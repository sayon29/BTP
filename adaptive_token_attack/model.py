"""CPU-only, full-batch NumPy Adam with analytical gradients.
Stable logaddexp BCE is mathematically equivalent to BCEWithLogitsLoss.
Only category logits, one positive calibration scale and one bias are learned.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from adaptive_token_attack.data import (validate_partition, frame_fingerprint, fingerprint, seal, check_seal, validate_execution, execution_labels)
from adaptive_token_attack.features import CATEGORIES, STATIC
from shared.sersem_baseline import write_json

DEFAULTS = dict(seed=17, max_epochs=500, learning_rate=.03, patience=40, min_delta=1e-6,
                regularization=.01, calibration_l2=1e-4, initial_scale=4., initial_noise=.001, gradient_clip=10.)


@dataclass
class Pool:
    role: str
    manifest: object
    counts: np.ndarray
    sums: np.ndarray
    labels: np.ndarray
    signature: str


def make_pool(manifest, role, store):
    validate_partition(manifest, role)
    counts, sums = [], []
    for row in manifest.to_dict('records'):
        arrays = store.get(row)
        if arrays is None:
            raise FileNotFoundError('Missing cached features: '+row['sample_id'])
        mask = arrays['eligible'].astype(bool)
        category, signals = arrays['category'][mask], arrays['signals'][mask]
        counts.append(np.bincount(category, minlength=len(CATEGORIES)))
        sums.append(np.bincount(category, weights=signals, minlength=len(CATEGORIES)))
    return Pool(role, manifest.copy(), np.asarray(counts, dtype=np.float64), np.asarray(sums, dtype=np.float64),
                manifest.is_member.to_numpy(dtype=np.float64),
                fingerprint({'manifest':frame_fingerprint(manifest), 'features':store.digest(manifest)}))


def sigmoid(value):
    return np.exp(-np.logaddexp(0., -np.asarray(value, dtype=np.float64)))


def parameters(theta):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (len(CATEGORIES)+2,) or not np.isfinite(theta).all():
        raise ValueError('Invalid adaptive parameters')
    shifted = theta[:-2]-theta[:-2].max()
    log_weights = shifted-np.log(np.exp(shifted).sum())
    weights = np.exp(log_weights)
    scale = float(np.logaddexp(0., theta[-2]))
    if not np.all(weights > 0) or scale <= 0:
        raise ValueError('Degenerate weights/scale')
    return weights, scale, float(theta[-1]), log_weights


def aggregate(theta, counts, sums):
    weights, _, _, _ = parameters(theta)
    counts, sums = np.asarray(counts), np.asarray(sums)
    if (counts.ndim != 2 or counts.shape != sums.shape or counts.shape[1] != len(CATEGORIES)
            or not np.isfinite(counts).all() or not np.isfinite(sums).all()
            or np.any(counts < 0) or np.any(sums < 0) or np.any(sums > counts)):
        raise ValueError('Invalid category sufficient statistics')
    denominator = counts @ weights
    if np.any(denominator <= 0):
        raise ValueError('Sample has no eligible targets')
    return (sums @ weights)/denominator


def bce_with_logits(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    if logits.shape != labels.shape or not np.isfinite(logits).all() or not np.isin(labels, [0.,1.]).all() or not logits.size:
        raise ValueError('Invalid BCE inputs')
    return float(np.mean(np.logaddexp(0., logits)-labels*logits))


def loss_gradient(theta, pool, cfg):
    weights, scale, bias, log_weights = parameters(theta)
    score = aggregate(theta, pool.counts, pool.sums)
    logits = scale*(score-.5)+bias
    bce = bce_with_logits(logits, pool.labels)
    residual = (sigmoid(logits)-pool.labels)/len(pool.labels)
    denominator = pool.counts @ weights
    dscore = weights[None,:]*(pool.sums-score[:,None]*pool.counts)/denominator[:,None]
    gradient = np.zeros_like(theta)
    gradient[:-2] = (residual*scale) @ dscore
    dscale = float(np.dot(residual, score-.5))
    gradient[-1] = residual.sum()
    prior = STATIC.astype(np.float64)/STATIC.sum(dtype=np.float64)
    kl = float(np.sum(prior*(np.log(prior)-log_weights)))
    calibration_penalty = cfg['calibration_l2']*((scale-cfg['initial_scale'])**2+bias**2)
    objective = bce+cfg['regularization']*kl+calibration_penalty
    gradient[:-2] += cfg['regularization']*(weights-prior)
    dscale += 2*cfg['calibration_l2']*(scale-cfg['initial_scale'])
    gradient[-2] = dscale*float(sigmoid(theta[-2]))
    gradient[-1] += 2*cfg['calibration_l2']*bias
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite adaptive objective/gradient')
    return float(objective), gradient, bce, kl


def validation_loss(theta, pool):
    _, scale, bias, _ = parameters(theta)
    return bce_with_logits(scale*(aggregate(theta,pool.counts,pool.sums)-.5)+bias, pool.labels)


def assert_disjoint(pool, protected):
    for other in protected:
        if (set(pool.manifest.sample_id) & set(other['sample_ids'])
                or set(pool.manifest.content_sha256) & set(other['content_hashes'])):
            raise ValueError('Training/validation/calibration/test data leakage')


def pool_provenance(pool):
    return dict(role=pool.role, signature=pool.signature, sample_ids=pool.manifest.sample_id.tolist(),
                content_hashes=pool.manifest.content_sha256.tolist())


def artifact(theta, train, validation, identity, cfg, best_loss):
    weights, scale, bias, _ = parameters(theta)
    return seal(dict(schema='adaptive-sersem-weights-v1', theta=theta.tolist(),
                     categories=list(CATEGORIES), weights=dict(zip(CATEGORIES, weights.tolist())),
                     static_prior=dict(zip(CATEGORIES, (STATIC/STATIC.sum()).astype(float).tolist())),
                     calibration_scale=scale, calibration_bias=bias, score_direction='larger_is_member; positive_scale',
                     training=pool_provenance(train), validation=pool_provenance(validation),
                     fit_identity=identity, hyperparameters=cfg, best_validation_bce=float(best_loss),
                     category_counts_train=dict(zip(CATEGORIES, train.counts.sum(axis=0).astype(int).tolist()))))


def fit(train, validation, cfg, out, static_gate, epoch_budget=None):
    """Resume exact Adam/early-stop state. Neither heldout nor calibration labels enter."""
    if train.role != 'shadow' or validation.role != 'validation':
        raise ValueError('Fit requires shadow training and fusion-fit validation pools only')
    validate_partition(train.manifest, 'shadow'); validate_partition(validation.manifest, 'validation')
    assert_disjoint(train, [pool_provenance(validation)])
    check_seal(static_gate)
    execution = static_gate.get('execution')
    if execution:
        validate_execution(execution, {'shadow':train.manifest, 'validation':validation.manifest}, complete=False)
    if not static_gate.get('verified'):
        raise ValueError('Static-score reproduction must pass before adaptive training')
    for pool in [train, validation]:
        if static_gate['pool_signatures'].get(pool.role) != pool.signature:
            raise ValueError('Static reproduction gate belongs to different features/manifests')
    if set(cfg)-set(DEFAULTS):
        raise ValueError('Unknown training hyperparameters')
    cfg = dict(DEFAULTS, **cfg)
    if (any(not np.isfinite(v) for v in cfg.values()) or cfg['max_epochs'] < 1 or cfg['patience'] < 1
            or cfg['learning_rate'] <= 0 or cfg['initial_scale'] <= 0 or cfg['gradient_clip'] <= 0
            or min(cfg['regularization'],cfg['calibration_l2'],cfg['min_delta'],cfg['initial_noise']) < 0):
        raise ValueError('Invalid training hyperparameters')
    identity = fingerprint(dict(training=train.signature, validation=validation.signature, cfg=cfg,
                         static_gate=static_gate['artifact_sha256'], implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    path = out/'training_checkpoint.json'
    if path.exists():
        saved = json.loads(path.read_text()); check_seal(saved)
        if saved['identity'] != identity:
            raise ValueError('Training configuration/feature identity changed; use a new output run')
        state = saved['state']
    else:
        rng = np.random.default_rng(cfg['seed'])
        theta = np.r_[np.log(STATIC.astype(np.float64))+rng.normal(0,cfg['initial_noise'],len(STATIC)),
                      np.log(np.expm1(cfg['initial_scale'])), 0.]
        state = dict(theta=theta.tolist(), m=np.zeros_like(theta).tolist(), v=np.zeros_like(theta).tolist(), epoch=0,
                     best_theta=theta.tolist(), best_loss=validation_loss(theta,validation), bad_epochs=0, history=[], complete=False)
    def export():
        value = artifact(np.array(state['best_theta']),train,validation,identity,cfg,state['best_loss'])
        payload = {k:v for k,v in value.items() if k != 'artifact_sha256'}
        payload.update(training_complete=state['complete'], static_gate_sha256=static_gate['artifact_sha256'],
                       pool_signatures=static_gate['pool_signatures'], execution=execution, **execution_labels(execution))
        return seal(payload)

    executed = 0
    while not state['complete'] and (epoch_budget is None or executed < epoch_budget):
        theta, m, v = [np.array(state[k],dtype=np.float64) for k in ['theta','m','v']]
        _, gradient, _, _ = loss_gradient(theta, train, cfg)
        norm = np.linalg.norm(gradient)
        if norm > cfg['gradient_clip']:
            gradient *= cfg['gradient_clip']/norm
        epoch = state['epoch']+1
        m = .9*m+.1*gradient; v = .999*v+.001*gradient**2
        theta -= cfg['learning_rate']*(m/(1-.9**epoch))/(np.sqrt(v/(1-.999**epoch))+1e-8)
        objective, _, train_bce, kl = loss_gradient(theta, train, cfg)
        val = validation_loss(theta, validation)
        if not np.isfinite(val):
            raise FloatingPointError('Nonfinite validation loss')
        if val < state['best_loss']-cfg['min_delta']:
            state.update(best_loss=val, best_theta=theta.tolist(), bad_epochs=0)
        else:
            state['bad_epochs'] += 1
        state['history'].append(dict(epoch=epoch,train_loss=objective,train_bce=train_bce,validation_bce=val,kl_to_static=kl,**execution_labels(execution)))
        state.update(theta=theta.tolist(), m=m.tolist(), v=v.tolist(), epoch=epoch,
                     complete=bool(epoch>=cfg['max_epochs'] or state['bad_epochs']>=cfg['patience']))
        write_json(path, seal(dict(identity=identity, state=state)))
        write_json(out/'best_weights.json', export())
        write_json(out/'training_history.json', state['history'])
        executed += 1
    result = export()
    if state['complete']:
        write_json(out/'weights.json', result)
    return result, state['complete']


def freeze(model, validation, out):
    check_seal(model)
    if not model.get('training_complete'):
        raise ValueError('Training must finish before freezing')
    if validation.role != 'validation' or model['validation']['signature'] != validation.signature:
        raise ValueError('Freeze requires the original validation pool')
    actual = validation_loss(np.array(model['theta']), validation)
    if abs(actual-model['best_validation_bce']) > 1e-12:
        raise ValueError('Best-checkpoint validation loss mismatch')
    payload = {k:v for k,v in model.items() if k != 'artifact_sha256'}
    payload.update(frozen=True, frozen_validation_bce=actual)
    result = seal(payload)
    write_json(Path(out)/'frozen_weights.json', result)
    return result


def score_pool(model, pool):
    check_seal(model)
    if not model.get('frozen'):
        raise ValueError('Validate and freeze weights before calibration/evaluation')
    theta = np.array(model['theta']); _, scale, bias, _ = parameters(theta)
    result = pool.manifest.copy()
    if model.get('execution'):
        for key,value in execution_labels(model['execution']).items():
            result[key] = value
    result['adaptive_score'] = aggregate(theta, pool.counts, pool.sums)
    result['adaptive_probability'] = sigmoid(scale*(result.adaptive_score.to_numpy()-.5)+bias)
    return result

"""Static reproduction gate, calibration-only thresholding and frozen evaluation."""
import json
from pathlib import Path

import numpy as np
from adaptive_token_attack.data import match_scores, fingerprint, seal, check_seal, validate_partitions, validate_execution, execution_labels
from adaptive_token_attack.features import static_score
from adaptive_token_attack.model import assert_disjoint, score_pool
from shared.research_execution import metric_report
from shared.sersem_baseline import write_json

# Fixed before looking at adaptive/test metrics; no automatic tolerance relaxation.
STATIC_ATOL = {'float32':2e-6, 'float16':2e-5, 'bfloat16':2e-5}


def verify_static(pools, saved_scores, store, dtype, out, execution=None):
    if set(pools) != {'shadow','validation','calibration','heldout'} or dtype not in STATIC_ATOL:
        raise ValueError('Require all four pools and a supported saved model dtype')
    validate_partitions({role:pool.manifest for role,pool in pools.items()},
                        expected_counts={role:len(pool.manifest) for role,pool in pools.items()})
    if execution:
        validate_execution(execution, {role:pool.manifest for role,pool in pools.items()})
    saved = match_scores(pools['heldout'].manifest, saved_scores).set_index('sample_id')
    records, max_cpu_error = [], 0.
    for role, pool in pools.items():
        for row in pool.manifest.to_dict('records'):
            arrays = store.get(row)
            if arrays is None:
                raise FileNotFoundError('Missing feature record: '+row['sample_id'])
            cpu = static_score(arrays); gpu = float(arrays['static_score_gpu'])
            max_cpu_error = max(max_cpu_error, abs(cpu-gpu))
            if role == 'heldout':
                reference = float(saved.loc[row['sample_id'],'greybox_score'])
                records.append(dict(sample_id=row['sample_id'], saved=reference, reconstructed=cpu,
                                    extraction_gpu=gpu, absolute_error=abs(cpu-reference), gpu_error=abs(gpu-reference)))
    maximum = max(max(r['absolute_error'],r['gpu_error']) for r in records)
    result = seal(dict(schema='adaptive-static-verification-v1', verified=bool(maximum<=STATIC_ATOL[dtype] and max_cpu_error<=2e-6),
                       execution=execution, **execution_labels(execution),
                       dtype=dtype, absolute_tolerance=STATIC_ATOL[dtype], relative_tolerance=0.,
                       max_saved_score_error=maximum, max_cpu_gpu_error=max_cpu_error,
                       compared_saved_samples=len(records), records=records,
                       pool_signatures={role:pool.signature for role,pool in pools.items()},
                       saved_scores_sha256=fingerprint(json.loads(saved.reset_index().to_json(orient='records')))))
    write_json(Path(out)/'static_verification.json',result)
    if not result['verified']:
        raise ValueError('Static SERSEM reproduction mismatch. Inspect static_verification.json; investigate target/tokenization/dtype/backend/weights/source/environment. Training is blocked.')
    return result


def check_pool(model, pool):
    check_seal(model)
    if model.get('execution'):
        validate_execution(model['execution'], {pool.role:pool.manifest}, complete=False)
    if model.get('pool_signatures',{}).get(pool.role) != pool.signature:
        raise ValueError('Pool differs from the feature cohort verified before training')


def calibrate(model, pool, out, target_fpr=.01):
    from sklearn.metrics import roc_curve
    if pool.role != 'calibration' or not 0 <= target_fpr < 1:
        raise ValueError('Threshold selection requires the threshold-calibration pool')
    check_pool(model,pool)
    assert_disjoint(pool,[model['training'],model['validation']])
    scored = score_pool(model,pool)
    y = scored.is_member.to_numpy(); scores = scored.adaptive_score.to_numpy()
    fpr,tpr,cuts = roc_curve(y,scores,drop_intermediate=False)
    indexes = np.flatnonzero((fpr<=target_fpr) & np.isfinite(cuts))
    cut = float(cuts[indexes[np.argmax(tpr[indexes])]]) if len(indexes) else float(np.nextafter(scores.max(),np.inf))
    predicted = scores>=cut
    result = seal(dict(schema='adaptive-threshold-v1', source_role='calibration', target_fpr=target_fpr,
                       **execution_labels(model.get('execution')),
                       threshold=cut, score_column='adaptive_score', frozen_model_sha256=model['artifact_sha256'],
                       calibration_signature=pool.signature, sample_ids=pool.manifest.sample_id.tolist(),
                       content_hashes=pool.manifest.content_sha256.tolist(),
                       calibration_tpr=float(predicted[y==1].mean()), calibration_fpr=float(predicted[y==0].mean()),
                       negatives=int(sum(y==0)), low_fpr_resolution_limited=bool(sum(y==0)<100)))
    write_json(Path(out)/'threshold.json',result)
    scored['adaptive_predicted_member'] = predicted
    scored.to_parquet(Path(out)/'calibration_scores.parquet',index=False)
    return result


def evaluate(model, threshold, heldout, saved_scores, out, expected_count=1000):
    check_pool(model,heldout); check_seal(threshold)
    execution = model.get('execution')
    if execution:
        configured = execution['expected_counts']['heldout']
        if expected_count not in {1000, configured}:
            raise ValueError('Evaluation count disagrees with execution selection')
        expected_count = configured
        if threshold.get('mode') != execution['mode']:
            raise ValueError('Smoke/full threshold mode mismatch')
    if heldout.role != 'heldout' or len(heldout.manifest) != expected_count:
        raise ValueError(f'Final evaluation requires the exact frozen {expected_count:,}-sample heldout pool')
    if (threshold['source_role'] != 'calibration' or threshold['score_column'] != 'adaptive_score'
            or threshold['frozen_model_sha256'] != model['artifact_sha256']
            or threshold['calibration_signature'] != model['pool_signatures']['calibration']):
        raise ValueError('Frozen model/threshold/calibration provenance mismatch')
    assert_disjoint(heldout,[model['training'],model['validation'],threshold])
    reference = match_scores(heldout.manifest,saved_scores)
    scored = score_pool(model,heldout)
    for column in ['lumia_score','greybox_score','fusion_score']:
        scored[column] = reference[column].to_numpy()
    predicted = scored.adaptive_score.to_numpy()>=threshold['threshold']
    scored['adaptive_predicted_member'] = predicted
    y = scored.is_member.to_numpy()
    columns = ['adaptive_score','greybox_score','lumia_score','fusion_score']
    identity = fingerprint(dict(model=model['artifact_sha256'],threshold=threshold['artifact_sha256'],heldout=heldout.signature,
                           saved_scores=json.loads(reference.to_json(orient='records'))))
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists() and json.loads((out/'metrics.json').read_text())['evaluation_identity'] != identity:
        raise ValueError('This run already evaluated a different frozen model/cohort; use a new run')
    def calibrated(group):
        labels = group.is_member.to_numpy(); detected = group.adaptive_predicted_member.to_numpy()
        return dict(tpr=float(detected[labels==1].mean()) if np.any(labels==1) else None,
                    fpr=float(detected[labels==0].mean()) if np.any(labels==0) else None)
    report = dict(schema='adaptive-metrics-v1', evaluation_identity=identity, metrics=metric_report(scored,columns),
                  execution=execution, **execution_labels(execution),
                  calibrated=dict(threshold=threshold['threshold'],overall=calibrated(scored),
                                  by_language={lang:calibrated(group) for lang,group in scored.groupby('language')}),
                  coverage=dict(requested=expected_count,matched=len(scored),fraction=len(scored)/expected_count,
                                by_language=scored.groupby('language').size().astype(int).to_dict()),
                  learned_weights=model['weights'],calibration_scale=model['calibration_scale'],calibration_bias=model['calibration_bias'],
                  frozen_model_sha256=model['artifact_sha256'],threshold_sha256=threshold['artifact_sha256'],
                  methodology='All four ROC reports recomputed on identical saved IDs/labels/language/content. Threshold uses calibration only. Larger adaptive_score is member-like; positive monotone calibration.',
                  provenance={'lumia':'pinned upstream scoring, reused saved scores', 'greybox':'existing SERSEM Section 2.1 reconstruction',
                              'fusion':'existing experimental baseline fusion, reused saved scores', 'adaptive':'our supervised category-weighting experiment'},
                  limitations=['No author-result equivalence claim','Empirical low-FPR ROC thresholds are reporting diagnostics only',
                               'No per-language fitting; rare/absent categories can be weakly identified',
                               'Equal-weight overlaps use our deterministic attribution convention'])
    if execution and execution['mode'] == 'smoke':
        report['status'] = 'Preliminary smoke evaluation; resolution-limited; not a full experimental result'
        negatives = int(sum(y==0))
        report['fpr_resolution'] = dict(heldout_negatives=negatives, heldout_fpr_step=1/negatives,
                                         calibration_negatives=threshold['negatives'], calibration_fpr_step=1/threshold['negatives'])
        report['limitations'].append('Smoke subsets only; empirical low-FPR estimates are coarse. See fpr_resolution for the selected negative counts.')
        for metrics in report['metrics'].values():
            for summary in [metrics['overall'], *metrics['by_language'].values()]:
                summary.update(preliminary=True, resolution_limited=True)
        for summary in [report['calibrated']['overall'], *report['calibrated']['by_language'].values()]:
            summary.update(preliminary=True, resolution_limited=True)
    scored.to_parquet(out/'heldout_adaptive_scores.parquet',index=False)
    scored.to_csv(out/'heldout_adaptive_scores.csv',index=False)
    write_json(out/'metrics.json',report)
    return report

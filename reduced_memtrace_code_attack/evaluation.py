"""Fixed shadow-only RF, validation diagnostics, calibration-only threshold."""
from pathlib import Path
import warnings
import numpy as np
from adaptive_token_attack.data import IDENTITY, match_scores, frame_fingerprint, validate_partitions
from shared.research_execution import metric_report
from shared.sersem_baseline import write_json
from reduced_memtrace_code_attack.runner import atomic_parquet
from reduced_memtrace_code_attack import METHOD_NAME, METHOD_STATUS


def exact_match(manifest, scores, columns):
    if not set(IDENTITY+columns) <= set(scores) or scores.sample_id.duplicated().any():
        raise ValueError('Missing identity fields or duplicate sample IDs')
    joined = manifest[IDENTITY].merge(scores[IDENTITY+columns], on='sample_id', how='left',
        validate='one_to_one', suffixes=('', '_saved'))
    for key in IDENTITY[1:]:
        if not joined[key].eq(joined[key+'_saved']).all():
            raise ValueError('Missing/mismatched comparison identity: '+key)
    if not np.isfinite(joined[columns].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite or incomplete matched scores')
    return joined[IDENTITY+columns]


def calibrate(frame):
    from sklearn.metrics import roc_curve
    if not frame.split_role.eq('threshold_calibration').all() or frame.is_member.nunique() != 2:
        raise ValueError('Threshold selection requires calibration only with both classes')
    y, s = frame.is_member.to_numpy(), frame.memtrace_score.to_numpy()
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite calibration scores')
    fpr, tpr, cuts = roc_curve(y, s, drop_intermediate=False)
    valid = np.flatnonzero((fpr <= .01) & np.isfinite(cuts))
    cut = float(cuts[valid[np.argmax(tpr[valid])]]) if len(valid) else float(np.nextafter(s.max(), np.inf))
    p = s >= cut
    return dict(source_role='calibration', target_fpr=.01, threshold=cut, direction='higher_is_member',
        negatives=int(sum(y==0)), members=int(sum(y==1)), calibration_manifest=frame_fingerprint(frame),
        calibration_tpr=float(p[y==1].mean()), calibration_fpr=float(p[y==0].mean()),
        empirical_fpr_resolution=1/int(sum(y==0)))


def fit_classifier(x, labels, fixed):
    from sklearn.ensemble import RandomForestClassifier
    if not np.isfinite(x).all() or set(labels) != {0, 1}:
        raise ValueError('Require finite development features and both classes')
    model = RandomForestClassifier(**fixed['classifier'])
    model.fit(x, labels)
    return model


def member_probabilities(model, x):
    """Fixed tree-order FP64 accumulation avoids parallel prediction sum roundoff."""
    probability = np.zeros((len(x), len(model.classes_)), dtype=np.float64)
    for tree in model.estimators_:
        probability += tree.predict_proba(x)
    return probability[:, list(model.classes_).index(1)] / len(model.estimators_)


def evaluate(features, selected, static_scores, optional, identity, out, mode, timing):
    import joblib
    out = Path(out)
    validate_partitions(selected, identity['full_counts'] if mode=='full' else identity['smoke_counts'])
    expected = set().union(*(set(f.sample_id) for f in selected.values()))
    if set(features.sample_id) != expected or features.sample_id.duplicated().any():
        raise ValueError('Require exactly all selected feature identities')
    names = identity['schema']['names']
    pools = {role:exact_match(frame, features, names) for role, frame in selected.items()}
    train = pools['shadow']
    model = fit_classifier(train[names].to_numpy(), train.is_member.to_numpy(), identity)
    joblib.dump(model, out/'classifier.joblib')
    # No validation-based search or refit. Language/identity never enter the classifier.
    for pool in pools.values():
        pool['memtrace_score'] = member_probabilities(model, pool[names].to_numpy())
    write_json(out/'validation_metrics.json', metric_report(pools['validation'], ['memtrace_score']))
    threshold = calibrate(pools['calibration'])
    write_json(out/'threshold.json', threshold)
    held = pools['heldout'][IDENTITY+['memtrace_score']].copy()
    atomic_parquet(held, out/'heldout_memtrace_scores.parquet')
    baseline = match_scores(selected['heldout'], static_scores)
    for source, dest in [('greybox_score','static_sersem_score'), ('lumia_score','lumia_score'), ('fusion_score','fusion_score')]:
        held[dest] = baseline[source].to_numpy()
    omitted = {}
    for column, source in optional.items():
        try:
            matched = exact_match(selected['heldout'], source, [column])
            held[column] = matched[column].to_numpy()
        except ValueError as exc:
            omitted[column] = str(exc)
            warnings.warn('Optional comparison omitted: '+column+': '+str(exc))
    columns = [c for c in held if c.endswith('_score')]
    def deployed(group):
        y, p = group.is_member.to_numpy(), group.memtrace_score.to_numpy() >= threshold['threshold']
        return dict(tpr=float(p[y==1].mean()), fpr=float(p[y==0].mean()),
            members=int(sum(y==1)), negatives=int(sum(y==0)))
    importance = dict(zip(names, model.feature_importances_.tolist()))
    families = {family:float(sum(v for v,f in zip(model.feature_importances_, identity['schema']['families']) if f==family))
                for family in sorted(set(identity['schema']['families']))}
    report = dict(method_name=METHOD_NAME, method_status=METHOD_STATUS, mode=mode, preliminary=mode=='smoke', status='complete',
        coverage=dict(requested=len(selected['heldout']), valid_matched=len(held), invalid=0,
            original_heldout_requested=1000, original_heldout_fraction=len(held)/1000,
            by_language=held.groupby('language').size().to_dict()),
        deployed_calibrated_threshold=dict(**threshold, overall=deployed(held),
            by_language={l:deployed(g) for l,g in held.groupby('language')}),
        heldout_diagnostic_roc_only=metric_report(held, columns),
        feature_importance=importance, feature_family_importance=families, optional_omitted=omitted,
        timing=timing,
        limitations=['Reduced paper-informed implementation, not exact memTrace; author code unavailable',
            '100 full-run calibration negatives give 1% empirical FPR steps and substantial sampling uncertainty; no population FPR guarantee',
            'Held-out ROC thresholds diagnostic only; deployed threshold uses calibration only',
            'Per-language and smoke low-FPR estimates are coarse',
            'Impurity feature importance is descriptive association, not causal memorization evidence',
            'Exact sample pairings; previous attacks retain different analysis spans'])
    # pandas describe creates tuple column keys; JSON artifacts require plain string keys.
    report['effective_token_counts'] = {role:{c:{str(k):float(v) for k,v in g[c].describe().items()}
        for c in ['valid_token_count','effective_target_tokens','truncated_tokens']} for role,g in features.groupby('split_role')}
    diagnostics = report['heldout_diagnostic_roc_only']
    primary = diagnostics['memtrace_score']['overall']
    report['paired_differences'] = {c:dict(auc_delta_memtrace_minus_baseline=primary['auc_roc']-diagnostics[c]['overall']['auc_roc'],
        diagnostic_tpr_delta=primary['tpr_at_fpr_le_1pct']-diagnostics[c]['overall']['tpr_at_fpr_le_1pct'])
        for c in columns if c != 'memtrace_score'}
    atomic_parquet(held, out/'matched_comparison.parquet')
    write_json(out/'metrics.json', report)
    return report

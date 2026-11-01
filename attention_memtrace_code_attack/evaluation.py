"""Fixed shadow-only RF, validation diagnostics, calibration-only threshold."""
from pathlib import Path
import warnings
import numpy as np
from adaptive_token_attack.data import IDENTITY, match_scores, frame_fingerprint, validate_partitions
from shared.research_execution import metric_report
from shared.sersem_baseline import write_json
from attention_memtrace_code_attack.runner import atomic_parquet
from attention_memtrace_code_attack import METHOD_NAME, METHOD_STATUS


from reduced_memtrace_code_attack.evaluation import exact_match, calibrate, fit_classifier, member_probabilities


def paired_bootstrap(frame, primary, references, settings):
    """Paired stratified held-out resampling of fixed scores; no model fitting/tuning."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(settings['seed'])
    groups = [g.index.to_numpy() for _,g in frame.reset_index(drop=True).groupby(['language','is_member'])]
    y = frame.is_member.to_numpy()
    scores = {c:frame[c].to_numpy() for c in [primary]+references}
    deltas = {c:[] for c in references}
    for _ in range(settings['replicates']):
        indices = np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        primary_auc = roc_auc_score(y[indices],scores[primary][indices])
        for column in references:
            deltas[column].append(primary_auc-roc_auc_score(y[indices],scores[column][indices]))
    return {c:dict(auc_delta_95pct_ci=[float(v) for v in np.quantile(values,[.025,.975])],
        replicates=settings['replicates'],seed=settings['seed'],strata=settings['strata'],
        interpretation='Conditional on the fixed trained classifiers and this cohort; no retraining uncertainty or population guarantee')
        for c,values in deltas.items()}


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
        pool['attention_memtrace_score'] = member_probabilities(model, pool[names].to_numpy())
    write_json(out/'validation_metrics.json', metric_report(pools['validation'], ['attention_memtrace_score']))
    threshold = calibrate(pools['calibration'].rename(columns={'attention_memtrace_score':'memtrace_score'}))
    write_json(out/'threshold.json', threshold)
    held = pools['heldout'][IDENTITY+['attention_memtrace_score']].copy()
    atomic_parquet(held, out/'heldout_attention_memtrace_scores.parquet')
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
        y, p = group.is_member.to_numpy(), group.attention_memtrace_score.to_numpy() >= threshold['threshold']
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
        limitations=['Single combined adaptation, not exact memTrace; sparsity threshold/head-entropy aggregation underspecified in paper; author code unavailable',
            '100 full-run calibration negatives give 1% empirical FPR steps and substantial sampling uncertainty; no population FPR guarantee',
            'Held-out ROC thresholds diagnostic only; deployed threshold uses calibration only',
            'Per-language and smoke low-FPR estimates are coarse',
            'Impurity feature importance is descriptive association, not causal memorization evidence',
            'Exact sample pairings; previous attacks retain different analysis spans'])
    # pandas describe creates tuple column keys; JSON artifacts require plain string keys.
    report['effective_token_counts'] = {role:{c:{str(k):float(v) for k,v in g[c].describe().items()}
        for c in ['valid_token_count','effective_target_tokens','truncated_tokens']} for role,g in features.groupby('split_role')}
    diagnostics = report['heldout_diagnostic_roc_only']
    primary = diagnostics['attention_memtrace_score']['overall']
    report['paired_differences'] = {c:dict(auc_delta_attention_memtrace_minus_baseline=primary['auc_roc']-diagnostics[c]['overall']['auc_roc'],
        diagnostic_tpr_delta=primary['tpr_at_fpr_le_1pct']-diagnostics[c]['overall']['tpr_at_fpr_le_1pct'])
        for c in columns if c != 'attention_memtrace_score'}
    report['paired_bootstrap'] = paired_bootstrap(held, 'attention_memtrace_score',
        [c for c in columns if c != 'attention_memtrace_score'], identity['paired_bootstrap'])
    report['feature_counts'] = dict(reduced=identity['schema']['preserved_prefix_features'],
        attention=identity['schema']['attention_features'], combined=len(names))
    report['attention_backend'] = 'eager'
    report['prefix_reuse'] = identity.get('prefix_reuse', {'status':'not_attached'})
    report['limitations'].append('Fresh Reduced prefix uses identical formulas with eager attention; FP16 backend roundoff can differ from the saved SDPA run. Verified saved prefix reuse preserves those values exactly.')
    atomic_parquet(held, out/'matched_comparison.parquet')
    write_json(out/'metrics.json', report)
    return report

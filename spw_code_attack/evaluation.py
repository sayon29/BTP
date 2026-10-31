"""Exact-identity paired evaluation; deployed threshold uses calibration only."""
from pathlib import Path
import numpy as np
from adaptive_token_attack.data import IDENTITY, match_scores, frame_fingerprint, validate_partitions, COUNTS, SMOKE_SIZES
from spw_code_attack.runner import atomic_parquet
from spw_code_attack import METHOD_NAME, METHOD_STATUS
from shared.research_execution import metric_report
from shared.sersem_baseline import write_json


def exact_match(manifest, scores, columns, strict=True):
    fields = IDENTITY if strict else ['sample_id','language','is_member','content_sha256']
    if not set(fields+columns) <= set(scores) or scores.sample_id.duplicated().any():
        raise ValueError('Missing identity/score fields or duplicate sample IDs')
    joined = manifest[fields].merge(scores[fields+columns],on='sample_id',how='left',validate='one_to_one',suffixes=('','_saved'))
    for key in fields[1:]:
        if not joined[key].eq(joined[key+'_saved']).all():
            raise ValueError('Score identity mismatch: '+key)
    if not np.isfinite(joined[columns].to_numpy(dtype=float)).all():
        raise ValueError('Missing/nonfinite matched scores')
    return joined[fields+columns]


def calibrate(frame):
    from sklearn.metrics import roc_curve
    if not frame.split_role.eq('threshold_calibration').all() or frame.is_member.nunique()!=2:
        raise ValueError('Threshold selection requires calibration only and both classes')
    y,s = frame.is_member.to_numpy(),frame.spw_inspired_score.to_numpy()
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite calibration score')
    fpr,tpr,cuts = roc_curve(y,s,drop_intermediate=False)
    valid = np.flatnonzero((fpr<=.01)&np.isfinite(cuts))
    cut = float(cuts[valid[np.argmax(tpr[valid])]]) if len(valid) else float(np.nextafter(s.max(),np.inf))
    pred = s>=cut
    return dict(source_role='calibration', target_fpr=.01, threshold=cut, direction='higher_is_member',
                calibration_manifest=frame_fingerprint(frame),negatives=int(sum(y==0)),
                calibration_tpr=float(pred[y==1].mean()),calibration_fpr=float(pred[y==0].mean()))


def evaluate(scored, selected, static_scores, adaptive_scores, out, mode, timing=None):
    if mode not in {'smoke','full'}:
        raise ValueError('MODE must be smoke or full')
    validate_partitions(selected, COUNTS if mode=='full' else SMOKE_SIZES)
    held = selected['heldout']
    all_ids = set().union(*(set(f.sample_id) for f in selected.values()))
    required_ids = set(selected['calibration'].sample_id) | set(held.sample_id)
    if (not required_ids <= set(scored.sample_id) <= all_ids
            or scored.sample_id.duplicated().any()):
        raise ValueError('Require complete calibration/heldout cohorts, no unknown or duplicate IDs')
    calibration = exact_match(selected['calibration'],scored,['spw_inspired_score'])
    threshold = calibrate(calibration)
    write_json(Path(out)/'threshold.json',threshold)
    matched = exact_match(held,scored,['spw_inspired_score'])
    static = match_scores(held,static_scores)
    adaptive = exact_match(held,adaptive_scores,['adaptive_score']) if adaptive_scores is not None else None
    matched['static_sersem_score'] = static.greybox_score.to_numpy()
    matched['lumia_score'] = static.lumia_score.to_numpy()
    matched['fusion_score'] = static.fusion_score.to_numpy()
    if adaptive is not None:
        matched['adaptive_sersem_score'] = adaptive.adaptive_score.to_numpy()
    matched['spw_inspired_predicted_member'] = matched.spw_inspired_score>=threshold['threshold']
    columns = ['spw_inspired_score','static_sersem_score','lumia_score','fusion_score']
    if adaptive is not None:
        columns.append('adaptive_sersem_score')
    diagnostic = metric_report(matched,columns)
    def deployed(group):
        y = group.is_member.to_numpy(); p = group.spw_inspired_predicted_member.to_numpy()
        return dict(tpr=float(p[y==1].mean()) if np.any(y==1) else None,
                    achieved_fpr=float(p[y==0].mean()) if np.any(y==0) else None,
                    negatives=int(sum(y==0)))
    def differences(group):
        metrics = metric_report(group,columns)
        ref = metrics['spw_inspired_score']['overall']
        return {c:{'auc_delta_spw_inspired_minus_baseline':ref['auc_roc']-metrics[c]['overall']['auc_roc'],
                   'diagnostic_tpr_delta_spw_inspired_minus_baseline':ref['tpr_at_fpr_le_1pct']-metrics[c]['overall']['tpr_at_fpr_le_1pct']}
                for c in columns[1:]}
    report = dict(method_name=METHOD_NAME,method_status=METHOD_STATUS,paper_faithful=False,
        adaptive_comparison_included=adaptive is not None,
        timing=timing or {},mode=mode,preliminary=mode=='smoke',
        coverage=dict(selected_requested=len(held),matched=len(matched),selected_fraction=len(matched)/len(held),
                      original_heldout_requested=1000,original_heldout_fraction=len(matched)/1000,
                      by_language=matched.groupby('language').size().astype(int).to_dict()),
        deployed_calibrated_threshold=dict(**threshold,overall=deployed(matched),
            by_language={l:deployed(g) for l,g in matched.groupby('language')}),
        heldout_diagnostic_roc_only=diagnostic,
        paired_differences=dict(overall=differences(matched),by_language={l:differences(g) for l,g in matched.groupby('language')}),
        fpr_resolution=dict(calibration=1/threshold['negatives'],heldout=1/int(sum(matched.is_member==0))),
        limitations=['SPW-inspired approximation: full paper equations, algorithm and defaults unavailable',
                     'Held-out ROC thresholds are diagnostics only, never deployed thresholds',
                     'Dynamic-halving, weight and Bernoulli-JS equations are ours; no published SPW-MIA equivalence claim',
                     'First 256 eligible targets are a compute adaptation; published context settings were unavailable',
                     'Low-FPR per-language and smoke estimates have coarse empirical resolution',
                     'Saved methods retain their original baseline spans; pairings match exact identities, not analysis spans'])
    atomic_parquet(matched,Path(out)/'matched_comparison.parquet')
    write_json(Path(out)/'metrics.json',report)
    return report

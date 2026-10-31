"""Exact-identity paired evaluation; deployed threshold uses calibration only."""
from pathlib import Path
import numpy as np
from adaptive_token_attack.data import IDENTITY, match_scores, frame_fingerprint, validate_partitions, COUNTS, SMOKE_SIZES
from recall_code_attack.runner import atomic_parquet
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
    y,s = frame.is_member.to_numpy(),frame.recall_score.to_numpy()
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite calibration score')
    fpr,tpr,cuts = roc_curve(y,s,drop_intermediate=False)
    valid = np.flatnonzero((fpr<=.01)&np.isfinite(cuts))
    cut = float(cuts[valid[np.argmax(tpr[valid])]]) if len(valid) else float(np.nextafter(s.max(),np.inf))
    pred = s>=cut
    return dict(source_role='calibration', target_fpr=.01, threshold=cut, direction='higher_is_member',
                calibration_manifest=frame_fingerprint(frame),negatives=int(sum(y==0)),
                calibration_tpr=float(pred[y==1].mean()),calibration_fpr=float(pred[y==0].mean()))


def evaluate(scored, selected, static_scores, adaptive_scores, out, mode):
    if mode not in {'smoke','full'}:
        raise ValueError('MODE must be smoke or full')
    validate_partitions(selected, COUNTS if mode=='full' else SMOKE_SIZES)
    held = selected['heldout']
    all_ids = set().union(*(set(f.sample_id) for f in selected.values()))
    if set(scored.sample_id)!=all_ids or scored.sample_id.duplicated().any():
        raise ValueError('Require complete selected cohort, no dropped or extra IDs')
    calibration = exact_match(selected['calibration'],scored,['recall_score'])
    threshold = calibrate(calibration)
    write_json(Path(out)/'threshold.json',threshold)
    matched = exact_match(held,scored,['recall_score'])
    static = match_scores(held,static_scores)
    matched['static_sersem_score'] = static.greybox_score.to_numpy()
    matched['lumia_score'] = static.lumia_score.to_numpy()
    matched['fusion_score'] = static.fusion_score.to_numpy()
    matched['recall_predicted_member'] = matched.recall_score>=threshold['threshold']
    columns = ['recall_score','static_sersem_score','lumia_score','fusion_score']
    if adaptive_scores is not None:
        adaptive = exact_match(held,adaptive_scores,['adaptive_score'])
        matched['adaptive_sersem_score'] = adaptive.adaptive_score.to_numpy()
        columns.insert(2, 'adaptive_sersem_score')
    diagnostic = metric_report(matched,columns)
    def deployed(group):
        y = group.is_member.to_numpy(); p = group.recall_predicted_member.to_numpy()
        return dict(tpr=float(p[y==1].mean()) if np.any(y==1) else None,
                    achieved_fpr=float(p[y==0].mean()) if np.any(y==0) else None,
                    negatives=int(sum(y==0)))
    def differences(group):
        metrics = metric_report(group,columns)
        ref = metrics['recall_score']['overall']
        return {c:{'auc_delta_recall_minus_baseline':ref['auc_roc']-metrics[c]['overall']['auc_roc'],
                   'diagnostic_tpr_delta_recall_minus_baseline':ref['tpr_at_fpr_le_1pct']-metrics[c]['overall']['tpr_at_fpr_le_1pct']}
                for c in columns[1:]}
    report = dict(mode=mode,preliminary=mode=='smoke',
        coverage=dict(selected_requested=len(held),matched=len(matched),selected_fraction=len(matched)/len(held),
                      original_heldout_requested=1000,original_heldout_fraction=len(matched)/1000,
                      by_language=matched.groupby('language').size().astype(int).to_dict()),
        deployed_calibrated_threshold=dict(**threshold,overall=deployed(matched),
            by_language={l:deployed(g) for l,g in matched.groupby('language')}),
        heldout_diagnostic_roc_only=diagnostic,
        paired_differences=dict(overall=differences(matched),by_language={l:differences(g) for l,g in matched.groupby('language')}),
        fpr_resolution=dict(calibration=1/threshold['negatives'],heldout=1/int(sum(matched.is_member==0))),
        comparisons=dict(adaptive_sersem_included='adaptive_sersem_score' in columns,
            paired_methods=columns),
        limitations=['Code/prefix/identical-mask adaptation, no paper-result equivalence claim',
                     'Adaptive comparison is omitted when no exact-identity adaptive score file is attached',
                     'Held-out ROC thresholds are diagnostics only, never deployed thresholds',
                     'Prefix overlap screening excludes fixed exact n-grams; cannot certify semantic unrelatedness',
                     'Saved non-member labels are assumed valid; no independent pretraining membership proof',
                     'Low-FPR per-language and smoke estimates have coarse empirical resolution',
                     'Saved comparison scores retain their original complete baseline spans; context truncation, if any, is reported'])
    atomic_parquet(matched,Path(out)/'matched_comparison.parquet')
    write_json(Path(out)/'metrics.json',report)
    return report

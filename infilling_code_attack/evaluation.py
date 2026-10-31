"""Calibrate only on calibration; explicitly report invalid rows and common-ID coverage."""
from pathlib import Path
import numpy as np
from adaptive_token_attack.data import IDENTITY,COUNTS,SMOKE_SIZES,validate_partitions,match_scores,frame_fingerprint
from shared.research_execution import metric_report
from shared.sersem_baseline import write_json
from infilling_code_attack.runner import atomic_parquet
from infilling_code_attack import METHOD_NAME,METHOD_STATUS


def exact_match(manifest,scores,columns):
    fields=IDENTITY
    if not set(fields+columns)<=set(scores) or scores.sample_id.duplicated().any():raise ValueError('Missing/duplicate score identities')
    joined=manifest[fields].merge(scores[fields+columns],on='sample_id',how='left',validate='one_to_one',suffixes=('','_saved'))
    for field in fields[1:]:
        if not joined[field].eq(joined[field+'_saved']).all():raise ValueError('Score identity mismatch: '+field)
    if not np.isfinite(joined[columns].to_numpy(dtype=float)).all():raise ValueError('Missing/nonfinite matched score')
    return joined[fields+columns]


def calibrate(frame):
    from sklearn.metrics import roc_curve
    if not frame.split_role.eq('threshold_calibration').all():raise ValueError('Calibration partition required')
    if frame.is_member.nunique()!=2:return dict(status='unavailable',reason='Both valid calibration classes required',source_role='calibration',threshold=None)
    y=frame.is_member.to_numpy();s=frame.infilling_score.to_numpy()
    if not np.isfinite(s).all():raise ValueError('Nonfinite valid calibration scores')
    f,t,c=roc_curve(y,s,drop_intermediate=False)
    eligible=np.flatnonzero((f<=.01)&np.isfinite(c))
    cut=float(c[eligible[np.argmax(t[eligible])]]) if len(eligible) else float(np.nextafter(s.max(),np.inf))
    p=s>=cut
    return dict(status='calibrated',source_role='calibration',threshold=cut,target_fpr=.01,direction='higher_is_member',
        calibration_manifest=frame_fingerprint(frame),calibration_tpr=float(p[y==1].mean()),calibration_fpr=float(p[y==0].mean()),
        negatives=int(sum(y==0)),members=int(sum(y==1)))


def evaluate(scored,selected,static_scores,adaptive_scores,out,mode,timing=None):
    import pandas as pd
    if mode not in {'smoke','full'}:raise ValueError('Invalid MODE')
    validate_partitions(selected,COUNTS if mode=='full' else SMOKE_SIZES)
    manifest=pd.concat(list(selected.values()),ignore_index=True)
    if set(scored.sample_id)!=set(manifest.sample_id):raise ValueError('Incomplete or extra scored sample IDs')
    exact_match(manifest,scored,[])
    if not scored.status.isin(['valid','invalid']).all():raise ValueError('Unknown scoring status')
    valid=scored.status.eq('valid')
    if not np.isfinite(scored.loc[valid,'infilling_score'].to_numpy(dtype=float)).all() or scored.loc[~valid,'infilling_score'].notna().any():
        raise ValueError('Score validity flag disagreement')
    # Require complete baseline/adaptive identity coverage BEFORE explicit invalid filtering.
    base=match_scores(selected['heldout'],static_scores)
    adaptive=exact_match(selected['heldout'],adaptive_scores,['adaptive_score'])
    held_valid=selected['heldout'][selected['heldout'].sample_id.isin(scored.loc[valid,'sample_id'])]
    cal_valid=selected['calibration'][selected['calibration'].sample_id.isin(scored.loc[valid,'sample_id'])]
    calibration=exact_match(cal_valid,scored.loc[valid],['infilling_score']);threshold=calibrate(calibration)
    write_json(Path(out)/'threshold.json',threshold)
    matched=exact_match(held_valid,scored.loc[valid],['infilling_score'])
    baseline=base.set_index('sample_id');a=adaptive.set_index('sample_id')
    for dest,source in [('static_sersem_score','greybox_score'),('lumia_score','lumia_score'),('fusion_score','fusion_score')]:
        matched[dest]=baseline.loc[matched.sample_id,source].to_numpy()
    matched['adaptive_sersem_score']=a.loc[matched.sample_id,'adaptive_score'].to_numpy()
    if threshold['threshold'] is not None:matched['infilling_predicted_member']=matched.infilling_score>=threshold['threshold']
    else:matched['infilling_predicted_member']=pd.array([None]*len(matched),dtype='boolean')
    columns=['infilling_score','static_sersem_score','adaptive_sersem_score','lumia_score','fusion_score']
    diagnostic=metric_report(matched,columns)
    def deployed(group):
        if threshold['threshold'] is None:return dict(status='threshold unavailable',tpr=None,fpr=None)
        y=group.is_member.to_numpy();p=group.infilling_predicted_member.to_numpy(dtype=bool)
        return dict(tpr=float(p[y==1].mean()) if np.any(y==1) else None,
                    fpr=float(p[y==0].mean()) if np.any(y==0) else None)
    def difference(group):
        metrics=metric_report(group,columns);ref=metrics['infilling_score']['overall']
        return {c:{name:(ref[key]-metrics[c]['overall'][key]) if key in ref and key in metrics[c]['overall'] else None
                   for name,key in [('auc_delta_infilling_minus_baseline','auc_roc'),('diagnostic_tpr_delta','tpr_at_fpr_le_1pct')]}
                for c in columns[1:]}
    invalid=scored.loc[~valid];held_invalid=invalid[invalid.sample_id.isin(selected['heldout'].sample_id)]
    report=dict(method_name=METHOD_NAME,method_status=METHOD_STATUS,mode=mode,preliminary=mode=='smoke',
        status='complete' if invalid.empty else 'complete_with_explicit_invalid_samples',
        coverage=dict(requested=len(selected['heldout']),processed=len(selected['heldout']),valid_matched=len(matched),
            invalid=len(held_invalid),selected_valid_fraction=len(matched)/len(selected['heldout']),
            original_heldout_requested=1000,original_heldout_fraction=len(matched)/1000,
            by_language={lang:dict(requested=len(group),valid=int(sum(matched.language==lang)),invalid=int(sum(held_invalid.language==lang)))
                         for lang,group in selected['heldout'].groupby('language')}),
        invalid_samples=dict(total=len(invalid),by_role=invalid.groupby('split_role').size().astype(int).to_dict(),
            reasons=invalid.invalid_reason.value_counts().astype(int).to_dict(),heldout_ids=held_invalid.sample_id.tolist()),
        effective_token_counts=dict(by_partition={role:scored[scored.sample_id.isin(frame.sample_id)].effective_analysis_tokens.describe().to_dict()
                                                  for role,frame in selected.items()}),
        deployed_calibrated_threshold=dict(**threshold,overall=deployed(matched),by_language={l:deployed(g) for l,g in matched.groupby('language')}),
        heldout_diagnostic_roc_only=diagnostic,paired_differences=dict(overall=difference(matched),by_language={l:difference(g) for l,g in matched.groupby('language')}),
        timing=timing or {},limitations=['Code/domain, 128-token span and FP16/FP64 arithmetic adaptations; no benchmark-result equivalence claim',
            'Pinned author-code context-specific normalization and end handling, not literal Appendix D pseudocode typos',
            'Invalid IDs are retained in scores/coverage; all five comparisons use exactly the same explicitly valid cohort',
            'Diagnostic held-out ROC thresholds never deploy; no direction or parameter search',
            'Baseline scores retain their original spans; identities match, analysis spans differ',
            'Low-FPR and per-language smoke estimates are empirically coarse'])
    atomic_parquet(matched,Path(out)/'matched_comparison.parquet');write_json(Path(out)/'metrics.json',report)
    return report

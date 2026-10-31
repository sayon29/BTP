"""Kaggle orchestration and atomic export; no target construction on import."""
import json
from pathlib import Path
import time
from adaptive_token_attack.data import IDENTITY
from infilling_code_attack.engine import Engine
from infilling_code_attack import METHOD_NAME, METHOD_STATUS
from shared.sersem_baseline import write_json


def atomic_parquet(frame,path):
    path=Path(path);tmp=path.with_suffix('.parquet.tmp')
    frame.to_parquet(tmp,index=False);tmp.replace(path)


def audit_report(path):
    """Persisted actual calls, including failed calls; flag interrupted accounting."""
    pending={};malformed=0;calls=0;heads=0;seconds=0.
    if Path(path).is_file():
        with Path(path).open() as file:
            for line in file:
                try:
                    value=json.loads(line)
                    if value['event']=='started':pending[value['sequence']]=value['operation']
                    elif value['event']=='finished':
                        pending.pop(value['sequence'],None)
                        calls+=value['transformer_forward_calls'];heads+=value['head_forward_calls'];seconds+=value['seconds']
                    else:malformed+=1
                except (json.JSONDecodeError,KeyError,TypeError):malformed+=1
    return dict(transformer_forward_calls=calls,lm_head_forward_calls=heads,model_seconds=seconds,
                interrupted_unaccounted_operations=len(pending),malformed_log_lines=malformed,
                exact_accounting=not pending and not malformed)


def run_scores(selected,plans,identity,cache,predictor_factory,monitor,out):
    import pandas as pd
    out=Path(out);rows=pd.concat(list(selected.values()),ignore_index=True)
    engine=Engine(cache,identity,predictor_factory,monitor.guard,out/'forward_calls.jsonl')
    records=[];started=time.perf_counter()
    try:
        for i,row in enumerate(rows.to_dict('records')):
            result=engine.score(row,plans[row['sample_id']])
            scalar={k:v for k,v in result.items() if k not in {'token_scores','raw_log_ratios','greedy_ids','bottom_positions'}}
            records.append(dict({k:row[k] for k in IDENTITY},**plans[row['sample_id']]['metadata'],method_name=METHOD_NAME,method_status=METHOD_STATUS,**scalar))
            progress=dict(method_name=METHOD_NAME,statistic='pinned-author-normalization; paper Eqs 9-11; code/span adaptation',completed=i+1,requested=len(rows),
                sample_id=row['sample_id'],seconds_this_session=time.perf_counter()-started,
                actual_session_forward_calls=engine.forward_calls,actual_session_head_calls=engine.head_calls,
                last_sample_seconds=result['session_seconds'],last_sample_forward_calls=result['session_forward_calls'])
            write_json(out/'progress.json',progress)
            monitor.tick('infilling_scores',i+1,len(rows))
    finally:
        engine.scorer=None
    scored=pd.DataFrame(records)
    monitor.guard();atomic_parquet(scored,out/'infilling_scores.parquet')
    timing=dict(method_name=METHOD_NAME,method_status=METHOD_STATUS,statistic='pinned-author-normalization; paper Eqs 9-11; code/span adaptation',
        session_wall_seconds=time.perf_counter()-started,session_wall_seconds_per_example=(time.perf_counter()-started)/len(scored),
        session_transformer_forward_calls=engine.forward_calls,
        session_transformer_calls_per_example=engine.forward_calls/len(scored),
        session_head_forward_calls=engine.head_calls,session_model_seconds=engine.model_seconds,
        mean_session_seconds_per_example=float(scored.session_seconds.mean()),
        mean_cached_result_computation_seconds=float(scored.computation_seconds.mean()),
        cached_samples=int(scored.cache_hit.sum()),samples=len(scored),persistent_call_audit=audit_report(out/'forward_calls.jsonl'),
        timing_note='Cached result timing excludes work before interrupted partial-sample recovery. Persistent audit counts all recorded attempts; hard-killed unfinished operations are explicitly unknown. Cache import does not import historic call logs.')
    write_json(out/'timing.json',timing)
    return scored,timing

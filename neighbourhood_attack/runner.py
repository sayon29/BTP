"""Standalone masked-MLM neighbourhood MIA. Kaggle CUDA execution only.
S(x)=mean_j normalized_causal_NLL(neighbour_j)-normalized_causal_NLL(x).
No LUMIA imports, hidden states, probes or fusion are used by this runner.
"""
import contextlib
import hashlib
import json
import sqlite3
import time
import zlib
from pathlib import Path
from shared.sersem_baseline import write_json
from shared.research_execution import MILESTONES, metric_report, auc_metrics


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class Cache:
    def __init__(self,path):
        self.path=Path(path)
        self.con=sqlite3.connect(self.path)
        self.con.execute('PRAGMA cache_size=-16384')
        self.con.execute('CREATE TABLE IF NOT EXISTS cache(kind TEXT,key TEXT,payload BLOB,checksum TEXT,PRIMARY KEY(kind,key))')
        self.con.commit()
    def get(self,kind,key):
        row=self.con.execute('SELECT payload,checksum FROM cache WHERE kind=? AND key=?',(kind,key)).fetchone()
        if not row: return None
        if hashlib.sha256(row[0]).hexdigest()!=row[1]: raise ValueError('Checkpoint checksum mismatch')
        return json.loads(zlib.decompress(row[0]))
    def put(self,kind,key,value):
        packed=zlib.compress(json.dumps(value,allow_nan=False).encode('utf-8'))
        self.con.execute('INSERT OR REPLACE INTO cache VALUES(?,?,?,?)',(kind,key,packed,hashlib.sha256(packed).hexdigest()))
        self.con.commit()
    def rows(self,kind):
        for key,packed,checksum in self.con.execute('SELECT key,payload,checksum FROM cache WHERE kind=? ORDER BY rowid',(kind,)):
            if hashlib.sha256(packed).hexdigest()!=checksum: raise ValueError('Result checkpoint checksum mismatch')
            yield key,json.loads(zlib.decompress(packed))
    def close(self): self.con.close()


def reuse_nll_cache(cache, source_run, cfg):
    """Copy exact scalar NLLs across K runs only after target/math validation."""
    if not source_run: return 0
    source_run=Path(source_run)
    if source_run.resolve()==cache.path.parent.resolve(): return 0
    previous=json.loads((source_run/'config.json').read_text())
    keys=['model_id','model_revision','dtype','max_length','target_nll_version','runner_sha256','versions']
    if any(previous[k]!=cfg[k] for k in keys):
        raise ValueError('Cannot reuse NLLs from another target/math/version')
    uri=(source_run/'neighbourhood.sqlite').resolve().as_uri()+'?mode=ro'
    copied=0
    with contextlib.closing(sqlite3.connect(uri,uri=True)) as source:
        for key,packed,checksum in source.execute("SELECT key,payload,checksum FROM cache WHERE kind='nll'"):
            if hashlib.sha256(packed).hexdigest()!=checksum: raise ValueError('Imported NLL checksum mismatch')
            cache.con.execute("INSERT OR IGNORE INTO cache VALUES('nll',?,?,?)",(key,packed,checksum));copied+=1
            if copied%1000==0: cache.con.commit()
    cache.con.commit();return copied


class TargetNLL:
    def __init__(self,model,tokenizer,cfg,cache):
        self.model,self.tokenizer,self.cfg,self.cache=model,tokenizer,cfg,cache
        if model.device.type!='cuda': raise RuntimeError('CUDA target required')
        self.identity=sha(json.dumps({k:cfg[k] for k in ['model_id','model_revision','dtype','max_length','target_nll_version','runner_sha256']},sort_keys=True))
        self.calls=0; self.texts=0; self.seconds=0; self.monitor=None
    def key(self,text): return sha(self.identity+sha(text))
    def score(self,texts):
        import torch
        import torch.nn.functional as F
        pending={self.key(text):text for text in texts if self.cache.get('nll',self.key(text)) is None}
        items=list(pending.items())
        for start in range(0,len(items),self.cfg['target_batch_size']):
            if self.monitor is not None: self.monitor.guard()
            chunk=items[start:start+self.cfg['target_batch_size']]
            encodings=[self.tokenizer(text,add_special_tokens=True,truncation=True,max_length=self.cfg['max_length']) for _,text in chunk]
            eligible=[i for i,e in enumerate(encodings) if len(e.input_ids)>=2]
            for i,e in enumerate(encodings):
                if len(e.input_ids)<2:
                    self.cache.put('nll',chunk[i][0],dict(nll=None,tokens=len(e.input_ids),reason='no_causal_target'))
            if not eligible: continue
            width=max(len(encodings[i].input_ids) for i in eligible)
            pad=self.tokenizer.pad_token_id
            if pad is None: pad=self.tokenizer.eos_token_id
            if pad is None: raise ValueError('No existing padding/EOS token; tokenizer vocabulary will not be mutated')
            ids=torch.full((len(eligible),width),pad,device=self.model.device,dtype=torch.long)
            mask=torch.zeros_like(ids)
            for row,i in enumerate(eligible):
                seq=encodings[i].input_ids
                ids[row,:len(seq)]=torch.tensor(seq,device=self.model.device);mask[row,:len(seq)]=1
            torch.cuda.synchronize(); started=time.perf_counter()
            with torch.no_grad():
                output=self.model(input_ids=ids,attention_mask=mask,use_cache=False,output_hidden_states=False,return_dict=True)
                # logits[t] predicts input_ids[t+1]. First token and all pads excluded.
                total=torch.zeros(len(eligible),dtype=torch.float64,device=self.model.device)
                counts=(mask[:,1:]*mask[:,:-1]).sum(dim=1)
                for offset in range(0,width-1,self.cfg['nll_logit_chunk_size']):
                    end=min(width-1,offset+self.cfg['nll_logit_chunk_size'])
                    block=output.logits[:,offset:end,:].float()
                    losses=F.cross_entropy(block.transpose(1,2),ids[:,offset+1:end+1],reduction='none')
                    valid=mask[:,offset+1:end+1]*mask[:,offset:end]
                    total+=(losses*valid).sum(dim=1,dtype=torch.float64)
                values=(total/counts).cpu().tolist()
            torch.cuda.synchronize(); elapsed=time.perf_counter()-started
            self.calls+=1;self.texts+=len(eligible);self.seconds+=elapsed
            for row,i in enumerate(eligible):
                value=float(values[row])
                if not __import__('math').isfinite(value): raise ValueError('Nonfinite causal NLL')
                self.cache.put('nll',chunk[i][0],dict(nll=value,tokens=len(encodings[i].input_ids),
                               target_batch_seconds=elapsed,amortized_seconds=elapsed/len(eligible)))
            del output,ids,mask,total,block,losses
        return [self.cache.get('nll',self.key(text)) for text in texts]


def load_target(cfg):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError('Enable Kaggle CUDA; CPU inference is not supported')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['model_revision'],use_fast=True,
                  cache_dir=cfg['model_cache_dir'],local_files_only=cfg['model_local_files_only'])
    if not tokenizer.is_fast: raise RuntimeError('Fast character offsets required')
    model=AutoModelForCausalLM.from_pretrained(cfg['model_id'],revision=cfg['model_revision'],
                  cache_dir=cfg['model_cache_dir'],local_files_only=cfg['model_local_files_only'],
                  torch_dtype=getattr(torch,cfg['dtype']),device_map={'':0})
    model.eval();model.config.use_cache=False
    return model,tokenizer


def validate_heldout_manifest(manifest):
    """Fail closed: this experiment reuses the existing 1,000 IDs verbatim."""
    required = {'sample_id','language','is_member','membership','content_sha256',
                'official_split','source_row','split_role'}
    if (not required <= set(manifest) or len(manifest) != 1000
            or manifest.sample_id.duplicated().any() or manifest[list(required)].isna().any().any()
            or not manifest.split_role.eq('heldout_test').all()):
        raise ValueError('Require the existing 1,000-sample SERSEM held-out manifest; do not resample')
    if not manifest.membership.map({'member':1,'non-member':0}).eq(manifest.is_member).all():
        raise ValueError('Invalid membership labels')


def validate_saved_scores(manifest, baseline):
    required = {'sample_id','language','is_member','content_sha256',
                'lumia_score','greybox_score','fusion_score'}
    import numpy as np
    if not required <= set(baseline) or baseline.sample_id.duplicated().any():
        raise ValueError('Require saved LUMIA, grey-box and fusion scores with unique identities')
    joined = manifest.merge(baseline, on='sample_id', how='left', suffixes=('', '_sersem'), validate='one_to_one')
    for field in ['language','is_member','content_sha256']:
        if not joined[field].eq(joined[field+'_sersem']).all():
            raise ValueError('Saved SERSEM score identity mismatch or missing ID: ' + field)
    if not np.isfinite(joined[['lumia_score','greybox_score','fusion_score']].to_numpy(dtype=float)).all():
        raise ValueError('Missing/nonfinite saved SERSEM scores')


def evaluate_results(results,manifest,baseline_path,out,partial=False):
    import numpy as np
    import pandas as pd
    out=Path(out)
    if results.empty: return {'status':'no completed results'}
    if results.sample_id.duplicated().any() or manifest.sample_id.duplicated().any(): raise ValueError('Duplicate sample IDs')
    reference=manifest.set_index('sample_id')
    for row in results.itertuples(index=False):
        if row.sample_id not in reference.index: raise ValueError('Unexpected result ID')
        expected=reference.loc[row.sample_id]
        if int(row.is_member)!=int(expected.is_member) or row.language!=expected.language or row.content_sha256!=expected.content_sha256:
            raise ValueError('Result label/language/content identity changed')
    valid=results[results.status.eq('ok') & results.neighbourhood_score.notna()].copy()
    coverage=dict(requested=len(manifest),completed=len(results),valid=len(valid),failed=int(sum(results.status.ne('ok'))),
                  coverage_fraction=len(valid)/len(manifest) if len(manifest) else 0.,
                  partial=partial,all_requested_ids_completed=set(results.sample_id)==set(manifest.sample_id))
    coverage['by_language_and_label']={}
    for (language,label),group in manifest.groupby(['language','is_member']):
        coverage['by_language_and_label'][f'{language}/{label}']=dict(requested=len(group),
            completed=int(results.sample_id.isin(group.sample_id).sum()),
            valid=int(valid.sample_id.isin(group.sample_id).sum()),
            coverage_fraction=float(valid.sample_id.isin(group.sample_id).sum()/len(group)))
    report={'coverage':coverage,'score_formula':'mean(neighbour NLL)-original NLL',
            'metrics':metric_report(valid,['neighbourhood_score']) if len(valid) else {},
            'comparison_status':'SERSEM score file unavailable; rerun report when it is available'}
    configuration_path=out/'config.json'
    if configuration_path.exists():
        configuration=json.loads(configuration_path.read_text())
        report['run_scope']={k:configuration[k] for k in ['site_importance','mode','k','model_id','model_revision','dataset_id','dataset_revision','max_length','dtype','seed','heldout_manifest_sha256','full_heldout_manifest_sha256','baseline_scores_sha256','structural_weight_provenance'] if k in configuration}
    if baseline_path and Path(baseline_path).is_file() and len(valid):
        baseline=pd.read_parquet(baseline_path) if str(baseline_path).endswith('.parquet') else pd.read_csv(baseline_path)
        if baseline.sample_id.duplicated().any(): raise ValueError('Duplicate SERSEM sample IDs')
        columns=[c for c in ['lumia_score','greybox_score','fusion_score'] if c in baseline]
        joined=valid.merge(baseline,on='sample_id',how='inner',suffixes=('','_sersem'),validate='one_to_one')
        for field in ['language','is_member','content_sha256']:
            if field+'_sersem' not in joined or not joined[field].eq(joined[field+'_sersem']).all():
                raise ValueError('SERSEM/neighbourhood IDs do not have identical labels/language/content: '+field)
        for column in columns:
            if not np.isfinite(joined[column]).all(): raise ValueError('Invalid SERSEM scores')
        report['comparison_status']='exact matched IDs/labels on common successful cohort; see coverage'
        report['comparison_coverage']=dict(matched=len(joined),neighbourhood_valid=len(valid),
                                         baseline_available=len(baseline),requested=len(manifest),
                                         complete=bool(set(joined.sample_id)==set(manifest.sample_id)))
        report['matched_metrics']=metric_report(joined,['neighbourhood_score']+columns) if len(joined) else {}
        overlap={}
        if len(joined):
            _, ours=auc_metrics(joined,'neighbourhood_score')
            for column in columns:
                _, detected=auc_metrics(joined,column)
                overlap[column]=dict(neighbourhood_detected_members=len(ours),sersem_detected_members=len(detected),
                                     intersection=len(ours&detected),neighbourhood_only=len(ours-detected),
                                     sersem_only=len(detected-ours),union=len(ours|detected),
                                     jaccard=len(ours&detected)/len(ours|detected) if ours|detected else None,
                                     intersection_ids=sorted(ours&detected),neighbourhood_only_ids=sorted(ours-detected))
        report['member_overlap_at_empirical_fpr_le_1pct']=overlap
        joined.to_parquet(out/('matched_partial.parquet' if partial else 'matched_comparison.parquet'),index=False)
    if (out/'runtime.json').exists(): report['runtime']=json.loads((out/'runtime.json').read_text())
    history_path=out/'runtime_history.jsonl'
    if history_path.exists():
        sessions={}
        for line in history_path.read_text().splitlines():
            event=json.loads(line); session=event.get('session_id','legacy')
            sessions[session]=max(sessions.get(session,0),event['session_seconds'])
        report['observed_runtime_across_sessions_seconds']=sum(sessions.values())
        report['runtime_scope']='Recorded generation/scoring/I/O elapsed; excludes target initialization/setup and unlogged work at abrupt interruption.'
    write_json(out/('partial_metrics.json' if partial else 'metrics.json'),report)
    return report


def run(cfg,manifest,source_store,model,tokenizer,out,monitor,baseline_scores=''):
    import pandas as pd
    import torch
    from neighbourhood_attack.perturb import Perturber
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    identity=sha(json.dumps({k:v for k,v in cfg.items() if k not in ['model_cache_dir','mlm_cache_dir','model_local_files_only',
                        'mlm_local_files_only','target_batch_size','mlm_batch_size','source_chunk_size','nll_reuse_from']},sort_keys=True))
    keys={r.sample_id:sha(identity+r.sample_id+r.content_sha256) for r in manifest.itertuples(index=False)}
    cache=Cache(out/'neighbourhood.sqlite')
    reused=reuse_nll_cache(cache,cfg.get('nll_reuse_from',''),cfg)
    if reused: write_json(out/'imported_nll_cache.json',dict(source=cfg['nll_reuse_from'],records_checked=reused))
    target=TargetNLL(model,tokenizer,cfg,cache); target.monitor=monitor
    completed={sid:cache.get('result',key) for sid,key in keys.items()}
    completed={sid:value for sid,value in completed.items() if value is not None}
    initial=len(completed); started=time.perf_counter();generator=None
    session_id=__import__('uuid').uuid4().hex
    def snapshot(n):
        records=pd.DataFrame(list(completed.values())[:n])
        records.to_parquet(out/f'checkpoint_{n}_scores.parquet',index=False)
        checkpoint=evaluate_results(records,manifest,baseline_scores,out,partial=True)
        write_json(out/f'checkpoint_{n}_metrics.json',checkpoint)
    try:
        for n in MILESTONES:
            if len(completed)>=n and not (out/f'checkpoint_{n}_scores.parquet').exists(): snapshot(n)
        pending=manifest[~manifest.sample_id.isin(completed)]
        for batch in source_store.batches(pending,cfg['source_chunk_size']):
            monitor.guard()
            generated=[]
            generation_started=time.perf_counter()
            missing=[r for r in batch.itertuples(index=False) if cache.get('generation',keys[r.sample_id]) is None]
            if missing:
                if generator is None:
                    generator=Perturber(cfg)
                    write_json(out/'mlm_loading_info.json',generator.loading_info)
                generator.model.to('cuda')
                for row in missing:
                    monitor.guard()
                    try:
                        neighbours,failures=generator.generate(row.content,row.language,tokenizer)
                    except (ValueError,SyntaxError,UnicodeError,RecursionError) as error:
                        neighbours,failures=[],[dict(reason='transformation_exception',error_type=type(error).__name__,message=str(error)[:500])]
                    cache.put('generation',keys[row.sample_id],dict(neighbours=neighbours,failures=failures,selected_sites=generator.last_selection))
                if cfg['park_mlm_on_cpu']:
                    generator.model.to('cpu');torch.cuda.empty_cache()
            generation_seconds=time.perf_counter()-generation_started
            texts=[]
            for row in batch.itertuples(index=False):
                generated.append(cache.get('generation',keys[row.sample_id]))
                texts.append(row.content);texts.extend(n['content'] for n in generated[-1]['neighbours'])
            scoring_started=time.perf_counter(); target.score(texts)
            scoring_seconds=time.perf_counter()-scoring_started
            for row,group in zip(batch.to_dict(orient='records'),generated):
                original=target.score([row['content']])[0]
                neighbours=target.score([n['content'] for n in group['neighbours']])
                status='ok' if len(neighbours)==cfg['k'] and original['nll'] is not None and all(n['nll'] is not None for n in neighbours) else 'transformation_or_nll_failure'
                result={k:v for k,v in row.items() if k not in ['content','manifest_position']}
                result.update(status=status,original_nll=original['nll'],neighbour_nlls=[n['nll'] for n in neighbours],
                              neighbour_count=len(neighbours),neighbourhood_score=sum(n['nll'] for n in neighbours)/cfg['k']-original['nll'] if status=='ok' else None,
                              transformation_failures=group['failures'],selected_sites=group['selected_sites'],
                              nll_failures=([dict(reason=original.get('reason','original_nll_unavailable'),source='original')] if original['nll'] is None else [])
                                  + [dict(reason=n.get('reason','neighbour_nll_unavailable'),source='neighbour',index=i) for i,n in enumerate(neighbours) if n['nll'] is None],
                              neighbour_transformations=[{k:v for k,v in n.items() if k != 'content'} for n in group['neighbours']],
                              chunk_generation_seconds=generation_seconds,chunk_scoring_seconds=scoring_seconds,
                              chunk_seconds_amortized=(generation_seconds+scoring_seconds)/len(batch))
                monitor.guard()
                cache.put('result',keys[row['sample_id']],result);completed[row['sample_id']]=result
                if len(completed) in MILESTONES and not (out/f'checkpoint_{len(completed)}_scores.parquet').exists(): snapshot(len(completed))
            del texts,generated
            monitor.tick('neighbourhood_completed',len(completed),len(manifest),force=True)
            elapsed=time.perf_counter()-started; fresh=len(completed)-initial
            event=dict(session_id=session_id,completed=len(completed),newly_completed=fresh,session_seconds=elapsed,
                       examples_per_second=fresh/elapsed if elapsed else None,
                       remaining_seconds_from_observed_rate=(len(manifest)-len(completed))*elapsed/fresh if fresh else None,
                       target_forward_batches=target.calls,target_texts_scored=target.texts,target_nll_seconds=target.seconds,
                       note='Measured current-session generation+NLL+I/O; early ETA is length/cache-mixture dependent')
            write_json(out/'runtime.json',event)
            with (out/'runtime_history.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        records=pd.DataFrame([completed[sid] for sid in manifest.sample_id])
        records.to_parquet(out/'neighbourhood_scores.parquet',index=False)
        records.to_csv(out/'neighbourhood_scores.csv',index=False)
        report=evaluate_results(records,manifest,baseline_scores,out)
        if (out/'runtime.json').exists():
            report['runtime']=json.loads((out/'runtime.json').read_text());write_json(out/'metrics.json',report)
        return report
    finally:
        if generator is not None and cfg['park_mlm_on_cpu']:
            generator.model.to('cpu')
        cache.close()

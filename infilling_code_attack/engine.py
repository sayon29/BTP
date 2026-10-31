"""One fixed Infilling Score; independent original, altered and sample checkpoints."""
import json
import time
from infilling_code_attack.cache import fingerprint,sample_identity
from infilling_code_attack.method import future_positions,token_score,aggregate,InvalidSample


class Engine:
    def __init__(self,cache,identity,factory,guard=lambda:None,audit_path=None):
        self.cache,self.identity,self.factory,self.guard=cache,identity,factory,guard
        self.signature=fingerprint(identity);self.scorer=None;self.audit_path=audit_path
        self.forward_calls=0;self.head_calls=0;self.model_seconds=0.

    def query_key(self,request):return fingerprint(dict(scoring_sha256=self.signature,request=request))

    def query(self,requests):
        unique={self.query_key(r):r for r in requests};results={};missing=[]
        for key,request in unique.items():
            value=self.cache.get('query',key)
            if value is None:missing.append((key,request))
            else:results[key]=value
        size=self.identity['altered_batch_size']
        # Bucket similar lengths: right-padded exact causal calls, not a context approximation.
        missing.sort(key=lambda pair:(len(pair[1]['ids']),pair[0]))
        for start in range(0,len(missing),size):
            batch=missing[start:start+size];self.guard()
            if self.scorer is None:self.scorer=self.factory()
            self.scorer.synchronize();before=self.scorer.forward_calls;heads=self.scorer.head_calls
            sequence=time.time_ns();began=time.perf_counter()
            def log(event):
                if self.audit_path:
                    with open(self.audit_path,'a') as file:file.write(json.dumps(event,allow_nan=False)+'\n');file.flush()
            log(dict(event='started',sequence=sequence,operation='predict_batch',queries=len(batch)))
            try:values=self.scorer.predict_batch([r for _,r in batch])
            finally:
                self.scorer.synchronize();seconds=time.perf_counter()-began
                calls=self.scorer.forward_calls-before;hc=self.scorer.head_calls-heads
                self.forward_calls+=calls;self.head_calls+=hc;self.model_seconds+=seconds
                log(dict(event='finished',sequence=sequence,transformer_forward_calls=calls,head_forward_calls=hc,seconds=seconds))
            if len(values)!=len(batch):raise ValueError('Missing query result')
            for (key,request),value in zip(batch,values):
                if set(value)!=set(map(str,request['positions'])):raise ValueError('Missing/aligned probability statistics')
                self.guard();self.cache.put('query',key,value);results[key]=value
        return [results[self.query_key(r)] for r in requests]

    def score(self,row,plan):
        key=fingerprint(dict(sample=sample_identity(row),plan=plan,scoring_sha256=self.signature))
        saved=self.cache.get('sample',key)
        if saved is not None:return dict(saved,cache_hit=True,session_seconds=0.,session_forward_calls=0,session_head_calls=0)
        began=time.perf_counter();before=self.forward_calls;heads=self.head_calls
        summary=dict(status='invalid',invalid_reason=plan['invalid_reason'],infilling_score=None,bottom_count=0,bottom_positions=[],token_scores=[],raw_log_ratios=[],greedy_ids=[])
        if not plan['invalid_reason']:
            ids=plan['input_ids'];n=len(ids);targets=plan['target_positions'];m=self.identity['future_tokens']
            original=self.query([dict(ids=ids,positions=list(range(1,n)))])[0]
            requests=[];request_positions=[];altered={}
            for t in targets:
                greedy=original[str(t)]['greedy_id'];future=future_positions(t,n,m)
                # Identical branch is algebraically zero; no repeated model call.
                if greedy==ids[t] or not future:
                    altered[t]={str(j):original[str(j)] for j in future}
                else:
                    branch=list(ids[:future[-1]+1]);branch[t]=greedy
                    requests.append(dict(ids=branch,positions=future));request_positions.append(t)
            if requests:
                for t,value in zip(request_positions,self.query(requests)):altered[t]=value
            try:
                values=[];raw=[]
                for t in targets:
                    value,ratio=token_score(original,altered[t],t,n,m);values.append(value);raw.append(ratio)
                summary=dict(aggregate(targets,values,self.identity['bottom_fraction']),status='valid',invalid_reason=None,
                    token_scores=values,raw_log_ratios=raw,greedy_ids=[original[str(t)]['greedy_id'] for t in targets])
            except InvalidSample as error:summary['invalid_reason']=str(error)
        summary.update(sample_cache_key=key,scored_target_tokens=len(plan['target_positions']),
            span_ids_sha256=fingerprint(plan['input_ids']),target_ids_sha256=fingerprint([plan['input_ids'][t] for t in plan['target_positions']]),
            computation_seconds=time.perf_counter()-began,transformer_forward_calls=self.forward_calls-before,head_forward_calls=self.head_calls-heads)
        self.guard();self.cache.put('sample',key,summary)
        return dict(summary,cache_hit=False,session_seconds=summary['computation_seconds'],
                    session_forward_calls=summary['transformer_forward_calls'],session_head_calls=summary['head_forward_calls'])

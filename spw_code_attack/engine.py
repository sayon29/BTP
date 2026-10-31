"""Dynamic shortening with resumable canonical-prefix and per-sample caches."""
from collections import defaultdict
import json
import time
from spw_code_attack.cache import fingerprint, sample_identity
from spw_code_attack.method import next_length, aggregate, validate_prediction
from spw_code_attack import METHOD_NAME, METHOD_STATUS


def prediction_key(prefix, identity):
    signature=identity if isinstance(identity,str) else fingerprint(identity)
    return fingerprint(dict(inference_sha256=signature,prefix_ids=prefix,positions='zero-based'))


def select(record, true_id):
    value=dict(log_true=record['log_true'][str(true_id)],log_greedy=record['log_greedy'],greedy_id=record['greedy_id'])
    validate_prediction(value)
    return value


class Engine:
    def __init__(self, cache, identity, predictor_factory, guard=lambda:None, audit_path=None):
        self.cache,self.identity,self.factory,self.guard=cache,identity,predictor_factory,guard
        self.predictor=None;self.audit_path=audit_path
        self.inference_signature=fingerprint(identity)
        self.forward_calls=0;self.head_calls=0;self.model_seconds=0.

    def _get_predictor(self):
        if self.predictor is None:self.predictor=self.factory()
        return self.predictor

    def _call(self, function, *args):
        self.guard()
        predictor=self._get_predictor();predictor.synchronize()
        event=dict(event='started',operation=function,sequence=time.time_ns())
        def log(value):
            if self.audit_path:
                with open(self.audit_path,'a') as file:
                    file.write(json.dumps(value,allow_nan=False)+'\n');file.flush()
        log(event)
        before,heads=predictor.forward_calls,predictor.head_calls
        started=time.perf_counter()
        try:
            return getattr(predictor,function)(*args)
        finally:
            predictor.synchronize()
            elapsed=time.perf_counter()-started
            calls=predictor.forward_calls-before;head_calls=predictor.head_calls-heads
            self.forward_calls+=calls;self.head_calls+=head_calls;self.model_seconds+=elapsed
            log(dict(event='finished',operation=function,sequence=event['sequence'],
                     transformer_forward_calls=calls,head_forward_calls=head_calls,seconds=elapsed))

    def _store(self,key,record):
        for token in record['log_true']:select(record,int(token))
        self.guard()
        self.cache.extend_prediction(key,record)

    def query(self, requests):
        """requests=(prefix IDs, true ID); group exact prefixes then equal lengths."""
        grouped={}
        for prefix,target in requests:
            key=prediction_key(prefix,self.inference_signature)
            item=grouped.setdefault(key,dict(prefix=prefix,targets=set()))
            item['targets'].add(int(target))
        records={};pending=defaultdict(list)
        for key,item in grouped.items():
            record=self.cache.get('prediction',key)
            if record is not None:
                missing=item['targets']-set(map(int,record['log_true']))
                if missing:
                    record=self._call('extend',record,sorted(missing));self._store(key,record)
                records[key]=record
            else:
                pending[len(item['prefix'])].append((key,item))
        batch_size=self.identity['shortened_batch_size']
        for length in sorted(pending):
            items=sorted(pending[length],key=lambda pair:pair[0])
            for start in range(0,len(items),batch_size):
                batch=items[start:start+batch_size]
                values=self._call('predict_batch',[v['prefix'] for _,v in batch],[sorted(v['targets']) for _,v in batch])
                if len(values)!=len(batch):raise ValueError('Incomplete model batch')
                for (key,_),record in zip(batch,values):self._store(key,record);records[key]=record
        return [select(records[prediction_key(prefix,self.inference_signature)],target) for prefix,target in requests]

    def originals(self,plan):
        ids,positions=plan['input_ids'],plan['target_positions']
        # One teacher-forced pass for missing original predictions only. Already
        # cached probabilities/head projections are never queried again; their
        # tokens can still occur as context within a different, longer prefill.
        keys=[prediction_key(ids[:t],self.inference_signature) for t in positions]
        saved=[self.cache.get('prediction',key) for key in keys]
        missing=[t for t,value in zip(positions,saved) if value is None]
        if missing:
            values=self._call('original',ids,missing,self.identity['original_forward_chunk_tokens'])
            if set(values)!=set(missing):raise ValueError('Incomplete original targets')
            for t in missing:self._store(prediction_key(ids[:t],self.inference_signature),values[t])
        return self.query([(ids[:t],ids[t]) for t in positions])

    def score(self,row,plan):
        sample_key=fingerprint(dict(sample=sample_identity(row),plan=plan,scoring=self.identity))
        cached=self.cache.get('sample',sample_key)
        if cached is not None:return dict(cached,cache_hit=True,session_seconds=0.,session_forward_calls=0,session_head_calls=0)
        start=time.perf_counter();before=self.forward_calls;heads=self.head_calls
        original=self.originals(plan)
        ids=plan['input_ids'];positions=plan['target_positions']
        states=[dict(original_length=t,accepted_length=t,accepted=o,active=t>1,attempts=[]) for t,o in zip(positions,original)]
        # Each stage's prediction records commit before advancing to the next.
        # On crash, deterministically replay control flow entirely from cache.
        while any(s['active'] for s in states):
            active=[i for i,s in enumerate(states) if s['active']]
            requests=[]
            for i in active:
                t=positions[i];length=next_length(states[i]['accepted_length'])
                requests.append((ids[t-length:t],ids[t]))
            values=self.query(requests)
            for i,value,(prefix,_) in zip(active,values,requests):
                state=states[i];length=len(prefix)
                agrees=value['greedy_id']==original[i]['greedy_id']
                state['attempts'].append(dict(length=length,agrees=agrees,**value))
                if agrees:
                    state['accepted_length']=length;state['accepted']=value
                state['active']=agrees and length>1
        summary=aggregate(original,states)
        summary.update(method_name=METHOD_NAME,method_status=METHOD_STATUS,paper_faithful=False,
            scored_target_tokens=len(positions),original_predictions=original,shortening_states=states,
            preparation=plan['metadata'],target_ids_sha256=fingerprint([ids[t] for t in positions]),
            computation_seconds=time.perf_counter()-start,transformer_forward_calls=self.forward_calls-before,
            head_forward_calls=self.head_calls-heads,sample_cache_key=sample_key)
        self.guard();self.cache.put('sample',sample_key,summary)
        return dict(summary,cache_hit=False,session_seconds=summary['computation_seconds'],
                    session_forward_calls=summary['transformer_forward_calls'],session_head_calls=summary['head_forward_calls'])

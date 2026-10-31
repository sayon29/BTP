"""Synthetic CPU tests only. No datasets or pretrained model downloads."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from adaptive_token_attack.data import COUNTS, validate_partitions, select_partitions
from spw_code_attack.cache import Cache, fingerprint, sample_identity
from spw_code_attack.data import make_plan, prepare_plans, fixed_configuration
from spw_code_attack.engine import Engine, prediction_key
from spw_code_attack.method import bernoulli_js, token_weight, next_length, straightforward, aggregate
from spw_code_attack.inference import encode_hidden, decode_hidden
from spw_code_attack.evaluation import exact_match, calibrate, evaluate
from spw_code_attack.runner import run_scores, audit_report


def partitions():
    roles=dict(shadow='shadow_train',validation='fusion_fit',calibration='threshold_calibration',heldout='heldout_test')
    languages=['Go','Java','Python','Ruby','Rust'];result={};offset=0
    for role,count in COUNTS.items():
        rows=[]
        for i in range(count):
            lang=languages[(i//2)%5];split='test' if role=='heldout' else 'train'
            rows.append(dict(sample_id=f'{lang}/{split}/{offset+i}',source_row=offset+i,language=lang,
                official_split=split,split_role=roles[role],is_member=i%2,
                membership='member' if i%2 else 'non-member',content_sha256=hashlib.sha256(f'{role}:{i}'.encode()).hexdigest()))
        result[role]=pd.DataFrame(rows);offset+=count
    return result


class FakePredictor:
    """Deterministic causal vocabulary with a prefix-length-dependent decision."""
    def __init__(self):self.forward_calls=0;self.head_calls=0;self.prefix_evaluations=[]
    def synchronize(self):pass
    def distribution(self,prefix):
        logits=np.array([-.7,.3,-.2,.1],dtype=float)
        # Last token changes confidence; short prefixes can change the winner.
        logits[(sum(prefix)+len(prefix)//3)%4]+=.7
        return logits-np.log(np.exp(logits).sum())
    def record(self,prefix,targets):
        lp=self.distribution(prefix);g=int(lp.argmax())
        return dict(greedy_id=g,log_greedy=float(lp[g]),log_true={str(t):float(lp[t]) for t in targets},
                    terminal=list(lp))
    def predict_batch(self,prefixes,targets):
        assert len({len(p) for p in prefixes})==1
        self.forward_calls+=1;self.head_calls+=len(prefixes);self.prefix_evaluations.extend(map(tuple,prefixes))
        return [self.record(p,t) for p,t in zip(prefixes,targets)]
    def original(self,ids,positions,chunk_size):
        self.forward_calls+=math.ceil(max(positions)/chunk_size);self.head_calls+=len(positions)
        self.prefix_evaluations.extend(tuple(ids[:t]) for t in positions)
        return {t:self.record(ids[:t],[ids[t]]) for t in positions}
    def extend(self,record,targets):
        self.head_calls+=1
        result=dict(record,log_true=dict(record['log_true']))
        result['log_true'].update({str(t):float(record['terminal'][t]) for t in targets})
        return result
    def slow(self,prefix,target):
        r=self.record(prefix,[target]);return dict(greedy_id=r['greedy_id'],log_true=r['log_true'][str(target)],log_greedy=r['log_greedy'])


class FakeTokenizer:
    def __call__(self,text,add_special_tokens=True,truncation=False,max_length=None,**kwargs):
        n=int(text) if text.isdigit() else 300
        ids=[i%4 for i in range(n)];ids=ids[:max_length] if truncation else ids
        return dict(input_ids=ids,attention_mask=[1]*len(ids),offset_mapping=[(i,i+1) for i in range(len(ids))],special_tokens_mask=[0]*len(ids))


class SPWSynthetic(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.cache=Cache(self.out/'cache.sqlite');self.predictor=FakePredictor()
        self.identity=dict(fixed_configuration(),baseline_max_length=8192,context_limit=16384)
        self.original=partitions();self.selected,_=select_partitions(self.original,'smoke',seed=42)
        self.row=self.selected['heldout'].iloc[0].to_dict()
        self.plan=dict(input_ids=[0,1,2,3,0,1,2,3,0,2,1],target_positions=list(range(1,11)),metadata={'synthetic':True})
        self.engine=Engine(self.cache,self.identity,lambda:self.predictor,audit_path=self.out/'calls.jsonl')

    def tearDown(self):self.cache.close();self.temp.cleanup()

    def test_published_fidelity_is_never_claimed(self):
        self.assertFalse(self.identity['paper_faithful']);self.assertEqual(self.identity['seed'],42)
        source=json.loads(Path('spw_code_attack/research_access.json').read_text())
        self.assertIn('All numbered method equations',source['unavailable'])

    def test_js_known_values_symmetry_and_endpoints(self):
        self.assertAlmostEqual(bernoulli_js(0,1),math.log(2))
        self.assertEqual(bernoulli_js(.4,.4),0)
        self.assertEqual(bernoulli_js(0,0),0)
        self.assertEqual(bernoulli_js(1,1),0)
        self.assertAlmostEqual(bernoulli_js(.1,.8),bernoulli_js(.8,.1))
        for p,q in [(float('nan'),.5),(-.1,.2),(1.1,.5)]:
            with self.assertRaises(ValueError):bernoulli_js(p,q)

    def test_weights_exact_and_bounded(self):
        self.assertEqual(token_weight(8,8,math.log(.2),math.log(.2)),1.)
        self.assertEqual(token_weight(8,2,math.log(.2),math.log(.2)),1.75)
        self.assertGreater(token_weight(8,2,math.log(.2),math.log(.2)),token_weight(8,2,math.log(.2),math.log(.9)))
        with self.assertRaises(ValueError):token_weight(2,3,-1,-1)

    def test_dynamic_schedule_edges(self):
        self.assertIsNone(next_length(1));self.assertEqual(next_length(9),4);self.assertEqual(next_length(4),2)
        with self.assertRaises(ValueError):next_length(0)

    def test_optimized_equals_straightforward(self):
        fast=self.engine.score(self.row,self.plan)
        slow,states=straightforward(self.plan,self.predictor.slow)
        self.assertAlmostEqual(fast['spw_inspired_score'],slow['spw_inspired_score'],places=14)
        self.assertEqual(fast['weights'],slow['weights'])
        self.assertEqual([s['accepted_length'] for s in fast['shortening_states']],[s['accepted_length'] for s in states])
        self.assertEqual(fast['shortening_states'][0]['attempts'],[])
        self.assertTrue(all(1<=w<=2 for w in fast['weights']))

    def test_first_mismatch_stops_even_if_later_suffix_would_agree(self):
        class NonMonotonic(FakePredictor):
            def distribution(inner,prefix):
                logits=np.zeros(4);logits[0 if len(prefix)==4 else 1]=2.
                return logits-np.log(np.exp(logits).sum())
        predictor=NonMonotonic()
        engine=Engine(self.cache,self.identity,lambda:predictor)
        plan=dict(input_ids=[1]*9,target_positions=[8],metadata={})
        result=engine.score(self.row,plan)
        state=result['shortening_states'][0]
        self.assertEqual([a['length'] for a in state['attempts']],[4])
        self.assertEqual(state['accepted_length'],8)
        self.assertEqual(result['weights'],[1.])
        self.assertEqual([len(p) for p in predictor.prefix_evaluations],[8,4])

    def test_score_direction_and_finite_no_labels(self):
        o=[dict(log_true=math.log(.8),log_greedy=math.log(.8),greedy_id=1)]
        s=[dict(original_length=2,accepted_length=1,accepted=o[0])]
        self.assertEqual(aggregate(o,s)['spw_inspired_score'],0.)
        bad=[dict(log_true=math.log(.1),log_greedy=math.log(.8),greedy_id=1)]
        self.assertLess(aggregate(bad,s)['spw_inspired_score'],0.)
        a=self.engine.score(self.row,self.plan)
        b=self.engine.score(dict(self.row,is_member=1-self.row['is_member']),self.plan)
        self.assertEqual(a['spw_inspired_score'],b['spw_inspired_score']);self.assertTrue(b['cache_hit'])

    def test_deduplicate_and_batch_compatible_lengths(self):
        requests=[([1,2],0),([1,2],1),([1,2],0),([2,2],0),([1],2)]
        values=self.engine.query(requests)
        self.assertEqual(len(values),5)
        self.assertEqual(self.predictor.forward_calls,2)
        self.assertEqual(len(self.predictor.prefix_evaluations),3)
        # A different true token for the same prefix uses cached terminal state.
        before=self.predictor.forward_calls
        self.engine.query([([1,2],3)])
        self.assertEqual(self.predictor.forward_calls,before)
        self.assertEqual(self.predictor.head_calls,4)

    def test_resume_no_duplicate_calls(self):
        first=self.engine.score(self.row,self.plan);before=self.predictor.forward_calls
        self.cache.close();self.cache=Cache(self.out/'cache.sqlite')
        engine=Engine(self.cache,self.identity,lambda:(_ for _ in ()).throw(AssertionError('must not load model')))
        second=engine.score(self.row,self.plan)
        self.assertEqual(first['spw_inspired_score'],second['spw_inspired_score'])
        self.assertTrue(second['cache_hit']);self.assertEqual(second['session_forward_calls'],0)
        self.assertEqual(before,self.predictor.forward_calls)

    def test_partial_recovery_preserves_committed_prefixes(self):
        original_store=self.engine._store;counter=[0]
        def fail_after_two(key,record):
            original_store(key,record);counter[0]+=1
            if counter[0]==2:raise RuntimeError('crash after durable prediction commit')
        with patch.object(self.engine,'_store',side_effect=fail_after_two):
            with self.assertRaises(RuntimeError):self.engine.score(self.row,self.plan)
        protected={tuple(self.plan['input_ids'][:t]) for t in self.plan['target_positions'][:2]}
        self.predictor.prefix_evaluations=[]
        result=self.engine.score(self.row,self.plan)
        self.assertFalse(protected & set(self.predictor.prefix_evaluations))
        reference,_=straightforward(self.plan,self.predictor.slow)
        self.assertAlmostEqual(result['spw_inspired_score'],reference['spw_inspired_score'],places=14)

    def test_prediction_extension_immutable(self):
        self.engine.query([([1,2],0)])
        key=prediction_key([1,2],self.identity);record=self.cache.get('prediction',key)
        bad=copy.deepcopy(record);bad['log_greedy']-=.1
        with self.assertRaises(ValueError):self.cache.extend_prediction(key,bad)
        bad=copy.deepcopy(record);bad['log_true']['0']-=.1
        with self.assertRaises(ValueError):self.cache.extend_prediction(key,bad)

    def test_checksum_cache_import_and_version_invalidation(self):
        self.engine.score(self.row,self.plan)
        other=Cache(self.out/'other.sqlite');other.reuse_from(self.cache.path)
        engine=Engine(other,self.identity,lambda:(_ for _ in ()).throw(AssertionError('cached')))
        self.assertTrue(engine.score(self.row,self.plan)['cache_hit']);other.close()
        self.assertNotEqual(prediction_key([1],self.identity),prediction_key([1],dict(self.identity,seed=43)))
        self.cache.con.execute("UPDATE cache SET sha='bad'");self.cache.con.commit()
        with self.assertRaises(ValueError):self.engine.score(self.row,self.plan)

    def test_eligibility_span_and_initial_token(self):
        plan=make_plan('300',FakeTokenizer(),self.identity)
        self.assertEqual(plan['target_positions'],list(range(1,257)))
        self.assertEqual(len(plan['input_ids']),257)
        self.assertFalse(plan['metadata']['short_sample'])
        short=make_plan('5',FakeTokenizer(),self.identity)
        self.assertEqual(short['target_positions'],[1,2,3,4]);self.assertTrue(short['metadata']['short_sample'])
        with self.assertRaises(ValueError):make_plan('1',FakeTokenizer(),self.identity)
        cap=make_plan('300',FakeTokenizer(),dict(self.identity,context_limit=100))
        self.assertEqual(cap['metadata']['context_or_baseline_truncated_tokens'],200)
        self.assertEqual(len(cap['target_positions']),99)

    def test_special_and_empty_offset_targets_excluded(self):
        class Specials(FakeTokenizer):
            def __call__(self,*args,**kwargs):
                e=super().__call__(*args,**kwargs)
                e['special_tokens_mask'][2]=1;e['offset_mapping'][3]=(0,0)
                return e
        plan=make_plan('10',Specials(),self.identity)
        self.assertNotIn(0,plan['target_positions']);self.assertNotIn(2,plan['target_positions']);self.assertNotIn(3,plan['target_positions'])
        self.assertEqual(plan['input_ids'][2],2)  # excluded target still remains in context

    def test_lossless_terminal_cache(self):
        for dtype in (np.float16,np.float32):
            hidden=np.array([.123456,-7.43,5.15],dtype=dtype)
            decoded=decode_hidden(encode_hidden(hidden),len(hidden),str(hidden.dtype))
            np.testing.assert_array_equal(decoded,hidden)

    def test_partitions_stratification_identity_and_leakage(self):
        self.assertEqual({k:len(v) for k,v in self.selected.items()},self.identity['smoke_counts'])
        for frame in self.selected.values():
            for _,group in frame.groupby('language'):self.assertEqual(group.is_member.sum(),len(group)//2)
        full,_=select_partitions(self.original,'full',seed=42)
        for k in full:pd.testing.assert_frame_equal(full[k],self.original[k])
        bad=copy.deepcopy(self.original);bad['shadow'].loc[0,'content_sha256']=bad['heldout'].iloc[0].content_sha256
        with self.assertRaises(ValueError):validate_partitions(bad)

    def test_exact_comparison_and_calibration_only(self):
        f=self.selected['heldout'].copy();f['spw_inspired_score']=-.1
        exact_match(self.selected['heldout'],f,['spw_inspired_score'])
        bad=f.copy();bad.loc[0,'source_row']=-1
        with self.assertRaises(ValueError):exact_match(self.selected['heldout'],bad,['spw_inspired_score'])
        with self.assertRaises(ValueError):calibrate(f)
        cal=self.selected['calibration'].copy();cal['spw_inspired_score']=-.1
        result=calibrate(cal);self.assertGreater(result['threshold'],-.1);self.assertEqual(result['calibration_fpr'],0.)

    def test_paired_metrics_and_no_heldout_threshold_selection(self):
        scores=pd.concat(list(self.selected.values()),ignore_index=True);scores['spw_inspired_score']=-.4+scores.is_member*.2
        base=self.selected['heldout'].copy();adaptive=base.copy()
        for key in ['lumia_score','greybox_score','fusion_score']:base[key]=base.is_member*.2
        adaptive['adaptive_score']=adaptive.is_member*.3
        with patch('spw_code_attack.evaluation.atomic_parquet'):
            first=evaluate(scores,self.selected,base,adaptive,self.out,'smoke')
            scores.loc[scores.split_role.eq('heldout_test'),'spw_inspired_score']*=-1
            second=evaluate(scores,self.selected,base,adaptive,self.out,'smoke')
        self.assertEqual(first['deployed_calibrated_threshold']['threshold'],second['deployed_calibrated_threshold']['threshold'])
        self.assertEqual(first['coverage']['original_heldout_fraction'],.03)
        self.assertEqual(len(first['heldout_diagnostic_roc_only']),5)
        self.assertFalse(first['paper_faithful'])

    def test_audit_calls_and_interrupted_unknown_accounting(self):
        self.engine.score(self.row,self.plan)
        report=audit_report(self.out/'calls.jsonl')
        self.assertEqual(report['transformer_forward_calls'],self.predictor.forward_calls)
        self.assertTrue(report['exact_accounting'])
        with (self.out/'calls.jsonl').open('a') as file:file.write(json.dumps(dict(event='started',sequence=1,operation='predict_batch'))+'\n')
        self.assertEqual(audit_report(self.out/'calls.jsonl')['interrupted_unaccounted_operations'],1)

    def test_preparation_and_orchestration_cached_resume(self):
        monitor=SimpleNamespace(guard=lambda:None,tick=lambda *args:None)
        contents={}
        for role,frame in self.original.items():
            for i,row in enumerate(frame.to_dict('records')):contents[row['sample_id']]=f'{role}:{i}'
        class Sources:
            calls=0
            def batches(inner,manifest,size):
                inner.calls+=1
                for start in range(0,len(manifest),size):
                    frame=manifest.iloc[start:start+size].copy();frame['content']=frame.sample_id.map(contents);yield frame
        class Tiny(FakeTokenizer):
            def __call__(self,text,**kwargs):return super().__call__('5',**kwargs)
        source=Sources();plans=prepare_plans(self.selected,Tiny(),self.identity,self.cache,source,monitor)
        again=prepare_plans(self.selected,Tiny(),self.identity,self.cache,source,monitor)
        self.assertEqual(plans,again);self.assertEqual(source.calls,1)
        with patch('spw_code_attack.runner.atomic_parquet'):
            scored,timing=run_scores(self.selected,plans,self.identity,self.cache,lambda:self.predictor,monitor,self.out)
            restored,timing2=run_scores(self.selected,plans,self.identity,self.cache,lambda:(_ for _ in ()).throw(AssertionError('no calls')),monitor,self.out)
        self.assertEqual(len(scored),150)
        self.assertTrue(restored.cache_hit.all());self.assertEqual(timing2['session_transformer_forward_calls'],0)
        np.testing.assert_array_equal(scored.spw_inspired_score,restored.spw_inspired_score)

    def test_notebook_syntax_when_present(self):
        path=Path('notebooks/05_spw_code.ipynb')
        if not path.exists():self.skipTest('Notebook is a separate artifact')
        nb=json.loads(path.read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));self.assertEqual(c['outputs'],[])
        self.assertIn('inspired',''.join(nb['cells'][0]['source']))


class TorchSynthetic(unittest.TestCase):
    def test_tiny_random_starcoder_alignment_batch_equivalence_and_extension(self):
        try:
            import torch
            from transformers import Starcoder2Config,Starcoder2ForCausalLM
        except ImportError:self.skipTest('Optional torch/transformers unavailable locally; no downloads')
        from spw_code_attack.inference import TorchPredictor
        threads=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,threads)
        torch.manual_seed(42)
        cfg=Starcoder2Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=512,sliding_window=512,
            attention_dropout=0.,embedding_dropout=0.,residual_dropout=0.)
        cfg._attn_implementation='eager'
        model=Starcoder2ForCausalLM(cfg).eval().requires_grad_(False);predictor=TorchPredictor(model)
        ids=[1,2,3,4,5,6,7];positions=list(range(1,len(ids)))
        records=predictor.original(ids,positions,3)
        with torch.inference_mode():lp=torch.log_softmax(model(torch.tensor([ids]),use_cache=False).logits.float(),-1)[0]
        for t in positions:
            self.assertAlmostEqual(records[t]['log_true'][str(ids[t])],float(lp[t-1,ids[t]]),places=5)
            self.assertEqual(records[t]['greedy_id'],int(lp[t-1].argmax()))
        batch=predictor.predict_batch([[1,2],[3,4]],[[3],[5]])
        solo=predictor.predict_batch([[1,2]],[[3]])
        self.assertAlmostEqual(batch[0]['log_true']['3'],solo[0]['log_true']['3'],places=5)
        before=predictor.forward_calls
        extended=predictor.extend(batch[0],[7]);self.assertEqual(predictor.forward_calls,before)
        fresh=predictor.predict_batch([[1,2]],[[7]])[0]
        self.assertAlmostEqual(extended['log_true']['7'],fresh['log_true']['7'],places=5)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            cache=Cache(Path(tmp)/'torch.sqlite')
            identity=dict(fixed_configuration())
            engine=Engine(cache,identity,lambda:predictor)
            row=dict(sample_id='Python/test/1',language='Python',content_sha256='synthetic',
                official_split='test',source_row=1,split_role='heldout_test')
            plan=dict(input_ids=ids,target_positions=positions,metadata={})
            fast=engine.score(row,plan)
            def single(prefix,target):
                r=predictor.predict_batch([prefix],[[target]])[0]
                return dict(greedy_id=r['greedy_id'],log_greedy=r['log_greedy'],log_true=r['log_true'][str(target)])
            slow,states=straightforward(plan,single)
            self.assertAlmostEqual(fast['spw_inspired_score'],slow['spw_inspired_score'],places=6)
            np.testing.assert_allclose(fast['weights'],slow['weights'],atol=1e-6,rtol=0)
            cache.close()



if __name__=='__main__':unittest.main()

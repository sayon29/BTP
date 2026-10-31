"""CPU synthetic validation: causal Bayes derivation, author indexing and resume."""
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
from adaptive_token_attack.data import COUNTS,select_partitions,validate_partitions
from infilling_code_attack.cache import Cache
from infilling_code_attack.data import make_plan,prepare_plans,fixed_configuration
from infilling_code_attack.method import statistics,standardized,token_score,future_positions,aggregate,reference,bayes_log_ratio,InvalidSample
from infilling_code_attack.engine import Engine
from infilling_code_attack.evaluation import evaluate,exact_match,calibrate
from infilling_code_attack.runner import run_scores,audit_report


def partitions():
    roles=dict(shadow='shadow_train',validation='fusion_fit',calibration='threshold_calibration',heldout='heldout_test')
    langs=['Go','Java','Python','Ruby','Rust'];parts={};offset=0
    for role,count in COUNTS.items():
        rows=[]
        for i in range(count):
            lang=langs[(i//2)%5];split='test' if role=='heldout' else 'train'
            rows.append(dict(sample_id=f'{lang}/{split}/{offset+i}',source_row=offset+i,language=lang,
                official_split=split,split_role=roles[role],is_member=i%2,membership='member' if i%2 else 'non-member',
                content_sha256=hashlib.sha256(f'{role}:{i}'.encode()).hexdigest()))
        parts[role]=pd.DataFrame(rows);offset+=count
    return parts


class FakeScorer:
    def __init__(self):self.forward_calls=0;self.head_calls=0;self.requests=[]
    def synchronize(self):pass
    def distribution(self,prefix):
        logits=np.array([.6,-.4,.3,-.1]);logits[(sum(prefix)+len(prefix))%4]+=.8
        return logits-np.log(np.exp(logits).sum())
    def predict_batch(self,requests):
        self.forward_calls+=1;self.head_calls+=1;self.requests.extend(copy.deepcopy(requests))
        return [{str(t):statistics(self.distribution(r['ids'][:t]),r['ids'][t]) for t in r['positions']} for r in requests]


class Tokenizer:
    def __call__(self,text,add_special_tokens=True,truncation=False,max_length=None,**kwargs):
        n=int(text) if text.isdigit() else 12
        ids=[i%4 for i in range(n)];ids=ids[:max_length] if truncation else ids
        return dict(input_ids=ids,attention_mask=[1]*len(ids),special_tokens_mask=[0]*len(ids),offset_mapping=[(i,i+1) for i in range(len(ids))])


class InfillingSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name);self.cache=Cache(self.out/'cache.sqlite')
        self.identity=dict(fixed_configuration(),baseline_max_length=8192,context_limit=16384)
        self.scorer=FakeScorer();self.engine=Engine(self.cache,self.identity,lambda:self.scorer,audit_path=self.out/'calls.jsonl')
        self.original=partitions();self.selected,_=select_partitions(self.original,'smoke',seed=42)
        self.row=self.selected['heldout'].iloc[0].to_dict();self.plan=make_plan('14',Tokenizer(),self.identity)

    def tearDown(self):self.cache.close();self.tmp.cleanup()

    def test_weighted_predictive_mean_variance_and_sigma_floor(self):
        p=np.array([.7,.2,.1]);lp=np.log(p);r=statistics(lp,1)
        mu=float(p@lp);var=float(p@((lp-mu)**2))
        self.assertAlmostEqual(r['mu'],mu);self.assertAlmostEqual(r['sigma'],math.sqrt(var))
        self.assertNotAlmostEqual(r['sigma'],float(lp.std()))
        uniform=statistics(np.log([.5,.5]),0);self.assertEqual(uniform['sigma_used'],1e-6)

    def test_raw_bayes_ratio_equals_four_term_causal_calculation(self):
        prefix=[0,1];true=2;g=int(self.scorer.distribution(prefix).argmax());future=[3,1]
        posterior=bayes_log_ratio(prefix,true,g,future,self.scorer.distribution,4)
        lp=self.scorer.distribution(prefix);a=float(lp[true]);b=float(lp[g]);ca=prefix+[true];cb=prefix+[g]
        for token in future:
            a+=float(self.scorer.distribution(ca)[token]);b+=float(self.scorer.distribution(cb)[token])
            ca+=[token];cb+=[token]
        self.assertAlmostEqual(posterior,a-b,places=14)

    def test_ratio_is_not_clipped_at_one(self):
        def predict(prefix):
            if len(prefix)==1:return np.log([.5,.4,.1])
            return np.log([.01,.01,.98]) if prefix[1]==1 else np.log([.98,.01,.01])
        r=bayes_log_ratio([2],1,0,[2],predict,3)
        self.assertGreater(r,0);self.assertAlmostEqual(r,math.log(.4*.98/(.5*.01)))

    def test_author_boundary_loop_translation_for_every_position(self):
        for n in [3,7,128]:
            for t in range(1,n):
                i=t-1;expected=[j+2 for j in range(i,min(i+5,n-3))]
                self.assertEqual(future_positions(t,n),expected)
        self.assertEqual(future_positions(1,10),[2,3,4,5,6])
        self.assertEqual(future_positions(8,10),[]);self.assertEqual(future_positions(9,10),[])

    def test_separate_altered_context_normalization_not_shared_sigma(self):
        original={str(t):dict(log_true=-2.-t/10,log_greedy=-.5,greedy_id=0,mu=-1.,sigma=.5+t/10,sigma_used=.5+t/10) for t in range(1,8)}
        alt={str(t):dict(log_true=-3.,log_greedy=-.4,greedy_id=0,mu=-.7,sigma=2.,sigma_used=2.) for t in range(2,7)}
        s,_=token_score(original,alt,1,8,5)
        expected=standardized(original['1'])-standardized(original['1'],'log_greedy')
        expected+=sum((original[str(j)]['log_true']-original[str(j)]['mu'])/original[str(j)]['sigma_used']
                      -(alt[str(j)]['log_true']-alt[str(j)]['mu'])/alt[str(j)]['sigma_used'] for j in range(2,7))
        self.assertAlmostEqual(s,expected)
        wrong=sum((original[str(j)]['log_true']-alt[str(j)]['log_true'])/original[str(j)]['sigma_used'] for j in range(2,7))
        self.assertNotAlmostEqual(s-(standardized(original['1'])-standardized(original['1'],'log_greedy')),wrong)

    def test_bottom_twenty_percent_is_mean_not_sum_or_full_average(self):
        scores=[-4,-3,-2,-1,0,1,2,3,4,5]
        r=aggregate(list(range(10)),scores)
        self.assertEqual(r['bottom_count'],2);self.assertEqual(r['infilling_score'],-3.5);self.assertEqual(r['bottom_positions'],[0,1])
        with self.assertRaises(InvalidSample):aggregate([1,2],[.1,.2])

    def test_optimized_matches_independent_prefix_reference(self):
        fast=self.engine.score(self.row,self.plan);slow=reference(self.plan,self.scorer.distribution)
        self.assertEqual(fast['status'],'valid')
        self.assertAlmostEqual(fast['infilling_score'],slow['infilling_score'],places=13)
        np.testing.assert_allclose(fast['token_scores'],slow['token_scores'],atol=1e-13,rtol=0)
        np.testing.assert_allclose(fast['raw_log_ratios'],slow['raw_log_ratios'],atol=1e-13,rtol=0)

    def test_altered_context_truncation_preserves_causal_statistics(self):
        self.engine.score(self.row,self.plan)
        for r in self.scorer.requests[1:]:
            self.assertEqual(max(r['positions']),len(r['ids'])-1)
            self.assertLessEqual(len(r['positions']),5)
            self.assertLess(len(r['ids']),len(self.plan['input_ids']))

    def test_complete_cache_skips_model_and_labels(self):
        first=self.engine.score(self.row,self.plan)
        self.cache.close();self.cache=Cache(self.out/'cache.sqlite')
        engine=Engine(self.cache,self.identity,lambda:(_ for _ in ()).throw(AssertionError('no model')))
        second=engine.score(dict(self.row,is_member=1-self.row['is_member']),self.plan)
        self.assertEqual(first['infilling_score'],second['infilling_score']);self.assertTrue(second['cache_hit'])
        self.assertEqual(second['session_forward_calls'],0)

    def test_partial_query_checkpoint_recovery(self):
        original_put=self.cache.put;commits=[0];committed=set()
        def fail(kind,key,payload):
            original_put(kind,key,payload)
            if kind=='query':
                committed.add(key);commits[0]+=1
                if commits[0]==2:raise RuntimeError('interrupted after commit')
        with patch.object(self.cache,'put',side_effect=fail):
            with self.assertRaises(RuntimeError):self.engine.score(self.row,self.plan)
        self.scorer.requests=[];result=self.engine.score(self.row,self.plan)
        self.assertFalse({self.engine.query_key(r) for r in self.scorer.requests}&committed)
        self.assertAlmostEqual(result['infilling_score'],reference(self.plan,self.scorer.distribution)['infilling_score'])

    def test_duplicate_queries_have_one_model_evaluation(self):
        r=dict(ids=[0,1,2,3],positions=[1,2,3]);values=self.engine.query([r,r,r])
        self.assertEqual(values[0],values[2]);self.assertEqual(self.scorer.forward_calls,1)
        self.engine.query([r]);self.assertEqual(self.scorer.forward_calls,1)

    def test_identity_version_and_checksum(self):
        a=self.engine.score(self.row,self.plan)
        r=dict(ids=[1,2],positions=[1]);other=Engine(self.cache,dict(self.identity,author_commit='changed'),lambda:self.scorer)
        self.assertNotEqual(self.engine.query_key(r),other.query_key(r))
        self.cache.con.execute("UPDATE cache SET sha='corrupt' WHERE kind='sample'");self.cache.con.commit()
        with self.assertRaises(ValueError):self.engine.score(self.row,self.plan)

    def test_span_contiguity_special_mask_and_short_counts(self):
        p=make_plan('300',Tokenizer(),self.identity)
        self.assertEqual(len(p['input_ids']),128);self.assertEqual(p['input_ids'],[i%4 for i in range(128)])
        self.assertEqual(p['target_positions'],list(range(1,128)))
        self.assertEqual(p['metadata']['effective_analysis_tokens'],127);self.assertEqual(p['metadata']['positions_without_future'],2)
        self.assertFalse(p['metadata']['short_span']);self.assertTrue(p['metadata']['fewer_than_128_eligible_tokens'])
        class Specials(Tokenizer):
            def __call__(inner,*args,**kwargs):
                e=super().__call__(*args,**kwargs)
                if len(e['input_ids'])>3:e['special_tokens_mask'][2]=1;e['offset_mapping'][3]=(0,0)
                return e
        s=make_plan('14',Specials(),self.identity)
        self.assertNotIn(2,s['target_positions']);self.assertNotIn(3,s['target_positions'])
        self.assertEqual(s['input_ids'][2],2)

    def test_invalid_short_samples_are_not_zero_scores_or_model_calls(self):
        for length in ['1','2','3','5']:
            p=make_plan(length,Tokenizer(),self.identity);r=self.engine.score(self.row,p)
            self.assertEqual(r['status'],'invalid');self.assertIsNone(r['infilling_score'])
        self.assertEqual(self.scorer.forward_calls,0)

    def test_partition_integrity_balanced_seed_42(self):
        self.assertEqual({k:len(v) for k,v in self.selected.items()},self.identity['smoke_counts'])
        for frame in self.selected.values():
            for _,g in frame.groupby('language'):self.assertEqual(g.is_member.sum(),len(g)//2)
        full,_=select_partitions(self.original,'full',seed=42)
        for k in full:pd.testing.assert_frame_equal(full[k],self.original[k])
        bad=copy.deepcopy(self.original);bad['shadow'].loc[0,'content_sha256']=bad['heldout'].iloc[0].content_sha256
        with self.assertRaises(ValueError):validate_partitions(bad)

    def evaluation_frames(self):
        scored=pd.concat(list(self.selected.values()),ignore_index=True);scored['infilling_score']=-1.+scored.is_member*.5
        scored['status']='valid';scored['invalid_reason']=None;scored['effective_analysis_tokens']=127
        baseline=self.selected['heldout'].copy();adaptive=baseline.copy()
        for c in ['greybox_score','lumia_score','fusion_score']:baseline[c]=baseline.is_member*.2
        adaptive['adaptive_score']=adaptive.is_member*.3
        return scored,baseline,adaptive

    def test_exact_match_and_no_heldout_threshold_selection(self):
        scored,baseline,adaptive=self.evaluation_frames()
        with patch('infilling_code_attack.evaluation.atomic_parquet'):
            first=evaluate(scored,self.selected,baseline,adaptive,self.out,'smoke')
            changed=scored.copy();changed.loc[changed.split_role.eq('heldout_test'),'infilling_score']*=-1
            second=evaluate(changed,self.selected,baseline,adaptive,self.out,'smoke')
        self.assertEqual(first['deployed_calibrated_threshold']['threshold'],second['deployed_calibrated_threshold']['threshold'])
        self.assertEqual(first['coverage']['valid_matched'],30)
        self.assertEqual(len(first['heldout_diagnostic_roc_only']),5)
        bad=adaptive.copy();bad.loc[0,'content_sha256']='changed'
        with self.assertRaises(ValueError):exact_match(self.selected['heldout'],bad,['adaptive_score'])

    def test_explicit_invalid_cohort_and_coverage(self):
        scored,baseline,adaptive=self.evaluation_frames();sid=self.selected['heldout'].iloc[0].sample_id
        scored.loc[scored.sample_id.eq(sid),['infilling_score','status','invalid_reason']]=[None,'invalid','short']
        with patch('infilling_code_attack.evaluation.atomic_parquet'):
            report=evaluate(scored,self.selected,baseline,adaptive,self.out,'smoke')
        self.assertEqual(report['coverage']['processed'],30);self.assertEqual(report['coverage']['valid_matched'],29)
        self.assertEqual(report['invalid_samples']['heldout_ids'],[sid]);self.assertEqual(report['invalid_samples']['total'],1)

    def test_all_invalid_heldout_and_calibration_report_without_fabrication(self):
        scored,baseline,adaptive=self.evaluation_frames()
        mask=scored.split_role.isin(['heldout_test','threshold_calibration'])
        scored.loc[mask,['infilling_score','status','invalid_reason']]=[None,'invalid','short']
        with patch('infilling_code_attack.evaluation.atomic_parquet'):
            report=evaluate(scored,self.selected,baseline,adaptive,self.out,'smoke')
        self.assertEqual(report['coverage']['valid_matched'],0)
        self.assertIsNone(report['deployed_calibrated_threshold']['threshold'])
        self.assertNotIn('auc_roc',report['heldout_diagnostic_roc_only']['infilling_score']['overall'])

    def test_preparation_orchestration_and_audit(self):
        contents={r['sample_id']:f'{role}:{i}' for role,frame in self.original.items() for i,r in enumerate(frame.to_dict('records'))}
        class Sources:
            calls=0
            def batches(inner,manifest,size):
                inner.calls+=1
                for start in range(0,len(manifest),size):
                    f=manifest.iloc[start:start+size].copy();f['content']=f.sample_id.map(contents);yield f
        source=Sources();monitor=SimpleNamespace(guard=lambda:None,tick=lambda *args:None)
        plans=prepare_plans(self.selected,Tokenizer(),self.identity,self.cache,source,monitor)
        self.assertEqual(plans,prepare_plans(self.selected,Tokenizer(),self.identity,self.cache,source,monitor));self.assertEqual(source.calls,1)
        with patch('infilling_code_attack.runner.atomic_parquet'):
            scored,timing=run_scores(self.selected,plans,self.identity,self.cache,lambda:self.scorer,monitor,self.out)
            again,timing2=run_scores(self.selected,plans,self.identity,self.cache,lambda:(_ for _ in ()).throw(AssertionError('cached')),monitor,self.out)
        self.assertEqual(len(scored),150);self.assertEqual(timing2['session_transformer_forward_calls'],0)
        self.assertTrue(again.cache_hit.all());self.assertEqual(audit_report(self.out/'forward_calls.jsonl')['transformer_forward_calls'],self.scorer.forward_calls)

    def test_notebook_syntax(self):
        p=Path('notebooks/06_infilling_code.ipynb')
        if not p.exists():self.skipTest('Separate notebook artifact not in source ZIP')
        nb=json.loads(p.read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));self.assertEqual(c['outputs'],[])


class TorchSynthetic(unittest.TestCase):
    def test_native_indexing_padding_moments_and_optimized_reference_equivalence(self):
        try:
            import torch
            from transformers import Starcoder2Config,Starcoder2ForCausalLM
        except ImportError:self.skipTest('Optional torch/transformers unavailable locally; no downloads')
        from infilling_code_attack.inference import TorchScorer
        threads=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,threads)
        torch.manual_seed(42)
        cfg=Starcoder2Config(vocab_size=8,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=256,sliding_window=256,
            attention_dropout=0.,embedding_dropout=0.,residual_dropout=0.);cfg._attn_implementation='eager'
        model=Starcoder2ForCausalLM(cfg).eval().requires_grad_(False);scorer=TorchScorer(model,0,3)
        ids=[0,1,2,3,4,5,6,7,0];request=dict(ids=ids,positions=list(range(1,len(ids))))
        values=scorer.predict_batch([request,dict(ids=[0,2,3,4],positions=[1,2,3])])
        with torch.inference_mode():lp=torch.log_softmax(model(torch.tensor([ids]),use_cache=False).logits.float(),-1)[0]
        for t in request['positions']:
            expected=statistics(lp[t-1].double().numpy(),ids[t])
            for field in ['log_true','log_greedy','mu','sigma']:self.assertAlmostEqual(values[0][str(t)][field],expected[field],places=5)
        solo=scorer.predict_batch([dict(ids=[0,2,3,4],positions=[1,2,3])])[0]
        for t in solo:self.assertAlmostEqual(solo[t]['log_true'],values[1][t]['log_true'],places=5)
        def predict(prefix):
            with torch.inference_mode():return torch.log_softmax(model(torch.tensor([prefix]),use_cache=False).logits[0,-1].float(),-1).double().numpy()
        with tempfile.TemporaryDirectory() as tmp:
            cache=Cache(Path(tmp)/'torch.sqlite');identity=fixed_configuration();engine=Engine(cache,identity,lambda:scorer)
            row=dict(sample_id='Python/test/1',language='Python',content_sha256='test',official_split='test',source_row=1,split_role='heldout_test')
            plan=dict(input_ids=ids,target_positions=list(range(1,len(ids))),metadata={},invalid_reason=None)
            fast=engine.score(row,plan);slow=reference(plan,predict)
            self.assertAlmostEqual(fast['infilling_score'],slow['infilling_score'],places=4);cache.close()
        self.assertFalse(any(p.requires_grad for p in model.parameters()))


if __name__=='__main__':unittest.main()

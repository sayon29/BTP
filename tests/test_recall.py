"""CPU synthetic validation only; never download a dataset or pretrained model."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from adaptive_token_attack.data import COUNTS, select_partitions, validate_partitions
from recall_code_attack.cache import Cache, fingerprint, sample_identity
from recall_code_attack.prefix import approved_pool, choose_prefix, substantive_overlap
from recall_code_attack.scoring import aligned_targets, ratio, target_positions, score_plan
from recall_code_attack.evaluation import exact_match, calibrate, evaluate


def partitions():
    languages = ['Go','Java','Python','Ruby','Rust']
    roles = dict(shadow='shadow_train',validation='fusion_fit',calibration='threshold_calibration',heldout='heldout_test')
    frames = {}; offset=0
    for role,count in COUNTS.items():
        rows=[]
        for i in range(count):
            lang=languages[(i//2)%5]; split='test' if role=='heldout' else 'train'
            rows.append(dict(sample_id=f'{lang}/{split}/{offset+i}',language=lang,source_row=offset+i,
                official_split=split,split_role=roles[role],is_member=i%2,
                membership='member' if i%2 else 'non-member',
                content_sha256=hashlib.sha256(f'{role}:{i}'.encode()).hexdigest()))
        frames[role]=pd.DataFrame(rows);offset+=count
    return frames


class RecallSynthetic(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.original=partitions()
        self.selected,self.execution=select_partitions(self.original,'smoke',seed=17)
        self.row=self.selected['heldout'].iloc[0].to_dict()
        self.cache=Cache(self.out/'cache.sqlite')
        self.plan=dict(base_ids=[1,2,3,4],prefix_ids=[5]*128,target_ids_sha256=fingerprint([2,3,4]),prefix_hash=fingerprint([5]*128))

    def tearDown(self):
        self.cache.close();self.temp.cleanup()

    def candidate(self,i):
        return dict(sample_id=f'{self.row["language"]}/train/{i}',language=self.row['language'],
                    content_sha256=str(i),official_split='train',split_role='shadow_train',is_member=0,
                    prefix_ids=list(range(200+i*200,328+i*200)),prefix_text=f'prefixUnique{i} '+' '.join(f'v{i}_{j}' for j in range(128)))

    def test_smoke_counts_and_strata(self):
        self.assertEqual({k:len(v) for k,v in self.selected.items()},dict(shadow=80,validation=20,calibration=20,heldout=30))
        for f in self.selected.values():
            self.assertEqual(len(f.language.unique()),5)
            for _,g in f.groupby('language'):
                self.assertEqual(int(g.is_member.sum()),len(g)//2)
        other,_=select_partitions({k:v.iloc[::-1].copy() for k,v in self.original.items()},'smoke',seed=17)
        for k in other:
            self.assertEqual(other[k].sample_id.tolist(),self.selected[k].sample_id.tolist())

    def test_full_exact_identity_order(self):
        full,_=select_partitions(self.original,'full')
        for k in full:
            pd.testing.assert_frame_equal(full[k],self.original[k])

    def test_incomplete_manifest_rejected(self):
        self.original['shadow']=self.original['shadow'].iloc[:-1]
        with self.assertRaises(ValueError): select_partitions(self.original,'smoke')

    def test_partition_identity_and_content_contamination(self):
        for column in ['sample_id','content_sha256']:
            bad=copy.deepcopy(self.original)
            bad['calibration'].loc[0,column]=bad['heldout'].iloc[0][column]
            with self.assertRaises(ValueError): validate_partitions(bad)

    def test_pool_only_shadow_nonmembers(self):
        pool=approved_pool(self.original)
        self.assertEqual(len(pool),800)
        self.assertTrue(pool.is_member.eq(0).all())
        self.assertFalse(set(pool.sample_id)&set(self.original['heldout'].sample_id))
        self.assertFalse(set(pool.content_sha256)&set(self.original['heldout'].content_sha256))
        bad=copy.deepcopy(self.original)
        bad['shadow'].loc[0,'official_split']='test'
        with self.assertRaises(ValueError): approved_pool(bad)

    def test_prefix_deterministic_label_independent(self):
        pool=[self.candidate(i) for i in range(5)]
        a,_=choose_prefix(self.row,'targetUnique',list(range(20)),pool)
        target=dict(self.row,is_member=1-self.row['is_member'])
        b,_=choose_prefix(target,'targetUnique',list(range(20)),pool[::-1])
        self.assertEqual(a,b)
        self.assertEqual(len(a['prefix_ids']),128)
        self.assertEqual(a['language'],self.row['language'])

    def test_prefix_self_duplicate_language_and_heldout_exclusions(self):
        self_candidate=dict(self.candidate(0),sample_id=self.row['sample_id'])
        duplicate=dict(self.candidate(1),content_sha256=self.row['content_sha256'])
        wrong_language=dict(self.candidate(2),language='not-a-language')
        valid=self.candidate(3)
        got,_=choose_prefix(self.row,'targetUnique',[1,2,3],[self_candidate,duplicate,wrong_language,valid])
        self.assertEqual(got,valid)
        with self.assertRaises(ValueError):
            choose_prefix(self.row,'targetUnique',[1,2,3],[dict(valid,official_split='test')])
        with self.assertRaises(ValueError):
            choose_prefix(self.row,'targetUnique',[1,2,3],[dict(valid,is_member=1)])

    def test_substantive_overlap(self):
        self.assertTrue(substantive_overlap(list(range(128)),list(range(90,200)),'alpha','beta'))
        text=' '.join(f'code_{i}' for i in range(30))
        self.assertTrue(substantive_overlap([1]*128,[2]*128,text,text.replace(' ','\n')))
        self.assertTrue(substantive_overlap([1]*128,[2]*128,'def tiny ( ) return 1','tiny ( ) return 1'))
        self.assertFalse(substantive_overlap([1]*128,[2]*128,'prefix_variable','unrelated_target'))
        with self.assertRaises(ValueError):
            choose_prefix(self.row,text,list(range(1000,1030)),[dict(self.candidate(0),prefix_text=text)])

    def test_identical_target_span_and_context_limit(self):
        base=list(range(300));prefix=list(range(128))
        kept,meta=aligned_targets(base,prefix,256)
        self.assertEqual(kept,base[:128])
        self.assertEqual(meta['context_truncated_tokens'],172)
        self.assertEqual(meta['target_ids_sha256'],fingerprint(base[1:128]))
        self.assertEqual((prefix+kept)[129:],kept[1:])
        self.assertLessEqual(len(prefix+kept),256)
        with self.assertRaises(ValueError): aligned_targets([1],prefix,256)

    def test_autoregressive_alignment_and_prefix_mask(self):
        # A hand-constructed next-token distribution has known target NLLs.
        ids=[0,1,2,3,1]; logits=np.zeros((4,4)); logits[np.arange(4),ids[1:]]=np.log(3.)
        def score(values,start,chunk):
            total=0.;count=0;selected=[]
            for a in range(0,len(values)-1,chunk):
                b=min(a+chunk,len(values)-1)
                first,end=target_positions(len(values),start,a,b)
                for i in range(first,end):
                    z=logits[i] if start==1 else logits[i-128]
                    total+=z[values[i+1]]-np.log(np.exp(z).sum());count+=1;selected.append(values[i+1])
            return total/count,selected
        expected=np.log(.5)
        for chunk in [1,2,256]:
            ll,targets=score(ids,1,chunk)
            conditional,other=score([3]*128+ids,129,chunk)
            self.assertAlmostEqual(ll,expected)
            self.assertAlmostEqual(conditional,expected)
            self.assertEqual(targets,ids[1:]);self.assertEqual(other,targets)
        self.assertEqual(ratio(-1.,-2.),2.)
        self.assertEqual(ratio(-2.,-1.),.5)

    def test_ratio_finite_no_clipping(self):
        for a,b in [(0.,-1.),(float('nan'),-1.),(-1.,float('inf')),(1.,-1.)]:
            with self.assertRaises(FloatingPointError): ratio(a,b)

    def test_cache_resume_no_duplicate_model_calls(self):
        calls=[]
        def ll(model,ids,start,chunk):
            calls.append((ids,start));return -2. if start==1 else -3.
        kwargs=dict(row=self.row,plan=self.plan,identity={'chunk_size':256},cache=self.cache,
                    model_factory=lambda:object(),ll_function=ll)
        first=score_plan(**kwargs)
        self.assertEqual(len(calls),2)
        self.cache.close();self.cache=Cache(self.out/'cache.sqlite');kwargs['cache']=self.cache
        second=score_plan(**kwargs)
        self.assertEqual(first,second);self.assertEqual(len(calls),2)
        self.assertEqual(calls[0][0][calls[0][1]:],calls[1][0][calls[1][1]:])
        self.assertEqual(first['recall_score'],1.5)

    def test_interrupted_pair_resumes_only_missing_branch(self):
        calls=[]
        def interrupted(model,ids,start,chunk):
            calls.append(start)
            if start!=1: raise RuntimeError('simulated interruption')
            return -2.
        args=(self.row,self.plan,{'chunk_size':256},self.cache,lambda:object())
        with self.assertRaises(RuntimeError): score_plan(*args,ll_function=interrupted)
        def restored(model,ids,start,chunk): calls.append(start);return -3.
        score_plan(*args,ll_function=restored)
        self.assertEqual(calls,[1,129,129])

    def test_cache_identity_changes_invalidate(self):
        calls=[]
        def ll(*args): calls.append(1);return -2.
        for identity in [{'chunk_size':256,'revision':'a'},{'chunk_size':256,'revision':'b'}]:
            score_plan(self.row,self.plan,identity,self.cache,lambda:object(),ll_function=ll)
        self.assertEqual(len(calls),4)
        self.assertEqual(sample_identity(self.row),sample_identity(dict(self.row,is_member=999)))
        self.assertNotEqual(sample_identity(self.row),sample_identity(dict(self.row,content_sha256='changed')))

    def test_cache_checksum_and_import(self):
        self.cache.put('x','a',{'answer':42})
        other=Cache(self.out/'other.sqlite')
        other.reuse_from(self.cache.path)
        self.assertEqual(other.get('x','a'),{'answer':42});other.close()
        self.cache.con.execute("UPDATE cache SET sha='bad'");self.cache.con.commit()
        with self.assertRaises(ValueError): self.cache.get('x','a')

    def test_exact_scores_reject_identity_drift_and_missing(self):
        f=self.selected['heldout'].copy();f['recall_score']=1.
        exact_match(self.selected['heldout'],f,['recall_score'])
        for field,value in [('source_row',-1),('content_sha256','changed'),('language','C'),('is_member',4)]:
            bad=f.copy();bad.loc[0,field]=value
            with self.assertRaises(ValueError):exact_match(self.selected['heldout'],bad,['recall_score'])
        with self.assertRaises(ValueError):exact_match(self.selected['heldout'],f.iloc[:-1],['recall_score'])
        with self.assertRaises(ValueError):exact_match(self.selected['heldout'],pd.concat([f,f.iloc[:1]]),['recall_score'])

    def test_calibration_only_ties_and_zero_fpr(self):
        f=self.selected['calibration'].copy();f['recall_score']=1.
        threshold=calibrate(f)
        self.assertGreater(threshold['threshold'],1.)
        self.assertEqual(threshold['calibration_fpr'],0.)
        with self.assertRaises(ValueError): calibrate(f.assign(split_role='heldout_test'))

    def test_evaluation_paired_full_coverage_and_no_test_threshold_leakage(self):
        selected=self.selected
        scored=pd.concat(list(selected.values()),ignore_index=True)
        scored['recall_score']=1.+scored.is_member*.1
        static=selected['heldout'].copy()
        for c in ['greybox_score','lumia_score','fusion_score']: static[c]=static.is_member*.2
        adaptive=static.copy();adaptive['adaptive_score']=adaptive.is_member*.3
        with patch('recall_code_attack.evaluation.atomic_parquet'):
            report=evaluate(scored,selected,static,adaptive,self.out,'smoke')
        self.assertEqual(report['coverage']['matched'],30)
        self.assertEqual(report['coverage']['original_heldout_fraction'],.03)
        self.assertEqual(len(report['heldout_diagnostic_roc_only']),5)
        # Change all held-out scores while keeping calibration fixed.
        changed=scored.copy();mask=changed.split_role.eq('heldout_test')
        changed.loc[mask,'recall_score']=100.-changed.loc[mask,'recall_score']
        with patch('recall_code_attack.evaluation.atomic_parquet'):
            other=evaluate(changed,selected,static,adaptive,self.out,'smoke')
        self.assertEqual(report['deployed_calibrated_threshold']['threshold'],other['deployed_calibrated_threshold']['threshold'])

    def test_preparation_and_lazy_orchestration_resume(self):
        from recall_code_attack.runner import prepare_plans, run_scores
        contents={}
        for role,frame in self.original.items():
            for i,row in enumerate(frame.to_dict('records')):
                contents[row['sample_id']]=f'{role}:{i}'
        class Tokenizer:
            def __call__(self,text,add_special_tokens=True,truncation=False,max_length=None):
                offset=int(hashlib.sha256(text.encode()).hexdigest()[:12],16)*1000
                ids=list(range(offset,offset+200))
                return {'input_ids':ids[:max_length] if truncation else ids}
            def decode(self,ids,**kwargs):
                return ' '.join(f'token_{i}' for i in ids)
        class Sources:
            calls=0
            def batches(inner,manifest,size):
                inner.calls+=1
                for start in range(0,len(manifest),size):
                    batch=manifest.iloc[start:start+size].copy()
                    batch['content']=[contents[sid] for sid in batch.sample_id]
                    yield batch
        monitor=SimpleNamespace(guard=lambda:None,tick=lambda *args:None)
        identity=dict(baseline_max_length=180,context_limit=240,chunk_size=256)
        source=Sources()
        plans,pool_hash=prepare_plans(self.original,self.selected,Tokenizer(),identity,self.cache,source,monitor)
        self.assertEqual(len(plans),150);self.assertEqual(source.calls,1)
        approved=approved_pool(self.original).set_index('sample_id')
        for sid,p in plans.items():
            self.assertNotEqual(sid,p['prefix_sample_id'])
            self.assertIn(p['prefix_sample_id'],approved.index)
            self.assertEqual(len(p['prefix_ids']),128)
            self.assertEqual(len(p['base_ids']),112)
            self.assertEqual(p['baseline_truncated_tokens'],20)
            self.assertEqual(p['context_truncated_tokens'],68)
            self.assertEqual(p['scored_target_tokens'],111)
        again,_=prepare_plans(self.original,self.selected,Tokenizer(),identity,self.cache,source,monitor)
        self.assertEqual(plans,again);self.assertEqual(source.calls,1)
        factories=[];calls=[]
        def factory(): factories.append(1);return object()
        def ll(model,ids,start,chunk): calls.append(start);return -2. if start==1 else -3.
        with patch('recall_code_attack.runner.atomic_parquet'), patch('recall_code_attack.scoring.mean_log_likelihood',side_effect=ll):
            first=run_scores(self.selected,plans,identity,self.cache,factory,monitor,self.out)
            second=run_scores(self.selected,plans,identity,self.cache,factory,monitor,self.out)
        self.assertEqual(len(factories),1);self.assertEqual(len(calls),300)
        self.assertTrue(first.recall_score.eq(1.5).all())
        pd.testing.assert_frame_equal(first,second)

    def test_notebook_compiles_and_has_no_sweep_or_execution_output(self):
        if not Path('notebooks/04_recall_code.ipynb').is_file():
            self.skipTest('Separate notebook file is not part of the source ZIP')
        nb=json.loads(Path('notebooks/04_recall_code.ipynb').read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':
                ast.parse(''.join(c['source']))
                self.assertEqual(c['outputs'],[])
        self.assertTrue(any('MODE = ' in ''.join(c['source']) for c in nb['cells']))


class OptionalTorchSynthetic(unittest.TestCase):
    def test_tiny_random_causal_model_chunk_equivalence(self):
        try:
            import torch
            from transformers import Starcoder2Config, Starcoder2ForCausalLM
        except ImportError:
            self.skipTest('Optional torch/transformers unavailable; no installation/download required for local CPU suite')
        from recall_code_attack.scoring import mean_log_likelihood
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(42)
        config=Starcoder2Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=256,sliding_window=256,
            attention_dropout=0.,embedding_dropout=0.,residual_dropout=0.)
        config._attn_implementation='eager'
        model=Starcoder2ForCausalLM(config).eval().requires_grad_(False)
        ids=[2,3,4,5,6,7,8]
        for prefix in [[],[9]*128]:
            full=prefix+ids;start=len(prefix)+1
            with torch.inference_mode():
                logits=model(torch.tensor([full]),use_cache=False).logits[0,:-1].float()
                labels=torch.tensor(full[1:])
                lp=torch.log_softmax(logits,dim=-1).gather(1,labels[:,None]).squeeze(1)
                expected=float(lp[start-1:].double().mean())
            for chunk in [1,3,256]:
                actual=mean_log_likelihood(model,full,start,chunk)
                self.assertAlmostEqual(actual,expected,places=5)


if __name__=='__main__': unittest.main()

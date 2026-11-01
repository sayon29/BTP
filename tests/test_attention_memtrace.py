"""One combined attack: mathematical checks and mock workflow; no pretrained downloads."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
import torch
from transformers import Starcoder2Config, Starcoder2ForCausalLM

from adaptive_token_attack.data import fingerprint, IDENTITY, validate_partitions, load_run, resolve_paths, select_partitions
from attention_memtrace_code_attack.features import feature_schema, attention_schema, reduce_attention_layer, reduce_attentions
from attention_memtrace_code_attack.data import fixed_configuration, verified_reduced_prefixes, make_plan, cache_key
from attention_memtrace_code_attack.inference import Extractor
from attention_memtrace_code_attack.cache import Cache
from attention_memtrace_code_attack.runner import run_features
from attention_memtrace_code_attack.evaluation import evaluate, paired_bootstrap
from attention_memtrace_code_attack.workflow import require_completed_smoke
from attention_memtrace_code_attack.bootstrap import stage_source, MARKER
from attention_memtrace_code_attack.archive import create_results_archive, create_partial_archive, REQUIRED_OUTPUTS
from reduced_memtrace_code_attack.features import feature_schema as original_schema
from reduced_memtrace_code_attack.inference import Extractor as ReducedExtractor
from reduced_memtrace_code_attack.evaluation import fit_classifier, member_probabilities, calibrate
from reduced_memtrace_code_attack.cache import Cache as OriginalCache
from test_reduced_memtrace import synthetic_partitions, Tokenizer


def tiny_model(dtype=torch.float32, layers=2):
    torch.manual_seed(42)
    cfg=Starcoder2Config(vocab_size=16,hidden_size=16,intermediate_size=24,
        num_hidden_layers=layers,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=128,bos_token_id=0,eos_token_id=1,
        attention_dropout=0.,residual_dropout=0.,embedding_dropout=0.,sliding_window=4096)
    cfg._attn_implementation='eager'
    return Starcoder2ForCausalLM(cfg).to(dtype).eval()


class AttentionMathematics(unittest.TestCase):
    def setUp(self):
        self.schema=attention_schema(1,2)
        self.p=torch.tensor([[[[1.,0,0],[.5,.5,0],[.25,.25,.5]],
                              [[1.,0,0],[.25,.75,0],[0.,0.,1.]]]])
        self.mask=torch.ones(1,3,dtype=torch.long)
    def vector(self):return reduce_attention_layer(self.p,self.mask,self.schema)[0]
    def test_row_sums_and_normalization(self):
        _,audit=reduce_attention_layer(self.p.half(),self.mask,self.schema)
        self.assertEqual(float(audit[0]),0.)
        bad=self.p*.5
        with self.assertRaisesRegex(ValueError,'row sums'):reduce_attention_layer(bad,self.mask,self.schema)
    def test_probability_validation(self):
        for value in [float('nan'),float('inf'),-1.]:
            bad=self.p.clone();bad[0,0,1,0]=value
            with self.assertRaises(FloatingPointError):reduce_attention_layer(bad,self.mask,self.schema)
    def test_future_masking(self):
        bad=self.p.clone();bad[0,0,0,1]=.05
        with self.assertRaisesRegex(ValueError,'future/padding'):reduce_attention_layer(bad,self.mask,self.schema)
    def test_entropy(self):
        mean=self.p[0].mean(dim=0).numpy()
        expected=np.mean([-sum(x*np.log2(x) for x in row if x>0) for row in mean])
        self.assertAlmostEqual(float(self.vector()[0]),expected,places=6)
    def test_head_entropy_and_focus(self):
        vector=self.vector()
        # After six layer scalars: h0 entropy/focus, then h1 entropy/focus.
        self.assertAlmostEqual(float(vector[6]),(0+1+1.5)/3,places=6)
        self.assertAlmostEqual(float(vector[7]),(1+.5+.5)/3,places=6)
        h1_entropy=(-.25*np.log2(.25)-.75*np.log2(.75))/3
        self.assertAlmostEqual(float(vector[8]),h1_entropy,places=6)
        self.assertAlmostEqual(float(vector[9]),(1+.75+1)/3,places=6)
    def test_concentration(self):
        self.assertAlmostEqual(float(self.vector()[1]),(1+.625+.75)/3,places=6)
    def test_sparsity_valid_pairs_only(self):
        # Mean valid probability = 3 queries / 6 allowed pairs = .5.
        # Values [1,.375,.625,.125,.125,.75] -> 3/6 below .5.
        self.assertAlmostEqual(float(self.vector()[2]),.5)
    def test_self_previous_bias(self):
        self.assertAlmostEqual(float(self.vector()[3]),(1+.625+.75)/3,places=6)
        self.assertAlmostEqual(float(self.vector()[4]),(.375+.125)/2,places=6)
    def test_distance(self):
        self.assertAlmostEqual(float(self.vector()[5]),(.375+2*.125+.125)/3,places=6)
    def test_padding_ignored(self):
        padded=torch.zeros(1,2,5,5);padded[:,:,:3,:3]=self.p
        # Padded query rows can contain junk; valid queries cannot attend to padded keys.
        padded[:,:,3:,:]=float('nan')
        actual,_=reduce_attention_layer(padded,torch.tensor([[1,1,1,0,0]]),self.schema)
        torch.testing.assert_close(actual,self.vector())
        padded[0,0,1,4]=.1
        with self.assertRaisesRegex(ValueError,'future/padding'):reduce_attention_layer(padded,torch.tensor([[1,1,1,0,0]]),self.schema)
    def test_zero_probabilities_fp16(self):
        x,_=reduce_attention_layer(self.p.half(),self.mask,self.schema)
        self.assertEqual(x.dtype,torch.float32);self.assertTrue(torch.isfinite(x).all())
        torch.testing.assert_close(x,self.vector())
    def test_missing_layer_or_wrong_heads(self):
        with self.assertRaisesRegex(ValueError,'Every transformer layer'):reduce_attentions(None,self.mask,self.schema)
        with self.assertRaises(ValueError):reduce_attentions((self.p,),self.mask,attention_schema(2,2))
        with self.assertRaisesRegex(ValueError,'query-head'):reduce_attention_layer(self.p[:,:1],self.mask,self.schema)
    def test_short_sequence(self):
        with self.assertRaisesRegex(ValueError,'two valid'):reduce_attention_layer(torch.ones(1,2,1,1),torch.ones(1,1),self.schema)
    def test_combined_schema_prefix(self):
        schema=feature_schema(30,24);original=original_schema(30)
        self.assertEqual(schema['names'][:167],original['names'])
        self.assertEqual(schema['families'][:167],original['families'])
        self.assertEqual(schema['reduced'],original)
        self.assertEqual(len(schema['names']),1787)
        self.assertEqual(schema['attention_features'],1620)
        self.assertEqual(len(set(schema['names'])),1787)
        self.assertEqual(schema,feature_schema(30,24))


class PinnedModelChecks(unittest.TestCase):
    def test_eager_query_heads_and_causal_rows(self):
        model=tiny_model()
        with torch.inference_mode():
            out=model(input_ids=torch.tensor([[1,2,3,4]]),attention_mask=torch.ones(1,4),
                output_attentions=True,output_hidden_states=True,use_cache=False)
        self.assertEqual(len(out.attentions),2)
        for p in out.attentions:
            self.assertEqual(tuple(p.shape),(1,4,4,4)) # four query heads, two KV heads
            torch.testing.assert_close(p.sum(-1),torch.ones(1,4,4))
            self.assertEqual(float(p.triu(1).abs().max()),0.)
        self.assertIsNone(out.past_key_values)
    def test_reduced_prefix_exact_and_single_forward(self):
        model=tiny_model(layers=30);schema=feature_schema(30,4)
        plan=dict(input_ids=[1,2,3,4],attention_mask=[1]*4,target_positions=[1,2,3])
        expected=ReducedExtractor(model,schema['reduced']).one(plan)['features']
        self.assertEqual(len(expected),167)
        calls=[]
        hook=model.register_forward_hook(lambda *args:calls.append(1))
        try:result=Extractor(model,schema).one(plan)
        finally:hook.remove()
        self.assertEqual(len(calls),1)
        np.testing.assert_array_equal(result['features'][:len(expected)],expected)
        self.assertEqual(len(result['features']),len(schema['names']))
        self.assertFalse(model.training);self.assertTrue(all(not p.requires_grad for p in model.parameters()))
    def test_preserves_verified_saved_prefix_without_reductions(self):
        schema=feature_schema(2,4);prefix=[.25]*len(schema['reduced']['names'])
        plan=dict(input_ids=[1,2,3],attention_mask=[1,1,1],target_positions=[1,2],verified_reduced_features=prefix)
        with patch('attention_memtrace_code_attack.inference.reduce_hidden',side_effect=AssertionError('Unnecessary recomputation')):
            r=Extractor(tiny_model(),schema).one(plan)
        self.assertEqual(r['features'][:len(prefix)],prefix)
        self.assertEqual(r['output_shapes']['hidden_states'],[])
    def test_fp16_model_stability(self):
        model=tiny_model(torch.float16);schema=feature_schema(2,4)
        r=Extractor(model,schema).one(dict(input_ids=[1,2,3],attention_mask=[1,1,1],target_positions=[1,2]))
        self.assertTrue(np.isfinite(r['features']).all())
        self.assertLess(r['attention_audit']['max_raw_row_sum_error'],.002)
    def test_supported_eager_matches_sdpa_tiny_fp32(self):
        # Same model and weights; supported backend selection, no attention monkey patch.
        model=tiny_model();ids=torch.tensor([[1,2,3,4]])
        with torch.inference_mode():
            eager=model(input_ids=ids,output_attentions=True,use_cache=False).logits
            model.config._attn_implementation='sdpa'
            sdpa=model(input_ids=ids,output_attentions=False,use_cache=False).logits
        torch.testing.assert_close(eager,sdpa,atol=1e-6,rtol=1e-5)
        with self.assertRaisesRegex(ValueError,'eager'):Extractor(model,feature_schema(2,4))


class Workflow(unittest.TestCase):
    def test_frozen_settings(self):
        cfg=fixed_configuration()
        self.assertEqual(cfg['analysis_tokens'],128);self.assertEqual(cfg['seed'],42)
        self.assertEqual(cfg['classifier']['n_estimators'],300)
        self.assertEqual(cfg['expected_query_heads'],24)
    def test_cache_reopen_import_and_schema_rejection(self):
        identity=dict(schema=feature_schema(2,2),attention_backend='eager',model_revision='test')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=Cache(root/'a.sqlite',identity)
            a.put('features','key',dict(features=[1.,2.]));a.close()
            b=Cache(root/'b.sqlite',identity);b.reuse_from(root/'a.sqlite')
            self.assertEqual(b.get('features','key')['features'],[1.,2.]);b.close()
            changed=dict(identity,attention_backend='sdpa')
            with self.assertRaisesRegex(ValueError,'incompatible'):Cache(root/'a.sqlite',changed)
            c=Cache(root/'c.sqlite',changed)
            with self.assertRaisesRegex(ValueError,'Cannot import'):c.reuse_from(root/'a.sqlite')
            old=OriginalCache(root/'reduced.sqlite');old.put('features','old',{'features':[1.]});old.close()
            with self.assertRaisesRegex(ValueError,'Reduced cache'):
                Cache(root/'reduced.sqlite',identity)
            with self.assertRaisesRegex(ValueError,'Cannot import'):c.reuse_from(root/'reduced.sqlite')
            c.close()
    def test_labels_split_and_token_planning_preserved(self):
        identity=dict(analysis_tokens=128,baseline_max_length=8192,model_config=dict(max_position_embeddings=8192))
        p=make_plan('01234567'*20,Tokenizer(),identity)
        self.assertEqual(p['metadata']['effective_target_tokens'],127)
        selected=synthetic_partitions();validate_partitions(selected,dict(shadow=80,validation=20,calibration=20,heldout=30))
        row=selected['shadow'].iloc[0].to_dict()
        self.assertNotEqual(cache_key(row,identity),cache_key(dict(row,is_member=1-row['is_member']),identity))
    def test_rf_and_calibration_reused(self):
        x=np.random.default_rng(42).normal(size=(30,5));y=np.arange(30)%2
        a=fit_classifier(x,y,fixed_configuration());b=fit_classifier(x,y,fixed_configuration())
        np.testing.assert_array_equal(member_probabilities(a,x),member_probabilities(b,x))
        f=synthetic_partitions()['calibration'].copy();f['memtrace_score']=f.is_member*.7+.1
        self.assertLessEqual(calibrate(f)['calibration_fpr'],.01)
        f['split_role']='heldout_test'
        with self.assertRaises(ValueError):calibrate(f)
    def test_missing_reduced_provenance_never_reused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'config.json').write_text('')
            with self.assertWarns(UserWarning):
                prefixes,report=verified_reduced_prefixes(Path(d)/'heldout_memtrace_scores.parquet',synthetic_partitions(),dict(schema=feature_schema(30,24)))
            self.assertFalse(prefixes);self.assertEqual(report['status'],'unverified_or_missing_provenance')
    def test_provenance_validated_prefix_reuse(self):
        identity=dict(fixed_configuration(),schema=feature_schema(2,4),sources={
            'reduced_memtrace_code_attack/'+f:'hash' for f in ['features.py','data.py','inference.py']},
            tokenizer_revision='rev',tokenizer_sha256='hash',tokenizer_special_tokens={},baseline_max_length=8192,
            versions={'transformers':'4.52.4'},model_config={'model_type':'starcoder2'},reduced_attention_backend='eager')
        selected=synthetic_partitions();rows=pd.concat(list(selected.values()),ignore_index=True)
        frame=rows[IDENTITY].copy()
        values=pd.DataFrame(np.full((len(frame),len(identity['schema']['reduced']['names'])),.25),columns=identity['schema']['reduced']['names'])
        frame=pd.concat([frame,values],axis=1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frame.to_parquet(root/'features.parquet',index=False)
            old=dict(identity,schema=identity['schema']['reduced'],numerical_implementation=identity['reduced_numerical_implementation'])
            (root/'config.json').write_text(json.dumps(dict(scoring=old)))
            (root/'feature_schema.json').write_text(json.dumps(old['schema']))
            (root/'run_manifest.json').write_text(json.dumps(dict(status='complete',artifacts={'features.parquet':hashlib.sha256((root/'features.parquet').read_bytes()).hexdigest()})))
            prefixes,report=verified_reduced_prefixes(root/'heldout_memtrace_scores.parquet',selected,identity)
            self.assertEqual(len(prefixes),150);self.assertEqual(report['status'],'verified_reuse')
    def test_directory_zip_staging_and_hash_safety(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';(source/MARKER).parent.mkdir(parents=True)
            (source/'a.py').write_text('x=1\n')
            raw=json.dumps({'a.py':hashlib.sha256((source/'a.py').read_bytes()).hexdigest()}).encode()
            (source/MARKER).write_bytes(raw);pin=hashlib.sha256(raw).hexdigest()
            stage_source(source,root/'one',pin,source)
            z=source/'attention_memtrace_kaggle_source.zip'
            with zipfile.ZipFile(z,'w') as a:
                a.write(source/'a.py','a.py');a.write(source/MARKER,MARKER)
            # Autodiscovery prefers extracted layout when ZIP is present beside it.
            stage_source(source,root/'two',pin)
            stage_source(source,root/'three',pin,upload_zip=z)
            with zipfile.ZipFile(z,'a') as a:a.writestr('../unsafe','x')
            with self.assertRaises(ValueError):stage_source(source,root/'unsafe',pin,upload_zip=z)
    def test_archives_exclude_model_caches(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir()
            for name in REQUIRED_OUTPUTS:
                if name!='feature_cache.sqlite':(out/name).write_text('{}')
            cache=Cache(out/'feature_cache.sqlite',dict(schema=feature_schema(2,2)));cache.close()
            (out/'hf_cache').mkdir();(out/'hf_cache/weights.safetensors').write_bytes(b'not weights')
            path=create_results_archive(out,root,'test',0)
            with zipfile.ZipFile(path) as a:self.assertFalse(any('hf_cache' in n for n in a.namelist()))
            self.assertTrue(create_partial_archive(out,root,'test',0).exists())
    def test_paired_bootstrap_deterministic(self):
        frame=synthetic_partitions()['heldout'].copy()
        frame['a']=np.linspace(0,1,len(frame));frame['b']=frame.a.to_numpy()[::-1]
        settings=dict(seed=42,replicates=20,strata=['language','is_member'])
        self.assertEqual(paired_bootstrap(frame,'a',['b'],settings),paired_bootstrap(frame,'a',['b'],settings))
    def test_notebook_config_syntax_and_single_fit(self):
        path=Path(__file__).resolve().parents[1]/'notebooks/09_attention_memtrace_code.ipynb'
        if not path.exists():self.skipTest('Notebook provided separately')
        nb=json.loads(path.read_text());source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        self.assertIn('MODE = "smoke"',source);self.assertIn('REDUCED_MEMTRACE_SCORES',source)
        self.assertEqual(source.count('report = evaluate('),1)
        self.assertIn("eager_cfg['resources']['attention_backend'] = 'eager'",source)
        evaluation=Path(__file__).resolve().parents[1]/'attention_memtrace_code_attack/evaluation.py'
        self.assertEqual(evaluation.read_text().count('model = fit_classifier('),1)
    def test_smoke_gate_and_resume_proof(self):
        identity=dict(fixed_configuration(),schema=feature_schema(2,2))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'full';out.mkdir();smoke=root/'smoke';smoke.mkdir()
            with self.assertRaisesRegex(ValueError,'smoke experiment first'):
                require_completed_smoke(identity,'',out)
            artifacts={
                'config.json':dict(mode='smoke',scoring=identity),
                'metrics.json':dict(status='complete',coverage=dict(valid_matched=30),feature_counts=dict(combined=len(identity['schema']['names']))),
                'run_manifest.json':dict(status='complete',selected_counts=identity['smoke_counts'])}
            for name,payload in artifacts.items():(smoke/name).write_text(json.dumps(payload))
            proof=require_completed_smoke(identity,smoke/'feature_cache.sqlite',out)
            self.assertTrue(proof['smoke_complete'])
            # Same-mode full resume carries a sealed proof and needs no smoke input mount.
            self.assertEqual(proof,require_completed_smoke(identity,'',out))
            with self.assertRaisesRegex(ValueError,'different combined'):
                require_completed_smoke(dict(identity,seed=43),'',out)
    def test_complete_mock_workflow_one_rf_and_cache_resume(self):
        selected=synthetic_partitions();identity=dict(fixed_configuration(),schema=feature_schema(2,2))
        rows=pd.concat(list(selected.values()),ignore_index=True)
        plans={sid:dict(input_ids=[0,1,2,3],attention_mask=[1]*4,target_positions=[1,2,3],
            metadata=dict(source_token_count=4,analysis_input_tokens=4,valid_token_count=4,
                effective_target_tokens=3,truncated_tokens=0,truncated=False)) for sid in rows.sample_id}
        schema=identity['schema']
        class Monitor:
            def guard(self):pass
            def tick(self,*args):pass
            def snapshot(self):return dict(rss_gib=.1,disk_free_gib=10.)
        class FakeExtractor:
            calls=0
            def one(self,plan):
                self.calls+=1
                return dict(features=[.25]*len(schema['names']),seconds=.01,forward_seconds=.008,
                    attention_aggregation_seconds=.001,gpu_peak_allocated_gib=1.,gpu_peak_reserved_gib=2.)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cache=Cache(out/'feature_cache.sqlite',identity);extractor=FakeExtractor()
            features,timing=run_features(selected,plans,identity,cache,lambda:extractor,Monitor(),out)
            self.assertEqual(len(features),150)
            self.assertEqual(extractor.calls,151) # 150 samples + one synthetic max-span stress
            probe=json.loads((out/'resource_probe.json').read_text())
            self.assertEqual(probe['probe_fresh_forwards'],5)
            self.assertEqual(probe['combined_feature_count'],len(schema['names']))
            def forbidden():raise AssertionError('Cache-only resume must not load model')
            again,_=run_features(selected,plans,identity,cache,forbidden,Monitor(),out)
            self.assertEqual(len(again),150)
            held=selected['heldout'][IDENTITY].copy()
            for column in ['lumia_score','greybox_score','fusion_score']:held[column]=np.linspace(0,1,len(held))
            optional=selected['heldout'][IDENTITY].copy();optional['reduced_memtrace_score']=.5
            with patch('attention_memtrace_code_attack.evaluation.fit_classifier',wraps=fit_classifier) as fit, \
                 patch('attention_memtrace_code_attack.evaluation.paired_bootstrap',return_value={}):
                report=evaluate(features,selected,held,{'reduced_memtrace_score':optional},identity,out,'smoke',timing)
                self.assertEqual(fit.call_count,1)
            self.assertEqual(report['coverage']['valid_matched'],30)
            self.assertEqual(report['feature_counts']['combined'],len(schema['names']))
            self.assertIn('reduced_memtrace_score',report['paired_differences'])
            self.assertTrue((out/'heldout_attention_memtrace_scores.parquet').is_file())
            self.assertLessEqual(report['deployed_calibrated_threshold']['calibration_fpr'],.01)
            cache.close()
    def test_memory_gate_stops_without_dropping_features(self):
        selected=synthetic_partitions();identity=dict(fixed_configuration(),schema=feature_schema(2,2))
        rows=pd.concat(list(selected.values()),ignore_index=True)
        plans={sid:dict(input_ids=[0,1],attention_mask=[1,1],target_positions=[1],metadata=dict(source_token_count=2)) for sid in rows.sample_id}
        class Monitor:
            def guard(self):pass
            def tick(self,*args):pass
            def snapshot(self):return {}
        class Unsafe:
            def one(self,plan):
                return dict(features=[.25]*len(identity['schema']['names']),gpu_peak_allocated_gib=12.,gpu_peak_reserved_gib=12.)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cache=Cache(out/'feature_cache.sqlite',identity)
            with self.assertRaisesRegex(RuntimeError,'11 GiB'):
                run_features(selected,plans,identity,cache,Unsafe,Monitor(),out)
            self.assertTrue((out/'memory_stop.json').is_file())
            self.assertEqual(len(pd.read_parquet(out/'features.parquet')),0)
            cache.close()
    def test_saved_reduced_auc_and_identities(self):
        root=Path(__file__).resolve().parents[1]
        score=root/'Results/Memtrace/heldout_memtrace_scores.parquet'
        if not score.exists():self.skipTest('Saved experimental results not bundled')
        from sklearn.metrics import roc_auc_score
        from reduced_memtrace_code_attack.evaluation import exact_match
        original,_,_=load_run(resolve_paths(root/'Results/Sersem_Baseline'))
        saved=pd.read_parquet(score);exact_match(original['heldout'],saved,['memtrace_score'])
        self.assertAlmostEqual(roc_auc_score(saved.is_member,saved.memtrace_score),.7033,places=6)


if __name__=='__main__':unittest.main()

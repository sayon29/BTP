"""Synthetic mathematical and workflow checks; no pretrained weights/downloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
import torch

from adaptive_token_attack.data import fingerprint, IDENTITY, load_run, resolve_paths, select_partitions, validate_partitions
from reduced_memtrace_code_attack.features import (selected_boundaries, feature_schema,
    reduce_hidden, confidence_features, combine_features)
from reduced_memtrace_code_attack.data import make_plan, cache_key, fixed_configuration
from reduced_memtrace_code_attack.inference import Extractor
from reduced_memtrace_code_attack.cache import Cache
from reduced_memtrace_code_attack.evaluation import calibrate, fit_classifier, member_probabilities, evaluate, exact_match
from reduced_memtrace_code_attack.runner import run_features, probe_rows
from reduced_memtrace_code_attack.bootstrap import stage_source, MARKER
from reduced_memtrace_code_attack.archive import create_results_archive, create_partial_archive, REQUIRED_OUTPUTS


class Tokenizer:
    model_max_length = 8192
    def __call__(self, text, **kw):
        ids = [int(c) % 8 for c in text]
        if kw.get('truncation'):
            ids = ids[:kw['max_length']]
        return dict(input_ids=ids, attention_mask=[1]*len(ids), special_tokens_mask=[0]*len(ids),
                    offset_mapping=[(i,i+1) for i in range(len(ids))])


class MockModel(torch.nn.Module):
    device = torch.device('cpu')
    def forward(self, input_ids, attention_mask, **kwargs):
        self.last_kwargs = kwargs
        h = input_ids.float()[..., None].repeat(1,1,2)
        logits = torch.zeros((*input_ids.shape, 8))
        # Predict the next integer modulo eight with high confidence.
        logits.scatter_(2, ((input_ids+1)%8)[...,None], 4.)
        return types.SimpleNamespace(hidden_states=(h,h+1,h+2), logits=logits, attentions=None)


def synthetic_partitions():
    result = {}
    counts = dict(shadow=80,validation=20,calibration=20,heldout=30)
    roles = dict(shadow='shadow',validation='fusion_fit',calibration='threshold_calibration',heldout='heldout_test')
    index = 0
    for role, count in counts.items():
        rows=[]
        for i in range(count):
            language=['Go','Java','Python','Ruby','Rust'][(i//2)%5]
            split='test' if role=='heldout' else 'train'
            rows.append(dict(sample_id=f'{language}/{split}/{index}', language=language, is_member=i%2,
                content_sha256=hashlib.sha256(str(index).encode()).hexdigest(),official_split=split,
                source_row=index,split_role=roles[role],membership='member' if i%2 else 'non-member'))
            index+=1
        result[role]=pd.DataFrame(rows)
    return result


class Mathematics(unittest.TestCase):
    def setUp(self):
        self.schema=feature_schema(2)
        self.a=torch.tensor([[[0.,0.],[3.,4.],[99.,99.]]],dtype=torch.float16)
        self.b=torch.tensor([[[0.,0.],[6.,8.],[99.,99.]]],dtype=torch.float16)
        self.mask=torch.tensor([[1,1,0]])
    def test_layer_indexing(self):
        self.assertEqual(selected_boundaries(30),[0,8,15,23,30])
        self.assertEqual(len(feature_schema(30)['names']),167)
        with self.assertRaises(ValueError):reduce_hidden((self.a,),self.mask,self.schema)
    def test_shape(self):
        with self.assertRaises(ValueError):reduce_hidden((self.a,self.b,self.a[:,:,:1]),self.mask,self.schema)
    def test_euclidean(self):
        h,_=reduce_hidden((self.a,self.b,self.b),self.mask,self.schema)
        self.assertEqual(float(h[0]),2.5);self.assertEqual(float(h[1]),2.5)
    def test_cosine_and_zero_norm(self):
        h,_=reduce_hidden((self.a,self.b,self.b),self.mask,self.schema)
        self.assertAlmostEqual(float(h[2]),.5);self.assertAlmostEqual(float(h[3]),.5)
    def test_fp32(self):
        h,c=reduce_hidden((self.a,self.b,self.b),self.mask,self.schema)
        self.assertTrue(all(v.dtype==torch.float32 for v in h+c))
        large=torch.full((1,2,2),60000.,dtype=torch.float16)
        h,_=reduce_hidden((large,-large,-large),torch.ones(1,2),self.schema)
        self.assertTrue(torch.isfinite(h[0]));self.assertGreater(float(h[0]),65504)
    def test_activations(self):
        h,_=reduce_hidden((self.a,self.b,self.b),self.mask,self.schema)
        np.testing.assert_allclose([float(v) for v in h[8:13]], [1.75,2.5,np.std([0,0,3,4]),4,.5],rtol=1e-6)
    def test_context(self):
        _,c=reduce_hidden((self.a,self.b,self.b),self.mask,self.schema)
        np.testing.assert_allclose([float(v) for v in c[:3]],[2.5,0,2.5])
    def test_entropy_and_margin(self):
        z=torch.tensor([[0.,0.],[np.log(3),0.]],dtype=torch.float32)
        v=confidence_features(z,torch.tensor([0,0]))
        entropy=(-.75*np.log(.75)-.25*np.log(.25)+np.log(2))/2
        self.assertAlmostEqual(float(v[2]),.625)
        self.assertAlmostEqual(float(v[4]),.25)
        self.assertAlmostEqual(float(v[5]),entropy,places=6)
    def test_entropy_shift_invariance(self):
        z=torch.tensor([[2.,4.,-3.]])
        a=confidence_features(z,torch.tensor([2]));b=confidence_features(z+100,torch.tensor([2]))
        np.testing.assert_allclose([float(v) for v in a],[float(v) for v in b],atol=1e-4)
    def test_causal_alignment_and_flags(self):
        model=MockModel();plan=make_plan('012345',Tokenizer(),dict(analysis_tokens=128,baseline_max_length=8192,model_config=dict(max_position_embeddings=8192)))
        r=Extractor(model,self.schema).one(plan)
        idx=self.schema['names'].index('confidence.nll_mean')
        self.assertAlmostEqual(r['features'][idx],np.log(1+7*np.exp(-4)),places=5)
        self.assertFalse(model.last_kwargs['use_cache']);self.assertFalse(model.last_kwargs['output_attentions'])
        self.assertFalse(model.training)
    def test_tiny_starcoder2_representation_contract(self):
        # Randomly initialized tiny architecture, never from_pretrained/downloaded.
        from transformers import Starcoder2Config, Starcoder2ForCausalLM
        cfg=Starcoder2Config(vocab_size=16,hidden_size=8,intermediate_size=16,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
            max_position_embeddings=128,bos_token_id=0,eos_token_id=1,attention_dropout=0.)
        model=Starcoder2ForCausalLM(cfg).eval()
        ids=torch.tensor([[1,2,3,4]])
        captured={}
        def capture_norm(module,args,output):captured['final']=output.detach().clone()
        hook=model.model.norm.register_forward_hook(capture_norm)
        try:
            with torch.inference_mode():
                output=model(input_ids=ids,output_hidden_states=True,output_attentions=False,use_cache=False)
                self.assertEqual(len(output.hidden_states),3)
                self.assertEqual(output.hidden_states[0].shape,(1,4,8))
                torch.testing.assert_close(output.hidden_states[0],model.model.embed_tokens(ids))
                torch.testing.assert_close(output.hidden_states[-1],captured['final'])
                self.assertIsNone(output.attentions)
                h,c=reduce_hidden(output.hidden_states,torch.ones_like(ids),self.schema)
                confidence=confidence_features(output.logits[0,:-1],ids[0,1:])
                self.assertEqual(len(combine_features(h,confidence,c,self.schema)),len(self.schema['names']))
        finally:hook.remove()
    def test_feature_order(self):
        self.assertEqual(feature_schema(30),feature_schema(30))
        self.assertEqual(len(set(self.schema['names'])),len(self.schema['names']))
        self.assertTrue(self.schema['names'][-1].startswith('context.'))
    def test_nonfinite_rejected(self):
        with self.assertRaises(FloatingPointError):combine_features([torch.tensor(float('nan'))],[],[],dict(names=['a']))
    def test_padding_ignored(self):
        h,c=reduce_hidden((self.a,self.b,self.b),self.mask,self.schema)
        a=self.a[:,:2];b=self.b[:,:2]
        hh,cc=reduce_hidden((a,b,b),torch.ones(1,2),self.schema)
        np.testing.assert_array_equal([float(x) for x in h+c],[float(x) for x in hh+cc])
    def test_truncation(self):
        identity=dict(analysis_tokens=128,baseline_max_length=8192,model_config=dict(max_position_embeddings=8192))
        plan=make_plan('1'*200,Tokenizer(),identity)
        self.assertEqual(len(plan['input_ids']),128);self.assertEqual(plan['metadata']['truncated_tokens'],72)
        self.assertEqual(len(plan['target_positions']),127)
        with self.assertRaises(ValueError):make_plan('1',Tokenizer(),identity)


class Workflow(unittest.TestCase):
    def test_frozen_config(self):
        self.assertEqual(fixed_configuration()['classifier']['n_estimators'],300)
    def test_partitions_and_overlap(self):
        p=synthetic_partitions();validate_partitions(p,dict(shadow=80,validation=20,calibration=20,heldout=30))
        p['validation'].loc[0,'content_sha256']=p['shadow'].iloc[0].content_sha256
        with self.assertRaisesRegex(ValueError,'overlap'):validate_partitions(p,dict(shadow=80,validation=20,calibration=20,heldout=30))
    def test_saved_baseline_loader(self):
        path=Path(__file__).resolve().parents[1]/'Results/Sersem_Baseline'
        if not (path/'shadow_manifest.parquet').exists():self.skipTest('Local baseline not bundled in source ZIP')
        original,_,cfg=load_run(resolve_paths(path))
        a,_=select_partitions(original,'smoke',seed=42);b,_=select_partitions(original,'smoke',seed=42)
        self.assertEqual(sum(map(len,a.values())),150)
        for role in a:self.assertEqual(a[role].sample_id.tolist(),b[role].sample_id.tolist())
        self.assertEqual(cfg['model_revision'],fixed_configuration()['model_revision'])
    def test_rf_reproducibility(self):
        x=np.random.default_rng(42).normal(size=(30,4));y=np.arange(30)%2
        a=fit_classifier(x,y,fixed_configuration());b=fit_classifier(x,y,fixed_configuration())
        np.testing.assert_array_equal(member_probabilities(a,x),member_probabilities(b,x))
    def test_calibration(self):
        frame=pd.DataFrame(dict(is_member=[0]*100+[1]*100,memtrace_score=np.arange(200)/200,split_role='threshold_calibration'))
        # Fingerprint needs all identity fields, independent of calibration math.
        for key in IDENTITY:
            if key not in frame:frame[key]=np.arange(200).astype(str)
        result=calibrate(frame)
        self.assertLessEqual(result['calibration_fpr'],.01);self.assertEqual(result['calibration_tpr'],1)
        frame['split_role']='heldout_test'
        with self.assertRaises(ValueError):calibrate(frame)
    def test_cache_restart_and_identity(self):
        row=synthetic_partitions()['shadow'].iloc[0].to_dict()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.sqlite';c=Cache(path);key=cache_key(row,{'model_revision':'a'})
            c.put('features',key,dict(features=[1.,2.]));c.close()
            c=Cache(Path(d)/'b.sqlite');c.reuse_from(path)
            self.assertEqual(c.get('features',key)['features'],[1.,2.])
            self.assertIsNone(c.get('features',cache_key(row,{'model_revision':'b'})))
            changed=dict(row,is_member=1-row['is_member']);self.assertNotEqual(cache_key(changed,{}),cache_key(row,{}))
            c.close()
    def test_source_zip_and_extracted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input';source.mkdir();(source/'a.py').write_text('x=1\n')
            (source/MARKER).parent.mkdir();raw=json.dumps({'a.py':hashlib.sha256((source/'a.py').read_bytes()).hexdigest()}).encode()
            (source/MARKER).write_bytes(raw);pin=hashlib.sha256(raw).hexdigest()
            stage_source(source,root/'extracted',pin,source)
            z=root/'package.zip'
            with zipfile.ZipFile(z,'w') as a:
                a.write(source/'a.py','a.py');a.write(source/MARKER,MARKER)
            stage_source(source,root/'unzipped',pin,upload_zip=z)
            with zipfile.ZipFile(z,'a') as a:a.writestr('../unsafe','bad')
            with self.assertRaises(ValueError):stage_source(source,root/'unsafe',pin,upload_zip=z)
    def test_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir()
            for name in REQUIRED_OUTPUTS:
                if name!='feature_cache.sqlite':(out/name).write_text('{}')
            c=Cache(out/'feature_cache.sqlite');c.close()
            (out/'hf_cache').mkdir();(out/'hf_cache/model.safetensors').write_bytes(b'not a real model')
            target=create_results_archive(out,root,'test',0)
            with zipfile.ZipFile(target) as a:self.assertFalse(any('hf_cache' in n for n in a.namelist()))
            self.assertTrue(create_partial_archive(out,root,'partial',0).is_file())
    def test_monitor(self):
        from shared.resource_pipeline import Monitor,ResourcePause
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(resources=dict(disk_reserve_gib=0,ram_reserve_gib=0,session_budget_minutes=0,log_interval_seconds=0))
            m=Monitor(cfg,d);m.guard();self.assertIn('rss_gib',m.snapshot());m.tick('test',1,2)
            cfg['resources']['disk_reserve_gib']=1e12
            with self.assertRaises(ResourcePause):m.guard()
    def test_end_to_end_and_cache_only_resume(self):
        selected=synthetic_partitions();identity=dict(fixed_configuration(),schema=feature_schema(2))
        rows=pd.concat(list(selected.values()),ignore_index=True)
        plans={r['sample_id']:dict(input_ids=[0,1,2,3],attention_mask=[1]*4,target_positions=[1,2,3],
            metadata=dict(source_token_count=4,analysis_input_tokens=4,valid_token_count=4,effective_target_tokens=3,truncated_tokens=0,truncated=False)) for r in rows.to_dict('records')}
        class Monitor:
            def guard(self):pass
            def tick(self,*a):pass
            def snapshot(self):return dict(rss_gib=.1,disk_free_gib=1.,gpu_peak_allocated_gib=0.)
        self.assertEqual(len(probe_rows(selected['shadow'],plans)),5)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cache=Cache(out/'feature_cache.sqlite')
            features,timing=run_features(selected,plans,identity,cache,lambda:Extractor(MockModel(),identity['schema']),Monitor(),out)
            def no_model():raise AssertionError('Cache-only resume must not load model')
            again,_=run_features(selected,plans,identity,cache,no_model,Monitor(),out)
            self.assertEqual(len(features),150);self.assertEqual(len(again),150)
            held=selected['heldout'][IDENTITY].copy()
            for c in ['lumia_score','greybox_score','fusion_score']:held[c]=np.linspace(0,1,len(held))
            report=evaluate(features,selected,held,{},identity,out,'smoke',timing)
            self.assertEqual(report['coverage']['valid_matched'],30)
            self.assertTrue((out/'classifier.joblib').is_file())
            self.assertLessEqual(report['deployed_calibrated_threshold']['calibration_fpr'],.01)
            cache.close()
    def test_memory_guard_preserves_partial_features(self):
        selected=synthetic_partitions();identity=dict(fixed_configuration(),schema=feature_schema(2))
        rows=pd.concat(list(selected.values()),ignore_index=True)
        plans={sid:dict(input_ids=[0,1],attention_mask=[1,1],target_positions=[1],
            metadata=dict(source_token_count=2)) for sid in rows.sample_id}
        class Monitor:
            def guard(self):pass
            def tick(self,*args):pass
            def snapshot(self):return {}
        class UnsafeExtractor:
            def one(self,plan):
                return dict(features=[0.]*len(identity['schema']['names']),gpu_peak_allocated_gib=12.,gpu_peak_reserved_gib=12.)
        with tempfile.TemporaryDirectory() as d:
            cache=Cache(Path(d)/'feature_cache.sqlite')
            with self.assertRaisesRegex(RuntimeError,'GPU peak'):
                run_features(selected,plans,identity,cache,UnsafeExtractor,Monitor(),d)
            self.assertTrue((Path(d)/'memory_stop.json').exists())
            self.assertEqual(len(pd.read_parquet(Path(d)/'features.parquet')),0)
            cache.close()
    def test_exact_identity_pairing(self):
        f=synthetic_partitions()['heldout'];scores=f.copy();scores['score']=.5
        scores.loc[0,'content_sha256']='bad'
        with self.assertRaises(ValueError):exact_match(f,scores,['score'])
    def test_notebook_syntax_and_settings(self):
        import ast
        path=Path(__file__).resolve().parents[1]/'notebooks/08_reduced_memtrace_code.ipynb'
        if not path.exists():self.skipTest('Notebook supplied separately on Kaggle')
        nb=json.loads(path.read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        self.assertIn('MODE = "smoke"',''.join(nb['cells'][1]['source']))


if __name__=='__main__':unittest.main()

"""Synthetic CPU tests only: no torch import, models, downloads or inference."""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from adaptive_token_attack.data import (FILES, COUNTS, resolve_paths, load_run, validate_partitions,
    match_scores, seal, check_seal, SMOKE_SIZES, IDENTITY, fingerprint, select_partitions, validate_execution)
from adaptive_token_attack.features import (STATIC, CATEGORIES, Categorizer, FeatureStore, align_tokens,
    static_score, validate_record, extract_manifest)
from adaptive_token_attack.model import (DEFAULTS, make_pool, parameters, aggregate, bce_with_logits,
    loss_gradient, validation_loss, fit, freeze, score_pool)
from adaptive_token_attack.evaluation import verify_static, calibrate, evaluate


def frames(counts=None):
    counts = counts or {'shadow':40, 'validation':20, 'calibration':20, 'heldout':20}
    roles = {'shadow':'shadow_train','validation':'fusion_fit','calibration':'threshold_calibration','heldout':'heldout_test'}
    result = {}; offset=0
    for role,count in counts.items():
        rows=[]
        for i in range(count):
            language = 'Python' if i%4 < 2 else 'Go'
            split = 'test' if role=='heldout' else 'train'
            text = f'{role}:{i}'
            rows.append(dict(sample_id=f'{language}/{split}/{offset+i}', source_row=offset+i,
                             language=language, official_split=split, split_role=roles[role],
                             is_member=i%2, membership='member' if i%2 else 'non-member',
                             content_sha256=hashlib.sha256(text.encode()).hexdigest(), content=text))
        result[role]=pd.DataFrame(rows);offset+=count
    return result


def token_record(row):
    label=row['is_member']
    # Identifier signal is useful, comment signal is misleading. Other categories absent.
    signal=np.array([0., .85 if label else .15, .2 if label else .8,
                     .8 if label else .2, .25 if label else .75],dtype=np.float32)
    category=np.array([0,1,6,1,6],dtype=np.uint8)
    weights=STATIC[category].copy();weights[0]=0
    z=np.zeros(5,dtype=np.float32);z[1:]=np.log(signal[1:]/(1-signal[1:]))
    arrays=dict(input_ids=np.arange(5,dtype=np.int32),offsets=np.array([[0,0],[0,1],[1,2],[2,3],[3,4]],dtype=np.int32),
                attention_mask=np.ones(5,dtype=np.uint8),special_tokens_mask=np.array([1,0,0,0,0],dtype=np.uint8),
                eligible=np.array([0,1,1,1,1],dtype=np.uint8),category=category,signals=signal,z=z,static_weights=weights,
                static_score_gpu=np.asarray(0.,dtype=np.float32),ast_has_error=np.asarray(False),
                n_tokens=np.asarray(5,dtype=np.int32),n_targets=np.asarray(4,dtype=np.int32))
    arrays['static_score_gpu']=np.asarray(static_score(arrays),dtype=np.float32)
    return arrays


class SyntheticExperiment(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
        self.parts=frames();self.store=FeatureStore(self.out/'features.sqlite',{'test':'synthetic'})
        for frame in self.parts.values():
            for row in frame.to_dict('records'):self.store.put(row,token_record(row))
        self.pools={role:make_pool(frame,role,self.store) for role,frame in self.parts.items()}
        self.scores=self.parts['heldout'].drop(columns='content').copy()
        self.scores['greybox_score']=[static_score(token_record(r)) for r in self.scores.to_dict('records')]
        self.scores['lumia_score']=self.scores.is_member*.6+.2
        self.scores['fusion_score']=.9*self.scores.lumia_score+.1*self.scores.greybox_score
        self.gate=verify_static(self.pools,self.scores,self.store,'float32',self.out)
        self.cfg=dict(max_epochs=180,patience=200,min_delta=0.,seed=17)

    def tearDown(self):
        self.store.close();self.tmp.cleanup()

    def train(self):
        model,complete=fit(self.pools['shadow'],self.pools['validation'],self.cfg,self.out,self.gate)
        self.assertTrue(complete)
        return freeze(model,self.pools['validation'],self.out)

    def test_static_reconstruction_and_aggregation(self):
        theta=np.r_[np.log(STATIC.astype(float)),0.,0.]
        scores=aggregate(theta,self.pools['shadow'].counts,self.pools['shadow'].sums)
        expected=[static_score(token_record(row)) for row in self.parts['shadow'].to_dict('records')]
        np.testing.assert_allclose(scores,expected,rtol=0,atol=1e-15)
        self.assertTrue(self.gate['verified']);self.assertEqual(self.gate['compared_saved_samples'],20)

    def test_strict_static_gate_blocks_training_on_mismatch(self):
        bad=self.scores.copy();bad.loc[0,'greybox_score']+=.001
        with self.assertRaisesRegex(ValueError,'reproduction mismatch'):
            verify_static(self.pools,bad,self.store,'float32',self.out)
        failed=json.loads((self.out/'static_verification.json').read_text())
        self.assertFalse(failed['verified'])
        with self.assertRaisesRegex(ValueError,'must pass'):
            fit(self.pools['shadow'],self.pools['validation'],self.cfg,self.out,failed)

    def test_analytic_gradient_matches_finite_difference(self):
        theta=np.r_[np.log(STATIC.astype(float)),3.,.1]
        cfg=dict(DEFAULTS)
        _,gradient,_,_=loss_gradient(theta,self.pools['shadow'],cfg)
        finite=[]
        for i in range(len(theta)):
            plus=theta.copy();minus=theta.copy();plus[i]+=1e-5;minus[i]-=1e-5
            finite.append((loss_gradient(plus,self.pools['shadow'],cfg)[0]-loss_gradient(minus,self.pools['shadow'],cfg)[0])/2e-5)
        np.testing.assert_allclose(gradient,finite,atol=1e-9,rtol=1e-6)
        self.assertGreater(np.linalg.norm(gradient[:-2]),0)
        self.assertNotEqual(gradient[-2],0);self.assertNotEqual(gradient[-1],0)

    def test_optimization_weights_absent_categories_and_direction(self):
        initial=np.r_[np.log(STATIC.astype(float)),np.log(np.expm1(4.)),0.]
        old=validation_loss(initial,self.pools['validation'])
        model=self.train();theta=np.array(model['theta']);weights,scale,_,_=parameters(theta)
        self.assertAlmostEqual(weights.sum(),1.)
        self.assertTrue((weights>0).all());self.assertGreater(scale,0)
        self.assertLess(validation_loss(theta,self.pools['validation']),old)
        self.assertGreater(model['weights']['identifier']/model['weights']['comment'],.1)
        self.assertEqual(model['category_counts_train']['string'],0)
        scored=score_pool(model,self.pools['heldout'])
        self.assertGreater(scored.loc[scored.is_member.eq(1),'adaptive_score'].mean(),scored.loc[scored.is_member.eq(0),'adaptive_score'].mean())
        ordered=scored.sort_values('adaptive_score')
        self.assertTrue(np.all(np.diff(ordered.adaptive_probability)>=0))

    def test_deterministic_training_and_exact_resume(self):
        cfg=dict(self.cfg,max_epochs=50)
        full,complete=fit(self.pools['shadow'],self.pools['validation'],cfg,self.out/'full',self.gate)
        self.assertTrue(complete)
        partial,complete=fit(self.pools['shadow'],self.pools['validation'],cfg,self.out/'resumed',self.gate,epoch_budget=13)
        self.assertFalse(complete)
        with self.assertRaisesRegex(ValueError,'finish'):
            freeze(partial,self.pools['validation'],self.out)
        resumed,complete=fit(self.pools['shadow'],self.pools['validation'],cfg,self.out/'resumed',self.gate)
        self.assertTrue(complete);self.assertEqual(full,resumed)
        again,_=fit(self.pools['shadow'],self.pools['validation'],cfg,self.out/'again',self.gate)
        self.assertEqual(full,again)
        changed=dict(cfg,regularization=.5)
        with self.assertRaisesRegex(ValueError,'identity changed'):
            fit(self.pools['shadow'],self.pools['validation'],changed,self.out/'full',self.gate)

    def test_early_stopping_and_regularization_gradient(self):
        cfg=dict(self.cfg,max_epochs=20,patience=3,min_delta=.1,learning_rate=1e-12)
        model,complete=fit(self.pools['shadow'],self.pools['validation'],cfg,self.out,self.gate)
        state=json.loads((self.out/'training_checkpoint.json').read_text())['state']
        self.assertTrue(complete);self.assertEqual(state['epoch'],3)
        theta=np.r_[np.log(STATIC.astype(float)),3.,.1];theta[0]+=1
        _,g0,_,_=loss_gradient(theta,self.pools['shadow'],dict(DEFAULTS,regularization=0.))
        _,g1,_,_=loss_gradient(theta,self.pools['shadow'],dict(DEFAULTS,regularization=1.))
        weights=parameters(theta)[0];prior=STATIC.astype(float)/STATIC.sum(dtype=float)
        np.testing.assert_allclose((g1-g0)[:-2],weights-prior,atol=1e-14)
        self.assertTrue('validation_bce' in state['history'][0])

    def test_leakage_and_role_checks(self):
        with self.assertRaisesRegex(ValueError,'shadow training'):
            fit(self.pools['heldout'],self.pools['validation'],self.cfg,self.out,self.gate)
        bad=copy.deepcopy(self.parts)
        bad['calibration'].loc[0,'content_sha256']=bad['heldout'].loc[0,'content_sha256']
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_partitions(bad,{k:len(v) for k,v in bad.items()})
        bad=copy.deepcopy(self.parts)
        bad['shadow'].loc[0,'official_split']='test'
        with self.assertRaisesRegex(ValueError,'official split'):
            validate_partitions(bad,{k:len(v) for k,v in bad.items()})
        model=self.train()
        with self.assertRaisesRegex(ValueError,'threshold-calibration'):
            calibrate(model,self.pools['heldout'],self.out)
        threshold=calibrate(model,self.pools['calibration'],self.out)
        payload={k:v for k,v in threshold.items() if k!='artifact_sha256'}
        payload['sample_ids']=[self.parts['heldout'].iloc[0].sample_id]
        with self.assertRaisesRegex(ValueError,'leakage'):
            evaluate(model,seal(payload),self.pools['heldout'],self.scores,self.out,expected_count=20)

    def test_matched_id_evaluation_and_output_schemas(self):
        model=self.train();threshold=calibrate(model,self.pools['calibration'],self.out)
        self.assertLessEqual(threshold['calibration_fpr'],.01)
        with self.assertRaisesRegex(ValueError,'1,000'):
            evaluate(model,threshold,self.pools['heldout'],self.scores,self.out)
        report=evaluate(model,threshold,self.pools['heldout'],self.scores.sample(frac=1,random_state=2),self.out,expected_count=20)
        self.assertEqual(report['coverage']['fraction'],1.)
        self.assertEqual(report['coverage']['matched'],20)
        self.assertEqual(set(report['metrics']),{'adaptive_score','greybox_score','lumia_score','fusion_score'})
        self.assertEqual(set(report['metrics']['adaptive_score']['by_language']),{'Python','Go'})
        for field in ['auc_roc','tpr_at_fpr_le_1pct','low_fpr_resolution_limited']:
            self.assertIn(field,report['metrics']['adaptive_score']['overall'])
        for field in ['theta','weights','static_prior','calibration_scale','calibration_bias','training','validation','pool_signatures','artifact_sha256']:
            self.assertIn(field,model)
        saved=pd.read_parquet(self.out/'heldout_adaptive_scores.parquet')
        self.assertTrue({'sample_id','content_sha256','adaptive_score','adaptive_probability','adaptive_predicted_member','greybox_score','lumia_score','fusion_score'}<=set(saved))
        self.assertEqual(set(saved.sample_id),set(self.parts['heldout'].sample_id))
        bad=self.scores.copy();bad.loc[0,'content_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            evaluate(model,threshold,self.pools['heldout'],bad,self.out,expected_count=20)
        with self.assertRaises(ValueError):match_scores(self.parts['heldout'],self.scores.iloc[:-1])

    def test_feature_cache_schema_checksum_and_identity(self):
        row=self.parts['shadow'].iloc[0].to_dict();arrays=self.store.get(row)
        validate_record(arrays)
        self.assertEqual(arrays['z'].shape,arrays['category'].shape)
        bad=dict(row,is_member=1-row['is_member'])
        with self.assertRaisesRegex(ValueError,'identity/label/role'):
            self.store.get(bad)
        with self.assertRaisesRegex(ValueError,'identity changed'):
            FeatureStore(self.store.path,{'test':'different'})
        self.store.con.execute('UPDATE features SET checksum=? WHERE sample_id=?',('bad',row['sample_id']));self.store.con.commit()
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.store.get(row)

    def test_read_only_attached_cache_reuse_and_incomplete_failure(self):
        attached=FeatureStore(self.store.path,{'test':'synthetic'},read_only=True)
        try:
            factory=Mock(side_effect=AssertionError('must not load target'))
            result=extract_manifest(self.parts['shadow'],attached,None,factory,None)
            self.assertEqual(result['extracted'],0);factory.assert_not_called()
            row=self.parts['shadow'].iloc[0].to_dict()
            with self.assertRaisesRegex(RuntimeError,'read-only'):attached.put(row,token_record(row))
            self.store.con.execute('DELETE FROM features WHERE sample_id=?',(row['sample_id'],));self.store.con.commit()
            with self.assertRaisesRegex(FileNotFoundError,'missing sample IDs'):
                extract_manifest(self.parts['shadow'],attached,None,factory,None)
            factory.assert_not_called()
        finally:attached.close()

    def test_complete_cache_skips_target_factory_and_source(self):
        factory=Mock(side_effect=AssertionError('must not load a target'))
        source=SimpleNamespace(batches=Mock(side_effect=AssertionError('must not read source')))
        report=extract_manifest(self.parts['shadow'],self.store,source,factory,Mock())
        self.assertEqual(report['extracted'],0);factory.assert_not_called()

    def test_per_sample_extraction_resume_with_synthetic_extractor(self):
        manifest=self.parts['shadow'].iloc[:2].copy()
        calls=[]
        class Extractor:
            def one(self,text,language):
                calls.append(text)
                row=manifest.loc[manifest.content.eq(text)].iloc[0].to_dict()
                return token_record(row)
        source=SimpleNamespace(batches=lambda pending,size:iter([manifest[manifest.sample_id.isin(pending.sample_id)]]))
        monitor=SimpleNamespace(guard=lambda:None,tick=lambda *args:None)
        store=FeatureStore(self.out/'fresh.sqlite',{'synthetic':True})
        try:
            extract_manifest(manifest,store,source,Extractor,monitor)
            extract_manifest(manifest,store,source,Extractor,monitor)
            self.assertEqual(len(calls),2)
        finally:store.close()


class AlignmentAndDataTests(unittest.TestCase):
    def test_token_alignment_special_padding_empty_and_max_overlap(self):
        char_weights=np.array([1.,10.,5.,5.],dtype=np.float32)
        category=np.array([1,6,3,4],dtype=np.uint8)
        offsets=np.array([[0,1],[0,2],[2,3],[3,3],[3,4],[3,4]])
        labels,weights,eligible=align_tokens(offsets,[0,0,1,0,0,0],[1,1,1,1,0,0],category,char_weights)
        np.testing.assert_array_equal(eligible,[0,1,0,0,0,0])
        self.assertEqual(labels[1],6);self.assertEqual(weights[1],10.)
        with self.assertRaises(ValueError):align_tokens([[0,0],[0,5]],[1,0],[1,1],category,char_weights)

    def test_invalid_features_and_empty_samples_fail_explicitly(self):
        row=frames()['shadow'].iloc[0].to_dict();record=token_record(row)
        record['signals'][1]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):validate_record(record)
        record=token_record(row);record['n_targets']=np.asarray(3,dtype=np.int32)
        with self.assertRaisesRegex(ValueError,'counts disagree'):validate_record(record)
        with self.assertRaisesRegex(ValueError,'no eligible targets'):
            aggregate(np.zeros(len(CATEGORIES)+2),np.zeros((1,len(CATEGORIES))),np.zeros((1,len(CATEGORIES))))

    def test_bce_is_stable_and_correct(self):
        self.assertAlmostEqual(bce_with_logits(np.array([0.,0.]),np.array([0.,1.])),np.log(2))
        self.assertEqual(bce_with_logits(np.array([1000.,-1000.]),np.array([1.,0.])),0.)
        self.assertEqual(bce_with_logits(np.array([-1000.,1000.]),np.array([1.,0.])),1000.)
        with self.assertRaises(ValueError):bce_with_logits(np.array([np.inf]),np.array([1.]))

    def test_missing_and_ambiguous_runtime_manifest_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError) as error:resolve_paths(input_root=root)
            for name in FILES.values():self.assertIn(name,str(error.exception))
            for run in ['one','two']:
                (root/run).mkdir();(root/run/FILES['shadow']).touch()
            with self.assertRaisesRegex(ValueError,'Multiple saved runs'):resolve_paths(input_root=root)
            with self.assertRaises(FileNotFoundError) as error:resolve_paths(root/'one')
            self.assertNotIn('shadow:',str(error.exception))

    def test_load_exact_3000_synthetic_identities_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parts=frames(COUNTS)
            for role,frame in parts.items():frame.to_parquet(root/FILES[role],index=False)
            scores=parts['heldout'].copy()
            for c in ['lumia_score','greybox_score','fusion_score']:scores[c]=.5
            scores.to_parquet(root/FILES['scores'],index=False)
            cfg={k:'synthetic' for k in ['model_id','model_revision','dataset_id','dataset_revision','dtype','max_length','seed','greybox','versions','upstream_sha']}
            (root/'config.json').write_text(json.dumps(cfg))
            loaded,reference,_=load_run(resolve_paths(input_root=root))
            self.assertEqual({k:len(v) for k,v in loaded.items()},COUNTS)
            self.assertEqual(len(reference),1000)
            parts['shadow'].iloc[:-1].to_parquet(root/FILES['shadow'],index=False)
            with self.assertRaisesRegex(ValueError,'expected 1600'):load_run(resolve_paths(root))

    def test_overlap_attribution_matches_real_structural_weight_method(self):
        # Real CPU parser; dictionary and linter are explicit synthetic fixtures.
        import tree_sitter, tree_sitter_python
        python_parser=tree_sitter.Parser(tree_sitter.Language(tree_sitter_python.language()))
        categorizer=Categorizer.__new__(Categorizer)
        categorizer.parsers={'Python':python_parser}
        categorizer.dictionary=SimpleNamespace(check=lambda word:word!='fremdwort')
        categorizer.extractor=SimpleNamespace(split_identifier=lambda name:[name])
        categorizer.linter=SimpleNamespace(extract_errors=lambda text,language:[])
        categorizer.psych=re.compile(r'(TODO|FIXME|HACK|Note to self)',re.I)
        text='x = "é hello"\nshort longidentifier fremdwort\n# plain TODO\n'
        labels,weights,_=categorizer.characters(text,'Python')
        np.testing.assert_array_equal(STATIC[labels],weights)
        for value,index in [('x',0),('hello',3),('short',1),('longidentifier',2),('fremdwort',5),('plain',6),('TODO',7)]:
            self.assertEqual(labels[text.index(value)],index)
        categorizer.linter=SimpleNamespace(extract_errors=lambda text,language:[(1,7)])
        labels,_,_=categorizer.characters(text,'Python')
        self.assertEqual(labels[6],4)  # equal 5 weight: linter takes precedence over string


class ExecutionModeTests(unittest.TestCase):
    def setUp(self):
        self.parts=frames(COUNTS)
        languages=['Python','Go','Java','Ruby','Rust']
        for frame in self.parts.values():
            frame['language']=[languages[(i//2)%5] for i in range(len(frame))]
            frame['sample_id']=frame.language+'/'+frame.official_split+'/'+frame.source_row.astype(str)
        self.scores=self.parts['heldout'].copy()
        self.scores['greybox_score']=[static_score(token_record(row)) for row in self.scores.to_dict('records')]
        self.scores['lumia_score']=.2+.6*self.scores.is_member
        self.scores['fusion_score']=.9*self.scores.lumia_score+.1*self.scores.greybox_score

    def test_deterministic_balanced_stratified_subset_preserves_identity(self):
        chosen,execution=select_partitions(self.parts,'smoke',seed=17)
        reversed_input={role:frame.iloc[::-1] for role,frame in self.parts.items()}
        again,execution_again=select_partitions(reversed_input,'smoke',seed=17)
        different,_=select_partitions(self.parts,'smoke',seed=18)
        self.assertEqual(execution['expected_counts'],SMOKE_SIZES)
        self.assertTrue(any(set(chosen[k].sample_id)!=set(different[k].sample_id) for k in chosen))
        for role,frame in chosen.items():
            pd.testing.assert_frame_equal(frame,again[role])
            self.assertEqual(len(frame),SMOKE_SIZES[role])
            self.assertEqual(frame.is_member.value_counts().to_dict(),{0:len(frame)//2,1:len(frame)//2})
            counts=frame.groupby(['language','is_member']).size()
            self.assertEqual(len(counts),10);self.assertLessEqual(counts.max()-counts.min(),1)
            reference=self.parts[role].set_index('sample_id').loc[frame.sample_id].reset_index()
            pd.testing.assert_frame_equal(frame[IDENTITY+['membership']],reference[IDENTITY+['membership']])
        self.assertEqual(execution['selected_manifests'],execution_again['selected_manifests'])
        validate_execution(execution,chosen)
        sizes=dict(SMOKE_SIZES,shadow=82)
        chosen,_=select_partitions(self.parts,'smoke',sizes,seed=17)
        self.assertEqual(len(chosen['shadow']),82)
        by_language=chosen['shadow'].groupby(['language','is_member']).size().unstack()
        np.testing.assert_array_equal(by_language[0],by_language[1])
        self.assertEqual(by_language[0].max()-by_language[0].min(),1)

    def test_complete_originals_checked_first_in_both_modes(self):
        truncated=dict(self.parts,heldout=self.parts['heldout'].iloc[:-1])
        for mode in ['smoke','full']:
            with self.assertRaisesRegex(ValueError,'expected 1000'):
                select_partitions(truncated,mode)
        broken=copy.deepcopy(self.parts)
        broken['shadow'].loc[0,'source_row']=-1
        with self.assertRaisesRegex(ValueError,'source rows'):
            select_partitions(broken,'smoke')
        broken=copy.deepcopy(self.parts)
        broken['validation'].loc[0,'content_sha256']=broken['heldout'].loc[0,'content_sha256']
        with self.assertRaisesRegex(ValueError,'leakage'):
            select_partitions(broken,'smoke')
        for sizes in [dict(SMOKE_SIZES,shadow=81),dict(SMOKE_SIZES,heldout=8),dict(SMOKE_SIZES,shadow=1602)]:
            with self.assertRaisesRegex(ValueError,'smoke count'):
                select_partitions(self.parts,'smoke',sizes)

    def test_selected_identity_leakage_and_count_checks_remain_strict(self):
        selected,execution=select_partitions(self.parts,'smoke')
        changed=copy.deepcopy(selected)
        changed['shadow'].loc[0,'content_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            validate_execution(execution,changed)
        changed=copy.deepcopy(selected)
        changed['calibration'].loc[0,'content_sha256']=changed['heldout'].loc[0,'content_sha256']
        with self.assertRaisesRegex(ValueError,'leakage'):
            validate_execution(execution,changed)
        changed=dict(selected,heldout=selected['heldout'].iloc[:-2])
        with self.assertRaisesRegex(ValueError,'expected 30'):
            validate_execution(execution,changed)

    def test_full_mode_keeps_every_row_and_production_counts(self):
        full,execution=select_partitions(self.parts,'full')
        for role,frame in full.items():pd.testing.assert_frame_equal(frame,self.parts[role])
        self.assertEqual(execution['expected_counts'],COUNTS)
        self.assertEqual(execution['original_manifests'],execution['selected_manifests'])
        payload={k:v for k,v in execution.items() if k!='artifact_sha256'}
        payload['expected_counts']=dict(COUNTS,heldout=30)
        with self.assertRaisesRegex(ValueError,'production counts'):
            validate_execution(seal(payload),full)

    def test_partial_smoke_feature_cache_reuse_full_identity_checks(self):
        smoke,_=select_partitions(self.parts,'smoke')
        selected=pd.concat(list(smoke.values()),ignore_index=True)
        full=pd.concat(list(self.parts.values()),ignore_index=True)
        identity={'model':'synthetic-frozen','revision':'abc','dtype':'float32','max_length':5}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=FeatureStore(root/'smoke.sqlite',identity)
            for row in selected.to_dict('records'):source.put(row,token_record(row))
            source.close()
            destination=FeatureStore(root/'full.sqlite',identity)
            try:
                imported=destination.reuse_from(root/'smoke.sqlite',full)
                self.assertEqual(imported['imported'],150)
                self.assertEqual(destination.con.execute('SELECT COUNT(*) FROM features').fetchone()[0],150)
                self.assertEqual(destination.reuse_from(root/'smoke.sqlite',full)['imported'],0)
                self.assertEqual(sum(destination.get(row) is None for row in full.to_dict('records')),2850)
                tampered=selected.iloc[:1].copy();tampered.loc[0,'content_sha256']='0'*64
                with self.assertRaisesRegex(ValueError,'identity/label/role'):
                    destination.reuse_from(root/'smoke.sqlite',tampered)
            finally:destination.close()
            incompatible=FeatureStore(root/'wrong-target.sqlite',dict(identity,revision='different'))
            try:
                with self.assertRaisesRegex(ValueError,'identity changed'):
                    incompatible.reuse_from(root/'smoke.sqlite',full)
            finally:incompatible.close()
            self.assertEqual({p.name for p in root.iterdir()},{'smoke.sqlite','full.sqlite','wrong-target.sqlite'})

    def test_smoke_and_full_end_to_end_synthetic_only(self):
        # In-memory synthetic features keep the 3,000-example mode test inexpensive.
        class MemoryStore:
            read_only=False
            def __init__(self):self.records={}
            def get(self,row):
                value=self.records.get(row['sample_id'])
                if value is None:return None
                metadata,arrays=value
                if metadata!={k:row[k] for k in IDENTITY}:raise ValueError('synthetic identity mismatch')
                return arrays
            def put(self,row,arrays):
                validate_record(arrays)
                self.records[row['sample_id']]=({k:row[k] for k in IDENTITY},arrays)
            def digest(self,manifest):
                for row in manifest.to_dict('records'):
                    if self.get(row) is None:raise FileNotFoundError(row['sample_id'])
                return fingerprint(manifest[IDENTITY].to_dict('records'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results={};store=MemoryStore()
            for mode in ['smoke','full']:
                selected,execution=select_partitions(self.parts,mode,seed=17)
                all_selected=pd.concat(list(selected.values()),ignore_index=True)
                source=SimpleNamespace(batches=lambda pending,size:iter([pending]))
                by_text=all_selected.set_index('content')
                requested=[]
                class SyntheticExtractor:
                    def one(self,text,language):
                        requested.append(text)
                        return token_record(by_text.loc[text].to_dict())
                out=root/mode;out.mkdir();store.path=out/'synthetic.sqlite'
                summary=extract_manifest(all_selected,store,source,SyntheticExtractor,SimpleNamespace(guard=lambda:None,tick=lambda *a:None))
                self.assertEqual(len(requested),150 if mode=='smoke' else 2850)
                self.assertEqual(summary['requested'],150 if mode=='smoke' else 3000)
                pools={role:make_pool(frame,role,store) for role,frame in selected.items()}
                gate=verify_static(pools,self.scores,store,'float32',out,execution=execution)
                self.assertEqual(gate['compared_saved_samples'],30 if mode=='smoke' else 1000)
                self.assertEqual(gate['absolute_tolerance'],2e-6)
                weights,complete=fit(pools['shadow'],pools['validation'],dict(max_epochs=3,patience=5,seed=17),out,gate)
                self.assertTrue(complete);self.assertEqual(weights['mode'],mode)
                frozen=freeze(weights,pools['validation'],out)
                threshold=calibrate(frozen,pools['calibration'],out)
                report=evaluate(frozen,threshold,pools['heldout'],self.scores,out)
                expected=30 if mode=='smoke' else 1000
                self.assertEqual(report['coverage']['requested'],expected)
                self.assertEqual(report['coverage']['matched'],expected)
                self.assertEqual(report['preliminary'],mode=='smoke')
                if mode=='smoke':
                    self.assertTrue(report['resolution_limited']);self.assertTrue(threshold['preliminary'])
                    for metrics in report['metrics'].values():
                        for row in [metrics['overall'],*metrics['by_language'].values()]:
                            self.assertTrue(row['preliminary']);self.assertTrue(row['resolution_limited'])
                    self.assertEqual(set(weights['training']['sample_ids']),set(selected['shadow'].sample_id))
                    bad=self.scores.copy()
                    bad.loc[bad.sample_id.eq(selected['heldout'].iloc[0].sample_id),'greybox_score']+=.001
                    with self.assertRaisesRegex(ValueError,'reproduction mismatch'):
                        verify_static(pools,bad,store,'float32',out,execution=execution)
                else:
                    smoke=results['smoke']
                    with self.assertRaisesRegex(ValueError,'identity changed'):
                        fit(pools['shadow'],pools['validation'],dict(max_epochs=3,patience=5,seed=17),root/'smoke',gate)
                    with self.assertRaisesRegex(ValueError,'mode mismatch'):
                        evaluate(frozen,smoke['threshold'],pools['heldout'],self.scores,out)
                    with self.assertRaisesRegex(ValueError,'selection/count/identity mismatch'):
                        evaluate(smoke['model'],smoke['threshold'],pools['heldout'],self.scores,out)
                    # Full scores still use precisely the same aggregate; mode only adds provenance.
                    expected_scores=aggregate(np.array(frozen['theta']),pools['heldout'].counts,pools['heldout'].sums)
                    saved=pd.read_parquet(out/'heldout_adaptive_scores.parquet')
                    np.testing.assert_array_equal(saved.adaptive_score.to_numpy(),expected_scores)
                results[mode]={'model':frozen,'threshold':threshold,'gate':gate}
            self.assertNotEqual(results['smoke']['model']['fit_identity'],results['full']['model']['fit_identity'])


if __name__=='__main__':unittest.main()

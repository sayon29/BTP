"""CPU tests: real parsers/ASTs, deterministic fake MLM/tokenizer, no model loads.
Run: python3 -m unittest discover -s tests -v
Requires numpy, pandas, sklearn, pyarrow, tree-sitter and five grammar wheels.
"""
import ast
import hashlib
from pathlib import Path
import random
import re
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from neighbourhood_attack.perturb import Perturber, parser, select_sites, sites, validate_change
from neighbourhood_attack.runner import Cache, evaluate_results, validate_heldout_manifest, validate_saved_scores, run
from shared.weighted_anomaly import StructuralWeights, WeightedAnomalyMIA


class Tokenizer:
    """Character offsets with one initial special token, bounded target context."""
    def __call__(self, text, max_length, **kwargs):
        text = text[:max_length-1]
        return SimpleNamespace(input_ids=[0]+[ord(c) for c in text],
                               offset_mapping=[(0,0)]+[(i,i+1) for i in range(len(text))],
                               special_tokens_mask=[1]+[0]*len(text))


def generator(policy='sersem', count=5, weights=None):
    gen = Perturber.__new__(Perturber)  # never initialize/download an MLM
    gen.cfg = dict(seed=17, k=count, max_length=4096, max_sites=12, mlm_batch_size=4,
                   allow_python_local_rename=True, allow_comment_insertion=False, site_importance=policy)
    gen.structural = SimpleNamespace(char_weights=lambda text, language:
                                    (np.array(weights if weights is not None else [10.]*len(text)), False)) if policy == 'sersem' else None
    gen.proposals = lambda text, candidates: [['return','fresh','other','third','fourth','fifth'] for _ in candidates]
    return gen


CODE = '# header\n# header\ndef fun(arg):\n    local = arg + 1\n    # alpha bravo charlie delta echo foxtrot\n    return local\n'


class SelectionTests(unittest.TestCase):
    def test_exact_target_projection_and_max_over_rename_occurrences(self):
        enc = SimpleNamespace(offset_mapping=[(0,1),(1,3),(3,4),(4,6)],special_tokens_mask=[0,0,1,0])
        spans = [dict(kind='python_local', start=1,end=2,occurrences=[(1,2),(5,6)]),
                 dict(kind='comment',start=0,end=1,occurrences=[(0,1)]),
                 dict(kind='comment_insert',start=6,end=6,occurrences=[(6,6)])]
        got = select_sites(spans, enc, [100.,1.,3.,99.,5.,10.], 'sersem', random.Random(0), 3)
        self.assertEqual(got[0]['importance'], 10.)
        self.assertEqual([s['importance'] for s in got[1:]], [0.,0.])

    def test_high_weight_first_and_reproducible_ties(self):
        weights = np.ones(len(CODE)); start = CODE.index('local');weights[start:start+5]=20
        gen = generator(weights=weights)
        a, _ = gen.generate(CODE,'Python',Tokenizer())
        b, _ = gen.generate(CODE,'Python',Tokenizer())
        self.assertEqual(a,b)
        self.assertEqual(a[0]['transformation'],'python_local')
        self.assertEqual(a[0]['importance'],20.)
        self.assertEqual(len(a),5)
        self.assertEqual(len({n['start'] for n in a}),5)
        self.assertEqual(len({n['content_sha256'] for n in a}),5)

    def test_diversity_and_one_transformation_per_neighbour(self):
        neighbours, failures = generator().generate(CODE,'Python',Tokenizer())
        self.assertEqual(len({n['start'] for n in neighbours}),5)
        for n in neighbours:
            expected = CODE
            for a,b in sorted(n['occurrences'],reverse=True):
                expected = expected[:a]+n['replacement']+expected[b:]
            self.assertEqual(n['content'],expected)
            ast.parse(n['content'])
        self.assertTrue(any(f['reason']=='ineligible_mlm_replacement' and 'site' in f for f in failures))

    def test_fewer_sites_round_robin_and_explicit_failure(self):
        text = '# first\n# second\nx = 1\n# alpha bravo\n'
        gen = generator(); neighbours, _ = gen.generate(text,'Python',Tokenizer())
        self.assertEqual(len(neighbours),5)
        self.assertEqual(len({n['start'] for n in neighbours[:2]}),2)
        gen.proposals = lambda text, sites: [['fresh'] for _ in sites]
        neighbours, failures = gen.generate(text,'Python',Tokenizer())
        self.assertEqual(len(neighbours),2)
        self.assertEqual(failures[-1]['reason'],'insufficient_distinct_safe_neighbours')
        self.assertEqual(failures[-1]['required'],5)

    def test_uniform_policy_remains_available(self):
        neighbours, _ = generator('uniform').generate(CODE,'Python',Tokenizer())
        self.assertEqual(len(neighbours),5)
        self.assertTrue(all(n['importance'] is None and n['importance_source']=='uniform' for n in neighbours))

    def test_missing_weights_never_fabricated(self):
        enc = Tokenizer()(CODE,max_length=4096)
        with self.assertRaisesRegex(RuntimeError,'unavailable'):
            select_sites([],enc,None,'sersem',random.Random(0),12)
        with self.assertRaises(ValueError):
            select_sites([],enc,None,'invented',random.Random(0),12)

    def test_visible_context_boundary(self):
        gen = generator(); gen.cfg['max_length']=20
        neighbours, failures = gen.generate(CODE,'Python',Tokenizer())
        self.assertEqual(neighbours,[])
        self.assertEqual(failures[0]['reason'],'no_safe_visible_transformation')


class SafetyTests(unittest.TestCase):
    def test_consistent_rename_and_partial_rename_rejected(self):
        eligible, tree, _ = sites(CODE,'Python',len(CODE),True,False)
        site = next(s for s in eligible if s['kind']=='python_local')
        replacement = 'fresh'
        changed = CODE
        for a,b in sorted(site['occurrences'],reverse=True):
            changed = changed[:a]+replacement+changed[b:]
        self.assertTrue(validate_change(CODE,changed,'Python',tree,site,replacement)[0])
        a,b = site['occurrences'][0]
        self.assertFalse(validate_change(CODE,CODE[:a]+replacement+CODE[b:],'Python',tree,site,replacement)[0])

    def test_reflection_calls_and_attributes_excluded(self):
        for statement in ['return locals()', 'return arg.value', 'return str(arg)']:
            text = 'def fun(arg):\n    local = arg\n    '+statement+'\n'
            eligible, _, _ = sites(text,'Python',len(text),True,False)
            self.assertFalse(any(s['kind']=='python_local' for s in eligible))

    def test_comment_preserves_noncomment_tokens_across_languages(self):
        examples = {'Python':'x = 1', 'Ruby':'x = 1', 'Java':'class Test {}',
                    'Go':'package main\nfunc main() {}', 'Rust':'fn main() {}'}
        for language, code in examples.items():
            marker = '#' if language in {'Python','Ruby'} else '//'
            text = marker+' header\n'+marker+' header\n'+code+'\n'+marker+' alpha bravo\n'
            eligible, tree, error = sites(text,language,len(text),True,False)
            self.assertIsNone(error)
            site = next(s for s in eligible if s['kind']=='comment')
            a,b = site['start'],site['end']
            candidate = text[:a]+'fresh'+text[b:]
            self.assertTrue(validate_change(text,candidate,language,tree,site,'fresh')[0],language)
            bad = candidate.replace(code,code+' invalid syntax !!!')
            self.assertFalse(validate_change(text,bad,language,tree,site,'fresh')[0],language)

    def test_directives_and_original_syntax_error(self):
        text = '# first\n# second\nx = 1\n# noqa important\n'
        self.assertEqual(sites(text,'Python',len(text),False,False)[0],[])
        neighbours, failures = generator().generate('def broken(:\n','Python',Tokenizer())
        self.assertEqual(neighbours,[])
        self.assertEqual(failures[0]['reason'],'original_syntax_error')

    def test_comment_insertion_is_verified(self):
        text = '# first\n# second\nx = 1\n'
        candidates, tree, _ = sites(text,'Python',len(text),False,True)
        site = next(s for s in candidates if s['kind']=='comment_insert')
        candidate = text+site['prefix']+'fresh'+site['suffix']
        self.assertTrue(validate_change(text,candidate,'Python',tree,site,'fresh')[0])


class StructuralTests(unittest.TestCase):
    def test_shared_weight_method_and_all_weight_categories(self):
        self.assertIs(WeightedAnomalyMIA.char_weights,StructuralWeights.char_weights)
        text = 'x = "hello"\nshort longidentifier fremdwort\n# plain\nTODO'
        # Real syntax parser; dictionary/linter are explicit test fixtures only.
        provider = StructuralWeights.__new__(StructuralWeights)
        provider.parsers = {'Python':parser('Python')}
        provider.dictionary = SimpleNamespace(check=lambda word: word!='fremdwort')
        provider.extractor = SimpleNamespace(split_identifier=lambda name:[name])
        provider.linter = SimpleNamespace(extract_errors=lambda text,language:[])
        provider.psych = re.compile(r'(TODO|FIXME|HACK|Note to self)',re.I)
        weights, _ = provider.char_weights(text,'Python')
        for value, weight in [('x',.1),('hello',5.),('short',1.),('longidentifier',3.),('fremdwort',10.),('plain',10.),('TODO',10.)]:
            self.assertAlmostEqual(float(weights[text.index(value)]),weight)
        provider.linter = SimpleNamespace(extract_errors=lambda text,language:[(1,1)])
        weights, _ = provider.char_weights(text,'Python')
        self.assertEqual(weights[0],5.)


class ReportingTests(unittest.TestCase):
    def frame(self, count=1000):
        return pd.DataFrame([dict(sample_id=str(i),language='Python' if i%4<2 else 'Go',
                                  is_member=i%2,membership='member' if i%2 else 'non-member',
                                  content_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                                  official_split='test',source_row=i,split_role='heldout_test') for i in range(count)])

    def test_exact_manifest_and_score_guards(self):
        manifest = self.frame();validate_heldout_manifest(manifest)
        for bad in [manifest.iloc[:999],pd.concat([manifest.iloc[:999],manifest.iloc[:1]])]:
            with self.assertRaises(ValueError):validate_heldout_manifest(bad)
        baseline = manifest.copy()
        for c in ['lumia_score','greybox_score','fusion_score']:baseline[c]=baseline.is_member.astype(float)
        validate_saved_scores(manifest,baseline)
        for bad in [baseline.iloc[:999],baseline.drop(columns='fusion_score')]:
            with self.assertRaises(ValueError):validate_saved_scores(manifest,bad)
        baseline.loc[0,'content_sha256']='changed'
        with self.assertRaises(ValueError):validate_saved_scores(manifest,baseline)

    def test_matched_subset_metrics_coverage_and_identity(self):
        manifest = self.frame(12)
        baseline = manifest.copy()
        for c in ['lumia_score','greybox_score','fusion_score']:baseline[c]=baseline.is_member.astype(float)
        results = manifest.copy();results['neighbourhood_score']=results.is_member.astype(float)
        results['status']='ok'; results.loc[0,'status']='failure';results.loc[0,'neighbourhood_score']=np.nan
        baseline = baseline.iloc[2:].copy()  # common successful cohort has 10 examples
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'baseline.parquet';baseline.to_parquet(path,index=False)
            report=evaluate_results(results,manifest,path,tmp)
            self.assertEqual(report['coverage']['valid'],11)
            self.assertEqual(report['comparison_coverage']['matched'],10)
            for c in ['neighbourhood_score','lumia_score','greybox_score','fusion_score']:
                metrics=report['matched_metrics'][c]
                self.assertEqual(metrics['overall']['n'],10)
                self.assertEqual(metrics['overall']['auc_roc'],1.)
                self.assertEqual(metrics['overall']['tpr_at_fpr_le_1pct'],1.)
                self.assertEqual(set(metrics['by_language']),{'Go','Python'})
            results.loc[2,'language']='Rust'
            with self.assertRaises(ValueError):evaluate_results(results,manifest,path,tmp)

    def test_run_formula_audit_export_and_resume_without_models(self):
        manifest = self.frame(4);manifest["language"] = "Python"
        frame = manifest.copy();frame['content'] = [CODE]*4
        class Store:
            def batches(self, pending, size):
                if len(pending):
                    yield frame[frame.sample_id.isin(pending.sample_id)]
        class NLL:
            calls = texts = seconds = 0
            def __init__(self, *args):pass
            def score(self, texts):return [{'nll':len(t)/10.} for t in texts]
        gen = generator()
        gen.loading_info = {}
        gen.model = SimpleNamespace(to=lambda device:None)
        monitor = SimpleNamespace(guard=lambda:None, tick=lambda *a,**kw:None)
        cfg = dict(gen.cfg, source_chunk_size=4, park_mlm_on_cpu=True)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {'torch':SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))}), \
                 patch('neighbourhood_attack.perturb.Perturber', return_value=gen) as factory, \
                 patch('neighbourhood_attack.runner.TargetNLL', NLL):
                first = run(cfg,manifest,Store(),None,Tokenizer(),tmp,monitor)
                factory.assert_called_once()
                factory.reset_mock()
                second = run(cfg,manifest,Store(),None,Tokenizer(),tmp,monitor)
                factory.assert_not_called()
            self.assertEqual(first['metrics'],second['metrics'])
            saved = pd.read_parquet(Path(tmp)/'neighbourhood_scores.parquet')
            for row in saved.itertuples():
                self.assertEqual(row.neighbour_count,5)
                self.assertAlmostEqual(row.neighbourhood_score,sum(row.neighbour_nlls)/5-row.original_nll)
                self.assertEqual(len(row.neighbour_transformations),5)
                self.assertTrue(len(row.selected_sites)>0)
                self.assertTrue(all(n['importance']==10. for n in row.neighbour_transformations))

    def test_checkpoint_roundtrip_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.sqlite';cache=Cache(path)
            payload={'selected_sites':[{'importance':10.,'start':3}], 'failures':[{'reason':'test'}]}
            cache.put('generation','key',payload);cache.close()
            cache=Cache(path);self.assertEqual(cache.get('generation','key'),payload)
            cache.con.execute("UPDATE cache SET checksum='invalid'");cache.con.commit()
            with self.assertRaises(ValueError):cache.get('generation','key')
            cache.close()


if __name__ == '__main__':
    unittest.main()

"""Deployment regression tests using temporary files and synthetic scores only."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd
from spw_code_attack.archive import REQUIRED_OUTPUTS, create_results_archive
from spw_code_attack.data import fixed_configuration
from spw_code_attack.evaluation import evaluate
from spw_code_attack.runner import workload_estimate
from test_spw import partitions
from adaptive_token_attack.data import select_partitions


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_optional_adaptive_and_only_required_inference_roles(self):
        selected, _ = select_partitions(partitions(), 'smoke', seed=42)
        scored = pd.concat([selected['calibration'], selected['heldout']], ignore_index=True)
        scored['spw_inspired_score'] = -.4 + .2*scored.is_member
        base = selected['heldout'].copy()
        for key in ('greybox_score', 'lumia_score', 'fusion_score'):
            base[key] = base.is_member*.2
        with tempfile.TemporaryDirectory() as temp, patch('spw_code_attack.evaluation.atomic_parquet'):
            report = evaluate(scored, selected, base, None, temp, 'smoke')
            self.assertFalse(report['adaptive_comparison_included'])
            self.assertEqual(len(report['heldout_diagnostic_roc_only']), 4)
            self.assertEqual(report['coverage']['matched'], 30)
            with self.assertRaises(ValueError):
                evaluate(scored.iloc[1:], selected, base, None, temp, 'smoke')
            wrong = selected['heldout'].copy()
            wrong['adaptive_score'] = 0.
            wrong.loc[0, 'content_sha256'] = 'changed'
            with self.assertRaises(ValueError):
                evaluate(scored, selected, base, wrong, temp, 'smoke')

    def test_frozen_configuration_rejects_edits(self):
        with patch.object(Path, 'read_bytes', return_value=b'{}'):
            with self.assertRaisesRegex(ValueError, 'Frozen scientific'):
                fixed_configuration()

    def test_measured_estimate_and_cache_only(self):
        frame = pd.DataFrame(dict(cache_hit=[False, True, False],
            session_seconds=[2., 0., 4.], session_forward_calls=[10, 0, 20]))
        result = workload_estimate(frame)
        self.assertEqual(result['measured_samples'], 2)
        self.assertEqual(result['estimated_full_scoring_minutes'], 60.)
        self.assertEqual(result['estimated_full_transformer_calls'], 18000)
        self.assertEqual(workload_estimate(frame.iloc[1:2])['status'], 'unavailable_cache_only')

    def outputs(self, out):
        out.mkdir()
        for name in REQUIRED_OUTPUTS:
            if name.endswith('.sqlite'):
                with sqlite3.connect(out/name) as connection:
                    connection.execute('CREATE TABLE results (id INTEGER)')
            else:
                (out/name).write_text('{}')

    def test_archive_complete_outputs_and_crc(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp);out = work/'outputs';self.outputs(out)
            archive = create_results_archive(out, work, 'smoke-test', 0)
            with zipfile.ZipFile(archive) as zipped:
                self.assertIsNone(zipped.testzip())
                self.assertTrue(set(REQUIRED_OUTPUTS) <= set(zipped.namelist()))
            self.assertTrue((out/'spw_cache.sqlite').is_file())
            (out/'metrics.json').unlink()
            with self.assertRaises(FileNotFoundError):
                create_results_archive(out, work, 'incomplete', 0)

    def test_archive_rejects_active_sqlite_and_low_disk(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp);out = work/'outputs';self.outputs(out)
            sidecar = out/'spw_cache.sqlite-wal';sidecar.write_bytes(b'active')
            with self.assertRaises(RuntimeError):
                create_results_archive(out, work, 'active', 0)
            sidecar.unlink()
            with patch('spw_code_attack.archive.shutil.disk_usage') as disk:
                disk.return_value.free = 0
                with self.assertRaisesRegex(RuntimeError, 'Insufficient disk'):
                    create_results_archive(out, work, 'full-disk')
            with self.assertRaises(ValueError):
                create_results_archive(out, out/'nested', 'nested', 0)

    def test_notebook_wiring_and_source_bundle(self):
        path = ROOT/'notebooks/07_spw_code.ipynb'
        if not path.is_file():
            self.skipTest('Notebook and upload ZIP are separate artifacts')
        notebook = json.loads(path.read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        self.assertIn('require_cuda=True', code)
        self.assertIn("prepare_plans(scoring_partitions,", code)
        self.assertIn('run_scores(scoring_partitions,', code)
        self.assertNotIn("rglob('heldout_adaptive_scores.parquet')", code)
        self.assertIn("write_json(OUT/'environment.json',environment)", code)
        self.assertIn('create_results_archive(OUT, WORK_ROOT, RUN_ID, DISK_RESERVE_GIB)', code)
        self.assertEqual(path.read_bytes(), (ROOT/'notebooks/05_spw_code.ipynb').read_bytes())
        archive_path = ROOT/'notebooks/spw_kaggle_source.zip'
        with zipfile.ZipFile(archive_path) as archive:
            self.assertIsNone(archive.testzip())
            manifest_bytes = archive.read('spw_code_attack/source_manifest.json')
            self.assertIn(hashlib.sha256(manifest_bytes).hexdigest(), code)
            manifest = json.loads(manifest_bytes)
            self.assertEqual(set(archive.namelist()), set(manifest)|{'spw_code_attack/source_manifest.json'})
            for name, expected in manifest.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected, name)
                self.assertEqual((ROOT/name).read_bytes(), archive.read(name), name)
            self.assertEqual(archive.read('spw_code_attack/requirements-kaggle.txt'),
                             archive.read('shared/requirements-kaggle.txt'))

    def test_source_bootstrap_zip_extracted_repeat_and_tampering(self):
        notebook = ROOT/'notebooks/07_spw_code.ipynb'
        if not notebook.exists():
            self.skipTest('Notebook delivered separately from source ZIP')
        bootstrap = ''.join(json.loads(notebook.read_text())['cells'][3]['source'])
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            # Exercise the actual notebook bootstrap, with a minimal synthetic manifest.
            inputs = work/'input';source = inputs/'nested/dataset';source.mkdir(parents=True)
            names = ['shared/kaggle_setup.py','shared/requirements-kaggle.txt']
            names += ['spw_code_attack/'+n for n in ('reproducibility.json','method.py','inference.py','evaluation.py')]
            manifest = {}
            for name in names:
                p = source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
                manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            path = source/'spw_code_attack/source_manifest.json'
            path.write_text(json.dumps(manifest))
            pin = hashlib.sha256(path.read_bytes()).hexdigest()
            lines = bootstrap.splitlines()
            lines = [f'SOURCE_MANIFEST_SHA256 = {pin!r}' if l.startswith('SOURCE_MANIFEST_SHA256 =') else l for l in lines]
            code = '\n'.join(lines)
            def run(project, upload):
                env = dict(Path=Path, INPUT_ROOT=inputs, PROJECT_ROOT=project,
                           UPLOAD_SOURCE=upload, UPLOAD_ZIP=upload/'spw_kaggle_source.zip')
                exec(compile(code,'notebook-bootstrap','exec'), env)
                return env
            import sys
            with patch.object(sys, 'path', list(sys.path)):
                extracted = work/'extracted';run(extracted, source);run(extracted, source)
                self.assertEqual((extracted/names[0]).read_bytes(),(source/names[0]).read_bytes())
                with zipfile.ZipFile(source/'spw_kaggle_source.zip','w') as archive:
                    for name in names+['spw_code_attack/source_manifest.json']:
                        archive.write(source/name,name)
                # Remove extracted source; force unique mount discovery and ZIP path.
                import shutil
                shutil.rmtree(source/'shared');shutil.rmtree(source/'spw_code_attack')
                zipped = work/'zipped';run(zipped, inputs/'missing')
                self.assertTrue((zipped/names[0]).is_file())
                (zipped/names[0]).write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    run(zipped, source)
                with zipfile.ZipFile(source/'spw_kaggle_source.zip','a') as archive:
                    archive.writestr('../outside','unsafe')
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    run(work/'unsafe', source)


if __name__ == '__main__':
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from shared import kaggle_setup as setup


LOCK = {
    'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128',
    'torchaudio': '2.11.0+cu128', 'triton': '3.6.0',
    'transformers': '4.52.4', 'dill': '0.4.0', 'numpy': '2.1.3',
}
BASELINE_LOCK = {
    'transformers': '4.52.4', 'tokenizers': '0.21.4', 'huggingface-hub': '0.30.2',
    'accelerate': '1.6.0', 'datasets': '4.6.1', 'dill': '0.4.0',
    'multiprocess': '0.70.18', 'fsspec': '2025.12.0', 'numpy': '2.1.3',
    'pandas': '2.3.3', 'pyarrow': '23.0.1', 'scikit-learn': '1.6.1',
    'matplotlib': '3.10.0', 'flake8': '7.3.0', 'pyenchant': '3.3.0',
    'tree-sitter': '0.25.2', 'tree-sitter-go': '0.25.0', 'tree-sitter-java': '0.23.5',
    'tree-sitter-python': '0.25.0', 'tree-sitter-ruby': '0.23.1', 'tree-sitter-rust': '0.24.0',
    'psutil': '5.9.5', 'torch': '2.11.0+cu128', 'contourpy': '1.3.3',
    'h5py': '3.16.0', 'kiwisolver': '1.5.1', 'llvmlite': '0.44.0',
    'numba': '0.61.2', 'pillow': '13.0.0', 'scipy': '1.16.3',
    'torchaudio': '2.11.0+cu128', 'torchvision': '0.26.0+cu128', 'triton': '3.6.0',
}


class KaggleSetupTests(unittest.TestCase):
    def reqfile(self, directory, entries=None):
        path = Path(directory) / 'requirements.txt'
        path.write_text('\n'.join(f'{k}=={v}' for k, v in (entries or LOCK).items()) + '\n')
        return path

    def patch_world(self, versions, loaded=(), pip_side_effect=None):
        def snapshot():
            return dict(versions)

        def run(command, **kwargs):
            if pip_side_effect:
                pip_side_effect(command, versions)
            return SimpleNamespace(returncode=0, stdout='mocked pip\n')

        return [patch.object(setup, 'installed_version', side_effect=lambda n: versions.get(setup.normalize(n))),
                patch.object(setup, 'distribution_snapshot', side_effect=snapshot),
                patch.object(setup, 'loaded_distribution_names', return_value=set(loaded)),
                patch.object(setup, 'stale_modules', return_value=[]),
                patch.object(setup.subprocess, 'run', side_effect=run)]

    def enter(self, stack):
        self.patch_mocks = [item.start() for item in stack]
        self.addCleanup(lambda: [item.stop() for item in reversed(stack)])

    def test_fresh_environment_installs_only_missing_pins(self):
        with tempfile.TemporaryDirectory() as d:
            versions = {'torch': LOCK['torch'], 'numpy': LOCK['numpy']}
            def install(command, found):
                for spec in command:
                    if '==' in str(spec) and str(spec).split('==', 1)[0] in LOCK:
                        name, value = str(spec).split('==', 1)
                        found[name] = value
            stack = self.patch_world(versions, pip_side_effect=install)
            self.enter(stack)
            report = setup.install_dependencies(self.reqfile(d), d)
            argv = self.patch_mocks[-1].call_args.args[0]
            self.assertIn('--no-deps', argv)
            self.assertEqual(set(report['changed_packages']), set(LOCK) - {'torch', 'numpy'})
            self.assertFalse(report['restart_required'])

    def test_correct_environment_is_noop_on_repeated_calls(self):
        with tempfile.TemporaryDirectory() as d:
            versions = dict(LOCK)
            stack = self.patch_world(versions)
            self.enter(stack)
            first = setup.install_dependencies(self.reqfile(d), d)
            second = setup.install_dependencies(self.reqfile(d), d)
            self.assertFalse(first['pip_called'])
            self.assertFalse(second['pip_called'])
            self.patch_mocks[-1].assert_not_called()

    def test_transformers_fix_does_not_touch_correct_cuda_torch(self):
        with tempfile.TemporaryDirectory() as d:
            versions = dict(LOCK, transformers='5.16.1')
            stack = self.patch_world(versions, pip_side_effect=lambda c, v: v.update(transformers='4.52.4'))
            self.enter(stack)
            setup.install_dependencies(self.reqfile(d), d)
            argv = self.patch_mocks[-1].call_args.args[0]
            self.assertIn('transformers==4.52.4', argv)
            self.assertNotIn('torch==2.11.0+cu128', argv)
            self.assertNotIn(setup.PYTORCH_CUDA_INDEX, argv)

    def test_cpu_torch_is_replaced_with_exact_cuda_build(self):
        with tempfile.TemporaryDirectory() as d:
            versions = dict(LOCK, torch='2.11.0+cpu', torchvision='0.26.0+cpu', torchaudio='2.11.0+cpu')
            def install(command, found):
                found.update({k: LOCK[k] for k in ('torch', 'torchvision', 'torchaudio')})
            stack = self.patch_world(versions, pip_side_effect=install)
            self.enter(stack)
            setup.install_dependencies(self.reqfile(d), d)
            argv = self.patch_mocks[-1].call_args.args[0]
            self.assertIn(setup.PYTORCH_CUDA_INDEX, argv)
            self.assertIn('torch==2.11.0+cu128', argv)
            self.assertIn('torchvision==0.26.0+cu128', argv)
            self.assertIn('torchaudio==2.11.0+cu128', argv)

    def test_loaded_correct_dill_is_never_reinstalled(self):
        with tempfile.TemporaryDirectory() as d:
            versions = dict(LOCK)
            stack = self.patch_world(versions, loaded={'dill'})
            self.enter(stack)
            setup.install_dependencies(self.reqfile(d), d)
            self.patch_mocks[-1].assert_not_called()

    def test_loaded_wrong_dill_requires_one_restart_then_no_install(self):
        with tempfile.TemporaryDirectory() as d:
            versions = dict(LOCK, dill='0.3.8')
            def install(command, found): found['dill'] = '0.4.0'
            stack = self.patch_world(versions, loaded={'dill'}, pip_side_effect=install)
            self.enter(stack)
            with self.assertRaisesRegex(RuntimeError, 'Restart the Kaggle kernel ONCE'):
                setup.install_dependencies(self.reqfile(d), d)
            self.assertEqual(json.loads((Path(d)/'sersem_installation.json').read_text())['changed_loaded_packages'], ['dill'])
            self.patch_mocks[-1].assert_called_once()
            # Simulate a new kernel: the exact wheel is present, so setup is a no-op.
            setup.stale_modules = lambda reqs=None: []
            setup.loaded_distribution_names = lambda: {'dill'}
            report = setup.install_dependencies(self.reqfile(d), d)
            self.assertFalse(report['pip_called'])
            self.patch_mocks[-1].assert_called_once()

    def test_loaded_wrong_dill_without_restart_never_reinstalls(self):
        with tempfile.TemporaryDirectory() as d:
            versions = dict(LOCK, dill='0.4.0')
            stack = self.patch_world(versions, loaded={'dill'})
            self.enter(stack)
            setup.stale_modules = lambda reqs=None: ['dill: loaded 0.3.8, installed 0.4.0']
            with self.assertRaisesRegex(RuntimeError, 'prior package change'):
                setup.install_dependencies(self.reqfile(d), d)
            self.patch_mocks[-1].assert_not_called()

    def test_conflicting_duplicate_exact_pins_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.reqfile(d, {'torch': '2.11.0+cu128'})
            path.write_text(path.read_text() + 'torch==2.11.0+cpu\n')
            with self.assertRaisesRegex(ValueError, 'conflicting exact pins'):
                setup.install_dependencies(path, d)

    def test_local_cuda_version_suffix_is_part_of_exact_pin(self):
        with tempfile.TemporaryDirectory() as d:
            reqs = setup.active_requirements(self.reqfile(d, {'torch': '2.11.0+cu128'}))
            self.assertEqual(setup.validate_requirement_versions(reqs, lambda _: '2.11.0+cu128')['torch'],
                             '2.11.0+cu128')
            with self.assertRaisesRegex(RuntimeError, r'installed 2\.11\.0\+cpu'):
                setup.validate_requirement_versions(reqs, lambda _: '2.11.0+cpu')

    def test_exact_saved_baseline_lock_has_all_33_versions(self):
        root = Path(__file__).resolve().parents[1]
        reqs = setup.active_requirements(root/'shared/requirements-kaggle.txt')
        saved_config = root/'Results/Sersem_Baseline/config.json'
        expected = BASELINE_LOCK
        if saved_config.is_file():
            expected = json.loads(saved_config.read_text())['versions']
        self.assertEqual({setup.normalize(r.name): next(iter(r.specifier)).version for r in reqs},
                         {setup.normalize(k): v for k, v in expected.items()})
        infilling_requirements = root/'infilling_code_attack/requirements-kaggle.txt'
        if infilling_requirements.is_file():
            self.assertEqual(infilling_requirements.read_text(), (root/'shared/requirements-kaggle.txt').read_text())

    def test_changed_loaded_dependency_reports_restart_and_prevents_pip_loop(self):
        with tempfile.TemporaryDirectory() as d:
            versions = dict(LOCK, transformers='5.16.1')
            def install(command, found): found['transformers'] = '4.52.4'
            stack = self.patch_world(versions, loaded={'transformers'}, pip_side_effect=install)
            self.enter(stack)
            with self.assertRaisesRegex(RuntimeError, 'Restart the Kaggle kernel ONCE'):
                setup.install_dependencies(self.reqfile(d), d)
            setup.stale_modules = lambda reqs=None: ['transformers: loaded 5.16.1, installed 4.52.4']
            with self.assertRaisesRegex(RuntimeError, 'prior package change'):
                setup.install_dependencies(self.reqfile(d), d)
            self.patch_mocks[-1].assert_called_once()


if __name__ == '__main__':
    unittest.main()

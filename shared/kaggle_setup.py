"""Pinned, idempotent Kaggle setup shared by SERSEM-derived experiments.

The requirements file is the saved SERSEM environment lock.  Installation is
metadata-driven and never uses pip's resolver to replace already-validated
transitive packages.  A loaded module that is actually changed requires one
kernel restart; a subsequent fresh-kernel call sees the satisfied lock and
does not install again.
"""
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

MODULE_NAMES = {
    'huggingface-hub': 'huggingface_hub', 'scikit-learn': 'sklearn',
    'pyenchant': 'enchant', 'tree-sitter': 'tree_sitter',
    'torchvision': 'torchvision', 'torchaudio': 'torchaudio',
}
EXTRA_MODULE_NAMES = {
    'numpy': 'numpy', 'pandas': 'pandas', 'matplotlib': 'matplotlib',
    'scipy': 'scipy', 'pyarrow': 'pyarrow', 'torch': 'torch',
    'transformers': 'transformers', 'datasets': 'datasets',
    'tokenizers': 'tokenizers', 'dill': 'dill', 'multiprocess': 'multiprocess',
    'fsspec': 'fsspec', 'triton': 'triton', 'torchaudio': 'torchaudio',
    'torchvision': 'torchvision', 'huggingface-hub': 'huggingface_hub',
}
PYTORCH_CUDA_INDEX = 'https://download.pytorch.org/whl/cu128'


def requirements(path):
    from packaging.requirements import Requirement
    return [Requirement(line.strip()) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def normalize(name):
    from packaging.utils import canonicalize_name
    return canonicalize_name(name)


def active_requirements(path):
    return [r for r in requirements(path) if r.marker is None or r.marker.evaluate()]


def validate_requirement_pins(reqs):
    """Reject duplicate package entries unless their specifiers agree."""
    from packaging.specifiers import SpecifierSet
    grouped = {}
    for req in reqs:
        grouped.setdefault(normalize(req.name), []).append(req)
    conflicts = []
    for name, entries in grouped.items():
        if len(entries) < 2:
            continue
        pins = set()
        for entry in entries:
            for spec in entry.specifier:
                if spec.operator in {'==', '==='} and '*' not in spec.version:
                    pins.add(spec.version)
        if len(pins) > 1:
            conflicts.append(f'{name}: conflicting exact pins {sorted(pins)}')
            continue
        if pins:
            pin = next(iter(pins))
            if not all(SpecifierSet(str(e.specifier)).contains(pin, prereleases=True) for e in entries):
                conflicts.append(f'{name}: exact pin {pin} violates another requirement')
        elif len({str(e.specifier) for e in entries}) > 1:
            # The lock file is intentionally canonical: repeated entries are
            # almost always a packaging error, and resolving arbitrary PEP 440
            # set intersections here would hide that error.
            conflicts.append(f'{name}: duplicate nonidentical constraints')
    if conflicts:
        raise ValueError('Conflicting requirements: ' + '; '.join(conflicts))


def installed_version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def validate_requirement_versions(reqs, lookup=None):
    """Validate and return every pinned distribution with PEP 440 semantics."""
    lookup = lookup or installed_version
    versions = {}
    failures = []
    for req in reqs:
        actual = lookup(req.name)
        if actual is None or not req.specifier.contains(actual, prereleases=True):
            failures.append(f'{req}: installed {actual}')
        else:
            versions[normalize(req.name)] = actual
    if failures:
        raise RuntimeError('Environment differs from the pinned SERSEM lock: ' + '; '.join(failures))
    return versions


def distribution_snapshot():
    """Use the same selected metadata lookup as installed_version()."""
    names = {normalize(d.metadata['Name']) for d in metadata.distributions()
             if d.metadata.get('Name')}
    return {name: metadata.version(name) for name in sorted(names)}


def loaded_distribution_names():
    """Return distributions whose import roots are already in this process."""
    package_modules = metadata.packages_distributions()
    loaded_roots = {name.split('.')[0] for name in sys.modules}
    loaded = {normalize(dist) for root in loaded_roots
              for dist in package_modules.get(root, [])}
    for package, module_name in EXTRA_MODULE_NAMES.items():
        if module_name in loaded_roots:
            loaded.add(normalize(package))
    return loaded


def stale_modules(reqs=None):
    """Find loaded code whose on-disk distribution was replaced in this kernel."""
    from packaging.version import Version
    names = set(EXTRA_MODULE_NAMES)
    if reqs is not None:
        names.update(normalize(r.name) for r in reqs)
    stale = []
    for package in sorted(names):
        module_name = EXTRA_MODULE_NAMES.get(package, MODULE_NAMES.get(package, package.replace('-', '_')))
        module = sys.modules.get(module_name)
        loaded = getattr(module, '__version__', None) if module is not None else None
        disk = installed_version(package)
        if loaded is not None and disk is not None and Version(str(loaded)) != Version(str(disk)):
            stale.append(f'{module_name}: loaded {loaded}, installed {disk}')
    return stale


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def install_dependencies(requirements_path, work_root):
    """Install only missing/wrong lock entries; restart only after loaded code changes."""
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        raise RuntimeError('This setup targets Python 3.11–3.13; use a supported Kaggle runtime.')
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    reqs = active_requirements(requirements_path)
    validate_requirement_pins(reqs)

    stale = stale_modules(reqs)
    if stale:
        raise RuntimeError('A prior package change is still loaded. Restart the Kaggle kernel before setup: '
                           + '; '.join(stale))

    before = distribution_snapshot()
    pending = [req for req in reqs if (installed_version(req.name) is None
               or not req.specifier.contains(installed_version(req.name), prereleases=True))]
    loaded = loaded_distribution_names()
    install_log = Path(work_root) / 'sersem_pip_install.log'
    if pending:
        # no-deps is safe here because every locked direct/transitive package
        # is checked explicitly below. It prevents a small Transformers fix
        # from replacing a correct CUDA wheel or an unrelated Kaggle package.
        command = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                   '--no-deps', '--only-binary',
                   'numpy,pandas,matplotlib,scipy,scikit-learn,pyarrow,tokenizers']
        if any(normalize(req.name) in {'torch', 'torchvision', 'torchaudio'} for req in pending):
            command += ['--extra-index-url', PYTORCH_CUDA_INDEX]
        command += [str(req) for req in pending]
        print('Installing only unsatisfied SERSEM lock entries:', ', '.join(map(str, pending)))
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        install_log.write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(f'Pinned dependency installation failed. Full log: {install_log}\n'
                               + process.stdout[-12000:])
        print(process.stdout)
    else:
        print('All pinned requirements are satisfied; pip was not called.')

    after = distribution_snapshot()
    try:
        exact_versions = validate_requirement_versions(reqs)
    except RuntimeError as exc:
        raise RuntimeError(str(exc) + f'. Inspect {install_log}') from exc
    changed = sorted(name for name in before.keys() | after.keys()
                     if before.get(name) != after.get(name))
    changed_loaded = sorted(set(changed) & loaded)
    report = {'python': sys.version, 'requirements': exact_versions,
              'changed_packages': changed, 'changed_loaded_packages': changed_loaded,
              'restart_required': bool(changed_loaded), 'pip_called': bool(pending)}
    _write_json(Path(work_root) / 'sersem_installation.json', report)
    if changed_loaded:
        raise RuntimeError('Installation changed already-loaded packages. Restart the Kaggle kernel ONCE, '
                           'then rerun the setup cell. On the fresh kernel all satisfied pins are skipped. '
                           'Changed modules: ' + ', '.join(changed_loaded))
    return report


def validate_environment(requirements_path, require_cuda=False):
    """Validate exact pinned versions and import APIs after dependency setup."""
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    reqs = active_requirements(requirements_path)
    validate_requirement_pins(reqs)
    stale = stale_modules(reqs)
    if stale:
        raise RuntimeError('Restart the Kaggle kernel before validation: ' + '; '.join(stale))
    versions = validate_requirement_versions(reqs)
    try:
        for name in ['numpy', 'pandas', 'matplotlib', 'pyarrow', 'sklearn', 'datasets',
                     'transformers', 'tokenizers', 'huggingface_hub', 'torch']:
            importlib.import_module(name)
        from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError('Dependency preflight failed before data/model loading. If packages were changed, '
                           'restart the kernel; otherwise inspect the chained import error.') from exc
    if require_cuda:
        import torch
        if torch.version.cuda != '12.8' or not torch.cuda.is_available():
            raise RuntimeError(f'ReCaLL requires the saved CUDA build torch 2.11.0+cu128 and a visible Kaggle GPU; '
                               f'loaded torch={torch.__version__}, CUDA runtime={torch.version.cuda}, '
                               f'cuda_available={torch.cuda.is_available()}. Select Tesla T4 and rerun setup.')
    return {'python': sys.version.split()[0], 'versions': versions,
            'cuda_runtime': _cuda_runtime(), 'cuda_available': _cuda_available()}


def _cuda_runtime():
    torch = sys.modules.get('torch')
    return getattr(getattr(torch, 'version', None), 'cuda', None)


def _cuda_available():
    torch = sys.modules.get('torch')
    return bool(torch is not None and torch.cuda.is_available())

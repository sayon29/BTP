"""Kaggle dependency setup; no model construction, inference or attack execution.
Scientific/CUDA packages already in Kaggle are constrained to installed versions.
"""
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

# Constrain existing numerical, compiled-extension and accelerator distributions,
# including transitive dependencies. This prevents pip replacing preloaded binaries.
PROTECTED = {
    'numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn', 'pyarrow',
    'torch', 'torchvision', 'torchaudio', 'triton', 'numba', 'llvmlite',
    'contourpy', 'kiwisolver', 'pillow', 'h5py',
}
MODULE_NAMES = {
    'huggingface-hub': 'huggingface_hub', 'scikit-learn': 'sklearn',
    'pyenchant': 'enchant', 'tree-sitter': 'tree_sitter',
}


def requirements(path):
    from packaging.requirements import Requirement
    return [Requirement(line.strip()) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def installed_version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def normalize(name):
    from packaging.utils import canonicalize_name
    return canonicalize_name(name)


def active_requirements(path):
    return [r for r in requirements(path) if r.marker is None or r.marker.evaluate()]


def stale_modules():
    stale = []
    for package, module_name in {'numpy': 'numpy', 'pandas': 'pandas',
                                 'matplotlib': 'matplotlib', 'scipy': 'scipy',
                                 'pyarrow': 'pyarrow', 'scikit-learn': 'sklearn',
                                 'torch': 'torch', 'transformers': 'transformers',
                                 'datasets': 'datasets', 'tokenizers': 'tokenizers',
                                 'huggingface-hub': 'huggingface_hub'}.items():
        module = sys.modules.get(module_name)
        loaded = getattr(module, '__version__', None) if module is not None else None
        disk = installed_version(package)
        if loaded is not None and disk is not None:
            from packaging.version import Version
            # Torch can expose a CUDA local suffix absent in distribution metadata.
            differs = Version(str(loaded)).public != Version(disk).public if package == 'torch' else Version(str(loaded)) != Version(disk)
            if differs:
                stale.append(f'{module_name}: loaded {loaded}, installed {disk}')
    return stale


def distribution_snapshot():
    """Select each distribution consistently with metadata.version(name).

    Kaggle may retain duplicate dist-info directories. Taking the last entry
    from metadata.distributions() can freeze an inactive NumPy 1.26 while the
    active lookup and imported module report NumPy 2.x. Never use iteration
    order to choose a version.
    """
    names = {normalize(d.metadata['Name']) for d in metadata.distributions() if d.metadata.get('Name')}
    return {name: metadata.version(name) for name in sorted(names)}


def distribution_diagnostics(names):
    """Metadata only; no scientific package imports or model execution."""
    names = set(names)
    result = {}
    for name in sorted(names):
        selected = metadata.distribution(name)
        result[name] = {'selected_version': selected.version,
                        'selected_metadata_path': str(getattr(selected, '_path', selected.locate_file(''))),
                        'discovered_metadata': []}
    for dist in metadata.distributions():
        name = dist.metadata.get('Name')
        if name and normalize(name) in names:
            result[normalize(name)]['discovered_metadata'].append(
                {'version': dist.version, 'path': str(getattr(dist, '_path', dist.locate_file('')))})
    return result


def install_dependencies(requirements_path, work_root):
    """Idempotent installer. Called manually on Kaggle, never by local QA."""
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        raise RuntimeError('This setup targets Python 3.11–3.13; use a supported Kaggle runtime.')
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    stale = stale_modules()
    if stale:
        raise RuntimeError('Restart the Kaggle kernel before setup: ' + '; '.join(stale))
    reqs = active_requirements(requirements_path)
    pending = [r for r in reqs if installed_version(r.name) is None
               or not r.specifier.contains(installed_version(r.name), prereleases=True)]
    conflicts = [str(r) + ' (installed ' + str(installed_version(r.name)) + ')'
                 for r in pending if normalize(r.name) in PROTECTED and installed_version(r.name) is not None]
    if conflicts:
        raise RuntimeError('Kaggle scientific stack is outside supported ranges; it will NOT be replaced in this kernel. '
                           'Use a fresh compatible Kaggle image. Conflicts: ' + ', '.join(conflicts))
    # Use the same active distribution lookup as pending/conflict checks.
    before = distribution_snapshot()
    protected_versions = {name: version for name, version in before.items()
                          if name in PROTECTED or name.startswith('nvidia-')}
    diagnostics = distribution_diagnostics(protected_versions)
    diagnostic_path = Path(work_root) / 'sersem_dependency_diagnostics.json'
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2))
    # Invariants are checked BEFORE pip, so a stale/inconsistent constraint
    # cannot trigger another numerical-package install/resolver loop.
    for req in reqs:
        selected = protected_versions.get(normalize(req.name))
        if selected is not None and not req.specifier.contains(selected, prereleases=True):
            raise RuntimeError(f'Contradictory generated constraint: {req.name}=={selected} vs {req}. '
                               f'No install attempted. Diagnostics: {diagnostic_path}')
    constraints = Path(work_root) / 'sersem_runtime_constraints.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(protected_versions.items())))
    print('Active NumPy constraint:', protected_versions.get('numpy', '(not installed)'))
    # Track selected distributions consistently before/after installation.
    package_modules = metadata.packages_distributions()
    loaded_roots = {name.split('.')[0] for name in sys.modules}
    loaded_distributions = {normalize(name) for root in loaded_roots for name in package_modules.get(root, [])}
    for req in reqs:
        if MODULE_NAMES.get(normalize(req.name), normalize(req.name).replace('-', '_')) in loaded_roots:
            loaded_distributions.add(normalize(req.name))
    if pending:
        print('Installing missing/mismatched dependencies:', ', '.join(str(r) for r in pending))
        # Pass the COMPLETE requirements set so the resolver respects satisfied
        # direct pins as well as pending ones. No --upgrade / --force-reinstall.
        command = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                   '--only-binary', 'numpy,pandas,matplotlib,scipy,scikit-learn,pyarrow,tokenizers',
                   '-c', str(constraints), '-r', str(requirements_path)]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_path = Path(work_root) / 'sersem_pip_install.log'
        log_path.write_text(process.stdout)
        if process.returncode:
            # Preserve the actual resolver explanation, not just CalledProcessError.
            raise RuntimeError(f'pip failed. Full log: {log_path}; selected/duplicate package metadata: '
                               f'{diagnostic_path}.\n' + process.stdout[-12000:])
        print(process.stdout)

    else:
        print('All Python requirements already satisfied; skipping pip.')
    after = distribution_snapshot()
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    modified_protected = [name for name, version in protected_versions.items() if after.get(name) != version]
    if modified_protected:
        raise RuntimeError('Protected package changed unexpectedly: ' + ', '.join(modified_protected))
    report = {'python': sys.version, 'protected_versions': protected_versions,
              'changed_packages': changed, 'restart_required': bool(set(changed) & loaded_distributions)}
    (Path(work_root) / 'sersem_installation.json').write_text(json.dumps(report, indent=2))
    if report['restart_required']:
        raise RuntimeError('Installation finished. Restart the Kaggle kernel ONCE, then rerun setup in order. '
                           'Already-loaded packages changed: ' + ', '.join(sorted(set(changed) & loaded_distributions)))
    return report


def validate_environment(requirements_path):
    """Version/import preflight only; no model instance or network request."""
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    stale = stale_modules()
    if stale:
        raise RuntimeError('Restart the Kaggle kernel: ' + '; '.join(stale))
    versions = {}
    for req in active_requirements(requirements_path):
        actual = installed_version(req.name)
        if actual is None or not req.specifier.contains(actual, prereleases=True):
            raise RuntimeError(f'{req}: installed {actual}. Run dependency setup before initialization.')
        versions[normalize(req.name)] = actual
    try:
        for name in ['numpy', 'pandas', 'matplotlib', 'pyarrow', 'sklearn', 'datasets',
                     'transformers', 'tokenizers', 'huggingface_hub']:
            importlib.import_module(name)
        from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError('Dependency preflight failed before data/model loading. If packages were changed, '
                           'restart the kernel; otherwise inspect the chained import error.') from exc
    versions['torch'] = metadata.version('torch')
    for name in sorted(PROTECTED - set(versions)):
        actual = installed_version(name)
        if actual is not None:
            versions[name] = actual
    return {'python': sys.version.split()[0], 'versions': versions}

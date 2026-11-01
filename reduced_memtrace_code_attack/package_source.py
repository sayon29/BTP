"""Package existing verified helpers plus the independent Reduced memTrace source."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    shared = ['sersem_baseline.py','weighted_anomaly.py','kaggle_setup.py','resource_data.py',
        'resource_pipeline.py','research_execution.py','cache_setup.py','requirements-kaggle.txt','upstream_manifest.json']
    files = ['shared/'+name for name in shared]
    files += ['adaptive_token_attack/'+name for name in ['data.py','features.py']]
    files += [str(p.relative_to(root)) for p in sorted((root/'reduced_memtrace_code_attack').iterdir())
              if p.is_file() and p.name != 'source_manifest.json']
    upstream = json.loads((root/'shared/upstream_manifest.json').read_text())
    files += ['SERSEM-Poisoned-Chalice-Competition-2026/'+name for name in upstream['source_sha256']]
    files += ['tests/test_reduced_memtrace.py','tests/test_kaggle_setup.py']
    files = sorted(set(files))
    hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    manifest = root/'reduced_memtrace_code_attack/source_manifest.json'
    manifest.write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    notebook = root/'notebooks/08_reduced_memtrace_code.ipynb'
    nb = json.loads(notebook.read_text())
    updates = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        lines = ''.join(cell['source']).splitlines(True)
        for node in ast.parse(''.join(lines)).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_MANIFEST_SHA256' for t in node.targets):
                lines[node.lineno-1:node.end_lineno] = ['SOURCE_MANIFEST_SHA256 = '+repr(pin)+'\n']
                updates += 1
        cell['source'] = lines
    if updates != 1:
        raise ValueError('Require exactly one notebook source pin')
    notebook.write_text(json.dumps(nb,indent=1)+'\n')
    target = root/'notebooks/reduced_memtrace_kaggle_source.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files+['reduced_memtrace_code_attack/source_manifest.json']:
            archive.write(root/name,name)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Source ZIP CRC failure')
    print(target)
    print('Manifest SHA256:',pin)


if __name__ == '__main__':
    main()

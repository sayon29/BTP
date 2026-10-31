"""Portable, verified Kaggle ZIP; never rewrites any existing attack artifact."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    shared = ['sersem_baseline.py','weighted_anomaly.py','kaggle_setup.py','resource_data.py',
              'resource_pipeline.py','research_execution.py','cache_setup.py','requirements-kaggle.txt','upstream_manifest.json']
    files = ['shared/'+p for p in shared]
    files += ['adaptive_token_attack/'+p for p in ['data.py','features.py']]
    files += [str(p.relative_to(root)) for p in sorted((root/'recall_code_attack').glob('*'))
              if p.is_file() and p.name != 'source_manifest.json']
    upstream = json.loads((root/'shared/upstream_manifest.json').read_text())
    files += ['SERSEM-Poisoned-Chalice-Competition-2026/'+p for p in upstream['source_sha256']]
    files += ['tests/test_recall.py', 'tests/test_kaggle_setup.py']
    files = sorted(set(files))
    hashes = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
    manifest = root/'recall_code_attack/source_manifest.json'
    manifest.write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    notebook = root/'notebooks/04_recall_code.ipynb'
    content = json.loads(notebook.read_text()); updated=0
    for cell in content['cells']:
        if cell['cell_type']!='code':
            continue
        lines = ''.join(cell['source']).splitlines(True)
        for node in ast.parse(''.join(lines)).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_MANIFEST_SHA256' for t in node.targets):
                lines[node.lineno-1:node.end_lineno] = ['SOURCE_MANIFEST_SHA256 = '+repr(digest)+'\n']; updated+=1
        cell['source'] = lines
    if updated != 1:
        raise ValueError('Require one notebook manifest pin')
    notebook.write_text(json.dumps(content,indent=1)+'\n')
    target = root/'notebooks/recall_kaggle_source.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files+['recall_code_attack/source_manifest.json']:
            archive.write(root/name,name)
    print(target)
    print('Source manifest SHA256:',digest)


if __name__=='__main__':
    main()

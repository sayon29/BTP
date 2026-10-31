"""Build only adaptive sources; never rewrite baseline/neighbourhood artifacts."""
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
    files += [str(p.relative_to(root)) for p in sorted((root/'adaptive_token_attack').glob('*.py'))]
    files += ['adaptive_token_attack/README.md']
    upstream = json.loads((root/'shared/upstream_manifest.json').read_text())
    files += ['SERSEM-Poisoned-Chalice-Competition-2026/'+name for name in upstream['source_sha256']]
    hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    source = root/'adaptive_token_attack/source_manifest.json'
    source.write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    notebook = root/'notebooks/03_adaptive_sersem.ipynb'
    content = json.loads(notebook.read_text()); updated=0
    for cell in content['cells']:
        if cell['cell_type'] != 'code':
            continue
        lines = ''.join(cell['source']).splitlines(True)
        for node in ast.parse(''.join(lines)).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_MANIFEST_SHA256' for t in node.targets):
                lines[node.lineno-1:node.end_lineno] = ['SOURCE_MANIFEST_SHA256 = '+repr(digest)+'\n'];updated+=1
        cell['source'] = lines
    if updated != 1:
        raise ValueError('Require one source manifest fingerprint in adaptive notebook')
    notebook.write_text(json.dumps(content,indent=1)+'\n')
    target=root/'notebooks/adaptive_kaggle_source.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files+['adaptive_token_attack/source_manifest.json']:
            archive.write(root/name,name)
    print(target)


if __name__ == '__main__':
    main()

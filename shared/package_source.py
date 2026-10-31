"""Build allowlisted research source ZIPs; exclude weights/data/history/secrets."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
shared_names=['sersem_baseline.py','weighted_anomaly.py','kaggle_setup.py','resource_data.py','resource_pipeline.py','research_execution.py','cache_setup.py','requirements-kaggle.txt','upstream_manifest.json']
fingerprints={name:hashlib.sha256((root/'shared'/name).read_bytes()).hexdigest() for name in shared_names}
neighbourprints={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['neighbourhood_attack/runner.py','neighbourhood_attack/perturb.py']}
for notebook_path in [root/'notebooks/01_sersem_baseline.ipynb',root/'notebooks/02_sersem_neighbourhood.ipynb']:
    notebook=json.loads(notebook_path.read_text());updated=0
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPECTED_FRAMEWORK_SHA256' for t in node.targets):
                lines=source.splitlines(True)
                lines[node.lineno-1:node.end_lineno]=['EXPECTED_FRAMEWORK_SHA256 = '+repr(fingerprints)+'\n']
                source=''.join(lines);updated+=1
        if 'sys.path.insert(0, str(PROJECT_ROOT))' in source:
            marker='# Verify standalone neighbourhood source as well.\n'
            if marker in source:source=source[:source.index(marker)]+source[source.index('# End neighbourhood source verification.\n')+len('# End neighbourhood source verification.\n'):]
            extra=marker+'EXPECTED_NEIGHBOURHOOD_SHA256 = '+repr(neighbourprints)+'\n'
            extra+='for relative_name, expected_hash in EXPECTED_NEIGHBOURHOOD_SHA256.items():\n    if hashlib.sha256((PROJECT_ROOT / relative_name).read_bytes()).hexdigest() != expected_hash:\n        raise RuntimeError("Outdated neighbourhood helper; use refreshed source and a fresh PROJECT_ROOT")\n# End neighbourhood source verification.\n'
            source=source.replace('sys.path.insert(0, str(PROJECT_ROOT))',extra+'sys.path.insert(0, str(PROJECT_ROOT))')
        cell['source']=source.splitlines(True)
    if updated!=1:raise ValueError('Expected one framework fingerprint declaration: '+str(notebook_path))
    notebook_path.write_text(json.dumps(notebook,indent=1)+'\n')
manifest=json.loads((root/'shared/upstream_manifest.json').read_text())
files=['README.md','AUDIT.md','KAGGLE_RESOURCES.md','RESEARCH_RUN.md','neighbourhood_attack/README.md']
files+=['shared/'+name for name in shared_names]
files+=['shared/author_source_search.json']+list(neighbourprints)
files+=['SERSEM-Poisoned-Chalice-Competition-2026/'+name for name in manifest['source_sha256']]
for target in [root/'notebooks/sersem_kaggle_source.zip',root/'notebooks/neighbourhood_kaggle_source.zip']:
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:archive.write(root/name,name)
    print(target)

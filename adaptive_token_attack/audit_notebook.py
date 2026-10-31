"""Static audit only; no notebook execution, data access or target imports."""
import ast
import hashlib
import json
from pathlib import Path
import runpy


def main():
    root=Path(__file__).resolve().parents[1]
    audit=runpy.run_path(str(root/'shared/audit_notebook.py'))['audit_block']
    notebook=root/'notebooks/03_adaptive_sersem.ipynb'
    content=json.loads(notebook.read_text());defined=set();cells=[]
    ids=[c['id'] for c in content['cells']]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate notebook cell IDs')
    for index,cell in enumerate(content['cells']):
        if cell['cell_type']!='code':
            continue
        if cell['execution_count'] is not None or cell['outputs']:
            raise ValueError('Adaptive notebook must have no saved execution/output')
        tree=ast.parse(''.join(cell['source']))
        defined=audit(tree.body,defined);cells.append(index)
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_MANIFEST_SHA256' for t in node.targets):
                expected=ast.literal_eval(node.value)
                actual=hashlib.sha256((root/'adaptive_token_attack/source_manifest.json').read_bytes()).hexdigest()
                if actual!=expected:
                    raise ValueError('Rebuild adaptive source package before notebook audit')
    source=json.loads((root/'adaptive_token_attack/source_manifest.json').read_text())
    for name,expected in source.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Package source changed; rebuild: '+name)
    print(f'Adaptive notebook static audit passed: {len(cells)} code cells, sequential bindings, empty outputs and source fingerprints.')


if __name__=='__main__':main()

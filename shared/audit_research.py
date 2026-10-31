"""Static syntax/config/binding/API audit; NEVER imports attack dependencies."""
import ast
import json
from pathlib import Path
from audit_notebook import audit_block, configured_paths

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/02_sersem_neighbourhood.ipynb'
nb=json.loads(path.read_text());defined=set();cells=[];trees=[]
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code':continue
    assert cell['execution_count'] is None and not cell['outputs']
    tree=ast.parse(''.join(cell['source']),filename=f'{path.name}:{i}')
    defined=audit_block(tree.body,defined);trees.append(tree)
    cells.append({'index':i,'id':cell['id']})
configured=set()
for tree in trees:
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cfg' for t in node.targets):
            configured |= configured_paths(node.value)
required=set()
for name in ['neighbourhood_attack/runner.py','neighbourhood_attack/perturb.py']:
    tree=ast.parse((ROOT/name).read_text(),filename=name)
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):
            owner=node.value
            if isinstance(owner,ast.Name) and owner.id=='cfg' or isinstance(owner,ast.Attribute) and isinstance(owner.value,ast.Name) and owner.value.id=='self' and owner.attr=='cfg':
                required.add((node.slice.value,))
missing=required-configured
if missing:raise ValueError(f'Missing neighbourhood cfg keys: {missing}')
exports={}
for directory in ['shared','neighbourhood_attack']:
    for helper in (ROOT/directory).glob('*.py'):
        tree=ast.parse(helper.read_text(),filename=str(helper));module=directory+'.'+helper.stem;functions={}
        for node in tree.body:
            fn=node if isinstance(node,ast.FunctionDef) else next((n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None) if isinstance(node,ast.ClassDef) else None
            if fn is not None:
                params=[a.arg for a in fn.args.posonlyargs+fn.args.args]
                if isinstance(node,ast.ClassDef):params=params[1:]
                functions[node.name]=(params,len(params)-len(fn.args.defaults))
        exports[module]=functions
imports={};calls=[]
for tree in trees:
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module in exports:
            for item in node.names:
                if item.name not in exports[node.module]:raise ValueError('Missing helper export '+item.name)
                imports[item.asname or item.name]=exports[node.module][item.name]
for tree in trees:
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in imports:
            params,minimum=imports[node.func.id];count=len(node.args)+len(node.keywords)
            if not minimum<=count<=len(params) or any(k.arg not in params for k in node.keywords):raise ValueError('Helper arity mismatch '+node.func.id)
            calls.append(node.func.id)
report=dict(status='static checks passed; no attacks/notebook cells executed',cells=cells,
            required_cfg_fields=sorted('.'.join(p) for p in required),helper_calls_checked=calls,
            limitations='Does not validate CUDA, wheels, grammar behavior, perturbation safety/success, scores, memory or throughput.')
path.with_suffix('.static_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Static neighbourhood audit passed: {len(cells)} code cells, {len(required)} cfg keys, {len(calls)} helper calls.')

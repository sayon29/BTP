"""Static notebook audit only. Parses source; never imports/executes attack code."""
import ast
import builtins
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/01_sersem_baseline.ipynb'


def referenced_globals(node):
    # AST scopes work for both Python 3.12's inlined comprehensions and older
    # interpreters; comprehension targets must not become notebook globals.
    names = set()
    def visit(value, bound):
        if isinstance(value, ast.Name):
            if isinstance(value.ctx, ast.Load) and value.id not in bound:
                names.add(value.id)
        elif isinstance(value, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            local = set(bound)
            for gen in value.generators:
                visit(gen.iter, local)
                local |= target_names(gen.target)
                for condition in gen.ifs:
                    visit(condition, local)
            for item in ([value.key, value.value] if isinstance(value, ast.DictComp) else [value.elt]):
                visit(item, local)
        elif isinstance(value, ast.Lambda):
            for default in value.args.defaults + [x for x in value.args.kw_defaults if x is not None]:
                visit(default, bound)
            params = {a.arg for a in value.args.posonlyargs + value.args.args + value.args.kwonlyargs}
            if value.args.vararg:
                params.add(value.args.vararg.arg)
            if value.args.kwarg:
                params.add(value.args.kwarg.arg)
            visit(value.body, bound | params)
        else:
            for child in ast.iter_child_nodes(value):
                visit(child, bound)
    visit(node, set())
    return names


def target_names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def audit_block(body, defined):
    defined = set(defined)
    def require(node):
        if node is None:
            return
        missing = referenced_globals(node) - defined - set(dir(builtins))
        if missing:
            raise ValueError(f'Line {node.lineno}: names used before definition: {sorted(missing)}')
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for item in stmt.names:
                defined.add(item.asname or (item.name.split('.')[0] if isinstance(stmt, ast.Import) else item.name))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            require(stmt.value)
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, (ast.Subscript, ast.Attribute)):
                    require(target)
                defined |= target_names(target)
        elif isinstance(stmt, ast.AugAssign):
            require(stmt.target); require(stmt.value)
            defined |= target_names(stmt.target)
        elif isinstance(stmt, ast.If):
            require(stmt.test)
            yes = audit_block(stmt.body, defined)
            no = audit_block(stmt.orelse, defined)
            if yes is None and no is None:
                return None
            defined = no if yes is None else yes if no is None else yes & no
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            require(stmt.iter)
            audit_block(stmt.body, defined | target_names(stmt.target))
            audit_block(stmt.orelse, defined)
            # A loop can run zero times: its new names are not definite afterward.
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            local = set(defined)
            for item in stmt.items:
                missing = referenced_globals(item.context_expr) - local - set(dir(builtins))
                if missing:
                    raise ValueError(f'Undefined with-context names: {sorted(missing)}')
                if item.optional_vars is not None:
                    local |= target_names(item.optional_vars)
            after = audit_block(stmt.body, local)
            if after is None:
                return None
            defined = after
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for value in stmt.decorator_list + stmt.args.defaults + [x for x in stmt.args.kw_defaults if x is not None]:
                require(value)
            defined.add(stmt.name)
        elif isinstance(stmt, (ast.Raise, ast.Return)):
            for value in ast.iter_child_nodes(stmt):
                require(value)
            return None
        elif isinstance(stmt, (ast.Expr, ast.Assert)):
            require(stmt)
        elif not isinstance(stmt, ast.Pass):
            raise ValueError(f'Static checker needs explicit support for {type(stmt).__name__}')
    return defined


def cfg_path(node, aliases=None):
    aliases = aliases or {}
    if isinstance(node, ast.Name):
        return () if node.id == 'cfg' else aliases.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr == 'cfg' and isinstance(node.value, ast.Name) and node.value.id == 'self':
        return ()
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        parent = cfg_path(node.value, aliases)
        if parent is not None:
            return parent + (node.slice.value,)
    return None


def configured_paths(node, prefix=()):
    paths = set()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        fields = [(k.arg, k.value) for k in node.keywords]
    elif isinstance(node, ast.Dict):
        fields = [(k.value, v) for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)]
    else:
        return paths
    for key, value in fields:
        path = prefix + (key,)
        paths.add(path)
        paths |= configured_paths(value, path)
    return paths


def main():
    nb = json.loads(NOTEBOOK.read_text())
    if nb['nbformat'] != 4 or len({c['id'] for c in nb['cells']}) != len(nb['cells']):
        raise ValueError('Invalid notebook version or duplicate cell IDs')
    defined, cell_reports, trees = set(), [], []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        if cell['execution_count'] is not None or cell['outputs']:
            raise ValueError('Deliverable must have no execution state/results')
        tree = ast.parse(''.join(cell['source']), filename=f'{NOTEBOOK.name}:cell{i}')
        previous = set(defined)
        defined = audit_block(tree.body, defined)
        if defined is None:
            raise ValueError(f'Cell {i} always terminates')
        cell_reports.append({'cell_index': i, 'cell_id': cell['id'], 'new_definite_names': sorted(defined - previous)})
        trees.append(tree)
    helper_files = ['shared/sersem_baseline.py', 'shared/weighted_anomaly.py', 'shared/kaggle_setup.py', 'shared/resource_data.py', 'shared/resource_pipeline.py', 'shared/research_execution.py', 'shared/cache_setup.py']
    required, apis, module_apis = set(), {}, {}
    for file in helper_files:
        tree = ast.parse((ROOT / file).read_text(), filename=file)
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                path = cfg_path(node.value)
                if path is not None:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            aliases[target.id] = path
        for node in ast.walk(tree):
            path = cfg_path(node, aliases)
            if path:
                required.add(path)
            # prepare_probes obtains identity via cfg[k] over a constant key list.
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.GeneratorExp)):
                for gen in node.generators:
                    if isinstance(gen.iter, (ast.List, ast.Tuple)) and isinstance(gen.target, ast.Name):
                        uses_key = any(isinstance(x, ast.Subscript) and cfg_path(x.value) == ()
                                       and isinstance(x.slice, ast.Name) and x.slice.id == gen.target.id for x in ast.walk(node))
                        if uses_key:
                            required |= {(v.value,) for v in gen.iter.elts if isinstance(v, ast.Constant) and isinstance(v.value, str)}
        exports = {}
        for node in tree.body:
            name, fn = None, None
            if isinstance(node, ast.FunctionDef):
                name, fn = node.name, node
            elif isinstance(node, ast.ClassDef):
                name = node.name
                fn = next((x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == '__init__'), None)
            if fn is not None:
                args = [a.arg for a in fn.args.posonlyargs + fn.args.args]
                if isinstance(node, ast.ClassDef):
                    args = args[1:]
                exports[name] = {'arguments': args, 'required_count': len(args) - len(fn.args.defaults)}
        module_apis[file[:-3].replace('/', '.')] = exports
    configured = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cfg' for t in node.targets):
                configured |= configured_paths(node.value)
            if isinstance(node, ast.ImportFrom) and node.module in module_apis:
                for item in node.names:
                    if item.name not in module_apis[node.module]:
                        raise ValueError(f'Missing exported helper {node.module}.{item.name}')
                    apis[item.asname or item.name] = module_apis[node.module][item.name]
    missing = required - configured
    if missing:
        raise ValueError(f'Missing configuration fields: {sorted(missing)}')
    calls = []
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in apis:
                api = apis[node.func.id]
                provided = len(node.args) + len(node.keywords)
                if any(isinstance(x, ast.Starred) for x in node.args) or not api['required_count'] <= provided <= len(api['arguments']):
                    raise ValueError(f'Helper call arity mismatch: {node.func.id}')
                if any(k.arg not in api['arguments'] for k in node.keywords):
                    raise ValueError(f'Unknown keyword for {node.func.id}')
                calls.append(node.func.id)
    report = {'status': 'static checks passed; no notebook/helper/attack execution',
              'analysis_scope': 'sequential definite bindings, comprehension scopes, helper exports/call arity, configuration field coverage',
              'code_cells': cell_reports, 'required_cfg_fields': sorted('.'.join(p) for p in required),
              'configured_cfg_fields': sorted('.'.join(p) for p in configured),
              'imported_helper_signatures': apis, 'helper_calls_checked': calls,
              'runtime_limitations': 'Does not verify third-party APIs, CUDA execution, package resolution, live dataset schemas, or scores.'}
    output = NOTEBOOK.with_suffix('.static_audit.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Static audit passed: {len(cell_reports)} code cells, {len(required)} cfg fields, {len(calls)} helper calls. Report: {output}')


if __name__ == '__main__':
    main()

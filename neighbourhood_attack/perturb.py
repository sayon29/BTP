"""Conservative code-aware MLM perturbations. No execution at import.
Comments first; Python local renames are restricted and AST-verified.
This is our standalone research attack, not recovered author SERSEM code.
"""
import ast
import hashlib
import keyword
import random
import re

PARSERS = {}
DIRECTIVE = re.compile(r'(?i)(coding\s*[:=]|encoding|noqa|nolint|type\s*:|pylint|pragma|go:|\+build|cgo|^\s*//\s*line\s|rustfmt|clippy|rubocop|frozen_string_literal|eslint|sourceMappingURL|doctest|fmt:|noinspection|@|\\)')


def parser(language):
    if language not in PARSERS:
        import importlib
        import tree_sitter
        module = importlib.import_module('tree_sitter_' + {'Go':'go','Java':'java','Python':'python','Ruby':'ruby','Rust':'rust'}[language])
        PARSERS[language] = tree_sitter.Parser(tree_sitter.Language(module.language()))
    return PARSERS[language]


def leaves_without_comments(tree, raw):
    values = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if 'comment' in node.type:
            continue
        if not node.children:
            values.append((node.type, raw[node.start_byte:node.end_byte]))
        else:
            stack.extend(reversed(node.children))
    return values


def _offset(text, lineno, byte_column):
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[:lineno - 1])) + len(lines[lineno - 1].encode('utf-8')[:byte_column].decode('utf-8'))


def sites(text, language, visible_end, allow_rename, allow_insert):
    raw = text.encode('utf-8'); tree = parser(language).parse(raw)
    if tree.root_node.has_error:
        return [], tree, 'original_syntax_error'
    result = []
    cgo = language == 'Go' and bool(re.search(r'\bimport\s*(?:\(\s*)?"C"', text))
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if 'comment' in node.type:
            value = raw[node.start_byte:node.end_byte].decode('utf-8')
            if cgo or node.start_point.row < 2 or value.startswith(('/**', '///', '//!', '#!')) or DIRECTIVE.search(value):
                continue
            base = len(raw[:node.start_byte].decode('utf-8'))
            for match in re.finditer(r'\b[A-Za-z]{4,}\b', value):
                start, end = base + match.start(), base + match.end()
                if end <= visible_end:
                    result.append(dict(kind='comment', start=start, end=end, old=match.group(), occurrences=[(start,end)]))
        else:
            stack.extend(reversed(node.children))
    if allow_insert and not cgo:
        marker = '# ' if language in {'Python','Ruby'} else '// '
        # Add only a plain comment on an existing blank line or after the file.
        # Whole noncomment syntax/token equality is verified for every proposal.
        cursor = 0
        for line_number,line in enumerate(text.splitlines(keepends=True)):
            if line_number >= 2 and not line.strip() and cursor <= visible_end:
                result.append(dict(kind='comment_insert',start=cursor,end=cursor,old='',
                                   occurrences=[(cursor,cursor)],prefix=marker,suffix='\n'))
            cursor += len(line)
        if len(text.rstrip()) <= visible_end:
            result.append(dict(kind='comment_insert',start=len(text),end=len(text),old='',
                               occurrences=[(len(text),len(text))],prefix='\n'+marker,suffix='\n'))
    if language == 'Python' and allow_rename and not re.search(r'\b(?:__code__|__dict__|locals|globals|eval|exec|inspect|frame|vars)\b',text):
        try:
            root = ast.parse(text, type_comments=True)
        except SyntaxError:
            return result, tree, None
        # No calls, attributes, imports, comprehensions, nested scopes, reflection,
        # annotations/defaults/decorators or global/nonlocal declarations.
        forbidden = (ast.Call, ast.Attribute, ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                     ast.Lambda, ast.ClassDef, ast.AsyncFunctionDef, ast.ListComp, ast.SetComp,
                     ast.DictComp, ast.GeneratorExp, ast.Yield, ast.YieldFrom, ast.NamedExpr)
        for fn in root.body:
            if not isinstance(fn, ast.FunctionDef) or fn.decorator_list or fn.returns or fn.args.defaults or any(x is not None for x in fn.args.kw_defaults):
                continue
            if any(a.annotation for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs) or fn.args.vararg or fn.args.kwarg:
                continue
            nodes = [n for statement in fn.body for n in ast.walk(statement)]
            if any(isinstance(n, forbidden + (ast.FunctionDef,)) for n in nodes):
                continue
            arguments = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
            names = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} - arguments
            for name in sorted(names):
                if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{2,}', name) or keyword.iskeyword(name) or name.startswith('__'):
                    continue
                occurrences = [(_offset(text,n.lineno,n.col_offset), _offset(text,n.end_lineno,n.end_col_offset))
                               for n in nodes if isinstance(n,ast.Name) and n.id == name]
                if occurrences and max(b for a,b in occurrences) <= visible_end:
                    result.append(dict(kind='python_local', start=occurrences[0][0], end=occurrences[0][1], old=name,
                                       occurrences=occurrences, function_line=fn.lineno, identifiers=sorted({n.id for n in ast.walk(root) if isinstance(n,ast.Name)})))
    return result, tree, None


def validate_change(original, candidate, language, tree, site, replacement):
    if site['kind'] == 'python_local':
        expected = original
        for a, b in sorted(site['occurrences'], reverse=True):
            expected = expected[:a] + replacement + expected[b:]
        if candidate != expected:
            return False, 'rename_not_consistent'
    raw = candidate.encode('utf-8'); changed = parser(language).parse(raw)
    if changed.root_node.has_error:
        return False, 'changed_syntax_error'
    if site['kind'].startswith('comment'):
        if leaves_without_comments(tree, original.encode('utf-8')) != leaves_without_comments(changed,raw):
            return False, 'noncomment_tokens_changed'
        if language == 'Python':
            try:
                if ast.dump(ast.parse(original,type_comments=True),include_attributes=False) != ast.dump(ast.parse(candidate,type_comments=True),include_attributes=False):
                    return False, 'python_ast_changed'
            except SyntaxError:
                return False, 'python_ast_invalid'
    else:
        try:
            before, after = ast.parse(original,type_comments=True), ast.parse(candidate,type_comments=True)
            fn = next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.lineno == site['function_line'])
            for statement in fn.body:
                for node in ast.walk(statement):
                    if isinstance(node,ast.Name) and node.id == replacement:
                        node.id = site['old']
            if ast.dump(before,include_attributes=False) != ast.dump(after,include_attributes=False):
                return False, 'rename_ast_not_equivalent'
        except (SyntaxError, StopIteration):
            return False, 'rename_validation_failed'
    return True, None


def select_sites(candidates, encoded, char_weights, policy, rng, max_sites):
    """Project exact SERSEM target-token weights; max over affected target tokens.

    Multi-occurrence renames are one site. Insertions have no original span and
    receive zero importance, rather than an invented structural weight.
    """
    if policy not in {'sersem', 'uniform'}:
        raise ValueError('site_importance must be sersem or uniform')
    tokens = []
    if policy == 'sersem':
        if char_weights is None:
            raise RuntimeError('SERSEM weights unavailable; explicitly choose uniform to proceed')
        for i, ((a, b), special) in enumerate(zip(encoded.offset_mapping, encoded.special_tokens_mask)):
            if not 0 <= a <= b <= len(char_weights):
                raise ValueError('Invalid structural token offsets')
            if i > 0 and not special and a < b:
                tokens.append((a, b, float(max(char_weights[a:b]))))
    for site in candidates:
        weights = [w for a, b, w in tokens
                   if any(a < end and b > start for start, end in site['occurrences'])]
        site['importance'] = max(weights, default=0.) if policy == 'sersem' else None
        site['importance_source'] = ('shared.weighted_anomaly.StructuralWeights.char_weights'
                                     if policy == 'sersem' else 'uniform')
    comments = [s for s in candidates if s['kind'] == 'comment']
    renames = [s for s in candidates if s['kind'] == 'python_local']
    insertions = [s for s in candidates if s['kind'] == 'comment_insert']
    rng.shuffle(comments); rng.shuffle(renames)
    if policy == 'uniform':
        # Retain the previous uniform-within-kind policy and reserved insertion.
        ordered = ((comments + renames + insertions)[:max_sites-1] + insertions[:1]
                   if insertions else (comments + renames)[:max_sites])
    else:
        ordered = sorted(comments + renames + insertions, key=lambda s: -s['importance'])[:max_sites]
    return list({(s['kind'], s['start'], s['end']): s for s in ordered}.values())


def site_record(site):
    return {k: site[k] for k in ['kind', 'start', 'end', 'occurrences', 'importance', 'importance_source']}


class Perturber:
    def __init__(self, cfg):
        policy = cfg.get('site_importance', 'sersem')
        if policy not in {'sersem', 'uniform'}:
            raise ValueError('site_importance must be sersem or uniform')
        self.structural = None
        if policy == 'sersem':
            from shared.weighted_anomaly import StructuralWeights
            self.structural = StructuralWeights()
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('Kaggle CUDA required; no CPU inference path')
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['mlm_id'], revision=cfg['mlm_revision'], use_fast=True,
                              cache_dir=cfg['mlm_cache_dir'], local_files_only=cfg['mlm_local_files_only'])
        self.model, info = AutoModelForMaskedLM.from_pretrained(cfg['mlm_id'], revision=cfg['mlm_revision'],
                              cache_dir=cfg['mlm_cache_dir'], local_files_only=cfg['mlm_local_files_only'],
                              torch_dtype=torch.float32, output_loading_info=True)
        missing_head = [k for k in info.get('missing_keys',[]) if 'lm_head' in k or 'predictions' in k]
        if missing_head or info.get('mismatched_keys') or self.model.config.model_type != 'roberta' or self.tokenizer.mask_token_id is None:
            raise RuntimeError(f'MLM checkpoint is not validated: {info}')
        if info.get('missing_keys'):
            raise RuntimeError('Refusing randomly initialized MLM tensors: ' + str(info['missing_keys']))
        self.loading_info = info
        self.model.eval()
        self.model.to('cpu' if cfg['park_mlm_on_cpu'] else 'cuda')

    def proposals(self, text, candidates):
        import torch
        tok = self.tokenizer; length = self.cfg['mlm_max_length']
        left_budget = (length-3)//2; right_budget = length-3-left_budget
        sequences = []
        for site in candidates:
            left_text = text[max(0,site['start']-4000):site['start']] + (site['prefix'] if site['kind']=='comment_insert' else '')
            left = tok.encode(left_text, add_special_tokens=False)[-left_budget:]
            right = tok.encode(text[site['end']:site['end']+4000], add_special_tokens=False)[:right_budget]
            ids = [tok.bos_token_id] + left + [tok.mask_token_id] + right + [tok.eos_token_id]
            sequences.append((ids,len(left)+1))
        result = []
        for start in range(0,len(sequences),self.cfg['mlm_batch_size']):
            chunk = sequences[start:start+self.cfg['mlm_batch_size']]
            width = max(len(ids) for ids,_ in chunk)
            ids = torch.full((len(chunk),width),tok.pad_token_id,dtype=torch.long,device='cuda')
            mask = torch.zeros_like(ids)
            for i,(seq,position) in enumerate(chunk):
                ids[i,:len(seq)] = torch.tensor(seq,device='cuda'); mask[i,:len(seq)] = 1
            with torch.no_grad():
                output = self.model(input_ids=ids,attention_mask=mask,output_hidden_states=False,return_dict=True)
                for i,(_,position) in enumerate(chunk):
                    indexes = output.logits[i,position].topk(self.cfg['mlm_top_k']).indices.tolist()
                    result.append([tok.decode([j],clean_up_tokenization_spaces=False).strip() for j in indexes if j not in tok.all_special_ids])
            del output,ids,mask
        return result

    def generate(self, text, language, target_tokenizer):
        self.last_selection = []
        encoded = target_tokenizer(text, truncation=True,max_length=self.cfg['max_length'],add_special_tokens=True,return_offsets_mapping=True,return_special_tokens_mask=True)
        visible_end = max((end for start,end in encoded.offset_mapping),default=0)
        candidates, tree, error = sites(text,language,visible_end,self.cfg['allow_python_local_rename'],self.cfg['allow_comment_insertion'])
        if error:
            return [], [{'reason':error}]
        seed = int(hashlib.sha256((str(self.cfg['seed'])+language+text).encode()).hexdigest()[:16],16)
        rng = random.Random(seed)
        char = self.structural.char_weights(text, language)[0] if self.structural is not None else None
        candidates = select_sites(candidates, encoded, char, self.cfg.get('site_importance', 'sersem'), rng, self.cfg['max_sites'])
        self.last_selection = [site_record(s) for s in candidates]
        if not candidates:
            return [], [{'reason':'no_safe_visible_transformation'}]
        neighbours, failures, seen = [], [], set()
        deferred = []

        def attempt(site, replacement):
            detail = dict(site=site_record(site), replacement=replacement)
            reason = None
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{2,}', replacement) or keyword.iskeyword(replacement):
                reason = 'ineligible_mlm_replacement'
            elif replacement == site['old']:
                reason = 'unchanged_replacement'
            elif site['kind'] == 'python_local' and replacement in site['identifiers']:
                reason = 'identifier_collision'
            if reason:
                failures.append(dict(reason=reason, **detail)); return False
            changed = text
            for a, b in sorted(site['occurrences'], reverse=True):
                value = site.get('prefix','') + replacement + site.get('suffix','') if site['kind']=='comment_insert' else replacement
                changed = changed[:a] + value + changed[b:]
            sha = hashlib.sha256(changed.encode()).hexdigest()
            if sha in seen:
                failures.append(dict(reason='duplicate_neighbour', **detail)); return False
            valid, reason = validate_change(text, changed, language, tree, site, replacement)
            if not valid:
                failures.append(dict(reason=reason, **detail)); return False
            if target_tokenizer(changed,truncation=True,max_length=self.cfg['max_length'],add_special_tokens=True).input_ids == encoded.input_ids:
                failures.append(dict(reason='no_change_in_target_context', **detail)); return False
            seen.add(sha)
            neighbours.append(dict(content=changed, content_sha256=sha, transformation=site['kind'],
                                   replaced=site['old'], replacement=replacement, **site_record(site)))
            return True

        # First pass: at most one neighbour from each important eligible site.
        # Each candidate is always an independent transformation of the original.
        for begin in range(0, len(candidates), self.cfg['mlm_batch_size']):
            chunk = candidates[begin:begin+self.cfg['mlm_batch_size']]
            for site, proposals in zip(chunk, self.proposals(text, chunk)):
                if not proposals:
                    failures.append(dict(reason='no_mlm_proposals', site=site_record(site)))
                for index, replacement in enumerate(proposals):
                    if attempt(site, replacement):
                        deferred.append((site, proposals[index+1:])); break
                if len(neighbours) == self.cfg['k']:
                    return neighbours, failures
        # Fewer than K productive sites: round-robin remaining MLM proposals.
        while deferred:
            remaining = []
            for site, proposals in deferred:
                for index, replacement in enumerate(proposals):
                    if attempt(site, replacement):
                        remaining.append((site, proposals[index+1:])); break
                if len(neighbours) == self.cfg['k']:
                    return neighbours, failures
            deferred = remaining
        failures.append(dict(reason='insufficient_distinct_safe_neighbours',found=len(neighbours),required=self.cfg['k']))
        return neighbours,failures

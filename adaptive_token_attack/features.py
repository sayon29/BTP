"""Frozen, Kaggle-only extraction; compact CPU-readable checkpoints.
The baseline is untouched. Attribution is checked against its actual weights.
"""
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from adaptive_token_attack.data import IDENTITY, fingerprint
from shared.sersem_baseline import write_json
from shared.weighted_anomaly import StructuralWeights

CATEGORIES = ('standard', 'identifier', 'long_identifier', 'string', 'linter',
              'multilingual_identifier', 'comment', 'psychological_marker')
STATIC = np.array([.1, 1., 3., 5., 5., 10., 10., 10.], dtype=np.float32)
SCHEMA = 'adaptive-sersem-per-token-v1'


class Categorizer(StructuralWeights):
    def characters(self, text, language):
        """Maximum weight wins; equal weights use the last category in CATEGORIES.

        Categorization is our attribution convention, not an author category API.
        Numeric weights MUST match the existing char_weights implementation exactly.
        """
        import re
        reference, parse_error = self.char_weights(text, language)
        category = np.zeros(len(text), dtype=np.uint8)
        raw = text.encode('utf-8')

        def apply(a, b, index):
            old = category[a:b]
            replace = (STATIC[index] > STATIC[old]) | ((STATIC[index] == STATIC[old]) & (index > old))
            category[a:b] = np.where(replace, index, old)

        nodes = [self.parsers[language].parse(raw).root_node]
        while nodes:
            node = nodes.pop()
            a = len(raw[:node.start_byte].decode('utf-8', errors='replace'))
            b = len(raw[:node.end_byte].decode('utf-8', errors='replace'))
            index = None
            if 'comment' in node.type:
                index = 6
            elif 'string' in node.type:
                index = 3
            elif 'identifier' in node.type:
                ident = text[a:b]; words = self.extractor.split_identifier(ident)
                unknown = sum(len(w) >= 4 and not self.dictionary.check(w) for w in words)
                if words and unknown > 0 and unknown / len(words) >= .5:
                    index = 5
                elif len(ident) >= 10:
                    index = 2
                elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{3,}', ident):
                    index = 1
            if index is not None:
                apply(a, b, index)
            nodes.extend(node.children)
        for match in self.psych.finditer(text):
            apply(match.start(), match.end(), 7)
        lines = text.splitlines(keepends=True)
        starts = np.cumsum([0]+[len(line) for line in lines])
        for line, column in self.linter.extract_errors(text, language):
            if not (1 <= line <= len(lines) and column >= 1):
                raise ValueError('Invalid linter position')
            offset = starts[line-1]+column-1
            apply(max(0, offset-2), min(len(text), offset+3), 4)
        if not np.array_equal(STATIC[category], reference):
            raise ValueError('Category attribution no longer reproduces exact SERSEM structural weights')
        return category, reference, parse_error


def align_tokens(offsets, special, attention, char_category, char_weights):
    n = len(offsets)
    if not (len(special) == len(attention) == n):
        raise ValueError('Token/mask alignment mismatch')
    weights = np.zeros(n, dtype=np.float32)
    category = np.zeros(n, dtype=np.uint8)
    eligible = np.zeros(n, dtype=np.uint8)
    for i, (a, b) in enumerate(offsets):
        if not 0 <= a <= b <= len(char_weights):
            raise ValueError('Invalid character offsets')
        if i > 0 and not special[i] and attention[i] and attention[i-1] and a < b:
            weights[i] = char_weights[a:b].max()
            labels = char_category[a:b]
            category[i] = labels[STATIC[labels] == weights[i]].max()
            eligible[i] = 1
    if not eligible.any() or not np.array_equal(weights[eligible.astype(bool)], STATIC[category[eligible.astype(bool)]]):
        raise ValueError('No valid targets or category/token weights disagree')
    return category, weights, eligible


def static_score(arrays):
    mask = arrays['eligible'].astype(bool)
    weights = arrays['static_weights'][mask].astype(np.float64)
    return float(np.dot(weights, arrays['signals'][mask].astype(np.float64))/weights.sum())


def validate_record(arrays):
    required = {'input_ids', 'offsets', 'attention_mask', 'special_tokens_mask', 'eligible',
                'category', 'z', 'signals', 'static_weights', 'static_score_gpu', 'ast_has_error', 'n_tokens', 'n_targets'}
    if set(arrays) != required:
        raise ValueError('Feature schema mismatch')
    n = len(arrays['input_ids'])
    if n < 2:
        raise ValueError('At least two target-context tokens required')
    for key in required - {'static_score_gpu', 'ast_has_error', 'n_tokens', 'n_targets'}:
        if arrays[key].shape != ((n, 2) if key == 'offsets' else (n,)):
            raise ValueError('Feature token alignment mismatch: '+key)
        if not np.isfinite(arrays[key]).all():
            raise ValueError('Nonfinite token features: '+key)
    for key in ['attention_mask', 'special_tokens_mask', 'eligible']:
        if not np.isin(arrays[key], [0, 1]).all():
            raise ValueError('Invalid feature mask')
    mask = arrays['eligible'].astype(bool)
    if arrays['n_tokens'].shape != () or arrays['n_targets'].shape != () or int(arrays['n_tokens']) != n or int(arrays['n_targets']) != int(mask.sum()):
        raise ValueError('Cached token counts disagree with masks')
    if (not mask.any() or mask[0] or np.any(mask & (arrays['special_tokens_mask'] != 0))
            or np.any(mask & (arrays['attention_mask'] == 0))):
        raise ValueError('Invalid causal target mask')
    if np.any(mask[1:] & (arrays['attention_mask'][:-1] == 0)):
        raise ValueError('Target without causal context')
    if not np.isin(arrays['category'], np.arange(len(CATEGORIES))).all():
        raise ValueError('Invalid category IDs')
    if not np.array_equal(arrays['static_weights'][mask], STATIC[arrays['category'][mask]]):
        raise ValueError('Static weights/category assignments disagree')
    if np.any(arrays['static_weights'][~mask] != 0):
        raise ValueError('Masked token has nonzero weight')
    if np.any((arrays['signals'] < 0) | (arrays['signals'] > 1)):
        raise ValueError('Invalid sigmoid signals')
    # Stable sigmoid check; features preserve the GPU float32 sigmoid values.
    z = arrays['z'][mask].astype(np.float64)
    expected = np.exp(-np.logaddexp(0., -z))
    if not np.allclose(arrays['signals'][mask], expected, atol=2e-7, rtol=2e-7):
        raise ValueError('Signals disagree with sigmoid(z)')
    if not np.isfinite(arrays['static_score_gpu']) or not 0 <= float(arrays['static_score_gpu']) <= 1:
        raise ValueError('Invalid static GPU score')
    if abs(static_score(arrays)-float(arrays['static_score_gpu'])) > 2e-6:
        raise ValueError('Cached features do not reconstruct extraction-time static score')


class FeatureStore:
    """One compressed NPZ per sample in SQLite, atomic commits and SHA256 checks."""
    def __init__(self, path, identity, read_only=False):
        self.path = Path(path); self.read_only = read_only
        if read_only and not self.path.is_file():
            raise FileNotFoundError('Missing attached feature cache: '+str(self.path))
        self.extraction_identity = identity
        self.identity = fingerprint({'schema': SCHEMA, 'extraction': identity})
        if read_only:
            self.con = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro', uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(self.path)
        self.con.execute('PRAGMA cache_size=-16384')
        if not read_only:
            self.con.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
            self.con.execute('CREATE TABLE IF NOT EXISTS features (sample_id TEXT PRIMARY KEY, payload BLOB, checksum TEXT)')
        saved = self.con.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if saved and saved[0] != self.identity:
            self.con.close(); raise ValueError('Feature cache extraction/config/source identity changed; use a new run')
        if not saved:
            if read_only or self.con.execute('SELECT COUNT(*) FROM features').fetchone()[0]:
                self.con.close(); raise ValueError('Feature cache lacks extraction provenance')
            self.con.execute("INSERT INTO meta VALUES('identity',?)", (self.identity,)); self.con.commit()

    def get(self, row):
        saved = self.con.execute('SELECT payload,checksum FROM features WHERE sample_id=?', (row['sample_id'],)).fetchone()
        if saved is None:
            return None
        payload, checksum = saved
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError('Feature checksum mismatch: '+row['sample_id'])
        with np.load(io.BytesIO(payload), allow_pickle=False) as file:
            metadata = json.loads(str(file['metadata']))
            arrays = {k: file[k] for k in file.files if k != 'metadata'}
        if metadata != {k: row[k] for k in IDENTITY}:
            raise ValueError('Feature sample identity/label/role mismatch: '+row['sample_id'])
        validate_record(arrays)
        return arrays

    def put(self, row, arrays):
        if self.read_only:
            raise RuntimeError('Attached feature cache is read-only')
        validate_record(arrays)
        data = io.BytesIO()
        np.savez_compressed(data, metadata=json.dumps({k: row[k] for k in IDENTITY}, sort_keys=True), **arrays)
        payload = data.getvalue()
        self.con.execute('INSERT OR REPLACE INTO features VALUES(?,?,?)',
                         (row['sample_id'], payload, hashlib.sha256(payload).hexdigest())); self.con.commit()

    def reuse_from(self, path, manifest):
        """Import only compatible per-token rows, never fitted models/thresholds.

        A smoke cache may be partial for a full manifest. Copy original payloads
        after checksum and exact metadata validation; leave missing IDs for normal
        extraction. Source stays read-only, destination commits each imported row.
        """
        if self.read_only:
            raise RuntimeError('Feature reuse requires a writable destination cache')
        if Path(path).resolve() == self.path.resolve():
            raise ValueError('Feature reuse source and destination must differ')
        source = FeatureStore(path, self.extraction_identity, read_only=True)
        imported = 0
        try:
            for row in manifest.to_dict('records'):
                arrays = source.get(row)  # verifies checksum, IDs, labels, hashes and original role
                if arrays is None:
                    continue
                if self.get(row) is not None:
                    continue
                payload, checksum = source.con.execute('SELECT payload,checksum FROM features WHERE sample_id=?', (row['sample_id'],)).fetchone()
                self.con.execute('INSERT INTO features VALUES(?,?,?)', (row['sample_id'], payload, checksum))
                self.con.commit(); imported += 1
        finally:
            source.close()
        return dict(source=str(path), requested=len(manifest), imported=imported,
                    note='Only per-token features imported; no fitted weights, optimizer state or thresholds')

    def digest(self, manifest):
        checks = []
        for row in manifest.to_dict('records'):
            if self.get(row) is None:
                raise FileNotFoundError('Missing cached feature sample: '+row['sample_id'])
            checks.append(self.con.execute('SELECT checksum FROM features WHERE sample_id=?', (row['sample_id'],)).fetchone()[0])
        return fingerprint({'identity': self.identity, 'samples': checks})

    def close(self):
        self.con.close()


class FrozenExtractor:
    def __init__(self, model, tokenizer, cfg):
        import torch
        if not torch.cuda.is_available() or model.device.type != 'cuda':
            raise RuntimeError('Target feature extraction is Kaggle CUDA only')
        if cfg['greybox']['epsilon'] <= 0 or cfg['greybox']['logit_chunk_size'] < 1:
            raise ValueError('Invalid saved grey-box numerical configuration')
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.structural = Categorizer()
        model.eval(); model.requires_grad_(False)

    def one(self, text, language):
        import torch
        char_category, char_weights, error = self.structural.characters(text, language)
        enc = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.cfg['max_length'],
                             add_special_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True)
        offsets = enc.pop('offset_mapping')[0].numpy().astype(np.int32)
        special = enc.pop('special_tokens_mask')[0].numpy().astype(np.uint8)
        attention = enc.attention_mask[0].numpy().astype(np.uint8)
        category, weights, eligible = align_tokens(offsets, special, attention, char_category, char_weights)
        n = len(weights)
        ids = enc.input_ids[0].numpy().astype(np.int32)
        inputs = enc.to(self.model.device)
        z_values = np.zeros(n, dtype=np.float32); signals = np.zeros(n, dtype=np.float32)
        gpu_weights = torch.tensor(weights[1:], dtype=torch.float32, device=self.model.device)
        with torch.no_grad():
            output = self.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
            logits = output.logits[0, :-1, :]; targets = inputs.input_ids[0, 1:]
            numerator = torch.zeros((), dtype=torch.float32, device=self.model.device)
            for start in range(0, n-1, self.cfg['greybox']['logit_chunk_size']):
                end = min(n-1, start+self.cfg['greybox']['logit_chunk_size'])
                block = logits[start:end].float()
                var, mean = torch.var_mean(block, dim=-1, correction=0)
                correct = block.gather(1, targets[start:end, None]).squeeze(1)
                z = (correct-mean)/var.sqrt().clamp_min(self.cfg['greybox']['epsilon'])
                signal = torch.sigmoid(z)
                numerator += (signal * gpu_weights[start:end]).sum()
                z_values[start+1:end+1] = z.cpu().numpy()
                signals[start+1:end+1] = signal.cpu().numpy()
            value = float((numerator/gpu_weights.sum()).item())
        arrays = dict(input_ids=ids, offsets=offsets, attention_mask=attention, special_tokens_mask=special,
                      category=category, static_weights=weights, eligible=eligible, z=z_values, signals=signals,
                      static_score_gpu=np.asarray(value, dtype=np.float32), ast_has_error=np.asarray(error),
                      n_tokens=np.asarray(n,dtype=np.int32), n_targets=np.asarray(int(eligible.sum()),dtype=np.int32))
        validate_record(arrays)
        return arrays


def extraction_identity(cfg, project_root):
    root = Path(project_root)
    names = ['adaptive_token_attack/features.py', 'shared/weighted_anomaly.py', 'shared/upstream_manifest.json']
    upstream = root/'SERSEM-Poisoned-Chalice-Competition-2026'
    manifest = json.loads((root/'shared/upstream_manifest.json').read_text())
    if cfg['upstream_sha'] != manifest['commit_sha']:
        raise ValueError('Baseline/upstream commit mismatch')
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((upstream/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned upstream source mismatch: '+name)
    return {'target': {k: cfg[k] for k in ['model_id','model_revision','dataset_id','dataset_revision','dtype','max_length','greybox','versions','upstream_sha']},
            'attention_backend': cfg.get('resources', {}).get('attention_backend', 'upstream'),
            'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}}


def load_frozen_target(cfg, cache_dir, local_files_only=True):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle GPU; no CPU target inference')
    options = dict(cache_dir=str(cache_dir), local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['model_revision'], use_fast=True, **options)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer offsets required')
    backend = cfg.get('resources', {}).get('attention_backend', 'upstream')
    if backend != 'upstream':
        options['attn_implementation'] = backend
    model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], revision=cfg['model_revision'],
                torch_dtype=getattr(torch, cfg['dtype']), device_map={'': 0}, **options)
    model.eval(); model.requires_grad_(False)
    return model, tokenizer


def extract_manifest(manifest, store, source_store, extractor_factory, monitor):
    """Lazy model loading: a complete cache never instantiates the target."""
    rows = manifest.to_dict('records')
    pending = manifest[[store.get(row) is None for row in rows]]
    if pending.empty:
        return {'requested': len(manifest), 'extracted': 0, 'cached': len(manifest)}
    if store.read_only:
        raise FileNotFoundError('Read-only feature cache is incomplete; missing sample IDs: '+str(pending.sample_id.tolist()[:10]))
    extractor = extractor_factory()
    done = 0
    for batch in source_store.batches(pending, 8):
        for row in batch.to_dict('records'):
            monitor.guard()
            if hashlib.sha256(row['content'].encode()).hexdigest() != row['content_sha256']:
                raise ValueError('Source content hash mismatch')
            try:
                arrays = extractor.one(row['content'], row['language'])
                store.put(row, arrays)
            except Exception as exc:
                with (store.path.parent/'extraction_failures.jsonl').open('a') as file:
                    file.write(json.dumps({'sample_id':row['sample_id'], 'error':type(exc).__name__, 'reason':str(exc)})+'\n')
                raise
            done += 1
            monitor.tick('adaptive_features', done, len(pending))
    store.digest(manifest)  # detect a source iterator that did not yield every ID
    return {'requested':len(manifest), 'extracted':done, 'cached':len(manifest)-done}

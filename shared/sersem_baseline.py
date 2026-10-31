"""Kaggle-only orchestration around the pinned, unchanged upstream attack.
Importing this file does not load a model. No attacks are executed locally.
"""
import contextlib
import hashlib
import io
import json
import random
import types
from pathlib import Path

UPSTREAM_URL = 'https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026.git'
UPSTREAM_SHA = '413f56040e5b4805bcf15ed794dec56bc4e16b41'
LANGUAGES = ['Go', 'Java', 'Python', 'Ruby', 'Rust']


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def validate_environment(requirements_path):
    """Compatibility entry point; scientific dependencies use supported ranges."""
    from shared.kaggle_setup import validate_environment as validate
    return validate(requirements_path)


def load_data(cfg, out):
    """Train split stays development; official test is reserved for evaluation."""
    import pandas as pd
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
    from sklearn.model_selection import train_test_split
    configs = get_dataset_config_names(cfg['dataset_id'], revision=cfg['dataset_revision'])
    if not set(LANGUAGES) <= set(configs):
        raise ValueError(f'Unexpected language configs: {configs}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policy = cfg.get('invalid_content_policy', 'drop')
    if policy not in {'drop', 'error'}:
        raise ValueError('invalid_content_policy must be drop or error')
    frames, schema, excluded_frames = [], {}, []
    exclusion_columns = ['sample_id', 'source_row', 'language', 'official_split',
                         'membership', 'is_member', 'exclusion_reason', 'content_type']
    quality = {'policy': policy, 'status': 'loading', 'splits': {},
               'total_invalid_content_rows': 0,
               'evaluation_population': 'nonempty string content, before sampling; exclusions affect eligible population'}
    def save_quality():
        excluded = pd.concat(excluded_frames, ignore_index=True) if excluded_frames else pd.DataFrame(columns=exclusion_columns)
        excluded.to_parquet(out / 'excluded_content_rows.parquet', index=False)
        write_json(out / 'data_quality.json', quality)
        write_json(out / 'dataset_schema.json', schema)
    for lang in LANGUAGES:
        splits = get_dataset_split_names(cfg['dataset_id'], lang, revision=cfg['dataset_revision'])
        if not {'train', 'test'} <= set(splits):
            raise ValueError(f'{lang}: expected official train/test, found {splits}')
        for split in ['train', 'test']:
            ds = load_dataset(cfg['dataset_id'], lang, split=split, revision=cfg['dataset_revision'])
            schema[f'{lang}/{split}'] = {'features': str(ds.features), 'rows': len(ds), 'fingerprint': ds._fingerprint}
            if not {'content', 'membership'} <= set(ds.column_names):
                raise ValueError(f'Unexpected fields: {ds.column_names}')
            df = ds.to_pandas()
            labels = set(df.membership.unique())
            if labels != {'member', 'non-member'}:
                raise ValueError(f'Unrecognized labels in {lang}/{split}: {labels}')
            if 'language' in df and not df.language.eq(lang).all():
                raise ValueError('Language field disagrees with config')
            df['language'], df['official_split'] = lang, split
            df['is_member'] = df.membership.map({'member': 1, 'non-member': 0})
            # Assign stable source IDs BEFORE exclusions; never renumber source rows.
            df['source_row'] = range(len(df))
            df['sample_id'] = [f'{lang}/{split}/{i}' for i in range(len(df))]
            null = df.content.isna()
            strings = df.content.map(lambda x: isinstance(x, str))
            blank = df.content.map(lambda x: isinstance(x, str) and not bool(x.strip()))
            reasons = pd.Series('', index=df.index, dtype='object')
            reasons.loc[null] = 'null_content'
            reasons.loc[~null & ~strings] = 'non_string_content'
            reasons.loc[blank] = 'empty_or_whitespace_content'
            invalid = reasons.ne('')
            if invalid.any():
                excluded = df.loc[invalid, exclusion_columns[:6]].copy()
                excluded['exclusion_reason'] = reasons.loc[invalid]
                excluded['content_type'] = df.loc[invalid, 'content'].map(lambda x: type(x).__name__)
                excluded_frames.append(excluded[exclusion_columns])
            valid = df.loc[~invalid].copy()
            counts = {str(reason): int(count) for reason, count in reasons.loc[invalid].value_counts().items()}
            quality['splits'][f'{lang}/{split}'] = {
                'source_rows': len(df), 'eligible_content_rows': len(valid),
                'invalid_content_rows': int(invalid.sum()), 'reasons': counts,
                'labels_before': {str(k): int(v) for k, v in df.membership.value_counts().items()},
                'labels_after': {str(k): int(v) for k, v in valid.membership.value_counts().items()},
            }
            quality['total_invalid_content_rows'] += int(invalid.sum())
            # Persist incrementally even when strict mode or a later split fails.
            save_quality()
            if invalid.any():
                print(f'[Data quality] {lang}/{split}: {int(invalid.sum())}/{len(df)} invalid-content rows; {counts}; policy={policy}')
                if policy == 'error':
                    examples = df.loc[invalid, 'sample_id'].head(5).tolist()
                    raise ValueError(f'{lang}/{split}: invalid content at {examples}. See {out / "excluded_content_rows.parquet"}. '
                                     'Use invalid_content_policy=drop in a NEW run to exclude these rows before sampling.')
            # Never coerce nulls to strings, invent content, strip accepted source,
            # or substitute model features/scores for invalid input.
            valid['content_sha256'] = valid.content.map(digest)
            frames.append(valid)
    all_df = pd.concat(frames, ignore_index=True)
    if all_df.groupby('content_sha256').is_member.nunique().max() > 1:
        raise ValueError('Identical content has conflicting membership labels')
    # Hold out every official test content hash, including unsampled rows.
    test_hashes = set(all_df.loc[all_df.official_split.eq('test'), 'content_sha256'])
    dev = all_df[all_df.official_split.eq('train') & ~all_df.content_sha256.isin(test_hashes)]
    test = all_df[all_df.official_split.eq('test')]
    def sample(frame, cap):
        frame = frame.drop_duplicates('content_sha256')
        parts = []
        for (lang, label), group in frame.groupby(['language', 'is_member']):
            if len(group) < cap // 2:
                raise ValueError(f'Insufficient {lang}/{label}: {len(group)}')
            parts.append(group.sample(cap // 2, random_state=cfg['seed']))
        if len(parts) != 10:
            raise ValueError('Missing language/label group')
        return pd.concat(parts, ignore_index=True)
    dev = sample(dev, cfg['dev_per_language'])
    test = sample(test, cfg['test_per_language'])
    shadow, calibration = train_test_split(dev, test_size=0.2, random_state=cfg['seed'], stratify=dev.language + dev.is_member.astype(str))
    shadow, calibration, test = [x.reset_index(drop=True) for x in (shadow, calibration, test)]
    for df, role in [(shadow, 'shadow'), (calibration, 'calibration'), (test, 'heldout_test')]:
        df['split_role'] = role
    quality['status'] = 'complete'
    quality['sampled_rows'] = {'shadow': len(shadow), 'calibration': len(calibration), 'heldout_test': len(test)}
    save_quality()
    return shadow, calibration, test


def build_attack(cfg, upstream, out):
    import sys
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU required. Enable an accelerator; CPU execution is forbidden.')
    if cfg['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU lacks bf16 support. Explicitly select float16 and a new run ID.')
    # Single GPU intentionally; no CPU offload or implicit quantization.
    free, total = torch.cuda.mem_get_info()
    minimum = 9 if '3b' in cfg['model_id'].lower() else 19
    if free < minimum * 2**30:
        raise RuntimeError(f'Only {free/2**30:.1f} GiB free; need at least {minimum} GiB for loading/headroom. Long sequences may need more.')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    sys.path.insert(0, str(upstream / 'code'))
    from LumiaAttack import LumiaAttack
    from ASTExtractor import AST_AVAILABLE, ENCHANT_AVAILABLE
    if not AST_AVAILABLE or not ENCHANT_AVAILABLE:
        raise RuntimeError('Parser/dictionary dependencies failed; fix the environment before running.')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['model_revision'], use_fast=True,
                                               **({'cache_dir': cfg['resources']['model_cache_dir'],
                                                   'local_files_only': cfg['resources']['model_local_files_only']}
                                                  if cfg.get('resources', {}).get('enabled') else {}))
    if not tokenizer.is_fast:
        raise RuntimeError('Fast tokenizer with offsets required')
    resource_settings = cfg.get('resources', {})
    load_options = {}
    if resource_settings.get('enabled'):
        load_options.update(cache_dir=resource_settings['model_cache_dir'],
                            local_files_only=resource_settings['model_local_files_only'])
        if resource_settings['attention_backend'] != 'upstream':
            load_options['attn_implementation'] = resource_settings['attention_backend']
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], revision=cfg['model_revision'], torch_dtype=getattr(torch, cfg['dtype']), device_map={'': 0}, **load_options)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError('Insufficient GPU memory. Choose a larger GPU or explicitly use a shorter smoke run; no algorithm fallback.') from exc
    model.eval()
    if resource_settings.get('enabled') and resource_settings['disable_kv_cache']:
        model.config.use_cache = False  # unused KV outputs; no change to causal hidden/logit operations
    args = types.SimpleNamespace(model_name=cfg['model_id'], max_length=cfg['max_length'], lumia_cache_dir=str(out / 'probes'), lumia_epochs=100, lumia_hidden_dim=128, lumia_lr=1e-3, lumia_batch_size=32, lumia_ensemble_size=5)
    attack = LumiaAttack(args, model, tokenizer)
    if len(attack._get_layer_modules()) != model.config.num_hidden_layers:
        raise RuntimeError('Model layer discovery incompatible')
    original = attack._extract_activations
    if resource_settings.get('enabled'):
        from shared.resource_pipeline import safe_extract
        attack._upstream_extract = original
        attack._extract_activations = lambda texts: safe_extract(attack, texts)
        return attack
    cache = out / 'features'; cache.mkdir(exist_ok=True)
    identity = json.dumps(cfg, sort_keys=True)
    def extract(texts):
        # Once probes are frozen, only selected-layer features are consumed.
        # Upstream still computes/hooks all layers; discard unused pooled outputs.
        selected = list(attack.ensemble_layers) if attack.probes else list(range(attack.num_layers))
        layers = {i: [] for i in selected}
        for start in range(0, len(texts), cfg['feature_chunk_size']):
            chunk = texts[start:start + cfg['feature_chunk_size']]
            key = digest(identity + json.dumps(selected) + json.dumps([digest(t) for t in chunk]))
            path = cache / f'{key}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    values = {i: saved[str(i)] for i in layers}
            else:
                log = io.StringIO()
                # Upstream catches all exceptions and substitutes zeros. Reject those chunks.
                before = {id(m): set(m._forward_hooks) for m in attack._get_layer_modules()}
                try:
                    with contextlib.redirect_stdout(log):
                        values = original(chunk)
                finally:
                    for m in attack._get_layer_modules():
                        for hook_id in set(m._forward_hooks) - before[id(m)]:
                            del m._forward_hooks[hook_id]
                if 'Error extracting activations' in log.getvalue():
                    raise RuntimeError('Upstream extraction failed (possibly GPU OOM); refusing zero features. ' + log.getvalue()[-2000:])
            for i in layers:
                x = values[i]
                if x.shape != (len(chunk), 2 * model.config.hidden_size) or not np.isfinite(x).all():
                    raise RuntimeError(f'Invalid activations at layer {i}: {x.shape}')
                layers[i].append(x)
            if not path.exists():
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **{str(i): values[i] for i in layers}); tmp.replace(path)
            print(f'Features {min(start + len(chunk), len(texts))}/{len(texts)} (cached by content/config)')
        result = {}
        for i in list(layers):
            chunks = layers.pop(i)
            result[i] = np.vstack(chunks)
            del chunks
        attack.activation_dim = 2 * model.config.hidden_size
        return result
    attack._extract_activations = extract
    return attack


def prepare_probes(attack, cfg, shadow, calibration, test, out):
    import shutil
    probe = attack.cache_dir / f"probes_{cfg['model_id'].replace('/', '_')}.pkl"
    manifest_path = probe.with_suffix('.manifest.json')
    identity = {k: cfg[k] for k in ['model_id', 'model_revision', 'dataset_id', 'dataset_revision', 'dtype', 'max_length', 'seed']}
    identity['upstream_sha'] = UPSTREAM_SHA
    identity['runner_sha256'] = cfg['runner_sha256']
    identity['versions'] = cfg['versions']
    if cfg.get('resources', {}).get('enabled'):
        identity['feature_storage_dtype'] = cfg['resources']['feature_dtype']
    protected = set(calibration.content_sha256) | set(test.content_sha256)
    if cfg['probe_mode'] == 'verified':
        if not cfg['trust_probe_pickle']:
            raise ValueError('Verify source/provenance before enabling trust_probe_pickle; pickle executes code.')
        source = Path(cfg['probe_path'])
        source_manifest = source.with_suffix('.manifest.json')
        if not source.exists() or not source_manifest.exists():
            raise FileNotFoundError('Verified probes require a pickle and .manifest.json; no fallback')
        probe.parent.mkdir(exist_ok=True)
        if source.resolve() != probe.resolve():
            shutil.copy2(source, probe); shutil.copy2(source_manifest, manifest_path)
    elif cfg['probe_mode'] != 'train':
        raise ValueError('probe_mode must be train or verified')
    if (cfg.get('resources', {}).get('enabled') and cfg['probe_mode'] == 'train'
            and probe.exists() and not manifest_path.exists() and (out / 'probe_training.pkl').exists()):
        # Recover a final upstream probe save interrupted before its manifest.
        # Completed layer states are validated by train_grouped before rewriting.
        probe.unlink()
    if probe.exists():
        if not manifest_path.exists():
            raise RuntimeError('Probe lacks provenance manifest. Use a new output directory.')
        meta = json.loads(manifest_path.read_text())
        if meta['identity'] != identity or meta['sha256'] != hashlib.sha256(probe.read_bytes()).hexdigest():
            raise ValueError('Probe identity/hash mismatch')
        trained = set(meta['training_content_hashes'])
        if meta.get('training_official_split') != 'train':
            raise ValueError('Probe provenance must establish official train-only development')
        ids = meta.get('training_sample_ids', [])
        if len(ids) != len(meta['training_content_hashes']) or any('/train/' not in x for x in ids):
            raise ValueError('Missing/incompatible training sample identity provenance')
        if not trained or protected & trained:
            raise ValueError('Probe training/calibration/test overlap or missing training provenance')
        if cfg['probe_mode'] == 'train' and trained != set(shadow.content_sha256):
            raise ValueError('Cached probe training pool changed')
        if not attack._load_probes():
            raise RuntimeError('Probe loading failed; no fallback')
        for role, indices in meta.get('internal_indices', {}).items():
            if cfg['probe_mode'] == 'train':
                shadow.loc[indices, 'split_role'] = role
    else:
        if cfg['probe_mode'] != 'train':
            raise FileNotFoundError('Missing verified probe')
        if cfg.get('resources', {}).get('enabled'):
            from shared.resource_pipeline import train_grouped
            idx = train_grouped(attack, cfg, shadow, out)
        else:
            idx = attack.train_probes(shadow.content.tolist(), shadow.is_member.tolist())
        internal = dict(zip(['shadow_train', 'shadow_early_stop', 'shadow_layer_selection'], [x.tolist() for x in idx]))
        for role, indices in internal.items():
            shadow.loc[indices, 'split_role'] = role
        write_json(manifest_path, {'identity': identity, 'sha256': hashlib.sha256(probe.read_bytes()).hexdigest(), 'training_content_hashes': shadow.content_sha256.tolist(), 'training_sample_ids': shadow.sample_id.tolist(), 'training_official_split': 'train', 'internal_indices': internal})
    expected = 2 * attack.model.config.hidden_size
    layers = attack.ensemble_layers or [attack.best_layer]
    if attack.activation_dim != expected or not layers or not set(layers) <= set(attack.probes) or any(i < 0 or i >= attack.num_layers for i in layers):
        raise ValueError('Probe architecture/layers incompatible')
    if len(layers) != min(5, attack.num_layers):
        raise ValueError('Expected original top-five layer ensemble')
    return probe


def score(attack, df, path):
    import numpy as np
    result = df.copy()
    if not attack.probes or not set(attack.ensemble_layers) <= set(attack.probes):
        raise RuntimeError('Missing probes; random upstream fallback forbidden')
    scores, vectors = attack.compute_scores_with_vectors(df.content.tolist(), df.language.tolist())
    if len(scores) != len(df) or not np.isfinite(scores).all():
        raise RuntimeError('Invalid scores')
    result['lumia_score'], result['lumia_confidence_vector'] = scores, vectors
    result.drop(columns=['content']).to_parquet(path, index=False)
    return result


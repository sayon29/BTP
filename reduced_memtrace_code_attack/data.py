"""First 128 baseline-tokenizer input positions, no artificial padding/BOS."""
import hashlib
import json
from pathlib import Path
from adaptive_token_attack.data import fingerprint, IDENTITY
from reduced_memtrace_code_attack.features import feature_schema


def fixed_configuration():
    path = Path(__file__).with_name('reproducibility.json')
    spec = json.loads(path.read_text())
    if fingerprint(spec) != FROZEN_SPEC_SHA256:
        raise ValueError('Frozen scientific configuration changed')
    return spec


def scoring_identity(cfg, tokenizer, model_config, sources):
    fixed = fixed_configuration()
    for key in ['model_id', 'model_revision', 'dtype', 'seed', 'dataset_id', 'dataset_revision']:
        if cfg[key] != fixed[key]:
            raise ValueError('Saved baseline identity mismatch: '+key)
    if model_config.num_hidden_layers != fixed['expected_num_hidden_layers'] or model_config.hidden_size != fixed['expected_hidden_size']:
        raise ValueError('Unexpected baseline model architecture')
    if not tokenizer.is_fast or tokenizer.truncation_side != 'right' or model_config.model_type != 'starcoder2':
        raise ValueError('Require baseline fast right-truncating StarCoder2 tokenizer/model')
    return dict(fixed, tokenizer_revision=cfg['model_revision'], tokenizer_sha256=hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode()).hexdigest(), tokenizer_special_tokens=tokenizer.special_tokens_map,
        model_config=model_config.to_dict(), baseline_max_length=int(cfg['max_length']),
        schema=feature_schema(model_config.num_hidden_layers), versions=cfg['versions'], sources=sources,
        attention_backend=cfg.get('resources', {}).get('attention_backend', 'upstream'))


def make_plan(text, tokenizer, identity):
    cap = min(identity['analysis_tokens'], identity['baseline_max_length'],
              identity['model_config']['max_position_embeddings'], int(tokenizer.model_max_length))
    full = tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
    enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=cap,
                    return_special_tokens_mask=True, return_offsets_mapping=True)
    ids, attention = enc['input_ids'], enc.get('attention_mask', [1]*len(enc['input_ids']))
    targets = [i for i, ((a,b), special) in enumerate(zip(enc['offset_mapping'], enc['special_tokens_mask']))
               if i > 0 and attention[i] and attention[i-1] and not special and a < b]
    if sum(attention) < 2 or not targets:
        raise ValueError('Source has fewer than two valid tokens or no predictable code targets')
    return dict(input_ids=ids, attention_mask=attention, target_positions=targets,
        metadata=dict(source_token_count=len(full), analysis_input_tokens=len(ids),
            valid_token_count=sum(attention), effective_target_tokens=len(targets),
            truncated_tokens=max(0, len(full)-len(ids)), truncated=len(full)>len(ids)))


def cache_key(row, identity):
    # Labels are bound to cached features for integrity, but never enter model inputs.
    return fingerprint(dict(sample={k:row[k] for k in IDENTITY}, scientific=identity))


def prepare_plans(selected, tokenizer, identity, cache, source_store, monitor):
    import pandas as pd
    rows = pd.concat(list(selected.values()), ignore_index=True)
    pending = rows[[cache.get('plan', cache_key(r, identity)) is None for r in rows.to_dict('records')]]
    seen = set()
    for batch in source_store.batches(pending, 8) if len(pending) else []:
        for row in batch.to_dict('records'):
            monitor.guard()
            if row['sample_id'] not in set(pending.sample_id) or row['sample_id'] in seen:
                raise ValueError('Unexpected/duplicate source sample')
            if hashlib.sha256(row['content'].encode()).hexdigest() != row['content_sha256']:
                raise ValueError('Source content hash mismatch')
            cache.put('plan', cache_key(row, identity), make_plan(row['content'], tokenizer, identity))
            seen.add(row['sample_id'])
        monitor.tick('reduced_memtrace_plans', len(seen), len(pending))
    if seen != set(pending.sample_id):
        raise ValueError('Incomplete source retrieval')
    return {r['sample_id']: cache.get('plan', cache_key(r, identity)) for r in rows.to_dict('records')}

FROZEN_SPEC_SHA256 = '42f180ef654920ac4b3249ba5c82abcb0567b97bca470aeaf92003f6c4c58479'

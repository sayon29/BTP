"""Reuse original token planning and enforce frozen combined scientific identity."""
import hashlib
import json
from pathlib import Path
import warnings
from adaptive_token_attack.data import fingerprint, IDENTITY
from reduced_memtrace_code_attack.data import (make_plan, prepare_plans, cache_key,
    scoring_identity as reduced_identity)
from reduced_memtrace_code_attack.evaluation import exact_match
from attention_memtrace_code_attack.features import feature_schema


def fixed_configuration():
    spec = json.loads(Path(__file__).with_name('reproducibility.json').read_text())
    if fingerprint(spec) != FROZEN_SPEC_SHA256:
        raise ValueError('Frozen combined scientific configuration changed')
    return spec


def scoring_identity(cfg, tokenizer, model_config, sources):
    old = reduced_identity(cfg, tokenizer, model_config, sources)
    fixed = fixed_configuration()
    if model_config.num_attention_heads != fixed['expected_query_heads'] or model_config.num_key_value_heads != fixed['expected_kv_heads']:
        raise ValueError('Unexpected StarCoder2 GQA architecture')
    identity = dict(old)
    identity.update(fixed)
    identity.update(attention_backend='eager', reduced_attention_backend=old['attention_backend'],
        schema=feature_schema(model_config.num_hidden_layers, model_config.num_attention_heads))
    return identity


def verified_reduced_prefixes(scores_path, selected, identity):
    """Optional exact prefix reuse only with full provenance and hash validation.

    Score comparison remains available even when old feature provenance is absent.
    """
    import pandas as pd
    if scores_path is None:
        return {}, dict(status='not_attached')
    root = Path(scores_path).parent
    paths = {name:root/name for name in ['config.json','feature_schema.json','run_manifest.json','features.parquet']}
    try:
        cfg = json.loads(paths['config.json'].read_text())
        schema = json.loads(paths['feature_schema.json'].read_text())
        manifest = json.loads(paths['run_manifest.json'].read_text())
        old = cfg['scoring']
        if schema != identity['schema']['reduced'] or old['schema'] != schema:
            raise ValueError('Reduced schema differs from unchanged 167-feature implementation')
        for key in ['model_id','model_revision','tokenizer_revision','tokenizer_sha256','tokenizer_special_tokens',
                    'dataset_id','dataset_revision','seed','dtype','analysis_tokens','baseline_max_length',
                    'token_policy','classifier','boundary_policy','versions','model_config']:
            if old[key] != identity[key]:
                raise ValueError('Reduced feature identity mismatch: '+key)
        if old['numerical_implementation'] != identity['reduced_numerical_implementation']:
            raise ValueError('Reduced numerical implementation differs')
        if old['attention_backend'] != identity['reduced_attention_backend']:
            raise ValueError('Saved prefix attention backend differs from original baseline convention')
        # Match implementation files determining the numeric prefix, not prose or run selection.
        for name in ['features.py','data.py','inference.py']:
            key = 'reduced_memtrace_code_attack/'+name
            if old['sources'][key] != identity['sources'][key]:
                raise ValueError('Reduced extraction source differs: '+name)
        digest = hashlib.sha256(paths['features.parquet'].read_bytes()).hexdigest()
        if manifest.get('status') != 'complete' or manifest.get('artifacts',{}).get('features.parquet') != digest:
            raise ValueError('Reduced feature artifact lacks completed-run checksum')
        frame = pd.read_parquet(paths['features.parquet'])
        selected_frame = pd.concat(list(selected.values()),ignore_index=True)
        verified = exact_match(selected_frame, frame, schema['names'])
        result = {r['sample_id']:[r[n] for n in schema['names']] for r in verified.to_dict('records')}
        return result, dict(status='verified_reuse', features_sha256=digest,
            config_sha256=hashlib.sha256(paths['config.json'].read_bytes()).hexdigest(),
            reduced_attention_backend=old['attention_backend'])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        warnings.warn('Reduced features will be computed once in the attention forward; saved prefix reuse unavailable: '+str(exc))
        return {}, dict(status='unverified_or_missing_provenance', reason=str(exc))

FROZEN_SPEC_SHA256 = '53dd244ff76a102a51c93ca8848af23dd0d6218c841dbbbb636ca5d65f422c40'

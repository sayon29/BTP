"""Prepare fixed plans and run frozen inference; caller supplies existing helpers."""
import hashlib
import json
from pathlib import Path
import re

from adaptive_token_attack.data import IDENTITY, frame_fingerprint
from recall_code_attack.cache import fingerprint, sample_identity
from recall_code_attack.prefix import approved_pool, choose_prefix
from recall_code_attack.scoring import aligned_targets, score_plan
from shared.sersem_baseline import LANGUAGES, write_json


def scoring_identity(cfg, tokenizer, model_config, source_hashes):
    if cfg['model_id'] != 'bigcode/starcoder2-3b' or cfg['dtype'] != 'float16':
        raise ValueError('This fixed experiment requires bigcode/starcoder2-3b / float16')
    for key in ('model_revision','dataset_revision'):
        if not re.fullmatch(r'[a-f0-9]{40}', cfg[key]):
            raise ValueError('Require immutable 40-character '+key+' from saved baseline')
    if tokenizer.truncation_side != 'right':
        raise ValueError('Baseline right truncation required')
    if model_config.model_type != 'starcoder2':
        raise ValueError('Unexpected model architecture')
    limit = min(int(model_config.max_position_embeddings), int(tokenizer.model_max_length))
    return dict(schema='recall-code-v1', model_id=cfg['model_id'], model_revision=cfg['model_revision'],
                tokenizer_revision=cfg['model_revision'], tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                tokenizer_special_tokens=tokenizer.special_tokens_map, dataset_id=cfg['dataset_id'], dataset_revision=cfg['dataset_revision'],
                dtype='float16', logsoftmax_dtype='float32', reduction_dtype='float64', versions=cfg['versions'],
                baseline_max_length=int(cfg['max_length']), context_limit=limit,
                sliding_window=getattr(model_config,'sliding_window',None), model_config=model_config.to_dict(),
                attention_backend=cfg.get('resources',{}).get('attention_backend','upstream'),
                chunk_size=256, prefix_tokens=128, prefix_seed=42, prefix_policy='hash-ranked-shadow-nonmember-same-language-v1',
                overlap_policy='32-model-token-or-20-lexical-token-or-complete-short-inclusion-v1',
                target_policy='baseline-add-special-true-right-truncation; skip-anchor-in-both; tail-truncate-to-pos-limit',
                score='mean_LL_conditioned / mean_LL_unconditioned; high_is_member',
                sources=source_hashes, baseline_loss_reuse=False)


def plan_key(row, identity, pool_hash):
    return fingerprint(dict(sample=sample_identity(row), scoring=identity, pool=pool_hash))


def atomic_parquet(frame, path):
    path = Path(path)
    tmp = path.with_suffix('.parquet.tmp')
    frame.to_parquet(tmp,index=False)
    tmp.replace(path)


def prepare_plans(original, selected, tokenizer, identity, cache, source_store, monitor):
    import pandas as pd
    pool = approved_pool(original)
    if set(original['heldout'].language) != set(LANGUAGES):
        raise ValueError('Require Go, Java, Python, Ruby and Rust')
    pool_hash = frame_fingerprint(pool)
    all_rows = pd.concat(list(selected.values()),ignore_index=True)
    pending = all_rows[[cache.get('plan',plan_key(r,identity,pool_hash)) is None for r in all_rows.to_dict('records')]]
    if pending.empty:
        return {r['sample_id']:cache.get('plan',plan_key(r,identity,pool_hash)) for r in all_rows.to_dict('records')}, pool_hash
    # Cache only the selected source texts and approved pool, never a full split.
    wanted = pd.concat([pool,pending],ignore_index=True).drop_duplicates('sample_id')
    texts = {}
    for batch in source_store.batches(wanted,8):
        for row in batch.to_dict('records'):
            monitor.guard()
            if hashlib.sha256(row['content'].encode()).hexdigest() != row['content_sha256']:
                raise ValueError('Source identity mismatch')
            texts[row['sample_id']] = row['content']
    if set(texts) != set(wanted.sample_id):
        raise ValueError('Missing selected source text')
    candidates = []
    for row in pool.to_dict('records'):
        ids = tokenizer(texts[row['sample_id']], add_special_tokens=False, truncation=True, max_length=128)['input_ids']
        decoded = tokenizer.decode(ids,clean_up_tokenization_spaces=False)
        if len(ids) == 128 and decoded.strip():
            candidates.append(dict(row, prefix_ids=ids, prefix_text=decoded))
    for n,row in enumerate(pending.to_dict('records')):
        monitor.guard()
        text = texts[row['sample_id']]
        full_ids = tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
        # This is exactly the baseline's tokenizer input span before context adjustment.
        base_ids = tokenizer(text,add_special_tokens=True,truncation=True,max_length=identity['baseline_max_length'])['input_ids']
        prefix,rejected = choose_prefix(row,text,full_ids,candidates)
        kept,metadata = aligned_targets(base_ids,prefix['prefix_ids'],identity['context_limit'])
        plan = dict(metadata, base_ids=kept, prefix_ids=prefix['prefix_ids'],
                    prefix_sample_id=prefix['sample_id'],prefix_content_sha256=prefix['content_sha256'],
                    prefix_language=prefix['language'],prefix_official_split=prefix['official_split'],
                    prefix_split_role=prefix['split_role'],prefix_is_member=0,
                    prefix_hash=fingerprint(prefix['prefix_ids']), pool_sha256=pool_hash,
                    original_input_tokens=len(full_ids),baseline_truncated_tokens=len(full_ids)-len(base_ids),
                    overlap_rejections=rejected)
        cache.put('plan',plan_key(row,identity,pool_hash),plan)
        monitor.tick('recall_plans',n+1,len(pending))
    return {r['sample_id']:cache.get('plan',plan_key(r,identity,pool_hash)) for r in all_rows.to_dict('records')}, pool_hash


def run_scores(selected, plans, identity, cache, model_factory, monitor, out):
    import pandas as pd
    rows = pd.concat(list(selected.values()),ignore_index=True)
    results, prefix_records = [], []
    for row in rows.to_dict('records'):
        plan = plans[row['sample_id']]
        meta = {k:v for k,v in plan.items() if k not in {'base_ids','prefix_ids'}}
        prefix_records.append(dict(sample_identity(row),**meta,prefix_token_ids=plan['prefix_ids']))
    monitor.guard()
    atomic_parquet(pd.DataFrame(prefix_records),Path(out)/'prefix_manifest.parquet')
    model_holder = []
    def lazy_model():
        if not model_holder:
            model_holder.append(model_factory())
        return model_holder[0]
    try:
        for n,row in enumerate(rows.to_dict('records')):
            plan = plans[row['sample_id']]
            result = score_plan(row,plan,identity,cache,lazy_model,monitor.guard)
            meta = {k:v for k,v in plan.items() if k not in {'base_ids','prefix_ids'}}
            results.append(dict({k:row[k] for k in IDENTITY},**meta,**result))
            monitor.tick('recall_scores',n+1,len(rows))
    finally:
        model_holder.clear()
    scored = pd.DataFrame(results)
    monitor.guard()
    atomic_parquet(scored,Path(out)/'recall_scores.parquet')
    return scored

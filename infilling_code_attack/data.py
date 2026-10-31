"""Saved manifests and deterministic contiguous span, no partition construction."""
import hashlib
import json
from pathlib import Path
import re
from infilling_code_attack.cache import fingerprint,sample_identity
from infilling_code_attack.method import future_positions
from shared.sersem_baseline import LANGUAGES


def fixed_configuration():
    return json.loads(Path(__file__).with_name('reproducibility.json').read_text())


def scoring_identity(cfg,tokenizer,model_config,sources):
    fixed=fixed_configuration()
    if cfg['model_id']!=fixed['model_id'] or cfg['dtype']!='float16':raise ValueError('Require bigcode/starcoder2-3b / float16')
    if not tokenizer.is_fast or tokenizer.truncation_side!='right' or model_config.model_type!='starcoder2':
        raise ValueError('Require baseline fast/right tokenizer and native StarCoder2')
    for field in ('model_revision','dataset_revision'):
        if not re.fullmatch('[a-f0-9]{40}',cfg[field]):raise ValueError('Immutable revision required: '+field)
    return dict(fixed,model_revision=cfg['model_revision'],tokenizer_revision=cfg['model_revision'],
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        special_tokens=tokenizer.special_tokens_map,dataset_id=cfg['dataset_id'],dataset_revision=cfg['dataset_revision'],
        baseline_max_length=int(cfg['max_length']),context_limit=min(int(model_config.max_position_embeddings),int(tokenizer.model_max_length)),
        model_config=model_config.to_dict(),versions=cfg['versions'],sources=sources,
        attention_backend=cfg.get('resources',{}).get('attention_backend','upstream'))


def make_plan(text,tokenizer,identity):
    cap=min(identity['span_tokens'],identity['baseline_max_length'],identity['context_limit'])
    enc=tokenizer(text,add_special_tokens=True,truncation=True,max_length=cap,
                  return_offsets_mapping=True,return_special_tokens_mask=True)
    ids=enc['input_ids'];n=len(ids);offsets=enc['offset_mapping'];special=enc['special_tokens_mask']
    attention=enc.get('attention_mask',[1]*n)
    if not(len(offsets)==len(special)==len(attention)==n):raise ValueError('Tokenizer/mask alignment mismatch')
    positions=[t for t,(a,b) in enumerate(offsets) if t>0 and attention[t] and attention[t-1] and not special[t] and a<b]
    complete=len(tokenizer(text,add_special_tokens=True,truncation=False)['input_ids'])
    metadata=dict(span_tokens=n,span_start=0,span_end_exclusive=n,complete_input_tokens=complete,
        truncated_input_tokens=max(0,complete-n),short_span=n<identity['span_tokens'],
        effective_analysis_tokens=len(positions),fewer_than_128_eligible_tokens=len(positions)<128,
        positions_without_future=sum(not future_positions(t,n,identity['future_tokens']) for t in positions),
        positions_with_reduced_future=sum(len(future_positions(t,n,identity['future_tokens']))<identity['future_tokens'] for t in positions),
        first_input_token_scored=False)
    reason='Fewer than three span tokens' if n<3 else 'Zero bottom-fraction token count' if int(len(positions)*identity['bottom_fraction'])<1 else None
    return dict(input_ids=ids,target_positions=positions,metadata=metadata,invalid_reason=reason)


def plan_key(row,identity):return fingerprint(dict(sample=sample_identity(row),scoring=identity))


def prepare_plans(selected,tokenizer,identity,cache,source_store,monitor):
    import pandas as pd
    rows=pd.concat(list(selected.values()),ignore_index=True)
    if set(rows.language)!=set(LANGUAGES):raise ValueError('Require all five saved languages')
    pending=rows[[cache.get('plan',plan_key(r,identity)) is None for r in rows.to_dict('records')]]
    wanted=set(pending.sample_id);seen=set()
    if wanted:
        for batch in source_store.batches(pending,8):
            for row in batch.to_dict('records'):
                monitor.guard()
                if row['sample_id'] not in wanted or row['sample_id'] in seen:raise ValueError('Unexpected/duplicate source identity')
                if hashlib.sha256(row['content'].encode()).hexdigest()!=row['content_sha256']:raise ValueError('Source content hash mismatch')
                cache.put('plan',plan_key(row,identity),make_plan(row['content'],tokenizer,identity))
                seen.add(row['sample_id']);monitor.tick('infilling_plans',len(seen),len(wanted))
        if seen!=wanted:raise ValueError('Missing selected source content')
    return {r['sample_id']:cache.get('plan',plan_key(r,identity)) for r in rows.to_dict('records')}

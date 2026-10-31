"""Unchanged saved partitions; baseline tokenizer eligibility, first 256 targets."""
import hashlib
import json
from pathlib import Path
import re

from adaptive_token_attack.data import IDENTITY, fingerprint, frame_fingerprint
from shared.sersem_baseline import LANGUAGES
from spw_code_attack.cache import sample_identity
from spw_code_attack import METHOD_NAME, METHOD_STATUS


FROZEN_SPEC_SHA256 = 'b1b2fa359faebed5dbde24deef6eb9326010af421f6083a8e15dcaeade1db607'


def fixed_configuration():
    path = Path(__file__).with_name('reproducibility.json')
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_SPEC_SHA256:
        raise ValueError('Frozen scientific configuration changed; requires a reviewed new experiment')
    return json.loads(path.read_text())


def scoring_identity(cfg,tokenizer,model_config,sources):
    fixed=fixed_configuration()
    if cfg['model_id']!=fixed['model_id'] or cfg['dtype']!='float16':
        raise ValueError('Require frozen bigcode/starcoder2-3b in float16')
    for key in ('model_revision','dataset_revision'):
        if not re.fullmatch('[a-f0-9]{40}',cfg[key]):raise ValueError('Require immutable '+key)
    if not tokenizer.is_fast or tokenizer.truncation_side!='right' or model_config.model_type!='starcoder2':
        raise ValueError('Require baseline fast tokenizer with right truncation and StarCoder2')
    return dict(fixed,model_revision=cfg['model_revision'],tokenizer_revision=cfg['model_revision'],
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        tokenizer_special_tokens=tokenizer.special_tokens_map,dataset_id=cfg['dataset_id'],
        dataset_revision=cfg['dataset_revision'],baseline_max_length=int(cfg['max_length']),
        context_limit=min(int(model_config.max_position_embeddings),int(tokenizer.model_max_length)),
        model_config=model_config.to_dict(),versions=cfg['versions'],sources=sources,
        attention_backend=cfg.get('resources',{}).get('attention_backend','upstream'),
        method_status=METHOD_STATUS,method_name=METHOD_NAME)


def make_plan(text,tokenizer,identity):
    cap=min(identity['baseline_max_length'],identity['context_limit'])
    enc=tokenizer(text,add_special_tokens=True,truncation=True,max_length=cap,
                  return_offsets_mapping=True,return_special_tokens_mask=True)
    ids=enc['input_ids'];offsets=enc['offset_mapping']
    special=enc['special_tokens_mask'];attention=enc.get('attention_mask',[1]*len(ids))
    if not (len(ids)==len(offsets)==len(special)==len(attention)):
        raise ValueError('Tokenizer alignment mismatch')
    eligible=[i for i,(a,b) in enumerate(offsets) if i>0 and attention[i] and attention[i-1] and not special[i] and a<b]
    selected=eligible[:identity['analysis_target_tokens']]
    if not selected:raise ValueError('No eligible predictable code target tokens')
    complete_ids=tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
    end=selected[-1]+1
    metadata=dict(complete_input_tokens=len(complete_ids),baseline_context_cap=cap,
        capped_input_tokens=len(ids),context_or_baseline_truncated_tokens=max(0,len(complete_ids)-len(ids)),
        capped_eligible_targets=len(eligible),selected_target_tokens=len(selected),analysis_input_tokens=end,
        analysis_tail_input_tokens_omitted=len(ids)-end,analysis_eligible_targets_omitted=len(eligible)-len(selected),
        short_sample=len(selected)<identity['analysis_target_tokens'],
        first_input_token_scored=False,first_target_position=selected[0],last_target_position=selected[-1])
    return dict(input_ids=ids[:end],target_positions=selected,metadata=metadata)


def plan_key(row,identity):
    return fingerprint(dict(sample=sample_identity(row),scoring=identity))


def prepare_plans(selected,tokenizer,identity,cache,source_store,monitor):
    import pandas as pd
    rows=pd.concat(list(selected.values()),ignore_index=True)
    if set(rows.language)!=set(LANGUAGES):raise ValueError('Require all five saved languages')
    pending=rows[[cache.get('plan',plan_key(r,identity)) is None for r in rows.to_dict('records')]]
    if not pending.empty:
        seen=set()
        for batch in source_store.batches(pending,8):
            for row in batch.to_dict('records'):
                monitor.guard()
                if row['sample_id'] not in set(pending.sample_id) or row['sample_id'] in seen:
                    raise ValueError('Unexpected/duplicate source identity')
                if hashlib.sha256(row['content'].encode()).hexdigest()!=row['content_sha256']:
                    raise ValueError('Selected source content changed')
                cache.put('plan',plan_key(row,identity),make_plan(row['content'],tokenizer,identity))
                seen.add(row['sample_id']);monitor.tick('spw_inspired_plans',len(seen),len(pending))
        if seen!=set(pending.sample_id):raise ValueError('Incomplete selected source rows')
    return {r['sample_id']:cache.get('plan',plan_key(r,identity)) for r in rows.to_dict('records')}

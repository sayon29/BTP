"""Mean target log-likelihood ratio, high = member. No prefix labels scored."""
import math
from recall_code_attack.cache import fingerprint, sample_identity


def aligned_targets(base_ids, prefix_ids, context_limit):
    if len(prefix_ids) != 128 or context_limit < 130:
        raise ValueError('Require 128 prefix tokens and space for two candidate tokens')
    # Preserve baseline tokenization, including its first context/anchor token.
    kept = base_ids[:context_limit-len(prefix_ids)]
    if len(kept) < 2:
        raise ValueError('Candidate has no predictable baseline target tokens')
    return kept, dict(baseline_input_tokens=len(base_ids), retained_input_tokens=len(kept),
                      scored_target_tokens=len(kept)-1, context_truncated_tokens=len(base_ids)-len(kept),
                      target_ids_sha256=fingerprint(kept[1:]))


def ratio(unconditioned, conditioned):
    if not all(math.isfinite(x) and x < 0 for x in (unconditioned, conditioned)):
        raise FloatingPointError('LLs must be finite and strictly negative; denominator is never clipped')
    score = conditioned/unconditioned
    if not math.isfinite(score):
        raise FloatingPointError('Nonfinite ReCaLL ratio')
    return score


def target_positions(length, target_start, start, end):
    """Logit/label slice bounds shared by CPU alignment tests and GPU scoring."""
    if not 1 <= target_start < length or not 0 <= start < end <= length-1:
        raise ValueError("Invalid causal scoring interval")
    first = max(start, target_start-1)
    return first, end


def mean_log_likelihood(model, ids, target_start, chunk_size=256):
    """Shift once: logits at i predict ids[i+1]. KV chunks bound vocabulary memory.

    Scored tokens are ids[target_start:] in BOTH branches. The last sequence
    token need not be forwarded because no later target is scored. KV cache is
    native Transformers; attention/position context is continuous between chunks.
    FP16 target forward, FP32 log-softmax, FP64 summation of token log probs.
    """
    import torch
    if not 1 <= target_start < len(ids) or chunk_size < 1:
        raise ValueError('Empty/invalid target mask')
    total, count, past = 0., 0, None
    with torch.inference_mode():
        for start in range(0, len(ids)-1, chunk_size):
            end = min(start+chunk_size, len(ids)-1)
            inputs = torch.tensor([ids[start:end]], dtype=torch.long, device=model.device)
            output = model(input_ids=inputs, attention_mask=torch.ones((1,end), dtype=torch.long, device=model.device),
                           past_key_values=past, use_cache=True, return_dict=True, output_hidden_states=False)
            past = output.past_key_values
            first, _ = target_positions(len(ids),target_start,start,end)
            if first < end:
                logits = output.logits[0, first-start:end-start].float()
                targets = torch.tensor(ids[first+1:end+1], dtype=torch.long, device=model.device)
                logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:,None]).squeeze(1)
                if not torch.isfinite(logp).all():
                    raise FloatingPointError('Nonfinite token log probability')
                total += float(logp.double().sum().item())
                count += len(targets)
            del output
    if count != len(ids)-target_start:
        raise AssertionError('Autoregressive target mask mismatch')
    return total/count


def score_plan(row, plan, identity, cache, model_factory, guard=lambda: None, ll_function=None):
    """Each valid branch skips its model call, including partial-pair resume."""
    ll_function = ll_function or mean_log_likelihood
    key = fingerprint(dict(sample=sample_identity(row), scoring=identity, plan=plan))
    values = {}
    for branch in ('unconditioned','conditioned'):
        branch_key = fingerprint([key,branch])
        saved = cache.get('ll', branch_key)
        if saved is None:
            guard()
            prefix = [] if branch == 'unconditioned' else plan['prefix_ids']
            ll = ll_function(model_factory(), prefix+plan['base_ids'], len(prefix)+1,
                                     identity['chunk_size'])
            if not math.isfinite(ll) or ll >= 0:
                raise FloatingPointError('Invalid LL')
            saved = dict(ll=ll, target_hash=plan['target_ids_sha256'])
            cache.put('ll', branch_key, saved)
        if saved['target_hash'] != plan['target_ids_sha256']:
            raise ValueError('Cached target span differs')
        values[branch] = saved['ll']
    return dict(unconditioned_ll=values['unconditioned'], conditioned_ll=values['conditioned'],
                recall_score=ratio(values['unconditioned'],values['conditioned']), cache_key=key)

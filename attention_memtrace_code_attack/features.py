"""All specified attention families; append to the unmodified Reduced schema."""
from adaptive_token_attack.data import fingerprint
from reduced_memtrace_code_attack.features import feature_schema as reduced_schema

LAYER_STATISTICS = ('entropy_mean', 'concentration', 'sparsity', 'self_attention_bias',
                    'previous_token_bias', 'mean_attention_distance')
HEAD_STATISTICS = ('entropy_mean', 'maximum_focus_mean')


def attention_schema(layers, heads):
    if layers < 1 or heads < 1:
        raise ValueError('Require positive transformer layer/query-head counts')
    names = []
    for layer in range(layers):
        names += [f'attention.{layer:02d}.{s}' for s in LAYER_STATISTICS]
        for head in range(heads):
            names += [f'attention.{layer:02d}.head.{head:02d}.{s}' for s in HEAD_STATISTICS]
    payload = dict(version='attention-probability-reductions-v1', layers=layers, query_heads=heads,
        names=names, families=['attention']*len(names), layer_statistics=list(LAYER_STATISTICS),
        head_statistics=list(HEAD_STATISTICS), entropy_base=2, entropy_epsilon=1e-12,
        sparsity_threshold='mean normalized head-mean probability over valid causal pairs; strict less-than; our frozen convention',
        row_sum_atol=.002, masked_probability_atol=1e-7,
        normalization='validate raw valid-query probabilities first; zero masked keys; FP32 row-renormalize allowed keys',
        query_aggregation='unweighted mean across valid queries; previous bias excludes first valid query',
        pair_domain='valid query/key positions with original key index <= original query index',
        special_token_policy='include all nonpadding input positions, matching Reduced hidden features')
    return dict(payload, sha256=fingerprint(payload))


def feature_schema(layers, heads):
    reduced, attention = reduced_schema(layers), attention_schema(layers, heads)
    payload = dict(version='combined-attention-memtrace-v1', reduced=reduced, attention=attention,
        names=reduced['names']+attention['names'], families=reduced['families']+attention['families'],
        preserved_prefix_features=len(reduced['names']), attention_features=len(attention['names']))
    return dict(payload, sha256=fingerprint(payload))


def reduce_attention_layer(attention, valid_mask, schema):
    """Return GPU scalar vector + audit tensors; no raw probabilities returned/cached."""
    import torch
    heads = schema['query_heads']
    if attention is None or attention.ndim != 4 or attention.shape[:2] != (1, heads):
        raise ValueError('Missing attention or wrong query-head shape; KV heads are not query heads')
    t = attention.shape[-1]
    if attention.shape[-2] != t or valid_mask.shape != (1,t):
        raise ValueError('Require square, full-sequence aligned attention maps')
    indices = valid_mask[0].bool().nonzero().flatten()
    if len(indices) < 2:
        raise ValueError('Attention features require at least two valid tokens')
    # Drop padded query rows before any arithmetic, but retain every key for leakage validation.
    p = attention[0].index_select(1, indices).float()
    allowed = (torch.arange(t, device=p.device)[None,:] <= indices[:,None]) & valid_mask[0].bool()[None,:]
    if not torch.isfinite(p).all() or (p < 0).any() or (p > 1+schema['row_sum_atol']).any():
        raise FloatingPointError('Attention must contain finite post-softmax probabilities in [0,1]')
    leak = p.masked_select(~allowed[None,:,:]).abs()
    leak_max = leak.max() if leak.numel() else p.new_zeros(())
    if leak_max > schema['masked_probability_atol']:
        raise ValueError('Nonzero future/padding attention probability; causal mask/backend is invalid')
    p = p.masked_fill(~allowed[None,:,:], 0.)
    sums = p.sum(dim=-1, keepdim=True)
    error = (sums-1).abs().max()
    if error > schema['row_sum_atol'] or (sums <= 0).any():
        raise ValueError('Attention row sums fail frozen post-softmax tolerance')
    p = p/sums
    mean = p.mean(dim=0)
    eps = schema['entropy_epsilon']
    layer_entropy = -(mean*torch.log2(mean+eps)).sum(dim=-1).mean()
    head_entropy = -(p*torch.log2(p+eps)).sum(dim=-1).mean(dim=-1)
    head_focus = p.max(dim=-1).values.mean(dim=-1)
    values = mean.masked_select(allowed)
    tau = values.mean()
    concentration = mean.max(dim=-1).values.mean()
    sparsity = (values < tau).float().mean()
    q = torch.arange(len(indices), device=p.device)
    self_bias = mean[q, indices].mean()
    # Previous valid token: adjacent real token order, even with internal padding.
    previous_bias = mean[q[1:], indices[:-1]].mean()
    distance = (indices[:,None]-torch.arange(t,device=p.device)[None,:]).abs().float()
    mean_distance = (mean*distance).sum()/mean.sum()
    vector = torch.cat((torch.stack((layer_entropy, concentration, sparsity, self_bias,
        previous_bias, mean_distance)), torch.stack((head_entropy,head_focus), dim=1).flatten()))
    if not torch.isfinite(vector).all():
        raise FloatingPointError('Nonfinite attention feature vector')
    return vector, torch.stack((error,leak_max,tau))


def reduce_attentions(attentions, valid_mask, schema):
    import torch
    if attentions is None or len(attentions) != schema['layers']:
        raise ValueError('Every transformer layer must return attention; no feature fallback permitted')
    vectors, audits = [], []
    for attention in attentions:
        vector, audit = reduce_attention_layer(attention, valid_mask, schema)
        vectors.append(vector)
        audits.append(audit)
    vector = torch.cat(vectors)
    if len(vector) != len(schema['names']):
        raise ValueError('Incomplete attention feature schema')
    # Only compact scalars cross to CPU. FP32 maps are local to each layer reduction.
    audit = torch.stack(audits).cpu().tolist()
    return vector.cpu().tolist(), dict(max_raw_row_sum_error=max(r[0] for r in audit),
        maximum_masked_probability=max(r[1] for r in audit), sparsity_thresholds=[r[2] for r in audit])

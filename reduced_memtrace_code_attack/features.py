"""Label-free FP32 feature reductions; Hugging Face hidden-state tuple convention."""
from adaptive_token_attack.data import fingerprint


def selected_boundaries(layers):
    if layers < 1:
        raise ValueError('At least one transformer layer required')
    return sorted({(layers * i + 2) // 4 for i in range(5)})


def feature_schema(layers):
    names, families = [], []
    def add(family, prefix, statistics):
        for statistic in statistics:
            names.append(prefix + '.' + statistic)
            families.append(family)
    for layer in range(layers):
        add('transition', f'transition.{layer:02d}_{layer+1:02d}',
            ['distance_mean', 'distance_std', 'cosine_mean', 'cosine_std'])
    for layer in selected_boundaries(layers):
        add('activation', f'activation.{layer:02d}',
            ['abs_mean', 'rms', 'std', 'abs_max', 'near_zero_fraction'])
    add('confidence', 'confidence', ['nll_mean', 'nll_std', 'max_probability_mean',
        'max_probability_std', 'top_two_margin_mean', 'entropy_mean', 'entropy_std'])
    for layer in selected_boundaries(layers):
        add('context', f'context.{layer:02d}', ['distance_mean', 'distance_std', 'distance_max'])
    payload = dict(version='reduced-memtrace-fp32-v1', layers=layers,
        selected_boundaries=selected_boundaries(layers), names=names, families=families,
        hidden_state_convention='HF tuple: embeddings, intermediate block inputs, final normalized output',
        embedding_transition=True, std_correction=0, cosine_eps=1e-8, near_zero_threshold=1e-3)
    return dict(payload, sha256=fingerprint(payload))


def reduce_hidden(hidden_states, valid_mask, schema):
    import torch
    import torch.nn.functional as F
    if len(hidden_states) != schema['layers'] + 1:
        raise ValueError('Unexpected hidden-state layer count')
    shape = hidden_states[0].shape
    if len(shape) != 3 or shape[0] != 1 or valid_mask.shape != shape[:2]:
        raise ValueError('Require batch size one and aligned token mask')
    if any(h.shape != shape for h in hidden_states):
        raise ValueError('Hidden-state shape mismatch')
    mask = valid_mask[0].bool()
    if int(mask.sum()) < 2:
        raise ValueError('At least two valid tokens required')
    values = []
    def stats(x):
        return [x.mean(), x.std(correction=0)]
    for left, right in zip(hidden_states[:-1], hidden_states[1:]):
        a, b = left[0, mask].float(), right[0, mask].float()
        distance = torch.linalg.vector_norm(b-a, dim=-1)
        cosine = F.cosine_similarity(b, a, dim=-1, eps=schema['cosine_eps'])
        values.extend(stats(distance) + stats(cosine))
    for layer in schema['selected_boundaries']:
        h = hidden_states[layer][0, mask].float()
        values.extend([h.abs().mean(), h.square().mean().sqrt(), h.std(correction=0),
                       h.abs().max(), (h.abs() < schema['near_zero_threshold']).float().mean()])
    # Context features are placed after confidence in the public schema.
    context = []
    for layer in schema['selected_boundaries']:
        h = hidden_states[layer][0, mask].float()
        running = h.cumsum(dim=0) / torch.arange(1, len(h)+1, device=h.device).float()[:, None]
        evolution = torch.linalg.vector_norm(running[1:]-running[:-1], dim=-1)
        context.extend(stats(evolution) + [evolution.max()])
    return values, context


def confidence_features(logits, targets):
    """Logits already aligned to next-token targets; never copy vocabulary arrays to CPU."""
    import torch
    if logits.ndim != 2 or logits.shape[0] != len(targets) or logits.shape[1] < 2 or not len(targets):
        raise ValueError('Invalid logits/target alignment')
    z = logits.float()
    normalizer = torch.logsumexp(z, dim=-1)
    nll = normalizer - z.gather(1, targets[:, None]).squeeze(1)
    top = z.topk(2, dim=-1).values
    probabilities = (top-normalizer[:, None]).exp()
    # logsumexp(z) - E_p[z], computed in FP32 on the device.
    entropy = normalizer - ((z-normalizer[:, None]).exp()*z).sum(dim=-1)
    return [nll.mean(), nll.std(correction=0), probabilities[:, 0].mean(),
        probabilities[:, 0].std(correction=0), (probabilities[:, 0]-probabilities[:, 1]).mean(),
        entropy.mean(), entropy.std(correction=0)]


def combine_features(hidden, confidence, context, schema):
    import torch
    result = torch.stack(hidden + confidence + context).float()
    if len(result) != len(schema['names']) or not torch.isfinite(result).all():
        raise FloatingPointError('Feature vector has incorrect size or nonfinite values')
    return result.cpu().tolist()

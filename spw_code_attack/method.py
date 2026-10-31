"""Our frozen fallback equations, never represented as published SPW-MIA equations."""
import math
import numpy as np


def next_length(length):
    if not isinstance(length,int) or length < 1:
        raise ValueError('Positive integer prefix length required')
    return length//2 if length > 1 else None


def validate_prediction(value):
    p,q,g=value['log_true'],value['log_greedy'],value['greedy_id']
    if not (math.isfinite(p) and math.isfinite(q) and p <= q+1e-6 and q <= 1e-6
            and isinstance(g,int) and g >= 0):
        raise ValueError('Invalid original/shortened token probabilities')


def token_weight(length, accepted_length, original_log_true, accepted_log_true):
    if not 1 <= accepted_length <= length:
        raise ValueError('Invalid accepted suffix length')
    p,ps=math.exp(original_log_true),math.exp(accepted_log_true)
    if not (0 <= p <= 1 and 0 <= ps <= 1):
        raise ValueError('Invalid probability')
    return 1.+(1.-accepted_length/length)*(1.-abs(p-ps))


def bernoulli_js(p,q):
    """Exact zero-safe JS in nats, no smoothing epsilon or fitted scale."""
    if not (np.isfinite(p) and np.isfinite(q) and 0 <= p <= 1 and 0 <= q <= 1):
        raise ValueError('Probabilities must be finite in [0,1]')
    a=np.array([p,1.-p],dtype=np.float64)
    b=np.array([q,1.-q],dtype=np.float64)
    m=(a+b)/2.
    def kl(x):
        active=x>0
        return float(np.sum(x[active]*(np.log(x[active])-np.log(m[active]))))
    return max(0.,.5*(kl(a)+kl(b)))


def aggregate(original, states):
    if not original or len(original)!=len(states):
        raise ValueError('Require all eligible targets')
    weights,divergences=[],[]
    for o,s in zip(original,states):
        validate_prediction(o);validate_prediction(s['accepted'])
        weights.append(token_weight(s['original_length'],s['accepted_length'],o['log_true'],s['accepted']['log_true']))
        divergences.append(bernoulli_js(math.exp(o['log_true']),math.exp(o['log_greedy'])))
    score=-float(np.average(divergences,weights=weights))
    if not math.isfinite(score) or score > 1e-12:
        raise FloatingPointError('Nonfinite or wrong-direction approximation score')
    return dict(spw_inspired_score=score,weights=weights,token_divergences=divergences,
                mean_weight=float(np.mean(weights)),mean_js=float(np.mean(divergences)))


def straightforward(plan, predict):
    """Slow reference for CPU equivalence tests: one full prefix per query."""
    original,states=[],[]
    ids=plan['input_ids']
    for t in plan['target_positions']:
        o=predict(ids[:t],ids[t]);validate_prediction(o)
        accepted=o;length=t
        while length > 1:
            short=next_length(length)
            candidate=predict(ids[t-short:t],ids[t]);validate_prediction(candidate)
            if candidate['greedy_id'] != o['greedy_id']:
                break
            length,accepted=short,candidate
        original.append(o)
        states.append(dict(original_length=t,accepted_length=length,accepted=accepted))
    return aggregate(original,states),states

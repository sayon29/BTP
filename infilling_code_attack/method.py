"""Transparent normalized Infilling Score and exact small-vocabulary Bayes check."""
import math
import numpy as np


class InvalidSample(ValueError):
    pass


def future_positions(t,n,m=5):
    """Literal zero-based translation of author range(i,min(i+m,N-3)), i=t-1."""
    if not 1<=t<n or m<0:raise ValueError('Invalid causal target position')
    return list(range(t+1,min(t+m,n-2)+1))


def statistics(log_probs,target,min_sigma=1e-6):
    lp=np.asarray(log_probs,dtype=np.float64)
    if lp.ndim!=1 or not np.isfinite(lp).all():raise InvalidSample('Nonfinite probability distribution')
    probabilities=np.exp(lp)
    # Exact central population variance under the predictive distribution.
    mu=float(np.sum(probabilities*lp));variance=float(np.sum(probabilities*(lp-mu)**2))
    sigma=math.sqrt(max(0.,variance));g=int(lp.argmax())
    return dict(log_true=float(lp[target]),log_greedy=float(lp[g]),greedy_id=g,
                mu=mu,sigma=sigma,sigma_used=max(sigma,min_sigma))


def standardized(value,log_field='log_true'):
    sigma=value['sigma_used']
    terms=(value[log_field],value['mu'],sigma)
    if not all(math.isfinite(v) for v in terms) or sigma<=0:raise InvalidSample('Invalid sufficient statistics')
    return (value[log_field]-value['mu'])/sigma


def token_score(original,altered,t,n,m=5):
    """Author-normalized Eq 10. Altered moments MUST be recomputed per context."""
    own=original[str(t)]
    score=standardized(own)-standardized(own,'log_greedy')
    raw=own['log_true']-own['log_greedy']
    for j in future_positions(t,n,m):
        score+=standardized(original[str(j)])-standardized(altered[str(j)])
        raw+=original[str(j)]['log_true']-altered[str(j)]['log_true']
    if not math.isfinite(score) or not math.isfinite(raw):raise InvalidSample('Nonfinite token score')
    return float(score),float(raw)


def aggregate(positions,scores,fraction=.2):
    values=np.asarray(scores,dtype=np.float64)
    if len(positions)!=len(values) or not np.isfinite(values).all():raise InvalidSample('Incomplete/nonfinite token scores')
    k=int(len(values)*fraction)
    if not k:raise InvalidSample('Zero bottom-fraction token count')
    ranked=sorted(range(len(values)),key=lambda i:(values[i],positions[i]))[:k]
    return dict(infilling_score=float(np.mean(values[ranked])),bottom_count=k,
                bottom_positions=[positions[i] for i in ranked])


def reference(plan,predict,min_sigma=1e-6,m=5,fraction=.2):
    """Independent-prefix reference: predict(prefix) returns normalized log probs."""
    ids=plan['input_ids'];n=len(ids)
    original={str(t):statistics(predict(ids[:t]),ids[t],min_sigma) for t in range(1,n)}
    scores=[];raw=[]
    for t in plan['target_positions']:
        greedy=original[str(t)]['greedy_id'];alt_ids=list(ids);alt_ids[t]=greedy
        altered={str(j):statistics(predict(alt_ids[:j]),ids[j],min_sigma) for j in future_positions(t,n,m)}
        score,ratio=token_score(original,altered,t,n,m);scores.append(score);raw.append(ratio)
    return dict(aggregate(plan['target_positions'],scores,fraction),token_scores=scores,raw_log_ratios=raw)


def bayes_log_ratio(prefix,true_token,greedy_token,future,predict,vocabulary_size):
    """Tiny-test ONLY: explicitly sum the Bayes partition, then take its ratio."""
    prior=np.asarray(predict(prefix),dtype=float);weights=[]
    for token in range(vocabulary_size):
        value=float(prior[token]);context=prefix+[token]
        for target in future:
            value+=float(predict(context)[target]);context=context+[target]
        weights.append(value)
    maximum=max(weights);log_z=maximum+math.log(sum(math.exp(x-maximum) for x in weights))
    return (weights[true_token]-log_z)-(weights[greedy_token]-log_z)

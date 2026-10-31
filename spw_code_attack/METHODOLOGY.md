# SPW-inspired methodology and fidelity boundary

## Paper access and fidelity boundary

Paper: Linyun Xie et al., *Shortening the prefix! Members and non-members exhibit divergent behavior*, Knowledge-Based Systems 346, 116100 (2026), [DOI](https://doi.org/10.1016/j.knosys.2026.116100). Access attempted 2026-10-30.

The [publisher preview](https://www.sciencedirect.com/science/article/abs/pii/S0950705126008269) supplied the abstract, introduction and a method outline. It describes prediction-based dynamic prefix shortening, token weights, and a scoring term comparing true-token and greedy-token probabilities. Its outline places the probability discussion in §3.3, shortening behavior in §3.4, and scoring in §3.5. **The complete contents of those sections were not obtained.** Full article/PDF fetches failed; metadata endpoints did not supply method text; exact-title PDF/arXiv/GitHub searches did not locate accessible full equations or author code. `research_access.json` records the individual attempts.

Unavailable and **not invented as paper facts**:

* All numbered mathematical equations and algorithm pseudocode.
* Exact shortening schedule, acceptance/stopping/selection criteria.
* Weight definition and normalization.
* Divergence definition, orientation and combination into the sample score.
* Recommended/default parameters and experimental context lengths.

Consequently, none of the equations below has a claimed source equation number. They are **our own explicit approximation**, motivated only by the accessible high-level description. Full original SPW-MIA would require the complete method text and a separate implementation audit. Every config, metric report, score file, run manifest and notebook identifies `paper_faithful=False`; the score column is `spw_inspired_score`. Do not describe these results as published SPW-MIA reproduction or paper-equivalent performance.

## Our single frozen mathematical specification

Use token IDs `x[0], ..., x[L-1]` from the saved baseline tokenizer. Let `T` be the first at most 256 eligible target positions. Position 0 is unpredicted; token t uses context `c_t = x[:t]`. Prefix lengths count all actual IDs, including unscored tokens left as context. Compute under the original prefix:

```
p_t = P(x[t] | c_t)
g_t = argmax_v P(v | c_t)
q_t = P(g_t | c_t) = max_v P(v | c_t)
```

The greedy prediction is the next token under the **true sample prefix**, not a separately generated rollout. Ties use the smallest vocabulary index. Original true and greedy log probabilities and IDs are cached. No reference model or external prefix is used.

**A1 — dynamic shortening (ours).** Start with accepted length `ell=t` and accepted true probability `p*_t=p_t`. While `ell>1`, propose `k=floor(ell/2)` and evaluate the actual suffix `x[t-k:t]`, starting its positions at zero. If its greedy prediction equals the original `g_t`, accept `ell=k`, retain its true-token probability `p*_t`, and continue. Otherwise stop at the first mismatch. Stop also at length 1. Never evaluate an empty prefix. Do not assume greedy agreement is monotonic: this is the specified greedy halving path, not a binary search for a globally shortest suffix. This is one dynamic path per token, **not a grid or best-result selection**.

**A2 — token weights (ours).** Let `ell*_t` be the final accepted length:

```
w_t = 1 + (1 - ell*_t/t) * (1 - abs(p_t - p*_t))
```

Weights lie in `[1,2)`; unshortened tokens have weight 1. Both factors lie in `[0,1]`, so larger successfully removed fractions and more stable true-token probabilities increase emphasis without an unbounded ratio or tuning scale. A failed proposal is recorded but never accepted. This assignment is a precommitted heuristic about stable prediction under context removal; it is not a verified paper formula or a proof of membership.

**A3 — probability divergence and sample score (ours).** Define two Bernoulli distributions per original-prefix token, `B_t=(p_t,1-p_t)` and `G_t=(q_t,1-q_t)`, with `M_t=(B_t+G_t)/2`:

```
D_t = JS(B_t, G_t)
    = 0.5 * KL(B_t || M_t) + 0.5 * KL(G_t || M_t)
S(x) = -sum_t(w_t * D_t) / sum_t(w_t)
```

Use natural logarithms and the exact `0*log(0)=0` convention, without smoothing epsilon. JS is finite, symmetric and bounded by `log(2)`. The score lies in `[-log(2),0]`. True-token probability equals greedy probability when the true token is a maximizer, giving zero divergence, the maximum possible score. Thus **higher scores are member-like** under the predeclared accuracy/confidence intuition. This direction is never estimated or flipped using labels. The accessible paper mentions comparing collections of probabilities but does not reveal its divergence; our Bernoulli construction must **not** be attributed to it.

There is only this score. No NLL-difference substitute, score-definition selection, reference model, hyperparameter sweep or learned fusion is available. `reproducibility.json` freezes seed 42, these rules, 256 eligible targets, four equal-length shortened prefixes per batch, 256-token original-forward chunks, precision and provenance. A1–A3 are labels for this README only, not paper equation numbers.

## Tokenization and code adaptation

Reuse the saved baseline's pinned model/tokenizer revision and tokenizer conventions: fast tokenizer, `add_special_tokens=True`, right truncation, no retokenization of concatenated text. First cap at `min(baseline.max_length, model.max_position_embeddings, tokenizer.model_max_length)`. An eligible target has position >0, a nonempty character offset, attention on both it and its predecessor, and is not a special token. This matches the adaptive eligibility predicate without invoking AST/category weighting. Select the first **256** eligible positions identically across Go, Java, Python, Ruby and Rust. All earlier input IDs, even excluded targets, remain in context. There is no artificial BOS/EOS, language prompt, external context or deletion of whitespace tokens.

Retain short samples, record their actual target count, and flag `short_sample=True`; zero eligible targets fail. Record complete and capped input counts, omitted analysis/context tokens, selected count and exact target-ID SHA256. Analysis input ends after the last selected target. Logit at position `t-1` predicts `x[t]`. The original first input token is never scored; a first eligible token with one predecessor has no shortened context and weight 1. Every shortened suffix resets its own positions to zero. Respect the native attention/sliding-window implementation of the pinned model.

Analyzing 256 eligible targets is a **compute-budget adaptation**. The original context length could not be retrieved, so we cannot state whether or by how much this shortens its experiments. The original method also concerns natural language; source-code behavior across these five languages is a new domain and is not evidence of paper equivalence.


## Verification ledger — 2026-10-31

The publisher's full article returned HTTP 403. The Elsevier XML endpoint was inaccessible. DOI/title/SPW-MIA searches did not locate accessible full equations, supplementary algorithms or official code. The available publisher preview supports only the high-level ideas of shortening prefixes, token weighting and comparing actual/greedy probability collections. See [publisher](https://doi.org/10.1016/j.knosys.2026.116100) and `research_access.json`.

| Required published detail | Section requested | Equation number | Verification |
|---|---|---|---|
| Actual and greedy probability collections | 3.3 | Unavailable | High-level description only |
| Dynamic shortening schedule, acceptance and stop | 3.4 | Unavailable | Not verified |
| Token weight and normalization | 3.5 | Unavailable | Not verified |
| Divergence, aggregation, membership direction | 3.5 | Unavailable | Not verified |
| Pretrained-model defaults | Experiments | Unavailable | Not verified |

The section references follow the previously recorded preview outline and the requested audit; they are not evidence that those sections were read in full. A1–A3 above have no paper equation numbers. Tests verify these adaptation equations only. A published-equation test cannot honestly pass until the full method becomes available. This package is a separately named experiment; implementing the original method remains unresolved.

## Fixed experiment and inference scope

Seed 42; frozen `bigcode/starcoder2-3b` FP16 with the saved immutable revision; first 256 eligible targets; original decoder chunks of 256; shortened equal-length batches of four; natural logarithms; FP32 log-softmax and FP64 reductions. No reference model, fine-tuning, sweep, learned score or held-out orientation choice. The halving rule, weights, Bernoulli distributions and sign are local scientific choices, not published defaults. Relative to the paper, source code, baseline tokenization/context cap and the 256-target budget are domain/compute adaptations.

Validate and save all original and selected manifests. Score only the 200 calibration and 1,000 held-out examples in full mode (20 and 30 in smoke). Shadow and validation labels do not train or tune this unsupervised method. Calibration fixes one global threshold before held-out evaluation. Complete held-out coverage is required; scoring errors stop with resumable checkpoints, never silently reduce the cohort. Optional adaptive comparison requires exact identity matching and explicit attachment.

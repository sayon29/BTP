# Single combined Attention-Augmented memTrace

Name: **Attention-Augmented memTrace (StarCoder2 code adaptation)**. This is the sole primary attack: all 167 frozen Reduced features followed by every attention family explicitly specified in the request. One random forest consumes the entire vector. No attention-only model, ablation, search, feature selection or competing configuration is implemented.

Reference: Makhija et al., [Neural Breadcrumbs, EACL 2026](https://aclanthology.org/2026.eacl-long.262.pdf), Section 2, especially p. 5610. The paper's attention formulas motivate this extension. Its supplement does not resolve the adaptive threshold or enumerate the reported 705-dimensional Pythia attention block. The linked [author repository](https://github.com/amazon-science/NeuralBreadcrumbs-MIA-EACL-2026) returned HTTP 404 on 2026-11-01. Consequently, this implements **all specified attention families**, not a verified reconstruction of every undocumented author-code feature. No exact original memTrace reproduction or AUC improvement is claimed.

## Preserved scientific settings

The original `reduced_memtrace_code_attack` files are imported unchanged. Its feature-schema dictionary is nested verbatim, and its 167 names, ordering, families, transition/activation/confidence/context reductions and token planner are reused directly. Adjacent transitions include embeddings and final normalization. Five selected boundaries remain `[0,8,15,23,30]`. See the unchanged [Reduced methodology](../reduced_memtrace_code_attack/METHODOLOGY.md) for their equations. Attention additions begin at column 167, never replace a Reduced column.

Target/tokenizer: `bigcode/starcoder2-3b`, immutable revision `733247c55e3f73af49ce8e9c7949bf14af205928`. Dataset/revision, seed 42, first 128 input positions, baseline tokenizer special-token policy, original four partitions and all 1,000 held-out identities remain unchanged. No membership or language enters feature extraction or classifier inputs. Confidence retains shifted eligible targets; attention uses all nonpadding input positions, including any tokenizer-supplied special tokens, like the original hidden features.

The supported eager backend materializes actual post-softmax weights. Fresh Reduced features use precisely the original functions in the same eager forward. FP16 eager and SDPA can differ numerically despite identical equations and weights, so bitwise equality to a historical SDPA run is not promised. A tiny 30-layer eager model verifies all 167 prefix values exactly against the original extractor on that same model/backend. A separate tiny FP32 comparison checks eager/SDPA agreement within fixed numerical tolerance. These are implementation tests, not additional attack experiments.

If attached Reduced results include a completed-run checksum, exact matching feature schema, matching numerical/extraction source hashes, model/tokenizer identity and all selected sample identities, the saved 167 values are reused exactly and their reductions are skipped. Otherwise the identical formulas run once on the attention-enabled forward. Reuse status and hashes are frozen before extraction and bound into cache identity, ensuring smoke/full compatibility. Empty or incomplete old JSON metadata cannot authorize feature reuse; readable identity-matched scores can still support comparison.

## Attention definitions and frozen conventions

For layer l, the raw tensor must have shape `(1,H,T,T)` with **H query heads**. StarCoder2-3B has 30 layers, 24 query heads and 2 KV heads; GQA expansion produces 24 different attention distributions. Features are never indexed by KV-head count.

Let Q be valid query positions, K_t = {s: s is valid and s≤t}, and Ω = {(t,s): t∈Q, s∈K_t}. All indices refer to original input positions. Valid-query probabilities must be finite, nonnegative, at most 1+.002, and have allowed-key row sums within .002 of one. Future/padded-key probabilities must have absolute value ≤1e-7. Invalid padded query rows are ignored before arithmetic. Fail on invalid rows; never repair logits or missing maps into probabilities.

After validation, cast to FP32, mask invalid keys and normalize each allowed row by its measured sum, correcting FP16 quantization drift. Call the result p(h,t,s). Set b(t,s) = mean_h p(h,t,s), and epsilon=1e-12. The normalization and tolerances are fixed numerical conventions, not paper defaults. Existing FP16 underflow zeros remain zeros; missing probability mass cannot be recovered from returned tensors.

For every layer, append these six scalars, in this order:

1. **Entropy:** mean over valid queries of −sum_{s∈K_t} b(t,s) log₂(b(t,s)+epsilon).
2. **Concentration:** mean_t max_{s∈K_t} b(t,s).
3. **Sparsity:** mean_{(t,s)∈Ω} 1[b(t,s)<tau_l], with **tau_l = mean_{Ω} b(t,s)**.
4. **Self-attention bias:** mean_t b(t,t).
5. **Previous-token bias:** mean over all valid queries after the first of b(t,previous_valid_token(t)). For contiguous unpadded inputs this is b(t,t−1). Padding gaps do not count as predecessor tokens.
6. **Mean distance:** sum_{Ω}|t−s|b(t,s) / sum_{Ω}b(t,s).

Then, for each query head in index order, append:

- **Head entropy:** mean_t [−sum_{s∈K_t} p(h,t,s) log₂(p(h,t,s)+epsilon)].
- **Head maximum focus:** mean_t max_{s∈K_t} p(h,t,s).

This keeps per-head information rather than replacing it with a summary over heads. The paper does not specify a different per-head entropy aggregate, so the valid-query mean is our explicit choice. Concentration is the head-mean-map focus; head focus is separate. No unverified global maximum, arbitrary additional moments, positional features or inferred dimensions are invented.

The paper describes adaptive sparsity but omits its formula. **Our threshold rule is a documented adaptation**, not a recovered author default. After row normalization and for contiguous T, tau=2/(T+1), so this threshold depends on length rather than layer magnitude. The resulting below-threshold fraction still depends on the attention distribution. Both numerator and denominator exclude causal/padding zeros; genuine zero-valued allowed probabilities count. Thresholds are saved in audit metadata, not added as classifier features. No threshold choice depends on membership performance.

Dimension: each layer contributes `6 + 2H`. For L=30, H=24: **1,620 attention dimensions**, plus **167 Reduced dimensions**, totaling **1,787**. Full exact names, layer/head identifiers, conventions and hashes are in `attention_feature_schema.json` and `feature_schema.json`.

## Supported extraction and memory

The official Transformers **4.52.4** StarCoder2 source was inspected from its PyPI wheel. `eager_attention_forward` expands KV groups, applies the causal mask, computes softmax in FP32 and casts weights back to query dtype; evaluation disables attention dropout. `Starcoder2Attention` explicitly falls back from SDPA when attentions are requested. Load directly with `attn_implementation='eager'` through the existing loader's resource option: no private function replacement, weight changes or second model copy.

One inference-mode full-sequence forward per uncached sample returns hidden states, logits and all layer attentions. Frozen FP16 parameters, eval mode, batch one, no gradients, generation or KV cache. Verified saved-prefix reuse turns off hidden-state output and skips original reductions, while the same eager pass still supplies every attention layer. Vocabulary logits are never copied/cached; only feature scalars reach CPU. Full attention matrices are never written to disk.

The public full-output tuple retains 30 FP16 attention maps for one sample. At 128 positions this is `30×24×128×128×2` bytes = **22.5 MiB**. This bounded payload makes private streaming patches unnecessary. Reduce one layer at a time in FP32, discard its temporary maps on return and release the entire output before the next sample. No previous sample's GPU tensors survive and no per-batch empty_cache is used. The final post-extraction collection follows the successful Reduced workflow.

The 22.5 MiB payload is not a peak-memory prediction: eager workspace, allocator behavior, retained hidden states/logits and FP32 reductions also matter. The previous ~5.77 GiB Reduced peak and ~375-second extraction are user-reported context, not measurements of this extension. There is no measured T4 runtime/peak for this code yet.

Before each session with pending extraction, execute **five actual attention-enabled shadow forwards**, one per language at staggered source-length quantiles. Include a separate synthetic 128-position stress if all selected real probe inputs are shorter. Probes use no held-out labels or tuning. New probe features become ordinary cache entries; already completed entries are not overwritten. Both peak allocation and peak reservation must stay ≤11 GiB. Every later sample retains the same gate. Missing layers/heads, mask violations, nonfinite features or OOM stop clearly and preserve committed work. No span reduction or attention-feature fallback exists.

CUDA events installed through public forward hooks time attention modules, including Q/K/V projections, RoPE and output projection; this is not a claimed isolated softmax time. Record full attention-enabled forward time, attention-module time, Reduced and attention aggregation times, total seconds, shapes/counts, CPU RAM, disk and GPU allocation/reservation. The post-smoke ETA extrapolates only measured fresh extraction to 3,000 examples; it excludes environment setup, source retrieval, RF training and score-only bootstrap work.

## Single classifier and evaluation

Reuse the exact original RF training and deterministic tree-order scoring functions. Parameters remain 300 estimators, depth 12, leaf minimum 5, balanced class weights, random_state=42, n_jobs=-1. Fit once per run on the entire selected development partition, using all 1,787 columns. Validation is diagnostic only; no refit/search/scaling/selection. The smoke fit checks the same configuration that will be fitted once on all 1,600 development rows for the sole full experiment.

Use the original calibration-only empirical FPR≤1% threshold procedure. Report deployed held-out TPR/FPR separately from diagnostic held-out ROC cutoffs. Full calibration has 100 negatives: 1% empirical steps with substantial uncertainty, not a population FPR guarantee. Report original coverage and language metrics.

Saved Reduced, LUMIA, fusion, static and Adaptive comparisons use exact matched identities/content hashes/labels. Static/LUMIA/fusion are already in the required baseline Dataset; Reduced/Adaptive attachments are optional. No previous attack is rerun and reference AUC constants are never substituted for paired scores. The local Reduced score Parquet independently reproduces **0.7033** on the 1,000 matched held-out samples.

For fixed saved score comparisons, use a preregistered 2,000-draw paired bootstrap, seed 42, resampling within language/membership strata. It estimates percentile 95% intervals for the pooled AUC difference. This is score analysis only, not additional model fitting or model selection, and does not include uncertainty from retraining. More features alone do not establish improvement. Impurity importance is descriptive association, not causal memorization evidence.

## Persistence and one-full-run workflow

Validate all original saved partitions before deterministic smoke selection; never regenerate splits. Transactional DELETE-mode SQLite commits every sample, caching only token plans and scalar features. Cache metadata binds the entire scientific identity and combined schema; incompatible/Reduced-only imports are rejected before copying rows. Compatible smoke-to-full imports preserve features; full-to-full resume retains the same identity. Save source/input hashes, schema hashes, numerical versions, backend, optional-prefix provenance, classifier configuration and output artifacts.

Full mode requires completed compatible smoke evidence. It accepts the smoke cache from CACHE_REUSE_FROM (or finds the conventional writable smoke output), verifies config/counts/metrics, and writes a sealed `smoke_validation.json` proof. This proof travels with full outputs for later full-to-full resume, so a resumed full run need not reattach smoke separately. Science remains fixed between smoke and full.

Archive closed/checkpointed SQLite and top-level JSON/JSONL/Parquet/SQLite/joblib artifacts. Model weights and nested Hugging Face cache directories are excluded. Partial runs have explicit partial ZIPs, never a completed result claim.

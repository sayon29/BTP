# Reduced memTrace methodology

**Reduced memTrace (hidden-state, activation, confidence and context-evolution features)** is a paper-informed reduced implementation, not exact memTrace reproduction. Reference: Makhija et al., [Neural Breadcrumbs, EACL 2026, Section 2](https://aclanthology.org/2026.eacl-long.262.pdf), pp. 5610–5611. The supplied [official repository](https://github.com/amazon-science/NeuralBreadcrumbs-MIA-EACL-2026) returned HTTP 404 on 2026-11-01; no author source files were available for verification. `research_access.json` records access limits. The paper's reported performance is not an expectation for this experiment.

## Frozen identity and representation

Baseline model and tokenizer: `bigcode/starcoder2-3b`, revision `733247c55e3f73af49ce8e9c7949bf14af205928`. Dataset: `AISE-TUDelft/Poisoned-Chalice`, revision `ae3f42011bf55ab2910c83a342caf8a3c02bcd5d`. The saved completed Infilling configuration independently records 30 layers and hidden width 3072. The loaded architecture must match. Seed 42; FP16 frozen weights; inference mode; batch size one; no generation, gradients, KV cache or attention matrices.

Tokenize with the baseline fast tokenizer and `add_special_tokens=True`, right truncation to the first **128 input positions**, bounded also by baseline/model/tokenizer limits. This gives at most 127 predictable targets. No additional BOS is inserted. Record complete source token count, valid input count, selected target count, truncation flag and omitted-token count. Hidden features include every non-padding position, including tokenizer-supplied special positions. Confidence excludes special targets, zero-width offsets and positions without a valid predecessor. Fewer than two valid input positions or no predictable targets fails explicitly; no undefined or nonfinite vector is accepted or silently imputed.

Boundary 0 is embeddings. Use the Hugging Face `hidden_states` tuple, with intermediate pre-block states and the final normalized output at boundary L. Include embedding-to-first-block transition; the final transition includes terminal normalization and must not be interpreted as a pure residual-block update. These representation conventions are fixed, and mock tensor tests check the tuple contract. Actual architecture/API behavior still requires the Kaggle smoke test.

All 30 adjacent transitions are used. Selected boundaries use integer round-half-up of `L * [0,.25,.5,.75,1]`, deduplicated: **[0, 8, 15, 23, 30]**. This uses boundary depth, including embeddings, for the five relative positions.

## Features and numerical definitions

Let valid hidden vectors in token order be h(l,t), t=1..T. All differences, norms, running sums and statistics use FP32. Standard deviation is population standard deviation (`correction=0`); no sample-unbiased estimator. Flatten token and hidden dimensions for activation statistics. Natural logarithms throughout.

A. For each l=0..29:

- D(l,t) = ||h(l+1,t) - h(l,t)||₂.
- C(l,t) = dot(h(l+1,t),h(l,t)) / [max(||h(l+1,t)||₂,1e-8) max(||h(l,t)||₂,1e-8)]. Zero-vector cosine is zero.
- Mean and standard deviation of D, followed by mean and standard deviation of C: **120 features**.

B. At each selected boundary, over all valid activation elements x:

- mean(|x|), sqrt(mean(x²)), std(x), max(|x|), mean(1[|x| < 1e-3]): **25 features**.

C. For eligible target position j, use logits z at position j−1 and target input ID at j. Define Z=logsumexp(z), p(v)=exp(z(v)−Z), NLL=Z−z(target), q₁ and q₂ the top two probabilities, E=Z−sum_v p(v)z(v).

- Mean NLL, std NLL, mean q₁, std q₁, mean(q₁−q₂), mean E, std E: **7 features**.

Confidence uses the final model logits only. Entropy uses log-sum-exp normalization; vocabulary-sized temporaries remain on device and die before the next example. No vocabulary tensors are cached or copied to CPU.

D. At each selected boundary, m(l,t) = sum_{i=1..t} h(l,i)/t. For t=2..T, G(l,t)=||m(l,t)−m(l,t−1)||₂. Mean, std and max G: **15 features**.

Concatenation order: all A transitions in increasing depth; B at increasing selected depths; C; D at increasing depths. Exactly **167 dimensions**. `feature_schema.json` enumerates every exact name and family in order and includes its SHA256. No extra language, length, identity, timing or label features enter the classifier. Counts and resource measurements remain metadata.

## Differences from published memTrace

The paper describes many more transition aggregates (including extrema and locations), normalized surprises, richer activation patterns, attention statistics, token-position features, confidence stability and broader layerwise confidence analysis. This implementation omits attention matrices and every unrequested feature. The paper's cumulative-mean context-change equation, Euclidean/cosine transition definitions and final confidence concepts inform the retained equations. Our activation RMS/absolute mean/std, chosen aggregates, threshold, 128-position cap, five-depth sampling, FP32 reductions, architecture and representation conventions are explicit reduced choices. An author-code near-zero default could not be verified; use the requested 1e-3.

The paper describes standardized features and nested classifier searches/cross-validation with an 80/20 division. Here raw features feed a fixed random forest (trees do not require z-score scaling). Fit only the original 1,600 shadow examples with 300 trees, max_depth=12, min_samples_leaf=5, balanced class weights, random_state=42, n_jobs=-1. No search, validation selection or refit. Probability prediction sums estimators in a fixed FP64 order to eliminate scheduling-dependent summation roundoff. Validation is diagnostic only.

Calibration alone chooses a higher-is-member threshold maximizing empirical TPR among thresholds with FPR≤.01; ties choose the first/highest threshold. If no finite allowable threshold exists, use nextafter(max_score,+∞), rejecting all examples. The 100 full-run calibration negatives permit 1% empirical increments, with substantial uncertainty: this is not a population FPR guarantee. Held-out ROC-AUC and TPR at FPR≤1% use thresholds derived from held-out scores **only as diagnostic summaries**, never for deployment or tuning. Full evaluation uses all 1,000 original held-out identities. Smoke estimates are explicitly preliminary.

Saved comparisons require exact content hashes, identities and labels. Prior baselines keep their own analysis spans. Optional missing, ambiguous or mismatched result files are omitted, never rerun. Feature importances are impurity-based descriptive associations, not causal evidence of memorization; correlated features can share importance.

## Memory and execution

One full model forward per uncached sample, including hidden states and final logits; no redundant transformer forward for independent feature families. FP16 hidden tuple size is approximately 31×128×3072×2 bytes = **23.25 MiB**, and FP16 logits approximately 128×49152×2 = **12 MiB**. These are tensor payload estimates, not measured GPU peaks. Model weights, attention workspace, FP32 temporary allocations and allocator reservations add overhead. Given prior ~6 GiB weights, roughly **6–8 GiB** is an engineering expectation, not a guarantee. The five-shadow-sample probe measures actual allocation/reservation and times before continuation; both peaks must stay ≤11 GiB. Every subsequent fresh batch retains this guard. No per-batch empty_cache; a single post-extraction collection releases unused model allocations before RF fit.

Probe selection uses only shadow identities/languages and source lengths: one per language at increasing source-length quantiles, including the longest in the final language. Complete input token lengths are recorded; if all real samples are short, a separate synthetic 128-position stress check is required before the remaining workload. No held-out labels or performance inform features. Cached probe examples are reused on resume; all fresh batches remain guarded. ETA extrapolates measured extraction to 3,000 samples, excludes setup/source retrieval/RF fitting and is not a deadline guarantee.

Transactional DELETE-mode SQLite stores only plans and compact finite feature vectors. Each key binds all saved identity fields (including labels for integrity), pinned target/tokenizer/configuration, schema hash, numerical version, source hashes and versions. Labels do not enter tokenization or extraction. The scientific identity excludes smoke/full execution selection, so compatible features can be imported from smoke into full. Changed science yields cache misses. Every sample commits independently; partial Parquet, progress and cache survive controlled resource stops. Archive only top-level result artifacts after closing/checking SQLite; never include HF caches or weights.

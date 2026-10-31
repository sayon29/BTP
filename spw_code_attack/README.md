# SPW-inspired code experiment (approximation)

**This package does not implement verified, full published SPW-MIA.** The user explicitly permitted an honestly named fallback when the full equations could not be retrieved. This is that fallback: **SPW-inspired dynamic-halving weighted Bernoulli-JS**, frozen before any validation or held-out results. All existing attacks, source helpers, notebooks and saved partitions are unchanged.

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

## Baseline integration and identities

Unchanged components reused:

* `adaptive_token_attack.data`: complete saved-run loading, identity validation, partition validation and stratified selection.
* `adaptive_token_attack.features.load_frozen_target`: same frozen model loader as notebook 03.
* `shared.kaggle_setup`, `SourceStore`, `Monitor`, `restore_run`, `ensure_model_snapshot`, `metric_report`, atomic JSON writer.

Required files are the existing `shadow_manifest.parquet`, `fusion_fit_manifest.parquet`, `threshold_calibration_manifest.parquet`, `heldout_test_manifest.parquet`, baseline `heldout_test_scores.parquet`, baseline `config.json`, and saved adaptive `heldout_adaptive_scores.parquet`. Never reconstruct splits. Complete originals are validated before selection: unique IDs/content, correct official train/test source positions, labels and roles, no cross-partition identity/hash overlap. Saved full counts are **1600/200/200/1000**. Smoke selects **80/20/20/30**, balanced by language and membership, with **seed 42**. Full retains every saved row and its order. Development and validation receive scores but do not tune the method. Only calibration chooses the deployed threshold.

Notebook 03 may have used a different smoke selection seed. Attach **full adaptive held-out scores**, or a file covering all exact SPW-selected smoke IDs. Missing IDs are an error before target inference; the experiment never silently reduces coverage or substitutes another seed. Full SPW requires full adaptive scores.

`SourceStore.batches` reads only selected sample content from existing Arrow caches or HF streaming, validates hashes and stops after required rows. Streaming may traverse earlier rows/shards to reach a saved source position; no entire training split is materialized/prepared. Cached plans skip repeat text reads. Baseline/API numerical versions must match exactly, following notebook 03's existing check. Baseline scalar losses and token-probability caches are not presumed compatible and are not reused.

## Kaggle efficiency and equivalence

StarCoder2 is frozen, in eval mode, with `torch.inference_mode()` for every decoder/head call. The predictor uses the native StarCoder2 decoder and LM head, no training or auxiliary model. FP16 forward/head, FP32 log-softmax, FP64 probability/weight/divergence reductions. No logits or intermediate full-layer activations are cached.

* Original-prefix probabilities: teacher forcing over the needed initial sequence, native KV-cache chunks of 256. One pass collects all missing original target predictions. Already cached prediction/head results are retained; common earlier tokens may still be processed as context in a **different longer sequence**. KV states are not shared across unrelated examples.
* Shortened-prefix requests: group identical token-ID sequences, collect requested true IDs, then batch only equal lengths, maximum four, no padding. Stop each token dynamically at the first mismatch. No approximate KV reuse across distinct shortened prefixes.
* Each canonical prefix cache stores greedy ID/log probability, queried true-token log probabilities, and its terminal hidden vector. FP16 hidden vectors are compressed **losslessly** with zlib/base64; float32 is retained losslessly for CPU tests. A new true-token query for an existing prefix needs only a head projection of the cached terminal state, no transformer call. LM head projection uses one row both initially and on extension. Probability cache extension may only add entries, never alter established values.
* Prefix keys bind exact ID sequence, zero-based position policy, immutable model/tokenizer versions, tokenizer serialization, precision, native model config/backend, all fixed method settings and source hashes. Sample/result keys additionally bind original identity/content and exact analysis plan. Labels are excluded. Repeated identical prefixes share inference across samples/languages without mixing their identities or labels. Smoke/full share compatible keys.
* Mathematical equivalence is tested against a straightforward independent-prefix implementation using deterministic CPU synthetic distributions; optional tiny random CPU StarCoder2 tests check shifted logits, teacher forcing, equal-length batching and cached-head extensions. Native GPU FP16 batch/chunk kernels can produce ordinary roundoff differences; no bitwise GPU equivalence claim is made. The optimized algorithm does not change shortening/weight/score rules for speed.

Caching terminal vectors and token-level traces can consume **several GiB**. Disk/RAM/session budget checks occur before model/cache operations and outputs; allowed model downloads and cache imports are allocation checked. Nothing is auto-deleted. T4 memory is bounded by short batches and native original chunks, but full mode may exceed a single session's time/disk budget and require persisted resume. Full mode has no promised runtime in the absence of actual model measurements.

`forward_calls.jsonl` logs started/finished operations with actual transformer and LM-head forward counts and synchronized model time, including failed operations when Python cleanup executes. `progress.json` records each sample's runtime/calls. `timing.json`, metrics and run manifest report session calls, per-example times, cache hits and persistent call audit. A hard-killed operation may never log completion: it is **flagged as unknown**, not silently counted as zero or claimed exact. Imported caches do not bring the previous run's call history; result computation times remain descriptive and are not an estimate of total historical GPU cost.

## Checkpointing and crash recovery

`spw_cache.sqlite`: checksummed JSON, SQLite synchronous FULL, DELETE rollback journals, one committed prediction/cache extension at a time and a separate complete-sample checkpoint. All original/shortened true and greedy probabilities, greedy IDs, accepted lengths, proposal traces, weights and divergence terms are auditable in the cache. Original sample tokens/plans are retained there too. It is a data/source-provenance cache, not model training state.

On interruption, rebuild control flow deterministically from committed prediction records. A cached full result causes zero model calls; missing decisions alone are evaluated. A query killed **before commit** must be repeated; queries with committed results are reused. JSON/Parquet exports use temporary files and atomic replacement. `run_manifest.status` distinguishes preparing/scoring from complete. Preserve rollback journals if an interruption leaves one. Stop active cells before copying a database. Save the whole OUT directory with Kaggle **Save Version with outputs** before ending a session.

Same-run resume: attach saved OUT, keep MODE/RUN_ID/scientific identity, set `RESUME_FROM` to saved OUT. Changed identity fails. Smoke → full: new full RUN_ID, `RESUME_FROM=''`, optional `CACHE_REUSE_FROM='/kaggle/input/.../spw_cache.sqlite'`. Only matching cached data is used; no thresholds are transferred. Source changes invalidate keys. Operational cache/source paths and resource budgets may move without changing scientific identity. Source ZIP and notebook pins are rebuilt together.

## Evaluation

`S` direction is fixed before labels. On the separate calibration pool, choose the highest-TPR empirical threshold satisfying FPR <=.01, with conservative descending-threshold tie order. If no finite threshold qualifies, use the next representable value above the maximum calibration score, predicting none. Apply this **single global threshold** unchanged to held-out rows and each language.

Report calibrated held-out TPR/FPR separately from `heldout_diagnostic_roc_only`, which contains ROC-AUC, maximum TPR at empirical FPR <=1%, its achieved FPR and diagnostic threshold. Diagnostic held-out thresholds never deploy or choose the sign/score. No interpolation. Smoke and language-level low-FPR estimates are coarse and explicitly resolution limited.

Compare `spw_inspired_score`, saved static SERSEM (`greybox_score`), adaptive SERSEM (`adaptive_score`), LUMIA and baseline fusion on exactly matched IDs, content hashes, labels and languages; adaptive also validates complete source/role provenance. Reject missing, duplicate, changed and nonfinite rows. Never silently inner-join to fewer IDs. Report per-language AUC and ROC, globally calibrated per-language TPR/FPR, and paired metric differences. No new fusion, significance test or confidence claim. Baseline methods keep their full original spans while this experiment analyzes 256 eligible tokens, so identity pairing is not a claim of span equivalence.

Full mode requires 1000/1000 held-out coverage. Smoke has 30/30 selected coverage and 30/1000 original coverage (3%), clearly preliminary. Sample time and actual call statistics are included in metrics; development/validation/calibration scoring cost is included in total session counters.

## Files and generated artifacts

Package: `__init__.py`, `method.py`, `inference.py`, `engine.py`, `data.py`, `runner.py`, `evaluation.py`, `cache.py`, `package_source.py`, `requirements-kaggle.txt`, `reproducibility.json`, `research_access.json`, `source_manifest.json`, this README. Separate test suite: `tests/test_spw.py`. Notebook: `notebooks/05_spw_code.ipynb`. Upload: `notebooks/spw_kaggle_source.zip`.

Kaggle produces:

* `config.json`, `reproducibility.json`, `research_access.json`, `source_version_hashes.json`, `input_paths.json`.
* `metrics.json`, `spw_scores.parquet`, `matched_comparison.parquet`.
* `spw_cache.sqlite`, `run_manifest.json`, `threshold.json`, selected partition snapshots.
* `timing.json`, `forward_calls.jsonl`, `progress.json`, resource logs and model-cache setup report.

No real-data metrics/scores/cache are fabricated locally; those are generated by manual Kaggle inference.

## Upload and exact configuration

From repository root:

```bash
python3 -m spw_code_attack.package_source
python3 -m unittest discover -s tests -p test_spw.py -v
```

Upload `notebooks/spw_kaggle_source.zip` as a Kaggle Dataset, import notebook 05, enable Tesla T4 and Internet for streaming. The source bootstrap and protected dependency installer follow the working adaptive notebook, including explicit pinned HF cache setup and version validation. The dependency file exactly matches existing baseline pins; Kaggle scientific/CUDA packages are protected from replacement. No expensive local model inference is needed. Rebuild the ZIP after source/test/spec/README edits; only notebook 05's source pin is updated.

First configuration cell, common settings:

```python
UPLOAD_SOURCE = INPUT_ROOT / 'your-spw-source-dataset'
BASELINE_RUN = '/kaggle/input/your-baseline-dataset/path/to/run'
ADAPTIVE_SCORES = '/kaggle/input/your-adaptive-dataset/path/heldout_adaptive_scores.parquet'
MODEL_CACHE_DIR = '/kaggle/input/your-target-cache'  # HF cache root, as in notebook 03
SOURCE_ACCESS = 'streaming'
MODEL_LOCAL_FILES_ONLY = True
ALLOW_MODEL_DOWNLOAD = False
DISK_RESERVE_GIB = 3.
SESSION_BUDGET_MINUTES = 480
```

An empty `MODEL_CACHE_DIR` inherits the saved baseline path. If intentionally downloading the pinned model on Kaggle, set `ALLOW_MODEL_DOWNLOAD=True`; the helper verifies the prepared local snapshot before inference. For offline source access set `SOURCE_ACCESS='cached'` and `DATASET_CACHE_DIR` to an existing HF Arrow cache. No full dataset download/preparation is called. Resolve ambiguous input discoveries with explicit paths/`PATH_OVERRIDES`.

Smoke:

```python
MODE = 'smoke'
SMOKE_SIZES = dict(shadow=80, validation=20, calibration=20, heldout=30)
RUN_ID = f'spw-inspired-code-{MODE}-v1'
RESUME_FROM = ''
CACHE_REUSE_FROM = ''
```

Full:

```python
MODE = 'full'
RUN_ID = f'spw-inspired-code-{MODE}-v1'
RESUME_FROM = ''
CACHE_REUSE_FROM = ''  # optionally saved smoke spw_cache.sqlite
```

Run the configuration cell again after changing MODE so OUT updates, then run subsequent cells in order. Use full saved adaptive scores for full, and preferably also for smoke. Seed 42, 256 targets, shortening/weight/divergence rules, score direction, chunk/batch sizes are fixed in the hashed specification and must not be tuned on results.

## Lightweight CPU validation

The suite tests known Bernoulli-JS values/endpoints, exact weights, dynamic shortening/initial-prefix cases, original/short target spans, special/empty-offset masking, optimized vs straightforward numerical equivalence, equal-length batching, deduplication, cached terminal extension, immutable probabilities/checksums, version invalidation, crash recovery, no repeated calls for valid results, label independence, source/split leakage rejection, exact comparison identities, calibration-only thresholds, ties, low-FPR coverage, call accounting and portable orchestration. Optional PyTorch tests instantiate a tiny random CPU StarCoder2 without downloads. If PyTorch is absent locally that test skips; Kaggle setup supplies it and the notebook runs the suite before pretrained inference. The notebook syntax check skips inside the ZIP because its notebook is a separate artifact.

# Infilling Score on frozen StarCoder2-3B

Standalone source-code adaptation of Raoof et al., ICLR 2025. Actual target inference runs manually on Kaggle, never during local development. Existing attacks/notebooks/helpers and saved data remain unchanged.

## Verified mathematical source

Read [paper §§3.1–3.2, Eqs. (5)–(12)](https://proceedings.iclr.cc/paper_files/paper/2025/file/5709163243753c9c9ab7f4b4e5a8766d-Paper-Conference.pdf), experiment settings, appendices A/C/D, and the [author repository at commit `54ee252e378afffe09f4dbde5c2ca3adbf8ccfe0`](https://github.com/neginraoof/infilling-score/tree/54ee252e378afffe09f4dbde5c2ca3adbf8ccfe0). Exact file/blob provenance and implementation decisions are in `research_provenance.json`.

For token `x[t]`, let `c=x[:t]`, `g=argmax_v P(v|c)`, and `f` be the selected following true tokens. Bayes gives

```
P(v | c,f) = P(f | c,v) P(v | c) / Z(c,f)
R_t = P(x[t] | c,f) / P(g | c,f)
    = P(f | c,x[t]) P(x[t] | c) / [P(f | c,g) P(g | c)]
log R_t = log P(x[t]|c) - log P(g|c)
          + sum_j [log P(x[j]|x[:j]) - log P(x[j]|altered[:j])]
```

`altered` changes **only** `x[t]` to `g`. The partition `Z` cancels: no enumeration of alternative vocabulary entries is needed in target queries. The ratio may exceed one; no clipping to `[0,1]` is valid. Standard vocabulary softmax/moment reductions still cost O(vocabulary size); it is the **number of model queries**, not all arithmetic, that avoids vocabulary-sized enumeration.

The published detection statistic standardizes the causal components before aggregating. Write `l_j(v)=log P(v|x[:j])`, `p_j(v)=exp(l_j(v))`, and use the original prefix's predictive moments

```
mu_j    = sum_v p_j(v) * l_j(v)
sigma_j = sqrt(sum_v p_j(v) * (l_j(v) - mu_j)^2)
z_j(v)  = (l_j(v) - mu_j) / max(sigma_j, 1e-6)
```

Define `z_alt_j` analogously from the **altered prefix's own distribution**. The implemented single score is

```
s_t = z_t(x[t]) - z_t(g)
      + sum_{j in F_t} [z_j(x[j]) - z_alt_j(x[j])]
K   = floor(0.2 * number_of_eligible_positions)
S(x)= arithmetic_mean(the K smallest s_t)
```

**Higher `S` is member-like**, fixed from Eq. (12), not inferred from any labels. No raw-ratio/normalized-score selection occurs. Raw `log R_t` is saved only as an audit feature; evaluation uses `infilling_score=S` exclusively. In general, the normalized statistic is **not numerically equal** to raw `log R_t`; using just the raw ratio would omit published normalization. Likewise, this is not a ratio of sequence NLLs, fill-mask accuracy, a FIM prompt, or a heuristic suffix probability. StarCoder2's native causal decoder/head supplies every component; no bidirectional head, retraining or reference model is assumed.

### Resolving paper/source discrepancies explicitly

The [author's optimized implementation](https://github.com/neginraoof/infilling-score/blob/54ee252e378afffe09f4dbde5c2ca3adbf8ccfe0/infilling_score/optimizations/infill.py) computes separate original/altered moments at each future position. This resolves Eq. (10)'s reused moment symbols. Appendix D instead reuses the initial position's moments in its future loop, and its final pseudocode omits the averaging divisor printed in Eq. (11). We follow the **verified author code and Eq. (11)'s mean**, not those inconsistent pseudocode lines. No claim of literal agreement with every printed pseudocode line or bitwise author-run results is made.

Pinned author details retained:

* Original causal greedy token, using deterministic smallest-index `argmax`, never a generated rollout.
* Context-specific predictive probability-weighted moments; not unweighted logit variance.
* Five future tokens and a **20%** bottom mean, chosen from the main experimental configuration before evaluating data. Appendix A also discusses 10%; no alternative is tried here.
* Author numerical floor `min_sigma=1e-6` from [`utils/constants.py`](https://github.com/neginraoof/infilling-score/blob/54ee252e378afffe09f4dbde5c2ca3adbf8ccfe0/infilling_score/utils/constants.py). Optional log-probability/score clipping is off. No nonfinite values are replaced with member-like zero scores.
* The exact end handling below, including partial and empty future windows. No silently shortened-target cohort requiring five future tokens at every position.

## Span selection, eligibility and boundaries

One frozen policy for Go, Java, Python, Ruby and Rust: baseline fast tokenizer, pinned revision, `add_special_tokens=True`, right truncation, attention/special masks and character offsets. Take the **first contiguous 128 encoded input IDs**, capped also by existing baseline/positional/tokenizer limits. Native tokenizer-added BOS/EOS, if any, count toward those IDs. Do not append a BOS, EOS, FIM delimiter, mask, suffix prompt or outside context. Internal excluded targets are retained in context; no filtering and retokenization of source strings.

A scoring position `t` has `t>0`, real attention on it and its predecessor, non-special mask, and nonempty character offset. This reuses the adaptive eligibility predicate without AST/category weighting. All such positions in the span are processed. Position zero supplies initial context because there is no LM distribution for an empty tensor. If a native BOS was added, it naturally serves as context. This special/offset eligibility is an explicit code-domain adaptation; the author text implementation scores all predictable IDs.

For span length `N`, the pinned code's logit-index loop is `i=0..N-2`, target `t=i+1`; its future loop is `j in range(i,min(i+m,N-3))`, true target `j+2`. Our zero-based translation is exactly

```
F_t = {t+1, ..., min(t+5, N-2)}
```

The final input token is never a **future** term, while the final two scoring targets have no future terms. They are still included through their causal standardized difference. This slightly unusual rule comes from both [optimized and straightforward author code](https://github.com/neginraoof/infilling-score/blob/54ee252e378afffe09f4dbde5c2ca3adbf8ccfe0/infilling_score/models/detector.py); it is not an invented complete-window boundary. The synthetic suite checks the loop translation at every position.

A 128-ID span ordinarily yields **127 eligible scoring tokens**, not 128, because the first ID is unpredicted; specials/empty offsets can reduce it further. Record `span_tokens`, `effective_analysis_tokens`, `fewer_than_128_eligible_tokens`, `short_span`, per-boundary future counts, exact ID hashes and truncation counts. A full-length span is distinguished from a genuinely short source sample even when both have fewer than 128 eligible targets. Short sources retain all their IDs; never extend them to another span length. The fixed m remains 5 even on short sources, using the same author boundary rule.

`N<3` is invalid as in the author's entry check. If `floor(.2 * eligible_count)==0`, record an **invalid sample with null score**, rather than the author's orchestration fallback of assigning zero. Nonfinite final statistics are also invalid, never silently omitted or replaced. OOM, budget stops and cache corruption raise for recovery instead of being classified as sample invalidity. Invalid rows stay in the score table and coverage report. No alternate score is substituted.

The 128-token-ID budget/domain selection is an adaptation: the paper also evaluates longer contexts, and its benchmark subset lengths/tokenization do not imply this exact StarCoder2 span. It is not an exact reproduction of the paper's experimental setup or reported performance.

## Queries and implementation

`method.reference` is an intentionally simple independent-prefix calculation. For each real target prefix, it obtains a distribution and sufficient statistics. For each changed token it re-evaluates future prefixes independently. `bayes_log_ratio` explicitly enumerates a **tiny synthetic vocabulary only**, demonstrating partition cancellation numerically. Production never calls this enumerator.

The optimized path:

1. One original teacher-forced decoder pass collects all required true/greedy log probabilities, greedy IDs, means and sigmas. Causal alignment is `hidden/logit[t-1] -> input_id[t]`.
2. Reuse the same original statistics for every target. If greedy equals truth, the altered branch is identical and its contribution is exactly zero, so skip that query. Targets with no future use only the original distribution.
3. For other targets, change exactly that token, retain true subsequent tokens, and trim after the last required future label. This does not affect any earlier causal prediction. Batch similar lengths in groups of four, using right padding with an attention mask and explicit zero-based positions. No tokenizer resize or new padding token; use existing pad/EOS ID or zero only in masked padding slots.
4. Native StarCoder2 decoder and LM head run under `torch.inference_mode()`, eval, all parameters frozen. Project only requested hidden rows in blocks of 32. Reduce transient vocabulary logits immediately to sufficient statistics. No complete logits, hidden vectors or per-layer activations are saved.
5. FP16 model/head, FP32 log-softmax, stable **FP64 central population variance** and score reductions. This is mathematically equivalent to the probability-weighted variance definition but avoids cancellation in `E[l^2]-E[l]^2`. It is not bitwise equal to the repository's lower-precision arithmetic. CUDA batch/attention kernels can introduce ordinary floating-point differences; CPU reference agreement is tested with explicit tolerances.

Worst-case query count is linear in eligible positions, with many queries handled per batched transformer forward; no vocabulary-sized candidate loop. `forward_calls.jsonl` and timing separately count actual native decoder and LM-head forwards, synchronized operation wall time, per-example/session cost and cache hits. Hard-killed operations without a completion record are marked **unknown**, not falsely reported as exact zero cost. Importing a warm cache does not import previous call logs, so reports clearly distinguish this run's measured calls from stored result timing.

## Saved manifests and Kaggle conventions

Reuse unchanged `adaptive_token_attack.data` loading, complete partition validation and stratified selection; `adaptive_token_attack.features.load_frozen_target`; `shared.kaggle_setup`, `SourceStore`, `Monitor`, `restore_run`, `ensure_model_snapshot`, `metric_report` and atomic JSON writer. Source hashes bind these exact files. No existing attack runs or changes.

Required saved baseline files:

| Role | Filename | Full | Smoke |
|---|---|---:|---:|
| Development | shadow_manifest.parquet | 1600 | 80 |
| Validation | fusion_fit_manifest.parquet | 200 | 20 |
| Calibration | threshold_calibration_manifest.parquet | 200 | 20 |
| Held-out | heldout_test_manifest.parquet | 1000 | 30 |

Also attach baseline `config.json`, `heldout_test_scores.parquet`, and adaptive `heldout_adaptive_scores.parquet`. Complete originals are validated before smoke selection. Exact source-row/sample ID, content hash, language, role, membership and train/test disjointness checks remain active. No new split generation. Seed 42 balances smoke rows by language and membership, independently of inference labels. Full preserves every original row/order. The saved adaptive smoke seed can differ: full adaptive scores are recommended to cover all selected seed-42 IDs. Missing baseline/adaptive identities fail before inference, not via a silent inner join.

Source access is streaming or existing cached Arrow, bounded to needed source identities and stopping after they are found. Streaming can traverse earlier rows/shards to reach a saved source position; no full train-dataset preparation/download is forced. Content SHA256 is checked against manifests. Complete cached span plans skip source access. No baseline scalar loss or token statistics are assumed compatible with this attack.

Notebook setup mirrors working notebook 03: verified portable ZIP/extracted source before imports, protected numerical/CUDA packages, the same pinned dependency specification and exact saved baseline/API-version equality. Model cache preparation is explicit and disk checked. Default model download is disabled. Enable T4 and Internet when streaming. Actual T4 model execution and throughput remain unmeasured until the notebook runs; full mode can need saved-output resume across sessions.

## Checkpoint and resource behavior

`infilling_cache.sqlite` uses checksummed JSON records, synchronous FULL transactions and DELETE rollback journals. Per-query keys bind exact token IDs, scoring positions, source implementation hashes, full fixed configuration, immutable model/tokenizer revisions, tokenizer serialization, versions, native model config/backend and precision. Sample/result keys additionally bind original sample/content/source identity and span plan. Labels do not enter inference keys or query construction. Equivalent exact query contexts share cached results across samples; smoke/full share compatible keys.

Every completed original/altered query commits separately. Replaying a partial sample uses these records; a valid complete sample causes zero further model calls. A call interrupted **before commit** must be repeated. Query statistics and per-sample token scores, raw log ratios, greedy IDs and bottom-position selection remain auditable in SQLite. Operational paths/budgets may move; scientific/source changes invalidate keys. Disk/RAM/session guards precede operations and exports, allowed model downloads/cache imports have allocation checks, and no cache is deleted automatically.

JSON/Parquet exports replace temporary files atomically. Run status distinguishes preparing/scoring/complete/complete-with-invalid. Save the **entire OUT** using Kaggle Save Version **with outputs** before ending the session. Stop active cells before copying SQLite, and preserve any rollback journal left after interruption. Same-run resume copies trusted outputs without overwriting different files. Cache import into a new full run reuses only exact matching entries and never transfers a calibrated threshold.

## Evaluation and generated outputs

Threshold selection uses only valid calibration rows and empirical FPR <=1%, maximizing TPR with conservative ROC threshold order. Ties that admit no finite qualifying threshold use the next representable value above maximum calibration score. Apply one global threshold unchanged to held-out/per-language rows. If both valid calibration classes are not present, report threshold unavailable.

`heldout_diagnostic_roc_only` reports ROC-AUC, maximum TPR at FPR <=1%, achieved FPR and diagnostic thresholds; these held-out-label diagnostics never deploy, flip sign or select a score. Per-language ROC/AUC and calibrated TPR/FPR are separate. Smoke/per-language low-FPR estimates are coarse, with resolution flags in ROC reports.

All saved held-out identities must have baseline and adaptive scores **before** any explicit invalid filtering. Pair Infilling, static SERSEM (`greybox_score`), adaptive (`adaptive_score`), LUMIA and saved fusion on the exact same valid IDs/content/labels. Report paired metric differences without an implied significance claim. The saved baseline methods retain their original analysis spans, while this attack uses 128 IDs. Identity pairing does not imply span equivalence.

Full processes 1000/1000 held-out rows. Coverage separately reports processed, valid matched, invalid and per-language counts. Invalid score rows stay in `infilling_scores.parquet`, and all five methods use the same explicitly reported valid cohort. AUC is unavailable if valid labels lack both classes; no scores or performance are fabricated. Smoke processes 30 selected held-out rows and reports its fraction of the original 1000. All selected development/validation/calibration rows also receive a result, without fitting or method selection.

Manual Kaggle execution produces:

* `config.json`, `metrics.json`, `infilling_scores.parquet`, `matched_comparison.parquet`.
* `run_manifest.json`, `infilling_cache.sqlite`, `threshold.json`.
* `reproducibility.json`, `research_provenance.json`, `source_version_hashes.json`, `input_paths.json`, selected manifest snapshots.
* `timing.json`, `forward_calls.jsonl`, `progress.json`, resource logs and model-cache setup report.

Real-data metrics/Parquet/cache outputs are intentionally not fabricated locally. Source deliverables are this package, `tests/test_infilling.py`, notebook 06 and its upload ZIP.

## Prepare upload and run

From repository root:

```bash
python3 -m infilling_code_attack.package_source
python3 -m unittest discover -s tests -p test_infilling.py -v
```

Upload `notebooks/infilling_kaggle_source.zip` as a Kaggle Dataset. Import `notebooks/06_infilling_code.ipynb`, enable Tesla T4, attach source/baseline/adaptive/model cache Datasets, and run cells in order. The source manifest and notebook pin are rebuilt together; only notebook 06's pin is modified. Rebuild after editing this package/spec/tests/README.

Common first-cell paths:

```python
UPLOAD_SOURCE = INPUT_ROOT / 'your-infilling-source-dataset'
BASELINE_RUN = '/kaggle/input/your-baseline-dataset/path/to/run'
ADAPTIVE_SCORES = '/kaggle/input/your-adaptive-dataset/path/heldout_adaptive_scores.parquet'
MODEL_CACHE_DIR = '/kaggle/input/your-model-cache'  # HF cache root, as in notebook 03
SOURCE_ACCESS = 'streaming'
MODEL_LOCAL_FILES_ONLY = True
ALLOW_MODEL_DOWNLOAD = False
DISK_RESERVE_GIB = 3.
SESSION_BUDGET_MINUTES = 480
```

An empty model cache path inherits the baseline path. Intentionally downloading the pinned target on Kaggle requires `ALLOW_MODEL_DOWNLOAD=True`; the helper then verifies a local snapshot. For offline source access use `SOURCE_ACCESS='cached'` and an existing Arrow `DATASET_CACHE_DIR`. Ambiguous saved runs require explicit `BASELINE_RUN`/`PATH_OVERRIDES`; ambiguous adaptive files require `ADAPTIVE_SCORES`.

Smoke:

```python
MODE = 'smoke'
SMOKE_SIZES = dict(shadow=80, validation=20, calibration=20, heldout=30)
RUN_ID = f'infilling-code-{MODE}-v1'
RESUME_FROM = ''
CACHE_REUSE_FROM = ''
```

Full:

```python
MODE = 'full'
RUN_ID = f'infilling-code-{MODE}-v1'
RESUME_FROM = ''
CACHE_REUSE_FROM = ''  # optionally point to saved smoke infilling_cache.sqlite
```

Re-execute the configuration cell after changing MODE so OUT updates. Full requires saved adaptive scores for every original held-out ID. Same-run recovery sets `RESUME_FROM` to attached saved OUT with the same mode/RUN_ID/configuration. Smoke-to-full uses a new full run and `CACHE_REUSE_FROM` rather than RESUME_FROM. Seed 42, contiguous span 128, future tokens 5, bottom fraction .2, normalization, boundaries and direction are frozen in the hashed specification; no grid is supplied.

Package structure: `method.py` (derivation/reference), `inference.py` (GPU sufficient statistics), `engine.py` (query/result checkpoints), `data.py` (span plans), `evaluation.py`, `runner.py`, `cache.py`, `package_source.py`, `__init__.py`, requirements and hashed JSON specifications/provenance.

## Synthetic validation

Tests cover explicit tiny-vocabulary Bayes normalization versus the causal four-term expansion; ratios above one; probability-weighted moments and sigma floor; separate altered-context normalization; exact author-loop translation; bottom-fraction **mean**; causal alignment; optimized/reference numerical agreement; cropped/right-padded query equivalence; special/BOS-like and short-span cases; source identities/partition leakage; label-independent inference; checksum/version invalidation; partial-query recovery; no extra calls for committed queries/results; exact paired evaluation; invalid/all-invalid coverage; calibration-only thresholds and audit counts.

The optional Torch test builds a tiny **random CPU StarCoder2**, without downloading pretrained weights, and exercises shifted logits, moments, masked right padding, freezing and optimized/reference score agreement. It skips when Torch/Transformers are absent locally; Kaggle dependencies provide them and the notebook runs the suite before target inference. The separate notebook syntax test skips inside the source ZIP, where the notebook is not embedded. No expensive model inference is part of local QA.

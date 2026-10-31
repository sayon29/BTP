# Adaptive SERSEM token weighting

This branch learns interpretable category weights on **cached frozen-target signals**. It leaves the baseline and neighbourhood branches unchanged. Use [notebook 03](../notebooks/03_adaptive_sersem.ipynb) on Kaggle; no target models, LUMIA probes or GPU work are needed to develop/test the CPU optimizer.

## Mathematics and provenance

The reference is `shared/weighted_anomaly.py`, our **SERSEM Section 2.1 reconstruction**, not recovered author scoring code. The pinned upstream `ASTExtractor.py` supplies parsers, dictionary and identifier splitting; `LinterExtractor.py` supplies linter positions. Those source files and LUMIA have SHA256 provenance in `shared/upstream_manifest.json`. LUMIA and the experimental baseline fusion are compared through saved scores only. No pretrained target or LUMIA training is performed here.

Extraction preserves the reference tokenizer/revision, `add_special_tokens=True`, inherited truncation/context/dtype and attention backend. For target token t, vocabulary moments are uniform **population** moments of the raw float32 logits at t-1:

```
z_t = (logit[t-1, token_t] - mean_v(logit[t-1,v])) / max(std_v(logit[t-1,v]), epsilon)
signal_t = sigmoid(z_t)
S = sum_t(weight_t * signal_t) / sum_t(weight_t)
```

This is neither causal NLL nor softmax-normalized token probability. First/unpredicted tokens, specials and empty character offsets have zero weight. Extraction uses one unpadded sample per forward, matching the original baseline. Logit chunks use its configured epsilon/chunk length and float32 accumulation order. No vocabulary logits are persisted.

The existing reference exposes numeric character weights rather than category labels. Our `Categorizer` attributes those weights to these supported triggers:

| Category | Original weight |
|---|---:|
| standard/background | 0.1 |
| regular identifier | 1 |
| long identifier | 3 |
| string | 5 |
| linter anomaly | 5 |
| multilingual identifier | 10 |
| comment | 10 |
| psychological marker | 10 |

The largest numeric weight wins at each character. Equal-weight ties use the **last** category in the table. Token weights use the original maximum over their character spans; equal-weight categories within a token use the same tie rule. This attribution convention is ours. Each character's attributed weight is checked for exact equality with the actual `StructuralWeights.char_weights` result; drift raises an error. This includes its original identifier boundaries, dictionary heuristics, unrestricted string-length weighting, psychological patterns and linter neighborhoods. We do not substitute the upstream AST extractor's different numeric mask.

The optimizer learns `w = softmax(theta)` with one parameter per category. A shared positive `softplus(rho)` scale and scalar bias give classification logits `scale * (S - 0.5) + bias`. The prior is the normalized original category-weight vector; normalization cancels in S, so it exactly reproduces the static aggregation. Initialize near log(original weights), with small seeded noise. Penalize `KL(prior || w)` plus a small calibration L2 penalty. A positive scale makes raw S and calibrated probabilities rank samples in the same membership direction. No per-language models, neural token classifier, attention network or fusion are added.

Training uses full-batch **NumPy Adam**, analytical gradients, float64 arithmetic and stable `mean(logaddexp(0, logits) - label * logits)` BCE, mathematically equivalent to `BCEWithLogitsLoss`. Finite-difference CPU tests verify all parameter gradients. Only eight category logits, rho and bias are optimized; StarCoder2 is never involved in this stage. Category counts and signal sums are exact sufficient statistics for S. Missing categories contribute zero to both numerator and denominator; an all-empty sample raises an error.

## Saved data and runtime checks

Set `BASELINE_RUN` or the notebook's six `PATH_OVERRIDES`:

| Path key | Default filename | Required role/count |
|---|---|---|
| shadow | `shadow_manifest.parquet` | development, 1,600 |
| validation | `fusion_fit_manifest.parquet` | early stopping, 200 |
| calibration | `threshold_calibration_manifest.parquet` | threshold selection only, 200 |
| heldout | `heldout_test_manifest.parquet` | final evaluation, 1,000 |
| scores | `heldout_test_scores.parquet` | saved LUMIA/grey-box/fusion scores |
| config | `config.json` | original pinned baseline configuration |

Discovery scans `/kaggle/input` for saved run directories. Multiple candidates require explicit selection, including when saved adaptive outputs also contain a shadow manifest. Missing files are listed together. Importing modules does not look for data. No function reconstructs splits. Development/validation/calibration must use official train rows; held-out uses official test rows. Validate unique IDs/content hashes, correct role and membership labels, source positions and disjoint IDs **and hashes** across all four pools. Original shadow internal roles (`shadow_train`, `shadow_early_stop`, `shadow_layer_selection`) are accepted together as the saved 1,600-example pool.

The held-out score file must cover every frozen held-out identity and include `lumia_score`, `greybox_score`, `fusion_score`, labels, language and content hashes. Additional baseline rows are ignored; no held-out rows are silently dropped. Default production sizes are enforced. Both modes load and validate every original manifest and all 1,000 saved held-out score identities first. Full mode enforces the production sizes. Smoke mode then selects the configured smaller cohorts from those validated originals and enforces the selected counts, identities and disjointness at later stages.

## Kaggle run

1. Locally build the small source package with `python3 adaptive_token_attack/package_source.py`. Upload `notebooks/adaptive_kaggle_source.zip` as a Kaggle Dataset and upload notebook 03. This command only updates adaptive artifacts.
2. Attach the original saved baseline run and existing HF target/dataset caches. Edit `UPLOAD_SOURCE`, `BASELINE_RUN` (or per-file paths), `DATASET_CACHE_DIR`, and `MODEL_CACHE_DIR`. Start a fresh kernel. Restore the saved baseline dependency versions; the notebook rejects a different API/numerical environment.
3. Choose `MODE="smoke"` (default) or `MODE="full"`, then run setup and manifest loading. Both require exactly 1,600/200/200/1,000 original saved rows. Missing development files stop execution here, without generating alternatives. Smoke selects subsets only after validating these complete originals; full retains every original row in its original order.
4. Run feature extraction on Kaggle GPU. The default uses attached caches and permits no target download. If explicitly needed on Kaggle, set `ALLOW_MODEL_DOWNLOAD=True`; target identity still comes from the saved config. `SOURCE_ACCESS="streaming"` is an explicit alternative to the existing Arrow cache. The pipeline never calls `SourceStore.manifests()` or prepares a new split.
5. Up to **150** frozen single-example forwards in default smoke mode, or **3,000** in full mode, extract compact token data; every committed sample is reused on resume. An interrupted forward not yet committed may repeat. An extraction failure is logged and raised; training cannot use an incomplete cache. A complete cache skips target loading and source reading. Set `RUN_EXTRACTION=False` for cache-only CPU work. For a new hyperparameter experiment, use a new `RUN_ID` and set `FEATURE_CACHE_PATH` to an attached compatible `token_features.sqlite`; `FEATURE_CACHE_READ_ONLY=True` is the default for that path. A complete attached cache requires no target inference; an incomplete read-only cache lists missing IDs before loading the target. Keep that external cache attached on subsequent resumes.
6. Verify reconstructed static scores before training. CPU reconstruction must agree with extraction-time GPU scores within absolute 2e-6. Both must reproduce the actual saved grey-box scores for every selected held-out ID (30 in default smoke, 1,000 in full) within absolute 2e-6 for float32 or 2e-5 for float16/bfloat16 (relative tolerance zero). Differences write per-sample diagnostics and **block fitting**. Investigate model revision, tokenization, source text, dtype/backend, dependencies or categorization; do not auto-relax tolerance. Only saved scores, not held-out labels, are used in this fidelity check. Development samples additionally get internal CPU/GPU reconstruction checks; historical development grey-box scores were not required to be saved by the baseline.
7. Run CPU fitting, then the separate validation/freezing cell. Run calibration on its selected separate examples (20 in default smoke, 200 in full), then final evaluation once weights/threshold are frozen. Do not select hyperparameters from held-out performance.
8. Read `metrics.json`: all four overall/per-language ROC reports use the exact same selected held-out IDs (30 in smoke, 1,000 in full). Smoke reports, calibrated metrics and score exports carry explicit preliminary/resolution-limited labels. Adaptive raw-score ROC-AUC and TPR at empirical FPR <=1% are separate from TPR/FPR at the calibration-selected threshold. Inspect category weights/counts and report coverage. No adaptive fusion is performed.

Default hyperparameters: baseline seed, 500 epochs maximum, learning rate .03, patience 40, minimum validation BCE improvement 1e-6, KL coefficient .01, calibration L2 coefficient 1e-4, initial scale 4, initialization noise .001, gradient norm cap 10. These are configurable in `TRAINING_OPTIONS`. Early stopping uses validation BCE, excluding regularization. Hyperparameters and source/data/cache identities are bound into checkpoints; changing them requires a new run ID.

### Smoke first, then full

Start with:

```python
MODE = "smoke"
SMOKE_SIZES = dict(shadow=80, validation=20, calibration=20, heldout=30)
RUN_ID = "adaptive-sersem-smoke-v1"
RUN_EXTRACTION = True
RESUME_FROM = ""
FEATURE_CACHE_PATH = ""
FEATURE_REUSE_FROM = ""
```

Edit the source/baseline/model/dataset paths and run notebook cells in order on Kaggle GPU. Only these **150 selected samples** are extracted, trained/validated/calibrated/evaluated. The selector ranks saved IDs by SHA256 of the baseline seed, partition, language and membership; it selects independently **within** each original partition, never creates new splits. Each language gets an equal number of members and non-members; language totals differ by at most one membership pair for configurable counts that are not multiples of 10. The default five-language quotas per class are 8/2/2/3. Sizes must be even, include at least one member/non-member pair per language, and fit the saved strata; incompatible quotas fail rather than silently change sizes. Selected rows retain their exact original IDs, source positions, labels, roles and hashes. `selection.json` records mode, policy, seed, counts and original/selected fingerprints; selected manifest snapshots record the exact rows.

Inspect `static_verification.json`, `training_history.json`, `frozen_weights.json`, `threshold.json`, `metrics.json` and the score exports. Static comparison must cover the selected 30 IDs against their saved grey-box scores at the same strict tolerance as full mode. All smoke performance reports are **preliminary and resolution-limited**. With the default 15 held-out negatives, empirical FPR changes in steps of 1/15; the 10 calibration negatives give steps of 1/10. A smoke TPR at <=1% FPR therefore describes a zero-observed-false-positive operating point, not a precise estimate of performance at 1% FPR. Smoke checks pipeline execution and fidelity, and should not be used to tune from held-out performance.

Save the entire smoke output directory with Kaggle Save Version **with outputs**. For the full experiment, restart the kernel and use:

```python
MODE = "full"
RUN_ID = "adaptive-sersem-full-v1"
RUN_EXTRACTION = True
RESUME_FROM = ""  # never restore smoke fitted artifacts into full mode
FEATURE_CACHE_PATH = ""  # full mode needs its own writable cache for remaining samples
FEATURE_REUSE_FROM = "/kaggle/input/EDIT-saved-smoke-output/token_features.sqlite"
```

`RUN_ID` is derived automatically from `MODE` in the notebook, or can be explicitly changed for a fresh experiment. The full selector leaves all 1,600/200/200/1,000 rows and their order unchanged; it rejects truncated production manifests. Optional `FEATURE_REUSE_FROM` imports only the compatible per-token records for selected full IDs into the new full run's SQLite cache. It checks extraction/model identity, checksums, IDs, labels, content hashes and original split roles. Missing IDs remain pending for extraction. With complete default smoke features, exactly 150 records can be reused and 2,850 full records remain. The importer copies **no weights, optimizer checkpoints, calibration thresholds or metrics**. All feature-extraction configuration and code fingerprints must match; incompatible cache provenance fails explicitly. Leave `FEATURE_REUSE_FROM=""` to extract from scratch.

Full fitting starts fresh, verifies all 1,000 saved static scores, trains on all 1,600 development samples, early-stops on the original 200 fitting samples, calibrates on the original 200 calibration samples, and evaluates all 1,000 held-out IDs. Smoke and full configurations, selection/gate fingerprints, fitted artifacts and checkpoints are distinct. Changing mode, sizes or selected IDs cannot resume an existing fitted run. Use `RESUME_FROM` only for a saved run of the **same mode and configuration**; mode mismatches fail before copying artifacts. Feature cache identity intentionally excludes execution mode/selected cohort so verified per-token records are reusable across modes. Keep `RUN_EXTRACTION=True` for first-time runs; it infers only missing selected samples. No LUMIA or StarCoder2 training is performed.

## Outputs and resumption

Persist the **whole OUT directory** with Kaggle Save Version **with outputs**, including SQLite journals. Next session attach it and the original baseline/caches, retain the same `RUN_ID`, and set `RESUME_FROM` to the trusted saved adaptive directory. Files are restored without overwriting different contents. Only operational cache paths and resource budgets may change within a resumed mode; mode/selected cohorts are immutable. Full-batch Adam moments, parameters, patience state, best parameters and history are saved atomically after every epoch; resume is deterministic. Frozen models, thresholds and cache records have checksums and exact identity checks. Baseline outputs are read-only inputs.

| Artifact | Schema/content |
|---|---|
| `config.json`, `environment.json`, `input_paths.json` | source/package hashes, baseline config/score hashes, exact manifest fingerprints, training hyperparameters, model/API versions and current input locations |
| `selection.json`, `{role}_manifest.parquet` | mode/seed/counts and original/selected fingerprints; selected row snapshots retaining original identities/roles |
| `imported_features.json` | optional per-token cache import source, requested/imported counts; no fitted artifacts |
| `token_features.sqlite` | extraction identity in `meta`; `features(sample_id, payload, checksum)` contains compressed NPZ arrays and JSON identity/label metadata, loaded with `allow_pickle=False` |
| Per-token NPZ arrays | `input_ids:int32[n]`, `offsets:int32[n,2]`, `category:uint8[n]`, `z/signals/static_weights:float32[n]`, `attention_mask/special_tokens_mask/eligible:uint8[n]` |
| NPZ scalars | `n_tokens`, `n_targets`, `static_score_gpu`, `ast_has_error`; token counts and masks are cross-checked |
| `feature_extraction.json`, `extraction_failures.jsonl` | cache/extraction counts; exact failed sample IDs and reasons |
| `static_verification.json` | checksummed gate, tolerance, saved/reconstructed/extraction scores and errors, all pool feature signatures |
| `training_checkpoint.json` | checksummed optimizer state, epoch, best loss/parameters, patience and history |
| `training_history.json`, `best_weights.json`, `weights.json` | training/validation loss history; best and completed models; categories, normalized weights/prior, calibration scale/bias, training/validation IDs/hashes, feature signatures and hyperparameters |
| `frozen_weights.json` | validated completed best model, frozen flag and checksum |
| `threshold.json`, `calibration_scores.parquet` | frozen model binding, calibration-only IDs/hashes and empirical FPR, threshold and calibration scores |
| `heldout_adaptive_scores.parquet/csv` | original identities/roles, raw `adaptive_score`, `adaptive_probability`, prediction and three saved baseline score columns |
| `metrics.json` | mode and preliminary/resolution-limited labels (smoke), matched selected-cohort coverage, global/per-language ROC-AUC and TPR at <=1% FPR for all four methods, adaptive calibrated TPR/FPR, weights, model/threshold hashes and limitations |

## CPU tests and limitations

```bash
python3 -m unittest discover -s tests -p 'test_adaptive.py' -v
python3 adaptive_token_attack/audit_notebook.py
```

Tests use synthetic identities and token signals: alignment/masks/overlaps, exact static reconstruction, finite-difference gradients, BCE stability, absent categories, weight normalization/optimization, early stopping, determinism/resumption, runtime discovery, leakage rejection, cache integrity, cache reuse, matched-ID evaluation and output schemas. Mode tests additionally cover deterministic balanced language-stratified selection, validation of complete originals before selection, selected-cohort identity/leakage guards, smoke/full end-to-end synthetic execution, strict static tolerances, partial smoke-to-full feature reuse and rejection of smoke fitted artifacts in full mode. Dependencies are NumPy, pandas, scikit-learn, pyarrow, tree-sitter and its Python grammar. No models, model downloads, GPU inference or GPU training run locally.

This is an experimental supervised extension of a reconstruction, not a recovered SERSEM implementation or a promise of improvement. Category attribution, especially equal-weight overlaps, adds an explicit convention. Rare/absent categories have limited identifiable evidence and regularization can dominate them. A common weight vector may fit languages unevenly. Low-FPR estimates from 200 calibration examples (and per-language test subsets) are coarse; reports flag fewer than 100 negatives. Reproducing scalar scores does not prove tokenwise equivalence to unpublished author code. GPU memory/throughput, real numerical fidelity and the final experiment await manual Kaggle execution. Your existing baseline results remain untouched.

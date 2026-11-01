# Reduced memTrace for StarCoder2-3B

Paper-informed reduced implementation, **not exact memTrace**. No pretrained model or Kaggle GPU experiment was run during local implementation. See [METHODOLOGY.md](METHODOLOGY.md), [reproducibility.json](reproducibility.json), and [feature_schema.json](feature_schema.json). Weak AUC is a valid experimental outcome.

## Reused infrastructure, inspected before implementation

Inspected `03_adaptive_sersem.ipynb`, corrected ReCaLL `04_recall_code.ipynb`, corrected Infilling `06_infilling_code.ipynb`, SPW-inspired `07_spw_code.ipynb`, their Python packages and the saved SERSEM baseline configuration. Existing working notebooks and shared helpers are unchanged.

| Component | Reuse and rationale |
|---|---|
| `shared/kaggle_setup.py`, `shared/requirements-kaggle.txt` | Latest common tested installer, used in corrected ReCaLL/Infilling/SPW; exact saved environment, only missing/incompatible pins installed, CUDA wheel index when needed, no force reinstall; explicit one-restart handling for changed loaded modules. Prefer this over duplicating older Adaptive notebook installation logic. |
| `shared/cache_setup.ensure_model_snapshot` | Checks pinned cached model files, sizes/header completeness, missing weights and disk reserve; explicit opt-in download. |
| `adaptive_token_attack.features.load_frozen_target` | Existing FP16 frozen GPU target/tokenizer loader and baseline attention backend. No baseline feature/probe extraction is rerun. |
| `adaptive_token_attack.data` | Validates complete saved partitions, source IDs/content hashes/labels, cross-partition overlap, deterministic balanced smoke subsets, fingerprints and static score matching. |
| `shared.resource_data.SourceStore` | Existing pinned source lookup, selected-text hash verification, bounded batches, cached Arrow or streaming access. |
| `shared.resource_pipeline.Monitor`, `restore_run` | Existing RAM/disk/session guards and collision-safe restore. Session budget begins at monitor creation, after dependency initialization. |
| `shared.research_execution.metric_report` | Existing overall/language ROC diagnostics with explicit low-FPR resolution. |
| SPW transactional cache and archive pattern | Cache implementation copied unchanged into the new package; SHA-checked JSON payloads, DELETE journaling, FULL synchronization, imports and per-example commits. Archive adapted for memTrace artifacts and partial runs, excluding model/cache directories. |
| ReCaLL/Infilling/SPW notebook packaging pattern | Manifest-pinned ZIP/extracted source handling, test-before-model flow, run/source/input hashes and archive cell. New stdlib bootstrap adds explicit candidate discovery and path checks. |

Saved baseline pins include `torch==2.11.0+cu128`, `transformers==4.52.4`, `tokenizers==0.21.4`, `dill==0.4.0`, `triton==3.6.0`. Do not replace them with generic latest or CPU PyTorch. The notebook validates exact baseline package versions and runs a CUDA allocation/FP16 matmul to exercise CUDA/cuBLAS library loading. Imports or library failures abort setup. A single requested kernel restart is followed by an idempotent rerun; never loop reinstalling dill. HF_HOME is `/kaggle/working/hf_cache`; empty MODEL_CACHE_DIR inherits the saved baseline's cache directory. If cache paths moved, set MODEL_CACHE_DIR to the actual HF hub cache, not an arbitrary snapshots parent.

## Kaggle instructions

1. Create a Kaggle Dataset containing **`notebooks/reduced_memtrace_kaggle_source.zip`**. Suggested dataset name: `reduced-memtrace-kaggle-source`. Upload the ZIP as-is or upload its extracted package tree. The notebook prints candidates; set UPLOAD_SOURCE or UPLOAD_ZIP only when discovery is ambiguous. The filename is exact; Kaggle dataset slug normalization is handled by recursive discovery.
2. Import **`notebooks/08_reduced_memtrace_code.ipynb`** and select Tesla T4. Attach your new source Dataset and the required baseline Dataset **`sayonsujitmondal/sersem-baseline-1000-results3`**, mounted at `/kaggle/input/datasets/sayonsujitmondal/sersem-baseline-1000-results3`. PATH_OVERRIDES supports individual paths if that mount differs.
3. Model weights must be available in a usable pinned Hugging Face cache. Default `MODEL_LOCAL_FILES_ONLY=True`, `ALLOW_MODEL_DOWNLOAD=False` deliberately avoids a surprise download. Set MODEL_CACHE_DIR for an attached cache. If no cache exists, enable Internet and explicitly set `ALLOW_MODEL_DOWNLOAD=True`; the reused helper downloads missing pinned files once, subject to disk guards. This does not require an additional attack-results Dataset. The baseline manifests themselves do not substitute for pretrained weights.
4. Default SOURCE_ACCESS=`streaming` retrieves exact pinned source rows, requiring Internet and any dataset/model access credentials. Alternatively use `cached` with a complete previously prepared pinned Arrow cache and DATASET_CACHE_DIR. No replacement splits are generated.
5. Run all cells with `MODE="smoke"`: **80 shadow, 20 validation, 20 calibration, 30 held-out**, total 150. Seed 42, span 128, batch 1. The shadow resource probe covers five languages and staggered lengths; a synthetic full-span stress executes if needed. Inspect `resource_probe.json`, `resource_stress.json` if present, `timing.json` and `metrics.json`. Cache-only resumes do not invent timings. Per-batch allocated/reserved peaks above 11 GiB abort continuation. Actual Kaggle memory and throughput must be measured.
6. For full evaluation change only `MODE="full"` in the first cell and rerun from that cell: **1,600/200/200/1,000**, total 3,000; the derived RUN_ID/OUT change automatically. All scientific settings remain fixed. To reuse smoke features, attach preserved smoke output and set CACHE_REUSE_FROM to its **feature_cache.sqlite file**. Leave RESUME_FROM empty for smoke-to-full.

Optional completed comparison Datasets: `adaptive-1000-results`, ReCaLL, Infilling, SPW-inspired when genuinely available. ADAPTIVE_SCORES can point to an exact file; otherwise unique filenames are discovered. Missing, ambiguous, unreadable or identity-mismatched files are omitted. Baseline's static SERSEM/LUMIA/fusion comparisons use saved scores; no prior model or score extraction is rerun. Filenames: `heldout_adaptive_scores.parquet`, `recall_scores.parquet`, `infilling_scores.parquet`, `spw_scores.parquet`. No SPW numbers are invented. Optional comparisons never become classifier inputs or tune settings.

## Features and evaluation

All 30 adjacent boundaries, including embeddings, yield 120 transition features. Selected boundaries `[0,8,15,23,30]` yield 25 activation and 15 context features; final causal logits yield 7 confidence features. Total **167**. The exact names/order/hash are in `feature_schema.json` (also exported from the loaded architecture). `features.py` is label-free. RF sees only those named columns; source lengths, language, identities and membership are metadata/targets. See the methodology for population std, cosine epsilon, near-zero threshold, special-token policy, final normalization and equations.

The fixed RF trains on shadow only. Validation reports diagnostics without search/refit. Calibration targets empirical FPR≤1%; held-out reports AUC, diagnostic low-FPR TPR, achieved FPR and deployed threshold rates overall/by language. No held-out parameter/direction searches. Calibration has only 100 negatives in full mode: 1% steps, substantial uncertainty, no population guarantee. Feature importance is descriptive association.

## Resume, outputs and downloads

Outputs: `/kaggle/working/reduced_memtrace_outputs/reduced-memtrace-<smoke|full>-v1/`. Each extracted sample commits independently. Features and token plans, never full hidden states, live in feature_cache.sqlite. Keys bind sample/label integrity, model/tokenizer revision/hash, feature schema, span, numerical version, source hashes and scientific configuration. Complete saved manifests are validated before smoke selection. Errors on invalid source/features are explicit and preserve prior checkpoints; incomplete runs cannot be reported complete.

Same-mode resume: Save Version with outputs, attach that output Dataset next session and set **RESUME_FROM to the previous run directory**. Keep scientific settings, source package and optional comparisons identical. Do not set it to a smoke directory when starting full; use CACHE_REUSE_FROM instead. SourceStore text/cache artifacts may be present in the results; model/HF cache directories remain outside it. The monitor stops at sample boundaries within 690 minutes from its creation. Persist results well before Kaggle terminates the session.

The final completed-results cell closes/checks SQLite, writes `/kaggle/working/<RUN_ID>.zip` and displays a download link. Download it and Save Version with outputs. On interruption skip classifier/evaluation/completed archive cells and run the optional partial-archive cell; download `<RUN_ID>-partial.zip`. A partial ZIP is for resuming, not completed results. Hard kills may omit the final Parquet snapshot, but already committed SQLite samples remain recoverable.

Expected complete artifacts: config.json, reproducibility.json, environment.json, feature_schema.json, features.parquet, feature_cache.sqlite, classifier.joblib, metrics.json, heldout_memtrace_scores.parquet, matched_comparison.parquet, threshold.json, run_manifest.json, source_version_hashes.json. Also selected partition manifests, resource_probe.json, optional resource_stress.json, resource_log.jsonl, progress.json, timing.json, extraction_measurements.json, validation_metrics.json, research_access.json, input_paths.json and model_cache_setup.json. Classifier reload requires the recorded scikit-learn environment; load only your trusted joblib artifacts.

## Local validation and source packaging

Use a Python environment with PyTorch, NumPy, pandas, scikit-learn, PyArrow and psutil; local tests use CPU mock tensors/models, never pretrained weights or remote datasets:

```bash
python3 -m unittest discover -s tests -p 'test_reduced_memtrace.py' -v
python3 -m unittest discover -s tests -p 'test_kaggle_setup.py' -v
python3 -m reduced_memtrace_code_attack.package_source
```

`VALIDATION.md` records actual local versions/results and limitations. Packager hashes every included source, pins that manifest in the new notebook, verifies ZIP CRC, and writes only the new notebook/package. It does not change existing working notebooks. Rebuild after changing any included source and upload the updated ZIP together with the updated notebook.

New source inventory: `__init__.py`, `features.py`, `inference.py`, `data.py`, `cache.py`, `runner.py`, `evaluation.py`, `bootstrap.py`, `archive.py`, `package_source.py`, `reproducibility.json`, `feature_schema.json`, `research_access.json`, `requirements-kaggle.txt`, `source_manifest.json`, `README.md`, `METHODOLOGY.md`, `VALIDATION.md`, all under `reduced_memtrace_code_attack/`; plus `tests/test_reduced_memtrace.py`, `notebooks/08_reduced_memtrace_code.ipynb` and `notebooks/reduced_memtrace_kaggle_source.zip`.

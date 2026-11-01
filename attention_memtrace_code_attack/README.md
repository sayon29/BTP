# One combined Attention-Augmented memTrace experiment

**1,787 features = all unchanged 167 Reduced features + 1,620 attention features.** One combined random forest, with the successful Reduced classifier parameters and original partitions. This is the sole primary experiment. Existing Reduced source, notebook, ZIP and saved results are left untouched.

See [METHODOLOGY.md](METHODOLOGY.md), [reproducibility.json](reproducibility.json), [feature_schema.json](feature_schema.json), [attention_feature_schema.json](attention_feature_schema.json) and [VALIDATION.md](VALIDATION.md).

## What is reused

Inspected notebooks 03 Adaptive, 04 corrected ReCaLL, 06 corrected Infilling, 07 SPW, 08 Reduced and their helper modules. Reuse the **unchanged notebook 08 environment setup cell** and exact `shared/requirements-kaggle.txt` lock/installer: torch 2.11.0+cu128, Transformers 4.52.4, tokenizer 0.21.4, dill 0.4.0, Triton 3.6.0. The installer changes only unsatisfied pins, preserves CUDA builds, and requests one kernel restart only if a loaded package changed. No new package or PyTorch version is needed for attention.

Existing snapshot/header/disk checks, pinned model/tokenizer loader, SourceStore, deterministic partition selection, manifest validation, Monitor and restore helpers remain unchanged. Original Reduced token planning, 167 feature reductions, RF, deterministic scoring and calibration are imported directly. The source ZIP bundles their unchanged dependencies. The new package adds explicit eager extraction, attention validation/reductions/schema, combined-cache compatibility, score comparisons and smoke evidence. Source staging accepts extracted mounts and ZIPs, preferring extracted files when both sit together.

## Upload and attach

1. Upload **`notebooks/attention_memtrace_kaggle_source.zip`** as a Kaggle Dataset, suggested name `attention-memtrace-kaggle-source`. Keep the ZIP filename or upload the extracted directory tree. You do **not** need to upload individual Python files separately. This ZIP contains the new package, unchanged Reduced/shared dependencies and tests.
2. Import **`notebooks/09_attention_memtrace_code.ipynb`**. Attach the new source Dataset and required baseline Dataset **`sayonsujitmondal/sersem-baseline-1000-results3`**. The first cell uses `/kaggle/input/datasets/sayonsujitmondal/sersem-baseline-1000-results3`; use PATH_OVERRIDES if files are mounted elsewhere. Source discovery prints candidates, so no guessed source slug or nonexistent ZIP path is required. Set UPLOAD_SOURCE to an extracted package root or UPLOAD_ZIP to a real ZIP if ambiguous.
3. Select Tesla T4. Preserve the pinned model cache. Default MODEL_LOCAL_FILES_ONLY=True and ALLOW_MODEL_DOWNLOAD=False avoid surprise downloads. Set MODEL_CACHE_DIR to the actual HF hub cache if moved. If weights are absent, explicitly set ALLOW_MODEL_DOWNLOAD=True with Internet enabled; the existing helper downloads only missing pinned files after disk checks. Tokenizer/model must match revision `733247c55e3f73af49ce8e9c7949bf14af205928`.
4. Default SOURCE_ACCESS=`streaming` requires Internet for exact pinned source rows. `cached` requires a previously prepared matching Arrow cache via DATASET_CACHE_DIR. HF_HOME remains `/kaggle/working/hf_cache`. Setup verifies CUDA and performs FP16 cuBLAS matmul before model loading; failures are explicit.
5. Optional: attach full Reduced memTrace outputs and set **REDUCED_MEMTRACE_SCORES** to their `heldout_memtrace_scores.parquet`, or let unique-filename discovery find it. Set **ADAPTIVE_SCORES** to `heldout_adaptive_scores.parquet` if needed. Missing/ambiguous/unmatched optional scores are omitted without blocking extraction. Static SERSEM, LUMIA and fusion use saved scores from the baseline Dataset. No prior attack is rerun.

Reduced saved features can be reused only when sibling `config.json`, `feature_schema.json`, `run_manifest.json` and `features.parquet` provide complete validated provenance, exact identity/schema/source matching and a completed-run feature checksum. Then the 167 values are copied exactly without recomputing their reductions. If metadata is missing/empty, the same original formulas run once in the eager forward; score comparison remains available. The local `Results/Memtrace` directory currently contains empty configuration/manifest JSON placeholders, so those local files cannot prove feature provenance. Nothing is edited or backfilled automatically.

Eager attention is necessary to obtain weights through the supported API. Fresh prefix formulas/order are identical, but FP16 backend rounding may differ from the historical SDPA run. The methodology distinguishes preserved definitions from bitwise reuse and records the chosen prefix source. Keep the same optional feature attachment across smoke and full so the scientific identity remains compatible.

## Smoke first, then one full experiment

Default first cell: MODE=`smoke`, seed 42, span 128 input positions, batch one, 690-minute monitor budget, 3 GiB disk reserve. Counts: **80 development / 20 validation / 20 calibration / 30 held-out = 150**.

Run all cells. The notebook first runs synthetic tests, then **five real attention-enabled GPU shadow probes** across languages/lengths. A synthetic max-span stress follows only if needed. Inspect `resource_probe.json`, optional `resource_stress.json`, `timing.json`, complete `feature_counts` in metrics and overall/per-language coverage. All 30 layers and 24 query heads must be present. Both allocated and reserved peaks above 11 GiB stop execution; features are never silently omitted. There are no T4 measurements for this new code yet; the old 5.77 GiB/roughly six-minute result is not an attention-enabled cost estimate.

For the sole full experiment:

```python
MODE = "full"
# Point to the completed combined smoke output, not the old Reduced cache:
CACHE_REUSE_FROM = "/kaggle/input/<your-smoke-output>/feature_cache.sqlite"
```

Re-execute the entire first cell so RUN_ID and OUT derive from full mode. If the conventional completed smoke output is still under WORK_ROOT, the notebook can find its cache automatically. All scientific settings stay unchanged. Full counts: **1,600 / 200 / 200 / 1,000 = 3,000**. Full mode verifies compatible completed smoke config/counts/metrics before extraction. Do not use RESUME_FROM to copy smoke output into full output; CACHE_REUSE_FROM imports only compatible features.

The full run trains exactly one combined RF on development. Calibration alone chooses the deployed threshold. No attention-only model, ablation, parameter sweep, feature selection, alternative configuration or held-out selection exists. Smoke and full are stages of the same fixed combined method.

## Preserve, download and resume

Outputs live at `/kaggle/working/attention_memtrace_outputs/attention-memtrace-<smoke|full>-v1/`. The final cell closes/checks SQLite, creates `/kaggle/working/<RUN_ID>.zip` and displays its download link. Download the ZIP and **Save Version with outputs**. Weights/HF cache directories are excluded.

For interruption: committed features remain in `feature_cache.sqlite`; partial `features.parquet` and progress are exported where execution can unwind. Skip classifier/evaluation/completed archive cells and run the optional partial-archive cell. Download `<RUN_ID>-partial.zip`. Hard kills may omit the last Parquet snapshot; SQLite commits remain usable. A partial archive is for resume, not a completed evaluation.

For same-mode resume, attach the previous output and set RESUME_FROM to its **run directory**, keeping RUN_ID/MODE, source version, scientific settings and optional prefix provenance unchanged. `smoke_validation.json` travels with full runs as proof of completed smoke, so full-to-full resume does not require the old smoke mount. CACHE_REUSE_FROM accepts only combined caches with exactly matching scientific/schema metadata; Reduced-only caches are rejected. Every new session with pending samples runs five real resource probes; previously committed sample vectors are retained unchanged. The Monitor stops at checkpoint boundaries for RAM/disk/session limits; persist outputs before Kaggle's hard cutoff.

## Features, comparisons and limitations

All 30 layers contribute layer entropy, concentration, allowed-pair sparsity, self bias, previous-token bias and mean distance; every query head contributes entropy and maximum focus. Exact feature identifiers include layer/head indices. No raw attention tensors are cached.

The paper leaves some conventions unspecified. Our adaptive sparsity threshold is the mean head-averaged normalized probability over valid causal pairs; per-head entropy uses a mean over valid queries. These are explicit frozen choices, not claimed author defaults. All eight specified attention families are present, but the unavailable author code's complete feature dimension cannot be certified. This remains an adaptation, not exact original memTrace.

Paired reports use actual ID-matched saved scores; no numeric constant is fabricated into a comparison. The existing local Reduced score file independently gives AUC **0.7033** on all matched 1,000 samples. Reports include AUC differences and fixed-score paired stratified bootstrap intervals (2,000 draws, seed 42), without retraining or choosing another attack. Low-FPR calibration has only 100 negatives in full mode and is coarse. More features do not imply improvement; weak/negative results remain valid.

## Files and local checks

New files under `attention_memtrace_code_attack/`: `__init__.py`, `features.py`, `inference.py`, `data.py`, `cache.py`, `runner.py`, `evaluation.py`, `workflow.py`, `bootstrap.py`, `archive.py`, `package_source.py`, `reproducibility.json`, `feature_schema.json`, `attention_feature_schema.json`, `research_access.json`, `requirements-kaggle.txt`, `source_manifest.json`, `README.md`, `METHODOLOGY.md`, `VALIDATION.md`. Also: `tests/test_attention_memtrace.py`, `notebooks/09_attention_memtrace_code.ipynb`, `notebooks/attention_memtrace_kaggle_source.zip`.

Expected full results: config.json, reproducibility.json, feature_schema.json, attention_feature_schema.json, environment.json, features.parquet, feature_cache.sqlite, classifier.joblib, metrics.json, heldout_attention_memtrace_scores.parquet, matched_comparison.parquet, threshold.json, run_manifest.json, resource_probe.json, timing.json, source_version_hashes.json. Also manifests, progress/resource logs, validation metrics, attention audits, prefix-reuse report and full-mode smoke proof. Feature importance appears in metrics, with descriptive rather than causal interpretation.

```bash
python3 -m unittest discover -s tests -p 'test_attention_memtrace.py' -v
python3 -m unittest discover -s tests -p 'test_kaggle_setup.py' -v
python3 -m attention_memtrace_code_attack.package_source
```

Use the pinned Kaggle environment or a local CPU test environment with the dependencies listed in VALIDATION. Packaging updates only notebook 09 and its source manifest/ZIP. Rebuild and upload the matching notebook/ZIP after changing included source. It never rebuilds or changes notebook 08 or Reduced outputs.

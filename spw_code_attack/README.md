# SPW-inspired code experiment

**The original SPW-MIA equations remain unverified.** This package implements the existing, explicitly named **SPW-inspired dynamic-halving weighted Bernoulli-JS** adaptation. It must not be reported as a reproduction of published SPW-MIA. See [METHODOLOGY.md](METHODOLOGY.md) for every local equation, the source ledger, unavailable published details and domain differences. No published parameters or equation numbers are invented.

## Deliverables

- Primary notebook: `notebooks/07_spw_code.ipynb`.
- Compatibility copy for existing uploads: `notebooks/05_spw_code.ipynb` (same cells and source pin).
- Upload package: `notebooks/spw_kaggle_source.zip`.
- Frozen settings: `spw_code_attack/reproducibility.json`, checked against a pinned SHA256 before use.
- Local tests: `tests/test_spw.py`, `tests/test_spw_workflow.py`, shared `tests/test_kaggle_setup.py`.
- Validation record: `spw_code_attack/VALIDATION.md`.

Scientific code stays in `method.py`, `inference.py` and `engine.py`; orchestration uses `data.py`, `runner.py`, `evaluation.py`, `cache.py` and `archive.py`. Existing attack implementations are unchanged by this work.

## One fixed scientific configuration

Frozen StarCoder2-3B; saved baseline model/tokenizer revision; five languages (Go, Java, Python, Ruby, Rust); seed 42; first 256 eligible targets. Original-prefix chunks 256 and same-length shortened batches four. FP16 inference, FP32 log-softmax, FP64 scalar reductions. No artificial BOS or padding. Logit `t-1` predicts the fixed target `x[t]`; shortened contexts end immediately before that same token.

Dynamic halving accepts shorter suffixes while their greedy token equals the original greedy token, stopping on the first mismatch or length one. Local weights emphasize retained prediction stability and removed prefix fraction. The score is the negative weighted mean Bernoulli-JS divergence of actual/greedy original-prefix probabilities. Higher is member-like, fixed without labels. These rules are **our adaptation**, not recovered paper mathematics. No reference model, training, sweeps or post-hoc reversal.

All saved manifests are validated before selection. Full counts are 1600/200/200/1000; smoke counts 80/20/20/30. Selection is deterministic and stratified with no new split. Only calibration and held-out receive GPU inference: **50 smoke / 1,200 full**. Development/validation manifests remain saved for provenance. Calibration fixes the deployed threshold at empirical FPR <=1%; held-out ROC thresholds are separately labelled diagnostics. Low-FPR estimates from small cohorts are coarse.

## Kaggle setup

1. Rebuild locally with `python3 -m spw_code_attack.package_source`.
2. Upload **`notebooks/spw_kaggle_source.zip`** as a Kaggle Dataset, e.g. `spw-kaggle-source`. Keep the ZIP filename unchanged. Alternatively upload its extracted contents, with `spw_code_attack/source_manifest.json` under the Dataset root.
3. Import **`notebooks/07_spw_code.ipynb`**, select a Tesla T4 and attach the source Dataset, the saved SERSEM baseline Dataset, and a prepared model-cache Dataset. Enable Internet for source streaming.
4. Set `UPLOAD_SOURCE` to the actual mounted Dataset directory. The loader can discover one unique source mount if the configured path is missing; ambiguous mounts require an explicit path. ZIP and extracted source files are hash checked. Use a fresh `PROJECT_ROOT` after updating source.
5. Keep the default baseline path `/kaggle/input/datasets/sayonsujitmondal/sersem-baseline-1000-results3`, adjusting it to the actual mount if necessary. `PATH_OVERRIDES` resolves ambiguous saved files. Never construct replacement partitions.
6. Set `MODEL_CACHE_DIR` to the attached Hugging Face cache root. Empty inherits the saved baseline cache path. Defaults are local-only and no model download. Explicitly set `ALLOW_MODEL_DOWNLOAD=True` only if the pinned snapshot must be downloaded. For cached dataset access, set `SOURCE_ACCESS='cached'` and point `DATASET_CACHE_DIR` to the existing Arrow cache; default is streaming.

`ADAPTIVE_SCORES=''` skips the adaptive comparison and performs no adaptive file discovery. Set it explicitly to `heldout_adaptive_scores.parquet` only when attaching it; every selected identity, label, language and provenance field must match. Full adaptive scores are recommended even for smoke. ReCaLL and Infilling results are never required. Saved SERSEM/LUMIA/fusion scores are reused, never recomputed.

The notebook reuses corrected ReCaLL notebook 04's setup cell and **shared** `kaggle_setup.py`, with the exact saved environment lock and `require_cuda=True`. It checks before installing, skips satisfied pins, preserves correct CUDA wheels and requests a restart only if loaded code changed. Pins include torch 2.11.0+cu128, torchvision 0.26.0+cu128, torchaudio 2.11.0+cu128, triton 3.6.0 and transformers 4.52.4. Validation failures stop execution. Local mocked setup checks pass; no claim is made that ReCaLL's or SPW's current setup has executed successfully on Kaggle.

Shared infrastructure also reuses Adaptive's saved-run loader, identity validation, selection and frozen model loader; shared source streaming, model snapshot setup, monitoring, restore logic and metric reporting. The source bootstrap/archive conventions follow the corrected existing notebooks. No independent installer was introduced.

## Smoke and full

Start with:

```python
MODE = 'smoke'
SMOKE_SIZES = dict(shadow=80, validation=20, calibration=20, heldout=30)
RUN_ID = f'spw-inspired-code-{MODE}-v1'
RESUME_FROM = ''
CACHE_REUSE_FROM = ''
```

After checking the actual smoke throughput and outputs, change `MODE='full'` and rerun the configuration cell so `RUN_ID`/`OUT` switch to a distinct full directory. Keep `RESUME_FROM=''`. Optionally set `CACHE_REUSE_FROM` to the saved smoke `spw_cache.sqlite`. All **1,000 original held-out examples** plus 200 calibration examples must finish. No other scientific setting changes.

Same-run resume: attach the saved entire output directory, keep the mode/configuration and set `RESUME_FROM` to it. Changed settings, inputs, optional comparisons or source versions require a new run identity. Smoke-to-full uses cache import, never a smoke threshold. Transactional checksummed SQLite entries preserve finished prefix/sample work; interrupted branches replay from durable entries. A graceful scoring interruption records status and closes the cache. Hard kills can leave incomplete call accounting, which is explicitly flagged.

## Outputs and runtime

Outputs go to `/kaggle/working/spw_outputs/<RUN_ID>/`: `config.json`, `reproducibility.json`, `research_access.json`, `environment.json`, `metrics.json`, `spw_scores.parquet`, `matched_comparison.parquet`, `run_manifest.json`, `spw_cache.sqlite`, selected manifests, calibrated threshold, source/dependency hashes, progress, call and resource logs, and `timing.json`.

The final cell verifies required artifacts and SQLite integrity, checks disk reserve, and archives the entire closed run to **`/kaggle/working/<RUN_ID>.zip`**. Download it or Save Version with outputs. A database with an active journal/WAL is rejected. Resource stops preserve existing results; caches are never automatically deleted.

`timing.json` gives measured transformer/head calls and a linear estimate for the 1,200-example full inference workload based on completed uncached examples. Cache-only runs report the estimate unavailable. Partial-cache recovery can underestimate fresh cost; setup/source retrieval are excluded and prefix lengths/sharing vary. No T4 runtime or successful Kaggle result is claimed before an actual smoke run. Full execution may require multiple resumed sessions. No approximate runtime shortcut changes the frozen mathematics.

## Local verification

```bash
python3 -m unittest discover -s tests -p 'test_spw*.py' -v
python3 -m unittest discover -s tests -p test_kaggle_setup.py -v
python3 -m spw_code_attack.package_source
```

The notebook runs both suites before target inference. Tests use synthetic probabilities and temporary files; an optional tiny random CPU StarCoder2 test runs only if torch/transformers are already available. No pretrained inference, package installation or dataset/model download runs locally. See `VALIDATION.md` for observed results and the skipped optional test.

# Local validation — 2026-11-01

No Kaggle GPU test or pretrained StarCoder2-3B inference was performed. No attack AUC, actual T4 peak VRAM or throughput is claimed. The expected 6–8 GiB range in METHODOLOGY is analytical planning; the notebook records real probe measurements and enforces an 11 GiB gate. No held-out data was used to select feature definitions or classifier settings.

Local environment: Python 3.12.3, torch 2.13.0+cpu, transformers 5.13.1, numpy 2.0.2, pandas 3.0.3, scikit-learn 1.9.0, pyarrow 19.0.1, psutil 7.2.1. Existing local packages were reused with temporary import links; existing Python environments were not modified. These local versions differ from the frozen Kaggle environment. The notebook enforces the exact saved baseline lock and reruns the synthetic checks on that environment before loading target weights.

Commands executed from the repository:

```bash
PYTHONPATH=/tmp/memtrace-local-deps /home/sayonmondal/plivo/env/bin/python \
  -m unittest discover -s tests -p test_reduced_memtrace.py -v
python3 -m unittest discover -s tests -p test_kaggle_setup.py -v
```

Results: **28 Reduced memTrace tests passed**, plus **11 existing dependency-setup tests passed**. The 28-test run took approximately 6.5 seconds; this is test-suite time, not experimental throughput. Tests cover:

- Hidden tuple dimensions, all-boundary indexing and round-half-up selected depths.
- Hand-checked Euclidean/cosine values, zero-vector cosine and FP16 inputs reduced in FP32, including distances that overflow FP16.
- Absolute activation mean/RMS/std/max/sparsity; running-mean context evolution; padding exclusion.
- Shifted causal targets, stable entropy, top-two margins and logit-shift invariance.
- Schema order/uniqueness, no NaN/Inf acceptance, 128-position truncation and short-source failure.
- Actual randomly initialized **tiny StarCoder2 architecture** (no pretrained download): tuple[0] equals embeddings, last state equals captured final norm output, expected shapes, no attentions, successful full feature extraction. This test used the local Transformers version above; pinned Kaggle behavior still requires running the notebook tests there.
- Saved local SERSEM baseline loading: all 3,000 manifest rows validated, original split/content overlap checks, exact pinned revision and deterministic 150-row smoke selection.
- RF reproducibility with fixed tree-order probability accumulation; calibration-only threshold and rejection of held-out calibration inputs.
- SQLite transaction/reopen/import, scientific/label identity separation and cache-only resume without loading a model.
- ZIP and extracted source staging, pinned hashes/path safety; completed and partial ZIP archiving without nested HF/model caches.
- RAM/disk/session monitor APIs, resource guard failure and checkpoint preservation.
- A complete **150-row synthetic pipeline** through features, classifier, validation, calibration, matched baselines and result artifacts. These are mock results, not BTP experimental results.
- Notebook code syntax and default smoke configuration.

The existing installer tests mock package metadata/pip/CUDA APIs to verify missing-only installs, unchanged CUDA packages, CPU-build rejection/replacement, exact pins and idempotent reruns. They do not claim that a fresh Kaggle installation or CUDA shared-library check ran locally.

The source ZIP is additionally extracted into a clean temporary directory, its manifest checked, and tests rerun from that directory. The baseline-loading test skips there because baseline results are intentionally not bundled. The notebook is supplied separately and copied alongside the clean extraction for its syntax check. ZIP CRC and SHA256 manifest verification complete locally. Logs are retained under `/tmp/memtrace-*.log` during this implementation session; they are not experimental results.

Clean-package results: 28 Reduced memTrace tests discovered, **27 passed and 1 intentionally skipped** (baseline files not bundled), approximately 6.1 seconds; all **11 dependency-setup tests passed**. All 40 included source files passed SHA256 manifest verification and the source archive passed CRC validation.

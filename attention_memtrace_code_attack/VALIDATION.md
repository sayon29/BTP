# Validation — 2026-11-01

**No pretrained StarCoder2-3B or Kaggle GPU experiment was executed during implementation.** No attention-augmented AUC, measured T4 peak VRAM or T4 runtime is claimed. The notebook defaults to the required 150-sample GPU smoke experiment and requires compatible completed smoke evidence before its sole full experiment.

Local validation uses Python 3.12.3, **Transformers 4.52.4**, tokenizers 0.21.4 and Hugging Face Hub 0.30.2. These three pinned packages were downloaded into `/tmp/attention-memtrace-test-deps` without changing installed user environments. Other locally reused packages: torch 2.13.0+cpu, numpy 2.0.2, pandas 3.0.3, scikit-learn 1.9.0, PyArrow 19.0.1 and psutil 7.2.1. Therefore the actual pinned Transformers attention implementation was tested, while the **full saved CUDA/scikit-learn environment still requires Kaggle validation**. The notebook preserves the complete original lock and reruns synthetic tests there before loading pretrained weights.

Commands:

```bash
PYTHONPATH=/tmp/attention-memtrace-test-deps:/tmp/memtrace-local-deps \
  /home/sayonmondal/plivo/env/bin/python -m unittest discover \
  -s tests -p test_attention_memtrace.py -v
python3 -m unittest discover -s tests -p test_kaggle_setup.py -v
```

Result: **33 attention/combined tests passed**, approximately 4.5 seconds; **11 unchanged dependency-setup tests passed**. Test-suite duration is not experimental inference throughput.

Validated:

- Hand-calculated mean-map entropy, per-head entropy/focus, concentration, allowed-pair sparsity, self/previous-token bias and mean distance.
- Row sums, normalization, causal future exclusion, padding exclusion, zero probabilities, FP16-to-FP32 behavior, NaN/Inf/negative probability rejection and short-sequence failure.
- All layers required; query-head versus KV-head shape checks and complete combined schema/order/hash.
- Actual randomly initialized tiny StarCoder2 under **Transformers 4.52.4**, including grouped-query attention (four query heads, two KV heads), causal row sums and returned probability maps.
- A **30-layer tiny architecture** preserving all 167 Reduced values exactly against the original extractor using the same eager model; a forward hook confirms the combined extractor performs one forward. No pretrained weights or remote dataset is used by these model tests.
- FP16 tiny model finite output and raw probability tolerance; same-weight tiny FP32 eager/SDPA agreement to atol=1e-6, rtol=1e-5. This does not imply bitwise equality of T4 FP16 backends.
- Validated saved-prefix reuse skips original reductions; absent/empty provenance falls back to the same original formulas, not untrusted vectors.
- Transactional close/reopen/import, combined identity compatibility and rejection of Reduced-only/incompatible caches.
- Original token planner, label/split identity binding, fixed RF and deterministic scoring, calibration-only threshold selection and rejection of held-out calibration.
- One full **150-row mock workflow**, one combined RF fit, five actual mock probes, max-span stress, cache-only resume, baseline matching and output artifacts. These mock results are not BTP experimental results.
- Explicit memory-stop behavior without dropping features; partial output persistence, completed/partial ZIP packaging and exclusion of nested HF/model caches.
- ZIP and extracted source staging, mixed-layout discovery, source checksums and unsafe-path rejection.
- Completed-smoke gate and sealed proof for full-to-full resume; deterministic paired score-only bootstrap.
- Notebook syntax, default smoke configuration, explicit eager loading and a single evaluation/model-fit path.

The existing 11 installer tests mock package metadata, pip and CUDA APIs to check exact baseline pins, missing-only installation, CPU/CUDA build handling, one-restart policy and idempotence. They are not fresh GPU environment installation claims. Notebook 09's environment setup cell is byte-for-byte identical to the successful notebook 08 cell, including CUDA allocation and FP16 cuBLAS matmul. Actual CUDA library loading and T4 memory/timing must pass when the user runs Kaggle.

Read-only inspection of `Results/Memtrace/heldout_memtrace_scores.parquet` verified all 1,000 original baseline identities/labels/content hashes and recalculated ROC-AUC **0.7032999999999999**. Empty JSON files in that directory were not edited or used to invent provenance. Saved Reduced inputs, source, notebook and previous ZIP are protected by before/after SHA256 checks: all **33 pre-existing protected files** remained unchanged.

The final source archive is CRC checked and manifest-pinned to notebook 09. Clean-extraction tests run outside the repository, with the notebook supplied alongside the source package; the saved-results test intentionally skips there because experimental datasets are not bundled. Local execution logs are under `/tmp/attention-memtrace-*.log` for this session.

Clean-package result: **32 passed, 1 intentional skip** out of the same 33 tests (saved experimental scores are not bundled), plus **11 installer tests passed**. All **61 bundled source files** passed SHA256 verification, and the source ZIP passed CRC checking. Existing unlabelled/Reduced cache contents are also rejected when opening a combined cache, as well as on import.

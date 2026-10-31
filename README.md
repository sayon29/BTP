# SERSEM baseline: Stage 1

**Fresh-session setup:** both notebooks now accept the extracted `sersem-kaggle-source3` mount and explicitly prepare missing pinned model/Arrow caches on Kaggle, with a 3 GiB setup reserve. Model loading remains local-only after preparation. See the exact settings/cell sequence in [RESEARCH_RUN.md](RESEARCH_RUN.md).

**Active task: two-attack research execution.** Run [notebook 01](notebooks/01_sersem_baseline.ipynb) for timing/SERSEM and [notebook 02](notebooks/02_sersem_neighbourhood.ipynb) for standalone code-aware neighbourhood MIA on exactly matched saved IDs. Follow [RESEARCH_RUN.md](RESEARCH_RUN.md). The rebuilt source ZIPs contain both runners; notebook 01 defaults to timing-only before fitting. Existing baseline mathematics/checkpoint identities are preserved. No Stage 2/submission restructuring or adaptive importance is underway.


**Deferred reference:** the [2027 competition audit](KAGGLE_RESOURCES.md#icse-2027-competition-audit-and-proposed-runtime-plan--2026-10-23) is historical planning, outside the present two-attack research scope. Use RESEARCH_RUN.md for the current execution instructions.

The feasibility plan uses hypothetical **3/6/12-hour** budgets. Your T4 measurements are development resources only. The proposed competition path must work from supplied/attached assets without runtime downloads; target-specific Stage 2 probe fitting, if required, must fit within that fresh execution. See the audit’s pre-submission/scoring boundary.

Prepared for manual execution on Kaggle GPU. **No model inference, attack validation, weight download, or notebook execution has been performed locally. No published result is claimed.**

- Original checkout: `SERSEM-Poisoned-Chalice-Competition-2026/`, with Git history intact and tracked files unchanged.
- Upstream: https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026
- Pinned commit: `413f56040e5b4805bcf15ed794dec56bc4e16b41`.
- Paper: https://arxiv.org/abs/2604.01147
- Notebook: [notebooks/01_sersem_baseline.ipynb](notebooks/01_sersem_baseline.ipynb).
- Code/paper differences and limitations: [AUDIT.md](AUDIT.md).
- Static cell/configuration/API audit: `python3 shared/audit_notebook.py`; report: `notebooks/01_sersem_baseline.static_audit.json`. This parses source only.
- Setup helper: `shared/kaggle_setup.py` preserves compatible Kaggle numerical/CUDA packages.
- Runtime helpers: `shared/sersem_baseline.py` (original LUMIA orchestration), `shared/weighted_anomaly.py` (reconstructed grey-box and experimental fusion), `shared/resource_data.py` and `shared/resource_pipeline.py` (bounded ingestion/extraction/scoring/resume); dependency ranges/pins: `shared/requirements-kaggle.txt`.
- Author-artifact search: `shared/author_source_search.json` (including the unresolved Figshare HTTP 403).
- Source checksums and parquet metadata: `shared/upstream_manifest.json`, `shared/upstream_parquet_inventory.json`.
- `neighbourhood_attack/` now implements the standalone research comparison; `adaptive_token_attack/` remains a placeholder. No adaptive or submission-specific work is included.

The corrected baseline separates three provenance levels:

| Branch | Implementation |
|---|---|
| `lumia` | Direct unchanged upstream pooling, probe training/selection and top-five mean, with external failure guards |
| `greybox` | Our Section 2.1 reconstruction: uniform population raw-logit Z-score → sigmoid → sum-normalized structural weighted mean; language-aware AST/dictionary/linting |
| `fusion` | Our configurable convex mixture `alpha*lumia + (1-alpha)*greybox`; fixed or selected on fusion-fit data |

No author WeightedAnomalyMIA implementation or cross-branch fusion was located across accessible Git history, branches/releases, related code or the HF file listing. The cited Figshare archive returned HTTP 403 and remains uninspected. [AUDIT.md](AUDIT.md) records the search, exact equation, every reconstruction assumption and the difference between the original LUMIA mask and reconstructed grey-box mask. Figure 1 does not supply a cross-branch numerical fusion rule. The presence of published 3B/7B probe files does not verify their training lineage.

## Kaggle

Use the refreshed [notebook](notebooks/01_sersem_baseline.ipynb) **and** [source ZIP](notebooks/sersem_kaggle_source.zip) together. Follow [KAGGLE_RESOURCES.md](KAGGLE_RESOURCES.md) for capacity, validation, persistence and resume instructions. The notebook defaults to a new `sersem-project-resources-v1` project directory, `MODE="resource_validation"`, a new run ID, faithful 8192-token fp16 execution and all three branches. The user reports that the earlier three-branch smoke run succeeded; the new resource path has received static checks only.

1. Attach the source ZIP as a private Kaggle Dataset, enable a T4 GPU and Internet, and edit `UPLOAD_ZIP` to its mount. The allowlisted archive excludes secrets, upstream Git history, datasets, probes and weights. Alternatively use `SOURCE_MODE="github"` with your framework repository and an immutable commit SHA. Rebuild with `python3 shared/package_source.py` after edits.
2. Complete pinned model/Arrow caches are reused. With the reported fresh 19.50 GiB free disk, `AUTO_DOWNLOAD_MODEL_IF_MISSING=True` explicitly prepares missing target assets on Kaggle before local-only model loading, with at least 3 GiB setup reserve and metadata-based size checks. Set `MODEL_CACHE_DIR` to the HF Hub cache root (default `/kaggle/working/hf_cache/hub`), or a complete attached cache. Model/dataset revisions are pinned in config.json and reused on resume. No model is downloaded locally by source/static checks.
3. `DATA_ACCESS="cached"` reads memory-mapped Arrow files; `AUTO_PREPARE_DATASET_IF_MISSING=True` explicitly prepares missing language caches after disk checks. Set `DATASET_CACHE_DIR` before running installation/import cells, especially for attached caches. Alternatively choose `DATA_ACCESS="streaming"` explicitly to avoid full Arrow preparation, accepting repeated remote scans. No mode switches automatically. See RESEARCH_RUN.md for the preparation allowance and resume procedure.
4. Run cells in order. Resource validation uses 1,600 shadow rows plus separate 200-row fusion and 200-row threshold pools; held-out metrics are disabled by default. Inspect `resource_preflight.json`, `resource_validation.json`, `loaded_model_resources.json` and `resource_log.jsonl`. Confirm `exercised_full_token_cap` before interpreting the GPU check as evidence about 8192-token capacity. Do not tune resources or fusion on held-out results.
5. Only after validation, start a **new RUN_ID** with `MODE="full"`. This selects 25,000 shadow, 3,125 fusion-fit, 3,125 threshold-calibration and 25,000 held-out rows. It scans all official source rows for exclusions/conflicts/overlap, but does not train on every official row. Keep `FAITHFUL_MODE=True`; shorter context, float16 feature storage or an attention override require explicit nonfaithful configuration and a separate experiment. `BRANCHES=["lumia"]`, `["greybox"]`, `["fusion"]` and combinations are independently supported; fusion requires both underlying scores.
6. Persist the **whole run directory** before ending a session. Kaggle working storage is ephemeral. Use Save Version **with outputs**, attach those outputs to the next session, or manually export a private checkpoint Dataset/external copy. Nothing is automatically uploaded/published. Set `RESUME_FROM` to the trusted saved run directory and retain the same RUN_ID/settings/versions. Complete instructions and the required checkpoint files are in [KAGGLE_RESOURCES.md](KAGGLE_RESOURCES.md).

Initialization defines every helper import, `cfg`, `OUT`, resolved revisions, seeds and actual dependency versions before data work. The Python 3.13 installer preserves compatible loaded numerical/CUDA packages instead of repeatedly replacing NumPy or Matplotlib. It pins attack APIs, validates generated constraints against selected installed metadata, records duplicate dist-info diagnostics and captures pip errors. Restart only when instructed after changing already loaded dependencies. Current Kaggle package resolution and third-party APIs still require manual validation.

Official train provides probe development, separate fusion fitting and separate threshold calibration; official test stays held out. The complete official-test content hash set is excluded from development. Invalid null/non-string/blank content is filtered before sampling with stable source row IDs and recorded counts (`excluded_content_rows.csv` in the bounded runner, `data_quality.json`). Accepted source text is unchanged. Unknown labels, conflicting labels, insufficient samples, configuration drift and content/role overlap fail explicitly. Earlier legacy-loader runs used exclusions Parquet; the new CSV avoids materializing all exclusions in RAM.

The resource runner retains only metadata and compact score/checkpoint records for the selected experiment, a configurable bounded compressed text cache, and one group of memory-mapped pooled features. It calls upstream `_train_probe` unchanged, preserves its internal splits/MLP/top-five selection and deletes a group's temporary features only after all its probes are committed. It does not stream the original trainer's full per-layer train/validation tensors. GPU logits are transient and remain potentially large. There is no automatic quantization, CPU offload, context reduction or cache eviction. See the resource guide for the T4 feasibility limits.

Do not run the upstream shell script: it sources its tracked `.env`, downloads an unverified probe and aliases 3B probes for other models. This framework imports only hash-verified upstream Python sources and never reads that `.env`. Keep saved probe/checkpoint pickles private and load only trusted own-run artifacts.

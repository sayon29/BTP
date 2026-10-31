# SERSEM-guided neighbourhood membership inference

Run [notebook 02](../notebooks/02_sersem_neighbourhood.ipynb) manually on Kaggle GPU using the **existing 1,000-row SERSEM held-out manifest**, config, and saved LUMIA, grey-box and fusion scores. No LUMIA/StarCoder2 training occurs. The notebook validates all saved score identities before loading models.

`SITE_IMPORTANCE="sersem"` (default) calls `StructuralWeights.char_weights` from `shared/weighted_anomaly.py`, also inherited by the grey-box scorer. Its existing weighting implementation is unchanged. These are the exact weights of this repository's **SERSEM reconstruction**, not a claim to recovered author weights. Missing parser/dictionary/linter dependencies raise an error; no approximate weight fallback is used. Token weights use the existing max-over-character-span projection, excluding the first token, specials and empty offsets. A site's importance is the maximum overlapping target-token weight across its affected occurrences. This site aggregation is our selection policy. Comment insertion has no original span and receives selection priority zero.

Rank eligible sites by decreasing importance with seeded tie shuffling. Try one valid MLM replacement at each site before revisiting productive sites round-robin. Each neighbour independently transforms the original: one comment substitution/insertion or one consistent Python local rename (possibly multiple occurrences). `K=5` requires five distinct validated texts; failures never reduce K or substitute originals. `SITE_IMPORTANCE="uniform"` retains the old uniform-within-kind ordering and reserved insertion, with the same diversity-first generation pass.

CodeBERT `microsoft/codebert-base-mlm` proposes replacements in a bounded context. Existing tree-sitter syntax/noncomment-token checks and conservative Python consistent-renaming/AST checks remain. No unrestricted single-occurrence identifier replacement is added. Selected-site importance, character locations, transformation types, replacements and failed-proposal reasons are saved in SQLite and score exports. `selected_sites` is the bounded candidate pool; `neighbour_transformations` identifies accepted changes.

Scoring remains **mean of K whole-sequence normalized causal neighbour NLLs minus original normalized causal NLL**, using the same pinned StarCoder2 revision/context/precision as SERSEM. No local-window NLL is implemented. Reports include overall/per-language ROC-AUC, empirical TPR at FPR <=1%, coverage by language/label, and all three saved SERSEM branches recomputed on exactly the same successful comparison subset.

For a small smoke test, edit the notebook's source/baseline/cache paths, keep `MODE="smoke"`, `K=5`, `SITE_IMPORTANCE="sersem"`, `SMOKE_PER_LANGUAGE_CLASS=3`, and `RUN_ID="neighbourhood-sersem-k5-smoke-v2"`. Run cells in order (up to 30 existing IDs). Inspect `metrics.json`, selected sites, transformations and failures; low-FPR estimates on this small cohort are resolution-limited.

For the full run, restart the kernel, set `MODE="full"` and `RUN_ID="neighbourhood-sersem-k5-full-v2"`, and retain the other scientific settings. This requests all 1,000 saved IDs. Optional `NLL_REUSE_FROM` may point to the compatible smoke output. Persist the entire output directory with Kaggle Save Version **with outputs**. Resume with the same run ID/config and `RESUME_FROM` pointing to saved outputs; only operational paths/batch sizes may change. Old generation checkpoints cannot be reused across importance/code changes. SQLite checksums, per-text NLL caches, manifest/config fingerprints and resource guards remain active.

CPU tests (real parsers; fake tokenizer/MLM; no models):

```bash
python3 -m unittest discover -s tests -v
```

Test dependencies: numpy, pandas, scikit-learn, pyarrow, tree-sitter and the Python/Java/Go/Ruby/Rust grammar wheels. Kaggle instructions and dependency pins are in [RESEARCH_RUN.md](../RESEARCH_RUN.md). Source packaging: `python3 shared/package_source.py`; static checks: `python3 shared/audit_research.py`.

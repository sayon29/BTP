# Local validation — 2026-10-31

No pretrained StarCoder2-3B inference, model/dataset download, dependency installation, or Kaggle execution was performed locally.

- `python3 -m unittest discover -s tests -p 'test_spw*.py' -v`: 29 tests collected; 28 passed, one optional tiny random CPU StarCoder2 test skipped because torch/transformers are unavailable locally.
- `python3 -m unittest discover -s tests -p test_kaggle_setup.py -v`: 11 passed with mocked package metadata/pip. These verify exact CUDA pins, installation decisions, no-op repeated setup, correct dill preservation, and one-restart behavior.
- Notebook code cells parse; saved outputs are empty. Tests verify notebook wiring, both notebook copies, required CUDA preflight, optional adaptive input, scoring roles, environment output, and archive invocation.
- ZIP CRC, manifest hashes, exact ZIP membership, live-source byte equality, source pin, and dependency lock equality are checked.
- Source bootstrap executed from synthetic extracted files and a ZIP in temporary directories, including unique mount discovery, repeated loading, hash tampering and path traversal rejection.
- Archive checks cover required artifacts, SQLite integrity/sidecars, CRC, reserve space and destination isolation.
- Existing numerical tests verify **local adaptation** equations against straightforward calculations, token alignment, greedy extraction, shortening boundaries, determinism, cache extension/identity/recovery, manifest selection and calibration-only thresholds. Additional tests cover calibration/held-out-only inference and absent/mismatched adaptive comparisons.

The standalone upload package is also extracted to a fresh temporary directory and tested there. Notebook/ZIP-dependent checks skip there because those artifacts are delivered separately. No results or model data are bundled.

## Remaining limits

Published equations, algorithm and pretrained-model defaults remain unavailable. There is no verified published-equation test or full SPW-MIA implementation. Exact method reproduction requires the complete paper or authoritative source.

The optional random-model alignment test will run in Kaggle after setup; local analytical tests are not GPU numerical validation. The current corrected shared setup passed local decision tests but actual Kaggle package availability, T4 CUDA execution and throughput remain unverified. Smoke must measure runtime before assessing feasibility within 480 minutes; interrupted runs can resume. No attack metrics have been fabricated.

## Exact files changed in this task

Created:

- `notebooks/07_spw_code.ipynb`
- `spw_code_attack/archive.py`
- `spw_code_attack/METHODOLOGY.md`
- `spw_code_attack/VALIDATION.md`
- `tests/test_spw_workflow.py`

Modified/rebuilt:

- `README.md`
- `notebooks/05_spw_code.ipynb` (compatibility copy)
- `notebooks/spw_kaggle_source.zip`
- `spw_code_attack/README.md`
- `spw_code_attack/data.py`
- `spw_code_attack/evaluation.py`
- `spw_code_attack/package_source.py`
- `spw_code_attack/reproducibility.json`
- `spw_code_attack/requirements-kaggle.txt`
- `spw_code_attack/research_access.json`
- `spw_code_attack/runner.py`
- `spw_code_attack/source_manifest.json`

Pre-existing modifications to ReCaLL, Infilling and shared setup were inspected and reused as available; they were not edited in this task. Adaptive and Neighbourhood implementations were not changed.

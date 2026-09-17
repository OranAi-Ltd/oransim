# Research experiments

This repository hosts code for **OranSim: Outcome-Calibrated World Simulation for Pre-Launch Social Media Marketing Evaluation** and **Learning Collective Dynamics with Differentiable Gaussian Representations**.

| Study | Entry | Available evidence |
| --- | --- | --- |
| Pre-launch evaluation | [prelaunch](prelaunch/README.md) | Synthetic counterfactual audit; KuaiRand/Open Bandit policy evaluation; X5 randomized treatment ranking |
| Collective dynamics | [gaussian_population](gaussian_population/README.md) | Four aggregate evaluation windows; trained Gaussian, GRU and DeepAR models; controlled recovery experiments |

Install from the repository root with Python 3.10 or later:

```bash
python3 -m pip install -e '.[research,dev]'
PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/research -q
python3 backend/scripts/verify_research_artifacts.py
```

Training and evaluation commands are in `backend/scripts/`; reusable models live in `backend/oransim/world_model/`. Frozen evaluation output is under `experiments/*/results/`. Model snapshots and de-identified aggregate evaluation inputs are under `models/gaussian_population/`. New evaluation output goes to ignored `outputs/`.

The release includes public-data and synthetic experiments. Proprietary platform calibration datasets, credentials, internal documents, and manuscript files are excluded. Numeric XHS calibration weights and prepared-feature inference are available in [models/xhs_outcome](../models/xhs_outcome/README.md). Consequently, the closed-data calibration and campaign results of the OranSim study cannot be fully reproduced from this release.

Results retain the original experiment seeds and metrics. Historical `source_sha256` fields describe the code used at experiment time; namespace and path adaptation changes the released code hashes. `artifacts.json` records the actual released model/input/result bytes. Full historical evaluations are retained, including metrics where baselines outperform the proposed model.

Dataset and derived-artifact attribution: [DATA_LICENSES.md](DATA_LICENSES.md). The source-code license remains Apache-2.0; dataset-derived artifacts have the separate terms described there.

## Release validation

The release was checked with Python 3.12 and PyTorch 2.11 on CPU. Frozen-model replay completed all four windows (90 neural evaluations and 12 history selections); all 172,224 numeric values matched the archived behavior results exactly. The numerical aggregation checks and all four count-baseline runs completed. A 500-scenario synthetic counterfactual smoke run exercised the public generation/evaluation path; it does not replace the archived full experiment.

The XHS v3/v3.1 export reproduced original fitted-model predictions and PCA projections exactly on synthetic numeric inputs. No proprietary training or validation records were used in release verification. Full training was not rerun. The test suite includes the existing application tests and the research tests; CPU research CI installs its own optional dependencies.

# Collective dynamics with differentiable Gaussian representations

## Frozen-model replay

Four windows are included: KuaiRand standard recommendation and randomized exposure, and Online Retail II 2010 and 2011. Inputs contain daily cohort/campaign counts, joint binary-pattern counts and features; raw customer, creator and transaction records are omitted. Gaussian/GRU snapshots are JSON parameters with training-end state. DeepAR `.pt` files contain tensor state dictionaries and are loaded with `weights_only=True`.

```bash
python3 backend/scripts/evaluate_population_behavior.py \
  --out outputs/gaussian_population/behavior \
  --cache outputs/gaussian_population/behavior-cache
python3 backend/scripts/evaluate_population_count_baselines.py \
  --out outputs/gaussian_population/count-baselines
```

The behavior replay restores the published checkpoints, selects history baselines using validation data, and scores the held-out windows. It reads the DeepAR selection metadata from `results/external-baselines-2026-09-16/selections.json`. Compare the replay's `results.json` with `results/behavior-evaluation-2026-09-16/results.json`.

The frozen seeds are 20260915, 20260916 and 20260917. One-, three- and seven-day forecasts score the endpoint day. KuaiRand user hash bucket 1 supplies the aggregate benchmark; bucket 0 belongs to the earlier development experiments. Calendar dates overlap, so the later bucket is a population-sample replication. Joint NLL and Brier scores describe observed joint marks; conditional-rate and end-to-end count metrics use different exposure weights, as specified by each protocol JSON.

## Training

DeepAR can be retrained directly from the included aggregate records:

```bash
python3 backend/scripts/evaluate_population_deepar.py \
  --out outputs/gaussian_population/deepar \
  --snapshots outputs/gaussian_population/deepar-checkpoints
```

Gaussian models and original cohort construction can be retrained from the upstream public datasets. Download and extract [KuaiRand-Pure](https://zenodo.org/records/10439422) so its four required CSVs are in the directory passed to `--data`:

```bash
python3 backend/scripts/evaluate_population_readiness.py \
  --data data/kuairand/raw/KuaiRand-Pure/KuaiRand-Pure/data \
  --out outputs/gaussian_population/kuairand \
  --private outputs/gaussian_population/kuairand-checkpoints
```

The files are `log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv`, `log_random_4_22_to_5_08_pure.csv`, and `video_features_basic_pure.csv`. Use `--help` for the epoch budget; frozen protocols in `results/paper-readiness/kuairand/` record the original training settings. `--private` is the inherited CLI name for a local checkpoint/output directory.

For Online Retail II, download the [UCI workbook](https://archive.ics.uci.edu/dataset/502/online+retail+ii), placing `online_retail_II.xlsx` under `data/online_retail_ii/raw/`:

```bash
python3 backend/scripts/evaluate_retail_aggregation.py \
  --out outputs/gaussian_population/retail \
  --private outputs/gaussian_population/retail-checkpoints
```

The runner creates its Parquet cache from both workbook sheets. Retail observations are completed positive-price invoices and basket marks; visits, unpurchased alternatives and stock are unobserved.

## Controlled experiments and numerical checks

```bash
python3 backend/scripts/evaluate_differentiable_dynamics.py --out outputs/gaussian_population/dynamics
python3 backend/scripts/evaluate_differentiable_recovery.py --out outputs/gaussian_population/choice
python3 backend/scripts/check_population_aggregation_theory.py --out outputs/gaussian_population/theory.json
```

`results/differentiable-market-v3/` preserves the original recovery, quadrature and gradient audits. Choice snapshots are in `models/gaussian_population/controlled/choice/`. Original benchmark results and protocols are under `results/paper-readiness/`; the supplemental DeepAR comparison and behavior scoring are in their dated directories. Retraining may differ with library versions; use frozen-model replay to verify released weights without refitting.

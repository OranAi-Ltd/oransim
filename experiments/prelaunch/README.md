# OranSim pre-launch evaluation

The public experiment scripts cover known synthetic counterfactuals, logged exposure-policy evaluation and randomized treatment ranking. Closed platform calibration data are excluded. The v3 and v3.1 calibration weights are released as native LightGBM trees plus PCA parameters; see [checkpoint usage and feature contract](../../models/xhs_outcome/README.md).

## Synthetic counterfactual recovery

Install optional causal estimators, generate synthetic scenarios, then run the audit:

```bash
python3 -m pip install -e '.[research,causal-research]'
python3 backend/scripts/gen_synthetic_data.py --what scenarios \
  --out outputs/prelaunch/synthetic --n-kols 1000 --n-scenarios 20000 --seed 2027
python3 backend/scripts/run_public_counterfactual_audit.py \
  --data outputs/prelaunch/synthetic/scenarios_v0_1.jsonl \
  --out-dir outputs/prelaunch/public_cf
```

The audit reports factual and counterfactual outcomes, individual treatment-effect error, and cohort calibration. Results apply to the generator's known mechanism. Frozen metrics and cohort summaries are in `results/public_cf/`.

## Public exposure and treatment protocols

Acquire datasets from their upstream providers; raw records are not distributed here.

- [KuaiRand-Pure](https://github.com/chongminggao/KuaiRand): randomized video exposure. `prepare_kuairand.py` defaults to the randomized slice; retain this default for the policy-value experiment.
- [X5 RetailHero](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html): randomized marketing treatment, customer features and binary outcomes. Use `prepare_x5.py --raw-dir <downloaded-directory>` to construct `data/x5/processed/uplift.parquet`.
- [Open Bandit Pipeline](https://github.com/st-tech/zr-obp): random and Bernoulli Thompson Sampling logs. The historical C3 implementation uses `obp==0.4.1`; install it in a compatible separate Python environment if its legacy dependency constraints conflict with the main environment.

```bash
python3 backend/scripts/prepare_kuairand.py --raw-dir data/kuairand/raw
python3 backend/scripts/run_oranbench_baselines.py --cards C1 --seed 42
python3 backend/scripts/run_oranbench_baselines.py --cards C2 --seeds 42,137,256
python3 backend/scripts/run_oranbench_baselines.py --cards C3 --seed 42
python3 backend/scripts/run_x5_budget_case_study.py --seeds 42 137 256 \
  --out-dir outputs/prelaunch/x5
```

`results/protocols/` records the exposure-policy and treatment-ranking evaluations; `results/x5/` contains reach-budget curves and bootstrap summaries. Target policies, propensity diagnostics and seed-specific results are preserved in the JSON. X5 ranking evidence does not provide observed individual counterfactual outcomes.

# Model Card — Oransim Model Zoo

> **Status as of v0.1.0-alpha:** architecture + training + inference code ship
> for all models; pretrained weights (where applicable) land starting v0.2.

Oransim ships a model zoo across two layers:

1. **World models** (funnel KPI prediction): Causal Transformer (primary),
   LightGBM Quantile (baseline)
2. **Diffusion forecasters** (14-day cascade): Causal Neural Hawkes (primary),
   Parametric Hawkes (baseline)

---

## 1. CausalTransformerWorldModel (primary)

- **Family**: Causal Transformer for Treatment-Effect Prediction
- **Architecture**: 6-layer × 256-dim encoder with explicit
  *treatment / covariate / outcome* token factorization, DAG-aware attention
  bias, per-arm counterfactual heads, representation-balancing regularizer
- **Parameters**: ≈ 4.3M (default config)
- **Code**: `oransim.world_model.CausalTransformerWorldModel`
- **Config**: `CausalTransformerWMConfig`
- **Inference**: requires PyTorch (`pip install 'oransim[ml]'`)
- **Weights**: *coming_soon* — will train on 100k synthetic dataset and
  ship at https://github.com/ORAN-cgsj/oransim/releases starting v0.2

### Key references

- CaT — Melnychuk, Frauen, Feuerriegel, ICML 2022 (arXiv:2204.07258)
- CausalDAG-Transformer — Zhang et al. 2023
- TARNet — Shalit, Johansson, Sontag, ICML 2017
- Dragonnet — Shi, Blei, Veitch, NeurIPS 2019
- BCAUSS — Tesei et al. 2021
- CInA — Arik & Pfister, NeurIPS 2023
- HSIC — Gretton, Bousquet, Smola, Schölkopf, 2005

### Intended use

- Primary: marketing KPI prediction under explicit treatment assignment
- Primary: counterfactual queries (`do(T = arm)`) for scenario comparison
- Primary: reducing confounding via representation balancing when
  treatment is non-random (most real-world marketing data)
- Not intended: point prediction without uncertainty bands, attribution
  modeling for fraud detection, or non-causal general regression tasks

### Known limitations

1. Counterfactual heads use discretized treatment arms — continuous
   interventions (budget = $127,438 vs. budget = $150,000) require
   discretization to the nearest arm or use of the LightGBM baseline with
   direct budget features.
2. DAG attention bias is only as good as the provided DAG. Oransim ships
   the 64-node Pearl SCM as the default DAG; custom DAGs can be installed
   via `model.set_dag_from_edges(...)`.
3. Balancing loss (HSIC) regularises but does not guarantee bias removal
   under strong hidden confounding — standard caveat for observational
   causal inference.

---

## 2. LightGBMQuantileWorldModel (baseline)

- **Family**: Gradient-boosted quantile regression (Koenker 2005, Ke et al. 2017)
- **Architecture**: ``len(kpis) × len(quantiles)`` independent LightGBM
  boosters, optional PCA feature projection (default 32 components)
- **Parameters**: ≈ 800k LightGBM leaves (depends on data size)
- **Code**: `oransim.world_model.LightGBMQuantileWorldModel`
- **Config**: `LightGBMWMConfig`
- **Inference**: sub-millisecond CPU, zero GPU dependency
- **Weights**: *coming_soon* — same 100k synthetic training corpus

### Intended use

- Production latency-sensitive deployments
- Ablation baseline vs. Causal Transformer on OrancBench
- Fallback when PyTorch is unavailable

### Known limitations

- No explicit counterfactual head (treatment effect must be inferred
  by feature perturbation, a weaker contract than the Transformer's
  per-arm heads)
- Quantile regressors are trained independently → can cross (P35 > P65);
  default pipeline includes a post-hoc sort step, documented in v0.2

---

## 3. CausalNeuralHawkesProcess (primary diffusion)

- **Family**: Transformer Hawkes Process with causal / treatment typing
- **Architecture**: 4-layer × 128-dim self-attention encoder over
  (time, event_type, treatment_type), softplus intensity decoder; NLL
  training with Monte Carlo compensator estimator
- **Parameters**: ≈ 900k (default config)
- **Code**: `oransim.diffusion.CausalNeuralHawkesProcess`
- **Config**: `CausalNeuralHawkesConfig`
- **Inference**: requires PyTorch (`pip install 'oransim[ml]'`)
- **Weights**: *coming_soon* — will train on synthetic event streams and
  ship at release starting v0.2

### Key references

- Neural Hawkes — Mei & Eisner, NeurIPS 2017
- Transformer Hawkes — Zuo, Jiang, Zheng et al., ICML 2020
- Intensity-Free TPPs — Shchur et al., ICLR 2020
- Neural Spatio-Temporal Point Processes — Chen et al., ICLR 2021
- Counterfactual TPPs — Geng, Xu, Huang et al., NeurIPS 2022
- Counterfactual TPPs — Noorbakhsh & Rodriguez, 2022
- Ogata thinning sampler — Ogata, 1981

### Intended use

- Primary: 14-day multivariate cascade forecasting (impressions, likes,
  reshares, comments, saves, conversions)
- Primary: counterfactual rollouts — e.g., "what if we had stopped
  boosting on day 3" (`mute_at_min`) or "what if the boost factor had
  been 2× / 0.5×" (`treatment_boost_factor`)
- Not intended: sub-minute real-time forecasts (inference involves
  Monte Carlo rollouts ≈ 100-500ms per scenario)

### Known limitations

1. Forecast sampler uses a simple thinning scheme — replaced with batched
   importance sampling in v0.5 for a ~5× speedup
2. Intervention grammar in v0.1-v0.2 supports `{mute_at_min,
   treatment_boost_factor}`; richer per-event interventions (force event
   type, force time) land in v0.5

---

## 4. ParametricHawkes (baseline diffusion)

- **Family**: Classical multivariate Hawkes process (Hawkes 1971) with
  exponential kernels
- **Architecture**: ``len(event_types) × len(event_types)`` excitation
  matrix + per-type baseline rates + per-type decay constants
- **Parameters**: ≈ 50 (K² + 2K where K = number of event types)
- **Code**: `oransim.diffusion.ParametricHawkes`
- **Config**: `ParametricHawkesConfig`
- **Inference**: millisecond CPU, zero-dependency
- **Weights**: *coming_soon* — closed-form EM estimator; parameters small
  enough to ship directly

### Intended use

- OrancBench baseline against Causal Neural Hawkes
- Minimum-viable diffusion forecast when no ML stack is available
- Closed-form log-likelihood for likelihood-based calibration checks

---

## 5. Training Data (shared across all models)

- **Source**: Synthetic data generated by `scripts/gen_synthetic_data.py`
  (landing with v0.2)
- **Size**: 100k samples (planned)
- **License**: Apache-2.0 / CC0
- **Full documentation**: see [`data_card.md`](data_card.md)

**Crucially**: all shipped weights train on synthetic data. No real
KOL / note / user / brand data enters the open-source model zoo.
OranAI Enterprise Edition separately trains on proprietary real-world
data; those benchmarks are published under NDA and are not part of the
OSS release.

---

## 6. Reporting & Support

- Bug reports: https://github.com/ORAN-cgsj/oransim/issues
- Security: `cto@orannai.com` (see `SECURITY.md`)
- Enterprise inquiries: `cto@orannai.com`

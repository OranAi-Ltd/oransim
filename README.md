<div align="center">
<img src="assets/wordmark.svg" alt="Oransim" width="640"/>

### Causal Digital Twin for Marketing at Scale

<p>
  <a href="https://github.com/ORAN-cgsj/oransim/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/ORAN-cgsj/oransim?color=blue"></a>
  <a href="https://pypi.org/project/oransim/"><img alt="PyPI" src="https://img.shields.io/pypi/v/oransim?label=PyPI"></a>
  <a href="https://pypi.org/project/oransim/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/oransim"></a>
  <a href="https://github.com/ORAN-cgsj/oransim/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ORAN-cgsj/oransim/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/ORAN-cgsj/oransim/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/ORAN-cgsj/oransim?style=social"></a>
  <a href="https://oran.cn/oransim"><img alt="Website" src="https://img.shields.io/badge/website-oran.cn-FF6B35"></a>
</p>

<p>
  <strong>🇬🇧 English</strong> · <a href="README.zh-CN.md">🇨🇳 中文</a>
</p>

<p><em>Reason. Simulate. Intervene.<br/>Predict any marketing decision before you spend a dollar.</em></p>
</div>

---

## TL;DR

**Oransim** is an open-source **causal digital twin** for marketing performance prediction. Upload a creative, a budget, and a KOL list — in 60 seconds, get:

- 📈 Predicted impressions, clicks, conversions, ROI (with calibrated P35/P50/P65 uncertainty)
- 🔄 **Counterfactuals** — "what if I'd used a different creative / more budget / another KOL?" — via Pearl-style `do()` and a dedicated counterfactual head
- 🗣️ Virtual-user feedback in natural language (10 LLM-powered personas)
- 📊 14-day diffusion curve with intervention rollouts
- 🧭 Recommended next actions, ranked

### The causal stack

Oransim is built **causal-first** — counterfactual reasoning is first-class, not an afterthought:

- 🧠 **Causal Transformer World Model** — 6-layer multi-head self-attention with explicit *treatment / covariate / outcome* factorization, DAG-aware attention bias, per-arm counterfactual heads, and a representation-balancing loss. Draws from recent work: **CaT** (Melnychuk et al. ICML 2022), **CausalDAG-Transformer**, **BCAUSS**, **CInA** (Arik & Pfister NeurIPS 2023), **TARNet / Dragonnet**. ([arch details](#causal-transformer-world-model))
- ⚡ **Causal Neural Hawkes Process** — Transformer-parameterized temporal point process for 14-day diffusion with *treatment vs control* event typing and intervention-aware intensity. Follows **Mei & Eisner (NeurIPS 2017)**, **Zuo et al. (ICML 2020)**, **Geng et al. (NeurIPS 2022)** on counterfactual TPPs. ([arch details](#causal-neural-hawkes-process))
- 🌐 **64-node Structural Causal Model** — Pearl's 3-step counterfactual evaluation (abduction → action → prediction) over a hand-designed marketing funnel graph (117 edges), with mediators for group discourse (Sunstein 2017) and information cascades.
- 👥 **1M-agent population** — Iterative Proportional Fitting (IPF, Deming & Stephan 1940) baseline calibrated to demographic priors; pluggable `PopulationSynthesizer` interface with Bayesian-network (v0.2), CTGAN (v0.5), and Causal-DAG-guided TabDDPM (v1.0 research) variants on the roadmap. Top-10k salient agents get LLM personas for qualitative feedback.
- 🧪 **LightGBM Quantile baseline** — fast zero-dependency fallback, three quantile regressors (P35/P50/P65) per KPI. Retained for production latency targets and benchmark comparison.

**Plug-and-play out of the box** — v0.1.1-alpha ships the synthetic demo corpus (2.3 MB — 200 KOLs, 2k scenarios, 100 event streams) **and a pretrained LightGBM demo pkl** (2.7 MB, R² 0.69–0.89 on synthetic eval). Clone, install, set an LLM API key, and the full prediction pipeline works immediately — no separate training step required. The research-grade Causal Transformer and Causal Neural Hawkes weights train on the 100k synthetic dataset and ship starting v0.2; today v0.1.1-alpha includes the full architecture, training loop, and inference code — run `pip install 'oransim[ml]'` to unlock them.

> 🏢 **Enterprise edition** — OranAI trains the same architectures on **continuously-updated proprietary real-world data** (1M+ labeled campaigns), with **higher-performance vertical model variants** (beauty / fashion / 3C / F&B / luxury / auto) and **bespoke model customization** (on-premise, domain-specific DAGs, branded persona libraries). Contact `cto@orannai.com`.

---

## 🚀 Quickstart (60 seconds)

```bash
# 1. Clone and install
git clone https://github.com/ORAN-cgsj/oransim.git
cd oransim
pip install -e '.[dev]'

# 2. Run backend (mock mode — no API key required)
LLM_MODE=mock PORT=8001 python backend/run.py &

# 3. Run frontend
python -m http.server 8090 --directory frontend

# 4. Open http://localhost:8090 → click "🔥 Trending Preset" → "🚀 Predict"
```

To use real LLMs, set `LLM_MODE=api` + `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL`. See [docs/en/quickstart.md](docs/en/quickstart.md).

> **Note:** v0.1.0-alpha ships skeleton code only. Full backend (including the web demo and screenshots) lands in v0.2 (see [ROADMAP.md](ROADMAP.md)). Follow the repo to get notified.

---

## ✨ Why Oransim

|  | Traditional Analytics | AutoML / Black-Box Predictors | **Oransim** |
|---|---|---|---|
| World model | Rule-based | Tree / GBDT / generic DNN | ✅ **Causal Transformer** (CaT / CausalDAG-Transformer) with treatment/covariate/outcome factorization |
| Counterfactuals | ❌ | ❌ | ✅ **Per-arm counterfactual heads** (TARNet / Dragonnet) + Pearl 3-step `do()` evaluation |
| Causal bias reduction | ❌ | ❌ | ✅ **Representation balancing** loss (HSIC / adversarial-IPTW, BCAUSS) |
| Causal graph structure | ❌ | ❌ | ✅ **DAG-aware attention bias** over a 64-node Pearl SCM (117 edges) |
| Diffusion forecasting | Linear decay | Generic time-series DNN | ✅ **Causal Neural Hawkes** (Transformer Hawkes + Geng 2022 intervention TPP) |
| Agent-level simulation | ❌ | ❌ Aggregate only | ✅ 1M IPF-calibrated virtual consumers + 10k LLM persona agents |
| Platform coverage | Single platform | Single platform | ✅ **PlatformAdapter × DataProvider** two-axis extension |
| Budget saturation | ❌ Linear | ❌ Linear | ✅ Hill saturation (Dubé & Manchanda 2005) + frequency fatigue (Naik & Raman 2003) |
| Interpretability | Moderate | Low (SHAP at best) | ✅ SCM paths + per-head attention + agent reasoning traces |
| Amortized inference | ❌ | Per-problem retrain | ✅ **In-context amortization** (CInA, Arik & Pfister NeurIPS 2023) |
| Cost | Licensing fees | API costs | ✅ Apache-2.0 + self-hosted (`[ml]` extras optional) |

Built by practitioners frustrated with both ends of the market — academic simulators that don't ship, and enterprise tools that don't explain.

---

## 🏗️ Architecture

<div align="center">
<img src="assets/architecture.svg" alt="Oransim architecture diagram" width="100%"/>
</div>

A typical prediction request flows: **Creative + Budget** → **PlatformAdapter** (pulls data via pluggable **DataProvider**) → **Causal Transformer World Model** (factual + per-arm counterfactual quantile predictions) + **Agent Layer** (1M IPF + 10k LLM personas) → **Causal Engine** (64-node Pearl SCM + 3-step `do()` counterfactuals) → **Causal Neural Hawkes** (14-day diffusion with intervention rollout) → **Prediction JSON** (14–19 schemas). *LightGBM quantile and parametric Hawkes are available as fast baselines via the registry.*

Two-axis extensibility:
- **Platform** axis — XHS today; TikTok / Instagram / YouTube Shorts / Douyin on roadmap
- **Data Provider** axis — pluggable per platform (Synthetic / CSV / JSON / OpenAPI / your own)

See [`docs/en/architecture.md`](docs/en/architecture.md) for the full design.

---

## 🌐 Platform Adapter Matrix

| Platform             | Region   | Status  | Data Provider                       | World Model          | Milestone |
|----------------------|----------|---------|-------------------------------------|----------------------|-----------|
| 🔴 XHS / RedNote     | Greater China | ✅ v1   | Synthetic / CSV / JSON / OpenAPI | Causal Transformer + LightGBM baseline | — |
| ⚫ TikTok            | Global   | 🟡 stub | —                                 | —                    | v0.5 (Q3 2026) |
| 🟣 Instagram Reels   | Global   | 🟡 stub | —                                 | —                    | v0.5 (Q4 2026) |
| 🔴 YouTube Shorts    | Global   | 🟡 stub | —                                 | —                    | v0.7 (Q1 2027) |
| 🔵 Douyin            | Greater China | 🟡 stub | —                                 | —                    | v0.5 (Q3 2026) |
| ⚪ Twitter / X       | Global   | 📋 planned | —                             | —                    | v0.5 |
| 📺 Bilibili          | Greater China | 📋 planned | —                        | —                    | v1.0 |
| ✒️ LinkedIn          | Global   | 📋 planned | —                             | —                    | v1.0 |

**Want another platform?** Open an [Adapter Request](https://github.com/ORAN-cgsj/oransim/issues/new?template=adapter_request.yml) — we prioritize based on community demand.

---

## 📊 What You Get — 14 to 19 Schemas

A single `/api/predict` call returns structured outputs across these schemas:

1. **total_kpis** — aggregate impressions / clicks / conversions / cost / revenue / CTR / CVR / ROI with P35/P50/P65 bands
2. **per_platform** — KPIs broken down per platform adapter
3. **per_kol** — KOL-level attribution
4. **diffusion_curve** — 14-day daily impression/engagement forecast (Causal Neural Hawkes; parametric Hawkes as baseline)
5. **cate** — Conditional Average Treatment Effect across agent demographics
6. **counterfactual** — "What if" branching: alternative creative / budget / KOL
7. **soul_feedback** — 10 LLM persona reactions in natural language
8. **group_chat** — simulated group conversation dynamics (Sunstein 2017 polarization)
9. **discourse** — second-wave mediator impact estimation
10. **final_report** — LLM-generated executive summary
11. **verdict** — top-line recommendation (greenlight / optimize / kill)
12. **kol_optimizer** — optimal KOL mix given objective
13. **kol_content_match** — creative × KOL compatibility scoring
14. **tag_lift** — incremental performance from tag/targeting choices
15. **mediator_impact** — path analysis from discourse/group_chat to funnel
16. **brand_memory** — longitudinal brand preference updates
17. **sandbox_snapshot** — serialized session state for "undo / redo"
18. **audit_trace** — explainability — which agents, which paths, which weights
19. **benchmark** — performance against OrancBench

See [`docs/en/schemas/`](docs/en/schemas/) for JSON schema definitions.

---

## 🧠 Under the Hood

<details>
<summary><b>Structural Causal Model (SCM)</b> — 64 nodes, 117 edges</summary>

Pearl's SCM framework (Pearl 2009) with three-step counterfactual evaluation:
1. **Abduction** — update latent noise terms given evidence
2. **Action** — apply `do()` intervention
3. **Prediction** — propagate through the modified SCM

The graph is hand-designed by domain experts covering the marketing funnel from impression → awareness → consideration → conversion → repeat purchase → brand memory, with mediators for group discourse (Sunstein 2017) and information cascades (Bikhchandani et al. 1992).
</details>

<details>
<summary><b>Agent Population</b> — 1M IPF-calibrated virtual consumers</summary>

Generated via Iterative Proportional Fitting (IPF / Deming-Stephan 1940) against real Chinese demographic distributions (age × gender × region × income × platform). Each agent carries:
- Demographics + psychographics
- Platform-specific engagement priors
- Niche/category affinity vectors
- Time-of-day activity curves
- Social graph embeddings
</details>

<details>
<summary><b>Soul Agents</b> — 10k LLM personas for qualitative feedback</summary>

The top-10k most salient agents for a scenario are upgraded to LLM-backed personas. Default model: `gpt-5.4`. Each persona:
- Generates a persona card from its demographic vector
- Evaluates the creative (reaction / emotional response / intent)
- Optionally participates in simulated group chats (Sunstein 2017 group polarization)
- Feeds second-wave mediators back into the causal graph

Cost controlled via:
- In-flight request coalescing (leader/follower dedup pattern)
- Persona card caching
- Configurable `SOUL_POOL_N` (default 100 for demo; production tiers scale via Ray, see roadmap)
</details>

<details id="causal-transformer-world-model">
<summary><b>Causal Transformer World Model</b> — primary (research-grade)</summary>

A 6-layer × 256-dim causal Transformer that ingests heterogeneous campaign features and predicts three quantile levels (P35/P50/P65) for each funnel KPI. Architecture lifts ideas from the recent causal-Transformer literature:

- **Token-type factorization** (CaT, Melnychuk et al. ICML 2022) — inputs split into *Covariate* (platform, demographic, time), *Treatment* (creative embedding, budget, KOL), and *Outcome* (KPIs) tokens with distinct type embeddings
- **DAG-aware attention** (CausalDAG-Transformer) — attention mask derived from the 64-node Pearl SCM restricts each token to attend to topological ancestors; per-head learnable gate on the bias
- **Per-arm counterfactual heads** (TARNet, Shalit et al. ICML 2017 / Dragonnet, Shi et al. NeurIPS 2019) — one quantile head per discrete treatment arm enables `predict_factual` vs `predict_counterfactual(do(T=t'))` with a single forward pass
- **Representation balancing** (BCAUSS + CaT) — HSIC (Gretton et al. 2005) or adversarial-IPTW loss decorrelates the learned representation from treatment assignment, reducing bias in counterfactual predictions
- **In-context amortization** (CInA, Arik & Pfister NeurIPS 2023, optional) — model can condition on a context set of prior campaigns for amortized zero-shot causal inference

Core component: `oransim.world_model.CausalTransformerWorldModel`. Training loop, counterfactual rollout, and save/load are shipped in v0.1.0-alpha; pretrained weights land in v0.2.

```python
from oransim.world_model import get_world_model, CausalTransformerWMConfig

wm = get_world_model("causal_transformer", config=CausalTransformerWMConfig(
    dag_attention_bias=True,
    balancing_loss="hsic",
    use_counterfactual_head=True,
))
pred = wm.predict(features)                         # factual
cf = wm.counterfactual(features, arm_idx=2)         # do(T = arm 2)
```

*Requires* `pip install 'oransim[ml]'` (brings in PyTorch). Falls back gracefully to LightGBM if torch is unavailable.
</details>

<details>
<summary><b>LightGBM Quantile World Model</b> — fast baseline</summary>

Three quantile regressors (P35, P50, P65) per KPI. Sub-millisecond inference, zero GPU requirement. Feature engineering includes creative embeddings (OpenAI `text-embedding-3-small`), platform priors, KOL features, temporal signals, and PCA-reduced behavioral features. Refs: Ke et al. 2017 (LightGBM), Koenker 2005 (Quantile Regression).

Kept as the production default until the Causal Transformer checkpoints ship in v0.2. Also used as an ablation baseline in OrancBench.

```python
wm = get_world_model("lightgbm_quantile")
```
</details>

<details>
<summary><b>Budget Model</b> — Hill saturation + frequency fatigue</summary>

Instead of naive linear budget scaling:

$$\text{effective\_impr\_ratio}(x) = \frac{(1+K) \cdot x}{K + x}$$

Michaelis-Menten / Hill saturation (Dubé & Manchanda 2005), combined with frequency fatigue (Naik & Raman 2003) on CTR/CVR:

$$\text{ctr\_decay}(r) = \max(0.5, 1.0 - 0.08 \cdot \max(0, \log_2 r))$$

This captures diminishing returns, an optimal budget point, and realistic campaign dynamics.
</details>

<details id="causal-neural-hawkes-process">
<summary><b>Causal Neural Hawkes Process</b> — primary diffusion forecaster</summary>

Transformer-parameterized neural temporal point process for 14-day cascading engagement forecasting, with first-class support for counterfactual rollouts under `do()` interventions.

Architectural references:

- **Mei & Eisner (NeurIPS 2017)** — *The Neural Hawkes Process* — continuous-time neural intensity function, foundation of the field
- **Zuo et al. (ICML 2020)** — *Transformer Hawkes Process* — self-attention encoder replacing the original CT-LSTM; directly the backbone of this implementation
- **Shchur et al. (ICLR 2020)** — *Intensity-Free Learning of TPPs* — closed-form inter-event-time head for fast sampling
- **Chen et al. (ICLR 2021)** — *Neural Spatio-Temporal Point Processes* — Monte Carlo estimator for the log-likelihood compensator
- **Geng et al. (NeurIPS 2022)** — *Counterfactual Temporal Point Processes* — the intervention semantics for marked point processes
- **Noorbakhsh & Rodriguez (2022)** — *Counterfactual Temporal Point Processes* — formalizes `do()` queries on event streams

Explicit treatment/control event typing (`organic` vs `paid_boost`) and an intervention-aware intensity decoder enable queries like "what if we had stopped boosting on day 3" via a counterfactual rollout loop.

Core component: `oransim.diffusion.CausalNeuralHawkesProcess`. Architecture, training loop (NLL with MC compensator), forecast sampler (Ogata thinning), and counterfactual rollout are shipped in v0.1.0-alpha; pretrained weights land in v0.2.

```python
from oransim.diffusion import get_diffusion_model

nh = get_diffusion_model("causal_neural_hawkes")
factual = nh.forecast(seed_events=[(0, "impression"), (12, "like")])
cf = nh.counterfactual_forecast(
    seed_events,
    intervention={"mute_at_min": 4320}  # stop boosting 3 days in
)
```

*Requires* `pip install 'oransim[ml]'`.
</details>

<details>
<summary><b>Parametric Hawkes</b> — classical baseline</summary>

Exponential-kernel multivariate Hawkes process (Hawkes 1971). Closed-form intensity and log-likelihood; Ogata (1981) thinning sampler. Zero-dependency fallback and the baseline against which the Causal Neural Hawkes is evaluated on OrancBench.

```python
ph = get_diffusion_model("parametric_hawkes")
```
</details>

<details>
<summary><b>Sandbox</b> — incremental recomputation for "what if"</summary>

Scenario sessions persist state so users can iterate: "change budget from 100k to 150k, how does ROI move?" Incremental recomputation avoids redoing the full agent simulation when only budget changes. The 1M-agent pool is cached; counterfactual evaluation uses union-semantics CATE over reached vs. unreached populations.
</details>

---

## 📈 Benchmarks

Phase 1 benchmarks are based on **100k synthetic samples** — see [`data/models/data_card.md`](data/models/data_card.md) for the data-generating process.

| Metric | R² (synthetic) | Baseline (linear) | Notes |
|--------|---------------|-------------------|-------|
| `second_wave_click`     | 0.30 | 0.18 | PRS quantile median |
| `first_wave_conversion` | 0.33 | 0.21 | PRS quantile median |
| `cascade_lift`          | 0.39 | 0.25 | Second-wave mediator |
| `roi_point_estimate`    | 0.33 | 0.19 | Single-shot regression |
| `retention_7d`          | 0.29 | 0.17 | Longitudinal |

> ⚠️ **Reproducibility disclaimer** — these numbers reflect synthetic data. Real-world performance depends on (1) data quality of your chosen DataProvider, (2) platform match, (3) vertical/industry. **OranAI Enterprise Edition** trains on proprietary real-world data and publishes separate benchmarks under NDA.

See [`docs/en/benchmarks/`](docs/en/benchmarks/) for the full protocol.

---

## 🗺️ Roadmap — Highlights

See [ROADMAP.md](ROADMAP.md) for the full 3-horizon × 8-theme plan. Teasers:

**v0.2 (Q3 2026) — shipping pretrained weights**
- 📦 Trained Causal Transformer + Causal Neural Hawkes checkpoints on the 100k synthetic corpus
- TikTok + Douyin adapter MVPs
- Docker Compose · MkDocs · CI

**v0.5 (Q4 2026 – Q1 2027)**
- 🎯 **Cross-platform transfer learning** — pretrain on XHS, fine-tune on TikTok
- 🎯 **Multi-LLM-format adapters** — native Anthropic Messages, Gemini, Bedrock, Qwen DashScope
- 🎯 **10k soul agents on Ray cluster**
- Instagram / YouTube Shorts / Douyin adapters MVP

**v1.0+ (2027)**
- 🎯 **Causal Foundation Model** — pretrain on 10M+ campaigns
- 🎯 **Closed-loop AI media buying** — real-time optimization with safety constraints
- 🎯 **Differential privacy + Federated learning** — for brand-proprietary training
- 15+ platforms, multi-modal creative understanding, vertical sub-benchmarks

---

## 🏢 OranAI Enterprise Edition

Oransim OSS ships on synthetic data for transparency and reproducibility. **OranAI Enterprise Edition** provides:

- 📊 **Real-world training data** — continuously updated 1M+ labeled campaigns across beauty, fashion, 3C, F&B, luxury, auto
- ⚡ **SLA-backed hosted inference** — 99.9% uptime, sub-second response
- 🎯 **Vertical world models** — beauty / fashion / electronics / F&B specialized calibration
- 🤝 **White-glove onboarding** — custom adapter development, integration support, training
- 🔒 **On-premise deployment** — with SOC 2 / ISO 27001 / GDPR compliance path
- 🎓 **Managed model updates** — no downtime model refresh as platforms evolve

**Contact:** `cto@orannai.com` · [Book a demo](mailto:cto@orannai.com?subject=Oransim%20Enterprise%20Demo)

---

## 🤝 Contributing

We love contributions — platform adapters, world-model improvements, docs, benchmarks, translations, bug fixes.

- **Start here**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Sign off commits** per [DCO](CONTRIBUTING.md#developer-certificate-of-origin-dco): `git commit -s`
- **Good first issues**: [see labels](https://github.com/ORAN-cgsj/oransim/issues?q=is%3Aissue+label%3A%22good+first+issue%22)
- **Platform adapter requests**: [file here](https://github.com/ORAN-cgsj/oransim/issues/new?template=adapter_request.yml)

By contributing, you agree your contribution is licensed under Apache-2.0. No CLA required.

---

## 📚 Citation

If you use Oransim in research, please cite:

```bibtex
@software{oransim2026,
  author       = {Yin, Fakong and {Oransim contributors}},
  title        = {Oransim: Causal Digital Twin for Marketing at Scale},
  version      = {0.1.0-alpha},
  date         = {2026-04-18},
  url          = {https://github.com/ORAN-cgsj/oransim},
  organization = {OranAI Ltd.}
}
```

See [CITATION.cff](CITATION.cff) for `cffconvert`-compatible metadata.

---

## 📜 License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

`Copyright (c) 2026 OranAI Ltd. (橙果视界（深圳）科技有限公司) and Oransim contributors.`

Third-party dependencies retain their original licenses. We are not affiliated with Xiaohongshu, ByteDance, Meta, Google, or any other platform mentioned in this repository.

---

## 💫 Team

Oransim is built by **[OranAI Ltd.](https://oran.cn)** (橙果视界（深圳）科技有限公司).

**Core Maintainers**
- **Fakong Yin** — CTO & Core Architect · [cto@orannai.com](mailto:cto@orannai.com) · [@ORAN-cgsj](https://github.com/ORAN-cgsj)

**Open roles** — we're hiring researchers (Causal ML, RL, Agent-based Simulation) and engineers (Platform, Infra). Reach out at [cto@orannai.com](mailto:cto@orannai.com).

Contributors appear on [`CONTRIBUTORS.md`](CONTRIBUTORS.md) (auto-generated).

---

## ⭐ Star History

<a href="https://star-history.com/#ORAN-cgsj/oransim&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ORAN-cgsj/oransim&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ORAN-cgsj/oransim&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ORAN-cgsj/oransim&type=Date" />
  </picture>
</a>

---

<div align="center">
Built with ☕ in Shenzhen by <a href="https://oran.cn">OranAI</a>. If Oransim helps your work, please ⭐ star the repo — it powers our open-source commitment.
</div>

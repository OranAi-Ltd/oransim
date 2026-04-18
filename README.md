<div align="center">
<img src="assets/wordmark.svg" alt="Oransim" width="640"/>

### Causal Digital Twin for Marketing at Scale

<p>
  <a href="https://github.com/OranAi-Ltd/oransim/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/OranAi-Ltd/oransim?color=blue"></a>
  <a href="https://pypi.org/project/oransim/"><img alt="PyPI" src="https://img.shields.io/pypi/v/oransim?label=PyPI"></a>
  <a href="https://pypi.org/project/oransim/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/oransim"></a>
  <a href="https://github.com/OranAi-Ltd/oransim/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/OranAi-Ltd/oransim/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/OranAi-Ltd/oransim/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/OranAi-Ltd/oransim?style=social"></a>
  <a href="https://oran.cn/oransim"><img alt="Website" src="https://img.shields.io/badge/website-oran.cn-FF6B35"></a>
</p>

<p>
  <strong>🇬🇧 English</strong> · <a href="README.zh-CN.md">🇨🇳 中文</a>
</p>

<p><em>Reason. Simulate. Intervene.<br/>Predict any marketing decision before you spend a dollar.</em></p>
</div>

---

<p align="center">
<img src="assets/screenshots/hero.png" alt="Oransim hero · 60-second prediction with counterfactual reasoning over a 1M-agent society" width="100%"/>
</p>

## TL;DR

> **Think of Oransim as Figma for ad prediction.** Paste your ad copy, move a slider, see *why* the prediction changes — and what would happen if you'd chosen differently. Counterfactual reasoning built in, not bolted on.

**Oransim** is an open-source **causal digital twin** for marketing performance prediction. Upload a creative, a budget, and a KOL list — in 60 seconds, get:

- 📈 Predicted impressions, clicks, conversions, ROI (with calibrated uncertainty bands)
- 🔄 **Counterfactuals** — *"what if I'd used a different creative / more budget / another KOL?"* — asked and answered in one click
- 🗣️ Virtual-user feedback in natural language (10 LLM-powered personas reading your actual copy)
- 📊 14-day diffusion curve with intervention rollouts (*"what if we'd stopped boosting on day 3?"*)
- 🧭 Recommended next actions, ranked

**Plug-and-play out of the box** — v0.2 ships the synthetic demo corpus (2.3 MB — 200 KOLs, 2k scenarios, 100 event streams) **and a pretrained LightGBM demo model** (R² 0.69–0.89 on synthetic eval). Clone, install, set an LLM API key, and the full prediction pipeline works immediately — no separate training step required.

The research-grade Causal Transformer and Causal Neural Hawkes ship **architecture + training loop + inference code** today (`pip install 'oransim[ml]'`). **Pretrained weights are deliberately held back** until [OrancBench v0.5](ROADMAP.md#v05--mid-q4-2026--q1-2027) introduces causal-native evaluation tasks where these architectures can demonstrate real structural advantage over the LightGBM baseline. Honesty over optics.

<details>
<summary><b>🧠 The causal stack</b> — research lineage for each component (click to expand)</summary>

Oransim is built **causal-first** — counterfactual reasoning is first-class, not an afterthought:

- 🧠 **Causal Transformer World Model** — 6-layer multi-head self-attention with explicit *treatment / covariate / outcome* factorization, DAG-aware attention bias, per-arm counterfactual heads, and a representation-balancing loss. Draws from recent work: **CaT** (Melnychuk et al. ICML 2022), **CausalDAG-Transformer**, **BCAUSS**, **CInA** (Arik & Pfister NeurIPS 2023), **TARNet / Dragonnet**. ([arch details](#causal-transformer-world-model))
- ⚡ **Causal Neural Hawkes Process** — Transformer-parameterized temporal point process for 14-day diffusion with *treatment vs control* event typing and intervention-aware intensity. Follows **Mei & Eisner (NeurIPS 2017)**, **Zuo et al. (ICML 2020)**, **Geng et al. (NeurIPS 2022)** on counterfactual TPPs. ([arch details](#causal-neural-hawkes-process))
- 🌐 **64-node Structural Causal Model** — Pearl's 3-step counterfactual evaluation (abduction → action → prediction) over a hand-designed marketing funnel graph (117 edges), with mediators for group discourse (Sunstein 2017) and information cascades.
- 👥 **1M-agent population** — Iterative Proportional Fitting (IPF, Deming & Stephan 1940) baseline calibrated to demographic priors; pluggable `PopulationSynthesizer` interface with Bayesian-network (v0.2), CTGAN (v0.5), and Causal-DAG-guided TabDDPM (v1.0 research) variants on the roadmap. Top-10k salient agents get LLM personas for qualitative feedback.
- 🧪 **LightGBM Quantile baseline** — fast zero-dependency fallback, three quantile regressors (P35/P50/P65) per KPI. Retained for production latency targets and benchmark comparison.

</details>

> 🏢 **Enterprise edition** — OranAI trains the same architectures on **continuously-updated proprietary real-world data** (1M+ labeled campaigns), with **higher-performance vertical model variants** (beauty / fashion / 3C / F&B / luxury / auto) and **bespoke model customization** (on-premise, domain-specific DAGs, branded persona libraries). Contact `cto@orannai.com`.

---

## 🚀 Quickstart (60 seconds)

```bash
# 1. Clone and install
git clone https://github.com/OranAi-Ltd/oransim.git
cd oransim
pip install -e '.[dev]'

# 2. Run backend (mock mode — no API key required)
LLM_MODE=mock PORT=8001 python backend/run.py &

# 3. Run frontend
python -m http.server 8090 --directory frontend

# 4. Open http://localhost:8090 → click "🔥 Trending Preset" → "🚀 Predict"
```

To use real LLMs, set `LLM_MODE=api` + `LLM_API_KEY` + `LLM_MODEL`. Select the native request format via `LLM_PROVIDER` (`openai` · `anthropic` · `gemini` · `qwen`); `openai` is the default and also covers DeepSeek / vLLM / any OpenAI-compat gateway. See [docs/en/quickstart.md](docs/en/quickstart.md) and [.env.example](.env.example).

> **Note:** v0.1.0-alpha ships skeleton code only. Full backend (including the web demo) lands in v0.2 (see [ROADMAP.md](ROADMAP.md)). Follow the repo to get notified.

---

## 🎬 See It In Action

<table>
<tr>
<td width="50%" valign="top">

**Three-panel working UI** — left: creative + budget + sliders · center: KPI / 百万智能体 / AI-群聊 tabs (+「更多 ›」dropdown for deep analysis) · right: per-persona LLM reactions.

<img src="assets/screenshots/main-three-col.png" alt="Three-panel prediction UI" width="100%"/>

</td>
<td width="50%" valign="top">

**Opinion-propagation through a 1M-agent society** — drop an ad copy, watch color-coded opinion waves (green=click / purple=high intent / red=skip / blue=curious) ripple outward from KOL seeds, cascading to their followers in real time.

<img src="assets/screenshots/society-100m.png" alt="Opinion propagation over 1M agents" width="100%"/>

</td>
</tr>
</table>

---

## ✨ Why Oransim

|  | Traditional Analytics | AutoML / Black-Box Predictors | **Oransim** |
|---|---|---|---|
| **Answers "why did the prediction change?"** | Partial — rule trace | ❌ Opaque (SHAP at best) | ✅ Every prediction traces back through the causal graph, per-agent reasoning, and attention paths |
| **Answers "what if I'd done X instead?"** | ❌ Re-run from scratch | ❌ Model doesn't know | ✅ Native counterfactual heads — ask `do(creative=B)` in one forward pass |
| **Sees individual user reactions** | Aggregates only | Aggregates only | ✅ 1M simulated consumers + 10k LLM personas reading your actual copy |
| **Predicts 14-day diffusion + intervention** | Linear decay | Generic time-series | ✅ Self-exciting point process that handles "what if we stopped boosting on day 3" |
| **Realistic budget curves** | ❌ Linear = 2× budget = 2× results | ❌ Same | ✅ Diminishing returns + frequency fatigue (real-world marketing economics) |
| **Removes spurious correlations** | ❌ | ❌ | ✅ Representation balancing loss decorrelates learned features from treatment assignment |
| **Transfers to a new campaign without retraining** | ❌ Redo the analysis | ❌ Per-problem retrain | ✅ In-context amortization — model conditions on your prior campaigns at inference time |
| **Multiple platforms** | Single platform | Single platform | ✅ 5 adapters shipped (XHS / TikTok / IG / YouTube / Douyin), 2-axis extensible |
| **Cost** | Per-seat licensing | API tokens per call | ✅ Apache-2.0 · self-hosted · free |

<details>
<summary>Technical references for each row</summary>

- *Why explanation*: Pearl SCM path tracing (64 nodes, 117 edges) + per-head attention maps + agent reasoning traces
- *Counterfactual heads*: TARNet (Shalit ICML 2017), Dragonnet (Shi NeurIPS 2019); Pearl 3-step abduction → action → prediction
- *LLM personas*: top-10k salient agents upgraded to LLM-backed personas (Park et al. 2023 Generative Agents)
- *14-day diffusion*: Causal Neural Hawkes (Mei & Eisner 2017 + Zuo ICML 2020 + Geng NeurIPS 2022 counterfactual TPP)
- *Budget curves*: Hill saturation (Dubé & Manchanda 2005) + frequency fatigue (Naik & Raman 2003)
- *Balancing loss*: HSIC (Gretton 2005) or adversarial-IPTW · BCAUSS · CaT (Melnychuk ICML 2022)
- *In-context amortization*: CInA (Arik & Pfister NeurIPS 2023)

</details>

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
| ⚫ TikTok            | Global   | 🟢 MVP  | Synthetic                        | LightGBM baseline    | v0.5 (real panels) |
| 🟣 Instagram Reels   | Global   | 🟢 MVP  | Synthetic                        | LightGBM baseline    | v0.5 (real panels) |
| 🔴 YouTube Shorts    | Global   | 🟢 MVP  | Synthetic                        | LightGBM baseline    | v0.5 (real panels) |
| 🔵 Douyin            | Greater China | 🟢 MVP | Synthetic                        | LightGBM baseline    | v0.5 (real panels) |
| ⚪ Twitter / X       | Global   | 📋 planned | —                             | —                    | v0.5 |
| 📺 Bilibili          | Greater China | 📋 planned | —                        | —                    | v1.0 |
| ✒️ LinkedIn          | Global   | 📋 planned | —                             | —                    | v1.0 |

**Want another platform?** Open an [Adapter Request](https://github.com/OranAi-Ltd/oransim/issues/new?template=adapter_request.yml) — we prioritize based on community demand.

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
<summary><b>Universal Embedding Bus (UEB)</b> — text-only today, multi-modal hooks for v0.5</summary>

Every data source (creative copy, KOL bio, user comment, fan-profile tabular record, platform event stream) flows through a shared `Embedder` ABC that produces a fixed-dim vector. Downstream modules (world_model / agent / causal) never see modality-specific code — the registry is modality-generic.

**Shipped today (v0.2)**:
- `RealTextEmbedder` — OpenAI-compatible `text-embedding-3-small` via the same gateway as `soul_llm` (one key for everything). Falls back to a deterministic hash embedder if the API is unavailable.
- `TabularEmbedder`, `CategoricalEmbedder`, `TimeSeriesEmbedder`, `GeoEmbedder`, `EventEmbedder` — non-learned baselines.

**Stubs for v0.5** (raise `NotImplementedError` pointing to ROADMAP.md#v05 if called):
- `ImageEmbedderStub` — planned backends: CLIP / Qwen-VL / SigLIP / ImageBind
- `VideoEmbedderStub` — planned backends: I-JEPA v2 / TimeSformer / VideoMAE v2 / Qwen-VL video
- `AudioEmbedderStub` — planned backends: Whisper-v3 encoder / CLAP / AudioMAE

Dropping a real implementation in is a ~50-line `Embedder` subclass with no downstream changes. See `backend/oransim/runtime/embedding_bus.py`.

</details>

<details>
<summary><b>LightGBM Quantile World Model</b> — fast baseline</summary>

Three quantile regressors (P35, P50, P65) per KPI. Sub-millisecond inference, zero GPU requirement. Feature engineering includes creative embeddings (OpenAI `text-embedding-3-small` — text only today, multi-modal via the UEB above lands in v0.5), platform priors, KOL features, temporal signals, and PCA-reduced behavioral features. Refs: Ke et al. 2017 (LightGBM), Koenker 2005 (Quantile Regression).

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
- ✅ **Multi-LLM-format adapters** — native Anthropic Messages, Gemini, Qwen DashScope shipped in v0.2; Bedrock Converse + native streaming roadmap item
- 🎯 **10k soul agents on Ray cluster**
- ✅ Instagram / YouTube Shorts / Douyin adapters MVP

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
- **Good first issues**: [see labels](https://github.com/OranAi-Ltd/oransim/issues?q=is%3Aissue+label%3A%22good+first+issue%22)
- **Platform adapter requests**: [file here](https://github.com/OranAi-Ltd/oransim/issues/new?template=adapter_request.yml)

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
  url          = {https://github.com/OranAi-Ltd/oransim},
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
- **Fakong Yin** — CTO & Core Architect · [cto@orannai.com](mailto:cto@orannai.com) · [@OranAi-Ltd](https://github.com/OranAi-Ltd)

**Open roles** — we're hiring researchers (Causal ML, RL, Agent-based Simulation) and engineers (Platform, Infra). Reach out at [cto@orannai.com](mailto:cto@orannai.com).

Contributors appear on [`CONTRIBUTORS.md`](CONTRIBUTORS.md) (auto-generated).

---

## ⭐ Star History

<a href="https://star-history.com/#OranAi-Ltd/oransim&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OranAi-Ltd/oransim&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OranAi-Ltd/oransim&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=OranAi-Ltd/oransim&type=Date" />
  </picture>
</a>

---

<div align="center">
Built with ☕ in Shenzhen by <a href="https://oran.cn">OranAI</a>. If Oransim helps your work, please ⭐ star the repo — it powers our open-source commitment.
</div>

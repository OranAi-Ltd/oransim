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

- 📈 Predicted impressions, clicks, conversions, ROI (with uncertainty bands)
- 🔄 Counterfactuals — "what if I'd used a different creative / more budget / another KOL?"
- 🗣️ Virtual-user feedback in natural language (10 LLM-powered personas)
- 📊 14-day diffusion curve
- 🧭 Recommended next actions, ranked

Built on structural causal models (Pearl 2009), agent-based simulation (1M IPF-calibrated virtual consumers + 10k LLM soul agents), Hawkes-process diffusion forecasting, and a LightGBM quantile world model.

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
| Causal reasoning | ❌ Correlation only | ❌ | ✅ Pearl 3-step counterfactuals |
| Agent-level simulation | ❌ | ❌ Aggregate only | ✅ 1M IPF + 10k LLM personas |
| Platform coverage | Single platform | Single platform | ✅ Multi-platform adapter framework |
| Budget saturation | ❌ Linear | ❌ Linear | ✅ Hill curve + frequency fatigue |
| Interpretability | Moderate | Low (SHAP at best) | ✅ SCM paths + agent reasoning traces |
| Cost | Licensing fees | API costs | ✅ Apache-2.0 + self-hosted |

Built by practitioners frustrated with both ends of the market — academic simulators that don't ship, and enterprise tools that don't explain.

---

## 🏗️ Architecture

<div align="center">
<img src="assets/architecture.svg" alt="Oransim architecture diagram" width="100%"/>
</div>

A typical prediction request flows: **Creative + Budget** → **PlatformAdapter** (pulls data via pluggable **DataProvider**) → **World Model** (LightGBM quantile) + **Agent Layer** (1M IPF + 10k LLM personas) → **Causal Engine** (SCM + Pearl counterfactuals) → **Diffusion** (Hawkes 14-day forecast) → **Prediction JSON** (14–19 schemas).

Two-axis extensibility:
- **Platform** axis — XHS today; TikTok / Instagram / YouTube Shorts / Douyin on roadmap
- **Data Provider** axis — pluggable per platform (Synthetic / CSV / JSON / OpenAPI / your own)

See [`docs/en/architecture.md`](docs/en/architecture.md) for the full design.

---

## 🌐 Platform Adapter Matrix

| Platform             | Region   | Status  | Data Provider                       | World Model          | Milestone |
|----------------------|----------|---------|-------------------------------------|----------------------|-----------|
| 🔴 XHS / RedNote     | Greater China | ✅ v1   | Synthetic / CSV / JSON / OpenAPI | LightGBM Quantile    | — |
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
4. **diffusion_curve** — 14-day daily impression/engagement forecast (Hawkes process)
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

<details>
<summary><b>World Model</b> — LightGBM Quantile Regression</summary>

Three quantile models (P35, P50, P65) for each KPI, providing ~30% confidence intervals. Feature engineering includes creative embeddings (OpenAI `text-embedding-3-small`), platform priors, KOL features, temporal signals, and PCA-reduced behavioral features.
</details>

<details>
<summary><b>Budget Model</b> — Hill saturation + frequency fatigue</summary>

Instead of naive linear budget scaling:

$$\text{effective\_impr\_ratio}(x) = \frac{(1+K) \cdot x}{K + x}$$

Michaelis-Menten / Hill saturation (Dubé & Manchanda 2005), combined with frequency fatigue (Naik & Raman 2003) on CTR/CVR:

$$\text{ctr\_decay}(r) = \max(0.5, 1.0 - 0.08 \cdot \max(0, \log_2 r))$$

This captures diminishing returns, an optimal budget point, and realistic campaign dynamics.
</details>

<details>
<summary><b>Diffusion</b> — Hawkes process (Neural Hawkes on roadmap)</summary>

Self-exciting point process modeling cascading engagement over 14 days. Captures virality, decay, and cross-platform spillover. Neural Hawkes (Mei & Eisner 2017) with Transformer intensity on the v0.5 roadmap.
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

**v0.5 (Q3 2026 – Q1 2027)**
- 🎯 **Neural Hawkes Process** — learned intensity functions for diffusion
- 🎯 **Transformer World Model** — attention over user × creative × platform × time
- 🎯 **Cross-platform transfer learning** — pretrain on XHS, fine-tune on TikTok
- 🎯 **Multi-LLM-format adapters** — native Anthropic Messages, Gemini, Bedrock, Qwen DashScope
- 🎯 **10k soul agents on Ray cluster**
- TikTok / Instagram / YouTube Shorts / Douyin adapters MVP

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
- **Fakong Yin** — CTO & Core Architect · [cto@orannai.com](mailto:cto@orannai.com) · [GitHub](https://github.com/[TBD:%20GITHUB_HANDLE])

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

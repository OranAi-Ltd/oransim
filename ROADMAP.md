# Oransim Roadmap

> **Updated:** 2026-04-18 · **Current version:** 0.1.0-alpha
>
> This roadmap is aspirational. Item inclusion does not guarantee delivery; priorities shift with user feedback, research breakthroughs, and commercial signal. Give a 👍 reaction on the [tracking issue](https://github.com/oranai/oransim/issues) to help us prioritize.

Oransim's roadmap is organized across three horizons and eight themes. Each horizon is cumulative — v1.0 ships everything from v0.2 and v0.5 plus new items.

- **v0.2 · Near** — 6~12 weeks (target Q3 2026)
- **v0.5 · Mid** — 3~6 months (target Q4 2026 – Q1 2027)
- **v1.0+ · Far** — 2027 and beyond (vision / conference-grade research)

Themes:
1. 🧠 **Models & Algorithms**
2. 🌐 **Platforms**
3. 🔌 **LLM Providers**
4. 📊 **Data & Benchmarks**
5. 🏗️ **Infrastructure**
6. 📦 **SDK & Integrations**
7. 🌱 **Ecosystem**
8. 📄 **Research & Publications**

---

## v0.2 · Near (Q3 2026)

### 🧠 Models & Algorithms
- Canonical schema v1.1 with richer field coverage (creative metadata, audience micro-segments, time-of-day priors)
- World model retains LightGBM quantile (P35/P50/P65) as the baseline
- Incremental PCA retraining pipeline with documented procedure
- Hill saturation + frequency-fatigue curves formalized as public API

### 🌐 Platforms
- TikTok adapter MVP (notes + KOL endpoints at minimum)
- Douyin adapter MVP (field schema close to TikTok)
- XHS adapter docs hardened (`docs/en/platforms/xhs.md`)

### 🔌 LLM Providers
- OpenAI-compatible client hardening (model fallback chain, retry with jitter)
- `.env.example` with all supported providers configured

### 📊 Data & Benchmarks
- OrancBench v0.1 — 50 synthetic scenarios with auto-scored outputs
- Model card v1.0 for shipped pkl
- Data card v1.0 for synthetic dataset

### 🏗️ Infrastructure
- Docker Compose one-shot launch
- MkDocs site deployed to GitHub Pages (`/docs/en/` + `/docs/zh/`)
- CI green: ruff + black + pytest with coverage gate

### 📦 SDK & Integrations
- Python SDK stub (`pip install oransim-sdk`) with basic CLI

### 🌱 Ecosystem
- Discord server launch
- Weekly "Office Hours" livestream (first 8 weeks)
- Blog launch on `https://oran.cn/oransim/blog`
- Hacker News / Product Hunt announcement

### 📄 Research & Publications
- Blog post: *Causal Data Augmentation for Sparse Marketing Prediction*
- Technical note: *Hill Saturation + Frequency Fatigue — Why Linear Budget Scaling Is Wrong*

---

## v0.5 · Mid (Q4 2026 – Q1 2027)

### 🧠 Models & Algorithms
- 🎯 **Neural Hawkes Process** (Mei & Eisner 2017) — replace parametric Hawkes with a neural intensity function (RNN or Transformer-parameterized), better long-tail diffusion capture, marked events
- 🎯 **Transformer World Model** — attention over (user × creative × platform × time) as a drop-in alternative to LightGBM quantile; evaluate against current baseline on OrancBench v0.5
- 🎯 **Cross-platform transfer learning** — pretrain world model on XHS data, fine-tune on TikTok with few-shot adapter layer; quantify transfer gain
- **Vision-Language Model (VLM) creative understanding** — CLIP / Qwen-VL / GPT-5.4-Vision to embed image + video creatives; compare against text-only baseline
- **RLHF for soul agents** — fine-tune the LLM persona layer using real marketer thumbs-up/down feedback

### 🌐 Platforms
- Instagram adapter MVP (Reels + Feed)
- YouTube Shorts adapter MVP
- Twitter / X adapter stub → MVP (tweets + replies + retweets)
- WeChat 视频号 adapter stub
- Twitch adapter stub

### 🔌 LLM Providers
- 🎯 **Multi-LLM-format adapters** — first-class support for:
  - Anthropic Messages API (native format)
  - Google Gemini generateContent API
  - AWS Bedrock (Claude / Llama / Mistral)
  - Azure OpenAI
  - xAI Grok API
  - DeepSeek native API
  - Qwen DashScope native API
  - Alternative: dependency on `litellm` as unified backend

### 📊 Data & Benchmarks
- OrancBench v0.5 — 500 scenarios with human rater scores
- Public leaderboard at `https://oran.cn/oransim/leaderboard`
- Synthetic generator v2 — comparative study of Copula vs GMM vs VAE
- Evaluation protocol spec (`docs/en/benchmarks/protocol.md`)

### 🏗️ Infrastructure
- Ray cluster support for 10k+ soul agents in parallel
- Kubernetes Helm chart
- GPU inference support (vLLM / TGI for local open-weight LLMs)
- Prometheus exporter + OpenTelemetry tracing
- Grafana dashboard templates

### 📦 SDK & Integrations
- TypeScript SDK (Node.js + browser)
- GraphQL API alongside REST
- Slack / Discord / Teams bots (scenario query)
- Notion / Linear integrations for campaign tracking
- VS Code extension — scenario YAML authoring with inline preview

### 🌱 Ecosystem
- **Plugin registry** — npm-style index for community-published platform adapters and data providers
- **Hosted demo** — `https://oran.cn/demo` (public try-before-clone)
- Monthly newsletter — 10 highlighted community-contributed adapters / models

### 📄 Research & Publications
- 📄 Submit to **ICML 2027**: *Causal Data Augmentation via Simulation Pretraining for Sparse Marketing Data*
- 📄 Submit to **KDD 2026** (short track): *Neural Hawkes for Cross-Platform Marketing Diffusion Forecasting*
- 📄 Submit to **WWW 2027** (demo track): *Oransim — An Open Platform for Causal Marketing Simulation*

---

## v1.0+ · Far (2027+)

### 🧠 Models & Algorithms
- 🎯 **Causal Foundation Model** — pretrain on 10M+ cross-industry, cross-platform campaigns; few-shot fine-tune per brand; the "BERT / GPT moment" for marketing causality
- 🎯 **Closed-loop AI media buying** — real-time prediction + automatic campaign parameter adjustment; production-grade RL with safety constraints
- **Differential privacy** — (ε, δ)-DP training for brand-proprietary data; publish DP-utility tradeoff curves
- **Federated learning** — cross-brand federated training without data egress from brand infrastructure
- **Streaming real-time prediction** — Kafka-native, sub-second refresh on live campaign data

### 🌐 Platforms
- 15+ platforms covered: LinkedIn, Pinterest, Snapchat, Reddit, Twitch, Bilibili, Zhihu, Kuaishou, Weibo, WeChat 公众号, etc.
- Cross-platform campaign orchestration — upload once, predict across all platforms simultaneously

### 🔌 LLM Providers
- **OranAI-tuned foundation model** — open-weight base + real marketing corpus fine-tune; available via Hugging Face

### 📊 Data & Benchmarks
- OrancBench v1.0 — 5,000 scenarios, multimodal (text + image + video)
- Public leaderboard with continuously updated SOTA
- Vertical benchmarks: beauty, fashion, 3C, F&B, luxury, auto

### 🏗️ Infrastructure
- Multi-tenant SaaS backend (row-level security, per-tenant encryption keys)
- On-premise enterprise deployment
- SOC 2 / ISO 27001 / GDPR compliance
- Multi-region (US / EU / APAC)

### 📦 SDK & Integrations
- Official plugins for Salesforce, HubSpot, Adobe Marketing Cloud
- Native integrations with Meta Ads, Google Ads, TikTok Ads, Pinterest Ads
- Creative asset generation via Flux / Midjourney / SD3 / Sora-class video models
- Zapier / Make / n8n low-code connectors

### 🌱 Ecosystem
- Annual **Oransim Conference**
- **OranAI Certified Partner** program (agencies, consultancies)
- Academic research grant program (extending MUSE collaboration)
- Open university curriculum for marketing causality

### 📄 Research & Publications
- 📄 **NeurIPS 2027**: *Foundation Models for Causal Reasoning in Agent-Based Marketing Simulation*
- 📄 **WWW 2027**: *Cross-Platform Transfer Learning for Creative Content Performance Prediction*
- 📄 **CHI 2027**: *Virtual User Personas: Bridging Quantitative and Qualitative Marketing Research*
- 📄 **KDD 2027**: *Closed-Loop Campaign Optimization via Causal Reinforcement Learning*

---

## Explicitly Out of Scope

To keep the project focused, these are **not** on the roadmap (may change with community demand):

- Generic recommendation systems (beyond marketing-adjacent use cases)
- Consumer-facing social media analytics dashboards (we're an infrastructure/framework, not an end-user BI tool)
- Real-time bidding (RTB) SSP/DSP — adjacent market, different compliance profile
- Generic time-series forecasting (use Prophet / Neural Prophet / GluonTS instead)

---

## Getting Involved

- Open an [Issue](https://github.com/oranai/oransim/issues) to propose roadmap changes
- Pick up a `good first issue` or `help wanted` item
- Join [Discussions](https://github.com/oranai/oransim/discussions) for design conversations
- Enterprise sponsors influencing the roadmap: contact `cto@orannai.com`

## Changelog of the Roadmap

- **2026-04-18** — Initial publication with three horizons

<div align="center">
<img src="assets/wordmark.svg" alt="Oransim" width="640"/>

### Predict your next campaign's ROI before spending a dollar.

<p>
  <a href="https://github.com/OranAi-Ltd/oransim/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/OranAi-Ltd/oransim?color=blue"></a>
  <a href="https://github.com/OranAi-Ltd/oransim/releases"><img alt="Release" src="https://img.shields.io/github/v/tag/OranAi-Ltd/oransim?label=release&color=blue"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="https://github.com/OranAi-Ltd/oransim/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/OranAi-Ltd/oransim/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/OranAi-Ltd/oransim/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/OranAi-Ltd/oransim?style=social"></a>
  <a href="https://oran.cn"><img alt="Website" src="https://img.shields.io/badge/website-oran.cn-FF6B35"></a>
</p>

<p>
  <strong>🇬🇧 English</strong> · <a href="README.zh-CN.md">🇨🇳 中文</a>
</p>

<p><em>Causal simulation for enterprise growth teams.<br/>Audit the engine, license the data.</em></p>
</div>

---

<p align="center">
<img src="assets/screenshots/hero.png" alt="Oransim hero · 60-second prediction with counterfactual reasoning over a agent-based society" width="100%"/>
</p>

**For enterprise CMOs** — predict your next campaign's ROI before spending: **4.3M+ indexed 小红书 notes · 2.1M+ creators (达人) across 15 verticals · 100,000+ surveyed consumer panel**, refreshed daily via licensed platform APIs. Counterfactual reasoning engine running on a **1M+ virtual consumer society** with LLM-backed soul personas reading your actual creatives. Transparent causal logic, open-sourced so you can audit it before licensing data access.

*The OSS repo you're reading is the same causal engine running on a 21k-note demo corpus — try it, audit the mechanism end-to-end, then explore the live Enterprise data panel at [datacenter.oran.cn](https://datacenter.oran.cn/) or contact `cto@orannai.com` for licensed access.*

---

## Who we are

**OranAI Ltd. (橙果视界（深圳）科技有限公司)** — a Shenzhen-based AI marketing company founded May 2024, closed a **multi-million-dollar angel+ round** led by [Cloud Angels Fund, with participation from Leaguer Venture Capital and Jinshajiang United Capital](https://36kr.com/p/3442645125141897). We co-operate the [Tencent Cloud × OranAI AIGC Design Lab](https://caijing.chinadaily.com.cn/a/202412/26/WS676d01b5a310b59111daaff3.html), run our in-house multimodal matrix (**Oran-VL 7B** / **Oran-XVL 72B**) behind four products — **PhotoG** (creative agent) · **DataG** (insight engine) · **VoyaAI** (strategy co-pilot) · **[DataCenter](https://datacenter.oran.cn/)** (real-time creator + note panel explorer) — and serve **70+ enterprise clients** across beauty, FMCG, consumer electronics, and DTC outbound — including [Timekettle and Hyundai Motor (Pharos IV Best Prize)](https://m.tech.china.com/articles/20260117/202601171798695.html), with 2025 revenue crossing **RMB 20M**.

**Oransim is the causal engine inside that stack.** When a CMO using OranAI asks *"what if we swapped KOL A for B on day 3 of this campaign?"* — the `do()`-operator, the per-arm counterfactual heads, and the 14-day Hawkes rollout that answer the question all live in this repository. We open-sourced it under Apache-2.0 so enterprise buyers can audit the reasoning end-to-end — **trust the engine, then license the data panel.**

<sub>As featured in: [PR Newswire](https://www.prnewswire.com/news-releases/oranai-raises-multi-million-dollar-angel-funding-to-lead-ai-content-marketing-through-its-ai-agent-photog-302548911.html) · [亿邦动力](https://www.ebrun.com/20250520/579947.shtml) · [新浪科技](https://finance.sina.com.cn/tech/roll/2024-11-26/doc-incxkhus4289659.shtml) · [腾讯新闻](https://news.qq.com/rain/a/20250714A07JHO00) · [DoNews](https://www.donews.com/news/detail/5/3670706.html)</sub>

---

## What it solves

Three campaign decisions that break traditional tools but collapse to one Oransim workflow:

### 1. Pre-launch 
> *"I have 4 creative videos × 3 KOL shortlists × 2 budget tiers — which combination has the highest ROI?"*

Traditional approach: A/B test for 2 weeks, burn ¥500k to learn. **Oransim**: 60-second simulation on ¥0, rank all 24 combinations with P35/P65 confidence bands, pick top 3 to actually test.

### 2. Mid-campaign 
> *"Day 3 CTR is below target. Can I swap out 2 KOLs and reallocate budget to 3 others — and how much ROI shifts?"*

Traditional approach: data team rebuilds a dashboard overnight. **Oransim**: `do(kol=swap_A_for_B, day=3)` counterfactual rollout in 30 seconds — shows the 14-day path diff with the intervention applied.

### 3. Post-mortem 
> *"This campaign underperformed. If we'd spent on 小红书 instead of 抖音, what would we have gotten?"*

Traditional approach: retrospective analysis, ambiguous conclusion. **Oransim**: load actuals + `do(platform_alloc={xhs: 1.0})`, get the counterfactual ROI curve over the same agent population — confident attribution of what would have happened.

All three run on the same engine. Below is how it's built and why you can trust it.

---

## Why current tools can't answer these three questions

Every marketing intelligence tool answers part of the question. None answer all three campaign decisions above on the same data:

| The 3 CMO questions | What existing tools do | What's missing |
|---|---|---|
| **Pre-launch ROI ranking** for 24 creative × KOL × budget combinations | Classical **Marketing Mix Modelers** fit the total revenue curve — one number per period | Can't tell you *which combination*: MMM is a total, not a per-arm counterfactual |
| **Mid-campaign intervention** — what if I swap a KOL on day 3? | **Customer Data Platforms** report what already happened — click funnel, cohort retention | Can't roll forward under a `do()` — DMPs are observational, not causal |
| **Post-mortem counterfactual** — what if we'd spent on 小红书 instead of 抖音? | **Black-box predictors** (AutoML, LLM "predict ROI") output a number with no derivation | Can't audit the reasoning — SHAP plots ≠ a causal graph |

Oransim sits in the gap: **per-arm counterfactuals** (pre-launch ranking) · **temporal `do()`-rollout** (mid-campaign swap) · **transparent causal graph** (post-mortem audit). One engine, three decisions.

---

## Why you can trust it — three signals, pick what your stakeholders care about

### 🔬 Mechanism · audit the engine yourself

The OSS repo you're reading is the **full causal engine**, not a marketing demo. Clone it, run it on your scenarios, trace any prediction back through the 64-node causal graph to which agent decision and which budget-curve calculation produced it. No "trust us, it's ML" — every prediction is decomposable.

```bash
git clone https://github.com/OranAi-Ltd/oransim.git && cd oransim
pip install -e '.[dev]' && python -m uvicorn oransim.api:app --port 8001 &
curl http://localhost:8001/api/graph/inspect   # the causal graph, in JSON
```

### 📊 Data · what Enterprise licenses get you beyond the OSS demo

The OSS ships a 21k-note reference corpus — enough to validate the mechanism, not enough to power production campaigns. Enterprise Edition runs on a continuously refreshed licensed panel, explorable live at **[datacenter.oran.cn](https://datacenter.oran.cn/)**:

| Asset | Scale | Source |
|---|---|---|
| 小红书 notes | **4,300,000+**, daily refresh | Licensed platform APIs + in-house crawlers |
| Creators (达人) | **2,100,000+** across 15 verticals — 美妆 · 护肤 · 穿搭 · 3C · 食饮 · 母婴 · 家居 · 汽车 · 汽车后市场 · 健身 · 理财 · 奢品 · 宠物 · 医美 · 旅行 · spanning KOL (top + mid tier), KOC (waist, 1k–50k fans), and long-tail creators | Platform signal + fan-profile metadata |
| Consumer panel | **100,000+** verified 小红书 users, surveyed monthly | Opt-in recruitment |

*Browse the live panel at **[datacenter.oran.cn](https://datacenter.oran.cn/)** · contact [`cto@orannai.com`](mailto:cto@orannai.com?subject=Oransim%20Enterprise%20Data%20Access) for licensed integration.*

### 📚 Research · 12-year tech lineage behind every layer

Oransim isn't a "vibes LLM" — every layer traces to 2010–2024 peer-reviewed literature:

<details>
<summary>Architecture + research lineage (click to expand)</summary>

- **Per-arm counterfactual heads** — TARNet (Shalit ICML 2017) · Dragonnet (Shi NeurIPS 2019)
- **Representation balancing** — HSIC (Gretton 2005) · adversarial-IPTW · BCAUSS · CaT (Melnychuk ICML 2022)
- **In-context amortization** — CInA (Arik & Pfister NeurIPS 2023)
- **Causal Neural Hawkes Process** — Mei & Eisner NeurIPS 2017 + Zuo ICML 2020 + Geng NeurIPS 2022 counterfactual TPP
- **Budget curves** — Hill saturation (Dubé & Manchanda 2005) + frequency fatigue (Naik & Raman 2003)
- **SCM** — Pearl 3-step (abduction → action → prediction), 64 nodes / 117 edges, discourse + cascade mediators (Sunstein 2017 · Bikhchandani 1992)
- **Agent population** — IPF / Deming-Stephan 1940 baseline

See `backend/oransim/{world_model,diffusion,causal}/` — every file has inline citations.
</details>

---

## 🚀 Quickstart (60 seconds)

```bash
# 1. Clone and install
git clone https://github.com/OranAi-Ltd/oransim.git
cd oransim
pip install -e '.[dev]'

# 2. Run backend (mock mode — no API key required)
LLM_MODE=mock python -m uvicorn oransim.api:app --port 8001 &

# 3. Run frontend
python -m http.server 8090 --directory frontend

# 4. Open http://localhost:8090 → click "⚡ 极速" → "🚀 Predict"
```

Mock mode returns deterministic stubs — good for CI / first look, but every LLM-driven feature (soul personas, group-chat, comment-section discourse, LLM calibration of KPIs) falls back to templates. **To unlock the real pipeline, switch to api mode:**

```bash
LLM_MODE=api \
LLM_API_KEY=sk-xxxxx \
LLM_MODEL=gpt-5.4 \
python -m uvicorn oransim.api:app --port 8001 &
```

Pick the native request format with `LLM_PROVIDER` — defaults to `openai` (also covers DeepSeek / vLLM / any OpenAI-compat gateway):

<details>
<summary>Per-provider recommended config (click)</summary>

| `LLM_PROVIDER` | `LLM_BASE_URL` | `LLM_MODEL` example | Key env |
|---|---|---|---|
| `openai` *(default)* | `https://api.openai.com/v1` | `gpt-5.4` · `gpt-4o-mini` | `OPENAI_API_KEY` or `LLM_API_KEY` |
| `openai` (DeepSeek) | `https://api.deepseek.com/v1` | `deepseek-chat` | `LLM_API_KEY` |
| `openai` (vLLM local) | `http://localhost:8000/v1` | any served model | `LLM_API_KEY=local` |
| `anthropic` | `https://api.anthropic.com` (default) | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` or `LLM_API_KEY` |
| `gemini` | Google default | `gemini-2.5-pro` · `gemini-2.5-flash` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `LLM_API_KEY` |
| `qwen` | `https://dashscope.aliyuncs.com/api/v1` (default) | `qwen-plus` · `qwen-turbo` | `DASHSCOPE_API_KEY` / `QWEN_API_KEY` / `LLM_API_KEY` |

Full reference in [`.env.example`](.env.example); extended retry / fallback-chain options in [`docs/en/quickstart.md`](docs/en/quickstart.md).

</details>

The frontend shows a yellow banner at the top whenever the backend is still in mock (or has no key set) — click ✕ to dismiss for the session.

> **Running right now · what's real vs aspirational**
> - ✅ **Working today** — full backend (`POST /api/predict` · `/api/adapters` · `/api/sandbox/*`, split across `api_routers/` since api.py 1730-line god-file refactor) · full frontend (hero · 9 tabs · cascade animation · modular `js/*.js`) · LightGBM quantile baseline pkl shipped · 5 platform adapters (XHS v1 legacy + TikTok agent-level w/ FYP RL + IG / YouTube Shorts / Douyin MVP) · learned amortized abduction (pure-numpy MLP q(U|O)) · multi-LLM providers (OpenAI-compat · Anthropic · Gemini · Qwen).
> - 🟡 **Code-complete, weights pending** — Causal Transformer world model + Causal Neural Hawkes diffusion — architecture + training loop + inference + thinning sampler all shipped; pretrained weights land with OrancBench v0.5.
> - 📋 **Roadmap-only** — Twitter / Bilibili / LinkedIn adapters · multi-modal embedders (image/video/audio stubs only today) · Ray cluster · hosted demo.

---

## 🎬 See It In Action

<table>
<tr>
<td width="50%" valign="top">

**Three-panel working UI** — left: creative + budget + sliders · center: KPI / Agent pool / AI group-chat tabs (+「更多 ›」dropdown for deep analysis) · right: per-persona LLM reactions.

<img src="assets/screenshots/main-three-col.png" alt="Three-panel prediction UI" width="100%"/>

</td>
<td width="50%" valign="top">

**Opinion-propagation through a agent-based society** — drop an ad copy, watch color-coded opinion waves (green=click / purple=high intent / red=skip / blue=curious) ripple outward from KOL seeds, cascading to their followers in real time.

<img src="assets/screenshots/society-100m.png" alt="Opinion propagation over the agent population" width="100%"/>

</td>
</tr>
</table>

---

## 🏗️ Architecture

<div align="center">
<img src="assets/architecture.svg" alt="Oransim architecture diagram" width="100%"/>
</div>

A typical prediction request flows: **Creative + Budget** → **PlatformAdapter** (pulls data via pluggable **DataProvider**) → **World Model** (factual + counterfactual predictions) + **Agent Layer** (POP_SIZE-scalable IPF + LLM personas) → **Causal Engine** (64-node causal graph + `do()` counterfactuals) → **Diffusion** (14-day intervention-aware rollout) → **Prediction JSON** (14–19 schemas).

**What runs where:**

| Surface | Default (ships today) | Research-grade (opt-in) |
|---|---|---|
| World model | LightGBM quantile baseline (`data/models/world_model_demo.pkl`) + hand-coded structural formula | `CausalTransformerWorldModel` (CaT / TARNet / Dragonnet / CInA) — train locally, or swap in via `POST /api/v2/world_model/predict?model=causal_transformer` |
| Diffusion | Parametric exponential-kernel Hawkes (Hawkes 1971) | `CausalNeuralHawkesProcess` (Mei & Eisner + Zuo et al. + Geng et al.) — same opt-in pattern: `POST /api/v2/diffusion/forecast?model=causal_neural_hawkes` |
| Agents | `StatisticalAgents` (vectorised, CPU) | `SoulAgentPool` LLM personas (enable via `use_llm=true` on `/api/predict`) |
| Sandbox | Budget-only slider uses a Hill-saturation + frequency-fatigue closed form (`mode: "fast_approx"` in the response) so the slider is responsive. Non-budget edits (creative / alloc / KOL) trigger a real model re-run (`mode: "counterfactual"` or `"full_rerun"`). | — |

*The registry is the extension point. Default `/api/predict` uses the baseline stack because it's what ships with weights today; `/api/v2/*` is how you A/B swap in the research stack once you've trained it. Both routes share the same SCM / agent / Hawkes plumbing.*

Two-axis extensibility:
- **Platform** axis — XHS (legacy, v1 live) + TikTok / Instagram / YouTube Shorts / Douyin (MVP on synthetic); Twitter / Bilibili / LinkedIn on roadmap
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

> *What "MVP" actually means here*: XHS is the canonical v1 adapter with real data-provider paths (CSV / JSON / OpenAPI). TikTok / IG / YouTube Shorts / Douyin ship as **config-differentiated wrappers** over the same `PlatformAdapter` interface (each has distinct CPM / CTR / CVR / duration priors — see `backend/oransim/platforms/{platform}/adapter.py`), all driven by the synthetic LightGBM baseline. They pass shape tests end-to-end but don't yet have platform-specific DataProviders hooked up; that's what "v0.5 (real panels)" means in the milestone column.

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

<details id="causal-graph">
<summary><b>Causal Graph</b> — 64 nodes, 117 edges</summary>

Hand-designed by domain experts covering the marketing funnel: impression → awareness → consideration → conversion → repeat purchase → brand memory, with mediators for group discourse (Sunstein 2017) and information cascades (Bikhchandani et al. 1992).

The graph includes long-term feedback loops (e.g. `repeat_purchase → brand_equity → ecpm_bid → next-cycle impression_dist`). This is intentional — it reflects real marketing physics, not a modeling artifact. Strict Pearl-style abduction on cycles is undefined; our `do()` evaluation uses the cyclic-SCM generalization of Bongers et al. 2021 ([Foundations of Structural Causal Models with Cycles and Latent Variables](https://arxiv.org/abs/1611.06221)), treating the 25-node feedback SCC as a fixed-point solve rather than a topological forward pass.

The 3-step evaluation in code:
1. **Abduction** — at the agent layer, re-use the sampled noise from baseline; at the graph layer, per-node residuals are frozen
2. **Action** — apply `do()` intervention (supported nodes listed in `/api/dag`'s `intervenable: true` set)
3. **Prediction** — topologically sort the acyclic condensation, solve each SCC by numerical iteration (2–3 passes empirically converge on the shipped graph)

A time-unrolled DAG projection IS available in the OSS release via `oransim.causal.scm.dag_dict_unrolled(n_steps=K)` — each original node becomes `N_t0, N_t1, ..., N_t{K-1}`; feedback edges cross time (`src_ti → dst_t{i+1}`), non-feedback edges replicate within each slice. At `n_steps=2` the shipped graph's 64 nodes + 117 edges (cyclic) unroll to 128 nodes + 220 edges (strict DAG, 14 feedback edges detected automatically via DFS back-edge analysis). Downstream modules that need strict acyclicity (CausalDAG-Transformer attention on a true DAG, textbook Pearl three-step abduction) can consume the unrolled view. The cyclic native graph + SCC condensation remains the default because it keeps the node count small and matches the shipped Transformer's 7-token input layout.

A full equilibrium-solver with fixed-point guarantees for the cyclic native graph is an Enterprise Edition upgrade; the OSS release offers the unrolled-DAG path as the acyclic alternative.
</details>

<details>
<summary><b>Agent Population</b> — POP_SIZE-scalable IPF-calibrated virtual consumers</summary>

Generated via Iterative Proportional Fitting (IPF / Deming-Stephan 1940) against real Chinese demographic distributions (age × gender × region × income × platform). Each agent carries:
- Demographics + psychographics
- Platform-specific engagement priors
- Niche/category affinity vectors
- Time-of-day activity curves
- Social graph embeddings
</details>

<details>
<summary><b>Soul Agents</b> — LLM personas for qualitative feedback</summary>

The top-K most salient agents for a scenario are upgraded to LLM-backed personas (`SOUL_POOL_N` configurable; default 100 for demo, scalable via Ray in the Enterprise Edition). Default model: `gpt-5.4`. Each persona:
- Generates a persona card from its demographic vector
- Evaluates the creative (reaction / emotional response / intent)
- Optionally participates in simulated group chats (Sunstein 2017 group polarization)
- Feeds second-wave mediators back into the causal graph

**Two modes, explicit trade-off**:

- **Template mode** (`use_llm=False`, default) — click decision is a Bernoulli draw against the statistical `click_prob` (+40% niche-match lift); the persona picks a consistent template ``reason`` / ``comment`` / ``feel``. Zero LLM cost, deterministic given seed, used for CATE / ROI numerical reproducibility.
- **LLM-decider mode** (`use_llm=True`, Park et al. 2023 Generative Agents style) — a real LLM gets the full persona card + creative + KOL context and returns a structured JSON (`will_click`, `reason`, `comment`, `feel`, `purchase_intent_7d`). **The LLM's ``will_click`` is the agent's decision** (not overridden by Bernoulli); the statistical `click_prob` is available as a prior in the prompt. Response tagged `source: "llm"`. Trade-off: adds non-determinism per persona; for strict reproducibility stay in template mode or pin `LLM_TEMPERATURE=0`.

Cost controlled via:
- In-flight request coalescing (leader/follower dedup pattern)
- Persona card caching
- Configurable `SOUL_POOL_N`
</details>

<details id="causal-transformer-world-model">
<summary><b>Causal Transformer World Model</b> — primary (research-grade)</summary>

A 6-layer × 256-dim causal Transformer that ingests heterogeneous campaign features and predicts three quantile levels (P35/P50/P65) for each funnel KPI. Architecture lifts ideas from the recent causal-Transformer literature:

- **Token-type factorization** (CaT, Melnychuk et al. ICML 2022) — inputs split into *Covariate* (platform, demographic, time), *Treatment* (creative embedding, budget, KOL), and *Outcome* (KPIs) tokens with distinct type embeddings
- **DAG-aware attention** (CausalDAG-Transformer) — attention mask derived from the 64-node causal graph restricts each token to attend to topological ancestors; per-head learnable gate on the bias. Because the shipped graph is cyclic (see §[Causal Graph](#causal-graph)), ancestry is defined on the graph's **SCC condensation**: within a feedback SCC all nodes are mutually ancestral, across SCCs the standard DAG ancestor relation applies (Bongers 2021 §3.2). Reference implementation shipped in `CausalTransformerWorldModel.set_dag_from_edges()` and toggleable via `dag_attention_bias=True`. The OSS release defaults to the LightGBM baseline path; **pretrained CT checkpoints with DAG attention enabled ship with the Enterprise Edition** (see §[OranAI Enterprise Edition](#enterprise)).
- **Per-arm counterfactual heads** (TARNet, Shalit et al. ICML 2017 / Dragonnet, Shi et al. NeurIPS 2019) — one quantile head per discrete treatment arm enables `predict_factual` vs `predict_counterfactual(do(T=t'))` with a single forward pass
- **Representation balancing** (BCAUSS + CaT) — HSIC (Gretton et al. 2005) or adversarial-IPTW loss decorrelates the learned representation from treatment assignment, reducing bias in counterfactual predictions
- **In-context amortization** (CInA, Arik & Pfister NeurIPS 2023, optional) — model can condition on a context set of prior campaigns for amortized zero-shot causal inference

Core component: `oransim.world_model.CausalTransformerWorldModel`. Training loop, counterfactual rollout, and save/load are shipped today; pretrained weights land with OrancBench v0.5.

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

Three quantile regressors (P35, P50, P65) per KPI. Sub-millisecond inference, zero GPU requirement. Refs: Ke et al. 2017 (LightGBM), Koenker 2005 (Quantile Regression).

**Shipped pkl** (`data/models/world_model_demo.pkl`, `feature_version: demo_v2`, ~3 MB) consumes **23 features**: 7 tabular (`platform_id`, `niche_idx`, `budget`, `budget_bucket`, `kol_tier_idx`, `kol_fan_count`, `kol_engagement_rate`) + 16 PCA-reduced text-embedding dimensions. The embedding input is a deterministic caption per scenario (`"春季 {niche} 新品种草 · {tier} KOL · {budget_bucket}"`) passed through `RealTextEmbedder` — same embedder the rest of the stack uses (UEB, soul-agent persona matching, `kol_content_match`, `search_elasticity`). When `OPENAI_API_KEY` is set, it hits `text-embedding-3-small`; without a key, it falls back to the deterministic SHA-256 hash embedder so training / inference is still reproducible offline. PCA components ship inside the pkl and are applied at inference time via `POST /api/v2/world_model/predict?model=lightgbm_quantile`. R² on the 200 held-out from 2,000 synthetic scenarios: impressions 0.88 · clicks 0.79 · conversions 0.71 · revenue 0.75.

The Causal Transformer path consumes the full-dim creative embedding natively (without PCA) once weights land with OrancBench v0.5; the demo LightGBM pkl is the CPU-only fallback until then.

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

Core component: `oransim.diffusion.CausalNeuralHawkesProcess`. Architecture, training loop (NLL with MC compensator), forecast sampler (Ogata thinning), and counterfactual rollout are shipped today; pretrained weights land with OrancBench v0.5.

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

Scenario sessions persist state so users can iterate: "change budget from 100k to 150k, how does ROI move?" Incremental recomputation avoids redoing the full agent simulation when only budget changes. The agent pool is cached; counterfactual evaluation uses union-semantics CATE over reached vs. unreached populations.
</details>

---

## 📈 Benchmarks

Phase 1 benchmarks are based on the shipped synthetic corpus (**2,000 scenarios + 100 event streams + 50 OrancBench tasks** — reproducible from the files under [`data/synthetic/`](data/synthetic/) and [`data/benchmarks/`](data/benchmarks/)). See [`data/models/data_card.md`](data/models/data_card.md) for the data-generating process. The R² numbers below were run on 10% held-out of those 2k scenarios; larger-corpus numbers land with OrancBench v0.5.

| Metric | R² (synthetic) | Baseline (linear) | Notes |
|--------|---------------|-------------------|-------|
| `second_wave_click`     | 0.30 | 0.18 | PRS quantile median |
| `first_wave_conversion` | 0.33 | 0.21 | PRS quantile median |
| `cascade_lift`          | 0.39 | 0.25 | Second-wave mediator |
| `roi_point_estimate`    | 0.33 | 0.19 | Single-shot regression |
| `retention_7d`          | 0.29 | 0.17 | Longitudinal |

> ⚠️ **Honest reproducibility framing** — this is a **closed-loop evaluation**: the same synthetic data generator (`backend/scripts/gen_synthetic_data.py`) produces both training and held-out splits, and we evaluate our own model on our own generative process. This measures **"does the model fit our generative assumptions"**, not external validity. For real marketing-decision accuracy you need either (a) an independent real-panel benchmark (Enterprise Edition uses proprietary real-world data) or (b) a public benchmark with out-of-distribution campaigns — the OrancBench v0.5 plan (see ROADMAP.md) is our attempt at the latter.

See [`docs/en/benchmarks/`](docs/en/benchmarks/) for the full protocol.

---

## 🗺️ Roadmap — Highlights

See [ROADMAP.md](ROADMAP.md) for the full 3-horizon × 8-theme plan. Teasers:

**v0.2 (Q3 2026) — shipping pretrained weights**
- 📦 Trained Causal Transformer + Causal Neural Hawkes checkpoints on an expanded synthetic corpus (targeting ~100k scenarios for OrancBench v0.5)
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

The OSS you just read is the **causal engine**. Both editions run on the same Apache-2.0 code — the differences below span **8 dimensions**: data, pretrained weights, algorithms, learning loop, governance, integrations, team product, runtime. Audit the engine in this repo, then license the production stack.

> **💼 Business contact** — [`cto@orannai.com`](mailto:cto@orannai.com?subject=Oransim%20Business%20Inquiry) · pricing · data licensing · pilot · on-prem deployment · typical reply < 24h · or browse the live panel at **[datacenter.oran.cn](https://datacenter.oran.cn/)** first.

### Capability matrix

#### 📊 Data · real-world panel

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **Data panel** | 21k demo 小红书 notes + 3k KOLs | **4.3M+ notes · 2.1M+ 达人 (KOL + KOC + long-tail) · 100k+ consumer panel**, daily refresh · live at [datacenter.oran.cn](https://datacenter.oran.cn/) `[licensed platform APIs · ClickHouse]` |
| **Vertical calibration** | Generic priors | **10+ verticals** each calibrated — beauty · 3C · auto · luxury · medical aesthetics · … `[per-vertical fan_profile pkl + CPM–conversion curve fits]` |
| **Competitor panel** | — | Competitor KOL rosters + historical CPM/CVR 实盘 data `[public disclosures + third-party licensed feeds]` |

#### 🧠 Models · pretrained weights

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **World-model checkpoints** | All 3 models ship with `pretrained_url: "coming_soon"` · falls back to LightGBM baseline | **Pretrained CausalTransformer + CausalNeuralHawkes** with DAG-attention enabled `[trained on 10M+ real impressions · DAG mask derived from the 64-node SCM]` |
| **LLM soul agents** | Text LLM via your API key | Full multimodal — reads your actual creatives (image + video + audio) `[proprietary multimodal backbone · details under NDA]` |
| **Client-specific fine-tuning** | Shared generic baseline | Fine-tuned on **your real campaign data** · monthly incremental updates |

#### 🧮 Algorithms · solvers & posteriors

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **Counterfactual posterior** | Sample-reuse + closed-form Bayesian shrink + pure-numpy MLP amortizer | **Normalizing-flow learned posterior** · proper Pearl Step-1 abduction on cyclic graphs `[sbi NPE / SNPE]` |
| **Cyclic SCM equilibrium** | Time-unrolled DAG (acyclic approximation) + linear-SCC Banach fixed-point (requires ρ < 1) | **Non-linear equilibrium solver** with contraction guarantees on arbitrary cyclic SCMs `[Bongers 2021 §5 + damped Picard + spectral-radius monitoring]` |
| **Synthetic population** | IPF marginal matching (1-way marginals → 8-dim joint · ignores pairwise) | **Bayesian-net / diffusion joint synthesizer** · preserves pairwise + higher-order structure `[bnlearn · TabDDPM · both return HTTP 501 in OSS]` |
| **KOL matching** | Heuristic cosine (creative embed × KOL interest vector) | **Learned cross-attention encoder** · creative tokens × KOL-persona tokens `[transformer cross-attention · trained on real CPM-conversion labels]` |
| **Tag / trend extraction** | jieba tokenizer on 21k synthetic notes (static) | **Real-panel index** · daily refresh from live platform feeds `[Kafka → ClickHouse]` |

#### 🔁 Learning loop

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **Incremental learning from actuals** | Static model · manual retrain | Post-campaign actuals auto-stream back into the training set |
| **Cross-campaign brand memory** | Per-request brand memory only | 12-month continuous brand-equity tracking · avoids re-targeting the same cohort |

#### 🧭 Governance

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **Audit trail** | Local logs | Tamper-evident signed audit chain per prediction (input + model version + data snapshot, fully replayable) |
| **Approval workflow** | — | Strategy → budget → go-live multi-stage approval |
| **Rollback / version control** | — | Model-version + data-version + campaign-version binding · one-click rollback |
| **Compliance** | — | SOC 2 / ISO 27001 path · GDPR · 中国《个人信息保护法》 |

#### 🔗 Integrations

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **Martech connectors** | — | 巨量引擎 / 磁力引擎 / 小红书千帆 / 腾讯广告 / Google Ads / Meta Ads · official APIs |
| **CRM / CDP bidirectional sync** | — | Salesforce · SAP CDP · Adobe AEP · customer-owned CDP |
| **SSO / RBAC** | — | SAML 2.0 · OIDC · role-based permissions |

#### 👥 Team product

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **Multi-tenant isolation** | Single-tenant, local | Strict tenant isolation · competitor data physically segregated |
| **Collaboration** | CLI | Planner / buyer / approver multi-role workflow · Lark / Slack webhooks |
| **Saved scenario library** | No persistence | Scenario catalog + decision-chain traceability |

#### ⚙️ Runtime

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **Agent runtime** | Single-process Python · 100k agents (`SOUL_POOL_N ≤ 1000` LLM personas) | **Distributed Ray actor pool** · 1M+ agents · 10k+ LLM personas in parallel `[Ray 2.x + vLLM batched inference]` |
| **Shared state** | Process-local singletons + multi-worker startup WARNING | **Redis-backed shared state** · sandbox / brand-memory / UEB consistent across workers `[Redis 7 + asyncio client]` |

#### 🎧 Managed service

| | Oransim OSS | OranAI Enterprise |
|---|---|---|
| **Deployment** | Local / your cloud | Hosted · on-prem · hybrid · 99.9% SLA · sub-second · 全球加速 |
| **Onboarding** | Self-serve docs | White-glove — custom adapter dev · integration · training |
| **Model updates** | Community cadence | Managed — zero-downtime refresh as platforms evolve |

### Typical pilot (2 weeks, ¥0 commitment)

1. **Day 1–3 · Scope call** — we pick 2–3 of your active campaigns as test scenarios
2. **Day 4–10 · Simulation** — you give us creative + KOL shortlist + historical KPIs → we run counterfactual simulation → present ranked recommendations
3. **Day 11–14 · In-market validation** — you execute one recommendation in market → we compare our pre-launch prediction vs actuals → calibration report

**Exit criteria**: our pre-launch P35/P65 bands contain the actual KPI **≥ 80% of the time**. If not, pilot ends, no charge. If yes, we talk pricing.

### Contact

All inbound → [`cto@orannai.com`](mailto:cto@orannai.com) · typical reply < 24h. Tag the subject so we route it right:

- **[Business]** — pricing · demo · data licensing · API integration · on-prem deployment
- **[Pilot]** — book the 2-week pilot described above
- **[Investor]** / **[Partner]** — investors / strategic partners
- **[Press]** — media inquiries

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
  author       = {{OranAI Ltd. and Oransim contributors}},
  title        = {Oransim: Causal Simulation for Enterprise Growth Teams},
  version      = {0.2.0-alpha},
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

Built by **[OranAI Ltd.](https://oran.cn)** (橙果视界（深圳）科技有限公司). See §[Who we are](#who-we-are) above for company context.

### Core Maintainer

**Fakong Yin (尹法空)** · CTO & Core Architect, OranAI Ltd. · [`cto@orannai.com`](mailto:cto@orannai.com) · [@OranAi-Ltd](https://github.com/OranAi-Ltd)

Sole author of this repository's causal engine — 64-node Pearl SCM, per-arm counterfactual world model, causal neural Hawkes diffusion layer, Universal Embedding Bus, 8-router FastAPI backend, 5 platform adapters (XHS · TikTok · Douyin · Instagram Reels · YouTube Shorts), the LightGBM quantile baseline pipeline, and the 9-tab production frontend. End-to-end range across marketing strategy · ad-tech product · causal ML / RL / agent-based simulation · backend + data infrastructure — rare for a single engineer.

Git log speaks for itself: `git log --author="Fakong Yin" --oneline | wc -l`.

**Open roles** — we're hiring researchers (Causal ML · RL · agent-based simulation) and engineers (platform · data · infra). Reach out at [`cto@orannai.com`](mailto:cto@orannai.com).

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

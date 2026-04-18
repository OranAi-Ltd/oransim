# Architecture

> **Status:** Coming soon (v0.2). This page will document the full system architecture in depth.

For v0.1.0-alpha, see the [README Architecture section](https://github.com/deterministically/oransim/blob/main/README.md#%EF%B8%8F-architecture) and the [architecture.svg](https://raw.githubusercontent.com/deterministically/oransim/main/assets/architecture.svg) diagram.

## Design Principles

1. **Two-axis extensibility** — Platform × DataProvider. Adding a new platform requires implementing `PlatformAdapter`; adding a new data source for an existing platform requires implementing `DataProvider`.
2. **Canonical schemas as contracts** — `CanonicalKOL`, `CanonicalNote`, `CanonicalFanProfile` are the internal vocabulary. Adapters consume canonical types; providers emit canonical types.
3. **Causal first** — every prediction is either a direct `do()` intervention or a counterfactual. Correlation-only shortcuts are rejected at API boundary.
4. **Cost-aware agent scaling** — soul agent LLM calls are dedup-coalesced by (persona, creative, platform) key. Scaling to 10k agents is an infrastructure concern (Ray), not an algorithmic one.
5. **Synthetic-data-first OSS** — all shipped models train on synthetic data. Real-data training is an internal/paid tier concern.

## Module Layout

```
backend/oransim/
├── api.py               # FastAPI entry
├── platforms/           # PlatformAdapter × DataProvider
│   ├── base.py
│   ├── xhs/ (v1 reference)
│   └── {tiktok,instagram,youtube_shorts,douyin}/ (roadmap)
├── data/schema/         # CanonicalKOL / CanonicalNote / CanonicalFanProfile
├── agents/              # soul / discourse / group_chat
├── causal/              # SCM + Pearl counterfactual + CATE
├── diffusion/           # Hawkes (Neural Hawkes on roadmap)
├── runtime/             # CCG DAG + event bus + embedder
└── sandbox/             # scenario sessions + incremental recompute
```

## Prediction Flow

1. `POST /api/predict` with creative + budget + platform allocation + KOL list
2. For each platform: `PlatformAdapter.simulate_impression(creative, budget, ...)` using the bound `DataProvider`
3. World model estimates funnel rates (P35/P50/P65 via LightGBM quantile)
4. Agent layer draws decisions for each of 1M IPF-calibrated agents; top-10k upgrade to LLM personas
5. Causal engine runs Pearl 3-step for each counterfactual branch
6. Hawkes process projects 14-day diffusion
7. Soul agent feedback + discourse mediator signals feed back into funnel weights
8. Final JSON: 14–19 schemas, returned to client

See the [ROADMAP](https://github.com/deterministically/oransim/blob/main/ROADMAP.md) for Neural Hawkes + Transformer world model details.

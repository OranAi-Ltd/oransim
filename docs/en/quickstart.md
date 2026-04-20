# Quickstart

> **Status:** Coming soon (v0.2). This page will cover full installation, environment configuration, and running your first prediction.

For v0.1.0-alpha, see the root [README.md](https://github.com/OranAi-Ltd/oransim/blob/main/README.md#-quickstart-60-seconds).

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODE` | `mock` | `mock` for offline deterministic responses; `api` to call a real LLM |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `LLM_API_KEY` | (unset) | API key for the above endpoint |
| `LLM_MODEL` | `gpt-5.4` | Model identifier |
| `LLM_STREAM` | `1` | Stream responses when possible |
| `LLM_CONCURRENCY` | `15` | Max parallel LLM requests |
| `SOUL_POOL_N` | `100` | Number of LLM persona agents |
| `PORT` | `8001` | Backend API port |

## Deployment: run a single worker

Oransim v0.2 stores the runtime state (population, agents, world model,
Embedding Bus indexes, brand-memory cache) in **process-local**
singletons. There is no cross-worker synchronization in the OSS build.

**Deploy with a single worker.** Setting `WEB_CONCURRENCY >= 2`,
`WORKERS >= 2`, or `UVICORN_WORKERS >= 2` triggers a loud startup
`WARNING` — every extra worker holds its own independent copy of the
~GB runtime state, and sandbox / brand-memory / UEB state will diverge
across requests. A shared-state Redis backend is on the Enterprise
Edition roadmap; until it lands, single-worker is the correct choice
for the OSS tier.

See [ROADMAP.md](https://github.com/OranAi-Ltd/oransim/blob/main/ROADMAP.md) for when the full backend lands.

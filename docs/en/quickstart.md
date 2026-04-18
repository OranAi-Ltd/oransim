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

See [ROADMAP.md](https://github.com/OranAi-Ltd/oransim/blob/main/ROADMAP.md) for when the full backend lands.

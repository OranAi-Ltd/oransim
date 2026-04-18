# Platforms

Oransim organises platform integrations along **two orthogonal axes**:

1. **`PlatformAdapter`** — platform-specific semantics (what a post looks
   like, what a conversion is, what a KOL is). Each platform ships an
   adapter under `oransim.platforms.<name>`.
2. **`DataProvider`** — how the platform's data is sourced (synthetic,
   CSV export, JSON dump, OpenAPI endpoint, or custom). Providers map
   vendor-specific formats into the canonical Pydantic schemas.

Adding a new platform requires only an adapter. Adding a new data source
for an existing platform requires only a provider. The two axes never
need to change together.

## Adapter matrix

| Platform | Region | Status | Data Provider | Milestone |
|---|---|---|---|---|
| 🔴 Xiaohongshu / XHS | Greater China | ✅ v1 | Synthetic / CSV / JSON / OpenAPI | — |
| ⚫ TikTok | Global | 🟢 MVP | Synthetic | v0.5 (real panels) |
| 🟣 Instagram Reels | Global | 🟡 stub | — | v0.5 (Q4 2026) |
| 🔴 YouTube Shorts | Global | 🟡 stub | — | v0.7 (Q1 2027) |
| 🔵 Douyin | Greater China | 🟢 MVP | Synthetic | v0.5 (real panels) |
| ⚪ Twitter / X | Global | 📋 planned | — | v0.5 |
| 📺 Bilibili | Greater China | 📋 planned | — | v1.0 |
| ✒️ LinkedIn | Global | 📋 planned | — | v1.0 |

Request a new adapter via the [adapter request
template](https://github.com/ORAN-cgsj/oransim/issues/new?template=adapter_request.yml).

## Further reading

- [Writing an adapter](writing-an-adapter.md)
- [Writing a DataProvider](writing-a-provider.md)

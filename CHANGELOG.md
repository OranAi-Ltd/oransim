# Changelog

All notable changes to Oransim are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Phase 2: code desensitization + audit log
- Phase 3: platform abstraction refactor + synthetic data generator + test suite

## [0.1.0-alpha] — 2026-04-18

### Added
- Initial public repository
- Flagship bilingual README (EN + 中文) with hero banner, platform adapter matrix, technical deep-dive, roadmap summary, enterprise edition section, contributing guide, citation, star history
- `ROADMAP.md` — 3-horizon × 8-theme ambitious roadmap (Neural Hawkes, Transformer world model, Causal Foundation Model, multi-LLM native formats, closed-loop AI media buying, differential privacy, federated learning, 15+ platform coverage)
- Apache-2.0 `LICENSE` + `NOTICE`
- `CITATION.cff` (CFF 1.2.0)
- `SECURITY.md` vulnerability disclosure policy
- `CONTRIBUTING.md` with Developer Certificate of Origin (DCO) sign-off
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1)
- GitHub issue templates: bug report, feature request, platform adapter request
- GitHub PR template, `FUNDING.yml`, `CODEOWNERS`
- Directory skeleton for Phase 3:
  - `backend/oransim/platforms/{base,xhs,tiktok,instagram,youtube_shorts,douyin}/`
  - `backend/oransim/data/schema/`, `agents/`, `causal/`, `diffusion/`, `runtime/`, `sandbox/`
  - `tests/`, `examples/`, `docker/`, `docs/{en,zh}/`
- SVG visual assets: logo, wordmark, social preview, architecture diagram
- Python package metadata (`pyproject.toml`) for future `pip install oransim`

### Notes
- This is a skeleton release — full backend lands in v0.2.
- Platform stubs (TikTok/Instagram/YouTube Shorts/Douyin) raise `NotImplementedError` when accessed; this is intentional and tracks the roadmap.
- Benchmarks in README are based on synthetic data from the internal (non-public) data generator.

[Unreleased]: https://github.com/oranai/oransim/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/oranai/oransim/releases/tag/v0.1.0-alpha

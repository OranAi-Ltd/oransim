# Changelog

All notable changes to Oransim are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Causal Transformer World Model** (`oransim.world_model.CausalTransformerWorldModel`) —
  research-grade causal Transformer with token-type factorization
  (covariate/treatment/outcome), DAG-aware attention bias, per-arm
  counterfactual heads, and HSIC/adversarial-IPTW representation-balancing
  loss. Integrates CaT (Melnychuk et al. ICML 2022), CausalDAG-Transformer,
  TARNet/Dragonnet, BCAUSS, CInA (Arik & Pfister NeurIPS 2023). Full
  architecture + training loop + counterfactual rollout shipped; pretrained
  weights arrive in v0.2.
- **LightGBM Quantile World Model** (`oransim.world_model.LightGBMQuantileWorldModel`) —
  fast baseline retained for production latency-sensitive deployments and
  OrancBench ablations.
- **Causal Neural Hawkes Process** (`oransim.diffusion.CausalNeuralHawkesProcess`) —
  Transformer-parameterized neural temporal point process with causal event
  typing (organic vs paid_boost) and intervention-aware intensity. Based on
  Mei & Eisner (NeurIPS 2017), Zuo et al. (ICML 2020), Geng et al. (NeurIPS
  2022 counterfactual TPP), Ogata (1981 thinning). Full forecast + counter-
  factual rollout + NLL training with Monte Carlo compensator shipped;
  pretrained weights arrive in v0.2.
- **Parametric Hawkes** (`oransim.diffusion.ParametricHawkes`) — classical
  exponential-kernel multivariate Hawkes baseline (Hawkes 1971).
- **Training scripts** (`backend/scripts/train_transformer_wm.py`,
  `backend/scripts/train_neural_hawkes.py`) — CLI entry points for
  training; gracefully fail with helpful messages until the synthetic
  data generator lands in v0.2.
- **Model registry** (`get_world_model(name)`, `get_diffusion_model(name)`) —
  select variants by string, lazy-import the underlying module.
- **Optional `[ml]` extras** in `pyproject.toml` — `pip install 'oransim[ml]'`
  brings in PyTorch + einops to unlock the research-grade models; omitting
  the extra keeps the baselines fully usable.
- README + Chinese mirror upgraded to center on the full causal stack
  (Causal Transformer + Causal Neural Hawkes + Pearl SCM + counterfactual
  heads); LightGBM repositioned as a fast baseline with sub-millisecond
  inference.
- Comprehensive model card (`data/models/model_card.md`) covering the
  four-model zoo with per-model architecture, references, intended use,
  and known limitations.

### Planned
- Phase 2: code desensitization + audit log
- Phase 3: synthetic data generator (unlocks weight training) + test
  suite + Docker + CI + MkDocs site

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

[Unreleased]: https://github.com/ORAN-cgsj/oransim/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/ORAN-cgsj/oransim/releases/tag/v0.1.0-alpha

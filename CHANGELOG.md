# Changelog

All notable changes to Oransim are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0-alpha] — 2026-04-18

The v0.2 release. Ships the platform-adapter axis (TikTok + Douyin MVPs),
the second population synthesizer (Bayesian network), the canonical
Pydantic schema contract, the public budget-response API, OpenAI-client
hardening (retry + backoff + fallback), a MkDocs Material documentation
site, and expanded model / data card v1.0 documentation.

### Added
- **Canonical schemas v1.1** (`oransim.data.schema`) — CanonicalKOL,
  CanonicalNote, CanonicalNoteMetrics, CanonicalFanProfile,
  CanonicalScenario Pydantic models. Every adapter and provider now
  implements the same contract.
- **Public budget-curve API** (`oransim.world_model.budget`) —
  `hill_saturation`, `frequency_fatigue`, `apply_budget_curves`, and
  `BudgetCurveConfig`. Previously-embedded formulas promoted to cited
  first-class functions (Dubé & Manchanda 2005; Naik & Raman 2003).
- **OpenAI-compat HTTP client** (`oransim.runtime.http_client.post_json`)
  — exponential backoff with full jitter on retryable statuses,
  short-circuit on non-retryable statuses, optional
  `LLM_MODEL_FALLBACK` chain rotated per retry.
- **.env.example** — new file documenting every env var with provider
  cheatsheet for OpenAI / DeepSeek / Qwen (DashScope) / Moonshot / xAI /
  Together AI / Fireworks AI / local vLLM.
- **BayesianNetworkSynthesizer** — first non-IPF population synthesizer,
  a hand-specified BN over 6 demographic variables. Respects
  conditional dependencies (e.g., mean income increases with education)
  that IPF cannot represent.
- **TikTok adapter MVP** (`oransim.platforms.tiktok`) —
  TikTokAdapter + TikTokAdapterConfig + TikTokSyntheticProvider with
  global priors (USD CPM 5.8, FYP algorithm discovery, sub-minute
  attention curve, young-skewed fan demographics). Replaces the day-one
  NotImplementedError stub.
- **Douyin adapter MVP** (`oransim.platforms.douyin`) —
  DouyinAdapter + DouyinAdapterConfig + DouyinSyntheticProvider with
  Greater-China priors (RMB CPM 35, livestream conversion boost 1.25×,
  broader age spread). Replaces the stub.
- **MkDocs Material documentation site** (`mkdocs.yml` + docs/*) —
  strict-mode build clean; docs/en/platforms/index.md landing page.
  CI workflow at `.github/workflows/docs.yml` runs `mkdocs build
  --strict` on every docs change.
- **Model card v1.0 / Data card v1.0** — full documentation for the
  five-model zoo (CausalTransformerWM, LightGBMQuantileWM,
  CausalNeuralHawkes, ParametricHawkes, IPFSynthesizer) with
  per-model architecture, references, intended use, limitations, and
  the shipped R² numbers on synthetic eval + OrancBench.

### Fixed
- Hill-saturation double-scaling bug in the newly-added TikTok + Douyin
  adapters — baseline impressions now computed at `reference_budget` so
  Hill acts as a budget-ratio multiplier, not a multiplier on top of the
  linear `budget/CPM` scaling. Verified by
  `test_tiktok_adapter_mvp` asserting doubled budget → < 2× impressions.

### Tests
- 34 pass in ~6 s (up from 27). New coverage: canonical schemas,
  budget-curve public API, http_client retry/fallback semantics,
  .env.example shipping, BayesNet synthesizer conditional dependency
  (mean income rising with education), TikTok + Douyin adapter Hill /
  livestream-boost paths.

## [0.1.2-alpha] — 2026-04-18

### Added
- **PopulationSynthesizer abstraction** (`oransim.data.synthesizers`) — IPF
  baseline wrapping `generate_population` plus stubs for
  `BayesianNetworkSynthesizer` (v0.2), `TabDDPMSynthesizer` (v0.5),
  `CausalDAGTabDDPMSynthesizer` (v1.0 research), and `CTGAN` (v0.5). Pick via
  `get_synthesizer(name)`. ROADMAP adds "Causal Population Synthesis" as a
  v1.0 research item combining tabular diffusion with DAG-guided scoring.
- **OrancBench v0.1** — 50 scenarios (20 easy + 20 medium + 10 hard) shipped
  at `data/benchmarks/orancbench_v0_1.jsonl`, with ground-truth outcomes
  generated from the same Hill-saturation + frequency-fatigue + Hawkes
  process used by the synthetic corpus. Loader + scorer in
  `oransim.benchmarks`; runner at `backend/scripts/run_orancbench.py`. The
  shipped LightGBM baseline scores impressions R² 0.98/0.60/0.41 on
  easy/medium/hard — a real evaluation gradient.
- **4 example Jupyter notebooks** at `examples/` — quickstart · counterfactual
  reasoning · custom platform adapter · soul-agent personas.
- **Docker Compose** — `docker/Dockerfile` (multi-stage, slim) +
  `docker/docker-compose.yml` for one-command launch of backend+frontend.
- **GitHub Actions CI** — `.github/workflows/ci.yml` runs pytest on Python
  3.10/3.11/3.12, ruff + black in lint mode, and a dedicated
  desensitization gate stage.

### Tests
- **27 smoke tests** (up from 21), all pass in 13.7 s without PyTorch.
  New coverage: synthesizer registry + deferred NotImplementedError,
  OrancBench scenario schema + loader/scorer round-trip on ground-truth
  predictions (R² > 0.99), CI workflow present + has required stages,
  Docker artifacts shipped, example notebooks are valid ipynb JSON.

## [0.1.1-alpha] — 2026-04-18

### Added
- **Plug-and-play demo artifacts** — 2.3 MB of deterministic synthetic data
  (200 KOLs, 500 notes, 2k scenarios, 100 event streams) + a pretrained
  2.7 MB LightGBM quantile world model (R² on synthetic eval: impressions
  0.886, clicks 0.778, conversions 0.727, revenue 0.687) shipped at
  `data/synthetic/` and `data/models/world_model_demo.pkl`. Community can
  clone → set `LLM_API_KEY` → run, no separate data-gen step required.
- **`backend/scripts/train_lightgbm_demo.py`** — the trainer that produced
  the shipped pkl; retrain on your own data via the documented CLI.
- **Frontend `frontend/index.html`** — desensitized port of the 2422-line
  internal demo UI. All vendor-specific references scrubbed.
- **Training-script JSONL loaders** —
  `_load_dataset(...)` in `train_transformer_wm.py` now reads the
  scenario JSONL, applies a deterministic hash-based stand-in for the
  1536-d creative embedding, expands 7 scalar features into the full
  tensor dict CausalTransformerNet expects, and yields batched dicts
  (factual + counterfactual targets + treatment_arm).
  `_load_streams(...)` in `train_neural_hawkes.py` reads the event-stream
  JSONL directly. End-to-end training now reachable on any machine with
  `pip install 'oransim[ml]'`.
- **RBF-kernel HSIC** — `CausalTransformerWorldModel` supports
  `balancing_kernel="rbf"` in addition to linear, via new config fields
  `balancing_kernel` (default "linear") and `balancing_rbf_sigma`.
- **Compensator estimator choice** — `CausalNeuralHawkesConfig.compensator`
  lets users pick between the default rectangle-rule approximation and a
  (future) "mc" Monte Carlo estimator via `n_mc_samples`.

### Fixed
- Bulletproof desensitization: `test_no_sensitive_terms_in_package` now
  runs a case-insensitive scan (`hui[-_]?tun`, `tu[-_]?zi`, `cg[-_]?api`,
  `灰豚`, plus internal absolute paths) across 14 file extensions in
  backend / frontend / docs / assets / .github / root markdown files.
  Hardens the gate against capitalization regressions like the earlier
  `Huitun` oversight.
- `data/fan_profile.py` had four uncaught `Huitun` (capitalised) comments
  left by the original migration; scrubbed.
- `ROADMAP.md` + README "Roadmap Highlights" were listing already-shipped
  Neural Hawkes + Transformer WM as future v0.5 targets. Reworded to mark
  them as shipped; new v0.2 item is "pretrained weight release."
- `gen_synthetic_data.py` docstring claimed `.parquet` output when actual
  file is `.jsonl`; `train_transformer_wm.py` default `--data` path fixed.
- `HANDOFF.md` removed — it leaked `/home/projects/sim/` absolute paths.
- `ParametricHawkes.forecast` gained a hard `max_events=2000` cap alongside
  the existing `max_iters=20000`, preventing hangs under aggressive
  self-excitation priors.
- Neural Hawkes compensator now uses the intensity at `lam[0, i-1]` (the
  state BEFORE observing event `i`) rather than `lam[0, i]` — fixes a
  subtle acausal leak in the training NLL.
- `_hsic_unbiased` renamed to `_hsic_biased` (the formula was always the
  biased estimator; docstring corrected).
- `counterfactual_forecast` in Neural Hawkes now clones the intensity
  tensor before in-place scaling to avoid leaf-variable errors in future
  training-time rollouts.
- Five stale `from ..world_model.model import ...` imports in
  `causal/counterfactual.py`, `diffusion/legacy_hawkes.py`,
  `agents/{agent_provider,cross_platform,statistical}.py` redirected to
  the new `platforms.xhs.world_model_legacy` path.
- LLM defaults in `agents/soul_llm.py` corrected from the ported
  `api.deepseek.com` / `deepseek-chat` to the README-documented
  `api.openai.com/v1` / `gpt-5.4`.
- README broken Team link `[TBD: GITHUB_HANDLE]` → `@deterministically`.
- `.gitignore` now excludes `*.pt` and `*.safetensors` checkpoints.
- `(1.0 + 1.0) * r / ...` magic constant in `gen_synthetic_data.py`
  replaced with a named `K_SAT` constant and explanatory docstring.

### Tests
- **21 smoke tests**, all pass in ~12 s without PyTorch installed.
  Cover package imports, registries, torch-deferral, parametric Hawkes
  baseline, LightGBM demo pkl loading + prediction, synthetic data
  generator determinism and regression (hang-guard), synthetic CLI e2e,
  FastAPI bootstrap + route inventory, SCM graph shape (64 nodes /
  117 edges), CATE union semantics, population determinism, creative
  generator, and the desensitization gate.

## [0.1.0-alpha] — 2026-04-18

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

[Unreleased]: https://github.com/deterministically/oransim/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/deterministically/oransim/releases/tag/v0.1.0-alpha

# OrancBench — Benchmark Suite

> **Status:** Coming soon (v0.2: OrancBench v0.1 with 50 synthetic scenarios).

## Purpose

**OrancBench** is a benchmark suite for evaluating causal digital twin frameworks on marketing prediction tasks. It measures:

1. **Prediction accuracy** — R² / MAE / calibration on held-out scenarios
2. **Counterfactual fidelity** — quality of "what if" estimates vs. ground truth (synthetic)
3. **Cross-platform transfer** — how well a model trained on XHS generalizes to TikTok/Instagram
4. **Agent reasoning quality** — human rater scores on soul agent feedback realism

## Roadmap

- **v0.1 (Q3 2026)** — 50 synthetic scenarios, auto-scored
- **v0.5 (Q4 2026)** — 500 scenarios with human rater scores + public leaderboard
- **v1.0 (2027)** — 5,000 multimodal scenarios (text + image + video) + vertical sub-benchmarks

See the full [ROADMAP](https://github.com/ORAN-cgsj/oransim/blob/main/ROADMAP.md) for details.

## Reproducibility Note

v0.1.0-alpha benchmark numbers in the root README are produced from **100k synthetic samples**. They are not directly comparable to real-world production traffic. See [`data/models/data_card.md`](https://github.com/ORAN-cgsj/oransim/blob/main/data/models/data_card.md) for data-generating process.

Enterprise-edition real-world benchmarks are published separately under NDA.

# Data Card — Oransim Synthetic Training Corpus

- **Dataset name**: Oransim Synthetic Marketing Campaigns
- **Version**: **v1.0** (2026-04-18)
- **License**: Apache-2.0 / CC0 (the data itself is public domain)
- **Companion model card**: [`model_card.md`](model_card.md)

## Summary

A fully synthetic corpus of simulated marketing campaigns produced by
`backend/scripts/gen_synthetic_data.py`. No real KOL, note, user, or brand
data enters this dataset. It is the exclusive training source for all
pretrained models shipped with Oransim OSS.

## Generation process

| Artefact | Process | Distributional basis |
|---|---|---|
| KOL demographics | IPF-calibrated sampling | Published Chinese creator-economy aggregates |
| KOL fan counts | Log-normal per niche | Industry-reported niche-level creator tier distributions |
| KOL engagement rates | Beta per niche | Industry-reported engagement benchmarks |
| Notes (text + metrics) | Template-filling + KOL linkage | Aggregate note-per-day + engagement-rate priors |
| Fan profiles | Static aggregate per niche | Public demographic yearbooks |
| Training scenarios | Budget × KOL × niche sampling + Hill + fatigue | `apply_budget_curves()` + stochastic noise |
| Event streams | Hawkes-like self-exciting PP | Calibrated to match 14-day engagement envelope |

**Crucial property**: no real instance-level data enters the pipeline.
All samples are draws from distributions matched to published aggregate
statistics, not from real observations.

## Shipped files (v0.1.2-alpha)

| File | Rows | Size | Purpose |
|---|---|---|---|
| `synthetic_kols.json` | 200 | 49 KB | KOL pool |
| `synthetic_notes.json` | 500 | 208 KB | Note pool (English + Chinese text) |
| `notes_v3.json` | 500 | 208 KB | Back-compat alias for `synthetic_notes.json` |
| `synthetic_fan_profiles.json` | 10 niches | 4 KB | Per-niche demographic distributions |
| `niche_priors_calibrated.json` | 10 niches | 4 KB | Back-compat alias |
| `scenarios_v0_1.jsonl` | 2 000 | 770 KB | Training scenarios for the world model |
| `event_streams_v0_1.jsonl` | 100 × 14-day streams | 1.1 MB | Training corpus for the Neural Hawkes |

Total shipped demo data: **2.3 MB**. The deterministic seed (42) ensures
byte-identical regeneration. Users can scale up by rerunning the generator
with their own `--n-*` arguments.

## Canonical schema

All records conform to the Pydantic schemas in
`backend/oransim/data/schema/canonical.py`:

- `CanonicalKOL` — 200 instances
- `CanonicalNote` — 500 instances
- `CanonicalFanProfile` — 10 instances (one per niche)
- `CanonicalScenario` — 2 000 instances
- Event streams use the flat list-of-(time, event_type) form documented in
  `oransim/diffusion/base.py`.

Schema version: `1.1`.

## Intended use

| Use case | Supported? |
|---|---|
| Training the shipped LightGBM baseline | ✅ |
| Training the Causal Transformer world model | ✅ |
| Training the Causal Neural Hawkes diffusion model | ✅ |
| OrancBench reproducibility | ✅ |
| Testing platform-adapter code | ✅ |
| Benchmarking alternative causal models | ✅ |
| Real-world deployment | ❌ — use a platform-specific `DataProvider` |
| Academic benchmarking vs real data | ❌ — use held-out real data |

## Known limitations

1. **No real-instance fidelity** — the corpus matches published aggregate
   statistics, not real individual campaigns. Metrics derived against this
   corpus do not transfer to production without re-calibration.
2. **Chinese-market bias** — XHS-focused priors. Global-market deployment
   requires platform-specific `DataProvider`s and recalibration.
3. **Text-only modality** — no image/video creative embeddings. Multimodal
   generation lands in v0.5.
4. **No adversarial examples** — fraud, bots, spam, and brand-safety edge
   cases are not explicitly modelled.
5. **Stationary distributions** — no temporal drift is simulated. Real
   creator economies shift month-over-month.

## Ethical considerations

- The corpus contains **no personal information** about real users, KOLs,
  or brands. Synthetic names are drawn from a randomised vocabulary of
  English adjective-noun pairs.
- OSS users training on this dataset can freely redistribute derived
  models under Apache-2.0.
- OranAI Enterprise Edition trains on proprietary real-world data under
  separate agreements; those models are not distributed with the OSS
  release.

## Reproducibility

```bash
python -m backend.scripts.gen_synthetic_data \
    --out data/synthetic --seed 42 \
    --n-kols 200 --n-notes 500 --n-scenarios 2000 --n-streams 100
```

Deterministic output. Identical seed → byte-identical files.

## Versioning

| Version | Changes |
|---|---|
| **v1.0** | First documented release. 2.3 MB demo bundle shipped with v0.1.2-alpha. |
| v0.5 | Planned: 500 k+ scenarios, comparative Copula vs GMM vs VAE generator study. |
| v1.0+ | Planned: 5 M multimodal scenarios. |

## Maintenance

Please report data-quality issues via
<https://github.com/deterministically/oransim/issues>. Contact `cto@orannai.com`
for commercial-data-licensing enquiries.

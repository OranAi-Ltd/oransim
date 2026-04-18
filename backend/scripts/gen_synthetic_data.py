"""Synthetic data generator for Oransim.

Produces the reproducible OSS training corpus:

- ``synthetic_kols.json``      — 1 000 fake KOLs across 10 niches
- ``synthetic_notes.json``     — 10 000 fake notes with text, metrics, KOL link
- ``synthetic_fan_profiles.json`` — per-niche fan demographic distributions
- ``scenarios_v0_1.parquet``   — 100 000 (covariate, treatment, outcome, arm)
  records for training the Causal Transformer World Model
  (``backend.scripts.train_transformer_wm``)
- ``event_streams_v0_1.jsonl`` — 50 000 event streams (per-campaign 14-day
  timelines) for training the Causal Neural Hawkes Process
  (``backend.scripts.train_neural_hawkes``)

All outputs are drawn from published aggregate statistics (Chinese
demographic yearbook distributions, typical XHS / RedNote engagement-rate
ranges) — **no real KOL, note, user, or brand data is used**. The OSS
model zoo trains exclusively on this synthetic output; the OranAI
Enterprise Edition retrains on proprietary real-world data.

Usage
-----

    python -m backend.scripts.gen_synthetic_data \\
        --out data/synthetic \\
        --n-kols 1000 --n-notes 10000 --n-scenarios 100000 --n-streams 50000 \\
        --seed 42

Deterministic — same seed produces byte-identical output. Skips any
output file that already exists unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------- config


NICHES = [
    ("beauty",     "美妆"),
    ("fashion",    "服装"),
    ("food",       "食饮"),
    ("electronics","3C"),
    ("travel",     "旅游"),
    ("parenting",  "母婴"),
    ("fitness",    "健身"),
    ("home",       "家居"),
    ("beverage",   "饮品"),
    ("pet",        "宠物"),
]

# Aggregate fan-count distribution (log-normal per niche, params rough-fit
# against publicly-reported XHS creator economy stats)
NICHE_FAN_COUNT_LN = {
    # (mu_log, sigma_log) for lognormal fan-count distribution
    "beauty":      (9.5, 1.6),
    "fashion":     (9.3, 1.6),
    "food":        (9.2, 1.5),
    "electronics": (9.0, 1.4),
    "travel":      (9.1, 1.5),
    "parenting":   (8.9, 1.4),
    "fitness":     (8.8, 1.4),
    "home":        (8.7, 1.4),
    "beverage":    (8.8, 1.4),
    "pet":         (8.6, 1.3),
}

# Aggregate engagement-rate distribution (beta per niche)
NICHE_ENGAGE_BETA = {
    "beauty":      (2.0, 60.0),   # mean ≈ 3.2%
    "fashion":     (2.1, 55.0),
    "food":        (2.5, 50.0),
    "electronics": (1.8, 65.0),
    "travel":      (2.3, 55.0),
    "parenting":   (2.4, 50.0),
    "fitness":     (2.2, 55.0),
    "home":        (1.9, 60.0),
    "beverage":    (2.3, 55.0),
    "pet":         (2.6, 48.0),
}

# Audience demographics (approx age-band distributions per niche, 7 bands)
# Ages: 14-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+
NICHE_AGE_DIST = {
    "beauty":      [0.03, 0.36, 0.39, 0.15, 0.05, 0.015, 0.005],
    "fashion":     [0.05, 0.32, 0.36, 0.17, 0.07, 0.025, 0.005],
    "food":        [0.04, 0.22, 0.34, 0.22, 0.12, 0.05,  0.01],
    "electronics": [0.03, 0.25, 0.38, 0.20, 0.09, 0.04,  0.01],
    "travel":      [0.02, 0.18, 0.32, 0.25, 0.15, 0.06,  0.02],
    "parenting":   [0.01, 0.08, 0.44, 0.35, 0.08, 0.03,  0.01],
    "fitness":     [0.02, 0.25, 0.38, 0.22, 0.10, 0.025, 0.005],
    "home":        [0.01, 0.10, 0.32, 0.30, 0.18, 0.07,  0.02],
    "beverage":    [0.05, 0.28, 0.35, 0.18, 0.10, 0.03,  0.01],
    "pet":         [0.02, 0.22, 0.40, 0.22, 0.10, 0.03,  0.01],
}

# Gender distribution per niche (female share)
NICHE_FEMALE_SHARE = {
    "beauty":      0.90,
    "fashion":     0.78,
    "food":        0.60,
    "electronics": 0.32,
    "travel":      0.56,
    "parenting":   0.88,
    "fitness":     0.54,
    "home":        0.72,
    "beverage":    0.62,
    "pet":         0.68,
}

EVENT_TYPES = [
    ("impression", "organic"),
    ("impression", "paid_boost"),
    ("like",       "organic"),
    ("comment",    "organic"),
    ("share",      "organic"),
    ("save",       "organic"),
    ("conversion", "organic"),
]


# --------------------------------------------------------------- kol / notes


@dataclass
class SyntheticKOL:
    kol_id: str
    nickname: str
    niche_en: str
    niche_zh: str
    fan_count: int
    avg_engagement_rate: float
    tier: str  # nano / micro / mid / macro / mega
    region: str
    joined_year: int


def _kol_tier(fan_count: int) -> str:
    if fan_count < 10_000:
        return "nano"
    if fan_count < 100_000:
        return "micro"
    if fan_count < 500_000:
        return "mid"
    if fan_count < 5_000_000:
        return "macro"
    return "mega"


FAKE_ADJ = [
    "Aurora", "Crimson", "Velvet", "Solar", "Neon", "Oak", "Linen",
    "Crystal", "Indigo", "Coral", "Ember", "Meadow", "Cloud", "Ivory",
    "Amber", "Storm", "Silver", "Jade", "Harbor", "Prairie",
]
FAKE_NOUN = [
    "Studio", "Lab", "Journal", "Files", "Diary", "Club", "Notebook",
    "Agenda", "Gallery", "Monthly", "Atelier", "Bureau", "Press",
    "Chronicle", "Works", "Record", "Society", "Muse", "Archive",
]


def _fake_nickname(rng: random.Random) -> str:
    return f"{rng.choice(FAKE_ADJ)}{rng.choice(FAKE_NOUN)}"


def generate_kols(rng: random.Random, n: int) -> list[SyntheticKOL]:
    """Draw ``n`` KOLs stratified across niches."""
    kols: list[SyntheticKOL] = []
    regions = ["CN-East", "CN-South", "CN-North", "CN-West", "CN-Central"]
    for i in range(n):
        niche_en, niche_zh = NICHES[i % len(NICHES)]
        mu, sigma = NICHE_FAN_COUNT_LN[niche_en]
        fan_count = max(500, int(rng.lognormvariate(mu, sigma)))
        a, b = NICHE_ENGAGE_BETA[niche_en]
        er = min(0.5, max(0.005, rng.betavariate(a, b)))
        kols.append(
            SyntheticKOL(
                kol_id=f"KOL_{i:06d}",
                nickname=_fake_nickname(rng),
                niche_en=niche_en,
                niche_zh=niche_zh,
                fan_count=fan_count,
                avg_engagement_rate=round(er, 4),
                tier=_kol_tier(fan_count),
                region=rng.choice(regions),
                joined_year=rng.randint(2018, 2024),
            )
        )
    rng.shuffle(kols)
    return kols


NOTE_TEMPLATES_EN = [
    "Just tried the new {adj} {noun} — thoughts?",
    "Three things I learned about {noun}s this month",
    "{adj} {noun} review: worth the hype?",
    "Before/after using the {adj} {noun}",
    "Honest take on {adj} {noun} after 30 days",
    "Why {adj} {noun} didn't work for me (and what did)",
    "The only {noun} guide you'll ever need",
    "Budget {adj} {noun} vs premium — real comparison",
]

NOTE_TEMPLATES_ZH = [
    "试了新的 {adj_zh} {noun_zh}，你们怎么看？",
    "这个月关于 {noun_zh} 我学到的三件事",
    "{adj_zh} {noun_zh} 实测：值得这个热度吗？",
    "用了 {adj_zh} {noun_zh} 的前后对比",
    "用了 30 天 {adj_zh} {noun_zh} 的真实感受",
    "为什么 {adj_zh} {noun_zh} 对我没效果（以及什么有效）",
    "你需要的唯一 {noun_zh} 指南",
    "平价 {adj_zh} {noun_zh} vs 高端——真实对比",
]

ADJ_ZH = ["清新", "醇厚", "质朴", "现代", "古典", "极简", "温暖", "柔和"]
NOUN_ZH = ["方案", "体验", "选择", "搭配", "组合", "系列", "款式"]


def generate_notes(
    rng: random.Random, kols: list[SyntheticKOL], n: int
) -> list[dict[str, Any]]:
    """Generate ``n`` synthetic notes tied to random KOLs."""
    notes: list[dict[str, Any]] = []
    for i in range(n):
        kol = rng.choice(kols)
        # Engagement drawn from KOL's niche distribution
        a, b = NICHE_ENGAGE_BETA[kol.niche_en]
        er = min(0.5, max(0.001, rng.betavariate(a, b)))
        # Impressions ~ follower_count × reach_factor × noise
        reach_factor = rng.lognormvariate(-1.3, 0.6)
        impressions = max(100, int(kol.fan_count * reach_factor))
        likes = int(impressions * er * 0.7)
        comments = int(impressions * er * 0.12)
        shares = int(impressions * er * 0.06)
        saves = int(impressions * er * 0.12)
        tpl_en = rng.choice(NOTE_TEMPLATES_EN)
        tpl_zh = rng.choice(NOTE_TEMPLATES_ZH)
        adj = rng.choice(FAKE_ADJ).lower()
        noun = rng.choice(FAKE_NOUN).lower()
        text_en = tpl_en.format(adj=adj, noun=noun)
        text_zh = tpl_zh.format(adj_zh=rng.choice(ADJ_ZH), noun_zh=rng.choice(NOUN_ZH))

        notes.append(
            {
                "note_id": f"NOTE_{i:07d}",
                "kol_id": kol.kol_id,
                "niche": kol.niche_en,
                "platform": "xhs",
                "text_en": text_en,
                "text_zh": text_zh,
                "publish_day": rng.randint(1, 120),  # days since campaign start baseline
                "metrics": {
                    "impressions": impressions,
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "saves": saves,
                    "engagement_rate": round(er, 4),
                },
            }
        )
    return notes


# ----------------------------------------------------------- fan profiles


def generate_fan_profiles() -> dict[str, dict[str, Any]]:
    """Per-niche aggregate demographic distributions — deterministic, no rng."""
    profiles: dict[str, dict[str, Any]] = {}
    for niche_en, _ in NICHES:
        profiles[niche_en] = {
            "age_bands": ["14-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            "age_dist": NICHE_AGE_DIST[niche_en],
            "female_share": NICHE_FEMALE_SHARE[niche_en],
            "male_share": round(1.0 - NICHE_FEMALE_SHARE[niche_en], 3),
            "source": "synthetic v0.1 (distribution-matched to published aggregates)",
        }
    return profiles


# --------------------------------------------------------------- scenarios


def _budget_effect(budget: float) -> float:
    """Hill saturation used for the outcome process (Dubé & Manchanda 2005)."""
    K = 50_000.0
    r = budget / K
    return (1.0 + 1.0) * r / (1.0 + r)  # K_sat = 1.0 → asymptote 2x


def _freq_fatigue(budget: float, kol_er: float) -> float:
    """Frequency-fatigue damping on CTR (Naik & Raman 2003)."""
    impressions_proxy = budget * 0.002
    if impressions_proxy <= 1.0:
        return 1.0
    return max(0.5, 1.0 - 0.08 * math.log2(impressions_proxy))


def generate_scenarios(
    rng: random.Random,
    kols: list[SyntheticKOL],
    n: int,
) -> Iterable[dict[str, Any]]:
    """Yield ``n`` scenario records for world-model training.

    Each record bundles:

    - covariates: platform_id, demo_feat (24-dim), time_feat (4-dim)
    - treatments: creative_embed (1536-dim), kol_feat (16-dim), budget
    - treatment_arm: discrete 0..3 bucket (budget × KOL-tier combo)
    - targets: {impressions, clicks, conversions, revenue}
    - cf_targets: same KPIs under a nearby alternative arm (for CaT-style
      counterfactual supervision)
    """
    for i in range(n):
        kol = rng.choice(kols)
        # Budget bucket: 0 (small) / 1 (medium) / 2 (large) / 3 (xlarge)
        budget_bucket = rng.choices([0, 1, 2, 3], weights=[0.4, 0.35, 0.18, 0.07])[0]
        budget_base = {0: 10_000, 1: 40_000, 2: 120_000, 3: 400_000}[budget_bucket]
        budget = budget_base * rng.uniform(0.85, 1.15)

        # KOL-tier encoding drives impression ceiling
        tier_boost = {"nano": 0.8, "micro": 1.0, "mid": 1.4, "macro": 2.1, "mega": 3.2}[kol.tier]

        # Treatment arm: combine budget bucket × KOL tier into 4 buckets
        # (simple discretisation for per-arm counterfactual head)
        arm = min(3, budget_bucket // 2 + (0 if kol.tier in ("nano", "micro") else 2))

        # Synthesise KPIs with Hill saturation + frequency fatigue
        base_ctr = 0.04 * (0.5 + kol.avg_engagement_rate)
        base_cvr = 0.015 * (0.6 + kol.avg_engagement_rate)
        budget_factor = _budget_effect(budget)
        fatigue = _freq_fatigue(budget, kol.avg_engagement_rate)

        noise_impr = rng.gauss(1.0, 0.15)
        impressions = max(
            100.0,
            budget * 0.002 * budget_factor * tier_boost * noise_impr,
        )
        clicks = max(0.0, impressions * base_ctr * fatigue * rng.gauss(1.0, 0.2))
        conversions = max(0.0, clicks * base_cvr * rng.gauss(1.0, 0.25))
        revenue = conversions * rng.uniform(80.0, 300.0)

        # Counterfactual arm = (arm + 1) mod 4 for supervision
        cf_arm = (arm + 1) % 4
        cf_budget = budget * {0: 0.6, 1: 1.0, 2: 1.4, 3: 1.8}[cf_arm]
        cf_budget_factor = _budget_effect(cf_budget)
        cf_impressions = max(100.0, cf_budget * 0.002 * cf_budget_factor * tier_boost * rng.gauss(1.0, 0.15))
        cf_clicks = max(0.0, cf_impressions * base_ctr * _freq_fatigue(cf_budget, kol.avg_engagement_rate) * rng.gauss(1.0, 0.2))
        cf_conversions = max(0.0, cf_clicks * base_cvr * rng.gauss(1.0, 0.25))
        cf_revenue = cf_conversions * rng.uniform(80.0, 300.0)

        # Features (low-dim for JSON size; real training loader expands these)
        yield {
            "scenario_id": f"SCN_{i:08d}",
            "platform_id": 0,  # xhs
            "niche": kol.niche_en,
            "budget": round(budget, 2),
            "budget_bucket": budget_bucket,
            "kol_tier": kol.tier,
            "kol_fan_count": kol.fan_count,
            "kol_engagement_rate": kol.avg_engagement_rate,
            "treatment_arm": arm,
            "cf_arm": cf_arm,
            "targets": {
                "impressions": round(impressions, 1),
                "clicks":      round(clicks, 1),
                "conversions": round(conversions, 1),
                "revenue":     round(revenue, 2),
            },
            "cf_targets": {
                "impressions": round(cf_impressions, 1),
                "clicks":      round(cf_clicks, 1),
                "conversions": round(cf_conversions, 1),
                "revenue":     round(cf_revenue, 2),
            },
        }


# --------------------------------------------------------------- streams


def generate_event_streams(
    rng: random.Random, kols: list[SyntheticKOL], n: int,
    *, max_events_per_stream: int = 500,
) -> Iterable[list[tuple[float, str]]]:
    """Yield ``n`` 14-day event streams for Neural Hawkes training.

    Uses a simple inhomogeneous Poisson + Hawkes-style self-excitation with a
    sliding 24h window. The window is maintained with an integer head pointer
    (O(1) amortised per event), so streams for macro/mega-tier KOLs do not
    blow up quadratically. A hard cap of ``max_events_per_stream`` events
    prevents runaway cascades.
    """
    from collections import deque

    horizon_min = 14 * 24 * 60
    window_min = 24 * 60
    weights = [3, 1, 2, 1, 1, 1, 0.3]

    # Calibrated base rates (events/min). Previous values were ~10x too high
    # for macro/mega and caused runaway self-excitation.
    base_rate = {
        "nano":  0.001,
        "micro": 0.004,
        "mid":   0.010,
        "macro": 0.025,
        "mega":  0.050,
    }
    excitement_per_recent = 0.0008  # was 0.003 — damp self-excitation

    for _ in range(n):
        kol = rng.choice(kols)
        base = base_rate[kol.tier]
        stream: list[tuple[float, str]] = []
        recent_times: "deque[float]" = deque()
        t = 0.0

        while t < horizon_min and len(stream) < max_events_per_stream:
            lam_base = base * math.exp(-t / (6 * 24 * 60))  # decay over 6 days
            # Prune sliding window: drop events older than t - window_min
            while recent_times and t - recent_times[0] > window_min:
                recent_times.popleft()
            excitement = excitement_per_recent * len(recent_times)
            lam = lam_base + excitement
            if lam <= 0:
                break
            dt = -math.log(max(1e-9, rng.random())) / lam
            t = t + dt
            if t >= horizon_min:
                break
            picks = rng.choices(range(len(EVENT_TYPES)), weights=weights)[0]
            name, treatment = EVENT_TYPES[picks]
            ev_name = f"paid_{name}" if treatment == "paid_boost" else name
            stream.append((round(t, 2), ev_name))
            recent_times.append(t)

        yield stream


# ------------------------------------------------------------------- main


def write_json(path: Path, data: Any, force: bool) -> bool:
    if path.exists() and not force:
        print(f"  [skip] {path} (use --force to overwrite)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [write] {path}  ({path.stat().st_size // 1024}KB)")
    return True


def write_jsonl(path: Path, rows: Iterable[Any], force: bool) -> bool:
    if path.exists() and not force:
        print(f"  [skip] {path} (use --force to overwrite)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    print(f"  [write] {path}  ({n} rows, {path.stat().st_size // 1024}KB)")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Oransim synthetic training data.")
    parser.add_argument("--out", default="data/synthetic", help="Output directory.")
    parser.add_argument("--n-kols", type=int, default=1000)
    parser.add_argument("--n-notes", type=int, default=10000)
    parser.add_argument("--n-scenarios", type=int, default=100000)
    parser.add_argument("--n-streams", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--what",
        choices=["all", "kols", "notes", "fan_profiles", "scenarios", "streams"],
        default="all",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    rng = random.Random(args.seed)

    print(f"[oransim] seed={args.seed}  out={out}")

    kols: list[SyntheticKOL] = []
    if args.what in ("all", "kols", "notes", "scenarios", "streams"):
        print(f"[oransim] generating {args.n_kols} KOLs …")
        kols = generate_kols(rng, args.n_kols)
        if args.what in ("all", "kols"):
            write_json(out / "synthetic_kols.json", [dataclasses.asdict(k) for k in kols], args.force)

    if args.what in ("all", "notes"):
        print(f"[oransim] generating {args.n_notes} notes …")
        notes = generate_notes(rng, kols, args.n_notes)
        write_json(out / "synthetic_notes.json", notes, args.force)
        # Also emit notes_v3.json for back-compat with legacy agents
        write_json(out / "notes_v3.json", notes, args.force)

    if args.what in ("all", "fan_profiles"):
        print("[oransim] generating fan profiles …")
        profiles = generate_fan_profiles()
        write_json(out / "synthetic_fan_profiles.json", profiles, args.force)
        # niche_priors_calibrated.json — back-compat with fan_profile.py
        write_json(out / "niche_priors_calibrated.json", profiles, args.force)

    if args.what in ("all", "scenarios"):
        print(f"[oransim] generating {args.n_scenarios} training scenarios …")
        # Stream to JSONL to avoid holding 100k dicts in RAM
        path = out / "scenarios_v0_1.jsonl"
        if path.exists() and not args.force:
            print(f"  [skip] {path} (use --force to overwrite)")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for row in generate_scenarios(rng, kols, args.n_scenarios):
                    f.write(json.dumps(row, ensure_ascii=False))
                    f.write("\n")
            print(f"  [write] {path}  ({path.stat().st_size // 1024}KB)")

    if args.what in ("all", "streams"):
        print(f"[oransim] generating {args.n_streams} event streams …")
        path = out / "event_streams_v0_1.jsonl"
        if path.exists() and not args.force:
            print(f"  [skip] {path} (use --force to overwrite)")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            n = 0
            with open(path, "w", encoding="utf-8") as f:
                for stream in generate_event_streams(rng, kols, args.n_streams):
                    f.write(json.dumps({"events": stream}, ensure_ascii=False))
                    f.write("\n")
                    n += 1
            print(f"  [write] {path}  ({n} streams, {path.stat().st_size // 1024}KB)")

    print("[oransim] done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

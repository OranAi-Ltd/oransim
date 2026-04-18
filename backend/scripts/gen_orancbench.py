"""Generate the canonical OrancBench v0.1 scenario file.

Produces 50 deterministic scenarios (20 easy / 20 medium / 10 hard) stratified
across niches, budget buckets, and KOL tiers, with ground-truth outcomes
synthesised from the same Hill-saturation + frequency-fatigue process used
by ``gen_synthetic_data.py``.

Usage:

    python -m backend.scripts.gen_orancbench \\
        --out data/benchmarks/orancbench_v0_1.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

NICHES = [
    "beauty",
    "fashion",
    "food",
    "electronics",
    "travel",
    "parenting",
    "fitness",
    "home",
    "beverage",
    "pet",
]
TIERS = ["nano", "micro", "mid", "macro", "mega"]
CAPTIONS_EN = {
    "beauty": ["Aurora morning serum", "Matte lip restocked", "30-day glow diary"],
    "fashion": ["Linen summer capsule", "Autumn coat edit", "Vintage denim revival"],
    "food": ["5-minute brunch bowl", "Spice-rack pantry hacks", "No-fry air recipes"],
    "electronics": ["Budget mechanical kb", "Noise-cancel for the gym", "Portable SSD teardown"],
    "travel": ["Hidden-gem Chengdu tea", "Off-peak Bali route", "Carry-on packing grid"],
    "parenting": ["Sleep-regression rescue", "Weekend craft bundle", "Screen-time that works"],
    "fitness": ["Pilates reformer at home", "Marathon base-build", "Recovery stretch flow"],
    "home": ["Studio declutter day", "Rental-friendly wall tricks", "Lamp ratio 101"],
    "beverage": ["Cold-brew without gear", "Tasting notes decoded", "Low-ABV summer pick"],
    "pet": ["Puppy crate week 1", "Enrichment on $10", "Senior-dog mobility plan"],
}
TIER_BOOST = {"nano": 0.8, "micro": 1.0, "mid": 1.4, "macro": 2.1, "mega": 3.2}
TIER_FAN_MEAN = {
    "nano": 5_000,
    "micro": 40_000,
    "mid": 200_000,
    "macro": 1_500_000,
    "mega": 8_000_000,
}


def _budget_effect(budget: float) -> float:
    K, K_SAT = 50_000.0, 1.0
    r = budget / K
    return (1.0 + K_SAT) * r / (K_SAT + r)


def _freq_fatigue(budget: float) -> float:
    impressions_proxy = budget * 0.002
    if impressions_proxy <= 1.0:
        return 1.0
    return max(0.5, 1.0 - 0.08 * math.log2(impressions_proxy))


def _synth_ground_truth(budget: float, tier: str, er: float, rng: random.Random) -> dict:
    base_ctr = 0.04 * (0.5 + er)
    base_cvr = 0.015 * (0.6 + er)
    factor = _budget_effect(budget)
    fatigue = _freq_fatigue(budget)
    impressions = max(100.0, budget * 0.002 * factor * TIER_BOOST[tier] * rng.gauss(1.0, 0.10))
    clicks = max(0.0, impressions * base_ctr * fatigue * rng.gauss(1.0, 0.15))
    conversions = max(0.0, clicks * base_cvr * rng.gauss(1.0, 0.20))
    revenue = conversions * rng.uniform(80.0, 300.0)
    return {
        "impressions": round(impressions, 1),
        "clicks": round(clicks, 1),
        "conversions": round(conversions, 1),
        "revenue": round(revenue, 2),
    }


def generate(seed: int, n_easy: int = 20, n_medium: int = 20, n_hard: int = 10) -> list[dict]:
    rng = random.Random(seed)
    scenarios: list[dict] = []
    idx = 1

    def _scenario(niche: str, tier: str, budget_bucket: int, difficulty: str) -> dict:
        nonlocal idx
        base_budget = {0: 10_000, 1: 40_000, 2: 120_000, 3: 400_000}[budget_bucket]
        budget = base_budget * rng.uniform(0.85, 1.15)
        fan_mean = TIER_FAN_MEAN[tier]
        fan_count = max(500, int(rng.lognormvariate(math.log(fan_mean), 0.4)))
        er = min(0.3, max(0.005, rng.betavariate(2.0, 55.0)))
        caption = rng.choice(CAPTIONS_EN[niche])
        gt = _synth_ground_truth(budget, tier, er, rng)
        scn = {
            "scenario_id": f"ORB-{idx:04d}",
            "niche": niche,
            "platform": "xhs",
            "budget": round(budget, 2),
            "budget_bucket": budget_bucket,
            "kol_tier": tier,
            "kol_fan_count": fan_count,
            "kol_engagement_rate": round(er, 4),
            "creative_caption": caption,
            "difficulty": difficulty,
            "ground_truth": gt,
        }
        idx += 1
        return scn

    easy_niches = ["beauty", "fashion", "food", "beverage"]
    easy_tiers = ["micro", "mid"]
    for i in range(n_easy):
        scenarios.append(
            _scenario(
                rng.choice(easy_niches),
                rng.choice(easy_tiers),
                budget_bucket=rng.choice([1, 2]),
                difficulty="easy",
            )
        )
    for i in range(n_medium):
        scenarios.append(
            _scenario(
                rng.choice(NICHES),
                rng.choice(TIERS),
                budget_bucket=rng.choice([0, 1, 2, 3]),
                difficulty="medium",
            )
        )
    hard_niches = ["pet", "parenting", "home", "fitness"]
    hard_tiers = ["nano", "mega"]
    for i in range(n_hard):
        scenarios.append(
            _scenario(
                rng.choice(hard_niches),
                rng.choice(hard_tiers),
                budget_bucket=rng.choice([0, 3]),
                difficulty="hard",
            )
        )
    return scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate OrancBench v0.1 scenarios.")
    parser.add_argument("--out", default="data/benchmarks/orancbench_v0_1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-easy", type=int, default=20)
    parser.add_argument("--n-medium", type=int, default=20)
    parser.add_argument("--n-hard", type=int, default=10)
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = generate(args.seed, args.n_easy, args.n_medium, args.n_hard)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")
    print(f"[orancbench] wrote {len(scenarios)} scenarios to {out_path}")
    print(f"[orancbench]   easy={args.n_easy}  medium={args.n_medium}  hard={args.n_hard}")
    print(f"[orancbench]   size = {out_path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

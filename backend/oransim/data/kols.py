"""Mock KOL (influencer) library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .creatives import _hash_emb


@dataclass
class KOL:
    id: str
    name: str
    platform: str  # "douyin" / "xhs" / ...
    fan_count: int
    interaction_rate: float
    price_cny: int
    niche: str  # "beauty" / "mom" / "tech" / "food" / ...
    emb: np.ndarray  # (64,) — audience interest fingerprint


# Niche list + bias captions are driven by data/niches.json via the loader
# in oransim.config.niches. Extending the registry there is the only edit
# needed to support new niches throughout the codebase.
from oransim.config import niches as _niche_registry

NICHES = _niche_registry.niche_keys()
NICHE_BIAS_CAPTIONS = _niche_registry.bias_captions()


def generate_kol_library(n_per_platform: int = 50, seed: int = 7) -> list[KOL]:
    rng = np.random.default_rng(seed)
    kols = []
    for platform in ["douyin", "xhs"]:
        for i in range(n_per_platform):
            niche = rng.choice(NICHES)
            # fan count log-uniform from 10k to 10M
            fans = int(10 ** rng.uniform(4, 7))
            inter = float(np.clip(rng.normal(0.04, 0.02), 0.005, 0.15))
            # price ≈ fans * inter * market_rate
            price = int(fans * inter * rng.uniform(2, 8))
            emb = _hash_emb(NICHE_BIAS_CAPTIONS[niche], seed_offset=i)
            kols.append(
                KOL(
                    id=f"{platform}_{niche}_{i:03d}",
                    name=f"{niche}达人#{i:03d}",
                    platform=platform,
                    fan_count=fans,
                    interaction_rate=inter,
                    price_cny=price,
                    niche=niche,
                    emb=emb,
                )
            )
    return kols


def pick_kol_by_spec(kols: list[KOL], platform: str, niche: str = None, budget: int = None) -> KOL:
    cand = [k for k in kols if k.platform == platform]
    if niche:
        cand = [k for k in cand if k.niche == niche] or cand
    if budget:
        cand = [k for k in cand if k.price_cny <= budget] or cand
    if not cand:
        return kols[0]
    return max(cand, key=lambda k: k.interaction_rate * np.log(k.fan_count))

"""Mock KOL (influencer) library."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from .creatives import _hash_emb


@dataclass
class KOL:
    id: str
    name: str
    platform: str           # "douyin" / "xhs" / ...
    fan_count: int
    interaction_rate: float
    price_cny: int
    niche: str              # "beauty" / "mom" / "tech" / "food" / ...
    emb: np.ndarray         # (64,) — audience interest fingerprint


NICHES = ["beauty", "mom", "tech", "food", "fashion", "fitness", "finance", "travel"]
NICHE_BIAS_CAPTIONS = {
    "beauty": "美妆 口红 粉底 测评",
    "mom": "母婴 育儿 妈妈 辅食",
    "tech": "数码 机械 评测 极客",
    "food": "美食 探店 做饭 家常",
    "fashion": "穿搭 vintage 设计师 小众",
    "fitness": "健身 减脂 跑步 瑜伽",
    "finance": "理财 基金 省钱 性价比",
    "travel": "旅行 vlog 攻略 风景",
}


def generate_kol_library(n_per_platform: int = 50, seed: int = 7) -> List[KOL]:
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
            kols.append(KOL(
                id=f"{platform}_{niche}_{i:03d}",
                name=f"{niche}达人#{i:03d}",
                platform=platform,
                fan_count=fans,
                interaction_rate=inter,
                price_cny=price,
                niche=niche,
                emb=emb,
            ))
    return kols


def pick_kol_by_spec(kols: List[KOL], platform: str, niche: str = None, budget: int = None) -> KOL:
    cand = [k for k in kols if k.platform == platform]
    if niche: cand = [k for k in cand if k.niche == niche] or cand
    if budget: cand = [k for k in cand if k.price_cny <= budget] or cand
    if not cand:
        return kols[0]
    return max(cand, key=lambda k: k.interaction_rate * np.log(k.fan_count))

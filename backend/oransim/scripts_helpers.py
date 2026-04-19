"""Shared helpers used by both training scripts and the API.

Lives inside the ``oransim`` package so the API can import it as
``from oransim.scripts_helpers import ...`` without depending on the
``backend/scripts/`` path layout.
"""

from __future__ import annotations

_BUDGET_BUCKETS_ZH = ["小预算", "中档预算", "大预算", "超大预算"]
_NICHE_ZH = {
    "beauty": "美妆",
    "fashion": "穿搭",
    "food": "美食",
    "electronics": "数码",
    "travel": "旅行",
    "parenting": "母婴",
    "fitness": "健身",
    "home": "家居",
    "beverage": "饮品",
    "pet": "宠物",
}
_TIER_ZH = {
    "nano": "尾部",
    "micro": "尾部",
    "mid": "腰部",
    "macro": "腰部头部",
    "mega": "头部",
}


def caption_for_demo_pkl(features: dict, config: dict | None = None) -> str:
    """Deterministic caption used to embed a scenario for the demo LightGBM
    pkl. Must stay identical between training (see
    ``backend/scripts/train_lightgbm_demo.py::_caption_for``) and inference,
    since the pkl's PCA projection was fit on this exact text.
    """
    niche = features.get("niche", "beauty")
    tier = features.get("kol_tier", "micro")
    bucket_idx = int(features.get("budget_bucket", 0))
    buckets = (config or {}).get("budget_buckets_zh", _BUDGET_BUCKETS_ZH)
    bucket = buckets[min(bucket_idx, len(buckets) - 1)]
    zh_niche = _NICHE_ZH.get(niche, niche)
    zh_tier = _TIER_ZH.get(tier, tier)
    return f"春季 {zh_niche} 新品种草 · {zh_tier} KOL · {bucket}"

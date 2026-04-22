"""T2-A2 kol_content_match — creative × KOL compatibility scoring.

Given a creative brief (own brand + category + caption + target niches),
score every KOL in the synthetic pool by:

1. Niche match:    +0.45 if KOL's niche ∈ target_niches
2. Tier fit:       +0.25 for mid-tier, +0.15 for others (mid-tier has highest
                   cost-efficiency in our heuristic model)
3. Text overlap:   +0.30 jaccard of 2-char n-grams between caption and the
                   KOL's nickname — proxy for content-semantic alignment.

If caller leaves ``target_niches`` empty, we fall back to
``detect_niche_from_caption`` to sniff one from caption keywords —
avoids the "food post matched to beauty KOL" failure mode when the UI
forgets to pass the niche field.

Returns the top_k KOLs with explanations on the top_n.

Enterprise Edition: swap the scoring function with a real cross-attention
encoder trained on historical creative/KOL performance data.
"""

from __future__ import annotations

import time
import uuid

from .kol_optimizer import _EN_TO_ZH, _classify_tier, _load_pool


# caption → Chinese niche name · simple keyword match.
# Fallback for when target_niches is empty — prevents unrelated niches
# (e.g. beauty) from winning the ranking on a food caption via tier
# and text-overlap alone.
#
# Niche labels MUST match the synthetic pool's ``niche_zh`` vocabulary
# (see data/synthetic_kols.json): 健身 / 3C / 宠物 / 食饮 / 家居 / 服装
# / 饮品 / 旅游 / 美妆 / 母婴.  Drinks have their own niche separate
# from 食饮 — order matters since first hit wins.
_CAPTION_NICHE_KW = [
    ("饮品", ["奶茶", "咖啡", "拿铁", "美式", "手冲", "茶饮", "气泡水",
             "汽水", "饮料", "下午茶", "果汁", "冰茶"]),
    ("美妆", ["美妆", "口红", "粉底", "眼影", "腮红", "遮瑕", "底妆", "素颜",
             "妆前", "化妆", "彩妆", "香水", "护肤", "面膜", "精华", "防晒",
             "水乳", "面霜", "洗面奶", "护肤品"]),
    ("食饮", ["美食", "探店", "做饭", "家常", "餐厅", "零食", "甜品", "小吃",
             "便当", "拉面", "火锅", "烧烤", "轻食", "外卖", "寿司", "烘焙"]),
    ("服装", ["穿搭", "搭配", "服装", "时尚", "通勤", "约会", "OOTD",
             "潮流", "外套", "裙子", "裤子", "上衣", "西装"]),
    ("3C",   ["数码", "手机", "耳机", "笔记本", "平板", "电脑", "相机",
             "键盘", "游戏本", "科技", "3C"]),
    ("母婴", ["母婴", "育儿", "妈妈", "辅食", "宝宝", "儿童", "婴儿",
             "童装", "奶粉", "纸尿裤", "孕期"]),
    ("健身", ["健身", "减脂", "跑步", "瑜伽", "运动", "健身房", "增肌",
             "塑形", "普拉提"]),
    ("旅游", ["旅行", "旅游", "vlog", "攻略", "酒店", "机票", "景点",
             "风景", "出行", "自驾", "露营", "度假"]),
    ("宠物", ["宠物", "猫", "狗", "宠粮", "猫粮", "狗粮", "铲屎", "毛孩",
             "喵", "汪", "猫砂"]),
    ("家居", ["家居", "家具", "沙发", "装修", "床垫", "收纳", "家电",
             "清洁", "家纺"]),
]


def detect_niche_from_caption(caption: str) -> str | None:
    """Sniff the most likely niche from a caption. First hit wins; None if no
    keyword matches."""
    if not caption:
        return None
    for niche, kws in _CAPTION_NICHE_KW:
        if any(k in caption for k in kws):
            return niche
    return None


def _ngram_set(s: str, n: int = 2) -> set[str]:
    s = (s or "").strip()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _normalize_niche(n: str) -> str:
    return _EN_TO_ZH.get(n, n)


def match_kol_content(
    own_brand: str = "本品牌",
    category: str = "通用",
    target_niches: list[str] | None = None,
    caption: str = "",
    top_k: int = 10,
    explain_top_n: int = 3,
) -> dict:
    pool = _load_pool()
    if not pool:
        return {"_error": "synthetic KOL pool missing", "rows": []}

    # Fallback: when caller did not pass target_niches, sniff one from the
    # caption. Without this, every KOL gets niche_score=0 and the ranking
    # degenerates to tier+text_overlap — which lets off-niche KOLs win.
    effective_niches = list(target_niches or [])
    if not effective_niches:
        sniffed = detect_niche_from_caption(caption)
        if sniffed:
            effective_niches = [sniffed]

    target_zh = {_normalize_niche(n) for n in effective_niches}
    cap_grams = _ngram_set(caption, 2) | _ngram_set(caption, 3)

    scored: list[dict] = []
    for k in pool:
        fans = int(k.get("fan_count", 0) or 0)
        if fans < 500:
            continue
        niche_zh = k.get("niche_zh") or _normalize_niche(k.get("niche_en", ""))
        tier = _classify_tier(fans)

        niche_score = 0.45 if niche_zh in target_zh else 0.0
        if tier == "腰部":
            tier_score = 0.25
        elif tier == "尾部":
            tier_score = 0.22
        elif tier == "KOC":
            tier_score = 0.18
        else:
            tier_score = 0.15  # 头部
        name_grams = _ngram_set(k.get("nickname", ""), 2)
        text_score = 0.30 * _jaccard(cap_grams, name_grams)

        total = niche_score + tier_score + text_score
        scored.append(
            {
                "kol_id": k.get("kol_id"),
                "name": k.get("nickname"),
                "niche": niche_zh,
                "tier": tier,
                "fans": fans,
                "match_score": round(total, 3),
                "niche_match": round(niche_score, 3),
                "tier_fit": round(tier_score, 3),
                "text_overlap": round(text_score, 3),
            }
        )

    scored.sort(key=lambda r: -r["match_score"])
    top = scored[:top_k]

    brief = {
        "own_brand": own_brand,
        "category": category,
        "target_niches": effective_niches,
        "target_niches_source": (
            "caller" if (target_niches or []) else
            ("caption_sniff" if effective_niches else "none")
        ),
        "caption_preview": (caption or "")[:80],
    }
    for i, row in enumerate(top[:explain_top_n]):
        why: list[str] = []
        if row["niche_match"] > 0:
            why.append(f"垂类匹配（{row['niche']}）")
        why.append(f"{row['tier']} 层级，粉丝 {row['fans']:,}")
        if row["text_overlap"] > 0.05:
            why.append(f"文案 n-gram 重合 {row['text_overlap']:.0%}")
        row["explanation"] = "；".join(why)

    return {
        "run_id": f"kcm_{uuid.uuid4().hex[:8]}",
        "brief": brief,
        "agent_model": "heuristic_match_v1",
        "total_cost_cny": 0,
        "candidate_pool_size": len(scored),
        "top_k": len(top),
        "rows": top,
        "data_source": "synthetic_kols (200 KOLs)",
        "note": "Heuristic on synthetic pool — Enterprise Edition uses a learned encoder.",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

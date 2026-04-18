"""T2-A2 kol_content_match — creative × KOL compatibility scoring.

Given a creative brief (own brand + category + caption + target niches),
score every KOL in the synthetic pool by:

1. Niche match:    +0.45 if KOL's niche ∈ target_niches
2. Tier fit:       +0.25 for mid-tier, +0.15 for others (mid-tier has highest
                   cost-efficiency in our heuristic model)
3. Text overlap:   +0.30 jaccard of 2-char n-grams between caption and the
                   KOL's nickname — proxy for content-semantic alignment.

Returns the top_k KOLs with explanations on the top_n.

Enterprise Edition: swap the scoring function with a real cross-attention
encoder trained on historical creative/KOL performance data.
"""

from __future__ import annotations

import time
import uuid

from .kol_optimizer import _EN_TO_ZH, _classify_tier, _load_pool


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

    target_zh = {_normalize_niche(n) for n in (target_niches or [])}
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
        "target_niches": target_niches or [],
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

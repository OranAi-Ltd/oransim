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


# caption → Chinese niche name · keyword match (with LLM fallback).
# Fallback for when target_niches is empty — prevents unrelated niches
# (e.g. beauty) from winning the ranking on a food caption via tier
# and text-overlap alone.
#
# Niche labels MUST match the synthetic pool's ``niche_zh`` vocabulary
# (see data/synthetic_kols.json): 健身 / 3C / 宠物 / 食饮 / 家居 / 服装
# / 饮品 / 旅游 / 美妆 / 母婴. Skincare terms are folded into 美妆 — the
# pool has no dedicated 护肤 niche.
#
# The 旅游 list had three over-broad words (vlog / 攻略 / 出行) that
# appeared in all kinds of non-travel captions (约会 vlog, 口红攻略,
# 出行妆), which used to pull every caption without strong features into
# the travel bucket. Removed — kept only high-specificity travel words.
# Intentionally kept local (not sourced from the niche registry): this is the
# caption→ZH-niche short-list used by soul prompts, whose ZH keys differ from
# the registry canonical zh labels (饮品/食饮/3C/旅游 vs 饮品/美食/数码/旅行).
# Editing the niche registry won't break this; if you want a new niche to
# participate in caption detection, add a new entry here too.
_CAPTION_NICHE_KW = [
    ("饮品", ["奶茶", "咖啡", "拿铁", "美式", "手冲", "茶饮", "气泡水",
             "汽水", "饮料", "下午茶", "果汁", "冰茶"]),
    ("美妆", ["美妆", "口红", "粉底", "眼影", "腮红", "遮瑕", "底妆", "素颜",
             "妆前", "化妆", "彩妆", "香水",
             "护肤", "面膜", "精华", "防晒", "水乳", "面霜", "洗面奶", "护肤品",
             "眼霜", "眼膜", "眼周", "眼部", "黑眼圈", "细纹", "抗老", "抗皱",
             "祛痘", "痘痘", "痘印", "保湿", "补水", "毛孔", "黑头",
             "妆容", "淡妆", "浓妆", "约会妆", "通勤妆", "持妆", "脱妆", "卸妆",
             "高光", "眉笔", "睫毛膏", "唇膏", "手霜", "身体乳"]),
    ("食饮", ["美食", "探店", "做饭", "家常", "餐厅", "零食", "甜品", "小吃",
             "便当", "拉面", "火锅", "烧烤", "轻食", "外卖", "寿司", "烘焙"]),
    ("服装", ["穿搭", "搭配", "服装", "时尚", "通勤", "约会", "OOTD",
             "潮流", "外套", "裙子", "裤子", "上衣", "西装",
             "毛衣", "卫衣", "牛仔", "连衣裙"]),
    ("3C",   ["数码", "手机", "耳机", "笔记本", "平板", "电脑", "相机",
             "键盘", "游戏本", "科技", "3C"]),
    ("母婴", ["母婴", "育儿", "妈妈", "辅食", "宝宝", "儿童", "婴儿",
             "童装", "奶粉", "纸尿裤", "孕期"]),
    ("健身", ["健身", "减脂", "跑步", "瑜伽", "运动", "健身房", "增肌",
             "塑形", "普拉提"]),
    # Travel: only keep high-specificity travel/transport/lodging terms to
    # avoid false positives on captions like "约会 vlog" or "口红攻略".
    ("旅游", ["旅行", "旅游", "酒店", "民宿", "机票", "景点", "风景",
             "自驾", "露营", "度假", "旅程", "景区", "路线"]),
    ("宠物", ["宠物", "猫", "狗", "宠粮", "猫粮", "狗粮", "铲屎", "毛孩",
             "喵", "汪", "猫砂"]),
    ("家居", ["家居", "家具", "沙发", "装修", "床垫", "收纳", "家电",
             "清洁", "家纺"]),
]

_VALID_NICHES = {t[0] for t in _CAPTION_NICHE_KW}


def detect_niche_from_caption(caption: str, use_llm_fallback: bool = True) -> str | None:
    """Sniff the most likely niche from a caption.

    Strategy:
      1. Count keyword hits per niche; take the niche with the most hits.
         Ties break by ``_CAPTION_NICHE_KW`` order (饮品 > 美妆 > 食饮 > ...).
         Avoids the old "first hit wins" bias that let a single stray
         keyword derail the ranking when another niche had more evidence.
      2. If no keyword hit and ``use_llm_fallback=True`` (default), ask
         the LLM to pick one of the niches from the vocabulary above.
         Cached in-process; transient LLM failures return None.

    Pass ``use_llm_fallback=False`` on latency-sensitive hot paths.
    Returns None when both keyword + LLM produce no match.
    """
    if not caption:
        return None
    best_niche = None
    best_count = 0
    for niche, kws in _CAPTION_NICHE_KW:
        c = sum(1 for k in kws if k in caption)
        if c > best_count:
            best_count = c
            best_niche = niche
    if best_niche:
        return best_niche
    if use_llm_fallback:
        return _detect_niche_via_llm(caption)
    return None


# ── LLM niche classifier ─────────────────────────────────────────────────
_NICHE_LLM_CACHE: dict[str, str | None] = {}
_NICHE_LLM_CACHE_MAX = 500

_NICHE_LLM_SYSTEM = """你是内容分类助手。给定一条社媒文案，从 10 个候选赛道里选出**最贴合的一个**。只输出 JSON。"""

_NICHE_LLM_PROMPT = """文案:
{caption}

候选赛道（只能从这 10 个里选一个）:
- 饮品：奶茶/咖啡/茶饮/气泡水
- 美妆：口红/化妆/香水/护肤/面膜/眼部护理
- 食饮：餐饮/做饭/探店/甜品/烘焙
- 服装：穿搭/OOTD/时尚搭配/西装
- 3C：手机/耳机/电脑/相机/数码/潮玩/盲盒
- 母婴：育儿/辅食/童装/奶粉
- 健身：运动/减脂/瑜伽/跑步
- 旅游：旅行/酒店/景点/自驾/露营
- 宠物：猫/狗/宠粮/铲屎
- 家居：家具/装修/收纳/家电

必须选一个最相近的（即使完全不契合也选最近似的，**禁止输出 none**）。
盲盒/潮玩/手办/模型 属于 3C；茶馆 属于 饮品；烘焙 属于 食饮；露营装备 属于 旅游。

严格 JSON: {{"niche": "xxx"}}  xxx 必须是上述 10 个中文赛道名之一。"""


def _detect_niche_via_llm(caption: str) -> str | None:
    key = caption.strip()[:200]
    if key in _NICHE_LLM_CACHE:
        return _NICHE_LLM_CACHE[key]
    result: str | None = None
    try:
        from .soul_llm import MODEL, call_llm_json_with_retry, llm_available

        if llm_available():
            body = {
                "model": MODEL,
                "temperature": 0.1,
                "max_tokens": 40,
                "messages": [
                    {"role": "system", "content": _NICHE_LLM_SYSTEM},
                    {"role": "user", "content": _NICHE_LLM_PROMPT.format(caption=key)},
                ],
            }
            # Retry with backoff + strict-JSON hint on parse failures
            parsed, _usage = call_llm_json_with_retry(body, max_retries=2, timeout=8)
            niche = parsed.get("niche")
            if niche in _VALID_NICHES:
                result = niche
    except Exception:
        result = None
    if len(_NICHE_LLM_CACHE) >= _NICHE_LLM_CACHE_MAX:
        _NICHE_LLM_CACHE.pop(next(iter(_NICHE_LLM_CACHE)))
    _NICHE_LLM_CACHE[key] = result
    return result


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

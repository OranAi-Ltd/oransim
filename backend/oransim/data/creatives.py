"""Mock creative / KOL / campaign objects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .population import INTEREST_DIM


@dataclass
class Creative:
    """A single ad creative (video / image post)."""

    id: str
    caption: str
    duration_sec: float
    visual_style: str  # "bright" / "dark" / "minimal" / "flashy"
    music_mood: str  # "upbeat" / "calm" / "asmr" / "dramatic"
    has_celeb: bool
    aigc_score: float  # 0..1, higher = more obvious AI-generated
    content_emb: np.ndarray  # (64,) aligned to population.interest space
    creator_id: str | None = None
    predicted_quality: float = 0.5
    audit_risk: float = 0.0  # 0..1, higher = compliance risk → throttle
    category_hint: str = "general"  # for season factor: beverage/apparel_warm/travel/...


def _hash_emb(text: str, dim: int = INTEREST_DIM, seed_offset: int = 0) -> np.ndarray:
    """Deterministic text → embedding via hash (mocks CLIP/Qwen embedding)."""
    h = hashlib.sha256((text + str(seed_offset)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.normal(0, 1, dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


_RISK_KEYWORDS = ["最", "第一", "100%", "包治", "速效", "彻底根治", "国家级", "无效退款", "顶级"]
_AIGC_HINT_KW = ["AI 生成", "AIGC", "数字人", "虚拟主播"]
# Caption → category_hint keyword table. Niche synonyms come from the niche
# registry (data/niches.json via oransim.config.niches); season-factor-only
# categories (apparel_warm for winter, delivery for weather events) stay local
# since they don't map to any niche.
from oransim.config import niches as _niche_registry

_CATEGORY_KW = dict(_niche_registry.synonyms())
_CATEGORY_KW["apparel_warm"] = ["羽绒", "外套", "保暖", "毛衣"]
_CATEGORY_KW["delivery"] = ["外卖", "美团", "饿了么"]


def make_creative(
    creative_id: str,
    caption: str,
    *,
    duration_sec: float = 15.0,
    visual_style: str = "bright",
    music_mood: str = "upbeat",
    has_celeb: bool = False,
    aigc_score: float | None = None,
    target_audience_hint: str = "",
) -> Creative:
    # Build content_emb: caption embedding + structured bias for known axes.
    emb = _hash_emb(caption)
    # bias gender (dims 0-7)
    if any(w in caption for w in ["美妆", "口红", "粉底", "妈妈", "母婴", "姐妹"]):
        emb[:8] += 0.5
    if any(w in caption for w in ["兄弟", "电竞", "机械", "刀片"]):
        emb[:8] -= 0.5
    # bias age (dims 8-23) — young-leaning
    if any(w in caption for w in ["学生", "开学", "校园", "Z世代", "二次元"]):
        emb[8:24] -= 0.4
    if any(w in caption for w in ["养生", "长辈", "退休"]):
        emb[8:24] += 0.4
    # city tier (dims 24-39) — high-tier leaning
    if any(w in caption for w in ["小众", "设计师", "咖啡", "独立", "vintage"]):
        emb[24:40] -= 0.4
    if any(w in caption for w in ["性价比", "实惠", "省钱", "团购"]):
        emb[24:40] += 0.4
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    # mock a quality score from caption length + celebrity
    quality = 0.4 + (0.15 if has_celeb else 0) + min(len(caption), 40) / 200
    quality = float(np.clip(quality, 0.2, 0.9))

    # Audit risk: violations of advertising law (绝对化用语 etc.)
    risk_hits = sum(1 for k in _RISK_KEYWORDS if k in caption)
    audit_risk = float(min(0.9, risk_hits * 0.25))

    # AIGC tag: explicit hints OR random base
    if aigc_score is None:
        aigc_score = 0.55 if any(k in caption for k in _AIGC_HINT_KW) else 0.05

    # Category hint for season factor
    cat = "general"
    for c, kws in _CATEGORY_KW.items():
        if any(k in caption for k in kws):
            cat = c
            break

    return Creative(
        id=creative_id,
        caption=caption,
        duration_sec=duration_sec,
        visual_style=visual_style,
        music_mood=music_mood,
        has_celeb=has_celeb,
        aigc_score=float(aigc_score),
        content_emb=emb,
        predicted_quality=quality,
        audit_risk=audit_risk,
        category_hint=cat,
    )

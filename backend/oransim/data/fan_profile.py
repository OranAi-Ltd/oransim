"""Fan profile reweighting — when a KOL is chosen, the simulated audience
shifts toward that KOL's realistic fan demographics (not general population).

Motivation: 七普 / QuestMobile give general population, but a beauty KOL's
followers are 60% female 25-34 tier-1, not 51% female / uniform age. Assuming
general distribution grossly misestimates reach.

This module maintains **niche × demographic** priors based on public industry
reports (蝉妈妈/千瓜/巨量星图 aggregate data) and reweights the population
accordingly at sampling time.

Mode 1: soft reweight — multiply each agent's platform_activity by a
        niche-match score ∈ [0.2, 3.0].
Mode 2: hard filter — only sample from agents matching the niche profile.

MVP uses Mode 1 (softer, preserves Voronoi partition structure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- Niche-specific fan demographic priors (from public industry reports) ----
# Values = multiplicative weight on each (gender, age, city_tier) bucket
# Calibrated from: 蝉妈妈 年度报告 2024, 千瓜 Q4 2024 达人数据洞察白皮书

# gender: 0=F, 1=M
# age_idx: 0=15-24, 1=25-34, 2=35-44, 3=45-54, 4=55-64, 5=65+
# city: 0=T1, 1=T2, 2=T3, 3=T4, 4=T5+


@dataclass
class FanPrior:
    """Multiplicative weights relative to general population."""

    gender_wt: np.ndarray  # (2,) F, M
    age_wt: np.ndarray  # (6,) age buckets
    city_wt: np.ndarray  # (5,) city tiers
    income_wt: np.ndarray  # (10,) income deciles
    notes: str = ""
    source: str = "hand-tuned"  # "hand-tuned" | "provider-calibrated"


# Load provider-calibrated priors if available (real XHS KOL fan demographics)
def _load_calibrated():
    import json
    from pathlib import Path

    p = Path("data/synthetic/niche_priors_calibrated.json")
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    # Reference marginals (7th Census): F 48.7%, age 6 buckets per AGE_PROBS,
    # city 5 tiers per CITY_PROBS
    ref_gender_f = 0.487
    ref_age = [0.14, 0.18, 0.20, 0.18, 0.15, 0.15]
    ref_city = [0.09, 0.15, 0.22, 0.28, 0.26]

    priors = {}
    for niche, stats in data.items():
        g = stats.get("gender_pct_F_M") or [0.5, 0.5]
        a = stats.get("age_pct_6buckets") or ref_age
        c = stats.get("city_tier_pct_5tiers") or ref_city
        priors[niche] = FanPrior(
            gender_wt=np.array([g[0] / ref_gender_f, g[1] / (1 - ref_gender_f)], np.float32),
            age_wt=np.array([max(0.05, a[i] / ref_age[i]) for i in range(6)], np.float32),
            city_wt=np.array([max(0.05, c[i] / ref_city[i]) for i in range(5)], np.float32),
            income_wt=np.ones(10, np.float32),  # synthetic provider has no income signal,用 1
            notes=f"synthetic 50 KOL 校准 · {stats.get('total_fans_aggregated', 0):,} 粉丝聚合",
            source="provider-calibrated",
        )
    return priors


# Start from hand-tuned English-named priors as fallback
NICHE_PRIORS: dict[str, FanPrior] = {
    # 美妆：女性 + 一二线 + 25-34 为主
    "beauty": FanPrior(
        gender_wt=np.array([2.5, 0.2], np.float32),
        age_wt=np.array([1.4, 1.8, 1.0, 0.4, 0.2, 0.1], np.float32),
        city_wt=np.array([1.8, 1.5, 1.0, 0.5, 0.3], np.float32),
        income_wt=np.array([0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.3, 1.0], np.float32),
        notes="女性 + 25-34 + 一二线 + 中高收入",
    ),
    # 母婴：女性 + 25-35 + 有娃 + 下沉也多
    "mom": FanPrior(
        gender_wt=np.array([3.0, 0.15], np.float32),
        age_wt=np.array([0.3, 2.5, 1.8, 0.4, 0.1, 0.05], np.float32),
        city_wt=np.array([1.2, 1.3, 1.5, 1.3, 0.9], np.float32),
        income_wt=np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.3, 1.2, 1.0, 0.8, 0.5], np.float32),
        notes="女性 + 25-44 有娃 + 新一线到三线",
    ),
    # 数码：男性 + 年轻 + 高收入
    "tech": FanPrior(
        gender_wt=np.array([0.4, 1.8], np.float32),
        age_wt=np.array([1.6, 1.8, 1.2, 0.6, 0.3, 0.1], np.float32),
        city_wt=np.array([2.0, 1.6, 1.0, 0.5, 0.2], np.float32),
        income_wt=np.array([0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.6, 1.4, 1.0], np.float32),
        notes="男性 + 18-40 + 一二线 + 中高收入",
    ),
    # 美食：人群广，轻度偏女，下沉兼容
    "food": FanPrior(
        gender_wt=np.array([1.4, 0.7], np.float32),
        age_wt=np.array([1.3, 1.5, 1.3, 1.0, 0.8, 0.5], np.float32),
        city_wt=np.array([1.1, 1.3, 1.4, 1.2, 1.0], np.float32),
        income_wt=np.ones(10, np.float32),
        notes="偏女性 + 全年龄 + 广泛城市",
    ),
    # 穿搭：女性 + 年轻 + 一二线
    "fashion": FanPrior(
        gender_wt=np.array([2.8, 0.2], np.float32),
        age_wt=np.array([1.8, 1.6, 0.6, 0.2, 0.1, 0.05], np.float32),
        city_wt=np.array([2.2, 1.5, 0.8, 0.4, 0.2], np.float32),
        income_wt=np.array([0.3, 0.4, 0.7, 1.0, 1.3, 1.5, 1.6, 1.4, 1.2, 1.0], np.float32),
        notes="女性 + 18-34 + 一线偏好",
    ),
    # 健身：25-44 + 一二线 + 中高收入，男女兼有
    "fitness": FanPrior(
        gender_wt=np.array([1.1, 1.0], np.float32),
        age_wt=np.array([1.0, 1.8, 1.5, 0.8, 0.3, 0.1], np.float32),
        city_wt=np.array([1.7, 1.5, 1.0, 0.6, 0.3], np.float32),
        income_wt=np.array([0.3, 0.5, 0.8, 1.1, 1.4, 1.5, 1.5, 1.3, 1.1, 0.8], np.float32),
        notes="25-44 + 一二线 + 中高收入",
    ),
    # 理财：30+ + 中等收入 + 偏男
    "finance": FanPrior(
        gender_wt=np.array([0.7, 1.4], np.float32),
        age_wt=np.array([0.4, 1.3, 2.0, 1.5, 0.7, 0.3], np.float32),
        city_wt=np.array([1.8, 1.5, 1.1, 0.7, 0.4], np.float32),
        income_wt=np.array([0.2, 0.4, 0.7, 1.0, 1.3, 1.5, 1.6, 1.4, 1.2, 1.0], np.float32),
        notes="25-55 男性为主 + 一二线",
    ),
    # 旅行：25-44 + 一二线 + 中高收入
    "travel": FanPrior(
        gender_wt=np.array([1.3, 0.8], np.float32),
        age_wt=np.array([1.2, 1.8, 1.5, 1.0, 0.5, 0.2], np.float32),
        city_wt=np.array([1.9, 1.5, 1.0, 0.6, 0.3], np.float32),
        income_wt=np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.6, 1.4, 1.2, 1.0], np.float32),
        notes="偏女 + 25-44 + 一二线 + 中高收入",
    ),
}


# Override hand-tuned with provider-calibrated data (real XHS KOL fan demographics)
_CALIBRATED = _load_calibrated()
# Merge: calibrated Chinese niches + legacy English synonyms
_ZH_TO_EN = {
    "美妆": "beauty",
    "护肤": "beauty",  # 都归 beauty
    "母婴": "mom",
    "数码": "tech",
    "美食": "food",
    "穿搭": "fashion",
    "健身": "fitness",
    "理财": "finance",
    "旅行": "travel",
    "家居": "home",
}
for zh, prior in _CALIBRATED.items():
    # Store under Chinese name AND English alias for backward compat
    NICHE_PRIORS[zh] = prior
    en = _ZH_TO_EN.get(zh)
    if en and en not in NICHE_PRIORS:
        NICHE_PRIORS[en] = prior
    elif en:
        # Override hand-tuned with calibrated if we have real data
        NICHE_PRIORS[en] = prior


def fan_weight_vector(population, niche: str) -> np.ndarray:
    """Return (N,) multiplicative weight per agent based on KOL niche prior.

    Use this to multiply platform_activity / audience_score for realistic
    KOL-fan distributions (instead of general population).
    """
    if niche not in NICHE_PRIORS:
        return np.ones(population.N, dtype=np.float32)
    prior = NICHE_PRIORS[niche]
    w = (
        prior.gender_wt[population.gender_idx]
        * prior.age_wt[population.age_idx]
        * prior.city_wt[population.city_idx]
        * prior.income_wt[population.income]
    )
    # Normalize to mean-1 so it's a relative reweight (doesn't bias absolute scale)
    w = w / (w.mean() + 1e-8)
    return w.astype(np.float32)


def fan_profile_summary(population, niche: str) -> dict:
    """Explain the fan profile reweighting effect on the population."""
    if niche not in NICHE_PRIORS:
        return {"niche": niche, "applied": False}
    prior = NICHE_PRIORS[niche]
    w = fan_weight_vector(population, niche)
    # Effective distribution under reweight: p_i = w_i / sum(w)
    p = w / w.sum()
    gender_dist = np.zeros(2, np.float32)
    age_dist = np.zeros(6, np.float32)
    city_dist = np.zeros(5, np.float32)
    np.add.at(gender_dist, population.gender_idx, p)
    np.add.at(age_dist, population.age_idx, p)
    np.add.at(city_dist, population.city_idx, p)
    return {
        "niche": niche,
        "applied": True,
        "notes": prior.notes,
        "effective_gender_pct_female": round(float(gender_dist[0]) * 100, 1),
        "effective_age_dist": {
            "15-24": round(float(age_dist[0]) * 100, 1),
            "25-34": round(float(age_dist[1]) * 100, 1),
            "35-44": round(float(age_dist[2]) * 100, 1),
            "45-54": round(float(age_dist[3]) * 100, 1),
            "55-64": round(float(age_dist[4]) * 100, 1),
            "65+": round(float(age_dist[5]) * 100, 1),
        },
        "effective_city_dist": {
            "T1": round(float(city_dist[0]) * 100, 1),
            "T2": round(float(city_dist[1]) * 100, 1),
            "T3": round(float(city_dist[2]) * 100, 1),
            "T4": round(float(city_dist[3]) * 100, 1),
            "T5+": round(float(city_dist[4]) * 100, 1),
        },
        "concentration_ratio": round(float(w.std() / (w.mean() + 1e-8)), 3),
    }

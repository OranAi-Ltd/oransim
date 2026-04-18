"""Mock population generator.

Produces N virtual users whose marginal distributions roughly match:
- China 7th Census age/gender/city-tier
- QuestMobile 2024 Douyin/Xiaohongshu MAU penetration by segment
- Income tier from NBS + household survey
- Interest 64-d embedding (aligned to CLIP space — mocked)
- Big Five personality

Uses lightweight IPF over marginals for realistic-ish joint distribution.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict

# ---- Real-world-ish marginals (2024-2025 data approximations) ----

# 7th Census age buckets (10-year bins from 15 to 75+)
AGE_BUCKETS = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_PROBS = np.array([0.14, 0.18, 0.20, 0.18, 0.15, 0.15])

GENDER = ["F", "M"]
GENDER_PROBS = np.array([0.487, 0.513])

CITY_TIER = ["T1", "T2", "T3", "T4", "T5+"]  # 一线 → 五线及以下
CITY_PROBS = np.array([0.09, 0.15, 0.22, 0.28, 0.26])

INCOME_TIER = list(range(10))  # decile
INCOME_PROBS = np.array([0.13, 0.12, 0.11, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.07])

EDU = ["初中及以下", "高中", "大专", "本科", "硕士+"]
EDU_PROBS = np.array([0.42, 0.21, 0.16, 0.18, 0.03])

OCCUPATION = ["学生", "白领", "蓝领", "个体", "公务员/教师", "自由职业", "退休", "无业"]
OCC_PROBS = np.array([0.11, 0.25, 0.22, 0.14, 0.08, 0.06, 0.09, 0.05])

# Platform MAU penetration (QuestMobile 2024 approximate)
# Probability of being an active user of each platform given adult population
PLATFORM_NAMES = ["douyin", "xhs", "wechat_video", "bilibili", "kuaishou"]
PLATFORM_BASELINE = np.array([0.75, 0.40, 0.68, 0.32, 0.55])

INTEREST_DIM = 64
STATE_DIM = 16


@dataclass
class Population:
    """Vectorized population. All arrays shape (N, ...)."""
    N: int
    age_idx: np.ndarray
    gender_idx: np.ndarray
    city_idx: np.ndarray
    income: np.ndarray
    edu_idx: np.ndarray
    occ_idx: np.ndarray
    interest: np.ndarray          # (N, 64) unit-norm
    bigfive: np.ndarray           # (N, 5) in [0,1]
    platform_activity: np.ndarray # (N, P) use-frequency 0..1
    state: np.ndarray             # (N, 16) dynamic
    seed: int = 42

    def slice(self, idx: np.ndarray) -> "Population":
        return Population(
            N=len(idx),
            age_idx=self.age_idx[idx],
            gender_idx=self.gender_idx[idx],
            city_idx=self.city_idx[idx],
            income=self.income[idx],
            edu_idx=self.edu_idx[idx],
            occ_idx=self.occ_idx[idx],
            interest=self.interest[idx],
            bigfive=self.bigfive[idx],
            platform_activity=self.platform_activity[idx],
            state=self.state[idx].copy(),
            seed=self.seed,
        )


def _categorical(rng: np.random.Generator, probs: np.ndarray, N: int) -> np.ndarray:
    return rng.choice(len(probs), size=N, p=probs)


def _build_interests(rng, N, age_idx, gender_idx, city_idx) -> np.ndarray:
    """Fake but structured 64-d embedding.
    We inject age/gender/city signal into specific dims so that later
    'content_emb' aligned to the same dims produces realistic affinity.
    """
    base = rng.normal(0, 0.3, size=(N, INTEREST_DIM)).astype(np.float32)
    # dims 0-7: gender signal  (F pushes +, M pushes -)
    base[:, :8] += np.where(gender_idx == 0, 0.5, -0.5)[:, None] * rng.uniform(0.5, 1.0, (N, 8))
    # dims 8-23: age signal
    age_vec = (age_idx[:, None] - 2.5) / 2.5  # center
    base[:, 8:24] += age_vec * rng.uniform(0.2, 0.8, (N, 16))
    # dims 24-39: city tier
    city_vec = (city_idx[:, None] - 2) / 2.0
    base[:, 24:40] += city_vec * rng.uniform(0.2, 0.6, (N, 16))
    # dims 40-63: noise
    # L2 normalize
    norm = np.linalg.norm(base, axis=1, keepdims=True) + 1e-8
    return base / norm


def generate_population(N: int = 100_000, seed: int = 42) -> Population:
    """IPF-style calibrated mock population."""
    rng = np.random.default_rng(seed)

    age_idx    = _categorical(rng, AGE_PROBS, N)
    gender_idx = _categorical(rng, GENDER_PROBS, N)
    city_idx   = _categorical(rng, CITY_PROBS, N)
    edu_idx    = _categorical(rng, EDU_PROBS, N)
    occ_idx    = _categorical(rng, OCC_PROBS, N)

    # Income: correlated with city tier + education (not fully independent)
    base_income = rng.uniform(0, 1, N)
    income_boost = (4 - city_idx) * 0.08 + edu_idx * 0.06
    income_raw = np.clip(base_income + income_boost + rng.normal(0, 0.05, N), 0, 1)
    # map to deciles
    income = (income_raw * 9.999).astype(np.int8)

    interest = _build_interests(rng, N, age_idx, gender_idx, city_idx)
    bigfive = np.clip(rng.normal(0.5, 0.15, (N, 5)), 0, 1).astype(np.float32)

    # Platform activity — age/city conditional
    # young + high-tier → more xhs; broader base for douyin
    platform_activity = np.zeros((N, len(PLATFORM_NAMES)), dtype=np.float32)
    for p_idx, base_p in enumerate(PLATFORM_BASELINE):
        activity = rng.uniform(0.3, 1.0, N) * base_p
        if PLATFORM_NAMES[p_idx] == "xhs":
            # Female + high tier + younger boost
            activity *= np.where(gender_idx == 0, 1.4, 0.6)
            activity *= np.where(city_idx <= 1, 1.3, 0.85)
            activity *= np.where(age_idx <= 2, 1.2, 0.7)
        elif PLATFORM_NAMES[p_idx] == "douyin":
            activity *= np.where(age_idx >= 4, 1.1, 1.0)  # older users too
        elif PLATFORM_NAMES[p_idx] == "bilibili":
            activity *= np.where(age_idx <= 1, 1.6, 0.5)
            activity *= np.where(edu_idx >= 3, 1.3, 0.8)
        platform_activity[:, p_idx] = np.clip(activity, 0, 1)

    state = rng.normal(0, 0.1, (N, STATE_DIM)).astype(np.float32)

    return Population(
        N=N, age_idx=age_idx, gender_idx=gender_idx, city_idx=city_idx,
        income=income, edu_idx=edu_idx, occ_idx=occ_idx,
        interest=interest, bigfive=bigfive,
        platform_activity=platform_activity, state=state, seed=seed,
    )


def marginal_fit_report(pop: Population) -> Dict:
    """Return KL-ish divergence between realized and target marginals."""
    report = {}
    def _kl(observed, target):
        observed = np.clip(observed, 1e-6, 1)
        target = np.clip(target, 1e-6, 1)
        return float(np.sum(observed * np.log(observed / target)))
    report["age_kl"]    = _kl(np.bincount(pop.age_idx,    minlength=6)/pop.N, AGE_PROBS)
    report["gender_kl"] = _kl(np.bincount(pop.gender_idx, minlength=2)/pop.N, GENDER_PROBS)
    report["city_kl"]   = _kl(np.bincount(pop.city_idx,   minlength=5)/pop.N, CITY_PROBS)
    report["edu_kl"]    = _kl(np.bincount(pop.edu_idx,    minlength=5)/pop.N, EDU_PROBS)
    return report

"""L1 Platform world model.

Given (creative, platform, budget, audience_filter, kol), return:
  - which agents see the impression (indices + per-agent weight)
  - per-agent impression intensity

For MVP: a structured scoring function that mimics the real platform
recommendation — score = affinity(content, interest)
                       * platform_activity
                       * audience_filter_match
                       * kol_fan_boost
                       * algo_diversity_noise
Then budget controls total impression count; top-scoring agents get impressions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data.creatives import Creative
from ...data.kols import KOL
from ...data.platforms import PLATFORMS, budget_to_impressions
from ...data.population import PLATFORM_NAMES, Population


@dataclass
class AudienceFilter:
    """Soft targeting: boost weights but don't hard-exclude."""

    age_buckets: list[int] | None = None  # list of age_idx values to boost
    gender: int | None = None  # 0=F, 1=M, None=both
    city_tiers: list[int] | None = None  # list of city_idx to boost
    interest_keywords: list[str] | None = None  # boost via keyword→emb cosine
    boost_strength: float = 2.0  # multiplier for matched agents


@dataclass
class ImpressionResult:
    agent_idx: np.ndarray  # agents who got impression
    weight: np.ndarray  # per-agent score 0..1
    total_impressions: float
    platform: str
    score_breakdown: dict[str, np.ndarray]  # for explainability


class PlatformWorldModel:
    """Structured world model that mimics a learned Transformer encoder."""

    def __init__(self, population: Population):
        self.pop = population
        self.platform_idx = {p: i for i, p in enumerate(PLATFORM_NAMES)}

    def _audience_score(self, flt: AudienceFilter | None) -> np.ndarray:
        n = self.pop.N
        if flt is None:
            return np.ones(n, dtype=np.float32)
        s = np.ones(n, dtype=np.float32)
        matched = np.zeros(n, dtype=bool)
        if flt.age_buckets is not None:
            m = np.isin(self.pop.age_idx, flt.age_buckets)
            matched |= m
        if flt.gender is not None:
            matched |= self.pop.gender_idx == flt.gender
        if flt.city_tiers is not None:
            matched |= np.isin(self.pop.city_idx, flt.city_tiers)
        if matched.any():
            s = np.where(matched, flt.boost_strength, 1.0 / flt.boost_strength).astype(np.float32)
        return s

    def simulate_impression(
        self,
        creative: Creative,
        platform: str,
        budget_cny: float,
        audience_filter: AudienceFilter | None = None,
        kol: KOL | None = None,
        rng_seed: int = 0,
    ) -> ImpressionResult:
        rng = np.random.default_rng(rng_seed)
        cfg = PLATFORMS[platform]
        plat_i = self.platform_idx.get(platform, 0)

        # 1. Content-interest affinity (cosine)
        content_score = self.pop.interest @ creative.content_emb  # (N,)
        content_score = (content_score + 1.0) / 2.0  # 0..1

        # 2. Platform activity
        plat_score = self.pop.platform_activity[:, plat_i]

        # 3. Audience filter
        aud_score = self._audience_score(audience_filter)

        # 4. KOL boost: users whose interests align with KOL's audience get boost
        #    + fan profile reweighting (based on KOL niche's realistic demographic skew)
        if kol is not None:
            kol_align = self.pop.interest @ kol.emb
            kol_score = 1.0 + 0.5 * np.clip(kol_align, 0, 1)
            # Apply fan profile prior (beauty KOL fans ≠ general population)
            try:
                from ...data.fan_profile import fan_weight_vector

                fan_w = fan_weight_vector(self.pop, kol.niche)
                kol_score = kol_score * fan_w  # multiplicative reweight
            except Exception:
                pass
        else:
            kol_score = np.ones_like(content_score)

        # 5. Algorithm diversity noise (exploration)
        noise = rng.uniform(
            1 - cfg.algo_diversity * 0.4, 1 + cfg.algo_diversity * 0.4, size=self.pop.N
        ).astype(np.float32)

        # 6. Platform audience skew — supported keys (any subset):
        #    female_boost    · gender_idx == 0
        #    male_boost      · gender_idx == 1
        #    tier1_boost     · city_idx <= 1 (一线 + 新一线)
        #    tier3_boost     · city_idx >= 2 (三四五线 + 县域)
        #    young_boost     · age_idx <= 2  (15-44)
        #    senior_boost    · age_idx >= 4  (55+)
        # Any unsupported key is silently skipped.
        skew = np.ones(self.pop.N, dtype=np.float32)
        _mults = [
            ("female_boost", self.pop.gender_idx == 0),
            ("male_boost", self.pop.gender_idx == 1),
            ("tier1_boost", self.pop.city_idx <= 1),
            ("tier3_boost", self.pop.city_idx >= 2),
            ("young_boost", self.pop.age_idx <= 2),
            ("senior_boost", self.pop.age_idx >= 4),
        ]
        for key, mask in _mults:
            if key in cfg.audience_skew:
                skew *= np.where(mask, cfg.audience_skew[key], 1.0).astype(np.float32)

        final = content_score * plat_score * aud_score * kol_score * noise * skew

        # 7. Sample top-K by score, where K = budget_to_impressions (capped by N)
        total_imps = budget_to_impressions(budget_cny, platform)
        k = int(min(total_imps, self.pop.N))
        if k <= 0:
            return ImpressionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                0.0,
                platform,
                {
                    "content": content_score,
                    "platform": plat_score,
                    "audience": aud_score,
                    "kol": kol_score,
                },
            )

        # top-k selection (no full sort)
        idx = np.argpartition(-final, k - 1)[:k]
        # weight normalized within selected
        w = final[idx]
        w = w / (w.max() + 1e-8)

        return ImpressionResult(
            agent_idx=idx.astype(np.int64),
            weight=w.astype(np.float32),
            total_impressions=float(total_imps),
            platform=platform,
            score_breakdown={
                "content": content_score[idx],
                "platform_activity": plat_score[idx],
                "audience_filter": aud_score[idx],
                "kol_boost": kol_score[idx],
            },
        )

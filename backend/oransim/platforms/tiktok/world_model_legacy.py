"""TikTok platform world model.

TikTok shares ByteDance's recommendation stack with Douyin, so the
underlying user-affinity signal is reused from the ``douyin`` activity
column on the population matrix (there is no separate ``tiktok`` column
in ``PLATFORM_NAMES`` — adding one would require retraining the
downstream world-model pkl).

What differs from Douyin is the scoring dynamics:

  - **FYP exploration dominates targeting.** Explicit audience filters
    have less lever here than on XHS/instagram — the algorithm routes
    content largely by engagement signal, not by declared segment. So
    we dampen the ``audience_filter`` contribution to 60% of its
    XHS-equivalent strength.
  - **Duration retention is a first-class score factor.** TikTok
    retention collapses past ~30 s for most categories. We apply a
    duration-keyed multiplier so short-enough videos win regardless of
    content-interest ties.
  - **Higher algo noise.** ``algo_diversity = 0.92`` (see
    ``data/platforms.py``) already exposes this, but we apply an
    additional ±10% draw on top so FYP's pseudorandom exploration is
    visible at the per-impression level.

This module is consumed by ``TikTokAdapter.simulate_impression_agents``
and by ``TikTokRecSysRLSimulator`` for the FYP cold-start loop.
"""

from __future__ import annotations

import numpy as np

from ...data.creatives import Creative
from ...data.kols import KOL
from ...data.platforms import PLATFORMS, budget_to_impressions
from ..xhs.world_model_legacy import (
    AudienceFilter,
    ImpressionResult,
    PlatformWorldModel,
)

# Reuse XHS's ImpressionResult / AudienceFilter / PlatformWorldModel
# base class so downstream consumers (counterfactual, cross_platform)
# remain platform-agnostic.
__all__ = ["AudienceFilter", "ImpressionResult", "TikTokWorldModel"]


class TikTokWorldModel(PlatformWorldModel):
    """FYP-aware world model extending the generic PlatformWorldModel.

    Overrides ``simulate_impression`` to:

      - dampen audience-filter lever (FYP exploration vs. explicit targeting)
      - apply duration-keyed retention multiplier
      - add an extra layer of algo exploration noise

    Everything else (KOL fan profile, platform activity lookup, budget
    → impression count) is inherited.
    """

    # Audience filter is dampened toward 1.0; matched agents still boost,
    # but unmatched agents are not as heavily punished as on XHS.
    _audience_damp = 0.6

    # Duration scoring (sigmoid-like peak around 24s, half at 42s).
    _duration_peak_sec = 24.0
    _duration_half_sec = 42.0

    # Extra FYP exploration noise on top of the platform base diversity.
    _fyp_noise_pct = 0.10

    def _duration_retention(self, creative: Creative) -> float:
        d = float(getattr(creative, "duration_sec", 20.0)) or 20.0
        retention = 1.0 / (1.0 + ((d / self._duration_half_sec) ** 2))
        # Normalize so a 20s video sits ~1.0 and extremes stay in [0.35, 1.15].
        return float(max(0.35, min(1.15, retention * 1.2)))

    def simulate_impression(
        self,
        creative: Creative,
        platform: str,
        budget_cny: float,
        audience_filter: AudienceFilter | None = None,
        kol: KOL | None = None,
        rng_seed: int = 0,
    ) -> ImpressionResult:
        # Force platform="tiktok" so PLATFORMS lookup + CPM math uses the
        # TikTok config even if callers pass a legacy alias.
        if platform != "tiktok":
            platform = "tiktok"
        rng = np.random.default_rng(rng_seed)
        cfg = PLATFORMS[platform]
        # FYP routes via the ByteDance stack → share Douyin's activity
        # column rather than treating TikTok as an unknown platform.
        plat_i = self.platform_idx.get("douyin", 0)

        # 1. Content-interest affinity (cosine, shifted to 0..1).
        content_score = self.pop.interest @ creative.content_emb
        content_score = (content_score + 1.0) / 2.0

        # 2. Platform activity (douyin column proxies TikTok).
        plat_score = self.pop.platform_activity[:, plat_i]

        # 3. Audience filter — dampened toward 1.0 to reflect FYP routing.
        aud_raw = self._audience_score(audience_filter)
        aud_score = 1.0 + (aud_raw - 1.0) * self._audience_damp

        # 4. KOL boost (same formula as base) + niche fan profile reweight.
        if kol is not None:
            kol_align = self.pop.interest @ kol.emb
            kol_score = 1.0 + 0.5 * np.clip(kol_align, 0, 1)
            try:
                from ...data.fan_profile import fan_weight_vector

                fan_w = fan_weight_vector(self.pop, kol.niche)
                kol_score = kol_score * fan_w
            except Exception:
                pass
        else:
            kol_score = np.ones_like(content_score)

        # 5. Base algo diversity noise + extra FYP noise draw.
        base_noise = rng.uniform(
            1 - cfg.algo_diversity * 0.4,
            1 + cfg.algo_diversity * 0.4,
            size=self.pop.N,
        ).astype(np.float32)
        fyp_noise = rng.uniform(
            1 - self._fyp_noise_pct, 1 + self._fyp_noise_pct, size=self.pop.N
        ).astype(np.float32)
        noise = base_noise * fyp_noise

        # 6. Duration retention (scalar, applied uniformly).
        retention = self._duration_retention(creative)

        final = content_score * plat_score * aud_score * kol_score * noise * retention

        total_imps = budget_to_impressions(budget_cny, platform)
        k = int(min(total_imps, self.pop.N))
        if k <= 0:
            empty = np.array([], dtype=np.int64)
            return ImpressionResult(
                agent_idx=empty,
                weight=np.array([], dtype=np.float32),
                total_impressions=0.0,
                platform=platform,
                score_breakdown={
                    "content": content_score,
                    "platform_activity": plat_score,
                    "audience_filter": aud_score,
                    "kol_boost": kol_score,
                    "duration_retention": np.full(self.pop.N, retention, dtype=np.float32),
                },
            )

        idx = np.argpartition(-final, k - 1)[:k]
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
                "duration_retention": np.full(len(idx), retention, dtype=np.float32),
            },
        )

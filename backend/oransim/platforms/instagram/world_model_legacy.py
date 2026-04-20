"""Instagram platform world model.

Instagram Reels runs a Meta-native recommendation stack that is
FYP-adjacent but preserves more explicit-targeting signal than
TikTok (saves, follows, and shares-to-Stories are first-class
engagement outputs). We subclass the XHS ``PlatformWorldModel`` base
(the generic per-agent ranker) rather than TikTok's FYP subclass, so
the audience-filter lever stays closer to full strength:

  - ``_audience_damp = 0.85`` — respect explicit-targeting more than
    TikTok (``0.6``) but less than XHS (base ``1.0``).
  - Slightly longer tolerated duration peak (~26 s).
  - Reels exploration noise narrower than FYP (algo_diversity driven).
  - ``platform_activity`` uses the Instagram column on the population
    matrix (added lazily; falls through to Douyin column if IG isn't
    a registered activity column — current OSS v0.2 state).

``ImpressionResult`` schema kept identical to XHS/TikTok so downstream
consumers (sandbox, counterfactual) remain platform-agnostic.
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

__all__ = ["AudienceFilter", "ImpressionResult", "InstagramWorldModel"]


class InstagramWorldModel(PlatformWorldModel):
    """Instagram Reels-tuned world model."""

    _audience_damp = 0.85
    _duration_peak_sec = 26.0
    _duration_half_sec = 48.0
    _reels_noise_pct = 0.07

    def _duration_retention(self, creative: Creative) -> float:
        d = float(getattr(creative, "duration_sec", 22.0)) or 22.0
        retention = 1.0 / (1.0 + ((d / self._duration_half_sec) ** 2))
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
        platform = "instagram"
        rng = np.random.default_rng(rng_seed)
        cfg = PLATFORMS[platform]
        # No native 'instagram' column on the population matrix — fall
        # through to the most similar CN column (Douyin's FYP signal
        # best approximates Reels' content-affinity score). Swap in a
        # real column when the next population revision includes it.
        plat_i = self.platform_idx.get("instagram", self.platform_idx.get("douyin", 0))

        content_score = (self.pop.interest @ creative.content_emb + 1.0) / 2.0
        plat_score = self.pop.platform_activity[:, plat_i]
        aud_raw = self._audience_score(audience_filter)
        aud_score = 1.0 + (aud_raw - 1.0) * self._audience_damp

        if kol is not None:
            kol_align = self.pop.interest @ kol.emb
            kol_score = 1.0 + 0.5 * np.clip(kol_align, 0, 1)
        else:
            kol_score = np.ones_like(content_score)

        base_noise = rng.uniform(
            1 - cfg.algo_diversity * 0.35,
            1 + cfg.algo_diversity * 0.35,
            size=self.pop.N,
        ).astype(np.float32)
        reels_noise = rng.uniform(
            1 - self._reels_noise_pct, 1 + self._reels_noise_pct, size=self.pop.N
        ).astype(np.float32)
        noise = base_noise * reels_noise
        retention = self._duration_retention(creative)
        skew = self._audience_skew_multiplier(cfg)

        final = content_score * plat_score * aud_score * kol_score * noise * retention * skew
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

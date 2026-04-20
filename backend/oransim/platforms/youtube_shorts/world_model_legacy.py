"""YouTube Shorts platform world model.

YouTube Shorts sits between long-form YouTube's subscriber-driven
retention physics and TikTok's pure FYP physics:

  - Explicit targeting (subscribe graph, channel affinity) survives
    further into the ranking path than on TikTok, so we keep
    ``_audience_damp = 0.8`` (tighter than TikTok ``0.6``, looser than
    XHS ``1.0``).
  - Peak retention duration is ~30 s — Shorts cap at 60 s and completion
    curves flatten later than on Douyin.
  - Shorts exploration noise sits between TikTok's FYP draw and
    Instagram's Reels draw.
  - Platform activity falls through to Douyin's column (same FYP
    content-affinity baseline) until a native YT column is added to
    the population matrix.

Downstream ``ImpressionResult`` shape matches XHS/TikTok.
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

__all__ = ["AudienceFilter", "ImpressionResult", "YouTubeShortsWorldModel"]


class YouTubeShortsWorldModel(PlatformWorldModel):
    """YouTube Shorts-tuned world model."""

    _audience_damp = 0.8
    _duration_peak_sec = 30.0
    _duration_half_sec = 55.0
    _shorts_noise_pct = 0.09

    def _duration_retention(self, creative: Creative) -> float:
        d = float(getattr(creative, "duration_sec", 28.0)) or 28.0
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
        platform = "youtube_shorts"
        rng = np.random.default_rng(rng_seed)
        cfg = PLATFORMS[platform]
        plat_i = self.platform_idx.get("youtube_shorts", self.platform_idx.get("douyin", 0))

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
            1 - cfg.algo_diversity * 0.4,
            1 + cfg.algo_diversity * 0.4,
            size=self.pop.N,
        ).astype(np.float32)
        shorts_noise = rng.uniform(
            1 - self._shorts_noise_pct, 1 + self._shorts_noise_pct, size=self.pop.N
        ).astype(np.float32)
        noise = base_noise * shorts_noise
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

"""Douyin platform world model.

Douyin runs on the same ByteDance recommendation stack as TikTok, so
we inherit :class:`TikTokWorldModel` and override only the physics
parameters that differ between the CN (Douyin) and global (TikTok)
products:

  - **Peak retention duration lower on Douyin (~18 s)** — CN user
    baseline tolerates shorter videos before drop-off. ``_duration_peak_sec``
    / ``_duration_half_sec`` both tightened.
  - **Livestream-commerce tilt.** Douyin's ad inventory is heavily
    weighted toward livestream GMV; if the creative visual_style is
    ``"livestream"`` we apply an extra multiplier that TikTok does not.
    This flips on through the scored ``content_score`` path rather than
    a platform-wide scalar so agents with low livestream propensity
    aren't over-boosted.
  - **Platform string routed to ``"douyin"``** (not ``"tiktok"``) so
    ``PLATFORMS["douyin"]`` CPM / skew is used, and the
    ``platform_activity`` column lookup goes through Douyin directly.

Everything else (FYP exploration noise, dampened audience filter,
score_breakdown shape) is inherited. Downstream consumers (sandbox,
counterfactual) stay platform-agnostic because the returned
``ImpressionResult`` schema is identical to TikTok's.
"""

from __future__ import annotations

import numpy as np

from ...data.creatives import Creative
from ...data.kols import KOL
from ...data.platforms import PLATFORMS, budget_to_impressions
from ..tiktok.world_model_legacy import TikTokWorldModel
from ..xhs.world_model_legacy import AudienceFilter, ImpressionResult

__all__ = ["AudienceFilter", "ImpressionResult", "DouyinWorldModel"]


class DouyinWorldModel(TikTokWorldModel):
    """Greater-China Douyin tuning of the FYP world model."""

    _duration_peak_sec = 18.0
    _duration_half_sec = 35.0
    _livestream_boost = 1.25

    def _livestream_multiplier(self, creative: Creative) -> float:
        if getattr(creative, "visual_style", "") == "livestream":
            return float(self._livestream_boost)
        return 1.0

    def simulate_impression(
        self,
        creative: Creative,
        platform: str,
        budget_cny: float,
        audience_filter: AudienceFilter | None = None,
        kol: KOL | None = None,
        rng_seed: int = 0,
    ) -> ImpressionResult:
        platform = "douyin"
        rng = np.random.default_rng(rng_seed)
        cfg = PLATFORMS[platform]
        plat_i = self.platform_idx.get("douyin", 0)

        content_score = (self.pop.interest @ creative.content_emb + 1.0) / 2.0
        plat_score = self.pop.platform_activity[:, plat_i]
        aud_raw = self._audience_score(audience_filter)
        aud_score = 1.0 + (aud_raw - 1.0) * self._audience_damp

        if kol is not None:
            kol_align = self.pop.interest @ kol.emb
            kol_score = 1.0 + 0.5 * np.clip(kol_align, 0, 1)
            try:
                from ...data.fan_profile import fan_weight_vector

                kol_score = kol_score * fan_weight_vector(self.pop, kol.niche)
            except Exception:
                pass
        else:
            kol_score = np.ones_like(content_score)

        base_noise = rng.uniform(
            1 - cfg.algo_diversity * 0.4,
            1 + cfg.algo_diversity * 0.4,
            size=self.pop.N,
        ).astype(np.float32)
        fyp_noise = rng.uniform(
            1 - self._fyp_noise_pct, 1 + self._fyp_noise_pct, size=self.pop.N
        ).astype(np.float32)
        noise = base_noise * fyp_noise

        retention = self._duration_retention(creative)
        livestream_mult = self._livestream_multiplier(creative)
        skew = self._audience_skew_multiplier(cfg)

        final = (
            content_score
            * plat_score
            * aud_score
            * kol_score
            * noise
            * retention
            * livestream_mult
            * skew
        )
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
                    "livestream_boost": np.full(self.pop.N, livestream_mult, dtype=np.float32),
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
                "livestream_boost": np.full(len(idx), livestream_mult, dtype=np.float32),
            },
        )

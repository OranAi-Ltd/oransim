"""Instagram Reels platform adapter.

Global Meta short-video surface. Unlike TikTok's FYP-first discovery,
Reels inherits a "follow-graph + recommendation" hybrid: content from
followed accounts mixes with recommended Reels at a roughly 30:70 ratio.
The adapter's priors reflect this — lower cold-start penalty because
existing followers provide baseline distribution, but higher variance
on breakout reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...data.schema import CanonicalKOL
from ...world_model.budget import BudgetCurveConfig, apply_budget_curves
from ..base import PlatformAdapter


@dataclass
class InstagramAdapterConfig:
    """Instagram Reels-specific calibration overrides."""

    platform_id: str = "instagram"
    region: str = "global"

    # Reels CPM in USD — Meta auction pricing, generally higher than TikTok
    # due to higher-value advertiser competition (brands + direct response).
    cpm_usd: float = 7.2

    # Existing follower graph shortens cold-start vs a pure-FYP platform.
    cold_start_days: float = 0.3

    # Baseline CTR lower than TikTok — Reels audience is less
    # shopping-primed than TikTok's FYP crowd.
    base_ctr: float = 0.014
    base_cvr: float = 0.009

    # Reels prefers 15-30 s content; drops off sharply past 60 s.
    duration_peak_sec: float = 18.0
    duration_half_sec: float = 38.0

    # Audio hook multiplier — Reels ads using trending audio see ~20% lift.
    trending_audio_boost: float = 1.20

    budget_curve: BudgetCurveConfig = field(default_factory=BudgetCurveConfig)


class InstagramAdapter(PlatformAdapter):
    """Instagram Reels platform adapter — v0.2 MVP."""

    platform_id: str = "instagram"

    def __init__(
        self,
        data_provider: Any = None,
        config: InstagramAdapterConfig | None = None,
    ):
        self.config = config or InstagramAdapterConfig()
        self.data_provider = data_provider

    def simulate_impression(
        self,
        creative: Any,
        budget: float,
        *,
        reference_budget: float = 50_000.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        cfg = self.config
        ref_impressions = (reference_budget / max(cfg.cpm_usd, 0.01)) * 1000.0
        duration = float(getattr(creative, "duration_sec", 18.0)) or 18.0
        duration_retention = 1.0 / (1.0 + ((duration / cfg.duration_half_sec) ** 2))
        duration_retention = max(0.35, min(1.15, duration_retention * 1.2))

        # Trending-audio boost: surfaces when `creative.music_mood` is
        # "trending" (or any non-empty value — kept permissive for MVP).
        has_trending_audio = getattr(creative, "music_mood", "") == "trending"
        audio_boost = cfg.trending_audio_boost if has_trending_audio else 1.0

        curves = apply_budget_curves(
            baseline_impressions=ref_impressions,
            baseline_clicks=ref_impressions * cfg.base_ctr * audio_boost,
            baseline_conversions=ref_impressions * cfg.base_ctr * cfg.base_cvr * audio_boost,
            budget_ratio=budget / reference_budget,
            config=cfg.budget_curve,
        )
        return {
            "platform": cfg.platform_id,
            "impressions": curves["impressions"] * duration_retention,
            "clicks": curves["clicks"] * duration_retention,
            "conversions": curves["conversions"] * duration_retention,
            "factors": {
                "cpm_usd":              cfg.cpm_usd,
                "duration_retention":   round(duration_retention, 3),
                "trending_audio_boost": audio_boost,
                "effective_impr_ratio": curves["effective_impr_ratio"],
                "ctr_decay":            curves["ctr_decay"],
                "cvr_decay":            curves["cvr_decay"],
                "cold_start_days":      cfg.cold_start_days,
            },
        }

    def simulate_conversion(self, impression: Any, **_kwargs: Any) -> dict[str, Any]:
        imp = impression.get("impressions", 0.0) if isinstance(impression, dict) else 0.0
        return {"conversions": imp * self.config.base_ctr * self.config.base_cvr}

    def get_kol(self, kol_id: str) -> CanonicalKOL:
        if self.data_provider is None:
            raise RuntimeError(
                "InstagramAdapter has no data_provider attached. "
                "Bind one: adapter.data_provider = InstagramSyntheticProvider()"
            )
        return self.data_provider.fetch_kol(kol_id)

"""YouTube Shorts platform adapter.

Global. Distinctive traits compared to TikTok / Instagram / Douyin:

- **Search-driven long tail** — YouTube's universal search surfaces Shorts
  beyond the Shorts feed itself, giving impressions a slower-decaying
  distribution than pure feed platforms.
- **Cross-format spillover** — Shorts KOLs often have matching long-form
  channels, so engagement can trigger subscribe-to-channel flows that
  compound over time.
- **Sensitive to subscribe-button prominence** — `creative.has_subscribe_cta`
  drives a measurable conversion lift on this platform specifically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...data.schema import CanonicalKOL
from ...world_model.budget import BudgetCurveConfig, apply_budget_curves
from ..base import PlatformAdapter


@dataclass
class YouTubeShortsAdapterConfig:
    """YouTube Shorts-specific calibration overrides."""

    platform_id: str = "youtube_shorts"
    region: str = "global"

    # CPM USD — premium advertiser base + Google auction pricing.
    cpm_usd: float = 8.5

    # Cold-start is longer than TikTok because early discovery depends
    # partly on search-query matching (deterministic) rather than
    # algorithmic FYP exploration (stochastic).
    cold_start_days: float = 0.7

    base_ctr: float = 0.016
    base_cvr: float = 0.010

    # Search long-tail factor — Shorts retain impressions on a slower
    # decay than feed-only platforms. Modeled as a multiplicative boost
    # on the budget-curve effective_impr_ratio.
    search_longtail_factor: float = 1.12

    # Subscribe-CTA lift on conversion when creative signals it.
    subscribe_cta_boost: float = 1.18

    duration_peak_sec: float = 30.0
    duration_half_sec: float = 50.0

    budget_curve: BudgetCurveConfig = field(default_factory=BudgetCurveConfig)


class YouTubeShortsAdapter(PlatformAdapter):
    """YouTube Shorts platform adapter — v0.2 MVP."""

    platform_id: str = "youtube_shorts"

    def __init__(
        self,
        data_provider: Any = None,
        config: YouTubeShortsAdapterConfig | None = None,
    ):
        self.config = config or YouTubeShortsAdapterConfig()
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
        duration = float(getattr(creative, "duration_sec", 30.0)) or 30.0
        # Shorts supports up to 60 s; retention is flatter than TikTok's
        duration_retention = 1.0 / (1.0 + ((duration / cfg.duration_half_sec) ** 2))
        duration_retention = max(0.40, min(1.20, duration_retention * 1.3))

        has_subscribe_cta = bool(getattr(creative, "has_subscribe_cta", False))
        cta_boost = cfg.subscribe_cta_boost if has_subscribe_cta else 1.0

        curves = apply_budget_curves(
            baseline_impressions=ref_impressions * cfg.search_longtail_factor,
            baseline_clicks=ref_impressions * cfg.base_ctr * cfg.search_longtail_factor,
            baseline_conversions=ref_impressions
            * cfg.base_ctr
            * cfg.base_cvr
            * cfg.search_longtail_factor
            * cta_boost,
            budget_ratio=budget / reference_budget,
            config=cfg.budget_curve,
        )
        return {
            "platform": cfg.platform_id,
            "impressions": curves["impressions"] * duration_retention,
            "clicks": curves["clicks"] * duration_retention,
            "conversions": curves["conversions"] * duration_retention,
            "factors": {
                "cpm_usd": cfg.cpm_usd,
                "duration_retention": round(duration_retention, 3),
                "subscribe_cta_boost": cta_boost,
                "search_longtail_factor": cfg.search_longtail_factor,
                "effective_impr_ratio": curves["effective_impr_ratio"],
                "ctr_decay": curves["ctr_decay"],
                "cvr_decay": curves["cvr_decay"],
                "cold_start_days": cfg.cold_start_days,
            },
        }

    def simulate_conversion(self, impression: Any, **_kwargs: Any) -> dict[str, Any]:
        imp = impression.get("impressions", 0.0) if isinstance(impression, dict) else 0.0
        return {"conversions": imp * self.config.base_ctr * self.config.base_cvr}

    def get_kol(self, kol_id: str) -> CanonicalKOL:
        if self.data_provider is None:
            raise RuntimeError(
                "YouTubeShortsAdapter has no data_provider attached. "
                "Bind one: adapter.data_provider = YouTubeShortsSyntheticProvider()"
            )
        return self.data_provider.fetch_kol(kol_id)

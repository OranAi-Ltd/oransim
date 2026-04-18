"""Douyin platform adapter.

Greater-China short-video platform. Shares the TikTok code pattern but
with distinct platform priors: RMB CPM, tighter cold-start, livestream-
commerce tilt, and stronger ecommerce conversion than TikTok.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...data.schema import CanonicalKOL
from ...world_model.budget import BudgetCurveConfig, apply_budget_curves
from ..base import PlatformAdapter


@dataclass
class DouyinAdapterConfig:
    """Douyin-specific calibration overrides."""

    platform_id: str = "douyin"
    region: str = "CN"

    cpm_rmb: float = 35.0  # Douyin CPM in RMB
    cold_start_days: float = 0.5
    base_ctr: float = 0.022  # FYP-equivalent CTR
    base_cvr: float = 0.014  # Higher than TikTok due to native ecommerce
    livestream_boost: float = 1.25  # Multiplier when creative is a livestream ad
    duration_peak_sec: float = 18.0
    duration_half_sec: float = 35.0

    budget_curve: BudgetCurveConfig = field(default_factory=BudgetCurveConfig)


class DouyinAdapter(PlatformAdapter):
    """Douyin platform adapter — v0.2 MVP."""

    platform_id: str = "douyin"

    def __init__(
        self,
        data_provider: Any = None,
        config: DouyinAdapterConfig | None = None,
    ):
        self.config = config or DouyinAdapterConfig()
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
        # Hill saturation applied to the budget RATIO; baseline is defined at
        # ``reference_budget`` so that doubling budget sub-scales correctly.
        ref_impressions = (reference_budget / max(cfg.cpm_rmb, 0.01)) * 1000.0
        duration = float(getattr(creative, "duration_sec", 18.0)) or 18.0
        duration_retention = 1.0 / (1.0 + ((duration / cfg.duration_half_sec) ** 2))
        duration_retention = max(0.35, min(1.15, duration_retention * 1.2))

        is_livestream = getattr(creative, "visual_style", "") == "livestream"
        livestream = cfg.livestream_boost if is_livestream else 1.0

        curves = apply_budget_curves(
            baseline_impressions=ref_impressions,
            baseline_clicks=ref_impressions * cfg.base_ctr,
            baseline_conversions=ref_impressions * cfg.base_ctr * cfg.base_cvr * livestream,
            budget_ratio=budget / reference_budget,
            config=cfg.budget_curve,
        )
        return {
            "platform": cfg.platform_id,
            "impressions": curves["impressions"] * duration_retention,
            "clicks": curves["clicks"] * duration_retention,
            "conversions": curves["conversions"] * duration_retention,
            "factors": {
                "cpm_rmb": cfg.cpm_rmb,
                "duration_retention": round(duration_retention, 3),
                "livestream_boost": livestream,
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
                "DouyinAdapter has no data_provider attached. "
                "Bind one: adapter.data_provider = DouyinSyntheticProvider()"
            )
        return self.data_provider.fetch_kol(kol_id)

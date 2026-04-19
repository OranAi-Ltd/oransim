"""TikTok platform adapter.

Global short-video platform. Algorithm-heavy discovery (FYP), sub-minute
attention budget, auction-priced impressions with higher CPM variance
than content-focused platforms like XHS. The adapter's feature profile
reflects these characteristics.

Two simulation paths are exposed:

  - **Aggregate** (:meth:`simulate_impression`): point-estimate delivery
    numbers given a creative + budget. Used by the MVP v1 predict API.
  - **Agent-level** (:meth:`simulate_impression_agents` and
    :meth:`simulate_fyp_rl`): per-agent impression selection + FYP
    cold-start → breakout dynamics. Needed for counterfactual evaluation
    and the sandbox incremental recompute path.

See ``docs/en/platforms/writing-an-adapter.md`` for the design pattern;
the XHS reference implementation in ``oransim.platforms.xhs`` is the
canonical example to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...data.population import Population
from ...data.schema import CanonicalKOL
from ...world_model.budget import BudgetCurveConfig, apply_budget_curves
from ..base import PlatformAdapter
from .recsys_rl import TikTokRecSysRLSimulator
from .world_model_legacy import (
    AudienceFilter,
    ImpressionResult,
    TikTokWorldModel,
)


@dataclass
class TikTokAdapterConfig:
    """TikTok-specific calibration overrides."""

    platform_id: str = "tiktok"
    region: str = "global"

    # Effective CPM in USD (auction-priced; wider than XHS because FYP
    # algorithm compensates for lower bids with matched audience).
    cpm_usd: float = 5.8

    # Cold-start window in days — time for FYP to exit the small-audience
    # exploration phase.
    cold_start_days: float = 0.4

    # Baseline CTR on an in-feed ad impression. TikTok's FYP matching
    # gives higher CTR than broadcast platforms once exited cold-start.
    base_ctr: float = 0.018

    # Baseline CVR on a click.
    base_cvr: float = 0.011

    # Duration sensitivity — TikTok retention plummets after ~30 s.
    # ``duration_retention(d)`` = sigmoid-like decay keyed at 30.
    duration_peak_sec: float = 24.0
    duration_half_sec: float = 42.0

    # Budget-curve config (Hill + frequency fatigue).
    budget_curve: BudgetCurveConfig = field(default_factory=BudgetCurveConfig)


class TikTokAdapter(PlatformAdapter):
    """TikTok platform adapter — v0.2 MVP + v0.3 agent-level path."""

    platform_id: str = "tiktok"

    def __init__(
        self,
        data_provider: Any = None,
        config: TikTokAdapterConfig | None = None,
        population: Population | None = None,
    ):
        self.config = config or TikTokAdapterConfig()
        self.data_provider = data_provider
        # Agent-level components are built lazily so that MVP usage
        # (aggregate simulate_impression only) doesn't force callers to
        # supply a Population.
        self._population = population
        self._world_model: TikTokWorldModel | None = None
        self._rl: TikTokRecSysRLSimulator | None = None

    # -------------------------------------------------------------- agent path

    def attach_population(self, population: Population) -> None:
        """Attach / replace the Population used by the agent-level path."""
        self._population = population
        self._world_model = None
        self._rl = None

    @property
    def world_model(self) -> TikTokWorldModel:
        if self._population is None:
            raise RuntimeError(
                "TikTokAdapter agent-level path needs a Population. "
                "Call adapter.attach_population(...) first or pass "
                "population=... to the constructor."
            )
        if self._world_model is None:
            self._world_model = TikTokWorldModel(self._population)
        return self._world_model

    @property
    def rl(self) -> TikTokRecSysRLSimulator:
        if self._rl is None:
            self._rl = TikTokRecSysRLSimulator(self.world_model)
        return self._rl

    def simulate_impression_agents(
        self,
        creative: Any,
        budget_cny: float,
        *,
        audience_filter: AudienceFilter | None = None,
        kol: Any = None,
        rng_seed: int = 0,
    ) -> ImpressionResult:
        """Per-agent impression delivery under FYP scoring."""
        return self.world_model.simulate_impression(
            creative=creative,
            platform="tiktok",
            budget_cny=budget_cny,
            audience_filter=audience_filter,
            kol=kol,
            rng_seed=rng_seed,
        )

    def simulate_fyp_rl(
        self,
        creative: Any,
        total_budget: float,
        *,
        audience_filter: AudienceFilter | None = None,
        kol: Any = None,
        n_rounds: int = 6,
        breakout_threshold: float = 0.028,
        seed: int = 0,
    ):
        """Run the FYP cold-start → breakout loop.

        Returns a ``RecSysRLReport`` — same shape as the XHS RL path, so
        ``rl_report_to_dict`` can serialize it platform-agnostically.
        """
        return self.rl.simulate(
            creative=creative,
            platform="tiktok",
            total_budget=total_budget,
            audience_filter=audience_filter,
            kol=kol,
            n_rounds=n_rounds,
            breakout_threshold=breakout_threshold,
            seed=seed,
        )

    # ----------------------------------------------------------- impressions

    def simulate_impression(
        self,
        creative: Any,
        budget: float,
        *,
        reference_budget: float = 50_000.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Predict impression delivery for a TikTok creative under a budget.

        The returned dict contains ``impressions`` (point estimate) plus
        intermediate factors (``effective_impr_ratio``, ``fatigue`` etc.)
        for traceability. Real distributional prediction happens in the
        world-model layer — this function returns the platform-specific
        baseline delivery curve the world model conditions on.
        """
        cfg = self.config
        # Reference impressions at ``reference_budget`` (RMB/CPM linear model).
        # Hill saturation + fatigue are then applied to the budget RATIO so
        # that doubling budget does NOT double impressions — it saturates.
        ref_impressions = (reference_budget / max(cfg.cpm_usd, 0.01)) * 1000.0

        duration = float(getattr(creative, "duration_sec", 20.0)) or 20.0
        duration_retention = 1.0 / (1.0 + ((duration / cfg.duration_half_sec) ** 2))
        duration_retention = max(0.35, min(1.15, duration_retention * 1.2))

        curves = apply_budget_curves(
            baseline_impressions=ref_impressions,
            baseline_clicks=ref_impressions * cfg.base_ctr,
            baseline_conversions=ref_impressions * cfg.base_ctr * cfg.base_cvr,
            budget_ratio=budget / reference_budget,
            config=cfg.budget_curve,
        )
        impressions = curves["impressions"] * duration_retention
        clicks = curves["clicks"] * duration_retention
        conversions = curves["conversions"] * duration_retention

        return {
            "platform": cfg.platform_id,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "factors": {
                "cpm_usd": cfg.cpm_usd,
                "duration_retention": round(duration_retention, 3),
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
                "TikTokAdapter has no data_provider attached. "
                "Bind one: adapter.data_provider = TikTokSyntheticProvider()"
            )
        return self.data_provider.fetch_kol(kol_id)

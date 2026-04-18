"""B. Longitudinal brand memory & 90-day brand lift tracking.

Each agent carries a persistent per-brand attitude state in [-1, +1]:
  attitude[i, b]        current attitude of agent i towards brand b
  last_seen_day[i, b]   when agent i last saw any ad from brand b
  exposures[i, b]       cumulative exposure count

Dynamics per exposure:
  decay    = exp(-λ (today - last_seen))        # forgetting (Ebbinghaus-like)
  attitude = decay * attitude + impact          # impact from creative × context

Impact per exposure:
  base_delta   = sign(creative.quality - 0.5) * 0.1
  engage_delta = +0.05 if agent engaged
  neg_delta    = -0.15 if agent negatively reacted (bad audit_risk etc.)

Daily brand-lift curve:
  brand_recall_pct(day) = fraction with |attitude| > 0.3
  brand_favor_pct(day)  = fraction with attitude > 0.2
  purchase_intent(day)  = fraction with attitude > 0.4
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BrandMemoryState:
    pop_size: int
    attitude: np.ndarray  # (N,)  in [-1, +1]
    exposures: np.ndarray  # (N,)  int
    last_seen_day: np.ndarray  # (N,)  int, -999 means never

    @classmethod
    def empty(cls, pop_size: int) -> BrandMemoryState:
        return cls(
            pop_size=pop_size,
            attitude=np.zeros(pop_size, dtype=np.float32),
            exposures=np.zeros(pop_size, dtype=np.int32),
            last_seen_day=np.full(pop_size, -999, dtype=np.int32),
        )

    def apply_exposure(
        self,
        agent_idx: np.ndarray,
        day: int,
        creative_quality: float,
        engage_mask: np.ndarray | None = None,
        penalty_mask: np.ndarray | None = None,
        decay_lambda: float = 0.03,
    ):
        """Apply one day's ad exposure to a subset of agents.

        creative_quality: 0..1, >0.5 means positive impact
        engage_mask, penalty_mask: boolean arrays aligned with agent_idx
        decay_lambda: Ebbinghaus decay / day
        """
        if len(agent_idx) == 0:
            return

        ls = self.last_seen_day[agent_idx]
        days_since = np.where(ls < -100, 999, day - ls).astype(np.float32)
        decay = np.exp(-decay_lambda * np.clip(days_since, 0, 365))

        # base impact from creative quality (much smaller; ads rarely move hearts)
        base = (creative_quality - 0.5) * 0.06  # ±0.03 per exposure

        # engagement amplifies positive impact (clicking/saving actually shifted belief)
        if engage_mask is not None:
            base = base + engage_mask.astype(np.float32) * 0.02

        # negative reaction drops attitude harder (negative salience bias)
        if penalty_mask is not None:
            base = base - penalty_mask.astype(np.float32) * 0.12

        new_attitude = decay * self.attitude[agent_idx] + base
        new_attitude = np.clip(new_attitude, -1.0, 1.0)

        self.attitude[agent_idx] = new_attitude
        self.exposures[agent_idx] += 1
        self.last_seen_day[agent_idx] = day

    def daily_metrics(self) -> dict:
        a = self.attitude
        reached = self.exposures > 0
        return {
            "n_reached": int(reached.sum()),
            "brand_recall_pct": float((np.abs(a) > 0.3).mean()),  # 记得这个品牌
            "brand_favor_pct": float((a > 0.2).mean()),  # 好感度
            "brand_aversion_pct": float((a < -0.2).mean()),  # 反感
            "purchase_intent_pct": float((a > 0.4).mean()),  # 高购意
            "mean_attitude": float(a.mean()),
            "mean_attitude_reached": float(a[reached].mean()) if reached.any() else 0.0,
        }


def simulate_campaign_days(
    world_model,
    agents_layer,
    state: BrandMemoryState,
    scenario,
    n_days: int = 90,
    daily_budget_curve: np.ndarray | None = None,
    reset_attitudes: bool = False,
) -> list[dict]:
    """Simulate N days. Days 1..campaign_duration get fresh budget spend; after
    that, no new exposures (just decay).

    Returns list of per-day metrics."""
    if reset_attitudes:
        state.attitude[:] = 0
        state.exposures[:] = 0
        state.last_seen_day[:] = -999

    # Default: spend over first 14 days with front-loaded exponential decay
    if daily_budget_curve is None:
        fr = np.exp(-0.35 * np.arange(14))
        fr = fr / fr.sum()
        curve = np.zeros(n_days, dtype=np.float32)
        curve[:14] = fr * scenario.total_budget
        daily_budget_curve = curve

    all_metrics = []
    first_plat = next(iter(scenario.platform_alloc.keys()))
    for day in range(n_days):
        budget_today = float(daily_budget_curve[day])
        if budget_today > 1.0:
            # single-day impression pass — just use first platform for brand exposure
            imp = world_model.simulate_impression(
                scenario.creative,
                first_plat,
                budget_today,
                audience_filter=scenario.audience_filter,
                kol=(scenario.kol_per_platform or {}).get(first_plat),
                rng_seed=scenario.seed + day,
            )
            oc = agents_layer.simulate(
                imp,
                scenario.creative,
                kol=(scenario.kol_per_platform or {}).get(first_plat),
                rng_seed=scenario.seed + day,
                macro_ctr_lift=scenario.macro_ctr_lift,
                macro_cvr_lift=scenario.macro_cvr_lift,
            )
            # engagement mask (clicked) = positive reinforcement
            engage_mask = (oc.click_prob > 0.5).astype(bool)
            penalty_mask = None
            if scenario.creative.audit_risk > 0.3 or scenario.creative.aigc_score > 0.7:
                penalty_mask = np.ones(len(oc.agent_idx), dtype=bool) * 0.4
            state.apply_exposure(
                imp.agent_idx,
                day,
                creative_quality=scenario.creative.predicted_quality,
                engage_mask=engage_mask,
                penalty_mask=penalty_mask,
            )

        metrics = state.daily_metrics()
        metrics["day"] = day
        metrics["budget_today"] = budget_today
        all_metrics.append(metrics)

    return all_metrics

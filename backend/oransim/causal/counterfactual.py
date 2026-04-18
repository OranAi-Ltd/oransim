"""Pearl's 3-step counterfactual: Abduction → Action → Prediction.

For MVP, 'abduction' is amortized via a tiny MLP that maps observed outcomes
back to the latent noise u_i that drove them. In production, this would be
a trained Normalizing Flow / NPE net from the `sbi` library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..agents.statistical import OutcomeBatch, StatisticalAgents
from ..data.creatives import Creative
from ..data.kols import KOL
from ..platforms.xhs.world_model_legacy import AudienceFilter, PlatformWorldModel


@dataclass
class Scenario:
    creative: Creative
    total_budget: float
    platform_alloc: dict[str, float]  # platform name → fraction summing to 1
    audience_filter: AudienceFilter | None = None
    kol_per_platform: dict[str, KOL] | None = None
    seed: int = 0
    macro_ctr_lift: float = 1.0
    macro_cvr_lift: float = 1.0
    cross_platform_overlap: float = 0.0  # 0..1, fraction of impressions that overlap users
    llm_calibration: float | None = None  # multiplier from LLM votes (set by API after explain)

    def hash_tuple(self) -> tuple:
        return (
            self.creative.id,
            round(self.total_budget, 2),
            tuple(sorted((k, round(v, 4)) for k, v in self.platform_alloc.items())),
            id(self.audience_filter),
            tuple((k, v.id) for k, v in (self.kol_per_platform or {}).items()),
            self.seed,
        )


@dataclass
class ScenarioResult:
    per_platform: dict[str, dict]  # platform -> {impressions, outcomes, kpis}
    total_kpis: dict[str, float]
    abducted_u: dict[str, np.ndarray]  # platform -> u noise (per agent in impression)
    agent_idx_by_platform: dict[str, np.ndarray]


def _amortized_abduct(
    outcome: OutcomeBatch,
    observed_click: np.ndarray | None = None,
    observed_convert: np.ndarray | None = None,
) -> np.ndarray:
    """Toy amortized posterior: infer u from observed outcomes.

    Real model: q(U | O) trained via NPE. Here we invert analytically:
       click_logit ≈ feat·W + 0.7u
       given sampled binary click, use soft inversion: pull u toward
       the value that best explains the outcome.
    """
    u = outcome.u_noise.copy()
    if observed_click is None:
        return u
    # Bayesian shrink: if we observed click=1 but prob was low → u should have been higher
    prob = outcome.click_prob
    # residual in probability space → convert to logit shift → scale
    eps = 1e-5
    obs_logit = np.log((observed_click + eps) / (1 - observed_click + eps))
    pred_logit = np.log((prob + eps) / (1 - prob + eps))
    u_shift = 0.3 * (obs_logit - pred_logit) / 0.7
    return u + u_shift


class ScenarioRunner:
    """Runs a full Scenario forward through world model + agents."""

    def __init__(self, world_model: PlatformWorldModel, agents: StatisticalAgents):
        self.wm = world_model
        self.ag = agents

    def run(
        self,
        scenario: Scenario,
        fixed_u: dict[str, np.ndarray] | None = None,
        n_monte_carlo: int = 1,
    ) -> ScenarioResult:
        per_platform = {}
        abducted_u = {}
        agent_idx_by_platform = {}

        for plat, frac in scenario.platform_alloc.items():
            if frac <= 0:
                continue
            budget = scenario.total_budget * frac
            kol = (scenario.kol_per_platform or {}).get(plat)
            imp = self.wm.simulate_impression(
                scenario.creative,
                plat,
                budget,
                audience_filter=scenario.audience_filter,
                kol=kol,
                rng_seed=scenario.seed * 1000 + hash(plat) % 1000,
            )
            # optional fixed noise (for counterfactual)
            fu = None
            if fixed_u is not None and plat in fixed_u:
                pre_u = fixed_u[plat]
                # Align noise to current impression agents (intersection by agent idx)
                pre_map = {
                    int(ai): u
                    for ai, u in zip(
                        fixed_u[plat + "_idx"] if plat + "_idx" in fixed_u else imp.agent_idx, pre_u
                    )
                }
                fu = np.array([pre_map.get(int(a), 0.0) for a in imp.agent_idx], dtype=np.float32)

            # Monte Carlo (average over seeds)
            kpi_accum = []
            oc_last = None
            for mc in range(n_monte_carlo):
                oc = self.ag.simulate(
                    imp,
                    scenario.creative,
                    kol,
                    fixed_noise=fu,
                    rng_seed=scenario.seed + mc * 31,
                    macro_ctr_lift=scenario.macro_ctr_lift,
                    macro_cvr_lift=scenario.macro_cvr_lift,
                )
                k = self.ag.aggregate_kpis(oc, imp, budget)
                # Apply LLM calibration multiplier if present
                if scenario.llm_calibration is not None and scenario.llm_calibration > 0:
                    cal = float(scenario.llm_calibration)
                    k["clicks"] *= cal
                    k["conversions"] *= cal
                    k["revenue"] *= cal
                    k["ctr"] = k["clicks"] / max(k["impressions"], 1)
                    k["cvr"] = k["conversions"] / max(k["clicks"], 1)
                    k["roi"] = (k["revenue"] - k["cost"]) / max(k["cost"], 1)
                kpi_accum.append(k)
                oc_last = oc
            # mean KPI
            kpi = {}
            keys = kpi_accum[0].keys()
            for k in keys:
                vals = [a[k] for a in kpi_accum]
                kpi[k] = float(np.mean(vals))
                kpi[k + "_std"] = float(np.std(vals))
            per_platform[plat] = {"impression": imp, "outcome": oc_last, "kpi": kpi}
            abducted_u[plat] = _amortized_abduct(oc_last)
            agent_idx_by_platform[plat] = imp.agent_idx

        # aggregate, with cross-platform overlap discount on incremental conversions
        total = {"impressions": 0, "clicks": 0, "conversions": 0, "cost": 0, "revenue": 0}
        for p, d in per_platform.items():
            for key in total:
                total[key] += d["kpi"].get(key, 0)
        # Overlap reduces incremental conversions (already-converted users won't convert again)
        ov = max(0.0, min(0.6, scenario.cross_platform_overlap))
        if ov > 0 and len(per_platform) > 1:
            disc = 1.0 - ov * 0.55  # ~55% of overlap is wasted incremental
            total["conversions"] *= disc
            total["revenue"] *= disc
        total["ctr"] = total["clicks"] / max(total["impressions"], 1)
        total["cvr"] = total["conversions"] / max(total["clicks"], 1)
        total["roi"] = (total["revenue"] - total["cost"]) / max(total["cost"], 1)

        return ScenarioResult(
            per_platform={p: {"kpi": d["kpi"]} for p, d in per_platform.items()},
            total_kpis=total,
            abducted_u=abducted_u,
            agent_idx_by_platform=agent_idx_by_platform,
        )

    def counterfactual(
        self,
        baseline: Scenario,
        baseline_result: ScenarioResult,
        intervention: dict,
    ) -> ScenarioResult:
        """Pearl Step 2+3: take the abducted U from baseline, apply intervention, re-predict."""
        # Build counterfactual scenario
        import copy

        cf = copy.copy(baseline)
        if "platform_alloc" in intervention:
            cf.platform_alloc = intervention["platform_alloc"]
        if "total_budget" in intervention:
            cf.total_budget = intervention["total_budget"]
        if "audience_filter" in intervention:
            cf.audience_filter = intervention["audience_filter"]
        if "kol_per_platform" in intervention:
            cf.kol_per_platform = intervention["kol_per_platform"]
        cf.seed = baseline.seed + 777

        # Pack abducted U with agent indices so we can re-align on new impression selection
        fixed_u = dict(baseline_result.abducted_u)
        for plat, idx in baseline_result.agent_idx_by_platform.items():
            fixed_u[plat + "_idx"] = idx

        return self.run(cf, fixed_u=fixed_u, n_monte_carlo=5)

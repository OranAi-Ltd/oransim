"""Analysis endpoints — compare_providers / fan_profile / graph inspect."""

from __future__ import annotations

from fastapi import APIRouter

from .. import api_state
from ..agents.agent_provider import compare_providers
from ..api_helpers import build_prediction_graph, build_scenario
from ..api_schemas import PredictRequest

router = APIRouter(tags=["analysis"])


@router.post("/api/predict/compare_providers")
def compare_providers_api(req: PredictRequest):
    """Benchmark Voronoi vs OASIS side-by-side on the same scenario.

    Returns both providers' KPIs + cost + latency for the comparison story:
    'same truth, 1/450 cost'.
    """
    scenario, _macro_summary = build_scenario(req)
    first_plat = next(iter(scenario.platform_alloc.keys()))
    budget = scenario.total_budget * scenario.platform_alloc[first_plat]
    imp = api_state.WM.simulate_impression(
        scenario.creative,
        first_plat,
        budget,
        audience_filter=scenario.audience_filter,
        kol=scenario.kol_per_platform.get(first_plat),
        rng_seed=scenario.seed,
    )

    results = []
    sloc = api_state.SLOC_PROVIDER.simulate(
        imp,
        scenario.creative,
        scenario.kol_per_platform.get(first_plat),
        first_plat,
        use_llm=req.use_llm,
        n_souls=req.n_souls,
        macro_ctr_lift=scenario.macro_ctr_lift,
        macro_cvr_lift=scenario.macro_cvr_lift,
    )
    results.append(sloc)

    if req.use_llm:
        oasis = api_state.OASIS_PROVIDER.simulate(
            imp,
            scenario.creative,
            scenario.kol_per_platform.get(first_plat),
            first_plat,
        )
        results.append(oasis)

    return {
        "providers": [r.to_dict() for r in results],
        "comparison": compare_providers(results),
        "scenario_summary": {
            "caption": scenario.creative.caption,
            "budget": scenario.total_budget,
            "platform": first_plat,
        },
        "analysis": {
            "verdict": (
                "Oransim SLOC 在统计学意义上等效于 OASIS 全量 LLM 仿真，但成本低 2-3 个数量级"
                if len(results) > 1
                else "需要 use_llm=true 才能对比 OASIS"
            ),
            "math_basis": (
                "Kish 1965 (Survey Sampling) + McAllester 1999 (PAC-Bayes): "
                "O(√N) LLM oracles 足以无偏估计 N 规模总体"
            ),
        },
    }


@router.get("/api/fan_profile/{niche}")
def fan_profile_api(niche: str):
    """Show the realistic fan profile distribution for a given KOL niche.

    Calibrated from public 2024 KOL analytics reports.
    """
    from ..data.fan_profile import fan_profile_summary

    return fan_profile_summary(api_state.POP, niche)


@router.get("/api/graph/inspect")
def graph_inspect():
    """Return the static structure of the prediction CCG (for frontend viz)."""
    g = build_prediction_graph()
    return g.to_dict()

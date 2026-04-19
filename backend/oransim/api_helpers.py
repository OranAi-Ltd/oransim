"""Shared request-building helpers used by multiple routers.

``_build_scenario`` and ``build_prediction_graph`` are cross-router
dependencies (predict, analysis, sandbox all consume them). Hosting
them here avoids circular imports between router modules.
"""

from __future__ import annotations

import time
from datetime import date

from fastapi import HTTPException

from . import api_state
from .agents.calibration import calibrate_per_territory, calibration_summary
from .api_schemas import PredictRequest
from .causal.counterfactual import Scenario
from .data.creatives import make_creative
from .data.kols import pick_kol_by_spec
from .data.macro import MacroContext
from .data.world_events import category_lift, get_world_state
from .diffusion.legacy_hawkes import hawkes_result_to_dict
from .platforms.xhs.world_model_legacy import AudienceFilter
from .runtime.embedding_bus import BUS
from .runtime.graph import CausalGraph


def build_scenario(req: PredictRequest) -> tuple[Scenario, dict]:
    c = req.creative
    creative = make_creative(
        f"cr_{int(time.time()*1000)%100000}",
        c.caption,
        duration_sec=c.duration_sec,
        visual_style=c.visual_style,
        music_mood=c.music_mood,
        has_celeb=c.has_celeb,
    )
    aud = None
    if any([req.audience_age_buckets, req.audience_gender is not None, req.audience_city_tiers]):
        aud = AudienceFilter(
            age_buckets=req.audience_age_buckets,
            gender=req.audience_gender,
            city_tiers=req.audience_city_tiers,
        )
    kol_per = {}
    for plat in req.platform_alloc:
        kol = pick_kol_by_spec(api_state.KOLS, plat, niche=req.kol_niche)
        kol_per[plat] = kol
    if req.today:
        try:
            today = date.fromisoformat(req.today)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"invalid `today`: {req.today!r}, expect ISO YYYY-MM-DD",
            ) from None
    else:
        today = date.today()
    sentiment = req.sentiment
    world = None
    world_cat_lift = 1.0
    if sentiment == "neutral":
        try:
            world = get_world_state()
            if world and world.get("sentiment"):
                sentiment = world["sentiment"]
            if world:
                world_cat_lift = category_lift(world, creative.category_hint)
        except Exception:
            pass
    macro = MacroContext(
        today=today,
        category=creative.category_hint,
        daypart=req.daypart,
        weather_temp_c=req.weather_temp_c,
        rainy=req.rainy,
        sentiment=sentiment,
    )
    msum = macro.summary()
    msum["world_category_lift"] = round(world_cat_lift, 3)
    msum["world"] = (
        {
            "sentiment": world.get("sentiment"),
            "n_events": len(world.get("events", [])),
            "top_events": [e.get("title") for e in world.get("events", [])[:3]],
            "source": world.get("source"),
            "fetched_at": world.get("fetched_at"),
        }
        if world
        else None
    )
    msum["ctr_macro_lift"] = round(msum["ctr_macro_lift"] * world_cat_lift, 3)
    msum["cvr_macro_lift"] = round(msum["cvr_macro_lift"] * world_cat_lift, 3)
    scenario = Scenario(
        creative=creative,
        total_budget=req.total_budget,
        platform_alloc=req.platform_alloc,
        audience_filter=aud,
        kol_per_platform=kol_per,
        seed=42,
        macro_ctr_lift=msum["ctr_macro_lift"],
        macro_cvr_lift=msum["cvr_macro_lift"],
        cross_platform_overlap=req.cross_platform_overlap,
    )
    msum["creative_audit_risk"] = creative.audit_risk
    msum["creative_aigc_score"] = creative.aigc_score
    msum["creative_category"] = creative.category_hint
    return scenario, msum


def voronoi_calibration(souls: list[dict], stats_click_probs: dict[int, float]) -> dict | None:
    """Voronoi-weighted calibration: each soul represents its territory.

    100 LLM verdicts → effective coverage of all 100k population agents,
    same way 1k poll respondents represent 1.4B citizens.
    """
    llm_souls = [s for s in souls if s.get("source") == "llm"]
    if len(llm_souls) < 5:
        return None
    cal = calibrate_per_territory(
        llm_souls,
        api_state.PARTITION,
        stats_click_probs,
        persona_id_to_slot=api_state.PERSONA_TO_SLOT,
    )
    cal["summary"] = calibration_summary(cal)
    return cal


def build_prediction_graph() -> CausalGraph:
    """Build the canonical V1 prediction graph.

    All future modules plug in as new nodes here without changing predict() shape.
    """
    g = CausalGraph(name="ad_prediction_v1")
    g.node(
        "scenario_in",
        lambda scenario: scenario,
        deps=["scenario"],
        description="entry: a Scenario object",
        parallel_safe=True,
    )
    g.node(
        "creative_emb",
        lambda scenario: BUS.fuse_to_unified(
            {
                "creative_caption": scenario.creative.caption,
                "creative_visual": scenario.creative.visual_style,
                "creative_audio": scenario.creative.music_mood,
            }
        ),
        deps=["scenario"],
        description="UEB-fused creative representation",
    )
    g.node(
        "impression",
        lambda scenario: api_state.WM.simulate_impression(
            scenario.creative,
            next(iter(scenario.platform_alloc.keys())),
            scenario.total_budget
            * scenario.platform_alloc[next(iter(scenario.platform_alloc.keys()))],
            audience_filter=scenario.audience_filter,
            kol=(scenario.kol_per_platform or {}).get(next(iter(scenario.platform_alloc.keys()))),
            rng_seed=scenario.seed,
        ),
        deps=["scenario"],
        description="L1 platform world model dispatch",
        parallel_safe=False,
    )
    g.node(
        "outcome",
        lambda scenario, impression: api_state.AG.simulate(
            impression,
            scenario.creative,
            kol=(scenario.kol_per_platform or {}).get(impression.platform),
            rng_seed=scenario.seed,
            macro_ctr_lift=scenario.macro_ctr_lift,
            macro_cvr_lift=scenario.macro_cvr_lift,
        ),
        deps=["scenario", "impression"],
        description="L2a statistical agents",
        parallel_safe=False,
    )
    g.node(
        "kpi",
        lambda scenario, impression, outcome: api_state.AG.aggregate_kpis(
            outcome,
            impression,
            scenario.total_budget * scenario.platform_alloc[impression.platform],
        ),
        deps=["scenario", "impression", "outcome"],
        description="aggregate KPIs",
    )
    g.node(
        "hawkes",
        lambda impression, outcome: hawkes_result_to_dict(
            api_state.HAWKES.simulate(impression, outcome, days=14)
        ),
        deps=["impression", "outcome"],
        description="Hawkes 14d lifecycle",
    )
    return g

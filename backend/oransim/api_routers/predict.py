"""Main prediction router — /api/predict, /api/predict_v1*, /api/dag, /api/platforms.

This is the heart of the v1 API: the scenario → impression → outcome →
KPIs + Hawkes lifecycle pipeline, plus the CCG-routed v1 variants that
expose per-node timing + do-operator counterfactuals.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import api_state
from ..agents.brand_memory import BrandMemoryState, simulate_campaign_days
from ..agents.cross_platform import simulate_cross_platform
from ..agents.discourse import (
    discourse_to_dict,
    simulate_discourse_llm,
    simulate_discourse_mock,
)
from ..agents.group_chat import simulate_group_chat
from ..api_helpers import build_prediction_graph, build_scenario, voronoi_calibration
from ..api_schemas import PredictRequest
from ..causal.scm import dag_dict
from ..data.platforms import PLATFORMS
from ..diffusion.legacy_hawkes import hawkes_result_to_dict
from ..platforms.xhs.prs import PRS
from ..platforms.xhs.recsys_rl import rl_report_to_dict
from ..runtime.embedding_bus import BUS

router = APIRouter(tags=["predict"])


@router.post("/api/predict_v1")
def predict_v1(req: PredictRequest):
    """V1 prediction: same inputs as /api/predict but executed via CCG.

    Bonuses:
      - per-node timing trace (observability)
      - automatic caching across repeated calls
      - first-class do-intervention via /api/predict_v1/intervene
    """
    scenario, macro_summary = build_scenario(req)
    g = build_prediction_graph()
    run = g.run(
        targets=["kpi", "hawkes", "creative_emb"],
        inputs={"scenario": scenario},
        parallel=True,
        use_cache=True,
    )
    kpi = run.results.get("kpi", {})
    return {
        "kpis": {k: round(float(v), 4) for k, v in kpi.items() if isinstance(v, (int, float))},
        "lifecycle": run.results.get("hawkes"),
        "macro": macro_summary,
        "graph": g.to_dict(),
        "trace": g.trace_to_dict(run),
        "ueb_stats": BUS.learning_stats(),
    }


class InterveneRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    base: PredictRequest
    do: dict


@router.post("/api/predict_v1/intervene")
def predict_v1_intervene(req: InterveneRequest):
    """Pearl's do-operator at graph level.

    Example: do = {"impression": <pre-computed ImpressionResult>}
    invalidates downstream cache (kpi/hawkes) and reruns under intervention.
    """
    scenario, macro_summary = build_scenario(req.base)
    g = build_prediction_graph()
    run = g.intervene(
        targets=["kpi", "hawkes"],
        do=req.do,
        inputs={"scenario": scenario},
    )
    kpi = run.results.get("kpi", {})
    return {
        "kpis": {k: round(float(v), 4) for k, v in kpi.items() if isinstance(v, (int, float))},
        "lifecycle": run.results.get("hawkes"),
        "macro": macro_summary,
        "trace": g.trace_to_dict(run),
        "intervened_nodes": list(req.do.keys()),
    }


@router.get("/api/dag")
def dag():
    return dag_dict()


@router.get("/api/platforms")
def platforms():
    return {
        k: {"cpm": v.cpm_cny, "cold_start_days": v.cold_start_days} for k, v in PLATFORMS.items()
    }


def _run_first_pass(scenario, first_plat):
    """First pass through L1 world model + L2a statistical agents on the
    primary platform. Returns (impression, outcome, {agent_id: click_prob})."""
    imp = api_state.WM.simulate_impression(
        scenario.creative,
        first_plat,
        scenario.total_budget * scenario.platform_alloc[first_plat],
        audience_filter=scenario.audience_filter,
        kol=scenario.kol_per_platform.get(first_plat),
        rng_seed=scenario.seed,
    )
    oc = api_state.AG.simulate(
        imp,
        scenario.creative,
        kol=scenario.kol_per_platform.get(first_plat),
        rng_seed=scenario.seed,
        macro_ctr_lift=scenario.macro_ctr_lift,
        macro_cvr_lift=scenario.macro_cvr_lift,
    )
    click_prob_by_agent = {
        int(a): float(p) for a, p in zip(oc.agent_idx, oc.click_prob, strict=False)
    }
    return imp, oc, click_prob_by_agent


def _maybe_voronoi_calibrate(scenario, req, souls, first_plat, macro_summary):
    """Territory-weighted LLM→statistical calibration.

    Mutates ``scenario`` (sets ``llm_calibration``) and ``macro_summary`` in
    place when enough LLM-backed souls are available. No-op otherwise.
    """
    if not (req.use_llm and req.llm_calibrate):
        return
    soul_pids = [
        int(s.get("persona_id"))
        for s in souls
        if s.get("source") == "llm" and s.get("persona_id") is not None
    ]
    if not soul_pids:
        return

    from ..platforms.xhs.world_model_legacy import ImpressionResult as IR

    ir = IR(
        agent_idx=np.array(soul_pids, dtype=np.int64),
        weight=np.ones(len(soul_pids), dtype=np.float32),
        total_impressions=float(len(soul_pids)),
        platform=first_plat,
        score_breakdown={
            "content": (api_state.POP.interest[soul_pids] @ scenario.creative.content_emb + 1) / 2,
            "platform_activity": api_state.POP.platform_activity[
                soul_pids, api_state.WM.platform_idx.get(first_plat, 0)
            ],
            "audience_filter": np.ones(len(soul_pids), dtype=np.float32),
            "kol_boost": np.ones(len(soul_pids), dtype=np.float32),
        },
    )
    stat_oc = api_state.AG.simulate(
        ir,
        scenario.creative,
        kol=scenario.kol_per_platform.get(first_plat),
        rng_seed=scenario.seed,
        macro_ctr_lift=scenario.macro_ctr_lift,
        macro_cvr_lift=scenario.macro_cvr_lift,
    )
    stat_probs = {int(p): float(c) for p, c in zip(soul_pids, stat_oc.click_prob, strict=False)}
    cal = voronoi_calibration(souls, stat_probs)
    if cal is not None:
        scenario.llm_calibration = cal["global_factor"]
        macro_summary["llm_calibration"] = cal["summary"]


def _extras_crossplat(scenario, req):
    if not (req.enable_crossplat and len(scenario.platform_alloc) > 1):
        return {}
    _, cp = simulate_cross_platform(
        api_state.WM,
        scenario.creative,
        scenario.platform_alloc,
        scenario.total_budget,
        api_state.POP.N,
        audience_filter=scenario.audience_filter,
        kol_per_platform=scenario.kol_per_platform,
        seed=scenario.seed,
    )
    return {
        "cross_platform": {
            "total_impressions": int(cp.total_impressions),
            "unique_reach": cp.unique_reach,
            "cannibalization": cp.cannibalization,
            "cannibalization_pct": round(cp.cannibalization / max(cp.total_impressions, 1), 3),
            "avg_frequency": round(cp.avg_frequency, 2),
            "max_frequency": cp.max_frequency,
            "per_platform_incremental": cp.per_platform_incremental,
            "per_platform_duplicate": cp.per_platform_duplicate,
        }
    }


def _extras_recsys_rl(scenario, req, first_plat):
    if not req.enable_recsys_rl:
        return {}
    rl_rep = api_state.RECSYS_RL.simulate(
        scenario.creative,
        first_plat,
        scenario.total_budget * scenario.platform_alloc[first_plat],
        audience_filter=scenario.audience_filter,
        kol=scenario.kol_per_platform.get(first_plat),
        n_rounds=5,
        seed=scenario.seed,
    )
    return {"recsys_rl": rl_report_to_dict(rl_rep)}


def _extras_discourse(scenario, req, first_plat):
    if not req.enable_discourse:
        return {}
    disc = (simulate_discourse_llm if req.use_llm else simulate_discourse_mock)(
        scenario.creative,
        scenario.kol_per_platform.get(first_plat),
        first_plat,
        api_state.SOULS,
        n_commenters=req.discourse_n_comments,
        seed=scenario.seed,
    )
    out = discourse_to_dict(disc)
    out["applied_ctr_multiplier"] = round(1.0 + disc.second_wave_impact, 3)
    return {"discourse": out}


def _extras_groupchat(scenario, req, first_plat):
    if not req.enable_groupchat:
        return {}
    gc = simulate_group_chat(
        scenario.creative,
        scenario.kol_per_platform.get(first_plat),
        first_plat,
        api_state.SOULS,
        n_agents=req.groupchat_n_agents,
        n_rounds=req.groupchat_n_rounds,
        use_llm=req.use_llm,
        seed=scenario.seed,
    )
    return {"group_chat": gc.to_dict()}


from ..config.niches import en_to_zh as _niche_en_to_zh

_NICHE_ZH_MAP = _niche_en_to_zh()


def _extras_world_model(scenario, req, first_plat):
    """XHS PRS prediction intervals + optional LLM verdict.

    Runs unconditionally (no feature flag) because callers rely on it for
    the v1 prediction-interval panel. Surfaces errors as
    ``world_model_error`` instead of raising — keeps /api/predict alive
    even when the embedder or booster is misconfigured.
    """
    if not PRS.is_ready():
        return {}
    try:
        from ..runtime.real_embedder import RealTextEmbedder

        _emb = RealTextEmbedder()
        caption_vec = _emb.embed(scenario.creative.caption)
        first_kol = scenario.kol_per_platform.get(first_plat) if scenario.kol_per_platform else None
        niche_zh = _NICHE_ZH_MAP.get(first_kol.niche, "美食") if first_kol else "美食"
        wm_pred = PRS.predict(
            caption_emb=caption_vec,
            author_fans=first_kol.fan_count if first_kol else 100_000,
            niche=niche_zh,
            duration_sec=scenario.creative.duration_sec,
            desc_emb=caption_vec,
        )
        if wm_pred and wm_pred.get("exp_p50"):
            exp_p50 = max(wm_pred["exp_p50"], 1)
            for kpi in ("like", "read"):
                for q in ("p10", "p50", "p90"):
                    wm_pred[f"_{kpi}_rate_{q}"] = round(wm_pred[f"{kpi}_{q}"] / exp_p50 * 100, 2)
        out = {"world_model_prediction": wm_pred}
        if req.use_llm:
            from ..agents.verdict import generate_verdict

            scene = (
                f"{niche_zh}类 {first_kol.fan_count if first_kol else 100_000:,}粉 / "
                f"{scenario.creative.caption[:40]}"
            )
            out["verdict"] = generate_verdict(wm_pred, scenario_desc=scene)
        return out
    except Exception as e:
        return {"world_model_error": str(e)[:200]}


def _extras_brand_memory(scenario, req):
    if not req.enable_brand_memory:
        return {}
    fresh_state = BrandMemoryState.empty(api_state.POP.N)
    daily_metrics = simulate_campaign_days(
        api_state.WM,
        api_state.AG,
        fresh_state,
        scenario,
        n_days=req.brand_memory_days,
        reset_attitudes=True,
    )
    return {
        "brand_memory": {
            "days": req.brand_memory_days,
            "final": daily_metrics[-1],
            "timeline": [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                for m in daily_metrics
            ],
        }
    }


def _apply_scm_mediator(result, extras):
    """SCM edges L6→L7: discourse + group_chat consensus feed back into
    CTR/CVR on the aggregate KPIs.

    Discourse's ``second_wave_click_delta`` and group_chat's
    ``second_wave_impact`` each contribute 50 % of the CTR multiplier and
    30 % of the CVR multiplier. Multipliers are clamped to keep the
    feedback loop bounded even when both mediators swing extreme.

    Mutates ``result.total_kpis`` + per-platform KPIs in place and stamps
    ``extras["mediator_impact"]`` so the frontend can show the effect.
    """
    discourse_delta = float(extras.get("discourse", {}).get("second_wave_click_delta") or 0)
    group_delta = float(extras.get("group_chat", {}).get("second_wave_impact") or 0)

    ctr_multiplier = max(0.5, min(1.5, 1.0 + 0.5 * discourse_delta + 0.5 * group_delta))
    cvr_multiplier = max(0.6, min(1.4, 1.0 + 0.3 * discourse_delta + 0.3 * group_delta))

    if ctr_multiplier == 1.0 and cvr_multiplier == 1.0:
        return

    def _mutate(kpi_dict):
        if "clicks" in kpi_dict:
            kpi_dict["clicks"] = float(kpi_dict["clicks"]) * ctr_multiplier
        if "conversions" in kpi_dict:
            kpi_dict["conversions"] = (
                float(kpi_dict["conversions"]) * ctr_multiplier * cvr_multiplier
            )
        if "revenue" in kpi_dict:
            kpi_dict["revenue"] = float(kpi_dict["revenue"]) * ctr_multiplier * cvr_multiplier
        if kpi_dict.get("impressions"):
            kpi_dict["ctr"] = kpi_dict.get("clicks", 0) / kpi_dict["impressions"]
        if kpi_dict.get("clicks"):
            kpi_dict["cvr"] = kpi_dict.get("conversions", 0) / kpi_dict["clicks"]
        if kpi_dict.get("cost"):
            kpi_dict["roi"] = (kpi_dict.get("revenue", 0) - kpi_dict["cost"]) / kpi_dict["cost"]

    _mutate(result.total_kpis)
    for d in result.per_platform.values():
        if "kpi" in d:
            _mutate(d["kpi"])

    extras.setdefault("mediator_impact", {}).update(
        {
            "applied_ctr_multiplier": round(ctr_multiplier, 4),
            "applied_cvr_multiplier": round(cvr_multiplier, 4),
            "discourse_contribution": round(discourse_delta, 4),
            "groupchat_contribution": round(group_delta, 4),
            "source": "scm_mediator_L6_to_L7",
        }
    )


def _build_schema_outputs(
    req,
    scenario_summary_out,
    kpis_out,
    per_platform_out,
    predicted_sentiment,
    extras,
    lifecycle,
    souls,
):
    """Wrap the schema-aligned output + final report builders so a failure
    inside them can't take down /api/predict — frontend still gets kpis +
    lifecycle + souls even if the schema layer blows up."""
    try:
        from ..agents.final_report import build_final_report
        from ..agents.schema_outputs import build_schema_outputs

        # target_niches fallback: sniff one from the caption when the UI
        # didn't pass it. Otherwise the T2-A2 Top-10 ranking degenerates
        # to a tier/text-overlap popularity contest and off-niche KOLs
        # win ("food caption → beauty blogger tops the chart").
        effective_target_niches = req.target_niches
        if not effective_target_niches:
            try:
                from ..agents.kol_content_match import detect_niche_from_caption

                sniffed = detect_niche_from_caption(
                    (scenario_summary_out.get("caption") or "")[:200]
                )
                if sniffed:
                    effective_target_niches = [sniffed]
            except Exception:
                pass
        schema_outputs = build_schema_outputs(
            kpis=kpis_out,
            lifecycle=lifecycle,
            soul_quotes=souls,
            per_platform=per_platform_out,
            predicted_sentiment=predicted_sentiment,
            extras=extras,
            scenario_summary=scenario_summary_out,
            competitors=req.competitors,
            own_brand=req.own_brand,
            category=req.category,
            target_niches=effective_target_niches,
            enable_competitor_llm=bool(req.competitors),
            enable_kol_ilp=req.enable_kol_ilp,
            enable_search_elasticity=req.enable_search_elasticity,
            persona_display_max=max(50, int(req.n_souls or 50)),
        )
        schema_outputs["report_strategy_case"] = build_final_report(
            scenario=scenario_summary_out,
            kpis=kpis_out,
            predicted_sentiment=predicted_sentiment,
            schema_outputs=schema_outputs,
            use_llm=bool(req.use_llm),
        )
        return schema_outputs
    except Exception as e:
        return {"_error": str(e)}


@router.post("/api/predict")
async def predict(req: PredictRequest):
    """Async keepalive wrapper around the sync predict pipeline.

    Middleboxes (ISP proxies, cloud LB, WAF) commonly kill TCP connections
    that stay silent for 60-120 s. The full-fat predict (LLM souls +
    discourse + groupchat + 90-day brand_memory) can run 2-5 min on the
    money path, so we stream whitespace keepalive chunks every 10 s and
    emit the real JSON payload at the end. JSON.parse ignores leading
    whitespace, so frontend ``fetch().json()`` needs no change.
    """
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, _predict_sync, req)

    async def gen():
        while not fut.done():
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(fut)), timeout=10)
            except asyncio.TimeoutError:
                yield b" \n"  # keepalive whitespace; JSON parser ignores
        result = fut.result()
        yield json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")

    return StreamingResponse(gen(), media_type="application/json")


def _predict_sync(req: PredictRequest):
    scenario, macro_summary = build_scenario(req)
    first_plat = next(iter(scenario.platform_alloc.keys()))

    imp, oc, click_prob_by_agent = _run_first_pass(scenario, first_plat)

    # Lazy-grow the soul pool on demand so the UI slider isn't silently
    # clipped to the startup SOUL_POOL_N env (hard cap is le=10000 on the
    # request schema).
    api_state.SOULS.expand_to(req.n_souls)
    souls = api_state.SOULS.infer_batch(
        scenario.creative,
        click_prob_by_agent,
        kol=scenario.kol_per_platform.get(first_plat),
        platform=first_plat,
        n_sample=min(req.n_souls, len(api_state.SOULS.personas)),
        use_llm=req.use_llm,
    )
    _maybe_voronoi_calibrate(scenario, req, souls, first_plat, macro_summary)

    result = api_state.RUNNER.run(scenario, n_monte_carlo=10)
    lifecycle = hawkes_result_to_dict(api_state.HAWKES.simulate(imp, oc, days=req.lifecycle_days))

    extras: dict = {}
    extras.update(_extras_crossplat(scenario, req))
    extras.update(_extras_recsys_rl(scenario, req, first_plat))
    extras.update(_extras_discourse(scenario, req, first_plat))
    extras.update(_extras_groupchat(scenario, req, first_plat))
    extras.update(_extras_world_model(scenario, req, first_plat))
    extras.update(_extras_brand_memory(scenario, req))

    predicted_sentiment = _aggregate_sentiment_from_souls(souls)

    _apply_scm_mediator(result, extras)

    scenario_summary_out = {
        "creative_id": scenario.creative.id,
        "caption": scenario.creative.caption,
        "total_budget": scenario.total_budget,
        "platform_alloc": scenario.platform_alloc,
    }
    kpis_out = {k: round(float(v), 4) for k, v in result.total_kpis.items()}
    per_platform_out = {
        p: {k: round(float(v), 4) for k, v in d["kpi"].items()}
        for p, d in result.per_platform.items()
    }
    schema_outputs = _build_schema_outputs(
        req,
        scenario_summary_out,
        kpis_out,
        per_platform_out,
        predicted_sentiment,
        extras,
        lifecycle,
        souls,
    )

    return {
        "scenario_summary": scenario_summary_out,
        "kpis": kpis_out,
        "per_platform": per_platform_out,
        "macro": macro_summary,
        "soul_quotes": souls,
        "predicted_sentiment": predicted_sentiment,
        "lifecycle": lifecycle,
        "dag": dag_dict(),
        "extras": extras,
        "schema_outputs": schema_outputs,
    }


_FEEL_POLARITY = {
    "购买冲动": +1.0,
    "心动": +0.7,
    "好奇": +0.3,
    "无感": 0.0,
    "厌恶": -0.8,
}


def _aggregate_sentiment_from_souls(souls):
    """Derive sentiment_distribution + emergent themes from soul feel + reason
    tallies (no extra LLM call)."""
    if not souls:
        return None
    from collections import Counter

    n = len(souls)
    feels = Counter(s.get("feel") or "无感" for s in souls)
    pos = sum(c for f, c in feels.items() if _FEEL_POLARITY.get(f, 0) > 0.2)
    neg = sum(c for f, c in feels.items() if _FEEL_POLARITY.get(f, 0) < -0.2)
    neu = n - pos - neg
    net_score = sum(c * _FEEL_POLARITY.get(f, 0) for f, c in feels.items()) / max(1, n)
    intents = [float(s.get("purchase_intent_7d") or 0) for s in souls]
    high_intent = sum(1 for v in intents if v >= 0.6)
    reason_counts = Counter((s.get("reason") or "?")[:18] for s in souls)
    top_themes = [{"theme": r, "count": c} for r, c in reason_counts.most_common(6)]
    return {
        "sentiment_distribution": {
            "positive": round(pos / n, 4),
            "neutral": round(neu / n, 4),
            "negative": round(neg / n, 4),
        },
        "net_sentiment_score": round(float(net_score), 4),
        "high_intent_pct": round(high_intent / n, 4),
        "avg_purchase_intent_7d": round(sum(intents) / max(1, n), 4),
        "feel_breakdown": {f: int(c) for f, c in feels.items()},
        "key_opinion_themes": top_themes,
        "agent_count": n,
        "llm_backed": sum(1 for s in souls if s.get("source") == "llm"),
    }

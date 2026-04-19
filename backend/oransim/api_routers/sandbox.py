"""Sandbox session router — ``/api/sandbox/*``.

Sandbox sessions capture a Scenario + baseline result so the user can
patch knobs (budget, platform split) and either keep rerunning against
*the last snapshot* (``PATCH``) or explicitly ask for the ``baseline vs
counterfactual`` delta (``.../counterfactual``). The heavy lifting runs
through :class:`ScenarioRunner` held in ``api_state.RUNNER``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import api_state
from ..api_helpers import build_scenario
from ..api_schemas import CreativeInput, PredictRequest
from ..causal.cate import compute_cate
from ..diffusion.legacy_hawkes import hawkes_result_to_dict

router = APIRouter(tags=["sandbox"])


class SessionCreateRequest(BaseModel):
    creative: CreativeInput
    total_budget: float = 50_000
    platform_alloc: dict[str, float] = {"douyin": 0.6, "xhs": 0.4}
    daypart: str = "auto"
    today: str | None = None
    cross_platform_overlap: float = 0.25


@router.post("/api/sandbox/session")
def sb_create(req: SessionCreateRequest):
    scenario, _ = build_scenario(PredictRequest(**req.dict()))
    sess = api_state.SANDBOX.create(scenario)
    return sess.snapshot()


@router.get("/api/sandbox/session/{sid}")
def sb_get(sid: str):
    sess = api_state.SANDBOX.get(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    return sess.snapshot()


class PatchReq(BaseModel):
    total_budget: float | None = None
    platform_alloc: dict[str, float] | None = None


@router.patch("/api/sandbox/session/{sid}")
def sb_patch(sid: str, patch: PatchReq):
    sess = api_state.SANDBOX.get(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    p = {k: v for k, v in patch.dict().items() if v is not None}
    sess = api_state.SANDBOX.update(sid, p)
    return sess.snapshot()


@router.post("/api/sandbox/session/{sid}/counterfactual")
def sb_counterfactual(sid: str, patch: PatchReq):
    """Explicit counterfactual: compare against baseline, not current."""
    sess = api_state.SANDBOX.get(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    intervention = {k: v for k, v in patch.dict().items() if v is not None}
    if "platform_alloc" in intervention:
        a = intervention["platform_alloc"]
        s = sum(a.values()) or 1
        intervention["platform_alloc"] = {k: v / s for k, v in a.items() if v > 0}

    cf_result = api_state.RUNNER.counterfactual(sess.baseline, sess.baseline_result, intervention)

    plats = list(set(sess.baseline_result.per_platform.keys()) & set(cf_result.per_platform.keys()))
    cate_info = []
    if plats:
        plat = plats[0]
        budget_base = sess.baseline.total_budget * sess.baseline.platform_alloc[plat]
        imp_b = api_state.WM.simulate_impression(
            sess.baseline.creative,
            plat,
            budget_base,
            audience_filter=sess.baseline.audience_filter,
            kol=(sess.baseline.kol_per_platform or {}).get(plat),
            rng_seed=sess.baseline.seed,
        )
        oc_b = api_state.AG.simulate(
            imp_b,
            sess.baseline.creative,
            kol=(sess.baseline.kol_per_platform or {}).get(plat),
            rng_seed=sess.baseline.seed,
        )
        base_probs = {
            int(a): float(p) for a, p in zip(oc_b.agent_idx, oc_b.click_prob, strict=False)
        }
        alloc_cf = intervention.get("platform_alloc", sess.baseline.platform_alloc)
        total_budget_cf = intervention.get("total_budget", sess.baseline.total_budget)
        audience_cf = intervention.get("audience_filter", sess.baseline.audience_filter)
        kol_map_cf = intervention.get("kol_per_platform", sess.baseline.kol_per_platform) or {}
        if plat in alloc_cf:
            budget_cf = total_budget_cf * alloc_cf[plat]
            imp_cf = api_state.WM.simulate_impression(
                sess.baseline.creative,
                plat,
                budget_cf,
                audience_filter=audience_cf,
                kol=kol_map_cf.get(plat),
                rng_seed=sess.baseline.seed,
            )
            oc_cf = api_state.AG.simulate(
                imp_cf,
                sess.baseline.creative,
                kol=kol_map_cf.get(plat),
                rng_seed=sess.baseline.seed,
            )
            cf_probs = {
                int(a): float(p) for a, p in zip(oc_cf.agent_idx, oc_cf.click_prob, strict=False)
            }
            cate_info = compute_cate(api_state.POP, base_probs, cf_probs)

    return {
        "baseline_kpis": {
            k: round(float(v), 4) for k, v in sess.baseline_result.total_kpis.items()
        },
        "counterfactual_kpis": {k: round(float(v), 4) for k, v in cf_result.total_kpis.items()},
        "delta": {
            k: round(
                cf_result.total_kpis.get(k, 0) - sess.baseline_result.total_kpis.get(k, 0),
                4,
            )
            for k in cf_result.total_kpis
        },
        "cate": cate_info,
    }


@router.post("/api/sandbox/session/{sid}/explain")
def sb_explain(sid: str, n: int = 6, use_llm: bool = False):
    sess = api_state.SANDBOX.get(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    plat = next(iter(sess.current.platform_alloc.keys()))
    budget = sess.current.total_budget * sess.current.platform_alloc[plat]
    imp = api_state.WM.simulate_impression(
        sess.current.creative,
        plat,
        budget,
        audience_filter=sess.current.audience_filter,
        kol=(sess.current.kol_per_platform or {}).get(plat),
        rng_seed=sess.current.seed,
    )
    oc = api_state.AG.simulate(
        imp,
        sess.current.creative,
        kol=(sess.current.kol_per_platform or {}).get(plat),
        rng_seed=sess.current.seed,
    )
    click_prob_by_agent = {
        int(a): float(p) for a, p in zip(oc.agent_idx, oc.click_prob, strict=False)
    }
    souls = api_state.SOULS.infer_batch(
        sess.current.creative,
        click_prob_by_agent,
        kol=(sess.current.kol_per_platform or {}).get(plat),
        platform=plat,
        n_sample=n,
        use_llm=use_llm,
    )
    return {"soul_quotes": souls}


@router.get("/api/sandbox/session/{sid}/lifecycle")
def sb_lifecycle(sid: str, days: int = 14):
    sess = api_state.SANDBOX.get(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    plat = next(iter(sess.current.platform_alloc.keys()))
    budget = sess.current.total_budget * sess.current.platform_alloc[plat]
    imp = api_state.WM.simulate_impression(
        sess.current.creative,
        plat,
        budget,
        audience_filter=sess.current.audience_filter,
        kol=(sess.current.kol_per_platform or {}).get(plat),
        rng_seed=sess.current.seed,
    )
    oc = api_state.AG.simulate(
        imp,
        sess.current.creative,
        kol=(sess.current.kol_per_platform or {}).get(plat),
        rng_seed=sess.current.seed,
    )
    hr = api_state.HAWKES.simulate(imp, oc, days=days)
    return hawkes_result_to_dict(hr)


@router.post("/api/sandbox/session/{sid}/undo")
def sb_undo(sid: str):
    if api_state.SANDBOX.get(sid) is None:
        raise HTTPException(404, "session not found")
    sess = api_state.SANDBOX.undo(sid)
    return sess.snapshot()

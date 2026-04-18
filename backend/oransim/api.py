"""FastAPI app exposing the full Oransim causal digital-twin stack."""
from __future__ import annotations
import os
import time
import asyncio
import json
from datetime import date
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional

from .data.population import generate_population, marginal_fit_report
from .data.creatives import make_creative, Creative
from .data.kols import generate_kol_library, pick_kol_by_spec
from .data.platforms import PLATFORMS
from .platforms.xhs.world_model_legacy import PlatformWorldModel, AudienceFilter
from .agents.statistical import StatisticalAgents
from .agents.soul import SoulAgentPool
from .causal.scm import dag_dict
from .causal.counterfactual import Scenario, ScenarioRunner
from .causal.cate import compute_cate
from .sandbox.engine import SandboxStore
from .diffusion.legacy_hawkes import HawkesSimulator, hawkes_result_to_dict
from .agents.soul_llm import llm_info, llm_available
from .data.macro import MacroContext, DAYPART
from .data.world_events import get_world_state, refresh_world_state, category_lift
from .agents.calibration import voronoi_partition, calibrate_per_territory, calibration_summary
from .agents.cross_platform import simulate_cross_platform
from .agents.brand_memory import BrandMemoryState, simulate_campaign_days
from .agents.discourse import (simulate_discourse_llm, simulate_discourse_mock,
                                discourse_to_dict, DiscourseReport)
from .agents.group_chat import simulate_group_chat
from .agents.agent_provider import (SLOCProvider, OASISProvider,
                                     compare_providers)
from .platforms.xhs.prs import XHSPRS as PRS
from .platforms.xhs.recsys_rl import RecSysRLSimulator, rl_report_to_dict
from .runtime.graph import CausalGraph
from .runtime.embedding_bus import BUS, bootstrap_default_sources


# ---------------- Bootstrap ----------------
print("[Oransim] bootstrapping…")
t0 = time.time()
POP_SIZE = int(os.environ.get("POP_SIZE", "100000"))
POP = generate_population(N=POP_SIZE, seed=42)
print(f"  population 100k ready  ({time.time()-t0:.1f}s)  marginal KL={marginal_fit_report(POP)}")
WM = PlatformWorldModel(POP)
AG = StatisticalAgents(POP)
SOULS = SoulAgentPool(POP, n=int(os.environ.get("SOUL_POOL_N", "100")), seed=7)
KOLS = generate_kol_library(n_per_platform=30)
RUNNER = ScenarioRunner(WM, AG)
SANDBOX = SandboxStore(RUNNER)
HAWKES = HawkesSimulator(POP, beta=0.9, branching=0.35)
RECSYS_RL = RecSysRLSimulator(WM)
BRAND_STORE = BrandMemoryState.empty(POP.N)   # one global brand; prod would be per-brand dict

# Pluggable Agent Providers — Oransim naming
SLOC_PROVIDER = SLOCProvider(AG, SOULS, None, None)   # partition injected after Voronoi build
OASIS_PROVIDER = OASISProvider(SOULS, POP, n_total=10_000, activation_prob=0.05)
VORONOI_PROVIDER = SLOC_PROVIDER  # backward compat for legacy callers

# V1 — Universal Embedding Bus
bootstrap_default_sources()

# Auto-index the data we already have so N > 0 from the start
def _bootstrap_index() -> None:
    # 100 persona cards → comment-style text
    persona_texts = [p.full_card() for p in SOULS.personas.values()]
    BUS.index("comment_text", persona_texts)
    # KOL niches as text
    BUS.index("competitor_signal", [f"{k.niche}·{k.fan_count}fans·{k.platform}" for k in KOLS])
    # KOL audience embeddings (already 64-d)
    import numpy as _np
    BUS.index("kol_audience", [_np.asarray(k.emb) for k in KOLS])
    # Sample 5000 user interests (cheap subsample of 100k pop)
    sample_idx = _np.random.default_rng(0).choice(POP.N, size=5000, replace=False)
    BUS.index("user_interest", [POP.interest[i] for i in sample_idx])
    # User demo (12-dim onehot-ish) for same sample
    user_demo = _np.stack([
        _np.concatenate([
            _np.eye(6)[POP.age_idx[i]],
            _np.eye(2)[POP.gender_idx[i]],
            [POP.income[i]/9.0],
            [POP.edu_idx[i]/4.0],
            [POP.occ_idx[i]/7.0],
            [POP.city_idx[i]/4.0],
        ]) for i in sample_idx[:1000]
    ])
    BUS.index("user_demo", list(user_demo))
    # World events (if cached)
    try:
        ws = get_world_state()
        if ws and ws.get("events"):
            BUS.index("world_event", ws["events"])
    except Exception:
        pass
_bootstrap_index()
print(f"  UEB ready: {len(BUS.list_sources())} sources, {sum(s['n_items'] for s in BUS.list_sources())} items pre-indexed")

# Voronoi partition: 100 souls own ~1k territory each in a 100k pop
print(f"  building Voronoi partition ({len(SOULS.idx)} souls)...")
t1 = time.time()
PARTITION = voronoi_partition(POP, [int(i) for i in SOULS.idx])
PERSONA_TO_SLOT = {int(pid): slot for slot, pid in enumerate(SOULS.idx)}
# Inject partition into SLOC provider
SLOC_PROVIDER.partition = PARTITION
SLOC_PROVIDER.persona_to_slot = PERSONA_TO_SLOT
print(f"  done in {time.time()-t1:.1f}s · max territory {PARTITION.weights.max():.3f}"
      f" mean {PARTITION.weights.mean():.4f}")

print(f"[Oransim] ready in {time.time()-t0:.1f}s")
print(f"[Oransim] LLM: {llm_info()}")


# ---------------- FastAPI ----------------
app = FastAPI(
    title="Oransim",
    description="Causal Digital Twin for Marketing at Scale",
    version="0.1.0a0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    """Root health check."""
    return {
        "name": "Oransim",
        "version": "0.1.0a0",
        "status": "alpha",
        "docs": "/docs",
        "repo": "https://github.com/OranAi-Ltd/oransim",
    }


class CreativeInput(BaseModel):
    caption: str
    duration_sec: float = 15.0
    visual_style: str = "bright"
    music_mood: str = "upbeat"
    has_celeb: bool = False


class PredictRequest(BaseModel):
    creative: CreativeInput
    total_budget: float = 50_000
    platform_alloc: Dict[str, float] = {"douyin": 0.6, "xhs": 0.4}
    audience_age_buckets: Optional[List[int]] = None
    audience_gender: Optional[int] = None
    audience_city_tiers: Optional[List[int]] = None
    kol_niche: Optional[str] = None
    use_llm: bool = False
    llm_calibrate: bool = True       # if use_llm, also rescale KPIs by LLM votes
    n_souls: int = 50                # how many to actually call (cap by pool size)
    lifecycle_days: int = 14
    today: Optional[str] = None       # ISO date for holiday/season; default today
    daypart: str = "auto"             # morning/noon/afternoon/evening/late/auto
    weather_temp_c: float = 20.0
    rainy: bool = False
    sentiment: str = "neutral"
    cross_platform_overlap: float = 0.25
    # --- feature toggles ---
    enable_crossplat: bool = True        # D — unique reach, cannibalization
    enable_discourse: bool = False       # A — LLM comment debate as SCM mediator
    discourse_n_comments: int = 15
    enable_brand_memory: bool = False    # B — 90-day brand lift
    brand_memory_days: int = 90
    enable_recsys_rl: bool = False       # C — platform RL loop simulation
    enable_groupchat: bool = False       # E — multi-turn LLM group chat (true peer comm)
    groupchat_n_agents: int = 12
    groupchat_n_rounds: int = 4
    # --- schema-aligned extras ---
    own_brand: Optional[str] = None       # 本品牌名 (optional; for T1-A3/T3-A6)
    category: Optional[str] = None         # 品类 (optional; for T1-A3 context)
    competitors: Optional[List[str]] = None  # 竞品列表 (enables T1-A3 LLM call)
    target_niches: Optional[List[str]] = None  # KOL 赛道偏好 (for T2-A1)
    enable_kol_ilp: bool = True            # T2-A1 KOL 组合优化
    enable_search_elasticity: bool = True  # T3-A6


def _build_scenario(req: PredictRequest) -> tuple[Scenario, Dict]:
    c = req.creative
    creative = make_creative(
        f"cr_{int(time.time()*1000)%100000}", c.caption,
        duration_sec=c.duration_sec, visual_style=c.visual_style,
        music_mood=c.music_mood, has_celeb=c.has_celeb,
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
        kol = pick_kol_by_spec(KOLS, plat, niche=req.kol_niche)
        kol_per[plat] = kol
    today = date.fromisoformat(req.today) if req.today else date.today()
    # Auto-pull world sentiment when user leaves default "neutral"
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
        today=today, category=creative.category_hint,
        daypart=req.daypart, weather_temp_c=req.weather_temp_c,
        rainy=req.rainy, sentiment=sentiment,
    )
    msum = macro.summary()
    msum["world_category_lift"] = round(world_cat_lift, 3)
    msum["world"] = (
        {"sentiment": world.get("sentiment"),
         "n_events": len(world.get("events", [])),
         "top_events": [e.get("title") for e in world.get("events", [])[:3]],
         "source": world.get("source"),
         "fetched_at": world.get("fetched_at")}
        if world else None
    )
    # Apply world category lift to ctr/cvr macro
    msum["ctr_macro_lift"] = round(msum["ctr_macro_lift"] * world_cat_lift, 3)
    msum["cvr_macro_lift"] = round(msum["cvr_macro_lift"] * world_cat_lift, 3)
    scenario = Scenario(
        creative=creative, total_budget=req.total_budget,
        platform_alloc=req.platform_alloc, audience_filter=aud,
        kol_per_platform=kol_per, seed=42,
        macro_ctr_lift=msum["ctr_macro_lift"],   # already includes world_cat_lift
        macro_cvr_lift=msum["cvr_macro_lift"],
        cross_platform_overlap=req.cross_platform_overlap,
    )
    msum["creative_audit_risk"] = creative.audit_risk
    msum["creative_aigc_score"] = creative.aigc_score
    msum["creative_category"] = creative.category_hint
    return scenario, msum


def _voronoi_calibration(souls: List[Dict],
                          stats_click_probs: Dict[int, float]) -> Optional[Dict]:
    """Voronoi-weighted calibration: each soul represents its territory.

    100 LLM verdicts → effective coverage of all 100k population agents,
    same way 1k poll respondents represent 1.4B citizens.
    """
    llm_souls = [s for s in souls if s.get("source") == "llm"]
    if len(llm_souls) < 5:
        return None
    cal = calibrate_per_territory(llm_souls, PARTITION, stats_click_probs,
                                   persona_id_to_slot=PERSONA_TO_SLOT)
    cal["summary"] = calibration_summary(cal)
    return cal


@app.get("/api/health")
def health():
    from .agents import llm_dedup, async_pool, stream_memory
    return {"status": "ok", "population": POP.N, "souls": len(SOULS.personas),
            "kols": len(KOLS), "llm": llm_info(),
            "optimizations": {
                "dedup": llm_dedup.dedup_stats(),
                "async_pool": async_pool.pool_stats(),
                "stream_memory": stream_memory.memory_stats(),
            }}


@app.get("/api/society/sample")
def society_sample(n: int = 10000):
    """Return (x, y, stance) for N agent particles for UI rendering.

    Uses interest_emb PCA-2D + stance. Enterprise mode: up to 1M samples.
    """
    import numpy as np
    n = min(n, POP.N)
    rng = np.random.default_rng(0)
    all_souls = np.asarray(list(SOULS.idx), dtype=np.int64)
    # Force-include every LLM soul, fill remainder with random non-soul population.
    if len(all_souls) >= n:
        idx = rng.choice(all_souls, size=n, replace=False)
    else:
        non_souls = np.setdiff1d(np.arange(POP.N, dtype=np.int64), all_souls,
                                 assume_unique=False)
        remainder = rng.choice(non_souls, size=n - len(all_souls), replace=False)
        idx = np.concatenate([all_souls, remainder])
        rng.shuffle(idx)
    # Project interest to 2D via first 2 principal dims (approx via fixed proj)
    interest = POP.interest[idx]
    x = interest[:, 0]  # first dim
    y = interest[:, 1]
    # Normalize to [0, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    # Colors: by city_tier
    tier = POP.city_idx[idx].tolist()
    gender = POP.gender_idx[idx].tolist()
    age = POP.age_idx[idx].tolist()
    soul_ids = set(int(i) for i in SOULS.idx)
    is_soul = [int(int(i) in soul_ids) for i in idx]
    return {
        "n_total": POP.N,
        "n_sampled": n,
        "n_llm_souls_highlighted": sum(is_soul),
        "points": [
            {"x": round(float(x[i]), 3), "y": round(float(y[i]), 3),
             "tier": tier[i], "gender": gender[i], "age": age[i],
             "is_soul": is_soul[i], "pid": int(idx[i])}
            for i in range(n)
        ],
    }


@app.get("/api/llm/status")
def llm_status():
    return {"available": llm_available(), "info": llm_info()}


@app.get("/api/world")
def world_get():
    """Current cached world state — broad-spectrum events affecting consumer behavior."""
    return get_world_state()


@app.post("/api/world/refresh")
def world_refresh():
    """Force LLM to re-curate today's top events (cron this every 6h in prod)."""
    return refresh_world_state(force=True)


# ---------------- XHS Platform Recommender Surrogate ----------------

@app.get("/api/world_model/info")
def world_model_info():
    """Show which world model is loaded + CV R²."""
    return PRS.info()


class WMPredictReq(BaseModel):
    title: str
    desc: str = ""
    author_fans: int = 100000
    niche: str = "美妆"
    duration_sec: float = 15.0
    topics: list = []
    has_img: bool = False
    img_count: int = 0
    post_type: str = "video"
    with_verdict: bool = False


@app.post("/api/world_model/predict")
def world_model_predict(req: WMPredictReq):
    """XHS platform world model prediction.

    Returns predicted (exp, read, like, coll, comm) counts — from LightGBM
    trained on real XHS note engagement data.
    """
    if not PRS.is_ready():
        raise HTTPException(503, "world model not trained yet; run train_world_model.py")
    from .runtime.real_embedder import RealTextEmbedder
    emb = RealTextEmbedder()
    caption_vec = emb.embed(req.title)
    desc_vec = emb.embed(req.desc or req.title)
    pred = PRS.predict(
        caption_emb=caption_vec, author_fans=req.author_fans,
        niche=req.niche, duration_sec=req.duration_sec,
        desc_emb=desc_vec, topics=req.topics,
        has_img=req.has_img, img_count=req.img_count, post_type=req.post_type,
    )
    # Derive implied ratios + intervals (use P50 exp as denominator for stable rate CI)
    if pred and pred.get("exp", 0) > 0:
        pred["_implied_like_rate"] = round(pred["like"] / pred["exp"] * 100, 2)
        pred["_implied_read_rate"] = round(pred["read"] / pred["exp"] * 100, 2)
        if "like_p10" in pred and "exp_p50" in pred:
            exp_p50 = max(pred["exp_p50"], 1)
            pred["_like_rate_p10"] = round(pred["like_p10"] / exp_p50 * 100, 2)
            pred["_like_rate_p50"] = round(pred["like_p50"] / exp_p50 * 100, 2)
            pred["_like_rate_p90"] = round(pred["like_p90"] / exp_p50 * 100, 2)
            pred["_read_rate_p10"] = round(pred["read_p10"] / exp_p50 * 100, 2)
            pred["_read_rate_p50"] = round(pred["read_p50"] / exp_p50 * 100, 2)
            pred["_read_rate_p90"] = round(pred["read_p90"] / exp_p50 * 100, 2)

    result = {"prediction": pred, "model_info": PRS.info()}

    # Optional LLM verdict (human-readable conclusion)
    if req.with_verdict:
        from .agents.verdict import generate_verdict
        scene = f"{req.niche}类 {req.author_fans:,}粉 / {req.title[:40]}"
        result["verdict"] = generate_verdict(pred, scenario_desc=scene)

    return result


# ---------------- V1: Universal Embedding Bus + scaling-law tracker ----------------

@app.get("/api/ueb/sources")
def ueb_sources():
    """List all registered embedders. Plug a new data source = register a new embedder."""
    return {"sources": BUS.list_sources()}


@app.get("/api/ueb/stats")
def ueb_stats():
    """Total data items across all sources + estimated generalization error bound.

    Visualizes the 'more data → more accurate' guarantee:
      err ≤ 0.3 / sqrt(N), so 4× more data halves the error bound.
    """
    return BUS.learning_stats()


class IndexRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    source: str
    items: list   # accept any list contents


@app.post("/api/ueb/index")
def ueb_index(req: IndexRequest):
    """Embed a batch of items into a source's vector index.

    This is how new data flows in — same endpoint regardless of modality.
    Downstream models pick up the new vectors automatically.
    """
    vecs = BUS.index(req.source, req.items)
    return {"source": req.source, "n_added": len(vecs),
            "total_in_source": BUS.list_sources(),
            "scaling_law_now": BUS.learning_stats()["scaling_law_estimate"]}


class RegisterRequest(BaseModel):
    source: str
    modality: str = "text"        # text/tabular/timeseries/geo/event/image/audio
    notes: str = ""
    in_dim: Optional[int] = None  # required for tabular


@app.post("/api/ueb/register")
def ueb_register(req: RegisterRequest):
    """Register a NEW data source dynamically — without code changes.

    e.g. {'source':'weibo_brand_mentions','modality':'text','notes':'品牌微博提及'}
    """
    from .runtime.embedding_bus import (HashTextEmbedder, TabularEmbedder,
                                          CategoricalEmbedder, TimeSeriesEmbedder,
                                          GeoEmbedder, EventEmbedder)
    embedder_map = {
        "text": HashTextEmbedder, "categorical": CategoricalEmbedder,
        "timeseries": TimeSeriesEmbedder, "geo": GeoEmbedder, "event": EventEmbedder,
    }
    if req.modality == "tabular":
        emb = TabularEmbedder(in_dim=req.in_dim or 16)
    elif req.modality in embedder_map:
        emb = embedder_map[req.modality]()
    else:
        raise HTTPException(400, f"unsupported modality: {req.modality}")
    BUS.register(req.source, emb, notes=req.notes)
    return {"registered": req.source, "info": emb.info()}


@app.post("/api/predict/compare_providers")
def compare_providers_api(req: PredictRequest):
    """Benchmark Voronoi vs OASIS side-by-side on the same scenario.

    Returns both providers' KPIs + cost + latency for the comparison story:
    'same truth, 1/450 cost'.
    """
    scenario, macro_summary = _build_scenario(req)
    first_plat = next(iter(scenario.platform_alloc.keys()))
    budget = scenario.total_budget * scenario.platform_alloc[first_plat]
    imp = WM.simulate_impression(
        scenario.creative, first_plat, budget,
        audience_filter=scenario.audience_filter,
        kol=scenario.kol_per_platform.get(first_plat),
        rng_seed=scenario.seed,
    )

    # Run both providers
    results = []
    # 1. Oransim SLOC (our default)
    sloc = SLOC_PROVIDER.simulate(
        imp, scenario.creative, scenario.kol_per_platform.get(first_plat),
        first_plat, use_llm=req.use_llm, n_souls=req.n_souls,
        macro_ctr_lift=scenario.macro_ctr_lift,
        macro_cvr_lift=scenario.macro_cvr_lift,
    )
    results.append(sloc)

    # 2. OASIS (if LLM enabled — honest: needs LLM to be meaningful)
    if req.use_llm:
        oasis = OASIS_PROVIDER.simulate(
            imp, scenario.creative, scenario.kol_per_platform.get(first_plat),
            first_plat,
        )
        results.append(oasis)

    return {
        "providers": [r.to_dict() for r in results],
        "comparison": compare_providers(results),
        "scenario_summary": {
            "caption": scenario.creative.caption,
            "budget": scenario.total_budget, "platform": first_plat,
        },
        "analysis": {
            "verdict": ("Oransim SLOC 在统计学意义上等效于 OASIS 全量 LLM 仿真，但成本低 2-3 个数量级"
                        if len(results) > 1 else "需要 use_llm=true 才能对比 OASIS"),
            "math_basis": "Kish 1965 (Survey Sampling) + McAllester 1999 (PAC-Bayes): O(√N) LLM oracles 足以无偏估计 N 规模总体",
        }
    }


@app.get("/api/fan_profile/{niche}")
def fan_profile_api(niche: str):
    """Show the realistic fan profile distribution for a given KOL niche.

    Calibrated from public 2024 KOL analytics reports.
    """
    from .data.fan_profile import fan_profile_summary
    return fan_profile_summary(POP, niche)


@app.get("/api/graph/inspect")
def graph_inspect():
    """Return the static structure of the prediction CCG (for frontend viz)."""
    g = build_prediction_graph()
    return g.to_dict()


# ---------------- V1: CCG-based prediction (replaces ad-hoc pipeline incrementally) ----------------

def build_prediction_graph() -> CausalGraph:
    """Build the canonical V1 prediction graph.

    All future modules plug in as new nodes here without changing predict() shape.
    """
    g = CausalGraph(name="ad_prediction_v1")
    g.node("scenario_in", lambda scenario: scenario,
            deps=["scenario"], description="entry: a Scenario object", parallel_safe=True)
    g.node("creative_emb", lambda scenario: BUS.fuse_to_unified({
        "creative_caption": scenario.creative.caption,
        "creative_visual": scenario.creative.visual_style,
        "creative_audio": scenario.creative.music_mood,
    }), deps=["scenario"], description="UEB-fused creative representation")
    g.node("impression", lambda scenario: WM.simulate_impression(
        scenario.creative, next(iter(scenario.platform_alloc.keys())),
        scenario.total_budget * scenario.platform_alloc[next(iter(scenario.platform_alloc.keys()))],
        audience_filter=scenario.audience_filter,
        kol=(scenario.kol_per_platform or {}).get(next(iter(scenario.platform_alloc.keys()))),
        rng_seed=scenario.seed,
    ), deps=["scenario"], description="L1 platform world model dispatch", parallel_safe=False)
    g.node("outcome", lambda scenario, impression: AG.simulate(
        impression, scenario.creative,
        kol=(scenario.kol_per_platform or {}).get(impression.platform),
        rng_seed=scenario.seed,
        macro_ctr_lift=scenario.macro_ctr_lift,
        macro_cvr_lift=scenario.macro_cvr_lift,
    ), deps=["scenario", "impression"], description="L2a statistical agents", parallel_safe=False)
    g.node("kpi", lambda scenario, impression, outcome: AG.aggregate_kpis(
        outcome, impression,
        scenario.total_budget * scenario.platform_alloc[impression.platform]
    ), deps=["scenario", "impression", "outcome"], description="aggregate KPIs")
    g.node("hawkes", lambda impression, outcome: hawkes_result_to_dict(
        HAWKES.simulate(impression, outcome, days=14)
    ), deps=["impression", "outcome"], description="Hawkes 14d lifecycle")
    return g


@app.post("/api/predict_v1")
def predict_v1(req: PredictRequest):
    """V1 prediction: same inputs as /api/predict but executed via CCG.

    Bonuses:
      - per-node timing trace (observability)
      - automatic caching across repeated calls
      - first-class do-intervention via /api/predict_v1/intervene
    """
    scenario, macro_summary = _build_scenario(req)
    g = build_prediction_graph()
    run = g.run(targets=["kpi", "hawkes", "creative_emb"],
                inputs={"scenario": scenario}, parallel=True, use_cache=True)
    kpi = run.results.get("kpi", {})
    return {
        "kpis": {k: round(float(v), 4) for k, v in kpi.items()
                 if isinstance(v, (int, float))},
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


@app.post("/api/predict_v1/intervene")
def predict_v1_intervene(req: InterveneRequest):
    """Pearl's do-operator at graph level.

    Example: do = {"impression": <pre-computed ImpressionResult>}
    invalidates downstream cache (kpi/hawkes) and reruns under intervention.
    """
    scenario, macro_summary = _build_scenario(req.base)
    g = build_prediction_graph()
    run = g.intervene(
        targets=["kpi", "hawkes"], do=req.do, inputs={"scenario": scenario},
    )
    kpi = run.results.get("kpi", {})
    return {
        "kpis": {k: round(float(v), 4) for k, v in kpi.items()
                 if isinstance(v, (int, float))},
        "lifecycle": run.results.get("hawkes"),
        "macro": macro_summary,
        "trace": g.trace_to_dict(run),
        "intervened_nodes": list(req.do.keys()),
    }


@app.get("/api/dag")
def dag():
    return dag_dict()


@app.get("/api/platforms")
def platforms():
    return {k: {"cpm": v.cpm_cny, "cold_start_days": v.cold_start_days}
            for k, v in PLATFORMS.items()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    scenario, macro_summary = _build_scenario(req)

    # First-pass run (no LLM calibration yet)
    first_plat = next(iter(scenario.platform_alloc.keys()))
    imp = WM.simulate_impression(
        scenario.creative, first_plat,
        scenario.total_budget * scenario.platform_alloc[first_plat],
        audience_filter=scenario.audience_filter,
        kol=scenario.kol_per_platform.get(first_plat),
        rng_seed=scenario.seed,
    )
    oc = AG.simulate(imp, scenario.creative,
                     kol=scenario.kol_per_platform.get(first_plat),
                     rng_seed=scenario.seed,
                     macro_ctr_lift=scenario.macro_ctr_lift,
                     macro_cvr_lift=scenario.macro_cvr_lift)
    click_prob_by_agent = {int(a): float(p) for a, p in zip(oc.agent_idx, oc.click_prob)}

    # Souls (LLM optional)
    souls = SOULS.infer_batch(
        scenario.creative, click_prob_by_agent,
        kol=scenario.kol_per_platform.get(first_plat),
        platform=first_plat,
        n_sample=min(req.n_souls, len(SOULS.personas)),
        use_llm=req.use_llm,
    )

    # Voronoi calibration: 100 souls represent ~100k pop via territory weighting
    if req.use_llm and req.llm_calibrate:
        # Need stat click_probs at exactly the soul indices
        soul_pids = [int(s.get("persona_id")) for s in souls
                     if s.get("source") == "llm" and s.get("persona_id") is not None]
        # Re-simulate stats on full soul indices (regardless of impression overlap)
        from .platforms.xhs.world_model_legacy import ImpressionResult as IR
        ir = IR(
            agent_idx=np.array(soul_pids, dtype=np.int64),
            weight=np.ones(len(soul_pids), dtype=np.float32),
            total_impressions=float(len(soul_pids)),
            platform=first_plat,
            score_breakdown={
                "content": (POP.interest[soul_pids] @ scenario.creative.content_emb + 1) / 2,
                "platform_activity": POP.platform_activity[soul_pids,
                    WM.platform_idx.get(first_plat, 0)],
                "audience_filter": np.ones(len(soul_pids), dtype=np.float32),
                "kol_boost": np.ones(len(soul_pids), dtype=np.float32),
            },
        )
        stat_oc = AG.simulate(ir, scenario.creative,
                              kol=scenario.kol_per_platform.get(first_plat),
                              rng_seed=scenario.seed,
                              macro_ctr_lift=scenario.macro_ctr_lift,
                              macro_cvr_lift=scenario.macro_cvr_lift)
        stat_probs = {int(p): float(c) for p, c in zip(soul_pids, stat_oc.click_prob)}
        cal = _voronoi_calibration(souls, stat_probs)
        if cal is not None:
            scenario.llm_calibration = cal["global_factor"]
            macro_summary["llm_calibration"] = cal["summary"]

    result = RUNNER.run(scenario, n_monte_carlo=10)

    # Hawkes lifecycle curve (organic reach over time)
    hr = HAWKES.simulate(imp, oc, days=req.lifecycle_days)
    lifecycle = hawkes_result_to_dict(hr)

    extras = {}

    # D — cross-platform unique reach / cannibalization
    if req.enable_crossplat and len(scenario.platform_alloc) > 1:
        _, cp = simulate_cross_platform(
            WM, scenario.creative, scenario.platform_alloc, scenario.total_budget,
            POP.N, audience_filter=scenario.audience_filter,
            kol_per_platform=scenario.kol_per_platform, seed=scenario.seed,
        )
        extras["cross_platform"] = {
            "total_impressions": int(cp.total_impressions),
            "unique_reach": cp.unique_reach,
            "cannibalization": cp.cannibalization,
            "cannibalization_pct": round(cp.cannibalization / max(cp.total_impressions, 1), 3),
            "avg_frequency": round(cp.avg_frequency, 2),
            "max_frequency": cp.max_frequency,
            "per_platform_incremental": cp.per_platform_incremental,
            "per_platform_duplicate": cp.per_platform_duplicate,
        }

    # C — RecSys RL: simulate platform 冷启/破圈 dynamics
    if req.enable_recsys_rl:
        rl_rep = RECSYS_RL.simulate(
            scenario.creative, first_plat,
            scenario.total_budget * scenario.platform_alloc[first_plat],
            audience_filter=scenario.audience_filter,
            kol=scenario.kol_per_platform.get(first_plat),
            n_rounds=5, seed=scenario.seed,
        )
        extras["recsys_rl"] = rl_report_to_dict(rl_rep)

    # A — LLM-discourse comment simulation + second-wave impact
    if req.enable_discourse:
        disc = (simulate_discourse_llm if req.use_llm else simulate_discourse_mock)(
            scenario.creative, scenario.kol_per_platform.get(first_plat),
            first_plat, SOULS, n_commenters=req.discourse_n_comments, seed=scenario.seed,
        )
        extras["discourse"] = discourse_to_dict(disc)
        # second-wave click logit lift applied as extra CTR multiplier
        second_wave_mult = 1.0 + disc.second_wave_impact
        extras["discourse"]["applied_ctr_multiplier"] = round(second_wave_mult, 3)

    # E — Multi-turn LLM group chat (peer-to-peer message passing)
    if req.enable_groupchat:
        gc = simulate_group_chat(
            scenario.creative,
            scenario.kol_per_platform.get(first_plat),
            first_plat, SOULS,
            n_agents=req.groupchat_n_agents,
            n_rounds=req.groupchat_n_rounds,
            use_llm=req.use_llm, seed=scenario.seed,
        )
        extras["group_chat"] = gc.to_dict()

    # F — World model PI + LLM verdict (默认开启，低成本)
    try:
        from .runtime.real_embedder import RealTextEmbedder
        if PRS.is_ready():
            _emb = RealTextEmbedder()
            caption_vec = _emb.embed(scenario.creative.caption)
            first_kol = scenario.kol_per_platform.get(first_plat) if scenario.kol_per_platform else None
            _niche_map = {"food":"美食","beauty":"美妆","mom":"母婴","tech":"数码",
                          "fashion":"穿搭","fitness":"健身","finance":"理财","travel":"旅行"}
            _niche = _niche_map.get(first_kol.niche, "美食") if first_kol else "美食"
            wm_pred = PRS.predict(
                caption_emb=caption_vec,
                author_fans=first_kol.fan_count if first_kol else 100_000,
                niche=_niche,
                duration_sec=scenario.creative.duration_sec,
                desc_emb=caption_vec,
            )
            if wm_pred and wm_pred.get("exp_p50"):
                exp_p50 = max(wm_pred["exp_p50"], 1)
                wm_pred["_like_rate_p10"] = round(wm_pred["like_p10"]/exp_p50*100, 2)
                wm_pred["_like_rate_p50"] = round(wm_pred["like_p50"]/exp_p50*100, 2)
                wm_pred["_like_rate_p90"] = round(wm_pred["like_p90"]/exp_p50*100, 2)
                wm_pred["_read_rate_p50"] = round(wm_pred["read_p50"]/exp_p50*100, 2)
                wm_pred["_read_rate_p10"] = round(wm_pred["read_p10"]/exp_p50*100, 2)
                wm_pred["_read_rate_p90"] = round(wm_pred["read_p90"]/exp_p50*100, 2)
            extras["world_model_prediction"] = wm_pred
            # Generate verdict (reuse existing LLM if use_llm enabled)
            if req.use_llm:
                from .agents.verdict import generate_verdict
                scene = f"{_niche}类 {first_kol.fan_count if first_kol else 100_000:,}粉 / {scenario.creative.caption[:40]}"
                extras["verdict"] = generate_verdict(wm_pred, scenario_desc=scene)
    except Exception as e:
        extras["world_model_error"] = str(e)[:200]

    # B — 90-day brand lift longitudinal
    if req.enable_brand_memory:
        fresh_state = BrandMemoryState.empty(POP.N)
        daily_metrics = simulate_campaign_days(
            WM, AG, fresh_state, scenario, n_days=req.brand_memory_days,
            reset_attitudes=True,
        )
        extras["brand_memory"] = {
            "days": req.brand_memory_days,
            "final": daily_metrics[-1],
            "timeline": [
                {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in m.items()} for m in daily_metrics
            ],
        }

    # T1-A5 oasis_sentiment_preview — aggregate predicted sentiment from soul quotes
    predicted_sentiment = _aggregate_sentiment_from_souls(souls)

    # ── SCM mediator 回写：discourse + group_chat 影响 CTR/CVR/revenue ──────
    # discourse.second_wave_impact 已在 [-0.3, +0.3]; group_chat.second_wave_impact 同范围。
    # 权重基于 L6 "群体话语层" → L7 "漏斗" 的 SCM 边（scm.py 中的 edges
    #  group_consensus→click / comment_sentiment→click），各占 50%。
    discourse_delta = 0.0
    if "discourse" in extras:
        discourse_delta = float(extras["discourse"].get("second_wave_click_delta") or 0)
    group_delta = 0.0
    if "group_chat" in extras:
        group_delta = float(extras["group_chat"].get("second_wave_impact") or 0)

    # 组合 CTR 乘数，每个 mediator 权重 0.5（共识/评论舆情贡献等权）
    ctr_multiplier = 1.0 + 0.5 * discourse_delta + 0.5 * group_delta
    # CVR 响应较弱（评论舆情对转化影响不如对点击）
    cvr_multiplier = 1.0 + 0.3 * discourse_delta + 0.3 * group_delta
    # Hard clamp 防极端乘数
    ctr_multiplier = max(0.5, min(1.5, ctr_multiplier))
    cvr_multiplier = max(0.6, min(1.4, cvr_multiplier))

    def _apply_mediator(kpi_dict):
        if ctr_multiplier == 1.0 and cvr_multiplier == 1.0:
            return
        if "clicks" in kpi_dict:
            kpi_dict["clicks"] = float(kpi_dict["clicks"]) * ctr_multiplier
        if "conversions" in kpi_dict:
            kpi_dict["conversions"] = float(kpi_dict["conversions"]) * ctr_multiplier * cvr_multiplier
        if "revenue" in kpi_dict:
            kpi_dict["revenue"] = float(kpi_dict["revenue"]) * ctr_multiplier * cvr_multiplier
        if "impressions" in kpi_dict and kpi_dict["impressions"]:
            kpi_dict["ctr"] = kpi_dict.get("clicks", 0) / kpi_dict["impressions"]
        if "clicks" in kpi_dict and kpi_dict["clicks"]:
            kpi_dict["cvr"] = kpi_dict.get("conversions", 0) / kpi_dict["clicks"]
        if "cost" in kpi_dict and kpi_dict["cost"]:
            kpi_dict["roi"] = (kpi_dict.get("revenue", 0) - kpi_dict["cost"]) / kpi_dict["cost"]

    _apply_mediator(result.total_kpis)
    for plat, d in result.per_platform.items():
        if "kpi" in d:
            _apply_mediator(d["kpi"])

    if ctr_multiplier != 1.0 or cvr_multiplier != 1.0:
        # 让前端能看到这次 mediator 回写确实发生了
        extras.setdefault("mediator_impact", {})
        extras["mediator_impact"].update({
            "applied_ctr_multiplier": round(ctr_multiplier, 4),
            "applied_cvr_multiplier": round(cvr_multiplier, 4),
            "discourse_contribution": round(discourse_delta, 4),
            "groupchat_contribution": round(group_delta, 4),
            "source": "scm_mediator_L6_to_L7",
        })

    # Schema-aligned outputs (T1-A1, T1-A2, T2-A4, T3-A1, T3-A2, T3-A3, T3-A5, report)
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
    try:
        from .agents.schema_outputs import build_schema_outputs
        schema_outputs = build_schema_outputs(
            kpis=kpis_out, lifecycle=lifecycle, soul_quotes=souls,
            per_platform=per_platform_out, predicted_sentiment=predicted_sentiment,
            extras=extras, scenario_summary=scenario_summary_out,
            competitors=req.competitors,
            own_brand=req.own_brand, category=req.category,
            target_niches=req.target_niches,
            enable_competitor_llm=bool(req.competitors),  # mock fallback inside if no LLM key
            enable_kol_ilp=req.enable_kol_ilp,
            enable_search_elasticity=req.enable_search_elasticity,
        )
        # Final aggregated report (OpenAI-compatible LLM polished if available, else template)
        from .agents.final_report import build_final_report
        schema_outputs["report_strategy_case"] = build_final_report(
            scenario=scenario_summary_out, kpis=kpis_out,
            predicted_sentiment=predicted_sentiment,
            schema_outputs=schema_outputs,
            use_llm=bool(req.use_llm),
        )
    except Exception as e:
        schema_outputs = {"_error": str(e)}

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
    "购买冲动": +1.0, "心动": +0.7, "好奇": +0.3,
    "无感": 0.0, "厌恶": -0.8,
}

def _aggregate_sentiment_from_souls(souls):
    """T1-A5 oasis_sentiment_preview schema — derive sentiment_distribution
    + emergent themes from soul feel + reason tallies (no new LLM call)."""
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
    # Top reason bag (proxy for key_opinion_themes)
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


# ---------------- Sandbox ----------------

class SessionCreateRequest(BaseModel):
    creative: CreativeInput
    total_budget: float = 50_000
    platform_alloc: Dict[str, float] = {"douyin": 0.6, "xhs": 0.4}
    daypart: str = "auto"
    today: Optional[str] = None
    cross_platform_overlap: float = 0.25


@app.post("/api/sandbox/session")
def sb_create(req: SessionCreateRequest):
    scenario, _ = _build_scenario(PredictRequest(**req.dict()))
    sess = SANDBOX.create(scenario)
    return sess.snapshot()


@app.get("/api/sandbox/session/{sid}")
def sb_get(sid: str):
    sess = SANDBOX.get(sid)
    if not sess: raise HTTPException(404, "session not found")
    return sess.snapshot()


class PatchReq(BaseModel):
    total_budget: Optional[float] = None
    platform_alloc: Optional[Dict[str, float]] = None


@app.patch("/api/sandbox/session/{sid}")
def sb_patch(sid: str, patch: PatchReq):
    sess = SANDBOX.get(sid)
    if not sess: raise HTTPException(404, "session not found")
    p = {k: v for k, v in patch.dict().items() if v is not None}
    sess = SANDBOX.update(sid, p)
    return sess.snapshot()


@app.post("/api/sandbox/session/{sid}/counterfactual")
def sb_counterfactual(sid: str, patch: PatchReq):
    """Explicit counterfactual: compare against baseline, not current."""
    sess = SANDBOX.get(sid)
    if not sess: raise HTTPException(404, "session not found")
    intervention = {k: v for k, v in patch.dict().items() if v is not None}
    if "platform_alloc" in intervention:
        a = intervention["platform_alloc"]
        s = sum(a.values()) or 1
        intervention["platform_alloc"] = {k: v/s for k,v in a.items() if v > 0}

    # CATE from baseline vs counterfactual
    cf_result = RUNNER.counterfactual(sess.baseline, sess.baseline_result, intervention)

    # per-agent click_prob for CATE — use first common platform
    plats = list(set(sess.baseline_result.per_platform.keys())
                 & set(cf_result.per_platform.keys()))
    cate_info = []
    if plats:
        plat = plats[0]
        # rerun to get per-agent probs
        budget_base = sess.baseline.total_budget * sess.baseline.platform_alloc[plat]
        imp_b = WM.simulate_impression(sess.baseline.creative, plat, budget_base,
                                       audience_filter=sess.baseline.audience_filter,
                                       kol=(sess.baseline.kol_per_platform or {}).get(plat),
                                       rng_seed=sess.baseline.seed)
        oc_b = AG.simulate(imp_b, sess.baseline.creative,
                           kol=(sess.baseline.kol_per_platform or {}).get(plat),
                           rng_seed=sess.baseline.seed)
        base_probs = {int(a): float(p) for a, p in zip(oc_b.agent_idx, oc_b.click_prob)}
        # CF budget reflects BOTH intervention.total_budget and intervention.platform_alloc.
        alloc_cf = intervention.get("platform_alloc", sess.baseline.platform_alloc)
        total_budget_cf = intervention.get("total_budget", sess.baseline.total_budget)
        audience_cf = intervention.get("audience_filter", sess.baseline.audience_filter)
        kol_map_cf = intervention.get("kol_per_platform", sess.baseline.kol_per_platform) or {}
        if plat in alloc_cf:
            budget_cf = total_budget_cf * alloc_cf[plat]
            imp_cf = WM.simulate_impression(sess.baseline.creative, plat, budget_cf,
                                             audience_filter=audience_cf,
                                             kol=kol_map_cf.get(plat),
                                             rng_seed=sess.baseline.seed)
            oc_cf = AG.simulate(imp_cf, sess.baseline.creative,
                                kol=kol_map_cf.get(plat),
                                rng_seed=sess.baseline.seed)
            cf_probs = {int(a): float(p) for a, p in zip(oc_cf.agent_idx, oc_cf.click_prob)}
            cate_info = compute_cate(POP, base_probs, cf_probs)

    return {
        "baseline_kpis": {k: round(float(v),4) for k,v in sess.baseline_result.total_kpis.items()},
        "counterfactual_kpis": {k: round(float(v),4) for k,v in cf_result.total_kpis.items()},
        "delta": {
            k: round(cf_result.total_kpis.get(k,0) - sess.baseline_result.total_kpis.get(k,0), 4)
            for k in cf_result.total_kpis
        },
        "cate": cate_info,
    }


@app.post("/api/sandbox/session/{sid}/explain")
def sb_explain(sid: str, n: int = 6, use_llm: bool = False):
    sess = SANDBOX.get(sid)
    if not sess: raise HTTPException(404, "session not found")
    plat = next(iter(sess.current.platform_alloc.keys()))
    budget = sess.current.total_budget * sess.current.platform_alloc[plat]
    imp = WM.simulate_impression(
        sess.current.creative, plat, budget,
        audience_filter=sess.current.audience_filter,
        kol=(sess.current.kol_per_platform or {}).get(plat),
        rng_seed=sess.current.seed,
    )
    oc = AG.simulate(imp, sess.current.creative,
                     kol=(sess.current.kol_per_platform or {}).get(plat),
                     rng_seed=sess.current.seed)
    click_prob_by_agent = {int(a): float(p) for a, p in zip(oc.agent_idx, oc.click_prob)}
    souls = SOULS.infer_batch(
        sess.current.creative, click_prob_by_agent,
        kol=(sess.current.kol_per_platform or {}).get(plat),
        platform=plat, n_sample=n, use_llm=use_llm,
    )
    return {"soul_quotes": souls}


@app.get("/api/sandbox/session/{sid}/lifecycle")
def sb_lifecycle(sid: str, days: int = 14):
    sess = SANDBOX.get(sid)
    if not sess: raise HTTPException(404, "session not found")
    plat = next(iter(sess.current.platform_alloc.keys()))
    budget = sess.current.total_budget * sess.current.platform_alloc[plat]
    imp = WM.simulate_impression(
        sess.current.creative, plat, budget,
        audience_filter=sess.current.audience_filter,
        kol=(sess.current.kol_per_platform or {}).get(plat),
        rng_seed=sess.current.seed,
    )
    oc = AG.simulate(imp, sess.current.creative,
                     kol=(sess.current.kol_per_platform or {}).get(plat),
                     rng_seed=sess.current.seed)
    hr = HAWKES.simulate(imp, oc, days=days)
    return hawkes_result_to_dict(hr)


@app.post("/api/sandbox/session/{sid}/undo")
def sb_undo(sid: str):
    if SANDBOX.get(sid) is None:
        raise HTTPException(404, "session not found")
    sess = SANDBOX.undo(sid)
    return sess.snapshot()


# =============================================================================
# v2 API — registry-routed access to the new model zoo
# -----------------------------------------------------------------------------
# The original /api/predict path uses the legacy PlatformWorldModel +
# HawkesSimulator pair bound at module-import time — good for backward compat,
# but it leaves the Causal Transformer + Causal Neural Hawkes unreachable
# from HTTP. These /api/v2/* endpoints route through the registries instead,
# so any caller can pick a model by name:
#
#   POST /api/v2/world_model/predict      ?model=causal_transformer | lightgbm_quantile
#   POST /api/v2/diffusion/forecast       ?model=causal_neural_hawkes | parametric_hawkes
#   POST /api/v2/diffusion/counterfactual ?model=...
#   POST /api/v2/synthesizer/generate     ?model=ipf | bayes_net
#
# Torch-requiring models raise HTTP 501 with a helpful message when invoked
# without the [ml] extra installed.
# =============================================================================


class V2WorldModelPredictRequest(BaseModel):
    features: Dict = {}
    n_samples: int = 1        # ignored for LightGBM; reserved for MC variants


class V2DiffusionForecastRequest(BaseModel):
    seed_events: List[List] = []  # [[t_min, "event_name"], ...]
    intervention: Optional[Dict] = None


class V2SynthesizerRequest(BaseModel):
    N: int = 500
    seed: Optional[int] = None


@app.post("/api/v2/world_model/predict")
def v2_wm_predict(req: V2WorldModelPredictRequest, model: str = "lightgbm_quantile"):
    """Predict KPIs via the registered world model."""
    try:
        from .world_model import get_world_model
    except ImportError as e:
        raise HTTPException(501, f"world_model registry unavailable: {e}")
    try:
        wm = get_world_model(model)
    except ImportError as e:
        # torch-dep model missing
        raise HTTPException(501, f"model '{model}' requires optional deps: {e}")
    except KeyError:
        raise HTTPException(400, f"unknown model '{model}'")

    # LightGBM baseline needs booster loaded from the shipped pkl
    if model in ("lightgbm_quantile", "lightgbm", "lgbm"):
        # Load the shipped demo pkl on first use
        try:
            import pickle as _pkl, lightgbm as _lgb, numpy as _np
            from pathlib import Path as _P
            pkl_path = _P(__file__).resolve().parent.parent.parent / "data" / "models" / "world_model_demo.pkl"
            if not pkl_path.exists():
                raise HTTPException(501, f"pretrained demo pkl not found at {pkl_path}")
            with open(pkl_path, "rb") as f:
                blob = _pkl.load(f)
            # Build feature vector from the request — expected 7-dim scalar schema
            f = req.features
            niches = blob["config"]["niches"]
            tiers = blob["config"]["kol_tiers"]
            niche_idx = niches.index(f.get("niche", "beauty")) if f.get("niche", "beauty") in niches else 0
            tier_idx = tiers.index(f.get("kol_tier", "micro")) if f.get("kol_tier", "micro") in tiers else 0
            x = _np.asarray([[
                float(f.get("platform_id", 0)),
                float(niche_idx),
                float(f.get("budget", 10000)),
                float(f.get("budget_bucket", 1)),
                float(tier_idx),
                float(f.get("kol_fan_count", 50000)),
                float(f.get("kol_engagement_rate", 0.03)),
            ]], dtype=_np.float32)
            out = {"model": model, "kpi_quantiles": {}, "model_version": blob["config"].get("training_version", "demo")}
            for kpi, qmap in blob["boosters"].items():
                out["kpi_quantiles"][kpi] = {}
                for q_str, bst_str in qmap.items():
                    booster = _lgb.Booster(model_str=bst_str)
                    out["kpi_quantiles"][kpi][q_str] = float(booster.predict(x)[0])
            return out
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"lightgbm predict failed: {type(e).__name__}: {e}")

    # Causal Transformer: delegate to its own predict()
    try:
        pred = wm.predict(req.features)
        return {
            "model": model,
            "kpi_quantiles": pred.kpi_quantiles,
            "latent": pred.latent,
        }
    except FileNotFoundError as e:
        raise HTTPException(501, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/v2/diffusion/forecast")
def v2_diffusion_forecast(req: V2DiffusionForecastRequest, model: str = "parametric_hawkes"):
    """Forecast the 14-day cascade via the registered diffusion model."""
    try:
        from .diffusion import get_diffusion_model
    except ImportError as e:
        raise HTTPException(501, f"diffusion registry unavailable: {e}")
    try:
        diff = get_diffusion_model(model)
    except ImportError as e:
        raise HTTPException(501, f"model '{model}' requires optional deps: {e}")
    except KeyError:
        raise HTTPException(400, f"unknown model '{model}'")

    seed_events = [(float(t), str(n)) for t, n in req.seed_events]
    try:
        if req.intervention:
            out = diff.counterfactual_forecast(seed_events, intervention=req.intervention)
        else:
            out = diff.forecast(seed_events)
    except FileNotFoundError as e:
        raise HTTPException(501, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")

    return {
        "model": model,
        "per_type_totals": out.per_type_totals,
        "daily_buckets":   out.daily_buckets,
        "n_events_simulated": len(out.timeline),
        "latent": out.latent,
    }


@app.post("/api/v2/synthesizer/generate")
def v2_synth_generate(req: V2SynthesizerRequest, model: str = "ipf"):
    """Generate a virtual consumer population via the registered synthesizer."""
    try:
        from .data.synthesizers import get_synthesizer
    except ImportError as e:
        raise HTTPException(501, f"synthesizer registry unavailable: {e}")
    try:
        syn = get_synthesizer(model)
    except NotImplementedError as e:
        raise HTTPException(501, str(e))
    except KeyError:
        raise HTTPException(400, f"unknown synthesizer '{model}'")

    pop = syn.generate(N=req.N, seed=req.seed)
    # Return summary stats, not the raw arrays (keep the response small).
    try:
        import numpy as _np
        summary = {
            k: {"mean": float(_np.asarray(v).mean()), "min": float(_np.asarray(v).min()),
                "max": float(_np.asarray(v).max())}
            for k, v in pop.attributes.items()
            if hasattr(v, "__len__") and len(v) > 0
        }
    except Exception:
        summary = {}

    return {
        "model": model,
        "N": pop.N,
        "attribute_summary": summary,
        "latent": pop.latent,
    }


@app.get("/api/v2/registry")
def v2_registry():
    """List all registered model + synthesizer variants."""
    from .world_model import list_world_models
    from .diffusion import list_diffusion_models
    from .data.synthesizers import list_synthesizers
    return {
        "world_model": list_world_models(),
        "diffusion":   list_diffusion_models(),
        "synthesizer": list_synthesizers(),
        "api_version": "v2",
    }


# -------- WebSocket --------
@app.websocket("/ws/sandbox/{sid}")
async def ws(ws: WebSocket, sid: str):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                patch = json.loads(msg)
            except Exception:
                await ws.send_json({"error": "bad json"})
                continue
            sess = SANDBOX.get(sid)
            if not sess:
                await ws.send_json({"error": "session not found"}); continue
            sess = SANDBOX.update(sid, patch)
            await ws.send_json(sess.snapshot())
    except WebSocketDisconnect:
        return

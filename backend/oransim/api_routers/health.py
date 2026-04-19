"""Health / society / LLM / world / world_model router.

Groups the "infra-ish" endpoints that either ping liveness
(``/api/health``, ``/api/llm/status``) or expose the platform's baseline
state (``/api/society/sample`` for the star-map, ``/api/world*`` for the
macro event feed, ``/api/world_model/*`` for the XHS PRS).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import api_state
from ..agents.soul_llm import llm_available, llm_info
from ..data.world_events import get_world_state, refresh_world_state
from ..platforms.xhs.prs import PRS

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health():
    from ..agents import async_pool, llm_dedup, stream_memory

    return {
        "status": "ok",
        "population": api_state.POP.N,
        "souls": len(api_state.SOULS.personas),
        "kols": len(api_state.KOLS),
        "llm": llm_info(),
        "optimizations": {
            "dedup": llm_dedup.dedup_stats(),
            "async_pool": async_pool.pool_stats(),
            "stream_memory": stream_memory.memory_stats(),
        },
    }


@router.get("/api/society/sample")
def society_sample(n: int = 10000):
    """Return (x, y, stance) for N agent particles for UI rendering.

    Uses interest_emb PCA-2D + stance. Enterprise mode: up to 1M samples.
    """
    import numpy as np

    n = min(n, api_state.POP.N)
    rng = np.random.default_rng(0)
    all_souls = np.asarray(list(api_state.SOULS.idx), dtype=np.int64)
    if len(all_souls) >= n:
        idx = rng.choice(all_souls, size=n, replace=False)
    else:
        non_souls = np.setdiff1d(
            np.arange(api_state.POP.N, dtype=np.int64), all_souls, assume_unique=False
        )
        remainder = rng.choice(non_souls, size=n - len(all_souls), replace=False)
        idx = np.concatenate([all_souls, remainder])
        rng.shuffle(idx)
    interest = api_state.POP.interest[idx]
    x = interest[:, 0]
    y = interest[:, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    tier = api_state.POP.city_idx[idx].tolist()
    gender = api_state.POP.gender_idx[idx].tolist()
    age = api_state.POP.age_idx[idx].tolist()
    soul_ids = set(int(i) for i in api_state.SOULS.idx)
    is_soul = [int(int(i) in soul_ids) for i in idx]
    return {
        "n_total": api_state.POP.N,
        "n_sampled": n,
        "n_llm_souls_highlighted": sum(is_soul),
        "points": [
            {
                "x": round(float(x[i]), 3),
                "y": round(float(y[i]), 3),
                "tier": tier[i],
                "gender": gender[i],
                "age": age[i],
                "is_soul": is_soul[i],
                "pid": int(idx[i]),
            }
            for i in range(n)
        ],
    }


@router.get("/api/llm/status")
def llm_status():
    return {"available": llm_available(), "info": llm_info()}


@router.get("/api/world")
def world_get():
    """Current cached world state — broad-spectrum events affecting consumer behavior."""
    return get_world_state()


@router.post("/api/world/refresh")
def world_refresh():
    """Force LLM to re-curate today's top events (cron this every 6h in prod)."""
    return refresh_world_state(force=True)


@router.get("/api/world_model/info")
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


@router.post("/api/world_model/predict")
def world_model_predict(req: WMPredictReq):
    """XHS platform world model prediction.

    Returns predicted (exp, read, like, coll, comm) counts — from LightGBM
    trained on real XHS note engagement data.
    """
    if not PRS.is_ready():
        raise HTTPException(503, "world model not trained yet; run train_world_model.py")
    from ..runtime.real_embedder import RealTextEmbedder

    emb = RealTextEmbedder()
    caption_vec = emb.embed(req.title)
    desc_vec = emb.embed(req.desc or req.title)
    pred = PRS.predict(
        caption_emb=caption_vec,
        author_fans=req.author_fans,
        niche=req.niche,
        duration_sec=req.duration_sec,
        desc_emb=desc_vec,
        topics=req.topics,
        has_img=req.has_img,
        img_count=req.img_count,
        post_type=req.post_type,
    )
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

    if req.with_verdict:
        from ..agents.verdict import generate_verdict

        scene = f"{req.niche}类 {req.author_fans:,}粉 / {req.title[:40]}"
        result["verdict"] = generate_verdict(pred, scenario_desc=scene)

    return result

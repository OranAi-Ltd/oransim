"""v2 registry-routed API — ``/api/v2/*``.

The original ``/api/predict`` path uses the legacy ``PlatformWorldModel``
+ ``HawkesSimulator`` pair bound at module-import time — good for
backward compat, but it leaves the Causal Transformer + Causal Neural
Hawkes unreachable from HTTP. These ``/api/v2/*`` endpoints route
through the registries instead, so any caller can pick a model by name:

  POST /api/v2/world_model/predict      ?model=causal_transformer | lightgbm_quantile
  POST /api/v2/diffusion/forecast       ?model=causal_neural_hawkes | parametric_hawkes
  POST /api/v2/synthesizer/generate     ?model=ipf | bayes_net

Torch-requiring models raise HTTP 501 with a helpful message when
invoked without the ``[ml]`` extra installed.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["v2"])


class V2WorldModelPredictRequest(BaseModel):
    features: dict = {}
    n_samples: int = 1  # ignored for LightGBM; reserved for MC variants


class V2DiffusionForecastRequest(BaseModel):
    seed_events: list[list] = []  # [[t_min, "event_name"], ...]
    intervention: dict | None = None


class V2SynthesizerRequest(BaseModel):
    N: int = 500
    seed: int | None = None


@router.post("/api/v2/world_model/predict")
def v2_wm_predict(req: V2WorldModelPredictRequest, model: str = "lightgbm_quantile"):
    """Predict KPIs via the registered world model."""
    try:
        from ..world_model import get_world_model
    except ImportError as e:
        raise HTTPException(501, f"world_model registry unavailable: {e}") from e
    try:
        wm = get_world_model(model)
    except ImportError as e:
        raise HTTPException(501, f"model '{model}' requires optional deps: {e}") from e
    except KeyError:
        raise HTTPException(400, f"unknown model '{model}'") from None

    if model in ("lightgbm_quantile", "lightgbm", "lgbm"):
        try:
            import pickle as _pkl
            from pathlib import Path as _P

            import lightgbm as _lgb
            import numpy as _np

            pkl_path = (
                _P(__file__).resolve().parent.parent.parent.parent
                / "data"
                / "models"
                / "world_model_demo.pkl"
            )
            if not pkl_path.exists():
                raise HTTPException(501, f"pretrained demo pkl not found at {pkl_path}")
            with open(pkl_path, "rb") as f:
                blob = _pkl.load(f)
            f = req.features
            niches = blob["config"]["niches"]
            tiers = blob["config"]["kol_tiers"]
            niche_idx = (
                niches.index(f.get("niche", "beauty")) if f.get("niche", "beauty") in niches else 0
            )
            tier_idx = (
                tiers.index(f.get("kol_tier", "micro"))
                if f.get("kol_tier", "micro") in tiers
                else 0
            )
            scalar = _np.asarray(
                [
                    float(f.get("platform_id", 0)),
                    float(niche_idx),
                    float(f.get("budget", 10000)),
                    float(f.get("budget_bucket", 1)),
                    float(tier_idx),
                    float(f.get("kol_fan_count", 50000)),
                    float(f.get("kol_engagement_rate", 0.03)),
                ],
                dtype=_np.float32,
            )
            if blob.get("pca") is not None:
                from ..runtime.real_embedder import RealTextEmbedder
                from ..scripts_helpers import caption_for_demo_pkl  # type: ignore

                embedder = RealTextEmbedder()
                caption = caption_for_demo_pkl(f, blob["config"])
                emb = embedder.embed(caption).astype(_np.float32)
                comps = _np.asarray(blob["pca"]["components"], dtype=_np.float32)
                mean = _np.asarray(blob["pca"]["mean"], dtype=_np.float32)
                emb_pca = (emb - mean) @ comps.T
                scalar = _np.concatenate([scalar, emb_pca.astype(_np.float32)])
            x = scalar.reshape(1, -1)
            out = {
                "model": model,
                "kpi_quantiles": {},
                "model_version": blob["config"].get("training_version", "demo"),
            }
            for kpi, qmap in blob["boosters"].items():
                out["kpi_quantiles"][kpi] = {}
                for q_str, bst_str in qmap.items():
                    booster = _lgb.Booster(model_str=bst_str)
                    out["kpi_quantiles"][kpi][q_str] = float(booster.predict(x)[0])
            return out
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"lightgbm predict failed: {type(e).__name__}: {e}") from e

    try:
        pred = wm.predict(req.features)
        return {
            "model": model,
            "kpi_quantiles": pred.kpi_quantiles,
            "latent": pred.latent,
        }
    except FileNotFoundError as e:
        raise HTTPException(501, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


@router.post("/api/v2/diffusion/forecast")
def v2_diffusion_forecast(req: V2DiffusionForecastRequest, model: str = "parametric_hawkes"):
    """Forecast the 14-day cascade via the registered diffusion model."""
    try:
        from ..diffusion import get_diffusion_model
    except ImportError as e:
        raise HTTPException(501, f"diffusion registry unavailable: {e}") from e
    try:
        diff = get_diffusion_model(model)
    except ImportError as e:
        raise HTTPException(501, f"model '{model}' requires optional deps: {e}") from e
    except KeyError:
        raise HTTPException(400, f"unknown model '{model}'") from None

    seed_events = [(float(t), str(n)) for t, n in req.seed_events]
    try:
        if req.intervention:
            out = diff.counterfactual_forecast(seed_events, intervention=req.intervention)
        else:
            out = diff.forecast(seed_events)
    except FileNotFoundError as e:
        raise HTTPException(501, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e

    return {
        "model": model,
        "per_type_totals": out.per_type_totals,
        "daily_buckets": out.daily_buckets,
        "n_events_simulated": len(out.timeline),
        "latent": out.latent,
    }


@router.post("/api/v2/synthesizer/generate")
def v2_synth_generate(req: V2SynthesizerRequest, model: str = "ipf"):
    """Generate a virtual consumer population via the registered synthesizer."""
    try:
        from ..data.synthesizers import get_synthesizer
    except ImportError as e:
        raise HTTPException(501, f"synthesizer registry unavailable: {e}") from e
    try:
        syn = get_synthesizer(model)
    except NotImplementedError as e:
        raise HTTPException(501, str(e)) from e
    except KeyError:
        raise HTTPException(400, f"unknown synthesizer '{model}'") from None

    pop = syn.generate(N=req.N, seed=req.seed)
    try:
        import numpy as _np

        summary = {
            k: {
                "mean": float(_np.asarray(v).mean()),
                "min": float(_np.asarray(v).min()),
                "max": float(_np.asarray(v).max()),
            }
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


@router.get("/api/v2/registry")
def v2_registry():
    """List all registered model + synthesizer variants."""
    from ..data.synthesizers import list_synthesizers
    from ..diffusion import list_diffusion_models
    from ..world_model import list_world_models

    return {
        "world_model": list_world_models(),
        "diffusion": list_diffusion_models(),
        "synthesizer": list_synthesizers(),
        "api_version": "v2",
    }

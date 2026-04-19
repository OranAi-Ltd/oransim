"""Unified Embedding Bus router — ``/api/ueb/*``.

Exposes the UEB's registered sources, indexing, and dynamic source
registration. BUS itself is a module-level singleton in
``oransim.runtime.embedding_bus`` so we import it directly rather than
routing through ``api_state``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..runtime.embedding_bus import BUS

router = APIRouter(tags=["ueb"])


@router.get("/api/ueb/sources")
def ueb_sources():
    """List all registered embedders. Plug a new data source = register a new embedder."""
    return {"sources": BUS.list_sources()}


@router.get("/api/ueb/stats")
def ueb_stats():
    """Total data items across all sources + estimated generalization error bound.

    Visualizes the 'more data → more accurate' guarantee:
      err ≤ 0.3 / sqrt(N), so 4× more data halves the error bound.
    """
    return BUS.learning_stats()


class IndexRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    source: str
    items: list  # accept any list contents


@router.post("/api/ueb/index")
def ueb_index(req: IndexRequest):
    """Embed a batch of items into a source's vector index.

    This is how new data flows in — same endpoint regardless of modality.
    Downstream models pick up the new vectors automatically.
    """
    vecs = BUS.index(req.source, req.items)
    return {
        "source": req.source,
        "n_added": len(vecs),
        "total_in_source": BUS.list_sources(),
        "scaling_law_now": BUS.learning_stats()["scaling_law_estimate"],
    }


class RegisterRequest(BaseModel):
    source: str
    modality: str = "text"  # text/tabular/timeseries/geo/event/image/audio
    notes: str = ""
    in_dim: int | None = None  # required for tabular


@router.post("/api/ueb/register")
def ueb_register(req: RegisterRequest):
    """Register a NEW data source dynamically — without code changes.

    e.g. {'source':'weibo_brand_mentions','modality':'text','notes':'品牌微博提及'}
    """
    from ..runtime.embedding_bus import (
        CategoricalEmbedder,
        EventEmbedder,
        GeoEmbedder,
        HashTextEmbedder,
        TabularEmbedder,
        TimeSeriesEmbedder,
    )

    embedder_map = {
        "text": HashTextEmbedder,
        "categorical": CategoricalEmbedder,
        "timeseries": TimeSeriesEmbedder,
        "geo": GeoEmbedder,
        "event": EventEmbedder,
    }
    if req.modality == "tabular":
        emb = TabularEmbedder(in_dim=req.in_dim or 16)
    elif req.modality in embedder_map:
        emb = embedder_map[req.modality]()
    else:
        raise HTTPException(400, f"unsupported modality: {req.modality}")
    BUS.register(req.source, emb, notes=req.notes)
    return {"registered": req.source, "info": emb.info()}

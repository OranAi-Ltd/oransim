"""Platform adapter router — ``/api/adapters*``.

Lazy-built registry of the 4 MVP ``PlatformAdapter`` implementations.
The legacy XHS pipeline inside ``/api/predict`` still uses
``PlatformWorldModel`` — these endpoints expose the new adapter surface
so users can call platform-specific simulations directly.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["adapters"])

_ADAPTER_CACHE: dict = {}


def _build_adapter_registry():
    from ..platforms.douyin.adapter import DouyinAdapter
    from ..platforms.instagram.adapter import InstagramAdapter
    from ..platforms.tiktok.adapter import TikTokAdapter
    from ..platforms.youtube_shorts.adapter import YouTubeShortsAdapter

    return {
        "tiktok": TikTokAdapter(),
        "douyin": DouyinAdapter(),
        "instagram": InstagramAdapter(),
        "youtube_shorts": YouTubeShortsAdapter(),
    }


def _get_adapter(platform_id: str):
    if not _ADAPTER_CACHE:
        _ADAPTER_CACHE.update(_build_adapter_registry())
    if platform_id not in _ADAPTER_CACHE:
        raise HTTPException(
            404,
            f"unknown platform_id {platform_id!r}; "
            f"available: {sorted(_ADAPTER_CACHE)} + 'xhs' (legacy)",
        )
    return _ADAPTER_CACHE[platform_id]


@router.get("/api/adapters")
def list_platform_adapters():
    """List PlatformAdapter instances (status + config)."""
    if not _ADAPTER_CACHE:
        _ADAPTER_CACHE.update(_build_adapter_registry())
    out = []
    for pid, adapter in _ADAPTER_CACHE.items():
        cfg = adapter.config
        out.append(
            {
                "platform_id": pid,
                "adapter_class": type(adapter).__name__,
                "status": "mvp",
                "data_provider": (
                    type(adapter.data_provider).__name__ if adapter.data_provider else None
                ),
                "config": {
                    "cpm_usd": getattr(cfg, "cpm_usd", None),
                    "base_ctr": getattr(cfg, "base_ctr", None),
                    "base_cvr": getattr(cfg, "base_cvr", None),
                    "duration_half_sec": getattr(cfg, "duration_half_sec", None),
                },
            }
        )
    out.append(
        {
            "platform_id": "xhs",
            "adapter_class": "PlatformWorldModel (legacy)",
            "status": "v1",
            "data_provider": "synthetic / CSV / JSON / OpenAPI",
            "config": {"note": "legacy pipeline · runs inside /api/predict"},
        }
    )
    return {"platforms": out, "total": len(out)}


class AdapterImpressionRequest(BaseModel):
    caption: str = "demo creative"
    duration_sec: float = 15.0
    budget: float = 50_000.0
    reference_budget: float = 50_000.0


@router.post("/api/adapters/{platform_id}/simulate_impression")
def adapter_simulate_impression(platform_id: str, req: AdapterImpressionRequest):
    """Run a single adapter.simulate_impression call (no agent-level draw)."""
    adapter = _get_adapter(platform_id)
    creative = type("Creative", (), {"caption": req.caption, "duration_sec": req.duration_sec})()
    result = adapter.simulate_impression(
        creative=creative,
        budget=float(req.budget),
        reference_budget=float(req.reference_budget),
    )
    return {"platform_id": platform_id, **result}

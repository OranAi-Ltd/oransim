"""Douyin platform adapter (MVP).

v0.2 status: MVP — synthetic-data-driven, calibrated to Greater-China
priors distinct from TikTok's global profile.
"""

from .adapter import DouyinAdapter, DouyinAdapterConfig
from .providers.synthetic import DouyinSyntheticProvider
from .prs import PRS, DouyinPRS
from .recsys_rl import DouyinRecSysRLSimulator
from .world_model_legacy import DouyinWorldModel

__all__ = [
    "PRS",
    "DouyinAdapter",
    "DouyinAdapterConfig",
    "DouyinPRS",
    "DouyinRecSysRLSimulator",
    "DouyinSyntheticProvider",
    "DouyinWorldModel",
]

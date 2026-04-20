"""Instagram Reels adapter (MVP).

v0.2 status: MVP — synthetic-data-driven. Emphasis on Reels rather than
Feed posts since Reels is now Meta's primary short-video surface.
"""

from .adapter import InstagramAdapter, InstagramAdapterConfig
from .providers.synthetic import InstagramSyntheticProvider
from .prs import PRS, InstagramPRS
from .recsys_rl import InstagramRecSysRLSimulator
from .world_model_legacy import InstagramWorldModel

__all__ = [
    "PRS",
    "InstagramAdapter",
    "InstagramAdapterConfig",
    "InstagramPRS",
    "InstagramRecSysRLSimulator",
    "InstagramSyntheticProvider",
    "InstagramWorldModel",
]

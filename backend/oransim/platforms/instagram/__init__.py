"""Instagram Reels adapter (MVP).

v0.2 status: MVP — synthetic-data-driven. Emphasis on Reels rather than
Feed posts since Reels is now Meta's primary short-video surface.
"""

from .adapter import InstagramAdapter, InstagramAdapterConfig
from .providers.synthetic import InstagramSyntheticProvider

__all__ = ["InstagramAdapter", "InstagramAdapterConfig", "InstagramSyntheticProvider"]

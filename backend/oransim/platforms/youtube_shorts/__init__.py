"""YouTube Shorts adapter (MVP).

v0.2 status: MVP — synthetic-data-driven. YouTube Shorts inherits
YouTube's search + recommendation infrastructure plus a dedicated
short-form feed, giving it structurally longer discovery tails than
pure-feed platforms.
"""

from .adapter import YouTubeShortsAdapter, YouTubeShortsAdapterConfig
from .providers.synthetic import YouTubeShortsSyntheticProvider

__all__ = [
    "YouTubeShortsAdapter",
    "YouTubeShortsAdapterConfig",
    "YouTubeShortsSyntheticProvider",
]

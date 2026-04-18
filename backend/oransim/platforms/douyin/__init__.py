"""Douyin platform adapter (MVP).

v0.2 status: MVP — synthetic-data-driven, calibrated to Greater-China
priors distinct from TikTok's global profile.
"""

from .adapter import DouyinAdapter, DouyinAdapterConfig
from .providers.synthetic import DouyinSyntheticProvider

__all__ = ["DouyinAdapter", "DouyinAdapterConfig", "DouyinSyntheticProvider"]

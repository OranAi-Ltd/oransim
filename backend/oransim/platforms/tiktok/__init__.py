"""TikTok platform adapter (MVP).

v0.2 status: MVP — synthetic-data-driven. Enough to plug into
``oransim.api`` and return realistic per-platform KPI predictions.
Next: production adapter against a real TikTok Research API /
third-party panel (roadmap v0.5).
"""

from .adapter import TikTokAdapter, TikTokAdapterConfig
from .providers.synthetic import TikTokSyntheticProvider

__all__ = ["TikTokAdapter", "TikTokAdapterConfig", "TikTokSyntheticProvider"]

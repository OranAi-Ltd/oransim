"""YouTube Shorts cold-start → breakout simulator.

Inherits the TikTok 6-round geometric template but uses a slightly
lower breakout threshold (CTR > 2.5 %) because Shorts' subscribe-graph
lift boosts same-channel re-impressions earlier, so the "distribution
widens" check should fire sooner in absolute CTR terms.
"""

from __future__ import annotations

from ...data.creatives import Creative
from ...data.kols import KOL
from ..tiktok.recsys_rl import TikTokRecSysRLSimulator
from ..xhs.recsys_rl import RecSysRLReport
from .world_model_legacy import YouTubeShortsWorldModel

__all__ = ["YouTubeShortsRecSysRLSimulator", "RecSysRLReport"]


class YouTubeShortsRecSysRLSimulator(TikTokRecSysRLSimulator):
    """Shorts-calibrated cold-start breakout simulator."""

    def __init__(self, world_model: YouTubeShortsWorldModel):
        super().__init__(world_model)

    def simulate(
        self,
        creative: Creative,
        platform: str = "youtube_shorts",
        total_budget: float = 0.0,
        audience_filter=None,
        kol: KOL | None = None,
        n_rounds: int = 6,
        lr: float = 0.3,
        breakout_threshold: float = 0.025,
        seed: int = 0,
    ) -> RecSysRLReport:
        return super().simulate(
            creative,
            platform=platform,
            total_budget=total_budget,
            audience_filter=audience_filter,
            kol=kol,
            n_rounds=n_rounds,
            lr=lr,
            breakout_threshold=breakout_threshold,
            seed=seed,
        )

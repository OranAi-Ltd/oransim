"""Douyin FYP cold-start → breakout simulator.

Thin subclass of :class:`TikTokRecSysRLSimulator` that:

  - Routes through ``DouyinWorldModel`` (so the livestream boost +
    CN duration tuning flow into the per-round engagement proxy), and
  - Uses ``budget_to_impressions("douyin")`` for CPM math instead of
    TikTok's USD-normalized CPM.

All other FYP dynamics (6 geometric rounds, 2.8 % breakout threshold,
wide exploration noise) are inherited unchanged — the ByteDance
recommendation stack is shared between the two products.
"""

from __future__ import annotations

from ..tiktok.recsys_rl import TikTokRecSysRLSimulator
from ..xhs.recsys_rl import RecSysRLReport
from .world_model_legacy import DouyinWorldModel

__all__ = ["DouyinRecSysRLSimulator", "RecSysRLReport"]


class DouyinRecSysRLSimulator(TikTokRecSysRLSimulator):
    """Douyin-calibrated cold-start breakout simulator."""

    def __init__(self, world_model: DouyinWorldModel):
        super().__init__(world_model)

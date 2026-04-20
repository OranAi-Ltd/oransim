"""Instagram Reels cold-start → breakout simulator.

Shares TikTok's 6-round geometric cold-start shape with one
Reels-specific knob exposed today:

  - **Higher breakout threshold** (CTR > 3.2 % vs TikTok 2.8 %) — Reels
    leans on save/share in addition to CTR so the distribution-widening
    bar is slightly higher.

The narrower exploration-noise window that TikTok uses (``[0.5, 1.5]``
→ want ``[0.6, 1.4]`` for Reels) is not parameterized on the parent
class yet — wiring that up is tracked under the "expose scoring knobs
on RecSysRLSimulator" v0.3 follow-up. Until then Reels uses TikTok's
noise, which slightly over-estimates early-round variance.

Everything else (round fractions, weight-update shape) is inherited
from :class:`TikTokRecSysRLSimulator`.
"""

from __future__ import annotations

from ...data.creatives import Creative
from ...data.kols import KOL
from ..tiktok.recsys_rl import TikTokRecSysRLSimulator
from ..xhs.recsys_rl import RecSysRLReport
from .world_model_legacy import InstagramWorldModel

__all__ = ["InstagramRecSysRLSimulator", "RecSysRLReport"]


class InstagramRecSysRLSimulator(TikTokRecSysRLSimulator):
    """Reels-calibrated cold-start breakout simulator."""

    def __init__(self, world_model: InstagramWorldModel):
        super().__init__(world_model)

    def simulate(
        self,
        creative: Creative,
        platform: str = "instagram",
        total_budget: float = 0.0,
        audience_filter=None,
        kol: KOL | None = None,
        n_rounds: int = 6,
        lr: float = 0.3,
        breakout_threshold: float = 0.032,
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

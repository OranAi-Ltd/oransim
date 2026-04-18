"""D. Cross-platform identity resolution + cannibalization.

Same virtual user appears on multiple platforms; exposure on platform A induces
fatigue on platform B's subsequent exposure of the same agent.

Metrics produced:
  total_impressions      sum over all platforms (may double-count same user)
  unique_reach           how many DISTINCT users got impressed
  cannibalization        total - unique (wasted duplicate spend)
  incremental_per_plat   per-platform NEW user reach (not seen on earlier plats)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.creatives import Creative
from ..data.kols import KOL
from ..platforms.xhs.world_model_legacy import ImpressionResult, PlatformWorldModel


@dataclass
class CrossPlatformReach:
    total_impressions: float
    unique_reach: int
    cannibalization: int  # duplicate exposures
    exposure_count: np.ndarray  # (N_pop,) times each agent saw the ad
    per_platform_incremental: dict[str, int]  # plat → new users on this plat
    per_platform_duplicate: dict[str, int]
    max_frequency: int
    avg_frequency: float  # mean exposures for reached users


def simulate_cross_platform(
    wm: PlatformWorldModel,
    creative: Creative,
    platform_alloc: dict[str, float],
    total_budget: float,
    pop_size: int,
    audience_filter=None,
    kol_per_platform: dict[str, KOL] = None,
    seed: int = 0,
) -> tuple[dict[str, ImpressionResult], CrossPlatformReach]:
    """Run impression simulation sequentially per platform; track cumulative reach.

    Returns per-platform impression results (unchanged from single-platform) plus
    cross-platform reach metrics.
    """
    per_platform: dict[str, ImpressionResult] = {}
    exposure_count = np.zeros(pop_size, dtype=np.int32)
    per_platform_incremental: dict[str, int] = {}
    per_platform_duplicate: dict[str, int] = {}

    seen_before = np.zeros(pop_size, dtype=bool)
    for plat, frac in platform_alloc.items():
        if frac <= 0:
            continue
        budget = total_budget * frac
        kol = (kol_per_platform or {}).get(plat)
        imp = wm.simulate_impression(
            creative,
            plat,
            budget,
            audience_filter=audience_filter,
            kol=kol,
            rng_seed=seed * 1000 + hash(plat) % 1000,
        )
        per_platform[plat] = imp

        idx = imp.agent_idx
        mask_dup = seen_before[idx]
        mask_new = ~mask_dup
        per_platform_incremental[plat] = int(mask_new.sum())
        per_platform_duplicate[plat] = int(mask_dup.sum())

        exposure_count[idx] += 1
        seen_before[idx] = True

    unique_reach = int(seen_before.sum())
    total_impressions = int(exposure_count.sum())
    cannibalization = total_impressions - unique_reach
    reached_mask = exposure_count > 0
    avg_freq = float(exposure_count[reached_mask].mean()) if reached_mask.any() else 0.0
    max_freq = int(exposure_count.max()) if reached_mask.any() else 0

    return per_platform, CrossPlatformReach(
        total_impressions=float(total_impressions),
        unique_reach=unique_reach,
        cannibalization=cannibalization,
        exposure_count=exposure_count,
        per_platform_incremental=per_platform_incremental,
        per_platform_duplicate=per_platform_duplicate,
        max_frequency=max_freq,
        avg_frequency=avg_freq,
    )


def fatigue_lift(
    exposure_count: np.ndarray, agent_idx: np.ndarray, halflife: float = 2.0
) -> np.ndarray:
    """Multiplier to click_logit: 1st exposure = 1.0, 2nd = 0.7, 3rd = 0.5 ...
    Returns (len(agent_idx),) array of multipliers in (0, 1]."""
    freq = exposure_count[agent_idx].astype(np.float32)
    # 1st view: freq=1 (no penalty).  Higher exposures get diminishing lift.
    return np.power(0.5, np.clip(freq - 1, 0, 10) / halflife)

"""Platform-level priors: CPM, ECPM baselines, audience reach dynamics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class PlatformConfig:
    name: str
    cpm_cny: float          # cost per 1000 impressions
    conversion_cost: float  # cost per action estimate
    cold_start_days: float
    algo_diversity: float   # 0..1 : how broadly the recommender explores
    audience_skew: Dict[str, float]  # soft biases applied on top of agent platform_activity

PLATFORMS: Dict[str, PlatformConfig] = {
    "douyin": PlatformConfig(
        name="douyin",
        cpm_cny=35.0, conversion_cost=4.5,
        cold_start_days=0.5, algo_diversity=0.85,
        audience_skew={},
    ),
    "xhs": PlatformConfig(
        name="xhs",
        cpm_cny=48.0, conversion_cost=3.2,
        cold_start_days=1.2, algo_diversity=0.65,
        audience_skew={"female_boost": 1.3, "tier1_boost": 1.15},
    ),
}


def budget_to_impressions(budget_cny: float, platform: str) -> float:
    cfg = PLATFORMS[platform]
    return 1000.0 * budget_cny / cfg.cpm_cny

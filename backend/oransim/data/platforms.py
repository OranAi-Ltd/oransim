"""Platform-level priors: CPM, ECPM baselines, audience reach dynamics.

Audience-skew data sources
--------------------------
``audience_skew`` values below are **derived from public industry data**
(not free-hand heuristics). Each multiplier is ``platform_share /
national_baseline``, clamped to ``[0.6, 1.8]``.

National baselines (see ``data/population.py``):
  - gender:      女 48.7 % / 男 51.3 %
  - city tier:   tier1+2 = 24 % (一线 9 + 新一线/二线 15)
  - age young:   15-44 = 52 % · age senior: 55+ = 30 %

Per-platform sources:
  - 抖音 Douyin:       QuestMobile《2024 年新媒体生态盘点》(Dec 2024)
                       + Statista 2024 gender. MAU 748M (2024-09). Tier
                       breakdown directional from QuestMobile narrative.
  - 小红书 XHS:        QuestMobile 同上 · 最 cited 的一家。MAU 242M,
                       女 65 % / tier1+2 55.4 % / ≤24岁 34 %
                       (QuestMobile verbatim cells).
  - TikTok (global):   DataReportal (Jan 2025) · ad-audience 基础,
                       1.59B reach, 男 55.7 % / 女 44.3 %, 18-24
                       30.7 % + 25-34 35.3 %. City-tier N/A.
  - 哔哩哔哩 Bilibili: Bilibili 2024 Q4 财报（341M MAU）+ QuestMobile
                       定性数据 · 16-35 占 62.25 %, 性别比 57/43
                       interpolated。
  - 快手 Kuaishou:     Kuaishou 2024 Q4 公告（714.8M MAU, +3.3% YoY）
                       + Statista 2024 性别和下沉分布。
  - Instagram, YT Shorts (placeholder, NOT point-cited):
                       Meta FY24 + Alphabet Q4 2024 investor letters
                       + DataReportal aggregate demographics. Values
                       are directional placeholders so the 3-layer
                       platform stack can boot; re-calibrate against a
                       real panel before quoting numbers to clients.

Supported keys in ``audience_skew`` (all multiplicative, centered at 1.0):

  female_boost · male_boost         · gender skew
  tier1_boost  · tier3_boost        · city tier (≤1 = 一线新一线, ≥2 = 三线及以下)
  young_boost  · senior_boost       · age skew (≤2 = 15-44, ≥4 = 55+)

Caveats:
  - QuestMobile 公开摘要对抖音/快手的城市 tier 只给定性描述，具体
    数字 interpolated from narrative.
  - Bilibili 性别比是 aggregator 数据（非财报一手），如上线商用需对
    Q4 2024 earnings call transcript 做 hard-cite.
  - TikTok ad-audience 基础（18+），没有 13-17 段，young_boost 偏保守。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlatformConfig:
    name: str
    cpm_cny: float  # cost per 1000 impressions
    conversion_cost: float  # cost per action estimate
    cold_start_days: float
    algo_diversity: float  # 0..1 : how broadly the recommender explores
    audience_skew: dict[str, float]  # soft biases applied on top of agent platform_activity


PLATFORMS: dict[str, PlatformConfig] = {
    "douyin": PlatformConfig(
        name="douyin",
        cpm_cny=35.0,
        conversion_cost=4.5,
        cold_start_days=0.5,
        algo_diversity=0.85,
        audience_skew={
            # gender: 男 53 % / 女 47 % (Statista 2024, cited)
            "female_boost": 0.97,
            "male_boost": 1.03,
            # tier: QuestMobile 2024 narrative — Douyin tier1+2 >
            # national average but less concentrated than XHS/B站.
            # Interpolated from "Douyin/Kuaishou tier3+ higher than
            # XHS/B/Weibo" narrative.
            "tier1_boost": 1.6,
            "tier3_boost": 0.85,
            # age: broadly representative, mild young skew
            "young_boost": 1.05,
            "senior_boost": 0.7,
        },
    ),
    "xhs": PlatformConfig(
        name="xhs",
        cpm_cny=48.0,
        conversion_cost=3.2,
        cold_start_days=1.2,
        algo_diversity=0.65,
        audience_skew={
            # QuestMobile 2024 cited: 女 65 % → 65/48.7 ≈ 1.33
            "female_boost": 1.33,
            "male_boost": 0.68,
            # tier1+2 55.4 % → 55.4/24 ≈ 2.31, capped to 1.8
            "tier1_boost": 1.8,
            "tier3_boost": 0.6,
            # 15-44 ≈ 73 % → 73/52 ≈ 1.4
            "young_boost": 1.4,
            "senior_boost": 0.6,
        },
    ),
    # TikTok ≈ global Douyin on ByteDance's FYP stack. CPM is quoted in USD
    # upstream but we normalize to CNY here (1 USD ≈ 7.2 CNY) so the generic
    # budget_to_impressions helper works uniformly across adapters. The
    # TikTokAdapterConfig keeps the USD figure for USD-denominated ad budgets.
    "tiktok": PlatformConfig(
        name="tiktok",
        cpm_cny=42.0,
        conversion_cost=3.8,
        cold_start_days=0.4,
        algo_diversity=0.92,
        audience_skew={
            # DataReportal Jan 2025 cited: 男 55.7 % / 女 44.3 %
            "female_boost": 0.91,
            "male_boost": 1.08,
            # Global platform — CN city-tier N/A. Leave tier multipliers
            # at 1.0 by omission.
            # Age 18-44 ≈ 82 % → 82/52 ≈ 1.58 (floors for 18-24 only,
            # real floor if 15-17 were counted would be higher)
            "young_boost": 1.6,
            # 55+ 8.4 % → 8.4/30 ≈ 0.28, floored to 0.6
            "senior_boost": 0.6,
        },
    ),
    "bilibili": PlatformConfig(
        name="bilibili",
        cpm_cny=38.0,
        conversion_cost=5.0,
        cold_start_days=1.0,
        algo_diversity=0.70,
        audience_skew={
            # aggregator data 2024: 男 57 % / 女 43 %. Soft-cite.
            "female_boost": 0.88,
            "male_boost": 1.11,
            # QuestMobile 2024: B站 与 XHS / Weibo 同属 "一二线高集中",
            # tier1+2 ≈ 55 % interpolated → 55/24 ≈ 2.29, cap 1.8
            "tier1_boost": 1.8,
            "tier3_boost": 0.6,
            # Bilibili 2024: 16-35 占 62.25 %, under 30 占 78.7 %,
            # under 35 占 ~86 % (cited aggregate).
            # Implied 15-44 ≈ 90 % → 90/52 ≈ 1.73
            "young_boost": 1.73,
            # 55+ ≈ 4 % → 4/30 ≈ 0.13, floor 0.6
            "senior_boost": 0.6,
        },
    ),
    "kuaishou": PlatformConfig(
        name="kuaishou",
        cpm_cny=30.0,
        conversion_cost=4.0,
        cold_start_days=0.6,
        algo_diversity=0.80,
        audience_skew={
            # Statista 2024 cited: 男 54 % / 女 46 %
            "female_boost": 0.94,
            "male_boost": 1.05,
            # Statista 2024: "two in five users in tier 3/4" → T3+4
            # alone = 40%, implies T3+ ≥ 60 %. tier1+2 ≈ 35 %.
            # 35/24 ≈ 1.46 (cited, uncapped).
            "tier1_boost": 1.46,
            "tier3_boost": 0.86,
            # Aggregate: 18-35 ≈ 45 %, under 35 ≈ 70 %, 50+ ≈ 20 %.
            # Implied 15-44 ≈ 68 % → 68/52 ≈ 1.31
            "young_boost": 1.31,
            # 55+ ≈ 20 % → 20/30 ≈ 0.67
            "senior_boost": 0.67,
        },
    ),
    # ------------------------------------------------------------------
    # Global platforms (non-CN baseline). CPMs normalized to CNY
    # (1 USD ≈ 7.2) so budget_to_impressions stays uniform. The
    # audience_skew here is a **directional placeholder** derived from
    # Meta/Alphabet investor-letter public talking points + DataReportal
    # aggregate demographics — NOT a point-cited verbatim audience
    # breakdown like XHS. Flagged below so production callers know to
    # re-calibrate against a real panel before quoting to clients.
    "instagram": PlatformConfig(
        name="instagram",
        cpm_cny=55.0,  # ~7.5 USD global avg 2024 aggregator data
        conversion_cost=4.2,
        cold_start_days=0.6,
        algo_diversity=0.78,
        audience_skew={
            # Meta FY24 + DataReportal: global ad-audience 女 ≈ 49 %,
            # 男 ≈ 51 %. Near-balanced; very slight male tilt directional.
            "female_boost": 1.0,
            "male_boost": 1.0,
            # Global platform — CN tier not applicable. Leave at 1.0.
            # 18-34 ≈ 60 % of ad audience (DataReportal aggregate).
            # Implied 15-44 ≈ 75 % → 75/52 ≈ 1.44, conservative 1.35.
            "young_boost": 1.35,
            # 55+ ≈ 10-12 % → 11/30 ≈ 0.37, floored at 0.6.
            "senior_boost": 0.6,
        },
    ),
    "youtube_shorts": PlatformConfig(
        name="youtube_shorts",
        cpm_cny=28.0,  # Shorts CPMs noticeably lower than long-form YT
        conversion_cost=5.5,
        cold_start_days=0.8,
        algo_diversity=0.75,
        audience_skew={
            # Alphabet Q4 2024 + DataReportal: YT 全站男 54 % / 女 46 %,
            # Shorts 面板年纪略低但性别近似。Directional placeholder.
            "female_boost": 0.94,
            "male_boost": 1.05,
            # Global platform — CN tier not applicable.
            # Shorts 13-34 ≈ 65 % (aggregate from third-party panels);
            # 15-44 ≈ 80 % → 80/52 ≈ 1.54, conservative 1.45.
            "young_boost": 1.45,
            # 55+ ≈ 8 % → 8/30 ≈ 0.27, floored at 0.6.
            "senior_boost": 0.6,
        },
    ),
}


def budget_to_impressions(budget_cny: float, platform: str) -> float:
    cfg = PLATFORMS[platform]
    return 1000.0 * budget_cny / cfg.cpm_cny

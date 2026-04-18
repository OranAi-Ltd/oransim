"""Douyin synthetic data provider — Greater-China priors."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from ....data.schema import (
    CanonicalFanProfile,
    CanonicalKOL,
    CanonicalNote,
    CanonicalNoteMetrics,
)


NICHES = [
    "beauty", "fashion", "food", "electronics", "travel",
    "parenting", "fitness", "home", "beverage", "pet",
    "finance", "ecommerce",
]
TIERS = ["nano", "micro", "mid", "macro", "mega"]
TIER_FAN_MEAN = {"nano": 12_000, "micro": 90_000, "mid": 500_000, "macro": 3_500_000, "mega": 25_000_000}

CITY_TIERS = ["tier1", "tier2", "tier3", "tier4", "tier5"]
REGION_LABELS = ["CN-East", "CN-South", "CN-North", "CN-West", "CN-Central"]

FAKE_ADJ_ZH = ["晨光", "星河", "暖阳", "微风", "碧空", "松风", "月影", "清溪"]
FAKE_NOUN_ZH = ["日记", "笔记", "小站", "工坊", "栈", "书", "录", "档"]


def _seed_from(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (2**31)


@dataclass
class DouyinSyntheticProvider:
    """Synthetic DataProvider backing DouyinAdapter."""

    seed: int = 42

    def fetch_kol(self, kol_id: str) -> CanonicalKOL:
        rng = random.Random(self.seed ^ _seed_from(kol_id))
        niche = rng.choice(NICHES)
        tier = rng.choices(TIERS, weights=[0.40, 0.32, 0.18, 0.08, 0.02])[0]
        mean = TIER_FAN_MEAN[tier]
        fan_count = max(500, int(rng.lognormvariate(math.log(mean), 0.45)))
        er = min(0.15, max(0.006, rng.betavariate(2.2, 55.0)))
        region = rng.choices(REGION_LABELS, weights=[0.30, 0.22, 0.20, 0.14, 0.14])[0]
        nickname = f"{rng.choice(FAKE_ADJ_ZH)}{rng.choice(FAKE_NOUN_ZH)}"
        return CanonicalKOL(
            kol_id=kol_id,
            nickname=nickname,
            platform="douyin",
            niche=niche,
            tier=tier,
            fan_count=fan_count,
            avg_engagement_rate=round(er, 4),
            region=region,
            joined_year=rng.randint(2018, 2025),
            verified=(tier in ("macro", "mega")),
            average_views_per_post=fan_count * rng.uniform(0.14, 0.48),
            avg_posting_interval_days=round(rng.uniform(0.3, 2.5), 2),
            fan_profile=self.fetch_fan_profile(kol_id),
            custom_metadata={"source": "douyin_synthetic_v0.2"},
        )

    def search_notes(
        self,
        keyword: str,
        *,
        max_results: int = 20,
        **_kwargs: object,
    ) -> list[CanonicalNote]:
        rng = random.Random(self.seed ^ _seed_from(keyword))
        out: list[CanonicalNote] = []
        for i in range(max_results):
            er = min(0.18, max(0.005, rng.betavariate(2.2, 55.0)))
            impressions = rng.randint(10_000, 1_500_000)
            likes = int(impressions * er * 0.72)
            comments = int(impressions * er * 0.06)
            shares = int(impressions * er * 0.10)
            saves = int(impressions * er * 0.12)
            out.append(
                CanonicalNote(
                    note_id=f"DY-{keyword[:8]}-{i:04d}",
                    kol_id=f"DY_KOL_{rng.randint(1, 9999):06d}",
                    platform="douyin",
                    niche=rng.choice(NICHES),
                    text=f"{keyword} · {rng.choice(FAKE_ADJ_ZH)}{rng.choice(FAKE_NOUN_ZH)}",
                    text_language="zh",
                    media_types=["video"],
                    duration_sec=rng.uniform(8.0, 55.0),
                    tags=[f"#{keyword}", f"#{rng.choice(NICHES)}"],
                    metrics=CanonicalNoteMetrics(
                        impressions=impressions,
                        likes=likes,
                        comments=comments,
                        shares=shares,
                        saves=saves,
                        engagement_rate=round(er, 4),
                    ),
                    publish_day=rng.randint(1, 180),
                    publish_hour_of_day=rng.choice([7, 8, 12, 19, 20, 21, 22]),
                )
            )
        return out

    def fetch_fan_profile(self, kol_id: str) -> CanonicalFanProfile:
        rng = random.Random(self.seed ^ _seed_from(kol_id + "fanprof"))
        # Douyin is broader-age than TikTok — meaningful 25-44 presence
        age = [
            max(0.0, rng.gauss(0.04, 0.015)),
            max(0.0, rng.gauss(0.22, 0.04)),
            max(0.0, rng.gauss(0.36, 0.05)),
            max(0.0, rng.gauss(0.22, 0.04)),
            max(0.0, rng.gauss(0.10, 0.03)),
            max(0.0, rng.gauss(0.04, 0.015)),
            max(0.0, rng.gauss(0.02, 0.01)),
        ]
        s = sum(age) or 1.0
        age = [a / s for a in age]
        female = min(0.90, max(0.25, rng.gauss(0.58, 0.07)))
        region_dist = [0.30, 0.22, 0.20, 0.14, 0.14]
        return CanonicalFanProfile(
            age_dist=age,
            gender_dist=[female, 1.0 - female],
            region_dist=region_dist,
            region_labels=REGION_LABELS,
            source="douyin_synthetic_v0.2",
        )

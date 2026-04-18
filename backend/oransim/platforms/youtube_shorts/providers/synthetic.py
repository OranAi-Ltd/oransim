"""YouTube Shorts synthetic data provider."""

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
    "gaming", "education", "tech_review",  # YouTube-heavy niches
]
TIERS = ["nano", "micro", "mid", "macro", "mega"]
# YouTube rewards long-tail subscribers — tier thresholds run slightly
# higher than TikTok / Instagram on the upper end.
TIER_FAN_MEAN = {"nano": 10_000, "micro": 80_000, "mid": 500_000, "macro": 4_000_000, "mega": 25_000_000}

FAKE_ADJ = ["Apex", "Prism", "Vertex", "Cobalt", "Axial", "Quartz", "Vanta", "Stellar"]
FAKE_NOUN = ["Lab", "Works", "Channel", "Review", "Breakdown", "Show", "Feed", "Tune"]


def _seed_from(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (2**31)


@dataclass
class YouTubeShortsSyntheticProvider:
    """Synthetic DataProvider backing YouTubeShortsAdapter."""

    seed: int = 42

    def fetch_kol(self, kol_id: str) -> CanonicalKOL:
        rng = random.Random(self.seed ^ _seed_from(kol_id))
        niche = rng.choice(NICHES)
        tier = rng.choices(TIERS, weights=[0.42, 0.30, 0.18, 0.08, 0.02])[0]
        mean = TIER_FAN_MEAN[tier]
        fan_count = max(500, int(rng.lognormvariate(math.log(mean), 0.45)))
        # YouTube engagement rate is typically lower than TikTok / IG
        # because views don't always translate into explicit reactions.
        er = min(0.12, max(0.003, rng.betavariate(1.8, 58.0)))
        country = rng.choices(
            ["US", "IN", "BR", "UK", "JP", "ID", "OTHER"],
            weights=[0.28, 0.12, 0.08, 0.07, 0.06, 0.06, 0.33],
        )[0]
        return CanonicalKOL(
            kol_id=kol_id,
            nickname=f"{rng.choice(FAKE_ADJ)}{rng.choice(FAKE_NOUN)}",
            platform="youtube_shorts",
            niche=niche,
            tier=tier,
            fan_count=fan_count,
            avg_engagement_rate=round(er, 4),
            region=country,
            joined_year=rng.randint(2015, 2025),
            verified=(tier in ("macro", "mega")),
            average_views_per_post=fan_count * rng.uniform(0.10, 0.40),
            avg_posting_interval_days=round(rng.uniform(0.7, 5.0), 2),
            fan_profile=self.fetch_fan_profile(kol_id),
            custom_metadata={"source": "youtube_shorts_synthetic_v0.2"},
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
            er = min(0.12, max(0.004, rng.betavariate(1.8, 58.0)))
            # Shorts impressions often larger than other platforms thanks
            # to search long-tail delivery.
            impressions = rng.randint(20_000, 2_500_000)
            likes = int(impressions * er * 0.68)
            comments = int(impressions * er * 0.09)
            shares = int(impressions * er * 0.09)
            saves = int(impressions * er * 0.14)
            out.append(
                CanonicalNote(
                    note_id=f"YS-{keyword[:8]}-{i:04d}",
                    kol_id=f"YS_KOL_{rng.randint(1, 9999):06d}",
                    platform="youtube_shorts",
                    niche=rng.choice(NICHES),
                    text=f"{keyword} — {rng.choice(FAKE_ADJ)} take in 60s",
                    text_language="en",
                    media_types=["video"],
                    duration_sec=rng.uniform(15.0, 60.0),
                    tags=[f"#{keyword}", f"#{rng.choice(NICHES)}", "#shorts"],
                    metrics=CanonicalNoteMetrics(
                        impressions=impressions,
                        likes=likes,
                        comments=comments,
                        shares=shares,
                        saves=saves,
                        engagement_rate=round(er, 4),
                    ),
                    publish_day=rng.randint(1, 180),
                    publish_hour_of_day=rng.choice([7, 11, 14, 17, 19, 20, 21]),
                )
            )
        return out

    def fetch_fan_profile(self, kol_id: str) -> CanonicalFanProfile:
        rng = random.Random(self.seed ^ _seed_from(kol_id + "fanprof"))
        # YouTube viewership spans widest — 18-44 is the bulk but long-tail
        # holds into 45+ for education / tech niches.
        age = [
            max(0.0, rng.gauss(0.06, 0.02)),
            max(0.0, rng.gauss(0.25, 0.04)),
            max(0.0, rng.gauss(0.30, 0.05)),
            max(0.0, rng.gauss(0.20, 0.04)),
            max(0.0, rng.gauss(0.12, 0.03)),
            max(0.0, rng.gauss(0.05, 0.02)),
            max(0.0, rng.gauss(0.02, 0.01)),
        ]
        s = sum(age) or 1.0
        age = [a / s for a in age]
        # YouTube skews slightly more male than IG / TikTok
        female = min(0.80, max(0.25, rng.gauss(0.45, 0.08)))
        return CanonicalFanProfile(
            age_dist=age,
            gender_dist=[female, 1.0 - female],
            region_dist=[0.28, 0.12, 0.08, 0.07, 0.06, 0.06, 0.33],
            region_labels=["US", "IN", "BR", "UK", "JP", "ID", "OTHER"],
            source="youtube_shorts_synthetic_v0.2",
        )

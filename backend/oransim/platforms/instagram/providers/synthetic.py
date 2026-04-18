"""Instagram Reels synthetic data provider."""

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
    "wellness", "photography",  # Instagram-heavy niches
]
TIERS = ["nano", "micro", "mid", "macro", "mega"]
TIER_FAN_MEAN = {"nano": 6_000, "micro": 50_000, "mid": 280_000, "macro": 1_800_000, "mega": 10_000_000}

FAKE_ADJ = ["Pastel", "Coral", "Ivory", "Amber", "Sage", "Dusk", "Meadow", "Linen"]
FAKE_NOUN = ["Studio", "Atelier", "Collective", "Edit", "Journal", "Notes", "Library", "Story"]


def _seed_from(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (2**31)


@dataclass
class InstagramSyntheticProvider:
    """Synthetic DataProvider backing InstagramAdapter."""

    seed: int = 42

    def fetch_kol(self, kol_id: str) -> CanonicalKOL:
        rng = random.Random(self.seed ^ _seed_from(kol_id))
        niche = rng.choice(NICHES)
        tier = rng.choices(TIERS, weights=[0.50, 0.28, 0.14, 0.07, 0.01])[0]
        mean = TIER_FAN_MEAN[tier]
        fan_count = max(500, int(rng.lognormvariate(math.log(mean), 0.4)))
        er = min(0.16, max(0.006, rng.betavariate(2.0, 52.0)))
        country = rng.choices(
            ["US", "UK", "IN", "BR", "DE", "JP", "OTHER"],
            weights=[0.32, 0.10, 0.12, 0.10, 0.06, 0.05, 0.25],
        )[0]
        return CanonicalKOL(
            kol_id=kol_id,
            nickname=f"{rng.choice(FAKE_ADJ)}{rng.choice(FAKE_NOUN)}",
            platform="instagram",
            niche=niche,
            tier=tier,
            fan_count=fan_count,
            avg_engagement_rate=round(er, 4),
            region=country,
            joined_year=rng.randint(2016, 2025),
            verified=(tier in ("macro", "mega")),
            average_views_per_post=fan_count * rng.uniform(0.18, 0.52),
            avg_posting_interval_days=round(rng.uniform(1.0, 6.0), 2),
            fan_profile=self.fetch_fan_profile(kol_id),
            custom_metadata={"source": "instagram_synthetic_v0.2"},
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
            er = min(0.16, max(0.005, rng.betavariate(2.0, 52.0)))
            impressions = rng.randint(3_000, 500_000)
            likes = int(impressions * er * 0.75)
            comments = int(impressions * er * 0.06)
            shares = int(impressions * er * 0.08)
            saves = int(impressions * er * 0.11)
            out.append(
                CanonicalNote(
                    note_id=f"IG-{keyword[:8]}-{i:04d}",
                    kol_id=f"IG_KOL_{rng.randint(1, 9999):06d}",
                    platform="instagram",
                    niche=rng.choice(NICHES),
                    text=f"{keyword} · {rng.choice(FAKE_ADJ)} take",
                    text_language="en",
                    media_types=["video"] if rng.random() < 0.7 else ["image"],
                    duration_sec=rng.uniform(9.0, 60.0),
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
                    publish_hour_of_day=rng.randint(8, 22),
                )
            )
        return out

    def fetch_fan_profile(self, kol_id: str) -> CanonicalFanProfile:
        rng = random.Random(self.seed ^ _seed_from(kol_id + "fanprof"))
        # Instagram skews slightly older than TikTok
        age = [
            max(0.0, rng.gauss(0.04, 0.015)),
            max(0.0, rng.gauss(0.28, 0.05)),
            max(0.0, rng.gauss(0.36, 0.05)),
            max(0.0, rng.gauss(0.18, 0.04)),
            max(0.0, rng.gauss(0.08, 0.02)),
            max(0.0, rng.gauss(0.03, 0.015)),
            max(0.0, rng.gauss(0.01, 0.008)),
        ]
        s = sum(age) or 1.0
        age = [a / s for a in age]
        female = min(0.90, max(0.35, rng.gauss(0.58, 0.07)))
        return CanonicalFanProfile(
            age_dist=age,
            gender_dist=[female, 1.0 - female],
            region_dist=[0.32, 0.10, 0.12, 0.10, 0.06, 0.05, 0.25],
            region_labels=["US", "UK", "IN", "BR", "DE", "JP", "OTHER"],
            source="instagram_synthetic_v0.2",
        )

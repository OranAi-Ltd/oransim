"""TikTok synthetic data provider — deterministic fake KOL / note data.

Good enough to exercise the TikTokAdapter end-to-end without access to a
real TikTok Research API key. All returned objects are CanonicalKOL /
CanonicalNote instances so downstream code is agnostic to the provider.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from ....data.schema import (
    CanonicalFanProfile,
    CanonicalKOL,
    CanonicalNote,
    CanonicalNoteMetrics,
)

NICHES = [
    "beauty",
    "fashion",
    "food",
    "electronics",
    "travel",
    "parenting",
    "fitness",
    "home",
    "beverage",
    "pet",
    "gaming",
    "comedy",  # TikTok-heavy niches
]
TIERS = ["nano", "micro", "mid", "macro", "mega"]
TIER_FAN_MEAN = {
    "nano": 8_000,
    "micro": 60_000,
    "mid": 350_000,
    "macro": 2_500_000,
    "mega": 15_000_000,
}

FAKE_ADJ = ["Velvet", "Neon", "Indigo", "Solar", "Cobalt", "Amber", "Jade", "Crimson"]
FAKE_NOUN = ["Studio", "Diaries", "Club", "Lab", "Bureau", "Press", "Archive", "Works"]


def _seed_from(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (2**31)


@dataclass
class TikTokSyntheticProvider:
    """Synthetic DataProvider backing TikTokAdapter.

    Deterministic: the same ``kol_id`` always returns the same profile.
    """

    seed: int = 42

    # -------------------- KOL ------------------------------------------

    def fetch_kol(self, kol_id: str) -> CanonicalKOL:
        rng = random.Random(self.seed ^ _seed_from(kol_id))
        niche = rng.choice(NICHES)
        tier = rng.choices(TIERS, weights=[0.45, 0.30, 0.15, 0.08, 0.02])[0]
        mean = TIER_FAN_MEAN[tier]
        fan_count = max(500, int(rng.lognormvariate(__import__("math").log(mean), 0.4)))
        er = min(0.18, max(0.008, rng.betavariate(2.0, 48.0)))
        country_weights = [0.35, 0.12, 0.10, 0.08, 0.05, 0.30]
        countries = ["US", "ID", "BR", "MX", "JP", "OTHER"]
        country = rng.choices(countries, weights=country_weights)[0]
        return CanonicalKOL(
            kol_id=kol_id,
            nickname=f"{rng.choice(FAKE_ADJ)}{rng.choice(FAKE_NOUN)}",
            platform="tiktok",
            niche=niche,
            tier=tier,
            fan_count=fan_count,
            avg_engagement_rate=round(er, 4),
            region=country,
            joined_year=rng.randint(2019, 2025),
            verified=(tier in ("macro", "mega")),
            average_views_per_post=fan_count * rng.uniform(0.12, 0.45),
            avg_posting_interval_days=round(rng.uniform(0.5, 4.0), 2),
            fan_profile=self.fetch_fan_profile(kol_id),
            custom_metadata={"source": "tiktok_synthetic_v0.2"},
        )

    # -------------------- Notes / posts --------------------------------

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
            note_id = f"TT-{keyword[:8]}-{i:04d}"
            er = min(0.20, max(0.005, rng.betavariate(2.0, 50.0)))
            impressions = rng.randint(5_000, 800_000)
            likes = int(impressions * er * 0.7)
            comments = int(impressions * er * 0.08)
            shares = int(impressions * er * 0.12)
            saves = int(impressions * er * 0.10)
            out.append(
                CanonicalNote(
                    note_id=note_id,
                    kol_id=f"TT_KOL_{rng.randint(1, 9999):06d}",
                    platform="tiktok",
                    niche=rng.choice(NICHES),
                    text=f"{keyword} — {rng.choice(FAKE_ADJ)} take, rate /10",
                    text_language="en",
                    media_types=["video"],
                    duration_sec=rng.uniform(9.0, 58.0),
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
                    publish_hour_of_day=rng.randint(6, 23),
                )
            )
        return out

    # -------------------- Fan profile ----------------------------------

    def fetch_fan_profile(self, kol_id: str) -> CanonicalFanProfile:
        rng = random.Random(self.seed ^ _seed_from(kol_id + "fanprof"))
        # TikTok skews young — heavier weight on 18-24 / 25-34 bands
        age = [
            max(0.0, rng.gauss(0.08, 0.02)),  # 14-17
            max(0.0, rng.gauss(0.38, 0.05)),  # 18-24
            max(0.0, rng.gauss(0.32, 0.05)),  # 25-34
            max(0.0, rng.gauss(0.14, 0.03)),  # 35-44
            max(0.0, rng.gauss(0.06, 0.02)),  # 45-54
            max(0.0, rng.gauss(0.015, 0.01)),  # 55-64
            max(0.0, rng.gauss(0.005, 0.005)),  # 65+
        ]
        s = sum(age) or 1.0
        age = [a / s for a in age]
        female = min(0.95, max(0.30, rng.gauss(0.55, 0.08)))
        return CanonicalFanProfile(
            age_dist=age,
            gender_dist=[female, 1.0 - female],
            region_dist=[0.35, 0.12, 0.10, 0.08, 0.05, 0.30],
            region_labels=["US", "ID", "BR", "MX", "JP", "OTHER"],
            source="tiktok_synthetic_v0.2",
        )

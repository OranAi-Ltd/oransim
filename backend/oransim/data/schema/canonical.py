"""Canonical Pydantic schemas — the cross-platform contract.

Every ``PlatformAdapter`` consumes these types. Every ``DataProvider`` is
responsible for mapping its vendor-specific format into these types.
Introducing a new platform never requires changes to providers; adding a
new provider never requires changes to platforms. The canonical schemas
are the shared vocabulary between the two axes.

All schemas are versioned via ``schema_version``. Breaking changes bump
the major number (v1 → v2) and ship a migration. Minor additions
(v1.0 → v1.1) are backwards-compatible: fields are only added as
``Optional`` to avoid breaking existing providers.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.1"


class CanonicalFanProfile(BaseModel):
    """Aggregate fan-demographic distribution for a KOL or niche.

    Every field is a list of bucket weights summing to ~1.0. Providers
    that do not have a specific slice may leave the field empty.
    """

    model_config = ConfigDict(extra="forbid")

    # Age: 14-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+
    age_dist: list[float] = Field(default_factory=list)
    age_bands: list[str] = Field(
        default_factory=lambda: ["14-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    )

    # Gender share: [female, male]
    gender_dist: list[float] = Field(default_factory=list)

    # Region share — provider-specific but by convention 5 tiers:
    # [CN-East, CN-South, CN-North, CN-West, CN-Central] for XHS/Douyin;
    # ISO country codes (top-N) for global platforms.
    region_dist: list[float] = Field(default_factory=list)
    region_labels: list[str] = Field(default_factory=list)

    # Income tertile: [low, mid, high]; optional, often unavailable
    income_dist: Optional[list[float]] = None

    # Niche / interest affinity (scores, not a distribution — each niche
    # independently in [0, 1])
    niche_affinity: dict[str, float] = Field(default_factory=dict)

    source: str = "unknown"
    schema_version: str = SCHEMA_VERSION


class CanonicalKOL(BaseModel):
    """Unified KOL representation across platforms."""

    model_config = ConfigDict(extra="forbid")

    kol_id: str
    nickname: str
    platform: str           # "xhs" | "tiktok" | "instagram" | ...
    niche: str              # beauty | fashion | food | ...
    tier: str               # nano | micro | mid | macro | mega
    fan_count: int
    avg_engagement_rate: float   # share of fans interacting on a typical post
    region: Optional[str] = None

    # Optional enrichments — providers may skip these
    joined_year: Optional[int] = None
    verified: Optional[bool] = None
    average_views_per_post: Optional[float] = None
    avg_posting_interval_days: Optional[float] = None
    fan_profile: Optional[CanonicalFanProfile] = None
    custom_metadata: dict[str, str] = Field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION


class CanonicalNoteMetrics(BaseModel):
    """Engagement metrics shipped alongside a note/post."""

    model_config = ConfigDict(extra="forbid")

    impressions: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    engagement_rate: Optional[float] = None


class CanonicalNote(BaseModel):
    """Unified post representation across platforms."""

    model_config = ConfigDict(extra="forbid")

    note_id: str
    kol_id: str
    platform: str
    niche: str

    # Text content (may be empty for image/video-only posts)
    text: str = ""
    text_language: str = "zh"       # ISO-639

    # Creative metadata
    media_types: list[str] = Field(default_factory=list)  # ["image", "video", "text"]
    duration_sec: Optional[float] = None                   # for video/audio
    tags: list[str] = Field(default_factory=list)

    # Engagement
    metrics: CanonicalNoteMetrics = Field(default_factory=CanonicalNoteMetrics)

    # Temporal
    publish_day: Optional[int] = None           # days since an arbitrary epoch
    publish_hour_of_day: Optional[int] = None   # 0-23

    custom_metadata: dict[str, str] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


class CanonicalScenario(BaseModel):
    """Full prediction-request bundle.

    A ``PlatformAdapter`` receives a ``CanonicalScenario`` and returns
    structured predictions across the 14-19 output schemas.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    platform: str

    # Treatment — under marketer control
    creative_text: str
    creative_media_types: list[str] = Field(default_factory=list)
    budget: float
    kol_ids: list[str] = Field(default_factory=list)

    # Covariates — not under control
    niche: str
    target_audience: dict[str, list] = Field(default_factory=dict)
    launch_day: Optional[int] = None
    launch_hour_of_day: Optional[int] = None

    schema_version: str = SCHEMA_VERSION


__all__ = [
    "SCHEMA_VERSION",
    "CanonicalFanProfile",
    "CanonicalKOL",
    "CanonicalNote",
    "CanonicalNoteMetrics",
    "CanonicalScenario",
]

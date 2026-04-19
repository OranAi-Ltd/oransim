"""Pydantic request/response models shared across routers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreativeInput(BaseModel):
    caption: str
    duration_sec: float = 15.0
    visual_style: str = "bright"
    music_mood: str = "upbeat"
    has_celeb: bool = False


class PredictRequest(BaseModel):
    creative: CreativeInput
    # Budget bounded at 10B — well above any realistic campaign, short of overflow.
    total_budget: float = Field(default=50_000, ge=0, le=1e10)
    platform_alloc: dict[str, float] = {"douyin": 0.6, "xhs": 0.4}
    audience_age_buckets: list[int] | None = None
    audience_gender: int | None = None
    audience_city_tiers: list[int] | None = None
    kol_niche: str | None = None
    use_llm: bool = False
    llm_calibrate: bool = True  # if use_llm, also rescale KPIs by LLM votes
    # Hard cap at 1000 — beyond that, LLM latency and memory cost run away.
    # Further capped at runtime by the actual soul pool size.
    n_souls: int = Field(default=50, ge=0, le=1000)
    lifecycle_days: int = Field(default=14, ge=1, le=60)
    today: str | None = None  # ISO date for holiday/season; default today
    daypart: str = "auto"  # morning/noon/afternoon/evening/late/auto
    weather_temp_c: float = 20.0
    rainy: bool = False
    sentiment: str = "neutral"
    cross_platform_overlap: float = 0.25
    # --- feature toggles ---
    enable_crossplat: bool = True  # D — unique reach, cannibalization
    enable_discourse: bool = False  # A — LLM comment debate as SCM mediator
    discourse_n_comments: int = Field(default=15, ge=0, le=200)
    enable_brand_memory: bool = False  # B — 90-day brand lift
    brand_memory_days: int = Field(default=90, ge=1, le=365)
    enable_recsys_rl: bool = False  # C — platform RL loop simulation
    enable_groupchat: bool = False  # E — multi-turn LLM group chat
    groupchat_n_agents: int = Field(default=12, ge=2, le=50)
    groupchat_n_rounds: int = Field(default=4, ge=1, le=20)
    # --- schema-aligned extras ---
    own_brand: str | None = None  # 本品牌名 (optional; for T1-A3/T3-A6)
    category: str | None = None  # 品类 (optional; for T1-A3 context)
    competitors: list[str] | None = None  # 竞品列表 (enables T1-A3 LLM call)
    target_niches: list[str] | None = None  # KOL 赛道偏好 (for T2-A1)
    enable_kol_ilp: bool = True  # T2-A1 KOL 组合优化
    enable_search_elasticity: bool = True  # T3-A6

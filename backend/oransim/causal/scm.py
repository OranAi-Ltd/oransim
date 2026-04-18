"""Industrial-grade Structural Causal Model for ad funnel.

50+ nodes across 8 layers, mirroring real marketing causal chains:

  L1 Macro            外生宏观（天气/节日/舆情/竞品/监管/供应链）
  L2 Brand            品牌存量（认知/记忆/态度/历史购买）
  L3 Decision         可干预决策（预算/出价/人群/达人/创意/节奏）
  L4 Distribution     算法分发（曝光/频次/覆盖/算法放大）
  L5 User State       用户状态（注意/疲劳/情绪/理解/先验）
  L6 Discourse        群体话语（评论/口碑/peer影响/共识）
  L7 Funnel           漏斗（回忆/点击/互动/搜索/页面/加购/转化/复购）
  L8 Outcome          产出（直销/归因/品牌 lift/LTV/ROI/有机增量）

Edges encode causal direction. Mediators marked. Time-varying nodes flagged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SCMNode:
    name: str
    layer: str  # L1..L8 for visualization grouping
    category: str  # exogenous / decision / mediator / outcome / state
    label_zh: str
    intervenable: bool = False
    time_varying: bool = False
    computed_by: str = ""  # module name that actually computes this in V1
    description: str = ""


# ============================================================================
# Nodes
# ============================================================================

NODES: list[SCMNode] = [
    # ---- L1 Macro / Exogenous ----
    SCMNode("macro_env", "L1", "exogenous", "宏观环境", description="GDP/CPI/消费信心"),
    SCMNode("weather", "L1", "exogenous", "天气", time_varying=True, computed_by="macro.py"),
    SCMNode("season", "L1", "exogenous", "季节", computed_by="macro.py"),
    SCMNode("holiday_calendar", "L1", "exogenous", "节假日", computed_by="macro.py"),
    SCMNode("dow", "L1", "exogenous", "星期", computed_by="macro.py"),
    SCMNode(
        "public_sentiment",
        "L1",
        "exogenous",
        "全民舆情",
        time_varying=True,
        computed_by="world_events.py",
    ),
    SCMNode("competitor_action", "L1", "exogenous", "竞品动作", time_varying=True),
    SCMNode("regulatory_state", "L1", "exogenous", "监管状态", description="广告法/平台审核紧/松"),
    SCMNode("supply_state", "L1", "exogenous", "供应链状态", description="缺货/正常/过剩"),
    # ---- L2 Brand (latent stocks) ----
    SCMNode("brand_awareness", "L2", "state", "品牌认知度", time_varying=True),
    SCMNode(
        "brand_recall", "L2", "state", "品牌回忆", time_varying=True, computed_by="brand_memory.py"
    ),
    SCMNode(
        "brand_favor", "L2", "state", "品牌好感", time_varying=True, computed_by="brand_memory.py"
    ),
    SCMNode(
        "brand_aversion",
        "L2",
        "state",
        "品牌反感",
        time_varying=True,
        computed_by="brand_memory.py",
    ),
    SCMNode("prior_purchase", "L2", "state", "历史购买", description="复购客户标记"),
    SCMNode("brand_equity", "L2", "state", "品牌资产", description="长期累积价值"),
    # ---- L3 Decision (intervenable) ----
    SCMNode("total_budget", "L3", "decision", "总预算", intervenable=True),
    SCMNode("platform_alloc", "L3", "decision", "平台分配", intervenable=True),
    SCMNode("audience_pkg", "L3", "decision", "人群包", intervenable=True),
    SCMNode("kol_choice", "L3", "decision", "达人选择", intervenable=True),
    SCMNode("creative_caption", "L3", "decision", "文案", intervenable=True),
    SCMNode("creative_visual", "L3", "decision", "视觉", intervenable=True),
    SCMNode("creative_audio", "L3", "decision", "音频", intervenable=True),
    SCMNode(
        "bid_strategy",
        "L3",
        "decision",
        "出价策略",
        intervenable=True,
        description="CPM/CPC/CPA/oCPX",
    ),
    SCMNode("frequency_cap", "L3", "decision", "频次上限", intervenable=True),
    SCMNode(
        "pacing", "L3", "decision", "投放节奏", intervenable=True, description="均匀/前置/后置"
    ),
    SCMNode("daypart_alloc", "L3", "decision", "时段分配", intervenable=True),
    SCMNode("ab_variants", "L3", "decision", "AB 变体", intervenable=True),
    # ---- L4 Distribution ----
    SCMNode("ecpm_bid", "L4", "mediator", "ECPM 竞价", computed_by="platforms.py"),
    SCMNode("impression_dist", "L4", "mediator", "曝光分发", computed_by="world_model.py"),
    SCMNode(
        "recsys_amplification",
        "L4",
        "mediator",
        "算法放大",
        computed_by="recsys_rl.py",
        description="冷启破圈系数",
    ),
    SCMNode("exposure_count", "L4", "mediator", "曝光次数", time_varying=True),
    SCMNode("unique_reach", "L4", "mediator", "唯一覆盖", computed_by="cross_platform.py"),
    SCMNode("frequency", "L4", "mediator", "曝光频次", computed_by="cross_platform.py"),
    SCMNode("audience_match", "L4", "mediator", "人群匹配度", computed_by="world_model.py"),
    SCMNode(
        "organic_amplification",
        "L4",
        "mediator",
        "自然扩散",
        computed_by="hawkes.py",
        description="Hawkes 二次曝光",
    ),
    # ---- L5 User State ----
    SCMNode("attention", "L5", "state", "注意力"),
    SCMNode(
        "user_fatigue",
        "L5",
        "state",
        "用户疲劳",
        time_varying=True,
        computed_by="cross_platform.py",
    ),
    SCMNode("user_mood", "L5", "state", "用户情绪", time_varying=True),
    SCMNode("comprehension", "L5", "state", "信息理解度"),
    SCMNode("prior_belief", "L5", "state", "先验态度", description="刷之前对品牌的认知"),
    # ---- L6 Discourse / Social ----
    SCMNode("comment_sentiment", "L6", "mediator", "评论情绪", computed_by="discourse.py"),
    SCMNode("group_consensus", "L6", "mediator", "群聊共识", computed_by="group_chat.py"),
    SCMNode("group_polarization", "L6", "mediator", "群体极化", computed_by="group_chat.py"),
    SCMNode("peer_influence", "L6", "mediator", "同伴影响"),
    SCMNode("viral_score", "L6", "mediator", "病毒系数", computed_by="hawkes.py"),
    # ---- L7 Funnel ----
    SCMNode("ad_recall", "L7", "mediator", "广告回忆"),
    SCMNode("click", "L7", "mediator", "点击", computed_by="statistical.py"),
    SCMNode("engagement", "L7", "mediator", "互动", description="点赞/收藏/评论/分享"),
    SCMNode("brand_search_lift", "L7", "mediator", "品牌搜索增量"),
    SCMNode("site_visit", "L7", "mediator", "落地页/小程序访问"),
    SCMNode("product_view", "L7", "mediator", "商品页浏览"),
    SCMNode("add_to_cart", "L7", "mediator", "加购"),
    SCMNode("conversion", "L7", "mediator", "下单转化", computed_by="statistical.py"),
    SCMNode("repurchase", "L7", "mediator", "复购", time_varying=True),
    # ---- L8 Outcome ----
    SCMNode("direct_revenue", "L8", "outcome", "直接 GMV"),
    SCMNode("attributed_revenue", "L8", "outcome", "归因 GMV"),
    SCMNode("brand_lift_aware", "L8", "outcome", "品牌认知 lift"),
    SCMNode("brand_lift_favor", "L8", "outcome", "品牌好感 lift"),
    SCMNode("brand_lift_intent", "L8", "outcome", "购意 lift"),
    SCMNode("ltv_increment", "L8", "outcome", "LTV 增量"),
    SCMNode("roas", "L8", "outcome", "ROAS"),
    SCMNode("roi", "L8", "outcome", "ROI"),
    SCMNode("organic_search_uplift", "L8", "outcome", "自然搜索增量"),
    SCMNode("nps_delta", "L8", "outcome", "NPS 变化"),
]


# ============================================================================
# Edges (~120)
# ============================================================================

EDGES: list[tuple[str, str]] = [
    # ---- L1 → others ----
    ("macro_env", "user_mood"),
    ("macro_env", "brand_equity"),
    ("weather", "user_mood"),
    ("weather", "site_visit"),
    ("season", "ecpm_bid"),
    ("season", "click"),
    ("holiday_calendar", "ecpm_bid"),
    ("holiday_calendar", "click"),
    ("holiday_calendar", "conversion"),
    ("dow", "user_mood"),
    ("dow", "frequency"),
    ("public_sentiment", "comment_sentiment"),
    ("public_sentiment", "user_mood"),
    ("competitor_action", "ecpm_bid"),
    ("competitor_action", "audience_match"),
    ("regulatory_state", "impression_dist"),
    ("regulatory_state", "creative_caption"),
    ("supply_state", "conversion"),
    # ---- L2 brand → user ----
    ("brand_awareness", "ad_recall"),
    ("brand_awareness", "click"),
    ("brand_recall", "click"),
    ("brand_recall", "ad_recall"),
    ("brand_favor", "click"),
    ("brand_favor", "conversion"),
    ("brand_aversion", "click"),
    ("prior_purchase", "repurchase"),
    ("prior_purchase", "click"),
    ("brand_equity", "ecpm_bid"),
    # ---- L3 decisions → distribution ----
    ("total_budget", "impression_dist"),
    ("total_budget", "ecpm_bid"),
    ("total_budget", "roi"),  # cost side
    ("platform_alloc", "impression_dist"),
    ("platform_alloc", "audience_match"),
    ("audience_pkg", "audience_match"),
    ("audience_pkg", "impression_dist"),
    ("kol_choice", "impression_dist"),
    ("kol_choice", "click"),  # direct effect (达人信任)
    ("kol_choice", "comment_sentiment"),
    ("creative_caption", "comprehension"),
    ("creative_caption", "click"),
    ("creative_visual", "attention"),
    ("creative_visual", "click"),
    ("creative_audio", "attention"),
    ("bid_strategy", "ecpm_bid"),
    ("bid_strategy", "impression_dist"),
    ("frequency_cap", "frequency"),
    ("frequency_cap", "user_fatigue"),
    ("pacing", "exposure_count"),
    ("pacing", "organic_amplification"),
    ("daypart_alloc", "audience_match"),
    ("daypart_alloc", "user_mood"),
    ("ab_variants", "click"),
    # ---- L4 distribution → user / discourse ----
    ("ecpm_bid", "impression_dist"),
    ("impression_dist", "exposure_count"),
    ("impression_dist", "unique_reach"),
    ("impression_dist", "audience_match"),
    ("impression_dist", "recsys_amplification"),
    ("recsys_amplification", "exposure_count"),
    ("recsys_amplification", "viral_score"),
    ("exposure_count", "frequency"),
    ("exposure_count", "user_fatigue"),
    ("exposure_count", "ad_recall"),
    ("frequency", "user_fatigue"),
    ("frequency", "ad_recall"),
    ("unique_reach", "brand_lift_aware"),
    ("audience_match", "click"),
    ("organic_amplification", "exposure_count"),
    ("organic_amplification", "viral_score"),
    # ---- L5 user state → funnel ----
    ("attention", "ad_recall"),
    ("attention", "click"),
    ("user_fatigue", "click"),
    ("user_fatigue", "engagement"),
    ("user_mood", "click"),
    ("user_mood", "conversion"),
    ("comprehension", "click"),
    ("comprehension", "conversion"),
    ("prior_belief", "click"),
    ("prior_belief", "conversion"),
    # ---- L6 discourse → second-wave funnel ----
    ("comment_sentiment", "click"),
    ("comment_sentiment", "conversion"),
    ("comment_sentiment", "viral_score"),
    ("group_consensus", "comment_sentiment"),
    ("group_polarization", "viral_score"),
    ("group_polarization", "comment_sentiment"),
    ("peer_influence", "click"),
    ("peer_influence", "engagement"),
    ("viral_score", "organic_amplification"),
    ("viral_score", "brand_search_lift"),
    # ---- L7 funnel chain ----
    ("ad_recall", "click"),
    ("click", "engagement"),
    ("click", "site_visit"),
    ("click", "brand_search_lift"),
    ("engagement", "viral_score"),
    ("engagement", "conversion"),
    ("engagement", "peer_influence"),
    ("brand_search_lift", "site_visit"),
    ("brand_search_lift", "organic_search_uplift"),
    ("site_visit", "product_view"),
    ("product_view", "add_to_cart"),
    ("add_to_cart", "conversion"),
    ("conversion", "repurchase"),
    ("conversion", "direct_revenue"),
    ("conversion", "attributed_revenue"),
    # ---- L7 → L2 (long-term feedback) ----
    ("ad_recall", "brand_recall"),
    ("engagement", "brand_favor"),
    ("repurchase", "brand_equity"),
    ("repurchase", "ltv_increment"),
    # ---- L8 outcomes ----
    ("direct_revenue", "roi"),
    ("direct_revenue", "roas"),
    ("attributed_revenue", "roi"),
    ("attributed_revenue", "roas"),
    ("brand_lift_aware", "brand_awareness"),  # feedback
    ("brand_lift_favor", "brand_favor"),  # feedback
    ("brand_lift_intent", "conversion"),  # measurement → funnel feedback
    ("ltv_increment", "roi"),
    ("organic_search_uplift", "attributed_revenue"),
    ("comment_sentiment", "nps_delta"),
]


# Deduplicate edges (some helpers may double-add)
EDGES = list({(s, t) for s, t in EDGES})


# ============================================================================
# Node lookups
# ============================================================================

NODE_BY_NAME: dict[str, SCMNode] = {n.name: n for n in NODES}
INTERVENABLE: set[str] = {n.name for n in NODES if n.intervenable}
LAYERS = sorted({n.layer for n in NODES})
LAYER_LABELS = {
    "L1": "宏观/外生",
    "L2": "品牌存量",
    "L3": "投放决策",
    "L4": "算法分发",
    "L5": "用户状态",
    "L6": "群体话语",
    "L7": "漏斗",
    "L8": "产出",
}
LAYER_COLOR = {
    "L1": "#8b93a7",
    "L2": "#ffc857",
    "L3": "#6ea8fe",
    "L4": "#bd93f9",
    "L5": "#8be9fd",
    "L6": "#ff79c6",
    "L7": "#5ed39b",
    "L8": "#ff7a85",
}


def dag_dict() -> dict:
    """Rich SCM dict for visualization + intervention selectors."""
    return {
        "n_nodes": len(NODES),
        "n_edges": len(EDGES),
        "nodes": [
            {
                "name": n.name,
                "label_zh": n.label_zh,
                "layer": n.layer,
                "layer_label": LAYER_LABELS[n.layer],
                "category": n.category,
                "intervenable": n.intervenable,
                "time_varying": n.time_varying,
                "computed_by": n.computed_by,
                "color": LAYER_COLOR[n.layer],
            }
            for n in NODES
        ],
        "edges": [list(e) for e in EDGES],
        "intervenable": list(INTERVENABLE),
        "layers": [
            {
                "id": L,
                "label": LAYER_LABELS[L],
                "color": LAYER_COLOR[L],
                "nodes": [n.name for n in NODES if n.layer == L],
            }
            for L in LAYERS
        ],
        "stats": {
            "by_layer": {L: sum(1 for n in NODES if n.layer == L) for L in LAYERS},
            "by_category": {
                c: sum(1 for n in NODES if n.category == c) for c in {n.category for n in NODES}
            },
            "intervenable_count": len(INTERVENABLE),
            "time_varying_count": sum(1 for n in NODES if n.time_varying),
            "computed_count": sum(1 for n in NODES if n.computed_by),
        },
    }

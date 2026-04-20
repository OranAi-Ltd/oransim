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


# ============================================================================
# Time-unrolled DAG (acyclic projection of the cyclic causal graph)
# ============================================================================
#
# The shipped graph encodes long-term marketing feedback loops (repeat purchase
# → brand equity → CPM bid → next-cycle impression distribution, etc.). These
# loops make the graph strictly cyclic — see README §Causal Graph + Bongers
# et al. 2021 for the cyclic-SCM framing. For downstream modules that require
# a strict DAG (e.g. the CausalDAG-Transformer attention bias, or any classic
# Pearl abduction), we expose a time-unrolled projection:
#
#   - Node N becomes copies N_t0, N_t1, ..., N_t{K-1}
#   - Non-feedback edges are replicated inside each time slice
#   - Feedback edges cross time: ``src_t{i} → dst_t{i+1}`` (at the last slice
#     the feedback edge is dropped — its effect would land outside the horizon)
#
# The result is a proper DAG regardless of the cycle structure of the original.


def _find_feedback_edges() -> set[tuple[str, str]]:
    """Return the set of back-edges in a DFS traversal of the causal graph.

    An edge ``(u, v)`` is a "back edge" if, during DFS, we encounter ``v`` as
    a GRAY (on-stack) ancestor of ``u``. Back-edges are exactly the edges
    that participate in cycles; removing them produces an acyclic subgraph.
    The specific set returned depends on DFS start order but always satisfies:
    edges ∖ feedback-set is acyclic.
    """
    adj: dict[str, list[str]] = {n.name: [] for n in NODES}
    for s, t in EDGES:
        adj[s].append(t)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n.name: WHITE for n in NODES}
    feedback: set[tuple[str, str]] = set()

    # Iterative DFS — avoids recursion-limit on dense graphs.
    for start in NODES:
        if color[start.name] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(start.name, 0)]
        while stack:
            u, child_idx = stack[-1]
            if child_idx == 0:
                color[u] = GRAY
            children = adj[u]
            if child_idx < len(children):
                stack[-1] = (u, child_idx + 1)
                v = children[child_idx]
                c = color[v]
                if c == WHITE:
                    stack.append((v, 0))
                elif c == GRAY:
                    feedback.add((u, v))
                # BLACK → forward/cross edge, not feedback
            else:
                color[u] = BLACK
                stack.pop()
    return feedback


def dag_dict_unrolled(n_steps: int = 2) -> dict:
    """Acyclic time-unrolled projection of the causal graph.

    Each original node becomes ``n_steps`` time-indexed copies (``N_t0``..
    ``N_t{K-1}``). Non-feedback edges are replicated within each time slice;
    feedback edges cross to the next time slice. Resulting graph is a strict
    DAG (verified by :func:`tests.test_causal_invariants.test_dag_dict_unrolled
    _is_strict_dag`).

    Parameters
    ----------
    n_steps
        How many time slices. 2 is the minimum to acyclically represent all
        current feedback edges. 3+ lets downstream models reason about multi-
        horizon feedback (e.g. brand equity at t influences bid at t+1 which
        influences impression distribution at t+2).

    Returns
    -------
    A dict with the same shape as :func:`dag_dict` but with the time-unrolled
    graph, plus a ``feedback_edges`` list listing which original edges were
    treated as feedback (for debugging / visualization).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    feedback = _find_feedback_edges()

    unrolled_nodes: list[dict] = []
    for t in range(n_steps):
        for n in NODES:
            unrolled_nodes.append(
                {
                    "name": f"{n.name}_t{t}",
                    "original_name": n.name,
                    "time_step": t,
                    "label_zh": f"{n.label_zh}@t{t}",
                    "layer": n.layer,
                    "layer_label": LAYER_LABELS[n.layer],
                    "category": n.category,
                    "intervenable": n.intervenable and t == 0,  # interventions at t=0 only
                    "time_varying": n.time_varying,
                    "computed_by": n.computed_by,
                    "color": LAYER_COLOR[n.layer],
                }
            )

    unrolled_edges: list[list[str]] = []
    for s, d in EDGES:
        if (s, d) in feedback:
            # Cross-time edge: src at t, dst at t+1. Drop at the last slice
            # (no t+1 exists there).
            for t in range(n_steps - 1):
                unrolled_edges.append([f"{s}_t{t}", f"{d}_t{t+1}"])
        else:
            # Within-slice edge: replicate in every time step
            for t in range(n_steps):
                unrolled_edges.append([f"{s}_t{t}", f"{d}_t{t}"])

    return {
        "n_nodes": len(unrolled_nodes),
        "n_edges": len(unrolled_edges),
        "n_steps": n_steps,
        "nodes": unrolled_nodes,
        "edges": unrolled_edges,
        "feedback_edges": [list(e) for e in sorted(feedback)],
        "intervenable": [n["name"] for n in unrolled_nodes if n["intervenable"]],
        "stats": {
            "n_original_nodes": len(NODES),
            "n_original_edges": len(EDGES),
            "n_feedback_edges": len(feedback),
            "n_within_slice_edges": len(EDGES) - len(feedback),
        },
    }


# ============================================================================
# Fixed-point equilibrium under do() — Bongers et al. 2021 cyclic-SCM treatment
# ============================================================================
#
# For the feedback SCC (nodes participating in long-term marketing loops
# like repurchase → brand_equity → ecpm_bid → next-cycle impression_dist),
# a single topological forward pass is ill-defined. Following Bongers 2021
# §5, we treat the SCC as a **linear structural system** ``x = M x + b``
# whose equilibrium ``x* = (I - M)^(-1) b`` is the result of the do()
# intervention. ``M`` is the SCC adjacency with per-edge linear
# coefficients; ``b`` absorbs (i) exogenous inputs from non-SCC parents
# and (ii) ``do()`` clamps (intervention fixes the node value and zeros
# out its in-edges).
#
# See :mod:`oransim.causal.fixed_point` for the generic solver; this
# section defines the Oransim-specific SCC extraction + linear-system
# assembly + the public ``equilibrium_under_do`` entry point.


def get_feedback_scc() -> set[str]:
    """Return the set of node names in the single feedback strongly-connected
    component of the causal graph.

    The shipped SCM has exactly one non-singleton SCC (verified by
    :func:`tests.test_causal_invariants.test_scm_shape_is_intentional_cyclic_graph`),
    containing the long-term brand↔funnel feedback loop. This is the
    component Bongers-style fixed-point evaluation applies to.

    ``networkx`` is imported lazily so this module stays importable in
    environments without the dev extras (CI tests for the cyclic graph
    already require networkx).
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "networkx is required for SCC extraction. Install via "
            "pip install 'oransim[dev]' or pip install networkx."
        ) from e

    g = nx.DiGraph()
    for n in NODES:
        g.add_node(n.name)
    for s, t in EDGES:
        g.add_edge(s, t)

    sccs = [scc for scc in nx.strongly_connected_components(g) if len(scc) > 1]
    if not sccs:
        return set()
    # Return the largest (the shipped graph has exactly one non-singleton SCC).
    return max(sccs, key=len)


def _default_edge_weights(scc: set[str]) -> dict[tuple[str, str], float]:
    """Assign a default per-edge coefficient for edges inside the SCC.

    The linear model ``x = M x + b`` needs a per-edge weight. Without a
    calibration dataset we use a uniform weight per node's in-degree,
    rescaled to keep the spectral radius strictly below 1 (a necessary
    condition for a well-posed equilibrium — see
    :func:`oransim.causal.fixed_point.solve_linear_scm`).

    For the shipped 25-node SCC this keeps the system contractive by a
    comfortable margin (ρ ≈ 0.7 in practice). Production Enterprise
    deployments calibrate these weights from real-campaign data;
    overriding via the ``weights`` argument of
    :func:`equilibrium_under_do` is the extension point.
    """
    in_degree: dict[str, int] = {n: 0 for n in scc}
    scc_edges: list[tuple[str, str]] = []
    for s, t in EDGES:
        if s in scc and t in scc:
            in_degree[t] += 1
            scc_edges.append((s, t))
    # Per-edge weight = 0.75 / max(1, in_degree(t)) — sum of incoming
    # coefficients per node is ≤ 0.75, so spectral radius is bounded by 0.75
    # (Gerschgorin circles + scaled adjacency argument).
    return {(s, t): 0.75 / max(1, in_degree[t]) for (s, t) in scc_edges}


def equilibrium_under_do(
    intervention: dict[str, float] | None = None,
    *,
    weights: dict[tuple[str, str], float] | None = None,
    exogenous: dict[str, float] | None = None,
    method: str = "linear_closed_form",
):
    """Compute equilibrium values of the feedback SCC under ``do(X=x)``.

    Parameters
    ----------
    intervention : mapping ``node_name → clamped_value`` for ``do()`` nodes.
        Nodes in the intervention dict have their in-edges zeroed (per
        Pearl's truncated-factorisation rule) and their value forced.
        Nodes outside the SCC are ignored (they are upstream / downstream
        of the cycle, not part of the equilibrium).
    weights : mapping ``(source, target) → coefficient`` — override the
        default per-edge weight assignment. Missing entries fall back to
        the default.
    exogenous : mapping ``node_name → b_i`` — exogenous input for that
        SCC node (captures contributions from non-SCC parents). Missing
        entries default to 0.
    method : ``"linear_closed_form"`` (default, Bongers 2021 §5) or
        ``"banach"`` (generic damped Picard iteration, useful when the
        structural functions are non-linear).

    Returns
    -------
    dict with ``equilibrium`` (node name → equilibrium value),
    ``spectral_radius`` (diagnostic; iff ``< 1`` the system is
    contractive and Banach iteration would also converge), ``converged``
    (bool), ``method`` (string), and ``scc_size``.
    """
    from .fixed_point import banach_iterate, solve_linear_scm

    scc = sorted(get_feedback_scc())  # sort for deterministic node ordering
    if not scc:
        return {
            "equilibrium": {},
            "spectral_radius": 0.0,
            "converged": True,
            "method": method,
            "scc_size": 0,
            "note": "no non-trivial feedback SCC in the current graph",
        }

    idx = {name: i for i, name in enumerate(scc)}
    n = len(scc)
    intervention = intervention or {}
    exogenous = exogenous or {}
    default_w = _default_edge_weights(set(scc))
    if weights is None:
        weights = {}

    # Build adjacency matrix M (rows = target, cols = source, so x = M x)
    import numpy as np

    M = np.zeros((n, n), dtype=np.float64)
    for s, t in EDGES:
        if s in idx and t in idx:
            M[idx[t], idx[s]] = weights.get((s, t), default_w.get((s, t), 0.0))

    # Apply do() truncation: intervened nodes have their in-edges zeroed.
    clamps = np.zeros(n, dtype=np.float64)
    for node, value in intervention.items():
        if node in idx:
            i = idx[node]
            M[i, :] = 0.0  # ignore parents (truncated factorisation)
            clamps[i] = float(value)

    # Build b vector: exogenous inputs + do() clamps
    b = np.zeros(n, dtype=np.float64)
    for node, val in exogenous.items():
        if node in idx:
            b[idx[node]] = float(val)
    b = b + clamps  # clamps are absorbed as forced value

    if method == "linear_closed_form":
        result = solve_linear_scm(M, b)
    elif method == "banach":
        # Define f(x) = M x + b, iterate from zero
        def f(x: np.ndarray) -> np.ndarray:
            return M @ x + b

        result = banach_iterate(f, np.zeros(n), tol=1e-6, max_iter=200)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'linear_closed_form' or 'banach'")

    equilibrium = {name: float(result.x[i]) for name, i in idx.items()}
    return {
        "equilibrium": equilibrium,
        "spectral_radius": result.spectral_radius,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "residual_inf": result.residual_inf,
        "method": result.method,
        "scc_size": n,
    }

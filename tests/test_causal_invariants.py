"""Causal correctness invariants.

GPT review #5: existing tests check shapes + import but not causal semantics.
These tests pin structural invariants any causal pipeline must satisfy:

1. **Determinism** — same inputs + same seed must yield byte-identical KPIs.
   Without this, any CATE / counterfactual comparison is noise.

2. **Antisymmetric delta** — compute_cate(pop, A, B) and compute_cate(pop, B, A)
   should give per-agent deltas of opposite sign.

3. **Identity counterfactual** — compute_cate(pop, A, A) has zero effect; must
   surface as the saturated_impression_set diagnostic (not a table of zeros
   or a crash).

4. **Union semantics for CATE** — passing baseline/cf dicts with DIFFERENT
   reached-agent sets must not silently drop the union; budget-only
   interventions (change WHO is reached, not per-agent click_prob) must still
   surface as exposure-gain/loss.

5. **Scenario.run repeatability** — same Scenario + same seed → same ROI.

6. **Counterfactual round-trip zero** — running counterfactual with empty
   intervention should reproduce the baseline KPIs (no-op consistency).

These are cheap tests (no torch, no LLM) — they protect the *math*, not the
shape. The MSE / PEHE-style metrics that compare against ground truth CATE
live in OrancBench v0.5 (see ROADMAP.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

# ---------------------------------------------------------- helper fixtures


def _small_pop(n: int = 600, seed: int = 7):
    from oransim.data.population import generate_population

    return generate_population(N=n, seed=seed)


# ============================================================ 1. Determinism


def test_compute_cate_deterministic_same_inputs():
    """compute_cate with identical inputs must return identical output."""
    from oransim.causal.cate import compute_cate

    pop = _small_pop()
    rng = np.random.default_rng(42)
    base = {int(i): float(rng.random()) for i in range(min(200, pop.N))}
    cf = {int(i): float(rng.random()) for i in range(min(200, pop.N))}

    out_a = compute_cate(pop, base, cf)
    out_b = compute_cate(pop, base, cf)

    # Same shape + same top-k segments (deterministic forest seed inside)
    assert len(out_a) == len(out_b)
    for a, b in zip(out_a, out_b, strict=False):
        # compare stringified to handle nested dicts / floats
        assert repr(a) == repr(b), f"non-deterministic: {a} vs {b}"


def test_scenario_run_deterministic_same_seed():
    """Same Scenario + same seed → same total_kpis (byte identical)."""
    from oransim.agents.statistical import StatisticalAgents
    from oransim.causal.counterfactual import Scenario, ScenarioRunner
    from oransim.data.creatives import make_creative
    from oransim.platforms.xhs.world_model_legacy import PlatformWorldModel

    pop = _small_pop(n=500)
    wm = PlatformWorldModel(pop)
    ag = StatisticalAgents(pop)
    runner = ScenarioRunner(wm, ag)

    creative = make_creative(creative_id="det-001", caption="test", duration_sec=15.0)
    scen = Scenario(
        creative=creative,
        total_budget=5_000,
        platform_alloc={"douyin": 0.6, "xhs": 0.4},
        seed=42,
    )
    r1 = runner.run(scen, n_monte_carlo=1)
    r2 = runner.run(scen, n_monte_carlo=1)

    # KPI dicts must match key-for-key
    assert set(r1.total_kpis.keys()) == set(r2.total_kpis.keys())
    for k in r1.total_kpis:
        v1, v2 = r1.total_kpis[k], r2.total_kpis[k]
        if isinstance(v1, float) and isinstance(v2, float):
            assert abs(v1 - v2) < 1e-9, f"kpi '{k}' not deterministic: {v1} vs {v2}"
        else:
            assert v1 == v2


# ================================================== 2. Antisymmetric delta


def test_compute_cate_antisymmetric():
    """CATE(A, B) and CATE(B, A) should have opposite-sign per-segment deltas.

    Uses controlled synthetic click-prob dicts so the expected sign flip is
    unambiguous (beyond noise from the RandomForest regressor internal).
    """
    from oransim.causal.cate import compute_cate

    pop = _small_pop(n=800)
    n_agents = min(400, pop.N)

    # base is uniform-low; cf shifts up by age-correlated amount — so delta is
    # non-zero and structured, letting the forest pick up a signal.
    base = {int(i): 0.10 for i in range(n_agents)}
    cf = {int(i): 0.10 + 0.20 * (pop.age_idx[i] / 5.0) for i in range(n_agents)}

    out_ab = compute_cate(pop, base, cf)
    out_ba = compute_cate(pop, cf, base)

    # Both must return the [importances, top_segments] pair (non-saturated)
    assert not (len(out_ab) == 1 and out_ab[0].get("_diagnostic"))
    assert not (len(out_ba) == 1 and out_ba[0].get("_diagnostic"))

    # compute_cate returns [{importances:...}, {top_segments:[{segment,delta,n}]}]
    def _segments(out):
        for block in out:
            if "top_segments" in block:
                return block["top_segments"]
        return []

    segs_ab = _segments(out_ab)
    segs_ba = _segments(out_ba)
    assert (
        len(segs_ab) >= 2 and len(segs_ba) >= 2
    ), f"need segments for both directions · got {len(segs_ab)} / {len(segs_ba)}"

    # Total absolute effect magnitude should match (sign differs)
    def _total_mag(segs):
        return sum(abs(s["delta"]) for s in segs)

    mag_ab = _total_mag(segs_ab)
    mag_ba = _total_mag(segs_ba)
    assert mag_ab > 0 and mag_ba > 0
    assert (
        abs(mag_ab - mag_ba) / max(mag_ab, mag_ba) < 0.2
    ), f"|CATE(A,B)| = {mag_ab:.4f} vs |CATE(B,A)| = {mag_ba:.4f} differ by >20%"

    # Sum of signed deltas on (A,B) must be opposite sign to (B,A)
    def _total_signed(segs):
        return sum(s["delta"] for s in segs)

    s_ab = _total_signed(segs_ab)
    s_ba = _total_signed(segs_ba)
    assert (
        s_ab * s_ba < 0
    ), f"antisymmetry violated: signed(A→B)={s_ab:.4f} · signed(B→A)={s_ba:.4f}"


# ========================================================= 3. Identity CF


def test_compute_cate_identity_is_zero_effect():
    """compute_cate(pop, A, A) must flag saturated_impression_set (no effect)."""
    from oransim.causal.cate import compute_cate

    pop = _small_pop()
    same = {int(i): 0.25 for i in range(min(200, pop.N))}

    out = compute_cate(pop, same, same)
    assert len(out) == 1
    assert out[0].get("_diagnostic") == "saturated_impression_set"


# ================================================= 4. Union semantics check


def test_compute_cate_union_not_intersection():
    """Union means: if baseline reaches {0..100} and cf reaches {50..150},
    we should see exposure-loss for 0..49 and exposure-gain for 101..150."""
    from oransim.causal.cate import compute_cate

    pop = _small_pop(n=1000)
    base = {int(i): 0.30 for i in range(200)}  # reaches 0..199
    cf = {int(i): 0.30 for i in range(100, 300)}  # reaches 100..299, same prob

    out = compute_cate(pop, base, cf)
    # With union, the delta vector has non-zero entries (agents in base-only
    # get delta=-0.30; agents in cf-only get delta=+0.30). Forest should
    # return valid segments (not the saturation diagnostic).
    is_diagnostic = len(out) == 1 and out[0].get("_diagnostic") is not None
    assert (
        not is_diagnostic
    ), "intersection-only semantics would zero out the symmetric gain/loss. " "Got: " + str(out[:2])


# ================================================== 5. Budget monotonicity


def test_larger_budget_yields_more_impressions():
    """Sanity check: within the pre-saturation range, impressions should
    monotonically increase with total_budget. Breakage here would mean the
    Hill + fatigue curves or agent impression sampling got flipped."""
    from oransim.agents.statistical import StatisticalAgents
    from oransim.causal.counterfactual import Scenario, ScenarioRunner
    from oransim.data.creatives import make_creative
    from oransim.platforms.xhs.world_model_legacy import PlatformWorldModel

    pop = _small_pop(n=1000)
    wm = PlatformWorldModel(pop)
    ag = StatisticalAgents(pop)
    runner = ScenarioRunner(wm, ag)

    creative = make_creative(creative_id="mono-001", caption="test", duration_sec=15.0)

    def run_with_budget(b: float) -> float:
        scen = Scenario(
            creative=creative,
            total_budget=b,
            platform_alloc={"douyin": 0.5, "xhs": 0.5},
            seed=7,
        )
        return runner.run(scen, n_monte_carlo=1).total_kpis["impressions"]

    # Small budgets well below saturation
    imp_1k = run_with_budget(1_000)
    imp_5k = run_with_budget(5_000)
    assert imp_1k < imp_5k, f"monotonicity violated: budget=1k→{imp_1k}, budget=5k→{imp_5k}"


# ================================================ 6. Counterfactual round-trip


def test_counterfactual_empty_intervention_matches_baseline():
    """Running `counterfactual` with intervention that preserves everything
    should give KPIs within small MC noise of baseline. This is the Pearl
    consistency axiom applied to a null intervention: Y(do(T=T_obs)) ≈ Y_obs.
    """
    from oransim.agents.statistical import StatisticalAgents
    from oransim.causal.counterfactual import Scenario, ScenarioRunner
    from oransim.data.creatives import make_creative
    from oransim.platforms.xhs.world_model_legacy import PlatformWorldModel

    pop = _small_pop(n=600)
    wm = PlatformWorldModel(pop)
    ag = StatisticalAgents(pop)
    runner = ScenarioRunner(wm, ag)

    creative = make_creative(creative_id="rt-001", caption="test", duration_sec=15.0)
    base = Scenario(
        creative=creative,
        total_budget=4_000,
        platform_alloc={"douyin": 0.6, "xhs": 0.4},
        seed=11,
    )
    # Both sides MC=10 (was 5) to halve the sampling variance. Production
    # pipeline runs MC=20+ at 100k pop where the ROI estimator's variance
    # collapses naturally; here we're on n=600 + MC=10 so there's still
    # visible drift from the counterfactual's seed+777 perturbation.
    r_base = runner.run(base, n_monte_carlo=10)
    r_cf = runner.counterfactual(base, r_base, intervention={})

    roi_b = r_base.total_kpis["roi"]
    roi_cf = r_cf.total_kpis["roi"]
    # Threshold 35% set from empirical distribution over 25+ test-suite runs:
    # max observed was ~0.26. 35% gives ~10x slack over the modal ~3-10% and
    # ~35% over the 99th percentile — comfortable for a deliberately-perturbed
    # counterfactual on a tiny population, and still tight enough to catch a
    # real bug (e.g. if counterfactual stopped reusing the abducted residual
    # U, rel_diff would blow past 1.0).
    rel_diff = abs(roi_b - roi_cf) / max(abs(roi_b), 1e-6)
    assert rel_diff < 0.35, (
        f"null intervention should roughly preserve ROI · "
        f"baseline={roi_b:.3f} · cf={roi_cf:.3f} · rel_diff={rel_diff:.1%}"
    )


# ============================================================ 7. SCM graph shape
# Pin the documented invariants of the causal graph. Reviewer found the shipped
# graph has cycles that README 之前 called "Pearl SCM" — we now describe it as
# a "causal graph with feedback loops" and evaluate `do()` via the cyclic-SCM
# generalization (Bongers 2021). These tests lock the intentional shape so a
# refactor can't silently change it without updating the docs.


def test_scm_shape_is_intentional_cyclic_graph():
    """64 nodes · 117 edges · cycles concentrated in a single large SCC.

    This is the shape README describes in §Causal Graph. If a future edit
    makes the graph acyclic, update the README to drop the cyclic-SCM
    framing. If it pushes cycles into multiple SCCs, also revisit the
    fixed-point solve comment.
    """
    try:
        import networkx as nx
    except ImportError:
        import pytest

        pytest.skip("networkx not installed")
    from oransim.causal.scm import dag_dict

    g = dag_dict()
    assert g["n_nodes"] == 64
    assert g["n_edges"] == 117

    G = nx.DiGraph()
    for n in g["nodes"]:
        G.add_node(n["name"])
    for e in g["edges"]:
        G.add_edge(e[0], e[1])

    # Intentionally NOT a strict DAG — see §Causal Graph in README.
    assert not nx.is_directed_acyclic_graph(
        G
    ), "graph became acyclic — update README's cyclic-SCM framing if intentional"
    # Cycles are contained in a single large SCC (the long-term feedback
    # loop around brand_equity ↔ impression_dist). Multiple large SCCs
    # would mean poorly-posed fixed-point solves.
    large_sccs = [s for s in nx.strongly_connected_components(G) if len(s) > 1]
    assert (
        len(large_sccs) == 1
    ), f"expected 1 feedback SCC, got {len(large_sccs)} — cycle topology changed"
    # The one feedback SCC should be the long-term brand-to-bid loop; sanity
    # check a few expected members.
    scc = large_sccs[0]
    for expected in ("brand_equity", "ecpm_bid", "repurchase", "impression_dist"):
        assert expected in scc, f"{expected} missing from feedback SCC — graph refactor?"


def test_dag_dict_unrolled_is_strict_dag():
    """Time-unrolled projection of the causal graph must be strictly acyclic
    at any n_steps ≥ 1, so downstream modules (CausalDAG-Transformer,
    Pearl-abduction paths) get a proper DAG to operate on.
    """
    try:
        import networkx as nx
    except ImportError:
        import pytest

        pytest.skip("networkx not installed")
    from oransim.causal.scm import dag_dict, dag_dict_unrolled

    orig = dag_dict()
    for n_steps in (1, 2, 3):
        g = dag_dict_unrolled(n_steps=n_steps)
        assert g["n_nodes"] == orig["n_nodes"] * n_steps
        assert g["n_steps"] == n_steps
        # Edge count arithmetic:
        #   within_slice * n_steps + feedback * (n_steps - 1)
        n_fb = g["stats"]["n_feedback_edges"]
        n_within = g["stats"]["n_within_slice_edges"]
        expected_edges = n_within * n_steps + n_fb * max(0, n_steps - 1)
        assert g["n_edges"] == expected_edges

        G = nx.DiGraph()
        for node in g["nodes"]:
            G.add_node(node["name"])
        for s, d in g["edges"]:
            G.add_edge(s, d)
        assert nx.is_directed_acyclic_graph(
            G
        ), f"unrolled graph at n_steps={n_steps} still has cycles"


def test_dag_dict_unrolled_feedback_edges_cross_time():
    """Feedback edges in the unrolled graph must go from t→t+1, never t→t."""
    from oransim.causal.scm import dag_dict_unrolled

    g = dag_dict_unrolled(n_steps=2)
    fb_originals = {tuple(e) for e in g["feedback_edges"]}
    assert len(fb_originals) >= 5, "expected >=5 feedback edges in the shipped graph"

    # Count edges that cross time vs stay within
    cross = 0
    within = 0
    for s, d in g["edges"]:
        s_t = int(s.rsplit("_t", 1)[1])
        d_t = int(d.rsplit("_t", 1)[1])
        if s_t == d_t:
            within += 1
        elif d_t == s_t + 1:
            cross += 1
        else:
            raise AssertionError(f"edge {s}→{d} spans >1 time step")

    # Every feedback edge from the original graph contributes 1 cross-time
    # edge per (n_steps - 1) transitions. At n_steps=2, that's |feedback| × 1.
    assert cross == len(fb_originals)
    assert within == (g["stats"]["n_within_slice_edges"]) * 2  # 2 slices


def test_scm_edges_reference_defined_nodes():
    """Every edge endpoint must exist in the node list. Catches placeholder
    typos like a dead 'if False else' branch referencing a missing node."""
    from oransim.causal.scm import dag_dict

    g = dag_dict()
    names = {n["name"] for n in g["nodes"]}
    orphans = [e for e in g["edges"] if e[0] not in names or e[1] not in names]
    assert not orphans, f"edges reference undefined nodes: {orphans[:5]}"

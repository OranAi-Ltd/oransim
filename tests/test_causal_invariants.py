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
    # Both sides MC=5 so we compare two 5-sample means, not 1-sample vs 5.
    # counterfactual() uses seed+777 internally, so the comparison still
    # measures "is a null intervention close to baseline" under that shift.
    r_base = runner.run(base, n_monte_carlo=5)
    r_cf = runner.counterfactual(base, r_base, intervention={})

    roi_b = r_base.total_kpis["roi"]
    roi_cf = r_cf.total_kpis["roi"]
    # 25% rel_diff tolerance: the residual drift comes from seed+777 + independent
    # MC draws. With a small test population (n=600) the tail of KPI variance
    # leaks through even at MC=5; the production pipeline runs MC=20+ at 100k pop
    # where this naturally shrinks.
    rel_diff = abs(roi_b - roi_cf) / max(abs(roi_b), 1e-6)
    assert rel_diff < 0.25, (
        f"null intervention should roughly preserve ROI · "
        f"baseline={roi_b:.3f} · cf={roi_cf:.3f} · rel_diff={rel_diff:.1%}"
    )

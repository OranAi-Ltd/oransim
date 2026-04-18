"""Contract tests for the 4 schema-output agents that were previously 0-byte
stubs and are now real implementations (kol_optimizer / kol_content_match /
tag_lift / content_type_coef).

Each test pins the response shape + at least one substantive field so that a
future regression (accidental revert to empty file, schema rename, broken
niche lookup) fails the CI gate instead of silently returning ``{"_error"}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def test_tag_lift_returns_non_empty_rows_for_synthetic_niche():
    from oransim.agents.tag_lift import compute_tag_lift

    out = compute_tag_lift(target_niche="美妆", top_k=5, min_support=2)
    assert "_error" not in out, f"tag_lift errored: {out}"
    assert out["target_niche"] == "beauty", "zh→en niche resolution broken"
    assert out["n_notes_global"] > 0
    assert out["rows"], "no Lift rows surfaced"
    row = out["rows"][0]
    for k in ("tag", "lift", "count_in_niche", "p_niche", "p_global"):
        assert k in row, f"tag_lift row missing field {k}"
    assert row["lift"] > 0


def test_tag_lift_accepts_english_niche_key():
    from oransim.agents.tag_lift import compute_tag_lift

    out = compute_tag_lift(target_niche="beauty", top_k=3, min_support=1)
    assert "_error" not in out
    assert out["target_niche"] == "beauty"


def test_content_type_coef_returns_format_coefficients():
    from oransim.agents.content_type_coef import compute_content_type_coefficients

    out = compute_content_type_coefficients(target_niche="美妆")
    assert "_error" not in out, f"content_type_coef errored: {out}"
    assert out["n_notes_total"] > 0
    assert out["rows"], "no format coefficients surfaced"
    row = out["rows"][0]
    for k in ("content_type", "coefficient", "mean_engagement_rate", "recommendation"):
        assert k in row
    assert row["coefficient"] > 0
    assert row["recommendation"] in ("优先投放", "可选", "不推荐")


def test_kol_optimizer_milp_solves_and_respects_budget():
    from oransim.agents.kol_optimizer import optimize_kol_mix

    out = optimize_kol_mix(total_budget=50_000, target_niches=["美妆"])
    assert "_error" not in out, f"kol_optimizer errored: {out}"
    assert out["total_selected"] > 0, "milp picked nothing"
    # MILP should succeed or fall back to greedy, not fail entirely
    assert out["solver_status"] in (
        "milp_optimal",
        "greedy_fallback",
        "greedy_fallback_no_scipy",
        "greedy_fallback_infeasible",
    ) or out["solver_status"].startswith("greedy_fallback_")
    # Budget constraint: estimated_cost should not exceed budget
    assert out["estimated_cost"] <= out["budget"] * 1.01  # tiny float slack
    # Every selected KOL has the full schema shape
    kol = out["selected_kols"][0]
    for k in ("kol_id", "name", "niche", "tier", "fans", "cost", "roi"):
        assert k in kol


def test_kol_content_match_returns_ranked_top_k():
    from oransim.agents.kol_content_match import match_kol_content

    out = match_kol_content(
        own_brand="珂莱欧",
        category="美妆",
        target_niches=["beauty"],
        caption="春季裸感底妆",
        top_k=5,
    )
    assert "_error" not in out, f"kol_content_match errored: {out}"
    assert out["top_k"] == 5
    assert len(out["rows"]) == 5
    # Ranked by match_score descending
    scores = [r["match_score"] for r in out["rows"]]
    assert scores == sorted(scores, reverse=True)
    # Top pick should have been boosted by niche match (score > 0)
    assert out["rows"][0]["match_score"] > 0


def test_kol_optimizer_reinvest_ranking_shape():
    from oransim.agents.kol_optimizer import optimize_kol_mix, reinvest_ranking

    plan = optimize_kol_mix(total_budget=30_000, target_niches=["美妆"])
    rows = reinvest_ranking(plan["selected_kols"])
    assert len(rows) == plan["total_selected"]
    for r in rows:
        assert r["recommendation"] in ("优先复投", "观望", "替换")
        assert r["cost_trend"] in ("rising", "stable", "falling")
        assert r["reinvest_rank"] >= 1


def test_schema_outputs_integration_no_error_field_leaks():
    """Integration: the wrapper in schema_outputs.py should now produce these
    4 sections with real payloads, not the {"_error": ...} fallback."""
    from oransim.agents.schema_outputs import build_schema_outputs

    scenario_summary = {
        "total_budget": 50_000,
        "platform_alloc": {"douyin": 0.5, "xhs": 0.5},
        "caption": "春季裸感底妆测评",
    }
    fake_kpis = {
        "impressions": 100000.0,
        "clicks": 2000.0,
        "conversions": 200.0,
        "revenue": 30000.0,
        "cost": 50000.0,
        "ctr": 0.02,
        "cvr": 0.1,
        "roi": -0.4,
    }
    out = build_schema_outputs(
        kpis=fake_kpis,
        lifecycle={"day_axis": list(range(14)), "total_daily": [0] * 14},
        soul_quotes=[],
        per_platform={"xhs": {"kpi": fake_kpis}, "douyin": {"kpi": fake_kpis}},
        predicted_sentiment={"positive": 0.6, "neutral": 0.3, "negative": 0.1},
        extras={},
        scenario_summary=scenario_summary,
        target_niches=["beauty"],
        enable_kol_ilp=True,
        enable_search_elasticity=False,  # avoid loading optional model
    )
    # These 4 schema blocks used to be silent "_error" — now they must surface
    # a non-error payload.
    for key in (
        "T2_A1_kol_mix_optimization",
        "T2_A2_kol_content_match",
        "T2_A3_tag_lift_ranking",
        "T3_A7_content_type_coefficient",
    ):
        payload = out.get(key)
        assert payload is not None, f"{key} missing from schema_outputs"
        assert not (
            isinstance(payload, dict) and "_error" in payload
        ), f"{key} still fell into _error branch: {payload}"

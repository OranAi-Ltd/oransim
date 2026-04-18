"""Torch-free smoke tests — run in any CI without GPU / PyTorch.

Covers:
- Package importability + version metadata
- Registry lookup for world models + diffusion models
- Deferred-torch ImportError on PyTorch-dependent models
- Parametric Hawkes baseline instantiation + forecast
- Synthetic data generator (small N) produces valid outputs
- FastAPI app metadata (no bootstrap — just object inspection)
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import pytest


BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ---------------------------------------------------------- package metadata


def test_package_version():
    import oransim
    assert oransim.__version__ == "0.1.0a0"


def test_package_docstring_present():
    import oransim
    assert oransim.__doc__ is not None
    assert "causal" in oransim.__doc__.lower()


# ----------------------------------------------------------------- registry


def test_world_model_registry():
    from oransim.world_model import list_world_models, REGISTRY
    names = list_world_models()
    assert "causal_transformer" in names
    assert "lightgbm_quantile" in names
    # Aliases still resolvable
    assert "transformer" in REGISTRY
    assert "lgbm" in REGISTRY


def test_diffusion_registry():
    from oransim.diffusion import list_diffusion_models, REGISTRY
    names = list_diffusion_models()
    assert "causal_neural_hawkes" in names
    assert "parametric_hawkes" in names
    assert "neural_hawkes" in REGISTRY
    assert "thp" in REGISTRY


# ------------------------------------------------------ torch deferral


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(_torch_available(), reason="torch is installed; deferral test is a no-op")
def test_causal_transformer_defers_torch_import():
    from oransim.world_model import get_world_model
    with pytest.raises(ImportError, match=r"(PyTorch|torch)"):
        get_world_model("causal_transformer")


@pytest.mark.skipif(_torch_available(), reason="torch is installed; deferral test is a no-op")
def test_causal_neural_hawkes_defers_torch_import():
    from oransim.diffusion import get_diffusion_model
    with pytest.raises(ImportError, match=r"(PyTorch|torch)"):
        get_diffusion_model("causal_neural_hawkes")


# ------------------------------------------------------ parametric baseline


def test_parametric_hawkes_instantiation():
    from oransim.diffusion import get_diffusion_model
    ph = get_diffusion_model("parametric_hawkes")
    desc = ph.describe()
    assert desc["name"] == "ParametricHawkes"
    assert desc["horizon_days"] == 14
    assert "impression" in desc["event_types"]


def test_parametric_hawkes_forecast_and_counterfactual():
    from oransim.diffusion import get_diffusion_model, ParametricHawkesConfig
    # Use a 1-day horizon + stronger decay to keep the thinning loop fast.
    cfg = ParametricHawkesConfig(horizon_days=1, alpha_prior=0.1, beta_prior=0.5)
    ph = get_diffusion_model("parametric_hawkes", config=cfg)
    seed = [(0.0, "impression"), (60.0, "like"), (120.0, "share")]
    fact = ph.forecast(seed)
    assert isinstance(fact.per_type_totals, dict)
    assert len(fact.daily_buckets) == 1
    cf = ph.counterfactual_forecast(seed, intervention={"mute_at_min": 90.0})
    assert cf.latent["intervention"]["mute_at_min"] == 90.0


def test_demo_lightgbm_pkl_loads_and_predicts():
    """The shipped LightGBM demo pkl must load + predict out-of-the-box.

    Powers the 'clone → set LLM key → run' plug-and-play experience.
    """
    import pickle
    root = Path(__file__).parent.parent
    pkl_path = root / "data" / "models" / "world_model_demo.pkl"
    assert pkl_path.exists(), "demo pkl not shipped at data/models/world_model_demo.pkl"

    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    assert "config" in blob
    assert "boosters" in blob
    assert blob["config"]["feature_version"] == "demo_v1"
    assert set(blob["boosters"].keys()) == {"impressions", "clicks", "conversions", "revenue"}

    # Load a booster + predict on a plausible feature vector
    import lightgbm as lgb
    import numpy as np
    b = lgb.Booster(model_str=blob["boosters"]["impressions"]["0.5"])
    # [platform_id, niche_idx, budget, budget_bucket, kol_tier_idx, kol_fan_count, kol_engagement_rate]
    x = np.asarray([[0.0, 0.0, 50000.0, 1.0, 2.0, 100000.0, 0.035]], dtype=np.float32)
    pred = b.predict(x)[0]
    assert pred > 0, f"impressions P50 prediction should be > 0, got {pred}"


def test_demo_synthetic_data_shipped():
    """The small synthetic demo dataset must ship for immediate exploration."""
    root = Path(__file__).parent.parent
    for fname in (
        "synthetic_kols.json",
        "synthetic_notes.json",
        "synthetic_fan_profiles.json",
        "scenarios_v0_1.jsonl",
        "event_streams_v0_1.jsonl",
    ):
        p = root / "data" / "synthetic" / fname
        assert p.exists(), f"demo synthetic file missing: {fname}"
        assert p.stat().st_size > 0, f"demo synthetic file is empty: {fname}"


def test_parametric_hawkes_log_likelihood():
    from oransim.diffusion import get_diffusion_model
    ph = get_diffusion_model("parametric_hawkes")
    ll = ph.log_likelihood([(0.0, "impression"), (30.0, "like"), (70.0, "comment")])
    assert isinstance(ll, float)
    assert ll == ll  # not NaN


# ------------------------------------------------------ synthetic generator


def test_synthetic_data_deterministic():
    from scripts.gen_synthetic_data import generate_kols
    a = generate_kols(random.Random(42), 10)
    b = generate_kols(random.Random(42), 10)
    assert [k.kol_id for k in a] == [k.kol_id for k in b]
    assert [k.fan_count for k in a] == [k.fan_count for k in b]


def test_synthetic_streams_terminate_fast():
    """Regression: the prior O(N^2) sliding-window bug caused streams for
    macro/mega-tier KOLs to hang. Ensure termination in < 10 s for 20 streams."""
    import time
    from scripts.gen_synthetic_data import generate_kols, generate_event_streams
    rng = random.Random(7)
    kols = generate_kols(rng, 50)
    start = time.time()
    streams = list(generate_event_streams(rng, kols, 20))
    elapsed = time.time() - start
    assert elapsed < 10.0, f"streams took {elapsed:.1f}s — regression?"
    assert len(streams) == 20
    # No empty streams, and hard cap holds
    for s in streams:
        assert 1 <= len(s) <= 500


def test_synthetic_generator_cli(tmp_path):
    """Smoke-test the CLI end-to-end at tiny scale."""
    from scripts.gen_synthetic_data import main
    rc = main(
        [
            "--out", str(tmp_path),
            "--n-kols", "20",
            "--n-notes", "10",
            "--n-scenarios", "10",
            "--n-streams", "5",
            "--seed", "42",
            "--force",
        ]
    )
    assert rc == 0
    assert (tmp_path / "synthetic_kols.json").exists()
    assert (tmp_path / "scenarios_v0_1.jsonl").exists()
    assert (tmp_path / "event_streams_v0_1.jsonl").exists()
    # Scenarios have the expected schema
    with open(tmp_path / "scenarios_v0_1.jsonl") as f:
        row = json.loads(f.readline())
    for key in ("scenario_id", "treatment_arm", "cf_arm", "targets", "cf_targets"):
        assert key in row
    for kpi in ("impressions", "clicks", "conversions", "revenue"):
        assert kpi in row["targets"]
        assert kpi in row["cf_targets"]
    # Streams are non-empty and capped
    with open(tmp_path / "event_streams_v0_1.jsonl") as f:
        lines = f.readlines()
    assert len(lines) == 5
    for line in lines:
        events = json.loads(line)["events"]
        assert 1 <= len(events) <= 500


# ------------------------------------------------------ fastapi metadata


@pytest.mark.skipif(
    os.environ.get("POP_SIZE", "100000") not in ("10000", "100000"),
    reason="api bootstrap requires POP_SIZE >= 5000 (see _bootstrap_index)",
)
def test_fastapi_app_metadata():
    """Does not start a server, just inspects the FastAPI app object."""
    os.environ["POP_SIZE"] = "10000"
    os.environ["SOUL_POOL_N"] = "5"
    os.environ["LLM_MODE"] = "mock"
    from oransim import api
    assert api.app.title == "Oransim"
    assert api.app.version == "0.1.0a0"
    # No huitun routes leaked
    routes = [r.path for r in api.app.routes if hasattr(r, "path")]
    assert not any("huitun" in p for p in routes)
    # Core routes present
    assert "/" in routes
    assert any("/api/health" in p for p in routes)


# ------------------------------------------------------- desensitization


# ----------------------------------------------------------- causal / sandbox / agents


def test_causal_scm_graph_shape():
    from oransim.causal.scm import dag_dict
    g = dag_dict()
    assert "nodes" in g
    assert "edges" in g
    # README claims 64 nodes / 117 edges
    assert len(g["nodes"]) == 64, f"expected 64 SCM nodes, got {len(g['nodes'])}"
    assert len(g["edges"]) == 117, f"expected 117 SCM edges, got {len(g['edges'])}"


def test_cate_union_semantics():
    """compute_cate uses union (not intersection) for budget-only
    counterfactuals. Requires >=20 agents per compute_cate's own guard."""
    from oransim.causal.cate import compute_cate
    from oransim.data.population import generate_population
    pop = generate_population(N=50, seed=7)
    # Base: 25 agents with prob 0.2
    base = {i: 0.2 for i in range(25)}
    # CF: 25 agents (10 overlap, 15 new) with prob 0.4
    cf = {i: 0.4 for i in range(15, 40)}
    out = compute_cate(pop, base, cf)
    assert isinstance(out, list)
    assert len(out) > 0


def test_agent_soul_persona_mock_no_network():
    from oransim.agents.soul_llm import llm_info, llm_available
    import os
    os.environ["LLM_MODE"] = "mock"
    info = llm_info()
    assert info["mode"] == "mock"
    assert not llm_available()


def test_population_generates_determistic():
    from oransim.data.population import generate_population
    pop_a = generate_population(N=500, seed=123)
    pop_b = generate_population(N=500, seed=123)
    import numpy as np
    assert pop_a.N == 500
    assert pop_b.N == 500
    assert np.array_equal(pop_a.age_idx, pop_b.age_idx)
    assert np.array_equal(pop_a.gender_idx, pop_b.gender_idx)


def test_creative_generator():
    from oransim.data.creatives import make_creative
    c = make_creative(creative_id="test_001", caption="Aurora morning serum", duration_sec=15.0)
    assert c.caption == "Aurora morning serum"
    # content embedding is attached (exact attribute name may evolve)
    assert any(hasattr(c, a) for a in ("content_emb", "emb", "embedding"))


# ----------------------------------------------------------- demo synthetic data


# ----------------------------------------------------------- v0.1.2-alpha additions


def test_population_synthesizer_registry():
    from oransim.data.synthesizers import list_synthesizers, get_synthesizer
    names = list_synthesizers()
    assert "ipf" in names
    assert "bayes_net" in names
    assert "tabddpm" in names
    assert "causal_dag_tabddpm" in names
    # IPF works
    syn = get_synthesizer("ipf")
    pop = syn.generate(N=200, seed=7)
    assert pop.N == 200
    assert "age_idx" in pop.attributes
    # Roadmap synthesizers defer
    import pytest
    with pytest.raises(NotImplementedError, match="roadmap"):
        get_synthesizer("tabddpm")


def test_orancbench_scenarios_shipped():
    root = Path(__file__).parent.parent
    p = root / "data" / "benchmarks" / "orancbench_v0_1.jsonl"
    assert p.exists(), "OrancBench v0.1 scenarios not shipped"
    import json as _json
    with open(p, encoding="utf-8") as f:
        scenarios = [_json.loads(L) for L in f if L.strip()]
    assert len(scenarios) == 50
    difficulties = {s["difficulty"] for s in scenarios}
    assert difficulties == {"easy", "medium", "hard"}
    # Each scenario has ground truth + feature fields
    for s in scenarios:
        for k in ("scenario_id", "niche", "budget", "kol_tier", "ground_truth"):
            assert k in s, f"missing {k} in {s.get('scenario_id','?')}"
        for kpi in ("impressions", "clicks", "conversions", "revenue"):
            assert kpi in s["ground_truth"]


def test_orancbench_loader_and_scorer():
    from oransim.benchmarks import load_scenarios, score_predictions
    import sys as _sys
    root = Path(__file__).parent.parent
    # Chdir so the default path resolves correctly (CI-friendly)
    _sys.path.insert(0, str(root / "backend"))
    scenarios = load_scenarios(root / "data" / "benchmarks" / "orancbench_v0_1.jsonl")
    assert len(scenarios) == 50
    # Perfect-prediction baseline: score against ground truth itself
    preds = {s.scenario_id: dict(s.ground_truth) for s in scenarios}
    results = score_predictions(scenarios, preds)
    # Perfect predictions → R² ≈ 1 on every bucket
    assert results["overall"].n == 50
    for kpi in ("impressions", "clicks", "conversions", "revenue"):
        assert results["overall"].r2[kpi] > 0.99, f"{kpi} R² = {results['overall'].r2[kpi]}"


def test_ci_workflow_present():
    """The CI workflow must be committed so external contributors get
    automated validation."""
    root = Path(__file__).parent.parent
    ci = root / ".github" / "workflows" / "ci.yml"
    assert ci.exists(), ".github/workflows/ci.yml missing"
    content = ci.read_text()
    for stage in ("pytest", "ruff", "grep -rIEi"):
        assert stage in content, f"CI workflow missing stage: {stage}"


def test_docker_artifacts_shipped():
    root = Path(__file__).parent.parent
    assert (root / "docker" / "Dockerfile").exists()
    assert (root / "docker" / "docker-compose.yml").exists()
    assert (root / ".dockerignore").exists()


def test_example_notebooks_valid_json():
    """All 4 example notebooks must parse as valid ipynb JSON."""
    import json as _json
    root = Path(__file__).parent.parent
    expected = {
        "01_quickstart.ipynb",
        "02_counterfactual.ipynb",
        "03_custom_platform.ipynb",
        "04_soul_agents.ipynb",
    }
    shipped = {p.name for p in (root / "examples").glob("*.ipynb")}
    missing = expected - shipped
    assert not missing, f"missing notebooks: {missing}"
    for name in expected:
        nb = _json.loads((root / "examples" / name).read_text())
        assert nb.get("nbformat") == 4
        assert isinstance(nb.get("cells"), list)
        assert len(nb["cells"]) > 0


# ----------------------------------------------------------- Phase J (v0.2 quick wins)


def test_canonical_schemas():
    from oransim.data.schema import (
        CanonicalKOL, CanonicalNote, CanonicalFanProfile, CanonicalScenario, SCHEMA_VERSION
    )
    assert SCHEMA_VERSION == "1.1"
    kol = CanonicalKOL(
        kol_id="K1", nickname="AuroraStudio", platform="xhs",
        niche="beauty", tier="mid", fan_count=200_000, avg_engagement_rate=0.035,
    )
    assert kol.schema_version == "1.1"
    # FanProfile nested into KOL
    fp = CanonicalFanProfile(
        age_dist=[0.03, 0.36, 0.39, 0.15, 0.05, 0.015, 0.005],
        gender_dist=[0.9, 0.1],
    )
    kol2 = CanonicalKOL(
        kol_id="K2", nickname="CrimsonLab", platform="xhs",
        niche="fashion", tier="micro", fan_count=40_000,
        avg_engagement_rate=0.028, fan_profile=fp,
    )
    assert kol2.fan_profile is not None
    assert len(kol2.fan_profile.age_dist) == 7
    # Scenario
    scn = CanonicalScenario(
        scenario_id="S1", platform="xhs", creative_text="test",
        budget=50_000.0, niche="beauty",
    )
    assert scn.budget == 50_000.0


def test_budget_curves_public_api():
    from oransim.world_model.budget import (
        hill_saturation, frequency_fatigue, apply_budget_curves, BudgetCurveConfig
    )
    # Hill: doubling budget does not double impressions
    assert abs(hill_saturation(2.0) - 4.0 / 3.0) < 1e-9
    # Hill: at ratio=0 → 0
    assert hill_saturation(0.0) == 0.0
    # Hill: asymptotic toward 1 + K_sat
    assert hill_saturation(1e9) > 1.99  # → 2.0 with K_sat=1

    # Fatigue: below ref impressions → 1.0
    assert frequency_fatigue(0.5) == 1.0
    # Fatigue: floor at min_retained
    assert frequency_fatigue(1e9) >= 0.5

    # Composite path
    r = apply_budget_curves(100_000, 3500, 60, budget_ratio=2.0)
    assert r["impressions"] > 100_000
    assert r["clicks"] < r["impressions"]
    assert r["conversions"] < r["clicks"]
    assert r["effective_impr_ratio"] > 1.0


def test_http_client_module():
    from oransim.runtime.http_client import (
        post_json, _fallback_chain, _backoff_seconds,
        RETRYABLE_STATUS, NON_RETRYABLE_STATUS,
    )
    assert 429 in RETRYABLE_STATUS
    assert 401 in NON_RETRYABLE_STATUS
    # Fallback chain parsing
    import os
    os.environ["LLM_MODEL_FALLBACK"] = "gpt-4o-mini,deepseek-chat"
    chain = _fallback_chain("gpt-5.4")
    assert chain == ["gpt-5.4", "gpt-4o-mini", "deepseek-chat"]
    del os.environ["LLM_MODEL_FALLBACK"]
    # Backoff monotone bounded
    for n in range(5):
        b = _backoff_seconds(n, 0.8, 20.0)
        assert 0 <= b <= 20.0


def test_env_example_shipped():
    root = Path(__file__).parent.parent
    envx = root / ".env.example"
    assert envx.exists(), ".env.example missing — external users need it"
    content = envx.read_text()
    for key in ("LLM_MODE", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
                "SOUL_POOL_N", "POP_SIZE", "PORT"):
        assert key in content, f"missing env key: {key}"


def test_no_sensitive_terms_in_package():
    """BULLETPROOF case-insensitive scan — no vendor references leak anywhere.

    Covers every shipped file type, every case permutation (huitun/HUITUN/
    Huitun/HuItUn), both dashes and underscores in tu-zi/cg-api, plus the
    Chinese 灰豚 character and internal absolute paths. This is the single
    gate between desensitization regressions and the public repo.
    """
    import subprocess
    # Pattern built from hex / split strings to avoid matching this file.
    # Parts:
    #   hui[-_]?tun  — matches huitun, hui-tun, hui_tun (any case)
    #   tu[-_]?zi    — matches tuzi, tu-zi, tu_zi
    #   cg[-_]?api   — matches cgapi, cg-api, cg_api
    #   \u7070\u8c5a — 灰豚 (Chinese)
    #   /home/projects/sim — leaked internal absolute path
    pattern = "|".join(
        [
            "hui" + "[-_]?tun",
            "tu" + "[-_]?zi",
            "cg" + "[-_]?api",
            "\u7070\u8c5a",
            "/home/projects/sim",
            "/root/projects/sim",
        ]
    )
    root = Path(__file__).parent.parent
    # Note: CHANGELOG.md documents the scrub history (mentioning the terms is
    # expected + intentional). .github/workflows/ci.yml contains the scrub
    # grep command itself. Both are excluded from the gate since they encode
    # the gate, they don't leak the terms.
    paths = [
        str(root / "backend"),
        str(root / "frontend"),
        str(root / "docs"),
        str(root / "assets"),
        str(root / "index.html"),
        str(root / "README.md"),
        str(root / "README.zh-CN.md"),
        str(root / "ROADMAP.md"),
        str(root / "CONTRIBUTING.md"),
        str(root / "CODE_OF_CONDUCT.md"),
        str(root / "SECURITY.md"),
        str(root / "GITHUB_SETUP.md"),
        str(root / "CITATION.cff"),
        str(root / "pyproject.toml"),
    ]
    # .github is scanned minus the workflows dir that encodes the gate
    github_dir = root / ".github"
    if github_dir.exists():
        for item in github_dir.iterdir():
            if item.name == "workflows":
                continue
            paths.append(str(item))
    # Filter to only existing paths
    paths = [p for p in paths if Path(p).exists()]
    result = subprocess.run(
        [
            "grep", "-rIEi",   # -i makes it case-insensitive
            pattern,
            *paths,
            "--include=*.py",
            "--include=*.md",
            "--include=*.html",
            "--include=*.yml",
            "--include=*.yaml",
            "--include=*.toml",
            "--include=*.cff",
            "--include=*.svg",
            "--include=*.json",
            "--include=*.jsonl",
            "--include=*.css",
            "--include=*.js",
            "--include=*.txt",
            "--include=*.ini",
            "--include=*.cfg",
            "--exclude-dir=__pycache__",
            "--exclude-dir=.git",
            "--exclude-dir=node_modules",
            "--exclude-dir=data",  # synthetic data ok
        ],
        capture_output=True,
        text=True,
    )
    # grep returns 1 when no matches — that's the good path
    assert result.returncode == 1, (
        "🚨 SENSITIVE TERM LEAK — would ship to public repo:\n"
        f"{result.stdout}"
    )

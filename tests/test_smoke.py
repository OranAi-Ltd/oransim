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
from pathlib import Path

import pytest

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ---------------------------------------------------------- package metadata


def test_package_version():
    import oransim

    assert oransim.__version__ == "0.2.0a0"


def test_package_docstring_present():
    import oransim

    assert oransim.__doc__ is not None
    assert "causal" in oransim.__doc__.lower()


# ----------------------------------------------------------------- registry


def test_world_model_registry():
    from oransim.world_model import REGISTRY, list_world_models

    names = list_world_models()
    assert "causal_transformer" in names
    assert "lightgbm_quantile" in names
    # Aliases still resolvable
    assert "transformer" in REGISTRY
    assert "lgbm" in REGISTRY


def test_diffusion_registry():
    from oransim.diffusion import REGISTRY, list_diffusion_models

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


@pytest.mark.skipif(not _torch_available(), reason="requires torch")
def test_counterfactual_head_vectorized_matches_naive_loop():
    """Pin the vectorized CounterfactualHead — it must produce identical output
    to the prior .item()-loop implementation, since the speedup win only
    matters if numerics are preserved. Regressing to the loop would silently
    bring back per-row GPU syncs at training time.
    """
    import torch
    import torch.nn as nn
    from oransim.world_model import get_world_model

    wm = get_world_model("causal_transformer")
    # cf_heads is a ModuleList with one CounterfactualHead per KPI; grab one.
    cf_heads_list = getattr(wm, "cf_heads", None) or getattr(wm, "_net", None).cf_heads
    assert cf_heads_list is not None, "counterfactual head not constructed"
    head = cf_heads_list[0]
    assert hasattr(head, "arms") and len(head.arms) >= 2

    torch.manual_seed(7)
    for m in head.arms[0].modules():
        if isinstance(m, nn.Linear):
            d_model = m.in_features
            break

    B = 32
    h = torch.randn(B, d_model)
    arm_idx = torch.randint(0, len(head.arms), (B,))

    head.eval()
    with torch.no_grad():
        y_vec = head(h, arm_idx)
        # Reference: naive loop, slice arm-by-arm
        y_ref = torch.cat(
            [head.arms[int(arm_idx[b].item())](h[b : b + 1]) for b in range(B)],
            dim=0,
        )

    assert y_vec.shape == y_ref.shape
    # FP noise tolerance
    assert (y_vec - y_ref).abs().max().item() < 1e-5


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
    from oransim.diffusion import ParametricHawkesConfig, get_diffusion_model

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
    # demo_v2+: tabular + PCA-reduced text embedding. Older demo_v1 pkls
    # (tabular-only, 7 features) still load but don't carry a "pca" block.
    assert blob["config"]["feature_version"] in ("demo_v1", "demo_v2")
    assert set(blob["boosters"].keys()) == {"impressions", "clicks", "conversions", "revenue"}

    import lightgbm as lgb
    import numpy as np

    b = lgb.Booster(model_str=blob["boosters"]["impressions"]["0.5"])
    feat_dim = b.num_feature()
    assert feat_dim in (7, 23), f"unexpected booster feature dim {feat_dim}"

    # Build a feature vector matching whichever version this pkl was trained at
    x_scalar = np.asarray([0.0, 0.0, 50000.0, 1.0, 2.0, 100000.0, 0.035], dtype=np.float32)
    if feat_dim == 23:
        # demo_v2: concat the same PCA-projected embedding the trainer used
        assert "pca" in blob, "demo_v2 pkl must carry PCA components"
        comps = np.asarray(blob["pca"]["components"], dtype=np.float32)
        mean = np.asarray(blob["pca"]["mean"], dtype=np.float32)
        emb_dim = int(blob["config"]["embedding_dim_raw"])
        assert comps.shape == (16, emb_dim)
        emb_pca = (np.zeros(emb_dim, dtype=np.float32) - mean) @ comps.T
        x = np.concatenate([x_scalar, emb_pca]).reshape(1, -1)
    else:
        x = x_scalar.reshape(1, -1)
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

    from scripts.gen_synthetic_data import generate_event_streams, generate_kols

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
            "--out",
            str(tmp_path),
            "--n-kols",
            "20",
            "--n-notes",
            "10",
            "--n-scenarios",
            "10",
            "--n-streams",
            "5",
            "--seed",
            "42",
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
    assert api.app.version == "0.2.0a0"
    # No internal-vendor routes leaked to the public API
    routes = [r.path for r in api.app.routes if hasattr(r, "path")]
    _forbidden_route = bytes.fromhex("68756974756e").decode()  # decoded at runtime
    assert not any(_forbidden_route in p for p in routes)
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
    import os

    from oransim.agents.soul_llm import llm_available, llm_info

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
    from oransim.data.synthesizers import get_synthesizer, list_synthesizers

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
    # Bayes net works (v0.2 lands)
    bn = get_synthesizer("bayes_net")
    bpop = bn.generate(N=200, seed=7)
    assert bpop.N == 200
    assert "income_tertile_idx" in bpop.attributes
    assert bpop.latent["backend"] == "bayesian_network"
    # Future synthesizers still defer
    import pytest

    with pytest.raises(NotImplementedError, match="roadmap"):
        get_synthesizer("tabddpm")


def test_bayes_net_synthesizer_respects_conditionals():
    """Higher-education buckets should skew toward higher income tertiles —
    a joint-dependency property IPF cannot represent."""
    import numpy as np
    from oransim.data.synthesizers import get_synthesizer

    bn = get_synthesizer("bayes_net")
    pop = bn.generate(N=5000, seed=42)
    edu = np.asarray(pop.attributes["edu_idx"])
    inc = np.asarray(pop.attributes["income_tertile_idx"])
    # Mean income tertile should increase with edu level
    means = [float(inc[edu == e].mean()) if (edu == e).any() else 0.0 for e in range(5)]
    # Undergrad (3) and grad (4) must have strictly higher mean income tertile
    # than junior (1) — core dependency the BN encodes that IPF loses.
    assert (
        means[4] > means[1]
    ), f"grad mean income {means[4]:.3f} should exceed junior {means[1]:.3f}"
    assert (
        means[3] > means[1]
    ), f"undergrad mean income {means[3]:.3f} should exceed junior {means[1]:.3f}"


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
    import sys as _sys

    from oransim.benchmarks import load_scenarios, score_predictions

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
    # The vendor-scrub grep was retired from public CI (self-defeating — it
    # enumerated the terms it was guarding against). The check lives in
    # test_no_sensitive_terms_in_package instead.
    for stage in ("pytest", "ruff"):
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


# ----------------------------------------------------------- Phase O (v0.2 gap closure)


def test_v2_endpoints_wired_to_registries():
    """Gap 1: /api/v2/* endpoints must route through model registries."""
    import os

    os.environ["POP_SIZE"] = "10000"
    os.environ["SOUL_POOL_N"] = "5"
    os.environ["LLM_MODE"] = "mock"
    from fastapi.testclient import TestClient
    from oransim import api

    c = TestClient(api.app)

    # Registry introspection
    r = c.get("/api/v2/registry")
    assert r.status_code == 200
    j = r.json()
    assert "causal_transformer" in j["world_model"]
    assert "causal_neural_hawkes" in j["diffusion"]
    assert "bayes_net" in j["synthesizer"]

    # LightGBM baseline — shipped pkl path
    r = c.post(
        "/api/v2/world_model/predict?model=lightgbm_quantile",
        json={
            "features": {
                "niche": "beauty",
                "kol_tier": "mid",
                "budget": 80_000,
                "budget_bucket": 2,
                "kol_fan_count": 240_000,
                "kol_engagement_rate": 0.042,
            }
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert "kpi_quantiles" in j
    assert j["kpi_quantiles"]["impressions"]["0.5"] > 0

    # Parametric Hawkes forecast
    r = c.post(
        "/api/v2/diffusion/forecast?model=parametric_hawkes",
        json={"seed_events": [[0.0, "impression"], [60.0, "like"]]},
    )
    assert r.status_code == 200
    assert r.json()["n_events_simulated"] > 0

    # Synthesizer
    r = c.post("/api/v2/synthesizer/generate?model=bayes_net", json={"N": 100, "seed": 42})
    assert r.status_code == 200
    assert r.json()["N"] == 100

    # Deferred synthesizer returns 501 not 500
    r = c.post("/api/v2/synthesizer/generate?model=tabddpm", json={"N": 100})
    assert r.status_code == 501


@pytest.mark.skipif(not _torch_available(), reason="CInA context test requires torch")
def test_cina_in_context_path():
    """Gap 2: CausalTransformer must accept a context= argument."""
    import torch
    from oransim.world_model import CausalTransformerWMConfig, CausalTransformerWorldModel

    cfg = CausalTransformerWMConfig(n_layers=2, d_model=64, n_heads=4, dag_attention_bias=False)
    wm = CausalTransformerWorldModel(cfg)

    def _make_features(seed):
        g = torch.Generator().manual_seed(seed)
        return {
            "creative_embed": torch.randn(cfg.creative_embed_dim, generator=g),
            "kol_feat": torch.randn(cfg.kol_feature_dim, generator=g),
            "demo_feat": torch.randn(cfg.demographic_feature_dim, generator=g),
            "platform_id": torch.tensor(0, dtype=torch.long),
            "budget": torch.tensor([0.5]),
            "time_feat": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        }

    query = _make_features(42)
    pred_no_ctx = wm.predict(query)
    assert pred_no_ctx.latent["context_size"] == 0

    ctx_entry = _make_features(100)
    ctx_entry["outcome"] = torch.tensor([1.0, 0.02, 0.001, 50.0])
    ctx_entry2 = _make_features(101)
    ctx_entry2["outcome"] = torch.tensor([0.8, 0.015, 0.0008, 42.0])
    pred_ctx = wm.predict(query, context=[ctx_entry, ctx_entry2])
    assert pred_ctx.latent["context_size"] == 2
    a = pred_no_ctx.kpi_quantiles["impressions"][0.50]
    b = pred_ctx.kpi_quantiles["impressions"][0.50]
    assert abs(a - b) > 1e-6, "CInA context token had no effect on the prediction"


@pytest.mark.skipif(not _torch_available(), reason="requires torch")
def test_cnhp_kv_cache_matches_full_forward():
    """Pin that the KV-cache incremental path produces the same hidden state
    (and thus the same intensity) as a full-sequence forward. Regression
    protection for the forecast() speedup.
    """
    import torch
    from oransim.diffusion import CausalNeuralHawkesConfig, CausalNeuralHawkesProcess

    torch.manual_seed(42)
    cfg = CausalNeuralHawkesConfig(n_layers=2, d_model=32, n_heads=4, dropout=0.0)
    nh = CausalNeuralHawkesProcess(cfg)
    net = nh._net
    net.eval()

    # 7-event stream so the incremental + full forwards can both produce a
    # last-position intensity to compare.
    events = [
        (0.0, "impression"),
        (15.0, "like"),
        (30.0, "impression"),
        (45.0, "share"),
        (60.0, "conversion"),
        (90.0, "like"),
        (120.0, "save"),
    ]
    type_ids, dt, treatment_ids, _ = nh._prep_stream(events)

    with torch.no_grad():
        # Full-seq path (what training uses)
        h_full = net(type_ids, dt, treatment_ids)
        lam_full = net.intensity(h_full)[:, -1]

        # Cached path: seed with the first N-1 events, then add the last one
        # incrementally. Final intensity at the new token should match lam_full.
        caches = net.new_kv_caches()
        _ = net(
            type_ids[:, :-1],
            dt[:, :-1],
            treatment_ids[:, :-1],
            kv_caches=caches,
            is_incremental=False,
        )
        h_inc = net(
            type_ids[:, -1:],
            dt[:, -1:],
            treatment_ids[:, -1:],
            kv_caches=caches,
            is_incremental=True,
        )
        lam_inc = net.intensity(h_inc)[:, -1]

    assert lam_full.shape == lam_inc.shape
    max_abs = (lam_full - lam_inc).abs().max().item()
    assert max_abs < 1e-5, f"KV-cache incremental diverges from full forward: {max_abs:.2e}"


@pytest.mark.skipif(not _torch_available(), reason="requires torch")
def test_dag_attention_handles_cyclic_scm_via_scc_condensation():
    """`set_dag_from_edges` must produce a meaningful mask even when the
    input graph has cycles. The shipped SCM contains a 25-node feedback
    SCC (README §Causal Graph); the old transitive-closure code silently
    collapsed every SCC member into the ancestor set of every other,
    degrading the bias to no-op. We now condense the graph to its SCC
    DAG and use "SCC-level" ancestry.
    """
    import torch
    from oransim.causal.scm import dag_dict
    from oransim.world_model import CausalTransformerWMConfig, CausalTransformerWorldModel

    torch.manual_seed(0)
    cfg = CausalTransformerWMConfig(n_layers=2, d_model=32, n_heads=4, dag_attention_bias=True)
    wm = CausalTransformerWorldModel(cfg)
    g = dag_dict()
    name_to_idx = {n["name"]: i for i, n in enumerate(g["nodes"])}
    edges = [(name_to_idx[p], name_to_idx[c]) for p, c in g["edges"]]

    # CLS + early-graph + in-SCC mix of tokens
    tokens = [
        -1,  # CLS
        name_to_idx["creative_caption"],
        name_to_idx["kol_choice"],
        name_to_idx["total_budget"],
        name_to_idx["macro_env"],
        name_to_idx["impression_dist"],  # in the 25-node feedback SCC
    ]
    wm.set_dag_from_edges(g["n_nodes"], edges, tokens)
    mask = wm._net._dag_bias
    assert mask is not None
    L = len(tokens)
    assert mask.shape == (L, L)

    # Mask must be meaningful — not all-zero (no-op), not all-inf
    n_blocked = int(torch.isinf(mask).sum().item())
    assert 0 < n_blocked < L * L, f"mask degenerate: blocked {n_blocked}/{L * L}"

    # CLS (token 0) is free → nobody blocks from/to CLS
    for j in range(L):
        assert not torch.isinf(mask[0, j]), "CLS row should have no -inf"
        assert not torch.isinf(mask[j, 0]), "CLS col should have no -inf"

    # impression_dist is in the feedback SCC, which is downstream of
    # creative / kol / budget / macro — so its row should allow attending
    # FROM all of those tokens (no -inf in its row across those cols).
    impr_row = L - 1
    for j in range(1, L - 1):  # the non-CLS, non-impr tokens
        assert not torch.isinf(
            mask[impr_row, j]
        ), f"impression_dist should be able to attend from token {j}"

    # creative_caption's row should block attention FROM impression_dist
    # (the SCC is NOT an ancestor of creative_caption in the condensation).
    creative_row = 1
    assert torch.isinf(
        mask[creative_row, impr_row]
    ), "creative_caption should not receive attention from impression_dist"


@pytest.mark.skipif(not _torch_available(), reason="requires torch")
def test_cnhp_batched_fit_equivalence():
    """Batched NLL (padded mini-batch) over K streams equals the sum of
    unbatched log-likelihoods on each stream. Pins that fit()'s batch_size
    path matches the reference per-stream computation to FP noise.
    """
    import torch
    from oransim.diffusion import CausalNeuralHawkesConfig, CausalNeuralHawkesProcess

    cfg = CausalNeuralHawkesConfig(n_layers=2, d_model=32, n_heads=4, dropout=0.0)
    torch.manual_seed(0)
    nh = CausalNeuralHawkesProcess(cfg)
    nh._net.eval()

    streams = [
        [(0.0, "impression"), (20.0, "like"), (60.0, "share")],
        [
            (0.0, "impression"),
            (15.0, "like"),
            (30.0, "impression"),
            (50.0, "conversion"),
        ],
        [
            (0.0, "impression"),
            (10.0, "comment"),
            (40.0, "like"),
            (80.0, "share"),
            (90.0, "save"),
        ],
    ]

    with torch.no_grad():
        type_ids, dt, treatment_ids, times, mask = nh._prep_batch(streams)
        h = nh._net(type_ids, dt, treatment_ids)
        lam = nh._net.intensity(h)
        event_lam = torch.gather(lam, 2, type_ids.unsqueeze(-1)).squeeze(-1)
        log_sum = (torch.log(event_lam.clamp(min=1e-12)) * mask).sum()
        comp = nh._integrate_compensator_batched(lam, times, mask)
        batched_nll = float(-(log_sum - comp).item())
        unbatched_nll = -sum(nh.log_likelihood(s) for s in streams)

    assert (
        abs(batched_nll - unbatched_nll) < 1e-2
    ), f"batched NLL {batched_nll} diverges from unbatched sum {unbatched_nll}"


@pytest.mark.skipif(not _torch_available(), reason="requires torch")
def test_cnhp_kv_cache_rollback_reverts_state():
    """Rejection in Ogata thinning rolls back the virtual-τ token from the
    cache. After rollback, a subsequent incremental call with the SAME new
    token must produce the same hidden state as it did before the rollback.
    """
    import torch
    from oransim.diffusion import CausalNeuralHawkesConfig, CausalNeuralHawkesProcess

    torch.manual_seed(7)
    cfg = CausalNeuralHawkesConfig(n_layers=2, d_model=32, n_heads=4, dropout=0.0)
    nh = CausalNeuralHawkesProcess(cfg)
    net = nh._net
    net.eval()

    type_ids, dt, treatment_ids, _ = nh._prep_stream(
        [(0.0, "impression"), (20.0, "like"), (40.0, "share")]
    )
    caches = net.new_kv_caches()
    with torch.no_grad():
        net(type_ids, dt, treatment_ids, kv_caches=caches, is_incremental=False)

        virtual_type = torch.tensor([[0]], dtype=torch.long)
        virtual_dt = torch.tensor([[5.0]], dtype=torch.float32)
        virtual_tr = torch.tensor([[0]], dtype=torch.long)

        h1 = net(virtual_type, virtual_dt, virtual_tr, kv_caches=caches, is_incremental=True)
        net.rollback_last_token(caches)
        h2 = net(virtual_type, virtual_dt, virtual_tr, kv_caches=caches, is_incremental=True)

    assert (h1 - h2).abs().max().item() < 1e-6, "rollback did not restore state"


@pytest.mark.skipif(not _torch_available(), reason="MC compensator test requires torch")
def test_mc_compensator_branch():
    """Gap 3: CausalNeuralHawkes compensator modes must actually branch."""
    import torch
    from oransim.diffusion import CausalNeuralHawkesConfig, CausalNeuralHawkesProcess

    events = [(float(i * 15.0), "impression" if i % 2 == 0 else "like") for i in range(12)]

    def _ll(mode):
        cfg = CausalNeuralHawkesConfig(
            n_layers=2, d_model=32, n_heads=4, compensator=mode, n_mc_samples=8
        )
        # Seed BEFORE model construction so default-init weights are
        # reproducible AND non-degenerate (constant-intensity weights would
        # collapse all three quadrature modes to the same integral).
        torch.manual_seed(0)
        nh = CausalNeuralHawkesProcess(cfg)
        return nh.log_likelihood(events)

    ll_rect = _ll("rectangle")
    ll_trap = _ll("trapezoidal")
    ll_mc = _ll("mc")
    for v in (ll_rect, ll_trap, ll_mc):
        assert isinstance(v, float)
        assert v == v  # not NaN
    # Rectangle vs trapezoidal differ whenever λ changes between adjacent
    # events; MC adds per-call sampling noise on top. Floor chosen to
    # distinguish true mode-branch from "same code path ran twice".
    assert abs(ll_trap - ll_rect) > 1e-6
    assert abs(ll_mc - ll_rect) > 1e-6


# ----------------------------------------------------------- Phase J (v0.2 quick wins)


def test_tiktok_adapter_mvp():
    """TikTok adapter + synthetic provider must fulfill the canonical contract."""
    from oransim.data.schema import CanonicalKOL, CanonicalNote
    from oransim.platforms.tiktok import TikTokAdapter, TikTokSyntheticProvider

    provider = TikTokSyntheticProvider(seed=42)
    adapter = TikTokAdapter(data_provider=provider)
    assert adapter.platform_id == "tiktok"

    # get_kol returns a canonical type
    kol = adapter.get_kol("TT_KOL_000042")
    assert isinstance(kol, CanonicalKOL)
    assert kol.platform == "tiktok"
    assert kol.tier in ("nano", "micro", "mid", "macro", "mega")
    assert kol.fan_profile is not None

    # search_notes returns canonical list
    notes = provider.search_notes("beauty", max_results=5)
    assert len(notes) == 5
    assert all(isinstance(n, CanonicalNote) for n in notes)
    assert all(n.platform == "tiktok" for n in notes)

    # Impressions grow non-linearly with budget (Hill saturation)
    from types import SimpleNamespace

    creative = SimpleNamespace(caption="ad copy", duration_sec=20.0)
    pred_1x = adapter.simulate_impression(creative, budget=50_000)
    pred_2x = adapter.simulate_impression(creative, budget=100_000)
    # Hill: doubling budget should NOT double impressions
    assert pred_2x["impressions"] > pred_1x["impressions"]
    assert pred_2x["impressions"] < 2.0 * pred_1x["impressions"]


def test_tiktok_agent_level_impression():
    """TikTok agent-level simulate_impression_agents must select real agent indices."""
    from oransim.data.creatives import make_creative
    from oransim.data.population import generate_population
    from oransim.platforms.tiktok import TikTokAdapter

    pop = generate_population(N=800, seed=7)
    adapter = TikTokAdapter(population=pop)
    creative = make_creative("CR_TT_01", "新品开箱 小众设计师", duration_sec=18.0)

    result = adapter.simulate_impression_agents(creative, budget_cny=30_000, rng_seed=0)
    # Non-empty selection
    assert len(result.agent_idx) > 0
    # Indices within population range
    assert int(result.agent_idx.max()) < pop.N
    # Weights in [0, 1]
    assert float(result.weight.max()) <= 1.0 + 1e-6
    # TikTok-specific score channel is present
    assert "duration_retention" in result.score_breakdown


def test_tiktok_fyp_rl_breakout_shape():
    """FYP RL loop produces a RecSysRLReport with the expected round layout."""
    from oransim.data.creatives import make_creative
    from oransim.data.population import generate_population
    from oransim.platforms.tiktok import TikTokAdapter
    from oransim.platforms.xhs.recsys_rl import rl_report_to_dict

    pop = generate_population(N=1200, seed=11)
    adapter = TikTokAdapter(population=pop)
    creative = make_creative("CR_TT_02", "夏日清单 好物分享", duration_sec=22.0)

    report = adapter.simulate_fyp_rl(creative, total_budget=60_000, seed=3)
    assert len(report.rounds) > 0 and len(report.rounds) <= 6
    # Platform-agnostic serializer accepts the TikTok report unchanged.
    serialized = rl_report_to_dict(report)
    assert serialized["n_rounds"] == len(report.rounds)
    # TikTok fractions grow geometrically; later rounds reach more agents.
    sizes = [len(r.impression_idx) for r in report.rounds]
    assert sizes[-1] >= sizes[0]


def test_tiktok_prs_stub_not_ready_in_oss():
    """TikTok PRS is a deliberate stub in OSS — is_ready must be False."""
    from oransim.platforms.tiktok import TikTokPRS

    prs = TikTokPRS()
    assert prs.is_ready() is False
    info = prs.info()
    assert info["loaded"] is False
    # predict() must fall through (empty dict) rather than raise when stubbed.
    import numpy as np

    assert prs.predict(caption_emb=np.zeros(64, np.float32), author_fans=1000, niche="beauty") == {}


def test_tiktok_adapter_agent_path_errors_without_population():
    """Accessing world_model / rl without a Population must raise clearly."""
    from oransim.platforms.tiktok import TikTokAdapter

    adapter = TikTokAdapter()
    import pytest

    with pytest.raises(RuntimeError, match="needs a Population"):
        _ = adapter.world_model


def test_instagram_adapter_mvp():
    """Instagram Reels adapter + synthetic provider must fulfill the canonical contract."""
    from oransim.data.schema import CanonicalKOL
    from oransim.platforms.instagram import InstagramAdapter, InstagramSyntheticProvider

    provider = InstagramSyntheticProvider(seed=42)
    adapter = InstagramAdapter(data_provider=provider)
    assert adapter.platform_id == "instagram"

    kol = adapter.get_kol("IG_KOL_000042")
    assert isinstance(kol, CanonicalKOL)
    assert kol.platform == "instagram"
    assert kol.fan_profile is not None

    from types import SimpleNamespace

    normal = SimpleNamespace(caption="promo", duration_sec=18.0, music_mood="upbeat")
    trending = SimpleNamespace(caption="promo", duration_sec=18.0, music_mood="trending")
    p_normal = adapter.simulate_impression(normal, budget=50_000)
    p_trending = adapter.simulate_impression(trending, budget=50_000)
    # Impressions unchanged but clicks and conversions boosted by trending-audio factor
    assert abs(p_normal["impressions"] - p_trending["impressions"]) < 1e-6
    assert p_trending["clicks"] > p_normal["clicks"]
    assert p_trending["conversions"] > p_normal["conversions"]


def test_youtube_shorts_adapter_mvp():
    """YouTube Shorts adapter + synthetic provider must fulfill the canonical contract."""
    from oransim.data.schema import CanonicalKOL
    from oransim.platforms.youtube_shorts import (
        YouTubeShortsAdapter,
        YouTubeShortsSyntheticProvider,
    )

    provider = YouTubeShortsSyntheticProvider(seed=42)
    adapter = YouTubeShortsAdapter(data_provider=provider)
    assert adapter.platform_id == "youtube_shorts"

    kol = adapter.get_kol("YS_KOL_000042")
    assert isinstance(kol, CanonicalKOL)
    assert kol.platform == "youtube_shorts"

    from types import SimpleNamespace

    no_cta = SimpleNamespace(caption="clip", duration_sec=30.0, has_subscribe_cta=False)
    with_cta = SimpleNamespace(caption="clip", duration_sec=30.0, has_subscribe_cta=True)
    p_no = adapter.simulate_impression(no_cta, budget=50_000)
    p_yes = adapter.simulate_impression(with_cta, budget=50_000)
    # Impressions + clicks unaffected; conversions lifted by subscribe CTA
    assert abs(p_no["impressions"] - p_yes["impressions"]) < 1e-6
    assert abs(p_no["clicks"] - p_yes["clicks"]) < 1e-6
    assert p_yes["conversions"] > p_no["conversions"]
    # Search long-tail factor > 1 (Shorts has that structural prior)
    assert p_no["factors"]["search_longtail_factor"] > 1.0


def test_douyin_adapter_mvp():
    """Douyin mirrors TikTok but with Greater-China priors + livestream boost."""
    from oransim.data.schema import CanonicalKOL
    from oransim.platforms.douyin import DouyinAdapter, DouyinSyntheticProvider

    provider = DouyinSyntheticProvider(seed=42)
    adapter = DouyinAdapter(data_provider=provider)
    assert adapter.platform_id == "douyin"

    kol = adapter.get_kol("DY_KOL_000042")
    assert isinstance(kol, CanonicalKOL)
    assert kol.platform == "douyin"
    assert kol.region and kol.region.startswith("CN")

    # Livestream boost
    from types import SimpleNamespace

    video = SimpleNamespace(caption="normal ad", duration_sec=18.0, visual_style="bright")
    livestream = SimpleNamespace(
        caption="live commerce", duration_sec=18.0, visual_style="livestream"
    )
    pred_video = adapter.simulate_impression(video, budget=50_000)
    pred_live = adapter.simulate_impression(livestream, budget=50_000)
    # Livestream has same impressions but higher conversions
    assert abs(pred_video["impressions"] - pred_live["impressions"]) < 1e-6
    assert pred_live["conversions"] > pred_video["conversions"]


def test_canonical_schemas():
    from oransim.data.schema import (
        SCHEMA_VERSION,
        CanonicalFanProfile,
        CanonicalKOL,
        CanonicalScenario,
    )

    assert SCHEMA_VERSION == "1.1"
    kol = CanonicalKOL(
        kol_id="K1",
        nickname="AuroraStudio",
        platform="xhs",
        niche="beauty",
        tier="mid",
        fan_count=200_000,
        avg_engagement_rate=0.035,
    )
    assert kol.schema_version == "1.1"
    # FanProfile nested into KOL
    fp = CanonicalFanProfile(
        age_dist=[0.03, 0.36, 0.39, 0.15, 0.05, 0.015, 0.005],
        gender_dist=[0.9, 0.1],
    )
    kol2 = CanonicalKOL(
        kol_id="K2",
        nickname="CrimsonLab",
        platform="xhs",
        niche="fashion",
        tier="micro",
        fan_count=40_000,
        avg_engagement_rate=0.028,
        fan_profile=fp,
    )
    assert kol2.fan_profile is not None
    assert len(kol2.fan_profile.age_dist) == 7
    # Scenario
    scn = CanonicalScenario(
        scenario_id="S1",
        platform="xhs",
        creative_text="test",
        budget=50_000.0,
        niche="beauty",
    )
    assert scn.budget == 50_000.0


def test_budget_curves_public_api():
    from oransim.world_model.budget import (
        apply_budget_curves,
        frequency_fatigue,
        hill_saturation,
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
        NON_RETRYABLE_STATUS,
        RETRYABLE_STATUS,
        _backoff_seconds,
        _fallback_chain,
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
    for key in (
        "LLM_MODE",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "SOUL_POOL_N",
        "POP_SIZE",
        "PORT",
    ):
        assert key in content, f"missing env key: {key}"


def test_no_sensitive_terms_in_package():
    """Case-insensitive scan — no internal-vendor / internal-path tokens leak.

    Patterns are hex-decoded at runtime so this file itself does not contain
    the plaintext tokens it's guarding against (which would defeat the check).
    """
    import subprocess

    # Build grep pattern from hex-encoded fragments so this test file itself
    # does not contain the plaintext tokens it's trying to detect. Two forms:
    #   - full hex strings → literal substring
    #   - "PREFIX+SEP+SUFFIX" → matches optional -/_ between PREFIX and SUFFIX
    def _decode_hex(h: str) -> str:
        return bytes.fromhex(h).decode("utf-8")

    # literal tokens (will be matched verbatim, case-insensitive via -Ei)
    _literal = [
        "68756974756e",  # ascii vendor
        "e781b0e8b19a",  # CJK vendor (utf-8)
        "2f686f6d652f70726f6a656374732f73696d",  # internal path /home/projects/sim
        "2f726f6f742f70726f6a656374732f73696d",  # internal path /root/projects/sim
    ]
    # (prefix_hex, suffix_hex) for tokens allowing optional -/_ between halves
    _joined = [
        ("7475", "7a69"),  # tu[-_]?zi
        ("6367", "617069"),  # cg[-_]?api
    ]
    parts = [_decode_hex(t) for t in _literal]
    parts.extend(f"{_decode_hex(p)}[-_]?{_decode_hex(s)}" for p, s in _joined)
    pattern = "|".join(parts)
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
            "grep",
            "-rIEi",  # -i makes it case-insensitive
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
        "🚨 SENSITIVE TERM LEAK — would ship to public repo:\n" f"{result.stdout}"
    )

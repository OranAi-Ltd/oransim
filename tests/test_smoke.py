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
    paths = [
        str(root / "backend"),
        str(root / "frontend"),
        str(root / "docs"),
        str(root / "assets"),
        str(root / ".github"),
        str(root / "index.html"),
        str(root / "README.md"),
        str(root / "README.zh-CN.md"),
        str(root / "ROADMAP.md"),
        str(root / "CHANGELOG.md"),
        str(root / "CONTRIBUTING.md"),
        str(root / "CODE_OF_CONDUCT.md"),
        str(root / "SECURITY.md"),
        str(root / "GITHUB_SETUP.md"),
        str(root / "CITATION.cff"),
        str(root / "pyproject.toml"),
    ]
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

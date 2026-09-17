"""Checks of the independent daily-count baseline's probability and time cutoffs."""

from copy import deepcopy
from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parents[2]
spec = spec_from_file_location(
    "population_count_baselines", ROOT / "backend/scripts/evaluate_population_count_baselines.py"
)
baseline = module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_nb_nll_matches_scipy_including_zero_counts():
    counts = np.array([0, 1, 20, 900])
    means = np.array([0.01, 2, 25, 700.0])
    for size in [0.1, 1.0, 1000.0]:
        np.testing.assert_allclose(
            baseline.nll(counts, means, size),
            -nbinom.logpmf(counts, size, size / (size + means)),
            atol=1e-9,
        )


def test_frozen_origin_means_and_daily_observation_updates():
    counts = [3, 20, 5, 40, 12, 9, 30, 50, 1, 20, 40]
    rows = [
        {"date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(), "count": y}
        for i, y in enumerate(counts)
    ]
    records = {"train": rows[:2], "validation": rows[2:4], "test": rows[4:]}
    parameters = {
        "stationary_mean": 10.0,
        "omega": 5.0,
        "alpha": 0.2,
        "beta": 0.3,
        "dispersion": 3.0,
    }
    original = baseline.score_records(records, parameters)["date_level"]
    first = [r for r in original if r["origin"] == rows[4]["date"]]
    _, expected_next = baseline.conditional_means(counts[:4], parameters)
    for row in first:
        expected = 10 + 0.5 ** (row["horizon"] - 1) * (expected_next - 10)
        assert row["predicted_exposures"] == pytest.approx(expected)
        assert row["target"] == rows[4 + row["horizon"] - 1]["date"]
    altered = deepcopy(records)
    for row in altered["test"]:
        row["count"] += 1000
    changed = baseline.score_records(altered, parameters)["date_level"]
    assert [r["predicted_exposures"] for r in changed if r["origin"] == rows[4]["date"]] == [
        r["predicted_exposures"] for r in first
    ]
    before = next(r for r in original if r["origin"] == rows[5]["date"] and r["horizon"] == 1)
    after = next(r for r in changed if r["origin"] == rows[5]["date"] and r["horizon"] == 1)
    assert after["predicted_exposures"] - before["predicted_exposures"] == pytest.approx(200.0)
    assert records["test"][0]["count"] == 12


def test_aggregate_input_calendar_and_stationary_parameter_constraints():
    raw = {
        name: [{"date": f"2020-01-0{i}", "patterns": [[0, 2], [3, 1]]}]
        for i, name in enumerate(["train", "validation", "test"], start=1)
    }
    assert baseline.extract_records(raw)["test"][0]["count"] == 6
    raw["test"][0]["date"] = "2020-01-05"
    with pytest.raises(ValueError, match="consecutive"):
        baseline.extract_records(raw)
    for a, b in [(-12, -12), (12, 12), (12, -12), (-12, 12)]:
        parameters = baseline.decode([np.log(10), a, b, np.log(20)])
        assert parameters["omega"] > 0
        assert 0 <= parameters["alpha"] + parameters["beta"] < 1

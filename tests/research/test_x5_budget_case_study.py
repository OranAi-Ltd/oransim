from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "backend" / "scripts" / "run_x5_budget_case_study.py"
SPEC = importlib.util.spec_from_file_location("run_x5_budget_case_study", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_uplift_at_fraction_matches_hand_calculation() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    score = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    uplift, n_top, n_treated, n_control = MODULE.uplift_at_fraction(y, treatment, score, 0.5)
    assert n_top == 4
    assert n_treated == 2
    assert n_control == 2
    assert uplift == pytest.approx(1.0)


def test_cumulative_gain_and_incremental_scaling() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    score = np.arange(8, 0, -1)
    uplift, n_top, _, _ = MODULE.uplift_at_fraction(y, treatment, score, 0.5)
    assert uplift * n_top == pytest.approx(4.0)
    assert uplift * 10_000 == pytest.approx(10_000.0)


def test_bootstrap_is_deterministic_and_ordered() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1] * 20)
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1] * 20)
    score = np.tile(np.arange(8, 0, -1), 20) + np.repeat(np.linspace(0, 0.01, 20), 8)
    first = MODULE.stratified_binary_bootstrap_ci(y, treatment, score, 0.5, samples=200, seed=42)
    second = MODULE.stratified_binary_bootstrap_ci(y, treatment, score, 0.5, samples=200, seed=42)
    assert first == second
    assert first[0] <= first[1]


def test_sklift_metrics_are_finite_on_hand_example() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1] * 4)
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1] * 4)
    score = np.arange(len(y), 0, -1, dtype=float)
    metrics = MODULE.full_ranking_metrics(y, treatment, score)
    assert set(metrics) == {"auuc", "qini"}
    assert all(np.isfinite(v) for v in metrics.values())

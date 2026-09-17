from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_kuairand_population_dynamics", ROOT / "backend/scripts/run_kuairand_population_dynamics.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
from oransim.data.gaussian_response import GaussianLogitResponse, GaussianResponseConfig


def test_more_exposures_provide_more_information_and_zero_exposures_do_not_update() -> None:
    state = GaussianLogitResponse(np.array([0.5, 0.5, 0.5]), GaussianResponseConfig())
    before_mean, before_cov = state.mean.copy(), state.covariance.copy()
    state.observe(np.array([5, 500, 0]), np.array([10, 1000, 0]))
    assert state.covariance[1, 0, 0] < state.covariance[0, 0, 0] < before_cov[0, 0, 0]
    np.testing.assert_array_equal(state.mean[2], before_mean[2])
    np.testing.assert_array_equal(state.covariance[2], before_cov[2])


def test_extreme_counts_remain_finite_and_covariance_psd() -> None:
    for kwargs in (
        {"level_variance": np.nan},
        {"trend_variance": np.inf},
        {"trend_damping": -0.1},
        {"effective_exposure_cap": -1},
    ):
        with pytest.raises(ValueError):
            GaussianResponseConfig(**kwargs)
    for prior in ([], [[0.5]], [np.nan], [-0.1], [1.1]):
        with pytest.raises(ValueError):
            GaussianLogitResponse(np.asarray(prior), GaussianResponseConfig())
    state = GaussianLogitResponse(
        np.array([1e-8, 1 - 1e-8]), GaussianResponseConfig(0.01, 0.001, 0.8)
    )
    for _ in range(12):
        state.advance()
        state.observe(np.array([100000, 0]), np.array([100000, 100000]))
    assert np.isfinite(state.mean).all()
    assert np.linalg.eigvalsh(state.covariance).min() >= -1e-10
    assert (state.forecast(7) > 0).all() and (state.forecast(7) < 1).all()
    assert state.forecast()[0] > 0.99 and state.forecast()[1] < 0.01


def test_power_likelihood_retains_proportion_with_lower_confidence() -> None:
    uncapped = GaussianLogitResponse(np.array([0.5]), GaussianResponseConfig())
    capped = GaussianLogitResponse(
        np.array([0.5]), GaussianResponseConfig(effective_exposure_cap=100)
    )
    for state in (uncapped, capped):
        state.observe(np.array([8000]), np.array([10000]))
    assert capped.covariance[0, 0, 0] > uncapped.covariance[0, 0, 0]
    assert 0.5 < capped.forecast()[0] < uncapped.forecast()[0]


def test_forecast_is_frozen_and_does_not_mutate_state() -> None:
    state = GaussianLogitResponse(np.array([0.3]), GaussianResponseConfig(0.01, 0.001, 0.8))
    state.mean[0, 1] = 0.1
    before = state.mean.copy(), state.covariance.copy()
    assert state.forecast(7)[0] > state.forecast(1)[0]
    np.testing.assert_array_equal(state.mean, before[0])
    np.testing.assert_array_equal(state.covariance, before[1])


@pytest.mark.parametrize(
    "candidate",
    [
        {"family": "last_value", "strength": 20.0},
        {"family": "ewma", "alpha": 0.35, "strength": 20.0},
        {"family": "gaussian_logit", "level_variance": 0.01},
    ],
)
def test_predictions_cannot_consume_current_or_future_targets(candidate: dict) -> None:
    exposures = np.full((12, 2), 100.0)
    successes = np.tile(np.array([20.0, 70.0]), (12, 1))
    original = MODULE.response_predictions(successes, exposures, 4, candidate)
    changed = successes.copy()
    changed[7:] = 100 - changed[7:]
    perturbed = MODULE.response_predictions(changed, exposures, 4, candidate)
    for horizon in MODULE.HORIZONS:
        np.testing.assert_array_equal(original[horizon][4:8], perturbed[horizon][4:8])
    assert not np.allclose(original[1][8], perturbed[1][8])


def test_multiday_endpoints_stay_inside_phase_and_align_with_origins() -> None:
    values = np.arange(12, dtype=float)[:, None]
    prediction, successes, exposures = MODULE.phase_arrays(values, values, values, 4, 10, 3)
    np.testing.assert_array_equal(prediction[:, 0], [4, 5, 6, 7])
    np.testing.assert_array_equal(successes[:, 0], [6, 7, 8, 9])
    with pytest.raises(ValueError):
        MODULE.phase_arrays(values, values, values, 4, 6, 3)


def test_aggregation_retains_failures_and_assigns_unseen_users_cold_start() -> None:
    history = pd.DataFrame({"user_id": [10, 10, 20], "is_click": [0, 1, 0]})
    segments, _ = MODULE.fit_segments(history)
    future = pd.DataFrame(
        {"user_id": [10, 30, 30], "is_click": [0, 1, 0], "date": pd.to_datetime(["2022-04-22"] * 3)}
    )
    _, successes, exposures = MODULE.daily_counts(future, segments)
    assert exposures.sum() == 3 and successes.sum() == 1
    assert exposures[0, 8] == 2 and successes[0, 8] == 1
    assert 30 not in segments.index


def test_test_targets_do_not_change_validation_model_selection() -> None:
    dates = pd.date_range("2022-01-01", periods=15)
    exposures = np.full((15, 2), 100.0)
    successes = np.tile([20.0, 60.0], (15, 1))
    first = MODULE.evaluate_policy(dates, successes, exposures, 5, 8)
    successes[8:] = 100 - successes[8:]
    second = MODULE.evaluate_policy(dates, successes, exposures, 5, 8)
    for horizon in ("1", "3"):
        a, b = first["horizons"][horizon], second["horizons"][horizon]
        assert a["validation_selected_response"] == b["validation_selected_response"]
        assert (
            a["validation_selected_composition_alpha"] == b["validation_selected_composition_alpha"]
        )
        for family in a["response_methods"]:
            assert (
                a["response_methods"][family]["parameters"]
                == b["response_methods"][family]["parameters"]
            )
            assert (
                a["response_methods"][family]["validation"]
                == b["response_methods"][family]["validation"]
            )
    assert first["horizons"]["7"]["status"] == "Open"

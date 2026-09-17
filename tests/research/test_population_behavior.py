"""Independent checks for aggregate behavior scores and causal baselines."""

import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "backend/scripts/evaluate_population_behavior.py"
SPEC = importlib.util.spec_from_file_location("behavior_evaluation", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_aggregate_scores_equal_explicit_events():
    outcomes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[[2, 1, 0, 1], [1, 0, 2, 1]], [[1, 2, 1, 0], [0, 1, 1, 2]]], dtype=float)
    q = np.array(
        [[[0.4, 0.2, 0.1, 0.3], [0.1, 0.2, 0.4, 0.3]], [[0.2, 0.4, 0.3, 0.1], [0.1, 0.1, 0.2, 0.6]]]
    )
    arrivals = np.array([[5, 2], [4, 6]], dtype=float)
    result = m.score(y, q, arrivals, outcomes, np.array([0, 1]))
    losses = []
    squares = []
    for t in range(2):
        for c in range(2):
            p = q[t, c] @ outcomes
            for index, count in enumerate(y[t, c].astype(int)):
                losses.extend([-np.log(q[t, c, index])] * count)
                squares.extend([(p - outcomes[index]) ** 2] * count)
    np.testing.assert_allclose(result["metrics"]["joint_nll"], np.mean(losses))
    np.testing.assert_allclose(result["metrics"]["head_brier"], np.mean(squares, axis=0))
    expected = np.abs((arrivals[..., None] * (q @ outcomes)).sum(1) - (y @ outcomes).sum(1)).mean(0)
    np.testing.assert_allclose(result["metrics"]["head_action_mae"], expected)
    assert all(
        sum(row["events"] for row in bins) == y.sum()
        for bins in result["diagnostics"]["calibration"]
    )


def test_zero_arrival_dates_do_not_become_zero_response_rates():
    outcomes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[[0, 0, 1, 1]], [[0, 0, 0, 0]]], dtype=float)
    q = np.full_like(y, 0.25)
    result = m.score(y, q, np.array([[2.0], [2.0]]), outcomes, np.array([0]))
    assert result["diagnostics"]["valid_rate_dates"] == 1
    np.testing.assert_allclose(result["metrics"]["head_rate_mae_pp"], [50, 0])
    # Positive forecasts on the zero-count day must still incur count error.
    np.testing.assert_allclose(result["metrics"]["head_action_mae"], [1, 0.5])


def test_history_forecasts_use_only_preorigin_observations():
    data = {
        "train": np.array([[[2.0, 1, 0, 1]], [[1.0, 1, 1, 1]]]),
        "validation": np.array([[[1.0, 0, 2, 1]]]),
        "test": np.array([[[1.0, 2, 1, 0]], [[0.0, 1, 1, 2]], [[0.0, 0, 2, 2]]]),
    }
    for method in m.BASELINES:
        original = m.history_forecasts(
            data, "test", method, 10, 3 if method == "ewma_history" else None
        )
        changed = {k: v.copy() for k, v in data.items()}
        changed["test"][0] *= 10
        perturbed = m.history_forecasts(
            changed, "test", method, 10, 3 if method == "ewma_history" else None
        )
        for h in [1, 3]:
            np.testing.assert_array_equal(original[h][0][0], perturbed[h][0][0])
            np.testing.assert_array_equal(original[h][0][1], perturbed[h][0][1])
        if method != "training_history":
            assert not np.array_equal(original[1][1][1], perturbed[1][1][1])
        for q, n in original[1]:
            np.testing.assert_allclose(q.sum(-1), 1)
            assert (q > 0).all() and (n >= 0).all()

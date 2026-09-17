from dataclasses import replace

import numpy as np
import pytest
from oransim.world_model.population_learning import count_nll, fit_world, marginal_count_nll, replay
from oransim.world_model.population_loop import (
    LoopConfig,
    PopulationWorldLoop,
    WorldAction,
    WorldObservation,
)


def example():
    world = PopulationWorldLoop(
        [0.6, 0.4],
        [[1.0], [1.0]],
        [[-1.0], [0.5]],
        [[0.2], [0.2]],
        LoopConfig(base_arrivals=300, population_size=500, novelty_decay=0),
    )
    actions = [
        WorldAction(
            np.array([0.5 * np.sin(i), -0.3 * np.cos(i)]), np.array([0.0, i % 3 * 0.5]), 1 + i % 2
        )
        for i in range(10)
    ]
    truth = world.branch(np.random.default_rng(81))
    rng = np.random.default_rng(82)
    obs = []
    for action in actions:
        _, o, _ = truth.simulate_step(action, rng, False)
        obs.append(WorldObservation(o.day, o.exposures, o.responses))
    return world, actions, obs


def test_snapshot_preserves_update_and_forecast(tmp_path):
    world, actions, obs = example()
    world.observe(actions[0], obs[0])
    world.save(tmp_path / "world.json")
    restored = PopulationWorldLoop.load(tmp_path / "world.json")
    world.observe(actions[1], obs[1])
    restored.observe(actions[1], obs[1])
    assert world.to_dict() == restored.to_dict()
    before = world.to_dict()
    assert world.forecast_distribution(actions[2:5], 32, 5) == restored.forecast_distribution(
        actions[2:5], 32, 5
    )
    assert world.to_dict() == before
    branch = world.branch(np.random.default_rng(11))
    branch.save(tmp_path / "branch.json")
    loaded = PopulationWorldLoop.load(tmp_path / "branch.json")
    assert branch.rollout(actions[:2], 42, False)[1] == loaded.rollout(actions[:2], 42, False)[1]
    with pytest.raises(ValueError):
        loaded.observe(actions[2], obs[2])


def test_collapsed_sampling_obeys_same_conditional_moments():
    world, actions, _ = example()
    world.config = replace(world.config, base_arrivals=10000)
    branch = world.branch(np.random.default_rng(10))
    rng = np.random.default_rng(11)
    prediction = branch.predict(actions[0])
    counts = [branch._sample(actions[0], rng, False)[1].responses.sum() for _ in range(1200)]
    assert abs(np.mean(counts) - prediction.expected_responses) < 4 * np.sqrt(
        prediction.response_count_variance / 1200
    )
    assert np.var(counts) / prediction.response_count_variance == pytest.approx(1, rel=0.15)


def test_training_is_chronological_nonmutating_and_improves_own_objectives():
    world, actions, obs = example()
    before = world.to_dict()
    fitted, report = fit_world(world, actions[:8], obs[:8], max_iterations=5)
    assert world.to_dict() == before and fitted.state.day == 8
    assert all(
        b["objective_after"] <= b["objective_before"] + 1e-10
        for b in report["blocks"]
        if b["accepted"]
    )
    assert report["identification"]["numerical_rank"] <= len(
        report["identification"]["parameter_names"]
    )
    future, _, _, _ = replay(fitted, actions[8:], obs[8:])
    assert future.state.day == 10
    with pytest.raises(ValueError):
        replay(fitted, actions[:2], obs[:2])
    with pytest.raises(ValueError):
        fit_world(world, actions[:3], obs[:2])
    with pytest.raises(ValueError):
        fit_world(
            world,
            actions[:3],
            [WorldObservation(i, np.zeros(2, int), np.zeros(2, int)) for i in range(3)],
        )
    assert count_nll(0, 0, 10) == 0


def test_arrival_scoring_integrates_the_same_uncertainty_as_forecasting():
    from scipy.integrate import quad

    world, actions, _ = example()
    action = actions[0]
    world._state.arrival_posterior_variance = 0.3
    for n in (150, 400, 1000):
        integral = quad(
            lambda x, n=n: np.exp(
                -count_nll(
                    n, float(world._arrival_rates(action, x)), world.config.arrival_dispersion
                )
                - x * x / (2 * 0.3)
            )
            / np.sqrt(2 * np.pi * 0.3),
            -8,
            8,
            epsabs=1e-12,
        )[0]
        assert marginal_count_nll(n, world, action) == pytest.approx(-np.log(integral), abs=0.005)
    prediction = world.predict(action)
    samples = world.forecast_distribution([action], 3000, 998)
    assert samples["mean"][0][0] == pytest.approx(prediction.expected_exposures, rel=0.05)


def test_selection_updates_delivery_without_changing_population():
    world, actions, obs = example()
    world.config = replace(world.config, selection_process_variance=0.05)
    # Enable a learned delivery state with a proper log-ratio prior.
    world._state.selection_covariance[:] = 0.1
    before = world.state.population_weights.copy()
    world.observe(actions[0], WorldObservation(0, np.array([10, 990]), np.array([2, 500])))
    assert world.predict(actions[0]).exposure_weights[1] > 0.85
    np.testing.assert_array_equal(world.state.population_weights, before)
    assert np.linalg.eigvalsh(world.state.selection_covariance).min() >= 0
    restored = PopulationWorldLoop.from_dict(world.to_dict())
    assert restored.to_dict() == world.to_dict()


def test_saved_world_cli_observe_fit_and_forecast(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    world, actions, obs = example()
    world.save(tmp_path / "initial.json")
    records = [
        {
            "action": {
                "content_logits": a.content_logits.tolist(),
                "targeting_logits": a.targeting_logits.tolist(),
                "exposure_effort": a.exposure_effort,
            },
            "observation": {
                "day": o.day,
                "exposures": o.exposures.tolist(),
                "responses": o.responses.tolist(),
            },
        }
        for a, o in zip(actions[:4], obs[:4], strict=False)
    ]
    (tmp_path / "records.json").write_text(json.dumps(records))
    cli = Path(__file__).resolve().parents[2] / "backend/scripts/population_world.py"
    for operation in ("observe", "fit"):
        subprocess.run(
            [
                sys.executable,
                str(cli),
                operation,
                "--model",
                str(tmp_path / "initial.json"),
                "--records",
                str(tmp_path / "records.json"),
                "--out",
                str(tmp_path / f"{operation}.json"),
                "--max-iterations",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert PopulationWorldLoop.load(tmp_path / f"{operation}.json").state.day == 4
    subprocess.run(
        [
            sys.executable,
            str(cli),
            "forecast",
            "--model",
            str(tmp_path / "fit.json"),
            "--records",
            str(tmp_path / "records.json"),
            "--out",
            str(tmp_path / "forecast.json"),
            "--trajectories",
            "16",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    forecast = json.loads((tmp_path / "forecast.json").read_text())
    assert forecast["start_day"] == 4 and len(forecast["mean"]) == 4

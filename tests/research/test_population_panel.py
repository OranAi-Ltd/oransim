from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from oransim.world_model.population_behavior import BehaviorBatch, JointBehaviorModel
from oransim.world_model.population_loop import LoopConfig, PopulationWorldLoop, WorldAction
from oransim.world_model.population_panel import (
    PanelEvent,
    PopulationWorldPanel,
    SharedArrivalFilter,
    UserMemory,
)


def panel():
    cfg = LoopConfig(
        base_arrivals=30,
        population_size=20,
        novelty_decay=0,
        selection_process_variance=0.02,
        social_gain=0.1,
        fatigue_gain=0.05,
        recommendation_gain=0,
        momentum_arrival_gain=0,
        mixture_gain=0,
    )
    worlds = {
        c: PopulationWorldLoop(
            [0.6, 0.4],
            [[0.5, 0.5], [0.5, 0.5]],
            [[-1.0, 0.0], [-0.5, 0.5]],
            [[0.1, 0.1], [0.1, 0.1]],
            cfg,
        )
        for c in ["a", "b"]
    }
    model = PopulationWorldPanel(worlds, ["click", "like"], [0.4, 0.1])
    actions = {c: WorldAction(np.zeros(2), np.zeros(2)) for c in worlds}
    events = [
        PanelEvent(
            float(i + 1), f"u{i%4}", "a" if i % 3 else "b", i % 4 % 2, (i % 2, int(i % 5 == 0))
        )
        for i in range(20)
    ]
    return model, actions, events


def test_joint_distribution_normalizes_and_marginals_match_sampling():
    model = JointBehaviorModel(["click", "like"], ["x"], [0.4, 0.1])
    model.dependencies[1, 0] = 2.0
    model.loadings[1] = 0.7
    outcomes = np.array(list(product([0, 1], repeat=2)))
    batch = BehaviorBatch(np.zeros((4, 1)), np.full((4, 1), 0.2), np.ones((4, 1)), outcomes)
    mass = np.exp(model.log_probability(batch))
    assert mass.sum() == pytest.approx(1)
    expected = mass @ outcomes
    np.testing.assert_allclose(model.marginals(batch)[0], expected)
    rng = np.random.default_rng(55)
    observed = np.array([model.sample([0], 0.2, rng) for _ in range(12000)])
    np.testing.assert_allclose(observed.mean(0), expected, atol=0.015)


def test_joint_gaussian_mixture_readout_matches_latent_population_sampling():
    model, _, _ = panel()
    behavior = model.behavior
    behavior.dependencies[1, 0] = 1.8
    behavior.loadings[1] = 0.6
    nodes, weights = model._latent("a", 0)
    features = np.zeros(len(behavior.feature_names))
    batch = BehaviorBatch(features[None, :], nodes[None, :], weights[None, :], np.zeros((1, 2)))
    expected = behavior.marginals(batch)[0]
    rng = np.random.default_rng(178)
    state = model.worlds["a"].state
    observed = []
    for _ in range(16000):
        k = rng.choice(2, p=state.mixture_weights[0])
        latent = rng.normal(
            state.means[0, k],
            np.sqrt(state.heterogeneity_variance[0, k] + state.mean_posterior_variance[0]),
        )
        observed.append(behavior.sample(features, latent, rng))
    np.testing.assert_allclose(np.mean(observed, axis=0), expected, atol=0.015)


def test_joint_fit_improves_unseen_dependent_actions():
    rng = np.random.default_rng(77)
    teacher = JointBehaviorModel(["click", "like"], ["x"], [0.5, 0.1])
    teacher.dependencies[1, 0] = 3.0
    teacher.coefficients[1, 1] = 0.8
    x = rng.normal(size=(1600, 1))
    u = rng.normal(size=(1600, 1))
    y = np.array([teacher.sample(a, z[0], rng) for a, z in zip(x, u, strict=False)])
    batch = BehaviorBatch(x, u, np.ones_like(u), y)
    learner = JointBehaviorModel(teacher.heads, teacher.feature_names, [0.5, 0.1])
    before = -learner.log_probability(batch.take(np.arange(1000, 1600))).mean()
    report = learner.fit(batch.take(np.arange(1000)), max_iterations=50)
    after = -learner.log_probability(batch.take(np.arange(1000, 1600))).mean()
    assert report["accepted"] and after < before - 0.05
    assert learner.dependencies[1, 0] > 1.0


def test_user_memory_decay_cross_campaign_and_delayed_outcomes():
    memory = UserMemory(2, 100.0)
    memory.observe("u", "a", 0, [0, 0])
    memory.observe("u", "b", 10, [0, 0])
    assert memory.features("u", "b", 10)[0] > memory.features("u", "b", 10)[1]
    assert memory.features("u", "a", 10)[3] == 0
    memory.complete("u", "a", 0, [1, 0])
    assert memory.features("u", "a", 10)[3] > 0
    assert memory.features("u", "a", 110)[0] < memory.features("u", "a", 10)[0]
    with pytest.raises(ValueError):
        memory.features("u", "a", 9)


def test_common_environment_updates_unobserved_campaign_and_has_gauge():
    shared = SharedArrivalFilter(2, 0.1, 0.001)
    independent = SharedArrivalFilter(2, 0, 0.001)
    for model in (shared, independent):
        model.observe([100, 100], [100, 100], np.array([1000.0, 1000.0]))
        model.observe([400, 0], [100, 0], np.array([1000.0, 1000.0]))
    assert shared.mean[1] > independent.mean[1] + 0.5
    assert (shared.mean - shared.common).sum() == pytest.approx(0)
    assert np.linalg.eigvalsh(shared.covariance).min() > 0


def test_panel_observation_timing_atomicity_and_persistence(tmp_path):
    model, actions, events = panel()
    other = deepcopy(model)
    altered = [
        PanelEvent(e.timestamp, e.user, e.campaign, e.group, tuple(1 - v for v in e.outcomes))
        for e in events
    ]
    _, batch = model.observe_day(events, actions, end_timestamp=100.0)
    _, changed = other.observe_day(altered, actions, end_timestamp=100.0)
    # Same-day outcomes cannot leak into features of subsequent same-day events.
    np.testing.assert_array_equal(batch.features, changed.features)
    assert model.memory.features("u0", "a", 101)[3] != other.memory.features("u0", "a", 101)[3]
    before = deepcopy(model.to_dict())
    with pytest.raises(ValueError):
        model.observe_day(events, actions)
    assert model.to_dict() == before
    model.save(tmp_path / "private.json")
    restored = PopulationWorldPanel.load(tmp_path / "private.json")
    assert (tmp_path / "private.json").stat().st_mode & 0o777 == 0o600
    assert restored.to_dict() == model.to_dict()
    assert restored.forecast([actions] * 2, 8, 33) == model.forecast([actions] * 2, 8, 33)
    assert model.to_dict() == before
    with pytest.raises(ValueError):
        model.simulate_day(actions, np.random.default_rng(1))
    simulated = model.branch(1)
    simulated.save(tmp_path / "branch.json")
    loaded = PopulationWorldPanel.load(tmp_path / "branch.json")
    with pytest.raises(ValueError):
        loaded.observe_day([], actions)
    assert simulated.simulate_day(actions, np.random.default_rng(8)) == loaded.simulate_day(
        actions, np.random.default_rng(8)
    )


def test_shared_process_noise_is_trainable_on_synchronized_counts():
    rng = np.random.default_rng(49)
    model = SharedArrivalFilter(3, 0.001, 0.001)
    common = np.cumsum(rng.normal(0, 0.3, 16))
    base = np.full((16, 3), 10000.0)
    counts = rng.poisson(base * np.exp(common[:, None]))
    result = model.fit(counts, base, np.full(3, 10000.0))
    assert result["accepted"] and result["working_nll_after"] < result["working_nll_before"]
    assert model.shared_variance > model.individual_variance


def test_content_and_persistent_memory_change_joint_simulated_actions():
    model, actions, events = panel()
    model.observe_day(events, actions, end_timestamp=100.0)
    memory_column = model.behavior.feature_names.index("log_exposures") + 1
    model.behavior.coefficients[0, memory_column] = -0.8
    without = deepcopy(model)
    without.memory_enabled = False
    with_memory = model.forecast([actions] * 2, 16, 909)
    no_memory = without.forecast([actions] * 2, 16, 909)
    assert np.array(with_memory["mean"])[:, :, 1].sum() < np.array(no_memory["mean"])[:, :, 1].sum()
    elevated = {c: WorldAction(np.full(2, 2.0), a.targeting_logits) for c, a in actions.items()}
    high = model.forecast([elevated] * 2, 16, 909)
    assert np.array(high["mean"])[:, :, 1].sum() > np.array(with_memory["mean"])[:, :, 1].sum()


def test_panel_cli_observe_and_forecast(tmp_path):
    import json
    import subprocess
    import sys
    from dataclasses import asdict
    from pathlib import Path

    model, actions, events = panel()
    model.save(tmp_path / "initial.json")
    encoded = {
        c: {
            "content_logits": a.content_logits.tolist(),
            "targeting_logits": a.targeting_logits.tolist(),
        }
        for c, a in actions.items()
    }
    (tmp_path / "observations.json").write_text(
        json.dumps(
            [
                {
                    "day": 0,
                    "actions": encoded,
                    "events": [asdict(e) for e in events],
                    "end_timestamp": 100.0,
                }
            ]
        )
    )
    (tmp_path / "plan.json").write_text(json.dumps([encoded]))
    cli = Path(__file__).resolve().parents[2] / "backend/scripts/population_panel.py"
    subprocess.run(
        [
            sys.executable,
            str(cli),
            "observe",
            "--model",
            str(tmp_path / "initial.json"),
            "--records",
            str(tmp_path / "observations.json"),
            "--out",
            str(tmp_path / "updated.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert PopulationWorldPanel.load(tmp_path / "updated.json").day == 1
    subprocess.run(
        [
            sys.executable,
            str(cli),
            "forecast",
            "--model",
            str(tmp_path / "updated.json"),
            "--records",
            str(tmp_path / "plan.json"),
            "--out",
            str(tmp_path / "forecast.json"),
            "--trajectories",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    forecast = __import__("json").loads((tmp_path / "forecast.json").read_text())
    assert forecast["columns"] == ["exposures", "click", "like"] and forecast["start_day"] == 1


def test_log_calendar_uses_absolute_time_when_source_date_disagrees(tmp_path):
    import importlib.util
    from pathlib import Path

    import pandas as pd

    path = Path(__file__).resolve().parents[2] / "backend/scripts/train_population_panel.py"
    spec = importlib.util.spec_from_file_location("panel_research_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    timestamp = pd.Timestamp("2022-04-12 15:30:00", tz="UTC").value // 1_000_000
    row = {
        "user_id": 1,
        "video_id": 1,
        "date": 20220413,
        "time_ms": timestamp,
        "tab": 1,
        **dict.fromkeys(runner.HEADS, 0),
    }
    for name in [
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ]:
        pd.DataFrame([row]).to_csv(tmp_path / name, index=False)
    pd.DataFrame([{"video_id": 1, "author_id": 1}]).to_csv(
        tmp_path / "video_features_basic_pure.csv", index=False
    )
    for frame in runner.read_data(tmp_path, 1):
        assert frame.date.iloc[0] == pd.Timestamp("2022-04-12")

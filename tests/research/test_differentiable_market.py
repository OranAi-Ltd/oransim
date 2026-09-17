import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from oransim.world_model.differentiable_market import (
    DTYPE,
    DifferentiableMarket,
    MarketBatch,
    MarketConfig,
    MarketDay,
    MarketRuntime,
    fit_market,
)

torch.set_num_threads(1)


def fixture(dimensions=1, family="gaussian"):
    model = DifferentiableMarket(
        MarketConfig(
            2,
            2,
            2,
            ("click", "like"),
            components=1 if family == "point" else 2,
            dimensions=dimensions,
            quadrature=7,
            family=family,
        )
    )
    with torch.no_grad():
        model.dependencies[1, 0] = 1.3
        model.behavior_features.copy_(torch.tensor([[0.3, -0.2], [-0.2, 0.1]]))
    batch = MarketBatch.create(
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [[0.2, 0.1], [0.4, -0.2], [1.0, 0.3], [-0.1, 0.4]],
        [200, 100, 200, 100],
    )
    return model, batch


def test_joint_mass_contributions_and_zero_effort():
    model, batch = fixture()
    out = model(batch)
    torch.testing.assert_close(out["joint_log_prob"].exp().sum(-1), torch.ones(4, dtype=DTYPE))
    torch.testing.assert_close(out["component_actions"].sum(1), out["actions"])
    torch.testing.assert_close(out["component_exposures"].sum(1), out["exposures"])
    assert torch.all(out["actions"] <= out["exposures"][:, None])
    assert torch.all(out["reach_poisson"] <= batch.population)
    batch.effort.zero_()
    zero = model(batch)
    assert not zero["exposures"].any() and not zero["actions"].any()
    assert torch.isfinite(zero["joint_log_prob"]).all()
    with pytest.raises(ValueError):
        model.loss(batch, torch.ones(4, 4, dtype=DTYPE))


def test_mass_scaling_permutation_and_valid_covariance():
    model, batch = fixture(2)
    with torch.no_grad():
        model.raw_cholesky[:, :, 1, 0] = 0.2
    out = model(batch)
    order = torch.tensor([2, 0, 3, 1])
    reordered = MarketBatch(*(getattr(batch, k)[order] for k in batch.__dataclass_fields__))
    torch.testing.assert_close(model(reordered)["actions"], out["actions"][order])
    twice = deepcopy(batch)
    twice.population *= 2
    torch.testing.assert_close(model(twice)["actions"], 2 * out["actions"])
    chol = model.cholesky()
    assert torch.linalg.eigvalsh(chol @ chol.transpose(-1, -2)).min() > 0


def test_market_loss_reaches_every_population_and_observation_block():
    model, batch = fixture(2)
    patterns = torch.tensor([[7, 3, 6, 2], [8, 4, 2, 1], [9, 2, 8, 5], [8, 1, 3, 2]], dtype=DTYPE)
    parts, _ = model.loss(batch, patterns)
    loss = parts["joint_nll"] + 0.1 * parts["count_nll"]
    loss.backward()
    for name in [
        "mixture_logits",
        "means",
        "raw_cholesky",
        "intensity_loading",
        "intensity_features",
        "behavior_features",
        "dependencies",
    ]:
        grad = getattr(model, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 1e-9, name
    # Central differences check the actual integrated objective, not a surrogate.
    for name, index in [
        ("means", (0, 0, 0)),
        ("raw_cholesky", (0, 0, 0, 0)),
        ("mixture_logits", (0, 0)),
    ]:
        parameter = getattr(model, name)
        analytical = float(parameter.grad[index])
        original = float(parameter[index].detach())
        values = []
        for offset in [1e-5, -1e-5]:
            with torch.no_grad():
                parameter[index] = original + offset
            p, _ = model.loss(batch, patterns)
            values.append(float((p["joint_nll"] + 0.1 * p["count_nll"]).detach()))
        with torch.no_grad():
            parameter[index] = original
        assert analytical == pytest.approx((values[0] - values[1]) / 2e-5, rel=2e-4, abs=1e-7)


def test_dynamic_gradient_and_label_free_rollout():
    model, batch = fixture()
    observed = torch.tensor([[7, 3, 6, 2], [8, 4, 2, 1], [9, 2, 8, 5], [8, 1, 3, 2]], dtype=DTYPE)
    state = model.initial_state()
    out = model(batch, state)
    state = model.transition(state, batch, out, observed)
    outputs, end = model.rollout([batch, batch], state)
    outputs[-1]["actions"].sum().backward()
    for name in ["raw_retention", "raw_correction", "feedback"]:
        grad = getattr(model, name).grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert end.day == 3 and state.day == 1
    assert abs(float(end.supply.sum().detach())) < 1e-10


def test_choice_mass_price_response_and_availability():
    model, _ = fixture(2)
    attributes = [[1.0, 0.2], [0.4, 1.0]]
    mass = [200.0, 100.0]
    result = model.choice(attributes, mass, prices=[1.0, 2.0])
    expensive = model.choice(attributes, mass, prices=[2.0, 2.0])
    assert float(result["demand"].sum().detach()) == pytest.approx(300.0)
    torch.testing.assert_close(result["component_demand"].sum((0, 1)), result["demand"])
    assert expensive["demand"][1] < result["demand"][1]
    unavailable = model.choice(attributes, mass, available=[False, False])
    assert (
        float(unavailable["demand"][0].detach()) == pytest.approx(300.0)
        and not unavailable["demand"][1:].any()
    )
    result["demand"][1].backward()
    assert model.raw_cholesky.grad.abs().sum() > 0


def test_runtime_save_replay_forecast_isolation_and_atomic_failure(tmp_path):
    model, batch = fixture()
    runtime = MarketRuntime(model)
    patterns = torch.ones(4, 4, dtype=DTYPE) * 5
    day = MarketDay("2022-01-01", batch, patterns)
    runtime.observe(day)
    before = runtime.state.to_dict()
    path = tmp_path / "model.json"
    runtime.save(path)
    restored = MarketRuntime.load(path)
    assert path.stat().st_mode & 0o777 == 0o600
    a = runtime.forecast([batch, batch], trajectories=128, seed=17)
    b = restored.forecast([batch, batch], trajectories=128, seed=17)
    assert a == b and runtime.state.to_dict() == before
    assert np.max(np.abs(np.array(a["mean"])[0] - np.array(a["deterministic_mean_path"])[0])) < 10
    with pytest.raises(ValueError):
        runtime.observe(day)
    assert runtime.state.to_dict() == before
    branch = runtime.branch()
    branch.save(tmp_path / "branch.json")
    with pytest.raises(ValueError):
        MarketRuntime.load(tmp_path / "branch.json").observe(day)


def test_aggregate_supervision_can_fit_unseen_contexts():
    teacher, batch = fixture()
    days = []
    with torch.no_grad():
        teacher.behavior_bias[0] = torch.tensor([1.0, -0.8])
        teacher.behavior_bias[1] = torch.tensor([-0.6, 0.4])
        teacher.intensity_bias.fill_(0.4)
        for t in range(8):
            b = deepcopy(batch)
            b.features = b.features + (0.1 * t)
            out = teacher(b)
            patterns = out["exposures"][:, None] * out["joint_log_prob"].exp()
            days.append(MarketDay(f"2022-01-{t+1:02d}", b, patterns))
    student, _ = fixture()
    before, _ = student.loss(days[-1].batch, days[-1].patterns)
    state, report = fit_market(student, days[:5], days[5:7], epochs=35, patience=10)
    after, _ = student.loss(days[-1].batch, days[-1].patterns)
    assert after["joint_nll"] < before["joint_nll"] - 0.02
    assert report["selected_epoch"] > 0


def test_future_forecast_ignores_unavailable_history_features():
    from dataclasses import replace

    original, batch = fixture()
    model = DifferentiableMarket(replace(original.config, history_feature_indices=(1,)))
    with torch.no_grad():
        model.behavior_features[:, 1] = 1.0
    future = deepcopy(batch)
    future.features[:, 1] = 1000.0
    runtime = MarketRuntime(model)
    a = runtime.forecast([batch, batch], trajectories=8, seed=1)
    b = runtime.forecast([batch, future], trajectories=8, seed=1)
    assert a == b


def test_choice_aggregate_training_and_cli_roundtrip(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from oransim.world_model.market_choice_learning import choice_loss, fit_choices

    teacher, _ = fixture()
    rng = np.random.default_rng(11)
    records = []
    with torch.no_grad():
        teacher.choice_features.copy_(torch.tensor([1.1, -0.8], dtype=DTYPE))
        for i in range(7):
            record = {
                "attributes": rng.normal(size=(3, 2)).tolist(),
                "population": [200.0, 100.0],
                "prices": [0.4, 0.8, 1.2],
            }
            record["counts"] = teacher.choice(**record)["demand"].tolist()
            records.append(record)
    student, batch = fixture()
    before = float(choice_loss(student, records[-1]).detach())
    fit_choices(student, records[:5], records[5:6], epochs=40)
    assert float(choice_loss(student, records[-1]).detach()) < before - 0.01
    runtime = MarketRuntime(student)
    snapshot = tmp_path / "model.json"
    runtime.save(snapshot)
    plans = tmp_path / "plan.json"
    plans.write_text(json.dumps([batch.to_dict()]))
    script = Path(__file__).resolve().parents[2] / "backend/scripts/differentiable_market.py"
    output = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "forecast",
            "--model",
            str(snapshot),
            "--records",
            str(plans),
            "--out",
            str(output),
            "--trajectories",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["trajectories"] == 8
    branch_path = tmp_path / "branch.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "simulate",
            "--model",
            str(snapshot),
            "--records",
            str(plans),
            "--out",
            str(output),
            "--save-model",
            str(branch_path),
        ],
        check=True,
        capture_output=True,
    )
    assert MarketRuntime.load(branch_path).simulation
    assert MarketRuntime.load(snapshot).state.day == 0
    assert MarketRuntime.load(branch_path).state.day == 1


def test_splitting_identical_cohort_preserves_feedback():
    model, batch = fixture()
    state = model.initial_state()
    out = model(batch)
    doubled = MarketBatch(
        batch.group.repeat_interleave(2),
        batch.campaign.repeat_interleave(2),
        batch.features.repeat_interleave(2, dim=0),
        batch.population.repeat_interleave(2) / 2,
        batch.effort.repeat_interleave(2),
    )
    split = model(doubled)
    a = model.transition(state, batch, out)
    b = model.transition(state, doubled, split)
    for k in ["shift", "supply", "common", "fatigue", "memory"]:
        torch.testing.assert_close(getattr(a, k), getattr(b, k))


def test_daily_chronology_and_zero_opportunity_contracts():
    model, batch = fixture()
    counts = torch.ones(4, 4, dtype=DTYPE)
    a = MarketDay("2022-01-01", batch, counts)
    b = MarketDay("2022-01-03", batch, counts)
    with pytest.raises(ValueError):
        fit_market(model, [a, b], epochs=1)
    result = model.choice([[1.0, 2.0]], [0.0, 0.0])
    assert not result["demand"].any() and result["purchase_probability"] == 0
    runtime = MarketRuntime(model)
    runtime.observe(a)
    with pytest.raises(ValueError):
        runtime.observe(b)


def test_event_likelihood_equals_cohort_likelihood_for_identical_contexts():
    model, batch = fixture()
    state = model.initial_state()
    out = model(batch, state)
    outcomes = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=DTYPE)
    observed = model.event_log_probability(batch, outcomes, state)
    torch.testing.assert_close(observed, out["joint_log_prob"][torch.arange(4), torch.arange(4)])
    groups = batch.group.repeat_interleave(4)
    campaigns = batch.campaign.repeat_interleave(4)
    features = batch.features.repeat_interleave(4, dim=0)
    events = MarketBatch.create(groups, campaigns, features, torch.ones(16, dtype=DTYPE))
    day = MarketDay(
        "2022-01-01", batch, torch.ones(4, 4, dtype=DTYPE), events, outcomes.repeat(4, 1)
    )
    parts, _ = model.day_loss(day, state)
    torch.testing.assert_close(parts["joint_nll"], parts["cohort_joint_nll"])
    parts["joint_nll"].backward()
    assert model.raw_cholesky.grad.abs().sum() > 0
    assert model.mixture_logits.grad.abs().sum() > 0

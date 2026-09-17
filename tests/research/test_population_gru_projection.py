import torch
from oransim.world_model.differentiable_market import DTYPE, MarketBatch, MarketConfig
from oransim.world_model.population_comparisons import GRUMarket
from oransim.world_model.population_gru_projection import (
    ProjectedGRUMarket,
    load_projected_gru,
    save_projected_gru,
)


def fixture():
    config = MarketConfig(
        2,
        1,
        1,
        ("click", "like"),
        family="point",
        components=1,
        dimensions=4,
        learn_population=False,
    )
    model = ProjectedGRUMarket(config)
    batch = MarketBatch.create([0, 1], [0, 0], [[0.1], [-0.3]], [100.0, 200.0])
    return model, batch


def test_all_hidden_coordinates_can_train_the_click_readout():
    model, batch = fixture()
    legacy = GRUMarket(model.config)
    state = model.initial_state()
    state.shift = torch.tensor([[0.2, -0.3, 0.4, 0.1], [-0.1, 0.5, -0.2, 0.3]], dtype=DTYPE)
    torch.testing.assert_close(
        model(batch, state)["joint_log_prob"], legacy(batch, state)["joint_log_prob"]
    )
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    loss = -model.event_log_probability(batch, labels, state).sum()
    loss.backward()
    assert model.behavior_projection.requires_grad and not model.behavior_loading.requires_grad
    assert torch.all(model.behavior_projection.grad[0].abs() > 1e-6)
    before = model(batch, state)["marginals"][:, 0].detach()
    with torch.no_grad():
        model.behavior_projection[0, 2] += 2.0
    after = model(batch, state)["marginals"][:, 0].detach()
    assert torch.all((after - before).abs() > 0.01)
    torch.testing.assert_close(
        model.event_log_probability(batch, labels, state),
        model(batch, state)["joint_log_prob"][torch.arange(2), torch.tensor([2, 1])],
    )


def test_projected_gru_snapshot_preserves_learned_projection_and_future(tmp_path):
    model, batch = fixture()
    with torch.no_grad():
        model.behavior_projection[0] = torch.tensor([1.2, -0.6, 0.9, 0.3], dtype=DTYPE)
    _, state = model.rollout([batch, batch, batch])
    path = tmp_path / "model.json"
    save_projected_gru(model, path, state, {"training_task": "aggregate-behavior"})
    restored, restored_state, metadata = load_projected_gru(path)
    actual, _ = model.rollout([batch, batch], state)
    recovered, _ = restored.rollout([batch, batch], restored_state)
    for expected, found in zip(actual, recovered, strict=False):
        torch.testing.assert_close(expected["joint_log_prob"], found["joint_log_prob"])
        torch.testing.assert_close(expected["exposures"], found["exposures"])
    assert restored.behavior_projection.requires_grad
    assert path.stat().st_mode & 0o777 == 0o600
    assert restored_state.day == 3 and metadata["training_task"] == "aggregate-behavior"

from dataclasses import replace

import torch
from oransim.world_model.differentiable_market import (
    DTYPE,
    DifferentiableMarket,
    MarketBatch,
    MarketConfig,
)
from oransim.world_model.population_comparisons import (
    GRUMarket,
    SeparatedAggregationMarket,
    load_comparison,
    save_comparison,
)


def setup():
    cfg = MarketConfig(2, 1, 1, ("click", "like"), quadrature=9)
    full = DifferentiableMarket(cfg)
    separate = SeparatedAggregationMarket(cfg)
    with torch.no_grad():
        full.intensity_loading.fill_(1.0)
    separate.load_state_dict(full.state_dict())
    batch = MarketBatch.create([0, 1], [0, 0], [[0.0], [1.0]], [100.0, 200.0])
    return full, separate, batch


def test_separation_keeps_counts_and_changes_selected_response():
    full, separate, batch = setup()
    a = full(batch)
    b = separate(batch)
    torch.testing.assert_close(a["exposures"], b["exposures"])
    assert torch.all(a["marginals"][:, 0] > b["marginals"][:, 0])
    torch.testing.assert_close(b["component_actions"].sum(1), b["actions"])
    with torch.no_grad():
        full.intensity_loading.zero_()
        separate.intensity_loading.zero_()
    torch.testing.assert_close(full(batch)["joint_log_prob"], separate(batch)["joint_log_prob"])


def test_separate_event_probability_matches_enumerated_joint():
    _, model, batch = setup()
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    lp = model.event_log_probability(batch, y)
    torch.testing.assert_close(
        lp, model(batch)["joint_log_prob"][torch.arange(2), torch.tensor([2, 1])]
    )


def test_gru_future_gradient_and_comparison_snapshot(tmp_path):
    full, _, batch = setup()
    model = GRUMarket(
        replace(full.config, family="point", components=1, dimensions=4, learn_population=False)
    )
    state = model.initial_state()
    outputs, end = model.rollout([batch, batch, batch], state)
    outputs[-1]["actions"].sum().backward()
    assert model.gru.weight_hh.grad.abs().sum() > 0
    path = tmp_path / "gru.json"
    save_comparison(model, path, end)
    restored, rs, _ = load_comparison(path)
    torch.testing.assert_close(restored(batch, rs)["actions"], model(batch, end)["actions"])
    assert path.stat().st_mode & 0o777 == 0o600 and rs.day == 3

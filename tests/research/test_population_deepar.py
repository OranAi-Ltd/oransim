"""Forecast information and likelihood checks for the external baseline."""

import importlib.util
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[2] / "backend/scripts/evaluate_population_deepar.py"
SPEC = importlib.util.spec_from_file_location("population_deepar", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture():
    torch.manual_seed(3)
    model = m.DeepAR(4, 2, 2, 8, True)
    x = torch.randn(6, 2, 4, dtype=m.DTYPE)
    patterns = torch.randint(0, 5, (6, 2, 4)).to(m.DTYPE)
    model.setup(x[:3], patterns[:3])
    return model, x, patterns


def test_likelihood_and_normalized_marks():
    model, x, patterns = fixture()
    counts = patterns.sum(-1)
    dist, logq, _ = model(x, torch.cat((torch.zeros_like(counts[:1]), counts[:-1])))
    torch.testing.assert_close(logq.exp().sum(-1), torch.ones_like(counts))
    torch.testing.assert_close(dist.variance, dist.mean + dist.mean.square() / dist.total_count)
    # Independent NB probability calculation catches mean/dispersion mistakes.
    r, mu = dist.total_count, dist.mean
    expected = (
        torch.lgamma(counts + r)
        - torch.lgamma(r)
        - torch.lgamma(counts + 1)
        + r * (r.log() - (r + mu).log())
        + counts * (mu.log() - (r + mu).log())
    )
    torch.testing.assert_close(dist.log_prob(counts), expected)


def test_future_history_frozen_and_targets_unused():
    model, x, patterns = fixture()
    data = dict(x=x[:3], y=patterns[:3])
    with torch.no_grad():
        _, state, previous = m.objective(model, data)
    a = m.forecast(model, x[3:], previous, state, 2, 64, 10)
    changed = x[3:].clone()
    changed[1:, :, 2:] += 1000
    b = m.forecast(model, changed, previous, state, 2, 64, 10)
    for left, right in zip(a, b, strict=False):
        torch.testing.assert_close(left["mean"], right["mean"])
        torch.testing.assert_close(left["logq"], right["logq"])
        torch.testing.assert_close(left["logq"].exp().sum(-1), torch.ones(2, dtype=m.DTYPE))


def test_causal_filter_and_snapshot(tmp_path):
    model, x, patterns = fixture()
    counts = patterns.sum(-1)
    lag = torch.cat((torch.zeros_like(counts[:1]), counts[:-1]))
    with torch.no_grad():
        expected, expected_q, _ = model(x, lag)
        state = None
        for t in range(len(x)):
            found, found_q, state = model(x[t : t + 1], lag[t : t + 1], state)
            torch.testing.assert_close(found.mean, expected.mean[t : t + 1])
            torch.testing.assert_close(found_q, expected_q[t : t + 1])
        path = tmp_path / "state.pt"
        torch.save(model.state_dict(), path)
        restored = m.DeepAR(4, 2, 2, 8, True)
        restored.load_state_dict(torch.load(path, weights_only=True))
        found, found_q, _ = restored(x, lag)
        torch.testing.assert_close(found.mean, expected.mean)
        torch.testing.assert_close(found_q, expected_q)


def test_test_outcomes_only_update_subsequent_origins():
    model, x, patterns = fixture()
    data = {
        split: dict(
            x=x[a:b], y=patterns[a:b].clone(), dates=[f"2026-01-{d+1:02d}" for d in range(a, b)]
        )
        for split, a, b in [("train", 0, 2), ("validation", 2, 3), ("test", 3, 6)]
    }
    original = m.CONFIG["samples"]
    m.CONFIG["samples"] = 64
    try:
        a = m.score(model, data, 2, 7)
        data["test"]["y"][0] *= 10
        b = m.score(model, data, 2, 7)
        first = data["test"]["dates"][0]
        for left, right in zip(a["date_level"], b["date_level"], strict=False):
            if left["origin"] == first:
                assert left["predicted_exposures"] == right["predicted_exposures"]
    finally:
        m.CONFIG["samples"] = original

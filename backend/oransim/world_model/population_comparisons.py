"""Task-matched research comparators; frozen market implementation stays unchanged.

SeparatedAggregationMarket retains the exposure model but uses E[ell] E[p]
for expected marks. GRUMarket replaces the handcrafted transition with a GRU.
Comparison snapshots have their own format so ordinary runtime loading cannot
silently reinterpret a comparator as the proposed model.
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F  # noqa: N812 - standard PyTorch alias

from .differentiable_market import DTYPE, DifferentiableMarket, MarketConfig, MarketState


class SeparatedAggregationMarket(DifferentiableMarket):
    def _unselected_log_probability(self, batch, state, outcomes):
        g, c = batch.group, batch.campaign
        x = (batch.features - self.feature_center) / self.feature_scale
        latent, logw = self.population_nodes(g, state)
        first = self.means.new_zeros(1, self.config.dimensions)
        first[0, 0] = 1.0
        loading = torch.cat([first, self.behavior_loading], 0)
        base = (
            self.behavior_bias[g]
            + self.behavior_campaign[c]
            + x @ self.behavior_features.T
            + state.memory[g] @ self.memory_coefficients.T
            + state.fatigue[g, None] * self.fatigue_coefficients
            + state.common * self.environment_coefficients
        )
        dependency = outcomes @ torch.tril(self.dependencies, -1).T
        if outcomes.ndim == 2 and len(outcomes) == len(batch.group):
            logits = (base + dependency)[:, None, None, :] + latent @ loading.T
            lp = (outcomes[:, None, None, :] * logits - F.softplus(logits)).sum(-1)
            return torch.logsumexp((lp + logw).flatten(1), 1)
        raise ValueError("observed binary vectors required")

    def event_log_probability(self, batch, outcomes, state=None):
        self.validate(batch)
        if (
            outcomes.shape != (len(batch.group), len(self.config.heads))
            or not torch.isfinite(outcomes).all()
            or torch.any((outcomes != 0) & (outcomes != 1))
        ):
            raise ValueError("one binary vector per event required")
        return self._unselected_log_probability(
            batch, self.initial_state() if state is None else state, outcomes
        )

    def forward(self, batch, state=None):
        state = self.initial_state() if state is None else state
        out = super().forward(batch, state)
        g, c = batch.group, batch.campaign
        x = (batch.features - self.feature_center) / self.feature_scale
        u, logw = self.population_nodes(g, state)
        first = self.means.new_zeros(1, self.config.dimensions)
        first[0, 0] = 1.0
        loading = torch.cat([first, self.behavior_loading], 0)
        base = (
            self.behavior_bias[g]
            + self.behavior_campaign[c]
            + x @ self.behavior_features.T
            + state.memory[g] @ self.memory_coefficients.T
            + state.fatigue[g, None] * self.fatigue_coefficients
            + state.common * self.environment_coefficients
        )
        logits = (
            base[:, None, None, None, :]
            + (u @ loading.T)[:, :, :, None, :]
            + (self.outcomes @ torch.tril(self.dependencies, -1).T)[None, None, None, :, :]
        )
        lp = (self.outcomes[None, None, None, :, :] * logits - F.softplus(logits)).sum(-1)
        component_joint = (lp.exp() * logw.exp()[:, :, :, None]).sum(2)
        joint = torch.logsumexp((lp + logw[:, :, :, None]).flatten(1, 2), 1)
        marginal = joint.exp() @ self.outcomes
        out.update(
            joint_log_prob=joint,
            marginals=marginal,
            actions=out["exposures"][:, None] * marginal,
            component_actions=out["exposures"][:, None, None] * (component_joint @ self.outcomes),
        )
        return out


class GRUMarket(DifferentiableMarket):
    """Same observation likelihood with deterministic recurrent population state."""

    def __init__(self, config):
        if config.family != "point" or config.learn_population:
            raise ValueError("GRU comparator uses deterministic point population")
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.gru = torch.nn.GRUCell(len(config.heads) + 2, config.dimensions, dtype=DTYPE)
        for name in [
            "raw_retention",
            "raw_correction",
            "feedback",
            "fatigue_feedback",
            "memory_coefficients",
            "fatigue_coefficients",
            "environment_coefficients",
        ]:
            getattr(self, name).requires_grad_(False)

    def transition(self, state, batch, out, patterns=None):
        g = batch.group
        c = self.config

        def grouped(v):
            return v.new_zeros((c.groups,) + v.shape[1:]).index_add(0, g, v)

        n = out["exposures"] if patterns is None else patterns.sum(-1)
        actions = out["actions"] if patterns is None else patterns @ self.outcomes
        gn = grouped(n)
        expected = grouped(out["exposures"])
        rates = (grouped(actions) + 2 * self.reference_rates) / (gn[:, None] + 2)
        mass = (
            batch.population.new_zeros(c.campaigns * c.groups)
            .index_add(0, batch.campaign * c.groups + g, batch.population)
            .reshape(c.campaigns, c.groups)
            .max(0)
            .values
        )
        features = torch.cat(
            [
                rates,
                torch.log1p(gn / mass.clamp_min(1))[:, None],
                (torch.log1p(gn) - torch.log1p(expected))[:, None],
            ],
            1,
        )
        result = self.initial_state()
        result.shift = self.gru(features, state.shift)
        result.day = state.day + 1
        return result


def save_comparison(model, path, state, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "format": "population-comparison",
        "version": 1,
        "model_class": type(model).__name__,
        "config": asdict(model.config),
        "parameters": {k: v.detach().tolist() for k, v in model.state_dict().items()},
        "state": state.to_dict(),
        "metadata": metadata or {},
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".comparison-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, allow_nan=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_comparison(path):
    value = json.loads(Path(path).read_text())
    if value.get("format") != "population-comparison" or value.get("version") != 1:
        raise ValueError("unsupported comparison snapshot")
    cls = {
        "DifferentiableMarket": DifferentiableMarket,
        "SeparatedAggregationMarket": SeparatedAggregationMarket,
        "GRUMarket": GRUMarket,
    }[value["model_class"]]
    model = cls(MarketConfig(**value["config"]))
    params = {
        k: torch.tensor(v, dtype=model.state_dict()[k].dtype)
        for k, v in value["parameters"].items()
    }
    if not all(torch.isfinite(v).all() for v in params.values()):
        raise ValueError("nonfinite snapshot")
    model.load_state_dict(params)
    return model, MarketState.from_dict(value["state"]), value.get("metadata", {})

"""Supplementary GRU comparator with a trainable projection for every mark.

The original comparison module and its experiment remain frozen. This class
retains its GRU, exposure readout and input information, and frees the first
behavior head's recurrent readout alongside every other behavior head.
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - standard PyTorch alias

from .differentiable_market import MarketConfig, MarketState
from .population_comparisons import GRUMarket


class ProjectedGRUMarket(GRUMarket):
    def __init__(self, config):
        super().__init__(config)
        first = self.means.new_zeros(1, config.dimensions)
        first[0, 0] = 1.0
        # Identical initial probabilities to GRUMarket; all rows can now learn.
        self.behavior_projection = nn.Parameter(
            torch.cat([first, self.behavior_loading.detach().clone()], 0)
        )
        self.behavior_loading.requires_grad_(False)

    def behavior_logits(self, batch, state):
        x = (batch.features - self.feature_center) / self.feature_scale
        g, c = batch.group, batch.campaign
        latent, _ = self.population_nodes(g, state)
        # The comparator has exactly one deterministic point per cohort.
        return (
            self.behavior_bias[g]
            + self.behavior_campaign[c]
            + x @ self.behavior_features.T
            + state.memory[g] @ self.memory_coefficients.T
            + state.fatigue[g, None] * self.fatigue_coefficients
            + state.common * self.environment_coefficients
            + latent[:, 0, 0] @ self.behavior_projection.T
        )

    def forward(self, batch, state=None):
        state = self.initial_state() if state is None else state
        result = super().forward(batch, state)
        logits = self.behavior_logits(batch, state)[:, None, :]
        logits = logits + (self.outcomes @ torch.tril(self.dependencies, -1).T)[None, :, :]
        joint = (self.outcomes[None] * logits - F.softplus(logits)).sum(-1)
        marginal = joint.exp() @ self.outcomes
        actions = result["exposures"][:, None] * marginal
        result.update(
            joint_log_prob=joint,
            marginals=marginal,
            actions=actions,
            component_actions=actions[:, None, :],
        )
        return result

    def event_log_probability(self, batch, outcomes, state=None):
        self.validate(batch)
        if (
            outcomes.shape != (len(batch.group), len(self.config.heads))
            or not torch.isfinite(outcomes).all()
            or torch.any((outcomes != 0) & (outcomes != 1))
        ):
            raise ValueError("one binary vector per observed exposure required")
        state = self.initial_state() if state is None else state
        logits = self.behavior_logits(batch, state) + outcomes @ torch.tril(self.dependencies, -1).T
        return (outcomes * logits - F.softplus(logits)).sum(-1)


def save_projected_gru(model, path, state, metadata=None):
    if not isinstance(model, ProjectedGRUMarket):
        raise ValueError("ProjectedGRUMarket snapshot required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "format": "population-projected-gru",
        "version": 1,
        "config": asdict(model.config),
        "parameters": {k: v.detach().tolist() for k, v in model.state_dict().items()},
        "state": state.to_dict(),
        "metadata": metadata or {},
    }
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".projected-gru-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_projected_gru(path):
    value = json.loads(Path(path).read_text())
    if value.get("format") != "population-projected-gru" or value.get("version") != 1:
        raise ValueError("unsupported projected GRU snapshot")
    config = dict(value["config"])
    config["heads"] = tuple(config["heads"])
    model = ProjectedGRUMarket(MarketConfig(**config))
    parameters = {
        k: torch.tensor(v, dtype=model.state_dict()[k].dtype)
        for k, v in value["parameters"].items()
    }
    if not all(torch.isfinite(v).all() for v in parameters.values()):
        raise ValueError("nonfinite projected GRU parameters")
    model.load_state_dict(parameters)
    state = MarketState.from_dict(value["state"])
    initial = model.initial_state()
    for name in ["shift", "supply", "common", "fatigue", "memory"]:
        current = getattr(state, name)
        if current.shape != getattr(initial, name).shape or not torch.isfinite(current).all():
            raise ValueError("invalid projected GRU state")
    if state.day < 0:
        raise ValueError("invalid projected GRU date index")
    return model, state, value.get("metadata", {})

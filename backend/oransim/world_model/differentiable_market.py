"""Differentiable population-to-market observation and recurrent state model.

Research-only Torch implementation. Cells are disjoint population cohorts within
one campaign. Population mass is an observed input, distinct from mixture mass.
Exposure counts use an NB working likelihood; marks form a normalized joint
binary distribution. All latent quadrature nodes stay in the autograd graph.
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date as calendar_date
from datetime import timedelta
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - standard PyTorch alias

DTYPE = torch.float64


@dataclass(frozen=True)
class MarketConfig:
    groups: int
    campaigns: int
    features: int
    heads: tuple[str, ...] = ("click", "long_view", "like")
    components: int = 2
    dimensions: int = 1
    quadrature: int = 9
    family: str = "gaussian"
    learn_population: bool = True
    dynamic: bool = True
    dispersion: float = 50.0
    seed: int = 20260912
    history_feature_indices: tuple[int, ...] = ()

    def __post_init__(self):
        if (
            min(self.groups, self.campaigns, self.components, self.dimensions, self.quadrature) < 1
            or self.features < 0
        ):
            raise ValueError("invalid model dimensions")
        if not 1 <= len(self.heads) <= 8 or len(set(self.heads)) != len(self.heads):
            raise ValueError("one to eight distinct binary heads required")
        if self.family not in ("gaussian", "discrete", "point") or self.dispersion <= 0:
            raise ValueError("invalid family or dispersion")
        if any(i < 0 or i >= self.features for i in self.history_feature_indices):
            raise ValueError("invalid history feature indices")
        if self.family == "point" and self.components != 1:
            raise ValueError("point family uses one component")


@dataclass
class MarketBatch:
    group: torch.Tensor
    campaign: torch.Tensor
    features: torch.Tensor
    population: torch.Tensor
    effort: torch.Tensor

    @classmethod
    def create(cls, group, campaign, features, population, effort=None):
        population = torch.as_tensor(population, dtype=DTYPE)
        return cls(
            torch.as_tensor(group, dtype=torch.long),
            torch.as_tensor(campaign, dtype=torch.long),
            torch.as_tensor(features, dtype=DTYPE),
            population,
            torch.ones_like(population) if effort is None else torch.as_tensor(effort, dtype=DTYPE),
        )

    def to_dict(self):
        return {
            name: getattr(self, name).detach().cpu().tolist() for name in self.__dataclass_fields__
        }


@dataclass
class MarketState:
    shift: torch.Tensor
    supply: torch.Tensor
    common: torch.Tensor
    fatigue: torch.Tensor
    memory: torch.Tensor
    day: int = 0

    def detached(self):
        return MarketState(
            *(
                getattr(self, k).detach().clone()
                for k in ("shift", "supply", "common", "fatigue", "memory")
            ),
            self.day,
        )

    def to_dict(self):
        return {
            k: (v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v)
            for k, v in vars(self).items()
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            *(
                torch.tensor(value[k], dtype=DTYPE)
                for k in ("shift", "supply", "common", "fatigue", "memory")
            ),
            int(value["day"]),
        )


@dataclass
class MarketDay:
    date: str
    batch: MarketBatch
    patterns: torch.Tensor
    event_batch: MarketBatch | None = None
    event_outcomes: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["date"],
            MarketBatch.create(**value["batch"]),
            torch.tensor(value["patterns"], dtype=DTYPE),
            MarketBatch.create(**value["event_batch"]) if value.get("event_batch") else None,
            (
                torch.tensor(value["event_outcomes"], dtype=DTYPE)
                if value.get("event_outcomes") is not None
                else None
            ),
        )

    def to_dict(self):
        return {
            "date": self.date,
            "batch": self.batch.to_dict(),
            "patterns": self.patterns.tolist(),
            "event_batch": None if self.event_batch is None else self.event_batch.to_dict(),
            "event_outcomes": None if self.event_outcomes is None else self.event_outcomes.tolist(),
        }


class DifferentiableMarket(nn.Module):
    def __init__(self, config: MarketConfig):
        super().__init__()
        self.config = config
        g, c, f, k, d, h = (
            config.groups,
            config.campaigns,
            config.features,
            config.components,
            config.dimensions,
            len(config.heads),
        )
        # Explicit local generator avoids changing callers' random state.
        rng = torch.Generator().manual_seed(config.seed)
        self.mixture_logits = nn.Parameter(
            torch.zeros(g, k, dtype=DTYPE),
            requires_grad=config.learn_population and config.family != "point",
        )
        mu = torch.randn(g, k, d, generator=rng, dtype=DTYPE) * 0.08
        mu[:, :, 0] += torch.linspace(-0.6, 0.6, k, dtype=DTYPE) if k > 1 else 0.0
        if config.family == "point":
            mu.zero_()
        self.means = nn.Parameter(
            mu, requires_grad=config.learn_population and config.family != "point"
        )
        scale = torch.zeros(g, k, d, d, dtype=DTYPE)
        scale.diagonal(dim1=-2, dim2=-1).fill_(float(np.log(np.expm1(0.45))))
        self.raw_cholesky = nn.Parameter(
            scale, requires_grad=config.learn_population and config.family == "gaussian"
        )
        self.register_buffer("prior_means", mu.clone())
        self.register_buffer("feature_center", torch.zeros(f, dtype=DTYPE))
        self.register_buffer("feature_scale", torch.ones(f, dtype=DTYPE))
        self.register_buffer("reference_rates", torch.full((g, h), 0.1, dtype=DTYPE))
        self.register_buffer("base_log_intensity", torch.full((c, g), -1.0, dtype=DTYPE))
        self.intensity_bias = nn.Parameter(torch.zeros(c, g, dtype=DTYPE))
        self.intensity_features = nn.Parameter(torch.zeros(f, dtype=DTYPE))
        self.intensity_loading = nn.Parameter(torch.randn(d, generator=rng, dtype=DTYPE) * 0.05)
        self.behavior_bias = nn.Parameter(torch.zeros(g, h, dtype=DTYPE))
        self.behavior_campaign = nn.Parameter(torch.zeros(c, h, dtype=DTYPE))
        self.behavior_features = nn.Parameter(torch.zeros(h, f, dtype=DTYPE))
        self.behavior_loading = nn.Parameter(
            torch.randn(h - 1, d, generator=rng, dtype=DTYPE) * 0.2
        )
        self.dependencies = nn.Parameter(torch.zeros(h, h, dtype=DTYPE))
        # Recurrent, group-level exposure fatigue and released behavior memory.
        self.memory_coefficients = nn.Parameter(torch.zeros(h, h, dtype=DTYPE))
        self.fatigue_coefficients = nn.Parameter(torch.zeros(h, dtype=DTYPE))
        self.environment_coefficients = nn.Parameter(torch.zeros(h, dtype=DTYPE))
        self.raw_retention = nn.Parameter(
            torch.tensor([2.0, 4.0, 1.0, 0.0], dtype=DTYPE), requires_grad=config.dynamic
        )
        self.raw_correction = nn.Parameter(
            torch.tensor([1.0, 1.0], dtype=DTYPE), requires_grad=config.dynamic
        )
        self.feedback = nn.Parameter(
            torch.randn(h, d, generator=rng, dtype=DTYPE) * 0.02, requires_grad=config.dynamic
        )
        self.fatigue_feedback = nn.Parameter(
            torch.zeros(d, dtype=DTYPE), requires_grad=config.dynamic
        )
        # Choice alternatives share one preference distribution; utility varies by attributes.
        self.choice_features = nn.Parameter(torch.zeros(f, dtype=DTYPE))
        self.choice_interaction = nn.Parameter(torch.randn(f, d, generator=rng, dtype=DTYPE) * 0.1)
        self.raw_price_sensitivity = nn.Parameter(torch.zeros(d + 1, dtype=DTYPE))
        outcomes = torch.tensor(list(product([0.0, 1.0], repeat=h)), dtype=DTYPE)
        self.register_buffer("outcomes", outcomes)
        if config.family != "gaussian":
            nodes = torch.zeros(1, d, dtype=DTYPE)
            weights = torch.ones(1, dtype=DTYPE)
        elif d <= 2:
            x, w = np.polynomial.hermite.hermgauss(config.quadrature)
            index = np.array(list(product(range(len(x)), repeat=d)))
            nodes = torch.tensor(np.sqrt(2) * x[index], dtype=DTYPE)
            weights = torch.tensor(np.prod((w / np.sqrt(np.pi))[index], axis=1), dtype=DTYPE)
        else:
            # Fixed scrambled quasi-Monte Carlo nodes permit deterministic backprop.
            uniform = (
                torch.quasirandom.SobolEngine(d, scramble=True, seed=config.seed)
                .draw(config.quadrature)
                .to(DTYPE)
            )
            nodes = np.sqrt(2) * torch.erfinv(2 * uniform.clamp(1e-8, 1 - 1e-8) - 1)
            weights = torch.full((len(nodes),), 1 / len(nodes), dtype=DTYPE)
        self.register_buffer("nodes", nodes)
        self.register_buffer("node_weights", weights)

    def initial_state(self):
        c = self.config
        return MarketState(
            self.means.new_zeros(c.groups, c.dimensions),
            self.means.new_zeros(c.campaigns),
            self.means.new_zeros(()),
            self.means.new_zeros(c.groups),
            self.means.new_zeros(c.groups, len(c.heads)),
        )

    def validate(self, batch):
        c = self.config
        b = len(batch.group)
        if (
            batch.group.shape != (b,)
            or batch.campaign.shape != (b,)
            or batch.features.shape != (b, c.features)
            or batch.population.shape != (b,)
            or batch.effort.shape != (b,)
        ):
            raise ValueError("incompatible cohort shapes")
        if (
            b == 0
            or torch.any(batch.group < 0)
            or torch.any(batch.group >= c.groups)
            or torch.any(batch.campaign < 0)
            or torch.any(batch.campaign >= c.campaigns)
        ):
            raise ValueError("invalid group or campaign")
        if (
            not all(
                torch.isfinite(t).all() for t in (batch.features, batch.population, batch.effort)
            )
            or torch.any(batch.population < 0)
            or torch.any(batch.effort < 0)
        ):
            raise ValueError("finite inputs and nonnegative population/effort required")

    def cholesky(self):
        lower = torch.tril(self.raw_cholesky, diagonal=-1)
        diag = F.softplus(self.raw_cholesky.diagonal(dim1=-2, dim2=-1)) + 1e-4
        return lower + torch.diag_embed(diag)

    def population_nodes(self, group, state):
        means = self.means[group] + state.shift[group, None, :]
        if self.config.family == "gaussian":
            latent = means[:, :, None, :] + torch.einsum(
                "bkij,qj->bkqi", self.cholesky()[group], self.nodes
            )
        else:
            latent = means[:, :, None, :]
        logweights = (
            F.log_softmax(self.mixture_logits[group], -1)[:, :, None]
            + self.node_weights.log()[None, None, :]
        )
        return latent, logweights

    def forward(self, batch: MarketBatch, state: MarketState | None = None):
        self.validate(batch)
        state = self.initial_state() if state is None else state
        g, c = batch.group, batch.campaign
        x = (batch.features - self.feature_center) / self.feature_scale
        u, logw = self.population_nodes(g, state)
        lograte = (
            self.base_log_intensity[c, g]
            + self.intensity_bias[c, g]
            + x @ self.intensity_features
            + state.common
            + state.supply[c]
        )[:, None, None] + u @ self.intensity_loading
        rate = lograte.clamp(-16, 10).exp() * batch.effort[:, None, None]
        intensity = (logw.exp() * rate).sum((1, 2))
        # Conditional marks given exposure are tilted by the exposure intensity.
        # At zero effort use the limiting conditional distribution at effort 1.
        tilted = logw + lograte.clamp(-16, 10)
        exposure_logw = tilted - torch.logsumexp(tilted.flatten(1), 1)[:, None, None]
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
        logits = base[:, None, None, :] + u @ loading.T
        dependency = self.outcomes @ torch.tril(self.dependencies, diagonal=-1).T
        logits = logits[:, :, :, None, :] + dependency[None, None, None, :, :]
        lp = (self.outcomes[None, None, None, :, :] * logits - F.softplus(logits)).sum(-1)
        joint_logp = torch.logsumexp((lp + exposure_logw[:, :, :, None]).flatten(1, 2), 1)
        marginal = joint_logp.exp() @ self.outcomes
        exposures = batch.population * intensity
        # Conditional Poisson reach, integrated over individual heterogeneity.
        reach = batch.population * (logw.exp() * (-torch.expm1(-rate))).sum((1, 2))
        component_exposures = batch.population[:, None] * (logw.exp() * rate).sum(2)
        component_joint = (lp.exp() * logw.exp()[:, :, :, None] * rate[:, :, :, None]).sum(
            2
        ) * batch.population[:, None, None]
        return {
            "exposures": exposures,
            "reach_poisson": reach,
            "joint_log_prob": joint_logp,
            "marginals": marginal,
            "actions": exposures[:, None] * marginal,
            "component_exposures": component_exposures,
            "component_actions": component_joint @ self.outcomes,
        }

    def event_log_probability(self, batch, outcomes, state=None):
        self.validate(batch)
        state = self.initial_state() if state is None else state
        if (
            outcomes.shape != (len(batch.group), len(self.config.heads))
            or not torch.isfinite(outcomes).all()
            or torch.any((outcomes != 0) & (outcomes != 1))
        ):
            raise ValueError("one binary vector per observed exposure required")
        g, c = batch.group, batch.campaign
        x = (batch.features - self.feature_center) / self.feature_scale
        u, logw = self.population_nodes(g, state)
        lograte = (
            self.base_log_intensity[c, g]
            + self.intensity_bias[c, g]
            + x @ self.intensity_features
            + state.common
            + state.supply[c]
        )[:, None, None] + u @ self.intensity_loading
        tilted = logw + lograte.clamp(-16, 10)
        logw = tilted - torch.logsumexp(tilted.flatten(1), 1)[:, None, None]
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
            + outcomes @ torch.tril(self.dependencies, -1).T
        )
        logits = base[:, None, None, :] + u @ loading.T
        conditional = (outcomes[:, None, None, :] * logits - F.softplus(logits)).sum(-1)
        return torch.logsumexp((conditional + logw).flatten(1), 1)

    def day_loss(self, day, state=None):
        parts, out = self.loss(day.batch, day.patterns, state)
        if day.event_batch is not None:
            if day.event_outcomes is None or len(day.event_batch.group) != int(day.patterns.sum()):
                raise ValueError("event labels must cover exactly the aggregate exposures")
            parts["cohort_joint_nll"] = parts["joint_nll"]
            parts["joint_nll"] = -self.event_log_probability(
                day.event_batch, day.event_outcomes, state
            ).mean()
        return parts, out

    def loss(self, batch, patterns, state=None):
        out = self(batch, state)
        n = patterns.sum(-1)
        if (
            patterns.shape != out["joint_log_prob"].shape
            or not torch.isfinite(patterns).all()
            or torch.any(patterns < 0)
        ):
            raise ValueError("invalid joint pattern counts")
        if torch.any((out["exposures"] == 0) & (n > 0)):
            raise ValueError("positive counts at zero population or effort")
        mean = out["exposures"].clamp_min(1e-12)
        r = self.config.dispersion
        # Exact NB log likelihood and multinomial marks, omitting mark combinatorial constant.
        nb = (
            torch.lgamma(n + r)
            - torch.lgamma(n + 1)
            - torch.lgamma(torch.tensor(r, dtype=DTYPE))
            + r * (np.log(r) - torch.log(r + mean))
            + n * (torch.log(mean) - torch.log(r + mean))
        )
        return {
            "count_nll": -nb.mean(),
            "joint_nll": -(patterns * out["joint_log_prob"]).sum() / n.sum().clamp_min(1),
        }, out

    def parameter_counts(self):
        c = self.config
        g, k, d, h = c.groups, c.components, c.dimensions, len(c.heads)
        population = (
            0
            if not c.learn_population or c.family == "point"
            else g * (k - 1 + k * d + (k * d * (d + 1) // 2 if c.family == "gaussian" else 0))
        )
        excluded = {
            "mixture_logits",
            "means",
            "raw_cholesky",
            "raw_price_sensitivity",
            "choice_features",
            "choice_interaction",
        }
        observation = sum(
            p.numel() for n, p in self.named_parameters() if p.requires_grad and n not in excluded
        )
        observation -= h * (h + 1) // 2  # masked dependency entries
        if not c.dynamic:
            observation -= h * h + 2 * h  # recurrent feature coefficients are inactive
        return {
            "effective_population": population,
            "effective_exposure_behavior_and_transition": observation,
            "effective_total": population + observation,
            "allocated_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def regularizer(self):
        penalty = self.means.new_zeros(())
        for name, p in self.named_parameters():
            if (
                p.requires_grad
                and name
                not in ("raw_cholesky", "raw_retention", "raw_correction", "raw_price_sensitivity")
                and not name.startswith("choice_")
            ):
                penalty = penalty + p.square().sum()
        if self.config.family == "gaussian" and self.config.learn_population:
            scale = self.cholesky()
            diag = scale.diagonal(dim1=-2, dim2=-1)
            penalty = (
                penalty
                + ((diag / 0.45).square() - 2 * torch.log(diag / 0.45) - 1).sum()
                + torch.tril(scale, -1).square().sum()
            )
        return penalty

    def transition(self, state, batch, out, patterns=None):
        """Release observed or expected daily feedback after scoring the whole day.

        Means/memory here are cohort-level sufficient features. User-level histories
        can additionally enter batch.features; they are prepared outside autograd.
        """
        if not self.config.dynamic:
            result = self.initial_state()
            result.day = state.day + 1
            return result
        c = self.config
        g = batch.group
        j = batch.campaign
        n = out["exposures"] if patterns is None else patterns.sum(-1)
        actions = out["actions"] if patterns is None else patterns @ self.outcomes

        def sum_by(values, index, size):
            return values.new_zeros((size,) + values.shape[1:]).index_add(0, index, values)

        gn = sum_by(n, g, c.groups)
        ga = sum_by(actions, g, c.groups)
        # Each campaign may see the same population; use mean over campaign cells.
        campaign_mass = sum_by(batch.population, j * c.groups + g, c.campaigns * c.groups).reshape(
            c.campaigns, c.groups
        )
        gm = campaign_mass.max(0).values
        dose = gn / gm.clamp_min(1)
        rates = (ga + 2 * self.reference_rates) / (gn[:, None] + 2)
        cn = sum_by(n, j, c.campaigns)
        ce = sum_by(out["exposures"], j, c.campaigns)
        innovation = torch.log1p(cn) - torch.log1p(ce)
        active = sum_by(batch.effort * batch.population, j, c.campaigns) > 0
        common_innovation = (innovation * active).sum() / active.sum().clamp_min(1)
        rho = torch.sigmoid(self.raw_retention)
        gain = torch.sigmoid(self.raw_correction)
        common = rho[1] * state.common + gain[0] * common_innovation
        supply = rho[1] * state.supply + gain[1] * active * (innovation - common_innovation)
        supply = supply - supply.mean()  # fixed common/contrast gauge
        fatigue = rho[2] * state.fatigue + (1 - rho[2]) * (-torch.expm1(-dose))
        memory = rho[3] * state.memory + (1 - rho[3]) * rates
        shift = (
            rho[0] * state.shift
            + (rates - self.reference_rates) @ self.feedback
            - fatigue[:, None] * self.fatigue_feedback
        )
        return MarketState(shift, supply, common, fatigue, memory, state.day + 1)

    def future_batch(self, template, origin):
        features = template.features.clone()
        if self.config.history_feature_indices:
            if not torch.equal(template.group, origin.group) or not torch.equal(
                template.campaign, origin.campaign
            ):
                raise ValueError("future history requires aligned cohorts")
            indices = list(self.config.history_feature_indices)
            features[:, indices] = origin.features[:, indices]
        return MarketBatch(
            template.group, template.campaign, features, origin.population, template.effort
        )

    def rollout(self, batches, state=None):
        state = self.initial_state() if state is None else state
        outputs = []
        for batch in batches:
            out = self(batch, state)
            outputs.append(out)
            state = self.transition(state, batch, out)
        return outputs, state

    def choice(self, attributes, population, prices=None, available=None, state=None):
        """One opportunity per mass unit; column 0 is the outside/no-purchase option."""
        state = self.initial_state() if state is None else state
        x = torch.as_tensor(attributes, dtype=DTYPE)
        mass = torch.as_tensor(population, dtype=DTYPE)
        if (
            x.ndim != 2
            or len(x) == 0
            or x.shape[1] != self.config.features
            or mass.shape != (self.config.groups,)
            or torch.any(mass < 0)
            or not torch.isfinite(x).all()
            or not torch.isfinite(mass).all()
        ):
            raise ValueError("invalid choice attributes or opportunity mass")
        price = (
            torch.zeros(len(x), dtype=DTYPE)
            if prices is None
            else torch.as_tensor(prices, dtype=DTYPE)
        )
        mask = (
            torch.ones(len(x), dtype=torch.bool)
            if available is None
            else torch.as_tensor(available, dtype=torch.bool)
        )
        if (
            price.shape != (len(x),)
            or mask.shape != (len(x),)
            or not torch.isfinite(price).all()
            or torch.any(price < 0)
        ):
            raise ValueError("invalid prices or availability")
        group = torch.arange(self.config.groups)
        u, logw = self.population_nodes(group, state)
        utility = (x @ self.choice_features)[None, None, None, :] + torch.einsum(
            "gkqd,bd->gkqb", u, x @ self.choice_interaction
        )
        sensitivity = F.softplus(self.raw_price_sensitivity[0] + u @ self.raw_price_sensitivity[1:])
        utility = utility - sensitivity[:, :, :, None] * price
        utility = utility.masked_fill(~mask, -torch.inf)
        probabilities = torch.softmax(
            torch.cat([torch.zeros_like(utility[..., :1]), utility], -1), -1
        )
        component = mass[:, None, None] * (probabilities * logw.exp()[:, :, :, None]).sum(2)
        group_demand = component.sum(1)
        demand = group_demand.sum(0)
        purchased = demand[1:].sum()
        return {
            "demand": demand,
            "group_demand": group_demand,
            "component_demand": component,
            "market_share": demand[1:] / purchased.clamp_min(1e-12),
            "purchase_probability": torch.where(
                mass.sum() > 0,
                1 - demand[0] / mass.sum().clamp_min(1e-12),
                torch.zeros_like(demand[0]),
            ),
        }

    def configure_features(self, days):
        x = torch.cat(
            [(d.batch if d.event_batch is None else d.event_batch).features for d in days]
        )
        self.feature_center.copy_(x.mean(0))
        self.feature_scale.copy_(x.std(0, unbiased=False).clamp_min(0.1))

    def save(self, path, state=None, metadata=None, simulation=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "differentiable-market",
            "version": 1,
            "config": asdict(self.config),
            "parameters": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
            "state": (self.initial_state() if state is None else state).to_dict(),
            "metadata": metadata or {},
            "simulation": bool(simulation),
        }
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".market-")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text + "\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if value.get("format") != "differentiable-market" or value.get("version") != 1:
            raise ValueError("unsupported market snapshot")
        config = dict(value["config"])
        config["heads"] = tuple(config["heads"])
        model = cls(MarketConfig(**config))
        state_dict = {
            k: torch.tensor(v, dtype=model.state_dict()[k].dtype)
            for k, v in value["parameters"].items()
        }
        if not all(torch.isfinite(v).all() for v in state_dict.values()):
            raise ValueError("nonfinite model parameters")
        model.load_state_dict(state_dict, strict=True)
        state = MarketState.from_dict(value["state"])
        expected = model.initial_state()
        for k in ("shift", "supply", "common", "fatigue", "memory"):
            v = getattr(state, k)
            if v.shape != getattr(expected, k).shape or not torch.isfinite(v).all():
                raise ValueError("invalid recurrent state")
        if state.day < 0:
            raise ValueError("invalid state day")
        return model, state, value.get("metadata", {}), value.get("simulation", False)


def score_days(model, days, state=None, update=True):
    state = model.initial_state() if state is None else state
    rows = []
    with torch.no_grad():
        for day in days:
            losses, out = model.day_loss(day, state)
            n = day.patterns.sum(-1)
            actual = day.patterns @ model.outcomes
            rows.append(
                {
                    "date": day.date,
                    "events": int(n.sum()),
                    "joint_nll": float(losses["joint_nll"]),
                    "count_nll": float(losses["count_nll"]),
                    "cohort_joint_nll": float(losses.get("cohort_joint_nll", losses["joint_nll"])),
                    "exposures": float(n.sum()),
                    "predicted_exposures": float(out["exposures"].sum()),
                    "actual_actions": actual.sum(0).tolist(),
                    "predicted_actions": out["actions"].sum(0).tolist(),
                }
            )
            if update:
                state = model.transition(state, day.batch, out, day.patterns)
    return rows, state.detached()


def fit_market(
    model,
    train,
    validation=(),
    epochs=120,
    learning_rate=0.025,
    regularization=0.001,
    count_weight=0.05,
    multi_step_weight=0.0,
    patience=25,
):
    """Chronological teacher forcing plus optional label-free two-step forecasts.

    Validation only selects an epoch. It never contributes gradients, feature
    statistics or parameter updates. Aggregate mark counts are sufficient for
    exchangeable observations with identical cohort inputs.
    """
    if not train or epochs < 1:
        raise ValueError("training days and positive epochs required")
    dates = [calendar_date.fromisoformat(d.date) for d in [*train, *validation]]
    if any((b - a).days != 1 for a, b in zip(dates, dates[1:], strict=False)):
        raise ValueError("daily training and validation records must be consecutive and disjoint")
    model.configure_features(train)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    trace = []
    best = float("inf")
    best_weights = None
    best_epoch = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        state = model.initial_state()
        loss = state.common
        for i, day in enumerate(train):
            parts, out = model.day_loss(day, state)
            loss = loss + parts["joint_nll"] + count_weight * parts["count_nll"]
            if multi_step_weight and i + 1 < len(train):
                forecast_state = model.transition(state, day.batch, out)
                future_batch = model.future_batch(train[i + 1].batch, day.batch)
                future, _ = model.loss(future_batch, train[i + 1].patterns, forecast_state)
                loss = loss + multi_step_weight * (
                    future["joint_nll"] + count_weight * future["count_nll"]
                )
            state = model.transition(state, day.batch, out, day.patterns)
        loss = loss / len(train) + regularization * model.regularizer() / max(
            1, sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite market training objective")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite market gradients")
        optimizer.step()
        _, end = score_days(model, train)
        val, _ = score_days(model, validation or train, end if validation else None)
        objective = np.mean([v["joint_nll"] + count_weight * v["count_nll"] for v in val])
        trace.append(
            {
                "epoch": epoch + 1,
                "train_objective": float(loss.detach()),
                "validation_objective": float(objective),
            }
        )
        if objective < best - 1e-7:
            best = objective
            best_epoch = epoch + 1
            best_weights = deepcopy(model.state_dict())
        if epoch + 1 - best_epoch >= patience:
            break
    model.load_state_dict(best_weights)
    _, state = score_days(model, train)
    return state, {
        "epochs": len(trace),
        "selected_epoch": best_epoch,
        "validation_objective": best,
        "trace": trace,
    }


class MarketRuntime:
    """Persistent real/simulated state with isolated deterministic and stochastic forecasts."""

    def __init__(self, model, state=None, simulation=False, metadata=None):
        self.model = model
        self.state = model.initial_state() if state is None else state
        self.simulation = simulation
        self.metadata = metadata or {}

    def observe(self, day):
        if self.simulation:
            raise ValueError("real observations require a real runtime")
        current = calendar_date.fromisoformat(day.date)
        if (
            self.metadata.get("last_date")
            and (current - calendar_date.fromisoformat(self.metadata["last_date"])).days != 1
        ):
            raise ValueError("observations must advance exactly one day; include empty days")
        rows, next_state = score_days(self.model, [day], self.state)
        self.state = next_state
        self.metadata = {**self.metadata, "last_date": day.date}
        return rows[0]

    def forecast(self, batches, trajectories=128, seed=0):
        if not batches or trajectories < 2:
            raise ValueError("forecast needs plans and at least two trajectories")
        batches = [self.model.future_batch(b, batches[0]) for b in batches]
        paths = []
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            deterministic, _ = self.model.rollout(batches, self.state)
            for _ in range(trajectories):
                state = self.state.detached()
                path = []
                for batch in batches:
                    out = self.model(batch, state)
                    means = out["exposures"].numpy()
                    r = self.model.config.dispersion
                    count = rng.poisson(rng.gamma(r, means / r))
                    probabilities = out["joint_log_prob"].exp().numpy()
                    probabilities /= probabilities.sum(1, keepdims=True)
                    patterns = torch.tensor(
                        np.array(
                            [
                                rng.multinomial(int(n), p)
                                for n, p in zip(count, probabilities, strict=False)
                            ]
                        ),
                        dtype=DTYPE,
                    )
                    actions = patterns @ self.model.outcomes
                    path.append([float(count.sum()), *actions.sum(0).tolist()])
                    state = self.model.transition(state, batch, out, patterns)
                paths.append(path)
        array = np.array(paths)
        return {
            "trajectories": trajectories,
            "mean": array.mean(0).tolist(),
            "median": np.median(array, axis=0).tolist(),
            "lower": np.quantile(array, 0.05, axis=0).tolist(),
            "upper": np.quantile(array, 0.95, axis=0).tolist(),
            "deterministic_mean_path": [
                [float(o["exposures"].sum()), *o["actions"].sum(0).tolist()] for o in deterministic
            ],
        }

    def simulate(self, batch, seed=0):
        if not self.simulation:
            raise ValueError("simulation requires an isolated branch")
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            out = self.model(batch, self.state)
            r = self.model.config.dispersion
            n = rng.poisson(rng.gamma(r, out["exposures"].numpy() / r))
            p = out["joint_log_prob"].exp().numpy()
            p /= p.sum(1, keepdims=True)
            patterns = torch.tensor(
                np.array([rng.multinomial(int(v), q) for v, q in zip(n, p, strict=False)]),
                dtype=DTYPE,
            )
            self.state = self.model.transition(self.state, batch, out, patterns).detached()
        if self.metadata.get("last_date"):
            self.metadata = {
                **self.metadata,
                "last_date": str(
                    calendar_date.fromisoformat(self.metadata["last_date"]) + timedelta(days=1)
                ),
            }
        return patterns

    def branch(self):
        return MarketRuntime(
            deepcopy(self.model), self.state.detached(), True, deepcopy(self.metadata)
        )

    def save(self, path):
        self.model.save(path, self.state, self.metadata, self.simulation)

    @classmethod
    def load(cls, path):
        model, state, metadata, simulation = DifferentiableMarket.load(path)
        return cls(model, state, simulation, metadata)

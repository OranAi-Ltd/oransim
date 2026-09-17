"""研究级 G–C–P 日步长闭环：到达、触达、潜在人群、行为、反馈及滤波。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax

from oransim.data.gaussian_population_dynamics import ilr_basis


@dataclass(frozen=True)
class LoopConfig:
    base_arrivals: float = 1000.0
    arrival_dispersion: float = 50.0
    population_size: float = 10000.0
    recommendation_gain: float = 0.5
    momentum_arrival_gain: float = 1.0
    novelty_decay: float = 0.01
    momentum_retention: float = 0.7
    fatigue_retention: float = 0.85
    response_retention: float = 0.95
    social_gain: float = 0.2
    fatigue_gain: float = 0.15
    fatigue_selection_gain: float = 1.0
    mixture_gain: float = 0.2
    mixture_refresh: float = 0.02
    response_process_variance: float = 0.001
    arrival_process_variance: float = 0.01
    selection_process_variance: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("configuration must be finite")
        if min(self.base_arrivals, self.arrival_dispersion, self.population_size) <= 0:
            raise ValueError("positive arrival scale, dispersion and population required")
        for name in ["momentum_retention", "fatigue_retention", "response_retention"]:
            if not 0 <= getattr(self, name) < 1:
                raise ValueError("retention must be in [0,1)")
        if not 0 < self.mixture_refresh <= 1:
            raise ValueError("mixture refresh must be in (0,1]")
        if (
            min(
                self.response_process_variance,
                self.arrival_process_variance,
                self.selection_process_variance,
                self.novelty_decay,
            )
            < 0
        ):
            raise ValueError("noise and novelty decay must be nonnegative")


@dataclass(frozen=True)
class WorldAction:
    """当日已知的 G、内容与投放输入；effort 为相对投放量，不换算货币。"""

    content_logits: np.ndarray
    targeting_logits: np.ndarray
    exposure_effort: float = 1.0
    platform_log_supply: float = 0.0
    attention_logit: float = 0.0


@dataclass(frozen=True)
class WorldObservation:
    day: int
    exposures: np.ndarray
    responses: np.ndarray
    provenance: str = "real"


@dataclass
class LoopState:
    day: int
    population_weights: np.ndarray
    mixture_weights: np.ndarray
    means: np.ndarray
    heterogeneity_variance: np.ndarray
    mean_posterior_variance: np.ndarray
    fatigue: np.ndarray
    selection_mean: np.ndarray
    selection_covariance: np.ndarray
    momentum: float = 0.0
    log_arrival_offset: float = 0.0
    arrival_posterior_variance: float = 0.1
    cumulative_exposures: int = 0
    cumulative_responses: int = 0


@dataclass(frozen=True)
class WorldPrediction:
    day: int
    expected_exposures: float
    exposure_weights: np.ndarray
    group_probabilities: np.ndarray
    component_probabilities: np.ndarray
    expected_responses: float
    response_count_variance: float


@dataclass(frozen=True)
class MicroBatch:
    group: np.ndarray
    component: np.ndarray
    latent_response: np.ndarray
    response: np.ndarray


class PopulationWorldLoop:
    """同一 G–C–P 状态同时供给宏观积分和真实潜在个体抽样。

    每组共享响应校正的高斯后验；群内成分差异与方差由机制传播。
    聚合二元反馈仅校正组偏移，不假装能识别每个成分的全部参数。
    """

    def __init__(
        self,
        population_weights,
        mixture_weights,
        means,
        heterogeneity_variance,
        config=LoopConfig(),  # noqa: B008 - frozen, immutable configuration
        mean_posterior_variance=0.1,
        arrival_posterior_variance=0.1,
    ):
        q = np.asarray(population_weights, float)
        pi = np.asarray(mixture_weights, float)
        mu = np.asarray(means, float)
        variance = np.asarray(heterogeneity_variance, float)
        if (
            q.ndim != 1
            or not len(q)
            or pi.ndim != 2
            or pi.shape[0] != len(q)
            or pi.shape[1] < 1
            or mu.shape != pi.shape
            or variance.shape != pi.shape
        ):
            raise ValueError("incompatible population arrays")
        if (
            not all(np.isfinite(x).all() for x in (q, pi, mu, variance))
            or np.any(q < 0)
            or q.sum() <= 0
            or np.any(pi <= 0)
            or np.any(variance < 0)
        ):
            raise ValueError("invalid population probabilities or latent variance")
        if (
            not np.isfinite(mean_posterior_variance)
            or mean_posterior_variance < 0
            or not np.isfinite(arrival_posterior_variance)
            or arrival_posterior_variance < 0
        ):
            raise ValueError("invalid posterior variance")
        self.config = config
        self._prior_means = mu.copy()
        self._prior_pi = pi / pi.sum(1, keepdims=True)
        self._prior_variance = variance.copy()
        self._selection_basis = ilr_basis(len(q)) if len(q) > 1 else np.zeros((0, 1))
        self._state = LoopState(
            0,
            q / q.sum(),
            self._prior_pi.copy(),
            mu.copy(),
            variance.copy(),
            np.full(len(q), mean_posterior_variance),
            np.zeros(len(q)),
            np.zeros(len(q) - 1),
            np.eye(len(q) - 1) * (0.1 if config.selection_process_variance > 0 else 0.0),
            arrival_posterior_variance=arrival_posterior_variance,
        )
        self._simulation = False
        self._conditioned_parameters = False
        self._nodes, self._weights = np.polynomial.hermite.hermgauss(24)
        self._weights /= np.sqrt(np.pi)
        self._reference_rate = float(
            self._state.population_weights
            @ (self._prior_pi * self._component_response(np.zeros(len(q)), mu, variance)).sum(1)
        )

    @property
    def state(self):
        return deepcopy(self._state)

    def branch(self, rng=None):
        """复制当前世界；传入 rng 时抽取一次共享参数后验，供成对方案比较。"""
        branch = deepcopy(self)
        branch._simulation = True
        if rng is not None:
            s = branch._state
            s.means += (
                rng.normal(size=len(s.fatigue))[:, None]
                * np.sqrt(s.mean_posterior_variance)[:, None]
            )
            s.log_arrival_offset += float(rng.normal() * np.sqrt(s.arrival_posterior_variance))
            if s.selection_covariance.size and np.any(s.selection_covariance):
                s.selection_mean += rng.multivariate_normal(
                    np.zeros(len(s.selection_mean)), s.selection_covariance
                )
            s.selection_covariance[:] = 0
            s.mean_posterior_variance[:] = 0
            s.arrival_posterior_variance = 0
            branch._conditioned_parameters = True
        return branch

    def _action(self, action):
        shape = self._state.population_weights.shape
        content = np.asarray(action.content_logits, float)
        target = np.asarray(action.targeting_logits, float)
        if (
            content.shape != shape
            or target.shape != shape
            or not np.isfinite(content).all()
            or not np.isfinite(target).all()
            or not all(
                np.isfinite(x)
                for x in [
                    action.exposure_effort,
                    action.platform_log_supply,
                    action.attention_logit,
                ]
            )
            or action.exposure_effort < 0
        ):
            raise ValueError("invalid or unavailable action inputs")
        return content, target

    def _component_response(self, content, means=None, variance=None):
        s = self._state
        mu = s.means if means is None else means
        var = (
            s.heterogeneity_variance + s.mean_posterior_variance[:, None]
            if variance is None
            else variance
        )
        logits = (
            content[:, None, None] + mu[:, :, None] + np.sqrt(2 * var)[:, :, None] * self._nodes
        )
        return expit(logits) @ self._weights

    def predict(self, action):
        content, target = self._action(action)
        s = self._state
        c = self.config
        p_comp = self._component_response(content + action.attention_logit)
        p = (s.mixture_weights * p_comp).sum(1)
        selection = (
            target
            + c.recommendation_gain * p
            - c.fatigue_selection_gain * s.fatigue
            + self._selection_basis.T @ s.selection_mean
        )
        logw = np.full_like(selection, -np.inf)
        positive = s.population_weights > 0
        logw[positive] = np.log(s.population_weights[positive]) + selection[positive]
        w = softmax(logw)
        if s.selection_covariance.size and np.any(s.selection_covariance):
            # Symmetric cubature in orthogonal log-ratio space; covariance retains
            # correlations and is invariant to choosing a reference category.
            eigen, vectors = np.linalg.eigh(s.selection_covariance)
            points = vectors * np.sqrt(np.maximum(eigen, 0) * len(eigen))
            shifts = self._selection_basis.T @ points
            w = (
                softmax(logw[:, None] + shifts, axis=0).sum(1)
                + softmax(logw[:, None] - shifts, axis=0).sum(1)
            ) / (2 * len(eigen))
        arrivals = float(
            self._arrival_rates(action, np.sqrt(2 * s.arrival_posterior_variance) * self._nodes)
            @ self._weights
        )
        if not np.isfinite(arrivals):
            raise ValueError("arrival expectation overflow")
        rate = float(w @ p)
        # Conditional count variance, given arrival offset and marginal response readout.
        # Shared parameter uncertainty requires branch Monte Carlo, not this expression.
        return WorldPrediction(
            s.day,
            float(arrivals),
            w,
            p,
            p_comp,
            float(arrivals * rate),
            float(arrivals * rate + (arrivals * rate) ** 2 / c.arrival_dispersion),
        )

    def _arrival_rates(self, action, offsets):
        s = self._state
        c = self.config
        log_supply = np.clip(
            action.platform_log_supply
            + s.log_arrival_offset
            + c.momentum_arrival_gain * s.momentum
            - c.novelty_decay * s.day
            + offsets,
            -8,
            8,
        )
        return c.base_arrivals * action.exposure_effort * np.exp(log_supply)

    def _sample(self, action, rng, materialize_agents=True):
        if not self._simulation:
            raise RuntimeError("sample and advance only an explicit simulation branch")
        # A simulation path must condition on a parameter draw so shared uncertainty
        # is not incorrectly resampled independently for each person.
        if not self._conditioned_parameters:
            raise RuntimeError("create a branch with rng to sample shared posterior parameters")
        prediction = self.predict(action)
        c = self.config
        s = self._state
        total = int(
            rng.poisson(
                rng.gamma(
                    c.arrival_dispersion, prediction.expected_exposures / c.arrival_dispersion
                )
            )
        )
        if not materialize_agents:
            n = rng.multinomial(total, prediction.exposure_weights)
            k = rng.binomial(n, prediction.group_probabilities)
            return prediction, WorldObservation(s.day, n, k, "simulated"), None
        group = rng.choice(len(s.fatigue), size=total, p=prediction.exposure_weights)
        component = np.empty(total, int)
        for g in range(len(s.fatigue)):
            mask = group == g
            component[mask] = rng.choice(
                s.means.shape[1], size=int(mask.sum()), p=s.mixture_weights[g]
            )
        latent = rng.normal(
            s.means[group, component], np.sqrt(s.heterogeneity_variance[group, component])
        )
        content, _ = self._action(action)
        response = rng.binomial(1, expit(content[group] + action.attention_logit + latent))
        n = np.bincount(group, minlength=len(s.fatigue))
        k = np.bincount(group, weights=response, minlength=len(s.fatigue)).astype(int)
        return (
            prediction,
            WorldObservation(s.day, n, k, "simulated"),
            MicroBatch(group, component, latent, response),
        )

    def _check_observation(self, observation):
        n = np.asarray(observation.exposures)
        k = np.asarray(observation.responses)
        if (
            observation.day != self._state.day
            or n.shape != self._state.fatigue.shape
            or k.shape != n.shape
            or not np.issubdtype(n.dtype, np.integer)
            or not np.issubdtype(k.dtype, np.integer)
            or np.any(n < 0)
            or np.any(k < 0)
            or np.any(k > n)
        ):
            raise ValueError("dated per-group integer exposure/response counts required")
        return n, k

    def _assimilate(self, action, prediction, n, k):
        s = self._state
        c = self.config
        total = int(n.sum())
        if prediction.expected_exposures > 0:
            residual = np.log((total + 0.5) / (float(self._arrival_rates(action, 0.0)) + 0.5))
            noise = 1 / (total + 0.5) + 1 / c.arrival_dispersion
            gain = s.arrival_posterior_variance / (s.arrival_posterior_variance + noise)
            s.log_arrival_offset += gain * residual
            s.arrival_posterior_variance *= 1 - gain
        if total and s.selection_covariance.size and np.any(s.selection_covariance):
            b = self._selection_basis
            content, target = self._action(action)
            selection = (
                target
                + c.recommendation_gain * prediction.group_probabilities
                - c.fatigue_selection_gain * s.fatigue
            )
            baseline = np.log(np.maximum(s.population_weights, 1e-12)) + selection
            measured = b @ (np.log(n + 0.5) - baseline)
            noise = (b * (1 / (n + 0.5))) @ b.T
            prior = s.selection_covariance
            gain = np.linalg.solve(prior + noise, prior).T
            s.selection_mean += gain @ (measured - s.selection_mean)
            remainder = np.eye(len(measured)) - gain
            s.selection_covariance = remainder @ prior @ remainder.T + gain @ noise @ gain.T
        content, _ = self._action(action)
        for g in np.flatnonzero(n > 0):
            prior_var = s.mean_posterior_variance[g]
            if prior_var == 0:
                continue
            centers = (
                content[g]
                + action.attention_logit
                + s.means[g, :, None]
                + np.sqrt(2 * s.heterogeneity_variance[g, :, None]) * self._nodes
            )

            def probability(shift, centers=centers, g=g):
                z = expit(centers + shift)
                return float(s.mixture_weights[g] @ (z @ self._weights)), float(
                    s.mixture_weights[g] @ ((z * (1 - z)) @ self._weights)
                )

            def objective(shift, g=g, prior_var=prior_var, probability=probability):
                p, _ = probability(shift)
                p = np.clip(p, 1e-12, 1 - 1e-12)
                return (
                    -(k[g] * np.log(p) + (n[g] - k[g]) * np.log1p(-p)) + 0.5 * shift**2 / prior_var
                )

            fitted = minimize_scalar(
                objective, bounds=(-12.0, 12.0), method="bounded", options={"xatol": 1e-8}
            )
            if not fitted.success:
                raise RuntimeError("group response calibration failed")
            p, derivative = probability(fitted.x)
            s.means[g] += fitted.x
            # Expected-information Gaussian approximation, distinct from heterogeneity.
            s.mean_posterior_variance[g] = 1 / (
                1 / prior_var + n[g] * derivative**2 / max(p * (1 - p), 1e-12)
            )

    def _feedback(self, prediction, n, k, rng=None):
        s = self._state
        c = self.config
        total = int(n.sum())
        if total:
            surprise = float(k.sum() / total - self._reference_rate)
            s.momentum = c.momentum_retention * s.momentum + (1 - c.momentum_retention) * surprise
        else:
            s.momentum *= c.momentum_retention
        dose = n / np.maximum(c.population_size * s.population_weights, 1.0)
        s.fatigue = c.fatigue_retention * s.fatigue + (1 - c.fatigue_retention) * (-np.expm1(-dose))
        rho = c.response_retention
        s.means = (
            rho * s.means
            + (1 - rho) * self._prior_means
            + c.social_gain * s.momentum
            - c.fatigue_gain * s.fatigue[:, None]
        )
        s.heterogeneity_variance = (
            rho**2 * s.heterogeneity_variance + (1 - rho**2) * self._prior_variance
        )
        centered = prediction.component_probabilities - prediction.group_probabilities[:, None]
        evolved = softmax(np.log(s.mixture_weights) + c.mixture_gain * centered, axis=1)
        s.mixture_weights = (1 - c.mixture_refresh) * evolved + c.mixture_refresh * self._prior_pi
        if rng is not None and self._conditioned_parameters:
            s.means += rng.normal(size=len(s.fatigue))[:, None] * np.sqrt(
                c.response_process_variance
            )
            s.log_arrival_offset += float(rng.normal() * np.sqrt(c.arrival_process_variance))
            if c.selection_process_variance:
                s.selection_mean += rng.normal(size=len(s.selection_mean)) * np.sqrt(
                    c.selection_process_variance
                )
        else:
            s.mean_posterior_variance = (
                rho**2 * s.mean_posterior_variance + c.response_process_variance
            )
            s.arrival_posterior_variance += c.arrival_process_variance
            s.selection_covariance += np.eye(len(s.selection_mean)) * c.selection_process_variance
        s.cumulative_exposures += total
        s.cumulative_responses += int(k.sum())
        s.day += 1

    def observe(self, action, observation):
        """真实观测校正真实世界；失败更新保持原状态。"""
        if self._simulation or observation.provenance != "real":
            raise ValueError("real feedback belongs only to the real observation state")
        n, k = self._check_observation(observation)
        prediction = self.predict(action)
        before = deepcopy(self._state)
        try:
            self._assimilate(action, prediction, n, k)
            self._feedback(prediction, n, k)
        except Exception:
            self._state = before
            raise
        return prediction

    def simulate_step(self, action, rng, materialize_agents=True):
        prediction, observation, batch = self._sample(action, rng, materialize_agents)
        self._feedback(prediction, observation.exposures, observation.responses, rng)
        return prediction, observation, batch

    def rollout(self, actions, seed=0, materialize_agents=True):
        """冻结真实状态后执行完整模拟路径；返回的分支可继续分叉。"""
        rng = np.random.default_rng(seed)
        branch = self.branch(rng)
        records = []
        for action in actions:
            predicted, observed, _ = branch.simulate_step(action, rng, materialize_agents)
            records.append(
                {
                    "day": observed.day,
                    "action": {
                        "content_logits": np.asarray(action.content_logits).tolist(),
                        "targeting_logits": np.asarray(action.targeting_logits).tolist(),
                        "exposure_effort": action.exposure_effort,
                        "platform_log_supply": action.platform_log_supply,
                        "attention_logit": action.attention_logit,
                    },
                    "predicted_exposures": predicted.expected_exposures,
                    "predicted_responses": predicted.expected_responses,
                    "exposures": int(observed.exposures.sum()),
                    "responses": int(observed.responses.sum()),
                    "momentum": branch._state.momentum,
                    "fatigue": branch._state.fatigue.tolist(),
                    "means": branch._state.means.tolist(),
                    "mixture_weights": branch._state.mixture_weights.tolist(),
                }
            )
        return branch, records

    def set_population_observation(self, day, weights):
        """同源人口画像仅更新人口存量；曝光占比不替代人口画像。"""
        q = np.asarray(weights, float)
        if (
            self._simulation
            or day != self._state.day
            or q.shape != self._state.population_weights.shape
            or not np.isfinite(q).all()
            or np.any(q < 0)
            or q.sum() <= 0
        ):
            raise ValueError("invalid dated real population observation")
        self._state.population_weights = q / q.sum()

    def forecast_distribution(self, actions, trajectories=128, seed=0, coverage=0.9):
        """通过参数后验与闭环路径积分预测区间；计数折叠避免逐曝光内存。"""
        if not isinstance(trajectories, int) or trajectories < 2 or not 0 < coverage < 1:
            raise ValueError("at least two trajectories and a valid coverage required")
        actions = list(actions)
        if not actions:
            raise ValueError("forecast actions required")
        for action in actions:
            self._action(action)
        paths = []
        for child in np.random.SeedSequence(seed).spawn(trajectories):
            rng = np.random.default_rng(child)
            branch = self.branch(rng)
            counts = []
            for action in actions:
                _, obs, _ = branch.simulate_step(action, rng, materialize_agents=False)
                counts.append([int(obs.exposures.sum()), int(obs.responses.sum())])
            paths.append(counts)
        values = np.asarray(paths, float)
        alpha = (1 - coverage) / 2
        return {
            "start_day": self._state.day,
            "trajectories": trajectories,
            "coverage": coverage,
            "mean": values.mean(0).tolist(),
            "lower": np.quantile(values, alpha, axis=0).tolist(),
            "upper": np.quantile(values, 1 - alpha, axis=0).tolist(),
            "cumulative_mean": values.sum(1).mean(0).tolist(),
            "cumulative_lower": np.quantile(values.sum(1), alpha, axis=0).tolist(),
            "cumulative_upper": np.quantile(values.sum(1), 1 - alpha, axis=0).tolist(),
        }

    def to_dict(self):
        def arrays(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: arrays(v) for k, v in value.items()}
            return value

        return arrays(
            {
                "format": "population-world-loop",
                "version": 2,
                "config": asdict(self.config),
                "state": asdict(self._state),
                "prior_means": self._prior_means,
                "prior_pi": self._prior_pi,
                "prior_variance": self._prior_variance,
                "reference_rate": self._reference_rate,
                "simulation": self._simulation,
                "conditioned_parameters": self._conditioned_parameters,
            }
        )

    @classmethod
    def from_dict(cls, data):
        if data.get("format") != "population-world-loop" or data.get("version") not in (1, 2):
            raise ValueError("unsupported population model format")
        raw = data["state"]
        config = LoopConfig(**data["config"])
        model = cls(
            raw["population_weights"],
            data["prior_pi"],
            data["prior_means"],
            data["prior_variance"],
            config,
        )
        state = dict(raw)
        if data["version"] == 1:
            state["selection_mean"] = model._state.selection_mean
            state["selection_covariance"] = model._state.selection_covariance
        names = [
            "population_weights",
            "mixture_weights",
            "means",
            "heterogeneity_variance",
            "mean_posterior_variance",
            "fatigue",
            "selection_mean",
            "selection_covariance",
        ]
        for name in names:
            state[name] = np.asarray(state[name], float)
            if (
                state[name].shape != getattr(model._state, name).shape
                or not np.isfinite(state[name]).all()
            ):
                raise ValueError("invalid saved population state")
        covariance = state["selection_covariance"]
        if not np.allclose(covariance, covariance.T) or (
            covariance.size and np.linalg.eigvalsh(covariance).min() < -1e-10
        ):
            raise ValueError("invalid selection covariance")
        for name in [
            "population_weights",
            "mixture_weights",
            "heterogeneity_variance",
            "mean_posterior_variance",
            "fatigue",
        ]:
            if np.any(state[name] < 0):
                raise ValueError("negative probability or variance in snapshot")
        if (
            not np.isclose(state["population_weights"].sum(), 1.0)
            or not np.allclose(state["mixture_weights"].sum(1), 1.0)
            or np.any(state["mixture_weights"] <= 0)
            or np.any(state["fatigue"] > 1)
        ):
            raise ValueError("invalid snapshot probability normalization")
        for name in ["day", "cumulative_exposures", "cumulative_responses"]:
            if not isinstance(state[name], int) or isinstance(state[name], bool) or state[name] < 0:
                raise ValueError("invalid snapshot counts")
        if state["cumulative_responses"] > state["cumulative_exposures"]:
            raise ValueError("snapshot responses exceed exposures")
        for name in ["momentum", "log_arrival_offset", "arrival_posterior_variance"]:
            if not np.isfinite(state[name]):
                raise ValueError("invalid snapshot scalar")
        if abs(state["momentum"]) > 1 or state["arrival_posterior_variance"] < 0:
            raise ValueError("invalid snapshot state bounds")
        if (
            not isinstance(data["simulation"], bool)
            or not isinstance(data["conditioned_parameters"], bool)
            or (data["conditioned_parameters"] and not data["simulation"])
        ):
            raise ValueError("invalid snapshot branch flags")
        if not np.isfinite(data["reference_rate"]) or not 0 <= data["reference_rate"] <= 1:
            raise ValueError("invalid reference response rate")
        model._state = LoopState(**state)
        model._reference_rate = float(data["reference_rate"])
        model._simulation = data["simulation"]
        model._conditioned_parameters = data["conditioned_parameters"]
        return model

    def save(self, path):
        import os
        import tempfile

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = (
            tempfile.NamedTemporaryFile(  # noqa: SIM115 - entered below; name needed for cleanup
                mode="w", dir=path.parent, prefix=path.name + ".", delete=False
            )
        )
        try:
            with handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, allow_nan=False, indent=2)
            os.replace(handle.name, path)
        finally:
            if os.path.exists(handle.name):
                os.unlink(handle.name)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

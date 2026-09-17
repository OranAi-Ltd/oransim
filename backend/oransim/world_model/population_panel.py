"""Research G–C–P panel with joint actions and persistent, timestamped users.

Campaigns share a Gaussian log-arrival filter. The common coordinate is defined
as the arithmetic mean; campaign contrasts sum to zero. It describes the sampled
pool, not an identified platform intervention. Private snapshots include users.
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .population_behavior import BehaviorBatch, JointBehaviorModel
from .population_loop import PopulationWorldLoop


class UserMemory:
    def __init__(self, heads, half_life_seconds=86400.0):
        if not np.isfinite(half_life_seconds) or half_life_seconds <= 0:
            raise ValueError("positive memory half-life required")
        self.heads = int(heads)
        self.half_life_seconds = float(half_life_seconds)
        self.users = {}
        self.pairs = {}

    def _at(self, record, timestamp):
        if record is None:
            return 0.0, np.zeros(self.heads), 0.0
        last, count, outcomes = record
        if timestamp < last:
            raise ValueError("user memory cannot travel backwards")
        factor = 2 ** (-(timestamp - last) / self.half_life_seconds)
        return count * factor, np.asarray(outcomes) * factor, (timestamp - last) / 86400.0

    def features(self, user, campaign, timestamp):
        if not np.isfinite(timestamp):
            raise ValueError("finite event timestamp required")
        count, outcomes, gap = self._at(self.users.get(str(user)), timestamp)
        pair, _, _ = self._at(self.pairs.get((str(user), str(campaign))), timestamp)
        # Features contain only earlier events; unknown users have exact zero memory.
        return np.r_[np.log1p(count), np.log1p(pair), np.log1p(gap), outcomes / (count + 2)]

    def observe(self, user, campaign, timestamp, outcomes):
        y = np.asarray(outcomes, float)
        if y.shape != (self.heads,) or not np.isfinite(y).all() or np.any((y != 0) & (y != 1)):
            raise ValueError("invalid memory actions")
        self.features(user, campaign, timestamp)
        for collection, key in [(self.users, str(user)), (self.pairs, (str(user), str(campaign)))]:
            count, previous, _ = self._at(collection.get(key), timestamp)
            collection[key] = [float(timestamp), count + 1, (previous + y).tolist()]

    def complete(self, user, campaign, timestamp, outcomes):
        """Release an earlier exposure's outcomes at the observation boundary."""
        y = np.asarray(outcomes, float)
        for collection, key in [(self.users, str(user)), (self.pairs, (str(user), str(campaign)))]:
            last, count, previous = collection[key]
            if timestamp > last:
                raise ValueError("complete a recorded exposure only")
            updated = np.asarray(previous) + y * 2 ** (-(last - timestamp) / self.half_life_seconds)
            if np.any(updated > count + 1e-8):
                raise ValueError("outcomes completed more than once")
            collection[key] = [last, count, updated.tolist()]

    def to_dict(self):
        return {
            "heads": self.heads,
            "half_life_seconds": self.half_life_seconds,
            "users": self.users,
            "pairs": [[*key, value] for key, value in self.pairs.items()],
        }

    @classmethod
    def from_dict(cls, value):
        memory = cls(value["heads"], value["half_life_seconds"])
        memory.users = deepcopy(value["users"])
        memory.pairs = {(u, c): record for u, c, record in value["pairs"]}
        for record in [*memory.users.values(), *memory.pairs.values()]:
            t, n, y = record
            if (
                not np.isfinite([t, n, *y]).all()
                or n < 0
                or len(y) != memory.heads
                or np.any(np.asarray(y) < 0)
                or np.any(np.asarray(y) > n + 1e-8)
            ):
                raise ValueError("invalid saved memory")
        return memory


class SharedArrivalFilter:
    def __init__(self, campaigns, shared_variance=0.02, individual_variance=0.02):
        if campaigns < 1:
            raise ValueError("campaigns required")
        self.mean = np.zeros(campaigns)
        self.covariance = np.eye(campaigns) * 0.1
        self.shared_variance = float(shared_variance)
        self.individual_variance = float(individual_variance)
        self._noise()

    def _noise(self):
        if (
            min(self.shared_variance, self.individual_variance) < 0
            or not np.isfinite([self.shared_variance, self.individual_variance]).all()
        ):
            raise ValueError("invalid environment variance")
        j = len(self.mean)
        return self.individual_variance * np.eye(j) + self.shared_variance * np.ones((j, j))

    @property
    def common(self):
        return float(self.mean.mean())

    def observe(self, counts, scales, dispersion):
        counts = np.asarray(counts, float)
        scales = np.asarray(scales, float)
        r = np.asarray(dispersion, float)
        if (
            counts.shape != self.mean.shape
            or scales.shape != counts.shape
            or np.any(counts < 0)
            or np.any(scales < 0)
            or not np.isfinite([*counts, *scales, *r]).all()
        ):
            raise ValueError("invalid panel arrivals")
        active = scales > 0
        if np.any((~active) & (counts > 0)):
            raise ValueError("observed arrivals conflict with zero effort")
        indices = np.flatnonzero(active)
        if len(indices):
            observed = np.log((counts[active] + 0.5) / scales[active])
            noise = np.diag(1 / (counts[active] + 0.5) + 1 / r[active])
            covariance = self.covariance[np.ix_(indices, indices)] + noise
            gain = np.linalg.solve(covariance, self.covariance[indices, :]).T
            self.mean += gain @ (observed - self.mean[indices])
            self.covariance -= gain @ self.covariance[indices, :]
            self.covariance = (self.covariance + self.covariance.T) / 2
        self.covariance += self._noise()

    def fit(self, counts, scales, dispersion, shared=True):
        """Fit process noise by chronological Gaussian log-count innovations.

        This is a working likelihood for transformed counts; binary response
        likelihoods and NB count scores are reported separately by the runner.
        """
        counts = np.asarray(counts, float)
        scales = np.asarray(scales, float)
        if counts.ndim != 2 or counts.shape != scales.shape or len(counts) < 3:
            raise ValueError("three aligned panel days required")
        start = deepcopy(self)

        def objective(theta):
            f = deepcopy(start)
            f.individual_variance = float(np.exp(theta[0]))
            f.shared_variance = float(np.exp(theta[1])) if shared else 0.0
            score = 0.0
            for n, base in zip(counts, scales, strict=False):
                # Predictable observation covariance uses prior expected counts.
                expected = base * np.exp(np.clip(f.mean, -8, 8))
                v = f.covariance + np.diag(1 / (expected + 0.5) + 1 / dispersion)
                error = np.log((n + 0.5) / base) - f.mean
                _, logdet = np.linalg.slogdet(v)
                score += 0.5 * (
                    logdet + error @ np.linalg.solve(v, error) + len(n) * np.log(2 * np.pi)
                )
                f.observe(n, base, dispersion)
            return score / len(counts)

        theta = np.log(
            [max(self.individual_variance, 1e-5)]
            + ([max(self.shared_variance, 1e-5)] if shared else [])
        )
        solved = minimize(
            objective, theta, method="L-BFGS-B", bounds=[(np.log(1e-5), np.log(1.0))] * len(theta)
        )
        before = objective(theta)
        accepted = bool(np.isfinite(solved.fun) and solved.fun <= before)
        if accepted:
            self.individual_variance = float(np.exp(solved.x[0]))
            self.shared_variance = float(np.exp(solved.x[1])) if shared else 0.0
        elif not shared:
            self.shared_variance = 0.0
        return {
            "working_nll_before": before,
            "working_nll_after": float(solved.fun),
            "accepted": accepted,
            "optimizer_success": bool(solved.success),
            "shared_variance": self.shared_variance,
            "individual_variance": self.individual_variance,
        }

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "shared_variance": self.shared_variance,
            "individual_variance": self.individual_variance,
        }

    @classmethod
    def from_dict(cls, value):
        f = cls(len(value["mean"]), value["shared_variance"], value["individual_variance"])
        f.mean = np.asarray(value["mean"], float)
        f.covariance = np.asarray(value["covariance"], float)
        if (
            f.covariance.shape != (len(f.mean), len(f.mean))
            or not np.isfinite(f.mean).all()
            or not np.isfinite(f.covariance).all()
            or not np.allclose(f.covariance, f.covariance.T)
            or np.linalg.eigvalsh(f.covariance).min() < -1e-8
        ):
            raise ValueError("invalid shared Gaussian state")
        return f


@dataclass(frozen=True)
class PanelEvent:
    timestamp: float
    user: str
    campaign: str
    group: int
    outcomes: tuple
    content: float = 0.0  # known logit shift, not a future outcome


class PopulationWorldPanel:
    def __init__(self, worlds, heads, rates=None, memory=True, dependent=True, shared=True):
        if not worlds or len({w.state.day for w in worlds.values()}) != 1:
            raise ValueError("synchronized campaign worlds required")
        if any(w._simulation for w in worlds.values()):
            raise ValueError("initialize a real panel from real worlds")
        if len({w.state.means.shape for w in worlds.values()}) != 1:
            raise ValueError("campaigns must use the same group and mixture schema")
        self.worlds = deepcopy(worlds)
        self.campaigns = list(worlds)
        self.heads = list(heads)
        self.memory = UserMemory(len(heads))
        self.memory_enabled = bool(memory)
        self.shared_enabled = bool(shared)
        names = [
            "common_supply",
            "campaign_momentum",
            "group_fatigue",
            "log_exposures",
            "log_campaign_exposures",
            "log_gap_days",
        ] + ["memory_" + h for h in heads]
        self.behavior = JointBehaviorModel(heads, names, rates, dependent)
        self.arrivals = SharedArrivalFilter(len(worlds), shared_variance=0.02 if shared else 0.0)
        self.users = {}
        self.last_timestamp = None
        self._simulation = False
        self._nodes, self._weights = np.polynomial.hermite.hermgauss(12)
        self._weights /= np.sqrt(np.pi)
        self._sync_arrivals()

    @property
    def day(self):
        return next(iter(self.worlds.values())).state.day

    def _sync_arrivals(self):
        for j, key in enumerate(self.campaigns):
            self.worlds[key]._state.log_arrival_offset = float(self.arrivals.mean[j])
            self.worlds[key]._state.arrival_posterior_variance = (
                0.0 if self._simulation else float(self.arrivals.covariance[j, j])
            )

    def _feature(self, event):
        s = self.worlds[event.campaign]._state
        history = self.memory.features(event.user, event.campaign, event.timestamp)
        if not self.memory_enabled:
            history = np.zeros_like(history)
        return np.r_[
            self.arrivals.common if self.shared_enabled else 0.0,
            s.momentum,
            s.fatigue[event.group],
            history,
        ]

    def _latent(self, campaign, group, attention=0.0):
        s = self.worlds[campaign]._state
        variance = s.heterogeneity_variance[group] + s.mean_posterior_variance[group]
        nodes = (
            s.means[group, :, None] + attention + np.sqrt(2 * variance[:, None]) * self._nodes
        ).ravel()
        weights = (s.mixture_weights[group, :, None] * self._weights).ravel()
        return nodes, weights

    def _validate_day(self, events, actions):
        if set(actions) != set(self.campaigns):
            raise ValueError("one action per campaign required")
        for key, a in actions.items():
            self.worlds[key]._action(a)
        previous = self.last_timestamp
        identities = dict(self.users)
        for e in events:
            if (
                e.campaign not in self.worlds
                or not isinstance(e.user, str)
                or not 0 <= e.group < len(self.worlds[e.campaign]._state.fatigue)
            ):
                raise ValueError("invalid panel event identity")
            if (
                not np.isfinite(e.timestamp)
                or not np.isfinite(e.content)
                or (previous is not None and e.timestamp < previous)
            ):
                raise ValueError("events must be chronologically ordered")
            y = np.asarray(e.outcomes)
            if y.shape != (len(self.heads),) or np.any((y != 0) & (y != 1)):
                raise ValueError("invalid joint actions")
            if e.user in identities and identities[e.user] != e.group:
                raise ValueError("a user has inconsistent group identity")
            identities[e.user] = e.group
            previous = e.timestamp

    def _features_and_remember(self, events, actions):
        x = []
        u = []
        w = []
        y = []
        by_campaign = {key: [] for key in self.campaigns}
        for i, e in enumerate(events):
            action = actions[e.campaign]
            shifted = PanelEvent(
                e.timestamp,
                e.user,
                e.campaign,
                e.group,
                e.outcomes,
                e.content + action.content_logits[e.group],
            )
            x.append(self._feature(shifted))
            nodes, weights = self._latent(
                e.campaign, e.group, action.attention_logit + shifted.content
            )
            u.append(nodes)
            w.append(weights)
            y.append(e.outcomes)
            by_campaign[e.campaign].append(i)
            self.memory.observe(e.user, e.campaign, e.timestamp, np.zeros(len(self.heads)))
            self.users[e.user] = e.group
        if events:
            self.last_timestamp = events[-1].timestamp
        if not events:
            return None, by_campaign
        return BehaviorBatch(np.array(x), np.array(u), np.array(w), np.array(y)), by_campaign

    def observe_day(self, events, actions, score=True, end_timestamp=None):
        """Atomic real update; all campaign/environment predictions precede labels.

        Event memory is sequential within the day. Gaussian population and shared
        environment updates occur after the day's events, with no within-day leak.
        """
        if self._simulation:
            raise ValueError("simulated panel cannot assimilate real evidence")
        events = list(events)
        self._validate_day(events, actions)
        if end_timestamp is not None and (
            not np.isfinite(end_timestamp)
            or (events and end_timestamp < events[-1].timestamp)
            or (self.last_timestamp is not None and end_timestamp < self.last_timestamp)
        ):
            raise ValueError("invalid observation boundary")
        before = deepcopy(self)
        try:
            result = self._observe_day(events, actions, score)
            if end_timestamp is not None:
                self.last_timestamp = float(end_timestamp)
            return result
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(before.__dict__)
            raise

    def _observe_day(self, events, actions, score):
        predictions = {key: world.predict(actions[key]) for key, world in self.worlds.items()}
        common_before = self.arrivals.common
        batch, indices = self._features_and_remember(events, actions)
        metrics = {}
        if batch is not None and score:
            logp = self.behavior.log_probability(batch)
            p = np.clip(self.behavior.marginals(batch), 1e-9, 1 - 1e-9)
            y = batch.outcomes
            metrics = {
                "events": len(events),
                "joint_nll": float(-logp.mean()),
                "head_log_loss": (-(y * np.log(p) + (1 - y) * np.log1p(-p))).mean(0).tolist(),
                "predicted_actions": p.sum(0).tolist(),
                "observed_actions": y.sum(0).astype(int).tolist(),
            }
        counts = []
        scales = []
        dispersion = []
        for key, world in self.worlds.items():
            selected = indices[key]
            a = actions[key]
            prediction = predictions[key]
            s = world._state
            n = np.zeros(len(s.fatigue), int)
            k = n.copy()
            for i in selected:
                n[events[i].group] += 1
                k[events[i].group] += int(events[i].outcomes[0])
            counts.append(int(n.sum()))
            scales.append(
                world.config.base_arrivals
                * a.exposure_effort
                * np.exp(
                    a.platform_log_supply
                    + world.config.momentum_arrival_gain * s.momentum
                    - world.config.novelty_decay * s.day
                )
            )
            dispersion.append(world.config.arrival_dispersion)
            # Update the Gaussian population from the joint head's actual likelihood.
            for group in np.flatnonzero(n):
                prior = float(s.mean_posterior_variance[group])
                if prior <= 0:
                    continue
                rows = [i for i in selected if events[i].group == group]
                local = batch.take(rows)
                # Condition on the unknown shared group offset: integrate intrinsic
                # heterogeneity only, not the same group posterior a second time.
                local.latent_nodes = (
                    (
                        s.means[group, :, None]
                        + a.attention_logit
                        + np.sqrt(2 * s.heterogeneity_variance[group, :, None]) * self._nodes
                    )
                    .reshape(1, -1)
                    .repeat(len(rows), axis=0)
                )
                local.latent_nodes += np.array(
                    [events[i].content + a.content_logits[group] for i in rows]
                )[:, None]

                def loss(shift, local=local, prior=prior):
                    return -self.behavior.log_probability(local, shift).sum() + shift * shift / (
                        2 * prior
                    )

                fit = minimize_scalar(loss, bounds=(-8, 8), method="bounded")
                if not fit.success:
                    raise RuntimeError("joint population assimilation failed")
                step = 1e-3
                curvature = (loss(fit.x + step) - 2 * loss(fit.x) + loss(fit.x - step)) / step**2
                s.means[group] += fit.x
                s.mean_posterior_variance[group] = min(prior, 1 / max(curvature, 1 / prior))
            # Reuse selection assimilation and deterministic feedback; joint head
            # above owns response assimilation, panel owns correlated arrivals.
            variance = s.mean_posterior_variance.copy()
            s.mean_posterior_variance[:] = 0
            s.arrival_posterior_variance = 0
            world._assimilate(a, prediction, n, k)
            s.mean_posterior_variance = variance
            world._feedback(prediction, n, k)
        self.arrivals.observe(counts, scales, np.array(dispersion))
        self._sync_arrivals()
        for e in events:
            self.memory.complete(e.user, e.campaign, e.timestamp, e.outcomes)
        return {
            "day": self.day - 1,
            "common_before": common_before,
            "common_after": self.arrivals.common,
            "campaign_contrasts": (self.arrivals.mean - self.arrivals.common).tolist(),
            "observed_exposures": counts,
            "predicted_exposures": [predictions[c].expected_exposures for c in self.campaigns],
            "metrics": metrics,
        }, batch

    def branch(self, seed=0):
        rng = np.random.default_rng(seed)
        branch = deepcopy(self)
        branch._simulation = True
        branch.arrivals.mean = rng.multivariate_normal(self.arrivals.mean, self.arrivals.covariance)
        branch.arrivals.covariance[:] = 0
        branch.worlds = {key: world.branch(rng) for key, world in self.worlds.items()}
        branch._sync_arrivals()
        return branch

    def simulate_day(self, actions, rng, start_timestamp=None, duration=86400.0, max_events=200000):
        if not self._simulation:
            raise ValueError("simulate only an explicit panel branch")
        self._validate_day([], actions)
        if not self.users:
            raise ValueError("supply an observed user pool before persistent-user simulation")
        start = (
            (self.last_timestamp + 1 if self.last_timestamp is not None else 0.0)
            if start_timestamp is None
            else float(start_timestamp)
        )
        if (
            not np.isfinite([start, duration]).all()
            or duration <= 0
            or (self.last_timestamp is not None and start < self.last_timestamp)
        ):
            raise ValueError("invalid simulation clock")
        predictions = {key: w.predict(actions[key]) for key, w in self.worlds.items()}
        schedule = []
        pools = {
            g: [u for u, group in self.users.items() if group == g]
            for g in set(self.users.values())
        }
        for key, world in self.worlds.items():
            p = predictions[key]
            r = world.config.arrival_dispersion
            total = int(rng.poisson(rng.gamma(r, p.expected_exposures / r)))
            if len(schedule) + total > max_events:
                raise ValueError(
                    "requested individual simulation exceeds max_events; use a smaller exposure plan or raise the explicit budget"
                )
            groups = rng.choice(len(p.exposure_weights), size=total, p=p.exposure_weights)
            for t, g in zip(rng.uniform(start, start + duration, total), groups, strict=False):
                if int(g) not in pools:
                    # Cold groups get persistent synthetic identities, never real IDs.
                    user = f"synthetic:{int(g)}"
                    pools[int(g)] = [user]
                user = pools[int(g)][rng.integers(len(pools[int(g)]))]
                schedule.append((float(t), key, user, int(g)))
        schedule.sort()
        counts = {key: np.zeros(len(w._state.fatigue), int) for key, w in self.worlds.items()}
        responses = {
            key: np.zeros((len(w._state.fatigue), len(self.heads)), int)
            for key, w in self.worlds.items()
        }
        completed = []
        for t, key, user, g in schedule:
            world = self.worlds[key]
            s = world._state
            a = actions[key]
            component = rng.choice(s.means.shape[1], p=s.mixture_weights[g])
            latent = (
                rng.normal(s.means[g, component], np.sqrt(s.heterogeneity_variance[g, component]))
                + a.attention_logit
                + a.content_logits[g]
            )
            e = PanelEvent(
                t, user, key, g, tuple([0] * len(self.heads)), float(a.content_logits[g])
            )
            y = self.behavior.sample(self._feature(e), latent, rng)
            self.memory.observe(user, key, t, np.zeros(len(self.heads)))
            completed.append((user, key, t, y))
            self.users[user] = g
            counts[key][g] += 1
            responses[key][g] += y
        for user, key, t, y in completed:
            self.memory.complete(user, key, t, y)
        for key, world in self.worlds.items():
            world._feedback(predictions[key], counts[key], responses[key][:, 0], rng)
        self.arrivals.mean += rng.multivariate_normal(
            np.zeros(len(self.campaigns)), self.arrivals._noise()
        )
        self.last_timestamp = start + duration
        self._sync_arrivals()
        return {
            "day": self.day - 1,
            "exposures": [int(counts[c].sum()) for c in self.campaigns],
            "actions": [responses[c].sum(0).tolist() for c in self.campaigns],
            "common_supply": self.arrivals.common,
        }

    def forecast(self, plans, trajectories=32, seed=0, max_events=200000):
        if not plans or trajectories < 2:
            raise ValueError("plans and at least two trajectories required")
        paths = []
        for child in np.random.SeedSequence(seed).spawn(trajectories):
            rng = np.random.default_rng(child)
            branch = self.branch(rng)
            rows = []
            for actions in plans:
                result = branch.simulate_day(actions, rng, max_events=max_events)
                rows.append(np.column_stack([result["exposures"], result["actions"]]))
            paths.append(rows)
        values = np.asarray(paths, float)
        return {
            "start_day": self.day,
            "campaigns": self.campaigns,
            "columns": ["exposures", *self.heads],
            "trajectories": trajectories,
            "mean": values.mean(0).tolist(),
            "lower_90": np.quantile(values, 0.05, axis=0).tolist(),
            "upper_90": np.quantile(values, 0.95, axis=0).tolist(),
        }

    def to_dict(self):
        return {
            "format": "population-world-panel",
            "version": 2,
            "campaigns": self.campaigns,
            "heads": self.heads,
            "worlds": {key: w.to_dict() for key, w in self.worlds.items()},
            "memory": self.memory.to_dict(),
            "behavior": self.behavior.to_dict(),
            "arrivals": self.arrivals.to_dict(),
            "users": self.users,
            "memory_enabled": self.memory_enabled,
            "shared_enabled": self.shared_enabled,
            "simulation": self._simulation,
            "last_timestamp": self.last_timestamp,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = (
            tempfile.NamedTemporaryFile(  # noqa: SIM115 - entered below; name needed for cleanup
                mode="w", dir=path.parent, delete=False
            )
        )
        try:
            with temporary:
                json.dump(self.to_dict(), temporary, allow_nan=False)
            os.replace(temporary.name, path)  # NamedTemporaryFile creates a private 0600 file.
        finally:
            if os.path.exists(temporary.name):
                os.unlink(temporary.name)

    @classmethod
    def from_dict(cls, value):
        if value.get("format") != "population-world-panel" or value.get("version") != 2:
            raise ValueError("unsupported panel snapshot")
        panel = cls.__new__(cls)
        panel.campaigns = list(value["campaigns"])
        panel.heads = list(value["heads"])
        panel.worlds = {
            key: PopulationWorldLoop.from_dict(value["worlds"][key]) for key in panel.campaigns
        }
        panel.memory = UserMemory.from_dict(value["memory"])
        panel.behavior = JointBehaviorModel.from_dict(value["behavior"])
        panel.arrivals = SharedArrivalFilter.from_dict(value["arrivals"])
        panel.users = deepcopy(value["users"])
        panel.memory_enabled = value["memory_enabled"]
        panel.shared_enabled = value["shared_enabled"]
        panel._simulation = value["simulation"]
        panel.last_timestamp = value["last_timestamp"]
        if (
            len({w.state.day for w in panel.worlds.values()}) != 1
            or len(panel.arrivals.mean) != len(panel.campaigns)
            or panel.heads != panel.behavior.heads
            or panel.memory.heads != len(panel.heads)
        ):
            raise ValueError("inconsistent panel snapshot")
        if any(w._simulation != panel._simulation for w in panel.worlds.values()):
            raise ValueError("inconsistent simulation provenance")
        if panel.last_timestamp is not None and not np.isfinite(panel.last_timestamp):
            raise ValueError("invalid panel time")
        panel._nodes, panel._weights = np.polynomial.hermite.hermgauss(12)
        panel._weights /= np.sqrt(np.pi)
        panel._sync_arrivals()
        return panel

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

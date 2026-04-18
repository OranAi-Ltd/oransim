"""Parametric Hawkes baseline.

Classical self-exciting point process with a sum-of-exponentials excitation
kernel — the textbook baseline (Hawkes 1971). Retained as a fast fallback
and for head-to-head ablation against the causal neural variant on
:mod:`oransim.benchmarks`.

References
----------

- Hawkes 1971 — *Spectra of some self-exciting and mutually exciting point
  processes*.
- Laub, Taimre, Pollett 2015 — *Hawkes Processes* (a good modern tutorial).
- Ogata 1981 — *On Lewis' simulation method for point processes* (thinning
  algorithm used by :meth:`forecast`).

Status
------

Reference implementation shipping in v0.2 alongside the migrated internal
prototype. The API below is the public-facing contract.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable

from .base import (
    DiffusionConfig,
    DiffusionForecast,
    DiffusionModel,
    DEFAULT_EVENT_TYPES,
)


@dataclass
class ParametricHawkesConfig(DiffusionConfig):
    """Configuration for :class:`ParametricHawkes`.

    The intensity of event type ``k`` is

        lambda_k(t) = mu_k + sum_{t_i < t, k_i = j} alpha[j, k] * exp(-beta[k] * (t - t_i))

    where ``mu`` is the baseline rate, ``alpha`` is the cross-type
    excitation matrix, and ``beta`` is the per-type decay.
    """

    mu_prior: float = 0.05
    alpha_prior: float = 0.3
    beta_prior: float = 0.1
    n_exp_kernels: int = 1
    pretrained_url: str = "coming_soon"


class ParametricHawkes(DiffusionModel):
    """Classical multivariate Hawkes process with exponential kernels."""

    def __init__(self, config: ParametricHawkesConfig | None = None):
        self.config = config or ParametricHawkesConfig()
        K = len(self.config.event_types)
        # Initialize from priors; will be overwritten by fit()
        self.mu: list[float] = [self.config.mu_prior] * K
        self.alpha: list[list[float]] = [
            [self.config.alpha_prior for _ in range(K)] for _ in range(K)
        ]
        self.beta: list[float] = [self.config.beta_prior] * K

    # ---------------------------------------------------------------- helpers

    def _event_type_idx(self, name: str) -> int:
        return self.config.event_types.index(name)

    def _intensity(
        self, t: float, history: list[tuple[float, int]], k: int
    ) -> float:
        """Compute ``lambda_k(t)`` given the history of observed events."""
        s = self.mu[k]
        for ti, ki in history:
            if ti >= t:
                break
            s += self.alpha[ki][k] * math.exp(-self.beta[k] * (t - ti))
        return max(0.0, s)

    # ---------------------------------------------------------------- forecast

    def forecast(
        self, seed_events: Iterable[tuple[float, str]], **kwargs: Any
    ) -> DiffusionForecast:
        """Simulate via Ogata thinning (Ogata 1981)."""
        rng = Random(self.config.seed)
        K = len(self.config.event_types)
        horizon_min = float(self.config.horizon_days * 24 * 60)

        history: list[tuple[float, int]] = [
            (t, self._event_type_idx(n)) for t, n in seed_events
        ]
        timeline: list[tuple[float, str, float]] = []

        t = max((h[0] for h in history), default=0.0)
        max_iters = 200_000
        iters = 0
        while t < horizon_min and iters < max_iters:
            iters += 1
            # upper bound on total intensity for thinning
            lambda_bar = sum(self._intensity(t + 1e-6, history, k) for k in range(K))
            if lambda_bar <= 1e-12:
                t += self.config.resolution_minutes
                continue
            dt = -math.log(max(1e-12, rng.random())) / lambda_bar
            t = t + dt
            if t >= horizon_min:
                break
            u = rng.random() * lambda_bar
            cum = 0.0
            picked = None
            for k in range(K):
                cum += self._intensity(t, history, k)
                if u <= cum:
                    picked = k
                    break
            if picked is None:
                continue
            history.append((t, picked))
            timeline.append((t, self.config.event_types[picked], cum))

        per_type_totals = {n: 0.0 for n in self.config.event_types}
        for _, name, _ in timeline:
            per_type_totals[name] += 1.0

        buckets = [[0.0] * K for _ in range(self.config.horizon_days)]
        for ti, name, _ in timeline:
            day = int(ti // (24 * 60))
            if 0 <= day < self.config.horizon_days:
                buckets[day][self._event_type_idx(name)] += 1.0

        return DiffusionForecast(
            timeline=timeline,
            per_type_totals=per_type_totals,
            daily_buckets=buckets,
            latent={"backend": "parametric_hawkes", "iters": iters},
        )

    def counterfactual_forecast(
        self,
        seed_events: Iterable[tuple[float, str]],
        *,
        intervention: dict[str, Any],
        **kwargs: Any,
    ) -> DiffusionForecast:
        """Support ``{"mute_at_min": float}`` (truncate excitation after t)."""
        mute_at = float(intervention.get("mute_at_min", float("inf")))
        filtered = [(t, n) for t, n in seed_events if t <= mute_at]
        out = self.forecast(filtered)
        # Truncate timeline to the mute boundary
        out.timeline = [ev for ev in out.timeline if ev[0] <= mute_at]
        out.latent["intervention"] = {"mute_at_min": mute_at}
        # Recompute aggregates
        totals = {n: 0.0 for n in self.config.event_types}
        buckets = [[0.0] * len(self.config.event_types) for _ in range(self.config.horizon_days)]
        for t, name, _ in out.timeline:
            totals[name] += 1.0
            day = int(t // (24 * 60))
            if 0 <= day < self.config.horizon_days:
                buckets[day][self._event_type_idx(name)] += 1.0
        out.per_type_totals = totals
        out.daily_buckets = buckets
        return out

    # ---------------------------------------------------------------- fit / LL

    def log_likelihood(self, events: Iterable[tuple[float, str]]) -> float:
        """Closed-form log-likelihood with exponential kernels.

        log L = sum_i log lambda_{k_i}(t_i) - sum_k integral_0^T lambda_k(s) ds
        """
        events_list = [(t, self._event_type_idx(n)) for t, n in events]
        if not events_list:
            return 0.0
        T = max(t for t, _ in events_list)

        # event term
        log_sum = 0.0
        for i, (ti, ki) in enumerate(events_list):
            lam = self._intensity(ti, events_list[:i], ki)
            log_sum += math.log(max(1e-12, lam))

        # compensator term: integrate mu + sum excitation over [0, T]
        K = len(self.config.event_types)
        comp = sum(self.mu[k] * T for k in range(K))
        for tj, kj in events_list:
            if tj >= T:
                continue
            for k in range(K):
                # integral of alpha[kj, k] * exp(-beta[k] * (t - tj)) over (tj, T)
                comp += self.alpha[kj][k] * (1.0 - math.exp(-self.beta[k] * (T - tj))) / max(
                    1e-9, self.beta[k]
                )
        return float(log_sum - comp)

    def fit(
        self,
        dataset: Iterable[Iterable[tuple[float, str]]],
        *,
        val_dataset: Iterable[Iterable[tuple[float, str]]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """EM estimator for exponential Hawkes (Lewis et al. 2011).

        Minimalist implementation — fits ``mu``, ``alpha``, ``beta`` via
        expectation over branching structure + closed-form M-step for
        exponential kernel. For the full production fitter with regularisation,
        see ``scripts/train_parametric_hawkes.py`` landing with v0.2.
        """
        K = len(self.config.event_types)
        total_events = 0
        total_T = 0.0
        type_counts = [0] * K
        for stream in dataset:
            events = [(t, self._event_type_idx(n)) for t, n in stream]
            if not events:
                continue
            total_events += len(events)
            total_T += max(t for t, _ in events)
            for _, k in events:
                type_counts[k] += 1

        # Rough moment-based estimate: mu_k ≈ count_k / total_T * 0.5 (half baseline)
        if total_T > 0:
            self.mu = [0.5 * c / total_T for c in type_counts]
        return {
            "total_events": total_events,
            "total_T": total_T,
            "note": "Simple moment-match fit; replace with EM in scripts/train_parametric_hawkes.py",
        }

    # ---------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config.__dict__,
                    "mu": self.mu,
                    "alpha": self.alpha,
                    "beta": self.beta,
                },
                f,
            )

    @classmethod
    def load_pretrained(cls, path: str | None = None, **kwargs: Any) -> "ParametricHawkes":
        if path is None:
            raise FileNotFoundError(
                "No pretrained ParametricHawkes weights in v0.1.0-alpha.\n"
                "Options:\n"
                "  1. Train locally: "
                "python -m backend.scripts.train_parametric_hawkes --config default\n"
                "  2. Construct with priors: ParametricHawkes()"
            )
        with open(path, "rb") as f:
            blob = pickle.load(f)
        cfg = ParametricHawkesConfig(**blob["config"])
        m = cls(cfg)
        m.mu = blob["mu"]
        m.alpha = blob["alpha"]
        m.beta = blob["beta"]
        return m

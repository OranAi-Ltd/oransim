"""Gaussian logit states observed through grouped Binomial response counts.

This models a response probability conditional on an exposed group. Population
composition is a separate simplex-valued quantity. No causal interpretation of
fatigue, recommendation policy, or the unobserved complete exposure history is
implied by the fitted local trend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(probability) - np.log1p(-probability)


def pooled_prior(
    successes: np.ndarray, exposures: np.ndarray, strength: float = 20.0
) -> np.ndarray:
    """Empirical-Bayes group shrinkage, fitted exclusively on training counts."""
    total = (np.sum(successes) + 0.5) / (np.sum(exposures) + 1.0)
    return (np.asarray(successes) + strength * total) / (np.asarray(exposures) + strength)


@dataclass(frozen=True)
class GaussianResponseConfig:
    level_variance: float = 0.01
    trend_variance: float = 0.0
    trend_damping: float = 0.0
    effective_exposure_cap: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(
            [
                self.level_variance,
                self.trend_variance,
                self.trend_damping,
                self.effective_exposure_cap,
            ]
        ).all():
            raise ValueError("Configuration values must be finite")
        if self.level_variance < 0 or self.trend_variance < 0:
            raise ValueError("Process variances must be nonnegative")
        if not 0 <= self.trend_damping <= 1:
            raise ValueError("Trend damping must lie in [0, 1]")
        if self.effective_exposure_cap < 0:
            raise ValueError("Effective exposure cap must be nonnegative")


class GaussianLogitResponse:
    """Laplace Binomial filter with damped local linear dynamics.

    A zero cap uses the Binomial likelihood unchanged. A positive cap applies a
    power likelihood with n_eff=min(n, cap), retaining the observed proportion.
    This is an explicit robustness option for correlated repeated exposures,
    not an assertion that its uncertainty intervals have calibrated coverage.
    """

    def __init__(self, prior: np.ndarray, config: GaussianResponseConfig):
        prior = np.asarray(prior, dtype=float)
        if prior.ndim != 1 or not len(prior) or not np.isfinite(prior).all():
            raise ValueError("Prior must be a nonempty finite vector")
        if np.any(prior < 0) or np.any(prior > 1):
            raise ValueError("Prior probabilities must lie in [0, 1]")
        self.config = config
        self.mean = np.column_stack((logit(prior), np.zeros(len(prior))))
        self.covariance = np.tile(np.diag([1.0, 0.05]), (len(prior), 1, 1))
        self.transition = np.asarray([[1.0, config.trend_damping], [0.0, config.trend_damping]])
        self.process = np.diag([config.level_variance, config.trend_variance])
        self._nodes, self._weights = np.polynomial.hermite.hermgauss(9)

    def advance(self) -> None:
        self.mean = self.mean @ self.transition.T
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Predict a future day's probability without consuming observations."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        mean, covariance = self.mean.copy(), self.covariance.copy()
        for _ in range(horizon):
            mean = mean @ self.transition.T
            covariance = self.transition @ covariance @ self.transition.T + self.process
        # Integrate sigmoid under the Gaussian, rather than sigmoid(E[logit]).
        logits = mean[:, :1] + np.sqrt(2 * np.maximum(covariance[:, :1, 0], 0)) * self._nodes
        return sigmoid(logits) @ self._weights / np.sqrt(np.pi)

    def observe(self, successes: np.ndarray, exposures: np.ndarray) -> None:
        successes, exposures = np.asarray(successes, dtype=float), np.asarray(
            exposures, dtype=float
        )
        if successes.shape != (len(self.mean),) or exposures.shape != successes.shape:
            raise ValueError("Counts must match the number of groups")
        if not np.all(np.isfinite(successes)) or not np.all(np.isfinite(exposures)):
            raise ValueError("Counts must be finite")
        if np.any(successes < 0) or np.any(exposures < successes):
            raise ValueError("Require 0 <= successes <= exposures")
        if self.config.effective_exposure_cap:
            scale = np.minimum(1.0, self.config.effective_exposure_cap / np.maximum(exposures, 1))
            successes, exposures = successes * scale, exposures * scale
        active = exposures > 0
        if not active.any():
            return
        prior_mean = self.mean[:, 0].copy()
        variance = np.maximum(self.covariance[:, 0, 0], 1e-12)
        mode = prior_mean.copy()
        # Convex negative log posterior: bounded Newton steps handle 0/n and n/n.
        for _ in range(50):
            probability = sigmoid(mode)
            gradient = (mode - prior_mean) / variance + exposures * probability - successes
            precision = 1 / variance + exposures * probability * (1 - probability)
            step = np.clip(gradient / precision, -4.0, 4.0)
            mode -= step
            if np.max(np.abs(step)) < 1e-10:
                break
        column = self.covariance[:, :, 0].copy()
        self.mean += column / variance[:, None] * (mode - prior_mean)[:, None]
        probability = sigmoid(mode)
        information = exposures * probability * (1 - probability)
        reduction = information / (1 + variance * information)
        self.covariance -= reduction[:, None, None] * column[:, :, None] * column[:, None, :]
        self.covariance = (self.covariance + self.covariance.transpose(0, 2, 1)) / 2

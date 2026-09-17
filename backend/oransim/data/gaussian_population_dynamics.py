"""Experimental population forecasts in orthogonal log-ratio coordinates.

The observation covariance is expressed in the same coordinates as the state.
Time is measured in days.  A forecast is a read-only projection; only actual
observations advance the filter.  This module has no production integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np


def ilr_basis(categories: int) -> np.ndarray:
    """Rows are an orthonormal basis of the zero-sum log-composition space."""
    if categories < 2:
        raise ValueError("a composition needs at least two categories")
    basis = np.zeros((categories - 1, categories), dtype=float)
    for row in range(categories - 1):
        scale = math.sqrt((row + 1) * (row + 2))
        basis[row, : row + 1] = 1 / scale
        basis[row, row + 1] = -(row + 1) / scale
    return basis


def close_composition(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("composition must be a vector with at least two entries")
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("composition must have finite nonnegative positive mass")
    # Numerical zero replacement is not a count or an effective sample size.
    values = np.maximum(values / values.sum(), 1e-6)
    return values / values.sum()


def to_ilr(values: np.ndarray) -> np.ndarray:
    values = close_composition(values)
    return ilr_basis(len(values)) @ np.log(values)


def from_ilr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    logits = ilr_basis(len(values) + 1).T @ values
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def difference_noise_variance(histories: list[np.ndarray], floor: float = 1e-5) -> float:
    """Estimate a scalar coordinate-space noise scale from training differences.

    E[||delta z||^2]/(2d) includes both reporting noise and genuine changes.
    It is a regularized working noise scale, not an identified measurement
    error variance.  Pooling makes short individual histories usable and the
    isotropic scale preserves invariance to the chosen orthonormal basis.
    """
    sum_squares = 0.0
    coordinates = 0
    for history in histories:
        history = np.asarray(history, dtype=float)
        if history.ndim != 2 or len(history) < 2:
            continue
        differences = np.diff(history, axis=0)
        sum_squares += float(np.square(differences).sum())
        coordinates += int(differences.size)
    return max(float(floor), sum_squares / (2 * coordinates)) if coordinates else float(floor)


@dataclass(frozen=True)
class DynamicsConfig:
    # Q/R per day.  The slope diffusion uses the same ratio, scaled by 0.05.
    process_ratio: float = 1.0
    observation_scale: float = 1.0
    slope_half_life_days: float | None = None

    def __post_init__(self):
        if self.process_ratio < 0 or not math.isfinite(self.process_ratio):
            raise ValueError("process_ratio must be finite and nonnegative")
        if self.observation_scale <= 0 or not math.isfinite(self.observation_scale):
            raise ValueError("observation_scale must be finite and positive")
        if self.slope_half_life_days is not None and (
            self.slope_half_life_days <= 0 or not math.isfinite(self.slope_half_life_days)
        ):
            raise ValueError("slope half life must be finite and positive")


def _days(later: datetime, earlier: datetime) -> float:
    if later.tzinfo is None or earlier.tzinfo is None:
        raise ValueError("timestamps must be timezone aware")
    days = (later - earlier).total_seconds() / 86400
    if days < 0:
        raise ValueError("forecast/update cannot precede filter state")
    return days


@dataclass(frozen=True)
class LatentForecast:
    mean: np.ndarray
    predictive_variance: float
    at_time: datetime


class GaussianDynamicsFilter:
    """Isotropic multivariate local-level or damped-trend Kalman filter.

    Coordinates share a 2x2 level/slope covariance.  This makes rotations of
    the ILR basis equivalent and avoids a privileged reference category.
    """

    def __init__(
        self,
        first: np.ndarray,
        at_time: datetime,
        noise_variance: float,
        config: DynamicsConfig = DynamicsConfig(),  # noqa: B008 - frozen, immutable configuration
    ):
        _days(at_time, at_time)
        first = np.asarray(first, dtype=float)
        if first.ndim != 1 or not len(first) or not np.isfinite(first).all():
            raise ValueError("latent observations must be finite nonempty vectors")
        if noise_variance <= 0 or not math.isfinite(noise_variance):
            raise ValueError("noise variance must be finite and positive")
        self.config = config
        self.noise = float(noise_variance) * config.observation_scale
        self.mean = np.stack([first.copy(), np.zeros_like(first)])
        slope_variance = (
            self.noise * config.process_ratio * 0.05 if config.slope_half_life_days else 0.0
        )
        self.covariance = np.diag([self.noise, slope_variance])
        self.state_time = at_time

    def _project(self, at_time: datetime) -> tuple[np.ndarray, np.ndarray]:
        dt = _days(at_time, self.state_time)
        if dt == 0:
            return self.mean.copy(), self.covariance.copy()
        q_level = self.noise * self.config.process_ratio
        if self.config.slope_half_life_days is None:
            transition = np.eye(2)
            process = np.diag([q_level * dt, 0.0])
        else:
            # Exact integrated Ornstein-Uhlenbeck transition and covariance.
            rate = math.log(2) / self.config.slope_half_life_days
            decay = math.exp(-rate * dt)
            integrated = -math.expm1(-rate * dt) / rate
            slope_integral = -math.expm1(-2 * rate * dt) / (2 * rate)
            cross = (integrated - slope_integral) / rate
            level_integral = (dt - 2 * integrated + slope_integral) / rate**2
            # Avoid subtractive cancellation for very small intervals.
            if rate * dt < 1e-4:
                level_integral = dt**3 / 3
                cross = dt**2 / 2
            transition = np.array([[1.0, integrated], [0.0, decay]])
            q_slope = q_level * 0.05
            process = np.array(
                [
                    [q_level * dt + q_slope * max(0.0, level_integral), q_slope * cross],
                    [q_slope * cross, q_slope * slope_integral],
                ]
            )
        covariance = transition @ self.covariance @ transition.T + process
        return transition @ self.mean, (covariance + covariance.T) / 2

    def forecast(self, at_time: datetime) -> LatentForecast:
        mean, covariance = self._project(at_time)
        return LatentForecast(mean[0], max(1e-12, float(covariance[0, 0]) + self.noise), at_time)

    def update(self, observation: np.ndarray, at_time: datetime) -> None:
        if _days(at_time, self.state_time) <= 0:
            raise ValueError("new observations must have strictly increasing timestamps")
        observation = np.asarray(observation, dtype=float)
        if observation.shape != self.mean[0].shape or not np.isfinite(observation).all():
            raise ValueError("latent observation has invalid shape or values")
        mean, covariance = self._project(at_time)
        gain = covariance[:, 0] / (covariance[0, 0] + self.noise)
        self.mean = mean + gain[:, None] * (observation - mean[0])[None, :]
        residual = np.eye(2) - np.outer(gain, [1.0, 0.0])
        self.covariance = residual @ covariance @ residual.T + self.noise * np.outer(gain, gain)
        self.state_time = at_time


def anchored_drift(
    times_days: np.ndarray,
    latent: np.ndarray,
    target_day: float,
    window_days: float = 7.0,
    half_life_days: float = 3.0,
    activity_gate: bool = True,
) -> np.ndarray:
    """以最新状态为锚，用持续变化的速度作有界外推。

    径向截断削弱单次画像修订；有变化的区间比例控制趋势可信度。
    单次跳变更新当前水平，多次同向变化才持续影响预测速度。
    """
    times = np.asarray(times_days, dtype=float)
    z = np.asarray(latent, dtype=float)
    if (
        times.ndim != 1
        or z.ndim != 2
        or len(times) != len(z)
        or not len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(z).all()
        or not np.isfinite(target_day)
        or np.any(np.diff(times) <= 0)
        or target_day < times[-1]
        or window_days <= 0
        or half_life_days <= 0
    ):
        raise ValueError("invalid dated latent history or forecast settings")
    keep = times >= times[-1] - window_days
    t, values = times[keep], z[keep]
    if len(t) < 3:
        return z[-1].copy()
    dt = np.diff(t)
    velocity = np.diff(values, axis=0) / dt[:, None]
    norms = np.linalg.norm(velocity, axis=1)
    nonzero = norms > 1e-8
    if not nonzero.any():
        return z[-1].copy()
    cap = 3 * np.median(norms[nonzero])
    clipped = velocity * np.minimum(1.0, cap / np.maximum(norms, 1e-12))[:, None]
    weights = dt / dt.sum()
    drift = weights @ clipped
    coherence = np.linalg.norm(drift) / max(float(weights @ np.linalg.norm(clipped, axis=1)), 1e-12)
    if activity_gate:
        drift *= float(weights @ nonzero) ** 2 * coherence
    rate = np.log(2) / half_life_days
    effective_horizon = -np.expm1(-rate * (target_day - t[-1])) / rate
    return z[-1] + effective_horizon * drift


def calibration_radius(errors: np.ndarray, coverage: float = 0.9) -> float:
    """独立校准样本的有限样本残差分位数；时间相关性另行实测。

    调用方按账号汇总多个预测起点，避免把相关日期当作独立账号。
    """
    scores = np.asarray(errors, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("calibration errors must be finite nonnegative scores")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    rank = min(len(scores), int(math.ceil((len(scores) + 1) * coverage)))
    return float(np.partition(scores, rank - 1)[rank - 1])

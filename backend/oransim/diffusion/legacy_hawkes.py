"""Segment-level sparse Hawkes process for 7-day organic reach curve.

Background (plain words): when a user engages with an ad, they cause secondary
exposures to people in their social neighborhood — friends, followers, feed-
neighbors with similar tastes. Hawkes models this "self-exciting" cascade.

We avoid N×N (10^10 cells for 100k agents) by aggregating to segments:
  segment = (age_bucket × gender × city_tier)  →  6 × 2 × 5 = 60 buckets

Influence matrix A (60×60) is sparse-friendly: homophily makes most mass
land on same-segment or adjacent segments. Exponential decay kernel.

Ogata recurrence:
  S(t+Δt) = S(t) · exp(-β·Δt) + new_impulses
  λ(t)    = μ(t) + branching · A · S(t)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..agents.statistical import OutcomeBatch
from ..data.population import Population
from ..platforms.xhs.world_model_legacy import ImpressionResult

N_AGE = 6
N_GENDER = 2
N_CITY = 5
N_SEG = N_AGE * N_GENDER * N_CITY  # 60


def _seg_id(age_idx: np.ndarray, gender_idx: np.ndarray, city_idx: np.ndarray) -> np.ndarray:
    return age_idx * (N_GENDER * N_CITY) + gender_idx * N_CITY + city_idx


def _seg_decode(s: int):
    a = s // (N_GENDER * N_CITY)
    g = (s // N_CITY) % N_GENDER
    c = s % N_CITY
    return a, g, c


def build_influence_matrix(homophily: float = 3.0, cross_weight: float = 0.15) -> np.ndarray:
    """60×60 row-stochastic-ish influence matrix.
    Diagonal boost (homophily), softened distance-decay to other segments.
    """
    A = np.zeros((N_SEG, N_SEG), dtype=np.float32)
    for i in range(N_SEG):
        ai, gi, ci = _seg_decode(i)
        for j in range(N_SEG):
            aj, gj, cj = _seg_decode(j)
            if i == j:
                w = homophily
            else:
                dist = abs(ai - aj) + (0.8 if gi != gj else 0) + abs(ci - cj) * 0.6
                w = cross_weight * np.exp(-0.5 * dist)
            A[i, j] = w
        A[i] /= A[i].sum()
    return A


@dataclass
class HawkesResult:
    days: int
    dt: float
    t_axis: np.ndarray  # (T,) days
    paid_curve: np.ndarray  # (T,) from budget (exogenous μ)
    organic_curve: np.ndarray  # (T,) secondary from Hawkes
    total_curve: np.ndarray  # (T,)
    seg_curves: np.ndarray  # (N_SEG, T) decomposed
    branching_ratio: float  # effective reproduction number (like R0)
    peak_day: float  # day index of peak organic rate


class HawkesSimulator:
    def __init__(self, population: Population, beta: float = 0.9, branching: float = 0.35):
        """
        beta       : decay rate per day (0.9 → e-fold ≈ 1.1 day)
        branching  : virality coefficient. <1 sub-critical (decays), >1 explodes.
        """
        self.pop = population
        self.beta = beta
        self.branching = branching
        self.A = build_influence_matrix()
        self._seg_cache = _seg_id(population.age_idx, population.gender_idx, population.city_idx)

    def simulate(
        self,
        impression: ImpressionResult,
        outcome: OutcomeBatch,
        days: int = 14,
        dt: float = 0.25,  # 6-hour tick
        paid_daily_fraction: np.ndarray | None = None,
    ) -> HawkesResult:
        """Return paid + organic cumulative + rate curves over `days`."""
        T = int(days / dt) + 1
        t_axis = np.arange(T) * dt

        # ---- 1. Seed (initial engagement per segment) ----
        idx = impression.agent_idx
        if len(idx) == 0:
            zeros = np.zeros(T, dtype=np.float32)
            return HawkesResult(
                days=days,
                dt=dt,
                t_axis=t_axis,
                paid_curve=zeros,
                organic_curve=zeros,
                total_curve=zeros,
                seg_curves=np.zeros((N_SEG, T), dtype=np.float32),
                branching_ratio=self.branching,
                peak_day=0.0,
            )
        segs = self._seg_cache[idx]
        # engagement "mass" per agent = click_prob + 0.5*engage_prob (virality = clicks+shares)
        mass = outcome.click_prob + 0.5 * outcome.engage_prob

        # distribute over paid schedule: campaigns are front-loaded (50% in day 1-2)
        if paid_daily_fraction is None:
            base = np.exp(-0.4 * np.arange(days))
            paid_daily_fraction = base / base.sum()

        # aggregate to segment × time for paid exogenous injection
        mu = np.zeros((N_SEG, T), dtype=np.float32)
        # scatter mass to segment
        seg_mass_total = np.zeros(N_SEG, dtype=np.float32)
        np.add.at(seg_mass_total, segs, mass)

        for d in range(days):
            t_lo = int(d / dt)
            t_hi = min(int((d + 1) / dt), T)
            per_tick = paid_daily_fraction[d] / max(t_hi - t_lo, 1)
            mu[:, t_lo:t_hi] += seg_mass_total[:, None] * per_tick

        # ---- 2. Ogata-recurrent simulation ----
        S = np.zeros(N_SEG, dtype=np.float32)  # accumulated excitation
        organic_per_tick = np.zeros((N_SEG, T), dtype=np.float32)
        decay = np.exp(-self.beta * dt)
        AT = self.A.T.astype(np.float32)  # influence is "j excites i" → use A^T

        for t in range(T):
            # excitation rate contributed now = branching · (A^T @ S)
            lam = self.branching * (AT @ S)
            organic_per_tick[:, t] = lam * dt
            # add new impulses: paid + organic
            new_impulse = mu[:, t] + organic_per_tick[:, t]
            S = S * decay + new_impulse

        paid_curve = mu.sum(axis=0)  # per tick rate
        organic_curve = organic_per_tick.sum(axis=0)
        total_curve = paid_curve + organic_curve
        seg_curves = mu + organic_per_tick

        # aggregate to cumulative daily
        peak_tick = int(np.argmax(organic_curve))
        peak_day = float(peak_tick * dt)

        # return cumulative daily bins
        # reshape ticks → days
        ticks_per_day = int(1.0 / dt)

        def _to_daily(x):
            nd = len(x) // ticks_per_day
            return x[: nd * ticks_per_day].reshape(nd, ticks_per_day).sum(axis=1)

        return HawkesResult(
            days=days,
            dt=dt,
            t_axis=t_axis[: int(days * ticks_per_day)][::ticks_per_day][:days],
            paid_curve=_to_daily(paid_curve),
            organic_curve=_to_daily(organic_curve),
            total_curve=_to_daily(total_curve),
            seg_curves=np.stack([_to_daily(seg_curves[s]) for s in range(N_SEG)]),
            branching_ratio=float(self.branching),
            peak_day=peak_day,
        )


def hawkes_result_to_dict(hr: HawkesResult) -> dict:
    return {
        "days": hr.days,
        "day_axis": list(range(hr.days)),
        "paid_daily": [round(float(x), 2) for x in hr.paid_curve],
        "organic_daily": [round(float(x), 2) for x in hr.organic_curve],
        "total_daily": [round(float(x), 2) for x in hr.total_curve],
        "branching_ratio": hr.branching_ratio,
        "peak_day": hr.peak_day,
        "organic_share": round(float(hr.organic_curve.sum() / max(hr.total_curve.sum(), 1e-6)), 3),
    }

"""Voronoi / nearest-neighbor calibration.

Core idea: 100 LLM souls represent N (e.g. 100k or 1M) population agents the
same way a 1000-person poll represents 1.4B citizens.

Algorithm:
  1. Build (N, D) feature matrix for the whole population
  2. For each agent, find its nearest LLM soul by feature distance
       → defines a Voronoi partition of feature space
  3. Each soul's territory size / N = its sample weight
  4. Compute per-soul correction factor = LLM_verdict / stat_prediction
  5. Apply factor to all population agents in that soul's territory
       → 100 LLM verdicts effectively rescale prediction for all N agents
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def _build_features(population, indices: Optional[np.ndarray] = None) -> np.ndarray:
    if indices is None:
        idx = np.arange(population.N)
    else:
        idx = np.asarray(indices)
    interest = population.interest[idx]                       # (N, 64) L2-normed
    age = (population.age_idx[idx][:, None] / 5.0).astype(np.float32)
    gender = population.gender_idx[idx][:, None].astype(np.float32)
    city = (population.city_idx[idx][:, None] / 4.0).astype(np.float32)
    income = (population.income[idx][:, None] / 9.0).astype(np.float32)
    edu = (population.edu_idx[idx][:, None] / 4.0).astype(np.float32)
    return np.concatenate(
        [interest * 0.4, age, gender, city, income, edu], axis=1
    ).astype(np.float32)


@dataclass
class VoronoiPartition:
    soul_indices: np.ndarray          # (S,) population indices of souls
    nearest: np.ndarray               # (N,) for each agent → soul slot
    weights: np.ndarray               # (S,) territory size / N
    feat_pop: np.ndarray              # (N, D)
    feat_souls: np.ndarray            # (S, D)


def voronoi_partition(population, soul_indices: List[int],
                      chunk: int = 20_000) -> VoronoiPartition:
    """Compute Voronoi partition of population by soul features."""
    feat_pop = _build_features(population)                     # (N, D)
    feat_souls = _build_features(population, indices=np.array(soul_indices))  # (S, D)
    N, S = feat_pop.shape[0], feat_souls.shape[0]

    nearest = np.zeros(N, dtype=np.int32)
    # squared L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    soul_sq = (feat_souls ** 2).sum(axis=1)                    # (S,)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        block = feat_pop[s:e]                                  # (B, D)
        block_sq = (block ** 2).sum(axis=1, keepdims=True)     # (B, 1)
        # cross term
        cross = block @ feat_souls.T                           # (B, S)
        d = block_sq + soul_sq[None, :] - 2 * cross
        nearest[s:e] = d.argmin(axis=1)

    counts = np.bincount(nearest, minlength=S).astype(np.float32)
    weights = counts / max(counts.sum(), 1)
    return VoronoiPartition(
        soul_indices=np.array(soul_indices, dtype=np.int64),
        nearest=nearest, weights=weights,
        feat_pop=feat_pop, feat_souls=feat_souls,
    )


def calibrate_per_territory(
    souls: List[Dict],
    partition: VoronoiPartition,
    stat_click_probs_by_persona: Dict[int, float],
    eps: float = 0.05,
    persona_id_to_slot: Optional[Dict[int, int]] = None,
) -> Dict:
    """For each soul → territory, compute correction factor; produce both
    a global (weighted-geomean) factor and per-segment factors.

    Returns:
      {
        "per_soul_factor": (S,) array,
        "global_factor": float,           # log-weighted-mean
        "soul_weights": (S,) array,
        "soul_verdicts": (S,) array (0/1),
        "stat_at_souls": (S,) array (0..1),
        "covered_population_share": float (= 1.0 if all assigned),
      }
    """
    S = len(souls)
    soul_verdicts = np.zeros(S, dtype=np.float32)
    stat_at_souls = np.zeros(S, dtype=np.float32)
    factors = np.zeros(S, dtype=np.float32)

    for i, s in enumerate(souls):
        soul_verdicts[i] = 1.0 if s.get("will_click") else 0.0
        pid = s.get("persona_id")
        if pid is not None:
            stat_at_souls[i] = stat_click_probs_by_persona.get(int(pid), eps)
        else:
            stat_at_souls[i] = eps
        factors[i] = (soul_verdicts[i] + eps) / (stat_at_souls[i] + eps)

    factors = np.clip(factors, 0.2, 5.0)

    # Pick the subset of partition weights corresponding to these souls.
    if persona_id_to_slot is not None:
        slots = np.array([persona_id_to_slot.get(int(s.get("persona_id", -1)), -1)
                          for s in souls], dtype=np.int32)
        valid = slots >= 0
        if valid.any():
            weights = np.zeros(S, dtype=np.float32)
            weights[valid] = partition.weights[slots[valid]]
        else:
            weights = np.full(S, 1.0 / S, dtype=np.float32)
    elif S == len(partition.weights):
        weights = partition.weights
    else:
        # uneven — fall back to uniform subset weights
        weights = np.full(S, 1.0 / S, dtype=np.float32)
    weights = weights / max(weights.sum(), 1e-8)

    log_factors = np.log(factors + 1e-6)
    global_factor = float(np.exp(np.sum(weights * log_factors)))
    return {
        "per_soul_factor": [round(float(f), 3) for f in factors],
        "global_factor": round(global_factor, 3),
        "soul_weights": [round(float(w), 4) for w in weights],
        "soul_verdicts": soul_verdicts.tolist(),
        "stat_at_souls": [round(float(p), 3) for p in stat_at_souls],
        "n_souls": S,
        "n_population_covered": int((weights > 0).sum()),
    }


def calibration_summary(cal: Dict) -> Dict:
    """Compact summary for UI."""
    if not cal:
        return {}
    factors = np.array(cal["per_soul_factor"])
    weights = np.array(cal["soul_weights"])
    return {
        "global_factor": cal["global_factor"],
        "n_souls_active": int((weights > 0).sum()),
        "n_souls_total": cal["n_souls"],
        "factor_p10": float(np.percentile(factors, 10)),
        "factor_p50": float(np.percentile(factors, 50)),
        "factor_p90": float(np.percentile(factors, 90)),
        "max_territory_weight": float(weights.max()),
        "mean_territory_weight": float(weights.mean()),
    }

"""CATE (Conditional Average Treatment Effect) estimation.

Uses a small sklearn RandomForest approximation instead of EconML CausalForest
to keep MVP dependency light. API-compatible with swap later.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..data.population import Population


def compute_cate(
    population: Population,
    baseline_click_prob: dict[int, float],
    cf_click_prob: dict[int, float],
    top_k_segments: int = 5,
) -> list[dict]:
    """Given per-agent click probs under baseline vs counterfactual,
    fit a forest on agent features → delta, then extract top segments.
    """
    # Use UNION not intersection so budget-only interventions (which only
    # change who gets impressed, not per-agent click_prob) still surface as
    # exposure-gain / exposure-loss signals. Agent missing from one side → 0.
    # Previously this used intersection which zeroed out budget-only CATE.
    base_keys = set(baseline_click_prob.keys())
    cf_keys = set(cf_click_prob.keys())
    union = sorted(base_keys | cf_keys)
    if len(union) < 20:
        return []
    idx = np.array(union, dtype=np.int64)
    delta = np.array(
        [cf_click_prob.get(i, 0.0) - baseline_click_prob.get(i, 0.0) for i in union],
        dtype=np.float32,
    )

    # When pop.N caps both imp sets (impression saturation), base==cf agent sets
    # and per-agent click_prob is identical → delta is all-zero by model design.
    # Return an explicit diagnostic instead of a silent table of zeros so callers
    # can tell "model is saturated" apart from "intervention has no effect".
    if float(np.abs(delta).max()) < 1e-9:
        return [
            {
                "_diagnostic": "saturated_impression_set",
                "note": (
                    f"base 触达 {len(base_keys)} / cf 触达 {len(cf_keys)} / 交集 "
                    f"{len(base_keys & cf_keys)} · 所有 common agent click_prob 相同 → "
                    "delta=0。多为预算已使 impression 饱和至 pop.N，建议增大 POP_SIZE 或改用 alloc/kol/audience 干预观察 CATE"
                ),
                "base_reached": len(base_keys),
                "cf_reached": len(cf_keys),
                "common": len(base_keys & cf_keys),
            }
        ]

    X = np.concatenate(
        [
            population.age_idx[idx][:, None],
            population.gender_idx[idx][:, None],
            population.city_idx[idx][:, None],
            population.income[idx][:, None],
            population.edu_idx[idx][:, None],
        ],
        axis=1,
    ).astype(np.float32)

    rf = RandomForestRegressor(n_estimators=40, min_samples_leaf=20, random_state=0)
    rf.fit(X, delta)
    # feature importance = approx. CATE driver
    importances = dict(
        zip(
            ["age", "gender", "city_tier", "income_decile", "education"],
            rf.feature_importances_.tolist(),
        )
    )
    # Segment analysis: mean delta per age × gender cell
    segments = []
    for a in range(6):
        for g in range(2):
            mask = (population.age_idx[idx] == a) & (population.gender_idx[idx] == g)
            if mask.sum() < 10:
                continue
            seg_delta = float(delta[mask].mean())
            seg_n = int(mask.sum())
            segments.append(
                {
                    "segment": f"{['15-24','25-34','35-44','45-54','55-64','65+'][a]}·{'女' if g==0 else '男'}",
                    "delta": seg_delta,
                    "n": seg_n,
                }
            )
    segments.sort(key=lambda s: abs(s["delta"]), reverse=True)
    return [{"importances": importances}, {"top_segments": segments[:top_k_segments]}]

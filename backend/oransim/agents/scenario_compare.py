"""T3-A4 scenario_comparison + Wilcoxon — paired statistical comparison.

Given baseline scenario A and intervention scenario B, run N Monte Carlo
samples of each and apply Wilcoxon signed-rank test on per-sample KPI deltas.

Why Wilcoxon (not t-test):
  - Doesn't assume normal distribution (KPIs are right-skewed)
  - Robust to outliers
  - Works on small N (we typically use N=10-20 paired samples)

Output: schema T3-A4 row.
"""
from __future__ import annotations
import time
import uuid
from typing import Dict, List, Optional, Callable

import numpy as np


def compare_scenarios(scenario_a_label: str, scenario_b_label: str,
                      samples_a: List[Dict], samples_b: List[Dict],
                      kpi_keys: Optional[List[str]] = None) -> Dict:
    """Compare two scenarios via Wilcoxon signed-rank test.

    samples_a / samples_b: list of KPI dicts (e.g. [{roi: 1.2, ctr: 0.013}, ...])
    Must be PAIRED (same length, same random seed across iterations).
    """
    if not samples_a or not samples_b:
        return {"_error": "empty samples", "rows": []}
    if len(samples_a) != len(samples_b):
        return {"_error": f"unpaired samples: A={len(samples_a)} B={len(samples_b)}"}

    kpi_keys = kpi_keys or ["roi", "ctr", "cvr", "impressions", "clicks", "conversions"]

    rows = []
    for k in kpi_keys:
        a_vals = np.asarray([float(s.get(k, 0) or 0) for s in samples_a])
        b_vals = np.asarray([float(s.get(k, 0) or 0) for s in samples_b])
        if len(a_vals) < 3:
            continue
        # Wilcoxon signed-rank test on paired differences
        try:
            from scipy.stats import wilcoxon
            diff = b_vals - a_vals
            if np.allclose(diff, 0):
                stat, p = 0.0, 1.0
                test_name = "wilcoxon (zero-diff)"
            else:
                res = wilcoxon(b_vals, a_vals, alternative="two-sided",
                               zero_method="zsplit")
                stat = float(res.statistic)
                p = float(res.pvalue)
                test_name = "wilcoxon_signed_rank"
        except ImportError:
            # Manual fallback
            stat, p = _manual_wilcoxon(a_vals, b_vals)
            test_name = "manual_wilcoxon"
        except Exception as e:
            stat, p = None, None
            test_name = f"error: {e}"

        a_mean = float(a_vals.mean())
        b_mean = float(b_vals.mean())
        delta = b_mean - a_mean
        delta_pct = (delta / a_mean * 100) if a_mean else None

        rows.append({
            "kpi": k,
            "scenario_a_mean": round(a_mean, 4),
            "scenario_a_std": round(float(a_vals.std()), 4),
            "scenario_b_mean": round(b_mean, 4),
            "scenario_b_std": round(float(b_vals.std()), 4),
            "delta_absolute": round(delta, 4),
            "delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
            "wilcoxon_statistic": round(stat, 3) if stat is not None else None,
            "p_value": round(p, 4) if p is not None else None,
            "is_significant_05": bool(p is not None and p < 0.05),
            "is_significant_01": bool(p is not None and p < 0.01),
            "test_method": test_name,
            "n_samples": int(len(a_vals)),
        })

    # Recommend scenario
    sig_b_better = sum(1 for r in rows if r.get("is_significant_05")
                       and (r.get("delta_pct") or 0) > 0)
    sig_b_worse = sum(1 for r in rows if r.get("is_significant_05")
                      and (r.get("delta_pct") or 0) < 0)
    if sig_b_better > sig_b_worse:
        recommended = "B"
        reason = f"B 在 {sig_b_better} 项 KPI 显著优于 A (p<0.05)"
    elif sig_b_worse > sig_b_better:
        recommended = "A"
        reason = f"A 在 {sig_b_worse} 项 KPI 显著优于 B (p<0.05)"
    else:
        recommended = "tie"
        reason = "无显著差异，建议看长期/品牌指标"

    return {
        "comparison_id": f"comp_{uuid.uuid4().hex[:8]}",
        "scenario_a_desc": scenario_a_label,
        "scenario_b_desc": scenario_b_label,
        "n_samples": int(len(samples_a)),
        "kpi_comparison": rows,
        "recommended_scenario": recommended,
        "recommendation_reason": reason,
        "test_family": "Wilcoxon signed-rank (non-parametric, paired)",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _manual_wilcoxon(a: np.ndarray, b: np.ndarray):
    """Fallback signed-rank test if scipy unavailable. Z-approximation."""
    diff = b - a
    nz = diff[diff != 0]
    n = len(nz)
    if n < 5: return 0.0, 1.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    W_pos = float(ranks[nz > 0].sum())
    W_neg = float(ranks[nz < 0].sum())
    W = min(W_pos, W_neg)
    mu_W = n * (n + 1) / 4
    sigma_W = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sigma_W == 0: return W, 1.0
    z = (W - mu_W) / sigma_W
    # 2-sided p ≈ 2 × (1 - Φ(|z|))
    from .content_type_coef import _phi
    p = 2 * (1 - _phi(abs(z)))
    return W, max(min(p, 1.0), 0.0)


def quick_compare_from_predict(predict_fn: Callable, base_req: Dict,
                                intervention: Dict, n_samples: int = 10) -> Dict:
    """Helper: re-run /api/predict N times with different seeds for both scenarios.

    predict_fn: a callable that takes a request dict + seed → returns full predict response
    base_req: baseline PredictRequest dict
    intervention: dict to merge into base_req for scenario B
    """
    samples_a, samples_b = [], []
    for i in range(n_samples):
        try:
            ra = predict_fn({**base_req, "_seed": i})
            samples_a.append(ra.get("kpis", {}))
        except Exception:
            pass
        try:
            req_b = {**base_req, **intervention, "_seed": i}
            rb = predict_fn(req_b)
            samples_b.append(rb.get("kpis", {}))
        except Exception:
            pass
    label_b = " / ".join(f"{k}={v}" for k, v in intervention.items())[:60]
    return compare_scenarios(
        scenario_a_label="baseline",
        scenario_b_label=f"intervention: {label_b}",
        samples_a=samples_a, samples_b=samples_b,
    )

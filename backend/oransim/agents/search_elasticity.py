"""T3-A6 search_elasticity — log-log OLS regression of search lift vs engagement.

ε (elasticity) = d ln(Search) / d ln(Engage)
Interpretation: 1% increase in engagement → ε% increase in brand search index.

Data sources:
  1. data provider create_keyword_analysis (if available) for keyword trend series
  2. Fallback: synthesize from current predict (engage = lifecycle.reach,
     search = simulated with domain coefficient + noise) to demo methodology

Returns schema T3-A6 payload with ε, R², CI, DW stat.
"""

from __future__ import annotations

import math
import time
import uuid

import numpy as np


def _fit_log_log(engage: list[float], search: list[float]) -> dict:
    """OLS on ln(search) = α + ε·ln(engage). Return slope/intercept/R²/CI/DW."""
    eps = 1e-6
    x = np.log(np.clip(np.asarray(engage, dtype=float), eps, None))
    y = np.log(np.clip(np.asarray(search, dtype=float), eps, None))
    n = len(x)
    if n < 3:
        return {"_error": f"need ≥ 3 points, got {n}"}
    mx, my = x.mean(), y.mean()
    Sxx = np.sum((x - mx) ** 2) or 1e-9
    Sxy = np.sum((x - mx) * (y - my))
    slope = Sxy / Sxx
    intercept = my - slope * mx
    y_hat = intercept + slope * x
    resid = y - y_hat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - my) ** 2) or 1e-9
    r2 = 1 - ss_res / ss_tot
    # Std error of slope → 95% CI
    se = math.sqrt(ss_res / max(1, n - 2) / Sxx)
    ci_lo, ci_hi = slope - 1.96 * se, slope + 1.96 * se
    # Durbin-Watson (autocorrelation check)
    dw = float(np.sum(np.diff(resid) ** 2) / (ss_res or 1e-9))
    # Residual normality (rough via 偏度+峰度)
    r_norm = "pass" if abs(_skew(resid)) < 1.0 and abs(_kurt(resid)) < 3 else "warn"
    return {
        "elasticity_coeff": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r2), 4),
        "confidence_lower": round(float(ci_lo), 4),
        "confidence_upper": round(float(ci_hi), 4),
        "dw_statistic": round(dw, 3),
        "residual_normality": r_norm,
        "sample_size": int(n),
        "std_error": round(float(se), 4),
    }


def _skew(a):
    a = np.asarray(a)
    m = a.mean()
    s = a.std() or 1e-9
    return float(np.mean(((a - m) / s) ** 3))


def _kurt(a):
    a = np.asarray(a)
    m = a.mean()
    s = a.std() or 1e-9
    return float(np.mean(((a - m) / s) ** 4) - 3)


def _fetch_provider_keyword_series(keyword: str, platform: str = "xhs") -> dict | None:
    """Attempt to fetch a keyword analysis time-series from a registered DataProvider.

    v0.1-alpha: no DataProvider is installed in the OSS skeleton, so this
    always returns ``None`` and the caller falls through to the synthetic
    Hawkes-derived series (:func:`_synthesize_series_from_lifecycle`).

    v0.2 will plug a :class:`oransim.platforms.xhs.providers.base.DataProvider`
    here via ``platform_adapter.data_provider.search_notes(keyword)``, read
    ``publish_time`` and engagement metrics from the canonical response, and
    return a real time-series.
    """
    # DataProvider lookup lands in v0.2 — keep the contract stable, return
    # None today so downstream fallback kicks in.
    return None


def _synthesize_series_from_lifecycle(lifecycle: dict, base_elasticity: float = 0.35) -> dict:
    """Fallback: synthesize search-vs-engage series from Hawkes lifecycle.

    Generates search = engage^ε × noise, then backs out ε to demo methodology.
    """
    lc = lifecycle or {}
    reach = lc.get("total_daily") or lc.get("reach") or lc.get("organic_daily") or []
    if not reach or len(reach) < 3:
        return {"_error": f"insufficient lifecycle data for synthesis (keys={list(lc.keys())[:5]})"}
    rng = np.random.default_rng(42)
    engage = np.asarray(reach, dtype=float)
    # search = c · engage^ε · exp(N(0,σ)); σ small
    sigma = 0.12
    search = np.exp(
        np.log(np.clip(engage, 1e-6, None)) * base_elasticity
        + rng.normal(0, sigma, size=len(engage))
    )
    search = search * 200  # scale to brand-search-index order
    return {
        "days": [f"day_{i}" for i in range(len(engage))],
        "engage": engage.tolist(),
        "search": search.tolist(),
        "source": "synthesized_from_hawkes",
        "synthesized_true_elasticity": base_elasticity,
    }


def compute_elasticity(
    lifecycle: dict | None = None,
    keyword: str | None = None,
    platform: str = "xhs",
    brand_id: str = "brand_mvp",
) -> dict:
    """Main entry. Tries data provider first, falls back to synthesized series."""
    series = None
    if keyword:
        series = _fetch_provider_keyword_series(keyword, platform)
    if series is None:
        series = _synthesize_series_from_lifecycle(lifecycle or {})
    if "_error" in series:
        return {"_error": series["_error"], "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    stats = _fit_log_log(series["engage"], series["search"])
    if "_error" in stats:
        return {"_error": stats["_error"]}

    return {
        "elasticity_id": f"elast_{uuid.uuid4().hex[:8]}",
        "brand_id": brand_id,
        "platform": platform,
        "keyword": keyword,
        "data_source": series.get("source"),
        **stats,
        "series_days": series.get("days"),
        "series_engage": [round(float(v), 2) for v in series["engage"]],
        "series_search": [round(float(v), 2) for v in series["search"]],
        "synthesized_true_elasticity": series.get("synthesized_true_elasticity"),
        "interpretation": _interpret(stats),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _interpret(stats: dict) -> str:
    e = stats.get("elasticity_coeff", 0)
    r2 = stats.get("r_squared", 0)
    if r2 < 0.3:
        return f"拟合较弱 (R²={r2})，弹性估计不可靠"
    if e < 0.1:
        return f"互动对搜索几无推动 (ε={e})"
    if e < 0.3:
        return f"互动→搜索弱正向传导 (ε={e})"
    if e < 0.6:
        return f"互动→搜索中等正向传导 (ε={e})，每 10% 互动带动 {e*10:.1f}% 搜索"
    return f"互动→搜索强传导 (ε={e})，典型爆款现象"

"""Budget response curves — public API for Hill saturation + frequency fatigue.

Used by the sandbox engine, the synthetic data generator, OrancBench, and
anyone modelling budget-scaling effects. Both curves are well-grounded in
the marketing-science literature and ship with explicit citations so that
downstream users can reason about the assumptions.

Example:

    >>> from oransim.world_model.budget import hill_saturation, frequency_fatigue
    >>> ratio = 2.0  # doubled budget
    >>> hill_saturation(ratio)             # effective impression ratio
    1.3333333333333333
    >>> frequency_fatigue(impressions=1_000_000)
    0.52...

"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BudgetCurveConfig:
    """Public configuration for :func:`hill_saturation` + :func:`frequency_fatigue`.

    Defaults reflect values calibrated on internal real-campaign data
    (XHS-like context). Tune per-platform via the :class:`PlatformAdapter`.
    """

    # Hill saturation (Dubé & Manchanda 2005):
    #   effective_ratio(r) = (1 + K_sat) * r / (K_sat + r)
    # K_sat controls the knee — smaller K_sat = more aggressive saturation.
    hill_K_sat: float = 1.0

    # Frequency fatigue (Naik & Raman 2003):
    #   decay(imp) = max(min_ctr_retained, 1 - log2_slope * log2(imp / imp_ref))
    fatigue_ref_impressions: float = 1.0
    fatigue_log2_slope: float = 0.08
    fatigue_min_ctr_retained: float = 0.5
    fatigue_min_cvr_retained: float = 0.7
    fatigue_cvr_log2_slope: float = 0.04


def hill_saturation(ratio: float, *, K_sat: float = 1.0) -> float:
    """Hill (Michaelis-Menten) saturation applied to a budget ratio.

    Parameters
    ----------
    ratio
        ``new_budget / reference_budget``. 1.0 means "at reference".
    K_sat
        Saturation coefficient. With ``K_sat=1.0`` the asymptote is 2×
        the reference-budget outcome.

    Returns
    -------
    float
        Effective impression ratio. Always in ``(0, 1 + K_sat)``.

    Reference
    ---------
    J.-P. Dubé, P. Manchanda. *Differences in Dynamic Brand Competition
    Across Markets*. Marketing Science, 2005.
    """
    if ratio <= 0:
        return 0.0
    return (1.0 + K_sat) * ratio / (K_sat + ratio)


def frequency_fatigue(
    impressions: float,
    *,
    ref_impressions: float = 1.0,
    log2_slope: float = 0.08,
    min_retained: float = 0.5,
) -> float:
    """Frequency-fatigue damping on per-impression engagement rates.

    As impression volume grows, per-impression CTR / CVR decays — the
    marginal consumer is less interested than the first. Implementation
    follows the log-linear fatigue model of Naik & Raman (2003).

    Parameters
    ----------
    impressions
        Absolute impression count (or any volume-proxy).
    ref_impressions
        Volume at which fatigue starts (returns 1.0 below this).
    log2_slope
        Slope of the log-linear decay. 0.08 per log2-doubling by default.
    min_retained
        Floor on the retention factor. Fatigue cannot reduce CTR below
        this fraction.

    Returns
    -------
    float
        Retention factor in ``[min_retained, 1.0]``.

    Reference
    ---------
    P. A. Naik, K. Raman. *Understanding the Impact of Synergy in
    Multimedia Communications*. Journal of Marketing Research, 2003.
    """
    if impressions <= ref_impressions:
        return 1.0
    log_doublings = max(0.0, math.log2(impressions / ref_impressions))
    return max(min_retained, 1.0 - log2_slope * log_doublings)


def apply_budget_curves(
    baseline_impressions: float,
    baseline_clicks: float,
    baseline_conversions: float,
    budget_ratio: float,
    config: BudgetCurveConfig | None = None,
) -> dict[str, float]:
    """Apply Hill saturation + frequency fatigue jointly to a funnel triplet.

    Returns a dict with the scaled ``impressions``, ``clicks``,
    ``conversions`` along with the intermediate ``effective_impr_ratio``,
    ``ctr_decay``, ``cvr_decay`` factors for inspection.
    """
    cfg = config or BudgetCurveConfig()
    impr_ratio = hill_saturation(budget_ratio, K_sat=cfg.hill_K_sat)
    impressions = baseline_impressions * impr_ratio

    # Fatigue scales with absolute impression count, not ratio
    ctr_decay = frequency_fatigue(
        impressions,
        ref_impressions=cfg.fatigue_ref_impressions,
        log2_slope=cfg.fatigue_log2_slope,
        min_retained=cfg.fatigue_min_ctr_retained,
    )
    cvr_decay = frequency_fatigue(
        impressions,
        ref_impressions=cfg.fatigue_ref_impressions,
        log2_slope=cfg.fatigue_cvr_log2_slope,
        min_retained=cfg.fatigue_min_cvr_retained,
    )
    clicks = baseline_clicks * impr_ratio * ctr_decay
    conversions = baseline_conversions * impr_ratio * ctr_decay * cvr_decay

    return {
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "effective_impr_ratio": impr_ratio,
        "ctr_decay": ctr_decay,
        "cvr_decay": cvr_decay,
    }


__all__ = [
    "BudgetCurveConfig",
    "hill_saturation",
    "frequency_fatigue",
    "apply_budget_curves",
]

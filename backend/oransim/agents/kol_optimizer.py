"""T2-A1 kol_mix_optimization — integer-programming KOL portfolio selection.

Inputs
------
- KOL pool from ``data/synthetic/synthetic_kols.json`` (200 synthetic KOLs with
  ``fan_count`` + ``avg_engagement_rate`` + ``tier`` + ``niche_zh``).
- Total budget (CNY).
- Target niches (preference ordering).
- KOL:KOC ratio constraint.

Objective: max Σ ROI_i × selected_i
Subject to: Σ cost_i × selected_i ≤ budget
            Σ koc_i × selected_i ≥ min_koc_ratio × Σ selected_i

Solver: scipy.optimize.milp; falls back to greedy ROI/cost on missing scipy or
infeasibility. Both paths return the same schema.

Enterprise Edition swaps the synthetic pool for continuously-updated real-panel
KOL data with true CPE / ROI feedback.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid

import numpy as np

_POOL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "synthetic", "synthetic_kols.json"
)
_POOL: list[dict] | None = None


def _load_pool() -> list[dict]:
    global _POOL
    if _POOL is None:
        try:
            with open(os.path.abspath(_POOL_PATH), encoding="utf-8") as f:
                _POOL = json.load(f)
        except Exception:
            _POOL = []
    return _POOL


def _classify_tier(fans: int) -> str:
    if fans >= 1_000_000:
        return "头部"
    if fans >= 300_000:
        return "腰部"
    if fans >= 50_000:
        return "尾部"
    return "KOC"


def _estimated_cpe(fans: int) -> float:
    """Industry heuristic CPE by tier (CNY per engagement)."""
    if fans >= 1_000_000:
        return 3.5
    if fans >= 300_000:
        return 2.0
    if fans >= 50_000:
        return 1.2
    return 0.5


def _estimated_cost(fans: int, interaction_rate: float) -> float:
    cpe = _estimated_cpe(fans)
    expected_engage = fans * max(0.001, interaction_rate)
    return cpe * expected_engage


def _estimated_roi(fans: int, niche_match: bool = True) -> float:
    """Heuristic ROI: diminishing returns on fan count (peak ~30k fans);
    niche match adds a multiplicative bonus."""
    log_fans = math.log10(max(1, fans))
    efficiency = 1 + 1.8 * math.exp(-((log_fans - 4.5) ** 2) / 2)
    return round(efficiency * (2.0 if niche_match else 1.2), 3)


_EN_TO_ZH = {
    "beauty": "美妆",
    "fashion": "穿搭",
    "food": "美食",
    "electronics": "数码",
    "travel": "旅行",
    "parenting": "母婴",
    "fitness": "健身",
    "finance": "理财",
    "mom": "母婴",
    "tech": "数码",
}


def _normalize_niche(n: str) -> str:
    return _EN_TO_ZH.get(n, n)


def _gather_candidates(
    target_niches: list[str] | None = None,
    max_per_niche: int = 80,
    platform: str = "xhs",
) -> list[dict]:
    pool = _load_pool()
    if not pool:
        return []
    target_zh = {_normalize_niche(n) for n in (target_niches or [])}

    by_niche: dict[str, list[dict]] = {}
    for k in pool:
        niche = k.get("niche_zh") or _normalize_niche(k.get("niche_en", ""))
        by_niche.setdefault(niche, []).append(k)

    cands: list[dict] = []
    niches_to_scan = target_zh if target_zh else set(by_niche.keys())
    for niche in niches_to_scan:
        items = by_niche.get(niche, [])[:max_per_niche]
        for k in items:
            fans = int(k.get("fan_count", 0) or 0)
            if fans < 500:
                continue
            ir = float(k.get("avg_engagement_rate") or 0.03)
            cost = _estimated_cost(fans, ir)
            match = niche in target_zh if target_zh else True
            roi = _estimated_roi(fans, niche_match=match)
            cands.append(
                {
                    "kol_id": k.get("kol_id", "?"),
                    "name": k.get("nickname", "?"),
                    "niche": niche,
                    "platform": platform,
                    "fans": fans,
                    "tier": _classify_tier(fans),
                    "interaction_rate": round(ir, 4),
                    "estimated_cost": round(cost, 1),
                    "estimated_reach": int(fans * 0.35),
                    "estimated_engagement": int(fans * ir),
                    "estimated_roi": roi,
                }
            )
    return cands


def _greedy_knapsack(cands: list[dict], budget: float, min_koc_ratio: float = 0.5) -> list[dict]:
    sorted_cands = sorted(cands, key=lambda c: -c["estimated_roi"] / max(1, c["estimated_cost"]))
    picked: list[dict] = []
    spent = 0.0
    for c in sorted_cands:
        if spent + c["estimated_cost"] > budget:
            continue
        picked.append(c)
        spent += c["estimated_cost"]
    # Rebalance to hit min_koc_ratio
    n_koc = sum(1 for p in picked if p["tier"] == "KOC")
    if len(picked) > 0 and n_koc / len(picked) < min_koc_ratio:
        head_idx = [i for i, p in enumerate(picked) if p["tier"] == "头部"]
        koc_cands = [c for c in sorted_cands if c["tier"] == "KOC" and c not in picked]
        while head_idx and koc_cands and n_koc / max(1, len(picked)) < min_koc_ratio:
            idx = head_idx.pop()
            picked[idx] = koc_cands.pop(0)
            n_koc += 1
    return picked


def _milp_solve(
    cands: list[dict], budget: float, min_koc_ratio: float = 0.5
) -> tuple[list[dict], str]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return _greedy_knapsack(cands, budget, min_koc_ratio), "greedy_fallback_no_scipy"
    n = len(cands)
    if n == 0:
        return [], "empty"
    c = -np.array([x["estimated_roi"] for x in cands])
    A_cost = np.array([x["estimated_cost"] for x in cands]).reshape(1, n)
    koc_flag = np.array([1.0 if x["tier"] == "KOC" else 0.0 for x in cands])
    A_ratio = ((1 - min_koc_ratio) * np.ones(n) - koc_flag).reshape(1, n)
    A = np.vstack([A_cost, A_ratio])
    cons = LinearConstraint(A, ub=np.array([budget, 0.0]))
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    try:
        res = milp(c, constraints=cons, integrality=np.ones(n), bounds=bounds)
        if not res.success or res.x is None:
            return _greedy_knapsack(cands, budget, min_koc_ratio), "greedy_fallback_infeasible"
        picked = [cands[i] for i in range(n) if res.x[i] > 0.5]
        return picked, "milp_optimal"
    except Exception as e:
        return _greedy_knapsack(cands, budget, min_koc_ratio), f"greedy_fallback_{type(e).__name__}"


def optimize_kol_mix(
    total_budget: float,
    target_niches: list[str] | None = None,
    objective: str = "max_roi",
    min_koc_ratio: float = 0.5,
    max_per_niche: int = 80,
    platform: str = "xhs",
) -> dict:
    """Schema T2-A1 kol_mix_optimization payload."""
    cands = _gather_candidates(target_niches, max_per_niche, platform)
    picked, solver = _milp_solve(cands, total_budget, min_koc_ratio)

    kol_count = sum(1 for p in picked if p["tier"] in ("头部", "腰部"))
    koc_count = sum(1 for p in picked if p["tier"] in ("尾部", "KOC"))
    total_cost = sum(p["estimated_cost"] for p in picked)
    total_reach = sum(p["estimated_reach"] for p in picked)
    total_engage = sum(p["estimated_engagement"] for p in picked)
    avg_roi = sum(p["estimated_roi"] for p in picked) / max(1, len(picked)) if picked else 0.0

    return {
        "plan_id": f"plan_{uuid.uuid4().hex[:8]}",
        "budget": total_budget,
        "kol_count": kol_count,
        "koc_count": koc_count,
        "total_selected": len(picked),
        "kol_koc_ratio": f"{kol_count}:{koc_count}",
        "candidate_pool_size": len(cands),
        "selected_kol_ids": [p["kol_id"] for p in picked],
        "selected_kols": [
            {
                "kol_id": p["kol_id"],
                "name": p["name"],
                "niche": p["niche"],
                "tier": p["tier"],
                "fans": p["fans"],
                "cost": p["estimated_cost"],
                "reach": p["estimated_reach"],
                "roi": p["estimated_roi"],
            }
            for p in picked
        ],
        "estimated_total_reach": total_reach,
        "estimated_total_engagement": total_engage,
        "estimated_roi": round(avg_roi, 3),
        "estimated_cost": round(total_cost, 1),
        "budget_utilization": round(total_cost / max(1, total_budget), 3),
        "optimization_objective": objective,
        "solver_status": solver,
        "target_niches": target_niches,
        "data_source": "synthetic_kols (200 KOLs)",
        "note": "Synthetic pool — Enterprise Edition swaps in a real-panel KOL index.",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def reinvest_ranking(selected_kols: list[dict]) -> list[dict]:
    """T2-A5 kol_reinvest_ranking — score by ROI × reach efficiency."""
    rows = []
    for p in selected_kols:
        eff = p["reach"] / max(1, p["cost"])
        score = p["roi"] * math.sqrt(eff / 100)
        if score > 0.9:
            rec, trend = "优先复投", "rising"
        elif score > 0.6:
            rec, trend = "观望", "stable"
        else:
            rec, trend = "替换", "falling"
        rows.append(
            {
                "kol_id": p["kol_id"],
                "name": p["name"],
                "reinvest_score": round(score, 3),
                "historical_roi_avg": p["roi"],
                "audience_match_score": round(min(1.0, p["roi"] / 3), 3),
                "cost_trend": trend,
                "recommendation": rec,
            }
        )
    rows.sort(key=lambda r: -r["reinvest_score"])
    for i, r in enumerate(rows):
        r["reinvest_rank"] = i + 1
    return rows

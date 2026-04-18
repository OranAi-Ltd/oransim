"""OrancBench v0.1 — scenario loader, scorer, and data types.

Scenarios ship at ``data/benchmarks/orancbench_v0_1.jsonl`` (committed with
the release). The dataset generator lives at
``backend/scripts/gen_orancbench.py``.

Usage:

    from oransim.benchmarks import load_scenarios, score_predictions

    scenarios = load_scenarios()
    preds = {s.scenario_id: my_predict(s) for s in scenarios}
    report = score_predictions(scenarios, preds)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class OrancBenchScenario:
    scenario_id: str
    niche: str
    platform: str
    budget: float
    budget_bucket: int
    kol_tier: str
    kol_fan_count: int
    kol_engagement_rate: float
    creative_caption: str
    difficulty: str
    ground_truth: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "OrancBenchScenario":
        return cls(
            scenario_id=d["scenario_id"],
            niche=d["niche"],
            platform=d["platform"],
            budget=float(d["budget"]),
            budget_bucket=int(d["budget_bucket"]),
            kol_tier=d["kol_tier"],
            kol_fan_count=int(d["kol_fan_count"]),
            kol_engagement_rate=float(d["kol_engagement_rate"]),
            creative_caption=d["creative_caption"],
            difficulty=d["difficulty"],
            ground_truth=dict(d.get("ground_truth") or {}),
        )


@dataclass
class OrancBenchResult:
    """Per-difficulty metrics produced by :func:`score_predictions`."""

    n: int
    r2: dict[str, float]                     # per-KPI R² on median prediction
    mape: dict[str, float]                   # per-KPI MAPE (%)
    per_scenario: list[dict[str, Any]] = field(default_factory=list)


DEFAULT_PATH = Path("data/benchmarks/orancbench_v0_1.jsonl")


def load_scenarios(path: str | Path = DEFAULT_PATH) -> list[OrancBenchScenario]:
    """Load OrancBench scenarios from the committed JSONL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"OrancBench scenario file not found at {p}. "
            "Regenerate with:\n    python -m backend.scripts.gen_orancbench"
        )
    scenarios: list[OrancBenchScenario] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scenarios.append(OrancBenchScenario.from_dict(json.loads(line)))
    return scenarios


def _r2(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    mean = sum(y_true) / len(y_true)
    ss_tot = sum((y - mean) ** 2 for y in y_true)
    ss_res = sum((y - p) ** 2 for y, p in zip(y_true, y_pred))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def _mape(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    pct = []
    for y, p in zip(y_true, y_pred):
        if abs(y) < 1e-9:
            continue
        pct.append(abs((y - p) / y))
    return 100.0 * sum(pct) / max(1, len(pct))


def score_predictions(
    scenarios: Iterable[OrancBenchScenario],
    predictions: dict[str, dict[str, float]],
    *,
    kpis: tuple[str, ...] = ("impressions", "clicks", "conversions", "revenue"),
    by_difficulty: bool = True,
) -> dict[str, OrancBenchResult]:
    """Score a batch of predictions against OrancBench ground truth.

    ``predictions`` maps ``scenario_id -> {kpi_name: float}``. Missing
    scenario IDs are skipped. Returns a dict keyed by "overall" plus
    difficulty buckets ("easy", "medium", "hard") when ``by_difficulty``
    is true.
    """
    scenarios = list(scenarios)
    groups: dict[str, list[OrancBenchScenario]] = {"overall": scenarios}
    if by_difficulty:
        for d in ("easy", "medium", "hard"):
            groups[d] = [s for s in scenarios if s.difficulty == d]

    out: dict[str, OrancBenchResult] = {}
    for group_name, group in groups.items():
        if not group:
            continue
        y_true_per_kpi = {k: [] for k in kpis}
        y_pred_per_kpi = {k: [] for k in kpis}
        per_scenario: list[dict[str, Any]] = []
        for s in group:
            if s.scenario_id not in predictions:
                continue
            pred = predictions[s.scenario_id]
            entry = {"scenario_id": s.scenario_id, "difficulty": s.difficulty}
            for k in kpis:
                yt = float(s.ground_truth.get(k, 0.0))
                yp = float(pred.get(k, 0.0))
                y_true_per_kpi[k].append(yt)
                y_pred_per_kpi[k].append(yp)
                entry[f"{k}_true"] = yt
                entry[f"{k}_pred"] = yp
            per_scenario.append(entry)

        n = len(per_scenario)
        out[group_name] = OrancBenchResult(
            n=n,
            r2={k: _r2(y_true_per_kpi[k], y_pred_per_kpi[k]) for k in kpis},
            mape={k: _mape(y_true_per_kpi[k], y_pred_per_kpi[k]) for k in kpis},
            per_scenario=per_scenario,
        )
    return out

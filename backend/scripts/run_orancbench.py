"""Run a predictor against OrancBench v0.1 and report metrics.

By default uses the shipped LightGBM demo baseline
(`data/models/world_model_demo.pkl`) but accepts any Python callable via
``--predictor module:function`` — the function must take an
:class:`OrancBenchScenario` and return a dict of per-KPI predictions.

Usage:

    python -m backend.scripts.run_orancbench \\
        --data data/benchmarks/orancbench_v0_1.jsonl \\
        --predictor oransim.benchmarks.baselines:lightgbm_demo_predict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _baseline_predict_factory():
    """Load the shipped LightGBM demo pkl and return a predict callable."""
    import pickle
    import numpy as np
    import lightgbm as lgb

    pkl_path = Path("data/models/world_model_demo.pkl")
    if not pkl_path.exists():
        print(f"[error] baseline pkl not found at {pkl_path}. "
              "Train: python -m backend.scripts.train_lightgbm_demo", file=sys.stderr)
        sys.exit(3)
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)

    niches = blob["config"]["niches"]
    tiers = blob["config"]["kol_tiers"]
    boosters = {
        kpi: {float(q): lgb.Booster(model_str=s) for q, s in per_q.items()}
        for kpi, per_q in blob["boosters"].items()
    }

    def _predict(scn) -> dict[str, float]:
        niche_idx = niches.index(scn.niche) if scn.niche in niches else 0
        tier_idx = tiers.index(scn.kol_tier) if scn.kol_tier in tiers else 0
        x = np.asarray(
            [
                [
                    0.0,
                    float(niche_idx),
                    float(scn.budget),
                    float(scn.budget_bucket),
                    float(tier_idx),
                    float(scn.kol_fan_count),
                    float(scn.kol_engagement_rate),
                ]
            ],
            dtype=np.float32,
        )
        out = {}
        for kpi, qmap in boosters.items():
            out[kpi] = float(qmap[0.5].predict(x)[0])
        return out

    return _predict


def _load_custom(spec: str):
    """Resolve ``module:function`` to a callable."""
    module, _, func = spec.partition(":")
    if not func:
        raise ValueError(f"predictor spec must be 'module:function', got: {spec!r}")
    import importlib
    mod = importlib.import_module(module)
    return getattr(mod, func)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OrancBench v0.1.")
    parser.add_argument("--data", default="data/benchmarks/orancbench_v0_1.jsonl")
    parser.add_argument("--predictor", default=None,
                        help="Python path 'module:function'. Default: LightGBM demo baseline.")
    parser.add_argument("--out", default=None, help="Write detailed JSON results to this path.")
    args = parser.parse_args(argv)

    try:
        from oransim.benchmarks import load_scenarios, score_predictions
    except ImportError as e:
        print(f"[error] import: {e}", file=sys.stderr)
        return 2

    predictor = _load_custom(args.predictor) if args.predictor else _baseline_predict_factory()

    scenarios = load_scenarios(args.data)
    predictions = {s.scenario_id: predictor(s) for s in scenarios}
    results = score_predictions(scenarios, predictions)

    print()
    print(f"=== OrancBench v0.1 — {args.predictor or 'lightgbm_demo (baseline)'} ===")
    for group, r in results.items():
        print()
        print(f"  [{group}]  n = {r.n}")
        print(f"    KPI            |     R²    |   MAPE%")
        print(f"    ---------------|-----------|--------")
        for kpi in ("impressions", "clicks", "conversions", "revenue"):
            print(f"    {kpi:15s}| {r.r2[kpi]:+8.3f}  | {r.mape[kpi]:6.1f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {group: {"n": r.n, "r2": r.r2, "mape": r.mape, "per_scenario": r.per_scenario}
                 for group, r in results.items()},
                f, indent=2, ensure_ascii=False,
            )
        print(f"\n  detailed results → {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

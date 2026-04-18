"""Train the demo LightGBM quantile world model.

Reads the synthetic scenarios produced by ``gen_synthetic_data.py`` and fits
three P35/P50/P65 quantile regressors per KPI (impressions, clicks,
conversions, revenue), producing a compact pkl that ships with the OSS
release and powers the "clone → set LLM key → run" plug-and-play experience.

Feature vector (7-dim scalar, by design — keeps the pkl small and the
inference path CPU-only, zero-GPU):

    [platform_id, niche_idx, budget, budget_bucket, kol_tier_idx,
     kol_fan_count, kol_engagement_rate]

For the full 1600-dim feature pipeline with creative embeddings + KOL
audience + demographic + time-of-day tokens, see the Causal Transformer
training script ``train_transformer_wm.py`` (v0.2).

Usage:

    python -m backend.scripts.train_lightgbm_demo \\
        --data data/synthetic/scenarios_v0_1.jsonl \\
        --out  data/models/world_model_demo.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

NICHES = [
    "beauty",
    "fashion",
    "food",
    "electronics",
    "travel",
    "parenting",
    "fitness",
    "home",
    "beverage",
    "pet",
]
KOL_TIERS = ["nano", "micro", "mid", "macro", "mega"]
KPIS = ("impressions", "clicks", "conversions", "revenue")
QUANTILES = (0.35, 0.50, 0.65)


def _vectorize(row: dict) -> np.ndarray:
    niche_idx = NICHES.index(row["niche"]) if row["niche"] in NICHES else 0
    tier_idx = KOL_TIERS.index(row["kol_tier"]) if row["kol_tier"] in KOL_TIERS else 0
    return np.asarray(
        [
            float(row.get("platform_id", 0)),
            float(niche_idx),
            float(row["budget"]),
            float(row["budget_bucket"]),
            float(tier_idx),
            float(row["kol_fan_count"]),
            float(row["kol_engagement_rate"]),
        ],
        dtype=np.float32,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the demo LightGBM quantile world model.")
    parser.add_argument("--data", default="data/synthetic/scenarios_v0_1.jsonl")
    parser.add_argument("--out", default="data/models/world_model_demo.pkl")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        import lightgbm as lgb
    except ImportError:
        print("[error] lightgbm not installed. pip install lightgbm", file=sys.stderr)
        return 2

    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"[error] {data_path} not found. "
            "Generate synthetic data first:\n"
            "    python -m backend.scripts.gen_synthetic_data --out data/synthetic",
            file=sys.stderr,
        )
        return 3

    print(f"[train] loading {data_path} …")
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"[train]   loaded {len(rows)} scenarios")

    X = np.stack([_vectorize(r) for r in rows])
    y_all = {
        kpi: np.asarray([float(r["targets"][kpi]) for r in rows], dtype=np.float32) for kpi in KPIS
    }
    print(f"[train]   feature shape: {X.shape}")

    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(len(rows) * args.val_frac))
    idx = rng.permutation(len(rows))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    boosters: dict[str, dict[float, Any]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for kpi in KPIS:
        boosters[kpi] = {}
        y = y_all[kpi]
        y_val = y[val_idx]
        print(f"\n[train] === {kpi} ===")
        for q in QUANTILES:
            params = {
                "objective": "quantile",
                "alpha": float(q),
                "num_leaves": 2**args.max_depth - 1,
                "learning_rate": args.learning_rate,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 5,
                "min_child_samples": 10,
                "lambda_l2": 0.1,
                "max_depth": args.max_depth,
                "verbose": -1,
            }
            dtrain = lgb.Dataset(X[train_idx], y[train_idx])
            dval = lgb.Dataset(X[val_idx], y_val, reference=dtrain)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=args.n_estimators,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            boosters[kpi][q] = booster
            pred = booster.predict(X[val_idx])
            # Pinball loss on val
            diff = y_val - pred
            pinball = float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
            # R² on val
            ss_res = float(((y_val - pred) ** 2).sum())
            ss_tot = float(((y_val - y_val.mean()) ** 2).sum())
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            metrics.setdefault(kpi, {})[f"pinball_q{int(q*100)}"] = pinball
            metrics[kpi][f"r2_q{int(q*100)}"] = r2
            print(
                f"  q={q:.2f}  best_iter={booster.best_iteration}  pinball={pinball:.3f}  R²={r2:.3f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dumped = {
        "config": {
            "kpis": KPIS,
            "quantiles": QUANTILES,
            "feature_names": [
                "platform_id",
                "niche_idx",
                "budget",
                "budget_bucket",
                "kol_tier_idx",
                "kol_fan_count",
                "kol_engagement_rate",
            ],
            "niches": NICHES,
            "kol_tiers": KOL_TIERS,
            "feature_version": "demo_v1",
            "training_version": "0.1.1-alpha",
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        },
        "boosters": {
            kpi: {str(q): bst.model_to_string() for q, bst in per_q.items()}
            for kpi, per_q in boosters.items()
        },
        "metrics": metrics,
    }
    with open(out_path, "wb") as f:
        pickle.dump(dumped, f)
    size_kb = out_path.stat().st_size // 1024
    print(f"\n[train] saved {out_path} ({size_kb} KB)")
    print("[train] metrics summary:")
    for kpi, m in metrics.items():
        r2_med = m.get("r2_q50", 0.0)
        print(f"  {kpi:15s}  R²(P50)={r2_med:+.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

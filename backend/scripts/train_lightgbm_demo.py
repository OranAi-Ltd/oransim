"""Train the demo LightGBM quantile world model.

Reads the synthetic scenarios produced by ``gen_synthetic_data.py`` and fits
three P35/P50/P65 quantile regressors per KPI (impressions, clicks,
conversions, revenue), producing a compact pkl that ships with the OSS
release and powers the "clone → set LLM key → run" plug-and-play experience.

Feature vector (23-dim = 7 scalar + 16 text-embedding-PCA):

    scalar (7):       [platform_id, niche_idx, budget, budget_bucket,
                       kol_tier_idx, kol_fan_count, kol_engagement_rate]
    embedding (16):   PCA-reduced RealTextEmbedder output on a synthesized
                      creative brief caption ("春季 {niche} {tier} KOL ·
                      预算 {budget_bucket_label}"). Uses the fallback
                      hash-embedder when OPENAI_API_KEY is not set, so the
                      pipeline is reproducible + offline. The PCA components
                      ship inside the pkl and are applied at inference time
                      to the same caption template.

The embedding path is the same RealTextEmbedder / UEB that the rest of the
OSS stack uses — soul-agent persona matching, kol_content_match (T2-A2),
search_elasticity (T3-A6). Keeping the demo pkl consistent with that
pipeline was the point of upgrading feature_version from demo_v1 (tabular
only) to demo_v2 (tabular + PCA embedding).

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

# Allow running as either a script or an installed package entry-point
_THIS_DIR = Path(__file__).resolve().parent
_BACKEND = _THIS_DIR.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Single source of truth for caption logic — shared with the API's inference
# path so the pkl's PCA projection is applied to the same text it was
# trained on.
from oransim.scripts_helpers import _BUDGET_BUCKETS_ZH  # noqa: E402
from oransim.scripts_helpers import caption_for_demo_pkl as _caption_for

_BUDGET_BUCKETS = _BUDGET_BUCKETS_ZH

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

    X_scalar = np.stack([_vectorize(r) for r in rows])

    # Embed each scenario's synthetic caption via RealTextEmbedder (falls back
    # to deterministic hash embedder when OPENAI_API_KEY is not set). Then
    # PCA-reduce to EMB_PCA_DIM so the pkl stays small.
    print("[train] embedding synthetic captions …")
    from oransim.runtime.real_embedder import RealTextEmbedder

    embedder = RealTextEmbedder()
    captions = [_caption_for(r) for r in rows]
    emb_full = embedder.embed_batch(captions)  # [N, 1536]
    print(
        f"[train]   captions={len(captions)}  embed_shape={emb_full.shape}  "
        f"api_hits={embedder._api_hits}  fallback_hits={embedder._fallback_hits}"
    )

    from sklearn.decomposition import PCA

    EMB_PCA_DIM = 16
    pca = PCA(n_components=EMB_PCA_DIM, random_state=args.seed)
    emb_pca = pca.fit_transform(emb_full).astype(np.float32)
    print(
        f"[train]   PCA({EMB_PCA_DIM}) explained var = "
        f"{pca.explained_variance_ratio_.sum():.3f}"
    )

    X = np.concatenate([X_scalar, emb_pca], axis=1)
    y_all = {
        kpi: np.asarray([float(r["targets"][kpi]) for r in rows], dtype=np.float32) for kpi in KPIS
    }
    print(f"[train]   final feature shape: {X.shape}  (7 scalar + {EMB_PCA_DIM} embed-PCA)")

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
                *[f"caption_emb_pca_{i}" for i in range(EMB_PCA_DIM)],
            ],
            "niches": NICHES,
            "kol_tiers": KOL_TIERS,
            "budget_buckets_zh": _BUDGET_BUCKETS,
            "feature_version": "demo_v2",
            "training_version": "0.2.0-alpha",
            "embedding_model": embedder.model,
            "embedding_dim_raw": int(emb_full.shape[1]),
            "embedding_pca_dim": EMB_PCA_DIM,
            "embedding_pca_explained_var": float(pca.explained_variance_ratio_.sum()),
            "embedding_api_hits": embedder._api_hits,
            "embedding_fallback_hits": embedder._fallback_hits,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        },
        "pca": {
            "components": pca.components_.astype(np.float32),  # [EMB_PCA_DIM, 1536]
            "mean": pca.mean_.astype(np.float32),  # [1536]
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
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

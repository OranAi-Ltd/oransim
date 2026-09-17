"""Run the X5 budget-constrained targeting case study.

All learners share the same joint-stratified train/test split within a seed.
The script writes only aggregate metrics and split hashes; customer identifiers
never enter the result artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oransim.paths import REPO_ROOT  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "outputs/prelaunch/x5"
METHOD_ORDER = [
    "random_ranking",
    "two_model_logistic",
    "s_learner",
    "t_learner",
    "x_learner",
    "dr_learner",
    "causal_forest_dml",
]
METHOD_LABELS = {
    "random_ranking": "Random ranking",
    "two_model_logistic": "Two-model logistic",
    "s_learner": "S-Learner",
    "t_learner": "T-Learner",
    "x_learner": "X-Learner",
    "dr_learner": "DRLearner",
    "causal_forest_dml": "CausalForestDML",
}
SCORE_QUANTIZATION_DECIMALS = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_id_hash(ids: pd.Series) -> str:
    values = sorted(str(x) for x in ids)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    work = df.copy()
    feature_cols = [c for c in work.columns if c.startswith("X_")]
    for col in feature_cols:
        if "date" in col or "_ts" in col:
            dt = pd.to_datetime(work[col], errors="coerce")
            work[col] = dt.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    cat_cols = [c for c in feature_cols if not is_numeric_dtype(work[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    y = pd.to_numeric(work["outcome"], errors="raise").astype(int).to_numpy()
    t = pd.to_numeric(work["treatment"], errors="raise").astype(int).to_numpy()
    return work[feature_cols], y, t, num_cols, cat_cols


def preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=0.0,
    )


def uplift_at_fraction(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    fraction: float,
) -> tuple[float, int, int, int]:
    """Empirical treated-minus-control outcome rate in the top fraction."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n_top = max(1, int(np.floor(len(y) * fraction)))
    idx = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:n_top]
    yt = np.asarray(y)[idx]
    tt = np.asarray(treatment)[idx]
    treated = yt[tt == 1]
    control = yt[tt == 0]
    if not len(treated) or not len(control):
        raise ValueError("top subset must contain both treatment arms")
    uplift = float(treated.mean() - control.mean())
    return uplift, n_top, int(len(treated)), int(len(control))


def stratified_binary_bootstrap_ci(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    fraction: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Treatment/control-stratified nonparametric bootstrap for binary means."""
    n_top = max(1, int(np.floor(len(y) * fraction)))
    idx = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:n_top]
    yt = np.asarray(y)[idx]
    tt = np.asarray(treatment)[idx]
    treated = yt[tt == 1]
    control = yt[tt == 0]
    if not len(treated) or not len(control):
        raise ValueError("top subset must contain both treatment arms")
    rng = np.random.default_rng(seed)
    # For binary outcomes, resampling observations within an arm is exactly a
    # Binomial(n, empirical-rate) draw for the number of positive outcomes.
    boot_t = rng.binomial(len(treated), float(treated.mean()), size=samples) / len(treated)
    boot_c = rng.binomial(len(control), float(control.mean()), size=samples) / len(control)
    delta = boot_t - boot_c
    return float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))


def full_ranking_metrics(
    y: np.ndarray, treatment: np.ndarray, score: np.ndarray
) -> dict[str, float]:
    from sklift.metrics import qini_auc_score, uplift_auc_score

    return {
        "auuc": float(uplift_auc_score(y, score, treatment)),
        "qini": float(qini_auc_score(y, score, treatment)),
    }


@dataclass
class FittedScores:
    scores: dict[str, np.ndarray]
    configs: dict[str, dict[str, Any]]


def fit_all_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    t_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    forest_trees: int,
    causal_forest_trees: int,
) -> FittedScores:
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    from econml.metalearners import SLearner, TLearner, XLearner

    scores: dict[str, np.ndarray] = {}
    configs: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(seed + 90_001)
    scores["random_ranking"] = rng.random(len(x_test))
    configs["random_ranking"] = {"library": "numpy", "fit_rows": 0}

    logistic_t = LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
    logistic_c = LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
    logistic_t.fit(x_train[t_train == 1], y_train[t_train == 1])
    logistic_c.fit(x_train[t_train == 0], y_train[t_train == 0])
    scores["two_model_logistic"] = (
        logistic_t.predict_proba(x_test)[:, 1] - logistic_c.predict_proba(x_test)[:, 1]
    )
    configs["two_model_logistic"] = {
        "library": "sklearn.linear_model.LogisticRegression",
        "fit_rows": int(len(x_train)),
    }

    base_reg = RandomForestRegressor(
        n_estimators=forest_trees,
        min_samples_leaf=50,
        random_state=seed,
        n_jobs=-1,
    )
    base_clf = RandomForestClassifier(
        n_estimators=forest_trees,
        min_samples_leaf=50,
        random_state=seed,
        n_jobs=1,
    )
    serial_reg = clone(base_reg).set_params(n_jobs=1)

    learner = SLearner(overall_model=clone(base_reg))
    learner.fit(y_train, t_train, X=x_train)
    scores["s_learner"] = np.asarray(learner.effect(x_test)).ravel()
    configs["s_learner"] = {
        "library": "econml.metalearners.SLearner",
        "fit_rows": int(len(x_train)),
        "forest_trees": forest_trees,
    }

    learner = TLearner(models=clone(base_reg))
    learner.fit(y_train, t_train, X=x_train)
    scores["t_learner"] = np.asarray(learner.effect(x_test)).ravel()
    configs["t_learner"] = {
        "library": "econml.metalearners.TLearner",
        "fit_rows": int(len(x_train)),
        "forest_trees": forest_trees,
    }

    learner = XLearner(models=clone(serial_reg), cate_models=clone(serial_reg))
    learner.fit(y_train, t_train, X=x_train)
    scores["x_learner"] = np.asarray(learner.effect(x_test)).ravel()
    configs["x_learner"] = {
        "library": "econml.metalearners.XLearner",
        "fit_rows": int(len(x_train)),
        "forest_trees": forest_trees,
        "n_jobs": 1,
    }

    learner = DRLearner(
        model_propensity=LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed),
        model_regression=clone(serial_reg),
        model_final=clone(serial_reg),
        cv=3,
        random_state=seed,
    )
    learner.fit(y_train, t_train, X=x_train)
    scores["dr_learner"] = np.asarray(learner.effect(x_test)).ravel()
    configs["dr_learner"] = {
        "library": "econml.dr.DRLearner",
        "fit_rows": int(len(x_train)),
        "forest_trees": forest_trees,
        "n_jobs": 1,
    }

    learner = CausalForestDML(
        model_y=clone(base_clf),
        model_t=clone(base_clf),
        discrete_outcome=True,
        discrete_treatment=True,
        n_estimators=causal_forest_trees,
        min_samples_leaf=50,
        cv=3,
        random_state=seed,
        n_jobs=1,
    )
    with threadpool_limits(limits=1):
        learner.fit(y_train, t_train, X=x_train)
    scores["causal_forest_dml"] = np.asarray(learner.effect(x_test)).ravel()
    configs["causal_forest_dml"] = {
        "library": "econml.dml.CausalForestDML",
        "fit_rows": int(len(x_train)),
        "forest_trees": causal_forest_trees,
        "n_jobs": 1,
    }
    for method in scores:
        scores[method] = np.round(scores[method], SCORE_QUANTIZATION_DECIMALS)
        configs[method]["score_quantization_decimals"] = SCORE_QUANTIZATION_DECIMALS
    return FittedScores(scores=scores, configs=configs)


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def package_versions() -> dict[str, str]:
    import econml
    import lightgbm
    import matplotlib as mpl
    import sklearn
    import sklift

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scikit-uplift": sklift.__version__,
        "econml": econml.__version__,
        "lightgbm": lightgbm.__version__,
        "matplotlib": mpl.__version__,
    }


def git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def plot_curve(curve: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = plt.get_cmap("tab10")
    for i, method in enumerate(METHOD_ORDER):
        block = curve[curve["method"] == method]
        agg = (
            block.groupby("target_fraction", sort=True)["incremental_conversions_per_10000"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.plot(
            agg["target_fraction"] * 100,
            agg["mean"],
            marker="o",
            linewidth=1.8,
            label=METHOD_LABELS[method],
            color=colors(i),
        )
        ax.fill_between(
            agg["target_fraction"] * 100,
            agg["mean"] - agg["std"].fillna(0),
            agg["mean"] + agg["std"].fillna(0),
            color=colors(i),
            alpha=0.10,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Targeted customers (% of test set)")
    ax.set_ylabel("Estimated incremental conversions per 10,000 targeted")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "x5_budget_curve.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "x5_budget_curve.svg", bbox_inches="tight")
    plt.close(fig)


def write_case_study(summary: dict[str, Any], curve: pd.DataFrame, out_dir: Path) -> None:
    methods = summary["aggregate"]["methods"]
    q10 = {name: row["fractions"]["0.10"]["uplift"]["mean"] for name, row in methods.items()}
    q30 = {name: row["fractions"]["0.30"]["uplift"]["mean"] for name, row in methods.items()}
    best10 = max(q10, key=q10.get)
    best30 = max(q30, key=q30.get)
    best_auuc = max(methods, key=lambda name: methods[name]["auuc"]["mean"])
    lines = [
        "# Case Study: Budget-Constrained Targeting under Randomized Message Exposure",
        "",
        (
            "X5 RetailHero contains 200,039 customers assigned by an advertiser-randomized "
            "message experiment. We jointly stratified treatment and outcome and used the "
            "same 75/25 customer split for every learner within each of three seeds. "
            "Preprocessing was fitted on the training customers only."
        ),
        "",
        (
            f"At a 10% targeting budget, {METHOD_LABELS[best10]} produced the largest mean "
            f"uplift ({q10[best10]:.4f}), corresponding to an estimated "
            f"{q10[best10] * 10000:.1f} incremental conversions per 10,000 targeted "
            "customers. "
            f"At 30%, {METHOD_LABELS[best30]} led with mean uplift {q30[best30]:.4f} "
            f"({q30[best30] * 10000:.1f} per 10,000)."
        ),
        "",
        (
            f"{METHOD_LABELS[best_auuc]} attained the highest integrated ranking score "
            f"(AUUC {methods[best_auuc]['auuc']['mean']:.4f} ± "
            f"{methods[best_auuc]['auuc']['std']:.4f}; Qini "
            f"{methods[best_auuc]['qini']['mean']:.4f} ± "
            f"{methods[best_auuc]['qini']['std']:.4f}). The method selected for broad "
            "ranking quality can therefore differ from the method selected under a fixed "
            "contact budget."
        ),
        "",
        (
            "The estimates compare observed treated and control outcomes among customers "
            "ranked at each cutoff. Individual treatment effects remain unobserved, so the "
            "case supports budget-dependent ranking decisions rather than pointwise CATE "
            "accuracy."
        ),
    ]
    (out_dir / "x5_case_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    data_path = args.data.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data_path)
    required = {"client_id", "treatment", "outcome"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")

    x, y, treatment, num_cols, cat_cols = prepare_features(df)
    strat = (
        pd.Series(treatment, index=df.index).astype(str)
        + "_"
        + pd.Series(y, index=df.index).astype(str)
    )
    curve_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    method_configs: dict[str, dict[str, Any]] = {}

    for seed in args.seeds:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.25,
            random_state=seed,
            stratify=strat,
        )
        prep = preprocessor(num_cols, cat_cols)
        x_train = prep.fit_transform(x.iloc[train_idx])
        x_test = prep.transform(x.iloc[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        t_train, t_test = treatment[train_idx], treatment[test_idx]
        fitted = fit_all_scores(
            x_train,
            y_train,
            t_train,
            x_test,
            seed=seed,
            forest_trees=args.forest_trees,
            causal_forest_trees=args.causal_forest_trees,
        )
        method_configs = fitted.configs
        split_info = {
            "seed": int(seed),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_index_hash": stable_id_hash(df.iloc[train_idx]["client_id"]),
            "test_index_hash": stable_id_hash(df.iloc[test_idx]["client_id"]),
            "train_joint_counts": {
                f"t{arm}_y{outcome}": int(np.sum((t_train == arm) & (y_train == outcome)))
                for arm in (0, 1)
                for outcome in (0, 1)
            },
            "test_joint_counts": {
                f"t{arm}_y{outcome}": int(np.sum((t_test == arm) & (y_test == outcome)))
                for arm in (0, 1)
                for outcome in (0, 1)
            },
            "methods": {},
        }
        for method in METHOD_ORDER:
            score = fitted.scores[method]
            ranking = full_ranking_metrics(y_test, t_test, score)
            method_seed = {"auuc": ranking["auuc"], "qini": ranking["qini"], "fractions": {}}
            for frac in args.target_fractions:
                uplift, n_top, n_t, n_c = uplift_at_fraction(y_test, t_test, score, frac)
                ci_low, ci_high = stratified_binary_bootstrap_ci(
                    y_test,
                    t_test,
                    score,
                    frac,
                    samples=args.bootstrap_samples,
                    seed=seed * 100_000 + METHOD_ORDER.index(method) * 1_000 + int(frac * 100),
                )
                row = {
                    "seed": int(seed),
                    "method": method,
                    "target_fraction": float(frac),
                    "n_targeted": n_top,
                    "n_treated": n_t,
                    "n_control": n_c,
                    "uplift": uplift,
                    "uplift_ci95_low": ci_low,
                    "uplift_ci95_high": ci_high,
                    "incremental_conversions_per_10000": uplift * 10_000,
                    "incremental_conversions_per_10000_ci95_low": ci_low * 10_000,
                    "incremental_conversions_per_10000_ci95_high": ci_high * 10_000,
                    "cumulative_gain": uplift * n_top,
                    "auuc": ranking["auuc"],
                    "qini": ranking["qini"],
                    "train_index_hash": split_info["train_index_hash"],
                    "test_index_hash": split_info["test_index_hash"],
                }
                curve_rows.append(row)
                method_seed["fractions"][f"{frac:.2f}"] = {
                    key: row[key]
                    for key in (
                        "n_targeted",
                        "n_treated",
                        "n_control",
                        "uplift",
                        "uplift_ci95_low",
                        "uplift_ci95_high",
                        "incremental_conversions_per_10000",
                        "cumulative_gain",
                    )
                }
            split_info["methods"][method] = method_seed
        seed_summaries.append(split_info)

    curve = pd.DataFrame(curve_rows)
    curve.to_csv(out_dir / "x5_budget_curve.csv", index=False, float_format="%.12g")
    aggregate: dict[str, Any] = {"methods": {}}
    for method in METHOD_ORDER:
        block = curve[curve["method"] == method]
        method_row: dict[str, Any] = {
            "auuc": mean_std(block.groupby("seed", sort=True)["auuc"].first().tolist()),
            "qini": mean_std(block.groupby("seed", sort=True)["qini"].first().tolist()),
            "fractions": {},
        }
        for frac in args.target_fractions:
            fblock = block[np.isclose(block["target_fraction"], frac)]
            method_row["fractions"][f"{frac:.2f}"] = {
                "uplift": mean_std(fblock["uplift"].tolist()),
                "incremental_conversions_per_10000": mean_std(
                    fblock["incremental_conversions_per_10000"].tolist()
                ),
                "cumulative_gain": mean_std(fblock["cumulative_gain"].tolist()),
            }
        aggregate["methods"][method] = method_row

    summary = {
        "study": "x5_budget_constrained_targeting",
        "dataset": "X5 RetailHero advertiser-randomized A/B experiment",
        "rows": int(len(df)),
        "treatment_rate": float(treatment.mean()),
        "outcome_rate": float(y.mean()),
        "input_sha256": sha256_file(data_path),
        "split": "joint treatment-outcome stratified 75/25",
        "seeds": [int(s) for s in args.seeds],
        "target_fractions": [float(x) for x in args.target_fractions],
        "bootstrap": {
            "scheme": "treatment/control-stratified nonparametric bootstrap within each ranked subset",
            "samples": int(args.bootstrap_samples),
            "interval": "percentile 95%",
        },
        "method_configs": method_configs,
        "seed_runs": seed_summaries,
        "aggregate": aggregate,
        "privacy": {
            "customer_ids_written": False,
            "split_identifiers": "SHA-256 of sorted customer IDs",
        },
    }
    (out_dir / "x5_budget_case_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_curve(curve, out_dir)
    write_case_study(summary, curve, out_dir)
    command = " ".join([sys.executable, *sys.argv])
    log = {
        "command": command,
        "git_commit": git_commit(),
        "input": str(data_path),
        "input_sha256": summary["input_sha256"],
        "packages": package_versions(),
        "wall_seconds": round(time.time() - started, 3),
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "x5_reproduction.log").write_text(
        json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "x5" / "processed" / "uplift.parquet",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 256])
    parser.add_argument(
        "--target-fractions",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.50],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--forest-trees", type=int, default=80)
    parser.add_argument("--causal-forest-trees", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.causal_forest_trees % 4:
        raise SystemExit("--causal-forest-trees must be divisible by 4")
    if args.bootstrap_samples < 10:
        raise SystemExit("--bootstrap-samples must be at least 10")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

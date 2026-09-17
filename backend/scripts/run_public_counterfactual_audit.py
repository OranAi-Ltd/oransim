"""Audit factual, counterfactual, ITE, and CATE recovery on public synthetic data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oransim.paths import REPO_ROOT  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "outputs/prelaunch/public_cf"
KPIS = ("impressions", "clicks", "conversions", "revenue")
METHODS = (
    "global_mean",
    "arm_mean",
    "lightgbm_substitution",
    "t_learner",
    "causal_forest_dml",
)
CF_BUDGET_MULTIPLIER = {0: 0.6, 1: 1.0, 2: 1.4, 3: 1.8}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def generate_data(source_repo: Path, n: int, seed: int, target_dir: Path) -> Path:
    cmd = [
        sys.executable,
        str(source_repo / "backend" / "scripts" / "gen_synthetic_data.py"),
        "--out",
        str(target_dir),
        "--what",
        "scenarios",
        "--n-kols",
        "1000",
        "--n-scenarios",
        str(n),
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, cwd=source_repo, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr or proc.stdout)
    path = target_dir / "scenarios_v0_1.jsonl"
    if not path.exists():
        raise RuntimeError("generator did not create scenarios_v0_1.jsonl")
    return path


def load_rows(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    flat = []
    for row in records:
        item = {
            key: row[key]
            for key in (
                "scenario_id",
                "platform_id",
                "niche",
                "budget",
                "budget_bucket",
                "kol_tier",
                "kol_fan_count",
                "kol_engagement_rate",
                "treatment_arm",
                "cf_arm",
            )
        }
        item["cf_budget"] = float(row["budget"]) * CF_BUDGET_MULTIPLIER[int(row["cf_arm"])]
        for kpi in KPIS:
            item[f"factual_{kpi}"] = float(row["targets"][kpi])
            item[f"counterfactual_{kpi}"] = float(row["cf_targets"][kpi])
        flat.append(item)
    return pd.DataFrame(flat).sort_values("scenario_id", kind="stable").reset_index(drop=True)


def split_by_scenario_id(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frame) < 10:
        raise ValueError("at least 10 scenarios are required")
    n = len(frame)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = frame.iloc[:n_train].copy()
    validation = frame.iloc[n_train : n_train + n_val].copy()
    test = frame.iloc[n_train + n_val :].copy()
    ids = [set(block["scenario_id"]) for block in (train, validation, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise AssertionError("scenario split overlap")
    return train, validation, test


def covariates(frame: pd.DataFrame, arm: pd.Series, budget: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "platform_id": frame["platform_id"].astype(str),
            "niche": frame["niche"].astype(str),
            "budget_log": np.log1p(budget.astype(float)),
            "budget_bucket": frame["budget_bucket"].astype(str),
            "kol_tier": frame["kol_tier"].astype(str),
            "kol_fans_log": np.log1p(frame["kol_fan_count"].astype(float)),
            "kol_engagement_rate": frame["kol_engagement_rate"].astype(float),
            "arm": arm.astype(str),
        },
        index=frame.index,
    )


def feature_preprocessor(include_arm: bool) -> ColumnTransformer:
    numeric = ["budget_log", "kol_fans_log", "kol_engagement_rate"]
    categorical = ["platform_id", "niche", "budget_bucket", "kol_tier"]
    if include_arm:
        categorical.append("arm")
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )


def lgbm(seed: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=32,
        min_child_samples=20,
        reg_alpha=0.3,
        reg_lambda=0.3,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


class PotentialOutcomePredictor:
    def predict(self, frame: pd.DataFrame, arm: pd.Series, budget: pd.Series) -> np.ndarray:
        raise NotImplementedError


class GlobalMeanPredictor(PotentialOutcomePredictor):
    def __init__(self, y_log: np.ndarray):
        self.mean = y_log.mean(axis=0)

    def predict(self, frame: pd.DataFrame, arm: pd.Series, budget: pd.Series) -> np.ndarray:
        return np.tile(self.mean, (len(frame), 1))


class ArmMeanPredictor(PotentialOutcomePredictor):
    def __init__(self, treatment: np.ndarray, y_log: np.ndarray):
        self.global_mean = y_log.mean(axis=0)
        self.means = {
            value: y_log[treatment == value].mean(axis=0) for value in sorted(set(treatment))
        }

    def predict(self, frame: pd.DataFrame, arm: pd.Series, budget: pd.Series) -> np.ndarray:
        return np.stack([self.means.get(int(value), self.global_mean) for value in arm], axis=0)


class GlobalLightGBMPredictor(PotentialOutcomePredictor):
    def __init__(self, train: pd.DataFrame, y_log: np.ndarray, seed: int):
        self.pre = feature_preprocessor(include_arm=True)
        x = covariates(train, train["treatment_arm"], train["budget"])
        tx = self.pre.fit_transform(x)
        self.models = []
        for k in range(len(KPIS)):
            model = lgbm(seed)
            model.fit(tx, y_log[:, k])
            self.models.append(model)

    def predict(self, frame: pd.DataFrame, arm: pd.Series, budget: pd.Series) -> np.ndarray:
        x = self.pre.transform(covariates(frame, arm, budget))
        return np.stack([model.predict(x) for model in self.models], axis=1)


class TLearnerPredictor(PotentialOutcomePredictor):
    def __init__(self, train: pd.DataFrame, y_log: np.ndarray, seed: int):
        self.pre = feature_preprocessor(include_arm=False)
        x = covariates(train, train["treatment_arm"], train["budget"]).drop(columns=["arm"])
        tx = self.pre.fit_transform(x)
        treatment = train["treatment_arm"].to_numpy(dtype=int)
        self.models: dict[int, list[Any]] = {}
        for arm in sorted(set(treatment)):
            mask = treatment == arm
            self.models[arm] = []
            for k in range(len(KPIS)):
                model = lgbm(seed)
                model.fit(tx[mask], y_log[mask, k])
                self.models[arm].append(model)

    def predict(self, frame: pd.DataFrame, arm: pd.Series, budget: pd.Series) -> np.ndarray:
        x = covariates(frame, arm, budget).drop(columns=["arm"])
        tx = self.pre.transform(x)
        out = np.zeros((len(frame), len(KPIS)), dtype=float)
        arm_values = arm.to_numpy(dtype=int)
        for arm_value, models in self.models.items():
            mask = arm_values == arm_value
            if mask.any():
                out[mask] = np.stack([model.predict(tx[mask]) for model in models], axis=1)
        return out


class CausalForestPredictor(PotentialOutcomePredictor):
    """One-vs-reference CausalForestDML potential-outcome adapter."""

    def __init__(
        self,
        train: pd.DataFrame,
        y_log: np.ndarray,
        seed: int,
        n_estimators: int,
    ):
        from econml.dml import CausalForestDML

        self.pre = feature_preprocessor(include_arm=False)
        x = covariates(train, train["treatment_arm"], train["budget"]).drop(columns=["arm"])
        tx = self.pre.fit_transform(x)
        treatment = train["treatment_arm"].to_numpy(dtype=int)
        self.reference_models = []
        ref = treatment == 0
        for k in range(len(KPIS)):
            model = lgbm(seed)
            model.fit(tx[ref], y_log[ref, k])
            self.reference_models.append(model)
        self.forests: dict[int, Any] = {}
        for arm in (1, 2, 3):
            mask = np.isin(treatment, [0, arm])
            binary = (treatment[mask] == arm).astype(int)
            forest = CausalForestDML(
                model_y=RandomForestRegressor(
                    n_estimators=60,
                    min_samples_leaf=20,
                    random_state=seed,
                    n_jobs=-1,
                ),
                model_t=LogisticRegression(max_iter=500, random_state=seed),
                discrete_treatment=True,
                n_estimators=n_estimators,
                min_samples_leaf=20,
                cv=3,
                random_state=seed,
                n_jobs=-1,
            )
            forest.fit(y_log[mask], binary, X=tx[mask])
            self.forests[arm] = forest

    def predict(self, frame: pd.DataFrame, arm: pd.Series, budget: pd.Series) -> np.ndarray:
        x = covariates(frame, arm, budget).drop(columns=["arm"])
        tx = self.pre.transform(x)
        ref = np.stack([model.predict(tx) for model in self.reference_models], axis=1)
        out = ref.copy()
        values = arm.to_numpy(dtype=int)
        for arm_value, forest in self.forests.items():
            mask = values == arm_value
            if mask.any():
                effect = np.asarray(forest.effect(tx[mask]))
                if effect.ndim == 1:
                    effect = effect[:, None]
                out[mask] = ref[mask] + effect
        return out


def build_predictors(
    train: pd.DataFrame,
    seed: int,
    causal_forest_trees: int,
) -> tuple[dict[str, PotentialOutcomePredictor], dict[str, dict[str, Any]]]:
    y_raw = train[[f"factual_{k}" for k in KPIS]].to_numpy(dtype=float)
    y_log = np.log1p(y_raw)
    treatment = train["treatment_arm"].to_numpy(dtype=int)
    predictors: dict[str, PotentialOutcomePredictor] = {
        "global_mean": GlobalMeanPredictor(y_log),
        "arm_mean": ArmMeanPredictor(treatment, y_log),
        "lightgbm_substitution": GlobalLightGBMPredictor(train, y_log, seed),
        "t_learner": TLearnerPredictor(train, y_log, seed),
    }
    status = {name: {"status": "completed"} for name in predictors}
    try:
        predictors["causal_forest_dml"] = CausalForestPredictor(
            train, y_log, seed, causal_forest_trees
        )
        status["causal_forest_dml"] = {
            "status": "completed",
            "design": "one-vs-arm-0 CausalForestDML with an arm-0 outcome anchor",
            "n_estimators": causal_forest_trees,
        }
    except Exception as exc:
        status["causal_forest_dml"] = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return predictors, status


def metric_rows(
    method: str,
    truth_f: np.ndarray,
    truth_cf: np.ndarray,
    pred_f: np.ndarray,
    pred_cf: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    truth_effect = truth_cf - truth_f
    pred_effect = pred_cf - pred_f
    for k, kpi in enumerate(KPIS):
        denom = float(np.mean(np.abs(truth_cf[:, k]))) + 1e-12
        nonzero = np.abs(truth_effect[:, k]) > 1e-12
        sign_accuracy = (
            float(np.mean(np.sign(pred_effect[nonzero, k]) == np.sign(truth_effect[nonzero, k])))
            if nonzero.any()
            else float("nan")
        )
        corr = spearmanr(truth_effect[:, k], pred_effect[:, k], nan_policy="omit").statistic
        rows.append(
            {
                "method": method,
                "kpi": kpi,
                "factual_r2": float(r2_score(truth_f[:, k], pred_f[:, k])),
                "factual_mae": float(mean_absolute_error(truth_f[:, k], pred_f[:, k])),
                "counterfactual_mae": float(mean_absolute_error(truth_cf[:, k], pred_cf[:, k])),
                "counterfactual_normalized_mae": float(
                    mean_absolute_error(truth_cf[:, k], pred_cf[:, k]) / denom
                ),
                "ite_mae": float(mean_absolute_error(truth_effect[:, k], pred_effect[:, k])),
                "ite_sign_accuracy": sign_accuracy,
                "ite_rank_correlation": float(corr) if np.isfinite(corr) else float("nan"),
            }
        )
    return rows


def cate_rows(
    method: str,
    test: pd.DataFrame,
    truth_effect: np.ndarray,
    pred_effect: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    dimensions = {
        "budget_bucket": test["budget_bucket"].astype(str),
        "kol_tier": test["kol_tier"].astype(str),
        "niche": test["niche"].astype(str),
        "arm_x_kol_tier": (
            test["treatment_arm"].astype(str)
            + "->"
            + test["cf_arm"].astype(str)
            + "|"
            + test["kol_tier"].astype(str)
        ),
    }
    for dim, labels in dimensions.items():
        for value in sorted(labels.unique()):
            mask = (labels == value).to_numpy()
            for k, kpi in enumerate(KPIS):
                rows.append(
                    {
                        "method": method,
                        "dimension": dim,
                        "segment": value,
                        "kpi": kpi,
                        "n": int(mask.sum()),
                        "true_cate": float(truth_effect[mask, k].mean()),
                        "predicted_cate": float(pred_effect[mask, k].mean()),
                    }
                )
    return rows


def add_cate_r2(metrics: pd.DataFrame, cate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, kpi), block in cate.groupby(["method", "kpi"], sort=False):
        eligible = block[block["dimension"].isin(["budget_bucket", "kol_tier", "niche"])]
        score = (
            float(r2_score(eligible["true_cate"], eligible["predicted_cate"]))
            if len(eligible) >= 2
            else float("nan")
        )
        rows.append({"method": method, "kpi": kpi, "segment_cate_r2": score})
    return metrics.merge(pd.DataFrame(rows), on=["method", "kpi"], how="left")


def plot_cate(cate: pd.DataFrame, out_dir: Path) -> None:
    block = cate[cate["dimension"] == "arm_x_kol_tier"]
    methods = list(block["method"].drop_duplicates())
    display_methods = []
    omitted_methods = []
    for method in methods:
        diverged = False
        for kpi in KPIS:
            part = block[(block["method"] == method) & (block["kpi"] == kpi)]
            truth_scale = max(1.0, float(part["true_cate"].abs().max()))
            if float(part["predicted_cate"].abs().max()) > 100 * truth_scale:
                diverged = True
                break
        (omitted_methods if diverged else display_methods).append(method)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    for ax, kpi in zip(axes.ravel(), KPIS, strict=False):
        kblock = block[block["kpi"] == kpi]
        shown = kblock[kblock["method"].isin(display_methods)]
        for method in display_methods:
            part = kblock[kblock["method"] == method]
            ax.scatter(
                part["true_cate"],
                part["predicted_cate"],
                s=18,
                alpha=0.70,
                label=method,
            )
        lo = min(shown["true_cate"].min(), shown["predicted_cate"].min())
        hi = max(shown["true_cate"].max(), shown["predicted_cate"].max())
        ax.plot([lo, hi], [lo, hi], color="#555555", linewidth=0.8)
        ax.set_title(kpi)
        ax.set_xlabel("True segment CATE")
        ax.set_ylabel("Predicted segment CATE")
        ax.grid(alpha=0.20)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    if omitted_methods:
        fig.text(
            0.5,
            0.012,
            "Omitted from axes after >100x scale divergence: "
            + ", ".join(omitted_methods)
            + "; complete metrics remain in the CSV.",
            ha="center",
            fontsize=7.5,
        )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out_dir / "public_cf_cate_calibration.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "public_cf_cate_calibration.svg", bbox_inches="tight")
    plt.close(fig)


def plot_effect_recovery(predictions: pd.DataFrame, out_dir: Path) -> None:
    methods = [m for m in METHODS if f"{m}_effect_revenue" in predictions.columns]
    truth = predictions["true_effect_revenue"].to_numpy()
    truth_scale = max(1.0, float(np.max(np.abs(truth))))
    display_methods = [
        method
        for method in methods
        if float(np.max(np.abs(predictions[f"{method}_effect_revenue"]))) <= 100 * truth_scale
    ]
    omitted_methods = [method for method in methods if method not in display_methods]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    order = np.argsort(truth)
    bins = np.array_split(order, 20)
    x = np.array([truth[idx].mean() for idx in bins])
    ax.plot(x, x, color="#555555", linewidth=1.0, label="Ideal")
    for method in display_methods:
        values = predictions[f"{method}_effect_revenue"].to_numpy()
        y = np.array([values[idx].mean() for idx in bins])
        ax.plot(x, y, marker="o", markersize=3, label=method)
    ax.set_xlabel("True revenue effect (test-bin mean)")
    ax.set_ylabel("Predicted revenue effect")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    if omitted_methods:
        ax.text(
            0.01,
            0.02,
            "Omitted after >100x scale divergence: "
            + ", ".join(omitted_methods)
            + "; see complete metrics.",
            transform=ax.transAxes,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "public_cf_effect_recovery.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "public_cf_effect_recovery.svg", bbox_inches="tight")
    plt.close(fig)


def package_versions() -> dict[str, str]:
    import econml
    import lightgbm
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "lightgbm": lightgbm.__version__,
        "econml": econml.__version__,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_repo = args.source_repo.resolve() if args.source_repo else None
    source_status_before: str | None = None
    source_commit = args.source_commit or "provided-data"
    if args.data is None:
        if source_repo is None:
            raise SystemExit("provide --data or --source-repo")
        source_status_before = git_output(source_repo, "status", "--short")
        if source_status_before:
            raise SystemExit(
                f"source repository must be clean before read-only audit:\n{source_status_before}"
            )
        source_commit = git_output(source_repo, "rev-parse", "HEAD")

    with tempfile.TemporaryDirectory(prefix="oransim-aaai27-public-cf-") as temp_name:
        temp_dir = Path(temp_name)
        data_path = (
            args.data.resolve()
            if args.data
            else generate_data(source_repo, args.n_scenarios, args.seed, temp_dir)
        )
        input_sha = sha256_file(data_path)
        frame = load_rows(data_path)
        train, validation, test = split_by_scenario_id(frame)
        predictors, method_status = build_predictors(train, args.seed, args.causal_forest_trees)
        truth_f = test[[f"factual_{k}" for k in KPIS]].to_numpy(dtype=float)
        truth_cf = test[[f"counterfactual_{k}" for k in KPIS]].to_numpy(dtype=float)
        predictions = pd.DataFrame({"scenario_id": test["scenario_id"].to_numpy()})
        for k, kpi in enumerate(KPIS):
            predictions[f"true_factual_{kpi}"] = truth_f[:, k]
            predictions[f"true_counterfactual_{kpi}"] = truth_cf[:, k]
            predictions[f"true_effect_{kpi}"] = truth_cf[:, k] - truth_f[:, k]
        metrics_rows = []
        all_cate_rows = []
        identity_checks = {}
        for method, predictor in predictors.items():
            factual_log = predictor.predict(test, test["treatment_arm"], test["budget"])
            counterfactual_log = predictor.predict(test, test["cf_arm"], test["cf_budget"])
            identity_log = factual_log
            identity_checks[method] = float(np.max(np.abs(identity_log - factual_log)))
            pred_f = np.maximum(0.0, np.expm1(factual_log))
            pred_cf = np.maximum(0.0, np.expm1(counterfactual_log))
            metrics_rows.extend(metric_rows(method, truth_f, truth_cf, pred_f, pred_cf))
            effect = pred_cf - pred_f
            all_cate_rows.extend(cate_rows(method, test, truth_cf - truth_f, effect))
            for k, kpi in enumerate(KPIS):
                predictions[f"{method}_factual_{kpi}"] = pred_f[:, k]
                predictions[f"{method}_counterfactual_{kpi}"] = pred_cf[:, k]
                predictions[f"{method}_effect_{kpi}"] = effect[:, k]

        metrics = add_cate_r2(pd.DataFrame(metrics_rows), pd.DataFrame(all_cate_rows))
        cate = pd.DataFrame(all_cate_rows)
        metrics.to_csv(out_dir / "public_cf_metrics.csv", index=False, float_format="%.12g")
        predictions.to_csv(out_dir / "public_cf_predictions.csv", index=False, float_format="%.12g")
        cate.to_csv(out_dir / "public_cf_cate_segments.csv", index=False, float_format="%.12g")
        plot_cate(cate, out_dir)
        plot_effect_recovery(predictions, out_dir)

    source_status_after: str | None = None
    if source_repo is not None and args.data is None:
        source_status_after = git_output(source_repo, "status", "--short")
        if source_status_after:
            raise SystemExit(
                f"source repository changed during read-only audit:\n{source_status_after}"
            )
    summary = {
        "study": "public_synthetic_paired_counterfactual_audit",
        "source_repository": str(source_repo) if source_repo is not None else "provided-data",
        "source_commit": source_commit,
        "source_clean_before": (
            source_status_before == "" if source_status_before is not None else None
        ),
        "source_clean_after": (
            source_status_after == "" if source_status_after is not None else None
        ),
        "generator": {
            "seed": int(args.seed),
            "n_scenarios": int(len(frame)),
            "input_sha256": input_sha,
            "counterfactual_budget_rule": {
                str(key): value for key, value in CF_BUDGET_MULTIPLIER.items()
            },
        },
        "split": {
            "policy": "scenario_id lexical order 80/10/10",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "train_validation_overlap": 0,
            "train_test_overlap": 0,
            "validation_test_overlap": 0,
        },
        "methods": {
            **method_status,
            "oransim_counterfactual_head": {
                "status": "deferred",
                "reason": "the public source repository contains architecture code but no trained CausalTransformer counterfactual-head checkpoint",
            },
        },
        "overlap_diagnostic": {
            "conditional_overlap_available": False,
            "reason": "treatment_arm is a deterministic function of budget_bucket and KOL tier in the public generator",
            "observed_consequence": "T-Learner extrapolates across unsupported arms and CausalForestDML diverges; complete failure metrics are retained",
        },
        "identity_intervention_policy": "reuse the same computed potential outcome for identical arm and budget inputs",
        "identity_intervention_max_abs_effect_log_scale": identity_checks,
        "metrics": metrics.replace({np.nan: None}).to_dict(orient="records"),
        "cate_primary_interpretation": "segment means because factual and counterfactual generator paths contain independent noise",
        "privacy": {
            "synthetic_rows_only": True,
            "prediction_identifiers": "synthetic scenario_id only",
            "private_data_used": False,
        },
    }
    (out_dir / "public_cf_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reproduction = {
        "command": " ".join([sys.executable, *sys.argv]),
        "source_commit": source_commit,
        "generator_seed": args.seed,
        "input_sha256": input_sha,
        "packages": package_versions(),
        "wall_seconds": round(time.time() - started, 3),
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "public_cf_reproduction.log").write_text(
        json.dumps(reproduction, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument(
        "--source-commit",
        default="",
        help="Optional provenance label when --data is supplied directly.",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--n-scenarios", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--causal-forest-trees", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.causal_forest_trees % 4:
        raise SystemExit("--causal-forest-trees must be divisible by 4")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

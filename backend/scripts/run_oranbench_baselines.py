"""Run OranBench sprint baselines and write JSON result artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oransim.paths import DATA_DIR, REPO_ROOT

RESULTS_DIR = REPO_ROOT / "outputs/prelaunch/protocols"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[oranbench] wrote {path}")


def ci_mean(values: np.ndarray, alpha: float = 1.96) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    if len(values) <= 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "stderr": 0.0}
    stderr = float(np.nanstd(values, ddof=1) / math.sqrt(len(values)))
    return {
        "mean": mean,
        "ci95_low": mean - alpha * stderr,
        "ci95_high": mean + alpha * stderr,
        "stderr": stderr,
    }


def bootstrap_ratio(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    numerator_values = np.asarray(numerator_values, dtype=float)
    denominator_values = np.asarray(denominator_values, dtype=float)
    if len(numerator_values) != len(denominator_values):
        raise ValueError("numerator_values and denominator_values must have the same length")
    if len(numerator_values) == 0:
        return {
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "stderr": float("nan"),
            "bootstrap_samples": 0,
        }

    denom = float(np.sum(denominator_values))
    mean = float(np.sum(numerator_values) / denom) if denom != 0 else float("nan")
    if n_bootstrap <= 1 or len(numerator_values) == 1:
        return {
            "mean": mean,
            "ci95_low": mean,
            "ci95_high": mean,
            "stderr": 0.0,
            "bootstrap_samples": int(max(n_bootstrap, 0)),
        }

    n = len(numerator_values)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_denom = float(np.sum(denominator_values[idx]))
        samples[i] = (
            float(np.sum(numerator_values[idx]) / boot_denom) if boot_denom != 0 else np.nan
        )
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return {
            "mean": mean,
            "ci95_low": mean,
            "ci95_high": mean,
            "stderr": float("nan"),
            "bootstrap_samples": int(n_bootstrap),
        }
    return {
        "mean": mean,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "stderr": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        "bootstrap_samples": int(n_bootstrap),
    }


def ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights**2))
    if denom <= 0:
        return 0.0
    return float((np.sum(weights) ** 2) / denom)


def c1_kuairand(args: argparse.Namespace) -> dict[str, Any]:
    path = DATA_DIR / "kuairand" / "processed" / "clicks.parquet"
    df = pd.read_parquet(path)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sort_values("timestamp").iloc[: args.max_rows].copy()

    reward_col = args.kuairand_reward
    reward = pd.to_numeric(df[reward_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if reward_col == "reward_watch_time":
        reward = np.log1p(reward)
    p0 = pd.to_numeric(df["propensity"], errors="coerce").to_numpy(dtype=float)

    n_items = int(df["item_id"].nunique())
    top_k = 10
    epsilon = 0.1
    item_popularity = df.groupby("item_id").size().sort_values(ascending=False)
    greedy_items = list(item_popularity.head(top_k).index)
    is_greedy = df["item_id"].isin(greedy_items).to_numpy()
    target_prob = np.where(
        is_greedy, epsilon / n_items + (1.0 - epsilon) / top_k, epsilon / n_items
    )
    weights = target_prob / p0
    logging_mean = ci_mean(reward)
    ips_values = weights * reward
    ips = ci_mean(ips_values)
    bootstrap_rng = np.random.default_rng(int(args.seed) + 10_001)
    snips = bootstrap_ratio(weights * reward, weights, args.bootstrap_samples, bootstrap_rng)
    snips_mean = float(snips["mean"])

    clip_grid = [1, 2, 5, 10, 20, 50]
    clipping_curve = []
    for clip in clip_grid:
        wc = np.minimum(weights, clip)
        clipping_curve.append(
            {
                "clip": clip,
                "snips": float(np.sum(wc * reward) / np.sum(wc)),
                "ips": float(np.mean(wc * reward)),
                "ess": ess(wc),
            }
        )

    alpha = 20.0
    global_mean = float(np.mean(reward))
    item_stats = (
        pd.DataFrame({"item_id": df["item_id"].to_numpy(), "reward": reward})
        .groupby("item_id")["reward"]
        .agg(["sum", "count"])
    )
    item_mean = (item_stats["sum"] + alpha * global_mean) / (item_stats["count"] + alpha)
    observed_items = pd.Index(df["item_id"].unique())
    target_prob_by_item = pd.Series(epsilon / n_items, index=observed_items, dtype=float)
    target_prob_by_item.loc[target_prob_by_item.index.isin(greedy_items)] += (1.0 - epsilon) / top_k
    model_reward_by_item = item_mean.reindex(observed_items).fillna(global_mean)
    direct_method_value = float(
        np.sum(target_prob_by_item.to_numpy() * model_reward_by_item.to_numpy())
    )
    logged_model_reward = item_mean.reindex(df["item_id"]).fillna(global_mean).to_numpy()
    dr_values = direct_method_value + weights * (reward - logged_model_reward)
    dr = ci_mean(dr_values)
    switch_threshold = 100.0
    switch_values = direct_method_value + np.where(
        weights <= switch_threshold, weights * (reward - logged_model_reward), 0.0
    )
    switch_dr = ci_mean(switch_values)
    ips_snips_relative_delta = float(abs(ips["mean"] - snips_mean) / max(abs(ips["mean"]), 1e-12))

    result = {
        "card": "C1",
        "dataset": "KuaiRand-Pure random slice",
        "seed": int(args.seed),
        "rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_items": n_items,
        "propensity": {
            "source": "uniform_random_pool_1/7583",
            "unique_values": sorted({float(x) for x in pd.Series(p0).dropna().unique()})[:5],
        },
        "reward": reward_col,
        "target_policy": {
            "type": "synthetic_epsilon_greedy",
            "epsilon": epsilon,
            "greedy_rule": "top-10 items by popularity in the KuaiRand Pure random slice",
            "top_k": top_k,
            "greedy_items": [str(x) for x in greedy_items],
            "purpose": "exercise estimator variance-reduction differences; not a deployed production policy",
        },
        "estimators": {
            "logging_policy_mean": logging_mean,
            "ips": ips,
            "snips": snips,
            "dr": dr,
            "switch_dr": switch_dr,
        },
        "diagnostics": {
            "ess": ess(weights),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            "weight_mean": float(np.mean(weights)),
            "clipping_curve": clipping_curve,
            "direct_method_value": direct_method_value,
            "switch_threshold": switch_threshold,
            "ips_snips_relative_delta": ips_snips_relative_delta,
            "bootstrap_samples": int(args.bootstrap_samples),
            "redline_checks": {
                "nontrivial_weight_range": bool(np.min(weights) < 1.0 < np.max(weights)),
                "ips_snips_differ_at_least_1pct": bool(ips_snips_relative_delta >= 0.01),
                "dr_differs_from_ips": bool(abs(dr["mean"] - ips["mean"]) > 1e-8),
                "ess_below_n": bool(ess(weights) < len(df)),
            },
        },
    }
    write_json(RESULTS_DIR / "c1_kuairand_ope.json", result)
    return result


def _prep_x5_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    df = df.copy()
    for c in df.columns:
        if c.startswith("X_") and ("date" in c or "_ts" in c):
            dt = pd.to_datetime(df[c], errors="coerce")
            df[c] = dt.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    X = df[feature_cols]
    y = pd.to_numeric(df["outcome"], errors="coerce").fillna(0).astype(int)
    t = pd.to_numeric(df["treatment"], errors="coerce").fillna(0).astype(int)
    cat_cols = [c for c in feature_cols if not is_numeric_dtype(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return X, y, t, num_cols, cat_cols


def _preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=0.0,
    )


def sklift_metrics(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> dict[str, float]:
    from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

    y = np.asarray(y, dtype=int)
    t = np.asarray(t, dtype=int)
    score = np.asarray(score, dtype=float).ravel()
    return {
        "auuc": float(uplift_auc_score(y, score, t)),
        "qini": float(qini_auc_score(y, score, t)),
        "uplift_at_10": float(uplift_at_k(y, score, t, strategy="overall", k=0.1)),
        "uplift_at_30": float(uplift_at_k(y, score, t, strategy="overall", k=0.3)),
    }


def c2_x5(args: argparse.Namespace, write: bool = True) -> dict[str, Any]:
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    from econml.metalearners import SLearner, TLearner, XLearner

    df = pd.read_parquet(DATA_DIR / "x5" / "processed" / "uplift.parquet")
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).copy()
    X, y, t, num_cols, cat_cols = _prep_x5_features(df)
    strat = df[["treatment", "outcome"]].astype(str).agg("_".join, axis=1)
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.25, random_state=args.seed, stratify=strat
    )
    pre = _preprocessor(num_cols, cat_cols)
    X_train = pre.fit_transform(X.loc[train_idx])
    X_test = pre.transform(X.loc[test_idx])
    y_train = y.loc[train_idx].to_numpy()
    y_test = y.loc[test_idx].to_numpy()
    t_train = t.loc[train_idx].to_numpy()
    t_test = t.loc[test_idx].to_numpy()

    results: dict[str, Any] = {}

    base_lr = LogisticRegression(max_iter=500, solver="lbfgs")
    base_rf_reg = RandomForestRegressor(
        n_estimators=80, min_samples_leaf=50, random_state=args.seed, n_jobs=-1
    )
    base_rf_clf = RandomForestClassifier(
        n_estimators=80, min_samples_leaf=50, random_state=args.seed, n_jobs=-1
    )

    s_learner = SLearner(overall_model=clone(base_rf_reg))
    s_learner.fit(y_train, t_train, X=X_train)
    s_score = s_learner.effect(X_test)
    results["s_learner"] = {
        **sklift_metrics(y_test, t_test, s_score),
        "library": "econml.metalearners.SLearner",
    }

    t_learner = TLearner(models=clone(base_rf_reg))
    t_learner.fit(y_train, t_train, X=X_train)
    t_score = t_learner.effect(X_test)
    results["t_learner"] = {
        **sklift_metrics(y_test, t_test, t_score),
        "library": "econml.metalearners.TLearner",
    }

    x_learner = XLearner(models=clone(base_rf_reg), cate_models=clone(base_rf_reg))
    x_learner.fit(y_train, t_train, X=X_train)
    x_score = x_learner.effect(X_test)
    results["x_learner"] = {
        **sklift_metrics(y_test, t_test, x_score),
        "library": "econml.metalearners.XLearner",
    }

    dr_learner = DRLearner(
        model_propensity=clone(base_lr),
        model_regression=clone(base_rf_reg),
        model_final=clone(base_rf_reg),
        cv=3,
        random_state=args.seed,
    )
    dr_learner.fit(y_train, t_train, X=X_train)
    dr_score = dr_learner.effect(X_test)
    results["r_learner_dml"] = {
        **sklift_metrics(y_test, t_test, dr_score),
        "library": "econml.dr.DRLearner",
        "note": "DRLearner is used as the sprint DML residualization baseline for the R-learner row.",
    }

    cf_train_n = min(len(train_idx), 60_000)
    cf_test_n = min(len(test_idx), 20_000)
    cf = CausalForestDML(
        model_y=clone(base_rf_clf),
        model_t=clone(base_rf_clf),
        discrete_outcome=True,
        discrete_treatment=True,
        n_estimators=200,
        min_samples_leaf=50,
        cv=3,
        random_state=args.seed,
    )
    cf.fit(y_train[:cf_train_n], t_train[:cf_train_n], X=X_train[:cf_train_n])
    cf_score = cf.effect(X_test[:cf_test_n])
    results["causal_forest_dml"] = {
        **sklift_metrics(y_test[:cf_test_n], t_test[:cf_test_n], cf_score),
        "library": "econml.dml.CausalForestDML",
        "train_rows": int(cf_train_n),
        "test_rows": int(cf_test_n),
    }

    outcome_auc_model = Pipeline(
        [
            ("pre", _preprocessor(num_cols, cat_cols)),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )
    outcome_auc_model.fit(X.loc[train_idx], y.loc[train_idx])
    outcome_auc = float(
        roc_auc_score(y.loc[test_idx], outcome_auc_model.predict_proba(X.loc[test_idx])[:, 1])
    )

    results["dragonnet_tarnet"] = {
        "status": "deferred",
        "reason": "CPU sprint gate; neural baseline not required unless compute permits",
    }
    payload = {
        "card": "C2",
        "dataset": "X5 RetailHero",
        "seed": int(args.seed),
        "rows": int(len(df)),
        "test_rows": int(len(test_idx)),
        "treatment_rate": float(t.mean()),
        "outcome_rate": float(y.mean()),
        "metrics_note": "AUUC, Qini, and uplift@k are computed with scikit-uplift metrics on held-out treatment/control outcomes; no PEHE is reported.",
        "outcome_auc_sanity": outcome_auc,
        "baselines": results,
    }
    if write:
        write_json(RESULTS_DIR / "c2_x5_uplift.json", payload)
    return payload


def _metric_mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return {"mean": float(np.nanmean(arr)), "std": 0.0}
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr, ddof=1))}


def c2_x5_multi_seed(args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    payloads = []
    seed_dir = RESULTS_DIR / "seed_runs"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed)
        payload = c2_x5(seed_args, write=False)
        payloads.append(payload)
        write_json(seed_dir / f"c2_seed{seed}.json", payload)

    primary = payloads[0]
    aggregate = {
        **primary,
        "seed": int(seeds[0]),
        "seeds": [int(s) for s in seeds],
        "seed_runs_source": "backend/scripts/run_oranbench_baselines.py --cards C2 --seeds",
        "baselines": {},
    }
    metric_names = ["auuc", "qini", "uplift_at_10", "uplift_at_30"]
    baseline_names = sorted(
        {
            name
            for payload in payloads
            for name, row in payload["baselines"].items()
            if isinstance(row, dict) and all(metric in row for metric in metric_names)
        }
    )
    for name in baseline_names:
        first_row = primary["baselines"][name]
        row = {key: value for key, value in first_row.items() if key not in metric_names}
        row["seed_runs"] = [
            {
                "seed": int(payload["seed"]),
                **{metric: float(payload["baselines"][name][metric]) for metric in metric_names},
            }
            for payload in payloads
        ]
        for metric in metric_names:
            stats = _metric_mean_std([run[metric] for run in row["seed_runs"]])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        aggregate["baselines"][name] = row
    for name, row in primary["baselines"].items():
        if name not in aggregate["baselines"]:
            aggregate["baselines"][name] = row

    write_json(RESULTS_DIR / "c2_x5_uplift.json", aggregate)
    return aggregate


def patch_obp_pandas_compat() -> list[str]:
    """Patch pandas 3 positional-argument breaks in OBP 0.4.1's bundled loader."""
    applied: list[str] = []
    orig_drop = pd.DataFrame.drop

    def drop_compat(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        return orig_drop(
            self,
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )

    pd.DataFrame.drop = drop_compat
    applied.append("DataFrame.drop(axis positional -> keyword)")

    orig_concat = pd.concat

    def concat_compat(
        objs,
        axis=0,
        join="outer",
        ignore_index=False,
        keys=None,
        levels=None,
        names=None,
        verify_integrity=False,
        sort=False,
        copy=None,
    ):
        return orig_concat(
            objs,
            axis=axis,
            join=join,
            ignore_index=ignore_index,
            keys=keys,
            levels=levels,
            names=names,
            verify_integrity=verify_integrity,
            sort=sort,
            copy=copy,
        )

    pd.concat = concat_compat
    applied.append("pandas.concat(axis positional -> keyword)")
    return applied


def _interval(estimator, kwargs: dict[str, Any], seed: int) -> dict[str, float]:
    interval = estimator.estimate_interval(**kwargs, n_bootstrap_samples=200, random_state=seed)
    return {
        "mean": float(interval["mean"]),
        "ci95_low": float(interval["95.0% CI (lower)"]),
        "ci95_high": float(interval["95.0% CI (upper)"]),
    }


def _obd_pair(behavior_policy: str, args: argparse.Namespace) -> dict[str, Any]:
    from obp.dataset import OpenBanditDataset
    from obp.ope import (
        DoublyRobust,
        DoublyRobustWithShrinkage,
        InverseProbabilityWeighting,
        SelfNormalizedInverseProbabilityWeighting,
        SwitchDoublyRobust,
    )

    dataset = OpenBanditDataset(behavior_policy=behavior_policy, campaign="all")
    feedback = dataset.obtain_batch_bandit_feedback()
    reward = feedback["reward"]
    action = feedback["action"]
    pscore = feedback["pscore"]
    position = feedback["position"]
    n_rounds = int(feedback["n_rounds"])
    n_actions = int(feedback["n_actions"])
    len_list = int(np.max(position) + 1)
    epsilon = 0.1
    top_k = 10
    action_counts = np.bincount(action, minlength=n_actions)
    greedy_actions = np.argsort(-action_counts)[:top_k]
    target_policy = np.full(n_actions, epsilon / n_actions, dtype=float)
    target_policy[greedy_actions] += (1.0 - epsilon) / top_k
    action_dist = np.tile(target_policy[None, :, None], (n_rounds, 1, len_list))

    alpha = 20.0
    global_mean = float(np.mean(reward))
    reward_sums = np.bincount(action, weights=reward, minlength=n_actions)
    reward_counts = np.bincount(action, minlength=n_actions)
    reward_model = (reward_sums + alpha * global_mean) / (reward_counts + alpha)
    estimated_rewards = np.tile(reward_model[None, :, None], (n_rounds, 1, len_list))

    target_probs_logged_actions = target_policy[action]
    weights = target_probs_logged_actions / pscore
    base_kwargs = {
        "reward": reward,
        "action": action,
        "pscore": pscore,
        "action_dist": action_dist,
        "position": position,
    }
    model_kwargs = {**base_kwargs, "estimated_rewards_by_reg_model": estimated_rewards}
    estimators = {
        "ips": (InverseProbabilityWeighting(), base_kwargs),
        "snips": (SelfNormalizedInverseProbabilityWeighting(), base_kwargs),
        "dr": (DoublyRobust(), model_kwargs),
        "switch_dr": (SwitchDoublyRobust(tau=5.0), model_kwargs),
        "dr_os": (DoublyRobustWithShrinkage(lambda_=10.0), model_kwargs),
    }
    estimates: dict[str, Any] = {}
    for name, (estimator, kwargs) in estimators.items():
        estimates[name] = {
            "value": float(estimator.estimate_policy_value(**kwargs)),
            **_interval(estimator, kwargs, args.seed),
            "obp_estimator_name": estimator.estimator_name,
        }

    clip_grid = [1, 2, 5, 10, 20, 50]
    clipping_curve = []
    for clip in clip_grid:
        wc = np.minimum(weights, clip)
        clipping_curve.append(
            {
                "clip": clip,
                "snips": float(np.sum(wc * reward) / np.sum(wc)),
                "ips": float(np.mean(wc * reward)),
                "ess": ess(wc),
            }
        )

    return {
        "logging_policy": behavior_policy,
        "campaign": "all",
        "rows": n_rounds,
        "n_actions": n_actions,
        "len_list": len_list,
        "reward_mean_logged": float(np.mean(reward)),
        "target_policy": {
            "type": "synthetic_epsilon_greedy",
            "epsilon": epsilon,
            "greedy_rule": f"top-{top_k} OBD actions by logged popularity for this behavior-policy slice",
            "top_k": top_k,
            "greedy_actions": [int(x) for x in greedy_actions],
        },
        "estimators": estimates,
        "diagnostics": {
            "ess": ess(weights),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            "weight_mean": float(np.mean(weights)),
            "clipping_curve": clipping_curve,
        },
    }


def c3_obd(args: argparse.Namespace) -> dict[str, Any]:
    import obp

    compat_patches = patch_obp_pandas_compat()
    pairs = [_obd_pair("random", args), _obd_pair("bts", args)]
    payload = {
        "card": "C3",
        "dataset": "Open Bandit Dataset bundled sample",
        "seed": int(args.seed),
        "obp_version": obp.__version__,
        "compatibility_patches": compat_patches,
        "policy_pairs": pairs,
        "main_pair": "random -> synthetic_epsilon_greedy_top10",
        "summary": {
            "random_switch_dr": pairs[0]["estimators"]["switch_dr"],
            "random_dr_os": pairs[0]["estimators"]["dr_os"],
            "random_ess": pairs[0]["diagnostics"]["ess"],
            "bts_switch_dr": pairs[1]["estimators"]["switch_dr"],
            "bts_dr_os": pairs[1]["estimators"]["dr_os"],
            "bts_ess": pairs[1]["diagnostics"]["ess"],
        },
        "note": "OBP 0.4.1 exposes DoublyRobustWithShrinkage (dr-os), not a MoreRobustDoublyRobust class; dr-os is reported as the robust-DR row for this sprint artifact.",
    }
    write_json(RESULTS_DIR / "c3_obd_ope.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", default="C1,C2,C3")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", default="", help="Comma-separated seed list for multi-seed C2 aggregation."
    )
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument(
        "--kuairand-reward",
        default="reward_long_view",
        choices=["reward_click", "reward_like", "reward_long_view", "reward_watch_time"],
    )
    args = parser.parse_args()

    selected = {c.strip().upper() for c in args.cards.split(",") if c.strip()}
    if "C1" in selected:
        c1_kuairand(args)
    if "C2" in selected:
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            c2_x5_multi_seed(args, seeds)
        else:
            c2_x5(args)
    if "C3" in selected:
        c3_obd(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Separate population representation, mechanism ablations and count forecasting.

Reuse frozen full-model evidence; train matched heterogeneity alternatives.
All distribution alternatives retain the same Gaussian state-estimation scheme.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oransim.world_model.population_behavior import BehaviorBatch
from oransim.world_model.population_loop import PopulationWorldLoop
from oransim.world_model.population_panel import PopulationWorldPanel
from train_population_panel import initialize, metrics, read_data, replay, write


def representation(panel, family):
    """Match initial first two moments except for the no-heterogeneity ablation."""
    result = deepcopy(panel)
    for key, world in result.worlds.items():
        s = world.state
        mean = (s.mixture_weights * s.means).sum(1)
        variance = (
            s.mixture_weights * (s.heterogeneity_variance + (s.means - mean[:, None]) ** 2)
        ).sum(1)
        if family == "single_gaussian":
            pi = np.ones((len(mean), 1))
            mu = mean[:, None]
            var = variance[:, None]
        elif family == "two_point":
            pi = np.full((len(mean), 2), 0.5)
            mu = mean[:, None] + np.sqrt(variance)[:, None] * np.array([-1, 1])
            var = np.zeros_like(mu)
        elif family == "point_mass":
            pi = np.ones((len(mean), 1))
            mu = mean[:, None]
            var = np.zeros_like(mu)
        else:
            raise ValueError("unknown heterogeneity representation")
        result.worlds[key] = PopulationWorldLoop(
            s.population_weights,
            pi,
            mu,
            var,
            world.config,
            float(s.mean_posterior_variance[0]),
            s.arrival_posterior_variance,
        )
    result._sync_arrivals()
    return result


def train(initial, actions, frame, dates, seed, max_iterations, max_events):
    counts = np.array(
        [
            [len(frame[(frame.date == date) & (frame.campaign == c)]) for c in initial.campaigns]
            for date in dates
        ],
        float,
    )
    scales = np.tile(
        [initial.worlds[c].config.base_arrivals for c in initial.campaigns], (len(dates), 1)
    )
    arrival = initial.arrivals.fit(
        counts, scales, np.full(len(initial.campaigns), 50.0), shared=True
    )
    behavior = deepcopy(initial.behavior)
    fits = []
    for _ in range(2):
        state = deepcopy(initial)
        state.behavior = deepcopy(behavior)
        _, batches = replay(state, actions, frame, dates)
        batch = BehaviorBatch.concatenate(batches)
        if len(batch.features) > max_events:
            indices = np.sort(
                np.random.default_rng(seed).choice(len(batch.features), max_events, replace=False)
            )
            batch = batch.take(indices)
        fits.append(behavior.fit(batch, max_iterations))
    final = deepcopy(initial)
    final.behavior = behavior
    replay(final, actions, frame, dates)
    parameter_count = (
        behavior.coefficients.size
        + len(behavior.heads) * (len(behavior.heads) - 1) // 2
        + len(behavior.heads)
        - 1
    )
    return final, {
        "arrival": arrival,
        "behavior": fits,
        "learned_behavior_parameters": parameter_count,
    }


def paired_daily_difference(reference, candidate, seed=20260908, replicates=2000, block=3):
    if [r["date"] for r in reference] != [r["date"] for r in candidate]:
        raise ValueError("paired dates required")
    n = np.array([r["metrics"]["events"] for r in reference])
    if not np.array_equal(n, [r["metrics"]["events"] for r in candidate]):
        raise ValueError("paired exposure counts required")
    difference = np.array(
        [
            b["metrics"]["joint_nll"] - a["metrics"]["joint_nll"]
            for a, b in zip(reference, candidate, strict=False)
        ]
    )
    rng = np.random.default_rng(seed)
    estimates = []
    length = len(n)
    for _ in range(replicates):
        starts = rng.integers(length, size=int(np.ceil(length / block)))
        indices = ((starts[:, None] + np.arange(block)) % length).ravel()[:length]
        estimates.append(float(np.average(difference[indices], weights=n[indices])))
    return {
        "candidate_minus_gaussian_nll": float(np.average(difference, weights=n)),
        "circular_day_block_range_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "block_days": block,
        "replicates": replicates,
        "days": length,
        "interpretation": "Descriptive paired block resampling on a short, reused test period; not a causal or external-generalization significance claim.",
    }


def fit_log_trend(series, start, end):
    z = np.log1p(np.asarray(series, float))
    delta = np.diff(z)
    indices = np.arange(max(start, 2), end)
    x = np.column_stack([np.ones(len(indices)), delta[indices - 2]])
    y = delta[indices - 1]
    return np.linalg.solve(x.T @ x + np.diag([0.0, 1.0]), x.T @ y)


def baseline_paths(history, horizon, trend):
    history = list(np.asarray(history, float))
    if len(history) < 7:
        raise ValueError("seven observed days required for all matched baselines")
    seasonal = history.copy()
    linear = history.copy()
    for _ in range(horizon):
        seasonal.append(seasonal[-7])
        change = trend[0] + trend[1] * (np.log1p(linear[-1]) - np.log1p(linear[-2]))
        linear.append(float(np.expm1(np.clip(np.log1p(linear[-1]) + change, 0, 20))))
    return {
        "last_value": np.repeat(history[-1], horizon),
        "weekly_naive": np.array(seasonal[-horizon:]),
        "ridge_log_trend": np.array(linear[-horizon:]),
    }


def count_evaluation(model, frame, train_dates, val_dates, test_dates, seed):
    """Exact marginal log-arrival propagation for this experiment's fixed action.

    These models have zero momentum-arrival gain and zero novelty decay. Joint
    endpoint draws therefore avoid replaying individual actions to score counts.
    """
    campaigns = model.campaigns
    f = deepcopy(model.arrivals)
    for w in model.worlds.values():
        if w.config.momentum_arrival_gain != 0 or w.config.novelty_decay != 0:
            raise ValueError(
                "analytic count propagation requires the frozen experiment configuration"
            )
    scales = np.array([model.worlds[c].config.base_arrivals for c in campaigns])
    disp = np.array([model.worlds[c].config.arrival_dispersion for c in campaigns])
    dates = pd.date_range(frame.date.min(), frame.date.max())
    counts = np.array(
        [[len(frame[(frame.date == d) & (frame.campaign == c)]) for c in campaigns] for d in dates],
        float,
    )
    totals = counts.sum(1)
    lookup = {d: i for i, d in enumerate(dates)}
    trend = fit_log_trend(totals, lookup[train_dates[0]], lookup[train_dates[-1]] + 1)
    for date in val_dates:
        f.observe(counts[lookup[date]], scales, disp)
    output = {str(h): [] for h in (1, 3, 7)}
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    weights /= np.sqrt(np.pi)
    for date in test_dates:
        i = lookup[date]
        horizon = min(7, len(dates) - i)
        baselines = baseline_paths(totals[:i], horizon, trend)
        for h in (1, 3, 7):
            if h > horizon:
                continue
            covariance = f.covariance + (h - 1) * f._noise()
            mean = float(
                (
                    (
                        scales[:, None]
                        * np.exp(
                            np.clip(
                                f.mean[:, None] + np.sqrt(2 * np.diag(covariance))[:, None] * nodes,
                                -8,
                                8,
                            )
                        )
                    )
                    @ weights
                ).sum()
            )
            rng = np.random.default_rng(np.random.SeedSequence([seed, i, h]))
            latent = rng.multivariate_normal(f.mean, covariance, size=8192)
            means = scales * np.exp(np.clip(latent, -8, 8))
            samples = rng.poisson(rng.gamma(disp, means / disp)).sum(1)
            forecast = {name: float(values[h - 1]) for name, values in baselines.items()}
            forecast.update(
                {
                    "panel_predictive_mean": mean,
                    "panel_predictive_median": float(np.median(samples)),
                }
            )
            output[str(h)].append(
                {
                    "origin": str(date.date()),
                    "target": str(dates[i + h - 1].date()),
                    "actual": float(totals[i + h - 1]),
                    "forecasts": forecast,
                    "panel_interval_90": np.quantile(samples, [0.05, 0.95]).tolist(),
                }
            )
        f.observe(counts[i], scales, disp)
    summary = {}
    for h, rows in output.items():
        summary[h] = {"origins": len(rows), "models": {}}
        for name in rows[0]["forecasts"]:
            errors = np.array([r["forecasts"][name] - r["actual"] for r in rows])
            summary[h]["models"][name] = {
                "mae": float(np.abs(errors).mean()),
                "rmse": float(np.sqrt(np.square(errors).mean())),
            }
    return {
        "rows": output,
        "summary": summary,
        "trend_coefficients": trend.tolist(),
        "scope": "Same frozen origin for all models. MAE is aligned with predictive median; mean and RMSE are also reported. Only total recorded exposures are scored.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "experiments/gaussian_population/results/population-panel-v4",
    )
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/kuairand/raw/KuaiRand-Pure/KuaiRand-Pure/data"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/contribution-evaluation-v1"
    )
    parser.add_argument(
        "--private",
        type=Path,
        default=ROOT / "outputs/gaussian_population/contribution-evaluation-v1",
    )
    args = parser.parse_args()
    reference = json.loads((args.reference / "results.json").read_text())
    old = json.loads((args.reference / "protocol.json").read_text())
    for name, digest in old["code_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("reference implementation changed; frozen evidence cannot be reused")
    for name, digest in old["data_sha256"].items():
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != digest:
            raise ValueError("reference data changed")
    protocol = {
        "version": "contribution-evaluation-v1",
        "reference_protocol_sha256": hashlib.sha256(
            (args.reference / "protocol.json").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": old["seed"],
        "comparison": "Group-level intrinsic heterogeneity only. Gaussian filtering uncertainty is held fixed across families. Single Gaussian and two-point priors match mixture mean and variance; point mass removes heterogeneity.",
        "training": "Same users, dates, parameter counts, two fitting rounds, regularization and iteration budget as reference. Gaussian full-model results are reused without refitting.",
        "count_forecast": "Separate fixed-origin 1/3/7-day evaluation; all models see the same past. No count metric enters representation selection.",
        "history_notice": old["history_notice"],
    }
    if (args.out / "protocol.json").exists() and json.loads(
        (args.out / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("changed protocol requires a new directory")
    write(args.out / "protocol.json", protocol)
    raw = read_data(args.data, old["user_hash_divisor"])
    output = {}
    for policy, frame in [
        ("standard", pd.concat(raw[:2]).sort_values(["date", "time_ms"], kind="stable")),
        ("random", raw[2]),
    ]:
        split = reference[policy]["split"]
        warm = pd.Timestamp(split["warmup_end"])
        train_end = pd.Timestamp(split["train_end"])
        val_end = pd.Timestamp(split["validation_end"])
        train_dates = pd.date_range(warm + pd.Timedelta(days=1), train_end)
        val_dates = pd.date_range(train_end + pd.Timedelta(days=1), val_end)
        test_dates = pd.date_range(val_end + pd.Timedelta(days=1), frame.date.max())
        history = raw[0] if policy == "random" else frame[frame.date <= warm]
        initial, actions, prepared = initialize(frame, history, warm, {})
        reference_selection = json.loads((args.reference / policy / "selection.json").read_text())
        validation = {"gaussian_mixture": reference_selection["validation"]["full"]}
        tests = {"gaussian_mixture": reference[policy]["test"]["full"]}
        fits = {}
        models = {}
        for family in ("single_gaussian", "two_point", "point_mass"):
            print(f"{policy}: train {family}", flush=True)
            candidate = representation(initial, family)
            fitted, report = train(
                candidate,
                actions,
                prepared,
                train_dates,
                old["seed"],
                old["max_iterations"],
                old["max_train_events"],
            )
            fits[family] = report
            fitted.save(args.private / policy / f"{family}-train.json")
            rows, _ = replay(fitted, actions, prepared, val_dates, score=True)
            validation[family] = {"metrics": metrics(rows), "daily": rows}
            models[family] = fitted
        selected = min(validation, key=lambda name: validation[name]["metrics"]["joint_nll"])
        write(
            args.out / policy / "selection.json",
            {"selected": selected, "validation": validation, "fits": fits},
        )
        for family, model in models.items():
            rows, _ = replay(model, actions, prepared, test_dates, score=True)
            tests[family] = {"metrics": metrics(rows), "daily": rows}
            model.save(args.private / policy / f"{family}-final.json")
        differences = {
            family: paired_daily_difference(
                tests["gaussian_mixture"]["daily"], tests[family]["daily"], old["seed"]
            )
            for family in models
        }
        source_model = PopulationWorldPanel.load(
            ROOT / reference[policy]["private_model_directory"] / "full-train.json"
        )
        counts = count_evaluation(
            source_model, prepared, train_dates, val_dates, test_dates, old["seed"]
        )
        write(args.out / policy / "count-forecast.json", counts)
        mechanisms = {
            name: paired_daily_difference(
                reference[policy]["test"]["full"]["daily"], r["daily"], old["seed"]
            )
            for name, r in reference[policy]["test"].items()
            if name != "full"
        }
        output[policy] = {
            "selected_representation": selected,
            "representations": tests,
            "paired_differences": differences,
            "existing_mechanism_ablations": mechanisms,
            "count_forecast_summary": counts["summary"],
        }
        write(args.out / "results.json", output)
        print(f"{policy}: complete; validation selected {selected}", flush=True)


if __name__ == "__main__":
    main()

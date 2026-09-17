#!/usr/bin/env python3
"""Train-only NB-INGARCH(1,1) comparison on the frozen readiness records.

Each task is reduced to one integer total per day by summing joint-pattern
counts. The model is Y_t|F_(t-1) ~ NB(mean=lambda_t, size=r), with
lambda_(t+1)=omega+alpha*Y_t+beta*lambda_t. A stable parameterization enforces
omega>0, alpha,beta>=0 and alpha+beta<1. Validation/test observations only
update the filter; parameter estimation uses training dates exclusively.

Forecast horizons 1/3/7 refer to endpoint-day counts, matching the existing
market scripts. Their mean paths replace unknown future counts by conditional
means. One-step NB NLL is the model's conditional NLL; multi-step NLL is an
explicit plug-in NB diagnostic, since the marginal multi-step law is a mixture.
The NLL unit is one daily total, not the mean-cell count NLL of the market model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import gammaln, softmax

ROOT = Path(__file__).resolve().parents[2]
SLOTS = {
    "kuairand/standard": "kuairand/standard/aggregate/records.json",
    "kuairand/random": "kuairand/random/aggregate/records.json",
    "retail/2010": "retail/2010/records.json",
    "retail/2011": "retail/2011/records.json",
}
CONFIG = {
    "horizons": [1, 3, 7],
    "stationarity_margin": 1e-6,
    "stationary_mean_lower": 0.001,
    "stationary_mean_upper_rule": "max(10, 100*train_mean, 10*train_max)",
    "dispersion_bounds": [0.01, 100000.0],
    "coefficient_logit_bounds": [-12.0, 12.0],
    "starts_alpha_beta": [[0.05, 0.05], [0.4, 0.1], [0.18, 0.72], [0.72, 0.18]],
    "optimizer": "L-BFGS-B",
    "maxiter": 2000,
    "ftol": 1e-12,
    "gtol": 1e-6,
    "fallback": "Powell with the same bounds if no L-BFGS-B start converges",
    "selection": "lowest training mean NB NLL among converged starts; finite minimum if none converge",
    "initialization": "lambda at the first training day equals fitted stationary mean",
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def nll(count, mean, dispersion):
    """NB2 negative log probability; Var(Y|past)=mean+mean**2/dispersion."""
    y, mu, r = np.asarray(count), np.asarray(mean), dispersion
    return (
        gammaln(r)
        + gammaln(y + 1)
        - gammaln(y + r)
        + r * np.log1p(mu / r)
        + y * (np.log(r + mu) - np.log(mu))
    )


def decode(theta):
    mean, dispersion = np.exp(theta[0]), np.exp(theta[3])
    weights = softmax([theta[1], theta[2], 0.0])
    alpha, beta = (1.0 - CONFIG["stationarity_margin"]) * weights[:2]
    return {
        "stationary_mean": float(mean),
        "omega": float(mean * (1 - alpha - beta)),
        "alpha": float(alpha),
        "beta": float(beta),
        "dispersion": float(dispersion),
    }


def conditional_means(counts, parameters):
    intensity = parameters["stationary_mean"]
    result = []
    for count in counts:
        result.append(intensity)
        intensity = (
            parameters["omega"] + parameters["alpha"] * count + parameters["beta"] * intensity
        )
    return np.asarray(result), float(intensity)


def fit_counts(train):
    train = np.asarray(train, dtype=float)
    if (
        len(train) < 2
        or np.any(~np.isfinite(train))
        or np.any(train < 0)
        or np.any(train != np.floor(train))
    ):
        raise ValueError("at least two finite nonnegative integer training counts are required")
    initial_mean = max(float(train.mean()), CONFIG["stationary_mean_lower"])
    upper = max(10.0, 100 * float(train.mean()), 10 * float(train.max()))
    bounds = [
        (np.log(CONFIG["stationary_mean_lower"]), np.log(upper)),
        tuple(CONFIG["coefficient_logit_bounds"]),
        tuple(CONFIG["coefficient_logit_bounds"]),
        tuple(np.log(CONFIG["dispersion_bounds"])),
    ]
    variance = float(train.var())
    initial_size = np.clip(
        initial_mean**2 / max(variance - initial_mean, 0.001), *CONFIG["dispersion_bounds"]
    )

    def objective(theta):
        parameters = decode(theta)
        means, _ = conditional_means(train, parameters)
        return float(nll(train, means, parameters["dispersion"]).mean())

    attempts = []
    for alpha, beta in CONFIG["starts_alpha_beta"]:
        scale = 1.0 - CONFIG["stationarity_margin"]
        leftover = scale - alpha - beta
        theta = np.array(
            [
                np.log(initial_mean),
                np.log(alpha / leftover),
                np.log(beta / leftover),
                np.log(initial_size),
            ]
        )
        fit = minimize(
            objective,
            theta,
            method="L-BFGS-B",
            bounds=bounds,
            options={k: CONFIG[k] for k in ["maxiter", "ftol", "gtol"]},
        )
        attempts.append((fit, "L-BFGS-B"))
    if not any(f.success and np.isfinite(f.fun) for f, _ in attempts):
        finite = [f for f, _ in attempts if np.isfinite(f.fun)]
        if not finite:
            raise RuntimeError("all train-only INGARCH starts returned nonfinite losses")
        fit = minimize(
            objective,
            min(finite, key=lambda f: f.fun).x,
            method="Powell",
            bounds=bounds,
            options={"maxiter": CONFIG["maxiter"], "ftol": CONFIG["ftol"], "xtol": 1e-8},
        )
        attempts.append((fit, "Powell"))
    eligible = [i for i, (f, _) in enumerate(attempts) if f.success and np.isfinite(f.fun)]
    if not eligible:
        eligible = [i for i, (f, _) in enumerate(attempts) if np.isfinite(f.fun)]
    chosen = min(eligible, key=lambda i: attempts[i][0].fun)
    fit, method = attempts[chosen]
    active_bounds = [
        i
        for i, (low, high) in enumerate(bounds)
        if min(abs(fit.x[i] - low), abs(fit.x[i] - high)) < 1e-3
    ]
    return {
        "parameters": decode(fit.x),
        "training_mean_nll": float(fit.fun),
        "converged": bool(fit.success),
        "optimizer": method,
        "selected_start": chosen,
        "active_bound_coordinates": active_bounds,
        "coordinate_order": ["log stationary mean", "alpha logit", "beta logit", "log dispersion"],
        "bounds": [[float(a), float(b)] for a, b in bounds],
        "iterations": int(fit.nit),
        "message": str(fit.message),
        "attempts": [
            {
                "method": m,
                "converged": bool(f.success),
                "training_mean_nll": float(f.fun),
                "iterations": int(f.nit),
                "message": str(f.message),
            }
            for f, m in attempts
        ],
    }


def extract_records(raw):
    result = {}
    ordered_dates = []
    for split in ["train", "validation", "test"]:
        values = []
        if not raw[split]:
            raise ValueError(f"{split} split must be nonempty")
        for record in raw[split]:
            patterns = np.asarray(record["patterns"], dtype=float)
            if (
                patterns.ndim != 2
                or np.any(~np.isfinite(patterns))
                or np.any(patterns < 0)
                or np.any(patterns != np.floor(patterns))
            ):
                raise ValueError("joint pattern counts must be finite nonnegative integers")
            stamp = date.fromisoformat(record["date"])
            ordered_dates.append(stamp)
            values.append({"date": stamp.isoformat(), "count": int(patterns.sum())})
        result[split] = values
    if any(
        b - a != timedelta(days=1) for a, b in zip(ordered_dates, ordered_dates[1:], strict=False)
    ):
        raise ValueError("splits must jointly form consecutive increasing calendar days")
    return result


def score_records(records, parameters):
    """Condition each test origin only on observations before that origin."""
    past = records["train"] + records["validation"]
    _, intensity = conditional_means([d["count"] for d in past], parameters)
    rows = []
    for index, origin in enumerate(records["test"]):
        mean = intensity
        remaining = len(records["test"]) - index
        for h in range(1, min(max(CONFIG["horizons"]), remaining) + 1):
            if h in CONFIG["horizons"]:
                target = records["test"][index + h - 1]
                observed = target["count"]
                rows.append(
                    {
                        "origin": origin["date"],
                        "target": target["date"],
                        "horizon": h,
                        "observed": observed,
                        "exposures": observed,
                        "predicted_exposures": float(mean),
                        "count_nll": float(nll(observed, mean, parameters["dispersion"])),
                        "count_nll_kind": (
                            "conditional_NB" if h == 1 else "plug_in_NB_at_endpoint_mean"
                        ),
                    }
                )
            mean = parameters["omega"] + (parameters["alpha"] + parameters["beta"]) * mean
        intensity = (
            parameters["omega"]
            + parameters["alpha"] * origin["count"]
            + parameters["beta"] * intensity
        )
    summaries = {}
    for h in CONFIG["horizons"]:
        chosen = [r for r in rows if r["horizon"] == h]
        errors = np.array([r["predicted_exposures"] - r["observed"] for r in chosen])
        summaries[str(h)] = {
            "origins": len(chosen),
            "mae": float(np.abs(errors).mean()) if chosen else None,
            "rmse": float(np.sqrt(np.mean(errors**2))) if chosen else None,
            "count_nll": float(np.mean([r["count_nll"] for r in chosen])) if chosen else None,
        }
    return {"summary": summaries, "date_level": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private", type=Path, default=ROOT / "models/gaussian_population/benchmarks"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/paper-readiness"
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail after saving available results if a required records file is missing.",
    )
    args = parser.parse_args(argv)
    protocol = {
        "method": "negative_binomial_INGARCH_1_1",
        "config": CONFIG,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "input_slots": SLOTS,
        "private_root": str(args.private.resolve()),
        "observations": "Total joint-pattern counts per day; only aggregate records consumed.",
        "training": "Conditional NB2 MLE on training dates only; initial mean is fitted stationary mean.",
        "evaluation": "Validation and past test counts update intensity without parameter refitting; endpoint means at horizons 1/3/7.",
        "likelihood_scope": "NLL per daily total; one-step exact conditional NB, multi-step plug-in NB diagnostic, unlike the mean-cell NB NLL of market models.",
        "reference": "https://doi.org/10.1111/j.1467-9892.2006.00496.x",
        "reference_scope": "Original reference is Poisson INGARCH; this baseline uses an NB2 observation extension.",
    }
    protocol_path = args.out / "count-baseline-protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("source or config changed; use a new output directory")
    else:
        write_json(protocol_path, protocol)  # Persist before loading outcomes or fitting any task.
    result_path = args.out / "count-baselines.json"
    result = {
        "method": protocol["method"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "likelihood_scope": protocol["likelihood_scope"],
        "datasets": {},
        "missing": [],
    }
    previous = json.loads(result_path.read_text()) if result_path.exists() else {}
    for key, relative in SLOTS.items():
        path = args.private / relative
        if not path.exists():
            result["missing"].append(key)
            continue
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        cached = previous.get("datasets", {}).get(key)
        if cached and cached["records_sha256"] == source_hash:
            result["datasets"][key] = cached
            continue
        records = extract_records(json.loads(path.read_text()))
        fit = fit_counts([d["count"] for d in records["train"]])
        scores = score_records(records, fit["parameters"])
        result["datasets"][key] = {
            "records_path": str(path.relative_to(args.private)),
            "records_sha256": source_hash,
            "count_object": (
                "sampled logged exposures"
                if key.startswith("kuairand")
                else "positive transaction events in the fixed historical customer/product population"
            ),
            "splits": {
                s: {
                    "start": rows[0]["date"],
                    "end": rows[-1]["date"],
                    "days": len(rows),
                    "total_count": sum(d["count"] for d in rows),
                }
                for s, rows in records.items()
            },
            "fit": fit,
            **scores,
        }
        print(
            key,
            "converged=",
            fit["converged"],
            "one_day_MAE=",
            scores["summary"]["1"]["mae"],
            flush=True,
        )
    result["complete"] = not result["missing"]
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(result_path, result)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "available": list(result["datasets"]),
                "missing": result["missing"],
            }
        )
    )
    return 1 if args.require_all and result["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())

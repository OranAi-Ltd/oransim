"""Chronological KuaiRand-Pure tab=1 grouped response/composition experiment.

Run: python3 backend/scripts/run_kuairand_population_dynamics.py
Only aggregate metrics are exported. All hyperparameter selection uses validation
endpoints. h>1 forecasts freeze the origin state; daily rolling updates resume
only when advancing to the next origin. Standard and random policies have
separate training priors, fitted states, validation selection, and test results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from oransim.data.gaussian_response import (
    GaussianLogitResponse,
    GaussianResponseConfig,
    pooled_prior,
)

SOURCE = "https://github.com/chongminggao/KuaiRand/blob/main/README.md"
HORIZONS = (1, 3, 7)


def fit_segments(history: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Fit fixed 4 response x 2 exposure quantiles on pre-intervention logs."""
    stats = history.groupby("user_id").is_click.agg(["sum", "size"])
    global_rate = (stats["sum"].sum() + 0.5) / (stats["size"].sum() + 1)
    rate = (stats["sum"] + 20 * global_rate) / (stats["size"] + 20)
    rate_edges = np.unique(np.quantile(rate, [0.25, 0.5, 0.75]))
    volume_edges = np.unique(np.quantile(stats["size"], [0.5]))
    labels = pd.Series(
        np.searchsorted(rate_edges, rate, side="right") * 2
        + np.searchsorted(volume_edges, stats["size"], side="right"),
        index=stats.index,
    )
    return labels, {
        "rate_quantile_edges": rate_edges.tolist(),
        "exposure_quantile_edges": volume_edges.tolist(),
        "training_users": len(stats),
        "known_groups": 8,
        "cold_start_group": 8,
        "smoothing_exposures": 20,
    }


def daily_counts(
    frame: pd.DataFrame, segments: pd.Series
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = frame.assign(group=frame.user_id.map(segments).fillna(8).astype(int))
    dates = pd.date_range(frame.date.min(), frame.date.max(), freq="D")
    grouped = frame.groupby(["date", "group"]).is_click.agg(["sum", "size"])
    index = pd.MultiIndex.from_product([dates, range(9)], names=["date", "group"])
    grouped = grouped.reindex(index, fill_value=0)
    return (
        dates,
        grouped["sum"].to_numpy().reshape(-1, 9).astype(float),
        grouped["size"].to_numpy().reshape(-1, 9).astype(float),
    )


def response_candidates() -> list[dict]:
    candidates = [{"family": "fixed_train", "strength": 20.0}]
    candidates += [
        {"family": "last_value", "strength": strength} for strength in (0.5, 20.0, 200.0)
    ]
    candidates += [
        {"family": "ewma", "alpha": alpha, "strength": strength}
        for alpha, strength in itertools.product((0.05, 0.15, 0.35, 0.7), (20.0, 200.0))
    ]
    for level, trend, cap in itertools.product(
        (0.0001, 0.001, 0.01, 0.1), ((0.0, 0.0), (0.0001, 0.8), (0.001, 0.8)), (0.0, 200.0, 2000.0)
    ):
        candidates.append(
            {
                "family": "gaussian_logit",
                **asdict(GaussianResponseConfig(level, trend[0], trend[1], cap)),
            }
        )
    return candidates


def response_predictions(
    successes: np.ndarray, exposures: np.ndarray, train_end: int, candidate: dict
) -> dict[int, np.ndarray]:
    """Entry [h][origin] predicts endpoint origin+h-1 before observing origin."""
    prior = pooled_prior(successes[:train_end].sum(0), exposures[:train_end].sum(0))
    family = candidate["family"]
    model = None
    state = prior.copy()
    if family == "gaussian_logit":
        model = GaussianLogitResponse(
            prior, GaussianResponseConfig(**{k: v for k, v in candidate.items() if k != "family"})
        )
    predictions = {h: np.full_like(successes, np.nan) for h in HORIZONS}
    for day in range(len(successes)):
        if day >= train_end:
            for h in HORIZONS:
                predictions[h][day] = model.forecast(h) if model is not None else state
        if model is not None:
            model.advance()
            model.observe(successes[day], exposures[day])
        elif family != "fixed_train":
            strength = candidate["strength"]
            rate = (successes[day] + strength * prior) / (exposures[day] + strength)
            active = exposures[day] > 0
            alpha = candidate.get("alpha", 1.0)
            state[active] = (1 - alpha) * state[active] + alpha * rate[active]
    return predictions


def composition_predictions(
    exposures: np.ndarray, train_end: int, alpha: float
) -> dict[int, np.ndarray]:
    counts = exposures[:train_end].sum(0)
    prior = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
    state = prior.copy()
    predictions = {h: np.full_like(exposures, np.nan) for h in HORIZONS}
    for day, counts in enumerate(exposures):
        if day >= train_end:
            for h in HORIZONS:
                predictions[h][day] = state
        if counts.sum() and alpha:
            pi = (counts + 20 * prior) / (counts.sum() + 20)
            state = (1 - alpha) * state + alpha * pi
    return predictions


def phase_arrays(
    prediction: np.ndarray,
    successes: np.ndarray,
    exposures: np.ndarray,
    start: int,
    stop: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stop - start < horizon:
        raise ValueError("Phase is shorter than forecast horizon")
    # Every frozen forecast window stays fully inside its evaluation phase.
    return (
        prediction[start : stop - horizon + 1],
        successes[start + horizon - 1 : stop],
        exposures[start + horizon - 1 : stop],
    )


def response_metrics(
    probability: np.ndarray,
    successes: np.ndarray,
    exposures: np.ndarray,
    composition: np.ndarray | None = None,
) -> dict:
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    total = exposures.sum()
    weight = exposures.sum(1)
    actual = successes.sum(1) / weight
    # Actual exposures here are scoring weights, never forecasting inputs.
    conditional_rate = (probability * exposures).sum(1) / weight
    result = {
        "binomial_log_loss": float(
            -(
                successes * np.log(probability) + (exposures - successes) * np.log1p(-probability)
            ).sum()
            / total
        ),
        "bernoulli_brier": float(
            (successes * (1 - probability) ** 2 + (exposures - successes) * probability**2).sum()
            / total
        ),
        "conditional_total_rate_rmse": float(
            np.sqrt(np.average((conditional_rate - actual) ** 2, weights=weight))
        ),
        "group_rate_rmse": float(
            np.sqrt(
                (
                    exposures
                    * (
                        probability
                        - np.divide(
                            successes, exposures, out=np.zeros_like(successes), where=exposures > 0
                        )
                    )
                    ** 2
                ).sum()
                / total
            )
        ),
        "forecast_origins": len(probability),
        "endpoint_exposures": int(total),
    }
    if composition is not None:
        joint_rate = (probability * composition).sum(1)
        result["forecast_composition_total_rate_rmse"] = float(
            np.sqrt(np.average((joint_rate - actual) ** 2, weights=weight))
        )
    return result


def composition_metrics(prediction: np.ndarray, exposures: np.ndarray) -> dict:
    actual = exposures / exposures.sum(1, keepdims=True)
    return {
        "multinomial_log_loss": float(
            -(exposures * np.log(np.maximum(prediction, 1e-12))).sum() / exposures.sum()
        ),
        "mean_total_variation": float(np.abs(actual - prediction).sum(1).mean() / 2),
        "forecast_origins": len(exposures),
    }


def evaluate_policy(
    dates: pd.DatetimeIndex,
    successes: np.ndarray,
    exposures: np.ndarray,
    train_end: int,
    validation_end: int,
) -> dict:
    phases = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(dates)),
    }
    result = {
        "splits": {
            name: {
                "start": str(dates[a].date()),
                "end": str(dates[b - 1].date()),
                "days": b - a,
                "exposures": int(exposures[a:b].sum()),
                "responses": int(successes[a:b].sum()),
            }
            for name, (a, b) in phases.items()
        },
        "horizons": {},
    }
    candidates = response_candidates()
    forecasts = [
        response_predictions(successes, exposures, train_end, candidate) for candidate in candidates
    ]
    pi_candidates = (0.0, 0.1, 0.35, 0.7, 1.0)
    pi_forecasts = [composition_predictions(exposures, train_end, alpha) for alpha in pi_candidates]
    for horizon in HORIZONS:
        if validation_end - train_end < horizon or len(dates) - validation_end < horizon:
            result["horizons"][str(horizon)] = {
                "status": "Open",
                "reason": "Validation or test phase is shorter than horizon; no test-driven tuning or cross-split frozen forecast.",
            }
            continue
        validation = [
            response_metrics(
                *phase_arrays(
                    pred[horizon], successes, exposures, train_end, validation_end, horizon
                )
            )
            for pred in forecasts
        ]
        best_by_family = {
            family: min(
                (i for i, candidate in enumerate(candidates) if candidate["family"] == family),
                key=lambda i: validation[i]["binomial_log_loss"],
            )
            for family in ("fixed_train", "last_value", "ewma", "gaussian_logit")
        }
        pi_val = []
        for pred in pi_forecasts:
            p, _, n = phase_arrays(
                pred[horizon], successes, exposures, train_end, validation_end, horizon
            )
            pi_val.append(composition_metrics(p, n))
        pi_best = min(range(len(pi_candidates)), key=lambda i: pi_val[i]["multinomial_log_loss"])
        pi_test, _, test_n = phase_arrays(
            pi_forecasts[pi_best][horizon],
            successes,
            exposures,
            validation_end,
            len(dates),
            horizon,
        )
        methods = {}
        for family, best in best_by_family.items():
            test_p, test_k, test_n = phase_arrays(
                forecasts[best][horizon], successes, exposures, validation_end, len(dates), horizon
            )
            methods[family] = {
                "parameters": candidates[best],
                "validation": validation[best],
                "test": response_metrics(test_p, test_k, test_n, pi_test),
            }
        winner = min(
            best_by_family, key=lambda family: methods[family]["validation"]["binomial_log_loss"]
        )
        composition = {}
        for i, alpha in enumerate(pi_candidates):
            p, _, n = phase_arrays(
                pi_forecasts[i][horizon], successes, exposures, validation_end, len(dates), horizon
            )
            composition[str(alpha)] = {"validation": pi_val[i], "test": composition_metrics(p, n)}
        result["horizons"][str(horizon)] = {
            "status": "measured",
            "forecast_protocol": "Predict endpoint h days ahead from state available before origin day; no observations from frozen forecast window. Next origin may consume previous actual day.",
            "validation_selected_response": winner,
            "validation_selected_composition_alpha": pi_candidates[pi_best],
            "response_methods": methods,
            "composition_methods": composition,
        }
    # Descriptive test-only decomposition. Both dates must expose every group:
    # assigning p=0 to an absent group would invent a within-group rate change.
    test_k, test_n = successes[validation_end:], exposures[validation_end:]
    observed_p = np.divide(test_k, test_n, out=np.zeros_like(test_k), where=test_n > 0)
    observed_pi = test_n / test_n.sum(1, keepdims=True)
    valid = np.all(test_n[1:] > 0, axis=1) & np.all(test_n[:-1] > 0, axis=1)
    dp, dpi = np.diff(observed_p, axis=0), np.diff(observed_pi, axis=0)
    within = (dp * (observed_pi[1:] + observed_pi[:-1]) / 2).sum(1)
    mix = (dpi * (observed_p[1:] + observed_p[:-1]) / 2).sum(1)
    result["descriptive_daily_rate_change"] = {
        "period": "test",
        "eligible_adjacent_day_pairs": int(valid.sum()),
        "composition_mean_absolute_change": (
            float(np.abs(mix[valid]).mean()) if valid.any() else None
        ),
        "within_group_mean_absolute_change": (
            float(np.abs(within[valid]).mean()) if valid.any() else None
        ),
        "maximum_decomposition_error": (
            float(np.max(np.abs((within + mix - np.diff(test_k.sum(1) / test_n.sum(1)))[valid])))
            if valid.any()
            else None
        ),
        "interpretation": "Exact symmetric exposure-share and within-group response accounting, restricted to adjacent test dates with every group exposed. Within-group shifts still include content/user selection and are not causal fatigue.",
    }
    result["candidate_counts"] = {"response": len(candidates), "composition": len(pi_candidates)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "data/kuairand/raw/KuaiRand-Pure/KuaiRand-Pure/data"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/gaussian_population/results/kuairand-dynamics.json",
    )
    args = parser.parse_args()
    frames, sources = {}, []
    for name in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ):
        raw = pd.read_csv(args.raw_dir / name, usecols=["user_id", "date", "tab", "is_click"])
        if not raw.is_click.isin([0, 1]).all():
            raise ValueError(f"Nonbinary response in {name}")
        frame = raw.loc[raw.tab == 1].copy()
        frame["date"] = pd.to_datetime(frame.date.astype(str), format="%Y%m%d")
        frames[name] = frame
        sources.append(
            {
                "file": name,
                "all_exposures": len(raw),
                "included_tab1_exposures": len(frame),
                "excluded_other_tab_exposures": len(raw) - len(frame),
                "included_zero_responses": int((frame.is_click == 0).sum()),
                "included_users": frame.user_id.nunique(),
                "first_date": str(frame.date.min().date()),
                "last_date": str(frame.date.max().date()),
            }
        )
    history = frames["log_standard_4_08_to_4_21_pure.csv"]
    segments, segment_info = fit_segments(history)
    standard = pd.concat([history, frames["log_standard_4_22_to_5_08_pure.csv"]], ignore_index=True)
    random = frames["log_random_4_22_to_5_08_pure.csv"]
    result = {
        "experiment": "kuairand_pure_gaussian_logit_response_v1",
        "source_documentation": SOURCE,
        "target": "is_click binary feedback (click or valid_play depending on UI), restricted to scenario tab=1",
        "sources": sources,
        "grouping": {
            **segment_info,
            "fit_period": "2022-04-09 through 2022-04-21, standard tab=1 only",
            "features": "Per-user smoothed past response rate and count of candidate-pool exposures; no static cumulative user/video features",
            "frozen_for_validation_and_test": True,
            "random_cold_start_exposures": int((~random.user_id.isin(segments.index)).sum()),
        },
        "model_notes": {
            "latent": "Per-group Gaussian level/trend in logit response space; Binomial observation with Laplace update",
            "numeric": "Stable sigmoid/logit, bounded Newton steps, covariance symmetry, 9-point Gauss-Hermite probability integration, training-only group-prior shrinkage",
            "initialization": "The training aggregate sets an empirical prior mean with weak fixed logit variance 1. Training is replayed to fit final level/trend; counts are not inserted into initial precision. This reuses training outcomes for the empirical center, and posterior coverage is not calibrated.",
            "effective_exposure_cap": "0 = exact Binomial likelihood; positive values temper the likelihood for repeated/correlated observations. No interval calibration claim.",
            "composition": "pi is the group share of candidate-pool tab=1 exposures, not demographic population mass. It is forecast separately from response p.",
            "selection": "Each family/horizon tuned by pooled validation endpoint log loss; dynamic/static winner chosen on validation only. Test scored after parameters fixed.",
            "uncertainty": "Few independent dates and overlapping multi-day windows; point metrics only, no significance or long-horizon generalization claim.",
        },
        "limitations": [
            "Pure retains only candidate-video-pool logs; complete sequential exposure histories and total fatigue doses are unavailable.",
            "Random insertion of items does not randomize daily exposure volume, group composition, or time; dynamics remain predictive/descriptive.",
            "Standard and random policies are fitted independently. Results do not establish cross-policy or cross-platform transfer.",
            "Content composition can change within each fixed historical user group. Tab filtering controls scenario only; no undocumented tab-to-UI mapping is assumed.",
            "Cold-start users share one fixed group. Groups do not encode social influence or interpretable psychological states.",
        ],
        "policies": {},
    }
    for policy, frame, train_end, validation_end in (
        ("random", random, 7, 10),
        ("standard", standard, 13, 20),
    ):
        dates, successes, exposures = daily_counts(frame, segments)
        result["policies"][policy] = evaluate_policy(
            dates, successes, exposures, train_end, validation_end
        )
        print(
            f"Finished {policy}: {len(dates)} dates, {int(exposures.sum())} exposures", flush=True
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(str(args.out))
    for policy, output in result["policies"].items():
        for horizon, metrics in output["horizons"].items():
            if metrics["status"] == "measured":
                print(
                    policy,
                    horizon,
                    "selected=",
                    metrics["validation_selected_response"],
                    {
                        name: round(m["test"]["binomial_log_loss"], 6)
                        for name, m in metrics["response_methods"].items()
                    },
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

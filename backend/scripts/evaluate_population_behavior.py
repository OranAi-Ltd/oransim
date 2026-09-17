#!/usr/bin/env python3
"""Replay frozen models for behavior, cohort, calibration and multistep scores.

No neural fitting. History baseline selection completes across all windows
before test scoring. Conditional scores use observed endpoint exposure weights;
end-to-end action counts use predicted endpoint arrivals.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_population_deepar as ar
from evaluate_population_count_baselines import SLOTS, write_json
from oransim.world_model.differentiable_market import DifferentiableMarket, MarketDay, score_days
from oransim.world_model.population_comparisons import load_comparison
from oransim.world_model.population_gru_projection import load_projected_gru

ORIGINAL = ROOT / "models/gaussian_population/benchmarks"
EXTERNAL = ROOT / "models/gaussian_population/deepar"
SAVED = ROOT / "experiments/gaussian_population/results"
SEEDS = [20260915, 20260916, 20260917]
HORIZONS = [1, 3, 7]
BASELINES = ["training_history", "last_history", "ewma_history"]
VARIANTS = ["joint_gaussian", "fixed_gaussian", "joint_discrete", "direct", "no_dynamics"]


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(raw):
    pairs = list(
        zip(raw["train"][0]["batch"]["campaign"], raw["train"][0]["batch"]["group"], strict=False)
    )
    order = np.asarray(sorted(range(len(pairs)), key=pairs.__getitem__))
    for rows in raw.values():
        for row in rows:
            assert list(zip(row["batch"]["campaign"], row["batch"]["group"], strict=False)) == pairs
            assert row.get("event_batch") is None
    data = {
        split: np.asarray([d["patterns"] for d in days], dtype=float)[:, order]
        for split, days in raw.items()
    }
    days = {split: [MarketDay.from_dict(d) for d in rows] for split, rows in raw.items()}
    groups = np.asarray(raw["train"][0]["batch"]["group"])[order]
    outcomes = np.asarray(
        list(itertools.product([0.0, 1.0], repeat=int(np.log2(data["train"].shape[-1]))))
    )
    return dict(data=data, days=days, groups=groups, order=order, outcomes=outcomes, raw=raw)


def history_forecasts(data, split, method, tau, half_life=None):
    train = data["train"]
    prior = train.sum((0, 1)) + 0.5
    prior /= prior.sum()
    average = train.mean(0)
    statistic = train.sum(0) if method == "training_history" else train[0].copy()
    decay = 0.5 ** (1 / half_life) if half_life else 0.0
    if method == "last_history":
        statistic = train[-1].copy()
    if method == "ewma_history":
        for day in train[1:]:
            statistic = decay * statistic + (1 - decay) * day
    if split == "test" and method != "training_history":
        for day in data["validation"]:
            statistic = decay * statistic + (1 - decay) * day
    result = {h: [] for h in HORIZONS}
    for index, day in enumerate(data[split]):
        q = (statistic + tau * prior) / (statistic.sum(-1, keepdims=True) + tau)
        count = average.sum(-1) if method == "training_history" else statistic.sum(-1)
        for h in HORIZONS:
            if index + h <= len(data[split]):
                result[h].append((q.copy(), count.copy()))
        if method != "training_history":
            statistic = decay * statistic + (1 - decay) * day
    return result


def select_history(data):
    selections = {}
    for method in BASELINES:
        candidates = []
        for tau in [1.0, 10.0, 100.0]:
            for half_life in ([1.0, 3.0, 7.0] if method == "ewma_history" else [None]):
                q = np.asarray(
                    [p[0] for p in history_forecasts(data, "validation", method, tau, half_life)[1]]
                )
                y = data["validation"]
                loss = float(-(y * np.log(q)).sum() / y.sum())
                candidates.append(dict(tau=tau, half_life=half_life, validation_nll=loss))
        selections[method] = dict(
            selected=min(candidates, key=lambda c: c["validation_nll"]), candidates=candidates
        )
    return selections


@torch.no_grad()
def neural_forecasts(model, state, prepared):
    days, order = prepared["days"], prepared["order"]
    _, state = score_days(model, days["validation"], state)
    result = {h: [] for h in HORIZONS}
    for index, origin in enumerate(days["test"]):
        plans = [
            model.future_batch(day.batch, origin.batch) for day in days["test"][index : index + 7]
        ]
        outputs, _ = model.rollout(plans, state)
        for h in HORIZONS:
            if h <= len(outputs):
                out = outputs[h - 1]
                result[h].append(
                    (out["joint_log_prob"].exp().numpy()[order], out["exposures"].numpy()[order])
                )
        _, state = score_days(model, [origin], state)
    return result


@torch.no_grad()
def deepar_forecasts(key, seed, prepared, selections):
    data = ar.tensorize(prepared["raw"])
    x = data["train"]["x"]
    slot = f"{key}/deepar_joint/{seed}"
    model = ar.DeepAR(
        x.shape[-1], prepared["outcomes"].shape[-1], x.shape[1], selections[slot]["width"], True
    )
    model.load_state_dict(torch.load(EXTERNAL / f"{slot}.pt", weights_only=True))
    _, state, previous = ar.objective(model, data["train"])
    _, state, previous = ar.objective(model, data["validation"], state, previous)
    result = {h: [] for h in HORIZONS}
    test = data["test"]
    for index in range(len(test["x"])):
        outputs = ar.forecast(
            model,
            test["x"][index : index + 7],
            previous,
            state,
            7 if key.startswith("retail") else 2,
            ar.CONFIG["samples"],
            seed + index,
        )
        for h in HORIZONS:
            if h <= len(outputs):
                out = outputs[h - 1]
                result[h].append((out["logq"].exp().numpy(), out["mean"].numpy()))
        _, _, state = model(test["x"][index : index + 1], previous[None], state)
        previous = test["y"][index].sum(-1)
    return result


def score(y, q, arrivals, outcomes, groups):
    """Scores from sufficient pattern counts, including zero-arrival days."""
    assert y.shape == q.shape and arrivals.shape == y.shape[:2]
    assert np.isfinite(q).all() and np.isfinite(arrivals).all() and (arrivals >= 0).all()
    np.testing.assert_allclose(q.sum(-1), 1.0, atol=1e-9)
    n = y.sum(-1)
    a = y @ outcomes
    p = q @ outcomes
    total_n = n.sum()
    daily_n = n.sum(-1)
    valid = daily_n > 0
    actual = a.sum(1)
    predicted = (arrivals[..., None] * p).sum(1)
    conditional = (n[..., None] * p).sum(1)
    nll = float(-(y * np.log(q.clip(1e-300))).sum() / total_n)
    brier = (a * (1 - p) ** 2 + (n[..., None] - a) * p**2).sum((0, 1)) / total_n
    rate_error = 100 * (conditional[valid] - actual[valid]) / daily_n[valid, None]
    head_rate_mae = np.abs(rate_error).mean(0)
    count_error = predicted - actual
    cond_error = conditional - actual
    head_count_mae = np.abs(count_error).mean(0)
    group_maes, group_nll, group_events = [], [], []
    for group in sorted(set(groups)):
        cells = groups == group
        gn = n[:, cells].sum(1)
        mask = gn > 0
        ga = a[:, cells].sum(1)
        gp = (n[:, cells, None] * p[:, cells]).sum(1)
        group_maes.append((100 * np.abs(gp[mask] - ga[mask]) / gn[mask, None]).mean(0).tolist())
        group_nll.append(float(-(y[:, cells] * np.log(q[:, cells].clip(1e-300))).sum() / gn.sum()))
        group_events.append(int(gn.sum()))
    pair_indices = list(itertools.combinations(range(outcomes.shape[1]), 2))
    pair_basis = np.stack([outcomes[:, i] * outcomes[:, j] for i, j in pair_indices], axis=1)
    pair_true = (y @ pair_basis).sum(1)
    pair_pred = (n[..., None] * (q @ pair_basis)).sum(1)
    pair_mae = (100 * np.abs(pair_pred[valid] - pair_true[valid]) / daily_n[valid, None]).mean(0)
    conditionals = []
    for index, (i, j) in enumerate(pair_indices):
        observed_denom = actual[:, i].sum()
        predicted_denom = conditional[:, i].sum()
        conditionals.append(
            dict(
                given=i,
                response=j,
                observed=(
                    None
                    if observed_denom == 0
                    else float(pair_true[:, index].sum() / observed_denom)
                ),
                predicted=float(pair_pred[:, index].sum() / max(predicted_denom, 1e-300)),
                observed_condition_count=float(observed_denom),
            )
        )
    calibration, eces = [], []
    for h in range(outcomes.shape[1]):
        bins = np.minimum((p[..., h] * 10).astype(int), 9)
        rows = []
        for b in range(10):
            mask = bins == b
            weight = float(n[mask].sum())
            rows.append(
                dict(
                    bin=b,
                    events=weight,
                    predicted=(
                        None if weight == 0 else float((n[mask] * p[..., h][mask]).sum() / weight)
                    ),
                    observed=None if weight == 0 else float(a[..., h][mask].sum() / weight),
                )
            )
        eces.append(
            100
            * sum(
                row["events"] * abs(row["predicted"] - row["observed"])
                for row in rows
                if row["events"]
            )
            / total_n
        )
        calibration.append(rows)
    metrics = dict(
        joint_nll=nll,
        brier_macro=float(brier.mean()),
        rate_mae_pp=float(head_rate_mae.mean()),
        group_rate_mae_pp=float(np.mean(group_maes)),
        worst_group_rate_mae_pp=float(np.mean(group_maes, axis=1).max()),
        pair_rate_mae_pp=float(pair_mae.mean()),
        action_mae=float(head_count_mae.mean()),
        action_rmse=float(np.sqrt((count_error**2).mean(0)).mean()),
        conditional_action_mae=float(np.abs(cond_error).mean()),
        calibration_ece_pp=float(np.mean(eces)),
        arrival_mae=float(np.abs(arrivals.sum(1) - daily_n).mean()),
        head_brier=brier.tolist(),
        head_rate_mae_pp=head_rate_mae.tolist(),
        head_action_mae=head_count_mae.tolist(),
        head_action_rmse=np.sqrt((count_error**2).mean(0)).tolist(),
        head_conditional_action_mae=np.abs(cond_error).mean(0).tolist(),
        head_calibration_ece_pp=eces,
        group_head_rate_mae_pp=group_maes,
        group_joint_nll=group_nll,
        pair_rate_mae_pp_each=pair_mae.tolist(),
    )
    diagnostic = dict(
        events=int(total_n),
        origins=len(y),
        valid_rate_dates=int(valid.sum()),
        group_events=group_events,
        head_positive_counts=actual.sum(0).tolist(),
        head_prevalence=(actual.sum(0) / total_n).tolist(),
        calibration=calibration,
        conditional_pairs=conditionals,
        daily=dict(
            observed_actions=actual.tolist(),
            predicted_actions=predicted.tolist(),
            conditional_predicted_actions=conditional.tolist(),
            observed_arrivals=daily_n.tolist(),
            predicted_arrivals=arrivals.sum(1).tolist(),
        ),
    )
    return dict(metrics=metrics, diagnostics=diagnostic)


def summarize_runs(runs):
    result = {}
    for h in map(str, HORIZONS):
        records = [r[h]["metrics"] for r in runs.values()]
        result[h] = {}
        for metric in records[0]:
            values = np.asarray([r[metric] for r in records])
            result[h][metric] = dict(
                mean=values.mean(0).tolist(),
                seed_sd=(
                    values.std(0, ddof=1).tolist()
                    if len(values) > 1
                    else np.zeros_like(values[0]).tolist()
                ),
            )
    return result


def snapshot(key, method, seed):
    if method == "projected_gru":
        return ORIGINAL / "kuairand-gru-projection" / key.split("/")[1] / f"{seed}-train.json"
    name = "static_mixture" if key.startswith("kuairand") and method == "no_dynamics" else method
    return ORIGINAL / SLOTS[key].replace("records.json", f"{seed}-{name}-train.json")


def load_model(path):
    form = read(path)["format"]
    if form == "differentiable-market":
        return DifferentiableMarket.load(path)[:3]
    if form == "population-comparison":
        return load_comparison(path)
    return load_projected_gru(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/behavior-evaluation"
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "outputs/gaussian_population/behavior-cache"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepared = {k: prepare(read(ORIGINAL / v)) for k, v in SLOTS.items()}
    methods = {
        k: [
            *VARIANTS,
            *(["separated", "gru", "projected_gru"] if k.startswith("kuairand") else []),
            "deepar_joint",
            *BASELINES,
        ]
        for k in prepared
    }
    snapshots = {
        str(snapshot(k, m, s).relative_to(ROOT)): digest(snapshot(k, m, s))
        for k, ms in methods.items()
        for m in ms
        if m not in [*BASELINES, "deepar_joint"]
        for s in SEEDS
    }
    snapshots.update(
        {
            str(p.relative_to(ROOT)): digest(p)
            for p in EXTERNAL.rglob("*.pt")
            if "deepar_joint" in str(p)
        }
    )
    protocol = dict(
        seeds=SEEDS,
        horizons=HORIZONS,
        methods=methods,
        source_sha256=digest(Path(__file__)),
        inputs={k: digest(ORIGINAL / v) for k, v in SLOTS.items()},
        snapshots=snapshots,
        baseline=dict(
            tau=[1, 10, 100],
            half_life=[1, 3, 7],
            prior="Training pooled pattern counts plus .5 per pattern",
            selection="validation event-weighted joint NLL",
            count="training mean, last cell count, or EWMA cell count; frozen at origin",
        ),
        status="Supplementary analysis after original test results were known. No neural refits or test tuning.",
        scope="Conditional probabilities: actual endpoint exposure weights; end-to-end actions: predicted arrivals times probabilities.",
        averaging="Head macro; event weighted Brier/NLL; day-mean rate MAE; equal groups of within-group date/head MAE; zero-exposure dates excluded only from rates.",
        calibration="10 fixed equal-width probability bins; no fitted recalibration",
        pairs="all unordered pairs; conditional diagnostic predicts later head given earlier head",
        deepar=dict(samples=ar.CONFIG["samples"], source_sha256=digest(Path(ar.__file__))),
    )
    if (args.out / "protocol.json").exists():
        assert (
            read(args.out / "protocol.json") == protocol
        ), "changed protocol requires new directory"
    write_json(args.out / "protocol.json", protocol)
    selections = {k: select_history(p["data"]) for k, p in prepared.items()}
    write_json(args.out / "history-selections.json", selections)
    arselections = read(SAVED / "external-baselines-2026-09-16/selections.json")
    original = read(SAVED / "paper-readiness/summary.json")
    arresults = read(SAVED / "external-baselines-2026-09-16/results.json")
    results = {}
    for key, p in prepared.items():
        results[key] = dict(
            heads=read(snapshot(key, "joint_gaussian", SEEDS[0]))["config"]["heads"],
            groups=sorted(set(p["groups"].tolist())),
            dates=[day.date for day in p["days"]["test"]],
            pairs=list(itertools.combinations(range(p["outcomes"].shape[1]), 2)),
            models={},
        )
        for method in methods[key]:
            runs = {}
            for seed in ([0] if method in BASELINES else SEEDS):
                cache = args.cache / key / f"{method}-{seed}.npz"
                if cache.exists():
                    saved = np.load(cache)
                    forecasts = {
                        h: list(zip(saved[f"q{h}"], saved[f"n{h}"], strict=False)) for h in HORIZONS
                    }
                else:
                    if method in BASELINES:
                        selection = selections[key][method]["selected"]
                        forecasts = history_forecasts(
                            p["data"], "test", method, selection["tau"], selection["half_life"]
                        )
                    elif method == "deepar_joint":
                        forecasts = deepar_forecasts(key, seed, p, arselections)
                    else:
                        model, state, _ = load_model(snapshot(key, method, seed))
                        forecasts = neural_forecasts(model, state, p)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache,
                        **{
                            f"{kind}{h}": np.asarray([x[i] for x in forecasts[h]])
                            for h in HORIZONS
                            for i, kind in enumerate(["q", "n"])
                        },
                    )
                    cache.chmod(0o600)
                runs[str(seed)] = {}
                for h, values in forecasts.items():
                    q, arrivals = map(np.asarray, zip(*values, strict=False))
                    runs[str(seed)][str(h)] = score(
                        p["data"]["test"][h - 1 :], q, arrivals, p["outcomes"], p["groups"]
                    )
                if method == "deepar_joint":
                    for h in HORIZONS:
                        current = runs[str(seed)][str(h)]["metrics"]
                        old = arresults[key][method]["runs"][str(seed)]["summary"][str(h)]
                        np.testing.assert_allclose(
                            [current["joint_nll"], current["arrival_mae"]],
                            [old["joint_nll"], old["mae"]],
                            rtol=1e-9,
                            atol=1e-9,
                        )
                print(key, method, seed, "scored", flush=True)
            summary = summarize_runs(runs)
            if method not in [*BASELINES, "deepar_joint"]:
                old = (
                    original["kuairand"][key.split("/")[1]]["aggregate"]["models"]
                    if key.startswith("kuairand")
                    else original["retail"]["windows"][key.split("/")[1]]["models"]
                )
                name = (
                    "static_mixture"
                    if key.startswith("kuairand") and method == "no_dynamics"
                    else method
                )
                np.testing.assert_allclose(
                    summary["1"]["joint_nll"]["mean"],
                    old[name]["joint_nll"]["mean"],
                    rtol=1e-9,
                    atol=1e-9,
                )
            results[key]["models"][method] = dict(runs=runs, summary=summary)
            write_json(args.out / "results.json", results)
    write_json(
        args.out / "complete.json",
        dict(
            windows=4,
            methods_per_window={k: len(v) for k, v in methods.items()},
            neural_replays=sum(3 * (len(v) - 3) for v in methods.values()),
            history_selections=12,
        ),
    )
    print("Behavior evaluation complete.", flush=True)


if __name__ == "__main__":
    main()

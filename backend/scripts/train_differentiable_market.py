#!/usr/bin/env python3
"""Train and compare differentiable cohort aggregation on chronological KuaiRand.

Public outputs are anonymous aggregates. User-level memory remains in ignored
private preprocessing snapshots; daily cohort features average only prior users'
histories. Forward plans freeze history features at the forecast origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oransim.world_model.differentiable_market import (
    DTYPE,
    DifferentiableMarket,
    MarketBatch,
    MarketConfig,
    MarketDay,
    MarketRuntime,
    fit_market,
    score_days,
)
from train_population_panel import HEADS, initialize, read_data, write

VARIANTS = ("fixed_gaussian", "joint_gaussian", "joint_discrete", "direct", "no_dynamics")
FEATURES = [
    "weekday_sin",
    "weekday_cos",
    "log_prior_exposures",
    "log_prior_campaign_exposures",
    "log_gap_days",
    *["prior_" + h for h in HEADS],
]


def prepare(raw, policy, event_history=False):
    if policy == "standard":
        frame = pd.concat(raw[:2]).sort_values(["date", "time_ms"], kind="stable")
        warm = pd.Timestamp("2022-04-12")
        history = frame[frame.date <= warm]
        train_end = pd.Timestamp("2022-04-21")
        val_end = pd.Timestamp("2022-04-28")
    else:
        frame = raw[2]
        warm = pd.Timestamp("2022-04-24")
        history = raw[0]
        train_end = pd.Timestamp("2022-04-28")
        val_end = pd.Timestamp("2022-05-01")
    panel, _, frame = initialize(frame, history, warm, {})
    campaigns = panel.campaigns
    users = panel.users.copy()
    memory = panel.memory
    # Population denominator is anchored to initialized historical-user stock.
    mass = (
        next(iter(panel.worlds.values())).state.population_weights
        * next(iter(panel.worlds.values())).config.population_size
    )
    groups = np.tile(np.arange(9), len(campaigns))
    cidx = np.repeat(np.arange(len(campaigns)), 9)
    powers = 2 ** np.arange(len(HEADS) - 1, -1, -1)
    members = [[u for u, g in users.items() if g == group] for group in range(9)]
    warmframe = frame[frame.date <= warm]
    days = []
    for date in pd.date_range(warm + pd.Timedelta(days=1), frame.date.max()):
        timestamp = float(date.tz_localize("Asia/Shanghai").timestamp())
        x = []
        for c in campaigns:
            for g in range(9):
                features = (
                    np.mean([memory.features(u, c, timestamp) for u in members[g]], axis=0)
                    if members[g]
                    else np.zeros(3 + len(HEADS))
                )
                x.append(
                    [
                        np.sin(2 * np.pi * date.dayofweek / 7),
                        np.cos(2 * np.pi * date.dayofweek / 7),
                        *features,
                    ]
                )
        counts = np.zeros((len(groups), 2 ** len(HEADS)))
        local = frame[frame.date == date]
        cnum = local.campaign.map({c: i for i, c in enumerate(campaigns)}).to_numpy()
        code = local[HEADS].to_numpy() @ powers
        np.add.at(counts, (cnum * 9 + local.group.to_numpy(), code), 1)
        batch = MarketBatch.create(groups, cidx, x, np.tile(mass, len(campaigns)))
        event_batch = None
        event_y = None
        if event_history and len(local):
            event_x = []
            event_g = []
            event_c = []
            event_values = []
            released = []
            for row in local.itertuples(index=False):
                timestamp = float(row.time_ms) / 1000
                features = memory.features(str(row.user_id), row.campaign, timestamp)
                event_x.append(
                    [
                        np.sin(2 * np.pi * date.dayofweek / 7),
                        np.cos(2 * np.pi * date.dayofweek / 7),
                        *features,
                    ]
                )
                event_g.append(int(row.group))
                event_c.append(campaigns.index(row.campaign))
                y = [int(getattr(row, h)) for h in HEADS]
                event_values.append(y)
                memory.observe(str(row.user_id), row.campaign, timestamp, [0] * len(HEADS))
                released.append((str(row.user_id), row.campaign, timestamp, y))
            for u, c, t, y in released:
                memory.complete(u, c, t, y)
            event_batch = MarketBatch.create(event_g, event_c, event_x, np.ones(len(local)))
            event_y = torch.tensor(event_values, dtype=DTYPE)
        else:
            for row in local.itertuples(index=False):
                memory.observe(
                    str(row.user_id),
                    row.campaign,
                    float(row.time_ms) / 1000,
                    [getattr(row, h) for h in HEADS],
                )
        days.append(
            MarketDay(
                str(date.date()), batch, torch.tensor(counts, dtype=DTYPE), event_batch, event_y
            )
        )
    split = {
        "train": [d for d in days if d.date <= str(train_end.date())],
        "validation": [d for d in days if str(train_end.date()) < d.date <= str(val_end.date())],
        "test": [d for d in days if d.date > str(val_end.date())],
    }
    prior = {}
    for c, world in panel.worlds.items():
        prior[c] = {"means": world.state.means.tolist()}
    # Warm-up counts determine initial observation intercepts; train never sees test labels.
    initial_counts = np.zeros((len(campaigns), 9))
    initial_actions = np.zeros((9, len(HEADS)))
    initial_n = np.zeros(9)
    for row in warmframe.itertuples(index=False):
        ci = campaigns.index(row.campaign)
        initial_counts[ci, row.group] += 1
        initial_n[row.group] += 1
        initial_actions[row.group] += [getattr(row, h) for h in HEADS]
    num_warm = max(1, (warm - frame.date.min()).days + 1)
    global_rate = (warmframe[HEADS].sum().to_numpy() + 0.5) / (len(warmframe) + 1)
    rates = (initial_actions + 20 * global_rate) / (initial_n[:, None] + 20)
    logintensity = np.log((initial_counts + 1) / (num_warm * mass[None, :]))
    next_date = frame.date.max() + pd.Timedelta(days=1)
    next_timestamp = float(next_date.tz_localize("Asia/Shanghai").timestamp())
    next_features = []
    for c in campaigns:
        for g in range(9):
            history_mean = (
                np.mean([memory.features(u, c, next_timestamp) for u in members[g]], axis=0)
                if members[g]
                else np.zeros(3 + len(HEADS))
            )
            next_features.append(
                [
                    np.sin(2 * np.pi * next_date.dayofweek / 7),
                    np.cos(2 * np.pi * next_date.dayofweek / 7),
                    *history_mean,
                ]
            )
    info = {
        "policy": policy,
        "warmup_end": str(warm.date()),
        "train_end": str(train_end.date()),
        "validation_end": str(val_end.date()),
        "test_end": days[-1].date,
        "recorded_events": len(frame),
        "historical_users": len(users),
        "population_mass": mass.tolist(),
        "days": {k: len(v) for k, v in split.items()},
        "events": {k: int(sum(d.patterns.sum() for d in v)) for k, v in split.items()},
    }
    return (
        split,
        logintensity,
        rates,
        info,
        memory.to_dict(),
        torch.tensor(next_features, dtype=DTYPE),
    )


def make_model(seed, variant, logintensity, rates):
    config = MarketConfig(
        9,
        5,
        len(FEATURES),
        tuple(HEADS),
        quadrature=7,
        seed=seed,
        history_feature_indices=tuple(range(2, len(FEATURES))),
    )
    if variant == "fixed_gaussian":
        config = replace(config, learn_population=False)
    elif variant == "joint_discrete":
        config = replace(config, family="discrete", components=3)
    elif variant == "direct":
        config = replace(config, family="point", components=1, learn_population=False)
    elif variant == "no_dynamics":
        config = replace(config, dynamic=False)
    elif variant != "joint_gaussian":
        raise ValueError("unknown variant")
    model = DifferentiableMarket(config)
    # Same non-population initialization across all families, even if RNG shapes differ.
    reference = DifferentiableMarket(
        replace(config, family="gaussian", components=2, learn_population=True, dynamic=True)
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (
                name not in ("mixture_logits", "means", "raw_cholesky")
                and p.shape == dict(reference.named_parameters())[name].shape
            ):
                p.copy_(dict(reference.named_parameters())[name])
        model.base_log_intensity.copy_(torch.tensor(logintensity, dtype=DTYPE))
        model.reference_rates.copy_(torch.tensor(rates, dtype=DTYPE))
        model.behavior_bias.copy_(
            torch.logit(torch.tensor(rates, dtype=DTYPE).clamp(1e-5, 1 - 1e-5))
        )
        if variant == "joint_discrete":
            pi = reference.mixture_logits.softmax(-1)
            mean = (pi[:, :, None] * reference.means).sum(1)
            var = (
                pi[:, :, None]
                * (
                    reference.cholesky().square().sum(-1)
                    + (reference.means - mean[:, None, :]).square()
                )
            ).sum(1)
            model.means.copy_(
                mean[:, None, :]
                + torch.sqrt(1.5 * var)[:, None, :]
                * torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)[None, :, None]
            )
            model.prior_means.copy_(model.means)
    return model


def summarize(rows):
    n = np.array([r["events"] for r in rows])
    pred = np.array([r["predicted_exposures"] for r in rows])
    actual = np.array([r["exposures"] for r in rows])
    return {
        "days": len(rows),
        "events": int(n.sum()),
        "joint_nll": float(np.average([r["joint_nll"] for r in rows], weights=n)),
        "count_nll": float(np.mean([r["count_nll"] for r in rows])),
        "exposure_mae": float(np.abs(pred - actual).mean()),
        "exposure_rmse": float(np.sqrt(np.square(pred - actual).mean())),
        "action_mae": np.abs(
            np.array([r["predicted_actions"] for r in rows])
            - np.array([r["actual_actions"] for r in rows])
        )
        .mean(0)
        .tolist(),
    }


def multi_step(model, split, state):
    rows = []
    for i, day in enumerate(split["test"]):
        for h in (1, 3, 7):
            if i + h > len(split["test"]):
                continue
            plans = [model.future_batch(d.batch, day.batch) for d in split["test"][i : i + h]]
            with torch.no_grad():
                outputs, _ = model.rollout(plans, state)
                out = outputs[-1]
            target = split["test"][i + h - 1]
            # Mark scoring is also at the frozen origin; future memory features stay frozen.
            n = target.patterns.sum()
            nll = float(-(target.patterns * out["joint_log_prob"]).sum() / n.clamp_min(1))
            rows.append(
                {
                    "origin": day.date,
                    "target": target.date,
                    "horizon": h,
                    "events": int(n),
                    "joint_nll": nll,
                    "predicted_exposures": float(out["exposures"].sum()),
                    "exposures": float(n),
                }
            )
        _, state = score_days(model, [day], state)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/kuairand/raw/KuaiRand-Pure/KuaiRand-Pure/data"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/differentiable-market-v1"
    )
    parser.add_argument(
        "--private",
        type=Path,
        default=ROOT / "outputs/gaussian_population/differentiable-market-v1",
    )
    parser.add_argument("--event-history", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260912, 20260913, 20260914])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--policies", nargs="+", default=["standard", "random"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [
        Path(__file__),
        ROOT / "backend/oransim/world_model/differentiable_market.py",
        ROOT / "backend/scripts/train_population_panel.py",
    ]
    protocol = {
        "version": 1,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "variants": args.variants,
        "user_hash_divisor": 64,
        "features": FEATURES,
        "heads": HEADS,
        "individual_event_history": args.event_history,
        "loss": "mean daily joint NLL + 0.05 mean cell NB NLL + 0.15 two-step loss; regularization 0.001 normalized by trainable parameter count",
        "cohorts": "campaign x historical user group; prior individual histories averaged over fixed historical users for macro forecasts",
        "behavior_training": (
            "per-event histories with day-end outcome release"
            if args.event_history
            else "same-context cohort joint pattern counts"
        ),
        "selection": "validation composite objective selects epoch and variant before test; test is reported for every prespecified variant",
        "scope": "Retrospective previously examined KuaiRand dates; no new unseen external holdout. Counts refer to recorded sample exposures, not platform population.",
        "forecast": "History features and population mass freeze at each origin; weekday is known; recurrent memory evolves from generated feedback.",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.data.glob("*.csv")
            if p.name.startswith(("log_", "video_features_basic"))
        },
    }
    if (args.out / "protocol.json").exists() and json.loads(
        (args.out / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("changed protocol requires a new output directory")
    write(args.out / "protocol.json", protocol)
    raw = read_data(args.data, 64)
    results = {}
    for policy in args.policies:
        split, logintensity, rates, info, memory, next_features = prepare(
            raw, policy, args.event_history
        )
        results[policy] = {"data": info, "runs": {}}
        private = args.private / policy
        private.mkdir(parents=True, exist_ok=True)
        # No identifiers in public result artifacts.
        for name, payload in [
            ("records.json", {k: [d.to_dict() for d in v] for k, v in split.items()}),
            ("memory.json", memory),
        ]:
            path = private / name
            path.write_text(json.dumps(payload))
            path.chmod(0o600)
        for seed in args.seeds:
            fitted = {}
            selection = {}
            for variant in args.variants:
                print(f"{policy} {seed} {variant}: fitting", flush=True)
                start = time.perf_counter()
                model = make_model(seed, variant, logintensity, rates)
                train_state, report = fit_market(
                    model,
                    split["train"],
                    split["validation"],
                    epochs=args.epochs,
                    multi_step_weight=0.15,
                    patience=20,
                )
                elapsed = time.perf_counter() - start
                validation, state = score_days(model, split["validation"], train_state)
                selection[variant] = {
                    "fit": report,
                    "validation": summarize(validation),
                    "train_seconds": elapsed,
                    "trainable_parameters": model.parameter_counts(),
                }
                model.save(
                    private / f"{seed}-{variant}-train.json",
                    train_state,
                    {"features": FEATURES, "last_date": info["train_end"]},
                )
                fitted[variant] = (model, state)
                print(
                    f'{policy} {seed} {variant}: validation NLL {summarize(validation)["joint_nll"]:.6f}, {elapsed:.1f}s',
                    flush=True,
                )
            chosen = min(selection, key=lambda v: selection[v]["fit"]["validation_objective"])
            write(
                args.out / policy / f"{seed}-selection.json",
                {"selected": chosen, "candidates": selection},
            )
            for variant, (model, state) in fitted.items():
                rows, end = score_days(model, split["test"], state)
                report = {
                    "test": summarize(rows),
                    "daily": rows,
                    "multi_step": multi_step(model, split, state),
                    "train_seconds": selection[variant]["train_seconds"],
                    "selected_epoch": selection[variant]["fit"]["selected_epoch"],
                    "trainable_parameters": selection[variant]["trainable_parameters"],
                }
                results[policy]["runs"][f"{seed}-{variant}"] = report
                model.save(
                    private / f"{seed}-{variant}-final.json",
                    end,
                    {"features": FEATURES, "last_date": info["test_end"]},
                )
                if variant == chosen:
                    # Complete standalone continuation using the final observed history template.
                    last = split["test"][-1].batch
                    plans = []
                    for h in range(1, 4):
                        x = next_features.clone()
                        date = pd.Timestamp(info["test_end"]) + pd.Timedelta(days=h)
                        x[:, 0] = np.sin(2 * np.pi * date.dayofweek / 7)
                        x[:, 1] = np.cos(2 * np.pi * date.dayofweek / 7)
                        plans.append(
                            MarketBatch(last.group, last.campaign, x, last.population, last.effort)
                        )
                    forecast = MarketRuntime(model, end).forecast(plans, trajectories=64, seed=seed)
                    write(
                        args.out / policy / f"{seed}-continuation.json",
                        {"variant": chosen, "forecast": forecast},
                    )
                    path = private / f"{seed}-continuation-plan.json"
                    write(path, [p.to_dict() for p in plans])
                    path.chmod(0o600)
            write(args.out / "results.json", results)
        print(f"{policy}: all experiments completed", flush=True)


if __name__ == "__main__":
    main()

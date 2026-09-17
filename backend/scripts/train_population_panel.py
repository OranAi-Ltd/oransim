#!/usr/bin/env python3
"""Chronological multi-action, remembered-user and shared-environment research audit.

Snapshots containing user memory are written only to an ignored private directory.
Public results contain aggregates and anonymous campaign indices.
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
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oransim.world_model.population_behavior import BehaviorBatch
from oransim.world_model.population_loop import LoopConfig, PopulationWorldLoop, WorldAction
from oransim.world_model.population_panel import PanelEvent, PopulationWorldPanel
from run_kuairand_population_dynamics import fit_segments

HEADS = ["is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward", "is_hate"]
VARIANTS = {
    "full": {},
    "no_memory": {"memory": False},
    "no_action_dependencies": {"dependent": False},
    "no_shared_environment": {"shared": False},
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_data(path, divisor):
    names = [
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ]
    frames = []
    for name in names:
        frame = (
            pd.read_csv(
                path / name, usecols=["user_id", "video_id", "time_ms", "date", "tab", *HEADS]
            )
            .query("tab == 1")
            .copy()
        )
        chosen = {
            int(u)
            for u in frame.user_id.unique()
            if int.from_bytes(hashlib.sha256(str(int(u)).encode()).digest()[:8], "big") % divisor
            == 0
        }
        frame = frame[frame.user_id.isin(chosen)].copy()
        # Source date buckets overlap in absolute time. Derive a single monotone
        # calendar from event timestamps, so memory never consumes a future row.
        frame["date"] = (
            pd.to_datetime(frame.time_ms, unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        frames.append(frame)
    metadata = pd.read_csv(
        path / "video_features_basic_pure.csv", usecols=["video_id", "author_id"]
    )
    for i, frame in enumerate(frames):
        frames[i] = frame.merge(
            metadata, on="video_id", how="left", validate="many_to_one"
        ).sort_values(["date", "time_ms"], kind="stable")
    return frames


def initialize(frame, history, warmup_end, options):
    warmup = frame[frame.date <= warmup_end].copy()
    segments, _ = fit_segments(history)
    top = warmup.author_id.value_counts().head(4).index.tolist()
    mapping = {author: f"campaign_{i}" for i, author in enumerate(top)}
    campaigns = [*mapping.values(), "campaign_other"]
    frame = frame.copy()
    frame["campaign"] = frame.author_id.map(mapping).fillna("campaign_other")
    frame["group"] = frame.user_id.map(segments).fillna(8).astype(int)
    warmup = frame[frame.date <= warmup_end]
    users = segments.value_counts().reindex(range(9), fill_value=0).to_numpy(float) + 1
    q = users / users.sum()
    global_rate = (warmup[HEADS].sum().to_numpy() + 0.5) / (len(warmup) + 1)
    worlds = {}
    actions = {}
    for campaign in campaigns:
        local = warmup[warmup.campaign == campaign]
        nk = local.groupby("group").is_click.agg(["sum", "size"]).reindex(range(9), fill_value=0)
        rate = (nk["sum"].to_numpy() + 20 * global_rate[0]) / (nk["size"].to_numpy() + 20)
        counts = nk["size"].to_numpy() + 1
        mix = counts / counts.sum()
        cfg = LoopConfig(
            base_arrivals=max(1.0, float((local.date == warmup_end).sum())),
            population_size=float(users.sum()),
            novelty_decay=0,
            response_retention=0.99,
            response_process_variance=0.005,
            selection_process_variance=0.03,
            arrival_process_variance=0,
            recommendation_gain=0,
            fatigue_selection_gain=0,
            momentum_arrival_gain=0,
            social_gain=0,
            fatigue_gain=0,
            mixture_gain=0,
        )
        worlds[campaign] = PopulationWorldLoop(
            q,
            np.full((9, 2), 0.5),
            logit(rate)[:, None] + np.array([-0.4, 0.4]),
            np.full((9, 2), 0.1),
            cfg,
        )
        actions[campaign] = WorldAction(np.zeros(9), np.log(mix / q))
    panel = PopulationWorldPanel(worlds, HEADS, global_rate, **options)
    # Warmup is entirely before the first evaluated/training event. It supplies
    # population priors and persistent history, without double-filtering counts.
    for row in warmup.itertuples(index=False):
        panel.memory.observe(
            str(row.user_id),
            row.campaign,
            float(row.time_ms) / 1000,
            [getattr(row, h) for h in HEADS],
        )
        panel.users[str(row.user_id)] = int(row.group)
    panel.last_timestamp = float(
        (warmup_end + pd.Timedelta(days=1)).tz_localize("Asia/Shanghai").timestamp()
    )
    # KuaiRand log timestamps are checked rather than inferred from row position.
    after = frame[frame.date > warmup_end]
    if len(after) and after.time_ms.min() / 1000 < panel.last_timestamp:
        raise ValueError("date partition disagrees with event timestamps")
    return panel, actions, frame


def events_for(frame, date):
    rows = frame[frame.date == date]
    return [
        PanelEvent(
            float(row.time_ms) / 1000,
            str(row.user_id),
            row.campaign,
            int(row.group),
            tuple(int(getattr(row, h)) for h in HEADS),
        )
        for row in rows.itertuples(index=False)
    ]


def replay(panel, actions, frame, dates, score=False):
    rows = []
    batches = []
    for date in dates:
        result, batch = panel.observe_day(
            events_for(frame, date),
            actions,
            score=score,
            end_timestamp=float(
                (date + pd.Timedelta(days=1)).tz_localize("Asia/Shanghai").timestamp()
            ),
        )
        result["date"] = str(date.date())
        rows.append(result)
        if batch is not None:
            batches.append(batch)
    return rows, batches


def metrics(rows):
    selected = [r for r in rows if r["metrics"]]
    weights = np.array([r["metrics"]["events"] for r in selected])
    actual = np.array([r["observed_exposures"] for r in rows])
    predicted = np.array([r["predicted_exposures"] for r in rows])
    return {
        "days": len(rows),
        "events": int(weights.sum()),
        "joint_nll": float(
            np.average([r["metrics"]["joint_nll"] for r in selected], weights=weights)
        ),
        "head_log_loss": np.average(
            [r["metrics"]["head_log_loss"] for r in selected], weights=weights, axis=0
        ).tolist(),
        "exposure_mae_per_campaign": np.abs(actual - predicted).mean(0).tolist(),
        "total_exposure_mae": float(np.abs(actual.sum(1) - predicted.sum(1)).mean()),
    }


def last_day_baseline(frame, train_dates, test_dates):
    train = frame[frame.date.isin(train_dates)]
    global_rate = (train[HEADS].sum().to_numpy() + 0.5) / (len(train) + 1)
    aggregate = train.groupby("group")[HEADS].sum().reindex(range(9), fill_value=0).to_numpy()
    n = train.groupby("group").size().reindex(range(9), fill_value=0).to_numpy()
    prior = (aggregate + 20 * global_rate) / (n[:, None] + 20)
    loss = []
    total = 0
    count_errors = []
    for date in test_dates:
        earlier = frame[frame.date == date - pd.Timedelta(days=1)]
        k = earlier.groupby("group")[HEADS].sum().reindex(range(9), fill_value=0).to_numpy()
        n = earlier.groupby("group").size().reindex(range(9), fill_value=0).to_numpy()
        p = (k + 20 * prior) / (n[:, None] + 20)
        current = frame[frame.date == date]
        y = current[HEADS].to_numpy()
        prob = np.clip(p[current.group.to_numpy()], 1e-9, 1 - 1e-9)
        loss.append((-(y * np.log(prob) + (1 - y) * np.log1p(-prob))).sum(0))
        total += len(y)
        count_errors.append(abs(len(current) - len(earlier)))
    head = np.sum(loss, axis=0) / total
    return {
        "joint_nll": float(head.sum()),
        "head_log_loss": head.tolist(),
        "total_exposure_mae": float(np.mean(count_errors)),
        "scope": "Independent binary heads, smoothed previous-day group rates; previous-day total exposure count.",
    }


def run(args):
    names = [
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
        "video_features_basic_pure.csv",
    ]
    source_names = [
        "backend/scripts/train_population_panel.py",
        "backend/oransim/world_model/population_panel.py",
        "backend/oransim/world_model/population_behavior.py",
        "backend/oransim/world_model/population_loop.py",
        "backend/oransim/data/gaussian_population_dynamics.py",
        "backend/scripts/run_kuairand_population_dynamics.py",
    ]
    protocol = {
        "version": "population-panel-v4",
        "heads": HEADS,
        "user_hash_divisor": args.user_divisor,
        "max_train_events": args.max_train_events,
        "max_iterations": args.max_iterations,
        "conditional_fit_rounds": 2,
        "variants": VARIANTS,
        "seed": args.seed,
        "forecast_trajectories": args.trajectories,
        "selection": "Lowest validation event-weighted joint negative log probability among four variants, before test replay.",
        "availability": "All same-day exposure memory is available in event order; outcomes become available at the day boundary. No current-day outcomes enter user features.",
        "calendar": "Asia/Shanghai dates derived from time_ms. The original date column has overlapping absolute time ranges and is not used to order events or define train/test boundaries.",
        "campaigns": "Top four authors in policy-specific warmup plus a fixed other bucket; campaign names in public artifacts are anonymous.",
        "population": "Historical user groups; fixed two-component Gaussian priors; no individual mixture recovery claim.",
        "history_notice": "Retrospective reuse of previously examined dates; no pristine external holdout claim.",
        "data_sha256": {n: hashlib.sha256((args.data / n).read_bytes()).hexdigest() for n in names},
        "code_sha256": {
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in source_names
        },
    }
    if (args.out / "protocol.json").exists() and json.loads(
        (args.out / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("changed protocol requires a new output directory")
    write(args.out / "protocol.json", protocol)
    raw = read_data(args.data, args.user_divisor)
    results = {}
    for policy, frame, warm_end, train_end, val_end in [
        (
            "standard",
            pd.concat(raw[:2]).sort_values(["date", "time_ms"], kind="stable"),
            "2022-04-12",
            "2022-04-21",
            "2022-04-28",
        ),
        ("random", raw[2], "2022-04-24", "2022-04-28", "2022-05-01"),
    ]:
        warm_end = pd.Timestamp(warm_end)
        train_end = pd.Timestamp(train_end)
        val_end = pd.Timestamp(val_end)
        history = raw[0] if policy == "random" else frame[frame.date <= warm_end]
        train_dates = pd.date_range(warm_end + pd.Timedelta(days=1), train_end)
        val_dates = pd.date_range(train_end + pd.Timedelta(days=1), val_end)
        test_dates = pd.date_range(val_end + pd.Timedelta(days=1), frame.date.max())
        candidates = {}
        validation = {}
        fits = {}
        for name, options in VARIANTS.items():
            print(f"{policy}: train {name}", flush=True)
            initial, actions, prepared = initialize(frame, history, warm_end, options)
            counts = np.array(
                [
                    [
                        len(prepared[(prepared.date == date) & (prepared.campaign == c)])
                        for c in initial.campaigns
                    ]
                    for date in train_dates
                ],
                float,
            )
            scales = np.tile(
                [initial.worlds[c].config.base_arrivals for c in initial.campaigns],
                (len(train_dates), 1),
            )
            arrival_fit = initial.arrivals.fit(
                counts, scales, np.full(len(initial.campaigns), 50.0), shared=initial.shared_enabled
            )
            reports = []
            behavior = deepcopy(initial.behavior)
            for iteration in range(2):
                training = deepcopy(initial)
                training.behavior = deepcopy(behavior)
                _, batches = replay(training, actions, prepared, train_dates)
                batch = BehaviorBatch.concatenate(batches)
                if len(batch.features) > args.max_train_events:
                    indices = np.sort(
                        np.random.default_rng(args.seed).choice(
                            len(batch.features), args.max_train_events, replace=False
                        )
                    )
                    batch = batch.take(indices)
                reports.append(behavior.fit(batch, args.max_iterations))
            trained = deepcopy(initial)
            trained.behavior = behavior
            replay(trained, actions, prepared, train_dates)
            fits[name] = {"arrival": arrival_fit, "behavior_rounds": reports}
            trained.save(args.private / policy / f"{name}-train.json")
            rows, _ = replay(trained, actions, prepared, val_dates, score=True)
            validation[name] = {"metrics": metrics(rows), "daily": rows}
            candidates[name] = trained
        selected = min(validation, key=lambda n: validation[n]["metrics"]["joint_nll"])
        write(
            args.out / policy / "selection.json",
            {"selected": selected, "validation": validation, "fits": fits},
        )
        # Frozen trajectory before any test labels enter the selected model.
        print(f"{policy}: selected {selected}; frozen seven-day simulation", flush=True)
        forecast = candidates[selected].forecast(
            [actions] * min(7, len(test_dates)), args.trajectories, args.seed, max_events=100000
        )
        write(args.out / policy / "frozen-forecast.json", forecast)
        tests = {}
        for name, model in candidates.items():
            print(f"{policy}: test {name}", flush=True)
            rows, _ = replay(model, actions, prepared, test_dates, score=True)
            tests[name] = {"metrics": metrics(rows), "daily": rows}
            model.save(args.private / policy / f"{name}-final.json")
            if (
                PopulationWorldPanel.load(args.private / policy / f"{name}-final.json").to_dict()
                != model.to_dict()
            ):
                raise RuntimeError("panel snapshot mismatch")
        plan = [
            {
                c: {
                    "content_logits": a.content_logits.tolist(),
                    "targeting_logits": a.targeting_logits.tolist(),
                }
                for c, a in actions.items()
            }
        ] * 3
        write(args.out / policy / "continuation-plan.json", plan)
        results[policy] = {
            "selected": selected,
            "test": tests,
            "baseline": last_day_baseline(prepared, train_dates, test_dates),
            "split": {
                "warmup_end": str(warm_end.date()),
                "train_end": str(train_end.date()),
                "validation_end": str(val_end.date()),
                "test_end": str(test_dates[-1].date()),
            },
            "sample": {
                "events": len(prepared),
                "users": int(prepared.user_id.nunique()),
                "campaigns": len(actions),
                "training_positive_counts": prepared[prepared.date.isin(train_dates)][HEADS]
                .sum()
                .astype(int)
                .tolist(),
            },
            "private_model_directory": str((args.private / policy).relative_to(ROOT)),
        }
        write(args.out / "results.json", results)
        print(f"{policy}: complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/kuairand/raw/KuaiRand-Pure/KuaiRand-Pure/data"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/population-panel-v4"
    )
    parser.add_argument(
        "--private", type=Path, default=ROOT / "outputs/gaussian_population/population-panel-v4"
    )
    parser.add_argument("--user-divisor", type=int, default=64)
    parser.add_argument("--max-train-events", type=int, default=20000)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260908)
    run(parser.parse_args())

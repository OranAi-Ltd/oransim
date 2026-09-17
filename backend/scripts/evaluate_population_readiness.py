#!/usr/bin/env python3
"""Frozen supplementary KuaiRand cohort, aggregation and comparator experiment.

The user hash bucket is disjoint from the differentiable-market-v3 bucket.
Calendar dates are reused, so this is a population-sample replication, not a
new temporal holdout. Pure-aggregate training reads only cohort counts and
lagged cohort summaries after historical group initialization.
"""

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
from evaluate_population_contributions import baseline_paths, fit_log_trend
from oransim.world_model.differentiable_market import (
    DTYPE,
    MarketBatch,
    MarketDay,
    fit_market,
    score_days,
)
from oransim.world_model.population_comparisons import (
    GRUMarket,
    SeparatedAggregationMarket,
    load_comparison,
    save_comparison,
)
from train_differentiable_market import HEADS, make_model, multi_step, prepare, summarize, write

VARIANTS = [
    "joint_gaussian",
    "fixed_gaussian",
    "joint_discrete",
    "separated",
    "static_mixture",
    "direct",
    "gru",
]


def read_bucket(path, bucket):
    frames = []
    for name in [
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ]:
        frame = (
            pd.read_csv(path / name, usecols=["user_id", "video_id", "time_ms", "tab", *HEADS])
            .query("tab == 1")
            .copy()
        )
        chosen = {
            int(u)
            for u in frame.user_id.unique()
            if int.from_bytes(hashlib.sha256(str(int(u)).encode()).digest()[:8], "big") % 64
            == bucket
        }
        frame = frame[frame.user_id.isin(chosen)].copy()
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
    return [
        f.merge(metadata, on="video_id", how="left", validate="many_to_one").sort_values(
            ["date", "time_ms"], kind="stable"
        )
        for f in frames
    ]


def aggregate_records(split, logintensity, rates):
    template = split["train"][0].batch
    g = template.group.numpy()
    old_n = np.exp(logintensity).reshape(-1) * template.population.numpy()
    old_a = old_n[:, None] * rates[g]
    result = {}
    for name, days in split.items():
        result[name] = []
        for day in days:
            group_n = np.bincount(g, weights=old_n, minlength=9)
            x = day.batch.features.numpy().copy()
            x[:, 2] = np.log1p(group_n[g] / template.population.numpy().clip(1))
            x[:, 3] = np.log1p(old_n / template.population.numpy().clip(1))
            x[:, 4] = 0.0
            x[:, 5:] = (old_a + 2 * rates[g]) / (old_n[:, None] + 2)
            batch = MarketBatch(
                day.batch.group,
                day.batch.campaign,
                torch.tensor(x, dtype=DTYPE),
                day.batch.population,
                day.batch.effort,
            )
            result[name].append(MarketDay(day.date, batch, day.patterns.clone()))
            old_n = 0.5 * old_n + 0.5 * day.patterns.sum(-1).numpy()
            outcomes = np.array(
                list(__import__("itertools").product([0.0, 1.0], repeat=len(HEADS)))
            )
            old_a = 0.5 * old_a + 0.5 * (day.patterns.numpy() @ outcomes)
    return result


def comparator(seed, variant, logintensity, rates):
    mapped = (
        "no_dynamics"
        if variant == "static_mixture"
        else ("joint_gaussian" if variant in ("separated", "gru") else variant)
    )
    reference = make_model(seed, mapped, logintensity, rates)
    if variant == "separated":
        model = SeparatedAggregationMarket(reference.config)
        model.load_state_dict(reference.state_dict())
        return model
    if variant == "gru":
        model = GRUMarket(
            replace(
                reference.config, family="point", components=1, dimensions=4, learn_population=False
            )
        )
        with torch.no_grad():
            own = model.state_dict()
            for name, value in reference.state_dict().items():
                if (
                    name
                    not in (
                        "means",
                        "prior_means",
                        "mixture_logits",
                        "raw_cholesky",
                        "nodes",
                        "node_weights",
                    )
                    and name in own
                    and own[name].shape == value.shape
                ):
                    own[name].copy_(value)
        return model
    return reference


def baselines(split):
    days = sum(split.values(), [])
    values = np.array([float(d.patterns.sum()) for d in days])
    start = len(split["train"]) + len(split["validation"])
    trend = fit_log_trend(values, 0, len(split["train"]))
    rows = []
    for i in range(start, len(days)):
        paths = baseline_paths(values[:i], min(7, len(days) - i), trend)
        for h in [1, 3, 7]:
            if i + h > len(days):
                continue
            rows.append(
                {
                    "origin": days[i].date,
                    "target": days[i + h - 1].date,
                    "horizon": h,
                    "observed": float(values[i + h - 1]),
                    "predictions": {name: float(v[h - 1]) for name, v in paths.items()},
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/kuairand/raw/KuaiRand-Pure/KuaiRand-Pure/data"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/paper-readiness/kuairand"
    )
    parser.add_argument(
        "--private", type=Path, default=ROOT / "outputs/gaussian_population/checkpoints/kuairand"
    )
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [
        Path(__file__),
        ROOT / "backend/oransim/world_model/population_comparisons.py",
        ROOT / "backend/oransim/world_model/differentiable_market.py",
        ROOT / "backend/scripts/train_differentiable_market.py",
        ROOT / "backend/scripts/train_population_panel.py",
        ROOT / "backend/oransim/world_model/population_panel.py",
    ]
    protocol = {
        "bucket": 1,
        "modulus": 64,
        "excluded_v3_bucket": 0,
        "seeds": [20260915, 20260916, 20260917],
        "regimes": {"aggregate": VARIANTS, "event": ["joint_gaussian"]},
        "epochs": args.epochs,
        "patience": 20,
        "selection": "same validation daily mark NLL + .05 mean-cell NB NLL, no test-dependent choices",
        "training": "Adam .025, clip10, normalized L2 .001, two-step weight .15; inherited v3 training",
        "aggregate_visibility": "Historical fixed group priors; thereafter only cohort joint pattern counts, weekday and prior cohort count/mark EMA. No per-user history features or event likelihood.",
        "event_visibility": "Same events and dates; individual history is additional observed information. Cross-regime differences include history granularity.",
        "sample_scope": "Hash bucket disjoint from v3. Previously studied calendar dates. This is population-sample replication, not a pristine temporal holdout.",
        "comparisons": {
            "primary": "aggregate joint_gaussian versus fixed_gaussian",
            "mechanism": "joint versus separated expectation with identical parameters; static mixture and deterministic GRU alternatives",
            "information": "event joint_gaussian versus aggregate joint_gaussian",
            "uncertainty": "Seed mean/SD plus exploratory paired circular 3-day block-bootstrap over test dates; no multiple-comparison significance claim",
        },
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
        raise ValueError("changed protocol requires new output directory")
    write(args.out / "protocol.json", protocol)
    raw = read_bucket(args.data, 1)
    results = {}
    for policy in ["standard", "random"]:
        event, logintensity, rates, info, _, _ = prepare(raw, policy, event_history=True)
        aggregate = aggregate_records(event, logintensity, rates)
        results[policy] = {"data": info, "regimes": {}}
        for regime, split in [("aggregate", aggregate), ("event", event)]:
            private = args.private / policy / regime
            path = private / "records.json"
            write(path, {k: [d.to_dict() for d in v] for k, v in split.items()})
            path.chmod(0o600)
            result = {"runs": {}, "count_baselines": baselines(split)}
            results[policy]["regimes"][regime] = result
            for seed in protocol["seeds"]:
                fitted = {}
                selected = {}
                for variant in protocol["regimes"][regime]:
                    print(f"{policy} {regime} {seed} {variant}: fitting", flush=True)
                    model = comparator(seed, variant, logintensity, rates)
                    start = time.perf_counter()
                    state, fit = fit_market(
                        model,
                        split["train"],
                        split["validation"],
                        epochs=args.epochs,
                        multi_step_weight=0.15,
                        patience=20,
                    )
                    elapsed = time.perf_counter() - start
                    save_comparison(
                        model,
                        private / f"{seed}-{variant}-train.json",
                        state,
                        {"last_date": info["train_end"]},
                    )
                    val, end = score_days(model, split["validation"], state)
                    selected[variant] = {
                        "fit": fit,
                        "validation": summarize(val),
                        "seconds": elapsed,
                        "allocated_trainable": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                    }
                    fitted[variant] = (model, end)
                    print(
                        f'{policy} {regime} {seed} {variant}: validation {summarize(val)["joint_nll"]:.6f}; {elapsed:.1f}s',
                        flush=True,
                    )
                winner = min(selected, key=lambda v: selected[v]["fit"]["validation_objective"])
                write(
                    args.out / policy / regime / f"{seed}-selection.json",
                    {"selected": winner, "candidates": selected},
                )
                for variant, (model, state) in fitted.items():
                    rows, end = score_days(model, split["test"], state)
                    result["runs"][f"{seed}-{variant}"] = {
                        "test": summarize(rows),
                        "daily": rows,
                        "multi_step": multi_step(model, split, state),
                        "train_seconds": selected[variant]["seconds"],
                        "selected_epoch": selected[variant]["fit"]["selected_epoch"],
                        "allocated_trainable": selected[variant]["allocated_trainable"],
                    }
                    path = private / f"{seed}-{variant}-final.json"
                    save_comparison(model, path, end, {"last_date": info["test_end"]})
                    restored, rs, _ = load_comparison(path)
                    with torch.no_grad():
                        torch.testing.assert_close(
                            restored(split["test"][-1].batch, rs)["actions"],
                            model(split["test"][-1].batch, end)["actions"],
                        )
                write(args.out / "results.json", results)
    print("All population-sample and observation-granularity runs complete.", flush=True)


if __name__ == "__main__":
    main()

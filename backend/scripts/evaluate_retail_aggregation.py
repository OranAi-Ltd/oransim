#!/usr/bin/env python3
"""Independent aggregate-only marked-transaction evaluation on UCI Online Retail II.

The event unit is an observed positive invoice. Population anchors are historical
customer counts, never visits or purchase opportunities. Models receive only
daily cohort pattern counts, calendar features and lagged aggregate summaries.
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
from oransim.world_model.differentiable_market import (
    DTYPE,
    DifferentiableMarket,
    MarketBatch,
    MarketConfig,
    MarketDay,
    fit_market,
    score_days,
)

HEADS = ("high_value", "high_quantity", "many_skus")
FEATURES = [
    *[f"weekday_{i}" for i in range(7)],
    "log_lag_arrivals_per_anchor",
    *[f"lag_{h}_rate" for h in HEADS],
]
VARIANTS = ("fixed_gaussian", "joint_gaussian", "joint_discrete", "direct", "no_dynamics")
WINDOWS = {
    "2010": {
        "warmup_start": "2009-12-01",
        "warmup_end": "2009-12-31",
        "train_end": "2010-02-28",
        "validation_end": "2010-03-14",
        "test_end": "2010-03-31",
    },
    "2011": {
        "warmup_start": "2010-12-01",
        "warmup_end": "2010-12-31",
        "train_end": "2011-02-28",
        "validation_end": "2011-03-14",
        "test_end": "2011-03-31",
    },
}


def write(path, value, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if private:
        path.chmod(0o600)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_cache(path):
    if path.exists():
        return
    workbook = path.parent.parent / "raw/online_retail_II.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"download/extract official UCI workbook to {workbook}")
    sheets = pd.read_excel(workbook, sheet_name=None)
    frame = pd.concat(sheets.values(), ignore_index=True)
    for col in ("Invoice", "StockCode", "Description", "Country"):
        frame[col] = frame[col].astype("string")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def invoices(raw):
    frame = pd.read_parquet(raw)
    audit = {"raw_rows": len(frame), "exact_duplicate_rows_removed": int(frame.duplicated().sum())}
    frame = frame.drop_duplicates().copy()
    # Invoice prefixes identify cancellations. Merchandise codes begin with digits;
    # POST/DOT/discount/bank-charge/manual entries are not product quantities.
    stages = [
        ("cancellation_rows", ~frame.Invoice.str.upper().str.startswith("C", na=False)),
        ("nonpositive_quantity_or_price_rows", (frame.Quantity > 0) & (frame.Price > 0)),
        ("non_merchandise_rows", frame.StockCode.str.match(r"^\d", na=False)),
        ("missing_invoice_or_time_rows", frame.Invoice.notna() & frame.InvoiceDate.notna()),
    ]
    for name, mask in stages:
        mask = mask.reindex(frame.index)
        audit[name + "_removed"] = int((~mask).sum())
        frame = frame[mask].copy()
    frame["value"] = frame.Quantity * frame.Price
    grouped = frame.groupby("Invoice", sort=False, dropna=False)
    # An invoice whose customer or timestamp is internally inconsistent is excluded.
    inconsistent = grouped.agg(
        customers=("Customer ID", lambda x: x.nunique(dropna=False)),
        timestamps=("InvoiceDate", "nunique"),
    )
    bad = inconsistent.index[(inconsistent.customers > 1) | (inconsistent.timestamps > 1)]
    audit["inconsistent_invoices_removed"] = len(bad)
    frame = frame[~frame.Invoice.isin(bad)]
    out = (
        frame.groupby("Invoice", sort=False)
        .agg(
            date=("InvoiceDate", "min"),
            customer=("Customer ID", "first"),
            value=("value", "sum"),
            quantity=("Quantity", "sum"),
            skus=("StockCode", "nunique"),
        )
        .reset_index(drop=True)
    )
    out["date"] = out.date.dt.normalize()
    audit.update(
        {
            "retained_line_rows": len(frame),
            "positive_merchandise_invoices": len(out),
            "anonymous_invoices": int(out.customer.isna().sum()),
            "known_customer_ids": int(out.customer.nunique()),
            "retained_date_min": str(out.date.min().date()),
            "retained_date_max": str(out.date.max().date()),
        }
    )
    return out, audit


def prepare(frame, window):
    w = WINDOWS[window]
    warm = frame[frame.date.between(w["warmup_start"], w["warmup_end"])].copy()
    stats = (
        warm.dropna(subset=["customer"])
        .groupby("customer")
        .agg(visits=("value", "size"), spend=("value", "sum"))
    )
    repeats = stats[stats.visits > 1]
    spend_threshold = float(repeats.spend.median()) if len(repeats) else float(stats.spend.median())
    groups = pd.Series(
        np.where(stats.visits == 1, 0, np.where(stats.spend <= spend_threshold, 1, 2)),
        index=stats.index,
    )
    # Unknown customers are an arrival stream on one fixed reference unit.
    # No future count of visitors or future customer membership is supplied.
    mass = np.r_[groups.value_counts().reindex(range(3), fill_value=0).to_numpy(float), 1.0]
    if np.any(mass[:3] <= 0):
        raise ValueError("the frozen grouping requires three populated warmup cohorts")
    thresholds = {key: float(warm[key].median()) for key in ("value", "quantity", "skus")}
    local = frame[frame.date.between(w["warmup_start"], w["test_end"])].copy()
    local["group"] = local.customer.map(groups).fillna(3).astype(int)
    local["pattern"] = sum(
        (local[key] > thresholds[key]).astype(int) * power
        for key, power in zip(("value", "quantity", "skus"), (4, 2, 1), strict=False)
    )
    full = pd.date_range(w["warmup_start"], w["test_end"])
    table = np.zeros((len(full), 4, 8), dtype=float)
    di = ((local.date - full[0]) / pd.Timedelta(days=1)).to_numpy(int)
    np.add.at(table, (di, local.group.to_numpy(), local.pattern.to_numpy()), 1.0)
    outcome = np.array(list(__import__("itertools").product((0.0, 1.0), repeat=3)))
    warm_n = (pd.Timestamp(w["warmup_end"]) - full[0]).days + 1
    warm_counts = table[:warm_n].sum(0)
    warm_events = warm_counts.sum(1)
    global_rate = (warm_counts.sum(0) @ outcome + 0.5) / (warm_events.sum() + 1.0)
    rates = (warm_counts @ outcome + 20.0 * global_rate) / (warm_events[:, None] + 20.0)
    intercept = np.log((warm_events + 1.0) / (warm_n * mass))[None, :]
    ema_n = warm_events / warm_n
    ema_a = (warm_counts @ outcome) / warm_n
    days = []
    for i, date in enumerate(full[warm_n:], warm_n):
        weekday = np.tile(np.eye(7)[date.dayofweek], (4, 1))
        past = np.c_[np.log1p(ema_n / mass), (ema_a + 2.0 * global_rate) / (ema_n[:, None] + 2.0)]
        batch = MarketBatch.create(np.arange(4), np.zeros(4, dtype=int), np.c_[weekday, past], mass)
        patterns = table[i]
        days.append(MarketDay(str(date.date()), batch, torch.tensor(patterns, dtype=DTYPE)))
        ema_n = 0.85 * ema_n + 0.15 * patterns.sum(1)
        ema_a = 0.85 * ema_a + 0.15 * (patterns @ outcome)
    split = {
        "train": [d for d in days if d.date <= w["train_end"]],
        "validation": [d for d in days if w["train_end"] < d.date <= w["validation_end"]],
        "test": [d for d in days if d.date > w["validation_end"]],
    }
    info = {
        **w,
        "window": window,
        "warmup_invoices": int(warm_events.sum()),
        "known_warmup_customers": len(stats),
        "population_anchors": mass.tolist(),
        "unknown_group_anchor": "one fixed reference unit; not a person/visit/opportunity count",
        "repeat_spend_threshold_gbp": spend_threshold,
        "mark_thresholds": thresholds,
        "days": {k: len(v) for k, v in split.items()},
        "invoices": {k: int(sum(d.patterns.sum() for d in v)) for k, v in split.items()},
        "group_invoices": {
            k: torch.stack([d.patterns.sum(1) for d in v]).sum(0).int().tolist()
            for k, v in split.items()
        },
    }
    return split, intercept, rates, info, table[:warm_n].sum((1, 2)).tolist()


def make_model(seed, variant, intercept, rates):
    config = MarketConfig(
        4,
        1,
        len(FEATURES),
        HEADS,
        quadrature=7,
        seed=seed,
        history_feature_indices=tuple(range(7, len(FEATURES))),
    )
    reference = DifferentiableMarket(config)
    if variant == "fixed_gaussian":
        config = replace(config, learn_population=False)
    elif variant == "joint_discrete":
        config = replace(config, family="discrete", components=3)
    elif variant == "direct":
        config = replace(config, family="point", components=1, learn_population=False)
    elif variant == "no_dynamics":
        config = replace(config, dynamic=False)
    elif variant != "joint_gaussian":
        raise ValueError(variant)
    model = DifferentiableMarket(config)
    with torch.no_grad():
        ref_parameters = dict(reference.named_parameters())
        for name, value in model.named_parameters():
            if name not in ("mixture_logits", "means", "raw_cholesky"):
                value.copy_(ref_parameters[name])
        model.base_log_intensity.copy_(torch.tensor(intercept, dtype=DTYPE))
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
    events = np.array([r["events"] for r in rows])
    error = np.array([r["predicted_exposures"] - r["exposures"] for r in rows])
    return {
        "days": len(rows),
        "transactions": int(events.sum()),
        "joint_nll": float(np.average([r["joint_nll"] for r in rows], weights=events)),
        "count_nll": float(np.mean([r["count_nll"] for r in rows])),
        "transaction_mae": float(np.abs(error).mean()),
        "transaction_rmse": float(np.sqrt(np.square(error).mean())),
    }


def forecasts(model, split, state):
    rows = []
    for i, day in enumerate(split["test"]):
        for horizon in (1, 3, 7):
            if i + horizon > len(split["test"]):
                continue
            plans = [model.future_batch(d.batch, day.batch) for d in split["test"][i : i + horizon]]
            with torch.no_grad():
                out = model.rollout(plans, state)[0][-1]
            target = split["test"][i + horizon - 1]
            n = float(target.patterns.sum())
            rows.append(
                {
                    "origin": day.date,
                    "target": target.date,
                    "horizon": horizon,
                    "events": int(n),
                    "joint_nll": float(
                        -(target.patterns * out["joint_log_prob"]).sum() / max(n, 1)
                    ),
                    "transactions": n,
                    "predicted_transactions": float(out["exposures"].sum()),
                }
            )
        _, state = score_days(model, [day], state)
    return rows


def baselines(split, warm_totals):
    past = list(warm_totals) + [
        float(d.patterns.sum()) for d in split["train"] + split["validation"]
    ]
    rows = []
    for i, day in enumerate(split["test"]):
        for horizon in (1, 3, 7):
            if i + horizon > len(split["test"]):
                continue
            target = split["test"][i + horizon - 1]
            rows.append(
                {
                    "origin": day.date,
                    "target": target.date,
                    "horizon": horizon,
                    "transactions": float(target.patterns.sum()),
                    "last": past[-1],
                    "weekly": past[-7 + (horizon - 1) % 7],
                }
            )
        past.append(float(day.patterns.sum()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/online_retail_ii/processed/raw.parquet"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/paper-readiness/retail"
    )
    parser.add_argument(
        "--private", type=Path, default=ROOT / "outputs/gaussian_population/checkpoints/retail"
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260915, 20260916, 20260917])
    parser.add_argument("--windows", nargs="+", default=list(WINDOWS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare_cache(args.data)
    core = ROOT / "backend/oransim/world_model/differentiable_market.py"
    workbook = args.data.parent.parent / "raw/online_retail_II.xlsx"
    protocol = {
        "dataset": "UCI Online Retail II",
        "windows": WINDOWS,
        "seeds": args.seeds,
        "variants": args.variants,
        "epochs": args.epochs,
        "patience": 20,
        "features": FEATURES,
        "heads": HEADS,
        "lag_ema_retention": 0.85,
        "loss": "mean daily mark NLL + .05 mean cell NB NLL + .15 two-step loss + .001 normalized regularization",
        "supervision": "daily cohort joint counts only; no event batches or individual dynamic histories",
        "marks": "basket value, positive merchandise units and unique SKUs greater than warmup medians",
        "groups": "warmup single invoice; repeat below/at repeat-spend median; repeat above median; unknown/missing customer",
        "population": "fixed warmup customer stocks; unknown stream uses one reference unit",
        "event_semantics": "positive merchandise invoice; not exposure, visit or purchase opportunity",
        "future_inputs": "known weekday; lag aggregate features and population anchors frozen at origin",
        "selection": "validation composite selects epoch and model before all test evaluation",
        "data_sha256": {args.data.name: digest(args.data)},
        "workbook_sha256": digest(workbook) if workbook.exists() else None,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch_threads": 1,
        },
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), core)},
    }
    path = args.out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("changed protocol requires a new output directory")
    write(path, protocol)
    frame, audit = invoices(args.data)
    write(args.out / "data-audit.json", audit)
    results = {}
    for window in args.windows:
        split, intercept, rates, info, warm_totals = prepare(frame, window)
        results[window] = {"data": info, "baselines": baselines(split, warm_totals), "runs": {}}
        private = args.private / window
        write(
            private / "records.json", {k: [d.to_dict() for d in v] for k, v in split.items()}, True
        )
        for seed in args.seeds:
            fitted, selection = {}, {}
            for variant in args.variants:
                print(f"{window} {seed} {variant}: fitting", flush=True)
                start = time.perf_counter()
                model = make_model(seed, variant, intercept, rates)
                state, fit = fit_market(
                    model,
                    split["train"],
                    split["validation"],
                    epochs=args.epochs,
                    multi_step_weight=0.15,
                    patience=20,
                )
                validation, val_state = score_days(model, split["validation"], state)
                elapsed = time.perf_counter() - start
                selection[variant] = {
                    "fit": fit,
                    "validation": summarize(validation),
                    "train_seconds": elapsed,
                    "parameters": model.parameter_counts(),
                }
                model.save(
                    private / f"{seed}-{variant}-train.json",
                    state,
                    {
                        "last_date": info["train_end"],
                        "training_task": "marked_positive_transactions",
                    },
                )
                fitted[variant] = (model, val_state)
                print(
                    f'{window} {seed} {variant}: validation NLL {summarize(validation)["joint_nll"]:.6f}, {elapsed:.1f}s',
                    flush=True,
                )
            selected = min(selection, key=lambda v: selection[v]["fit"]["validation_objective"])
            write(
                args.out / window / f"{seed}-selection.json",
                {"selected": selected, "candidates": selection},
            )
            for variant, (model, state) in fitted.items():
                rows, end = score_days(model, split["test"], state)
                results[window]["runs"][f"{seed}-{variant}"] = {
                    "test": summarize(rows),
                    "daily": rows,
                    "multi_step": forecasts(model, split, state),
                    "train_seconds": selection[variant]["train_seconds"],
                    "selected_epoch": selection[variant]["fit"]["selected_epoch"],
                    "parameters": model.parameter_counts(),
                }
                model.save(
                    private / f"{seed}-{variant}-final.json",
                    end,
                    {
                        "last_date": info["test_end"],
                        "training_task": "marked_positive_transactions",
                    },
                )
            write(args.out / "results.json", results)
        print(f"{window}: all experiments completed", flush=True)


if __name__ == "__main__":
    main()

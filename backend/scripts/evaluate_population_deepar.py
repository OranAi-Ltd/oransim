#!/usr/bin/env python3
"""DeepAR-NB and a joint-mark adaptation on frozen aggregate population records.

The shared LSTM consumes scaled previous counts and available covariates. NB2
parameters generate counts; the joint adaptation adds an autoregressive binary
mark head. Entire short training sequences are used as batches (no window
sampling). Validation chooses width and epoch before any test predictions.
Forecasts integrate sampled NB count histories; future aggregate covariates and
population are frozen at the origin, as in the Gaussian comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - standard PyTorch alias

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_population_count_baselines import SLOTS, extract_records, write_json

DTYPE = torch.float64
CONFIG = dict(
    seeds=[20260915, 20260916, 20260917],
    widths=[8, 16],
    lr=0.01,
    epochs={"kuairand": 300, "retail": 120},
    patience=20,
    weight_decay=0.001,
    clip=10.0,
    samples=2048,
    horizons=[1, 3, 7],
)


class DeepAR(nn.Module):
    def __init__(self, features, heads, cells, width, joint):
        super().__init__()
        self.width, self.joint = width, joint
        self.lstm = nn.LSTM(features + cells + 1, width).to(DTYPE)
        self.count = nn.Linear(width, 2).to(DTYPE)
        self.mark = nn.Linear(width, heads).to(DTYPE) if joint else None
        self.dependencies = nn.Parameter(
            torch.zeros(heads, heads, dtype=DTYPE), requires_grad=joint
        )
        self.register_buffer(
            "outcomes", torch.tensor(list(product([0.0, 1.0], repeat=heads)), dtype=DTYPE)
        )
        self.register_buffer("center", torch.zeros(features, dtype=DTYPE))
        self.register_buffer("spread", torch.ones(features, dtype=DTYPE))
        self.register_buffer("scale", torch.ones(cells, dtype=DTYPE))
        self.register_buffer("identity", torch.eye(cells, dtype=DTYPE))

    def setup(self, x, patterns):
        with torch.no_grad():
            self.center.copy_(x.mean((0, 1)))
            self.spread.copy_(x.std((0, 1), correction=0).clamp_min(1e-6))
            self.scale.copy_(1 + patterns.sum(-1).mean(0))
            if self.joint:
                counts = patterns.sum((0, 1))
                rates = (counts @ self.outcomes + 0.5) / (counts.sum() + 1)
                self.mark.bias.copy_(torch.logit(rates.clamp(0.001, 0.999)))

    def forward(self, x, previous, state=None):
        # x: time x (sample replicas * cells) x features.
        copies = x.shape[1] // len(self.scale)
        scale = self.scale.repeat(copies)
        ids = self.identity.repeat(copies, 1).expand(len(x), -1, -1)
        inputs = torch.cat(
            ((x - self.center) / self.spread, ids, (previous / scale)[..., None]), -1
        )
        hidden, state = self.lstm(inputs, state)
        raw = self.count(hidden)
        mu = (F.softplus(raw[..., 0]) + 1e-8) * scale
        alpha = (F.softplus(raw[..., 1]) + 1e-8) / scale.sqrt()
        distribution = torch.distributions.NegativeBinomial(
            total_count=1 / alpha, logits=(mu * alpha).log()
        )
        logq = None
        if self.joint:
            logits = (
                self.mark(hidden)[..., None, :]
                + self.outcomes @ torch.tril(self.dependencies, -1).T
            )
            logq = (self.outcomes * logits - F.softplus(logits)).sum(-1)
        return distribution, logq, state


def tensorize(records):
    """Canonical cell order; no event records, no target-dependent features."""
    result = {}
    canonical = None
    for split in ("train", "validation", "test"):
        xs, ys, dates = [], [], []
        for row in records[split]:
            if row.get("event_batch") is not None or row.get("event_outcomes") is not None:
                raise ValueError("aggregate records required")
            batch = row["batch"]
            pairs = list(zip(batch["campaign"], batch["group"], strict=False))
            order = sorted(range(len(pairs)), key=pairs.__getitem__)
            keys = [pairs[i] for i in order]
            if canonical is None:
                canonical = keys
            if keys != canonical or len(set(keys)) != len(keys):
                raise ValueError("each date must contain the same unique cells")
            # Reference population and effort are also known model inputs.
            x = np.c_[batch["features"], np.log1p(batch["population"]), np.log1p(batch["effort"])][
                order
            ]
            y = np.asarray(row["patterns"])[order]
            if np.any(~np.isfinite(x)) or np.any(y < 0) or np.any(y != np.floor(y)):
                raise ValueError("invalid features/counts")
            xs.append(x)
            ys.append(y)
            dates.append(row["date"])
        result[split] = dict(
            x=torch.tensor(np.asarray(xs), dtype=DTYPE),
            y=torch.tensor(np.asarray(ys), dtype=DTYPE),
            dates=dates,
        )
    extract_records(records)  # Also checks consecutive dates and count validity.
    return result


def objective(model, data, state=None, previous=None):
    x, patterns = data["x"], data["y"]
    counts = patterns.sum(-1)
    previous = torch.zeros_like(counts[:1]) if previous is None else previous[None]
    lag = torch.cat((previous, counts[:-1]))
    distribution, logq, state = model(x, lag, state)
    count_loss = -distribution.log_prob(counts).mean()
    loss = count_loss
    if model.joint:
        mark_loss = -(patterns * logq).sum((1, 2)) / counts.sum(1).clamp_min(1)
        loss = mark_loss.mean() + 0.05 * count_loss
    return loss, state, counts[-1]


def fit(data, joint, seed, epochs):
    candidates = []
    best_model, best_value = None, float("inf")
    for width in CONFIG["widths"]:
        torch.manual_seed(seed)
        train = data["train"]
        model = DeepAR(
            train["x"].shape[-1],
            int(np.log2(train["y"].shape[-1])),
            train["x"].shape[1],
            width,
            joint,
        )
        model.setup(train["x"], train["y"])
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
        best, selected, snapshot, stale = float("inf"), 0, None, 0
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss, _, _ = objective(model, train)
            parameters = [p for p in model.parameters() if p.requires_grad]
            regularizer = sum(p.square().sum() for p in parameters) / sum(
                p.numel() for p in parameters
            )
            (loss + CONFIG["weight_decay"] * regularizer).backward()
            nn.utils.clip_grad_norm_(parameters, CONFIG["clip"], error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                _, state, previous = objective(model, train)
                valid, _, _ = objective(model, data["validation"], state, previous)
                value = float(valid)
            if not np.isfinite(value):
                raise RuntimeError("nonfinite validation objective")
            if value < best - 1e-8:
                best, selected, snapshot, stale = value, epoch, deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= CONFIG["patience"]:
                break
        model.load_state_dict(snapshot)
        candidates.append(
            dict(width=width, selected_epoch=selected, epochs_run=epoch, validation=best)
        )
        if best < best_value:
            best_model, best_value = model, best
    return best_model, dict(
        candidates=candidates,
        width=best_model.width,
        validation=best_value,
        parameters=sum(p.numel() for p in best_model.parameters() if p.requires_grad),
    )


@torch.no_grad()
def forecast(model, x, previous, state, calendar_features, samples, seed):
    """No outcomes are accepted. Freeze every non-calendar covariate at origin."""
    x = x.clone()
    x[:, :, calendar_features:] = x[:1, :, calendar_features:]
    count, logq, next_state = model(x[:1], previous[None], state)
    outputs = [dict(mean=count.mean[0], logq=None if logq is None else logq[0])]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        prev = count.sample((samples,))[:, 0].reshape(1, -1)
        current = tuple(v.repeat(1, samples, 1) for v in next_state)
        for features in x[1:]:
            dist, q, current = model(features.repeat(samples, 1)[None], prev, current)
            mean = dist.mean.reshape(samples, -1).mean(0)
            # Conditional on a target arrival, histories are weighted by their
            # arrival intensities, rather than giving each sampled history equal weight.
            marginal_logq = None
            if q is not None:
                rates = dist.mean.reshape(samples, -1)
                probs = q.exp().reshape(samples, len(model.scale), -1)
                marginal_logq = ((rates[..., None] * probs).sum(0) / rates.sum(0)[:, None]).log()
            outputs.append(dict(mean=mean, logq=marginal_logq))
            prev = dist.sample()
    return outputs


@torch.no_grad()
def score(model, data, calendar_features, seed):
    _, state, previous = objective(model, data["train"])
    _, state, previous = objective(model, data["validation"], state, previous)
    test, rows = data["test"], []
    for index, origin in enumerate(test["dates"]):
        outputs = forecast(
            model,
            test["x"][index : index + 7],
            previous,
            state,
            calendar_features,
            CONFIG["samples"],
            seed + index,
        )
        for h in CONFIG["horizons"]:
            if h > len(outputs):
                continue
            out, target = outputs[h - 1], index + h - 1
            observed = test["y"][target].sum()
            row = dict(
                origin=origin,
                target=test["dates"][target],
                horizon=h,
                exposures=float(observed),
                predicted_exposures=float(out["mean"].sum()),
            )
            if out["logq"] is not None:
                row["joint_nll"] = float(
                    -(test["y"][target] * out["logq"]).sum() / observed.clamp_min(1)
                )
            rows.append(row)
        _, _, state = model(test["x"][index : index + 1], previous[None], state)
        previous = test["y"][index].sum(-1)
    summary = {}
    for h in CONFIG["horizons"]:
        chosen = [r for r in rows if r["horizon"] == h]
        if not chosen:
            continue
        errors = np.asarray([r["predicted_exposures"] - r["exposures"] for r in chosen])
        values = dict(
            origins=len(chosen),
            mae=float(np.abs(errors).mean()),
            rmse=float(np.sqrt(np.mean(errors**2))),
        )
        if model.joint:
            total = sum(r["exposures"] for r in chosen)
            values["joint_nll"] = sum(r["joint_nll"] * r["exposures"] for r in chosen) / max(
                1.0, total
            )
        summary[str(h)] = values
    return dict(summary=summary, date_level=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "models/gaussian_population/benchmarks"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/gaussian_population/external-baselines-2026-09-16",
    )
    parser.add_argument(
        "--snapshots", type=Path, default=ROOT / "outputs/gaussian_population/deepar-checkpoints"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = dict(
        config=CONFIG,
        reference="https://doi.org/10.1016/j.ijforecast.2019.07.001",
        status="Supplement specified after original Gaussian and GRU test results were known.",
        adaptation="Shared one-layer LSTM; full short sequences, uniform cell weighting; train-only scales; one-hot cell identifiers. DeepAR-joint adds autoregressive binary mark readout with daily mark NLL + .05 cell count NLL.",
        inputs="Original aggregate covariates plus reference population, effort, cell identity, and previous count. No individual histories or fitted Gaussian weights.",
        selection="Each method/seed selects width and epoch by validation; all 24 selections completed before test scoring; no test retuning.",
        forecast="2048 sampled NB count histories; conditional endpoint means; non-calendar covariates frozen at origin; marks weighted by endpoint arrival intensity.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={
            k: hashlib.sha256((args.input / v).read_bytes()).hexdigest() for k, v in SLOTS.items()
        },
        environment=dict(torch=torch.__version__, numpy=np.__version__, dtype="float64", threads=1),
    )
    path = args.out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("changed protocol: choose a new output directory")
    write_json(path, protocol)
    prepared = {k: tensorize(json.loads((args.input / v).read_text())) for k, v in SLOTS.items()}
    selections = {}
    # Train and freeze all models first, so test outcomes cannot influence selection.
    for key, data in prepared.items():
        for method, joint in [("deepar_nb", False), ("deepar_joint", True)]:
            for seed in CONFIG["seeds"]:
                slot = f"{key}/{method}/{seed}"
                start = time.perf_counter()
                model, selection = fit(data, joint, seed, CONFIG["epochs"][key.split("/")[0]])
                selection["train_seconds"] = time.perf_counter() - start
                snapshot = args.snapshots / f"{slot}.pt"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), snapshot)
                snapshot.chmod(0o600)
                selections[slot] = dict(
                    **selection, snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()
                )
                write_json(args.out / "selections.json", selections)
                print(slot, "selected", selection, flush=True)
    results = {}
    for key, data in prepared.items():
        results[key] = {}
        for method, joint in [("deepar_nb", False), ("deepar_joint", True)]:
            runs = {}
            for seed in CONFIG["seeds"]:
                slot = f"{key}/{method}/{seed}"
                train = data["train"]
                model = DeepAR(
                    train["x"].shape[-1],
                    int(np.log2(train["y"].shape[-1])),
                    train["x"].shape[1],
                    selections[slot]["width"],
                    joint,
                )
                model.load_state_dict(torch.load(args.snapshots / f"{slot}.pt", weights_only=True))
                runs[str(seed)] = score(model, data, 7 if key.startswith("retail") else 2, seed)
            summary = {}
            for h in CONFIG["horizons"]:
                metrics = ["mae", "rmse"] + (["joint_nll"] if joint else [])
                summary[str(h)] = dict(origins=runs[str(seed)]["summary"][str(h)]["origins"])
                for metric in metrics:
                    values = [r["summary"][str(h)][metric] for r in runs.values()]
                    summary[str(h)][metric] = dict(
                        mean=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1))
                    )
            results[key][method] = dict(runs=runs, summary=summary)
            write_json(args.out / "results.json", results)
            print(key, method, summary["1"], flush=True)
    write_json(
        args.out / "complete.json",
        dict(
            completed_utc=datetime.now(timezone.utc).isoformat(),
            fits=48,
            selected_models=24,
            tasks=list(results),
        ),
    )


if __name__ == "__main__":
    main()

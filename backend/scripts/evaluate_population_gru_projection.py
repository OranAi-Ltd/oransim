#!/usr/bin/env python3
"""Post-audit GRU readout supplement, separate from the frozen primary family.

All seven behavior heads receive a trainable four-dimensional hidden projection.
Initial probabilities, aggregate inputs, count likelihood, recurrent state,
training budget and seeds match the original GRU. Some original test results had
already been inspected when this structural correction was specified.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_population_readiness import comparator
from oransim.world_model.differentiable_market import MarketDay, fit_market, score_days
from oransim.world_model.population_comparisons import load_comparison
from oransim.world_model.population_gru_projection import (
    ProjectedGRUMarket,
    load_projected_gru,
    save_projected_gru,
)
from train_differentiable_market import multi_step, summarize, write


def build(seed, intercept, rates):
    legacy = comparator(seed, "gru", intercept, rates)
    model = ProjectedGRUMarket(legacy.config)
    loaded = model.load_state_dict(legacy.state_dict(), strict=False)
    assert loaded.missing_keys == ["behavior_projection"] and not loaded.unexpected_keys
    with torch.no_grad():
        model.behavior_projection.zero_()
        model.behavior_projection[0, 0] = 1.0
        model.behavior_projection[1:].copy_(legacy.behavior_loading)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "models/gaussian_population/benchmarks/kuairand"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/gaussian_population/paper-readiness/kuairand-gru-projection",
    )
    parser.add_argument(
        "--private",
        type=Path,
        default=ROOT / "outputs/gaussian_population/checkpoints/kuairand-gru-projection",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    seeds = [20260915, 20260916, 20260917]
    origin = ROOT / "experiments/gaussian_population/results/paper-readiness/kuairand/protocol.json"
    original_protocol = json.loads(origin.read_text())
    assert original_protocol["seeds"] == seeds and original_protocol["bucket"] == 1
    sources = [
        Path(__file__),
        ROOT / "backend/oransim/world_model/population_gru_projection.py",
        ROOT / "backend/oransim/world_model/population_comparisons.py",
        ROOT / "backend/oransim/world_model/differentiable_market.py",
        ROOT / "backend/scripts/evaluate_population_readiness.py",
        ROOT / "backend/scripts/train_differentiable_market.py",
    ]
    private_sources = {}
    prepared = {}
    for policy in ["standard", "random"]:
        folder = args.input / policy / "aggregate"
        record_path = folder / "records.json"
        # These two immutable buffers originate in warm-up, never fitted weights.
        warmup_source = folder / f"{seeds[0]}-joint_gaussian-train.json"
        base, _, _ = load_comparison(warmup_source)
        split = {
            k: [MarketDay.from_dict(d) for d in days]
            for k, days in json.loads(record_path.read_text()).items()
        }
        if any(
            d.event_batch is not None or d.event_outcomes is not None
            for days in split.values()
            for d in days
        ):
            raise ValueError("supplement requires pure aggregate records")
        prepared[policy] = (split, base.base_log_intensity.numpy(), base.reference_rates.numpy())
        for path in [record_path, warmup_source]:
            private_sources[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    protocol = {
        "status": "Post-audit enhanced baseline; specified after inspection of some primary-family test results.",
        "selection_scope": "Only validation selects each run epoch. No entry into the original family selection; no retuning after supplementary test scores.",
        "model": "ProjectedGRUMarket",
        "policies": ["standard", "random"],
        "regime": "aggregate",
        "bucket": 1,
        "seeds": seeds,
        "epochs": original_protocol["epochs"],
        "patience": original_protocol["patience"],
        "optimization": "Same fit_market: Adam .025, clip norm 10, normalized regularizer .001, count .05 and two-step .15.",
        "change": "Replace the first fixed behavior row and remaining latent behavior loading with an H x 4 trainable matrix; legacy loading frozen. All H rows learn.",
        "initialization": "Exactly legacy GRU output: first row e1, other rows copied. No data-dependent extra initialization.",
        "retained": "Same four-dimensional group GRU state, observation features, exposure likelihood and readout, autoregressive head dependencies, dates, warm-up priors and validation rule.",
        "information": "Only fixed historical groups, joint cohort counts, weekday, and lagged cohort count/mark EMA. Event records absent.",
        "warmup_priors": "Only immutable base_log_intensity and reference_rates buffers read from original train snapshots; no fitted parameter or recurrent state transferred.",
        "forecast": "Origin history columns and population frozen; 1/3/7-day endpoint scores.",
        "original_protocol_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "private_source_sha256": private_sources,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "dtype": "float64",
            "threads": 1,
        },
    }
    if (args.out / "protocol.json").exists() and json.loads(
        (args.out / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("changed supplementary protocol requires new output directory")
    # Record the fully specified supplement before fitting or scoring its tests.
    write(args.out / "protocol.json", protocol)
    results = {}
    snapshots = []
    for policy, (split, intercept, rates) in prepared.items():
        runs = {}
        results[policy] = {"runs": runs}
        for seed in seeds:
            model = build(seed, intercept, rates)
            print(f"{policy} {seed} projected_gru: fitting", flush=True)
            start = time.perf_counter()
            state, fit = fit_market(
                model,
                split["train"],
                split["validation"],
                epochs=protocol["epochs"],
                multi_step_weight=0.15,
                patience=protocol["patience"],
            )
            seconds = time.perf_counter() - start
            train_path = args.private / policy / f"{seed}-train.json"
            save_projected_gru(
                model, train_path, state, {"last_date": split["train"][-1].date, "supplement": True}
            )
            validation, origin_state = score_days(model, split["validation"], state)
            write(
                args.out / policy / f"{seed}-selection.json",
                {
                    "fit": fit,
                    "validation": summarize(validation),
                    "allocated_trainable": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "family_selection": "Separate supplementary baseline; only epoch selection.",
                },
            )
            rows, end = score_days(model, split["test"], origin_state)
            paths = multi_step(model, split, origin_state)
            final_path = args.private / policy / f"{seed}-final.json"
            save_projected_gru(
                model, final_path, end, {"last_date": split["test"][-1].date, "supplement": True}
            )
            for path, expected_state in [(train_path, state), (final_path, end)]:
                restored, restored_state, _ = load_projected_gru(path)
                with torch.no_grad():
                    found, _ = restored.rollout([split["test"][-1].batch] * 3, restored_state)
                    expected, _ = model.rollout([split["test"][-1].batch] * 3, expected_state)
                    for left, right in zip(found, expected, strict=False):
                        torch.testing.assert_close(left["actions"], right["actions"])
                        torch.testing.assert_close(left["exposures"], right["exposures"])
                if path.stat().st_mode & 0o777 != 0o600:
                    raise RuntimeError("private snapshot permissions")
                snapshots.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "three_step_restoration_passed": True,
                    }
                )
            runs[str(seed)] = {
                "test": summarize(rows),
                "daily": rows,
                "multi_step": paths,
                "train_seconds": seconds,
                "selected_epoch": fit["selected_epoch"],
                "allocated_trainable": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "learned_click_projection": model.behavior_projection[0].detach().tolist(),
            }
            print(
                f'{policy} {seed}: test NLL={runs[str(seed)]["test"]["joint_nll"]:.6f}', flush=True
            )
            write(args.out / "results.json", results)
    summary = {}
    for policy, record in results.items():
        runs = list(record["runs"].values())
        summary[policy] = {
            metric: {
                "mean": float(np.mean([r["test"][metric] for r in runs])),
                "seed_sd": float(np.std([r["test"][metric] for r in runs], ddof=1)),
            }
            for metric in ["joint_nll", "count_nll", "exposure_mae", "exposure_rmse"]
        }
        summary[policy]["multi_step"] = {}
        for horizon in [1, 3, 7]:
            values = []
            for r in runs:
                rows = [x for x in r["multi_step"] if x["horizon"] == horizon]
                values.append(
                    {
                        "cohort_joint_nll": float(
                            np.average(
                                [x["joint_nll"] for x in rows], weights=[x["events"] for x in rows]
                            )
                        ),
                        "exposure_mae": float(
                            np.mean([abs(x["predicted_exposures"] - x["exposures"]) for x in rows])
                        ),
                    }
                )
            summary[policy]["multi_step"][str(horizon)] = {
                "origins": len(rows),
                **{
                    metric: {
                        "mean": float(np.mean([v[metric] for v in values])),
                        "seed_sd": float(np.std([v[metric] for v in values], ddof=1)),
                    }
                    for metric in ["cohort_joint_nll", "exposure_mae"]
                },
            }
    write(
        args.out / "summary.json",
        {
            "scope": protocol["status"],
            "policies": summary,
            "runs": 6,
            "restored_snapshots": len(snapshots),
        },
    )
    write(
        args.out / "snapshot-checks.json", {"snapshots": snapshots, "all_restorations_passed": True}
    )
    print("All six supplementary GRU fits and twelve snapshot restorations complete.", flush=True)


if __name__ == "__main__":
    main()

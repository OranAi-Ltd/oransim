#!/usr/bin/env python3
"""Fit, observe, forecast and evaluate purchase choices with a saved market model.

Training input: {config, train: [MarketDay], validation: [MarketDay]}.
Observe input: [MarketDay]; forecast input: [MarketBatch].
Choice input: {attributes: products x features, population: group opportunities,
              prices: optional product prices, available: optional booleans}.
Choice-fit input: {config, train: [choice records with counts including outside], validation: [...]}.
Model state is private JSON, written atomically with mode 0600.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from oransim.world_model.differentiable_market import (
    DifferentiableMarket,
    MarketBatch,
    MarketConfig,
    MarketDay,
    MarketRuntime,
    fit_market,
)
from oransim.world_model.market_choice_learning import fit_choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=["fit", "observe", "forecast", "choice", "fit-choice", "simulate"]
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.out.resolve() in {
        args.model.resolve(),
        *([] if args.save_model is None else [args.save_model.resolve()]),
    }:
        raise ValueError("result output must differ from model snapshots")
    torch.set_num_threads(1)
    records = json.loads(args.records.read_text())
    if args.operation in ("fit", "fit-choice"):
        cfg = dict(records["config"])
        cfg["heads"] = tuple(cfg.get("heads", ("click", "long_view", "like")))
        model = DifferentiableMarket(MarketConfig(**cfg))
        if args.operation == "fit":
            days = [MarketDay.from_dict(d) for d in records["train"]]
            val = [MarketDay.from_dict(d) for d in records.get("validation", [])]
            state, result = fit_market(model, days, val, epochs=args.epochs, multi_step_weight=0.15)
            runtime = MarketRuntime(model, state, metadata={"last_date": days[-1].date})
        else:
            result = fit_choices(
                model, records["train"], records.get("validation", []), epochs=args.epochs
            )
            runtime = MarketRuntime(model, metadata={"training_task": "purchase_choice"})
        runtime.save(args.model)
    else:
        runtime = MarketRuntime.load(args.model)
        if args.operation == "observe":
            # A failed record never changes the on-disk snapshot; commit only after all records succeed.
            result = [runtime.observe(MarketDay.from_dict(d)) for d in records]
            runtime.save(args.save_model or args.model)
        elif args.operation == "forecast":
            result = runtime.forecast(
                [MarketBatch.create(**r) for r in records], args.trajectories, args.seed
            )
        elif args.operation == "simulate":
            branch = runtime.branch()
            result = []
            for i, r in enumerate(records):
                patterns = branch.simulate(MarketBatch.create(**r), args.seed + i)
                result.append({"day": branch.state.day, "pattern_counts": patterns.tolist()})
            if not args.save_model:
                raise ValueError("simulate requires --save-model for the isolated branch")
            branch.save(args.save_model)
        else:
            if runtime.metadata.get("training_task") != "purchase_choice":
                raise ValueError("choice prediction requires a snapshot trained by fit-choice")
            with torch.no_grad():
                result = {
                    k: v.tolist()
                    for k, v in runtime.model.choice(**records, state=runtime.state).items()
                }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

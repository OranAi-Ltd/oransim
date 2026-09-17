#!/usr/bin/env python3
"""Cross-generator aggregate choice recovery and recurrent gradient diagnostics.

Known Gaussian and discrete populations generate noisy aggregate market choices.
Training observes only outside/product counts over varying attributes and prices.
Test reports exact-probability KL on fresh contexts, separately from sampling error.
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from oransim.world_model.differentiable_market import (
    DTYPE,
    DifferentiableMarket,
    MarketConfig,
)
from oransim.world_model.market_choice_learning import fit_choices


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def records_for(teacher, seed, n):
    rng = np.random.default_rng(seed)
    records = []
    oracle = []
    for i in range(n):
        # Three products, two measured attributes, and one no-purchase option.
        attributes = rng.normal(size=(3, 2))
        prices = rng.uniform(0.1, 2.5, size=3)
        mass = [2400.0, 1600.0]
        record = {"attributes": attributes.tolist(), "population": mass, "prices": prices.tolist()}
        with torch.no_grad():
            probability = teacher.choice(**record)["demand"].numpy() / sum(mass)
        record["counts"] = rng.multinomial(int(sum(mass)), probability / probability.sum()).tolist()
        records.append(record)
        oracle.append(probability)
    return records, np.array(oracle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/differentiable-market-v1"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = MarketConfig(2, 1, 2, ("click", "like"), dimensions=2, quadrature=7, seed=20260912)
    sources = [
        Path(__file__),
        ROOT / "backend/oransim/world_model/differentiable_market.py",
        ROOT / "backend/oransim/world_model/market_choice_learning.py",
    ]
    protocol = {
        "seeds": [20260912, 20260913, 20260914],
        "train_markets": 20,
        "validation_markets": 8,
        "test_markets": 40,
        "opportunities_per_market": 4000,
        "teachers": ["gaussian", "discrete"],
        "candidates": ["fixed_gaussian", "joint_gaussian", "joint_discrete", "point"],
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "scope": "Known synthetic market generators; aggregate-only training; tests on fresh randomized attributes/prices. No real sales or causal effect claim.",
    }
    write(args.out / "recovery-protocol.json", protocol)
    results = {}
    for family in protocol["teachers"]:
        teacher = DifferentiableMarket(replace(config, family=family))
        with torch.no_grad():
            teacher.means.copy_(
                torch.tensor([[[-1.5, 0.6], [1.2, -0.4]], [[-0.8, -1.0], [1.8, 1.0]]], dtype=DTYPE)
            )
            teacher.mixture_logits.copy_(torch.tensor([[0.4, -0.4], [-0.3, 0.3]], dtype=DTYPE))
            teacher.choice_features.copy_(torch.tensor([0.5, -0.25], dtype=DTYPE))
            teacher.choice_interaction.copy_(torch.tensor([[1.1, 0.3], [-0.2, 0.9]], dtype=DTYPE))
            teacher.raw_price_sensitivity.copy_(torch.tensor([0.1, 0.2, -0.15], dtype=DTYPE))
        results[family] = {}
        for seed in protocol["seeds"]:
            train, _ = records_for(teacher, seed, 20)
            val, _ = records_for(teacher, seed + 100, 8)
            test, oracle = records_for(teacher, seed + 200, 40)
            for variant in protocol["candidates"]:
                cfg = replace(config, seed=seed)
                if variant == "fixed_gaussian":
                    cfg = replace(cfg, learn_population=False)
                if variant == "joint_discrete":
                    cfg = replace(cfg, family="discrete", components=4)
                if variant == "point":
                    cfg = replace(cfg, family="point", components=1, learn_population=False)
                model = DifferentiableMarket(cfg)
                start = time.perf_counter()
                report = fit_choices(model, train, val, epochs=180, patience=30)
                with torch.no_grad():
                    predicted = np.array(
                        [
                            model.choice(r["attributes"], r["population"], r["prices"])[
                                "demand"
                            ].numpy()
                            / 4000
                            for r in test
                        ]
                    )
                kl = float(
                    (oracle * (np.log(oracle) - np.log(np.maximum(predicted, 1e-15)))).sum(1).mean()
                )

                def high_order(original):
                    refined = DifferentiableMarket(replace(original.config, quadrature=25))
                    weights = original.state_dict()
                    weights["nodes"] = refined.nodes
                    weights["node_weights"] = refined.node_weights
                    refined.load_state_dict(weights)
                    return refined

                high = high_order(model)
                high_teacher = high_order(teacher)
                with torch.no_grad():
                    high_predicted = np.array(
                        [
                            high.choice(r["attributes"], r["population"], r["prices"])[
                                "demand"
                            ].numpy()
                            / 4000
                            for r in test
                        ]
                    )
                    high_oracle = np.array(
                        [
                            high_teacher.choice(r["attributes"], r["population"], r["prices"])[
                                "demand"
                            ].numpy()
                            / 4000
                            for r in test
                        ]
                    )
                high_kl = float(
                    (
                        high_oracle
                        * (np.log(high_oracle) - np.log(np.maximum(high_predicted, 1e-15)))
                    )
                    .sum(1)
                    .mean()
                )
                model.save(
                    ROOT
                    / "outputs/gaussian_population"
                    / args.out.name
                    / "choice"
                    / family
                    / f"{seed}-{variant}.json",
                    metadata={"training_task": "purchase_choice"},
                )

                results[family][f"{seed}-{variant}"] = {
                    "kl": kl,
                    "share_mae": float(np.abs(predicted - oracle).mean()),
                    "seconds": time.perf_counter() - start,
                    "selected_epoch": report["selected_epoch"],
                    "parameter_counts": model.parameter_counts(),
                    "kl_high_order": high_kl,
                    "quadrature_max_probability_change": float(
                        np.abs(high_predicted - predicted).max()
                    ),
                    "oracle_quadrature_max_probability_change": float(
                        np.abs(high_oracle - oracle).max()
                    ),
                    "oracle": oracle.tolist(),
                    "predicted": predicted.tolist(),
                }
                print(f"{family} {seed} {variant}: KL {kl:.6f}", flush=True)
                write(args.out / "choice-recovery.json", results)
    print("Aggregate choice recovery complete", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Known recurrent market recovery from noisy aggregate observations."""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/differentiable-market-v2"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = MarketConfig(2, 2, 2, ("click", "like"), quadrature=9, dispersion=80.0)
    teacher = DifferentiableMarket(config)
    with torch.no_grad():
        teacher.base_log_intensity.fill_(-0.7)
        teacher.behavior_bias.copy_(torch.tensor([[-2.0, -2.3], [-1.3, -1.8]], dtype=DTYPE))
        teacher.behavior_features.copy_(torch.tensor([[1.5, -0.2], [0.4, 0.6]], dtype=DTYPE))
        teacher.memory_coefficients.copy_(torch.tensor([[4.0, -0.2], [0.8, 0.5]], dtype=DTYPE))
        teacher.dependencies[1, 0] = 1.1
        teacher.feedback[0, 0] = 0.8
        teacher.raw_retention[0] = 1.3
        teacher.raw_retention[3] = 0.8
        teacher.intensity_features[0] = 0.2
    results = {}
    protocol = {
        "seeds": [20260912, 20260913, 20260914],
        "train_days": 20,
        "validation_days": 8,
        "test_days": 12,
        "teacher_config": vars(config),
        "teacher_parameters": {k: v.tolist() for k, v in teacher.state_dict().items()},
        "scope": "Synthetic known recurrent generator, fresh temporal innovations; expected-response KL evaluated against the true latent state; no real causal interpretation.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dynamics-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for seed in protocol["seeds"]:
        rng = np.random.default_rng(seed)
        truth = MarketRuntime(teacher).branch()
        days = []
        oracle = []
        for t in range(40):
            x = rng.normal(0, 0.9, (4, 2))
            batch = MarketBatch.create(
                [0, 1, 0, 1], [0, 0, 1, 1], x, [1000.0, 800.0, 1000.0, 800.0]
            )
            with torch.no_grad():
                oracle.append(teacher(batch, truth.state)["joint_log_prob"].exp())
            counts = truth.simulate(batch, seed + t)
            days.append(MarketDay(str(date(2020, 1, 1) + timedelta(days=t)), batch, counts))
        for variant in ["dynamic", "no_dynamics"]:
            model = DifferentiableMarket(replace(config, dynamic=variant == "dynamic", seed=seed))
            state, report = fit_market(
                model, days[:20], days[20:28], epochs=150, patience=25, multi_step_weight=0.15
            )
            _, state = score_days(model, days[20:28], state)
            kls = []
            errors = []
            with torch.no_grad():
                for t, day in enumerate(days[28:], 28):
                    out = model(day.batch, state)
                    p = oracle[t]
                    kls.append(
                        float(
                            (p * (p.clamp_min(1e-15).log() - out["joint_log_prob"])).sum(-1).mean()
                        )
                    )
                    errors.append(float((out["marginals"] - p @ model.outcomes).abs().mean()))
                    state = model.transition(state, day.batch, out, day.patterns)
            results[f"{seed}-{variant}"] = {
                "joint_kl": float(np.mean(kls)),
                "marginal_probability_mae": float(np.mean(errors)),
                "selected_epoch": report["selected_epoch"],
                "epochs": report["epochs"],
            }
            print(seed, variant, results[f"{seed}-{variant}"], flush=True)
            (args.out / "dynamics-recovery.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

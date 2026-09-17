#!/usr/bin/env python3
"""Cross-generator distribution recovery under known randomized content shifts.

Both Gaussian and discrete teachers are evaluated; a Gaussian teacher alone
would make a Gaussian advantage largely a consequence of the test design.
"""

import argparse
import hashlib
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oransim.world_model.population_behavior import BehaviorBatch, JointBehaviorModel
from train_population_panel import write


def quadrature(family, content):
    content = np.asarray(content, float)
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    weights /= np.sqrt(np.pi)
    if family == "gaussian_mixture":
        u = (np.array([-2.0, 2.0])[:, None] + np.sqrt(2) * nodes).ravel()
        w = np.tile(weights / 2, 2)
    elif family == "two_point":
        u = np.array([-np.sqrt(5.0), np.sqrt(5.0)])
        w = np.full(2, 0.5)
    elif family == "point_mass":
        u = np.array([0.0])
        w = np.ones(1)
    else:
        raise ValueError("unknown representation")
    return u[None, :] + content[:, None], np.tile(w, (len(content), 1))


def batch(family, content, outcomes):
    u, w = quadrature(family, content)
    return BehaviorBatch(np.empty((len(content), 0)), u, w, np.asarray(outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/gaussian_population/contribution-evaluation-v1"
    )
    args = parser.parse_args()
    seed = 20260908
    n = 8000
    result = {}
    teacher = JointBehaviorModel(["click", "like", "share"], [], [0.5, 0.2, 0.1])
    teacher.loadings = np.array([1.0, 1.5, -0.8])
    teacher.dependencies[1, 0] = 1.2
    teacher.dependencies[2, 1] = 0.9
    outcomes = np.array(list(product([0, 1], repeat=3)))
    grid = np.linspace(-3, 3, 61)
    test_content = np.repeat(grid, len(outcomes))
    test_y = np.tile(outcomes, (len(grid), 1))
    protocol = {
        "seed": seed,
        "train_events": n,
        "train_content_range": [-1, 1],
        "evaluation_content_grid": grid.tolist(),
        "teachers": ["gaussian_mixture", "two_point"],
        "candidates": ["gaussian_mixture", "two_point", "point_mass"],
        "scope": "Specified generators with known content logit shifts; deterministic probability recovery across a wider intervention range, not real platform treatment effects.",
        "teacher": teacher.to_dict(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "behavior_source_sha256": hashlib.sha256(
            (ROOT / "backend/oransim/world_model/population_behavior.py").read_bytes()
        ).hexdigest(),
    }
    write(args.out / "recovery-protocol.json", protocol)
    for teacher_family in protocol["teachers"]:
        rng = np.random.default_rng(seed)
        content = rng.uniform(-1, 1, n)
        latent = (
            rng.choice([-2.0, 2.0], n) + rng.normal(size=n)
            if teacher_family == "gaussian_mixture"
            else rng.choice([-np.sqrt(5.0), np.sqrt(5.0)], n)
        )
        y = np.array(
            [teacher.sample([], z + c, rng) for z, c in zip(latent, content, strict=False)]
        )
        oracle_batch = batch(teacher_family, test_content, test_y)
        logp = teacher.log_probability(oracle_batch)
        probability = np.exp(logp).reshape(len(grid), -1)
        if not np.allclose(probability.sum(1), 1):
            raise RuntimeError("teacher probability does not normalize")
        entropy = float(-(probability * logp.reshape(probability.shape)).sum(1).mean())
        oracle_marginals = probability @ outcomes
        fits = {}
        for family in protocol["candidates"]:
            model = JointBehaviorModel(teacher.heads, [], [0.5, 0.2, 0.1])
            fit = model.fit(batch(family, content, y), max_iterations=200)
            candidate = batch(family, test_content, test_y)
            predicted_logp = model.log_probability(candidate).reshape(probability.shape)
            cross_entropy = float(-(probability * predicted_logp).sum(1).mean())
            marginals = np.exp(predicted_logp) @ outcomes
            if cross_entropy < entropy - 1e-8:
                raise RuntimeError("negative KL indicates inconsistent probability evaluation")
            fits[family] = {
                "fit": fit,
                "excess_joint_nll_over_oracle": cross_entropy - entropy,
                "marginal_probability_mae": np.abs(marginals - oracle_marginals).mean(0).tolist(),
                "content_response_curves": marginals.tolist(),
            }
        result[teacher_family] = {
            "oracle_entropy": entropy,
            "oracle_curves": oracle_marginals.tolist(),
            "candidates": fits,
        }
    write(args.out / "representation-recovery.json", result)
    print("Cross-generator recovery evaluation complete")


if __name__ == "__main__":
    main()

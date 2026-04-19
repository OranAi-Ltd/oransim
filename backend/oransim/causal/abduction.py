"""Learned amortized abduction for counterfactual prediction.

Motivation
----------

Pearl's three-step counterfactual is:

    1. Abduction:    infer p(U | O) from observed outcome O
    2. Action:       apply do(T = t')
    3. Prediction:   resample descendants under the intervention

For OSS we've historically used two abduction paths in
``_amortized_abduct``:

  - **Sample-reuse** (default): assume the sampled ``u_noise`` from the
    simulator IS the counterfactual-world residual. This is Pearl's
    "fix U, change do(T), resample descendants" reading when no external
    realization is available; it's NOT a posterior over U given an
    observation.
  - **Bayesian shrink** (opt-in): a closed-form heuristic that pulls
    ``u`` toward a value consistent with an observed click rate. Works
    but is not a learned posterior.

This module adds the third option — a **learned amortizer** ``q(U | O)``.
Given (observed_click, click_prob) per agent, a small MLP predicts the
posterior mean of ``U``. The MLP is trained on samples drawn from the
simulator's own generative process, so the amortizer is aligned with
whatever forward model the agent layer uses.

We implement a minimal single-hidden-layer MLP with numpy + SGD so that
the OSS release has zero extra dependencies beyond what it already ships.
The ``sbi`` library's NPE / SNPE path (proper normalizing-flow posterior)
is Enterprise-Edition-only.

Reproducibility: ``fit_abduction_mlp`` is deterministic given a seed;
the trained weights round-trip stable across platforms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AbductionMLPWeights:
    """Compact weights for a 2-input → H-hidden → 1-output MLP.

    Layout kept minimal so the pkl / dict serialization is small and
    reviewable. Activation: ReLU on hidden, identity on output.
    """

    W1: np.ndarray  # [H, 2]
    b1: np.ndarray  # [H]
    W2: np.ndarray  # [1, H]
    b2: np.ndarray  # [1]
    feature_stats: dict  # { "click_prob_mean", "click_prob_std", "click_mean", ... }
    meta: dict  # { "n_samples", "epochs", "final_loss", ... }

    def apply(self, click_prob: np.ndarray, observed_click: np.ndarray) -> np.ndarray:
        """Forward pass → predicted u_shift per agent."""
        cp_n = (click_prob - self.feature_stats["click_prob_mean"]) / (
            self.feature_stats["click_prob_std"] + 1e-8
        )
        ck_n = (observed_click - self.feature_stats["click_mean"]) / (
            self.feature_stats["click_std"] + 1e-8
        )
        X = np.stack([cp_n, ck_n], axis=-1).astype(np.float32)  # [N, 2]
        h = np.maximum(0.0, X @ self.W1.T + self.b1)  # [N, H]
        y = h @ self.W2.T + self.b2  # [N, 1]
        return y.squeeze(-1)


def _simulate_training_batch(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (u, click_prob, click) triples from the same generative process
    the simulator uses internally:

        u ~ N(0, 1)
        logit(click_prob) ~ N(0, 1.5) (covers ~95% range [-3, 3])
        click ~ Bernoulli(sigmoid(logit(click_prob) + 0.7 * u))

    The 0.7 factor reflects how strongly u perturbs the click decision in
    the shipped StatisticalAgents (matches the constant in the existing
    Bayesian shrink formula).
    """
    u = rng.standard_normal(n).astype(np.float32)
    base_logit = rng.normal(0.0, 1.5, size=n).astype(np.float32)
    effective_logit = base_logit + 0.7 * u
    click_prob = 1.0 / (1.0 + np.exp(-base_logit))
    click = (rng.random(n) < 1.0 / (1.0 + np.exp(-effective_logit))).astype(np.float32)
    return u, click_prob, click


def fit_abduction_mlp(
    n_samples: int = 20_000,
    hidden: int = 16,
    epochs: int = 40,
    lr: float = 0.02,
    seed: int = 17,
) -> AbductionMLPWeights:
    """Train the amortizer on simulator rollouts. Deterministic given seed.

    Loss: mean-squared error between predicted u_shift and the true sampled u.
    Optimizer: plain SGD with momentum (kept minimal so it's pure numpy).

    Returns trained ``AbductionMLPWeights`` — serializable via dataclass.
    """
    rng = np.random.default_rng(seed)
    u, click_prob, click = _simulate_training_batch(n_samples, rng)
    # Feature normalization (from train set)
    stats = {
        "click_prob_mean": float(click_prob.mean()),
        "click_prob_std": float(click_prob.std()),
        "click_mean": float(click.mean()),
        "click_std": float(click.std() or 1.0),
    }
    cp_n = (click_prob - stats["click_prob_mean"]) / (stats["click_prob_std"] + 1e-8)
    ck_n = (click - stats["click_mean"]) / (stats["click_std"] + 1e-8)
    X = np.stack([cp_n, ck_n], axis=-1).astype(np.float32)
    y = u.reshape(-1, 1)

    W1 = rng.normal(0, 1.0 / np.sqrt(2), size=(hidden, 2)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), size=(1, hidden)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)

    mW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2)
    momentum = 0.9
    batch = 512
    n = X.shape[0]
    final_loss = 0.0

    for epoch in range(epochs):
        perm = rng.permutation(n)
        Xp, yp = X[perm], y[perm]
        losses = []
        for i in range(0, n, batch):
            xb = Xp[i : i + batch]
            yb = yp[i : i + batch]
            # forward
            z1 = xb @ W1.T + b1
            h = np.maximum(0.0, z1)
            yh = h @ W2.T + b2
            err = yh - yb
            loss = float((err**2).mean())
            losses.append(loss)
            # backward
            dL_dyh = (2.0 / xb.shape[0]) * err  # [B, 1]
            dL_dW2 = dL_dyh.T @ h  # [1, H]
            dL_db2 = dL_dyh.sum(axis=0)  # [1]
            dh = dL_dyh @ W2  # [B, H]
            dz1 = dh * (z1 > 0)  # ReLU grad
            dL_dW1 = dz1.T @ xb  # [H, 2]
            dL_db1 = dz1.sum(axis=0)  # [H]
            # SGD + momentum
            mW1 = momentum * mW1 - lr * dL_dW1
            W1 += mW1
            mb1 = momentum * mb1 - lr * dL_db1
            b1 += mb1
            mW2 = momentum * mW2 - lr * dL_dW2
            W2 += mW2
            mb2 = momentum * mb2 - lr * dL_db2
            b2 += mb2
        final_loss = float(np.mean(losses))

    return AbductionMLPWeights(
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        feature_stats=stats,
        meta={
            "n_samples": int(n),
            "hidden": int(hidden),
            "epochs": int(epochs),
            "lr": float(lr),
            "seed": int(seed),
            "final_loss": final_loss,
        },
    )


# Pre-trained weights cache — fit once per process on first use so downstream
# calls are cheap. Weights are small (~1 KB) so we don't bother pickling.
_CACHED_WEIGHTS: AbductionMLPWeights | None = None


def get_pretrained_abductor() -> AbductionMLPWeights:
    global _CACHED_WEIGHTS
    if _CACHED_WEIGHTS is None:
        _CACHED_WEIGHTS = fit_abduction_mlp()
    return _CACHED_WEIGHTS

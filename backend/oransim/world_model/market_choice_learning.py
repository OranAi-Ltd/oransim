"""Aggregate purchase-choice likelihood for the differentiable market model."""

from __future__ import annotations

from copy import deepcopy

import torch

from .differentiable_market import DTYPE


def choice_loss(model, record):
    result = model.choice(
        record["attributes"], record["population"], record.get("prices"), record.get("available")
    )
    counts = torch.as_tensor(record["counts"], dtype=DTYPE)
    if (
        counts.shape != result["demand"].shape
        or not torch.isfinite(counts).all()
        or torch.any(counts < 0)
    ):
        raise ValueError("choice counts include outside option followed by each product")
    total = counts.sum()
    mass = torch.as_tensor(record["population"], dtype=DTYPE).sum()
    if not torch.isclose(total, mass, atol=1e-6, rtol=1e-8):
        raise ValueError("observed choice counts must exhaust the declared opportunities")
    if torch.any((result["demand"] == 0) & (counts > 0)):
        raise ValueError("observations include unavailable products")
    probability = result["demand"] / mass.clamp_min(1)
    return -(counts * probability.clamp_min(1e-15).log()).sum() / total.clamp_min(1)


def fit_choices(model, train, validation=(), epochs=150, learning_rate=0.03, patience=25):
    if not train or epochs < 1:
        raise ValueError("choice training records and positive epochs required")
    names = {
        "mixture_logits",
        "means",
        "raw_cholesky",
        "choice_features",
        "choice_interaction",
        "raw_price_sensitivity",
    }
    parameters = [p for n, p in model.named_parameters() if n in names and p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    best = float("inf")
    best_weights = None
    best_epoch = 0
    trace = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.stack([choice_loss(model, row) for row in train]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite choice likelihood")
        # Weak parameter penalty, including variance anchors, controls degeneracy.
        penalty = sum(p.square().sum() for p in parameters) / max(
            1, sum(p.numel() for p in parameters)
        )
        objective = loss + 1e-4 * penalty
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite choice gradient")
        optimizer.step()
        with torch.no_grad():
            score = float(
                torch.stack([choice_loss(model, row) for row in validation or train]).mean()
            )
        trace.append(
            {"epoch": epoch + 1, "train_nll": float(loss.detach()), "validation_nll": score}
        )
        if score < best - 1e-8:
            best = score
            best_epoch = epoch + 1
            best_weights = deepcopy(model.state_dict())
        if epoch + 1 - best_epoch >= patience:
            break
    model.load_state_dict(best_weights)
    return {
        "epochs": len(trace),
        "selected_epoch": best_epoch,
        "validation_nll": best,
        "trace": trace,
    }

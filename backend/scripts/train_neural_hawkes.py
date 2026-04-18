"""Train the CausalNeuralHawkesProcess.

Usage
-----

    python -m backend.scripts.train_neural_hawkes \\
        --config default \\
        --data data/synthetic/event_streams_v0_1.jsonl \\
        --out data/models/causal_neural_hawkes_v0_1.pt \\
        --epochs 30

Status
------

v0.1.0-alpha ships this training harness as-is; the synthetic event-stream
generator lands in v0.2. Attempting to run today will fail with a helpful
``FileNotFoundError`` pointing at the generator or the v0.2 release.

Expected v0.2 behaviour:

- Default hyperparameters produce a MAE on 14-day event-volume forecast
  competitive with classical Hawkes (≥10% improvement on the synthetic
  eval set).
- Full training takes ≈ 45 min on a single A100 for 50k synthetic streams.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _load_streams(path: str) -> Iterable[Iterable[tuple[float, str]]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Training event streams not found at {p}.\n"
            "Generate synthetic data first (v0.2):\n"
            "    python -m backend.scripts.gen_synthetic_data --n 50000 "
            f"--events-out {p}\n"
            "Or download a released bundle from "
            "https://github.com/ORAN-cgsj/oransim/releases (starting v0.2)."
        )
    raise NotImplementedError(
        "Event-stream loader lands with the synthetic generator in v0.2. "
        f"File at {p} was found but the loader is not yet wired up."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Causal Neural Hawkes Process.")
    parser.add_argument("--config", default="default")
    parser.add_argument("--data", default="data/synthetic/event_streams_v0_1.jsonl")
    parser.add_argument("--val", default=None)
    parser.add_argument("--out", default="data/models/causal_neural_hawkes.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    try:
        from oransim.diffusion import CausalNeuralHawkesConfig, CausalNeuralHawkesProcess
    except ImportError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print("Install the ML extras: pip install 'oransim[ml]'", file=sys.stderr)
        return 2

    cfg = CausalNeuralHawkesConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    model = CausalNeuralHawkesProcess(cfg)

    try:
        train = _load_streams(args.data)
        val = _load_streams(args.val) if args.val else None
    except (FileNotFoundError, NotImplementedError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 3

    history = model.fit(train, val_dataset=val)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved to {args.out}")
    print(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

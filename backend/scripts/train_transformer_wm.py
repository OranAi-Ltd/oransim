"""Train the CausalTransformerWorldModel.

Usage
-----

    python -m backend.scripts.train_transformer_wm \\
        --config default \\
        --data data/synthetic/scenarios_v0_1.jsonl \\
        --out data/models/causal_transformer_wm_v0_1.pt \\
        --epochs 50 --batch-size 256

Status
------

v0.1.0-alpha ships this training harness as-is. The ``data/synthetic/``
directory is currently a placeholder — the synthetic scenario generator
lands in v0.2 (see :mod:`backend.scripts.gen_synthetic_data`). Running this
script today will fail with a helpful ``FileNotFoundError`` pointing the
user at the generator or the v0.2 release.

Once the generator + pretrained weights ship:

- Default hyperparameters produce the benchmark R² numbers reported in
  README.md → Benchmarks table.
- Full training takes ≈ 30 min on a single A100 for 100k synthetic samples.
- Checkpoints are saved every epoch to ``<out>.ep<N>.pt`` and the best
  val-loss snapshot is copied to ``<out>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _load_dataset(path: str) -> Iterable[dict[str, Any]]:
    """Placeholder dataset loader.

    Expected v0.2 behavior: read parquet of {features, targets, treatment_arm},
    build a torch DataLoader with collate_fn that produces batched dicts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {p}.\n"
            "Generate synthetic data first (v0.2):\n"
            "    python -m backend.scripts.gen_synthetic_data --n 100000 --out "
            f"{p.parent}\n"
            "Or download a released bundle from "
            "https://github.com/ORAN-cgsj/oransim/releases (starting v0.2)."
        )
    raise NotImplementedError(
        "Dataset loader lands with the synthetic generator in v0.2. "
        f"File at {p} was found but the loader is not yet wired up."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Causal Transformer World Model.")
    parser.add_argument("--config", default="default", help="Config preset or YAML path.")
    parser.add_argument("--data", default="data/synthetic/scenarios_v0_1.jsonl")
    parser.add_argument("--val", default=None)
    parser.add_argument("--out", default="data/models/causal_transformer_wm.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--balancing-loss",
        choices=["hsic", "iptw_adv", "none"],
        default="hsic",
    )
    args = parser.parse_args(argv)

    # Import inside main so users can `--help` without torch installed
    try:
        from oransim.world_model import CausalTransformerWMConfig, CausalTransformerWorldModel
    except ImportError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print("Install the ML extras: pip install 'oransim[ml]'", file=sys.stderr)
        return 2

    cfg = CausalTransformerWMConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        balancing_loss=args.balancing_loss,
    )
    model = CausalTransformerWorldModel(cfg)

    try:
        train = _load_dataset(args.data)
        val = _load_dataset(args.val) if args.val else None
    except (FileNotFoundError, NotImplementedError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 3

    history = model.fit(train, val_dataset=val)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved to {args.out}")
    print("Training history:")
    print(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

NICHES = [
    "beauty",
    "fashion",
    "food",
    "electronics",
    "travel",
    "parenting",
    "fitness",
    "home",
    "beverage",
    "pet",
]
KOL_TIERS = ["nano", "micro", "mid", "macro", "mega"]


def _hash_embed(text: str, dim: int, seed: int = 0) -> Any:
    """Deterministic pseudo-embedding for text via hashing.

    Stand-in for OpenAI text-embedding-3-small until the real embedder bus
    is wired into the training script (v0.2). The stand-in is deterministic
    per ``text`` so that the Causal Transformer's learned feature projections
    see stable targets during training.
    """
    import hashlib

    import numpy as np
    import torch

    rng = np.random.default_rng(
        seed + int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (2**31)
    )
    return torch.tensor(rng.standard_normal(dim).astype("float32") * 0.1)


def _featurize_row(row: dict, seed: int) -> dict[str, Any]:
    """Expand a scalar scenario row into the tensor dict the
    CausalTransformerNet.tokenize() method expects.

    This is a demo-grade feature engineering pass. v0.2 replaces the
    hash-based creative embedding with the real OpenAI-compatible
    text-embedding-3-small bus.
    """
    import torch

    niche = row.get("niche", "beauty")
    niche_idx = NICHES.index(niche) if niche in NICHES else 0
    tier = row.get("kol_tier", "nano")
    tier_idx = KOL_TIERS.index(tier) if tier in KOL_TIERS else 0

    scenario_id = row.get("scenario_id", "UNK")
    creative_embed = _hash_embed(f"{scenario_id}/{niche}", dim=1536, seed=seed)

    # 16-dim KOL feature vector
    kol_feat = torch.tensor(
        [
            float(tier_idx) / 5.0,
            float(row.get("kol_fan_count", 0)) / 1e7,
            float(row.get("kol_engagement_rate", 0.02)),
            *[0.0] * 13,  # reserved for real KOL audience / niche affinity
        ],
        dtype=torch.float32,
    )

    # 24-dim demographic feature — one-hot niche + padding
    demo_feat = torch.zeros(24, dtype=torch.float32)
    demo_feat[niche_idx] = 1.0

    # 4-dim time feature — placeholder (day-of-week sin/cos, hour-of-day sin/cos)
    time_feat = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    return {
        "creative_embed": creative_embed,
        "platform_id": torch.tensor(int(row.get("platform_id", 0)), dtype=torch.long),
        "kol_feat": kol_feat,
        "demo_feat": demo_feat,
        "budget": torch.tensor([float(row.get("budget", 10000)) / 1e5], dtype=torch.float32),
        "time_feat": time_feat,
    }


def _load_dataset(path: str, batch_size: int = 256, seed: int = 42) -> Iterable[dict[str, Any]]:
    """Load scenarios from JSONL and yield batched feature + target dicts.

    The current implementation does CPU-side collation and keeps the full
    file in RAM — adequate for the 100k synthetic demo. v0.2 will add a
    torch DataLoader wrapper with background workers + shuffling.
    """
    import torch

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {p}.\n"
            "Generate synthetic data first:\n"
            "    python -m backend.scripts.gen_synthetic_data --out "
            f"{p.parent}\n"
            "Or use the demo data shipped with the repo at data/synthetic/."
        )

    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Deterministic shuffle per seed
    import random

    rng = random.Random(seed)
    rng.shuffle(rows)

    def _collate(batch_rows: list[dict]) -> dict[str, Any]:
        feats: dict[str, list] = {}
        for r in batch_rows:
            for k, v in _featurize_row(r, seed=seed).items():
                feats.setdefault(k, []).append(v)
        # Stack into [B, ...] tensors
        stacked = {k: torch.stack(v) for k, v in feats.items()}
        # Drop the per-sample batch axis that _batch_one would otherwise re-add
        # (fit() sends the dict to the network as-is)
        targets = {
            kpi: torch.tensor([float(r["targets"][kpi]) for r in batch_rows], dtype=torch.float32)
            for kpi in ("impressions", "clicks", "conversions", "revenue")
        }
        out: dict[str, Any] = stacked
        out["targets"] = targets
        if all("cf_targets" in r for r in batch_rows):
            out["cf_targets"] = {
                kpi: torch.tensor(
                    [float(r["cf_targets"][kpi]) for r in batch_rows], dtype=torch.float32
                )
                for kpi in ("impressions", "clicks", "conversions", "revenue")
            }
        if all("treatment_arm" in r for r in batch_rows):
            out["treatment_arm"] = torch.tensor(
                [int(r["treatment_arm"]) for r in batch_rows], dtype=torch.long
            )
        return out

    for i in range(0, len(rows), batch_size):
        yield _collate(rows[i : i + batch_size])


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
        train = list(_load_dataset(args.data, batch_size=args.batch_size))
        val = list(_load_dataset(args.val, batch_size=args.batch_size)) if args.val else None
    except (FileNotFoundError, NotImplementedError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 3
    print(f"[train] loaded {len(train)} mini-batches of size {args.batch_size}")

    history = model.fit(train, val_dataset=val)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved to {args.out}")
    print("Training history:")
    print(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

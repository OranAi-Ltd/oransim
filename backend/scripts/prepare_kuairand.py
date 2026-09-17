"""Prepare KuaiRand raw CSV files into an OranBench C1 parquet table.

The script is intentionally schema-tolerant because KuaiRand Pure/1K/27K ship
slightly different side files. It only processes CSVs that contain both a user
identifier and an item/video identifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oransim.paths import KUAIRAND_PROCESSED, KUAIRAND_RAW

USER_COLS = ["user_id", "userId", "uid", "user", "userID"]
ITEM_COLS = ["item_id", "video_id", "videoId", "photo_id", "vid", "item", "itemID"]
TIME_COLS = ["timestamp", "time", "time_ms", "ts", "date", "datetime"]
PROPENSITY_COLS = ["propensity", "pscore", "prob", "logging_prob", "policy_prob", "action_prob"]
REWARD_ALIASES = {
    "reward_click": ["click", "is_click", "clicked"],
    "reward_like": ["like", "is_like", "liked"],
    "reward_long_view": ["long_view", "is_long_view", "longview"],
    "reward_watch_time": [
        "watch_time",
        "watch_time_ms",
        "play_time",
        "play_time_ms",
        "play_duration",
    ],
}
KUAIRAND_PURE_RANDOM_POOL_SIZE = 7_583
KUAIRAND_PURE_UNIFORM_PROPENSITY = 1.0 / KUAIRAND_PURE_RANDOM_POOL_SIZE


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def score_csv(path: Path) -> tuple[int, dict[str, str | None]]:
    try:
        sample = pd.read_csv(path, nrows=200)
    except Exception:
        return 0, {}
    cols = list(sample.columns)
    mapping = {
        "user_id": first_present(cols, USER_COLS),
        "item_id": first_present(cols, ITEM_COLS),
        "timestamp": first_present(cols, TIME_COLS),
        "propensity": first_present(cols, PROPENSITY_COLS),
    }
    for out_col, aliases in REWARD_ALIASES.items():
        mapping[out_col] = first_present(cols, aliases)
    score = int(mapping["user_id"] is not None) + int(mapping["item_id"] is not None)
    score += sum(1 for out_col in REWARD_ALIASES if mapping[out_col] is not None)
    return score, mapping


def candidate_csvs(
    raw_dir: Path, include_standard: bool = False
) -> list[tuple[Path, dict[str, str | None]]]:
    found = []
    for path in sorted(raw_dir.rglob("*.csv")):
        if not include_standard and "random" not in path.name.lower():
            continue
        score, mapping = score_csv(path)
        if mapping.get("user_id") and mapping.get("item_id") and score >= 3:
            found.append((path, mapping))
    return found


def normalize_chunk(
    chunk: pd.DataFrame,
    mapping: dict[str, str | None],
    source: Path,
    default_propensity: float | None,
) -> pd.DataFrame:
    out = pd.DataFrame()
    out["user_id"] = chunk[mapping["user_id"]].astype(str)
    out["item_id"] = chunk[mapping["item_id"]].astype(str)
    out["action_id"] = out["item_id"]
    if mapping.get("timestamp"):
        out["timestamp"] = chunk[mapping["timestamp"]]
    else:
        out["timestamp"] = pd.NA
    if mapping.get("propensity"):
        out["propensity"] = pd.to_numeric(chunk[mapping["propensity"]], errors="coerce")
        if default_propensity is not None:
            out["propensity"] = out["propensity"].fillna(default_propensity)
    elif default_propensity is not None:
        out["propensity"] = default_propensity
    else:
        out["propensity"] = pd.NA
    for out_col in REWARD_ALIASES:
        src = mapping.get(out_col)
        if src:
            out[out_col] = pd.to_numeric(chunk[src], errors="coerce")
        else:
            out[out_col] = pd.NA
    out["source_file"] = source.name
    return out


def find_meta_csv(raw_dir: Path, prefix: str) -> Path | None:
    """Recursively locate the first CSV whose basename starts with ``prefix``."""
    for path in raw_dir.rglob(f"{prefix}*.csv"):
        return path
    return None


def build_user_features(raw_dir: Path, out_path: Path) -> dict | None:
    csv = find_meta_csv(raw_dir, "user_features")
    if not csv:
        return None
    df = pd.read_csv(csv)
    df["user_id"] = df["user_id"].astype(str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    info = {
        "source": str(csv),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "output": str(out_path),
    }
    print(
        f"[kuairand-prepare] user_features: {info['rows']} rows × {info['cols']} cols → {out_path}"
    )
    return info


def build_video_features(raw_dir: Path, out_path: Path) -> dict | None:
    """Basic video features only (basic_*.csv). Statistic CSV is intentionally
    skipped — its columns (``play_duration``, ``complete_play_cnt``, etc.) are
    cumulative target leakage for the watch-time prediction task."""
    csv = find_meta_csv(raw_dir, "video_features_basic")
    if not csv:
        return None
    df = pd.read_csv(csv)
    df["video_id"] = df["video_id"].astype(str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    info = {
        "source": str(csv),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "output": str(out_path),
    }
    print(
        f"[kuairand-prepare] video_features_basic: {info['rows']} rows × {info['cols']} cols → {out_path}"
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=KUAIRAND_RAW)
    parser.add_argument("--out", type=Path, default=KUAIRAND_PROCESSED / "clicks.parquet")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test row cap.")
    parser.add_argument(
        "--include-standard",
        action="store_true",
        help="Include non-random standard logs. Default is C1 random-exposure slice only.",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Also export user_features.parquet + video_features.parquet "
        "(basic only; statistic skipped to avoid watch-time leakage). v0.4+.",
    )
    parser.add_argument(
        "--uniform-propensity",
        type=float,
        default=KUAIRAND_PURE_UNIFORM_PROPENSITY,
        help="Default propensity for KuaiRand Pure random-exposed rows. Use 0 to leave missing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cands = candidate_csvs(args.raw_dir, include_standard=args.include_standard)
    if not cands:
        raise SystemExit(f"No KuaiRand random interaction CSVs found under {args.raw_dir}")
    print("[kuairand-prepare] candidate CSVs:")
    for path, mapping in cands:
        print(f"  - {path} :: {mapping}")

    if args.dry_run:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    propensity_present = False
    default_propensity = args.uniform_propensity if args.uniform_propensity > 0 else None
    user_set: set[str] = set()
    item_set: set[str] = set()
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    try:
        for path, mapping in cands:
            usecols = sorted({v for v in mapping.values() if v})
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunksize):
                norm = normalize_chunk(chunk, mapping, path, default_propensity=default_propensity)
                propensity_present = propensity_present or norm["propensity"].notna().any()
                if args.max_rows and total + len(norm) > args.max_rows:
                    norm = norm.iloc[: max(args.max_rows - total, 0)]
                if norm.empty:
                    continue
                user_set.update(norm["user_id"].unique().tolist())
                item_set.update(norm["item_id"].unique().tolist())
                table = pa.Table.from_pandas(norm, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(args.out, schema, compression="snappy")
                else:
                    table = table.select(schema.names).cast(schema, safe=False)
                writer.write_table(table)
                total += len(norm)
                if args.max_rows and total >= args.max_rows:
                    break
            if args.max_rows and total >= args.max_rows:
                break
    finally:
        if writer is not None:
            writer.close()

    if total == 0:
        raise SystemExit("No rows produced from candidate KuaiRand CSVs")
    manifest = {
        "rows": int(total),
        "n_users": len(user_set),
        "n_items": len(item_set),
        "propensity_status": "present" if propensity_present else "missing_or_unmapped",
        "propensity_source": (
            f"uniform_random_pool_1/{KUAIRAND_PURE_RANDOM_POOL_SIZE}"
            if default_propensity == KUAIRAND_PURE_UNIFORM_PROPENSITY
            else ("custom_uniform" if default_propensity is not None else "missing_or_unmapped")
        ),
        "uniform_propensity": default_propensity,
        "output": str(args.out),
        "sources": [str(p) for p, _ in cands],
        "include_standard": bool(args.include_standard),
        "include_meta": bool(args.include_meta),
    }
    if args.include_meta:
        user_info = build_user_features(args.raw_dir, args.out.parent / "user_features.parquet")
        video_info = build_video_features(args.raw_dir, args.out.parent / "video_features.parquet")
        manifest["user_features"] = user_info
        manifest["video_features"] = video_info
    (args.out.parent / "kuairand_prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

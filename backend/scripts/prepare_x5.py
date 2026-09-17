"""Prepare X5 RetailHero data into an OranBench C2 uplift parquet table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oransim.paths import DATA_DIR

X5_DIR = DATA_DIR / "x5"
X5_RAW = X5_DIR / "raw"
X5_PROCESSED = X5_DIR / "processed"


def find_raw(raw_dir: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = raw_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing {stem}.csv(.gz) under {raw_dir}")


def id_col(df: pd.DataFrame) -> str:
    for col in ("client_id", "customer_id", "user_id"):
        if col in df.columns:
            return col
    raise KeyError(f"no client/customer id column in {list(df.columns)}")


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def aggregate_purchases(path: Path, chunksize: int, max_rows: int = 0) -> pd.DataFrame:
    aggs = []
    rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        cid = id_col(chunk)
        chunk = chunk.rename(columns={cid: "client_id"})
        rows += len(chunk)

        amount_col = first_present(
            list(chunk.columns),
            [
                "purchase_sum",
                "transaction_sum",
                "amount",
                "sales_sum",
                "trn_sum_from_iss",
                "regular_points_received",
            ],
        )
        date_col = first_present(
            list(chunk.columns),
            ["transaction_datetime", "purchase_datetime", "date", "datetime", "transaction_date"],
        )
        product_col = first_present(
            list(chunk.columns), ["product_id", "item_id", "material", "sku"]
        )
        store_col = first_present(list(chunk.columns), ["store_id", "shop_id"])

        grouped = (
            chunk.groupby("client_id", sort=False).size().rename("X_purchase_count").to_frame()
        )
        if amount_col:
            grouped["X_total_spend"] = (
                pd.to_numeric(chunk[amount_col], errors="coerce").groupby(chunk["client_id"]).sum()
            )
        if product_col:
            grouped["X_unique_products"] = chunk.groupby("client_id")[product_col].nunique()
        if store_col:
            grouped["X_unique_stores"] = chunk.groupby("client_id")[store_col].nunique()
        if date_col:
            dates = pd.to_datetime(chunk[date_col], errors="coerce")
            grouped["first_purchase_ts"] = dates.groupby(chunk["client_id"]).min()
            grouped["last_purchase_ts"] = dates.groupby(chunk["client_id"]).max()
        aggs.append(grouped.reset_index())
        if max_rows and rows >= max_rows:
            break

    if not aggs:
        return pd.DataFrame(columns=["client_id", "X_purchase_count"])

    agg = pd.concat(aggs, ignore_index=True)
    sum_cols = [
        c
        for c in agg.columns
        if c.startswith("X_") and c not in {"X_unique_products", "X_unique_stores"}
    ]
    nunique_cols = [c for c in ("X_unique_products", "X_unique_stores") if c in agg.columns]
    spec = {c: "sum" for c in sum_cols + nunique_cols}
    if "first_purchase_ts" in agg.columns:
        spec["first_purchase_ts"] = "min"
    if "last_purchase_ts" in agg.columns:
        spec["last_purchase_ts"] = "max"
    out = agg.groupby("client_id", as_index=False).agg(spec)
    if "X_total_spend" in out.columns and "X_purchase_count" in out.columns:
        out["X_avg_spend"] = out["X_total_spend"] / out["X_purchase_count"].clip(lower=1)
    if "last_purchase_ts" in out.columns:
        max_ts = pd.to_datetime(out["last_purchase_ts"], errors="coerce").max()
        out["X_recency_days"] = (
            max_ts - pd.to_datetime(out["last_purchase_ts"], errors="coerce")
        ).dt.days
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=X5_RAW)
    parser.add_argument("--out", type=Path, default=X5_PROCESSED / "uplift.parquet")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-purchase-rows", type=int, default=0, help="Optional smoke-test cap.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_path = find_raw(args.raw_dir, "uplift_train")
    clients_path = find_raw(args.raw_dir, "clients")
    purchases_path = find_raw(args.raw_dir, "purchases")

    print(f"[x5-prepare] train={train_path}")
    print(f"[x5-prepare] clients={clients_path}")
    print(f"[x5-prepare] purchases={purchases_path}")
    if args.dry_run:
        return 0

    train = pd.read_csv(train_path)
    clients = pd.read_csv(clients_path)
    train = train.rename(
        columns={id_col(train): "client_id", "treatment_flg": "treatment", "target": "outcome"}
    )
    clients = clients.rename(columns={id_col(clients): "client_id"})
    if "treatment" not in train.columns or "outcome" not in train.columns:
        raise SystemExit(
            "uplift_train must contain treatment_flg/treatment and target/outcome columns"
        )

    purchases = aggregate_purchases(
        purchases_path, chunksize=args.chunksize, max_rows=args.max_purchase_rows
    )
    df = train.merge(clients, on="client_id", how="left").merge(
        purchases, on="client_id", how="left"
    )
    for col in df.columns:
        if col.startswith("X_"):
            df[col] = df[col].fillna(0)
    feature_cols = [c for c in df.columns if c not in {"client_id", "treatment", "outcome"}]
    renamed = {}
    for col in feature_cols:
        if not col.startswith("X_"):
            renamed[col] = f"X_{col}"
    df = df.rename(columns=renamed)
    df["propensity_estimate"] = float(pd.to_numeric(df["treatment"], errors="coerce").mean())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    manifest = {
        "rows": int(len(df)),
        "treatment_rate": float(pd.to_numeric(df["treatment"], errors="coerce").mean()),
        "outcome_rate": float(pd.to_numeric(df["outcome"], errors="coerce").mean()),
        "feature_columns": [c for c in df.columns if c.startswith("X_")],
        "output": str(args.out),
        "purchase_rows_capped": int(args.max_purchase_rows),
    }
    (args.out.parent / "x5_prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

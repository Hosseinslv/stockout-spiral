"""Stage 1. When and where do stockouts happen?

Descriptives are written in SQL (sql/*.sql, run with DuckDB on the parquet
file). The regression asks which conditions make a stockout more likely,
comparing each store and each product with itself (fixed effects).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def run_sql(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Run sql/<name>.sql against the table `sales`."""
    con = duckdb.connect()
    con.register("sales", df[[c for c in df.columns if c != "true_hours_demand"]])
    return con.execute((SQL_DIR / f"{name}.sql").read_text()).df()


def stockout_regression(df: pd.DataFrame):
    """Share of open hours out of stock, explained by discount, rain, holiday.

    Store, product and weekday fixed effects; standard errors clustered by store.
    `discount` is the price ratio (1.0 = full price), so depth = 1 - discount.
    """
    d = pd.DataFrame({
        "stockout_share": df["stock_hour6_22_cnt"] / 16,
        "discount_depth": 1 - df["discount"],
        "rain": df["precpt"],
        "holiday": df["holiday_flag"],
    })
    absorb = pd.DataFrame({
        "store": df["store_id"], "product": df["product_id"],
        "weekday": pd.to_datetime(df["dt"]).dt.dayofweek,
    }).astype("category")
    model = AbsorbingLS(d["stockout_share"], d[["discount_depth", "rain", "holiday"]], absorb=absorb,
                        drop_absorbed=True)
    return model.fit(cov_type="clustered", clusters=df["store_id"].to_numpy())

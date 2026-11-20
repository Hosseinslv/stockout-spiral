"""Data loading and a small simulator with the same layout as FreshRetailNet-50K.

One row = one store-product-day. The two list columns hold 24 hourly values:
    hours_sale          sales in each hour of the day
    hours_stock_status  1 if the product was out of stock in that hour, else 0
    stock_hour6_22_cnt  number of out-of-stock hours between 06:00 and 22:00

The simulator exists for two reasons:
  1. the code can be tested without downloading the real data;
  2. it produces the *true* demand, which the real data never shows,
     so we can check that every method does what we claim it does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OPEN_HOURS = np.arange(6, 22)  # the dataset reports stock status for 06:00-22:00

COLUMNS = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "dt",
    "sale_amount", "hours_sale", "stock_hour6_22_cnt", "hours_stock_status",
    "discount", "holiday_flag", "activity_flag", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
]


def load_real(data_dir: str | Path = "data/raw", split: str = "train") -> pd.DataFrame:
    """Load the real dataset after `python scripts/download_data.py`."""
    path = Path(data_dir) / "data" / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_data.py first.")
    df = pd.read_parquet(path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Columns differ from what the code expects. Missing: {sorted(missing)}")
    df["dt"] = pd.to_datetime(df["dt"])
    # stock_hour6_22_cnt counts the OUT-of-stock hours between 06:00 and 22:00
    # (0 = in stock all day). Check that on a sample before trusting it.
    head = df.head(2000)
    counted = np.stack(head["hours_stock_status"].to_numpy())[:, 6:22].sum(axis=1)
    if not (counted == head["stock_hour6_22_cnt"].to_numpy()).all():
        raise ValueError("stock_hour6_22_cnt does not match hours_stock_status[6:22]. "
                         "Check which value means out of stock.")
    return df


def simulate(n_stores: int = 20, n_products: int = 25, n_days: int = 90,
             service_quantile: float = 0.75, seed: int = 0) -> pd.DataFrame:
    """Simulate store-product-day rows in the FreshRetailNet-50K layout.

    Demand in each open hour is Poisson. Each morning the store holds a fixed
    stock; once it is sold the product is out of stock for the rest of the day.
    Stockouts therefore happen on high-demand days, which is exactly why
    censoring is not random.

    Extra column `true_hours_demand` holds the demand that would have been
    served with unlimited stock. Real data does not have it.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-03-28", periods=n_days, freq="D")

    # hourly shape of the day: a morning and an evening peak
    h = OPEN_HOURS
    shape = np.exp(-0.5 * ((h - 9) / 2.0) ** 2) + 1.3 * np.exp(-0.5 * ((h - 18) / 2.5) ** 2) + 0.15
    profile = shape / shape.sum()

    rows = []
    for s in range(n_stores):
        store_scale = rng.lognormal(0, 0.3)
        rain = rng.gamma(0.4, 6.0, n_days) * (rng.random(n_days) < 0.3)
        for p in range(n_products):
            base = rng.lognormal(2.6, 0.7) * store_scale          # mean daily demand
            promo = rng.random(n_days) < 0.15
            discount = np.where(promo, rng.choice([0.9, 0.8, 0.7], n_days), 1.0)
            dow = 1 + 0.25 * np.isin(dates.day_of_week, [5, 6])
            shock = rng.gamma(4.0, 1 / 4.0, n_days)               # day-to-day noise
            lam_day = base * dow * (1 + 2.5 * (1 - discount)) * (1 + 0.01 * rain) * shock
            stock = max(1, int(np.quantile(rng.poisson(base * rng.gamma(4.0, 1 / 4.0, 500)),
                                           service_quantile)))
            for d in range(n_days):
                demand = np.zeros(24, dtype=int)
                demand[h] = rng.poisson(lam_day[d] * profile)
                cum = demand.cumsum()
                sales = np.diff(np.minimum(cum, stock), prepend=0)
                out = (cum >= stock).astype(int)                  # sold out from this hour on
                out[:6] = 0
                out[22:] = 0
                rows.append((
                    s % 5, s, s % 3, p % 4, p % 8, p % 16, p, dates[d],
                    float(sales.sum()), sales.astype(float), int(out[h].sum()), out,
                    float(discount[d]), int(dates[d].day_of_week == 6), int(promo[d]),
                    float(rain[d]), 20.0, 60.0, 2.0, demand.astype(float),
                ))
    return pd.DataFrame(rows, columns=COLUMNS + ["true_hours_demand"])


def hourly_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (sales, out_of_stock) as two arrays of shape (rows, 24)."""
    sales = np.stack(df["hours_sale"].to_numpy()).astype(float)
    out = np.stack(df["hours_stock_status"].to_numpy()).astype(int)
    return sales, out


def series_id(df: pd.DataFrame) -> np.ndarray:
    """Integer id per store-product series, aligned with the rows of df."""
    return df.groupby(["store_id", "product_id"], sort=False).ngroup().to_numpy()

"""Stage 2. A test with a known answer.

Idea
----
Real data never shows the demand that was lost during a stockout, so we cannot
check a recovery method against the truth. The way round it:

  1. keep only the store-product-days with no stockout (demand fully observed);
  2. hide part of each day ourselves, as if a stockout had happened;
  3. ask each method to recover what we hid;
  4. compare with the truth, which we still hold.

We hide demand in two ways:
  * "random"         the stockout starts at a random hour, unrelated to demand;
  * "demand_driven"  the store holds a fixed stock and sells out when demand is
                     high. This is how stockouts really happen.

The gap between the two is a finding in itself: a method that looks fine under
random censoring can still be biased under realistic censoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPEN = slice(6, 22)


# ---------------------------------------------------------------- censoring --
def censor_random(demand: np.ndarray, share_days: float = 0.3,
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Censor a random share of days from a random hour between 10:00 and 20:00."""
    rng = rng or np.random.default_rng(0)
    n = demand.shape[0]
    out = np.zeros_like(demand, dtype=int)
    hit = rng.random(n) < share_days
    start = rng.integers(10, 21, n)
    hours = np.arange(24)[None, :]
    out[(hours >= start[:, None]) & (hours < 22) & hit[:, None]] = 1
    return demand * (1 - out), out


def censor_demand_driven(demand: np.ndarray, sid: np.ndarray,
                         stock_quantile: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Give every series a fixed daily stock; sales stop once it is sold."""
    daily = pd.Series(demand.sum(axis=1))
    stock = daily.groupby(sid).transform(lambda x: x.quantile(stock_quantile)).to_numpy()
    cum = demand.cumsum(axis=1)
    sales = np.diff(np.minimum(cum, stock[:, None]), axis=1, prepend=0)
    out = (cum >= stock[:, None]).astype(int)
    out[:, :6] = 0
    out[:, 22:] = 0
    out[daily.to_numpy() < stock] = 0          # never sold out on that day
    return sales, out


# ----------------------------------------------------------------- recovery --
def _profiles(sales: np.ndarray, out: np.ndarray, sid: np.ndarray, min_days: int = 5) -> np.ndarray:
    """Share of a day's demand that falls in each hour, per series.

    Estimated only from days the analyst sees in full. Series with too few such
    days use the profile of all series together.
    """
    clean = out.sum(axis=1) == 0
    hours = pd.DataFrame(sales[clean]).groupby(sid[clean]).sum()
    n_clean = pd.Series(clean).groupby(sid).sum()
    overall = sales[clean].sum(axis=0)
    overall = overall / overall.sum()
    prof = np.tile(overall, (sid.max() + 1, 1))
    ok = hours.index[(n_clean.reindex(hours.index) >= min_days) & (hours.sum(axis=1) > 0)]
    prof[ok] = hours.loc[ok].to_numpy() / hours.loc[ok].sum(axis=1).to_numpy()[:, None]
    return prof[sid]


def recover_naive(sales, out, sid):
    """Treat sales as demand. This is what a forecaster does by default."""
    return sales.sum(axis=1)


def recover_profile_scaling(sales, out, sid, min_exposure: float = 0.05):
    """Scale up the day: sales seen / share of the day that was observed."""
    prof = _profiles(sales, out, sid)
    exposure = (prof * (1 - out)).sum(axis=1)
    seen = (sales * (1 - out)).sum(axis=1)
    est = seen / np.maximum(exposure, min_exposure)
    return np.where(out.sum(axis=1) == 0, sales.sum(axis=1), est)


def recover_rate_exposure(sales, out, sid):
    """Add the series' average rate for the hours that were hidden.

    rate = total sales / total observed share of days (the Poisson estimate
    with exposure). Demand = sales seen + rate x share of the day hidden.
    """
    prof = _profiles(sales, out, sid)
    exposure = (prof * (1 - out)).sum(axis=1)
    seen = (sales * (1 - out)).sum(axis=1)
    rate = (pd.Series(seen).groupby(sid).transform("sum")
            / pd.Series(exposure).groupby(sid).transform("sum")).to_numpy()
    return seen + rate * (1 - exposure)


METHODS = {
    "naive (sales = demand)": recover_naive,
    "profile scaling": recover_profile_scaling,
    "rate x exposure": recover_rate_exposure,
}


# --------------------------------------------------------------- evaluation --
def evaluate(true_daily: np.ndarray, est_daily: np.ndarray, censored: np.ndarray) -> dict:
    """Bias and error, on censored days and on all days."""
    t, e = true_daily[censored], est_daily[censored]
    return {
        "bias_censored_days_%": 100 * (e.sum() - t.sum()) / t.sum(),
        "wape_censored_days_%": 100 * np.abs(e - t).sum() / t.sum(),
        "bias_all_days_%": 100 * (est_daily.sum() - true_daily.sum()) / true_daily.sum(),
    }


def run_experiment(demand: np.ndarray, sid: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Run both censoring mechanisms and all recovery methods. One table out."""
    rng = np.random.default_rng(seed)
    mechanisms = {
        "random": censor_random(demand, rng=rng),
        "demand_driven": censor_demand_driven(demand, sid),
    }
    true_daily = demand.sum(axis=1)
    rows = []
    for mech, (sales, out) in mechanisms.items():
        censored = out.sum(axis=1) > 0
        for name, fn in METHODS.items():
            res = evaluate(true_daily, fn(sales, out, sid), censored)
            rows.append({"censoring": mech, "method": name,
                         "days_censored_%": 100 * censored.mean(), **res})
    return pd.DataFrame(rows).round(2)

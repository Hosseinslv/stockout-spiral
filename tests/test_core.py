import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import censoring, data, newsvendor  # noqa: E402


def _truth():
    df = data.simulate(n_stores=6, n_products=10, n_days=60, seed=1)
    demand = np.stack(df["true_hours_demand"].to_numpy())
    return df, demand, data.series_id(df)


def test_simulated_sales_never_exceed_demand():
    df, demand, _ = _truth()
    sales, out = data.hourly_arrays(df)
    assert (sales <= demand + 1e-9).all()
    assert 0.02 < out[:, 6:22].mean() < 0.6


def test_censoring_only_removes_demand():
    _, demand, sid = _truth()
    for sales, out in (censoring.censor_random(demand), censoring.censor_demand_driven(demand, sid)):
        assert (sales <= demand + 1e-9).all()
        assert out.sum() > 0


def test_naive_underestimates_and_recovery_helps():
    _, demand, sid = _truth()
    res = censoring.run_experiment(demand, sid).set_index(["censoring", "method"])
    for mech in ("random", "demand_driven"):
        naive = res.loc[(mech, "naive (sales = demand)"), "bias_censored_days_%"]
        scaled = res.loc[(mech, "profile scaling"), "bias_censored_days_%"]
        assert naive < -5
        assert abs(scaled) < abs(naive)


def test_newsvendor():
    assert newsvendor.critical_ratio(3, 1) == 0.75
    assert newsvendor.cost(10, 12, 3, 1) == 6
    assert newsvendor.cost(10, 7, 3, 1) == 3

"""Stage 4. Turn a forecast into an order and put a price on being wrong.

For a perishable product ordered once a day:
    cost = lost margin on every unit of unmet demand
         + purchase cost of every unit thrown away
The best order is the demand quantile at the critical ratio cu / (cu + co).
"""
from __future__ import annotations

import numpy as np


def critical_ratio(margin: float, waste_cost: float) -> float:
    """Service level that balances lost sales against waste."""
    return margin / (margin + waste_cost)


def order_quantity(demand_samples: np.ndarray, margin: float, waste_cost: float) -> float:
    """Best order given samples (or a forecast distribution) of demand."""
    return float(np.quantile(demand_samples, critical_ratio(margin, waste_cost)))


def cost(order: np.ndarray, demand: np.ndarray, margin: float, waste_cost: float) -> np.ndarray:
    """Cost of each order against the demand that actually occurred."""
    order, demand = np.asarray(order, float), np.asarray(demand, float)
    return margin * np.maximum(demand - order, 0) + waste_cost * np.maximum(order - demand, 0)

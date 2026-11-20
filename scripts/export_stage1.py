"""Collect the stage 1 outputs into one Excel workbook.

Run after run_stage1.py:
    python scripts/export_stage1.py            # real data
    python scripts/export_stage1.py --simulated
"""
import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--simulated", action="store_true")
args = parser.parse_args()

out = Path("outputs") / ("simulated" if args.simulated else "real")
hour = pd.read_csv(out / "stockout_by_hour.csv")
disc = pd.read_csv(out / "stockout_by_discount.csv")
series = pd.read_csv(out / "series_summary.csv")
reg_text = (out / "stockout_regression.txt").read_text()

# Regression coefficients, parsed from the saved summary table.
rows = []
for line in reg_text.splitlines():
    parts = line.split()
    if parts and parts[0] in {"discount_depth", "rain", "holiday"}:
        rows.append([parts[0], *map(float, parts[1:7])])
reg = pd.DataFrame(rows, columns=["parameter", "estimate", "std_err", "t_stat",
                                  "p_value", "ci_lower", "ci_upper"])

share = series["open_hours_out_of_stock_pct"]
overview = pd.DataFrame([
    ("Store-product series", len(series)),
    ("Stores", series["store_id"].nunique()),
    ("Products", series["product_id"].nunique()),
    ("Store-product-days", int(disc["store_product_days"].sum())),
    ("Open hours (06-22) out of stock, %",
     round((disc["open_hours_out_of_stock_pct"] * disc["store_product_days"]).sum()
           / disc["store_product_days"].sum(), 2)),
    ("Days with any stockout, %",
     round((disc["days_with_any_stockout_pct"] * disc["store_product_days"]).sum()
           / disc["store_product_days"].sum(), 2)),
    ("Series out of stock > 25 % of open hours", int((share > 25).sum())),
    ("Series out of stock > 50 % of open hours", int((share > 50).sum())),
    ("Median fully observed days per series", float(series["fully_observed_days"].median())),
], columns=["measure", "value"])

path = out / "stage1_results.xlsx"
with pd.ExcelWriter(path, engine="openpyxl") as xl:
    for name, table in [("overview", overview), ("by_hour", hour), ("by_discount", disc),
                        ("regression", reg), ("series_summary", series)]:
        table.to_excel(xl, sheet_name=name, index=False)
    for ws in xl.book.worksheets:
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(
                12, min(45, max(len(str(c.value)) for c in col[:200]) + 2))
        ws.freeze_panes = "A2"

print(overview.to_string(index=False))
print(f"\nSaved {path}")

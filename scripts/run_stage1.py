"""Stage 1: describe stockouts. Usage:
    python scripts/run_stage1.py            # real data (after download_data.py)
    python scripts/run_stage1.py --simulated
    
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data, describe  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--simulated", action="store_true")
args = parser.parse_args()

df = data.simulate() if args.simulated else data.load_real()
tag = "simulated" if args.simulated else "real"
out = Path("outputs") / tag
out.mkdir(parents=True, exist_ok=True)

for name in ["stockout_by_hour", "stockout_by_discount", "series_summary"]:
    table = describe.run_sql(df, name)
    table.to_csv(out / f"{name}.csv", index=False)
    print(f"\n== {name} ==\n{table.head(16).to_string(index=False)}")

fit = describe.stockout_regression(df)
(out / "stockout_regression.txt").write_text(str(fit.summary))
print("\n== regression: share of open hours out of stock ==")
print(fit.summary.tables[1])

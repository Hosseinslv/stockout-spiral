"""Stage 2: hide demand we know, and see which method recovers it. Usage:
    python scripts/run_stage2.py            # real data (after download_data.py)
    python scripts/run_stage2.py --simulated
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import censoring, data  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--simulated", action="store_true")
parser.add_argument("--max-rows", type=int, default=1_000_000)
args = parser.parse_args()

df = data.simulate(service_quantile=0.9) if args.simulated else data.load_real()

# Ground truth = days with no stockout at all, so sales equal demand.
clean = df[df["stock_hour6_22_cnt"] == 0]
# Keep series with enough fully observed days to learn an hourly profile.
n = clean.groupby(["store_id", "product_id"])["dt"].transform("size")
clean = clean[n >= 20].head(args.max_rows).reset_index(drop=True)

demand, _ = data.hourly_arrays(clean)
sid = data.series_id(clean)
print(f"{len(clean):,} fully observed store-product-days from {sid.max() + 1:,} series")

table = censoring.run_experiment(demand, sid)
tag = "simulated" if args.simulated else "real"
out = Path("outputs") / tag
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "stage2_recovery.csv", index=False)
print(table.to_string(index=False))

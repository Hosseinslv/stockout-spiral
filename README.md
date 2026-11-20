# Learning from what you didn't sell

**Censored demand, AI forecasts and the stockout spiral in fresh retail**

Hossein Salavatinejad · work in progress, started November 2026

## The question

When a product sells out, the sales record shows what was sold, not what
customers wanted. A forecasting model trained on those sales forecasts too
low, the store orders too little, and the product sells out again.

1. How much do modern forecasting models underestimate demand when they learn
   from sales instead of demand?
2. What does that cost in lost sales and waste once the forecast drives orders?
3. Does the error grow when the model keeps retraining on the sales its own
   orders produced?
4. Which correction is worth using?

## Data

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
(Wang et al., 2025): 50,000 store-product series from 898 stores, hourly sales
and hourly stock status for perishable products, with discounts and weather.
Licence CC BY 4.0. The data is not stored in this repository.

## Design

| Stage | Question | Method | Code | Status |
|---|---|---|---|---|
| 1 | When and where do stockouts happen? | SQL descriptives; regression with store, product and weekday fixed effects | `sql/`, `src/describe.py` | Code ready |
| 2 | Which method recovers hidden demand? | Hide demand in fully observed days, recover it, compare with the truth | `src/censoring.py` | Code ready |
| 3 | How biased are forecasting models? | Gradient boosting and a pretrained time-series model, trained on sales vs. recovered demand | | Planned |
| 4 | What does the bias cost? | Daily order for a perishable product; lost sales against waste | `src/newsvendor.py` | Cost function ready |
| 5 | Does the bias grow? | Simulation in which orders follow the forecast and the forecast retrains on the result | | Planned |

Stage 2 hides demand in two ways: at random, and by letting a fixed stock sell
out on high-demand days. The second is how stockouts actually arise, and a
method that looks unbiased under random censoring need not be under the second.

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests                    # checks on simulated data

python scripts/run_stage1.py --simulated  # no download needed
python scripts/run_stage2.py --simulated

python scripts/download_data.py           # real data
python scripts/run_stage1.py
python scripts/run_stage2.py
```

Tables are written to `outputs/`.

## Results

None on the real data yet. The code has so far been run only on simulated
data (`src/data.py`), which copies the layout of the real dataset and also
keeps the true demand so that each method can be checked.

## Known limitations

- The data comes from one Chinese online fresh-food retailer. Whether patterns
  carry over to Swedish store retail is an open question.
- The fully observed days used as ground truth in stage 2 are days without a
  stockout, so they are lower-demand days than average.
- Substitution between products is ignored so far: demand lost on one product
  may show up as sales of another (Steeneck, Eng-Larsson and Jauffred, 2022).

## References

See [`docs/READING.md`](docs/READING.md).

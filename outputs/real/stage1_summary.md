# Stage 1 results: when and where do stockouts happen?

Data: FreshRetailNet-50K, training split. 50,000 store-product series
(898 stores, 865 products), 90 days each (2024-03-28 to 2024-06-25),
4.5 million store-product-days. Open hours are 06:00 to 22:00 (16 hours).

Full tables: `stage1_results.xlsx` (sheets: overview, by_hour, by_discount,
regression, series_summary). Raw outputs: the CSV files and
`stockout_regression.txt` in this folder.

## Headline numbers

| Measure | Value |
|---|---|
| Open hours out of stock | 19.9 % |
| Days with at least one out-of-stock hour | 44.3 % |
| Series out of stock more than 25 % of open hours | 7,898 (15.8 %) |
| Series out of stock more than 50 % of open hours | 699 (1.4 %) |
| Fully observed days per series (median) | 52 of 90 |

Stockouts are common. On almost half of all days the sales record is
censored for at least part of the day, so a model trained on raw sales
learns from truncated demand on a large share of its data.

## 1. Stockouts build up over the day

| Hour | 06 | 07 | 09 | 12 | 15 | 18 | 20 | 21 |
|---|---|---|---|---|---|---|---|---|
| Out of stock, % | 6.2 | 5.0 | 7.1 | 14.8 | 20.0 | 33.6 | 38.8 | 41.0 |

The rate is lowest at 07:00 (5.0 %) and rises steadily to 41.0 % in the
last open hour. The slightly higher rate at 06:00 is consistent with some
products not yet being restocked when the store opens. So most of the hidden
demand sits in the afternoon and evening. A recovery method has to
extrapolate the evening share of demand, which is exactly the part that is
least often observed.

## 2. Discounts: little difference in the raw data, a clear effect within series

| Price | Store-product-days | Open hours out of stock, % | Days with any stockout, % | Avg daily sales |
|---|---|---|---|---|
| Full price | 2,184,531 | 20.3 | 43.0 | 0.83 |
| Up to 10 % off | 721,282 | 17.3 | 43.5 | 1.02 |
| 10-20 % off | 847,707 | 20.4 | 46.2 | 0.93 |
| More than 20 % off | 746,480 | 20.7 | 46.4 | 1.53 |

In the raw comparison, the share of open hours out of stock barely changes
with the discount, while the share of days with any stockout rises from
43 % to 46 %. Discounts are not given at random, though: they may go to
products or stores that are restocked differently.

The regression compares each store and each product with itself (store,
product and weekday fixed effects, standard errors clustered by store):

| Variable | Estimate | Std. error | p-value | 95 % CI |
|---|---|---|---|---|
| Discount depth (1 - price ratio) | 0.117 | 0.006 | < 0.001 | 0.106 to 0.128 |
| Rain (precipitation) | -0.0001 | 0.0001 | 0.26 | -0.0003 to 0.0001 |
| Holiday | 0.0008 | 0.0012 | 0.49 | -0.0016 to 0.0033 |

A discount 10 percentage points deeper raises the share of open hours out
of stock by about 1.2 percentage points, around 6 % of the 19.9 % average.
Rain and holidays show no detectable effect. The model explains little of
the variation within series (R² 0.06 with fixed effects, 0.001 without),
so day-to-day stockouts are mostly driven by things these three variables
do not capture.

This matters for the later stages. Promotions raise demand and also raise
the chance that the extra demand goes unrecorded, so a model trained on
sales will tend to understate the promotion effect.

## 3. Stockouts are concentrated in some products

The series with the most stockouts are dominated by a few products. Product
757 (category 20) appears in 11 of the 16 series with the highest stockout
share, out of stock 76-83 % of open hours across many different stores. That
points to a product-level supply or ordering issue rather than a store
issue. The most extreme series (store 519, product 151) is out of stock 97 %
of open hours and has a single fully observed day.

## Caveats

- 16,039 rows (0.36 %) have a price ratio of 0, which the code reads as a
  100 % discount. They are probably data errors. They are too few to move
  the headline numbers, but they should be checked or dropped before the
  regression is reported formally. 34 rows have a price ratio slightly
  above 1 and are counted as full price.
- "Fully observed" days (no stockout) are 55.7 % of all days, but they are
  lower-demand days than average. Stage 2 uses them as ground truth, which
  should be kept in mind.
- The regression is descriptive. It shows where stockouts co-occur with
  discounts, not that discounts cause them.

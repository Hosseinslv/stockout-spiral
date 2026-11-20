-- Share of store-product-days that are out of stock, for each hour of the day.
-- hours_stock_status is a list of 24 values; UNNEST turns it into 24 rows.
WITH hourly AS (
    SELECT
        UNNEST(hours_stock_status)            AS out_of_stock,
        UNNEST(range(24))                     AS hour
    FROM sales
)
SELECT
    hour,
    ROUND(100.0 * AVG(out_of_stock), 2)       AS stockout_rate_pct,
    COUNT(*)                                  AS n
FROM hourly
WHERE hour BETWEEN 6 AND 21
GROUP BY hour
ORDER BY hour;

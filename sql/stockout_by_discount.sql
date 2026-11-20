-- Are promoted products out of stock more often?
-- discount is the price ratio: 1.0 = full price, 0.8 = 20 % off.
SELECT
    CASE
        WHEN discount >= 0.999 THEN '0  full price'
        WHEN discount >= 0.90  THEN '1  up to 10 % off'
        WHEN discount >= 0.80  THEN '2  10-20 % off'
        ELSE                        '3  more than 20 % off'
    END                                              AS discount_band,
    COUNT(*)                                         AS store_product_days,
    ROUND(100.0 * AVG(stock_hour6_22_cnt / 16.0), 2) AS open_hours_out_of_stock_pct,
    ROUND(100.0 * AVG(CASE WHEN stock_hour6_22_cnt > 0 THEN 1 ELSE 0 END), 2)
                                                     AS days_with_any_stockout_pct,
    ROUND(AVG(sale_amount), 2)                       AS avg_daily_sales
FROM sales
GROUP BY 1
ORDER BY 1;

-- One row per store-product series: how much it sells and how often it runs out.
-- Useful for Excel: export and build a pivot of stockout rate by category.
SELECT
    store_id,
    product_id,
    first_category_id,
    COUNT(*)                                             AS days,
    ROUND(AVG(sale_amount), 3)                           AS avg_daily_sales,
    ROUND(100.0 * AVG(stock_hour6_22_cnt / 16.0), 2) AS open_hours_out_of_stock_pct,
    SUM(CASE WHEN stock_hour6_22_cnt = 0 THEN 1 ELSE 0 END) AS fully_observed_days
FROM sales
GROUP BY store_id, product_id, first_category_id
ORDER BY open_hours_out_of_stock_pct DESC;

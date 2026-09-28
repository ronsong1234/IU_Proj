-- Class balance per split (label 0 = normal, 1 = pneumonia), plus an overall row.
-- Run against output/pneumoniamnist.duckdb, e.g.:
--   duckdb output/pneumoniamnist.duckdb < sql/split_summary.sql
SELECT
    COALESCE(split, 'all')                                  AS split,
    COUNT(*)                                                AS total_images,
    COUNT(*) FILTER (WHERE label = 0)                       AS normal_images,
    COUNT(*) FILTER (WHERE label = 1)                       AS pneumonia_images,
    ROUND(100.0 * COUNT(*) FILTER (WHERE label = 1) / COUNT(*), 2) AS pct_pneumonia
FROM image_metadata
GROUP BY ROLLUP (split)
ORDER BY
    CASE split WHEN 'train' THEN 1 WHEN 'val' THEN 2 WHEN 'test' THEN 3 ELSE 4 END;

-- How big is the normal vs pneumonia difference, and is it consistent across splits?
-- diff = pneumonia mean - normal mean; cohens_d = diff / pooled SD
-- (|d| ~0.2 small, ~0.5 medium, ~0.8 large).
WITH long AS (
    UNPIVOT (
        SELECT split, label, mean_intensity, std_intensity
        FROM image_metadata
    )
    ON mean_intensity, std_intensity
    INTO NAME statistic VALUE value
),
by_class AS (
    SELECT COALESCE(split, 'all') AS split, statistic, label,
           COUNT(*) AS n, AVG(value) AS m, VAR_SAMP(value) AS v
    FROM long
    GROUP BY GROUPING SETS ((split, statistic, label), (statistic, label))
)
SELECT
    statistic,
    n.split,
    ROUND(n.m, 2)                                                     AS normal_mean,
    ROUND(p.m, 2)                                                     AS pneumonia_mean,
    ROUND(p.m - n.m, 2)                                               AS diff,
    ROUND((p.m - n.m) / SQRT(((n.n - 1) * n.v + (p.n - 1) * p.v) / (n.n + p.n - 2)), 2) AS cohens_d
FROM by_class n
JOIN by_class p USING (split, statistic)
WHERE n.label = 0 AND p.label = 1
ORDER BY statistic,
    CASE n.split WHEN 'train' THEN 1 WHEN 'val' THEN 2 WHEN 'test' THEN 3 ELSE 4 END;

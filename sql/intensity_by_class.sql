-- Per-image intensity statistics compared between normal and pneumonia images (all splits).
-- One row per (statistic, class): mean, SD, median and quartiles of that statistic across images.
WITH long AS (
    UNPIVOT (
        SELECT label_name, mean_intensity, std_intensity, median_intensity,
               min_intensity::DOUBLE AS min_intensity, max_intensity::DOUBLE AS max_intensity
        FROM image_metadata
    )
    ON mean_intensity, std_intensity, median_intensity, min_intensity, max_intensity
    INTO NAME statistic VALUE value
)
SELECT
    statistic,
    label_name                               AS class,
    COUNT(*)                                 AS n_images,
    ROUND(AVG(value), 2)                     AS mean,
    ROUND(STDDEV_SAMP(value), 2)             AS sd,
    ROUND(QUANTILE_CONT(value, 0.25), 2)     AS q1,
    ROUND(MEDIAN(value), 2)                  AS median,
    ROUND(QUANTILE_CONT(value, 0.75), 2)     AS q3
FROM long
GROUP BY statistic, label_name
ORDER BY
    CASE statistic WHEN 'mean_intensity' THEN 1 WHEN 'std_intensity' THEN 2
                   WHEN 'median_intensity' THEN 3 WHEN 'min_intensity' THEN 4 ELSE 5 END,
    class;

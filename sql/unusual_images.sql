-- Images with unusual brightness or contrast, by robust z-score (all splits, both classes).
--   robust_z = (x - median) / (1.4826 * MAD),  MAD = median(|x - median|)
-- The median/MAD are not dragged by the outliers themselves (unlike mean/SD), and
-- 1.4826 makes the scale match an SD for normal data. |robust_z| > 3.5 is the
-- Iglewicz & Hoaglin (1993) outlier cutoff.
-- One row per (image, reason); rank_in_reason 1 = most extreme.
WITH ref AS (
    SELECT MEDIAN(mean_intensity) AS med_mean, MEDIAN(std_intensity) AS med_std
    FROM image_metadata
),
scale AS (
    SELECT 1.4826 * MEDIAN(ABS(mean_intensity - med_mean)) AS mad_mean,
           1.4826 * MEDIAN(ABS(std_intensity - med_std))   AS mad_std
    FROM image_metadata, ref
),
scored AS (
    SELECT m.image_id, m.split, m.split_index, m.label_name, m.pixel_sha1,
           m.mean_intensity, m.std_intensity, m.min_intensity, m.max_intensity,
           (m.mean_intensity - med_mean) / mad_mean AS z_mean,
           (m.std_intensity  - med_std)  / mad_std  AS z_std
    FROM image_metadata m, ref, scale
),
flagged AS (
    SELECT *, 'very dark'           AS reason, z_mean AS z FROM scored WHERE z_mean < -3.5
    UNION ALL
    SELECT *, 'very bright',                   z_mean      FROM scored WHERE z_mean >  3.5
    UNION ALL
    SELECT *, 'very low contrast',             z_std       FROM scored WHERE z_std  < -3.5
    UNION ALL
    SELECT *, 'very high contrast',            z_std       FROM scored WHERE z_std  >  3.5
)
SELECT
    reason, image_id, split, split_index, label_name,
    ROUND(mean_intensity, 1) AS mean_intensity,
    ROUND(std_intensity, 1)  AS std_intensity,
    min_intensity, max_intensity, pixel_sha1,
    ROUND(z, 2)              AS robust_z,
    ROW_NUMBER() OVER (PARTITION BY reason ORDER BY ABS(z) DESC) AS rank_in_reason
FROM flagged
ORDER BY reason, rank_in_reason;

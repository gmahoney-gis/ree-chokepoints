-- Query B (task 2.17): GDELT tension totals per country, root code and month, BEFORE the
-- distance filter (context only; never part of the score). Same window, root codes,
-- location types, duplicate rule and source buckets as Query A. Country codes are
-- FIPS 10-4 (GDELT's own); S05 converts them to ISO-3.

WITH filtered AS (
  SELECT
    GLOBALEVENTID, SQLDATE, DATEADDED, EventRootCode, NumSources,
    Actor1Code, Actor2Code, ActionGeo_CountryCode, ActionGeo_FeatureID
  FROM `gdelt-bq.gdeltv2.events_partitioned`
  WHERE
    _PARTITIONTIME >= TIMESTAMP('2024-09-30')
    AND _PARTITIONTIME < TIMESTAMP('2025-10-08')
    AND SQLDATE BETWEEN 20241001 AND 20250930
    AND EventRootCode IN ('10','11','12','13','14','15','16','17')
    AND ActionGeo_Type IN (3, 4)
),

reports AS (
  SELECT
    *,
    FIRST_VALUE(GLOBALEVENTID) OVER ranked AS merged_id,
    ROW_NUMBER() OVER ranked AS report_rank,
    MAX(NumSources) OVER same_event AS numsources_max,
    SUM(NumSources) OVER same_event AS numsources_sum,
    CAST(FORMAT_TIMESTAMP('%Y%m%d%H%M%S',
         TIMESTAMP_ADD(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING)),
                       INTERVAL 30 DAY)) AS INT64) AS mention_cutoff
  FROM filtered
  WINDOW
    same_event AS (PARTITION BY SQLDATE, EventRootCode, Actor1Code, Actor2Code, ActionGeo_FeatureID),
    ranked AS (same_event ORDER BY NumSources DESC, GLOBALEVENTID)
),

mentions AS (
  SELECT GLOBALEVENTID, MentionTimeDate, LOWER(MentionSourceName) AS outlet
  FROM `gdelt-bq.gdeltv2.eventmentions_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('2024-09-30')
    AND _PARTITIONTIME < TIMESTAMP('2025-11-07')
),

outlets AS (
  SELECT r.merged_id, COUNT(DISTINCT m.outlet) AS outlet_count
  FROM reports AS r
  JOIN mentions AS m
    ON m.GLOBALEVENTID = r.GLOBALEVENTID
   AND m.MentionTimeDate <= r.mention_cutoff
  GROUP BY r.merged_id
),

merged AS (
  SELECT r.*, IFNULL(o.outlet_count, 0) AS outlet_count
  FROM reports AS r
  LEFT JOIN outlets AS o USING (merged_id)
  WHERE r.report_rank = 1
)

SELECT
  ActionGeo_CountryCode AS country_fips,
  CAST(EventRootCode AS INT64) AS root_code,
  DIV(SQLDATE, 100) AS month,                       -- yyyymm
  LEAST(outlet_count, 3) AS outlet_bucket,
  LEAST(numsources_max, 3) AS numsources_bucket,
  LEAST(numsources_sum, 3) AS numsources_sum_bucket,
  COUNT(*) AS event_count,
  SUM(outlet_count) AS total_outlets,
  SUM(numsources_max) AS total_numsources
FROM merged
GROUP BY country_fips, root_code, month, outlet_bucket, numsources_bucket, numsources_sum_bucket
ORDER BY country_fips, month, root_code, outlet_bucket, numsources_bucket, numsources_sum_bucket

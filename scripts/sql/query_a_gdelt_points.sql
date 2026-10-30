-- Query A (tasks 2.16 and 2.18): GDELT tension events near the chokepoints, one row per
-- location, day, root code, international flag, state_actor flag and source buckets.
-- Settings used (config/settings.yaml): events.window_start/window_end, events.gdelt
-- (cameo_root_codes, keep_location_types, dedupe_on), events.prefilter_radius_km.
-- Chokepoint points: all 28 in config/chokepoint_screening.csv (as S05's ACLED prefilter;
-- gated straits use their point). No URLs, outlet names or actor names leave BigQuery.

WITH chokepoints AS (
  SELECT chokepoint_id, name, ST_GEOGPOINT(lon, lat) AS point
  FROM UNNEST([
    STRUCT('chokepoint1' AS chokepoint_id, 'Suez Canal' AS name, 32.43688221 AS lon, 30.59334599 AS lat),
    STRUCT('chokepoint2' AS chokepoint_id, 'Panama Canal' AS name, -79.76723825 AS lon, 9.120512367 AS lat),
    STRUCT('chokepoint3' AS chokepoint_id, 'Bosporus Strait' AS name, 29.09150126 AS lon, 41.16928167 AS lat),
    STRUCT('chokepoint4' AS chokepoint_id, 'Bab el-Mandeb Strait' AS name, 43.34954476 AS lon, 12.78859715 AS lat),
    STRUCT('chokepoint5' AS chokepoint_id, 'Malacca Strait' AS name, 102.6651061 AS lon, 1.516954817 AS lat),
    STRUCT('chokepoint6' AS chokepoint_id, 'Strait of Hormuz' AS name, 56.85984844 AS lon, 26.29685349 AS lat),
    STRUCT('chokepoint7' AS chokepoint_id, 'Cape of Good Hope' AS name, 20.88273666 AS lon, -34.92728556 AS lat),
    STRUCT('chokepoint8' AS chokepoint_id, 'Gibraltar Strait' AS name, -5.754895722 AS lon, 35.94227416 AS lat),
    STRUCT('chokepoint9' AS chokepoint_id, 'Dover Strait' AS name, 1.505839717 AS lon, 51.03022414 AS lat),
    STRUCT('chokepoint10' AS chokepoint_id, 'Oresund Strait' AS name, 12.85079477 AS lon, 55.50784043 AS lat),
    STRUCT('chokepoint11' AS chokepoint_id, 'Taiwan Strait' AS name, 119.8313644 AS lon, 24.72350972 AS lat),
    STRUCT('chokepoint12' AS chokepoint_id, 'Korea Strait' AS name, 129.209206 AS lon, 34.13076811 AS lat),
    STRUCT('chokepoint13' AS chokepoint_id, 'Tsugaru Strait' AS name, 140.3532743 AS lon, 41.32803599 AS lat),
    STRUCT('chokepoint14' AS chokepoint_id, 'Luzon Strait' AS name, 121.352295 AS lon, 20.48889071 AS lat),
    STRUCT('chokepoint15' AS chokepoint_id, 'Lombok Strait' AS name, 115.801433 AS lon, -8.419145141 AS lat),
    STRUCT('chokepoint16' AS chokepoint_id, 'Ombai Strait' AS name, 125.0909814 AS lon, -8.3985412 AS lat),
    STRUCT('chokepoint17' AS chokepoint_id, 'Bohai Strait' AS name, 120.9000492 AS lon, 38.37304425 AS lat),
    STRUCT('chokepoint18' AS chokepoint_id, 'Torres Strait' AS name, 142.2475287 AS lon, -9.862534132 AS lat),
    STRUCT('chokepoint19' AS chokepoint_id, 'Sunda Strait' AS name, 105.7752199 AS lon, -5.96684057 AS lat),
    STRUCT('chokepoint20' AS chokepoint_id, 'Makassar Strait' AS name, 119.2570654 AS lon, 0.352280587 AS lat),
    STRUCT('chokepoint21' AS chokepoint_id, 'Magellan Strait' AS name, -69.59483246 AS lon, -52.64033967 AS lat),
    STRUCT('chokepoint22' AS chokepoint_id, 'Yucatan Channel' AS name, -85.64727922 AS lon, 21.81525419 AS lat),
    STRUCT('chokepoint23' AS chokepoint_id, 'Windward Passage' AS name, -73.6975399 AS lon, 19.98618719 AS lat),
    STRUCT('chokepoint24' AS chokepoint_id, 'Mona Passage' AS name, -67.71135449 AS lon, 18.44873557 AS lat),
    STRUCT('chokepoint25' AS chokepoint_id, 'Balabac Strait' AS name, 117.1146217 AS lon, 7.413614993 AS lat),
    STRUCT('chokepoint26' AS chokepoint_id, 'Bering Strait' AS name, -165.5498286 AS lon, 65.96648859 AS lat),
    STRUCT('chokepoint27' AS chokepoint_id, 'Mindoro Strait' AS name, 120.4034279 AS lon, 12.46831537 AS lat),
    STRUCT('chokepoint28' AS chokepoint_id, 'Kerch Strait' AS name, 36.5439017 AS lon, 45.26676309 AS lat)
  ])
),

-- All chokepoint points as one multipoint: "within 500 km of any point" = within 500 km of this.
all_points AS (
  SELECT ST_UNION_AGG(point) AS points FROM chokepoints
),

-- Step 1. Window, root codes 10-17, city/town locations, valid coordinates.
filtered AS (
  SELECT
    GLOBALEVENTID, SQLDATE, DATEADDED, EventRootCode, NumSources,
    Actor1Code, Actor2Code, Actor1CountryCode, Actor2CountryCode,
    Actor1Type1Code, Actor1Type2Code, Actor1Type3Code,
    Actor2Type1Code, Actor2Type2Code, Actor2Type3Code,
    ActionGeo_Type, ActionGeo_CountryCode, ActionGeo_Lat, ActionGeo_Long, ActionGeo_FeatureID
  FROM `gdelt-bq.gdeltv2.events_partitioned`
  WHERE
    -- Partitions are the day GDELT added the event. One day before the window (time-zone
    -- safety) to 7 days after it (late-added events); this only limits the bytes scanned,
    -- the SQLDATE line below is the real window.
    _PARTITIONTIME >= TIMESTAMP('2024-09-30')
    AND _PARTITIONTIME < TIMESTAMP('2025-10-08')
    AND SQLDATE BETWEEN 20241001 AND 20250930                      -- events.window_start / window_end
    AND EventRootCode IN ('10','11','12','13','14','15','16','17') -- events.gdelt.cameo_root_codes
    AND ActionGeo_Type IN (3, 4)                                   -- events.gdelt.keep_location_types
    AND ActionGeo_Lat BETWEEN -90 AND 90
    AND ActionGeo_Long BETWEEN -180 AND 180
),

-- Step 2. Within 500 km (events.prefilter_radius_km, the largest ring) of any chokepoint point.
near AS (
  SELECT f.*
  FROM filtered AS f CROSS JOIN all_points AS a
  WHERE ST_DWITHIN(SAFE.ST_GEOGPOINT(f.ActionGeo_Long, f.ActionGeo_Lat), a.points, 500000)
),

-- Step 3. Group repeated reports of one event (events.gdelt.dedupe_on: date,
-- event_root_code, actor1, actor2, location). Every report is kept for now, labelled with
-- merged_id = the ID of the report with the most sources, which stands for the event.
reports AS (
  SELECT
    *,
    FIRST_VALUE(GLOBALEVENTID) OVER ranked AS merged_id,
    ROW_NUMBER() OVER ranked AS report_rank,
    MAX(NumSources) OVER same_event AS numsources_max,
    SUM(NumSources) OVER same_event AS numsources_sum,
    -- Mentions count only up to 30 days after GDELT first recorded this report
    -- (same follow-up for every event; DATEADDED and MentionTimeDate are yyyymmddhhmmss).
    CAST(FORMAT_TIMESTAMP('%Y%m%d%H%M%S',
         TIMESTAMP_ADD(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING)),
                       INTERVAL 30 DAY)) AS INT64) AS mention_cutoff
  FROM near
  WINDOW
    same_event AS (PARTITION BY SQLDATE, EventRootCode, Actor1Code, Actor2Code, ActionGeo_FeatureID),
    ranked AS (same_event ORDER BY NumSources DESC, GLOBALEVENTID)
),

-- Step 4. Mentions: one row per article (or other document) that mentions an event.
-- Partitions = day of the mention: from the first event partition to 30 days after the
-- last one (2025-10-07 + 30 days = 2025-11-06, so up to but not including 2025-11-07).
mentions AS (
  SELECT GLOBALEVENTID, MentionTimeDate, LOWER(MentionSourceName) AS outlet
  FROM `gdelt-bq.gdeltv2.eventmentions_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('2024-09-30')
    AND _PARTITIONTIME < TIMESTAMP('2025-11-07')
),

-- Step 5. Distinct news outlets per merged event, across all of its reports.
outlets AS (
  SELECT r.merged_id, COUNT(DISTINCT m.outlet) AS outlet_count
  FROM reports AS r
  JOIN mentions AS m
    ON m.GLOBALEVENTID = r.GLOBALEVENTID
   AND m.MentionTimeDate <= r.mention_cutoff
  GROUP BY r.merged_id
),

-- Step 6. One row per merged event (the report with the most sources), with its outlet count.
merged AS (
  SELECT r.*, IFNULL(o.outlet_count, 0) AS outlet_count   -- 0 = no mention found (S05 logs these)
  FROM reports AS r
  LEFT JOIN outlets AS o USING (merged_id)
  WHERE r.report_rank = 1
),

-- Step 7. Flags and source buckets (1, 2 or 3; 3 means 3 or more).
flagged AS (
  SELECT
    *,
    -- international: both actor countries known and different
    (IFNULL(Actor1CountryCode, '') != '' AND IFNULL(Actor2CountryCode, '') != ''
     AND Actor1CountryCode != Actor2CountryCode) AS international,
    -- state_actor: either actor typed as government, military, police or intelligence.
    -- A bare country code (no type code) is not a state actor.
    EXISTS (
      SELECT 1 FROM UNNEST([Actor1Type1Code, Actor1Type2Code, Actor1Type3Code,
                            Actor2Type1Code, Actor2Type2Code, Actor2Type3Code]) AS t
      WHERE t IN ('GOV', 'MIL', 'COP', 'SPY')
    ) AS state_actor,
    LEAST(outlet_count, 3) AS outlet_bucket,              -- main: distinct outlets (0 = none found)
    LEAST(numsources_max, 3) AS numsources_bucket,        -- fallback: largest NumSources
    LEAST(numsources_sum, 3) AS numsources_sum_bucket     -- fallback (b): NumSources added up
  FROM merged
)

SELECT
  ActionGeo_Lat AS lat,
  ActionGeo_Long AS lon,
  ActionGeo_FeatureID AS feature_id,
  ActionGeo_CountryCode AS country_fips,
  ActionGeo_Type AS location_type,
  SQLDATE AS event_date,
  CAST(EventRootCode AS INT64) AS root_code,
  international,
  state_actor,
  outlet_bucket,
  numsources_bucket,
  numsources_sum_bucket,
  COUNT(*) AS event_count,
  SUM(outlet_count) AS total_outlets,
  SUM(numsources_max) AS total_numsources
FROM flagged
GROUP BY lat, lon, feature_id, country_fips, location_type, event_date, root_code,
         international, state_actor, outlet_bucket, numsources_bucket, numsources_sum_bucket
ORDER BY event_date, root_code, country_fips, feature_id

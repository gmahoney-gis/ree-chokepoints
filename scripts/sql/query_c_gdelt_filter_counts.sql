-- Query C (optional, for the S05 log): how many GDELT rows each filter keeps and removes,
-- before duplicates are merged. Queries A and B only return what was kept; CLAUDE.md asks
-- the log to show what each filter removed (methodology: "the number removed is logged").
-- One row per location type. Same 28 chokepoint points as Query A. Counts are reports
-- (rows) before duplicates are merged; Query A and B give the counts after merging.

WITH chokepoints AS (
  SELECT ST_GEOGPOINT(lon, lat) AS point
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
all_points AS (SELECT ST_UNION_AGG(point) AS points FROM chokepoints),

window_rows AS (
  SELECT EventRootCode, ActionGeo_Type, ActionGeo_Lat, ActionGeo_Long
  FROM `gdelt-bq.gdeltv2.events_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('2024-09-30')
    AND _PARTITIONTIME < TIMESTAMP('2025-10-08')
    AND SQLDATE BETWEEN 20241001 AND 20250930
)

SELECT
  w.ActionGeo_Type AS location_type,
  COUNT(*) AS rows_in_window,
  COUNTIF(w.EventRootCode IN ('10','11','12','13','14','15','16','17')) AS rows_root_10_17,
  COUNTIF(w.EventRootCode IN ('10','11','12','13','14','15','16','17')
          AND w.ActionGeo_Lat BETWEEN -90 AND 90 AND w.ActionGeo_Long BETWEEN -180 AND 180
          AND ST_DWITHIN(SAFE.ST_GEOGPOINT(w.ActionGeo_Long, w.ActionGeo_Lat), a.points, 500000))
    AS rows_root_10_17_within_500km
FROM window_rows AS w CROSS JOIN all_points AS a
GROUP BY location_type
ORDER BY location_type

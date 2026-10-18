"""S03: chokepoint points and true-distance zones (task 2.9).

Reads the hand-made screening file (config/chokepoint_screening.csv, tasks 2.7-2.8)
and writes two layers to the project database:

- chokepoints: one point for every candidate, with its included flag and reason.
- zones: a true-distance circle for each included chokepoint only. The radius is
  chokepoints.zone_radius_km, or the override for that chokepoint by name.

Each circle is built in an azimuthal equidistant projection centred on its own
chokepoint (distances from the centre are true there), then converted to EPSG:4326.
The projection uses the same spherical Earth radius as common.great_circle_km, so
the zone edge sits at exactly the radius by the project's great-circle distance.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.ops import transform

import common

REQUIRED_COLUMNS = ["chokepoint_id", "name", "lat", "lon", "included", "reason",
                    "narrowest_width_km", "territorial_seas_overlap"]


def read_screening(settings, log):
    """Read the screening file and check it before anything is built."""
    path = common.ROOT / settings["chokepoints"]["screening_file"]
    table = pd.read_csv(path, dtype={"chokepoint_id": str, "name": str, "included": str,
                                     "reason": str, "territorial_seas_overlap": str})
    log.info("Read %s: %d rows", path, len(table))

    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"Screening file is missing columns: {missing}")
    if table["chokepoint_id"].duplicated().any():
        dupes = table.loc[table["chokepoint_id"].duplicated(), "chokepoint_id"].tolist()
        raise ValueError(f"Duplicate chokepoint IDs in screening file: {dupes}")

    table["included"] = table["included"].str.strip().str.lower()
    bad = table.loc[~table["included"].isin(["yes", "no"]), "chokepoint_id"].tolist()
    if bad:
        raise ValueError(f"included must be yes or no; check {bad}")
    if not (table["lat"].between(-90, 90).all() and table["lon"].between(-180, 180).all()):
        raise ValueError("Screening file has coordinates outside the valid range")
    return table


def assign_radii(table, settings, log):
    """Give every chokepoint its zone radius in km: the default or its named override."""
    default = settings["chokepoints"]["zone_radius_km"]
    overrides = settings["chokepoints"].get("zone_radius_overrides_km") or {}
    unknown = sorted(set(overrides) - set(table["name"]))
    if unknown:
        raise ValueError(f"zone_radius_overrides_km names not in screening file: {unknown}")

    table["zone_radius_km"] = table["name"].map(overrides).fillna(default).astype(float)
    log.info("Task 2.9 | default zone radius %s km; overrides: %s", default,
             ", ".join(f"{name} {km} km" for name, km in overrides.items()) or "none")
    return table


def build_points(table, settings):
    """Task 2.9: point layer for every candidate chokepoint, included or not."""
    geometry = gpd.points_from_xy(table["lon"], table["lat"])
    return gpd.GeoDataFrame(table, geometry=geometry, crs=settings["project"]["storage_crs"])


def true_distance_circle(lon, lat, radius_km, settings):
    """Circle of radius_km around (lon, lat), returned in storage coordinates.

    Built in an azimuthal equidistant projection centred on the point, on a sphere
    with the project Earth radius, so every edge vertex is radius_km away by
    great-circle distance.
    """
    radius_m = settings["project"]["earth_radius_km"] * 1000
    local = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +R={radius_m} +units=m +no_defs")
    to_storage = Transformer.from_crs(local, settings["project"]["storage_crs"], always_xy=True)
    circle = Point(0, 0).buffer(radius_km * 1000,
                                quad_segs=settings["chokepoints"]["zone_quad_segments"])
    zone = transform(to_storage.transform, circle)

    # A zone across the 180th meridian would wrap around the globe in EPSG:4326.
    min_lon, _, max_lon, _ = zone.bounds
    if max_lon - min_lon > 180:
        raise ValueError(f"Zone at ({lon}, {lat}) crosses the antimeridian; split it before writing")
    return zone


def build_zones(points, settings, log):
    """Task 2.9: zone layer for included chokepoints only."""
    included = points[points["included"] == "yes"]
    zones = gpd.GeoDataFrame(
        included[["chokepoint_id", "name", "zone_radius_km", "included", "reason"]].copy(),
        geometry=[true_distance_circle(lon, lat, r, settings)
                  for lon, lat, r in zip(included["lon"], included["lat"], included["zone_radius_km"])],
        crs=settings["project"]["storage_crs"],
    )
    check_zone_edges(zones, included, settings, log)
    return zones


def check_zone_edges(zones, included, settings, log):
    """Measure each zone edge back to its centre by great-circle distance and log it."""
    worst = 0.0
    for zone, lon, lat, radius in zip(zones.geometry, included["lon"], included["lat"],
                                      included["zone_radius_km"]):
        edge = np.asarray(zone.exterior.coords)
        dist = common.great_circle_km(lon, lat, edge[:, 0], edge[:, 1], settings)
        worst = max(worst, float(np.abs(dist - radius).max()))
    log.info("Task 2.9 | zone edge check: largest gap between edge distance and radius = %.6f km", worst)
    if worst > 0.01:
        raise ValueError(f"Zone edges are up to {worst:.4f} km off their radius")


def log_counts(points, zones, log):
    """Log the counts reported in Lab 7."""
    excluded = points[points["included"] == "no"]
    log.info("Task 2.9 | chokepoints in screening file: %d", len(points))
    common.log_filter(log, "Task 2.9 | included chokepoints", int((points["included"] == "yes").sum()),
                      len(excluded), "included = no in screening file")
    for _, row in excluded.iterrows():
        log.info("Task 2.9 | excluded: %s (%s): %s", row["name"], row["chokepoint_id"], row["reason"])
    log.info("Task 2.9 | zones written: %d", len(zones))
    for _, row in zones.iterrows():
        log.info("Task 2.9 | zone %-13s %-22s radius %5.0f km", row["chokepoint_id"], row["name"],
                 row["zone_radius_km"])


def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)

    table = read_screening(settings, log)
    table = assign_radii(table, settings, log)
    points = build_points(table, settings)
    zones = build_zones(points, settings, log)
    log_counts(points, zones, log)

    common.write_layer(points, settings, "chokepoints", log)
    common.write_layer(zones, settings, "zones", log)
    log.info("Run of s03_chokepoints finished")


if __name__ == "__main__":
    main()

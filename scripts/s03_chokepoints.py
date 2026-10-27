"""S03: chokepoint points and true-distance zones (task 2.9).

Reads the hand-made screening file (config/chokepoint_screening.csv, tasks 2.7-2.8)
and writes two layers to the project database:

- chokepoints: one point for every candidate, with its included flag and reason.
- zones: a true-distance circle for each included chokepoint only. The radius is
  chokepoints.zone_radius_km, or the override for that chokepoint by name.

- gates: a gate line for each chokepoint listed in config/chokepoint_gates.csv
  (Decision 12: the four wide straits). S06 and S08 use the gate in place of the zone.

Each circle is built in an azimuthal equidistant projection centred on its own
chokepoint (distances from the centre are true there), then converted to EPSG:4326.
The projection uses the same spherical Earth radius as common.great_circle_km, so
the zone edge sits at exactly the radius by the project's great-circle distance.

Each gate is checked before it is written: its chokepoint must be included, its WKT must
be a line, and both ends must lie on land (Natural Earth countries). The log gives its
great-circle length, the km over water and the number of searoute network edges crossing it.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import searoute
from pyproj import CRS, Transformer
from shapely import wkt
from shapely.affinity import translate
from shapely.errors import ShapelyError
from shapely.geometry import LineString, Point
from shapely.ops import transform, unary_union

import common

REQUIRED_COLUMNS = ["chokepoint_id", "name", "lat", "lon", "included", "reason",
                    "narrowest_width_km", "territorial_seas_overlap"]
GATE_COLUMNS = ["chokepoint_id", "name", "wkt", "source", "notes"]
WATER_STEP_DEG = 0.005     # gate pieces are this short (about 0.5 km) when measuring km over water


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


def read_gates(points, settings, log):
    """Decision 12: read the gate file and check its IDs and lines.

    Stops with an error if an ID is not an included chokepoint, its name differs from the
    screening file, or its WKT does not parse as a line.
    """
    path = common.ROOT / settings["chokepoints"]["gates_file"]
    table = pd.read_csv(path, dtype=str)
    log.info("Decision 12 | read %s: %d gates", path, len(table))
    missing = [c for c in GATE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"Gate file is missing columns: {missing}")

    included = points[points["included"] == "yes"].set_index("chokepoint_id")["name"]
    lines = []
    for row in table.itertuples():
        if row.chokepoint_id not in included.index:
            raise ValueError(f"Gate file: {row.chokepoint_id} ({row.name}) is not an included chokepoint")
        if row.name != included[row.chokepoint_id]:
            raise ValueError(f"Gate file: {row.chokepoint_id} is named '{row.name}', but "
                             f"'{included[row.chokepoint_id]}' in the screening file")
        try:
            line = wkt.loads(row.wkt)
        except (ShapelyError, TypeError) as err:
            raise ValueError(f"Gate file: WKT for {row.chokepoint_id} does not parse: {err}") from err
        if line.geom_type != "LineString" or not line.is_valid or len(line.coords) < 2:
            raise ValueError(f"Gate file: WKT for {row.chokepoint_id} must be a LINESTRING")
        lines.append(line)
    if table["chokepoint_id"].duplicated().any():
        raise ValueError("Gate file has more than one gate for a chokepoint")
    return gpd.GeoDataFrame(table[GATE_COLUMNS[:2] + GATE_COLUMNS[3:]], geometry=lines,
                            crs=settings["project"]["storage_crs"])


def read_land(settings, log):
    """Decision 12: Natural Earth country polygons, used as land for the gate checks."""
    path = common.get_path(settings, "raw_files.natural_earth_countries")
    land = pyogrio.read_dataframe(path, columns=["ADMIN"])
    log.info("Read %s (%d features)", path.name, len(land))
    return land


def network_edges(settings):
    """Every searoute network edge as a line, without the zero-length edges joining 180 and -180."""
    lines = [LineString([u, v]) for u, v in searoute.setup_M().edges()
             if not (abs(u[0]) == 180 and abs(v[0]) == 180 and u[1] == v[1] and u[0] != v[0])]
    return gpd.GeoSeries(lines, crs=settings["project"]["storage_crs"])


def line_km(coords, settings):
    """Great-circle length of a line in km, summed between its vertices."""
    xy = np.asarray(coords)
    return float(np.sum(common.great_circle_km(xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1], settings)))


def check_gates(gates, land, edges, settings, log):
    """Decision 12: both gate ends on land; log length, km over water and edges crossing.

    Edges are tested as S06 tests routes: straight lines in longitude/latitude, with the
    gate copied 360 degrees east and west.
    """
    for i, row in gates.iterrows():
        coords = list(row.geometry.coords)
        for end in (coords[0], coords[-1]):
            if not land.contains(Point(end)).any():
                raise ValueError(f"Gate {row.chokepoint_id} ({row['name']}): end {end} is not on land")
        near = unary_union(land.geometry[land.intersects(row.geometry.buffer(0.5))].values)
        water = row.geometry.segmentize(WATER_STEP_DEG).difference(near)
        water_km = sum(line_km(p.coords, settings) for p in getattr(water, "geoms", [water]) if not p.is_empty)
        crossing = set()
        for dx in (-360.0, 0.0, 360.0):
            crossing |= set(edges.index[edges.intersects(translate(row.geometry, xoff=dx))])
        gates.loc[i, "length_km"] = round(line_km(coords, settings), 1)
        gates.loc[i, "water_km"] = round(water_km, 1)
        gates.loc[i, "network_edges"] = len(crossing)
        log.info("Decision 12 | gate %-13s %-16s %d vertices, ends on land | length %6.1f km | over "
                 "water %6.1f km | network edges crossing %d", row.chokepoint_id, row["name"],
                 len(coords), gates.loc[i, "length_km"], water_km, len(crossing))
    gates["network_edges"] = gates["network_edges"].astype("int64")
    return gates


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
    # Task 2.9 / Decision 12: gate lines for the wide straits.
    gates = read_gates(points, settings, log)
    gates = check_gates(gates, read_land(settings, log), network_edges(settings), settings, log)

    common.write_layer(points, settings, "chokepoints", log)
    common.write_layer(zones, settings, "zones", log)
    common.write_layer(gates, settings, "gates", log)
    log.info("Run of s03_chokepoints finished")


if __name__ == "__main__":
    main()

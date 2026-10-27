"""Shared tools used by every script (task 0.4).

- Settings: load config/settings.yaml and resolve paths from it.
- Logging: one log per script run, with standard filter-count lines.
- Distance: great-circle (haversine) distance in kilometres.
- Country codes: clean ISO-3 codes, flag non-country codes, apply remaps.
- GeoPackage: read and write project layers by their settings key.
- Events to count: the IMO GISIS counting rule (Decision 9), so every script drops the
  same incidents.
- Crossing geometry: the gate line or zone each chokepoint is crossed at (Decision 12).
- Sea routes: searoute paths, their length in nautical miles, splitting at 180 degrees
  (S06 and S08).
"""

import logging
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio
import searoute
import yaml
from shapely.geometry import LineString, MultiLineString

# The repository root is the folder above scripts/, so paths work from any folder.
ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

def load_settings(path=None):
    """Read the settings file and return it as a dictionary."""
    path = Path(path) if path else SETTINGS_FILE
    with open(path, encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    settings["_settings_file"] = str(path)
    return settings


def get_path(settings, key):
    """Return the full path for an entry under `paths:`.

    Nested entries use dots, for example "raw_files.comtrade_dir".
    Stops with an error if the entry is missing or still marked TODO,
    so a script never quietly reads a file that has not been named yet.
    """
    value = settings["paths"]
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"No setting paths.{key} in {settings['_settings_file']}")
        value = value[part]
    if value is None or str(value).strip().upper() == "TODO":
        raise ValueError(f"Setting paths.{key} is still TODO; fill it in settings.yaml first")
    return ROOT / value


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def start_log(script_name, settings, log_dir=None):
    """Start a log that writes to logs/<script_name>.log and to the screen.

    The file is replaced on each run, so a rerun leaves only its own log.
    """
    log_dir = Path(log_dir) if log_dir else get_path(settings, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}.log"

    log = logging.getLogger(script_name)
    log.setLevel(logging.INFO)
    log.propagate = False
    # Remove handlers from an earlier run in the same Python session (run_all.py).
    for handler in list(log.handlers):
        handler.close()
        log.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    screen_handler = logging.StreamHandler(sys.stdout)
    screen_handler.setFormatter(fmt)
    log.addHandler(file_handler)
    log.addHandler(screen_handler)

    log.info("Run of %s started %s", script_name, datetime.now().isoformat(timespec="seconds"))
    log.info("Settings file: %s", settings["_settings_file"])
    log.info("Python %s on %s", platform.python_version(), platform.platform())
    return log


def log_filter(log, step, kept, removed, reason):
    """Write one standard line recording what a filter kept and removed.

    Example: "Task 2.3 | aggregate partners | kept 15,120 | removed 829 | reason: not a country"
    """
    log.info("%s | kept %s | removed %s | reason: %s", step, f"{kept:,}", f"{removed:,}", reason)


# ---------------------------------------------------------------------
# Great-circle distance
# ---------------------------------------------------------------------

def great_circle_km(lon1, lat1, lon2, lat2, settings):
    """Great-circle distance in km between points given in degrees (haversine).

    Longitude comes first, matching GeoPandas x/y. Accepts single numbers or
    arrays/columns of equal (or broadcastable) length. Uses the Earth radius
    from settings (project.earth_radius_km).
    """
    radius = settings["project"]["earth_radius_km"]
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    # Rounding can push a slightly above 1 for near-antipodal points.
    distance = 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(distance) if distance.ndim == 0 else distance


def distance_matrix_km(lons_a, lats_a, lons_b, lats_b, settings):
    """Distances from every point in set A (rows) to every point in set B (columns).

    Used, for example, for every event against every chokepoint.
    """
    lons_a = np.asarray(lons_a, dtype=float)[:, None]
    lats_a = np.asarray(lats_a, dtype=float)[:, None]
    lons_b = np.asarray(lons_b, dtype=float)[None, :]
    lats_b = np.asarray(lats_b, dtype=float)[None, :]
    return great_circle_km(lons_a, lats_a, lons_b, lats_b, settings)


# ---------------------------------------------------------------------
# ISO-3 country codes
# ---------------------------------------------------------------------

def clean_iso3(values):
    """Trim spaces and convert to upper case (Comtrade pads some codes, e.g. "_X ").

    Accepts a single code or a pandas Series; missing values stay missing.
    """
    if isinstance(values, pd.Series):
        return values.astype("string").str.strip().str.upper()
    if values is None or pd.isna(values):
        return values
    return str(values).strip().upper()


def is_aggregate(values, settings):
    """True where a code is not a country (World, regions, bunkers, free zones...).

    The list of codes is trade.aggregate_codes in settings.
    """
    codes = set(settings["trade"]["aggregate_codes"])
    cleaned = clean_iso3(values)
    if isinstance(cleaned, pd.Series):
        return cleaned.isin(codes)
    return cleaned in codes


def remap_iso3(values, settings):
    """Clean codes and replace those listed in trade.iso3_remap (e.g. S19 -> TWN).

    The calling script logs how many rows and how much value were remapped.
    """
    remap = settings["trade"].get("iso3_remap") or {}
    cleaned = clean_iso3(values)
    if isinstance(cleaned, pd.Series):
        return cleaned.replace(remap)
    return remap.get(cleaned, cleaned)


# ---------------------------------------------------------------------
# GeoPackage layers
# ---------------------------------------------------------------------

def layer_name(settings, key):
    """Return the GeoPackage layer name for a key under `layers:` in settings."""
    layers = settings["layers"]
    if key not in layers:
        raise KeyError(f"Unknown layer '{key}'; add it under layers: in settings.yaml")
    return layers[key]


def _database(settings, database):
    """Use the given database path, or the project database from settings."""
    return Path(database) if database else get_path(settings, "database")


def list_layers(settings, database=None):
    """Return the names of the layers in the database (empty if it does not exist)."""
    path = _database(settings, database)
    if not path.exists():
        return []
    return [name for name, _geom_type in pyogrio.list_layers(path)]


def read_layer(settings, key, database=None):
    """Read a layer by its settings key; returns a GeoDataFrame or a plain table."""
    path = _database(settings, database)
    name = layer_name(settings, key)
    if name not in list_layers(settings, path):
        raise ValueError(f"Layer '{name}' is not in {path}")
    return pyogrio.read_dataframe(path, layer=name)


def write_layer(df, settings, key, log=None, database=None):
    """Write (replace) one layer, leaving the other layers in the database untouched.

    Spatial layers must be in the storage CRS (project.storage_crs, EPSG:4326).
    Plain tables without geometry are written as attribute-only layers.
    """
    path = _database(settings, database)
    name = layer_name(settings, key)
    if isinstance(df, gpd.GeoDataFrame) and df.geometry.name in df.columns:
        storage_crs = settings["project"]["storage_crs"]
        if df.crs is None or not df.crs.equals(storage_crs):
            raise ValueError(f"Layer '{name}' has CRS {df.crs}; reproject to {storage_crs} before writing")
    path.parent.mkdir(parents=True, exist_ok=True)
    pyogrio.write_dataframe(df, path, layer=name, driver="GPKG")
    if log:
        log.info("Wrote layer '%s' (%s rows) to %s", name, f"{len(df):,}", path)


# ---------------------------------------------------------------------
# Events to count (Decision 9)
# ---------------------------------------------------------------------

GISIS_SOURCE = "IMO GISIS"              # the source label S05 gives IMO GISIS incidents
COUNT_RULES = ("crew_harm", "all")      # allowed values of events.imo_gisis.count_rule


def countable_events(events, settings, log=None, count_rule=None):
    """Return the events to count after the IMO GISIS counting rule (Decision 9).

    "crew_harm" (the main run) drops IMO GISIS incidents where crew were not threatened,
    injured or held (crew_harm is not 1); "all" keeps every incident (S10 alternative).
    Events from other sources are always kept. count_rule defaults to
    events.imo_gisis.count_rule; S10 passes the alternative here instead.
    """
    rule = count_rule or settings["events"]["imo_gisis"]["count_rule"]
    if rule not in COUNT_RULES:
        raise ValueError(f"Unknown IMO GISIS count_rule '{rule}'; use one of {COUNT_RULES}")
    gisis = events["source"] == GISIS_SOURCE
    if rule == "crew_harm":
        drop = gisis & (events["crew_harm"] != 1)
        reason = "crew not threatened, injured or held (crew_harm)"
    else:
        drop = pd.Series(False, index=events.index)
        reason = "none (count_rule all keeps every incident)"
    if log:
        log_filter(log, f"Decision 9 | IMO GISIS count_rule {rule}", int((gisis & ~drop).sum()),
                   int(drop.sum()), reason)
    return events[~drop].reset_index(drop=True)


# ---------------------------------------------------------------------
# Chokepoint crossing geometry
# ---------------------------------------------------------------------

def crossing_geometries(zones, settings, database=None):
    """The shape a route must touch to cross each chokepoint (Decision 12).

    A chokepoint with a gate line (the gates layer written by S03) is crossed when a route
    crosses the gate; every other chokepoint is crossed when a route intersects its zone.
    Used by S06 (crossings, task 4.3) and S08 (closures). `zones` holds the chokepoints to
    test (chokepoint_id, name, geometry). Returns chokepoint_id, name, crossing_method
    ("gate" or "zone") and geometry, one row per zone.
    """
    gates = read_layer(settings, "gates", database)
    unknown = sorted(set(gates["chokepoint_id"]) - set(zones["chokepoint_id"]))
    if unknown:
        raise ValueError(f"Gates for chokepoints that are not in the zones given: {unknown}")
    gate_by_id = gates.set_index("chokepoint_id").geometry
    shapes = zones[["chokepoint_id", "name"]].copy()
    has_gate = shapes["chokepoint_id"].isin(gate_by_id.index)
    shapes["crossing_method"] = np.where(has_gate, "gate", "zone")
    geometry = [gate_by_id[cid] if gate else zone
                for cid, gate, zone in zip(shapes["chokepoint_id"], has_gate, zones.geometry)]
    return gpd.GeoDataFrame(shapes, geometry=geometry, crs=zones.crs)


# ---------------------------------------------------------------------
# Sea routes (S06 task 4.2, S08 task 4.13)
# ---------------------------------------------------------------------

KM_PER_NM = 1.852          # definition of the nautical mile


def route_line(origin, destination, restrictions, network=None):
    """searoute coordinates between two network nodes, or None when no route exists.

    `network` is a searoute Marnet to route on (searoute's M=); None uses searoute's
    own network. S08 passes an edited copy with a chokepoint's edges removed. Returns
    (coords, traversed passages). The coordinates are continuous across 180 degrees
    (longitudes may pass +-180).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # searoute warns when restrictions leave no path
        try:
            feature = searoute.searoute(list(origin), list(destination), units="naut",
                                        restrictions=restrictions, return_passages=True, M=network)
        except nx.NetworkXNoPath:
            # Removing edges can cut a node off; searoute then raises instead of returning inf.
            return None, []
    coords = [tuple(map(float, c)) for c in feature["geometry"]["coordinates"]]
    if len(coords) < 2 or not np.isfinite(feature["properties"]["length"]):
        return None, []
    return coords, feature["properties"]["traversed_passages"]


def length_nm(coords, settings):
    """Route length in nautical miles, summed great-circle distance between vertices."""
    xy = np.asarray(coords)
    km = great_circle_km(xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1], settings)
    return float(np.sum(km)) / KM_PER_NM


def split_antimeridian(coords):
    """Split a continuous line at +-180 and shift every part into -180..180.

    Each vertex belongs to the 360-degree "copy" k = floor((lon + 180) / 360). Where two
    vertices fall in different copies, the segment is cut at the boundary between them
    (latitude interpolated linearly) and a new part starts on the other side. A vertex
    lying exactly on a boundary (the network has nodes on 180 and -180) joins the copy of
    its neighbour, so a route that only touches 180 degrees is not split.
    """
    copies = [int(np.floor((x + 180.0) / 360.0)) for x, _ in coords]
    on_boundary = [(x + 180.0) % 360.0 == 0 for x, _ in coords]
    for order in (range(1, len(coords)), range(len(coords) - 2, -1, -1)):
        for i in order:
            neighbour = copies[i - 1] if order.step == 1 else copies[i + 1]
            if on_boundary[i] and neighbour in (copies[i], copies[i] - 1):
                copies[i] = neighbour

    parts, current = [], [coords[0]]
    for i in range(1, len(coords)):
        (x0, y0), (x1, y1) = coords[i - 1], coords[i]
        k0, k1 = copies[i - 1], copies[i]
        while k0 != k1:
            step = 1 if k1 > k0 else -1
            boundary = -180.0 + 360.0 * (k0 + (1 if step > 0 else 0))
            y_cut = y0 if x1 == x0 else y0 + (y1 - y0) * (boundary - x0) / (x1 - x0)
            current.append((boundary, y_cut))
            parts.append((current, k0))
            current = [(boundary, y_cut)]
            k0 += step
        current.append((x1, y1))
    parts.append((current, copies[-1]))

    lines = []
    for part, k in parts:
        shifted = [(x - 360.0 * k, y) for x, y in part]
        shifted = [p for j, p in enumerate(shifted) if j == 0 or p != shifted[j - 1]]
        if len(shifted) >= 2:
            lines.append(LineString(shifted))
    return MultiLineString(lines)

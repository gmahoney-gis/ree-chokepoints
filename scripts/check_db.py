"""check_db: integrity check of the project database (task 3.2; layer names, task 3.1).

Reads data/project.gpkg and never changes it. Each check writes one line to
logs/check_db.log:

    PASS | group | check | counts
    WARN | group | check | counts      (expected gap or known, already-logged case)
    FAIL | group | check | counts      (a real problem; the script stops with an error)
    INFO | group | check | counts      (counts for the record)

Two stages (run_all.py runs both):
    --stage inputs   straight after S05: the S01-S05 layers (check_db.inputs_layers)
    --stage all      at the end of the run: every layer under layers:

Groups:
    A layers       every layer is present with rows (task 3.1); expected gaps from settings
    B crs          EPSG:4326, geometry types, no empty/invalid shapes, coordinates in range
    C unique       IDs unique in every layer
    D fields       required fields present and filled; no "<NA>"/"nan" text or blank strings
    E codes        ISO-3 codes, HS codes, main run (Decision 1), shares and values
    F ports        a usable port or gateway for every seaborne trading country (task 3.2)
    G chokepoints  screening file, one zone per included chokepoint, gates (Decision 12),
                   zone radii (Decision 11), true-distance circles
    J events       event window (Decision 7), sources, IMO GISIS fields (Decision 9),
                   ACLED filters, prefilter distance
    H joins        (all) sub-flows, routes, crossings, detours and costs link up
    I prefixes     (all) chokepoint_scores columns follow the prefix rule
    K values       (all) shares, route and detour lengths, null rules (Decision 10)

Licensed data: events are reported as counts only. IDs are listed (at most
check_db.max_ids_listed) only for project-made keys: flows, sub-flows, ports,
chokepoints and routes.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

import common

LEVELS = ("PASS", "WARN", "FAIL", "INFO")

# Expected geometry type of each layer (None = table without geometry).
GEOMETRY_TYPES = {
    "trade_flows": None, "chokepoints": "Point", "zones": "Polygon", "gates": "LineString",
    "ports": "Point", "country_port_shares": None, "events": "Point",
    "gdelt_country_context": None, "subflows": None, "routes": "MultiLineString",
    "route_crossings": None, "detours": None, "detour_routes": "MultiLineString",
    "reroute_costs": None, "chokepoint_scores": None,
}

# Fields that make each row unique.
UNIQUE_KEYS = {
    "trade_flows": ["flow_id"], "chokepoints": ["chokepoint_id"], "zones": ["chokepoint_id"],
    "gates": ["chokepoint_id"], "ports": ["port_id"], "events": ["event_id"],
    "subflows": ["subflow_id"], "routes": ["route_id"],
    "route_crossings": ["flow_id", "scenario", "chokepoint_id"],
    "detours": ["subflow_id", "chokepoint_id"], "detour_routes": ["route_id"],
    "reroute_costs": ["subflow_id", "chokepoint_id"], "chokepoint_scores": ["chokepoint_id"],
}

# Table 3 key fields as stored: they must exist and be filled in every row.
REQUIRED_FIELDS = {
    "trade_flows": ["flow_id", "origin_iso3", "dest_iso3", "hs_code", "in_main_run", "value_usd",
                    "mode_source", "maritime_share", "maritime_value_usd"],
    "chokepoints": ["chokepoint_id", "name", "included", "reason", "zone_radius_km"],
    "zones": ["chokepoint_id", "name", "zone_radius_km", "included", "reason"],
    "gates": ["chokepoint_id", "name", "source"],
    "ports": ["port_id", "name", "country_iso3", "harbor_size", "is_gateway", "snap_dist_km",
              "snap_flag", "use_for_routing"],
    "events": ["event_id", "source", "category", "event_type", "event_date"],
    "subflows": ["subflow_id", "flow_id", "origin_iso3", "dest_iso3", "origin_port", "dest_port",
                 "share", "maritime_value_usd", "same_port"],
    "routes": ["route_id", "origin_port", "dest_port", "scenario", "length_nm"],
    "route_crossings": ["flow_id", "scenario", "chokepoint_id"],
    "detours": ["subflow_id", "flow_id", "chokepoint_id", "origin_port", "dest_port",
                "old_length_nm", "no_route", "port_in_zone"],
    "detour_routes": ["route_id", "origin_port", "dest_port", "chokepoint_id", "old_length_nm",
                      "new_length_nm"],
    "reroute_costs": ["subflow_id", "flow_id", "chokepoint_id", "no_route", "port_in_zone"],
    "chokepoint_scores": ["chokepoint_id", "name"],
}

# Text that means "missing" written as a value (for example by astype(str) on a null).
NA_TEXT = ["<NA>", "nan", "NaN", "None", "NaT"]

ACLED_SOURCE = "ACLED"                     # the source label S05 gives ACLED events
KNOWN_SOURCES = [ACLED_SOURCE, common.GISIS_SOURCE, "GDELT"]
CATEGORIES = ["violence", "tension"]


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

class Report:
    """Collects the result of every check and writes one log line per check."""

    def __init__(self, log, max_ids):
        self.log = log
        self.max_ids = max_ids
        self.counts = dict.fromkeys(LEVELS, 0)

    def add(self, level, group, check, detail=""):
        """Record one result at the given level and log it."""
        self.counts[level] += 1
        line = f"{level} | {group} | {check}" + (f" | {detail}" if detail else "")
        write = {"WARN": self.log.warning, "FAIL": self.log.error}.get(level, self.log.info)
        write(line)

    def test(self, group, check, ok, detail="", ids=None, level_if_not_ok="FAIL"):
        """PASS if ok, otherwise level_if_not_ok. `ids` (project-made keys only) are listed on failure."""
        if not ok and ids is not None and len(ids):
            detail = f"{detail} | {self.list_ids(ids)}" if detail else self.list_ids(ids)
        self.add("PASS" if ok else level_if_not_ok, group, check, detail)

    def list_ids(self, ids):
        """At most max_ids IDs, sorted, with how many more there are."""
        ids = sorted({str(i) for i in ids})
        shown = ", ".join(ids[:self.max_ids])
        more = len(ids) - self.max_ids
        return f"IDs: {shown}" + (f" ... and {more} more" if more > 0 else "")


def as_bool(series):
    """A stored true/false field as plain booleans (nulls count as false)."""
    return series.fillna(False).astype(bool)


def usd(value):
    """Format a dollar value for the log."""
    return f"{value:,.0f} USD"


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------

def stage_keys(stage, settings):
    """Settings keys of the layers checked in this stage."""
    if stage == "inputs":
        return list(settings["check_db"]["inputs_layers"])
    return list(settings["layers"])


def read_layers(keys, present, settings, database, log):
    """Read every layer of this stage that is in the database: {settings key: table}."""
    layers = {}
    for key in keys:
        if common.layer_name(settings, key) in present:
            layers[key] = common.read_layer(settings, key, database)
    log.info("Read %d of %d layers for this stage: %s", len(layers), len(keys), ", ".join(layers))
    return layers


# ---------------------------------------------------------------------
# A. Layers present (task 3.1)
# ---------------------------------------------------------------------

def check_layers(keys, present, layers, settings, report):
    """Task 3.1: every layer of the stage is in the database under its Table 3 name."""
    expected = settings["check_db"]["expected_missing_layers"] or {}
    for key in keys:
        name = common.layer_name(settings, key)
        if key in layers:
            rows = len(layers[key])
            report.test("A layers", f"{name} present with rows", rows > 0, f"{rows:,} rows")
            if key in expected:
                report.add("WARN", "A layers", f"{name} is listed in check_db.expected_missing_layers "
                           "but is now built", "remove it from the list")
        elif key in expected:
            report.add("WARN", "A layers", f"{name} not built yet", f"expected gap: {expected[key]}")
        else:
            report.add("FAIL", "A layers", f"{name} missing", "not in check_db.expected_missing_layers")
    extra = sorted(set(present) - set(settings["layers"].values()))
    report.test("A layers", "no layers outside layers: in settings", not extra,
                f"{len(extra)} extra: {', '.join(extra)}" if extra else "")


# ---------------------------------------------------------------------
# B. CRS and geometry
# ---------------------------------------------------------------------

def check_geometry(layers, settings, database, report):
    """Every spatial layer is in the storage CRS, of the expected type, with sound shapes."""
    storage_crs = settings["project"]["storage_crs"]
    for key, df in layers.items():
        name = common.layer_name(settings, key)
        info = pyogrio.read_info(database, layer=name)
        expected = GEOMETRY_TYPES.get(key, "unknown")
        got = info["geometry_type"]
        if expected == "unknown":
            report.add("INFO", "B crs", f"{name} geometry type", f"{got} (no expected type recorded)")
        else:
            report.test("B crs", f"{name} geometry type {expected or 'none (table)'}",
                        got == expected, f"found {got or 'none'}")
        if got is None:
            continue
        report.test("B crs", f"{name} CRS {storage_crs}", info["crs"] == storage_crs, f"found {info['crs']}")
        empty = int((df.geometry.isna() | df.geometry.is_empty).sum())
        invalid = int((~df.geometry.is_valid).sum()) - empty
        b = df.geometry.bounds
        outside = int(((b.minx < -180) | (b.maxx > 180) | (b.miny < -90) | (b.maxy > 90)).sum())
        report.test("B crs", f"{name} shapes filled, valid, inside +-180 / +-90",
                    empty == 0 and invalid == 0 and outside == 0,
                    f"empty {empty}, invalid {invalid}, out of range {outside}")


# ---------------------------------------------------------------------
# C. Unique IDs, D. required fields
# ---------------------------------------------------------------------

def check_unique(layers, report):
    """Each layer's key fields identify one row."""
    for key, cols in UNIQUE_KEYS.items():
        if key not in layers:
            continue
        dup = layers[key].duplicated(cols, keep=False)
        ids = None if key == "events" or len(cols) > 1 else layers[key].loc[dup, cols[0]]
        report.test("C unique", f"{key} unique on {' + '.join(cols)}", not dup.any(),
                    f"{int(layers[key].duplicated(cols).sum())} duplicates", ids)


def check_fields(layers, report):
    """Required fields exist and are filled; no missing-value text or blank strings anywhere."""
    for key, cols in REQUIRED_FIELDS.items():
        if key not in layers:
            continue
        df = layers[key]
        missing = [c for c in cols if c not in df.columns]
        empty = {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}
        detail = "; ".join(filter(None, [f"missing fields {missing}" if missing else "",
                                         f"empty values {empty}" if empty else ""]))
        report.test("D fields", f"{key} required fields present and filled", not missing and not empty, detail)
    for key, df in layers.items():
        text = df.drop(columns="geometry", errors="ignore").select_dtypes(include=["object", "string"])
        bad = {}
        for col in text.columns:
            values = text[col].dropna().astype(str).str.strip()
            n = int(values.isin(NA_TEXT).sum() + (values == "").sum())
            if n:
                bad[col] = n
        report.test("D fields", f"{key} no '<NA>'/'nan' text or blank strings", not bad,
                    f"values per field {bad}" if bad else "")


# ---------------------------------------------------------------------
# E. Codes and values
# ---------------------------------------------------------------------

def check_codes(layers, settings, report):
    """ISO-3 and HS codes, the main-run flag (Decision 1), shares and values."""
    iso3 = re.compile(settings["trade"]["iso3_pattern"])
    for key, cols in {"trade_flows": ["origin_iso3", "dest_iso3"], "ports": ["country_iso3"],
                      "subflows": ["origin_iso3", "dest_iso3"]}.items():
        if key not in layers:
            continue
        for col in cols:
            bad = ~layers[key][col].astype(str).str.match(iso3)
            report.test("E codes", f"{key}.{col} is an ISO-3 code", not bad.any(),
                        f"{int(bad.sum())} not ISO-3", layers[key].loc[bad, col])
    if "trade_flows" not in layers:
        return
    flows = layers["trade_flows"]
    main_codes = settings["trade"]["hs_codes_main"]
    all_codes = set(main_codes) | set(settings["trade"]["hs_codes_sensitivity"])
    bad = ~flows["hs_code"].isin(all_codes)
    report.test("E codes", "trade_flows.hs_code in the settings HS codes", not bad.any(),
                f"{int(bad.sum())} other codes", flows.loc[bad, "flow_id"])
    wrong = as_bool(flows["in_main_run"]) != flows["hs_code"].isin(main_codes)
    report.test("E codes", "in_main_run true only for the main HS codes (Decision 1)", not wrong.any(),
                f"{int(wrong.sum())} wrong", flows.loc[wrong, "flow_id"])
    same = flows["origin_iso3"] == flows["dest_iso3"]
    report.test("E codes", "no flow from a country to itself", not same.any(),
                f"{int(same.sum())} flows", flows.loc[same, "flow_id"])
    bad = (~flows["maritime_share"].between(0, 1) | (flows["value_usd"] < 0)
           | (flows["maritime_value_usd"] < 0))
    report.test("E codes", "maritime_share between 0 and 1, values not negative", not bad.any(),
                f"{int(bad.sum())} flows", flows.loc[bad, "flow_id"])
    if "subflows" in layers:
        main_ids = flows.loc[as_bool(flows["in_main_run"]), "flow_id"]
        bad = ~layers["subflows"]["flow_id"].isin(main_ids)
        report.test("E codes", "sub-flows only from main-run flows (Decision 1)", not bad.any(),
                    f"{int(bad.sum())} sub-flows", layers["subflows"].loc[bad, "subflow_id"])


# ---------------------------------------------------------------------
# F. Ports for every trading country (task 3.2)
# ---------------------------------------------------------------------

def check_ports(layers, settings, report):
    """Every country with seaborne main-run trade has a usable port or a landlocked gateway."""
    if not {"trade_flows", "ports"} <= set(layers):
        return
    flows, ports = layers["trade_flows"], layers["ports"]
    sea = flows[as_bool(flows["in_main_run"]) & (flows["maritime_value_usd"] > 0)]
    traders = set(sea["origin_iso3"]) | set(sea["dest_iso3"])
    usable = set(ports.loc[as_bool(ports["use_for_routing"]), "country_iso3"])
    no_port = sorted(traders - usable)
    report.test("F ports", f"every seaborne trading country ({len(traders)}) has a usable port or gateway",
                not no_port, f"{len(no_port)} without", no_port)

    gateway_file = pd.read_csv(common.ROOT / settings["ports"]["landlocked_gateways_file"])
    listed = set(gateway_file["country_iso3"]) & traders
    has_gateway = set(ports.loc[as_bool(ports["is_gateway"]), "country_iso3"])
    report.test("F ports", f"every landlocked trader in the gateways file ({len(listed)}) has its gateway port",
                listed <= has_gateway, f"{len(listed - has_gateway)} without", listed - has_gateway)

    flagged = int(as_bool(ports["snap_flag"]).sum())
    off = int((~as_bool(ports["use_for_routing"])).sum())
    report.add("INFO", "F ports", "ports far from the sea-lane network (logged by S04)",
               f"{flagged} farther than {settings['ports']['snap_warning_km']} km, "
               f"{off} set use_for_routing = false")


# ---------------------------------------------------------------------
# G. Chokepoints, zones and gates
# ---------------------------------------------------------------------

def included_ids(chokepoints):
    """IDs of the chokepoints with included = yes."""
    return set(chokepoints.loc[chokepoints["included"].astype(str).str.lower() == "yes", "chokepoint_id"])


def check_chokepoints(layers, settings, report):
    """Screening file, one zone per included chokepoint, gate lines and zone radii."""
    if not {"chokepoints", "zones", "gates"} <= set(layers):
        return
    points, zones, gates = layers["chokepoints"], layers["zones"], layers["gates"]
    screening = pd.read_csv(common.ROOT / settings["chokepoints"]["screening_file"])
    diff = set(points["chokepoint_id"]) ^ set(screening["chokepoint_id"])
    report.test("G chokepoints", f"chokepoint IDs match the screening file ({len(screening)})", not diff,
                f"{len(diff)} differ", diff)

    included = included_ids(points)
    no_zone = included - set(zones["chokepoint_id"])
    extra = set(zones["chokepoint_id"]) - included
    report.test("G chokepoints", f"every included chokepoint ({len(included)}) has a zone, "
                "no zone for an excluded one", not no_zone and not extra,
                f"without zone {len(no_zone)}, extra zones {len(extra)}", no_zone | extra)

    gate_file = pd.read_csv(common.ROOT / settings["chokepoints"]["gates_file"])
    diff = set(gates["chokepoint_id"]) ^ set(gate_file["chokepoint_id"])
    report.test("G chokepoints", f"gate lines match the gates file ({len(gate_file)}; Decision 12)",
                not diff, f"{len(diff)} differ", diff)
    report.test("G chokepoints", "gate lines only on included chokepoints",
                set(gates["chokepoint_id"]) <= included, "", set(gates["chokepoint_id"]) - included)

    # Decision 11: the default radius, or the override named in settings.
    default = settings["chokepoints"]["zone_radius_km"]
    overrides = settings["chokepoints"]["zone_radius_overrides_km"] or {}
    expected = zones["name"].map(overrides).fillna(default)
    wrong = zones["zone_radius_km"] != expected
    report.test("G chokepoints", "zone radius = settings (default or override; Decision 11)",
                not wrong.any(), f"{int(wrong.sum())} differ", zones.loc[wrong, "chokepoint_id"])

    centre = points.set_index("chokepoint_id").geometry
    outside = [cid for cid, zone in zip(zones["chokepoint_id"], zones.geometry)
               if not zone.contains(centre[cid])]
    report.test("G chokepoints", "each zone contains its chokepoint", not outside,
                f"{len(outside)} do not", outside)

    # True-distance circles: every vertex lies one radius from the centre along the Earth's surface.
    tolerance = settings["check_db"]["zone_radius_tolerance"]
    off = []
    worst = 0.0
    for cid, radius, zone in zip(zones["chokepoint_id"], zones["zone_radius_km"], zones.geometry):
        lon, lat = np.asarray(zone.exterior.coords).T
        km = common.great_circle_km(centre[cid].x, centre[cid].y, lon, lat, settings)
        error = float(np.abs(km - radius).max() / radius)
        worst = max(worst, error)
        if error > tolerance:
            off.append(cid)
    report.test("G chokepoints", f"zones are true-distance circles (within {tolerance:.1%} of the radius)",
                not off, f"largest difference {worst:.4%}", off)


# ---------------------------------------------------------------------
# J. Events (counts only: licensed data)
# ---------------------------------------------------------------------

def check_events(layers, settings, report):
    """Event window, sources, the IMO GISIS and ACLED fields, and the prefilter distance."""
    if "events" not in layers:
        return
    events = layers["events"]
    events_cfg = settings["events"]

    dates = pd.to_datetime(events["event_date"])
    start, end = pd.Timestamp(events_cfg["window_start"]), pd.Timestamp(events_cfg["window_end"])
    outside = ~dates.between(start, end)
    report.test("J events", f"event dates inside {start.date()} to {end.date()} (Decision 7)",
                not outside.any(), f"{dates.min().date()} to {dates.max().date()}, outside {int(outside.sum())}")

    counts = events["source"].value_counts().to_dict()
    report.test("J events", "sources are known", set(counts) <= set(KNOWN_SOURCES), str(counts))
    counts = events["category"].value_counts().to_dict()
    report.test("J events", "categories are violence or tension", set(counts) <= set(CATEGORIES), str(counts))

    acled = events[events["source"] == ACLED_SOURCE]
    gisis = events[events["source"] == common.GISIS_SOURCE]
    ok = acled["imo_match"].notna().all() and acled["crew_harm"].isna().all()
    report.test("J events", "ACLED rows have imo_match and no crew_harm (Decision 9)", ok,
                f"{len(acled):,} rows, imo_match true {int(as_bool(acled['imo_match']).sum())}")
    ok = gisis["crew_harm"].notna().all() and gisis["imo_match"].isna().all()
    report.test("J events", "IMO GISIS rows have crew_harm and no imo_match (Decision 9)", ok,
                f"{len(gisis):,} rows, crew_harm true {int(as_bool(gisis['crew_harm']).sum())}")

    bad = ~acled["event_type"].isin(events_cfg["acled"]["event_types"])
    report.test("J events", "ACLED event types as in settings", not bad.any(), f"{int(bad.sum())} other")
    bad = acled["geo_precision"].isin(events_cfg["acled"]["exclude_geo_precision"])
    report.test("J events", "no ACLED event with an excluded geo precision", not bad.any(), f"{int(bad.sum())} found")
    bad = acled["fatalities"].isna() | (acled["fatalities"] < 0)
    report.test("J events", "ACLED fatalities filled and not negative", not bad.any(), f"{int(bad.sum())} rows")

    # The overrides are applied before the prefilter and the ACLED merge, so some may be gone.
    overrides = set(events_cfg["imo_gisis"]["crew_harm_overrides"] or {})
    found = len(overrides & set(gisis["event_id"]))
    report.add("INFO", "J events", "crew_harm overrides found in the layer",
               f"{found} of {len(overrides)} (the rest were dropped by the prefilter or merged into ACLED)")

    # S05 keeps events within the prefilter radius of any candidate chokepoint (task 2.18),
    # included or not, so the test uses every chokepoint in the layer.
    if "chokepoints" in layers:
        points = layers["chokepoints"]
        km = common.distance_matrix_km(events.geometry.x, events.geometry.y,
                                       points.geometry.x, points.geometry.y, settings).min(axis=1)
        radius = events_cfg["prefilter_radius_km"]
        far = int((km > radius).sum())
        report.test("J events", f"every event within {radius} km of a candidate chokepoint ({len(points)})",
                    far == 0, f"{far} farther")


# ---------------------------------------------------------------------
# H. Joins (stage all)
# ---------------------------------------------------------------------

def check_routes(layers, settings, report):
    """Sub-flows, ports and routes link up; every routed sub-flow has a route in each scenario."""
    if not {"trade_flows", "ports", "subflows", "routes"} <= set(layers):
        return
    flows, ports, subflows, routes = (layers[k] for k in ("trade_flows", "ports", "subflows", "routes"))
    bad = ~subflows["flow_id"].isin(flows["flow_id"])
    report.test("H joins", "subflows.flow_id in trade_flows", not bad.any(), f"{int(bad.sum())} not found",
                subflows.loc[bad, "subflow_id"])
    port_ids = set(ports["port_id"])
    for key, df, id_col in (("subflows", subflows, "subflow_id"), ("routes", routes, "route_id")):
        bad = ~df["origin_port"].isin(port_ids) | ~df["dest_port"].isin(port_ids)
        report.test("H joins", f"{key} ports in the ports layer", not bad.any(), f"{int(bad.sum())} not found",
                    df.loc[bad, id_col])

    scenarios = settings["routing"]["scenarios"]
    other = set(routes["scenario"]) - set(scenarios)
    report.test("H joins", "routes.scenario in the settings scenarios", not other, f"other: {sorted(other)}")

    same = as_bool(subflows["same_port"])
    report.add("INFO", "H joins", "sub-flows left unrouted (both ports on one network node; logged by S06)",
               f"{int(same.sum())} sub-flows, {usd(subflows.loc[same, 'maritime_value_usd'].sum())}")
    routed = subflows[~same]
    pair = routed["origin_port"] + "__" + routed["dest_port"]
    for scenario, spec in scenarios.items():
        here = routes[routes["scenario"] == scenario]
        have = set(here["origin_port"] + "__" + here["dest_port"])
        missing = ~pair.isin(have)
        unused = ~(here["origin_port"] + "__" + here["dest_port"]).isin(set(pair))
        report.test("H joins", f"every {scenario} route is used by a sub-flow", not unused.any(),
                    f"{int(unused.sum())} unused", here.loc[unused, "route_id"])
        detail = (f"{pair[missing].nunique()} port pairs, {int(missing.sum())} sub-flows, "
                  f"{usd(routed.loc[missing, 'maritime_value_usd'].sum())}")
        if spec.get("closed"):
            # Ports that cannot be reached with this scenario's chokepoints closed (logged by S06).
            report.test("H joins", f"every routed sub-flow has a {scenario} route", not missing.any(),
                        detail + " with no sea route when the scenario's chokepoints are closed",
                        pair[missing], level_if_not_ok="WARN")
        else:
            report.test("H joins", f"every routed sub-flow has a {scenario} route", not missing.any(),
                        detail, pair[missing])


def check_crossings(layers, settings, report):
    """Crossings, detours, reroute costs, detour lines and scores point at real rows."""
    needed = {"chokepoints", "subflows", "route_crossings", "detours", "detour_routes",
              "reroute_costs", "chokepoint_scores"}
    if not needed <= set(layers):
        return
    included = included_ids(layers["chokepoints"])
    subflows, crossings = layers["subflows"], layers["route_crossings"]
    detours, lines, costs = layers["detours"], layers["detour_routes"], layers["reroute_costs"]

    bad = ~crossings["flow_id"].isin(subflows["flow_id"])
    report.test("H joins", "route_crossings.flow_id in subflows", not bad.any(), f"{int(bad.sum())} not found",
                crossings.loc[bad, "flow_id"])
    other = set(crossings["scenario"]) - set(settings["routing"]["scenarios"])
    report.test("H joins", "route_crossings.scenario in the settings scenarios", not other, f"other: {sorted(other)}")
    for key, df in (("route_crossings", crossings), ("detours", detours), ("detour_routes", lines),
                    ("reroute_costs", costs)):
        bad = set(df["chokepoint_id"]) - included
        report.test("H joins", f"{key}.chokepoint_id are included chokepoints", not bad,
                    f"{len(bad)} others", bad)

    bad = ~detours["subflow_id"].isin(subflows["subflow_id"])
    report.test("H joins", "detours.subflow_id in subflows", not bad.any(), f"{int(bad.sum())} not found",
                detours.loc[bad, "subflow_id"])
    flow_of = subflows.set_index("subflow_id")["flow_id"]
    wrong = detours["flow_id"] != detours["subflow_id"].map(flow_of)
    report.test("H joins", "detours.flow_id is the flow of its sub-flow", not wrong.any(),
                f"{int(wrong.sum())} differ", detours.loc[wrong, "subflow_id"])

    # S08 closes each chokepoint a flow crosses in the routes_from scenario: one detour per crossing.
    base = crossings[crossings["scenario"] == settings["redundancy"]["routes_from"]]
    crossed = set(base["flow_id"] + " @ " + base["chokepoint_id"])
    detoured = set(detours["flow_id"] + " @ " + detours["chokepoint_id"])
    report.test("H joins", "one detour per baseline crossing (Decision 10)", crossed == detoured,
                f"crossings without detour {len(crossed - detoured)}, detours without crossing "
                f"{len(detoured - crossed)}", crossed ^ detoured)

    keys = set(detours["subflow_id"] + " @ " + detours["chokepoint_id"])
    cost_keys = set(costs["subflow_id"] + " @ " + costs["chokepoint_id"])
    report.test("H joins", "reroute_costs has exactly one row per detour (Decision 13)", keys == cost_keys,
                f"{len(keys ^ cost_keys)} differ", keys ^ cost_keys)

    rerouted = detours[~as_bool(detours["no_route"]) & ~as_bool(detours["port_in_zone"])]
    want = set(rerouted["origin_port"] + "__" + rerouted["dest_port"] + " @ " + rerouted["chokepoint_id"])
    have = set(lines["origin_port"] + "__" + lines["dest_port"] + " @ " + lines["chokepoint_id"])
    report.test("H joins", "one detour_routes line per rerouted port pair and chokepoint", want == have,
                f"detours without line {len(want - have)}, lines without detour {len(have - want)}", want ^ have)

    scored = set(layers["chokepoint_scores"]["chokepoint_id"])
    report.test("H joins", f"chokepoint_scores has every included chokepoint ({len(included)}) and no other",
                scored == included, f"{len(scored ^ included)} differ", scored ^ included)


# ---------------------------------------------------------------------
# I. chokepoint_scores prefix rule (stage all)
# ---------------------------------------------------------------------

def check_prefixes(layers, settings, report):
    """chokepoint_scores holds chokepoint_id, name and only columns with an owned prefix."""
    if "chokepoint_scores" not in layers:
        return
    prefixes = settings["check_db"]["score_prefixes"]
    columns = [c for c in layers["chokepoint_scores"].columns if c not in ("chokepoint_id", "name", "geometry")]
    other = [c for c in columns if not c.startswith(tuple(prefixes))]
    report.test("I prefixes", "chokepoint_scores columns all start with an owned prefix", not other,
                f"{len(other)} other: {', '.join(other)}" if other else f"{len(columns)} columns")
    per_prefix = ", ".join(f"{p} ({owner}) {sum(c.startswith(p) for c in columns)}"
                           for p, owner in prefixes.items())
    report.add("INFO", "I prefixes", "columns per prefix", per_prefix)


# ---------------------------------------------------------------------
# K. Values (stage all)
# ---------------------------------------------------------------------

def check_values(layers, settings, report):
    """Sub-flow shares, route and detour lengths, and the null rules of Decision 10."""
    if "subflows" in layers:
        total = layers["subflows"].groupby("flow_id")["share"].sum()
        bad = ~np.isclose(total, 1.0)
        report.test("K values", "sub-flow shares add up to 1 for each flow", not bad.any(),
                    f"{int(bad.sum())} flows", total.index[bad])
    if "routes" in layers:
        bad = ~(layers["routes"]["length_nm"] > 0)
        report.test("K values", "route lengths greater than 0", not bad.any(), f"{int(bad.sum())} routes",
                    layers["routes"].loc[bad, "route_id"])
    if not {"detours", "gates", "chokepoint_scores"} <= set(layers):
        return
    detours = layers["detours"]
    no_route, in_zone = as_bool(detours["no_route"]), as_bool(detours["port_in_zone"])
    rerouted = detours[~no_route & ~in_zone]
    tolerance = settings["check_db"]["length_tolerance_nm"]
    shorter = rerouted["new_length_nm"] < rerouted["old_length_nm"] - tolerance
    report.test("K values", f"no detour shorter than its original route (tolerance {tolerance} nm)",
                not shorter.any(), f"{int(shorter.sum())} shorter", rerouted.loc[shorter, "subflow_id"])
    bad = detours.loc[no_route, "new_length_nm"].notna()
    report.test("K values", "no_route detours have no new length (Decision 10d)", not bad.any(),
                f"{int(no_route.sum())} no_route rows, {int(bad.sum())} with a length")
    at_gate = in_zone & detours["chokepoint_id"].isin(layers["gates"]["chokepoint_id"])
    report.test("K values", "no port_in_zone detour at a gated chokepoint (Decision 12)", not at_gate.any(),
                f"{int(at_gate.sum())} rows", detours.loc[at_gate, "subflow_id"])

    scores = layers["chokepoint_scores"]
    null = scores["redund_penalty_pct"].isna()
    no_flows = scores["redund_nflows_affected"] == 0
    report.test("K values", "redund_penalty_pct null exactly where no flow is affected (Decision 10e)",
                (null == no_flows).all(), f"{int(null.sum())} null, {int(no_flows.sum())} without affected flows",
                scores.loc[null != no_flows, "chokepoint_id"])
    if "impact_days" in scores:
        report.add("INFO", "K values", "chokepoints without a reroute cost (impact_days null; Decision 13)",
                   f"{int(scores['impact_days'].isna().sum())} of {len(scores)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    """Read the stage and the (test-only) database path."""
    parser = argparse.ArgumentParser(description="Integrity check of the project database (task 3.2).")
    parser.add_argument("--stage", choices=["inputs", "all"], default="all",
                        help="inputs: the S01-S05 layers (after S05); all: every layer (default)")
    parser.add_argument("--database", type=Path, default=None,
                        help="database to check instead of paths.database (for testing only)")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    database = args.database.resolve() if args.database else common.get_path(settings, "database")
    keys = stage_keys(args.stage, settings)
    log.info("Stage: %s (%s)", args.stage,
             "S01-S05 layers, run after S05" if args.stage == "inputs" else "every layer, run at the end")
    log.info("Database: %s", database)
    if not database.exists():
        log.error("FAIL | A layers | database exists | not found")
        sys.exit(f"check_db: database not found: {database}")

    present = common.list_layers(settings, database)
    layers = read_layers(keys, present, settings, database, log)
    report = Report(log, settings["check_db"]["max_ids_listed"])

    check_layers(keys, present, layers, settings, report)
    check_geometry(layers, settings, database, report)
    check_unique(layers, report)
    check_fields(layers, report)
    check_codes(layers, settings, report)
    check_ports(layers, settings, report)
    check_chokepoints(layers, settings, report)
    check_events(layers, settings, report)
    if args.stage == "all":
        check_routes(layers, settings, report)
        check_crossings(layers, settings, report)
        check_prefixes(layers, settings, report)
        check_values(layers, settings, report)

    c = report.counts
    log.info("Summary | stage %s | %d checks | PASS %d | WARN %d | FAIL %d | INFO %d",
             args.stage, sum(c.values()), c["PASS"], c["WARN"], c["FAIL"], c["INFO"])
    if c["FAIL"]:
        log.error("check_db FAILED: %d check(s); see the FAIL lines above", c["FAIL"])
        sys.exit(f"check_db: {c['FAIL']} check(s) failed (stage {args.stage}); see logs/check_db.log")
    log.info("check_db passed (stage %s)", args.stage)


if __name__ == "__main__":
    main()

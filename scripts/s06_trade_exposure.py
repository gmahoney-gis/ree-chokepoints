"""S06: trade exposure of each chokepoint and routing validation (tasks 4.1-4.5).

Lab 7 first pass (ports.assignment_first_pass = nearest_gateway):

1. Task 4.1: one sub-flow per main-run flow with seaborne value (share 1). The origin
   and destination ports are the pair of candidate ports (use_for_routing = true; the
   gateway for landlocked countries) whose snapped network points are closest along the
   Earth's surface. Flows whose two ports snap to the same network node are logged and
   left unrouted. -> subflows table.
2. Task 4.2: every unique port pair is routed once per scenario with searoute. Each
   scenario closes the searoute passages of its routing.scenarios chokepoints, plus
   routing.always_closed (the Arctic routes) in every scenario. Lengths are in nautical
   miles, measured along the route with common.great_circle_km. searoute returns
   routes that cross 180 degrees as one continuous line (longitudes past +-180), so each
   route is split at the antimeridian and stored as a MultiLineString inside +-180.
   -> routes layer.
3. Task 4.3: a flow crosses a chokepoint if its route intersects that chokepoint's zone
   (included chokepoints only). Crossings are found on the split routes, then checked
   again on the continuous routes against zones shifted by -360, 0 and +360 degrees; the
   log reports whether the two agree. -> route_crossings table.
4. Task 4.4: Equation 1, the value and weight of crossing flows summed for each
   chokepoint and scenario, and the number of crossing flows. S06 owns the
   chokepoint_scores columns starting with "trade_" and keeps every other column.
5. Task 4.5: each chokepoint's share of crossing flows under the validation scenario is
   compared with its share of PortWatch container + general-cargo transits (Spearman).
   -> outputs/tables/validation.csv.

Test the routing first with:  python scripts/s06_trade_exposure.py --test-pairs 5
(routes the first 5 port pairs in every scenario, logs their lengths, writes nothing).
"""

import argparse
import csv
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import searoute
from scipy.stats import spearmanr
from searoute.classes.passages import Passage
from shapely.affinity import translate
from shapely.geometry import LineString, MultiLineString, Point

import common

KM_PER_NM = 1.852          # definition of the nautical mile
# Field suffix for each scenario in chokepoint_scores.
SCENARIO_SUFFIX = {"baseline": "base", "red_sea_diversion": "redsea"}
S06_PREFIX = "trade_"      # S06 owns every chokepoint_scores column starting with this
PW_ID = "portid"           # PortWatch chokepoint ID (matches the screening file's chokepoint_id)
PW_NAME = "portname"
PW_COUNT = "vessel_count_{}"


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_flows(settings, log):
    """Task 4.1: main-run flows with seaborne value."""
    flows = common.read_layer(settings, "trade_flows")
    keep = flows["in_main_run"] & (flows["maritime_value_usd"] > 0)
    common.log_filter(log, "Task 4.1 | trade flows in main run with maritime_value_usd > 0",
                      int(keep.sum()), int((~keep).sum()), "not in main run, or no seaborne value")
    flows = flows[keep].reset_index(drop=True)
    log.info("Task 4.1 | flows %s | maritime value %s USD | maritime weight %s kg "
             "(%d flows have no weight)", f"{len(flows):,}", f"{flows['maritime_value_usd'].sum():,.0f}",
             f"{flows['maritime_weight_kg'].sum():,.0f}", int(flows["maritime_weight_kg"].isna().sum()))
    return flows


def read_ports(settings, log):
    """Task 4.1: ports S04 marked for routing (gateways carry the landlocked country's code)."""
    ports = common.read_layer(settings, "ports")
    keep = ports["use_for_routing"].astype(bool)
    common.log_filter(log, "Task 4.1 | ports with use_for_routing = true", int(keep.sum()),
                      int((~keep).sum()), "flagged by S04 (far from the network, country has another port)")
    return pd.DataFrame(ports[keep].drop(columns="geometry")).reset_index(drop=True)


def read_zones(settings, log):
    """Task 4.3: zones of the included chokepoints."""
    zones = common.read_layer(settings, "zones")
    included = zones["included"].str.strip().str.lower() == "yes"
    common.log_filter(log, "Task 4.3 | included chokepoint zones", int(included.sum()),
                      int((~included).sum()), "included is not 'yes' in the screening file")
    return zones[included][["chokepoint_id", "name", "geometry"]].reset_index(drop=True)


# ---------------------------------------------------------------------
# Task 4.1: sub-flows
# ---------------------------------------------------------------------

def closest_pair(origin_ports, dest_ports, settings):
    """The origin and destination port whose snapped network points are closest (km)."""
    distance = common.distance_matrix_km(origin_ports["snap_lon"], origin_ports["snap_lat"],
                                         dest_ports["snap_lon"], dest_ports["snap_lat"], settings)
    i, j = np.unravel_index(np.argmin(distance), distance.shape)
    return origin_ports.iloc[i], dest_ports.iloc[j], float(distance[i, j])


def build_subflows(flows, ports, settings, log):
    """Task 4.1: one sub-flow per flow (share 1) between its closest pair of ports."""
    by_country = {iso3: group for iso3, group in ports.groupby("country_iso3")}
    no_port = sorted((set(flows["origin_iso3"]) | set(flows["dest_iso3"])) - set(by_country))
    if no_port:
        raise ValueError(f"Countries with seaborne trade but no routable port: {no_port}")

    # Flows of the same country pair (different HS codes) share the same ports.
    pairs = {}
    for o, d in flows[["origin_iso3", "dest_iso3"]].drop_duplicates().itertuples(index=False):
        po, pd_, km = closest_pair(by_country[o], by_country[d], settings)
        same = (po["snap_lon"], po["snap_lat"]) == (pd_["snap_lon"], pd_["snap_lat"])
        pairs[(o, d)] = (po["port_id"], pd_["port_id"], round(km, 1), same)
    log.info("Task 4.1 | country pairs %s | candidate ports compared per pair: nearest pair "
             "of snapped network points", f"{len(pairs):,}")

    chosen = flows.apply(lambda f: pairs[(f["origin_iso3"], f["dest_iso3"])], axis=1)
    subflows = flows[["flow_id", "origin_iso3", "dest_iso3", "hs_code"]].copy()
    subflows["subflow_id"] = subflows["flow_id"] + "_1"
    subflows["origin_port"] = [c[0] for c in chosen]
    subflows["dest_port"] = [c[1] for c in chosen]
    subflows["share"] = 1.0
    subflows["maritime_value_usd"] = flows["maritime_value_usd"] * subflows["share"]
    subflows["maritime_weight_kg"] = flows["maritime_weight_kg"] * subflows["share"]
    subflows["pair_distance_km"] = [c[2] for c in chosen]
    subflows["same_port"] = [bool(c[3]) for c in chosen]

    same = subflows[subflows["same_port"]]
    log.info("Task 4.1 | sub-flows %s (one per flow, share 1)", f"{len(subflows):,}")
    common.log_filter(log, "Task 4.1 | sub-flows with different origin and destination network points",
                      int((~subflows["same_port"]).sum()), len(same),
                      f"both ports snap to the same network node; left unrouted, "
                      f"{same['maritime_value_usd'].sum():,.0f} USD")
    for r in same.itertuples():
        log.info("Task 4.1 |   unrouted %s (%s -> %s): %s USD", r.flow_id, r.origin_port,
                 r.dest_port, f"{r.maritime_value_usd:,.0f}")
    cols = ["subflow_id", "flow_id", "origin_iso3", "dest_iso3", "hs_code", "origin_port",
            "dest_port", "share", "maritime_value_usd", "maritime_weight_kg",
            "pair_distance_km", "same_port"]
    return subflows[cols]


# ---------------------------------------------------------------------
# Task 4.2: routing
# ---------------------------------------------------------------------

def scenario_restrictions(settings, log):
    """Task 4.2: searoute passage tags closed in each scenario (its own plus always_closed)."""
    routing = settings["routing"]
    mapping = routing["searoute_passages"]
    # valid_passages() also returns the class's own methods on Python 3.11; keep the tags.
    valid = {p for p in Passage.valid_passages() if isinstance(p, str)}
    log.info("Task 4.2 | searoute %s passage tags: %s", searoute_version(), ", ".join(sorted(valid)))
    unknown = sorted((set(mapping.values()) | set(routing["always_closed"])) - valid)
    if unknown:
        raise ValueError(f"routing settings name searoute passages that do not exist: {unknown}")

    restrictions = {}
    for name, scenario in routing["scenarios"].items():
        missing = [c for c in scenario["closed"] if c not in mapping]
        if missing:
            raise ValueError(f"Scenario {name} closes {missing}, which have no searoute passage in "
                             "routing.searoute_passages (use the editable network, S08)")
        tags = list(dict.fromkeys(list(routing["always_closed"]) + [mapping[c] for c in scenario["closed"]]))
        restrictions[name] = tags
        log.info("Task 4.2 | scenario %s: closed chokepoints %s -> searoute restrictions %s",
                 name, scenario["closed"] or "none", tags)
    return restrictions


def searoute_version():
    """Installed searoute version."""
    from importlib.metadata import version
    return version("searoute")


def route_line(origin, destination, restrictions):
    """Task 4.2: searoute coordinates between two network nodes, or None when no route exists.

    The coordinates are continuous across 180 degrees (longitudes may pass +-180).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # searoute warns when restrictions leave no path
        feature = searoute.searoute(list(origin), list(destination), units="naut",
                                    restrictions=restrictions, return_passages=True)
    coords = [tuple(map(float, c)) for c in feature["geometry"]["coordinates"]]
    if len(coords) < 2 or not np.isfinite(feature["properties"]["length"]):
        return None, []
    return coords, feature["properties"]["traversed_passages"]


def length_nm(coords, settings):
    """Route length in nautical miles, summed great-circle distance between vertices."""
    xy = np.asarray(coords)
    km = common.great_circle_km(xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1], settings)
    return float(np.sum(km)) / KM_PER_NM


def split_antimeridian(coords):
    """Task 4.2: split a continuous line at +-180 and shift every part into -180..180.

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


def route_pairs(pairs, ports, restrictions, settings, log):
    """Task 4.2: route each unique port pair once per scenario.

    Returns the routes (one row per pair and scenario, with the continuous line kept for
    the crossing check) and the failed pairs.
    """
    snap = ports.drop_duplicates("port_id").set_index("port_id")[["snap_lon", "snap_lat"]]
    rows, failures = [], []
    for scenario, tags in restrictions.items():
        passages = {}
        for o, d in pairs:
            coords, traversed = route_line(tuple(snap.loc[o]), tuple(snap.loc[d]), tags)
            if coords is None:
                failures.append({"origin_port": o, "dest_port": d, "scenario": scenario})
                continue
            for p in traversed:
                passages[p] = passages.get(p, 0) + 1
            geometry = split_antimeridian(coords)
            rows.append({"route_id": f"{o}__{d}__{scenario}", "origin_port": o, "dest_port": d,
                         "scenario": scenario, "length_nm": round(length_nm(coords, settings), 1),
                         "crosses_180": len(geometry.geoms) > 1,
                         "max_lat": max(c[1] for c in coords),
                         "continuous": LineString(coords), "geometry": geometry})
        n_ok = sum(r["scenario"] == scenario for r in rows)
        n_fail = sum(f["scenario"] == scenario for f in failures)
        log.info("Task 4.2 | scenario %s: port pairs %s | routed %s | failed (no sea route with "
                 "these restrictions) %s | routes crossing 180 degrees %d", scenario,
                 f"{len(pairs):,}", f"{n_ok:,}", f"{n_fail:,}",
                 sum(r["crosses_180"] for r in rows if r["scenario"] == scenario))
        log.info("Task 4.2 | scenario %s: routes through each searoute passage: %s", scenario,
                 ", ".join(f"{p} {n:,}" for p, n in sorted(passages.items())) or "none")
    routes = gpd.GeoDataFrame(rows, geometry="geometry", crs=settings["project"]["storage_crs"])
    return routes, pd.DataFrame(failures, columns=["origin_port", "dest_port", "scenario"])


def log_failures(failures, subflows, log):
    """Task 4.2: flows and value on the port pairs that could not be routed."""
    for scenario, group in failures.groupby("scenario"):
        hit = subflows.merge(group, on=["origin_port", "dest_port"])
        log.info("Task 4.2 | scenario %s: failed pairs carry %d sub-flows, %s USD", scenario,
                 len(hit), f"{hit['maritime_value_usd'].sum():,.0f}")
        for r in group.itertuples():
            on_pair = hit[(hit["origin_port"] == r.origin_port) & (hit["dest_port"] == r.dest_port)]
            log.info("Task 4.2 |   failed %s -> %s (%d sub-flows, %s USD)", r.origin_port,
                     r.dest_port, len(on_pair), f"{on_pair['maritime_value_usd'].sum():,.0f}")


def arctic_check(routes, settings, log):
    """Task 4.2: routes north of routing.arctic_check_lat (should be none through the Arctic)."""
    limit = settings["routing"]["arctic_check_lat"]
    north = routes[routes["max_lat"] > limit]
    log.info("Task 4.2 | check: routes reaching north of %.2f N: %d%s", limit, len(north),
             "".join(f"\n    {r.route_id} (max lat {r.max_lat:.1f}, {r.length_nm:,.0f} nm)"
                     for r in north.itertuples()))


def test_routing(pairs, ports, restrictions, settings, log):
    """Task 4.2 test: route the first few pairs in every scenario and log the lengths."""
    routes, failures = route_pairs(pairs, ports, restrictions, settings, log)
    for r in routes.itertuples():
        log.info("Test | %-40s %-18s %9s nm  parts %d  max lat %.1f", f"{r.origin_port} -> {r.dest_port}",
                 r.scenario, f"{r.length_nm:,.0f}", len(r.geometry.geoms), r.max_lat)
    for r in failures.itertuples():
        log.info("Test | %-40s %-18s no route", f"{r.origin_port} -> {r.dest_port}", r.scenario)


# ---------------------------------------------------------------------
# Task 4.3: crossings
# ---------------------------------------------------------------------

def route_zone_hits(routes, zones, log):
    """Task 4.3: (route_id, chokepoint_id) for every route that intersects a zone.

    Found on the split routes, then checked on the continuous routes against the zones
    copied 360 degrees east and west; the log says whether the two methods agree.
    """
    hits = gpd.sjoin(routes[["route_id", "geometry"]], zones[["chokepoint_id", "geometry"]],
                     predicate="intersects")[["route_id", "chokepoint_id"]]
    found = set(map(tuple, hits.values))

    copies = pd.concat([zones.assign(geometry=zones.geometry.apply(translate, xoff=dx))
                        for dx in (-360.0, 0.0, 360.0)], ignore_index=True)
    continuous = gpd.GeoDataFrame(routes[["route_id"]], geometry=routes["continuous"].values,
                                  crs=routes.crs)
    check = gpd.sjoin(continuous, gpd.GeoDataFrame(copies, crs=routes.crs), predicate="intersects")
    checked = set(map(tuple, check[["route_id", "chokepoint_id"]].drop_duplicates().values))
    log.info("Task 4.3 | route-zone intersections %s | check on continuous routes with zones "
             "shifted +-360: %s | %s", f"{len(found):,}", f"{len(checked):,}",
             "agree" if found == checked else
             f"DISAGREE: only split {sorted(found - checked)}; only continuous {sorted(checked - found)}")
    near_180 = routes.loc[routes["crosses_180"], "route_id"]
    log.info("Task 4.3 | routes crossing 180 degrees that cross a zone: %d of %d",
             hits.loc[hits["route_id"].isin(near_180), "route_id"].nunique(), len(near_180))
    return hits.drop_duplicates()


def endpoint_zone_check(routes, ports, zones, log):
    """Task 4.3: routes that start or end inside a zone count as crossing it; list them."""
    snap = ports.drop_duplicates("port_id").set_index("port_id")
    lon = snap["snap_lon"].where(snap["snap_lon"] <= 180, snap["snap_lon"] - 360)
    points = gpd.GeoDataFrame({"port_id": snap.index},
                              geometry=[Point(x, y) for x, y in zip(lon, snap["snap_lat"])],
                              crs=zones.crs)
    inside = gpd.sjoin(points, zones, predicate="within")
    log.info("Task 4.3 | ports whose network point lies inside a zone (their routes cross it): %d%s",
             len(inside), "".join(f"\n    {r.port_id} in {r.name}" for r in inside.itertuples()))


def build_crossings(hits, routes, subflows, log):
    """Task 4.3: one row per flow, scenario and chokepoint crossed."""
    routed = subflows[~subflows["same_port"]]
    on_route = routes[["route_id", "origin_port", "dest_port", "scenario"]].merge(
        routed[["subflow_id", "flow_id", "origin_port", "dest_port"]], on=["origin_port", "dest_port"])
    sub_cross = on_route.merge(hits, on="route_id")[["subflow_id", "flow_id", "scenario", "chokepoint_id"]]
    crossings = sub_cross[["flow_id", "scenario", "chokepoint_id"]].drop_duplicates()
    for scenario, group in crossings.groupby("scenario"):
        routed_flows = on_route.loc[on_route["scenario"] == scenario, "flow_id"].nunique()
        log.info("Task 4.3 | scenario %s: routed flows %s | crossings %s | flows crossing at "
                 "least one chokepoint %s | flows crossing none %s", scenario, f"{routed_flows:,}",
                 f"{len(group):,}", f"{group['flow_id'].nunique():,}",
                 f"{routed_flows - group['flow_id'].nunique():,}")
    return crossings.reset_index(drop=True), sub_cross


# ---------------------------------------------------------------------
# Task 4.4: Equation 1
# ---------------------------------------------------------------------

def trade_exposure(sub_cross, subflows, chokepoints, settings, log):
    """Task 4.4: Equation 1 by value and weight, and the number of crossing flows."""
    values = sub_cross.merge(subflows[["subflow_id", "maritime_value_usd", "maritime_weight_kg"]],
                             on="subflow_id")
    scores = chokepoints[["chokepoint_id", "name"]].copy()
    for scenario in settings["routing"]["scenarios"]:
        s = SCENARIO_SUFFIX[scenario]
        group = values[values["scenario"] == scenario].groupby("chokepoint_id")
        scores[f"trade_value_{s}"] = scores["chokepoint_id"].map(group["maritime_value_usd"].sum()).fillna(0.0)
        scores[f"trade_wt_{s}"] = scores["chokepoint_id"].map(group["maritime_weight_kg"].sum()).fillna(0.0)
        scores[f"trade_nflows_{s}"] = (scores["chokepoint_id"].map(group["flow_id"].nunique())
                                       .fillna(0).astype("int64"))

    main = [n for n, sc in settings["routing"]["scenarios"].items() if sc.get("main")][0]
    measure = "value" if settings["trade"]["measure_main"] == "value_usd" else "wt"
    rank_value = f"trade_{measure}_{SCENARIO_SUFFIX[main]}"
    scores[f"trade_rank_{SCENARIO_SUFFIX[main]}"] = (scores[rank_value].rank(ascending=False, method="min")
                                                     .astype("int64"))
    log.info("Task 4.4 | main ranking field: %s -> trade_rank_%s (1 = most exposed; ties share "
             "the higher rank)", rank_value, SCENARIO_SUFFIX[main])
    return scores


def log_trade_ranking(scores, log):
    """Log both scenarios side by side, sorted by the baseline value."""
    total = scores[["trade_value_base", "trade_value_redsea"]].sum()
    log.info("Task 4.4 | trade exposure by chokepoint (value USD, share of the column total, flows):")
    log.info("  %4s  %-22s %16s %7s %6s   %16s %7s %6s", "rank", "chokepoint", "baseline USD", "share",
             "flows", "red sea USD", "share", "flows")
    for r in scores.sort_values(["trade_rank_base", "name"]).itertuples():
        log.info("  %4d  %-22s %16s %6.1f%% %6d   %16s %6.1f%% %6d", r.trade_rank_base, r.name,
                 f"{r.trade_value_base:,.0f}", 100 * r.trade_value_base / total.iloc[0], r.trade_nflows_base,
                 f"{r.trade_value_redsea:,.0f}", 100 * r.trade_value_redsea / total.iloc[1],
                 r.trade_nflows_redsea)


def merge_scores(s06, settings, log):
    """Replace S06's columns in chokepoint_scores and keep every other script's columns."""
    if common.layer_name(settings, "chokepoint_scores") not in common.list_layers(settings):
        log.info("chokepoint_scores does not exist yet; creating it")
        return s06
    old = pd.DataFrame(common.read_layer(settings, "chokepoint_scores"))
    old = old.drop(columns=[c for c in old.columns if c == "geometry"])
    keep = [c for c in old.columns if c not in ("chokepoint_id", "name") and not c.startswith(S06_PREFIX)]
    log.info("chokepoint_scores exists (%d rows); replacing %d S06 columns, keeping %d others: %s",
             len(old), len(old.columns) - len(keep) - 2, len(keep), keep or "none")
    dropped = sorted(set(old["chokepoint_id"]) - set(s06["chokepoint_id"]))
    if dropped:
        log.warning("Rows for chokepoints no longer included are removed: %s", dropped)
    return s06.merge(old[["chokepoint_id"] + keep], on="chokepoint_id", how="left")


# ---------------------------------------------------------------------
# Task 4.5: validation
# ---------------------------------------------------------------------

def read_portwatch(settings, log):
    """Task 4.5: PortWatch chokepoint file as text, with a check of the field alignment."""
    path = common.get_path(settings, "raw_files.portwatch_chokepoints")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    widths = sorted({len(r) for r in rows[1:]})
    log.info("Read %s: %d rows | header fields %d | fields per row %s", path.name, len(rows) - 1,
             len(rows[0]), widths)
    if widths != [len(rows[0])]:
        raise ValueError(f"{path.name}: header has {len(rows[0])} names but rows have {widths} fields")
    pw = pd.DataFrame(rows[1:], columns=rows[0])
    count_cols = [c for c in pw.columns if c.startswith(PW_COUNT.format(""))]

    # Alignment check: vessel_count_total should equal the sum of the vessel-type counts.
    counts = pw[count_cols].astype(int)
    parts = counts.drop(columns=PW_COUNT.format("total")).sum(axis=1)
    off = pw.loc[parts != counts[PW_COUNT.format("total")], PW_NAME].tolist()
    log.info("Task 4.5 | check: vessel_count_total = sum of %s in %d of %d rows%s",
             "+".join(c.replace(PW_COUNT.format(""), "") for c in count_cols if not c.endswith("total")),
             len(pw) - len(off), len(pw), f" (differs: {off})" if off else "")

    show = pw[pw[PW_NAME].isin(["Suez Canal", "Malacca Strait", "Panama Canal"])]
    log.info("Task 4.5 | PortWatch vessel-count columns as read (text, unconverted):")
    log.info("    %s", " | ".join([PW_ID, PW_NAME] + count_cols))
    for _, r in show.iterrows():
        log.info("    %s", " | ".join(r[[PW_ID, PW_NAME] + count_cols]))
    return pw


def validate(scores, pw, settings, log):
    """Task 4.5: Spearman rho between modeled flow shares and PortWatch transit shares."""
    v = settings["routing"]["validation"]
    s = SCENARIO_SUFFIX[v["scenario"]]
    type_cols = [PW_COUNT.format(t) for t in v["vessel_types"]]
    pw = pw.set_index(PW_ID)
    missing = sorted(set(scores["chokepoint_id"]) - set(pw.index))
    if missing:
        raise ValueError(f"Chokepoints missing from the PortWatch file: {missing}")

    table = scores[["chokepoint_id", "name", f"trade_nflows_{s}", "trade_nflows_base"]].copy()
    for c in type_cols:
        table[c] = table["chokepoint_id"].map(pw[c].astype(int))
    table["pw_transits"] = table[type_cols].sum(axis=1)
    table["model_share"] = table[f"trade_nflows_{s}"] / table[f"trade_nflows_{s}"].sum()
    table["pw_share"] = table["pw_transits"] / table["pw_transits"].sum()
    table["model_rank"] = table["model_share"].rank(ascending=False, method="min").astype("int64")
    table["pw_rank"] = table["pw_share"].rank(ascending=False, method="min").astype("int64")

    rho, p = spearmanr(table["model_share"], table["pw_share"])
    passed = bool(rho >= v["pass_mark_spearman"])
    log.info("Task 4.5 | %s flow shares vs PortWatch %s shares, %d chokepoints: Spearman rho = "
             "%.3f, p = %.2g -> %s (pass mark %.2f)", v["scenario"], "+".join(v["vessel_types"]),
             len(table), rho, p, "PASS" if passed else "BELOW PASS MARK", v["pass_mark_spearman"])
    rho_b, p_b = spearmanr(table["trade_nflows_base"], table["pw_share"])
    log.info("Task 4.5 | for reference only, baseline flow shares: rho = %.3f, p = %.2g", rho_b, p_b)
    for r in table.sort_values("pw_rank").itertuples():
        log.info("    PortWatch rank %2d  model rank %2d  %-22s pw share %5.1f%%  model share %5.1f%%",
                 r.pw_rank, r.model_rank, r.name, 100 * r.pw_share, 100 * r.model_share)

    table = table.drop(columns="trade_nflows_base").rename(columns={f"trade_nflows_{s}": "model_nflows"})
    table.insert(2, "scenario", v["scenario"])
    table["model_share"] = table["model_share"].round(4)
    table["pw_share"] = table["pw_share"].round(4)
    table["spearman_rho"] = round(float(rho), 4)
    table["p_value"] = float(p)
    table["pass_mark"] = v["pass_mark_spearman"]
    table["passed"] = passed
    return table.sort_values("pw_rank")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-pairs", type=int, default=None,
                        help="route only the first N port pairs, log their lengths, write nothing")
    return parser.parse_args()


def main(test_pairs=None):
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)

    flows = read_flows(settings, log)
    ports = read_ports(settings, log)
    subflows = build_subflows(flows, ports, settings, log)
    restrictions = scenario_restrictions(settings, log)

    # Task 4.2: unique port pairs among the routed sub-flows.
    routed = subflows[~subflows["same_port"]]
    pairs = list(routed[["origin_port", "dest_port"]].drop_duplicates().itertuples(index=False, name=None))
    log.info("Task 4.2 | unique port pairs to route: %s (from %s sub-flows)", f"{len(pairs):,}",
             f"{len(routed):,}")
    if test_pairs:
        test_routing(pairs[:test_pairs], ports, restrictions, settings, log)
        log.info("Test run finished; nothing written")
        return

    routes, failures = route_pairs(pairs, ports, restrictions, settings, log)
    log_failures(failures, subflows, log)
    arctic_check(routes, settings, log)

    # Task 4.3: crossings.
    zones = read_zones(settings, log)
    hits = route_zone_hits(routes, zones, log)
    endpoint_zone_check(routes, ports, zones, log)
    crossings, sub_cross = build_crossings(hits, routes, subflows, log)

    # Task 4.4: Equation 1.
    scores = trade_exposure(sub_cross, subflows, zones, settings, log)
    log_trade_ranking(scores, log)

    # Task 4.5: validation.
    tables = common.get_path(settings, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    validation = validate(scores, read_portwatch(settings, log), settings, log)
    validation.to_csv(tables / "validation.csv", index=False)
    log.info("Wrote %s (%d rows)", tables / "validation.csv", len(validation))

    common.write_layer(subflows, settings, "subflows", log)
    common.write_layer(routes.drop(columns=["continuous", "max_lat"]), settings, "routes", log)
    common.write_layer(crossings, settings, "route_crossings", log)
    common.write_layer(merge_scores(scores, settings, log), settings, "chokepoint_scores", log)
    log.info("Run of s06_trade_exposure finished")


if __name__ == "__main__":
    main(parse_args().test_pairs)

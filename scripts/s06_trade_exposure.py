"""S06: trade exposure of each chokepoint and routing validation (tasks 4.1-4.5).

1. Task 4.1: sub-flows -> subflows table. ports.assignment chooses how (Decision 14):
   - principal_ports (main run): each main-run flow with seaborne value is split into one
     sub-flow per pair of an origin export port and a destination import port from
     country_port_shares (S04). A sub-flow's share is the origin port's export share times
     the destination port's import share, so a flow's shares sum to 1; its value and weight
     are the flow's times the share (Decision 14d). Landlocked gateways and BMU have share 1.
   - nearest_gateway (S10 alternative; the Lab 7/8 first pass): one sub-flow per flow
     (share 1) between the candidate ports (use_for_routing = true) whose snapped network
     points are closest along the Earth's surface.
   Each port's snapped network point gets an ID ("lon_lat"); a sub-flow's route_key is its
   origin and destination point IDs. Sub-flows whose two ports snap to the same network
   point (same_port = true) have no sea leg: they keep their value, get no route, cross
   no chokepoint, and are logged.
2. Task 4.2: every unique pair of network points is routed once per scenario with
   searoute, and the line is copied to every port pair that shares those points.
   routing.always_closed (the Arctic routes) is closed in every scenario. The baseline
   closes nothing else. A scenario that closes chokepoints (red_sea_diversion) reroutes
   only the routes whose baseline crosses one of them, with its closed chokepoints and its
   diverted_also_closed ones (Panama, so displaced flows go round the Cape; task 4.6
   correction); every other route is the same as in the baseline. Lengths are in nautical
   miles, measured along the route with common.great_circle_km. searoute returns
   routes that cross 180 degrees as one continuous line (longitudes past +-180), so each
   route is split at the antimeridian and stored as a MultiLineString inside +-180.
   -> routes layer, one row per port pair and scenario (origin_point, dest_point and
   route_key say which network-point route it copies).
3. Task 4.3: a route crosses a chokepoint if it intersects that chokepoint's zone
   (included chokepoints only), or its gate line where it has one (Decision 12, the four
   wide straits; common.crossing_geometries). Crossings are found on the split routes, then checked
   again on the continuous routes against zones shifted by -360, 0 and +360 degrees; the
   log reports whether the two agree. -> route_crossings table: one row per flow, scenario
   and chokepoint crossed, with the share and value of the flow's sub-flows that cross.
4. Task 4.4: Equation 1, the value and weight of crossing sub-flows summed for each
   chokepoint and scenario, and the number of flows with a crossing sub-flow. S06 owns the
   chokepoint_scores columns starting with "trade_" and keeps every other column.
5. Task 4.5: each chokepoint's share of crossing flows under the validation scenario is
   compared with its share of PortWatch container + general-cargo transits (Spearman),
   summed from the daily transit file over routing.validation.transit_period. A flow
   counts by the share of it that crosses (model_wflows; equal to the flow count when every
   flow has one sub-flow); the count of flows with any crossing sub-flow (model_nflows) is
   logged for reference. Two more variants are kept for transparency: the summary-file
   counts and the daily counts, both against the scenario with Panama left open for
   displaced flows (the first run). -> outputs/tables/validation.csv (column "variant";
   is_main marks the main result).
6. Context: PortWatch transits (all vessel types, and container only) per calendar year
   for routing.red_sea_context chokepoints. -> outputs/tables/red_sea_context.csv.
7. Comparison with the first pass (paths.first_pass_database): trade exposure per
   chokepoint old vs new with rank change, the validation rho, and the Red Sea pairs with
   no sea route. Logged only.

Test the routing first with:  python scripts/s06_trade_exposure.py --test-pairs 5
(routes the first 5 network-point pairs in every scenario, logs their lengths, writes nothing).
Test the whole run on a sample with:  python scripts/s06_trade_exposure.py --sample-flows 50
(runs every step on 50 flows, writes nothing).
"""

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from searoute.classes.passages import Passage
from shapely.affinity import translate
from shapely.geometry import LineString, Point

import common

# Field suffix for each scenario in chokepoint_scores.
SCENARIO_SUFFIX = {"baseline": "base", "red_sea_diversion": "redsea"}
S06_PREFIX = "trade_"      # S06 owns every chokepoint_scores column starting with this
PW_ID = "portid"           # PortWatch chokepoint ID (matches the screening file's chokepoint_id)
PW_NAME = "portname"
PW_COUNT = "vessel_count_{}"     # summary file: vessel counts by type
DAILY_COUNT = "n_{}"            # daily file: transits by type ("total" = all types)
# Validation variants written to validation.csv ({year} = transit_period start year).
VARIANT_SUMMARY = "summary_file_panama_open"
VARIANT_DAILY_OPEN = "daily{year}_panama_open"
VARIANT_DAILY_MAIN = "daily{year}_cape_diversion"
SAMPLE_SEED = 1            # --sample-flows: fixed random sample, so test runs repeat
SUBFLOW_COLUMNS = ["subflow_id", "flow_id", "origin_iso3", "dest_iso3", "hs_code", "origin_port",
                   "dest_port", "share", "maritime_value_usd", "maritime_weight_kg",
                   "pair_distance_km", "same_port", "origin_point", "dest_point", "route_key"]


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_flows(settings, log, sample=None):
    """Task 4.1: main-run flows with seaborne value (optionally a random sample, for tests)."""
    flows = common.read_layer(settings, "trade_flows")
    keep = flows["in_main_run"] & (flows["maritime_value_usd"] > 0)
    common.log_filter(log, "Task 4.1 | trade flows in main run with maritime_value_usd > 0",
                      int(keep.sum()), int((~keep).sum()), "not in main run, or no seaborne value")
    flows = flows[keep].reset_index(drop=True)
    if sample:
        flows = flows.sample(n=min(sample, len(flows)), random_state=SAMPLE_SEED).reset_index(drop=True)
        log.info("Test run: random sample of %d flows (seed %d); nothing is written", len(flows), SAMPLE_SEED)
    log.info("Task 4.1 | flows %s | maritime value %s USD | maritime weight %s kg "
             "(%d flows have no weight)", f"{len(flows):,}", f"{flows['maritime_value_usd'].sum():,.0f}",
             f"{flows['maritime_weight_kg'].sum():,.0f}", int(flows["maritime_weight_kg"].isna().sum()))
    return flows


def read_ports(settings, log):
    """Task 4.1: ports S04 marked for routing, each with the ID of its snapped network point.

    Gateways carry the landlocked country's code. Ports that snap to the same network node
    share a point ID, so their routes are found once.
    """
    ports = common.read_layer(settings, "ports")
    keep = ports["use_for_routing"].astype(bool)
    common.log_filter(log, "Task 4.1 | ports with use_for_routing = true", int(keep.sum()),
                      int((~keep).sum()), "not used by the port assignment (S04)")
    ports = pd.DataFrame(ports[keep].drop(columns="geometry")).reset_index(drop=True)
    ports["point_id"] = [f"{x:.5f}_{y:.5f}" for x, y in zip(ports["snap_lon"], ports["snap_lat"])]
    # The ID must stand for exactly one network point.
    per_id = ports.groupby("point_id")[["snap_lon", "snap_lat"]].nunique()
    if (per_id > 1).any().any():
        raise ValueError("Two network points round to the same point_id; use more decimals")
    log.info("Task 4.1 | ports %d on %d distinct network points", len(ports), ports["point_id"].nunique())
    return ports


def read_port_shares(settings, log):
    """Task 4.1 / Decision 14: each country's principal export and import ports with their shares."""
    shares = pd.DataFrame(common.read_layer(settings, "country_port_shares"))
    log.info("Task 4.1 | country_port_shares: %d rows | %d exporting countries | %d importing countries",
             len(shares), shares.loc[shares["side"] == "export", "country_iso3"].nunique(),
             shares.loc[shares["side"] == "import", "country_iso3"].nunique())
    return shares


def read_zones(settings, log):
    """Task 4.3: zones of the included chokepoints."""
    zones = common.read_layer(settings, "zones")
    included = zones["included"].str.strip().str.lower() == "yes"
    common.log_filter(log, "Task 4.3 | included chokepoint zones", int(included.sum()),
                      int((~included).sum()), "included is not 'yes' in the screening file")
    return zones[included][["chokepoint_id", "name", "geometry"]].reset_index(drop=True)


def read_crossing_shapes(zones, settings, log):
    """Task 4.3 / Decision 12: the gate line where a chokepoint has one, otherwise its zone."""
    shapes = common.crossing_geometries(zones, settings)
    counts = shapes["crossing_method"].value_counts()
    log.info("Task 4.3 | crossing test: gate line %d, zone %d (Decision 12)",
             counts.get("gate", 0), counts.get("zone", 0))
    for r in shapes[shapes["crossing_method"] == "gate"].itertuples():
        log.info("Task 4.3 |   gate: %s %s", r.chokepoint_id, r.name)
    return shapes


# ---------------------------------------------------------------------
# Task 4.1: sub-flows
# ---------------------------------------------------------------------

def closest_pair(origin_ports, dest_ports, settings):
    """The origin and destination port whose snapped network points are closest (km)."""
    distance = common.distance_matrix_km(origin_ports["snap_lon"], origin_ports["snap_lat"],
                                         dest_ports["snap_lon"], dest_ports["snap_lat"], settings)
    i, j = np.unravel_index(np.argmin(distance), distance.shape)
    return origin_ports.iloc[i], dest_ports.iloc[j], float(distance[i, j])


def build_nearest_subflows(flows, ports, settings, log):
    """Task 4.1, nearest_gateway: one sub-flow per flow (share 1) between its closest pair of ports."""
    by_country = {iso3: group for iso3, group in ports.groupby("country_iso3")}
    no_port = sorted((set(flows["origin_iso3"]) | set(flows["dest_iso3"])) - set(by_country))
    if no_port:
        raise ValueError(f"Countries with seaborne trade but no routable port: {no_port}")

    # Flows of the same country pair (different HS codes) share the same ports.
    pairs = {}
    for o, d in flows[["origin_iso3", "dest_iso3"]].drop_duplicates().itertuples(index=False):
        po, pd_, _ = closest_pair(by_country[o], by_country[d], settings)
        pairs[(o, d)] = (po["port_id"], pd_["port_id"])
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
    return subflows


def build_principal_subflows(flows, shares, settings, log):
    """Task 4.1, principal_ports (Decision 14d): one sub-flow per origin export port x
    destination import port, share = export share x import share."""
    side = {}
    for name, end in (("export", "origin"), ("import", "dest")):
        side[end] = (shares[shares["side"] == name][["country_iso3", "port_id", "share", "rank"]]
                     .rename(columns={"country_iso3": f"{end}_iso3", "port_id": f"{end}_port",
                                      "share": f"share_{end}", "rank": f"rank_{end}"}))
    for end, label in (("origin", "export"), ("dest", "import")):
        missing = sorted(set(flows[f"{end}_iso3"]) - set(side[end][f"{end}_iso3"]))
        if missing:
            raise ValueError(f"Countries with seaborne trade but no principal {label} ports: {missing}")

    subflows = (flows[["flow_id", "origin_iso3", "dest_iso3", "hs_code", "maritime_value_usd",
                       "maritime_weight_kg"]]
                .merge(side["origin"], on="origin_iso3").merge(side["dest"], on="dest_iso3")
                .sort_values(["flow_id", "rank_origin", "rank_dest"]).reset_index(drop=True))
    subflows["share"] = subflows["share_origin"] * subflows["share_dest"]
    subflows["subflow_id"] = (subflows["flow_id"] + "_"
                              + (subflows.groupby("flow_id").cumcount() + 1).astype(str))
    subflows["maritime_value_usd"] = subflows["maritime_value_usd"] * subflows["share"]
    subflows["maritime_weight_kg"] = subflows["maritime_weight_kg"] * subflows["share"]

    per_flow = subflows.groupby("flow_id").size()
    log.info("Task 4.1 | sub-flows %s from %s flows (origin export ports x destination import "
             "ports) | per flow: 1 sub-flow %d flows, median %d, max %d (%s)", f"{len(subflows):,}",
             f"{len(per_flow):,}", int((per_flow == 1).sum()), int(per_flow.median()),
             int(per_flow.max()), per_flow.idxmax())
    return subflows


def add_points(subflows, flows, ports, settings, log):
    """Task 4.1: network points, route key and same-point flag; checks shares and totals."""
    port = ports.drop_duplicates("port_id").set_index("port_id")
    for end in ("origin", "dest"):
        subflows[f"{end}_point"] = subflows[f"{end}_port"].map(port["point_id"])
        if subflows[f"{end}_point"].isna().any():
            raise ValueError(f"Sub-flow {end} ports missing from the routable ports layer")
    subflows["route_key"] = subflows["origin_point"] + "__" + subflows["dest_point"]
    subflows["same_port"] = subflows["origin_point"] == subflows["dest_point"]
    subflows["pair_distance_km"] = np.round(common.great_circle_km(
        subflows["origin_port"].map(port["snap_lon"]), subflows["origin_port"].map(port["snap_lat"]),
        subflows["dest_port"].map(port["snap_lon"]), subflows["dest_port"].map(port["snap_lat"]),
        settings), 1)

    # Checks: shares sum to 1 per flow; sub-flow values add back to the flow values.
    total = subflows.groupby("flow_id")["share"].sum()
    value_gap = abs(subflows["maritime_value_usd"].sum() - flows["maritime_value_usd"].sum())
    log.info("Task 4.1 | check: shares sum to 1 for %d of %d flows | sub-flow value total differs from "
             "the flow total by %s USD", int(np.isclose(total, 1.0).sum()), len(total), f"{value_gap:,.2f}")
    if not np.isclose(total, 1.0).all() or value_gap > 1:
        raise ValueError("Sub-flow shares or values do not add up to the flows")

    same = subflows[subflows["same_port"]]
    common.log_filter(log, "Task 4.1 | sub-flows with different origin and destination network points",
                      int((~subflows["same_port"]).sum()), len(same),
                      f"both ports snap to the same network point: no sea leg, no route, cross no "
                      f"chokepoint; value kept in the totals, {same['maritime_value_usd'].sum():,.0f} USD")
    for r in same.itertuples():
        log.info("Task 4.1 |   same network point %s (%s -> %s): %s USD", r.subflow_id, r.origin_port,
                 r.dest_port, f"{r.maritime_value_usd:,.0f}")
    routed = subflows[~subflows["same_port"]]
    log.info("Task 4.1 | routed sub-flows %s | port pairs %s | network-point pairs %s",
             f"{len(routed):,}", f"{routed[['origin_port', 'dest_port']].drop_duplicates().shape[0]:,}",
             f"{routed['route_key'].nunique():,}")
    return subflows[SUBFLOW_COLUMNS]


def build_subflows(flows, ports, settings, log):
    """Task 4.1: sub-flows by the ports.assignment in the settings (Decision 14)."""
    assignment = settings["ports"]["assignment"]
    log.info("Task 4.1 | port assignment: %s (ports.assignment; Decision 14)", assignment)
    if assignment == "principal_ports":
        subflows = build_principal_subflows(flows, read_port_shares(settings, log), settings, log)
    elif assignment == "nearest_gateway":
        subflows = build_nearest_subflows(flows, ports, settings, log)
    else:
        raise ValueError(f"Unknown ports.assignment '{assignment}'")
    return add_points(subflows, flows, ports, settings, log)


# ---------------------------------------------------------------------
# Task 4.2: routing
# ---------------------------------------------------------------------

def scenario_restrictions(settings, log):
    """Task 4.2: searoute passage tags closed in each scenario.

    Returns {scenario: {"closed": names, "tags": tags, "diverted_tags": tags}}: "tags"
    closes always_closed plus the scenario's chokepoints; "diverted_tags" adds the
    diverted_also_closed chokepoints, used only for routes displaced by the closure.
    """
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
        also = scenario.get("diverted_also_closed") or []
        missing = [c for c in scenario["closed"] + also if c not in mapping]
        if missing:
            raise ValueError(f"Scenario {name} closes {missing}, which have no searoute passage in "
                             "routing.searoute_passages (use the editable network, S08)")
        tags = list(dict.fromkeys(list(routing["always_closed"]) + [mapping[c] for c in scenario["closed"]]))
        diverted = list(dict.fromkeys(tags + [mapping[c] for c in also]))
        restrictions[name] = {"closed": scenario["closed"], "tags": tags, "diverted_tags": diverted}
        log.info("Task 4.2 | scenario %s: closed chokepoints %s -> searoute restrictions %s%s",
                 name, scenario["closed"] or "none", tags,
                 f"; displaced routes also close {also} -> {diverted}" if also else "")
    return restrictions


def searoute_version():
    """Installed searoute version."""
    from importlib.metadata import version
    return version("searoute")


def point_coords(ports):
    """Task 4.2: {point_id: (lon, lat)} of every network point in use."""
    points = ports.drop_duplicates("point_id").set_index("point_id")
    return {pid: (lon, lat) for pid, lon, lat in zip(points.index, points["snap_lon"], points["snap_lat"])}


def route_pairs(pairs, points, tags, scenario, settings, log, label=None):
    """Task 4.2: route each pair of network points once with one set of searoute restrictions.

    Returns the routes, one per network-point pair (with the continuous line and the
    searoute passages traversed, kept for the checks), and the failed pairs.
    """
    rows, failures, passages = [], [], {}
    for o, d in pairs:
        key = f"{o}__{d}"
        coords, traversed = common.route_line(points[o], points[d], tags)
        if coords is None:
            failures.append({"route_key": key, "scenario": scenario})
            continue
        for p in set(traversed):
            passages[p] = passages.get(p, 0) + 1
        geometry = common.split_antimeridian(coords)
        rows.append({"route_id": f"{key}__{scenario}", "route_key": key, "origin_point": o,
                     "dest_point": d, "scenario": scenario,
                     "length_nm": round(common.length_nm(coords, settings), 1),
                     "crosses_180": len(geometry.geoms) > 1,
                     "max_lat": max(c[1] for c in coords), "passages": sorted(set(traversed)),
                     "continuous": LineString(coords), "geometry": geometry})
    label = label or scenario
    log.info("Task 4.2 | %s (restrictions %s): network-point pairs %s | routed %s | failed (no sea "
             "route with these restrictions) %s | routes crossing 180 degrees %d", label, tags,
             f"{len(pairs):,}", f"{len(rows):,}", f"{len(failures):,}", sum(r["crosses_180"] for r in rows))
    log.info("Task 4.2 | %s: routes through each searoute passage: %s", label,
             ", ".join(f"{p} {n:,}" for p, n in sorted(passages.items())) or "none")
    routes = gpd.GeoDataFrame(rows, columns=list(ROUTE_COLUMNS), geometry="geometry",
                              crs=settings["project"]["storage_crs"])
    return routes, pd.DataFrame(failures, columns=["route_key", "scenario"])


ROUTE_COLUMNS = ("route_id", "route_key", "origin_point", "dest_point", "scenario", "length_nm",
                 "crosses_180", "max_lat", "passages", "continuous", "geometry")


def scenario_routes(scenario, spec, base_routes, base_hits, zones, points, tags, settings, log, label):
    """Task 4.2 / 4.6: routes for a scenario that closes chokepoints.

    Routes whose baseline crosses a closed chokepoint's zone are rerouted with `tags`;
    every other route is copied from the baseline under the scenario's label.
    """
    closed_ids = set(zones.loc[zones["name"].isin(spec["closed"]), "chokepoint_id"])
    displaced = set(base_hits.loc[base_hits["chokepoint_id"].isin(closed_ids), "route_id"])
    moved = base_routes[base_routes["route_id"].isin(displaced)]
    kept = base_routes[~base_routes["route_id"].isin(displaced)].copy()
    kept["scenario"] = scenario
    kept["route_id"] = kept["route_key"] + "__" + scenario

    pairs = list(moved[["origin_point", "dest_point"]].itertuples(index=False, name=None))
    rerouted, failures = route_pairs(pairs, points, tags, scenario, settings, log,
                                     label=f"{label}, displaced routes")
    log.info("Task 4.2 | %s: baseline routes crossing %s: %d, rerouted (%d failed) | other routes "
             "copied from the baseline: %d", label, " or ".join(spec["closed"]), len(moved),
             len(failures), len(kept))
    routes = gpd.GeoDataFrame(pd.concat([kept, rerouted], ignore_index=True), geometry="geometry",
                              crs=base_routes.crs)
    return routes, failures, moved


def log_failures(failures, subflows, log):
    """Task 4.2: sub-flows, port pairs and value on the network-point pairs that could not be routed."""
    for scenario, group in failures.groupby("scenario"):
        hit = subflows.merge(group, on="route_key")
        log.info("Task 4.2 | scenario %s: failed network-point pairs %d carry %d port pairs, %d sub-flows, "
                 "%s USD", scenario, len(group), hit[["origin_port", "dest_port"]].drop_duplicates().shape[0],
                 len(hit), f"{hit['maritime_value_usd'].sum():,.0f}")
        for key, on_key in hit.groupby("route_key"):
            for (o, d), on_pair in on_key.groupby(["origin_port", "dest_port"]):
                log.info("Task 4.2 |   failed %s -> %s (%d sub-flows, %s USD)", o, d, len(on_pair),
                         f"{on_pair['maritime_value_usd'].sum():,.0f}")


def arctic_check(routes, settings, log):
    """Task 4.2: routes north of routing.arctic_check_lat (should be none through the Arctic)."""
    limit = settings["routing"]["arctic_check_lat"]
    north = routes[routes["max_lat"] > limit]
    log.info("Task 4.2 | check: network-point routes reaching north of %.2f N: %d%s", limit, len(north),
             "".join(f"\n    {r.route_id} (max lat {r.max_lat:.1f}, {r.length_nm:,.0f} nm)"
                     for r in north.itertuples()))


def correction_log(scenario, moved, rerouted, open_variant, subflows, log):
    """Task 4.6: log the Cape correction for displaced routes and what it changed."""
    on_moved = subflows.merge(moved[["route_key"]], on="route_key")
    log.info("Task 4.6 | correction | %s: %d baseline network-point routes crossing the closed "
             "chokepoints (%d sub-flows, %s USD) are rerouted with diverted_also_closed as well | reason: "
             "the first run sent displaced Northeast Asia-Europe flows through the Panama Canal "
             "(shortest network distance), but the proposal defines the scenario as forcing ships "
             "round the Cape of Good Hope; Panama stays open for all other flows", scenario,
             len(moved), len(on_moved), f"{on_moved['maritime_value_usd'].sum():,.0f}")
    ids = set(moved["route_key"] + "__" + scenario)
    for name, routes in (("Cape diversion", rerouted), ("Panama open (first run)", open_variant)):
        sub = routes[routes["route_id"].isin(ids)]
        log.info("Task 4.6 | displaced routes, %s: %d routed | via south_africa %d | via panama "
                 "%d | median length %s nm", name, len(sub),
                 int(sub["passages"].apply(lambda p: "south_africa" in p).sum()),
                 int(sub["passages"].apply(lambda p: "panama" in p).sum()),
                 f"{sub['length_nm'].median():,.0f}")


def test_routing(pairs, points, restrictions, settings, log):
    """Task 4.2 test: route the first few pairs with every restriction set and log the lengths."""
    sets = {f"{name} ({key})": spec[key] for name, spec in restrictions.items()
            for key in ("tags", "diverted_tags") if key == "tags" or spec["diverted_tags"] != spec["tags"]}
    found = [route_pairs(pairs, points, tags, label, settings, log) for label, tags in sets.items()]
    routes = pd.concat([r for r, _ in found], ignore_index=True)
    failures = pd.concat([f for _, f in found], ignore_index=True)
    for r in routes.itertuples():
        log.info("Test | %-52s %-34s %9s nm  parts %d  max lat %.1f", r.route_key,
                 r.scenario, f"{r.length_nm:,.0f}", len(r.geometry.geoms), r.max_lat)
    for r in failures.itertuples():
        log.info("Test | %-52s %-34s no route", r.route_key, r.scenario)


def port_pair_routes(routes, subflows, log):
    """Task 4.2: one routes row per port pair and scenario, copying its network-point route.

    S08, S08b, S11 and check_db join routes to sub-flows on origin_port and dest_port.
    """
    pairs = (subflows.loc[~subflows["same_port"], ["origin_port", "dest_port", "route_key"]]
             .drop_duplicates())
    rows = pairs.merge(pd.DataFrame(routes.drop(columns=["route_id", "continuous", "max_lat", "passages"])),
                       on="route_key")
    rows["route_id"] = rows["origin_port"] + "__" + rows["dest_port"] + "__" + rows["scenario"]
    cols = ["route_id", "origin_port", "dest_port", "scenario", "length_nm", "crosses_180",
            "origin_point", "dest_point", "route_key", "geometry"]
    out = gpd.GeoDataFrame(rows[cols], geometry="geometry", crs=routes.crs)
    for scenario, group in out.groupby("scenario"):
        log.info("Task 4.2 | routes layer, %s: %s network-point routes copied to %s port pairs", scenario,
                 f"{group['route_key'].nunique():,}", f"{len(group):,}")
    return out.sort_values(["scenario", "origin_port", "dest_port"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Task 4.3: crossings
# ---------------------------------------------------------------------

def route_zone_hits(routes, zones, log, label):
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
    log.info("Task 4.3 | %s: route-zone intersections %s | check on continuous routes with zones "
             "shifted +-360: %s | %s", label, f"{len(found):,}", f"{len(checked):,}",
             "agree" if found == checked else
             f"DISAGREE: only split {sorted(found - checked)}; only continuous {sorted(checked - found)}")
    near_180 = routes.loc[routes["crosses_180"], "route_id"]
    log.info("Task 4.3 | %s: routes crossing 180 degrees that cross a zone: %d of %d", label,
             hits.loc[hits["route_id"].isin(near_180), "route_id"].nunique(), len(near_180))
    return hits.drop_duplicates()


def endpoint_zone_check(ports, zones, log):
    """Task 4.3: routes that start or end inside a zone count as crossing it; list the ports."""
    snap = ports.drop_duplicates("port_id").set_index("port_id")
    lon = snap["snap_lon"].where(snap["snap_lon"] <= 180, snap["snap_lon"] - 360)
    points = gpd.GeoDataFrame({"port_id": snap.index},
                              geometry=[Point(x, y) for x, y in zip(lon, snap["snap_lat"])],
                              crs=zones.crs)
    inside = gpd.sjoin(points, zones, predicate="within")
    log.info("Task 4.3 | ports whose network point lies inside a zone (their routes cross it): %d%s",
             len(inside), "".join(f"\n    {r.port_id} in {r.name}" for r in inside.itertuples()))


def build_crossings(hits, routes, subflows, log, label=""):
    """Task 4.3: sub-flow crossings, and one row per flow, scenario and chokepoint crossed.

    crossing_share and value_usd are the share and value of the flow's sub-flows that cross.
    """
    routed = subflows[~subflows["same_port"]]
    on_route = routes[["route_id", "route_key", "scenario"]].merge(
        routed[["subflow_id", "flow_id", "route_key", "share", "maritime_value_usd"]], on="route_key")
    sub_cross = on_route.merge(hits, on="route_id")
    crossings = (sub_cross.groupby(["flow_id", "scenario", "chokepoint_id"], as_index=False)
                 .agg(crossing_share=("share", "sum"), value_usd=("maritime_value_usd", "sum")))
    crossings["crossing_share"] = crossings["crossing_share"].round(6)
    crossings["value_usd"] = crossings["value_usd"].round(2)
    for scenario, group in crossings.groupby("scenario"):
        here = on_route[on_route["scenario"] == scenario]
        crossing_subs = sub_cross.loc[sub_cross["scenario"] == scenario, "subflow_id"].nunique()
        routed_flows = here["flow_id"].nunique()
        log.info("Task 4.3 | scenario %s%s: routed sub-flows %s (flows %s) | sub-flows crossing at least "
                 "one chokepoint %s | flow-chokepoint crossings %s | flows with a crossing sub-flow %s | "
                 "flows crossing none %s", scenario, label, f"{len(here):,}", f"{routed_flows:,}",
                 f"{crossing_subs:,}", f"{len(group):,}", f"{group['flow_id'].nunique():,}",
                 f"{routed_flows - group['flow_id'].nunique():,}")
    sub_cross = sub_cross[["subflow_id", "flow_id", "scenario", "chokepoint_id", "share"]]
    return crossings.reset_index(drop=True), sub_cross


# ---------------------------------------------------------------------
# Task 4.4: Equation 1
# ---------------------------------------------------------------------

def trade_exposure(sub_cross, subflows, chokepoints, settings, log):
    """Task 4.4: Equation 1 by value and weight, and the number of flows with a crossing sub-flow."""
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
    """Task 4.5: PortWatch summary file, with a check of the field alignment.

    Returns the vessel counts by type (columns named by vessel type), indexed by portid.
    """
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
    counts.columns = [c.replace(PW_COUNT.format(""), "") for c in count_cols]
    return counts.set_index(pw[PW_ID])


def read_daily(settings, log):
    """Task 4.5: PortWatch daily transits, one row per chokepoint and day."""
    path = common.get_path(settings, "raw_files.portwatch_transits_daily")
    daily = pd.read_csv(path, encoding="utf-8-sig")
    daily["date"] = pd.to_datetime(daily["date"], format="%Y/%m/%d")
    log.info("Read %s: %s rows | %d chokepoints | %s to %s", path.name, f"{len(daily):,}",
             daily[PW_ID].nunique(), daily["date"].min().date(), daily["date"].max().date())
    dupes = int(daily.duplicated(["date", PW_ID]).sum())
    if dupes:
        raise ValueError(f"{path.name}: {dupes} repeated chokepoint-days")
    return daily


def period_transits(daily, settings, log):
    """Task 4.5: transits by vessel type summed over routing.validation.transit_period."""
    period = settings["routing"]["validation"]["transit_period"]
    start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    inside = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    days = inside.groupby(PW_ID).size()
    expected = (end - start).days + 1
    short = days[days != expected]
    log.info("Task 4.5 | daily transits %s to %s: %s rows | days per chokepoint %d expected, %s",
             start.date(), end.date(), f"{len(inside):,}", expected,
             "all complete" if short.empty else f"incomplete: {short.to_dict()}")
    types = settings["routing"]["validation"]["vessel_types"]
    sums = inside.groupby(PW_ID)[[DAILY_COUNT.format(t) for t in types]].sum()
    sums.columns = types
    return sums


def flow_counts(sub_cross, scenario):
    """Task 4.5: crossing flows per chokepoint, weighted by the share that crosses, and any-crossing."""
    here = sub_cross[sub_cross["scenario"] == scenario].groupby("chokepoint_id")
    return here["share"].sum(), here["flow_id"].nunique()


def validate_variant(variant, chokepoints, counts, transits, is_main, settings, log):
    """Task 4.5: Spearman rho between modeled flow shares and PortWatch transit shares.

    counts: (share-weighted crossing flows, flows with any crossing sub-flow) per
    chokepoint_id; the shares use the weighted count. transits: counts by vessel type per portid.
    """
    v = settings["routing"]["validation"]
    types = v["vessel_types"]
    missing = sorted(set(chokepoints["chokepoint_id"]) - set(transits.index))
    if missing:
        raise ValueError(f"Chokepoints missing from the PortWatch data: {missing}")

    weighted, anyflow = counts
    table = chokepoints[["chokepoint_id", "name"]].copy()
    table.insert(0, "variant", variant)
    table["scenario"] = v["scenario"]
    table["model_nflows"] = table["chokepoint_id"].map(anyflow).fillna(0).astype("int64")
    table["model_wflows"] = table["chokepoint_id"].map(weighted).fillna(0.0)
    for t in types:
        table[f"pw_{t}"] = table["chokepoint_id"].map(transits[t]).astype("int64")
    table["pw_transits"] = table[[f"pw_{t}" for t in types]].sum(axis=1)
    table["model_share"] = table["model_wflows"] / table["model_wflows"].sum()
    table["pw_share"] = table["pw_transits"] / table["pw_transits"].sum()
    table["model_rank"] = table["model_share"].rank(ascending=False, method="min").astype("int64")
    table["pw_rank"] = table["pw_share"].rank(ascending=False, method="min").astype("int64")

    rho, p = spearmanr(table["model_share"], table["pw_share"])
    rho_any, p_any = spearmanr(table["model_nflows"], table["pw_share"])
    passed = bool(rho >= v["pass_mark_spearman"])
    log.info("Task 4.5 | variant %s%s: share-weighted flow shares vs PortWatch %s shares, %d chokepoints: "
             "Spearman rho = %.3f, p = %.2g -> %s (pass mark %.2f) | for reference, flows with any "
             "crossing sub-flow: rho = %.3f, p = %.2g", variant, " (main result)" if is_main else "",
             "+".join(types), len(table), rho, p, "PASS" if passed else "BELOW PASS MARK",
             v["pass_mark_spearman"], rho_any, p_any)
    for r in table.sort_values(["pw_rank", "name"]).itertuples():
        log.info("    PortWatch rank %2d  model rank %2d  %-22s pw share %5.1f%%  model share %5.1f%%",
                 r.pw_rank, r.model_rank, r.name, 100 * r.pw_share, 100 * r.model_share)

    table["model_wflows"] = table["model_wflows"].round(3)
    table["model_share"] = table["model_share"].round(4)
    table["pw_share"] = table["pw_share"].round(4)
    table["spearman_rho"] = round(float(rho), 4)
    table["p_value"] = float(p)
    table["pass_mark"] = v["pass_mark_spearman"]
    table["passed"] = passed
    table["is_main"] = is_main
    return table.sort_values(["pw_rank", "name"]), float(rho)


def red_sea_context(daily, settings, log):
    """Context: PortWatch transits per calendar year, all vessel types and container only."""
    context = settings["routing"]["red_sea_context"]
    names = context["chokepoints"]
    unknown = sorted(set(names) - set(daily[PW_NAME]))
    if unknown:
        raise ValueError(f"routing.red_sea_context chokepoints not in the daily file: {unknown}")
    sub = daily[daily[PW_NAME].isin(names) & daily["date"].dt.year.isin(context["years"])]
    table = (sub.assign(year=sub["date"].dt.year)
             .groupby([PW_ID, PW_NAME, "year"])
             .agg(days=("date", "size"), transits_all=(DAILY_COUNT.format("total"), "sum"),
                  transits_container=(DAILY_COUNT.format("container"), "sum"))
             .reset_index().rename(columns={PW_ID: "chokepoint_id", PW_NAME: "name"}))
    order = {name: i for i, name in enumerate(names)}
    table = table.sort_values(["name", "year"], key=lambda c: c.map(order) if c.name == "name" else c)
    log.info("Context | PortWatch transits per calendar year (all types / container):")
    for r in table.itertuples():
        log.info("    %-22s %d  days %3d  all %7s  container %6s", r.name, r.year, r.days,
                 f"{r.transits_all:,}", f"{r.transits_container:,}")
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------
# Comparison with the first pass
# ---------------------------------------------------------------------

def no_route_pairs(subflows, routes, scenario):
    """Routed sub-flows with no route in a scenario: (port pairs, sub-flows, value USD)."""
    routed = subflows[~subflows["same_port"].astype(bool)]
    have = set(zip(*(routes.loc[routes["scenario"] == scenario, c] for c in ("origin_port", "dest_port"))))
    missing = routed[[p not in have for p in zip(routed["origin_port"], routed["dest_port"])]]
    return (missing[["origin_port", "dest_port"]].drop_duplicates().shape[0], len(missing),
            missing["maritime_value_usd"].sum())


def compare_first_pass(scores, rho_new, subflows, routes, zones, transits, settings, log):
    """Log trade exposure, validation rho and Red Sea no-route pairs: first pass vs this run."""
    path = common.get_path(settings, "first_pass_database")
    if not path.exists():
        log.warning("Comparison | first-pass database %s not found; comparison skipped", path)
        return
    old = pd.DataFrame(common.read_layer(settings, "chokepoint_scores", database=path))
    log.info("Comparison | first pass: %s (ports.assignment nearest_gateway, WPI ports)", path)
    both = scores[["chokepoint_id", "name", "trade_value_base", "trade_value_redsea"]].merge(
        old[["chokepoint_id", "trade_value_base", "trade_value_redsea"]], on="chokepoint_id",
        how="left", suffixes=("", "_old"))
    for s in ("base", "redsea"):
        both[f"rank_{s}"] = both[f"trade_value_{s}"].rank(ascending=False, method="min").astype(int)
        both[f"rank_{s}_old"] = both[f"trade_value_{s}_old"].rank(ascending=False, method="min").astype(int)
    log.info("Comparison | trade exposure per chokepoint, value USD (rank change: + = moved up):")
    log.info("  %-22s %14s %14s %4s %4s %4s   %14s %14s %4s %4s %4s", "chokepoint", "base old", "base new",
             "old", "new", "chg", "red sea old", "red sea new", "old", "new", "chg")
    for r in both.sort_values(["rank_base", "name"]).itertuples():
        log.info("  %-22s %14s %14s %4d %4d %+4d   %14s %14s %4d %4d %+4d", r.name,
                 f"{r.trade_value_base_old:,.0f}", f"{r.trade_value_base:,.0f}", r.rank_base_old, r.rank_base,
                 r.rank_base_old - r.rank_base, f"{r.trade_value_redsea_old:,.0f}",
                 f"{r.trade_value_redsea:,.0f}", r.rank_redsea_old, r.rank_redsea,
                 r.rank_redsea_old - r.rank_redsea)
    for s in ("base", "redsea"):
        rho, _ = spearmanr(both[f"trade_value_{s}_old"], both[f"trade_value_{s}"])
        log.info("Comparison | trade_value_%s: Spearman rho old vs new %.3f", s, rho)

    # Validation rho of the first pass, recomputed the same way (each flow there has share 1).
    v = settings["routing"]["validation"]
    old_cross = pd.DataFrame(common.read_layer(settings, "route_crossings", database=path))
    n_old = old_cross[old_cross["scenario"] == v["scenario"]].groupby("chokepoint_id")["flow_id"].nunique()
    observed = transits[v["vessel_types"]].sum(axis=1)
    rho_old, _ = spearmanr(zones["chokepoint_id"].map(n_old).fillna(0), zones["chokepoint_id"].map(observed))
    log.info("Comparison | validation rho (%s, main variant): first pass %.3f (recomputed; Lab 8 reported "
             "0.641 before the Decision 12 gate lines) | this run %.3f", v["scenario"], rho_old, rho_new)

    old_sub = pd.DataFrame(common.read_layer(settings, "subflows", database=path))
    old_routes = pd.DataFrame(common.read_layer(settings, "routes", database=path))
    for label, (sf, rt) in (("first pass", (old_sub, old_routes)), ("this run", (subflows, routes))):
        n_pairs, n_sub, value = no_route_pairs(sf, rt, "red_sea_diversion")
        log.info("Comparison | red_sea_diversion port pairs with no sea route, %s: %d pairs, %d sub-flows, "
                 "%s USD", label, n_pairs, n_sub, f"{value:,.0f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-pairs", type=int, default=None,
                        help="route only the first N network-point pairs, log their lengths, write nothing")
    parser.add_argument("--sample-flows", type=int, default=None,
                        help="run every step on a random sample of N flows, write nothing")
    return parser.parse_args()


def main(test_pairs=None, sample_flows=None):
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)

    flows = read_flows(settings, log, sample=sample_flows)
    ports = read_ports(settings, log)
    points = point_coords(ports)
    subflows = build_subflows(flows, ports, settings, log)
    restrictions = scenario_restrictions(settings, log)

    # Task 4.2: unique network-point pairs among the routed sub-flows.
    routed = subflows[~subflows["same_port"]]
    pairs = list(routed[["origin_point", "dest_point"]].drop_duplicates().itertuples(index=False, name=None))
    log.info("Task 4.2 | unique network-point pairs to route: %s (from %s sub-flows)", f"{len(pairs):,}",
             f"{len(routed):,}")
    if test_pairs:
        test_routing(pairs[:test_pairs], points, restrictions, settings, log)
        log.info("Test run finished; nothing written")
        return

    # Task 4.2: baseline first; other scenarios reroute only the routes they displace.
    base = [n for n, sc in settings["routing"]["scenarios"].items() if sc.get("main")][0]
    base_routes, base_failures = route_pairs(pairs, points, restrictions[base]["tags"], base, settings, log)
    zones = read_zones(settings, log)
    # Task 4.3 / Decision 12: crossings are tested on the gate where there is one, else the zone.
    shapes = read_crossing_shapes(zones, settings, log)
    base_hits = route_zone_hits(base_routes, shapes, log, base)
    all_routes, all_hits, all_failures = [base_routes], [base_hits], [base_failures]
    open_variant = {}
    for name, spec in restrictions.items():
        if name == base:
            continue
        routes_s, fail_s, moved = scenario_routes(name, spec, base_routes, base_hits, shapes, points,
                                                  spec["diverted_tags"], settings, log, label=name)
        all_routes.append(routes_s)
        all_hits.append(route_zone_hits(routes_s, shapes, log, name))
        all_failures.append(fail_s)
        if spec["diverted_tags"] != spec["tags"]:
            # Validation variant: displaced routes with diverted_also_closed left open (first run).
            label = f"{name}, diverted_also_closed open (validation variant)"
            open_routes, _, _ = scenario_routes(name, spec, base_routes, base_hits, shapes, points,
                                                spec["tags"], settings, log, label=label)
            open_variant[name] = (open_routes, route_zone_hits(open_routes, shapes, log, label))
            correction_log(name, moved, routes_s, open_routes, subflows, log)

    routes = gpd.GeoDataFrame(pd.concat(all_routes, ignore_index=True), geometry="geometry",
                              crs=base_routes.crs)
    hits = pd.concat(all_hits, ignore_index=True)
    log_failures(pd.concat(all_failures, ignore_index=True), subflows, log)
    arctic_check(routes, settings, log)

    # Task 4.3: crossings.
    # A port cannot lie inside a gate line, so only zone-tested chokepoints are checked.
    endpoint_zone_check(ports, shapes[shapes["crossing_method"] == "zone"], log)
    crossings, sub_cross = build_crossings(hits, routes, subflows, log)

    # Task 4.4: Equation 1.
    scores = trade_exposure(sub_cross, subflows, zones, settings, log)
    log_trade_ranking(scores, log)

    # Task 4.5: validation, three variants (the main one last).
    validation_settings = settings["routing"]["validation"]
    v_scenario = validation_settings["scenario"]
    counts_main = flow_counts(sub_cross, v_scenario)
    counts_open = counts_main
    if v_scenario in open_variant:
        open_routes, open_hits = open_variant[v_scenario]
        _, open_sub = build_crossings(open_hits, open_routes, subflows, log,
                                      label=" (diverted_also_closed open)")
        counts_open = flow_counts(open_sub, v_scenario)
    daily = read_daily(settings, log)
    transits = period_transits(daily, settings, log)
    year = validation_settings["transit_period"]["start"][:4]
    variants = [
        validate_variant(VARIANT_SUMMARY, zones, counts_open, read_portwatch(settings, log),
                         False, settings, log),
        validate_variant(VARIANT_DAILY_OPEN.format(year=year), zones, counts_open, transits,
                         False, settings, log),
        validate_variant(VARIANT_DAILY_MAIN.format(year=year), zones, counts_main, transits,
                         True, settings, log),
    ]
    validation = pd.concat([t for t, _ in variants], ignore_index=True)
    rho_main = variants[-1][1]
    weighted_base, _ = flow_counts(sub_cross, base)
    observed = transits[validation_settings["vessel_types"]].sum(axis=1)
    rho_b, p_b = spearmanr(zones["chokepoint_id"].map(weighted_base).fillna(0),
                           zones["chokepoint_id"].map(observed))
    log.info("Task 4.5 | for reference only, %s share-weighted flow shares vs daily %s transits: "
             "rho = %.3f, p = %.2g", base, year, rho_b, p_b)

    # Task 4.2: the routes layer has one row per port pair and scenario.
    port_routes = port_pair_routes(routes, subflows, log)
    compare_first_pass(scores, rho_main, subflows, port_routes, zones, transits, settings, log)
    if sample_flows:
        log.info("Test run finished; nothing written")
        return

    tables = common.get_path(settings, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    validation.to_csv(tables / "validation.csv", index=False)
    log.info("Wrote %s (%d rows, %d variants)", tables / "validation.csv", len(validation),
             validation["variant"].nunique())
    context = red_sea_context(daily, settings, log)
    context.to_csv(tables / "red_sea_context.csv", index=False)
    log.info("Wrote %s (%d rows)", tables / "red_sea_context.csv", len(context))

    common.write_layer(subflows, settings, "subflows", log)
    common.write_layer(port_routes, settings, "routes", log)
    common.write_layer(crossings, settings, "route_crossings", log)
    common.write_layer(merge_scores(scores, settings, log), settings, "chokepoint_scores", log)
    log.info("Run of s06_trade_exposure finished")


if __name__ == "__main__":
    args = parse_args()
    main(args.test_pairs, args.sample_flows)

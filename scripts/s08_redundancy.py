"""S08: route redundancy of each chokepoint (tasks 4.12-4.14).

1. Task 4.12: an editable copy of searoute's own sea-lane network (copy.deepcopy of
   searoute.setup_M(); setup_M() itself is never edited, because searoute caches it and
   every caller would see the change). The network's edges are tested against each
   chokepoint's crossing shape from common.crossing_geometries: the gate line for the
   four wide straits (Decision 12), otherwise the zone circle (Decision 11). An edge is
   removed if it intersects the shape (Decision 10a), with the shape copied 360 degrees
   east and west as in S03 and S06. The 12 zero-length edges joining 180 and -180 are
   skipped. Only edges are removed, never nodes, so every port snaps to the same node
   as in S06.
2. Check: every baseline port pair (redundancy.routes_from) is routed on the unedited
   copy. The lengths must equal S06's routes layer, and the network edges each path uses
   must agree with S06's route_crossings (a sub-flow is affected by a closure when its
   flow crosses the chokepoint and its path uses a removed edge). S08 stops if either
   check fails.
3. Task 4.13: one chokepoint is closed at a time on a fresh copy of the network; every
   other passage stays open, including Panama when Suez is closed (Decision 10b).
   routing.always_closed (the Arctic routes) stays closed, as in S06. Each affected port
   pair is routed once and mapped back to its sub-flows. Sub-flows from or to a port whose
   snapped network point lies inside the closed zone are not rerouted (Decision 10c;
   zones only, never gates). After each reroute S08 checks that the new path does not
   touch the closed gate or zone (leak test) and stops if it does.
4. Task 4.14: Equation 4, the detour penalty (percent extra distance, weighted by trade
   value) over the affected sub-flows that are not at a port inside the zone. Sub-flows
   with no remaining sea route get the largest single detour penalty observed across all
   closures (redundancy.no_alternative_penalty = max_observed; Decision 10d) and count
   toward the share of value with no sea alternative. A chokepoint with no sub-flows in
   Equation 4 gets a null penalty (Decision 10e).
5. Outputs (Decision 10f): S08 owns the chokepoint_scores columns starting with
   "redund_" and keeps every other column; per-sub-flow detours go to the detours table
   and rerouted lines to the detour_routes layer. S08 never writes to routes.

Test first with:  python scripts/s08_redundancy.py --only "Suez Canal" "Korea Strait"
(closes only the chokepoints named, logs the results, writes nothing; the no-route
penalty is then the largest among the chokepoints tested).
"""

import argparse
import copy
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import searoute
from shapely.affinity import translate
from shapely.geometry import LineString, Point

import common

S08_PREFIX = "redund_"     # S08 owns every chokepoint_scores column starting with this
SHIFTS = (-360.0, 0.0, 360.0)   # copies of a shape tested against lines near 180 degrees
DETOUR_COLUMNS = ["subflow_id", "flow_id", "chokepoint_id", "origin_port", "dest_port",
                  "old_length_nm", "new_length_nm", "extra_nm", "detour_pct", "no_route",
                  "port_in_zone"]
DETOUR_ROUTE_COLUMNS = ["route_id", "origin_port", "dest_port", "chokepoint_id", "old_length_nm",
                        "new_length_nm", "extra_nm", "crosses_180", "max_lat", "geometry"]


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_inputs(settings, log):
    """Baseline sub-flows, routes and crossings, and the snapped port points."""
    scenario = settings["redundancy"]["routes_from"]
    subflows = common.read_layer(settings, "subflows")
    routed = ~subflows["same_port"].astype(bool)
    common.log_filter(log, "Task 4.13 | sub-flows with a route", int(routed.sum()), int((~routed).sum()),
                      "both ports snap to the same network node (left unrouted by S06)")
    subflows = subflows[routed].reset_index(drop=True)

    routes = common.read_layer(settings, "routes")
    routes = pd.DataFrame(routes[routes["scenario"] == scenario].drop(columns="geometry"))
    crossings = common.read_layer(settings, "route_crossings")
    crossings = crossings[crossings["scenario"] == scenario][["flow_id", "chokepoint_id"]]
    log.info("Task 4.13 | scenario %s (redundancy.routes_from): routes %s | crossings %s",
             scenario, f"{len(routes):,}", f"{len(crossings):,}")

    ports = common.read_layer(settings, "ports")
    snap = ports.drop_duplicates("port_id").set_index("port_id")[["snap_lon", "snap_lat"]]
    return subflows, routes, crossings, snap


def read_shapes(settings, log, only=None):
    """The gate line or zone of each included chokepoint (Decision 12), optionally a few by name."""
    zones = common.read_layer(settings, "zones")
    included = zones["included"].str.strip().str.lower() == "yes"
    zones = zones[included][["chokepoint_id", "name", "geometry"]].reset_index(drop=True)
    shapes = common.crossing_geometries(zones, settings)
    if only:
        unknown = sorted(set(only) - set(shapes["name"]))
        if unknown:
            raise ValueError(f"--only names chokepoints that are not included: {unknown}")
        shapes = shapes[shapes["name"].isin(only)].reset_index(drop=True)
        log.info("Test run: closing only %s; nothing is written", ", ".join(shapes["name"]))
    counts = shapes["crossing_method"].value_counts()
    log.info("Task 4.12 | chokepoints to close: %d (gate line %d, zone %d)", len(shapes),
             counts.get("gate", 0), counts.get("zone", 0))
    return shapes


# ---------------------------------------------------------------------
# Task 4.12: editable network
# ---------------------------------------------------------------------

def network_edges(network, settings, log):
    """Every network edge as a line, without the zero-length edges joining 180 and -180."""
    edges = list(network.edges())
    joins = [abs(u[0]) == 180 and abs(v[0]) == 180 and u[1] == v[1] and u[0] != v[0]
             for u, v in edges]
    kept = [e for e, join in zip(edges, joins) if not join]
    log.info("Task 4.12 | network copy: nodes %s | edges %s | zero-length 180/-180 edges skipped "
             "in the zone test %d", f"{network.number_of_nodes():,}", f"{len(edges):,}", sum(joins))
    return gpd.GeoDataFrame({"edge": kept}, geometry=[LineString(e) for e in kept],
                            crs=settings["project"]["storage_crs"])


def edges_to_remove(edges, shapes, log):
    """Task 4.12 / Decision 10a: the edges that intersect each chokepoint's gate or zone."""
    removed = {}
    for r in shapes.itertuples():
        hits = set()
        for dx in SHIFTS:
            hits |= set(edges.sindex.query(translate(r.geometry, xoff=dx), predicate="intersects"))
        removed[r.chokepoint_id] = [edges["edge"].iloc[i] for i in sorted(hits)]
    return removed


def shifted(shape):
    """The shape and its copies 360 degrees east and west."""
    return [translate(shape, xoff=dx) for dx in SHIFTS]


# ---------------------------------------------------------------------
# Check: baseline reproduced, affected sub-flows agree with route_crossings
# ---------------------------------------------------------------------

def baseline_paths(network, pairs, snap, routes, tags, settings, log):
    """Route every baseline pair on the unedited copy; return the edges each path uses.

    Stops if any length differs from S06's routes layer.
    """
    old = routes.set_index(["origin_port", "dest_port"])["length_nm"]
    network.restrictions = tags
    paths, differ = {}, []
    for o, d in pairs:
        coords, _ = common.route_line(tuple(snap.loc[o]), tuple(snap.loc[d]), tags, network=network)
        length = round(common.length_nm(coords, settings), 1) if coords else None
        if length is None or abs(length - old[(o, d)]) > 0.01:
            differ.append((o, d, old[(o, d)], length))
        _, nodes = network.shortest_path(tuple(snap.loc[o]), tuple(snap.loc[d]))
        paths[(o, d)] = {frozenset(e) for e in zip(nodes[:-1], nodes[1:])}
    log.info("Check | baseline rerouted on the unedited copy: %s pairs | lengths equal to S06 routes: "
             "%s | differ: %d", f"{len(pairs):,}", f"{len(pairs) - len(differ):,}", len(differ))
    if differ:
        raise ValueError(f"Baseline not reproduced on the network copy (first 5): {differ[:5]}")
    return paths


def affected_subflows(cid, removed, paths, subflows, crossings):
    """Sub-flows affected by closing one chokepoint, after checking S06 agrees.

    A sub-flow is affected when its flow crosses the chokepoint (route_crossings) and its
    baseline path uses a removed edge. Returns (affected, problems): problems lists flows
    S06 says cross but none of whose paths use a removed edge, and paths that use a
    removed edge although S06 says their flow does not cross.
    """
    gone = {frozenset(e) for e in removed}
    uses = {pair for pair, edges in paths.items() if edges & gone}
    on_edge = subflows[[pair in uses for pair in zip(subflows["origin_port"], subflows["dest_port"])]]
    crossing = set(crossings.loc[crossings["chokepoint_id"] == cid, "flow_id"])
    affected = on_edge[on_edge["flow_id"].isin(crossing)]
    problems = {"crossing_flows_without_removed_edge": sorted(crossing - set(affected["flow_id"])),
                "removed_edge_without_crossing": sorted(set(on_edge["flow_id"]) - crossing)}
    return affected.reset_index(drop=True), problems


# ---------------------------------------------------------------------
# Task 4.13: close one chokepoint and reroute
# ---------------------------------------------------------------------

def ports_in_zone(shape_row, snap):
    """Decision 10c: ports whose snapped network point lies inside a zone (never a gate)."""
    if shape_row.crossing_method != "zone":
        return set()
    inside = set()
    for port_id, (lon, lat) in snap.iterrows():
        if shape_row.geometry.contains(Point(lon if lon <= 180 else lon - 360, lat)):
            inside.add(port_id)
    return inside


def reroute(shape_row, removed, base_network, pairs, snap, tags, settings):
    """Task 4.13: close one chokepoint on a fresh copy and route each pair once.

    Returns {pair: coords or None (no sea route)}. Stops if a new path touches the
    closed gate or zone (leak test).
    """
    network = copy.deepcopy(base_network)
    network.remove_edges_from(removed)
    copies = shifted(shape_row.geometry)
    found = {}
    for o, d in pairs:
        coords, _ = common.route_line(tuple(snap.loc[o]), tuple(snap.loc[d]), tags, network=network)
        if coords is not None and any(LineString(coords).intersects(c) for c in copies):
            raise ValueError(f"Leak test failed: closing {shape_row.name} ({shape_row.chokepoint_id}), "
                             f"the new route {o} -> {d} still touches its {shape_row.crossing_method}")
        found[(o, d)] = coords
    return found


def detour_rows(cid, affected, in_zone, found, routes, settings):
    """One detours row per affected sub-flow; one detour_routes row per rerouted pair."""
    old = routes.set_index(["origin_port", "dest_port"])["length_nm"]
    new_len = {pair: round(common.length_nm(c, settings), 1) for pair, c in found.items() if c is not None}
    rows = affected[["subflow_id", "flow_id", "origin_port", "dest_port"]].copy()
    rows.insert(2, "chokepoint_id", cid)
    pairs = list(zip(rows["origin_port"], rows["dest_port"]))
    rows["old_length_nm"] = [old[p] for p in pairs]
    rows["new_length_nm"] = [new_len.get(p, np.nan) for p in pairs]
    rows["extra_nm"] = (rows["new_length_nm"] - rows["old_length_nm"]).round(1)
    rows["detour_pct"] = (100 * rows["extra_nm"] / rows["old_length_nm"]).round(3)
    # Explicit bool: an empty table (no affected sub-flows) would otherwise get object columns.
    rows["port_in_zone"] = pd.Series([o in in_zone or d in in_zone for o, d in pairs],
                                     index=rows.index, dtype=bool)
    rows["no_route"] = pd.Series([(not z) and p not in new_len for p, z in zip(pairs, rows["port_in_zone"])],
                                 index=rows.index, dtype=bool)

    lines = []
    for (o, d), coords in found.items():
        if coords is None:
            continue
        geometry = common.split_antimeridian(coords)
        lines.append({"route_id": f"{o}__{d}__closed_{cid}", "origin_port": o, "dest_port": d,
                      "chokepoint_id": cid, "old_length_nm": old[(o, d)], "new_length_nm": new_len[(o, d)],
                      "extra_nm": round(new_len[(o, d)] - old[(o, d)], 1),
                      "crosses_180": len(geometry.geoms) > 1, "max_lat": max(c[1] for c in coords),
                      "geometry": geometry})
    return rows[DETOUR_COLUMNS], lines


def close_all(shapes, removed, base_network, paths, subflows, crossings, routes, snap, tags, settings, log):
    """Task 4.13: close each chokepoint in turn; return the detours table and the new lines."""
    all_rows, all_lines = [], []
    for r in shapes.itertuples():
        cid = r.chokepoint_id
        affected, problems = affected_subflows(cid, removed[cid], paths, subflows, crossings)
        if any(problems.values()):
            raise ValueError(f"{r.name}: route_crossings and the removed edges disagree: {problems}")
        in_zone = ports_in_zone(r, snap)
        pairs = sorted(set(zip(affected["origin_port"], affected["dest_port"])))
        to_route = [(o, d) for o, d in pairs if o not in in_zone and d not in in_zone]
        found = reroute(r, removed[cid], base_network, to_route, snap, tags, settings)
        rows, lines = detour_rows(cid, affected, in_zone, found, routes, settings)
        log.info("Task 4.13 | %-22s %-4s edges removed %3d | affected pairs %3d, sub-flows %3d | "
                 "port in zone %3d pairs | rerouted %3d pairs | no route %3d pairs | leak test passed",
                 r.name, r.crossing_method, len(removed[cid]), len(pairs), len(affected),
                 len(pairs) - len(to_route), sum(c is not None for c in found.values()),
                 sum(c is None for c in found.values()))
        if in_zone:
            log.info("Task 4.13 |   ports inside the zone: %s", ", ".join(sorted(in_zone)))
        all_rows.append(rows)
        all_lines.extend(lines)
    detours = pd.concat(all_rows, ignore_index=True)
    detours[["no_route", "port_in_zone"]] = detours[["no_route", "port_in_zone"]].astype(bool)
    lines = gpd.GeoDataFrame(all_lines, columns=DETOUR_ROUTE_COLUMNS, geometry="geometry",
                             crs=settings["project"]["storage_crs"])
    return detours, lines


def log_special(detours, subflows, shapes, log):
    """List the pairs with no sea route and the pairs at a port inside the zone, with value."""
    value = detours.merge(subflows[["subflow_id", "maritime_value_usd"]], on="subflow_id")
    names = shapes.set_index("chokepoint_id")["name"]
    for flag, label in (("no_route", "no sea route"), ("port_in_zone", "port inside the zone")):
        sub = value[value[flag]]
        for (cid, o, d), g in sub.groupby(["chokepoint_id", "origin_port", "dest_port"], sort=False):
            log.info("Task 4.13 |   %s, %s: %s -> %s (%d sub-flows, %s USD)", names[cid], label, o, d,
                     len(g), f"{g['maritime_value_usd'].sum():,.0f}")


def log_checks(detours, lines, settings, log):
    """Negative detours (should be none) and detour routes reaching the Arctic."""
    negative = detours[detours["extra_nm"] < 0]
    log.info("Check | rerouted sub-flows with a shorter new route than the baseline: %d%s", len(negative),
             f" (smallest {negative['extra_nm'].min()} nm)" if len(negative) else "")
    limit = settings["routing"]["arctic_check_lat"]
    north = lines[lines["max_lat"] > limit]
    log.info("Check | detour routes reaching north of %.2f N: %d%s", limit, len(north),
             "".join(f"\n    {r.route_id} (max lat {r.max_lat:.1f})" for r in north.itertuples()))


# ---------------------------------------------------------------------
# Task 4.14: Equation 4
# ---------------------------------------------------------------------

def largest_penalty(detours, subflows, shapes, settings, log):
    """Decision 10d: the largest single detour penalty across all closures."""
    rule = settings["redundancy"]["no_alternative_penalty"]
    if rule != "max_observed":
        raise ValueError(f"redundancy.no_alternative_penalty '{rule}' is not supported (only max_observed)")
    routed = detours.dropna(subset=["detour_pct"])
    if routed.empty:
        log.info("Task 4.14 | no rerouted sub-flows, so no largest penalty")
        return np.nan
    top = routed.loc[routed["detour_pct"].idxmax()]
    value = subflows.set_index("subflow_id").loc[top["subflow_id"], "maritime_value_usd"]
    name = shapes.set_index("chokepoint_id").loc[top["chokepoint_id"], "name"]
    log.info("Task 4.14 | largest detour penalty (given to sub-flows with no sea route): %.3f%% | "
             "sub-flow %s (%s -> %s, %s USD) when %s is closed: %s -> %s nm", top["detour_pct"],
             top["subflow_id"], top["origin_port"], top["dest_port"], f"{value:,.0f}", name,
             f"{top['old_length_nm']:,.1f}", f"{top['new_length_nm']:,.1f}")
    return float(top["detour_pct"])


def weighted_mean(values, weights):
    """Value-weighted mean, or null when there is nothing to average."""
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() > 0 else np.nan


def redundancy_scores(shapes, removed, detours, subflows, penalty):
    """Task 4.14: Equation 4 and the counts for each chokepoint (redund_ columns)."""
    d = detours.merge(subflows[["subflow_id", "maritime_value_usd"]], on="subflow_id")
    rows = []
    for r in shapes.itertuples():
        g = d[d["chokepoint_id"] == r.chokepoint_id]
        in_eq = g[~g["port_in_zone"]]
        routed = in_eq[~in_eq["no_route"]]
        noroute = in_eq[in_eq["no_route"]]
        portzone = g[g["port_in_zone"]]
        # Equation 4: rerouted sub-flows at their detour, no-route sub-flows at the largest penalty.
        pct = np.where(in_eq["no_route"], penalty, in_eq["detour_pct"])
        value_eq = in_eq["maritime_value_usd"].sum()
        rows.append({
            "chokepoint_id": r.chokepoint_id,
            "redund_edges_removed": len(removed[r.chokepoint_id]),
            "redund_nflows_affected": g["flow_id"].nunique(),
            "redund_value_affected": g["maritime_value_usd"].sum(),
            "redund_nflows_rerouted": routed["flow_id"].nunique(),
            "redund_nflows_noroute": noroute["flow_id"].nunique(),
            "redund_value_noroute": noroute["maritime_value_usd"].sum(),
            "redund_nflows_portzone": portzone["flow_id"].nunique(),
            "redund_value_portzone": portzone["maritime_value_usd"].sum(),
            "redund_median_pct": float(routed["detour_pct"].median()) if len(routed) else np.nan,
            "redund_penalty_routed_pct": weighted_mean(routed["detour_pct"], routed["maritime_value_usd"]),
            "redund_noalt_pct": 100 * noroute["maritime_value_usd"].sum() / value_eq if value_eq > 0 else np.nan,
            "redund_penalty_pct": weighted_mean(pct, in_eq["maritime_value_usd"]),
        })
    return pd.DataFrame(rows)


def log_scores(scores, shapes, log):
    """Log one line per chokepoint, sorted by the detour penalty."""
    table = shapes[["chokepoint_id", "name", "crossing_method"]].merge(scores, on="chokepoint_id")
    table = table.sort_values("redund_penalty_pct", ascending=False, na_position="last")
    log.info("Task 4.14 | route redundancy by chokepoint (flows; penalties in %):")
    log.info("  %-22s %-4s %5s %8s %8s %8s %8s %8s %8s %8s %8s", "chokepoint", "test", "edges", "affected",
             "rerouted", "no route", "port zn", "median", "routed", "no-alt", "Eq.4")
    fmt = lambda v: "null" if pd.isna(v) else f"{v:.1f}"
    for r in table.itertuples():
        log.info("  %-22s %-4s %5d %8d %8d %8d %8d %8s %8s %8s %8s", r.name, r.crossing_method,
                 r.redund_edges_removed, r.redund_nflows_affected, r.redund_nflows_rerouted,
                 r.redund_nflows_noroute, r.redund_nflows_portzone, fmt(r.redund_median_pct),
                 fmt(r.redund_penalty_routed_pct), fmt(r.redund_noalt_pct), fmt(r.redund_penalty_pct))


def merge_scores(s08, settings, log):
    """Replace S08's columns in chokepoint_scores and keep every other script's columns."""
    if common.layer_name(settings, "chokepoint_scores") not in common.list_layers(settings):
        raise ValueError("chokepoint_scores does not exist; run S06 first")
    old = pd.DataFrame(common.read_layer(settings, "chokepoint_scores"))
    old = old.drop(columns=[c for c in old.columns if c == "geometry" or c.startswith(S08_PREFIX)])
    missing = sorted(set(old["chokepoint_id"]) ^ set(s08["chokepoint_id"]))
    if missing:
        raise ValueError(f"chokepoint_scores and the closed chokepoints differ: {missing}")
    log.info("chokepoint_scores exists (%d rows); writing %d S08 columns, keeping %d others",
             len(old), len(s08.columns) - 1, len(old.columns) - 2)
    return old.merge(s08, on="chokepoint_id", how="left")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", metavar="NAME", default=None,
                        help="close only these chokepoints (names as in the screening file); writes nothing")
    return parser.parse_args()


def main(only=None):
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)

    subflows, routes, crossings, snap = read_inputs(settings, log)
    shapes = read_shapes(settings, log, only)
    tags = list(settings["routing"]["always_closed"])
    log.info("Task 4.13 | searoute restrictions in every closure: %s (routing.always_closed)", tags)

    # Task 4.12: an editable copy; searoute's cached network is never edited.
    base_network = copy.deepcopy(searoute.setup_M())
    edges = network_edges(base_network, settings, log)
    removed = edges_to_remove(edges, shapes, log)

    # Check: the copy reproduces S06's baseline; record the edges each path uses.
    pairs = list(routes[["origin_port", "dest_port"]].itertuples(index=False, name=None))
    paths = baseline_paths(base_network, pairs, snap, routes, tags, settings, log)

    # Task 4.13: one closure at a time.
    detours, lines = close_all(shapes, removed, base_network, paths, subflows, crossings, routes,
                               snap, tags, settings, log)
    log_special(detours, subflows, shapes, log)
    log_checks(detours, lines, settings, log)

    # Task 4.14: Equation 4.
    penalty = largest_penalty(detours, subflows, shapes, settings, log)
    scores = redundancy_scores(shapes, removed, detours, subflows, penalty)
    log_scores(scores, shapes, log)
    log.info("Task 4.14 | detours rows %s (baseline route_crossings rows for these chokepoints: %s)",
             f"{len(detours):,}", f"{crossings['chokepoint_id'].isin(shapes['chokepoint_id']).sum():,}")

    if only:
        log.info("Test run finished; nothing written")
        return
    common.write_layer(detours, settings, "detours", log)
    common.write_layer(lines.drop(columns="max_lat"), settings, "detour_routes", log)
    common.write_layer(merge_scores(scores, settings, log), settings, "chokepoint_scores", log)
    log.info("Run of s08_redundancy finished")


if __name__ == "__main__":
    main(parse_args().only)

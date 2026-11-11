"""S11c: web layers and the Leaflet web map (tasks 6.9 and 6.12; Decision 17).

Stage 1: the chokepoints. Stage 2a: the trade routes and the scenario switch. Stage 2b: the zones
and gate lines (docs/map/data/zones.geojson only; no database layer) and the tidied ranked list.

1. web_chokepoints (task 6.9): the scored chokepoints with the chokepoint_scores columns the
   pop-ups show, unchanged, plus short_name, the trade shares and impact_note. Written to the
   project database (layer name from settings) and as GeoJSON to paths.web_data.
1b. web_routes (task 6.9, Decision 17b): for each routing scenario, the routes cut into sea-lane
   segments (one segment per pair of consecutive vertices, both directions together), the
   sub-flow value of every route using a segment summed onto it, and runs of segments with equal
   value joined into one line. The database keeps full detail (scenario, value_usd, length_nm);
   one GeoJSON per scenario is simplified (outputs.web.route_simplify_deg). Port pairs with no
   sea route in a scenario are not drawn; the page states how many.
2. docs/index.html (task 6.12): the page template (paths.web_template) filled with the settings
   in outputs.web, the house colours in outputs.style and the score classes; Leaflet comes from
   a CDN with integrity hashes. docs/.nojekyll so GitHub Pages serves the files as they are.
3. The base map (outputs.web.basemap.kind): natural_earth writes land.geojson, Natural Earth
   countries without Antarctica (as on the paper maps), simplified, to paths.web_data; tiles
   uses the tile service in the settings and writes no file.

Only derived counts are published; no event locations (ACLED terms). Paper and poster outputs
are not read or changed. The log gives the rows and fields written, each file's size and the
total size of the published files against outputs.web.max_page_mb.
"""

import gzip
import hashlib
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import MultiPolygon, Polygon

import common
import s11_figures as s11

# chokepoint_scores columns copied unchanged into web_chokepoints (the pop-up fields).
SCORE_FIELDS = [
    "name",
    "score_rank", "score_composite",                                 # rank and score
    "score_trade", "score_conflict", "score_redund",                 # components, 0-1
    "trade_value_base", "trade_value_redsea",                        # trade, US$
    "viol_n_250", "tens_n_250",                                      # events within the main ring
    "redund_penalty_routed_pct", "redund_noalt_pct",                 # detour and no-alternative share
    "redund_nflows_affected", "score_redund_filled",
    "impact_days", "impact_days_low", "impact_days_high",            # reroute impact (Decision 13)
    "impact_voyage_cost", "impact_voyage_cost_low", "impact_voyage_cost_high",
    "impact_holding_cost", "impact_holding_cost_low", "impact_holding_cost_high",
    "score_trade_redsea", "score_composite_redsea",                  # Red Sea scenario (task 5.4)
    "score_rank_redsea", "score_closed_redsea",
    "sens_rank_min", "sens_rank_max",                                # sensitivity (task 5.7)
    "sens_grid_rank_min", "sens_grid_rank_max",
]

TEMPLATE_KEYS = ["TITLE", "LEAFLET_BASE", "LEAFLET_CSS_SRI", "LEAFLET_JS_SRI", "BACKGROUND", "CONFIG"]


# ---------------------------------------------------------------------
# Task 6.9: web_chokepoints
# ---------------------------------------------------------------------

def check_fields(settings):
    """The pop-up fields use the main ring (conflict.ring_main_km); stop if the settings moved it."""
    ring = settings["conflict"]["ring_main_km"]
    for field in ("viol_n_250", "tens_n_250"):
        if not field.endswith(f"_{ring}"):
            raise ValueError(f"{field} is not the main ring ({ring} km); update SCORE_FIELDS")


def total_maritime_value(settings, log):
    """Sum of maritime_value_usd over all sub-flows: the denominator of the trade shares."""
    subflows = pd.DataFrame(common.read_layer(settings, "subflows"))
    total = float(subflows["maritime_value_usd"].sum())
    log.info("Task 6.9 | trade share denominator: %s sub-flows, US$ %s maritime REE trade",
             f"{len(subflows):,}", f"{total:,.0f}")
    return total


def web_chokepoints(settings, log):
    """Task 6.9: one point per scored chokepoint with its pop-up fields."""
    points = common.read_layer(settings, "chokepoints")
    scores = pd.DataFrame(common.read_layer(settings, "chokepoint_scores"))
    missing = [f for f in SCORE_FIELDS if f not in scores.columns]
    if missing:
        raise ValueError(f"chokepoint_scores lacks {missing}; run S06-S10 first")
    web = points[["chokepoint_id", "geometry"]].merge(scores[["chokepoint_id"] + SCORE_FIELDS],
                                                      on="chokepoint_id")
    web = gpd.GeoDataFrame(web, geometry="geometry", crs=points.crs)
    common.log_filter(log, "Task 6.9 | chokepoints with scores", len(web), len(points) - len(web),
                      "not included in the study (no chokepoint_scores row)")

    short = settings["outputs"]["poster"]["short_names"]
    web.insert(2, "short_name", web["chokepoint_id"].map(short))
    if web["short_name"].isna().any():
        raise ValueError("outputs.poster.short_names lacks " + ", ".join(web.loc[web["short_name"].isna(), "name"]))
    total = total_maritime_value(settings, log)
    web["trade_share_pct"] = 100 * web["trade_value_base"] / total
    web["trade_share_redsea_pct"] = 100 * web["trade_value_redsea"] / total
    # Why a chokepoint has no reroute cost (Decision 13); null where it has one.
    notes = web.apply(s11.impact_note, axis=1)
    web["impact_note"] = notes.where(notes != "", None)
    web = web.sort_values("score_rank").reset_index(drop=True)

    top = web.nsmallest(settings["outputs"]["web"]["top_n_labels"], "score_rank")
    log.info("Task 6.9 | top %d (numbered): %s", len(top),
             ", ".join(f"{r} {n} {s:.3f}" for r, n, s in zip(top["score_rank"], top["short_name"], top["score_composite"])))
    log.info("Task 6.9 | Red Sea diversion closed (hollow): %s",
             ", ".join(web.loc[web["score_closed_redsea"].astype(bool), "short_name"]))
    log.info("Task 6.9 | no reroute cost: %s",
             ", ".join(f"{n} ({note})" for n, note in zip(web["short_name"], web["impact_note"]) if pd.notna(note)))
    log.info("Task 6.9 | trade shares: largest %s %.1f%%; sum over chokepoints %.1f%% "
             "(over 100%% because routes cross several chokepoints)",
             web.loc[web["trade_share_pct"].idxmax(), "short_name"], web["trade_share_pct"].max(),
             web["trade_share_pct"].sum())
    return web


def write_geojson(gdf, path, settings, log, decimals=None):
    """GeoJSON (RFC 7946) with rounded coordinates and numbers, for the web page."""
    web = settings["outputs"]["web"]
    out = gdf.copy()
    floats = out.select_dtypes(include="float").columns
    out[floats] = out[floats].round(web["value_decimals"] if decimals is None else decimals)
    path.parent.mkdir(parents=True, exist_ok=True)
    pyogrio.write_dataframe(out, path, driver="GeoJSON",
                            layer_options={"COORDINATE_PRECISION": web["coord_decimals"], "RFC7946": "YES"})
    log.info("Wrote %s (%d features, %d fields, %.1f KB)", path.relative_to(common.ROOT).as_posix(),
             len(out), len(out.columns) - 1, path.stat().st_size / 1024)


def polygon_parts(geom):
    """The polygons of a shape as one MultiPolygon (None if simplifying left none)."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    parts = []
    for part in getattr(geom, "geoms", []):      # MultiPolygon or GeometryCollection
        polygons = polygon_parts(part)
        if polygons is not None:
            parts.extend(polygons.geoms)
    return MultiPolygon(parts) if parts else None


def write_land(settings, path, log):
    """Base map land and borders (basemap.kind natural_earth): the paper maps' Natural Earth
    countries (s11.read_basemap, no Antarctica), simplified for the web."""
    tol = settings["outputs"]["web"]["basemap"]["land_simplify_deg"]
    land = s11.read_basemap(settings, log)[["geometry"]].copy()
    land["geometry"] = land.geometry.simplify(tol, preserve_topology=True).map(polygon_parts)
    kept = land[land.geometry.notna()]
    common.log_filter(log, f"Task 6.12 | base map countries simplified to {tol} deg", len(kept),
                      len(land) - len(kept), "no polygon left after simplifying")
    pyogrio.write_dataframe(kept, path, driver="GeoJSON",
                            layer_options={"COORDINATE_PRECISION": settings["outputs"]["web"]["coord_decimals"]})
    log.info("Wrote %s (%d countries, %.1f KB)", path.relative_to(common.ROOT).as_posix(), len(kept),
             path.stat().st_size / 1024)


# ---------------------------------------------------------------------
# Task 6.9 (Stage 2a): web_routes
# ---------------------------------------------------------------------

PAIR = ["origin_port", "dest_port"]


def usd_text(value):
    """Dollars as the page writes them (the page's usd()): US$ 134 K, US$ 2.15 M."""
    a = abs(value)
    if a >= 1e9:
        return f"US$ {value / 1e9:.2f} B"
    if a >= 1e6:
        return f"US$ {value / 1e6:.{2 if a < 1e7 else 1}f} M"
    if a >= 1e3:
        return f"US$ {value / 1e3:.{1 if a < 1e5 else 0}f} K"
    return f"US$ {value:,.0f}"


def gc_length_nm(lines, settings):
    """Great-circle length (nm) of each line in an array: the sum over consecutive vertices."""
    coords, idx = shapely.get_coordinates(lines, return_index=True)
    same = idx[:-1] == idx[1:]
    km = common.great_circle_km(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1], settings)
    return np.bincount(idx[:-1][same], weights=km[same], minlength=len(lines)) / common.KM_PER_NM


def pair_values(settings, log):
    """Sub-flow value per port pair (routed sub-flows only), the routes and the totals."""
    routes = common.read_layer(settings, "routes")
    subflows = pd.DataFrame(common.read_layer(settings, "subflows"))
    same = subflows["same_port"].fillna(False).astype(bool)
    pairs = (subflows[~same].groupby(PAIR, as_index=False)
             .agg(value_usd=("maritime_value_usd", "sum"), n_subflows=("subflow_id", "size")))
    totals = {"all": float(subflows["maritime_value_usd"].sum()),
              "same_port": float(subflows.loc[same, "maritime_value_usd"].sum()), "n_same_port": int(same.sum())}
    log.info("Task 6.9 | routes: %s rows; sub-flows %s (US$ %s), of which %d same-port sub-flows "
             "(US$ %s) have no route in any scenario and are not drawn", f"{len(routes):,}",
             f"{len(subflows):,}", f"{totals['all']:,.0f}", totals["n_same_port"], f"{totals['same_port']:,.0f}")
    return routes, pairs, totals


def no_route_summary(pairs, routed, ports, log, scenario):
    """Port pairs with sub-flows but no route in this scenario, and the ports cut off: ports
    that no route of this scenario reaches. Every no-route pair must start or end at one."""
    keys = routed[PAIR].drop_duplicates()
    missing = pairs.merge(keys, on=PAIR, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    reached = set(keys["origin_port"]) | set(keys["dest_port"])
    cut_off = sorted((set(missing["origin_port"]) | set(missing["dest_port"])) - reached)
    unexplained = ~(missing["origin_port"].isin(cut_off) | missing["dest_port"].isin(cut_off))
    if unexplained.any():
        raise ValueError(f"{scenario}: {int(unexplained.sum())} no-route port pairs touch no cut-off port; "
                         "outputs.web.no_route_note would be wrong")
    names = ports.set_index("port_id").loc[cut_off, "name"].tolist()
    summary = {"pairs": len(missing), "n_subflows": int(missing["n_subflows"].sum()),
               "value": float(missing["value_usd"].sum()), "ports": names}
    log.info("Task 6.9 | %s | no sea route: %d port pairs, %d sub-flows, US$ %s (not drawn); every one "
             "starts or ends at one of %d cut-off ports: %s", scenario, summary["pairs"], summary["n_subflows"],
             f"{summary['value']:,.0f}", len(names), ", ".join(names) or "none")
    return summary


def route_segments(routed):
    """Each route cut into segments between consecutive vertices, with the route's value.
    Both directions become one segment: the endpoints are put in a fixed order (west, then south, first)."""
    parts = routed[["value_usd", "geometry"]].explode(index_parts=False)
    coords, idx = shapely.get_coordinates(parts.geometry.values, return_index=True)
    keep = (idx[:-1] == idx[1:]) & (coords[:-1] != coords[1:]).any(axis=1)   # same part, not zero length
    a, b = coords[:-1][keep], coords[1:][keep]
    value = parts["value_usd"].to_numpy()[idx[:-1][keep]]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    p, q = np.where(swap[:, None], b, a), np.where(swap[:, None], a, b)
    return pd.DataFrame({"x0": p[:, 0], "y0": p[:, 1], "x1": q[:, 0], "y1": q[:, 1], "value_usd": value})


def antimeridian_jumps(lines):
    """Per line, the positions where consecutive vertices are more than 180 degrees of longitude apart."""
    coords, idx = shapely.get_coordinates(lines, return_index=True)
    jump = (idx[:-1] == idx[1:]) & (np.abs(np.diff(coords[:, 0])) > 180)
    return coords, idx, jump


def split_jumps(lines):
    """Cut lines wherever consecutive vertices are more than 180 degrees of longitude apart, so no
    line is drawn across the whole map. Returns the lines, the source line of each, and the number of cuts."""
    coords, idx, jump = antimeridian_jumps(lines)
    if not jump.any():
        return lines, np.arange(len(lines)), 0
    # A new part starts after every jump and at every new line.
    start = np.r_[True, (idx[1:] != idx[:-1]) | jump]
    part = np.cumsum(start) - 1
    counts = np.bincount(part)
    ok = counts[part] >= 2                         # a part needs two vertices
    new = shapely.linestrings(coords[ok], indices=np.unique(part[ok], return_inverse=True)[1])
    source = idx[start][counts >= 2]
    return new, source, int(jump.sum())


def merge_segments(segments, settings):
    """Sum the value per segment, round it, and join runs of segments with equal value into lines."""
    web = settings["outputs"]["web"]
    seg = (segments.groupby(["x0", "y0", "x1", "y1"], as_index=False, sort=False)["value_usd"].sum())
    seg["value_usd"] = seg["value_usd"].round(web["route_join_decimals"])
    seg = seg.sort_values("value_usd", kind="stable").reset_index(drop=True)
    group, values = pd.factorize(seg["value_usd"], sort=True)
    lines = shapely.linestrings(np.stack([seg[["x0", "y0"]].to_numpy(), seg[["x1", "y1"]].to_numpy()], axis=1))
    joined = shapely.line_merge(shapely.multilinestrings(lines, indices=group))
    parts, which = shapely.get_parts(joined, return_index=True)
    return len(seg), parts, np.asarray(values)[which]


def busiest(lines, values, points, settings):
    """Value of the busiest line and the chokepoint nearest its midpoint (for the log)."""
    i = int(np.argmax(values))
    mid = shapely.line_interpolate_point(lines[i], 0.5, normalized=True)
    km = common.great_circle_km(points.geometry.x, points.geometry.y, mid.x, mid.y, settings)
    j = int(np.argmin(km))
    return (f"US$ {values[i]:,.0f} at {mid.y:.2f}, {mid.x:.2f} "
            f"({km[j]:,.0f} km from {points['name'].iloc[j]})")


def web_routes(settings, data_dir, log):
    """Task 6.9: web_routes in the database (full detail) and one simplified GeoJSON per scenario.
    Returns the layer, the files and each scenario's no-route summary."""
    web = settings["outputs"]["web"]
    routes, pairs, totals = pair_values(settings, log)
    ports = common.read_layer(settings, "ports")
    points = common.read_layer(settings, "chokepoints")
    layers, files, no_route = [], {}, {}
    for scenario in settings["routing"]["scenarios"]:
        here = routes[routes["scenario"] == scenario]
        routed = here.merge(pairs, on=PAIR, how="inner")
        common.log_filter(log, f"Task 6.9 | {scenario} | routes used by sub-flows", len(routed),
                          len(here) - len(routed), "no sub-flow uses the route")
        no_route[scenario] = no_route_summary(pairs, routed, ports, log, scenario)
        routed_value = float(routed["value_usd"].sum())
        closure = routed_value + no_route[scenario]["value"] + totals["same_port"] - totals["all"]
        log.info("Task 6.9 | %s | routed sub-flow value US$ %s; routed + no route + same port - all "
                 "sub-flows = US$ %.6f", scenario, f"{routed_value:,.2f}", closure)
        if abs(closure) > 1:
            raise ValueError(f"{scenario}: routed, no-route and same-port values do not add up to the sub-flows")

        # Reference: sum of value x length over the unmerged routes.
        reference = float((routed["value_usd"] * gc_length_nm(routed.geometry.values, settings)).sum())
        segments = route_segments(routed)
        n_unique, lines, values = merge_segments(segments, settings)
        lines, source, cuts = split_jumps(lines)
        values = values[source]
        length = gc_length_nm(lines, settings)
        ratio = float((values * length).sum()) / reference
        log.info("Task 6.9 | %s | %s route segments -> %s distinct sea-lane segments -> %s lines after "
                 "joining equal values; lines cut at a jump over 180 deg: %d", scenario, f"{len(segments):,}",
                 f"{n_unique:,}", f"{len(lines):,}", cuts)
        log.info("Task 6.9 | %s | sum of value x length, lines / unmerged routes = %.6f", scenario, ratio)
        if abs(ratio - 1) > settings["check_db"]["web_routes_tolerance"]:
            raise ValueError(f"{scenario}: merged lines do not keep value x length (ratio {ratio:.9f})")
        log.info("Task 6.9 | %s | busiest segment: %s", scenario, busiest(lines, values, points, settings))
        layers.append(gpd.GeoDataFrame({"scenario": scenario, "value_usd": values, "length_nm": length},
                                       geometry=lines, crs=routes.crs))

        # Web file: simplified, thin lines first so busy lines draw on top.
        tol = web["route_simplify_deg"]
        small = shapely.simplify(lines, tol, preserve_topology=True)
        small, source, cuts_small = split_jumps(small)
        ratio_small = float((values[source] * gc_length_nm(small, settings)).sum()) / reference
        decimals = web["route_value_decimals"]
        value_out = np.round(values[source], decimals)
        out = gpd.GeoDataFrame({"value_usd": value_out.astype("int64") if decimals == 0 else value_out},
                               geometry=small, crs=routes.crs).sort_values("value_usd", kind="stable")
        path = data_dir / f"routes_{scenario}.geojson"
        write_geojson(out, path, settings, log, decimals=decimals)
        # Check the written file: no jump over 180 deg once coordinates are rounded.
        written = gpd.read_file(path)
        _, _, jump = antimeridian_jumps(written.geometry.values)
        log.info("Task 6.9 | %s | web file simplified to %s deg: value x length ratio %.6f (simplifying "
                 "shortens lines); %d further cuts; jumps over 180 deg left in the file: %d; "
                 "%.1f KB, %.1f KB gzipped", scenario, tol, ratio_small, cuts_small, int(jump.sum()),
                 path.stat().st_size / 1024, len(gzip.compress(path.read_bytes())) / 1024)
        if jump.any():
            raise ValueError(f"{path.name} has lines that jump across the map at +-180")
        files[scenario] = path
    layer = pd.concat(layers, ignore_index=True)
    return gpd.GeoDataFrame(layer, geometry="geometry", crs=routes.crs), files, no_route


# ---------------------------------------------------------------------
# Stage 2b: chokepoint zones and gate lines (Decisions 11 and 12)
# ---------------------------------------------------------------------

def write_zones(settings, chokepoint_ids, path, log):
    """zones.geojson: the zone circles of the scored chokepoints, simplified, and the gate lines.
    A gated strait is drawn by its gate line only; its circle is for the paper maps."""
    zones = common.read_layer(settings, "zones")
    gates = common.read_layer(settings, "gates")
    zones = zones[zones["chokepoint_id"].isin(chokepoint_ids)]
    gated = zones["chokepoint_id"].isin(gates["chokepoint_id"])
    common.log_filter(log, "Stage 2b | zone circles drawn", int((~gated).sum()), int(gated.sum()),
                      "gated strait: drawn by its gate line (Decision 12)")
    missing = set(gates["chokepoint_id"]) - set(chokepoint_ids)
    if missing:
        raise ValueError(f"gates layer has chokepoints that are not scored: {sorted(missing)}")
    tol = settings["outputs"]["web"]["zone_simplify_deg"]
    circles = zones.loc[~gated, ["chokepoint_id", "geometry"]].assign(kind="zone")
    circles["geometry"] = circles.geometry.simplify(tol, preserve_topology=True)
    lines = gates[["chokepoint_id", "geometry"]].assign(kind="gate")
    out = gpd.GeoDataFrame(pd.concat([circles, lines], ignore_index=True), geometry="geometry", crs=zones.crs)
    out = out[["kind", "geometry"]]
    pyogrio.write_dataframe(out, path, driver="GeoJSON", geometry_type="Unknown",
                            layer_options={"COORDINATE_PRECISION": settings["outputs"]["web"]["coord_decimals"],
                                           "RFC7946": "YES"})
    vertices = int(shapely.get_num_coordinates(out.geometry.values).sum())
    log.info("Stage 2b | wrote %s (%d circles simplified to %s deg, %d gate lines, %s vertices, %.1f KB, "
             "%.1f KB gzipped)", path.relative_to(common.ROOT).as_posix(), len(circles), tol, len(lines),
             f"{vertices:,}", path.stat().st_size / 1024, len(gzip.compress(path.read_bytes())) / 1024)


def no_route_text(settings, summary):
    """The panel note for a scenario with port pairs that have no sea route (None if there are none)."""
    if not summary["pairs"]:
        return None
    names = summary["ports"]
    ports = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return settings["outputs"]["web"]["no_route_note"].format(
        pairs=f"{summary['pairs']:,}", value=usd_text(summary["value"]), ports=ports)


# ---------------------------------------------------------------------
# Task 6.12: the web page
# ---------------------------------------------------------------------

def page_config(settings, data_files, no_route):
    """Everything the page script needs, from the settings (one JSON block in the page)."""
    out = settings["outputs"]
    web = out["web"]
    ring = settings["conflict"]["ring_main_km"]
    scenarios = settings["scoring"]["scenarios"]
    as_of = pd.Timestamp(web["as_of"])
    return {
        "title": web["title"],
        "subtitle": web["subtitle"],
        "author": web["author"],
        "as_of": web["as_of"],
        "as_of_text": f"Results as of {as_of.day} {as_of:%b %Y} · {web['as_of_note']}",
        "repo_url": web["repo_url"],
        "how_to_read": web["how_to_read"],
        "exposure_kinds": [[name, text.format(ring_km=ring)] for name, text in web["exposure_kinds"]],
        "red_sea": web["red_sea"],
        "caveat": web["caveat"],
        "data_note": web["data_note"],
        "credits": web["credits"],
        "ring_km": ring,
        "basemap": web["basemap"],
        "max_zoom": web["basemap"]["max_zoom"][web["basemap"]["kind"]],
        "start_view": web["start_view"],
        "max_bounds": max_bounds(web),
        "max_bounds_viscosity": web["max_bounds_viscosity"],
        "min_zoom": web["min_zoom"],
        "narrow_screen_px": web["narrow_screen_px"],
        "top_n_labels": web["top_n_labels"],
        "marker_radius_px": web["marker_radius_px"],
        "rank_min_radius_px": web["rank_min_radius_px"],
        "hit_radius_px": web["hit_radius_px"],
        "class_breaks": out["score_class_breaks"],
        "style": out["style"],
        "route": {k: web[k] for k in ("route_width_px", "route_opacity", "route_tap_tolerance_px",
                                      "route_legend", "route_popup", "route_popup_note")},
        "legend_button": web["legend_button"],
        "zones": {k: web[k] for k in ("zone_style", "zones_label", "zones_note")},
        "closed_label": web["closed_label"],
        "closed_heading": web["closed_heading"],
        "list_zoom": web["list_zoom"],
        # Each scenario's fields (scoring.scenarios suffix); the main one has no suffix.
        "scenarios": [
            {"key": key, "label": web["scenario_labels"][key],
             "score": f"score_composite_{cfg['suffix']}" if "suffix" in cfg else "score_composite",
             "rank": f"score_rank_{cfg['suffix']}" if "suffix" in cfg else "score_rank",
             "trade": f"score_trade_{cfg['suffix']}" if "suffix" in cfg else "score_trade",
             "trade_value": cfg["trade"],
             "trade_share": "trade_share_redsea_pct" if "suffix" in cfg else "trade_share_pct",
             "closed": f"score_closed_{cfg['suffix']}" if "suffix" in cfg else None,
             # Scoring scenarios share their keys with routing.scenarios (checked in main).
             "routes": f"routes_{key}",
             "no_route_note": no_route_text(settings, no_route[key])}
            for key, cfg in scenarios.items()],
        "data": data_files,
    }


def max_bounds(web):
    """Pan limits for the page: lat [south, north], the east-west band [west, east] and the free range
    used when the band is not applied. The south edge mirrors the north edge about start_view.lat in
    Web Mercator, so where the screen is taller than the area Leaflet centres it on the opening
    latitude and the opening view does not move."""
    start, b = web["start_view"], web["max_bounds"]
    if not b["west"] < start["lon"] < b["east"]:
        raise ValueError("outputs.web.max_bounds west..east must contain start_view.lon")
    mercator = lambda lat: np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    south = np.degrees(2 * np.arctan(np.exp(2 * mercator(start["lat"]) - mercator(b["north"]))) - np.pi / 2)
    return {"lat": [round(float(south), 6), b["north"]], "band": [b["west"], b["east"]],
            "free": [start["lon"] - b["lon_half_width"], start["lon"] + b["lon_half_width"]]}


def fill_template(settings, config, log):
    """docs/index.html from the template: placeholders {{KEY}} replaced, none left over.
    The settings JSON goes in last, so text in the settings is never taken for a placeholder."""
    web = settings["outputs"]["web"]
    template = common.get_path(settings, "web_template").read_text(encoding="utf-8")
    values = {
        "TITLE": web["title"],
        "LEAFLET_BASE": web["leaflet"]["base"],
        "LEAFLET_CSS_SRI": web["leaflet"]["css_sri"],
        "LEAFLET_JS_SRI": web["leaflet"]["js_sri"],
        "BACKGROUND": settings["outputs"]["style"]["background"],
        # "</" escaped so no text in the settings can close the script element.
        "CONFIG": json.dumps(config, ensure_ascii=False, indent=1).replace("</", "<\\/"),
    }
    for key in TEMPLATE_KEYS:
        token = "{{" + key + "}}"
        if token not in template:
            raise ValueError(f"Template lacks {token}")
        if key != "CONFIG":
            template = template.replace(token, values[key])
    left = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", template)) - {"{{CONFIG}}"})
    if left:
        raise ValueError(f"Template has placeholders S11c does not fill: {', '.join(left)}")
    template = template.replace("{{CONFIG}}", values["CONFIG"])
    page = common.get_path(settings, "web_export") / "index.html"
    page.write_text(template, encoding="utf-8", newline="\n")
    log.info("Task 6.12 | wrote %s (%.1f KB)", page.relative_to(common.ROOT).as_posix(), page.stat().st_size / 1024)
    return page


def page_weight(settings, files, log, when):
    """Total size of the files the page loads from this site, against outputs.web.max_page_mb
    (the limit applies to the files as stored; GitHub Pages sends them gzipped, also logged)."""
    total = sum(f.stat().st_size for f in files)
    zipped = sum(len(gzip.compress(f.read_bytes())) for f in files)
    limit = settings["outputs"]["web"]["max_page_mb"]
    log.info("Task 6.12 | files the page loads %s: %d, %.1f KB (%.1f KB gzipped as served; limit %s MB; "
             "Leaflet and basemap tiles come from CDNs and are not counted)", when, len(files), total / 1024,
             zipped / 1024, limit)
    if total > limit * 1024 * 1024:
        log.warning("Task 6.12 | published files exceed outputs.web.max_page_mb (%s MB)", limit)


def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    check_fields(settings)
    export = common.get_path(settings, "web_export")
    data_dir = common.get_path(settings, "web_data")
    unmatched = set(settings["scoring"]["scenarios"]) - set(settings["routing"]["scenarios"])
    if unmatched:
        raise ValueError(f"scoring.scenarios {sorted(unmatched)} have no routing scenario of the same name")

    # Task 6.9: web_chokepoints in the database and as GeoJSON.
    points = web_chokepoints(settings, log)
    common.write_layer(points, settings, "web_chokepoints", log)
    chokepoint_file = data_dir / "chokepoints.geojson"
    write_geojson(points, chokepoint_file, settings, log)

    # Task 6.9 (Stage 2a): web_routes in the database and one GeoJSON per scenario.
    routes, route_files, no_route = web_routes(settings, data_dir, log)
    common.write_layer(routes, settings, "web_routes", log)

    # Task 6.12: the page. Data URLs carry a hash of the file's content, so browsers fetch a file
    # again whenever it changes and keep using their copy while it does not.
    published = [chokepoint_file] + list(route_files.values())

    # Stage 2b: zone circles and gate lines, one file loaded when the box is first ticked.
    zones_file = data_dir / "zones.geojson"
    write_zones(settings, set(points["chokepoint_id"]), zones_file, log)
    published.append(zones_file)
    land_file = data_dir / "land.geojson"
    kind = settings["outputs"]["web"]["basemap"]["kind"]
    if kind == "natural_earth":
        write_land(settings, land_file, log)
        published.append(land_file)
    elif kind != "tiles":
        raise ValueError(f"outputs.web.basemap.kind must be natural_earth or tiles, not {kind}")
    elif land_file.exists():
        land_file.unlink()
        log.info("Task 6.12 | base map from tiles: removed %s", land_file.relative_to(common.ROOT).as_posix())
    log.info("Task 6.12 | base map: %s", kind)
    data_files = {f.stem: f"{f.relative_to(export).as_posix()}?v={hashlib.sha256(f.read_bytes()).hexdigest()[:10]}"
                  for f in published}
    page = fill_template(settings, page_config(settings, data_files, no_route), log)
    bounds = max_bounds(settings["outputs"]["web"])
    log.info("Task 6.12 | pan limit: latitude %.2f to %.2f (centred on the opening view in Web Mercator); "
             "longitude %g to %g while the window is narrower than that band, else %g to %g; min zoom %s",
             *bounds["lat"], *bounds["band"], *bounds["free"], settings["outputs"]["web"]["min_zoom"])
    nojekyll = export / ".nojekyll"
    nojekyll.write_text("", encoding="utf-8")
    log.info("Task 6.12 | wrote %s (GitHub Pages serves files as they are)", nojekyll.relative_to(common.ROOT).as_posix())
    # At start the page loads every file but the routes of the other scenarios (fetched on first
    # switch) and the zones (fetched when the box is first ticked).
    first = settings["scoring"]["scenarios"]
    later = [route_files[k] for k in list(first)[1:]]
    page_weight(settings, [page] + [f for f in published if f not in later + [zones_file]], log, "at start")
    page_weight(settings, [page] + [f for f in published if f != zones_file], log, "after every scenario is shown")
    page_weight(settings, [page] + published, log, "with every scenario and the zones")
    log.info("Run of s11c_web_map finished")


if __name__ == "__main__":
    main()

"""S11c: web layers and the Leaflet web map (tasks 6.9 and 6.12; Decision 17).

Stage 1 (this version): the chokepoints-only map.

1. web_chokepoints (task 6.9): the scored chokepoints with the chokepoint_scores columns the
   pop-ups show, unchanged, plus short_name, the trade shares and impact_note. Written to the
   project database (layer name from settings) and as GeoJSON to paths.web_data.
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

import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
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


def write_geojson(gdf, path, settings, log):
    """GeoJSON (RFC 7946) with rounded coordinates and numbers, for the web page."""
    web = settings["outputs"]["web"]
    out = gdf.copy()
    floats = out.select_dtypes(include="float").columns
    out[floats] = out[floats].round(web["value_decimals"])
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
# Task 6.12: the web page
# ---------------------------------------------------------------------

def page_config(settings, data_files):
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
        "caveat": web["caveat"],
        "data_note": web["data_note"],
        "credits": web["credits"],
        "ring_km": ring,
        "basemap": web["basemap"],
        "max_zoom": web["basemap"]["max_zoom"][web["basemap"]["kind"]],
        "start_view": web["start_view"],
        "narrow_screen_px": web["narrow_screen_px"],
        "top_n_labels": web["top_n_labels"],
        "marker_radius_px": web["marker_radius_px"],
        "rank_min_radius_px": web["rank_min_radius_px"],
        "hit_radius_px": web["hit_radius_px"],
        "class_breaks": out["score_class_breaks"],
        "style": out["style"],
        # Each scenario's fields (scoring.scenarios suffix); the main one has no suffix.
        "scenarios": [
            {"key": key, "label": web["scenario_labels"][key],
             "score": f"score_composite_{cfg['suffix']}" if "suffix" in cfg else "score_composite",
             "rank": f"score_rank_{cfg['suffix']}" if "suffix" in cfg else "score_rank",
             "trade": f"score_trade_{cfg['suffix']}" if "suffix" in cfg else "score_trade",
             "trade_value": cfg["trade"],
             "trade_share": "trade_share_redsea_pct" if "suffix" in cfg else "trade_share_pct",
             "closed": f"score_closed_{cfg['suffix']}" if "suffix" in cfg else None}
            for key, cfg in scenarios.items()],
        "data": data_files,
    }


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


def page_weight(settings, files, log):
    """Total size of the files the page loads from this site, against outputs.web.max_page_mb."""
    total = sum(f.stat().st_size for f in files)
    limit = settings["outputs"]["web"]["max_page_mb"]
    log.info("Task 6.12 | published files the page loads: %d, %.1f KB (limit %s MB; Leaflet and "
             "basemap tiles come from CDNs and are not counted)", len(files), total / 1024, limit)
    if total > limit * 1024 * 1024:
        log.warning("Task 6.12 | published files exceed outputs.web.max_page_mb (%s MB)", limit)


def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    check_fields(settings)
    export = common.get_path(settings, "web_export")
    data_dir = common.get_path(settings, "web_data")
    as_of = settings["outputs"]["web"]["as_of"]

    # Task 6.9: web_chokepoints in the database and as GeoJSON.
    points = web_chokepoints(settings, log)
    common.write_layer(points, settings, "web_chokepoints", log)
    chokepoint_file = data_dir / "chokepoints.geojson"
    write_geojson(points, chokepoint_file, settings, log)

    # Task 6.12: the page. Data URLs carry the as-of date so browsers fetch new data after an update.
    published = [chokepoint_file]
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
    data_files = {f.stem: f"{f.relative_to(export).as_posix()}?v={as_of}" for f in published}
    page = fill_template(settings, page_config(settings, data_files), log)
    nojekyll = export / ".nojekyll"
    nojekyll.write_text("", encoding="utf-8")
    log.info("Task 6.12 | wrote %s (GitHub Pages serves files as they are)", nojekyll.relative_to(common.ROOT).as_posix())
    page_weight(settings, [page] + published, log)
    log.info("Run of s11c_web_map finished")


if __name__ == "__main__":
    main()

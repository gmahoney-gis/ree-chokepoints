#!/usr/bin/env python3
"""
maritime_network_chokepoints.py
-------------------------------
Builds GIS-ready vector layers of the searoute maritime network (Eurostat
MARNET) and shows where it passes through the IMF PortWatch chokepoints.

Output: maritime_output/maritime_network_chokepoints.gpkg with these layers
  marnet_network        every network segment (line), with length_km and the
                        searoute passage tag (suez, malacca, ...)
  chokepoints           the 28 PortWatch points
  chokepoint_zones      a circle of --zone-km radius around each point (polygon)
  network_at_chokepoint network segments clipped to each zone (line), tagged
                        with cp_id / cp_name
  example_routes        illustrative shortest routes between major ports,
                        with a baseline and a Red Sea closure scenario (line)
plus GeoJSON copies (maritime_output/geojson/) for web upload, and
chokepoint_network_check.csv: for each chokepoint, how far it is from the
network and how many segments cross its zone.

Usage
  pip install searoute geopandas
  python maritime_network_chokepoints.py
  python maritime_network_chokepoints.py --zone-km 50 --no-routes
"""

import argparse
import json
import math
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import searoute as sr
from pyproj import CRS, Geod
from shapely.affinity import translate
from shapely.geometry import LineString, MultiLineString, Point, box, shape
from shapely.ops import transform
from pyproj import Transformer

GEOD = Geod(ellps="WGS84")
WORLD = box(-180, -90, 180, 90)

# Illustrative port pairs (lon, lat). These are NOT your REE trade flows yet;
# they show how the network routes major trade lanes through chokepoints.
PORTS = {
    "Shanghai":    (122.05, 31.00),
    "Kuantan":     (103.45, 3.95),
    "Fremantle":   (115.70, -32.05),
    "Rotterdam":   (3.90, 51.98),
    "Hamburg":     (8.40, 54.00),
    "Yokohama":    (139.75, 35.35),
    "Busan":       (129.10, 35.00),
    "Los Angeles": (-118.25, 33.70),
    "Houston":     (-94.70, 29.30),
}
PAIRS = [
    ("Shanghai", "Rotterdam"), ("Shanghai", "Hamburg"), ("Shanghai", "Houston"),
    ("Shanghai", "Los Angeles"), ("Shanghai", "Yokohama"), ("Shanghai", "Busan"),
    ("Kuantan", "Yokohama"), ("Kuantan", "Rotterdam"), ("Fremantle", "Kuantan"),
]
SCENARIOS = {
    "baseline": ["northwest"],                                   # searoute default
    "red_sea_closed": ["northwest", "babalmandab", "suez"],
    # The shortest path around a Red Sea closure can go via Panama; carriers in
    # 2024-25 actually diverted round the Cape, so also test that case.
    "red_sea_and_panama_closed": ["northwest", "babalmandab", "suez", "panama"],
}


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def fix_antimeridian(geom):
    """MARNET stores some Pacific segments with longitudes beyond +/-180.
    Split them at the 180th meridian and shift the overflow back into range."""
    if geom is None or geom.is_empty:
        return geom
    minx, _, maxx, _ = geom.bounds
    if minx >= -180 and maxx <= 180:
        return geom
    parts = []
    for shift, window in ((0, WORLD), (-360, box(180, -90, 540, 90)), (360, box(-540, -90, -180, 90))):
        piece = geom.intersection(window)
        if not piece.is_empty:
            parts.append(translate(piece, xoff=shift))
    lines = []
    for p in parts:
        if isinstance(p, LineString):
            lines.append(p)
        elif hasattr(p, "geoms"):
            lines.extend(g for g in p.geoms if isinstance(g, LineString))
    lines = [l for l in lines if l.length > 0]
    if not lines:
        return None
    return lines[0] if len(lines) == 1 else MultiLineString(lines)


def geodesic_km(geom):
    if geom is None or geom.is_empty:
        return 0.0
    return round(GEOD.geometry_length(geom) / 1000.0, 2)


def geodesic_circle(lon, lat, radius_km):
    """True-distance circle: buffer in a projection centred on the point."""
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    to_m = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True).transform
    to_ll = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True).transform
    circle = transform(to_m, Point(lon, lat)).buffer(radius_km * 1000, quad_segs=32)
    return fix_antimeridian_polygon(transform(to_ll, circle))


def fix_antimeridian_polygon(poly):
    minx, _, maxx, _ = poly.bounds
    if maxx - minx < 180:
        return poly
    # Circle wrapped across 180: rebuild with continuous longitudes, then split.
    from shapely.geometry import Polygon
    xs, ys = poly.exterior.coords.xy
    xs = [x + 360 if x < 0 else x for x in xs]
    p = Polygon(zip(xs, ys))
    east = p.intersection(box(0, -90, 180, 90))
    west = translate(p.intersection(box(180, -90, 360, 90)), xoff=-360)
    return east.union(west)


def nearest_network_km(lon, lat, network):
    """Approximate distance (km) from a point to the nearest network segment."""
    cand = network.cx[lon - 5:lon + 5, lat - 5:lat + 5]
    if cand.empty:
        cand = network
    best = math.inf
    pt = Point(lon, lat)
    for g in cand.geometry:
        np_ = g.interpolate(g.project(pt)) if isinstance(g, LineString) else \
            min((s.interpolate(s.project(pt)) for s in g.geoms), key=lambda q: q.distance(pt))
        _, _, d = GEOD.inv(lon, lat, np_.x, np_.y)
        best = min(best, d / 1000.0)
    return round(best, 1)


# ----------------------------------------------------------------------------
# Labeled preview map (PNG)
# ----------------------------------------------------------------------------
# Label offsets in points (dx, dy, horizontal alignment) for crowded areas.
# Any chokepoint not listed gets the default: just right of the point.
LABEL_OFFSETS = {
    "Malacca Strait":       (-12, -10, "right"),
    "Sunda Strait":         (-12, -12, "right"),
    "Lombok Strait":        (-6, -18, "right"),
    "Ombai Strait":         (6, -20, "left"),
    "Torres Strait":        (10, -6, "left"),
    "Makassar Strait":      (22, -6, "left"),
    "Balabac Strait":       (30, 2, "left"),
    "Mindoro Strait":       (-30, 4, "right"),
    "Luzon Strait":         (24, 0, "left"),
    "Taiwan Strait":        (-28, 6, "right"),
    "Bohai Strait":         (-14, 6, "right"),
    "Korea Strait":         (10, -12, "left"),
    "Tsugaru Strait":       (10, 4, "left"),
    "Yucatan Channel":      (-10, 10, "right"),
    "Windward Passage":     (0, 16, "center"),
    "Mona Passage":         (12, -6, "left"),
    "Panama Canal":         (-10, -12, "right"),
    "Dover Strait":         (-10, 8, "right"),
    "Oresund Strait":       (8, 8, "left"),
    "Gibraltar Strait":     (-10, -12, "right"),
    "Bosporus Strait":      (-8, 10, "right"),
    "Kerch Strait":         (10, 6, "left"),
    "Suez Canal":           (-10, -10, "right"),
    "Bab el-Mandeb Strait": (-10, -12, "right"),
    "Strait of Hormuz":     (10, 4, "left"),
}


def make_preview_png(network, zones, at_cp, cp_pts, routes, path):
    """World map of the network, chokepoint zones, intersections and routes,
    with every chokepoint labeled. For slides/review, not for GIS."""
    import matplotlib
    matplotlib.use("Agg")                      # draw to file, no window needed
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(18, 9))
    network.plot(ax=ax, color="#a9c0c4", linewidth=0.3, zorder=1)
    if routes is not None:
        styles = {"baseline": ("#1f3c88", "-", 1.4),
                  "red_sea_closed": ("#7b3fa0", ":", 1.2),
                  "red_sea_and_panama_closed": ("#e08a00", "--", 1.2)}
        for scen, (col, ls, lw) in styles.items():
            sub = routes[routes.scenario == scen]
            if not sub.empty:
                sub.plot(ax=ax, color=col, linestyle=ls, linewidth=lw, zorder=2)
    zones.plot(ax=ax, facecolor="none", edgecolor="#c0392b", linewidth=1.2, zorder=3)
    at_cp.plot(ax=ax, color="#c0392b", linewidth=3, zorder=4)
    cp_pts.plot(ax=ax, color="black", markersize=10, zorder=5)

    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    for r in cp_pts.itertuples():
        dx, dy, ha = LABEL_OFFSETS.get(r.cp_name, (6, 3, "left"))
        ax.annotate(r.cp_name, xy=(r.lon, r.lat), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=7.5, path_effects=halo, zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.5,
                                    shrinkA=0, shrinkB=2) if abs(dx) > 12 or abs(dy) > 12 else None)

    legend = [Line2D([0], [0], color="#a9c0c4", lw=1, label="Maritime network (MARNET)"),
              Line2D([0], [0], color="#c0392b", lw=2, label="Network inside chokepoint zone"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                     markeredgecolor="#c0392b", markersize=9, label="Chokepoint zone"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
                     markersize=5, label="PortWatch chokepoint")]
    if routes is not None:
        legend += [Line2D([0], [0], color="#1f3c88", lw=1.5, label="Example route: baseline"),
                   Line2D([0], [0], color="#7b3fa0", lw=1.5, ls=":", label="Red Sea closed"),
                   Line2D([0], [0], color="#e08a00", lw=1.5, ls="--", label="Red Sea + Panama closed")]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 80)
    ax.set_title("Maritime routing network and IMF PortWatch chokepoints", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview map: {Path(path).resolve()}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="PortWatch_IMF_Chokepoints.csv")
    ap.add_argument("--out", default="maritime_output")
    ap.add_argument("--zone-km", type=float, default=50,
                    help="radius of each chokepoint zone used for the intersection (default 50)")
    ap.add_argument("--zone-override", default="",
                    help='per-chokepoint radius, e.g. "Luzon Strait=150,Torres Strait=100"')
    ap.add_argument("--no-routes", action="store_true", help="skip the example routes")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "geojson").mkdir(parents=True, exist_ok=True)
    gpkg = out / "maritime_network_chokepoints.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    # 1. Network --------------------------------------------------------------
    net_path = Path(sr.__file__).parent / "data" / "marnet_searoute.geojson"
    with open(net_path) as fh:
        fc = json.load(fh)
    rows = []
    for i, f in enumerate(fc["features"]):
        g = fix_antimeridian(shape(f["geometry"]))
        if g is None:
            continue
        rows.append({"edge_id": i, "passage": (f.get("properties") or {}).get("passage"),
                     "length_km": geodesic_km(g), "geometry": g})
    network = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    print(f"Network: {len(network)} segments, {network.length_km.sum():,.0f} km in total")

    # 2. Chokepoints and zones -----------------------------------------------
    csv = Path(args.csv)
    if not csv.exists() and Path("PortWatch IMF Chokepoints.csv").exists():
        csv = Path("PortWatch IMF Chokepoints.csv")
    cps = pd.read_csv(csv)[["portid", "portname", "lat", "lon",
                                 "vessel_count_total"]].dropna(subset=["lat", "lon"])
    cps = cps.rename(columns={"portid": "cp_id", "portname": "cp_name"})
    cp_pts = gpd.GeoDataFrame(cps, geometry=gpd.points_from_xy(cps.lon, cps.lat), crs="EPSG:4326")
    override = {}
    for item in filter(None, (x.strip() for x in args.zone_override.split(","))):
        name, km = item.rsplit("=", 1)
        override[name.strip().lower()] = float(km)
    cps["zone_km"] = [override.get(n.lower(), args.zone_km) for n in cps.cp_name]
    cp_pts["zone_km"] = cps["zone_km"].values
    zones = cp_pts.drop(columns="geometry").copy()
    zones = gpd.GeoDataFrame(zones, geometry=[geodesic_circle(r.lon, r.lat, r.zone_km)
                                              for r in cps.itertuples()], crs="EPSG:4326")

    # 3. Intersection ---------------------------------------------------------
    at_cp = gpd.overlay(network, zones[["cp_id", "cp_name", "geometry"]],
                        how="intersection", keep_geom_type=True)
    at_cp["clip_km"] = at_cp.geometry.apply(geodesic_km)
    print(f"Network segments inside chokepoint zones: {len(at_cp)}")

    check = []
    for r in cps.itertuples():
        sub = at_cp[at_cp.cp_id == r.cp_id]
        dist = nearest_network_km(r.lon, r.lat, network)
        check.append({"cp_id": r.cp_id, "cp_name": r.cp_name, "zone_km": r.zone_km,
                      "nearest_network_km": dist,
                      "segments_in_zone": int(len(sub)),
                      "network_km_in_zone": round(sub.clip_km.sum(), 1),
                      "searoute_passage": ", ".join(sorted({p for p in sub.passage.dropna()})) or "",
                      "on_network": "yes" if len(sub) else "NO"})
    check = pd.DataFrame(check)
    check.to_csv(out / "chokepoint_network_check.csv", index=False)
    cp_pts = cp_pts.merge(check[["cp_id", "nearest_network_km", "segments_in_zone", "on_network"]], on="cp_id")
    zones = zones.merge(check[["cp_id", "segments_in_zone", "on_network"]], on="cp_id")

    # 4. Example routes -------------------------------------------------------
    routes = None
    if not args.no_routes:
        rrows = []
        for scen, restr in SCENARIOS.items():
            for o, d in PAIRS:
                feat = sr.searoute(PORTS[o], PORTS[d], units="km", restrictions=restr,
                                   append_orig_dest=True, return_passages=True)
                coords = feat["geometry"]["coordinates"]
                if len(coords) < 2:
                    continue
                g = fix_antimeridian(LineString(coords))
                rrows.append({"route": f"{o} - {d}", "origin": o, "dest": d, "scenario": scen,
                              "length_km": round(feat["properties"]["length"], 0),
                              "passages": ", ".join(feat["properties"].get("traversed_passages", [])),
                              "geometry": g})
        routes = gpd.GeoDataFrame(rrows, crs="EPSG:4326")
        # Which PortWatch chokepoints does each route pass through?
        hits = gpd.sjoin(routes[["route", "scenario", "geometry"]], zones[["cp_name", "geometry"]],
                         predicate="intersects", how="left")
        cp_lists = hits.groupby(["route", "scenario"])["cp_name"].apply(
            lambda s: "; ".join(sorted(s.dropna().unique())))
        routes["chokepoints"] = [cp_lists.get((r, s), "") for r, s in zip(routes.route, routes.scenario)]
        base = routes[routes.scenario == "baseline"].set_index("route").length_km
        routes["detour_pct"] = [round(100 * (l / base[r] - 1), 1) if r in base else None
                                for r, l in zip(routes.route, routes.length_km)]
        routes.drop(columns="geometry").to_csv(out / "example_routes.csv", index=False)
        print(f"Example routes: {len(routes)}")

    # 5. Write ------------------------------------------------------------------
    layers = {"marnet_network": network, "chokepoints": cp_pts, "chokepoint_zones": zones,
              "network_at_chokepoint": at_cp}
    if routes is not None:
        layers["example_routes"] = routes
    for name, gdf in layers.items():
        gdf.to_file(gpkg, layer=name, driver="GPKG")
        gdf.to_file(out / "geojson" / f"{name}.geojson", driver="GeoJSON")

    print("\nChokepoints and the network:")
    print(check.to_string(index=False))
    missing = check[check.on_network == "NO"]
    if len(missing):
        print(f"\n{len(missing)} chokepoint(s) have no network segment inside their zone: "
              f"{', '.join(missing.cp_name)}. Widen them with --zone-override, or note that the "
              "routing model does not send traffic through them.")
    print(f"\nGeoPackage: {gpkg.resolve()}")
    make_preview_png(network, zones, at_cp, cp_pts, routes, out / "maritime_network_preview.png")


if __name__ == "__main__":
    main()

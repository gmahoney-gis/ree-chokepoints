"""
Maritime routing network (searoute) and IMF PortWatch chokepoints
Produces a 16:9 PNG for a PowerPoint slide.

Install once (PyCharm Terminal):  pip install searoute shapely pandas matplotlib
"""
import json
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import pandas as pd
import searoute
from matplotlib.lines import Line2D
from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, box, shape
from shapely.strtree import STRtree

# ---- Settings you may change -------------------------------------------------
CSV_PATH = r"D:\GEOG797 Data\PortWatch IMF Chokepoints.csv"
OUTPUT_PNG = r"D:\GEOG797 Data\maritime_network_chokepoints.png"
ZONE_KM = 50                                          # radius around each chokepoint
WIDE_STRAITS = {"Luzon Strait": 150, "Torres Strait": 100}   # wider zones
SHOW_KM = 500                                         # highlighted route length shown around each chokepoint
TITLE = "Maritime Routing Network and Global Chokepoints"
HIDE_OCEAN_GRID = True    # hide MARNET's open-ocean lattice (display only; routing unaffected)
THEME = "dark"            # "dark" (briefing style) or "light"

THEMES = {
    "dark": dict(ocean="#0b1d2e", land="#243443", coast="#5b7385", coast_lw=0.5,
                 net="#4fb3d9", net_alpha=0.55, net_glow=0.07,
                 hit="#ffb000", hit_glow=0.30, point_face="#0b1d2e", halo_alpha=0.22,
                 text="#f2f2f2", text_halo="#0b1d2e", leader="#9fb3c2",
                 title="#ffffff", note="#9fb3c2", page="#0b1d2e"),
    "light": dict(ocean="#e3eef4", land="#f5f2eb", coast="#3d3d3d", coast_lw=0.6,
                  net="#7f9eab", net_alpha=1.0, net_glow=0.0,
                  hit="#d62728", hit_glow=0.0, point_face="white", halo_alpha=0.18,
                  text="#222222", text_halo="white", leader="#444444",
                  title="#111111", note="#555555", page="white"),
}
C = THEMES[THEME]

# Natural Earth land polygons (downloaded once, then reused)
LAND_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_50m_land.geojson")
LAND_FILE = Path(OUTPUT_PNG).with_name("ne_50m_land.geojson")

# Label position for each chokepoint: (points right, points up, alignment)
LABELS = {
    "Malacca Strait": (-10, -10, "right"),  "Sunda Strait": (-10, -12, "right"),
    "Lombok Strait": (-4, -18, "right"),    "Ombai Strait": (6, -20, "left"),
    "Torres Strait": (10, -6, "left"),      "Makassar Strait": (22, -6, "left"),
    "Balabac Strait": (30, 2, "left"),      "Mindoro Strait": (-30, 4, "right"),
    "Luzon Strait": (24, 0, "left"),        "Taiwan Strait": (-24, 14, "right"),
    "Bohai Strait": (-14, 6, "right"),      "Korea Strait": (10, -12, "left"),
    "Tsugaru Strait": (10, 4, "left"),      "Yucatan Channel": (-10, -2, "right"),
    "Windward Passage": (0, 16, "center"),  "Mona Passage": (12, -6, "left"),
    "Panama Canal": (-10, -12, "right"),    "Dover Strait": (-10, 8, "right"),
    "Oresund Strait": (8, 8, "left"),       "Gibraltar Strait": (-10, -12, "right"),
    "Bosporus Strait": (-8, 10, "right"),   "Kerch Strait": (10, 6, "left"),
    "Suez Canal": (-10, -10, "right"),      "Bab el-Mandeb Strait": (-10, -12, "right"),
    "Strait of Hormuz": (8, -10, "left"),
}


# ---- Helpers -----------------------------------------------------------------
def split_at_180(line):
    """Some network lines run past longitude 180; cut them and wrap them back."""
    pieces = []
    for shift, window in ((0, box(-180, -90, 180, 90)),
                          (-360, box(180, -90, 540, 90)),
                          (360, box(-540, -90, -180, 90))):
        part = line.intersection(window)
        if part.is_empty:
            continue
        for g in getattr(part, "geoms", [part]):
            if isinstance(g, LineString) and g.length > 0:
                pieces.append(translate(g, xoff=shift))
    return pieces


def on_grid(xy, step=5):
    """True if a point sits exactly on MARNET's 5-degree open-ocean lattice."""
    return all(abs(v / step - round(v / step)) < 1e-6 for v in xy[:2])


def polygon_patch(poly, **style):
    """Matplotlib patch for a shapely polygon, keeping holes (e.g. Caspian Sea)."""
    rings = [poly.exterior] + list(poly.interiors)
    path = MplPath.make_compound_path(*[MplPath(list(r.coords)) for r in rings])
    return PathPatch(path, **style)


def zone(lon, lat, km):
    """Circle of radius km around a point, drawn in degrees."""
    dlat = km / 111.32
    dlon = km / (111.32 * math.cos(math.radians(lat)))
    return Polygon([(lon + dlon * math.cos(math.radians(a)),
                     lat + dlat * math.sin(math.radians(a))) for a in range(0, 360, 5)])


# ---- 1. Maritime network from the searoute package ----------------------------
net_file = Path(searoute.__file__).parent / "data" / "marnet_searoute.geojson"
features = json.loads(net_file.read_text())["features"]
lines = []
for f in features:
    g = shape(f["geometry"])
    for ln in getattr(g, "geoms", [g]):
        # The open-ocean lattice: segments that start or end on a 5-degree grid node
        if HIDE_OCEAN_GRID and (on_grid(ln.coords[0]) or on_grid(ln.coords[-1])):
            continue
        lines.append(ln)
network = [piece for ln in lines for piece in split_at_180(ln)]
print(f"Network segments drawn: {len(network)}")

# ---- 1b. Land (continent outlines) ---------------------------------------------
if not LAND_FILE.exists():
    print("Downloading Natural Earth land polygons (one time)...")
    urllib.request.urlretrieve(LAND_URL, LAND_FILE)
land = [shape(f["geometry"]) for f in json.loads(LAND_FILE.read_text())["features"]]

# ---- 2. Chokepoints ------------------------------------------------------------
cps = pd.read_csv(CSV_PATH)[["portname", "lat", "lon"]]

# ---- 3. Network segments that pass through each chokepoint zone ---------------
tree = STRtree(network)
hits = []
for r in cps.itertuples():
    z = zone(r.lon, r.lat, WIDE_STRAITS.get(r.portname, ZONE_KM))
    found = tree.query(z, predicate="intersects")
    # Trim each segment to SHOW_KM around the chokepoint so long ocean
    # segments are not highlighted across whole oceans.
    piece = [network[i].intersection(zone(r.lon, r.lat, SHOW_KM)) for i in found]
    hits += [g for p in piece for g in getattr(p, "geoms", [p]) if isinstance(g, LineString)]
    print(f"{r.portname:<22} {len(found):>3} route segments pass through")

# ---- 4. Map --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.33, 5.6))      # full slide width; leaves room for a slide title
fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.06)
fig.patch.set_facecolor(C["page"])
ax.set_facecolor(C["ocean"])

for ln in network:
    x, y = ln.xy
    if C["net_glow"]:                                # soft glow under each route
        ax.plot(x, y, color=C["net"], linewidth=2.0, alpha=C["net_glow"], zorder=1)
    ax.plot(x, y, color=C["net"], linewidth=0.4, alpha=C["net_alpha"], zorder=1)

# Land drawn over the network hides inland waterways; coastlines outline continents
for g in land:
    for poly in getattr(g, "geoms", [g]):
        ax.add_patch(polygon_patch(poly, facecolor=C["land"], edgecolor=C["coast"],
                                   linewidth=C["coast_lw"], zorder=1.5))

for ln in hits:
    x, y = ln.xy
    if C["hit_glow"]:
        ax.plot(x, y, color=C["hit"], linewidth=6, alpha=C["hit_glow"], zorder=3)
    ax.plot(x, y, color=C["hit"], linewidth=2.2, solid_capstyle="round", zorder=3)

ax.scatter(cps.lon, cps.lat, s=520, color=C["hit"], alpha=C["halo_alpha"], linewidth=0, zorder=2)
ax.scatter(cps.lon, cps.lat, s=34, facecolor=C["point_face"], edgecolor=C["hit"],
           linewidth=1.6, zorder=4)

halo = [pe.withStroke(linewidth=2.5, foreground=C["text_halo"])]
for r in cps.itertuples():
    dx, dy, ha = LABELS.get(r.portname, (6, 3, "left"))
    leader = dict(arrowstyle="-", color=C["leader"], linewidth=0.5, shrinkA=0, shrinkB=3) \
        if abs(dx) > 12 or abs(dy) > 12 else None
    ax.annotate(r.portname, (r.lon, r.lat), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", fontsize=7.5, fontweight="bold", color=C["text"],
                path_effects=halo, arrowprops=leader, zorder=5)

ax.legend(handles=[
    Line2D([0], [0], color=C["net"], lw=1.2, label="Maritime routing network (searoute / MARNET)"),
    Line2D([0], [0], color=C["hit"], lw=3, label=f"Routes passing through a chokepoint (within {SHOW_KM} km)"),
    Line2D([0], [0], marker="o", lw=0, markerfacecolor=C["point_face"], markeredgecolor=C["hit"],
           markeredgewidth=1.6, markersize=7, label="IMF PortWatch chokepoint (n = 28)")],
    loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9, frameon=False,
    labelcolor=C["text"])

ax.set_xlim(-180, 180)
ax.set_ylim(-60, 75)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(TITLE, fontsize=16, fontweight="bold", pad=10, color=C["title"])
ax.text(1.0, -0.075, f"Data: searoute (Eurostat MARNET); IMF PortWatch; Natural Earth. Chokepoint zone radius "
        f"{ZONE_KM} km (Luzon 150 km, Torres 100 km).", transform=ax.transAxes,
        ha="right", fontsize=7, color=C["note"])

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor=C["page"])
print(f"Saved: {OUTPUT_PNG}")
plt.show()
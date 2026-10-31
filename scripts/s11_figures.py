"""S11: figures and results table (tasks 6.1-6.8; final pass for the paper).

Reads the project database and the S06/S09/S10 tables, then draws:

1. Task 6.1: ranked composite map -> fig_6_1_composite_map.png. Chokepoints as circles
   sized by score_composite and shaded by the fixed classes in outputs.score_class_breaks,
   over faint baseline routes (width proportional to trade); the top outputs.top_n_labels
   are labelled with their rank.
2. Task 6.2: component maps -> fig_6_2_components.png. Three stacked world maps:
   (a) trade exposure, (b) conflict (score_conflict: the violence-tension mix the score
   uses) and (c) redundancy (score_redund_pct, Equation 4 with Decision 16's cap).
3. Task 6.3: scenario route maps -> fig_6_3_routes.png. One map per routing scenario,
   line width proportional to the summed maritime_value_usd of the sub-flows on each route.
4. Task 6.4: close-ups -> fig_6_4_closeups.png. The top outputs.closeups_top_n
   chokepoints: violence density surface (fill), tension density (contours), distance
   rings, the zone or gate line, and baseline routes. Densities use the Equation 3 kernel
   at outputs.closeup_bandwidth_km; no event points are drawn (ACLED terms, Decision 9).
5. Task 6.5: trade vs. conflict chart -> fig_6_5_trade_conflict.png. Quadrants from S09's
   quad_trade_conflict; Spearman result from outputs/tables/correlation.csv.
6. Task 6.6: score breakdown -> fig_6_6_score_breakdown.png. Stacked score_part_* bars.
7. Task 6.7: sensitivity rank ranges -> fig_6_7_sensitivity.png. One-at-a-time and
   weight-grid best-to-worst rank (S10's sens_ columns) with the main rank marked.
8. Task 6.8: results table -> outputs/tables/results_table.csv and fig_6_8_results_table.png,
   with the Decision 8 reroute impact (extra days, cost per voyage, REE holding cost).
9. Decision 8: reroute impact -> fig_reroute_impact.png. Extra days and cost per voyage
   with S08b's low-high ranges; chokepoints without a cost say why.
10. Context: Red Sea transits -> fig_red_sea_context.png (outputs/tables/red_sea_context.csv).

World maps use outputs.static_map_crs (Equal Earth). Close-ups use an azimuthal
equidistant projection centred on each chokepoint, so the distance rings are true circles.
Every figure uses the house style in outputs.style (the dark theme of Lab 6 Figure 1) and
is saved twice: outputs.screen_dpi in paths.figures and outputs.print_dpi in
paths.figures_print. Each figure's caption is written to the log. Task 6.9 (web layers)
waits for the web map.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")   # draw to files only, no window
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from shapely.geometry import LineString, Polygon, box

import common

MAX_MARKER_AREA = 700        # points^2 for the largest circle
MIN_MARKER_AREA = 8          # smallest non-zero circle, so it stays visible
ZERO_MARKER_AREA = 14        # hollow marker for zero values
MAX_ROUTE_WIDTH = 6.0        # points for the route with the most trade
MIN_ROUTE_WIDTH = 0.15
CIRCLE_ALPHA = 0.5           # fill opacity of the proportional circles (edge stays solid)
COMPOSITE_CLASS_AREAS = np.array([24, 48, 80, 115, 150])    # task 6.1: points^2 per score class, low to high
COMPOSITE_ROUTE_SCALE = 0.3                                 # task 6.1: routes thinner than on fig 6.3
DENSITY_UNIT_KM2 = 10_000    # close-up densities are shown as events per 10,000 km^2
DENSITY_RANGE = 1e4          # colour scale of the violence surface spans 4 orders of magnitude
DISTANCE_CHUNK = 2_000       # event locations per block when computing a close-up surface
TENSION_LEVELS = [1, 10, 100, 1000, 10000]   # close-up tension contours, events per 10,000 km^2
EMPTY_CELL = "\u2013"           # task 6.8: en dash in empty table cells


# ---------------------------------------------------------------------
# House style (outputs.style)
# ---------------------------------------------------------------------

def apply_style(style):
    """Set matplotlib defaults to the Figure 1 dark theme so every figure matches."""
    plt.rcParams.update({
        "font.family": style["font"],
        "figure.facecolor": style["background"],
        "savefig.facecolor": style["background"],
        "axes.facecolor": style["background"],
        "axes.edgecolor": style["borders"],
        "axes.labelcolor": style["text"],
        "axes.titlecolor": style["title"],
        "text.color": style["text"],
        "xtick.color": style["note"],
        "ytick.color": style["note"],
        "legend.labelcolor": style["text"],
    })


def halo(style):
    """Path effect that outlines label text in the background colour (Figure 1 labels)."""
    return [pe.withStroke(linewidth=style["label_halo_width"], foreground=style["label_halo"])]


def bottom_legend(ax, handles, labels, title=None, y=-0.01, **kw):
    """Horizontal legend centred under the axes, unframed, as in Figure 1."""
    return ax.legend(handles, labels, title=title, loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=len(handles), frameon=False, **kw)


def source_note(fig, text, style):
    """Small source note at the bottom right of the figure (space reserved by the layout)."""
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    fig.text(0.995, 0.008, text, ha="right", va="bottom", fontsize=6.5, color=style["note"])


def clean_axes(ax, sides=("top", "right")):
    """Hide chart spines, as in the Figure 1 charts."""
    for side in sides:
        ax.spines[side].set_visible(False)


def violence_sources(settings):
    """Caption wording for what the violence counts include (Decision 9 count_rule)."""
    if settings["events"]["imo_gisis"]["count_rule"] == "crew_harm":
        gisis = "piracy incidents in which crew were threatened, injured or held (IMO GISIS)"
    else:
        gisis = "all piracy and armed-robbery incidents (IMO GISIS)"
    return f"violent events (ACLED) and {gisis}"


def save_figure(fig, name, settings, log):
    """Save the screen copy (paths.figures) and the print copy (paths.figures_print)."""
    out = settings["outputs"]
    paths = []
    for key, dpi in (("figures", out["screen_dpi"]), ("figures_print", out["print_dpi"])):
        folder = common.get_path(settings, key)
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / name, dpi=dpi)
        log.info("Wrote %s (%d dpi)", folder / name, dpi)
        paths.append(folder / name)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_inputs(settings, log):
    """Read chokepoint points with their scores, the routes and the sub-flows."""
    points = common.read_layer(settings, "chokepoints")
    scores = pd.DataFrame(common.read_layer(settings, "chokepoint_scores"))
    scores = scores.drop(columns=[c for c in scores.columns if c == "geometry"])
    # chokepoint_scores holds the included chokepoints only; take their points.
    chokepoints = points[["chokepoint_id", "geometry"]].merge(scores, on="chokepoint_id")
    chokepoints = gpd.GeoDataFrame(chokepoints, geometry="geometry", crs=points.crs)
    log.info("Read %d chokepoint points; %d have scores (included)", len(points), len(chokepoints))

    routes = common.read_layer(settings, "routes")
    subflows = pd.DataFrame(common.read_layer(settings, "subflows"))
    log.info("Read %s routes and %s sub-flows", f"{len(routes):,}", f"{len(subflows):,}")
    return chokepoints, routes, subflows


def read_basemap(settings, log):
    """Natural Earth countries without Antarctica (no trade or chokepoints there)."""
    path = common.get_path(settings, "raw_files.natural_earth_countries")
    countries = gpd.read_file(path, columns=["ADMIN"])
    kept = countries[countries["ADMIN"] != "Antarctica"]
    common.log_filter(log, "Base map | Natural Earth countries", len(kept),
                      len(countries) - len(kept), "Antarctica left off the world maps")
    return kept


def impact_note(row):
    """Why a chokepoint has no reroute cost (Decision 13): no affected flows, or no sea route."""
    if pd.notna(row["impact_days"]):
        return ""
    return "no affected flows" if row["redund_nflows_affected"] == 0 else "no sea route"


# ---------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------

def world_outline(crs, south=-60):
    """The edge of the world map (longitude -180 to 180, latitude south to 90) in `crs`."""
    lons = np.linspace(-180, 180, 361)
    lats = np.linspace(south, 90, 151)
    ring = ([(x, south) for x in lons] + [(180, y) for y in lats]
            + [(x, 90) for x in lons[::-1]] + [(-180, y) for y in lats[::-1]])
    return gpd.GeoSeries([Polygon(ring)], crs="EPSG:4326").to_crs(crs)


def draw_basemap(ax, basemap, outline, style):
    """Ocean-filled map frame with dark land and no axes."""
    outline.plot(ax=ax, color=style["background"], edgecolor=style["borders"], linewidth=0.5, zorder=0)
    basemap.plot(ax=ax, color=style["land"], edgecolor=style["borders"], linewidth=0.2, zorder=1)
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = outline.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def marker_areas(values, vmax):
    """Circle area proportional to the value; small non-zero values get a minimum size."""
    return np.maximum(MAX_MARKER_AREA * values / vmax, MIN_MARKER_AREA)


# Candidate label positions (offset in points, horizontal and vertical alignment), tried in order.
LABEL_CANDIDATES = [((6, 4), "left", "bottom"), ((6, -4), "left", "top"),
                    ((-6, 4), "right", "bottom"), ((-6, -4), "right", "top"),
                    ((0, 9), "center", "bottom"), ((0, -9), "center", "top"),
                    ((8, 0), "left", "center"), ((-8, 0), "right", "center")]


# Map labels with a short leader line to their circle (task 6.1), nearest positions first.
def diagonal_candidates(distances):
    """Label positions on the four diagonals at each distance (points), nearest first."""
    return [((dx, dy), "left" if dx > 0 else "right", "bottom" if dy > 0 else "top")
            for dist in distances
            for dx, dy in ((dist, dist * 0.6), (dist, -dist * 0.6), (-dist, dist * 0.6), (-dist, -dist * 0.6))]


# Task 6.1: short leader lines only. Task 6.2: farther positions too, for the large circles there.
MAP_LEADER_NEAR = diagonal_candidates((12, 18, 26))
MAP_LEADER_CANDIDATES = (diagonal_candidates((12, 18, 26, 36, 48))
                         + [((0, dy), "center", "bottom" if dy > 0 else "top") for dy in (22, -22, 34, -34)]
                         + [((dx, 0), "left" if dx > 0 else "right", "center") for dx in (24, -24, 36, -36)])


# Scatter labels: beside the point first, so each label sits level with its own point.
# Farther positions (with a leader line) are the fallback where labels crowd together.
SCATTER_CANDIDATES = ([((7, 0), "left", "center"), ((-7, 0), "right", "center")] + LABEL_CANDIDATES
                      + [((dx, dy), "left" if dx > 0 else "right", "center")
                         for dy in (12, -12, 20, -20, 28, -28) for dx in (10, -10)]
                      + [((dx, dy), "center", "top" if dy < 0 else "bottom")
                         for dy in (-18, 18, -30, 30, -42) for dx in (0, 22, -22)])


def text_box(label, renderer):
    """Screen box of a label's text only (an annotation's own box also covers its leader line)."""
    label.update_positions(renderer)
    return matplotlib.text.Text.get_window_extent(label, renderer)


def place_labels(ax, xs, ys, texts, avoid_xy=(), avoid_lines=(), avoid_boxes=(), inside=False,
                 clearance=0, candidates=LABEL_CANDIDATES, point_radius=4, point_radii=None, **text_kw):
    """Place each label at the first candidate position that overlaps nothing placed so far.

    A simple greedy placer (no extra library): labels are drawn in the order given. A
    label must not overlap an earlier label, a point in `avoid_xy` (data coordinates,
    taken as a square of `point_radius` points, or of each point's own radius in
    `point_radii`) or a box in `avoid_boxes` (display
    coordinates); with `inside`, it must also stay inside the axes. Every point in
    `avoid_xy` other than the label's own keeps at least `clearance` points between its
    edge and the label, so a
    label never sits against a neighbour's point and reads as belonging to its own. It
    should not cross a line in `avoid_lines` (("h", y) or ("v", x) in data
    coordinates): every candidate is tried with that rule first, then without it. If every
    candidate still fails, the first one is used. Returns the labels that cross a line
    and the labels that fell back to the first position.
    """
    renderer = ax.figure.canvas.get_renderer()
    radii = [point_radius] * len(avoid_xy) if point_radii is None else list(point_radii)
    pads = [r * ax.figure.dpi / 72 for r in radii]
    taken = [matplotlib.transforms.Bbox.from_extents(px - r, py - r, px + r, py + r)
             for (px, py), r in zip((ax.transData.transform(p) for p in avoid_xy), pads)] + list(avoid_boxes)
    lines = [(kind, ax.transData.transform((v, 0))[0] if kind == "v" else ax.transData.transform((0, v))[1])
             for kind, v in avoid_lines]
    frame = ax.get_window_extent(renderer)
    points_px = [ax.transData.transform(p) for p in avoid_xy]
    clear_px = clearance * ax.figure.dpi / 72
    leaders = []                       # leader lines placed so far (screen coordinates)

    def gap(bbox, px, py):
        """Screen distance from a point to the nearest edge of a box (0 if inside)."""
        return np.hypot(max(bbox.x0 - px, 0, px - bbox.x1), max(bbox.y0 - py, 0, py - bbox.y1))

    def leader(bbox, own):
        """The line from a point to the nearest edge of its label."""
        return LineString([own, (min(max(own[0], bbox.x0), bbox.x1), min(max(own[1], bbox.y0), bbox.y1))])

    def fits(bbox, own, check_lines):
        if any(bbox.overlaps(t) for t in taken if not t.contains(*own)):   # a label may touch its own point
            return False
        # A leader line may not cross another leader line or another label, and a label may not
        # sit on another leader line.
        line, area = leader(bbox, own), box(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        if any(line.crosses(other) or area.intersects(other) for other in leaders):
            return False
        if line.length > 0 and any(line.intersects(box(t.x0, t.y0, t.x1, t.y1)) for t in taken
                                   if not t.contains(*own)):
            return False
        if any(gap(bbox, px, py) < clear_px + r for (px, py), r in zip(points_px, pads)
               if np.hypot(px - own[0], py - own[1]) > 1):                  # other points keep their distance
            return False
        if check_lines and any((bbox.y0 < pos < bbox.y1) if kind == "h" else (bbox.x0 < pos < bbox.x1)
                               for kind, pos in lines):
            return False
        return not inside or (frame.x0 <= bbox.x0 and bbox.x1 <= frame.x1
                              and frame.y0 <= bbox.y0 and bbox.y1 <= frame.y1)

    crossed, fallbacks = [], []
    for x, y, text in zip(xs, ys, texts):
        own = ax.transData.transform((x, y))
        chosen = None
        for check_lines in ([True, False] if lines else [False]):
            for offset, ha, va in candidates:
                label = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points",
                                    ha=ha, va=va, **text_kw)
                if fits(text_box(label, renderer).expanded(1.06, 1.25), own, check_lines):
                    chosen = label
                    break
                label.remove()
            if chosen is not None:
                if lines and not check_lines:
                    crossed.append(text.replace("\n", " / "))
                break
        if chosen is None:
            fallbacks.append(text.replace("\n", " / "))
            offset, ha, va = candidates[0]
            chosen = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points",
                                 ha=ha, va=va, **text_kw)
        taken.append(text_box(chosen, renderer))
        leaders.append(leader(taken[-1], own))
    return crossed, fallbacks


def leader_labels(ax, rows, labels, points, radii, style, candidates=MAP_LEADER_CANDIDATES):
    """Bold haloed map labels with a short leader line to their own circle (figs 6.1 and 6.2).

    Every chokepoint in `points` (radius in points in `radii`) is kept clear of the labels.
    Call after the figure layout is fixed, so the positions hold. Returns place_labels' result.
    """
    return place_labels(ax, rows.geometry.x, rows.geometry.y, labels,
                 avoid_xy=list(zip(points.geometry.x, points.geometry.y)), point_radii=radii, clearance=2,
                 candidates=candidates,
                 arrowprops={"arrowstyle": "-", "color": style["text"], "lw": 0.5, "shrinkA": 0, "shrinkB": 0},
                 fontsize=6.5, fontweight="bold", color=style["text"], path_effects=halo(style), zorder=5)


def top_rows(points, field, n, always=()):
    """The n chokepoints with the largest values of `field`, plus any IDs in `always`."""
    top = points[points[field] > 0].nlargest(n, field)
    extra = points[points["chokepoint_id"].isin(always) & ~points["chokepoint_id"].isin(top["chokepoint_id"])]
    return pd.concat([top, extra])


def circles(ax, x, y, areas, color, **kw):
    """Proportional circles: translucent fill, solid edge in the same colour."""
    return ax.scatter(x, y, s=areas, facecolor=to_rgba(color, CIRCLE_ALPHA), edgecolor=color,
                      linewidth=0.7, **kw)


def zero_markers(ax, x, y, style, **kw):
    """Small hollow light-grey markers for zero values."""
    return ax.scatter(x, y, s=ZERO_MARKER_AREA, facecolor="none", edgecolor=style["zero_marker"],
                      linewidth=0.7, **kw)


def size_legend(ax, vmax, refs, fmt, color, title, style, note=None):
    """Horizontal legend of reference circle sizes plus the hollow zero marker.

    The unit leads the row as a handle-less entry; a title above the row would be
    overlapped by the largest circle, which matplotlib does not count in the row height.
    An optional `note` ends the row as another handle-less entry.
    """
    handles = [Line2D([], [], linestyle="none")]
    handles += [circles(ax, [], [], marker_areas(np.array([r]), vmax), color) for r in refs]
    labels = [title] + [fmt(r) for r in refs]
    handles.append(zero_markers(ax, [], [], style))
    labels.append("0")
    if note:
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(note)
    legend = bottom_legend(ax, handles, labels, fontsize=6.5, columnspacing=2.0,
                           handletextpad=1.8, borderpad=2.2)   # padding holds the largest circle
    legend.get_texts()[0].set_color(style["note"])


def route_values(routes, subflows, log):
    """Sum maritime_value_usd of the sub-flows using each route (joined on the port pair)."""
    per_pair = (subflows.groupby(["origin_port", "dest_port"], as_index=False)["maritime_value_usd"]
                .sum().rename(columns={"maritime_value_usd": "route_value_usd"}))
    routes = routes.merge(per_pair, on=["origin_port", "dest_port"], how="left")
    unmatched = routes["route_value_usd"].isna().sum()
    routes["route_value_usd"] = routes["route_value_usd"].fillna(0.0)
    log.info("Task 6.3 | route values: %s routes, %d without a matching sub-flow (value 0); "
             "%d same-port sub-flows have no route", f"{len(routes):,}", unmatched,
             int(subflows["same_port"].sum()))
    for scenario, group in routes.groupby("scenario"):
        log.info("Task 6.3 | %-18s %s routes, US$ %s, %d split at the antimeridian", scenario,
                 f"{len(group):,}", f"{group['route_value_usd'].sum():,.0f}",
                 int(group["crosses_180"].sum()))
    return routes


def route_widths(values, vmax):
    """Line width proportional to the route's trade value, with a thin minimum."""
    return np.maximum(MAX_ROUTE_WIDTH * values / vmax, MIN_ROUTE_WIDTH)


def baseline_routes(routes):
    """The baseline-scenario routes, thin first so thick routes draw on top."""
    return routes[routes["scenario"] == "baseline"].sort_values("route_value_usd")


# ---------------------------------------------------------------------
# Task 6.1: ranked composite map
# ---------------------------------------------------------------------

def score_classes(settings):
    """Colour map and norm for the fixed composite classes (outputs.score_class_breaks)."""
    breaks = settings["outputs"]["score_class_breaks"]
    colors = settings["outputs"]["style"]["score_scale"]
    if len(colors) != len(breaks) - 1:
        raise ValueError("outputs.style.score_scale needs one colour per class of score_class_breaks")
    return ListedColormap(colors), BoundaryNorm(breaks, len(colors)), breaks


def fig_composite(points, routes, basemap, outline, settings, log):
    """Task 6.1: chokepoints sized and shaded by composite score over faint baseline routes."""
    style = settings["outputs"]["style"]
    n = settings["outputs"]["top_n_labels"]
    cmap, _, breaks = score_classes(settings)
    base = baseline_routes(routes)
    vmax_route = routes["route_value_usd"].max()

    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    draw_basemap(ax, basemap, outline, style)
    # Routes are context only: faint and thin, width still proportional to trade.
    base.plot(ax=ax, color=style["route"],
              linewidth=route_widths(base["route_value_usd"], vmax_route) * COMPOSITE_ROUTE_SCALE,
              alpha=0.15, zorder=2)
    # Graduated circles: one size and one shade per class; higher scores drawn on top.
    order = points.sort_values("score_composite")
    classes = np.digitize(order["score_composite"], breaks[1:-1])
    ax.scatter(order.geometry.x, order.geometry.y, s=COMPOSITE_CLASS_AREAS[classes],
               facecolor=cmap(classes), edgecolor=style["zero_marker"], linewidth=0.5, zorder=3)

    # Legend: one entry per class, at its circle size.
    handles = [Line2D([], [], linestyle="none")]
    labels = ["Composite score"]
    for i, color in enumerate(cmap.colors):
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(COMPOSITE_CLASS_AREAS[i]),
                              markerfacecolor=color, markeredgecolor=style["zero_marker"], markeredgewidth=0.5))
        labels.append(f"{breaks[i]:.1f}–{breaks[i + 1]:.1f}")
    legend = bottom_legend(ax, handles, labels, fontsize=6.5, columnspacing=1.6, handletextpad=1.0,
                           borderpad=1.2)
    legend.get_texts()[0].set_color(style["note"])
    source_note(fig, "Data: UN Comtrade; IMF PortWatch; ACLED; IMO GISIS; GDELT; searoute; Natural Earth. "
                     "Equal Earth projection.", style)

    # Top-n labels with short leader lines, placed after the layout is fixed so positions hold.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    top = points.nsmallest(n, "score_rank")
    radius = np.sqrt(COMPOSITE_CLASS_AREAS[-1]) / 2          # every circle counted at the largest size
    _, fallbacks = leader_labels(ax, top, [f"{int(r)}  {name}" for r, name in zip(top["score_rank"], top["name"])],
                                 points, [radius] * len(points), style, candidates=MAP_LEADER_NEAR)
    counts, _ = np.histogram(points["score_composite"], bins=breaks)
    log.info("Task 6.1 | chokepoints per class: %s",
             ", ".join(f"{lo:.1f}-{hi:.1f}: {c}" for lo, hi, c in zip(breaks[:-1], breaks[1:], counts)))
    log.info("Task 6.1 | labelled: %s; fell back to the first position (may overlap): %s",
             ", ".join(top["name"]), ", ".join(fallbacks) or "none")

    w = settings["scoring"]["weights"]
    caption = (f"Figure 6.1. Composite vulnerability score of 2025 maritime rare earth trade at the "
               f"{len(points)} included chokepoints (Equal Earth). Circle size and shade show the score "
               f"class (fixed breaks {', '.join(f'{b:.1f}' for b in breaks)}, "
               f"so the classes stay the same if the analysis is rerun). The score is the weighted sum "
               f"(weights trade {w['trade']:.2f}, conflict {w['conflict']:.2f}, redundancy "
               f"{w['redundancy']:.2f}) of rescaled trade exposure, conflict exposure (violence and "
               f"tension within {settings['conflict']['ring_main_km']} km) and route redundancy. The top "
               f"{n} are labelled with their rank. Faint lines are the baseline sea routes between "
               f"PortWatch principal ports, width proportional to trade value. Sources: UN Comtrade, "
               f"IMF PortWatch, ACLED, IMO GISIS, GDELT, Natural Earth.")
    return save_figure(fig, "fig_6_1_composite_map.png", settings, log), caption


# ---------------------------------------------------------------------
# Task 6.2: component maps
# ---------------------------------------------------------------------

def component_panel(ax, points, field, color, title, legend_refs, fmt, legend_title, style, note=None):
    """One component map: proportional circles and hollow zeros. Returns each circle's radius
    in points (for the labels, drawn later) and the number of zeros."""
    zero = points[points[field] == 0]
    nonzero = points[points[field] > 0].sort_values(field, ascending=False)  # small on top
    vmax = points[field].max()
    circles(ax, nonzero.geometry.x, nonzero.geometry.y, marker_areas(nonzero[field], vmax), color,
            zorder=3)
    zero_markers(ax, zero.geometry.x, zero.geometry.y, style, zorder=3)
    size_legend(ax, vmax, legend_refs, fmt, color, legend_title, style, note)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    areas = np.where(points[field] > 0, marker_areas(points[field], vmax), ZERO_MARKER_AREA)
    return np.sqrt(areas) / 2, len(zero)


def fig_components(points, basemap, outline, settings, log):
    """Task 6.2: trade (a), conflict (b) and redundancy (c) maps, stacked."""
    n = settings["outputs"]["top_n_labels"]
    always = settings["outputs"]["always_label"]
    ring = settings["conflict"]["ring_main_km"]
    trade_field = settings["scoring"]["correlation_main"]["trade"]
    mix = settings["scoring"]["conflict_mix"]
    style = settings["outputs"]["style"]

    cap = points["score_redund_pct"].max()
    cap_note = f"{cap:.1f}% = no-route cap (no sea route; Decision 16)"

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 13.4), layout="constrained")
    for ax in axes:
        draw_basemap(ax, basemap, outline, style)
    panels = {"trade": trade_field, "conflict": "score_conflict", "redundancy": "score_redund_pct"}
    radii, zeros = {}, {}
    radii["trade"], zeros["trade"] = component_panel(
        axes[0], points, trade_field, style["route"],
        "(a) Trade exposure: maritime REE trade crossing each chokepoint",
        [1e7, 1e8, 5e8], lambda v: f"${v / 1e6:,.0f} M", "US$ (2025)", style)
    radii["conflict"], zeros["conflict"] = component_panel(
        axes[1], points, "score_conflict", style["highlight"],
        f"(b) Conflict exposure: violence and tension within {ring} km (rescaled, 0–1)",
        [0.25, 0.5, 1.0], lambda v: f"{v:.2f}", "Score", style)
    radii["redundancy"], zeros["redundancy"] = component_panel(
        axes[2], points, "score_redund_pct", style["redundancy"],
        "(c) Route redundancy: extra distance if the chokepoint closes",
        [10, 30, 60], lambda v: f"{v:.0f}%", "Extra distance", style, note=cap_note)
    source_note(fig, "Data: UN Comtrade; IMF PortWatch; ACLED; IMO GISIS; GDELT; searoute; Natural Earth. "
                     "Equal Earth projection.", style)

    # Labels with leader lines, placed after the layout is fixed so positions hold.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax, (key, field) in zip(axes, panels.items()):
        rows = top_rows(points, field, n, always)
        _, fallbacks = leader_labels(ax, rows, rows["name"], points, radii[key], style)
        log.info("Task 6.2 | labelled %s: %s; fell back to the first position (may overlap): %s", key,
                 ", ".join(rows["name"]), ", ".join(fallbacks) or "none")

    at_cap = points.loc[np.isclose(points["score_redund_pct"], cap), "name"]
    filled = points.loc[points["score_redund_filled"].astype(bool), "name"]
    w = settings["events"]
    caption = (f"Figure 6.2. The three components of the composite score at the {len(points)} included "
               f"maritime chokepoints (Equal Earth); circle area is proportional to the value. (a) Value "
               f"of 2025 maritime rare earth trade (HS {', '.join(settings['trade']['hs_codes_main'])}) "
               f"whose baseline route crosses each chokepoint. (b) Conflict exposure: the mix "
               f"({mix['violence']:.2f} violence, {mix['tension']:.2f} tension) of the rescaled number of "
               f"{violence_sources(settings)} and the rescaled number of GDELT tension events, both "
               f"within {ring} km, {w['window_start']} to {w['window_end']}. (c) Route redundancy "
               f"(Equation 4): the value-weighted extra sea distance of the affected trade if the "
               f"chokepoint closes; trade with no sea route gets the largest penalty among chokepoints "
               f"whose trade can still be rerouted ({cap:.1f}%, shared by {len(at_cap)}: "
               f"{', '.join(at_cap)}); {cap_note}. Hollow markers show zero values ({zeros['trade']} for trade, "
               f"{zeros['conflict']} for conflict, where 0 is the lowest chokepoint after rescaling, "
               f"{zeros['redundancy']} for redundancy"
               f"{'; ' + ', '.join(filled) + ' has no affected flows and is set to 0' if len(filled) else ''}). "
               f"The top {n} on each panel are labelled (leader lines point to each circle), and "
               f"{', '.join(points.loc[points['chokepoint_id'].isin(always), 'name'])} on every panel. "
               f"Sources: UN Comtrade, IMF PortWatch, ACLED, IMO GISIS, GDELT, Natural Earth.")
    return save_figure(fig, "fig_6_2_components.png", settings, log), caption


# ---------------------------------------------------------------------
# Task 6.3: scenario route maps
# ---------------------------------------------------------------------

def fig_routes(routes, points, basemap, outline, settings, log):
    """Task 6.3: baseline and Red Sea diversion routes, one map each, same width scale."""
    scenarios = list(settings["routing"]["scenarios"])
    vmax = routes["route_value_usd"].max()
    style = settings["outputs"]["style"]
    # Chokepoints as in Figure 1: background-filled dot with an amber ring.
    point_style = {"facecolor": style["background"], "edgecolor": style["highlight"], "linewidth": 1.0}

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(7.5, 4.1 * len(scenarios)), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, scenarios):
        draw_basemap(ax, basemap, outline, style)
        group = routes[routes["scenario"] == scenario].sort_values("route_value_usd")  # thick on top
        group.plot(ax=ax, color=style["route"], linewidth=route_widths(group["route_value_usd"], vmax),
                   alpha=0.55, zorder=2)
        ax.scatter(points.geometry.x, points.geometry.y, s=12, zorder=4, **point_style)
        closed = settings["routing"]["scenarios"][scenario]["closed"]
        subtitle = "all chokepoints open" if not closed else "closed: " + ", ".join(closed)
        ax.set_title(f"{scenario.replace('_', ' ').capitalize()} ({subtitle})", loc="left",
                     fontsize=9, fontweight="bold")

    # One legend for both panels (same width scale), under the last map.
    refs = [1e7, 1e8, vmax]
    handles = [Line2D([], [], color=style["route"], alpha=0.55, linewidth=route_widths(np.array([r]), vmax)[0])
               for r in refs]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=4,
                          markerfacecolor=style["background"], markeredgecolor=style["highlight"],
                          markeredgewidth=1.0))
    labels = [f"${r / 1e6:,.0f} M" for r in refs] + ["Chokepoint"]
    legend = bottom_legend(axes[-1], handles, labels, title="Route trade, US$ (2025)", fontsize=6.5,
                           title_fontsize=6.5, columnspacing=2.0)
    legend.get_title().set_color(style["note"])
    source_note(fig, "Data: UN Comtrade; IMF PortWatch; searoute; Natural Earth. Equal Earth projection.", style)

    diversion = settings["routing"]["scenarios"].get("red_sea_diversion", {})
    share = settings["ports"]["principal_port_share"]
    caption = (f"Figure 6.3. Modelled sea routes of 2025 maritime rare earth trade between principal "
               f"ports (IMF PortWatch ports that together carry {100 * share:.0f}% of each country's "
               f"maritime exports or imports; landlocked countries use a gateway port; the World Port "
               f"Index fills one country PortWatch lacks) under the baseline and the Red Sea diversion "
               f"({', '.join(diversion.get('closed', []))} closed; displaced flows rerouted round the "
               f"Cape of Good Hope). Line width is proportional to the summed value of the sub-flows "
               f"using each route, on the same scale in both panels; dots are the {len(points)} included "
               f"chokepoints. Routes from searoute, split at the antimeridian. Equal Earth projection. "
               f"Sources: UN Comtrade, IMF PortWatch, NGA World Port Index, Natural Earth.")
    return save_figure(fig, "fig_6_3_routes.png", settings, log), caption


# ---------------------------------------------------------------------
# Task 6.4: chokepoint close-ups
# ---------------------------------------------------------------------

def local_crs(point):
    """Azimuthal equidistant projection centred on a chokepoint (true distance from the centre)."""
    return f"+proj=aeqd +lat_0={point.y} +lon_0={point.x} +datum=WGS84 +units=m +no_defs"


def window_lonlat(crs, half_m):
    """The close-up window (a square in the local projection) as a lon/lat bounding box."""
    edge = np.linspace(-half_m, half_m, 41)
    xs = np.concatenate([edge, np.full_like(edge, half_m), edge[::-1], np.full_like(edge, -half_m)])
    ys = np.concatenate([np.full_like(edge, -half_m), edge, np.full_like(edge, half_m), edge[::-1]])
    corners = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=crs).to_crs("EPSG:4326")
    return corners.total_bounds


def by_location(events, weights):
    """Sum event weights at each distinct location, so each place is computed once."""
    table = pd.DataFrame({"x": events.geometry.x, "y": events.geometry.y, "w": weights})
    return table.groupby(["x", "y"], as_index=False)["w"].sum()


def density_surface(grid_lon, grid_lat, places, settings):
    """Equation 3 kernel (S07, task 4.9) at outputs.closeup_bandwidth_km, read at every grid cell.

    Distances are great-circle; the result is events per DENSITY_UNIT_KM2.
    """
    h = settings["outputs"]["closeup_bandwidth_km"]
    total = np.zeros(len(grid_lon))
    for start in range(0, len(places), DISTANCE_CHUNK):
        chunk = places.iloc[start:start + DISTANCE_CHUNK]
        d = common.distance_matrix_km(chunk["x"], chunk["y"], grid_lon, grid_lat, settings)
        kernel = np.exp(-d ** 2 / (2.0 * h ** 2)) / (2.0 * np.pi * h ** 2)
        total += (kernel * chunk["w"].to_numpy()[:, None]).sum(axis=0)
    return total * DENSITY_UNIT_KM2


def closeup_data(point, violence, tension, settings):
    """Grid, violence and tension surfaces for one chokepoint, in its local projection."""
    out = settings["outputs"]
    half_m, cell_m = out["closeup_half_width_km"] * 1000, out["closeup_cell_km"] * 1000
    crs = local_crs(point)
    axis = np.arange(-half_m, half_m + cell_m / 2, cell_m)
    gx, gy = np.meshgrid(axis, axis)
    cells = gpd.GeoSeries(gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=crs).to_crs("EPSG:4326")
    # Events that can reach the window: within its half-diagonal plus 4 bandwidths of the centre.
    reach = np.sqrt(2) * out["closeup_half_width_km"] + 4 * out["closeup_bandwidth_km"]
    surfaces = {}
    for key, places in (("violence", violence), ("tension", tension)):
        d = common.distance_matrix_km(places["x"], places["y"], [point.x], [point.y], settings)[:, 0]
        near = places[d <= reach]
        surfaces[key] = density_surface(cells.x.to_numpy(), cells.y.to_numpy(), near, settings).reshape(gx.shape)
        surfaces[f"{key}_places"] = len(near)
    return crs, gx, gy, surfaces


def closeup_panel(ax, row, data, routes, zones, gates, basemap, vmax_route, norm, cmap, settings):
    """One close-up: land, violence fill, tension contours, routes, rings and the zone or gate."""
    style = settings["outputs"]["style"]
    out = settings["outputs"]
    crs, gx, gy, surf = data
    half_m = out["closeup_half_width_km"] * 1000
    rings = settings["conflict"]["ring_distances_km"]
    # Events were kept only within events.prefilter_radius_km of a chokepoint, so the surface is
    # shown only inside the largest ring.
    outside = np.hypot(gx, gy) > max(rings) * 1000
    lon0, lat0, lon1, lat1 = window_lonlat(crs, half_m)
    frame = box(-half_m, -half_m, half_m, half_m)

    ax.set_facecolor(style["background"])
    # Clipped in lon/lat only: projected country outlines can be invalid, and the axis limits crop them.
    land = basemap.cx[lon0:lon1, lat0:lat1].clip(box(lon0, lat0, lon1, lat1)).to_crs(crs)
    land.plot(ax=ax, color=style["land"], edgecolor="none", zorder=1)
    ax.pcolormesh(gx, gy, np.ma.masked_where(outside | (surf["violence"] < norm.vmin), surf["violence"]),
                  cmap=cmap, norm=norm, alpha=0.7, shading="nearest", zorder=2)
    land.boundary.plot(ax=ax, color=style["borders"], linewidth=0.4, zorder=3)   # coasts over the fill
    levels = [lvl for lvl in TENSION_LEVELS if lvl <= surf["tension"][~outside].max()]
    if levels:
        cs = ax.contour(gx, gy, np.ma.masked_where(outside, surf["tension"]), levels=levels,
                        colors=style["text"], linewidths=0.5, alpha=0.8, zorder=3)
        ax.clabel(cs, fmt=lambda v: f"{v:,.0f}", fontsize=5, colors=style["text"])
    local = routes.cx[lon0:lon1, lat0:lat1]
    if len(local):
        local = local.to_crs(crs).clip(frame)
        local.plot(ax=ax, color=style["route"], linewidth=route_widths(local["route_value_usd"], vmax_route),
                   alpha=0.45, zorder=4)
    for r in rings:
        ax.add_patch(Circle((0, 0), r * 1000, fill=False, edgecolor=style["note"], linewidth=0.6,
                            linestyle=(0, (3, 2)), zorder=5))
        ax.text(0, -r * 1000, f"{r} km", ha="center", va="bottom", fontsize=5, color=style["note"],
                path_effects=halo(style), zorder=6)
    # Decision 12: a gated chokepoint is drawn as its gate line, any other as its zone outline.
    gate = gates[gates["chokepoint_id"] == row["chokepoint_id"]]
    outline = gate.geometry if len(gate) else zones[zones["chokepoint_id"] == row["chokepoint_id"]].boundary
    gpd.GeoSeries(outline.to_numpy(), crs="EPSG:4326").to_crs(crs).plot(
        ax=ax, color=style["highlight"], linewidth=1.2, zorder=6)
    ax.scatter([0], [0], s=14, facecolor=style["background"], edgecolor=style["highlight"],
               linewidth=1.0, zorder=7)
    ax.set_xlim(-half_m, half_m)
    ax.set_ylim(-half_m, half_m)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{int(row['score_rank'])}. {row['name']} ({row['score_composite']:.2f})", loc="left",
                 fontsize=8, fontweight="bold")


def fig_closeups(points, routes, basemap, settings, log):
    """Task 6.4: close-ups of the top chokepoints (density surfaces, rings, zone, routes)."""
    style = settings["outputs"]["style"]
    top_n = settings["outputs"]["closeups_top_n"]
    top = points.nsmallest(top_n, "score_rank").to_crs("EPSG:4326")
    zones = common.read_layer(settings, "zones")
    gates = common.read_layer(settings, "gates")

    events = common.read_layer(settings, "events")
    violence = common.countable_events(events[events["category"] == "violence"], settings, log)
    tension = common.countable_tension(events, settings, log)
    violence_places = by_location(violence, np.ones(len(violence)))
    tension_places = by_location(tension, tension["event_count"].to_numpy())
    log.info("Task 6.4 | %s violence events at %s places; %s tension rows (%s events) at %s places",
             f"{len(violence):,}", f"{len(violence_places):,}", f"{len(tension):,}",
             f"{int(tension['event_count'].sum()):,}", f"{len(tension_places):,}")

    data = {cid: closeup_data(row.geometry, violence_places, tension_places, settings)
            for cid, row in top.set_index("chokepoint_id").iterrows()}
    vmax = max(d[3]["violence"].max() for d in data.values())
    norm = LogNorm(vmin=vmax / DENSITY_RANGE, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("violence", style["score_scale"])
    base = baseline_routes(routes)
    vmax_route = routes["route_value_usd"].max()

    ncols = 3
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 2.75 * nrows + 0.9), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, top.iterrows()):
        d = data[row["chokepoint_id"]]
        closeup_panel(ax, row, d, base, zones, gates, basemap, vmax_route, norm, cmap, settings)
        log.info("Task 6.4 | %-20s violence peak %.2f, tension peak %.1f per %s km2 (%s / %s places within reach)",
                 row["name"], d[3]["violence"].max(), d[3]["tension"].max(), f"{DENSITY_UNIT_KM2:,}",
                 f"{d[3]['violence_places']:,}", f"{d[3]['tension_places']:,}")
    for ax in axes[top_n:]:
        ax.set_axis_off()

    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(),
                       location="bottom", shrink=0.45, aspect=35, pad=0.02)
    bar.set_label(f"Violent events per {DENSITY_UNIT_KM2:,} km² (kernel density)", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_edgecolor(style["borders"])
    handles = [Line2D([], [], color=style["text"], linewidth=0.5),
               Line2D([], [], color=style["note"], linewidth=0.6, linestyle=(0, (3, 2))),
               Line2D([], [], color=style["highlight"], linewidth=1.2),
               Line2D([], [], color=style["route"], linewidth=1.5, alpha=0.6)]
    labels = [f"Tension events per {DENSITY_UNIT_KM2:,} km²", "Distance rings", "Chokepoint zone",
              "Baseline routes"]
    bar.ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -3.6), ncol=4, frameon=False,
                  fontsize=6.5)
    source_note(fig, "Data: ACLED; IMO GISIS; GDELT; UN Comtrade; IMF PortWatch; Natural Earth. "
                     "Azimuthal equidistant projection centred on each chokepoint.", style)

    h = settings["outputs"]["closeup_bandwidth_km"]
    w = settings["events"]
    caption = (f"Figure 6.4. Conflict around the {top_n} highest-scoring chokepoints (composite rank and "
               f"score in each title). Shading: kernel density of {violence_sources(settings)}; contours: "
               f"kernel density of GDELT tension events (CAMEO roots "
               f"{w['gdelt']['main_filter']['root_codes'][0]}-{w['gdelt']['main_filter']['root_codes'][-1]}, "
               f"at least {w['gdelt']['main_filter']['min_sources']} news outlets), both Gaussian kernels "
               f"with a {h} km bandwidth (Equation 3; narrower than the {settings['conflict']['density_main_km']} km "
               f"used in the density sensitivity test, so local patterns show), {w['window_start']} to "
               f"{w['window_end']}, on one scale for all panels. Individual events are not shown. Dashed circles are the "
               f"{', '.join(str(r) for r in settings['conflict']['ring_distances_km'])} km rings; the amber "
               f"outline is the chokepoint zone used for crossings and closures; cyan lines are baseline "
               f"sea routes. Events were downloaded within {w['prefilter_radius_km']} km of each "
               f"chokepoint, so the density surface ends at the {w['prefilter_radius_km']} km ring. "
               f"Each panel is {2 * settings['outputs']['closeup_half_width_km']:,} km across in an "
               f"azimuthal equidistant projection centred on the chokepoint, so the rings are true "
               f"distances. Sources: ACLED, IMO GISIS, GDELT, UN Comtrade, IMF PortWatch, Natural Earth.")
    return save_figure(fig, "fig_6_4_closeups.png", settings, log), caption


# ---------------------------------------------------------------------
# Task 6.5: trade vs. conflict chart
# ---------------------------------------------------------------------

def read_correlation(settings, log):
    """The main Spearman result from outputs/tables/correlation.csv (written by S09)."""
    path = common.get_path(settings, "tables") / "correlation.csv"
    table = pd.read_csv(path)
    main = table[table["variant"] == "main"].iloc[0]
    log.info("Task 6.5 | read %s: main rho = %.3f, p = %.3f, n = %d", path, main["rho"],
             main["p_value"], main["n"])
    return main


def label_groups(points, xf, yf):
    """Group chokepoints that share the same (x, y) so they get one stacked label."""
    return points.groupby([xf, yf])["name"].apply(lambda s: sorted(s)).reset_index()


def quadrant_check(points, xf, yf, x_med, y_med, settings):
    """Stop if S09's quad_trade_conflict disagrees with the medians and the tie rule."""
    above = (lambda v, m: v > m) if settings["scoring"]["quadrant_tie"] == "low" else (lambda v, m: v >= m)
    expected = (np.where(above(points[xf], x_med), "high_trade", "low_trade").astype(object) + "_"
                + np.where(above(points[yf], y_med), "high_violence", "low_violence"))
    if (expected != points["quad_trade_conflict"].to_numpy()).any():
        raise ValueError("quad_trade_conflict (S09) disagrees with the medians and scoring.quadrant_tie")


def margin_labels(ax, xm, ym, style):
    """Name the quadrant halves outside the plot (top: trade; right: violence)."""
    mid = ax.transAxes.inverted().transform(ax.transData.transform((xm, ym)))
    kw = {"transform": ax.transAxes, "fontsize": 7, "color": style["note"], "fontstyle": "italic"}
    ax.text(mid[0] - 0.01, 1.01, "← Low trade", ha="right", va="bottom", **kw)
    ax.text(mid[0] + 0.01, 1.01, "High trade →", ha="left", va="bottom", **kw)
    ax.text(1.01, mid[1] + 0.01, "High violence →", ha="left", va="bottom", rotation=90, **kw)
    ax.text(1.01, mid[1] - 0.01, "← Low violence", ha="left", va="top", rotation=90, **kw)


def fig_trade_conflict(points, settings, log):
    """Task 6.5: trade (x) vs. violence (y) on log(1 + x) axes, quadrants from S09."""
    xf = settings["scoring"]["correlation_main"]["trade"]
    yf = settings["scoring"]["correlation_main"]["conflict"]
    main = read_correlation(settings, log)
    x, y = np.log1p(points[xf]), np.log1p(points[yf])
    x_med, y_med = points[xf].median(), points[yf].median()
    style = settings["outputs"]["style"]
    quadrant_check(points, xf, yf, x_med, y_med, settings)
    quadrants = points["quad_trade_conflict"].value_counts().sort_index()
    on_line = points.loc[(points[xf] == x_med) | (points[yf] == y_med), "name"]

    fig, ax = plt.subplots(figsize=(7.5, 6.6), layout="constrained")
    ax.axvline(np.log1p(x_med), color=style["note"], linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(np.log1p(y_med), color=style["note"], linestyle="--", linewidth=0.8, zorder=1)
    ax.scatter(x, y, s=36, color=style["highlight"], edgecolor=style["background"], linewidth=0.8,
               zorder=3)

    ax.margins(x=0.06, y=0.06)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(-1.6, ax.get_ylim()[1])     # room under the zero row for its labels (log(1 + x) units)
    margin_labels(ax, np.log1p(x_med), np.log1p(y_med), style)

    # Axis ticks at round values of the original units.
    xticks = [0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    ax.set_xticks(np.log1p(xticks), ["0", "$10 K", "$100 K", "$1 M", "$10 M", "$100 M", "$1 B"])
    yticks = [0, 1, 10, 100, 1000, 10000]
    ax.set_yticks(np.log1p(yticks), [f"{t:,}" for t in yticks])
    ax.set_xlabel(f"Trade exposure, US$ 2025 ({xf}, log(1 + x) scale)", fontsize=8)
    ring = settings["conflict"]["ring_main_km"]
    ax.set_ylabel(f"Violent events within {ring} km ({yf}, log(1 + x) scale)", fontsize=8)
    ax.tick_params(labelsize=7)
    clean_axes(ax)
    stats = ax.text(0.02, 0.98, f"Spearman ρ = {main['rho']:.2f}\np = {main['p_value']:.2f}, n = {int(main['n'])}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.4", "fc": style["background"], "ec": style["borders"],
                          "lw": 0.6})
    source_note(fig, "Data: UN Comtrade; IMF PortWatch; searoute; ACLED; IMO GISIS.", style)

    # Constrained layout runs once on the finished frame, then is frozen so label positions hold.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    # Every chokepoint labelled; coincident points share one stacked label. Labels avoid the
    # points, each other, the statistics box and (where they can) the median lines. The
    # crowded high-trade side is labelled first, while it still has room.
    groups = label_groups(points, xf, yf).sort_values([xf, yf], ascending=False)
    crossed, fallbacks = place_labels(
        ax, np.log1p(groups[xf]), np.log1p(groups[yf]), ["\n".join(names) for names in groups["name"]],
        avoid_xy=list(zip(x, y)), avoid_lines=[("v", np.log1p(x_med)), ("h", np.log1p(y_med))],
        avoid_boxes=[stats.get_window_extent()], inside=True, clearance=4, candidates=SCATTER_CANDIDATES,
        arrowprops={"arrowstyle": "-", "color": style["note"], "lw": 0.4, "shrinkA": 0, "shrinkB": 3},
        fontsize=6, color=style["text"], linespacing=1.1, zorder=4, path_effects=halo(style))
    log.info("Task 6.5 | %d labels placed; cross a median line: %s; fell back to the first position "
             "(may overlap): %s", len(groups), ", ".join(crossed) or "none", ", ".join(fallbacks) or "none")
    log.info("Task 6.5 | quadrants from S09 quad_trade_conflict: %s",
             ", ".join(f"{q} {n}" for q, n in quadrants.items()))

    # The log goes to a cp1252 console on Windows, so the caption spells out "rho".
    verdict = "no significant association" if main["p_value"] >= 0.05 else "significant at p < 0.05"
    tie = settings["scoring"]["quadrant_tie"]
    caption = (f"Figure 6.5. Trade exposure against violence exposure for the {len(points)} included "
               f"chokepoints, both on log(1 + x) axes. Violence exposure is the number of "
               f"{violence_sources(settings)} within {ring} km. Dashed lines mark the medians (trade "
               f"US$ {x_med:,.0f}; violence {y_med:,.0f} events) and divide the four quadrants ("
               f"{'; '.join(f'{q.replace(chr(95), chr(32))} {n}' for q, n in quadrants.items())}). A value exactly at the median counts "
               f"as {tie} ({', '.join(on_line)}). The violence median is only {y_med:,.0f} events, so the "
               f"\"high violence\" side includes chokepoints with very few events. "
               f"{int(main['zero_trade'])} {'chokepoint has' if main['zero_trade'] == 1 else 'chokepoints have'} "
               f"zero trade and {int(main['zero_conflict'])} have zero violent events; coincident points "
               f"share one label. Spearman rho = "
               f"{main['rho']:.2f} (p = {main['p_value']:.2f}): {verdict}. Sources: UN Comtrade, IMF PortWatch, "
               f"searoute, ACLED, IMO GISIS.")
    return save_figure(fig, "fig_6_5_trade_conflict.png", settings, log), caption


# ---------------------------------------------------------------------
# Task 6.6: score breakdown
# ---------------------------------------------------------------------

def fig_breakdown(points, settings, log):
    """Task 6.6: horizontal bars split into the three weighted component contributions."""
    style = settings["outputs"]["style"]
    table = points.sort_values("score_rank")
    parts = [("score_part_trade", "Trade", style["route"]),
             ("score_part_conflict", "Conflict", style["highlight"]),
             ("score_part_redund", "Redundancy", style["redundancy"])]
    check = table[[p for p, _, _ in parts]].sum(axis=1) - table["score_composite"]
    if check.abs().max() > 1e-6:
        raise ValueError("score_part_* columns do not add up to score_composite")

    fig, ax = plt.subplots(figsize=(7.5, 7.4), layout="constrained")
    ys = np.arange(len(table))
    left = np.zeros(len(table))
    for field, label, color in parts:
        ax.barh(ys, table[field], left=left, height=0.72, color=color, edgecolor=style["background"],
                linewidth=1.0, label=label, zorder=2)
        left += table[field].to_numpy()
    for yv, (_, row) in zip(ys, table.iterrows()):
        note = "  redundancy set to 0 (no affected flows)" if row["score_redund_filled"] else ""
        ax.text(row["score_composite"] + 0.008, yv, f"{row['score_composite']:.2f}{note}", va="center",
                fontsize=6.5, color=style["text"])
    ax.set_yticks(ys, [f"{int(r)}. {n}" for r, n in zip(table["score_rank"], table["name"])], fontsize=7)
    ax.tick_params(axis="y", colors=style["text"], length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Composite score (sum of weighted, rescaled components)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color=style["grid"], linewidth=0.6, zorder=0)
    clean_axes(ax, ("top", "right", "left"))
    handles, labels = ax.get_legend_handles_labels()
    bottom_legend(ax, handles, labels, fontsize=7, y=-0.06)
    source_note(fig, "Data: UN Comtrade; IMF PortWatch; ACLED; IMO GISIS; GDELT; searoute.", style)

    shares = table[[p for p, _, _ in parts]].sum() / table["score_composite"].sum()
    log.info("Task 6.6 | share of all score points: %s",
             ", ".join(f"{label} {100 * shares[f]:.0f}%" for f, label, _ in parts))
    w = settings["scoring"]["weights"]
    caption = (f"Figure 6.6. Composite score of each of the {len(points)} included chokepoints, split into "
               f"the contributions of trade exposure, conflict exposure and route redundancy (each "
               f"component rescaled to 0-1 and multiplied by its weight: {w['trade']:.2f}, "
               f"{w['conflict']:.2f}, {w['redundancy']:.2f}); sorted by rank. Across all chokepoints, "
               f"trade gives {100 * shares['score_part_trade']:.0f}% of the score points, conflict "
               f"{100 * shares['score_part_conflict']:.0f}% and redundancy "
               f"{100 * shares['score_part_redund']:.0f}%. Sources: UN Comtrade, IMF PortWatch, ACLED, "
               f"IMO GISIS, GDELT.")
    return save_figure(fig, "fig_6_6_score_breakdown.png", settings, log), caption


# ---------------------------------------------------------------------
# Task 6.7: sensitivity rank ranges
# ---------------------------------------------------------------------

def fig_sensitivity(points, settings, log):
    """Task 6.7: best-to-worst rank under one-at-a-time tests and the weight grid."""
    style = settings["outputs"]["style"]
    table = points.sort_values("score_rank")
    summary = pd.read_csv(common.get_path(settings, "tables") / "sensitivity.csv")
    one = summary[summary["group"] == "one_at_a_time"]
    n_run, not_run = int((one["status"] == "run").sum()), one.loc[one["status"] != "run", "alternative"]
    n_grid = int((summary["group"] == "weight_grid").sum())

    fig, ax = plt.subplots(figsize=(7.5, 7.4), layout="constrained")
    ys = np.arange(len(table))
    offset = 0.17
    ax.hlines(ys - offset, table["sens_rank_min"], table["sens_rank_max"], color=style["text"],
              linewidth=2.2, zorder=2, label=f"One at a time ({n_run} tests)")
    ax.hlines(ys + offset, table["sens_grid_rank_min"], table["sens_grid_rank_max"], color=style["note"],
              linewidth=2.2, zorder=2, label=f"Weight grid ({n_grid} sets)")
    for field, yo, color in (("sens_rank", -offset, style["text"]), ("sens_grid_rank", offset, style["note"])):
        for end in ("min", "max"):
            ax.scatter(table[f"{field}_{end}"], ys + yo, s=10, color=color, zorder=3, linewidth=0)
    ax.scatter(table["score_rank"], ys, s=46, color=style["highlight"], edgecolor=style["background"],
               linewidth=0.8, zorder=4, label="Main-run rank")
    ax.set_yticks(ys, table["name"], fontsize=7)
    ax.tick_params(axis="y", colors=style["text"], length=0)
    ax.invert_yaxis()
    ax.set_xlim(0.3, len(table) + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25, len(table)])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Rank (1 = most exposed)", fontsize=8)
    ax.grid(axis="x", color=style["grid"], linewidth=0.6, zorder=0)
    clean_axes(ax, ("top", "right", "left"))
    handles, labels = ax.get_legend_handles_labels()
    bottom_legend(ax, handles, labels, fontsize=7, y=-0.06)
    source_note(fig, "Sensitivity tests: S10 (sensitivity.csv, sensitivity_ranks.csv).", style)

    wide = table.nlargest(3, "sens_rank_range")
    log.info("Task 6.7 | widest one-at-a-time ranges: %s",
             ", ".join(f"{r['name']} {r['sens_rank_min']}-{r['sens_rank_max']}" for _, r in wide.iterrows()))
    stable = table[(table["sens_rank_max"] <= settings["outputs"]["top_n_labels"])]
    top = settings["outputs"]["top_n_labels"]
    always = (f"{len(stable)} chokepoints stay in the top {top} under every one-at-a-time test "
              f"({', '.join(stable['name'])})" if len(stable) else
              f"No chokepoint stays in the top {top} under every one-at-a-time test")
    caption = (f"Figure 6.7. Sensitivity of each chokepoint's rank (amber dot: main run). The upper line "
               f"spans the best to worst rank over the main run and {n_run} one-at-a-time alternatives "
               f"(one choice changed at a time, Table 4); the lower line spans the {n_grid} weight "
               f"combinations of the weight grid (each weight 0.1 to 0.8 in steps of 0.1, summing to 1). "
               f"{always}. Not run (they need a full "
               f"reroute): {', '.join(not_run) or 'none'}. Source: S10 sensitivity tests.")
    return save_figure(fig, "fig_6_7_sensitivity.png", settings, log), caption


# ---------------------------------------------------------------------
# Task 6.8: results table
# ---------------------------------------------------------------------

def results_table(points, settings, log):
    """Task 6.8: one row per chokepoint with the score, components, rank ranges and impact."""
    ring = settings["conflict"]["ring_main_km"]
    trade = settings["scoring"]["correlation_main"]["trade"]
    columns = ["score_rank", "name", "chokepoint_id", "score_composite", trade, f"viol_n_{ring}",
               f"tens_n_{ring}", "score_redund_pct", "score_trade", "score_conflict", "score_redund",
               "sens_rank_min", "sens_rank_max", "sens_grid_rank_min", "sens_grid_rank_max",
               "impact_days", "impact_days_low", "impact_days_high", "impact_voyage_cost",
               "impact_voyage_cost_low", "impact_voyage_cost_high", "impact_holding_cost",
               "impact_holding_cost_low", "impact_holding_cost_high"]
    table = points.sort_values("score_rank")
    out = pd.DataFrame(table[columns])
    out["impact_note"] = table.apply(impact_note, axis=1).to_numpy()
    path = common.get_path(settings, "tables") / "results_table.csv"
    out.to_csv(path, index=False)
    log.info("Task 6.8 | wrote %s (%d rows, %d columns)", path, len(out), len(out.columns))
    return out


def fig_results_table(table, settings, log):
    """Task 6.8: the results table as a formatted figure for the paper."""
    style = settings["outputs"]["style"]
    ring = settings["conflict"]["ring_main_km"]
    trade = settings["scoring"]["correlation_main"]["trade"]

    def small(v, scale, digits, floor):
        """Value in units of `scale`; below `floor` shown as "<floor" rather than a misleading 0."""
        if pd.isna(v):
            return EMPTY_CELL
        return f"<{floor:g}" if 0 < v / scale < floor else f"{v / scale:,.{digits}f}"

    rows = []
    for _, r in table.iterrows():
        note = r["impact_note"]
        rows.append([f"{int(r['score_rank'])}", r["name"], f"{r['score_composite']:.3f}",
                     small(r[trade], 1e6, 1, 0.1), f"{int(r[f'viol_n_{ring}']):,}", f"{int(r[f'tens_n_{ring}']):,}",
                     f"{r['score_redund_pct']:.1f}",
                     f"{int(r['sens_rank_min'])}–{int(r['sens_rank_max'])}",
                     f"{int(r['sens_grid_rank_min'])}–{int(r['sens_grid_rank_max'])}",
                     note or f"{r['impact_days']:.2f}", EMPTY_CELL if note else small(r["impact_voyage_cost"], 1e3, 0, 1),
                     EMPTY_CELL if note else small(r["impact_holding_cost"], 1e3, 0, 1)])
    headers = ["Rank", "Chokepoint", "Score", "Trade\nUS$ M", "Violent\nevents", "Tension\nevents",
               "Redund.\n%", "Rank range\none at a time", "Rank range\nweight grid", "Extra days\nat sea",
               "Cost per\nvoyage US$ K", "REE holding\ncost US$ K"]

    fig, ax = plt.subplots(figsize=(10, 8.2), layout="constrained")
    ax.set_axis_off()
    widths = [0.04, 0.15, 0.06, 0.07, 0.07, 0.07, 0.07, 0.09, 0.09, 0.11, 0.09, 0.09]
    tbl = ax.table(cellText=rows, colLabels=headers, colWidths=widths, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.8)
    tbl.scale(1, 1.32)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor(style["grid"])
        cell.set_linewidth(0.5)
        cell.set_facecolor(style["background"] if i % 2 else "#0f2438")
        cell.get_text().set_color(style["text"])
        if i == 0:
            cell.set_facecolor(style["land"])
            cell.get_text().set_color(style["title"])
            cell.get_text().set_fontweight("bold")
            cell.set_height(cell.get_height() * 1.9)
        elif rows[i - 1][9] in ("no sea route", "no affected flows") and j == 9:
            cell.get_text().set_color(style["note"])
            cell.get_text().set_fontstyle("italic")
    source_note(fig, "Data: S09 composite scores; S10 sensitivity; S08b reroute impact.", style)
    # Footnotes at the bottom left, in space reserved below the table.
    cap = table["score_redund_pct"].max()
    footnotes = [f"Redund. {cap:.1f}% = no-route cap (Decision 16).",
                 "REE holding cost is a 2025 total, not per voyage (Decision 13)."]
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.text(0.005, 0.008, "\n".join(footnotes), ha="left", va="bottom", fontsize=6.5, color=style["note"],
             linespacing=1.4)
    log.info("Task 6.8 | formatted table: %d rows", len(rows))

    caption = (f"Table 6.8. Results for the {len(rows)} included chokepoints, by composite rank. Trade: "
               f"2025 maritime REE trade crossing the chokepoint (baseline); violent and tension events "
               f"within {ring} km; redundancy: Equation 4 extra distance (Decision 16 cap). Rank ranges: "
               f"best to worst rank under the one-at-a-time sensitivity tests (main run included) and "
               f"under the weight grid. Reroute impact (reported beside the score, not part of it; "
               f"Decision 8): value-weighted mean extra days at sea and extra cost per typical container-"
               f"ship voyage if the chokepoint closes, and the 2025 REE holding cost of the delay; "
               f"chokepoints whose affected trade has no sea route, or with no affected flows, have no "
               f"cost (shown as a dash). {' '.join(footnotes)} The full table with low/high ranges is "
               f"outputs/tables/results_table.csv.")
    return save_figure(fig, "fig_6_8_results_table.png", settings, log), caption


# ---------------------------------------------------------------------
# Decision 8: reroute impact ("days and dollars")
# ---------------------------------------------------------------------

def fig_impact(points, settings, log):
    """Extra days at sea and extra cost per voyage with S08b's low-high ranges, by rank."""
    style = settings["outputs"]["style"]
    table = points.sort_values("score_rank").copy()
    table["note"] = table.apply(impact_note, axis=1)
    ys = np.arange(len(table))
    costed_rows = (table["note"] == "").to_numpy()
    panels = [("impact_days", "(a) Extra days at sea per voyage", 1.0),
              ("impact_voyage_cost", "(b) Extra cost per voyage, US$ M", 1e6)]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 7.4), sharey=True, layout="constrained")
    for ax, (field, title, scale) in zip(axes, panels):
        has = table[costed_rows]
        ax.hlines(ys[costed_rows], has[f"{field}_low"] / scale, has[f"{field}_high"] / scale,
                  color=style["note"], linewidth=1.6, zorder=2)
        ax.scatter(has[field] / scale, ys[costed_rows], s=30, color=style["text"],
                   edgecolor=style["background"], linewidth=0.8, zorder=3)
        for yv, note in zip(ys, table["note"]):
            if note:
                ax.text(0.01, yv, note, transform=ax.get_yaxis_transform(), va="center", fontsize=6.5,
                        color=style["note"], fontstyle="italic")
        ax.set_xlim(left=0)
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", color=style["grid"], linewidth=0.6, zorder=0)
        clean_axes(ax, ("top", "right", "left"))
    axes[0].set_yticks(ys, [f"{int(r)}. {n}" for r, n in zip(table["score_rank"], table["name"])], fontsize=7)
    axes[0].tick_params(axis="y", colors=style["text"], length=0)
    axes[1].tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5, markerfacecolor=style["text"],
                      markeredgecolor=style["background"]),
               Line2D([], [], color=style["note"], linewidth=1.6)]
    fig.legend(handles, ["Main estimate (value-weighted mean)", "Low–high range (one parameter at a time)"],
               loc="outside upper center", ncol=2, frameon=False, fontsize=7)
    source_note(fig, "Data: S08 detours; S08b reroute cost (settings reroute_impact).", style)

    costed = table[table["note"] == ""]
    for note in ("no sea route", "no affected flows"):
        log.info("Decision 8 | %s: %s", note, ", ".join(table.loc[table["note"] == note, "name"]) or "none")
    log.info("Decision 8 | %d chokepoints costed; largest extra days %s (%.2f), largest cost per voyage %s "
             "(US$ %s)", len(costed), costed.loc[costed["impact_days"].idxmax(), "name"],
             costed["impact_days"].max(), costed.loc[costed["impact_voyage_cost"].idxmax(), "name"],
             f"{costed['impact_voyage_cost'].max():,.0f}")
    p = settings["reroute_impact"]
    caption = (f"Figure (reroute impact). Extra days at sea (a) and extra cost per typical container-ship "
               f"voyage (b) if each chokepoint closed, for the {len(costed)} chokepoints whose affected "
               f"trade can be rerouted by sea; rows follow the composite rank. Dots are value-weighted "
               f"means over the rerouted sub-flows (Decision 13); lines span the lowest and highest "
               f"result when one parameter at a time (speed and fuel use paired, fuel price, charter "
               f"rate) is set to its low or high value. Cost = extra days x (daily charter cost + daily "
               f"fuel use x fuel price) plus or minus the change in Suez and Panama tolls. \"No sea route\": "
               f"all affected trade has no sea alternative or starts or ends at a port inside the zone; "
               f"\"no affected flows\": no 2025 REE trade crosses it. Panama's extra cost is low relative to "
               f"its extra days because detours avoid the Panama Canal toll. This is an impact layer reported "
               f"beside the score, not part of it (Decision 8). Sources: S08, S08b; parameters in "
               f"settings reroute_impact ({p['vessel_type'].replace('_', ' ')}).")
    return save_figure(fig, "fig_reroute_impact.png", settings, log), caption


# ---------------------------------------------------------------------
# Context: Red Sea transits
# ---------------------------------------------------------------------

def fig_red_sea(settings, log):
    """Grouped bars: container transits per year at the Red Sea and Cape chokepoints."""
    path = common.get_path(settings, "tables") / "red_sea_context.csv"
    table = pd.read_csv(path)
    context = settings["routing"]["red_sea_context"]
    names, years = context["chokepoints"], context["years"]
    log.info("Context | read %s (%d rows)", path, len(table))
    style = settings["outputs"]["style"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    width = 0.8 / len(years)
    xs = np.arange(len(names))
    for i, (year, color) in enumerate(zip(years, style["year_scale"])):
        values = [table.loc[(table["name"] == n) & (table["year"] == year), "transits_container"].sum()
                  for n in names]
        bars = ax.bar(xs + (i - (len(years) - 1) / 2) * width, values, width * 0.92, color=color,
                      label=str(year), zorder=2)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in values], fontsize=6.5, padding=2)
        log.info("Context | %d container transits: %s", year,
                 ", ".join(f"{n} {v:,.0f}" for n, v in zip(names, values)))
    ax.set_xticks(xs, names, fontsize=8)
    ax.set_ylabel("Container ship transits per year", fontsize=8)
    ax.tick_params(axis="x", colors=style["text"])
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color=style["grid"], linewidth=0.6, zorder=0)
    clean_axes(ax, ("top", "right", "left"))
    handles, labels = ax.get_legend_handles_labels()
    bottom_legend(ax, handles, labels, fontsize=7, y=-0.08)
    source_note(fig, "Data: IMF PortWatch.", style)

    first, last = (table[table["year"] == yr].set_index("name")["transits_container"] for yr in (years[0], years[-1]))
    change = "; ".join(f"{n} {100 * (last[n] / first[n] - 1):+.0f}%" for n in names)
    caption = (f"Figure (context). Container ship transits per calendar year through the "
               f"{', '.join(names[:-1])} and past the {names[-1]}, {years[0]}-{years[-1]} (2024 "
               f"has 366 days). From {years[0]} to {years[-1]}, {change}. This supports the Red Sea "
               f"diversion scenario as the realistic one for 2025. Source: IMF PortWatch daily "
               f"chokepoint transits.")
    return save_figure(fig, "fig_red_sea_context.png", settings, log), caption


# ---------------------------------------------------------------------

def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    crs = settings["outputs"]["static_map_crs"]
    apply_style(settings["outputs"]["style"])

    chokepoints, routes, subflows = read_inputs(settings, log)
    routes = route_values(routes, subflows, log)
    basemap = read_basemap(settings, log)
    world = basemap.to_crs(crs)
    outline = world_outline(crs)
    points = chokepoints.to_crs(crs)
    routes_eq = routes.to_crs(crs)
    log.info("Static world maps projected to %s", crs)

    figures = [
        fig_composite(points, routes_eq, world, outline, settings, log),
        fig_components(points, world, outline, settings, log),
        fig_routes(routes_eq, points, world, outline, settings, log),
        fig_closeups(chokepoints, routes, basemap, settings, log),
        fig_trade_conflict(chokepoints, settings, log),
        fig_breakdown(chokepoints, settings, log),
        fig_sensitivity(chokepoints, settings, log),
        fig_results_table(results_table(chokepoints, settings, log), settings, log),
        fig_impact(chokepoints, settings, log),
        fig_red_sea(settings, log),
    ]
    for paths, caption in figures:
        log.info("Caption %s: %s", paths[0].name, caption)
    log.info("Run of s11_figures finished")


if __name__ == "__main__":
    main()

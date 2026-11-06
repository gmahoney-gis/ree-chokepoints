"""S11b: poster versions of five figures (task 6.14).

Draws each figure at its width on the 36 x 24 in poster, so that at 100% size all text is at
least outputs.poster.font_pt.minimum. Reads the same database and tables as S11 and imports
its helpers; s11_figures.py and the paper figures are not changed. Labels are plain language,
use the short names in outputs.poster.short_names, and carry no data line (the poster footer
credits the data). Written to paths.figures_poster at outputs.poster.dpi:

1. fig_6_1_composite_map.png: ranked composite map (top outputs.top_n_labels labelled).
2. fig_6_5_trade_conflict.png: trade vs. violence; every point drawn, those in
   outputs.poster.trade_conflict.labels named.
3. fig_6_6_score_breakdown.png: stacked score parts, all chokepoints, by rank.
4. fig_reroute_impact.png: extra days and cost per voyage for the outputs.poster.impact.rows
   largest extra-days values, largest first; chokepoints with no sea route are listed below.
5. fig_study_area.png: sea-lane network and all chokepoints on a flat lon/lat map (rebuilt from
   the report's Figure 1, scripts/prototypes/searoute_visual.py, on the current zones and gates).

Map and scatter labels sit at hand-set positions from the settings. The run stops if any drawn
text is below the minimum or leaves the figure, or if a label's nearest circle is not its own.
The log gives each figure's final size and the counts from these checks.
"""

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import searoute
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from shapely.geometry import LineString

import common
import s03_chokepoints as s03
import s11_figures as s11


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def short_names(points, poster):
    """Short poster name of each chokepoint, in row order."""
    return points["chokepoint_id"].map(poster["short_names"])


def poster_halo(settings):
    """S11's label halo at the poster's stroke width."""
    return s11.halo({**settings["outputs"]["style"],
                     "label_halo_width": settings["outputs"]["poster"]["label_halo_width"]})


def named_labels(ax, xs, ys, texts, offsets, line_color, settings, **text_kw):
    """Haloed labels with a leader to each point.

    Each offset is [x, y (points), alignment], with an optional fourth value: where along the
    bottom of the label the leader starts (0 left, 1 right). Without it the leader aims at the
    label's centre.
    """
    poster = settings["outputs"]["poster"]
    labels = []
    for x, y, text, (dx, dy, ha, *start) in zip(xs, ys, texts, offsets):
        arrow = {"arrowstyle": "-", "color": line_color, "lw": poster["leader_width"],
                 "shrinkA": poster["leader_gap_label"], "shrinkB": poster["leader_gap_point"]}
        if start:
            arrow["relpos"] = (start[0], 0)
        labels.append(ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                                  fontsize=poster["font_pt"]["label"], zorder=5, arrowprops=arrow,
                                  path_effects=poster_halo(settings), **text_kw))
    return labels


def box_gap(box, x, y, radius):
    """Screen distance from a box to the edge of a circle at (x, y); negative if they overlap."""
    return np.hypot(max(box.x0 - x, 0, x - box.x1), max(box.y0 - y, 0, y - box.y1)) - radius


def check_labels(fig, ax, labels, points, radii_pt, name, settings, log):
    """Check hand-set labels: stop unless each one's nearest circle is its own point.

    `points` holds chokepoint_id, x and y (data coordinates) and `radii_pt` the circle radii in
    points. A pair in outputs.poster.nearest_circle_exceptions is left out of the test and logged.
    Also logs how many labels overlap another label, touch another circle or leave the figure.
    """
    exceptions = settings["outputs"]["poster"]["nearest_circle_exceptions"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [s11.text_box(label, renderer) for label in labels]
    radii = np.broadcast_to(radii_pt, len(points)) * fig.dpi / 72
    centres = ax.transData.transform(points[["x", "y"]].to_numpy())
    ids = points["chokepoint_id"].to_numpy()
    touching = 0
    for label, box in zip(labels, boxes):
        is_own = np.hypot(*(centres - ax.transData.transform(label.xy)).T) < 1
        gaps = np.array([box_gap(box, x, y, r) for (x, y), r in zip(centres, radii)])
        excused = ids == exceptions.get(ids[is_own][0])
        own_gap, other_gap = gaps[is_own].min(), gaps[~is_own & ~excused].min()
        if other_gap <= own_gap:
            raise ValueError(f"{name}: the circle nearest the label '{label.get_text()}' is not its own "
                             f"(own {own_gap:.1f} px, other {other_gap:.1f} px)")
        touching += int(gaps[~is_own].min() <= 0)
        if excused.any() and gaps[excused].min() <= own_gap:
            log.info("%s | exception: '%s' is nearer %s than its own circle (outputs.poster."
                     "nearest_circle_exceptions)", name, label.get_text(), ids[excused][0])
    overlaps = sum(a.overlaps(b) for a, b in itertools.combinations(boxes, 2))
    outside = sum(b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1 for b in boxes)
    log.info("%s | %d labels pass the nearest-circle test: %d overlap another label, %d touch another "
             "circle, %d leave the figure", name, len(labels), overlaps, touching, outside)


def point_table(points, x=None, y=None):
    """Chokepoint IDs with plotting coordinates (the point geometry unless x and y are given)."""
    return pd.DataFrame({"chokepoint_id": points["chokepoint_id"].to_numpy(),
                         "x": points.geometry.x.to_numpy() if x is None else np.asarray(x),
                         "y": points.geometry.y.to_numpy() if y is None else np.asarray(y)})


def drawn_texts(fig):
    """Every text the figure actually draws (not the axis text of axes that are switched off)."""
    hidden = {t for ax in fig.axes if not ax.axison for axis in (ax.xaxis, ax.yaxis) for t in axis.findobj(Text)}
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t not in hidden]


def save_poster(fig, name, settings, log):
    """Check all drawn text, then save at poster size and resolution.

    Stops if any text is below the poster minimum, leaves the figure or overlaps other text.
    """
    poster = settings["outputs"]["poster"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = drawn_texts(fig)
    smallest = min(t.get_fontsize() for t in texts)
    if smallest < poster["font_pt"]["minimum"]:
        raise ValueError(f"{name}: text at {smallest} pt is below the poster minimum")
    boxes = [Text.get_window_extent(t, renderer) for t in texts]
    cut = [t.get_text() for t, b in zip(texts, boxes)
           if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1]
    if cut:
        raise ValueError(f"{name}: text leaves the figure: {cut}")
    clashes = [(a.get_text(), b.get_text()) for (a, x), (b, y) in itertools.combinations(zip(texts, boxes), 2)
               if x.overlaps(y)]
    if clashes:
        raise ValueError(f"{name}: text overlaps: {clashes}")
    folder = common.get_path(settings, "figures_poster")
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=poster["dpi"])
    width, height = fig.get_size_inches()
    log.info("%s | %d drawn texts, smallest %.1f pt, no text overlaps", name, len(texts), smallest)
    log.info("Wrote %s (%.2f x %.2f in, %d dpi)", folder / name, width, height, poster["dpi"])
    plt.close(fig)


# ---------------------------------------------------------------------
# Composite map
# ---------------------------------------------------------------------

def poster_composite(points, routes, basemap, outline, settings, log):
    """Composite map at poster size: S11 fig_composite with larger circles, text and routes."""
    out = settings["outputs"]
    style, poster = out["style"], out["poster"]
    spec, font = poster["composite"], poster["font_pt"]
    cmap, _, breaks = s11.score_classes(settings)
    areas = np.array(spec["class_areas"])

    xmin, ymin, xmax, ymax = outline.total_bounds
    width, margin, band = spec["width_in"], spec["map_margin_in"], spec["legend_band_in"]
    map_width = width - 2 * margin
    map_height = map_width * (ymax - ymin) / (xmax - xmin)
    height = margin + map_height + band
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([margin / width, band / height, map_width / width, map_height / height])
    s11.draw_basemap(ax, basemap, outline, style)

    base = s11.baseline_routes(routes)
    widths = np.maximum(spec["route_max_width"] * base["route_value_usd"] / routes["route_value_usd"].max(),
                        spec["route_min_width"])
    base.plot(ax=ax, color=style["route"], linewidth=widths, alpha=spec["route_alpha"], zorder=2)
    order = points.sort_values("score_composite")
    classes = np.digitize(order["score_composite"], breaks[1:-1])
    ax.scatter(order.geometry.x, order.geometry.y, s=areas[classes], facecolor=cmap(classes),
               edgecolor=style["zero_marker"], linewidth=spec["marker_edge_width"], zorder=3)

    # Legend: one entry per class, at its circle size.
    handles = [Line2D([], [], linestyle="none")]
    labels = ["Composite score"]
    for i, color in enumerate(cmap.colors):
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(areas[i]),
                              markerfacecolor=color, markeredgecolor=style["zero_marker"],
                              markeredgewidth=spec["marker_edge_width"]))
        labels.append(f"{breaks[i]:.1f}–{breaks[i + 1]:.1f}")
    legend = s11.bottom_legend(ax, handles, labels, fontsize=font["legend"],
                               columnspacing=spec["legend_column_spacing"],
                               handletextpad=spec["legend_handle_pad"], borderpad=spec["legend_border_pad"])
    legend.get_texts()[0].set_color(style["note"])

    # Top-n labels; S11b stops with a KeyError if a new top-n chokepoint has no position yet.
    top = points.nsmallest(out["top_n_labels"], "score_rank")
    labels_drawn = named_labels(
        ax, top.geometry.x, top.geometry.y,
        [f"{int(r)}  {n}" for r, n in zip(top["score_rank"], short_names(top, poster))],
        [spec["labels"][cid] for cid in top["chokepoint_id"]], style["text"], settings,
        fontweight="bold", color=style["text"])
    name = "fig_6_1_composite_map.png"
    check_labels(fig, ax, labels_drawn, point_table(order), np.sqrt(areas[classes]) / 2, name, settings, log)
    save_poster(fig, name, settings, log)


# ---------------------------------------------------------------------
# Trade vs. conflict chart
# ---------------------------------------------------------------------

def poster_trade_conflict(points, settings, log):
    """Trade vs. violence at poster size: S11 fig_trade_conflict with named points only."""
    style, poster = settings["outputs"]["style"], settings["outputs"]["poster"]
    spec, font = poster["trade_conflict"], poster["font_pt"]
    xf = settings["scoring"]["correlation_main"]["trade"]
    yf = settings["scoring"]["correlation_main"]["conflict"]
    main = s11.read_correlation(settings, log)
    x, y = np.log1p(points[xf]), np.log1p(points[yf])
    x_med, y_med = points[xf].median(), points[yf].median()
    s11.quadrant_check(points, xf, yf, x_med, y_med, settings)

    fig, ax = plt.subplots(figsize=(spec["width_in"], spec["height_in"]), layout="constrained")
    ax.axvline(np.log1p(x_med), color=style["note"], linestyle="--", linewidth=spec["median_line_width"], zorder=1)
    ax.axhline(np.log1p(y_med), color=style["note"], linestyle="--", linewidth=spec["median_line_width"], zorder=1)
    ax.scatter(x, y, s=spec["marker_area"], color=style["highlight"], edgecolor=style["background"],
               linewidth=spec["marker_edge_width"], zorder=3)

    ax.margins(x=spec["axis_margin"], y=spec["axis_margin"])
    ax.set_xlim(ax.get_xlim()[0], spec["x_max_log"])
    ax.set_ylim(spec["y_min_log"], ax.get_ylim()[1] + spec["y_top_pad_log"])
    x_ticks = spec["x_ticks"]
    ax.set_xticks(np.log1p([value for value, _ in x_ticks]), [text for _, text in x_ticks])
    ax.set_yticks(np.log1p(spec["y_ticks"]), [f"{t:,}" for t in spec["y_ticks"]])
    ax.set_xlabel(f"REE trade crossing, {settings['project']['study_year']} (log scale)", fontsize=font["label"])
    ring = settings["conflict"]["ring_main_km"]
    ax.set_ylabel(f"Violent events within {ring} km (log scale)", fontsize=font["label"])
    ax.tick_params(labelsize=font["tick"], length=poster["tick_length"], width=poster["axis_line_width"])
    s11.clean_axes(ax)
    for spine in ax.spines.values():
        spine.set_linewidth(poster["axis_line_width"])

    # Quadrant words: trade above the plot, violence inside it beside the y axis.
    mid = ax.transAxes.inverted().transform(ax.transData.transform((np.log1p(x_med), np.log1p(y_med))))
    words = {"transform": ax.transAxes, "fontsize": font["label"], "color": style["note"], "fontstyle": "italic"}
    word_y, word_x, word_gap = spec["trade_words_y"], spec["violence_words_x"], spec["violence_words_gap"]
    ax.text(mid[0] - spec["quadrant_gap"], word_y, "← Low trade", ha="right", va="bottom", **words)
    ax.text(mid[0] + spec["quadrant_gap"], word_y, "High trade →", ha="left", va="bottom", **words)
    ax.text(word_x, mid[1] + word_gap, "High violence", ha="left", va="bottom", **words)
    ax.text(word_x, mid[1] - word_gap, "Low violence", ha="left", va="top", **words)
    ax.text(*spec["stats_box_xy"], f"Spearman ρ = {main['rho']:.2f}\np = {main['p_value']:.2f}, n = {int(main['n'])}",
            transform=ax.transAxes, va="top", fontsize=font["label"],
            bbox={"boxstyle": f"round,pad={spec['stats_box_pad']}", "fc": style["background"], "ec": style["borders"],
                  "lw": spec["stats_box_line_width"]})

    # Constrained layout runs once on the finished frame, then is frozen so label positions hold.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    named = points[points["chokepoint_id"].isin(spec["labels"])]
    labels = named_labels(ax, np.log1p(named[xf]), np.log1p(named[yf]), short_names(named, poster),
                          [spec["labels"][cid] for cid in named["chokepoint_id"]], style["note"], settings,
                          color=style["text"])
    name = "fig_6_5_trade_conflict.png"
    check_labels(fig, ax, labels, point_table(points, x, y), np.sqrt(spec["marker_area"]) / 2, name, settings, log)
    save_poster(fig, name, settings, log)


# ---------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------

def poster_breakdown(points, settings, log):
    """Score breakdown at poster size: S11 fig_breakdown with short names and one note."""
    style, poster = settings["outputs"]["style"], settings["outputs"]["poster"]
    spec, font = poster["breakdown"], poster["font_pt"]
    table = points.sort_values("score_rank")
    parts = [("score_part_trade", "Trade", style["route"]),
             ("score_part_conflict", "Conflict", style["highlight"]),
             ("score_part_redund", "Redundancy", style["redundancy"])]
    if (table[[p for p, _, _ in parts]].sum(axis=1) - table["score_composite"]).abs().max() > 1e-6:
        raise ValueError("score_part_* columns do not add up to score_composite")

    height = len(table) * spec["row_pitch_in"] + spec["frame_in"]
    fig, ax = plt.subplots(figsize=(spec["width_in"], height), layout="constrained")
    ys = np.arange(len(table))
    left = np.zeros(len(table))
    for field, label, color in parts:
        ax.barh(ys, table[field], left=left, height=spec["bar_height"], color=color,
                edgecolor=style["background"], linewidth=spec["bar_edge_width"], label=label, zorder=2)
        left += table[field].to_numpy()
    for yv, (_, row) in zip(ys, table.iterrows()):
        note = "  no affected flows" if row["score_redund_filled"] else ""
        ax.text(row["score_composite"] + spec["value_gap"], yv, f"{row['score_composite']:.2f}{note}",
                va="center", fontsize=font["label"], color=style["text"])
    ax.set_yticks(ys, [f"{int(r)}. {n}" for r, n in zip(table["score_rank"], short_names(table, poster))],
                  fontsize=font["tick"])
    ax.tick_params(axis="y", colors=style["text"], length=0)
    ax.invert_yaxis()
    ax.set_ylim(len(table) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks(spec["x_ticks"])
    ax.set_xlabel("Composite score", fontsize=font["label"])
    ax.tick_params(axis="x", labelsize=font["tick"], length=poster["tick_length"], width=poster["axis_line_width"])
    ax.grid(axis="x", color=style["grid"], linewidth=spec["grid_width"], zorder=0)
    s11.clean_axes(ax, ("top", "right", "left"))
    ax.spines["bottom"].set_linewidth(poster["axis_line_width"])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(parts), frameon=False, fontsize=font["legend"],
               columnspacing=spec["legend_column_spacing"], handletextpad=spec["legend_handle_pad"])
    fig.canvas.draw()
    pitch = ax.get_window_extent().height / fig.dpi / len(table)
    log.info("fig_6_6_score_breakdown.png | %d rows at %.3f in each", len(table), pitch)
    save_poster(fig, "fig_6_6_score_breakdown.png", settings, log)


# ---------------------------------------------------------------------
# Reroute impact
# ---------------------------------------------------------------------

def poster_impact(points, settings, log):
    """Reroute impact at poster size: the largest extra-days values, largest first, in both panels."""
    style, poster = settings["outputs"]["style"], settings["outputs"]["poster"]
    spec, font = poster["impact"], poster["font_pt"]
    notes = points.apply(s11.impact_note, axis=1)
    table = points[notes == ""].nlargest(spec["rows"], "impact_days")
    no_route = points[notes == "no sea route"].sort_values("score_rank")
    ys = np.arange(len(table))
    panels = [("impact_days", "(a) Extra\ndays at sea", 1.0, spec["x_ticks_days"]),
              ("impact_voyage_cost", "(b) Extra\ncost, US$ M", 1e6, spec["x_ticks_cost"])]

    fig, axes = plt.subplots(1, 2, figsize=(spec["width_in"], spec["height_in"]), sharey=True, layout="constrained")
    for ax, (field, title, scale, ticks) in zip(axes, panels):
        ax.hlines(ys, table[f"{field}_low"] / scale, table[f"{field}_high"] / scale, color=style["note"],
                  linewidth=spec["range_line_width"], zorder=2)
        ax.scatter(table[field] / scale, ys, s=spec["marker_area"], color=style["text"],
                   edgecolor=style["background"], linewidth=spec["marker_edge_width"], zorder=3)
        ax.set_xlim(left=0)
        ax.set_xticks(ticks)
        ax.set_title(title, loc="left", fontsize=font["panel_title"], fontweight="bold")
        ax.tick_params(axis="x", labelsize=font["tick"], length=poster["tick_length"], width=poster["axis_line_width"])
        ax.grid(axis="x", color=style["grid"], linewidth=spec["grid_width"], zorder=0)
        s11.clean_axes(ax, ("top", "right", "left"))
        ax.spines["bottom"].set_linewidth(poster["axis_line_width"])
    axes[0].set_yticks(ys, short_names(table, poster), fontsize=font["tick"])
    axes[0].tick_params(axis="y", colors=style["text"], length=0)
    axes[1].tick_params(axis="y", length=0)
    axes[0].invert_yaxis()

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=spec["legend_marker_size"],
                      markerfacecolor=style["text"], markeredgecolor=style["background"]),
               Line2D([], [], color=style["note"], linewidth=spec["range_line_width"])]
    fig.legend(handles, ["Trade-weighted average", "Low–high range"], loc="outside upper center", ncol=2,
               frameon=False, fontsize=font["legend"], columnspacing=spec["legend_column_spacing"],
               handletextpad=spec["legend_handle_pad"])
    band = spec["note_band_in"] / spec["height_in"]
    fig.get_layout_engine().set(rect=(0, band, 1, 1 - band))
    fig.text(0.5, band / 2, "No sea route: " + ", ".join(short_names(no_route, poster)), ha="center", va="center",
             fontsize=font["label"], color=style["note"])
    log.info("fig_reroute_impact.png | rows: %s", ", ".join(short_names(table, poster)))
    save_poster(fig, "fig_reroute_impact.png", settings, log)


# ---------------------------------------------------------------------
# Study-area map
# ---------------------------------------------------------------------

def network_lines(settings, log):
    """searoute's sea-lane lines inside -180..180, without the open-ocean lattice.

    A line that starts or ends on the lattice of ocean_grid_step_deg is left out, as in the
    report's Figure 1 (display only; routing is unaffected).
    """
    spec = settings["outputs"]["poster"]["study_area"]
    step = spec["ocean_grid_step_deg"]

    def on_lattice(point):
        return all(abs(v / step - round(v / step)) < 1e-6 for v in point[:2])

    source = gpd.read_file(Path(searoute.__file__).parent / "data" / spec["network_file"])
    lines, hidden = [], 0
    for geometry in source.geometry:
        for line in getattr(geometry, "geoms", [geometry]):
            coords = [point[:2] for point in line.coords]
            if on_lattice(coords[0]) or on_lattice(coords[-1]):
                hidden += 1
            else:
                lines += list(common.split_antimeridian(coords).geoms)
    log.info("fig_study_area.png | network: %s pieces drawn, %s lattice lines left out", f"{len(lines):,}", f"{hidden:,}")
    return gpd.GeoSeries(lines, crs=settings["project"]["storage_crs"])


def highlighted_routes(lines, shapes, chokepoints, settings):
    """Network lines that pass through a chokepoint's zone or gate, cut to highlight_km around it."""
    reach = settings["outputs"]["poster"]["study_area"]["highlight_km"]
    pieces = []
    for shape in shapes.itertuples():
        point = chokepoints.loc[chokepoints["chokepoint_id"] == shape.chokepoint_id].geometry.iloc[0]
        circle = s03.true_distance_circle(point.x, point.y, reach, settings)
        hits = lines.iloc[lines.sindex.query(shape.geometry, predicate="intersects")]
        for part in hits.intersection(circle):
            pieces += [g for g in getattr(part, "geoms", [part]) if isinstance(g, LineString) and not g.is_empty]
    return pieces


def line_collection(lines, color, width, alpha, zorder):
    """A set of lines as one matplotlib collection."""
    return LineCollection([np.array(line.coords)[:, :2] for line in lines], colors=color, linewidths=width,
                          alpha=alpha, zorder=zorder, capstyle="round")


def poster_study_area(basemap, settings, log):
    """Study-area map: sea lanes, routes through each chokepoint, and all chokepoints (flat lon/lat)."""
    style, poster = settings["outputs"]["style"], settings["outputs"]["poster"]
    spec, font = poster["study_area"], poster["font_pt"]
    points = common.read_layer(settings, "chokepoints")
    zones = common.read_layer(settings, "zones")
    zones = zones[zones["included"].str.strip().str.lower() == "yes"].reset_index(drop=True)
    shapes = common.crossing_geometries(zones, settings)

    lines = network_lines(settings, log)
    highlight = highlighted_routes(lines, shapes, points, settings)
    log.info("fig_study_area.png | %d chokepoints drawn; %d with a zone or gate (%s has none); %s highlighted "
             "pieces within %d km", len(points), len(shapes),
             ", ".join(points.loc[~points["chokepoint_id"].isin(shapes["chokepoint_id"]), "name"]),
             f"{len(highlight):,}", spec["highlight_km"])

    west, east, south, north = spec["extent"]
    width, margin, band = spec["width_in"], spec["map_margin_in"], spec["legend_band_in"]
    map_width = width - 2 * margin
    map_height = map_width * (north - south) / (east - west)
    height = margin + map_height + band
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([margin / width, band / height, map_width / width, map_height / height])
    ax.set_facecolor(style["background"])

    ax.add_collection(line_collection(lines, style["route"], spec["network_glow_width"], spec["network_glow_alpha"], 1))
    ax.add_collection(line_collection(lines, style["route"], spec["network_width"], spec["network_alpha"], 1))
    # Land over the network hides inland waterways; the coast outlines the continents.
    land = gpd.GeoSeries([basemap.union_all()], crs=basemap.crs)
    land.plot(ax=ax, color=style["land"], edgecolor=style["borders"], linewidth=spec["coast_width"], zorder=1.5)
    ax.add_collection(line_collection(highlight, style["highlight"], spec["highlight_glow_width"],
                                      spec["highlight_glow_alpha"], 3))
    ax.add_collection(line_collection(highlight, style["highlight"], spec["highlight_width"], 1.0, 3))
    ax.scatter(points.geometry.x, points.geometry.y, s=spec["halo_area"], color=style["highlight"],
               alpha=spec["halo_alpha"], linewidth=0, zorder=2)
    ax.scatter(points.geometry.x, points.geometry.y, s=spec["point_area"], facecolor=style["background"],
               edgecolor=style["highlight"], linewidth=spec["point_edge_width"], zorder=4)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")
    ax.set_axis_off()

    handles = [Line2D([], [], color=style["route"], linewidth=spec["legend_line_width"] / 2),
               Line2D([], [], color=style["highlight"], linewidth=spec["legend_line_width"]),
               Line2D([], [], marker="o", linestyle="none", markersize=spec["legend_marker_size"],
                      markerfacecolor=style["background"], markeredgecolor=style["highlight"],
                      markeredgewidth=spec["point_edge_width"])]
    s11.bottom_legend(ax, handles, ["Sea lanes", "Through a chokepoint", f"Chokepoint ({len(points)})"],
                      fontsize=font["legend"], columnspacing=spec["legend_column_spacing"],
                      handletextpad=spec["legend_handle_pad"], borderpad=spec["legend_border_pad"])

    named = points[points["chokepoint_id"].isin(spec["labels"])]
    labels = named_labels(ax, named.geometry.x, named.geometry.y, short_names(named, poster),
                          [spec["labels"][cid] for cid in named["chokepoint_id"]], style["note"], settings,
                          fontweight="bold", color=style["text"])
    name = "fig_study_area.png"
    check_labels(fig, ax, labels, point_table(points), np.sqrt(spec["point_area"]) / 2, name, settings, log)
    save_poster(fig, name, settings, log)


# ---------------------------------------------------------------------

def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    crs = settings["outputs"]["static_map_crs"]
    s11.apply_style(settings["outputs"]["style"])

    chokepoints, routes, subflows = s11.read_inputs(settings, log)
    routes = s11.route_values(routes, subflows, log)
    basemap = s11.read_basemap(settings, log)

    poster_composite(chokepoints.to_crs(crs), routes.to_crs(crs), basemap.to_crs(crs),
                     s11.world_outline(crs), settings, log)
    poster_trade_conflict(chokepoints, settings, log)
    poster_breakdown(chokepoints, settings, log)
    poster_impact(chokepoints, settings, log)
    poster_study_area(basemap, settings, log)
    log.info("Run of s11b_poster_figures finished")


if __name__ == "__main__":
    main()

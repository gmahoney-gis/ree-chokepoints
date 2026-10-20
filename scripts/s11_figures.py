"""S11: draft figures (tasks 6.2, 6.3, 6.5; first pass for Lab 7).

Reads the project database and the S06/S09 tables, then draws:

1. Task 6.2: component maps -> fig_6_2_components.png. Two stacked world maps of the
   included chokepoints: (a) trade exposure (trade_value_base) and (b) violence
   exposure (viol_n_250), circle area proportional to the value. Zero-value
   chokepoints are small hollow markers; the top outputs.top_n_labels are labelled.
2. Task 6.3: scenario route maps -> fig_6_3_routes.png. One map per routing scenario,
   line width proportional to the summed maritime_value_usd of the sub-flows using
   each route. Routes are the antimeridian-split geometries written by S06.
3. Task 6.5: trade vs. conflict chart -> fig_6_5_trade_conflict.png. Scatter on
   log(1 + x) axes with median lines (the quadrant split) and the main Spearman
   result from outputs/tables/correlation.csv.
4. Context: Red Sea transits -> fig_red_sea_context.png. Container transits per year
   from outputs/tables/red_sea_context.csv (written by S06).

World maps use outputs.static_map_crs (Equal Earth) over the Natural Earth countries.
Maps show chokepoint values only, never individual events. Each figure's draft caption
is written to the log. Tasks 6.1, 6.4 and 6.6-6.9 are not part of the first pass.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")   # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import Polygon

import common

DPI = 300
# Okabe-Ito colours (colourblind-safe) and neutral greys for the base map.
TRADE_COLOR = "#0072B2"      # blue
VIOLENCE_COLOR = "#D55E00"   # vermilion
ROUTE_COLOR = "#0072B2"
POINT_COLOR = "#222222"
LAND_COLOR = "#E6E6E6"
BORDER_COLOR = "#FFFFFF"
OCEAN_COLOR = "#F7FBFF"
TEXT_COLOR = "#222222"
MUTED_TEXT = "#555555"
YEAR_COLORS = ["#9ECAE1", "#4292C6", "#08519C"]   # one hue, light (earliest) to dark

MAX_MARKER_AREA = 700        # points^2 for the largest circle
MIN_MARKER_AREA = 8          # smallest non-zero circle, so it stays visible
ZERO_MARKER_AREA = 14        # hollow marker for zero values
MAX_ROUTE_WIDTH = 6.0        # points for the route with the most trade
MIN_ROUTE_WIDTH = 0.15


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


def draw_basemap(ax, basemap, outline):
    """Ocean-filled map frame with grey land and no axes."""
    outline.plot(ax=ax, color=OCEAN_COLOR, edgecolor="#BBBBBB", linewidth=0.5, zorder=0)
    basemap.plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, zorder=1)
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


# Scatter labels: beside the point first, so each label sits level with its own point.
SCATTER_CANDIDATES = [((7, 0), "left", "center"), ((-7, 0), "right", "center")] + LABEL_CANDIDATES


def place_labels(ax, xs, ys, texts, avoid_xy=(), candidates=LABEL_CANDIDATES, **text_kw):
    """Place each label at the first candidate position that overlaps no earlier label.

    A simple greedy placer (no extra library): labels are drawn in the order given,
    and a label also avoids the points in `avoid_xy` (data coordinates). If every
    candidate overlaps, the first one is used.
    """
    renderer = ax.figure.canvas.get_renderer()
    taken = []
    avoid = [ax.transData.transform(p) for p in avoid_xy]
    for x, y, text in zip(xs, ys, texts):
        chosen = None
        for offset, ha, va in candidates:
            label = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points",
                                ha=ha, va=va, **text_kw)
            box = label.get_window_extent(renderer).expanded(1.05, 1.15)
            hits = (any(box.overlaps(t) for t in taken)
                    or any(box.contains(px, py) for px, py in avoid))
            if not hits:
                chosen = label
                break
            label.remove()
        if chosen is None:
            offset, ha, va = candidates[0]
            chosen = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points",
                                 ha=ha, va=va, **text_kw)
        taken.append(chosen.get_window_extent(renderer))


def label_top(ax, points, field, n):
    """Label the n chokepoints with the largest values of `field`."""
    top = points[points[field] > 0].nlargest(n, field)
    place_labels(ax, top.geometry.x, top.geometry.y, top["name"],
                 fontsize=6.5, color=TEXT_COLOR, zorder=5,
                 bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75})
    return list(top["name"])


def size_legend(ax, vmax, refs, fmt, color, title):
    """Legend of reference circle sizes plus the hollow zero marker."""
    handles = [plt.scatter([], [], s=marker_areas(np.array([r]), vmax)[0], color=color,
                           alpha=0.6, edgecolor="white", linewidth=0.5) for r in refs]
    labels = [fmt(r) for r in refs]
    handles.append(plt.scatter([], [], s=ZERO_MARKER_AREA, facecolor="none", edgecolor=MUTED_TEXT,
                               linewidth=0.7))
    labels.append("0")
    ax.legend(handles, labels, title=title, loc="lower left", fontsize=6, title_fontsize=6.5,
              frameon=True, framealpha=0.9, edgecolor="none", labelspacing=2.2, borderpad=1.2,
              handletextpad=2.0)


# ---------------------------------------------------------------------
# Task 6.2: component maps
# ---------------------------------------------------------------------

def component_panel(ax, points, field, color, title, legend_refs, fmt, legend_title, n_labels):
    """One component map: proportional circles, hollow zeros, top-n labels."""
    zero = points[points[field] == 0]
    nonzero = points[points[field] > 0].sort_values(field, ascending=False)  # small on top
    vmax = points[field].max()
    ax.scatter(nonzero.geometry.x, nonzero.geometry.y, s=marker_areas(nonzero[field], vmax),
               color=color, alpha=0.6, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(zero.geometry.x, zero.geometry.y, s=ZERO_MARKER_AREA, facecolor="none",
               edgecolor=MUTED_TEXT, linewidth=0.7, zorder=3)
    labelled = label_top(ax, points, field, n_labels)
    size_legend(ax, vmax, legend_refs, fmt, color, legend_title)
    ax.set_title(title, loc="left", fontsize=9, color=TEXT_COLOR)
    return labelled, len(zero)


def fig_components(points, basemap, outline, settings, out, log):
    """Task 6.2: trade (a) and violence (b) exposure maps, stacked."""
    n = settings["outputs"]["top_n_labels"]
    ring = settings["conflict"]["ring_main_km"]
    trade_field = settings["scoring"]["correlation_main"]["trade"]
    viol_field = settings["scoring"]["correlation_main"]["conflict"]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.2))
    for ax in axes:
        draw_basemap(ax, basemap, outline)
    top_trade, zero_trade = component_panel(
        axes[0], points, trade_field, TRADE_COLOR, "(a) Trade exposure: maritime REE trade crossing each chokepoint",
        [1e7, 1e8, 5e8], lambda v: f"${v / 1e6:,.0f} M", "US$ (2025)", n)
    top_viol, zero_viol = component_panel(
        axes[1], points, viol_field, VIOLENCE_COLOR,
        f"(b) Violence exposure: violent events within {ring} km",
        [100, 1000, 5000], lambda v: f"{v:,.0f}", "Events", n)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    w = settings["events"]
    caption = (f"Figure 6.2 (draft). Component exposure at the {len(points)} included maritime "
               f"chokepoints (Equal Earth). (a) Value of 2025 maritime rare earth trade "
               f"(HS {', '.join(settings['trade']['hs_codes_main'])}) whose baseline route crosses "
               f"each chokepoint zone; circle area is proportional to value. (b) ACLED violent "
               f"events within {ring} km, {w['window_start']} to {w['window_end']}; circle area is "
               f"proportional to the count. Hollow markers show zero values ({zero_trade} for "
               f"trade, {zero_viol} for violence). The top {n} on each panel are labelled. "
               f"Sources: UN Comtrade, ACLED, Natural Earth.")
    log.info("Task 6.2 | labelled trade: %s", ", ".join(top_trade))
    log.info("Task 6.2 | labelled violence: %s", ", ".join(top_viol))
    return caption


# ---------------------------------------------------------------------
# Task 6.3: scenario route maps
# ---------------------------------------------------------------------

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


def fig_routes(routes, subflows, points, basemap, outline, settings, crs, out, log):
    """Task 6.3: baseline and Red Sea diversion routes, one map each, same width scale."""
    routes = route_values(routes, subflows, log).to_crs(crs)
    scenarios = list(settings["routing"]["scenarios"])
    vmax = routes["route_value_usd"].max()

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(7.5, 4.1 * len(scenarios)))
    for ax, scenario in zip(np.atleast_1d(axes), scenarios):
        draw_basemap(ax, basemap, outline)
        group = routes[routes["scenario"] == scenario].sort_values("route_value_usd")  # thick on top
        group.plot(ax=ax, color=ROUTE_COLOR, linewidth=route_widths(group["route_value_usd"], vmax),
                   alpha=0.55, zorder=2)
        ax.scatter(points.geometry.x, points.geometry.y, s=10, color=POINT_COLOR,
                   edgecolor="white", linewidth=0.4, zorder=4)
        closed = settings["routing"]["scenarios"][scenario]["closed"]
        subtitle = "all chokepoints open" if not closed else "closed: " + ", ".join(closed)
        ax.set_title(f"{scenario.replace('_', ' ').capitalize()} ({subtitle})", loc="left",
                     fontsize=9, color=TEXT_COLOR)

    refs = [1e7, 1e8, vmax]
    handles = [Line2D([], [], color=ROUTE_COLOR, alpha=0.55, linewidth=route_widths(np.array([r]), vmax)[0])
               for r in refs]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=3.5, color=POINT_COLOR))
    labels = [f"${r / 1e6:,.0f} M" for r in refs] + ["Chokepoint"]
    np.atleast_1d(axes)[0].legend(handles, labels, title="Route trade, US$ (2025)", loc="lower left",
                                  fontsize=6, title_fontsize=6.5, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    diversion = settings["routing"]["scenarios"].get("red_sea_diversion", {})
    caption = (f"Figure 6.3 (draft). Modelled sea routes of 2025 maritime rare earth trade "
               f"between principal ports (first pass: nearest World Port Index ports) under the "
               f"baseline and the Red Sea diversion ({', '.join(diversion.get('closed', []))} closed; "
               f"displaced flows rerouted round the Cape of Good Hope). Line width is proportional "
               f"to the summed value of the sub-flows using each route, on the same scale in both "
               f"panels; dots are the {len(points)} included chokepoints. Routes from searoute, "
               f"split at the antimeridian. Equal Earth projection. Sources: UN Comtrade, NGA "
               f"World Port Index, Natural Earth.")
    return caption


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


def fig_trade_conflict(points, settings, out, log):
    """Task 6.5: trade (x) vs. violence (y) on log(1 + x) axes, split at the medians."""
    xf = settings["scoring"]["correlation_main"]["trade"]
    yf = settings["scoring"]["correlation_main"]["conflict"]
    main = read_correlation(settings, log)
    x, y = np.log1p(points[xf]), np.log1p(points[yf])
    x_med, y_med = points[xf].median(), points[yf].median()

    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    ax.axvline(np.log1p(x_med), color=MUTED_TEXT, linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(np.log1p(y_med), color=MUTED_TEXT, linestyle="--", linewidth=0.8, zorder=1)
    ax.scatter(x, y, s=36, color=TRADE_COLOR, edgecolor="white", linewidth=0.8, zorder=3)

    # Every chokepoint labelled; coincident points share one stacked label.
    # Axis limits are fixed first so label positions do not shift after placement.
    groups = label_groups(points, xf, yf)
    ax.margins(x=0.08, y=0.08)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    place_labels(ax, np.log1p(groups[xf]), np.log1p(groups[yf]),
                 ["\n".join(names) for names in groups["name"]], avoid_xy=list(zip(x, y)),
                 candidates=SCATTER_CANDIDATES,
                 arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.4, "shrinkA": 0, "shrinkB": 3},
                 fontsize=6, color=TEXT_COLOR, linespacing=1.1, zorder=4,
                 bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.85})

    # Axis ticks at round values of the original units.
    xticks = [0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    ax.set_xticks(np.log1p(xticks), ["0", "$10 K", "$100 K", "$1 M", "$10 M", "$100 M", "$1 B"])
    yticks = [0, 1, 10, 100, 1000, 10000]
    ax.set_yticks(np.log1p(yticks), [f"{t:,}" for t in yticks])
    ax.set_xlabel(f"Trade exposure, US$ 2025 ({xf}, log(1 + x) scale)", fontsize=8)
    ring = settings["conflict"]["ring_main_km"]
    ax.set_ylabel(f"Violent events within {ring} km ({yf}, log(1 + x) scale)", fontsize=8)
    ax.tick_params(labelsize=7, colors=MUTED_TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BBBBBB")
    ax.text(0.02, 0.98, f"Spearman ρ = {main['rho']:.2f}\np = {main['p_value']:.2f}, n = {int(main['n'])}",
            transform=ax.transAxes, va="top", fontsize=8, color=TEXT_COLOR,
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#BBBBBB", "lw": 0.6})
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    # The log goes to a cp1252 console on Windows, so the caption spells out "rho".
    verdict = "no significant association" if main["p_value"] >= 0.05 else "significant at p < 0.05"
    caption = (f"Figure 6.5 (draft). Trade exposure against violence exposure for the "
               f"{len(points)} included chokepoints, both on log(1 + x) axes. Dashed lines mark "
               f"the medians (trade US$ {x_med:,.0f}; violence {y_med:,.0f} events) and divide "
               f"the four quadrants. The violence median is only {y_med:,.0f} events within "
               f"{ring} km, so the \"high violence\" side includes chokepoints with very few "
               f"events. {int(main['zero_trade'])} chokepoints have zero trade and "
               f"{int(main['zero_conflict'])} have zero violent events; coincident points share "
               f"one label. Spearman rho = {main['rho']:.2f} (p = {main['p_value']:.2f}): "
               f"{verdict}. Sources: UN Comtrade, ACLED.")
    return caption


# ---------------------------------------------------------------------
# Context: Red Sea transits
# ---------------------------------------------------------------------

def fig_red_sea(settings, out, log):
    """Grouped bars: container transits per year at the Red Sea and Cape chokepoints."""
    path = common.get_path(settings, "tables") / "red_sea_context.csv"
    table = pd.read_csv(path)
    context = settings["routing"]["red_sea_context"]
    names, years = context["chokepoints"], context["years"]
    log.info("Context | read %s (%d rows)", path, len(table))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / len(years)
    xs = np.arange(len(names))
    for i, (year, color) in enumerate(zip(years, YEAR_COLORS)):
        values = [table.loc[(table["name"] == n) & (table["year"] == year), "transits_container"].sum()
                  for n in names]
        bars = ax.bar(xs + (i - (len(years) - 1) / 2) * width, values, width * 0.92, color=color,
                      label=str(year), zorder=2)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in values], fontsize=6.5, color=TEXT_COLOR,
                     padding=2)
        log.info("Context | %d container transits: %s", year,
                 ", ".join(f"{n} {v:,.0f}" for n, v in zip(names, values)))
    ax.set_xticks(xs, names, fontsize=8)
    ax.set_ylabel("Container ship transits per year", fontsize=8)
    ax.tick_params(axis="y", labelsize=7, colors=MUTED_TEXT)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(years))
    ax.text(1.0, -0.12, "Source: IMF PortWatch", transform=ax.transAxes, ha="right",
            fontsize=7, color=MUTED_TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    first, last = (table[table["year"] == yr].set_index("name")["transits_container"] for yr in (years[0], years[-1]))
    change = "; ".join(f"{n} {100 * (last[n] / first[n] - 1):+.0f}%" for n in names)
    caption = (f"Figure (context, draft). Container ship transits per calendar year through the "
               f"{', '.join(names[:-1])} and past the {names[-1]}, {years[0]}-{years[-1]} (2024 "
               f"has 366 days). From {years[0]} to {years[-1]}, {change}. This supports the Red Sea "
               f"diversion scenario as the realistic one for 2025. Source: IMF PortWatch daily "
               f"chokepoint transits.")
    return caption


# ---------------------------------------------------------------------

def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    figures = common.get_path(settings, "figures")
    figures.mkdir(parents=True, exist_ok=True)
    crs = settings["outputs"]["static_map_crs"]

    chokepoints, routes, subflows = read_inputs(settings, log)
    basemap = read_basemap(settings, log).to_crs(crs)
    outline = world_outline(crs)
    points = chokepoints.to_crs(crs)
    log.info("Static maps projected to %s", crs)

    captions = {
        "fig_6_2_components.png": fig_components(points, basemap, outline, settings,
                                                 figures / "fig_6_2_components.png", log),
        "fig_6_3_routes.png": fig_routes(routes, subflows, points, basemap, outline, settings, crs,
                                         figures / "fig_6_3_routes.png", log),
        "fig_6_5_trade_conflict.png": fig_trade_conflict(chokepoints, settings,
                                                         figures / "fig_6_5_trade_conflict.png", log),
        "fig_red_sea_context.png": fig_red_sea(settings, figures / "fig_red_sea_context.png", log),
    }
    for name, caption in captions.items():
        log.info("Wrote %s (%d dpi)", figures / name, DPI)
        log.info("Caption %s: %s", name, caption)
    log.info("Run of s11_figures finished")


if __name__ == "__main__":
    main()

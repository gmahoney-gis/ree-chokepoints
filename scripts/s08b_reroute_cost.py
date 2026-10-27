"""S08b: reroute time and cost of each chokepoint closure (tasks 4.16-4.17).

An impact layer reported beside the score, never part of it (Decision 8). Every
parameter is read from reroute_impact in the settings file (task 4.15).

1. Task 4.16: each row of S08's detours table gets its extra days at sea,
   extra nm / (speed x 24).
2. Task 4.17: extra cost per voyage of one typical container ship,
   extra days x (charter per day + fuel t/day x fuel price) + change in canal tolls.
   A toll is added when the detour crosses a canal the baseline route did not, and
   subtracted when the baseline crossed it and the detour does not. Crossing uses the
   same test as S06 and S08: the line intersects the canal's shape from
   common.crossing_geometries, copied 360 degrees east and west. REE holding cost is
   trade value x interest rate x extra days / 365, summed per chokepoint (a total for
   2025 trade, not a per-voyage figure).
3. Decision 13: per chokepoint, the headline extra days and cost per voyage are
   value-weighted means over the rerouted sub-flows (reroute_impact.typical_voyage, the
   same weighting as Equation 4), with the median beside them. Sub-flows with no sea
   route, and sub-flows at a port inside the zone, get no cost (null) and are reported
   as counts and value.
4. Ranges (Decision 13): one parameter is changed at a time to its low and high value
   (speed and fuel use always together), and the smallest and largest result is kept.
5. Task 4.17 Suez check against ITF/OECD (reroute_impact.suez_check): S08's Suez-only
   closure, the Cape-only detour from S06's red_sea_diversion routes (the main
   comparison, Decision 10b), and one reference voyage routed via Suez and via the Cape;
   each with this project's parameters and with ITF's own charter rate and Suez toll.
6. Outputs: S08b owns the chokepoint_scores columns starting with "impact_"; the
   per-row costs go to the reroute_costs table (not published); outputs/tables gets
   reroute_cost.csv, reroute_cost_ranges.csv and suez_check.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import LineString

import common

S08B_PREFIX = "impact_"    # S08b owns every chokepoint_scores column starting with this
SHIFTS = (-360.0, 0.0, 360.0)   # copies of a shape tested against lines near 180 degrees
SHIP_PARAMETERS = ["speed_knots", "fuel_use_t_per_day", "fuel_price_usd_per_t",
                   "charter_rate_usd_per_day", "interest_rate_annual"]


# ---------------------------------------------------------------------
# Parameters (task 4.15)
# ---------------------------------------------------------------------

def read_parameters(settings, log):
    """Main parameter values, as a flat dict; canal tolls are keyed "toll_<canal key>"."""
    ri = settings["reroute_impact"]
    if ri["typical_voyage"] not in ("value_weighted_mean", "median"):
        raise ValueError(f"reroute_impact.typical_voyage '{ri['typical_voyage']}' is not supported")
    if ri["range_method"] != "one_at_a_time":
        raise ValueError(f"reroute_impact.range_method '{ri['range_method']}' is not supported")
    params = {key: ri[key]["value"] for key in SHIP_PARAMETERS}
    for key in ri["canal_chokepoints"].values():
        params[f"toll_{key}"] = ri["canal_tolls_usd_per_transit"][key]["value"]
    log.info("Task 4.15 | reroute parameters (main value, low, high; source):")
    for key in SHIP_PARAMETERS:
        p = ri[key]
        log.info("  %-26s %12s %12s %12s  %s", key, p["value"], p.get("low"), p.get("high"), p["source"])
    for key in ri["canal_chokepoints"].values():
        p = ri["canal_tolls_usd_per_transit"][key]
        log.info("  %-26s %12s %12s %12s  %s", f"toll {key}", p["value"], p["low"], p["high"], p["source"])
    log.info("Task 4.16 | ship cost per day at sea: %s USD (charter %s + %s t/day x %s USD/t) | %s nm per day",
             f"{daily_cost(params):,.1f}", params["charter_rate_usd_per_day"], params["fuel_use_t_per_day"],
             params["fuel_price_usd_per_t"], f"{params['speed_knots'] * 24:,.0f}")
    log.info("Decision 13 | headline per chokepoint: %s (reroute_impact.typical_voyage); median beside it",
             ri["typical_voyage"])
    return params


def daily_cost(params):
    """Cost of one day at sea: charter plus fuel."""
    return params["charter_rate_usd_per_day"] + params["fuel_use_t_per_day"] * params["fuel_price_usd_per_t"]


def range_cases(settings, params):
    """Decision 13: [(case, parameter, level, params)], one parameter changed at a time.

    Parameters with paired_with change together with their partner (speed and fuel use).
    """
    ri = settings["reroute_impact"]
    groups, seen = [], set()
    for key in SHIP_PARAMETERS:
        if key in seen:
            continue
        group = [key] + [k for k in SHIP_PARAMETERS if ri[k].get("paired_with") == key]
        seen.update(group)
        groups.append((" + ".join(group), {k: ri[k] for k in group}))
    for key in ri["canal_chokepoints"].values():
        groups.append((f"toll_{key}", {f"toll_{key}": ri["canal_tolls_usd_per_transit"][key]}))

    cases = [("main", "none", "main", dict(params))]
    for name, members in groups:
        for level in ("low", "high"):
            changed = dict(params)
            changed.update({k: p[level] for k, p in members.items()})
            cases.append((f"{name} {level}", name, level, changed))
    return cases


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_inputs(settings, log):
    """S08's detours and detour_routes, S06's routes and crossings, sub-flow values."""
    detours = pd.DataFrame(common.read_layer(settings, "detours"))
    subflows = pd.DataFrame(common.read_layer(settings, "subflows"))
    subflows = subflows[~subflows["same_port"].astype(bool)]
    detours = detours.merge(subflows[["subflow_id", "maritime_value_usd"]], on="subflow_id", how="left")
    if detours["maritime_value_usd"].isna().any():
        raise ValueError("detours rows without a routed sub-flow; rerun S08")
    detour_routes = common.read_layer(settings, "detour_routes")
    routes = common.read_layer(settings, "routes")
    crossings = common.read_layer(settings, "route_crossings")
    log.info("Inputs | detours rows %s (sub-flows %s, chokepoints %d) | detour_routes %s | routes %s "
             "(%s) | route_crossings %s", f"{len(detours):,}", f"{detours['subflow_id'].nunique():,}",
             detours["chokepoint_id"].nunique(), f"{len(detour_routes):,}", f"{len(routes):,}",
             ", ".join(f"{s} {n}" for s, n in routes["scenario"].value_counts().items()), f"{len(crossings):,}")
    return detours, subflows, detour_routes, routes, crossings


def read_shapes(settings, log):
    """Crossing shapes of the included chokepoints, and the shape of each tolled canal."""
    zones = common.read_layer(settings, "zones")
    zones = zones[zones["included"].str.strip().str.lower() == "yes"]
    shapes = common.crossing_geometries(zones[["chokepoint_id", "name", "geometry"]].reset_index(drop=True),
                                        settings)
    canals = {}
    for name, key in settings["reroute_impact"]["canal_chokepoints"].items():
        row = shapes[shapes["name"] == name]
        if len(row) != 1:
            raise ValueError(f"reroute_impact.canal_chokepoints: '{name}' is not an included chokepoint")
        canals[key] = (row["chokepoint_id"].iloc[0], row["geometry"].iloc[0])
        log.info("Task 4.17 | canal %s: %s (%s), tested with its %s", key, name, row["chokepoint_id"].iloc[0],
                 row["crossing_method"].iloc[0])
    return shapes, canals


# ---------------------------------------------------------------------
# Canal crossings
# ---------------------------------------------------------------------

def crosses(line, shape):
    """True when a line touches the shape or its copy 360 degrees east or west (as in S06)."""
    return any(line.intersects(translate(shape, xoff=dx)) for dx in SHIFTS)


def canal_flags(lines, keys, canals):
    """One row per line: the key columns and a True/False column per canal key."""
    flags = lines[keys].copy()
    for key, (_, shape) in canals.items():
        flags[key] = [crosses(g, shape) for g in lines.geometry]
    return flags


def check_against_crossings(base_flags, subflows, crossings, canals, log):
    """The baseline canal flags must agree with S06's route_crossings at the flow level."""
    on_pair = subflows[["flow_id", "origin_port", "dest_port"]].merge(base_flags, on=["origin_port", "dest_port"])
    base = crossings[crossings["scenario"] == "baseline"]
    for key, (cid, _) in canals.items():
        by_geometry = set(on_pair.loc[on_pair[key], "flow_id"])
        by_s06 = set(base.loc[base["chokepoint_id"] == cid, "flow_id"])
        log.info("Check | baseline flows crossing %s: by geometry %d | in route_crossings %d | %s", key,
                 len(by_geometry), len(by_s06), "agree" if by_geometry == by_s06 else "DISAGREE")
        if by_geometry != by_s06:
            raise ValueError(f"{key}: canal test and route_crossings differ: only geometry "
                             f"{sorted(by_geometry - by_s06)[:5]}, only S06 {sorted(by_s06 - by_geometry)[:5]}")


def log_both_canals(base_flags, detours, canals, log):
    """List the baseline port pairs that cross every tolled canal (their detours skip several tolls)."""
    keys = list(canals)
    both = base_flags[base_flags[keys].all(axis=1)]
    log.info("Check | baseline port pairs crossing %s: %d", " and ".join(keys), len(both))
    for r in both.itertuples():
        rows = detours[(detours["origin_port"] == r.origin_port) & (detours["dest_port"] == r.dest_port)]
        log.info("  %s -> %s: %d sub-flows, %s USD", r.origin_port, r.dest_port, rows["subflow_id"].nunique(),
                 f"{rows.drop_duplicates('subflow_id')['maritime_value_usd'].sum():,.0f}")


# ---------------------------------------------------------------------
# Tasks 4.16-4.17: days and costs per detours row
# ---------------------------------------------------------------------

def toll_units(detours, base_flags, detour_flags, canals):
    """Per detours row and canal: +1 the detour adds the canal, -1 it skips it, 0 no change.

    Null for rows without a detour route (no sea route, or port inside the zone).
    """
    rows = detours.merge(base_flags, on=["origin_port", "dest_port"], how="left")
    rows = rows.merge(detour_flags, on=["origin_port", "dest_port", "chokepoint_id"], how="left",
                      suffixes=("_before", "_after"))
    costed = ~detours["no_route"].astype(bool) & ~detours["port_in_zone"].astype(bool)
    if rows.loc[costed, [f"{k}_after" for k in canals]].isna().any().any():
        raise ValueError("rerouted detours rows without a detour_routes line; rerun S08")
    for key in canals:
        rows[f"{key}_before"] = rows[f"{key}_before"].astype(bool)
        after = rows[f"{key}_after"].where(costed)
        rows[f"{key}_after"] = after
        rows[f"{key}_change"] = after.astype(float) - rows[f"{key}_before"].astype(float)
    rows["costed"] = costed.values
    return rows


def costs(rows, params, canals):
    """Tasks 4.16-4.17: extra days, toll change, voyage cost and holding cost per row."""
    days = rows["extra_nm"].where(rows["costed"]) / (params["speed_knots"] * 24)
    toll = sum(rows[f"{k}_change"] * params[f"toll_{k}"] for k in canals)
    voyage = days * daily_cost(params) + toll
    holding = rows["maritime_value_usd"] * params["interest_rate_annual"] * days / 365
    return pd.DataFrame({"extra_days": days, "toll_change_usd": toll, "voyage_cost_usd": voyage,
                         "holding_cost_usd": holding}, index=rows.index)


def cost_table(rows, main, canals):
    """The reroute_costs table: one row per detours row (S08), with days and costs."""
    keep = ["subflow_id", "flow_id", "chokepoint_id", "origin_port", "dest_port", "extra_nm"]
    flags = [f"{k}_{when}" for k in canals for when in ("before", "after")]
    table = pd.concat([rows[keep], main[["extra_days"]], rows[flags], main.drop(columns="extra_days")], axis=1)
    table.insert(len(table.columns) - 1, "value_usd", rows["maritime_value_usd"])
    table["no_route"] = rows["no_route"].astype(bool)
    table["port_in_zone"] = rows["port_in_zone"].astype(bool)
    table[[f"{k}_after" for k in canals]] = table[[f"{k}_after" for k in canals]].astype("boolean")
    table["extra_days"] = table["extra_days"].round(4)
    for c in ("toll_change_usd", "voyage_cost_usd", "holding_cost_usd"):
        table[c] = table[c].round(2)
    return table


def log_canal_changes(rows, shapes, canals, log):
    """Per chokepoint: sub-flows whose detour adds, skips or keeps each canal."""
    names = shapes.set_index("chokepoint_id")["name"]
    costed = rows[rows["costed"]]
    log.info("Task 4.17 | canal changes per closure (sub-flows; added = detour crosses a canal the baseline did not):")
    for cid, g in costed.groupby("chokepoint_id", sort=False):
        parts = []
        for key in canals:
            added, skipped = int((g[f"{key}_change"] > 0).sum()), int((g[f"{key}_change"] < 0).sum())
            if added or skipped:
                parts.append(f"{key} added {added}, skipped {skipped}")
        if parts:
            log.info("  %-22s %s", names[cid], " | ".join(parts))


# ---------------------------------------------------------------------
# Decision 13: per chokepoint
# ---------------------------------------------------------------------

def weighted_mean(values, weights):
    """Value-weighted mean, or null when there is nothing to average."""
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() > 0 else np.nan


def headline(values, weights, how):
    """The per-chokepoint figure for a typical voyage (reroute_impact.typical_voyage)."""
    if how == "value_weighted_mean":
        return weighted_mean(values, weights)
    return float(values.median()) if len(values) else np.nan


def chokepoint_figures(rows, result, how):
    """Headline days, voyage cost and holding cost per chokepoint for one set of parameters."""
    d = pd.concat([rows[["chokepoint_id", "costed", "maritime_value_usd"]], result], axis=1)
    d = d[d["costed"]]
    out = {}
    for cid, g in d.groupby("chokepoint_id"):
        w = g["maritime_value_usd"]
        out[cid] = {"days": headline(g["extra_days"], w, how), "voyage_cost": headline(g["voyage_cost_usd"], w, how),
                    "holding_cost": float(g["holding_cost_usd"].sum())}
    return out


def impact_scores(shapes, rows, main, cases, canals, how, log):
    """Decision 13: the impact_ columns, and the long table of every range case."""
    figures = {case: chokepoint_figures(rows, costs(rows, p, canals), how) for case, _, _, p in cases}
    d = pd.concat([rows, main], axis=1)
    records, ranges = [], []
    for r in shapes.itertuples():
        g = d[d["chokepoint_id"] == r.chokepoint_id]
        c, nocost = g[g["costed"]], g[~g["costed"]]
        w = c["maritime_value_usd"]
        added = c[[f"{k}_change" for k in canals]].gt(0).any(axis=1)
        skipped = c[[f"{k}_change" for k in canals]].lt(0).any(axis=1)
        per_case = pd.DataFrame([figures[case].get(r.chokepoint_id, {"days": np.nan, "voyage_cost": np.nan,
                                                                     "holding_cost": np.nan})
                                 for case, _, _, _ in cases])
        if len(c):
            for (case, parameter, level, _), f in zip(cases, per_case.itertuples()):
                ranges.append({"chokepoint_id": r.chokepoint_id, "name": r.name, "case": case,
                               "parameter": parameter, "level": level, "extra_days": round(f.days, 4),
                               "voyage_cost_usd": round(f.voyage_cost, 2), "holding_cost_usd": round(f.holding_cost, 2)})
        records.append({
            "chokepoint_id": r.chokepoint_id,
            "impact_nflows_costed": c["flow_id"].nunique(),
            "impact_value_costed": w.sum(),
            "impact_nflows_nocost": nocost["flow_id"].nunique(),
            "impact_value_nocost": nocost["maritime_value_usd"].sum(),
            "impact_days": headline(c["extra_days"], w, how),
            "impact_days_median": float(c["extra_days"].median()) if len(c) else np.nan,
            "impact_days_max": float(c["extra_days"].max()) if len(c) else np.nan,
            "impact_days_low": per_case["days"].min() if len(c) else np.nan,
            "impact_days_high": per_case["days"].max() if len(c) else np.nan,
            "impact_toll_change": headline(c["toll_change_usd"], w, how),
            "impact_voyage_cost": headline(c["voyage_cost_usd"], w, how),
            "impact_voyage_cost_median": float(c["voyage_cost_usd"].median()) if len(c) else np.nan,
            "impact_voyage_cost_low": per_case["voyage_cost"].min() if len(c) else np.nan,
            "impact_voyage_cost_high": per_case["voyage_cost"].max() if len(c) else np.nan,
            "impact_holding_cost": float(c["holding_cost_usd"].sum()) if len(c) else np.nan,
            "impact_holding_cost_low": per_case["holding_cost"].min() if len(c) else np.nan,
            "impact_holding_cost_high": per_case["holding_cost"].max() if len(c) else np.nan,
            "impact_nflows_canal_added": c.loc[added, "flow_id"].nunique(),
            "impact_nflows_canal_skipped": c.loc[skipped, "flow_id"].nunique(),
        })
    scores = pd.DataFrame(records)
    for col in scores.columns:
        if col.startswith("impact_days"):
            scores[col] = scores[col].round(4)
        elif col.startswith(("impact_value", "impact_toll", "impact_voyage", "impact_holding")):
            scores[col] = scores[col].round(2)
    log.info("Decision 13 | range cases (one change at a time from the main values): %d | %s", len(cases) - 1,
             ", ".join(case for case, _, _, _ in cases[1:]))
    return scores, pd.DataFrame(ranges)


def log_scores(scores, shapes, rows, log):
    """One line per chokepoint, sorted by the headline voyage cost; then the no-cost rows."""
    t = shapes[["chokepoint_id", "name"]].merge(scores, on="chokepoint_id")
    t = t.sort_values("impact_voyage_cost", ascending=False, na_position="last")
    log.info("Tasks 4.16-4.17 | reroute time and cost by chokepoint (days; USD; flows):")
    log.info("  %-22s %5s %7s %7s %7s %13s %11s %11s %21s %10s %5s %5s %5s", "chokepoint", "flows", "days",
             "median", "max", "days range", "toll chg", "voyage", "voyage range", "holding", "add", "skip",
             "nocst")
    num = lambda v, f: "null" if pd.isna(v) else format(v, f)
    for r in t.itertuples():
        log.info("  %-22s %5d %7s %7s %7s %13s %11s %11s %21s %10s %5d %5d %5d", r.name, r.impact_nflows_costed,
                 num(r.impact_days, ".2f"), num(r.impact_days_median, ".2f"), num(r.impact_days_max, ".2f"),
                 f"{num(r.impact_days_low, '.2f')}-{num(r.impact_days_high, '.2f')}",
                 num(r.impact_toll_change, ",.0f"), num(r.impact_voyage_cost, ",.0f"),
                 f"{num(r.impact_voyage_cost_low, ',.0f')}-{num(r.impact_voyage_cost_high, ',.0f')}",
                 num(r.impact_holding_cost, ",.0f"), r.impact_nflows_canal_added, r.impact_nflows_canal_skipped,
                 r.impact_nflows_nocost)
    costed = rows[rows["costed"]]
    log.info("Tasks 4.16-4.17 | detours rows %s: costed %s (sub-flows) | no sea route %s (%s USD) | port inside "
             "the zone %s (%s USD); both left null (Decision 13)", f"{len(rows):,}", f"{len(costed):,}",
             f"{int(rows['no_route'].sum()):,}", f"{rows.loc[rows['no_route'], 'maritime_value_usd'].sum():,.0f}",
             f"{int(rows['port_in_zone'].sum()):,}",
             f"{rows.loc[rows['port_in_zone'], 'maritime_value_usd'].sum():,.0f}")
    log.info("Task 4.17 | REE holding cost, all closures summed: %s USD (range %s-%s)",
             f"{scores['impact_holding_cost'].sum():,.0f}", f"{scores['impact_holding_cost_low'].sum():,.0f}",
             f"{scores['impact_holding_cost_high'].sum():,.0f}")


def log_negative(rows, main, shapes, log):
    """Costed rows whose detour is cheaper than the baseline (the tolls it skips exceed its ship cost)."""
    d = pd.concat([rows[["chokepoint_id", "origin_port", "dest_port", "costed"]], main], axis=1)
    neg = d[d["costed"] & (d["voyage_cost_usd"] < 0)]
    names = shapes.set_index("chokepoint_id")["name"]
    log.info("Check | costed sub-flows whose detour costs less than the baseline (tolls skipped > ship cost): %d",
             len(neg))
    for (cid, o, dd), g in neg.groupby(["chokepoint_id", "origin_port", "dest_port"]):
        log.info("  %-22s %s -> %s: %d sub-flows, %.2f days, %s USD per voyage", names[cid], o, dd, len(g),
                 g["extra_days"].iloc[0], f"{g['voyage_cost_usd'].iloc[0]:,.0f}")


# ---------------------------------------------------------------------
# Task 4.17: Suez check
# ---------------------------------------------------------------------

def itf_parameters(params, settings):
    """The main parameters with ITF's own charter rate and Suez toll (fuel stays ours)."""
    check = settings["reroute_impact"]["suez_check"]
    suez_key = settings["reroute_impact"]["canal_chokepoints"]["Suez Canal"]
    return {**params, "charter_rate_usd_per_day": check["itf_charter_usd_per_day"],
            f"toll_{suez_key}": check["itf_suez_toll_usd"]}


def check_row(label, rows, params, itf, canals, settings):
    """One Suez check row: value-weighted and median days and cost, with ITF ratios."""
    check = settings["reroute_impact"]["suez_check"]
    ours, theirs = costs(rows, params, canals), costs(rows, itf, canals)
    w = rows["maritime_value_usd"]
    days = weighted_mean(ours["extra_days"], w)
    cost = weighted_mean(ours["voyage_cost_usd"], w)
    cost_itf = weighted_mean(theirs["voyage_cost_usd"], w)
    return {"comparison": label, "n_subflows": len(rows), "value_usd": round(w.sum(), 2),
            "extra_nm": round(weighted_mean(rows["extra_nm"], w), 1),
            "extra_days": round(days, 3), "extra_days_median": round(float(ours["extra_days"].median()), 3),
            "voyage_cost_usd": round(cost, 0),
            "voyage_cost_median_usd": round(float(ours["voyage_cost_usd"].median()), 0),
            "voyage_cost_itf_params_usd": round(cost_itf, 0),
            "itf_extra_days": check["extra_days"], "itf_cost_one_way_usd": check["extra_cost_usd_one_way"],
            "ratio_days": round(days / check["extra_days"], 3),
            "ratio_cost": round(cost / check["extra_cost_usd_one_way"], 3),
            "ratio_cost_itf_params": round(cost_itf / check["extra_cost_usd_one_way"], 3)}


def cape_only_rows(routes, base_flags, subflows, canals, settings, log):
    """Decision 10b: baseline pairs crossing Suez against S06's red_sea_diversion routes, per sub-flow."""
    suez_key = settings["reroute_impact"]["canal_chokepoints"]["Suez Canal"]
    base = routes[routes["scenario"] == "baseline"][["origin_port", "dest_port", "length_nm"]]
    diverted = routes[routes["scenario"] == "red_sea_diversion"]
    after = canal_flags(diverted, ["origin_port", "dest_port"], canals)
    after["new_length_nm"] = diverted["length_nm"].values
    pairs = base_flags[base_flags[suez_key]].merge(base, on=["origin_port", "dest_port"])
    rows = pairs.merge(after, on=["origin_port", "dest_port"], how="left", suffixes=("_before", "_after"))
    missing = rows[rows["new_length_nm"].isna()]
    log.info("Task 4.17 | Cape-only: baseline pairs crossing %s %d | with a red_sea_diversion route %d | "
             "without (no sea route in S06): %s", suez_key, len(rows), len(rows) - len(missing),
             ", ".join(f"{o} -> {d}" for o, d in zip(missing["origin_port"], missing["dest_port"])) or "none")
    rows = rows.dropna(subset=["new_length_nm"])
    for key in canals:
        rows[f"{key}_change"] = rows[f"{key}_after"].astype(float) - rows[f"{key}_before"].astype(float)
    rows["extra_nm"] = rows["new_length_nm"] - rows["length_nm"]
    log.info("Task 4.17 | Cape-only: diversion routes that still cross a tolled canal: %s",
             ", ".join(f"{k} {int(rows[f'{k}_after'].astype(bool).sum())}" for k in canals))
    rows = rows.merge(subflows[["subflow_id", "origin_port", "dest_port", "maritime_value_usd"]],
                      on=["origin_port", "dest_port"])
    rows["costed"] = True
    return rows.reset_index(drop=True)


def reference_rows(settings, canals, log):
    """One reference voyage (reroute_impact.suez_check.reference_voyage) via Suez and via the Cape.

    The Cape route closes what S06's red_sea_diversion closes for displaced routes
    (closed plus diverted_also_closed), on searoute's own network.
    """
    ref = settings["reroute_impact"]["suez_check"]["reference_voyage"]
    routing = settings["routing"]
    always = list(routing["always_closed"])
    scenario = routing["scenarios"]["red_sea_diversion"]
    names = scenario["closed"] + (scenario.get("diverted_also_closed") or [])
    cape_tags = always + [routing["searoute_passages"][n] for n in names]
    lines = {}
    for label, tags in (("via Suez", always), ("via the Cape", cape_tags)):
        coords, _ = common.route_line(tuple(ref["origin"]), tuple(ref["destination"]), tags)
        if coords is None:
            raise ValueError(f"Reference voyage {ref['name']} has no route {label}")
        lines[label] = coords
    flags = {label: {k: crosses(LineString(c), canals[k][1]) for k in canals} for label, c in lines.items()}
    lengths = {label: common.length_nm(c, settings) for label, c in lines.items()}
    log.info("Task 4.17 | reference voyage %s: via Suez %s nm (%s) | via the Cape %s nm (%s; closed %s)", ref["name"],
             f"{lengths['via Suez']:,.1f}", ", ".join(k for k, v in flags["via Suez"].items() if v) or "no canal",
             f"{lengths['via the Cape']:,.1f}", ", ".join(k for k, v in flags["via the Cape"].items() if v) or "no canal",
             cape_tags)
    row = {"extra_nm": lengths["via the Cape"] - lengths["via Suez"], "maritime_value_usd": 1.0, "costed": True}
    for k in canals:
        row[f"{k}_change"] = float(flags["via the Cape"][k]) - float(flags["via Suez"][k])
    return pd.DataFrame([row]), ref["name"]


def suez_check(rows, routes, base_flags, subflows, params, canals, settings, log):
    """Task 4.17: the three Suez comparisons with ITF/OECD."""
    suez_id = canals[settings["reroute_impact"]["canal_chokepoints"]["Suez Canal"]][0]
    itf = itf_parameters(params, settings)
    s08 = rows[(rows["chokepoint_id"] == suez_id) & rows["costed"]].reset_index(drop=True)
    cape = cape_only_rows(routes, base_flags, subflows, canals, settings, log)
    ref, ref_name = reference_rows(settings, canals, log)
    table = pd.DataFrame([
        check_row("Cape-only (S06 red_sea_diversion; main comparison)", cape, params, itf, canals, settings),
        check_row("S08 Suez-only closure (Panama open)", s08, params, itf, canals, settings),
        check_row(f"Reference voyage {ref_name}", ref, params, itf, canals, settings),
    ])
    table.loc[table.index[-1], ["n_subflows", "value_usd"]] = [0, np.nan]
    check = settings["reroute_impact"]["suez_check"]
    log.info("Task 4.17 | Suez check against %s: %s extra days and %s USD one way", check["source"].split(".")[0],
             check["extra_days"], f"{check['extra_cost_usd_one_way']:,}")
    log.info("  %-52s %6s %7s %12s %12s %8s %8s %10s", "comparison", "days", "median", "cost", "cost ITF p.",
             "r days", "r cost", "r ITF p.")
    for r in table.itertuples():
        log.info("  %-52s %6.2f %7.2f %12s %12s %8.3f %8.3f %10.3f", r.comparison, r.extra_days, r.extra_days_median,
                 f"{r.voyage_cost_usd:,.0f}", f"{r.voyage_cost_itf_params_usd:,.0f}", r.ratio_days, r.ratio_cost,
                 r.ratio_cost_itf_params)
    log.info("  (ITF parameters: charter %s USD/day and Suez toll %s USD in place of ours; fuel unchanged)",
             f"{check['itf_charter_usd_per_day']:,}", f"{check['itf_suez_toll_usd']:,}")
    return table


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

def merge_scores(s08b, settings, log):
    """Replace S08b's columns in chokepoint_scores and keep every other script's columns."""
    if common.layer_name(settings, "chokepoint_scores") not in common.list_layers(settings):
        raise ValueError("chokepoint_scores does not exist; run S06 first")
    old = pd.DataFrame(common.read_layer(settings, "chokepoint_scores"))
    old = old.drop(columns=[c for c in old.columns if c == "geometry" or c.startswith(S08B_PREFIX)])
    missing = sorted(set(old["chokepoint_id"]) ^ set(s08b["chokepoint_id"]))
    if missing:
        raise ValueError(f"chokepoint_scores and the included chokepoints differ: {missing}")
    log.info("chokepoint_scores exists (%d rows); writing %d S08b columns, keeping %d others",
             len(old), len(s08b.columns) - 1, len(old.columns) - 2)
    return old.merge(s08b, on="chokepoint_id", how="left")


def write_tables(scores, ranges, check, shapes, settings, log):
    """Publishable summaries in outputs/tables (no individual trade records)."""
    tables = common.get_path(settings, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    summary = shapes[["chokepoint_id", "name"]].merge(scores, on="chokepoint_id")
    for name, table in (("reroute_cost.csv", summary), ("reroute_cost_ranges.csv", ranges),
                        ("suez_check.csv", check)):
        table.to_csv(tables / name, index=False)
        log.info("Wrote %s (%d rows)", tables / name, len(table))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)

    params = read_parameters(settings, log)
    cases = range_cases(settings, params)
    detours, subflows, detour_routes, routes, crossings = read_inputs(settings, log)
    shapes, canals = read_shapes(settings, log)

    # Task 4.17: which canals each baseline route and each detour route crosses.
    base_flags = canal_flags(routes[routes["scenario"] == "baseline"], ["origin_port", "dest_port"], canals)
    detour_flags = canal_flags(detour_routes, ["origin_port", "dest_port", "chokepoint_id"], canals)
    check_against_crossings(base_flags, subflows, crossings, canals, log)
    log_both_canals(base_flags, detours, canals, log)

    # Tasks 4.16-4.17: days and costs per detours row at the main values.
    rows = toll_units(detours, base_flags, detour_flags, canals)
    main_costs = costs(rows, params, canals)
    log_canal_changes(rows, shapes, canals, log)
    log_negative(rows, main_costs, shapes, log)

    # Decision 13: per chokepoint, with ranges.
    how = settings["reroute_impact"]["typical_voyage"]
    scores, ranges = impact_scores(shapes, rows, main_costs, cases, canals, how, log)
    log_scores(scores, shapes, rows, log)

    # Task 4.17: Suez check.
    check = suez_check(rows, routes, base_flags, subflows, params, canals, settings, log)

    common.write_layer(cost_table(rows, main_costs, canals), settings, "reroute_costs", log)
    common.write_layer(merge_scores(scores, settings, log), settings, "chokepoint_scores", log)
    write_tables(scores, ranges, check, shapes, settings, log)
    log.info("Run of s08b_reroute_cost finished")


if __name__ == "__main__":
    main()

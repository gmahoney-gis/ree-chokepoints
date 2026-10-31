"""S10: sensitivity tests (tasks 5.5-5.7; Decision 6).

Reruns S09's composite score with one choice changed at a time from the main run, then
with every weight combination, and summarises how much each chokepoint's rank moves.

1. Task 5.5: one run per alternative in sensitivity.alternatives (proposal Table 4 plus
   Decisions 8, 9, 15 and 16). Each run is the main run with one change, made either in
   a copy of the settings (for example scoring.transform: none) or by swapping one input
   column in a copy of chokepoint_scores (for example tens_n_250 -> tens_s1_n_250).
   Most alternatives use columns S06-S08b already stored. Two need a quick recount from
   the events layer: imo_count_rule (Decision 9) and tension_top_place (Decision 16(e)).
   port_assignment: nearest_gateway takes trade and redundancy from the Lab 7/8 first
   pass (paths.first_pass_database). Alternatives in sensitivity.not_run are listed as
   untested with their reason.
2. Task 5.6: the full weight grid (sensitivity.weight_grid; 36 combinations).
3. Task 5.7: each run's agreement with the main ranking (Spearman rho, top-n overlap,
   biggest mover) -> outputs/tables/sensitivity.csv; every chokepoint's rank in every
   run -> outputs/tables/sensitivity_ranks.csv; each chokepoint's best, worst and
   median rank and top-5 share -> sens_ columns in chokepoint_scores.

The scoring itself is S09's score() unchanged. S10 owns the chokepoint_scores columns
starting with "sens_" and keeps every other column. Only counts go in the log, never
event rows, place names or coordinates.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common
import s07_conflict_exposure as s07
import s09_composite as s09

S10_PREFIX = "sens_"
# trade_measure alternative -> (main field stem, alternative field stem) in S06's columns
TRADE_MEASURE_STEMS = {"net_weight_kg": ("trade_value_", "trade_wt_")}
# S08 columns S09 builds Equation 4 from (Decision 16)
REDUND_COLUMNS = ["redund_penalty_routed_pct", "redund_noalt_pct", "redund_penalty_pct"]
SUMMARY_COLUMNS = ["group", "alternative", "setting", "value", "status", "rho", "p_value",
                   "top5_overlap", "top10_overlap", "n_changed", "max_move", "biggest_mover", "reason"]


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_scores(settings, log, database=None, what="chokepoint_scores"):
    """Read chokepoint_scores as a plain table, without S10's own columns."""
    scores = pd.DataFrame(common.read_layer(settings, "chokepoint_scores", database))
    scores = scores.drop(columns=[c for c in scores.columns if c == "geometry" or c.startswith(S10_PREFIX)])
    log.info("%s read: %d chokepoints, %d columns", what, len(scores), len(scores.columns))
    return scores


def main_trade_field(settings):
    """The trade field of the main routing scenario (scoring.scenarios)."""
    return settings["scoring"]["scenarios"][s09.main_scenario(settings)]["trade"]


def label(key, value):
    """A short run name, e.g. ring_main_km=100 or conflict_mix=violence0.75_tension0.25."""
    if isinstance(value, dict):
        text = "_".join(f"{k}{v:g}" for k, v in value.items())
    elif isinstance(value, list):
        text = "-".join(str(v) for v in value)
    else:
        text = str(value)
    return f"{key}={text}"


# ---------------------------------------------------------------------
# Task 5.5: recounts from the events layer
# ---------------------------------------------------------------------

def by_chokepoint(values, chokepoints, scores):
    """Put per-chokepoint values (in the chokepoints layer's order) in chokepoint_scores' order."""
    series = pd.Series(np.asarray(values), index=chokepoints["chokepoint_id"].to_numpy())
    missing = set(scores["chokepoint_id"]) - set(series.index)
    if missing:
        raise ValueError(f"Included chokepoints and chokepoint_scores differ: {sorted(missing)}")
    return scores["chokepoint_id"].map(series)


def recount_violence(events, chokepoints, scores, settings, rule, log):
    """Decision 9: violent events within conflict.ring_main_km under one IMO GISIS count_rule."""
    violence = common.countable_events(events[events["category"] == "violence"], settings, log, count_rule=rule)
    distances = common.distance_matrix_km(violence.geometry.x, violence.geometry.y,
                                          chokepoints.geometry.x, chokepoints.geometry.y, settings)
    counts = (distances <= settings["conflict"]["ring_main_km"]).sum(axis=0)
    return by_chokepoint(counts, chokepoints, scores)


def recount_imo(events, chokepoints, scores, settings, rules, log):
    """Task 5.5 (imo_count_rule): ring counts for each alternative rule.

    The main rule is recounted first and must match S07's stored field, which shows the
    recount uses the same events and distances as S07.
    """
    field = s09.conflict_fields(settings)["violence"]
    main_rule = settings["events"]["imo_gisis"]["count_rule"]
    check = recount_violence(events, chokepoints, scores, settings, main_rule, log)
    if not (check == scores[field]).all():
        raise ValueError(f"Recount with count_rule {main_rule} differs from {field}; rerun S07")
    log.info("Task 5.5 | imo_count_rule: recount with the main rule (%s) matches %s for all %d chokepoints",
             main_rule, field, len(scores))
    out = {}
    for rule in rules:
        out[rule] = recount_violence(events, chokepoints, scores, settings, rule, log)
        changed = out[rule] != scores[field]
        log.info("Task 5.5 | imo_count_rule %s: %s changes at %d chokepoints: %s", rule, field,
                 int(changed.sum()), ", ".join(f"{n} {a}->{b}" for n, a, b in zip(
                     scores.loc[changed, "name"], scores.loc[changed, field], out[rule][changed])) or "none")
    return out


def recount_top_place(events, chokepoints, scores, settings, log):
    """Decision 16(e) (tension_top_place: removed): each chokepoint's main-ring tension
    with its single largest place left out.

    A place is one exact coordinate pair (as S07's log); events are weighted by
    event_count. The full count must match S07's stored field. Only each top place's
    country and share are logged, never its name or coordinates.
    """
    field = s09.conflict_fields(settings)["tension"]
    ring = settings["conflict"]["ring_main_km"]
    tension = common.countable_tension(events, settings, log)
    distances = common.distance_matrix_km(tension.geometry.x, tension.geometry.y,
                                          chokepoints.geometry.x, chokepoints.geometry.y, settings)
    places = pd.DataFrame({"x": tension.geometry.x, "y": tension.geometry.y,
                           "country": tension["country_iso3"].fillna("none"),
                           "w": tension["event_count"].to_numpy(dtype=float)})
    totals, removed, notes = [], [], []
    for j in range(len(chokepoints)):
        by_place = places[distances[:, j] <= ring].groupby(["x", "y", "country"])["w"].sum()
        total = by_place.sum()
        top = by_place.max() if len(by_place) else 0.0
        totals.append(total)
        removed.append(total - top)
        notes.append((chokepoints.at[j, "name"], total, by_place.idxmax()[2] if len(by_place) else "-",
                      100 * top / total if total else 0.0))
    check = by_chokepoint(totals, chokepoints, scores)
    if not np.allclose(check, scores[field]):
        raise ValueError(f"Recount of tension within {ring} km differs from {field}; rerun S07")
    log.info("Task 5.5 | tension_top_place: recount within %d km matches %s for all %d chokepoints; "
             "top place left out (country, share of the ring):", ring, field, len(scores))
    for name, total, country, share in sorted(notes, key=lambda n: -n[3]):
        log.info("Task 5.5 |   %-20s %9s events, top place %s %5.1f%% -> %9s", name, f"{int(total):,}",
                 country, share, f"{int(total * (1 - share / 100)):,}")
    return by_chokepoint(removed, chokepoints, scores).round().astype("int64")


# ---------------------------------------------------------------------
# Task 5.5: one change per alternative
# ---------------------------------------------------------------------

def with_setting(section, key):
    """An alternative that changes one entry in a copy of the settings."""
    def apply(scores, settings, value, inputs, log):
        changed = copy.deepcopy(settings)
        changed[section][key] = value
        return scores, changed
    return apply


def swap_column(scores, target, values):
    """A copy of chokepoint_scores with one input column replaced."""
    changed = scores.copy()
    changed[target] = values
    return changed


def stored_tension(field_for):
    """An alternative that swaps the main tension field for another stored S07 field."""
    def apply(scores, settings, value, inputs, log):
        ring = settings["conflict"]["ring_main_km"]
        stored = field_for(value, ring)
        if stored not in scores:
            raise ValueError(f"{stored} is not in chokepoint_scores; rerun S07")
        return swap_column(scores, s09.conflict_fields(settings)["tension"], scores[stored]), settings
    return apply


def imo_count_rule(scores, settings, value, inputs, log):
    """Decision 9: every IMO GISIS incident counted (recount)."""
    return swap_column(scores, s09.conflict_fields(settings)["violence"], inputs["imo"][value]), settings


def tension_top_place(scores, settings, value, inputs, log):
    """Decision 16(e): each chokepoint's largest single GDELT place left out (recount)."""
    if value != "removed":
        raise ValueError(f"tension_top_place '{value}' is not supported (only 'removed')")
    return swap_column(scores, s09.conflict_fields(settings)["tension"], inputs["top_place"]), settings


def trade_measure(scores, settings, value, inputs, log):
    """Table 4: trade measured by weight (kg) instead of value, from S06's stored field.

    Redundancy stays value-weighted (S08). Flows without a reported weight count as 0.
    """
    main, alternative = TRADE_MEASURE_STEMS[value]
    changed = copy.deepcopy(settings)
    scenario = changed["scoring"]["scenarios"][s09.main_scenario(settings)]
    field = scenario["trade"].replace(main, alternative)
    if field not in scores:
        raise ValueError(f"{field} is not in chokepoint_scores; rerun S06")
    scenario["trade"] = field
    log.info("Task 5.5 | trade_measure %s: trade field %s; redundancy stays value-weighted; "
             "flows without a reported weight count as 0 (see the S02 log)", value, field)
    return scores, changed


def redundancy_measure(scores, settings, value, inputs, log):
    """Decision 8: redundancy as S08b's extra cost per voyage instead of Equation 4's %.

    impact_voyage_cost (the value-weighted mean over rerouted sub-flows, including
    sub-flows whose detour is cheaper) replaces redund_penalty_routed_pct in a copy, so
    S09's Decision 16 rule applies unchanged: no-route trade gets the largest chokepoint
    cost (scoring.no_route_cap), a chokepoint with no affected flows gets 0. A negative
    chokepoint mean would be set to 0 and logged. The copied column holds USD, not %.
    """
    if value != "extra_cost_per_voyage":
        raise ValueError(f"redundancy_measure '{value}' is not supported")
    if settings["scoring"]["no_route_cap"] != "max_routed_chokepoint":
        raise ValueError("The cost-based redundancy needs scoring.no_route_cap: max_routed_chokepoint")
    cost = scores["impact_voyage_cost"].astype(float)
    mismatch = cost.isna() != scores["redund_penalty_routed_pct"].isna()
    if mismatch.any():
        raise ValueError(f"impact_voyage_cost and redund_penalty_routed_pct are null at different "
                         f"chokepoints: {list(scores.loc[mismatch, 'name'])}; rerun S08b")
    negative = cost < 0
    log.info("Task 5.5 | redundancy_measure %s: chokepoints with a negative mean cost set to 0: %s",
             value, ", ".join(scores.loc[negative, "name"]) or "none")
    cost = cost.clip(lower=0)
    capped = scores["redund_noalt_pct"] > 0
    log.info("Task 5.5 | redundancy_measure %s: no-route trade gets the largest chokepoint cost, "
             "$%s (%s); applied by share at: %s", value, f"{cost.max():,.0f}",
             scores.loc[cost.idxmax(), "name"], ", ".join(
                 f"{n} {s:.2f}%" for n, s in zip(scores.loc[capped, "name"], scores.loc[capped, "redund_noalt_pct"])))
    return swap_column(scores, "redund_penalty_routed_pct", cost), settings


def port_assignment(scores, settings, value, inputs, log):
    """Table 4: single nearest gateway (Lab 7/8 first pass) for trade and redundancy.

    Trade and S08's redundancy columns come from the first-pass snapshot; conflict stays
    on today's run. S09's current rules (Decision 16 cap and null fill) are applied.
    """
    if value != "nearest_gateway":
        raise ValueError(f"port_assignment '{value}' is not supported (only 'nearest_gateway')")
    first = inputs["first_pass"].set_index("chokepoint_id")
    if set(first.index) != set(scores["chokepoint_id"]):
        raise ValueError("The first-pass snapshot and chokepoint_scores have different chokepoints")
    changed = scores.copy()
    for field in [main_trade_field(settings)] + REDUND_COLUMNS:
        changed[field] = scores["chokepoint_id"].map(first[field])
    return changed, settings


HANDLERS = {
    "skew_transform": with_setting("scoring", "transform"),
    "ring_main_km": with_setting("conflict", "ring_main_km"),
    "method_main": with_setting("conflict", "method_main"),
    "conflict_mix": with_setting("scoring", "conflict_mix"),
    "gdelt_min_sources": stored_tension(lambda n, ring: f"tens_s{n}_n_{ring}"),
    "gdelt_root_codes": stored_tension(lambda codes, ring: f"tens_r{codes[0]}{codes[-1]}_n_{ring}"),
    "event_weighting": with_setting("conflict", "event_weighting_main"),
    "imo_count_rule": imo_count_rule,
    "port_assignment": port_assignment,
    "trade_measure": trade_measure,
    "combination": with_setting("scoring", "combination_main"),
    "redundancy_measure": redundancy_measure,
    "no_route_cap": with_setting("scoring", "no_route_cap"),
    "tension_ring_km": stored_tension(lambda r, ring: f"tens_n_{r}"),
    "tension_top_place": tension_top_place,
}


def rank_of(scores, settings):
    """S09's composite rank (tasks 5.2-5.3) for one set of inputs and settings."""
    return s09.score(scores, settings, main_trade_field(settings))["score_rank"]


def one_at_a_time(scores, settings, inputs, log):
    """Task 5.5: ranks for each alternative, and the list of alternatives not run.

    Returns ({label: ranks}, [rows for alternatives not run]).
    """
    sens = settings["sensitivity"]
    if sens["approach"] != "one_at_a_time":
        raise ValueError(f"sensitivity.approach '{sens['approach']}' is not supported")
    not_run = sens.get("not_run") or {}
    ranks, skipped = {}, []
    for key, values in sens["alternatives"].items():
        for value in values:
            name = label(key, value)
            if key in not_run:
                skipped.append({"group": "one_at_a_time", "alternative": name, "setting": key,
                                "value": str(value), "status": "not run", "reason": not_run[key]})
                log.info("Task 5.5 | %-45s NOT RUN: %s", name, not_run[key])
                continue
            if key not in HANDLERS:
                raise ValueError(f"No S10 rule for sensitivity alternative '{key}'; add one or list it in not_run")
            changed_scores, changed_settings = HANDLERS[key](scores, settings, value, inputs, log)
            ranks[name] = rank_of(changed_scores, changed_settings)
    return ranks, skipped


# ---------------------------------------------------------------------
# Task 5.6: weight grid
# ---------------------------------------------------------------------

def weight_grid(settings):
    """Task 5.6: every (trade, conflict, redundancy) weight set in steps of `step`, each at least
    `minimum`, summing to 1 (36 sets for 0.1 and 0.1)."""
    grid = settings["sensitivity"]["weight_grid"]
    units, low = round(1 / grid["step"]), round(grid["minimum"] / grid["step"])
    for t in range(low, units - 2 * low + 1):
        for c in range(low, units - t - low + 1):
            yield {"trade": t / units, "conflict": c / units, "redundancy": (units - t - c) / units}


def grid_runs(scores, settings, log):
    """Task 5.6: ranks for every weight set; the combination rule stays the main one."""
    ranks = {}
    for weights in weight_grid(settings):
        changed = copy.deepcopy(settings)
        changed["scoring"]["weights"] = weights
        name = f"weights=T{weights['trade']:.1f}_C{weights['conflict']:.1f}_R{weights['redundancy']:.1f}"
        ranks[name] = rank_of(scores, changed)
    log.info("Task 5.6 | weight grid: %d combinations (step %s, minimum %s)", len(ranks),
             settings["sensitivity"]["weight_grid"]["step"], settings["sensitivity"]["weight_grid"]["minimum"])
    return ranks


# ---------------------------------------------------------------------
# Task 5.7: summaries
# ---------------------------------------------------------------------

def compare(main, other, names, top_n):
    """Task 5.7: one run's agreement with the main ranking."""
    rho, p_value = spearmanr(main, other)
    change = other - main
    move = int(change.abs().max())
    movers = change.abs() == move
    row = {"status": "run", "rho": float(rho), "p_value": float(p_value),
           "n_changed": int((change != 0).sum()), "max_move": move,
           "biggest_mover": "none" if move == 0 else "; ".join(
               f"{n} ({a}->{b})" for n, a, b in zip(names[movers], main[movers], other[movers]))}
    for n in top_n:
        row[f"top{n}_overlap"] = len(set(main.index[main <= n]) & set(other.index[other <= n]))
    return row


def summary_table(main, runs, group, names, settings):
    """Task 5.7: one row per run."""
    rows = []
    for name, ranks in runs.items():
        setting, value = name.split("=", 1)
        rows.append({"group": group, "alternative": name, "setting": setting, "value": value,
                     **compare(main, ranks, names, settings["sensitivity"]["top_n"])})
    return rows


def rank_ranges(ranks, prefix, top):
    """Task 5.7: best, worst and median rank, range, and share of runs in the top `top`."""
    table = pd.DataFrame(ranks)
    out = pd.DataFrame(index=table.index)
    out[f"{prefix}_rank_min"] = table.min(axis=1).astype("int64")
    out[f"{prefix}_rank_max"] = table.max(axis=1).astype("int64")
    out[f"{prefix}_rank_median"] = table.median(axis=1)
    out[f"{prefix}_rank_range"] = out[f"{prefix}_rank_max"] - out[f"{prefix}_rank_min"]
    out[f"{prefix}_top{top}_share"] = (table <= top).mean(axis=1)
    return out


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log_first_pass(first, settings, log):
    """Caveats of the nearest_gateway test (logged with every run)."""
    trade = main_trade_field(settings)
    log.info("Task 5.5 | port_assignment nearest_gateway: trade (%s) and redundancy (%s) from the "
             "Lab 7/8 first pass (%s; WPI nearest gateway); conflict from today's run, because the "
             "snapshot has no tens_ columns. Rescored with today's S09 rules (Decision 16).",
             trade, ", ".join(REDUND_COLUMNS), common.get_path(settings, "first_pass_database").name)
    log.info("Task 5.5 |   first pass: zero trade at %s; null redundancy (no affected flows, set to 0) at %s; "
             "no-route cap would be %.1f%% (%s)",
             ", ".join(first.loc[first[trade] == 0, "name"]) or "none",
             ", ".join(first.loc[first["redund_penalty_pct"].isna(), "name"]) or "none",
             first["redund_penalty_routed_pct"].max(),
             first.loc[first["redund_penalty_routed_pct"].idxmax(), "name"])
    log.info("Task 5.5 |   caveats: cannot be compared for the Red Sea scenario, reroute costs (impact_) "
             "or the trade-conflict correlation; any other difference between the Lab 8 pass and "
             "today's S04/S06/S08 (not only the port assignment) is folded into this one test.")


def log_summary(table, log):
    """Task 5.7: agreement of each run with the main ranking."""
    log.info("Task 5.7 | agreement with the main ranking (rho; top-5 and top-10 overlap; chokepoints "
             "changing rank; biggest mover):")
    for _, row in table.iterrows():
        if row["status"] != "run":
            log.info("  %-45s not run: %s", row["alternative"], row["reason"])
            continue
        log.info("  %-45s rho %.3f  top5 %d  top10 %2d  changed %2d  %s", row["alternative"], row["rho"],
                 row["top5_overlap"], row["top10_overlap"], row["n_changed"], row["biggest_mover"])


def log_ranges(scores, summary, top, log):
    """Task 5.7: rank ranges for every chokepoint, by main rank."""
    log.info("Task 5.7 | rank per chokepoint: main; one-at-a-time min-median-max, top-%d share; "
             "weight grid min-median-max, top-%d share", top, top)
    table = scores[["name", "score_rank"]].join(summary).sort_values(["score_rank", "name"])
    for _, r in table.iterrows():
        log.info("  %2d  %-20s  %2d-%4.1f-%2d  %4.0f%%   grid %2d-%4.1f-%2d  %4.0f%%", r["score_rank"], r["name"],
                 r["sens_rank_min"], r["sens_rank_median"], r["sens_rank_max"], 100 * r[f"sens_top{top}_share"],
                 r["sens_grid_rank_min"], r["sens_grid_rank_median"], r["sens_grid_rank_max"],
                 100 * r[f"sens_grid_top{top}_share"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    tables = common.get_path(settings, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    alternatives = settings["sensitivity"]["alternatives"]
    top = min(settings["sensitivity"]["top_n"])

    scores = read_scores(settings, log)
    main_ranks = rank_of(scores, settings)
    if not (main_ranks == scores["score_rank"]).all():
        raise ValueError("S09's score() no longer reproduces score_rank; rerun S09 first")
    log.info("Main run reproduced: S09's score() gives the stored score_rank for all %d chokepoints", len(scores))

    # Inputs the alternatives need beyond chokepoint_scores.
    inputs = {}
    if "port_assignment" in alternatives:
        inputs["first_pass"] = read_scores(settings, log, common.get_path(settings, "first_pass_database"),
                                           "first-pass chokepoint_scores")
        log_first_pass(inputs["first_pass"], settings, log)
    if {"imo_count_rule", "tension_top_place"} & set(alternatives):
        chokepoints = s07.read_chokepoints(settings, log)
        events = s07.read_events(settings, log)
        if "imo_count_rule" in alternatives:
            inputs["imo"] = recount_imo(events, chokepoints, scores, settings, alternatives["imo_count_rule"], log)
        if "tension_top_place" in alternatives:
            inputs["top_place"] = recount_top_place(events, chokepoints, scores, settings, log)
        del events

    # Tasks 5.5-5.6: ranks for every run.
    single, skipped = one_at_a_time(scores, settings, inputs, log)
    grid = grid_runs(scores, settings, log)

    # Task 5.7: agreement with the main run and rank ranges.
    names = scores["name"]
    table = pd.DataFrame(summary_table(main_ranks, single, "one_at_a_time", names, settings) + skipped
                         + summary_table(main_ranks, grid, "weight_grid", names, settings))
    table = table.reindex(columns=SUMMARY_COLUMNS)
    log_summary(table, log)
    path = tables / "sensitivity.csv"
    table.to_csv(path, index=False)
    log.info("Wrote %s (%d rows: %d run one at a time, %d not run, %d weight sets)", path, len(table),
             len(single), len(skipped), len(grid))

    ranks = scores[["chokepoint_id", "name"]].assign(main=main_ranks)
    ranks = ranks.join(pd.DataFrame(single)).join(pd.DataFrame(grid)).sort_values(["main", "name"])
    path = tables / "sensitivity_ranks.csv"
    ranks.to_csv(path, index=False)
    log.info("Wrote %s (%d chokepoints x %d runs)", path, len(ranks), len(single) + len(grid) + 1)

    # One-at-a-time ranges include the main run ("across all runs"); the grid ranges
    # cover the grid's own weight sets (equal weights are not one of them).
    summary = rank_ranges({"main": main_ranks, **single}, "sens", top)
    summary = summary.join(rank_ranges(grid, "sens_grid", top))
    log.info("Task 5.7 | sens_ ranges over %d runs (main + %d alternatives); sens_grid_ over %d weight sets",
             len(single) + 1, len(single), len(grid))
    log_ranges(scores, summary, top, log)

    common.write_layer(scores.join(summary), settings, "chokepoint_scores", log)
    log.info("Run of s10_sensitivity finished")


if __name__ == "__main__":
    main()

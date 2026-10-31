"""S09: trade-conflict correlation and the composite score (tasks 5.1-5.4).

Reads the chokepoint_scores table (one row per included chokepoint, written by S06,
S07, S08 and S08b), then:

1. Task 5.1: Spearman rank correlation, with p-value, between trade exposure and
   conflict exposure for the main pair of fields (scoring.correlation_main) and each
   supplementary pair (scoring.correlation_supplementary). Each row also counts the
   chokepoints with zero trade and with zero conflict, because many tied zeros weaken
   a rank correlation on so few points. -> outputs/tables/correlation.csv
2. Quadrants for the trade-vs-conflict chart (task 6.5): each chokepoint is high or
   low trade and high or low violence, split at the median of the main fields
   ("high" = above the median; a value exactly at the median counts as
   scoring.quadrant_tie, Decision 16). -> quad_trade_conflict
3. Task 5.2: the three components on one 0-1 scale (Equation 5). Trade and conflict
   are transformed with log(1 + x) first. Conflict is the mix (scoring.conflict_mix) of
   rescaled violence and rescaled tension, not rescaled again. Redundancy is Equation 4
   with no-route sub-flows capped at scoring.no_route_cap (Decision 16, recomputed from
   S08's columns); a chokepoint with no affected flows (Bering) gets 0 and a flag.
4. Task 5.3: the weighted sum (Equation 6), its rank, and the multiplicative score
   (Equation 7, components rescaled to scoring.multiplicative_rescale_floor-1).
5. Task 5.4: the same score with each non-main scenario's trade field
   (scoring.scenarios); conflict and redundancy stay on the baseline. Chokepoints closed
   in the scenario are ranked and flagged. -> outputs/tables/scenarios.csv

S09 owns the chokepoint_scores columns starting with "quad_" and "score_" and keeps
every other column. The scoring functions take the settings as an argument, so S10
can call them with one setting changed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common

S09_PREFIXES = ("quad_", "score_")   # S09 owns every chokepoint_scores column starting with these
QUADRANT_FIELD = "quad_trade_conflict"
CSV_COLUMNS = ["variant", "rho", "p_value", "n", "zero_trade", "zero_conflict"]
NO_ROUTE_CAPS = ("max_routed_chokepoint", "max_observed")


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_scores(settings, log):
    """Read chokepoint_scores as a plain table and check the fields S09 needs exist."""
    scores = pd.DataFrame(common.read_layer(settings, "chokepoint_scores"))
    scores = scores.drop(columns=[c for c in scores.columns if c == "geometry"])
    log.info("chokepoint_scores read: %d chokepoints, %d columns", len(scores), len(scores.columns))
    needed = {f for pair in correlation_pairs(settings).values() for f in pair.values()}
    needed |= {s["trade"] for s in settings["scoring"]["scenarios"].values()}
    needed |= set(conflict_fields(settings).values())
    needed |= {"redund_penalty_pct", "redund_penalty_routed_pct", "redund_noalt_pct"}
    missing = sorted(needed - set(scores.columns))
    if missing:
        raise ValueError(f"chokepoint_scores lacks {missing}; run S06, S07 and S08 first")
    return scores


def correlation_pairs(settings):
    """Return {variant: {trade: field, conflict: field}}, main result first."""
    scoring = settings["scoring"]
    pairs = {"main": scoring["correlation_main"]}
    pairs.update(scoring.get("correlation_supplementary") or {})
    return pairs


def conflict_fields(settings):
    """The S07 fields for violence and tension, from conflict.method_main, ring_main_km and
    event_weighting_main (the same rule S07 uses for its main ranking).

    Fatality weighting applies to violence only; tension always counts events.
    """
    conflict = settings["conflict"]
    if conflict["method_main"] == "rings":
        fw = conflict["event_weighting_main"] != "count"
        ring = conflict["ring_main_km"]
        return {"violence": f"viol_{'fw' if fw else 'n'}_{ring}", "tension": f"tens_n_{ring}"}
    if conflict["method_main"] == "density":
        h = conflict["density_main_km"]
        return {"violence": f"viol_kde_{h}", "tension": f"tens_kde_{h}"}
    raise ValueError(f"Unknown conflict.method_main: {conflict['method_main']}")


# ---------------------------------------------------------------------
# Task 5.1: correlation
# ---------------------------------------------------------------------

def correlate(scores, variant, pair, log):
    """Task 5.1: Spearman rho and p-value for one trade/conflict pair, with zero counts."""
    trade, conflict = scores[pair["trade"]], scores[pair["conflict"]]
    rho, p_value = spearmanr(trade, conflict)
    row = {"variant": variant, "trade_field": pair["trade"], "conflict_field": pair["conflict"],
           "rho": float(rho), "p_value": float(p_value), "n": int(len(scores)),
           "zero_trade": int((trade == 0).sum()), "zero_conflict": int((conflict == 0).sum())}
    log.info("Task 5.1 | %-12s %s vs %s: rho = %.3f, p = %.3f, n = %d "
             "(zero trade %d, zero conflict %d)", variant, pair["trade"], pair["conflict"],
             row["rho"], row["p_value"], row["n"], row["zero_trade"], row["zero_conflict"])
    return row


def log_ties(table, log):
    """Task 5.1: note how many tied values each correlation rests on."""
    main = table.iloc[0]
    log.info("Task 5.1 | ties: %d of %d chokepoints have zero trade (%s) and %d have zero "
             "violent events (%s). Ties share an average rank, which weakens a rank "
             "correlation on this few points; scipy's p-value is the t-approximation.",
             main["zero_trade"], main["n"], main["trade_field"],
             main["zero_conflict"], main["conflict_field"])


# ---------------------------------------------------------------------
# Task 6.5 support: quadrants
# ---------------------------------------------------------------------

def quadrants(scores, settings, log):
    """Classify each chokepoint high/low trade and high/low violence at the median.

    A value exactly at the median counts as scoring.quadrant_tie (Decision 16).
    """
    split, tie = settings["scoring"]["quadrant_split"], settings["scoring"]["quadrant_tie"]
    if split != "median":
        raise ValueError(f"scoring.quadrant_split '{split}' is not supported (only 'median')")
    if tie not in ("low", "high"):
        raise ValueError(f"scoring.quadrant_tie '{tie}' must be 'low' or 'high'")
    pair = settings["scoring"]["correlation_main"]
    labels = {}
    for axis, short in (("trade", "trade"), ("conflict", "violence")):
        values = scores[pair[axis]]
        median = values.median()
        high = values >= median if tie == "high" else values > median
        at_median = values == median
        log.info("Task 6.5 | %s median = %s; high %d, low %d; exactly at the median %d (counted %s: %s)",
                 pair[axis], f"{median:,.6g}", int(high.sum()), int((~high).sum()),
                 int(at_median.sum()), tie, ", ".join(scores.loc[at_median, "name"]) or "none")
        labels[axis] = high.map({True: f"high_{short}", False: f"low_{short}"})
    return labels["trade"] + "_" + labels["conflict"]


def log_quadrants(scores, settings, log):
    """List the chokepoints in each quadrant, sorted by trade exposure."""
    pair = settings["scoring"]["correlation_main"]
    for quadrant, group in sorted(scores.groupby(QUADRANT_FIELD)):
        group = group.sort_values(pair["trade"], ascending=False)
        log.info("Quadrant %s (%d):", quadrant, len(group))
        for _, row in group.iterrows():
            log.info("    %-20s %s = %s  %s = %s", row["name"], pair["trade"],
                     f"{row[pair['trade']]:,.0f}", pair["conflict"], f"{row[pair['conflict']]:,.0f}")


# ---------------------------------------------------------------------
# Task 5.2: components on one scale
# ---------------------------------------------------------------------

def rescale(values, floor=0.0):
    """Equation 5: min-max rescale to floor..1 (0 = lowest chokepoint, 1 = highest).

    If every chokepoint has the same value there is nothing to rank, so all get floor.
    """
    values = values.astype(float)
    spread = values.max() - values.min()
    if spread == 0:
        return pd.Series(floor, index=values.index)
    return floor + (1 - floor) * (values - values.min()) / spread


def transform(values, component, settings):
    """Task 5.2: log(1 + x) for the components in scoring.transform_components."""
    scoring = settings["scoring"]
    if component not in scoring["transform_components"]:
        return values.astype(float)
    if scoring["transform"] == "log1p":
        return np.log1p(values.astype(float))
    if scoring["transform"] == "none":
        return values.astype(float)
    raise ValueError(f"Unknown scoring.transform: {scoring['transform']}")


def redundancy_penalty(scores, settings):
    """Decision 16: Equation 4 with no-route sub-flows capped at scoring.no_route_cap.

    S08 stores the penalty over rerouted sub-flows (redund_penalty_routed_pct) and the
    value share with no sea route (redund_noalt_pct), so Equation 4 is
    routed x (1 - share) + cap x share. "max_routed_chokepoint" caps at the largest
    redund_penalty_routed_pct; "max_observed" keeps S08's redund_penalty_pct (Decision 10(d)).
    Returns (penalty in %, cap in % or None). Chokepoints with no affected flows stay null.
    """
    rule = settings["scoring"]["no_route_cap"]
    if rule not in NO_ROUTE_CAPS:
        raise ValueError(f"Unknown scoring.no_route_cap '{rule}'; use one of {NO_ROUTE_CAPS}")
    if rule == "max_observed":
        return scores["redund_penalty_pct"].astype(float), None
    cap = float(scores["redund_penalty_routed_pct"].max())
    share = scores["redund_noalt_pct"] / 100
    penalty = scores["redund_penalty_routed_pct"].fillna(0) * (1 - share) + cap * share
    return penalty.where(scores["redund_penalty_pct"].notna()), cap


def fill_redundancy(penalty, settings):
    """Decision 16: a chokepoint with no affected flows (null penalty) gets 0, flagged."""
    rule = settings["scoring"]["null_redundancy"]
    if rule != "zero":
        raise ValueError(f"scoring.null_redundancy '{rule}' is not supported (only 'zero')")
    filled = penalty.isna()
    return penalty.fillna(0.0), filled


def components(scores, settings, trade_field, floor=0.0):
    """Task 5.2: rescaled trade, violence, tension, conflict and redundancy (floor..1).

    Conflict is the conflict_mix of rescaled violence and tension, not rescaled again
    (Decision 16), so with a floor it also runs floor..1.
    """
    fields = conflict_fields(settings)
    mix = settings["scoring"]["conflict_mix"]
    penalty, _ = redundancy_penalty(scores, settings)
    penalty, _ = fill_redundancy(penalty, settings)
    out = pd.DataFrame(index=scores.index)
    out["trade"] = rescale(transform(scores[trade_field], "trade", settings), floor)
    out["viol"] = rescale(transform(scores[fields["violence"]], "conflict", settings), floor)
    out["tens"] = rescale(transform(scores[fields["tension"]], "conflict", settings), floor)
    out["conflict"] = mix["violence"] * out["viol"] + mix["tension"] * out["tens"]
    out["redund"] = rescale(transform(penalty, "redundancy", settings), floor)
    return out


# ---------------------------------------------------------------------
# Task 5.3: combined scores
# ---------------------------------------------------------------------

def rank(values):
    """1 = most vulnerable; ties share the higher rank (as S07)."""
    return values.rank(ascending=False, method="min").astype("int64")


def weighted_sum(parts, settings):
    """Equation 6: the weighted sum and each component's contribution (weight x component)."""
    weights = settings["scoring"]["weights"]
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"scoring.weights must sum to 1, not {sum(weights.values())}")
    contrib = pd.DataFrame({c: weights[c] * parts[c] for c in ("trade", "conflict")}, index=parts.index)
    contrib["redund"] = weights["redundancy"] * parts["redund"]
    return contrib.sum(axis=1), contrib


def multiplicative(scores, settings, trade_field):
    """Equation 7: product of the components (rescaled floor..1) raised to their weights."""
    weights = settings["scoring"]["weights"]
    parts = components(scores, settings, trade_field, settings["scoring"]["multiplicative_rescale_floor"])
    return (parts["trade"] ** weights["trade"] * parts["conflict"] ** weights["conflict"]
            * parts["redund"] ** weights["redundancy"])


def score(scores, settings, trade_field):
    """Tasks 5.2-5.3 for one trade field: components, both scores and their ranks.

    The combination rule in scoring.combination_main sets which score `rank` follows;
    S10 changes it to test the other.
    """
    parts = components(scores, settings, trade_field)
    total, contrib = weighted_sum(parts, settings)
    mult = multiplicative(scores, settings, trade_field)
    main = {"weighted_sum": total, "multiplicative": mult}.get(settings["scoring"]["combination_main"])
    if main is None:
        raise ValueError(f"Unknown scoring.combination_main: {settings['scoring']['combination_main']}")
    out = parts.add_prefix("score_")
    out = out.join(contrib.add_prefix("score_part_"))
    out["score_composite"] = total
    out["score_mult"] = mult
    out["score_rank"] = rank(main)
    out["score_mult_rank"] = rank(mult)
    return out


def main_scenario(settings):
    """The routing scenario marked main: true (the baseline)."""
    main = [s for s, cfg in settings["routing"]["scenarios"].items() if cfg.get("main")]
    if len(main) != 1:
        raise ValueError(f"routing.scenarios must have exactly one main scenario, not {main}")
    return main[0]


def composite(scores, settings, log=None):
    """Tasks 5.2-5.4: every score_ column for the main scenario and each other scenario."""
    penalty, cap = redundancy_penalty(scores, settings)
    penalty, filled = fill_redundancy(penalty, settings)
    main = main_scenario(settings)
    scenarios = settings["scoring"]["scenarios"]
    out = score(scores, settings, scenarios[main]["trade"])
    out.insert(0, "score_redund_pct", penalty)
    out.insert(1, "score_redund_filled", filled)
    for name, cfg in scenarios.items():
        if name == main:
            continue
        suffix = cfg["suffix"]
        other = score(scores, settings, cfg["trade"])
        closed = settings["routing"]["scenarios"][name]["closed"]
        out[f"score_trade_{suffix}"] = other["score_trade"]
        out[f"score_composite_{suffix}"] = other["score_composite"]
        out[f"score_rank_{suffix}"] = other["score_rank"]
        out[f"score_rank_change_{suffix}"] = other["score_rank"] - out["score_rank"]
        out[f"score_closed_{suffix}"] = scores["name"].isin(closed)
    if log:
        log_redundancy(scores, penalty, filled, cap, settings, log)
    return out


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log_redundancy(scores, penalty, filled, cap, settings, log):
    """Decision 16: the cap, the chokepoints it changes, and the filled nulls."""
    rule = settings["scoring"]["no_route_cap"]
    if cap is None:
        log.info("Decision 16 | no_route_cap %s: S08's redund_penalty_pct used unchanged", rule)
    else:
        setter = scores.loc[scores["redund_penalty_routed_pct"].idxmax(), "name"]
        log.info("Decision 16 | no_route_cap %s: no-route sub-flows capped at %.1f%% (set by %s); "
                 "S08's largest penalty was %.1f%%", rule, cap, setter, scores["redund_penalty_pct"].max())
        changed = ~np.isclose(penalty, scores["redund_penalty_pct"].fillna(0))
        for _, row in scores[changed].iterrows():
            log.info("Decision 16 |   %-20s Eq. 4 %.1f%% -> %.1f%% (no sea route for %.1f%% of value)",
                     row["name"], row["redund_penalty_pct"], penalty[row.name], row["redund_noalt_pct"])
    log.info("Decision 16 | null redundancy filled with 0 (flag score_redund_filled): %s",
             ", ".join(scores.loc[filled, "name"]) or "none")


def log_scores(scores, out, settings, log):
    """Task 5.3: every chokepoint by rank, with its components and both scores."""
    fields = conflict_fields(settings)
    log.info("Task 5.2 | fields: trade %s, violence %s, tension %s, redundancy score_redund_pct; "
             "transform %s on %s; conflict = %s", settings["scoring"]["scenarios"][main_scenario(settings)]["trade"],
             fields["violence"], fields["tension"], settings["scoring"]["transform"],
             settings["scoring"]["transform_components"], settings["scoring"]["conflict_mix"])
    log.info("Task 5.3 | composite (%s) by rank; components 0-1:", settings["scoring"]["combination_main"])
    log.info("  %4s  %-20s %6s %6s %6s %6s %6s  %6s  %6s %4s", "rank", "chokepoint", "trade", "viol",
             "tens", "confl", "redund", "score", "mult", "rank")
    table = scores[["name"]].join(out).sort_values(["score_rank", "name"])
    for _, row in table.iterrows():
        log.info("  %4d  %-20s %6.3f %6.3f %6.3f %6.3f %6.3f  %6.3f  %6.3f %4d", row["score_rank"], row["name"],
                 row["score_trade"], row["score_viol"], row["score_tens"], row["score_conflict"],
                 row["score_redund"], row["score_composite"], row["score_mult"], row["score_mult_rank"])
    rho, p = spearmanr(out["score_composite"], out["score_mult"])
    log.info("Task 5.3 | weighted sum vs multiplicative: Spearman rho = %.3f, p = %.2g; "
             "chokepoints in both top 10: %d", rho, p,
             len(set(out.index[out["score_rank"] <= 10]) & set(out.index[out["score_mult_rank"] <= 10])))


def log_scenarios(scores, out, settings, log):
    """Task 5.4: rank changes under each non-main scenario."""
    main = main_scenario(settings)
    for name, cfg in settings["scoring"]["scenarios"].items():
        if name == main:
            continue
        s = cfg["suffix"]
        change = out[f"score_rank_change_{s}"]
        rho, p = spearmanr(out["score_rank"], out[f"score_rank_{s}"])
        log.info("Task 5.4 | %s vs %s: rank Spearman rho = %.3f, p = %.2g; %d of %d chokepoints change rank",
                 name, main, rho, p, int((change != 0).sum()), len(out))
        main_trade = settings["scoring"]["scenarios"][main]["trade"]
        table = scores[["name", main_trade, cfg["trade"]]].join(out)
        for _, row in table.sort_values(f"score_rank_{s}").iterrows():
            log.info("  %-20s rank %2d -> %2d (%+d)  trade %s -> %s%s", row["name"], row["score_rank"],
                     row[f"score_rank_{s}"], row[f"score_rank_change_{s}"],
                     f"{row[main_trade]:,.0f}", f"{row[cfg['trade']]:,.0f}",
                     "  [closed in this scenario]" if row[f"score_closed_{s}"] else "")


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

def write_tables(scores, out, settings, tables, log):
    """P26 composite_scores.csv and P27 scenarios.csv."""
    fields = conflict_fields(settings)
    main = main_scenario(settings)
    trade = settings["scoring"]["scenarios"][main]["trade"]
    base = scores[["chokepoint_id", "name", trade, fields["violence"], fields["tension"]]]
    keep = [c for c in out.columns if not any(c.endswith(f"_{cfg['suffix']}")
            for n, cfg in settings["scoring"]["scenarios"].items() if n != main)]
    table = base.join(out[keep]).join(scores[[QUADRANT_FIELD]]).sort_values("score_rank")
    path = tables / "composite_scores.csv"
    table.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(table))

    rows = []
    for name, cfg in settings["scoring"]["scenarios"].items():
        if name == main:
            continue
        s = cfg["suffix"]
        part = scores[["chokepoint_id", "name"]].copy()
        part["scenario"] = name
        part["closed"] = out[f"score_closed_{s}"]
        part["trade_value_main"] = scores[trade]
        part["trade_value_scenario"] = scores[cfg["trade"]]
        part["trade_value_change"] = part["trade_value_scenario"] - part["trade_value_main"]
        part["score_trade_main"] = out["score_trade"]
        part["score_trade_scenario"] = out[f"score_trade_{s}"]
        part["score_main"] = out["score_composite"]
        part["score_scenario"] = out[f"score_composite_{s}"]
        part["rank_main"] = out["score_rank"]
        part["rank_scenario"] = out[f"score_rank_{s}"]
        part["rank_change"] = out[f"score_rank_change_{s}"]
        rows.append(part.sort_values("rank_scenario"))
    path = tables / "scenarios.csv"
    pd.concat(rows).to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, sum(len(r) for r in rows))


def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    tables = common.get_path(settings, "tables")
    tables.mkdir(parents=True, exist_ok=True)

    scores = read_scores(settings, log)

    # Task 5.1: main and supplementary correlations in one table.
    table = pd.DataFrame([correlate(scores, variant, pair, log)
                          for variant, pair in correlation_pairs(settings).items()])
    log_ties(table, log)
    out = tables / "correlation.csv"
    table[CSV_COLUMNS].to_csv(out, index=False)
    log.info("Wrote %s (%d rows)", out, len(table))

    # Replace only S09's columns; keep every other column.
    scores = scores.drop(columns=[c for c in scores.columns if c.startswith(S09_PREFIXES)])

    # Task 6.5 support: quadrants.
    scores[QUADRANT_FIELD] = quadrants(scores, settings, log)
    log_quadrants(scores, settings, log)

    # Tasks 5.2-5.4: components, composite, multiplicative, scenario comparison.
    result = composite(scores, settings, log)
    log_scores(scores, result, settings, log)
    log_scenarios(scores, result, settings, log)
    write_tables(scores, result, settings, tables, log)

    scores = scores.join(result)
    common.write_layer(scores, settings, "chokepoint_scores", log)
    log.info("Run of s09_composite finished")


if __name__ == "__main__":
    main()

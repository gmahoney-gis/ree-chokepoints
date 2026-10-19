"""S09: trade-conflict correlation (task 5.1; first pass for Lab 7).

Reads the chokepoint_scores table (one row per included chokepoint, written by S06
and S07), then:

1. Task 5.1: Spearman rank correlation, with p-value, between trade exposure and
   conflict exposure for the main pair of fields (scoring.correlation_main) and each
   supplementary pair (scoring.correlation_supplementary). Each row also counts the
   chokepoints with zero trade and with zero conflict, because many tied zeros weaken
   a rank correlation on so few points. -> outputs/tables/correlation.csv
2. Quadrants for the trade-vs-conflict chart (task 6.5): each chokepoint is high or
   low trade and high or low violence, split at the median of the main fields
   ("high" = above the median). Written to chokepoint_scores as
   trade_conflict_quadrant; every other column is left as it is.

The composite score (tasks 5.2-5.4) is not part of the Lab 7 first pass.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import common

QUADRANT_FIELD = "trade_conflict_quadrant"   # the only chokepoint_scores column S09 writes so far
CSV_COLUMNS = ["variant", "rho", "p_value", "n", "zero_trade", "zero_conflict"]


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_scores(settings, log):
    """Read chokepoint_scores as a plain table and check the fields S09 needs exist."""
    scores = pd.DataFrame(common.read_layer(settings, "chokepoint_scores"))
    scores = scores.drop(columns=[c for c in scores.columns if c == "geometry"])
    log.info("Task 5.1 | chokepoint_scores read: %d chokepoints, %d columns",
             len(scores), len(scores.columns))
    needed = {f for pair in correlation_pairs(settings).values() for f in pair.values()}
    missing = sorted(needed - set(scores.columns))
    if missing:
        raise ValueError(f"chokepoint_scores lacks {missing}; run S06 and S07 first")
    return scores


def correlation_pairs(settings):
    """Return {variant: {trade: field, conflict: field}}, main result first."""
    scoring = settings["scoring"]
    pairs = {"main": scoring["correlation_main"]}
    pairs.update(scoring.get("correlation_supplementary") or {})
    return pairs


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
    """Classify each chokepoint high/low trade and high/low violence at the median."""
    split = settings["scoring"]["quadrant_split"]
    if split != "median":
        raise ValueError(f"scoring.quadrant_split '{split}' is not supported (only 'median')")
    pair = settings["scoring"]["correlation_main"]
    labels = {}
    for axis, short in (("trade", "trade"), ("conflict", "violence")):
        values = scores[pair[axis]]
        median = values.median()
        high = values > median
        log.info("Task 6.5 | %s median = %s; above (high) %d, at or below (low) %d, "
                 "exactly at the median %d", pair[axis], f"{median:,.6g}",
                 int(high.sum()), int((~high).sum()), int((values == median).sum()))
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

    # Task 6.5 support: replace only the quadrant column; keep every other column.
    scores[QUADRANT_FIELD] = quadrants(scores, settings, log)
    log_quadrants(scores, settings, log)
    common.write_layer(scores, settings, "chokepoint_scores", log)
    log.info("Run of s09_composite finished")


if __name__ == "__main__":
    main()

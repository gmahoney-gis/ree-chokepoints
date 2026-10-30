"""S07: conflict exposure of each chokepoint (tasks 4.7-4.11).

Reads the included chokepoints and the events layer and scores two categories:
violence (ACLED events plus IMO GISIS incidents, prefix viol_) and tension (GDELT,
prefix tens_). For each category:

1. Task 4.7: great-circle distance from every event to every included chokepoint.
2. Task 4.8: ring counts at conflict.ring_distances_km (Equation 2). Violence counts each
   event once and also gives the fatality-weighted variant (fatalities + 1).
3. Task 4.9: density exposure at conflict.density_bandwidths_km (Equation 3,
   Gaussian kernel, distances and bandwidths in km, so values are events per km2).
4. Task 4.10: Spearman rank correlation between ring counts and density at each
   matching distance -> outputs/tables/conflict_methods.csv.
5. Task 4.11: chokepoint pairs whose rings overlap at each distance and the share of
   events they share -> outputs/tables/ring_overlap.csv.

Which events are counted:
- Violence: the IMO GISIS counting rule (Decision 9, events.imo_gisis.count_rule, through
  common.countable_events). Each row is one event (weight 1). Each IMO GISIS incident has
  0 fatalities, so it weighs 1 in the fatality-weighted variant.
- Tension: GDELT rows passing events.gdelt.main_filter (Decisions 5 and 15, through
  common.countable_tension). Each row stands for event_count merged events, so every
  count, density and overlap is weighted by event_count. GDELT has no fatalities, so
  there is no fatality-weighted variant for tension. For S10, S07 also writes ring counts
  for the other source thresholds (conflict.gdelt_min_sources_tested -> tens_s1_n_, tens_s3_n_)
  and the root-code alternatives (sensitivity.alternatives.gdelt_root_codes -> tens_r1317_n_).
  The log adds, per chokepoint, the international and state_actor shares and the share of
  the top location (Decision 15(d), (i)).

Results go to the chokepoint_scores table (one row per included chokepoint). Later
scripts add their own fields to that table, so S07 replaces only its own columns
(those starting with the prefixes in S07_PREFIXES) and keeps all others. A copy goes
to outputs/tables/conflict_exposure.csv. The log holds counts and per-chokepoint
summaries only, never individual event rows or place names.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common

# Field prefixes per event category; S07 owns every chokepoint_scores column that
# starts with one of these.
CATEGORY_PREFIX = {"violence": "viol", "tension": "tens"}
S07_PREFIXES = tuple(f"{p}_" for p in CATEGORY_PREFIX.values())


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def read_chokepoints(settings, log):
    """Read the chokepoints layer and keep the included ones."""
    chokepoints = common.read_layer(settings, "chokepoints")
    included = chokepoints["included"].str.strip().str.lower() == "yes"
    common.log_filter(log, "Task 4.7 | included chokepoints", int(included.sum()),
                      int((~included).sum()), "included is not 'yes' in the screening file")
    return chokepoints[included].reset_index(drop=True)


def read_events(settings, log):
    """Read the events layer and log what it holds by source and category."""
    events = common.read_layer(settings, "events")
    log.info("Task 4.7 | events read: %s; by source and category: %s", f"{len(events):,}",
             ", ".join(f"{s}/{c} {n:,}" for (s, c), n in
                       events.groupby(["source", "category"]).size().items()))
    return events


def violence_events(events, settings, log):
    """Violence rows after the IMO GISIS counting rule (Decision 9); each must weigh 1."""
    violence = common.countable_events(events[events["category"] == "violence"], settings, log)
    if (violence["event_count"] != 1).any():
        raise ValueError("Violence rows must have event_count 1 (ACLED and IMO GISIS are single events)")
    log.info("Task 4.7 | category violence: %s events", f"{len(violence):,}")
    return violence


def tension_variants(events, settings, log):
    """Tension rows for the main filter and each S10 alternative, keyed by field prefix.

    Main: events.gdelt.main_filter -> "tens". Source thresholds in
    conflict.gdelt_min_sources_tested other than the main one -> "tens_s<n>". Root-code
    lists in sensitivity.alternatives.gdelt_root_codes -> "tens_r<first><last>"
    (e.g. [13, 15, 16, 17] -> tens_r1317), each at the main source threshold.
    """
    main_sources = settings["events"]["gdelt"]["main_filter"]["min_sources"]
    variants = {"tens": common.countable_tension(events, settings, log)}
    for n in settings["conflict"]["gdelt_min_sources_tested"]:
        if n != main_sources:
            variants[f"tens_s{n}"] = common.countable_tension(events, settings, log, min_sources=n)
    for codes in settings["sensitivity"]["alternatives"]["gdelt_root_codes"]:
        name = f"tens_r{codes[0]}{codes[-1]}"
        if name in variants:
            raise ValueError(f"Two root-code alternatives share the field prefix {name}")
        variants[name] = common.countable_tension(events, settings, log, root_codes=codes)
    for name, rows in variants.items():
        log.info("Task 4.7 | category tension, %s: %s rows standing for %s events (sum of event_count)",
                 name, f"{len(rows):,}", f"{int(rows['event_count'].sum()):,}")
    return variants


# ---------------------------------------------------------------------
# Task 4.7: distances
# ---------------------------------------------------------------------

def event_distances(events, chokepoints, settings, log):
    """Task 4.7: distance in km from every event (rows) to every chokepoint (columns)."""
    distances = common.distance_matrix_km(events.geometry.x, events.geometry.y,
                                          chokepoints.geometry.x, chokepoints.geometry.y, settings)
    log.info("Task 4.7 | distance matrix: %s events x %d chokepoints; nearest-chokepoint "
             "distance median %.0f km, max %.0f km", f"{distances.shape[0]:,}", distances.shape[1],
             np.median(distances.min(axis=1)), distances.min(axis=1).max())
    return distances


def log_gisis_rings(events, distances, chokepoints, settings, log):
    """Log how many IMO GISIS incidents fall in each chokepoint's main ring (counts only)."""
    ring = settings["conflict"]["ring_main_km"]
    gisis = (events["source"] == common.GISIS_SOURCE).to_numpy()
    counts = (distances[gisis] <= ring).sum(axis=0)
    log.info("Task 4.8 | IMO GISIS incidents within %d km, chokepoints with at least one: %d",
             ring, int((counts > 0).sum()))
    for i in np.argsort(-counts, kind="stable"):
        if counts[i] > 0:
            log.info("Task 4.8 |   %-20s %3d", chokepoints.at[i, "name"], counts[i])


# ---------------------------------------------------------------------
# Task 4.8: rings (Equation 2)
# ---------------------------------------------------------------------

def ring_exposure(distances, weights, fatalities, prefix, settings, log):
    """Task 4.8: weighted event count (and fatality-weighted sum) within each ring distance.

    weights is 1 per violence event and event_count per GDELT row. fatalities is None
    for tension, which then gets no fatality-weighted (_fw_) field.
    """
    fields = {}
    for r in settings["conflict"]["ring_distances_km"]:
        inside = distances <= r
        fields[f"{prefix}_n_{r}"] = (inside * weights[:, None]).sum(axis=0).astype("int64")
        if fatalities is not None:
            fields[f"{prefix}_fw_{r}"] = (inside * (fatalities + 1.0)[:, None]).sum(axis=0).astype("int64")
        log.info("Task 4.8 | %s rings %d km: %d of %d chokepoints have no events; "
                 "events counted toward at least one chokepoint: %s", prefix, r,
                 int((fields[f"{prefix}_n_{r}"] == 0).sum()), distances.shape[1],
                 f"{int(weights[inside.any(axis=1)].sum()):,}")
    return fields


# ---------------------------------------------------------------------
# Task 4.9: density (Equation 3)
# ---------------------------------------------------------------------

def density_exposure(distances, weights, prefix, settings, log):
    """Task 4.9: Gaussian kernel density read at each chokepoint (Equation 3).

    D = sum over events of w/(2*pi*h^2) * exp(-d^2 / (2*h^2)), with d and h in km and
    w the event weight (1, or event_count for GDELT). S05 kept only events within
    events.prefilter_radius_km of a chokepoint, so the kernel tail beyond that radius
    is missing; the log gives the share of each event's kernel that lies beyond it
    (exp(-R^2 / 2h^2) for a 2-D Gaussian).
    """
    prefilter = settings["events"]["prefilter_radius_km"]
    fields = {}
    for h in settings["conflict"]["density_bandwidths_km"]:
        kernel = np.exp(-distances ** 2 / (2.0 * h ** 2)) / (2.0 * np.pi * h ** 2)
        fields[f"{prefix}_kde_{h}"] = (kernel * weights[:, None]).sum(axis=0)
        beyond = np.exp(-prefilter ** 2 / (2.0 * h ** 2))
        log.info("Task 4.9 | %s density h = %d km: share of a kernel beyond the %d km "
                 "prefilter radius = %.4f%s", prefix, h, prefilter, beyond,
                 "  (WARNING: events beyond the prefilter radius are missing, so this "
                 "bandwidth is truncated)" if beyond > 0.01 else "")
    return fields


# ---------------------------------------------------------------------
# Task 4.10: method comparison
# ---------------------------------------------------------------------

def compare_methods(scores, prefix, settings, log):
    """Task 4.10: Spearman correlation between ring counts and density at matching distances."""
    rings = settings["conflict"]["ring_distances_km"]
    bandwidths = settings["conflict"]["density_bandwidths_km"]
    rows = []
    for r in sorted(set(rings) & set(bandwidths)):
        ring_col, kde_col = f"{prefix}_n_{r}", f"{prefix}_kde_{r}"
        rho, p = spearmanr(scores[ring_col], scores[kde_col])
        rank_ring = scores[ring_col].rank(ascending=False, method="min")
        rank_kde = scores[kde_col].rank(ascending=False, method="min")
        rows.append({"category": prefix, "distance_km": r, "ring_field": ring_col,
                     "density_field": kde_col, "n_chokepoints": len(scores),
                     "spearman_rho": round(float(rho), 4), "p_value": float(p),
                     "max_rank_shift": int((rank_ring - rank_kde).abs().max())})
        log.info("Task 4.10 | %s %d km: Spearman rho = %.3f, p = %.2g, largest rank shift = %d",
                 prefix, r, rho, p, rows[-1]["max_rank_shift"])
    return rows


# ---------------------------------------------------------------------
# Task 4.11: ring overlap
# ---------------------------------------------------------------------

def ring_overlap(distances, weights, chokepoints, prefix, settings, log):
    """Task 4.11: pairs whose rings overlap at each distance and the events they share.

    Two rings of radius r overlap when the chokepoints are less than 2r apart. Events
    are weighted as in ring_exposure. share_of_union is shared events / events in either
    ring; share_of_a and share_of_b are shared events as a share of each chokepoint's own
    ring count.
    """
    centres = common.distance_matrix_km(chokepoints.geometry.x, chokepoints.geometry.y,
                                        chokepoints.geometry.x, chokepoints.geometry.y, settings)
    rows = []
    for r in settings["conflict"]["ring_distances_km"]:
        inside = distances <= r
        n_pairs = 0
        for a, b in combinations(range(len(chokepoints)), 2):
            if centres[a, b] >= 2 * r:
                continue
            n_pairs += 1
            n_a, n_b = int(weights[inside[:, a]].sum()), int(weights[inside[:, b]].sum())
            shared = int(weights[inside[:, a] & inside[:, b]].sum())
            union = n_a + n_b - shared
            rows.append({"category": prefix, "distance_km": r,
                         "chokepoint_a": chokepoints.at[a, "chokepoint_id"],
                         "name_a": chokepoints.at[a, "name"],
                         "chokepoint_b": chokepoints.at[b, "chokepoint_id"],
                         "name_b": chokepoints.at[b, "name"],
                         "centre_distance_km": round(float(centres[a, b]), 1),
                         "events_a": n_a, "events_b": n_b, "events_shared": shared,
                         "share_of_union": round(shared / union, 4) if union else np.nan,
                         "share_of_a": round(shared / n_a, 4) if n_a else np.nan,
                         "share_of_b": round(shared / n_b, 4) if n_b else np.nan})
        log.info("Task 4.11 | %s rings %d km: %d overlapping pairs", prefix, r, n_pairs)
    for row in rows:
        log.info("Task 4.11 |   %4d km  %-20s / %-20s %6.0f km apart  shared %6s events "
                 "(%.0f%% of union)", row["distance_km"], row["name_a"], row["name_b"],
                 row["centre_distance_km"], f"{row['events_shared']:,}",
                 100 * row["share_of_union"] if pd.notna(row["share_of_union"]) else 0)
    return rows


# ---------------------------------------------------------------------
# Tension context (log only; Decision 15(d) and (i))
# ---------------------------------------------------------------------

def log_tension_context(tension, distances, chokepoints, settings, log):
    """Log, per chokepoint within the main ring, the international and state_actor shares
    and the top location's share with its country.

    A location is one exact coordinate pair: every GDELT row placed at one feature has
    the same coordinates. Shares are weighted by event_count. Only countries and shares
    are logged, never place names or coordinates.
    """
    ring = settings["conflict"]["ring_main_km"]
    weights = tension["event_count"].to_numpy(dtype=float)
    international = tension["international"].fillna(False).to_numpy(dtype=bool)
    state_actor = tension["state_actor"].fillna(False).to_numpy(dtype=bool)
    places = pd.DataFrame({"x": tension.geometry.x, "y": tension.geometry.y,
                           "country": tension["country_iso3"].fillna("none"), "w": weights})
    log.info("Decision 15 | tension within %d km (main filter, weighted by event_count): "
             "international and state_actor shares; top location = the place with the most "
             "events, its country and share", ring)
    log.info("Decision 15 |   %-20s %9s %6s %6s  %s", "chokepoint", "events", "intl", "state", "top location")
    for j in range(len(chokepoints)):
        inside = distances[:, j] <= ring
        total = weights[inside].sum()
        if total == 0:
            log.info("Decision 15 |   %-20s %9d", chokepoints.at[j, "name"], 0)
            continue
        by_place = places[inside].groupby(["x", "y", "country"])["w"].sum()
        top = by_place.idxmax()
        log.info("Decision 15 |   %-20s %9s %5.0f%% %5.0f%%  %s %.0f%%", chokepoints.at[j, "name"],
                 f"{int(total):,}", 100 * weights[inside & international].sum() / total,
                 100 * weights[inside & state_actor].sum() / total, top[2], 100 * by_place.max() / total)
    log.info("Decision 15 | Limitation, reported not corrected: GDELT places many events at a few "
             "cities (Gaza for Suez, as GDELT codes Gaza as Israel; London, Istanbul, Taipei, Dubai, "
             "Manila elsewhere), so tension near a chokepoint partly measures news about a nearby "
             "city or capital.")


# ---------------------------------------------------------------------
# Scores table and outputs
# ---------------------------------------------------------------------

def main_field(settings, prefix):
    """The field the main ranking uses, from conflict.method_main / ring_main_km / weighting.

    Fatality weighting applies to violence only; tension always counts events.
    """
    conflict = settings["conflict"]
    if conflict["method_main"] == "rings":
        fw = conflict["event_weighting_main"] != "count" and prefix == CATEGORY_PREFIX["violence"]
        return f"{prefix}_{'fw' if fw else 'n'}_{conflict['ring_main_km']}"
    if conflict["method_main"] == "density":
        return f"{prefix}_kde_{conflict['density_main_km']}"
    raise ValueError(f"Unknown conflict.method_main: {conflict['method_main']}")


def add_rank(scores, prefix, settings, log):
    """Rank chokepoints on the main field (1 = most exposed; ties share the higher rank)."""
    value_field, rank_field = main_field(settings, prefix), f"{prefix}_rank"
    scores[rank_field] = scores[value_field].rank(ascending=False, method="min").astype("int64")
    log.info("Main ranking field: %s -> %s (1 = most exposed; ties share the higher rank)",
             value_field, rank_field)
    return value_field, rank_field


def log_rank_agreement(scores, settings, log):
    """Log the Spearman correlation between the violence and tension main fields."""
    viol, tens = main_field(settings, "viol"), main_field(settings, "tens")
    rho, p = spearmanr(scores[viol], scores[tens])
    log.info("Tension-violence rank agreement: Spearman rho (%s, %s) = %.3f, p = %.2g, n = %d",
             viol, tens, rho, p, len(scores))


def merge_scores(s07, settings, log):
    """Replace S07's columns in chokepoint_scores and keep every other script's columns."""
    if common.layer_name(settings, "chokepoint_scores") not in common.list_layers(settings):
        log.info("chokepoint_scores does not exist yet; creating it")
        return s07
    old = pd.DataFrame(common.read_layer(settings, "chokepoint_scores"))
    old = old.drop(columns=[c for c in old.columns if c == "geometry"])
    keep = [c for c in old.columns if c not in ("chokepoint_id", "name")
            and not c.startswith(S07_PREFIXES)]
    log.info("chokepoint_scores exists (%d rows); replacing %d S07 columns, keeping %d others: %s",
             len(old), len(old.columns) - len(keep) - 2, len(keep), keep or "none")
    dropped = sorted(set(old["chokepoint_id"]) - set(s07["chokepoint_id"]))
    if dropped:
        log.warning("Rows for chokepoints no longer included are removed: %s", dropped)
    return s07.merge(old[["chokepoint_id"] + keep], on="chokepoint_id", how="left")


def log_ranking(scores, prefix, rank_field, value_field, log):
    """Log the per-chokepoint summary for one category, sorted by its main rank."""
    cols = [c for c in scores.columns if c.startswith(f"{prefix}_") and c != rank_field]
    log.info("Per-chokepoint summary, ranked by %s:", value_field)
    for _, row in scores.sort_values([rank_field, "name"]).iterrows():
        log.info("  %2d  %-20s %s", row[rank_field], row["name"], "  ".join(
            f"{c}={row[c]:.3g}" if "_kde_" in c else f"{c}={int(row[c])}" for c in cols))


def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    tables = common.get_path(settings, "tables")
    tables.mkdir(parents=True, exist_ok=True)

    chokepoints = read_chokepoints(settings, log)
    events = read_events(settings, log)
    scores = chokepoints[["chokepoint_id", "name"]].copy()
    method_rows, overlap_rows = [], []

    # Violence: ACLED and IMO GISIS, one event per row (Decision 9 rule applied first).
    violence = violence_events(events, settings, log)
    distances = event_distances(violence, chokepoints, settings, log)
    log_gisis_rings(violence, distances, chokepoints, settings, log)
    weights = np.ones(len(violence))
    fatalities = violence["fatalities"].fillna(0).to_numpy(dtype=float)
    scores = scores.assign(**ring_exposure(distances, weights, fatalities, "viol", settings, log))
    scores = scores.assign(**density_exposure(distances, weights, "viol", settings, log))
    value_field, rank_field = add_rank(scores, "viol", settings, log)
    method_rows += compare_methods(scores, "viol", settings, log)
    overlap_rows += ring_overlap(distances, weights, chokepoints, "viol", settings, log)
    log_ranking(scores, "viol", rank_field, value_field, log)
    del distances

    # Tension: GDELT rows weighted by event_count (Decisions 5 and 15).
    log.info("Task 4.8 | tension: no fatality-weighted variant (GDELT records no fatalities); "
             "every count, density and overlap is weighted by event_count")
    variants = tension_variants(events, settings, log)
    del events
    for prefix, rows in variants.items():
        distances = event_distances(rows, chokepoints, settings, log)
        weights = rows["event_count"].to_numpy(dtype=float)
        scores = scores.assign(**ring_exposure(distances, weights, None, prefix, settings, log))
        if prefix != "tens":
            continue    # S10 alternatives: ring counts only
        scores = scores.assign(**density_exposure(distances, weights, prefix, settings, log))
        value_field, rank_field = add_rank(scores, prefix, settings, log)
        method_rows += compare_methods(scores, prefix, settings, log)
        overlap_rows += ring_overlap(distances, weights, chokepoints, prefix, settings, log)
        log_tension_context(rows, distances, chokepoints, settings, log)
    log_ranking(scores, "tens", "tens_rank", value_field, log)
    log_rank_agreement(scores, settings, log)

    methods = pd.DataFrame(method_rows)
    methods.to_csv(tables / "conflict_methods.csv", index=False)
    log.info("Wrote %s (%d rows)", tables / "conflict_methods.csv", len(methods))
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(tables / "ring_overlap.csv", index=False)
    log.info("Wrote %s (%d rows)", tables / "ring_overlap.csv", len(overlap))
    scores.to_csv(tables / "conflict_exposure.csv", index=False)
    log.info("Wrote %s (%d rows)", tables / "conflict_exposure.csv", len(scores))

    common.write_layer(merge_scores(scores, settings, log), settings, "chokepoint_scores", log)
    log.info("Run of s07_conflict_exposure finished")


if __name__ == "__main__":
    main()

"""S05: unified event layer (tasks 2.14, 2.15 and 2.18).

Reads the ACLED chunks (one CSV per chokepoint query, paths.raw_files.acled) and the
IMO GISIS piracy and armed-robbery export (paths.raw_files.imo_gisis; Decision 9), and
writes the events layer to the project database. ACLED steps, each logged with the
count kept and removed:

1. Read all chunks.
2. Remove repeated events (the same event ID in more than one chunk, where query
   areas overlap).
3. Keep events inside events.window_start to window_end (CLAUDE.md decision 7).
4. Keep events within events.prefilter_radius_km of any candidate chokepoint
   (great-circle distance, task 2.18).
5. Keep the event types in events.acled.event_types (task 2.14).
6. Remove events whose geo precision is in events.acled.exclude_geo_precision.

IMO GISIS steps (task 2.15):

G1. Read the date, area, coordinates and the two crew texts; give each incident an ID.
G2. Keep incidents inside the event window.
G3. Flag crew_harm (crew threatened, injured or held) from keywords in the settings,
    then apply the hand-checked overrides. Every incident is kept; S07 applies
    events.imo_gisis.count_rule.
G4. Convert coordinates from degrees and decimal minutes; remove incidents without both.
G5. Keep incidents within the prefilter radius (as step 4).
G6. An incident within 50 km and 1 day of a kept ACLED event is the same event: the
    ACLED record is kept and flagged imo_match, and the incident is removed.

GDELT steps (tasks 2.16-2.18, Decision 15). The filters, duplicate merging and outlet
counts ran in BigQuery (scripts/sql/); S05 reads the results and checks them:

T1. Read Query A (tension points) with fixed types and an exact column check.
T2. Re-check the window, root codes and location types; stop on any error.
T3. Re-check the prefilter distance (as step 4; nothing should be removed).
T4. Log the Query C filter counts (reports kept and removed by each filter).
T5. Convert FIPS country codes to ISO-3 (Natural Earth FIPS_10 plus the fixes in settings).
T6. Log the source-count checks and the rows per chokepoint within the main ring.
T7. Build the GDELT rows of the events layer: one row per Query A row, which holds
    event_count merged events at one place, day, root code, flags and source buckets.
T8. Write Query B as the gdelt_country_context table (task 2.17; context only).

Only the sources listed in events.sources are processed. The log holds counts only, never
individual event or incident rows, and no ship name, IMO number or narrative text is stored.
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

import common

KNOWN_SOURCES = ["acled", "imo_gisis", "gdelt"]

# Column names in the ACLED API export (checked against the raw chunks).
ACLED_COLUMNS = {
    "event_id_cnty": "event_id",
    "event_date": "event_date",
    "event_type": "event_type",
    "sub_event_type": "sub_event_type",
    "fatalities": "fatalities",
    "geo_precision": "geo_precision",
    "iso": "iso_numeric",
    "country": "country",
    "latitude": "latitude",
    "longitude": "longitude",
}

# IMO GISIS export (checked against the raw file): one sheet; only these columns are read.
# Ship names, IMO numbers and the other narrative columns are never read or stored.
GISIS_SHEET = "FullList"
GISIS_COLUMNS = {
    "Date": "event_date",
    "Area": "area",
    "Latitude": "lat_text",
    "Longitude": "lon_text",
    "Consequences for crew etc": "consequences",   # read only to set crew_harm
    "Incident details": "details",                 # read only to set crew_harm
}
GISIS_EVENT_TYPE = "Piracy and armed robbery"
# Degrees and decimal minutes, for example "1° 04.73' N".
DEGREES_MINUTES = re.compile(r"^\s*(\d+)\s*°\s*(\d+(?:\.\d+)?)\s*'\s*([NSEW])\s*$")

# GDELT BigQuery exports (checked against the raw files): exact columns and their types.
# The flags are written as TRUE/FALSE and are read as text, then converted.
GDELT_POINT_COLUMNS = {
    "lat": "float64", "lon": "float64", "feature_id": "string", "country_fips": "string",
    "location_type": "int64", "event_date": "string", "root_code": "int64",
    "international": "string", "state_actor": "string", "outlet_bucket": "int64",
    "numsources_bucket": "int64", "numsources_sum_bucket": "int64", "event_count": "int64",
    "total_outlets": "int64", "total_numsources": "int64",
}
GDELT_COUNTRY_COLUMNS = {
    "country_fips": "string", "root_code": "int64", "month": "int64", "outlet_bucket": "int64",
    "numsources_bucket": "int64", "numsources_sum_bucket": "int64", "event_count": "int64",
    "total_outlets": "int64", "total_numsources": "int64",
}
GDELT_FILTER_COLUMNS = {
    "location_type": "int64", "rows_in_window": "int64", "rows_root_10_17": "int64",
    "rows_root_10_17_within_500km": "int64",
}
GDELT_FLAGS = {"TRUE": True, "FALSE": False}
BUCKET_FIELDS = ["outlet_bucket", "numsources_bucket", "numsources_sum_bucket"]
# Query A's grouping key: sorting on it gives the event IDs a fixed order.
GDELT_SORT = ["event_date", "root_code", "country_fips", "feature_id", "latitude", "longitude",
              "international", "state_actor"] + BUCKET_FIELDS

# imo_match: ACLED events also reported to IMO (null for other rows).
# crew_harm: GISIS incidents where crew were threatened, injured or held (null for other rows).
# event_count: events a row stands for (1 for ACLED and GISIS; GDELT rows merge several).
# gdelt_sources: distinct news outlets, bucket 0/1/2/3 (3 = 3 or more); the main source count.
# numsources_bucket, numsources_sum_bucket: GDELT's NumSources (largest, and summed over
# merged reports), same buckets, kept for comparison. international, state_actor: GDELT flags.
GDELT_FIELDS = ["gdelt_sources", "numsources_bucket", "numsources_sum_bucket",
                "international", "state_actor"]
OUTPUT_FIELDS = ["event_id", "source", "category", "event_type", "sub_event_type",
                 "event_date", "fatalities", "geo_precision", "country_iso3",
                 "imo_match", "crew_harm", "event_count"] + GDELT_FIELDS
NO_GDELT_FIELDS = dict.fromkeys(GDELT_FIELDS, pd.NA)


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------

def read_acled_chunks(settings, log):
    """Step 1: read every ACLED chunk and log rows per chunk and in total."""
    folder = common.get_path(settings, "raw_files.acled")
    files = sorted(folder.glob("*.csv"), key=lambda p: (len(p.stem), p.stem))
    if not files:
        raise FileNotFoundError(f"No ACLED CSV files in {folder}")
    log.info("Task 2.14 | reading %d ACLED chunks from %s", len(files), folder)

    chunks = []
    for path in files:
        chunk = pd.read_csv(path, usecols=list(ACLED_COLUMNS), dtype={"event_id_cnty": str})
        log.info("Task 2.14 | chunk %-45s %8s rows", path.name, f"{len(chunk):,}")
        chunks.append(chunk)
    events = pd.concat(chunks, ignore_index=True).rename(columns=ACLED_COLUMNS)
    log.info("Task 2.14 | ACLED rows read in total: %s", f"{len(events):,}")
    return events


def remove_repeats(events, log):
    """Step 2: keep one row per event ID (query areas overlap, so chunks share events)."""
    before = len(events)
    distinct_rows = len(events.drop_duplicates())
    unique = events.drop_duplicates("event_id", keep="first").reset_index(drop=True)
    common.log_filter(log, "Task 2.14 | repeated event IDs", len(unique), before - len(unique),
                      "same event ID in more than one chunk")
    if distinct_rows != len(unique):
        log.warning("Task 2.14 | %s event IDs have rows that differ between chunks; first kept",
                    f"{distinct_rows - len(unique):,}")
    return unique


# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------

def filter_window(events, settings, log, task="Task 2.14"):
    """Step 3: keep events inside the event window, both dates inclusive (decision 7)."""
    start = pd.Timestamp(settings["events"]["window_start"])
    end = pd.Timestamp(settings["events"]["window_end"])
    events["event_date"] = pd.to_datetime(events["event_date"], format="%Y-%m-%d")
    inside = events["event_date"].between(start, end)
    log.info("%s | event dates in data: %s to %s", task, events["event_date"].min().date(),
             events["event_date"].max().date())
    common.log_filter(log, f"{task} | event window {start.date()} to {end.date()}",
                      int(inside.sum()), int((~inside).sum()), "event date outside window")
    return events[inside].reset_index(drop=True)


def filter_distance(events, settings, log, source="ACLED"):
    """Step 4 (task 2.18): keep events within the prefilter radius of any candidate chokepoint."""
    radius = settings["events"]["prefilter_radius_km"]
    chokepoints = common.read_layer(settings, "chokepoints")
    log.info("Task 2.18 | %s: distance to %d candidate chokepoints (all points in layer '%s', "
             "included or not)", source, len(chokepoints), common.layer_name(settings, "chokepoints"))

    bad = ~(events["latitude"].between(-90, 90) & events["longitude"].between(-180, 180))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} {source} events have coordinates outside the valid range")

    distances = common.distance_matrix_km(events["longitude"], events["latitude"],
                                          chokepoints.geometry.x, chokepoints.geometry.y, settings)
    events["nearest_km"] = distances.min(axis=1)
    near = events["nearest_km"] <= radius
    common.log_filter(log, f"Task 2.18 | within {radius} km of a chokepoint", int(near.sum()),
                      int((~near).sum()), f"farther than {radius} km from every candidate chokepoint")
    return events[near].reset_index(drop=True)


def compare_with_old_count(events, log):
    """Log the step-4 count against the unique-event count from the download script."""
    old = 136_791   # data/interim/acled_derived unique events file (MANIFEST.csv notes)
    log.info("Task 2.18 | check: %s events within radius here vs %s in the old derived file "
             "(difference %+d)", f"{len(events):,}", f"{old:,}", len(events) - old)


def log_type_counts(events, label, log):
    """Log the number of events of each type (counts only)."""
    counts = events["event_type"].value_counts()
    log.info("Task 2.14 | events by type %s:", label)
    for event_type, n in counts.items():
        log.info("Task 2.14 |   %-30s %8s", event_type, f"{n:,}")


def filter_event_types(events, settings, log):
    """Step 5 (task 2.14): keep the violent event types."""
    keep_types = settings["events"]["acled"]["event_types"]
    unknown = sorted(set(keep_types) - set(events["event_type"]))
    if unknown:
        log.warning("Task 2.14 | event types in settings not found in data: %s", unknown)

    log_type_counts(events, "before filter", log)
    keep = events["event_type"].isin(keep_types)
    common.log_filter(log, "Task 2.14 | event types", int(keep.sum()), int((~keep).sum()),
                      "event type not in events.acled.event_types")
    kept = events[keep].reset_index(drop=True)
    log_type_counts(kept, "after filter", log)
    return kept


def filter_geo_precision(events, settings, log):
    """Step 6 (task 2.14): remove region-level locations; log removals by country."""
    exclude = settings["events"]["acled"]["exclude_geo_precision"]
    log.info("Task 2.14 | geo precision before filter: %s", ", ".join(
        f"{p}: {n:,}" for p, n in events["geo_precision"].value_counts().sort_index().items()))
    drop = events["geo_precision"].isin(exclude)
    common.log_filter(log, f"Task 2.14 | geo precision {exclude}", int((~drop).sum()),
                      int(drop.sum()), "location precise only to a region")

    by_country = events.loc[drop, "country"].value_counts().head(10)
    log.info("Task 2.14 | removed for geo precision, top 10 countries:")
    for country, n in by_country.items():
        log.info("Task 2.14 |   %-30s %8s", country, f"{n:,}")
    return events[~drop].reset_index(drop=True)


# ---------------------------------------------------------------------
# Country codes and output
# ---------------------------------------------------------------------

def iso_numeric_lookup(settings):
    """ISO numeric code -> ISO-3 letters, from Natural Earth.

    Uses the _EH columns, which fill codes Natural Earth leaves as -99 in ISO_N3
    (France and Norway). Several Natural Earth rows can share a code, but they share
    the same ISO-3 too, so one row per code is kept.
    """
    path = common.get_path(settings, "raw_files.natural_earth_countries")
    ne = pyogrio.read_dataframe(path, columns=["ISO_N3_EH", "ISO_A3_EH"], read_geometry=False)
    ne["iso_numeric"] = pd.to_numeric(ne["ISO_N3_EH"], errors="coerce")
    ne = ne[(ne["iso_numeric"] > 0) & (ne["ISO_A3_EH"] != "-99")]
    if ne.groupby("iso_numeric")["ISO_A3_EH"].nunique().max() > 1:
        raise ValueError("Natural Earth gives more than one ISO-3 code for one numeric code")
    return ne.drop_duplicates("iso_numeric").set_index("iso_numeric")["ISO_A3_EH"]


def add_iso3(events, settings, log):
    """ACLED gives numeric ISO codes; convert them to ISO-3 (the project's country key).

    ACLED's own codes for seas (1002-1005) and for Akrotiri and Dhekelia have no ISO-3
    and stay empty; they are counted here, not removed.
    """
    lookup = iso_numeric_lookup(settings)
    events["country_iso3"] = events["iso_numeric"].map(lookup).astype("string")
    missing = events["country_iso3"].isna()
    log.info("Task 2.14 | ISO-3 codes assigned to %s events; %s left empty", f"{int((~missing).sum()):,}",
             f"{int(missing.sum()):,}")
    for country, n in events.loc[missing, "country"].value_counts().items():
        log.info("Task 2.14 |   no ISO-3 for ACLED area '%s': %s events", country, f"{n:,}")
    return events


def build_layer(events, settings):
    """Build the events layer with the fixed output fields and point geometry."""
    events = events.assign(source="ACLED", category="violence", imo_match=False, crew_harm=pd.NA,
                           event_count=1, **NO_GDELT_FIELDS)
    events["event_date"] = events["event_date"].dt.date
    events["fatalities"] = events["fatalities"].astype("int32")
    events["geo_precision"] = events["geo_precision"].astype("int32")
    geometry = gpd.points_from_xy(events["longitude"], events["latitude"])
    return gpd.GeoDataFrame(events[OUTPUT_FIELDS], geometry=geometry,
                            crs=settings["project"]["storage_crs"])


def process_acled(settings, log):
    """Run steps 1-6 for ACLED and return the cleaned events."""
    events = read_acled_chunks(settings, log)
    events = remove_repeats(events, log)
    events = filter_window(events, settings, log)
    events = filter_distance(events, settings, log)
    compare_with_old_count(events, log)
    events = filter_event_types(events, settings, log)
    events = filter_geo_precision(events, settings, log)
    events = add_iso3(events, settings, log)
    log.info("Task 2.14 | ACLED events kept: %s (%s fatalities)", f"{len(events):,}",
             f"{int(events['fatalities'].sum()):,}")
    return build_layer(events, settings)


# ---------------------------------------------------------------------
# IMO GISIS piracy and armed-robbery incidents (task 2.15, Decision 9)
# ---------------------------------------------------------------------

def read_imo_gisis(settings, log):
    """Step G1: read the GISIS export (only the columns needed) and give each incident an ID.

    Ship names, IMO numbers and the other columns are never read. The ID is
    IMO-<yyyymmdd>-<n>, numbered in file order within each date, so it stays the
    same as long as the file does.
    """
    path = common.get_path(settings, "raw_files.imo_gisis")
    events = pd.read_excel(path, sheet_name=GISIS_SHEET, usecols=list(GISIS_COLUMNS), dtype=str)
    events = events.rename(columns=GISIS_COLUMNS)
    log.info("Task 2.15 | IMO GISIS incidents read from %s: %s", path.name, f"{len(events):,}")
    blank = events["event_date"].isna()
    common.log_filter(log, "Task 2.15 | no date", int((~blank).sum()), int(blank.sum()),
                      "row has no incident date")
    events = events[~blank].reset_index(drop=True)
    number = events.groupby("event_date").cumcount() + 1
    events["event_id"] = "IMO-" + events["event_date"].str.replace("-", "") + "-" + number.astype(str)
    return events


def degrees_minutes(text):
    """Convert one GISIS coordinate such as "1° 04.73' N" to decimal degrees (S and W negative).

    Returns None when the text is empty or not in degrees and decimal minutes.
    """
    if not isinstance(text, str):
        return None
    match = DEGREES_MINUTES.match(text)
    if not match or float(match.group(2)) >= 60:
        return None
    value = int(match.group(1)) + float(match.group(2)) / 60
    return -value if match.group(3) in "SW" else value


def add_coordinates(events, log):
    """Step G4: parse latitude and longitude; remove incidents without both."""
    events["latitude"] = events["lat_text"].map(degrees_minutes).astype(float)
    events["longitude"] = events["lon_text"].map(degrees_minutes).astype(float)
    has_text = events["lat_text"].notna() & events["lon_text"].notna()
    parsed = events["latitude"].notna() & events["longitude"].notna()
    log.info("Task 2.15 | coordinates: %d with both, %d with neither, %d with only one; "
             "%d with both that could not be parsed", int(has_text.sum()),
             int((events["lat_text"].isna() & events["lon_text"].isna()).sum()),
             int((events["lat_text"].isna() ^ events["lon_text"].isna()).sum()),
             int((has_text & ~parsed).sum()))
    common.log_filter(log, "Task 2.15 | coordinates", int(parsed.sum()), int((~parsed).sum()),
                      "latitude or longitude missing or not in degrees and decimal minutes")
    return events[parsed].reset_index(drop=True)


def crew_harm_patterns(settings):
    """Build the regular expressions for the crew-harm rule from the settings.

    Returns (negation, {group: keyword pattern}). Keywords match at the start of a
    word. A negation is a negation word followed, within negation_window_words words
    joined only by spaces (so it never reaches past a comma or full stop), by a keyword.
    """
    rule = settings["events"]["imo_gisis"]
    groups = {name: "|".join(re.escape(k) for k in words)
              for name, words in rule["crew_harm_keywords"].items()}
    negation_words = "|".join(re.escape(w) for w in rule["negation_words"])
    window = rule["negation_window_words"]
    negation = re.compile(rf"\b(?:{negation_words})\b(?:\s+\w+){{0,{window}}}?\s+\b(?:{'|'.join(groups.values())})\w*")
    keywords = {name: re.compile(rf"\b(?:{pattern})") for name, pattern in groups.items()}
    return negation, keywords


def crew_harm_reasons(text, negation, keywords):
    """Return the keyword groups (injured, held, threatened) found in one text after negations."""
    cleaned = negation.sub(" ", text.lower())
    return [name for name, pattern in keywords.items() if pattern.search(cleaned)]


def flag_crew_harm(events, settings, log):
    """Step G3: crew_harm = crew threatened, injured or held (the main counting rule).

    Every incident is kept; S07 applies events.imo_gisis.count_rule. The hand-checked
    overrides in the settings replace the keyword result for the incidents they name.
    """
    negation, keywords = crew_harm_patterns(settings)
    # A full stop between the two texts, so a negation never runs from one into the other.
    text = events["consequences"].fillna("") + " . " + events["details"].fillna("")
    events["harm_reasons"] = text.map(lambda t: crew_harm_reasons(t, negation, keywords))
    events["keyword_harm"] = events["harm_reasons"].map(bool)
    events["crew_harm"] = events["keyword_harm"]
    for name in keywords:
        log.info("Task 2.15 | crew %s (keywords): %d", name,
                 int(events["harm_reasons"].map(lambda r: name in r).sum()))
    log.info("Task 2.15 | crew threatened, injured or held (keywords): %d of %d",
             int(events["keyword_harm"].sum()), len(events))
    apply_overrides(events, settings, log)
    log.info("Task 2.15 | by area (all / crew_harm): %s", ", ".join(
        f"{a} {n} / {int(events.loc[events['area'] == a, 'crew_harm'].sum())}"
        for a, n in events["area"].value_counts().items()))
    return events


def apply_overrides(events, settings, log):
    """Replace the keyword result with the hand-checked overrides; one log line per override."""
    overrides = settings["events"]["imo_gisis"]["crew_harm_overrides"] or {}
    for event_id, override in overrides.items():
        if set(override) != {"crew_harm", "reason"} or not isinstance(override["crew_harm"], bool):
            raise ValueError(f"crew_harm override {event_id} needs crew_harm (true/false) and reason")
        row = events["event_id"] == event_id
        if not row.any():
            log.warning("Task 2.15 | crew_harm override %s: event ID not in the data", event_id)
            continue
        keyword = bool(events.loc[row, "keyword_harm"].iloc[0])
        events.loc[row, "crew_harm"] = override["crew_harm"]
        log.info("Task 2.15 | crew_harm override %s: keywords %s -> %s (%s)%s", event_id,
                 str(keyword).lower(), str(override["crew_harm"]).lower(), override["reason"],
                 "" if keyword != override["crew_harm"] else "  [same as keywords]")
    to_true = int((events["crew_harm"] & ~events["keyword_harm"]).sum())
    to_false = int((~events["crew_harm"] & events["keyword_harm"]).sum())
    log.info("Task 2.15 | crew_harm review: keyword flags %d, overrides to true %d, overrides to "
             "false %d, crew_harm after overrides %d of %d", int(events["keyword_harm"].sum()),
             to_true, to_false, int(events["crew_harm"].sum()), len(events))


def log_hand_check(incidents, settings, log):
    """Log the hand check in counts, for the incidents within the prefilter radius."""
    log.info("Task 2.15 | hand check %s, within the prefilter radius: keyword flags %d, "
             "crew_harm after overrides %d of %d; keyword-unflagged incidents searched by hand %d",
             settings["events"]["imo_gisis"]["crew_harm_hand_check"],
             int(incidents["keyword_harm"].sum()), int(incidents["crew_harm"].sum()),
             len(incidents), int((~incidents["keyword_harm"]).sum()))


def match_acled(incidents, acled, settings, log):
    """Step G6: an incident within match_acled_km and match_acled_days of a kept ACLED event
    is the same event. The ACLED record is kept and flagged imo_match; the incident is removed.
    """
    rule = settings["events"]["imo_gisis"]
    max_km, max_days = rule["match_acled_km"], rule["match_acled_days"]
    acled_dates = pd.to_datetime(acled["event_date"])
    matched = pd.Series(False, index=incidents.index)
    for i, incident in incidents.iterrows():
        close_in_time = (acled_dates - incident["event_date"]).abs() <= pd.Timedelta(days=max_days)
        candidates = acled[close_in_time]
        km = common.great_circle_km(incident["longitude"], incident["latitude"],
                                    candidates.geometry.x, candidates.geometry.y, settings)
        hits = candidates.index[np.atleast_1d(km) <= max_km]
        if len(hits):
            matched[i] = True
            acled.loc[hits, "imo_match"] = True
    log.info("Task 2.15 | ACLED match rule: within %s km and dates at most %s day(s) apart, "
             "against the %s cleaned ACLED events", max_km, max_days, f"{len(acled):,}")
    common.log_filter(log, "Task 2.15 | same event as ACLED", int((~matched).sum()), int(matched.sum()),
                      "matches a kept ACLED event; the ACLED record is kept")
    log.info("Task 2.15 | matched incidents with crew_harm: %d; ACLED events flagged imo_match: %d",
             int(incidents.loc[matched, "crew_harm"].sum()), int(acled["imo_match"].sum()))
    return incidents[~matched].reset_index(drop=True), acled


def build_gisis_layer(incidents, settings):
    """Build the GISIS part of the events layer with the same fields as ACLED."""
    incidents = incidents.assign(source="IMO GISIS", category="violence",
                                 event_type=GISIS_EVENT_TYPE, sub_event_type=incidents["area"],
                                 fatalities=0, geo_precision=1, country_iso3=pd.NA, imo_match=pd.NA,
                                 event_count=1, **NO_GDELT_FIELDS)
    incidents["event_date"] = incidents["event_date"].dt.date
    incidents["fatalities"] = incidents["fatalities"].astype("int32")
    incidents["geo_precision"] = incidents["geo_precision"].astype("int32")
    geometry = gpd.points_from_xy(incidents["longitude"], incidents["latitude"])
    return gpd.GeoDataFrame(incidents[OUTPUT_FIELDS], geometry=geometry,
                            crs=settings["project"]["storage_crs"])


def process_imo_gisis(settings, log, acled):
    """Run steps G1-G6 for IMO GISIS; return the incidents layer and the flagged ACLED layer."""
    incidents = read_imo_gisis(settings, log)
    incidents = filter_window(incidents, settings, log, task="Task 2.15")      # step G2
    incidents = flag_crew_harm(incidents, settings, log)
    incidents = add_coordinates(incidents, log)
    incidents = filter_distance(incidents, settings, log, source="IMO GISIS")  # step G5
    log_hand_check(incidents, settings, log)
    log.info("Task 2.15 | within the prefilter radius, by area (all / crew_harm): %s", ", ".join(
        f"{a} {n} / {int(incidents.loc[incidents['area'] == a, 'crew_harm'].sum())}"
        for a, n in incidents["area"].value_counts().items()))
    incidents, acled = match_acled(incidents, acled, settings, log)
    log.info("Task 2.15 | IMO GISIS incidents kept: %d (crew_harm %d, other %d)", len(incidents),
             int(incidents["crew_harm"].sum()), int((~incidents["crew_harm"]).sum()))
    return build_gisis_layer(incidents, settings), acled


# ---------------------------------------------------------------------
# GDELT tension events (tasks 2.16-2.18, Decision 15)
# ---------------------------------------------------------------------

def read_gdelt_csv(settings, key, columns, log, nrows=None):
    """Read one GDELT BigQuery export with fixed types; stop unless the columns match exactly.

    nrows reads only the first rows (for a sample test); the row-count check is then skipped.
    """
    path = common.get_path(settings, f"raw_files.{key}")
    header = list(pd.read_csv(path, nrows=0).columns)
    if header != list(columns):
        raise ValueError(f"{path.name}: columns {header} differ from the expected {list(columns)}")
    table = pd.read_csv(path, dtype=columns, keep_default_na=False, na_values=[""], nrows=nrows)
    expected = settings["events"]["gdelt"]["expected_rows"][key.removeprefix("gdelt_")]
    if nrows is None and len(table) != expected:
        raise ValueError(f"{path.name}: {len(table):,} rows, but the BigQuery job returned {expected:,}")
    blank = table.isna().any(axis=1)
    if blank.any():
        raise ValueError(f"{path.name}: {int(blank.sum())} rows with an empty cell")
    log.info("Task 2.16 | read %s: %s rows (%s)%s", path.name, f"{len(table):,}",
             f"expected {expected:,}" if nrows is None else f"sample of the first {nrows:,}",
             "" if nrows is not None else ", matches the BigQuery job")
    return table


def check_gdelt_rule(table, ok, task, check, log):
    """Stop with an error if any row breaks a rule; otherwise log that every row passed."""
    bad = int((~ok).sum())
    if bad:
        raise ValueError(f"{task} | {check}: {bad:,} rows break the rule")
    common.log_filter(log, f"{task} | re-check {check}", len(table), 0, "none (every row passes)")


def check_gdelt_points(points, settings, log):
    """Step T2: every row inside the window, a main root code, a city location, valid buckets."""
    cfg = settings["events"]["gdelt"]
    for flag in ["international", "state_actor"]:
        if not points[flag].isin(list(GDELT_FLAGS)).all():
            raise ValueError(f"GDELT {flag} holds values other than TRUE/FALSE")
        points[flag] = points[flag].map(GDELT_FLAGS).astype(bool)
    points["event_date"] = pd.to_datetime(points["event_date"], format="%Y%m%d")
    start = pd.Timestamp(settings["events"]["window_start"])
    end = pd.Timestamp(settings["events"]["window_end"])
    log.info("Task 2.16 | event dates in data: %s to %s", points["event_date"].min().date(),
             points["event_date"].max().date())
    check_gdelt_rule(points, points["event_date"].between(start, end), "Task 2.16",
                     f"event window {start.date()} to {end.date()}", log)
    roots = cfg["main_filter"]["root_codes"]
    check_gdelt_rule(points, points["root_code"].isin(roots), "Task 2.16", f"root codes {roots}", log)
    types = cfg["keep_location_types"]
    check_gdelt_rule(points, points["location_type"].isin(types), "Task 2.16",
                     f"location types {types}", log)
    ok = (points["outlet_bucket"].between(0, 3) & points["numsources_bucket"].between(1, 3)
          & points["numsources_sum_bucket"].between(1, 3) & (points["event_count"] >= 1))
    check_gdelt_rule(points, ok, "Task 2.16", "source buckets (outlets 0-3, NumSources 1-3) and "
                     "event_count >= 1", log)
    return points.rename(columns={"lat": "latitude", "lon": "longitude"})


def log_gdelt_filter_counts(counts, points, settings, log):
    """Step T4: the reports each BigQuery filter kept and removed (Query C), then the merge."""
    types = settings["events"]["gdelt"]["keep_location_types"]
    roots = settings["events"]["gdelt"]["main_filter"]["root_codes"]
    radius = settings["events"]["prefilter_radius_km"]
    city = counts["location_type"].isin(types)
    in_window = int(counts["rows_in_window"].sum())
    in_roots = int(counts["rows_root_10_17"].sum())
    in_city = int(counts.loc[city, "rows_root_10_17"].sum())
    near = int(counts.loc[city, "rows_root_10_17_within_500km"].sum())
    merged = int(points["event_count"].sum())
    log.info("Task 2.16 | Query C (reports before duplicates are merged): %s reports in the "
             "event window", f"{in_window:,}")
    common.log_filter(log, f"Task 2.16 | root codes {roots}", in_roots, in_window - in_roots,
                      "CAMEO root code outside the main filter")
    common.log_filter(log, f"Task 2.16 | location types {types}", in_city, in_roots - in_city,
                      "located only to a country or state centroid (types 1, 2, 5) or not located (0)")
    log.info("Task 2.16 |   location types among root-code reports: %s", ", ".join(
        f"{t}: {n:,}" for t, n in zip(counts["location_type"], counts["rows_root_10_17"])))
    common.log_filter(log, f"Task 2.18 | within {radius} km of a chokepoint (BigQuery)", near,
                      in_city - near, f"farther than {radius} km from every candidate chokepoint")
    common.log_filter(log, "Task 2.16 | duplicate reports merged (Query A event_count)", merged,
                      near - merged, "same date, root code, actors and location as a report "
                      "with more sources")


def fips_iso3_lookup(settings):
    """FIPS 10-4 code -> ISO-3, from Natural Earth FIPS_10 and ISO_A3_EH, then the settings fixes.

    Natural Earth rows with -99 in either field are skipped. A FIPS code that Natural Earth
    gives to two countries must be listed in events.gdelt.fips_no_iso3, where it stays empty.
    """
    cfg = settings["events"]["gdelt"]
    path = common.get_path(settings, "raw_files.natural_earth_countries")
    ne = pyogrio.read_dataframe(path, columns=["FIPS_10", "ISO_A3_EH"], read_geometry=False)
    ne = ne[(ne["FIPS_10"] != "-99") & (ne["ISO_A3_EH"] != "-99")].drop_duplicates()
    shared = set(ne.loc[ne["FIPS_10"].duplicated(), "FIPS_10"])
    unlisted = shared - set(cfg["fips_no_iso3"])
    if unlisted:
        raise ValueError(f"FIPS codes with more than one ISO-3 in Natural Earth: {sorted(unlisted)}; "
                         "add them to fips_iso3_fixes or fips_no_iso3")
    lookup = ne[~ne["FIPS_10"].isin(shared)].set_index("FIPS_10")["ISO_A3_EH"].to_dict()
    lookup.update(cfg["fips_iso3_fixes"])
    for code in cfg["fips_no_iso3"]:
        lookup.pop(code, None)
    return lookup


def add_gdelt_iso3(table, lookup, settings, task, label, log):
    """Step T5: ISO-3 from the FIPS code; log the rows and events left empty, by code."""
    table["country_iso3"] = table["country_fips"].map(lookup).astype("string")
    missing = table["country_iso3"].isna()
    fixed = table["country_fips"].isin(list(settings["events"]["gdelt"]["fips_iso3_fixes"]))
    log.info("%s | %s: ISO-3 assigned to %s rows (%s through the fix list); %s rows "
             "(%s events) left empty", task, label, f"{int((~missing).sum()):,}", f"{int(fixed.sum()):,}",
             f"{int(missing.sum()):,}", f"{int(table.loc[missing, 'event_count'].sum()):,}")
    expected = set(settings["events"]["gdelt"]["fips_no_iso3"])
    by_code = table[missing].groupby("country_fips")["event_count"].agg(["size", "sum"])
    for code, row in by_code.sort_values("sum", ascending=False).iterrows():
        note = "" if code in expected else "  [not in fips_no_iso3: add it to the fixes or the list]"
        (log.info if not note else log.warning)(
            "%s |   no ISO-3 for FIPS %s: %s rows, %s events%s", task, code, f"{row['size']:,}",
            f"{row['sum']:,}", note)
    return table


def log_source_checks(points, log):
    """Step T6: outlet bucket 0, and how often the outlet count agrees with NumSources."""
    total_rows, total_events = len(points), int(points["event_count"].sum())
    zero = points["outlet_bucket"] == 0
    log.info("Task 2.16 | events in Query A rows: %s (sum of event_count over %s rows)",
             f"{total_events:,}", f"{total_rows:,}")
    log.info("Task 2.16 | outlet bucket 0 (no mention found within the mention window): %s rows, "
             "%s events", f"{int(zero.sum()):,}", f"{int(points.loc[zero, 'event_count'].sum()):,}")
    for field in BUCKET_FIELDS:
        rows = points.groupby(field).size()
        events = points.groupby(field)["event_count"].sum()
        log.info("Task 2.16 | %-21s rows / events by bucket: %s", field, ", ".join(
            f"{b}: {rows[b]:,} / {events[b]:,}" for b in rows.index))
    for field in BUCKET_FIELDS[1:]:
        same = points["outlet_bucket"] == points[field]
        at_two = (points["outlet_bucket"] >= 2) == (points[field] >= 2)
        log.info("Task 2.16 | outlet bucket vs %s: same bucket %.1f%% of rows (%.1f%% of events); "
                 "same side of 2+ sources %.1f%% of rows (%.1f%% of events)", field,
                 100 * same.mean(), 100 * points.loc[same, "event_count"].sum() / total_events,
                 100 * at_two.mean(), 100 * points.loc[at_two, "event_count"].sum() / total_events)
    table = pd.crosstab(points["outlet_bucket"], points["numsources_bucket"],
                        values=points["event_count"], aggfunc="sum").fillna(0).astype(int)
    log.info("Task 2.16 | events by outlet bucket (rows) and numsources_bucket (columns %s):",
             ", ".join(map(str, table.columns)))
    for bucket, row in table.iterrows():
        log.info("Task 2.16 |   outlets %d: %s", bucket, "  ".join(f"{n:>9,}" for n in row))


def log_gdelt_rings(points, settings, log):
    """Step T6: per chokepoint, GDELT rows and events within the main ring at the main source
    threshold, and the share of those events at the single most common location (feature ID)."""
    ring = settings["conflict"]["ring_main_km"]
    cfg = settings["events"]["gdelt"]["main_filter"]
    chokepoints = common.read_layer(settings, "chokepoints")
    kept = points[points["outlet_bucket"] >= cfg["min_sources"]].reset_index(drop=True)
    km = common.distance_matrix_km(kept["longitude"], kept["latitude"],
                                   chokepoints.geometry.x, chokepoints.geometry.y, settings)
    log.info("Task 2.18 | GDELT rows at %d+ outlets (any distance): %s rows, %s events", cfg["min_sources"],
             f"{len(kept):,}", f"{int(kept['event_count'].sum()):,}")
    log.info("Task 2.18 | GDELT within %d km of each chokepoint at %d+ outlets; top location = the "
             "feature ID with the most events, its share of the events and its country:",
             ring, cfg["min_sources"])
    log.info("Task 2.18 |   %-21s %-8s %8s %9s  %s", "chokepoint", "included", "rows", "events",
             "top location")
    for i, row in chokepoints.iterrows():
        near = kept[km[:, i] <= ring]
        events = int(near["event_count"].sum())
        top = ""
        if events:
            by_place = near.groupby(["feature_id", "country_iso3"], dropna=False)["event_count"].sum()
            (feature, iso3), n = by_place.idxmax(), by_place.max()
            top = f"{feature} ({iso3 if pd.notna(iso3) else 'no ISO-3'}) {100 * n / events:.0f}%"
        log.info("Task 2.18 |   %-21s %-8s %8s %9s  %s", row["name"], row["included"],
                 f"{len(near):,}", f"{events:,}", top)


def build_gdelt_layer(points, settings):
    """Step T7: GDELT rows of the events layer, with IDs GDELT-<yyyymmdd>-<n> in a fixed order."""
    labels = settings["events"]["gdelt"]["root_labels"]
    points = points.sort_values(GDELT_SORT, kind="stable").reset_index(drop=True)
    day = points["event_date"].dt.strftime("%Y%m%d")
    points["event_id"] = "GDELT-" + day + "-" + (points.groupby(day).cumcount() + 1).astype(str)
    points = points.assign(source="GDELT", category="tension",
                           event_type=points["root_code"].map(labels), sub_event_type=pd.NA,
                           fatalities=0, geo_precision=points["location_type"],
                           imo_match=pd.NA, crew_harm=pd.NA, gdelt_sources=points["outlet_bucket"])
    if points["event_type"].isna().any():
        raise ValueError("GDELT root codes without a label in events.gdelt.root_labels")
    points["event_date"] = points["event_date"].dt.date
    for field in ["fatalities", "geo_precision", "event_count"]:
        points[field] = points[field].astype("int32")
    geometry = gpd.points_from_xy(points["longitude"], points["latitude"])
    return gpd.GeoDataFrame(points[OUTPUT_FIELDS], geometry=geometry,
                            crs=settings["project"]["storage_crs"])


def build_country_context(settings, lookup, log):
    """Step T8 (task 2.17): GDELT tension totals per country, root code and month (Query B),
    before the distance filter. Context only; never part of the score."""
    table = read_gdelt_csv(settings, "gdelt_country", GDELT_COUNTRY_COLUMNS, log)
    cfg = settings["events"]["gdelt"]
    start = pd.Timestamp(settings["events"]["window_start"])
    end = pd.Timestamp(settings["events"]["window_end"])
    months = pd.to_datetime(table["month"].astype(str), format="%Y%m")
    check_gdelt_rule(table, months.between(start.replace(day=1), end), "Task 2.17",
                     f"months {start:%Y-%m} to {end:%Y-%m}", log)
    check_gdelt_rule(table, table["root_code"].isin(cfg["main_filter"]["root_codes"]), "Task 2.17",
                     "root codes", log)
    table = add_gdelt_iso3(table, lookup, settings, "Task 2.17", "country context", log)
    table.insert(2, "event_type", table["root_code"].map(cfg["root_labels"]))
    log.info("Task 2.17 | country context: %s events in %s countries (FIPS codes); %s at %d+ outlets",
             f"{int(table['event_count'].sum()):,}", table["country_fips"].nunique(),
             f"{int(table.loc[table['outlet_bucket'] >= cfg['main_filter']['min_sources'], 'event_count'].sum()):,}",
             cfg["main_filter"]["min_sources"])
    order = ["country_fips", "country_iso3"] + [c for c in table.columns
                                                if c not in ("country_fips", "country_iso3")]
    return table[order]


def process_gdelt(settings, log, nrows=None):
    """Run steps T1-T8 for GDELT; return the events rows and the country context table."""
    points = read_gdelt_csv(settings, "gdelt_points", GDELT_POINT_COLUMNS, log, nrows)
    points = check_gdelt_points(points, settings, log)
    before = len(points)
    points = filter_distance(points, settings, log, source="GDELT")                  # step T3
    if len(points) != before:
        log.warning("Task 2.18 | GDELT: %d rows beyond the radius here although BigQuery kept them",
                    before - len(points))
    counts = read_gdelt_csv(settings, "gdelt_filter_counts", GDELT_FILTER_COLUMNS, log)
    log_gdelt_filter_counts(counts, points, settings, log)
    lookup = fips_iso3_lookup(settings)
    points = add_gdelt_iso3(points, lookup, settings, "Task 2.16", "tension points", log)
    log_source_checks(points, log)
    log_gdelt_rings(points, settings, log)
    log.info("Task 2.16 | GDELT rows kept: %s (%s events; international %s rows, state_actor %s rows)",
             f"{len(points):,}", f"{int(points['event_count'].sum()):,}",
             f"{int(points['international'].sum()):,}", f"{int(points['state_actor'].sum()):,}")
    context = build_country_context(settings, lookup, log)
    return build_gdelt_layer(points, settings), context


def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)

    sources = settings["events"]["sources"]
    unknown = sorted(set(sources) - set(KNOWN_SOURCES))
    if unknown:
        raise ValueError(f"Unknown event sources in events.sources: {unknown}")
    for source in KNOWN_SOURCES:
        if source not in sources:
            log.info("Source %s: skipped (not in events.sources)", source)

    layers = []
    if "acled" in sources:
        acled = process_acled(settings, log)
        if "imo_gisis" in sources:
            gisis, acled = process_imo_gisis(settings, log, acled)
            layers.append(gisis)
        layers.insert(0, acled)
    elif "imo_gisis" in sources:
        raise ValueError("imo_gisis needs acled in events.sources (incidents are matched to ACLED)")
    context = None
    if "gdelt" in sources:
        gdelt, context = process_gdelt(settings, log)
        layers.append(gdelt)
    if not layers:
        raise ValueError("events.sources lists no source that S05 can process")

    events = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True),
                              crs=settings["project"]["storage_crs"])
    # Joining the sources mixes column types; set them again so empty values are written
    # as nulls (an object column would write the text "<NA>").
    for field in ["country_iso3", "sub_event_type"]:
        events[field] = events[field].astype("string")
    for flag in ["imo_match", "crew_harm", "international", "state_actor"]:
        events[flag] = events[flag].astype("boolean")   # true/false, empty where not applicable
    for field in BUCKET_FIELDS[1:] + ["gdelt_sources"]:
        events[field] = events[field].astype("Int32")   # empty for ACLED and IMO GISIS
    events["event_count"] = events["event_count"].astype("int32")
    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in the events layer")
    log.info("Task 2.18 | events layer: %s rows; by source (rows / events = sum of event_count): %s",
             f"{len(events):,}", ", ".join(f"{s} {n:,} / {e:,}" for s, (n, e) in events.groupby(
                 "source")["event_count"].agg(["size", "sum"]).iterrows()))
    common.write_layer(events, settings, "events", log)
    if context is not None:
        common.write_layer(context, settings, "gdelt_country_context", log)
    log.info("Run of s05_events finished")


if __name__ == "__main__":
    main()

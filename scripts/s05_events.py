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

Only the sources listed in events.sources are processed; GDELT (tasks 2.16-2.17) is
added later. The log holds counts only, never individual event or incident rows, and
no ship name, IMO number or narrative text is stored.
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

# imo_match: ACLED events also reported to IMO (null for GISIS rows).
# crew_harm: GISIS incidents where crew were threatened, injured or held (null for ACLED rows).
OUTPUT_FIELDS = ["event_id", "source", "category", "event_type", "sub_event_type",
                 "event_date", "fatalities", "geo_precision", "country_iso3",
                 "imo_match", "crew_harm"]


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
    events = events.assign(source="ACLED", category="violence", imo_match=False, crew_harm=pd.NA)
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
                                 fatalities=0, geo_precision=1, country_iso3=pd.NA, imo_match=pd.NA)
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
    if not layers:
        raise ValueError("events.sources lists no source that S05 can process")

    events = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True),
                              crs=settings["project"]["storage_crs"])
    # Joining the sources mixes column types; set them again so empty values are written
    # as nulls (an object column would write the text "<NA>").
    events["country_iso3"] = events["country_iso3"].astype("string")
    for flag in ["imo_match", "crew_harm"]:
        events[flag] = events[flag].astype("boolean")   # true/false, empty where not applicable
    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in the events layer")
    log.info("Task 2.18 | events layer: %s events; by source: %s", f"{len(events):,}",
             ", ".join(f"{s} {n:,}" for s, n in events["source"].value_counts().items()))
    common.write_layer(events, settings, "events", log)
    log.info("Run of s05_events finished")


if __name__ == "__main__":
    main()

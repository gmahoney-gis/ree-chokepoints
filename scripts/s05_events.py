"""S05: unified event layer (tasks 2.14 and 2.18; ACLED first pass for Lab 7).

Reads the ACLED chunks (one CSV per chokepoint query, paths.raw_files.acled) and
writes the events layer to the project database. Steps, each logged with the count
kept and removed:

1. Read all chunks.
2. Remove repeated events (the same event ID in more than one chunk, where query
   areas overlap).
3. Keep events inside events.window_start to window_end (CLAUDE.md decision 7).
4. Keep events within events.prefilter_radius_km of any candidate chokepoint
   (great-circle distance, task 2.18).
5. Keep the event types in events.acled.event_types (task 2.14).
6. Remove events whose geo precision is in events.acled.exclude_geo_precision.

Only the sources listed in events.sources are processed; ASAM (task 2.15) and GDELT
(tasks 2.16-2.17) are added later. Per the ACLED terms, the log holds counts only,
never individual event rows.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

import common

KNOWN_SOURCES = ["acled", "asam", "gdelt"]

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

OUTPUT_FIELDS = ["event_id", "source", "category", "event_type", "sub_event_type",
                 "event_date", "fatalities", "geo_precision", "country_iso3"]


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

def filter_window(events, settings, log):
    """Step 3: keep events inside the event window, both dates inclusive (decision 7)."""
    start = pd.Timestamp(settings["events"]["window_start"])
    end = pd.Timestamp(settings["events"]["window_end"])
    events["event_date"] = pd.to_datetime(events["event_date"], format="%Y-%m-%d")
    inside = events["event_date"].between(start, end)
    log.info("Task 2.14 | event dates in data: %s to %s", events["event_date"].min().date(),
             events["event_date"].max().date())
    common.log_filter(log, f"Task 2.14 | event window {start.date()} to {end.date()}",
                      int(inside.sum()), int((~inside).sum()), "event date outside window")
    return events[inside].reset_index(drop=True)


def filter_distance(events, settings, log):
    """Step 4 (task 2.18): keep events within the prefilter radius of any candidate chokepoint."""
    radius = settings["events"]["prefilter_radius_km"]
    chokepoints = common.read_layer(settings, "chokepoints")
    log.info("Task 2.18 | distance to %d candidate chokepoints (all points in layer '%s', "
             "included or not)", len(chokepoints), common.layer_name(settings, "chokepoints"))

    bad = ~(events["latitude"].between(-90, 90) & events["longitude"].between(-180, 180))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} ACLED events have coordinates outside the valid range")

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
    events = events.assign(source="ACLED", category="violence")
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
        layers.append(process_acled(settings, log))
    if not layers:
        raise ValueError("events.sources lists no source that S05 can process")

    events = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True),
                              crs=settings["project"]["storage_crs"])
    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in the events layer")
    log.info("Task 2.18 | events layer: %s events; by source: %s", f"{len(events):,}",
             ", ".join(f"{s} {n:,}" for s, n in events["source"].value_counts().items()))
    common.write_layer(events, settings, "events", log)
    log.info("Run of s05_events finished")


if __name__ == "__main__":
    main()

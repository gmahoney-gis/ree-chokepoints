"""S02: clean and reconcile the Comtrade trade flows (tasks 2.3-2.6).

Reads the S01 API files only (paths.raw_files.comtrade_api_dir):
2025 files with the transport-mode breakdown for all four HS codes, and the
2024 all-modes files for the main codes (top-10 trader check).

- Task 2.3: ISO-3 codes from reporterISO / partnerISO, remap S19 -> TWN, drop
  aggregate and other non-country codes, drop flows from a country to itself.
- Task 2.4: one record per origin-destination-HS flow from the all-modes rows
  (motCode 0): the importer's M record first, the exporter's X record (mirror)
  where the importer did not report.
- Task 2.5: maritime share of each flow: from the record's own mode breakdown
  where reported, otherwise the shared-land-border rule (Natural Earth).
- Task 2.6: coverage table -> outputs/tables/trade_coverage.csv.

Writes the trade_flows table to the project database. All four HS codes are
kept; in_main_run is true only for trade.hs_codes_main (CLAUDE.md decision 1).
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

import common

KEYS = ["reporterISO", "partnerISO", "flowCode", "cmdCode", "motCode"]
FLOW_KEYS = ["origin_iso3", "dest_iso3", "hs_code"]
COLUMNS = ["flow_id", "origin_iso3", "dest_iso3", "hs_code", "in_main_run", "value_usd",
           "net_weight_kg", "record_source", "mode_source", "maritime_share",
           "maritime_value_usd", "maritime_weight_kg"]


def usd(value):
    """Format a dollar value for the log."""
    return f"{value:,.0f} USD"


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------

def read_comtrade(settings, year, codes, kind, log):
    """Read the S01 files for one year: comtrade_<year>_<code>_<flow>_<kind>.csv."""
    folder = common.get_path(settings, "raw_files.comtrade_api_dir")
    frames = []
    for code in codes:
        for flow in settings["trade"]["flows"]:
            path = folder / f"comtrade_{year}_{code}_{flow}_{kind}.csv"
            df = pd.read_csv(path, dtype={"reporterISO": str, "partnerISO": str, "cmdCode": str},
                             usecols=KEYS + ["reporterDesc", "partnerDesc", "primaryValue", "netWgt"])
            log.info("Read %s (%s rows)", path.name, f"{len(df):,}")
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    log.info("%d | rows read %s | all-modes value %s", year, f"{len(df):,}",
             usd(df.loc[df["motCode"] == settings["trade"]["transport_mode"]["all_modes_code"],
                        "primaryValue"].sum()))
    return df


def country_names(df):
    """ISO-3 -> country name, from the reporter and partner descriptions."""
    names = pd.concat([df[["reporterISO", "reporterDesc"]].set_axis(["iso3", "name"], axis=1),
                       df[["partnerISO", "partnerDesc"]].set_axis(["iso3", "name"], axis=1)])
    return names.drop_duplicates("iso3").set_index("iso3")["name"]


# ---------------------------------------------------------------------
# Task 2.3: country codes
# ---------------------------------------------------------------------

def total_value(df, settings):
    """All-modes value of some rows (mode rows would count the same trade twice)."""
    return df.loc[df["motCode"] == settings["trade"]["transport_mode"]["all_modes_code"],
                  "primaryValue"].sum()


def clean_codes(df, settings, log, label):
    """Task 2.3: clean, remap and filter the reporter and partner codes."""
    trade = settings["trade"]
    step = f"Task 2.3 | {label}"
    for col in ["reporterISO", "partnerISO"]:
        df[col] = common.clean_iso3(df[col])

    # Task 2.3: remap S19 -> TWN, logging rows and value per code and column.
    for old, new in (trade.get("iso3_remap") or {}).items():
        for col in ["reporterISO", "partnerISO"]:
            hit = df[col] == old
            log.info("%s | remap %s -> %s in %s | rows %s | all-modes value %s", step, old, new,
                     col, f"{hit.sum():,}", usd(total_value(df[hit], settings)))
        df[["reporterISO", "partnerISO"]] = df[["reporterISO", "partnerISO"]].apply(
            common.remap_iso3, settings=settings)

    # Task 2.3: aggregate codes (World, regions, bunkers, n.e.s.).
    agg = common.is_aggregate(df["reporterISO"], settings) | common.is_aggregate(df["partnerISO"], settings)
    for code in trade["aggregate_codes"]:
        hit = (df["reporterISO"] == code) | (df["partnerISO"] == code)
        if hit.any():
            log.info("%s | aggregate code %s | rows %s | all-modes value %s", step, code,
                     f"{hit.sum():,}", usd(total_value(df[hit], settings)))
    common.log_filter(log, f"{step} | aggregate codes", int((~agg).sum()), int(agg.sum()),
                      f"not a country (trade.aggregate_codes); all-modes value "
                      f"{usd(total_value(df[agg], settings))}")
    df = df[~agg]

    # Task 2.3: any other code that is not a three-letter ISO-3 country code.
    pattern = re.compile(trade["iso3_pattern"])
    valid = df["reporterISO"].map(lambda c: bool(pattern.match(str(c)))) & \
        df["partnerISO"].map(lambda c: bool(pattern.match(str(c))))
    bad = df.loc[~valid]
    for code in sorted(set(bad["reporterISO"]) | set(bad["partnerISO"])):
        if not pattern.match(str(code)):
            hit = (bad["reporterISO"] == code) | (bad["partnerISO"] == code)
            log.info("%s | non-country code %r | rows %s | all-modes value %s", step, code,
                     f"{hit.sum():,}", usd(total_value(bad[hit], settings)))
    common.log_filter(log, f"{step} | other non-country codes", int(valid.sum()), int((~valid).sum()),
                      f"not an ISO-3 country code (trade.iso3_pattern); all-modes value "
                      f"{usd(total_value(bad, settings))}")
    df = df[valid]

    # Task 2.3: a country trading with itself (re-imports) is not a route.
    self_trade = df["reporterISO"] == df["partnerISO"]
    common.log_filter(log, f"{step} | origin = destination", int((~self_trade).sum()),
                      int(self_trade.sum()), f"country reporting trade with itself; all-modes value "
                      f"{usd(total_value(df[self_trade], settings))}")
    df = df[~self_trade]

    # A remap could make two rows share a key; add them up so each key is one record.
    before = len(df)
    df = df.groupby(KEYS, as_index=False)[["primaryValue", "netWgt"]].sum(min_count=1)
    if len(df) != before:
        log.info("%s | %d rows merged after the remap (same key)", step, before - len(df))
    return df


# ---------------------------------------------------------------------
# Task 2.4: importer record first
# ---------------------------------------------------------------------

def reconcile(df, settings, log, label):
    """Task 2.4: one all-modes record per flow, importer first, exporter mirror fills gaps."""
    totals = df[df["motCode"] == settings["trade"]["transport_mode"]["all_modes_code"]]
    # Importer M record: the partner is the origin. Exporter X record: the reporter is.
    imports = totals[totals["flowCode"] == "M"].rename(columns={"partnerISO": "origin_iso3",
                                                               "reporterISO": "dest_iso3"})
    exports = totals[totals["flowCode"] == "X"].rename(columns={"reporterISO": "origin_iso3",
                                                               "partnerISO": "dest_iso3"})
    cols = {"cmdCode": "hs_code", "primaryValue": "value_usd", "netWgt": "net_weight_kg"}
    imports = imports.rename(columns=cols).assign(record_source="importer")
    exports = exports.rename(columns=cols).assign(record_source="mirror")

    # Task 2.4: exporter records fill only the flows the importer did not report.
    have = pd.MultiIndex.from_frame(imports[FLOW_KEYS])
    mirror = exports[~pd.MultiIndex.from_frame(exports[FLOW_KEYS]).isin(have)]
    log.info("Task 2.4 | %s | importer M records %s | exporter X records %s | X records "
             "replaced by an importer record %s", label, f"{len(imports):,}", f"{len(exports):,}",
             f"{len(exports) - len(mirror):,}")
    flows = pd.concat([imports, mirror], ignore_index=True)[FLOW_KEYS + ["value_usd", "net_weight_kg",
                                                                         "record_source"]]
    flows["in_main_run"] = flows["hs_code"].isin(settings["trade"]["hs_codes_main"])
    return flows


def log_reconcile(flows, log, label):
    """Task 2.4: counts and value from each source, and flows without a weight."""
    for main, part in flows.groupby("in_main_run"):
        group = f"{label} {'main run' if main else 'sensitivity'}"
        total = part["value_usd"].sum()
        for source, sub in part.groupby("record_source"):
            log.info("Task 2.4 | %s | %-8s flows %s | value %s (%.1f%%)", group, source,
                     f"{len(sub):,}", usd(sub["value_usd"].sum()), 100 * sub["value_usd"].sum() / total)
        no_weight = part["net_weight_kg"].isna()
        zero_weight = part["net_weight_kg"].eq(0)
        log.info("Task 2.4 | %s | flows without a weight: %s missing (value %s), %s zero (value %s)",
                 group, f"{no_weight.sum():,}", usd(part.loc[no_weight, "value_usd"].sum()),
                 f"{zero_weight.sum():,}", usd(part.loc[zero_weight, "value_usd"].sum()))
        zero_value = part["value_usd"] <= 0
        if zero_value.any():
            log.info("Task 2.4 | %s | flows with zero value: %s (kept)", group, f"{zero_value.sum():,}")


# ---------------------------------------------------------------------
# Task 2.5: transport mode
# ---------------------------------------------------------------------

def mode_breakdown(df, settings):
    """Task 2.5: maritime, excluded and unclassified value per reported record."""
    mode = settings["trade"]["transport_mode"]
    rows = df[df["motCode"] != mode["all_modes_code"]].copy()
    groups = {"maritime": mode["maritime_codes"], "excluded": mode["excluded_codes"],
              "unclassified": mode["unclassified_codes"]}
    known = set().union(*groups.values())
    unknown = set(rows["motCode"]) - known
    if unknown:
        raise ValueError(f"motCode values {sorted(unknown)} are not listed in trade.transport_mode")
    rows["group"] = None
    for name, codes in groups.items():
        rows.loc[rows["motCode"].isin(codes), "group"] = name
    table = rows.pivot_table(index=["reporterISO", "partnerISO", "flowCode", "cmdCode"],
                             columns="group", values="primaryValue", aggfunc="sum", fill_value=0)
    return table.reindex(columns=list(groups), fill_value=0).reset_index()


def land_neighbours(settings, log):
    """Task 2.5: pairs of countries that share a land border in Natural Earth."""
    field = settings["trade"]["transport_mode"]["border_iso_field"]
    path = common.get_path(settings, "raw_files.natural_earth_countries")
    countries = pyogrio.read_dataframe(path, columns=[field])
    log.info("Read %s (%d features)", path.name, len(countries))
    countries = countries[countries[field] != "-99"].dissolve(by=field).reset_index()
    pairs = gpd.sjoin(countries, countries, predicate="intersects")
    pairs = pairs[pairs[f"{field}_left"] != pairs[f"{field}_right"]]
    neighbours = {frozenset(p) for p in zip(pairs[f"{field}_left"], pairs[f"{field}_right"])}
    log.info("Task 2.5 | Natural Earth countries (%s) %d | land-border pairs %d", field,
             len(countries), len(neighbours))
    return neighbours, set(countries[field])


def classify_mode(flows, df, settings, log):
    """Task 2.5: maritime share from the record's own breakdown, else the land-border rule."""
    breakdown = mode_breakdown(df, settings)
    # The record's reporter: the importer for importer records, the exporter for mirrors.
    is_imp = flows["record_source"] == "importer"
    flows["reporterISO"] = np.where(is_imp, flows["dest_iso3"], flows["origin_iso3"])
    flows["partnerISO"] = np.where(is_imp, flows["origin_iso3"], flows["dest_iso3"])
    flows["flowCode"] = np.where(is_imp, "M", "X")
    flows = flows.merge(breakdown.rename(columns={"cmdCode": "hs_code"}), how="left",
                        on=["reporterISO", "partnerISO", "flowCode", "hs_code"])
    flows[["maritime", "excluded", "unclassified"]] = flows[["maritime", "excluded", "unclassified"]].fillna(0)

    classified = flows["maritime"] + flows["excluded"]
    reported = classified > 0
    only_unclassified = ~reported & (flows["unclassified"] > 0)
    log.info("Task 2.5 | flows with a usable mode breakdown %s | breakdown with only "
             "unclassified modes (land-border rule used) %s | no breakdown %s",
             f"{reported.sum():,}", f"{only_unclassified.sum():,}",
             f"{(~reported & ~only_unclassified).sum():,}")
    unclassified = flows.loc[reported, "unclassified"].sum()
    log.info("Task 2.5 | reported breakdowns: unclassified value left out of the share %s "
             "(%.1f%% of those flows' value)", usd(unclassified),
             100 * unclassified / flows.loc[reported, "value_usd"].sum())

    # Task 2.5: land-border rule where no breakdown is usable.
    neighbours, ne_codes = land_neighbours(settings, log)
    border = [frozenset((o, d)) in neighbours for o, d in zip(flows["origin_iso3"], flows["dest_iso3"])]
    rule_share = np.where(border, 0.0, 1.0)
    missing = sorted((set(flows["origin_iso3"]) | set(flows["dest_iso3"])) - ne_codes)
    log.info("Task 2.5 | trade codes not in Natural Earth (treated as no land border): %s",
             ", ".join(missing) or "none")

    flows["mode_source"] = np.where(reported, "reported", "land_border_rule")
    flows["maritime_share"] = np.where(reported, flows["maritime"] / classified.where(reported, 1),
                                       rule_share)
    flows["maritime_value_usd"] = flows["value_usd"] * flows["maritime_share"]
    flows["maritime_weight_kg"] = flows["net_weight_kg"] * flows["maritime_share"]
    return flows


def log_modes(flows, names, log):
    """Task 2.5: counts and excluded value under each rule, with the largest excluded flows."""
    for main, part in flows.groupby("in_main_run"):
        group = "main run" if main else "sensitivity"
        total = part["value_usd"].sum()
        for rule, sub in part.groupby("mode_source"):
            excluded = sub["value_usd"] - sub["maritime_value_usd"]
            log.info("Task 2.5 | %s | %-16s flows %s | value %s | excluded as non-maritime %s "
                     "(%.1f%% of the rule's value, %.1f%% of all value) | fully overland flows %s",
                     group, rule, f"{len(sub):,}", usd(sub["value_usd"].sum()), usd(excluded.sum()),
                     100 * excluded.sum() / sub["value_usd"].sum(), 100 * excluded.sum() / total,
                     f"{(sub['maritime_share'] == 0).sum():,}")
            if not main:
                continue
            top = sub.assign(excluded=excluded).nlargest(10, "excluded")
            for _, row in top.iterrows():
                log.info("Task 2.5 | main run | %-16s top excluded | %s (%s) -> %s (%s) HS %s | "
                         "excluded %s | maritime share %.3f", rule, row["origin_iso3"],
                         names.get(row["origin_iso3"], ""), row["dest_iso3"],
                         names.get(row["dest_iso3"], ""), row["hs_code"], usd(row["excluded"]),
                         row["maritime_share"])


# ---------------------------------------------------------------------
# Task 2.6: coverage table
# ---------------------------------------------------------------------

def coverage_table(flows, flows_2024, reporters_2025, names, settings, log):
    """Task 2.6: top-10 traders in the check year, mirror share, non-maritime share."""
    n = settings["trade"]["coverage_top_n"]
    year = settings["project"]["coverage_check_year"]
    main_2024 = flows_2024[flows_2024["in_main_run"]]
    # Imports + exports per country, from the reconciled check-year flows.
    trade = (main_2024.groupby("origin_iso3")["value_usd"].sum()
             .add(main_2024.groupby("dest_iso3")["value_usd"].sum(), fill_value=0)
             .nlargest(n))
    rows = []
    for rank, (iso3, value) in enumerate(trade.items(), start=1):
        m, x = (iso3, "M") in reporters_2025, (iso3, "X") in reporters_2025
        rows.append({"section": f"top_traders_{year}", "rank": rank, "item": names.get(iso3, iso3),
                     "iso3": iso3, "value_usd": value, "share": value / (2 * main_2024["value_usd"].sum()),
                     "reported_2025_imports": m, "reported_2025_exports": x})
        log.info("Task 2.6 | top trader %2d %s %-28s %s | reported 2025: imports %s, exports %s",
                 rank, iso3, names.get(iso3, ""), usd(value), m, x)

    main = flows[flows["in_main_run"]]
    total = main["value_usd"].sum()
    study = f"main_run_{settings['project']['study_year']}"
    summary = [("total value", total)]
    summary.append(("value from mirror records", main.loc[main["record_source"] == "mirror", "value_usd"].sum()))
    for rule in ["reported", "land_border_rule"]:
        sub = main[main["mode_source"] == rule]
        summary.append((f"value classified by {rule}", sub["value_usd"].sum()))
        summary.append((f"excluded as non-maritime ({rule})",
                        (sub["value_usd"] - sub["maritime_value_usd"]).sum()))
    summary.append(("maritime value", main["maritime_value_usd"].sum()))
    for item, value in summary:
        rows.append({"section": study, "item": item, "value_usd": value, "share": value / total})
        log.info("Task 2.6 | %s | %-40s %s (%.1f%%)", study, item, usd(value), 100 * value / total)
    table = pd.DataFrame(rows, columns=["section", "rank", "item", "iso3", "value_usd", "share",
                                        "reported_2025_imports", "reported_2025_exports"])
    table["rank"] = table["rank"].astype("Int64")
    return table


# ---------------------------------------------------------------------

def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    trade = settings["trade"]
    study_year = settings["project"]["study_year"]
    check_year = settings["project"]["coverage_check_year"]

    # Tasks 2.3-2.5 on the study year, all four HS codes.
    raw = read_comtrade(settings, study_year, trade["hs_codes_main"] + trade["hs_codes_sensitivity"],
                        "bymode", log)
    names = country_names(raw).to_dict()
    names.update({new: names.get(new, "Taiwan") for new in (trade.get("iso3_remap") or {}).values()})
    clean = clean_codes(raw, settings, log, str(study_year))
    flows = reconcile(clean, settings, log, str(study_year))
    log_reconcile(flows, log, str(study_year))
    flows = classify_mode(flows, clean, settings, log)
    log_modes(flows, names, log)

    flows["flow_id"] = flows["origin_iso3"] + "_" + flows["dest_iso3"] + "_" + flows["hs_code"]
    flows = flows[COLUMNS].sort_values(["hs_code", "origin_iso3", "dest_iso3"]).reset_index(drop=True)
    if flows["flow_id"].duplicated().any():
        raise ValueError("Duplicate flow_id values in trade_flows")
    common.write_layer(flows, settings, "trade_flows", log)

    # Task 2.6: check-year flows (main codes, all modes) for the top-10 trader check.
    raw_2024 = read_comtrade(settings, check_year, trade["hs_codes_main"], "total", log)
    names = {**country_names(raw_2024).to_dict(), **names}
    flows_2024 = reconcile(clean_codes(raw_2024, settings, log, str(check_year)), settings, log,
                           str(check_year))
    main_clean = clean[clean["cmdCode"].isin(trade["hs_codes_main"])]
    reporters_2025 = set(zip(main_clean["reporterISO"], main_clean["flowCode"]))
    coverage = coverage_table(flows, flows_2024, reporters_2025, names, settings, log)
    tables = common.get_path(settings, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(tables / "trade_coverage.csv", index=False)
    log.info("Wrote %s (%d rows)", tables / "trade_coverage.csv", len(coverage))
    log.info("Run of s02_clean_trade finished")


if __name__ == "__main__":
    main()

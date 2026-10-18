"""S01: download Comtrade trade records through the API (tasks 2.1-2.2).

Uses comtradeapicall.getFinalData with the subscription key from the environment
variable named in trade.comtrade_api.key_env_var. The key is never written to a
file, the log, or the screen.

Requests (one per HS code and flow, to stay under the record limit):

- Task 2.1: study year, main and sensitivity HS codes, imports and exports, all
  reporters and partners, with the transport-mode breakdown (motCode 0 = all modes,
  plus individual modes where a country reports them).
- Task 2.2: coverage-check year, main HS codes, imports and exports, all reporters
  and partners, all modes only (for the top-10 trader check in S02).

Each response is saved unchanged as a CSV under paths.raw_files.comtrade_api_dir,
and its row in data/raw/MANIFEST.csv is added or replaced.

Usage:
    python scripts/s01_get_trade.py           # all requests
    python scripts/s01_get_trade.py --test    # one small request, nothing saved
"""

import argparse
import hashlib
import os
import sys
import time
from datetime import date
from pathlib import Path

import comtradeapicall
import pandas as pd

import common

# Reporters that must appear for the coverage check (Comtrade numeric codes).
CHECK_REPORTERS = {"156": "China", "392": "Japan", "842": "United States", "410": "Korea"}
API_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
PROVIDER = "UN Comtrade API (comtradeapicall)"
CMD_NAMES = {"280530": "rare earth metals, scandium and yttrium",
             "284610": "cerium compounds",
             "284690": "other rare earth compounds",
             "850511": "permanent magnets"}


def read_key(settings):
    """Return the API key from the environment, or stop with a clear message."""
    var = settings["trade"]["comtrade_api"]["key_env_var"]
    key = os.environ.get(var, "").strip()
    if not key:
        sys.exit(f"S01 stopped: environment variable {var} is not set. "
                 f"Set it to your Comtrade subscription key and rerun.")
    return key


def plan_requests(settings):
    """Tasks 2.1-2.2: list every request as a dict of year, code, flow and mode option."""
    trade = settings["trade"]
    requests = []
    # Task 2.1: study year with the transport-mode breakdown.
    for code in trade["hs_codes_main"] + trade["hs_codes_sensitivity"]:
        for flow in trade["flows"]:
            requests.append({"task": "2.1", "year": settings["project"]["study_year"],
                             "code": code, "flow": flow, "by_mode": True})
    # Task 2.2: coverage-check year, totals (all modes) only.
    for code in trade["hs_codes_main"]:
        for flow in trade["flows"]:
            requests.append({"task": "2.2", "year": settings["project"]["coverage_check_year"],
                             "code": code, "flow": flow, "by_mode": False})
    return requests


def file_name(req):
    """File name with year, HS code, flow and mode option, e.g. comtrade_2025_280530_M_bymode.csv."""
    return f"comtrade_{req['year']}_{req['code']}_{req['flow']}_{'bymode' if req['by_mode'] else 'total'}.csv"


def query_args(req, settings, reporter=None):
    """Arguments for getFinalData (without the key).

    partner2Code 0 and customsCode C00 keep only the totals of those breakdowns,
    so the only breakdown returned is transport mode. motCode None returns every
    mode; motCode 0 returns all modes combined.
    """
    api = settings["trade"]["comtrade_api"]
    return dict(typeCode=api["type_code"], freqCode=api["freq_code"], clCode=api["classification"],
                period=str(req["year"]), reporterCode=reporter, cmdCode=req["code"],
                flowCode=req["flow"], partnerCode=None, partner2Code="0", customsCode="C00",
                motCode=None if req["by_mode"] else "0", maxRecords=api["max_records"],
                breakdownMode="plus" if req["by_mode"] else "classic", includeDesc=True)


def fetch(key, args, settings, log):
    """Run one request, retrying after a wait; stop if it still fails.

    getFinalData returns None on an HTTP error (it prints the API's message itself).
    """
    api = settings["trade"]["comtrade_api"]
    for attempt in range(1 + api["retries"]):
        try:
            df = comtradeapicall.getFinalData(key, **args)
        except Exception as err:  # network errors; message never includes the key
            log.warning("Request failed (%s)", type(err).__name__)
            df = None
        if df is not None:
            return df
        if attempt < api["retries"]:
            log.warning("No data returned; retrying in %s s", api["retry_wait_seconds"])
            time.sleep(api["retry_wait_seconds"])
    raise RuntimeError(f"Comtrade request failed after {1 + api['retries']} attempts: "
                       f"{args['period']} {args['cmdCode']} {args['flowCode']}")


def summarise(df, name, settings, log):
    """Log records, reporters, primaryValue totals and the four check reporters for one file."""
    reporters = set(df["reporterCode"].astype(str)) if len(df) else set()
    if len(df):
        totals = df[(df["motCode"] == 0) & (df["partnerCode"] == 0)]
        world_value = totals["primaryValue"].sum()
        all_rows_value = df["primaryValue"].sum()
    else:
        world_value = all_rows_value = 0.0
    present = [n for c, n in CHECK_REPORTERS.items() if c in reporters]
    missing = [n for c, n in CHECK_REPORTERS.items() if c not in reporters]
    log.info("%s | records %s | reporters %d | primaryValue: partner World, all modes %s USD; "
             "sum of all rows %s USD (counts World and mode rows twice)",
             name, f"{len(df):,}", len(reporters), f"{world_value:,.0f}", f"{all_rows_value:,.0f}")
    log.info("%s | check reporters present: %s | missing: %s",
             name, ", ".join(present) or "none", ", ".join(missing) or "none")
    if len(df) and "motCode" in df:
        log.info("%s | records by motCode: %s", name,
                 ", ".join(f"{m}={n}" for m, n in df["motCode"].value_counts().sort_index().items()))
    return {"reporters": reporters, "world_value": world_value}


def md5(path):
    """MD5 checksum of a file, as recorded in the manifest."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_row(path, req, df, today):
    """One MANIFEST.csv row for a downloaded file."""
    flow = {"M": "imports", "X": "exports"}[req["flow"]]
    mode = ("transport-mode breakdown (motCode 0 plus reported modes)" if req["by_mode"]
            else "all modes only (motCode 0)")
    use = ("S02 sensitivity test only (CLAUDE.md decision 1)" if req["code"] == "850511"
           else "S02" if req["task"] == "2.1" else "S02 top-10 trader check (task 2.2)")
    return {
        "file_path": path.relative_to(common.ROOT).as_posix(),
        "dataset": f"Comtrade HS {req['code'][:4]}.{req['code'][4:]} {CMD_NAMES.get(req['code'], '')}, "
                   f"{req['year']} {flow}",
        "provider": PROVIDER,
        "source_url": API_URL,
        "download_date": today,
        "download_date_source": "S01 run date",
        "size_bytes": path.stat().st_size,
        "md5": md5(path),
        "rows": len(df),
        "coverage_period": f"{req['year']} (annual)",
        "used_by": use,
        "status": "raw",
        "notes": f"task {req['task']}; getFinalData: all reporters, all partners, flow {req['flow']}, "
                 f"partner2Code 0, customsCode C00, {mode}; "
                 f"{df['reporterCode'].nunique() if len(df) else 0} reporters; UTF-8",
    }


def update_manifest(rows, settings, log):
    """Add or replace this script's rows in data/raw/MANIFEST.csv (matched on file_path)."""
    path = common.get_path(settings, "raw") / "MANIFEST.csv"
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    new = pd.DataFrame(rows).astype(str)
    kept = manifest[~manifest["file_path"].isin(new["file_path"])]
    out = pd.concat([kept, new[manifest.columns]], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    log.info("Updated %s: %d rows written for S01 (%d replaced)", path, len(new),
             len(manifest) - len(kept))


def run_test(key, settings, log):
    """One small request (one reporter, one code); show the columns, save nothing."""
    req = {"task": "test", "year": settings["project"]["study_year"],
           "code": settings["trade"]["hs_codes_main"][0], "flow": "M", "by_mode": True}
    df = fetch(key, query_args(req, settings, reporter="842"), settings, log)
    log.info("Test request: reporter 842, %s %s %s | %d records", req["year"], req["code"], req["flow"], len(df))
    log.info("Columns (%d): %s", len(df.columns), ", ".join(df.columns))


def main():
    parser = argparse.ArgumentParser(description="Download Comtrade records (tasks 2.1-2.2).")
    parser.add_argument("--test", action="store_true", help="one small request, nothing saved")
    test = parser.parse_args().test

    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    key = read_key(settings)
    if test:
        run_test(key, settings, log)
        return

    api = settings["trade"]["comtrade_api"]
    out_dir = common.get_path(settings, "raw_files.comtrade_api_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    requests = plan_requests(settings)
    log.info("Planned %d requests (one per year, HS code and flow) into %s", len(requests), out_dir)
    rows, by_year = [], {}
    for i, req in enumerate(requests, start=1):
        if i > 1:
            time.sleep(api["pause_seconds"])
        df = fetch(key, query_args(req, settings), settings, log)
        name = file_name(req)
        log.info("Task %s | request %d of %d | %s %s %s %s | records %s", req["task"], i, len(requests),
                 req["year"], req["code"], req["flow"], "by mode" if req["by_mode"] else "total",
                 f"{len(df):,}")
        if len(df) >= api["max_records"]:
            log.warning("%s reached the record limit (%s); the response is probably truncated",
                        name, f"{api['max_records']:,}")
        path = out_dir / name
        df.to_csv(path, index=False, encoding="utf-8")
        stats = summarise(df, name, settings, log)
        by_year.setdefault((req["year"], req["code"] in settings["trade"]["hs_codes_main"]),
                           set()).update(stats["reporters"])
        rows.append(manifest_row(path, req, df, today))
        log.info("Wrote %s", path)

    update_manifest(rows, settings, log)

    # Summary: reporters across all files for each year (main codes).
    for (year, main_codes), reporters in sorted(by_year.items()):
        label = "main codes" if main_codes else "sensitivity codes"
        present = [n for c, n in CHECK_REPORTERS.items() if c in reporters]
        missing = [n for c, n in CHECK_REPORTERS.items() if c not in reporters]
        log.info("Summary %s %s | reporters in any file %d | check reporters present: %s | missing: %s",
                 year, label, len(reporters), ", ".join(present) or "none", ", ".join(missing) or "none")
    log.info("S01 finished: %d requests, %d files written", len(requests), len(rows))


if __name__ == "__main__":
    main()

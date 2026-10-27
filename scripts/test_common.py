"""Tests for common.py (task 0.4).

Run with:  python scripts/test_common.py
Writes only to a temporary folder; reads the raw Comtrade files without changing them.
"""

import copy
import sys
import tempfile
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SETTINGS = common.load_settings()
TMP = Path(tempfile.mkdtemp(prefix="test_common_"))


def approx(value, expected, tolerance):
    """True if value is within tolerance of expected."""
    return abs(value - expected) <= tolerance


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

def test_settings():
    assert SETTINGS["project"]["study_year"] == 2025
    assert common.get_path(SETTINGS, "database") == common.ROOT / "data" / "project.gpkg"
    assert common.get_path(SETTINGS, "raw_files.comtrade_dir").is_dir()
    # A TODO entry must raise ValueError. Test it on a copy with a made-up TODO entry, so the
    # test does not go stale when a real path is filled in (raw_files.asam was, see Decision 9).
    settings = copy.deepcopy(SETTINGS)
    settings["paths"]["raw_files"]["test_todo"] = "TODO"
    for bad_key, error in [("raw_files.test_todo", ValueError), ("no_such_path", KeyError)]:
        try:
            common.get_path(settings, bad_key)
        except error:
            pass
        else:
            raise AssertionError(f"get_path('{bad_key}') should raise {error.__name__}")


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def test_log():
    log_dir = TMP / "logs"
    for run in (1, 2):
        log = common.start_log("demo", SETTINGS, log_dir=log_dir)
        common.log_filter(log, f"Task 9.9 run {run}", 15120, 829, "not a country")
        for handler in log.handlers:
            handler.flush()
    text = (log_dir / "demo.log").read_text(encoding="utf-8")
    assert "run 2 | kept 15,120 | removed 829 | reason: not a country" in text
    assert "run 1" not in text, "second run should replace the log, not add to it"
    assert text.count("Run of demo started") == 1


# ---------------------------------------------------------------------
# Great-circle distance
# ---------------------------------------------------------------------

def test_distance():
    gc = common.great_circle_km
    london_paris = gc(-0.1278, 51.5074, 2.3522, 48.8566, SETTINGS)
    print(f"    London-Paris: {london_paris:.1f} km")
    assert approx(london_paris, 343.5, 2)
    assert gc(10, 20, 10, 20, SETTINGS) == 0
    assert approx(gc(0, 0, 1, 0, SETTINGS), 111.2, 0.1)
    dateline = gc(179, 0, -179, 0, SETTINGS)
    print(f"    Across the date line (179E to 179W): {dateline:.1f} km")
    assert approx(dateline, 222.4, 0.2)
    antipodes = gc(0, 0, 180, 0, SETTINGS)
    print(f"    Antipodes: {antipodes:.1f} km")
    assert approx(antipodes, np.pi * SETTINGS["project"]["earth_radius_km"], 0.01)
    # Columns give the same answer as single values.
    lons = pd.Series([-0.1278, 0, 179])
    lats = pd.Series([51.5074, 0, 0])
    many = gc(lons, lats, [2.3522, 1, -179], [48.8566, 0, 0], SETTINGS)
    assert np.allclose(many, [london_paris, gc(0, 0, 1, 0, SETTINGS), dateline])
    # Matrix: 2 points against 3 points.
    matrix = common.distance_matrix_km([0, 10], [0, 0], [0, 1, 2], [0, 0, 0], SETTINGS)
    assert matrix.shape == (2, 3)
    assert approx(matrix[0, 2], 2 * 111.195, 0.1)


# ---------------------------------------------------------------------
# ISO-3 codes
# ---------------------------------------------------------------------

def test_iso3_simple():
    assert common.clean_iso3(" chn") == "CHN"
    assert common.is_aggregate("_X ", SETTINGS)
    assert common.is_aggregate("W00", SETTINGS)
    assert not common.is_aggregate("CHN", SETTINGS)
    assert not common.is_aggregate("S19", SETTINGS), "S19 is remapped to TWN, not dropped"
    assert common.remap_iso3("S19", SETTINGS) == "TWN"
    assert common.remap_iso3("CHN", SETTINGS) == "CHN"
    codes = pd.Series(["X1 ", "usa", None, "S19"])
    assert common.is_aggregate(codes, SETTINGS).tolist() == [True, False, False, False]
    assert common.remap_iso3(codes, SETTINGS).tolist()[3] == "TWN"


def read_comtrade(path):
    """Read one raw Comtrade file as text (the files are a mix of UTF-8 and Latin-1)."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")


def test_iso3_on_comtrade():
    """Read-only pass over the real files: every code is either ISO-3 or listed in settings."""
    comtrade_dir = common.get_path(SETTINGS, "raw_files.comtrade_dir")
    for path in sorted(comtrade_dir.glob("*.csv")):
        df = read_comtrade(path)
        n = len(df)
        agg_partner = common.is_aggregate(df["partnerISO"], SETTINGS)
        agg_reporter = common.is_aggregate(df["reporterISO"], SETTINGS)
        s19 = (common.clean_iso3(df["partnerISO"]) == "S19") | (common.clean_iso3(df["reporterISO"]) == "S19")
        print(f"    {path.name}: {n:,} rows | aggregate partner {agg_partner.sum():,}"
              f" | aggregate reporter {agg_reporter.sum():,} | S19 rows {s19.sum():,}")
        for column in ("reporterISO", "partnerISO", "partner2ISO"):
            codes = common.remap_iso3(df[column], SETTINGS)
            codes = codes[~common.is_aggregate(codes, SETTINGS)]
            bad = codes[~codes.str.fullmatch(r"[A-Z]{3}")].unique().tolist()
            assert not bad, f"{path.name} {column}: unexpected codes {bad}"


# ---------------------------------------------------------------------
# GeoPackage
# ---------------------------------------------------------------------

def test_geopackage():
    db = TMP / "test.gpkg"
    points = gpd.GeoDataFrame(
        {"chokepoint_id": ["CP01", "CP02"], "name": ["Strait of Hormuz", "Strait of Malacca"]},
        geometry=[Point(56.4, 26.6), Point(101.0, 2.5)], crs="EPSG:4326")
    table = pd.DataFrame({"flow_id": ["F1", "F2", "F3"], "value_usd": [1.5, 2.0, 3.25]})

    common.write_layer(points, SETTINGS, "chokepoints", database=db)
    common.write_layer(table, SETTINGS, "trade_flows", database=db)
    common.write_layer(points, SETTINGS, "chokepoints", database=db)   # rewrite

    assert sorted(common.list_layers(SETTINGS, db)) == ["chokepoints", "trade_flows"], \
        "writing one layer must not remove the other"
    back = common.read_layer(SETTINGS, "chokepoints", database=db)
    assert len(back) == 2, f"rewrite should replace rows, got {len(back)}"
    assert back.crs.equals("EPSG:4326")
    assert back["chokepoint_id"].tolist() == ["CP01", "CP02"]
    assert back.geometry.iloc[0].equals(Point(56.4, 26.6))
    back_table = common.read_layer(SETTINGS, "trade_flows", database=db)
    assert list(back_table.columns) == ["flow_id", "value_usd"]
    assert back_table["value_usd"].tolist() == [1.5, 2.0, 3.25]

    for bad_call, error in [
        (lambda: common.write_layer(points.to_crs("EPSG:8857"), SETTINGS, "chokepoints", database=db), ValueError),
        (lambda: common.write_layer(points.set_crs(None, allow_override=True), SETTINGS, "chokepoints", database=db), ValueError),
        (lambda: common.write_layer(points, SETTINGS, "chokepionts", database=db), KeyError),
        (lambda: common.read_layer(SETTINGS, "events", database=db), ValueError),
    ]:
        try:
            bad_call()
        except error:
            pass
        else:
            raise AssertionError(f"expected {error.__name__}")


def main():
    tests = [test_settings, test_log, test_distance, test_iso3_simple, test_iso3_on_comtrade, test_geopackage]
    failed = 0
    for test in tests:
        print(f"{test.__name__}")
        try:
            test()
            print("    PASS")
        except Exception:
            failed += 1
            print("    FAIL")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed (temporary files in {TMP})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

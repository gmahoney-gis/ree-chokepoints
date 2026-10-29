"""S04: ports, port trade shares, landlocked gateways and links to the sea-lane network
(tasks 2.10-2.13).

ports.assignment chooses how countries get ports (Decision 14 in CLAUDE.md):

principal_ports (main run):
- Task 2.10: ports come from PortWatch (Ports.csv). Offshore oil terminals and mostly-tanker
  ports are removed first (ports.exclude_offshore_terminals, max_tanker_share,
  min_cargo_share), because REE ships in containers or as general cargo. The World Port
  Index (WPI) is used only for a trading country PortWatch lacks: its WPI harbors of the
  largest size class it has, with equal shares.
- Task 2.11: for each country, separately for exports and imports, the remaining shares are
  rescaled to 100% and ports are taken largest first until they reach
  ports.principal_port_share (the port that crosses it is included).
- Task 2.12: countries in config/landlocked_gateways.csv are always landlocked (also KAZ and
  MDA, which have PortWatch ports) and use the PortWatch port in gateway_portid, share 1.
- Task 2.13: each port is linked to the nearest node of the searoute sea-lane network. Ports
  farther than ports.snap_warning_km are kept and flagged (snap_flag). Only the hand-checked
  ports in ports.snap_exclude (nearest node in another sea) are dropped; their share moves to
  the country's other principal ports on that side. Ports whose network point lies inside a
  chokepoint zone are logged (Decision 10c).
  -> ports layer and country_port_shares table (S06 task 4.1 splits flows with them).

nearest_gateway (S10 alternative, proposal Table 4; the Lab 7/8 first pass):
- every trading country gets candidate WPI harbors of its largest size class; landlocked
  countries get the WPI gateway in the gateway file (a trading country with no WPI port is
  given the nearest foreign harbor, appended to the file for review). S06 routes each flow
  between the closest pair. -> ports layer only; country_port_shares is not written.

A port can appear more than once: once for its own country and once for each landlocked
country it serves as a gateway. port_id is "<country_iso3>_<portwatch id>" (PortWatch) or
"<country_iso3>_WPI<number>" (WPI), so it is unique and stable between runs.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import searoute
from shapely.geometry import Point

import common

# WPI column names (checked against the Pub 150 CSV header).
WPI_NUMBER = "World Port Index Number"
WPI_NAME = "Main Port Name"
WPI_COUNTRY = "Country Code"     # despite the name, this holds country names
WPI_LOCODE = "UN/LOCODE"
WPI_SIZE = "Harbor Size"
WPI_LAT = "Latitude"
WPI_LON = "Longitude"

# PortWatch Ports.csv column names (checked against the file header).
PW_ID = "portid"
PW_NAME = "portname"
PW_ISO3 = "ISO3"
PW_LAT = "lat"
PW_LON = "lon"
PW_LOCODE = "LOCODE"
PW_TOTAL = "vessel_count_total"
PW_TANKER = "vessel_count_tanker"
PW_CARGO = ["vessel_count_container", "vessel_count_general_cargo"]
PW_SHARE = {"export": "share_country_maritime_export", "import": "share_country_maritime_import"}

SIDES = ("export", "import")
GATEWAY_COLUMNS = ["country_iso3", "gateway_port", "gateway_wpi_number", "gateway_portid",
                   "gateway_country", "distance_km", "note"]
AUTO_NOTE = "auto first pass - review"
CENTRAL_ASIA_CHECK = ["KAZ", "UZB", "KGZ"]   # least certain gateways; review check only
TOP_TRADERS = ["CHN", "JPN", "USA", "KOR", "MYS", "DEU", "NLD"]   # listed port by port in the log

PORT_FIELDS = ["port_id", "name", "country_iso3", "source", "portwatch_id", "wpi_number", "locode",
               "harbor_size", "is_gateway", "principal_export", "principal_import",
               "share_export_pct", "share_import_pct", "snap_lon", "snap_lat", "snap_dist_km",
               "snap_flag", "use_for_routing", "geometry"]
SHARE_FIELDS = ["country_iso3", "side", "port_id", "rank", "share_raw_pct", "share_filtered_pct",
                "cumulative_pct", "share"]


# ---------------------------------------------------------------------
# Inputs shared by both assignments
# ---------------------------------------------------------------------

def trading_countries(settings, log):
    """Main-run flows with seaborne trade, and the countries that are their origin or destination."""
    flows = common.read_layer(settings, "trade_flows")
    keep = flows["in_main_run"] & (flows["maritime_share"] > 0)
    common.log_filter(log, "Task 2.10 | trade flows in main run with maritime_share > 0",
                      int(keep.sum()), int((~keep).sum()),
                      "not in main run, or no seaborne share")
    flows = flows[keep]
    countries = sorted(set(flows["origin_iso3"]) | set(flows["dest_iso3"]))
    log.info("Task 2.10 | trading countries (origin or destination): %d (exporters %d, importers %d)",
             len(countries), flows["origin_iso3"].nunique(), flows["dest_iso3"].nunique())
    return countries, flows


def read_countries(settings, log):
    """Natural Earth countries: name fields for matching WPI names, and one polygon per ISO-3."""
    field = settings["trade"]["transport_mode"]["border_iso_field"]
    name_fields = settings["ports"]["wpi_name_fields"]
    path = common.get_path(settings, "raw_files.natural_earth_countries")
    countries = pyogrio.read_dataframe(path, columns=[field] + name_fields)
    log.info("Read %s (%d features)", path.name, len(countries))
    countries = countries[countries[field] != "-99"]
    return countries.rename(columns={field: "iso3"})


def wpi_name_lookup(countries, settings):
    """Country name -> ISO-3, from the Natural Earth name fields, then the settings overrides."""
    lookup = {}
    for name_field in settings["ports"]["wpi_name_fields"]:
        for name, iso3 in zip(countries[name_field], countries["iso3"]):
            lookup.setdefault(str(name).strip(), iso3)
    lookup.update(settings["ports"]["wpi_country_iso3_overrides"] or {})
    return lookup


def read_wpi(settings, countries, log):
    """Task 2.10: read the WPI and give every port the ISO-3 code of its country."""
    path = common.get_path(settings, "raw_files.world_port_index")
    wpi = pd.read_csv(path, low_memory=False,
                      usecols=[WPI_NUMBER, WPI_NAME, WPI_COUNTRY, WPI_LOCODE, WPI_SIZE, WPI_LAT, WPI_LON])
    log.info("Read %s: %s ports", path.name, f"{len(wpi):,}")
    wpi[WPI_COUNTRY] = wpi[WPI_COUNTRY].astype("string").str.strip()
    wpi[WPI_SIZE] = wpi[WPI_SIZE].astype("string").str.strip()

    # Task 2.10: country names -> ISO-3 through Natural Earth names and the overrides.
    lookup = wpi_name_lookup(countries, settings)
    wpi["country_iso3"] = wpi[WPI_COUNTRY].map(lookup)
    overrides = set(settings["ports"]["wpi_country_iso3_overrides"] or {})
    log.info("Task 2.10 | WPI country names %d | matched to a Natural Earth name %d | "
             "matched through settings overrides %d",
             wpi[WPI_COUNTRY].nunique(),
             wpi.loc[~wpi[WPI_COUNTRY].isin(overrides) & wpi["country_iso3"].notna(), WPI_COUNTRY].nunique(),
             wpi.loc[wpi[WPI_COUNTRY].isin(overrides), WPI_COUNTRY].nunique())
    unmatched = wpi["country_iso3"].isna()
    common.log_filter(log, "Task 2.10 | WPI ports with an ISO-3 country", int((~unmatched).sum()),
                      int(unmatched.sum()), "country name not matched: "
                      + (", ".join(sorted(wpi.loc[unmatched, WPI_COUNTRY].unique())) or "none"))
    wpi = wpi[~unmatched]

    # Cross-check: the first two letters of the UN/LOCODE are the ISO-2 country code.
    iso2_to_iso3 = dict(zip(*pyogrio.read_dataframe(
        common.get_path(settings, "raw_files.natural_earth_countries"),
        columns=["ISO_A2_EH", "ISO_A3_EH"], read_geometry=False).values.T))
    locode_iso3 = wpi[WPI_LOCODE].astype("string").str.strip().str[:2].map(iso2_to_iso3)
    disagree = locode_iso3.notna() & (locode_iso3 != wpi["country_iso3"])
    log.info("Task 2.10 | check: ports whose UN/LOCODE country differs from the WPI country: %d "
             "(kept under the WPI country)%s", int(disagree.sum()),
             "".join(f"\n    {r[WPI_NAME]} ({r[WPI_COUNTRY]}, LOCODE {r[WPI_LOCODE]})"
                     for _, r in wpi[disagree].iterrows()))

    # Task 2.10: drop ports without a harbor size (needed to pick a country's largest class).
    sizes = settings["ports"]["wpi_size_fallback_order"]
    no_size = ~wpi[WPI_SIZE].isin(sizes)
    common.log_filter(log, "Task 2.10 | WPI ports with a harbor size", int((~no_size).sum()),
                      int(no_size.sum()), "harbor size blank")
    return wpi[~no_size].copy()


def read_gateway_file(settings, log):
    """Task 2.12: the reviewed gateway table, or an empty one on the first run."""
    path = common.ROOT / settings["ports"]["landlocked_gateways_file"]
    if not path.exists():
        log.info("Task 2.12 | %s not found; every gateway will be chosen automatically", path.name)
        return pd.DataFrame(columns=GATEWAY_COLUMNS)
    table = pd.read_csv(path, dtype={"country_iso3": str, "gateway_port": str, "gateway_portid": str,
                                     "gateway_country": str, "note": str})
    missing = [c for c in GATEWAY_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    dupes = table.loc[table["country_iso3"].duplicated(), "country_iso3"].tolist()
    if dupes:
        raise ValueError(f"Duplicate countries in {path.name}: {dupes}")
    log.info("Read %s: %d rows (%s)", path.name, len(table),
             ", ".join(f"{note}: {n}" for note, n in table["note"].value_counts().items()))
    return table[GATEWAY_COLUMNS]


def write_gateway_file(table, settings, log):
    """Task 2.12: write the gateway table back (existing rows unchanged, new rows at the end)."""
    path = common.ROOT / settings["ports"]["landlocked_gateways_file"]
    table.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(table))


def largest_size_class(own, settings):
    """A country's WPI harbors of the largest size class it has (empty if it has none)."""
    for size in settings["ports"]["wpi_size_fallback_order"]:
        if (own[WPI_SIZE] == size).any():
            return own[own[WPI_SIZE] == size], size
    return own.iloc[0:0], None


def country_centroids(countries):
    """Centroid of each country's (dissolved) Natural Earth polygon, as lon/lat.

    Computed on lon/lat coordinates: used only to pick the nearest port, and the
    countries it is used for (landlocked ones) are compact enough for this.
    """
    dissolved = countries[["iso3", "geometry"]].dissolve(by="iso3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # "geographic CRS" centroid warning
        centroids = dissolved.geometry.centroid
    return pd.DataFrame({"lon": centroids.x, "lat": centroids.y})


def nearest_gateway(iso3, pool, centroids, settings):
    """Task 2.12 (auto): the nearest foreign harbor in the pool to a country's centroid."""
    foreign = pool[pool["country_iso3"] != iso3]
    distance = common.great_circle_km(centroids.at[iso3, "lon"], centroids.at[iso3, "lat"],
                                      foreign[WPI_LON], foreign[WPI_LAT], settings)
    best = foreign.iloc[int(np.argmin(distance))]
    return {"country_iso3": iso3, "gateway_port": best[WPI_NAME],
            "gateway_wpi_number": int(best[WPI_NUMBER]), "gateway_portid": None,
            "gateway_country": best["country_iso3"],
            "distance_km": round(float(np.min(distance)), 1), "note": AUTO_NOTE}


def append_gateways(table, new, wpi, countries, settings, log):
    """Task 2.12: append the nearest WPI gateway for trading countries with no port of their own."""
    gateway_sizes = settings["ports"]["gateway_harbor_sizes"]
    if new:
        pool = wpi[wpi[WPI_SIZE].isin(gateway_sizes)].reset_index(drop=True)
        centroids = country_centroids(countries)
        missing = [c for c in new if c not in centroids.index]
        if missing:
            raise ValueError(f"No Natural Earth polygon for {missing}; cannot place a gateway")
        added = pd.DataFrame([nearest_gateway(c, pool, centroids, settings) for c in new],
                             columns=GATEWAY_COLUMNS)
        table = pd.concat([table, added], ignore_index=True)
    log.info("Task 2.12 | gateway rows kept unchanged from the file %d | appended (nearest %s "
             "harbor abroad to the country centroid) %d%s", len(table) - len(new),
             "/".join(gateway_sizes), len(new), f" ({', '.join(new)})" if new else "")
    return table


def gateway_trade(flows, gateways, log):
    """Task 2.12: main-run maritime value carried through each gateway (review aid).

    A flow counts for a gateway when the landlocked country is its origin or
    destination. Shares are of all main-run maritime value.
    """
    total = flows["maritime_value_usd"].sum()
    rows = []
    for g in gateways.itertuples():
        involved = (flows["origin_iso3"] == g.country_iso3) | (flows["dest_iso3"] == g.country_iso3)
        rows.append((g.country_iso3, g.gateway_port, flows.loc[involved, "maritime_value_usd"].sum()))
    rows.sort(key=lambda r: r[2], reverse=True)
    log.info("Task 2.12 | main-run maritime value through each gateway (total %s USD):%s",
             f"{total:,.0f}", "".join(f"\n    {iso3} via {port}: {value:,.0f} USD ({value / total:.3%})"
                                     for iso3, port, value in rows))

    # Review check: the Central Asian gateways are the least certain choices.
    check = [c for c in CENTRAL_ASIA_CHECK if c in set(gateways["country_iso3"])]
    involved = flows["origin_iso3"].isin(check) | flows["dest_iso3"].isin(check)
    share = flows.loc[involved, "maritime_value_usd"].sum() / total
    log.info("Task 2.12 | check: %s together carry %.3f%% of main-run maritime value (%s 1%%)",
             "+".join(check), share * 100, "above" if share > 0.01 else "not above")


def network_nodes(log):
    """Task 2.13: the searoute sea-lane network's nodes as a (lon, lat) array.

    searoute.setup_M() returns the network as a NetworkX graph (searoute Marnet)
    whose nodes are (lon, lat) tuples. searoute itself starts and ends routes at
    the nearest node, so ports are linked to nodes, not to points along edges.
    Node longitudes run past 180 near the date line; they are kept as they are,
    because S06 uses them as node keys.
    """
    network = searoute.setup_M()
    nodes = np.array(list(network.nodes), dtype=float)
    log.info("Task 2.13 | searoute %s sea-lane network: %s nodes, %s edges",
             searoute_version(), f"{network.number_of_nodes():,}", f"{network.number_of_edges():,}")
    return nodes


def searoute_version():
    """Installed searoute version (task 1.7 records it)."""
    from importlib.metadata import version
    return version("searoute")


def snap_to_network(ports, nodes, settings, log):
    """Task 2.13: nearest network node, its great-circle distance and the far-port flag.

    `ports` needs lon and lat columns. Each distinct port is snapped once.
    """
    places = ports.drop_duplicates(["lon", "lat"])[["lon", "lat"]].reset_index(drop=True)
    distance = common.distance_matrix_km(places["lon"], places["lat"], nodes[:, 0], nodes[:, 1], settings)
    nearest = distance.argmin(axis=1)
    places["snap_lon"] = nodes[nearest, 0]
    places["snap_lat"] = nodes[nearest, 1]
    places["snap_dist_km"] = distance[np.arange(len(places)), nearest].round(1)
    ports = ports.merge(places, on=["lon", "lat"], how="left")
    ports["snap_flag"] = ports["snap_dist_km"] > settings["ports"]["snap_warning_km"]
    log.info("Task 2.13 | snap distance km (%d ports): median %.1f | 95th percentile %.1f | max %.1f",
             len(ports), ports["snap_dist_km"].median(), ports["snap_dist_km"].quantile(0.95),
             ports["snap_dist_km"].max())
    return ports


# ---------------------------------------------------------------------
# principal_ports: Tasks 2.10-2.11 (PortWatch ports and shares)
# ---------------------------------------------------------------------

def read_portwatch(settings, log):
    """Task 2.10: PortWatch ports with their country shares (percent) and vessel calls."""
    path = common.get_path(settings, "raw_files.portwatch_ports")
    columns = [PW_ID, PW_NAME, PW_ISO3, PW_LAT, PW_LON, PW_LOCODE, PW_TOTAL, PW_TANKER,
               *PW_CARGO, *PW_SHARE.values()]
    pw = pd.read_csv(path, usecols=columns, encoding="utf-8-sig")
    log.info("Read %s: %s ports in %d countries", path.name, f"{len(pw):,}", pw[PW_ISO3].nunique())
    if pw[PW_ID].duplicated().any():
        raise ValueError(f"Duplicate PortWatch port IDs in {path.name}")
    # Shares are percent of the country's maritime trade; they must sum to about 100 per country.
    for side, col in PW_SHARE.items():
        sums = pw.groupby(PW_ISO3)[col].sum()
        log.info("Task 2.11 | PortWatch %s shares per country sum to %.2f-%.2f%% (percent, 0-100)",
                 side, sums.min(), sums.max())
    return pw


def filter_ports(pw, countries, settings, log):
    """Task 2.10 (Decision 14b): remove offshore oil terminals and mostly-tanker ports.

    Returns the kept ports and the removed ones (with the reason).
    """
    cfg = settings["ports"]
    calls = pw[PW_TOTAL].where(pw[PW_TOTAL] > 0)          # no calls: shares cannot be tested
    tanker_share = pw[PW_TANKER] / calls
    cargo_share = pw[PW_CARGO].sum(axis=1) / calls
    offshore = pw[PW_ID].str.startswith(cfg["offshore_terminal_prefix"]) & cfg["exclude_offshore_terminals"]
    tanker = (tanker_share >= cfg["max_tanker_share"]) & (cargo_share < cfg["min_cargo_share"])
    pw = pw.assign(tanker_call_share=tanker_share, cargo_call_share=cargo_share)
    pw["removed_reason"] = np.select([offshore, tanker], ["offshore oil terminal", "mostly tanker"], "")
    removed = pw["removed_reason"] != ""

    log.info("Task 2.10 | port filter: offshore terminals (portid starts '%s') removed: %s | "
             "mostly tanker (tanker calls >= %.0f%% and container + general cargo calls < %.0f%%) "
             "removed: %d | ports with no vessel calls (kept, test not applied): %d",
             cfg["offshore_terminal_prefix"], "yes" if cfg["exclude_offshore_terminals"] else "no",
             cfg["max_tanker_share"] * 100, cfg["min_cargo_share"] * 100,
             int((pw["removed_reason"] == "mostly tanker").sum()), int(calls.isna().sum()))
    common.log_filter(log, "Task 2.10 | PortWatch ports kept by the filter", int((~removed).sum()),
                      int(removed.sum()), f"offshore oil terminal {int(offshore.sum())}, "
                      f"mostly tanker {int((removed & ~offshore).sum())}")
    trading = pw[PW_ISO3].isin(countries)
    log.info("Task 2.10 | in trading countries: ports %d | removed %d (offshore %d, mostly tanker %d)",
             int(trading.sum()), int((removed & trading).sum()), int((offshore & trading).sum()),
             int((removed & ~offshore & trading).sum()))

    # Review list: removed ports with a large share, so no major cargo port is dropped unseen.
    limit = cfg["removed_port_report_pct"]
    big = removed & trading & ((pw[PW_SHARE["export"]] > limit) | (pw[PW_SHARE["import"]] > limit))
    listed = pw[big].sort_values([PW_ISO3, PW_NAME])
    log.info("Task 2.10 | removed ports in trading countries with more than %s%% of their country's "
             "export or import share: %d (calls: container / general cargo / tanker of total)%s",
             limit, len(listed), "".join(
                 f"\n    {r[PW_ISO3]} {r[PW_ID]} {r[PW_NAME]}: export {r[PW_SHARE['export']]:.1f}%, "
                 f"import {r[PW_SHARE['import']]:.1f}% | {r['removed_reason']} | calls "
                 f"{r['vessel_count_container']:,} / {r['vessel_count_general_cargo']:,} / "
                 f"{r[PW_TANKER]:,} of {r[PW_TOTAL]:,}" for _, r in listed.iterrows()))
    return pw[~removed].copy(), pw[removed].copy()


def principal_ports(kept, unfiltered, country, side, settings, log):
    """Task 2.11: one country's principal ports on one side (export or import).

    The filtered shares are rescaled to 100% and ports taken largest first until the
    running total reaches ports.principal_port_share. If the filter removed every port
    with a share on this side, the unfiltered ports are used (logged).
    """
    col = PW_SHARE[side]
    own = kept[(kept[PW_ISO3] == country) & (kept[col] > 0)]
    if own.empty:
        own = unfiltered[(unfiltered[PW_ISO3] == country) & (unfiltered[col] > 0)]
        log.info("Task 2.11 | %s %s: the filter removed every port with a share; unfiltered ports used",
                 country, side)
    if own.empty:
        return own
    own = own.sort_values([col, PW_ID], ascending=[False, True]).copy()
    own["share_raw_pct"] = own[col]
    own["share_filtered_pct"] = 100 * own[col] / own[col].sum()
    own["cumulative_pct"] = own["share_filtered_pct"].cumsum()
    before = own["cumulative_pct"] - own["share_filtered_pct"]
    own = own[before < settings["ports"]["principal_port_share"] * 100 - 1e-9].copy()
    own["rank"] = np.arange(1, len(own) + 1)
    own["side"] = side
    return own


def principal_shares(pw, kept, coastal, exporters, importers, settings, log):
    """Task 2.11: principal ports of every coastal trading country PortWatch covers."""
    tables, missing = [], []
    for country in coastal:
        for side, traders in (("export", exporters), ("import", importers)):
            if country not in traders:
                continue
            rows = principal_ports(kept, pw, country, side, settings, log)
            if rows.empty:
                missing.append(f"{country} {side}")
            tables.append(rows)
    if missing:
        raise ValueError(f"PortWatch gives no port with a share for: {missing}")
    shares = pd.concat(tables, ignore_index=True)
    shares["country_iso3"] = shares[PW_ISO3]
    shares["port_id"] = shares["country_iso3"] + "_" + shares[PW_ID]
    for side in SIDES:
        n = shares[shares["side"] == side].groupby("country_iso3").size()
        log.info("Task 2.11 | principal %s ports (%.0f%% rule): %d for %d countries | per country "
                 "min %d, median %.0f, max %d", side, settings["ports"]["principal_port_share"] * 100,
                 int(n.sum()), len(n), n.min(), n.median(), n.max())
    return shares


def wpi_fallback(wpi, countries, exporters, importers, settings, log):
    """Task 2.10 (Decision 14g): WPI ports, equal shares, for trading countries PortWatch lacks."""
    shares, ports = [], []
    for country in countries:
        own, size = largest_size_class(wpi[wpi["country_iso3"] == country], settings)
        if own.empty:
            raise ValueError(f"{country} has no PortWatch or WPI port; add it to the gateway file")
        log.info("Task 2.10 | %s not in PortWatch: %d WPI %s harbor(s), equal shares (%s)", country,
                 len(own), size, ", ".join(own[WPI_NAME]))
        own = own.assign(is_gateway=False)
        rows = wpi_port_rows(own)
        ports.append(rows)
        n = len(rows)
        for side, traders in (("export", exporters), ("import", importers)):
            if country in traders:
                shares.append(pd.DataFrame({
                    "country_iso3": country, "side": side, "port_id": rows["port_id"].values,
                    "rank": np.arange(1, n + 1), "share_raw_pct": np.nan,
                    "share_filtered_pct": 100 / n, "cumulative_pct": 100 * np.arange(1, n + 1) / n}))
    if not ports:
        return pd.DataFrame(columns=SHARE_FIELDS), pd.DataFrame()
    return pd.concat(shares, ignore_index=True), pd.concat(ports, ignore_index=True)


def portwatch_gateways(table, landlocked, exporters, importers, pw, removed, log):
    """Task 2.12 (Decision 14e): the PortWatch gateway port of each landlocked trading country.

    Gateways come from gateway_portid and are never filtered; share 1 on each side traded.
    """
    rows = table[table["country_iso3"].isin(landlocked)]
    blank = rows.loc[rows["gateway_portid"].isna(), "country_iso3"].tolist()
    if blank:
        raise ValueError(f"Gateway file rows without gateway_portid (add the PortWatch port): {blank}")
    by_id = pw.set_index(PW_ID)
    unknown = sorted(set(rows["gateway_portid"]) - set(by_id.index))
    if unknown:
        raise ValueError(f"gateway_portid values not in PortWatch: {unknown}")
    wrong = rows[by_id.loc[rows["gateway_portid"], PW_ISO3].values != rows["gateway_country"].values]
    if len(wrong):
        raise ValueError(f"gateway_portid in another country than gateway_country: {wrong['country_iso3'].tolist()}")

    filtered_out = sorted(set(rows["gateway_portid"]) & set(removed[PW_ID]))
    log.info("Task 2.12 | gateways in use: %d (PortWatch ports; %d of them would be removed by the "
             "port filter, kept because gateways are never filtered%s)%s", len(rows), len(filtered_out),
             f": {', '.join(filtered_out)}" if filtered_out else "",
             "".join(f"\n    {r.country_iso3} -> {r.gateway_port} ({r.gateway_portid}, {r.gateway_country}), "
                     f"{r.distance_km:,.0f} km [{r.note}]" for r in rows.itertuples()))

    port = by_id.loc[rows["gateway_portid"]]
    ports = pd.DataFrame({
        "port_id": (rows["country_iso3"] + "_" + rows["gateway_portid"]).values,
        "name": port[PW_NAME].values, "country_iso3": rows["country_iso3"].values,
        "source": "portwatch", "portwatch_id": rows["gateway_portid"].values, "wpi_number": np.nan,
        "locode": port[PW_LOCODE].values, "harbor_size": None, "is_gateway": True,
        "lon": port[PW_LON].values, "lat": port[PW_LAT].values})
    shares = [pd.DataFrame({"country_iso3": c, "side": side, "port_id": p, "rank": 1,
                            "share_raw_pct": np.nan, "share_filtered_pct": 100.0,
                            "cumulative_pct": 100.0}, index=[0])
              for c, p in zip(ports["country_iso3"], ports["port_id"])
              for side, traders in (("export", exporters), ("import", importers)) if c in traders]
    return ports, pd.concat(shares, ignore_index=True)


def portwatch_port_rows(shares, pw):
    """One ports-layer row per principal PortWatch port (country, port)."""
    ids = shares[["country_iso3", "port_id", PW_ID]].drop_duplicates("port_id")
    port = pw.set_index(PW_ID).loc[ids[PW_ID]]
    return pd.DataFrame({
        "port_id": ids["port_id"].values, "name": port[PW_NAME].values,
        "country_iso3": ids["country_iso3"].values, "source": "portwatch",
        "portwatch_id": ids[PW_ID].values, "wpi_number": np.nan, "locode": port[PW_LOCODE].values,
        "harbor_size": None, "is_gateway": False,
        "lon": port[PW_LON].values, "lat": port[PW_LAT].values})


def far_port_rule(shares, ports, settings, log):
    """Task 2.13 (Decision 14f): keep and flag far ports; drop only the hand-checked exclusions.

    Ports farther than ports.snap_warning_km stay in use (the network is sparse near some
    coasts). Ports listed in ports.snap_exclude (nearest node in another sea) are dropped
    and the remaining ports' shares on that side are rescaled. Returns the final shares
    (share sums to 1 per country and side).
    """
    info = ports.drop_duplicates("port_id").set_index("port_id")
    far = shares["port_id"].map(info["snap_flag"])
    log.info("Task 2.13 | far principal ports (> %s km), kept for routing and flagged: %d ports, "
             "%d share rows (filtered share of the country's side)%s",
             settings["ports"]["snap_warning_km"], shares.loc[far, "port_id"].nunique(), int(far.sum()),
             "".join(f"\n    {r.port_id} {info.at[r.port_id, 'name']} ({r.country_iso3} {r.side}): "
                     f"{r.share_filtered_pct:.1f}%, {info.at[r.port_id, 'snap_dist_km']:,.1f} km"
                     for r in shares[far].sort_values(["country_iso3", "side"]).itertuples()))

    exclude = settings["ports"]["snap_exclude"] or {}
    unused = sorted(set(exclude) - set(shares["port_id"]))
    if unused:
        log.info("Task 2.13 | snap_exclude ports that are not principal ports (nothing to drop): %s",
                 ", ".join(unused))
    drop = shares["port_id"].isin(exclude)
    left = (~drop).groupby([shares["country_iso3"], shares["side"]]).transform("any")
    if (drop & ~left).any():
        raise ValueError(f"snap_exclude leaves no port for: "
                         f"{sorted(set(shares.loc[drop & ~left, 'country_iso3']))}")
    log.info("Task 2.13 | ports dropped by snap_exclude (nearest node in another sea; share moved to "
             "the country's other principal ports on that side): %d share rows%s", int(drop.sum()),
             "".join(f"\n    {r.port_id} {info.at[r.port_id, 'name']} ({r.country_iso3} {r.side}): "
                     f"{r.share_filtered_pct:.1f}% moved, {info.at[r.port_id, 'snap_dist_km']:,.1f} km "
                     f"| {exclude[r.port_id]}" for r in shares[drop].itertuples()))

    shares = shares[~drop].copy()
    shares["share"] = shares["share_filtered_pct"] / shares.groupby(
        ["country_iso3", "side"])["share_filtered_pct"].transform("sum")
    return shares


def zone_check(ports, settings, log):
    """Task 2.13 (Decision 10c): principal ports whose network point lies inside a circle zone.

    Same test as S08 (ports_in_zone): gated chokepoints (Decision 12) have no port inside.
    Logged only; S06 and S08 apply the consequences.
    """
    zones = common.read_layer(settings, "zones")
    zones = zones[zones["included"].str.strip().str.lower() == "yes"]
    shapes = common.crossing_geometries(zones[["chokepoint_id", "name", "geometry"]], settings)
    shapes = shapes[shapes["crossing_method"] == "zone"]
    used = ports[ports["use_for_routing"]].drop_duplicates("port_id")
    hits = []
    for shape in shapes.itertuples():
        for r in used.itertuples():
            lon = r.snap_lon if r.snap_lon <= 180 else r.snap_lon - 360
            if shape.geometry.contains(Point(lon, r.snap_lat)):
                hits.append((shape.name, r))
    log.info("Task 2.13 | ports in use whose network point is inside an included circle zone "
             "(Decision 10c; S08 leaves their flows out of Equation 4): %d%s", len(hits), "".join(
                 f"\n    {name}: {r.port_id} {r.name} (export {r.share_export_pct:.1f}%, "
                 f"import {r.share_import_pct:.1f}% of the country, raw)" if r.source == "portwatch"
                 and not r.is_gateway else f"\n    {name}: {r.port_id} {r.name}" for name, r in hits))


def log_principal(shares, ports, log):
    """Principal ports of the largest traders, port by port, and counts for every country."""
    names = ports.drop_duplicates("port_id").set_index("port_id")["name"]
    for country in TOP_TRADERS:
        for side in SIDES:
            rows = shares[(shares["country_iso3"] == country) & (shares["side"] == side)]
            if len(rows):
                log.info("Task 2.11 | %s %s: %d ports, %.1f%% of the filtered share: %s", country, side,
                         len(rows), rows["share_filtered_pct"].sum(), ", ".join(
                             f"{names[p]} {s:.1%}" for p, s in zip(rows["port_id"], rows["share"])))
    counts = shares.pivot_table(index="country_iso3", columns="side", values="port_id",
                                aggfunc="count", fill_value=0)
    log.info("Task 2.11 | principal ports per country (export/import):%s", "".join(
        f"\n    {line}" for line in _wrap([f"{c} {int(r.get('export', 0))}/{int(r.get('import', 0))}"
                                         for c, r in counts.iterrows()], 12)))


def _wrap(items, per_line):
    """Join items into lines of per_line items."""
    return [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]


def run_principal_ports(settings, log, countries, flows, ne, wpi, gateways, nodes):
    """principal_ports assignment (main run): ports layer and country_port_shares table."""
    exporters, importers = set(flows["origin_iso3"]), set(flows["dest_iso3"])
    pw = read_portwatch(settings, log)

    # Task 2.12: listed countries are landlocked even when PortWatch has ports for them.
    landlocked = sorted(set(gateways["country_iso3"]) & set(countries))
    with_pw = [c for c in landlocked if c in set(pw[PW_ISO3])]
    coastal = [c for c in countries if c not in set(landlocked) and c in set(pw[PW_ISO3])]
    fallback = [c for c in countries if c not in set(landlocked) and c not in set(pw[PW_ISO3])]
    log.info("Task 2.10 | trading countries: PortWatch %d | WPI fallback (not in PortWatch) %d (%s) | "
             "landlocked, gateway file %d (of them with PortWatch ports, still treated as landlocked: %s)",
             len(coastal), len(fallback), ", ".join(fallback) or "none", len(landlocked),
             ", ".join(with_pw) or "none")

    # Tasks 2.10-2.11: filter, then the 80% rule on each side.
    kept, removed = filter_ports(pw, countries, settings, log)
    shares = principal_shares(pw, kept, coastal, exporters, importers, settings, log)
    ports = [portwatch_port_rows(shares, pw)]

    # Task 2.10: WPI fallback; Task 2.12: gateways.
    wpi_shares, wpi_ports = wpi_fallback(wpi, fallback, exporters, importers, settings, log)
    gw_ports, gw_shares = portwatch_gateways(gateways, landlocked, exporters, importers, pw, removed, log)
    gateway_trade(flows, gateways[gateways["country_iso3"].isin(countries)], log)
    shares = pd.concat([shares, wpi_shares, gw_shares], ignore_index=True)
    ports = pd.concat(ports + [wpi_ports, gw_ports], ignore_index=True)
    if ports["port_id"].duplicated().any():
        raise ValueError(f"Duplicate port IDs: {ports.loc[ports['port_id'].duplicated(), 'port_id'].tolist()}")

    # Raw PortWatch shares on every PortWatch port row (null for WPI and gateways), and the
    # sides on which each port is principal.
    raw = pw.set_index(PW_ID)
    own_pw = (ports["source"] == "portwatch") & ~ports["is_gateway"].astype(bool)
    for side in SIDES:
        ports[f"share_{side}_pct"] = np.where(own_pw, ports["portwatch_id"].map(raw[PW_SHARE[side]]), np.nan)
        on_side = set(shares.loc[shares["side"] == side, "port_id"])
        ports[f"principal_{side}"] = ports["port_id"].isin(on_side)

    # Task 2.13: snap, then the far-port rule.
    ports = snap_to_network(ports, nodes, settings, log)
    shares = far_port_rule(shares, ports, settings, log)
    ports["use_for_routing"] = ports["port_id"].isin(set(shares["port_id"]))
    zone_check(ports, settings, log)
    log_principal(shares, ports, log)

    write_gateway_file(gateways, settings, log)
    layer = to_layer(ports, settings)
    shares = shares[SHARE_FIELDS].sort_values(["country_iso3", "side", "rank"]).reset_index(drop=True)
    check_shares(shares, layer, countries, exporters, importers, log)
    log_summary(layer, countries, log)
    common.write_layer(layer, settings, "ports", log)
    common.write_layer(shares, settings, "country_port_shares", log)


def check_shares(shares, layer, countries, exporters, importers, log):
    """Shares sum to 1 per country and side, every trading side has ports, all ports are in use."""
    sums = shares.groupby(["country_iso3", "side"])["share"].sum()
    if not np.allclose(sums, 1):
        raise ValueError(f"Shares do not sum to 1: {sums[~np.isclose(sums, 1)].to_dict()}")
    have = set(zip(shares["country_iso3"], shares["side"]))
    need = {(c, "export") for c in exporters} | {(c, "import") for c in importers}
    if need - have:
        raise ValueError(f"Trading country sides without ports: {sorted(need - have)}")
    in_use = set(layer.loc[layer["use_for_routing"], "port_id"])
    if not set(shares["port_id"]) <= in_use:
        raise ValueError("country_port_shares lists ports not in use in the ports layer")
    log.info("Check | country_port_shares: %d rows, %d country-sides, each sums to 1; every exporter "
             "(%d) and importer (%d) has ports", len(shares), len(sums), len(exporters), len(importers))


# ---------------------------------------------------------------------
# nearest_gateway: the Lab 7/8 first pass (S10 alternative)
# ---------------------------------------------------------------------

def candidate_ports(wpi, countries, listed, settings, log):
    """Task 2.10: for each trading country, its WPI ports of the largest size class present.

    Countries listed in the gateway file are skipped: they are always treated as
    landlocked (task 2.12) and use their gateway instead.
    """
    overridden = [c for c in countries if c in listed and (wpi["country_iso3"] == c).any()]
    log.info("Task 2.12 | countries with WPI ports treated as landlocked (listed in the gateway "
             "file): %d (%s)", len(overridden), ", ".join(overridden) or "none")

    sizes = settings["ports"]["wpi_size_fallback_order"]
    chosen, level_used, no_port = [], {}, []
    for iso3 in countries:
        if iso3 in listed:
            continue
        own, size = largest_size_class(wpi[wpi["country_iso3"] == iso3], settings)
        if own.empty:
            no_port.append(iso3)
        else:
            chosen.append(own)
            level_used[iso3] = size

    for size in sizes:
        users = sorted(c for c, s in level_used.items() if s == size)
        log.info("Task 2.10 | countries using %s harbors: %d%s", size, len(users),
                 f" ({', '.join(users)})" if size != sizes[0] and users else "")
    log.info("Task 2.10 | trading countries not in the gateway file with no WPI port: %d (%s)",
             len(no_port), ", ".join(no_port) or "none")

    ports = pd.concat(chosen, ignore_index=True)
    ports["is_gateway"] = False
    return ports, no_port


def wpi_gateways(table, traded, wpi, log):
    """Task 2.12: one WPI gateway port row for each trading country in the gateway table."""
    unused = sorted(set(table["country_iso3"]) - set(traded))
    if unused:
        log.info("Task 2.12 | gateway rows for countries not trading in the main run (kept, not used): %s",
                 ", ".join(unused))

    # Gateway ports are looked up by WPI number, so a hand-edited row needs only a valid number.
    by_number = wpi.drop_duplicates(WPI_NUMBER).set_index(WPI_NUMBER)
    unknown = sorted(set(table["gateway_wpi_number"].astype(int)) - set(by_number.index))
    if unknown:
        raise ValueError(f"Gateway WPI numbers not in the World Port Index: {unknown}")

    in_use = table[table["country_iso3"].isin(traded)]
    ports = []
    for row in in_use.itertuples():
        port = by_number.loc[int(row.gateway_wpi_number)].copy()
        port[WPI_NUMBER] = int(row.gateway_wpi_number)
        port["country_iso3"] = row.country_iso3      # the port serves the landlocked country
        port["is_gateway"] = True
        ports.append(port)
    log.info("Task 2.12 | gateways in use: %d%s", len(in_use),
             "".join(f"\n    {r.country_iso3} -> {r.gateway_port} ({r.gateway_country}), "
                     f"{r.distance_km:,.0f} km [{r.note}]" for r in in_use.itertuples()))
    return pd.DataFrame(ports)


def wpi_port_rows(ports):
    """Shape WPI candidate and gateway ports into ports-layer rows."""
    return pd.DataFrame({
        "port_id": (ports["country_iso3"] + "_WPI" + ports[WPI_NUMBER].astype(int).astype(str)).values,
        "name": ports[WPI_NAME].values, "country_iso3": ports["country_iso3"].values, "source": "wpi",
        "portwatch_id": None, "wpi_number": ports[WPI_NUMBER].astype(int).values,
        "locode": ports[WPI_LOCODE].values, "harbor_size": ports[WPI_SIZE].values,
        "is_gateway": ports["is_gateway"].astype(bool).values, "principal_export": True,
        "principal_import": True, "share_export_pct": np.nan, "share_import_pct": np.nan,
        "lon": ports[WPI_LON].values, "lat": ports[WPI_LAT].values})


def run_nearest_gateway(settings, log, countries, flows, ne, wpi, gateways, nodes):
    """nearest_gateway assignment (S10 alternative): WPI candidate ports, ports layer only."""
    own_ports, no_port = candidate_ports(wpi, countries, set(gateways["country_iso3"]), settings, log)
    new = [c for c in no_port if c not in set(gateways["country_iso3"])]
    gateways = append_gateways(gateways, new, wpi, ne, settings, log)
    gateway_ports = wpi_gateways(gateways, countries, wpi, log)
    gateway_trade(flows, gateways[gateways["country_iso3"].isin(countries)], log)
    write_gateway_file(gateways, settings, log)

    ports = wpi_port_rows(pd.concat([own_ports, gateway_ports], ignore_index=True))
    if ports["port_id"].duplicated().any():
        raise ValueError(f"Duplicate port IDs: {ports.loc[ports['port_id'].duplicated(), 'port_id'].tolist()}")
    ports = snap_to_network(ports, nodes, settings, log)

    # Task 2.13: a flagged port is not routed from if its country has an unflagged candidate.
    has_unflagged = (~ports["snap_flag"]).groupby(ports["country_iso3"]).transform("any")
    ports["use_for_routing"] = ~(ports["snap_flag"] & has_unflagged)
    off = ports[~ports["use_for_routing"]]
    log.info("Task 2.13 | ports farther than %s km from the network: %d | set use_for_routing = false "
             "(country has an unflagged candidate): %d%s", settings["ports"]["snap_warning_km"],
             int(ports["snap_flag"].sum()), len(off),
             "".join(f"\n    {r.port_id} {r.name}: {r.snap_dist_km:,.1f} km" for r in off.itertuples()))

    layer = to_layer(ports, settings)
    log_summary(layer, countries, log)
    common.write_layer(layer, settings, "ports", log)
    log.info("country_port_shares not written: the nearest_gateway assignment uses no port shares")


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def to_layer(ports, settings):
    """The ports layer: fixed fields, point geometry in the storage CRS, sorted by port_id."""
    ports = ports.copy()
    for col in ("is_gateway", "principal_export", "principal_import", "snap_flag", "use_for_routing"):
        ports[col] = ports[col].astype(bool)
    ports["wpi_number"] = ports["wpi_number"].astype("Int64")
    geometry = gpd.points_from_xy(ports["lon"], ports["lat"])
    layer = gpd.GeoDataFrame(ports, geometry=geometry, crs=settings["project"]["storage_crs"])
    return layer[PORT_FIELDS].sort_values(["country_iso3", "port_id"]).reset_index(drop=True)


def log_summary(layer, countries, log):
    """Ports per country, so the log shows how many ports S06 will use."""
    used = layer[layer["use_for_routing"]]
    per_country = used.groupby("country_iso3").size().reindex(countries, fill_value=0)
    log.info("Summary | ports %d (in use %d; source portwatch %d, wpi %d; gateway %d) for %d countries | "
             "ports in use per country: min %d, median %.0f, max %d", len(layer), len(used),
             int((layer["source"] == "portwatch").sum()), int((layer["source"] == "wpi").sum()),
             int(layer["is_gateway"].sum()), len(per_country), per_country.min(),
             per_country.median(), per_country.max())
    top = per_country.sort_values(ascending=False).head(10)
    log.info("Summary | most ports in use: %s", ", ".join(f"{c} {n}" for c, n in top.items()))


def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    assignment = settings["ports"]["assignment"]
    runs = {"principal_ports": run_principal_ports, "nearest_gateway": run_nearest_gateway}
    if assignment not in runs:
        raise ValueError(f"Unknown ports.assignment '{assignment}'; use one of {list(runs)}")
    log.info("Port assignment: %s (ports.assignment; Decision 14)", assignment)

    countries, flows = trading_countries(settings, log)
    ne = read_countries(settings, log)
    wpi = read_wpi(settings, ne, log)
    gateways = read_gateway_file(settings, log)
    nodes = network_nodes(log)
    runs[assignment](settings, log, countries, flows, ne, wpi, gateways, nodes)
    log.info("Run finished")


if __name__ == "__main__":
    main()

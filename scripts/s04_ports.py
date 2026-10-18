"""S04: ports, landlocked gateways and links to the sea-lane network (tasks 2.10, 2.12, 2.13).

Lab 7 first pass (ports.assignment_first_pass = nearest_gateway). PortWatch ports are
not downloaded yet, so this pass uses the proposal's Table 4 "nearest gateway"
alternative with World Port Index (WPI) ports:

- Task 2.10: every country in the main run with seaborne trade gets candidate ports:
  its WPI harbors of the largest size class it has (ports.wpi_size_fallback_order).
- Task 2.12 (draft): a trading country with no WPI port gets the nearest foreign
  harbor of ports.gateway_harbor_sizes to its Natural Earth centroid. The choices are
  written to config/landlocked_gateways.csv for review.
- Task 2.13: each port is linked to the nearest node of the searoute sea-lane network
  (great-circle distance); ports farther than ports.snap_warning_km are flagged.

No trade shares are computed (task 2.11 waits for PortWatch): S06 chooses, for each
flow, the closest origin-destination pair of candidate ports.

A port can appear more than once: once for its own country and once for each
landlocked country it serves as a gateway. port_id is "<country_iso3>_WPI<number>",
so it is unique and stable between runs.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import searoute

import common

# WPI column names (checked against the Pub 150 CSV header).
WPI_NUMBER = "World Port Index Number"
WPI_NAME = "Main Port Name"
WPI_COUNTRY = "Country Code"     # despite the name, this holds country names
WPI_LOCODE = "UN/LOCODE"
WPI_SIZE = "Harbor Size"
WPI_LAT = "Latitude"
WPI_LON = "Longitude"

GATEWAY_COLUMNS = ["country_iso3", "gateway_port", "gateway_wpi_number", "gateway_country",
                   "distance_km", "note"]


def trading_countries(settings, log):
    """Countries that are an origin or destination of a main-run flow with seaborne trade."""
    flows = common.read_layer(settings, "trade_flows")
    keep = flows["in_main_run"] & (flows["maritime_share"] > 0)
    common.log_filter(log, "Task 2.10 | trade flows in main run with maritime_share > 0",
                      int(keep.sum()), int((~keep).sum()),
                      "not in main run, or no seaborne share")
    flows = flows[keep]
    countries = sorted(set(flows["origin_iso3"]) | set(flows["dest_iso3"]))
    log.info("Task 2.10 | trading countries (origin or destination): %d", len(countries))
    return countries


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

    sizes = settings["ports"]["wpi_size_fallback_order"]
    no_size = ~wpi[WPI_SIZE].isin(sizes)
    common.log_filter(log, "Task 2.10 | WPI ports with a harbor size", int((~no_size).sum()),
                      int(no_size.sum()), "harbor size blank")
    return wpi[~no_size].copy()


def candidate_ports(wpi, countries, settings, log):
    """Task 2.10: for each trading country, its WPI ports of the largest size class present."""
    sizes = settings["ports"]["wpi_size_fallback_order"]
    chosen, level_used, no_port = [], {}, []
    for iso3 in countries:
        own = wpi[wpi["country_iso3"] == iso3]
        for size in sizes:
            if (own[WPI_SIZE] == size).any():
                chosen.append(own[own[WPI_SIZE] == size])
                level_used[iso3] = size
                break
        else:
            no_port.append(iso3)

    for size in sizes:
        users = sorted(c for c, s in level_used.items() if s == size)
        log.info("Task 2.10 | countries using %s harbors: %d%s", size, len(users),
                 f" ({', '.join(users)})" if size != sizes[0] and users else "")
    log.info("Task 2.10 | trading countries with no WPI port: %d (%s)", len(no_port),
             ", ".join(no_port) or "none")

    ports = pd.concat(chosen, ignore_index=True)
    ports["is_gateway"] = False
    return ports, no_port


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


def landlocked_gateways(no_port, wpi, countries, settings, log):
    """Task 2.12 (draft): nearest foreign gateway harbor to each portless country's centroid."""
    gateway_sizes = settings["ports"]["gateway_harbor_sizes"]
    pool = wpi[wpi[WPI_SIZE].isin(gateway_sizes)].reset_index(drop=True)
    centroids = country_centroids(countries)
    missing = [c for c in no_port if c not in centroids.index]
    if missing:
        raise ValueError(f"No Natural Earth polygon for {missing}; cannot place a gateway")

    rows, ports = [], []
    for iso3 in no_port:
        foreign = pool[pool["country_iso3"] != iso3]
        distance = common.great_circle_km(centroids.at[iso3, "lon"], centroids.at[iso3, "lat"],
                                          foreign[WPI_LON], foreign[WPI_LAT], settings)
        best = foreign.iloc[int(np.argmin(distance))]
        rows.append({"country_iso3": iso3, "gateway_port": best[WPI_NAME],
                     "gateway_wpi_number": int(best[WPI_NUMBER]),
                     "gateway_country": best["country_iso3"],
                     "distance_km": round(float(np.min(distance)), 1),
                     "note": "auto first pass - review"})
        port = best.copy()
        port["country_iso3"] = iso3          # the port serves the landlocked country
        port["is_gateway"] = True
        ports.append(port)

    table = pd.DataFrame(rows, columns=GATEWAY_COLUMNS)
    log.info("Task 2.12 | gateways assigned (nearest %s harbor abroad to the country centroid): %d%s",
             "/".join(gateway_sizes), len(table),
             "".join(f"\n    {r.country_iso3} -> {r.gateway_port} ({r.gateway_country}), "
                     f"{r.distance_km:,.0f} km" for r in table.itertuples()))
    return table, pd.DataFrame(ports)


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
    """Task 2.13: nearest network node and its great-circle distance for every port."""
    distance = common.distance_matrix_km(ports[WPI_LON], ports[WPI_LAT],
                                         nodes[:, 0], nodes[:, 1], settings)
    nearest = distance.argmin(axis=1)
    ports["snap_lon"] = nodes[nearest, 0]
    ports["snap_lat"] = nodes[nearest, 1]
    ports["snap_dist_km"] = distance[np.arange(len(ports)), nearest].round(1)

    limit = settings["ports"]["snap_warning_km"]
    ports["snap_flag"] = ports["snap_dist_km"] > limit
    flagged = ports[ports["snap_flag"]].sort_values("snap_dist_km", ascending=False)
    log.info("Task 2.13 | snap distance km: median %.1f | 95th percentile %.1f | max %.1f",
             ports["snap_dist_km"].median(), ports["snap_dist_km"].quantile(0.95),
             ports["snap_dist_km"].max())
    log.info("Task 2.13 | ports farther than %s km from the network (snap_flag): %d of %d%s",
             limit, len(flagged), len(ports),
             "".join(f"\n    {r.port_id} {r.name} ({r.harbor_size}): {r.snap_dist_km:,.1f} km"
                     for r in flagged.itertuples()))
    return ports


def build_layer(ports, settings):
    """Shape the ports table into the fixed layer fields, with point geometry."""
    ports = ports.rename(columns={WPI_NAME: "name", WPI_SIZE: "harbor_size"})
    ports["port_id"] = ports["country_iso3"] + "_WPI" + ports[WPI_NUMBER].astype(int).astype(str)
    if ports["port_id"].duplicated().any():
        raise ValueError(f"Duplicate port IDs: {ports.loc[ports['port_id'].duplicated(), 'port_id'].tolist()}")
    ports["is_gateway"] = ports["is_gateway"].astype(bool)
    geometry = gpd.points_from_xy(ports[WPI_LON], ports[WPI_LAT])
    layer = gpd.GeoDataFrame(ports, geometry=geometry, crs=settings["project"]["storage_crs"])
    return layer.sort_values(["country_iso3", "port_id"]).reset_index(drop=True)


def log_summary(layer, countries, log):
    """Ports per country, so the log shows how many candidates S06 will compare."""
    per_country = layer.groupby("country_iso3").size().reindex(countries, fill_value=0)
    log.info("Summary | ports %d (own %d, gateway %d) for %d countries | ports per country: "
             "min %d, median %.0f, max %d", len(layer), int((~layer["is_gateway"]).sum()),
             int(layer["is_gateway"].sum()), len(per_country), per_country.min(),
             per_country.median(), per_country.max())
    counts = per_country.value_counts().sort_index()
    log.info("Summary | countries by number of ports: %s",
             ", ".join(f"{n} port{'s' if n != 1 else ''}: {c}" for n, c in counts.items()))
    top = per_country.sort_values(ascending=False).head(10)
    log.info("Summary | most ports: %s", ", ".join(f"{c} {n}" for c, n in top.items()))


def main():
    settings = common.load_settings()
    log = common.start_log(Path(__file__).stem, settings)
    log.info("Port assignment: %s (Lab 7 first pass, WPI ports)",
             settings["ports"]["assignment_first_pass"])

    countries = trading_countries(settings, log)
    ne = read_countries(settings, log)
    wpi = read_wpi(settings, ne, log)

    # Task 2.10: candidate ports; Task 2.12: gateways for countries without one.
    own_ports, no_port = candidate_ports(wpi, countries, settings, log)
    gateways, gateway_ports = landlocked_gateways(no_port, wpi, ne, settings, log)
    ports = pd.concat([own_ports, gateway_ports], ignore_index=True)

    gateway_file = common.ROOT / settings["ports"]["landlocked_gateways_file"]
    gateways.to_csv(gateway_file, index=False)
    log.info("Wrote %s (%d rows)", gateway_file, len(gateways))

    # Task 2.13: link each port to the sea-lane network.
    layer = build_layer(ports, settings)
    layer = snap_to_network(layer, network_nodes(log), settings, log)
    layer = layer[["port_id", "name", "country_iso3", "harbor_size", "is_gateway",
                   "snap_lon", "snap_lat", "snap_dist_km", "snap_flag", "geometry"]]

    log_summary(layer, countries, log)
    common.write_layer(layer, settings, "ports", log)
    log.info("Run finished")


if __name__ == "__main__":
    main()

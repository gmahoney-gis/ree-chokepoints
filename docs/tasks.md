# Task list

<!-- Generated from GEOG797_Analysis_Tracker.xlsx (13 Oct 2026). Task IDs match the tracker and the Analysis Plan of Attack. Track progress in the spreadsheet; this file is the specification. -->

Each script builds the tasks listed for it. "L7" marks tasks needed for the Lab 7 first pass. Product IDs (P01–P41) are listed at the end with their file or layer names and where they appear in the proposal.

## Script index

| Script | Tasks |
|---|---|
| common.py | 0.4 |
| run_all.py | 0.5 |
| S01 | 2.1, 2.2 |
| S02 | 2.3, 2.4, 2.5, 2.6 |
| S03 | 2.9 |
| S04 | 2.10, 2.11, 2.12, 2.13 |
| S05 | 2.14, 2.15, 2.16, 2.17, 2.18 |
| S01–S05 | 3.1 |
| check_db.py | 3.2 |
| S06 | 4.1, 4.2, 4.3, 4.4, 4.5 |
| S07 | 4.7, 4.8, 4.9, 4.10, 4.11 |
| S08 | 4.12, 4.13, 4.14 |
| S09 | 5.1, 5.2, 5.3, 5.4 |
| S10 | 5.5, 5.6, 5.7 |
| S11 | 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9 |

Tasks with script "—" are done by hand (downloads, checks, QGIS work, writing).

## Phase 0: Set up

| ID | Task | Script | Kind | Product | Depends on | L7 |
|---|---|---|---|---|---|---|
| 0.1 | Create a public GitHub repository with the agreed folder layout; add a .gitignore that excludes raw data and the project database. | — | Manual | P01 Project repository and Python environment | — | L7 |
| 0.2 | Create the Python environment (Python 3.11 with pandas, NumPy, GeoPandas, Shapely, pyogrio, searoute, NetworkX, SciPy, scikit-learn, Matplotlib, PyYAML, requests, comtradeapicall) and export it. | — | Manual | P01 Project repository and Python environment | 0.1 | L7 |
| 0.3 | Write the settings file with every parameter on the Settings sheet. | — | Manual | P02 Settings file | 0.1 | L7 |
| 0.4 | Write shared tools: load settings, start a log, great-circle distance, ISO-3 codes, read and write GeoPackage layers. | common.py | Script | P03 Shared tools and runner | 0.2 | L7 |
| 0.5 | Write the runner that runs S01–S11 in order, or one script by name. | run_all.py | Script | P03 Shared tools and runner | 0.4 | L7 |

## Phase 1: Gather

| ID | Task | Script | Kind | Product | Depends on | L7 |
|---|---|---|---|---|---|---|
| 1.1 | List every raw file with its source, URL, download date, and size. | — | Manual | P04 Data manifest and field checks | 0.1 | L7 |
| 1.2 | Download PortWatch port-level trade estimates for 2025 (all ports). | — | Manual | P05 PortWatch port-level trade data | — | L7 |
| 1.3 | Confirm the PortWatch chokepoint transits are broken down by vessel type (container, general cargo, tanker). | — | Manual | P04 Data manifest and field checks | — | L7 |
| 1.4 | Check whether the existing Comtrade download includes mode of transport; if not, plan a re-download through S01. | — | Manual | P04 Data manifest and field checks | — | L7 |
| 1.5 | Check that the GDELT data include the source-count and location-type fields and that the file size is manageable. | — | Manual | P04 Data manifest and field checks | — | L7 |
| 1.6 | Check that the ACLED export includes event type, fatalities, and the location-precision code. | — | Manual | P04 Data manifest and field checks | — | L7 |
| 1.7 | Record the World Port Index edition and the searoute version. | — | Manual | P04 Data manifest and field checks | — |  |

## Phase 2: Prepare

| ID | Task | Script | Kind | Product | Depends on | L7 |
|---|---|---|---|---|---|---|
| 2.1 | Download 2025 Comtrade imports and exports for all reporters (HS 2805.30, 2846.10, 2846.90; 8505.11 for the sensitivity test), with transport mode where reported. | S01 | Script | P06 Raw Comtrade downloads (2025; 2024 totals) | 0.5, 1.4 | L7 |
| 2.2 | Download 2024 totals for the same codes to identify the ten largest REE traders. | S01 | Script | P06 Raw Comtrade downloads (2025; 2024 totals) | 2.1 | L7 |
| 2.3 | Convert country codes to ISO-3; remove aggregate rows and log their value. | S02 | Script | P07 Reconciled trade flows | 2.1 | L7 |
| 2.4 | Reconcile flows: importer record first, exporter mirror record fills gaps; flag every fill. | S02 | Script | P07 Reconciled trade flows | 2.3 | L7 |
| 2.5 | Classify transport mode: reported mode where available, otherwise the shared-land-border rule. | S02 | Script | P07 Reconciled trade flows | 2.4 | L7 |
| 2.6 | Write the coverage report: top-10 traders reporting, share from mirror records, share excluded as non-maritime. | S02 | Script | P08 Trade coverage table | 2.2, 2.5 | L7 |
| 2.7 | Compile the 28 candidate chokepoints (cross-checked with UNCTAD, EIA, and Wang et al.) and assign permanent IDs. | — | Manual | P09 Chokepoint screening and justification table | — | L7 |
| 2.8 | Screen each candidate: both criteria, narrowest width measured in QGIS, territorial-sea overlap, decision, and reason. | — | Manual (QGIS) | P09 Chokepoint screening and justification table | 2.7 | L7 |
| 2.9 | Build chokepoint points and true-distance zones (50 km; Luzon 150 km; Torres 100 km). | S03 | Script | P10 Chokepoint points and zones | 2.8, 0.5 | L7 |
| 2.10 | Build the port list from PortWatch, using the World Port Index only where PortWatch lacks a country. | S04 | Script | P11 Ports and country port shares | 1.2, 0.5 | L7 |
| 2.11 | Compute port trade shares; keep each country’s ports that together carry at least 80% of its trade. | S04 | Script | P11 Ports and country port shares | 2.10 | L7 |
| 2.12 | Assign a foreign gateway port to each landlocked trading country. | S04 | Manual + script | P12 Landlocked gateway lookup | 2.10 |  |
| 2.13 | Link each port to the nearest point on the sea-lane network; flag ports far from it. | S04 | Script | P11 Ports and country port shares | 2.11 | L7 |
| 2.14 | ACLED: keep 2025 battles, explosions/remote violence, and violence against civilians; remove region-level locations. | S05 | Script | P13 Unified event layer | 1.6, 0.5 | L7 |
| 2.15 | ASAM: keep 2025 incidents; merge with ACLED events within 50 km and one day. | S05 | Script | P13 Unified event layer | 2.14 | L7 |
| 2.16 | GDELT: keep 2025 CAMEO 10–17 events located to a city or town; merge duplicates; keep the source count as a field. | S05 | Script | P13 Unified event layer | 1.5, 0.5 | L7 |
| 2.17 | Compute country-level GDELT tension totals for context, before the distance filter. | S05 | Script | P14 GDELT country context table | 2.16 |  |
| 2.18 | Keep events within 500 km of any candidate chokepoint; write the unified event layer. | S05 | Script | P13 Unified event layer | 2.9, 2.15, 2.16 | L7 |

## Phase 3: Consolidate

| ID | Task | Script | Kind | Product | Depends on | L7 |
|---|---|---|---|---|---|---|
| 3.1 | Write every prepared layer to the project database with the names in proposal Table 3. | S01–S05 | Script | P15 Verified project database | 2.6, 2.9, 2.13, 2.18 | L7 |
| 3.2 | Run the integrity check: unique IDs, ports for every trading country, zones for every chokepoint, WGS 84, required fields. | check_db.py | Script | P15 Verified project database | 3.1 | L7 |
| 3.3 | Open the database in QGIS and inspect each layer. | — | Manual (QGIS) | P15 Verified project database | 3.2 |  |

## Phase 4: Analyze

| ID | Task | Script | Kind | Product | Depends on | L7 |
|---|---|---|---|---|---|---|
| 4.1 | Split each trade flow into port-to-port sub-flows using the port shares. | S06 | Script | P16 Port-to-port sub-flows | 3.2 | L7 |
| 4.2 | Route every unique port pair under the baseline and Red Sea diversion; save each route once. | S06 | Script | P17 Routes and route crossings | 4.1 | L7 |
| 4.3 | Record which chokepoint zones each route crosses. | S06 | Script | P17 Routes and route crossings | 4.2 | L7 |
| 4.4 | Sum trade value and weight for each chokepoint and scenario (Equation 1). | S06 | Script | P18 Trade exposure values | 4.3 | L7 |
| 4.5 | Validate: compare diversion-route shares with PortWatch container and general-cargo transits (Spearman ρ, pass mark 0.5). | S06 | Script | P19 Routing validation table | 4.4, 1.3 | L7 |
| 4.6 | Review mismatched chokepoints; fix genuine errors only and log each fix. | — | Manual | P19 Routing validation table | 4.5 |  |
| 4.7 | Calculate the distance from every event to every chokepoint. | S07 | Script | P20 Conflict and tension exposure values | 3.2 | L7 |
| 4.8 | Count violence and tension events within 100, 250, and 500 km, with the fatality-weighted and GDELT-threshold variants. | S07 | Script | P20 Conflict and tension exposure values | 4.7 | L7 |
| 4.9 | Calculate density exposure at bandwidths of 100, 250, and 500 km (Equation 3). | S07 | Script | P20 Conflict and tension exposure values | 4.7 |  |
| 4.10 | Compare the ring and density rankings (Spearman). | S07 | Script | P21 Conflict method comparison | 4.8, 4.9 | L7 |
| 4.11 | List chokepoint pairs whose rings overlap at each distance and the share of events they share. | S07 | Script | P22 Ring overlap table | 4.7 |  |
| 4.12 | Build an editable copy of the sea-lane network so any chokepoint can be closed. | S08 | Script | P23 Editable sea-lane network | 3.2 |  |
| 4.13 | For each chokepoint, remove the network inside its zone, reroute affected baseline flows, and measure new lengths. | S08 | Script | P24 Route redundancy values | 4.12, 4.3 |  |
| 4.14 | Calculate the detour penalty (Equation 4) and the share of trade with no sea alternative. | S08 | Script | P24 Route redundancy values | 4.13 |  |

## Phase 5: Integrate

| ID | Task | Script | Kind | Product | Depends on | L7 |
|---|---|---|---|---|---|---|
| 5.1 | Calculate the trade–conflict correlation (Spearman, with p-value). | S09 | Script | P25 Trade–conflict correlation | 4.4, 4.8 | L7 |
| 5.2 | Transform trade and conflict with log(1 + x), rescale components to 0–1, and average violence and tension. | S09 | Script | P26 Composite scores and ranks | 4.4, 4.8, 4.14 |  |
| 5.3 | Calculate the composite score (Equation 6), ranks, and the multiplicative score (Equation 7). | S09 | Script | P26 Composite scores and ranks | 5.2 |  |
| 5.4 | Compare ranks under the baseline and Red Sea diversion. | S09 | Script | P27 Scenario comparison | 5.3 |  |
| 5.5 | Rerun the scoring with each Table 4 alternative, one change at a time. | S10 | Script | P28 Sensitivity results | 5.3 |  |
| 5.6 | Run the full weight grid (36 combinations). | S10 | Script | P28 Sensitivity results | 5.3 |  |
| 5.7 | Summarize best, worst, and median rank per chokepoint and each run’s agreement with the main run. | S10 | Script | P28 Sensitivity results | 5.5, 5.6 |  |

## Phase 6: Visualize

| ID | Task | Script | Kind | Product | Depends on | L7 |
|---|---|---|---|---|---|---|
| 6.1 | Draft the ranked vulnerability map (Equal Earth). | S11 | Script | P29 Ranked vulnerability map | 5.3 |  |
| 6.2 | Draft the component maps; for the first pass, the trade and conflict panels. | S11 | Script | P30 Component maps | 4.4, 4.8 | L7 |
| 6.3 | Draft the scenario route maps (baseline and Red Sea diversion side by side). | S11 | Script | P31 Scenario route maps | 4.2 | L7 |
| 6.4 | Draft chokepoint close-ups: events, distance rings, and density surface for the top chokepoints. | S11 | Script | P32 Chokepoint close-ups | 4.9, 5.3 |  |
| 6.5 | Draft the trade vs. conflict chart (four quadrants). | S11 | Script | P33 Trade vs. conflict chart | 5.1 | L7 |
| 6.6 | Draft the score breakdown chart. | S11 | Script | P34 Score breakdown chart | 5.3 |  |
| 6.7 | Draft the sensitivity rank-range chart. | S11 | Script | P35 Sensitivity chart | 5.7 |  |
| 6.8 | Write the results table (CSV and formatted). | S11 | Script | P36 Results table | 5.3, 5.7 |  |
| 6.9 | Write web layers: web_chokepoints, web_routes (merged segments, by scenario), and the density raster. | S11 | Script | P37 Web layers | 5.3, 4.2 |  |
| 6.10 | Build and save the QGIS project; style layers with fixed class breaks. | — | Manual (QGIS) | P38 QGIS project and final report maps | 6.9 |  |
| 6.11 | Finish the final report maps as QGIS print layouts. | — | Manual (QGIS) | P38 QGIS project and final report maps | 6.10 |  |
| 6.12 | Export the web map with qgis2web, push to GitHub Pages, check it online, and add the “as of” date. | — | Manual (QGIS) | P39 Interactive web map | 6.10 |  |
| 6.13 | Write the web-map update checklist in the README. | — | Manual | P39 Interactive web map | 6.12 |  |

## Phase 7: Report

| ID | Task | Script | Kind | Product | Depends on | L7 |
|---|---|---|---|---|---|---|
| 7.1 | Write the Lab 7 Results and Discussion section from the first-pass products. | — | Manual | P40 Lab 7 Results and Discussion | 2.6, 4.5, 4.10, 5.1, 6.2, 6.3, 6.5 | L7 |
| 7.2 | Update the Methodology for any adjustments the first pass reveals. | — | Manual | P40 Lab 7 Results and Discussion | 7.1 | L7 |
| 7.3 | Write the final report: full results, sensitivity, comparison with the 2026 Hormuz closure, limitations. | — | Manual | P41 Final report | All |  |

## Products

| ID | Product | Type | File or layer | Proposal reference | Objective | L7 |
|---|---|---|---|---|---|---|
| P01 | Project repository and Python environment | Code | `GitHub repository; environment.yml` | Methodology › Automation and reproducibility | All | L7 |
| P02 | Settings file | Code | `config/settings.yaml` | Automation and reproducibility (one settings file); Table 4 | All | L7 |
| P03 | Shared tools and runner | Code | `scripts/common.py; scripts/run_all.py` | Automation and reproducibility | All | L7 |
| P04 | Data manifest and field checks | Document | `data/raw/MANIFEST.csv` | Data › Data status; Data quality (download dates) | All | L7 |
| P05 | PortWatch port-level trade data | Dataset | `data/raw/portwatch_ports` | Step 1, item 4; Data processing (Ports); Table 1 | 2 | L7 |
| P06 | Raw Comtrade downloads (2025; 2024 totals) | Dataset | `data/raw/comtrade/` | Step 1, item 1; Table 1 | 2 | L7 |
| P07 | Reconciled trade flows | Layer / table | `trade_flows` | Step 1, item 1; Table 3; Data processing (Trade) | 2 | L7 |
| P08 | Trade coverage table | Table | `outputs/tables/trade_coverage.csv` | Data quality (UN Comtrade); Step 1, item 1 | 2 | L7 |
| P09 | Chokepoint screening and justification table | Table | `config/chokepoint_screening.csv` | Step 1, item 2 | 1 | L7 |
| P10 | Chokepoint points and zones | Layer | `chokepoints; zones` | Step 1, item 3; Table 3 | 1 | L7 |
| P11 | Ports and country port shares | Layer / table | `ports; country_port_shares` | Step 1, item 4; Table 3 | 2 | L7 |
| P12 | Landlocked gateway lookup | Table | `config/landlocked_gateways.csv` | Step 1, item 4 | 2 |  |
| P13 | Unified event layer | Layer | `events` | Step 1, item 5; Step 2, rule 3; Table 3 | 3 | L7 |
| P14 | GDELT country context table | Table | `gdelt_country_context` | Step 4 (context paragraph) | 3 |  |
| P15 | Verified project database | Database | `data/project.gpkg; logs/check_db.log` | Step 2; Table 3 | All | L7 |
| P16 | Port-to-port sub-flows | Table | `subflows` | Step 1, item 4 (sub-flows) | 2 | L7 |
| P17 | Routes and route crossings | Layer / table | `routes; route_crossings` | Step 3, items 1–3; Table 3 | 2 | L7 |
| P18 | Trade exposure values | Values | `chokepoint_scores (trade fields)` | Step 3, item 4; Equation 1 | 2 | L7 |
| P19 | Routing validation table | Table | `outputs/tables/validation.csv` | Step 3, item 5 | 2 | L7 |
| P20 | Conflict and tension exposure values | Values | `chokepoint_scores (conflict fields)` | Step 4, items 1–2; Equations 2–3 | 3 | L7 |
| P21 | Conflict method comparison | Table | `outputs/tables/conflict_methods.csv` | Step 4, item 3 | 3 | L7 |
| P22 | Ring overlap table | Table | `outputs/tables/ring_overlap.csv` | Step 4 (overlap paragraph) | 3 |  |
| P23 | Editable sea-lane network | Code / data | `Network graph (S08)` | Step 5, item 1 | 4 |  |
| P24 | Route redundancy values | Values | `chokepoint_scores (redundancy fields)` | Step 5; Equation 4 | 4 |  |
| P25 | Trade–conflict correlation | Table | `outputs/tables/correlation.csv` | Step 6, item 1 | Research question | L7 |
| P26 | Composite scores and ranks | Values | `chokepoint_scores (composite fields)` | Step 6, items 2–4; Equations 5–7 | 5 |  |
| P27 | Scenario comparison | Table | `outputs/tables/scenarios.csv` | Step 6 (scenario comparison paragraph) | 5 |  |
| P28 | Sensitivity results | Table | `outputs/tables/sensitivity.csv` | Step 6, item 5; Table 4 | 5 |  |
| P29 | Ranked vulnerability map | Map | `outputs/figures/` | Step 7; Table 5 | 5 |  |
| P30 | Component maps | Map | `outputs/figures/` | Step 7; Table 5 | 2–4 | L7 |
| P31 | Scenario route maps | Map | `outputs/figures/` | Step 7; Table 5 | 2 | L7 |
| P32 | Chokepoint close-ups | Map | `outputs/figures/` | Step 7; Table 5 | 3 |  |
| P33 | Trade vs. conflict chart | Chart | `outputs/figures/` | Step 7; Table 5 | Research question | L7 |
| P34 | Score breakdown chart | Chart | `outputs/figures/` | Step 7; Table 5 | 5 |  |
| P35 | Sensitivity chart | Chart | `outputs/figures/` | Step 7; Table 5 | 5 |  |
| P36 | Results table | Table | `outputs/tables/results.csv` | Step 7; Table 5 | 5 |  |
| P37 | Web layers | Layer | `web_chokepoints; web_routes; density raster` | Step 7; Table 3 | 5 |  |
| P38 | QGIS project and final report maps | Map | `qgis/project.qgz; map exports` | Step 7 | 5 |  |
| P39 | Interactive web map | Web map | `docs/ (GitHub Pages)` | Step 7; Table 5 | 5 |  |
| P40 | Lab 7 Results and Discussion | Report | `Lab 7 report` | New section after Methodology (Lab 7) | All | L7 |
| P41 | Final report | Report | `Final report` | Whole proposal | All |  |

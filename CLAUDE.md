# REE Chokepoint Vulnerability Analysis

GEOG797 capstone (University of Maryland, MS GEOINT), Garrett Mahoney.

The project scores each maritime chokepoint (about 27 straits and canals) on how exposed
2025 rare earth element (REE) trade is to conflict and tension there. The score combines
three components: REE trade crossing the chokepoint, violence and tension nearby, and how
far trade would detour if the chokepoint closed. Everything is scripted in Python so the
analysis can be rerun from the raw downloads.

## Key documents (read before writing code)

| File | What it is |
|---|---|
| `docs/methodology.md` | The proposal's Methodology section: the method, equations 1–7, Tables 2–5 |
| `docs/tasks.md` | Numbered tasks per script; the specification for each script |
| `config/settings.yaml` | Every parameter; scripts read values from here and nowhere else |

**Order of precedence when they disagree:** this file's "Decisions that override the
proposal" section, then `config/settings.yaml`, then `docs/tasks.md`, then
`docs/methodology.md`. If something is still unclear, ask rather than guess.

## Repository layout

```
CLAUDE.md
README.md               download steps, run steps, web-map update checklist
environment.yml
config/                 settings.yaml, chokepoint_screening.csv, landlocked_gateways.csv
data/                   NOT committed (see .gitignore)
  raw/                  original downloads + MANIFEST.csv; read-only
  interim/              working files a script may write
  project.gpkg          the project database
scripts/                common.py, run_all.py, check_db.py, s01_get_trade.py ... s11_figures.py
logs/                   one log per script run
outputs/tables/         derived tables (safe to publish)
outputs/figures/        draft figures
qgis/                   project.qgz
docs/                   methodology.md, tasks.md, and the qgis2web export for GitHub Pages
```

## Scripts

| Script | Purpose | Tasks |
|---|---|---|
| `common.py` | Load settings, logging, great-circle distance, ISO-3 codes, GeoPackage read/write | 0.4 |
| `run_all.py` | Run S01–S11 in order, or one script by name | 0.5 |
| `s01_get_trade.py` | Download Comtrade 2025 records (and 2024 totals) | 2.1–2.2 |
| `s02_clean_trade.py` | ISO-3 codes, reconcile flows, classify transport mode, coverage report | 2.3–2.6 |
| `s03_chokepoints.py` | Points and true-distance zones from the screening file | 2.9 |
| `s04_ports.py` | Principal ports, port trade shares, landlocked gateways, snapping | 2.10–2.13 |
| `s05_events.py` | ACLED, IMO GISIS, GDELT filtered and merged into one event layer | 2.14–2.18 |
| `check_db.py` | Integrity check of the project database | 3.2 |
| `s06_trade_exposure.py` | Sub-flows, routing (two scenarios), crossings, Equation 1, validation | 4.1–4.5 |
| `s07_conflict_exposure.py` | Distances, rings, density, method comparison, ring overlap | 4.7–4.11 |
| `s08_redundancy.py` | Editable network, close each chokepoint, Equation 4 | 4.12–4.14 |
| `s08b_reroute_cost.py` | Extra days at sea and extra cost of each chokepoint's detour (impact layer) | 4.15–4.17 |
| `s09_composite.py` | Trade–conflict correlation, Equations 5–7, ranks, scenario comparison | 5.1–5.4 |
| `s10_sensitivity.py` | One-at-a-time alternatives and the weight grid | 5.5–5.7 |
| `s11_figures.py` | Draft maps, charts, results table, web layers | 6.1–6.9 |

Run with `python scripts/run_all.py` (everything) or `python scripts/run_all.py s02` (one script).

## Conventions every script follows

- **Settings, not constants.** Read every parameter through `common.load_settings()`. No
  distances, years, codes, thresholds, or weights written into a script.
- **Raw data are read-only.** Never modify anything in `data/raw/`. Write new outputs only.
- **Log what happens.** Each run writes `logs/<script>.log` with what it read, how many
  records it kept and removed at each filter and why, and what it wrote. These counts are
  reported in Lab 7, so log them clearly.
- **Start the log the same way.** Every script starts its log with
  `common.start_log(Path(__file__).stem, settings)`, so the log is named after the script.
- **Fixed names.** Use the GeoPackage layer names under `layers:` in the settings file and
  keep field names stable between runs (QGIS styling and the web map depend on them).
  Each script's columns in chokepoint_scores start with a prefix no other script uses
  (trade_ S06, viol_/tens_ S07, redund_ S08, impact_ S08b, quad_/score_ S09, sens_ S10);
  a script clears only its own prefix. The list is `check_db.score_prefixes` in the settings
  file, and check_db.py fails on any other column.
- **Shared keys.** Countries are ISO-3 codes; chokepoints keep their permanent IDs from the
  screening file; flows have a flow ID; routes carry a scenario label.
- **Coordinates and distances.** Store layers in EPSG:4326. Measure distances along the
  Earth's surface (great-circle), never in degrees or a flat projection. Build zones as
  true-distance circles. Equal Earth (EPSG:8857) is for static maps only.
- **Rerunnable.** A script overwrites its own outputs, so rerunning it gives the same result.
- **Readable code.** Small functions with docstrings, plain-language comments that link
  back to task IDs (for example `# Task 2.4: importer record first`), and a `main()` guarded
  by `if __name__ == "__main__":`.
- **Libraries.** pandas, NumPy, GeoPandas, Shapely, pyogrio, searoute, NetworkX, SciPy,
  scikit-learn, Matplotlib, PyYAML, requests, comtradeapicall, pyarrow (reads the GDELT
  .parquet files). Ask before adding others.

## Data rules

- `data/` is never committed, except the manifest. ACLED's terms bar republishing its raw data.
- Only derived results (scores, routes, density surfaces, tables, figures) go in `outputs/`
  or the web map. Never put raw ACLED event points in anything published.
- `logs/` is committed and published. Logs contain counts and summaries only, never raw
  ACLED event rows or other individual records from licensed data.
- Record download dates in `data/raw/MANIFEST.csv`.
- `data/raw/MANIFEST.csv` is the only file under `data/` that is committed (see the
  `.gitignore` exception).

## Decisions that override the proposal

These were decided after the proposal text was written. Follow them where they differ.

1. **Permanent magnets (HS 8505.11) are a sensitivity test only.** Step 1, item 1 of the
   proposal lists it beside the main codes; it is never part of the main run.
2. **Ports come from PortWatch.** Use PortWatch port locations and trade shares as principal
   ports; use the World Port Index only for countries PortWatch lacks.
3. **Principal ports = those that together carry 80% of a country's trade**
   (`ports.principal_port_share`). Not yet in the proposal.
4. **Closing chokepoints uses an editable network.** searoute can close only some named
   passages. For S08 (and if needed S06), load the sea-lane network into NetworkX, remove
   segments inside a chokepoint's zone, and route on the edited network. Prototype early.
5. **GDELT source count is stored, not filtered away.** S05 keeps the source count as a field;
   S07 applies the 1/2/3-source thresholds.
6. **Sensitivity tests change one choice at a time** from the main run; only the weight grid
   is tested in full.
7. **Event window is 1 Oct 2024 to 30 Sep 2025** for ACLED, ASAM and GDELT
   (`events.window_start` / `window_end`), because the ACLED account only provides data older
   than 12 months. Trade stays calendar 2025. Scripts filter events by these settings, never
   by `study_year`.
8. **Reroute time and cost are an impact layer, not a score component.** For each chokepoint,
   S08b converts S08's extra route distance into extra days at sea
   (extra nautical miles / (speed x 24)) and extra cost: per typical container-ship voyage
   (extra days x (daily charter cost + daily fuel use x fuel price) +/- change in canal tolls)
   and REE cargo holding cost (trade value x annual interest rate x extra days / 365). These
   are reported beside the score (results table, figures, web map). The composite score keeps
   its three components; a cost-based redundancy measure is tested as one sensitivity
   alternative (Table 4). Parameters are in `reroute_impact` in the settings file.
9. **IMO GISIS replaces ASAM as the at-sea incident source.** The ASAM download has no
   records after 25 June 2024, so none fall in the event window. Piracy and armed-robbery
   incidents at sea come instead from the IMO GISIS Piracy and Armed Robbery module
   (`data/raw/GISIS/ListOfIncidents.xls`: all areas, 2 Oct 2024 to 30 Sep 2025, 200 incidents,
   193 with both coordinates in degrees and decimal minutes). Wherever the proposal, `docs/tasks.md`
   or the settings say ASAM, read IMO GISIS. The ASAM rules still apply: filter to the event
   window, treat an incident within 50 km of a kept (cleaned, violent) ACLED event, with dates at
   most one day apart, as the same event, and publish only derived results, never individual
   incident rows. On a match the ACLED record is kept and flagged `imo_match`; the incident is
   dropped. Counting rule: the main run counts only incidents where crew were threatened,
   injured or held (`crew_harm`), set from keywords and negation words in
   `events.imo_gisis` in the settings, corrected by a hand-checked overrides list there
   (event ID -> crew_harm and a short reason category); misses are fixed by overrides, not more
   keywords. S05 stores `crew_harm` on every incident and S07 applies `count_rule` (as with the
   GDELT source count, Decision 5); counting all incidents is an S10 alternative. S05 reads only
   the date, area, coordinates and the two crew texts; ship names, IMO numbers and narrative
   text are never stored or logged.
10. **How S08 closes chokepoints and where it writes.** (a) Closing a chokepoint removes every
   network edge that intersects its zone (the same test S06 uses for crossings), not only
   edges with an end inside; this replaces the wording of Decision 4. S08 routes on a copy of
   searoute's own network passed with `M=`, and skips the zero-length edges joining 180° and
   −180° when testing against zones. (b) One chokepoint is closed at a time; all other
   passages stay open, including Panama when Suez is closed. For the Suez check (task 4.17),
   S08b also reports a Cape-only figure from S06's Red Sea diversion routes. (c) Flows from or
   to a port inside the closed zone are not rerouted; they are left out of Equation 4 and
   reported as a separate count. (d) Flows with no remaining sea route get the largest penalty
   observed across all chokepoints and count toward the share with no sea alternative; a lower
   cap is tested in S10. (e) A chokepoint with no affected flows gets an empty (null) penalty;
   S09 decides how to treat it. (f) Outputs: redundancy columns in `chokepoint_scores` start
   with `redund_`; per-flow detours go in a new table `detours` (flow_id, chokepoint_id, old
   and new length in nm, no_route flag, port_in_zone flag); rerouted lines go in a new layer
   `detour_routes`. S08 never writes to `routes`. Zone sizes resolved by Decision 11.
11. **A chokepoint's zone spans the whole passage.** Otherwise ships can cross the strait
   outside the zone; the S08 groundwork found this at Taiwan, Korea, Luzon and Yucatán. Rule:
   radius = half the narrowest width + 10 km, rounded up to the next 10 km, never below
   `zone_radius_km` and never below an existing override (Torres, 100 km). Widths and sources
   are in `narrowest_width_km` and `width_source` in the screening file. Exception: the Korea
   Strait is 90 km, not 110, because at 100 km Busan falls inside the zone; the leak test
   showed 90 km closes both channels. Every radius is checked with the leak test (no rerouted
   path passes through the strait outside the zone). The Kanmon Strait and Seto Inland Sea
   route is accepted as a real alternative to the Korea Strait, for smaller ships only, and
   stated as a limitation. The same radii apply to S06 crossings and S08 closures.
   Amended by Decision 12 for the Taiwan, Korea and Luzon straits and the Yucatán Channel.
12. **Gate lines for the four wide straits (amends Decision 11).** For the Taiwan Strait,
   Korea Strait, Luzon Strait and Yucatán Channel, a circle wide enough to span the passage
   also caught routes that only clipped its edge without passing through (at Korea,
   Shanghai–Nagasaki was 46% of the value). These four use a gate line drawn across the
   passage from shore to shore, using islands as stepping stones, stored as WKT in
   `config/chokepoint_gates.csv` with the reason for each line. A route crosses the
   chokepoint if it crosses the gate, with no exceptions (so Busan and Incheon flows through
   the Western Channel and Europe–Wakamatsu flows through the Iki channel count). S08 closes
   a gated chokepoint by removing every network edge that crosses the gate; for a gate, no
   port is "inside the zone", so Decision 10(c) does not apply. Gate ends lie about 5 km
   inland on Natural Earth 10m countries, and no network edge passes round an end within
   50 km. The other 23 chokepoints keep circles under Decision 11 (Torres 100, Bohai 70,
   Makassar 80, Mona 80 km). The four gated straits keep a 50 km circle for maps only; the
   Korea 90 km exception no longer applies. The leak test and a clipping check (crossings
   whose closest approach exceeds 60% of the radius) were run for all 27 chokepoints.
   Limitations: a gate counts only ships that cross it (Taichung–Europe is not counted at
   Taiwan); results depend on where each line is drawn; Dalian–Seattle is about 54% of
   Korea's value; small edge clips remain at Dover (London–Antwerp, $53K) and Gibraltar
   (Tanger Med flows), under 0.1% of each.
13. **How S08b summarises reroute time and cost.** Per chokepoint, the headline extra days and
   extra cost per voyage are value-weighted means over rerouted sub-flows (the same weighting as
   Equation 4), with the median beside them. Canal tolls are added or subtracted by testing whether
   the baseline and detour routes cross Suez or Panama. Routing picks the shortest path and ignores
   tolls, so with Panama open some Suez detours go via Panama (a limitation). Flows with no sea
   route, and port-in-zone flows, get no cost (null) and are reported as counts and value. S08b
   reports low/high ranges itself, changing one parameter at a time (speed and fuel always paired)
   and taking the smallest and largest result. REE holding cost is a total for 2025 trade, not a
   per-voyage figure. The Suez check (task 4.17) compares the Cape-only detour from S06's Red Sea
   diversion routes with ITF/OECD (Merk 2024), plus a variant using ITF's own charter rate and toll.
14. **PortWatch principal ports (details Decisions 2 and 3).** `ports.assignment: principal_ports`
   in the main run. (a) Principal ports are chosen separately for exports and imports, from
   PortWatch's `share_country_maritime_export` and `_import` (percent of the country's AIS-derived
   maritime trade, 2019-25, all goods). (b) Before the 80% rule, S04 removes offshore oil terminals
   (portid `fso...`) and mostly-tanker ports: tanker calls at least half of all calls and
   container plus general-cargo calls under 10% (`ports.max_tanker_share`, `min_cargo_share`),
   because REE oxides and metals ship in containers or as general cargo, never by tanker. A
   plain tanker-majority test was rejected: it removed Singapore, Rotterdam, Antwerp, Ulsan and
   Houston. Ports with no vessel calls are kept. The filter uses vessel-call counts, not cargo
   volumes, so it is approximate; PortWatch shares cover all goods over 2019–25, not REE in 2025,
   and are used only to split each country's REE trade across its ports (a stated limitation). (c) The remaining shares are rescaled to 100% per
   country and side; ports are taken largest first until the running total reaches 80%, the
   crossing port included. (d) S06 splits each flow into sub-flows, one per pair of an origin
   export port and a destination import port; a sub-flow's share is the origin port's rescaled
   export share times the destination port's rescaled import share, so a flow's shares sum to 1.
   S06 routes each pair of network points once. (e) Landlocked countries in
   `config/landlocked_gateways.csv` use the PortWatch port in `gateway_portid` with share 1;
   KAZ (Caspian ports only) and MDA (Giurgiulesti, a Danube river port) stay landlocked although
   PortWatch lists ports for them. Gateways are used even if the filter would remove them.
   (f) Principal ports farther than `ports.snap_warning_km` from the sea-lane network are kept
   and flagged (`snap_flag`), because the network is sparse near some coasts (all 16 checked by
   hand on 29 Oct 2026; 15 snap to a node in the same sea and on the same coast). Only ports in
   `ports.snap_exclude` (hand-checked: nearest node in another sea; for now Ereğli, whose node
   is in the Sea of Marmara) are dropped, and their share is moved to the country's other
   principal ports on that side; S04 logs the share moved.
   (g) The World Port Index is used only for trading countries PortWatch lacks (BMU in the 2025
   run: its largest WPI size class, equal shares). (h) S10 alternatives: `port_assignment:
   nearest_gateway` (the Lab 7/8 WPI first pass, kept as functions in S04 and S06; its database
   is saved as `data/interim/lab8_first_pass.gpkg`) and `port_filter: all_ports` (no filter).

## How to work in this repository

- Build **one script per session**, in the order in `docs/tasks.md` (Phase 0, S01–S02, S03–S04,
  S05 + check_db, S07, S06, S08, S09–S10, S11).
- **Before writing code that reads a raw file, look at the file** (or ask for its column
  headers and first rows). Never invent column names.
- Test on a small sample first, then the full data. Report the counts from the log.
- Do not edit `docs/methodology.md`. If the method needs to change, say so and explain why.
- When a task is finished, say which task IDs it completes so the tracker can be updated.

## Current status

Lab 7 submitted 20 Oct 2026 (first pass of S01-S07, S09 task 5.1, S11 draft figures); frozen as git tag lab7-gold. Lab 8 submitted 27 Oct 2026; code frozen as git tag lab8-gold. Decision 9 recorded: IMO GISIS replaces ASAM. S05 updated 27 Oct 2026 (task 2.15): 136 IMO GISIS incidents in the events layer (25 crew_harm), 9 merged into ACLED events. S07 applies count_rule through common.countable_events (27 Oct 2026; S07, S09 5.1, S11 rerun): 25 GISIS incidents counted, 111 dropped; viol_n_250 changes only at Malacca 7->30, Sunda 6->7, Mindoro 57->58; main rho 0.094 (p 0.64), quadrants unchanged; S10 alternative imo_count_rule: [all] recorded. Reroute cost parameters P1-P8 recorded in settings (task 4.15); Suez check source recorded (task 4.17; comparison runs in S08b). Not built: S09 composite (5.2-5.4), S10, web map. check_db.py built 27 Oct 2026 (task 3.2; checks task 3.1): run by run_all after S05 (--stage inputs, 70 checks) and at the end (--stage all, 144 checks); 0 FAIL, WARN only for the expected gaps in `check_db.expected_missing_layers` and the 5 Red Sea diversion port pairs with no sea route (6 sub-flows, $26K). Decision 10 recorded (S08 closure rule and outputs). Decision 11 recorded; task 2.8 zone radii set. Decision 12 recorded (gate lines for four wide straits); S03/S06 updated and rerun. S08 built 27 Oct 2026 (tasks 4.12-4.14): 22 chokepoints scored, 5 null; largest penalty 627.3% (Gibraltar, BEL_DZA_284610_1) applied to Hormuz, Bosporus, Bohai. Decision 13 recorded; S08b built 27 Oct 2026 (tasks 4.16-4.17): 19 chokepoints costed, 8 null (3 no sea route, 5 no affected flows); Suez check Cape-only 9.59 days and $900K one way vs ITF/OECD 10 days and $850K (ratios 0.96, 1.06). Decision 14 recorded (PortWatch principal ports); S04 rebuilt 29 Oct 2026 (tasks 2.10-2.13): port filter removed 353 PortWatch ports in trading countries (143 offshore, 210 mostly tanker); 244 principal export ports (53 countries) and 395 import ports (101 countries), plus 23 PortWatch gateways and 1 WPI port (BMU); 16 far ports kept and flagged, 1 dropped by ports.snap_exclude (Ereğli, 3.5% of Turkey's imports moved); ports layer 481 rows (480 in use), country_port_shares 671 rows; S06 preview: 52,292 sub-flows, 21,619 network-point pairs to route (S08 projected about 66 min); check_db --stage inputs passes (78 checks, 0 FAIL). The first pass is saved as data/interim/lab8_first_pass.gpkg. Not yet rerun: S06 task 4.1 (next session), then S08, S08b, S09 5.1, S11. Until then subflows and routes use the old WPI port IDs, so check_db --stage all is expected to fail.
<!-- Update this section as work progresses. -->

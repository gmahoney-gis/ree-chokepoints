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
| `s05_events.py` | ACLED, ASAM, GDELT filtered and merged into one event layer | 2.14–2.18 |
| `check_db.py` | Integrity check of the project database | 3.2 |
| `s06_trade_exposure.py` | Sub-flows, routing (two scenarios), crossings, Equation 1, validation | 4.1–4.5 |
| `s07_conflict_exposure.py` | Distances, rings, density, method comparison, ring overlap | 4.7–4.11 |
| `s08_redundancy.py` | Editable network, close each chokepoint, Equation 4 | 4.12–4.14 |
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

## How to work in this repository

- Build **one script per session**, in the order in `docs/tasks.md` (Phase 0, S01–S02, S03–S04,
  S05 + check_db, S07, S06, S08, S09–S10, S11).
- **Before writing code that reads a raw file, look at the file** (or ask for its column
  headers and first rows). Never invent column names.
- Test on a small sample first, then the full data. Report the counts from the log.
- Do not edit `docs/methodology.md`. If the method needs to change, say so and explain why.
- When a task is finished, say which task IDs it completes so the tracker can be updated.

## Current status

Proposal (Lab 6) complete. Repository is on GitHub (gmahoney-gis/ree-chokepoints).
Data collected except PortWatch port data (Comtrade HS 8505.11 is now downloaded).
Phase 0 complete: common.py, test_common.py and run_all.py written and tested.
Next: task 1.1 (data manifest) and S01–S02.
<!-- Update this section as work progresses. -->

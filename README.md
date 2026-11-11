# Rare Earth Trade Vulnerability at Maritime Chokepoints

GEOG797 capstone, University of Maryland (MS GEOINT), Garrett Mahoney.

This project scores 27 maritime chokepoints (straits and canals) on how exposed 2025 seaborne
trade in rare earth elements (REEs) is to conflict there. Each score combines three parts: how
much REE trade passes through, how much violence and political tension there is nearby, and how
far ships would have to detour if the chokepoint closed. Everything is scripted in Python, so the
whole analysis can be rerun from the raw downloads.

**Live map:** https://gmahoney-gis.github.io/ree-chokepoints/

The code as submitted with the final report is git tag `paper-gold` (commit c8a29dd); later commits add
the poster figures and the web map and do not change the paper's results.

The method is in `docs/methodology.md`, the task list in `docs/tasks.md`, and every parameter in
`config/settings.yaml`. `CLAUDE.md` records the decisions made after the proposal.

## Set up the environment

1. Install Miniconda or Anaconda.
2. From the project folder, create the environment once:
   `conda env create -f environment.yml`
3. Activate it each time you work on the project: `conda activate ree`
4. After `environment.yml` changes: `conda env update -f environment.yml --prune`

S01 (the Comtrade download) also needs a UN Comtrade subscription key in the environment
variable `COMTRADE_API_KEY`. The key is never stored in the project.

## Get the data

`data/` is not in the repository. ACLED's terms do not allow its raw data to be republished,
and the other downloads are large. `data/raw/MANIFEST.csv` (the one committed file under
`data/`) lists every raw file: where it came from, when it was downloaded, its size and MD5 checksum,
and which script uses it (or why none does: kept for provenance, or superseded). Put the downloads at the paths it gives.

## Run the analysis

From the project folder, with the environment active:

- Everything, in order: `python scripts/run_all.py`
- One step: `python scripts/run_all.py s02`
- Several steps (always run in the standard order): `python scripts/run_all.py s06 s07 s09`

Step names, in run order:

| Name | What it does |
|---|---|
| `s01` | Download Comtrade 2025 trade records |
| `s02` | Clean trade: country codes, flows, transport mode |
| `s03` | Chokepoint points, zones and gate lines |
| `s04` | Principal ports and port trade shares |
| `s05` | ACLED, IMO GISIS and GDELT events |
| `check_db` | Check the input layers (S01-S05) |
| `s06` | Route trade and count chokepoint crossings |
| `s07` | Violence and tension near each chokepoint |
| `s08` | Close each chokepoint and measure the detour |
| `s08b` | Extra days at sea and extra cost of each detour |
| `s09` | Composite scores and the Red Sea scenario |
| `s10` | Sensitivity tests |
| `s11` | Paper figures and tables |
| `s11b` | Poster figures |
| `s11c` | Web map layers and page |
| `check_db_all` | Check every layer in the database |

Each step writes a log to `logs/`. Results go to `data/project.gpkg` (not committed),
`outputs/` (tables and figures) and `docs/` (the web map).

## Update the web map (task 6.13)

The page `docs/index.html` is built by `s11c` from `scripts/web/index_template.html` and the
settings. Do not edit `docs/index.html` by hand; the next run overwrites it.

1. Make the change in `config/settings.yaml` (all page text and map settings are under
   `outputs.web`), or in the template.
2. Rebuild the web map: `python scripts/run_all.py s11c`
3. Check the database: `python scripts/run_all.py check_db_all`. It must end with 0 FAIL.
4. Test locally:
   1. `cd docs`
   2. `python -m http.server 8000`
   3. Open http://localhost:8000/ in Chrome.
   4. Press F12; in the Network tab tick **Disable cache**.
   5. Press Ctrl+Shift+M, choose **iPhone 12 Pro**, and press F5.
   6. Check the desktop view the same way (Ctrl+Shift+M again), then stop the server with Ctrl+C,
      then `cd ..` to go back to the project folder (steps 2-3 run from there).
5. Commit and push the changes.
6. Wait for the green tick at https://github.com/gmahoney-gis/ree-chokepoints/actions
   (the "pages build and deployment" run, 1-2 minutes).
7. Check the live map on a phone in a private tab, so you do not see an older cached copy.
8. Only if the results changed (S09/S10 rerun), not for wording or style changes: update
   `outputs.web.as_of` and `as_of_note`, and check the hand-copied figures in
   `outputs.web.red_sea.text` against the sources the settings comment above it lists.
   Then repeat steps 2-7.

<!--
Source: GEOG797 Lab 6 proposal, Methodology section (13 Oct 2026), converted for Claude Code.
This is the proposal text as written. Where it differs from CLAUDE.md ("Decisions that override
the proposal") or config/settings.yaml, follow CLAUDE.md and the settings file.
Figure 2 (workflow diagram) is not reproduced here; docs/tasks.md lists the same steps as tasks.
Reference numbers in square brackets refer to the proposal's reference list.
-->

# Methodology

This project is an analysis and modeling project. It produces one reproducible vulnerability score for each maritime chokepoint, measuring how exposed 2025 rare earth element (REE) trade is to conflict and tension at that location. The score combines three measured components: how much REE trade passes through the chokepoint, how much violence and tension surround it, and how far that trade would have to detour if the chokepoint closed.

The design is cross-sectional: every input is restricted to calendar year 2025, and the result is a snapshot rather than a forecast. The method synthesizes two traditions identified in the literature review. From the GIS and vessel-tracking tradition [1, 8, 9], it borrows the conversion of country-level trade into routes across the ocean. From the conflict-mapping tradition [4, 15], it borrows the use of mapped conflict events, but measures them around a fixed point rather than across whole countries.

## Workflow overview

Table 2 summarizes the workflow as seven steps, listing the Python scripts that carry out each step, the objective it serves, and what it produces. Figure 2 shows the same workflow as a diagram, from the source datasets at the top to the final maps and charts at the bottom. The rest of this section describes the steps in order.

**Table 2.** Workflow summary.

| Step                                  | Scripts | Objective    | Main output                                                                            |
|----------------------|---------|------------|-----------------------------|
| 1\. Prepare the data                      | S01–S05     | 1 (supports 2–3) | Clean trade table, screened chokepoint layer, country–port lookup, and unified event layer |
| 2\. Consolidate in a project database     | —           | All              | One GeoPackage file with shared identifiers                                                |
| 3\. Measure trade exposure                | S06         | 2                | REE trade crossing each chokepoint, per routing scenario                                   |
| 4\. Measure conflict and tension exposure | S07         | 3                | Separate violence and tension values for each chokepoint                                   |
| 5\. Measure route redundancy              | S08         | 4                | Detour penalty for each chokepoint                                                         |
| 6\. Combine and test the score            | S09–S10     | 5                | Ranked composite score, sensitivity results, and scenario comparison                       |
| 7\. Visualize the results                 | S11         | 5                | Maps, charts, tables, and an interactive web map                                           |

## Automation and reproducibility

All steps are carried out by eleven short Python scripts (S01–S11) that run in a fixed order, each picking up where the previous one left off. Splitting the work this way means any single step can be checked, corrected, and rerun without repeating everything before it. The scripts rely on widely used open-source libraries: pandas and NumPy for tables, GeoPandas and Shapely for spatial data, the searoute package for sea routing, SciPy and scikit-learn for statistics and density estimation, and Matplotlib for draft graphics. The interactive web map is built in QGIS with the qgis2web plugin. Three conventions apply to every script:

-   **One settings file.** Every choice that could change a result (the study year, HS codes, zone sizes, buffer distances, density settings, and score weights) is stored in a single plain-text settings file rather than written into the scripts. Every assumption can be seen in one place, and the sensitivity tests in Step 6 simply change these settings and rerun.

-   **Raw data are never edited.** Each script reads from the original downloads or the project database and writes new outputs, so the raw files always remain available for checking.

-   **Every run is logged.** Each script records what it read, what it removed or filled in and why, and how many records it produced. These logs form the data-quality log cited throughout this proposal.

A single command runs S01 through S11 in order. The scripts, settings file, software versions, logs, and derived results such as the chokepoint scores will be kept together in a public, version-controlled repository on GitHub, which also hosts the web map (Step 7). Raw data are not included; instead, the repository documents how to download each source, because some providers, including ACLED, require registration and do not permit raw data to be republished. The repository also includes a short checklist for updating the web map after any rerun.

Visual workflow builders, such as ArcGIS ModelBuilder and the QGIS Graphical Modeler, were considered for automating the analysis. They were not used because the two most demanding steps, routing every trade flow across a global sea-lane network and rerouting all affected flows each time a chokepoint is closed, are far simpler to write, test, and repeat as scripts. Figure 2 serves the same purpose a model diagram would, showing every step and how data move between them.

**Figure 2.** Methodology workflow. Data move from top to bottom, from the source datasets to the final outputs. Solid arrows show data passing from one step to the next; dashed boxes and arrows show information used for validation or context that does not enter the score. Labels S01–S11 identify the Python script that performs each step (Table 2).

## Step 1: Prepare the data (S01–S05)

Each source dataset is first prepared on its own. The routine cleaning of trade, port, and event records is described under Data processing; this step adds the decisions that shape the analysis, including the chokepoint screening required by Objective 1.

1.  **Trade (S01–S02).** The 2025 Comtrade records for HS 2805.30, 2846.10, 2846.90, and 8505.11 are downloaded and reconciled into one origin–destination table. Following Peng and Guo [13], the importing country’s record is used where available, and the exporter’s mirror record fills any gap; the share of trade value filled this way is reported with the results. Both trade value (US dollars) and weight (kilograms) are kept. Only seaborne trade is routed. Where the reporting country records the mode of transport, as Comtrade does for many major traders, flows sent by road, rail, or air are excluded; this matters because some high-value rare earth metals travel by air. Where the mode is not reported, pairs that share a land border, such as Myanmar–China, are treated as overland and all other pairs as seaborne. The share of trade value excluded under each rule, and the largest excluded pairs, are reported with the results. Permanent magnets (HS 8505.11) are prepared the same way for the sensitivity test.

2.  **Chokepoints (S03, Objective 1).** The chokepoints are fixed before any trade or conflict data are examined, so the set cannot be shaped to fit the results [6]. The 28 PortWatch locations are cross-checked against the chokepoints named by UNCTAD [14], the U.S. Energy Information Administration [20], and Wang, Du, and Peng [1]. A candidate is kept only if it meets two criteria: a physical constraint (land on both sides narrows the passage, so ships cannot avoid it without a substantial detour) and navigational control (one or more coastal states can monitor, regulate, or obstruct passage). Width alone is not used as a cutoff, because some wide passages, such as the Taiwan Strait, are among the most geopolitically contested; how easily each chokepoint can be avoided is instead measured directly in Step 5. Each decision and its reason are recorded in a justification table, together with the narrowest width of each passage and whether the coastal states’ 12-nautical-mile territorial seas overlap across it. Candidates that fail, such as the Cape of Good Hope, stay in the sea-lane network as detour routes but receive no score. Only open-water passages such as the Cape of Good Hope are expected to fail, leaving roughly 27 chokepoints to score; S03 confirms the final count.

3.  **Chokepoint zones (S03).** Each retained chokepoint is represented by a point and a 50 km zone, widened to 150 km for the Luzon Strait and 100 km for the Torres Strait so the zone spans the full passage. The zone is used only to decide which routes pass through the chokepoint; the larger distances used to measure conflict are described in Step 4.

4.  **Ports (S04).** Each country is assigned its principal seaports from the World Port Index (large and medium harbors), with a foreign gateway for landlocked countries. Where a country has more than one principal port, its trade is split among them in proportion to each port’s 2025 share of the country’s seaborne trade, taken from IMF PortWatch port-level estimates [9]. Each flow is therefore divided into port-to-port sub-flows that carry their share of the flow’s value and weight. Because these shares cover all cargo rather than REE alone, they approximate where REE shipments leave and arrive. Each port is then linked to the nearest point on the sea-lane network.

5.  **Events (S05).** Violent events combine ACLED battles, explosions/remote violence, and violence against civilians with ASAM attacks on vessels; an ASAM record within 50 km and one day of an ACLED record is treated as the same event and counted once. Tension events are GDELT events coded as conflict short of physical violence (CAMEO root codes 10–17). Because GDELT is coded automatically from news reports, three further filters apply: an event must be reported by at least two separate news sources, it must be located to a city or town rather than only to a province or country, and repeated reports of the same event (same day, event type, actors, and location) are merged into one. The location filter matters most near small coastal states, because GDELT tends to place events at national centers or capitals when no exact location is reported [18]. Events located only to a country or province centroid are removed, because a centroid can fall near a chokepoint by accident; the number removed is logged.

**Inputs:** Comtrade, PortWatch (chokepoints and ports), UNCTAD, EIA, World Port Index, searoute network, ACLED, ASAM, and GDELT. **Outputs:** reconciled trade table, chokepoint and zone layers with the justification table, country–port lookup, and unified event layer.

## Step 2: Consolidate the data in a project database

The prepared datasets arrive in different formats (tables, point files, and line networks), at different levels of detail (countries, ports, and individual events), and with different naming conventions. They are combined in a single project database: one GeoPackage file, an open format that QGIS, ArcGIS Pro, and Python all read directly. Every analysis script reads from this file, so all steps work from the same version of the data. Three rules make the datasets fit together:

1.  **Shared identifiers.** Countries are always identified by their three-letter ISO code, which resolves naming differences between Comtrade, the World Port Index, and ACLED (for example, “Korea, Rep.” versus “South Korea”). Each chokepoint receives a permanent ID, each origin–destination trade flow a flow ID, and each modeled route a scenario label.

2.  **One location standard.** All layers are stored in latitude and longitude (WGS 84). Distances are measured along the curve of the Earth rather than on a flat map, because flat projections distort distance at a global scale.

3.  **One event format.** ACLED, ASAM, and GDELT records share a common set of fields: date, location, source, category (violence or tension), event type, fatalities where reported, and a location-precision flag.

The key link is between trade and geography. Trade flows connect to ports through country codes, ports connect to routes through the routing in Step 3, and routes connect to chokepoints wherever a route passes through a chokepoint zone. The result is a crossing table that lists, for every flow and scenario, which chokepoints it passes. This one table drives both the trade exposure (Step 3) and route redundancy (Step 5) calculations. Table 3 lists the layers the database will hold.

**Table 3.** Project database layers.

| Layer                   | Type      | Script | Key fields                                                                                                                                               |
|----------------|-----------|----------|------------------------------------|
| trade_flows                 | Table         | S02        | flow ID, origin ISO, destination ISO, HS code, value (US$), weight (kg), transport mode, maritime flag                                                      |
| chokepoints, zones          | Points, areas | S03        | chokepoint ID, name, zone radius, included (yes/no), reason                                                                                                  |
| ports                       | Points        | S04        | port ID, country ISO, harbor size, coast, trade share                                                                                                        |
| events                      | Points        | S05        | event ID, source, category, type, date, fatalities, precision flag                                                                                           |
| routes                      | Lines         | S06, S08   | flow ID, scenario, closed chokepoint (if any), length (nautical miles)                                                                                       |
| route_crossings             | Table         | S06        | flow ID, scenario, chokepoint ID                                                                                                                             |
| chokepoint_scores           | Table         | S06–S10    | chokepoint ID, each component, composite score, rank, rank range                                                                                             |
| web_chokepoints, web_routes | Points, lines | S11        | display fields with fixed names: chokepoint name, component scores, composite score, rank, rank range; route segment and its summed trade value, by scenario |

## Step 3: Measure trade exposure (S06, Objective 2)

Trade exposure answers the question “how much REE trade physically passes through this chokepoint?” Script S06 measures it in five steps:

1.  **Route each flow.** The searoute package draws the shortest path between the origin and destination ports along the Eurostat sea-lane network, following the port-level routing approach of Shen et al. [8] and Verschuur, Koks, and Hall [9].

2.  **Apply two scenarios.** Every flow is routed twice. The *baseline* uses the shortest available sea route and measures where REE trade structurally depends on each chokepoint; it drives the main score. The *Red Sea diversion* closes the Suez Canal and Bab el-Mandeb, forcing ships around the Cape of Good Hope. Because most Red Sea shipping rerouted this way through 2025 [3], the diversion shows how actual 2025 practice shifted exposure to other chokepoints. It is not used for the main score because the diversion is itself a response to conflict: scoring it would let Bab el-Mandeb’s insecurity lower its own measured exposure.

3.  **Record crossings.** A flow crosses a chokepoint if its route passes through that chokepoint’s zone. Each crossing is written to the crossing table. Chokepoints on the same corridor, such as the Suez Canal and Bab el-Mandeb, will carry nearly identical trade; they are scored separately because either can be closed on its own, but they are discussed together as one corridor in the results.

4.  **Total the trade.** The trade value of every crossing flow is summed for each chokepoint and scenario (Equation 1). The same sum is repeated using weight in place of value.

5.  **Check against observed shipping.** Each chokepoint’s share of modeled flows under the Red Sea diversion, the scenario that reflects 2025 practice, is compared with its share of 2025 PortWatch transits by container and general-cargo ships, the vessel types most likely to carry processed REE products. Tanker traffic is excluded because it would dominate oil chokepoints such as Hormuz. Agreement is measured with Spearman’s rank correlation, and a value of at least 0.5 is treated as acceptable. Because REE cargo is a tiny fraction of all shipping, this tests whether the routing sends trade through the chokepoints real cargo ships use, not whether REE volumes match. Chokepoints where the two sources disagree sharply are examined individually; errors found this way, such as a port linked to the wrong stretch of coastline, are corrected and logged, and no other adjustments are made to improve agreement. If agreement remains below 0.5, this is reported as a limitation of the trade component.

$$ T_c^{s} = \sum_{f} V_f \cdot x_{f,c}^{s} \qquad (\text{Eq. }1) $$

where *T* is the trade exposure of chokepoint *c* under scenario *s*; *V* is the 2025 trade value of flow *f*; and *x* equals 1 if the route of flow *f* crosses the zone of chokepoint *c* under scenario *s*, and 0 otherwise.

**Inputs:** reconciled trade table, country–port lookup, sea-lane network, and chokepoint zones. **Outputs:** route layers for both scenarios, the crossing table, and trade exposure by value and weight for each chokepoint and scenario.

## Step 4: Measure conflict and tension exposure (S07, Objective 3)

Conflict exposure replaces qualitative security ratings [2, 4] with measured counts of 2025 events near each chokepoint. Violence and tension are always measured separately, because a strait surrounded by diplomatic threats is a different situation from one where ships are being attacked. Script S07 measures each category in two ways and then compares them:

1.  **Distance rings.** Events are counted within 100, 250, and 500 km of each chokepoint, measured along the Earth’s surface (Equation 2). The three distances describe progressively wider zones of threat. At a typical cargo-ship speed of about 15 knots, 100 km (about 3–4 hours of steaming) covers the strait, its approaches, and the adjoining coastline; 250 km (about 9 hours) covers the coastal regions from which attacks on passing ships are launched; and 500 km (about 18 hours) covers the wider sea area ships commit to before entering the passage. Eklund and Butova show that insecurity spills into neighboring waters [4], which is why the measurement reaches beyond the strait itself. The main analysis uses 250 km, which captures this spillover while avoiding both the sparse counts expected at 100 km and the dilution and overlap between neighboring chokepoints at 500 km. The main analysis counts every event equally. A variant, tested in Table 4, weights each violent event by its reported fatalities plus one, so that events without deaths still count. Fatality weighting is not used in the main run because reported fatality figures are often estimates, and many attacks on shipping cause severe disruption with few or no deaths; weighting by deaths would favor land battles over attacks at sea.

2.  **Event density.** Instead of a hard cutoff, each event contributes an amount that declines smoothly with distance, and the total is read at the chokepoint (Equation 3). This is a kernel density estimate with a Gaussian (bell-shaped) kernel, tested with bandwidths of 100, 250, and 500 km, matching the ring distances so the two methods are compared at the same scales.

3.  **Compare the methods.** Both methods produce a ranking of chokepoints, and the two rankings are compared using Spearman’s rank correlation. Close agreement means the result does not depend on the method chosen; disagreement is reported and discussed as a finding. This comparison tests whether country-level spatial methods [15] can be adapted to scoring a single fixed point.

$$ E_c^{k}(r) = \sum_{e \in k} w_e \cdot I\left[\, d(e,c) \le r \,\right] \qquad (\text{Eq. }2) $$

where *E* is the exposure of chokepoint *c* to event category *k* (violence or tension) within distance *r*; *w* is the event weight (1 in the main analysis; fatalities plus one in the variant); *d* is the distance from event *e* to the chokepoint; and *I* equals 1 if the event falls inside the ring and 0 if not.

$$ D_c^{k} = \sum_{e \in k} \frac{1}{2\pi h^{2}} \exp\!\left( -\frac{d(e,c)^{2}}{2h^{2}} \right) \qquad (\text{Eq. }3) $$

where *D* is the density-based exposure of chokepoint *c* to event category *k*, and the bandwidth *h* controls how quickly an event’s influence fades with distance.

Distance rings around neighboring chokepoints can overlap at larger distances. At 500 km, for example, the rings around the Luzon and Taiwan Straits, the Windward and Mona Passages, and the Bosporus and Kerch Strait overlap, so a single event may count toward more than one chokepoint. This is intended: an event near two chokepoints genuinely threatens both, and each chokepoint’s score is calculated independently and never summed with another’s. S07 records which pairs of rings overlap at each distance and the share of events they share, so that similar scores for neighboring chokepoints can be interpreted correctly.

GDELT tension totals for the countries bordering each chokepoint are reported alongside the score as context but do not enter it.

**Inputs:** unified event layer and chokepoint points. **Outputs:** violence and tension exposure for each chokepoint, by method and distance, the method comparison, and a table of overlapping rings.

## Step 5: Measure route redundancy (S08, Objective 4)

Route redundancy turns Ulrichsen and Krane’s descriptive ratings of “primary” and “secondary” chokepoints [3] into a number: how much farther REE trade would travel if a chokepoint were closed. Script S08 reuses the baseline routes from Step 3 and repeats four steps for each chokepoint in turn:

1.  **Close the chokepoint.** The chokepoint is removed from the sea-lane network.

2.  **Reroute the affected trade.** Every flow that passed through the chokepoint is routed again, and its new length is measured in nautical miles.

3.  **Calculate the detour penalty.** The extra distance is expressed as a percentage of the original route length, then averaged across all affected flows with each flow weighted by its trade value (Equation 4).

4.  **Flag flows with no alternative.** If no sea route remains, the flow receives the largest detour penalty observed for any flow, and the share of trade value with no sea alternative is reported separately.

$$ R_c = \frac{\sum_{f \in F_c} V_f \cdot \dfrac{L_f^{-c} - L_f}{L_f}}{\sum_{f \in F_c} V_f} \times 100 \qquad (\text{Eq. }4) $$

where *R* is the detour penalty for chokepoint *c* (percent); *F* is the set of flows that use chokepoint *c*; *V* is each flow’s trade value; *L* is the original route length; and *L* with −*c* is the route length with chokepoint *c* closed. A high value means the trade has no good alternative.

Redundancy is calculated from the baseline routes, the same routes that drive the main score.

**Inputs:** sea-lane network, port-linked trade flows, and the crossing table. **Outputs:** detour penalty and no-alternative share for each chokepoint.

## Step 6: Combine and test the score (S09–S10, Objective 5)

This step answers the research question directly and then combines the three components into one ranked score. Script S09 carries out the first four tasks, and S10 carries out the fifth:

1.  **Test whether trade and conflict coincide.** Before anything is combined, the rankings of trade exposure and conflict exposure are compared using Spearman’s rank correlation. A strong positive value means the chokepoints that carry the most REE trade are also the most conflict-exposed.

2.  **Put the components on one scale.** Trade exposure is measured in dollars, conflict in event counts, and redundancy in percent, so each is rescaled to run from 0 (lowest chokepoint) to 1 (highest) (Equation 5). Trade values and event counts are both highly skewed: one or two routes carry most REE trade, and a few chokepoints near active wars record far more events than the rest. Both are therefore transformed with log(1 + *x*) before rescaling; adding 1 keeps chokepoints with no trade or no events at zero. Redundancy, already a percentage, is rescaled directly.

3.  **Combine the components.** The main score is a weighted sum with equal weights (Equation 6). Equal weighting is the standard starting point for composite indicators when there is no theoretical reason to favor one component [17], follows Lassoued’s equal-weighted Hormuz risk index [5], and gives each of the three measured objectives the same influence. Conflict exposure is the average of the rescaled violence and tension values; the two are weighted equally because the research question names active conflict and elevated geopolitical tension together. Chokepoints are then ranked from highest to lowest score.

4.  **Calculate an alternative score.** A weighted sum lets a chokepoint score high on conflict alone even if almost no REE trade passes through it. An alternative multiplicative score (Equation 7) rewards only chokepoints that are high on every component. For this version, components are rescaled from 0.01 to 1 so that the lowest chokepoint on one component is not automatically scored zero.

5.  **Test robustness (S10).** The scoring is rerun with each alternative in Table 4, changing one choice at a time from the main run; the weight combinations are tested in full. Each chokepoint’s best, worst, and median rank across all runs is recorded. Agreement between each alternative ranking and the main ranking is summarized with Spearman’s rank correlation.

$$ X'_c = \frac{X_c - \min(X)}{\max(X) - \min(X)} \qquad (\text{Eq. }5) $$

where *X* is any one component (after transformation, where applied) and *X′* is its rescaled value for chokepoint *c*.

$$ S_c = w_T\,T'_c + w_C\,C'_c + w_R\,R'_c, \qquad w_T + w_C + w_R = 1 \qquad (\text{Eq. }6) $$

where *S* is the composite score; *T′*, *C′*, and *R′* are the rescaled trade, conflict, and redundancy components; and the weights *w* sum to 1.

$$ S^{\times}_c = (T'_c)^{w_T} \cdot (C'_c)^{w_C} \cdot (R'_c)^{w_R} \qquad (\text{Eq. }7) $$

where *S*× is the multiplicative score, which is high only when all three components are high.

**Table 4.** Sensitivity tests.

| Choice             | Main run                                            | Alternatives tested                                     |
|-------------------|-----------------|------------------------------------|
| Component weights      | Equal (one-third each)                                  | Every combination in steps of 0.1, each weight at least 0.1 |
| Skew transform         | log(1 + x) for trade and conflict                       | No transform                                                |
| Conflict distance      | 250 km                                                  | 100 km and 500 km                                           |
| Conflict method        | Distance rings                                          | Event density                                               |
| Violence vs. tension   | Equal                                                   | Violence only (no GDELT); violence 0.75 and tension 0.25    |
| GDELT source threshold | At least 2 sources                                      | At least 1 source; at least 3 sources                       |
| Event weighting        | Each event counts once                                  | Violent events weighted by fatalities plus one              |
| Port assignment        | Split by PortWatch trade share                          | Single nearest gateway to each trading partner [8]        |
| Trade measure          | Value (US$)                                            | Weight (kg)                                                 |
| Products               | REE metals and compounds (HS 2805.30, 2846.10, 2846.90) | Adding permanent magnets (HS 8505.11)                       |
| Combination rule       | Weighted sum                                            | Multiplicative                                              |

A chokepoint whose rank barely moves across all runs is a robust finding; one that swings widely is reported as uncertain. Because the analysis covers fewer than 30 chokepoints, the rank correlations in Steps 3, 4, and 6 are reported with their p-values but interpreted as measures of agreement rather than as strong statistical tests.

The Red Sea diversion is reported as a separate scenario comparison rather than as a robustness test, because closing two chokepoints changes what is being measured rather than how it is measured. Trade exposure and composite ranks under the two scenarios are compared side by side, showing which chokepoints absorbed the rerouted trade in 2025.

**Inputs:** trade, conflict, and redundancy values from Steps 3–5. **Outputs:** the trade–conflict correlation, composite scores and ranks, the sensitivity results, and the scenario comparison.

## Step 7: Visualize the results (S11)

Script S11 produces the maps, charts, and tables in Table 5 directly from the project database, so every figure updates automatically if the analysis is rerun. Draft figures are produced in Python, and final report maps are finished in QGIS, the same software used for the web map, so the entire workflow uses free and open-source tools. All static world maps use the Equal Earth projection [29], which keeps the relative sizes of regions accurate, over Natural Earth land and coastlines, with colorblind-safe color schemes that match Figures 1 and 2. The web map uses Web Mercator, the standard projection of web basemaps, which suits interactive zooming but exaggerates areas near the poles, so it is not used for printed figures.

**Table 5.** Planned maps and charts.

| Product              | What it shows                                                                                                                 | Question it answers                                  |
|-----------------|---------------------------------|----------------------|
| Ranked vulnerability map | Each chokepoint as a circle sized and shaded by composite score, over sea routes drawn with width proportional to REE trade value | Where is REE trade most exposed?                         |
| Component maps           | Three matching world maps, one each for trade, conflict, and redundancy                                                           | Which factor drives each chokepoint’s score?             |
| Scenario route maps      | Baseline and Red Sea diversion routes side by side                                                                                | How did 2025 rerouting shift exposure?                   |
| Chokepoint close-ups     | 2025 events, distance rings, and the density surface for the highest-scoring chokepoints                                          | What does the conflict picture look like locally?        |
| Trade vs. conflict chart | Each chokepoint plotted by trade exposure and conflict exposure, divided into four quadrants                                      | Do trade and conflict coincide?                          |
| Score breakdown chart    | Horizontal bars, one per chokepoint, split into the three component contributions                                                 | Why does each chokepoint rank where it does?             |
| Sensitivity chart        | Each chokepoint’s main-run rank with a line spanning its best-to-worst rank                                                       | How confident is each ranking?                           |
| Results table            | Component values, composite score, rank, and rank range for every chokepoint                                                      | What are the exact figures?                              |
| Interactive web map      | Scored chokepoints, routes with a scenario toggle, and the conflict density surface, with pop-ups for every chokepoint            | Can every score be checked against the values behind it? |

Because the results combine several components, two routing scenarios, and a range of sensitivity tests across roughly 27 chokepoints, many of them clustered closely in Southeast Asia, static maps alone cannot show them fully. The layers will therefore also be published as an interactive web map, built in QGIS with the qgis2web plugin [31] and hosted on GitHub Pages alongside the project repository. Readers can zoom into dense regions, switch between routing scenarios, and open pop-ups giving each chokepoint’s component values, composite score, rank, and rank range, so that every score can be checked against the values behind it. Only derived layers (scores, routes, and event-density surfaces) are published; raw ACLED event points are not, in keeping with ACLED’s terms of use. To keep the map fast, routes are shown as merged segments carrying their summed trade value rather than as individual flows.

The web map is a snapshot: qgis2web copies the layers into the published files at the moment of export. To keep updates simple, S11 writes display-ready web layers with fixed names and fields to the project database, and a saved QGIS project [30] styles them with fixed class breaks on the 0–1 score scale. After any rerun of the analysis, the map is refreshed by re-exporting it from QGIS and uploading the new files, and its title carries an “as of” date so readers can tell which version they are viewing.

**Inputs:** composite scores and sensitivity results, route and event layers, and Natural Earth basemap layers. **Outputs:** final maps, charts, and tables, display-ready web layers, and the interactive web map hosted on GitHub Pages.

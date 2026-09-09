# External data contract

Only this document is distributed under `data/`. The tiny test fixture is separately located under `tests/fixtures/tiny_case/` and is wholly invented.

Obtain source data from the original providers under their applicable terms. Public download access does not establish redistribution permission. This repository does not redistribute SMART-DS, Texas7k, NOAA inputs, economic source documents, or derived research bundles. Confirm redistribution rights before creating a public data archive. No data license is assigned by this software release.

## Expected directories

| Directory | Contents supplied separately |
|---|---|
| `data/external/data_raw/transmission/` | Authorized Texas7k archive and mapping workbook |
| `data/external/data_raw/distribution/` | Authorized SMART-DS OpenDSS files and profiles |
| `data/external/parsed_distribution/` | Parsed bus, branch and load tables used by the benchmark converter |
| `data/external/hurricane_inputs/` | Filtered storm-track CSV and mainland polygon; optional source metadata inputs |
| `data/external/base_case/` | Original preparation-stage nodes, edges, loads and DC overlays if using the hurricane module directly |
| `data/external/benchmark_case/` | Paper-protocol network, anchor and damage schemas described below |
| `data/external/training/` | State metadata needed to rebuild lookahead training tensors |
| `data/external/evaluation/` | Controlled/fresh manifests, separate damage tables and locked GA cohort |
| `data/external/economics/` | Source-accounting tables needed for benefit-cost calculations |
| `data/external/plot_ready/` | Figure-specific tables, geometry and method palette |

Paths are configurable. Input preparation writes into ignored `outputs/testbed/` by default. To use its output for learning, set `--case-dir` or the training configuration's `paths.case_dir` to the generated benchmark directory.

## Network and restoration schemas

Within `benchmark_case/`:

| File | Required fields / contract |
|---|---|
| `network/nodes.csv` | Unique `node_id`; `voltage_kv`, `degree`, `x`, `y`; boolean `is_load`, `is_substation`, `is_transmission`, `is_distribution`, `is_data_center`, `is_critical`. Geographic conversion also uses `latitude`, `longitude`, `region_id`. |
| `network/edges.csv` | Unique `edge_id`; `from_node`, `to_node`, `length_km`, `voltage_kv`, `phases`, `hurricane_damage_component_type`, `repair_time_hours_placeholder`. Scenario construction additionally requires `is_hurricane_damage_eligible`, `repairable`. |
| `network/loads.csv` | Unique `load_id`; `node_id`, `baseline_p_kw`, `average_p_kw`, `peak_p_kw`, `load_type`, `is_critical`, `voll_usd_per_kwh`. Economic inventory accepts residential and commercial load groups, with critical status treated separately. |
| `network/data_centers.csv` | DC overlay table; construction exports IDs, connected-node identifiers, geography, facility/backup/UPS properties and export metadata. The final support controller takes its operative caps from `anchor_zones.csv`. |
| `dc_scenarios/anchor_zones.csv` | `anchor_zone_id`, JSON-encoded `anchor_load_ids`, `max_support_mw`; geographic construction also exports region, DC, feeder and zone-share fields. |
| `scenario_design/benchmark_hard_hurricane_scenario_manifest.csv` | `scenario_key`, scenario/source-storm identifiers, `split_role`, and named subset membership fields used by training selection. |
| `scenario_design/benchmark_hard_damage_components.csv` | `scenario_key`, `edge_id`, positive `repair_time_hours`; no repeated edge within a scenario. |

The reported simulator recognizes `T_SUPER_SOURCE_V03` as the synthetic supply node. Edge endpoints and load nodes must exist. Baseline loads use kW; support uses MW; repair intervals use hours; interruption accumulation uses kWh; ENS uses MWh. Supply is determined by connectivity plus prescribed support allocation.

## Training and evaluation inputs

`training/state_metadata.csv` supplies `state_id`, `scenario_key`, `step`, `split`, and `greedy_voll_edge_id`. These are archived schema names. The final learner reconstructs source-based candidate features and lookahead targets along the recorded behavior trajectories while preserving train/validation/test membership. A fresh validation bank must not enter training-state construction or checkpoint selection. The full metadata source and its provenance are required; the tiny fixture is not a substitute training dataset.

Evaluation manifests contain `scenario_key` and `eval_subset`, and may contain `random_seed` and source-storm/category metadata. Membership rows can overlap across subsets. Damage-bank overrides use the damage-table schema above. The locked GA cohort has unique scenario keys and retains each original row's seed; when absent, the source implementation uses `32 + row_index`. Preserve cohort order.

Trained checkpoints contain PyTorch state dictionaries plus architecture dimensions, graph edge indices where applicable, and protocol/selection metadata. Provide checkpoints from a trusted source. The package preserves serialized numerical field names and checkpoint keys from the research implementation; public method labels are defined by `policies/registry.py`.

## Economic and plotting inputs

The benefit-cost entry point requires paired resource episode results with `eval_subset`, `display_policy`, `scenario_key`, `dc_resource_configuration`, the archived economic-damage field, `energy_not_served_mwh`, `critical_energy_not_served_mwh`, and `dc_support_used_mwh`.

External economic ledgers supply:

- Event resource costs: `valuation_method`, `event_dc_support_mwh`, `event_provider_fuel_and_vom_dollars`, `selected_local_health_damage_dollars`, `event_climate_damage_dollars`.
- Capital summary: `cost_case`, `estimate_class`, `one_time_incremental_capital_2025_dollars`.
- Host-community incidence inputs: `rate_case`, `fiscal_case`, `annual_local_property_tax_revenue_dollars`, `annual_attributed_residential_rate_burden_dollars`, `annual_local_sales_tax_exemption_burden_dollars`, `annual_routine_generator_health_damage_dollars`, `annual_routine_generator_climate_damage_dollars`. Repeated rows must agree for the selected rate/fiscal case. YAML determines capital, fixed O&M and community shares; precomputed ratios and capital charges are not used.

These are separately maintained accounting inputs, not implied downloadable datasets. The copied enablement/community modules and source-bearing YAML files preserve the underlying cost and incidence calculations. Figure-specific input schemas and paths are described in [reproduction](../docs/reproduction.md).

## Source acquisition

Acquisition identifiers and inherited geographic source lineage are documented in [data provenance](../docs/data_provenance.md). Availability and redistribution terms must be checked with the providers.

The hurricane module expects a previously filtered HURDAT2-derived CSV and a mainland-boundary CSV. Their preprocessing and redistribution rights must accompany any separate bundle. The repository contains the existing downstream parsing, wind, fragility, damage and repair logic, but does not claim that these processed inputs are direct NOAA downloads.


## Analysis tables and compatibility fields

The stable post-processor consumes each explicitly supplied directory's `results_by_episode.csv` and `trajectory_by_step.csv`. Required episode fields include subset/scenario/policy/resource keys, `random_seed`, economic damage, ENS, critical ENS, support MWh, `episode_length`, runtime, action/zone/export-violation counts, `energy_depleted` and the JSON repair sequence. Trajectories require those identity keys and seed, integer step index, interval start/duration/end, initial and current total/critical unserved MW, interval/cumulative ENS and the selected repair. The exact column lists are `EPISODE_COLUMNS` and `TRAJECTORY_COLUMNS` in `evaluation/analysis_tables.py`.

Outputs preserve subset membership in `controlled/results_by_episode.csv` and `dc_anchor/dc_results_by_episode.csv`. Paired summaries feed comparative plots; `dc_anchor/dc_trajectory_unique_by_step.csv` removes verified repeated subset memberships. Fresh unique-bank output requires an explicit manifest. GA comparisons require completed search records and same-cohort policy evaluations. [Reproduction](../docs/reproduction.md) provides the complete command sequence and deduplication rules.

`voll_usd_per_kwh` and `greedy_voll_edge_id` are legacy compatibility fields. They are not the final economic objective. Numerical CSV/tensor keys and checkpoint directory stems from the archive also remain serialized compatibility names. Public display labels use the reported methods and support cases. No renaming of externally archived IDs is required.

# Configuration

Run commands from the repository root or pass `--root`. `DC_RESTORATION_ROOT` provides the same override. All public commands resolve relative paths from that root. No original workspace path is required. Use absolute CLI paths when inputs live elsewhere.

| Configuration | Role |
|---|---|
| `testbed/base.yaml` | Original source-acquisition and preparation parameters |
| `testbed/benchmark.yaml` | Selected composite benchmark, geographic transforms, final damage parameters and storm splits |
| `hazards/hurricane.yaml` | Existing downstream hurricane-module parameters and explicit external input locations |
| `restoration/anchors.yaml` | Reported controller/resource evaluation settings |
| `policies/training.yaml` | Shared final data, feature, BC, A2C and graph-PPO protocol |
| `policies/*.yaml` other than training | Stable method names and links to shared protocols |
| `evaluation/paper.yaml` | Reported methods, subsets, deterministic evaluation and resource settings |
| `evaluation/bounded_ga.yaml` | Locked GA cohort inputs, seed fallback, population and generation limits |
| `evaluation/figures.yaml`, `severity.yaml` | Figure-input paths and economic onset-severity parameters |
| `economics/interruption_damage.yaml` | Price year, CPI ratios, source URLs, dollar anchors and duration shapes |
| `economics/load_classes.yaml` | Recorded load-size classification thresholds |
| `economics/enablement.yaml`, `community.yaml`, `benefit_cost.yaml` | Source-based cost, incidence and final ledger settings |

The final shared training settings come from the matched encoder retraining implementation. Source configuration sections used only for unrelated reports or earlier comparisons are not active public protocols. Retained learning and restoration numeric settings are preserved; path values and public labels have been updated. Benefit-cost semantics have the intentional corrections detailed below. The manifest identifies the exact source file and retained sections for each copy.

Internal tensor, checkpoint, metadata and CSV field names remain compatible with the archive. Public method names are mapped at the interface. Output folders from an archival bundle may therefore need to be mapped to the named paths in these YAML files; no bulk renaming of that bundle is implied.

Override `--case-dir`, `--manifest`, `--damage-components`, `--checkpoint`, `--checkpoint-root`, `--output-dir` or `--device` where exposed by `--help`. The trainer's default CUDA selection follows the research workflow, while validation uses CPU. A device override makes code executable on an available device; it does not establish numerical equivalence to the archived GPU run.

All numerical changes should be saved as an explicitly reviewed protocol with a new output directory. Cached dataset tensors and resume checkpoints should only be reused with their original input/protocol identity. The supplied settings record training steps, seeds, periodic evaluation, resource limits and selection rules; increasing a budget or changing a seed creates a different experiment.

The BetterGrids `file_path` value in `testbed/base.yaml` is the original server-side download-form identifier, not a local filesystem path. It is retained as source acquisition metadata.


## Authoritative benefit-cost configuration

`configs/economics/benefit_cost.yaml` rejects missing and unknown keys. No apparent assumption is silently ignored. The public configuration contains only these inputs:

| Section | Keys and effect |
|---|---|
| `inputs` | `dc_results`, `event_costs`, `capital_summary`, `community_ledger`: explicit CSV paths |
| `analysis` | `selected_policy`, `physical_subset`: exact paired cohort; `event_cost_valuation_method`: exact source-ledger selector; `reference_event_interval_years`: community annualization and reference plot; `event_intervals_years`: lifecycle sensitivity grid |
| `lifecycle` | `capital_case`, `estimate_class`: exactly one capital row; `real_discount_rate`, `service_life_years`: computed CRF; `fixed_export_om_dollars_per_year`: annual incremental export-specific O&M |
| `community` | `rate_case`, `fiscal_case`: recurring-input selectors; `shares`: community fraction of annualized export capital and O&M; `reference_share`: annual comparison and reference plot |
| `environment` | `lifecycle_include_local_air_quality`, `lifecycle_include_climate`: event resource-cost switches; `host_include_local_air_quality`: routine and event local-health incidence switch |
| `output_dir` | Destination of the five calculated CSV ledgers; an explicit CLI `--output-dir` takes precedence |

Rates and lives must be finite, with nonnegative discount rate and positive life. CRF is `d / (1 - (1 + d)^(-n))`, or `1/n` at zero discount. Event intervals must be positive; community shares must lie in [0, 1]. Both grids must contain their configured reference values and must not contain duplicates. Environmental switches must be booleans. The shipped settings include local-health and event climate costs in lifecycle costs and local-health costs in community burden.

The capital ledger must have one row for the selected case and estimate class. The community ledger may repeat recurring inputs across resource/capital/share rows, but every row matching the selected rate/fiscal case must agree on its five recurring amounts. Only property-tax revenue, rate burden, sales-tax-exemption burden, routine local-health damage and routine climate damage are extracted. Community capital and fixed O&M are recomputed from YAML; precomputed ratios, capital charges and payer shares are ignored source-ledger columns, not public options.

Event unit rates preserve the archived arithmetic mean of each selected row's cost divided by delivered support MWh. Empty selections, nonpositive support denominators, nonfinite or negative source costs, duplicate cohort keys, mismatched scenarios and differing paired seeds fail clearly. Input currency columns retain the source's 2025-dollar schema; the calculator does not silently rebase prices.

The full-resource ledger always reports provider cost, local-health damage and climate damage separately and in total. The lifecycle-accounted cost follows its explicit switches. Primary host-community burden always excludes climate; there is no switch that relabels societal climate damage as local incidence. Routine and event climate values remain separate ledger columns and a broader-societal total. Routine generation costs are ordinary portfolio-operation effects, outside the incremental export-enablement lifecycle boundary. The annual typical/disaster comparison uses the Power-only reference and configured reference community share. It counts residual interruption damage in absolute ledgers and avoided damage once in incremental ledgers.

`economics/enablement.yaml` and `community.yaml` retain source-ledger construction parameters. Their legacy schema section keys remain for the copied library helpers; `run_benefit_cost.py` consumes only `benefit_cost.yaml` and its four external inputs. Descriptive assumptions removed from that file are documented here rather than presented as unused controls.

The only support-allocation mode is `critical_first_fixed_priority`. The trainer and evaluator read `anchor_allocation_mode` from the shared training configuration. `restoration/anchors.yaml` records the same final controller specification for review. Serialized feature, reward, checkpoint and metadata keys are retained for archive compatibility; they are not public scientific labels.

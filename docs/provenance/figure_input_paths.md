# Figure input tables

`configs/evaluation/figures.yaml` supplies all roots. `paper_tables` defaults to `outputs/analysis`; this tree is produced by `scripts/prepare_analysis_tables.py` from explicitly supplied completed evaluation directories. Missing families are not synthesized.

| Selector | Required files relative to the analysis root |
|---|---|
| `3` | `controlled/main_hard_subset_average.csv` |
| `4` | `bounded_ga/bounded_ga_comparison_by_scenario.csv`, `bounded_ga/bounded_ga_summary.csv` |
| `5` | `dc_anchor/dc_benefit_vs_no_anchor.csv`, `dc_anchor/dc_results_by_episode.csv` |
| `6` | `dc_anchor/dc_trajectory_unique_by_step.csv`, `controlled/results_by_episode.csv` |
| `7` | `fresh/unique_bank_results_by_episode.csv`, `fresh/results_by_episode.csv` |
| `s1` | `controlled/pairwise_vs_greedy.csv` |
| `s2` | `controlled/results_by_episode.csv` |
| `s3` | `dc_anchor/dc_results_by_episode.csv` |
| `s4` | `dc_anchor/dc_trajectory_unique_by_step.csv`, `controlled/results_by_episode.csv` |
| `s5` | `fresh/results_by_episode.csv`, `fresh/unique_bank_results_by_episode.csv` |

Policy figures require the external `palette` JSON, with a `methods` mapping for the displayed methods and their `color`, `matplotlib_marker` and `matplotlib_line_style` entries. The code normalizes legacy archive method labels on ingestion. Every displayed method, resource and figure-specific subset must have complete input rows.

`benefit_cost_tables` defaults to `outputs/benefit_cost`. Version A reads `anchor_enablement_bcr.csv` and `host_community_ratio.csv`. Version B reads `annual_four_case_comparison.csv` and `annual_incremental_bcr.csv`. The version selectors read only their own required files. [Reproduction](../reproduction.md) lists their exact PNG/PDF/SVG export names.

The `system` figure reads `nodes_for_map.csv`, `edges_for_map.csv`, `special_assets_for_map.csv` and `anchor_zones_for_map.csv` under `geometry_plot_ready/main_text_figures/fig01_system_map/`, plus `geometry_plot_ready/source_copies/geometry/loads.csv`. The `severity` figure uses its configured processed case, controlled/fresh manifests, damage components and reuse inventory. These geometry and scenario inputs remain external. Evaluation post-processing does not reconstruct geometry or generate damage realizations.

The exact numerical fields accessed by each figure remain in `src/dc_restoration/plotting/paper.py`; raw episode and interval contracts are documented in [the data guide](../../data/README.md). Archive numerical field names are retained for compatibility, while captions and exported policy/resource labels use current terminology.

# Execution and paper-artifact reproduction

This package executes the reported methods. Exact paper-artifact reproduction additionally requires the separately maintained processed case, scenario manifests and damage keys, training-state inventory, selected checkpoints, accounting ledgers, plot geometry and palette, and frozen evaluation cohorts. Research datasets, trained weights, frozen results, figures and manuscripts are not bundled. A separate archival repository may distribute authorized artifacts if one is created; no archive identifier or redistribution permission is assumed.

The quick start in [README](../README.md) validates six invented nodes and hand-constructed accounting records. It does not reproduce paper numbers. The release was checked on Windows, CPython 3.13.7, using the isolated dependencies recorded in [requirements-lock.txt](../requirements-lock.txt). That lock is not the archived GPU training environment.

## Evaluation to analysis tables

The following commands run from the release root **after** the required external files have been supplied. They are executable workflow instructions, not runs performed during packaging. `configs/evaluation/paper.yaml` names the controlled manifest and processed case. `outputs/training` must contain the selected checkpoints in the documented compatible layout, or supply `--checkpoint` for a single learned method. Keep historical nonzero-damage and intensity-scaled subsets labeled separately in each manifest.

```sh
python scripts/evaluate.py --method Greedy "Random Top-K" GraphRL-A2C GraphRL-PPO GraphBC MLP-A2C CNN-A2C --resource R1 --output-dir outputs/evaluation_controlled
python scripts/evaluate.py --method Greedy GraphRL-A2C --resource R0 --output-dir outputs/evaluation_r0
python scripts/evaluate.py --method Greedy GraphRL-A2C --resource R1 --output-dir outputs/evaluation_r1
python scripts/evaluate.py --method Greedy GraphRL-A2C --resource E12 --output-dir outputs/evaluation_e12
python scripts/evaluate.py --method Greedy GraphRL-A2C --resource E24 --output-dir outputs/evaluation_e24
python scripts/prepare_analysis_tables.py --controlled-dir outputs/evaluation_controlled --resource-dir outputs/evaluation_r0 --resource-dir outputs/evaluation_r1 --resource-dir outputs/evaluation_e12 --resource-dir outputs/evaluation_e24 --output-dir outputs/analysis
python scripts/run_benefit_cost.py --config configs/economics/benefit_cost.yaml
python scripts/generate_figures.py --figure 3
python scripts/generate_figures.py --figure 5
python scripts/generate_figures.py --figure 6
python scripts/generate_figures.py --figure benefit-cost-a
python scripts/generate_figures.py --figure benefit-cost-b
```

All evaluation directories must be supplied explicitly. Repeating `--controlled-dir`, `--resource-dir`, or `--fresh-dir` combines split method runs in that group. The post-processor validates all inputs before writing and requires an empty output directory. For a partial workflow, pass only the desired input groups; unavailable table families are not created. Missing inputs are never replaced by synthetic data. Individual methods may be analyzed with their Greedy reference, while a full comparative figure requires every displayed comparator and subset.

Each raw directory contains `results_by_episode.csv` and `trajectory_by_step.csv`. Each episode is keyed by subset, scenario, policy and support case; its random seed must agree across paired runs. Unknown methods, resource labels, malformed actions, missing trajectory steps, nonzero reported constraint violations and incomplete reference pairs fail validation. Identical duplicate copies may collapse. Conflicting copies fail. Repeated memberships of one scenario across overlapping subsets may collapse only after all episode and trajectory fields agree, including the seed. No repeated-seed average is inferred.

Controlled pairwise summaries retain overlapping subset memberships. `main_hard_subset_average.csv` equally averages the configured four hard-subset summaries, as in the archived implementation. The unique hard-cohort curves deduplicate scenario membership; they do not pool resource configurations. The archived percentage formulas, bootstrap seed 32 and 2,000 resamples are retained. Zero reference denominators use the archived epsilon convention in summary tables; figure-specific paired distributions retain their existing finite-value filtering. These conventions are recorded rather than silently unified.

## Unseen damage realizations

Provide a separate damage bank and a locked unique-bank manifest with `scenario_key` and `random_seed`. The manifest must exactly match the union of supplied fresh-run scenario/seed pairs for every policy; no inferred unique bank or overlap weighting is applied.

```sh
python scripts/evaluate.py --method Greedy "Random Top-K" GraphRL-A2C GraphRL-PPO GraphBC MLP-A2C CNN-A2C --manifest data/external/evaluation/fresh_subset_manifest.csv --damage-components data/external/evaluation/fresh_damage_components.csv --output-dir outputs/evaluation_fresh
python scripts/prepare_analysis_tables.py --fresh-dir outputs/evaluation_fresh --fresh-unique-manifest data/external/evaluation/fresh_unique_manifest.csv --output-dir outputs/analysis_fresh
```

Set `paper_tables: outputs/analysis_fresh` in a copy of the figure configuration for Figure 7 and the unseen-validation supplementary figure. Alternatively include the fresh arguments with the controlled/resource groups in a single new analysis directory. A unique-bank manifest can include additional realizations only when their completed evaluations are explicitly supplied too.

## Bounded GA comparison

The standalone search uses the final configured population, generation limits and locked cohort. It does not establish a global optimum. Supply a comparison evaluation on the same cohort for Greedy and GraphRL-A2C:

```sh
python scripts/run_bounded_ga.py --config configs/evaluation/bounded_ga.yaml --output-dir outputs/evaluation_ga
python scripts/evaluate.py --method Greedy GraphRL-A2C --manifest data/external/evaluation/ga_evaluation_manifest.csv --damage-components data/external/evaluation/ga_damage_components.csv --resource R1 --output-dir outputs/evaluation_ga_policies
python scripts/prepare_analysis_tables.py --ga-dir outputs/evaluation_ga --ga-evaluation-dir outputs/evaluation_ga_policies --output-dir outputs/analysis_ga
```

The GA evaluation manifest must contain the exact locked cohort plus `eval_subset` and `random_seed`. Set the figure configuration's `paper_tables` to that analysis directory to draw Figure 4, or combine these arguments with the other input groups in one new analysis directory. The GA table preserves `runtime_batch_elapsed_seconds` as elapsed batch completion time; it is not silently relabeled as isolated CPU time.

## Benefit-cost and figure inputs

[The data contract](../data/README.md) lists required accounting columns. YAML controls capital selection, real discount rate, service life, annual export-specific O&M, event intervals, community shares, rate/fiscal selectors and environmental accounting switches. The capital-recovery factor and community-funded charges are computed anew. Precomputed ratios and capital charges from the external community ledger are not used.

The five benefit-cost CSV ledgers feed two independent figure pathways. With the default configuration, exports are below `outputs/figures/benefit_cost/`:

| Selector | PNG exports |
|---|---|
| `benefit-cost-a` | `version_a/Figure08_anchor_enablement_bcr_600dpi.PNG`; `version_a/Figure09_host_community_value_ratio_600dpi.PNG` |
| `benefit-cost-b` | `version_b/Figure08_annual_community_balance_600dpi.PNG`; `version_b/Figure09_value_of_dc_presence_600dpi.PNG` |
| `benefit-cost-all` | Both versions |

PDF and SVG files use the same stems without `_600dpi`. Reference-interval and community-share captions are read from the generated ledgers. No generated figures are included in the package. System and onset-severity figures additionally require the separately maintained geometry, scenario and palette inputs described in [figure input paths](provenance/figure_input_paths.md). Post-processing evaluation outputs does not reconstruct geometry or generate scenarios.

## Scientific compatibility

The allocation default is unchanged: deterministic critical-first allocation with a fixed within-class priority. [Controller provenance](provenance/controller_trace.md) explains its difference from the SI wording. No paper-scale restoration or anchor rollout needs rerunning because of this release's preserved allocation mode. Adopting a newly defined economic dispatch rule would require fresh research-scale evaluations before claiming the same trajectories.

The corrected climate boundary changes community ratios and annual community comparisons. Their benefit-cost figures and manuscript values require updating from corrected ledgers; this needs accounting post-processing, not restoration retraining or trajectory regeneration. Only read-only external-ledger comparison and tiny software checks were performed during release refinement.

The citation credits Kamiar Khayambashi and Negin Alemazkoor, University of Virginia. MIT was selected. Copyright ownership and the notice year need confirmation before release; authorship does not establish ownership. Repository/archive URLs and any future DOI or release date should be added only when known.

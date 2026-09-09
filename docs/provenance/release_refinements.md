# Publication refinements and manual review

Intended repository: `graphrl-data-center-restoration`. Software title: **GraphRL for Data-Center-Supported Distribution Restoration**. The on-disk folder name is retained so the isolated review location and provenance remain stable. The import package remains `dc_restoration`.

This is a curated release derived from the authoritative archive. All changes occurred inside the isolated release folder. The archive and both supplied scientific-reference PDFs were read only. No Git repository, commit, remote or publication action was created.

## Scientific effects

- Numerical benefit-cost changes: routine and event climate damage are excluded from primary host-community burden and the annual local comparison. Local air-quality costs remain. Climate stays in lifecycle event resource costs and separate broader-societal columns. The source-incidence helper uses the same corrected boundary.
- YAML assumptions now control the calculation: capital selection, computed CRF, fixed export O&M, event intervals, payer shares, fiscal/rate cases and environmental switches. Changing these assumptions can change BCR values. With the shipped financial settings, the primary community-boundary correction changes the community and annual-comparison results; the paired restoration benefits are unchanged.
- Anchor allocation is unchanged. The explicit compatibility mode retains the archived critical-first, fixed within-class priority and original tie breaks, partial allocation, aggregate cap and depletion rules. No economic dispatch variant was invented. The external SI wording needs reconciliation with the [controller trace](controller_trace.md).
- No paper-scale restoration or anchor trajectories need rerunning for the preserved controller. Corrected benefit-cost figures and related manuscript values need updated accounting post-processing. Adopting any future dispatch-priority change would require new research-scale trajectory evaluations before claiming identical paper results.
- New post-processing validates explicit completed runs, scenario pairing and cohort memberships, using copied summary formulas. It does not train, generate scenarios, replay research trajectories or run GA.

## Validation and remaining decisions

The lightweight suite increased from **47 to 98 passing tests**. All 62 package modules import, all eight public scripts provide `--help`, and Ruff, formatting, dependency consistency, YAML/CFF parsing, Markdown local links, private-path, credential-pattern, size and artifact checks pass. See [packaging validation](packaging_validation.md) for the final audit and cleanup status.

Read-only external accounting ledgers were validated and compared using temporary target-local outputs. No resulting numerical ledger or research figure is included. Test figure exports are intercepted; no research figures were regenerated. No model training, scenario generation, GA search or paper-scale restoration evaluation was performed.

Included methods: GraphRL-A2C, GraphRL-PPO, GraphBC, MLP-A2C, CNN-A2C, Greedy, Random Top-K and Bounded GA. Datasets, trained checkpoints, frozen paper results, experiment outputs, manuscripts, scratch notebooks and generated figures remain excluded.

**Only release blocker:** confirm copyright ownership and the copyright notice year in [LICENSE](../../LICENSE). MIT is selected. The citation names Kamiar Khayambashi and Negin Alemazkoor with University of Virginia affiliations; authorship is not assumed to establish ownership. DOI, repository URL, release date and any separately licensed external archive are left unset until known.

Outside this task, the external manuscript still has outdated objective labels in the Figure 5 and Figure 6 captions. No manuscript was modified.

Local cleanup remains outstanding: automatic approval review rejected removal of the verified target-local validation environment, packaging scratch directory and Ruff cache with the reason "blocked by policy". They remain ignored and were not deleted. This does not change the candidate-file checks; the physical folder size includes these local files.

## Every changed or new repository candidate

Unchanged files are not listed. The source manifest covers every final candidate except its own self-hash. Temporary validation files and environments are not release candidates.

| File | Change class | Reason |
|---|---|---|
| `CITATION.cff` | Metadata | Set the supplied software title, two authors, University of Virginia affiliations, version and MIT license; omit unavailable identifiers. |
| `CONTRIBUTING.md` | Documentation | Replace provisional authorship text and distinguish authorship from copyright ownership. |
| `README.md` | Documentation | Adopt the requested identity and synthetic-testbed paragraph, correct accounting terminology, describe the post-processing workflow and verified authors. |
| `configs/economics/benefit_cost.yaml` | Numerical control/interface | Replace unused apparent assumptions with active source selectors, explicit financial/environmental controls and four required ledger paths. |
| `configs/economics/community.yaml` | Labels only | Use current resource labels while retaining numerical source parameters. |
| `configs/economics/enablement.yaml` | Labels only | Use current resource and assessment descriptions while retaining numerical source parameters and serialized section keys. |
| `configs/evaluation/figures.yaml` | Paths | Point policy figures at the new consolidated analysis directory. |
| `configs/evaluation/paper.yaml` | Labels only | Use canonical policy labels in public configuration values; preserve seeds, resource codes and selection parameters. |
| `configs/policies/training.yaml` | Interface | Record the unchanged fixed-priority compatibility mode explicitly. |
| `configs/restoration/anchors.yaml` | Specification | Record the same fixed-priority controller mode alongside the final resource specification. |
| `configs/testbed/base.yaml` | Description only | Replace geographic case-identity wording; preserve acquisition identifiers, coordinates and numerical values. |
| `data/README.md` | Documentation | Specify the active accounting and evaluation schemas, legacy fields, source lineage and redistribution boundary. |
| `docs/architecture.md` | Documentation | Include validated analysis tables and distinguish preserved restoration logic from intentional accounting corrections. |
| `docs/configuration.md` | Documentation | Document every active benefit-cost key, selectors, environmental boundaries, CRF and allocation compatibility mode. |
| `docs/data_provenance.md` | Documentation/provenance | Describe a curated derived release, intentional semantic corrections and inherited geographic identifiers. |
| `docs/methods.md` | Documentation | Describe the lifecycle and local-incidence cost boundaries and separate climate reporting. |
| `docs/model_scope.md` | Documentation | State fixed within-class priority and the assumed available, islandable support paths and excluded damage classes. |
| `docs/provenance/controller_trace.md` | New provenance | Trace final trajectory allocation, document exact legacy ordering and SI mismatch, and explain why no new dispatch rule is invented. |
| `docs/provenance/dependency_audit.md` | Validation record | Confirm the unchanged dependencies in the reinstalled, renamed isolated distribution. |
| `docs/provenance/figure_input_paths.md` | Documentation | Describe consolidated table families and the independent benefit-cost export contracts. |
| `docs/provenance/numerical_literal_audit.csv` | Provenance | Refresh the source comparison and explicitly distinguish reviewed accounting changes from unchanged numerical literals. |
| `docs/provenance/packaging_validation.md` | Validation record | Replace obsolete initial-stage claims with current checks, semantic changes, cleanup and the single copyright blocker. |
| `docs/provenance/release_refinements.md` | New review report | Catalogue every changed/new candidate file, scientific consequences, evidence and remaining release decisions. |
| `docs/provenance/source_file_manifest.csv` | Provenance | Preserve every existing source hash and recompute all destination hashes, including new curated files. |
| `docs/reproduction.md` | Documentation | Provide executable evaluation-to-analysis-to-accounting-to-figures commands, cohort rules, required external artifacts and rerun implications. |
| `pyproject.toml` | Metadata/interface | Set the intended distribution name and exact repository description; preserve dc_restoration imports and dependencies. |
| `scripts/prepare_analysis_tables.py` | New interface | Add the stable post-processing entry point without simulation, training or search. |
| `src/dc_restoration/anchors/controller.py` | Interface/documentation | Name and validate the existing compatibility mode and update resource descriptions; preserve sorting and cap/depletion equations. |
| `src/dc_restoration/anchors/priority.py` | New interface | Define the single supported fixed-priority compatibility mode and reject undefined dispatch rules. |
| `src/dc_restoration/cli.py` | Interface | Dispatch authoritative accounting and analysis tables; expose new figure selectors; emit canonical labels and explicit controller/depletion metadata. |
| `src/dc_restoration/economics/benefit_cost.py` | Numerical BCR correction/interface | Exclude climate from primary local burden and annual comparisons, preserve climate ledgers, compute CRF and O&M/share charges from strict YAML, validate source selections, rename figure pathways and use a headless backend. |
| `src/dc_restoration/economics/enablement.py` | Numerical BCR correction/labels | Apply the same climate boundary to the source-incidence helper and retain the broader-societal climate column; update public assessment labels. |
| `src/dc_restoration/economics/load_accounting.py` | Interface/documentation | Validate and document the fixed-priority mode without changing the allocation tuple or numerical rules. |
| `src/dc_restoration/evaluation/analysis_tables.py` | New post-processing | Validate explicit run directories, frozen scenario pairs, trajectories and cohort deduplication; construct all table families consumed by BCR and policy figures. |
| `src/dc_restoration/evaluation/bounded_ga.py` | Labels only | Use the current Greedy and Bounded GA labels in user-facing output; preserve search logic and budgets. |
| `src/dc_restoration/evaluation/statistics.py` | Copied/extracted post-processing | Copy the source before extracting three archived summary formulas; retain numerical formulas, bootstrap logic and source hash. |
| `src/dc_restoration/evaluation/trajectories.py` | Labels only | Display E12 and E24 consistently; preserve trajectory and support calculations. |
| `src/dc_restoration/labels.py` | Interface | Normalize archived policy/resource labels for output and ingestion while retaining numerical schema keys. |
| `src/dc_restoration/plotting/api.py` | Interface | Read consolidated tables and dispatch independent benefit-cost versions with input validation before drawing. |
| `src/dc_restoration/plotting/paper.py` | Interface/labels | Read both archived and canonical policy labels through the display adapter; use analysis-table paths and current support labels. |
| `src/dc_restoration/policies/training.py` | Interface/labels | Read the explicit fixed-priority mode and update progress labels; preserve learning, screening, rewards and checkpoint compatibility. |
| `src/dc_restoration/restoration/environment.py` | Interface | Expose and validate the existing allocation mode in the constructor; preserve numerical physical accounting. |
| `tests/test_analysis_tables.py` | New tests | Exercise invented analysis/BCR/plot schemas, fresh and GA ingestion, duplicates, missing pairs, invalid identities and CLI paths. |
| `tests/test_anchor_constraints.py` | Tests | Verify unchanged critical-first/fixed within-class partial allocation and rejection of undefined modes. |
| `tests/test_benefit_cost.py` | New numerical tests | Check climate boundaries, annual non-double-counting, every active assumption, strict configuration/ledger failures and CLI output selection. |
| `tests/test_benefit_cost_figures.py` | New interface tests | Draw synthetic figure objects with exports intercepted; verify independent versions, exact names and configured reference captions. |
| `tests/test_cli.py` | Tests | Cover the eighth help interface and pass actual tiny evaluator outputs through post-processing. |

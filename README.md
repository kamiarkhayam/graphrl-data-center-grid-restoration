# GraphRL for Data-Center-Supported Distribution Restoration

Research code for graph reinforcement learning and prescribed, power- and energy-constrained data-center support in sequential distribution-system restoration under hurricane-inspired disruptions, with resilience and benefit-cost evaluation.

The study uses a benchmark-derived synthetic multi-region distribution-system testbed constructed from transformed SMART-DS/OpenDSS feeder modules and synthetic source/backbone interfaces. It is an experimental restoration environment and does not represent a specific utility territory.

Intended GitHub repository name: `graphrl-data-center-restoration`. This curated research-software release is staged for manual review. No research data, trained weights, frozen paper results, or manuscripts are bundled. The quick start validates software behavior; it does not reproduce paper numbers.

## Reported methods

| Method | Implementation |
|---|---|
| GraphRL-A2C | Edge-informed graph message passing; lookahead behavior cloning followed by greedy-relative A2C |
| GraphRL-PPO | The same graph architecture and source-based objective, followed by PPO |
| GraphBC | Graph policy trained using lookahead behavior-cloning targets |
| MLP-A2C | Candidate-feature MLP, behavior-cloning initialization and A2C |
| CNN-A2C | Spatial raster encoder with candidate features, behavior-cloning initialization and A2C |
| Greedy | Selects the leading candidate under the source-based economic damage screen |
| Random Top-K | Samples uniformly from valid screened candidates |
| Bounded GA | Fixed-budget search over repair orders, initialized with a Greedy order |

The matched encoder comparison uses K = 50 candidates and a 32,768-step A2C budget. The shared anchor controller uses deterministic critical-first allocation with a fixed within-class priority. The `critical_first_fixed_priority` compatibility mode preserves archived trajectories. Its legacy priority fields are distinct from the source-based repair objective. See [methods](docs/methods.md) and [model scope](docs/model_scope.md).

## Installation

Use Python 3.11 or newer. The staging lock and CI target Windows with Python 3.13. Other platforms should resolve the dependencies from `pyproject.toml`; they have not been validated by this packaging task.

```sh
python -m venv .venv
```

Activate `.venv` using `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell. On the validated platform:

```sh
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e ".[dev]"
```

For a separately resolved environment, use `python -m pip install -e ".[dev]"`. The runtime includes PyTorch, NumPy, pandas, SciPy, NetworkX, Matplotlib, PyYAML, PyArrow, Requests, Shapely, OpenPyXL and DSS-Python. The dev group adds pytest and Ruff. The lock records the isolated staging environment, not the original training environment. Neural research runs used CUDA; tiny checks run on CPU.

## Quick start

Run from this folder:

```sh
python scripts/validate_installation.py
python -m pytest
```

Validation imports every package module and performs a short deterministic rollout on six invented nodes. Tests exercise connectivity, repair validity, candidate masks, support caps and energy depletion, economic units, and policy interfaces. No external download, training, scenario generation, GA, or paper evaluation is part of this quick start.

## External inputs and configuration

Supply your authorized inputs under `data/external/`, or configure other paths. [data/README.md](data/README.md) specifies the network, damage, manifest, training-state and accounting schemas. SMART-DS, NOAA-derived tracks and other source datasets are not redistributed here; confirm their terms directly with their providers.

`configs/testbed/`, `hazards/`, `restoration/`, `policies/`, `evaluation/` and `economics/` contain named destination copies of the relevant configurations. The shared learning protocol is `configs/policies/training.yaml`. Method-specific files reference it. Paths resolve from the repository root; use `--root` or `DC_RESTORATION_ROOT` when running elsewhere. See [configuration](docs/configuration.md).

Input preparation is an explicit stage:

```sh
python scripts/prepare_inputs.py --stage parse_opendss_distribution
python scripts/prepare_inputs.py --stage benchmark --config configs/testbed/benchmark.yaml
```

Other stages are listed by `--help`. Preparation may download or generate substantial inputs when explicitly invoked. It was not run during packaging.

## Training and evaluation

The following commands require the external data bundle. Each learned method uses the same final protocol; GraphBC stops after behavior cloning.

```sh
python scripts/train.py --method GraphRL-A2C
python scripts/train.py --method GraphRL-PPO
python scripts/train.py --method GraphBC
python scripts/train.py --method MLP-A2C
python scripts/train.py --method CNN-A2C
```

`--phase dataset`, `bc`, or `rl` selects an individual preparation/training phase. The `rl` phase requires its behavior-cloning checkpoint. `--case-dir`, `--output-dir`, `--config` and `--device` provide explicit overrides. Reusing an output directory reuses existing dataset caches and the copied trainer's resume logic; use a new output directory when changing inputs or protocol parameters.

```sh
python scripts/evaluate.py --method Greedy "Random Top-K"
python scripts/evaluate.py --method GraphRL-A2C --checkpoint-root outputs/training
python scripts/evaluate.py --method GraphRL-PPO GraphBC MLP-A2C CNN-A2C
python scripts/evaluate.py --method GraphRL-A2C --resource E12 --output-dir outputs/evaluation_e12
python scripts/evaluate.py --method GraphRL-A2C --resource E24 --output-dir outputs/evaluation_e24
```

Provide a locked manifest with `--manifest` and, for a separate damage bank, `--damage-components`. Resource comparisons support the paper's Greedy and GraphRL-A2C policies. Use distinct output folders to retain separate method/resource runs. Historical-nonzero and intensity-scaled stress subsets remain distinct. Evaluation reads trained checkpoints only when a learned method is requested.

```sh
python scripts/run_bounded_ga.py --config configs/evaluation/bounded_ga.yaml
python scripts/prepare_analysis_tables.py --resource-dir outputs/evaluation_r0 --resource-dir outputs/evaluation_r1 --resource-dir outputs/evaluation_e12 --resource-dir outputs/evaluation_e24 --output-dir outputs/analysis
python scripts/run_benefit_cost.py --config configs/economics/benefit_cost.yaml
python scripts/generate_figures.py --figure benefit-cost-a
python scripts/generate_figures.py --figure benefit-cost-b
```

Create all four resource runs before post-processing; the complete evaluation and analysis command sequence is in [reproduction](docs/reproduction.md).

Bounded GA requires its locked cohort and damage table and retains the reported population and generation budgets. Benefit-cost calculations require paired resource results and external source-accounting ledgers. The lifecycle export-enablement assessment includes event climate damage; the host-community incidence assessment excludes climate from its primary denominator and reports it separately. Every benefit-cost YAML key is validated and used. Figure generation requires the final plot-ready table schemas and a method palette, described in [reproduction](docs/reproduction.md); it does not train or evaluate policies. Every script supports `--help`.

## Repository structure

| Path | Purpose |
|---|---|
| `src/dc_restoration/` | Testbed, hazards, restoration, anchors, policies, evaluation, economics and plotting libraries |
| `configs/` | Final protocol and source-based accounting parameters |
| `scripts/` | Stable command interfaces |
| `tests/fixtures/tiny_case/` | Invented deterministic software fixture |
| `docs/` | Architecture, methods, scope, configuration and reproduction guidance |
| `docs/provenance/source_file_manifest.csv` | Source and destination hashes and extraction notes |
| `data/README.md` | External input contracts and redistribution boundary |

Generated inputs, results, weights, figures, logs and local environments are ignored. See [architecture](docs/architecture.md) for the dependency flow and [packaging validation](docs/provenance/packaging_validation.md) for the checks actually performed.

## Citation, license and contact

[CITATION.cff](CITATION.cff) contains the software title and available metadata. The software authors are Kamiar Khayambashi and Negin Alemazkoor, both affiliated with the University of Virginia. No DOI, repository URL, release date, journal citation or ORCID has been assigned here.

The software license is [MIT](LICENSE), as selected for this release. Confirmation of copyright ownership and the copyright notice year is the only release blocker. Authorship does not establish copyright ownership. The software license does not grant rights to external datasets or third-party dependencies. Once a repository URL is established, use its issue tracker for software questions; maintainer contact details will be added at release.

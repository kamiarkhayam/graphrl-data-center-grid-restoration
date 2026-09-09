# Publication-package validation

This curated research-software release derives from the authoritative archive. Intentional accounting corrections are documented in [release refinements](release_refinements.md); it is not represented as purely formatting or import work. No Git repository, commit, remote or publication action was created.

## Verified checks

| Check | Result |
|---|---|
| Lightweight suite | 98 passed (previous package: 47); last full run 4.78 seconds |
| Installation | 62 package modules imported; deterministic six-node fixture completed two repairs |
| Commands | All eight public scripts returned successful `--help` |
| Ruff | `python -m ruff check src scripts tests` passed |
| Formatting | `python -m ruff format --check src scripts tests` passed |
| Dependencies | Editable `graphrl-data-center-restoration` installed in the target-local isolated environment; `pip check` passed |
| Lock | Actual installed runtime/dev/build versions; Windows x86_64, CPython 3.13.7; no dependency changes needed |
| Markdown | Relative local links resolve |
| YAML and CFF | All parsed; supplied title, two authors, University of Virginia affiliations, version and MIT license checked |
| Public labels | Documentation, figure captions and progress messages use current terminology; frozen schema identifiers remain internal compatibility fields |
| Private paths and secrets | No private absolute workspace path in public Python/YAML; no credential patterns found |
| Artifact boundary | No datasets, frozen paper results, checkpoints, manuscripts, notebooks or generated figures are release candidates |
| Source size | Every candidate file is below 1 MiB |
| Provenance | Every candidate except the manifest itself has a destination hash; original source hashes are preserved and rechecked |

The complete source-workspace baseline covers 12,269 original files and 7,730,293,180 bytes. Its baseline-record SHA-256 is `ecf84fa4a5008e46d7b478fa58e5feba98023f7719ff633f17f968165b1e1476`. The final read-only comparison checks SHA-256, size and modification time, plus additions/removals outside the target. Both supplied reference PDFs are separately checked against their pre-refinement hashes.

Workspace audit status: all 12,269 files retain their SHA-256, size and modification time; zero additions or removals outside the target. Both supplied reference PDFs retain their pre-refinement hashes.

Cleanup status: automatic approval review rejected the exact target-local removal of `.venv`, `.packaging` and `.ruff_cache` with "blocked by policy". The directories were confirmed to resolve inside the release root; no deletion occurred. They remain ignored and are not repository candidates. Manual target-local cleanup remains outstanding.

Final repository candidates: **135 files, 966,220 bytes**. Largest file: 79,911 bytes.

Physical folder including ignored local validation files: **34,237 files, 1,196,579,435 bytes**. This is not a post-cleanup size; cleanup was blocked.

## Scientific evidence and limits

Regression tests verify full-resource climate cost, its exclusion from local burden, climate perturbation invariance of primary community ratios, local air-quality costs, and annual absolute/incremental accounting without double counting avoided interruption damage. They also verify supported financial/selective/environmental inputs, strict failures, the explicit fixed-priority allocation mode and the evaluation-to-analysis-to-accounting pipeline.

The external accounting ledgers were read only to validate schema compatibility and compare the corrected boundary. Those computed ledgers were written solely to ignored temporary storage. They are not public results. They remain in ignored packaging scratch storage because automatic approval review blocked cleanup. Plot-path tests create synthetic figure objects and intercept exports; no paper figures were regenerated.

Learning dispatch tests replace expensive functions with test doubles. Neural interface tests use untrained models for a tiny forward pass. Rollouts use only the invented fixture. No model training, scenario generation, GA search, paper-scale restoration or anchor evaluation was run. Exact paper-artifact reproduction has not been established.

Allocation behavior and learning/restoration numerical rules remain unchanged. The numerical-literal comparison is a structural check, not proof of semantic equivalence: it explicitly marks the reviewed accounting/interface changes and is complemented by behavioral tests. The controller trace explains the external SI wording difference. Corrected benefit-cost figures and associated manuscript values need accounting updates; preserved restoration trajectories do not need rerunning because of this refinement.

## Manual release review

The only release blocker is confirmation of copyright ownership and the copyright notice year in [LICENSE](../../LICENSE). MIT is selected; the supplied citation authors are Kamiar Khayambashi and Negin Alemazkoor, University of Virginia. No DOI, repository URL, release date, ORCID, journal citation or email address was invented. Repository/archive identifiers can be added when established, and redistribution rights must be verified before any separate research-data archive is released.

The manifest excludes its own row to avoid a self-referential hash. Its complete destination coverage is verified separately. All paths in the source column are relative archival provenance identifiers, not public method names or bundled artifacts.

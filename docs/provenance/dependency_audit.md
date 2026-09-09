# Dependency audit

Dependencies were installed into a new isolated environment inside the target folder. Imports, package installation, `pip check` and the tiny fixture suite were verified there. The lock does not use a global environment freeze.

| Import | Distribution | Staging version |
|---|---|---|
| `dss` | dss-python | 0.15.7 |
| `matplotlib` | matplotlib | 3.11.1 |
| `networkx` | networkx | 3.6.1 |
| `numpy` | numpy | 2.5.3 |
| `pandas` | pandas | 3.0.5 |
| `pyarrow` | pyarrow | 25.0.1 |
| `requests` | requests | 2.34.2 |
| `scipy` | scipy | 1.18.1 |
| `shapely` | shapely | 2.1.2 |
| `torch` | torch | 2.14.0 |
| `yaml` | PyYAML | 6.0.3 |

OpenPyXL is included because pandas reads the Excel transmission/mapping inputs; it is an engine dependency even though the copied callers import pandas. Pytest and Ruff are the separate development group. Setuptools is the package build dependency. Remaining lock entries are resolved transitive dependencies.

Verified platform: Windows x86_64, CPython 3.13.7. No CUDA training/evaluation was performed. The dependency set is not asserted to reproduce archived GPU numerics.

The distribution was reinstalled as `graphrl-data-center-restoration` for the refinement validation. No runtime, development or transitive dependency changes were needed. The target-local environment was checked before cleanup.

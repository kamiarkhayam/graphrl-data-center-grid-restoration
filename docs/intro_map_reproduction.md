# Introductory data-center growth and hurricane map

This independent spatial workflow reproduces the version-3 regional map. It requires
no restoration results, scenarios, checkpoints, or access to a research workspace.

## Install and configure

From the repository, install `python -m pip install -e ".[mapping]"`. Developers can
use `".[dev,mapping]"`. The optional mapping extra adds GeoPandas, pyproj and
pyogrio; importing the restoration package does not require those libraries.
The verified Windows/Python 3.13 combination is recorded in
`requirements-mapping-lock.txt`, which includes the base dependency lock.

The default [YAML configuration](../src/dc_restoration/spatial/intro_map.yaml) is
bundled in the installed package. Supply a copy with `--config path/to/map.yaml`
to change these three paths:

```yaml
cache_dir: data/external/intro_map/raw
prepared_dir: outputs/intro_map/data_processed
output_dir: outputs/intro_map/figures
```

Every command also accepts `--cache-dir`, `--prepared-dir`, `--output-dir`,
and the established `--root` option. Relative paths use `--root`,
`DC_RESTORATION_ROOT`, or the package's repository/current-directory resolution.
The three directories must be separate, nonoverlapping trees. Repository outputs
must use ignored paths. An explicit absolute destination can be outside a checkout.
Preparation and figure export require empty destination directories; select new
run directories instead of silently overwriting an analysis.

## Fetch, prepare, generate

```sh
dc-fetch-intro-map-sources --config path/to/map.yaml
dc-prepare-intro-map --config path/to/map.yaml
dc-generate-intro-map --config path/to/map.yaml
```

These installed commands work from outside the repository working directory.
Equivalent source-checkout wrappers are `scripts/fetch_intro_map_sources.py`,
`scripts/prepare_intro_map.py`, and `scripts/generate_intro_map.py`.
All support `--help`. Only the fetch command can use the network.

To regenerate from an existing version-3 raw cache, copy its contents into your
configured cache directory, including the `epri/`, `fema/`, and `census/`
subdirectories. Then run:

```sh
dc-fetch-intro-map-sources --config path/to/map.yaml --offline
dc-prepare-intro-map --config path/to/map.yaml
dc-generate-intro-map --config path/to/map.yaml
```

The bundled [source manifest](../src/dc_restoration/spatial/source_manifest.json)
records public URLs, retrieval timestamps, sizes and SHA-256 hashes for 15
archived downloads. It contains metadata only. Fetching archives missing files
locally and records actual retrieval times in the cache's `source_manifest.json`.
Existing cached bytes must match the pinned hashes. No archive, JavaScript source,
source dataset, derived spatial table, or generated figure is shipped in Git.

The EPRI dashboard asset is parsed as text; it is never executed. The extractor
requires the exact array grammar, unique scenario/state/year keys, the full
1,275-record dimensional structure, finite nonnegative values, and national/state
sums within the source's 0.026 rounding tolerance. A changed asset, schema, dataset
or URL fails clearly. To adopt newer source releases, review the sources and
scientific specification and deliberately update the pinned manifest. The code
does not discover and substitute a newer dashboard dataset automatically.

## Scientific and visual specification

Demand growth is medium-scenario annual electricity use in 2030 minus 2024, in
TWh/year. The nine selected coastal states are Texas, Louisiana, Mississippi,
Alabama, Florida, Georgia, South Carolina, North Carolina and Virginia. Circle
area is exactly `8.5 * growth_TWh` in Matplotlib points squared, without a
minimum-size floor. Symbol centers are state interior representative points,
**not facility locations**. EPRI's estimates include cryptocurrency mining.

FEMA `HRCN_AFREQ` is historical county-average annualized hurricane frequency,
in events/year. It is **not a future climate projection** and does not prescribe
restoration-event recurrence. Numerical display boundaries are
`[0, .01, .025, .05, .1, .2, .3, .5]`; they are not FEMA qualitative risk classes.
Not Applicable remains gray and distinct from a measured zero. Values are not
smoothed or interpolated. The downloaded geometry uses a 0.005-degree ArcGIS
simplification tolerance and five-decimal geographic precision. County polygons
are clipped to Census state land outlines; zero-area line/point remnants are
removed without changing polygon area. Joins use identifiers, not county names.

The projection remains EPSG:5070, with extent
`(-1450000, 2650000, 150000, 2380000)` in projected meters. Version-3 label
positions, locator, legends, thin boxed frame, DejaVu Serif font, navy
`#0C457D`, and the ordered gold-anchored `#B89B4D` hazard ramp are retained.

Explicit UTF-8 decoding preserves the downloaded name of county 35013, outside
the regional viewport; the archived version-3 processed table contains a character
encoding artifact in that name. Numerical values, FIPS joins and plotted state
labels are unaffected. Geospatial-library versions can also produce roundoff-scale
coordinate or serialization differences.

## Outputs and checks

All outputs use the established stem:

`FigureIntro_Gulf_Atlantic_data_center_growth_and_hurricane_exposure`

The figure directory contains `_600dpi.PNG` (4320 by 3000 pixels), vector
`.pdf` (7.2 by 5.0 inches), editable-text `.svg`, and `_preview.png`.
It also contains `regional_displayed_values.csv` and `figure_validation.json`.
The prepared directory contains the projection series, growth table, county
frequency table, county/state GeoJSON, national-sum checks, `validation.json`,
and `prepared_manifest.json`. The latter binds processed files to the pinned
raw hashes and is verified before plotting. Figure checks cover selection,
extent, circle areas, ordered colors, label overlap, and locator placement.

Ordinary tests use invented tables, geometries, or mocked HTTP responses and run
offline. A small figure test exercises the layout with synthetic polygons.
Neither tests nor CI download real inputs or run restoration experiments.

## Attribution and redistribution

The matching bibliography entries are in [intro_map_sources.bib](intro_map_sources.bib):

- `epri2026poweringintelligence`: [EPRI Powering Intelligence 2026](https://powering-intelligence.epri.com/dashboard/), retrieved September 9, 2026.
- `fema2025nri`: [FEMA National Risk Index Counties](https://www.arcgis.com/home/item.html?id=39485e8035d446a5bff03259508ae355), December 2025, version 1.20.0, retrieved September 9, 2026. Hazard metadata cover records through 2024.
- `census2024cartographic`: [Census 2024 cartographic state boundaries](https://www2.census.gov/geo/tiger/GENZ2024/shp/), 1:5 million, released in 2025 and retrieved September 9, 2026.

Public accessibility is not a redistribution license. The software MIT license
does not cover these external inputs. No EPRI dashboard data, raw JavaScript,
or derived demand table is redistributed; a blanket redistribution grant was
not established. Review [EPRI's copyright policy](https://www.epri.com/copyright-policy)
and [copyright permission process](https://copyright.epri.com/) before distributing
source materials or publishing a derivative figure.

FEMA's item metadata includes its terms, attribution and non-endorsement
requirements. The cache retains those terms in `fema/item_metadata.json`.
This product uses the Federal Emergency Management Agency's National Risk Index
dataset API or downloadable datasets but is not endorsed by FEMA. The Federal
Government or FEMA cannot vouch for the data or analyses derived from these data
after the data have been retrieved from the Agency's website(s).

The Census Bureau's [public-access policy](https://www2.census.gov/foia/ds_policies/ds027.pdf)
describes the general public-domain status of works created by its employees,
subject to applicable third-party material. This code-only integration redistributes
neither Census boundaries nor FEMA source/derived geometry. Retain source
attribution and review current source terms before sharing real data or figures.

"""Prepare the version-3 spatial inputs from verified local downloads, offline."""

import hashlib
import json

import numpy as np
import pandas as pd
from shapely import make_valid, union_all

from dc_restoration.paths import output_directory
from dc_restoration.spatial.sources import EPRI_ASSET, extract_epri, validate_epri, verify_cache

EXCLUDED = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}


def geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            'Install the optional mapping dependencies: pip install ".[mapping]"'
        ) from exc
    return gpd


def polygon_only(geometry):
    """Remove zero-area coastline-intersection remnants without changing area."""
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    parts = []
    for part in getattr(geometry, "geoms", []):
        if part.geom_type in ("Polygon", "MultiPolygon"):
            parts.append(part)
        elif part.geom_type == "GeometryCollection":
            parts.append(polygon_only(part))
    result = union_all(parts)
    if not np.isclose(result.area, geometry.area, rtol=1e-10, atol=0.001):
        raise ValueError("Polygon extraction changed the intersected land area")
    return result


def demand_growth(dc):
    """Annual TWh in medium 2030 minus medium 2024, joined by explicit year."""
    required = {"scenario", "state", "year", "annual_TWh"}
    if not required <= set(dc):
        raise ValueError("Demand table lacks required columns")
    if dc.duplicated(["scenario", "state", "year"]).any():
        raise ValueError("Duplicate demand scenario/state/year")
    medium = dc[dc.scenario.eq("medium") & dc.year.isin([2024, 2030])]
    growth = medium.pivot(index="state", columns="year", values="annual_TWh")
    if set(growth.columns) != {2024, 2030}:
        raise ValueError("Demand table must contain both 2024 and 2030")
    values = growth.apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Demand baselines and projections must be finite, nonnegative and paired")
    growth = growth.rename(columns={2024: "demand_2024_TWh", 2030: "demand_2030_TWh"})
    growth["growth_TWh"] = growth.demand_2030_TWh - growth.demand_2024_TWh
    if (growth.growth_TWh < 0).any():
        raise ValueError("Negative demand growth is not represented by the version-3 symbols")
    return growth.drop(index=["US", "AK", "HI"], errors="ignore").reset_index()


def join_growth(states, growth):
    if not states.STUSPS.is_unique or not growth.state.is_unique:
        raise ValueError("State identifiers must be unique")
    expected = set(states.STUSPS) - {"DC"}
    if set(growth.state) != expected:
        raise ValueError("EPRI/Census state join is incomplete or has unexpected states")
    result = states.merge(
        growth, left_on="STUSPS", right_on="state", how="left", validate="one_to_one"
    )
    if set(result.loc[result.growth_TWh.isna(), "STUSPS"]) != set(states.STUSPS) & {"DC"}:
        raise ValueError("Only DC may lack an EPRI projection; values are not imputed")
    return result


def validate_hazard(counties):
    if not {"STCOFIPS", "STATEABBRV", "HRCN_AFREQ", "HRCN_RISKR"} <= set(counties):
        raise ValueError("FEMA layer lacks required fields")
    if not counties.STCOFIPS.is_unique or counties.STCOFIPS.isna().any():
        raise ValueError("FEMA county identifiers must be unique and nonempty")
    present = counties.HRCN_AFREQ.notna()
    values = pd.to_numeric(counties.loc[present, "HRCN_AFREQ"], errors="coerce")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            "Hurricane frequencies must be nonnegative and finite; nulls stay Not Applicable"
        )
    if not counties.loc[~present, "HRCN_RISKR"].eq("Not Applicable").all():
        raise ValueError("Missing hurricane values are not designated Not Applicable in FEMA")


def verify_prepared(config):
    sources = verify_cache(config["cache_dir"])
    path = config["prepared_dir"] / "prepared_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Missing prepared_manifest.json; run dc-prepare-intro-map first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["sources"] != {k: v["sha256"] for k, v in sources.items()}:
        raise ValueError("Prepared map data use different source inputs")
    required = {"county_hurricane_frequency.geojson", "state_data_center_growth.geojson"}
    if not required <= set(manifest["files"]):
        raise ValueError("Prepared map manifest is incomplete")
    for name, digest in manifest["files"].items():
        target = (config["prepared_dir"] / name).resolve()
        if target.parent != config["prepared_dir"].resolve():
            raise ValueError("Prepared artifact path escapes its directory")
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prepared map data changed or are missing: {name}; prepare again")
    return sources


def prepare_data(config):
    gpd = geopandas()
    raw = config["cache_dir"]
    sources = verify_cache(raw)
    dc = extract_epri((raw / EPRI_ASSET).read_text(encoding="utf-8"))
    frames = []
    for offset in (0, 2000):
        data = json.loads((raw / f"fema/counties_{offset}.geojson").read_text(encoding="utf-8"))
        frames.append(gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326"))
    counties = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    if len(counties) != 3232:
        raise ValueError("Expected the version-3 FEMA 3,232-county source")
    validate_hazard(counties)
    counties = counties[~counties.STATEABBRV.isin(EXCLUDED)].copy()
    states = gpd.read_file(raw / "census/cb_2024_us_state_5m.zip")
    states = states[~states.STUSPS.isin(EXCLUDED)].copy()
    if len(states) != 49 or set(counties.STATEABBRV) != set(states.STUSPS):
        raise ValueError("FEMA/Census contiguous-US state identifiers do not match")
    growth = demand_growth(dc)
    states = join_growth(states, growth)
    states = states.to_crs(5070)
    counties = counties.to_crs(5070)
    states.geometry = states.geometry.map(make_valid)
    counties.geometry = counties.geometry.map(make_valid)
    land = states.geometry.union_all()
    counties.geometry = counties.geometry.intersection(land).map(polygon_only)
    if (
        counties.geometry.is_empty.any()
        or not counties.geometry.is_valid.all()
        or not counties.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    ):
        raise ValueError("Land clipping must retain valid, nonempty polygon-only counties")
    points = states.geometry.representative_point()
    states["symbol_x"], states["symbol_y"] = points.x, points.y
    out = output_directory(config["prepared_dir"], require_empty=True)
    dc.to_csv(out / "epri_state_projection_series.csv", index=False)
    growth.to_csv(out / "state_growth_2024_2030.csv", index=False)
    counties.drop(columns="geometry").to_csv(out / "county_hurricane_frequency.csv", index=False)
    counties.to_crs(4326).to_file(out / "county_hurricane_frequency.geojson", driver="GeoJSON")
    states.to_crs(4326).to_file(out / "state_data_center_growth.geojson", driver="GeoJSON")
    pd.DataFrame(validate_epri(dc)).to_csv(out / "epri_national_sum_checks.csv", index=False)
    audit = {
        "source_hashes_checked": len(sources),
        "fema_all_counties": 3232,
        "mapped_counties": len(counties),
        "mapped_states_plus_DC": len(states),
        "epri_records": len(dc),
        "mapped_states_with_EPRI_data": len(growth),
        "counties_with_frequency": int(counties.HRCN_AFREQ.notna().sum()),
        "counties_without_frequency": int(counties.HRCN_AFREQ.isna().sum()),
        "counties_with_zero_frequency": int(counties.HRCN_AFREQ.eq(0).sum()),
        "county_max_events_per_year": float(counties.HRCN_AFREQ.max()),
        "FEMA_version": "December 2025 (1.20.0)",
        "map_CRS": "EPSG:5070",
        "missing_DC_projection": "No separate EPRI row; omitted, not imputed.",
        "polygon_only_rendering": "Zero-area line/point remnants removed; polygon areas unchanged.",
        "all_checks_passed": True,
    }
    (out / "validation.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "sources": {k: v["sha256"] for k, v in sources.items()},
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()
        },
    }
    (out / "prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return audit

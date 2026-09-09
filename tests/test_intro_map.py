"""Offline synthetic checks for the introductory map; no research data downloads."""

import hashlib
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests
import yaml
from shapely.geometry import GeometryCollection, LineString, Point, box

from dc_restoration.plotting import intro_map
from dc_restoration.spatial import prepare, sources
from dc_restoration.spatial.configuration import load_map_config
from dc_restoration.spatial.prepare import demand_growth, join_growth, polygon_only, validate_hazard


@pytest.fixture
def invented_epri():
    records = []
    for scenario, years in [
        ("hist", range(2021, 2025)),
        *[(s, range(2024, 2031)) for s in ("low", "medium", "high")],
    ]:
        for state in sorted(sources.STATES | {"US"}):
            for year in years:
                n = (year - 2020) * (50 if state == "US" else 1)
                records.append(
                    f'{{scenario:"{scenario}",state:"{state}",year:{year},nominal_GW:{n},peak_GW:{n},annual_TWh:{n}}}'
                )
    return "As=[" + ",".join(records) + "],La=As"


def test_strict_extraction_and_growth(invented_epri):
    frame = sources.extract_epri(invented_epri)
    assert len(frame) == 1275
    growth = demand_growth(frame.sample(frac=1, random_state=7))
    assert len(growth) == 48
    assert set(growth.growth_TWh) == {6}
    assert set(growth.demand_2024_TWh) == {4}


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("As=[", "Changed=["),
        lambda s: s.replace("],La=As", ",unexpected()],La=As"),
        lambda s: s + s,
        lambda s: s.replace('scenario:"hist"', 'scenario:"unknown"', 1),
        lambda s: s.replace("annual_TWh:4", "annual_TWh:NaN", 1),
        lambda s: s.replace("nominal_GW:4", "nominal_GW:-1", 1),
    ],
)
def test_changed_assets_fail_without_javascript_execution(invented_epri, change):
    with pytest.raises(ValueError):
        sources.extract_epri(change(invented_epri))


@pytest.mark.parametrize("failure", ["duplicate", "missing", "negative", "infinite"])
def test_invalid_demand_pairs(failure):
    frame = pd.DataFrame(
        {
            "state": ["TX", "TX"],
            "scenario": ["medium", "medium"],
            "year": [2024, 2030],
            "annual_TWh": [2.0, 4.0],
        }
    )
    if failure == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif failure == "missing":
        frame = frame.iloc[:1]
    else:
        frame.loc[1, "annual_TWh"] = -1 if failure == "negative" else np.inf
    with pytest.raises(ValueError):
        demand_growth(frame)


def test_state_join_preserves_missing_dc_and_rejects_missing_states():
    states = pd.DataFrame({"STUSPS": ["TX", "VA", "DC"]})
    growth = pd.DataFrame({"state": ["VA", "TX"], "growth_TWh": [4, 6]})
    joined = join_growth(states, growth)
    assert joined.loc[joined.STUSPS.eq("TX"), "growth_TWh"].iloc[0] == 6
    assert joined.loc[joined.STUSPS.eq("DC"), "growth_TWh"].isna().all()
    with pytest.raises(ValueError, match="incomplete"):
        join_growth(states, growth.iloc[:1])


def test_not_applicable_is_distinct_from_zero():
    frame = pd.DataFrame(
        {
            "STCOFIPS": ["a", "b"],
            "STATEABBRV": ["TX", "TX"],
            "HRCN_AFREQ": [0, np.nan],
            "HRCN_RISKR": ["Very Low", "Not Applicable"],
        }
    )
    validate_hazard(frame)
    assert frame.HRCN_AFREQ.isna().sum() == 1 and frame.HRCN_AFREQ.eq(0).sum() == 1
    frame.loc[1, "HRCN_RISKR"] = "Unknown"
    with pytest.raises(ValueError, match="Not Applicable"):
        validate_hazard(frame)


def test_polygon_only_preserves_area_and_discards_zero_area_parts():
    polygon = box(0, 0, 2, 3)
    mixed = GeometryCollection(
        [polygon, Point(8, 8), GeometryCollection([LineString([(3, 0), (4, 0)])])]
    )
    result = polygon_only(mixed)
    assert result.geom_type == "Polygon" and result.equals(polygon) and result.area == 6
    assert polygon_only(GeometryCollection([Point(0, 0)])).is_empty


def test_symbol_area_and_ordered_hazard_palette():
    np.testing.assert_array_equal(intro_map.symbol_areas([0, 1, 2, 10]), [0, 8.5, 17, 85])
    for bad in ([-1], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            intro_map.symbol_areas(bad)
    from matplotlib.colors import to_rgb

    rgb = np.array([to_rgb(c) for c in intro_map.HAZARD_COLORS])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    assert np.all(np.diff(linear @ [0.2126, 0.7152, 0.0722]) < 0)
    assert intro_map.BLUE == "#0C457D" and intro_map.GOLD == "#B89B4D"


def test_config_paths_are_explicit_and_do_not_create_files(monkeypatch, tmp_path):
    monkeypatch.setenv("DC_RESTORATION_ROOT", str(tmp_path))
    result = load_map_config()
    assert result["cache_dir"] == tmp_path / "data/external/intro_map/raw"
    assert not list(tmp_path.iterdir())
    config = tmp_path / "paths.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "cache_dir": "data/raw",
                "prepared_dir": "outputs/data",
                "output_dir": "outputs/figures",
            }
        )
    )
    result = load_map_config(config, output_dir="outputs/custom")
    assert result["output_dir"] == tmp_path / "outputs/custom"
    with pytest.raises(ValueError, match="overlap"):
        load_map_config(config, output_dir="outputs/data/figures")
    with pytest.raises(ValueError, match="ignored"):
        load_map_config(config, output_dir="src")


def test_checksum_cache_and_mocked_fetch(monkeypatch, tmp_path, invented_epri):
    content = invented_epri.encode()
    reference = {
        sources.EPRI_ASSET: {
            "url": "https://example.invalid/pinned.js",
            "retrieved_utc": "original",
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    }
    monkeypatch.setattr(sources, "source_manifest", lambda: reference)
    calls = []
    client = SimpleNamespace(
        get=lambda *a, **k: (
            calls.append(a),
            SimpleNamespace(content=content, url=a[0], raise_for_status=lambda: None),
        )[1]
    )
    config = {"cache_dir": tmp_path / "cache"}
    sources.fetch_sources(config, session=client)
    receipt = (config["cache_dir"] / "source_manifest.json").read_bytes()
    sources.fetch_sources(config, session=client)
    assert (config["cache_dir"] / "source_manifest.json").read_bytes() == receipt
    sources.fetch_sources(config, offline=True)
    assert len(calls) == 1
    (config["cache_dir"] / sources.EPRI_ASSET).write_bytes(content + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        sources.fetch_sources(config, offline=True)


def test_failed_download_never_substitutes_data(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sources,
        "source_manifest",
        lambda: {"x.js": {"url": "https://example.invalid/x.js", "bytes": 1, "sha256": "x"}},
    )

    def fail(*a, **k):
        raise requests.HTTPError("404")

    with pytest.raises(RuntimeError, match="no replacement"):
        sources.fetch_sources({"cache_dir": tmp_path / "cache"}, session=SimpleNamespace(get=fail))
    assert not (tmp_path / "cache").exists()


def test_changed_download_is_rejected_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sources,
        "source_manifest",
        lambda: {
            "x.js": {
                "url": "https://example.invalid/x.js",
                "bytes": 1,
                "sha256": hashlib.sha256(b"a").hexdigest(),
            }
        },
    )
    response = SimpleNamespace(content=b"b", raise_for_status=lambda: None)
    client = SimpleNamespace(get=lambda *a, **k: response)
    with pytest.raises(ValueError, match="checksum"):
        sources.fetch_sources({"cache_dir": tmp_path / "cache"}, session=client)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("case", ["valid", "changed", "foreign_sources", "missing", "escape"])
def test_prepared_manifest_binds_inputs_and_artifacts(monkeypatch, tmp_path, case):
    records = {"invented_source": {"sha256": "synthetic"}}
    monkeypatch.setattr(prepare, "verify_cache", lambda p: records)
    names = ["county_hurricane_frequency.geojson", "state_data_center_growth.geojson"]
    for name in names:
        (tmp_path / name).write_bytes(b"invented artifact")
    manifest = {
        "sources": {"invented_source": "synthetic"},
        "files": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names
        },
    }
    if case == "changed":
        (tmp_path / names[0]).write_bytes(b"changed")
    elif case == "foreign_sources":
        manifest["sources"]["invented_source"] = "another cohort"
    elif case == "missing":
        del manifest["files"][names[0]]
    elif case == "escape":
        manifest["files"]["../outside"] = "unused"
    (tmp_path / "prepared_manifest.json").write_text(json.dumps(manifest))
    config = {"cache_dir": tmp_path / "cache", "prepared_dir": tmp_path}
    if case == "valid":
        assert prepare.verify_prepared(config) == records
    else:
        with pytest.raises(ValueError):
            prepare.verify_prepared(config)


@pytest.mark.parametrize(
    "entry,stage",
    [("fetch_main", "fetch"), ("prepare_main", "prepare"), ("generate_main", "generate")],
)
def test_cli_dispatch(monkeypatch, tmp_path, entry, stage):
    from dc_restoration.spatial import cli, prepare

    calls = []
    monkeypatch.setattr(sources, "fetch_sources", lambda c, **k: calls.append((c, k)))
    monkeypatch.setattr(prepare, "prepare_data", lambda c: calls.append((c, {})))
    monkeypatch.setattr(intro_map, "generate_map", lambda c: calls.append((c, {})))
    argv = [
        "map",
        "--cache-dir",
        str(tmp_path / "raw"),
        "--prepared-dir",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "figures"),
    ]
    if stage == "fetch":
        argv.append("--offline")
    monkeypatch.setattr(sys, "argv", argv)
    getattr(cli, entry)()
    assert calls[0][0]["cache_dir"] == tmp_path / "raw"
    if stage == "fetch":
        assert calls[0][1] == {"offline": True}


@pytest.mark.parametrize(
    "script", ["fetch_intro_map_sources", "prepare_intro_map", "generate_intro_map"]
)
def test_script_help(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", str(root / "scripts" / (script + ".py")), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--cache-dir" in result.stdout


def test_imports_work_without_mapping_extras_or_side_effects():
    code = """
import builtins, importlib
from pathlib import Path
import matplotlib.pyplot, numpy, pandas, requests, shapely, yaml
original=builtins.__import__
def blocked(name,*args,**kwargs):
    if name.split('.')[0] in {'geopandas','pyproj','pyogrio'}:
        raise AssertionError('Optional mapping import during package import')
    return original(name,*args,**kwargs)
def fail(*args,**kwargs):raise AssertionError('Import side effect')
builtins.__import__=blocked
Path.mkdir=Path.write_text=Path.write_bytes=fail
requests.Session.get=fail
matplotlib.pyplot.figure=fail
for name in ['spatial.sources','spatial.prepare','spatial.configuration','spatial.cli','plotting.intro_map']:
    importlib.import_module('dc_restoration.'+name)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_synthetic_map_smoke(tmp_path):
    gpd = pytest.importorskip("geopandas")
    import matplotlib.pyplot as plt

    rows = []
    for i, code in enumerate(intro_map.COASTAL):
        x, y = -500000 + 300000 * i, 900000
        rows.append(
            {
                "STUSPS": code,
                "NAME": code,
                "demand_2024_TWh": 1,
                "demand_2030_TWh": 2 + i,
                "growth_TWh": 1 + i,
                "symbol_x": x,
                "symbol_y": y,
                "geometry": box(x - 10000, y - 10000, x + 10000, y + 10000),
            }
        )
    for i, code in enumerate(["OK", "AR", "TN", "KY", "MO", "KS", "OH", "IN", "PA", "WV"]):
        x, y = 100000 + 150000 * i, 2200000
        rows.append(
            {
                "STUSPS": code,
                "NAME": code,
                "demand_2024_TWh": 1,
                "demand_2030_TWh": 2,
                "growth_TWh": 1,
                "symbol_x": x,
                "symbol_y": y,
                "geometry": box(x - 5000, y - 5000, x + 5000, y + 5000),
            }
        )
    states = gpd.GeoDataFrame(rows, crs="EPSG:5070")
    counties = states[["geometry"]].copy()
    counties["HRCN_AFREQ"] = [0, np.nan] + [0.1] * (len(counties) - 2)
    fig, selected, qa = intro_map.draw_map(counties, states)
    try:
        fig.savefig(tmp_path / "invented_map.png", dpi=40)
        assert (tmp_path / "invented_map.png").stat().st_size > 0
        assert tuple(fig.get_size_inches()) == (7.2, 5.0)
        assert len(selected) == 9 and qa["label_overlap_check_passed"]
        assert qa["hazard_class_boundaries_events_per_year"] == [
            0,
            0.01,
            0.025,
            0.05,
            0.1,
            0.2,
            0.3,
            0.5,
        ]
    finally:
        plt.close(fig)

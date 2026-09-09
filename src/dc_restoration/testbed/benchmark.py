from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dc_restoration.paths import repository_root
from dc_restoration.restoration.simulator import (
    RestorationSimulator,
    _haversine_km,
    _md_table,
    _resolve_path,
    _safe_bool,
    _write_text,
    _write_yaml,
)


def _make_dirs(case_dir: Path, out_dir: Path) -> Dict[str, Path]:
    names = [
        "candidate_downloads",
        "candidate_raw",
        "candidate_converted",
        "selected_case",
        "network",
        "dc_scenarios",
        "hurricane_scenarios",
        "restoration_env",
        "scenario_design",
        "diagnostics",
        "plots",
        "reports",
    ]
    out_names = [
        "configs",
        "candidate_search",
        "downloads",
        "candidate_audits",
        "conversions",
        "selected_case",
        "scenario_design",
        "diagnostics",
        "plots",
        "reports",
        "manifests",
    ]
    d = {"case": case_dir, "output": out_dir}
    for n in names:
        d[f"case_{n}"] = case_dir / n
    for n in out_names:
        d[n] = out_dir / n
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _clean_bus(bus: Any) -> str:
    # OpenDSS buses may include phase suffixes; restoration topology uses bus-level nodes.
    s = str(bus).strip().strip('"').strip("'")
    return s.split(".")[0]


def _load_raw_distribution(
    root: Path, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed = _resolve_path(root, config["paths"]["parsed_distribution_dir"])
    buses = pd.read_csv(parsed / "raw_buses.csv")
    lines = pd.read_csv(parsed / "raw_lines.csv")
    loads = pd.read_csv(parsed / "raw_loads.csv")
    transformers = pd.read_csv(parsed / "raw_transformers.csv")
    return buses, lines, loads, transformers


def convert_smartds_candidate(
    config: Dict[str, Any], root: Path, d: Dict[str, Path]
) -> Dict[str, pd.DataFrame]:
    buses, lines, loads_raw, transformers = _load_raw_distribution(root, config)
    raw_dir = _resolve_path(root, config["paths"]["parsed_distribution_dir"])
    dest_raw = d["case_candidate_raw"] / "SMART_DS_P49U_raw_OpenDSS"
    dest_raw.mkdir(parents=True, exist_ok=True)
    for fname in [
        "raw_buses.csv",
        "raw_lines.csv",
        "raw_loads.csv",
        "raw_transformers.csv",
        "raw_switches.csv",
    ]:
        src = raw_dir / fname
        if src.exists():
            shutil.copy2(src, dest_raw / fname)

    raw_cent_lat = float(buses["latitude"].mean())
    raw_cent_lon = float(buses["longitude"].mean())
    all_nodes: List[pd.DataFrame] = []
    all_edges: List[pd.DataFrame] = []
    all_loads: List[pd.DataFrame] = []
    all_substations: List[Dict[str, Any]] = []
    all_feeders: List[Dict[str, Any]] = []
    regions = []
    rng = np.random.default_rng(int(config["random_seed"]))

    for spec in config["geography"]["regions"]:
        prefix = spec["id_prefix"]
        scale = float(spec["coordinate_scale"])
        lat_shift = float(spec["latitude_shift"])
        lon_shift = float(spec["longitude_shift"])
        load_scale = float(spec["load_scale"])
        sub_id = f"{prefix}__SUB_BENCH"
        source_node = f"{prefix}__T_SOURCE"
        sub_node = f"{prefix}__SUBSTATION_SOURCE"
        b = buses.copy()
        b["raw_bus_id_clean"] = b["raw_bus_id"].map(_clean_bus)
        b["node_id"] = prefix + "__" + b["raw_bus_id_clean"].astype(str)
        b["source_benchmark_id"] = b["raw_bus_id_clean"]
        b["substation_id"] = sub_id
        b["feeder_id"] = prefix + "__" + b["feeder_id"].astype(str)
        b["latitude"] = (
            raw_cent_lat
            + (pd.to_numeric(b["latitude"], errors="coerce") - raw_cent_lat) * scale
            + lat_shift
        )
        b["longitude"] = (
            raw_cent_lon
            + (pd.to_numeric(b["longitude"], errors="coerce") - raw_cent_lon) * scale
            + lon_shift
        )
        b["x"] = b["longitude"]
        b["y"] = b["latitude"]
        b["layer"] = "distribution"
        b["node_type"] = "benchmark_raw_bus"
        b["voltage_kv"] = pd.to_numeric(b["base_kv"], errors="coerce").fillna(12.47)
        b["phase"] = b["phases"].astype(str).str.count(r"\.") + 1
        b["is_load"] = False
        b["is_generator"] = False
        b["is_substation"] = False
        b["is_transmission"] = False
        b["is_distribution"] = True
        b["is_data_center"] = False
        b["is_critical"] = False
        b["load_type"] = "unknown"
        b["voll_usd_per_kwh"] = 10.0
        b["baseline_p_kw"] = 0.0
        b["baseline_q_kvar"] = 0.0
        b["peak_p_kw"] = 0.0
        b["average_p_kw"] = 0.0
        b["priority_class"] = 5
        b["region_id"] = spec["region_id"]
        b["region_label"] = spec["region_id"]
        node_cols = [
            "node_id",
            "source_benchmark_id",
            "layer",
            "node_type",
            "substation_id",
            "feeder_id",
            "voltage_kv",
            "latitude",
            "longitude",
            "x",
            "y",
            "phase",
            "is_load",
            "is_generator",
            "is_substation",
            "is_transmission",
            "is_distribution",
            "is_data_center",
            "is_critical",
            "load_type",
            "voll_usd_per_kwh",
            "baseline_p_kw",
            "baseline_q_kvar",
            "peak_p_kw",
            "average_p_kw",
            "priority_class",
            "region_id",
            "region_label",
        ]
        nodes = b[node_cols].copy()

        # Add source/substation interface nodes.
        sub_lat = float(nodes["latitude"].mean()) + 0.035
        sub_lon = float(nodes["longitude"].mean()) - 0.035
        source_rows = pd.DataFrame(
            [
                {
                    "node_id": source_node,
                    "source_benchmark_id": "synthetic_transmission_source",
                    "layer": "transmission",
                    "node_type": "benchmark_region_source",
                    "substation_id": sub_id,
                    "feeder_id": sub_id,
                    "voltage_kv": 69.0,
                    "latitude": sub_lat + 0.015,
                    "longitude": sub_lon - 0.015,
                    "x": sub_lon - 0.015,
                    "y": sub_lat + 0.015,
                    "phase": 3,
                    "is_load": False,
                    "is_generator": True,
                    "is_substation": False,
                    "is_transmission": True,
                    "is_distribution": False,
                    "is_data_center": False,
                    "is_critical": False,
                    "load_type": "source",
                    "voll_usd_per_kwh": 0,
                    "baseline_p_kw": 0,
                    "baseline_q_kvar": 0,
                    "peak_p_kw": 0,
                    "average_p_kw": 0,
                    "priority_class": 0,
                    "region_id": spec["region_id"],
                    "region_label": spec["region_id"],
                },
                {
                    "node_id": sub_node,
                    "source_benchmark_id": "synthetic_substation_interface",
                    "layer": "interface",
                    "node_type": "benchmark_substation_interface",
                    "substation_id": sub_id,
                    "feeder_id": sub_id,
                    "voltage_kv": 12.47,
                    "latitude": sub_lat,
                    "longitude": sub_lon,
                    "x": sub_lon,
                    "y": sub_lat,
                    "phase": 3,
                    "is_load": False,
                    "is_generator": False,
                    "is_substation": True,
                    "is_transmission": False,
                    "is_distribution": False,
                    "is_data_center": False,
                    "is_critical": False,
                    "load_type": "source",
                    "voll_usd_per_kwh": 0,
                    "baseline_p_kw": 0,
                    "baseline_q_kvar": 0,
                    "peak_p_kw": 0,
                    "average_p_kw": 0,
                    "priority_class": 0,
                    "region_id": spec["region_id"],
                    "region_label": spec["region_id"],
                },
            ]
        )
        nodes = pd.concat([nodes, source_rows], ignore_index=True)

        # Build edge table from lines and transformers.
        line_edges = lines.copy()
        line_edges["edge_id"] = prefix + "__LINE__" + line_edges["raw_edge_id"].astype(str)
        line_edges["source_branch_id"] = line_edges["raw_edge_id"].astype(str)
        line_edges["from_node"] = prefix + "__" + line_edges["from_bus"].map(_clean_bus)
        line_edges["to_node"] = prefix + "__" + line_edges["to_bus"].map(_clean_bus)
        line_edges["component_type"] = np.where(
            _safe_bool(line_edges["is_switch"]), "switch", "overhead_line"
        )
        line_edges["hurricane_damage_component_type"] = np.where(
            _safe_bool(line_edges["is_switch"]), "switch", "overhead_line"
        )
        line_edges["length_km"] = (
            pd.to_numeric(line_edges["length"], errors="coerce").fillna(0.1) * scale
        )
        line_edges["edge_type"] = np.where(
            _safe_bool(line_edges["is_switch"]), "benchmark_switch", "benchmark_line"
        )
        line_edges["voltage_kv"] = 12.47
        line_edges["phases"] = (
            pd.to_numeric(line_edges["phases"], errors="coerce").fillna(3).astype(int)
        )
        line_edges["is_switch"] = _safe_bool(line_edges["is_switch"])
        line_edges["is_transformer"] = False
        line_edges["is_overhead"] = True
        line_edges["is_underground"] = False
        line_edges["repairable"] = True
        line_edges["is_hurricane_damage_eligible"] = True
        line_edges["repair_time_hours_placeholder"] = np.where(
            line_edges["is_switch"], 3.0, np.clip(4.0 + line_edges["length_km"] * 2.0, 4.0, 16.0)
        )
        line_edges["substation_id"] = sub_id
        line_edges["feeder_id"] = prefix + "__" + line_edges["feeder_id"].astype(str)
        line_edges["region_id"] = spec["region_id"]

        xf = transformers.copy()
        xf["edge_id"] = (
            prefix
            + "__XFMR__"
            + xf["raw_transformer_id"].astype(str).str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        )
        xf["source_branch_id"] = xf["raw_transformer_id"].astype(str)
        xf["from_node"] = prefix + "__" + xf["bus1"].map(_clean_bus)
        xf["to_node"] = prefix + "__" + xf["bus2"].map(_clean_bus)
        xf["component_type"] = "transformer"
        xf["hurricane_damage_component_type"] = "transformer"
        xf["length_km"] = 0.03
        xf["edge_type"] = "benchmark_transformer"
        xf["voltage_kv"] = pd.to_numeric(xf["kv_wdg1"], errors="coerce").fillna(12.47)
        xf["phases"] = pd.to_numeric(xf["phases"], errors="coerce").fillna(3).astype(int)
        xf["is_switch"] = False
        xf["is_transformer"] = True
        xf["is_overhead"] = False
        xf["is_underground"] = False
        xf["repairable"] = True
        xf["is_hurricane_damage_eligible"] = True
        xf["repair_time_hours_placeholder"] = 14.0
        xf["substation_id"] = sub_id
        xf["feeder_id"] = prefix + "__" + xf["feeder_id"].astype(str)
        xf["region_id"] = spec["region_id"]

        edge_cols = [
            "edge_id",
            "source_branch_id",
            "from_node",
            "to_node",
            "layer",
            "edge_type",
            "component_type",
            "voltage_kv",
            "phases",
            "length_km",
            "is_switch",
            "is_transformer",
            "is_overhead",
            "is_underground",
            "repairable",
            "is_hurricane_damage_eligible",
            "hurricane_damage_component_type",
            "repair_time_hours_placeholder",
            "substation_id",
            "feeder_id",
            "region_id",
        ]
        line_edges["layer"] = "distribution"
        xf["layer"] = "distribution"
        edges = pd.concat([line_edges[edge_cols], xf[edge_cols]], ignore_index=True)

        # Add missing endpoint nodes from edge endpoints with synthetic local coordinates.
        known = set(nodes["node_id"].astype(str))
        endpoints = set(edges["from_node"].astype(str)) | set(edges["to_node"].astype(str))
        missing = sorted(endpoints - known)
        if missing:
            feeder_centers = (
                nodes.groupby("feeder_id")[["latitude", "longitude"]].mean().to_dict("index")
            )
            missing_rows = []
            for j, node_id in enumerate(missing):
                feeder_candidates = edges.loc[
                    (edges["from_node"].eq(node_id)) | (edges["to_node"].eq(node_id)), "feeder_id"
                ]
                feeder_id = feeder_candidates.iloc[0] if not feeder_candidates.empty else sub_id
                center = feeder_centers.get(feeder_id, {"latitude": sub_lat, "longitude": sub_lon})
                missing_rows.append(
                    {
                        "node_id": node_id,
                        "source_benchmark_id": node_id.split("__", 1)[-1],
                        "layer": "distribution",
                        "node_type": "benchmark_missing_endpoint",
                        "substation_id": sub_id,
                        "feeder_id": feeder_id,
                        "voltage_kv": 12.47,
                        "latitude": float(center["latitude"]) + float(rng.normal(0, 0.001)),
                        "longitude": float(center["longitude"]) + float(rng.normal(0, 0.001)),
                        "x": float(center["longitude"]),
                        "y": float(center["latitude"]),
                        "phase": 3,
                        "is_load": False,
                        "is_generator": False,
                        "is_substation": False,
                        "is_transmission": False,
                        "is_distribution": True,
                        "is_data_center": False,
                        "is_critical": False,
                        "load_type": "unknown",
                        "voll_usd_per_kwh": 10,
                        "baseline_p_kw": 0,
                        "baseline_q_kvar": 0,
                        "peak_p_kw": 0,
                        "average_p_kw": 0,
                        "priority_class": 5,
                        "region_id": spec["region_id"],
                        "region_label": spec["region_id"],
                    }
                )
            nodes = pd.concat([nodes, pd.DataFrame(missing_rows)], ignore_index=True)

        # Loads and critical assignment.
        l = loads_raw.copy()
        l["node_id"] = prefix + "__" + l["bus"].map(_clean_bus)
        l["load_id"] = prefix + "__LD__" + l["raw_load_id"].astype(str)
        l["source_benchmark_load_id"] = l["raw_load_id"]
        l["substation_id"] = sub_id
        l["feeder_id"] = prefix + "__" + l["feeder_id"].astype(str)
        l["baseline_p_kw"] = pd.to_numeric(l["p_kw"], errors="coerce").fillna(0.0) * load_scale
        l["baseline_q_kvar"] = pd.to_numeric(l["q_kvar"], errors="coerce").fillna(0.0) * load_scale
        l["peak_p_kw"] = l["baseline_p_kw"] * 1.25
        l["average_p_kw"] = l["baseline_p_kw"] * 0.62
        l["load_type"] = l["inferred_customer_class"].fillna("commercial").astype(str)
        l["is_critical"] = False
        for feeder_id, group in l.groupby("feeder_id"):
            ncrit = max(1, int(math.ceil(len(group) * 0.045)))
            top_idx = group.sort_values("baseline_p_kw", ascending=False).head(ncrit).index
            l.loc[top_idx, "is_critical"] = True
        l["voll_usd_per_kwh"] = np.where(
            l["is_critical"],
            100.0,
            np.where(l["load_type"].str.contains("commercial", case=False, na=False), 25.0, 10.0),
        )
        l["is_data_center"] = False
        l["region_id"] = spec["region_id"]
        l["region_label"] = spec["region_id"]
        load_cols = [
            "load_id",
            "source_benchmark_load_id",
            "node_id",
            "feeder_id",
            "substation_id",
            "load_type",
            "voll_usd_per_kwh",
            "baseline_p_kw",
            "baseline_q_kvar",
            "peak_p_kw",
            "average_p_kw",
            "is_critical",
            "is_data_center",
            "profile_id",
            "region_id",
            "region_label",
        ]

        # Mark node load attributes.
        load_agg = l.groupby("node_id").agg(
            baseline_p_kw=("baseline_p_kw", "sum"),
            baseline_q_kvar=("baseline_q_kvar", "sum"),
            peak_p_kw=("peak_p_kw", "sum"),
            average_p_kw=("average_p_kw", "sum"),
            is_critical=("is_critical", "max"),
            voll_usd_per_kwh=("voll_usd_per_kwh", "max"),
            load_type=(
                "load_type",
                lambda s: (
                    "critical"
                    if any(l.loc[s.index, "is_critical"])
                    else s.mode().iloc[0]
                    if not s.mode().empty
                    else "unknown"
                ),
            ),
        )
        nodes = nodes.set_index("node_id")
        for nid, vals in load_agg.iterrows():
            if nid in nodes.index:
                nodes.loc[
                    nid,
                    [
                        "baseline_p_kw",
                        "baseline_q_kvar",
                        "peak_p_kw",
                        "average_p_kw",
                        "voll_usd_per_kwh",
                        "load_type",
                    ],
                ] = [
                    vals["baseline_p_kw"],
                    vals["baseline_q_kvar"],
                    vals["peak_p_kw"],
                    vals["average_p_kw"],
                    vals["voll_usd_per_kwh"],
                    vals["load_type"],
                ]
                nodes.loc[nid, "is_load"] = True
                nodes.loc[nid, "is_critical"] = bool(vals["is_critical"])
        nodes = nodes.reset_index()

        # Connect source to substation and substation to each feeder root. Then connect disconnected components by schematic non-damageable ties.
        extra_edges = [
            {
                "edge_id": f"{prefix}__BACKBONE_TO_SUB",
                "source_branch_id": "synthetic_backbone_to_substation",
                "from_node": source_node,
                "to_node": sub_node,
                "layer": "interface",
                "edge_type": "synthetic_source_connection",
                "component_type": "source_connection",
                "voltage_kv": 69.0,
                "phases": 3,
                "length_km": 0.2,
                "is_switch": False,
                "is_transformer": True,
                "is_overhead": False,
                "is_underground": False,
                "repairable": False,
                "is_hurricane_damage_eligible": False,
                "hurricane_damage_component_type": "not_applicable",
                "repair_time_hours_placeholder": 0.0,
                "substation_id": sub_id,
                "feeder_id": sub_id,
                "region_id": spec["region_id"],
            }
        ]
        for feeder_id, group in b.groupby("feeder_id"):
            clean_feeder = prefix + "__" + str(feeder_id)
            root_bus = group.sort_values("distance_from_substation").iloc[0]["raw_bus_id_clean"]
            extra_edges.append(
                {
                    "edge_id": f"{prefix}__FEEDER_SOURCE__{str(feeder_id).replace('--', '_').replace('-', '_')}",
                    "source_branch_id": "synthetic_feeder_source",
                    "from_node": sub_node,
                    "to_node": prefix + "__" + str(root_bus),
                    "layer": "interface",
                    "edge_type": "synthetic_feeder_head",
                    "component_type": "feeder_head",
                    "voltage_kv": 12.47,
                    "phases": 3,
                    "length_km": 0.05,
                    "is_switch": False,
                    "is_transformer": False,
                    "is_overhead": False,
                    "is_underground": False,
                    "repairable": False,
                    "is_hurricane_damage_eligible": False,
                    "hurricane_damage_component_type": "not_applicable",
                    "repair_time_hours_placeholder": 0.0,
                    "substation_id": sub_id,
                    "feeder_id": clean_feeder,
                    "region_id": spec["region_id"],
                }
            )
        edges = pd.concat([edges, pd.DataFrame(extra_edges)], ignore_index=True)
        g = nx.Graph()
        g.add_nodes_from(nodes["node_id"].astype(str))
        g.add_edges_from(
            edges[["from_node", "to_node"]].astype(str).itertuples(index=False, name=None)
        )
        comp_id = 0
        source_comp = nx.node_connected_component(g, source_node)
        for comp in list(nx.connected_components(g)):
            if comp & source_comp:
                continue
            comp_load = nodes.loc[nodes["node_id"].isin(comp), "baseline_p_kw"].sum()
            if comp_load <= 0:
                continue
            target = next(iter(comp))
            comp_id += 1
            edges = pd.concat(
                [
                    edges,
                    pd.DataFrame(
                        [
                            {
                                "edge_id": f"{prefix}__SCHEMATIC_CONNECTOR_{comp_id}",
                                "source_branch_id": "synthetic_component_connector",
                                "from_node": sub_node,
                                "to_node": target,
                                "layer": "interface",
                                "edge_type": "synthetic_component_connector",
                                "component_type": "connector",
                                "voltage_kv": 12.47,
                                "phases": 3,
                                "length_km": 0.1,
                                "is_switch": False,
                                "is_transformer": False,
                                "is_overhead": False,
                                "is_underground": False,
                                "repairable": False,
                                "is_hurricane_damage_eligible": False,
                                "hurricane_damage_component_type": "not_applicable",
                                "repair_time_hours_placeholder": 0.0,
                                "substation_id": sub_id,
                                "feeder_id": sub_id,
                                "region_id": spec["region_id"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        all_nodes.append(nodes)
        all_edges.append(edges)
        all_loads.append(l[load_cols])
        all_substations.append(
            {
                "substation_id": sub_id,
                "transmission_bus_id": source_node,
                "service_area": spec["region_id"],
                "region_label": spec["region_id"],
                "latitude": sub_lat,
                "longitude": sub_lon,
                "primary_voltage_kv": 69.0,
                "number_of_feeders": int(buses["feeder_id"].nunique()),
                "total_peak_load_mw": float(l["peak_p_kw"].sum() / 1000.0),
            }
        )
        for feeder_id, group in l.groupby("feeder_id"):
            all_feeders.append(
                {
                    "feeder_id": feeder_id,
                    "substation_id": sub_id,
                    "node_count": int(nodes["feeder_id"].eq(feeder_id).sum()),
                    "load_node_count": int(group["node_id"].nunique()),
                    "baseline_p_kw": float(group["baseline_p_kw"].sum()),
                    "region_id": spec["region_id"],
                }
            )
        regions.append(
            {
                **spec,
                "substation_id": sub_id,
                "source_node": source_node,
                "substation_node": sub_node,
            }
        )

    nodes = pd.concat(all_nodes, ignore_index=True).drop_duplicates("node_id", keep="first")
    edges = pd.concat(all_edges, ignore_index=True).drop_duplicates("edge_id", keep="first")
    loads = pd.concat(all_loads, ignore_index=True)
    substations = pd.DataFrame(all_substations)
    feeders = pd.DataFrame(all_feeders)
    regions_df = pd.DataFrame(regions)

    # Add global super source.
    super_source = {col: np.nan for col in nodes.columns}
    super_source.update(
        {
            "node_id": config["geography"]["super_source_node"],
            "source_benchmark_id": "synthetic_system_super_source",
            "layer": "transmission",
            "node_type": "system_super_source",
            "substation_id": "system_source",
            "feeder_id": "system_source",
            "voltage_kv": 230.0,
            "latitude": float(nodes["latitude"].mean()) + 0.05,
            "longitude": float(nodes["longitude"].mean()),
            "x": float(nodes["longitude"].mean()),
            "y": float(nodes["latitude"].mean()) + 0.05,
            "phase": 3,
            "is_load": False,
            "is_generator": True,
            "is_substation": False,
            "is_transmission": True,
            "is_distribution": False,
            "is_data_center": False,
            "is_critical": False,
            "load_type": "source",
            "voll_usd_per_kwh": 0,
            "baseline_p_kw": 0,
            "baseline_q_kvar": 0,
            "peak_p_kw": 0,
            "average_p_kw": 0,
            "priority_class": 0,
            "region_id": "SYSTEM",
            "region_label": "SYSTEM",
        }
    )
    nodes = pd.concat([nodes, pd.DataFrame([super_source])], ignore_index=True)
    backbone_edges = []
    for _, r in regions_df.iterrows():
        backbone_edges.append(
            {
                "edge_id": f"SYS_BACKBONE_{r['id_prefix']}",
                "source_branch_id": "synthetic_inter_region_backbone",
                "from_node": config["geography"]["super_source_node"],
                "to_node": r["source_node"],
                "layer": "transmission",
                "edge_type": "synthetic_transmission_backbone",
                "component_type": "backbone",
                "voltage_kv": 230.0,
                "phases": 3,
                "length_km": 1.0,
                "is_switch": False,
                "is_transformer": False,
                "is_overhead": False,
                "is_underground": False,
                "repairable": False,
                "is_hurricane_damage_eligible": False,
                "hurricane_damage_component_type": "not_applicable",
                "repair_time_hours_placeholder": 0.0,
                "substation_id": r["substation_id"],
                "feeder_id": r["substation_id"],
                "region_id": r["region_id"],
            }
        )
    edges = pd.concat([edges, pd.DataFrame(backbone_edges)], ignore_index=True)

    # Degree.
    g = nx.Graph()
    g.add_nodes_from(nodes["node_id"].astype(str))
    g.add_edges_from(edges[["from_node", "to_node"]].astype(str).itertuples(index=False, name=None))
    nodes["degree"] = nodes["node_id"].map(dict(g.degree())).fillna(0).astype(int)

    dcs, anchor_rows = build_dc_anchor_overlay(config, nodes, loads, regions_df)
    return {
        "nodes": nodes,
        "edges": edges,
        "loads": loads,
        "substations": substations,
        "feeders": feeders,
        "data_centers": dcs,
        "anchor_zones": anchor_rows,
        "regions": regions_df,
    }


def build_dc_anchor_overlay(
    config: Dict[str, Any], nodes: pd.DataFrame, loads: pd.DataFrame, regions: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_load_mw = float(loads["baseline_p_kw"].sum() / 1000.0)
    node_xy = nodes[["node_id", "latitude", "longitude", "region_id"]]
    dc_rows, anchor_rows = [], []
    for i, r in regions.iterrows():
        region_loads = loads.loc[loads["region_id"].eq(r["region_id"])].merge(
            node_xy[["node_id", "latitude", "longitude"]], on="node_id", how="left"
        )
        dc_lat = float(region_loads["latitude"].mean()) + (0.025 if i != 2 else -0.02)
        dc_lon = float(region_loads["longitude"].mean()) + (-0.025 if i == 0 else 0.025)
        region_loads["dc_distance_km"] = region_loads.apply(
            lambda x: _haversine_km(dc_lat, dc_lon, float(x["latitude"]), float(x["longitude"])),
            axis=1,
        )
        region_loads = region_loads.sort_values(
            ["dc_distance_km", "baseline_p_kw"], ascending=[True, False]
        )
        cap_mw = min(total_load_mw * 0.16, region_loads["baseline_p_kw"].sum() / 1000.0 * 0.32)
        chosen = []
        accum = 0.0
        for _, lr in region_loads.iterrows():
            mw = float(lr["baseline_p_kw"]) / 1000.0
            if accum + mw <= cap_mw or len(chosen) < 12:
                chosen.append(lr)
                accum += mw
            if accum >= cap_mw and len(chosen) >= 12:
                break
        chosen_df = pd.DataFrame(chosen)
        dc_id = f"BDC_{i + 1}_{r['id_prefix']}"
        connected_node = r["source_node"]
        anchor_zone_id = f"BENCH_ANCHOR_{r['id_prefix']}"
        load_ids = chosen_df["load_id"].dropna().astype(str).tolist() if not chosen_df.empty else []
        feeder_ids = (
            sorted(chosen_df["feeder_id"].dropna().astype(str).unique().tolist())
            if not chosen_df.empty
            else []
        )
        dc_rows.append(
            {
                "dc_id": dc_id,
                "name": f"Benchmark Synthetic Anchor {i + 1}",
                "latitude": dc_lat,
                "longitude": dc_lon,
                "connected_bus_id": connected_node,
                "connected_node_id": connected_node,
                "connected_node_layer": "transmission",
                "connection_voltage_kv": 69.0,
                "facility_load_mw": 26.0 + 3.0 * i,
                "critical_it_fraction": 0.7,
                "flexible_it_fraction": 0.3,
                "pue": 1.4,
                "critical_load_mw": 13.0 + i,
                "flexible_load_mw": 5.0 + i,
                "cooling_load_mw": 8.0 + i,
                "ups_power_mw": 14.0 + i,
                "ups_energy_mwh": 8.0 + i,
                "backup_gen_mw": 18.0 + i,
                "fuel_hours_at_critical_load": 48,
                "minimum_survival_hours": 24,
                "max_export_mw_anchor_mode": 4.0 + i,
                "anchor_energy_budget_mwh": 14.0 + 4 * i,
                "can_export_in_baseline": False,
                "can_export_in_anchor_mode": True,
                "anchor_zone_id": anchor_zone_id,
                "anchor_zone_load_mw": accum,
                "anchor_zone_load_share": accum / total_load_mw if total_load_mw else 0.0,
                "anchor_zone_load_ids": json.dumps(load_ids),
                "anchor_zone_feeder_ids": json.dumps(feeder_ids),
                "notes": "Synthetic benchmark DC overlay for restoration experiments; not a real facility.",
            }
        )
        anchor_rows.append(
            {
                "anchor_zone_id": anchor_zone_id,
                "dc_id": dc_id,
                "region_id": r["region_id"],
                "anchor_load_count": len(load_ids),
                "anchor_load_mw": accum,
                "anchor_load_share": accum / total_load_mw if total_load_mw else 0.0,
                "anchor_feeder_count": len(feeder_ids),
                "anchor_feeder_ids": json.dumps(feeder_ids),
                "anchor_load_ids": json.dumps(load_ids),
                "max_support_mw": 4.0 + i,
                "energy_budget_mwh": 14.0 + 4 * i,
            }
        )
    return pd.DataFrame(dc_rows), pd.DataFrame(anchor_rows)


def write_conversion_outputs(
    config: Dict[str, Any], d: Dict[str, Path], converted: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    cand_dir = d["case_candidate_converted"] / "SMART_DS_P49U_raw_OpenDSS_composite"
    cand_dir.mkdir(parents=True, exist_ok=True)
    for name in ["nodes", "edges", "loads", "data_centers"]:
        converted[name].to_csv(cand_dir / f"{name}.csv", index=False)
    for name in ["nodes", "edges", "loads", "data_centers", "substations", "feeders"]:
        converted[name].to_csv(d["case_selected_case"] / f"{name}.csv", index=False)
        converted[name].to_csv(d["case_network"] / f"{name}.csv", index=False)
        converted[name].to_csv(d["selected_case"] / f"{name}.csv", index=False)
    converted["anchor_zones"].to_csv(d["case_dc_scenarios"] / "anchor_zones.csv", index=False)
    converted["anchor_zones"].to_csv(d["selected_case"] / "anchor_zones.csv", index=False)
    converted["regions"].to_csv(d["case_network"] / "regions.csv", index=False)
    converted["regions"].to_csv(d["selected_case"] / "regions.csv", index=False)

    nodes, edges, loads, dcs, anchors = (
        converted["nodes"],
        converted["edges"],
        converted["loads"],
        converted["data_centers"],
        converted["anchor_zones"],
    )
    g = nx.Graph()
    g.add_nodes_from(nodes["node_id"].astype(str))
    g.add_edges_from(edges[["from_node", "to_node"]].astype(str).itertuples(index=False, name=None))
    source = config["geography"]["super_source_node"]
    source_comp = nx.node_connected_component(g, source) if source in g else set()
    load_nodes = set(loads["node_id"].astype(str))
    critical = loads.loc[_safe_bool(loads["is_critical"])]
    audit = {
        "candidate_name": "SMART_DS_P49U_raw_OpenDSS_composite",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "load_count": len(loads),
        "critical_load_count": len(critical),
        "substation_count": converted["substations"]["substation_id"].nunique(),
        "feeder_count": converted["feeders"]["feeder_id"].nunique(),
        "dc_anchor_count": len(dcs),
        "base_graph_connected": nx.is_connected(g),
        "orphan_load_count": len(load_nodes - set(g.nodes)),
        "unconnected_base_load_count": len(load_nodes - source_comp),
        "duplicate_node_ids": int(nodes["node_id"].duplicated().sum()),
        "duplicate_edge_ids": int(edges["edge_id"].duplicated().sum()),
        "duplicate_load_ids": int(loads["load_id"].duplicated().sum()),
        "hurricane_damage_eligible_edge_count": int(
            _safe_bool(edges["is_hurricane_damage_eligible"]).sum()
        ),
        "total_load_mw": float(loads["baseline_p_kw"].sum() / 1000.0),
        "max_single_anchor_load_share": float(anchors["anchor_load_share"].max()),
        "combined_anchor_zone_load_share": float(
            anchors["anchor_load_mw"].sum() / (loads["baseline_p_kw"].sum() / 1000.0)
        ),
        "latitude_span_degrees": float(nodes["latitude"].max() - nodes["latitude"].min()),
        "longitude_span_degrees": float(nodes["longitude"].max() - nodes["longitude"].min()),
        "approx_spatial_span_km": float(
            _haversine_km(
                nodes["latitude"].min(),
                nodes["longitude"].min(),
                nodes["latitude"].max(),
                nodes["longitude"].max(),
            )
        ),
    }
    pd.DataFrame([audit]).to_csv(cand_dir / "conversion_audit.csv", index=False)
    pd.DataFrame([audit]).to_csv(d["diagnostics"] / "benchmark_hard_network_audit.csv", index=False)
    anchors.to_csv(d["diagnostics"] / "benchmark_dc_anchor_zone_audit.csv", index=False)

    metadata = {
        "case_name": "final_case_v0_3_benchmark_hard_case",
        "selected_candidate": config["selected_candidate"],
        "benchmark_source": "SMART-DS/OpenDSS P49U local parsed raw feeder data",
        "benchmark_derived_parts": [
            "raw bus topology",
            "raw line topology",
            "raw transformer topology",
            "raw load placement/profile references",
            "feeder IDs",
        ],
        "synthetic_overlays": [
            "three-copy benchmark composition",
            "Houston/Galveston spatial transformation",
            "source/substation connectors",
            "DC anchor overlays",
            "hurricane damage scenarios",
        ],
        "not_real_utility_system": True,
    }
    _write_yaml(metadata, d["case_selected_case"] / "benchmark_source_metadata.yaml")
    _write_yaml(metadata, d["case_network"] / "benchmark_source_metadata.yaml")
    _write_yaml(
        {
            "case_name": "final_case_v0_3_benchmark_hard_case",
            "schema": "restoration node/edge/load/data_center CSV",
            "candidate_k_default": config["restoration_env"]["candidate_k"],
            "network_files": {
                "nodes": "network/nodes.csv",
                "edges": "network/edges.csv",
                "loads": "network/loads.csv",
                "data_centers": "network/data_centers.csv",
            },
        },
        d["case_network"] / "case_config.yaml",
    )
    _write_yaml(metadata, d["case"] / "case_config.yaml")

    features = []
    for _, r in nodes.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(r["longitude"]), float(r["latitude"])],
                },
                "properties": {
                    "node_id": str(r["node_id"]),
                    "region_id": str(r["region_id"]),
                    "node_type": str(r["node_type"]),
                    "is_load": bool(r["is_load"]),
                },
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    (d["case_network"] / "network.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    (d["case_selected_case"] / "network.geojson").write_text(json.dumps(geojson), encoding="utf-8")

    _write_text(
        cand_dir / "conversion_report.md",
        "# SMART-DS P49U Conversion Report\n\n"
        "The selected candidate was converted from local SMART-DS/OpenDSS parsed raw feeder files into the project restoration schema. "
        "The final v0_3 case uses three spatially transformed copies under synthetic substations; this is a benchmark composite, not a real utility system.\n\n"
        + _md_table(pd.DataFrame([audit]))
        + "\n",
    )
    _write_text(
        d["reports"] / "selected_benchmark_case_report.md",
        "# Selected Benchmark Case Report\n\n"
        "Selected system: **SMART_DS_P49U_raw_OpenDSS_composite**.\n\n"
        + _md_table(pd.DataFrame([audit]))
        + "\n\n## Benchmark-Derived vs Synthetic\n\n"
        "Benchmark-derived: raw SMART-DS/OpenDSS bus, line, transformer, load, and feeder topology. "
        "Synthetic overlays: three-copy composition, source connectors, Gulf Coast placement, critical-load labels, DC anchors, and hurricane damage realizations.\n",
    )
    _write_text(
        d["case_network"] / "selected_case_summary.md",
        "# Selected Benchmark Hard Case Summary\n\n" + _md_table(pd.DataFrame([audit])) + "\n",
    )
    _write_text(
        d["case_network"] / "smoke_test_report.md",
        "# Network Smoke Test Report\n\n"
        + (
            "PASS"
            if audit["base_graph_connected"] and audit["unconnected_base_load_count"] == 0
            else "FAIL"
        )
        + "\n\n"
        + _md_table(pd.DataFrame([audit]))
        + "\n",
    )
    return audit


class CachedRestorationSimulator(RestorationSimulator):
    def __init__(
        self, nodes: pd.DataFrame, edges: pd.DataFrame, loads: pd.DataFrame, damage: pd.DataFrame
    ):
        super().__init__(nodes, edges, loads, damage)
        self.source_node = "T_SUPER_SOURCE_V03"
        self._cache: Dict[Tuple[str, ...], Any] = {}

    def metrics(self, damaged_unrepaired: Iterable[str]):  # type: ignore[override]
        key = tuple(sorted(map(str, damaged_unrepaired)))
        if key not in self._cache:
            self._cache[key] = super().metrics(key)
        return self._cache[key]


def sample_damage_edges(
    rng: np.random.Generator,
    eligible: pd.DataFrame,
    storm_id: str,
    intensity: float,
    track_shift_km: float,
    count: int,
) -> List[str]:
    centers = {
        "AL031983": (29.75, -95.36),
        "AL011943": (29.92, -95.09),
        "AL011921": (29.58, -95.70),
        "AL011900": (29.70, -95.02),
        "AL021915": (29.95, -95.45),
        "AL051945": (29.52, -95.18),
    }
    lat, lon = centers.get(storm_id, (29.75, -95.35))
    lon += track_shift_km / 95.0
    dist = eligible.apply(
        lambda r: _haversine_km(lat, lon, float(r["mid_lat"]), float(r["mid_lon"])), axis=1
    ).to_numpy()
    spatial = np.exp(-(dist**2) / (2 * (34.0 + 8.0 * intensity) ** 2))
    comp = eligible["hurricane_damage_component_type"].astype(str)
    vuln = np.where(comp.eq("transformer"), 0.8, np.where(comp.eq("switch"), 0.5, 1.0))
    length = np.clip(
        pd.to_numeric(eligible["length_km"], errors="coerce").fillna(0.1).to_numpy(), 0.03, 3.5
    )
    weights = (0.08 + spatial) * vuln * np.sqrt(length) * intensity
    weights = np.maximum(weights, 1e-8)
    weights /= weights.sum()
    return list(
        rng.choice(
            eligible["edge_id"].astype(str).to_numpy(),
            size=min(count, len(eligible)),
            replace=False,
            p=weights,
        )
    )


def generate_benchmark_scenarios(config: Dict[str, Any], d: Dict[str, Path]) -> Dict[str, Any]:
    rng = np.random.default_rng(int(config["random_seed"]) + 19)
    nodes = pd.read_csv(d["case_network"] / "nodes.csv")
    edges = pd.read_csv(d["case_network"] / "edges.csv")
    loads = pd.read_csv(d["case_network"] / "loads.csv")
    anchors = pd.read_csv(d["case_dc_scenarios"] / "anchor_zones.csv")
    simulator = CachedRestorationSimulator(nodes, edges, loads, pd.DataFrame())
    pos = nodes.set_index("node_id")[["latitude", "longitude"]]
    eligible = edges.loc[
        _safe_bool(edges["is_hurricane_damage_eligible"]) & _safe_bool(edges["repairable"])
    ].copy()
    eligible = eligible.merge(pos, left_on="from_node", right_index=True, how="left").rename(
        columns={"latitude": "from_lat", "longitude": "from_lon"}
    )
    eligible = eligible.merge(pos, left_on="to_node", right_index=True, how="left").rename(
        columns={"latitude": "to_lat", "longitude": "to_lon"}
    )
    eligible["mid_lat"] = (eligible["from_lat"] + eligible["to_lat"]) / 2.0
    eligible["mid_lon"] = (eligible["from_lon"] + eligible["to_lon"]) / 2.0
    eligible = eligible.dropna(subset=["mid_lat", "mid_lon"])
    anchor_map = {}
    for _, a in anchors.iterrows():
        try:
            anchor_map[a["anchor_zone_id"]] = json.loads(a["anchor_load_ids"])
        except Exception:
            anchor_map[a["anchor_zone_id"]] = []
    rows, dmg_rows = [], []
    idx = 0
    for storm in config["scenarios"]["source_storms"]:
        for variant in config["scenarios"]["variants"]:
            for mc in range(1, int(config["scenarios"]["monte_carlo_per_variant"]) + 1):
                idx += 1
                intensity = float(variant["intensity_scale"])
                target = int(
                    np.clip(
                        rng.poisson(float(storm["base_damage_mean"]) * intensity**1.7)
                        + rng.integers(-2, 5),
                        5,
                        42,
                    )
                )
                if variant["augmentation_type"] != "historical_observed":
                    target = max(target, int(12 + 8 * intensity))
                damaged = sample_damage_edges(
                    rng,
                    eligible,
                    storm["source_storm_id"],
                    intensity,
                    float(variant["track_shift_km"]),
                    target,
                )
                metrics = simulator.metrics(damaged)
                attempts = 0
                while metrics.unserved_mw < 5.0 and attempts < 3:
                    target = min(len(eligible), target + 6)
                    damaged = sample_damage_edges(
                        rng,
                        eligible,
                        storm["source_storm_id"],
                        intensity + attempts * 0.05,
                        float(variant["track_shift_km"]),
                        target,
                    )
                    metrics = simulator.metrics(damaged)
                    attempts += 1
                skey = f"BV03_{idx:04d}_{storm['source_storm_id']}_{variant['suffix']}_MC{mc}"
                split = next(
                    (
                        role
                        for role, storms in config["scenarios"]["split_by_source_storm"].items()
                        if storm["source_storm_id"] in storms
                    ),
                    "train",
                )
                unserved_set = set(metrics.unserved_load_ids)
                anchor_unserved = 0.0
                for load_ids in anchor_map.values():
                    overlap = loads.loc[loads["load_id"].isin(unserved_set.intersection(load_ids))]
                    anchor_unserved += float(overlap["baseline_p_kw"].sum() / 1000.0)
                for j, eid in enumerate(damaged):
                    attrs = simulator.edge_attrs.get(eid, {})
                    comp = str(attrs.get("hurricane_damage_component_type", "overhead_line"))
                    base_rt = float(attrs.get("repair_time_hours_placeholder", 8.0) or 8.0)
                    rt = float(np.clip(rng.lognormal(math.log(max(base_rt, 3.0)), 0.35), 2.0, 48.0))
                    dmg_rows.append(
                        {
                            "scenario_key": skey,
                            "edge_id": eid,
                            "component_type": comp,
                            "repair_time_hours": rt,
                            "damage_order": j,
                        }
                    )
                rows.append(
                    {
                        "scenario_key": skey,
                        "scenario_id": f"BV03_{idx:04d}_{storm['source_storm_id']}_{variant['suffix']}",
                        "source_storm_id": storm["source_storm_id"],
                        "storm_name": storm["storm_name"],
                        "year": storm["year"],
                        "augmentation_type": variant["augmentation_type"],
                        "scenario_class": variant["scenario_class"],
                        "historical_or_augmented_flag": "historical"
                        if variant["augmentation_type"] == "historical_observed"
                        else "augmented",
                        "intensity_scaled_flag": variant["augmentation_type"] == "intensity_scaled",
                        "intensity_scale": intensity,
                        "track_shift_km": variant["track_shift_km"],
                        "monte_carlo_id": mc,
                        "random_seed": int(config["random_seed"]) + idx,
                        "split_role": split,
                        "initial_unserved_mw": metrics.unserved_mw,
                        "initial_critical_unserved_mw": metrics.critical_unserved_mw,
                        "initial_voll_cost_dollars_per_hour": metrics.voll_cost,
                        "initial_anchor_zone_unserved_mw": anchor_unserved,
                        "damaged_component_count": len(damaged),
                        "damaged_repairable_component_count": len(damaged),
                        "damaged_overhead_line_count": sum(
                            1
                            for e in damaged
                            if simulator.edge_component_type(e) == "overhead_line"
                        ),
                        "outage_island_count": metrics.island_count,
                        "expected_episode_length_repairs": len(damaged),
                        "anchor_support_feasible_t0": anchor_unserved
                        > config["scenarios"]["anchor_unserved_mw_threshold"],
                        "damaged_edge_ids_json": json.dumps(damaged),
                        "notes": "Benchmark-derived topology; historical rows use historical source storm labels, while intensity/track rows are stress/augmented and not historical observations.",
                    }
                )
    manifest = pd.DataFrame(rows)
    damage = pd.DataFrame(dmg_rows)
    manifest["benchmark_hard_nonzero"] = manifest["initial_unserved_mw"] > 0.05
    manifest["benchmark_hard_severe"] = (
        manifest["damaged_repairable_component_count"]
        >= config["scenarios"]["severe_damage_threshold"]
    ) & (manifest["initial_unserved_mw"] >= config["scenarios"]["severe_unserved_mw_threshold"])
    manifest["benchmark_hard_critical"] = (
        manifest["initial_critical_unserved_mw"]
        >= config["scenarios"]["critical_unserved_mw_threshold"]
    )
    manifest["benchmark_hard_anchor_relevant"] = (
        manifest["initial_anchor_zone_unserved_mw"]
        >= config["scenarios"]["anchor_unserved_mw_threshold"]
    )
    manifest["benchmark_hard_long_sequence"] = (
        manifest["damaged_repairable_component_count"]
        >= config["scenarios"]["destructive_damage_threshold"]
    )
    manifest["benchmark_hard_multi_island"] = (
        manifest["outage_island_count"] >= config["scenarios"]["multi_island_threshold"]
    )
    manifest["benchmark_hard_historical_nonzero"] = manifest["benchmark_hard_nonzero"] & manifest[
        "historical_or_augmented_flag"
    ].eq("historical")
    manifest["benchmark_hard_stress_intensity_scaled"] = manifest["augmentation_type"].isin(
        ["intensity_scaled", "augmented_track"]
    )
    manifest["benchmark_hard_train"] = manifest["split_role"].eq("train")
    manifest["benchmark_hard_validation"] = manifest["split_role"].eq("validation")
    manifest["benchmark_hard_test"] = manifest["split_role"].eq("test")
    manifest["benchmark_hard_destructive_15plus"] = (
        manifest["damaged_repairable_component_count"]
        >= config["scenarios"]["destructive_damage_threshold"]
    )

    for folder in [d["case_scenario_design"], d["case_hurricane_scenarios"], d["scenario_design"]]:
        manifest.to_csv(folder / "benchmark_hard_hurricane_scenario_manifest.csv", index=False)
        damage.to_csv(folder / "benchmark_hard_damage_components.csv", index=False)
    subset_cols = [c for c in manifest.columns if c.startswith("benchmark_hard_")]
    subset_manifest = manifest[
        [
            "scenario_key",
            "source_storm_id",
            "storm_name",
            "year",
            "augmentation_type",
            "split_role",
            "initial_unserved_mw",
            "initial_critical_unserved_mw",
            "initial_voll_cost_dollars_per_hour",
            "damaged_repairable_component_count",
            "outage_island_count",
        ]
        + subset_cols
    ]
    subset_manifest.to_csv(
        d["scenario_design"] / "benchmark_hard_scenario_subset_manifest.csv", index=False
    )
    subset_manifest.to_csv(
        d["case_scenario_design"] / "benchmark_hard_scenario_subset_manifest.csv", index=False
    )

    summary = []
    for c in subset_cols:
        p = manifest.loc[manifest[c]]
        if p.empty:
            continue
        summary.append(
            {
                "subset": c,
                "scenario_count": len(p),
                "source_storm_count": p["source_storm_id"].nunique(),
                "historical_count": int(p["historical_or_augmented_flag"].eq("historical").sum()),
                "stress_or_augmented_count": int(
                    ~p["historical_or_augmented_flag"].eq("historical").sum()
                )
                if False
                else int((p["historical_or_augmented_flag"] != "historical").sum()),
                "mean_damaged_repairable_components": p[
                    "damaged_repairable_component_count"
                ].mean(),
                "max_damaged_repairable_components": p["damaged_repairable_component_count"].max(),
                "mean_outage_islands": p["outage_island_count"].mean(),
                "mean_initial_unserved_mw": p["initial_unserved_mw"].mean(),
                "mean_initial_critical_unserved_mw": p["initial_critical_unserved_mw"].mean(),
                "mean_initial_voll_cost_dollars_per_hour": p[
                    "initial_voll_cost_dollars_per_hour"
                ].mean(),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(d["scenario_design"] / "benchmark_hard_scenario_summary.csv", index=False)
    summary_df.to_csv(d["diagnostics"] / "benchmark_hard_scenario_audit.csv", index=False)
    _write_text(
        d["reports"] / "benchmark_hard_hurricane_scenario_design_report.md",
        "# Benchmark Hard Hurricane Scenario Design Report\n\n"
        "The benchmark-derived v0_3 scenario library uses the same stress design principles as v0_2. Stress/intensity-scaled and track-shifted scenarios are explicitly labelled and are not historical observations.\n\n"
        + _md_table(summary_df)
        + "\n",
    )
    return {
        "scenario_count": len(manifest),
        "severe_count": int(manifest["benchmark_hard_severe"].sum()),
        "long_sequence_count": int(manifest["benchmark_hard_long_sequence"].sum()),
        "mean_severe_damage_count": float(
            manifest.loc[
                manifest["benchmark_hard_severe"], "damaged_repairable_component_count"
            ].mean()
        ),
        "max_damage_count": int(manifest["damaged_repairable_component_count"].max()),
        "mean_outage_islands": float(manifest["outage_island_count"].mean()),
        "mean_initial_voll_cost": float(manifest["initial_voll_cost_dollars_per_hour"].mean()),
    }

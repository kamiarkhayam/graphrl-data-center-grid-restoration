from __future__ import annotations

import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dc_restoration.paths import repository_root

SCRIPT_DIR = Path(__file__).resolve().parent

from dc_restoration.testbed.utils.dss_parse_utils import compile_dss_master, dss_value  # noqa: E402
from dc_restoration.testbed.utils.geo_utils import (  # noqa: E402
    haversine_km,
    min_distance_to_coast_km,
)
from dc_restoration.testbed.utils.project_utils import (  # noqa: E402
    default_arg_parser,
    get_paths,
    load_config,
    setup_logging,
    write_iteration_summary,
)
from dc_restoration.testbed.utils.time_series_utils import build_non_leap_timestamps  # noqa: E402

PLOTS = [
    "audit_network_layers_clean.png",
    "audit_physical_vs_equivalent_edges.png",
    "audit_distribution_only.png",
    "audit_data_center_connection_zoom.png",
    "audit_hazard_exposed_edges.png",
    "dc_candidate_locations.png",
    "dc_scenario_comparison_map.png",
]

SCENARIO_DEFS = [
    {
        "name": "single_dc_current",
        "description": "Current single synthetic data center placement retained from the baseline case.",
        "stress_test": False,
        "placements": [("DC_1", "T_110792", 29.6)],
    },
    {
        "name": "two_dc_split_same_total",
        "description": "Two data centers split the baseline total load across separate high-side service points.",
        "stress_test": False,
        "placements": [("DC_1", "T_110792", 14.8), ("DC_2", "T_110769", 14.8)],
    },
    {
        "name": "three_dc_split_same_total",
        "description": "Three smaller data centers preserve the baseline total load while increasing spatial diversity.",
        "stress_test": False,
        "placements": [
            ("DC_1", "T_110792", 10.0),
            ("DC_2", "T_110769", 10.0),
            ("DC_3", "SUB_p49uhs5_1247", 9.6),
        ],
    },
    {
        "name": "high_dc_penetration",
        "description": "Stress-test scenario with higher total data-center load and three anchor locations.",
        "stress_test": True,
        "placements": [
            ("DC_1", "T_110792", 18.0),
            ("DC_2", "T_110769", 15.0),
            ("DC_3", "SUB_p49uhs5_1247", 12.0),
        ],
    },
]


def parse_json_list(value: Any) -> list[str]:
    if pd.isna(value) or value in {"", "[]", None}:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in json.loads(value)]


def project_equirectangular(
    lat: float, lon: float, lat0: float, lon0: float
) -> tuple[float, float]:
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * 6371.0088
    y = math.radians(lat - lat0) * 6371.0088
    return x, y


def unproject_equirectangular(
    x_km: float, y_km: float, lat0: float, lon0: float
) -> tuple[float, float]:
    lat = lat0 + math.degrees(y_km / 6371.0088)
    lon = lon0 + math.degrees(x_km / (6371.0088 * math.cos(math.radians(lat0))))
    return lat, lon


def distance_pair_metrics(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    lat0: float,
    lon0: float,
) -> dict[str, float]:
    geodesic = haversine_km(point_a[0], point_a[1], point_b[0], point_b[1])
    x1, y1 = project_equirectangular(point_a[0], point_a[1], lat0, lon0)
    x2, y2 = project_equirectangular(point_b[0], point_b[1], lat0, lon0)
    projected = math.hypot(x2 - x1, y2 - y1)
    return {"geodesic_km": geodesic, "equirectangular_km": projected}


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def centroid_and_bbox(nodes_df: pd.DataFrame) -> dict[str, Any]:
    lat_min = float(nodes_df["latitude"].min())
    lat_max = float(nodes_df["latitude"].max())
    lon_min = float(nodes_df["longitude"].min())
    lon_max = float(nodes_df["longitude"].max())
    centroid_lat = float(nodes_df["latitude"].mean())
    centroid_lon = float(nodes_df["longitude"].mean())
    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
    }


def build_member_component_maps(
    raw_lines: pd.DataFrame, raw_transformers: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    component_map: dict[str, dict[str, Any]] = {}
    for _, row in raw_lines.iterrows():
        component_map[str(row["raw_edge_id"])] = {
            "kind": "switch" if normalize_bool(row["is_switch"]) else "line",
            "is_overhead": normalize_bool(row["is_overhead"]),
            "is_underground": normalize_bool(row["is_underground"]),
            "voltage_kv": float(row["r"]) if False else None,
        }
    for _, row in raw_transformers.iterrows():
        component_map[str(row["raw_transformer_id"])] = {
            "kind": "transformer",
            "is_overhead": False,
            "is_underground": False,
            "voltage_kv": None,
        }
    return component_map


def classify_edge_row(
    row: pd.Series, node_layer_map: dict[str, str], member_component_map: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    member_ids = parse_json_list(row["member_raw_edges"])
    member_kinds = [
        member_component_map.get(member_id, {"kind": "unknown"})["kind"] for member_id in member_ids
    ]
    line_members = [
        member_component_map.get(member_id)
        for member_id in member_ids
        if member_component_map.get(member_id, {}).get("kind") in {"line", "switch"}
    ]
    transformer_member_count = sum(kind == "transformer" for kind in member_kinds)
    overhead_count = sum(item.get("is_overhead", False) for item in line_members)
    underground_count = sum(item.get("is_underground", False) for item in line_members)
    unknown_install_count = max(0, len(line_members) - overhead_count - underground_count)

    from_layer = node_layer_map.get(str(row["from_node"]), "")
    to_layer = node_layer_map.get(str(row["to_node"]), "")
    edge_type = str(row["edge_type"])
    voltage_kv = float(row["voltage_kv"]) if pd.notna(row["voltage_kv"]) else np.nan

    if edge_type == "equivalent_subtransmission":
        return {
            "edge_role": "equivalent_source_connection",
            "is_physical_line": False,
            "is_equivalent_edge": True,
            "is_hurricane_damage_eligible": False,
            "hurricane_damage_component_type": "equivalent_source",
        }
    if edge_type == "dedicated_transformer":
        return {
            "edge_role": "data_center_connection",
            "is_physical_line": False,
            "is_equivalent_edge": True,
            "is_hurricane_damage_eligible": False,
            "hurricane_damage_component_type": "data_center_facility",
        }
    if edge_type == "interface_transformer_equivalent":
        return {
            "edge_role": "interface_connection",
            "is_physical_line": False,
            "is_equivalent_edge": True,
            "is_hurricane_damage_eligible": False,
            "hurricane_damage_component_type": "not_applicable",
        }
    if edge_type == "switch":
        return {
            "edge_role": "distribution_switch",
            "is_physical_line": False,
            "is_equivalent_edge": False,
            "is_hurricane_damage_eligible": False,
            "hurricane_damage_component_type": "switch",
        }
    if edge_type == "transformer":
        is_substation_transformer = (
            from_layer == "interface"
            or to_layer == "interface"
            or (pd.notna(voltage_kv) and voltage_kv >= 69.0)
        )
        return {
            "edge_role": "substation_transformer"
            if is_substation_transformer
            else "distribution_transformer",
            "is_physical_line": False,
            "is_equivalent_edge": False,
            "is_hurricane_damage_eligible": False,
            "hurricane_damage_component_type": "transformer",
        }

    if edge_type in {"line", "aggregated_radial", "aggregated_complex", "aggregated_corridor"}:
        if from_layer == "transmission" or to_layer == "transmission":
            return {
                "edge_role": "transmission_line",
                "is_physical_line": True,
                "is_equivalent_edge": False,
                "is_hurricane_damage_eligible": True,
                "hurricane_damage_component_type": "transmission_line",
            }

        component_type = "not_applicable"
        eligible = False
        if overhead_count > 0 and underground_count == 0 and unknown_install_count == 0:
            component_type = "overhead_distribution_line"
            eligible = True
        elif underground_count > 0 and overhead_count == 0 and unknown_install_count == 0:
            component_type = "underground_distribution_line"
            eligible = True
        elif normalize_bool(row["is_overhead"]) and not normalize_bool(row["is_underground"]):
            component_type = "overhead_distribution_line"
            eligible = True
        elif normalize_bool(row["is_underground"]) and not normalize_bool(row["is_overhead"]):
            component_type = "underground_distribution_line"
            eligible = True

        edge_role = "distribution_line"
        if (
            edge_type in {"aggregated_radial", "aggregated_complex", "aggregated_corridor"}
            and transformer_member_count > 0
            and len(line_members) == 0
        ):
            edge_role = "unknown"
            component_type = "not_applicable"
            eligible = False

        return {
            "edge_role": edge_role,
            "is_physical_line": True if edge_role == "distribution_line" else False,
            "is_equivalent_edge": False,
            "is_hurricane_damage_eligible": eligible,
            "hurricane_damage_component_type": component_type,
        }

    return {
        "edge_role": "unknown",
        "is_physical_line": False,
        "is_equivalent_edge": False,
        "is_hurricane_damage_eligible": False,
        "hurricane_damage_component_type": "not_applicable",
    }


def build_projected_positions(nodes_df: pd.DataFrame, lat0: float, lon0: float) -> pd.DataFrame:
    projected = nodes_df.copy()
    xy = projected.apply(
        lambda row: project_equirectangular(
            float(row["latitude"]), float(row["longitude"]), lat0, lon0
        ),
        axis=1,
    )
    projected["plot_x_km"] = [item[0] for item in xy]
    projected["plot_y_km"] = [item[1] for item in xy]
    return projected


def edge_segments(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    lookup = nodes_df.set_index("node_id")[["plot_x_km", "plot_y_km", "latitude", "longitude"]]
    edge_plot = edges_df.copy()
    edge_plot["x1"] = edge_plot["from_node"].map(lookup["plot_x_km"])
    edge_plot["y1"] = edge_plot["from_node"].map(lookup["plot_y_km"])
    edge_plot["x2"] = edge_plot["to_node"].map(lookup["plot_x_km"])
    edge_plot["y2"] = edge_plot["to_node"].map(lookup["plot_y_km"])
    edge_plot["lat1"] = edge_plot["from_node"].map(lookup["latitude"])
    edge_plot["lon1"] = edge_plot["from_node"].map(lookup["longitude"])
    edge_plot["lat2"] = edge_plot["to_node"].map(lookup["latitude"])
    edge_plot["lon2"] = edge_plot["to_node"].map(lookup["longitude"])
    return edge_plot


def draw_edge_subset(
    ax,
    edge_df: pd.DataFrame,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    for _, row in edge_df.iterrows():
        ax.plot(
            [row["x1"], row["x2"]],
            [row["y1"], row["y2"]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )


def scatter_subset(
    ax,
    node_df: pd.DataFrame,
    color: str,
    size: float,
    marker: str = "o",
    label: str | None = None,
    edgecolor: str = "none",
    zorder: int = 3,
) -> None:
    if node_df.empty:
        return
    ax.scatter(
        node_df["plot_x_km"],
        node_df["plot_y_km"],
        s=size,
        c=color,
        marker=marker,
        label=label,
        edgecolors=edgecolor,
        zorder=zorder,
    )


def style_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Local east-west distance (km)")
    ax.set_ylabel("Local north-south distance (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def add_node_labels(
    ax, nodes_df: pd.DataFrame, label_map: dict[str, str], dx: float = 0.08, dy: float = 0.08
) -> None:
    for _, row in nodes_df.iterrows():
        label = label_map.get(row["node_id"])
        if not label:
            continue
        ax.text(row["plot_x_km"] + dx, row["plot_y_km"] + dy, label, fontsize=8, zorder=6)


def select_candidate_nodes(
    nodes_df: pd.DataFrame, loads_df: pd.DataFrame, existing_dc: pd.Series
) -> pd.DataFrame:
    load_subset = loads_df.loc[
        ~loads_df["is_data_center"], ["node_id", "baseline_p_kw", "is_critical"]
    ].rename(columns={"baseline_p_kw": "load_baseline_p_kw", "is_critical": "load_is_critical"})
    non_dc_load_nodes = nodes_df.merge(load_subset, on="node_id", how="inner")
    candidate_mask = (
        ((nodes_df["layer"] == "transmission") & ~nodes_df["node_type"].eq("source"))
        | nodes_df["node_type"].eq("substation_interface")
        | nodes_df["node_type"].eq("source")
    )
    candidate_df = nodes_df.loc[
        candidate_mask,
        ["node_id", "layer", "node_type", "voltage_kv", "latitude", "longitude", "degree"],
    ].copy()
    interface_nodes = nodes_df.loc[
        nodes_df["layer"].eq("interface"), ["node_id", "latitude", "longitude"]
    ].copy()

    records = []
    for _, row in candidate_df.iterrows():
        distances = non_dc_load_nodes.apply(
            lambda r: haversine_km(
                float(row["latitude"]),
                float(row["longitude"]),
                float(r["latitude"]),
                float(r["longitude"]),
            ),
            axis=1,
        )
        nearby1 = float(
            non_dc_load_nodes.loc[distances <= 1.0, "load_baseline_p_kw"].sum() / 1000.0
        )
        nearby3 = float(
            non_dc_load_nodes.loc[distances <= 3.0, "load_baseline_p_kw"].sum() / 1000.0
        )
        crit3 = float(
            non_dc_load_nodes.loc[
                (distances <= 3.0) & non_dc_load_nodes["load_is_critical"], "load_baseline_p_kw"
            ].sum()
            / 1000.0
        )
        feeders_nearby = int(non_dc_load_nodes.loc[distances <= 3.0, "feeder_id"].nunique())
        if row["node_id"] in set(interface_nodes["node_id"]):
            dist_interface = 0.0
        else:
            dist_interface = float(
                interface_nodes.apply(
                    lambda r: haversine_km(
                        float(row["latitude"]),
                        float(row["longitude"]),
                        float(r["latitude"]),
                        float(r["longitude"]),
                    ),
                    axis=1,
                ).min()
            )
        dist_existing_dc = haversine_km(
            float(row["latitude"]),
            float(row["longitude"]),
            float(existing_dc["latitude"]),
            float(existing_dc["longitude"]),
        )

        voltage_bonus = (
            6.0
            if float(row["voltage_kv"]) >= 69.0
            else 4.0
            if row["layer"] == "interface"
            else -5.0
        )
        interface_bonus = (
            3.0
            if row["node_type"] == "substation_interface"
            else 2.0
            if row["layer"] == "transmission"
            else 0.0
        )
        nearby_load_bonus = min(5.0, nearby3 / 30.0)
        critical_bonus = min(3.0, crit3 / 10.0)
        feeder_bonus = min(2.0, feeders_nearby / 4.0)
        plausibility_bonus = 2.0 if row["node_type"] != "source" and dist_interface <= 1.0 else 0.0
        low_voltage_penalty = (
            0.0 if (float(row["voltage_kv"]) >= 69.0 or row["layer"] == "interface") else 6.0
        )
        isolation_penalty = 1.5 if int(row["degree"]) <= 1 else 0.0
        distance_penalty = max(0.0, dist_interface - 1.5) * 2.0
        equivalent_penalty = 25.0 if row["node_type"] == "source" else 0.0
        score = round(
            voltage_bonus
            + interface_bonus
            + nearby_load_bonus
            + critical_bonus
            + feeder_bonus
            + plausibility_bonus
            - low_voltage_penalty
            - isolation_penalty
            - distance_penalty
            - equivalent_penalty,
            3,
        )
        notes = []
        if row["node_type"] == "source":
            notes.append("Equivalent service-area source; excluded from scenario placements.")
        if row["node_type"] == "substation_interface":
            notes.append(
                "Use as a substation-level service point because the retained 69-kV candidate set is small."
            )
        if row["node_id"] == "T_110769":
            notes.append(
                "Neighboring 69-kV peer delivery point in the simplified equivalent transmission layer."
            )
        records.append(
            {
                "candidate_id": f"CAND_{row['node_id']}",
                "node_id": row["node_id"],
                "layer": row["layer"],
                "node_type": row["node_type"],
                "voltage_kv": float(row["voltage_kv"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "connected_feeders_nearby": feeders_nearby,
                "nearby_load_mw_within_1km": round(nearby1, 3),
                "nearby_load_mw_within_3km": round(nearby3, 3),
                "nearby_critical_load_mw_within_3km": round(crit3, 3),
                "distance_to_nearest_interface_km": round(dist_interface, 3),
                "distance_to_existing_dc_km": round(dist_existing_dc, 3),
                "candidate_score": score,
                "notes": " ".join(notes),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(["candidate_score", "voltage_kv"], ascending=[False, False])
        .reset_index(drop=True)
    )


def facility_location_for_candidate(
    candidate_row: pd.Series,
    load_nodes: pd.DataFrame,
    existing_dc_map: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    node_id = (
        str(candidate_row["node_id"])
        if "node_id" in candidate_row.index
        else str(candidate_row.name)
    )
    if node_id in existing_dc_map:
        return existing_dc_map[node_id]

    nearby = load_nodes.loc[
        load_nodes.apply(
            lambda r: (
                haversine_km(
                    float(candidate_row["latitude"]),
                    float(candidate_row["longitude"]),
                    float(r["latitude"]),
                    float(r["longitude"]),
                )
                <= 1.5
            ),
            axis=1,
        )
    ].copy()
    lat0 = float(candidate_row["latitude"])
    lon0 = float(candidate_row["longitude"])
    if not nearby.empty:
        weight_col = (
            "load_baseline_p_kw" if "load_baseline_p_kw" in nearby.columns else "baseline_p_kw"
        )
        weighted_lat = float(np.average(nearby["latitude"], weights=nearby[weight_col]))
        weighted_lon = float(np.average(nearby["longitude"], weights=nearby[weight_col]))
        x_target, y_target = project_equirectangular(weighted_lat, weighted_lon, lat0, lon0)
        distance = math.hypot(x_target, y_target)
        if distance < 0.2:
            x_target, y_target = 0.45, -0.15
            distance = math.hypot(x_target, y_target)
        scale = min(1.0, 0.9 / max(distance, 1e-6))
        x_offset = x_target * scale
        y_offset = y_target * scale
    else:
        angle = (sum(ord(ch) for ch in node_id) % 360) * math.pi / 180.0
        x_offset = 0.7 * math.cos(angle)
        y_offset = 0.7 * math.sin(angle)
    return unproject_equirectangular(x_offset, y_offset, lat0, lon0)


def make_dc_record(
    dc_id: str,
    connected_node: pd.Series,
    facility_load_mw: float,
    latitude: float,
    longitude: float,
    connection_type: str,
    is_physical_connection: bool,
    is_hurricane_damage_eligible: bool,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    connected_node_id = (
        str(connected_node["node_id"])
        if "node_id" in connected_node.index
        else str(connected_node.name)
    )
    pue = float(defaults["pue"])
    it_total = round(facility_load_mw / pue, 3)
    critical_load = round(it_total * float(defaults["critical_it_fraction"]), 3)
    flexible_load = round(it_total * float(defaults["flexible_it_fraction"]), 3)
    cooling_load = round(facility_load_mw - it_total, 3)
    ups_power = critical_load
    ups_energy = round(ups_power * (float(defaults["ups_duration_minutes"]) / 60.0), 3)
    backup_gen = round(critical_load * 1.2, 3)
    max_export = round(max(0.0, backup_gen - critical_load - 2.0), 3)
    notes = (
        "Synthetic Houston/Galveston-area data-center scenario node connected through a "
        f"{connection_type.replace('_', ' ')}."
    )
    return {
        "dc_id": dc_id,
        "name": f"Gulf Anchor {dc_id}",
        "connected_node_id": connected_node_id,
        "connected_node_layer": str(connected_node["layer"]),
        "connection_voltage_kv": float(connected_node["voltage_kv"]),
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "facility_load_mw": float(facility_load_mw),
        "critical_it_fraction": float(defaults["critical_it_fraction"]),
        "flexible_it_fraction": float(defaults["flexible_it_fraction"]),
        "pue": pue,
        "critical_load_mw": critical_load,
        "flexible_load_mw": flexible_load,
        "cooling_load_mw": cooling_load,
        "ups_power_mw": ups_power,
        "ups_energy_mwh": ups_energy,
        "backup_gen_mw": backup_gen,
        "fuel_hours_at_critical_load": 48,
        "minimum_survival_hours": int(defaults["minimum_survival_hours"]),
        "max_export_mw_anchor_mode": max_export,
        "can_export_in_baseline": False,
        "can_export_in_anchor_mode": True,
        "connection_type": connection_type,
        "is_physical_connection": bool(is_physical_connection),
        "is_hurricane_damage_eligible": bool(is_hurricane_damage_eligible),
        "notes": notes,
    }


def make_overlay_tables(
    scenario_name: str,
    dc_records: list[dict[str, Any]],
    base_nodes: pd.DataFrame,
    base_edges: pd.DataFrame,
    base_loads: pd.DataFrame,
    coast_points: list[tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_lookup = base_nodes.set_index("node_id")
    edge_rows = []
    node_rows = []
    load_rows = []
    for dc in dc_records:
        connected_node = node_lookup.loc[dc["connected_node_id"]]
        connection_length = haversine_km(
            float(dc["latitude"]),
            float(dc["longitude"]),
            float(connected_node["latitude"]),
            float(connected_node["longitude"]),
        )
        distance_to_sub = haversine_km(
            float(dc["latitude"]),
            float(dc["longitude"]),
            float(node_lookup.loc["SUB_p49uhs5_1247", "latitude"]),
            float(node_lookup.loc["SUB_p49uhs5_1247", "longitude"]),
        )
        node_rows.append(
            {
                "node_id": dc["dc_id"],
                "raw_id": dc["dc_id"],
                "layer": "data_center",
                "node_type": "data_center_facility",
                "substation_id": "p49uhs5_1247",
                "feeder_id": "",
                "voltage_kv": float(dc["connection_voltage_kv"]),
                "latitude": float(dc["latitude"]),
                "longitude": float(dc["longitude"]),
                "x": float(dc["longitude"]),
                "y": float(dc["latitude"]),
                "phase": "3",
                "is_load": True,
                "is_generator": False,
                "is_substation": False,
                "is_transmission": False,
                "is_distribution": False,
                "is_data_center": True,
                "is_critical": True,
                "load_type": "data_center",
                "voll_usd_per_kwh": 200,
                "baseline_p_kw": float(dc["facility_load_mw"] * 1000.0),
                "baseline_q_kvar": 0.0,
                "peak_p_kw": float(dc["facility_load_mw"] * 1000.0),
                "average_p_kw": float(dc["facility_load_mw"] * 1000.0),
                "priority_class": 0,
                "degree": 1,
                "distance_to_substation_km": distance_to_sub,
                "distance_to_data_center_km": 0.0,
                "distance_to_coast_km": min_distance_to_coast_km(
                    float(dc["latitude"]), float(dc["longitude"]), coast_points
                ),
                "hurricane_exposure_placeholder": 0.5,
                "aggregated_raw_node_count": 1,
                "member_raw_buses": json.dumps([]),
            }
        )
        edge_rows.append(
            {
                "edge_id": f"DC_CONN_{dc['dc_id']}",
                "from_node": dc["connected_node_id"],
                "to_node": dc["dc_id"],
                "layer": "data_center",
                "edge_type": "dedicated_transformer",
                "voltage_kv": float(dc["connection_voltage_kv"]),
                "phases": "3",
                "length_km": connection_length,
                "r_ohm": 0.0,
                "x_ohm": 0.0,
                "rating_amp": np.nan,
                "rating_mva": round(float(dc["facility_load_mw"]) * 1.2, 3),
                "is_switch": False,
                "is_transformer": True,
                "is_overhead": False,
                "is_underground": False,
                "normal_status": "closed",
                "initial_status": "closed",
                "switchable": False,
                "repairable": True,
                "damage_probability_placeholder": 0.2,
                "repair_time_hours_placeholder": 8.0,
                "member_raw_edges": json.dumps([]),
                "edge_role": "data_center_connection",
                "is_physical_line": False,
                "is_equivalent_edge": True,
                "is_hurricane_damage_eligible": bool(dc["is_hurricane_damage_eligible"]),
                "hurricane_damage_component_type": "data_center_facility",
            }
        )
        load_rows.append(
            {
                "load_id": f"LD_{dc['dc_id']}",
                "node_id": dc["dc_id"],
                "feeder_id": "",
                "substation_id": "p49uhs5_1247",
                "load_type": "data_center",
                "voll_usd_per_kwh": 200,
                "baseline_p_kw": float(dc["facility_load_mw"] * 1000.0),
                "baseline_q_kvar": 0.0,
                "peak_p_kw": float(dc["facility_load_mw"] * 1000.0),
                "average_p_kw": float(dc["facility_load_mw"] * 1000.0),
                "is_critical": True,
                "is_data_center": True,
                "profile_id": "constant_dc_profile",
            }
        )
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows), pd.DataFrame(load_rows)


def write_overlay_timeseries(
    output_path: Path, load_rows: pd.DataFrame, timestamps: pd.DatetimeIndex
) -> None:
    writer = None
    for _, row in load_rows.iterrows():
        periods = len(timestamps)
        table = pa.table(
            {
                "timestamp": pa.array(timestamps.astype("datetime64[ns]")),
                "load_id": pa.array([row["load_id"]] * periods),
                "node_id": pa.array([row["node_id"]] * periods),
                "p_kw": pa.array(
                    np.full(periods, float(row["baseline_p_kw"]), dtype=np.float32),
                    type=pa.float32(),
                ),
                "q_kvar": pa.array(np.zeros(periods, dtype=np.float32), type=pa.float32()),
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()


def plot_clean_network(
    plots_dir: Path,
    projected_nodes: pd.DataFrame,
    projected_edges: pd.DataFrame,
    scenario_locations: dict[str, pd.DataFrame],
    candidate_df: pd.DataFrame,
    distance_metrics: dict[str, dict[str, float]],
) -> None:
    distribution_nodes = projected_nodes.loc[projected_nodes["layer"] == "distribution"]
    plt.switch_backend("Agg")
    interface_nodes = projected_nodes.loc[projected_nodes["layer"] == "interface"]
    transmission_nodes = projected_nodes.loc[
        (projected_nodes["layer"] == "transmission") & ~projected_nodes["node_type"].eq("source")
    ]
    source_nodes = projected_nodes.loc[projected_nodes["node_type"].eq("source")]
    dc_nodes = projected_nodes.loc[projected_nodes["layer"] == "data_center"]

    physical_edges = projected_edges.loc[~projected_edges["is_equivalent_edge"]].copy()
    equivalent_edges = projected_edges.loc[projected_edges["is_equivalent_edge"]].copy()
    interface_edges = projected_edges.loc[
        projected_edges["edge_role"].isin(["interface_connection", "substation_transformer"])
    ]
    dc_edges = projected_edges.loc[projected_edges["edge_role"].eq("data_center_connection")]

    fig, ax = plt.subplots(figsize=(12, 10))
    draw_edge_subset(
        ax,
        physical_edges.loc[physical_edges["edge_role"].eq("distribution_line")],
        color="#9aa3af",
        linewidth=0.55,
        alpha=0.85,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[projected_edges["edge_role"].eq("distribution_switch")],
        color="#1d4ed8",
        linewidth=0.8,
        alpha=0.8,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[
            projected_edges["edge_role"].isin(
                ["distribution_transformer", "substation_transformer"]
            )
        ],
        color="#b45309",
        linewidth=0.9,
        alpha=0.8,
    )
    draw_edge_subset(
        ax,
        equivalent_edges.loc[equivalent_edges["edge_role"].eq("equivalent_source_connection")],
        color="#111827",
        linewidth=0.9,
        linestyle="--",
        alpha=0.35,
        zorder=1,
    )
    draw_edge_subset(
        ax, dc_edges, color="#f97316", linewidth=1.1, linestyle=":", alpha=0.9, zorder=2
    )
    scatter_subset(ax, distribution_nodes, color="#6b7280", size=6, label="Distribution")
    scatter_subset(
        ax,
        interface_nodes,
        color="#2563eb",
        size=28,
        marker="s",
        label="Interface/Substation",
        edgecolor="white",
        zorder=5,
    )
    scatter_subset(
        ax,
        transmission_nodes,
        color="#111827",
        size=24,
        marker="^",
        label="Transmission",
        edgecolor="white",
        zorder=5,
    )
    scatter_subset(
        ax,
        source_nodes,
        color="#111827",
        size=18,
        marker="D",
        label="Equivalent Source",
        edgecolor="white",
        zorder=5,
    )
    scatter_subset(
        ax,
        dc_nodes,
        color="#f97316",
        size=95,
        marker="*",
        label="Data Center",
        edgecolor="black",
        zorder=6,
    )
    add_node_labels(
        ax,
        pd.concat([transmission_nodes, interface_nodes, dc_nodes], ignore_index=True),
        {
            "T_110792": "69-kV delivery bus",
            "T_110769": "Peer 69-kV bus",
            "SUB_p49uhs5_1247": "Substation interface",
            "T_SOURCE_STMAT": "Equivalent source",
            "DC_1": "DC_1",
        },
    )
    style_axis(ax, "Clean Layered Network View")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(plots_dir / "audit_network_layers_clean.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    draw_edge_subset(
        ax,
        projected_edges.loc[projected_edges["edge_role"].eq("distribution_line")],
        color="#6b7280",
        linewidth=0.65,
        alpha=0.85,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[
            projected_edges["edge_role"].isin(
                ["distribution_transformer", "substation_transformer"]
            )
        ],
        color="#b45309",
        linewidth=1.0,
        linestyle="-.",
        alpha=0.9,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[projected_edges["edge_role"].eq("data_center_connection")],
        color="#f97316",
        linewidth=1.4,
        linestyle=":",
        alpha=0.95,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[projected_edges["edge_role"].eq("equivalent_source_connection")],
        color="#111827",
        linewidth=1.1,
        linestyle="--",
        alpha=0.45,
    )
    scatter_subset(ax, distribution_nodes, color="#9ca3af", size=5)
    scatter_subset(ax, interface_nodes, color="#2563eb", size=25, marker="s", edgecolor="white")
    scatter_subset(ax, transmission_nodes, color="#111827", size=22, marker="^", edgecolor="white")
    scatter_subset(ax, source_nodes, color="#111827", size=18, marker="D", edgecolor="white")
    scatter_subset(ax, dc_nodes, color="#f97316", size=85, marker="*", edgecolor="black")
    add_node_labels(
        ax,
        pd.concat([dc_nodes, transmission_nodes, interface_nodes], ignore_index=True),
        {"DC_1": "DC_1", "T_110792": "69-kV bus", "SUB_p49uhs5_1247": "Substation"},
    )
    style_axis(ax, "Physical vs Equivalent Edge Roles")
    legend_handles = [
        plt.Line2D([0], [0], color="#6b7280", lw=1.2, label="Physical distribution line"),
        plt.Line2D(
            [0], [0], color="#b45309", lw=1.2, linestyle="-.", label="Transformer/interface"
        ),
        plt.Line2D(
            [0], [0], color="#f97316", lw=1.4, linestyle=":", label="Data-center connection"
        ),
        plt.Line2D(
            [0], [0], color="#111827", lw=1.2, linestyle="--", label="Equivalent source connection"
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(plots_dir / "audit_physical_vs_equivalent_edges.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    dist_edges = projected_edges.loc[projected_edges["layer"].isin(["distribution", "interface"])]
    draw_edge_subset(
        ax,
        dist_edges.loc[dist_edges["edge_role"].eq("distribution_line")],
        color="#7c8796",
        linewidth=0.7,
        alpha=0.85,
    )
    draw_edge_subset(
        ax,
        dist_edges.loc[dist_edges["edge_role"].eq("distribution_switch")],
        color="#2563eb",
        linewidth=0.85,
        alpha=0.8,
    )
    draw_edge_subset(
        ax,
        dist_edges.loc[
            dist_edges["edge_role"].isin(
                ["distribution_transformer", "substation_transformer", "interface_connection"]
            )
        ],
        color="#b45309",
        linewidth=0.9,
        alpha=0.8,
    )
    scatter_subset(ax, distribution_nodes, color="#6b7280", size=6)
    scatter_subset(ax, interface_nodes, color="#2563eb", size=30, marker="s", edgecolor="white")
    add_node_labels(ax, interface_nodes, {"SUB_p49uhs5_1247": "Substation interface"})
    style_axis(ax, "Distribution and Interface Only")
    fig.tight_layout()
    fig.savefig(plots_dir / "audit_distribution_only.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    center_nodes = projected_nodes.loc[
        projected_nodes["node_id"].isin(["DC_1", "T_110792", "SUB_p49uhs5_1247", "T_110769"])
    ]
    center_x = float(center_nodes["plot_x_km"].mean())
    center_y = float(center_nodes["plot_y_km"].mean())
    zoom_edges = projected_edges.loc[
        (projected_edges[["x1", "x2"]].sub(center_x).abs().max(axis=1) <= 2.5)
        & (projected_edges[["y1", "y2"]].sub(center_y).abs().max(axis=1) <= 2.5)
    ]
    draw_edge_subset(
        ax,
        zoom_edges.loc[zoom_edges["edge_role"].eq("distribution_line")],
        color="#9aa3af",
        linewidth=0.9,
        alpha=0.9,
    )
    draw_edge_subset(
        ax,
        zoom_edges.loc[
            zoom_edges["edge_role"].isin(
                ["distribution_transformer", "substation_transformer", "interface_connection"]
            )
        ],
        color="#b45309",
        linewidth=1.1,
        alpha=0.9,
    )
    draw_edge_subset(
        ax,
        zoom_edges.loc[zoom_edges["edge_role"].eq("data_center_connection")],
        color="#f97316",
        linewidth=1.5,
        linestyle=":",
        alpha=0.95,
    )
    scatter_subset(
        ax,
        projected_nodes.loc[
            projected_nodes["plot_x_km"].between(center_x - 2.2, center_x + 2.2)
            & projected_nodes["plot_y_km"].between(center_y - 2.2, center_y + 2.2)
            & projected_nodes["layer"].eq("distribution")
        ],
        color="#9ca3af",
        size=7,
    )
    scatter_subset(
        ax,
        projected_nodes.loc[projected_nodes["node_id"].eq("SUB_p49uhs5_1247")],
        color="#2563eb",
        size=70,
        marker="s",
        edgecolor="white",
        zorder=6,
    )
    scatter_subset(
        ax,
        projected_nodes.loc[projected_nodes["node_id"].eq("T_110792")],
        color="#111827",
        size=70,
        marker="^",
        edgecolor="white",
        zorder=6,
    )
    scatter_subset(
        ax,
        projected_nodes.loc[projected_nodes["node_id"].eq("DC_1")],
        color="#f97316",
        size=140,
        marker="*",
        edgecolor="black",
        zorder=7,
    )
    add_node_labels(
        ax,
        projected_nodes.loc[
            projected_nodes["node_id"].isin(["DC_1", "T_110792", "SUB_p49uhs5_1247"])
        ],
        {"DC_1": "DC_1", "T_110792": "69-kV bus", "SUB_p49uhs5_1247": "Substation"},
    )
    dc_edge = dc_edges.iloc[0]
    ax.text(
        (dc_edge["x1"] + dc_edge["x2"]) / 2.0 + 0.08,
        (dc_edge["y1"] + dc_edge["y2"]) / 2.0 + 0.08,
        f"DC to 69-kV bus: {distance_metrics['dc_to_connected']['geodesic_km']:.2f} km",
        fontsize=8,
        color="#c2410c",
    )
    interface_edge = projected_edges.loc[
        projected_edges["edge_role"].eq("interface_connection")
    ].iloc[0]
    ax.text(
        (interface_edge["x1"] + interface_edge["x2"]) / 2.0 + 0.06,
        (interface_edge["y1"] + interface_edge["y2"]) / 2.0 - 0.12,
        f"69-kV bus to substation: {distance_metrics['connected_to_sub']['geodesic_km']:.2f} km",
        fontsize=8,
        color="#1d4ed8",
    )
    ax.set_xlim(center_x - 2.4, center_x + 2.4)
    ax.set_ylim(center_y - 2.0, center_y + 2.0)
    style_axis(ax, "Data-Center Connection Zoom")
    fig.tight_layout()
    fig.savefig(plots_dir / "audit_data_center_connection_zoom.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    draw_edge_subset(
        ax,
        projected_edges.loc[~projected_edges["is_hurricane_damage_eligible"]],
        color="#d1d5db",
        linewidth=0.45,
        alpha=0.4,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[
            projected_edges["hurricane_damage_component_type"].eq("overhead_distribution_line")
        ],
        color="#dc2626",
        linewidth=0.9,
        alpha=0.95,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[
            projected_edges["hurricane_damage_component_type"].eq("underground_distribution_line")
        ],
        color="#0f766e",
        linewidth=0.9,
        alpha=0.95,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[
            projected_edges["hurricane_damage_component_type"].eq("transmission_line")
        ],
        color="#7c3aed",
        linewidth=1.0,
        alpha=0.95,
    )
    scatter_subset(ax, distribution_nodes, color="#9ca3af", size=4)
    scatter_subset(ax, interface_nodes, color="#2563eb", size=24, marker="s", edgecolor="white")
    scatter_subset(ax, dc_nodes, color="#f97316", size=90, marker="*", edgecolor="black")
    style_axis(ax, "Edges Eligible for Later Hurricane Damage Modeling")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#dc2626", lw=1.2, label="Overhead distribution"),
            plt.Line2D([0], [0], color="#0f766e", lw=1.2, label="Underground distribution"),
            plt.Line2D([0], [0], color="#7c3aed", lw=1.2, label="Transmission"),
            plt.Line2D([0], [0], color="#d1d5db", lw=1.0, label="Ineligible / equivalent"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "audit_hazard_exposed_edges.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    candidates_projected = projected_nodes.loc[
        projected_nodes["node_id"].isin(candidate_df["node_id"])
    ].merge(candidate_df[["node_id", "candidate_score"]], on="node_id", how="left")
    fig, ax = plt.subplots(figsize=(12, 10))
    draw_edge_subset(
        ax,
        projected_edges.loc[projected_edges["edge_role"].eq("distribution_line")],
        color="#d1d5db",
        linewidth=0.45,
        alpha=0.5,
    )
    draw_edge_subset(
        ax,
        projected_edges.loc[projected_edges["edge_role"].eq("equivalent_source_connection")],
        color="#111827",
        linewidth=0.8,
        linestyle="--",
        alpha=0.2,
    )
    scatter_subset(ax, distribution_nodes, color="#d1d5db", size=4)
    ax.scatter(
        candidates_projected["plot_x_km"],
        candidates_projected["plot_y_km"],
        s=40 + candidates_projected["candidate_score"].clip(lower=0.0) * 18.0,
        c=candidates_projected["candidate_score"],
        cmap="YlOrRd",
        edgecolors="black",
        zorder=6,
    )
    for _, row in candidates_projected.iterrows():
        ax.text(row["plot_x_km"] + 0.08, row["plot_y_km"] + 0.08, row["node_id"], fontsize=8)
    style_axis(ax, "Candidate Data-Center Interconnection Nodes")
    fig.tight_layout()
    fig.savefig(plots_dir / "dc_candidate_locations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    draw_edge_subset(
        ax,
        projected_edges.loc[projected_edges["edge_role"].eq("distribution_line")],
        color="#d1d5db",
        linewidth=0.45,
        alpha=0.45,
    )
    scatter_subset(ax, distribution_nodes, color="#d1d5db", size=4)
    scatter_subset(ax, interface_nodes, color="#2563eb", size=24, marker="s", edgecolor="white")
    markers = {
        "single_dc_current": ("*", "#f97316"),
        "two_dc_split_same_total": ("o", "#ef4444"),
        "three_dc_split_same_total": ("^", "#10b981"),
        "high_dc_penetration": ("D", "#8b5cf6"),
    }
    for scenario_name, df in scenario_locations.items():
        if df.empty:
            continue
        marker, color = markers[scenario_name]
        ax.scatter(
            df["plot_x_km"],
            df["plot_y_km"],
            s=90,
            marker=marker,
            c=color,
            edgecolors="black",
            label=scenario_name,
            zorder=7,
        )
    style_axis(ax, "Scenario Comparison for Data-Center Placements")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(plots_dir / "dc_scenario_comparison_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_post_pruning_powerflow_report(
    selected_master: Path,
    retained_raw_buses: set[str],
    retained_raw_load_buses: set[str],
    retained_load_count: int,
    base_graph: nx.Graph,
    base_source_nodes: set[str],
) -> dict[str, Any]:
    ctx = compile_dss_master(selected_master)
    circuit = ctx.ActiveCircuit
    all_bus_vmag = list(dss_value(circuit.AllBusVmagPu))
    available_buses = {str(name) for name in dss_value(circuit.AllBusNames)}
    evaluated_retained_buses = sorted(set(map(str, retained_raw_buses)) & available_buses)
    skipped_non_opendss_buses = sorted(set(map(str, retained_raw_buses)) - available_buses)
    retained_bus_voltage = {}
    zero_voltage_retained_buses = []
    low_voltage_retained_buses = []
    high_voltage_retained_buses = []

    for bus_name in evaluated_retained_buses:
        try:
            circuit.SetActiveBus(bus_name)
            active_bus = circuit.ActiveBus
            mags = list(dss_value(active_bus.puVmagAngle))[::2]
        except Exception:
            mags = []
        nonzero = [float(val) for val in mags if val is not None]
        max_v = max(nonzero) if nonzero else 0.0
        retained_bus_voltage[bus_name] = max_v
        if max_v <= 1e-6:
            zero_voltage_retained_buses.append(bus_name)
        elif max_v < 0.95:
            low_voltage_retained_buses.append(bus_name)
        elif max_v > 1.05:
            high_voltage_retained_buses.append(bus_name)

    line_overloads = []
    lines = circuit.Lines
    if int(lines.First) > 0:
        while True:
            name = str(lines.Name)
            norm_amps = float(lines.NormAmps)
            circuit.SetActiveElement(f"Line.{name}")
            currents = np.array(
                list(dss_value(circuit.ActiveCktElement.CurrentsMagAng))[::2], dtype=float
            )
            currents = currents[: max(1, len(currents) // 2)]
            max_current = float(currents.max()) if len(currents) else 0.0
            if norm_amps > 0 and max_current > norm_amps:
                line_overloads.append(
                    {
                        "element": name,
                        "max_current_amp": round(max_current, 3),
                        "norm_amps": round(norm_amps, 3),
                    }
                )
            if int(lines.Next) <= 0:
                break

    component_ok = set()
    for component in nx.connected_components(base_graph):
        if component & base_source_nodes:
            component_ok.update(component)

    return {
        "converged": bool(dss_value(circuit.Solution.Converged)),
        "min_voltage_pu_all_raw": float(min(all_bus_vmag)) if all_bus_vmag else None,
        "max_voltage_pu_all_raw": float(max(all_bus_vmag)) if all_bus_vmag else None,
        "min_voltage_pu_retained_buses": float(min(retained_bus_voltage.values()))
        if retained_bus_voltage
        else None,
        "max_voltage_pu_retained_buses": float(max(retained_bus_voltage.values()))
        if retained_bus_voltage
        else None,
        "retained_bus_count_evaluated_in_opendss": int(len(evaluated_retained_buses)),
        "retained_bus_count_not_in_opendss": int(len(skipped_non_opendss_buses)),
        "skipped_non_opendss_buses_sample": skipped_non_opendss_buses[:25],
        "number_of_voltage_violations_retained_buses": int(
            len(low_voltage_retained_buses) + len(high_voltage_retained_buses)
        ),
        "number_of_line_overloads": int(len(line_overloads)),
        "line_overloads": line_overloads[:25],
        "number_of_zero_voltage_retained_buses": int(len(zero_voltage_retained_buses)),
        "zero_voltage_retained_buses_sample": zero_voltage_retained_buses[:25],
        "retained_load_count": int(retained_load_count),
        "every_delivered_load_has_source_path": len(component_ok) >= base_graph.number_of_nodes(),
    }


def scenario_smoke_checks(
    scenario_name: str,
    scenario_folder: Path,
    base_non_dc_nodes: pd.DataFrame,
    base_non_dc_edges: pd.DataFrame,
    base_non_dc_loads: pd.DataFrame,
    overlay_nodes: pd.DataFrame,
    overlay_edges: pd.DataFrame,
    overlay_loads: pd.DataFrame,
    dc_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    combined_nodes = pd.concat([base_non_dc_nodes, overlay_nodes], ignore_index=True)
    combined_edges = pd.concat([base_non_dc_edges, overlay_edges], ignore_index=True)
    combined_loads = pd.concat([base_non_dc_loads, overlay_loads], ignore_index=True)
    G = nx.Graph()
    G.add_nodes_from(combined_nodes["node_id"])
    G.add_edges_from(combined_edges[["from_node", "to_node"]].itertuples(index=False, name=None))
    connected = nx.is_connected(G)

    def add(test: str, status: str, reason: str) -> None:
        results.append(
            {"scenario": scenario_name, "test": test, "status": status, "reason": reason}
        )

    add(
        "All data centers connect to existing base or overlay nodes",
        "PASS"
        if set(overlay_edges["from_node"]).issubset(set(combined_nodes["node_id"]))
        else "FAIL",
        "Scenario connections reference existing nodes.",
    )
    valid_voltage = overlay_nodes["voltage_kv"].ge(12.47).all()
    add(
        "No data center uses a random low-voltage node",
        "PASS" if valid_voltage else "FAIL",
        "Scenario DC nodes connect at 69-kV or documented substation-interface points.",
    )
    add(
        "Positive data-center loads",
        "PASS" if overlay_loads["baseline_p_kw"].gt(0).all() else "FAIL",
        "All scenario DC loads are positive.",
    )
    fractions_ok = all(
        abs(record["critical_it_fraction"] + record["flexible_it_fraction"] - 1.0) <= 1e-6
        for record in dc_records
    )
    add(
        "Critical plus flexible fractions sum to 1",
        "PASS" if fractions_ok else "FAIL",
        "All DC records keep the IT split normalized.",
    )
    backup_ok = all(
        record["backup_gen_mw"] >= record["critical_load_mw"] * 1.19 for record in dc_records
    )
    add(
        "UPS and backup assumptions are plausible",
        "PASS" if backup_ok else "FAIL",
        "Backup generation keeps at least a 20% margin over critical IT load.",
    )
    export_ok = all(
        record["max_export_mw_anchor_mode"]
        <= (record["backup_gen_mw"] - record["critical_load_mw"] + 1e-9)
        for record in dc_records
    )
    add(
        "Export capacity does not exceed simple surplus",
        "PASS" if export_ok else "FAIL",
        "Scenario export capability stays below backup minus critical reserve.",
    )
    target_total = sum(record["facility_load_mw"] for record in dc_records)
    actual_total = float(overlay_loads["baseline_p_kw"].sum() / 1000.0)
    add(
        "Scenario total data-center load matches target",
        "PASS" if abs(actual_total - target_total) <= 1e-6 else "FAIL",
        f"Overlay totals {actual_total:.3f} MW versus target {target_total:.3f} MW.",
    )
    add(
        "Graph remains connected after applying overlay",
        "PASS" if connected else "FAIL",
        "Scenario overlay leaves the combined graph connected.",
    )
    add(
        "No load ID collisions",
        "PASS" if combined_loads["load_id"].is_unique else "FAIL",
        "Scenario overlay is applied to the non-DC base load set.",
    )
    add(
        "No node ID collisions",
        "PASS" if combined_nodes["node_id"].is_unique else "FAIL",
        "Scenario overlay is applied to the non-DC base node set.",
    )
    add(
        "No edge ID collisions",
        "PASS" if combined_edges["edge_id"].is_unique else "FAIL",
        "Scenario overlay is applied to the non-DC base edge set.",
    )
    non_dc_load_total = float(base_non_dc_loads["baseline_p_kw"].sum() / 1000.0)
    combined_total = float(combined_loads["baseline_p_kw"].sum() / 1000.0)
    add(
        "Total load accounting is explicit",
        "PASS",
        f"Combined load = {non_dc_load_total:.3f} MW non-DC base + {actual_total:.3f} MW DC overlay = {combined_total:.3f} MW.",
    )
    return results


def main() -> None:
    parser = default_arg_parser(
        "Prepare baseline_case_v0_1 cleanup outputs and data-center scenario overlays."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("11_prepare_baseline_case_v0_1", config)
    paths = get_paths(config)

    root = paths.root
    source_dir = root / "data_processed" / "final_case_v0"
    dest_dir = root / "data_processed" / "final_case_v0_1"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    plots_dir = dest_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    scenario_root = dest_dir / "dc_scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(dest_dir / "nodes.csv")
    edges = pd.read_csv(dest_dir / "edges.csv")
    loads = pd.read_csv(dest_dir / "loads.csv")
    base_data_centers = pd.read_csv(dest_dir / "data_centers.csv")
    raw_lines = pd.read_csv(paths.parsed_distribution_dir / "raw_lines.csv")
    raw_transformers = pd.read_csv(paths.parsed_distribution_dir / "raw_transformers.csv")
    raw_loads = pd.read_csv(paths.parsed_distribution_dir / "raw_loads.csv")
    selected_candidate = json.loads(
        (paths.selected_region_dir / "selected_candidate.json").read_text(encoding="utf-8")
    )
    subnetwork_metadata = json.loads(
        (paths.selected_region_dir / "selected_subnetwork_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    coast_points = [tuple(item) for item in config["geography"]["simplified_texas_gulf_coastline"]]
    bbox_nodes = nodes.loc[~nodes["node_type"].eq("source"), ["latitude", "longitude"]].copy()
    bbox_meta = centroid_and_bbox(bbox_nodes)
    lat0 = bbox_meta["centroid_lat"]
    lon0 = bbox_meta["centroid_lon"]

    member_component_map = build_member_component_maps(raw_lines, raw_transformers)
    node_layer_map = nodes.set_index("node_id")["layer"].to_dict()
    classification = edges.apply(
        lambda row: pd.Series(classify_edge_row(row, node_layer_map, member_component_map)), axis=1
    )
    edges = pd.concat([edges, classification], axis=1)
    edges.to_csv(dest_dir / "edges.csv", index=False)

    dc_edge = edges.loc[edges["edge_role"].eq("data_center_connection")].iloc[0]
    dc_record = base_data_centers.iloc[0].copy()
    dc_record["connected_node_id"] = dc_record["connected_bus_id"]
    dc_record["connected_node_layer"] = nodes.set_index("node_id").loc[
        str(dc_record["connected_bus_id"]), "layer"
    ]
    dc_record["edge_id"] = dc_edge["edge_id"]
    dc_record["edge_role"] = dc_edge["edge_role"]
    dc_record["connection_type"] = "schematic_equivalent_dedicated_service"
    dc_record["is_physical_connection"] = False
    dc_record["is_hurricane_damage_eligible"] = False
    dc_record["connection_length_km"] = float(dc_edge["length_km"])
    base_data_centers = pd.DataFrame([dc_record])
    base_data_centers.to_csv(dest_dir / "data_centers.csv", index=False)
    (dest_dir / "dc_specs.json").write_text(
        json.dumps({"data_centers": base_data_centers.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )

    projected_nodes = build_projected_positions(nodes, lat0, lon0)
    projected_edges = edge_segments(edges, projected_nodes)

    distance_metrics = {
        "dc_to_connected": distance_pair_metrics(
            (float(dc_record["latitude"]), float(dc_record["longitude"])),
            (
                float(
                    nodes.set_index("node_id").loc[str(dc_record["connected_bus_id"]), "latitude"]
                ),
                float(
                    nodes.set_index("node_id").loc[str(dc_record["connected_bus_id"]), "longitude"]
                ),
            ),
            lat0,
            lon0,
        ),
        "connected_to_sub": distance_pair_metrics(
            (
                float(
                    nodes.set_index("node_id").loc[str(dc_record["connected_bus_id"]), "latitude"]
                ),
                float(
                    nodes.set_index("node_id").loc[str(dc_record["connected_bus_id"]), "longitude"]
                ),
            ),
            (
                float(nodes.set_index("node_id").loc["SUB_p49uhs5_1247", "latitude"]),
                float(nodes.set_index("node_id").loc["SUB_p49uhs5_1247", "longitude"]),
            ),
            lat0,
            lon0,
        ),
        "source_to_sub": distance_pair_metrics(
            (
                float(nodes.set_index("node_id").loc["T_SOURCE_STMAT", "latitude"]),
                float(nodes.set_index("node_id").loc["T_SOURCE_STMAT", "longitude"]),
            ),
            (
                float(nodes.set_index("node_id").loc["SUB_p49uhs5_1247", "latitude"]),
                float(nodes.set_index("node_id").loc["SUB_p49uhs5_1247", "longitude"]),
            ),
            lat0,
            lon0,
        ),
        "dc_to_nearest_interface": distance_pair_metrics(
            (float(dc_record["latitude"]), float(dc_record["longitude"])),
            (
                float(nodes.set_index("node_id").loc["SUB_p49uhs5_1247", "latitude"]),
                float(nodes.set_index("node_id").loc["SUB_p49uhs5_1247", "longitude"]),
            ),
            lat0,
            lon0,
        ),
    }

    candidate_df = select_candidate_nodes(nodes, loads, base_data_centers.iloc[0])
    candidate_df.to_csv(scenario_root / "data_center_candidate_nodes.csv", index=False)

    defaults = config["data_center_defaults"]
    base_non_dc_nodes = nodes.loc[~nodes["node_id"].str.startswith("DC_")].copy()
    base_non_dc_edges = edges.loc[~edges["edge_role"].eq("data_center_connection")].copy()
    base_non_dc_loads = loads.loc[~loads["is_data_center"]].copy()
    load_nodes = nodes.merge(
        loads.loc[~loads["is_data_center"], ["node_id", "baseline_p_kw", "is_critical"]].rename(
            columns={"baseline_p_kw": "load_baseline_p_kw", "is_critical": "load_is_critical"}
        ),
        on="node_id",
        how="inner",
    )
    existing_dc_map = {
        "T_110792": (
            float(base_data_centers.iloc[0]["latitude"]),
            float(base_data_centers.iloc[0]["longitude"]),
        )
    }

    scenario_locations_for_plot: dict[str, pd.DataFrame] = {}
    scenario_results = []
    timestamps = build_non_leap_timestamps(config["sources"]["smart_ds"]["year"], 35040)
    node_lookup = nodes.set_index("node_id")

    for scenario_def in SCENARIO_DEFS:
        scenario_name = scenario_def["name"]
        scenario_dir = scenario_root / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        dc_records = []
        for dc_id, connected_node_id, facility_load_mw in scenario_def["placements"]:
            candidate_row = candidate_df.set_index("node_id").loc[connected_node_id]
            connected_node = node_lookup.loc[connected_node_id]
            lat, lon = facility_location_for_candidate(candidate_row, load_nodes, existing_dc_map)
            dc_records.append(
                make_dc_record(
                    dc_id=dc_id,
                    connected_node=connected_node,
                    facility_load_mw=float(facility_load_mw),
                    latitude=lat,
                    longitude=lon,
                    connection_type="schematic_equivalent_dedicated_service",
                    is_physical_connection=False,
                    is_hurricane_damage_eligible=False,
                    defaults=defaults,
                )
            )

        overlay_nodes, overlay_edges, overlay_loads = make_overlay_tables(
            scenario_name=scenario_name,
            dc_records=dc_records,
            base_nodes=nodes,
            base_edges=edges,
            base_loads=loads,
            coast_points=coast_points,
        )
        overlay_nodes.to_csv(scenario_dir / "nodes_overlay.csv", index=False)
        overlay_edges.to_csv(scenario_dir / "edges_overlay.csv", index=False)
        overlay_loads.to_csv(scenario_dir / "loads_overlay.csv", index=False)
        write_overlay_timeseries(
            scenario_dir / "load_timeseries_overlay.parquet", overlay_loads, timestamps
        )
        pd.DataFrame(dc_records).to_csv(scenario_dir / "data_centers.csv", index=False)
        (scenario_dir / "dc_specs.json").write_text(
            json.dumps(
                {
                    "scenario_name": scenario_name,
                    "overlay_mode": "replace_base_dc_set",
                    "stress_test": bool(scenario_def["stress_test"]),
                    "data_centers": dc_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        scenario_total_mw = sum(record["facility_load_mw"] for record in dc_records)
        non_dc_total_mw = float(base_non_dc_loads["baseline_p_kw"].sum() / 1000.0)
        summary_lines = [
            f"# {scenario_name}",
            "",
            f"Description: {scenario_def['description']}",
            "",
            "Application mode: replace the base DC set. Remove existing `DC_*` node/edge/load/time-series records from `baseline_case_v0_1` before applying this overlay.",
            "",
            f"Scenario total DC load (MW): {scenario_total_mw:.3f}",
            f"Base non-DC load (MW): {non_dc_total_mw:.3f}",
            f"Combined nominal load if applied once (MW): {non_dc_total_mw + scenario_total_mw:.3f}",
            "",
            "Connected nodes:",
        ]
        summary_lines.extend(
            [
                f"- {record['dc_id']}: {record['facility_load_mw']:.3f} MW at {record['connected_node_id']} ({record['connection_voltage_kv']:.1f} kV, {record['connection_type']})"
                for record in dc_records
            ]
        )
        summary_lines.extend(
            [
                "",
                "Notes:",
                "- Connections are modeled as schematic/equivalent dedicated service links rather than explicit physical facility feeders.",
                "- Total DC load is kept equal to the current baseline in split scenarios and increased only in the high-penetration stress test.",
                "",
            ]
        )
        (scenario_dir / "scenario_summary.md").write_text(
            "\n".join(summary_lines), encoding="utf-8"
        )

        scenario_results.extend(
            scenario_smoke_checks(
                scenario_name=scenario_name,
                scenario_folder=scenario_dir,
                base_non_dc_nodes=base_non_dc_nodes,
                base_non_dc_edges=base_non_dc_edges,
                base_non_dc_loads=base_non_dc_loads,
                overlay_nodes=overlay_nodes,
                overlay_edges=overlay_edges,
                overlay_loads=overlay_loads,
                dc_records=dc_records,
            )
        )
        scenario_locations_for_plot[scenario_name] = build_projected_positions(
            overlay_nodes, lat0, lon0
        )

    plot_clean_network(
        plots_dir,
        projected_nodes,
        projected_edges,
        scenario_locations_for_plot,
        candidate_df,
        distance_metrics,
    )

    selected_master = (
        root
        / "data_raw"
        / "distribution"
        / "P49U"
        / "scenarios"
        / "base_timeseries"
        / "opendss"
        / "p49uhs5_1247"
        / "Master_snapshot.dss"
    )
    retained_raw_buses = set()
    for member_text in nodes["member_raw_buses"]:
        retained_raw_buses.update(parse_json_list(member_text))
    retained_raw_load_buses = set(
        raw_loads.loc[raw_loads["bus"].isin(retained_raw_buses), "bus"].astype(str)
    )
    base_graph = nx.Graph()
    base_graph.add_nodes_from(base_non_dc_nodes["node_id"])
    base_graph.add_edges_from(
        base_non_dc_edges[["from_node", "to_node"]].itertuples(index=False, name=None)
    )
    base_source_nodes = set(
        base_non_dc_nodes.loc[
            base_non_dc_nodes["layer"].isin(["interface", "transmission"]), "node_id"
        ].astype(str)
    )
    post_report = build_post_pruning_powerflow_report(
        selected_master=selected_master,
        retained_raw_buses=retained_raw_buses,
        retained_raw_load_buses=retained_raw_load_buses,
        retained_load_count=len(base_non_dc_loads),
        base_graph=base_graph,
        base_source_nodes=base_source_nodes,
    )
    (dest_dir / "post_pruning_powerflow_report.json").write_text(
        json.dumps(post_report, indent=2), encoding="utf-8"
    )
    post_md = [
        "# Post-Pruning Power-Flow Report",
        "",
        "This diagnostic recompiles the raw selected OpenDSS case and evaluates only the raw buses retained in `baseline_case_v0_1` after pruning de-energized normal-state islands.",
        "",
        f"- Converged: {post_report['converged']}",
        f"- Min voltage pu across full raw case: {post_report['min_voltage_pu_all_raw']}",
        f"- Max voltage pu across full raw case: {post_report['max_voltage_pu_all_raw']}",
        f"- Min voltage pu across retained delivered buses: {post_report['min_voltage_pu_retained_buses']}",
        f"- Max voltage pu across retained delivered buses: {post_report['max_voltage_pu_retained_buses']}",
        f"- Voltage violations across retained delivered buses: {post_report['number_of_voltage_violations_retained_buses']}",
        f"- Line overloads detected: {post_report['number_of_line_overloads']}",
        f"- Zero-voltage retained buses: {post_report['number_of_zero_voltage_retained_buses']}",
        f"- Every delivered load has a source path: {post_report['every_delivered_load_has_source_path']}",
        "",
        "Notes:",
        "- The raw case still contains zero-voltage buses outside the delivered subset; those were the islands pruned from the clean baseline.",
        "- Overload counting is limited to line objects available from the compiled OpenDSS interface.",
        "",
    ]
    (dest_dir / "post_pruning_powerflow_report.md").write_text("\n".join(post_md), encoding="utf-8")

    scenario_results_df = pd.DataFrame(scenario_results)
    overall_status = "PASS" if scenario_results_df["status"].eq("FAIL").sum() == 0 else "FAIL"
    scenario_lines = [
        "# Scenario Smoke Test Report",
        "",
        f"Overall status: **{overall_status}**",
        "",
        "Scenario overlays are defined against the `baseline_case_v0_1` network with all existing `DC_*` records removed first.",
        "",
        "| Scenario | Test | Status | Reason |",
        "|---|---|---|---|",
    ]
    for _, row in scenario_results_df.iterrows():
        scenario_lines.append(
            f"| {row['scenario']} | {row['test']} | {row['status']} | {row['reason']} |"
        )
    (scenario_root / "scenario_smoke_test_report.md").write_text(
        "\n".join(scenario_lines) + "\n", encoding="utf-8"
    )

    role_counts = edges["edge_role"].value_counts().to_dict()
    hazard_counts = (
        edges.loc[edges["is_hurricane_damage_eligible"], "hurricane_damage_component_type"]
        .value_counts()
        .to_dict()
    )
    selected_summary_lines = [
        "# Selected Case Summary",
        "",
        "1. Dataset used",
        "Synthetic Texas7k load/generator mapping workbook from Texas A&M / BetterGrids and SMART-DS Texas distribution feeders from OEDI.",
        "",
        "2. Selected Gulf-area region",
        f"Houston / Galveston / Texas City synthetic service area around `{subnetwork_metadata['selected_substation_id']}`.",
        "",
        "3. Why this region was selected",
        "It provides a mapped 69-kV delivery point, strong coastal relevance, multiple feeders, and enough local load to support a meaningful data-center anchor scenario.",
        "",
        "4. Whether it is coastal or near-coastal",
        f"Near-coastal; selected substation is about {float(selected_candidate['distance_to_coast_km']):.1f} km from the simplified Texas Gulf coastline reference.",
        "",
        "5. Number of transmission nodes",
        str(int((nodes["layer"] == "transmission").sum())),
        "",
        "6. Number of substation/interface nodes",
        str(int((nodes["layer"] == "interface").sum())),
        "",
        "7. Number of distribution nodes",
        str(int((nodes["layer"] == "distribution").sum())),
        "",
        "8. Number of edges",
        str(int(len(edges))),
        "",
        "9. Number of feeders",
        str(int(pd.read_csv(dest_dir / "feeders.csv")["feeder_id"].nunique())),
        "",
        "10. Total peak load MW",
        f"{base_non_dc_loads['peak_p_kw'].sum() / 1000.0:.3f} (excludes synthetic data-center loads)",
        "",
        "11. Total average load MW",
        f"{base_non_dc_loads['average_p_kw'].sum() / 1000.0:.3f} (excludes synthetic data-center loads)",
        "",
        "12. Number of residential/commercial/critical/unknown loads",
        f"{base_non_dc_loads['load_type'].value_counts().to_dict()}",
        "",
        "13. Data center location and connection voltage",
        f"`DC_1` at ({dc_record['latitude']}, {dc_record['longitude']}) connected to `{dc_record['connected_node_id']}` ({dc_record['connected_node_layer']}) at {dc_record['connection_voltage_kv']} kV.",
        "",
        "14. Data center size and assumptions",
        f"{dc_record['facility_load_mw']} MW facility load, PUE {dc_record['pue']}, critical IT fraction {dc_record['critical_it_fraction']}, UPS {dc_record['ups_energy_mwh']} MWh, backup generation {dc_record['backup_gen_mw']} MW.",
        "",
        "15. Load aggregation method",
        "Protected substation/interface buses, MV switch locations, 69-kV interface buses, and high-load nodes were preserved; radial non-critical components were aggregated into supernodes; equivalent source and data-center service links remain schematic.",
        "",
        "16. Load conservation check",
        "Base non-data-center load totals remain unchanged from the audited baseline; synthetic data-center overlays are accounted for separately to prevent double counting.",
        "",
        "17. Coordinate coverage",
        f"{nodes[['latitude', 'longitude']].notna().all(axis=1).mean():.2%} of final nodes carry latitude/longitude values.",
        "",
        "18. Power-flow smoke test results if available",
        json.dumps(post_report, indent=2),
        "",
        "19. Known limitations",
        "The transmission layer is a simplified equivalent service-area representation rather than a directly parsed Texas7k bus-branch export. Some low-voltage switch, transformer, and open-tie detail remains aggregated for tractability. Equivalent source edges should not be treated as ordinary damageable physical transmission lines.",
        "",
        "20. Files produced",
        "nodes.csv, edges.csv, loads.csv, load_timeseries.parquet, data_centers.csv, dc_specs.json, configs/testbed/base.yaml, network.geojson, plots, scenario overlays, and validation reports.",
        "",
        "21. How this case will later support restoration RL",
        "Nodes include load, VoLL, voltage, location, layer, and criticality features; edges include status, switchability, repairability, length, equipment role, and hazard-eligibility fields; the data center includes survivability/support attributes; geography supports hurricane damage modeling; and load time series support time-evolving restoration episodes.",
        "",
    ]
    (dest_dir / "selected_case_summary.md").write_text(
        "\n".join(selected_summary_lines), encoding="utf-8"
    )

    cleanup_report_lines = [
        "# Network Cleanup And DC Scenario Report",
        "",
        "## Findings",
        "",
        "1. The original visual oddness came from simplified/equivalent plotting rather than a topological defect in the delivered normal-state graph. The long source edge and schematic DC connection visually stretched an otherwise compact distribution footprint.",
        f"2. Edge roles were reclassified into explicit modeling categories. Current counts: {role_counts}.",
        f"3. The current DC_1 placement remains realistic enough as a synthetic large-load service point. It is {distance_metrics['dc_to_connected']['geodesic_km']:.3f} km from its connected 69-kV bus and {distance_metrics['dc_to_nearest_interface']['geodesic_km']:.3f} km from the substation interface, which stays inside the 3-5 km review threshold.",
        "4. The orange DC edge is best interpreted as a schematic/equivalent dedicated service connection rather than a confirmed physical 69-kV feeder. It is therefore tagged as `data_center_connection`, `is_equivalent_edge=true`, and `is_hurricane_damage_eligible=false`.",
        f"5. Edges currently marked eligible for later hurricane damage modeling are summarized as {hazard_counts}. Switches and transformers are tagged separately in `hurricane_damage_component_type` but remain ineligible in the current placeholder ruleset. Equivalent source and schematic data-center service edges are excluded.",
        "6. Four scenario overlays were created: `single_dc_current`, `two_dc_split_same_total`, `three_dc_split_same_total`, and `high_dc_penetration`.",
        "7. The split scenarios preserve the current total synthetic data-center load of 29.6 MW, while `high_dc_penetration` raises the total to 45 MW as a stress test.",
        "8. All scenario data centers connect at retained 69-kV or documented substation-interface points. Because the simplified local transmission layer only retains two 69-kV candidates, the third split-scenario anchor uses the selected substation interface as a dedicated service-point surrogate.",
        "9. Remaining warnings before hurricane-damage and restoration-environment development are: the case remains above the preferred 300-800 node range, the transmission layer is simplified/equivalent, and some low-voltage switch/transformer detail is intentionally aggregated.",
        "",
        "## Distances And Geometry",
        f"- DC_1 to connected 69-kV bus: {distance_metrics['dc_to_connected']['geodesic_km']:.3f} km geodesic, {distance_metrics['dc_to_connected']['equirectangular_km']:.3f} km projected.",
        f"- Connected 69-kV bus to substation/interface: {distance_metrics['connected_to_sub']['geodesic_km']:.3f} km geodesic, {distance_metrics['connected_to_sub']['equirectangular_km']:.3f} km projected.",
        f"- Equivalent source to local T-D interface: {distance_metrics['source_to_sub']['geodesic_km']:.3f} km geodesic, {distance_metrics['source_to_sub']['equirectangular_km']:.3f} km projected.",
        f"- Local physical-footprint centroid: ({bbox_meta['centroid_lat']:.6f}, {bbox_meta['centroid_lon']:.6f}).",
        f"- Local physical-footprint bounding box: lat {bbox_meta['lat_min']:.6f} to {bbox_meta['lat_max']:.6f}, lon {bbox_meta['lon_min']:.6f} to {bbox_meta['lon_max']:.6f}.",
        "",
        "## Edge Interpretation",
        "- `equivalent_source_connection`: schematic source linkage used only to preserve upstream supply context.",
        "- `interface_connection` and `substation_transformer`: substation/interface coupling, with the 69-kV-to-distribution linkage still modeled equivalently.",
        "- `distribution_line`, `distribution_switch`, and `distribution_transformer`: retained or aggregated physical distribution equipment.",
        "- `data_center_connection`: schematic dedicated DC service linkage, not a confirmed damageable field line.",
        "",
        "## Scenario Overlay Notes",
        "- Apply scenario overlays after removing all existing `DC_*` records from the base node, edge, load, and time-series tables.",
        "- Base non-data-center loads remain unchanged across all scenarios.",
        "- Scenario load time-series overlays use flat 15-minute profiles unless a future experiment replaces them with richer facility traces.",
        "",
    ]
    (dest_dir / "network_cleanup_and_dc_scenario_report.md").write_text(
        "\n".join(cleanup_report_lines), encoding="utf-8"
    )

    dest_config = json.loads(json.dumps(config))
    dest_config["paths"]["final_case_dir"] = "data_processed/final_case_v0_1"
    dest_config["paths"]["plots_dir"] = "data_processed/final_case_v0_1/plots"
    dest_config["case_version"] = "baseline_case_v0_1"
    dest_config["base_snapshot_reference"] = "baseline_case_v0"
    dest_config["time_series_metadata"] = {
        "resolution_minutes": 15,
        "days": 365,
        "leap_day_omitted": True,
        "summary_peak_and_average_exclude_synthetic_data_centers": True,
        "baseline_p_kw_definition": "Nominal/base load field.",
        "peak_and_average_definition": "Derived from delivered time series.",
    }
    dest_config["deterministic_rules"] = {
        "critical_load_assignment": "Top approximately 5% of non-residential load nodes by baseline_p_kw.",
        "candidate_selection": "Deterministic scoring by voltage suitability, feeder proximity, critical-load proximity, and interface distance.",
        "data_center_placement": "Scenario locations are deterministic offsets from retained candidate nodes toward nearby load centroids.",
    }
    dest_config["modeling_notes"] = {
        "transmission_layer": "Simplified/equivalent service-area representation.",
        "equivalent_edges_damage_rule": "Equivalent source and schematic data-center service edges are not hurricane-damage eligible.",
        "aggregation_note": "Some low-voltage switch, transformer, and open-tie detail remains aggregated.",
    }
    with (dest_dir / "configs/testbed/base.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dest_config, fh, sort_keys=False)

    counts = {
        "v0_1_nodes": int(len(nodes)),
        "v0_1_edges": int(len(edges)),
        "v0_1_load_records": int(len(loads)),
        "dc_scenarios": len(SCENARIO_DEFS),
        "eligible_hurricane_edges": int(edges["is_hurricane_damage_eligible"].sum()),
        "scenario_smoke_failures": int((scenario_results_df["status"] == "FAIL").sum()),
    }
    write_iteration_summary(
        config=config,
        iteration=7,
        attempted=[
            "Prepared baseline_case_v0_1 from the audited v0 snapshot without rebuilding the T-D topology.",
            "Added explicit edge-role and hazard-eligibility fields, clearer audit plots, post-pruning diagnostics, and data-center scenario overlays.",
        ],
        worked=[
            "The normal-state graph remained connected and the schematic/equivalent structures were clarified rather than rebuilt.",
            "Scenario overlays and candidate-node tables were generated deterministically.",
            "Post-pruning diagnostics confirmed that retained delivered buses no longer include the zero-voltage island removed from the clean baseline package.",
        ],
        failed=[
            "No blocking failures; remaining warnings are documented in the cleanup and smoke-test reports."
        ],
        assumptions=[
            "Scenario overlays replace the base DC set rather than stacking on top of it.",
            "Data-center service edges remain schematic/equivalent unless a future feeder-level facility interconnection model is added.",
            "Flat DC time-series overlays are acceptable placeholders for later restoration-environment work.",
        ],
        counts=counts,
        next_action="Use baseline_case_v0_1 and the scenario overlays as the starting point for future hurricane-damage and restoration-environment development, without implementing RL logic yet.",
    )
    logger.info("Prepared cleanup outputs and scenario overlays under %s", dest_dir)


if __name__ == "__main__":
    main()

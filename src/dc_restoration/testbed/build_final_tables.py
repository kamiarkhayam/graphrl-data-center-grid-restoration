from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dc_restoration.paths import repository_root
from dc_restoration.testbed.aggregate_distribution_graph import dominant_class

SCRIPT_DIR = Path(__file__).resolve().parent

from dc_restoration.testbed.utils.geo_utils import (  # noqa: E402
    haversine_km,
    min_distance_to_coast_km,
)
from dc_restoration.testbed.utils.project_utils import (  # noqa: E402
    copy_config_to_final_case,
    default_arg_parser,
    get_paths,
    load_config,
    setup_logging,
)
from dc_restoration.testbed.utils.time_series_utils import (  # noqa: E402
    build_non_leap_timestamps,
    representative_day_mask,
)


def parse_json_list(text) -> list[str]:
    if pd.isna(text) or text == "":
        return []
    if isinstance(text, list):
        return text
    return list(json.loads(text))


def make_priority(load_type: str, is_critical: bool, is_data_center: bool) -> int:
    if is_data_center:
        return 0
    if is_critical:
        return 1
    order = {
        "industrial_or_large_commercial": 2,
        "commercial": 3,
        "residential": 4,
        "unknown": 5,
    }
    return order.get(load_type, 5)


def build_geojson(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, output_path: Path) -> None:
    features = []
    for _, row in nodes_df.iterrows():
        if pd.isna(row["longitude"]) or pd.isna(row["latitude"]):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
                "properties": {
                    "feature_type": "node",
                    "node_id": row["node_id"],
                    "layer": row["layer"],
                    "node_type": row["node_type"],
                    "voltage_kv": row["voltage_kv"],
                },
            }
        )
    node_lookup = nodes_df.set_index("node_id")[["longitude", "latitude"]].to_dict("index")
    for _, row in edges_df.iterrows():
        if row["from_node"] not in node_lookup or row["to_node"] not in node_lookup:
            continue
        p1 = node_lookup[row["from_node"]]
        p2 = node_lookup[row["to_node"]]
        if any(
            pd.isna(val)
            for val in [p1["longitude"], p1["latitude"], p2["longitude"], p2["latitude"]]
        ):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [p1["longitude"], p1["latitude"]],
                        [p2["longitude"], p2["latitude"]],
                    ],
                },
                "properties": {
                    "feature_type": "edge",
                    "edge_id": row["edge_id"],
                    "layer": row["layer"],
                    "edge_type": row["edge_type"],
                    "voltage_kv": row["voltage_kv"],
                },
            }
        )
    output_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )


def write_maps(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, plots_dir: Path) -> None:
    plt.switch_backend("Agg")
    plots_dir.mkdir(parents=True, exist_ok=True)
    node_lookup = nodes_df.set_index("node_id")

    def draw_base(ax, edge_subset):
        for _, edge in edge_subset.iterrows():
            if (
                edge["from_node"] not in node_lookup.index
                or edge["to_node"] not in node_lookup.index
            ):
                continue
            p1 = node_lookup.loc[edge["from_node"]]
            p2 = node_lookup.loc[edge["to_node"]]
            if any(
                pd.isna(v)
                for v in [p1["longitude"], p1["latitude"], p2["longitude"], p2["latitude"]]
            ):
                continue
            ax.plot(
                [p1["longitude"], p2["longitude"]],
                [p1["latitude"], p2["latitude"]],
                color="lightgray",
                linewidth=0.6,
                zorder=1,
            )

    # Layer map
    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, edges_df)
    for layer, color in {
        "transmission": "tab:red",
        "interface": "tab:orange",
        "distribution": "tab:blue",
        "data_center": "black",
    }.items():
        subset = nodes_df.loc[nodes_df["layer"].eq(layer)]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=18 if layer != "data_center" else 80,
            c=color,
            label=layer,
            zorder=2,
        )
    ax.legend()
    ax.set_title("Selected Synthetic Texas T-D Case by Layer")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "network_layer_map.png", dpi=200)
    plt.close(fig)

    # Distribution-only
    fig, ax = plt.subplots(figsize=(12, 9))
    dist_edges = edges_df.loc[edges_df["layer"].isin(["distribution", "interface"])]
    draw_base(ax, dist_edges)
    subset = nodes_df.loc[nodes_df["layer"].isin(["distribution", "interface"])]
    ax.scatter(subset["longitude"], subset["latitude"], s=12, c="tab:blue")
    ax.set_title("Distribution / Interface Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "distribution_only_map.png", dpi=200)
    plt.close(fig)

    # Data center placement
    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, edges_df)
    dist = nodes_df.loc[nodes_df["layer"].eq("distribution")]
    tx = nodes_df.loc[nodes_df["layer"].eq("transmission")]
    dc = nodes_df.loc[nodes_df["layer"].eq("data_center")]
    ax.scatter(
        dist["longitude"], dist["latitude"], s=10, c="steelblue", alpha=0.6, label="distribution"
    )
    ax.scatter(tx["longitude"], tx["latitude"], s=45, c="darkred", label="transmission")
    ax.scatter(dc["longitude"], dc["latitude"], s=120, c="black", marker="*", label="data_center")
    ax.legend()
    ax.set_title("Data Center Placement")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "data_center_placement_map.png", dpi=200)
    plt.close(fig)

    # Load size map
    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, edges_df)
    load_nodes = nodes_df.loc[nodes_df["baseline_p_kw"] > 0].copy()
    sizes = np.clip(np.sqrt(load_nodes["baseline_p_kw"].fillna(0)) * 1.6, 8, 180)
    ax.scatter(
        load_nodes["longitude"],
        load_nodes["latitude"],
        s=sizes,
        c=load_nodes["baseline_p_kw"],
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_title("Load Size Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "load_size_map.png", dpi=200)
    plt.close(fig)

    # Voltage map
    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, edges_df)
    scatter = ax.scatter(
        nodes_df["longitude"],
        nodes_df["latitude"],
        s=16,
        c=nodes_df["voltage_kv"],
        cmap="plasma",
        alpha=0.8,
    )
    fig.colorbar(scatter, ax=ax, label="Voltage kV")
    ax.set_title("Voltage-Level Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "voltage_level_map.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = default_arg_parser("Build final case tables, GeoJSON, plots, and summary files.")
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("09_build_final_tables", config)
    paths = get_paths(config)

    copy_config_to_final_case(config)

    selected = pd.read_json(paths.selected_region_dir / "selected_candidate.json", typ="series")
    metadata = json.loads(
        (paths.selected_region_dir / "selected_subnetwork_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    agg_nodes = pd.read_csv(paths.selected_region_dir / "aggregated_distribution_nodes.csv")
    agg_edges = pd.read_csv(paths.selected_region_dir / "aggregated_distribution_edges.csv")
    tx_nodes = pd.read_csv(paths.selected_region_dir / "transmission_interface_nodes.csv")
    tx_edges = pd.read_csv(paths.selected_region_dir / "transmission_interface_edges.csv")
    dc_df = pd.read_csv(paths.final_case_dir / "data_centers.csv")
    raw_loads = pd.read_csv(paths.parsed_distribution_dir / "raw_loads.csv")
    raw_buses = pd.read_csv(paths.parsed_distribution_dir / "raw_buses.csv")
    raw_lines = pd.read_csv(paths.parsed_distribution_dir / "raw_lines.csv")
    raw_transformers = pd.read_csv(paths.parsed_distribution_dir / "raw_transformers.csv")
    profile_table = pq.read_table(
        paths.parsed_distribution_dir / "raw_load_profiles.parquet"
    ).to_pydict()

    coast_points = [tuple(item) for item in config["geography"]["simplified_texas_gulf_coastline"]]

    # Distribution/interface nodes
    node_records = []
    load_candidate_records = []
    critical_target_fraction = 0.05

    load_type_values = config["voll_values_usd_per_kwh"]
    for _, row in agg_nodes.iterrows():
        member_load_ids = parse_json_list(row["member_raw_load_ids"])
        member_raw_buses = parse_json_list(row["member_raw_buses"])
        is_substation = row["agg_node_id"].startswith("SUB_")
        layer = "interface" if is_substation else "distribution"
        node_id = row["agg_node_id"]
        load_type = (
            row["dominant_customer_class"]
            if pd.notna(row["dominant_customer_class"])
            else "unknown"
        )
        if load_type not in load_type_values:
            load_type = "unknown"
        node_records.append(
            {
                "node_id": node_id,
                "raw_id": row["agg_node_id"],
                "layer": layer,
                "node_type": row.get("node_type", "distribution_bus"),
                "substation_id": metadata["selected_substation_id"],
                "feeder_id": row["feeder_id"],
                "voltage_kv": row["base_kv"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "x": row["longitude"],
                "y": row["latitude"],
                "phase": "3",
                "is_load": bool(row["baseline_p_kw"] > 0),
                "is_generator": False,
                "is_substation": is_substation,
                "is_transmission": False,
                "is_distribution": not is_substation,
                "is_data_center": False,
                "is_critical": False,
                "load_type": load_type,
                "voll_usd_per_kwh": load_type_values.get(load_type, load_type_values["unknown"]),
                "baseline_p_kw": float(row["baseline_p_kw"]),
                "baseline_q_kvar": float(row["baseline_q_kvar"]),
                "peak_p_kw": np.nan,
                "average_p_kw": np.nan,
                "priority_class": make_priority(load_type, False, False),
                "degree": 0,
                "distance_to_substation_km": 0.0,
                "distance_to_data_center_km": np.nan,
                "distance_to_coast_km": min_distance_to_coast_km(
                    float(row["latitude"]), float(row["longitude"]), coast_points
                )
                if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
                else np.nan,
                "hurricane_exposure_placeholder": config["placeholders"][
                    "hurricane_exposure_placeholder"
                ],
                "aggregated_raw_node_count": int(row["aggregated_raw_node_count"]),
                "member_raw_buses": row["member_raw_buses"],
                "member_raw_load_ids": json.dumps(member_load_ids),
            }
        )
        if row["baseline_p_kw"] > 0:
            load_candidate_records.append((node_id, load_type, float(row["baseline_p_kw"])))

    # Critical synthetic assignment
    load_nodes_df = pd.DataFrame(
        load_candidate_records, columns=["node_id", "load_type", "baseline_p_kw"]
    )
    critical_target = max(1, round(len(load_nodes_df) * critical_target_fraction))
    critical_pool = load_nodes_df.loc[~load_nodes_df["load_type"].eq("residential")].copy()
    if critical_pool.empty:
        critical_pool = load_nodes_df.copy()
    critical_ids = set(
        critical_pool.sort_values("baseline_p_kw", ascending=False).head(critical_target)["node_id"]
    )

    # Transmission nodes: keep source + selected + nearest peer
    peer_rows = tx_nodes.loc[tx_nodes["node_type"].eq("peer_delivery_bus")].head(1)
    tx_keep = pd.concat(
        [
            tx_nodes.loc[tx_nodes["node_id"].eq("T_SOURCE_STMAT")],
            tx_nodes.loc[tx_nodes["node_type"].eq("delivery_bus")],
            peer_rows,
        ],
        ignore_index=True,
    )
    for _, row in tx_keep.iterrows():
        node_records.append(
            {
                "node_id": row["node_id"],
                "raw_id": row["transmission_bus_id"],
                "layer": "transmission",
                "node_type": row["node_type"],
                "substation_id": metadata["selected_substation_id"]
                if row["node_type"] != "source"
                else "",
                "feeder_id": "",
                "voltage_kv": row["voltage_kv"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "x": row["longitude"],
                "y": row["latitude"],
                "phase": "3",
                "is_load": False,
                "is_generator": row["node_type"] == "source",
                "is_substation": False,
                "is_transmission": True,
                "is_distribution": False,
                "is_data_center": False,
                "is_critical": False,
                "load_type": "unknown",
                "voll_usd_per_kwh": load_type_values["unknown"],
                "baseline_p_kw": 0.0,
                "baseline_q_kvar": 0.0,
                "peak_p_kw": 0.0,
                "average_p_kw": 0.0,
                "priority_class": 5,
                "degree": 0,
                "distance_to_substation_km": haversine_km(
                    row["latitude"],
                    row["longitude"],
                    float(selected["latitude"]),
                    float(selected["longitude"]),
                ),
                "distance_to_data_center_km": np.nan,
                "distance_to_coast_km": min_distance_to_coast_km(
                    float(row["latitude"]), float(row["longitude"]), coast_points
                ),
                "hurricane_exposure_placeholder": config["placeholders"][
                    "hurricane_exposure_placeholder"
                ],
                "aggregated_raw_node_count": 1,
                "member_raw_buses": json.dumps([str(row["transmission_bus_id"])]),
                "member_raw_load_ids": json.dumps([]),
            }
        )

    # Data center node
    dc = dc_df.iloc[0]
    node_records.append(
        {
            "node_id": dc["dc_id"],
            "raw_id": dc["dc_id"],
            "layer": "data_center",
            "node_type": "data_center_facility",
            "substation_id": metadata["selected_substation_id"],
            "feeder_id": "",
            "voltage_kv": dc["connection_voltage_kv"],
            "latitude": dc["latitude"],
            "longitude": dc["longitude"],
            "x": dc["longitude"],
            "y": dc["latitude"],
            "phase": "3",
            "is_load": True,
            "is_generator": False,
            "is_substation": False,
            "is_transmission": False,
            "is_distribution": False,
            "is_data_center": True,
            "is_critical": True,
            "load_type": "data_center",
            "voll_usd_per_kwh": load_type_values["data_center"],
            "baseline_p_kw": float(dc["facility_load_mw"]) * 1000.0,
            "baseline_q_kvar": 0.0,
            "peak_p_kw": float(dc["facility_load_mw"]) * 1000.0,
            "average_p_kw": float(dc["facility_load_mw"]) * 1000.0,
            "priority_class": 0,
            "degree": 0,
            "distance_to_substation_km": haversine_km(
                float(dc["latitude"]),
                float(dc["longitude"]),
                float(selected["latitude"]),
                float(selected["longitude"]),
            ),
            "distance_to_data_center_km": 0.0,
            "distance_to_coast_km": min_distance_to_coast_km(
                float(dc["latitude"]), float(dc["longitude"]), coast_points
            ),
            "hurricane_exposure_placeholder": config["placeholders"][
                "hurricane_exposure_placeholder"
            ],
            "aggregated_raw_node_count": 1,
            "member_raw_buses": json.dumps([]),
            "member_raw_load_ids": json.dumps([]),
        }
    )

    nodes_df = pd.DataFrame(node_records).drop_duplicates(subset=["node_id"]).reset_index(drop=True)
    nodes_df.loc[nodes_df["node_id"].isin(critical_ids), "is_critical"] = True
    nodes_df.loc[nodes_df["node_id"].isin(critical_ids), "load_type"] = "critical"
    nodes_df.loc[nodes_df["node_id"].isin(critical_ids), "voll_usd_per_kwh"] = load_type_values[
        "critical"
    ]
    nodes_df.loc[nodes_df["node_id"].isin(critical_ids), "priority_class"] = 1

    # Add any connector buses still referenced by aggregated edges but not emitted as nodes.
    agg_endpoint_ids = set(agg_edges["from_node"]).union(set(agg_edges["to_node"]))
    existing_node_ids = set(nodes_df["node_id"])
    missing_endpoints = sorted(agg_endpoint_ids - existing_node_ids)
    raw_bus_lookup = raw_buses.set_index("raw_bus_id")
    connector_rows = []
    for bus_id in missing_endpoints:
        if bus_id not in raw_bus_lookup.index:
            continue
        bus_row = raw_bus_lookup.loc[bus_id]
        bus_loads = raw_loads.loc[raw_loads["bus"].eq(bus_id)]
        load_type = dominant_class(bus_loads) if not bus_loads.empty else "unknown"
        if load_type not in load_type_values:
            load_type = "unknown"
        connector_rows.append(
            {
                "node_id": bus_id,
                "raw_id": bus_id,
                "layer": "interface" if str(bus_id).startswith("sb6_") else "distribution",
                "node_type": "connector_raw_bus",
                "substation_id": metadata["selected_substation_id"],
                "feeder_id": bus_row["feeder_id"],
                "voltage_kv": bus_row["base_kv"],
                "latitude": bus_row["latitude"],
                "longitude": bus_row["longitude"],
                "x": bus_row["longitude"],
                "y": bus_row["latitude"],
                "phase": "3",
                "is_load": bool(bus_loads["p_kw"].sum() > 0),
                "is_generator": False,
                "is_substation": False,
                "is_transmission": False,
                "is_distribution": not str(bus_id).startswith("sb6_"),
                "is_data_center": False,
                "is_critical": False,
                "load_type": load_type,
                "voll_usd_per_kwh": load_type_values.get(load_type, load_type_values["unknown"]),
                "baseline_p_kw": float(bus_loads["p_kw"].sum()),
                "baseline_q_kvar": float(bus_loads["q_kvar"].sum()),
                "peak_p_kw": np.nan,
                "average_p_kw": np.nan,
                "priority_class": make_priority(load_type, False, False),
                "degree": 0,
                "distance_to_substation_km": 0.0,
                "distance_to_data_center_km": np.nan,
                "distance_to_coast_km": min_distance_to_coast_km(
                    float(bus_row["latitude"]), float(bus_row["longitude"]), coast_points
                )
                if pd.notna(bus_row["latitude"]) and pd.notna(bus_row["longitude"])
                else np.nan,
                "hurricane_exposure_placeholder": config["placeholders"][
                    "hurricane_exposure_placeholder"
                ],
                "aggregated_raw_node_count": 1,
                "member_raw_buses": json.dumps([bus_id]),
                "member_raw_load_ids": json.dumps(sorted(bus_loads["raw_load_id"].tolist())),
            }
        )
    if connector_rows:
        nodes_df = pd.concat([nodes_df, pd.DataFrame(connector_rows)], ignore_index=True)
        logger.info(
            "Added %s connector nodes required by aggregated edge endpoints.", len(connector_rows)
        )

    # Edges
    tx_keep_ids = set(
        tx_keep["node_id"].tolist() + ["T_SOURCE_STMAT", metadata["selected_transmission_node_id"]]
    )
    tx_edges_keep = tx_edges.loc[
        tx_edges["from_node"].isin(tx_keep_ids.union({f"SUB_{metadata['selected_substation_id']}"}))
        & tx_edges["to_node"].isin(tx_keep_ids.union({f"SUB_{metadata['selected_substation_id']}"}))
    ].copy()
    edge_records = []
    node_lookup_df = nodes_df.set_index("node_id")
    for _, row in tx_edges_keep.iterrows():
        edge_records.append(
            {
                "edge_id": row["edge_id"],
                "from_node": row["from_node"],
                "to_node": row["to_node"],
                "layer": "transmission" if "TE_" in row["edge_id"] else "interface",
                "edge_type": row["edge_type"],
                "voltage_kv": 230.0
                if "SOURCE" in row["edge_id"]
                else float(selected["voltage_kv"]),
                "phases": "3",
                "length_km": haversine_km(
                    float(node_lookup_df.loc[row["from_node"], "latitude"]),
                    float(node_lookup_df.loc[row["from_node"], "longitude"]),
                    float(node_lookup_df.loc[row["to_node"], "latitude"]),
                    float(node_lookup_df.loc[row["to_node"], "longitude"]),
                ),
                "r_ohm": 0.0,
                "x_ohm": 0.0,
                "rating_amp": np.nan,
                "rating_mva": np.nan,
                "is_switch": False,
                "is_transformer": "XFMR" in row["edge_id"],
                "is_overhead": False,
                "is_underground": False,
                "normal_status": "closed",
                "initial_status": "closed",
                "switchable": False,
                "repairable": True,
                "damage_probability_placeholder": config["placeholders"][
                    "damage_probability_placeholder"
                ],
                "repair_time_hours_placeholder": config["placeholders"][
                    "repair_time_hours_placeholder"
                ],
                "member_raw_edges": json.dumps([]),
            }
        )

    raw_line_lookup = raw_lines.set_index("raw_edge_id")
    raw_xf_lookup = raw_transformers.set_index("raw_transformer_id")
    for _, row in agg_edges.iterrows():
        member_edges = parse_json_list(row["member_raw_edges"])
        rating_amp_values = []
        phases_values = []
        rating_mva_values = []
        voltage_candidates = []
        for edge_id in member_edges:
            if edge_id in raw_line_lookup.index:
                r = raw_line_lookup.loc[edge_id]
                if pd.notna(r.get("normal_amps")):
                    rating_amp_values.append(float(r["normal_amps"]))
                if pd.notna(r.get("phases")):
                    phases_values.append(str(r["phases"]))
                fb = r["from_bus"]
                if fb in nodes_df["raw_id"].values:
                    pass
            elif edge_id in raw_xf_lookup.index:
                xfr = raw_xf_lookup.loc[edge_id]
                if pd.notna(xfr.get("kva")):
                    rating_mva_values.append(float(xfr["kva"]) / 1000.0)
                if pd.notna(xfr.get("phases")):
                    phases_values.append(
                        str(int(xfr["phases"])) if not pd.isna(xfr["phases"]) else "3"
                    )
        from_row = node_lookup_df.loc[row["from_node"]]
        to_row = node_lookup_df.loc[row["to_node"]]
        voltage_kv = max(
            float(from_row["voltage_kv"]) if pd.notna(from_row["voltage_kv"]) else 0.0,
            float(to_row["voltage_kv"]) if pd.notna(to_row["voltage_kv"]) else 0.0,
        )
        edge_type = row["edge_type"]
        layer = (
            "interface"
            if "SUB_" in str(row["from_node"]) or "SUB_" in str(row["to_node"])
            else "distribution"
        )
        edge_records.append(
            {
                "edge_id": row["edge_id"] if "edge_id" in row else row["agg_edge_id"],
                "from_node": row["from_node"],
                "to_node": row["to_node"],
                "layer": layer,
                "edge_type": edge_type,
                "voltage_kv": voltage_kv,
                "phases": phases_values[0] if phases_values else "3",
                "length_km": float(row["length_km"]) if pd.notna(row["length_km"]) else 0.0,
                "r_ohm": float(row["equivalent_r"]) if pd.notna(row["equivalent_r"]) else 0.0,
                "x_ohm": float(row["equivalent_x"]) if pd.notna(row["equivalent_x"]) else 0.0,
                "rating_amp": min(rating_amp_values) if rating_amp_values else np.nan,
                "rating_mva": min(rating_mva_values) if rating_mva_values else np.nan,
                "is_switch": bool(row["is_switch"])
                if pd.notna(row["is_switch"])
                else edge_type == "switch",
                "is_transformer": bool(row["is_transformer"])
                if pd.notna(row["is_transformer"])
                else edge_type == "transformer",
                "is_overhead": bool(row["is_overhead"]) if pd.notna(row["is_overhead"]) else False,
                "is_underground": bool(row["is_underground"])
                if pd.notna(row["is_underground"])
                else False,
                "normal_status": row["normal_status"]
                if pd.notna(row["normal_status"])
                else "closed",
                "initial_status": row["normal_status"]
                if pd.notna(row["normal_status"])
                else "closed",
                "switchable": bool(row["switchable"]) if pd.notna(row["switchable"]) else False,
                "repairable": True,
                "damage_probability_placeholder": config["placeholders"][
                    "damage_probability_placeholder"
                ],
                "repair_time_hours_placeholder": config["placeholders"][
                    "repair_time_hours_placeholder"
                ],
                "member_raw_edges": row["member_raw_edges"],
            }
        )

    # DC connection edge
    edge_records.append(
        {
            "edge_id": "DC_CONN_DC_1",
            "from_node": metadata["selected_transmission_node_id"],
            "to_node": "DC_1",
            "layer": "data_center",
            "edge_type": "dedicated_transformer",
            "voltage_kv": float(dc["connection_voltage_kv"]),
            "phases": "3",
            "length_km": haversine_km(
                float(dc["latitude"]),
                float(dc["longitude"]),
                float(selected["latitude"]),
                float(selected["longitude"]),
            ),
            "r_ohm": 0.0,
            "x_ohm": 0.0,
            "rating_amp": np.nan,
            "rating_mva": float(dc["facility_load_mw"]) * 1.2,
            "is_switch": False,
            "is_transformer": True,
            "is_overhead": False,
            "is_underground": True,
            "normal_status": "closed",
            "initial_status": "closed",
            "switchable": False,
            "repairable": True,
            "damage_probability_placeholder": config["placeholders"][
                "damage_probability_placeholder"
            ],
            "repair_time_hours_placeholder": config["placeholders"][
                "repair_time_hours_placeholder"
            ],
            "member_raw_edges": json.dumps([]),
        }
    )

    edges_df = pd.DataFrame(edge_records).drop_duplicates(subset=["edge_id"]).reset_index(drop=True)

    # Remove any normal-status islands that have no interface/transmission source path.
    # These appear in the raw OpenDSS case as de-energized pockets behind normally open ties,
    # and excluding them yields a cleaner baseline restoration case.
    G_normal = nx.Graph()
    G_normal.add_nodes_from(nodes_df["node_id"])
    closed_edge_mask = edges_df["normal_status"].astype(str).str.strip().str.lower().ne("open")
    G_normal.add_edges_from(
        edges_df.loc[closed_edge_mask, ["from_node", "to_node"]].itertuples(index=False, name=None)
    )
    source_nodes = set(
        nodes_df.loc[nodes_df["layer"].isin(["interface", "transmission"]), "node_id"].astype(str)
    )
    kept_nodes = set()
    for component in nx.connected_components(G_normal):
        if component & source_nodes:
            kept_nodes.update(component)
    dropped_nodes = sorted(set(nodes_df["node_id"]) - kept_nodes)
    if dropped_nodes:
        dropped_load_mw = float(
            nodes_df.loc[nodes_df["node_id"].isin(dropped_nodes), "baseline_p_kw"].sum() / 1000.0
        )
        logger.warning(
            "Dropping %s nodes (%.3f MW) from source-disconnected normal-status islands.",
            len(dropped_nodes),
            dropped_load_mw,
        )
        nodes_df = nodes_df.loc[nodes_df["node_id"].isin(kept_nodes)].copy().reset_index(drop=True)
        edges_df = (
            edges_df.loc[
                edges_df["from_node"].isin(kept_nodes) & edges_df["to_node"].isin(kept_nodes)
            ]
            .copy()
            .reset_index(drop=True)
        )

    # Degree and distances
    G_final = nx.Graph()
    for _, row in nodes_df.iterrows():
        G_final.add_node(row["node_id"])
    for _, row in edges_df.iterrows():
        G_final.add_edge(row["from_node"], row["to_node"])
    node_lookup_df = nodes_df.set_index("node_id")
    sub_lat = float(node_lookup_df.loc[f"SUB_{metadata['selected_substation_id']}", "latitude"])
    sub_lon = float(node_lookup_df.loc[f"SUB_{metadata['selected_substation_id']}", "longitude"])
    dc_lat = float(node_lookup_df.loc["DC_1", "latitude"])
    dc_lon = float(node_lookup_df.loc["DC_1", "longitude"])
    nodes_df["degree"] = nodes_df["node_id"].map(dict(G_final.degree()))
    nodes_df["distance_to_substation_km"] = nodes_df.apply(
        lambda row: (
            haversine_km(float(row["latitude"]), float(row["longitude"]), sub_lat, sub_lon)
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
            else np.nan
        ),
        axis=1,
    )
    nodes_df["distance_to_data_center_km"] = nodes_df.apply(
        lambda row: (
            haversine_km(float(row["latitude"]), float(row["longitude"]), dc_lat, dc_lon)
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
            else np.nan
        ),
        axis=1,
    )

    # Load profiles and timeseries
    profile_ids = profile_table["profile_id"]
    p_profile_arrays = {
        pid: np.array(arr, dtype=np.float32)
        for pid, arr in zip(profile_ids, profile_table["p_values"])
    }
    q_profile_arrays = {
        pid: np.array(arr, dtype=np.float32)
        for pid, arr in zip(profile_ids, profile_table["q_values"])
    }
    periods = len(next(iter(p_profile_arrays.values()))) if p_profile_arrays else 35040
    timestamps = build_non_leap_timestamps(config["sources"]["smart_ds"]["year"], periods)
    hourly_downsample_rule = str(
        config.get("representative_time_series", {}).get("hourly_downsample_rule", "1h")
    ).replace("H", "h")

    final_load_rows = []
    node_profile_coeffs = {}
    for _, node in nodes_df.loc[nodes_df["is_load"]].iterrows():
        node_id = node["node_id"]
        if node["is_data_center"]:
            final_load_rows.append(
                {
                    "load_id": "LD_DC_1",
                    "node_id": "DC_1",
                    "feeder_id": "",
                    "substation_id": metadata["selected_substation_id"],
                    "load_type": "data_center",
                    "voll_usd_per_kwh": load_type_values["data_center"],
                    "baseline_p_kw": float(node["baseline_p_kw"]),
                    "baseline_q_kvar": 0.0,
                    "peak_p_kw": float(node["baseline_p_kw"]),
                    "average_p_kw": float(node["baseline_p_kw"]),
                    "is_critical": True,
                    "is_data_center": True,
                    "profile_id": "constant_dc_profile",
                }
            )
            node_profile_coeffs[node_id] = {"constant": (float(node["baseline_p_kw"]), 0.0)}
            continue
        member_load_ids = parse_json_list(node["member_raw_load_ids"])
        subset = raw_loads.loc[raw_loads["raw_load_id"].isin(member_load_ids)].copy()
        coeffs = subset.groupby("profile_id").agg(p_kw=("p_kw", "sum"), q_kvar=("q_kvar", "sum"))
        node_profile_coeffs[node_id] = {
            pid: (float(vals["p_kw"]), float(vals["q_kvar"]))
            for pid, vals in coeffs.to_dict("index").items()
        }
        final_load_rows.append(
            {
                "load_id": f"LD_{node_id}",
                "node_id": node_id,
                "feeder_id": node["feeder_id"],
                "substation_id": node["substation_id"],
                "load_type": node["load_type"],
                "voll_usd_per_kwh": node["voll_usd_per_kwh"],
                "baseline_p_kw": float(node["baseline_p_kw"]),
                "baseline_q_kvar": float(node["baseline_q_kvar"]),
                "peak_p_kw": np.nan,
                "average_p_kw": np.nan,
                "is_critical": bool(node["is_critical"]),
                "is_data_center": False,
                "profile_id": ",".join(sorted(coeffs.index.astype(str).tolist())),
            }
        )

    load_ids_df = pd.DataFrame(final_load_rows)

    # Batch-write full-resolution timeseries
    full_ts_path = paths.final_case_dir / "load_timeseries.parquet"
    hourly_ts_path = paths.final_case_dir / "load_timeseries_hourly.parquet"
    day_ts_path = paths.final_case_dir / "load_timeseries_restoration_day.parquet"
    day_mask = representative_day_mask(
        timestamps,
        config["representative_time_series"]["representative_month"],
        config["representative_time_series"]["representative_day"],
    )

    writer = None
    writer_hourly = None
    writer_day = None
    peak_map = {}
    avg_map = {}

    for _, load_row in load_ids_df.iterrows():
        load_id = load_row["load_id"]
        node_id = load_row["node_id"]
        if load_row["is_data_center"]:
            p_series = np.full(periods, float(load_row["baseline_p_kw"]), dtype=np.float32)
            q_series = np.zeros(periods, dtype=np.float32)
        else:
            p_series = np.zeros(periods, dtype=np.float32)
            q_series = np.zeros(periods, dtype=np.float32)
            for profile_id, (p_coeff, q_coeff) in node_profile_coeffs[node_id].items():
                if profile_id not in p_profile_arrays:
                    continue
                p_series += p_coeff * p_profile_arrays[profile_id]
                q_series += q_coeff * q_profile_arrays[profile_id]
        peak_map[node_id] = float(p_series.max())
        avg_map[node_id] = float(p_series.mean())

        table = pa.table(
            {
                "timestamp": pa.array(timestamps.astype("datetime64[ns]")),
                "load_id": pa.array([load_id] * periods),
                "node_id": pa.array([node_id] * periods),
                "p_kw": pa.array(p_series, type=pa.float32()),
                "q_kvar": pa.array(q_series, type=pa.float32()),
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(full_ts_path, table.schema)
        writer.write_table(table)

        hourly_df = (
            pd.DataFrame({"timestamp": timestamps, "p_kw": p_series, "q_kvar": q_series})
            .set_index("timestamp")
            .resample(hourly_downsample_rule)
            .mean()
            .reset_index()
        )
        hourly_table = pa.table(
            {
                "timestamp": pa.array(hourly_df["timestamp"].values.astype("datetime64[ns]")),
                "load_id": pa.array([load_id] * len(hourly_df)),
                "node_id": pa.array([node_id] * len(hourly_df)),
                "p_kw": pa.array(hourly_df["p_kw"].astype(np.float32).values, type=pa.float32()),
                "q_kvar": pa.array(
                    hourly_df["q_kvar"].astype(np.float32).values, type=pa.float32()
                ),
            }
        )
        if writer_hourly is None:
            writer_hourly = pq.ParquetWriter(hourly_ts_path, hourly_table.schema)
        writer_hourly.write_table(hourly_table)

        day_table = pa.table(
            {
                "timestamp": pa.array(timestamps[day_mask].astype("datetime64[ns]")),
                "load_id": pa.array([load_id] * int(day_mask.sum())),
                "node_id": pa.array([node_id] * int(day_mask.sum())),
                "p_kw": pa.array(p_series[day_mask], type=pa.float32()),
                "q_kvar": pa.array(q_series[day_mask], type=pa.float32()),
            }
        )
        if writer_day is None:
            writer_day = pq.ParquetWriter(day_ts_path, day_table.schema)
        writer_day.write_table(day_table)

    if writer:
        writer.close()
    if writer_hourly:
        writer_hourly.close()
    if writer_day:
        writer_day.close()

    nodes_df.loc[nodes_df["node_id"].isin(peak_map), "peak_p_kw"] = nodes_df["node_id"].map(
        peak_map
    )
    nodes_df.loc[nodes_df["node_id"].isin(avg_map), "average_p_kw"] = nodes_df["node_id"].map(
        avg_map
    )
    load_ids_df["peak_p_kw"] = (
        load_ids_df["node_id"].map(peak_map).fillna(load_ids_df["baseline_p_kw"])
    )
    load_ids_df["average_p_kw"] = (
        load_ids_df["node_id"].map(avg_map).fillna(load_ids_df["baseline_p_kw"])
    )

    # Tables
    nodes_out = nodes_df.drop(columns=["member_raw_load_ids"])
    nodes_out.to_csv(paths.final_case_dir / "nodes.csv", index=False)
    edges_df.to_csv(paths.final_case_dir / "edges.csv", index=False)
    load_ids_df.to_csv(paths.final_case_dir / "loads.csv", index=False)

    feeder_rows = (
        nodes_df.loc[nodes_df["layer"].eq("distribution")]
        .groupby("feeder_id", as_index=False)
        .agg(
            substation_id=("substation_id", "first"),
            node_count=("node_id", "count"),
            load_node_count=("is_load", "sum"),
            baseline_p_kw=("baseline_p_kw", "sum"),
        )
    )
    feeder_rows.to_csv(paths.final_case_dir / "feeders.csv", index=False)

    substations_df = pd.DataFrame(
        [
            {
                "substation_id": metadata["selected_substation_id"],
                "transmission_bus_id": metadata["selected_transmission_bus_id"],
                "service_area": metadata["service_area"],
                "region_label": selected["region_label"],
                "latitude": selected["latitude"],
                "longitude": selected["longitude"],
                "primary_voltage_kv": selected["voltage_kv"],
                "number_of_feeders": int(feeder_rows["feeder_id"].nunique()),
                "total_peak_load_mw": selected["total_peak_load_mw"],
            }
        ]
    )
    substations_df.to_csv(paths.final_case_dir / "substations.csv", index=False)

    td_mapping = pd.read_csv(
        paths.parsed_transmission_dir / "transmission_distribution_mapping.csv"
    )
    td_mapping = td_mapping.loc[
        td_mapping["substation_id"].eq(metadata["selected_substation_id"])
    ].copy()
    td_mapping["selected_transmission_node_id"] = metadata["selected_transmission_node_id"]
    td_mapping.to_csv(paths.final_case_dir / "transmission_distribution_mapping.csv", index=False)

    build_geojson(nodes_out, edges_df, paths.final_case_dir / "network.geojson")
    write_maps(nodes_out, edges_df, paths.plots_dir)

    kept_raw_buses = set()
    for member_text in nodes_df["member_raw_buses"]:
        kept_raw_buses.update(parse_json_list(member_text))
    raw_loads_for_case = raw_loads.loc[raw_loads["bus"].isin(kept_raw_buses)].copy()

    total_raw_p = float(raw_loads_for_case["p_kw"].sum())
    total_final_p = float(load_ids_df.loc[~load_ids_df["is_data_center"], "baseline_p_kw"].sum())
    total_raw_q = float(raw_loads_for_case["q_kvar"].sum())
    total_final_q = float(load_ids_df.loc[~load_ids_df["is_data_center"], "baseline_q_kvar"].sum())
    conservation_p = (total_final_p - total_raw_p) / total_raw_p if total_raw_p else 0.0
    conservation_q = (total_final_q - total_raw_q) / total_raw_q if total_raw_q else 0.0
    coordinate_coverage = float(nodes_out["latitude"].notna().mean())

    counts_by_type = load_ids_df["load_type"].value_counts().to_dict()
    summary_lines = [
        "# Selected Case Summary",
        "",
        "1. Dataset used",
        "Synthetic Texas7k load/generator mapping workbook from Texas A&M / BetterGrids and SMART-DS Texas distribution feeders from OEDI.",
        "",
        "2. Selected Gulf-area region",
        f"Houston / Galveston / Texas City synthetic service area around `{metadata['selected_substation_id']}`.",
        "",
        "3. Why this region was selected",
        "It provides a mapped 69-kV delivery point, strong coastal relevance, multiple feeders, and enough local load to support a meaningful data-center anchor scenario.",
        "",
        "4. Whether it is coastal or near-coastal",
        f"Near-coastal; selected substation is about {float(selected['distance_to_coast_km']):.1f} km from the simplified Texas Gulf coastline reference.",
        "",
        "5. Number of transmission nodes",
        str(int((nodes_out["layer"] == "transmission").sum())),
        "",
        "6. Number of substation/interface nodes",
        str(int((nodes_out["layer"] == "interface").sum())),
        "",
        "7. Number of distribution nodes",
        str(int((nodes_out["layer"] == "distribution").sum())),
        "",
        "8. Number of edges",
        str(len(edges_df)),
        "",
        "9. Number of feeders",
        str(int(feeder_rows["feeder_id"].nunique())),
        "",
        "10. Total peak load MW",
        f"{load_ids_df.loc[~load_ids_df['is_data_center'], 'peak_p_kw'].sum() / 1000.0:.3f}",
        "",
        "11. Total average load MW",
        f"{load_ids_df.loc[~load_ids_df['is_data_center'], 'average_p_kw'].sum() / 1000.0:.3f}",
        "",
        "12. Number of residential/commercial/critical/unknown loads",
        f"{counts_by_type}",
        "",
        "13. Data center location and connection voltage",
        f"`DC_1` at ({dc['latitude']}, {dc['longitude']}) connected to `{metadata['selected_transmission_node_id']}` at {dc['connection_voltage_kv']} kV.",
        "",
        "14. Data center size and assumptions",
        f"{dc['facility_load_mw']} MW facility load, PUE {dc['pue']}, critical IT fraction {dc['critical_it_fraction']}, UPS {dc['ups_energy_mwh']} MWh, backup generation {dc['backup_gen_mw']} MW.",
        "",
        "15. Load aggregation method",
        "Protected substation/interface buses, MV switch locations, 69-kV transformer interface buses, and high-load nodes were preserved; radial non-critical components were aggregated into supernodes, and raw de-energized islands with no normal-status source path were dropped from the clean baseline case.",
        "",
        "16. Load conservation check",
        f"P error = {conservation_p:.4%}, Q error = {conservation_q:.4%}, measured against the retained source-connected raw distribution footprint.",
        "",
        "17. Coordinate coverage",
        f"{coordinate_coverage:.2%} of final nodes carry latitude/longitude values.",
        "",
        "18. Power-flow smoke test results if available",
        (paths.parsed_distribution_dir / "opendss_powerflow_smoke.json").read_text(
            encoding="utf-8"
        ),
        "",
        "19. Known limitations",
        "The accessible BetterGrids transmission archive exposed the official PowerWorld case plus mapping workbook but not a directly parseable bus-branch export, so the local transmission layer is represented as an equivalent service-area source plus mapped 69-kV delivery buses. Low-voltage service transformers and some internal switch detail remain partially aggregated to keep the case under the current size cap.",
        "",
        "20. Files produced",
        "nodes.csv, edges.csv, loads.csv, load_timeseries.parquet, data_centers.csv, dc_specs.json, configs/testbed/base.yaml, network.geojson, plots, and validation reports.",
        "",
        "21. How this case will later support restoration RL",
        "Nodes include load, VoLL, voltage, location, layer, and criticality features; edges include status, switchability, repairability, length, and equipment type; the data center includes survivability/support attributes; geography supports hurricane damage modeling; and load time series support time-evolving restoration episodes.",
        "",
    ]
    (paths.final_case_dir / "selected_case_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    logger.info("Final case tables and reports written under %s", paths.final_case_dir)


if __name__ == "__main__":
    main()

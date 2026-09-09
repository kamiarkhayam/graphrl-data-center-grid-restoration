from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .io_utils import apply_dc_overlay, selected_dc_scenario_dirs, strip_base_data_centers


def build_normal_state_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for _, node in nodes_df.iterrows():
        graph.add_node(node["node_id"], **node.to_dict())
    closed_edges = edges_df.copy()
    if "initial_status" in closed_edges.columns:
        closed_edges = closed_edges.loc[
            closed_edges["initial_status"].fillna("closed").str.lower() != "open"
        ]
    elif "normal_status" in closed_edges.columns:
        closed_edges = closed_edges.loc[
            closed_edges["normal_status"].fillna("closed").str.lower() != "open"
        ]
    for _, edge in closed_edges.iterrows():
        graph.add_edge(edge["from_node"], edge["to_node"], key=edge["edge_id"], **edge.to_dict())
    return graph


def source_nodes(nodes_df: pd.DataFrame) -> list[str]:
    sources = (
        nodes_df.loc[
            nodes_df["node_type"].astype(str).str.lower().eq("source")
            | (
                nodes_df["is_transmission"].fillna(False).astype(bool)
                & nodes_df["node_type"].astype(str).str.lower().str.contains("source")
            ),
            "node_id",
        ]
        .astype(str)
        .tolist()
    )
    if not sources:
        sources = (
            nodes_df.loc[nodes_df["layer"].astype(str).str.lower().eq("transmission"), "node_id"]
            .astype(str)
            .tolist()[:1]
        )
    return sources


def apply_damage(
    graph: nx.MultiGraph,
    damaged_edge_ids: set[str],
    damaged_node_ids: set[str],
) -> nx.MultiGraph:
    damaged_graph = graph.copy()
    for node_id in damaged_node_ids:
        if damaged_graph.has_node(node_id):
            damaged_graph.remove_node(node_id)
    for u, v, key in list(damaged_graph.edges(keys=True)):
        if key in damaged_edge_ids and damaged_graph.has_edge(u, v, key=key):
            damaged_graph.remove_edge(u, v, key=key)
    return damaged_graph


def compute_source_connected_set(graph: nx.MultiGraph, source_node_ids: list[str]) -> set[str]:
    connected: set[str] = set()
    for source in source_node_ids:
        if graph.has_node(source):
            connected.update(nx.node_connected_component(graph, source))
    return connected


def compute_outage_for_scenario(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    realized_damage_df: pd.DataFrame,
    dc_scenario_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    graph = build_normal_state_graph(nodes_df, edges_df)
    damaged_edges = set(
        realized_damage_df.loc[
            (realized_damage_df["component_kind"] == "edge")
            & realized_damage_df["damaged"].astype(bool),
            "component_id",
        ].astype(str)
    )
    damaged_nodes = set(
        realized_damage_df.loc[
            (realized_damage_df["component_kind"] == "node")
            & realized_damage_df["damaged"].astype(bool),
            "component_id",
        ].astype(str)
    )
    damaged_graph = apply_damage(graph, damaged_edges, damaged_nodes)
    sources = source_nodes(nodes_df)
    connected_nodes = compute_source_connected_set(damaged_graph, sources)
    components = (
        list(nx.connected_components(damaged_graph)) if damaged_graph.number_of_nodes() else []
    )
    largest_component_size = max((len(component) for component in components), default=0)

    node_rows: list[dict[str, Any]] = []
    for _, node in nodes_df.iterrows():
        node_id = str(node["node_id"])
        is_load = bool(node.get("is_load", False))
        is_source_connected = node_id in connected_nodes
        is_served = bool(is_load and is_source_connected)
        baseline_p_kw = float(node.get("baseline_p_kw", 0.0) or 0.0)
        voll = float(node.get("voll_usd_per_kwh", 0.0) or 0.0)
        initial_unserved_kw = 0.0 if is_served else baseline_p_kw if is_load else 0.0
        node_rows.append(
            {
                "scenario_id": realized_damage_df["scenario_id"].iloc[0],
                "storm_id": realized_damage_df["storm_id"].iloc[0],
                "monte_carlo_id": int(realized_damage_df["monte_carlo_id"].iloc[0]),
                "dc_scenario_name": dc_scenario_name,
                "node_id": node_id,
                "is_source_connected": bool(is_source_connected),
                "is_served_initially": bool(is_served),
                "is_load": is_load,
                "is_critical": bool(node.get("is_critical", False)),
                "is_data_center": bool(node.get("is_data_center", False)),
                "load_type": node.get("load_type"),
                "baseline_p_kw": baseline_p_kw,
                "voll_usd_per_kwh": voll,
                "initial_unserved_kw": initial_unserved_kw,
                "initial_unserved_cost_per_hour": initial_unserved_kw * voll,
            }
        )
    state_df = pd.DataFrame(node_rows)

    load_rows = state_df.loc[state_df["is_load"].astype(bool)].copy()
    dc_rows = load_rows.loc[load_rows["is_data_center"].astype(bool)]
    community_rows = load_rows.loc[~load_rows["is_data_center"].astype(bool)]
    critical_rows = community_rows.loc[community_rows["is_critical"].astype(bool)]
    total_load_kw = float(load_rows["baseline_p_kw"].sum())
    served_load_kw = float(
        load_rows["baseline_p_kw"].sum() - load_rows["initial_unserved_kw"].sum()
    )

    summary = {
        "scenario_id": realized_damage_df["scenario_id"].iloc[0],
        "storm_id": realized_damage_df["storm_id"].iloc[0],
        "monte_carlo_id": int(realized_damage_df["monte_carlo_id"].iloc[0]),
        "dc_scenario_name": dc_scenario_name,
        "damaged_edge_count": int(len(damaged_edges)),
        "damaged_node_count": int(len(damaged_nodes)),
        "unserved_load_count": int((load_rows["initial_unserved_kw"] > 0).sum()),
        "unserved_kw": float(load_rows["initial_unserved_kw"].sum()),
        "unserved_mw": float(load_rows["initial_unserved_kw"].sum() / 1000.0),
        "voll_weighted_unserved_cost_per_hour": float(
            load_rows["initial_unserved_cost_per_hour"].sum()
        ),
        "critical_unserved_kw": float(critical_rows["initial_unserved_kw"].sum()),
        "data_center_served_initially": bool((dc_rows["initial_unserved_kw"] <= 0).all())
        if not dc_rows.empty
        else True,
        "data_center_unserved_kw": float(dc_rows["initial_unserved_kw"].sum()),
        "number_of_components": int(len(components)),
        "largest_component_size": int(largest_component_size),
        "source_connected_load_fraction": float(served_load_kw / total_load_kw)
        if total_load_kw > 0
        else 1.0,
        "total_dc_mw": float(dc_rows["baseline_p_kw"].sum() / 1000.0),
        "number_of_data_centers": int(dc_rows["node_id"].nunique()),
        "dc_served_count_initially": int((dc_rows["initial_unserved_kw"] <= 0).sum()),
        "dc_unserved_count_initially": int((dc_rows["initial_unserved_kw"] > 0).sum()),
        "dc_unserved_mw_initially": float(dc_rows["initial_unserved_kw"].sum() / 1000.0),
        "community_unserved_mw_initially": float(
            community_rows["initial_unserved_kw"].sum() / 1000.0
        ),
        "critical_unserved_mw_initially": float(
            critical_rows["initial_unserved_kw"].sum() / 1000.0
        ),
    }
    return state_df, summary


def evaluate_all_dc_scenarios(
    config: dict[str, Any],
    realized_damage_df: pd.DataFrame,
    base_nodes: pd.DataFrame,
    base_edges: pd.DataFrame,
    base_loads: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dc_dirs = selected_dc_scenario_dirs(config)
    clean_nodes, clean_edges, clean_loads = strip_base_data_centers(
        base_nodes, base_edges, base_loads
    )
    all_states: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []

    for scenario_name, scenario_dir in dc_dirs.items():
        nodes_df, edges_df, loads_df, _ = apply_dc_overlay(
            clean_nodes, clean_edges, clean_loads, scenario_dir
        )
        _ = loads_df  # reserved for future consistency checks
        for (_, _, mc_id), group_df in realized_damage_df.groupby(
            ["scenario_id", "storm_id", "monte_carlo_id"], sort=True
        ):
            state_df, summary = compute_outage_for_scenario(
                nodes_df, edges_df, group_df.reset_index(drop=True), scenario_name
            )
            all_states.append(state_df)
            all_summaries.append(summary)

    return pd.concat(all_states, ignore_index=True), pd.DataFrame(all_summaries)

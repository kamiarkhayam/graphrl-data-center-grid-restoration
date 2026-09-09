from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

from dc_restoration.paths import repository_root

SCRIPT_DIR = Path(__file__).resolve().parent

from dc_restoration.testbed.utils.project_utils import (  # noqa: E402
    default_arg_parser,
    get_paths,
    load_config,
    setup_logging,
)

SMALL_LEAF_LOAD_KW = 50.0
HIGH_LOAD_QUANTILE = 0.95
MV_BRANCH_BASE_KV = 7.2


def json_list(values) -> str:
    return json.dumps(sorted(set(v for v in values if pd.notna(v))))


def dominant_class(load_subset: pd.DataFrame) -> str:
    if load_subset.empty:
        return "unknown"
    by_class = (
        load_subset.groupby("inferred_customer_class")["p_kw"].sum().sort_values(ascending=False)
    )
    if by_class.empty:
        return "unknown"
    return str(by_class.index[0])


def convert_length_to_km(length: float | None, unit: str | None) -> float:
    if length is None or pd.isna(length):
        return 0.0
    unit = (unit or "km").strip().lower()
    factors = {
        "km": 1.0,
        "kft": 0.3048,
        "m": 0.001,
        "mi": 1.60934,
        "ft": 0.0003048,
    }
    return float(length) * factors.get(unit, 1.0)


def main() -> None:
    parser = default_arg_parser(
        "Aggregate the selected distribution graph to a restoration-scale case."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("07_aggregate_distribution_graph", config)
    paths = get_paths(config)

    buses = pd.read_csv(paths.parsed_distribution_dir / "raw_buses.csv")
    lines = pd.read_csv(paths.parsed_distribution_dir / "raw_lines.csv")
    transformers = pd.read_csv(paths.parsed_distribution_dir / "raw_transformers.csv")
    loads = pd.read_csv(paths.parsed_distribution_dir / "raw_loads.csv")
    metadata = json.loads(
        (paths.selected_region_dir / "selected_subnetwork_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    substation_id = metadata["selected_substation_id"]
    root_bus = metadata["selected_substation_root_bus"]
    root_prefix = f"sb6_{substation_id}"

    bus_df = buses.set_index("raw_bus_id")
    load_by_bus = loads.groupby("bus", as_index=True).agg(
        baseline_p_kw=("p_kw", "sum"),
        baseline_q_kvar=("q_kvar", "sum"),
    )
    load_quantile = (
        float(load_by_bus["baseline_p_kw"].quantile(HIGH_LOAD_QUANTILE))
        if not load_by_bus.empty
        else 0.0
    )

    G = nx.Graph()
    for bus_id, row in bus_df.iterrows():
        G.add_node(
            bus_id,
            base_kv=row["base_kv"],
            feeder_id=row["feeder_id"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            substation_id=row["substation_id"],
        )

    edge_attrs: dict[tuple[str, str, str], dict] = {}

    for _, row in lines.iterrows():
        u = row["from_bus"]
        v = row["to_bus"]
        edge_key = ("line", str(row["raw_edge_id"]), "")
        attrs = {
            "raw_edge_id": str(row["raw_edge_id"]),
            "edge_type": "switch" if bool(row["is_switch"]) else "line",
            "length_km": convert_length_to_km(row["length"], row["length_unit"]),
            "r": 0.0
            if pd.isna(row["r"])
            else float(row["r"]) * convert_length_to_km(row["length"], row["length_unit"]),
            "x": 0.0
            if pd.isna(row["x"])
            else float(row["x"]) * convert_length_to_km(row["length"], row["length_unit"]),
            "is_switch": bool(row["is_switch"]),
            "is_transformer": False,
            "normal_status": row["normal_status"],
            "switchable": bool(row["is_switch"]),
            "feeder_id": row["feeder_id"],
            "voltage_kv": None,
            "is_overhead": bool(row["is_overhead"]) if pd.notna(row["is_overhead"]) else False,
            "is_underground": bool(row["is_underground"])
            if pd.notna(row["is_underground"])
            else False,
        }
        G.add_edge(u, v, key=edge_key, **attrs)
        edge_attrs[(u, v, attrs["raw_edge_id"])] = attrs

    for _, row in transformers.iterrows():
        u = row["bus1"]
        v = row["bus2"]
        if pd.isna(u) or pd.isna(v):
            continue
        kv1 = bus_df.loc[u, "base_kv"] if u in bus_df.index else None
        kva = float(row["kva"]) if pd.notna(row["kva"]) else None
        zbase = ((kv1 or 12.47) ** 2) / (kva / 1000.0) if kva and kva > 0 else 0.0
        percent_r = 0.0033831264
        percent_x = (float(row["xhl"]) / 100.0) if pd.notna(row["xhl"]) else 0.0
        attrs = {
            "raw_edge_id": str(row["raw_transformer_id"]),
            "edge_type": "transformer",
            "length_km": 0.0,
            "r": zbase * percent_r,
            "x": zbase * percent_x,
            "is_switch": False,
            "is_transformer": True,
            "normal_status": "closed",
            "switchable": False,
            "feeder_id": row["feeder_id"],
            "voltage_kv": max(kv1 or 0.0, bus_df.loc[v, "base_kv"] if v in bus_df.index else 0.0),
            "is_overhead": False,
            "is_underground": False,
        }
        G.add_edge(u, v, key=("xf", attrs["raw_edge_id"], ""), **attrs)

    protected = set()
    for node, data in G.nodes(data=True):
        kv = data.get("base_kv")
        if node == root_bus or str(node).startswith(root_prefix):
            protected.add(node)
        if G.degree(node) >= 3 and pd.notna(kv) and float(kv) >= MV_BRANCH_BASE_KV:
            protected.add(node)
    for node, row in load_by_bus.iterrows():
        if row["baseline_p_kw"] >= load_quantile:
            protected.add(node)
    for u, v, data in G.edges(data=True):
        if data["is_switch"]:
            # Preserve all explicit switch locations, including normally open ties,
            # so later restoration studies retain actionable switching structure.
            protected.add(u)
            protected.add(v)
        if data["is_transformer"]:
            ku = bus_df.loc[u, "base_kv"] if u in bus_df.index else None
            kv = bus_df.loc[v, "base_kv"] if v in bus_df.index else None
            if pd.notna(ku) and pd.notna(kv) and max(float(ku), float(kv)) >= 69.0:
                protected.add(u)
                protected.add(v)

    unprotected_nodes = [node for node in G.nodes if node not in protected]
    U = G.subgraph(unprotected_nodes).copy()

    boundary_merge_bus_members: dict[str, list[str]] = defaultdict(list)
    boundary_merge_load_ids: dict[str, list[str]] = defaultdict(list)
    boundary_merge_p: Counter = Counter()
    boundary_merge_q: Counter = Counter()

    agg_nodes = []
    agg_edges = []
    component_node_lookup: dict[str, str] = {}

    agg_counter = 0
    for component in nx.connected_components(U):
        comp_nodes = set(component)
        boundary_nodes = set()
        component_edges = set()
        component_has_transformer = False
        total_p = 0.0
        total_q = 0.0
        comp_loads = loads.loc[loads["bus"].isin(comp_nodes)].copy()
        total_p = float(comp_loads["p_kw"].sum()) if not comp_loads.empty else 0.0
        total_q = float(comp_loads["q_kvar"].sum()) if not comp_loads.empty else 0.0

        for node in comp_nodes:
            for nbr in G.neighbors(node):
                edge_data = G.get_edge_data(node, nbr)
                if nbr in protected:
                    boundary_nodes.add(nbr)
                if edge_data is None:
                    continue
                if edge_data.get("is_transformer"):
                    component_has_transformer = True
                component_edges.add(edge_data["raw_edge_id"])

        if (
            len(boundary_nodes) == 1
            and total_p < SMALL_LEAF_LOAD_KW
            and len(comp_nodes) <= 2
            and not component_has_transformer
        ):
            boundary = next(iter(boundary_nodes))
            boundary_merge_bus_members[boundary].extend(comp_nodes)
            boundary_merge_load_ids[boundary].extend(comp_loads["raw_load_id"].tolist())
            boundary_merge_p[boundary] += total_p
            boundary_merge_q[boundary] += total_q
            continue

        if len(boundary_nodes) == 2 and total_p == 0.0 and total_q == 0.0:
            boundary_list = sorted(boundary_nodes)
            agg_edges.append(
                {
                    "agg_edge_id": f"AGGE_{len(agg_edges) + 1:04d}",
                    "from_node": boundary_list[0],
                    "to_node": boundary_list[1],
                    "member_raw_edges": json.dumps(sorted(component_edges)),
                    "member_raw_buses": json.dumps(sorted(comp_nodes)),
                    "edge_type": "aggregated_corridor",
                }
            )
            continue

        agg_counter += 1
        agg_node_id = f"AGG_{agg_counter:04d}"
        for node in comp_nodes:
            component_node_lookup[node] = agg_node_id

        coords = bus_df.loc[list(comp_nodes), ["latitude", "longitude"]].dropna()
        latitude = float(coords["latitude"].mean()) if not coords.empty else None
        longitude = float(coords["longitude"].mean()) if not coords.empty else None
        agg_nodes.append(
            {
                "agg_node_id": agg_node_id,
                "boundary_node": json.dumps(sorted(boundary_nodes)),
                "feeder_id": Counter(bus_df.loc[list(comp_nodes), "feeder_id"]).most_common(1)[0][
                    0
                ],
                "member_raw_buses": json.dumps(sorted(comp_nodes)),
                "aggregated_raw_node_count": len(comp_nodes),
                "baseline_p_kw": total_p,
                "baseline_q_kvar": total_q,
                "dominant_customer_class": dominant_class(comp_loads),
                "latitude": latitude,
                "longitude": longitude,
                "base_kv": float(
                    pd.Series(bus_df.loc[list(comp_nodes), "base_kv"]).dropna().mode().iloc[0]
                )
                if not pd.Series(bus_df.loc[list(comp_nodes), "base_kv"]).dropna().empty
                else None,
                "member_raw_load_ids": json.dumps(sorted(comp_loads["raw_load_id"].tolist())),
                "node_type": "aggregated_radial_node"
                if len(boundary_nodes) == 1
                else "aggregated_complex_node",
            }
        )
        for boundary in sorted(boundary_nodes):
            agg_edges.append(
                {
                    "agg_edge_id": f"AGGE_{len(agg_edges) + 1:04d}",
                    "from_node": boundary,
                    "to_node": agg_node_id,
                    "member_raw_edges": json.dumps(sorted(component_edges)),
                    "member_raw_buses": json.dumps(sorted(comp_nodes)),
                    "edge_type": "aggregated_radial"
                    if len(boundary_nodes) == 1
                    else "aggregated_complex",
                }
            )

    final_protected_nodes = [node for node in protected if node not in component_node_lookup]

    node_rows = []
    substation_cluster = [
        node
        for node in final_protected_nodes
        if node == root_bus or str(node).startswith(root_prefix)
    ]
    sub_coords = bus_df.loc[substation_cluster, ["latitude", "longitude"]].dropna()
    sub_loads = loads.loc[loads["bus"].isin(substation_cluster)]
    node_rows.append(
        {
            "agg_node_id": f"SUB_{substation_id}",
            "node_type": "substation_interface",
            "feeder_id": substation_id,
            "member_raw_buses": json.dumps(sorted(substation_cluster)),
            "aggregated_raw_node_count": len(substation_cluster),
            "baseline_p_kw": float(sub_loads["p_kw"].sum()) + boundary_merge_p.get(root_bus, 0.0),
            "baseline_q_kvar": float(sub_loads["q_kvar"].sum())
            + boundary_merge_q.get(root_bus, 0.0),
            "dominant_customer_class": dominant_class(sub_loads),
            "latitude": float(sub_coords["latitude"].mean())
            if not sub_coords.empty
            else metadata["selected_latitude"],
            "longitude": float(sub_coords["longitude"].mean())
            if not sub_coords.empty
            else metadata["selected_longitude"],
            "base_kv": 12.47,
            "member_raw_load_ids": json.dumps(
                sorted(
                    sub_loads["raw_load_id"].tolist() + boundary_merge_load_ids.get(root_bus, [])
                )
            ),
        }
    )

    for node in sorted(set(final_protected_nodes) - set(substation_cluster)):
        node_loads = loads.loc[loads["bus"].eq(node)]
        node_rows.append(
            {
                "agg_node_id": node,
                "node_type": "protected_raw_bus",
                "feeder_id": bus_df.loc[node, "feeder_id"],
                "member_raw_buses": json.dumps(
                    sorted([node] + boundary_merge_bus_members.get(node, []))
                ),
                "aggregated_raw_node_count": 1 + len(boundary_merge_bus_members.get(node, [])),
                "baseline_p_kw": float(node_loads["p_kw"].sum()) + boundary_merge_p.get(node, 0.0),
                "baseline_q_kvar": float(node_loads["q_kvar"].sum())
                + boundary_merge_q.get(node, 0.0),
                "dominant_customer_class": dominant_class(
                    pd.concat(
                        [
                            node_loads,
                            loads.loc[
                                loads["raw_load_id"].isin(boundary_merge_load_ids.get(node, []))
                            ],
                        ]
                    )
                ),
                "latitude": bus_df.loc[node, "latitude"],
                "longitude": bus_df.loc[node, "longitude"],
                "base_kv": bus_df.loc[node, "base_kv"],
                "member_raw_load_ids": json.dumps(
                    sorted(
                        node_loads["raw_load_id"].tolist() + boundary_merge_load_ids.get(node, [])
                    )
                ),
            }
        )

    node_rows.extend(agg_nodes)
    agg_nodes_df = (
        pd.DataFrame(node_rows).drop_duplicates(subset=["agg_node_id"]).reset_index(drop=True)
    )

    raw_to_agg: dict[str, str] = {}
    for _, row in agg_nodes_df.iterrows():
        for raw_bus in json.loads(row["member_raw_buses"]):
            raw_to_agg[raw_bus] = row["agg_node_id"]

    concrete_edges = []
    for u, v, data in G.edges(data=True):
        from_node = raw_to_agg.get(u, u)
        to_node = raw_to_agg.get(v, v)
        if u in substation_cluster:
            from_node = f"SUB_{substation_id}"
        if v in substation_cluster:
            to_node = f"SUB_{substation_id}"
        if from_node == to_node:
            continue
        concrete_edges.append(
            {
                "from_node": from_node,
                "to_node": to_node,
                "raw_edge_id": data["raw_edge_id"],
                "edge_type": data["edge_type"],
                "length_km": data["length_km"],
                "r": data["r"],
                "x": data["x"],
                "is_switch": data["is_switch"],
                "is_transformer": data["is_transformer"],
                "normal_status": data["normal_status"],
                "switchable": data["switchable"],
                "feeder_id": data["feeder_id"],
                "is_overhead": data["is_overhead"],
                "is_underground": data["is_underground"],
            }
        )
    concrete_edges_df = pd.DataFrame(concrete_edges)

    grouped_edges = concrete_edges_df.groupby(
        ["from_node", "to_node", "edge_type"], as_index=False
    ).agg(
        member_raw_edges=("raw_edge_id", lambda x: json.dumps(sorted(set(map(str, x))))),
        length_km=("length_km", "sum"),
        r=("r", "sum"),
        x=("x", "sum"),
        is_switch=("is_switch", "max"),
        is_transformer=("is_transformer", "max"),
        normal_status=("normal_status", lambda x: "open" if "open" in set(x) else "closed"),
        switchable=("switchable", "max"),
        feeder_id=("feeder_id", lambda x: Counter(map(str, x)).most_common(1)[0][0]),
        is_overhead=("is_overhead", "max"),
        is_underground=("is_underground", "max"),
    )
    grouped_edges["agg_edge_id"] = [f"EDGE_{i:04d}" for i in range(1, len(grouped_edges) + 1)]

    corridor_edges_df = pd.DataFrame(agg_edges)
    if not corridor_edges_df.empty:
        corridor_edges_df["length_km"] = 0.0
        corridor_edges_df["r"] = 0.0
        corridor_edges_df["x"] = 0.0
        corridor_edges_df["is_switch"] = False
        corridor_edges_df["is_transformer"] = False
        corridor_edges_df["normal_status"] = "closed"
        corridor_edges_df["switchable"] = False
        corridor_edges_df["feeder_id"] = substation_id
        corridor_edges_df["is_overhead"] = False
        corridor_edges_df["is_underground"] = False
        corridor_edges_df = corridor_edges_df.rename(columns={"agg_edge_id": "agg_edge_id"})
        corridor_edges_df = corridor_edges_df[
            [
                "agg_edge_id",
                "from_node",
                "to_node",
                "edge_type",
                "member_raw_edges",
                "length_km",
                "r",
                "x",
                "is_switch",
                "is_transformer",
                "normal_status",
                "switchable",
                "feeder_id",
                "is_overhead",
                "is_underground",
            ]
        ]
    grouped_edges = grouped_edges.rename(columns={"r": "equivalent_r", "x": "equivalent_x"})
    grouped_edges["member_raw_buses"] = ""
    grouped_edges = grouped_edges.rename(columns={"agg_edge_id": "edge_id"})
    if not corridor_edges_df.empty:
        corridor_edges_df = corridor_edges_df.rename(
            columns={"agg_edge_id": "edge_id", "r": "equivalent_r", "x": "equivalent_x"}
        )
        edge_output = pd.concat([grouped_edges, corridor_edges_df], ignore_index=True, sort=False)
    else:
        edge_output = grouped_edges.copy()

    agg_nodes_df.to_csv(
        paths.selected_region_dir / "aggregated_distribution_nodes.csv", index=False
    )
    edge_output.to_csv(paths.selected_region_dir / "aggregated_distribution_edges.csv", index=False)

    logger.info(
        "Aggregation produced %s nodes and %s edges from %s raw buses and %s raw lines.",
        len(agg_nodes_df),
        len(edge_output),
        len(buses),
        len(lines),
    )
    summary = {
        "raw_bus_count": int(len(buses)),
        "raw_line_count": int(len(lines)),
        "raw_transformer_count": int(len(transformers)),
        "aggregated_node_count": int(len(agg_nodes_df)),
        "aggregated_edge_count": int(len(edge_output)),
        "load_quantile_kw": load_quantile,
        "small_leaf_merge_threshold_kw": SMALL_LEAF_LOAD_KW,
    }
    (paths.selected_region_dir / "aggregation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

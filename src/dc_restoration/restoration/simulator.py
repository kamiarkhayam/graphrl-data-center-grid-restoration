from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx
import pandas as pd
import yaml

from dc_restoration.paths import repository_root


def _write_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    use = df.head(max_rows).copy()
    try:
        return use.to_markdown(index=False)
    except Exception:
        return "```\n" + use.to_string(index=False) + "\n```"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _resolve_path(root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (root / p).resolve()


@dataclass
class OutageMetrics:
    unserved_mw: float
    critical_unserved_mw: float
    voll_cost: float
    island_count: int
    unserved_load_ids: List[str]


class RestorationSimulator:
    def __init__(
        self, nodes: pd.DataFrame, edges: pd.DataFrame, loads: pd.DataFrame, damage: pd.DataFrame
    ):
        self.nodes = nodes
        self.edges = edges
        self.loads = loads.copy()
        self.damage = damage.copy()
        self.base_graph = nx.Graph()
        self.base_graph.add_nodes_from(nodes["node_id"].astype(str))
        self.edge_endpoints = {
            str(r.edge_id): (str(r.from_node), str(r.to_node))
            for r in edges[["edge_id", "from_node", "to_node"]].itertuples(index=False)
        }
        self.edge_attrs = edges.set_index("edge_id").to_dict("index")
        self.base_graph.add_edges_from(
            (u, v, {"edge_id": eid}) for eid, (u, v) in self.edge_endpoints.items()
        )
        self.source_node = "T_SUPER_SOURCE"
        self.loads["is_critical"] = (
            _safe_bool(self.loads["is_critical"]) if "is_critical" in self.loads else False
        )
        self.load_by_node = self.loads.groupby("node_id").agg(
            baseline_p_kw=("baseline_p_kw", "sum"),
            critical_p_kw=("baseline_p_kw", lambda s: 0.0),
        )
        crit_kw_by_node = (
            self.loads.loc[self.loads["is_critical"]].groupby("node_id")["baseline_p_kw"].sum()
        )
        self.load_by_node["critical_p_kw"] = crit_kw_by_node.reindex(
            self.load_by_node.index
        ).fillna(0.0)
        self.voll_by_load = self.loads.set_index("load_id")[
            ["node_id", "baseline_p_kw", "voll_usd_per_kwh", "is_critical"]
        ]

    def metrics(self, damaged_unrepaired: Iterable[str]) -> OutageMetrics:
        damaged_set = set(map(str, damaged_unrepaired))
        g = self.base_graph.copy()
        for eid in damaged_set:
            uv = self.edge_endpoints.get(eid)
            if uv and g.has_edge(*uv):
                g.remove_edge(*uv)
        served = (
            nx.node_connected_component(g, self.source_node) if self.source_node in g else set()
        )
        unserved = self.loads.loc[~self.loads["node_id"].astype(str).isin(served)].copy()
        unserved_mw = float(unserved["baseline_p_kw"].sum() / 1000.0)
        crit_unserved_mw = float(
            unserved.loc[_safe_bool(unserved["is_critical"]), "baseline_p_kw"].sum() / 1000.0
        )
        voll = float((unserved["baseline_p_kw"] * unserved["voll_usd_per_kwh"]).sum())
        island_count = 0
        if not unserved.empty:
            for comp in nx.connected_components(g):
                if self.source_node in comp:
                    continue
                if any(n in set(unserved["node_id"].astype(str)) for n in comp):
                    island_count += 1
        return OutageMetrics(
            unserved_mw,
            crit_unserved_mw,
            voll,
            island_count,
            unserved["load_id"].astype(str).tolist(),
        )

    def edge_component_type(self, edge_id: str) -> str:
        attrs = self.edge_attrs.get(edge_id, {})
        return str(attrs.get("hurricane_damage_component_type", attrs.get("edge_type", "unknown")))

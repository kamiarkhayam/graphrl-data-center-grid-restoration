from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dc_restoration.paths import repository_root


def module_root() -> Path:
    return repository_root()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else module_root() / "configs/hazards/hurricane.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_markdown(path: str | Path, text: str) -> None:
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_base_network(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    base = Path(config["network_case_path"])
    return {
        "nodes": pd.read_csv(base / "nodes.csv"),
        "edges": pd.read_csv(base / "edges.csv"),
        "loads": pd.read_csv(base / "loads.csv"),
        "data_centers": pd.read_csv(base / "data_centers.csv"),
    }


def strip_base_data_centers(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    loads_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_nodes = nodes_df.loc[~nodes_df["is_data_center"].fillna(False)].copy()
    base_edges = edges_df.loc[
        ~(edges_df["edge_role"].fillna("") == "data_center_connection")
        & ~edges_df["from_node"].astype(str).str.startswith("DC_")
        & ~edges_df["to_node"].astype(str).str.startswith("DC_")
    ].copy()
    base_loads = loads_df.loc[~loads_df["is_data_center"].fillna(False)].copy()
    return base_nodes, base_edges, base_loads


def apply_dc_overlay(
    base_nodes: pd.DataFrame,
    base_edges: pd.DataFrame,
    base_loads: pd.DataFrame,
    scenario_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_dir = Path(scenario_dir)
    overlay_nodes = pd.read_csv(scenario_dir / "nodes_overlay.csv")
    overlay_edges = pd.read_csv(scenario_dir / "edges_overlay.csv")
    overlay_loads = pd.read_csv(scenario_dir / "loads_overlay.csv")
    overlay_dcs = pd.read_csv(scenario_dir / "data_centers.csv")
    nodes = pd.concat([base_nodes, overlay_nodes], ignore_index=True)
    edges = pd.concat([base_edges, overlay_edges], ignore_index=True)
    loads = pd.concat([base_loads, overlay_loads], ignore_index=True)
    return nodes, edges, loads, overlay_dcs


def selected_dc_scenario_dirs(config: dict[str, Any]) -> dict[str, Path]:
    base = Path(config["network_case_path"]) / "dc_scenarios"
    return {name: base / name for name in config["selected_dc_scenarios"]}

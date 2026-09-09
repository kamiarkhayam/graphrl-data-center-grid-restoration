from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .geo_utils import approximate_distance_km, midpoint_lonlat


def compute_max_wind_for_points(
    point_locations: np.ndarray,
    track_lonlat: np.ndarray,
    intensities_mps: np.ndarray,
    rmax_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact port of the vectorized Holland-wind evaluator from the old project.
    """
    distances = approximate_distance_km(point_locations, track_lonlat)
    intensities = intensities_mps.reshape(1, -1)
    rmax = rmax_km.reshape(1, -1)
    b_param = 1.05 * np.log(float(intensities_mps[0])) - 2.32
    term = np.where(distances < 1e-9, 1.0, (rmax / distances) ** b_param)
    winds = np.where(
        distances < 1e-9,
        np.tile(intensities, (point_locations.shape[0], 1)),
        intensities * term * np.exp(1.0 - term),
    )
    max_idx = np.argmax(winds, axis=1)
    max_wind = winds[np.arange(winds.shape[0]), max_idx]
    eye_locs = track_lonlat[max_idx]
    return max_wind, eye_locs


def _node_component_type(node_row: pd.Series) -> str:
    if (
        bool(node_row.get("is_substation", False))
        or str(node_row.get("node_type", "")).lower() == "substation_interface"
    ):
        return "substation"
    if bool(node_row.get("is_data_center", False)):
        return "data_center_facility"
    return "not_applicable"


def _node_damage_eligible(node_row: pd.Series) -> bool:
    return _node_component_type(node_row) == "substation"


def build_node_exposure_table(
    nodes_df: pd.DataFrame,
    sampled_tracks_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    node_points = nodes_df[["longitude", "latitude"]].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for _, storm in manifest_df.iterrows():
        storm_track = sampled_tracks_df.loc[
            sampled_tracks_df["scenario_id"] == storm["scenario_id"]
        ].copy()
        track = storm_track[["longitude", "latitude"]].to_numpy(dtype=float)
        intensities = storm_track["intensity_mps"].to_numpy(dtype=float)
        rmax = storm_track["rmax_km"].to_numpy(dtype=float)
        wind_speeds, eye_locs = compute_max_wind_for_points(node_points, track, intensities, rmax)
        for idx, node in nodes_df.iterrows():
            rows.append(
                {
                    "scenario_id": storm["scenario_id"],
                    "storm_id": storm["storm_id"],
                    "node_id": node["node_id"],
                    "node_type": node["node_type"],
                    "layer": node["layer"],
                    "latitude": float(node["latitude"]),
                    "longitude": float(node["longitude"]),
                    "wind_speed": float(wind_speeds[idx]),
                    "wind_speed_unit": "mps",
                    "exposure_metric": float(wind_speeds[idx]),
                    "damage_eligible": bool(_node_damage_eligible(node)),
                    "component_type": _node_component_type(node),
                    "eye_longitude": float(eye_locs[idx, 0]),
                    "eye_latitude": float(eye_locs[idx, 1]),
                }
            )
    return pd.DataFrame(rows)


def build_edge_exposure_table(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    sampled_tracks_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    node_lookup = nodes_df.set_index("node_id")[["longitude", "latitude"]]
    edge_base = edges_df.copy()
    edge_base["from_longitude"] = edge_base["from_node"].map(node_lookup["longitude"])
    edge_base["from_latitude"] = edge_base["from_node"].map(node_lookup["latitude"])
    edge_base["to_longitude"] = edge_base["to_node"].map(node_lookup["longitude"])
    edge_base["to_latitude"] = edge_base["to_node"].map(node_lookup["latitude"])
    edge_base["mid_longitude"] = (edge_base["from_longitude"] + edge_base["to_longitude"]) / 2.0
    edge_base["mid_latitude"] = (edge_base["from_latitude"] + edge_base["to_latitude"]) / 2.0
    midpoint_points = edge_base[["mid_longitude", "mid_latitude"]].to_numpy(dtype=float)
    endpoint_from = edge_base[["from_longitude", "from_latitude"]].to_numpy(dtype=float)
    endpoint_to = edge_base[["to_longitude", "to_latitude"]].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for _, storm in manifest_df.iterrows():
        storm_track = sampled_tracks_df.loc[
            sampled_tracks_df["scenario_id"] == storm["scenario_id"]
        ].copy()
        track = storm_track[["longitude", "latitude"]].to_numpy(dtype=float)
        intensities = storm_track["intensity_mps"].to_numpy(dtype=float)
        rmax = storm_track["rmax_km"].to_numpy(dtype=float)
        wind_mid, eye_mid = compute_max_wind_for_points(midpoint_points, track, intensities, rmax)
        wind_from, _ = compute_max_wind_for_points(endpoint_from, track, intensities, rmax)
        wind_to, _ = compute_max_wind_for_points(endpoint_to, track, intensities, rmax)
        wind_max_endpoint = np.maximum(wind_from, wind_to)
        for idx, edge in edge_base.iterrows():
            rows.append(
                {
                    "scenario_id": storm["scenario_id"],
                    "storm_id": storm["storm_id"],
                    "edge_id": edge["edge_id"],
                    "from_node": edge["from_node"],
                    "to_node": edge["to_node"],
                    "edge_role": edge["edge_role"],
                    "hurricane_damage_component_type": edge["hurricane_damage_component_type"],
                    "is_hurricane_damage_eligible": bool(edge["is_hurricane_damage_eligible"]),
                    "wind_speed_midpoint": float(wind_mid[idx]),
                    "wind_speed_max_endpoint": float(wind_max_endpoint[idx]),
                    "wind_speed_unit": "mps",
                    "exposure_metric": float(wind_mid[idx]),
                    "damage_eligible": bool(edge["is_hurricane_damage_eligible"]),
                    "eye_longitude": float(eye_mid[idx, 0]),
                    "eye_latitude": float(eye_mid[idx, 1]),
                    "from_longitude": float(edge["from_longitude"]),
                    "from_latitude": float(edge["from_latitude"]),
                    "to_longitude": float(edge["to_longitude"]),
                    "to_latitude": float(edge["to_latitude"]),
                }
            )
    return pd.DataFrame(rows)

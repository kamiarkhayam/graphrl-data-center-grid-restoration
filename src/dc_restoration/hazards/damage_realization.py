from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .fragility import (
    classify_edge_component,
    classify_node_component,
    conductor_failure_probability,
    pole_failure_probability_for_overhead_edges,
    sample_substation_states,
    substation_state_probabilities,
)
from .repair_times import sample_conductor_repair_times, sample_substation_repair_times


def build_component_damage_probabilities(
    node_exposure_df: pd.DataFrame,
    edge_exposure_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    enabled_types = set(config["damage_enabled_component_types"])
    pole_proxy = config["fallback_assumptions"]["overhead_distribution_pole_proxy"]
    node_lookup = nodes_df.set_index("node_id")
    edge_lookup = edges_df.set_index("edge_id")
    records: list[dict[str, Any]] = []

    for _, row in node_exposure_df.iterrows():
        base = node_lookup.loc[row["node_id"]]
        info = classify_node_component(base, enabled_types)
        damage_probability = 0.0
        if info["eligible"] and info["component_type"] == "substation":
            probs = substation_state_probabilities(np.array([row["wind_speed"]]))
            damage_probability = float(
                probs["prob_severe"].iloc[0] + probs["prob_complete"].iloc[0]
            )
        records.append(
            {
                "scenario_id": row["scenario_id"],
                "storm_id": row["storm_id"],
                "component_id": row["node_id"],
                "component_kind": "node",
                "component_type": info["component_type"],
                "wind_speed": float(row["wind_speed"]),
                "wind_speed_unit": row["wind_speed_unit"],
                "damage_probability": float(damage_probability),
                "fragility_model": info["fragility_model"],
                "eligible": bool(info["eligible"]),
                "notes": info["notes"],
            }
        )

    for _, row in edge_exposure_df.iterrows():
        base = edge_lookup.loc[row["edge_id"]]
        info = classify_edge_component(base, enabled_types)
        damage_probability = 0.0
        if info["eligible"] and info["component_type"] == "overhead_distribution_line":
            endpoints = np.array(
                [
                    [
                        row["from_longitude"],
                        row["from_latitude"],
                        row["to_longitude"],
                        row["to_latitude"],
                    ]
                ],
                dtype=float,
            )
            eye_locs = np.array([[row["eye_longitude"], row["eye_latitude"]]], dtype=float)
            span_length_m = float(base.get("length_km", 0.0) or 0.0) * 1000.0
            span_length_m = float(
                np.clip(
                    span_length_m, pole_proxy["span_length_m_min"], pole_proxy["span_length_m_max"]
                )
            )
            damage_probability = float(
                pole_failure_probability_for_overhead_edges(
                    np.array([row["wind_speed_midpoint"]], dtype=float),
                    eye_locs,
                    endpoints,
                    np.array([span_length_m], dtype=float),
                    pole_height_m=float(pole_proxy["pole_height_m"]),
                    pole_class=int(pole_proxy["ansi_class"]),
                )[0]
            )
        elif info["eligible"] and info["component_type"] == "transmission_line":
            endpoints = np.array(
                [
                    [
                        row["from_longitude"],
                        row["from_latitude"],
                        row["to_longitude"],
                        row["to_latitude"],
                    ]
                ],
                dtype=float,
            )
            eye_locs = np.array([[row["eye_longitude"], row["eye_latitude"]]], dtype=float)
            damage_probability = float(
                conductor_failure_probability(
                    np.array([row["wind_speed_midpoint"]], dtype=float),
                    eye_locs,
                    endpoints,
                )[0]
            )
        records.append(
            {
                "scenario_id": row["scenario_id"],
                "storm_id": row["storm_id"],
                "component_id": row["edge_id"],
                "component_kind": "edge",
                "component_type": info["component_type"],
                "wind_speed": float(row["wind_speed_midpoint"]),
                "wind_speed_unit": row["wind_speed_unit"],
                "damage_probability": float(damage_probability),
                "fragility_model": info["fragility_model"],
                "eligible": bool(info["eligible"]),
                "notes": info["notes"],
            }
        )

    return pd.DataFrame(records)


def realize_damage_scenarios(
    component_probabilities_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_mc = int(config["number_of_monte_carlo_realizations"])
    base_seed = int(config["random_seed"])
    realized_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    grouped = component_probabilities_df.groupby(["scenario_id", "storm_id"], sort=True)

    for group_index, ((scenario_id, storm_id), group_df) in enumerate(grouped, start=1):
        group_df = group_df.reset_index(drop=True)
        for monte_carlo_id in range(1, n_mc + 1):
            rng = np.random.default_rng(base_seed + group_index * 1000 + monte_carlo_id)
            severity = np.full(group_df.shape[0], "not_applicable", dtype=object)
            damaged = np.zeros(group_df.shape[0], dtype=bool)
            sampled_random_value = np.full(group_df.shape[0], np.nan, dtype=float)

            sub_mask = (
                (group_df["component_kind"] == "node")
                & (group_df["component_type"] == "substation")
                & group_df["eligible"].astype(bool)
            )
            if sub_mask.any():
                sub_idx = np.flatnonzero(sub_mask.to_numpy())
                sub_states = sample_substation_states(
                    group_df.loc[sub_mask, "wind_speed"].to_numpy(dtype=float), rng
                )
                severity[sub_idx] = sub_states
                damaged[sub_idx] = np.isin(sub_states, ["Severe", "Complete"])
                sampled_random_value[sub_idx] = rng.random(len(sub_idx))

            bin_mask = (
                group_df["eligible"].astype(bool)
                & ~sub_mask
                & group_df["component_type"].isin(
                    ["overhead_distribution_line", "transmission_line"]
                )
            )
            if bin_mask.any():
                bin_idx = np.flatnonzero(bin_mask.to_numpy())
                rand_vals = rng.random(len(bin_idx))
                probs = group_df.loc[bin_mask, "damage_probability"].to_numpy(dtype=float)
                failed = rand_vals < probs
                damaged[bin_idx] = failed
                severity[bin_idx] = np.where(failed, "failed", "none")
                sampled_random_value[bin_idx] = rand_vals

            sub_repair_times = sample_substation_repair_times(severity.tolist(), rng)
            line_repair_times = sample_conductor_repair_times(bin_mask.to_numpy() & damaged, rng)
            repair_times = np.zeros(group_df.shape[0], dtype=float)
            repair_times = np.maximum(repair_times, sub_repair_times)
            repair_times = np.maximum(repair_times, line_repair_times)

            for idx, row in group_df.iterrows():
                repair_model = "not_applicable"
                if row["component_type"] == "substation" and damaged[idx]:
                    repair_model = "legacy_substation_repair_time"
                elif (
                    row["component_type"] in {"overhead_distribution_line", "transmission_line"}
                    and damaged[idx]
                ):
                    repair_model = "legacy_conductor_repair_time"
                realized_record = {
                    "scenario_id": scenario_id,
                    "storm_id": storm_id,
                    "monte_carlo_id": monte_carlo_id,
                    "component_id": row["component_id"],
                    "component_kind": row["component_kind"],
                    "component_type": row["component_type"],
                    "damaged": bool(damaged[idx]),
                    "damage_probability": float(row["damage_probability"]),
                    "sampled_random_value": None
                    if np.isnan(sampled_random_value[idx])
                    else float(sampled_random_value[idx]),
                    "repair_time_hours": float(repair_times[idx]),
                    "severity_class": str(severity[idx]),
                }
                repair_record = {
                    "scenario_id": scenario_id,
                    "storm_id": storm_id,
                    "monte_carlo_id": monte_carlo_id,
                    "component_id": row["component_id"],
                    "component_type": row["component_type"],
                    "damaged": bool(damaged[idx]),
                    "repair_time_hours": float(repair_times[idx]),
                    "repair_time_model": repair_model,
                    "notes": row["notes"],
                }
                realized_rows.append(realized_record)
                repair_rows.append(repair_record)

    return pd.DataFrame(realized_rows), pd.DataFrame(repair_rows)

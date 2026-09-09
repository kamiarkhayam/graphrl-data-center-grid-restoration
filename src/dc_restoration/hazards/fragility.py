from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.special as sp
import scipy.stats as stats

SUBSTATION_FRAGILITY_PARAMS = {
    "Moderate": {"mu": 5.068, "sigma": 0.136},
    "Severe": {"mu": 5.204, "sigma": 0.147},
    "Complete": {"mu": 5.523, "sigma": 0.132},
}


def substation_state_probabilities(wind_speed_mps: np.ndarray) -> pd.DataFrame:
    wind_speed_mph = np.asarray(wind_speed_mps, dtype=float) * 2.23694
    ln_ws = np.log(np.maximum(wind_speed_mph, 1e-9))
    prob_complete = stats.norm.cdf(
        ln_ws,
        loc=SUBSTATION_FRAGILITY_PARAMS["Complete"]["mu"],
        scale=SUBSTATION_FRAGILITY_PARAMS["Complete"]["sigma"],
    )
    prob_severe = (
        stats.norm.cdf(
            ln_ws,
            loc=SUBSTATION_FRAGILITY_PARAMS["Severe"]["mu"],
            scale=SUBSTATION_FRAGILITY_PARAMS["Severe"]["sigma"],
        )
        - prob_complete
    )
    prob_moderate = (
        stats.norm.cdf(
            ln_ws,
            loc=SUBSTATION_FRAGILITY_PARAMS["Moderate"]["mu"],
            scale=SUBSTATION_FRAGILITY_PARAMS["Moderate"]["sigma"],
        )
        - prob_complete
        - prob_severe
    )
    prob_none = 1.0 - (prob_moderate + prob_severe + prob_complete)
    return pd.DataFrame(
        {
            "prob_none": np.clip(prob_none, 0.0, 1.0),
            "prob_moderate": np.clip(prob_moderate, 0.0, 1.0),
            "prob_severe": np.clip(prob_severe, 0.0, 1.0),
            "prob_complete": np.clip(prob_complete, 0.0, 1.0),
        }
    )


def sample_substation_states(wind_speed_mps: np.ndarray, rng: np.random.Generator) -> list[str]:
    probs = substation_state_probabilities(wind_speed_mps)
    rand = rng.random(len(probs))
    thresholds_1 = probs["prob_none"].to_numpy()
    thresholds_2 = thresholds_1 + probs["prob_moderate"].to_numpy()
    thresholds_3 = thresholds_2 + probs["prob_severe"].to_numpy()
    states = np.full(len(probs), "Complete", dtype=object)
    states[rand < thresholds_1] = "None"
    states[(rand >= thresholds_1) & (rand < thresholds_2)] = "Moderate"
    states[(rand >= thresholds_2) & (rand < thresholds_3)] = "Severe"
    return states.tolist()


def conductor_failure_probability(
    wind_speeds_mps: np.ndarray,
    eye_locations: np.ndarray,
    conductor_endpoints: np.ndarray,
) -> np.ndarray:
    wind_speeds = np.asarray(wind_speeds_mps, dtype=float)
    eye_locations = np.asarray(eye_locations, dtype=float)
    conductor_endpoints = np.asarray(conductor_endpoints, dtype=float)
    delta_x_cond = conductor_endpoints[:, 2] - conductor_endpoints[:, 0]
    delta_y_cond = conductor_endpoints[:, 3] - conductor_endpoints[:, 1]
    theta_conductor = np.degrees(np.arctan2(delta_y_cond, delta_x_cond))

    mid_x = (conductor_endpoints[:, 0] + conductor_endpoints[:, 2]) / 2.0
    mid_y = (conductor_endpoints[:, 1] + conductor_endpoints[:, 3]) / 2.0
    delta_x_eye = mid_x - eye_locations[:, 0]
    delta_y_eye = mid_y - eye_locations[:, 1]
    theta_wind = np.degrees(np.arctan2(delta_x_eye, -delta_y_eye))
    theta_attack = np.abs(theta_wind - theta_conductor) % 90
    theta_attack = np.minimum(theta_attack, 40)

    keys = np.array([0, 10, 20, 30, 40], dtype=float)
    mu1_vals = np.array([4.251, 4.266, 4.313, 4.398, 4.544])
    sigma1_vals = np.array([0.0245, 0.0244, 0.0245, 0.0283, 0.0385])
    mu2_vals = np.array([4.267, 4.282, 4.330, 4.414, 4.544])
    sigma2_vals = np.array([0.0363, 0.0362, 0.0365, 0.0369, 0.0385])
    vcut_vals = np.array([68, 69, 72, 77, 0])

    lower_idx = np.searchsorted(keys, theta_attack, side="right") - 1
    lower_idx = np.clip(lower_idx, 0, len(keys) - 1)
    upper_idx = np.clip(lower_idx + 1, 0, len(keys) - 1)
    lower_yaw = keys[lower_idx]
    upper_yaw = keys[upper_idx]
    diff = upper_yaw - lower_yaw
    weight = np.divide(
        theta_attack - lower_yaw, diff, out=np.zeros_like(theta_attack), where=diff != 0
    )

    mu1_interp = mu1_vals[lower_idx] * (1 - weight) + mu1_vals[upper_idx] * weight
    sigma1_interp = sigma1_vals[lower_idx] * (1 - weight) + sigma1_vals[upper_idx] * weight
    mu2_interp = mu2_vals[lower_idx] * (1 - weight) + mu2_vals[upper_idx] * weight
    sigma2_interp = sigma2_vals[lower_idx] * (1 - weight) + sigma2_vals[upper_idx] * weight
    vcut_interp = vcut_vals[lower_idx] * (1 - weight) + vcut_vals[upper_idx] * weight
    use_case1 = wind_speeds >= vcut_interp
    mu_final = np.where(use_case1, mu1_interp, mu2_interp)
    sigma_final = np.where(use_case1, sigma1_interp, sigma2_interp)
    z_score = (np.log(np.maximum(wind_speeds, 1e-9)) - mu_final) / (np.sqrt(2.0) * sigma_final)
    return np.clip(0.5 + 0.5 * sp.erf(z_score), 0.0, 1.0)


def pole_failure_probability_for_overhead_edges(
    wind_speeds_mps: np.ndarray,
    eye_locations: np.ndarray,
    edge_endpoints: np.ndarray,
    span_lengths_m: np.ndarray,
    pole_height_m: float = 12.0,
    pole_class: int = 5,
) -> np.ndarray:
    """
    Adapt the old wood-pole fragility to aggregated overhead feeder segments by
    treating each segment as a representative pole-supported span.
    """
    wind_speeds = np.asarray(wind_speeds_mps, dtype=float)
    eye_locations = np.asarray(eye_locations, dtype=float)
    edge_endpoints = np.asarray(edge_endpoints, dtype=float)
    span_lengths = np.asarray(span_lengths_m, dtype=float)

    delta_x = edge_endpoints[:, 2] - edge_endpoints[:, 0]
    delta_y = edge_endpoints[:, 3] - edge_endpoints[:, 1]
    theta_edge = np.degrees(np.arctan2(delta_y, delta_x))
    mid_x = (edge_endpoints[:, 0] + edge_endpoints[:, 2]) / 2.0
    mid_y = (edge_endpoints[:, 1] + edge_endpoints[:, 3]) / 2.0
    delta_x_eye = mid_x - eye_locations[:, 0]
    delta_y_eye = mid_y - eye_locations[:, 1]
    theta_wind = np.degrees(np.arctan2(delta_x_eye, -delta_y_eye))
    theta_attack = np.abs(theta_wind - theta_edge) % 90
    conductor_area = 3.0 * 0.011 * span_lengths
    age = 50.0

    fragility_params = {
        5: {
            "mu": [
                6.597,
                -0.01285,
                -0.08189,
                0.00685,
                -0.0654,
                0.00004979,
                -0.001368,
                0.005333,
                -0.0000007649,
                -0.000003429,
                -0.0001405,
                0.0002438,
                0.001743,
                -0.00001836,
                0.0005301,
            ],
            "sigma": [
                0.1497,
                0.0002763,
                0.002362,
                -0.002293,
                0.00006042,
                -0.0000005062,
                0.000001237,
                -0.00007755,
                -0.000002332,
                -0.00002446,
                0.00006829,
                -0.000004114,
                -0.00002209,
                0.00001401,
                -0.0000002003,
            ],
        }
    }
    mu_params = np.array(fragility_params[pole_class]["mu"], dtype=float)
    sigma_params = np.array(fragility_params[pole_class]["sigma"], dtype=float)

    mu_val = (
        mu_params[0]
        + mu_params[1] * theta_attack
        + mu_params[2] * conductor_area
        + mu_params[3] * age
        + mu_params[4] * pole_height_m
        + mu_params[5] * theta_attack**2
        + mu_params[6] * conductor_area * theta_attack
        + mu_params[7] * conductor_area**2
        + mu_params[8] * theta_attack * age
        + mu_params[9] * conductor_area * age
        + mu_params[10] * age**2
        + mu_params[11] * theta_attack * pole_height_m
        + mu_params[12] * conductor_area * pole_height_m
        + mu_params[13] * age * pole_height_m
        + mu_params[14] * pole_height_m**2
    )
    beta_val = (
        sigma_params[0]
        + sigma_params[1] * theta_attack
        + sigma_params[2] * conductor_area
        + sigma_params[3] * age
        + sigma_params[4] * pole_height_m
        + sigma_params[5] * theta_attack**2
        + sigma_params[6] * conductor_area * theta_attack
        + sigma_params[7] * conductor_area**2
        + sigma_params[8] * theta_attack * age
        + sigma_params[9] * conductor_area * age
        + sigma_params[10] * age**2
        + sigma_params[11] * theta_attack * pole_height_m
        + sigma_params[12] * conductor_area * pole_height_m
        + sigma_params[13] * age * pole_height_m
        + sigma_params[14] * pole_height_m**2
    )
    beta_val = np.maximum(beta_val, 1e-6)
    z_score = (np.log(np.maximum(wind_speeds * 2.23694, 1e-9)) - mu_val) / beta_val
    return np.clip(stats.norm.cdf(z_score), 0.0, 1.0)


def classify_edge_component(edge_row: pd.Series, enabled_types: set[str]) -> dict[str, Any]:
    component_type = str(edge_row.get("hurricane_damage_component_type", "not_applicable"))
    edge_role = str(edge_row.get("edge_role", "other"))
    is_equivalent = bool(edge_row.get("is_equivalent_edge", False))
    is_physical = bool(edge_row.get("is_physical_line", False))
    if is_equivalent or edge_role == "data_center_connection":
        return {
            "eligible": False,
            "component_type": component_type if component_type else "equivalent_source",
            "fragility_model": "excluded_equivalent_connection",
            "notes": "Equivalent or schematic connection excluded from hurricane damage.",
        }
    if component_type == "overhead_distribution_line" and is_physical:
        return {
            "eligible": component_type in enabled_types,
            "component_type": component_type,
            "fragility_model": "legacy_pole_fragility_adapted_to_overhead_distribution_edge",
            "notes": "Mapped to the old wood-pole fragility using representative span and pole geometry because the new case aggregates feeder segments instead of explicit pole nodes.",
        }
    if component_type == "transmission_line" and is_physical:
        return {
            "eligible": component_type in enabled_types,
            "component_type": component_type,
            "fragility_model": "legacy_conductor_fragility",
            "notes": "Physical transmission line treated with the old conductor fragility.",
        }
    if component_type == "underground_distribution_line":
        return {
            "eligible": False,
            "component_type": component_type,
            "fragility_model": "unsupported_underground_line",
            "notes": "Legacy code does not provide underground or flood fragility, so underground feeder segments are excluded for now.",
        }
    if component_type == "transformer":
        return {
            "eligible": False,
            "component_type": component_type,
            "fragility_model": "unsupported_transformer",
            "notes": "Legacy code does not include a dedicated transformer fragility curve.",
        }
    if component_type == "switch":
        return {
            "eligible": False,
            "component_type": component_type,
            "fragility_model": "unsupported_switch",
            "notes": "Legacy code does not include a dedicated switch fragility curve.",
        }
    return {
        "eligible": False,
        "component_type": component_type or "not_applicable",
        "fragility_model": "not_applicable",
        "notes": "No confident mapping to a legacy hurricane fragility category.",
    }


def classify_node_component(node_row: pd.Series, enabled_types: set[str]) -> dict[str, Any]:
    is_substation = (
        bool(node_row.get("is_substation", False))
        or str(node_row.get("node_type", "")).lower() == "substation_interface"
    )
    if is_substation:
        return {
            "eligible": "substation" in enabled_types,
            "component_type": "substation",
            "fragility_model": "legacy_substation_fragility",
            "notes": "Mapped to the old Hazus-style substation wind fragility.",
        }
    if bool(node_row.get("is_data_center", False)):
        return {
            "eligible": False,
            "component_type": "data_center_facility",
            "fragility_model": "excluded_data_center_facility",
            "notes": "Data-center facility internals are excluded from hurricane damage by default.",
        }
    return {
        "eligible": False,
        "component_type": "not_applicable",
        "fragility_model": "not_applicable",
        "notes": "No legacy node-level fragility mapping for this asset type.",
    }

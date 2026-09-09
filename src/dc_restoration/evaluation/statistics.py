"""Summary formulas extracted from the archived final paper implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dc_restoration.evaluation.trajectories import SUBSET_DISPLAY

EPS = 1e-9
GREEDY_RAW = "Greedy"


def _display_map(config):
    return {item["display_name"]: item["display_name"] for item in config["displayed_policies"]}


def _bootstrap_mean(values: np.ndarray, seed: int, n: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _pairwise_vs_greedy(
    frame: pd.DataFrame, subset_col: str, config: dict[str, Any]
) -> pd.DataFrame:
    metric_map = {
        "harmonized_economic_damage_dollars": "economic_damage",
        "unserved_mwh": "ens",
        "critical_unserved_mwh": "critical_ens",
    }
    keys = [subset_col, "scenario_key"]
    greedy = frame.loc[frame["policy"].eq(GREEDY_RAW), keys + list(metric_map)].rename(
        columns={metric: f"greedy_{name}" for metric, name in metric_map.items()}
    )
    paired = frame.loc[~frame["policy"].eq(GREEDY_RAW)].merge(
        greedy, on=keys, validate="many_to_one"
    )
    for metric, name in metric_map.items():
        paired[f"{name}_improvement_percent"] = (
            100.0
            * (paired[f"greedy_{name}"] - paired[metric])
            / paired[f"greedy_{name}"].clip(lower=EPS)
        )
    rows: list[dict[str, Any]] = []
    display = _display_map(config)
    for (subset, policy), part in paired.groupby([subset_col, "policy"], sort=False):
        econ = part["economic_damage_improvement_percent"].to_numpy(float)
        lo, hi = _bootstrap_mean(
            econ,
            int(config["random_seed"]) + len(rows),
            int(config["runtime"]["bootstrap_resamples"]),
        )
        rows.append(
            {
                subset_col: subset,
                "display_subset": SUBSET_DISPLAY.get(str(subset), str(subset)),
                "policy": policy,
                "display_policy": display.get(str(policy), str(policy)),
                "baseline_policy": GREEDY_RAW,
                "baseline_display_policy": display.get(GREEDY_RAW, GREEDY_RAW),
                "episode_count": int(part["scenario_key"].nunique()),
                "mean_economic_improvement_percent": float(np.mean(econ)),
                "median_economic_improvement_percent": float(np.median(econ)),
                "economic_bootstrap_95_low_percent": lo,
                "economic_bootstrap_95_high_percent": hi,
                "mean_ens_improvement_percent": float(part["ens_improvement_percent"].mean()),
                "median_ens_improvement_percent": float(part["ens_improvement_percent"].median()),
                "mean_critical_ens_improvement_percent": float(
                    part["critical_ens_improvement_percent"].mean()
                ),
                "median_critical_ens_improvement_percent": float(
                    part["critical_ens_improvement_percent"].median()
                ),
                "win_rate": float(
                    (
                        part["harmonized_economic_damage_dollars"]
                        < part["greedy_economic_damage"] - 1e-6
                    ).mean()
                ),
                "tie_rate": float(
                    (
                        np.abs(
                            part["harmonized_economic_damage_dollars"]
                            - part["greedy_economic_damage"]
                        )
                        <= 1e-6
                    ).mean()
                ),
                "loss_rate": float(
                    (
                        part["harmonized_economic_damage_dollars"]
                        > part["greedy_economic_damage"] + 1e-6
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame, subset_col: str, config: dict[str, Any]) -> pd.DataFrame:
    display = _display_map(config)
    rows: list[dict[str, Any]] = []
    for (subset, policy), part in frame.groupby([subset_col, "policy"], sort=False):
        rows.append(
            {
                subset_col: subset,
                "display_subset": SUBSET_DISPLAY.get(str(subset), str(subset)),
                "policy": policy,
                "display_policy": display.get(str(policy), str(policy)),
                "episode_count": int(part["scenario_key"].nunique()),
                "mean_harmonized_economic_damage_dollars": float(
                    part["harmonized_economic_damage_dollars"].mean()
                ),
                "mean_ens_mwh": float(part["unserved_mwh"].mean()),
                "mean_critical_ens_mwh": float(part["critical_unserved_mwh"].mean()),
                "mean_episode_length": float(part["episode_length"].mean()),
                "mean_runtime_seconds": float(part["runtime_seconds"].mean()),
                "invalid_action_count": int(part["invalid_action_count"].sum()),
                "export_cap_violation_count": int(part["export_cap_violation_count"].sum()),
                "support_outside_anchor_zone_count": int(
                    part["support_outside_anchor_zone_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)

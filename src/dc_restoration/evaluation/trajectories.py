from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from dc_restoration.anchors.controller import ResourceConfig
from dc_restoration.economics.duration_damage import _valuation_parameters
from dc_restoration.economics.interruption_damage import HarmonizedValuator
from dc_restoration.economics.load_accounting import _load_inventory, _support_by_load
from dc_restoration.paths import repository_root
from dc_restoration.policies.graph import EdgeConditionedMessagePassing, to_torch_batch
from dc_restoration.policies.training import GRAPH_KEYS, HarmonizedObservationAdapter
from dc_restoration.restoration.environment import V03RestorationModel
from dc_restoration.testbed.benchmark import CachedRestorationSimulator

ROOT = repository_root()

EPS = 1e-9

GREEDY_RAW = "Greedy Harmonized"

RESOURCE_DISPLAY = {
    "R0_no_dc_anchor": "No anchor",
    "R1_current_power_bounded_anchor": "Power-only reference",
    "E12_finite_energy_12h": "E12",
    "E24_finite_energy_24h": "E24",
}

SUBSET_DISPLAY = {
    "benchmark_hard_severe": "Severe",
    "benchmark_hard_long_sequence": "Long-sequence",
    "benchmark_hard_critical": "Critical",
    "benchmark_hard_anchor_relevant": "Anchor-relevant",
    "benchmark_hard_historical_nonzero": "Historical-nonzero",
    "benchmark_hard_validation": "Validation",
    "benchmark_hard_test": "Test",
    "benchmark_hard_stress_intensity_scaled": "Stress/intensity-scaled",
    "fresh_severe": "Fresh severe",
    "fresh_long_sequence": "Fresh long-sequence",
    "fresh_critical": "Fresh critical",
    "fresh_anchor_relevant": "Fresh anchor-relevant",
    "fresh_historical_nonzero": "Fresh historical-nonzero",
    "fresh_stress_intensity_scaled": "Fresh stress/intensity-scaled",
    "fresh_unique_bank": "Unique fresh bank",
    "fresh_overlap_adjusted": "Overlap-adjusted fresh aggregate",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load_context(
    config: dict[str, Any], damage_path: Path | None = None
) -> tuple[V03RestorationModel, pd.DataFrame, HarmonizedValuator, HarmonizedObservationAdapter]:
    case_dir = _resolve(config["paths"]["case_dir"])
    model = V03RestorationModel(case_dir)
    if damage_path is not None:
        damage = pd.read_csv(damage_path)
        model.damage = damage
        model.base = CachedRestorationSimulator(model.nodes, model.edges, model.loads, damage)
        model._metric_cache = {}
    node_cfg = _read_yaml(_resolve(config["paths"]["valuation_node_config"]))
    inventory = _load_inventory(model, node_cfg["commercial_size_classes"])
    duration_cfg = _read_yaml(_resolve(config["paths"]["valuation_duration_config"]))
    parameters, _ = _valuation_parameters(duration_cfg)
    valuator = HarmonizedValuator(
        inventory,
        parameters,
        float(config["valuation"]["critical_value_usd_per_kwh"]),
    )
    return model, inventory, valuator, HarmonizedObservationAdapter(model, valuator, inventory)


def _rates_summary(rates_kw: dict[str, float], inventory: pd.DataFrame) -> tuple[float, float]:
    rates = pd.Series(rates_kw, dtype=float)
    total = float(rates.sum()) / 1000.0
    critical_ids = set(inventory.loc[inventory["critical"].astype(bool), "load_id"].astype(str))
    critical = float(rates.loc[rates.index.astype(str).isin(critical_ids)].sum()) / 1000.0
    return total, critical


def _power_only_rates(model: V03RestorationModel, remaining: set[str]) -> dict[str, float]:
    base = model.base.metrics(tuple(sorted(remaining)))
    allocation = _support_by_load(model, base, sum(model.dc_support_by_anchor.values()))
    return {
        str(load_id): max(
            0.0,
            float(model.load_info[str(load_id)].get("baseline_p_kw", 0.0))
            - 1000.0 * float(allocation.get(str(load_id), 0.0)),
        )
        for load_id in map(str, base.unserved_load_ids)
    }


def _trajectory_from_order(
    model: V03RestorationModel,
    inventory: pd.DataFrame,
    valuator: HarmonizedValuator,
    scenario: str,
    policy: str,
    display_policy: str,
    order: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    damaged = set(map(str, model.scenario_edges(scenario)))
    repair_times = model.scenario_repair_times(scenario)
    repaired: set[str] = set()
    accumulated_kwh: dict[str, float] = {}
    initial_rates = _power_only_rates(model, damaged)
    initial_unserved, initial_critical = _rates_summary(initial_rates, inventory)
    total_connected = float(inventory["baseline_p_kw"].sum()) / 1000.0
    total_critical = (
        float(inventory.loc[inventory["critical"].astype(bool), "baseline_p_kw"].sum()) / 1000.0
    )
    cumulative_time = 0.0
    cumulative_ens = 0.0
    cumulative_critical_ens = 0.0
    cumulative_support = 0.0
    t_values: dict[str, float | None] = {
        "T50_total_hours": None,
        "T90_total_hours": None,
        "T95_total_hours": None,
        "T50_critical_hours": None,
        "T90_critical_hours": None,
        "T95_critical_hours": None,
    }
    full_time: float | None = None
    full_critical_time: float | None = None
    rows: list[dict[str, Any]] = []
    for edge_id in map(str, order):
        if edge_id not in damaged or edge_id in repaired:
            continue
        remaining = damaged - repaired
        rates = _power_only_rates(model, remaining)
        current_unserved, current_critical = _rates_summary(rates, inventory)
        if current_unserved <= EPS and current_critical <= EPS:
            full_time = cumulative_time
            full_critical_time = cumulative_time
            break
        duration = float(repair_times.get(edge_id, 8.0))
        support_mw = max(
            0.0,
            float(model.base.metrics(tuple(sorted(remaining))).unserved_mw) - current_unserved,
        )
        before_cost = valuator.cost_from_energy(accumulated_kwh)
        for load_id, rate_kw in rates.items():
            accumulated_kwh[load_id] = accumulated_kwh.get(load_id, 0.0) + rate_kw * duration
        after_cost = valuator.cost_from_energy(accumulated_kwh)
        step_ens = current_unserved * duration
        step_critical = current_critical * duration
        cumulative_ens += step_ens
        cumulative_critical_ens += step_critical
        cumulative_support += support_mw * duration
        rows.append(
            {
                "scenario_key": scenario,
                "policy": policy,
                "display_policy": display_policy,
                "dc_resource_configuration": "R1_current_power_bounded_anchor",
                "dc_resource_display": RESOURCE_DISPLAY["R1_current_power_bounded_anchor"],
                "step_index": len(rows),
                "interval_start_time_hours": cumulative_time,
                "interval_duration_hours": duration,
                "cumulative_time_hours": cumulative_time + duration,
                "total_connected_load_mw": total_connected,
                "total_critical_connected_load_mw": total_critical,
                "initial_unserved_mw": initial_unserved,
                "current_unserved_mw": current_unserved,
                "initially_interrupted_load_restored_fraction": (
                    1.0 - current_unserved / max(initial_unserved, EPS)
                    if initial_unserved > EPS
                    else np.nan
                ),
                "initial_critical_unserved_mw": initial_critical,
                "current_critical_unserved_mw": current_critical,
                "initially_interrupted_critical_load_restored_fraction": (
                    1.0 - current_critical / max(initial_critical, EPS)
                    if initial_critical > EPS
                    else np.nan
                ),
                "step_unserved_energy_mwh": step_ens,
                "cumulative_ens_mwh": cumulative_ens,
                "step_critical_unserved_energy_mwh": step_critical,
                "cumulative_critical_ens_mwh": cumulative_critical_ens,
                "step_harmonized_economic_damage_dollars": after_cost - before_cost,
                "cumulative_harmonized_economic_damage_dollars": after_cost,
                "dc_support_delivered_mw": support_mw,
                "dc_support_delivered_mwh": support_mw * duration,
                "cumulative_dc_support_mwh": cumulative_support,
                "selected_repair_action": edge_id,
                "invalid_action_count": 0,
                "export_cap_violation_count": 0,
                "support_outside_anchor_zone_count": 0,
            }
        )
        repaired.add(edge_id)
        cumulative_time += duration
        next_rates = _power_only_rates(model, damaged - repaired)
        next_unserved, next_critical = _rates_summary(next_rates, inventory)
        total_fraction = (
            1.0 - next_unserved / max(initial_unserved, EPS) if initial_unserved > EPS else np.nan
        )
        critical_fraction = (
            1.0 - next_critical / max(initial_critical, EPS) if initial_critical > EPS else np.nan
        )
        for threshold, key in [
            (0.50, "T50_total_hours"),
            (0.90, "T90_total_hours"),
            (0.95, "T95_total_hours"),
        ]:
            if (
                t_values[key] is None
                and np.isfinite(total_fraction)
                and total_fraction >= threshold
            ):
                t_values[key] = cumulative_time
        for threshold, key in [
            (0.50, "T50_critical_hours"),
            (0.90, "T90_critical_hours"),
            (0.95, "T95_critical_hours"),
        ]:
            if (
                t_values[key] is None
                and np.isfinite(critical_fraction)
                and critical_fraction >= threshold
            ):
                t_values[key] = cumulative_time
        if full_critical_time is None and next_critical <= EPS:
            full_critical_time = cumulative_time
        if next_unserved <= EPS:
            full_time = cumulative_time
            break
    episode = {
        "scenario_key": scenario,
        "policy": policy,
        "display_policy": display_policy,
        "dc_resource_configuration": "R1_current_power_bounded_anchor",
        "harmonized_economic_damage_dollars": valuator.cost_from_energy(accumulated_kwh),
        "energy_not_served_mwh": cumulative_ens,
        "critical_energy_not_served_mwh": cumulative_critical_ens,
        "episode_length": len(rows),
        "repairs_used": len(repaired),
        "time_to_full_restoration_hours": full_time,
        "time_to_full_critical_restoration_hours": full_critical_time,
        "dc_support_used_mwh": cumulative_support,
        "initial_unserved_mw": initial_unserved,
        "initial_critical_unserved_mw": initial_critical,
        "total_connected_load_mw": total_connected,
        "total_critical_connected_load_mw": total_critical,
        "normalized_total_service_loss_hours": (
            cumulative_ens / max(initial_unserved, EPS) if initial_unserved > EPS else np.nan
        ),
        "normalized_critical_service_loss_hours": (
            cumulative_critical_ens / max(initial_critical, EPS)
            if initial_critical > EPS
            else np.nan
        ),
        "ASIDI_type_total_hours": cumulative_ens / max(total_connected, EPS),
        "ASIDI_type_critical_hours": cumulative_critical_ens / max(total_critical, EPS),
        "invalid_action_count": 0,
        "export_cap_violation_count": 0,
        "support_outside_anchor_zone_count": 0,
        **t_values,
    }
    return rows, episode


def _hourly_curves(
    trajectories: pd.DataFrame,
    memberships: pd.DataFrame,
    subset_col: str,
    grid_hours: float,
) -> pd.DataFrame:
    expanded = trajectories.merge(
        memberships[["scenario_key", "policy", subset_col]].drop_duplicates(),
        on=["scenario_key", "policy"],
        how="inner",
        validate="many_to_many",
    )
    value_cols = [
        "initially_interrupted_load_restored_fraction",
        "initially_interrupted_critical_load_restored_fraction",
        "cumulative_ens_mwh",
        "cumulative_critical_ens_mwh",
        "cumulative_harmonized_economic_damage_dollars",
    ]
    sampled: list[dict[str, Any]] = []
    # Keep every episode in the aggregate through a common subset horizon.
    # Completed episodes are carried forward with full restoration and final
    # cumulative costs rather than silently dropping from later time points.
    subset_horizons = (
        expanded.groupby(subset_col, as_index=True)["cumulative_time_hours"].max().to_dict()
    )
    for (subset, policy, scenario), part in expanded.groupby(
        [subset_col, "policy", "scenario_key"], sort=False
    ):
        part = part.sort_values("interval_start_time_hours")
        starts = part["interval_start_time_hours"].to_numpy(float)
        final_time = float(part["cumulative_time_hours"].max())
        common_horizon = float(subset_horizons[subset])
        grid = np.arange(
            0.0,
            math.ceil(common_horizon / grid_hours) * grid_hours + grid_hours,
            grid_hours,
        )
        for point in grid:
            index = int(np.searchsorted(starts, point, side="right") - 1)
            index = max(index, 0)
            row = part.iloc[index]
            record = {
                subset_col: subset,
                "scenario_key": scenario,
                "policy": policy,
                "display_policy": row["display_policy"],
                "time_hours": float(point),
            }
            for column in value_cols:
                value = float(row[column]) if pd.notna(row[column]) else np.nan
                if point >= final_time:
                    if "restored_fraction" in column and np.isfinite(value):
                        value = 1.0
                    elif column == "cumulative_ens_mwh":
                        value = float(part[column].iloc[-1])
                    elif column == "cumulative_critical_ens_mwh":
                        value = float(part[column].iloc[-1])
                    elif column == "cumulative_harmonized_economic_damage_dollars":
                        value = float(part[column].iloc[-1])
                record[column] = value
            sampled.append(record)
    sampled_frame = pd.DataFrame(sampled)
    rows: list[dict[str, Any]] = []
    for (subset, policy, display, point), part in sampled_frame.groupby(
        [subset_col, "policy", "display_policy", "time_hours"], sort=False
    ):
        record = {
            subset_col: subset,
            "display_subset": SUBSET_DISPLAY.get(str(subset), str(subset)),
            "policy": policy,
            "display_policy": display,
            "time_hours": point,
            "episode_count": int(part["scenario_key"].nunique()),
        }
        for column in value_cols:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            record[f"mean_{column}"] = float(values.mean()) if len(values) else np.nan
            record[f"q25_{column}"] = float(values.quantile(0.25)) if len(values) else np.nan
            record[f"q75_{column}"] = float(values.quantile(0.75)) if len(values) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def _resource_rates(
    model: V03RestorationModel,
    remaining: set[str],
    cfg: ResourceConfig,
    energy_remaining_mwh: float,
    duration_hours: float,
) -> tuple[dict[str, float], float, float, Any]:
    base = model.base.metrics(tuple(sorted(remaining)))
    cap_mw = cfg.max_support_mw(sum(model.dc_support_by_anchor.values()))
    allocation = _support_by_load(model, base, cap_mw)
    potential_mwh = float(sum(allocation.values())) * duration_hours
    if cap_mw <= EPS or potential_mwh <= EPS:
        fraction = 0.0
        used_mwh = 0.0
    elif cfg.finite_energy_hours is None:
        fraction = 1.0
        used_mwh = potential_mwh
    else:
        used_mwh = min(max(0.0, energy_remaining_mwh), potential_mwh)
        fraction = used_mwh / max(potential_mwh, EPS)
    rates = {
        str(load_id): max(
            0.0,
            float(model.load_info[str(load_id)].get("baseline_p_kw", 0.0))
            - 1000.0 * float(allocation.get(str(load_id), 0.0)) * fraction,
        )
        for load_id in map(str, base.unserved_load_ids)
    }
    support_mw = float(sum(allocation.values())) * fraction
    return rates, used_mwh, support_mw, base


class HarmonizedResourceAdapter:
    """Apply the trained harmonized observation schema to a DC resource state."""

    def __init__(
        self,
        model: V03RestorationModel,
        inventory: pd.DataFrame,
        valuator: HarmonizedValuator,
        base_adapter: HarmonizedObservationAdapter,
    ) -> None:
        self.model = model
        self.inventory = inventory.set_index("load_id", drop=False)
        self.valuator = valuator
        self.base = base_adapter

    def build_state(
        self,
        damaged: Sequence[str],
        repair_times: dict[str, float],
        repaired: set[str],
        accumulated_energy_kwh: dict[str, float],
        cfg: ResourceConfig,
        energy_remaining_mwh: float,
        k: int,
    ) -> tuple[SimpleNamespace, list[dict[str, Any]], dict[str, np.ndarray], dict[str, float]]:
        damaged_list = list(map(str, damaged))
        remaining = set(damaged_list) - set(map(str, repaired))
        rates, _used, support_mw, base_physical = _resource_rates(
            self.model, remaining, cfg, energy_remaining_mwh, 1.0
        )
        unserved_mw, critical_unserved_mw = _rates_summary(
            rates, self.inventory.reset_index(drop=True)
        )
        damage_rate = self.valuator.marginal_one_hour_cost(accumulated_energy_kwh, rates)
        before = SimpleNamespace(
            unserved_mw=unserved_mw,
            critical_unserved_mw=critical_unserved_mw,
            dc_support_used_mw=support_mw,
            dc_anchor_active=bool(support_mw > EPS),
            outage_islands=int(base_physical.island_count),
        )
        scores: list[dict[str, Any]] = []
        for edge_id in remaining:
            after_rates, _after_used, _after_support, _after_base = _resource_rates(
                self.model,
                remaining - {edge_id},
                cfg,
                energy_remaining_mwh,
                1.0,
            )
            after_unserved, after_critical = _rates_summary(
                after_rates, self.inventory.reset_index(drop=True)
            )
            after_damage_rate = self.valuator.marginal_one_hour_cost(
                accumulated_energy_kwh, after_rates
            )
            repair_time = float(repair_times.get(edge_id, 8.0))
            gain = damage_rate - after_damage_rate
            scores.append(
                {
                    "edge_id": edge_id,
                    "repair_time": repair_time,
                    "immediate_harmonized_gain": gain,
                    "immediate_voll_gain": gain,
                    "immediate_critical_gain": critical_unserved_mw - after_critical,
                    "immediate_unserved_gain": unserved_mw - after_unserved,
                    "harmonized_gain_per_hour": gain / max(repair_time, EPS),
                    "voll_gain_per_hour": gain / max(repair_time, EPS),
                    "critical_gain_per_hour": (critical_unserved_mw - after_critical)
                    / max(repair_time, EPS),
                    "unserved_gain_per_hour": (unserved_mw - after_unserved)
                    / max(repair_time, EPS),
                }
            )
        scores.sort(
            key=lambda row: (
                row["immediate_harmonized_gain"],
                -row["repair_time"],
                row["edge_id"],
            ),
            reverse=True,
        )
        scores = scores[:k]

        node = np.concatenate(
            [
                self.base.static_node.copy(),
                np.zeros((len(self.base.node_ids), 4), dtype=np.float32),
            ],
            axis=1,
        )
        load_marginal = self.base._load_marginal_costs(accumulated_energy_kwh, rates)
        node_damage = np.zeros(len(self.base.node_ids), dtype=np.float32)
        node_unserved = np.zeros(len(self.base.node_ids), dtype=np.float32)
        node_critical = np.zeros(len(self.base.node_ids), dtype=np.float32)
        for load_id, rate_kw in rates.items():
            node_id = str(self.inventory.at[load_id, "node_id"])
            index = self.base.node_index.get(node_id)
            if index is None:
                continue
            node_damage[index] += float(load_marginal.get(load_id, 0.0)) / 1e6
            node_unserved[index] += float(rate_kw) / 1000.0
            if bool(self.inventory.at[load_id, "critical"]):
                node_critical[index] += float(rate_kw) / 1000.0
        node[:, 3] = node_damage
        node[:, -4] = node_unserved / 100.0
        node[:, -3] = node_critical / 50.0

        edge = np.concatenate(
            [
                self.base.static_edge.copy(),
                np.zeros((len(self.base.edge_ids), 3), dtype=np.float32),
            ],
            axis=1,
        )
        for edge_id in remaining:
            index = self.base.edge_index_map.get(edge_id)
            if index is None:
                continue
            edge[index, -3] = 1.0
            edge[index, -1] = float(repair_times.get(edge_id, 8.0)) / 24.0
            u, v = self.model.edge_endpoints.get(edge_id, ("", ""))
            if str(u) in self.base.node_index:
                node[self.base.node_index[str(u)], -2] += 0.1
            if str(v) in self.base.node_index:
                node[self.base.node_index[str(v)], -2] += 0.1
        for edge_id in repaired:
            index = self.base.edge_index_map.get(edge_id)
            if index is None:
                continue
            edge[index, -2] = 1.0
            u, v = self.model.edge_endpoints.get(edge_id, ("", ""))
            if str(u) in self.base.node_index:
                node[self.base.node_index[str(u)], -1] += 0.1
            if str(v) in self.base.node_index:
                node[self.base.node_index[str(v)], -1] += 0.1

        candidate_indices = np.zeros(k, dtype=np.int64)
        candidate_features = np.zeros((k, 12), dtype=np.float32)
        candidate_mask = np.zeros(k, dtype=bool)
        for index, score in enumerate(scores):
            candidate_indices[index] = self.base.edge_index_map.get(str(score["edge_id"]), 0)
            candidate_features[index] = self.base._candidate_features(
                before, score, damage_rate, len(scores), len(damaged_list)
            )
            candidate_mask[index] = True
        global_features = np.asarray(
            [
                unserved_mw / 100.0,
                critical_unserved_mw / 50.0,
                damage_rate / 1e6,
                len(remaining) / 60.0,
                len(repaired) / 60.0,
                len(damaged_list) / 60.0,
                support_mw / 20.0,
                float(support_mw > EPS),
                float(base_physical.island_count) / 50.0,
            ],
            dtype=np.float32,
        )
        observation = {
            "node_features": node.astype(np.float32),
            "edge_features": edge.astype(np.float32),
            "global_features": global_features,
            "candidate_edge_indices": candidate_indices,
            "candidate_features": candidate_features,
            "candidate_mask": candidate_mask,
        }
        return before, scores, observation, rates


def _dc_rollout(
    model: V03RestorationModel,
    inventory: pd.DataFrame,
    valuator: HarmonizedValuator,
    adapter: HarmonizedResourceAdapter,
    scenario: str,
    policy_name: str,
    display_policy: str,
    cfg: ResourceConfig,
    k: int,
    device: torch.device,
    policy: Any | None = None,
    edge_index: torch.Tensor | None = None,
    fixed_order: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    damaged = list(map(str, model.scenario_edges(scenario)))
    repair_times = model.scenario_repair_times(scenario)
    repaired: set[str] = set()
    accumulated: dict[str, float] = {}
    energy_budget = cfg.initial_energy_mwh(sum(model.dc_support_by_anchor.values()))
    energy_remaining = energy_budget
    initial_rates, _initial_used, _initial_support, _ = _resource_rates(
        model, set(damaged), cfg, energy_remaining, 1.0
    )
    initial_unserved, initial_critical = _rates_summary(initial_rates, inventory)
    total_connected = float(inventory["baseline_p_kw"].sum()) / 1000.0
    total_critical = (
        float(inventory.loc[inventory["critical"].astype(bool), "baseline_p_kw"].sum()) / 1000.0
    )
    rows: list[dict[str, Any]] = []
    order: list[str] = []
    elapsed = 0.0
    cumulative_ens = 0.0
    cumulative_critical = 0.0
    cumulative_support = 0.0
    depletion_time = np.nan
    message_hook_calls = 0
    hooks = []
    if policy is not None:

        def _hook(_module: Any, _inputs: Any, _output: Any) -> None:
            nonlocal message_hook_calls
            message_hook_calls += 1

        hooks = [
            module.register_forward_hook(_hook)
            for module in policy.modules()
            if isinstance(module, EdgeConditionedMessagePassing)
        ]
    started = time.time()
    fixed = list(map(str, fixed_order)) if fixed_order is not None else None
    fixed_position = 0
    while len(repaired) < len(damaged):
        if fixed is not None:
            # A precomputed action order does not need the expensive K-candidate
            # lookahead observation. Evaluate only the current physical state;
            # the interval below still uses the identical resource/controller
            # calculation and harmonized economic valuation.
            remaining = set(damaged) - repaired
            current_rates, _used, _support, _physical = _resource_rates(
                model, remaining, cfg, energy_remaining, 1.0
            )
            current_unserved, _current_critical = _rates_summary(current_rates, inventory)
            if current_unserved <= EPS:
                break
            chosen = None
            while fixed_position < len(fixed):
                candidate = fixed[fixed_position]
                fixed_position += 1
                if candidate in set(damaged) and candidate not in repaired:
                    chosen = candidate
                    break
            if chosen is None:
                break
        else:
            before, scores, observation, _decision_rates = adapter.build_state(
                damaged,
                repair_times,
                repaired,
                accumulated,
                cfg,
                energy_remaining,
                k,
            )
            if before.unserved_mw <= EPS or not scores:
                break
            if policy is None:
                chosen = str(scores[0]["edge_id"])
            else:
                batch = to_torch_batch(
                    {key: observation[key][None, ...] for key in GRAPH_KEYS}, device
                )
                with torch.no_grad():
                    action, _logp, _entropy, _value = policy.act(
                        batch, edge_index, deterministic=True
                    )
                chosen = str(scores[min(int(action.item()), len(scores) - 1)]["edge_id"])
        if chosen in repaired:
            raise RuntimeError(f"Repeated DC action {chosen} in {scenario}")
        remaining = set(damaged) - repaired
        duration = float(repair_times.get(chosen, 8.0))
        interval_rates, support_used, support_mw, base_physical = _resource_rates(
            model, remaining, cfg, energy_remaining, duration
        )
        current_unserved, current_critical = _rates_summary(interval_rates, inventory)
        before_cost = valuator.cost_from_energy(accumulated)
        for load_id, rate_kw in interval_rates.items():
            accumulated[load_id] = accumulated.get(load_id, 0.0) + rate_kw * duration
        after_cost = valuator.cost_from_energy(accumulated)
        cumulative_ens += current_unserved * duration
        cumulative_critical += current_critical * duration
        cumulative_support += support_used
        previous_energy = energy_remaining
        if cfg.finite_energy_hours is not None:
            energy_remaining = max(0.0, energy_remaining - support_used)
            if (
                previous_energy > EPS
                and energy_remaining <= EPS
                and not np.isfinite(depletion_time)
            ):
                depletion_time = elapsed + support_used / max(support_mw, EPS)
        rows.append(
            {
                "scenario_key": scenario,
                "policy": policy_name,
                "display_policy": display_policy,
                "dc_resource_configuration": cfg.code,
                "dc_resource_display": RESOURCE_DISPLAY[cfg.code],
                "step_index": len(rows),
                "interval_start_time_hours": elapsed,
                "interval_duration_hours": duration,
                "cumulative_time_hours": elapsed + duration,
                "initial_unserved_mw": initial_unserved,
                "current_unserved_mw": current_unserved,
                "initially_interrupted_load_restored_fraction": 1.0
                - current_unserved / max(initial_unserved, EPS)
                if initial_unserved > EPS
                else np.nan,
                "initial_critical_unserved_mw": initial_critical,
                "current_critical_unserved_mw": current_critical,
                "initially_interrupted_critical_load_restored_fraction": 1.0
                - current_critical / max(initial_critical, EPS)
                if initial_critical > EPS
                else np.nan,
                "step_unserved_energy_mwh": current_unserved * duration,
                "cumulative_ens_mwh": cumulative_ens,
                "step_critical_unserved_energy_mwh": current_critical * duration,
                "cumulative_critical_ens_mwh": cumulative_critical,
                "step_harmonized_economic_damage_dollars": after_cost - before_cost,
                "cumulative_harmonized_economic_damage_dollars": after_cost,
                "dc_support_delivered_mw": support_mw,
                "dc_support_delivered_mwh": support_used,
                "cumulative_dc_support_mwh": cumulative_support,
                "energy_remaining_mwh_before": previous_energy
                if cfg.finite_energy_hours is not None
                else np.nan,
                "energy_remaining_mwh_after": energy_remaining
                if cfg.finite_energy_hours is not None
                else np.nan,
                "selected_repair_action": chosen,
                "export_cap_violation_count": int(
                    support_mw > cfg.max_support_mw(sum(model.dc_support_by_anchor.values())) + 1e-6
                ),
                "energy_budget_violation_count": int(
                    cfg.finite_energy_hours is not None and support_used > previous_energy + 1e-6
                ),
                "support_outside_anchor_zone_count": 0,
                "invalid_action_count": 0,
            }
        )
        repaired.add(chosen)
        order.append(chosen)
        elapsed += duration
    for hook in hooks:
        hook.remove()
    finite = cfg.finite_energy_hours is not None
    episode = {
        "scenario_key": scenario,
        "policy": policy_name,
        "display_policy": display_policy,
        "dc_resource_configuration": cfg.code,
        "dc_resource_display": RESOURCE_DISPLAY[cfg.code],
        "harmonized_economic_damage_dollars": valuator.cost_from_energy(accumulated),
        "energy_not_served_mwh": cumulative_ens,
        "critical_energy_not_served_mwh": cumulative_critical,
        "episode_length": len(order),
        "repairs_used": len(repaired),
        "time_to_full_restoration_hours": elapsed,
        "dc_support_used_mwh": cumulative_support,
        "initial_energy_budget_mwh": energy_budget if finite else np.nan,
        "remaining_energy_mwh": energy_remaining if finite else np.nan,
        "energy_depleted": bool(finite and energy_remaining <= EPS),
        "depletion_time_hours": depletion_time,
        "useful_activation_rate": float(
            sum(row["dc_support_delivered_mwh"] > EPS for row in rows) / max(1, len(rows))
        ),
        "invalid_action_count": int(sum(row["invalid_action_count"] for row in rows)),
        "export_cap_violation_count": int(sum(row["export_cap_violation_count"] for row in rows)),
        "support_outside_anchor_zone_count": int(
            sum(row["support_outside_anchor_zone_count"] for row in rows)
        ),
        "energy_budget_violation_count": int(
            sum(row["energy_budget_violation_count"] for row in rows)
        ),
        "message_passing_hook_calls": message_hook_calls,
        "repair_sequence_edge_ids": json.dumps(order),
        "runtime_seconds": time.time() - started,
        "total_connected_load_mw": total_connected,
        "total_critical_connected_load_mw": total_critical,
        "normalized_total_service_loss_hours": cumulative_ens / max(initial_unserved, EPS)
        if initial_unserved > EPS
        else np.nan,
        "normalized_critical_service_loss_hours": cumulative_critical / max(initial_critical, EPS)
        if initial_critical > EPS
        else np.nan,
        "ASIDI_type_total_hours": cumulative_ens / max(total_connected, EPS),
        "ASIDI_type_critical_hours": cumulative_critical / max(total_critical, EPS),
    }
    return episode, rows


def _metric_reduction(
    episodes: pd.DataFrame,
    metric: str,
    policies: Sequence[str],
    subset_col: str,
) -> pd.DataFrame:
    baseline = episodes.loc[
        episodes["policy"].eq(GREEDY_RAW), [subset_col, "scenario_key", metric]
    ].rename(columns={metric: "baseline_value"})
    paired = episodes.loc[episodes["policy"].isin(policies)].merge(
        baseline, on=[subset_col, "scenario_key"], validate="many_to_one"
    )
    paired["reduction_percent"] = (
        100.0
        * (paired["baseline_value"] - paired[metric])
        / paired["baseline_value"].clip(lower=EPS)
    )
    return paired.groupby([subset_col, "policy", "display_policy"], as_index=False).agg(
        mean_reduction_percent=("reduction_percent", "mean"),
        median_reduction_percent=("reduction_percent", "median"),
        episode_count=("scenario_key", "nunique"),
    )


def _setup_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def _panel(axis: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
    )


def _export(fig: plt.Figure, d: dict[str, Path], stem: str) -> None:
    fig.savefig(d["figures/pdf"] / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(d["figures/svg"] / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(d["figures/png_600dpi"] / f"{stem}_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(d["figures/preview"] / f"{stem}_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scenario_initial_damage_rate(
    model: V03RestorationModel,
    inventory: pd.DataFrame,
    valuator: HarmonizedValuator,
    damaged_edges_json: str,
) -> tuple[float, float, float]:
    """Return no-anchor initial load severity and the first-hour damage rate."""
    damaged_edges = json.loads(str(damaged_edges_json))
    base = model.base.metrics(tuple(sorted(map(str, damaged_edges))))
    unserved_kw = {
        str(load_id): float(model.load_info[str(load_id)].get("baseline_p_kw", 0.0))
        for load_id in base.unserved_load_ids
    }
    inventory_by_load = inventory.set_index("load_id", drop=False)
    total_unserved_mw = sum(unserved_kw.values()) / 1000.0
    critical_unserved_mw = (
        sum(
            value
            for load_id, value in unserved_kw.items()
            if bool(inventory_by_load.loc[load_id, "critical"])
        )
        / 1000.0
    )
    # The nonlinear customer-damage functions do not have a single constant rate.
    # We define onset severity as the economic damage accumulated during hour one.
    damage_rate_usd_per_hour = valuator.marginal_one_hour_cost({}, unserved_kw)
    return total_unserved_mw, critical_unserved_mw, damage_rate_usd_per_hour


def _figure2(config: dict[str, Any], d: dict[str, Path]) -> None:
    controlled_manifest_path = _resolve(config["paths"]["benchmark_hard_scenario_manifest"])
    fresh_manifest_path = _resolve(config["paths"]["fresh_subset_manifest"])
    fresh_damage_path = _resolve(config["paths"]["fresh_damage_components"])
    controlled_manifest = pd.read_csv(controlled_manifest_path, low_memory=False)
    fresh_manifest = pd.read_csv(fresh_manifest_path, low_memory=False)

    controlled_model, controlled_inventory, controlled_valuator, _ = _load_context(config)
    fresh_model, fresh_inventory, fresh_valuator, _ = _load_context(config, fresh_damage_path)
    group_specs = [
        {
            "display_group": "Benchmark-hard library",
            "tick_label": "Benchmark-hard\nlibrary",
            "scenario_bank": "Benchmark-hard scenario library",
            "subset": "benchmark_hard_nonzero",
            "manifest": controlled_manifest,
            "model": controlled_model,
            "inventory": controlled_inventory,
            "valuator": controlled_valuator,
        },
        {
            "display_group": "Fresh historical-nonzero",
            "tick_label": "Fresh historical-\nnonzero",
            "scenario_bank": "Fresh unseen damage-realization bank",
            "subset": "fresh_historical_nonzero",
            "manifest": fresh_manifest.loc[
                fresh_manifest["eval_subset"].eq("fresh_historical_nonzero")
            ],
            "model": fresh_model,
            "inventory": fresh_inventory,
            "valuator": fresh_valuator,
        },
        {
            "display_group": "Fresh stress/intensity-scaled",
            "tick_label": "Fresh stress/\nintensity-scaled",
            "scenario_bank": "Fresh unseen damage-realization bank",
            "subset": "fresh_stress_intensity_scaled",
            "manifest": fresh_manifest.loc[
                fresh_manifest["eval_subset"].eq("fresh_stress_intensity_scaled")
            ],
            "model": fresh_model,
            "inventory": fresh_inventory,
            "valuator": fresh_valuator,
        },
    ]

    severity_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for spec in group_specs:
        manifest = spec["manifest"].drop_duplicates("scenario_key").copy()
        for row_index, row in manifest.iterrows():
            total_mw, critical_mw, damage_rate = _scenario_initial_damage_rate(
                spec["model"],
                spec["inventory"],
                spec["valuator"],
                row["damaged_edge_ids_json"],
            )
            common = {
                "scenario_key": str(row["scenario_key"]),
                "scenario_bank": spec["scenario_bank"],
                "subset": spec["subset"],
                "display_group": spec["display_group"],
                "historical_or_stress_label": (
                    "stress/intensity-scaled"
                    if bool(row.get("intensity_scaled_flag", False))
                    else "historical/non-stress"
                ),
            }
            for metric, value, unit in [
                (
                    "Damaged repairable components",
                    float(row["damaged_repairable_component_count"]),
                    "count",
                ),
                ("Initial unserved load", float(row["initial_unserved_mw"]), "MW"),
                (
                    "Initial critical unserved load",
                    float(row["initial_critical_unserved_mw"]),
                    "MW",
                ),
                (
                    "Initial economic interruption-damage rate",
                    damage_rate / 1_000_000.0,
                    "million USD h^-1",
                ),
            ]:
                severity_rows.append({**common, "metric": metric, "value": value, "unit": unit})
            audit_rows.append(
                {
                    **common,
                    "manifest_initial_unserved_mw": float(row["initial_unserved_mw"]),
                    "reconstructed_initial_unserved_mw": total_mw,
                    "initial_unserved_abs_error_mw": abs(
                        float(row["initial_unserved_mw"]) - total_mw
                    ),
                    "manifest_initial_critical_unserved_mw": float(
                        row["initial_critical_unserved_mw"]
                    ),
                    "reconstructed_initial_critical_unserved_mw": critical_mw,
                    "initial_critical_unserved_abs_error_mw": abs(
                        float(row["initial_critical_unserved_mw"]) - critical_mw
                    ),
                    "legacy_initial_voll_rate_usd_per_hour": float(
                        row["initial_voll_cost_dollars_per_hour"]
                    ),
                    "economic_interruption_damage_rate_usd_per_hour": damage_rate,
                    "valuation_definition": "economic damage accumulated over the first outage hour before anchor support",
                }
            )
            if (len(audit_rows) % 100) == 0:
                print(f"Figure 2 economic-severity reconstruction: {len(audit_rows)} scenarios")

    severity = pd.DataFrame(severity_rows)
    audit = pd.DataFrame(audit_rows)
    alignment_tolerance_mw = 1e-6
    audit["physical_alignment_pass"] = audit["initial_unserved_abs_error_mw"].le(
        alignment_tolerance_mw
    ) & audit["initial_critical_unserved_abs_error_mw"].le(alignment_tolerance_mw)
    if not bool(audit["physical_alignment_pass"].all()):
        raise RuntimeError("Figure 2 reconstruction does not align with manifest outage quantities")

    rng = np.random.default_rng(2303)
    severity["displayed_as_point"] = False
    summary_rows: list[dict[str, Any]] = []
    for (group, metric), part in severity.groupby(["display_group", "metric"], sort=False):
        values = part["value"].dropna().to_numpy(float)
        overlay_count = min(len(values), 120)
        displayed_index = part.index.to_numpy()
        if len(values) > overlay_count:
            displayed_index = rng.choice(displayed_index, size=overlay_count, replace=False)
        severity.loc[displayed_index, "displayed_as_point"] = True
        summary_rows.append(
            {
                "display_group": group,
                "metric": metric,
                "unit": part["unit"].iloc[0],
                "scenario_count": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "point_overlay_count": overlay_count,
            }
        )
    summary = pd.DataFrame(summary_rows)

    colors = {
        "Benchmark-hard library": "#4A4A4A",
        "Fresh historical-nonzero": "#5B83A6",
        "Fresh stress/intensity-scaled": "#B45C3C",
    }
    groups = [spec["display_group"] for spec in group_specs]
    tick_labels = [spec["tick_label"] for spec in group_specs]
    metric_panels = [
        ("Damaged repairable components", "Damaged repairable\ncomponents"),
        ("Initial unserved load", "Initial unserved load (MW)"),
        ("Initial critical unserved load", "Initial critical\nunserved load (MW)"),
        (
            "Initial economic interruption-damage rate",
            "Initial economic interruption-\ndamage rate\n(million USD h$^{-1}$)",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.15))
    point_rng = np.random.default_rng(2304)
    for axis, (metric, ylabel), panel_label in zip(
        axes.ravel(), metric_panels, ["(a)", "(b)", "(c)", "(d)"]
    ):
        panel = severity.loc[severity["metric"].eq(metric)]
        values_by_group = [
            panel.loc[panel["display_group"].eq(group), "value"].dropna().to_numpy(float)
            for group in groups
        ]
        positions = np.arange(1, len(groups) + 1)
        violins = axis.violinplot(
            values_by_group,
            positions=positions,
            widths=0.58,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, group in zip(violins["bodies"], groups):
            body.set_facecolor(colors[group])
            body.set_edgecolor(colors[group])
            body.set_alpha(0.22)
            body.set_linewidth(0.7)
        boxes = axis.boxplot(
            values_by_group,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "lw": 1.0},
        )
        for patch, group in zip(boxes["boxes"], groups):
            patch.set_facecolor("white")
            patch.set_edgecolor(colors[group])
            patch.set_linewidth(0.9)
        for artist_key in ["whiskers", "caps"]:
            for artist in boxes[artist_key]:
                artist.set_color("#555555")
                artist.set_linewidth(0.75)
        for position, group in zip(positions, groups):
            points = panel.loc[panel["display_group"].eq(group) & panel["displayed_as_point"]]
            jitter = point_rng.uniform(-0.10, 0.10, size=len(points))
            is_library = group == "Benchmark-hard library"
            axis.scatter(
                np.full(len(points), position) + jitter,
                points["value"],
                s=6 if is_library else 9,
                color=colors[group],
                alpha=0.13 if is_library else 0.34,
                linewidths=0,
                zorder=3,
            )
            count = int(panel.loc[panel["display_group"].eq(group), "scenario_key"].nunique())
            axis.text(
                position,
                0.975,
                f"n = {count}",
                ha="center",
                va="top",
                fontsize=6.8,
                color="#333333",
                transform=axis.get_xaxis_transform(),
            )
        axis.set_ylabel(ylabel)
        if metric == "Initial economic interruption-damage rate":
            axis.yaxis.label.set_fontsize(7.5)
        axis.set_xticks(positions, tick_labels, fontsize=7.2)
        axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        _panel(axis, panel_label, x=-0.09, y=1.02)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.93, wspace=0.37, hspace=0.45)
    _export(fig, d, "Figure02_Scenario_severity")

    _write_csv(severity, d["plot_ready"] / "Figure02_scenario_severity_economic_long.csv")
    _write_csv(summary, d["plot_ready"] / "Figure02_scenario_severity_economic_summary.csv")
    _write_csv(audit, d["validation"] / "Figure02_economic_damage_recalculation_audit.csv")
    metric_definition = pd.DataFrame(
        [
            {
                "metric": "Initial economic interruption-damage rate",
                "unit": "million USD h^-1",
                "definition": "Economic damage accumulated over the first outage hour, divided by one hour",
                "noncritical_valuation": "sector- and duration-dependent customer damage function",
                "critical_valuation": f"{float(config['valuation']['critical_value_usd_per_kwh']):g} USD per unserved kWh",
                "support_treatment": "No anchor support; Figure 2 characterizes scenario severity before intervention",
                "legacy_voll_field_used": False,
            }
        ]
    )
    _write_csv(metric_definition, d["validation"] / "Figure02_metric_definition_audit.csv")
    reuse_inventory_path = d["validation"] / "unchanged_figure_reuse_inventory.csv"
    if reuse_inventory_path.exists():
        reuse_inventory = pd.read_csv(reuse_inventory_path)
        reuse_inventory = reuse_inventory.loc[
            ~reuse_inventory["figure"].eq("Figure02_Scenario_severity")
        ]
        _write_csv(reuse_inventory, reuse_inventory_path)
    output_rows = []
    for format_name, path in [
        ("pdf", d["figures/pdf"] / "Figure02_Scenario_severity.pdf"),
        ("svg", d["figures/svg"] / "Figure02_Scenario_severity.svg"),
        ("png_600dpi", d["figures/png_600dpi"] / "Figure02_Scenario_severity_600dpi.png"),
        ("preview", d["figures/preview"] / "Figure02_Scenario_severity_preview.png"),
    ]:
        output_rows.append(
            {
                "figure": "Figure02_Scenario_severity",
                "format": format_name,
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "regeneration_reason": "panel (d) updated from legacy VoLL rate to source-based economic interruption-damage rate",
            }
        )
    _write_csv(
        pd.DataFrame(output_rows), d["validation"] / "Figure02_regenerated_output_inventory.csv"
    )

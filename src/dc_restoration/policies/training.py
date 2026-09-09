from __future__ import annotations

import csv
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from dc_restoration.economics.duration_damage import _valuation_parameters
from dc_restoration.economics.interruption_damage import (
    HarmonizedValuator,
    _accrue,
    _state_unserved_kw,
)
from dc_restoration.economics.load_accounting import _load_inventory
from dc_restoration.paths import repository_root
from dc_restoration.policies.encoders import (
    CNNEncoder,
    EncoderActorCritic,
    MLPScorer,
    build_spatial_index,
    candidate_extra_xy,
    rasterize_state,
)
from dc_restoration.policies.graph import (
    EdgeConditionedMessagePassing,
    TrueGNNActorCritic,
    V03GraphAdapter,
    compute_gae_arrays,
    load_true_gnn_checkpoint,
    save_true_gnn_checkpoint,
    to_torch_batch,
)
from dc_restoration.restoration.environment import V03RestorationModel, evaluate_order

ROOT = repository_root()

EPS = 1e-9

GRAPH_KEYS = [
    "node_features",
    "edge_features",
    "global_features",
    "candidate_edge_indices",
    "candidate_features",
    "candidate_mask",
]

RL_LOG_COLUMNS = [
    "architecture",
    "algorithm",
    "global_step",
    "mean_training_harmonized_damage_dollars",
    "mean_training_reward",
    "elapsed_minutes",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "gradient_norm",
    "periodic_main_improvement_vs_greedy_percent",
    "periodic_historical_improvement_vs_greedy_percent",
    "periodic_constraint_total",
    "periodic_episode_count",
]


def _read_rl_log(path: Path) -> pd.DataFrame:
    """Read the old variable-width log without dropping periodic-evaluation fields."""
    if not path.exists():
        return pd.DataFrame(columns=RL_LOG_COLUMNS)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return pd.DataFrame(columns=RL_LOG_COLUMNS)
    if rows[0] not in [RL_LOG_COLUMNS, RL_LOG_COLUMNS[:12]]:
        raise ValueError(f"Unrecognized RL log header: {path}")
    for row in rows[1:]:
        if len(row) not in [12, len(RL_LOG_COLUMNS)]:
            raise ValueError(f"Malformed RL log row in {path}: {len(row)} fields")
    frame = pd.DataFrame(
        [row + [""] * (len(RL_LOG_COLUMNS) - len(row)) for row in rows[1:]],
        columns=RL_LOG_COLUMNS,
    )
    for column in RL_LOG_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _rl_budget_complete(output_dir: Path, target: int) -> bool:
    """An interim best checkpoint is not proof of a completed training budget."""
    final_path = output_dir / "final.pt"
    if not final_path.exists():
        return False
    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("global_step", 0)) >= target


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _dirs(output: Path) -> dict[str, Path]:
    names = [
        "configs",
        "audit",
        "encoder_inputs",
        "checkpoints",
        "logs",
        "diagnostics",
        "development_evaluation",
        "controlled_evaluation",
        "fresh_scenario_bank",
        "fresh_evaluation",
        "manifests",
        "reports",
        "runtime_monitor",
    ]
    result = {"output": output}
    for name in names:
        result[name] = output / name
        result[name].mkdir(parents=True, exist_ok=True)
    return result


def _atomic_checkpoint_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


class HarmonizedObservationAdapter:
    """Builds candidate and graph observations without legacy-VoLL features."""

    def __init__(
        self,
        model: V03RestorationModel,
        valuator: HarmonizedValuator,
        inventory: pd.DataFrame,
    ) -> None:
        from dc_restoration.policies.graph import V03GraphAdapter

        self.model = model
        self.valuator = valuator
        self.inventory = inventory.set_index("load_id", drop=False)
        self.base = V03GraphAdapter(model)
        self.node_ids = self.base.node_ids
        self.node_index = self.base.node_index
        self.edge_ids = self.base.edge_ids
        self.edge_index_map = self.base.edge_index_map
        self.edge_index = self.base.edge_index
        self.static_node = self.base.static_node.copy()
        self.static_edge = self.base.static_edge.copy()
        self.node_loads = self.base.node_loads
        self.spatial_index: Any = None

    def _load_marginal_costs(
        self,
        energy_kwh: dict[str, float],
        unserved_kw: dict[str, float],
    ) -> pd.Series:
        rows = self.inventory
        before = pd.Series(energy_kwh, dtype=float).reindex(rows.index, fill_value=0.0)
        rates = pd.Series(unserved_kw, dtype=float).reindex(rows.index, fill_value=0.0)
        after = before + rates
        baseline = rows["baseline_p_kw"].astype(float).clip(lower=EPS)
        critical = rows["critical"].astype(bool)
        values = pd.Series(0.0, index=rows.index)
        values.loc[critical] = self.valuator.critical_value * rates.loc[critical]
        noncritical = ~critical
        if noncritical.any():
            bdur = before.loc[noncritical].to_numpy(float) / baseline.loc[noncritical].to_numpy(
                float
            )
            adur = after.loc[noncritical].to_numpy(float) / baseline.loc[noncritical].to_numpy(
                float
            )
            subset = rows.loc[noncritical]
            before_cdf = self.valuator._noncritical_cdf(subset, bdur)
            after_cdf = self.valuator._noncritical_cdf(subset, adur)
            values.loc[noncritical] = baseline.loc[noncritical].to_numpy(float) * (
                after_cdf - before_cdf
            )
        return values

    def _score_candidates(
        self,
        damaged: Iterable[str],
        repaired: set[str],
        repair_times: dict[str, float],
        energy_kwh: dict[str, float],
    ) -> tuple[Any, list[dict[str, Any]], dict[str, float], float]:
        remaining = set(map(str, damaged)) - set(map(str, repaired))
        before = self.model.metrics(remaining, "rule_based_anchor")
        before_rates, _ = _state_unserved_kw(self.model, remaining)
        before_damage_rate = self.valuator.marginal_one_hour_cost(energy_kwh, before_rates)
        scores: list[dict[str, Any]] = []
        for edge_id in remaining:
            after_remaining = remaining - {edge_id}
            after = self.model.metrics(after_remaining, "rule_based_anchor")
            after_rates, _ = _state_unserved_kw(self.model, after_remaining)
            after_damage_rate = self.valuator.marginal_one_hour_cost(energy_kwh, after_rates)
            repair_time = float(repair_times.get(edge_id, 8.0))
            damage_gain = float(before_damage_rate - after_damage_rate)
            critical_gain = float(before.critical_unserved_mw - after.critical_unserved_mw)
            unserved_gain = float(before.unserved_mw - after.unserved_mw)
            scores.append(
                {
                    "edge_id": edge_id,
                    "repair_time": repair_time,
                    "immediate_harmonized_gain": damage_gain,
                    "immediate_voll_gain": damage_gain,
                    "immediate_critical_gain": critical_gain,
                    "immediate_unserved_gain": unserved_gain,
                    "harmonized_gain_per_hour": damage_gain / max(repair_time, EPS),
                    "voll_gain_per_hour": damage_gain / max(repair_time, EPS),
                    "critical_gain_per_hour": critical_gain / max(repair_time, EPS),
                    "unserved_gain_per_hour": unserved_gain / max(repair_time, EPS),
                }
            )
        scores.sort(
            key=lambda row: (row["immediate_harmonized_gain"], -row["repair_time"], row["edge_id"]),
            reverse=True,
        )
        return before, scores, before_rates, float(before_damage_rate)

    def _candidate_features(
        self,
        before: Any,
        score: dict[str, Any],
        before_damage_rate: float,
        remaining: int,
        total: int,
    ) -> np.ndarray:
        gain = float(score["immediate_harmonized_gain"])
        repair_time = float(score["repair_time"])
        return np.asarray(
            [
                repair_time / 24.0,
                gain / 1e6,
                float(score["immediate_critical_gain"]) / 10.0,
                float(score["immediate_unserved_gain"]) / 50.0,
                gain / max(repair_time, EPS) / 1e6,
                float(score["critical_gain_per_hour"]) / 10.0,
                float(before.unserved_mw) / 100.0,
                float(before.critical_unserved_mw) / 50.0,
                before_damage_rate / 1e6,
                remaining / 60.0,
                total / 60.0,
                gain / max(before_damage_rate, 1.0),
            ],
            dtype=np.float32,
        )

    def build_state(
        self,
        damaged: Iterable[str],
        repair_times: dict[str, float],
        repaired: set[str],
        accumulated_energy_kwh: dict[str, float],
        k: int,
    ) -> tuple[Any, list[dict[str, Any]], dict[str, np.ndarray]]:
        damaged_list = list(map(str, damaged))
        repaired_set = set(map(str, repaired))
        before, all_scores, current_rates, damage_rate = self._score_candidates(
            damaged_list, repaired_set, repair_times, accumulated_energy_kwh
        )
        scores = all_scores[:k]

        node = np.concatenate(
            [self.static_node.copy(), np.zeros((len(self.node_ids), 4), dtype=np.float32)], axis=1
        )
        # Replace the legacy modeled-VoLL channel with current harmonized marginal damage.
        load_marginal = self._load_marginal_costs(accumulated_energy_kwh, current_rates)
        node_damage = np.zeros(len(self.node_ids), dtype=np.float32)
        for load_id, value in load_marginal.items():
            node_id = str(self.inventory.at[load_id, "node_id"])
            idx = self.node_index.get(node_id)
            if idx is not None:
                node_damage[idx] += float(value) / 1e6
        node[:, 3] = node_damage

        unserved_mw = np.zeros(len(self.node_ids), dtype=np.float32)
        unserved_critical = np.zeros(len(self.node_ids), dtype=np.float32)
        for load_id, rate_kw in current_rates.items():
            idx = self.node_index.get(str(self.inventory.at[load_id, "node_id"]))
            if idx is None:
                continue
            unserved_mw[idx] += float(rate_kw) / 1000.0
            if bool(self.inventory.at[load_id, "critical"]):
                unserved_critical[idx] += float(rate_kw) / 1000.0
        node[:, -4] = unserved_mw / 100.0
        node[:, -3] = unserved_critical / 50.0

        edge = np.concatenate(
            [self.static_edge.copy(), np.zeros((len(self.edge_ids), 3), dtype=np.float32)], axis=1
        )
        remaining_edges = set(damaged_list) - repaired_set
        for edge_id in remaining_edges:
            edge_idx = self.edge_index_map.get(edge_id)
            if edge_idx is None:
                continue
            edge[edge_idx, -3] = 1.0
            edge[edge_idx, -1] = float(repair_times.get(edge_id, 8.0)) / 24.0
            u, v = self.model.edge_endpoints.get(edge_id, ("", ""))
            if str(u) in self.node_index:
                node[self.node_index[str(u)], -2] += 0.1
            if str(v) in self.node_index:
                node[self.node_index[str(v)], -2] += 0.1
        for edge_id in repaired_set:
            edge_idx = self.edge_index_map.get(edge_id)
            if edge_idx is None:
                continue
            edge[edge_idx, -2] = 1.0
            u, v = self.model.edge_endpoints.get(edge_id, ("", ""))
            if str(u) in self.node_index:
                node[self.node_index[str(u)], -1] += 0.1
            if str(v) in self.node_index:
                node[self.node_index[str(v)], -1] += 0.1

        candidate_indices = np.zeros(k, dtype=np.int64)
        candidate_features = np.zeros((k, 12), dtype=np.float32)
        candidate_mask = np.zeros(k, dtype=bool)
        for index, score in enumerate(scores):
            candidate_indices[index] = self.edge_index_map.get(str(score["edge_id"]), 0)
            candidate_features[index] = self._candidate_features(
                before, score, damage_rate, len(scores), len(damaged_list)
            )
            candidate_mask[index] = True

        global_features = np.asarray(
            [
                float(before.unserved_mw) / 100.0,
                float(before.critical_unserved_mw) / 50.0,
                damage_rate / 1e6,
                len(remaining_edges) / 60.0,
                len(repaired_set) / 60.0,
                len(damaged_list) / 60.0,
                float(before.dc_support_used_mw) / 20.0,
                float(before.dc_anchor_active),
                float(before.outage_islands) / 50.0,
            ],
            dtype=np.float32,
        )
        return (
            before,
            scores,
            {
                "node_features": node.astype(np.float32),
                "edge_features": edge.astype(np.float32),
                "global_features": global_features,
                "candidate_edge_indices": candidate_indices,
                "candidate_features": candidate_features,
                "candidate_mask": candidate_mask,
            },
        )

    def local_features(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        indices = observation["candidate_edge_indices"]
        local_edge = observation["edge_features"][indices]
        return np.concatenate([observation["candidate_features"], local_edge], axis=1).astype(
            np.float32
        )


def _lookahead_cost(
    model: V03RestorationModel,
    valuator: HarmonizedValuator,
    damaged: list[str],
    repair_times: dict[str, float],
    repaired_before: set[str],
    energy_before: dict[str, float],
    order: list[str],
) -> float:
    repaired = set(repaired_before)
    energy = dict(energy_before)
    initial = valuator.cost_from_energy(energy)
    for edge_id in order:
        if edge_id in repaired:
            continue
        rates, _ = _state_unserved_kw(model, set(damaged) - repaired)
        if sum(rates.values()) <= EPS:
            break
        _accrue(energy, rates, float(repair_times.get(edge_id, 8.0)))
        repaired.add(edge_id)
    return valuator.cost_from_energy(energy) - initial


def _build_dataset(
    model: V03RestorationModel,
    adapter: HarmonizedObservationAdapter,
    valuator: HarmonizedValuator,
    source_metadata: pd.DataFrame,
    config: dict[str, Any],
    d: dict[str, Path],
) -> dict[str, Any]:
    output_npz = d["encoder_inputs"] / "harmonized_training_tensors.npz"
    output_meta = d["encoder_inputs"] / "harmonized_training_metadata.csv"
    if output_npz.exists() and output_meta.exists():
        loaded = np.load(output_npz)
        result = {key: loaded[key] for key in loaded.files}
        result["meta"] = pd.read_csv(output_meta)
        return result

    k = int(config["valuation"]["candidate_k"])
    horizon = int(config["valuation"]["lookahead_horizon_repairs"])
    graph_arrays: dict[str, list[np.ndarray]] = {key: [] for key in GRAPH_KEYS}
    local_x: list[np.ndarray] = []
    cnn_x: list[np.ndarray] = []
    labels: list[int] = []
    costs: list[np.ndarray] = []
    weights: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    states = source_metadata.sort_values(["scenario_key", "step", "state_id"]).reset_index(
        drop=True
    )
    grouped = list(states.groupby("scenario_key", sort=False))
    spatial = build_spatial_index(
        model,
        int(config["architectures"]["cnn_grid_resolution"]),
        [
            "damaged_unrepaired_count",
            "damaged_unrepaired_overhead",
            "unserved_load_mw",
            "critical_unserved_load_mw",
            "anchor_zone_load_mw",
            "dc_anchor_export_mw",
        ],
    )
    rasters: list[np.ndarray] = []
    start_time = time.time()

    for scenario_number, (scenario, scenario_states) in enumerate(grouped, start=1):
        damaged = model.scenario_edges(str(scenario))
        repair_times = model.scenario_repair_times(str(scenario))
        repaired: set[str] = set()
        energy: dict[str, float] = {}
        for _, state in scenario_states.sort_values("step").iterrows():
            before, candidate_scores, observation = adapter.build_state(
                damaged, repair_times, repaired, energy, k
            )
            if not candidate_scores or before.unserved_mw <= EPS:
                continue
            immediate_queue = [str(row["edge_id"]) for row in candidate_scores]
            greedy_order = immediate_queue[:horizon]
            greedy_cost = _lookahead_cost(
                model, valuator, damaged, repair_times, repaired, energy, greedy_order
            )
            candidate_cost = np.full(k, 1e6, dtype=np.float32)
            raw_costs: list[float] = []
            for candidate_index, candidate in enumerate(candidate_scores):
                edge_id = str(candidate["edge_id"])
                order = [edge_id] + [item for item in immediate_queue if item != edge_id][
                    : max(0, horizon - 1)
                ]
                value = _lookahead_cost(
                    model, valuator, damaged, repair_times, repaired, energy, order
                )
                raw_costs.append(value)
                candidate_rows.append(
                    {
                        "state_id": int(state["state_id"]),
                        "scenario_key": str(scenario),
                        "split": str(state["split"]),
                        "step": int(state["step"]),
                        "candidate_index": candidate_index,
                        "candidate_edge_id": edge_id,
                        "harmonized_lookahead_cost_dollars": value,
                        "harmonized_greedy_queue_cost_dollars": greedy_cost,
                    }
                )
            label = int(np.argmin(raw_costs))
            denominator = max(abs(greedy_cost), 1.0)
            candidate_cost[: len(raw_costs)] = np.asarray(raw_costs, dtype=np.float32) / denominator
            for key in GRAPH_KEYS:
                graph_arrays[key].append(observation[key])
            local = adapter.local_features(observation)
            local_x.append(local)
            extras = np.zeros((k, 6), dtype=np.float32)
            for index, candidate in enumerate(candidate_scores):
                extras[index] = candidate_extra_xy(spatial, str(candidate["edge_id"]))
            cnn_x.append(np.concatenate([local, extras], axis=1).astype(np.float32))
            rasters.append(rasterize_state(model, spatial, damaged, repaired))
            labels.append(label)
            costs.append(candidate_cost)
            improvement = 100.0 * (greedy_cost - raw_costs[label]) / denominator
            weights.append(
                1.0
                + float(config["bc_training"]["high_improvement_weight"])
                * max(improvement / 100.0, 0.0)
            )
            meta = state.to_dict()
            meta.update(
                {
                    "harmonized_label_edge_id": str(candidate_scores[label]["edge_id"]),
                    "harmonized_immediate_greedy_edge_id": immediate_queue[0],
                    "harmonized_label_improvement_vs_greedy": improvement / 100.0,
                    "harmonized_label_differs_from_greedy": str(candidate_scores[label]["edge_id"])
                    != immediate_queue[0],
                    "harmonized_candidate_count": len(candidate_scores),
                }
            )
            meta_rows.append(meta)

            behavior_edge = str(state["greedy_voll_edge_id"])
            current_rates, _ = _state_unserved_kw(model, set(damaged) - repaired)
            _accrue(energy, current_rates, float(repair_times.get(behavior_edge, 8.0)))
            repaired.add(behavior_edge)
        elapsed = (time.time() - start_time) / 60.0
        print(
            f"[economic-dataset] scenario={scenario_number}/{len(grouped)} "
            f"states={len(labels)} elapsed={elapsed:.1f}m",
            flush=True,
        )

    arrays: dict[str, np.ndarray] = {key: np.stack(value) for key, value in graph_arrays.items()}
    arrays.update(
        {
            "local_x": np.stack(local_x).astype(np.float32),
            "cnn_x": np.stack(cnn_x).astype(np.float32),
            "cnn_raster": np.stack(rasters).astype(np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "costs": np.stack(costs).astype(np.float32),
            "weights": np.asarray(weights, dtype=np.float32),
            "edge_index": adapter.edge_index.astype(np.int64),
        }
    )
    np.savez(output_npz, **arrays)
    metadata = pd.DataFrame(meta_rows)
    _write_csv(metadata, output_meta)
    pd.DataFrame(candidate_rows).to_parquet(
        d["encoder_inputs"] / "harmonized_candidate_lookahead_scores.parquet", index=False
    )
    audit = pd.DataFrame(
        [
            {
                "source_state_count": len(source_metadata),
                "rebuilt_state_count": len(metadata),
                "candidate_score_rows": len(candidate_rows),
                "train_states": int(metadata["split"].eq("train").sum()),
                "validation_states": int(metadata["split"].eq("validation").sum()),
                "test_states": int(metadata["split"].eq("test").sum()),
                "label_differs_from_harmonized_immediate_greedy_fraction": float(
                    metadata["harmonized_label_differs_from_greedy"].mean()
                ),
                "legacy_voll_in_candidate_features": False,
                "legacy_voll_in_global_features": False,
                "legacy_voll_node_channel_replaced": True,
                "lookahead_target_in_policy_inputs": False,
                "fresh_scenarios_used": False,
            }
        ]
    )
    _write_csv(audit, d["diagnostics"] / "harmonized_dataset_and_leakage_audit.csv")
    arrays["meta"] = metadata
    return arrays


def _graph_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "random_seed": int(config["random_seed"]),
        "true_gnn": {
            "hidden_dim": int(config["architectures"]["hidden_dim"]),
            "message_passing_layers": int(config["architectures"]["message_passing_layers"]),
        },
        "runtime": {"candidate_k": int(config["valuation"]["candidate_k"])},
        "bc_training": {
            "epochs": int(config["bc_training"]["true_gnn_epochs"]),
            "patience": int(config["bc_training"]["patience"]),
            "batch_size": int(config["bc_training"]["true_gnn_batch_size"]),
            "learning_rate": float(config["bc_training"]["learning_rate"]),
            "rank_weight": float(config["bc_training"]["rank_weight"]),
            "high_improvement_weight": float(config["bc_training"]["high_improvement_weight"]),
        },
    }


def _make_true_gnn(
    data: dict[str, Any], config: dict[str, Any], device: torch.device
) -> tuple[TrueGNNActorCritic, torch.Tensor, dict[str, int]]:
    dims = {
        "node_dim": int(data["node_features"].shape[-1]),
        "edge_dim": int(data["edge_features"].shape[-1]),
        "global_dim": int(data["global_features"].shape[-1]),
        "candidate_dim": int(data["candidate_features"].shape[-1]),
        "hidden_dim": int(config["architectures"]["hidden_dim"]),
        "message_passing_layers": int(config["architectures"]["message_passing_layers"]),
    }
    policy = TrueGNNActorCritic(
        dims["node_dim"],
        dims["edge_dim"],
        dims["global_dim"],
        dims["candidate_dim"],
        hidden=dims["hidden_dim"],
        layers=dims["message_passing_layers"],
    ).to(device)
    edge_index = torch.as_tensor(data["edge_index"], dtype=torch.long, device=device)
    return policy, edge_index, dims


def _bc_metrics(
    actor: Any,
    architecture: str,
    data: dict[str, Any],
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    if not len(indices):
        return {"top1": np.nan, "top3": np.nan, "top5": np.nan, "regret": np.nan}
    actor.eval()
    predictions: list[int] = []
    labels: list[int] = []
    regrets: list[float] = []
    top3_hits: list[bool] = []
    top5_hits: list[bool] = []
    batch_size = 4 if architecture == "true_gnn" else 128
    edge_index = torch.as_tensor(data["edge_index"], dtype=torch.long, device=device)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ids = indices[start : start + batch_size]
            if architecture == "true_gnn":
                batch = to_torch_batch({key: data[key][ids] for key in GRAPH_KEYS}, device)
                logits = actor(batch, edge_index)["logits"]
            else:
                x_key = "local_x" if architecture == "mlp" else "cnn_x"
                x = torch.as_tensor(data[x_key][ids], dtype=torch.float32, device=device)
                mask = torch.as_tensor(data["candidate_mask"][ids], dtype=torch.bool, device=device)
                raster = None
                if architecture == "cnn":
                    raster = torch.as_tensor(
                        data["cnn_raster"][ids], dtype=torch.float32, device=device
                    )
                logits = actor(x, mask, raster)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            y = data["labels"][ids]
            top3 = logits.topk(min(3, logits.shape[1]), dim=1).indices.detach().cpu().numpy()
            top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices.detach().cpu().numpy()
            valid_counts = data["candidate_mask"][ids].sum(axis=1)
            for row_index, prediction in enumerate(pred):
                labels.append(int(y[row_index]))
                predictions.append(int(prediction))
                top3_hits.append(int(y[row_index]) in top3[row_index])
                top5_hits.append(int(y[row_index]) in top5[row_index])
                valid = int(valid_counts[row_index])
                row_cost = data["costs"][ids[row_index]]
                regrets.append(float(row_cost[prediction] - row_cost[:valid].min()))
    return {
        "top1": float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        "top3": float(np.mean(top3_hits)),
        "top5": float(np.mean(top5_hits)),
        "regret": float(np.mean(regrets)),
    }


def _save_encoder_bc(
    actor: Any,
    architecture: str,
    path: Path,
    data: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "actor_state": actor.state_dict(),
            "candidate_dim": int(
                data["local_x"].shape[-1] if architecture == "mlp" else data["cnn_x"].shape[-1]
            ),
            "raster_shape": tuple(int(value) for value in data["cnn_raster"].shape[1:]),
            "hidden_dim": int(config["architectures"][f"{architecture}_hidden_dim"]),
            "metrics": metrics,
            "valuation": config["valuation"],
        },
        path,
    )


def _train_bc(
    architecture: str,
    data: dict[str, Any],
    config: dict[str, Any],
    d: dict[str, Path],
    device: torch.device,
) -> Path:
    metadata = data["meta"].reset_index(drop=True)
    train_indices = metadata.index[metadata["split"].eq("train")].to_numpy()
    validation_indices = metadata.index[metadata["split"].eq("validation")].to_numpy()
    rng = np.random.default_rng(int(config["random_seed"]))
    if architecture == "true_gnn":
        actor, edge_index, dims = _make_true_gnn(data, config, device)
        batch_size = int(config["bc_training"]["true_gnn_batch_size"])
        epochs = int(config["bc_training"]["true_gnn_epochs"])
    elif architecture == "mlp":
        actor = MLPScorer(
            int(data["local_x"].shape[-1]), int(config["architectures"]["mlp_hidden_dim"])
        ).to(device)
        edge_index = None
        dims = {}
        batch_size = int(config["bc_training"]["encoder_batch_size"])
        epochs = int(config["bc_training"]["encoder_epochs"])
    else:
        channels, grid, _ = data["cnn_raster"].shape[1:]
        actor = CNNEncoder(
            int(data["cnn_x"].shape[-1]),
            int(channels),
            int(grid),
            hidden=int(config["architectures"]["cnn_hidden_dim"]),
            conv_channels=int(config["architectures"]["cnn_conv_channels"]),
            spatial_dim=int(config["architectures"]["cnn_spatial_embedding_dim"]),
        ).to(device)
        edge_index = None
        dims = {}
        batch_size = int(config["bc_training"]["encoder_batch_size"])
        epochs = int(config["bc_training"]["encoder_epochs"])

    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=float(config["bc_training"]["learning_rate"]), weight_decay=1e-4
    )
    checkpoint_dir = d["checkpoints"] / f"{architecture}_harmonized_bc"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_regret = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    stale = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        actor.train()
        shuffled = rng.permutation(train_indices)
        batch_losses: list[float] = []
        for start in range(0, len(shuffled), batch_size):
            ids = shuffled[start : start + batch_size]
            labels = torch.as_tensor(data["labels"][ids], dtype=torch.long, device=device)
            target_costs = torch.as_tensor(data["costs"][ids], dtype=torch.float32, device=device)
            sample_weights = torch.as_tensor(
                data["weights"][ids], dtype=torch.float32, device=device
            )
            if architecture == "true_gnn":
                batch = to_torch_batch({key: data[key][ids] for key in GRAPH_KEYS}, device)
                logits = actor(batch, edge_index)["logits"]
            else:
                x_key = "local_x" if architecture == "mlp" else "cnn_x"
                x = torch.as_tensor(data[x_key][ids], dtype=torch.float32, device=device)
                mask = torch.as_tensor(data["candidate_mask"][ids], dtype=torch.bool, device=device)
                raster = (
                    None
                    if architecture == "mlp"
                    else torch.as_tensor(
                        data["cnn_raster"][ids], dtype=torch.float32, device=device
                    )
                )
                logits = actor(x, mask, raster)
            ce = F.cross_entropy(logits, labels, reduction="none")
            probabilities = torch.softmax(logits, dim=1)
            expected = (probabilities * target_costs).sum(dim=1)
            best = target_costs.gather(1, labels[:, None]).squeeze(1)
            ranking = torch.relu(expected - best)
            loss = (
                (ce + float(config["bc_training"]["rank_weight"]) * ranking) * sample_weights
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation = _bc_metrics(actor, architecture, data, validation_indices, device)
        training = _bc_metrics(actor, architecture, data, train_indices, device)
        row = {
            "architecture": architecture,
            "epoch": epoch,
            "loss": float(np.mean(batch_losses)),
            **{f"train_{key}": value for key, value in training.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        logs.append(row)
        if validation["regret"] < best_regret - 1e-7:
            best_regret = validation["regret"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch % int(config["bc_training"]["checkpoint_every_epochs"]) == 0:
            if architecture == "true_gnn":
                save_true_gnn_checkpoint(
                    actor,
                    edge_index,
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    _graph_config(config),
                    dims,
                    epoch,
                    row,
                )
            else:
                _save_encoder_bc(
                    actor, architecture, checkpoint_dir / f"epoch_{epoch:03d}.pt", data, config, row
                )
        print(
            f"[economic-bc-{architecture}] epoch={epoch}/{epochs} "
            f"val_top1={validation['top1']:.3f} val_regret={validation['regret']:.6f}",
            flush=True,
        )
        if stale >= int(config["bc_training"]["patience"]):
            break
    if best_state is not None:
        actor.load_state_dict(best_state)
    best_path = checkpoint_dir / "best_validation_regret.pt"
    best_metrics = _bc_metrics(actor, architecture, data, validation_indices, device)
    if architecture == "true_gnn":
        save_true_gnn_checkpoint(
            actor, edge_index, best_path, _graph_config(config), dims, len(logs), best_metrics
        )
    else:
        _save_encoder_bc(actor, architecture, best_path, data, config, best_metrics)
    _write_csv(pd.DataFrame(logs), d["logs"] / f"{architecture}_harmonized_bc_training_log.csv")
    return best_path


def _load_encoder_bc(
    path: Path, config: dict[str, Any], device: torch.device
) -> tuple[EncoderActorCritic, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = str(checkpoint["architecture"])
    candidate_dim = int(checkpoint["candidate_dim"])
    if architecture == "mlp":
        actor = MLPScorer(candidate_dim, int(checkpoint["hidden_dim"]))
    else:
        channels, grid, _ = checkpoint["raster_shape"]
        actor = CNNEncoder(
            candidate_dim,
            int(channels),
            int(grid),
            hidden=int(checkpoint["hidden_dim"]),
            conv_channels=int(config["architectures"]["cnn_conv_channels"]),
            spatial_dim=int(config["architectures"]["cnn_spatial_embedding_dim"]),
        )
    actor.load_state_dict(checkpoint["actor_state"])
    policy = EncoderActorCritic(actor, int(checkpoint["hidden_dim"])).to(device)
    return policy, checkpoint


def _graph_dependence(
    policy: TrueGNNActorCritic,
    edge_index: torch.Tensor,
    data: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    executed: list[str] = []
    hooks = []
    for name, module in policy.named_modules():
        if isinstance(module, EdgeConditionedMessagePassing):
            hooks.append(
                module.register_forward_hook(lambda _m, _i, _o, n=name: executed.append(n))
            )
    rows: list[dict[str, Any]] = []
    policy.eval()
    for state_index in range(min(10, len(data["labels"]))):
        batch = to_torch_batch({key: data[key][[state_index]] for key in GRAPH_KEYS}, device)
        with torch.no_grad():
            base = policy(batch, edge_index)["logits"][0]
            generator = torch.Generator(device=device).manual_seed(32000 + state_index)
            permutation = torch.randperm(edge_index.shape[1], generator=generator, device=device)
            perturbed_edge_index = edge_index.clone()
            # Rewire destinations while leaving all node/edge/global tensors fixed.
            perturbed_edge_index[1] = edge_index[1, permutation]
            changed = policy(batch, perturbed_edge_index)["logits"][0]
        mask = batch["candidate_mask"][0]
        rows.append(
            {
                "state_index": state_index,
                "maximum_logit_change": float((base[mask] - changed[mask]).abs().max().cpu()),
                "action_changed": int(base[mask].argmax()) != int(changed[mask].argmax()),
            }
        )
    for hook in hooks:
        hook.remove()
    frame = pd.DataFrame(rows)
    frame["message_passing_hook_calls"] = len(executed)
    frame["message_passing_verified"] = len(executed) > 0
    frame["graph_perturbation_changes_logits"] = frame["maximum_logit_change"].gt(1e-7)
    return frame


@dataclass
class HarmonizedTransition:
    graph_observation: Optional[dict[str, np.ndarray]]
    candidate_x: Optional[np.ndarray]
    candidate_mask: np.ndarray
    raster: Optional[np.ndarray]
    action: int
    log_probability: float
    value: float
    reward: float
    done: bool


def _entropy_coefficient(config: dict[str, Any], step: int) -> float:
    value = float(config["rl_training"]["entropy_schedule"][0][1])
    for boundary, scheduled in config["rl_training"]["entropy_schedule"]:
        if step >= int(boundary):
            value = float(scheduled)
    return value


def _spatial_index(model: V03RestorationModel, config: dict[str, Any]) -> Any:
    return build_spatial_index(
        model,
        int(config["architectures"]["cnn_grid_resolution"]),
        [
            "damaged_unrepaired_count",
            "damaged_unrepaired_overhead",
            "unserved_load_mw",
            "critical_unserved_load_mw",
            "anchor_zone_load_mw",
            "dc_anchor_export_mw",
        ],
    )


def _encoder_state_arrays(
    architecture: str,
    adapter: HarmonizedObservationAdapter,
    observation: dict[str, np.ndarray],
    scores: list[dict[str, Any]],
    spatial: Any,
    damaged: list[str],
    repaired: set[str],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    local = adapter.local_features(observation)
    if architecture == "mlp":
        return local, observation["candidate_mask"], None
    extras = np.zeros((local.shape[0], 6), dtype=np.float32)
    for index, score in enumerate(scores):
        extras[index] = candidate_extra_xy(spatial, str(score["edge_id"]))
    candidate_x = np.concatenate([local, extras], axis=1).astype(np.float32)
    raster = rasterize_state(adapter.model, spatial, damaged, repaired)
    return candidate_x, observation["candidate_mask"], raster


def _policy_action(
    architecture: str,
    policy: Any,
    edge_index: Optional[torch.Tensor],
    observation: dict[str, np.ndarray],
    candidate_x: Optional[np.ndarray],
    candidate_mask: np.ndarray,
    raster: Optional[np.ndarray],
    device: torch.device,
    deterministic: bool,
) -> tuple[int, float, float]:
    if architecture == "true_gnn":
        batch = to_torch_batch({key: observation[key][None, ...] for key in GRAPH_KEYS}, device)
        action, log_probability, _entropy, value = policy.act(
            batch, edge_index, deterministic=deterministic
        )
    else:
        x = torch.as_tensor(candidate_x[None, ...], dtype=torch.float32, device=device)
        mask = torch.as_tensor(candidate_mask[None, ...], dtype=torch.bool, device=device)
        raster_tensor = None
        if raster is not None:
            raster_tensor = torch.as_tensor(raster[None, ...], dtype=torch.float32, device=device)
        action, log_probability, _entropy, value = policy.act(
            x, mask, raster_tensor, deterministic=deterministic
        )
    return int(action.item()), float(log_probability.item()), float(value.item())


def _rollout_harmonized(
    architecture: str,
    model: V03RestorationModel,
    adapter: HarmonizedObservationAdapter,
    valuator: HarmonizedValuator,
    scenario: str,
    config: dict[str, Any],
    device: torch.device,
    policy: Optional[Any] = None,
    edge_index: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    collect_transitions: bool = False,
    greedy_cost: Optional[float] = None,
    random_seed: Optional[int] = None,
    decision_adapter: Optional[Any] = None,
) -> tuple[dict[str, Any], list[HarmonizedTransition]]:
    k = int(config["valuation"]["candidate_k"])
    damaged = model.scenario_edges(scenario)
    repair_times = model.scenario_repair_times(scenario)
    repaired: set[str] = set()
    energy: dict[str, float] = {}
    sequence: list[str] = []
    transitions: list[HarmonizedTransition] = []
    initial_rates, _ = _state_unserved_kw(model, set(damaged))
    initial_damage_rate = max(valuator.marginal_one_hour_cost({}, initial_rates), EPS)
    initial_physical = model.metrics(set(damaged), "rule_based_anchor")
    rng = random.Random(random_seed if random_seed is not None else int(config["random_seed"]))
    decision_adapter = decision_adapter or adapter
    if architecture == "cnn" and decision_adapter.spatial_index is None:
        decision_adapter.spatial_index = _spatial_index(model, config)
    spatial = decision_adapter.spatial_index if architecture == "cnn" else None
    start_time = time.time()

    while len(repaired) < len(damaged):
        if isinstance(decision_adapter, HarmonizedObservationAdapter):
            before, scores, observation = decision_adapter.build_state(
                damaged, repair_times, repaired, energy, k
            )
        else:
            before, scores, observation = decision_adapter.build_state(
                damaged, repair_times, repaired, k
            )
        if not scores or before.unserved_mw <= EPS:
            break
        candidate_x: Optional[np.ndarray] = None
        raster: Optional[np.ndarray] = None
        mask = observation["candidate_mask"]
        if architecture in {"mlp", "cnn"}:
            candidate_x, mask, raster = _encoder_state_arrays(
                architecture, decision_adapter, observation, scores, spatial, damaged, repaired
            )
        if architecture == "greedy":
            action_index, log_probability, value = 0, 0.0, 0.0
        elif architecture == "random":
            valid = int(mask.sum())
            action_index, log_probability, value = rng.randrange(max(valid, 1)), 0.0, 0.0
        else:
            action_index, log_probability, value = _policy_action(
                architecture,
                policy,
                edge_index,
                observation,
                candidate_x,
                mask,
                raster,
                device,
                deterministic,
            )
        action_index = min(action_index, len(scores) - 1)
        chosen = str(scores[action_index]["edge_id"])
        if chosen in repaired:
            raise RuntimeError(f"Invalid repeated action {chosen} in {scenario}")
        current_rates, _ = _state_unserved_kw(model, set(damaged) - repaired)
        after_remaining = (set(damaged) - repaired) - {chosen}
        after_rates, _ = _state_unserved_kw(model, after_remaining)
        before_damage_rate = valuator.marginal_one_hour_cost(energy, current_rates)
        after_damage_rate = valuator.marginal_one_hour_cost(energy, after_rates)
        after_physical = model.metrics(after_remaining, "rule_based_anchor")
        weights = config["rl_training"]["reward"]
        reward = (
            float(weights["harmonized_damage_gain_weight"])
            * (before_damage_rate - after_damage_rate)
            / initial_damage_rate
            + float(weights["unserved_gain_weight"])
            * (before.unserved_mw - after_physical.unserved_mw)
            / max(initial_physical.unserved_mw, EPS)
            + float(weights["critical_gain_weight"])
            * (before.critical_unserved_mw - after_physical.critical_unserved_mw)
            / max(initial_physical.critical_unserved_mw, EPS)
            - float(weights["remaining_harmonized_damage_weight"])
            * after_damage_rate
            / initial_damage_rate
            - float(weights["step_penalty"])
        )
        if collect_transitions:
            transitions.append(
                HarmonizedTransition(
                    graph_observation=observation if architecture == "true_gnn" else None,
                    candidate_x=candidate_x,
                    candidate_mask=mask,
                    raster=raster,
                    action=action_index,
                    log_probability=log_probability,
                    value=value,
                    reward=float(reward),
                    done=False,
                )
            )
        duration = float(repair_times.get(chosen, 8.0))
        _accrue(energy, current_rates, duration)
        repaired.add(chosen)
        sequence.append(chosen)

    economic_damage = valuator.cost_from_energy(energy)
    if transitions:
        if greedy_cost is not None:
            bonus = (
                float(config["rl_training"]["reward"]["episode_relative_weight"])
                * (greedy_cost - economic_damage)
                / max(greedy_cost, 1.0)
            )
            threshold = float(
                config["rl_training"]["reward"]["worse_than_greedy_penalty_threshold"]
            )
            if economic_damage > greedy_cost * (1.0 + threshold):
                bonus -= float(config["rl_training"]["reward"]["worse_than_greedy_penalty"])
            transitions[-1].reward += bonus
        transitions[-1].done = True
    physical = evaluate_order(model, damaged, repair_times, sequence, "rule_based_anchor")
    metrics = {
        **physical,
        "scenario_key": scenario,
        "harmonized_economic_damage_dollars": float(economic_damage),
        "repair_sequence_edge_ids": json.dumps(sequence),
        "runtime_seconds": time.time() - start_time,
        "training_reward_sum": float(sum(item.reward for item in transitions)),
        "transition_count": len(transitions),
    }
    return metrics, transitions


def _stack_harmonized_transitions(
    architecture: str, transitions: list[HarmonizedTransition]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "actions": np.asarray([item.action for item in transitions], dtype=np.int64),
        "old_log_probability": np.asarray(
            [item.log_probability for item in transitions], dtype=np.float32
        ),
        "values": np.asarray([item.value for item in transitions], dtype=np.float32),
        "rewards": np.asarray([item.reward for item in transitions], dtype=np.float32),
        "dones": np.asarray([item.done for item in transitions], dtype=bool),
    }
    if architecture == "true_gnn":
        for key in GRAPH_KEYS:
            result[key] = np.stack([item.graph_observation[key] for item in transitions])
    else:
        result["candidate_x"] = np.stack([item.candidate_x for item in transitions])
        result["candidate_mask"] = np.stack([item.candidate_mask for item in transitions])
        if architecture == "cnn":
            result["raster"] = np.stack([item.raster for item in transitions])
    return result


def _save_rl_checkpoint(
    architecture: str,
    algorithm: str,
    policy: Any,
    edge_index: Optional[torch.Tensor],
    path: Path,
    config: dict[str, Any],
    base_checkpoint: dict[str, Any],
    step: int,
    metrics: dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    sampler_rng: Optional[random.Random] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if architecture == "true_gnn":
        save_true_gnn_checkpoint(
            policy,
            edge_index,
            path,
            _graph_config(config),
            base_checkpoint["dims"],
            step,
            {**metrics, "algorithm": algorithm, "harmonized_objective": True},
        )
    else:
        torch.save(
            {
                **base_checkpoint,
                "architecture": architecture,
                "algorithm": algorithm,
                "actor_state": policy.actor.state_dict(),
                "value_state": policy.value.state_dict(),
                "global_step": step,
                "metrics": metrics,
                "harmonized_objective": True,
            },
            path,
        )
    if optimizer is not None:
        torch.save(
            {
                "optimizer_state": optimizer.state_dict(),
                "global_step": int(step),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "sampler_rng_state": sampler_rng.getstate() if sampler_rng is not None else None,
            },
            path.with_suffix(path.suffix + ".optimizer"),
        )


def _load_bc_policy(
    architecture: str,
    config: dict[str, Any],
    d: dict[str, Path],
    device: torch.device,
) -> tuple[Any, Optional[torch.Tensor], dict[str, Any]]:
    path = d["checkpoints"] / f"{architecture}_harmonized_bc" / "best_validation_regret.pt"
    if architecture == "true_gnn":
        return load_true_gnn_checkpoint(path, device)
    policy, checkpoint = _load_encoder_bc(path, config, device)
    return policy, None, checkpoint


def _greedy_reference(
    model: V03RestorationModel,
    adapter: HarmonizedObservationAdapter,
    valuator: HarmonizedValuator,
    scenarios: list[str],
    config: dict[str, Any],
    d: dict[str, Path],
    device: torch.device,
) -> dict[str, float]:
    path = d["development_evaluation"] / "harmonized_greedy_training_reference.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    complete = set(existing["scenario_key"].astype(str)) if not existing.empty else set()
    rows = []
    for index, scenario in enumerate(scenarios, start=1):
        if scenario in complete:
            continue
        metrics, _ = _rollout_harmonized(
            "greedy", model, adapter, valuator, scenario, config, device
        )
        rows.append(metrics)
        pd.DataFrame([metrics]).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"[economic-greedy-reference] {index}/{len(scenarios)}", flush=True)
    frame = pd.read_csv(path)
    return dict(
        zip(
            frame["scenario_key"].astype(str),
            frame["harmonized_economic_damage_dollars"].astype(float),
        )
    )


def _development_manifest(config: dict[str, Any]) -> pd.DataFrame:
    manifest = pd.read_csv(_resolve(config["paths"]["fixed_controlled_manifest"]))
    rows = []
    count = int(config["rl_training"]["periodic_eval_episodes_per_subset"])
    for subset in config["subsets"]["development"]:
        rows.append(manifest.loc[manifest["eval_subset"].eq(subset)].head(count))
    return pd.concat(rows, ignore_index=True)


def _periodic_evaluation(
    architecture: str,
    algorithm: str,
    policy: Any,
    edge_index: Optional[torch.Tensor],
    step: int,
    model: V03RestorationModel,
    adapter: HarmonizedObservationAdapter,
    valuator: HarmonizedValuator,
    config: dict[str, Any],
    d: dict[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    manifest = _development_manifest(config)
    unique_scenarios = manifest["scenario_key"].astype(str).drop_duplicates().tolist()
    greedy_costs = _greedy_reference(model, adapter, valuator, unique_scenarios, config, d, device)
    rows = []
    cache: dict[str, dict[str, Any]] = {}
    for _, membership in manifest.iterrows():
        scenario = str(membership["scenario_key"])
        if scenario not in cache:
            cache[scenario], _ = _rollout_harmonized(
                architecture,
                model,
                adapter,
                valuator,
                scenario,
                config,
                device,
                policy=policy,
                edge_index=edge_index,
                deterministic=True,
            )
        metrics = cache[scenario]
        greedy_cost = greedy_costs[scenario]
        rows.append(
            {
                "architecture": architecture,
                "algorithm": algorithm,
                "global_step": step,
                "eval_subset": str(membership["eval_subset"]),
                "scenario_key": scenario,
                "greedy_harmonized_damage_dollars": greedy_cost,
                "policy_harmonized_damage_dollars": metrics["harmonized_economic_damage_dollars"],
                "improvement_vs_greedy_percent": 100.0
                * (greedy_cost - metrics["harmonized_economic_damage_dollars"])
                / max(greedy_cost, 1.0),
                "invalid_action_count": int(metrics.get("invalid_action_count", 0)),
                "export_cap_violation_count": int(metrics.get("export_cap_violation_count", 0)),
                "support_outside_anchor_zone_count": int(
                    metrics.get("support_outside_anchor_zone_count", 0)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    path = d["development_evaluation"] / "periodic_physical_evaluation.csv"
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
    main = frame.loc[frame["eval_subset"].isin(config["subsets"]["main"])]
    historical = frame.loc[frame["eval_subset"].eq("benchmark_hard_historical_nonzero")]
    return {
        "periodic_main_improvement_vs_greedy_percent": float(
            main["improvement_vs_greedy_percent"].mean()
        ),
        "periodic_historical_improvement_vs_greedy_percent": float(
            historical["improvement_vs_greedy_percent"].mean()
        ),
        "periodic_constraint_total": int(
            frame[
                [
                    "invalid_action_count",
                    "export_cap_violation_count",
                    "support_outside_anchor_zone_count",
                ]
            ]
            .sum()
            .sum()
        ),
        "periodic_episode_count": len(frame),
    }


def _train_rl(
    architecture: str,
    algorithm: str,
    data: dict[str, Any],
    config: dict[str, Any],
    d: dict[str, Path],
    device: torch.device,
) -> Path:
    policy, edge_index, base_checkpoint = _load_bc_policy(architecture, config, d, device)
    model, _inventory, valuator, adapter = _load_context(config)
    training_scenarios = (
        data["meta"]
        .loc[data["meta"]["split"].eq("train"), "scenario_key"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    greedy_costs = _greedy_reference(
        model, adapter, valuator, training_scenarios, config, d, device
    )
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=float(config["rl_training"]["learning_rate"])
    )
    rng = random.Random(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]))
    torch.manual_seed(int(config["random_seed"]))
    target = int(config["rl_training"]["target_steps"])
    rollout_target = int(
        config["rl_training"]["n_steps_a2c" if algorithm == "a2c" else "n_steps_ppo"]
    )
    output_dir = d["checkpoints"] / f"{architecture}_harmonized_{algorithm}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "initial.pt").exists():
        _save_rl_checkpoint(
            architecture,
            algorithm,
            policy,
            edge_index,
            output_dir / "initial.pt",
            config,
            base_checkpoint,
            0,
            {},
        )
    legacy_log_path = d["logs"] / f"{architecture}_harmonized_{algorithm}_training_log.csv"
    log_path = d["logs"] / f"{architecture}_harmonized_{algorithm}_training_log_v2.csv"
    prior_log = _read_rl_log(log_path if log_path.exists() else legacy_log_path)
    global_step = 0
    best_score = -float("inf")
    best_guardrail = -float("inf")
    next_periodic = int(config["rl_training"]["periodic_eval_every_steps"])
    start_time = time.time()

    step_checkpoints = sorted(output_dir.glob("step_*.pt"))
    if step_checkpoints and not _rl_budget_complete(output_dir, target):
        resume_path = step_checkpoints[-1]
        policy, edge_index, resume_checkpoint = _load_rl_policy_path(
            architecture, resume_path, config, device
        )
        global_step = int(resume_checkpoint.get("global_step", 0))
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(config["rl_training"]["learning_rate"])
        )
        optimizer_path = resume_path.with_suffix(resume_path.suffix + ".optimizer")
        if not optimizer_path.exists():
            raise RuntimeError(f"Missing optimizer state for safe resume: {optimizer_path}")
        resume_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        if int(resume_state["global_step"]) != global_step:
            raise RuntimeError("Model/optimizer resume steps disagree")
        optimizer.load_state_dict(resume_state["optimizer_state"])
        exact_rng_resume = resume_state.get("sampler_rng_state") is not None
        if exact_rng_resume:
            rng.setstate(resume_state["sampler_rng_state"])
            np.random.set_state(resume_state["numpy_rng_state"])
            torch.set_rng_state(resume_state["torch_rng_state"])
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])
        else:
            # Legacy saves omitted RNG state; disclose a deterministic new stream.
            recovery_seed = int(config["random_seed"]) + global_step
            rng.seed(recovery_seed)
            np.random.seed(recovery_seed)
            torch.manual_seed(recovery_seed)
        resume_audit = {
            "checkpoint": str(resume_path),
            "resume_step": global_step,
            "last_logged_step": int(prior_log["global_step"].max()) if not prior_log.empty else 0,
            "optimizer_restored": True,
            "exact_rng_resume": exact_rng_resume,
            "rng_note": "restored"
            if exact_rng_resume
            else "Legacy RNG state absent; restart seed = configured seed + saved step. Not bitwise continuation.",
            "objective_or_hyperparameters_changed": False,
        }
        audit_path = (
            d["runtime_monitor"] / f"{architecture}_{algorithm}_resume_{time.time_ns()}.json"
        )
        audit_path.write_text(json.dumps(resume_audit, indent=2), encoding="utf-8")
        next_periodic = (
            global_step // int(config["rl_training"]["periodic_eval_every_steps"]) + 1
        ) * int(config["rl_training"]["periodic_eval_every_steps"])
        print(
            f"[economic-{architecture}-{algorithm}] resuming from {resume_path.name} at step {global_step}",
            flush=True,
        )
        if not prior_log.empty:
            if "periodic_main_improvement_vs_greedy_percent" in prior_log:
                valid = prior_log.dropna(subset=["periodic_main_improvement_vs_greedy_percent"])
                if not valid.empty:
                    best_score = float(valid["periodic_main_improvement_vs_greedy_percent"].max())
                    guardrail_values = valid["periodic_main_improvement_vs_greedy_percent"] + valid[
                        "periodic_historical_improvement_vs_greedy_percent"
                    ].fillna(0.0)
                    best_guardrail = float(guardrail_values.max())

    # Retain abandoned post-checkpoint rows in the old log or an archived v2 copy.
    if log_path.exists() and (prior_log["global_step"] > global_step).any():
        shutil.copy2(
            log_path, log_path.with_name(f"{log_path.stem}_before_resume_{time.time_ns()}.csv")
        )
    prior_log = prior_log.loc[prior_log["global_step"] <= global_step].copy()
    _write_csv(prior_log.reindex(columns=RL_LOG_COLUMNS), log_path)

    while global_step < target:
        transitions: list[HarmonizedTransition] = []
        episode_metrics: list[dict[str, Any]] = []
        while len(transitions) < rollout_target and global_step < target:
            scenario = rng.choice(training_scenarios)
            metrics, episode = _rollout_harmonized(
                architecture,
                model,
                adapter,
                valuator,
                scenario,
                config,
                device,
                policy=policy,
                edge_index=edge_index,
                deterministic=False,
                collect_transitions=True,
                greedy_cost=greedy_costs[scenario],
            )
            transitions.extend(episode)
            episode_metrics.append(metrics)
            global_step += len(episode)
        if not transitions:
            raise RuntimeError(f"No transitions collected for {architecture}-{algorithm}")
        stacked = _stack_harmonized_transitions(architecture, transitions)
        advantages, returns = compute_gae_arrays(
            stacked["rewards"],
            stacked["values"],
            stacked["dones"],
            gamma=float(config["rl_training"]["gamma"]),
            lam=float(config["rl_training"]["gae_lambda"]),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        indices = np.arange(len(transitions))
        epochs = int(config["rl_training"]["ppo_epochs_per_update"]) if algorithm == "ppo" else 1
        minibatch_size = int(config["rl_training"]["minibatch_size"])
        losses = []
        stop_for_kl = False
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), minibatch_size):
                ids = indices[start : start + minibatch_size]
                if architecture == "true_gnn":
                    batch = to_torch_batch({key: stacked[key][ids] for key in GRAPH_KEYS}, device)
                    output = policy(batch, edge_index)
                    logits, values = output["logits"], output["value"]
                else:
                    x = torch.as_tensor(
                        stacked["candidate_x"][ids], dtype=torch.float32, device=device
                    )
                    mask = torch.as_tensor(
                        stacked["candidate_mask"][ids], dtype=torch.bool, device=device
                    )
                    raster = None
                    if architecture == "cnn":
                        raster = torch.as_tensor(
                            stacked["raster"][ids], dtype=torch.float32, device=device
                        )
                    logits, values = policy.forward(x, mask, raster)
                distribution = torch.distributions.Categorical(logits=logits)
                actions = torch.as_tensor(stacked["actions"][ids], dtype=torch.long, device=device)
                log_probability = distribution.log_prob(actions)
                old_log_probability = torch.as_tensor(
                    stacked["old_log_probability"][ids], dtype=torch.float32, device=device
                )
                advantage = torch.as_tensor(advantages[ids], dtype=torch.float32, device=device)
                target_return = torch.as_tensor(returns[ids], dtype=torch.float32, device=device)
                if algorithm == "ppo":
                    ratio = torch.exp(log_probability - old_log_probability)
                    clip = float(config["rl_training"]["ppo_clip_range"])
                    policy_loss = -torch.min(
                        ratio * advantage,
                        torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage,
                    ).mean()
                    approximate_kl = float(
                        (old_log_probability - log_probability).mean().detach().cpu()
                    )
                else:
                    policy_loss = -(log_probability * advantage).mean()
                    approximate_kl = 0.0
                value_loss = F.mse_loss(values, target_return)
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + float(config["rl_training"]["value_loss_coefficient"]) * value_loss
                    - _entropy_coefficient(config, global_step) * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float(config["rl_training"]["max_grad_norm"])
                )
                optimizer.step()
                losses.append(
                    {
                        "loss": float(loss.detach().cpu()),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy.detach().cpu()),
                        "approximate_kl": approximate_kl,
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                    }
                )
                if algorithm == "ppo" and approximate_kl > float(
                    config["rl_training"]["ppo_target_kl"]
                ):
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break
        row: dict[str, Any] = {
            "architecture": architecture,
            "algorithm": algorithm,
            "global_step": global_step,
            "mean_training_harmonized_damage_dollars": float(
                np.mean([item["harmonized_economic_damage_dollars"] for item in episode_metrics])
            ),
            "mean_training_reward": float(
                np.mean([item["training_reward_sum"] for item in episode_metrics])
            ),
            "elapsed_minutes": (time.time() - start_time) / 60.0,
            **{key: float(np.mean([item[key] for item in losses])) for key in losses[0]},
        }
        if global_step >= next_periodic:
            periodic = _periodic_evaluation(
                architecture,
                algorithm,
                policy,
                edge_index,
                global_step,
                model,
                adapter,
                valuator,
                config,
                d,
                device,
            )
            row.update(periodic)
            next_periodic += int(config["rl_training"]["periodic_eval_every_steps"])
            score = float(periodic["periodic_main_improvement_vs_greedy_percent"])
            historical = float(periodic["periodic_historical_improvement_vs_greedy_percent"])
            constraints = int(periodic["periodic_constraint_total"])
            if constraints == 0 and score > best_score:
                best_score = score
                _save_rl_checkpoint(
                    architecture,
                    algorithm,
                    policy,
                    edge_index,
                    output_dir / "best_vs_greedy.pt",
                    config,
                    base_checkpoint,
                    global_step,
                    row,
                    optimizer,
                    rng,
                )
            guardrail = score + historical
            if constraints == 0 and historical > -5.0 and guardrail > best_guardrail:
                best_guardrail = guardrail
                _save_rl_checkpoint(
                    architecture,
                    algorithm,
                    policy,
                    edge_index,
                    output_dir / "best_guardrail.pt",
                    config,
                    base_checkpoint,
                    global_step,
                    row,
                    optimizer,
                    rng,
                )
        pd.DataFrame([row]).reindex(columns=RL_LOG_COLUMNS).to_csv(
            log_path, mode="a", header=not log_path.exists(), index=False
        )
        if global_step % int(config["rl_training"]["checkpoint_every_steps"]) < rollout_target:
            _save_rl_checkpoint(
                architecture,
                algorithm,
                policy,
                edge_index,
                output_dir / f"step_{global_step:06d}.pt",
                config,
                base_checkpoint,
                global_step,
                row,
                optimizer,
                rng,
            )
        print(
            f"[economic-{architecture}-{algorithm}] step={global_step}/{target} "
            f"damage={row['mean_training_harmonized_damage_dollars']:.1f} "
            f"periodic={row.get('periodic_main_improvement_vs_greedy_percent', float('nan')):.3f}",
            flush=True,
        )
    _save_rl_checkpoint(
        architecture,
        algorithm,
        policy,
        edge_index,
        output_dir / "final.pt",
        config,
        base_checkpoint,
        global_step,
        row,
        optimizer,
        rng,
    )
    if not (output_dir / "best_vs_greedy.pt").exists():
        _atomic_checkpoint_copy(output_dir / "final.pt", output_dir / "best_vs_greedy.pt")
    if not (output_dir / "best_guardrail.pt").exists():
        _atomic_checkpoint_copy(output_dir / "best_vs_greedy.pt", output_dir / "best_guardrail.pt")
    if architecture == "true_gnn":
        dependence = _graph_dependence(policy, edge_index, data, device)
        _write_csv(
            dependence,
            d["diagnostics"] / f"{architecture}_{algorithm}_graph_dependence_verification.csv",
        )
    return output_dir / "best_vs_greedy.pt"


def _load_rl_policy_path(
    architecture: str,
    path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Any, Optional[torch.Tensor], dict[str, Any]]:
    if architecture == "true_gnn":
        return load_true_gnn_checkpoint(path, device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    candidate_dim = int(checkpoint["candidate_dim"])
    if architecture == "mlp":
        actor = MLPScorer(candidate_dim, int(checkpoint["hidden_dim"]))
    else:
        channels, grid, _ = checkpoint["raster_shape"]
        actor = CNNEncoder(
            candidate_dim,
            int(channels),
            int(grid),
            hidden=int(checkpoint["hidden_dim"]),
            conv_channels=int(config["architectures"]["cnn_conv_channels"]),
            spatial_dim=int(config["architectures"]["cnn_spatial_embedding_dim"]),
        )
    actor.load_state_dict(checkpoint["actor_state"])
    policy = EncoderActorCritic(actor, int(checkpoint["hidden_dim"])).to(device)
    policy.value.load_state_dict(checkpoint["value_state"])
    return policy, None, checkpoint


def _load_context(
    config: dict[str, Any],
) -> tuple[V03RestorationModel, pd.DataFrame, HarmonizedValuator, HarmonizedObservationAdapter]:
    model = V03RestorationModel(
        _resolve(config["paths"]["case_dir"]),
        allocation_mode=config.get("anchor_allocation_mode", "critical_first_fixed_priority"),
    )
    duration_config = _load_yaml(_resolve(config["paths"]["duration_valuation_config"]))
    parameters, _ = _valuation_parameters(duration_config)
    node_config = _load_yaml(_resolve(config["paths"]["node_replay_config"]))
    inventory = _load_inventory(model, node_config["commercial_size_classes"])
    valuator = HarmonizedValuator(
        inventory,
        parameters,
        float(config["valuation"]["critical_value_usd_per_kwh"]),
    )
    adapter = HarmonizedObservationAdapter(model, valuator, inventory)
    return model, inventory, valuator, adapter


def _load_dataset(config: dict[str, Any], d: dict[str, Path]) -> dict[str, Any]:
    model, _inventory, valuator, adapter = _load_context(config)
    metadata = pd.read_csv(_resolve(config["paths"]["original_true_gnn_metadata"]))
    return _build_dataset(model, adapter, valuator, metadata, config, d)

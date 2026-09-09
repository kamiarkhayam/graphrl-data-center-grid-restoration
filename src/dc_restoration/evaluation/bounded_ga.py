from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from dc_restoration.economics.duration_damage import _valuation_parameters
from dc_restoration.economics.load_accounting import (
    RESOURCE_CONFIGS,
    _apply_node_valuations,
    _load_inventory,
    _support_by_load,
)
from dc_restoration.paths import repository_root
from dc_restoration.restoration.environment import V03RestorationModel
from dc_restoration.testbed.benchmark import CachedRestorationSimulator

EPS = 1e-9


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _append_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


class SourceDurationValuator:
    """Load-block economic valuation using only configured source-based functions."""

    def __init__(
        self, inventory: pd.DataFrame, parameters: dict[str, dict[str, Any]], legacy_critical: float
    ):
        self.inventory = inventory.set_index("load_id", drop=False)
        self.parameters = parameters
        self.legacy_critical = float(legacy_critical)

    def components(self, energy_kwh: dict[str, float] | pd.Series) -> pd.DataFrame:
        energy = pd.Series(energy_kwh, dtype=float).reindex(self.inventory.index, fill_value=0.0)
        frame = self.inventory.copy()
        frame["unserved_energy_kwh"] = energy.to_numpy(float)
        frame["equivalent_interruption_duration_hours"] = frame["unserved_energy_kwh"] / frame[
            "baseline_p_kw"
        ].astype(float).clip(lower=EPS)
        valued = _apply_node_valuations(frame.reset_index(drop=True), self.parameters, ["primary"])
        valued["source_duration_damage_dollars"] = valued[
            "node_duration_cost_primary_dollars"
        ].astype(float)
        critical = valued["critical"].astype(bool)
        valued["legacy_critical_100_sensitivity_dollars"] = np.where(
            critical,
            self.legacy_critical * valued["unserved_energy_kwh"].astype(float),
            valued["source_duration_damage_dollars"].astype(float),
        )
        return valued

    def cost(self, energy_kwh: dict[str, float] | pd.Series) -> float:
        return float(self.components(energy_kwh)["legacy_critical_100_sensitivity_dollars"].sum())

    def marginal_cost(
        self, accumulated: dict[str, float], rates_kw: dict[str, float], hours: float = 1.0
    ) -> float:
        before = pd.Series(accumulated, dtype=float).reindex(self.inventory.index, fill_value=0.0)
        rates = pd.Series(rates_kw, dtype=float).reindex(self.inventory.index, fill_value=0.0)
        return self.cost(before + rates * float(hours)) - self.cost(before)


def _load_model(case_dir: Path, damage_path: Path | None = None) -> V03RestorationModel:
    model = V03RestorationModel(case_dir)
    if damage_path is not None:
        damage = pd.read_csv(damage_path)
        model.damage = damage
        model.base = CachedRestorationSimulator(model.nodes, model.edges, model.loads, damage)
        model._metric_cache = {}
    return model


def _resource_rates(
    model: V03RestorationModel,
    remaining: set[str],
    resource_code: str,
    energy_remaining_mwh: float,
    duration_hours: float,
) -> tuple[dict[str, float], float, Any]:
    cfg = RESOURCE_CONFIGS[resource_code]
    base = model.base.metrics(tuple(sorted(remaining)))
    cap_mw = cfg.max_support_mw(sum(model.dc_support_by_anchor.values()))
    allocation = _support_by_load(model, base, cap_mw)
    potential = float(sum(allocation.values())) * float(duration_hours)
    if cap_mw <= EPS or potential <= EPS:
        fraction = 0.0
        used = 0.0
    elif cfg.finite_energy_hours is None:
        fraction = 1.0
        used = potential
    else:
        used = min(max(0.0, energy_remaining_mwh), potential)
        fraction = used / max(potential, EPS)
    rates: dict[str, float] = {}
    for load_id in map(str, base.unserved_load_ids):
        baseline = float(model.load_info[load_id].get("baseline_p_kw", 0.0))
        rates[load_id] = max(0.0, baseline - 1000.0 * allocation.get(load_id, 0.0) * fraction)
    return rates, used, base


def _simulate_source_greedy(
    model: V03RestorationModel,
    valuator: SourceDurationValuator,
    scenario_key: str,
    resource_code: str,
) -> dict[str, Any]:
    damaged = list(map(str, model.scenario_edges(scenario_key)))
    repair_times = model.scenario_repair_times(scenario_key)
    cfg = RESOURCE_CONFIGS[resource_code]
    energy_remaining = cfg.initial_energy_mwh(sum(model.dc_support_by_anchor.values()))
    accumulated: dict[str, float] = {}
    repaired: set[str] = set()
    sequence: list[str] = []
    support_mwh = 0.0
    started = time.time()
    while len(repaired) < len(damaged):
        remaining = set(damaged) - repaired
        rates, _used, _base = _resource_rates(
            model, remaining, resource_code, energy_remaining, 1.0
        )
        if sum(rates.values()) <= 1e-6:
            break
        current = valuator.marginal_cost(accumulated, rates)
        candidates: list[tuple[float, float, str]] = []
        for edge_id in remaining:
            after_rates, _after_used, _ = _resource_rates(
                model, remaining - {edge_id}, resource_code, energy_remaining, 1.0
            )
            gain = current - valuator.marginal_cost(accumulated, after_rates)
            candidates.append((float(gain), -float(repair_times.get(edge_id, 8.0)), str(edge_id)))
        if not candidates:
            break
        _gain, _negative_duration, chosen = max(candidates)
        duration = float(repair_times.get(chosen, 8.0))
        interval_rates, used, _ = _resource_rates(
            model, remaining, resource_code, energy_remaining, duration
        )
        for load_id, rate in interval_rates.items():
            accumulated[load_id] = accumulated.get(load_id, 0.0) + rate * duration
        if cfg.finite_energy_hours is not None:
            energy_remaining = max(0.0, energy_remaining - used)
        support_mwh += used
        repaired.add(chosen)
        sequence.append(chosen)
    valued = valuator.components(accumulated)
    critical = valued["critical"].astype(bool)
    return {
        "scenario_key": scenario_key,
        "dc_resource_configuration": resource_code,
        "policy": "greedy_source_duration_damage",
        "display_name": "Greedy",
        "source_duration_damage_dollars": float(valued["source_duration_damage_dollars"].sum()),
        "source_duration_noncritical_damage_dollars": float(
            valued.loc[~critical, "source_duration_damage_dollars"].sum()
        ),
        "source_duration_critical_damage_dollars": float(
            valued.loc[critical, "source_duration_damage_dollars"].sum()
        ),
        "legacy_critical_100_sensitivity_dollars": float(
            valued["legacy_critical_100_sensitivity_dollars"].sum()
        ),
        "paper_primary_economic_damage_dollars": float(
            valued["legacy_critical_100_sensitivity_dollars"].sum()
        ),
        "sector_duration_only_damage_dollars": float(
            valued["source_duration_damage_dollars"].sum()
        ),
        "energy_not_served_mwh": float(valued["unserved_energy_kwh"].sum()) / 1000.0,
        "critical_energy_not_served_mwh": float(valued.loc[critical, "unserved_energy_kwh"].sum())
        / 1000.0,
        "dc_support_used_mwh": support_mwh,
        "repairs_used": len(sequence),
        "repair_sequence_edge_ids": json.dumps(sequence),
        "runtime_seconds": time.time() - started,
        "invalid_action_count": 0,
        "export_cap_violation_count": 0,
        "support_outside_anchor_zone_count": 0,
    }


def _evaluate_source_order(
    model: V03RestorationModel,
    valuator: SourceDurationValuator,
    scenario_key: str,
    order: Sequence[str],
    resource_code: str = "R1_current_power_bounded_anchor",
) -> dict[str, Any]:
    damaged = set(map(str, model.scenario_edges(scenario_key)))
    repair_times = model.scenario_repair_times(scenario_key)
    cfg = RESOURCE_CONFIGS[resource_code]
    energy_remaining = cfg.initial_energy_mwh(sum(model.dc_support_by_anchor.values()))
    accumulated: dict[str, float] = {}
    repaired: set[str] = set()
    support_mwh = 0.0
    for raw_edge in order:
        edge_id = str(raw_edge)
        if edge_id not in damaged or edge_id in repaired:
            continue
        remaining = damaged - repaired
        duration = float(repair_times.get(edge_id, 8.0))
        rates, used, _ = _resource_rates(
            model, remaining, resource_code, energy_remaining, duration
        )
        for load_id, rate in rates.items():
            accumulated[load_id] = accumulated.get(load_id, 0.0) + rate * duration
        if cfg.finite_energy_hours is not None:
            energy_remaining = max(0.0, energy_remaining - used)
        support_mwh += used
        repaired.add(edge_id)
        if sum(rates.values()) <= 1e-6:
            break
    valued = valuator.components(accumulated)
    critical = valued["critical"].astype(bool)
    return {
        "source_duration_damage_dollars": float(valued["source_duration_damage_dollars"].sum()),
        "paper_primary_economic_damage_dollars": float(
            valued["legacy_critical_100_sensitivity_dollars"].sum()
        ),
        "sector_duration_only_damage_dollars": float(
            valued["source_duration_damage_dollars"].sum()
        ),
        "energy_not_served_mwh": float(valued["unserved_energy_kwh"].sum()) / 1000.0,
        "critical_energy_not_served_mwh": float(valued.loc[critical, "unserved_energy_kwh"].sum())
        / 1000.0,
        "dc_support_used_mwh": support_mwh,
        "repairs_used": len(repaired),
    }


def _worker_ga(task: tuple[str, int, int, int, list[str]]) -> dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_VALUATOR is None:
        raise RuntimeError("Harmonized GA worker was not initialized.")
    scenario, seed, population_size, generations, harmonized_greedy_order = task
    model = _WORKER_MODEL
    valuator = _WORKER_VALUATOR
    rng = random.Random(seed)
    damaged = list(map(str, model.scenario_edges(scenario)))
    seeds = [
        harmonized_greedy_order + [edge for edge in damaged if edge not in harmonized_greedy_order]
    ]
    population = [order.copy() for order in seeds if order]
    while len(population) < population_size:
        order = damaged.copy()
        rng.shuffle(order)
        population.append(order)
    cache: dict[str, float] = {}
    evaluations = 0

    def cost(order: list[str]) -> float:
        nonlocal evaluations
        signature = "|".join(order)
        if signature not in cache:
            cache[signature] = _evaluate_source_order(model, valuator, scenario, order)[
                "paper_primary_economic_damage_dollars"
            ]
            evaluations += 1
        return cache[signature]

    costs = [cost(order) for order in population]
    best_index = int(np.argmin(costs))
    best_order = population[best_index].copy()
    best_cost = float(costs[best_index])
    for _ in range(generations):
        ranked = [order for _, order in sorted(zip(costs, population), key=lambda item: item[0])]
        next_population = [order.copy() for order in ranked[: max(3, population_size // 4)]]
        while len(next_population) < population_size:
            child = rng.choice(ranked[: max(4, population_size // 2)]).copy()
            if len(child) >= 2:
                i, j = sorted(rng.sample(range(len(child)), 2))
                if rng.random() < 0.65:
                    child[i], child[j] = child[j], child[i]
                else:
                    segment = child[i:j]
                    del child[i:j]
                    insertion = rng.randrange(len(child) + 1)
                    child[insertion:insertion] = segment
            next_population.append(child)
        population = next_population
        costs = [cost(order) for order in population]
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            best_cost = float(costs[index])
            best_order = population[index].copy()
    metrics = _evaluate_source_order(model, valuator, scenario, best_order)
    return {
        "scenario_key": scenario,
        "policy": "bounded_ga_source_duration_damage",
        "display_name": "Bounded GA",
        **metrics,
        "fitness_evaluations": evaluations,
        "population": population_size,
        "generations": generations,
        "repair_sequence_edge_ids": json.dumps(best_order),
    }


def _run_harmonized_ga(
    cohort: pd.DataFrame,
    greedy: pd.DataFrame,
    case_dir: Path,
    damage_path: Path,
    duration_config: Path,
    node_config: Path,
    legacy_critical: float,
    workers: int,
    population: int,
    generations: int,
    output_path: Path,
) -> pd.DataFrame:
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    completed = set(existing["scenario_key"].astype(str)) if not existing.empty else set()
    orders = {
        str(row["scenario_key"]): json.loads(str(row["repair_sequence_edge_ids"]))
        for _, row in greedy.iterrows()
    }
    tasks = []
    for index, row in cohort.reset_index(drop=True).iterrows():
        scenario = str(row["scenario_key"])
        if scenario in completed:
            continue
        seed = int(row.get("random_seed", 32 + index))
        tasks.append((scenario, seed, population, generations, orders[scenario]))
    if tasks:
        started = time.time()
        with ProcessPoolExecutor(
            max_workers=max(1, min(workers, len(tasks))),
            initializer=_worker_init,
            initargs=(
                str(case_dir),
                str(damage_path),
                str(duration_config),
                str(node_config),
                legacy_critical,
            ),
        ) as pool:
            futures = {pool.submit(_worker_ga, task): task[0] for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                row["runtime_batch_elapsed_seconds"] = time.time() - started
                _append_rows([row], output_path)
                print(f"Bounded GA {index}/{len(futures)}", flush=True)
    return pd.read_csv(output_path)


_WORKER_MODEL: V03RestorationModel | None = None

_WORKER_VALUATOR: SourceDurationValuator | None = None


def _worker_init(
    case_dir: str, damage_path: str, duration_config: str, node_config: str, legacy_critical: float
) -> None:
    global _WORKER_MODEL, _WORKER_VALUATOR
    _WORKER_MODEL = _load_model(Path(case_dir), Path(damage_path) if damage_path else None)
    duration = _read_yaml(Path(duration_config))
    parameters, _ = _valuation_parameters(duration)
    node = _read_yaml(Path(node_config))
    inventory = _load_inventory(_WORKER_MODEL, node["commercial_size_classes"])
    _WORKER_VALUATOR = SourceDurationValuator(inventory, parameters, legacy_critical)

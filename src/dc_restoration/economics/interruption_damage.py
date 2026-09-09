from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from dc_restoration.economics.duration_damage import _cdf_per_interrupted_kw
from dc_restoration.economics.load_accounting import _support_by_load
from dc_restoration.paths import repository_root
from dc_restoration.restoration.environment import V03RestorationModel

EPS = 1e-9


class HarmonizedValuator:
    """Economic damage mapping applied to physical interruption trajectories."""

    def __init__(
        self,
        inventory: pd.DataFrame,
        parameters: dict[str, dict[str, Any]],
        critical_value_usd_per_kwh: float,
    ) -> None:
        self.inventory = inventory.set_index("load_id", drop=False)
        self.parameters = parameters["primary"]
        self.critical_value = float(critical_value_usd_per_kwh)

    def _noncritical_cdf(self, rows: pd.DataFrame, duration: np.ndarray) -> np.ndarray:
        result = np.zeros(len(rows), dtype=float)
        residential = rows["load_type"].astype(str).str.lower().eq("residential").to_numpy()
        if residential.any():
            p = self.parameters
            result[residential] = _cdf_per_interrupted_kw(
                duration[residential],
                p["residential_noncritical_24h"],
                p["residential_noncritical_points"],
                p["residential_noncritical_shape"],
            )
        if (~residential).any():
            p = self.parameters
            result[~residential] = _cdf_per_interrupted_kw(
                duration[~residential],
                p["commercial_24h"],
                p["commercial_points"],
                p["commercial_shape"],
            )
        return result

    def cost_from_energy(self, energy_kwh: dict[str, float] | pd.Series) -> float:
        energy = pd.Series(energy_kwh, dtype=float).reindex(self.inventory.index, fill_value=0.0)
        rows = self.inventory
        critical = rows["critical"].astype(bool).to_numpy()
        baseline = rows["baseline_p_kw"].astype(float).clip(lower=EPS).to_numpy()
        values = energy.to_numpy(float)
        duration = values / baseline
        cost = np.zeros(len(rows), dtype=float)
        cost[critical] = self.critical_value * values[critical]
        if (~critical).any():
            cdf = self._noncritical_cdf(rows.loc[~critical], duration[~critical])
            cost[~critical] = baseline[~critical] * cdf
        return float(cost.sum())

    def old_cost_from_energy(self, energy_kwh: dict[str, float] | pd.Series) -> float:
        energy = pd.Series(energy_kwh, dtype=float).reindex(self.inventory.index, fill_value=0.0)
        return float((energy * self.inventory["modeled_voll_usd_per_kwh"].astype(float)).sum())

    def marginal_one_hour_cost(
        self,
        accumulated_energy_kwh: dict[str, float],
        unserved_kw: dict[str, float],
    ) -> float:
        before = pd.Series(accumulated_energy_kwh, dtype=float).reindex(
            self.inventory.index, fill_value=0.0
        )
        rate = pd.Series(unserved_kw, dtype=float).reindex(self.inventory.index, fill_value=0.0)
        return self.cost_from_energy(before + rate) - self.cost_from_energy(before)


def _state_unserved_kw(
    model: V03RestorationModel,
    damaged_unrepaired: Iterable[str],
) -> tuple[dict[str, float], Any]:
    base = model.base.metrics(tuple(sorted(map(str, damaged_unrepaired))))
    allocation = _support_by_load(model, base, sum(model.dc_support_by_anchor.values()))
    rates: dict[str, float] = {}
    for load_id in map(str, base.unserved_load_ids):
        baseline = float(model.load_info[load_id].get("baseline_p_kw", 0.0))
        rates[load_id] = max(0.0, baseline - 1000.0 * allocation.get(load_id, 0.0))
    return rates, base


def _accrue(energy: dict[str, float], rates: dict[str, float], hours: float) -> None:
    for load_id, rate in rates.items():
        energy[load_id] = energy.get(load_id, 0.0) + float(rate) * float(hours)

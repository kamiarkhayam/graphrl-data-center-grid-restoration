from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dc_restoration.anchors.controller import RESOURCE_CONFIGS
from dc_restoration.economics.duration_damage import GROUPS, _cdf_per_interrupted_kw
from dc_restoration.paths import repository_root
from dc_restoration.restoration.environment import V03RestorationModel

EPS = 1e-9


def _load_inventory(model: V03RestorationModel, bins: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    small = float(bins["small_max_annual_mwh"])
    medium = float(bins["medium_max_annual_mwh"])
    for load_id, raw in model.load_info.items():
        load_type = str(raw.get("load_type", "commercial")).strip().lower()
        critical = bool(raw.get("is_critical", False))
        peak_kw = float(raw.get("peak_p_kw", raw.get("baseline_p_kw", 0.0)))
        average_kw = float(raw.get("average_p_kw", raw.get("baseline_p_kw", 0.0)))
        annual_mwh = average_kw * 8.76
        if load_type == "residential":
            size_class = "residential"
        elif annual_mwh <= small:
            size_class = "small_commercial"
        elif annual_mwh <= medium:
            size_class = "medium_commercial"
        else:
            size_class = "large_commercial"
        rows.append(
            {
                "load_id": str(load_id),
                "node_id": str(raw.get("node_id", "")),
                "load_type": load_type,
                "critical": critical,
                "valuation_group": f"{load_type}_{'critical' if critical else 'noncritical'}",
                "commercial_size_class": size_class,
                "baseline_p_kw": float(raw.get("baseline_p_kw", 0.0)),
                "average_p_kw": average_kw,
                "annual_energy_mwh": annual_mwh,
                "peak_p_kw": peak_kw,
                "modeled_voll_usd_per_kwh": float(raw.get("voll_usd_per_kwh", 0.0)),
            }
        )
    inventory = pd.DataFrame(rows)
    if set(inventory["valuation_group"].unique()) - set(GROUPS):
        raise ValueError("Unexpected load valuation group in case inventory.")
    return inventory


def _support_by_load(
    model: V03RestorationModel,
    base: Any,
    cap_mw: float,
) -> dict[str, float]:
    """Archived critical-first, fixed within-class allocation; retain tuple tie breaks."""
    from dc_restoration.anchors.priority import validate_allocation_mode

    validate_allocation_mode(model.allocation_mode)
    candidates: list[tuple[bool, float, float, str, str]] = []
    unserved = set(map(str, base.unserved_load_ids))
    for anchor_id, load_ids in model.anchor_load_ids.items():
        for load_id in load_ids:
            lid = str(load_id)
            if lid not in unserved or lid not in model.load_info:
                continue
            info = model.load_info[lid]
            candidates.append(
                (
                    bool(info.get("is_critical", False)),
                    float(info.get("voll_usd_per_kwh", 0.0)),
                    float(info.get("baseline_p_kw", 0.0)) / 1000.0,
                    str(anchor_id),
                    lid,
                )
            )
    candidates.sort(reverse=True)
    remaining = max(0.0, float(cap_mw))
    allocation: dict[str, float] = {}
    for _critical, _voll, load_mw, _anchor_id, load_id in candidates:
        if remaining <= EPS:
            break
        use = min(load_mw, remaining)
        allocation[load_id] = use
        remaining -= use
    return allocation


def _apply_node_valuations(
    loads: pd.DataFrame,
    parameters: dict[str, dict[str, Any]],
    cases: list[str],
) -> pd.DataFrame:
    valued = loads.copy()
    duration = valued["equivalent_interruption_duration_hours"].to_numpy(float)
    exposure_kw = valued["baseline_p_kw"].to_numpy(float)
    for case in cases:
        p = parameters[case]
        cost = np.zeros(len(valued), dtype=float)
        for group in GROUPS:
            mask = valued["valuation_group"].eq(group).to_numpy()
            if not mask.any():
                continue
            if group == "residential_noncritical":
                cdf = _cdf_per_interrupted_kw(
                    duration[mask],
                    p["residential_noncritical_24h"],
                    p["residential_noncritical_points"],
                    p["residential_noncritical_shape"],
                )
            elif group == "residential_critical":
                cdf = _cdf_per_interrupted_kw(
                    duration[mask],
                    p["residential_critical_24h"],
                    p["residential_critical_points"],
                    p["residential_critical_shape"],
                )
            else:
                cdf = _cdf_per_interrupted_kw(
                    duration[mask],
                    p["commercial_24h"],
                    p["commercial_points"],
                    p["commercial_shape"],
                )
            cost[mask] = exposure_kw[mask] * np.asarray(cdf, dtype=float)
        valued[f"node_duration_cost_{case}_dollars"] = cost
    return valued

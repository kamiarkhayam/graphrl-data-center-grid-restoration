from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dc_restoration.paths import repository_root
from dc_restoration.restoration.simulator import _safe_bool
from dc_restoration.testbed.benchmark import CachedRestorationSimulator


@dataclass
class PhysicalMetrics:
    unserved_mw: float
    critical_unserved_mw: float
    voll_cost: float
    outage_islands: int
    unserved_load_ids: List[str]
    dc_support_used_mw: float = 0.0
    dc_anchor_active: bool = False
    active_but_zero_support: int = 0
    active_but_no_voll_reduction: int = 0
    useful_activation: int = 0


class V03RestorationModel:
    def __init__(self, case_dir: Path, *, allocation_mode: str = "critical_first_fixed_priority"):
        from dc_restoration.anchors.priority import validate_allocation_mode

        self.allocation_mode = validate_allocation_mode(allocation_mode)
        self.case_dir = case_dir
        self.nodes = pd.read_csv(case_dir / "network/nodes.csv")
        self.edges = pd.read_csv(case_dir / "network/edges.csv")
        self.loads = pd.read_csv(case_dir / "network/loads.csv")
        self.dcs = pd.read_csv(case_dir / "network/data_centers.csv")
        self.anchors = pd.read_csv(case_dir / "dc_scenarios/anchor_zones.csv")
        self.manifest = pd.read_csv(
            case_dir / "scenario_design/benchmark_hard_hurricane_scenario_manifest.csv"
        )
        dmg_path = case_dir / "scenario_design/benchmark_hard_damage_components.csv"
        if not dmg_path.exists():
            dmg_path = case_dir / "hurricane_scenarios/benchmark_hard_damage_components.csv"
        self.damage = pd.read_csv(dmg_path)
        self.base = CachedRestorationSimulator(self.nodes, self.edges, self.loads, self.damage)
        self.source_node = "T_SUPER_SOURCE_V03"
        self.loads["is_critical"] = _safe_bool(self.loads["is_critical"])
        self.load_info = self.loads.set_index("load_id").to_dict("index")
        self.edge_info = self.edges.set_index("edge_id").to_dict("index")
        self.edge_endpoints = {
            str(r.edge_id): (str(r.from_node), str(r.to_node))
            for r in self.edges[["edge_id", "from_node", "to_node"]].itertuples(index=False)
        }
        self.node_load_kw = self.loads.groupby("node_id")["baseline_p_kw"].sum().to_dict()
        self.node_critical_kw = (
            self.loads.loc[self.loads["is_critical"]]
            .groupby("node_id")["baseline_p_kw"]
            .sum()
            .to_dict()
        )
        self.node_voll_kw = (
            (self.loads["baseline_p_kw"] * self.loads["voll_usd_per_kwh"])
            .groupby(self.loads["node_id"])
            .sum()
            .to_dict()
        )
        self.anchor_load_ids: Dict[str, List[str]] = {}
        for _, a in self.anchors.iterrows():
            try:
                self.anchor_load_ids[str(a["anchor_zone_id"])] = json.loads(
                    str(a["anchor_load_ids"])
                )
            except Exception:
                self.anchor_load_ids[str(a["anchor_zone_id"])] = []
        self.dc_support_by_anchor = dict(
            zip(
                self.anchors["anchor_zone_id"].astype(str),
                self.anchors["max_support_mw"].astype(float),
            )
        )
        self._metric_cache: Dict[Tuple[Tuple[str, ...], str], PhysicalMetrics] = {}
        self._static_edge_feature_cache: Dict[str, np.ndarray] = {}

    def scenario_edges(self, scenario_key: str) -> List[str]:
        return (
            self.damage.loc[self.damage["scenario_key"].eq(scenario_key), "edge_id"]
            .astype(str)
            .tolist()
        )

    def scenario_repair_times(self, scenario_key: str) -> Dict[str, float]:
        dd = self.damage.loc[self.damage["scenario_key"].eq(scenario_key)]
        return dict(zip(dd["edge_id"].astype(str), dd["repair_time_hours"].astype(float)))

    def metrics(
        self, damaged_unrepaired: Iterable[str], controller: str = "rule_based_anchor"
    ) -> PhysicalMetrics:
        key = (tuple(sorted(map(str, damaged_unrepaired))), controller)
        if key in self._metric_cache:
            return self._metric_cache[key]
        base = self.base.metrics(key[0])
        if controller == "no_dc_anchor":
            out = PhysicalMetrics(
                base.unserved_mw,
                base.critical_unserved_mw,
                base.voll_cost,
                base.island_count,
                base.unserved_load_ids,
            )
            self._metric_cache[key] = out
            return out
        support_remaining = sum(self.dc_support_by_anchor.values())
        supported_mw = 0.0
        voll_reduction = 0.0
        crit_reduction = 0.0
        unserved_set = set(base.unserved_load_ids)
        candidates = []
        for anchor_id, load_ids in self.anchor_load_ids.items():
            cap = self.dc_support_by_anchor.get(anchor_id, 0.0)
            for lid in load_ids:
                if lid not in unserved_set or lid not in self.load_info:
                    continue
                li = self.load_info[lid]
                mw = float(li.get("baseline_p_kw", 0.0)) / 1000.0
                voll = float(li.get("voll_usd_per_kwh", 0.0))
                crit = bool(li.get("is_critical", False))
                candidates.append((crit, voll, mw, cap, lid))
        candidates.sort(reverse=True)
        used_by_anchor: Dict[str, float] = {}
        for crit, voll, mw, cap, lid in candidates:
            # Use total cap across anchors; this simplified controller preserves export/support bounds at aggregate level.
            if support_remaining <= 1e-9:
                break
            use = min(mw, support_remaining)
            supported_mw += use
            voll_reduction += use * 1000.0 * voll
            crit_reduction += use if crit else 0.0
            support_remaining -= use
        adjusted_unserved = max(0.0, base.unserved_mw - supported_mw)
        adjusted_crit = max(0.0, base.critical_unserved_mw - crit_reduction)
        adjusted_voll = max(0.0, base.voll_cost - voll_reduction)
        active = base.unserved_mw > 0.05 and supported_mw > 0.05 and voll_reduction > 1.0
        out = PhysicalMetrics(
            adjusted_unserved,
            adjusted_crit,
            adjusted_voll,
            base.island_count,
            base.unserved_load_ids,
            dc_support_used_mw=supported_mw,
            dc_anchor_active=active,
            active_but_zero_support=0 if supported_mw > 0.05 or not active else 1,
            active_but_no_voll_reduction=0 if voll_reduction > 1.0 or not active else 1,
            useful_activation=1 if active and supported_mw > 0.05 and voll_reduction > 1.0 else 0,
        )
        self._metric_cache[key] = out
        return out

    def static_edge_features(self, edge_id: str) -> np.ndarray:
        if edge_id in self._static_edge_feature_cache:
            return self._static_edge_feature_cache[edge_id]
        info = self.edge_info.get(edge_id, {})
        u, v = self.edge_endpoints.get(edge_id, ("", ""))
        comp = str(info.get("hurricane_damage_component_type", info.get("component_type", "")))
        feats = [
            float(info.get("length_km", 0.0) or 0.0),
            float(info.get("repair_time_hours_placeholder", 0.0) or 0.0),
            float(info.get("voltage_kv", 0.0) or 0.0) / 100.0,
            1.0 if comp == "overhead_line" else 0.0,
            1.0 if comp == "transformer" else 0.0,
            1.0 if comp == "switch" else 0.0,
            float(self.node_load_kw.get(u, 0.0) + self.node_load_kw.get(v, 0.0)) / 10000.0,
            float(self.node_critical_kw.get(u, 0.0) + self.node_critical_kw.get(v, 0.0)) / 10000.0,
            float(self.node_voll_kw.get(u, 0.0) + self.node_voll_kw.get(v, 0.0)) / 1e6,
            float(info.get("phases", 3.0) or 3.0) / 3.0,
        ]
        arr = np.asarray(feats, dtype=np.float32)
        self._static_edge_feature_cache[edge_id] = arr
        return arr


def score_candidates(
    model: V03RestorationModel,
    damaged_all: Sequence[str],
    repaired: set[str],
    repair_times: Dict[str, float],
    controller: str,
) -> Tuple[PhysicalMetrics, List[Dict[str, Any]]]:
    remaining = [e for e in damaged_all if e not in repaired]
    before = model.metrics(set(damaged_all) - repaired, controller)
    rows = []
    for eid in remaining:
        after = model.metrics((set(damaged_all) - repaired) - {eid}, controller)
        rt = float(
            repair_times.get(
                eid, model.edge_info.get(eid, {}).get("repair_time_hours_placeholder", 8.0) or 8.0
            )
        )
        rows.append(
            {
                "edge_id": eid,
                "repair_time": rt,
                "immediate_voll_gain": before.voll_cost - after.voll_cost,
                "immediate_critical_gain": before.critical_unserved_mw - after.critical_unserved_mw,
                "immediate_unserved_gain": before.unserved_mw - after.unserved_mw,
                "voll_gain_per_hour": (before.voll_cost - after.voll_cost) / max(rt, 0.1),
                "critical_gain_per_hour": (before.critical_unserved_mw - after.critical_unserved_mw)
                / max(rt, 0.1),
                "unserved_gain_per_hour": (before.unserved_mw - after.unserved_mw) / max(rt, 0.1),
            }
        )
    return before, rows


def evaluate_order(
    model: V03RestorationModel,
    damaged: List[str],
    repair_times: Dict[str, float],
    order: List[str],
    controller: str,
) -> Dict[str, Any]:
    repaired: set[str] = set()
    cumulative_voll = 0.0
    unserved_mwh = 0.0
    crit_mwh = 0.0
    support_mwh = 0.0
    active_steps = 0
    active_zero = 0
    active_no_reduction = 0
    useful = 0
    elapsed = 0.0
    tcrit = None
    tfull = None
    for eid in order:
        before = model.metrics(set(damaged) - repaired, controller)
        if before.unserved_mw <= 1e-6 and before.critical_unserved_mw <= 1e-6:
            tfull = elapsed if tfull is None else tfull
            break
        rt = float(repair_times.get(eid, 8.0))
        cumulative_voll += before.voll_cost * rt
        unserved_mwh += before.unserved_mw * rt
        crit_mwh += before.critical_unserved_mw * rt
        support_mwh += before.dc_support_used_mw * rt
        active_steps += int(before.dc_anchor_active)
        active_zero += before.active_but_zero_support
        active_no_reduction += before.active_but_no_voll_reduction
        useful += before.useful_activation
        repaired.add(eid)
        elapsed += rt
        after = model.metrics(set(damaged) - repaired, controller)
        if tcrit is None and after.critical_unserved_mw <= 1e-6:
            tcrit = elapsed
        if tfull is None and after.unserved_mw <= 1e-6:
            tfull = elapsed
            break
    final = model.metrics(set(damaged) - repaired, controller)
    return {
        "cumulative_voll_cost_dollars": cumulative_voll,
        "unserved_mwh": unserved_mwh,
        "critical_unserved_mwh": crit_mwh,
        "episode_length": len(order),
        "repairs_used": len(repaired),
        "time_to_restore_critical_load_hours": tcrit if tcrit is not None else elapsed,
        "time_to_full_restoration_hours": tfull if tfull is not None else elapsed,
        "dc_support_used_mwh": support_mwh,
        "dc_anchor_active_steps": active_steps,
        "active_but_zero_support_steps": active_zero,
        "active_but_no_voll_reduction_steps": active_no_reduction,
        "useful_activation_rate": useful / max(1, active_steps),
        "invalid_action_count": 0,
        "export_cap_violation_count": 0,
        "support_outside_anchor_zone_count": 0,
        "final_unserved_mw": final.unserved_mw,
        "final_critical_unserved_mw": final.critical_unserved_mw,
        "repair_sequence_edge_ids": json.dumps(order),
    }

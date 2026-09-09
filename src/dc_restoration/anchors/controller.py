from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dc_restoration.paths import repository_root
from dc_restoration.restoration.environment import PhysicalMetrics, V03RestorationModel

EPS = 1e-9


@dataclass(frozen=True)
class ResourceConfig:
    code: str
    description: str
    cap_scale: float
    finite_energy_hours: Optional[float] = None
    is_reference: bool = False
    reference_dc_configuration: str = ""

    def max_support_mw(self, current_cap_mw: float) -> float:
        return float(current_cap_mw) * float(self.cap_scale)

    def initial_energy_mwh(self, current_cap_mw: float) -> float:
        if self.finite_energy_hours is None:
            return float("inf")
        return self.max_support_mw(current_cap_mw) * float(self.finite_energy_hours)


RESOURCE_CONFIGS: Dict[str, ResourceConfig] = {
    "R0_no_dc_anchor": ResourceConfig(
        "R0_no_dc_anchor",
        "No anchor: support disabled.",
        0.0,
        None,
        True,
        "no_dc_anchor",
    ),
    "R1_current_power_bounded_anchor": ResourceConfig(
        "R1_current_power_bounded_anchor",
        "Power-only reference: aggregate export cap without finite cumulative energy depletion.",
        1.0,
        None,
        True,
        "rule_based_anchor",
    ),
    "E12_finite_energy_12h": ResourceConfig(
        "E12_finite_energy_12h",
        "Current export cap with finite energy equal to 12 hours at rated aggregate export.",
        1.0,
        12.0,
    ),
    "E24_finite_energy_24h": ResourceConfig(
        "E24_finite_energy_24h",
        "Current export cap with finite energy equal to 24 hours at rated aggregate export.",
        1.0,
        24.0,
    ),
}


class ResourceSupportModel:
    """Path-dependent sensitivity wrapper; leaves V03RestorationModel untouched."""

    def __init__(self, model: V03RestorationModel):
        from dc_restoration.anchors.priority import validate_allocation_mode

        self.allocation_mode = validate_allocation_mode(model.allocation_mode)
        self.model = model
        self.current_aggregate_cap_mw = float(sum(model.dc_support_by_anchor.values()))
        self._potential_cache: Dict[Tuple[Tuple[str, ...], float], Dict[str, Any]] = {}

    def base_metrics(self, damaged_unrepaired: Iterable[str]) -> Any:
        key = tuple(sorted(map(str, damaged_unrepaired)))
        return self.model.base.metrics(key)

    def support_potential(self, damaged_unrepaired: Iterable[str], cap_mw: float) -> Dict[str, Any]:
        key = (tuple(sorted(map(str, damaged_unrepaired))), round(float(cap_mw), 9))
        if key in self._potential_cache:
            return self._potential_cache[key]
        base = self.model.base.metrics(key[0])
        if cap_mw <= EPS:
            out = {
                "base": base,
                "support_mw": 0.0,
                "critical_support_mw": 0.0,
                "voll_reduction_rate": 0.0,
                "support_by_anchor_mw": {str(a): 0.0 for a in self.model.dc_support_by_anchor},
                "critical_by_anchor_mw": {str(a): 0.0 for a in self.model.dc_support_by_anchor},
                "avoided_by_anchor_rate": {str(a): 0.0 for a in self.model.dc_support_by_anchor},
            }
            self._potential_cache[key] = out
            return out
        candidates: List[Tuple[bool, float, float, str, str]] = []
        unserved = set(base.unserved_load_ids)
        for anchor_id, load_ids in self.model.anchor_load_ids.items():
            for lid in load_ids:
                if lid not in unserved or lid not in self.model.load_info:
                    continue
                li = self.model.load_info[lid]
                mw = float(li.get("baseline_p_kw", 0.0)) / 1000.0
                voll = float(li.get("voll_usd_per_kwh", 0.0))
                crit = bool(li.get("is_critical", False))
                candidates.append((crit, voll, mw, str(anchor_id), str(lid)))
        # Archived fixed within-class priority; this is not the duration-dependent repair objective.
        candidates.sort(reverse=True)
        remaining = float(cap_mw)
        support = 0.0
        crit_support = 0.0
        voll_reduction = 0.0
        by_anchor = {str(a): 0.0 for a in self.model.dc_support_by_anchor}
        crit_by_anchor = {str(a): 0.0 for a in self.model.dc_support_by_anchor}
        avoided_by_anchor = {str(a): 0.0 for a in self.model.dc_support_by_anchor}
        for crit, voll, mw, anchor_id, _lid in candidates:
            if remaining <= EPS:
                break
            use = min(float(mw), remaining)
            support += use
            crit_support += use if crit else 0.0
            avoided = use * 1000.0 * voll
            voll_reduction += avoided
            by_anchor[anchor_id] = by_anchor.get(anchor_id, 0.0) + use
            if crit:
                crit_by_anchor[anchor_id] = crit_by_anchor.get(anchor_id, 0.0) + use
            avoided_by_anchor[anchor_id] = avoided_by_anchor.get(anchor_id, 0.0) + avoided
            remaining -= use
        out = {
            "base": base,
            "support_mw": support,
            "critical_support_mw": crit_support,
            "voll_reduction_rate": voll_reduction,
            "support_by_anchor_mw": by_anchor,
            "critical_by_anchor_mw": crit_by_anchor,
            "avoided_by_anchor_rate": avoided_by_anchor,
        }
        self._potential_cache[key] = out
        return out

    def decision_metrics(
        self, damaged_unrepaired: Iterable[str], cfg: ResourceConfig, energy_remaining_mwh: float
    ) -> PhysicalMetrics:
        cap = cfg.max_support_mw(self.current_aggregate_cap_mw)
        potential = self.support_potential(damaged_unrepaired, cap)
        base = potential["base"]
        if cap <= EPS or (cfg.finite_energy_hours is not None and energy_remaining_mwh <= EPS):
            return PhysicalMetrics(
                base.unserved_mw,
                base.critical_unserved_mw,
                base.voll_cost,
                base.island_count,
                base.unserved_load_ids,
            )
        support = float(potential["support_mw"])
        crit = float(potential["critical_support_mw"])
        voll = float(potential["voll_reduction_rate"])
        active = base.unserved_mw > 0.05 and support > 0.05 and voll > 1.0
        return PhysicalMetrics(
            max(0.0, base.unserved_mw - support),
            max(0.0, base.critical_unserved_mw - crit),
            max(0.0, base.voll_cost - voll),
            base.island_count,
            base.unserved_load_ids,
            dc_support_used_mw=support,
            dc_anchor_active=active,
            active_but_zero_support=0 if support > 0.05 or not active else 1,
            active_but_no_voll_reduction=0 if voll > 1.0 or not active else 1,
            useful_activation=1 if active and support > 0.05 and voll > 1.0 else 0,
        )

    def interval_accounting(
        self,
        damaged_unrepaired: Iterable[str],
        cfg: ResourceConfig,
        energy_remaining_mwh: float,
        duration_hours: float,
    ) -> Dict[str, Any]:
        cap = cfg.max_support_mw(self.current_aggregate_cap_mw)
        potential = self.support_potential(damaged_unrepaired, cap)
        base = potential["base"]
        full_support_mw = float(potential["support_mw"])
        full_crit_mw = float(potential["critical_support_mw"])
        full_voll_reduction = float(potential["voll_reduction_rate"])
        if cap <= EPS or full_support_mw <= EPS:
            fraction = 0.0
            energy_used = 0.0
        elif cfg.finite_energy_hours is None:
            fraction = 1.0
            energy_used = full_support_mw * duration_hours
        else:
            potential_mwh = full_support_mw * duration_hours
            energy_used = min(max(0.0, energy_remaining_mwh), potential_mwh)
            fraction = energy_used / max(potential_mwh, EPS)
        avg_support_mw = full_support_mw * fraction
        avg_crit_mw = full_crit_mw * fraction
        avg_voll_reduction = full_voll_reduction * fraction
        support_duration_hours = (
            energy_used / max(full_support_mw, EPS) if full_support_mw > EPS else 0.0
        )
        unmet_due_to_depletion_mwh = (
            max(0.0, full_support_mw * duration_hours - energy_used)
            if cfg.finite_energy_hours is not None
            else 0.0
        )
        by_anchor_mwh = {
            k: float(v) * duration_hours * fraction
            for k, v in potential["support_by_anchor_mw"].items()
        }
        active = base.unserved_mw > 0.05 and energy_used > EPS and avg_voll_reduction > 1.0
        return {
            "unserved_mw": max(0.0, base.unserved_mw - avg_support_mw),
            "critical_unserved_mw": max(0.0, base.critical_unserved_mw - avg_crit_mw),
            "voll_cost": max(0.0, base.voll_cost - avg_voll_reduction),
            "base_unserved_mw": base.unserved_mw,
            "base_critical_unserved_mw": base.critical_unserved_mw,
            "base_voll_cost": base.voll_cost,
            "support_mw_average": avg_support_mw,
            "support_mwh": energy_used,
            "support_duration_hours": support_duration_hours,
            "active": active,
            "useful": 1 if active and energy_used > EPS and avg_voll_reduction > 1.0 else 0,
            "active_but_zero": 1 if active and energy_used <= EPS else 0,
            "active_but_no_voll_reduction": 1 if active and avg_voll_reduction <= 1.0 else 0,
            "eligible_unserved_load_after_energy_depletion_mwh": unmet_due_to_depletion_mwh,
            "support_by_anchor_mwh": by_anchor_mwh,
            "energy_budget_violation_count": 1
            if cfg.finite_energy_hours is not None and energy_used - energy_remaining_mwh > 1e-6
            else 0,
            "no_support_after_depletion_violation_count": 1
            if cfg.finite_energy_hours is not None
            and energy_remaining_mwh <= EPS
            and energy_used > EPS
            else 0,
        }

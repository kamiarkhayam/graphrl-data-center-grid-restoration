import pytest

from dc_restoration.anchors.controller import RESOURCE_CONFIGS, ResourceSupportModel
from dc_restoration.economics.load_accounting import _support_by_load


def test_aggregate_export_cap_and_zone_eligibility(context):
    model, *_ = context
    remaining = set(model.scenario_edges("toy_damage"))
    result = model.metrics(remaining)
    allocation = _support_by_load(model, model.base.metrics(remaining), 15.0)
    assert result.dc_support_used_mw == pytest.approx(15)
    assert sum(allocation.values()) == pytest.approx(15)
    assert set(allocation) <= {"A", "B"}
    assert "D" not in allocation
    assert allocation["A"] == pytest.approx(15)  # critical-first allocation
    assert model.metrics(set()).dc_support_used_mw == 0


@pytest.mark.parametrize(
    "code,hours", [("E12_finite_energy_12h", 12), ("E24_finite_energy_24h", 24)]
)
def test_energy_budget_and_mid_interval_depletion(context, code, hours):
    model, *_ = context
    controller = ResourceSupportModel(model)
    cfg = RESOURCE_CONFIGS[code]
    energy = cfg.initial_energy_mwh(15)
    assert energy == 15 * hours
    state = set(model.scenario_edges("toy_damage"))
    result = controller.interval_accounting(state, cfg, energy, hours + 1)
    assert result["support_mwh"] == pytest.approx(energy)
    assert result["support_duration_hours"] == pytest.approx(hours)
    assert result["support_mw_average"] == pytest.approx(energy / (hours + 1))
    assert result["energy_budget_violation_count"] == 0
    exhausted = controller.interval_accounting(state, cfg, 0, 1)
    assert exhausted["support_mwh"] == 0
    assert exhausted["unserved_mw"] == pytest.approx(29)


def test_no_anchor_resource(context):
    model, *_ = context
    controller = ResourceSupportModel(model)
    cfg = RESOURCE_CONFIGS["R0_no_dc_anchor"]
    result = controller.interval_accounting(model.scenario_edges("toy_damage"), cfg, 0, 2)
    assert result["support_mwh"] == 0


def test_fixed_priority_compatibility_is_explicit(context):
    model, *_ = context
    assert model.allocation_mode == "critical_first_fixed_priority"
    # Both eligible blocks are noncritical: the fixed legacy field resolves priority.
    model.load_info["A"]["is_critical"] = False
    model.load_info["A"]["voll_usd_per_kwh"] = 1
    model.load_info["B"]["voll_usd_per_kwh"] = 2
    remaining = model.scenario_edges("toy_damage")
    allocation = _support_by_load(model, model.base.metrics(remaining), 15)
    assert allocation == {"B": 6, "A": 9}
    model.load_info["A"]["is_critical"] = True
    allocation = _support_by_load(model, model.base.metrics(remaining), 15)
    assert allocation == {"A": 15}


def test_unsupported_priority_cannot_silently_change_dispatch(context):
    from dc_restoration.anchors.priority import validate_allocation_mode

    with pytest.raises(ValueError, match="fixed within-class"):
        validate_allocation_mode("economic_priority")

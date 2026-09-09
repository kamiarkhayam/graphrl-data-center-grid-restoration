import numpy as np
import pytest

from dc_restoration.economics.duration_damage import _cdf_per_interrupted_kw
from dc_restoration.economics.interruption_damage import _accrue
from dc_restoration.evaluation.bounded_ga import SourceDurationValuator


def test_duration_function_known_values():
    value = _cdf_per_interrupted_kw(np.array([0, 12, 24, 48, 96]), 2, [24, 48, 96], [1, 2, 4])
    assert value == pytest.approx([0, 24, 48, 96, 192])


def test_critical_cost_units(context):
    _, _, valuator, _, _ = context
    assert valuator.cost_from_energy({}) == 0
    assert valuator.cost_from_energy({"A": 2000}) == pytest.approx(200000)


def test_marginal_damage_and_accrual(context):
    _, _, valuator, _, _ = context
    energy = {"B": 144000.0}
    rates = {"B": 6000.0}
    increment = valuator.marginal_one_hour_cost(energy, rates)
    original = valuator.cost_from_energy(energy)
    _accrue(energy, rates, 1.0)
    assert increment == pytest.approx(valuator.cost_from_energy(energy) - original)
    assert increment > 0


def test_ga_and_policy_economic_objectives_agree(context):
    _, inventory, valuator, _, _ = context
    # Both copied formulations must price the same load-level interruption identically.
    source = SourceDurationValuator(
        inventory, {"primary": valuator.parameters}, valuator.critical_value
    )
    energy = {"A": 1234, "B": 240000, "D": 120000}
    assert source.cost(energy) == pytest.approx(valuator.cost_from_energy(energy))

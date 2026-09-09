import pytest

from dc_restoration.restoration.validation import validate_case, validate_repair_sequence


def test_connectivity_and_repair(context):
    model, *_ = context
    validate_case(model)
    damaged = set(model.scenario_edges("toy_damage"))
    assert model.base.metrics(damaged).unserved_mw == pytest.approx(29)
    assert set(model.base.metrics(damaged).unserved_load_ids) == {"A", "B", "D"}
    assert model.base.metrics(damaged - {"e0"}).unserved_mw == pytest.approx(26)
    assert model.base.metrics(set()).unserved_mw == 0


@pytest.mark.parametrize("order", [["e0", "e0"], ["e3"], ["missing"]])
def test_invalid_repairs_are_rejected(context, order):
    model, *_ = context
    with pytest.raises(ValueError, match="unrepaired damaged"):
        validate_repair_sequence(model.scenario_edges("toy_damage"), order)


def test_partial_repair_order_is_valid(context):
    model, *_ = context
    assert validate_repair_sequence(model.scenario_edges("toy_damage"), ["e0"]) == ["e0"]

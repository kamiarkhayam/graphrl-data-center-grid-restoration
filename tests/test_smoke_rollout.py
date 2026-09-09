import json

import pytest
import torch

from dc_restoration.anchors.controller import RESOURCE_CONFIGS
from dc_restoration.evaluation.trajectories import (
    HarmonizedResourceAdapter,
    _dc_rollout,
    _trajectory_from_order,
)
from dc_restoration.policies.training import _rollout_harmonized
from dc_restoration.restoration.validation import validate_repair_sequence


@pytest.mark.parametrize("architecture", ["greedy", "random"])
def test_small_deterministic_rollout(context, architecture):
    model, inventory, valuator, adapter, config = context
    first, _ = _rollout_harmonized(
        architecture,
        model,
        adapter,
        valuator,
        "toy_damage",
        config,
        torch.device("cpu"),
        random_seed=32,
    )
    second, _ = _rollout_harmonized(
        architecture,
        model,
        adapter,
        valuator,
        "toy_damage",
        config,
        torch.device("cpu"),
        random_seed=32,
    )
    assert first["repair_sequence_edge_ids"] == second["repair_sequence_edge_ids"]
    assert (
        first["harmonized_economic_damage_dollars"] == second["harmonized_economic_damage_dollars"]
    )
    order = json.loads(first["repair_sequence_edge_ids"])
    validate_repair_sequence(model.scenario_edges("toy_damage"), order)
    assert 0 < len(order) <= 3
    _, metrics = _trajectory_from_order(
        model, inventory, valuator, "toy_damage", architecture, architecture, order
    )
    assert metrics["harmonized_economic_damage_dollars"] == pytest.approx(
        first["harmonized_economic_damage_dollars"]
    )


def test_small_finite_energy_rollout(context):
    model, inventory, valuator, adapter, _ = context
    resource_adapter = HarmonizedResourceAdapter(model, inventory, valuator, adapter)
    metrics, _ = _dc_rollout(
        model,
        inventory,
        valuator,
        resource_adapter,
        "toy_damage",
        "Greedy Harmonized",
        "Greedy",
        RESOURCE_CONFIGS["E12_finite_energy_12h"],
        5,
        torch.device("cpu"),
    )
    assert metrics["dc_support_used_mwh"] <= 180

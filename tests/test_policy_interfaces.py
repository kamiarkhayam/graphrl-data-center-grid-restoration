from pathlib import Path

import numpy as np
import pytest
import torch

from dc_restoration.policies.encoders import CNNEncoder, EncoderActorCritic, MLPScorer
from dc_restoration.policies.graph import to_torch_batch
from dc_restoration.policies.registry import METHODS, TRAINABLE
from dc_restoration.policies.training import _make_true_gnn


def test_candidate_mask_and_repaired_exclusion(context):
    model, _, _, adapter, _ = context
    damaged = model.scenario_edges("toy_damage")
    _, candidates, observation = adapter.build_state(
        damaged, model.scenario_repair_times("toy_damage"), {"e0"}, {}, 5
    )
    assert observation["candidate_mask"].tolist() == [True, True, False, False, False]
    assert {row["edge_id"] for row in candidates} == {"e1", "e2"}
    assert all(np.isfinite(value).all() for value in observation.values())


def test_all_method_configs_exist():
    root = Path(__file__).resolve().parents[1]
    assert len(METHODS) == 8
    assert len(TRAINABLE) == 5
    for name, method in METHODS.items():
        area = "evaluation" if name == "Bounded GA" else "policies"
        assert (root / "configs" / area / f"{method.configuration}.yaml").is_file()


@pytest.mark.parametrize("method", TRAINABLE)
def test_forward_interface_for_each_trainable_method(context, method):
    model, _, _, adapter, config = context
    torch.manual_seed(7)
    _, _, obs = adapter.build_state(
        model.scenario_edges("toy_damage"), model.scenario_repair_times("toy_damage"), set(), {}, 5
    )
    mask = torch.tensor(obs["candidate_mask"][None])
    if METHODS[method].architecture == "true_gnn":
        arrays = {k: v[None] for k, v in obs.items()}
        arrays["edge_index"] = adapter.edge_index
        policy, edge_index, _ = _make_true_gnn(arrays, config, torch.device("cpu"))
        with torch.no_grad():
            output = policy(to_torch_batch(arrays, torch.device("cpu")), edge_index)
        logits, value = output["logits"], output["value"]
    else:
        features = torch.tensor(adapter.local_features(obs)[None])
        if METHODS[method].architecture == "mlp":
            actor = MLPScorer(features.shape[-1], 16)
            raster = None
        else:
            actor = CNNEncoder(features.shape[-1], 6, 4, hidden=16, conv_channels=4, spatial_dim=8)
            raster = torch.zeros((1, 6, 4, 4))
        policy = EncoderActorCritic(actor, 16)
        with torch.no_grad():
            logits, value = policy(features, mask, raster)
    assert logits.shape == (1, 5)
    assert torch.isfinite(value).all()
    probability = torch.softmax(logits, dim=1)
    assert probability[0, 3:].sum() == 0
    assert probability.sum() == pytest.approx(1)

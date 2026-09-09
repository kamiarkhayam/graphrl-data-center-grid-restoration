"""Read-only installation validation on an invented six-node fixture."""

import importlib
import pkgutil

from dc_restoration.configuration import training_config
from dc_restoration.paths import repository_root


def validate() -> dict:
    import torch

    import dc_restoration
    from dc_restoration.policies import training
    from dc_restoration.restoration.validation import validate_case, validate_repair_sequence

    modules = list(pkgutil.walk_packages(dc_restoration.__path__, dc_restoration.__name__ + "."))
    for module in modules:
        importlib.import_module(module.name)
    config = training_config("configs/policies/training.yaml")
    config["paths"]["case_dir"] = str(repository_root() / "tests/fixtures/tiny_case")
    model, inventory, valuator, adapter = training._load_context(config)
    validate_case(model)
    metrics, _ = training._rollout_harmonized(
        "greedy", model, adapter, valuator, "toy_damage", config, torch.device("cpu")
    )
    import json

    order = json.loads(metrics["repair_sequence_edge_ids"])
    validate_repair_sequence(model.scenario_edges("toy_damage"), order)
    assert len(order) <= 3
    assert metrics["harmonized_economic_damage_dollars"] >= 0
    return {
        "imported_modules": len(modules),
        "fixture_nodes": len(model.nodes),
        "fixture_repairs": len(order),
        "status": "passed",
        "paper_runs_performed": False,
    }

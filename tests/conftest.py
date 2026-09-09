from pathlib import Path

import pytest

from dc_restoration.configuration import training_config
from dc_restoration.policies.training import _load_context


@pytest.fixture
def context(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("DC_RESTORATION_ROOT", str(root))
    config = training_config(root / "configs/policies/training.yaml")
    config["paths"]["case_dir"] = str(root / "tests/fixtures/tiny_case")
    return (*_load_context(config), config)

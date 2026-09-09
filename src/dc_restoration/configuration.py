"""Load final YAML configurations without changing scientific parameters."""

from pathlib import Path

import yaml

from dc_restoration.paths import require_file, resolve_path


def load_config(path: str | Path) -> dict:
    with require_file(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def training_config(path: str | Path) -> dict:
    value = load_config(path)
    if "training_config" in value:
        value = load_config(value["training_config"])
    return value


def use_case(config: dict, case: str | Path | None) -> None:
    if case is not None:
        config["paths"]["case_dir"] = str(resolve_path(case))

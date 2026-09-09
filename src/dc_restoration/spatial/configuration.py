"""Path configuration shared by the three explicit spatial-map commands."""

from importlib.resources import files
from pathlib import Path

import yaml

from dc_restoration.paths import require_file, resolve_output_path


def load_map_config(path=None, **overrides):
    text = (
        require_file(path).read_text(encoding="utf-8")
        if path is not None
        else files("dc_restoration.spatial").joinpath("intro_map.yaml").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(text)
    keys = {"cache_dir", "prepared_dir", "output_dir"}
    if not isinstance(config, dict) or set(config) != keys or set(overrides) - keys:
        raise ValueError("Map configuration requires only cache_dir, prepared_dir and output_dir")
    config.update({k: v for k, v in overrides.items() if v is not None})
    if any(not isinstance(v, (str, Path)) or not str(v).strip() for v in config.values()):
        raise ValueError("Map paths must be nonempty paths")
    resolved = {k: resolve_output_path(v) for k, v in config.items()}
    for name, path in resolved.items():
        for other, target in resolved.items():
            if name != other and (path == target or path.is_relative_to(target)):
                raise ValueError("Map cache, prepared and figure directories must not overlap")
    return resolved

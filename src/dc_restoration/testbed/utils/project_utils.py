from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dc_restoration.paths import repository_root, resolve_output_path

ROOT_DIR = repository_root()


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    raw_transmission_dir: Path
    raw_distribution_dir: Path
    raw_mapping_dir: Path
    parsed_transmission_dir: Path
    parsed_distribution_dir: Path
    candidate_regions_dir: Path
    selected_region_dir: Path
    final_case_dir: Path
    plots_dir: Path
    logs_dir: Path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_file = Path(config_path) if config_path else ROOT_DIR / "configs/testbed/base.yaml"
    with config_file.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_paths(config: dict[str, Any]) -> ProjectPaths:
    path_cfg = config["paths"]

    def resolve(key: str) -> Path:
        return (ROOT_DIR / path_cfg[key]).resolve()

    return ProjectPaths(
        root=ROOT_DIR,
        raw_transmission_dir=resolve("raw_transmission_dir"),
        raw_distribution_dir=resolve("raw_distribution_dir"),
        raw_mapping_dir=resolve("raw_mapping_dir"),
        parsed_transmission_dir=resolve("parsed_transmission_dir"),
        parsed_distribution_dir=resolve("parsed_distribution_dir"),
        candidate_regions_dir=resolve("candidate_regions_dir"),
        selected_region_dir=resolve("selected_region_dir"),
        final_case_dir=resolve("final_case_dir"),
        plots_dir=resolve("plots_dir"),
        logs_dir=resolve("logs_dir"),
    )


def ensure_project_dirs(paths: ProjectPaths) -> None:
    # Validate all configured trees before the first directory is created.
    for field in fields(paths):
        if field.name != "root":
            resolve_output_path(getattr(paths, field.name))
    for field in fields(paths):
        path = getattr(paths, field.name)
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)


def setup_logging(script_name: str, config: dict[str, Any]) -> logging.Logger:
    paths = get_paths(config)
    ensure_project_dirs(paths)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paths.logs_dir / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info("Logging to %s", log_path)
    return logger


def default_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs/testbed/base.yaml"),
        help="Path to configs/testbed/base.yaml",
    )
    return parser


def write_json(data: Any, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def write_text(text: str, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(text)


def write_iteration_summary(
    config: dict[str, Any],
    iteration: int,
    attempted: list[str],
    worked: list[str],
    failed: list[str],
    assumptions: list[str],
    counts: dict[str, Any],
    next_action: str,
) -> Path:
    paths = get_paths(config)
    summary_path = paths.logs_dir / f"iteration_summary_{iteration}.md"
    lines = [
        f"# Iteration {iteration}",
        "",
        "## What was attempted",
        *[f"- {item}" for item in attempted],
        "",
        "## What worked",
        *([f"- {item}" for item in worked] if worked else ["- None yet."]),
        "",
        "## What failed",
        *([f"- {item}" for item in failed] if failed else ["- No blocking failures."]),
        "",
        "## Selected assumptions",
        *([f"- {item}" for item in assumptions] if assumptions else ["- No new assumptions."]),
        "",
        "## Current counts",
        *(
            [f"- {key}: {value}" for key, value in counts.items()]
            if counts
            else ["- Not available yet."]
        ),
        "",
        "## Next action",
        f"- {next_action}",
        "",
    ]
    write_text("\n".join(lines), summary_path)
    return summary_path


def copy_config_to_final_case(config: dict[str, Any]) -> Path:
    paths = get_paths(config)
    out_path = paths.final_case_dir / "configs/testbed/base.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return out_path

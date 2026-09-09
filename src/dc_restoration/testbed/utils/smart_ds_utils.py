from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from .s3_utils import list_common_prefixes

SUBSTATION_RE = re.compile(r"^(p\d+[ru])hs(\d+)_([0-9]+)$", re.IGNORECASE)
SERVICE_AREA_RE = re.compile(r"^(P\d+[RU])/$")


@dataclass(slots=True)
class SmartDSSubstationRef:
    service_area: str
    substation_id: str
    distribution_voltage: float | None
    root_bus_name: str


def service_area_from_substation_id(substation_id: str) -> str:
    match = SUBSTATION_RE.match(substation_id)
    if not match:
        raise ValueError(f"Unexpected SMART-DS substation id format: {substation_id}")
    return match.group(1).upper()


def root_bus_name_from_substation_id(substation_id: str) -> str:
    match = SUBSTATION_RE.match(substation_id)
    if not match:
        raise ValueError(f"Unexpected SMART-DS substation id format: {substation_id}")
    return f"{match.group(1).lower()}hs{match.group(2)}"


def infer_distribution_voltage_from_substation_id(substation_id: str) -> float | None:
    match = SUBSTATION_RE.match(substation_id)
    if not match:
        return None
    try:
        return float(match.group(3)) / 100.0
    except ValueError:
        return None


def infer_service_area_prefix(dataset_prefix: str, service_area: str, scenario: str) -> str:
    return f"{dataset_prefix}/{service_area}/{scenario}"


def list_service_areas(base_url: str, dataset_prefix: str) -> list[str]:
    prefixes = list_common_prefixes(base_url, f"{dataset_prefix}/", delimiter="/")
    areas: list[str] = []
    for prefix in prefixes:
        part = prefix.rstrip("/").split("/")[-1]
        if SERVICE_AREA_RE.match(f"{part}/"):
            areas.append(part)
    return sorted(areas)


def fetch_text_from_s3(base_url: str, key: str) -> str:
    response = requests.get(f"{base_url}/{key}", timeout=120)
    response.raise_for_status()
    return response.text


def fetch_line_count_from_s3(base_url: str, key: str) -> int:
    response = requests.get(f"{base_url}/{key}", stream=True, timeout=120)
    response.raise_for_status()
    return sum(1 for _ in response.iter_lines())


def list_substation_feeder_ids(
    base_url: str,
    dataset_prefix: str,
    scenario: str,
    substation_id: str,
) -> list[str]:
    service_area = service_area_from_substation_id(substation_id)
    prefix = f"{dataset_prefix}/{service_area}/{scenario}/{substation_id}/"
    feeder_prefixes = list_common_prefixes(base_url, prefix, delimiter="/")
    feeder_ids = [item.rstrip("/").split("/")[-1] for item in feeder_prefixes]
    return sorted(feeder_ids)


def parse_buscoords_text(buscoords_text: str) -> dict[str, tuple[float, float]]:
    coordinates: dict[str, tuple[float, float]] = {}
    for line in buscoords_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        bus = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            continue
        coordinates[bus] = (y, x)
    return coordinates


def candidate_root_bus_names(substation_id: str) -> Iterable[str]:
    ref = root_bus_name_from_substation_id(substation_id)
    yield f"{ref}_69"
    yield f"{ref}_138"
    yield ref


def write_text_file(path: str | Path, text: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path

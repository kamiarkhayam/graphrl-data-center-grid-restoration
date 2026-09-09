from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

import dss
import pyarrow as pa
import pyarrow.parquet as pq
import requests

PROPERTY_RE = re.compile(r"\s*=\s*")
COMMENT_RE = re.compile(r"^\s*!|^\s*//")
LOADSHAPE_FILE_RE = re.compile(r"file=([^\)]+)")


def strip_inline_comment(line: str) -> str:
    if "!" in line:
        line = line.split("!", 1)[0]
    if "//" in line:
        line = line.split("//", 1)[0]
    return line.strip()


def dss_value(value: Any) -> Any:
    return value() if callable(value) else value


def parse_new_element_line(line: str, source_file: Path) -> dict[str, Any] | None:
    clean = strip_inline_comment(line)
    if not clean or COMMENT_RE.search(clean):
        return None
    if not clean.lower().startswith("new "):
        return None

    clean = PROPERTY_RE.sub("=", clean)
    tokens = shlex.split(clean, comments=False, posix=False)
    if len(tokens) < 2:
        return None

    element_ref = tokens[1]
    if "." in element_ref:
        element_class, element_name = element_ref.split(".", 1)
    else:
        element_class, element_name = "Unknown", element_ref

    props: dict[str, Any] = {}
    for token in tokens[2:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in props:
            if isinstance(props[key], list):
                props[key].append(value)
            else:
                props[key] = [props[key], value]
        else:
            props[key] = value

    return {
        "element_class": element_class,
        "element_name": element_name,
        "properties": props,
        "source_file": str(source_file),
    }


def parse_new_elements_from_file(path: Path) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_new_element_line(line, path)
        if parsed:
            elements.append(parsed)
    return elements


def bus_name_from_ref(bus_ref: str | None) -> str | None:
    if not bus_ref:
        return None
    return str(bus_ref).split(".", 1)[0]


def phases_from_bus_ref(bus_ref: str | None) -> str | None:
    if not bus_ref or "." not in str(bus_ref):
        return None
    return ".".join(str(bus_ref).split(".")[1:])


def normalize_bool_token(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"y", "yes", "true"}:
        return True
    if text in {"n", "no", "false"}:
        return False
    return None


def numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def infer_feeder_id_from_path(path: Path, substation_id: str) -> str:
    parent = path.parent.name
    if parent == substation_id:
        return substation_id
    return parent


def infer_line_installation(linecode: str | None) -> tuple[bool | None, bool | None]:
    if not linecode:
        return None, None
    code = linecode.upper()
    is_overhead = "OH" in code
    is_underground = "UG" in code or "UDG" in code
    return is_overhead, is_underground


def infer_customer_class(profile_id: str | None, load_name: str | None = None) -> str:
    text = f"{profile_id or ''} {load_name or ''}".lower()
    if "critical" in text:
        return "critical"
    if text.startswith("res_") or " res_" in text:
        return "residential"
    if text.startswith("com_") or " com_" in text:
        return "commercial"
    if any(tag in text for tag in ["ind_", "industrial", "lgcom", "large"]):
        return "industrial_or_large_commercial"
    return "unknown"


def extract_profile_definitions(loadshapes_path: Path) -> list[dict[str, Any]]:
    defs: list[dict[str, Any]] = []
    for item in parse_new_elements_from_file(loadshapes_path):
        if item["element_class"].lower() != "loadshape":
            continue
        props = item["properties"]
        p_match = LOADSHAPE_FILE_RE.search(str(props.get("mult", "")))
        q_match = LOADSHAPE_FILE_RE.search(str(props.get("qmult", "")))
        defs.append(
            {
                "profile_id": item["element_name"],
                "p_file": Path(p_match.group(1)).name if p_match else None,
                "q_file": Path(q_match.group(1)).name if q_match else None,
                "npts": int(float(props.get("npts", 0))),
                "interval_hours": float(props.get("interval", 0.25)),
                "source_file": str(loadshapes_path),
            }
        )
    return defs


def ensure_profile_csvs(
    base_url: str,
    profiles_prefix: str,
    destination_dir: Path,
    profile_defs: list[dict[str, Any]],
    logger,
) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    filenames = sorted(
        {
            filename
            for profile_def in profile_defs
            for filename in [profile_def.get("p_file"), profile_def.get("q_file")]
            if filename
        }
    )
    for filename in filenames:
        local_path = destination_dir / filename
        if local_path.exists():
            continue
        key = f"{profiles_prefix}/{filename}"
        response = requests.get(f"{base_url}/{key}", stream=True, timeout=120)
        response.raise_for_status()
        with local_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        downloaded.append(local_path)
    logger.info("Ensured %s profile CSVs under %s", len(filenames), destination_dir)
    return downloaded


def build_snapshot_master(original_master_path: Path) -> Path:
    output_path = original_master_path.with_name("Master_snapshot.dss")
    kept: list[str] = []
    for raw_line in original_master_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("solve ") or lower.startswith("plot ") or lower.startswith("export "):
            continue
        kept.append(raw_line)
    kept.extend(
        [
            "",
            "Set mode=snapshot",
            "Set controlmode=static",
            "Solve",
            "",
        ]
    )
    output_path.write_text("\n".join(kept), encoding="utf-8")
    return output_path


def compile_dss_master(master_path: Path):
    ctx = dss.dss.NewContext()
    ctx.Text.Command = f"Compile [{master_path.resolve().as_posix()}]"
    return ctx


def safe_active_bus_data(ctx, bus_name: str) -> dict[str, Any]:
    circuit = ctx.ActiveCircuit
    try:
        circuit.SetActiveBus(bus_name)
        bus = circuit.ActiveBus
        nodes = dss_value(bus.Nodes)
        kv_base_ln = dss_value(bus.kVBase)
    except Exception:
        return {
            "base_kv_ll": None,
            "phases": None,
            "distance": None,
            "x": None,
            "y": None,
        }
    if kv_base_ln and len(nodes) > 1:
        base_kv_ll = kv_base_ln * math.sqrt(3.0)
    else:
        base_kv_ll = kv_base_ln
    return {
        "base_kv_ll": float(base_kv_ll) if base_kv_ll else None,
        "phases": ".".join(str(node) for node in nodes) if len(nodes) else None,
        "distance": float(dss_value(bus.Distance)) if dss_value(bus.Distance) is not None else None,
        "x": float(dss_value(bus.x)) if dss_value(bus.x) is not None else None,
        "y": float(dss_value(bus.y)) if dss_value(bus.y) is not None else None,
    }


def circuit_voltage_report(ctx) -> dict[str, Any]:
    circuit = ctx.ActiveCircuit
    solution = circuit.Solution
    vmag_pu = list(dss_value(circuit.AllBusVmagPu))
    return {
        "converged": bool(dss_value(solution.Converged)),
        "num_buses": int(dss_value(circuit.NumBuses)),
        "num_nodes": int(dss_value(circuit.NumNodes)),
        "num_elements": int(dss_value(circuit.NumCktElements)),
        "voltage_pu_min": float(min(vmag_pu)) if vmag_pu else None,
        "voltage_pu_max": float(max(vmag_pu)) if vmag_pu else None,
        "total_power_kw_kvar": [float(x) for x in dss_value(circuit.TotalPower)],
    }


def write_profile_parquet(profile_rows: list[dict[str, Any]], output_path: Path) -> None:
    table = pa.table(
        {
            "profile_id": pa.array([row["profile_id"] for row in profile_rows]),
            "p_file": pa.array([row.get("p_file") for row in profile_rows]),
            "q_file": pa.array([row.get("q_file") for row in profile_rows]),
            "npts": pa.array([int(row["npts"]) for row in profile_rows], type=pa.int32()),
            "interval_hours": pa.array(
                [float(row["interval_hours"]) for row in profile_rows], type=pa.float32()
            ),
            "customer_class": pa.array(
                [row.get("customer_class", "unknown") for row in profile_rows]
            ),
            "p_values": pa.array(
                [row["p_values"] for row in profile_rows],
                type=pa.list_(pa.float32()),
            ),
            "q_values": pa.array(
                [row["q_values"] for row in profile_rows],
                type=pa.list_(pa.float32()),
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def read_profile_csv(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


def json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

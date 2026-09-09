from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from dc_restoration.paths import repository_root

SCRIPT_DIR = Path(__file__).resolve().parent

from dc_restoration.testbed.utils.dss_parse_utils import (  # noqa: E402
    build_snapshot_master,
    bus_name_from_ref,
    circuit_voltage_report,
    compile_dss_master,
    dss_value,
    ensure_profile_csvs,
    extract_profile_definitions,
    infer_customer_class,
    infer_feeder_id_from_path,
    infer_line_installation,
    json_dump,
    normalize_bool_token,
    numeric_or_none,
    parse_new_elements_from_file,
    phases_from_bus_ref,
    read_profile_csv,
    safe_active_bus_data,
    write_profile_parquet,
)
from dc_restoration.testbed.utils.project_utils import (  # noqa: E402
    default_arg_parser,
    get_paths,
    load_config,
    setup_logging,
)
from dc_restoration.testbed.utils.smart_ds_utils import (  # noqa: E402
    parse_buscoords_text,
    service_area_from_substation_id,
)


def scenario_root(paths, service_area: str, scenario: str) -> Path:
    return paths.raw_distribution_dir / service_area / Path(*Path(scenario).parts)


def gather_file_definitions(substation_root: Path, pattern: str) -> list[Path]:
    return sorted(substation_root.rglob(pattern))


def build_linecode_lookup(linecode_files: list[Path]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for path in linecode_files:
        for item in parse_new_elements_from_file(path):
            if item["element_class"].lower() != "linecode":
                continue
            lookup[item["element_name"]] = item["properties"]
    return lookup


def select_bus_feeder(
    bus_name: str, bus_feeder_votes: dict[str, Counter], substation_id: str
) -> str:
    votes = bus_feeder_votes.get(bus_name)
    if not votes:
        return substation_id
    return votes.most_common(1)[0][0]


def parse_selected_distribution(
    config: dict,
    substation_id: str,
    logger,
) -> dict:
    paths = get_paths(config)
    service_area = service_area_from_substation_id(substation_id)
    scenario = config["sources"]["smart_ds"]["scenario"]
    base_url = config["sources"]["smart_ds"]["s3_rest_base_url"]
    profiles_prefix = config["sources"]["smart_ds"]["profiles_prefix"]

    local_scenario_root = scenario_root(paths, service_area, scenario)
    local_substation_root = local_scenario_root / substation_id
    if not local_substation_root.exists():
        raise FileNotFoundError(f"Missing downloaded substation directory: {local_substation_root}")

    loadshape_path = local_substation_root / "LoadShapes.dss"
    profile_defs = extract_profile_definitions(loadshape_path)
    ensure_profile_csvs(
        base_url=base_url,
        profiles_prefix=profiles_prefix,
        destination_dir=paths.raw_distribution_dir / "profiles",
        profile_defs=profile_defs,
        logger=logger,
    )
    snapshot_master = build_snapshot_master(local_substation_root / "Master.dss")
    ctx = compile_dss_master(snapshot_master)
    powerflow_report = circuit_voltage_report(ctx)

    buscoords = parse_buscoords_text(
        (local_substation_root / "Buscoords.dss").read_text(encoding="utf-8")
    )
    linecode_lookup = build_linecode_lookup(
        gather_file_definitions(local_substation_root, "LineCodes.dss")
    )

    line_files = gather_file_definitions(local_substation_root, "Lines.dss")
    transformer_files = gather_file_definitions(local_substation_root, "Transformers.dss")
    regulator_files = gather_file_definitions(local_substation_root, "Regulators.dss")
    load_files = gather_file_definitions(local_substation_root, "Loads.dss")

    bus_feeder_votes: dict[str, Counter] = defaultdict(Counter)
    raw_line_rows: list[dict] = []
    raw_transformer_rows: list[dict] = []
    raw_load_rows: list[dict] = []

    circuit = ctx.ActiveCircuit
    line_name_set = set(dss_value(circuit.Lines.AllNames))

    for path in line_files:
        feeder_id = infer_feeder_id_from_path(path, substation_id)
        for item in parse_new_elements_from_file(path):
            if item["element_class"].lower() != "line":
                continue
            props = item["properties"]
            name = item["element_name"]
            bus1 = bus_name_from_ref(props.get("bus1"))
            bus2 = bus_name_from_ref(props.get("bus2"))
            if bus1:
                bus_feeder_votes[bus1][feeder_id] += 1
            if bus2:
                bus_feeder_votes[bus2][feeder_id] += 1

            if name in line_name_set:
                circuit.Lines.Name = name
                norm_amps = circuit.Lines.NormAmps
                r1 = circuit.Lines.R1
                x1 = circuit.Lines.X1
            else:
                norm_amps = None
                r1 = None
                x1 = None

            linecode = props.get("Linecode") or props.get("linecode")
            linecode_props = linecode_lookup.get(str(linecode), {})
            is_switch = normalize_bool_token(props.get("switch"))
            if is_switch is None and linecode:
                is_switch = "SWITCH" in str(linecode).upper() or "BREAKER" in str(linecode).upper()
            normal_enabled = normalize_bool_token(props.get("enabled"))
            is_overhead, is_underground = infer_line_installation(
                str(linecode) if linecode else None
            )

            raw_line_rows.append(
                {
                    "raw_edge_id": name,
                    "feeder_id": feeder_id,
                    "from_bus": bus1,
                    "to_bus": bus2,
                    "length": numeric_or_none(props.get("Length")),
                    "length_unit": props.get("Units"),
                    "phases": props.get("phases"),
                    "linecode": linecode,
                    "r": numeric_or_none(r1) or numeric_or_none(linecode_props.get("r1")),
                    "x": numeric_or_none(x1) or numeric_or_none(linecode_props.get("x1")),
                    "normal_amps": numeric_or_none(norm_amps)
                    or numeric_or_none(linecode_props.get("normamps")),
                    "is_switch": bool(is_switch),
                    "normal_status": "closed" if normal_enabled is not False else "open",
                    "is_overhead": is_overhead,
                    "is_underground": is_underground,
                    "source_file": str(path),
                }
            )

    for path in transformer_files:
        feeder_id = infer_feeder_id_from_path(path, substation_id)
        for item in parse_new_elements_from_file(path):
            if item["element_class"].lower() != "transformer":
                continue
            props = item["properties"]
            name = item["element_name"]
            buses = props.get("bus")
            if not isinstance(buses, list):
                buses = [buses] if buses else []
            bus_names = [bus_name_from_ref(bus) for bus in buses]
            for bus_name in bus_names:
                if bus_name:
                    bus_feeder_votes[bus_name][feeder_id] += 1

            raw_transformer_rows.append(
                {
                    "raw_transformer_id": name,
                    "feeder_id": feeder_id,
                    "substation_id": substation_id,
                    "bus1": bus_names[0] if len(bus_names) > 0 else None,
                    "bus2": bus_names[1] if len(bus_names) > 1 else None,
                    "phases": props.get("phases"),
                    "windings": numeric_or_none(props.get("windings")),
                    "kva": numeric_or_none(props.get("kva")),
                    "kv_wdg1": numeric_or_none(
                        props.get("Kv")[0] if isinstance(props.get("Kv"), list) else props.get("Kv")
                    ),
                    "xhl": numeric_or_none(props.get("XHL")),
                    "source_file": str(path),
                }
            )

    raw_switch_rows = [
        {
            "raw_switch_id": row["raw_edge_id"],
            "feeder_id": row["feeder_id"],
            "from_bus": row["from_bus"],
            "to_bus": row["to_bus"],
            "normal_status": row["normal_status"],
            "source_file": row["source_file"],
        }
        for row in raw_line_rows
        if row["is_switch"]
    ]

    for path in load_files:
        feeder_id = infer_feeder_id_from_path(path, substation_id)
        for item in parse_new_elements_from_file(path):
            if item["element_class"].lower() != "load":
                continue
            props = item["properties"]
            name = item["element_name"]
            bus = bus_name_from_ref(props.get("bus1"))
            if bus:
                bus_feeder_votes[bus][feeder_id] += 1
            profile_id = props.get("yearly") or props.get("daily") or props.get("duty")
            inferred_class = infer_customer_class(profile_id, name)
            raw_load_rows.append(
                {
                    "raw_load_id": name,
                    "bus": bus,
                    "feeder_id": feeder_id,
                    "substation_id": substation_id,
                    "p_kw": numeric_or_none(props.get("kW")),
                    "q_kvar": numeric_or_none(props.get("kvar")),
                    "customer_class": None,
                    "inferred_customer_class": inferred_class,
                    "profile_id": profile_id,
                    "phases": props.get("Phases") or phases_from_bus_ref(props.get("bus1")),
                    "voltage_kv": numeric_or_none(props.get("kV")),
                    "source_file": str(path),
                }
            )

    all_bus_names = set(buscoords.keys())
    for row in raw_line_rows:
        all_bus_names.update([row["from_bus"], row["to_bus"]])
    for row in raw_transformer_rows:
        all_bus_names.update([row["bus1"], row["bus2"]])
    for row in raw_load_rows:
        all_bus_names.add(row["bus"])
    all_bus_names = {name for name in all_bus_names if name}

    raw_bus_rows: list[dict] = []
    for bus_name in sorted(all_bus_names):
        lat_lon = buscoords.get(bus_name, (None, None))
        active_bus = safe_active_bus_data(ctx, bus_name)
        raw_bus_rows.append(
            {
                "raw_bus_id": bus_name,
                "feeder_id": select_bus_feeder(bus_name, bus_feeder_votes, substation_id),
                "substation_id": substation_id,
                "latitude": lat_lon[0],
                "longitude": lat_lon[1],
                "x": lat_lon[1],
                "y": lat_lon[0],
                "base_kv": active_bus["base_kv_ll"],
                "phases": active_bus["phases"],
                "distance_from_substation": active_bus["distance"],
            }
        )

    profile_def_lookup = {item["profile_id"]: item for item in profile_defs}
    profile_rows = []
    profiles_dir = paths.raw_distribution_dir / "profiles"
    for profile_id, profile_def in profile_def_lookup.items():
        inferred_class = infer_customer_class(profile_id)
        p_values = read_profile_csv(profiles_dir / profile_def["p_file"])
        q_values = read_profile_csv(profiles_dir / profile_def["q_file"])
        profile_rows.append(
            {
                "profile_id": profile_id,
                "p_file": profile_def["p_file"],
                "q_file": profile_def["q_file"],
                "npts": profile_def["npts"],
                "interval_hours": profile_def["interval_hours"],
                "customer_class": inferred_class,
                "p_values": p_values,
                "q_values": q_values,
            }
        )

    raw_buses_df = pd.DataFrame(raw_bus_rows)
    raw_lines_df = pd.DataFrame(raw_line_rows)
    raw_transformers_df = pd.DataFrame(raw_transformer_rows)
    raw_switches_df = pd.DataFrame(raw_switch_rows)
    raw_loads_df = pd.DataFrame(raw_load_rows)

    raw_buses_df.to_csv(paths.parsed_distribution_dir / "raw_buses.csv", index=False)
    raw_lines_df.to_csv(paths.parsed_distribution_dir / "raw_lines.csv", index=False)
    raw_transformers_df.to_csv(paths.parsed_distribution_dir / "raw_transformers.csv", index=False)
    raw_switches_df.to_csv(paths.parsed_distribution_dir / "raw_switches.csv", index=False)
    raw_loads_df.to_csv(paths.parsed_distribution_dir / "raw_loads.csv", index=False)
    write_profile_parquet(profile_rows, paths.parsed_distribution_dir / "raw_load_profiles.parquet")

    powerflow_report.update(
        {
            "service_area": service_area,
            "substation_id": substation_id,
            "profile_count": len(profile_rows),
            "raw_bus_count": len(raw_buses_df),
            "raw_line_count": len(raw_lines_df),
            "raw_transformer_count": len(raw_transformers_df),
            "raw_load_count": len(raw_loads_df),
        }
    )
    json_dump(powerflow_report, paths.parsed_distribution_dir / "opendss_powerflow_smoke.json")
    return powerflow_report


def main() -> None:
    parser = default_arg_parser(
        "Compile and parse the selected SMART-DS OpenDSS distribution model."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("05_parse_opendss_distribution", config)
    paths = get_paths(config)

    selected = pd.read_json(paths.selected_region_dir / "selected_candidate.json", typ="series")
    prototype_manifest = pd.read_csv(paths.selected_region_dir / "download_manifest.csv")
    prototype_row = prototype_manifest.loc[
        prototype_manifest["download_type"].eq("prototype")
    ].iloc[0]

    for label, substation_id in [
        ("prototype", prototype_row["substation_id"]),
        ("selected", selected["substation_id"]),
    ]:
        logger.info("Parsing %s OpenDSS case for substation %s", label, substation_id)
        report = parse_selected_distribution(config, str(substation_id), logger)
        logger.info(
            "%s compile report: converged=%s buses=%s loads=%s profiles=%s",
            label,
            report["converged"],
            report["raw_bus_count"],
            report["raw_load_count"],
            report["profile_count"],
        )


if __name__ == "__main__":
    main()

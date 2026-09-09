from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from dc_restoration.paths import repository_root

SCRIPT_DIR = Path(__file__).resolve().parent

from dc_restoration.testbed.utils.project_utils import (  # noqa: E402
    default_arg_parser,
    get_paths,
    load_config,
    setup_logging,
)
from dc_restoration.testbed.utils.smart_ds_utils import (  # noqa: E402
    infer_distribution_voltage_from_substation_id,
    root_bus_name_from_substation_id,
    service_area_from_substation_id,
)


def main() -> None:
    parser = default_arg_parser(
        "Extract transmission-to-distribution mapping tables from the Texas7k workbook."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("02_parse_td_mapping", config)
    paths = get_paths(config)

    load_map_path = paths.parsed_transmission_dir / "transmission_load_map.csv"
    if not load_map_path.exists():
        raise FileNotFoundError(f"Missing transmission load map: {load_map_path}")

    load_map = pd.read_csv(load_map_path)
    load_map["Connected To"] = load_map["Connected To"].astype(str).str.strip()
    load_map["NREL_Load_ID"] = load_map["NREL_Load_ID"].astype(str).str.strip()

    td_map = load_map.loc[load_map["Connected To"].eq("Distribution")].copy()
    td_map = td_map.rename(
        columns={
            "BusNum": "transmission_bus_id",
            "LoadID": "transmission_load_id",
            "IndividualPeakP": "individual_peak_p_mw",
            "IndividualPeakQ": "individual_peak_q_mvar",
            "PlanningP": "planning_p_mw",
            "PlanningQ": "planning_q_mvar",
            "OperationalP": "operational_p_mw",
            "OperationalQ": "operational_q_mvar",
            "PowerFactor": "power_factor",
            "NREL_Load_ID": "substation_id",
        }
    )
    td_map["service_area"] = td_map["substation_id"].map(service_area_from_substation_id)
    td_map["root_bus_name"] = td_map["substation_id"].map(root_bus_name_from_substation_id)
    td_map["distribution_voltage_kv"] = td_map["substation_id"].map(
        infer_distribution_voltage_from_substation_id
    )
    td_map["distribution_path_hint"] = td_map.apply(
        lambda row: (
            f"{config['sources']['smart_ds']['prefix']}/{row['service_area']}/"
            f"{config['sources']['smart_ds']['scenario']}/{row['substation_id']}/"
        ),
        axis=1,
    )

    td_map = td_map.sort_values(["transmission_bus_id", "substation_id", "transmission_load_id"])
    td_map_path = paths.parsed_transmission_dir / "transmission_distribution_mapping.csv"
    td_map.to_csv(td_map_path, index=False)

    summary = (
        td_map.groupby(["service_area", "substation_id"], as_index=False)
        .agg(
            transmission_bus_id=("transmission_bus_id", "first"),
            mapped_load_rows=("transmission_load_id", "count"),
            planning_p_mw=("planning_p_mw", "sum"),
            operational_p_mw=("operational_p_mw", "sum"),
        )
        .sort_values(["service_area", "substation_id"])
    )
    summary_path = paths.parsed_transmission_dir / "transmission_distribution_mapping_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info(
        "Parsed %s T-D mapping rows across %s unique substations.", len(td_map), len(summary)
    )


if __name__ == "__main__":
    main()

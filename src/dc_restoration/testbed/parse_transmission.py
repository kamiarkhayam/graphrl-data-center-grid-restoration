from __future__ import annotations

import json
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


def main() -> None:
    parser = default_arg_parser(
        "Parse the accessible Texas7k transmission-side artifacts into structured tables."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("01_parse_transmission", config)
    paths = get_paths(config)

    workbook_path = paths.raw_mapping_dir / "Texas7k_LoadGenMap.xlsx"
    if not workbook_path.exists():
        raise FileNotFoundError(f"Missing mapping workbook: {workbook_path}")

    load_map = pd.read_excel(workbook_path, sheet_name="LoadMap")
    gen_plants = pd.read_excel(workbook_path, sheet_name="GenPlants")
    gen_units = pd.read_excel(workbook_path, sheet_name="GenUnits")

    load_map.columns = [str(col).strip() for col in load_map.columns]
    gen_plants.columns = [str(col).strip() for col in gen_plants.columns]
    gen_units.columns = [str(col).strip() for col in gen_units.columns]

    load_map_path = paths.parsed_transmission_dir / "transmission_load_map.csv"
    gen_plants_path = paths.parsed_transmission_dir / "transmission_gen_plants.csv"
    gen_units_path = paths.parsed_transmission_dir / "transmission_gen_units.csv"
    load_map.to_csv(load_map_path, index=False)
    gen_plants.to_csv(gen_plants_path, index=False)
    gen_units.to_csv(gen_units_path, index=False)

    logger.info("Wrote %s load-map rows.", len(load_map))
    logger.info("Wrote %s generator-plant rows.", len(gen_plants))
    logger.info("Wrote %s generator-unit rows.", len(gen_units))

    accessible_metadata = {
        "transmission_case_archive_name": config["sources"]["transmission_archive"]["archive_name"],
        "archive_internal_powerworld_case": config["sources"]["transmission_archive"][
            "internal_powerworld_case"
        ],
        "accessible_artifacts": [
            "Texas7k_LoadGenMap.xlsx / LoadMap",
            "Texas7k_LoadGenMap.xlsx / GenPlants",
            "Texas7k_LoadGenMap.xlsx / GenUnits",
        ],
        "limitations": [
            config["notes"]["transmission_format_limitations"],
            "A parseable bus-branch model was not directly available in the downloaded BetterGrids bundle during this run.",
            "Transmission buses, generator buses, and load-delivery points will therefore be reconstructed from the official mapping workbook plus SMART-DS substation/interface coordinates where possible.",
        ],
    }
    metadata_path = paths.parsed_transmission_dir / "transmission_accessibility_report.json"
    metadata_path.write_text(json.dumps(accessible_metadata, indent=2), encoding="utf-8")
    logger.info("Wrote transmission accessibility report to %s", metadata_path)


if __name__ == "__main__":
    main()

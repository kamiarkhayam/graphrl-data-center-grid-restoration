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
from dc_restoration.testbed.utils.s3_utils import download_s3_object, list_objects  # noqa: E402
from dc_restoration.testbed.utils.smart_ds_utils import (  # noqa: E402
    list_substation_feeder_ids,
    service_area_from_substation_id,
)


def download_prefix_objects(
    base_url: str, prefix: str, destination_root: Path, logger
) -> list[Path]:
    downloaded: list[Path] = []
    for obj in list_objects(base_url, prefix):
        key = obj["key"]
        if key.endswith("/"):
            continue
        relative = key.split("SMART-DS/v0.9/2016/Full_Texas/", 1)[-1]
        local_path = destination_root / relative
        download_s3_object(base_url, key, local_path)
        downloaded.append(local_path)
    logger.info("Downloaded %s objects under %s", len(downloaded), prefix)
    return downloaded


def scenario_root_path(destination_root: Path, service_area: str, scenario: str) -> Path:
    scenario_parts = Path(scenario).parts
    return destination_root / service_area / Path(*scenario_parts)


def main() -> None:
    parser = default_arg_parser(
        "Download only the SMART-DS distribution folders required for the selected case."
    )
    parser.add_argument(
        "--prototype-substation",
        default="p1uhs0_1247",
        help="Prototype substation to download for the initial parser smoke test.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("04_download_distribution_subset", config)
    paths = get_paths(config)

    selected_path = paths.selected_region_dir / "selected_candidate.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected candidate metadata: {selected_path}")
    selected = pd.read_json(selected_path, typ="series")
    selected_substation = str(selected["substation_id"])
    service_area = service_area_from_substation_id(selected_substation)

    base_url = config["sources"]["smart_ds"]["s3_rest_base_url"]
    dataset_prefix = config["sources"]["smart_ds"]["prefix"]
    scenario = config["sources"]["smart_ds"]["scenario"]

    prototype_service_area = service_area_from_substation_id(args.prototype_substation)
    prototype_prefix = (
        f"{dataset_prefix}/{prototype_service_area}/{scenario}/{args.prototype_substation}/"
    )
    download_prefix_objects(base_url, prototype_prefix, paths.raw_distribution_dir, logger)

    prototype_root_prefix = f"{dataset_prefix}/{prototype_service_area}/{scenario}/"
    prototype_scenario_root = scenario_root_path(
        paths.raw_distribution_dir, prototype_service_area, scenario
    )
    for shared_file in ["Master.dss", "Buscoords.dss", "LoadShapes.dss"]:
        key = f"{prototype_root_prefix}{shared_file}"
        local = prototype_scenario_root / shared_file
        download_s3_object(base_url, key, local)
    logger.info("Downloaded prototype shared files for %s", prototype_service_area)

    selected_prefix = f"{dataset_prefix}/{service_area}/{scenario}/{selected_substation}/"
    selected_feeders = list_substation_feeder_ids(
        base_url, dataset_prefix, scenario, selected_substation
    )
    logger.info(
        "Selected substation %s has %s mapped feeder folders in SMART-DS.",
        selected_substation,
        len(selected_feeders),
    )
    download_prefix_objects(base_url, selected_prefix, paths.raw_distribution_dir, logger)

    selected_root_prefix = f"{dataset_prefix}/{service_area}/{scenario}/"
    selected_scenario_root = scenario_root_path(paths.raw_distribution_dir, service_area, scenario)
    for shared_file in ["Master.dss", "Buscoords.dss", "LoadShapes.dss"]:
        key = f"{selected_root_prefix}{shared_file}"
        local = selected_scenario_root / shared_file
        download_s3_object(base_url, key, local)
    logger.info("Downloaded selected shared files for %s", service_area)

    download_manifest = pd.DataFrame(
        [
            {
                "download_type": "prototype",
                "service_area": prototype_service_area,
                "substation_id": args.prototype_substation,
            },
            {
                "download_type": "selected",
                "service_area": service_area,
                "substation_id": selected_substation,
            },
        ]
    )
    manifest_path = paths.selected_region_dir / "download_manifest.csv"
    download_manifest.to_csv(manifest_path, index=False)
    logger.info("Wrote download manifest to %s", manifest_path)


if __name__ == "__main__":
    main()

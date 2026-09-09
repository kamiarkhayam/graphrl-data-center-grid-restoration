from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dc_restoration.paths import repository_root

SCRIPT_DIR = Path(__file__).resolve().parent

from dc_restoration.testbed.utils.geo_utils import (  # noqa: E402
    expand_bbox,
    min_distance_to_coast_km,
    point_in_bbox,
)
from dc_restoration.testbed.utils.project_utils import (  # noqa: E402
    default_arg_parser,
    get_paths,
    load_config,
    setup_logging,
)
from dc_restoration.testbed.utils.smart_ds_utils import (  # noqa: E402
    candidate_root_bus_names,
    fetch_line_count_from_s3,
    fetch_text_from_s3,
    list_substation_feeder_ids,
    parse_buscoords_text,
    service_area_from_substation_id,
)

BUS_SUFFIX_RE = re.compile(r"_(69|138|345)$")


def load_service_area_buscoords(
    base_url: str, dataset_prefix: str, scenario: str, service_area: str
) -> dict[str, tuple[float, float]]:
    key = f"{dataset_prefix}/{service_area}/{scenario}/Buscoords.dss"
    return parse_buscoords_text(fetch_text_from_s3(base_url, key))


def locate_substation_root(
    substation_id: str,
    buscoords: dict[str, tuple[float, float]],
) -> tuple[str | None, float | None, float | None, float | None]:
    for bus_name in candidate_root_bus_names(substation_id):
        if bus_name in buscoords:
            lat, lon = buscoords[bus_name]
            voltage = None
            match = BUS_SUFFIX_RE.search(bus_name)
            if match:
                voltage = float(match.group(1))
            return bus_name, lat, lon, voltage
    return None, None, None, None


def score_candidate(row: pd.Series, config: dict) -> float:
    preferred_min = config["selection"]["preferred_min_nodes"]
    preferred_max = config["selection"]["preferred_max_nodes"]
    hard_max = config["selection"]["hard_max_nodes"]
    score = 0.0

    if row["distance_to_coast_km"] <= 80:
        score += 3.0
    elif row["distance_to_coast_km"] <= 150:
        score += 2.0
    elif row["distance_to_coast_km"] <= 250:
        score += 1.0

    if row["voltage_kv"] in {69.0, 138.0}:
        score += 2.0

    if row["number_of_mapped_feeders"] >= 2:
        score += 2.0
    elif row["number_of_mapped_feeders"] == 1:
        score += 1.0

    if (
        row["estimated_aggregated_node_count"] >= preferred_min
        and row["estimated_aggregated_node_count"] <= preferred_max
    ):
        score += 3.0
    elif row["estimated_aggregated_node_count"] >= config["selection"]["min_total_nodes"]:
        score += 2.0
    elif row["estimated_aggregated_node_count"] >= 100:
        score += 0.5

    if row["total_peak_load_mw"] >= 80:
        score += 2.0
    elif row["total_peak_load_mw"] >= 30:
        score += 1.0

    if row["raw_distribution_node_count"] > hard_max * 3:
        score -= 2.0
    elif row["raw_distribution_node_count"] > preferred_max * 3:
        score -= 1.0

    if math.isnan(row["latitude"]) or math.isnan(row["longitude"]):
        score -= 3.0

    return round(score, 3)


def scan_service_area(
    service_area: str,
    substations: list[str],
    td_map: pd.DataFrame,
    config: dict,
) -> list[dict]:
    base_url = config["sources"]["smart_ds"]["s3_rest_base_url"]
    dataset_prefix = config["sources"]["smart_ds"]["prefix"]
    scenario = config["sources"]["smart_ds"]["scenario"]
    coast_points = [
        tuple(point) for point in config["geography"]["simplified_texas_gulf_coastline"]
    ]
    margin = config["selection"]["expanded_bbox_margin_degrees"]

    buscoords = load_service_area_buscoords(base_url, dataset_prefix, scenario, service_area)
    service_rows: list[dict] = []

    for substation_id in sorted(set(substations)):
        root_bus_name, lat, lon, voltage_kv = locate_substation_root(substation_id, buscoords)
        if lat is None or lon is None:
            continue

        region_label = None
        area_name = None
        for region in config["candidate_regions"]:
            if point_in_bbox(lat, lon, region):
                region_label = region["label"]
                area_name = region["approximate_city_area"]
                break
        if region_label is None:
            for region in config["candidate_regions"]:
                expanded = expand_bbox(region, margin)
                if point_in_bbox(lat, lon, expanded):
                    region_label = f"{region['label']}_expanded"
                    area_name = region["approximate_city_area"]
                    break
        if region_label is None:
            continue

        candidate_rows = td_map.loc[td_map["substation_id"].eq(substation_id)].copy()
        transmission_bus_id = int(candidate_rows["transmission_bus_id"].iloc[0])
        feeder_ids = list_substation_feeder_ids(base_url, dataset_prefix, scenario, substation_id)
        raw_node_count = fetch_line_count_from_s3(
            base_url,
            f"{dataset_prefix}/{service_area}/{scenario}/{substation_id}/Buscoords.dss",
        )
        estimated_aggregated_node_count = int(max(25, round(raw_node_count * 0.22)))
        total_peak_load_mw = float(candidate_rows["planning_p_mw"].sum())
        distance_to_coast_km = float(min_distance_to_coast_km(lat, lon, coast_points))
        notes = []
        if root_bus_name is None:
            notes.append("Missing explicit 69/138-kV root bus name.")
        if voltage_kv is None:
            notes.append("Root-bus voltage suffix was not parsed directly from Buscoords.")
        if estimated_aggregated_node_count < config["selection"]["min_total_nodes"]:
            notes.append("Would likely need feeder expansion or a neighboring substation.")

        row = {
            "candidate_id": f"{transmission_bus_id}_{substation_id}",
            "region_label": region_label,
            "approximate_city_area": area_name,
            "transmission_bus_id": transmission_bus_id,
            "substation_id": substation_id,
            "latitude": lat,
            "longitude": lon,
            "voltage_kv": voltage_kv,
            "number_of_mapped_feeders": len(feeder_ids),
            "raw_distribution_node_count": raw_node_count,
            "estimated_aggregated_node_count": estimated_aggregated_node_count,
            "total_peak_load_mw": total_peak_load_mw,
            "distance_to_coast_km": round(distance_to_coast_km, 3),
            "score": 0.0,
            "selected_flag": False,
            "notes": " ".join(notes),
        }
        row["score"] = score_candidate(pd.Series(row), config)
        service_rows.append(row)

    return service_rows


def main() -> None:
    parser = default_arg_parser(
        "Scan SMART-DS / Texas7k mappings for Gulf-coast candidate regions."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("03_scan_gulf_candidates", config)
    paths = get_paths(config)

    mapping_path = paths.parsed_transmission_dir / "transmission_distribution_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping table: {mapping_path}")
    td_map = pd.read_csv(mapping_path)

    service_area_substations: dict[str, list[str]] = defaultdict(list)
    for substation_id in td_map["substation_id"].dropna().astype(str).unique():
        service_area_substations[service_area_from_substation_id(substation_id)].append(
            substation_id
        )

    rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        future_map = {}
        for service_area, substations in sorted(service_area_substations.items()):
            logger.info(
                "Queueing service area %s (%s mapped substations)", service_area, len(substations)
            )
            future = executor.submit(scan_service_area, service_area, substations, td_map, config)
            future_map[future] = service_area

        for future in as_completed(future_map):
            service_area = future_map[future]
            try:
                service_rows = future.result()
                rows.extend(service_rows)
                logger.info(
                    "Completed service area %s with %s Gulf candidate substations.",
                    service_area,
                    len(service_rows),
                )
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning(
                    "Skipping service area %s due to scan failure: %s", service_area, exc
                )

    if not rows:
        raise RuntimeError(
            "No Gulf candidate substations were found in the current SMART-DS / mapping scan."
        )

    candidates = pd.DataFrame(rows).sort_values(
        ["score", "estimated_aggregated_node_count", "total_peak_load_mw"],
        ascending=[False, False, False],
    )
    best_idx = candidates.index[0]
    candidates.loc[best_idx, "selected_flag"] = True

    report_path = paths.final_case_dir / "candidate_region_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(report_path, index=False)
    logger.info("Wrote candidate report with %s rows to %s", len(candidates), report_path)

    selected_row = candidates.loc[best_idx].to_dict()
    selected_path = paths.selected_region_dir / "selected_candidate.json"
    pd.Series(selected_row).to_json(selected_path, indent=2)
    logger.info("Selected candidate: %s", selected_row["candidate_id"])

    substation_catalog = candidates[
        [
            "candidate_id",
            "substation_id",
            "region_label",
            "approximate_city_area",
            "latitude",
            "longitude",
            "voltage_kv",
            "distance_to_coast_km",
            "score",
        ]
    ].copy()
    substation_catalog.to_csv(paths.candidate_regions_dir / "substation_catalog.csv", index=False)

    plt.switch_backend("Agg")
    plt.figure(figsize=(11, 8))
    for region_name, region_df in candidates.groupby("approximate_city_area"):
        plt.scatter(
            region_df["longitude"],
            region_df["latitude"],
            s=40 + 1.5 * region_df["raw_distribution_node_count"].clip(upper=400),
            alpha=0.7,
            label=region_name,
        )
    selected = candidates.loc[candidates["selected_flag"]]
    plt.scatter(
        selected["longitude"],
        selected["latitude"],
        s=250,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Selected",
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("SMART-DS Gulf Candidate Regions")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.25)
    plot_path = paths.plots_dir / "candidate_regions_map.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info("Saved candidate map to %s", plot_path)


if __name__ == "__main__":
    main()

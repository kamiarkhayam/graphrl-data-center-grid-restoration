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
        "Place realistic substation/transmission-connected data center nodes."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("08_place_data_centers", config)
    paths = get_paths(config)

    selected = pd.read_json(paths.selected_region_dir / "selected_candidate.json", typ="series")
    metadata = json.loads(
        (paths.selected_region_dir / "selected_subnetwork_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    local_peak_mw = float(selected["total_peak_load_mw"])
    defaults = config["data_center_defaults"]
    facility_load_mw = min(
        defaults["max_facility_load_mw"],
        max(defaults["min_facility_load_mw"], round(local_peak_mw * 0.31, 1)),
    )
    facility_load_mw = min(
        facility_load_mw, round(local_peak_mw * defaults["max_load_fraction_of_local_peak"], 1)
    )
    if facility_load_mw < defaults["min_facility_load_mw"]:
        facility_load_mw = defaults["min_facility_load_mw"]

    pue = float(defaults["pue"])
    it_load_mw = round(facility_load_mw / pue, 3)
    critical_it_fraction = float(defaults["critical_it_fraction"])
    flexible_it_fraction = float(defaults["flexible_it_fraction"])
    critical_load_mw = round(it_load_mw * critical_it_fraction, 3)
    flexible_load_mw = round(it_load_mw * flexible_it_fraction, 3)
    cooling_load_mw = round(facility_load_mw - it_load_mw, 3)
    ups_power_mw = round(critical_load_mw, 3)
    ups_energy_mwh = round(ups_power_mw * (defaults["ups_duration_minutes"] / 60.0), 3)
    backup_gen_mw = round(critical_load_mw * 1.2, 3)
    max_export_mw = round(max(0.0, backup_gen_mw - critical_load_mw - 2.0), 3)

    data_center = {
        "dc_id": "DC_1",
        "name": defaults["name"],
        "latitude": round(float(selected["latitude"]) - 0.0085, 6),
        "longitude": round(float(selected["longitude"]) + 0.018, 6),
        "connected_bus_id": metadata["selected_transmission_node_id"],
        "connection_voltage_kv": float(selected["voltage_kv"]),
        "facility_load_mw": facility_load_mw,
        "critical_it_fraction": critical_it_fraction,
        "flexible_it_fraction": flexible_it_fraction,
        "pue": pue,
        "critical_load_mw": critical_load_mw,
        "flexible_load_mw": flexible_load_mw,
        "cooling_load_mw": cooling_load_mw,
        "ups_power_mw": ups_power_mw,
        "ups_energy_mwh": ups_energy_mwh,
        "backup_gen_mw": backup_gen_mw,
        "fuel_hours_at_critical_load": 48,
        "minimum_survival_hours": defaults["minimum_survival_hours"],
        "max_export_mw_anchor_mode": max_export_mw,
        "can_export_in_baseline": False,
        "can_export_in_anchor_mode": True,
        "notes": (
            "Synthetic Houston-area large commercial/data-center facility connected at the selected 69-kV delivery point. "
            "Anchor-mode export capability is stored only as a future scenario attribute."
        ),
    }

    out_csv = paths.final_case_dir / "data_centers.csv"
    pd.DataFrame([data_center]).to_csv(out_csv, index=False)
    out_json = paths.final_case_dir / "dc_specs.json"
    out_json.write_text(json.dumps({"data_centers": [data_center]}, indent=2), encoding="utf-8")
    logger.info("Wrote data center placement to %s and %s", out_csv, out_json)


if __name__ == "__main__":
    main()

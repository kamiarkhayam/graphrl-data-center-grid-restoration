from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from dc_restoration.paths import repository_root

SCRIPT_DIR = Path(__file__).resolve().parent

from dc_restoration.testbed.utils.geo_utils import haversine_km  # noqa: E402
from dc_restoration.testbed.utils.project_utils import (  # noqa: E402
    default_arg_parser,
    get_paths,
    load_config,
    setup_logging,
)
from dc_restoration.testbed.utils.smart_ds_utils import (  # noqa: E402
    parse_buscoords_text,
    root_bus_name_from_substation_id,
    service_area_from_substation_id,
)


def main() -> None:
    parser = default_arg_parser(
        "Build selected-region transmission/interface metadata from parsed artifacts."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging("06_extract_subnetwork", config)
    paths = get_paths(config)

    selected = pd.read_json(paths.selected_region_dir / "selected_candidate.json", typ="series")
    candidate_report = pd.read_csv(paths.final_case_dir / "candidate_region_report.csv")

    substation_id = str(selected["substation_id"])
    service_area = service_area_from_substation_id(substation_id)
    region_label = selected["region_label"]
    selected_bus_id = int(selected["transmission_bus_id"])
    selected_root_bus = (
        f"{root_bus_name_from_substation_id(substation_id)}_{int(selected['voltage_kv'])}"
    )

    scenario_root = (
        paths.raw_distribution_dir
        / service_area
        / Path(*Path(config["sources"]["smart_ds"]["scenario"]).parts)
    )
    top_buscoords = parse_buscoords_text(
        (scenario_root / "Buscoords.dss").read_text(encoding="utf-8")
    )
    if "st_mat" not in top_buscoords:
        raise KeyError(
            "Expected service-area source bus 'st_mat' not found in top-level Buscoords.dss"
        )

    selected_lat = float(selected["latitude"])
    selected_lon = float(selected["longitude"])

    peers = candidate_report.loc[
        candidate_report["region_label"].eq(region_label)
        & candidate_report["candidate_id"].ne(selected["candidate_id"])
    ].copy()
    peers["distance_km"] = peers.apply(
        lambda row: haversine_km(
            selected_lat, selected_lon, float(row["latitude"]), float(row["longitude"])
        ),
        axis=1,
    )
    peers = peers.sort_values(["distance_km", "score"], ascending=[True, False]).head(2)

    peer_nodes = []
    for _, row in peers.iterrows():
        peer_nodes.append(
            {
                "node_id": f"T_{int(row['transmission_bus_id'])}",
                "transmission_bus_id": int(row["transmission_bus_id"]),
                "substation_id": row["substation_id"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "voltage_kv": float(row["voltage_kv"]),
                "distance_to_selected_km": float(row["distance_km"]),
            }
        )

    metadata = {
        "selected_candidate_id": selected["candidate_id"],
        "service_area": service_area,
        "region_label": region_label,
        "transmission_source_bus_id": "st_mat",
        "transmission_source_voltage_kv": 230.0,
        "transmission_source_latitude": top_buscoords["st_mat"][0],
        "transmission_source_longitude": top_buscoords["st_mat"][1],
        "selected_transmission_bus_id": selected_bus_id,
        "selected_transmission_node_id": f"T_{selected_bus_id}",
        "selected_substation_id": substation_id,
        "selected_substation_root_bus": selected_root_bus,
        "selected_voltage_kv": float(selected["voltage_kv"]),
        "selected_latitude": selected_lat,
        "selected_longitude": selected_lon,
        "substation_interface_node_id": f"SUB_{substation_id}",
        "peer_transmission_nodes": peer_nodes,
        "assumption": (
            "Texas7k transmission bus-branch topology was not directly parseable from the accessible "
            "BetterGrids archive, so the local transmission/interface layer is represented as an "
            "equivalent service-area source bus plus nearby mapped 69-kV delivery points."
        ),
    }

    out_path = paths.selected_region_dir / "selected_subnetwork_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote selected subnetwork metadata to %s", out_path)

    tx_nodes = pd.DataFrame(
        [
            {
                "node_id": "T_SOURCE_STMAT",
                "transmission_bus_id": "st_mat",
                "node_type": "source",
                "latitude": metadata["transmission_source_latitude"],
                "longitude": metadata["transmission_source_longitude"],
                "voltage_kv": 230.0,
            },
            {
                "node_id": f"T_{selected_bus_id}",
                "transmission_bus_id": selected_bus_id,
                "node_type": "delivery_bus",
                "latitude": selected_lat,
                "longitude": selected_lon,
                "voltage_kv": float(selected["voltage_kv"]),
            },
            *[
                {
                    "node_id": peer["node_id"],
                    "transmission_bus_id": peer["transmission_bus_id"],
                    "node_type": "peer_delivery_bus",
                    "latitude": peer["latitude"],
                    "longitude": peer["longitude"],
                    "voltage_kv": peer["voltage_kv"],
                }
                for peer in peer_nodes
            ],
        ]
    )
    tx_edges = pd.DataFrame(
        [
            {
                "edge_id": f"TE_SOURCE_{selected_bus_id}",
                "from_node": "T_SOURCE_STMAT",
                "to_node": f"T_{selected_bus_id}",
                "edge_type": "equivalent_subtransmission",
            },
            *[
                {
                    "edge_id": f"TE_SOURCE_{peer['transmission_bus_id']}",
                    "from_node": "T_SOURCE_STMAT",
                    "to_node": peer["node_id"],
                    "edge_type": "equivalent_subtransmission",
                }
                for peer in peer_nodes
            ],
            {
                "edge_id": f"XFMR_T_{selected_bus_id}_SUB_{substation_id}",
                "from_node": f"T_{selected_bus_id}",
                "to_node": f"SUB_{substation_id}",
                "edge_type": "interface_transformer_equivalent",
            },
        ]
    )
    tx_nodes.to_csv(paths.selected_region_dir / "transmission_interface_nodes.csv", index=False)
    tx_edges.to_csv(paths.selected_region_dir / "transmission_interface_edges.csv", index=False)
    logger.info("Saved %s transmission/interface nodes and %s edges.", len(tx_nodes), len(tx_edges))


if __name__ == "__main__":
    main()

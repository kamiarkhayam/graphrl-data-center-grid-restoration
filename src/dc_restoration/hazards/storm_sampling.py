from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from shapely.geometry import Point, Polygon

from .geo_utils import approximate_distance_scalar_km
from .hurricane_data import extract_storm_track


def load_mainland_polygon(csv_path: str | Path) -> Polygon:
    coords = pd.read_csv(csv_path)
    polygon = Polygon(zip(coords["Longitude"], coords["Latitude"]))
    if not polygon.is_valid:
        raise ValueError(f"Invalid mainland polygon loaded from {csv_path}")
    return polygon


def find_first_landfall(
    track_lonlat: list[tuple[float, float]], polygon: Polygon
) -> tuple[int | None, tuple[float, float] | None]:
    for idx, (lon, lat) in enumerate(track_lonlat):
        if polygon.contains(Point(lon, lat)) and lon <= -85:
            return idx, (lon, lat)
    return None, None


def intensity_decay_with_distance(
    initial_intensity_mps: float, distance_km: float, kd: float = 0.003
) -> float:
    return float(initial_intensity_mps * np.exp(-kd * distance_km))


def compute_rmax_atlantic(vmax_mps: float) -> float:
    vmax_knots = vmax_mps * 1.94384
    return float(46.3 * np.exp(-0.0153 * vmax_knots))


def extend_track_if_needed(
    track_lonlat: list[tuple[float, float]],
    start_index: int,
    min_speed_mps: float = 20.0,
    step_km: float = 15.0,
    kd: float = 0.003,
    intensity_kts: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Port of the old `extend_track_if_needed` logic.

    If `intensity_kts` is supplied, it overrides the sampled gamma landfall
    intensity exactly like the legacy code path.
    """
    rng = rng or np.random.default_rng()
    extended_track = list(track_lonlat[start_index:])
    distances = [0.0]
    intensities: list[float] = []

    gamma_params = (2.8443029489547413, 22.07726110971529, 16.20587569107012)
    landfall_knots = float(stats.gamma.rvs(*gamma_params, random_state=rng))
    landfall_mps = landfall_knots * 0.514444

    intensity_change = (float(rng.lognormal(1.6564, 0.5396)) - 2.5) / 100.0 + 1.0
    landfall_mps *= intensity_change * 2.0

    if intensity_kts is not None and intensity_kts > 0:
        landfall_mps = float(intensity_kts) * 0.514444

    for idx in range(start_index + 1, len(track_lonlat)):
        d_step = approximate_distance_scalar_km(track_lonlat[idx - 1], track_lonlat[idx])
        distances.append(distances[-1] + d_step)

    for distance in distances:
        intensities.append(intensity_decay_with_distance(landfall_mps, distance, kd=kd))

    while intensities[-1] > min_speed_mps:
        if len(extended_track) > 1:
            p1, p2 = extended_track[-2], extended_track[-1]
        else:
            p1 = p2 = extended_track[-1]
        heading_dx = p2[0] - p1[0]
        heading_dy = p2[1] - p1[1]
        heading_dist = approximate_distance_scalar_km(p1, p2)
        if heading_dist < 1e-9:
            heading_dist = 0.0001
        scale = step_km / heading_dist
        new_lon = p2[0] + heading_dx * scale
        new_lat = p2[1] + heading_dy * scale
        extended_track.append((float(new_lon), float(new_lat)))
        distances.append(distances[-1] + step_km)
        intensities.append(intensity_decay_with_distance(landfall_mps, distances[-1], kd=kd))

    return extended_track, intensities


def build_sampled_storm_tracks(
    catalog_df: pd.DataFrame,
    sampled_storms_df: pd.DataFrame,
    mainland_polygon: Polygon,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(config["random_seed"]))
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    min_speed_mps = float(config["storm_model"]["min_speed_mps"])
    step_km = float(config["storm_model"]["extension_step_km"])
    kd = float(config["storm_model"]["intensity_decay_kd"])

    for _, sample in sampled_storms_df.iterrows():
        storm_id = sample["storm_id"]
        track_df = extract_storm_track(catalog_df, storm_id)
        track_lonlat = list(zip(track_df["longitude"], track_df["latitude"]))
        landfall_mask = track_df["is_landfall"].fillna(False).astype(bool)
        if landfall_mask.any():
            start_index = int(np.flatnonzero(landfall_mask.to_numpy())[0])
            landfall_pt = track_lonlat[start_index]
            start_method = "hurdat_landfall_flag"
            observed_landfall_wind_kts = float(track_df.loc[start_index, "max_wind"])
        else:
            start_index, landfall_pt = find_first_landfall(track_lonlat, mainland_polygon)
            if start_index is None:
                continue
            start_method = "polygon_landfall_fallback"
            observed_landfall_wind_kts = float(track_df.loc[start_index, "max_wind"])

        extended_track, intensities = extend_track_if_needed(
            track_lonlat=track_lonlat,
            start_index=start_index,
            min_speed_mps=min_speed_mps,
            step_km=step_km,
            kd=kd,
            intensity_kts=observed_landfall_wind_kts if observed_landfall_wind_kts > 0 else None,
            rng=rng,
        )
        original_points_after_start = len(track_lonlat) - start_index
        rmax_list = [compute_rmax_atlantic(value) for value in intensities]
        scenario_id = f"S{int(sample['selection_order']):03d}_{storm_id}"

        manifest_rows.append(
            {
                "scenario_id": scenario_id,
                "storm_id": storm_id,
                "storm_name": sample["storm_name"],
                "year": int(sample["year"]),
                "selection_order": int(sample["selection_order"]),
                "sampling_method": sample["sampling_method"],
                "sampling_seed": int(sample["sampling_seed"]),
                "landfall_distance_to_network_km": float(sample["landfall_distance_to_network_km"])
                if "landfall_distance_to_network_km" in sample
                else pd.NA,
                "track_source": "historical_filtered_hurdat2",
                "start_index": int(start_index),
                "start_method": start_method,
                "landfall_longitude": float(landfall_pt[0]),
                "landfall_latitude": float(landfall_pt[1]),
                "used_landfall_wind_kts": float(observed_landfall_wind_kts),
                "intensity_source": "observed_landfall_hurdat",
                "original_point_count": int(len(track_df)),
                "simulated_point_count": int(len(extended_track)),
            }
        )

        for point_index, ((lon, lat), intensity_mps, rmax_km) in enumerate(
            zip(extended_track, intensities, rmax_list)
        ):
            if point_index < original_points_after_start:
                original_row = track_df.iloc[start_index + point_index]
                phase = "historical_after_landfall"
                source_time_index = int(original_row["time_index"])
                source_timestamp_text = original_row["timestamp_text"]
                source_wind_kts = float(original_row["max_wind"])
            else:
                phase = "synthetic_inland_extension"
                source_time_index = pd.NA
                source_timestamp_text = pd.NA
                source_wind_kts = pd.NA
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "storm_id": storm_id,
                    "storm_name": sample["storm_name"],
                    "year": int(sample["year"]),
                    "selection_order": int(sample["selection_order"]),
                    "track_point_index": point_index,
                    "phase": phase,
                    "longitude": float(lon),
                    "latitude": float(lat),
                    "intensity_mps": float(intensity_mps),
                    "intensity_kts_equivalent": float(intensity_mps / 0.514444),
                    "rmax_km": float(rmax_km),
                    "source_time_index": source_time_index,
                    "source_timestamp_text": source_timestamp_text,
                    "source_wind_kts": source_wind_kts,
                }
            )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("selection_order").reset_index(drop=True)
    track_points_df = pd.DataFrame(rows)
    if not track_points_df.empty:
        track_points_df = track_points_df.sort_values(
            ["selection_order", "track_point_index"]
        ).reset_index(drop=True)
    return manifest_df, track_points_df

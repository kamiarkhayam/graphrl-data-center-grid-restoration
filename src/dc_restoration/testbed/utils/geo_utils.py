from __future__ import annotations

import math
from typing import Iterable

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def point_in_bbox(lat: float, lon: float, bbox: dict[str, float]) -> bool:
    return bbox["lat_min"] <= lat <= bbox["lat_max"] and bbox["lon_min"] <= lon <= bbox["lon_max"]


def expand_bbox(bbox: dict[str, float], margin: float) -> dict[str, float]:
    return {
        "lat_min": bbox["lat_min"] - margin,
        "lat_max": bbox["lat_max"] + margin,
        "lon_min": bbox["lon_min"] - margin,
        "lon_max": bbox["lon_max"] + margin,
    }


def min_distance_to_coast_km(
    lat: float,
    lon: float,
    coastline_points: Iterable[tuple[float, float]],
) -> float:
    return min(
        haversine_km(lat, lon, coast_lat, coast_lon) for coast_lat, coast_lon in coastline_points
    )

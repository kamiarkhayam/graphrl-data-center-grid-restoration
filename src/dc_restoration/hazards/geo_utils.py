from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def approximate_distance_km(
    p1: Sequence[float] | np.ndarray,
    p2: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Approximate geodesic distance in km using an equirectangular projection."""
    a1 = np.atleast_2d(np.asarray(p1, dtype=float))
    a2 = np.atleast_2d(np.asarray(p2, dtype=float))
    lon1 = a1[:, 0][:, None]
    lat1 = a1[:, 1][:, None]
    lon2 = a2[:, 0][None, :]
    lat2 = a2[:, 1][None, :]
    lat_mid = np.radians((lat1 + lat2) / 2.0)
    km_per_degree_lon = 111.321 * np.cos(lat_mid)
    km_per_degree_lat = 111.0
    d_lon = (lon2 - lon1) * km_per_degree_lon
    d_lat = (lat2 - lat1) * km_per_degree_lat
    return np.sqrt(d_lon**2 + d_lat**2)


def approximate_distance_scalar_km(
    p1: Sequence[float] | np.ndarray,
    p2: Sequence[float] | np.ndarray,
) -> float:
    return float(approximate_distance_km(p1, p2)[0, 0])


def midpoint_lonlat(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
) -> tuple[float, float]:
    return ((start_lon + end_lon) / 2.0, (start_lat + end_lat) / 2.0)


def centroid_from_frame(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> tuple[float, float]:
    clean = df[[lat_col, lon_col]].dropna()
    return (float(clean[lat_col].mean()), float(clean[lon_col].mean()))


def bounding_box_from_frame(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> dict[str, float]:
    clean = df[[lat_col, lon_col]].dropna()
    return {
        "min_lat": float(clean[lat_col].min()),
        "max_lat": float(clean[lat_col].max()),
        "min_lon": float(clean[lon_col].min()),
        "max_lon": float(clean[lon_col].max()),
    }


def rowwise_distances_km(
    left_lon: Iterable[float],
    left_lat: Iterable[float],
    right_lon: Iterable[float],
    right_lat: Iterable[float],
) -> np.ndarray:
    p1 = np.column_stack(
        [np.asarray(list(left_lon), dtype=float), np.asarray(list(left_lat), dtype=float)]
    )
    p2 = np.column_stack(
        [np.asarray(list(right_lon), dtype=float), np.asarray(list(right_lat), dtype=float)]
    )
    return approximate_distance_km(p1, p2).diagonal()

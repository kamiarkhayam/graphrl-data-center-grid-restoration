from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def saffir_simpson_category_from_kts(wind_kts: float) -> str:
    if pd.isna(wind_kts):
        return "unknown"
    if wind_kts < 34:
        return "tropical_depression"
    if wind_kts < 64:
        return "tropical_storm"
    if wind_kts < 83:
        return "category_1"
    if wind_kts < 96:
        return "category_2"
    if wind_kts < 113:
        return "category_3"
    if wind_kts < 137:
        return "category_4"
    return "category_5"


def load_filtered_hurdat_catalog(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    df = pd.read_csv(path)
    df["storm_id"] = df["storm_id"].astype(str)
    df["storm_name"] = df["storm_name"].astype(str)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["time"] = df["time"].astype(str).str.zfill(4)
    df["year"] = df["date"].str[:4].astype(int)
    df["time_index"] = df.groupby("storm_id").cumcount()
    df["timestamp_text"] = df["date"] + df["time"]
    df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["lon"], errors="coerce")
    df["max_wind"] = pd.to_numeric(df["wind_kts"], errors="coerce")
    df["pressure"] = pd.to_numeric(df["pressure_mb"], errors="coerce")
    df["category"] = df["max_wind"].map(saffir_simpson_category_from_kts)
    df["landfall_or_crossing_flag"] = df["is_landfall"].fillna(False).astype(bool)
    df["source_file"] = str(path)
    return df.sort_values(["storm_id", "date", "time"]).reset_index(drop=True)


def storm_level_summary(catalog_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        catalog_df.groupby("storm_id")
        .agg(
            storm_name=("storm_name", "first"),
            year=("year", "first"),
            point_count=("storm_id", "size"),
            has_landfall=("is_landfall", "max"),
            max_wind_kts=("max_wind", "max"),
            min_lat=("latitude", "min"),
            max_lat=("latitude", "max"),
            min_lon=("longitude", "min"),
            max_lon=("longitude", "max"),
        )
        .reset_index()
    )
    return summary


def sample_storms(
    storm_summary_df: pd.DataFrame,
    n_storms: int,
    seed: int,
    landfall_only: bool = True,
) -> pd.DataFrame:
    candidates = storm_summary_df.copy()
    if landfall_only:
        candidates = candidates.loc[candidates["has_landfall"].astype(bool)].copy()
    if candidates.empty:
        raise ValueError("No candidate storms available after filtering.")
    rng = np.random.default_rng(seed)
    n_take = min(n_storms, len(candidates))
    chosen = rng.choice(candidates["storm_id"].to_numpy(), size=n_take, replace=False)
    sampled = candidates.set_index("storm_id").loc[chosen].reset_index()
    sampled["selection_order"] = range(1, len(sampled) + 1)
    sampled["sampling_method"] = "uniform_without_replacement_historical_catalog"
    sampled["sampling_seed"] = seed
    return sampled


def extract_storm_track(catalog_df: pd.DataFrame, storm_id: str) -> pd.DataFrame:
    return (
        catalog_df.loc[catalog_df["storm_id"] == storm_id]
        .sort_values(["date", "time", "time_index"])
        .reset_index(drop=True)
    )

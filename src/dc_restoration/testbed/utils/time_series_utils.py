from __future__ import annotations

from datetime import datetime

import pandas as pd


def build_non_leap_timestamps(year: int, periods: int, freq_minutes: int = 15) -> pd.DatetimeIndex:
    freq = f"{freq_minutes}min"
    full = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:45:00",
        freq=freq,
    )
    no_leap = full[~((full.month == 2) & (full.day == 29))]
    if len(no_leap) == periods:
        return no_leap
    return pd.date_range(start=f"{year}-01-01 00:00:00", periods=periods, freq=freq)


def representative_day_mask(index: pd.DatetimeIndex, month: int, day: int) -> pd.Series:
    return (index.month == month) & (index.day == day)

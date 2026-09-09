from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dc_restoration.paths import repository_root

GROUPS = [
    "residential_noncritical",
    "residential_critical",
    "commercial_noncritical",
    "commercial_critical",
]


def _inflate(value: float, source_year: int, config: dict[str, Any]) -> float:
    inflation = config["inflation"]
    source = float(inflation[f"cpi_u_{source_year}"])
    target = float(inflation[f"cpi_u_{config['price_year']}"])
    return value * target / source


def _loglog_multiplier(
    duration_hours: Any, points_hours: list[float], multipliers: list[float]
) -> Any:
    duration = np.asarray(duration_hours, dtype=float)
    points = np.asarray(points_hours, dtype=float)
    values = np.asarray(multipliers, dtype=float)
    output = np.ones_like(duration)
    positive = duration > points[0]
    if positive.any():
        x = np.log(duration[positive])
        xp = np.log(points)
        yp = np.log(values)
        interpolated = np.interp(x, xp, yp)
        above = x > xp[-1]
        if above.any():
            slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
            interpolated[above] = yp[-1] + slope * (x[above] - xp[-1])
        output[positive] = np.exp(interpolated)
    return float(output) if output.ndim == 0 else output


def _cdf_per_interrupted_kw(
    duration_hours: Any,
    value_24h_usd_per_kwh: float,
    duration_points_hours: list[float],
    relative_total_damage: list[float],
) -> Any:
    duration = np.asarray(duration_hours, dtype=float)
    c24 = value_24h_usd_per_kwh * 24.0
    multiplier = _loglog_multiplier(duration, duration_points_hours, relative_total_damage)
    result = np.where(
        duration <= 24.0, value_24h_usd_per_kwh * np.maximum(duration, 0.0), c24 * multiplier
    )
    return float(result) if result.ndim == 0 else result


def _valuation_parameters(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    values = config["duration_dependent_valuation"]
    target_year = int(config["price_year"])
    points = [float(value) for value in values["duration_points_hours"]]

    def inflated_range(name: str, source_year: int) -> dict[str, float]:
        return {
            label: _inflate(float(value), source_year, config)
            for label, value in values[name].items()
        }

    rnc = inflated_range(
        "residential_noncritical_24h_usd_per_kwh",
        int(values["residential_24h_source_price_year"]),
    )
    rc24 = inflated_range(
        "residential_critical_24h_usd_per_kwh",
        int(values["residential_24h_source_price_year"]),
    )
    rc240 = inflated_range(
        "residential_critical_10day_usd_per_kwh",
        int(values["residential_critical_10day_source_price_year"]),
    )
    commercial24 = _inflate(
        float(values["commercial_24h_usd_per_kwh"]),
        int(values["commercial_24h_source_price_year"]),
        config,
    )

    residential_damage = np.asarray(
        values["residential_median_customer_damage_dollars"], dtype=float
    )
    smnr_damage = np.asarray(values["commercial_smnr_median_customer_damage_dollars"], dtype=float)
    lnrp_damage = np.asarray(values["commercial_lnrp_median_customer_damage_dollars"], dtype=float)
    residential_shape = (residential_damage / residential_damage[0]).tolist()
    smnr_shape = (smnr_damage / smnr_damage[0]).tolist()
    lnrp_shape = (lnrp_damage / lnrp_damage[0]).tolist()

    parameters: dict[str, dict[str, Any]] = {}
    for case in ["source_low", "primary", "source_high"]:
        range_key = {"source_low": "low", "primary": "central", "source_high": "high"}[case]
        critical_240_multiplier = (rc240[range_key] * 240.0) / (rc24[range_key] * 24.0)
        residential_at_240 = float(_loglog_multiplier(240.0, points, residential_shape))
        critical_points = [24.0, 240.0, 336.0, 720.0]
        critical_shape = [
            1.0,
            critical_240_multiplier,
            critical_240_multiplier * residential_shape[1] / residential_at_240,
            critical_240_multiplier * residential_shape[2] / residential_at_240,
        ]
        parameters[case] = {
            "residential_noncritical_24h": rnc[range_key],
            "residential_critical_24h": rc24[range_key],
            "residential_noncritical_points": points,
            "residential_noncritical_shape": residential_shape,
            "residential_critical_points": critical_points,
            "residential_critical_shape": critical_shape,
            "commercial_24h": commercial24,
            "commercial_points": points,
            "commercial_shape": lnrp_shape if case == "source_high" else smnr_shape,
        }

    rows = [
        {
            "parameter": "Residential noncritical 24-hour value",
            "source_value": str(values["residential_noncritical_24h_usd_per_kwh"]),
            "source_price_year": values["residential_24h_source_price_year"],
            "analysis_value_2025": str(rnc),
            "unit": "USD per unserved kWh",
            "source_url": values["residential_24h_source_url"],
            "use": "All modeled noncritical residential service",
        },
        {
            "parameter": "Residential critical 24-hour value",
            "source_value": str(values["residential_critical_24h_usd_per_kwh"]),
            "source_price_year": values["residential_24h_source_price_year"],
            "analysis_value_2025": str(rc24),
            "unit": "USD per unserved kWh",
            "source_url": values["residential_24h_source_url"],
            "use": "Modeled residential loads carrying the critical flag",
        },
        {
            "parameter": "Residential critical ten-day value",
            "source_value": str(values["residential_critical_10day_usd_per_kwh"]),
            "source_price_year": values["residential_critical_10day_source_price_year"],
            "analysis_value_2025": str(rc240),
            "unit": "USD per unserved kWh",
            "source_url": values["residential_critical_10day_source_url"],
            "use": "Ten-day anchor for limited critical residential service",
        },
        {
            "parameter": "Commercial 24-hour value",
            "source_value": commercial24,
            "source_price_year": target_year,
            "analysis_value_2025": commercial24,
            "unit": "USD per unserved kWh",
            "source_url": values["commercial_24h_source_url"],
            "use": "All modeled commercial service; critical commercial is reported separately without a premium",
        },
        {
            "parameter": "Long-duration normalized shapes",
            "source_value": f"Residential={residential_damage.tolist()}; SMNR={smnr_damage.tolist()}; LNRP={lnrp_damage.tolist()}",
            "source_price_year": values["long_duration_source_price_year"],
            "analysis_value_2025": f"Residential={residential_shape}; SMNR={smnr_shape}; LNRP={lnrp_shape}",
            "unit": "Relative total customer damage at 1, 14, and 30 days",
            "source_url": values["long_duration_source_url"],
            "use": "Duration shape only; raw Puerto Rico dollars are not transferred",
        },
    ]
    return parameters, pd.DataFrame(rows)

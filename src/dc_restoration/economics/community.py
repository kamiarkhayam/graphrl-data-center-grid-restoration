from __future__ import annotations

from typing import Any

import pandas as pd

from dc_restoration.paths import repository_root


def _capital_recovery_factor(rate: float, years: int) -> float:
    if rate == 0:
        return 1.0 / years
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def _annual_rate_path(
    config: dict[str, Any], loads: pd.DataFrame, total_facility_mw: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rate = config["ratepayer_effect"]
    analysis = config["analysis"]
    residential_mw = float(
        loads.loc[loads["load_type"].astype(str).str.lower() == "residential", "average_p_kw"]
        .astype(float)
        .sum()
        / 1000.0
    )
    residential_kwh = residential_mw * 1000.0 * 8760.0
    attribution = total_facility_mw / float(rate["portfolio_attribution_denominator_mw"])
    inflation = float(rate["cpi_u_2025"]) / float(rate["cpi_u_2024"])
    life = int(analysis["project_life_years"])
    discount = float(analysis["real_discount_rate"])
    crf = _capital_recovery_factor(discount, life)
    start_year = int(analysis["analysis_dollar_year"])
    path_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for case_key, case in rate["cases"].items():
        pv = 0.0
        for analysis_year in range(1, life + 1):
            calendar_year = start_year + analysis_year
            v2030 = float(case["monthly_2030_dollars_per_1000_kwh"])
            v2040 = float(case["monthly_2040_dollars_per_1000_kwh"])
            if calendar_year <= 2030:
                monthly_2024 = v2030 * max(0.0, (calendar_year - start_year) / (2030 - start_year))
            elif calendar_year <= 2040:
                monthly_2024 = v2030 + (v2040 - v2030) * (calendar_year - 2030) / 10.0
            else:
                monthly_2024 = v2040
            monthly_2025 = monthly_2024 * inflation
            rate_adder = monthly_2025 / 1000.0
            full_burden = residential_kwh * rate_adder
            attributed_burden = full_burden * attribution
            discounted = attributed_burden / (1.0 + discount) ** analysis_year
            pv += discounted
            path_rows.append(
                {
                    "rate_case": case_key,
                    "rate_case_display": case["display_name"],
                    "analysis_year": analysis_year,
                    "calendar_year": calendar_year,
                    "monthly_bill_adder_2025_dollars_per_1000_kwh": monthly_2025,
                    "rate_adder_2025_dollars_per_kwh": rate_adder,
                    "modeled_residential_average_load_mw": residential_mw,
                    "modeled_residential_annual_energy_kwh": residential_kwh,
                    "full_context_annual_residential_rate_burden_dollars": full_burden,
                    "v0_3_portfolio_attribution_share": attribution,
                    "attributed_annual_residential_rate_burden_dollars": attributed_burden,
                    "discounted_attributed_burden_dollars": discounted,
                }
            )
        summary_rows.append(
            {
                "rate_case": case_key,
                "rate_case_display": case["display_name"],
                "present_value_attributed_rate_burden_dollars": pv,
                "equivalent_annual_attributed_rate_burden_dollars": pv * crf,
                "portfolio_attribution_share": attribution,
                "attribution_note": config["documentation"]["tariff_warning"],
            }
        )
    return pd.DataFrame(path_rows), pd.DataFrame(summary_rows)


def _fiscal_summary(config: dict[str, Any], site_count: int) -> pd.DataFrame:
    fiscal = config["fiscal_effect"]
    block = float(fiscal["taxable_equipment_block_per_facility_dollars"])
    refresh = float(fiscal["equipment_refresh_years"])
    annual_local_sales_exemption = (
        block * site_count * float(fiscal["local_sales_tax_share"]) / refresh
    )
    rows = []
    for case_key, case in fiscal["cases"].items():
        annual_property = (
            float(case["five_year_property_tax_per_150m_block_dollars"]) * site_count / 5.0
        )
        derivation = {
            "low": "Published JLARC low endpoint across example localities.",
            "reference": "Arithmetic midpoint of the published $0.4M to $10.8M five-year range; not a separate observed locality.",
            "high": "Published JLARC high endpoint across example localities.",
        }[case_key]
        rows.append(
            {
                "fiscal_case": case_key,
                "fiscal_case_display": case["display_name"],
                "site_count": site_count,
                "assumed_taxable_equipment_basis_dollars": block * site_count,
                "annual_local_property_tax_revenue_dollars": annual_property,
                "annual_local_sales_tax_exemption_burden_dollars": annual_local_sales_exemption,
                "annual_net_local_fiscal_flow_dollars": annual_property
                - annual_local_sales_exemption,
                "property_tax_value_derivation": derivation,
                "accounting_note": "Host-locality fiscal incidence; revenue and exemption are transfers, not additive national welfare.",
            }
        )
    return pd.DataFrame(rows)

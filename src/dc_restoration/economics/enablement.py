from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dc_restoration.paths import repository_root


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


def _capital_recovery_factor(rate: float, years: int) -> float:
    if rate == 0:
        return 1.0 / years
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def _source_unit_costs(prior_items: pd.DataFrame) -> pd.DataFrame:
    base = prior_items.loc[prior_items["cost_scope"].eq("existing_intertie_retrofit")].copy()
    feeder = prior_items.loc[
        prior_items["cost_scope"].eq("dedicated_express_feeders")
        & prior_items["line_item"].eq("Routed underground dedicated feeder")
    ].copy()
    source = pd.concat([base, feeder], ignore_index=True)
    source = source.drop_duplicates(subset=["line_item"], keep="first")
    required = {
        "Switchgear modification",
        "Microgrid controller",
        "Cybersecurity and commissioning",
        "Communications, metering, relays, and breakers",
        "New SCADA integration",
        "Routed underground dedicated feeder",
    }
    missing = sorted(required - set(source["line_item"]))
    if missing:
        raise ValueError(f"Missing source cost line items: {missing}")
    numeric = ["quantity", "unit_cost_2022_dollars", "line_item_cost_2022_dollars"]
    source[numeric] = source[numeric].apply(pd.to_numeric, errors="raise")
    return source[
        [
            "line_item",
            "quantity",
            "quantity_unit",
            "unit_cost_2022_dollars",
            "line_item_cost_2022_dollars",
            "source",
        ]
    ].reset_index(drop=True)


def _capital_cost_cases(
    config: dict[str, Any], source_costs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    capital = config["incremental_capital"]
    cpi = float(capital["cpi_u_2025"]) / float(capital["cpi_u_2022"])
    contract_factor = float(capital["area_cost_factor"]) + float(
        capital["contingency_fraction_of_direct_subtotal"]
    )
    indirect_factor = (
        1.0
        + float(capital["overhead_fraction_of_loaded_contract"])
        + float(capital["design_fraction_of_loaded_contract"])
    )
    total_loading_factor = contract_factor * indirect_factor * cpi
    lookup = source_costs.set_index("line_item")
    line_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for case_key, case in capital["cost_cases"].items():
        included = list(case["included_line_items"])
        unknown = sorted(set(included) - set(lookup.index))
        if unknown:
            raise ValueError(f"Unknown source line items in {case_key}: {unknown}")
        direct_2022 = float(lookup.loc[included, "line_item_cost_2022_dollars"].sum())
        for estimate_class, class_multiplier in capital["estimate_class_multipliers"].items():
            class_multiplier = float(class_multiplier)
            total_2025 = direct_2022 * total_loading_factor * class_multiplier
            summary_rows.append(
                {
                    "cost_case": case_key,
                    "cost_case_display": case["display_name"],
                    "estimate_class": estimate_class,
                    "estimate_class_multiplier": class_multiplier,
                    "scope_completeness": case["completeness"],
                    "scope_role": case["role"],
                    "direct_subtotal_2022_dollars": direct_2022,
                    "loaded_cost_factor_2022_to_2025": total_loading_factor,
                    "one_time_incremental_capital_2025_dollars": total_2025,
                    "aggregate_export_mw": 15.0,
                    "capital_dollars_per_export_kw": total_2025 / 15000.0,
                    "is_primary_point_estimate": False,
                    "reason_no_primary_point": "Site export-readiness and fixed-O&M inputs are unresolved; the break-even envelope is primary.",
                }
            )
            for line_item in included:
                row = lookup.loc[line_item]
                line_rows.append(
                    {
                        "cost_case": case_key,
                        "cost_case_display": case["display_name"],
                        "estimate_class": estimate_class,
                        "line_item": line_item,
                        "quantity": float(row["quantity"]),
                        "quantity_unit": row["quantity_unit"],
                        "unit_cost_2022_dollars": float(row["unit_cost_2022_dollars"]),
                        "direct_line_item_2022_dollars": float(row["line_item_cost_2022_dollars"]),
                        "allocated_loaded_line_item_2025_dollars": float(
                            row["line_item_cost_2022_dollars"]
                        )
                        * total_loading_factor
                        * class_multiplier,
                        "source": row["source"],
                        "scope_role": case["role"],
                    }
                )
    return pd.DataFrame(line_rows), pd.DataFrame(summary_rows)


def _stage1(
    config: dict[str, Any], events: pd.DataFrame, cost_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = config["analysis"]
    crf = _capital_recovery_factor(
        float(analysis["real_discount_rate"]), int(analysis["project_life_years"])
    )
    fom_values = [
        float(value) for value in config["fixed_om"]["annual_incremental_fom_screening_dollars"]
    ]
    bcr_rows: list[dict[str, Any]] = []
    envelope_rows: list[dict[str, Any]] = []

    for event in events.itertuples(index=False):
        benefit = float(event.event_avoided_outage_damage_dollars)
        event_cost = float(event.selected_full_resource_event_cost_dollars)
        for recurrence in analysis["recurrence_interval_years"]:
            recurrence = float(recurrence)
            annual_benefit = benefit / recurrence
            annual_event_cost = event_cost / recurrence
            annual_net_before_capital_fom = annual_benefit - annual_event_cost
            for annual_fom in fom_values:
                affordable_annual_capital = max(0.0, annual_net_before_capital_fom - annual_fom)
                break_even_capital = affordable_annual_capital / crf
                envelope_rows.append(
                    {
                        "resource_case": event.resource_case_display,
                        "recurrence_interval_years": recurrence,
                        "event_avoided_outage_damage_dollars": benefit,
                        "event_full_resource_operating_cost_dollars": event_cost,
                        "annualized_avoided_outage_damage_dollars": annual_benefit,
                        "annualized_event_operating_cost_dollars": annual_event_cost,
                        "annual_incremental_fixed_om_dollars": annual_fom,
                        "annual_amount_available_for_capital_recovery_dollars": affordable_annual_capital,
                        "capital_recovery_factor": crf,
                        "maximum_one_time_incremental_capital_dollars": break_even_capital,
                        "decision_interpretation": "BCR >= 1 only when annual capital recovery plus fixed O&M does not exceed annual net event benefit.",
                    }
                )
                for estimate in cost_summary.itertuples(index=False):
                    capital = float(estimate.one_time_incremental_capital_2025_dollars)
                    annual_capital = capital * crf
                    total_annual_cost = annual_event_cost + annual_capital + annual_fom
                    bcr_rows.append(
                        {
                            "analysis_stage": "Lifecycle export-enablement assessment",
                            "analysis_label": config["documentation"]["stage1_label"],
                            "resource_case": event.resource_case_display,
                            "recurrence_interval_years": recurrence,
                            "cost_case": estimate.cost_case,
                            "cost_case_display": estimate.cost_case_display,
                            "estimate_class": estimate.estimate_class,
                            "scope_completeness": estimate.scope_completeness,
                            "event_avoided_outage_damage_dollars": benefit,
                            "event_full_resource_operating_cost_dollars": event_cost,
                            "annualized_avoided_outage_damage_dollars": annual_benefit,
                            "annualized_event_operating_cost_dollars": annual_event_cost,
                            "one_time_incremental_capital_dollars": capital,
                            "capital_recovery_factor": crf,
                            "equivalent_annual_capital_charge_dollars": annual_capital,
                            "annual_incremental_fixed_om_dollars": annual_fom,
                            "annual_total_incremental_resource_cost_dollars": total_annual_cost,
                            "engineering_resource_bcr": _safe_ratio(
                                annual_benefit, total_annual_cost
                            ),
                            "annual_net_resource_benefit_dollars": annual_benefit
                            - total_annual_cost,
                            "annual_fixed_om_headroom_at_this_capital_dollars": annual_net_before_capital_fom
                            - annual_capital,
                            "capital_is_one_time": True,
                            "payer_share_affects_stage1_resource_bcr": False,
                        }
                    )

    bcr = pd.DataFrame(bcr_rows)
    envelope = pd.DataFrame(envelope_rows)
    reference = bcr.loc[
        bcr["recurrence_interval_years"].eq(float(analysis["reference_recurrence_years"]))
        & bcr["estimate_class"].eq("reference")
        & bcr["annual_incremental_fixed_om_dollars"].eq(0.0)
    ].copy()
    decision = envelope.loc[
        envelope["recurrence_interval_years"].eq(float(analysis["reference_recurrence_years"]))
        & envelope["annual_incremental_fixed_om_dollars"].eq(0.0)
    ].copy()
    decision = decision[
        [
            "resource_case",
            "recurrence_interval_years",
            "event_avoided_outage_damage_dollars",
            "event_full_resource_operating_cost_dollars",
            "annualized_avoided_outage_damage_dollars",
            "annualized_event_operating_cost_dollars",
            "annual_amount_available_for_capital_recovery_dollars",
            "capital_recovery_factor",
            "maximum_one_time_incremental_capital_dollars",
            "decision_interpretation",
        ]
    ]
    return bcr, envelope, reference, decision


def _stage2(
    config: dict[str, Any],
    events: pd.DataFrame,
    cost_summary: pd.DataFrame,
    rate_summary: pd.DataFrame,
    fiscal_summary: pd.DataFrame,
    data_centers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = config["analysis"]
    stage2 = config["stage2"]
    crf = _capital_recovery_factor(
        float(analysis["real_discount_rate"]), int(analysis["project_life_years"])
    )
    reference_costs = cost_summary.loc[cost_summary["estimate_class"].eq("reference")].copy()
    fom_values = [
        float(value) for value in config["fixed_om"]["annual_incremental_fom_screening_dollars"]
    ]
    total_backup_mw = float(data_centers["backup_gen_mw"].astype(float).sum())
    routine_mwh = (
        total_backup_mw
        * float(stage2["routine_generator_annual_hours"])
        * float(stage2["routine_generator_average_capacity_fraction"])
    )
    routine_health = routine_mwh * float(stage2["routine_generator_local_health_dollars_per_mwh"])
    routine_climate = routine_mwh * float(
        stage2["routine_generator_climate_damage_dollars_per_mwh"]
    )
    rows: list[dict[str, Any]] = []

    for event in events.itertuples(index=False):
        for recurrence in analysis["recurrence_interval_years"]:
            recurrence = float(recurrence)
            annual_resilience = float(event.event_avoided_outage_damage_dollars) / recurrence
            annual_outage_health = float(event.selected_local_health_damage_dollars) / recurrence
            annual_outage_climate = float(event.event_climate_damage_dollars) / recurrence
            for estimate in reference_costs.itertuples(index=False):
                capital = float(estimate.one_time_incremental_capital_2025_dollars)
                annual_capital = capital * crf
                for annual_fom in fom_values:
                    for share in stage2["community_enablement_shares"]:
                        share = float(share)
                        public_capital = annual_capital * share
                        public_fom = annual_fom * share
                        for rate in rate_summary.itertuples(index=False):
                            for fiscal in fiscal_summary.itertuples(index=False):
                                total_benefit = annual_resilience + float(
                                    fiscal.annual_local_property_tax_revenue_dollars
                                )
                                total_burden = (
                                    public_capital
                                    + public_fom
                                    + float(rate.equivalent_annual_attributed_rate_burden_dollars)
                                    + float(fiscal.annual_local_sales_tax_exemption_burden_dollars)
                                    + routine_health
                                    + annual_outage_health
                                )
                                rows.append(
                                    {
                                        "analysis_stage": "Host-community incidence assessment",
                                        "analysis_label": config["documentation"]["stage2_label"],
                                        "resource_case": event.resource_case_display,
                                        "recurrence_interval_years": recurrence,
                                        "cost_case": estimate.cost_case,
                                        "cost_case_display": estimate.cost_case_display,
                                        "community_enablement_share": share,
                                        "annual_incremental_fixed_om_dollars": annual_fom,
                                        "community_fixed_om_share": share,
                                        "rate_case": rate.rate_case,
                                        "rate_case_display": rate.rate_case_display,
                                        "fiscal_case": fiscal.fiscal_case,
                                        "fiscal_case_display": fiscal.fiscal_case_display,
                                        "annualized_resilience_benefit_dollars": annual_resilience,
                                        "annual_local_property_tax_revenue_dollars": float(
                                            fiscal.annual_local_property_tax_revenue_dollars
                                        ),
                                        "annual_total_host_community_benefit_dollars": total_benefit,
                                        "one_time_total_incremental_capital_dollars": capital,
                                        "equivalent_annual_total_capital_charge_dollars": annual_capital,
                                        "annual_public_enablement_capital_dollars": public_capital,
                                        "annual_public_incremental_fixed_om_dollars": public_fom,
                                        "annual_private_or_noncommunity_capital_and_fom_dollars": (
                                            annual_capital + annual_fom
                                        )
                                        * (1.0 - share),
                                        "annual_attributed_residential_rate_burden_dollars": float(
                                            rate.equivalent_annual_attributed_rate_burden_dollars
                                        ),
                                        "annual_local_sales_tax_exemption_burden_dollars": float(
                                            fiscal.annual_local_sales_tax_exemption_burden_dollars
                                        ),
                                        "annual_routine_generator_health_damage_dollars": routine_health,
                                        "annual_routine_generator_climate_damage_dollars": routine_climate,
                                        "annualized_outage_export_health_damage_dollars": annual_outage_health,
                                        "annualized_outage_export_climate_damage_dollars": annual_outage_climate,
                                        "annual_broader_societal_climate_externality_dollars": routine_climate
                                        + annual_outage_climate,
                                        "annual_total_host_community_burden_dollars": total_burden,
                                        "host_community_benefit_to_burden_ratio": _safe_ratio(
                                            total_benefit, total_burden
                                        ),
                                        "annual_net_host_community_value_dollars": total_benefit
                                        - total_burden,
                                        "technical_resource_cost_changes_with_payer": False,
                                        "incidence_ratio_changes_with_payer": True,
                                        "ratio_scope_note": "Host-community incidence screen; not a national social welfare BCR.",
                                    }
                                )
    all_rows = pd.DataFrame(rows)
    primary = all_rows.loc[
        all_rows["recurrence_interval_years"].eq(float(analysis["reference_recurrence_years"]))
        & all_rows["rate_case"].eq(stage2["selected_rate_case"])
        & all_rows["fiscal_case"].eq(stage2["selected_fiscal_case"])
    ].copy()
    return all_rows, primary

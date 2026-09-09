"""Invented accounting units; these are not paper financial assumptions or artifacts."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from dc_restoration.configuration import load_config
from dc_restoration.economics import benefit_cost as bc


@pytest.fixture
def accounting_config(tmp_path):
    config = load_config("configs/economics/benefit_cost.yaml")
    config["analysis"]["physical_subset"] = "toy"
    config["analysis"]["event_cost_valuation_method"] = "toy costs"
    rows = []
    for subset, scale in [("toy", 1), ("other toy", 2)]:
        for policy, factor in [("GraphRL-A2C", 1), ("Greedy", 0.5)]:
            for scenario in ("invented-a", "invented-b"):
                for i, code in enumerate(["R0_no_dc_anchor", *bc.RESOURCE_CONFIGS.values()]):
                    rows.append(
                        {
                            "eval_subset": subset,
                            "display_policy": policy,
                            "scenario_key": scenario,
                            "random_seed": 32,
                            "dc_resource_configuration": code,
                            "harmonized_economic_damage_dollars": 1000 - 100 * i * factor * scale,
                            "energy_not_served_mwh": 10 - i,
                            "critical_energy_not_served_mwh": 5 - i,
                            "dc_support_used_mwh": i,
                        }
                    )
    dc = pd.DataFrame(rows)
    event = pd.DataFrame(
        [
            {
                "valuation_method": name,
                "event_dc_support_mwh": 1,
                "event_provider_fuel_and_vom_dollars": 10 * scale,
                "selected_local_health_damage_dollars": 2 * scale,
                "event_climate_damage_dollars": 5 * scale,
            }
            for name, scale in [("toy costs", 1), ("other costs", 2)]
        ]
    )
    capital = pd.DataFrame(
        [
            {
                "cost_case": case,
                "estimate_class": estimate,
                "one_time_incremental_capital_2025_dollars": amount,
            }
            for case, estimate, amount in [
                ("high_reuse_integration_screen", "reference", 1000),
                ("alternative", "reference", 2000),
                ("high_reuse_integration_screen", "upper", 3000),
            ]
        ]
    )
    community = pd.DataFrame(
        [
            {
                **dict(zip(bc.COMMUNITY_COLUMNS, [200 * f, 10 * r, 20 * f, 3, 7])),
                "rate_case": rate,
                "fiscal_case": fiscal,
            }
            for rate, r in [("reference_half_growth", 1), ("alternative", 2)]
            for fiscal, f in [("reference", 1), ("alternative", 2)]
        ]
    )
    for key, frame in [
        ("dc_results", dc),
        ("event_costs", event),
        ("capital_summary", capital),
        ("community_ledger", community),
    ]:
        path = tmp_path / f"{key}.csv"
        frame.to_csv(path, index=False)
        config["inputs"][key] = str(path)
    config["output_dir"] = str(tmp_path / "ledgers")
    return config


def test_climate_boundaries_and_annual_no_double_count(accounting_config):
    tables = bc.calculate(accounting_config)
    event = tables["event_ledger"].iloc[0]
    assert event.event_full_resource_cost_dollars == 10 + 2 + 5
    assert event.event_lifecycle_accounted_cost_dollars == event.event_full_resource_cost_dollars
    assert event.event_host_community_externality_dollars == 2
    host = tables["host_community_ratio"].iloc[0]  # zero community capital share
    assert host.annual_total_host_community_burden_dollars == pytest.approx(10 + 20 + 3 + 2 / 10)
    assert host.annual_broader_societal_climate_externality_dollars == pytest.approx(7 + 5 / 10)
    annual, incremental = tables["annual_four_case_comparison"], tables["annual_incremental_bcr"]
    for i in range(2):
        before, after = annual.iloc[2 * i], annual.iloc[2 * i + 1]
        assert (
            after.annual_net_community_value_dollars - before.annual_net_community_value_dollars
            == pytest.approx(incremental.iloc[i].incremental_net_value_dollars)
        )
    assert annual.iloc[3].annual_community_benefit_or_inflow_dollars == 200
    assert incremental.iloc[1].incremental_benefit_dollars == 200 + 300


def test_climate_perturbation_changes_only_societal_boundary(accounting_config):
    before = bc.calculate(accounting_config)
    for key, column in [
        ("event_costs", "event_climate_damage_dollars"),
        ("community_ledger", "annual_routine_generator_climate_damage_dollars"),
    ]:
        path = accounting_config["inputs"][key]
        frame = pd.read_csv(path)
        frame[column] *= 9
        frame.to_csv(path, index=False)
    after = bc.calculate(accounting_config)
    assert (
        after["anchor_enablement_bcr"].benefit_cost_ratio
        < before["anchor_enablement_bcr"].benefit_cost_ratio
    ).all()
    for table, column in [
        ("host_community_ratio", "host_community_benefit_to_burden_ratio"),
        ("annual_incremental_bcr", "incremental_benefit_cost_ratio"),
        ("annual_four_case_comparison", "annual_net_community_value_dollars"),
    ]:
        pd.testing.assert_series_equal(before[table][column], after[table][column])
    assert (
        after["host_community_ratio"].annual_broader_societal_climate_externality_dollars
        > before["host_community_ratio"].annual_broader_societal_climate_externality_dollars
    ).all()


@pytest.mark.parametrize(
    "section,key,value,table,column",
    [
        ("analysis", "selected_policy", "Greedy", "event_ledger", "mean_event_benefit_dollars"),
        ("analysis", "physical_subset", "other toy", "event_ledger", "mean_event_benefit_dollars"),
        (
            "analysis",
            "event_cost_valuation_method",
            "other costs",
            "event_ledger",
            "event_full_resource_cost_dollars",
        ),
        (
            "analysis",
            "reference_event_interval_years",
            20,
            "host_community_ratio",
            "annualized_resilience_benefit_dollars",
        ),
        (
            "lifecycle",
            "capital_case",
            "alternative",
            "anchor_enablement_bcr",
            "one_time_incremental_capital_dollars",
        ),
        (
            "lifecycle",
            "estimate_class",
            "upper",
            "anchor_enablement_bcr",
            "one_time_incremental_capital_dollars",
        ),
        (
            "lifecycle",
            "real_discount_rate",
            0.08,
            "anchor_enablement_bcr",
            "capital_recovery_factor",
        ),
        ("lifecycle", "service_life_years", 40, "anchor_enablement_bcr", "capital_recovery_factor"),
        (
            "lifecycle",
            "fixed_export_om_dollars_per_year",
            50,
            "anchor_enablement_bcr",
            "annual_total_cost_dollars",
        ),
        (
            "community",
            "rate_case",
            "alternative",
            "host_community_ratio",
            "annual_total_host_community_burden_dollars",
        ),
        (
            "community",
            "fiscal_case",
            "alternative",
            "host_community_ratio",
            "annual_total_host_community_benefit_dollars",
        ),
        ("community", "reference_share", 1.0, "annual_incremental_bcr", "incremental_cost_dollars"),
        (
            "environment",
            "lifecycle_include_climate",
            False,
            "anchor_enablement_bcr",
            "event_resource_cost_dollars",
        ),
        (
            "environment",
            "lifecycle_include_local_air_quality",
            False,
            "anchor_enablement_bcr",
            "event_resource_cost_dollars",
        ),
        (
            "environment",
            "host_include_local_air_quality",
            False,
            "host_community_ratio",
            "annual_total_host_community_burden_dollars",
        ),
    ],
)
def test_every_scalar_assumption_controls_outputs(
    accounting_config, section, key, value, table, column
):
    before = bc.calculate(accounting_config)
    config = deepcopy(accounting_config)
    config[section][key] = value
    after = bc.calculate(config)
    assert not np.allclose(before[table][column], after[table][column])


def test_assessment_grids_and_om_incidence(accounting_config):
    config = deepcopy(accounting_config)
    config["analysis"]["event_intervals_years"] = [10, 30]
    config["community"]["shares"] = [0.5, 0.75]
    config["lifecycle"]["fixed_export_om_dollars_per_year"] = 100
    tables = bc.calculate(config)
    assert set(tables["anchor_enablement_bcr"].event_interval_years) == {10, 30}
    assert set(tables["host_community_ratio"].community_enablement_share) == {0.5, 0.75}
    assert set(tables["host_community_ratio"].annual_public_incremental_fixed_om_dollars) == {
        50,
        75,
    }
    assert bc.capital_recovery_factor(0, 20) == 0.05


@pytest.mark.parametrize("section", [None, *bc.CONFIG_KEYS])
def test_unknown_keys_rejected(accounting_config, section):
    target = accounting_config if section is None else accounting_config[section]
    target["unused_option"] = 1
    with pytest.raises(ValueError, match="keys|requires"):
        bc.calculate(accounting_config)


@pytest.mark.parametrize(
    "failure",
    [
        "missing capital",
        "duplicate capital",
        "missing fiscal",
        "inconsistent recurring",
        "zero support",
        "missing pair",
        "wrong seed",
    ],
)
def test_bad_external_ledgers_fail_clearly(accounting_config, failure):
    key = (
        "capital_summary"
        if "capital" in failure
        else "community_ledger"
        if failure in ("missing fiscal", "inconsistent recurring")
        else "event_costs"
        if failure == "zero support"
        else "dc_results"
    )
    path = accounting_config["inputs"][key]
    frame = pd.read_csv(path)
    if failure.startswith("missing capital"):
        frame = frame.iloc[1:]
    elif failure == "duplicate capital":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif failure == "missing fiscal":
        frame["fiscal_case"] = "unavailable"
    elif failure == "inconsistent recurring":
        extra = frame.iloc[:1].copy()
        extra[bc.COMMUNITY_COLUMNS[0]] += 1
        frame = pd.concat([frame, extra])
    elif failure == "zero support":
        frame.loc[0, "event_dc_support_mwh"] = 0
    elif failure == "missing pair":
        frame = frame.iloc[1:]
    else:
        frame.loc[1, "random_seed"] = 99
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        bc.calculate(accounting_config)


def test_benefit_cost_cli_respects_config_and_output_override(
    monkeypatch, tmp_path, accounting_config
):
    import sys

    import yaml

    from dc_restoration import cli

    path = tmp_path / "accounting.yaml"
    path.write_text(yaml.safe_dump(accounting_config), encoding="utf-8")
    output = tmp_path / "explicit-output"
    monkeypatch.setattr(
        sys, "argv", ["run_benefit_cost", "--config", str(path), "--output-dir", str(output)]
    )
    cli.benefit_cost_main()
    assert len(list(output.glob("*.csv"))) == 5
    assert not (tmp_path / "ledgers").exists()


def test_source_incidence_helper_uses_same_climate_boundary():
    from dc_restoration.economics.enablement import _stage2

    config = load_config("configs/economics/enablement.yaml")
    config["analysis"]["recurrence_interval_years"] = [10]
    config["fixed_om"]["annual_incremental_fom_screening_dollars"] = [0]
    config["stage2"]["community_enablement_shares"] = [0.5]
    events = pd.DataFrame(
        [
            {
                "event_avoided_outage_damage_dollars": 100,
                "selected_local_health_damage_dollars": 2,
                "event_climate_damage_dollars": 5,
                "resource_case_display": "E12",
            }
        ]
    )
    capital = pd.DataFrame(
        [
            {
                "estimate_class": "reference",
                "one_time_incremental_capital_2025_dollars": 1000,
                "cost_case": "toy",
                "cost_case_display": "Toy",
            }
        ]
    )
    rate = pd.DataFrame(
        [
            {
                "equivalent_annual_attributed_rate_burden_dollars": 10,
                "rate_case": "reference_half_growth",
                "rate_case_display": "Toy rate",
            }
        ]
    )
    fiscal = pd.DataFrame(
        [
            {
                "annual_local_property_tax_revenue_dollars": 200,
                "annual_local_sales_tax_exemption_burden_dollars": 20,
                "fiscal_case": "reference",
                "fiscal_case_display": "Toy fiscal",
            }
        ]
    )
    data_centers = pd.DataFrame([{"backup_gen_mw": 1}])
    before, _ = _stage2(config, events, capital, rate, fiscal, data_centers)
    events["event_climate_damage_dollars"] *= 10
    config["stage2"]["routine_generator_climate_damage_dollars_per_mwh"] *= 10
    after, _ = _stage2(config, events, capital, rate, fiscal, data_centers)
    assert (
        before.iloc[0].host_community_benefit_to_burden_ratio
        == after.iloc[0].host_community_benefit_to_burden_ratio
    )
    assert (
        before.iloc[0].annual_broader_societal_climate_externality_dollars
        < after.iloc[0].annual_broader_societal_climate_externality_dollars
    )

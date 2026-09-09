"""Hand-constructed one-interval records, unrelated to the research dataset."""

import json
import sys

import pandas as pd
import pytest
from test_benefit_cost import accounting_config

from dc_restoration import cli
from dc_restoration.configuration import load_config
from dc_restoration.economics.benefit_cost import calculate
from dc_restoration.evaluation.analysis_tables import prepare_analysis_tables
from dc_restoration.plotting import paper

RESOURCES = [
    "R0_no_dc_anchor",
    "E12_finite_energy_12h",
    "E24_finite_energy_24h",
    "R1_current_power_bounded_anchor",
]


def toy_run(path, resource, subsets, policies=("Greedy", "GraphRL-A2C")):
    path.mkdir(parents=True)
    episodes, steps = [], []
    for subset in subsets:
        for policy in policies:
            i = RESOURCES.index(resource)
            mw = 10 - i
            metadata = {
                "eval_subset": subset,
                "scenario_key": "invented-only",
                "policy": policy,
                "display_policy": policy,
                "dc_resource_configuration": resource,
                "random_seed": 32,
            }
            episodes.append(
                {
                    **metadata,
                    "harmonized_economic_damage_dollars": (1000 - 100 * i)
                    * (0.9 if policy == "GraphRL-A2C" else 1),
                    "energy_not_served_mwh": mw * 2,
                    "critical_energy_not_served_mwh": mw,
                    "dc_support_used_mwh": 2 * i,
                    "episode_length": 1,
                    "runtime_seconds": 0.001,
                    "invalid_action_count": 0,
                    "export_cap_violation_count": 0,
                    "support_outside_anchor_zone_count": 0,
                    "energy_depleted": False,
                    "repair_sequence_edge_ids": '["e0"]',
                }
            )
            steps.append(
                {
                    **metadata,
                    "step_index": 0,
                    "interval_start_time_hours": 0,
                    "interval_duration_hours": 2,
                    "cumulative_time_hours": 2,
                    "initial_unserved_mw": mw,
                    "initial_critical_unserved_mw": mw / 2,
                    "current_unserved_mw": mw,
                    "current_critical_unserved_mw": mw / 2,
                    "step_unserved_energy_mwh": mw * 2,
                    "cumulative_ens_mwh": mw * 2,
                    "selected_repair_action": "e0",
                }
            )
    pd.DataFrame(episodes).to_csv(path / "results_by_episode.csv", index=False)
    pd.DataFrame(steps).to_csv(path / "trajectory_by_step.csv", index=False)
    return path


@pytest.fixture
def analysis_inputs(tmp_path):
    config = load_config("configs/evaluation/paper.yaml")
    subsets = config["subsets"]["controlled_main"] + ["toy"]
    controlled = toy_run(tmp_path / "controlled", RESOURCES[-1], subsets)
    resources = [
        toy_run(tmp_path / f"resource{i}", resource, subsets)
        for i, resource in enumerate(RESOURCES)
    ]
    return config, controlled, resources


def test_tiny_evaluation_to_analysis_to_benefit_cost_and_plot_schemas(
    tmp_path, analysis_inputs, accounting_config
):
    config, controlled, resources = analysis_inputs
    output = tmp_path / "analysis"
    tables = prepare_analysis_tables(
        config=config, output_dir=output, controlled_dirs=[controlled], resource_dirs=resources
    )
    assert len(tables["dc_anchor/dc_trajectory_unique_by_step"]) == 8
    assert len(tables["controlled/main_hard_subset_average"]) == 1
    assert tables["controlled/main_hard_subset_average"].iloc[
        0
    ].mean_economic_improvement_percent == pytest.approx(10)
    accounting_config["inputs"]["dc_results"] = str(output / "dc_anchor/dc_results_by_episode.csv")
    ledgers = calculate(accounting_config)
    assert set(ledgers["event_ledger"].resource_case) == {"E12", "E24", "Power-only reference"}
    curves = paper.unique_hard_cohort_dc_curves(
        tables["dc_anchor/dc_trajectory_unique_by_step"], pd.Series(["invented-only"])
    )
    assert set(curves.dc_resource_configuration) == set(RESOURCES)
    assert len(json.loads((output / "analysis_input_manifest.json").read_text())) == 10


def test_fresh_unique_manifest_and_ga_tables(tmp_path, analysis_inputs):
    config, _, _ = analysis_inputs
    fresh = toy_run(
        tmp_path / "fresh", RESOURCES[-1], ["fresh_historical_nonzero", "fresh_unique_bank"]
    )
    manifest = tmp_path / "unique.csv"
    pd.DataFrame([{"scenario_key": "invented-only", "random_seed": 32}]).to_csv(
        manifest, index=False
    )
    ga_dir = tmp_path / "ga"
    ga_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_key": "invented-only",
                "paper_primary_economic_damage_dollars": 660,
                "runtime_batch_elapsed_seconds": 0.002,
                "fitness_evaluations": 1,
                "population": 1,
                "generations": 1,
            }
        ]
    ).to_csv(ga_dir / "bounded_ga_results.csv", index=False)
    tables = prepare_analysis_tables(
        config=config,
        output_dir=tmp_path / "combined",
        fresh_dirs=[fresh],
        fresh_unique_manifest=manifest,
        ga_dir=ga_dir,
        ga_evaluation_dirs=[fresh],
    )
    assert len(tables["fresh/unique_bank_results_by_episode"]) == 2
    comparison = tables["bounded_ga/bounded_ga_comparison_by_scenario"].iloc[0]
    assert comparison.selected_improvement_vs_greedy_percent == pytest.approx(10)
    assert comparison.bounded_ga_improvement_vs_greedy_percent == pytest.approx(100 * 40 / 700)


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "unknown method",
        "unknown resource",
        "missing column",
        "trajectory mismatch",
        "missing baseline",
        "seed conflict",
        "resource pair",
    ],
)
def test_bad_inputs_fail_without_writing_output(tmp_path, analysis_inputs, failure):
    config, controlled, resources = analysis_inputs
    file = (resources[1] if failure == "resource pair" else controlled) / "results_by_episode.csv"
    frame = pd.read_csv(file)
    if failure == "duplicate":
        extra = frame.iloc[:1].copy()
        extra["harmonized_economic_damage_dollars"] += 1
        frame = pd.concat([frame, extra])
    elif failure == "unknown method":
        frame.loc[0, "policy"] = "Unknown"
    elif failure == "unknown resource":
        frame.loc[0, "dc_resource_configuration"] = "Unknown"
    elif failure == "missing column":
        frame = frame.drop(columns="runtime_seconds")
    elif failure == "trajectory mismatch":
        frame.loc[0, "episode_length"] = 2
    elif failure == "seed conflict":
        frame.loc[0, "random_seed"] = 99
    elif failure == "missing baseline":
        frame = frame[frame.policy.ne("Greedy")]
        tr_file = controlled / "trajectory_by_step.csv"
        tr = pd.read_csv(tr_file)
        tr[tr.policy.ne("Greedy")].to_csv(tr_file, index=False)
    else:
        frame = frame[frame.eval_subset.ne("toy")]
        tr_file = resources[1] / "trajectory_by_step.csv"
        tr = pd.read_csv(tr_file)
        tr[tr.eval_subset.ne("toy")].to_csv(tr_file, index=False)
    frame.to_csv(file, index=False)
    output = tmp_path / "should-not-exist"
    with pytest.raises(ValueError):
        prepare_analysis_tables(
            config=config, output_dir=output, controlled_dirs=[controlled], resource_dirs=resources
        )
    assert not output.exists()


def test_cli_paths_and_exact_duplicate_collapse(tmp_path, analysis_inputs, monkeypatch):
    _, controlled, _ = analysis_inputs
    output = tmp_path / "cli-analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_analysis_tables",
            "--controlled-dir",
            str(controlled),
            "--controlled-dir",
            str(controlled),
            "--output-dir",
            str(output),
        ],
    )
    cli.analysis_tables_main()
    assert len(pd.read_csv(output / "controlled/results_by_episode.csv")) == 10

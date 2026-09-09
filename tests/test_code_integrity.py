"""Regression checks for release interfaces, using only invented inputs."""

import importlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from test_analysis_tables import RESOURCES, toy_run
from test_benefit_cost import accounting_config

from dc_restoration import cli
from dc_restoration.configuration import load_config
from dc_restoration.economics import benefit_cost as bc
from dc_restoration.evaluation.analysis_tables import prepare_analysis_tables, read_runs
from dc_restoration.paths import output_directory, repository_root


def test_trajectory_seed_must_match_episode(tmp_path):
    run = toy_run(tmp_path / "run", RESOURCES[-1], ["toy"])
    path = run / "trajectory_by_step.csv"
    frame = pd.read_csv(path)
    frame["random_seed"] = 99
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="seed|orphan"):
        read_runs([run], role="toy", audit=[])


@pytest.mark.parametrize("seed", [None, -1, 1.5, float("inf"), "invalid"])
def test_accounting_requires_valid_pairing_seeds(accounting_config, seed):
    path = accounting_config["inputs"]["dc_results"]
    frame = pd.read_csv(path)
    if seed is None:
        frame = frame.drop(columns="random_seed")
    else:
        frame["random_seed"] = seed
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="random_seed"):
        bc.calculate(accounting_config)


def test_zero_repair_evaluation_remains_readable(monkeypatch, tmp_path):
    case = tmp_path / "case"
    shutil.copytree(repository_root() / "tests/fixtures/tiny_case", case)
    manifest = pd.DataFrame(
        [{"scenario_key": "toy_intact", "eval_subset": "toy", "random_seed": 32}]
    )
    manifest.to_csv(
        case / "scenario_design/benchmark_hard_hurricane_scenario_manifest.csv", index=False
    )
    selected = tmp_path / "manifest.csv"
    manifest.to_csv(selected, index=False)
    run = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--method",
            "Greedy",
            "Random Top-K",
            "--case-dir",
            str(case),
            "--manifest",
            str(selected),
            "--output-dir",
            str(run),
        ],
    )
    cli.evaluate_main()
    tables = prepare_analysis_tables(
        config=load_config("configs/evaluation/paper.yaml"),
        output_dir=tmp_path / "analysis",
        controlled_dirs=[run],
    )
    assert (tables["controlled/results_by_episode"].episode_length == 0).all()
    assert pd.read_csv(run / "trajectory_by_step.csv").empty


@pytest.mark.parametrize("seed", [1.5, -1, float("inf"), "invalid"])
def test_evaluation_rejects_invalid_seeds_before_rollout(monkeypatch, tmp_path, seed):
    from dc_restoration.policies import training

    selected = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"scenario_key": "toy_damage", "eval_subset": "toy", "random_seed": seed}]
    ).to_csv(selected, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--case-dir",
            str(repository_root() / "tests/fixtures/tiny_case"),
            "--manifest",
            str(selected),
            "--output-dir",
            str(tmp_path / "evaluation"),
        ],
    )
    monkeypatch.setattr(
        training, "_rollout_harmonized", lambda *a, **k: pytest.fail("invalid seed reached rollout")
    )
    with pytest.raises(SystemExit):
        cli.evaluate_main()
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize(
    "module,function",
    [
        ("dc_restoration.plotting.schematic", "setup_style"),
        ("dc_restoration.evaluation.trajectories", "_setup_style"),
    ],
)
def test_plot_styles_select_headless_backend(monkeypatch, module, function):
    calls = []
    monkeypatch.setattr(mpl, "use", lambda backend: calls.append(backend))
    getattr(importlib.import_module(module), function)()
    assert calls == ["Agg"]


@pytest.mark.parametrize("destination", ["src", "configs", ".", "unignored", "outputs/../tests"])
def test_repository_output_cannot_target_release_candidates(monkeypatch, tmp_path, destination):
    monkeypatch.setenv("DC_RESTORATION_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="ignored"):
        output_directory(destination)


def test_benefit_cost_rejects_nonempty_destination(monkeypatch, tmp_path, accounting_config):
    marker = tmp_path / "existing.csv"
    marker.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr("dc_restoration.configuration.load_config", lambda _: accounting_config)
    monkeypatch.setattr(sys, "argv", ["run_benefit_cost", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="empty"):
        cli.benefit_cost_main()
    assert marker.read_text() == "preserve me"
    assert not (tmp_path / "event_ledger.csv").exists()


@pytest.mark.parametrize(
    "key,column",
    [
        ("dc_results", "harmonized_economic_damage_dollars"),
        ("event_costs", "selected_local_health_damage_dollars"),
        ("capital_summary", "one_time_incremental_capital_2025_dollars"),
        ("community_ledger", bc.COMMUNITY_COLUMNS[0]),
    ],
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "invalid"])
def test_malformed_accounting_inputs_fail(accounting_config, key, column, value):
    path = accounting_config["inputs"][key]
    frame = pd.read_csv(path)
    frame[column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite nonnegative"):
        bc.calculate(accounting_config)


@pytest.mark.parametrize("failure", ["scenario", "seed", "duplicate", "missing"])
def test_fresh_manifest_cannot_substitute_cohorts(tmp_path, failure):
    run = toy_run(tmp_path / "fresh", RESOURCES[-1], ["fresh_unique_bank"])
    frame = pd.DataFrame([{"scenario_key": "invented-only", "random_seed": 32}])
    if failure == "scenario":
        frame["scenario_key"] = "different-toy"
    elif failure == "seed":
        frame["random_seed"] = 99
    elif failure == "duplicate":
        frame = pd.concat([frame, frame])
    else:
        frame = frame.iloc[:0]
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="manifest|Manifest"):
        prepare_analysis_tables(
            config=load_config("configs/evaluation/paper.yaml"),
            output_dir=tmp_path / "analysis",
            fresh_dirs=[run],
            fresh_unique_manifest=path,
        )
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize(
    "local,climate", [(False, False), (False, True), (True, False), (True, True)]
)
def test_lifecycle_switches_include_exact_cost_components(accounting_config, local, climate):
    accounting_config["environment"]["lifecycle_include_local_air_quality"] = local
    accounting_config["environment"]["lifecycle_include_climate"] = climate
    row = bc.calculate(accounting_config)["event_ledger"].iloc[0]
    assert row.event_lifecycle_accounted_cost_dollars == 10 + 2 * local + 5 * climate
    assert row.event_full_resource_cost_dollars == 17
    assert row.event_host_community_externality_dollars == 2


@pytest.mark.parametrize("failure", ["scenario", "seed", "duplicate"])
def test_ga_comparison_rejects_mismatched_cohorts(tmp_path, failure):
    run = toy_run(tmp_path / "evaluation", RESOURCES[-1], ["toy"])
    ga = tmp_path / "ga"
    ga.mkdir()
    frame = pd.DataFrame(
        [
            {
                "scenario_key": "different-toy" if failure == "scenario" else "invented-only",
                "paper_primary_economic_damage_dollars": 660,
                "runtime_batch_elapsed_seconds": 0.002,
                "fitness_evaluations": 1,
                "population": 1,
                "generations": 1,
            }
        ]
    )
    if failure == "duplicate":
        frame = pd.concat([frame, frame])
    if failure == "seed":
        for name in ("results_by_episode.csv", "trajectory_by_step.csv"):
            path = run / name
            rows = pd.read_csv(path)
            rows.loc[rows.policy.eq("GraphRL-A2C"), "random_seed"] = 99
            rows.to_csv(path, index=False)
    frame.to_csv(ga / "bounded_ga_results.csv", index=False)
    with pytest.raises(ValueError, match="GA"):
        prepare_analysis_tables(
            config=load_config("configs/evaluation/paper.yaml"),
            output_dir=tmp_path / "analysis",
            ga_dir=ga,
            ga_evaluation_dirs=[run],
        )
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize(
    "stage",
    [
        "parse_transmission",
        "parse_td_mapping",
        "scan_gulf_candidates",
        "download_distribution_subset",
        "parse_opendss_distribution",
        "extract_subnetwork",
        "aggregate_distribution_graph",
        "place_data_centers",
        "build_final_tables",
        "prepare_baseline_case_v0_1",
    ],
)
def test_prepare_stage_dispatch_without_preparing_data(monkeypatch, stage):
    calls = []

    def select(name):
        calls.append(name)
        return SimpleNamespace(main=lambda: calls.append(list(sys.argv)))

    monkeypatch.setattr(cli.importlib, "import_module", select)
    argv = ["prepare_inputs", "--stage", stage]
    monkeypatch.setattr(sys, "argv", argv)
    cli.prepare_main()
    assert calls == [
        "dc_restoration.testbed." + stage,
        ["prepare_inputs", "--config", str(repository_root() / "configs/testbed/base.yaml")],
    ]
    assert sys.argv is argv


@pytest.mark.parametrize("stage", ["benchmark", "benchmark_scenarios"])
def test_benchmark_dispatch_without_conversion_or_sampling(monkeypatch, stage):
    from dc_restoration.testbed import benchmark

    calls = []
    directories = {}
    monkeypatch.setattr(benchmark, "_make_dirs", lambda *a: directories)
    monkeypatch.setattr(benchmark, "convert_smartds_candidate", lambda *a: calls.append("convert"))
    monkeypatch.setattr(benchmark, "write_conversion_outputs", lambda *a: calls.append("write"))
    monkeypatch.setattr(
        benchmark, "generate_benchmark_scenarios", lambda *a: calls.append("scenarios")
    )
    monkeypatch.setattr(sys, "argv", ["prepare_inputs", "--stage", stage])
    cli.prepare_main()
    assert calls == (["convert", "write"] if stage == "benchmark" else ["scenarios"])


def test_ga_cli_dispatch_without_search(monkeypatch, tmp_path):
    from dc_restoration.evaluation import bounded_ga as ga

    cohort = tmp_path / "cohort.csv"
    pd.DataFrame([{"scenario_key": "toy_damage", "random_seed": 32}]).to_csv(cohort, index=False)
    case = repository_root() / "tests/fixtures/tiny_case"
    damage = case / "scenario_design/benchmark_hard_damage_components.csv"
    calls = []
    monkeypatch.setattr(ga, "_worker_init", lambda *a: calls.append(("init", a)))
    monkeypatch.setattr(
        ga,
        "_simulate_source_greedy",
        lambda *a: {"scenario_key": "toy_damage", "repair_sequence_edge_ids": "[]"},
    )
    monkeypatch.setattr(ga, "_run_harmonized_ga", lambda *a: calls.append(("ga", a)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bounded_ga",
            "--cohort",
            str(cohort),
            "--case-dir",
            str(case),
            "--damage-components",
            str(damage),
            "--output-dir",
            str(tmp_path / "ga"),
        ],
    )
    cli.ga_main()
    config = load_config("configs/evaluation/bounded_ga.yaml")
    assert [name for name, _ in calls] == ["init", "ga"]
    args = calls[1][1]
    assert args[2:4] == (case, damage)
    assert args[7:10] == tuple(
        config["runtime"][key]
        for key in ("harmonized_greedy_workers", "ga_population", "ga_generations")
    )
    assert args[-1] == tmp_path / "ga/bounded_ga_results.csv"


@pytest.mark.parametrize(
    "selector", ["system", "severity", "benefit-cost-a", "benefit-cost-b", "benefit-cost-all"]
)
def test_figure_cli_dispatch_without_drawing(monkeypatch, selector):
    from dc_restoration.plotting import api

    calls = []
    monkeypatch.setattr(api, "generate", lambda *args: calls.append(args))
    monkeypatch.setattr(
        sys, "argv", ["generate_figures", "--figure", selector, "--config", "custom.yaml"]
    )
    cli.figures_main()
    assert calls == [("custom.yaml", selector)]


def test_imports_do_not_access_research_inputs_or_start_work():
    code = r"""
import importlib
import pkgutil
import subprocess
import urllib.request
from pathlib import Path
import matplotlib.pyplot
import networkx
import numpy
import pandas
import pyarrow
import requests
import scipy
import shapely
import torch
import yaml

def forbidden(*args, **kwargs):
    raise AssertionError("Import attempted I/O, process creation, checkpoint loading, or CUDA initialization")

for name in ("read_csv", "read_parquet", "read_feather", "read_excel"):
    setattr(pandas, name, forbidden)
Path.mkdir = Path.write_text = Path.write_bytes = forbidden
requests.sessions.Session.request = urllib.request.urlopen = forbidden
subprocess.Popen = forbidden
torch.load = torch.cuda.init = torch.cuda._lazy_init = forbidden
import dc_restoration
names = [m.name for m in pkgutil.walk_packages(dc_restoration.__path__, dc_restoration.__name__ + ".")]
for name in names:
    importlib.import_module(name)
print(len(names))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) >= 62


def test_all_policy_plot_consumers_accept_analysis_tables(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    from dc_restoration.plotting import paper

    config = load_config("configs/evaluation/paper.yaml")
    subsets = [
        *paper.MAIN_SUBSETS,
        "benchmark_hard_historical_nonzero",
        "benchmark_hard_stress_intensity_scaled",
    ]
    controlled = toy_run(
        tmp_path / "controlled", RESOURCES[-1], subsets, policies=tuple(paper.DISPLAY)
    )
    resources = [
        toy_run(tmp_path / f"resource{i}", code, subsets) for i, code in enumerate(RESOURCES)
    ]
    fresh = toy_run(
        tmp_path / "fresh",
        RESOURCES[-1],
        [*paper.FRESH_HARD_SUBSETS, "fresh_historical_nonzero", "fresh_stress_intensity_scaled"],
        policies=tuple(paper.DISPLAY),
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"scenario_key": "invented-only", "random_seed": 32}]).to_csv(
        manifest, index=False
    )
    ga = tmp_path / "ga"
    ga.mkdir()
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
    ).to_csv(ga / "bounded_ga_results.csv", index=False)
    output = tmp_path / "analysis"
    prepare_analysis_tables(
        config=config,
        output_dir=output,
        controlled_dirs=[controlled],
        resource_dirs=resources,
        fresh_dirs=[fresh],
        fresh_unique_manifest=manifest,
        ga_dir=ga,
        ga_evaluation_dirs=[controlled],
    )
    # Exercise table consumers but suppress every figure export.
    monkeypatch.setattr(paper, "REFRESH", output)
    monkeypatch.setattr(paper, "OUT", tmp_path / "plot_consumers")
    (paper.OUT / "plot_ready").mkdir(parents=True)
    exported = []

    def capture(fig, family, stem):
        exported.append(stem)
        plt.close(fig)
        return {}

    monkeypatch.setattr(paper, "export", capture)
    monkeypatch.setattr(
        plt.Figure, "savefig", lambda *a, **k: pytest.fail("No figure files may be generated")
    )
    paper.setup_style()
    palette = {
        name: {"matplotlib_marker": "o", "color": "black", "matplotlib_line_style": "-"}
        for name in [*paper.DISPLAY, "Bounded GA"]
    }
    for number in ("3", "4", "5", "6", "7", "s1", "s2", "s3", "s4", "s5"):
        function = getattr(
            paper, "supplementary_" + number if number.startswith("s") else "main_figure" + number
        )
        function(palette) if number in ("3", "4", "5", "s1", "s2", "s5") else function()
    assert len(exported) == 10
    assert plt.get_fignums() == []


def test_capital_recovery_factor_and_community_capital_incidence(accounting_config):
    expected_crf = 0.031 * 1.031**20 / (1.031**20 - 1)
    assert bc.capital_recovery_factor(0.031, 20) == pytest.approx(expected_crf)
    host = bc.calculate(accounting_config)["host_community_ratio"]
    np.testing.assert_allclose(
        host.annual_public_enablement_capital_dollars,
        host.community_enablement_share * 1000 * expected_crf,
    )

"""Check dispatcher contracts without running research workloads."""

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dc_restoration import cli
from dc_restoration.configuration import load_config
from dc_restoration.paths import repository_root
from dc_restoration.policies.registry import TRAINABLE


@pytest.mark.parametrize(
    "name",
    [
        "prepare_inputs",
        "train",
        "evaluate",
        "prepare_analysis_tables",
        "run_bounded_ga",
        "run_benefit_cost",
        "generate_figures",
        "validate_installation",
    ],
)
def test_help_is_available(name):
    result = subprocess.run(
        [sys.executable, "-B", str(repository_root() / "scripts" / f"{name}.py"), "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("method", TRAINABLE)
def test_train_dispatch_only_requested_method(monkeypatch, tmp_path, method):
    from dc_restoration.policies import training

    calls = []
    monkeypatch.setattr(training, "_load_context", lambda config: None)
    monkeypatch.setattr(training, "_dirs", lambda output: {"output": output})
    monkeypatch.setattr(training, "_load_dataset", lambda config, dirs: {})
    monkeypatch.setattr(training, "_train_bc", lambda arch, *args: calls.append((arch, "bc")))
    monkeypatch.setattr(training, "_train_rl", lambda arch, alg, *args: calls.append((arch, alg)))
    monkeypatch.setattr(
        sys, "argv", ["train", "--method", method, "--device", "cpu", "--output-dir", str(tmp_path)]
    )
    cli.train_main()
    expected = cli.METHODS[method]
    assert calls[0] == (expected.architecture, "bc")
    assert calls[1:] == (
        [] if expected.algorithm == "bc" else [(expected.architecture, expected.algorithm)]
    )


def test_evaluate_cli_on_tiny_fixture(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"scenario_key": "toy_damage", "eval_subset": "synthetic_test", "random_seed": 32}]
    ).to_csv(manifest, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--method",
            "Greedy",
            "Random Top-K",
            "--case-dir",
            str(repository_root() / "tests/fixtures/tiny_case"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "evaluation"),
        ],
    )
    cli.evaluate_main()
    rows = pd.read_csv(tmp_path / "evaluation/results_by_episode.csv")
    assert set(rows["display_policy"]) == {"Greedy", "Random Top-K"}
    assert (rows["invalid_action_count"] == 0).all()
    from dc_restoration.evaluation.analysis_tables import prepare_analysis_tables

    tables = prepare_analysis_tables(
        config=load_config("configs/evaluation/paper.yaml"),
        output_dir=tmp_path / "analysis",
        controlled_dirs=[tmp_path / "evaluation"],
    )
    assert len(tables["controlled/pairwise_vs_greedy"]) == 1


@pytest.mark.parametrize("figure", ["3", "4", "5", "6", "7", "s1", "s2", "s3", "s4", "s5"])
def test_figure_dispatch_signature_without_drawing(monkeypatch, tmp_path, figure):
    from dc_restoration.plotting import api, paper

    palette = tmp_path / "palette.json"
    palette.write_text(json.dumps({"methods": {}}), encoding="utf-8")
    monkeypatch.setattr(
        api,
        "load_config",
        lambda _: {
            "output_dir": str(tmp_path),
            "paper_tables": str(tmp_path),
            "palette": str(palette),
        },
    )
    name = "supplementary_" + figure if figure.startswith("s") else "main_figure" + figure
    signature = inspect.signature(getattr(paper, name))
    calls = []

    def check(*args):
        signature.bind(*args)
        calls.append(True)

    monkeypatch.setattr(paper, name, check)
    monkeypatch.setattr(paper, "ensure_dirs", lambda: None)
    monkeypatch.setattr(paper, "setup_style", lambda: None)
    api.generate("unused", figure)
    assert calls == [True]


def test_preparation_shared_helper_import():
    from dc_restoration.testbed.build_final_tables import dominant_class

    rows = pd.DataFrame({"inferred_customer_class": ["residential", "commercial"], "p_kw": [1, 3]})
    assert dominant_class(rows) == "commercial"

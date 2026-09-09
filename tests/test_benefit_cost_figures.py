from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from test_benefit_cost import accounting_config

from dc_restoration.economics import benefit_cost as bc
from dc_restoration.plotting import api


@pytest.mark.parametrize(
    "selector,versions,count",
    [
        ("benefit-cost-a", {"version_a"}, 6),
        ("benefit-cost-b", {"version_b"}, 6),
        ("benefit-cost-all", {"version_a", "version_b"}, 12),
    ],
)
def test_independent_figure_paths(
    monkeypatch, tmp_path, accounting_config, selector, versions, count
):
    tables = bc.calculate(accounting_config)
    inputs = tmp_path / "input"
    inputs.mkdir()
    names = (
        ["anchor_enablement_bcr", "host_community_ratio"]
        if selector == "benefit-cost-a"
        else ["annual_four_case_comparison", "annual_incremental_bcr"]
        if selector == "benefit-cost-b"
        else list(tables)
    )
    for name in names:
        tables[name].to_csv(inputs / f"{name}.csv", index=False)
    saved = []
    monkeypatch.setattr(
        plt.Figure, "savefig", lambda _fig, path, **kwargs: saved.append(Path(path))
    )
    monkeypatch.setattr(
        api,
        "load_config",
        lambda _: {"output_dir": str(tmp_path / "figures"), "benefit_cost_tables": str(inputs)},
    )
    api.generate("test", selector)
    assert len(saved) == count
    assert {p.parent.name for p in saved} == versions
    assert {p.suffix for p in saved} == {".pdf", ".svg", ".PNG"}
    expected = {
        "version_a": {
            "Figure08_anchor_enablement_bcr_600dpi.PNG",
            "Figure09_host_community_value_ratio_600dpi.PNG",
        },
        "version_b": {
            "Figure08_annual_community_balance_600dpi.PNG",
            "Figure09_value_of_dc_presence_600dpi.PNG",
        },
    }
    assert {p.name for p in saved if p.suffix == ".PNG"} == set().union(
        *(expected[version] for version in versions)
    )
    assert plt.get_fignums() == []


def test_figure_reference_titles_follow_ledgers(monkeypatch, accounting_config):
    bc.setup_style()
    accounting_config["analysis"]["reference_event_interval_years"] = 20
    accounting_config["community"]["reference_share"] = 1
    tables = bc.calculate(accounting_config)
    titles = []

    def capture(fig, *args):
        titles.append(fig.axes[0].get_title())
        plt.close(fig)
        return {}

    monkeypatch.setattr(bc, "export_figure", capture)
    bc.figure_anchor_enablement(tables["anchor_enablement_bcr"])
    bc.figure_host_community(tables["host_community_ratio"])
    assert titles == ["(a) 20-year event interval", "(a) 100% community cost share"]

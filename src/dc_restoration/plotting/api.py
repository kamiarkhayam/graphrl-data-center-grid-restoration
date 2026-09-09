"""Configure paths for the preserved final figure functions."""

import json

from dc_restoration.configuration import load_config
from dc_restoration.labels import public_palette
from dc_restoration.paths import output_directory, require_file, resolve_path


def generate(config_path, figure):
    import pandas as pd

    config = load_config(config_path)
    output = output_directory(config["output_dir"])
    if figure == "system":
        from dc_restoration.plotting import schematic

        schematic.OUT = output / "system"
        schematic.PLOT_READY = resolve_path(config["geometry_plot_ready"])
        schematic.FIG1_IN = schematic.PLOT_READY / "main_text_figures" / "fig01_system_map"
        schematic.ensure_dirs()
        schematic.setup_style()
        schematic.make_figure1()
    elif figure == "severity":
        from dc_restoration.evaluation import trajectories

        severity = load_config(config["severity_config"])
        folders = [
            "figures/pdf",
            "figures/svg",
            "figures/png_600dpi",
            "figures/preview",
            "plot_ready",
            "validation",
        ]
        directories = {name: output_directory(output / "severity" / name) for name in folders}
        trajectories._setup_style()
        trajectories._figure2(severity, directories)
    elif figure in ("benefit-cost-a", "benefit-cost-b", "benefit-cost-all"):
        from dc_restoration.economics import benefit_cost as bc

        bc.OUT = output / "benefit_cost"
        inputs = resolve_path(config["benefit_cost_tables"])
        bc.setup_style()
        requested = []
        if figure in ("benefit-cost-a", "benefit-cost-all"):
            requested += [
                ("anchor_enablement_bcr", bc.figure_anchor_enablement),
                ("host_community_ratio", bc.figure_host_community),
            ]
        if figure in ("benefit-cost-b", "benefit-cost-all"):
            requested += [
                ("annual_four_case_comparison", bc.figure_annual_balance),
                ("annual_incremental_bcr", bc.figure_annual_incremental_bcr),
            ]
        # Validate every requested input before drawing the first figure.
        tables = [
            (pd.read_csv(require_file(inputs / f"{name}.csv")), draw) for name, draw in requested
        ]
        for table, draw in tables:
            draw(table)
    else:
        from dc_restoration.plotting import paper

        paper.REFRESH = resolve_path(config["paper_tables"])
        paper.OUT = output / "paper"
        paper.ensure_dirs()
        paper.setup_style()
        palette = public_palette(
            json.loads(require_file(config["palette"]).read_text(encoding="utf-8"))["methods"]
        )
        function = getattr(
            paper, "supplementary_" + figure if figure.startswith("s") else "main_figure" + figure
        )
        if figure in ("3", "4", "5", "s1", "s2", "s5"):
            function(palette)
        else:
            function()

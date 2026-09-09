from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dc_restoration.paths import repository_root

ROOT = repository_root()

REFRESH = ROOT / "outputs" / "analysis"

OUT = ROOT / "outputs" / "v0_3_paper_figures_complete_with_bcr_alternatives_v1"

GREEDY = "Greedy"

SELECTED = "GraphRL-A2C"

MAIN_SUBSETS = [
    "benchmark_hard_severe",
    "benchmark_hard_long_sequence",
    "benchmark_hard_critical",
    "benchmark_hard_anchor_relevant",
]

FRESH_HARD_SUBSETS = [
    "fresh_severe",
    "fresh_long_sequence",
    "fresh_critical",
    "fresh_anchor_relevant",
]

DISPLAY = {
    name: name
    for name in (
        "GraphRL-A2C",
        "GraphRL-PPO",
        "GraphBC",
        "MLP-A2C",
        "CNN-A2C",
        "Greedy",
        "Random Top-K",
    )
}


COMPETITIVE = ["GraphRL-A2C", "GraphRL-PPO", "GraphBC", "MLP-A2C", "CNN-A2C"]

RAW_BY_DISPLAY = {value: key for key, value in DISPLAY.items()}

SUBSET_LABEL = {
    "benchmark_hard_severe": "Severe-disruption subset",
    "benchmark_hard_long_sequence": "Long-sequence subset",
    "benchmark_hard_critical": "Critical-load subset",
    "benchmark_hard_anchor_relevant": "Anchor-relevant subset",
    "benchmark_hard_historical_nonzero": "Historical nonzero-damage subset",
    "benchmark_hard_stress_intensity_scaled": "Intensity-scaled sensitivity set",
    "fresh_historical_nonzero": "Historical nonzero-damage validation subset",
    "fresh_stress_intensity_scaled": "Intensity-scaled sensitivity set",
}

RESOURCE_ORDER = [
    "R0_no_dc_anchor",
    "E12_finite_energy_12h",
    "E24_finite_energy_24h",
    "R1_current_power_bounded_anchor",
]

RESOURCE_LABEL = {
    "R0_no_dc_anchor": "No anchor",
    "E12_finite_energy_12h": "E12",
    "E24_finite_energy_24h": "E24",
    "R1_current_power_bounded_anchor": "Power-only reference",
}

RESOURCE_COLOR = {
    "No anchor": "#7F7F7F",
    "E12": "#B89B4D",
    "E24": "#2A9D8F",
    "Power-only reference": "#1E6F3A",
}


def read_table(path, **kwargs):
    from dc_restoration.labels import public_display_columns

    return public_display_columns(pd.read_csv(path, **kwargs))


def setup_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def ensure_dirs() -> None:
    for family in [
        "main_text",
        "supplementary",
        "benefit_cost/version_A_lifecycle_and_community",
        "benefit_cost/version_B_annual_matrix",
    ]:
        for fmt in ["pdf", "svg", "png_600dpi", "preview"]:
            (OUT / family / fmt).mkdir(parents=True, exist_ok=True)
    for folder in [
        "captions",
        "plot_ready",
        "validation",
        "manuscript_png_all",
        "manuscript_drop_in_figs",
        "manuscript_drop_in_version_A_lifecycle_and_community",
        "manuscript_drop_in_version_B_annual_matrix",
    ]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)


def publish_manuscript_png(source: Path) -> None:
    """Publish the high-resolution PNG under the exact manuscript-style name."""
    shutil.copy2(source, OUT / "manuscript_png_all" / source.name)
    shutil.copy2(source, OUT / "manuscript_drop_in_figs" / source.name)


def style_axis(axis: plt.Axes, grid_axis: str = "x") -> None:
    # Preserve the full boxed-axis treatment used by the established paper figures.
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
    axis.grid(axis=grid_axis, color="#DCDCDC", lw=0.42, alpha=0.72)
    axis.set_axisbelow(True)


def panel(axis: plt.Axes, label: str, x: float = -0.11, y: float = 1.04) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
    )


def export(fig: plt.Figure, family: str, stem: str) -> dict[str, str]:
    destinations = {
        "pdf": OUT / family / "pdf" / f"{stem}.pdf",
        "svg": OUT / family / "svg" / f"{stem}.svg",
        "png_600dpi": OUT / family / "png_600dpi" / f"{stem}_600dpi.PNG",
        "preview": OUT / family / "preview" / f"{stem}_preview.png",
    }
    fig.savefig(destinations["pdf"], bbox_inches="tight")
    fig.savefig(destinations["svg"], bbox_inches="tight")
    fig.savefig(destinations["png_600dpi"], dpi=600, bbox_inches="tight")
    fig.savefig(destinations["preview"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    publish_manuscript_png(destinations["png_600dpi"])
    return {key: str(value) for key, value in destinations.items()}


def bootstrap_mean(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    spread = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return 100.0 * (center - spread), 100.0 * (center + spread)


def paired_reductions(
    data: pd.DataFrame,
    policy: str,
    metric: str,
    subsets: list[str] | None = None,
    unique_scenarios: bool = False,
) -> np.ndarray:
    part = data.copy()
    if subsets is not None:
        part = part[part["eval_subset"].isin(subsets)]
    keys = ["scenario_key"] if unique_scenarios else ["eval_subset", "scenario_key"]
    if unique_scenarios:
        part = part.drop_duplicates(["scenario_key", "policy"])
    base = part[part["policy"].eq(GREEDY)][keys + [metric]].rename(columns={metric: "baseline"})
    model = part[part["policy"].eq(policy)][keys + [metric]].rename(columns={metric: "model"})
    merged = model.merge(base, on=keys, how="inner")
    denom = merged["baseline"].to_numpy(float)
    values = np.full(denom.shape, np.nan, dtype=float)
    np.divide(
        denom - merged["model"].to_numpy(float),
        denom,
        out=values,
        where=np.abs(denom) > 1e-12,
    )
    values *= 100.0
    return values[np.isfinite(values)]


def support_reductions(
    data: pd.DataFrame,
    policy: str,
    resource: str,
    metric: str,
    subsets: list[str],
) -> np.ndarray:
    part = data[data["policy"].eq(policy) & data["eval_subset"].isin(subsets)]
    keys = ["eval_subset", "scenario_key"]
    base = part[part["dc_resource_configuration"].eq("R0_no_dc_anchor")][keys + [metric]].rename(
        columns={metric: "baseline"}
    )
    case = part[part["dc_resource_configuration"].eq(resource)][keys + [metric]].rename(
        columns={metric: "case"}
    )
    merged = case.merge(base, on=keys, how="inner")
    denom = merged["baseline"].to_numpy(float)
    values = np.full(denom.shape, np.nan, dtype=float)
    np.divide(
        denom - merged["case"].to_numpy(float),
        denom,
        out=values,
        where=np.abs(denom) > 1e-12,
    )
    values *= 100.0
    return values[np.isfinite(values)]


def main_figure3(pal: dict[str, dict[str, object]]) -> dict[str, str]:
    data = read_table(REFRESH / "controlled" / "main_hard_subset_average.csv")
    data["paper_policy"] = data["policy"].map(DISPLAY)
    order = COMPETITIVE
    data = data.set_index("paper_policy").reindex(order).reset_index()
    colors = [pal[name]["color"] for name in order]
    markers = [pal[name]["matplotlib_marker"] for name in order]
    fig = plt.figure(figsize=(7.1, 4.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], hspace=0.48, wspace=0.34)
    ax_a, ax_b, ax_c = (
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
    )
    y = np.arange(len(order))[::-1]
    values = data["mean_economic_improvement_percent"].to_numpy(float)
    ax_a.axvline(0, color="#222222", lw=0.7)
    for yi, value, color, marker in zip(y, values, colors, markers):
        ax_a.plot([0, value], [yi, yi], color=color, lw=1.25, alpha=0.75)
        ax_a.scatter(value, yi, color=color, marker=marker, s=42, zorder=3)
    ax_a.set_yticks(y, order)
    ax_a.set_xlabel("Economic-damage reduction vs Greedy (%)")
    style_axis(ax_a, "x")
    panel(ax_a, "(a)", -0.15)
    for index, row in data.iterrows():
        yi = y[index]
        marker = markers[index]
        ax_b.scatter(
            row["mean_ens_improvement_percent"],
            yi,
            facecolor=colors[index],
            edgecolor=colors[index],
            marker=marker,
            s=34,
            linewidth=0.8,
        )
        ax_b.scatter(
            row["mean_critical_ens_improvement_percent"],
            yi,
            facecolor="white",
            edgecolor=colors[index],
            marker=marker,
            s=34,
            linewidth=1.1,
        )
    ax_b.axvline(0, color="#222222", lw=0.7)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Reduction vs Greedy (%)")
    ax_b.legend(
        handles=[
            plt.Rectangle(
                (0, 0), 1, 1, facecolor="#555555", edgecolor="#555555", label="ENS (filled)"
            ),
            plt.Rectangle(
                (0, 0), 1, 1, facecolor="white", edgecolor="#555555", label="Critical ENS (open)"
            ),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    style_axis(ax_b, "x")
    panel(ax_b, "(b)", -0.12)
    ax_c.barh(
        y, 100.0 * data["mean_win_rate"].to_numpy(float), color=colors, alpha=0.82, height=0.64
    )
    ax_c.set_xlim(0, 100)
    ax_c.set_yticks([])
    ax_c.set_xlabel("Scenario win rate vs Greedy (%)")
    style_axis(ax_c, "x")
    panel(ax_c, "(c)", -0.12)
    for axis in [ax_a, ax_b, ax_c]:
        axis.set_ylim(-0.6, len(order) - 0.4)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.14, top=0.94)
    data.to_csv(
        OUT / "plot_ready" / "Figure03_comparative_restoration_performance.csv", index=False
    )
    return export(fig, "main_text", "Figure03_comparative_restoration_performance")


def main_figure4(pal: dict[str, dict[str, object]]) -> dict[str, str]:
    data = read_table(REFRESH / "bounded_ga" / "bounded_ga_comparison_by_scenario.csv")
    summary = read_table(REFRESH / "bounded_ga" / "bounded_ga_summary.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.75, 3.85), gridspec_kw={"wspace": 0.36})
    x = data["bounded_ga_improvement_vs_greedy_percent"].to_numpy(float)
    y = data["selected_improvement_vs_greedy_percent"].to_numpy(float)
    lo = min(np.nanmin(x), np.nanmin(y), 0.0)
    hi = max(np.nanmax(x), np.nanmax(y)) * 1.04
    axes[0].plot([lo, hi], [lo, hi], color="#8E8E8E", lw=0.9, zorder=1)
    axes[0].scatter(
        x,
        y,
        s=38,
        color=pal["GraphRL-A2C"]["color"],
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    axes[0].set_xlim(lo - 0.2, hi)
    axes[0].set_ylim(lo - 0.2, hi)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Bounded GA reduction vs Greedy (%)")
    axes[0].set_ylabel("GraphRL-A2C reduction vs Greedy (%)")
    style_axis(axes[0], "both")
    panel(axes[0], "(a)", -0.075)
    rng = np.random.default_rng(32)
    runtime = [
        data["selected_runtime_seconds"].to_numpy(float),
        data["runtime_batch_elapsed_seconds"].to_numpy(float),
    ]
    boxes = axes[1].boxplot(
        runtime,
        orientation="horizontal",
        positions=[1, 0],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        medianprops={"color": "white", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(
        boxes["boxes"], [pal["GraphRL-A2C"]["color"], pal["Bounded GA"]["color"]]
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    for pos, values, color in zip(
        [1, 0], runtime, [pal["GraphRL-A2C"]["color"], pal["Bounded GA"]["color"]]
    ):
        axes[1].scatter(
            values,
            pos + rng.uniform(-0.055, 0.055, len(values)),
            s=17,
            color=color,
            alpha=0.42,
            edgecolor="white",
            linewidth=0.25,
            zorder=5,
        )
    axes[1].set_yticks([1, 0], ["GraphRL-A2C", "Bounded GA"])
    axes[1].set_xlabel("Computation time per scenario (s)")
    axes[1].set_title("Computational demand", fontweight="normal", pad=8)
    max_runtime = float(np.nanmax(np.concatenate(runtime)))
    axes[1].set_xlim(0, max_runtime * 1.16)
    axes[1].set_ylim(-0.72, 1.45)
    axes[1].text(
        0.98,
        0.96,
        f"Bounded GA budget\nPopulation: {int(summary['population'])}\nGenerations: {int(summary['generations'])}\nMean fitness evals: {summary['mean_bounded_ga_fitness_evaluations']:.1f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "linewidth": 0.5, "alpha": 0.9},
    )
    style_axis(axes[1], "x")
    panel(axes[1], "(b)", -0.075)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.90)
    data.to_csv(OUT / "plot_ready" / "Figure04_GraphRL_A2C_vs_bounded_GA.csv", index=False)
    return export(fig, "main_text", "Figure04_GraphRL_A2C_vs_bounded_GA_final")


def aggregate_support_point(
    benefit: pd.DataFrame,
    policy: str,
    subsets: list[str],
    resource: str,
    column: str,
) -> float:
    rows = benefit[
        benefit["policy"].eq(policy)
        & benefit["eval_subset"].isin(subsets)
        & benefit["dc_resource_configuration"].eq(resource)
    ]
    return float(rows[column].mean())


def main_figure5(pal: dict[str, dict[str, object]]) -> dict[str, str]:
    benefit = read_table(REFRESH / "dc_anchor" / "dc_benefit_vs_no_anchor.csv")
    episodes = read_table(REFRESH / "dc_anchor" / "dc_results_by_episode.csv")
    resources = RESOURCE_ORDER
    labels = [RESOURCE_LABEL[r] for r in resources]
    x = np.arange(len(resources))
    fig, axes = plt.subplots(
        2, 2, figsize=(8.0, 5.25), gridspec_kw={"wspace": 0.34, "hspace": 0.46}
    )
    plotting_rows: list[dict[str, object]] = []
    for axis, subsets, title, lab in [
        (axes[0, 0], MAIN_SUBSETS, "Controlled hard subsets", "(a)"),
        (
            axes[0, 1],
            ["benchmark_hard_historical_nonzero"],
            "Historical nonzero-damage subset",
            "(b)",
        ),
    ]:
        for policy_name, offset, marker, linestyle in [
            (SELECTED, -0.055, "o", "-"),
            (GREEDY, 0.055, "s", "--"),
        ]:
            vals, lows, highs = [0.0], [0.0], [0.0]
            for index, resource in enumerate(resources[1:], start=1):
                value = aggregate_support_point(
                    benefit,
                    policy_name,
                    subsets,
                    resource,
                    "mean_economic_damage_reduction_vs_no_anchor_percent",
                )
                sample = support_reductions(
                    episodes, policy_name, resource, "harmonized_economic_damage_dollars", subsets
                )
                low, high = bootstrap_mean(sample, 500 + index + len(plotting_rows))
                vals.append(value)
                lows.append(value - low)
                highs.append(high - value)
                plotting_rows.append(
                    {
                        "panel": title,
                        "policy": DISPLAY[policy_name],
                        "resource": RESOURCE_LABEL[resource],
                        "metric": "Economic damage",
                        "value": value,
                        "lower_ci": low,
                        "upper_ci": high,
                    }
                )
            color = pal[DISPLAY[policy_name]]["color"]
            axis.errorbar(
                x + offset,
                vals,
                yerr=[lows, highs],
                color=color,
                marker=marker,
                ls=linestyle,
                lw=1.25,
                ms=4.5,
                capsize=2,
                label=DISPLAY[policy_name],
            )
        axis.axhline(0, color="#555555", lw=0.7)
        axis.set_xticks(x, labels)
        axis.set_ylabel("Economic-damage reduction\nrelative to no anchor (%)")
        axis.set_title(title, fontweight="normal")
        style_axis(axis, "y")
        panel(axis, lab, -0.09)
    axes[0, 0].legend(loc="lower center", bbox_to_anchor=(1.12, 1.14), frameon=False, ncol=2)
    for metric, marker, color in [
        ("mean_ens_reduction_vs_no_anchor_percent", "o", "#377A9F"),
        ("mean_critical_ens_reduction_vs_no_anchor_percent", "D", "#7A5195"),
    ]:
        values, lows, highs = [0.0], [0.0], [0.0]
        source_metric = (
            "energy_not_served_mwh"
            if "critical" not in metric
            else "critical_energy_not_served_mwh"
        )
        for index, resource in enumerate(resources[1:], start=1):
            value = aggregate_support_point(benefit, SELECTED, MAIN_SUBSETS, resource, metric)
            sample = support_reductions(episodes, SELECTED, resource, source_metric, MAIN_SUBSETS)
            low, high = bootstrap_mean(sample, 700 + index + len(plotting_rows))
            values.append(value)
            lows.append(value - low)
            highs.append(high - value)
            plotting_rows.append(
                {
                    "panel": "Physical outage reductions",
                    "policy": "GraphRL-A2C",
                    "resource": RESOURCE_LABEL[resource],
                    "metric": metric,
                    "value": value,
                    "lower_ci": low,
                    "upper_ci": high,
                }
            )
        label = "ENS" if "critical" not in metric else "Critical-load ENS"
        axes[1, 0].errorbar(
            x,
            values,
            yerr=[lows, highs],
            color=color,
            marker=marker,
            ls="-" if label == "ENS" else "--",
            lw=1.2,
            ms=4.5,
            capsize=2,
            label=label,
        )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Reduction relative to no anchor (%)")
    axes[1, 0].set_title("Physical outage reductions under GraphRL-A2C", fontweight="normal")
    axes[1, 0].legend(frameon=False, loc="upper left")
    style_axis(axes[1, 0], "y")
    panel(axes[1, 0], "(c)", -0.09)
    dep_resources = resources[1:3]
    dep_vals, dep_low, dep_high = [], [], []
    for index, resource in enumerate(dep_resources):
        part = episodes[
            episodes["policy"].eq(SELECTED)
            & episodes["eval_subset"].isin(MAIN_SUBSETS)
            & episodes["dc_resource_configuration"].eq(resource)
        ]
        successes = int(part["energy_depleted"].astype(str).str.lower().eq("true").sum())
        value = 100.0 * successes / len(part)
        low, high = wilson(successes, len(part))
        dep_vals.append(value)
        dep_low.append(value - low)
        dep_high.append(high - value)
        plotting_rows.append(
            {
                "panel": "Energy-budget depletion",
                "policy": "GraphRL-A2C",
                "resource": RESOURCE_LABEL[resource],
                "metric": "Episodes depleted",
                "value": value,
                "lower_ci": low,
                "upper_ci": high,
            }
        )
    axes[1, 1].bar(
        np.arange(2),
        dep_vals,
        color=[RESOURCE_COLOR["E12"], RESOURCE_COLOR["E24"]],
        width=0.58,
        alpha=0.86,
    )
    axes[1, 1].errorbar(
        np.arange(2),
        dep_vals,
        yerr=[dep_low, dep_high],
        fmt="none",
        ecolor="#333333",
        capsize=2,
        lw=0.9,
    )
    axes[1, 1].set_xticks(np.arange(2), ["E12", "E24"])
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_ylabel("Episodes with depleted\nenergy budget (%)")
    axes[1, 1].set_title("Energy-budget depletion under GraphRL-A2C", fontweight="normal")
    style_axis(axes[1, 1], "y")
    panel(axes[1, 1], "(d)", -0.09)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.88)
    pd.DataFrame(plotting_rows).to_csv(
        OUT / "plot_ready" / "Figure05_anchor_resource_response.csv", index=False
    )
    return export(fig, "main_text", "Figure05_anchor_resource_response_and_energy_depletion")


def unique_hard_cohort_dc_curves(traj: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Build hourly curves against each scenario's common no-anchor interruption."""
    traj = traj[traj["scenario_key"].isin(keys) & traj["policy"].eq(SELECTED)].copy()
    no_anchor_initial = (
        traj[traj["dc_resource_configuration"].eq("R0_no_dc_anchor")]
        .sort_values("step_index")
        .groupby("scenario_key", as_index=True)
        .first()[["initial_unserved_mw", "initial_critical_unserved_mw"]]
    )
    horizon = float(np.ceil(traj["cumulative_time_hours"].max()))
    grid = np.arange(0.0, horizon + 1.0, 1.0)
    sampled: list[dict[str, object]] = []
    for (scenario_key, resource), group in traj.groupby(
        ["scenario_key", "dc_resource_configuration"], sort=False
    ):
        group = group.sort_values("interval_start_time_hours")
        starts = group["interval_start_time_hours"].to_numpy(float)
        final_time = float(group["cumulative_time_hours"].max())
        final_ens = float(group["cumulative_ens_mwh"].iloc[-1])
        initial_unserved = float(no_anchor_initial.at[scenario_key, "initial_unserved_mw"])
        initial_critical = float(no_anchor_initial.at[scenario_key, "initial_critical_unserved_mw"])
        for point in grid:
            index = max(int(np.searchsorted(starts, point, side="right") - 1), 0)
            row = group.iloc[index]
            if point >= final_time:
                restored_total = 1.0
                restored_critical = (
                    1.0 if float(row["initial_critical_unserved_mw"]) > 1e-9 else np.nan
                )
                cumulative_ens = final_ens
            else:
                restored_total = float(
                    np.clip(
                        1.0 - float(row["current_unserved_mw"]) / max(initial_unserved, 1e-9),
                        0.0,
                        1.0,
                    )
                )
                restored_critical = (
                    float(
                        np.clip(
                            1.0 - float(row["current_critical_unserved_mw"]) / initial_critical,
                            0.0,
                            1.0,
                        )
                    )
                    if initial_critical > 1e-9
                    else np.nan
                )
                interval_ens_start = float(
                    row["cumulative_ens_mwh"] - row["step_unserved_energy_mwh"]
                )
                elapsed = np.clip(
                    point - float(row["interval_start_time_hours"]),
                    0.0,
                    float(row["interval_duration_hours"]),
                )
                cumulative_ens = interval_ens_start + float(row["current_unserved_mw"]) * elapsed
            sampled.append(
                {
                    "scenario_key": scenario_key,
                    "dc_resource_configuration": resource,
                    "time_hours": point,
                    "initially_interrupted_load_restored_fraction": restored_total,
                    "initially_interrupted_critical_load_restored_fraction": restored_critical,
                    "cumulative_ens_mwh": cumulative_ens,
                }
            )

    sampled_frame = pd.DataFrame(sampled)
    rows: list[dict[str, object]] = []
    value_columns = [
        "initially_interrupted_load_restored_fraction",
        "initially_interrupted_critical_load_restored_fraction",
        "cumulative_ens_mwh",
    ]
    for (resource, point), group in sampled_frame.groupby(
        ["dc_resource_configuration", "time_hours"], sort=False
    ):
        record: dict[str, object] = {
            "dc_resource_configuration": resource,
            "time_hours": point,
            "episode_count": int(group["scenario_key"].nunique()),
            "restoration_fraction_denominator": "Scenario-specific initial no-anchor interruption",
        }
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            mean = float(values.mean()) if len(values) else np.nan
            record[f"mean_{column}"] = mean
            record[f"q25_{column}"] = float(values.quantile(0.25)) if len(values) else np.nan
            record[f"q75_{column}"] = float(values.quantile(0.75)) if len(values) else np.nan
            if len(values) > 1:
                half_width = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
                lower, upper = mean - half_width, mean + half_width
                if "restored_fraction" in column:
                    lower, upper = max(0.0, lower), min(1.0, upper)
            else:
                lower, upper = mean, mean
            record[f"mean_95ci_low_{column}"] = lower
            record[f"mean_95ci_high_{column}"] = upper
        rows.append(record)
    return pd.DataFrame(rows)


def t90_reduction_by_resource(traj: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    traj = traj[traj["scenario_key"].isin(keys) & traj["policy"].eq(SELECTED)]
    no_anchor_initial = (
        traj[traj["dc_resource_configuration"].eq("R0_no_dc_anchor")]
        .sort_values("step_index")
        .groupby("scenario_key", as_index=True)
        .first()["initial_unserved_mw"]
    )
    times = []
    for (scenario_key, resource), group in traj.groupby(
        ["scenario_key", "dc_resource_configuration"]
    ):
        group = group.sort_values("interval_start_time_hours")
        initial_unserved = float(no_anchor_initial.at[scenario_key])
        restored_fraction = 1.0 - group["current_unserved_mw"].astype(float) / max(
            initial_unserved, 1e-9
        )
        hit = group[restored_fraction.ge(0.90)]
        # The recorded service state applies at interval start; if the final repair
        # first reaches 90%, the passage time is the episode's final interval end.
        t90 = (
            float(hit["interval_start_time_hours"].iloc[0])
            if len(hit)
            else float(group["cumulative_time_hours"].max())
        )
        times.append({"scenario_key": scenario_key, "resource": resource, "t90": t90})
    wide = pd.DataFrame(times).pivot(index="scenario_key", columns="resource", values="t90")
    rows = []
    for resource in RESOURCE_ORDER[1:]:
        values = (
            ((wide["R0_no_dc_anchor"] - wide[resource]) / wide["R0_no_dc_anchor"] * 100.0)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(float)
        )
        rows.append(
            {
                "resource": resource,
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "mean": float(np.mean(values)),
                "episode_count": len(values),
            }
        )
    return pd.DataFrame(rows)


def main_figure6() -> dict[str, str]:
    traj = read_table(REFRESH / "dc_anchor" / "dc_trajectory_unique_by_step.csv")
    controlled = read_table(
        REFRESH / "controlled" / "results_by_episode.csv",
        usecols=["eval_subset", "scenario_key"],
    )
    keys = controlled[controlled["eval_subset"].isin(MAIN_SUBSETS)][
        "scenario_key"
    ].drop_duplicates()
    aggregate = unique_hard_cohort_dc_curves(traj, keys)
    aggregate = aggregate.sort_values(["dc_resource_configuration", "time_hours"]).reset_index(
        drop=True
    )
    restoration_columns = [
        "mean_initially_interrupted_load_restored_fraction",
        "mean_initially_interrupted_critical_load_restored_fraction",
    ]
    for column in restoration_columns:
        display_column = f"display_{column}_7h_centered_mean"

        def smooth_for_display(values: pd.Series) -> pd.Series:
            smoothed = values.rolling(window=7, center=True, min_periods=1).mean()
            edge = min(3, len(values))
            smoothed.iloc[:edge] = values.iloc[:edge]
            smoothed.iloc[-edge:] = values.iloc[-edge:]
            return smoothed.clip(0.0, 1.0)

        aggregate[display_column] = aggregate.groupby("dc_resource_configuration", sort=False)[
            column
        ].transform(smooth_for_display)
    t90 = t90_reduction_by_resource(traj, keys)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), gridspec_kw={"wspace": 0.34, "hspace": 0.43})
    metrics = [
        (
            axes[0, 0],
            "display_mean_initially_interrupted_load_restored_fraction_7h_centered_mean",
            "Total interrupted load receiving service",
            "Fraction receiving service",
            110.0,
        ),
        (
            axes[0, 1],
            "display_mean_initially_interrupted_critical_load_restored_fraction_7h_centered_mean",
            "Critical interrupted load receiving service",
            "Fraction receiving service",
            110.0,
        ),
        (
            axes[1, 0],
            "mean_cumulative_ens_mwh",
            "Cumulative energy not served",
            "Cumulative ENS (MWh)",
            None,
        ),
    ]
    for axis, column, title, ylabel, xmax in metrics:
        for resource in RESOURCE_ORDER:
            part = aggregate[aggregate["dc_resource_configuration"].eq(resource)].sort_values(
                "time_hours"
            )
            if xmax is not None:
                part = part[part["time_hours"].le(xmax)]
            label = RESOURCE_LABEL[resource]
            axis.plot(
                part["time_hours"],
                part[column],
                color=RESOURCE_COLOR[label],
                ls="--" if label == "Power-only reference" else "-",
                lw=1.35,
                label=label,
            )
        axis.set_title(title, fontweight="normal")
        axis.set_xlabel("Restoration time (h)")
        axis.set_ylabel(ylabel)
        style_axis(axis, "both")
    panel(axes[0, 0], "(a)", -0.09)
    panel(axes[0, 1], "(b)", -0.09)
    panel(axes[1, 0], "(c)", -0.09)
    axes[0, 0].legend(loc="lower center", bbox_to_anchor=(1.13, 1.17), ncol=4, frameon=False)
    x = np.arange(3)
    medians = t90["median"].to_numpy(float)
    axes[1, 1].bar(
        x,
        medians,
        color=[
            RESOURCE_COLOR["E12"],
            RESOURCE_COLOR["E24"],
            RESOURCE_COLOR["Power-only reference"],
        ],
        width=0.58,
    )
    axes[1, 1].errorbar(
        x,
        medians,
        yerr=[medians - t90["q25"].to_numpy(float), t90["q75"].to_numpy(float) - medians],
        fmt="none",
        ecolor="#333333",
        capsize=2,
        lw=0.9,
    )
    axes[1, 1].set_xticks(x, ["E12", "E24", "Power-only reference"])
    axes[1, 1].set_ylabel("Median reduction in T90\nrelative to no anchor (%)")
    axes[1, 1].set_title("Recovery-time improvement", fontweight="normal")
    style_axis(axes[1, 1], "y")
    panel(axes[1, 1], "(d)", -0.09)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.87)
    aggregate.to_csv(
        OUT / "plot_ready" / "Figure06_anchor_supported_restoration_curves.csv", index=False
    )
    t90.to_csv(OUT / "plot_ready" / "Figure06_t90_reduction.csv", index=False)
    return export(fig, "main_text", "Figure06_anchor_supported_restoration_and_recovery_timing")


def fresh_pair_summary(data: pd.DataFrame, label: str, unique: bool) -> pd.DataFrame:
    rows = []
    for metric_label, metric in [
        ("Economic interruption damage", "harmonized_economic_damage_dollars"),
        ("ENS", "unserved_mwh"),
        ("Critical-load ENS", "critical_unserved_mwh"),
    ]:
        values = paired_reductions(data, SELECTED, metric, unique_scenarios=unique)
        low, high = bootstrap_mean(values, 1100 + len(rows) + (10 if unique else 0))
        rows.append(
            {
                "validation_set": label,
                "metric": metric_label,
                "mean_reduction_percent": float(np.mean(values)),
                "lower_ci": low,
                "upper_ci": high,
                "episode_count": len(values),
            }
        )
    economic = paired_reductions(
        data, SELECTED, "harmonized_economic_damage_dollars", unique_scenarios=unique
    )
    wins = int(np.sum(economic > 1e-9))
    ties = int(np.sum(np.abs(economic) <= 1e-9))
    low, high = wilson(wins, len(economic))
    rows.append(
        {
            "validation_set": label,
            "metric": "Win rate",
            "mean_reduction_percent": 100.0 * wins / len(economic),
            "lower_ci": low,
            "upper_ci": high,
            "episode_count": len(economic),
            "tie_count": ties,
        }
    )
    return pd.DataFrame(rows)


def main_figure7() -> dict[str, str]:
    unique = read_table(REFRESH / "fresh" / "unique_bank_results_by_episode.csv")
    fresh = read_table(REFRESH / "fresh" / "results_by_episode.csv")
    historical = fresh[fresh["eval_subset"].eq("fresh_historical_nonzero")]
    summary = pd.concat(
        [
            fresh_pair_summary(unique, "Unique fresh bank", True),
            fresh_pair_summary(historical, "Historical nonzero-damage", False),
        ],
        ignore_index=True,
    )
    colors = {"Unique fresh bank": "#0C457D", "Historical nonzero-damage": "#2A9D8F"}
    markers = {"Unique fresh bank": "o", "Historical nonzero-damage": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"wspace": 0.50})
    outcomes = ["Economic interruption damage", "ENS", "Critical-load ENS"]
    y = np.arange(3)[::-1]
    for set_index, validation_set in enumerate(colors):
        part = (
            summary[summary["validation_set"].eq(validation_set) & summary["metric"].isin(outcomes)]
            .set_index("metric")
            .reindex(outcomes)
        )
        vals = part["mean_reduction_percent"].to_numpy(float)
        axes[0].errorbar(
            vals,
            y + (set_index - 0.5) * 0.12,
            xerr=[vals - part["lower_ci"].to_numpy(float), part["upper_ci"].to_numpy(float) - vals],
            fmt=markers[validation_set],
            color=colors[validation_set],
            lw=1.0,
            capsize=2,
            ms=4.8,
            label=validation_set,
        )
    axes[0].axvline(0, color="#777777", lw=0.8)
    axes[0].set_yticks(y, ["Economic damage", "ENS", "Critical-load ENS"])
    axes[0].set_xlabel("Reduction relative to Greedy (%)")
    axes[0].set_title("Outcome reductions", fontweight="normal")
    style_axis(axes[0], "x")
    panel(axes[0], "(a)", -0.10)
    win_rows = summary[summary["metric"].eq("Win rate")]
    wy = np.asarray([0.58, 0.42])
    for yi, (_, row) in zip(wy, win_rows.iterrows()):
        value = float(row["mean_reduction_percent"])
        axes[1].errorbar(
            value,
            yi,
            xerr=[[value - float(row["lower_ci"])], [float(row["upper_ci"]) - value]],
            fmt=markers[row["validation_set"]],
            color=colors[row["validation_set"]],
            lw=1.0,
            capsize=2,
            ms=4.8,
        )
        axes[1].annotate(
            f"{value:.1f}%",
            xy=(value, yi),
            xytext=(-5, -8),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7.0,
            color=colors[row["validation_set"]],
            clip_on=True,
        )
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0.25, 0.75)
    axes[1].set_yticks(wy, ["Unique fresh\nbank", "Historical\nnonzero-damage"])
    axes[1].set_xlabel("Scenarios with lower economic damage (%)")
    axes[1].set_title("Scenario-level win rate", fontweight="normal")
    style_axis(axes[1], "x")
    panel(axes[1], "(b)", -0.10)
    fig.legend(loc="lower center", bbox_to_anchor=(0.51, 0.97), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.82)
    summary.to_csv(OUT / "plot_ready" / "Figure07_policy_validation.csv", index=False)
    return export(fig, "main_text", "Figure07_policy_validation_previously_unseen_damage")


def supplementary_s1(pal: dict[str, dict[str, object]]) -> dict[str, str]:
    source = read_table(REFRESH / "controlled" / "pairwise_vs_greedy.csv")
    source["paper_policy"] = source["policy"].map(DISPLAY)
    categories = MAIN_SUBSETS + [
        "benchmark_hard_historical_nonzero",
        "benchmark_hard_stress_intensity_scaled",
    ]
    plot_data = source[
        source["paper_policy"].isin(COMPETITIVE) & source["eval_subset"].isin(categories)
    ].copy()
    plot_data.to_csv(
        OUT / "plot_ready" / "FigureS01_controlled_subset_policy_comparison.csv", index=False
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.08, 1.06], "wspace": 0.78}
    )
    groups = [
        (MAIN_SUBSETS, "Disruption-characteristic subsets"),
        (categories[4:], "Historical and sensitivity sets"),
    ]
    offsets = np.linspace(-0.24, 0.24, len(COMPETITIVE))
    xmin = float(plot_data["economic_bootstrap_95_low_percent"].min()) - 1.0
    xmax = float(plot_data["economic_bootstrap_95_high_percent"].max()) + 1.0
    for axis, (subsets, title), lab in zip(axes, groups, ["(a)", "(b)"]):
        ybase = np.arange(len(subsets))[::-1]
        axis.axvline(0, color="#777777", lw=0.8)
        for index, policy_name in enumerate(COMPETITIVE):
            rows = plot_data[plot_data["paper_policy"].eq(policy_name)].set_index("eval_subset")
            vals, ys, low, high = [], [], [], []
            for j, subset in enumerate(subsets):
                row = rows.loc[subset]
                value = float(row["mean_economic_improvement_percent"])
                vals.append(value)
                ys.append(ybase[j] + offsets[index])
                low.append(value - float(row["economic_bootstrap_95_low_percent"]))
                high.append(float(row["economic_bootstrap_95_high_percent"]) - value)
            axis.errorbar(
                vals,
                ys,
                xerr=[low, high],
                fmt=pal[policy_name]["matplotlib_marker"],
                color=pal[policy_name]["color"],
                ms=5.0 if policy_name == "GraphRL-A2C" else 4.3,
                lw=1.0,
                capsize=2,
                label=policy_name,
            )
        axis.set_yticks(ybase, [SUBSET_LABEL[s] for s in subsets])
        axis.set_xlim(xmin, xmax)
        axis.set_title(title, fontweight="normal")
        style_axis(axis, "x")
        panel(axis, lab)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, 1.12), ncol=3, frameon=False)
    fig.supxlabel("Economic-damage reduction relative to Greedy (%)", y=0.055, fontsize=8.0)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.80, bottom=0.22)
    return export(fig, "supplementary", "FigureS01_controlled_subset_policy_comparison")


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    return values, np.arange(1, len(values) + 1) / len(values) if len(values) else np.array([])


def supplementary_s2(pal: dict[str, dict[str, object]]) -> dict[str, str]:
    controlled = read_table(REFRESH / "controlled" / "results_by_episode.csv")
    groups = [
        (
            "Controlled hard cohort",
            controlled[controlled["eval_subset"].isin(MAIN_SUBSETS)].drop_duplicates(
                ["scenario_key", "policy"]
            ),
            True,
        ),
        (
            "Historical nonzero-damage subset",
            controlled[controlled["eval_subset"].eq("benchmark_hard_historical_nonzero")],
            False,
        ),
        (
            "Intensity-scaled sensitivity set",
            controlled[controlled["eval_subset"].eq("benchmark_hard_stress_intensity_scaled")],
            False,
        ),
    ]
    rows = []
    for group_name, data, unique in groups:
        for policy_name in [RAW_BY_DISPLAY[name] for name in COMPETITIVE]:
            values = paired_reductions(
                data, policy_name, "harmonized_economic_damage_dollars", unique_scenarios=unique
            )
            for value in values:
                rows.append(
                    {
                        "scenario_set": group_name,
                        "policy": DISPLAY[policy_name],
                        "economic_damage_reduction_vs_greedy_percent": value,
                    }
                )
    plot_data = pd.DataFrame(rows)
    plot_data.to_csv(
        OUT / "plot_ready" / "FigureS02_scenario_level_policy_improvement_distributions.csv",
        index=False,
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True, gridspec_kw={"wspace": 0.20})
    xmin = float(plot_data["economic_damage_reduction_vs_greedy_percent"].min()) - 1.0
    xmax = float(plot_data["economic_damage_reduction_vs_greedy_percent"].max()) + 1.0
    for axis, (group_name, _, _), lab in zip(axes, groups, ["(a)", "(b)", "(c)"]):
        axis.axvline(0, color="#777777", lw=0.8)
        for policy_name in COMPETITIVE:
            values = plot_data[
                plot_data["scenario_set"].eq(group_name) & plot_data["policy"].eq(policy_name)
            ]["economic_damage_reduction_vs_greedy_percent"].to_numpy(float)
            xs, ys = ecdf(values)
            axis.step(
                xs,
                ys,
                where="post",
                color=pal[policy_name]["color"],
                lw=1.9 if policy_name == "GraphRL-A2C" else 1.15,
                ls=pal[policy_name]["matplotlib_line_style"],
                label=policy_name,
            )
        axis.set_title(group_name, fontweight="normal")
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(0, 1.02)
        style_axis(axis, "both")
        panel(axis, lab)
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)
    fig.supxlabel("Economic-damage reduction relative to Greedy (%)", y=0.055, fontsize=8.0)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.76)
    return export(fig, "supplementary", "FigureS02_scenario_level_policy_improvement_distributions")


def supplementary_s3() -> dict[str, str]:
    episodes = read_table(REFRESH / "dc_anchor" / "dc_results_by_episode.csv")
    categories = MAIN_SUBSETS + [
        "benchmark_hard_historical_nonzero",
        "benchmark_hard_stress_intensity_scaled",
    ]
    resource_plot = RESOURCE_ORDER[1:]
    metrics = [
        ("Economic damage", "harmonized_economic_damage_dollars"),
        ("ENS", "energy_not_served_mwh"),
        ("Critical-load ENS", "critical_energy_not_served_mwh"),
    ]
    rows = []
    for subset_index, subset in enumerate(categories):
        for resource_index, resource in enumerate(resource_plot):
            for metric_name, metric in metrics:
                values = support_reductions(episodes, SELECTED, resource, metric, [subset])
                low, high = bootstrap_mean(
                    values, 1300 + subset_index * 20 + resource_index * 4 + len(rows)
                )
                rows.append(
                    {
                        "subset": subset,
                        "resource": RESOURCE_LABEL[resource],
                        "metric": metric_name,
                        "value": float(np.mean(values)),
                        "lower_ci": low,
                        "upper_ci": high,
                        "episode_count": len(values),
                    }
                )
        for resource in RESOURCE_ORDER[1:3]:
            part = episodes[
                episodes["policy"].eq(SELECTED)
                & episodes["eval_subset"].eq(subset)
                & episodes["dc_resource_configuration"].eq(resource)
            ]
            successes = int(part["energy_depleted"].astype(str).str.lower().eq("true").sum())
            low, high = wilson(successes, len(part))
            rows.append(
                {
                    "subset": subset,
                    "resource": RESOURCE_LABEL[resource],
                    "metric": "Energy-budget depletion",
                    "value": 100.0 * successes / len(part),
                    "lower_ci": low,
                    "upper_ci": high,
                    "episode_count": len(part),
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(
        OUT / "plot_ready" / "FigureS03_extended_anchor_resource_sensitivity.csv", index=False
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.4), gridspec_kw={"wspace": 0.33, "hspace": 0.36})
    panel_specs = [
        ("Economic damage", "Economic-damage reduction", "Reduction relative to no anchor (%)"),
        ("ENS", "ENS reduction", "Reduction relative to no anchor (%)"),
        ("Critical-load ENS", "Critical-load ENS reduction", "Reduction relative to no anchor (%)"),
        (
            "Energy-budget depletion",
            "Energy-budget depletion",
            "Episodes with depleted energy budget (%)",
        ),
    ]
    offsets = {"E12": -0.18, "E24": 0.0, "Power-only reference": 0.18}
    ybase = np.arange(len(categories))[::-1]
    for axis, (metric, title, xlabel), lab in zip(
        axes.ravel(), panel_specs, ["(a)", "(b)", "(c)", "(d)"]
    ):
        axis.axvline(0, color="#777777", lw=0.8)
        resources = (
            ["E12", "E24"]
            if metric == "Energy-budget depletion"
            else ["E12", "E24", "Power-only reference"]
        )
        local_offsets = (
            {"E12": -0.10, "E24": 0.10} if metric == "Energy-budget depletion" else offsets
        )
        for resource in resources:
            part = data[data["metric"].eq(metric) & data["resource"].eq(resource)].set_index(
                "subset"
            )
            vals = np.array([float(part.loc[s, "value"]) for s in categories])
            low = vals - np.array([float(part.loc[s, "lower_ci"]) for s in categories])
            high = np.array([float(part.loc[s, "upper_ci"]) for s in categories]) - vals
            axis.errorbar(
                vals,
                ybase + local_offsets[resource],
                xerr=[low, high],
                fmt="o",
                color=RESOURCE_COLOR[resource],
                ms=4.6,
                lw=1.0,
                capsize=2,
                label=resource,
            )
        axis.set_yticks(ybase, [SUBSET_LABEL[s] for s in categories])
        if axis in [axes[0, 1], axes[1, 1]]:
            axis.set_yticklabels([])
        axis.set_title(title, fontweight="normal")
        axis.set_xlabel(xlabel)
        style_axis(axis, "x")
        panel(axis, lab)
    axes[0, 0].legend(loc="lower center", bbox_to_anchor=(1.13, 1.20), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.87, bottom=0.10)
    return export(fig, "supplementary", "FigureS03_extended_anchor_resource_sensitivity")


def supplementary_s4() -> dict[str, str]:
    traj = read_table(REFRESH / "dc_anchor" / "dc_trajectory_unique_by_step.csv")
    controlled = read_table(
        REFRESH / "controlled" / "results_by_episode.csv",
        usecols=["eval_subset", "scenario_key"],
    )
    keys = controlled[controlled["eval_subset"].isin(MAIN_SUBSETS)][
        "scenario_key"
    ].drop_duplicates()
    data = unique_hard_cohort_dc_curves(traj, keys).rename(
        columns={
            "dc_resource_configuration": "resource",
            "mean_initially_interrupted_load_restored_fraction": "mean_total",
            "q25_initially_interrupted_load_restored_fraction": "q25_total",
            "q75_initially_interrupted_load_restored_fraction": "q75_total",
            "mean_95ci_low_initially_interrupted_load_restored_fraction": "mean_95ci_low_total",
            "mean_95ci_high_initially_interrupted_load_restored_fraction": "mean_95ci_high_total",
            "mean_initially_interrupted_critical_load_restored_fraction": "mean_critical",
            "q25_initially_interrupted_critical_load_restored_fraction": "q25_critical",
            "q75_initially_interrupted_critical_load_restored_fraction": "q75_critical",
            "mean_95ci_low_initially_interrupted_critical_load_restored_fraction": "mean_95ci_low_critical",
            "mean_95ci_high_initially_interrupted_critical_load_restored_fraction": "mean_95ci_high_critical",
        }
    )
    data = data.sort_values(["resource", "time_hours"]).reset_index(drop=True)
    curve_columns = [
        "mean_total",
        "q25_total",
        "q75_total",
        "mean_95ci_low_total",
        "mean_95ci_high_total",
        "mean_critical",
        "q25_critical",
        "q75_critical",
        "mean_95ci_low_critical",
        "mean_95ci_high_critical",
    ]
    for column in curve_columns:
        display_column = f"display_{column}_7h_centered_mean"

        def smooth_for_display(values: pd.Series) -> pd.Series:
            smoothed = values.rolling(window=7, center=True, min_periods=1).mean()
            edge = min(3, len(values))
            smoothed.iloc[:edge] = values.iloc[:edge]
            smoothed.iloc[-edge:] = values.iloc[-edge:]
            return smoothed.clip(0.0, 1.0)

        data[display_column] = data.groupby("resource", sort=False)[column].transform(
            smooth_for_display
        )
    data.to_csv(
        OUT / "plot_ready" / "FigureS04_full_horizon_anchor_supported_trajectories.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.35), sharex=True, gridspec_kw={"wspace": 0.28})
    for axis, mean_col, q25_col, q75_col, title, ylabel, lab in [
        (
            axes[0],
            "display_mean_total_7h_centered_mean",
            "display_mean_95ci_low_total_7h_centered_mean",
            "display_mean_95ci_high_total_7h_centered_mean",
            "Total interrupted load receiving service",
            "Fraction of initially interrupted load\nreceiving service",
            "(a)",
        ),
        (
            axes[1],
            "display_mean_critical_7h_centered_mean",
            "display_mean_95ci_low_critical_7h_centered_mean",
            "display_mean_95ci_high_critical_7h_centered_mean",
            "Critical interrupted load receiving service",
            "Fraction of initially interrupted critical load\nreceiving service",
            "(b)",
        ),
    ]:
        for resource in RESOURCE_ORDER:
            part = data[data["resource"].eq(resource)].sort_values("time_hours")
            label = RESOURCE_LABEL[resource]
            axis.fill_between(
                part["time_hours"],
                part[q25_col],
                part[q75_col],
                color=RESOURCE_COLOR[label],
                alpha=0.10,
                lw=0,
            )
            axis.plot(
                part["time_hours"],
                part[mean_col],
                color=RESOURCE_COLOR[label],
                ls="--" if label == "Power-only reference" else "-",
                lw=1.45,
                label=label,
            )
        axis.set_title(title, fontweight="normal")
        axis.set_xlabel("Restoration time (h)")
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.02, 1.04)
        style_axis(axis, "both")
        panel(axis, lab)
    axes[1].legend(loc="lower center", bbox_to_anchor=(-0.08, 1.14), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.78, bottom=0.17)
    return export(fig, "supplementary", "FigureS04_full_horizon_anchor_supported_trajectories")


def fresh_category_data() -> pd.DataFrame:
    fresh = read_table(REFRESH / "fresh" / "results_by_episode.csv")
    unique = read_table(REFRESH / "fresh" / "unique_bank_results_by_episode.csv")
    categories = [
        ("Unique fresh bank", unique, True),
        (
            "Historical nonzero-damage validation subset",
            fresh[fresh["eval_subset"].eq("fresh_historical_nonzero")],
            False,
        ),
        (
            "Hard-disruption aggregate (overlap-aware)",
            fresh[fresh["eval_subset"].isin(FRESH_HARD_SUBSETS)].drop_duplicates(
                ["scenario_key", "policy"]
            ),
            True,
        ),
        (
            "Intensity-scaled sensitivity set",
            fresh[fresh["eval_subset"].eq("fresh_stress_intensity_scaled")],
            False,
        ),
    ]
    rows = []
    for category_index, (name, data, unique_flag) in enumerate(categories):
        for policy_name in [RAW_BY_DISPLAY[p] for p in COMPETITIVE]:
            values = paired_reductions(
                data,
                policy_name,
                "harmonized_economic_damage_dollars",
                unique_scenarios=unique_flag,
            )
            low, high = bootstrap_mean(values, 1600 + category_index * 20 + len(rows))
            rows.append(
                {
                    "validation_category": name,
                    "policy": DISPLAY[policy_name],
                    "value": float(np.mean(values)),
                    "lower_ci": low,
                    "upper_ci": high,
                    "episode_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def supplementary_s5(pal: dict[str, dict[str, object]]) -> dict[str, str]:
    data = fresh_category_data()
    data.to_csv(
        OUT / "plot_ready" / "FigureS05_full_previously_unseen_comparator_context.csv", index=False
    )
    categories = list(data["validation_category"].drop_duplicates())
    fig, axes = plt.subplots(
        2, 2, figsize=(8.4, 5.2), sharex=True, gridspec_kw={"wspace": 0.34, "hspace": 0.34}
    )
    axes = axes.ravel()
    y = np.arange(len(COMPETITIVE))[::-1]
    xmin = float(data["lower_ci"].min()) - 1.0
    xmax = float(data["upper_ci"].max()) + 1.0
    for axis, category, lab in zip(axes, categories, ["(a)", "(b)", "(c)", "(d)"]):
        axis.axvline(0, color="#777777", lw=0.8)
        part = data[data["validation_category"].eq(category)].set_index("policy")
        for index, policy_name in enumerate(COMPETITIVE):
            row = part.loc[policy_name]
            value = float(row["value"])
            axis.errorbar(
                value,
                y[index],
                xerr=[[value - float(row["lower_ci"])], [float(row["upper_ci"]) - value]],
                fmt=pal[policy_name]["matplotlib_marker"],
                color=pal[policy_name]["color"],
                ms=5.0 if policy_name == "GraphRL-A2C" else 4.4,
                lw=1.0,
                capsize=2,
            )
        axis.set_yticks(y, COMPETITIVE)
        if axis in [axes[1], axes[3]]:
            axis.set_yticklabels([])
        axis.set_xlim(xmin, xmax)
        axis.set_title(category, fontweight="normal")
        style_axis(axis, "x")
        panel(axis, lab)
    fig.supxlabel("Economic-damage reduction relative to Greedy (%)", y=0.050, fontsize=8.0)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.93, bottom=0.15)
    return export(fig, "supplementary", "FigureS05_full_previously_unseen_comparator_context")

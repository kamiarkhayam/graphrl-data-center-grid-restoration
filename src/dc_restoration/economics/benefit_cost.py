from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dc_restoration.labels import public_display_columns
from dc_restoration.paths import require_file
from dc_restoration.policies.registry import EVALUATED

# Resource codes are frozen external schema identifiers; labels are public terminology.
RESOURCE_CONFIGS = {
    "E12": "E12_finite_energy_12h",
    "E24": "E24_finite_energy_24h",
    "Power-only reference": "R1_current_power_bounded_anchor",
}
RESOURCE_ORDER = list(RESOURCE_CONFIGS)
RESOURCE_AXIS_LABELS = {"E12": "E12", "E24": "E24", "Power-only reference": "Power-only\nreference"}
RESOURCE_COLORS = {"E12": "#B89B4D", "E24": "#2A9D8F", "Power-only reference": "#0C457D"}
RESOURCE_MARKERS = {"E12": "s", "E24": "D", "Power-only reference": "o"}
RESOURCE_LINESTYLES = {"E12": "-", "E24": "-", "Power-only reference": "--"}
NAVY, TEAL, GOLD, RED = "#0C457D", "#2A9D8F", "#B89B4D", "#B55245"
DARK, MID_GRAY, GRID = "#303030", "#777777", "#DEDEDE"
# The figure dispatcher supplies this destination explicitly.
OUT = Path("outputs/figures/benefit_cost")

CONFIG_KEYS = {
    "inputs": {"dc_results", "event_costs", "capital_summary", "community_ledger"},
    "analysis": {
        "selected_policy",
        "physical_subset",
        "event_cost_valuation_method",
        "reference_event_interval_years",
        "event_intervals_years",
    },
    "lifecycle": {
        "capital_case",
        "estimate_class",
        "real_discount_rate",
        "service_life_years",
        "fixed_export_om_dollars_per_year",
    },
    "community": {"rate_case", "fiscal_case", "reference_share", "shares"},
    "environment": {
        "lifecycle_include_local_air_quality",
        "lifecycle_include_climate",
        "host_include_local_air_quality",
    },
}


def _number(value, label, *, minimum=0.0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not np.isfinite(value) or value < minimum or (positive and value == minimum):
        raise ValueError(f"{label} is outside its allowed range")


def validate_config(config: dict) -> dict:
    """Reject unknown/missing keys rather than silently ignoring public assumptions."""
    if not isinstance(config, dict) or set(config) != set(CONFIG_KEYS) | {"output_dir"}:
        raise ValueError(
            "Benefit-cost config requires inputs, analysis, lifecycle, community, environment and output_dir only"
        )
    for section, keys in CONFIG_KEYS.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"{section} keys must be exactly {sorted(keys)}")
    strings = [("output_dir", config["output_dir"])]
    strings += [(f"inputs.{k}", v) for k, v in config["inputs"].items()]
    strings += [
        (f"analysis.{k}", config["analysis"][k])
        for k in ("selected_policy", "physical_subset", "event_cost_valuation_method")
    ]
    strings += [
        (f"lifecycle.{k}", config["lifecycle"][k]) for k in ("capital_case", "estimate_class")
    ]
    strings += [(f"community.{k}", config["community"][k]) for k in ("rate_case", "fiscal_case")]
    for label, value in strings:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty string")
    if config["analysis"]["selected_policy"] not in EVALUATED:
        raise ValueError("analysis.selected_policy must name a reported evaluation method")
    life = config["lifecycle"]
    for key in ("real_discount_rate", "fixed_export_om_dollars_per_year"):
        _number(life[key], f"lifecycle.{key}")
    _number(life["service_life_years"], "lifecycle.service_life_years", positive=True)
    for section, sequence, reference, bounded in (
        ("analysis", "event_intervals_years", "reference_event_interval_years", False),
        ("community", "shares", "reference_share", True),
    ):
        values = config[section][sequence]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{section}.{sequence} must be a nonempty list")
        for value in [*values, config[section][reference]]:
            _number(value, f"{section}.{sequence}/{reference}", positive=not bounded)
            if bounded and value > 1:
                raise ValueError("Community shares must lie in [0, 1]")
        if len(set(values)) != len(values) or config[section][reference] not in values:
            raise ValueError(f"{section}.{sequence} must be unique and contain {reference}")
    if any(type(v) is not bool for v in config["environment"].values()):
        raise ValueError("Environmental accounting switches must be YAML booleans")
    return config


def read_csv(path: Path) -> pd.DataFrame:
    return public_display_columns(pd.read_csv(require_file(path), low_memory=False))


def _columns(frame, required, label):
    missing = set(required) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"{label}: missing required columns {sorted(missing)} or no rows")


def _finite_nonnegative(frame, columns, label):
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{label}.{column}: expected finite nonnegative values")


def capital_recovery_factor(rate: float, years: float) -> float:
    _number(rate, "real_discount_rate")
    _number(years, "service_life_years", positive=True)
    return 1.0 / years if rate == 0 else rate / (-np.expm1(-years * np.log1p(rate)))


def selected_capital_and_crf(config: dict) -> tuple[float, float]:
    capital = read_csv(config["inputs"]["capital_summary"])
    column = "one_time_incremental_capital_2025_dollars"
    _columns(capital, ["cost_case", "estimate_class", column], "Capital ledger")
    life = config["lifecycle"]
    selected = capital[
        capital["cost_case"].eq(life["capital_case"])
        & capital["estimate_class"].eq(life["estimate_class"])
    ]
    if len(selected) != 1:
        raise ValueError(
            "Capital ledger: expected exactly one row for configured capital_case and estimate_class"
        )
    _finite_nonnegative(selected, [column], "Capital ledger")
    return float(selected.iloc[0][column]), capital_recovery_factor(
        life["real_discount_rate"], life["service_life_years"]
    )


def build_event_ledger(config: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pair the explicit cohort; never interpolate or substitute missing resource runs."""
    analysis = config["analysis"]
    dc = read_csv(config["inputs"]["dc_results"])
    metrics = [
        "harmonized_economic_damage_dollars",
        "energy_not_served_mwh",
        "critical_energy_not_served_mwh",
        "dc_support_used_mwh",
    ]
    _columns(
        dc,
        [
            "eval_subset",
            "display_policy",
            "scenario_key",
            "dc_resource_configuration",
            "random_seed",
            *metrics,
        ],
        "DC results",
    )
    selected = dc[
        dc["eval_subset"].eq(analysis["physical_subset"])
        & dc["display_policy"].eq(analysis["selected_policy"])
    ].copy()
    _finite_nonnegative(selected, [*metrics, "random_seed"], "DC results")
    if not selected["random_seed"].eq(selected["random_seed"].astype(int)).all():
        raise ValueError("DC results: random_seed must be an integer")
    if selected["scenario_key"].isna().any():
        raise ValueError("DC results: null scenario keys")
    expected = {"R0_no_dc_anchor", *RESOURCE_CONFIGS.values()}
    if set(selected["dc_resource_configuration"]) != expected:
        raise ValueError(
            "DC results: selected cohort must contain exactly No anchor, E12, E24 and Power-only reference"
        )
    if selected.duplicated(["scenario_key", "dc_resource_configuration"]).any():
        raise ValueError(
            "DC results: duplicate scenario/resource keys; select one frozen evaluation per scenario"
        )
    no_anchor = selected[selected["dc_resource_configuration"].eq("R0_no_dc_anchor")].set_index(
        "scenario_key"
    )
    costs = read_csv(config["inputs"]["event_costs"])
    rate_columns = {
        "provider_fuel_and_vom_dollars_per_mwh": "event_provider_fuel_and_vom_dollars",
        "local_health_damage_dollars_per_mwh": "selected_local_health_damage_dollars",
        "climate_damage_dollars_per_mwh": "event_climate_damage_dollars",
    }
    _columns(
        costs,
        ["valuation_method", "event_dc_support_mwh", *rate_columns.values()],
        "Event cost ledger",
    )
    costs = costs[costs["valuation_method"].eq(analysis["event_cost_valuation_method"])]
    if costs.empty:
        raise ValueError("Event cost ledger: configured valuation method has no rows")
    _finite_nonnegative(
        costs, ["event_dc_support_mwh", *rate_columns.values()], "Event cost ledger"
    )
    if (costs["event_dc_support_mwh"] <= 0).any():
        raise ValueError("Event cost ledger: unit-rate rows require positive support MWh")
    # Preserve the archived arithmetic mean of per-row cost/MWh, not a ratio of totals.
    rates = {
        name: float((costs[column] / costs["event_dc_support_mwh"]).mean())
        for name, column in rate_columns.items()
    }
    env = config["environment"]
    rows = []
    for display, code in RESOURCE_CONFIGS.items():
        anchor = selected[selected["dc_resource_configuration"].eq(code)].set_index("scenario_key")
        if set(no_anchor.index) != set(anchor.index):
            raise ValueError(f"DC results: scenario pairing failed for {display}")
        anchor = anchor.loc[no_anchor.index]
        if not np.array_equal(no_anchor["random_seed"], anchor["random_seed"]):
            raise ValueError(f"DC results: paired random seeds differ for {display}")
        benefit = no_anchor[metrics[0]] - anchor[metrics[0]]
        support = float(anchor["dc_support_used_mwh"].mean())
        provider, health, climate = [support * rates[name] for name in rate_columns]
        rows.append(
            {
                "resource_case": display,
                "dc_resource_configuration": code,
                "episode_count": len(anchor),
                "mean_no_anchor_damage_dollars": float(no_anchor[metrics[0]].mean()),
                "mean_anchor_damage_dollars": float(anchor[metrics[0]].mean()),
                "mean_event_benefit_dollars": float(benefit.mean()),
                "mean_avoided_ens_mwh": float((no_anchor[metrics[1]] - anchor[metrics[1]]).mean()),
                "mean_avoided_critical_ens_mwh": float(
                    (no_anchor[metrics[2]] - anchor[metrics[2]]).mean()
                ),
                "mean_dc_support_mwh": support,
                "event_provider_fuel_and_vom_dollars": provider,
                "event_local_health_damage_dollars": health,
                "event_climate_damage_dollars": climate,
                "event_full_resource_cost_dollars": provider + health + climate,
                "event_lifecycle_accounted_cost_dollars": provider
                + health * env["lifecycle_include_local_air_quality"]
                + climate * env["lifecycle_include_climate"],
                "event_host_community_externality_dollars": health
                * env["host_include_local_air_quality"],
                "benefit_definition": "Paired No anchor minus support-case economic interruption damage",
                "source_subset": analysis["physical_subset"],
                "selected_policy": analysis["selected_policy"],
            }
        )
    return pd.DataFrame(rows), rates


def _ratio(benefit, cost):
    # An undefined ratio is retained as missing, never silently represented as zero.
    return benefit / cost if cost > 0 else np.nan


def build_anchor_enablement_bcr(event_ledger: pd.DataFrame, config: dict) -> pd.DataFrame:
    capital, crf = selected_capital_and_crf(config)
    annual_capital = capital * crf
    fixed_om = config["lifecycle"]["fixed_export_om_dollars_per_year"]
    rows = []
    for _, event in event_ledger.iterrows():
        for interval in config["analysis"]["event_intervals_years"]:
            benefit = event["mean_event_benefit_dollars"] / interval
            event_cost = event["event_lifecycle_accounted_cost_dollars"] / interval
            cost = annual_capital + fixed_om + event_cost
            rows.append(
                {
                    "assessment": "Lifecycle export-enablement assessment",
                    "resource_case": event["resource_case"],
                    "event_interval_years": interval,
                    "reference_event_interval_years": config["analysis"][
                        "reference_event_interval_years"
                    ],
                    "event_benefit_dollars": event["mean_event_benefit_dollars"],
                    "event_resource_cost_dollars": event["event_lifecycle_accounted_cost_dollars"],
                    "event_climate_damage_dollars": event["event_climate_damage_dollars"],
                    "one_time_incremental_capital_dollars": capital,
                    "capital_recovery_factor": crf,
                    "equivalent_annual_capital_cost_dollars": annual_capital,
                    "annual_fixed_om_dollars": fixed_om,
                    "annualized_event_benefit_dollars": benefit,
                    "annualized_event_resource_cost_dollars": event_cost,
                    "annual_total_cost_dollars": cost,
                    "benefit_cost_ratio": _ratio(benefit, cost),
                    "annual_net_value_dollars": benefit - cost,
                    **config["environment"],
                }
            )
    return pd.DataFrame(rows)


COMMUNITY_COLUMNS = [
    "annual_local_property_tax_revenue_dollars",
    "annual_attributed_residential_rate_burden_dollars",
    "annual_local_sales_tax_exemption_burden_dollars",
    "annual_routine_generator_health_damage_dollars",
    "annual_routine_generator_climate_damage_dollars",
]


def selected_community_parameters(config: dict) -> dict[str, float]:
    """Extract only rate/fiscal/environment inputs; recompute capital, shares and O&M."""
    source = read_csv(config["inputs"]["community_ledger"])
    _columns(source, ["rate_case", "fiscal_case", *COMMUNITY_COLUMNS], "Community ledger")
    selected = source[
        source["rate_case"].eq(config["community"]["rate_case"])
        & source["fiscal_case"].eq(config["community"]["fiscal_case"])
    ]
    if selected.empty:
        raise ValueError("Community ledger: no rows for configured rate_case and fiscal_case")
    _finite_nonnegative(selected, COMMUNITY_COLUMNS, "Community ledger")
    values = selected[COMMUNITY_COLUMNS].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            "Community ledger: inconsistent recurring inputs for selected rate/fiscal case"
        )
    return {key: float(values.iloc[0][key]) for key in COMMUNITY_COLUMNS}


def build_host_community_ratio(event_ledger: pd.DataFrame, config: dict) -> pd.DataFrame:
    recurring = selected_community_parameters(config)
    capital, crf = selected_capital_and_crf(config)
    fixed_om = config["lifecycle"]["fixed_export_om_dollars_per_year"]
    interval = config["analysis"]["reference_event_interval_years"]
    local_health = config["environment"]["host_include_local_air_quality"]
    rows = []
    for _, event in event_ledger.iterrows():
        for share in config["community"]["shares"]:
            resilience = event["mean_event_benefit_dollars"] / interval
            event_health = event["event_local_health_damage_dollars"] / interval
            event_climate = event["event_climate_damage_dollars"] / interval
            routine_health = recurring["annual_routine_generator_health_damage_dollars"]
            routine_climate = recurring["annual_routine_generator_climate_damage_dollars"]
            benefit = resilience + recurring["annual_local_property_tax_revenue_dollars"]
            burden = (
                share * (capital * crf + fixed_om)
                + sum(recurring[k] for k in COMMUNITY_COLUMNS[1:3])
                + local_health * (routine_health + event_health)
            )
            rows.append(
                {
                    **recurring,
                    "assessment": "Host-community incidence assessment",
                    "resource_case": event["resource_case"],
                    "community_enablement_share": share,
                    "reference_community_share": config["community"]["reference_share"],
                    "event_interval_years": interval,
                    "annualized_resilience_benefit_dollars": resilience,
                    "annual_total_host_community_benefit_dollars": benefit,
                    "annual_public_enablement_capital_dollars": share * capital * crf,
                    "annual_public_incremental_fixed_om_dollars": share * fixed_om,
                    "annualized_outage_export_health_damage_dollars": event_health,
                    "annualized_outage_export_climate_damage_dollars": event_climate,
                    "annual_broader_societal_climate_externality_dollars": routine_climate
                    + event_climate,
                    "annual_accounted_local_air_quality_damage_dollars": local_health
                    * (routine_health + event_health),
                    "annual_total_host_community_burden_dollars": burden,
                    "host_community_benefit_to_burden_ratio": _ratio(benefit, burden),
                    "annual_net_host_community_value_dollars": benefit - burden,
                    "host_include_local_air_quality": local_health,
                    "climate_in_primary_host_denominator": False,
                }
            )
    return pd.DataFrame(rows)


def build_annual_four_case_comparison(
    event_ledger: pd.DataFrame, host_community: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Absolute ledgers contain residual damage; incremental ledgers count avoided damage once."""
    event = event_ledger.set_index("resource_case").loc["Power-only reference"]
    host = host_community[
        host_community["resource_case"].eq("Power-only reference")
        & np.isclose(
            host_community["community_enablement_share"], config["community"]["reference_share"]
        )
    ].iloc[0]
    local_health = config["environment"]["host_include_local_air_quality"]
    normal = (
        sum(
            float(host[k])
            for k in [
                "annual_public_enablement_capital_dollars",
                "annual_public_incremental_fixed_om_dollars",
                *COMMUNITY_COLUMNS[1:3],
            ]
        )
        + local_health * host["annual_routine_generator_health_damage_dollars"]
    )
    tax = host["annual_local_property_tax_revenue_dollars"]
    routine_climate = host["annual_routine_generator_climate_damage_dollars"]
    rows, incremental = [], []
    for disaster, year in [(False, "Typical year"), (True, "Representative disaster year")]:
        for present in (False, True):
            damage = (
                (
                    event["mean_anchor_damage_dollars"]
                    if present
                    else event["mean_no_anchor_damage_dollars"]
                )
                if disaster
                else 0.0
            )
            externality = (
                event["event_host_community_externality_dollars"] if present and disaster else 0.0
            )
            burden = normal if present else 0.0
            inflow = tax if present else 0.0
            cost = burden + damage + externality
            climate = (
                (routine_climate + (event["event_climate_damage_dollars"] if disaster else 0.0))
                if present
                else 0.0
            )
            rows.append(
                {
                    "year_type": year,
                    "dc_presence": "With data centers" if present else "Without data centers",
                    "scenario_display": year + ("\nwith DCs" if present else "\nwithout DCs"),
                    "annual_community_benefit_or_inflow_dollars": inflow,
                    "annual_community_cost_or_burden_dollars": cost,
                    "annual_net_community_value_dollars": inflow - cost,
                    "customer_interruption_damage_dollars": damage,
                    "dc_related_annual_burden_dollars": burden,
                    "outage_export_externality_dollars": externality,
                    "routine_generator_climate_damage_dollars": routine_climate if present else 0.0,
                    "event_climate_damage_dollars": event["event_climate_damage_dollars"]
                    if present and disaster
                    else 0.0,
                    "broader_societal_climate_externality_dollars": climate,
                }
            )
        benefit = tax + (event["mean_event_benefit_dollars"] if disaster else 0.0)
        cost = normal + (event["event_host_community_externality_dollars"] if disaster else 0.0)
        incremental.append(
            {
                "year_type": year,
                "incremental_benefit_dollars": benefit,
                "incremental_cost_dollars": cost,
                "incremental_net_value_dollars": benefit - cost,
                "incremental_benefit_cost_ratio": _ratio(benefit, cost),
                "avoided_interruption_damage_dollars": event["mean_event_benefit_dollars"]
                if disaster
                else 0.0,
                "routine_generator_climate_damage_dollars": routine_climate,
                "event_climate_damage_dollars": event["event_climate_damage_dollars"]
                if disaster
                else 0.0,
                "broader_societal_climate_externality_dollars": routine_climate
                + (event["event_climate_damage_dollars"] if disaster else 0.0),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(incremental)


def calculate(config: dict) -> dict[str, pd.DataFrame]:
    validate_config(config)
    event, _ = build_event_ledger(config)
    lifecycle = build_anchor_enablement_bcr(event, config)
    community = build_host_community_ratio(event, config)
    annual, incremental = build_annual_four_case_comparison(event, community, config)
    return {
        "event_ledger": event,
        "anchor_enablement_bcr": lifecycle,
        "host_community_ratio": community,
        "annual_four_case_comparison": annual,
        "annual_incremental_bcr": incremental,
    }


def setup_style() -> None:
    """Use the typography and line weights established by the v0.3 paper figures."""
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    ax.grid(axis=grid_axis, color=GRID, lw=0.42, alpha=0.75)
    ax.set_axisbelow(True)


def money_millions(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=float) / 1_000_000.0


def export_figure(fig: plt.Figure, family: str, stem: str) -> dict[str, str]:
    base = OUT / family
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": base / f"{stem}.pdf",
        "svg": base / f"{stem}.svg",
        "png": base / f"{stem}_600dpi.PNG",
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {name: str(path) for name, path in paths.items()}


def label_ratio_bars(ax: plt.Axes, bars, values: np.ndarray) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold" if value >= 1.0 else "normal",
        )


def figure_anchor_enablement(lifecycle: pd.DataFrame) -> dict[str, str]:
    reference_interval = float(lifecycle["reference_event_interval_years"].iloc[0])
    intervals = sorted(lifecycle["event_interval_years"].unique())
    reference = lifecycle[np.isclose(lifecycle["event_interval_years"], reference_interval)]
    reference = reference.set_index("resource_case").loc[RESOURCE_ORDER].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), gridspec_kw={"wspace": 0.32})

    x = np.arange(len(reference))
    values = reference["benefit_cost_ratio"].to_numpy(float)
    bars = axes[0].bar(
        x,
        values,
        color=[RESOURCE_COLORS[name] for name in RESOURCE_ORDER],
        width=0.64,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    label_ratio_bars(axes[0], bars, values)
    axes[0].set_xticks(x, [RESOURCE_AXIS_LABELS[name] for name in RESOURCE_ORDER])
    axes[0].set_ylabel("Benefit / cost")
    axes[0].set_title(f"(a) {reference_interval:g}-year event interval")
    axes[0].set_ylim(0, max(1.55, values.max() + 0.20))
    axes[0].axhline(1.0, color=DARK, lw=0.9, ls="--", zorder=1)
    style_axis(axes[0])

    for resource in RESOURCE_ORDER:
        subset = lifecycle[lifecycle["resource_case"].eq(resource)].sort_values(
            "event_interval_years"
        )
        axes[1].plot(
            subset["event_interval_years"],
            subset["benefit_cost_ratio"],
            marker=RESOURCE_MARKERS[resource],
            ms=4.2,
            lw=1.3,
            ls=RESOURCE_LINESTYLES[resource],
            color=RESOURCE_COLORS[resource],
            label=resource,
        )
    axes[1].set_xscale("log")
    axes[1].set_xticks(intervals)
    axes[1].set_xticklabels([f"{value:g}" for value in intervals])
    axes[1].set_xlabel("Event interval (years)")
    axes[1].set_ylabel("Benefit / cost")
    axes[1].set_title("(b) Event-interval sensitivity")
    axes[1].set_ylim(bottom=0)
    axes[1].axhline(1.0, color=DARK, lw=0.9, ls="--", zorder=1)
    style_axis(axes[1])
    axes[1].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(bottom=0.19)
    return export_figure(fig, "version_a", "Figure08_anchor_enablement_bcr")


def figure_host_community(community: pd.DataFrame) -> dict[str, str]:
    reference_share = float(community["reference_community_share"].iloc[0])
    reference = community[np.isclose(community["community_enablement_share"], reference_share)]
    reference = reference.set_index("resource_case").loc[RESOURCE_ORDER].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), gridspec_kw={"wspace": 0.34})

    x = np.arange(len(reference))
    values = reference["host_community_benefit_to_burden_ratio"].to_numpy(float)
    bars = axes[0].bar(
        x,
        values,
        color=[RESOURCE_COLORS[name] for name in RESOURCE_ORDER],
        width=0.64,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    label_ratio_bars(axes[0], bars, values)
    axes[0].set_xticks(x, [RESOURCE_AXIS_LABELS[name] for name in RESOURCE_ORDER])
    axes[0].set_ylabel("Community benefit / burden")
    axes[0].set_title(f"(a) {100 * reference_share:g}% community cost share")
    axes[0].set_ylim(0, values.max() + 0.45)
    axes[0].axhline(1.0, color=DARK, lw=0.9, ls="--", zorder=1)
    style_axis(axes[0])

    for resource in RESOURCE_ORDER:
        subset = community[community["resource_case"].eq(resource)].sort_values(
            "community_enablement_share"
        )
        axes[1].plot(
            subset["community_enablement_share"].to_numpy(float) * 100.0,
            subset["host_community_benefit_to_burden_ratio"],
            marker=RESOURCE_MARKERS[resource],
            ms=4.2,
            lw=1.3,
            ls=RESOURCE_LINESTYLES[resource],
            color=RESOURCE_COLORS[resource],
            label=resource,
        )
    axes[1].set_xticks(sorted(community["community_enablement_share"].unique() * 100))
    axes[1].set_xlim(-5, 105)
    axes[1].set_xlabel("Community cost share (%)")
    axes[1].set_ylabel("Community benefit / burden")
    axes[1].set_title("(b) Cost-share sensitivity")
    y_min = float(community["host_community_benefit_to_burden_ratio"].min())
    y_max = float(community["host_community_benefit_to_burden_ratio"].max())
    axes[1].set_ylim(y_min - 0.08, y_max + 0.13)
    style_axis(axes[1])
    axes[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.02))
    fig.subplots_adjust(bottom=0.19)
    return export_figure(fig, "version_a", "Figure09_host_community_value_ratio")


def figure_annual_balance(annual: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.9), gridspec_kw={"wspace": 0.34})
    x = np.arange(len(annual))
    plot_labels = [
        "Typical\nNo DCs",
        "Typical\nWith DCs",
        "Disaster\nNo DCs",
        "Disaster\nWith DCs",
    ]
    benefits = money_millions(annual["annual_community_benefit_or_inflow_dollars"])
    costs = money_millions(annual["annual_community_cost_or_burden_dollars"])
    net = money_millions(annual["annual_net_community_value_dollars"])
    width = 0.36

    axes[0].bar(
        x - width / 2,
        benefits,
        width,
        color=TEAL,
        edgecolor="white",
        linewidth=0.5,
        label="Community benefit",
        zorder=3,
    )
    axes[0].bar(
        x + width / 2,
        costs,
        width,
        color=GOLD,
        edgecolor="white",
        linewidth=0.5,
        label="Community cost",
        zorder=3,
    )
    axes[0].set_xticks(x, plot_labels)
    axes[0].set_ylabel("Annual value (million $)")
    axes[0].set_title("(a) Benefits and costs")
    axes[0].set_ylim(0, max(float(benefits.max()), float(costs.max())) + 0.75)
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    style_axis(axes[0])

    colors = [MID_GRAY if np.isclose(value, 0.0) else (TEAL if value > 0 else RED) for value in net]
    bars = axes[1].bar(
        x,
        net,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for bar, value in zip(bars, net):
        offset = 0.12 if value >= 0 else -0.16
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.0,
            fontweight="bold" if abs(value) > 0.001 else "normal",
        )
    axes[1].axhline(0.0, color=DARK, lw=0.8)
    axes[1].set_xticks(x, plot_labels)
    axes[1].set_ylabel("Net community value (million $)")
    axes[1].set_title("(b) Net annual value")
    pad = max(abs(float(net.min())), abs(float(net.max()))) * 0.16
    axes[1].set_ylim(float(net.min()) - pad, float(net.max()) + pad)
    style_axis(axes[1])
    fig.subplots_adjust(bottom=0.23)
    return export_figure(fig, "version_b", "Figure08_annual_community_balance")


def figure_annual_incremental_bcr(incremental: pd.DataFrame) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(3.65, 2.75))
    x = np.arange(len(incremental))
    values = incremental["incremental_benefit_cost_ratio"].to_numpy(float)
    bars = ax.bar(
        x,
        values,
        width=0.58,
        color=[TEAL, NAVY],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    label_ratio_bars(ax, bars, values)
    ax.set_xticks(x, ["Typical year", "Representative\ndisaster year"])
    ax.set_ylabel("Benefit / cost")
    ax.set_ylim(0, values.max() + 0.65)
    ax.axhline(1.0, color=DARK, lw=0.9, ls="--", zorder=1)
    style_axis(ax)
    fig.subplots_adjust(bottom=0.18)
    return export_figure(fig, "version_b", "Figure09_value_of_dc_presence")

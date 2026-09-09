"""Validate and consolidate explicitly supplied, completed evaluation runs.

Subset membership is retained. Unique-bank tables remove repeated memberships only
after verifying identical episode and trajectory values across those memberships.
No simulator, checkpoint loader, trainer, or search routine is called here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dc_restoration.anchors.controller import RESOURCE_CONFIGS
from dc_restoration.evaluation.statistics import _pairwise_vs_greedy, _summary
from dc_restoration.evaluation.trajectories import RESOURCE_DISPLAY, SUBSET_DISPLAY
from dc_restoration.labels import public_display_columns
from dc_restoration.paths import output_directory, require_file, resolve_path
from dc_restoration.policies.registry import EVALUATED

EPISODE_KEY = ["eval_subset", "scenario_key", "policy", "dc_resource_configuration"]
PHYSICAL_KEY = EPISODE_KEY[1:]
METRICS = [
    "harmonized_economic_damage_dollars",
    "energy_not_served_mwh",
    "critical_energy_not_served_mwh",
]
COUNTS = ["invalid_action_count", "export_cap_violation_count", "support_outside_anchor_zone_count"]
EPISODE_COLUMNS = [
    *EPISODE_KEY,
    "display_policy",
    "random_seed",
    *METRICS,
    "dc_support_used_mwh",
    "episode_length",
    "runtime_seconds",
    *COUNTS,
    "energy_depleted",
    "repair_sequence_edge_ids",
]
TRAJECTORY_COLUMNS = [
    *EPISODE_KEY,
    "display_policy",
    "random_seed",
    "step_index",
    "interval_start_time_hours",
    "interval_duration_hours",
    "cumulative_time_hours",
    "initial_unserved_mw",
    "initial_critical_unserved_mw",
    "current_unserved_mw",
    "current_critical_unserved_mw",
    "step_unserved_energy_mwh",
    "cumulative_ens_mwh",
    "selected_repair_action",
]


def _required(frame, columns, label, *, allow_empty=False):
    missing = set(columns) - set(frame)
    if missing or (frame.empty and not allow_empty):
        raise ValueError(f"{label}: missing columns {sorted(missing)} or no rows")


def _unique(frame, keys, label, ignore=()):
    """Collapse exact copies; never choose between conflicting evaluations."""
    duplicates = frame[frame.duplicated(keys, keep=False)]
    compare = [c for c in frame if c not in {*keys, *ignore}]
    if (
        not duplicates.empty
        and (duplicates.groupby(keys, dropna=False)[compare].nunique(dropna=False) > 1).any().any()
    ):
        raise ValueError(f"{label}: conflicting duplicate or cohort-membership values for {keys}")
    return frame.drop_duplicates(keys).copy()


def _finite(frame, columns, label):
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{label}: {column} must be finite and nonnegative")


def _same_keys(actual, expected, keys, label):
    a = set(actual[keys].itertuples(index=False, name=None))
    b = set(expected[keys].itertuples(index=False, name=None))
    if a != b:
        raise ValueError(
            f"{label}: missing or unexpected scenario pairs ({len(a - b)} extra, {len(b - a)} missing)"
        )


def read_runs(directories, *, role, audit):
    episodes, trajectories = [], []
    for directory in directories:
        for filename, collector in [
            ("results_by_episode.csv", episodes),
            ("trajectory_by_step.csv", trajectories),
        ]:
            path = require_file(resolve_path(directory) / filename)
            frame = public_display_columns(pd.read_csv(path))
            collector.append(frame)
            audit.append(
                {
                    "role": role,
                    "input": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": len(frame),
                }
            )
    if not episodes:
        raise ValueError(f"{role}: provide at least one evaluation directory")
    ep, tr = pd.concat(episodes, ignore_index=True), pd.concat(trajectories, ignore_index=True)
    _required(ep, EPISODE_COLUMNS, role)
    _required(tr, TRAJECTORY_COLUMNS, f"{role} trajectories", allow_empty=True)
    for frame in (ep, tr):
        if (
            frame[EPISODE_KEY].isna().any().any()
            or (frame[EPISODE_KEY].astype(str).map(lambda v: not v.strip())).any().any()
        ):
            raise ValueError(f"{role}: scenario, subset, policy and resource keys must be nonempty")
        if (
            not set(frame["policy"]) <= set(EVALUATED)
            or not frame["policy"].eq(frame["display_policy"]).all()
        ):
            raise ValueError(f"{role}: policy/display names must agree and name supported methods")
        if not set(frame["dc_resource_configuration"]) <= set(RESOURCE_CONFIGS):
            raise ValueError(f"{role}: unknown support case")
        _finite(frame, ["random_seed"], role)
        if not frame["random_seed"].eq(frame["random_seed"].astype(int)).all():
            raise ValueError(f"{role}: random_seed must be an integer")
    _finite(
        ep, [*METRICS, "dc_support_used_mwh", "episode_length", "runtime_seconds", *COUNTS], role
    )
    _finite(tr, ["step_index", *TRAJECTORY_COLUMNS[8:-1]], f"{role} trajectories")
    if ep[COUNTS].to_numpy().sum() != 0:
        raise ValueError(f"{role}: input reports a restoration constraint violation")
    depleted = ep["energy_depleted"].astype(str).str.lower()
    if not depleted.isin(["true", "false"]).all():
        raise ValueError(f"{role}: energy_depleted must be boolean")
    ep["energy_depleted"] = depleted.eq("true")
    ep = _unique(ep, EPISODE_KEY, role)
    tr = _unique(tr, [*EPISODE_KEY, "step_index"], f"{role} trajectories")
    # Each scenario is one frozen run per policy/resource, even across overlapping subsets.
    unique_ep = _unique(ep, PHYSICAL_KEY, role, ignore=("eval_subset", "display_subset"))
    unique_tr = _unique(
        tr, [*PHYSICAL_KEY, "step_index"], role, ignore=("eval_subset", "display_subset")
    )
    if not tr.empty:
        pairing_key = [*EPISODE_KEY, "random_seed"]
        merged = (
            tr[pairing_key]
            .drop_duplicates()
            .merge(ep[pairing_key], how="left", on=pairing_key, indicator=True)
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{role}: orphan trajectory rows or mismatched random_seed")
    groups = {key: group for key, group in tr.groupby(EPISODE_KEY)}
    for _, row in ep.iterrows():
        key = tuple(row[k] for k in EPISODE_KEY)
        steps = groups.get(key, tr.iloc[:0]).sort_values("step_index")
        n = int(row["episode_length"])
        if n != row["episode_length"] or list(steps["step_index"]) != list(range(n)):
            raise ValueError(f"{role}: trajectory steps do not match episode_length for {key}")
        try:
            order = json.loads(row["repair_sequence_edge_ids"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{role}: malformed repair sequence for {key}") from exc
        if order != list(steps["selected_repair_action"]) or len(order) != len(set(order)):
            raise ValueError(f"{role}: repair order/trajectory mismatch for {key}")
        if n:
            if not np.allclose(
                steps["interval_start_time_hours"].to_numpy(),
                [0, *steps["cumulative_time_hours"].to_numpy()[:-1]],
            ):
                raise ValueError(f"{role}: non-contiguous trajectory intervals")
            if not np.allclose(
                steps["interval_start_time_hours"] + steps["interval_duration_hours"],
                steps["cumulative_time_hours"],
            ):
                raise ValueError(f"{role}: inconsistent interval duration")
            if not np.isclose(
                steps["step_unserved_energy_mwh"].sum(), row["energy_not_served_mwh"]
            ):
                raise ValueError(f"{role}: episode/trajectory ENS mismatch")
        elif any(row[m] != 0 for m in METRICS):
            raise ValueError(f"{role}: nonzero damage/ENS without trajectory intervals")
    for frame in (ep, tr, unique_ep, unique_tr):
        frame["display_subset"] = (
            frame["eval_subset"].map(SUBSET_DISPLAY).fillna(frame["eval_subset"])
        )
        frame["dc_resource_display"] = frame["dc_resource_configuration"].map(RESOURCE_DISPLAY)
    ep["unserved_mwh"] = ep["energy_not_served_mwh"]
    ep["critical_unserved_mwh"] = ep["critical_energy_not_served_mwh"]
    return ep, tr, unique_ep, unique_tr


def _method_pairs(ep, label):
    baseline = ep[ep["policy"].eq("Greedy")]
    if baseline.empty:
        raise ValueError(f"{label}: Greedy reference is required")
    for policy, part in ep.groupby("policy"):
        _same_keys(
            part, baseline, ["eval_subset", "scenario_key", "random_seed"], f"{label}/{policy}"
        )


def _summaries(ep, config, prefix):
    _method_pairs(ep, prefix)
    pair = _pairwise_vs_greedy(ep, "eval_subset", config)
    if pair.empty:
        raise ValueError(f"{prefix}: provide at least one method paired with Greedy")
    return {
        f"{prefix}/results_by_episode": ep,
        f"{prefix}/summary_by_subset": _summary(ep, "eval_subset", config),
        f"{prefix}/pairwise_vs_greedy": pair,
    }


def _resource_tables(ep, unique_tr):
    if not set(ep["policy"]) <= {"Greedy", "GraphRL-A2C"}:
        raise ValueError("Resource comparisons support Greedy and GraphRL-A2C")
    keys = ["eval_subset", "scenario_key", "policy", "random_seed"]
    base = ep[ep["dc_resource_configuration"].eq("R0_no_dc_anchor")]
    if base.empty:
        raise ValueError("Resource comparisons require No anchor")
    for resource in RESOURCE_CONFIGS:
        _same_keys(
            ep[ep["dc_resource_configuration"].eq(resource)],
            base,
            keys,
            f"Resource {RESOURCE_DISPLAY[resource]}",
        )
    paired = ep.merge(
        base[keys + METRICS].rename(columns={m: f"base_{m}" for m in METRICS}),
        on=keys,
        validate="many_to_one",
    )
    names = ["economic_damage", "ens", "critical_ens"]
    for metric, name in zip(METRICS, names):
        paired[f"{name}_reduction_vs_no_anchor_percent"] = (
            100
            * (paired[f"base_{metric}"] - paired[metric])
            / paired[f"base_{metric}"].clip(lower=1e-9)
        )
    group = [
        "eval_subset",
        "display_subset",
        "policy",
        "display_policy",
        "dc_resource_configuration",
        "dc_resource_display",
    ]
    benefit = (
        paired[~paired["dc_resource_configuration"].eq("R0_no_dc_anchor")]
        .groupby(group, as_index=False)
        .agg(
            episode_count=("scenario_key", "nunique"),
            **{
                f"mean_{name}_reduction_vs_no_anchor_percent": (
                    f"{name}_reduction_vs_no_anchor_percent",
                    "mean",
                )
                for name in names
            },
            energy_depletion_rate=("energy_depleted", "mean"),
            mean_dc_support_used_mwh=("dc_support_used_mwh", "mean"),
        )
    )
    return {
        "dc_anchor/dc_results_by_episode": ep,
        "dc_anchor/dc_trajectory_unique_by_step": unique_tr.drop(
            columns=["eval_subset", "display_subset"]
        ),
        "dc_anchor/dc_benefit_vs_no_anchor": benefit,
    }


def _ga_tables(directory, evaluation_dirs, audit):
    ep, _, unique, _ = read_runs(evaluation_dirs, role="ga-evaluation", audit=audit)
    if not ep["dc_resource_configuration"].eq("R1_current_power_bounded_anchor").all():
        raise ValueError("GA comparison requires Power-only reference evaluations")
    path = require_file(resolve_path(directory) / "bounded_ga_results.csv")
    ga = pd.read_csv(path)
    audit.append(
        {
            "role": "bounded-ga",
            "input": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(ga),
        }
    )
    required = [
        "scenario_key",
        "paper_primary_economic_damage_dollars",
        "runtime_batch_elapsed_seconds",
        "fitness_evaluations",
        "population",
        "generations",
    ]
    _required(ga, required, "GA ledger")
    if ga["scenario_key"].isna().any() or ga["scenario_key"].duplicated().any():
        raise ValueError("GA ledger: expected unique nonempty scenario keys")
    _finite(ga, required[1:], "GA ledger")
    for c in ("population", "generations"):
        if ga[c].nunique() != 1:
            raise ValueError("GA ledger: mixed search budgets")
    selected = unique[unique["policy"].eq("GraphRL-A2C")]
    greedy = unique[unique["policy"].eq("Greedy")]
    for data in (selected, greedy):
        _same_keys(ga, data, ["scenario_key"], "GA comparison")
    _same_keys(selected, greedy, ["scenario_key", "random_seed"], "GA reference seeds")
    comparison = ga.merge(
        selected[["scenario_key", METRICS[0], "runtime_seconds"]].rename(
            columns={
                METRICS[0]: "selected_economic_damage_dollars",
                "runtime_seconds": "selected_runtime_seconds",
            }
        ),
        on="scenario_key",
        validate="one_to_one",
    )
    comparison = comparison.merge(
        greedy[["scenario_key", METRICS[0]]].rename(
            columns={METRICS[0]: "greedy_economic_damage_dollars"}
        ),
        on="scenario_key",
        validate="one_to_one",
    )
    for prefix, column in [
        ("selected", "selected_economic_damage_dollars"),
        ("bounded_ga", "paper_primary_economic_damage_dollars"),
    ]:
        comparison[f"{prefix}_improvement_vs_greedy_percent"] = (
            100
            * (comparison["greedy_economic_damage_dollars"] - comparison[column])
            / comparison["greedy_economic_damage_dollars"].clip(lower=1e-9)
        )
    summary = pd.DataFrame(
        [
            {
                "cohort_size": len(ga),
                "population": ga["population"].iloc[0],
                "generations": ga["generations"].iloc[0],
                "mean_bounded_ga_fitness_evaluations": ga["fitness_evaluations"].mean(),
                "globally_optimal_claim": False,
            }
        ]
    )
    return {
        "bounded_ga/bounded_ga_comparison_by_scenario": comparison,
        "bounded_ga/bounded_ga_summary": summary,
    }


def prepare_analysis_tables(
    *,
    config,
    output_dir,
    controlled_dirs=(),
    resource_dirs=(),
    fresh_dirs=(),
    fresh_unique_manifest=None,
    ga_dir=None,
    ga_evaluation_dirs=(),
):
    if not any((controlled_dirs, resource_dirs, fresh_dirs, ga_dir)):
        raise ValueError("Supply at least one explicit evaluation input group")
    if bool(ga_dir) != bool(ga_evaluation_dirs):
        raise ValueError("--ga-dir and --ga-evaluation-dir must be supplied together")
    if bool(fresh_dirs) != bool(fresh_unique_manifest):
        raise ValueError("Fresh runs require an explicit unique-bank manifest, and vice versa")
    audit, tables = [], {}
    for label, directories in (("controlled", controlled_dirs), ("fresh", fresh_dirs)):
        if not directories:
            continue
        ep, _, _, _ = read_runs(directories, role=label, audit=audit)
        if not ep["dc_resource_configuration"].eq("R1_current_power_bounded_anchor").all():
            raise ValueError(f"{label}: method comparisons require Power-only reference")
        tables.update(_summaries(ep, config, label))
        if label == "controlled":
            pairs = tables["controlled/pairwise_vs_greedy"]
            main = pairs[pairs["eval_subset"].isin(config["subsets"]["controlled_main"])]
            if not main.empty:
                if set(main["eval_subset"]) != set(config["subsets"]["controlled_main"]):
                    raise ValueError(
                        "Controlled main summary requires every configured hard subset"
                    )
                tables["controlled/main_hard_subset_average"] = main.groupby(
                    ["policy", "display_policy"], as_index=False
                ).agg(
                    **{
                        c: (c, "mean")
                        for c in (
                            "mean_economic_improvement_percent",
                            "mean_ens_improvement_percent",
                            "mean_critical_ens_improvement_percent",
                        )
                    },
                    mean_win_rate=("win_rate", "mean"),
                )
        else:
            path = require_file(fresh_unique_manifest)
            manifest = pd.read_csv(path)
            _required(manifest, ["scenario_key", "random_seed"], "Fresh unique manifest")
            if manifest["scenario_key"].isna().any() or manifest["scenario_key"].duplicated().any():
                raise ValueError("Fresh unique manifest requires unique nonempty scenario keys")
            unique = _unique(
                ep, PHYSICAL_KEY, "Fresh bank", ignore=("eval_subset", "display_subset")
            )
            for _, group in unique.groupby("policy"):
                _same_keys(group, manifest, ["scenario_key", "random_seed"], "Fresh bank manifest")
            audit.append(
                {
                    "role": "fresh-unique-manifest",
                    "input": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": len(manifest),
                }
            )
            unique["eval_subset"] = "fresh_unique_bank"
            tables["fresh/unique_bank_results_by_episode"] = unique
    if resource_dirs:
        ep, _, _, unique_tr = read_runs(resource_dirs, role="resources", audit=audit)
        tables.update(_resource_tables(ep, unique_tr))
    if ga_dir:
        tables.update(_ga_tables(ga_dir, ga_evaluation_dirs, audit))
    # Finish all validation before creating outputs; reject a stale destination.
    output = resolve_path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output directory must be empty; choose a new run directory")
    output_directory(output)
    for name, frame in tables.items():
        destination = output / f"{name}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    (output / "analysis_input_manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return tables

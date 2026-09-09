"""Small command dispatchers; numerical work lives in copied library definitions."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from dc_restoration.policies.registry import EVALUATED, METHODS, TRAINABLE


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--root", type=Path, help="Repository root (or set DC_RESTORATION_ROOT)")
    return result


def configure_root(args: argparse.Namespace) -> None:
    if args.root is not None:
        os.environ["DC_RESTORATION_ROOT"] = str(args.root.resolve())


def train_main() -> None:
    command = parser("Train one reported method using the final shared training protocol.")
    command.add_argument("--method", required=True, choices=TRAINABLE)
    command.add_argument("--config", default="configs/policies/training.yaml")
    command.add_argument("--case-dir", type=Path)
    command.add_argument("--output-dir", type=Path)
    command.add_argument(
        "--device", default="cuda", help="Device for the requested run; paper work used CUDA"
    )
    command.add_argument("--phase", choices=("dataset", "bc", "rl", "all"), default="all")
    args = command.parse_args()
    configure_root(args)
    import torch

    from dc_restoration.configuration import training_config, use_case
    from dc_restoration.paths import resolve_output_path
    from dc_restoration.policies import training

    config = training_config(args.config)
    use_case(config, args.case_dir)
    method = METHODS[args.method]
    if args.method == "GraphBC" and args.phase == "rl":
        command.error("GraphBC has a behavior-cloning training phase; choose bc or all.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        command.error("CUDA is unavailable; choose an available device explicitly.")
    output = resolve_output_path(args.output_dir or config["paths"]["output_dir"])
    config["paths"]["output_dir"] = str(output)
    # Context validation occurs before any expensive work.
    training._load_context(config)
    directories = training._dirs(output)
    data = training._load_dataset(config, directories)
    if args.phase in ("bc", "all"):
        training._train_bc(method.architecture, data, config, directories, device)
    if args.phase in ("rl", "all") and method.algorithm != "bc":
        training._train_rl(method.architecture, method.algorithm, data, config, directories, device)
    print(f"Completed requested {args.phase} phase for {args.method}: {output}")


def evaluate_main() -> None:
    command = parser(
        "Evaluate reported methods on an explicitly supplied external scenario manifest."
    )
    command.add_argument("--method", nargs="+", choices=EVALUATED, default=["Greedy"])
    command.add_argument("--config", default="configs/evaluation/paper.yaml")
    command.add_argument("--case-dir", type=Path)
    command.add_argument("--manifest", type=Path)
    command.add_argument("--damage-components", type=Path)
    command.add_argument(
        "--checkpoint", type=Path, help="Selected checkpoint; requires exactly one learned method"
    )
    command.add_argument("--checkpoint-root", type=Path, help="Training output directory")
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--device", default="cpu")
    command.add_argument("--resource", choices=("R0", "R1", "E12", "E24"), default="R1")
    args = command.parse_args()
    configure_root(args)
    import numpy as np
    import pandas as pd
    import torch

    from dc_restoration.anchors.controller import RESOURCE_CONFIGS
    from dc_restoration.configuration import load_config, training_config, use_case
    from dc_restoration.evaluation import trajectories
    from dc_restoration.evaluation.analysis_tables import TRAJECTORY_COLUMNS
    from dc_restoration.paths import output_directory, require_file, resolve_path
    from dc_restoration.policies import training
    from dc_restoration.testbed.benchmark import CachedRestorationSimulator

    config = load_config(args.config)
    protocol = training_config(config["training_config"])
    use_case(protocol, args.case_dir or config["paths"]["case_dir"])
    model, inventory, valuator, adapter = training._load_context(protocol)
    if args.damage_components:
        model.damage = pd.read_csv(require_file(args.damage_components))
        model.base = CachedRestorationSimulator(model.nodes, model.edges, model.loads, model.damage)
        model._metric_cache.clear()
    manifest = pd.read_csv(require_file(args.manifest or config["paths"]["manifest"]))
    if not {"scenario_key", "eval_subset"}.issubset(manifest.columns):
        command.error("Manifest must have scenario_key and eval_subset columns.")
    if manifest.empty or manifest[["scenario_key", "eval_subset"]].isna().any().any():
        command.error("Manifest must contain nonempty scenario and subset keys.")
    seeds = pd.to_numeric(
        manifest.get("random_seed", pd.Series(config["random_seed"], index=manifest.index)),
        errors="coerce",
    )
    if not np.isfinite(seeds).all() or (seeds < 0).any() or not seeds.eq(seeds.round()).all():
        command.error("Manifest random_seed must be a finite nonnegative integer.")
    if args.checkpoint and len(args.method) != 1:
        command.error("--checkpoint requires one method.")
    if args.resource != "R1" and any(name not in ("Greedy", "GraphRL-A2C") for name in args.method):
        command.error("Resource comparisons in the paper use Greedy and GraphRL-A2C.")
    device = torch.device(args.device)
    output = output_directory(args.output_dir or config["paths"]["output_dir"], require_empty=True)
    training_output = resolve_path(args.checkpoint_root or protocol["paths"]["output_dir"])
    resources = {
        "R0": "R0_no_dc_anchor",
        "R1": "R1_current_power_bounded_anchor",
        "E12": "E12_finite_energy_12h",
        "E24": "E24_finite_energy_24h",
    }
    rows, step_rows = [], []
    for name in args.method:
        method = METHODS[name]
        policy, edge_index = None, None
        if method.algorithm is not None:
            filename = (
                "best_validation_regret.pt" if method.algorithm == "bc" else "best_vs_greedy.pt"
            )
            checkpoint = (
                args.checkpoint
                or training_output
                / "checkpoints"
                / f"{method.architecture}_harmonized_{method.algorithm}"
                / filename
            )
            policy, edge_index, _ = training._load_rl_policy_path(
                method.architecture, require_file(checkpoint), protocol, device
            )
            policy.eval()
        cache = {}
        for _, member in manifest.iterrows():
            scenario = str(member["scenario_key"])
            if scenario not in set(model.damage["scenario_key"].astype(str)) | set(
                model.manifest["scenario_key"].astype(str)
            ):
                raise ValueError(f"Scenario is absent from the supplied case: {scenario}")
            seed = int(member.get("random_seed", config["random_seed"]))
            key = (scenario, seed)
            if key not in cache:
                if args.resource == "R1":
                    metrics, _ = training._rollout_harmonized(
                        method.architecture,
                        model,
                        adapter,
                        valuator,
                        scenario,
                        protocol,
                        device,
                        policy=policy,
                        edge_index=edge_index,
                        deterministic=True,
                        random_seed=seed,
                    )
                    steps, physical = trajectories._trajectory_from_order(
                        model,
                        inventory,
                        valuator,
                        scenario,
                        method.archive_label,
                        name,
                        json.loads(metrics["repair_sequence_edge_ids"]),
                    )
                    metrics.update(physical)
                else:
                    resource = RESOURCE_CONFIGS[resources[args.resource]]
                    resource_adapter = trajectories.HarmonizedResourceAdapter(
                        model, inventory, valuator, adapter
                    )
                    metrics, steps = trajectories._dc_rollout(
                        model,
                        inventory,
                        valuator,
                        resource_adapter,
                        scenario,
                        method.archive_label,
                        name,
                        resource,
                        int(protocol["valuation"]["candidate_k"]),
                        device,
                        policy=policy,
                        edge_index=edge_index,
                    )
                cache[key] = metrics, steps
            metrics, steps = cache[key]
            metadata = {
                "eval_subset": str(member["eval_subset"]),
                "scenario_key": scenario,
                "policy": name,
                "anchor_allocation_mode": model.allocation_mode,
                "display_policy": name,
                "dc_resource_configuration": resources[args.resource],
                "random_seed": seed,
                "energy_depleted": bool(metrics.get("energy_depleted", False)),
            }
            rows.append({**metrics, **metadata})
            step_rows.extend({**row, **metadata} for row in steps)
    pd.DataFrame(rows).to_csv(output / "results_by_episode.csv", index=False)
    pd.DataFrame(step_rows, columns=None if step_rows else TRAJECTORY_COLUMNS).to_csv(
        output / "trajectory_by_step.csv", index=False
    )
    print(f"Evaluation written to {output}")


def analysis_tables_main() -> None:
    command = parser("Prepare analysis tables from explicitly supplied completed evaluation runs.")
    command.add_argument("--config", default="configs/evaluation/paper.yaml")
    command.add_argument("--controlled-dir", type=Path, action="append", default=[])
    command.add_argument("--resource-dir", type=Path, action="append", default=[])
    command.add_argument("--fresh-dir", type=Path, action="append", default=[])
    command.add_argument("--fresh-unique-manifest", type=Path)
    command.add_argument("--ga-dir", type=Path)
    command.add_argument("--ga-evaluation-dir", type=Path, action="append", default=[])
    command.add_argument("--output-dir", type=Path, required=True)
    args = command.parse_args()
    configure_root(args)
    from dc_restoration.configuration import load_config
    from dc_restoration.evaluation.analysis_tables import prepare_analysis_tables

    tables = prepare_analysis_tables(
        config=load_config(args.config),
        output_dir=args.output_dir,
        controlled_dirs=args.controlled_dir,
        resource_dirs=args.resource_dir,
        fresh_dirs=args.fresh_dir,
        fresh_unique_manifest=args.fresh_unique_manifest,
        ga_dir=args.ga_dir,
        ga_evaluation_dirs=args.ga_evaluation_dir,
    )
    print(f"Prepared {len(tables)} analysis tables in {args.output_dir}")


def ga_main() -> None:
    command = parser("Run the reported Bounded GA on an external locked cohort.")
    command.add_argument("--config", default="configs/evaluation/bounded_ga.yaml")
    command.add_argument("--case-dir", type=Path)
    command.add_argument("--cohort", type=Path)
    command.add_argument("--damage-components", type=Path)
    command.add_argument("--output-dir", type=Path)
    args = command.parse_args()
    configure_root(args)
    import pandas as pd

    from dc_restoration.configuration import load_config
    from dc_restoration.evaluation import bounded_ga as ga
    from dc_restoration.paths import output_directory, require_file, resolve_path

    config = load_config(args.config)
    paths = config["paths"]
    cohort = pd.read_csv(require_file(args.cohort or paths["cohort"]))
    if cohort["scenario_key"].duplicated().any():
        command.error("Locked cohort must have unique scenario keys.")
    case = resolve_path(args.case_dir or paths["case_dir"])
    damage = require_file(args.damage_components or paths["damage_components"])
    duration, nodes = require_file(paths["duration_config"]), require_file(paths["node_config"])
    critical = float(config["valuation"]["critical_service_value_usd_per_kwh"])
    output = output_directory(args.output_dir or paths["output_dir"], require_empty=True)
    ga._worker_init(str(case), str(damage), str(duration), str(nodes), critical)
    greedy = pd.DataFrame(
        [
            ga._simulate_source_greedy(
                ga._WORKER_MODEL,
                ga._WORKER_VALUATOR,
                str(row["scenario_key"]),
                "R1_current_power_bounded_anchor",
            )
            for _, row in cohort.iterrows()
        ]
    )
    greedy.to_csv(output / "greedy_seed_orders.csv", index=False)
    ga._run_harmonized_ga(
        cohort,
        greedy,
        case,
        damage,
        duration,
        nodes,
        critical,
        int(config["runtime"]["harmonized_greedy_workers"]),
        int(config["runtime"]["ga_population"]),
        int(config["runtime"]["ga_generations"]),
        output / "bounded_ga_results.csv",
    )


def prepare_main() -> None:
    stages = (
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
        "benchmark",
        "benchmark_scenarios",
    )
    command = parser("Prepare external testbed inputs; stages are explicit and may be expensive.")
    command.add_argument("--stage", required=True, choices=stages)
    command.add_argument(
        "--config", help="Defaults to base.yaml, or benchmark.yaml for benchmark stages"
    )
    args = command.parse_args()
    configure_root(args)
    from dc_restoration.configuration import load_config
    from dc_restoration.paths import repository_root, require_file, resolve_output_path

    if args.stage in ("benchmark", "benchmark_scenarios"):
        from dc_restoration.testbed import benchmark

        config = load_config(args.config or "configs/testbed/benchmark.yaml")
        paths = config["paths"]
        directories = benchmark._make_dirs(
            resolve_output_path(paths["benchmark_case_dir"]),
            resolve_output_path(paths["output_dir"]),
        )
        if args.stage == "benchmark":
            converted = benchmark.convert_smartds_candidate(config, repository_root(), directories)
            benchmark.write_conversion_outputs(config, directories, converted)
        else:
            benchmark.generate_benchmark_scenarios(config, directories)
    else:
        module = importlib.import_module("dc_restoration.testbed." + args.stage)
        original = sys.argv
        try:
            sys.argv = [
                original[0],
                "--config",
                str(require_file(args.config or "configs/testbed/base.yaml")),
            ]
            module.main()
        finally:
            sys.argv = original


def benefit_cost_main() -> None:
    command = parser("Calculate source-based benefit-cost ledgers from supplied external inputs.")
    command.add_argument("--config", default="configs/economics/benefit_cost.yaml")
    command.add_argument("--output-dir", type=Path)
    args = command.parse_args()
    configure_root(args)
    from dc_restoration.configuration import load_config
    from dc_restoration.economics.benefit_cost import calculate
    from dc_restoration.paths import output_directory

    config = load_config(args.config)
    tables = calculate(config)
    output = output_directory(args.output_dir or config["output_dir"], require_empty=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    print(f"Benefit-cost ledgers written to {output}")


def figures_main() -> None:
    command = parser("Draw paper figures from external plot-ready tables; no policy evaluation.")
    command.add_argument("--config", default="configs/evaluation/figures.yaml")
    command.add_argument(
        "--figure",
        required=True,
        choices=(
            "system",
            "severity",
            "3",
            "4",
            "5",
            "6",
            "7",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "benefit-cost-a",
            "benefit-cost-b",
            "benefit-cost-all",
        ),
    )
    args = command.parse_args()
    configure_root(args)
    from dc_restoration.plotting.api import generate

    generate(args.config, args.figure)

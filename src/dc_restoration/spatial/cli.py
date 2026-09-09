"""Use the package's argparse/root conventions for explicit spatial stages."""

from dc_restoration.cli import configure_root, parser
from dc_restoration.spatial.configuration import load_map_config


def _main(stage):
    command = parser(f"{stage.capitalize()} the version-3 introductory data-center/hurricane map.")
    command.add_argument(
        "--config", help="YAML paths; defaults to the configuration bundled with the package"
    )
    for key in ("cache-dir", "prepared-dir", "output-dir"):
        command.add_argument("--" + key)
    if stage == "fetch":
        command.add_argument(
            "--offline",
            action="store_true",
            help="Verify an existing cache without any network request",
        )
    args = command.parse_args()
    configure_root(args)
    try:
        config = load_map_config(
            args.config,
            cache_dir=args.cache_dir,
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
        )
        if stage == "fetch":
            from dc_restoration.spatial.sources import fetch_sources

            fetch_sources(config, offline=args.offline)
        elif stage == "prepare":
            from dc_restoration.spatial.prepare import prepare_data

            prepare_data(config)
        else:
            from dc_restoration.plotting.intro_map import generate_map

            generate_map(config)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        command.error(str(exc))
    print(f"Completed introduction-map {stage} stage")


def fetch_main():
    _main("fetch")


def prepare_main():
    _main("prepare")


def generate_main():
    _main("generate")

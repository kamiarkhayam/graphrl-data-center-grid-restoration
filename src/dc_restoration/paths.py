"""Explicit repository and external-data paths; imports never create directories."""

import os
from pathlib import Path


def repository_root() -> Path:
    """Use DC_RESTORATION_ROOT or locate the staged source repository."""
    configured = os.environ.get("DC_RESTORATION_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root() / path).resolve()


def require_file(value: str | Path) -> Path:
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Required external input is missing: {path}. See data/README.md.")
    return path


def resolve_output_path(value: str | Path) -> Path:
    """Keep repository outputs in ignored trees; allow explicit external destinations."""
    path = resolve_path(value)
    root = repository_root()
    if path.is_relative_to(root):
        parts = path.relative_to(root).parts
        ignored = {
            "data",
            "data_raw",
            "data_intermediate",
            "data_processed",
            "outputs",
            "results",
            "models",
            "checkpoints",
            "figures",
            "logs",
            ".runtime",
            ".packaging",
        }
        if not parts or parts[0] not in ignored or parts[:2] == ("data", "README.md"):
            raise ValueError("Repository output must be in an ignored directory such as outputs/")
    return path


def output_directory(value: str | Path, *, require_empty: bool = False) -> Path:
    path = resolve_output_path(value)
    if require_empty and path.exists() and any(path.iterdir()):
        raise ValueError("Analysis output directory must be empty; choose a new run directory")
    path.mkdir(parents=True, exist_ok=True)
    return path

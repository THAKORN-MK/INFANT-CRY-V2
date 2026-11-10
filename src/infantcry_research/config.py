"""Configuration loading and validation for the research workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment-specific branch
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


CANONICAL_DBL_LABELS = ("eairh", "eh", "heh", "neh", "owh")


class ConfigError(ValueError):
    """Raised when a project config is invalid or inconsistent."""


@dataclass(frozen=True)
class DatasetConfig:
    model_generation: str
    dataset_version: str
    doi: str | None
    dataset_root: Path
    metadata_filename: str
    class_order: tuple[str, ...]
    tiers: tuple[str, ...]
    columns: dict[str, str]
    label_source: str = "metadata_column"
    tier_source: str = "metadata_column"
    label_path_parent_levels: int = 1
    tier_path_parent_levels: int = 2
    cross_check_parent_label: bool = False
    grouping_identity_verified: bool = False
    exclusions_file: Path | None = None


def load_dataset_config(
    config_path: Path,
    *,
    project_root: Path | None = None,
    require_data_root: bool = False,
) -> DatasetConfig:
    """Load and validate the dataset config, resolving relative paths safely."""

    if yaml is None:
        raise ConfigError("PyYAML is required to read YAML configuration") from _YAML_IMPORT_ERROR

    path = Path(config_path).resolve()
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Dataset config must contain a YAML mapping")

    required = (
        "model_generation",
        "dataset_version",
        "dataset_root",
        "metadata_filename",
        "class_order",
        "tiers",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ConfigError(f"Dataset config is missing required keys: {', '.join(missing)}")

    model_generation = str(raw["model_generation"]).strip().lower()
    if not model_generation:
        raise ConfigError("model_generation must not be empty")
    dataset_version = str(raw["dataset_version"]).strip()
    if not dataset_version:
        raise ConfigError("dataset_version must not be empty")
    doi_value = raw.get("doi")
    doi = str(doi_value).strip() if doi_value not in (None, "") else None

    labels = raw["class_order"]
    if not isinstance(labels, list) or len(labels) != len(CANONICAL_DBL_LABELS):
        raise ConfigError("class_order must list all five original DBL labels exactly once")
    label_order = tuple(str(label).strip().lower() for label in labels)
    if len(set(label_order)) != len(label_order) or set(label_order) != set(CANONICAL_DBL_LABELS):
        raise ConfigError(
            "class_order must contain only the five original DBL labels: "
            + ", ".join(CANONICAL_DBL_LABELS)
        )

    tiers = raw["tiers"]
    if not isinstance(tiers, list) or not tiers or any(not str(tier).strip() for tier in tiers):
        raise ConfigError("tiers must be a non-empty list of names")
    column_map = raw.get("columns", {})
    if not isinstance(column_map, dict):
        raise ConfigError("columns must be a YAML mapping")
    label_source = str(raw.get("label_source", "metadata_column")).strip().casefold()
    tier_source = str(raw.get("tier_source", "metadata_column")).strip().casefold()
    if label_source not in {"metadata_column", "parent_folder"}:
        raise ConfigError("label_source must be 'metadata_column' or 'parent_folder'")
    if tier_source not in {"metadata_column", "parent_folder"}:
        raise ConfigError("tier_source must be 'metadata_column' or 'parent_folder'")
    cross_check_parent_label = raw.get("cross_check_parent_label", False)
    grouping_identity_verified = raw.get("grouping_identity_verified", False)
    if not isinstance(cross_check_parent_label, bool):
        raise ConfigError("cross_check_parent_label must be true or false")
    if not isinstance(grouping_identity_verified, bool):
        raise ConfigError("grouping_identity_verified must be true or false")
    try:
        label_parent_levels = int(raw.get("label_path_parent_levels", 1))
        tier_parent_levels = int(raw.get("tier_path_parent_levels", 2))
    except (TypeError, ValueError) as exc:
        raise ConfigError("folder-parent levels must be positive integers") from exc
    if label_parent_levels <= 0 or tier_parent_levels <= 0:
        raise ConfigError("folder-parent levels must be positive integers")

    root_value = Path(str(raw["dataset_root"]).strip())
    project_base = Path(project_root or path.parent.parent).resolve()
    root = root_value if root_value.is_absolute() else project_base / root_value
    root = root.resolve()
    if require_data_root and not root.is_dir():
        raise ConfigError(f"Dataset directory does not exist: {root}")

    exclusions_value = raw.get("exclusions_file")
    if exclusions_value in (None, ""):
        exclusions_file = None
    elif not isinstance(exclusions_value, str):
        raise ConfigError("exclusions_file must be a path string or null")
    else:
        exclusions_path = Path(exclusions_value.strip())
        exclusions_file = (
            exclusions_path
            if exclusions_path.is_absolute()
            else project_base / exclusions_path
        ).resolve()

    return DatasetConfig(
        model_generation=model_generation,
        dataset_version=dataset_version,
        doi=doi,
        dataset_root=root,
        metadata_filename=str(raw["metadata_filename"]).strip(),
        class_order=label_order,
        tiers=tuple(str(tier).strip() for tier in tiers),
        columns={str(key): str(value) for key, value in column_map.items() if value is not None},
        label_source=label_source,
        tier_source=tier_source,
        label_path_parent_levels=label_parent_levels,
        tier_path_parent_levels=tier_parent_levels,
        cross_check_parent_label=cross_check_parent_label,
        grouping_identity_verified=grouping_identity_verified,
        exclusions_file=exclusions_file,
    )

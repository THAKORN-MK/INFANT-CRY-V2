"""Read-only dataset inspection and validated, content-hashed manifests.

The module deliberately does not infer metadata columns or relabel classes. The
caller must configure the field names after inspecting the copied release.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infantcry_research.config import DatasetConfig


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus"}
RELEASE_MARKER_NAMES = {"release.json", "dataset_release.json", "version.json"}
GROUP_FIELD_PRIORITY = ("infant_id", "session_id", "family_id", "source_group", "clip_id")


class DatasetAuditError(ValueError):
    """Raised when the copied dataset cannot safely become a manifest."""


@dataclass(frozen=True)
class DatasetInspection:
    root: Path
    metadata_files: tuple[str, ...]
    metadata_columns: dict[str, tuple[str, ...]]
    folder_names: tuple[str, ...]
    version_markers: dict[str, str]
    audio_files: tuple[str, ...]

    @property
    def audio_file_count(self) -> int:
        return len(self.audio_files)


@dataclass(frozen=True)
class Record:
    sample_id: str
    path: Path
    relative_path: str
    raw_label: str
    tier: str
    group_id: str | None
    duration_s: float
    sha256: str
    sample_rate: int
    channels: int
    grouping_field: str | None


@dataclass(frozen=True)
class ExcludedRecord:
    sample_id: str
    relative_path: str
    raw_label: str
    tier: str
    group_id: str | None
    grouping_field: str | None
    duration_s: float
    sha256: str
    sample_rate: int
    channels: int
    reason: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_root: Path
    model_generation: str
    dataset_version: str
    doi: str | None
    metadata_filename: str
    records: tuple[Record, ...]
    source_record_count: int
    excluded_records: tuple[ExcludedRecord, ...]
    exclusions_file: Path | None
    exclusion_ledger_sha256: str | None
    class_counts_by_tier: dict[str, dict[str, int]]
    duplicate_family_count: int
    duplicate_record_count: int
    cross_label_duplicate_family_count: int
    source_cross_label_duplicate_family_count: int
    version_evidence_status: str
    version_markers: dict[str, str]
    grouping_scope: str
    grouping_identity_verified: bool
    label_source: str
    cross_check_parent_label: bool
    warnings: tuple[str, ...]


def inspect_dataset(dataset_root: Path, *, metadata_filename: str = "metadata.csv") -> DatasetInspection:
    """Inventory dataset files, metadata headers, directory names, and release markers.

    This is strictly read-only. It does not assume a class/tier directory layout
    and does not decide what any metadata column means.
    """

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise DatasetAuditError(f"Dataset directory does not exist: {root}")

    metadata_paths = sorted(
        path for path in root.rglob(metadata_filename) if path.is_file()
    )
    metadata_files = tuple(path.relative_to(root).as_posix() for path in metadata_paths)
    columns: dict[str, tuple[str, ...]] = {}
    for path in metadata_paths:
        relpath = path.relative_to(root).as_posix()
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None:
                    raise DatasetAuditError(f"Metadata file has no header: {relpath}")
                columns[relpath] = tuple(str(name) for name in reader.fieldnames)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise DatasetAuditError(f"Could not inspect metadata file {relpath}: {exc}") from exc

    folder_names = tuple(
        sorted(
            {
                path.relative_to(root).parts[0]
                for path in root.rglob("*")
                if path.is_dir() and path != root
            }
        )
    )
    audio_files = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
    )

    version_markers: dict[str, str] = {}
    markdown_release_patterns = {
        "README.md": re.compile(
            r"(?im)^\s*\*\*Version:\*\*\s*(v?\d+(?:\.\d+)*)\s*$"
        ),
        "DATASET_DESCRIPTION.md": re.compile(
            r"(?im)^\s*\*\*Release:\*\*\s*(v?\d+(?:\.\d+)*)\b"
        ),
    }
    for filename, pattern in markdown_release_patterns.items():
        path = root / filename
        if not path.is_file():
            continue
        try:
            contents = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            version_markers[filename] = "unreadable"
            continue
        match = pattern.search(contents)
        if match:
            version_markers[filename] = match.group(1).strip()

    for path in sorted(root.rglob("*.json")):
        if path.name.casefold() not in RELEASE_MARKER_NAMES:
            continue
        relpath = path.relative_to(root).as_posix()
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            version_markers[relpath] = "unreadable"
            continue
        value = None
        if isinstance(payload, dict):
            for key in ("dataset_version", "version", "release"):
                if payload.get(key) not in (None, ""):
                    value = payload[key]
                    break
        version_markers[relpath] = str(value).strip() if value is not None else "unspecified"

    return DatasetInspection(
        root=root,
        metadata_files=metadata_files,
        metadata_columns=columns,
        folder_names=folder_names,
        version_markers=version_markers,
        audio_files=audio_files,
    )


def build_manifest(dataset_root: Path, config: DatasetConfig) -> DatasetManifest:
    """Validate metadata and hash every referenced audio file without modifying it."""

    root = Path(dataset_root).resolve()
    if root != Path(config.dataset_root).resolve():
        # A CLI override is allowed, but it must remain a deliberate configured path.
        configured_root = Path(config.dataset_root).resolve()
        if not configured_root.is_dir() or root != configured_root:
            raise DatasetAuditError(
                f"Audit root {root} does not match configured dataset_root {configured_root}"
            )
    inspection = inspect_dataset(root, metadata_filename=config.metadata_filename)
    if not inspection.metadata_files:
        raise DatasetAuditError(
            f"Could not find {config.metadata_filename!r} under dataset root {root}"
        )
    if len(inspection.metadata_files) != 1:
        raise DatasetAuditError(
            "Expected exactly one configured metadata file; found: "
            + ", ".join(inspection.metadata_files)
        )

    relative_metadata = inspection.metadata_files[0]
    headers = inspection.metadata_columns[relative_metadata]
    mapping = config.columns
    required_fields = ["filepath"]
    if config.label_source == "metadata_column":
        required_fields.append("label")
    if config.tier_source == "metadata_column":
        required_fields.append("tier")
    missing_mapping = [field for field in required_fields if not mapping.get(field)]
    if missing_mapping:
        raise DatasetAuditError(
            "Configure metadata column names for: "
            + ", ".join(missing_mapping)
            + f". Inspect observed headers first: {list(headers)!r}"
        )
    missing_headers = [mapping[field] for field in required_fields if mapping[field] not in headers]
    if missing_headers:
        raise DatasetAuditError(
            f"Configured columns {missing_headers!r} are absent; observed headers: {list(headers)!r}"
        )

    _validate_release(config.dataset_version, inspection.version_markers)
    source_records = _read_records(root, relative_metadata, config, headers)
    if not source_records:
        raise DatasetAuditError("Metadata contains no audio records")
    referenced_paths = {record.relative_path for record in source_records}
    unreferenced_paths = sorted(set(inspection.audio_files) - referenced_paths)
    if unreferenced_paths:
        examples = ", ".join(unreferenced_paths[:5])
        suffix = " ..." if len(unreferenced_paths) > 5 else ""
        raise DatasetAuditError(
            f"Found {len(unreferenced_paths)} unreferenced audio files (would be implicit exclusions): "
            f"{examples}{suffix}"
        )

    exclusions, exclusion_ledger_sha256 = _load_exclusion_ledger(config.exclusions_file)
    source_records_by_path = {record.relative_path: record for record in source_records}
    unknown_exclusions = sorted(set(exclusions) - set(source_records_by_path))
    if unknown_exclusions:
        raise DatasetAuditError(
            "Exclusion ledger references paths absent from configured metadata: "
            + ", ".join(unknown_exclusions[:5])
        )
    for relative_path, (expected_sha256, _reason) in exclusions.items():
        record = source_records_by_path[relative_path]
        if record.sha256.casefold() != expected_sha256:
            raise DatasetAuditError(
                f"Exclusion ledger SHA-256 does not match current audio for {relative_path}"
            )

    source_cross_label_families = _cross_label_duplicate_families(source_records)
    excluded_paths = set(exclusions)
    for family in source_cross_label_families:
        family_paths = {record.relative_path for record in family}
        selected_paths = family_paths & excluded_paths
        if selected_paths and selected_paths != family_paths:
            raise DatasetAuditError(
                "Exclusion ledger must exclude the entire conflicting duplicate family; "
                "partial exclusion would retain an unverified label: "
                + ", ".join(sorted(family_paths))
            )

    excluded_records = tuple(
        ExcludedRecord(
            sample_id=record.sample_id,
            relative_path=record.relative_path,
            raw_label=record.raw_label,
            tier=record.tier,
            group_id=record.group_id,
            grouping_field=record.grouping_field,
            duration_s=record.duration_s,
            sha256=record.sha256,
            sample_rate=record.sample_rate,
            channels=record.channels,
            reason=exclusions[record.relative_path][1],
        )
        for record in source_records
        if record.relative_path in exclusions
    )
    records = [record for record in source_records if record.relative_path not in exclusions]
    if not records:
        raise DatasetAuditError("All metadata records are explicitly excluded from analysis")

    expected_labels = set(config.class_order)
    observed_labels = {record.raw_label.strip().casefold() for record in records}
    absent_labels = expected_labels - observed_labels
    if absent_labels:
        raise DatasetAuditError(
            "No records found for configured raw labels: " + ", ".join(sorted(absent_labels))
        )

    counts: dict[str, dict[str, int]] = {}
    for tier in config.tiers:
        tier_records = [record for record in records if record.tier.casefold() == tier.casefold()]
        counts[tier] = dict(
            sorted(Counter(record.raw_label.strip().casefold() for record in tier_records).items())
        )

    by_hash: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    duplicate_families = [family for family in by_hash.values() if len(family) > 1]
    cross_label_families = _cross_label_duplicate_families(records)

    record_grouping_fields = {record.grouping_field for record in records}
    complete_grouping_field = (
        next(iter(record_grouping_fields))
        if len(record_grouping_fields) == 1 and None not in record_grouping_fields
        else None
    )
    warnings: list[str] = []
    if excluded_records:
        warnings.append(
            f"Explicit curation ledger excludes {len(excluded_records)} records from analysis; "
            "source metadata and audio files were preserved."
        )
    if source_cross_label_families and not cross_label_families:
        warnings.append(
            f"The source metadata contained {len(source_cross_label_families)} exact-content "
            "duplicate families with conflicting labels; all family members were excluded "
            "through the configured ledger because labels could not be independently verified."
        )
    if cross_label_families:
        warnings.append(
            f"Found {len(cross_label_families)} exact-content duplicate families with conflicting "
            "labels; split creation and training are blocked until their source labels/audio are reviewed."
        )
    if not inspection.version_markers:
        version_status = "configured_only"
        warnings.append(
            "No release evidence was found in bundled documentation or release markers; the configured dataset version is not independently verified."
        )
    elif any(value.casefold() in {"unreadable", "unspecified"} for value in inspection.version_markers.values()):
        version_status = "incomplete"
        warnings.append("At least one release marker could not verify a dataset version.")
    else:
        version_status = "verified"
    if complete_grouping_field is None:
        grouping_scope = "mixed_or_missing_group_fields; exact_content_hash_guard_required"
        grouping_verified = False
        warnings.append(
            "Group IDs are missing or use mixed fields; exact-content grouping is required and "
            "subject-independent splitting is not established."
        )
    elif config.grouping_identity_verified:
        grouping_scope = f"verified_independent_group:{complete_grouping_field}"
        grouping_verified = True
    else:
        grouping_scope = f"configured_field_unverified:{complete_grouping_field}"
        grouping_verified = False
        warnings.append(
            f"Grouping column {complete_grouping_field!r} is present but its real-world identity semantics require verification."
        )

    return DatasetManifest(
        dataset_root=root,
        model_generation=config.model_generation,
        dataset_version=config.dataset_version,
        doi=config.doi,
        metadata_filename=relative_metadata,
        records=tuple(records),
        source_record_count=len(source_records),
        excluded_records=excluded_records,
        exclusions_file=config.exclusions_file,
        exclusion_ledger_sha256=exclusion_ledger_sha256,
        class_counts_by_tier=counts,
        duplicate_family_count=len(duplicate_families),
        duplicate_record_count=sum(len(family) for family in duplicate_families),
        cross_label_duplicate_family_count=len(cross_label_families),
        source_cross_label_duplicate_family_count=len(source_cross_label_families),
        version_evidence_status=version_status,
        version_markers=inspection.version_markers,
        grouping_scope=grouping_scope,
        grouping_identity_verified=grouping_verified,
        label_source=config.label_source,
        cross_check_parent_label=config.cross_check_parent_label,
        warnings=tuple(warnings),
    )


def _load_exclusion_ledger(
    path: Path | None,
) -> tuple[dict[str, tuple[str, str]], str | None]:
    if path is None:
        return {}, None
    if not path.is_file():
        raise DatasetAuditError(f"Configured exclusion ledger does not exist: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise DatasetAuditError(f"Exclusion ledger has no header: {path}")
            required = {"relative_path", "sha256", "reason"}
            missing = sorted(required - set(reader.fieldnames))
            if missing:
                raise DatasetAuditError(
                    "Exclusion ledger is missing required columns: " + ", ".join(missing)
                )
            entries: dict[str, tuple[str, str]] = {}
            for row_number, row in enumerate(reader, start=2):
                raw_path = (row.get("relative_path") or "").strip().replace("\\", "/")
                sha256 = (row.get("sha256") or "").strip().casefold()
                reason = (row.get("reason") or "").strip()
                path_parts = raw_path.split("/")
                if (
                    not raw_path
                    or Path(raw_path).is_absolute()
                    or any(part in {"", ".", ".."} for part in path_parts)
                ):
                    raise DatasetAuditError(
                        f"Exclusion ledger row {row_number} must use a normalized relative_path"
                    )
                if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
                    raise DatasetAuditError(
                        f"Exclusion ledger row {row_number} must include a 64-character SHA-256"
                    )
                if not reason:
                    raise DatasetAuditError(
                        f"Exclusion ledger row {row_number} must include a non-empty reason"
                    )
                if raw_path in entries:
                    raise DatasetAuditError(
                        f"Exclusion ledger contains a duplicate relative_path: {raw_path}"
                    )
                entries[raw_path] = (sha256, reason)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetAuditError(f"Could not read exclusion ledger {path}: {exc}") from exc
    return entries, _sha256(path)


def _cross_label_duplicate_families(records: list[Record]) -> list[list[Record]]:
    by_hash: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    return [
        family
        for family in by_hash.values()
        if len(family) > 1
        and len({record.raw_label.strip().casefold() for record in family}) > 1
    ]


def _load_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetAuditError(f"Could not read metadata {path}: {exc}") from exc


def _optional_value(row: dict[str, str], column: str | None) -> str | None:
    if not column:
        return None
    value = row.get(column, "")
    return value.strip() if value and value.strip() else None


def _read_records(
    root: Path,
    metadata_relative_path: str,
    config: DatasetConfig,
    headers: tuple[str, ...],
) -> list[Record]:
    rows = _load_rows(root / metadata_relative_path)
    if config.cross_check_parent_label:
        label_column = config.columns.get("label")
        if not label_column or label_column not in headers:
            raise DatasetAuditError(
                "cross_check_parent_label requires a configured label metadata column"
            )
    records: list[Record] = []
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        required_fields = ["filepath"]
        if config.label_source == "metadata_column":
            required_fields.append("label")
        if config.tier_source == "metadata_column":
            required_fields.append("tier")
        for field in required_fields:
            column = config.columns[field]
            value = row.get(column) or ""
            if not value.strip():
                raise DatasetAuditError(
                    f"Metadata row {row_number} has an empty required field {column!r}"
                )
        path_value = (row.get(config.columns["filepath"]) or "").strip().replace("\\", "/")
        folder_label = None
        if config.label_source == "parent_folder" or config.cross_check_parent_label:
            folder_label = _path_parent_value(
                path_value=path_value,
                levels=config.label_path_parent_levels,
                field="label",
                row_number=row_number,
            )
        metadata_label = (
            _optional_value(row, config.columns.get("label"))
            if config.columns.get("label") in headers
            else None
        )
        if config.cross_check_parent_label:
            if not metadata_label:
                raise DatasetAuditError(
                    f"Metadata row {row_number} has no value for its cross-checked label column"
                )
            if metadata_label.strip().casefold() != str(folder_label).strip().casefold():
                raise DatasetAuditError(
                    f"Metadata label {metadata_label!r} conflicts with parent-folder label "
                    f"{folder_label!r} at row {row_number}; inspect the official release"
                )
        raw_label = metadata_label if config.label_source == "metadata_column" else str(folder_label)
        normalized_label = raw_label.strip().casefold()
        if normalized_label not in set(config.class_order):
            raise DatasetAuditError(
                f"Metadata row {row_number} has unrecognized raw label {raw_label!r}; "
                f"allowed: {', '.join(config.class_order)}"
            )
        tier = (
            row[config.columns["tier"]]
            if config.tier_source == "metadata_column"
            else _path_parent_value(
                path_value=row[config.columns["filepath"]],
                levels=config.tier_path_parent_levels,
                field="tier",
                row_number=row_number,
            )
        )
        if tier.strip().casefold() not in {item.casefold() for item in config.tiers}:
            raise DatasetAuditError(
                f"Metadata row {row_number} has unrecognized Tier {tier!r}; "
                f"allowed: {', '.join(config.tiers)}"
            )

        relative = Path(path_value)
        if relative.is_absolute():
            raise DatasetAuditError(
                f"Metadata row {row_number} must use a relative audio path, got {path_value!r}"
            )
        audio_path = (root / relative).resolve()
        try:
            relative_audio_path = audio_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise DatasetAuditError(
                f"Metadata row {row_number} path escapes dataset root: {path_value!r}"
            ) from exc
        if relative_audio_path in seen_paths:
            raise DatasetAuditError(
                f"Metadata references the same audio file more than once: {relative_audio_path}"
            )
        seen_paths.add(relative_audio_path)
        if not audio_path.is_file():
            raise DatasetAuditError(f"Metadata row {row_number} references missing audio file: {relative_audio_path}")

        # clip_id is a grouping identifier in Tier 2, not a globally unique
        # record identifier. Tier 1 clip numbers are ordinal within a class.
        sample_id = relative_audio_path
        if sample_id in seen_ids:
            raise DatasetAuditError(f"Duplicate sample ID {sample_id!r} in metadata")
        seen_ids.add(sample_id)

        group_fields = GROUP_FIELD_PRIORITY
        if tier.strip().casefold() == "dunstan-core":
            # Tier 1 clip_id values are class-local sequence numbers, so they
            # must not tie otherwise unrelated records into the same split.
            group_fields = tuple(field for field in group_fields if field != "clip_id")
        group_field = next(
            (
                field
                for field in group_fields
                if _optional_value(row, config.columns.get(field))
            ),
            None,
        )
        group_value = _optional_value(row, config.columns.get(group_field)) if group_field else None
        group_id = f"{group_field}:{group_value}" if group_field and group_value else None
        sample_rate, channels, duration = _audio_info(audio_path, row_number)
        records.append(
            Record(
                sample_id=sample_id,
                path=audio_path,
                relative_path=relative_audio_path,
                raw_label=raw_label,
                tier=tier,
                group_id=group_id,
                duration_s=duration,
                sha256=_sha256(audio_path),
                sample_rate=sample_rate,
                channels=channels,
                grouping_field=group_field,
            )
        )
    return sorted(records, key=lambda item: item.relative_path)


def _audio_info(path: Path, row_number: int) -> tuple[int, int, float]:
    wave_declared_frames: int | None = None
    if path.suffix.casefold() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                wave_declared_frames = frames
                decoded_frames = 0
                while True:
                    block = handle.readframes(65_536)
                    if not block:
                        break
                    frame_bytes = sample_width * channels
                    if len(block) % frame_bytes:
                        raise DatasetAuditError(
                            f"Metadata row {row_number} contains an incomplete decoded audio frame: {path.name}"
                        )
                    decoded_frames += len(block) // frame_bytes
        except (OSError, wave.Error) as exc:
            raise DatasetAuditError(
                f"Could not decode WAV in metadata row {row_number}: {exc}"
            ) from exc
        if frames <= 0 or rate <= 0 or channels <= 0:
            raise DatasetAuditError(f"Metadata row {row_number} points to empty/invalid audio: {path.name}")
        if decoded_frames != frames:
            raise DatasetAuditError(
                f"Metadata row {row_number} decoded fewer samples than its header declares: "
                f"{decoded_frames} of {frames} frames in {path.name}"
            )

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        if path.suffix.casefold() == ".wav" and wave_declared_frames is not None:
            return rate, channels, frames / rate
        raise DatasetAuditError(
            "soundfile is required to inspect non-WAV audio metadata; install project requirements first"
        ) from exc

    try:
        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise DatasetAuditError(f"Metadata row {row_number} points to empty/invalid audio: {path.name}")
        decoded_frames = 0
        with sf.SoundFile(str(path)) as audio_file:
            while True:
                block = audio_file.read(65_536, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                if not np.isfinite(block).all():
                    raise DatasetAuditError(
                        f"Metadata row {row_number} contains non-finite decoded audio: {path.name}"
                    )
                decoded_frames += int(block.shape[0])
        if decoded_frames != int(info.frames):
            raise DatasetAuditError(
                f"Metadata row {row_number} decoded fewer samples than its header declares: "
                f"{decoded_frames} of {info.frames} frames in {path.name}"
            )
        if wave_declared_frames is not None and wave_declared_frames != decoded_frames:
            raise DatasetAuditError(
                f"Metadata row {row_number} WAV header declares {wave_declared_frames} frames but "
                f"the decoder returned {decoded_frames}: {path.name}"
            )
        return int(info.samplerate), int(info.channels), float(info.duration)
    except Exception as exc:
        if isinstance(exc, DatasetAuditError):
            raise
        raise DatasetAuditError(f"Could not inspect audio in metadata row {row_number}: {exc}") from exc


def _path_parent_value(*, path_value: str, levels: int, field: str, row_number: int) -> str:
    parts = Path(path_value.replace("\\", "/")).parts
    index = len(parts) - levels - 1
    if index < 0 or not parts[index].strip():
        raise DatasetAuditError(
            f"Metadata row {row_number} path does not contain the configured parent folder for {field}"
        )
    return parts[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_release(expected: str, markers: dict[str, str]) -> None:
    if not markers:
        return
    expected_match = _release_components(expected)
    if expected_match is None:
        raise DatasetAuditError(f"Configured dataset version is not recognized: {expected!r}")
    observed: dict[str, tuple[int, ...]] = {}
    for path, value in markers.items():
        version = _release_components(value)
        if version is None:
            raise DatasetAuditError(f"Could not verify release marker {path}: {value!r}")
        observed[path] = version
    if any(major != expected_match for major in observed.values()):
        details = ", ".join(f"{path}={value}" for path, value in markers.items())
        raise DatasetAuditError(
            f"Release evidence {details} conflicts with configured dataset release {expected}"
        )
    if len(set(observed.values())) > 1:
        raise DatasetAuditError("Dataset contains conflicting release/version markers")


def _release_components(value: str) -> tuple[int, ...] | None:
    match = re.fullmatch(r"\s*v?(\d+(?:\.\d+)*)\s*", value.casefold())
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))

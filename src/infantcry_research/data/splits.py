"""Deterministic grouped train/validation/locked-test partitioning."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from infantcry_research.data.manifest import Record


SPLIT_NAMES = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when a safe, class-supported grouped split cannot be made."""


@dataclass(frozen=True)
class SplitManifest:
    split_id: str
    source_manifest_sha256: str
    seed: int
    fractions: dict[str, float]
    group_key: str
    grouping_scope: str
    split_ids: dict[str, tuple[str, ...]]
    group_ids_by_split: dict[str, tuple[str, ...]]
    class_counts: dict[str, dict[str, int]]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SplitAudit:
    valid: bool
    split_sizes: dict[str, int]
    group_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    source_manifest_sha256: str


def make_splits(
    records: Sequence[Record],
    *,
    seed: int,
    group_key: str = "group_id",
    grouping_identity_verified: bool = False,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    search_trials: int = 128,
) -> SplitManifest:
    """Build one approximate stratified split without breaking groups or hashes.

    GroupShuffleSplit is sampled repeatedly, then the complete three-way split
    with the closest class and size proportions is selected. Class labels are
    used only to stratify the partition; they are not features or model inputs.
    """

    if not records:
        raise SplitError("Cannot split an empty manifest")
    if group_key not in {"group_id", "sha256"}:
        raise SplitError("group_key must be 'group_id' or 'sha256'")
    if search_trials < 1:
        raise SplitError("search_trials must be at least 1")
    if len(fractions) != 3 or any(not math.isfinite(value) or value <= 0 for value in fractions):
        raise SplitError("fractions must contain three positive finite values")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SplitError("train, validation, and test fractions must sum to 1")

    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise SplitError("sample_id values must be unique before splitting")

    component_ids = _component_ids(records, group_key)
    labels = [record.raw_label.strip().casefold() for record in records]
    classes = tuple(sorted(set(labels)))
    class_component_counts = {
        label: len({component_ids[i] for i, observed in enumerate(labels) if observed == label})
        for label in classes
    }
    unsupported = {
        label: count for label, count in class_component_counts.items() if count < len(SPLIT_NAMES)
    }
    if unsupported:
        formatted = ", ".join(f"{label}={count}" for label, count in unsupported.items())
        raise SplitError(
            "fewer than three independent groups for required class support in every split: "
            + formatted
        )

    candidate = _search_split(
        records,
        labels,
        component_ids,
        classes,
        fractions,
        seed,
        search_trials,
    )
    if candidate is None:
        raise SplitError(
            "Could not construct train/validation/test splits with every class represented. "
            "The group sizes or class/group distribution may make the requested fractions infeasible."
        )

    split_indices, class_counts = candidate
    id_splits = {
        name: tuple(sorted(records[index].sample_id for index in split_indices[position]))
        for position, name in enumerate(SPLIT_NAMES)
    }
    group_splits = {
        name: tuple(sorted({component_ids[index] for index in split_indices[position]}))
        for position, name in enumerate(SPLIT_NAMES)
    }
    source_hash = _source_manifest_hash(records)
    split_fractions = dict(zip(SPLIT_NAMES, fractions, strict=True))
    split_identity = hashlib.sha256(
        f"{source_hash}|{seed}|{group_key}|{split_fractions}|{id_splits}".encode("utf-8")
    ).hexdigest()[:12]
    grouping_scope, warnings = _grouping_description(
        records, group_key, grouping_identity_verified=grouping_identity_verified
    )
    manifest = SplitManifest(
        split_id=f"seed-{seed}-{split_identity}",
        source_manifest_sha256=source_hash,
        seed=seed,
        fractions=split_fractions,
        group_key=group_key,
        grouping_scope=grouping_scope,
        split_ids=id_splits,
        group_ids_by_split=group_splits,
        class_counts=class_counts,
        warnings=warnings,
    )
    validate_splits(records, manifest)
    return manifest


def validate_splits(
    records: Sequence[Record], split_manifest: SplitManifest
) -> SplitAudit:
    """Verify complete assignment, class support, group isolation, and hash isolation."""

    if split_manifest.group_key not in {"group_id", "sha256"}:
        raise SplitError("Split manifest uses an unsupported group key")
    if set(split_manifest.split_ids) != set(SPLIT_NAMES):
        raise SplitError("Split manifest must define exactly train, validation, and test")
    if _source_manifest_hash(records) != split_manifest.source_manifest_sha256:
        raise SplitError("Split manifest source manifest hash does not match the records")

    by_id = {record.sample_id: record for record in records}
    all_ids = [sample_id for name in SPLIT_NAMES for sample_id in split_manifest.split_ids[name]]
    if len(all_ids) != len(set(all_ids)):
        raise SplitError("A sample appears in more than one split")
    if set(all_ids) != set(by_id):
        missing = sorted(set(by_id) - set(all_ids))
        unknown = sorted(set(all_ids) - set(by_id))
        raise SplitError(f"Split assignment is incomplete or has unknown IDs; missing={missing[:5]}, unknown={unknown[:5]}")

    component_ids = _component_ids(records, split_manifest.group_key)
    component_by_sample = {
        record.sample_id: component_ids[index] for index, record in enumerate(records)
    }
    split_by_sample = {
        sample_id: split_name
        for split_name in SPLIT_NAMES
        for sample_id in split_manifest.split_ids[split_name]
    }
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[component_by_sample[record.sample_id]].add(split_by_sample[record.sample_id])
        hash_splits[record.sha256].add(split_by_sample[record.sample_id])
    if any(len(names) > 1 for names in group_splits.values()):
        raise SplitError("group leakage: a connected group is present in multiple splits")
    if any(len(names) > 1 for names in hash_splits.values()):
        raise SplitError("exact-content leakage: a SHA-256 family crosses splits")

    classes = tuple(sorted({record.raw_label.strip().casefold() for record in records}))
    class_counts: dict[str, dict[str, int]] = {}
    sizes: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for name in SPLIT_NAMES:
        assigned = [by_id[sample_id] for sample_id in split_manifest.split_ids[name]]
        counts = Counter(record.raw_label.strip().casefold() for record in assigned)
        missing_classes = [label for label in classes if counts[label] == 0]
        if missing_classes:
            raise SplitError(f"class support missing from {name}: {', '.join(missing_classes)}")
        class_counts[name] = dict(sorted(counts.items()))
        sizes[name] = len(assigned)
        group_counts[name] = len({component_by_sample[record.sample_id] for record in assigned})

    return SplitAudit(
        valid=True,
        split_sizes=sizes,
        group_counts=group_counts,
        class_counts=class_counts,
        source_manifest_sha256=split_manifest.source_manifest_sha256,
    )


def group_component_ids(records: Sequence[Record], group_key: str) -> dict[str, str]:
    """Map records to joint ID+exact-content components used by split isolation."""

    if group_key not in {"group_id", "sha256"}:
        raise SplitError("group_key must be 'group_id' or 'sha256'")
    components = _component_ids(records, group_key)
    return {record.sample_id: components[index] for index, record in enumerate(records)}


def _component_ids(records: Sequence[Record], group_key: str) -> list[str]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    first_index: dict[str, int] = {}
    for index, record in enumerate(records):
        key_value = getattr(record, group_key, None)
        tokens = [f"sha256:{record.sha256}"]
        if key_value not in (None, ""):
            tokens.append(f"{group_key}:{key_value}")
        for token in tokens:
            if token in first_index:
                union(index, first_index[token])
            else:
                first_index[token] = index

    members: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        members[find(index)].append(record.sample_id)
    component_by_root = {
        root: "group-" + hashlib.sha256("\n".join(sorted(sample_ids)).encode("utf-8")).hexdigest()[:16]
        for root, sample_ids in members.items()
    }
    return [component_by_root[find(index)] for index in range(len(records))]


def _search_split(
    records: Sequence[Record],
    labels: list[str],
    component_ids: list[str],
    classes: tuple[str, ...],
    fractions: tuple[float, float, float],
    seed: int,
    search_trials: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], dict[str, dict[str, int]]] | None:
    indices = np.arange(len(records))
    groups = np.asarray(component_ids, dtype=object)
    x = np.zeros((len(records), 1), dtype=np.uint8)
    all_label_counts = Counter(labels)
    best_score = math.inf
    best_result = None
    val_fraction_of_remainder = fractions[1] / (fractions[0] + fractions[1])
    inner_trials = min(32, max(12, search_trials // 8))

    test_splitter = GroupShuffleSplit(
        n_splits=search_trials, test_size=fractions[2], random_state=seed
    )
    for test_trial, (train_val_indices, test_indices) in enumerate(
        test_splitter.split(x, groups=groups)
    ):
        val_splitter = GroupShuffleSplit(
            n_splits=inner_trials,
            test_size=val_fraction_of_remainder,
            random_state=(seed + 7919 * (test_trial + 1)) % (2**32 - 1),
        )
        for train_local, val_local in val_splitter.split(
            x[train_val_indices], groups=groups[train_val_indices]
        ):
            split_indices = (
                tuple(int(index) for index in train_val_indices[train_local]),
                tuple(int(index) for index in train_val_indices[val_local]),
                tuple(int(index) for index in test_indices),
            )
            counts_by_split: list[Counter[str]] = []
            valid_support = True
            for selected in split_indices:
                counts = Counter(labels[index] for index in selected)
                if any(counts[label] == 0 for label in classes):
                    valid_support = False
                    break
                counts_by_split.append(counts)
            if not valid_support:
                continue
            score = _split_score(counts_by_split, all_label_counts, classes, fractions, len(records), split_indices)
            if score < best_score:
                best_score = score
                class_counts = {
                    name: dict(sorted(counts.items()))
                    for name, counts in zip(SPLIT_NAMES, counts_by_split, strict=True)
                }
                best_result = (split_indices, class_counts)
    return best_result


def _split_score(
    counts_by_split: Sequence[Counter[str]],
    all_label_counts: Counter[str],
    classes: tuple[str, ...],
    fractions: tuple[float, float, float],
    total_count: int,
    split_indices: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> float:
    score = 0.0
    for split_index, fraction in enumerate(fractions):
        for label in classes:
            target = all_label_counts[label] * fraction
            actual = counts_by_split[split_index][label]
            score += ((actual - target) / max(target, 1.0)) ** 2
        target_size = total_count * fraction
        actual_size = len(split_indices[split_index])
        score += 0.5 * ((actual_size - target_size) / max(target_size, 1.0)) ** 2
    return score


def _source_manifest_hash(records: Sequence[Record]) -> str:
    lines = [
        "\t".join(
            (
                record.sample_id,
                record.raw_label.strip().casefold(),
                record.tier.strip().casefold(),
                record.sha256,
                record.group_id or "",
                record.relative_path,
            )
        )
        for record in sorted(records, key=lambda item: item.sample_id)
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _grouping_description(
    records: Sequence[Record],
    group_key: str,
    *,
    grouping_identity_verified: bool = False,
) -> tuple[str, tuple[str, ...]]:
    if group_key == "sha256" or not any(record.group_id for record in records):
        return (
            "exact_content_hash_only",
            ("No usable subject/session group ID is available; this is not subject-independent testing.",),
        )
    fields = {record.grouping_field for record in records if record.group_id}
    if any(record.group_id is None for record in records) or len(fields) != 1:
        return (
            "mixed_or_missing_group_fields+exact_content_hash",
            ("Group fields are mixed/incomplete; exact-content grouping is retained, subject independence is unverified.",),
        )
    field = next(iter(fields))
    if grouping_identity_verified:
        return f"verified_independent_group:{field}", ()
    return (
        f"configured_field_unverified:{field}",
        (f"The {field!r} grouping field is unverified as a real-world subject/session identifier.",),
    )

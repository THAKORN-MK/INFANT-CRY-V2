"""Group-safe partition tests using synthetic manifests only."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from infantcry_research.data.manifest import Record
from infantcry_research.data.splits import SplitError, make_splits, validate_splits


LABELS = ("eairh", "eh", "heh", "neh", "owh")


def _records(*, groups_per_class: int = 20, include_group_ids: bool = True) -> list[Record]:
    records = []
    for group_index in range(groups_per_class):
        for label_index, label in enumerate(LABELS):
            sample_id = f"clip-{group_index:02d}-{label}"
            records.append(
                Record(
                    sample_id=sample_id,
                    path=Path("/synthetic") / f"{sample_id}.wav",
                    relative_path=f"{sample_id}.wav",
                    raw_label=label,
                    tier="InfantCry-1214",
                    group_id=f"family-{group_index}" if include_group_ids else None,
                    duration_s=1.0,
                    sha256=f"{group_index:08x}{label_index:056x}",
                    sample_rate=16_000,
                    channels=1,
                    grouping_field="family_id" if include_group_ids else None,
                )
            )
    return records


class GroupSplitTests(unittest.TestCase):
    def test_generates_deterministic_class_stratified_70_15_15_group_split(self) -> None:
        records = _records()

        split = make_splits(records, seed=42, group_key="group_id")
        audit = validate_splits(records, split)

        self.assertTrue(audit.valid)
        self.assertEqual(split.split_id, make_splits(records, seed=42, group_key="group_id").split_id)
        self.assertEqual(len(split.split_ids["train"]), 70)
        self.assertEqual(len(split.split_ids["validation"]), 15)
        self.assertEqual(len(split.split_ids["test"]), 15)
        self.assertEqual(set(split.class_counts["test"]), set(LABELS))

    def test_group_and_exact_duplicate_families_never_cross_splits(self) -> None:
        records = _records()
        duplicated = replace(
            records[0],
            sample_id="content-copy",
            path=Path("/synthetic/content-copy.wav"),
            relative_path="content-copy.wav",
            group_id="unrelated-family",
        )
        records[1] = replace(records[1], sha256=records[0].sha256)
        records.append(duplicated)

        split = make_splits(records, seed=123, group_key="group_id")
        split_by_id = {
            sample_id: name for name, sample_ids in split.split_ids.items() for sample_id in sample_ids
        }

        self.assertEqual(split_by_id[records[0].sample_id], split_by_id["content-copy"])
        self.assertEqual(split_by_id[records[0].sample_id], split_by_id[records[1].sample_id])
        self.assertTrue(validate_splits(records, split).valid)

    def test_absent_identity_ids_falls_back_to_exact_content_and_warns(self) -> None:
        records = _records(include_group_ids=False)

        split = make_splits(records, seed=2026, group_key="group_id")

        self.assertIn("exact_content_hash", split.grouping_scope)
        self.assertTrue(any("subject-independent" in warning for warning in split.warnings))
        self.assertTrue(validate_splits(records, split).valid)

    def test_group_scope_is_marked_verified_only_after_explicit_confirmation(self) -> None:
        split = make_splits(
            _records(), seed=42, group_key="group_id", grouping_identity_verified=True
        )

        self.assertEqual(split.grouping_scope, "verified_independent_group:family_id")
        self.assertEqual(split.warnings, ())

    def test_fails_when_each_class_lacks_three_independent_groups(self) -> None:
        with self.assertRaisesRegex(SplitError, "fewer than three independent groups"):
            make_splits(_records(groups_per_class=2), seed=42)

    def test_rejects_invalid_split_fractions(self) -> None:
        with self.assertRaisesRegex(SplitError, "sum to 1"):
            make_splits(_records(), seed=42, fractions=(0.7, 0.2, 0.2))

    def test_audit_detects_a_group_moved_between_splits(self) -> None:
        records = _records()
        valid = make_splits(records, seed=42)
        split_ids = {key: list(value) for key, value in valid.split_ids.items()}
        moved = split_ids["train"].pop()
        split_ids["test"].append(moved)
        tampered = replace(valid, split_ids={key: tuple(value) for key, value in split_ids.items()})

        with self.assertRaisesRegex(SplitError, "group leakage|class support|source manifest"):
            validate_splits(records, tampered)


if __name__ == "__main__":
    unittest.main()

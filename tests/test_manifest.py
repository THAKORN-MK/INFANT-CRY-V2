"""Synthetic tests for the DBL audit and manifest contract."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

from infantcry_research.config import DatasetConfig
from infantcry_research.data.manifest import (
    DatasetAuditError,
    build_manifest,
    inspect_dataset,
)
from infantcry_research.data.splits import group_component_ids


LABELS = ("eairh", "eh", "heh", "neh", "owh")


def _write_wav(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = bytes((seed + index) % 256 for index in range(320))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16_000)
        handle.writeframes(samples)


class DatasetManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dbl_v2"
        self.root.mkdir()
        self.metadata = self.root / "metadata.csv"
        self.rows: list[dict[str, str]] = []
        for tier_index, tier in enumerate(("Dunstan-Core", "InfantCry-1214")):
            for label_index, label in enumerate(LABELS):
                relpath = f"{tier}/{label}/{label_index}.wav"
                _write_wav(self.root / relpath, tier_index * 40 + label_index)
                self.rows.append(
                    {
                        "filepath": relpath,
                        "label": label,
                        "tier": tier,
                        "clip_id": f"{tier}-{label}",
                        "infant_id": f"infant-{label_index}",
                        "session_id": f"session-{tier_index}",
                        "family_id": "",
                        "source_group": f"source-{tier}-{label}",
                    }
                )
        self._write_metadata()
        (self.root / "README.md").write_text("**Version:** 1.0\n", encoding="utf-8")
        (self.root / "DATASET_DESCRIPTION.md").write_text(
            "**Release:** v1.0 · 2026-05-29\n", encoding="utf-8"
        )
        self.config = DatasetConfig(
            model_generation="v2",
            dataset_version="1.0",
            doi="10.17632/x493z8nmwc.1",
            dataset_root=self.root,
            metadata_filename="metadata.csv",
            class_order=LABELS,
            tiers=("Dunstan-Core", "InfantCry-1214"),
            columns={
                "filepath": "filepath",
                "label": "label",
                "tier": "tier",
                "clip_id": "clip_id",
                "infant_id": "infant_id",
                "session_id": "session_id",
                "family_id": "family_id",
                "source_group": "source_group",
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_metadata(self) -> None:
        with self.metadata.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_inspection_reports_metadata_columns_audio_and_release_evidence(self) -> None:
        inspection = inspect_dataset(self.root, metadata_filename="metadata.csv")

        self.assertEqual(inspection.metadata_columns["metadata.csv"], tuple(self.rows[0]))
        self.assertEqual(inspection.audio_file_count, len(self.rows))
        self.assertEqual(inspection.version_markers["README.md"], "1.0")
        self.assertEqual(inspection.version_markers["DATASET_DESCRIPTION.md"], "v1.0")
        self.assertIn("Dunstan-Core", inspection.folder_names)
        self.assertIn("InfantCry-1214", inspection.folder_names)

    def test_builds_manifest_without_relabeling_or_dropping_records(self) -> None:
        manifest = build_manifest(self.root, self.config)

        self.assertEqual(len(manifest.records), 10)
        self.assertEqual({record.raw_label for record in manifest.records}, set(LABELS))
        self.assertEqual(
            {record.tier for record in manifest.records}, {"Dunstan-Core", "InfantCry-1214"}
        )
        self.assertEqual(manifest.version_evidence_status, "verified")
        self.assertTrue(all(len(record.sha256) == 64 for record in manifest.records))
        self.assertTrue(all(record.duration_s == 0.02 for record in manifest.records))
        self.assertEqual(manifest.class_counts_by_tier["InfantCry-1214"]["owh"], 1)

    def test_clip_ids_are_not_sample_ids_and_only_group_tier2_records(self) -> None:
        for row in self.rows[:5]:
            row["clip_id"] = "1"
        self.rows[5]["clip_id"] = "shared-infant"
        self.rows[6]["clip_id"] = "shared-infant"
        self._write_metadata()
        config = DatasetConfig(
            **{
                **self.config.__dict__,
                "columns": {
                    "filepath": "filepath",
                    "label": "label",
                    "tier": "tier",
                    "clip_id": "clip_id",
                },
            }
        )

        manifest = build_manifest(self.root, config)

        self.assertEqual(len({record.sample_id for record in manifest.records}), 10)
        expected_sample_ids = {Path(row["filepath"]).as_posix() for row in self.rows}
        self.assertEqual({record.sample_id for record in manifest.records}, expected_sample_ids)

        tier1 = [record for record in manifest.records if record.tier == "Dunstan-Core"]
        self.assertTrue(all(record.group_id is None for record in tier1))
        self.assertTrue(all(record.grouping_field is None for record in tier1))

        shared_tier2_ids = {
            Path(row["filepath"]).as_posix() for row in self.rows[5:7]
        }
        tier2_by_id = {record.sample_id: record for record in manifest.records}
        shared_tier2 = [tier2_by_id[sample_id] for sample_id in shared_tier2_ids]
        self.assertEqual({record.group_id for record in shared_tier2}, {"clip_id:shared-infant"})
        self.assertEqual({record.grouping_field for record in shared_tier2}, {"clip_id"})

        components = group_component_ids(manifest.records, "group_id")
        self.assertEqual(components[shared_tier2[0].sample_id], components[shared_tier2[1].sample_id])
        self.assertEqual(
            len({components[record.sample_id] for record in tier1}),
            len(tier1),
        )

    def test_can_use_official_parent_folder_label_source_when_configured(self) -> None:
        rows_without_label = [{key: value for key, value in row.items() if key != "label"} for row in self.rows]
        with self.metadata.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows_without_label[0]))
            writer.writeheader()
            writer.writerows(rows_without_label)
        config = DatasetConfig(
            **{
                **self.config.__dict__,
                "label_source": "parent_folder",
                "label_path_parent_levels": 1,
                "columns": {
                    key: value for key, value in self.config.columns.items() if key != "label"
                },
            }
        )

        manifest = build_manifest(self.root, config)

        self.assertEqual({record.raw_label for record in manifest.records}, set(LABELS))

    def test_missing_metadata_audio_path_fails_instead_of_silently_dropping(self) -> None:
        self.rows[0]["filepath"] = "Dunstan-Core/eairh/missing.wav"
        self._write_metadata()

        with self.assertRaisesRegex(DatasetAuditError, "missing audio file"):
            build_manifest(self.root, self.config)

    def test_unreferenced_audio_file_is_reported_as_an_implicit_exclusion(self) -> None:
        _write_wav(self.root / "InfantCry-1214/owh/unlisted.wav", seed=91)

        with self.assertRaisesRegex(DatasetAuditError, "unreferenced audio files"):
            build_manifest(self.root, self.config)

    def test_unknown_label_fails_without_mapping_it_to_another_class(self) -> None:
        self.rows[0]["label"] = "crying"
        self._write_metadata()

        with self.assertRaisesRegex(DatasetAuditError, "unrecognized raw label"):
            build_manifest(self.root, self.config)

    def test_conflicting_release_marker_fails_closed(self) -> None:
        (self.root / "README.md").write_text("**Version:** 2.0\n", encoding="utf-8")

        with self.assertRaisesRegex(DatasetAuditError, "conflicts with configured dataset release 1.0"):
            build_manifest(self.root, self.config)

    def test_duplicate_content_is_reported_and_kept_for_group_aware_splitting(self) -> None:
        first = self.root / self.rows[0]["filepath"]
        second = self.root / self.rows[1]["filepath"]
        second.write_bytes(first.read_bytes())

        manifest = build_manifest(self.root, self.config)

        self.assertEqual(len(manifest.records), 10)
        self.assertEqual(manifest.duplicate_family_count, 1)
        self.assertEqual(manifest.duplicate_record_count, 2)
        self.assertEqual(manifest.cross_label_duplicate_family_count, 1)

    def test_explicit_exclusion_ledger_removes_conflicting_family_from_analysis(self) -> None:
        first_row = self.rows[1]  # Tier 1: eh
        second_row = self.rows[7]  # Tier 2: heh
        first_path = self.root / first_row["filepath"]
        second_path = self.root / second_row["filepath"]
        second_path.write_bytes(first_path.read_bytes())
        digest = hashlib.sha256(first_path.read_bytes()).hexdigest()
        reason = "Identical audio has conflicting source labels; no independent adjudication available."
        exclusions_file = self.root.parent / "exclusions.csv"
        with exclusions_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["relative_path", "sha256", "reason"])
            writer.writeheader()
            writer.writerows(
                [
                    {"relative_path": first_row["filepath"], "sha256": digest, "reason": reason},
                    {"relative_path": second_row["filepath"], "sha256": digest, "reason": reason},
                ]
            )
        config = DatasetConfig(**{**self.config.__dict__, "exclusions_file": exclusions_file})

        manifest = build_manifest(self.root, config)

        self.assertEqual(manifest.source_record_count, 10)
        self.assertEqual(len(manifest.records), 8)
        self.assertEqual(manifest.source_cross_label_duplicate_family_count, 1)
        self.assertEqual(manifest.cross_label_duplicate_family_count, 0)
        self.assertEqual(
            {item.relative_path for item in manifest.excluded_records},
            {first_row["filepath"], second_row["filepath"]},
        )
        self.assertTrue(all(item.reason == reason for item in manifest.excluded_records))
        self.assertTrue(first_path.is_file())
        self.assertTrue(second_path.is_file())

    def test_exclusion_ledger_must_exclude_every_record_in_conflicting_family(self) -> None:
        first_row = self.rows[1]  # Tier 1: eh
        second_row = self.rows[7]  # Tier 2: heh
        first_path = self.root / first_row["filepath"]
        second_path = self.root / second_row["filepath"]
        second_path.write_bytes(first_path.read_bytes())
        digest = hashlib.sha256(first_path.read_bytes()).hexdigest()
        exclusions_file = self.root.parent / "exclusions.csv"
        with exclusions_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["relative_path", "sha256", "reason"])
            writer.writeheader()
            writer.writerow(
                {
                    "relative_path": first_row["filepath"],
                    "sha256": digest,
                    "reason": "Unresolved conflicting labels.",
                }
            )
        config = DatasetConfig(**{**self.config.__dict__, "exclusions_file": exclusions_file})

        with self.assertRaisesRegex(DatasetAuditError, "entire conflicting duplicate family"):
            build_manifest(self.root, config)

    def test_exclusion_ledger_rejects_stale_audio_hash(self) -> None:
        row = self.rows[1]
        exclusions_file = self.root.parent / "exclusions.csv"
        with exclusions_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["relative_path", "sha256", "reason"])
            writer.writeheader()
            writer.writerow(
                {
                    "relative_path": row["filepath"],
                    "sha256": "0" * 64,
                    "reason": "Unresolved conflicting labels.",
                }
            )
        config = DatasetConfig(**{**self.config.__dict__, "exclusions_file": exclusions_file})

        with self.assertRaisesRegex(DatasetAuditError, "SHA-256 does not match"):
            build_manifest(self.root, config)

    def test_rejects_audio_that_has_a_valid_header_but_truncated_payload(self) -> None:
        audio_path = self.root / self.rows[0]["filepath"]
        audio_path.write_bytes(audio_path.read_bytes()[:-12])

        with self.assertRaises(DatasetAuditError):
            build_manifest(self.root, self.config)

    def test_optional_parent_folder_cross_check_rejects_conflicting_metadata_label(self) -> None:
        self.rows[0]["label"] = "eh"
        self._write_metadata()
        config = DatasetConfig(
            **{**self.config.__dict__, "cross_check_parent_label": True}
        )

        with self.assertRaisesRegex(DatasetAuditError, "Metadata label .* conflicts with parent-folder"):
            build_manifest(self.root, config)

    def test_missing_release_evidence_is_visible_as_unverified(self) -> None:
        (self.root / "README.md").unlink()
        (self.root / "DATASET_DESCRIPTION.md").unlink()

        manifest = build_manifest(self.root, self.config)

        self.assertEqual(manifest.version_evidence_status, "configured_only")
        self.assertTrue(any("release evidence" in warning for warning in manifest.warnings))

    def test_rejects_metadata_path_escape(self) -> None:
        self.rows[0]["filepath"] = "../../outside.wav"
        self._write_metadata()

        with self.assertRaisesRegex(DatasetAuditError, "escapes dataset root"):
            build_manifest(self.root, self.config)

    def test_missing_required_column_reports_observed_headers(self) -> None:
        config = DatasetConfig(
            **{
                **self.config.__dict__,
                "columns": {"filepath": "filepath", "label": "absent", "tier": "tier"},
            }
        )

        with self.assertRaisesRegex(DatasetAuditError, "observed headers"):
            build_manifest(self.root, config)


if __name__ == "__main__":
    unittest.main()

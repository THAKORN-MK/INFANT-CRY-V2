"""CLI tests for read-only inspection and validated manifest writes."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import yaml

from infantcry_research.cli import main


LABELS = ["eairh", "eh", "heh", "neh", "owh"]


class AuditCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.output = self.base / "generated_manifest"
        self.root.mkdir()
        self.rows = []
        for tier_index, tier in enumerate(("Dunstan-Core", "InfantCry-1214")):
            for index, label in enumerate(LABELS):
                relative = f"{tier}/{label}/{label}.wav"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16_000)
                    handle.writeframes(bytes([tier_index * len(LABELS) + index + 1, 0]) * 320)
                self.rows.append(
                    {
                        "filepath": relative,
                        "label": label,
                        "tier": tier,
                        "infant_id": f"infant-{index}",
                    }
                )
        with (self.root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (self.root / "README.md").write_text("**Version:** 1.0\n", encoding="utf-8")
        (self.root / "DATASET_DESCRIPTION.md").write_text(
            "**Release:** v1.0 · 2026-05-29\n", encoding="utf-8"
        )
        self.config = self.base / "dataset.yaml"
        self._write_config()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_config(
        self,
        columns: dict[str, str | None] | None = None,
        exclusions_file: Path | None = None,
    ) -> None:
        payload = {
            "model_generation": "v2",
            "dataset_version": "1.0",
            "doi": "10.17632/x493z8nmwc.1",
            "dataset_root": str(self.root),
            "metadata_filename": "metadata.csv",
            "class_order": LABELS,
            "tiers": ["Dunstan-Core", "InfantCry-1214"],
            "exclusions_file": str(exclusions_file) if exclusions_file else None,
            "columns": columns
            or {"filepath": "filepath", "label": "label", "tier": "tier", "infant_id": "infant_id"},
        }
        self.config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    def _invoke(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = main(list(args))
        return result, stdout.getvalue(), stderr.getvalue()

    def test_audit_writes_json_and_csv_only_after_validation_passes(self) -> None:
        status, stdout, stderr = self._invoke(
            "audit",
            "--dataset-root",
            str(self.root),
            "--config",
            str(self.config),
            "--output-dir",
            str(self.output),
        )

        self.assertEqual(status, 0)
        self.assertIn("semantics require verification", stderr)
        audit = json.loads((self.output / "dataset_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["record_count"], 10)
        self.assertEqual(audit["model_generation"], "v2")
        self.assertEqual(audit["dataset_version"], "1.0")
        self.assertEqual(audit["version_evidence_status"], "verified")
        self.assertTrue((self.output / "records.csv").is_file())
        self.assertIn("10 records", stdout)

    def test_audit_reports_explicit_exclusions_separately(self) -> None:
        first_row = self.rows[1]  # Tier 1: eh
        second_row = self.rows[7]  # Tier 2: heh
        first_path = self.root / first_row["filepath"]
        second_path = self.root / second_row["filepath"]
        second_path.write_bytes(first_path.read_bytes())
        digest = hashlib.sha256(first_path.read_bytes()).hexdigest()
        exclusions_file = self.base / "curation" / "exclusions.csv"
        exclusions_file.parent.mkdir()
        with exclusions_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["relative_path", "sha256", "reason"])
            writer.writeheader()
            for row in (first_row, second_row):
                writer.writerow(
                    {
                        "relative_path": row["filepath"],
                        "sha256": digest,
                        "reason": "Unresolved exact-content duplicate label conflict.",
                    }
                )
        self._write_config(exclusions_file=exclusions_file)

        status, stdout, stderr = self._invoke(
            "audit",
            "--dataset-root",
            str(self.root),
            "--config",
            str(self.config),
            "--output-dir",
            str(self.output),
        )

        self.assertEqual(status, 0, stderr)
        payload = json.loads((self.output / "dataset_audit.json").read_text(encoding="utf-8"))
        with (self.output / "excluded_records.csv").open(
            "r", newline="", encoding="utf-8"
        ) as stream:
            excluded_rows = list(csv.DictReader(stream))
        self.assertEqual(payload["source_record_count"], 10)
        self.assertEqual(payload["record_count"], 8)
        self.assertEqual(payload["excluded_record_count"], 2)
        self.assertEqual(payload["source_cross_label_duplicate_family_count"], 1)
        self.assertEqual(payload["cross_label_duplicate_family_count"], 0)
        self.assertEqual(len(excluded_rows), 2)
        self.assertTrue(all(row["reason"] for row in excluded_rows))
        self.assertIn("8 records for analysis", stdout)

    def test_audit_refuses_to_overwrite_existing_excluded_records_without_flag(self) -> None:
        self.output.mkdir()
        excluded_path = self.output / "excluded_records.csv"
        excluded_path.write_text("preserve", encoding="utf-8")

        status, _stdout, stderr = self._invoke(
            "audit",
            "--dataset-root",
            str(self.root),
            "--config",
            str(self.config),
            "--output-dir",
            str(self.output),
        )

        self.assertEqual(status, 2)
        self.assertIn("already exists", stderr)
        self.assertEqual(excluded_path.read_text(encoding="utf-8"), "preserve")

    def test_unmapped_columns_print_headers_and_do_not_create_manifest_files(self) -> None:
        self._write_config({"filepath": None, "label": None, "tier": None})

        status, _stdout, stderr = self._invoke(
            "audit",
            "--dataset-root",
            str(self.root),
            "--config",
            str(self.config),
            "--output-dir",
            str(self.output),
        )

        self.assertEqual(status, 2)
        self.assertIn("observed headers", stderr)
        self.assertIn("filepath", stderr)
        self.assertFalse(self.output.exists())

    def test_inspect_command_shows_observed_headers_without_writing_files(self) -> None:
        status, stdout, stderr = self._invoke(
            "inspect-dataset", "--dataset-root", str(self.root), "--metadata-filename", "metadata.csv"
        )

        payload = json.loads(stdout)
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(payload["audio_file_count"], 10)
        self.assertIn("infant_id", payload["metadata_columns"]["metadata.csv"])
        self.assertFalse(self.output.exists())

    def test_audit_refuses_to_overwrite_existing_manifest_without_flag(self) -> None:
        self.output.mkdir()
        (self.output / "dataset_audit.json").write_text("keep", encoding="utf-8")

        status, _stdout, stderr = self._invoke(
            "audit",
            "--dataset-root",
            str(self.root),
            "--config",
            str(self.config),
            "--output-dir",
            str(self.output),
        )

        self.assertEqual(status, 2)
        self.assertIn("already exists", stderr)
        self.assertEqual(
            (self.output / "dataset_audit.json").read_text(encoding="utf-8"), "keep"
        )


if __name__ == "__main__":
    unittest.main()

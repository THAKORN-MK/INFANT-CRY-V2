"""CLI lifecycle tests for consuming a completed dataset audit."""

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


class SplitCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.audit_dir = self.base / "audit"
        self.split_dir = self.base / "splits"
        self.root.mkdir()
        rows = []
        for group_index in range(20):
            for label_index, label in enumerate(LABELS):
                relative = f"InfantCry-1214/{label}/{group_index:02d}-{label}.wav"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                seed = group_index * len(LABELS) + label_index + 1
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16_000)
                    handle.writeframes(bytes([seed, 0]) * 320)
                rows.append(
                    {
                        "filepath": relative,
                        "label": label,
                        "tier": "InfantCry-1214",
                        "infant_id": f"infant-{group_index:02d}",
                    }
                )
        with (self.root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (self.root / "README.md").write_text("**Version:** 1.0\n", encoding="utf-8")
        (self.root / "DATASET_DESCRIPTION.md").write_text(
            "**Release:** v1.0 · 2026-05-29\n", encoding="utf-8"
        )
        self.config = self.base / "dataset.yaml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "model_generation": "v2",
                    "dataset_version": "1.0",
                    "doi": "10.17632/x493z8nmwc.1",
                    "dataset_root": str(self.root),
                    "metadata_filename": "metadata.csv",
                    "class_order": LABELS,
                    "tiers": ["Dunstan-Core", "InfantCry-1214"],
                    "columns": {
                        "filepath": "filepath",
                        "label": "label",
                        "tier": "tier",
                        "infant_id": "infant_id",
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _invoke(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                result = main(list(args))
            except SystemExit as exc:
                result = int(exc.code)
        return result, stdout.getvalue(), stderr.getvalue()

    def _audit(self, *, confirm_release: bool = False) -> tuple[int, str]:
        command = [
            "audit",
            "--dataset-root",
            str(self.root),
            "--config",
            str(self.config),
            "--output-dir",
            str(self.audit_dir),
        ]
        if confirm_release:
            command.append("--confirm-release")
        status, _stdout, stderr = self._invoke(*command)
        self.assertEqual(status, 0)
        return status, stderr

    def _make_splits(self) -> tuple[int, str, str]:
        return self._invoke(
            "make-splits",
            "--dataset-root",
            str(self.root),
            "--audit-dir",
            str(self.audit_dir),
            "--output-dir",
            str(self.split_dir),
            "--tier",
            "InfantCry-1214",
            "--seed",
            "42",
            "--group-key",
            "group_id",
        )

    def test_make_splits_consumes_hash_verified_audit_and_persists_locked_split(self) -> None:
        self._audit()

        status, stdout, stderr = self._make_splits()

        self.assertEqual(status, 0)
        self.assertIn("unverified as a real-world", stderr)
        self.assertIn("train=70", stdout)
        saved = list(self.split_dir.glob("*/split_manifest.json"))
        self.assertEqual(len(saved), 1)
        payload = json.loads(saved[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["tier"], "InfantCry-1214")
        self.assertTrue(payload["locked_test"])
        self.assertEqual(len(payload["split_ids"]["test"]), 15)
        training_manifest_path = saved[0].with_name("train_validation_records.csv")
        self.assertTrue(training_manifest_path.is_file())
        self.assertEqual(
            hashlib.sha256(training_manifest_path.read_bytes()).hexdigest(),
            payload["train_validation_records_sha256"],
        )
        with training_manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
            training_rows = list(csv.DictReader(stream))
        expected_train_validation = set(payload["split_ids"]["train"]) | set(
            payload["split_ids"]["validation"]
        )
        self.assertEqual({row["sample_id"] for row in training_rows}, expected_train_validation)
        self.assertTrue(
            set(payload["split_ids"]["test"]).isdisjoint(
                {row["sample_id"] for row in training_rows}
            )
        )

    def test_split_requires_verified_release_evidence(self) -> None:
        (self.root / "README.md").unlink()
        (self.root / "DATASET_DESCRIPTION.md").unlink()
        self._audit()

        status, _stdout, stderr = self._make_splits()

        self.assertEqual(status, 2)
        self.assertIn("release evidence is not verified", stderr)
        self.assertFalse(self.split_dir.exists())

    def test_split_accepts_audited_dataset_after_explicit_release_confirmation(self) -> None:
        (self.root / "README.md").unlink()
        (self.root / "DATASET_DESCRIPTION.md").unlink()
        self._audit(confirm_release=True)

        audit_payload = json.loads((self.audit_dir / "dataset_audit.json").read_text(encoding="utf-8"))
        status, _stdout, stderr = self._make_splits()

        self.assertEqual(status, 0)
        self.assertEqual(audit_payload["version_evidence_status"], "verified_by_user")
        self.assertEqual(audit_payload["release_verification_mode"], "user_confirmed")

    def test_split_rejects_audio_changed_since_audit(self) -> None:
        self._audit()
        wav_path = next(self.root.rglob("*.wav"))
        wav_path.write_bytes(wav_path.read_bytes() + b"changed")

        status, _stdout, stderr = self._make_splits()

        self.assertEqual(status, 2)
        self.assertIn("SHA-256 changed since dataset audit", stderr)
        self.assertFalse(self.split_dir.exists())

    def test_split_refuses_exact_content_duplicates_with_conflicting_labels(self) -> None:
        first = self.root / "InfantCry-1214/eairh/00-eairh.wav"
        conflicting = self.root / "InfantCry-1214/eh/00-eh.wav"
        conflicting.write_bytes(first.read_bytes())
        self._audit()

        status, _stdout, stderr = self._make_splits()

        self.assertEqual(status, 2)
        self.assertIn("cross-label exact-content duplicate", stderr)
        self.assertFalse(self.split_dir.exists())

    def test_split_is_immutable_on_rerun(self) -> None:
        self._audit()
        first_status, _stdout, _stderr = self._make_splits()
        self.assertEqual(first_status, 0)

        status, _stdout, stderr = self._make_splits()

        self.assertEqual(status, 2)
        self.assertIn("already exists", stderr)


if __name__ == "__main__":
    unittest.main()

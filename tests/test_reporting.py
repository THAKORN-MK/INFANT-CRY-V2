"""Report rendering tests for immutable paired comparison evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from infantcry_research.reporting.build_report import build_comparison_report
from infantcry_research.training.run_store import RunStoreError


class ComparisonReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package = self.root / "comparison" / "synthetic-split"
        self.package.mkdir(parents=True)
        (self.package / "candidates" / "cnn").mkdir(parents=True)
        (self.package / "candidates" / "mmt_reslstm").mkdir(parents=True)
        (self.package / "protocol.json").write_text(
            json.dumps({"protocol_hash": "p" * 64}), encoding="utf-8"
        )
        candidates = []
        for candidate_id, selected in (("cnn", False), ("mmt_reslstm", True)):
            candidate_dir = self.package / "candidates" / candidate_id
            (candidate_dir / "confusion_matrix.csv").write_text("truth,prediction\n", encoding="utf-8")
            (candidate_dir / "confusion_matrix.png").write_bytes(b"synthetic-image")
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "run_id": f"run-{candidate_id}",
                    "selected_by_validation": selected,
                    "validation_macro_f1": 0.80 if selected else 0.70,
                    "test_metrics": {
                        "accuracy": 0.75,
                        "balanced_accuracy": 0.74,
                        "macro_f1": 0.73,
                        "weighted_f1": 0.75,
                    },
                    "artifacts": f"candidates/{candidate_id}",
                }
            )
        summary = {
            "status": "complete",
            "evaluation_scope": "single_locked_internal_test_batch",
            "split_id": "synthetic-split",
            "selection_id": "selection-synthetic-split",
            "selection_sha256": "s" * 64,
            "protocol_hash": "p" * 64,
            "test_support": 12,
            "selected_by_validation": {
                "candidate_id": "mmt_reslstm",
                "run_id": "run-mmt_reslstm",
            },
            "test_scores_used_for_selection": False,
            "candidate_order": ["cnn", "mmt_reslstm"],
            "candidates": candidates,
        }
        (self.package / "comparison.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        files = {
            path.relative_to(self.package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.package.rglob("*"))
            if path.is_file()
        }
        manifest_bytes = json.dumps(
            {
                "selection_sha256": "s" * 64,
                "protocol_hash": "p" * 64,
                "files": files,
            },
            indent=2,
        ).encode("utf-8")
        (self.package / "artifact_hashes.json").write_bytes(manifest_bytes)
        self.ledger_dir = self.root / "manifests" / "test_access"
        self.ledger_dir.mkdir(parents=True)
        (self.ledger_dir / "synthetic-split.json").write_text(
            json.dumps(
                {
                    "state": "evaluated",
                    "comparison_dir": str(self.package.resolve()),
                    "selection_sha256": "s" * 64,
                    "protocol_hash": "p" * 64,
                    "evaluation_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_report_identifies_validation_selection_and_all_candidates(self) -> None:
        report = build_comparison_report(
            self.package, self.root / "reports", ledger_root=self.ledger_dir
        )
        content = report.read_text(encoding="utf-8")
        self.assertIn("Selected using validation", content)
        self.assertIn("mmt_reslstm", content)
        self.assertIn("cnn", content)
        self.assertIn("Test scores were not used for selection", content)
        self.assertIn("paired descriptive results only", content)
        self.assertIn("confusion_matrix.png", content)

    def test_report_rejects_changed_comparison_artifact(self) -> None:
        predictions = self.package / "candidates" / "cnn" / "confusion_matrix.csv"
        predictions.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(RunStoreError, "artifact hash mismatch"):
            build_comparison_report(
                self.package, self.root / "reports", ledger_root=self.ledger_dir
            )


if __name__ == "__main__":
    unittest.main()

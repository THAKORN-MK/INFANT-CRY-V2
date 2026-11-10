"""Immutable lifecycle tests for experiment run records."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from infantcry_research.training.run_store import (
    RunStoreError,
    claim_locked_split_access,
    create_run,
    load_run,
)


class RunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "runs"
        self.config = {"model": "cnn_bilstm_attention", "seed": 42}
        self.split = {"split_id": "seed-42-abc123", "locked_test": True}
        self.environment = {"python": "3.10.11", "device": "cpu"}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_creates_unique_run_with_frozen_config_split_and_environment_hashes(self) -> None:
        first = create_run(
            self.root,
            config=self.config,
            split_manifest=self.split,
            environment=self.environment,
            candidate_ids=("cnn_bilstm_attention",),
        )
        second = create_run(
            self.root,
            config=self.config,
            split_manifest=self.split,
            environment=self.environment,
            candidate_ids=("cnn_bilstm_attention",),
        )

        self.assertNotEqual(first.run_id, second.run_id)
        self.assertTrue((first.run_dir / "config.json").is_file())
        self.assertTrue((first.run_dir / "split_manifest.json").is_file())
        self.assertTrue((first.run_dir / "environment.json").is_file())
        self.assertEqual(len(first.hashes["config_sha256"]), 64)
        self.assertEqual(len(first.hashes["split_sha256"]), 64)
        self.assertEqual(first.status, "prepared")

    def test_refuses_to_reuse_a_run_id(self) -> None:
        create_run(
            self.root,
            config=self.config,
            split_manifest=self.split,
            environment=self.environment,
            candidate_ids=("cnn_bilstm_attention",),
            run_id="manual-run-001",
        )

        with self.assertRaisesRegex(RunStoreError, "already exists"):
            create_run(
                self.root,
                config=self.config,
                split_manifest=self.split,
                environment=self.environment,
                candidate_ids=("cnn_bilstm_attention",),
                run_id="manual-run-001",
            )

    def test_locked_test_requires_completed_run_and_preregistered_candidate(self) -> None:
        run = create_run(
            self.root,
            config=self.config,
            split_manifest=self.split,
            environment=self.environment,
            candidate_ids=("cnn_bilstm_attention", "mfcc_svm"),
            run_id="test-gate",
        )
        with self.assertRaisesRegex(RunStoreError, "completed run"):
            run.record_test_evaluation("cnn_bilstm_attention", "protocol-a", "eval-a")

        run.mark_training_started()
        run.mark_training_complete(selected_candidate_id="cnn_bilstm_attention")
        with self.assertRaisesRegex(RunStoreError, "not preregistered"):
            run.record_test_evaluation("new_candidate", "protocol-a", "eval-a")

        with self.assertRaisesRegex(RunStoreError, "Claim the locked-test"):
            run.record_test_evaluation("cnn_bilstm_attention", "protocol-a", "eval-a")
        run.begin_test_evaluation("cnn_bilstm_attention", "protocol-a")
        run.record_test_evaluation("cnn_bilstm_attention", "protocol-a", "eval-a")
        with self.assertRaisesRegex(RunStoreError, "already evaluated"):
            run.record_test_evaluation("cnn_bilstm_attention", "protocol-a", "eval-b")

        with self.assertRaisesRegex(RunStoreError, "already accessed or claimed"):
            run.begin_test_evaluation("cnn_bilstm_attention", "protocol-a")

    def test_only_validation_selected_candidate_can_claim_locked_test(self) -> None:
        run = create_run(
            self.root,
            config=self.config,
            split_manifest=self.split,
            environment=self.environment,
            candidate_ids=("cnn_bilstm_attention", "mfcc_svm"),
            run_id="selection-guard",
        )
        run.mark_training_started()
        run.mark_training_complete(selected_candidate_id="cnn_bilstm_attention")

        with self.assertRaisesRegex(RunStoreError, "selected from validation"):
            run.begin_test_evaluation("mfcc_svm", "protocol-a")

    def test_load_run_verifies_frozen_config_split_and_environment(self) -> None:
        run = create_run(
            self.root,
            config=self.config,
            split_manifest=self.split,
            environment=self.environment,
            candidate_ids=("cnn_bilstm_attention",),
            run_id="verified-load",
        )

        loaded = load_run(run.run_dir)
        self.assertEqual(loaded.run_id, "verified-load")
        (run.run_dir / "split_manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RunStoreError, "evidence hash mismatch"):
            load_run(run.run_dir)

    def test_split_test_access_is_global_and_one_shot_across_runs(self) -> None:
        ledger = self.root / "test-access"
        first = claim_locked_split_access(
            ledger,
            split_id="seed-42-split-a",
            candidate_id="cnn_bilstm_attention",
            run_id="run-a",
            protocol_hash="protocol-a",
        )

        self.assertTrue(first.is_file())
        with self.assertRaisesRegex(RunStoreError, "already been claimed or evaluated"):
            claim_locked_split_access(
                ledger,
                split_id="seed-42-split-a",
                candidate_id="mfcc_svm",
                run_id="run-b",
                protocol_hash="protocol-b",
            )

    def test_batch_test_claim_freezes_selection_and_exact_candidate_runs(self) -> None:
        from infantcry_research.training.run_store import (
            claim_locked_test_batch_access,
            complete_locked_test_batch_access,
        )

        candidates = [
            {
                "candidate_id": "majority",
                "run_id": "run-majority",
                "config_sha256": "a" * 64,
                "feature_config_sha256": "b" * 64,
            },
            {
                "candidate_id": "mmt_reslstm",
                "run_id": "run-mmt",
                "config_sha256": "c" * 64,
                "feature_config_sha256": "d" * 64,
            },
        ]
        claim = claim_locked_test_batch_access(
            self.root / "test-access-batch",
            split_id="seed-42-split-b",
            selection_id="selection-seed-42-split-b",
            selection_sha256="e" * 64,
            candidate_runs=candidates,
            protocol_hash="f" * 64,
        )

        frozen = json.loads(claim.read_text(encoding="utf-8"))
        self.assertEqual(frozen["selection_sha256"], "e" * 64)
        self.assertEqual(frozen["candidate_runs"], candidates)
        self.assertEqual(frozen["state"], "claimed_before_test_access")
        comparison_dir = self.root / "comparison"
        comparison_dir.mkdir()
        complete_locked_test_batch_access(
            claim,
            selection_sha256="e" * 64,
            protocol_hash="f" * 64,
            evaluation_sha256="1" * 64,
            comparison_dir=comparison_dir,
        )
        completed = json.loads(claim.read_text(encoding="utf-8"))
        self.assertEqual(completed["state"], "evaluated")
        self.assertEqual(completed["evaluation_sha256"], "1" * 64)
        with self.assertRaisesRegex(RunStoreError, "already been completed"):
            complete_locked_test_batch_access(
                claim,
                selection_sha256="e" * 64,
                protocol_hash="f" * 64,
                evaluation_sha256="2" * 64,
                comparison_dir=comparison_dir,
            )
        changed = [candidates[1], candidates[0]]
        with self.assertRaisesRegex(RunStoreError, "already been claimed or evaluated"):
            claim_locked_test_batch_access(
                self.root / "test-access-batch",
                split_id="seed-42-split-b",
                selection_id="selection-seed-42-split-b",
                selection_sha256="e" * 64,
                candidate_runs=changed,
                protocol_hash="f" * 64,
            )

    def test_cannot_change_validation_selection_after_training_completes(self) -> None:
        run = create_run(
            self.root,
            config=self.config,
            split_manifest=self.split,
            environment=self.environment,
            candidate_ids=("cnn_bilstm_attention",),
        )
        run.mark_training_started()
        run.mark_training_complete(selected_candidate_id="cnn_bilstm_attention")

        with self.assertRaisesRegex(RunStoreError, "already complete"):
            run.mark_training_complete(selected_candidate_id="mfcc_svm")


if __name__ == "__main__":
    unittest.main()

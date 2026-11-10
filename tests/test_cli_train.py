"""CLI tests for safe, validation-only candidate training."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml
import numpy as np

from infantcry_research.cli import DatasetAuditError, main


LABELS = ["eairh", "eh", "heh", "neh", "owh"]


class TrainCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.audit_dir = self.base / "audit"
        self.splits_dir = self.base / "splits"
        self.runs_dir = self.base / "runs"
        self.root.mkdir()
        rows = []
        for group_index in range(20):
            for label_index, label in enumerate(LABELS):
                relative = f"InfantCry-1214/{label}/{group_index:02d}-{label}.wav"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                sample = (group_index * len(LABELS) + label_index + 1) % 250 + 1
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16_000)
                    handle.writeframes(bytes([sample, 0]) * 320)
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
        self.dataset_config = self.base / "dataset.yaml"
        self.dataset_config.write_text(
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
        self.model_config = self.base / "models.yaml"
        self.model_config.write_text(
            yaml.safe_dump(
                {
                    "class_order": LABELS,
                    "primary_tier": "InfantCry-1214",
                    "input_shape": [196, 128, 1],
                    "features": {
                        "sample_rate": 16_000,
                        "sample_rate_status": "locked",
                        "mono": True,
                        "trim_top_db": 20,
                        "normalize_waveform": True,
                        "n_mfcc": 40,
                        "n_mels": 64,
                        "n_chroma": 12,
                        "n_fft": 2048,
                        "hop_length": 512,
                        "delta_width": 9,
                        "max_frames": 128,
                        "feature_order": ["mfcc", "delta", "delta2", "logmel", "chroma"],
                    },
                    "models": [
                        {"id": "majority", "enabled": True, "device": "cpu", "input": "labels_only"}
                    ],
                    "training": {
                        "epochs": 2,
                        "batch_size": 4,
                        "learning_rate": 0.001,
                        "early_stopping_patience": 1,
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
                status = main(list(args))
            except SystemExit as exc:
                status = int(exc.code)
        return status, stdout.getvalue(), stderr.getvalue()

    def _prepare(self) -> Path:
        status, _stdout, stderr = self._invoke(
            "audit",
            "--dataset-root",
            str(self.root),
            "--config",
            str(self.dataset_config),
            "--output-dir",
            str(self.audit_dir),
        )
        self.assertEqual(status, 0, stderr)
        status, stdout, stderr = self._invoke(
            "make-splits",
            "--dataset-root",
            str(self.root),
            "--audit-dir",
            str(self.audit_dir),
            "--output-dir",
            str(self.splits_dir),
            "--seed",
            "42",
        )
        self.assertEqual(status, 0, stderr)
        self.assertIn("Frozen split", stdout)
        return next(self.splits_dir.glob("*/split_manifest.json"))

    def test_qiu_candidates_dispatch_and_select_without_loading_test_records(self) -> None:
        split_path = self._prepare()
        candidates = (
            "qiu_mfcc_reslstm",
            "qiu_mmt_resnet",
            "qiu_mmt_lstm",
            "qiu_mmt_reslstm",
        )
        mfcc = {
            "feature_set": "mfcc_paper",
            "sample_rate": 16_000,
            "sample_rate_status": "locked",
            "mono": True,
            "trim_top_db": 20,
            "normalize_waveform": True,
            "n_mfcc": 42,
            "n_mels": 42,
            "n_fft": 512,
            "hop_length": 128,
            "mfcc_frames": 24,
        }
        mmt = {
            "feature_set": "mmt",
            "sample_rate": 16_000,
            "sample_rate_status": "locked",
            "mono": True,
            "trim_top_db": 20,
            "normalize_waveform": True,
            "n_mfcc": 42,
            "n_mels": 42,
            "n_fft": 512,
            "hop_length": 128,
            "mmt_frames": 24,
            "mmt_component_order": ["mfcc_time_doubled", "tonnetz_then_mel_time_concat"],
        }
        model_config = {
            "class_order": LABELS,
            "primary_tier": "InfantCry-1214",
            "feature_sets": {"mfcc_paper": mfcc, "mmt": mmt},
            "models": [
                {
                    "id": candidate,
                    "architecture": candidate,
                    "enabled": True,
                    "feature_set": "mfcc_paper" if candidate == "qiu_mfcc_reslstm" else "mmt",
                    "device": "cpu",
                    "mixed_precision": False,
                    "input_shape": [42, 24, 1]
                    if candidate == "qiu_mfcc_reslstm"
                    else [84, 48, 1],
                    "stem_filters": 4,
                    "residual_filters": [4, 8, 8, 8],
                    "lstm_units": [8, 4, 2],
                    "dropout": 0.1,
                }
                for candidate in candidates
            ],
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 0.001,
                "early_stopping_patience": 1,
                "selection_metric": "validation_macro_f1",
                "optimizer": "Adam",
                "deterministic_ops": True,
            },
        }
        self.model_config.write_text(
            yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8"
        )
        split = json.loads(split_path.read_text(encoding="utf-8"))
        test_ids = set(split["split_ids"]["test"])
        loader_calls: list[tuple[str, ...]] = []
        dispatched_models: list[str] = []

        def synthetic_features(records, sample_ids, audio_config):
            selected = tuple(sample_ids)
            self.assertTrue(set(selected).isdisjoint(test_ids))
            self.assertIn(audio_config.feature_set, {"mfcc_paper", "mmt"})
            loader_calls.append(selected)
            values = np.zeros((len(selected), *audio_config.tensor_shape), dtype=np.float32)
            labels = tuple(records[sample_id].raw_label for sample_id in selected)
            return values, labels, selected

        def train_stub(
            _tf,
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
            *,
            run,
            candidate_id,
            seed,
            epochs,
            batch_size,
            learning_rate,
            patience,
            device,
            class_order,
            validation_ids,
        ):
            del y_train, y_validation, seed, epochs, batch_size, learning_rate, patience
            del device, class_order
            self.assertEqual(tuple(x_train.shape[1:]), tuple(model.input_shape[1:]))
            self.assertEqual(tuple(x_validation.shape[1:]), tuple(model.input_shape[1:]))
            dispatched_models.append(model.name)
            run.mark_training_started()
            model_path = run.run_dir / "artifacts" / "best_model.keras"
            model_path.write_bytes(b"synthetic keras artifact; no model fit performed\n")
            validation_path = run.run_dir / "artifacts" / "validation_metrics.json"
            metrics = {
                "evaluation_scope": "validation_only",
                "candidate_id": candidate_id,
                "validation_support": len(validation_ids),
                "validation_macro_f1": (candidates.index(candidate_id) + 1) / 10.0,
            }
            validation_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            run.record_training_artifacts((model_path, validation_path))
            run.mark_training_complete(selected_candidate_id=candidate_id)
            return metrics

        original_record_loader = __import__(
            "infantcry_research.cli", fromlist=["_load_audited_records"]
        )._load_audited_records
        manifest_reads: list[str] = []

        def train_validation_manifest_only(records_path, dataset_root, *, sample_ids=None):
            manifest_reads.append(Path(records_path).name)
            self.assertEqual(Path(records_path).name, "train_validation_records.csv")
            return original_record_loader(records_path, dataset_root, sample_ids=sample_ids)

        with (
            patch(
                "infantcry_research.cli.configure_tensorflow_runtime",
                return_value={"selected_device": "cpu"},
            ),
            patch("infantcry_research.training.train.seed_tensorflow_experiment"),
            patch(
                "infantcry_research.training.data.load_feature_partition",
                side_effect=synthetic_features,
            ),
            patch("infantcry_research.cli.train_tensorflow_candidate", side_effect=train_stub),
            patch(
                "infantcry_research.cli._load_audited_records",
                side_effect=train_validation_manifest_only,
            ),
            patch(
                "infantcry_research.evaluation.comparison._load_test_records",
                side_effect=AssertionError(
                    "candidate development must not load locked Test records"
                ),
            ) as test_loader,
        ):
            run_dirs = []
            for candidate in candidates:
                status, stdout, stderr = self._invoke(
                    "train",
                    "--dataset-root", str(self.root),
                    "--audit-dir", str(self.audit_dir),
                    "--split-manifest", str(split_path),
                    "--config", str(self.model_config),
                    "--model", candidate,
                    "--device", "cpu",
                    "--output-dir", str(self.runs_dir),
                    "--run-id", f"synthetic-{candidate}",
                )
                self.assertEqual(status, 0, stderr)
                run_dirs.append(Path(json.loads(stdout)["run_dir"]))

            selection_args = ["select-candidate"]
            for run_dir in run_dirs:
                selection_args.extend(("--run-dir", str(run_dir)))
            selection_args.extend(
                (
                    "--models-config",
                    str(self.model_config),
                    "--output-dir",
                    str(self.base / "selections"),
                )
            )
            status, stdout, stderr = self._invoke(*selection_args)

        self.assertEqual(status, 0, stderr)
        self.assertEqual(
            dispatched_models,
            [
                "qiu_mfcc_reslstm_stage2",
                "qiu_mmt_resnet_stage2",
                "qiu_mmt_lstm_stage2",
                "qiu_mmt_reslstm_stage2",
            ],
        )
        self.assertEqual(len(loader_calls), 8)
        self.assertEqual(manifest_reads, ["train_validation_records.csv"] * 4)
        test_loader.assert_not_called()
        selection = json.loads(
            Path(json.loads(stdout)["selection_file"]).read_text(encoding="utf-8")
        )
        self.assertEqual(selection["candidate_set"], list(candidates))
        self.assertEqual(selection["selected_candidate_id"], "qiu_mmt_reslstm")
        self.assertEqual(
            len({entry["feature_config_sha256"] for entry in selection["ranking"]}), 2
        )
        self.assertFalse(selection["test_accessed"])

    def test_mmt_reslstm_training_freezes_candidate_specific_feature_contract(self) -> None:
        split_path = self._prepare()
        model_config = yaml.safe_load(self.model_config.read_text(encoding="utf-8"))
        standard = model_config.pop("features")
        model_config.pop("input_shape")
        standard["feature_set"] = "stage2_hybrid"
        mmt = dict(standard)
        mmt.update(
            {
                "feature_set": "mmt",
                "n_mfcc": 42,
                "n_mels": 42,
                "mmt_frames": 24,
                "n_fft": 512,
                "hop_length": 128,
            }
        )
        model_config["feature_sets"] = {"stage2_hybrid": standard, "mmt": mmt}
        model_config["models"] = [
            {
                "id": "mmt_reslstm",
                "enabled": True,
                "feature_set": "mmt",
                "device": "cpu",
                "mixed_precision": False,
                "stem_filters": 4,
                "residual_filters": [4, 8, 8, 8],
                "lstm_units": [8, 4, 2],
                "dropout": 0.1,
            }
        ]
        model_config["training"] = {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "selection_metric": "validation_macro_f1",
        }
        self.model_config.write_text(yaml.safe_dump(model_config), encoding="utf-8")
        loader_calls: list[tuple[str, ...]] = []

        def synthetic_features(records, sample_ids, audio_config):
            selected = tuple(sample_ids)
            loader_calls.append(selected)
            self.assertEqual(audio_config.feature_set, "mmt")
            self.assertEqual(audio_config.tensor_shape, (84, 48, 1))
            values = np.zeros((len(selected), *audio_config.tensor_shape), dtype=np.float32)
            labels = tuple(records[sample_id].raw_label for sample_id in selected)
            return values, labels, selected

        original_record_loader = __import__(
            "infantcry_research.cli", fromlist=["_load_audited_records"]
        )._load_audited_records
        manifest_reads: list[str] = []

        def train_validation_manifest_only(records_path, dataset_root, *, sample_ids=None):
            manifest_reads.append(Path(records_path).name)
            self.assertEqual(Path(records_path).name, "train_validation_records.csv")
            return original_record_loader(records_path, dataset_root, sample_ids=sample_ids)

        with (
            patch(
                "infantcry_research.cli._load_audited_records",
                side_effect=train_validation_manifest_only,
            ),
            patch(
                "infantcry_research.training.data.load_feature_partition",
                side_effect=synthetic_features,
            ),
        ):
            status, stdout, stderr = self._invoke(
                "train",
                "--dataset-root",
                str(self.root),
                "--audit-dir",
                str(self.audit_dir),
                "--split-manifest",
                str(split_path),
                "--config",
                str(self.model_config),
                "--model",
                "mmt_reslstm",
                "--device",
                "cpu",
                "--output-dir",
                str(self.runs_dir),
                "--run-id",
                "synthetic-mmt-reslstm",
            )

        self.assertEqual(status, 0, stderr)
        self.assertNotIn("error:", stderr.casefold())
        self.assertEqual(manifest_reads, ["train_validation_records.csv"])
        self.assertEqual(len(loader_calls), 2)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        self.assertTrue(
            all(
                not (set(sample_ids) & set(split["split_ids"]["test"]))
                for sample_ids in loader_calls
            )
        )
        run_info = json.loads(stdout)
        run_dir = Path(run_info["run_dir"])
        frozen_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(frozen_config["candidate_id"], "mmt_reslstm")
        self.assertEqual(frozen_config["input_shape"], [84, 48, 1])
        self.assertEqual(frozen_config["feature_config"]["feature_set"], "mmt")
        self.assertTrue((run_dir / "artifacts" / "best_model.keras").is_file())

    def test_validation_selection_accepts_distinct_feature_contracts_for_same_evidence(self) -> None:
        split_path = self._prepare()
        model_config = yaml.safe_load(self.model_config.read_text(encoding="utf-8"))
        standard = model_config.pop("features")
        model_config.pop("input_shape")
        standard["feature_set"] = "stage2_hybrid"
        mmt = dict(standard)
        mmt.update(
            {
                "feature_set": "mmt",
                "n_mfcc": 42,
                "n_mels": 42,
                "mmt_frames": 24,
                "n_fft": 512,
                "hop_length": 128,
            }
        )
        model_config["feature_sets"] = {"stage2_hybrid": standard, "mmt": mmt}
        model_config["models"] = [
            {"id": "majority", "enabled": True, "feature_set": "stage2_hybrid", "device": "cpu"},
            {
                "id": "mmt_reslstm",
                "enabled": True,
                "feature_set": "mmt",
                "device": "cpu",
                "mixed_precision": False,
                "input_shape": [84, 48, 1],
                "stem_filters": 4,
                "residual_filters": [4, 8, 8, 8],
                "lstm_units": [8, 4, 2],
                "dropout": 0.1,
            },
        ]
        model_config["training"] = {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "selection_metric": "validation_macro_f1",
        }
        self.model_config.write_text(yaml.safe_dump(model_config), encoding="utf-8")

        majority_status, majority_stdout, majority_stderr = self._invoke(
            "train",
            "--dataset-root", str(self.root),
            "--audit-dir", str(self.audit_dir),
            "--split-manifest", str(split_path),
            "--config", str(self.model_config),
            "--model", "majority",
            "--device", "cpu",
            "--output-dir", str(self.runs_dir),
            "--run-id", "selection-majority",
        )
        self.assertEqual(majority_status, 0, majority_stderr)
        majority_run = Path(json.loads(majority_stdout)["run_dir"])

        def synthetic_features(records, sample_ids, audio_config):
            selected = tuple(sample_ids)
            values = np.zeros((len(selected), *audio_config.tensor_shape), dtype=np.float32)
            return values, tuple(records[item].raw_label for item in selected), selected

        with patch(
            "infantcry_research.training.data.load_feature_partition",
            side_effect=synthetic_features,
        ):
            mmt_status, mmt_stdout, mmt_stderr = self._invoke(
                "train",
                "--dataset-root", str(self.root),
                "--audit-dir", str(self.audit_dir),
                "--split-manifest", str(split_path),
                "--config", str(self.model_config),
                "--model", "mmt_reslstm",
                "--device", "cpu",
                "--output-dir", str(self.runs_dir),
                "--run-id", "selection-mmt",
            )
        self.assertEqual(mmt_status, 0, mmt_stderr)
        mmt_run = Path(json.loads(mmt_stdout)["run_dir"])

        selection_status, selection_stdout, selection_stderr = self._invoke(
            "select-candidate",
            "--run-dir", str(majority_run),
            "--run-dir", str(mmt_run),
            "--models-config", str(self.model_config),
            "--output-dir", str(self.base / "selections"),
        )

        self.assertEqual(selection_status, 0, selection_stderr)
        selection_info = json.loads(selection_stdout)
        selection = json.loads(Path(selection_info["selection_file"]).read_text(encoding="utf-8"))
        self.assertEqual(set(selection["candidate_set"]), {"majority", "mmt_reslstm"})
        self.assertEqual(
            len({candidate["feature_config_sha256"] for candidate in selection["ranking"]}),
            2,
        )
        self.assertFalse(selection["test_accessed"])

        split = json.loads(split_path.read_text(encoding="utf-8"))
        ledger_dir = self.base / "manifests" / "test_access"
        original_loader = __import__(
            "infantcry_research.evaluation.comparison", fromlist=["_load_test_records"]
        )._load_test_records

        def assert_claim_precedes_record_read(records_path, dataset_root, sample_ids):
            claim_path = ledger_dir / f"{split['split_id']}.json"
            self.assertTrue(claim_path.is_file(), "test records were requested before ledger claim")
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            self.assertEqual(claim["state"], "claimed_before_test_access")
            return original_loader(records_path, dataset_root, sample_ids)

        def synthetic_test_features(records, sample_ids, audio_config):
            selected = tuple(sample_ids)
            self.assertEqual(selected, tuple(split["split_ids"]["test"]))
            self.assertEqual(audio_config.feature_set, "mmt")
            values = np.zeros((len(selected), *audio_config.tensor_shape), dtype=np.float32)
            labels = tuple(records[sample_id].raw_label for sample_id in selected)
            return values, labels, selected

        with (
            patch("infantcry_research.cli.PROJECT_ROOT", self.base),
            patch(
                "infantcry_research.evaluation.comparison._load_test_records",
                side_effect=assert_claim_precedes_record_read,
            ),
            patch(
                "infantcry_research.training.data.load_feature_partition",
                side_effect=synthetic_test_features,
            ),
        ):
            evaluation_status, evaluation_stdout, evaluation_stderr = self._invoke(
                "evaluate-test",
                "--dataset-root", str(self.root),
                "--audit-dir", str(self.audit_dir),
                "--selection-file", selection_info["selection_file"],
                "--output-dir", str(self.base / "comparisons"),
                "--bootstrap-resamples", "100",
                "--device", "cpu",
            )
        self.assertEqual(evaluation_status, 0, evaluation_stderr)
        self.assertNotIn("error:", evaluation_stderr.casefold())
        evaluation_info = json.loads(evaluation_stdout)
        comparison_dir = Path(evaluation_info["comparison_dir"])
        comparison = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(comparison["candidate_order"], ["majority", "mmt_reslstm"])
        self.assertEqual(comparison["test_support"], len(split["split_ids"]["test"]))
        self.assertEqual(comparison["test_scores_used_for_selection"], False)
        self.assertEqual(comparison["selected_by_validation"]["candidate_id"], selection["selected_candidate_id"])
        self.assertEqual(len(comparison["candidates"]), 2)
        claim = json.loads((ledger_dir / f"{split['split_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(claim["state"], "evaluated")
        self.assertEqual([item["candidate_id"] for item in claim["candidate_runs"]], ["majority", "mmt_reslstm"])
        for candidate_id in ("majority", "mmt_reslstm"):
            predictions_path = comparison_dir / "candidates" / candidate_id / "test_predictions.csv"
            with predictions_path.open("r", newline="", encoding="utf-8") as stream:
                prediction_ids = [row["sample_id"] for row in csv.DictReader(stream)]
            self.assertEqual(prediction_ids, split["split_ids"]["test"])
            self.assertTrue((predictions_path.parent / "confusion_matrix.png").is_file())

    def test_validation_selection_shared_evidence_guard_rejects_split_or_data_mismatch(self) -> None:
        from infantcry_research.cli import _assert_shared_candidate_evidence

        reference = {
            "split_sha256": "split-a",
            "dataset_evidence": "dataset-a",
            "seed": 42,
            "training_settings": {"epochs": 5},
        }
        for key, value in (("split_sha256", "split-b"), ("dataset_evidence", "dataset-b")):
            candidate = dict(reference)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                DatasetAuditError, "same split, dataset evidence, seed, and training controls"
            ):
                _assert_shared_candidate_evidence(reference, candidate)

    def test_majority_training_does_not_open_locked_test_audio(self) -> None:
        split_path = self._prepare()
        split = json.loads(split_path.read_text(encoding="utf-8"))
        records_path = self.audit_dir / "records.csv"
        with records_path.open("r", newline="", encoding="utf-8-sig") as stream:
            records_by_id = {row["sample_id"]: row for row in csv.DictReader(stream)}
        test_audio = [self.root / records_by_id[sample_id]["relative_path"] for sample_id in split["split_ids"]["test"]]
        for audio_path in test_audio:
            audio_path.unlink()

        status, stdout, stderr = self._invoke(
            "train",
            "--dataset-root",
            str(self.root),
            "--audit-dir",
            str(self.audit_dir),
            "--split-manifest",
            str(split_path),
            "--config",
            str(self.model_config),
            "--model",
            "majority",
            "--device",
            "cpu",
            "--output-dir",
            str(self.runs_dir),
            "--run-id",
            "synthetic-majority-test",
        )

        self.assertEqual((status, stderr), (0, ""))
        result = json.loads(stdout)
        self.assertEqual(result["evaluation_scope"], "validation_only")
        self.assertEqual(result["train_support"], len(split["split_ids"]["train"]))
        self.assertEqual(result["validation_support"], len(split["split_ids"]["validation"]))
        run_dir = self.runs_dir / "synthetic-majority-test"
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "complete")
        predictions = (run_dir / "artifacts" / "validation_predictions.csv").read_text(encoding="utf-8")
        self.assertFalse(any(sample_id in predictions for sample_id in split["split_ids"]["test"]))
        self.assertTrue(all(not audio_path.exists() for audio_path in test_audio))

    def test_training_rejects_tampered_split_manifest(self) -> None:
        split_path = self._prepare()
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["split_ids"]["validation"][0] = split["split_ids"]["test"][0]
        split_path.write_text(json.dumps(split), encoding="utf-8")

        status, _stdout, stderr = self._invoke(
            "train",
            "--dataset-root",
            str(self.root),
            "--audit-dir",
            str(self.audit_dir),
            "--split-manifest",
            str(split_path),
            "--config",
            str(self.model_config),
            "--model",
            "majority",
            "--device",
            "cpu",
            "--output-dir",
            str(self.runs_dir),
        )

        self.assertEqual(status, 2)
        self.assertIn("manifest hash", stderr)
        self.assertFalse(self.runs_dir.exists())

    def test_training_refuses_provisional_sample_rate(self) -> None:
        split_path = self._prepare()
        model_config = yaml.safe_load(self.model_config.read_text(encoding="utf-8"))
        model_config["features"]["sample_rate_status"] = "provisional"
        self.model_config.write_text(yaml.safe_dump(model_config), encoding="utf-8")

        status, _stdout, stderr = self._invoke(
            "train",
            "--dataset-root",
            str(self.root),
            "--audit-dir",
            str(self.audit_dir),
            "--split-manifest",
            str(split_path),
            "--config",
            str(self.model_config),
            "--model",
            "majority",
            "--device",
            "cpu",
            "--output-dir",
            str(self.runs_dir),
        )

        self.assertEqual(status, 2)
        self.assertIn("sample rate is still provisional", stderr.casefold())
        self.assertFalse(self.runs_dir.exists())

    def test_training_rejects_device_override_that_differs_from_frozen_candidate_config(self) -> None:
        split_path = self._prepare()

        status, _stdout, stderr = self._invoke(
            "train",
            "--dataset-root",
            str(self.root),
            "--audit-dir",
            str(self.audit_dir),
            "--split-manifest",
            str(split_path),
            "--config",
            str(self.model_config),
            "--model",
            "majority",
            "--device",
            "gpu",
            "--output-dir",
            str(self.runs_dir),
        )

        self.assertEqual(status, 2)
        self.assertIn("device differs from its frozen candidate config", stderr)
        self.assertFalse(self.runs_dir.exists())

    def test_training_rejects_mixed_precision_override_that_differs_from_candidate_config(self) -> None:
        split_path = self._prepare()

        status, _stdout, stderr = self._invoke(
            "train",
            "--dataset-root",
            str(self.root),
            "--audit-dir",
            str(self.audit_dir),
            "--split-manifest",
            str(split_path),
            "--config",
            str(self.model_config),
            "--model",
            "majority",
            "--device",
            "cpu",
            "--mixed-precision",
            "--output-dir",
            str(self.runs_dir),
        )

        self.assertEqual(status, 2)
        self.assertIn("mixed precision differs from its frozen candidate config", stderr)
        self.assertFalse(self.runs_dir.exists())

    def test_locked_test_is_claimed_once_and_emits_report_artifacts(self) -> None:
        split_path = self._prepare()
        train_status, train_stdout, train_stderr = self._invoke(
            "train",
            "--dataset-root",
            str(self.root),
            "--audit-dir",
            str(self.audit_dir),
            "--split-manifest",
            str(split_path),
            "--config",
            str(self.model_config),
            "--model",
            "majority",
            "--device",
            "cpu",
            "--output-dir",
            str(self.runs_dir),
            "--run-id",
            "synthetic-eval-test",
        )
        self.assertEqual((train_status, train_stderr), (0, ""))
        run_info = json.loads(train_stdout)
        run_dir = Path(run_info["run_dir"])

        original_model_config = self.model_config.read_bytes()
        modified_config = yaml.safe_load(original_model_config)
        modified_config["training"]["batch_size"] = 99
        self.model_config.write_text(yaml.safe_dump(modified_config), encoding="utf-8")
        changed_config_status, _changed_stdout, changed_config_stderr = self._invoke(
            "select-candidate",
            "--run-dir",
            str(run_dir),
            "--models-config",
            str(self.model_config),
            "--output-dir",
            str(self.base / "manifests" / "selections"),
        )
        self.assertEqual(changed_config_status, 2)
        self.assertIn("candidate protocol configuration changed", changed_config_stderr)
        self.model_config.write_bytes(original_model_config)

        selection_status, selection_stdout, selection_stderr = self._invoke(
            "select-candidate",
            "--run-dir",
            str(run_dir),
            "--models-config",
            str(self.model_config),
            "--output-dir",
            str(self.base / "manifests" / "selections"),
        )
        self.assertEqual((selection_status, selection_stderr), (0, ""))
        selection_info = json.loads(selection_stdout)
        selection_path = Path(selection_info["selection_file"])
        self.assertFalse(selection_info["test_accessed"])

        with patch("infantcry_research.cli.PROJECT_ROOT", self.base):
            evaluation_status, evaluation_stdout, evaluation_stderr = self._invoke(
                "evaluate-test",
                "--dataset-root",
                str(self.root),
                "--audit-dir",
                str(self.audit_dir),
                "--selection-file",
                str(selection_path),
                "--output-dir",
                str(self.base / "comparisons"),
                "--bootstrap-resamples",
                "100",
                "--device",
                "cpu",
            )

        self.assertEqual((evaluation_status, evaluation_stderr), (0, ""))
        evaluation = json.loads(evaluation_stdout)
        self.assertEqual(evaluation["test_support"], 15)
        comparison_dir = Path(evaluation["comparison_dir"])
        evaluation_dir = comparison_dir / "candidates" / "majority"
        self.assertTrue((evaluation_dir / "confusion_matrix.png").is_file())
        self.assertTrue((evaluation_dir / "confusion_matrix.csv").is_file())
        self.assertTrue((evaluation_dir / "test_predictions.csv").is_file())
        metric_payload = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metric_payload["evaluation_scope"], "locked_internal_test")
        self.assertEqual(metric_payload["support"], 15)
        self.assertEqual(metric_payload["uncertainty"]["status"], "unavailable")
        self.assertIn("not verified", metric_payload["uncertainty"]["reason"])

        split = json.loads(split_path.read_text(encoding="utf-8"))
        with (self.audit_dir / "records.csv").open("r", newline="", encoding="utf-8-sig") as stream:
            records_by_id = {row["sample_id"]: row for row in csv.DictReader(stream)}
        with patch("infantcry_research.cli.PROJECT_ROOT", self.base):
            retry_status, _retry_stdout, retry_stderr = self._invoke(
                "evaluate-test",
                "--dataset-root",
                str(self.root),
                "--audit-dir",
                str(self.audit_dir),
                "--selection-file",
                str(selection_path),
                "--output-dir",
                str(self.base / "comparisons"),
                "--bootstrap-resamples",
                "100",
                "--device",
                "cpu",
                "--protocol-id",
                "a-different-protocol-must-not-reopen-test",
            )
        self.assertEqual(retry_status, 2)
        self.assertIn("already been claimed or evaluated", retry_stderr)
        for sample_id in split["split_ids"]["test"]:
            (self.root / records_by_id[sample_id]["relative_path"]).unlink()

        reports_dir = self.base / "reports" / "stage2"
        with patch("infantcry_research.cli.PROJECT_ROOT", self.base):
            report_status, report_stdout, report_stderr = self._invoke(
                "report",
                "--comparison-dir",
                str(comparison_dir),
                "--output-dir",
                str(reports_dir),
            )
        self.assertEqual((report_status, report_stderr), (0, ""))
        report_path = Path(report_stdout.strip().removeprefix("Report created: "))
        report = report_path.read_text(encoding="utf-8")
        self.assertIn("Stage 2 paired comparison", report)
        self.assertIn("Selected by validation", report)
        self.assertIn("Test scores were not used for selection", report)
        self.assertIn("confusion_matrix.png", report)
        self.assertIn("Group-bootstrap interval: unavailable", report)


if __name__ == "__main__":
    unittest.main()

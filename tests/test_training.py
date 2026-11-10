"""Runtime and synthetic-only training lifecycle checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from infantcry_research.training.run_store import create_run
from infantcry_research.training.train import (
    TrainingError,
    configure_tensorflow_runtime,
    seed_tensorflow_experiment,
    train_classical_candidate,
    train_tensorflow_candidate,
)
from infantcry_research.models.baselines import MajorityClassifier


LABELS = ("eairh", "eh", "heh", "neh", "owh")


class TrainingTests(unittest.TestCase):
    def test_seed_is_applied_before_model_initialization_for_repeatable_weights(self) -> None:
        def build_with_seed() -> list[np.ndarray]:
            seed_tensorflow_experiment(tf, 17, deterministic_ops=True)
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(3,)),
                    tf.keras.layers.Dense(5, activation="relu"),
                    tf.keras.layers.Dense(2, activation="softmax"),
                ]
            )
            return model.get_weights()

        first = build_with_seed()
        second = build_with_seed()

        for first_weight, second_weight in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_weight, second_weight)

    def test_gpu_request_never_silently_falls_back_to_cpu(self) -> None:
        if tf.config.list_physical_devices("GPU"):
            self.skipTest("This host exposes a GPU; the CPU-only failure path is not applicable")

        with self.assertRaisesRegex(TrainingError, "GPU was requested but TensorFlow sees none"):
            configure_tensorflow_runtime(tf, device="gpu", require_gpu=True)

    def test_cpu_device_must_be_explicit_and_is_recorded(self) -> None:
        environment = configure_tensorflow_runtime(tf, device="cpu", require_gpu=False)

        self.assertEqual(environment["requested_device"], "cpu")
        self.assertEqual(environment["selected_device"], "cpu")
        self.assertEqual(environment["gpu_count"], len(tf.config.list_physical_devices("GPU")))

    def test_tensorflow_fit_uses_only_train_validation_and_completes_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = create_run(
                Path(directory) / "runs",
                config={"model": "tiny_test", "seed": 3},
                split_manifest={"split_id": "synthetic-only", "locked_test": True},
                environment={"selected_device": "cpu"},
                candidate_ids=("tiny_test",),
            )
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(2,)),
                    tf.keras.layers.Dense(4, activation="relu"),
                    tf.keras.layers.Dense(2, activation="softmax", dtype="float32"),
                ]
            )
            x_train = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
            y_train = np.asarray([0, 1, 1, 0], dtype=np.int32)
            x_validation = np.asarray([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
            y_validation = np.asarray([1, 0], dtype=np.int32)

            result = train_tensorflow_candidate(
                tf,
                model,
                x_train,
                y_train,
                x_validation,
                y_validation,
                run=run,
                candidate_id="tiny_test",
                seed=3,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                patience=1,
                device="cpu",
            )

            self.assertEqual(run.status, "complete")
            self.assertEqual(result["train_support"], 4)
            self.assertEqual(result["validation_support"], 2)
            self.assertTrue((run.run_dir / "artifacts" / "best_model.keras").is_file())
            self.assertTrue((run.run_dir / "logs" / "epochs.csv").is_file())
            self.assertNotIn("test", result)

    def test_classical_baseline_saves_only_validation_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = create_run(
                Path(directory) / "runs",
                config={"model": "majority", "seed": 42},
                split_manifest={"split_id": "synthetic-only", "locked_test": True},
                environment={"selected_device": "cpu"},
                candidate_ids=("majority",),
            )
            model = MajorityClassifier(LABELS)
            train_labels = np.asarray(["eairh"] * 6 + ["eh", "heh", "neh", "owh"])
            validation_labels = np.asarray(LABELS)
            validation_ids = tuple(f"validation-{index}" for index in range(5))

            result = train_classical_candidate(
                model,
                None,
                train_labels,
                None,
                validation_labels,
                run=run,
                candidate_id="majority",
                class_order=LABELS,
                validation_ids=validation_ids,
            )

            self.assertEqual(run.status, "complete")
            self.assertEqual(result["evaluation_scope"], "validation_only")
            self.assertEqual(result["train_support"], 10)
            saved = (run.run_dir / "artifacts" / "validation_predictions.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("validation-0", saved)
            self.assertNotIn("test", saved.casefold())


if __name__ == "__main__":
    unittest.main()

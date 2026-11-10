"""GPU-explicit runtime checks and train/validation-only Keras orchestration."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from infantcry_research.training.run_store import RunStore, RunStoreError


class TrainingError(RuntimeError):
    """Raised before or during training when the requested setup is invalid."""


def seed_tensorflow_experiment(tf: Any, seed: int, *, deterministic_ops: bool) -> None:
    """Set every supported random seed before constructing a TensorFlow model."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError as exc:
        raise TrainingError("TensorFlow does not expose keras.utils.set_random_seed") from exc
    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError as exc:
            raise TrainingError(
                "TensorFlow does not expose enable_op_determinism; use a compatible runtime "
                "or explicitly configure deterministic_ops: false"
            ) from exc


def configure_tensorflow_runtime(
    tf: Any,
    *,
    device: str,
    require_gpu: bool,
    mixed_precision: bool = False,
) -> dict[str, Any]:
    """Validate the requested device and record TensorFlow-visible hardware.

    There is intentionally no implicit GPU-to-CPU fallback. CPU execution is
    available only when explicitly requested with `device="cpu"`.
    """

    selected = str(device).strip().casefold()
    if selected not in {"cpu", "gpu"}:
        raise TrainingError("device must be explicitly set to 'cpu' or 'gpu'")
    if require_gpu and selected != "gpu":
        raise TrainingError("--require-gpu cannot be combined with an explicit CPU request")
    if mixed_precision and selected != "gpu":
        raise TrainingError("mixed precision is enabled only for an explicitly requested GPU run")

    physical_gpus = tf.config.list_physical_devices("GPU")
    if selected == "gpu" and not physical_gpus:
        raise TrainingError("GPU was requested but TensorFlow sees none; no CPU fallback was made")
    if selected == "gpu":
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                raise TrainingError(
                    f"Could not configure GPU memory growth before runtime initialization: {exc}"
                ) from exc
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16" if mixed_precision else "float32")
    except (AttributeError, ValueError) as exc:
        raise TrainingError(f"Could not set TensorFlow mixed-precision policy: {exc}") from exc

    return {
        "tensorflow_version": str(tf.__version__),
        "keras_version": str(getattr(tf.keras, "__version__", "bundled")),
        "requested_device": selected,
        "selected_device": selected,
        "gpu_count": len(physical_gpus),
        "gpu_names": [str(device.name) for device in physical_gpus],
        "mixed_precision": bool(mixed_precision),
    }


def train_tensorflow_candidate(
    tf: Any,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    run: RunStore,
    candidate_id: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: str,
    class_order: tuple[str, ...] | None = None,
    validation_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Fit on training data, choose best epoch on validation, and save evidence.

    The API has no test-set parameter: locked test files cannot accidentally be
    passed to `model.fit` through this function.
    """

    if run.status != "prepared":
        raise TrainingError(f"Run must be prepared before training, got {run.status!r}")
    if candidate_id not in run.metadata.get("candidate_ids", []):
        raise TrainingError(f"Candidate {candidate_id!r} was not preregistered in this run")
    if device not in {"cpu", "gpu"}:
        raise TrainingError("device must be explicitly set to 'cpu' or 'gpu'")
    if epochs < 1 or batch_size < 1 or patience < 0 or learning_rate <= 0:
        raise TrainingError("epochs/batch_size must be positive, patience non-negative, learning_rate positive")
    if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
        raise TrainingError("feature and label support do not match")
    if len(x_train) == 0 or len(x_validation) == 0:
        raise TrainingError("training and validation splits must both be non-empty")
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise TrainingError("input features contain NaN or infinite values")

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    artifact_dir = run.run_dir / "artifacts"
    log_dir = run.run_dir / "logs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = artifact_dir / "best_model.keras"
    history_path = log_dir / "epochs.csv"
    if best_model_path.exists() or history_path.exists():
        raise TrainingError("Training artifacts already exist; refusing to overwrite a prior attempt")

    class ValidationMacroF1(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            del epoch
            if logs is None:
                return
            val_probabilities = self.model.predict(
                x_validation, batch_size=batch_size, verbose=0
            )
            val_predictions = np.argmax(val_probabilities, axis=1)
            logs["val_macro_f1"] = float(
                f1_score(
                    np.asarray(y_validation).astype(int),
                    val_predictions,
                    labels=list(range(val_probabilities.shape[1])),
                    average="macro",
                    zero_division=0,
                )
            )

    callbacks = [
        ValidationMacroF1(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(str(history_path), append=False),
    ]

    run.mark_training_started()
    try:
        device_name = "/GPU:0" if device == "gpu" else "/CPU:0"
        with tf.device(device_name):
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_validation, y_validation),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
                shuffle=True,
            )
            validation_loss, validation_accuracy = model.evaluate(
                x_validation, y_validation, batch_size=batch_size, verbose=0
            )
            probabilities = model.predict(x_validation, batch_size=batch_size, verbose=0)

        y_true = np.asarray(y_validation).astype(int)
        y_pred = np.argmax(probabilities, axis=1)
        metrics = {
            "evaluation_scope": "validation_only",
            "candidate_id": candidate_id,
            "seed": seed,
            "train_support": int(len(y_train)),
            "validation_support": int(len(y_validation)),
            "epochs_completed": int(len(history.epoch)),
            "checkpoint_selection_metric": "validation_macro_f1",
            "best_epoch_zero_based": int(np.argmax(history.history["val_macro_f1"])),
            "validation_loss": float(validation_loss),
            "validation_accuracy": float(accuracy_score(y_true, y_pred)),
            "validation_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "validation_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        _write_validation_metrics(run.run_dir / "artifacts" / "validation_metrics.json", metrics)
        labels = class_order or tuple(str(index) for index in range(probabilities.shape[1]))
        true_names = [labels[int(index)] for index in y_true]
        predicted_names = [labels[int(index)] for index in y_pred]
        ids = validation_ids or tuple(f"validation-{index}" for index in range(len(y_true)))
        _write_validation_predictions(
            run.run_dir / "artifacts" / "validation_predictions.csv",
            ids,
            true_names,
            predicted_names,
            probabilities,
            labels,
        )
        run.record_training_artifacts(
            (
                best_model_path,
                history_path,
                run.run_dir / "artifacts" / "validation_metrics.json",
                run.run_dir / "artifacts" / "validation_predictions.csv",
            )
        )
        run.mark_training_complete(selected_candidate_id=candidate_id)
        return metrics
    except Exception as exc:
        if run.status == "running":
            try:
                run.mark_training_failed(reason=f"{type(exc).__name__}: {exc}")
            except RunStoreError:
                pass
        raise


def train_classical_candidate(
    estimator: Any,
    x_train: Any,
    y_train: np.ndarray,
    x_validation: Any,
    y_validation: np.ndarray,
    *,
    run: RunStore,
    candidate_id: str,
    class_order: tuple[str, ...],
    validation_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Fit a labels-only or scikit-learn baseline on train and score validation."""

    if run.status != "prepared":
        raise TrainingError(f"Run must be prepared before training, got {run.status!r}")
    if candidate_id not in run.metadata.get("candidate_ids", []):
        raise TrainingError(f"Candidate {candidate_id!r} was not preregistered in this run")
    if len(y_train) == 0 or len(y_validation) == 0:
        raise TrainingError("training and validation splits must both be non-empty")
    if len(y_validation) != len(validation_ids):
        raise TrainingError("validation IDs and labels have different support")

    run.mark_training_started()
    try:
        fit_features = x_train
        validation_features = x_validation
        if hasattr(estimator, "class_order") and x_train is None and x_validation is None:
            fit_features = np.empty((len(y_train), 0), dtype=np.float32)
            validation_features = np.empty((len(y_validation), 0), dtype=np.float32)
        estimator.fit(fit_features, y_train)
        predictions = np.asarray(estimator.predict(validation_features)).astype(str)
        source_probabilities = np.asarray(
            estimator.predict_proba(validation_features), dtype=np.float64
        )
        source_classes = tuple(str(label) for label in estimator.classes_)
        probabilities = np.zeros((len(y_validation), len(class_order)), dtype=np.float64)
        class_index = {label: index for index, label in enumerate(class_order)}
        for source_index, label in enumerate(source_classes):
            if label not in class_index:
                raise TrainingError(f"Estimator returned unconfigured class label {label!r}")
            probabilities[:, class_index[label]] = source_probabilities[:, source_index]
        y_true = np.asarray(y_validation).astype(str)
        metrics = {
            "evaluation_scope": "validation_only",
            "candidate_id": candidate_id,
            "train_support": int(len(y_train)),
            "validation_support": int(len(y_validation)),
            "validation_accuracy": float(accuracy_score(y_true, predictions)),
            "validation_balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "validation_macro_f1": float(
                f1_score(y_true, predictions, labels=list(class_order), average="macro", zero_division=0)
            ),
        }
        model_path = run.run_dir / "artifacts" / "model.joblib"
        metrics_path = run.run_dir / "artifacts" / "validation_metrics.json"
        predictions_path = run.run_dir / "artifacts" / "validation_predictions.csv"
        for path in (model_path, metrics_path, predictions_path):
            if path.exists():
                raise TrainingError(f"Training artifact already exists; refusing to overwrite: {path}")
        import joblib

        joblib.dump(estimator, model_path)
        _write_validation_metrics(metrics_path, metrics)
        _write_validation_predictions(
            predictions_path,
            validation_ids,
            list(y_true),
            list(predictions),
            probabilities,
            class_order,
        )
        run.record_training_artifacts((model_path, metrics_path, predictions_path))
        run.mark_training_complete(selected_candidate_id=candidate_id)
        return metrics
    except Exception as exc:
        if run.status == "running":
            try:
                run.mark_training_failed(reason=f"{type(exc).__name__}: {exc}")
            except RunStoreError:
                pass
        raise


def _write_validation_metrics(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise TrainingError(f"Validation metrics already exist; refusing to overwrite: {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_validation_predictions(
    path: Path,
    sample_ids: tuple[str, ...] | list[str],
    true_labels: list[str],
    predicted_labels: list[str],
    probabilities: np.ndarray,
    class_order: tuple[str, ...],
) -> None:
    if path.exists():
        raise TrainingError(f"Validation predictions already exist; refusing to overwrite: {path}")
    if len(sample_ids) != len(true_labels) or len(predicted_labels) != len(true_labels):
        raise TrainingError("Validation prediction IDs, truth, and predictions have different support")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "true_label", "predicted_label"]
            + [f"probability_{label}" for label in class_order],
        )
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            row = {
                "sample_id": sample_id,
                "true_label": true_labels[index],
                "predicted_label": predicted_labels[index],
            }
            row.update(
                {
                    f"probability_{label}": float(probabilities[index, class_index])
                    for class_index, label in enumerate(class_order)
                }
            )
            writer.writerow(row)

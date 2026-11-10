"""Single-write locked-test artifacts and their content-hash receipt."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def write_locked_test_artifacts(
    run_dir: Path,
    *,
    candidate_id: str,
    protocol_hash: str,
    class_order: Sequence[str],
    sample_ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_score: np.ndarray,
    metrics_payload: dict[str, object],
) -> tuple[Path, str]:
    """Write metrics, predictions, confusion matrix and hashes exactly once."""

    classes = tuple(str(value) for value in class_order)
    if not candidate_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("candidate_id contains unsafe path characters")
    if not protocol_hash or any(character not in "0123456789abcdef" for character in protocol_hash):
        raise ValueError("protocol_hash must be a lowercase hexadecimal identifier")
    truth = tuple(str(value) for value in y_true)
    predicted = tuple(str(value) for value in y_pred)
    identifiers = tuple(str(value) for value in sample_ids)
    scores = np.asarray(y_score, dtype=np.float64)
    if not identifiers or len(identifiers) != len(truth) or len(truth) != len(predicted):
        raise ValueError("test IDs, truth, and predictions must have equal non-zero support")
    if scores.shape != (len(truth), len(classes)):
        raise ValueError("test score matrix shape does not match test support and class order")

    evaluation_root = Path(run_dir).resolve() / "evaluations" / "test"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    destination = evaluation_root / f"{candidate_id}__{protocol_hash}"
    if destination.exists():
        raise FileExistsError(f"locked-test artifacts already exist: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{candidate_id}.", dir=evaluation_root))
    try:
        metrics_path = temporary / "metrics.json"
        metrics_path.write_bytes(
            (json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        )
        predictions_path = temporary / "test_predictions.csv"
        with predictions_path.open("w", newline="", encoding="utf-8") as stream:
            fields = ["sample_id", "true_label", "predicted_label"] + [
                f"probability_{label}" for label in classes
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, sample_id in enumerate(identifiers):
                row: dict[str, object] = {
                    "sample_id": sample_id,
                    "true_label": truth[index],
                    "predicted_label": predicted[index],
                }
                row.update(
                    {
                        f"probability_{label}": format(float(scores[index, class_index]), ".10g")
                        for class_index, label in enumerate(classes)
                    }
                )
                writer.writerow(row)

        matrix = confusion_matrix(truth, predicted, labels=list(classes))
        confusion_csv = temporary / "confusion_matrix.csv"
        with confusion_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true\\predicted", *classes])
            for label, row in zip(classes, matrix, strict=True):
                writer.writerow([label, *[int(value) for value in row]])

        figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            xlabel="Predicted label",
            ylabel="True label",
            title="Locked-test confusion matrix",
        )
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = int(matrix[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )
        figure.colorbar(image, ax=axis, label="Number of clips")
        figure.savefig(temporary / "confusion_matrix.png", dpi=180)
        plt.close(figure)

        file_hashes = {
            path.name: _sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": "1.0",
            "candidate_id": candidate_id,
            "protocol_hash": protocol_hash,
            "files": file_hashes,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        (temporary / "artifact_hashes.json").write_bytes(manifest_bytes)
        evaluation_hash = hashlib.sha256(manifest_bytes).hexdigest()
        os.replace(temporary, destination)
        return destination, evaluation_hash
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

"""Transparent multi-class metrics and group-level bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_score: np.ndarray,
    class_order: Sequence[str],
) -> dict[str, object]:
    """Calculate fixed-order metrics, requiring normalized class probabilities."""

    classes = tuple(str(label) for label in class_order)
    truth = np.asarray([str(value) for value in y_true], dtype=object)
    predicted = np.asarray([str(value) for value in y_pred], dtype=object)
    scores = np.asarray(y_score, dtype=np.float64)
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise ValueError("class_order must contain at least two unique labels")
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) == 0 or len(truth) != len(predicted):
        raise ValueError("truth and prediction arrays must be non-empty and have equal support")
    unknown = sorted((set(truth) | set(predicted)) - set(classes))
    if unknown:
        raise ValueError(f"observed labels are outside class_order: {unknown}")
    if scores.shape != (len(truth), len(classes)):
        raise ValueError(
            f"y_score must have shape {(len(truth), len(classes))}, got {scores.shape}"
        )
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("y_score must contain finite values in [0, 1]")
    if not np.allclose(scores.sum(axis=1), 1.0, rtol=0.0, atol=1e-4):
        raise ValueError("y_score rows must sum to one; pass class probabilities")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(classes),
        average=None,
        zero_division=0,
    )
    per_class: dict[str, dict[str, float | int | None]] = {}
    auc_values: list[float] = []
    for index, label in enumerate(classes):
        binary_truth = (truth == label).astype(np.int8)
        auc: float | None = None
        if np.any(binary_truth == 1) and np.any(binary_truth == 0):
            auc = float(roc_auc_score(binary_truth, scores[:, index]))
            auc_values.append(auc)
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "roc_auc_ovr": auc,
        }

    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=list(classes), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predicted, labels=list(classes), average="weighted", zero_division=0)
        ),
        "log_loss": float(log_loss(truth, scores, labels=list(classes))),
        "macro_ovr_auc": float(np.mean(auc_values)) if len(auc_values) == len(classes) else None,
        "per_class": per_class,
        "class_order": list(classes),
        "support": int(len(truth)),
    }


def group_bootstrap_intervals(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    groups: Sequence[str],
    *,
    class_order: Sequence[str],
    n_resamples: int = 2_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Percentile intervals from resampling whole groups, not individual clips."""

    truth = np.asarray([str(value) for value in y_true], dtype=object)
    predicted = np.asarray([str(value) for value in y_pred], dtype=object)
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    classes = tuple(str(label) for label in class_order)
    if len(truth) == 0 or len(truth) != len(predicted) or len(truth) != len(group_values):
        raise ValueError("truth, predictions, and groups must have equal non-zero support")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_values):
        members[group].append(index)
    group_ids = sorted(members)
    if len(group_ids) < 2:
        raise ValueError("group bootstrap requires at least two independent groups")

    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    for _ in range(n_resamples):
        selected_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
        selected = [index for group in selected_groups for index in members[str(group)]]
        sampled_truth = truth[selected]
        sampled_predicted = predicted[selected]
        metrics["accuracy"].append(float(accuracy_score(sampled_truth, sampled_predicted)))
        metrics["macro_f1"].append(
            float(
                f1_score(
                    sampled_truth,
                    sampled_predicted,
                    labels=list(classes),
                    average="macro",
                    zero_division=0,
                )
            )
        )

    alpha = (1.0 - confidence_level) / 2.0
    intervals = {
        metric: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }
        for metric, values in metrics.items()
    }
    return {
        "method": "percentile_group_bootstrap",
        "confidence_level": float(confidence_level),
        "n_resamples": int(n_resamples),
        "group_count": len(group_ids),
        "seed": int(seed),
        "accuracy": intervals["accuracy"],
        "macro_f1": intervals["macro_f1"],
    }

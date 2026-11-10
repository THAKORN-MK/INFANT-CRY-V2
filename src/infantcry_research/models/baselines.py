"""Reference classifiers for controlled Stage 2 comparisons."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


class MajorityClassifier:
    """Always predict the most frequent label observed in the training labels."""

    def __init__(self, class_order: Sequence[str]) -> None:
        self.class_order = tuple(class_order)
        if not self.class_order or len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order must contain unique labels")
        self.classes_ = np.asarray(self.class_order, dtype=object)
        self.class_counts_: Counter[str] | None = None
        self.majority_class_: str | None = None

    def fit(self, features: Any, labels: Sequence[str]) -> MajorityClassifier:
        del features  # This baseline intentionally uses no audio features.
        values = [str(label) for label in labels]
        if not values:
            raise ValueError("MajorityClassifier requires at least one training label")
        unknown = sorted(set(values) - set(self.class_order))
        if unknown:
            raise ValueError(f"Training labels outside configured class order: {unknown}")
        self.class_counts_ = Counter(values)
        order = {label: index for index, label in enumerate(self.class_order)}
        self.majority_class_ = min(
            self.class_counts_, key=lambda label: (-self.class_counts_[label], order[label])
        )
        return self

    def predict(self, features: Any) -> np.ndarray:
        if self.majority_class_ is None:
            raise ValueError("MajorityClassifier must be fitted before predict")
        count = _row_count(features)
        return np.full(count, self.majority_class_, dtype=object)

    def predict_proba(self, features: Any) -> np.ndarray:
        predictions = self.predict(features)
        probabilities = np.zeros((len(predictions), len(self.classes_)), dtype=np.float32)
        class_index = self.class_order.index(self.majority_class_ or "")
        probabilities[:, class_index] = 1.0
        return probabilities


def build_mfcc_svm(
    *,
    n_mfcc: int = 40,
    random_state: int = 42,
    kernel: str = "rbf",
    class_weight: str | dict[str, float] | None = "balanced",
) -> Pipeline:
    """Create a train-fitted standardization + RBF-SVM baseline.

    Input tensors may contain either MFCC-only features `(N, n_mfcc, T)` or the
    Stage 2 tensor `(N, 196, T, 1)`; in the latter case only the leading MFCC
    bins are pooled. Mean/std pooling is performed per clip before train-only
    scaling.
    """

    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be positive")
    if kernel not in {"linear", "poly", "rbf", "sigmoid"}:
        raise ValueError("kernel must be one of linear, poly, rbf, or sigmoid")
    return Pipeline(
        steps=(
            (
                "mfcc_pool",
                FunctionTransformer(
                    _pool_mfcc_statistics,
                    kw_args={"n_mfcc": n_mfcc},
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=kernel,
                    class_weight=class_weight,
                    probability=True,
                    random_state=random_state,
                ),
            ),
        )
    )


def build_logmel_cnn(
    tf: Any,
    *,
    input_shape: tuple[int, int, int] = (64, 128, 1),
    num_classes: int = 5,
    config: dict[str, Any] | None = None,
) -> Any:
    """Build a compact Log-Mel-only CNN baseline from the frozen model config."""

    if len(input_shape) != 3 or min(input_shape) <= 0 or num_classes < 2:
        raise ValueError("input_shape must be positive HWC dimensions and num_classes >= 2")
    options = config or {}
    filters = tuple(int(value) for value in options.get("cnn_filters", (16, 32, 64)))
    dense_units = tuple(int(value) for value in options.get("dense_units", (64,)))
    dropout = float(options.get("dropout", 0.25))
    if not filters or any(value <= 0 for value in filters):
        raise ValueError("cnn_filters must contain positive widths")
    if not dense_units or any(value <= 0 for value in dense_units):
        raise ValueError("dense_units must contain positive widths")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape, name="logmel_input")
    x = inputs
    for index, width in enumerate(filters, start=1):
        x = layers.Conv2D(
            width,
            3,
            padding="same",
            activation="relu",
            name=f"logmel_conv_{index}_{width}",
        )(x)
        if index < len(filters):
            x = layers.MaxPooling2D(pool_size=(2, 2), name=f"logmel_pool_{index}")(x)
    x = layers.GlobalAveragePooling2D(name="logmel_global_pool")(x)
    for index, width in enumerate(dense_units, start=1):
        x = layers.Dense(width, activation="relu", name=f"logmel_dense_{index}_{width}")(x)
        if dropout:
            x = layers.Dropout(dropout, name=f"logmel_dropout_{index}")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="class_probabilities"
    )(x)
    return tf.keras.Model(inputs, outputs, name="logmel_cnn_baseline")


def _pool_mfcc_statistics(features: np.ndarray, *, n_mfcc: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3:
        raise ValueError(f"Expected features shaped (N, bins, frames[,1]), got {values.shape}")
    if values.shape[1] < n_mfcc:
        raise ValueError(f"Feature tensor has {values.shape[1]} bins; expected at least {n_mfcc}")
    mfcc = values[:, :n_mfcc, :]
    mean = np.mean(mfcc, axis=2)
    standard_deviation = np.std(mfcc, axis=2)
    return np.concatenate((mean, standard_deviation), axis=1).astype(np.float32, copy=False)


def _row_count(features: Any) -> int:
    if features is None:
        return 1
    values = np.asarray(features)
    if values.ndim == 0:
        return 1
    return int(values.shape[0])

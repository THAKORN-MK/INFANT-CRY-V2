"""CNN-BiLSTM-Temporal-Attention Stage 2 candidate model."""

from __future__ import annotations

from typing import Any

from infantcry_research.models.layers import TemporalAttention


def build_cnn_bilstm_attention(
    tf: Any,
    *,
    input_shape: tuple[int, int, int] = (196, 128, 1),
    num_classes: int = 5,
    config: dict[str, Any] | None = None,
) -> Any:
    """Build a single-branch CNN-BiLSTM-attention classifier.

    The frequency axis is downsampled while time is preserved, then converted
    to a time-major sequence for the bidirectional recurrent layers.
    """

    options = config or {}
    filters = tuple(int(value) for value in options.get("cnn_filters", (32, 64, 128, 256)))
    recurrent_units = tuple(int(value) for value in options.get("bilstm_units", (128, 64)))
    dense_units = tuple(int(value) for value in options.get("dense_units", (256, 128, 64)))
    attention_units = int(options.get("attention_units", 64))
    dropout = float(options.get("dropout", 0.25))
    if len(input_shape) != 3 or min(input_shape) <= 0 or num_classes < 2:
        raise ValueError("input_shape must be positive HWC dimensions and num_classes >= 2")
    if not filters or any(value <= 0 for value in filters):
        raise ValueError("cnn_filters must contain positive widths")
    if len(recurrent_units) != 2 or any(value <= 0 for value in recurrent_units):
        raise ValueError("bilstm_units must contain two positive widths")
    if not dense_units or any(value <= 0 for value in dense_units):
        raise ValueError("dense_units must contain positive widths")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape, name="stage2_feature_input")
    x = inputs
    frequency_bins = input_shape[0]
    for index, width in enumerate(filters, start=1):
        x = layers.Conv2D(
            width,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
            name=f"cnn_conv_{index}",
        )(x)
        x = layers.BatchNormalization(name=f"cnn_bn_{index}")(x)
        x = layers.Activation("relu", name=f"cnn_relu_{index}")(x)
        x = layers.MaxPooling2D(
            pool_size=(2, 1), padding="valid", name=f"cnn_frequency_pool_{index}"
        )(x)
        frequency_bins //= 2
    if frequency_bins <= 0:
        raise ValueError("input frequency dimension is too small for the configured CNN pooling")

    # Conv tensor is [batch, frequency, time, channels]; make time the sequence axis.
    x = layers.Permute((2, 1, 3), name="time_major_permutation")(x)
    x = layers.Reshape(
        (input_shape[1], frequency_bins * filters[-1]), name="time_sequence_features"
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(recurrent_units[0], return_sequences=True, name="bilstm_forward_128"),
        name="bilstm_128",
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(recurrent_units[1], return_sequences=True, name="bilstm_forward_64"),
        name="bilstm_64",
    )(x)
    x = TemporalAttention(attention_units=attention_units, name="temporal_attention")(x)
    for index, width in enumerate(dense_units, start=1):
        x = layers.Dense(width, activation="relu", name=f"dense_{index}_{width}")(x)
        if dropout:
            x = layers.Dropout(dropout, name=f"dense_dropout_{index}")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="class_probabilities"
    )(x)
    return tf.keras.Model(inputs, outputs, name="cnn_bilstm_temporal_attention_stage2")

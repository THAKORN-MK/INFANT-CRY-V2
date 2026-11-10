"""Paper-informed residual CNN + LSTM classifier for Stage 2."""

from __future__ import annotations

from typing import Any


def build_mmt_reslstm(
    tf: Any,
    *,
    input_shape: tuple[int, int, int] = (84, 48, 1),
    num_classes: int = 5,
    config: dict[str, Any] | None = None,
) -> Any:
    """Build the MMT + ResLSTM candidate using TensorFlow/Keras.

    The CNN has the paper-described 7x7 stem and four residual stages. Its
    feature maps are converted into a time-major sequence before three
    unidirectional LSTM layers. This is ResLSTM, not the separate
    CNN-BiLSTM-attention candidate.
    """

    options = config or {}
    stem_filters = int(options.get("stem_filters", 32))
    residual_filters = tuple(
        int(value) for value in options.get("residual_filters", (64, 128, 256, 512))
    )
    lstm_units = tuple(int(value) for value in options.get("lstm_units", (256, 128, 64)))
    dropout = float(options.get("dropout", 0.3))

    if len(input_shape) != 3 or any(int(value) <= 0 for value in input_shape):
        raise ValueError("input_shape must contain three positive HWC dimensions")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if stem_filters <= 0:
        raise ValueError("stem_filters must be positive")
    if len(residual_filters) != 4 or any(value <= 0 for value in residual_filters):
        raise ValueError("residual_filters must contain four positive widths")
    if len(lstm_units) != 3 or any(value <= 0 for value in lstm_units):
        raise ValueError("lstm_units must contain three positive widths")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if min(input_shape[:2]) < 4:
        raise ValueError("input frequency and time dimensions must be >= 4 for stem pooling")

    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape, name="mmt_feature_input")
    x = layers.Conv2D(
        stem_filters,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name="reslstm_stem_conv7x7",
    )(inputs)
    x = layers.BatchNormalization(name="reslstm_stem_bn")(x)
    x = layers.Activation("relu", name="reslstm_stem_relu")(x)
    x = layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="same", name="reslstm_stem_maxpool"
    )(x)

    for index, width in enumerate(residual_filters, start=1):
        stride = 2 if index == 1 else 1
        x = _residual_block(
            tf,
            x,
            filters=width,
            stride=stride,
            dropout=dropout,
            name=f"reslstm_residual_stage_{index}_{width}",
        )

    # Conv output is [batch, frequency, time, channels]; expose time to LSTM.
    x = layers.Permute((2, 1, 3), name="reslstm_time_major_permutation")(x)
    time_steps = int(x.shape[1])
    frequency_steps = int(x.shape[2])
    channels = int(x.shape[3])
    x = layers.Reshape(
        (time_steps, frequency_steps * channels), name="reslstm_time_sequence"
    )(x)

    for index, units in enumerate(lstm_units, start=1):
        x = layers.LSTM(
            units,
            return_sequences=index < len(lstm_units),
            name=f"reslstm_lstm_{index}_{units}",
        )(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="reslstm_class_probabilities",
    )(x)
    return tf.keras.Model(inputs, outputs, name="mmt_reslstm_stage2")


def _residual_block(
    tf: Any,
    inputs: Any,
    *,
    filters: int,
    stride: int,
    dropout: float,
    name: str,
) -> Any:
    layers = tf.keras.layers
    shortcut = inputs

    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    if dropout:
        x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    input_channels = int(inputs.shape[-1])
    if stride != 1 or input_channels != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=(stride, stride),
            padding="same",
            use_bias=False,
            name=f"{name}_projection_conv1x1",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_projection_bn")(shortcut)

    x = layers.Add(name=f"{name}_skip_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_output_relu")(x)

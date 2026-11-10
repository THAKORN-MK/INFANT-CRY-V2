"""Paper-informed Stage 2 architectures for controlled local comparisons.

These builders implement explicit padding as described by Qiu et al. (2024)
while retaining the project's input tensors, class count, and training protocol.
They are adaptations for comparison, not claims of an exact reproduction.
"""

from __future__ import annotations

from typing import Any


def build_qiu_mfcc_reslstm(
    tf: Any,
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    config: dict[str, Any] | None = None,
) -> Any:
    """Build an MFCC-only ResLSTM comparison candidate."""

    return _build_reslstm(
        tf,
        input_shape=input_shape,
        num_classes=num_classes,
        config=config,
        input_name="mfcc_feature_input",
        model_name="qiu_mfcc_reslstm_stage2",
    )


def build_qiu_mmt_resnet(
    tf: Any,
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    config: dict[str, Any] | None = None,
) -> Any:
    """Build an MMT residual-CNN candidate with global-pooling head."""

    options = _validated_options(input_shape, num_classes, config)
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape, name="mmt_feature_input")
    x = _residual_feature_extractor(tf, inputs, options)
    x = layers.GlobalAveragePooling2D(name="qiu_resnet_global_average_pool")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="qiu_resnet_class_probabilities",
    )(x)
    return tf.keras.Model(inputs, outputs, name="qiu_mmt_resnet_stage2")


def build_qiu_mmt_lstm(
    tf: Any,
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    config: dict[str, Any] | None = None,
) -> Any:
    """Build an MMT-only stacked LSTM on frequency vectors per time frame."""

    options = _validated_options(input_shape, num_classes, config)
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape, name="mmt_feature_input")
    # Input order is [frequency, time, channel]; each LSTM step receives one
    # time frame with the frequency/channel axes flattened into a feature vector.
    x = layers.Permute((2, 1, 3), name="qiu_lstm_time_major_permutation")(inputs)
    time_steps = int(input_shape[1])
    frame_width = int(input_shape[0]) * int(input_shape[2])
    x = layers.Reshape(
        (time_steps, frame_width), name="qiu_lstm_frequency_frame_sequence"
    )(x)
    for index, units in enumerate(options["lstm_units"], start=1):
        x = layers.LSTM(
            units,
            return_sequences=index < len(options["lstm_units"]),
            name=f"qiu_lstm_layer_{index}_{units}",
        )(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="qiu_lstm_class_probabilities",
    )(x)
    return tf.keras.Model(inputs, outputs, name="qiu_mmt_lstm_stage2")


def build_qiu_mmt_reslstm(
    tf: Any,
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    config: dict[str, Any] | None = None,
) -> Any:
    """Build an explicitly padded MMT + residual CNN + LSTM candidate."""

    return _build_reslstm(
        tf,
        input_shape=input_shape,
        num_classes=num_classes,
        config=config,
        input_name="mmt_feature_input",
        model_name="qiu_mmt_reslstm_stage2",
    )


def _build_reslstm(
    tf: Any,
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    config: dict[str, Any] | None,
    input_name: str,
    model_name: str,
) -> Any:
    options = _validated_options(input_shape, num_classes, config)
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape, name=input_name)
    x = _residual_feature_extractor(tf, inputs, options)

    # Feature extractor output is [batch, frequency, time, channels].
    x = layers.Permute((2, 1, 3), name="qiu_reslstm_time_major_permutation")(x)
    time_steps = int(x.shape[1])
    frequency_steps = int(x.shape[2])
    channels = int(x.shape[3])
    x = layers.Reshape(
        (time_steps, frequency_steps * channels),
        name="qiu_reslstm_time_sequence",
    )(x)
    for index, units in enumerate(options["lstm_units"], start=1):
        x = layers.LSTM(
            units,
            return_sequences=index < len(options["lstm_units"]),
            name=f"qiu_reslstm_lstm_{index}_{units}",
        )(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="qiu_reslstm_class_probabilities",
    )(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


def _residual_feature_extractor(tf: Any, inputs: Any, options: dict[str, Any]) -> Any:
    layers = tf.keras.layers
    x = layers.ZeroPadding2D(
        padding=((2, 2), (2, 2)), name="qiu_stem_zero_padding"
    )(inputs)
    x = layers.Conv2D(
        options["stem_filters"],
        kernel_size=(7, 7),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        name="qiu_stem_conv7x7",
    )(x)
    x = layers.BatchNormalization(name="qiu_stem_batch_normalization")(x)
    x = layers.Activation("relu", name="qiu_stem_relu")(x)
    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same",
        name="qiu_stem_max_pool",
    )(x)

    for index, filters in enumerate(options["residual_filters"], start=1):
        x = _explicitly_padded_residual_block(
            tf,
            x,
            filters=filters,
            stride=2 if index == 1 else 1,
            dropout=options["dropout"],
            name=f"qiu_stage_{index}",
        )
    return x


def _explicitly_padded_residual_block(
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

    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=f"{name}_zero_padding1"
    )(inputs)
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="valid",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    if dropout:
        x = layers.Dropout(dropout, name=f"{name}_dropout")(x)

    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=f"{name}_zero_padding2"
    )(x)
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    input_channels = int(inputs.shape[-1])
    if stride != 1 or input_channels != filters:
        # A valid 1x1 stride projection produces ceil(n / stride), matching
        # the explicitly padded 3x3 main path for both even and odd sizes.
        shortcut = layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=(stride, stride),
            padding="valid",
            use_bias=False,
            name=f"{name}_projection_conv1x1",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_projection_bn")(shortcut)

    x = layers.Add(name=f"{name}_skip_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_output_relu")(x)


def _validated_options(
    input_shape: tuple[int, int, int],
    num_classes: int,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    if len(input_shape) != 3 or any(int(value) <= 0 for value in input_shape):
        raise ValueError("input_shape must contain three positive HWC dimensions")
    if min(input_shape[:2]) < 4:
        raise ValueError("input frequency and time dimensions must be >= 4 for stem pooling")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    options = config or {}
    stem_filters = int(options.get("stem_filters", 32))
    residual_filters = tuple(
        int(value) for value in options.get("residual_filters", (64, 128, 256, 512))
    )
    lstm_units = tuple(
        int(value) for value in options.get("lstm_units", (256, 128, 64))
    )
    dropout = float(options.get("dropout", 0.3))

    if stem_filters <= 0:
        raise ValueError("stem_filters must be positive")
    if len(residual_filters) != 4 or any(value <= 0 for value in residual_filters):
        raise ValueError("residual_filters must contain four positive widths")
    if len(lstm_units) != 3 or any(value <= 0 for value in lstm_units):
        raise ValueError("lstm_units must contain three positive widths")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    return {
        "stem_filters": stem_filters,
        "residual_filters": residual_filters,
        "lstm_units": lstm_units,
        "dropout": dropout,
    }

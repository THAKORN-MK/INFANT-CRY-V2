"""Serializable neural-network layers used by the candidate models."""

from __future__ import annotations

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised only in installations without TF
    tf = None


if tf is not None:

    @tf.keras.utils.register_keras_serializable(package="infantcry_research")
    class TemporalAttention(tf.keras.layers.Layer):
        """Learn a normalized temporal weighting over a sequence of hidden states."""

        def __init__(self, attention_units: int = 64, **kwargs):
            super().__init__(**kwargs)
            if attention_units <= 0:
                raise ValueError("attention_units must be positive")
            self.attention_units = int(attention_units)
            self.projection = tf.keras.layers.Dense(
                self.attention_units,
                activation="tanh",
                dtype=self.dtype_policy,
                name="attention_projection",
            )
            self.scorer = tf.keras.layers.Dense(
                1,
                use_bias=False,
                dtype=self.dtype_policy,
                name="attention_score",
            )

        def build(self, input_shape):
            sequence_shape = tf.TensorShape(input_shape)
            self.projection.build(sequence_shape)
            projected_shape = sequence_shape[:-1].concatenate(self.attention_units)
            self.scorer.build(projected_shape)
            super().build(input_shape)

        def call(self, inputs):
            scores = self.scorer(self.projection(inputs))
            weights = tf.nn.softmax(scores, axis=1)
            weights = tf.cast(weights, inputs.dtype)
            return tf.reduce_sum(inputs * weights, axis=1)

        def get_config(self):
            return {**super().get_config(), "attention_units": self.attention_units}

else:

    class TemporalAttention:  # pragma: no cover - import guard for optional TensorFlow
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ImportError("TensorFlow is required to build the attention model")

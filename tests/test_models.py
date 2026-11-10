"""Shape and contract tests for baseline and neural model builders."""

from __future__ import annotations

import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np
import tensorflow as tf

from infantcry_research.models.baselines import MajorityClassifier, build_logmel_cnn, build_mfcc_svm
from infantcry_research.models.cnn_bilstm_attention import build_cnn_bilstm_attention
from infantcry_research.models.layers import TemporalAttention


LABELS = ("eairh", "eh", "heh", "neh", "owh")


class ModelContractTests(unittest.TestCase):
    def test_majority_classifier_uses_training_labels_and_stable_tie_break(self) -> None:
        model = MajorityClassifier(class_order=LABELS).fit(None, ["eh", "eairh"])

        predictions = model.predict(np.zeros((3, 196, 128, 1), dtype=np.float32))

        self.assertEqual(predictions.tolist(), ["eairh"] * 3)
        self.assertEqual(model.classes_.tolist(), list(LABELS))
        self.assertEqual(model.predict_proba(None).shape, (1, len(LABELS)))

    def test_mfcc_svm_pools_only_mfcc_bins_and_scaler_fits_training_examples(self) -> None:
        model = build_mfcc_svm(n_mfcc=40, random_state=7)
        train_x = np.random.default_rng(4).normal(size=(20, 196, 32, 1)).astype(np.float32)
        train_y = np.asarray([LABELS[index % len(LABELS)] for index in range(20)])
        validation_x = np.random.default_rng(8).normal(size=(5, 196, 32, 1)).astype(np.float32)

        model.fit(train_x, train_y)
        fitted_mean = model.named_steps["scale"].mean_.copy()
        probabilities = model.predict_proba(validation_x)

        self.assertEqual(model.named_steps["mfcc_pool"].transform(train_x).shape, (20, 80))
        self.assertEqual(probabilities.shape, (5, len(LABELS)))
        np.testing.assert_array_equal(model.named_steps["scale"].mean_, fitted_mean)
        self.assertEqual(tuple(model.classes_), tuple(sorted(LABELS)))

    def test_mfcc_svm_uses_registered_kernel_and_class_weight(self) -> None:
        model = build_mfcc_svm(n_mfcc=40, kernel="linear", class_weight=None)

        self.assertEqual(model.named_steps["svc"].kernel, "linear")
        self.assertIsNone(model.named_steps["svc"].class_weight)

    def test_logmel_cnn_returns_five_class_softmax(self) -> None:
        model = build_logmel_cnn(
            tf,
            input_shape=(64, 128, 1),
            num_classes=5,
            config={"cnn_filters": [8, 16], "dense_units": [32], "dropout": 0.1},
        )

        probabilities = model(np.zeros((2, 64, 128, 1), dtype=np.float32), training=False).numpy()

        self.assertEqual(probabilities.shape, (2, 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)
        self.assertEqual(
            [layer.filters for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)],
            [8, 16],
        )

    def test_cnn_bilstm_attention_shape_layers_and_softmax_contract(self) -> None:
        model = build_cnn_bilstm_attention(tf, input_shape=(196, 128, 1), num_classes=5)

        probabilities = model(np.zeros((1, 196, 128, 1), dtype=np.float32), training=False).numpy()

        self.assertEqual(tuple(model.input_shape[1:]), (196, 128, 1))
        self.assertEqual(probabilities.shape, (1, 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(1), atol=1e-6)
        self.assertTrue(any("attention" in layer.name for layer in model.layers))
        self.assertEqual(
            [layer.filters for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)],
            [32, 64, 128, 256],
        )

    def test_attention_model_can_round_trip_in_keras_format(self) -> None:
        model = build_cnn_bilstm_attention(tf, input_shape=(196, 128, 1), num_classes=5)
        sample = np.zeros((1, 196, 128, 1), dtype=np.float32)
        expected = model(sample, training=False).numpy()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            actual = restored(sample, training=False).numpy()

        np.testing.assert_allclose(actual, expected, atol=1e-6)


class QiuComparatorModelTests(unittest.TestCase):
    @staticmethod
    def _small_config() -> dict[str, object]:
        return {
            "stem_filters": 4,
            "residual_filters": [4, 8, 8, 8],
            "lstm_units": [8, 4, 2],
            "dropout": 0.1,
        }

    @staticmethod
    def _assert_probabilities(
        test: unittest.TestCase,
        model: object,
        input_shape: tuple[int, int, int],
    ) -> None:
        probabilities = model(
            np.zeros((1, *input_shape), dtype=np.float32), training=False
        ).numpy()
        test.assertEqual(probabilities.shape, (1, 5))
        test.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(1), atol=1e-6)

    def test_reslstm_variants_preserve_even_and_odd_shapes_with_explicit_padding(self) -> None:
        module = import_module("infantcry_research.models.qiu_comparators")
        builders = (
            (module.build_qiu_mfcc_reslstm, "mfcc_feature_input"),
            (module.build_qiu_mmt_reslstm, "mmt_feature_input"),
        )

        for input_shape in ((42, 24, 1), (43, 25, 1)):
            for builder, input_name in builders:
                with self.subTest(builder=builder.__name__, input_shape=input_shape):
                    model = builder(
                        tf,
                        input_shape=input_shape,
                        num_classes=5,
                        config=self._small_config(),
                    )
                    self.assertEqual(tuple(model.input_shape[1:]), input_shape)
                    self.assertEqual(model.input.name.split(":")[0], input_name)
                    self._assert_probabilities(self, model, input_shape)

                    stem_padding = model.get_layer("qiu_stem_zero_padding")
                    self.assertIsInstance(stem_padding, tf.keras.layers.ZeroPadding2D)
                    self.assertEqual(stem_padding.padding, ((2, 2), (2, 2)))
                    self.assertEqual(model.get_layer("qiu_stem_conv7x7").padding, "valid")

                    for layer in model.layers:
                        is_three_by_three = (
                            isinstance(layer, tf.keras.layers.Conv2D)
                            and layer.kernel_size == (3, 3)
                        )
                        if is_three_by_three:
                            self.assertEqual(layer.padding, "valid", layer.name)
                            padding_name = layer.name.replace("_conv", "_zero_padding")
                            padding = model.get_layer(padding_name)
                            self.assertIsInstance(padding, tf.keras.layers.ZeroPadding2D)
                            self.assertEqual(padding.padding, ((1, 1), (1, 1)))

    def test_mmt_resnet_has_global_pooling_without_recurrent_layers(self) -> None:
        module = import_module("infantcry_research.models.qiu_comparators")
        model = module.build_qiu_mmt_resnet(
            tf,
            input_shape=(84, 48, 1),
            num_classes=5,
            config=self._small_config(),
        )

        self._assert_probabilities(self, model, (84, 48, 1))
        self.assertTrue(
            any(
                isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
                for layer in model.layers
            )
        )
        self.assertFalse(any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers))
        self.assertTrue(any(isinstance(layer, tf.keras.layers.Conv2D) for layer in model.layers))

    def test_mmt_lstm_flattens_frequency_per_frame_without_convolutions(self) -> None:
        module = import_module("infantcry_research.models.qiu_comparators")
        model = module.build_qiu_mmt_lstm(
            tf,
            input_shape=(84, 48, 1),
            num_classes=5,
            config=self._small_config(),
        )

        self._assert_probabilities(self, model, (84, 48, 1))
        self.assertEqual(
            [layer.units for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)],
            [8, 4, 2],
        )
        self.assertFalse(any(isinstance(layer, tf.keras.layers.Conv2D) for layer in model.layers))
        self.assertTrue(any(isinstance(layer, tf.keras.layers.Permute) for layer in model.layers))
        self.assertTrue(any(isinstance(layer, tf.keras.layers.Reshape) for layer in model.layers))

    def test_qiu_reslstm_round_trips_in_keras_format(self) -> None:
        module = import_module("infantcry_research.models.qiu_comparators")
        model = module.build_qiu_mmt_reslstm(
            tf,
            input_shape=(84, 48, 1),
            num_classes=5,
            config=self._small_config(),
        )
        sample = np.zeros((1, 84, 48, 1), dtype=np.float32)
        expected = model(sample, training=False).numpy()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qiu-mmt-reslstm.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            actual = restored(sample, training=False).numpy()

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_temporal_attention_round_trips_across_mixed_precision_policy(self) -> None:
        previous_policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            inputs = tf.keras.layers.Input(shape=(5, 3), name="sequence_input")
            sequence = tf.keras.layers.Dense(4, name="sequence_projection")(inputs)
            attended = TemporalAttention(attention_units=2, name="attention")(sequence)
            outputs = tf.keras.layers.Dense(
                5, activation="softmax", dtype="float32", name="class_probabilities"
            )(attended)
            model = tf.keras.Model(inputs, outputs)
            sample = np.ones((2, 5, 3), dtype=np.float32)
            expected = model(sample, training=False).numpy()

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mixed-precision-attention.keras"
                model.save(path)
                tf.keras.mixed_precision.set_global_policy("float32")
                restored = tf.keras.models.load_model(path)
                actual = restored(sample, training=False).numpy()

            self.assertEqual(actual.shape, (2, 5))
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual.sum(axis=1), np.ones(2), atol=1e-5)
            np.testing.assert_allclose(actual, expected, atol=1e-5)
        finally:
            tf.keras.mixed_precision.set_global_policy(previous_policy)

    def test_mmt_reslstm_has_residual_stages_stacked_lstm_and_softmax(self) -> None:
        module = import_module("infantcry_research.models.reslstm")
        model = module.build_mmt_reslstm(
            tf, input_shape=(84, 48, 1), num_classes=5
        )

        probabilities = model(
            np.zeros((1, 84, 48, 1), dtype=np.float32), training=False
        ).numpy()

        self.assertEqual(tuple(model.input_shape[1:]), (84, 48, 1))
        self.assertEqual(probabilities.shape, (1, 5))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(1), atol=1e-6)
        convolutions = [
            layer.filters
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        self.assertEqual(convolutions[0], 32)
        for width in (64, 128, 256, 512):
            self.assertIn(width, convolutions)
        self.assertEqual(
            [layer.units for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)],
            [256, 128, 64],
        )
        self.assertFalse(
            any(isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.layers)
        )
        self.assertFalse(any("attention" in layer.name.lower() for layer in model.layers))

    def test_mmt_reslstm_round_trips_with_configured_small_widths(self) -> None:
        module = import_module("infantcry_research.models.reslstm")
        model = module.build_mmt_reslstm(
            tf,
            input_shape=(84, 48, 1),
            num_classes=5,
            config={
                "stem_filters": 4,
                "residual_filters": [4, 8, 12, 16],
                "lstm_units": [16, 8, 4],
                "dropout": 0.1,
            },
        )
        sample = np.zeros((1, 84, 48, 1), dtype=np.float32)
        expected = model(sample, training=False).numpy()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mmt-reslstm.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            actual = restored(sample, training=False).numpy()

        np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

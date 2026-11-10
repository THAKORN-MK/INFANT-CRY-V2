"""Tests for model-specific locked-test inference contracts."""

from __future__ import annotations

import unittest

from infantcry_research.evaluation import comparison


class CandidateInputShapeTests(unittest.TestCase):
    def test_qiu_candidates_are_registered_as_keras_models(self) -> None:
        self.assertTrue(
            {
                "qiu_mfcc_reslstm",
                "qiu_mmt_resnet",
                "qiu_mmt_lstm",
                "qiu_mmt_reslstm",
            }.issubset(comparison.NEURAL_CANDIDATES)
        )

    def test_logmel_uses_candidate_shape_instead_of_full_feature_tensor_shape(self) -> None:
        frozen_run_config = {
            "input_shape": [196, 128, 1],
            "candidate_config": {"id": "logmel_cnn", "input_shape": [64, 128, 1]},
        }

        shape = getattr(comparison, "_candidate_model_input_shape", lambda _config: None)(
            frozen_run_config
        )

        self.assertEqual(shape, (64, 128, 1))

    def test_candidates_without_model_override_use_top_level_model_shape(self) -> None:
        frozen_run_config = {
            "input_shape": [196, 128, 1],
            "candidate_config": {"id": "cnn_bilstm_attention"},
        }

        shape = getattr(comparison, "_candidate_model_input_shape", lambda _config: None)(
            frozen_run_config
        )

        self.assertEqual(shape, (196, 128, 1))


if __name__ == "__main__":
    unittest.main()

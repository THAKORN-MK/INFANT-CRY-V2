"""Signal contract tests using generated sine waves, never dataset records."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

import infantcry_research.audio.features as feature_module
from infantcry_research.audio.features import (
    AudioConfig,
    AudioContractError,
    extract_stage2_features,
    load_and_preprocess_audio,
    preprocess_waveform,
)


def _sine(sample_rate: int, seconds: float = 1.2) -> np.ndarray:
    samples = int(sample_rate * seconds)
    time = np.arange(samples, dtype=np.float32) / sample_rate
    return (0.55 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AudioConfig()

    def test_preprocess_downmixes_and_resamples_to_configured_mono_rate(self) -> None:
        stereo = np.column_stack((_sine(44_100), _sine(44_100) * 0.5))

        mono = preprocess_waveform(stereo, 44_100, self.config)

        self.assertEqual(mono.ndim, 1)
        self.assertAlmostEqual(len(mono) / self.config.sample_rate, 1.2, delta=0.03)
        self.assertLessEqual(float(np.max(np.abs(mono))), 1.0)
        self.assertAlmostEqual(float(np.max(np.abs(mono))), 1.0, places=5)

    def test_stage2_feature_tensor_has_expected_shape_order_dtype_and_finite_values(self) -> None:
        waveform = preprocess_waveform(_sine(self.config.sample_rate), self.config.sample_rate, self.config)

        tensor = extract_stage2_features(waveform, self.config.sample_rate, self.config)

        self.assertEqual(tensor.shape, (196, 128, 1))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())

    def test_mmt_tensor_uses_configured_feature_contract(self) -> None:
        config = AudioConfig.from_mapping(
            {
                "feature_set": "mmt",
                "n_mfcc": 42,
                "n_mels": 42,
                "mmt_frames": 24,
                "n_fft": 2_048,
                "hop_length": 512,
            }
        )
        waveform = preprocess_waveform(
            _sine(config.sample_rate), config.sample_rate, config
        )

        tensor = feature_module.extract_audio_features(
            waveform, config.sample_rate, config
        )

        self.assertEqual(config.tensor_shape, (84, 48, 1))
        self.assertEqual(tensor.shape, config.tensor_shape)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())

    def test_mmt_blocks_follow_documented_geometry_consistent_order(self) -> None:
        config = AudioConfig.from_mapping(
            {
                "feature_set": "mmt",
                "n_mfcc": 42,
                "n_mels": 42,
                "mmt_frames": 2,
                "n_fft": 16,
                "hop_length": 4,
            }
        )
        waveform = np.ones(64, dtype=np.float32)
        mfcc = np.arange(84, dtype=np.float32).reshape(42, 2)
        mel = np.full((42, 2), 11.0, dtype=np.float32)
        tonnetz = np.arange(12, dtype=np.float32).reshape(6, 2)

        with (
            mock.patch.object(
                feature_module.librosa.feature,
                "melspectrogram",
                return_value=mel,
            ),
            mock.patch.object(
                feature_module.librosa,
                "power_to_db",
                return_value=np.zeros((42, 2), dtype=np.float32),
            ),
            mock.patch.object(
                feature_module.librosa.feature,
                "mfcc",
                return_value=mfcc,
            ),
            mock.patch.object(
                feature_module.librosa.feature,
                "tonnetz",
                return_value=tonnetz,
            ),
        ):
            tensor = feature_module.extract_audio_features(
                waveform, config.sample_rate, config
            )

        expected_mfcc = np.tile(mfcc, (1, 2))
        expected_tonnetz_then_mel = np.concatenate(
            (np.repeat(tonnetz, 7, axis=0), mel), axis=1
        )
        np.testing.assert_array_equal(tensor[:42, :, 0], expected_mfcc)
        np.testing.assert_array_equal(
            tensor[42:, :, 0], expected_tonnetz_then_mel
        )

    def test_mmt_extraction_is_deterministic(self) -> None:
        config = AudioConfig.from_mapping(
            {"feature_set": "mmt", "n_mfcc": 42, "n_mels": 42, "mmt_frames": 24}
        )
        waveform = preprocess_waveform(
            _sine(config.sample_rate, seconds=2.0), config.sample_rate, config
        )

        first = feature_module.extract_audio_features(
            waveform, config.sample_rate, config
        )
        second = feature_module.extract_audio_features(
            waveform, config.sample_rate, config
        )

        np.testing.assert_array_equal(first, second)

    def test_mfcc_paper_features_are_finite_deterministic_and_fixed_shape(self) -> None:
        config = AudioConfig.from_mapping(
            {
                "feature_set": "mfcc_paper",
                "sample_rate": 16_000,
                "n_mfcc": 42,
                "n_mels": 42,
                "n_fft": 2_048,
                "hop_length": 512,
                "mfcc_frames": 24,
            }
        )
        waveform = preprocess_waveform(
            _sine(config.sample_rate), config.sample_rate, config
        )

        first = feature_module.extract_audio_features(
            waveform, config.sample_rate, config
        )
        second = feature_module.extract_audio_features(
            waveform, config.sample_rate, config
        )

        self.assertEqual(config.tensor_shape, (42, 24, 1))
        self.assertEqual(first.shape, (42, 24, 1))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_mfcc_paper_extractor_returns_only_mfcc_values(self) -> None:
        config = AudioConfig.from_mapping(
            {
                "feature_set": "mfcc_paper",
                "n_mfcc": 42,
                "n_mels": 42,
                "n_fft": 16,
                "hop_length": 4,
                "mfcc_frames": 2,
            }
        )
        waveform = np.ones(64, dtype=np.float32)
        mel_power = np.full((42, 2), 3.0, dtype=np.float32)
        mfcc = np.arange(84, dtype=np.float32).reshape(42, 2)

        with (
            mock.patch.object(
                feature_module.librosa.feature,
                "melspectrogram",
                return_value=mel_power,
            ),
            mock.patch.object(
                feature_module.librosa,
                "power_to_db",
                return_value=np.zeros((42, 2), dtype=np.float32),
            ),
            mock.patch.object(
                feature_module.librosa.feature,
                "mfcc",
                return_value=mfcc,
            ),
        ):
            tensor = feature_module.extract_audio_features(
                waveform, config.sample_rate, config
            )

        np.testing.assert_array_equal(tensor[:, :, 0], mfcc)

    def test_mfcc_paper_rejects_non_positive_frame_count(self) -> None:
        with self.assertRaisesRegex(AudioContractError, "mfcc_frames"):
            AudioConfig.from_mapping(
                {
                    "feature_set": "mfcc_paper",
                    "n_mfcc": 42,
                    "n_mels": 42,
                    "mfcc_frames": 0,
                }
            )

    def test_audio_config_rejects_unknown_feature_set(self) -> None:
        with self.assertRaisesRegex(AudioContractError, "feature_set"):
            AudioConfig.from_mapping({"feature_set": "mystery"})

    def test_feature_extraction_is_deterministic_and_fixed_length(self) -> None:
        waveform = preprocess_waveform(
            _sine(self.config.sample_rate, seconds=8.0), self.config.sample_rate, self.config
        )

        first = extract_stage2_features(waveform, self.config.sample_rate, self.config)
        second = extract_stage2_features(waveform, self.config.sample_rate, self.config)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape[1], self.config.max_frames)

    def test_rejects_waveform_with_wrong_declared_sample_rate(self) -> None:
        with self.assertRaisesRegex(AudioContractError, "sample rate"):
            extract_stage2_features(_sine(16_000), 16_000, self.config)

    def test_rejects_silence_instead_of_creating_a_fake_feature_example(self) -> None:
        with self.assertRaisesRegex(AudioContractError, "silent"):
            preprocess_waveform(np.zeros(4_000, dtype=np.float32), 16_000, self.config)

    def test_short_non_silent_audio_uses_documented_frame_padding(self) -> None:
        short = _sine(self.config.sample_rate, seconds=0.02)
        waveform = preprocess_waveform(short, self.config.sample_rate, self.config)

        tensor = extract_stage2_features(waveform, self.config.sample_rate, self.config)

        self.assertEqual(tensor.shape, (196, 128, 1))
        self.assertTrue(np.isfinite(tensor).all())

    def test_loads_stereo_wav_and_normalizes_to_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cry.wav"
            mono = _sine(44_100)
            pcm = np.clip(np.column_stack((mono, mono * 0.25)) * 32_767, -32_768, 32_767).astype("<i2")
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44_100)
                handle.writeframes(pcm.tobytes())

            loaded, rate = load_and_preprocess_audio(path, self.config)

        self.assertEqual(rate, self.config.sample_rate)
        self.assertEqual(loaded.ndim, 1)
        self.assertAlmostEqual(float(np.max(np.abs(loaded))), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

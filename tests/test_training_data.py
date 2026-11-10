"""Training input loader tests to prove only requested split IDs are read."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from infantcry_research.audio.features import AudioConfig
from infantcry_research.data.manifest import Record
from infantcry_research.training.data import load_feature_partition


class TrainingDataTests(unittest.TestCase):
    def test_loader_uses_the_selected_mmt_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mmt-train.wav"
            time = np.arange(16_000, dtype=np.float32) / 16_000
            tone = (0.6 * np.sin(2 * np.pi * 440 * time) * 32_767).astype("<i2")
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(tone.tobytes())
            record = Record(
                sample_id="mmt-train",
                path=path,
                relative_path="mmt-train.wav",
                raw_label="eh",
                tier="InfantCry-1214",
                group_id="family-1",
                duration_s=1.0,
                sha256="c" * 64,
                sample_rate=16_000,
                channels=1,
                grouping_field="family_id",
            )
            config = AudioConfig.from_mapping(
                {
                    "feature_set": "mmt",
                    "sample_rate": 16_000,
                    "n_mfcc": 42,
                    "n_mels": 42,
                    "n_fft": 512,
                    "hop_length": 128,
                    "mmt_frames": 24,
                }
            )

            features, labels, sample_ids = load_feature_partition(
                {record.sample_id: record}, (record.sample_id,), config
            )

        self.assertEqual(features.shape, (1, 84, 48, 1))
        self.assertEqual(labels, ("eh",))
        self.assertEqual(sample_ids, ("mmt-train",))

    def test_loader_reads_only_requested_partition_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.wav"
            tone = (np.sin(np.arange(16_000) * 2 * np.pi * 440 / 16_000) * 20_000).astype("<i2")
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(tone.tobytes())
            records = {
                "train-1": Record(
                    sample_id="train-1",
                    path=path,
                    relative_path="train.wav",
                    raw_label="eh",
                    tier="InfantCry-1214",
                    group_id="family-1",
                    duration_s=1.0,
                    sha256="a" * 64,
                    sample_rate=16_000,
                    channels=1,
                    grouping_field="family_id",
                ),
                "locked-test-not-opened": Record(
                    sample_id="locked-test-not-opened",
                    path=root / "must-not-be-read.wav",
                    relative_path="must-not-be-read.wav",
                    raw_label="owh",
                    tier="InfantCry-1214",
                    group_id="family-2",
                    duration_s=1.0,
                    sha256="b" * 64,
                    sample_rate=16_000,
                    channels=1,
                    grouping_field="family_id",
                ),
            }
            config = AudioConfig(
                sample_rate=16_000,
                n_mfcc=13,
                n_mels=32,
                n_chroma=12,
                n_fft=512,
                hop_length=128,
                delta_width=9,
                max_frames=32,
            )

            features, labels, sample_ids = load_feature_partition(
                records, ("train-1",), config
            )

        self.assertEqual(features.shape, (1, 83, 32, 1))
        self.assertEqual(labels, ("eh",))
        self.assertEqual(sample_ids, ("train-1",))

    def test_loader_rejects_unknown_ids_before_reading_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown sample IDs"):
            load_feature_partition({}, ("unknown",), AudioConfig())


if __name__ == "__main__":
    unittest.main()

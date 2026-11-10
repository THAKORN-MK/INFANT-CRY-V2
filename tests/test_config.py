from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from infantcry_research.audio.features import AudioConfig
from infantcry_research.config import ConfigError, load_dataset_config


VALID_LABELS = ["eairh", "eh", "heh", "neh", "owh"]


def write_config(
    root: Path,
    *,
    labels: list[str] | None = None,
    data_path: str = "data/raw/db",
    exclusions_path: str | None = None,
) -> Path:
    path = root / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_generation": "v2",
                "dataset_version": "1.0",
                "doi": "10.17632/x493z8nmwc.1",
                "dataset_root": data_path,
                "metadata_filename": "metadata.csv",
                "exclusions_file": exclusions_path,
                "class_order": labels if labels is not None else VALID_LABELS,
                "tiers": ["Dunstan-Core", "InfantCry-1214"],
                "columns": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


class DatasetConfigTests(unittest.TestCase):
    def test_stage2_candidates_resolve_named_feature_sets_and_tensor_shapes(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        model_config = yaml.safe_load(
            (project_root / "configs" / "models_stage2.yaml").read_text(encoding="utf-8")
        )

        self.assertIn("feature_sets", model_config)
        profiles = model_config["feature_sets"]
        self.assertEqual(
            AudioConfig.from_mapping(profiles["stage2_hybrid"]).tensor_shape,
            (196, 128, 1),
        )
        self.assertEqual(
            AudioConfig.from_mapping(profiles["mmt"]).tensor_shape,
            (84, 48, 1),
        )
        candidates = {item["id"]: item for item in model_config["models"]}
        self.assertEqual(candidates["mmt_reslstm"]["feature_set"], "mmt")
        self.assertEqual(candidates["cnn_bilstm_attention"]["feature_set"], "stage2_hybrid")

    def test_mfcc_paper_feature_contract_has_42_bins_and_24_frames(self) -> None:
        config = AudioConfig.from_mapping(
            {
                "feature_set": "mfcc_paper",
                "n_mfcc": 42,
                "n_mels": 42,
                "mfcc_frames": 24,
            }
        )

        self.assertEqual(config.tensor_shape, (42, 24, 1))

    def test_qiu_comparison_config_freezes_four_registered_conditions(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        model_config = yaml.safe_load(
            (project_root / "configs" / "models_stage2_qiu_comparison.yaml").read_text(
                encoding="utf-8"
            )
        )
        candidates = {
            item["id"]: item for item in model_config["models"] if item.get("enabled") is True
        }
        expected = {
            "qiu_mfcc_reslstm",
            "qiu_mmt_resnet",
            "qiu_mmt_lstm",
            "qiu_mmt_reslstm",
        }

        self.assertEqual(set(candidates), expected)
        self.assertEqual(
            AudioConfig.from_mapping(model_config["feature_sets"]["mfcc_paper"]).tensor_shape,
            (42, 24, 1),
        )
        self.assertEqual(
            AudioConfig.from_mapping(model_config["feature_sets"]["mmt"]).tensor_shape,
            (84, 48, 1),
        )
        self.assertEqual(candidates["qiu_mfcc_reslstm"]["feature_set"], "mfcc_paper")
        self.assertTrue(
            all(
                candidates[candidate]["feature_set"] == "mmt"
                for candidate in expected - {"qiu_mfcc_reslstm"}
            )
        )
        self.assertEqual(
            {candidate["architecture"] for candidate in candidates.values()}, expected
        )
        self.assertEqual(
            model_config["training"],
            {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "early_stopping_patience": 8,
                "deterministic_ops": True,
                "optimizer": "Adam",
                "selection_metric": "validation_macro_f1",
            },
        )
        self.assertTrue(
            all(
                candidate["device"] == "gpu" and candidate["mixed_precision"] is True
                for candidate in candidates.values()
            )
        )

    def test_resolves_relative_dataset_root_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root)
            config = load_dataset_config(config_path, project_root=root)
            self.assertEqual(config.dataset_root, (root / "data/raw/db").resolve())
            self.assertEqual(config.class_order, tuple(VALID_LABELS))
            self.assertEqual(config.model_generation, "v2")
            self.assertEqual(config.dataset_version, "1.0")
            self.assertEqual(config.doi, "10.17632/x493z8nmwc.1")

    def test_resolves_relative_exclusion_ledger_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root, exclusions_path="data/curation/exclusions.csv")

            config = load_dataset_config(config_path, project_root=root)

            self.assertEqual(
                config.exclusions_file,
                (root / "data/curation/exclusions.csv").resolve(),
            )

    def test_rejects_unrecognized_class_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root, labels=["eairh", "eh", "heh", "neh", "other"])
            with self.assertRaisesRegex(ConfigError, "class_order"):
                load_dataset_config(config_path, project_root=root)

    def test_audit_mode_rejects_missing_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root)
            with self.assertRaisesRegex(ConfigError, "Dataset directory does not exist"):
                load_dataset_config(config_path, project_root=root, require_data_root=True)


if __name__ == "__main__":
    unittest.main()

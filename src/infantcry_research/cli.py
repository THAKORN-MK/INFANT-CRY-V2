"""Command-line entry points for dataset inspection and audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import replace
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from infantcry_research.config import ConfigError, load_dataset_config
from infantcry_research.audio.features import AudioConfig, AudioContractError
from infantcry_research.data.manifest import (
    DatasetAuditError,
    ExcludedRecord,
    DatasetInspection,
    DatasetManifest,
    Record,
    build_manifest,
    inspect_dataset,
)
from infantcry_research.data.splits import (
    SplitError,
    SplitManifest,
    group_component_ids,
    make_splits,
    validate_splits,
)
from infantcry_research.evaluation.artifacts import write_locked_test_artifacts
from infantcry_research.evaluation.metrics import classification_metrics, group_bootstrap_intervals
from infantcry_research.training.run_store import (
    RunStoreError,
    claim_locked_split_access,
    create_run,
    load_run,
)
from infantcry_research.training.train import (
    TrainingError,
    configure_tensorflow_runtime,
    train_classical_candidate,
    train_tensorflow_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEURAL_CANDIDATE_IDS = {
    "logmel_cnn",
    "cnn_bilstm_attention",
    "mmt_reslstm",
    "qiu_mfcc_reslstm",
    "qiu_mmt_resnet",
    "qiu_mmt_lstm",
    "qiu_mmt_reslstm",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infantcry-research")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect-dataset", help="show audio files and metadata headers without writing files"
    )
    inspect.add_argument("--dataset-root", required=True, type=Path)
    inspect.add_argument("--metadata-filename", default="metadata.csv")

    audit = commands.add_parser("audit", help="validate the copied dataset and write a manifest")
    audit.add_argument("--dataset-root", required=True, type=Path)
    audit.add_argument(
        "--config", default=PROJECT_ROOT / "configs" / "dataset_dbl_v2.yaml", type=Path
    )
    audit.add_argument("--output-dir", default=PROJECT_ROOT / "manifests", type=Path)
    audit.add_argument("--overwrite", action="store_true")
    audit.add_argument(
        "--confirm-release",
        action="store_true",
        help="attest that the local dataset release documentation and source record were checked",
    )

    split = commands.add_parser("make-splits", help="freeze a grouped train/validation/test split")
    split.add_argument("--dataset-root", required=True, type=Path)
    split.add_argument("--audit-dir", default=PROJECT_ROOT / "manifests", type=Path)
    split.add_argument("--output-dir", default=PROJECT_ROOT / "manifests" / "splits", type=Path)
    split.add_argument("--tier", default="InfantCry-1214")
    split.add_argument("--seed", required=True, type=int)
    split.add_argument("--group-key", choices=("group_id", "sha256"), default="group_id")
    split.add_argument("--fractions", nargs=3, type=float, default=(0.70, 0.15, 0.15))
    split.add_argument("--search-trials", type=int, default=128)

    train = commands.add_parser("train", help="train one registered candidate on train/validation only")
    train.add_argument("--dataset-root", required=True, type=Path)
    train.add_argument("--audit-dir", default=PROJECT_ROOT / "manifests", type=Path)
    train.add_argument("--split-manifest", required=True, type=Path)
    train.add_argument("--config", default=PROJECT_ROOT / "configs" / "models_stage2.yaml", type=Path)
    train.add_argument(
        "--model",
        required=True,
        choices=(
            "majority",
            "mfcc_svm",
            "logmel_cnn",
            "cnn_bilstm_attention",
            "mmt_reslstm",
            "qiu_mfcc_reslstm",
            "qiu_mmt_resnet",
            "qiu_mmt_lstm",
            "qiu_mmt_reslstm",
        ),
    )
    train.add_argument("--device", choices=("cpu", "gpu"))
    train.add_argument("--require-gpu", action="store_true")
    train.add_argument("--mixed-precision", action="store_true", default=None)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--run-id")
    train.add_argument("--output-dir", default=PROJECT_ROOT / "runs", type=Path)

    select = commands.add_parser(
        "select-candidate", help="freeze model choice using completed validation-only runs"
    )
    select.add_argument("--run-dir", action="append", required=True, type=Path)
    select.add_argument("--models-config", default=PROJECT_ROOT / "configs" / "models_stage2.yaml", type=Path)
    select.add_argument("--output-dir", default=PROJECT_ROOT / "manifests" / "selections", type=Path)

    evaluate = commands.add_parser(
        "evaluate-test", help="evaluate the complete frozen candidate set in one locked-test event"
    )
    evaluate.add_argument("--dataset-root", required=True, type=Path)
    evaluate.add_argument("--audit-dir", default=PROJECT_ROOT / "manifests", type=Path)
    evaluate.add_argument("--selection-file", required=True, type=Path)
    evaluate.add_argument("--output-dir", default=PROJECT_ROOT / "runs" / "comparisons", type=Path)
    evaluate.add_argument("--protocol-id", default="stage2-locked-test-batch-v1")
    evaluate.add_argument("--bootstrap-resamples", type=int, default=2_000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=42)
    evaluate.add_argument("--device", choices=("cpu", "gpu"), default="cpu")

    report = commands.add_parser("report", help="build a report from a verified run or comparison package")
    report_target = report.add_mutually_exclusive_group(required=True)
    report_target.add_argument("--run-dir", type=Path)
    report_target.add_argument("--comparison-dir", type=Path)
    report.add_argument(
        "--output-dir", default=PROJECT_ROOT / "reports" / "stage2", type=Path
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect-dataset":
            inspection = inspect_dataset(
                args.dataset_root, metadata_filename=args.metadata_filename
            )
            print(json.dumps(_inspection_payload(inspection), indent=2, ensure_ascii=False))
            return 0
        if args.command == "audit":
            return _run_audit(args)
        if args.command == "make-splits":
            return _run_make_splits(args)
        if args.command == "train":
            return _run_train(args)
        if args.command == "select-candidate":
            return _run_select_candidate(args)
        if args.command == "evaluate-test":
            return _run_evaluate_test(args)
        if args.command == "report":
            from infantcry_research.reporting.build_report import build_comparison_report, build_report

            if args.comparison_dir is not None:
                report_path = build_comparison_report(
                    args.comparison_dir,
                    args.output_dir,
                    ledger_root=PROJECT_ROOT / "manifests" / "test_access",
                )
            else:
                report_path = build_report(args.run_dir, args.output_dir)
            print(f"Report created: {report_path}")
            return 0
        raise DatasetAuditError(f"Unsupported command: {args.command}")
    except (ConfigError, DatasetAuditError, SplitError, TrainingError, RunStoreError, AudioContractError, OSError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_audit(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    json_path = output_dir / "dataset_audit.json"
    csv_path = output_dir / "records.csv"
    excluded_csv_path = output_dir / "excluded_records.csv"
    existing = [path for path in (json_path, csv_path, excluded_csv_path) if path.exists()]
    if existing and not args.overwrite:
        raise DatasetAuditError(
            "Manifest output already exists; use --overwrite only after preserving any audit "
            "evidence you need: " + ", ".join(str(path) for path in existing)
        )

    config = load_dataset_config(Path(args.config), project_root=PROJECT_ROOT)
    dataset_root = Path(args.dataset_root).resolve()
    config = replace(config, dataset_root=dataset_root)
    manifest = build_manifest(dataset_root, config)
    audit_payload = _manifest_payload(manifest)
    if args.confirm_release:
        if manifest.version_evidence_status != "verified":
            audit_payload["version_evidence_status"] = "verified_by_user"
            audit_payload["release_verification_mode"] = "user_confirmed"
            audit_payload["release_verification_note"] = (
                "The operator attested that the copied bundle documentation and its source release record were reviewed."
            )
    record_rows = [_record_payload(record) for record in manifest.records]
    excluded_rows = [_excluded_record_payload(record) for record in manifest.excluded_records]

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_outputs(
        [
            (
                json_path,
                lambda path: path.write_text(
                    json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                ),
            ),
            (csv_path, lambda path: _write_csv(path, record_rows)),
            (
                excluded_csv_path,
                lambda path: _write_csv(
                    path,
                    excluded_rows,
                    fieldnames=(
                        "sample_id",
                        "relative_path",
                        "raw_label",
                        "tier",
                        "group_id",
                        "grouping_field",
                        "duration_s",
                        "sample_rate",
                        "channels",
                        "sha256",
                        "reason",
                    ),
                ),
            ),
        ]
    )
    print(
        f"Audit complete: {len(manifest.records)} records for analysis from "
        f"{manifest.source_record_count} source metadata records; "
        f"{len(manifest.excluded_records)} explicitly excluded; "
        f"version evidence={manifest.version_evidence_status}; "
        f"outputs: {json_path}, {csv_path} and {excluded_csv_path}"
    )
    for warning in manifest.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _run_make_splits(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    audit_path = audit_dir / "dataset_audit.json"
    records_path = audit_dir / "records.csv"
    if not audit_path.is_file() or not records_path.is_file():
        raise DatasetAuditError(
            f"A completed dataset audit is required before splitting: {audit_dir}"
        )
    try:
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetAuditError(f"Could not read dataset audit {audit_path}: {exc}") from exc
    if audit_payload.get("version_evidence_status") not in {"verified", "verified_by_user"}:
        raise DatasetAuditError(
            "release evidence is not verified; review the bundled release documentation before splitting"
        )
    if Path(str(audit_payload.get("dataset_root", ""))).resolve() != dataset_root:
        raise DatasetAuditError("Dataset audit was created for a different dataset root")
    if int(audit_payload.get("cross_label_duplicate_family_count", 0)) > 0:
        raise DatasetAuditError(
            "cross-label exact-content duplicate families require manual label/audio review "
            "before splitting; no records were silently removed"
        )

    records = _load_audited_records(records_path, dataset_root)
    if len(records) != audit_payload.get("record_count"):
        raise DatasetAuditError("Dataset audit record count does not match records.csv")
    expected_tier = str(args.tier).strip().casefold()
    tier_records = [record for record in records if record.tier.strip().casefold() == expected_tier]
    if not tier_records:
        available = sorted({record.tier for record in records})
        raise DatasetAuditError(f"No records found for Tier {args.tier!r}; available tiers: {available}")

    split_manifest = make_splits(
        tier_records,
        seed=args.seed,
        group_key=args.group_key,
        grouping_identity_verified=bool(audit_payload.get("grouping_identity_verified", False)),
        fractions=tuple(args.fractions),
        search_trials=args.search_trials,
    )
    split_audit = validate_splits(tier_records, split_manifest)
    target_dir = Path(args.output_dir).resolve() / split_manifest.split_id
    if target_dir.exists():
        raise DatasetAuditError(
            f"Frozen split output already exists and will not be overwritten: {target_dir}"
        )

    payload = _split_payload(
        split_manifest,
        tier=args.tier,
        dataset_version=str(audit_payload["dataset_version"]),
        model_generation=str(audit_payload["model_generation"]),
    )
    train_validation_ids = set(split_manifest.split_ids["train"]) | set(
        split_manifest.split_ids["validation"]
    )
    train_validation_rows = [
        _record_payload(record)
        for record in sorted(tier_records, key=lambda item: item.sample_id)
        if record.sample_id in train_validation_ids
    ]
    train_validation_records = _csv_bytes(train_validation_rows)
    payload["schema_version"] = "1.1"
    payload["source_records_csv_sha256"] = _file_sha256(records_path)
    payload["train_validation_records_sha256"] = hashlib.sha256(
        train_validation_records
    ).hexdigest()
    payload["train_validation_record_count"] = len(train_validation_rows)
    audit_summary = {
        "schema_version": "1.0",
        "valid": split_audit.valid,
        "locked_test": True,
        "split_manifest_sha256": hashlib.sha256(
            (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        ).hexdigest(),
        "split_sizes": split_audit.split_sizes,
        "group_counts": split_audit.group_counts,
        "class_counts": split_audit.class_counts,
        "source_manifest_sha256": split_audit.source_manifest_sha256,
        "source_records_csv_sha256": payload["source_records_csv_sha256"],
        "train_validation_records_sha256": payload["train_validation_records_sha256"],
        "train_validation_record_count": len(train_validation_rows),
    }
    _write_split_directory(target_dir, payload, audit_summary, train_validation_records)
    print(
        f"Frozen split {split_manifest.split_id}: "
        f"train={split_audit.split_sizes['train']}, "
        f"validation={split_audit.split_sizes['validation']}, "
        f"test={split_audit.split_sizes['test']}; saved to {target_dir}"
    )
    for warning in split_manifest.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _resolve_feature_contract(
    raw_config: dict[str, object], model_spec: dict[str, object]
) -> tuple[dict[str, object], AudioConfig, tuple[int, int, int]]:
    """Resolve one candidate's named feature profile, retaining legacy config support."""

    feature_sets = raw_config.get("feature_sets")
    if feature_sets is None:
        feature_config = raw_config.get("features")
        if not isinstance(feature_config, dict):
            raise DatasetAuditError("Model config is missing Stage 2 feature settings")
        feature_config_raw = dict(feature_config)
        declared_feature_set = model_spec.get("feature_set")
        if declared_feature_set not in (None, "stage2_hybrid"):
            raise DatasetAuditError(
                "legacy single-feature config cannot select a non-default feature_set"
            )
        feature_config_raw.setdefault("feature_set", "stage2_hybrid")
    else:
        if not isinstance(feature_sets, dict):
            raise DatasetAuditError("feature_sets must be a mapping of names to audio contracts")
        feature_set_id = str(model_spec.get("feature_set", "")).strip()
        if not feature_set_id:
            raise DatasetAuditError("Every enabled candidate must name a feature_set")
        feature_config = feature_sets.get(feature_set_id)
        if not isinstance(feature_config, dict):
            raise DatasetAuditError(
                f"candidate selects an unknown feature_set: {feature_set_id!r}"
            )
        feature_config_raw = dict(feature_config)
        configured_id = str(feature_config_raw.get("feature_set", feature_set_id))
        if configured_id != feature_set_id:
            raise DatasetAuditError(
                f"feature_set name {feature_set_id!r} does not match its contract identifier {configured_id!r}"
            )
        feature_config_raw["feature_set"] = feature_set_id

    audio_config = AudioConfig.from_mapping(feature_config_raw)
    derived_shape = audio_config.tensor_shape
    if "feature_sets" not in raw_config:
        configured_shape = tuple(int(value) for value in raw_config.get("input_shape", ()))
        if configured_shape and configured_shape != derived_shape:
            raise DatasetAuditError(
                f"Configured input shape {configured_shape} does not match feature contract {derived_shape}"
            )
    candidate_shape = model_spec.get("input_shape")
    if candidate_shape is not None and tuple(int(value) for value in candidate_shape) != derived_shape:
        # The log-Mel CNN consumes a documented channel slice of the shared
        # standard tensor, so its smaller model shape is expected.
        if str(model_spec.get("id")) != "logmel_cnn":
            raise DatasetAuditError(
                f"Candidate input_shape {candidate_shape} does not match selected feature contract {derived_shape}"
            )
    return feature_config_raw, audio_config, derived_shape


def _assert_shared_candidate_evidence(
    reference: dict[str, object], candidate: dict[str, object]
) -> None:
    """Reject cross-candidate comparisons that differ on non-feature evidence."""

    invariant_keys = (
        "split_sha256",
        "dataset_evidence",
        "seed",
        "training_settings",
    )
    if any(reference.get(key) != candidate.get(key) for key in invariant_keys):
        raise DatasetAuditError(
            "Candidate runs do not share the same split, dataset evidence, seed, and training controls"
        )


def _run_train(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    audit_path = audit_dir / "dataset_audit.json"
    split_path = Path(args.split_manifest).resolve()
    training_records_path = split_path.with_name("train_validation_records.csv")
    split_audit_path = split_path.with_name("split_audit.json")
    audit_payload = _read_json(audit_path, "dataset audit")
    split_payload = _read_json(split_path, "frozen split manifest")
    split_audit = _read_json(split_audit_path, "frozen split audit")
    if split_audit.get("split_manifest_sha256") != _file_sha256(split_path):
        raise DatasetAuditError("Frozen split manifest hash does not match its audit")
    if audit_payload.get("version_evidence_status") not in {"verified", "verified_by_user"}:
        raise DatasetAuditError("Dataset release must be verified before training")
    if int(audit_payload.get("cross_label_duplicate_family_count", 0)) > 0:
        raise DatasetAuditError(
            "cross-label exact-content duplicate families must be resolved before training"
        )
    if Path(str(audit_payload.get("dataset_root", ""))).resolve() != dataset_root:
        raise DatasetAuditError("Dataset audit was created for a different dataset root")
    if split_audit.get("valid") is not True or split_audit.get("locked_test") is not True:
        raise DatasetAuditError("Split audit is not valid or does not mark the test split as locked")
    if split_payload.get("locked_test") is not True or split_payload.get("test_selection_allowed") is not False:
        raise DatasetAuditError("Split manifest must declare a locked test that cannot select models")
    if split_payload.get("source_manifest_sha256") != split_audit.get("source_manifest_sha256"):
        raise DatasetAuditError("Frozen split audit and manifest source hashes do not match")
    if split_payload.get("dataset_version") != audit_payload.get("dataset_version"):
        raise DatasetAuditError("Frozen split dataset version differs from the verified dataset audit")
    if split_payload.get("model_generation") != audit_payload.get("model_generation"):
        raise DatasetAuditError("Frozen split model generation differs from the verified dataset audit")

    split_ids = split_payload.get("split_ids")
    if not isinstance(split_ids, dict) or set(split_ids) != {"train", "validation", "test"}:
        raise DatasetAuditError("Split manifest must contain train, validation, and test ID lists")
    train_ids = tuple(str(value) for value in split_ids["train"])
    validation_ids = tuple(str(value) for value in split_ids["validation"])
    test_ids = tuple(str(value) for value in split_ids["test"])
    all_split_ids = train_ids + validation_ids + test_ids
    if not train_ids or not validation_ids or not test_ids or len(all_split_ids) != len(set(all_split_ids)):
        raise DatasetAuditError("Split manifest has empty partitions or overlapping sample IDs")
    recorded_sizes = split_audit.get("split_sizes", {})
    if any(recorded_sizes.get(name) != len(split_ids[name]) for name in ("train", "validation", "test")):
        raise DatasetAuditError("Frozen split audit sizes do not match the manifest ID lists")

    try:
        import yaml

        raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetAuditError(f"Could not read model configuration {args.config}: {exc}") from exc
    if not isinstance(raw_config, dict):
        raise DatasetAuditError("Model configuration must be a YAML mapping")
    class_order = tuple(str(label).strip().lower() for label in raw_config.get("class_order", ()))
    if class_order != ("eairh", "eh", "heh", "neh", "owh"):
        raise DatasetAuditError("Model config must preserve the five original DBL class labels in canonical order")
    model_specs = raw_config.get("models", [])
    enabled_candidate_ids = tuple(
        str(item["id"])
        for item in model_specs
        if isinstance(item, dict) and item.get("enabled") is True and item.get("id")
    )
    if not enabled_candidate_ids or len(set(enabled_candidate_ids)) != len(enabled_candidate_ids):
        raise DatasetAuditError("Enabled model candidate IDs must be non-empty and unique")
    model_spec = next(
        (item for item in model_specs if isinstance(item, dict) and item.get("id") == args.model),
        None,
    )
    if model_spec is None or not model_spec.get("enabled", False):
        raise DatasetAuditError(f"Model candidate {args.model!r} is missing or disabled in config")
    expected_tier = str(raw_config.get("primary_tier", "InfantCry-1214"))
    tier = str(split_payload.get("tier", ""))
    if tier.casefold() != expected_tier.casefold():
        raise DatasetAuditError(
            f"Split tier {tier!r} does not match configured primary tier {expected_tier!r}"
        )

    neural_model = args.model in NEURAL_CANDIDATE_IDS
    expected_device = str(model_spec.get("device", "gpu" if neural_model else "cpu")).casefold()
    device = args.device or expected_device
    if device.casefold() != expected_device:
        raise TrainingError(
            f"device differs from its frozen candidate config for {args.model}: "
            f"expected {expected_device!r}, got {device!r}"
        )
    expected_mixed_precision = bool(model_spec.get("mixed_precision", False))
    if args.mixed_precision is True and not expected_mixed_precision:
        raise TrainingError(
            f"mixed precision differs from its frozen candidate config for {args.model}: "
            "the candidate is configured with mixed_precision=false"
        )
    mixed_precision = expected_mixed_precision
    if not neural_model and device != "cpu":
        raise TrainingError("Majority and MFCC-SVM are CPU baselines; use --device cpu")
    if not neural_model and args.require_gpu:
        raise TrainingError("--require-gpu applies only to the neural-network candidates")
    if not neural_model and mixed_precision:
        raise TrainingError("mixed precision applies only to TensorFlow neural-network candidates")
    if neural_model:
        import tensorflow as tf

        runtime = configure_tensorflow_runtime(
            tf,
            device=device,
            require_gpu=bool(args.require_gpu),
            mixed_precision=mixed_precision,
        )
    else:
        runtime = {
            "requested_device": "cpu",
            "selected_device": "cpu",
            "gpu_count": 0,
            "gpu_names": [],
            "mixed_precision": mixed_precision,
        }

    feature_config_raw, audio_config, expected_shape = _resolve_feature_contract(
        raw_config, model_spec
    )
    if feature_config_raw.get("sample_rate_status") != "locked":
        raise DatasetAuditError(
            "Sample rate is still provisional; inspect the copied dataset and explicitly lock the "
            "chosen rate in configs/models_stage2.yaml before training"
        )
    train_ids_set = set(train_ids)
    validation_ids_set = set(validation_ids)
    requested_ids = train_ids_set | validation_ids_set
    expected_partition_hash = str(split_payload.get("train_validation_records_sha256", ""))
    source_records_hash = str(split_payload.get("source_records_csv_sha256", ""))
    if not expected_partition_hash or not source_records_hash:
        raise DatasetAuditError(
            "Frozen split is missing source-manifest and train/validation partition hashes; "
            "recreate the split with the current CLI"
        )
    if split_audit.get("source_records_csv_sha256") != source_records_hash:
        raise DatasetAuditError("Split audit and source records manifest hashes do not match")
    if split_payload.get("train_validation_record_count") != len(requested_ids):
        raise DatasetAuditError("Train/validation manifest count does not match the frozen split")
    if split_audit.get("train_validation_record_count") != len(requested_ids):
        raise DatasetAuditError("Split audit train/validation record count does not match the frozen split")
    if _file_sha256(training_records_path) != expected_partition_hash:
        raise DatasetAuditError("Frozen train/validation records manifest hash does not match the split")
    if split_audit.get("train_validation_records_sha256") != expected_partition_hash:
        raise DatasetAuditError("Split audit and train/validation records hashes do not match")
    records = _load_audited_records(
        training_records_path, dataset_root, sample_ids=requested_ids
    )
    records_by_id = {record.sample_id: record for record in records}
    if set(records_by_id) != requested_ids:
        missing = sorted(requested_ids - set(records_by_id))
        raise DatasetAuditError(
            f"Training/validation IDs are missing from the frozen partition manifest: {missing[:10]}"
        )
    if any(record.tier.casefold() != tier.casefold() for record in records):
        raise DatasetAuditError("The frozen train/validation IDs include a record from another Tier")
    unknown_labels = sorted({record.raw_label.strip().casefold() for record in records} - set(class_order))
    if unknown_labels:
        raise DatasetAuditError(f"Train/validation contains unconfigured labels: {unknown_labels}")

    seed = int(args.seed if args.seed is not None else split_payload.get("seed", 42))
    training_settings = dict(raw_config.get("training", {}))
    optimizer = str(training_settings.get("optimizer", "Adam")).strip().casefold()
    if optimizer != "adam":
        raise DatasetAuditError(
            f"Only the configured Adam optimizer is implemented; got {optimizer!r}"
        )
    epochs = int(args.epochs if args.epochs is not None else training_settings.get("epochs", 50))
    batch_size = int(
        args.batch_size if args.batch_size is not None else training_settings.get("batch_size", 32)
    )
    learning_rate = float(training_settings.get("learning_rate", 0.001))
    patience = int(training_settings.get("early_stopping_patience", 8))
    runtime.update(_package_environment(neural_model=neural_model))
    run_config = {
        "candidate_id": args.model,
        "candidate_protocol_sha256": _file_sha256(Path(args.config).resolve()),
        "registered_candidate_ids": list(enabled_candidate_ids),
        "candidate_protocol_snapshot": {
            "configuration_sha256": _file_sha256(Path(args.config).resolve()),
            "candidates": [
                item for item in model_specs if isinstance(item, dict) and item.get("enabled") is True
            ],
            "shared_training": training_settings,
            "features": feature_config_raw,
            "class_order": list(class_order),
        },
        "class_order": list(class_order),
        "input_shape": list(expected_shape),
        "candidate_config": model_spec,
        "feature_config": feature_config_raw,
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping_patience": patience,
            "device": device,
            "mixed_precision": mixed_precision,
            "selection_metric": training_settings.get("selection_metric", "validation_macro_f1"),
            "optimizer": "Adam",
            "deterministic_ops": bool(training_settings.get("deterministic_ops", True)),
        },
        "tier": tier,
        "dataset_evidence": {
            "dataset_root": str(dataset_root),
            "dataset_version": split_payload["dataset_version"],
            "model_generation": split_payload["model_generation"],
            "doi": audit_payload.get("doi"),
            "audit_sha256": _file_sha256(audit_path),
            "records_csv_sha256": source_records_hash,
        },
    }
    run = create_run(
        args.output_dir,
        config=run_config,
        split_manifest=split_payload,
        environment=runtime,
        candidate_ids=(args.model,),
        run_id=args.run_id,
    )
    try:
        train_labels = tuple(records_by_id[sample_id].raw_label.strip().casefold() for sample_id in train_ids)
        validation_labels = tuple(
            records_by_id[sample_id].raw_label.strip().casefold() for sample_id in validation_ids
        )
        if args.model == "majority":
            from infantcry_research.models.baselines import MajorityClassifier

            model = MajorityClassifier(class_order)
            result = train_classical_candidate(
                model,
                None,
                np.asarray(train_labels, dtype=object),
                None,
                np.asarray(validation_labels, dtype=object),
                run=run,
                candidate_id=args.model,
                class_order=class_order,
                validation_ids=validation_ids,
            )
        else:
            from infantcry_research.training.data import load_feature_partition

            x_train, _loaded_train_labels, _ = load_feature_partition(
                records_by_id, train_ids, audio_config
            )
            x_validation, _loaded_validation_labels, _ = load_feature_partition(
                records_by_id, validation_ids, audio_config
            )
            y_train = np.asarray([class_order.index(label) for label in train_labels], dtype=np.int32)
            y_validation = np.asarray(
                [class_order.index(label) for label in validation_labels], dtype=np.int32
            )
            if args.model == "mfcc_svm":
                from infantcry_research.models.baselines import build_mfcc_svm

                model = build_mfcc_svm(
                    n_mfcc=audio_config.n_mfcc,
                    random_state=seed,
                    kernel=str(model_spec.get("kernel", "rbf")),
                    class_weight=model_spec.get("class_weight", "balanced"),
                )
                result = train_classical_candidate(
                    model,
                    x_train,
                    np.asarray(train_labels, dtype=object),
                    x_validation,
                    np.asarray(validation_labels, dtype=object),
                    run=run,
                    candidate_id=args.model,
                    class_order=class_order,
                    validation_ids=validation_ids,
                )
            else:
                import tensorflow as tf
                from infantcry_research.training.train import seed_tensorflow_experiment

                seed_tensorflow_experiment(
                    tf,
                    seed,
                    deterministic_ops=bool(training_settings.get("deterministic_ops", True)),
                )

                if args.model == "logmel_cnn":
                    from infantcry_research.models.baselines import build_logmel_cnn

                    start = 3 * audio_config.n_mfcc
                    end = start + audio_config.n_mels
                    model_features_train = x_train[:, start:end, :, :]
                    model_features_validation = x_validation[:, start:end, :, :]
                    model = build_logmel_cnn(
                        tf,
                        input_shape=tuple(int(value) for value in model_spec.get("input_shape", (64, 128, 1))),
                        num_classes=len(class_order),
                        config=model_spec,
                    )
                elif args.model == "cnn_bilstm_attention":
                    from infantcry_research.models.cnn_bilstm_attention import build_cnn_bilstm_attention

                    model_features_train = x_train
                    model_features_validation = x_validation
                    model = build_cnn_bilstm_attention(
                        tf,
                        input_shape=expected_shape,
                        num_classes=len(class_order),
                        config=model_spec,
                    )
                elif args.model == "mmt_reslstm":
                    from infantcry_research.models.reslstm import build_mmt_reslstm

                    model_features_train = x_train
                    model_features_validation = x_validation
                    model = build_mmt_reslstm(
                        tf,
                        input_shape=expected_shape,
                        num_classes=len(class_order),
                        config=model_spec,
                    )
                else:
                    architecture = str(model_spec.get("architecture", ""))
                    if architecture != args.model:
                        raise TrainingError(
                            f"candidate {args.model!r} must declare its matching explicit "
                            "architecture; "
                            f"got {architecture!r}"
                        )
                    from infantcry_research.models.qiu_comparators import (
                        build_qiu_mfcc_reslstm,
                        build_qiu_mmt_lstm,
                        build_qiu_mmt_reslstm,
                        build_qiu_mmt_resnet,
                    )

                    architecture_builders = {
                        "qiu_mfcc_reslstm": build_qiu_mfcc_reslstm,
                        "qiu_mmt_resnet": build_qiu_mmt_resnet,
                        "qiu_mmt_lstm": build_qiu_mmt_lstm,
                        "qiu_mmt_reslstm": build_qiu_mmt_reslstm,
                    }
                    builder = architecture_builders.get(architecture)
                    if builder is None:
                        raise TrainingError(
                            f"unsupported explicit model architecture {architecture!r}"
                        )
                    model_features_train = x_train
                    model_features_validation = x_validation
                    model = builder(
                        tf,
                        input_shape=expected_shape,
                        num_classes=len(class_order),
                        config=model_spec,
                    )
                result = train_tensorflow_candidate(
                    tf,
                    model,
                    model_features_train,
                    y_train,
                    model_features_validation,
                    y_validation,
                    run=run,
                    candidate_id=args.model,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    patience=patience,
                    device=device,
                    class_order=class_order,
                    validation_ids=validation_ids,
                )
        print(json.dumps({"run_id": run.run_id, "run_dir": str(run.run_dir), **result}, indent=2))
        return 0
    except Exception as exc:
        if run.status in {"prepared", "running"}:
            try:
                run.mark_training_failed(reason=f"{type(exc).__name__}: {exc}")
            except RunStoreError:
                pass
        raise


def _load_audited_records(
    records_path: Path,
    dataset_root: Path,
    *,
    sample_ids: set[str] | None = None,
) -> list[Record]:
    required = {
        "sample_id",
        "relative_path",
        "raw_label",
        "tier",
        "duration_s",
        "sample_rate",
        "channels",
        "sha256",
    }
    records: list[Record] = []
    found_ids: set[str] = set()
    try:
        with records_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise DatasetAuditError(
                    "Audited records.csv is missing columns: " + ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, start=2):
                sample_id = row.get("sample_id") or ""
                if sample_ids is not None and sample_id not in sample_ids:
                    continue
                if sample_id in found_ids:
                    raise DatasetAuditError(
                        f"Duplicate audited sample_id at CSV row {row_number}: {sample_id!r}"
                    )
                found_ids.add(sample_id)
                relative_path = (row.get("relative_path") or "").replace("\\", "/")
                audio_path = (dataset_root / relative_path).resolve()
                try:
                    audio_path.relative_to(dataset_root)
                except ValueError as exc:
                    raise DatasetAuditError(
                        f"Audited row {row_number} path escapes dataset root: {relative_path!r}"
                    ) from exc
                if not audio_path.is_file():
                    raise DatasetAuditError(f"Audited audio file no longer exists: {relative_path}")
                if _file_sha256(audio_path) != row.get("sha256"):
                    raise DatasetAuditError(f"SHA-256 changed since dataset audit: {relative_path}")
                try:
                    records.append(
                        Record(
                            sample_id=sample_id,
                            path=audio_path,
                            relative_path=relative_path,
                            raw_label=row["raw_label"],
                            tier=row["tier"],
                            group_id=row.get("group_id") or None,
                            duration_s=float(row["duration_s"]),
                            sha256=row["sha256"],
                            sample_rate=int(row["sample_rate"]),
                            channels=int(row["channels"]),
                            grouping_field=row.get("grouping_field") or None,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise DatasetAuditError(
                        f"Invalid audited record at CSV row {row_number}: {exc}"
                    ) from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetAuditError(f"Could not read records CSV {records_path}: {exc}") from exc
    if sample_ids is not None:
        missing_ids = sorted(sample_ids - found_ids)
        if missing_ids:
            raise DatasetAuditError(
                f"Requested audited sample IDs are missing from records.csv: {missing_ids[:10]}"
            )
    return records


def _run_evaluate_test_single_legacy(args: argparse.Namespace) -> int:
    del args
    raise RunStoreError(
        "Single-candidate test evaluation is disabled; use the frozen batch evaluate-test command."
    )

    # Kept below only as historical implementation context; the public CLI has
    # no single-run route and cannot execute this code path.
    run = load_run(Path(args.run_dir))
    if run.status != "complete":
        raise RunStoreError("Only a completed train/validation run may access the locked test")
    run_config = _read_json(run.run_dir / "config.json", "frozen training config")
    split_payload = _read_json(run.run_dir / "split_manifest.json", "frozen run split")
    candidate_id = str(run_config.get("candidate_id", ""))
    if not candidate_id or run.metadata.get("selected_candidate_id") != candidate_id:
        raise RunStoreError("Run candidate is not the model selected using validation")
    selection_path = Path(args.selection_file).resolve()
    selection_evidence = run.metadata.get("selection_evidence")
    if not isinstance(selection_evidence, dict):
        raise RunStoreError("Freeze a cross-candidate validation selection before opening the test")
    if selection_evidence.get("selection_path") != str(selection_path):
        raise RunStoreError("Selection file does not match the evidence frozen in the selected run")
    if _file_sha256(selection_path) != selection_evidence.get("selection_sha256"):
        raise RunStoreError("Validation selection file changed after it was frozen")
    selection_payload = _read_json(selection_path, "validation selection")
    if (
        selection_payload.get("selected_run_id") != run.run_id
        or selection_payload.get("selected_candidate_id") != candidate_id
        or selection_payload.get("split_id") != split_payload.get("split_id")
    ):
        raise RunStoreError("Validation selection does not select this run/candidate/split")
    if not args.protocol_id.strip():
        raise DatasetAuditError("protocol-id must not be empty")
    if args.bootstrap_resamples < 100:
        raise DatasetAuditError("bootstrap-resamples must be at least 100")

    dataset_root = Path(args.dataset_root).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    audit_path = audit_dir / "dataset_audit.json"
    records_path = audit_dir / "records.csv"
    data_evidence = run_config.get("dataset_evidence")
    if not isinstance(data_evidence, dict):
        raise DatasetAuditError("Run is missing its frozen dataset audit references")
    if Path(str(data_evidence.get("dataset_root", ""))).resolve() != dataset_root:
        raise DatasetAuditError("Evaluation dataset root differs from the training dataset root")
    if _file_sha256(audit_path) != data_evidence.get("audit_sha256"):
        raise DatasetAuditError("Dataset audit changed since this model was trained")
    if _file_sha256(records_path) != data_evidence.get("records_csv_sha256"):
        raise DatasetAuditError("Dataset records manifest changed since this model was trained")
    audit_payload = _read_json(audit_path, "dataset audit")
    if audit_payload.get("version_evidence_status") not in {"verified", "verified_by_user"}:
        raise DatasetAuditError("Dataset release evidence is not verified")
    if Path(str(audit_payload.get("dataset_root", ""))).resolve() != dataset_root:
        raise DatasetAuditError("Dataset audit was created for a different dataset root")
    if split_payload.get("locked_test") is not True or split_payload.get("test_selection_allowed") is not False:
        raise DatasetAuditError("Run split does not preserve the locked-test policy")
    split_ids = split_payload.get("split_ids")
    if not isinstance(split_ids, dict) or not isinstance(split_ids.get("test"), list):
        raise DatasetAuditError("Run split is missing its locked-test sample IDs")
    test_ids = tuple(str(value) for value in split_ids["test"])
    if not test_ids or len(test_ids) != len(set(test_ids)):
        raise DatasetAuditError("Run locked-test IDs are empty or duplicated")

    # Verify held-out file hashes after selection but before consuming the
    # one-shot evaluation token.
    test_records = _load_audited_records(records_path, dataset_root, sample_ids=set(test_ids))
    by_id = {record.sample_id: record for record in test_records}

    class_order_raw = run_config.get("class_order")
    if not isinstance(class_order_raw, list):
        raise DatasetAuditError("Frozen run config is missing class_order")
    class_order = tuple(str(value) for value in class_order_raw)
    feature_config_raw = run_config.get("feature_config")
    if not isinstance(feature_config_raw, dict):
        raise DatasetAuditError("Frozen run config is missing the audio feature contract")
    audio_config = AudioConfig.from_mapping(feature_config_raw)
    tier = str(run_config.get("tier", ""))
    if any(record.tier.casefold() != tier.casefold() for record in test_records):
        raise DatasetAuditError("Locked-test sample IDs include a record outside the trained Tier")
    y_true = tuple(by_id[sample_id].raw_label.strip().casefold() for sample_id in test_ids)
    if set(y_true) - set(class_order):
        raise DatasetAuditError("Locked test contains labels outside the frozen class order")
    x_test: np.ndarray | None = None
    if candidate_id != "majority":
        from infantcry_research.training.data import load_feature_partition

        x_test, loaded_labels, loaded_ids = load_feature_partition(by_id, test_ids, audio_config)
        if loaded_ids != test_ids or tuple(label.strip().casefold() for label in loaded_labels) != y_true:
            raise DatasetAuditError("Decoded test records do not match the frozen test IDs/labels")
    protocol_descriptor = {
        "protocol_id": args.protocol_id,
        "split_id": split_payload.get("split_id"),
        "candidate_id": candidate_id,
        "class_order": class_order,
        "bootstrap_method": (
            "percentile_group_bootstrap"
            if str(split_payload.get("grouping_scope", "")).startswith("verified_independent_group:")
            else "not_reported_unverified_grouping"
        ),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact_hashes = run.metadata.get("training_artifact_hashes", {})
    candidate_model_name = "best_model.keras" if candidate_id in {"logmel_cnn", "cnn_bilstm_attention"} else "model.joblib"
    model_path = run.run_dir / "artifacts" / candidate_model_name
    if artifact_hashes.get(f"artifacts/{candidate_model_name}") is None:
        raise RunStoreError(f"Run does not have a hash-frozen model artifact: {candidate_model_name}")

    if candidate_id == "majority":
        import joblib

        model = joblib.load(model_path)
        x_test = np.empty((len(test_ids), 0), dtype=np.float32)
    elif candidate_id == "mfcc_svm":
        import joblib

        assert x_test is not None
        model = joblib.load(model_path)
    elif candidate_id in {"logmel_cnn", "cnn_bilstm_attention"}:
        import tensorflow as tf

        from infantcry_research.models.layers import TemporalAttention

        assert x_test is not None
        if args.device == "gpu":
            runtime = configure_tensorflow_runtime(
                tf, device="gpu", require_gpu=True, mixed_precision=False
            )
        else:
            runtime = configure_tensorflow_runtime(
                tf, device="cpu", require_gpu=False, mixed_precision=False
            )
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"TemporalAttention": TemporalAttention},
            compile=False,
        )
        if candidate_id == "logmel_cnn":
            start = 3 * audio_config.n_mfcc
            end = start + audio_config.n_mels
            x_test = x_test[:, start:end, :, :]
    else:
        raise DatasetAuditError(f"Unsupported frozen candidate: {candidate_id}")

    # Hashes, features, and the selected model artifact are preflighted. Consume
    # the one-shot token only when all setup is ready and just before prediction.
    claim_locked_split_access(
        PROJECT_ROOT / "manifests" / "test_access",
        split_id=str(split_payload.get("split_id", "")),
        candidate_id=candidate_id,
        run_id=run.run_id,
        protocol_hash=protocol_hash,
    )
    run.begin_test_evaluation(candidate_id, protocol_hash)

    if candidate_id in {"majority", "mfcc_svm"}:
        raw_scores = model.predict_proba(x_test)
        probabilities = _align_estimator_probabilities(model, raw_scores, class_order)
    else:
        with tf.device("/GPU:0" if args.device == "gpu" else "/CPU:0"):
            probabilities = np.asarray(model.predict(x_test, verbose=0), dtype=np.float64)

    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_pred = tuple(class_order[index] for index in np.argmax(probabilities, axis=1))
    metric_results = classification_metrics(y_true, y_pred, probabilities, class_order)
    test_component_ids = group_component_ids(
        test_records, str(split_payload.get("group_key", ""))
    )
    groups = tuple(test_component_ids[sample_id] for sample_id in test_ids)
    if str(split_payload.get("grouping_scope", "")).startswith("verified_independent_group:"):
        try:
            uncertainty = group_bootstrap_intervals(
                y_true,
                y_pred,
                groups,
                class_order=class_order,
                n_resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
                confidence_level=0.95,
            )
            uncertainty["status"] = "available"
        except ValueError as exc:
            uncertainty = {
                "status": "unavailable",
                "method": None,
                "reason": str(exc),
            }
    else:
        uncertainty = {
            "status": "unavailable",
            "method": None,
            "reason": (
                "A verified independent subject/session grouping field was not verified; "
                "recordings are not treated as independent groups."
            ),
        }
    metrics_payload: dict[str, object] = {
        "schema_version": "1.1",
        "evaluation_scope": "locked_internal_test",
        "candidate_id": candidate_id,
        "run_id": run.run_id,
        "protocol_hash": protocol_hash,
        "selection_sha256": selection_evidence.get("selection_sha256"),
        "split_id": split_payload.get("split_id"),
        "tier": tier,
        "grouping_scope": split_payload.get("grouping_scope"),
        "support": len(test_ids),
        "class_order": list(class_order),
        "metrics": metric_results,
        "uncertainty": uncertainty,
        "evaluation_device": args.device if candidate_id in {"logmel_cnn", "cnn_bilstm_attention"} else "cpu",
        "evaluation_runtime": runtime if candidate_id in {"logmel_cnn", "cnn_bilstm_attention"} else {"selected_device": "cpu"},
        "external_validation_performed": False,
    }
    evaluation_dir, evaluation_hash = write_locked_test_artifacts(
        run.run_dir,
        candidate_id=candidate_id,
        protocol_hash=protocol_hash,
        class_order=class_order,
        sample_ids=test_ids,
        y_true=y_true,
        y_pred=y_pred,
        y_score=probabilities,
        metrics_payload=metrics_payload,
    )
    receipt_path = run.record_test_evaluation(candidate_id, protocol_hash, evaluation_hash)
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "candidate_id": candidate_id,
                "evaluation_dir": str(evaluation_dir),
                "receipt": str(receipt_path),
                "evaluation_hash": evaluation_hash,
                "metrics": metric_results,
                "uncertainty": uncertainty,
            },
            indent=2,
        )
    )
    return 0


def _run_evaluate_test(args: argparse.Namespace) -> int:
    from infantcry_research.evaluation.comparison import evaluate_candidate_set

    comparison_dir = evaluate_candidate_set(
        Path(args.selection_file),
        dataset_root=Path(args.dataset_root),
        audit_dir=Path(args.audit_dir),
        ledger_root=PROJECT_ROOT / "manifests" / "test_access",
        output_root=Path(args.output_dir),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        protocol_id=str(args.protocol_id),
        device=str(args.device),
    )
    summary = _read_json(comparison_dir / "comparison.json", "locked-test comparison summary")
    print(
        json.dumps(
            {
                "comparison_dir": str(comparison_dir),
                "split_id": summary.get("split_id"),
                "selection_id": summary.get("selection_id"),
                "selected_by_validation": summary.get("selected_by_validation"),
                "test_support": summary.get("test_support"),
                "candidate_count": len(summary.get("candidates", [])),
                "test_scores_used_for_selection": False,
            },
            indent=2,
        )
    )
    return 0


def _run_select_candidate(args: argparse.Namespace) -> int:
    try:
        import yaml

        model_protocol = yaml.safe_load(Path(args.models_config).read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetAuditError(f"Could not read candidate-selection config: {exc}") from exc
    if not isinstance(model_protocol, dict):
        raise DatasetAuditError("Candidate-selection config must be a YAML mapping")
    expected_candidates = [
        str(item["id"])
        for item in model_protocol.get("models", [])
        if isinstance(item, dict) and item.get("enabled") is True
    ]
    if not expected_candidates:
        raise DatasetAuditError("No enabled model candidates are registered for selection")
    current_protocol_sha256 = _file_sha256(Path(args.models_config).resolve())
    expected_model_specs = {
        str(item["id"]): item
        for item in model_protocol.get("models", [])
        if isinstance(item, dict) and item.get("enabled") is True
    }

    candidates: list[dict[str, object]] = []
    common_split: dict[str, object] | None = None
    common_evidence: dict[str, object] | None = None
    observed_candidates: set[str] = set()
    for run_path in args.run_dir:
        run = load_run(run_path)
        if run.status != "complete":
            raise RunStoreError(f"Candidate run is not complete: {run.run_id}")
        run_config = _read_json(run.run_dir / "config.json", "candidate run config")
        candidate_id = str(run_config.get("candidate_id", ""))
        if run_config.get("candidate_protocol_sha256") != current_protocol_sha256:
            raise DatasetAuditError(
                "candidate protocol configuration changed after the run; selection requires "
                "the exact frozen models_stage2.yaml used before training"
            )
        if run_config.get("registered_candidate_ids") != expected_candidates:
            raise DatasetAuditError(
                "candidate set differs from the one frozen before training"
            )
        if not candidate_id or run.metadata.get("selected_candidate_id") != candidate_id:
            raise RunStoreError(f"Candidate identity is inconsistent in run {run.run_id}")
        if candidate_id in observed_candidates:
            raise DatasetAuditError(f"More than one run was supplied for candidate {candidate_id!r}")
        observed_candidates.add(candidate_id)
        if candidate_id not in expected_candidates:
            raise DatasetAuditError(f"Run contains unregistered/disabled candidate {candidate_id!r}")
        candidate_spec = expected_model_specs[candidate_id]
        if run.metadata.get("selection_evidence"):
            raise RunStoreError(f"Candidate run already has a frozen selection: {run.run_id}")
        split_hash = run.hashes["split_sha256"]
        split_payload = _read_json(run.run_dir / "split_manifest.json", "candidate frozen split")
        feature_key = json.dumps(
            run_config.get("feature_config"), sort_keys=True, separators=(",", ":")
        )
        feature_hash = hashlib.sha256(feature_key.encode("utf-8")).hexdigest()
        evidence_key = json.dumps(run_config.get("dataset_evidence"), sort_keys=True, separators=(",", ":"))
        training = run_config.get("training", {})
        if not isinstance(training, dict) or training.get("selection_metric") != "validation_macro_f1":
            raise DatasetAuditError("All candidates must use validation_macro_f1 as the predeclared selection metric")
        neural_model = candidate_id in NEURAL_CANDIDATE_IDS
        expected_device = str(
            candidate_spec.get("device", "gpu" if neural_model else "cpu")
        ).casefold()
        expected_mixed_precision = bool(candidate_spec.get("mixed_precision", False))
        if str(training.get("device", "")).casefold() != expected_device:
            raise DatasetAuditError(
                f"Run {run.run_id} device does not match its frozen candidate config: "
                f"expected {expected_device!r}"
            )
        if bool(training.get("mixed_precision", False)) != expected_mixed_precision:
            raise DatasetAuditError(
                f"Run {run.run_id} mixed-precision setting does not match its frozen candidate config"
            )
        training_settings_for_comparison = {
            key: training.get(key)
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "early_stopping_patience",
                "optimizer",
                "selection_metric",
                "deterministic_ops",
            )
        }
        run_seed = int(training.get("seed", -1))
        candidate_evidence: dict[str, object] = {
            "split_sha256": split_hash,
            "dataset_evidence": evidence_key,
            "seed": run_seed,
            "training_settings": training_settings_for_comparison,
        }
        if common_evidence is None:
            common_evidence = candidate_evidence
            common_split = split_payload
        else:
            _assert_shared_candidate_evidence(common_evidence, candidate_evidence)
        validation = _read_json(run.run_dir / "artifacts" / "validation_metrics.json", "validation metrics")
        if validation.get("evaluation_scope") != "validation_only":
            raise DatasetAuditError(f"Run {run.run_id} does not contain validation-only metrics")
        score = float(validation.get("validation_macro_f1", -1.0))
        if not 0.0 <= score <= 1.0:
            raise DatasetAuditError(f"Run {run.run_id} has an invalid validation macro-F1")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "validation_macro_f1": score,
                "validation_support": int(validation.get("validation_support", 0)),
                "config_sha256": run.hashes["config_sha256"],
                "feature_config_sha256": feature_hash,
            }
        )
    if observed_candidates != set(expected_candidates):
        missing = sorted(set(expected_candidates) - observed_candidates)
        raise DatasetAuditError(
            "Validation selection requires exactly one completed run for each enabled candidate; "
            f"missing: {missing}"
        )
    assert common_split is not None and common_evidence is not None
    candidate_order = {candidate_id: index for index, candidate_id in enumerate(expected_candidates)}
    candidates.sort(
        key=lambda item: (
            -float(item["validation_macro_f1"]),
            candidate_order[str(item["candidate_id"])],
        )
    )
    winner = candidates[0]
    split_id = str(common_split.get("split_id", ""))
    selection_id = f"selection-{split_id}"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / f"{split_id}.json"
    if selection_path.exists():
        raise DatasetAuditError(f"Selection already frozen and will not be overwritten: {selection_path}")
    payload = {
        "schema_version": "1.0",
        "selection_id": selection_id,
        "split_id": split_id,
        "split_sha256": common_evidence["split_sha256"],
        "candidate_protocol_sha256": current_protocol_sha256,
        "source_manifest_sha256": common_split.get("source_manifest_sha256"),
        "selection_metric": "validation_macro_f1",
        "candidate_set": expected_candidates,
        "ranking": candidates,
        "selected_candidate_id": winner["candidate_id"],
        "selected_run_id": winner["run_id"],
        "test_accessed": False,
        "selection_rule": "highest validation macro-F1; ties resolved by configured candidate order",
    }
    selection_bytes = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with selection_path.open("xb") as stream:
            stream.write(selection_bytes)
    except FileExistsError as exc:
        raise DatasetAuditError(f"Selection already frozen: {selection_path}") from exc
    selected_run = load_run(Path(str(winner["run_dir"])))
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    selected_run.record_validation_selection(
        selection_id=selection_id,
        selection_path=selection_path,
        selection_sha256=selection_sha256,
    )
    print(
        json.dumps(
            {
                "selection_id": selection_id,
                "selection_file": str(selection_path),
                "selected_candidate_id": winner["candidate_id"],
                "selected_run_id": winner["run_id"],
                "validation_macro_f1": winner["validation_macro_f1"],
                "test_accessed": False,
                "candidate_count": len(candidates),
            },
            indent=2,
        )
    )
    return 0


def _align_estimator_probabilities(
    estimator: object, scores: np.ndarray, class_order: tuple[str, ...]
) -> np.ndarray:
    source_classes = tuple(str(value) for value in getattr(estimator, "classes_", ()))
    probabilities = np.asarray(scores, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(source_classes):
        raise TrainingError("Estimator probability columns do not match estimator.classes_")
    output = np.zeros((probabilities.shape[0], len(class_order)), dtype=np.float64)
    class_index = {label: index for index, label in enumerate(class_order)}
    for source_index, label in enumerate(source_classes):
        if label not in class_index:
            raise TrainingError(f"Estimator emitted an unconfigured label: {label}")
        output[:, class_index[label]] = probabilities[:, source_index]
    row_sums = output.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise TrainingError("Estimator emitted zero total probability for a test record")
    return output / row_sums


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetAuditError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetAuditError(f"{description} must contain a JSON object: {path}")
    return payload


def _package_environment(*, neural_model: bool) -> dict[str, object]:
    packages = ("numpy", "librosa", "scikit-learn", "soundfile", "PyYAML", "tensorflow")
    versions: dict[str, str | None] = {}
    for name in packages:
        if name == "tensorflow" and not neural_model:
            continue
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_payload(
    split: SplitManifest,
    *,
    tier: str,
    dataset_version: str,
    model_generation: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "split_id": split.split_id,
        "dataset_version": dataset_version,
        "model_generation": model_generation,
        "tier": tier,
        "seed": split.seed,
        "fractions": split.fractions,
        "group_key": split.group_key,
        "grouping_scope": split.grouping_scope,
        "source_manifest_sha256": split.source_manifest_sha256,
        "locked_test": True,
        "test_selection_allowed": False,
        "split_ids": {name: list(ids) for name, ids in split.split_ids.items()},
        "group_ids_by_split": {
            name: list(ids) for name, ids in split.group_ids_by_split.items()
        },
        "class_counts": split.class_counts,
        "warnings": list(split.warnings),
    }


def _write_split_directory(
    target_dir: Path,
    split_payload: dict[str, object],
    audit_payload: dict[str, object],
    train_validation_records: bytes,
) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{target_dir.name}.", dir=target_dir.parent)
    )
    try:
        (temporary_dir / "split_manifest.json").write_bytes(
            (json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        )
        (temporary_dir / "split_audit.json").write_bytes(
            (json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        )
        (temporary_dir / "train_validation_records.csv").write_bytes(train_validation_records)
        os.replace(temporary_dir, target_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def _atomic_write_outputs(
    outputs: list[tuple[Path, Callable[[Path], None]]],
) -> None:
    temporary: list[Path] = []
    try:
        for destination, writer in outputs:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.stem}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
            temporary.append(temp_path)
            writer(temp_path)
        for source, (destination, _writer) in zip(temporary, outputs, strict=True):
            os.replace(source, destination)
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    fields = list(rows[0]) if rows else list(fieldnames or ())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        raise DatasetAuditError("Cannot create an empty train/validation records manifest")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _inspection_payload(inspection: DatasetInspection) -> dict[str, object]:
    sample_count = min(20, inspection.audio_file_count)
    return {
        "dataset_root": str(inspection.root),
        "metadata_files": list(inspection.metadata_files),
        "metadata_columns": {
            filename: list(headers) for filename, headers in inspection.metadata_columns.items()
        },
        "top_level_folder_names": list(inspection.folder_names),
        "version_markers": inspection.version_markers,
        "audio_file_count": inspection.audio_file_count,
        "audio_file_sample": list(inspection.audio_files[:sample_count]),
        "audio_file_sample_omitted_count": inspection.audio_file_count - sample_count,
        "inspection_only": True,
    }


def _record_payload(record: object) -> dict[str, object]:
    return {
        "sample_id": record.sample_id,
        "relative_path": record.relative_path,
        "raw_label": record.raw_label,
        "tier": record.tier,
        "group_id": record.group_id,
        "grouping_field": record.grouping_field,
        "duration_s": record.duration_s,
        "sample_rate": record.sample_rate,
        "channels": record.channels,
        "sha256": record.sha256,
    }


def _excluded_record_payload(record: ExcludedRecord) -> dict[str, object]:
    return {
        "sample_id": record.sample_id,
        "relative_path": record.relative_path,
        "raw_label": record.raw_label,
        "tier": record.tier,
        "group_id": record.group_id,
        "grouping_field": record.grouping_field,
        "duration_s": record.duration_s,
        "sample_rate": record.sample_rate,
        "channels": record.channels,
        "sha256": record.sha256,
        "reason": record.reason,
    }


def _manifest_payload(manifest: DatasetManifest) -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "dataset": "InfantCry-DBL",
        "model_generation": manifest.model_generation,
        "dataset_version": manifest.dataset_version,
        "doi": manifest.doi,
        "dataset_root": str(manifest.dataset_root),
        "metadata_file": manifest.metadata_filename,
        "source_record_count": manifest.source_record_count,
        "record_count": len(manifest.records),
        "excluded_record_count": len(manifest.excluded_records),
        "exclusion_ledger_file": (
            str(manifest.exclusions_file) if manifest.exclusions_file is not None else None
        ),
        "exclusion_ledger_sha256": manifest.exclusion_ledger_sha256,
        "class_counts_by_tier": manifest.class_counts_by_tier,
        "duplicate_family_count": manifest.duplicate_family_count,
        "duplicate_record_count": manifest.duplicate_record_count,
        "cross_label_duplicate_family_count": manifest.cross_label_duplicate_family_count,
        "source_cross_label_duplicate_family_count": manifest.source_cross_label_duplicate_family_count,
        "version_evidence_status": manifest.version_evidence_status,
        "version_markers": manifest.version_markers,
        "grouping_scope": manifest.grouping_scope,
        "grouping_identity_verified": manifest.grouping_identity_verified,
        "label_source": manifest.label_source,
        "cross_check_parent_label": manifest.cross_check_parent_label,
        "warnings": list(manifest.warnings),
        "records_file": "records.csv",
        "excluded_records_file": "excluded_records.csv",
    }


if __name__ == "__main__":
    raise SystemExit(main())

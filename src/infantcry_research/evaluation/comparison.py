"""One-shot, paired evaluation of a frozen validation-selected candidate set."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from infantcry_research.audio.features import AudioConfig
from infantcry_research.data.manifest import DatasetAuditError, Record
from infantcry_research.data.splits import group_component_ids
from infantcry_research.evaluation.metrics import classification_metrics, group_bootstrap_intervals
from infantcry_research.training.run_store import (
    RunStoreError,
    claim_locked_test_batch_access,
    complete_locked_test_batch_access,
    load_run,
)


NEURAL_CANDIDATES = {
    "logmel_cnn",
    "cnn_bilstm_attention",
    "mmt_reslstm",
    "qiu_mfcc_reslstm",
    "qiu_mmt_resnet",
    "qiu_mmt_lstm",
    "qiu_mmt_reslstm",
}
SHARED_TRAINING_FIELDS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "early_stopping_patience",
    "optimizer",
    "selection_metric",
    "deterministic_ops",
)


def _candidate_model_input_shape(config: dict[str, Any]) -> tuple[int, ...]:
    """Return the frozen input shape for the candidate model, not its feature tensor."""

    candidate_config = config.get("candidate_config")
    if isinstance(candidate_config, dict) and "input_shape" in candidate_config:
        raw_shape = candidate_config["input_shape"]
    else:
        raw_shape = config.get("input_shape", ())
    try:
        return tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError) as exc:
        raise DatasetAuditError("Candidate has an invalid frozen model input shape") from exc


def evaluate_candidate_set(
    selection_path: Path,
    *,
    dataset_root: Path,
    audit_dir: Path,
    ledger_root: Path,
    output_root: Path,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 42,
    protocol_id: str = "stage2-locked-test-batch-v1",
    device: str = "cpu",
    project_root: Path | None = None,
) -> Path:
    """Evaluate every preregistered candidate once, after validation selection.

    No locked-test records or waveforms are loaded until the global split ledger
    is claimed. A failure after that claim is intentionally terminal for this
    split; it cannot be retried or expanded into another test event.
    """

    selected_path = Path(selection_path).resolve()
    root = Path(dataset_root).resolve()
    audit_location = Path(audit_dir).resolve()
    out_root = Path(output_root).resolve()
    ledger_location = Path(ledger_root).resolve()
    del project_root  # The caller supplies explicit output and ledger roots.
    if device not in {"cpu", "gpu"}:
        raise DatasetAuditError("device must be explicitly 'cpu' or 'gpu'")
    if not protocol_id.strip():
        raise DatasetAuditError("protocol-id must not be empty")
    if bootstrap_resamples < 100:
        raise DatasetAuditError("bootstrap-resamples must be at least 100")
    if not selected_path.is_file():
        raise DatasetAuditError(f"Validation selection does not exist: {selected_path}")
    selection_sha = _sha256_file(selected_path)
    selection = _read_json(selected_path, "validation selection")
    if selection.get("test_accessed") is not False:
        raise DatasetAuditError("Validation selection must be frozen before test access")

    candidate_order = selection.get("candidate_set")
    ranking = selection.get("ranking")
    if (
        not isinstance(candidate_order, list)
        or not candidate_order
        or len(set(str(item) for item in candidate_order)) != len(candidate_order)
        or not isinstance(ranking, list)
        or len(ranking) != len(candidate_order)
    ):
        raise DatasetAuditError("Selection must contain one complete, unique ranking for every candidate")
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for item in ranking:
        if not isinstance(item, dict):
            raise DatasetAuditError("Selection ranking entries must be JSON objects")
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id in candidates_by_id or not candidate_id:
            raise DatasetAuditError("Selection ranking contains a duplicate or empty candidate ID")
        candidates_by_id[candidate_id] = item
    normalized_order = [str(value) for value in candidate_order]
    if set(normalized_order) != set(candidates_by_id):
        raise DatasetAuditError("Selection candidate list and ranking do not contain the same exact set")

    split_id = str(selection.get("split_id", ""))
    selection_id = str(selection.get("selection_id", ""))
    if not split_id or not selection_id:
        raise DatasetAuditError("Validation selection is missing its immutable selection/split identity")
    if (ledger_location / f"{split_id}.json").exists():
        raise RunStoreError(f"The locked test for split {split_id!r} has already been claimed or evaluated")
    destination = out_root / split_id
    if destination.exists():
        raise RunStoreError(f"Comparison package already exists and will not be overwritten: {destination}")

    audit_path = audit_location / "dataset_audit.json"
    records_path = audit_location / "records.csv"
    audit_payload = _read_json(audit_path, "dataset audit")
    if audit_payload.get("version_evidence_status") not in {"verified", "verified_by_user"}:
        raise DatasetAuditError("Dataset release evidence is not verified")
    if Path(str(audit_payload.get("dataset_root", ""))).resolve() != root:
        raise DatasetAuditError("Dataset audit was created for a different dataset root")
    audit_sha = _sha256_file(audit_path)
    if not records_path.is_file():
        raise DatasetAuditError(f"Audited records manifest is missing: {records_path}")

    selection_protocol_hash = str(selection.get("candidate_protocol_sha256", ""))
    selection_split_hash = str(selection.get("split_sha256", ""))
    class_order: tuple[str, ...] | None = None
    common_data_evidence: dict[str, Any] | None = None
    common_training: dict[str, Any] | None = None
    common_split: dict[str, Any] | None = None
    prepared: list[dict[str, Any]] = []
    model_candidates = [candidates_by_id[candidate_id] for candidate_id in normalized_order]

    for candidate_id, ranking_entry in zip(normalized_order, model_candidates, strict=True):
        run_path = Path(str(ranking_entry.get("run_dir", ""))).resolve()
        if not run_path.is_dir():
            raise DatasetAuditError(f"Selection run directory is missing for {candidate_id}: {run_path}")
        run = load_run(run_path)
        if run.status != "complete":
            raise RunStoreError(f"Candidate run is not complete: {run.run_id}")
        if ranking_entry.get("run_id") != run.run_id:
            raise DatasetAuditError(f"Run ID mismatch in selection ranking for {candidate_id}")
        config_path = run.run_dir / "config.json"
        config = _read_json(config_path, f"{candidate_id} frozen config")
        split_payload = _read_json(run.run_dir / "split_manifest.json", f"{candidate_id} frozen split")
        validation = _read_json(
            run.run_dir / "artifacts" / "validation_metrics.json", f"{candidate_id} validation metrics"
        )
        if config.get("candidate_id") != candidate_id or run.metadata.get("selected_candidate_id") != candidate_id:
            raise DatasetAuditError(f"Candidate identity mismatch for run {run.run_id}")
        if config.get("candidate_protocol_sha256") != selection_protocol_hash:
            raise DatasetAuditError("Candidate protocol hash differs from the frozen validation selection")
        if config.get("registered_candidate_ids") != normalized_order:
            raise DatasetAuditError("Candidate run did not freeze the exact selected candidate set")
        if run.hashes.get("split_sha256") != selection_split_hash:
            raise DatasetAuditError("Candidate run split hash differs from the frozen selection")
        if split_payload.get("split_id") != split_id:
            raise DatasetAuditError("Candidate run has a different split ID")
        if split_payload.get("locked_test") is not True or split_payload.get("test_selection_allowed") is not False:
            raise DatasetAuditError("Candidate run split does not enforce the locked-test policy")
        if validation.get("evaluation_scope") != "validation_only":
            raise DatasetAuditError(f"Candidate {candidate_id} lacks validation-only selection evidence")
        validation_score = float(validation.get("validation_macro_f1", -1.0))
        if not 0 <= validation_score <= 1 or validation_score != float(
            ranking_entry.get("validation_macro_f1", -2.0)
        ):
            raise DatasetAuditError(f"Validation score mismatch for candidate {candidate_id}")
        if validation_score != float(ranking_entry.get("validation_macro_f1", -2.0)):
            raise DatasetAuditError(f"Invalid validation score in selection for candidate {candidate_id}")

        run_data = config.get("dataset_evidence")
        run_training = config.get("training")
        run_classes = config.get("class_order")
        feature_payload = config.get("feature_config")
        if not isinstance(run_data, dict) or not isinstance(run_training, dict):
            raise DatasetAuditError(f"Candidate {candidate_id} is missing frozen data/training evidence")
        if not isinstance(run_classes, list) or not isinstance(feature_payload, dict):
            raise DatasetAuditError(f"Candidate {candidate_id} is missing class/feature contract")
        if Path(str(run_data.get("dataset_root", ""))).resolve() != root:
            raise DatasetAuditError("Evaluation dataset root differs from candidate training evidence")
        if run_data.get("audit_sha256") != audit_sha:
            raise DatasetAuditError("Dataset audit changed since candidate training")
        if not run_data.get("records_csv_sha256"):
            raise DatasetAuditError("Candidate is missing the frozen records-manifest hash")
        if class_order is None:
            class_order = tuple(str(value) for value in run_classes)
        elif tuple(str(value) for value in run_classes) != class_order:
            raise DatasetAuditError("Candidate class order differs from the other registered runs")
        training_contract = {key: run_training.get(key) for key in SHARED_TRAINING_FIELDS}
        evidence_contract = {
            "dataset_root": str(Path(str(run_data["dataset_root"])).resolve()),
            "dataset_version": run_data.get("dataset_version"),
            "audit_sha256": run_data.get("audit_sha256"),
            "records_csv_sha256": run_data.get("records_csv_sha256"),
        }
        if common_data_evidence is None:
            common_data_evidence = evidence_contract
            common_training = {"seed": run_training.get("seed"), **training_contract}
            common_split = split_payload
        else:
            if evidence_contract != common_data_evidence:
                raise DatasetAuditError("Candidates do not share the same frozen dataset evidence")
            if {"seed": run_training.get("seed"), **training_contract} != common_training:
                raise DatasetAuditError("Candidates do not share the same seed and training controls")
            if split_payload != common_split:
                raise DatasetAuditError("Candidates do not share the exact frozen split")

        feature_hash = _sha256_json(feature_payload)
        if feature_hash != ranking_entry.get("feature_config_sha256"):
            raise DatasetAuditError(f"Feature contract hash mismatch for candidate {candidate_id}")
        if run.hashes.get("config_sha256") != ranking_entry.get("config_sha256"):
            raise DatasetAuditError(f"Config hash mismatch for candidate {candidate_id}")
        artifact_name = "best_model.keras" if candidate_id in NEURAL_CANDIDATES else "model.joblib"
        artifact_relative = f"artifacts/{artifact_name}"
        if artifact_relative not in run.metadata.get("training_artifact_hashes", {}):
            raise RunStoreError(f"Model artifact is not hash-frozen for {candidate_id}: {artifact_relative}")
        audio_config = AudioConfig.from_mapping(feature_payload)
        expected_shape = _candidate_model_input_shape(config)
        if len(expected_shape) != 3:
            raise DatasetAuditError(f"Candidate {candidate_id} has an invalid frozen model input shape")
        if candidate_id != "logmel_cnn" and expected_shape != audio_config.tensor_shape:
            raise DatasetAuditError(f"Candidate {candidate_id} input shape differs from its feature contract")
        if validation.get("candidate_id") != candidate_id:
            raise DatasetAuditError(f"Validation metrics candidate mismatch for {candidate_id}")
        prepared.append(
            {
                "candidate_id": candidate_id,
                "run": run,
                "config": config,
                "feature_config": feature_payload,
                "feature_hash": feature_hash,
                "audio_config": audio_config,
                "model_path": run.run_dir / artifact_relative,
                "artifact_sha256": run.metadata["training_artifact_hashes"][artifact_relative],
                "validation_macro_f1": validation_score,
            }
        )

    assert class_order is not None and common_split is not None
    selected_id = str(selection.get("selected_candidate_id", ""))
    selected_run_id = str(selection.get("selected_run_id", ""))
    selected_entry = next((item for item in prepared if item["candidate_id"] == selected_id), None)
    if selected_entry is None or selected_entry["run"].run_id != selected_run_id:
        raise DatasetAuditError("Validation-selected candidate/run is not present in the complete candidate set")
    selection_link = selected_entry["run"].metadata.get("selection_evidence")
    if not isinstance(selection_link, dict):
        raise RunStoreError("Freeze the cross-candidate validation selection before opening the test")
    if (
        Path(str(selection_link.get("selection_path", ""))).resolve() != selected_path
        or selection_link.get("selection_sha256") != selection_sha
        or selection_link.get("selection_id") != selection_id
    ):
        raise RunStoreError("Selection file does not match the validation evidence frozen in the winning run")
    # Verify selection actually follows the predeclared validation rule; test
    # scores are neither available nor consulted here.
    candidate_tie_order = {candidate_id: index for index, candidate_id in enumerate(normalized_order)}
    expected_winner = max(
        prepared,
        key=lambda item: (
            float(item["validation_macro_f1"]),
            -candidate_tie_order[str(item["candidate_id"])],
        ),
    )
    if expected_winner["candidate_id"] != selected_id:
        raise DatasetAuditError("Frozen selected candidate does not match the declared validation selection rule")

    split_ids = common_split.get("split_ids")
    if not isinstance(split_ids, dict) or not isinstance(split_ids.get("test"), list):
        raise DatasetAuditError("Frozen split is missing locked-test IDs")
    test_ids = tuple(str(value) for value in split_ids["test"])
    if not test_ids or len(test_ids) != len(set(test_ids)):
        raise DatasetAuditError("Locked-test IDs are empty or duplicated")
    tier = str(selected_entry["config"].get("tier", ""))
    if not tier or any(str(item["config"].get("tier", "")).casefold() != tier.casefold() for item in prepared):
        raise DatasetAuditError("Candidate runs do not share the same frozen dataset tier")

    # Load models and validate the requested inference runtime before the claim.
    neural_items = [item for item in prepared if item["candidate_id"] in NEURAL_CANDIDATES]
    runtime: dict[str, Any] = {"selected_device": "cpu", "gpu_count": 0, "gpu_names": []}
    tf: Any | None = None
    if neural_items:
        import tensorflow as tf_module

        tf = tf_module
        from infantcry_research.training.train import configure_tensorflow_runtime

        runtime = configure_tensorflow_runtime(
            tf,
            device=device,
            require_gpu=device == "gpu",
            mixed_precision=False,
        )
    for item in prepared:
        candidate_id = str(item["candidate_id"])
        if candidate_id in NEURAL_CANDIDATES:
            assert tf is not None
            from infantcry_research.models.layers import TemporalAttention

            custom = {"TemporalAttention": TemporalAttention} if candidate_id == "cnn_bilstm_attention" else {}
            item["model"] = tf.keras.models.load_model(
                item["model_path"], custom_objects=custom, compile=False
            )
            item["evaluation_device"] = device
        elif candidate_id in {"majority", "mfcc_svm"}:
            import joblib

            item["model"] = joblib.load(item["model_path"])
            item["evaluation_device"] = "cpu"
        else:
            raise DatasetAuditError(f"Unsupported frozen candidate: {candidate_id}")

    protocol = {
        "schema_version": "1.0",
        "protocol_id": protocol_id,
        "evaluation_scope": "single_locked_internal_test_batch",
        "selection_id": selection_id,
        "selection_sha256": selection_sha,
        "split_id": split_id,
        "split_sha256": selection_split_hash,
        "candidate_order": normalized_order,
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "run_id": item["run"].run_id,
                "config_sha256": item["run"].hashes["config_sha256"],
                "feature_config_sha256": item["feature_hash"],
                "model_artifact_sha256": item["artifact_sha256"],
            }
            for item in prepared
        ],
        "selected_by_validation": {"candidate_id": selected_id, "run_id": selected_run_id},
        "class_order": list(class_order),
        "tier": tier,
        "evaluation_device": device,
        "bootstrap_method": (
            "percentile_group_bootstrap"
            if str(common_split.get("grouping_scope", "")).startswith("verified_independent_group:")
            else "not_reported_unverified_grouping"
        ),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "test_sample_ids_sha256": _sha256_json(list(test_ids)),
        "test_scores_used_for_selection": False,
    }
    protocol_hash = _sha256_json(protocol)

    # One global, write-once claim for this split and exact selection/candidate set.
    claim_path = claim_locked_test_batch_access(
        ledger_location,
        split_id=split_id,
        selection_id=selection_id,
        selection_sha256=selection_sha,
        candidate_runs=[
            {
                "candidate_id": item["candidate_id"],
                "run_id": item["run"].run_id,
                "config_sha256": item["run"].hashes["config_sha256"],
                "feature_config_sha256": item["feature_hash"],
            }
            for item in prepared
        ],
        protocol_hash=protocol_hash,
    )

    # From this point onward, all access to records.csv, test labels, and test
    # audio is covered by the global one-shot event.
    if _sha256_file(records_path) != common_data_evidence["records_csv_sha256"]:
        raise DatasetAuditError("Dataset records manifest changed since candidate training")
    test_records = _load_test_records(records_path, root, set(test_ids))
    by_id = {record.sample_id: record for record in test_records}
    ordered_records = [by_id[sample_id] for sample_id in test_ids]
    if any(record.tier.casefold() != tier.casefold() for record in ordered_records):
        raise DatasetAuditError("Locked-test sample IDs include a record outside the trained Tier")
    y_true = tuple(record.raw_label.strip().casefold() for record in ordered_records)
    if set(y_true) - set(class_order):
        raise DatasetAuditError("Locked test contains labels outside the frozen class order")
    test_component_ids = group_component_ids(ordered_records, str(common_split.get("group_key", "")))
    groups = tuple(test_component_ids[sample_id] for sample_id in test_ids)

    feature_cache: dict[str, tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]] = {}
    candidate_results: list[dict[str, Any]] = []
    for item in prepared:
        candidate_id = str(item["candidate_id"])
        model = item["model"]
        if candidate_id == "majority":
            model_input = np.empty((len(test_ids), 0), dtype=np.float32)
            raw_probabilities = model.predict_proba(model_input)
            probabilities = _align_probabilities(model, raw_probabilities, class_order)
        else:
            feature_key = str(item["feature_hash"])
            if feature_key not in feature_cache:
                from infantcry_research.training.data import load_feature_partition

                feature_cache[feature_key] = load_feature_partition(
                    by_id, test_ids, item["audio_config"]
                )
            features, labels, feature_ids = feature_cache[feature_key]
            if feature_ids != test_ids or tuple(value.strip().casefold() for value in labels) != y_true:
                raise DatasetAuditError("Decoded test records do not match frozen test IDs/labels")
            expected_shape = _candidate_model_input_shape(item["config"])
            if candidate_id == "mfcc_svm":
                model_input = features
                raw_probabilities = model.predict_proba(model_input)
                probabilities = _align_probabilities(model, raw_probabilities, class_order)
            else:
                if candidate_id == "logmel_cnn":
                    audio_config = item["audio_config"]
                    start = 3 * audio_config.n_mfcc
                    model_input = features[:, start : start + audio_config.n_mels, :, :]
                else:
                    model_input = features
                if tuple(model_input.shape[1:]) != expected_shape:
                    raise DatasetAuditError(
                        f"Decoded feature shape for {candidate_id} is {model_input.shape[1:]}; "
                        f"expected {expected_shape}"
                    )
                assert tf is not None
                with tf.device("/GPU:0" if device == "gpu" else "/CPU:0"):
                    probabilities = np.asarray(model.predict(model_input, verbose=0), dtype=np.float64)

        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (len(test_ids), len(class_order)) or not np.isfinite(probabilities).all():
            raise DatasetAuditError(f"Candidate {candidate_id} emitted invalid test probabilities")
        row_sums = probabilities.sum(axis=1)
        if np.any(row_sums <= 0):
            raise DatasetAuditError(f"Candidate {candidate_id} emitted zero-sum test probabilities")
        probabilities = probabilities / row_sums[:, None]
        predictions = tuple(class_order[index] for index in np.argmax(probabilities, axis=1))
        metrics = classification_metrics(y_true, predictions, probabilities, class_order)
        if str(common_split.get("grouping_scope", "")).startswith("verified_independent_group:"):
            try:
                uncertainty = group_bootstrap_intervals(
                    y_true,
                    predictions,
                    groups,
                    class_order=class_order,
                    n_resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                    confidence_level=0.95,
                )
                uncertainty["status"] = "available"
            except ValueError as exc:
                uncertainty = {"status": "unavailable", "method": None, "reason": str(exc)}
        else:
            uncertainty = {
                "status": "unavailable",
                "method": None,
                "reason": (
                    "A verified independent subject/session grouping field was not verified; "
                    "recordings are not treated as independent groups."
                ),
            }
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "run_id": item["run"].run_id,
                "protocol_hash": protocol_hash,
                "selected_by_validation": candidate_id == selected_id,
                "validation_macro_f1": item["validation_macro_f1"],
                "test_metrics": metrics,
                "uncertainty": uncertainty,
                "evaluation_device": item["evaluation_device"],
                "sample_ids": test_ids,
                "y_true": y_true,
                "y_pred": predictions,
                "y_score": probabilities,
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RunStoreError(f"Comparison package already exists and will not be overwritten: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{split_id}.", dir=out_root))
    try:
        (temporary / "protocol.json").write_text(
            json.dumps({**protocol, "protocol_hash": protocol_hash}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for result in candidate_results:
            _write_candidate_artifacts(
                temporary / "candidates" / str(result["candidate_id"]),
                result,
                class_order,
            )
        comparison = {
            "schema_version": "1.0",
            "status": "complete",
            "evaluation_scope": "single_locked_internal_test_batch",
            "split_id": split_id,
            "selection_id": selection_id,
            "selection_sha256": selection_sha,
            "protocol_hash": protocol_hash,
            "test_support": len(test_ids),
            "class_order": list(class_order),
            "candidate_order": normalized_order,
            "selected_by_validation": {"candidate_id": selected_id, "run_id": selected_run_id},
            "test_scores_used_for_selection": False,
            "interpretation": (
                "Test metrics are descriptive paired results only. The candidate was selected using "
                "validation macro-F1 before this one-time test event; test values must not be used "
                "to tune, re-rank, or promote a model."
            ),
            "candidates": [
                {
                    "candidate_id": result["candidate_id"],
                    "run_id": result["run_id"],
                    "selected_by_validation": result["selected_by_validation"],
                    "validation_macro_f1": result["validation_macro_f1"],
                    "test_metrics": result["test_metrics"],
                    "uncertainty": result["uncertainty"],
                    "evaluation_device": result["evaluation_device"],
                    "artifacts": f"candidates/{result['candidate_id']}",
                }
                for result in candidate_results
            ],
        }
        (temporary / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        file_hashes = {
            path.relative_to(temporary).as_posix(): _sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest_bytes = (
            json.dumps(
                {
                    "schema_version": "1.0",
                    "split_id": split_id,
                    "selection_sha256": selection_sha,
                    "protocol_hash": protocol_hash,
                    "files": file_hashes,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        (temporary / "artifact_hashes.json").write_bytes(manifest_bytes)
        evaluation_sha = hashlib.sha256(manifest_bytes).hexdigest()
        os.replace(temporary, destination)
        complete_locked_test_batch_access(
            claim_path,
            selection_sha256=selection_sha,
            protocol_hash=protocol_hash,
            evaluation_sha256=evaluation_sha,
            comparison_dir=destination,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return destination


def _write_candidate_artifacts(destination: Path, result: dict[str, Any], classes: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    metrics = {
        "schema_version": "1.0",
        "evaluation_scope": "locked_internal_test",
        "candidate_id": result["candidate_id"],
        "run_id": result["run_id"],
        "selected_by_validation": result["selected_by_validation"],
        "validation_macro_f1": result["validation_macro_f1"],
        "protocol_hash": result["protocol_hash"],
        "support": result["test_metrics"]["support"],
        "class_order": list(classes),
        "metrics": result["test_metrics"],
        "uncertainty": result["uncertainty"],
        "evaluation_device": result["evaluation_device"],
        "external_validation_performed": False,
        "test_scores_used_for_selection": False,
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (destination / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["sample_id", "true_label", "predicted_label"] + [f"probability_{label}" for label in classes]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sample_id in enumerate(result["sample_ids"]):
            row: dict[str, object] = {
                "sample_id": sample_id,
                "true_label": result["y_true"][index],
                "predicted_label": result["y_pred"][index],
            }
            row.update(
                {
                    f"probability_{label}": format(float(result["y_score"][index, class_index]), ".10g")
                    for class_index, label in enumerate(classes)
                }
            )
            writer.writerow(row)

    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(result["y_true"], result["y_pred"], labels=list(classes))
    with (destination / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\predicted", *classes])
        for label, values in zip(classes, matrix, strict=True):
            writer.writerow([label, *[int(value) for value in values]])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Locked-test confusion matrix — {result['candidate_id']}",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.colorbar(image, ax=axis, label="Number of clips")
    figure.savefig(destination / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def _load_test_records(records_path: Path, dataset_root: Path, sample_ids: set[str]) -> list[Record]:
    required = {"sample_id", "relative_path", "raw_label", "tier", "duration_s", "sample_rate", "channels", "sha256"}
    try:
        with records_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise DatasetAuditError("Audited records.csv is missing columns: " + ", ".join(sorted(missing)))
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetAuditError(f"Could not read locked-test records after claim: {exc}") from exc
    result: list[Record] = []
    found: set[str] = set()
    root = dataset_root.resolve()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in sample_ids:
            continue
        if sample_id in found:
            raise DatasetAuditError(f"Duplicate locked-test sample ID in records.csv: {sample_id}")
        found.add(sample_id)
        relative = str(row.get("relative_path") or "").replace("\\", "/")
        audio_path = (root / relative).resolve()
        try:
            audio_path.relative_to(root)
        except ValueError as exc:
            raise DatasetAuditError(f"Locked-test path escapes dataset root: {relative}") from exc
        if not audio_path.is_file():
            raise DatasetAuditError(f"Locked-test audio file no longer exists: {relative}")
        if _sha256_file(audio_path) != row.get("sha256"):
            raise DatasetAuditError(f"Locked-test audio SHA-256 differs from audit: {relative}")
        try:
            result.append(
                Record(
                    sample_id=sample_id,
                    path=audio_path,
                    relative_path=relative,
                    raw_label=str(row["raw_label"]),
                    tier=str(row["tier"]),
                    group_id=row.get("group_id") or None,
                    duration_s=float(row["duration_s"]),
                    sha256=str(row["sha256"]),
                    sample_rate=int(row["sample_rate"]),
                    channels=int(row["channels"]),
                    grouping_field=row.get("grouping_field") or None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetAuditError(f"Invalid locked-test record {sample_id}: {exc}") from exc
    if found != sample_ids:
        raise DatasetAuditError(f"Locked-test IDs missing from records.csv: {sorted(sample_ids - found)[:10]}")
    return result


def _align_probabilities(estimator: Any, scores: np.ndarray, class_order: tuple[str, ...]) -> np.ndarray:
    source_classes = tuple(str(value) for value in getattr(estimator, "classes_", ()))
    probabilities = np.asarray(scores, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(source_classes):
        raise DatasetAuditError("Estimator probability columns do not match fitted estimator classes")
    aligned = np.zeros((probabilities.shape[0], len(class_order)), dtype=np.float64)
    class_index = {label: index for index, label in enumerate(class_order)}
    for index, label in enumerate(source_classes):
        if label not in class_index:
            raise DatasetAuditError(f"Estimator emitted a class outside the frozen class order: {label}")
        aligned[:, class_index[label]] = probabilities[:, index]
    return aligned


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetAuditError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetAuditError(f"{description} must contain a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

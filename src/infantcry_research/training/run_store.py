"""Write-once experiment evidence and guarded run lifecycle."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunStoreError(ValueError):
    """Raised when a run would overwrite or violate its evidence lifecycle."""


@dataclass
class RunStore:
    run_dir: Path
    run_id: str
    hashes: dict[str, str]

    @property
    def status(self) -> str:
        return self._read_metadata()["status"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._read_metadata()

    def mark_training_started(self) -> None:
        metadata = self._read_metadata()
        if metadata["status"] != "prepared":
            raise RunStoreError(f"Training can start only from prepared status, got {metadata['status']!r}")
        metadata["status"] = "running"
        _append_event(metadata, "training_started")
        self._write_metadata(metadata)

    def mark_training_complete(self, *, selected_candidate_id: str) -> None:
        metadata = self._read_metadata()
        if metadata["status"] == "complete":
            raise RunStoreError("Run is already complete; selection metadata is immutable")
        if metadata["status"] != "running":
            raise RunStoreError(f"Run must be running before completion, got {metadata['status']!r}")
        if selected_candidate_id not in metadata["candidate_ids"]:
            raise RunStoreError(f"Selected candidate {selected_candidate_id!r} was not preregistered")
        metadata["selected_candidate_id"] = selected_candidate_id
        metadata["status"] = "complete"
        _append_event(metadata, "training_completed", selected_candidate_id=selected_candidate_id)
        self._write_metadata(metadata)

    def record_training_artifacts(self, paths: list[Path] | tuple[Path, ...]) -> None:
        """Freeze hashes for the trained model and its validation evidence."""

        metadata = self._read_metadata()
        if metadata["status"] != "running":
            raise RunStoreError("Training artifacts can be frozen only while the run is running")
        if metadata.get("training_artifact_hashes"):
            raise RunStoreError("Training artifact hashes are already frozen")
        if not paths:
            raise RunStoreError("At least one training artifact must be frozen")
        artifacts: dict[str, str] = {}
        for value in paths:
            path = Path(value).resolve()
            try:
                relative = path.relative_to(self.run_dir.resolve()).as_posix()
            except ValueError as exc:
                raise RunStoreError(f"Training artifact is outside the run directory: {path}") from exc
            if not path.is_file():
                raise RunStoreError(f"Training artifact does not exist: {path}")
            if relative in artifacts:
                raise RunStoreError(f"Duplicate training artifact path: {relative}")
            artifacts[relative] = _sha256_file(path)
        metadata["training_artifact_hashes"] = artifacts
        _append_event(metadata, "training_artifacts_frozen", artifact_count=len(artifacts))
        self._write_metadata(metadata)

    def record_validation_selection(
        self, *, selection_id: str, selection_path: Path, selection_sha256: str
    ) -> None:
        """Attach immutable cross-candidate validation selection evidence to the winner."""

        metadata = self._read_metadata()
        if metadata["status"] != "complete":
            raise RunStoreError("Validation selection requires a completed candidate run")
        if metadata.get("selection_evidence"):
            raise RunStoreError("A validation selection is already frozen for this run")
        path = Path(selection_path).resolve()
        if not path.is_file() or _sha256_file(path) != selection_sha256:
            raise RunStoreError("Validation selection file/hash could not be verified")
        metadata["selection_evidence"] = {
            "selection_id": _safe_component(selection_id),
            "selection_path": str(path),
            "selection_sha256": selection_sha256,
        }
        _append_event(metadata, "validation_selection_frozen", selection_id=selection_id)
        self._write_metadata(metadata)

    def mark_training_failed(self, *, reason: str) -> None:
        metadata = self._read_metadata()
        if metadata["status"] not in {"prepared", "running"}:
            raise RunStoreError(f"Cannot mark run failed from status {metadata['status']!r}")
        metadata["status"] = "failed"
        metadata["failure_reason"] = str(reason)
        _append_event(metadata, "training_failed", reason=str(reason))
        self._write_metadata(metadata)

    def begin_test_evaluation(self, candidate_id: str, protocol_hash: str) -> Path:
        """Claim a one-shot locked-test evaluation before any test records are loaded."""

        metadata = self._read_metadata()
        if metadata["status"] != "complete":
            raise RunStoreError("Locked test evaluation requires a completed run")
        if candidate_id not in metadata["candidate_ids"]:
            raise RunStoreError(f"Candidate {candidate_id!r} was not preregistered")
        if metadata.get("selected_candidate_id") != candidate_id:
            raise RunStoreError("Only the candidate selected from validation may access the locked test")
        if not isinstance(protocol_hash, str) or not protocol_hash.strip():
            raise RunStoreError("protocol_hash must be a non-empty string")
        safe_candidate = _safe_component(candidate_id)
        safe_protocol = _safe_component(protocol_hash)
        evaluation_dir = self.run_dir / "evaluations" / "test"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        claim_path = evaluation_dir / f"{safe_candidate}__{safe_protocol}.claim.json"
        result_path = evaluation_dir / f"{safe_candidate}__{safe_protocol}.json"
        candidate_lock = evaluation_dir / f"{safe_candidate}.locked_test_claim"
        if candidate_lock.exists() or list(evaluation_dir.glob(f"{safe_candidate}__*.json")):
            raise RunStoreError(
                f"Candidate {candidate_id!r} has already accessed or claimed the locked test"
            )
        claim = {
            "candidate_id": candidate_id,
            "protocol_hash": protocol_hash,
            "evaluation_scope": "locked_test",
            "state": "claimed_before_test_access",
            "created_at_utc": _now_utc(),
        }
        try:
            with candidate_lock.open("x", encoding="utf-8") as stream:
                stream.write(protocol_hash + "\n")
            with claim_path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(claim, indent=2, ensure_ascii=False) + "\n")
        except FileExistsError as exc:
            raise RunStoreError("This selected candidate already claimed the locked test") from exc
        _append_event(metadata, "locked_test_claimed", candidate_id=candidate_id, protocol_hash=protocol_hash)
        self._write_metadata(metadata)
        return claim_path

    def record_test_evaluation(
        self, candidate_id: str, protocol_hash: str, test_evaluation_hash: str
    ) -> Path:
        metadata = self._read_metadata()
        if metadata["status"] != "complete":
            raise RunStoreError("Locked test evaluation requires a completed run")
        if candidate_id not in metadata["candidate_ids"]:
            raise RunStoreError(f"Candidate {candidate_id!r} was not preregistered")
        if metadata.get("selected_candidate_id") != candidate_id:
            raise RunStoreError("Only the candidate selected from validation may be evaluated")
        for value, label in (
            (protocol_hash, "protocol_hash"),
            (test_evaluation_hash, "test_evaluation_hash"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RunStoreError(f"{label} must be a non-empty string")
        safe_candidate = _safe_component(candidate_id)
        safe_protocol = _safe_component(protocol_hash)
        evaluation_dir = self.run_dir / "evaluations" / "test"
        claim_path = evaluation_dir / f"{safe_candidate}__{safe_protocol}.claim.json"
        if not claim_path.is_file():
            raise RunStoreError("Claim the locked-test evaluation before reading test data")
        try:
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RunStoreError(f"Could not read locked-test claim {claim_path}: {exc}") from exc
        if claim.get("candidate_id") != candidate_id or claim.get("protocol_hash") != protocol_hash:
            raise RunStoreError("Locked-test claim identity does not match the evaluation receipt")
        path = evaluation_dir / f"{safe_candidate}__{safe_protocol}.json"
        if path.exists():
            raise RunStoreError(
                f"Candidate {candidate_id!r} was already evaluated under protocol {protocol_hash!r}"
            )
        payload = {
            "candidate_id": candidate_id,
            "protocol_hash": protocol_hash,
            "test_evaluation_hash": test_evaluation_hash,
            "evaluation_scope": "locked_test",
            "created_at_utc": _now_utc(),
        }
        _write_json_once(path, payload)
        _append_event(metadata, "locked_test_evaluated", candidate_id=candidate_id, protocol_hash=protocol_hash)
        self._write_metadata(metadata)
        return path

    def _read_metadata(self) -> dict[str, Any]:
        path = self.run_dir / "metadata.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RunStoreError(f"Could not read run metadata {path}: {exc}") from exc
        if payload.get("run_id") != self.run_id:
            raise RunStoreError(f"Run metadata identity mismatch at {path}")
        return payload

    def _write_metadata(self, payload: dict[str, Any]) -> None:
        _write_json_replace(self.run_dir / "metadata.json", payload)


def load_run(run_dir: Path) -> RunStore:
    """Reopen a run after verifying all write-once evidence hashes."""

    location = Path(run_dir).resolve()
    try:
        metadata = json.loads((location / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStoreError(f"Could not read run metadata at {location}: {exc}") from exc
    run_id = metadata.get("run_id")
    hashes = metadata.get("hashes")
    if run_id != location.name or not isinstance(hashes, dict):
        raise RunStoreError(f"Run identity or hashes are invalid at {location}")
    for key, filename in (
        ("config_sha256", "config.json"),
        ("split_sha256", "split_manifest.json"),
        ("environment_sha256", "environment.json"),
    ):
        path = location / filename
        expected = hashes.get(key)
        try:
            frozen_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RunStoreError(f"Could not read frozen run evidence {path}: {exc}") from exc
        if not isinstance(frozen_payload, dict) or _sha256_json(frozen_payload) != expected:
            raise RunStoreError(f"Frozen run evidence hash mismatch: {path}")
    for relative, expected in metadata.get("training_artifact_hashes", {}).items():
        path = (location / relative).resolve()
        try:
            path.relative_to(location)
        except ValueError as exc:
            raise RunStoreError(f"Unsafe training artifact path in run metadata: {relative}") from exc
        if not path.is_file() or _sha256_file(path) != expected:
            raise RunStoreError(f"Frozen training artifact hash mismatch: {path}")
    selection = metadata.get("selection_evidence")
    if selection:
        selection_path = Path(str(selection.get("selection_path", ""))).resolve()
        if not selection_path.is_file() or _sha256_file(selection_path) != selection.get("selection_sha256"):
            raise RunStoreError("Frozen cross-candidate validation selection hash mismatch")
    return RunStore(run_dir=location, run_id=str(run_id), hashes=dict(hashes))


def claim_locked_split_access(
    ledger_root: Path,
    *,
    split_id: str,
    candidate_id: str,
    run_id: str,
    protocol_hash: str,
) -> Path:
    """Atomically claim the one permitted test opening for a frozen split."""

    safe_split = _safe_component(split_id)
    safe_candidate = _safe_component(candidate_id)
    safe_run = _safe_component(run_id)
    safe_protocol = _safe_component(protocol_hash)
    root = Path(ledger_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{safe_split}.json"
    payload = {
        "schema_version": "1.0",
        "split_id": safe_split,
        "candidate_id": safe_candidate,
        "run_id": safe_run,
        "protocol_hash": safe_protocol,
        "state": "claimed_before_test_access",
        "created_at_utc": _now_utc(),
    }
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except FileExistsError as exc:
        raise RunStoreError(
            f"The locked test for split {split_id!r} has already been claimed or evaluated"
        ) from exc
    return path


def claim_locked_test_batch_access(
    ledger_root: Path,
    *,
    split_id: str,
    selection_id: str,
    selection_sha256: str,
    candidate_runs: list[dict[str, str]],
    protocol_hash: str,
) -> Path:
    """Atomically claim one locked-test event for a frozen validation-selected set."""

    safe_split = _safe_component(split_id)
    safe_selection = _safe_component(selection_id)
    for value, name in (
        (selection_sha256, "selection_sha256"),
        (protocol_hash, "protocol_hash"),
    ):
        if not _is_sha256(value):
            raise RunStoreError(f"{name} must be a lowercase SHA-256 digest")
    if not candidate_runs:
        raise RunStoreError("candidate_runs must be a non-empty frozen candidate list")

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in candidate_runs:
        candidate_id = _safe_component(str(item.get("candidate_id", "")))
        run_id = _safe_component(str(item.get("run_id", "")))
        if candidate_id in seen:
            raise RunStoreError(f"Duplicate candidate in locked-test batch: {candidate_id}")
        seen.add(candidate_id)
        config_hash = str(item.get("config_sha256", ""))
        feature_hash = str(item.get("feature_config_sha256", ""))
        if not _is_sha256(config_hash) or not _is_sha256(feature_hash):
            raise RunStoreError("Each candidate run must include valid config and feature SHA-256 hashes")
        normalized.append(
            {
                "candidate_id": candidate_id,
                "run_id": run_id,
                "config_sha256": config_hash,
                "feature_config_sha256": feature_hash,
            }
        )

    root = Path(ledger_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{safe_split}.json"
    payload = {
        "schema_version": "1.0",
        "split_id": safe_split,
        "selection_id": safe_selection,
        "selection_sha256": selection_sha256,
        "candidate_runs": normalized,
        "protocol_hash": protocol_hash,
        "state": "claimed_before_test_access",
        "created_at_utc": _now_utc(),
    }
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except FileExistsError as exc:
        raise RunStoreError(
            f"The locked test for split {split_id!r} has already been claimed or evaluated"
        ) from exc
    return path


def complete_locked_test_batch_access(
    claim_path: Path,
    *,
    selection_sha256: str,
    protocol_hash: str,
    evaluation_sha256: str,
    comparison_dir: Path,
) -> Path:
    """Mark a claimed batch event complete without changing its frozen identity."""

    path = Path(claim_path).resolve()
    for value, name in (
        (selection_sha256, "selection_sha256"),
        (protocol_hash, "protocol_hash"),
        (evaluation_sha256, "evaluation_sha256"),
    ):
        if not _is_sha256(value):
            raise RunStoreError(f"{name} must be a lowercase SHA-256 digest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStoreError(f"Could not read locked-test batch claim {path}: {exc}") from exc
    if payload.get("state") == "evaluated" or payload.get("evaluation_sha256"):
        raise RunStoreError("Locked-test batch has already been completed")
    if payload.get("state") != "claimed_before_test_access":
        raise RunStoreError("Locked-test batch claim is not in the claimed state")
    if payload.get("selection_sha256") != selection_sha256 or payload.get("protocol_hash") != protocol_hash:
        raise RunStoreError("Locked-test batch completion identity does not match its claim")
    destination = Path(comparison_dir).resolve()
    if not destination.is_dir():
        raise RunStoreError("Comparison package must exist before completing the test claim")
    payload["state"] = "evaluated"
    payload["evaluation_sha256"] = evaluation_sha256
    payload["comparison_dir"] = str(destination)
    payload["completed_at_utc"] = _now_utc()
    _write_json_replace(path, payload)
    return path


def create_run(
    run_root: Path,
    *,
    config: dict[str, Any],
    split_manifest: dict[str, Any],
    environment: dict[str, Any],
    candidate_ids: tuple[str, ...],
    run_id: str | None = None,
) -> RunStore:
    """Create a new run directory and freeze its input evidence exactly once."""

    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise RunStoreError("candidate_ids must be a non-empty list of unique model IDs")
    for candidate in candidate_ids:
        _safe_component(candidate)
    config_payload = _canonical_payload(config)
    split_payload = _canonical_payload(split_manifest)
    environment_payload = _canonical_payload(environment)
    config_hash = _sha256_json(config_payload)
    split_hash = _sha256_json(split_payload)
    environment_hash = _sha256_json(environment_payload)

    generated_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{config_hash[:8]}"
    _safe_component(generated_id)
    run_dir = Path(run_root).resolve() / generated_id
    if run_dir.exists():
        raise RunStoreError(f"Run ID already exists and will not be reused: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    hashes = {
        "config_sha256": config_hash,
        "split_sha256": split_hash,
        "environment_sha256": environment_hash,
    }
    try:
        _write_json_once(run_dir / "config.json", config_payload)
        _write_json_once(run_dir / "split_manifest.json", split_payload)
        _write_json_once(run_dir / "environment.json", environment_payload)
        (run_dir / "artifacts").mkdir()
        (run_dir / "logs").mkdir()
        metadata = {
            "schema_version": "1.0",
            "run_id": generated_id,
            "status": "prepared",
            "created_at_utc": _now_utc(),
            "candidate_ids": list(candidate_ids),
            "selected_candidate_id": None,
            "hashes": hashes,
            "locked_test_policy": "unopened_until_training_complete",
            "lifecycle": [{"event": "run_created", "at_utc": _now_utc()}],
        }
        _write_json_once(run_dir / "metadata.json", metadata)
    except Exception:
        # The caller owns this newly-created run directory; preserve partial evidence for diagnosis.
        raise
    return RunStore(run_dir=run_dir, run_id=generated_id, hashes=hashes)


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))
    except (TypeError, ValueError) as exc:
        raise RunStoreError(f"Run evidence is not JSON serializable: {exc}") from exc


def _sha256_json(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_component(value: str) -> str:
    if not isinstance(value, str) or not value or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise RunStoreError(f"Unsafe or empty run path component: {value!r}")
    if value in {".", ".."}:
        raise RunStoreError(f"Unsafe run path component: {value!r}")
    return value


def _append_event(metadata: dict[str, Any], event: str, **details: Any) -> None:
    metadata.setdefault("lifecycle", []).append(
        {"event": event, "at_utc": _now_utc(), **details}
    )


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RunStoreError(f"Evidence already exists and will not be overwritten: {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_json_replace(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)

"""Create a concise, evidence-linked report from an immutable completed run."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

from infantcry_research.training.run_store import RunStoreError, load_run


def build_report(run_dir: Path, report_root: Path) -> Path:
    """Write a report only when run evidence and its completion status verify."""

    run = load_run(run_dir)
    if run.status != "complete":
        raise RunStoreError("Reports can be generated only from completed runs")
    config = _read_json(run.run_dir / "config.json")
    metadata = run.metadata
    candidate = str(metadata.get("selected_candidate_id"))
    validation = _read_json(run.run_dir / "artifacts" / "validation_metrics.json")
    destination_root = Path(report_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{run.run_id}.md"
    if destination.exists():
        raise FileExistsError(f"A report already exists for this run and will not be overwritten: {destination}")

    run_config = config.get("candidate_config", {})
    training = config.get("training", {})
    feature_config = config.get("feature_config", {})
    lines = [
        f"<!-- generated-from-verified-run: {run.run_id} -->",
        f"# Stage 2 run `{run.run_id}`",
        "",
        "This report was generated from a completed run whose frozen configuration, split, "
        "environment, and training-artifact hashes were verified. It contains no clinical claim.",
        "",
        "## Run definition",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | InfantCry-DBL {config.get('dataset_evidence', {}).get('dataset_version', 'unspecified')} |",
        f"| Tier | {config.get('tier', 'unspecified')} |",
        f"| Candidate | `{candidate}` |",
        f"| Model settings | `{json.dumps(run_config, ensure_ascii=False, sort_keys=True)}` |",
        f"| Seed | {training.get('seed', 'not recorded')} |",
        f"| Device | {training.get('device', 'not recorded')} |",
        f"| Mixed precision | {training.get('mixed_precision', False)} |",
        f"| Audio features | `{json.dumps(feature_config, ensure_ascii=False, sort_keys=True)}` |",
        f"| Run status | `{metadata.get('status')}` |",
        "",
        "## Validation (used for model/checkpoint selection)",
        "",
        f"- Scope: `{validation.get('evaluation_scope', 'validation_only')}`",
        f"- Train support: {validation.get('train_support', 'not recorded')}",
        f"- Validation support: {validation.get('validation_support', 'not recorded')}",
        f"- Accuracy: {validation.get('validation_accuracy', 'not recorded')}",
        f"- Balanced accuracy: {validation.get('validation_balanced_accuracy', 'not recorded')}",
        f"- Macro-F1: {validation.get('validation_macro_f1', 'not recorded')}",
        "",
        "The validation score is not an independent test result. Candidate selection based on this "
        "score must be completed before opening the locked test.",
        "",
    ]

    test_receipts = sorted(
        path
        for path in (run.run_dir / "evaluations" / "test").glob("*.json")
        if not path.name.endswith(".claim.json")
    )
    if test_receipts:
        receipt = _read_json(test_receipts[0])
        protocol_hash = str(receipt.get("protocol_hash", ""))
        evaluation_dir = (
            run.run_dir
            / "evaluations"
            / "test"
            / f"{candidate}__{protocol_hash}"
        )
        _verify_evaluation_artifacts(evaluation_dir, receipt)
        test_metrics = _read_json(evaluation_dir / "metrics.json")
        relative_image = _relative_link(destination, evaluation_dir / "confusion_matrix.png")
        uncertainty = test_metrics.get("uncertainty", {})
        if uncertainty.get("status") == "available":
            uncertainty_line = (
                "- Group-bootstrap 95% macro-F1 interval: "
                f"{uncertainty.get('macro_f1')}"
            )
        else:
            uncertainty_line = (
                "- Group-bootstrap interval: unavailable — "
                f"{uncertainty.get('reason', 'independent grouping was not verified')}"
            )
        lines.extend(
            [
                "## Locked test (reported after model selection)",
                "",
                f"- Scope: `{test_metrics.get('evaluation_scope')}`",
                f"- Test support: {test_metrics.get('support')}",
                f"- Macro-F1: {test_metrics.get('metrics', {}).get('macro_f1')}",
                f"- Accuracy: {test_metrics.get('metrics', {}).get('accuracy')}",
                f"- Balanced accuracy: {test_metrics.get('metrics', {}).get('balanced_accuracy')}",
                uncertainty_line,
                "",
                f"![Locked-test confusion matrix]({relative_image})",
                "",
                f"[Test predictions]({_relative_link(destination, evaluation_dir / 'test_predictions.csv')}) · "
                f"[Full metrics]({_relative_link(destination, evaluation_dir / 'metrics.json')})",
                "",
                "This test result is internal to the audited InfantCry-DBL release and is not external validation.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Locked test",
                "",
                "Not evaluated. This is the intended state until the candidate and protocol are frozen.",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations to state with the result",
            "",
            f"- Grouping scope: `{config.get('split_grouping_scope', 'see frozen split manifest')}`; do not claim subject independence without verified identifiers.",
            "- The same public corpus may be used by prior work; this result is not external validation.",
            "- Cry-type labels are dataset labels and are not clinical diagnoses.",
            "- No accuracy target or generalization claim is implied by this report template.",
            "",
            f"Frozen run artifacts: `{run.run_dir}`.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    project_root = Path(__file__).resolve().parents[3]
    if destination_root == (project_root / "reports" / "stage2").resolve():
        _update_report_hub(project_root, destination_root)
    return destination


def build_comparison_report(
    comparison_dir: Path,
    report_root: Path,
    *,
    ledger_root: Path | None = None,
) -> Path:
    """Create a paired comparison report from a hash-verified batch package."""

    package = Path(comparison_dir).resolve()
    manifest_path = package / "artifact_hashes.json"
    manifest = _read_json(manifest_path)
    manifest_hash = _sha256(manifest_path)
    comparison = _read_json(package / "comparison.json")
    protocol = _read_json(package / "protocol.json")
    if comparison.get("status") != "complete":
        raise RunStoreError("Comparison report requires a completed locked-test package")
    if comparison.get("evaluation_scope") != "single_locked_internal_test_batch":
        raise RunStoreError("Comparison package has an unsupported evaluation scope")
    if comparison.get("test_scores_used_for_selection") is not False:
        raise RunStoreError("Comparison package does not preserve validation-only selection")
    if manifest.get("selection_sha256") != comparison.get("selection_sha256"):
        raise RunStoreError("Comparison manifest selection hash does not match its summary")
    if manifest.get("protocol_hash") != comparison.get("protocol_hash"):
        raise RunStoreError("Comparison manifest protocol hash does not match its summary")
    _verify_comparison_files(package, manifest)

    split_id = str(comparison.get("split_id", ""))
    ledger_location = (
        Path(ledger_root).resolve()
        if ledger_root is not None
        else Path(__file__).resolve().parents[3] / "manifests" / "test_access"
    )
    ledger_path = ledger_location / f"{split_id}.json"
    ledger = _read_json(ledger_path)
    if (
        ledger.get("state") != "evaluated"
        or ledger.get("comparison_dir") != str(package)
        or ledger.get("selection_sha256") != comparison.get("selection_sha256")
        or ledger.get("protocol_hash") != comparison.get("protocol_hash")
        or ledger.get("evaluation_sha256") != manifest_hash
    ):
        raise RunStoreError("Comparison package does not match the completed global test-access ledger")
    destination_root = Path(report_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{split_id}.comparison.md"
    if destination.exists():
        raise FileExistsError(f"A comparison report already exists and will not be overwritten: {destination}")
    selected = comparison.get("selected_by_validation", {})
    candidates = comparison.get("candidates", [])
    if not isinstance(selected, dict) or not isinstance(candidates, list) or not candidates:
        raise RunStoreError("Comparison package is missing selected-candidate or candidate metrics")

    lines = [
        f"<!-- generated-from-locked-test-batch: {split_id}; manifest-sha256: {manifest_hash} -->",
        f"# Stage 2 paired comparison — `{split_id}`",
        "",
        "This report summarizes one predeclared, paired evaluation event. All registered candidates "
        "were scored on the same locked-test sample IDs after model selection was frozen from validation.",
        "",
        "## At a glance",
        "",
        "| Item | Frozen evidence |",
        "|---|---|",
        f"| Selection | `{comparison.get('selection_id')}` |",
        f"| Selected using validation | `{selected.get('candidate_id')}` (`{selected.get('run_id')}`) |",
        f"| Test support | {comparison.get('test_support')} |",
        f"| Candidate count | {len(candidates)} |",
        f"| Protocol hash | `{comparison.get('protocol_hash')}` |",
        "| Test scores used for selection | No |",
        "",
        "## Candidate results",
        "",
        "The table preserves the preregistered candidate order; it is not ranked by test performance.",
        "",
        "| Candidate | Validation macro-F1 | Test accuracy | Test balanced accuracy | Test macro-F1 | Test weighted-F1 | Selected by validation | Confusion matrix |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    uncertainty_notes: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RunStoreError("Comparison candidate entries must be objects")
        candidate_id = str(candidate.get("candidate_id", ""))
        metrics = candidate.get("test_metrics", {})
        if not isinstance(metrics, dict):
            raise RunStoreError(f"Test metrics are missing for candidate {candidate_id}")
        artifacts = str(candidate.get("artifacts", f"candidates/{candidate_id}"))
        image = package / artifacts / "confusion_matrix.png"
        relative_image = _relative_link(destination, image)
        selected_mark = "Yes" if candidate.get("selected_by_validation") else "No"
        lines.append(
            "| `{candidate}` | {validation} | {accuracy} | {balanced} | {macro} | {weighted} | {selected} | "
            "[PNG]({image}) · [CSV]({csv}) |".format(
                candidate=candidate_id,
                validation=_fmt(candidate.get("validation_macro_f1")),
                accuracy=_fmt(metrics.get("accuracy")),
                balanced=_fmt(metrics.get("balanced_accuracy")),
                macro=_fmt(metrics.get("macro_f1")),
                weighted=_fmt(metrics.get("weighted_f1")),
                selected=selected_mark,
                image=relative_image,
                csv=_relative_link(destination, package / artifacts / "confusion_matrix.csv"),
            )
        )
        uncertainty = candidate.get("uncertainty", {})
        if isinstance(uncertainty, dict) and uncertainty.get("status") == "available":
            interval = uncertainty.get("macro_f1", {})
            uncertainty_notes.append(
                f"- `{candidate_id}` group-bootstrap 95% macro-F1 interval: "
                f"{interval.get('lower')}–{interval.get('upper')}"
            )
        elif isinstance(uncertainty, dict):
            uncertainty_notes.append(
                f"- `{candidate_id}` Group-bootstrap interval: unavailable — "
                f"{uncertainty.get('reason', 'verified independent groups were not available')}"
            )
    lines.extend(
        [
            "",
            "Uncertainty intervals:",
            "",
            *uncertainty_notes,
            "",
            "## Interpretation boundary",
            "",
            "- Test scores were not used for selection.",
            "- The model was selected using validation macro-F1 before this test event.",
            "- Test metrics are paired descriptive results only. They must not be used to choose, tune, "
            "re-rank, or promote a candidate; doing so would create test-set selection bias.",
            "- This is an internal held-out evaluation on the audited InfantCry-DBL release, not external validation.",
            "- Dataset labels are not clinical diagnoses; the result does not establish clinical or cross-device generalization.",
            "",
            "## Reproducibility and files",
            "",
            f"- Evaluation protocol: [protocol.json]({_relative_link(destination, package / 'protocol.json')})",
            f"- Comparison summary: [comparison.json]({_relative_link(destination, package / 'comparison.json')})",
            f"- Artifact hashes: [artifact_hashes.json]({_relative_link(destination, manifest_path)})",
            f"- Test predictions are stored under `{package / 'candidates'}`.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    project_root = Path(__file__).resolve().parents[3]
    if destination_root == (project_root / "reports" / "stage2").resolve():
        _update_report_hub(project_root, destination_root)
    return destination


def _verify_comparison_files(package: Path, manifest: dict[str, object]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RunStoreError("Comparison package has no artifact file hashes")
    for relative_name, expected_hash in files.items():
        path = (package / str(relative_name)).resolve()
        try:
            path.relative_to(package)
        except ValueError as exc:
            raise RunStoreError(f"Unsafe comparison artifact path: {relative_name}") from exc
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RunStoreError(f"Comparison artifact hash mismatch: {path}")


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _update_report_hub(project_root: Path, report_root: Path) -> None:
    hub = project_root / "reports" / "report_hub.md"
    stage_reports = sorted(report_root.glob("*.md"))
    rows = []
    for report in stage_reports:
        content = report.read_text(encoding="utf-8")
        first_line = content.splitlines()[0].strip() if content else ""
        if first_line.startswith("<!-- generated-from-verified-run: "):
            identity = first_line.removeprefix("<!-- generated-from-verified-run: ").removesuffix(" -->")
            record_type = "Candidate run"
            status = (
                "Training and locked-test evaluation complete"
                if "## Locked test (reported after model selection)" in content
                else "Training complete; batch comparison pending"
            )
        elif first_line.startswith("<!-- generated-from-locked-test-batch: "):
            identity = first_line.removeprefix("<!-- generated-from-locked-test-batch: ").split(";", 1)[0]
            record_type = "Paired comparison"
            status = "One-time locked-test batch complete"
        else:
            continue
        relative_report = Path(os.path.relpath(report, start=hub.parent)).as_posix()
        rows.append(
            f"| `{identity}` | {record_type} | {status} | [Open report]({relative_report}) |"
        )
    content = [
        "# Report Hub",
        "",
        "Verified Stage 2 experiment reports generated from immutable run artifacts. No results are "
        "shown here unless a completed run has a detailed report.",
        "",
        "| Split / Run ID | Record | Status | Report |",
        "|---|---|---|---|",
        *(rows or ["| — | — | No verified run reports yet | — |"]),
        "",
    ]
    hub.parent.mkdir(parents=True, exist_ok=True)
    hub.write_text("\n".join(content), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStoreError(f"Could not read verified report input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunStoreError(f"Report input must be a JSON object: {path}")
    return payload


def _relative_link(source: Path, target: Path) -> str:
    return Path(os.path.relpath(target, start=source.parent)).as_posix()


def _verify_evaluation_artifacts(evaluation_dir: Path, receipt: dict[str, object]) -> None:
    manifest_path = evaluation_dir / "artifact_hashes.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStoreError(f"Could not verify locked-test report artifacts: {exc}") from exc
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != receipt.get("test_evaluation_hash"):
        raise RunStoreError("Locked-test artifact manifest does not match its evaluation receipt")
    for name, expected in manifest.get("files", {}).items():
        path = (evaluation_dir / str(name)).resolve()
        try:
            path.relative_to(evaluation_dir.resolve())
        except ValueError as exc:
            raise RunStoreError(f"Unsafe report artifact path: {name}") from exc
        if not path.is_file() or _sha256(path) != expected:
            raise RunStoreError(f"Locked-test report artifact hash mismatch: {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

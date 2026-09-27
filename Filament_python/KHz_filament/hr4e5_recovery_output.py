"""Hash-bound optical output attempts and one active source locator for WP2."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .hr4e5_evidence import atomic_json, sha256_file


OUTPUT_ATTEMPT_SCHEMA = "khz_filament.hr4e5.wp2.output_attempt.v1"
ACTIVE_OUTPUT_SCHEMA = "khz_filament.hr4e5.wp2.active_output.v1"
_REQUIRED_OUTPUT = frozenset({
    "optical_run.json", "final_optical_field.npy", "scientific_ledger.npz",
    "sinks.hr3a_qion_samples.npy",
    "sinks.hr3a_qib_samples.npy",
    "sinks.hr3a_qraman_samples.npy",
    "sinks.hr3a_qthermal_samples.npy",
    "sinks.hr3b_delta_n_increment_samples.npy",
    "sinks.hr3b_delta_n_state_after_update_samples.npy",
})


def _inside(root: Path, path: str | Path, *, file: bool = False) -> Path:
    value = Path(path)
    if value.is_symlink():
        raise ValueError("output locator contains a symlink")
    resolved = value.resolve()
    if not resolved.is_relative_to(root) or (file and not resolved.is_file()):
        raise ValueError("output locator escapes campaign or names a missing file")
    return resolved


def _files(root: Path, output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError("output attempt contains a symlink directory")
            continue
        checked = _inside(root, path, file=True)
        if checked.is_symlink():
            raise ValueError("output attempt contains a symlink file")
        rows.append({"path": checked.relative_to(root).as_posix(),
                     "bytes": checked.stat().st_size, "sha256": sha256_file(checked)})
    present = {Path(row["path"]).relative_to(output.relative_to(root)).as_posix() for row in rows}
    if not _REQUIRED_OUTPUT.issubset(present):
        raise ValueError("optical output attempt lacks required scientific files")
    return rows


def qualify_output_attempt(*, campaign_root: str | Path, state_root: str | Path,
                           output_dir: str | Path, attempt_id: str,
                           campaign_id: str, scientific_pair_id: str,
                           trajectory: str, pulse: int, generation: str,
                           coordinator_epoch: str | int, writer_grant_id: str,
                           admission_identity_sha256: str,
                           implementation_identity_sha256: str) -> dict[str, Any]:
    """Snapshot a completed producer; never infer qualification from existence."""
    root = Path(campaign_root).resolve()
    state = _inside(root, state_root)
    output = _inside(root, output_dir)
    if not output.is_dir() or not re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_id):
        raise ValueError("output attempt path or ID is invalid")
    if trajectory not in {"R", "C"} or int(pulse) < 0:
        raise ValueError("output attempt trajectory/pulse is invalid")
    run_path = output / "optical_run.json"
    run = json.loads(_inside(root, run_path, file=True).read_text(encoding="utf-8"))
    if (run.get("status") != "PASS"
            or str(run.get("output_dir")) != str(output)
            or str(run.get("admission_identity_sha256")) != admission_identity_sha256
            or str(run.get("current_generation")) != generation):
        raise ValueError("optical producer result identity differs")
    record = {
        "schema": OUTPUT_ATTEMPT_SCHEMA, "status": "QUALIFIED_PRODUCER",
        "campaign_id": str(campaign_id), "scientific_pair_id": str(scientific_pair_id),
        "trajectory": trajectory, "pulse": int(pulse), "generation": str(generation),
        "logical_scientific_attempt": 0, "execution_attempt_id": attempt_id,
        "producer_coordinator_epoch": str(coordinator_epoch),
        "writer_grant_id": str(writer_grant_id),
        "admission_identity_sha256": admission_identity_sha256,
        "implementation_identity_sha256": implementation_identity_sha256,
        "output_dir": output.relative_to(root).as_posix(),
        "files": _files(root, output),
    }
    path = state / "output_attempts" / f"{attempt_id}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != record:
            raise ValueError("output attempt record changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, record, overwrite=False)
    return {**record, "record_path": str(path), "record_sha256": sha256_file(path)}


def validate_output_attempt(*, campaign_root: str | Path, record_path: str | Path,
                            expected_sha256: str | None = None) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    path = _inside(root, record_path, file=True)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("output attempt record hash changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != OUTPUT_ATTEMPT_SCHEMA or record.get("status") != "QUALIFIED_PRODUCER":
        raise ValueError("output attempt record is not qualified")
    output = _inside(root, root / str(record.get("output_dir", "")))
    if not output.is_dir():
        raise ValueError("qualified optical output is missing")
    seen = set()
    for row in record.get("files", []):
        relative = str(row.get("path", ""))
        if relative in seen:
            raise ValueError("output attempt has duplicate file identity")
        seen.add(relative)
        file = _inside(root, root / relative, file=True)
        if not file.is_relative_to(output) or file.stat().st_size != int(row.get("bytes", -1)):
            raise ValueError("output attempt file identity changed")
        if sha256_file(file) != str(row.get("sha256")):
            raise ValueError("output attempt file hash changed")
    present = {Path(name).relative_to(output.relative_to(root)).as_posix() for name in seen}
    if not _REQUIRED_OUTPUT.issubset(present):
        raise ValueError("output attempt required object set changed")
    return {**record, "record_path": str(path), "record_sha256": digest}


def activate_output_attempt(*, campaign_root: str | Path, state_root: str | Path,
                            record_path: str | Path, expected_record_sha256: str) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    state = _inside(root, state_root)
    qualified = validate_output_attempt(campaign_root=root, record_path=record_path,
                                        expected_sha256=expected_record_sha256)
    pointer_path = state / "ACTIVE_OUTPUT.json"
    old_hash = sha256_file(pointer_path) if pointer_path.is_file() else None
    if old_hash is not None:
        previous = validate_active_output(campaign_root=root, state_root=state)
        if (previous["scientific_pair_id"] != qualified["scientific_pair_id"]
                or previous["admission_identity_sha256"] != qualified["admission_identity_sha256"]
                or previous["implementation_identity_sha256"] != qualified["implementation_identity_sha256"]
                or previous["generation"] != qualified["generation"]):
            raise ValueError("replacement output does not belong to the active scientific pair")
        prior = json.loads(pointer_path.read_text(encoding="utf-8"))
        if (prior.get("record_path") == Path(record_path).resolve().relative_to(root).as_posix()
                and prior.get("record_sha256") == expected_record_sha256):
            return validate_active_output(campaign_root=root, state_root=state)
    pointer = {
        "schema": ACTIVE_OUTPUT_SCHEMA,
        "campaign_id": qualified["campaign_id"],
        "scientific_pair_id": qualified["scientific_pair_id"],
        "trajectory": qualified["trajectory"], "pulse": qualified["pulse"],
        "generation": qualified["generation"],
        "admission_identity_sha256": qualified["admission_identity_sha256"],
        "implementation_identity_sha256": qualified["implementation_identity_sha256"],
        "record_path": Path(record_path).resolve().relative_to(root).as_posix(),
        "record_sha256": expected_record_sha256,
        "previous_pointer_sha256": old_hash,
    }
    activation = state / "output_attempts" / f"activation_{qualified['execution_attempt_id']}.json"
    if activation.exists():
        if json.loads(activation.read_text(encoding="utf-8")) != pointer:
            raise ValueError("output activation lineage changed")
    else:
        atomic_json(activation, pointer, overwrite=False)
    atomic_json(pointer_path, pointer, overwrite=True)
    return validate_active_output(campaign_root=root, state_root=state)


def validate_active_output(*, campaign_root: str | Path, state_root: str | Path,
                           scientific_pair_id: str | None = None,
                           implementation_identity_sha256: str | None = None) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    state = _inside(root, state_root)
    pointer_path = _inside(root, state / "ACTIVE_OUTPUT.json", file=True)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema") != ACTIVE_OUTPUT_SCHEMA:
        raise ValueError("active output locator schema differs")
    if scientific_pair_id is not None and pointer.get("scientific_pair_id") != scientific_pair_id:
        raise ValueError("active output scientific pair identity differs")
    if (implementation_identity_sha256 is not None
            and pointer.get("implementation_identity_sha256") != implementation_identity_sha256):
        raise ValueError("REUSE_REJECTED / IMPLEMENTATION_IDENTITY_MISMATCH")
    record = validate_output_attempt(
        campaign_root=root, record_path=root / str(pointer.get("record_path", "")),
        expected_sha256=str(pointer.get("record_sha256", "")),
    )
    for field in ("campaign_id", "scientific_pair_id", "trajectory", "pulse", "generation",
                  "admission_identity_sha256", "implementation_identity_sha256"):
        if pointer.get(field) != record.get(field):
            raise ValueError(f"active output {field} differs from qualified attempt")
    return {**pointer, "output_dir": str(root / record["output_dir"]),
            "pointer_path": str(pointer_path), "pointer_sha256": sha256_file(pointer_path),
            "qualified_attempt": record}

"""CPU-Q3 private B/I two-pulse and independent NEXT1 reload executor.

The module composes the frozen J1 optical window, Q1 observer/comparator,
CPU-O1 B/I factory, and J3 HR-4 worker implementation. It does not implement
or alter scientific operators. Real execution is allocation-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


SCHEMA = "filament.cpu_q3.bi_two_pulse_reload.v1"
QUALIFICATION = "NOT_PRODUCTION_QUALIFIED"
CLASSIFICATION = "CPU_Q3_DIAGNOSTIC_ONLY"
IDENTITIES = {"B": "BASE_CPUFFT", "I": "BASE_CPUFFT_PLUS_IONIZATION"}
WINDOW_CASES = ("B_P1", "B_P2_CONT", "B_P2_RELOAD", "I_P1", "I_P2_CONT", "I_P2_RELOAD")
BRANCHES = ("B", "I")
SCREEN_IDS = tuple(f"screen-{index:02d}" for index in range(8))
FIELDS = ("delta_n", "vx", "vy")
EXPECTED_HR4_STEPS = 1000
EXPECTED_F_REP = 1000.0
EXPECTED_DT_HYDRO = 1e-6
WINDOW_TIMEOUT_SECONDS = 2700.0
FIRST_INTERVAL_SECONDS = 900.0
HYDRO_OUTER_LIMIT_SECONDS = 900.0
EXISTING_HR4_WORKER_LIMIT_SECONDS = 180.0
ALLOCATION_SECONDS = 7200.0
RESERVE_SECONDS = 900.0
MEMORY_BYTES = 320 * 1024**3
MEMORY_SAMPLE_SECONDS = 1.0
MEMORY_WARN_FRACTION = 0.80
MEMORY_ABORT_FRACTION = 0.90
MAX_STORAGE_BYTES = 16 * 1024**3
MAX_FILES = 50000
CONTRACT_PATH = Path(__file__).with_name("q3_comparison_contract.json")


class Q3Error(RuntimeError):
    pass


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_new_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())


def _write_atomic_new(path: Path, payload: bytes) -> None:
    """Publish one same-directory immutable file atomically, never replacing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    temp.unlink()


def _sha256_file(path: Path) -> str:
    from tools.hpc_ops import provenance_v2 as pv
    return pv.raw_sha256_file(Path(path))


def _contract_field_key(field: dict) -> tuple[str, str, str | None]:
    return (field.get("field_name"), field.get("scope", "science"),
            field.get("diagnostic_source"))


def load_static_contract(path: Path = CONTRACT_PATH) -> dict:
    """Load and fail-closed validate the pre-run, checked-in comparison matrix."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "filament.cpu_q3.comparison_contract.v1":
        raise Q3Error("unsupported frozen Q3 comparison contract")
    groups = value.get("groups")
    if not isinstance(groups, list) or len(groups) != 31:
        raise Q3Error("frozen Q3 contract must contain exactly 31 groups")
    by_id = {}
    for group in groups:
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not group_id or group_id in by_id:
            raise Q3Error("frozen Q3 contract has an empty or duplicate group id")
        if group.get("field_template") in {"first_boundary", "endpoint"}:
            template = value["source"].get(f"{group['field_template']}_expected_fields")
            expected_count = value["source"][f"{group['field_template']}_template_count"]
            if not isinstance(template, list) or len(template) != expected_count:
                raise Q3Error(f"frozen Q3 {group['field_template']} template count mismatch")
            group["expected_fields"] = template
        expected = group.get("expected_fields")
        if not isinstance(expected, list) or not expected:
            raise Q3Error(f"frozen Q3 group has no expected fields: {group_id}")
        keys = [_contract_field_key(field) for field in expected]
        if (len(keys) != len(set(keys)) or any(
                not isinstance(field.get("field_name"), str)
                or not field["field_name"]
                or field.get("scope") not in {"science", "diagnostics", "timing"}
                for field in expected)):
            raise Q3Error(f"frozen Q3 group fields are malformed or repeated: {group_id}")
        by_id[group_id] = group
    if len(by_id) != value.get("expected_group_count"):
        raise Q3Error("frozen Q3 comparison group count declaration mismatch")
    first = value["source"]["first_boundary_expected_fields"]
    endpoint = value["source"]["endpoint_expected_fields"]
    for name, rows in (("first_boundary", first), ("endpoint", endpoint)):
        reduced = sorted((row["field_name"], row["scope"], row.get("diagnostic_source"))
                         for row in rows)
        digest = hashlib.sha256(json.dumps(reduced, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        if digest != value["source"][f"{name}_template_sha256"]:
            raise Q3Error(f"frozen Q3 {name} field template hash mismatch")
    return {**value, "group_map": by_id, "raw_sha256": _sha256_file(path)}


def _expected_field_rows(group: dict) -> list[dict]:
    return list(group["expected_fields"])


def _require_raw_group(group: dict, rows: list[dict]) -> None:
    """Require exact, unique raw row coverage for one frozen group."""
    expected = [_contract_field_key(row) for row in _expected_field_rows(group)]
    observed = []
    for row in rows:
        if row.get("record_type") not in {"array_comparison", "exact_comparison"}:
            raise Q3Error("Q3 matrix accepts raw comparison records only")
        if (row.get("case_id") != group["case_id"]
                or row.get("identity") != group["implementation"]
                or row.get("implementation", group["implementation"]) != group["implementation"]
                or row.get("comparison_stage") != group["comparison_stage"]
                or row.get("reference_case_id") != group["reference_case_id"]):
            raise Q3Error(f"raw row belongs to the wrong Q3 group: {group['group_id']}")
        if not isinstance(row.get("metrics"), dict):
            raise Q3Error("Q3 raw comparison row has no metrics")
        observed.append(_contract_field_key(row))
    if len(observed) != len(set(observed)):
        raise Q3Error(f"Q3 raw comparison group contains duplicate fields: {group['group_id']}")
    if sorted(observed) != sorted(expected):
        raise Q3Error(f"Q3 raw comparison group does not match frozen fields: {group['group_id']}")


def _validate_sidecar(path: Path, group: dict, contract: dict) -> None:
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if (sidecar.get("schema") != "filament.cpu_q2.comparison_field_contract.v1"
            or sidecar.get("case_id") != group["case_id"]
            or sidecar.get("implementation") != group["implementation"]
            or sidecar.get("reference_case_id") != group["reference_case_id"]
            or sidecar.get("comparison_stage") != group["comparison_stage"]):
        raise Q3Error("optical comparison sidecar identity differs from frozen Q3 group")
    actual = [_contract_field_key(item) for item in sidecar.get("expected_fields", [])]
    expected = [_contract_field_key(item) for item in _expected_field_rows(group)]
    if len(actual) != len(set(actual)) or sorted(actual) != sorted(expected):
        raise Q3Error("runtime optical field sidecar differs from frozen Q3 template")


def build_q3_raw_once_gate(*, contract: dict, group_rows: dict[str, list[dict]],
                           identity_checks: dict[str, bool],
                           linux_environment_eligible: bool,
                           resource_budget_respected: bool,
                           matrix_status: str) -> dict:
    """Validate every expected raw group, then classify the combined raw rows once."""
    from tools.cpu_o1.numeric_gate import build_numeric_gate

    expected = set(contract["group_map"])
    if set(group_rows) != expected:
        missing, extra = sorted(expected - set(group_rows)), sorted(set(group_rows) - expected)
        raise Q3Error(f"Q3 raw matrix groups differ from frozen contract; missing={missing}; extra={extra}")
    rows = []
    counts = {}
    for group_id in sorted(expected):
        group = contract["group_map"][group_id]
        raw = group_rows[group_id]
        _require_raw_group(group, raw)
        rows.extend(raw)
        counts[group_id] = len(raw)
    checks = dict(identity_checks)
    checks.update({"comparison_contract_complete": True,
        "resource_budget_respected": bool(resource_budget_respected),
        "all_expected_raw_groups_complete": True})
    gate = build_numeric_gate(identity_checks=checks,
        linux_environment_eligible=bool(linux_environment_eligible), comparisons=rows)
    gate.update({"gate_scope": "REAL_Q3_BI_TWO_PULSE_RELOAD",
        "matrix_status": matrix_status, "frozen_contract_sha256": contract["raw_sha256"],
        "expected_group_count": len(expected), "group_row_counts": counts,
        "raw_comparison_row_count": len(rows),
        "classified_from_raw_metrics_once": True,
        "production_qualification": QUALIFICATION})
    return gate


def _validate_identity_and_hydro(input_root: Path, context: dict, hydro: dict) -> dict:
    """Bind f_rep to the sealed derived heat config and HR4 constants to HR4 source."""
    import numpy as np
    from KHz_filament import hr4
    from tools.hpc_ops import provenance_v2 as pv

    identity = hydro["identity"]
    identity_path = Path(input_root) / "N5_FORMAL_ADMISSION_IDENTITY.json"
    derived_path = Path(input_root) / "derived_n5_execution_config.json"
    identity_raw_sha = pv.raw_sha256_file(identity_path)
    derived_raw_sha = pv.raw_sha256_file(derived_path)
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    heat = derived.get("heat")
    effective = identity.get("effective_params")
    if not isinstance(heat, dict) or not isinstance(effective, dict):
        raise Q3Error("sealed derived heat or formal effective_params is missing")
    for name, expected in (("f_rep", EXPECTED_F_REP),):
        if float(heat.get(name, math.nan)) != expected:
            raise Q3Error(f"derived heat.{name} is not the approved sealed 1000 Hz value")
    if (float(identity.get("f_rep", math.nan)) != EXPECTED_F_REP
            or float(effective.get("f_rep", math.nan)) != EXPECTED_F_REP
            or float(identity.get("dt_hydro", math.nan)) != EXPECTED_DT_HYDRO
            or float(effective.get("dt_hydro", math.nan)) != EXPECTED_DT_HYDRO
            or float(hydro.get("dt_hydro", math.nan)) != EXPECTED_DT_HYDRO
            or int(hydro.get("n_steps", -1)) != EXPECTED_HR4_STEPS):
        raise Q3Error("sealed identity/effective schedule does not prove 1000 Hz, 1 us, 1000 steps")
    values = hr4.validate_hr4_parameters(
        chi=hydro["chi"], D_th=hr4.HR4_CHI, nu=hydro["nu"],
        gravity_x=hydro["gravity_x"], gravity_y=hydro["gravity_y"],
        x_min=hydro["x_min"], x_max=hydro["x_max"],
        y_min=hydro["y_min"], y_max=hydro["y_max"],
        dx=hydro["dx"], dy=hydro["dy"], dt_hydro=hydro["dt_hydro"],
        advection_scheme=hr4.HR4_ALLOWED_ADVECTION,
        diffusion_scheme=hr4.HR4_ALLOWED_DIFFUSION,
        time_integrator=hr4.HR4_ALLOWED_INTEGRATOR,
        grid_layout=hr4.HR4_ALLOWED_GRID_LAYOUT,
        boundary_delta_n=hr4.HR4_ALLOWED_DELTA_N_BOUNDARY,
        boundary_velocity=hr4.HR4_ALLOWED_VELOCITY_BOUNDARY)
    hr4_source = (Path(context["bench_root"]) if context.get("bench_root")
                  else None)
    if hr4_source is None:
        # Context from the frozen loader has no root field; derive the active source
        # from this module's repository root, then bind its committed blob via Git.
        repo_root = Path(__file__).resolve().parents[3]
    else:
        repo_root = hr4_source
    source_rel = "Filament_python/KHz_filament/hr4.py"
    source_blob = pv._git_blob_oid(repo_root, source_rel)
    source_bytes = pv._read_tracked(repo_root, source_rel)
    source_lf_sha = pv.canonical_lf_sha256(source_bytes)
    return {"f_rep_hz": EXPECTED_F_REP, "dt_hydro_s": EXPECTED_DT_HYDRO,
        "n_steps": EXPECTED_HR4_STEPS, "effective_hr4_parameters": values,
        "parameter_sources": {"f_rep": "derived_n5_execution_config.json:heat.f_rep and formal identity/effective_params",
            "dt_hydro": "N5_FORMAL_ADMISSION_IDENTITY.json top-level/effective_params",
            "chi_and_D_th": "frozen KHz_filament.hr4.HR4_CHI",
            "nu": "frozen KHz_filament.hr4.HR4_NU",
            "grid_and_n0": "verified sealed optical context",
            "remaining_hr4_constants": "frozen KHz_filament.hr4 constants"},
        "identity_path": str(identity_path), "identity_raw_sha256": identity_raw_sha,
        "identity_sha256": identity.get("identity_sha256"),
        "derived_config_path": str(derived_path), "derived_config_raw_sha256": derived_raw_sha,
        "hr4_source_path": source_rel, "hr4_source_git_blob_oid": source_blob,
        "hr4_source_canonical_lf_sha256": source_lf_sha,
        "heat_D_gas_compared_to_chi": False,
        "production_qualification": QUALIFICATION}


def _validate_next1_fields(fields: dict, *, shape=(8, 351, 301)) -> dict:
    from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    return j3._validate_fields(fields, shape=shape, label="Q3 NEXT1")


def _publish_checkpoint(path: Path, fields: dict, *, metadata: dict,
                        shape=(8, 351, 301), j3=None) -> dict:
    """Validate a same-directory temporary NPZ before atomic immutable publication."""
    import numpy as np
    if j3 is None:
        from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    values = _validate_next1_fields(fields, shape=shape)
    hashes = j3._field_hashes(values)
    payload = j3._npz_bytes(values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        raw_sha = j3.raw_sha256_file(temp)
        disk = j3._load_npz(temp, expected_raw_sha256=raw_sha,
            expected_field_sha256=hashes, shape=shape,
            label="Q3 temporary checkpoint before publication")
        if any(not np.array_equal(values[name].view(np.uint8), disk[name].view(np.uint8))
               for name in FIELDS):
            raise Q3Error("Q3 in-memory NEXT1 differs from validated temporary checkpoint")
        os.link(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    temp.unlink()
    disk = j3._load_npz(path, expected_raw_sha256=raw_sha,
        expected_field_sha256=hashes, shape=shape, label="Q3 published checkpoint")
    if any(not np.array_equal(values[name].view(np.uint8), disk[name].view(np.uint8))
           for name in FIELDS):
        raise Q3Error("Q3 in-memory checkpoint differs from atomically reloaded bytes")
    receipt = {"schema": SCHEMA, "status": "CHECKPOINT_VALIDATED",
        "checkpoint_kind": metadata.get("checkpoint_kind"),
        "branch": metadata.get("branch"), "completed_pulse": 1,
        "next_pulse": 2, "screen_ids": list(SCREEN_IDS),
        "field_shape": list(shape), "field_dtype": "float64",
        "raw_sha256": raw_sha, "field_canonical_sha256": hashes,
        "bytes": path.stat().st_size, "metadata": metadata,
        "memory_equals_disk_exact": True,
        "write_and_validation_seconds": time.perf_counter() - write_started,
        "production_qualification": QUALIFICATION}
    receipt_path = path.with_name(path.stem + ".receipt.json")
    marker_path = path.with_name(path.stem + ".COMPLETE.json")
    _write_atomic_new(receipt_path, _json_bytes(receipt))
    _write_atomic_new(marker_path, _json_bytes({"schema": SCHEMA,
        "status": "RECOVERABLE_COMPLETE", "checkpoint_raw_sha256": raw_sha,
        "receipt_sha256": _sha256_file(receipt_path)}))
    return {"path": path, "receipt_path": receipt_path, "marker_path": marker_path,
        "raw_sha256": raw_sha, "field_sha256": hashes, "fields": values,
        "reloaded_fields": disk, "receipt": receipt}


def _load_checkpoint(path: Path, *, expected_branch: str,
                     expected_shape=(8, 351, 301),
                     expected_metadata: dict | None = None, j3=None) -> tuple[dict, dict]:
    if j3 is None:
        from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    path = Path(path)
    receipt_path = path.with_name(path.stem + ".receipt.json")
    marker_path = path.with_name(path.stem + ".COMPLETE.json")
    if not receipt_path.is_file() or not marker_path.is_file():
        raise Q3Error("Q3 NEXT1 checkpoint is incomplete or not atomically finalized")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "CHECKPOINT_VALIDATED"
            or receipt.get("branch") != expected_branch
            or receipt.get("checkpoint_kind") != "NEXT1"
            or receipt.get("completed_pulse") != 1 or receipt.get("next_pulse") != 2
            or receipt.get("screen_ids") != list(SCREEN_IDS)
            or receipt.get("field_dtype") != "float64"
            or receipt.get("field_shape") != list(expected_shape)
            or (expected_metadata is not None and receipt.get("metadata") != expected_metadata)
            or marker.get("status") != "RECOVERABLE_COMPLETE"
            or marker.get("checkpoint_raw_sha256") != receipt.get("raw_sha256")
            or marker.get("receipt_sha256") != _sha256_file(receipt_path)):
        raise Q3Error("Q3 checkpoint branch/lineage/complete-marker identity mismatch")
    fields = j3._load_npz(path, expected_raw_sha256=receipt["raw_sha256"],
        expected_field_sha256=receipt["field_canonical_sha256"],
        shape=tuple(expected_shape), label="Q3 disk NEXT1")
    return fields, receipt


def _require_pre_lineage(pre_fields: dict, *, case_id: str,
                         expected_hashes: dict | None, initial_pre0: bool,
                         branch: str, expected_shape=(8, 351, 301)) -> dict:
    from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    if branch not in BRANCHES or not case_id.startswith(branch + "_"):
        raise Q3Error("PRE branch/case identity mismatch")
    values = j3._validate_fields(pre_fields, shape=expected_shape, label=f"{case_id} PRE")
    actual = j3._field_hashes(values)
    if case_id.endswith("P1"):
        if not initial_pre0 or branch not in {"B", "I"} or expected_hashes is None:
            raise Q3Error("P1 must use the sealed PRE0 full three-field state")
        if actual != expected_hashes:
            raise Q3Error("P1 PRE0 three-field identity mismatch")
    else:
        if initial_pre0 or expected_hashes is None or actual != expected_hashes:
            raise Q3Error("P2 input is not the full branch-local NEXT1 state")
    return actual


def _validate_screen_ordinals(rows: list[dict]) -> list[dict]:
    if len(rows) != 8:
        raise Q3Error("HR4 branch must provide exactly eight worker receipts")
    ordinals = [row.get("ordinal") for row in rows]
    if ordinals != list(range(8)) or len(set(ordinals)) != 8:
        raise Q3Error("HR4 screen ordinals are missing, duplicated, or out of schedule order")
    return rows


def _metadata_row(*, field_name: str, scope: str, diagnostic_source: str,
                  reference, candidate, identity: str, case_id: str,
                  reference_case_id: str, comparison_stage: str) -> dict:
    """Compare strict JSON metadata through the existing bounded array comparator."""
    import numpy as np
    from tools.cpu_o1 import q1_t0

    left = _json_bytes(reference)
    right = _json_bytes(candidate)
    row = q1_t0._array_comparison_row(field_name,
        np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8),
        scope=scope)
    row.update({"schema": "filament.cpu_q1.raw_comparison.v1",
        "record_type": "array_comparison", "identity": identity,
        "implementation": identity, "case_id": case_id,
        "reference_case_id": reference_case_id,
        "comparison_stage": comparison_stage,
        "diagnostic_source": diagnostic_source,
        "metadata_encoding": "strict_sorted_json_utf8",
        "reference_json_type": type(reference).__name__,
        "candidate_json_type": type(candidate).__name__,
        "reference_json_value": reference,
        "candidate_json_value": candidate})
    return row


def _array_row(*, field_name: str, reference, candidate, diagnostic_source: str,
               identity: str, case_id: str, reference_case_id: str,
               comparison_stage: str, zero_mask_path: Path | None = None) -> dict:
    from tools.cpu_o1 import q1_t0
    row = q1_t0._array_comparison_row(field_name, reference, candidate,
        scope="science", zero_mask_path=zero_mask_path)
    row.update({"schema": "filament.cpu_q1.raw_comparison.v1",
        "record_type": "array_comparison", "identity": identity,
        "implementation": identity, "case_id": case_id,
        "reference_case_id": reference_case_id,
        "comparison_stage": comparison_stage,
        "diagnostic_source": diagnostic_source})
    return row


def _write_group_rows(path: Path, *, group: dict, rows: list[dict]) -> None:
    from tools.cpu_o1.q1_t0_real import _write_comparison_field_contract
    _write_comparison_field_contract(path, case_id=group["case_id"],
        implementation=group["implementation"],
        reference_case_id=group["reference_case_id"],
        comparison_stage=group["comparison_stage"],
        expected_fields=_expected_field_rows(group))
    _append_jsonl(path, rows)
    _require_raw_group(group, rows)


def _read_raw_comparisons(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type") == "comparison_error":
            raise Q3Error(f"raw Q3 comparison error at {path}:{line_number}")
        if row.get("record_type") not in {"array_comparison", "exact_comparison"}:
            raise Q3Error(f"unexpected Q3 comparison record at {path}:{line_number}")
        rows.append(row)
    return rows


def _group_path(case_root: Path, group: dict) -> Path:
    stage = group["comparison_stage"]
    ref = group["reference_case_id"]
    case = group["case_id"]
    if stage == "first_boundary":
        root = case_root / case / "first_boundary"
        return root / ("self_check.jsonl" if ref == case
                       else f"comparison_to_{ref}.jsonl")
    if stage == "endpoint":
        root = case_root / case
        return root / ("endpoint_self_check.jsonl" if ref == case
                       else f"endpoint_comparison_to_{ref}.jsonl")
    return case_root / case / f"{stage}_to_{ref}.jsonl"


def _write_phase(path: Path, value: dict) -> None:
    """Atomically replace the Q3 supervisor's mutable phase pointer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def _run_optical_window(*, args, context: dict, identity: str, case_id: str,
                        pre_fields: dict, initial_pre0: bool,
                        expected_pre_hashes: dict, output_root: Path,
                        phase_path: Path, comparison_references=(),
                        contract: dict, group_rows: dict[str, list[dict]],
                        capture_pre_for_comparison: bool = False) -> dict:
    """Execute one real eight-interval Q3 case through the frozen J1/Q1 seam."""
    service_started = time.perf_counter()
    prior = []
    for receipt_path in Path(output_root).glob("*/q3_case_receipt.json"):
        prior.append(json.loads(receipt_path.read_text())["timing"]["service_wall_seconds_including_checks"])
    predicted = min(WINDOW_TIMEOUT_SECONDS, max(300.0, prior[-1] * 1.5)) if prior else WINDOW_TIMEOUT_SECONDS
    if time.time() + predicted > float(os.environ["Q3_WORK_DEADLINE_EPOCH"]):
        raise Q3Error("INSUFFICIENT_REMAINING_TIME_FOR_OPTICAL_STAGE")
    import numpy as np
    from importlib import import_module
    from tools.cpu_p1r import cpu_p1r_optical as p1r
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    from tools.cpu_o1 import q1_t0
    from tools.cpu_o1 import q1_t0_real as q1real
    from tools.cpu_o1.cpu_o1 import make_joint_propagator_factory

    case_root = Path(output_root) / case_id
    case_root.mkdir(parents=True, exist_ok=False)
    from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    source_validation_started = time.perf_counter()
    validated_source, validated_source_hash, source_validation = (
        p1r.validate_frozen_source(args.input_root, context["assets"]))
    source_validation_seconds = time.perf_counter() - source_validation_started
    if validated_source_hash != context["source_hash"]:
        raise Q3Error(f"{case_id} revalidated E_source differs from the loaded sealed source")
    source_file_raw_sha256 = source_validation.get("file_sha256")
    if source_file_raw_sha256 != p1r.SOURCE_RAW_SHA256:
        raise Q3Error(f"{case_id} revalidated E_source raw file identity mismatch")
    pre_validation_started = time.perf_counter()
    actual_pre = j3._validate_fields(pre_fields,
        shape=(8, int(context["grid"].Ny), int(context["grid"].Nx)),
        label=f"{case_id} actual scientific PRE")
    actual_pre_hashes = j3._field_hashes(actual_pre)
    if actual_pre_hashes != expected_pre_hashes:
        raise Q3Error(f"{case_id} PRE fields differ from the frozen branch lineage")
    _require_pre_lineage(actual_pre, case_id=case_id,
        expected_hashes=expected_pre_hashes, initial_pre0=initial_pre0,
        branch=identity, expected_shape=tuple(actual_pre["delta_n"].shape))
    pre_validation_seconds = time.perf_counter() - pre_validation_started

    source = np.array(validated_source, dtype=np.complex128, order="C", copy=True)
    if np.shares_memory(source, context["source"]):
        raise Q3Error("fresh E_source copy unexpectedly aliases the loaded frozen source")
    pre_entry_snapshot = ({name: np.array(actual_pre[name], order="C", copy=True)
                          for name in FIELDS}
                         if capture_pre_for_comparison else None)
    source_hash = p1r.verify_array_contract(source,
        shape=tuple(context["source"].shape), dtype="complex128",
        expected_hash=context["source_hash"], label=f"{case_id} fresh E_source")
    phase = {"schema": SCHEMA, "phase": "OPTICAL", "stage_id": case_id,
        "case_id": case_id,
        "implementation": identity,
        "progress_path": str(case_root / "propagation" / "observer_progress.jsonl"),
        "phase_started_utc": time.time(), "phase_started_monotonic": time.monotonic(),
        "timeout_seconds": WINDOW_TIMEOUT_SECONDS,
        "first_interval_timeout_seconds": FIRST_INTERVAL_SECONDS,
        "z_edges_m": [float(x) for x in context["schedule"].z_edges],
        "dz_intervals_m": [float(x) for x in context["schedule"].dz_intervals]}
    _write_phase(phase_path, phase)

    factory_started = time.perf_counter()
    refs = [(str(ref_id), Path(ref_root).resolve(strict=True))
            for ref_id, ref_root in comparison_references]
    diagnostic_contract = q1real._frozen_diag_list_contract(
        Path(import_module("KHz_filament.propagate").__file__).resolve(strict=True))
    observer_type = q1real._observer_class(j1, identity=identity,
        case_id=case_id, case_root=case_root,
        comparison_boundary_roots=[(ref_id, root / "first_boundary")
                                   for ref_id, root in refs],
        diagnostic_contract=diagnostic_contract)
    private_window = q1real._private_run_window_with_observer(j1, observer_type)
    factory = make_joint_propagator_factory(identity, kernel_threads=24,
        block_pixels=32, layout="NtB", temporal_workers=48, spatial_workers=24)
    capture = {}
    propagator_factory = q1_t0._scalar_capture_factory(factory, capture)
    factory_initialization_seconds = time.perf_counter() - factory_started
    candidate_receipt = None
    propagation_started = time.perf_counter()
    try:
        window_receipt = private_window(
            **{key: context[key] for key in (
                "grid", "beam", "prop", "ion", "heat", "raman", "schedule")},
            source=source, pre_fields=actual_pre,
            output_dir=case_root / "propagation",
            source_label=("Q3_sealed_PRE0" if initial_pre0
                          else f"Q3_{identity}_NEXT1_for_P2"),
            initial_pre0=initial_pre0, profile=False,
            source_hash=source_hash,
            pre0_hash=(context["assets"]["pre0_canonical_sha256"]
                       if initial_pre0 else None),
            stage_name=f"Q3_{case_id}", execution_scope="J3_REAL_INPUT",
            observe=True, propagator_factory=propagator_factory)
        propagation_seconds = time.perf_counter() - propagation_started
        observer = observer_type.last_instance
        if (observer is None or observer.q1_first_boundary is None
                or observer.q1_first_gate is None
                or observer.q1_first_gate.get("proceed_to_next_identity") is not True):
            raise Q3Error(f"{case_id} first-boundary exact gate did not pass")
        if not {"qacc", "payload", "values"}.issubset(capture):
            raise Q3Error(f"{case_id} propagator omitted its frozen scalar/Qacc return")
        q1_t0._write_new_array(case_root / "scientific_Qacc.npy", capture["qacc"])
        q1_t0._write_scalar_diagnostics(case_root / "scientific_scalar_diagnostics.json",
                                        identity, capture["payload"])
        dispatch = q1_t0._check_identity_dispatch(identity, factory,
            kernel_threads=24, temporal_workers=48, spatial_workers=24)
        if identity == "I":
            if ("ionization" not in dispatch.get("candidate_event_kinds", [])
                    or dispatch["execution"].get("kernel_threads") != 24):
                raise Q3Error(f"{case_id} did not prove the fixed I candidate dispatch")
        elif dispatch.get("candidate_event_count", 0) != 0:
            raise Q3Error(f"{case_id} baseline unexpectedly ran a compiled candidate")
        _, endpoint_gate = q1real._stage_endpoint_comparison(identity, None,
            case_root, case_id=case_id,
            comparison_roots=[(ref_id, root) for ref_id, root in refs])
        endpoint_cross_seconds = endpoint_gate.get(
            "cross_identity_comparison_seconds")
        if (not isinstance(endpoint_cross_seconds, (int, float))
                or not math.isfinite(float(endpoint_cross_seconds))
                or float(endpoint_cross_seconds) < 0.0):
            raise Q3Error(f"{case_id} endpoint cross-comparison timing is missing or invalid")
        if endpoint_gate.get("proceed_to_next_identity") is not True:
            raise Q3Error(f"{case_id} endpoint exact gate failed")
        first_cross_seconds = observer.q1_first_boundary.get(
            "cross_identity_comparison_seconds")
        if (not isinstance(first_cross_seconds, (int, float))
                or not math.isfinite(float(first_cross_seconds))
                or float(first_cross_seconds) < 0.0):
            raise Q3Error(f"{case_id} first-boundary cross-comparison timing is missing or invalid")
        candidate_events = list(getattr(factory.last_propagator,
            "_cpu_o1_candidate_events", ()))
        candidate_receipt = q1real._persist_candidate_events(
            case_root / "candidate_events.jsonl", case_id=case_id,
            implementation=identity, events=candidate_events, json_safe=j1._json_safe)
        if candidate_receipt.get("status") != "WRITTEN":
            raise Q3Error(f"{case_id} candidate event JSONL was not durably written")
        _validate_stage_groups(output_root=output_root, case_id=case_id,
            stages=("first_boundary", "endpoint"), contract=contract,
            group_rows=group_rows)
        service_seconds = time.perf_counter() - service_started
        cross_comparison_seconds = (float(first_cross_seconds)
                                    + float(endpoint_cross_seconds))
        if cross_comparison_seconds > service_seconds:
            raise Q3Error(f"{case_id} cross-comparison time exceeds inclusive service wall")
        result = {"schema": SCHEMA, "case_id": case_id,
            "identity": identity, "status": "COMPLETE_DIAGNOSTIC",
            "fresh_source": {"fresh_copy": True,
                "source_file_raw_sha256": source_file_raw_sha256,
                "source_canonical_sha256": source_hash,
                "source_validation": source_validation,
                "source_validation_seconds": source_validation_seconds,
                "shape": list(source.shape), "dtype": str(source.dtype),
                "c_contiguous": bool(source.flags.c_contiguous),
                "aliases_loaded_source": False},
            "pre_validation_seconds": pre_validation_seconds,
            "pre_fields_canonical_sha256_at_entry": actual_pre_hashes,
            "actual_entry_pre_fields_canonical_sha256": window_receipt[
                "input_PRE_fields_canonical_sha256"],
            "initial_pre0": bool(initial_pre0),
            "propagation_receipt": window_receipt, "dispatch": dispatch,
            "candidate_events": candidate_receipt,
            "first_boundary_gate": observer.q1_first_gate,
            "endpoint_gate": endpoint_gate,
            "timing": {"service_wall_seconds_including_checks": service_seconds,
                "service_wall_non_cross_comparison_seconds": (
                    service_seconds - cross_comparison_seconds),
                "service_wall_formula": (
                    "service_wall_non_cross_comparison_seconds + "
                    "first_boundary_cross_identity_comparison_seconds + "
                    "endpoint_cross_identity_comparison_seconds"),
                "factory_initialization_seconds": factory_initialization_seconds,
                "propagation_wall_seconds_including_first_boundary_compare": propagation_seconds,
                "propagation_wall_seconds": window_receipt["propagation_wall_seconds"],
                "output_validation_seconds": window_receipt["validation_output_seconds"],
                "first_boundary_comparison_seconds": float(first_cross_seconds),
                "endpoint_comparison_seconds": float(endpoint_cross_seconds),
                "cross_identity_comparison_total_seconds": cross_comparison_seconds}}
        if pre_entry_snapshot is not None:
            if j3._field_hashes(pre_entry_snapshot) != result["actual_entry_pre_fields_canonical_sha256"]:
                raise Q3Error(f"{case_id} PRE2 snapshot differs from the actual propagation entry")
            result["_actual_pre_fields_for_comparison"] = pre_entry_snapshot
        _write_new_json(case_root / "q3_case_receipt.json",
            {key: value for key, value in result.items()
             if not key.startswith("_")})
        return result
    finally:
        if candidate_receipt is None:
            events = list(getattr(getattr(factory, "last_propagator", None),
                "_cpu_o1_candidate_events", ()))
            try:
                candidate_receipt = q1real._persist_candidate_events(
                    case_root / "candidate_events.jsonl", case_id=case_id,
                    implementation=identity, events=events, json_safe=j1._json_safe)
            except BaseException as exc:
                candidate_receipt = {"status": "WRITE_FAILED",
                    "error": f"{type(exc).__name__}: {exc}"}
            try:
                _write_new_json(case_root / "candidate_event_finally_receipt.json",
                                candidate_receipt)
            except BaseException:
                pass


def _record_raw_group(*, output_root: Path, group_id: str,
                      contract: dict, group_rows: dict[str, list[dict]]) -> dict:
    group = contract["group_map"].get(group_id)
    if group is None:
        raise Q3Error(f"Q3 execution attempted an unplanned comparison group: {group_id}")
    path = _group_path(output_root, group)
    sidecar_path = path.with_name(path.name + ".contract.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise Q3Error(f"Q3 raw comparison or field contract is missing: {group_id}")
    _validate_sidecar(sidecar_path, group, contract)
    rows = _read_raw_comparisons(path)
    _require_raw_group(group, rows)
    from tools.cpu_o1.numeric_gate import build_numeric_gate
    gate = build_numeric_gate(identity_checks={"static_field_contract_complete": True},
        linux_environment_eligible=True, comparisons=rows)
    if gate.get("proceed_to_next_identity") is not True:
        raise Q3Error(f"Q3 numerical stage gate failed for {group_id}: {gate['classification']}")
    group_rows[group_id] = rows
    return {"group_id": group_id, "row_count": len(rows),
        "classification": gate["classification"], "raw_path": str(path),
        "field_contract_path": str(sidecar_path)}


def _validate_stage_groups(*, output_root: Path, case_id: str,
                           stages: tuple[str, ...], contract: dict,
                           group_rows: dict[str, list[dict]]) -> list[dict]:
    results = []
    for stage in stages:
        group_ids = sorted(group_id for group_id, group in contract["group_map"].items()
            if group.get("case_id") == case_id
            and group.get("comparison_stage") == stage)
        if not group_ids:
            raise Q3Error(f"Q3 frozen contract has no {stage} group for {case_id}")
        results.extend(_record_raw_group(output_root=output_root,
            group_id=group_id, contract=contract, group_rows=group_rows)
            for group_id in group_ids)
    return results


def _write_static_group(*, output_root: Path, group_id: str, contract: dict,
                        group_rows: dict[str, list[dict]], left_values: dict,
                        right_values: dict, array_prefix: str | None = None,
                        left_metadata: dict | None = None,
                        right_metadata: dict | None = None) -> dict:
    """Write exactly the pre-frozen field set for a Q3 hydro/lineage comparison."""
    group = contract["group_map"].get(group_id)
    if group is None:
        raise Q3Error(f"Q3 attempted an unplanned static comparison group: {group_id}")
    rows = []
    for spec in _expected_field_rows(group):
        name = spec["field_name"]
        if array_prefix is not None:
            if not name.startswith(array_prefix):
                raise Q3Error(f"unexpected field in array comparison contract: {name}")
            field = name[len(array_prefix):]
            left, right = left_values[field], right_values[field]
            row = _array_row(field_name=name, reference=left, candidate=right,
                diagnostic_source=spec.get("diagnostic_source", "NEXT1_checkpoint_array"),
                identity=group["implementation"], case_id=group["case_id"],
                reference_case_id=group["reference_case_id"],
                comparison_stage=group["comparison_stage"])
        else:
            key_prefix = "NEXT1_metadata." if name.startswith("NEXT1_metadata.") else ""
            if key_prefix:
                key = name[len(key_prefix):]
            elif name.startswith("NEXT1."):
                key = name[len("NEXT1."):]
            elif name.startswith("PRE2."):
                key = name[len("PRE2."):]
            else:
                raise Q3Error(f"unmapped metadata field in frozen Q3 contract: {name}")
            if left_metadata is None or right_metadata is None or key not in left_metadata or key not in right_metadata:
                raise Q3Error(f"required Q3 metadata value is absent: {name}")
            row = _metadata_row(field_name=name, scope=spec["scope"],
                diagnostic_source=spec.get("diagnostic_source", "NEXT1_checkpoint_science_metadata"),
                reference=left_metadata[key], candidate=right_metadata[key],
                identity=group["implementation"], case_id=group["case_id"],
                reference_case_id=group["reference_case_id"],
                comparison_stage=group["comparison_stage"])
        row["scope"] = spec["scope"]
        if spec.get("diagnostic_source") is not None:
            row["diagnostic_source"] = spec["diagnostic_source"]
        rows.append(row)
    path = _group_path(output_root, group)
    _write_group_rows(path, group=group, rows=rows)
    return _record_raw_group(output_root=output_root, group_id=group_id,
                             contract=contract, group_rows=group_rows)


def _schedule_metadata(context: dict) -> dict:
    import numpy as np
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    schedule = context["schedule"]
    z_edges = np.asarray(schedule.z_edges, dtype=np.float64)
    dz = np.asarray(schedule.dz_intervals, dtype=np.float64)
    metadata = j1._json_safe(schedule.as_metadata())
    return {"z_edges_m": [float(value) for value in z_edges],
        "dz_intervals_m": [float(value) for value in dz],
        "schedule_metadata": metadata,
        "schedule_identity_sha256": hashlib.sha256(_json_bytes({
            "z_edges_m": [float(value) for value in z_edges],
            "dz_intervals_m": [float(value) for value in dz],
            "schedule_metadata": metadata})).hexdigest()}


def _checkpoint_metadata(*, identity: str, context: dict, hydro: dict,
                         post_hashes: dict, hr4_identity: dict) -> dict:
    schedule = _schedule_metadata(context)
    return {"checkpoint_kind": "NEXT1", "branch": identity,
        "screen_ids": list(SCREEN_IDS),
        "screen_z_order": [{"screen_id": SCREEN_IDS[index], "ordinal": index,
            "z_start_m": schedule["z_edges_m"][index],
            "z_end_m": schedule["z_edges_m"][index + 1],
            "dz_m": schedule["dz_intervals_m"][index]} for index in range(8)],
        "z_edges_m": schedule["z_edges_m"],
        "dz_intervals_m": schedule["dz_intervals_m"],
        "schedule_metadata": schedule["schedule_metadata"],
        "schedule_identity_sha256": schedule["schedule_identity_sha256"],
        "grid_shape": [int(context["grid"].Nt), int(context["grid"].Ny),
                       int(context["grid"].Nx)],
        "field_shape": [8, int(context["grid"].Ny), int(context["grid"].Nx)],
        "field_dtype": "float64",
        "field_units": {"delta_n": "dimensionless", "vx": "m s-1", "vy": "m s-1"},
        "f_rep_hz": float(hydro["f_rep"]),
        "dt_hydro_s": float(hydro["dt_hydro"]), "n_steps": int(hydro["n_steps"]),
        "effective_hr4_parameters": hr4_identity["effective_hr4_parameters"],
        "source_canonical_sha256": context["source_hash"],
        "pre0_canonical_sha256": context["assets"]["pre0_canonical_sha256"],
        "identity_raw_sha256": hr4_identity["identity_raw_sha256"],
        "identity_sha256": hr4_identity["identity_sha256"],
        "derived_config_raw_sha256": hr4_identity["derived_config_raw_sha256"],
        "hr4_source_canonical_lf_sha256": hr4_identity["hr4_source_canonical_lf_sha256"],
        "post_field_canonical_sha256": post_hashes,
        "production_qualification": QUALIFICATION}


def _checkpoint_metadata_values(metadata: dict) -> dict:
    return {"screen_ids": metadata["screen_ids"],
        "z_edges_m": metadata["z_edges_m"], "dz_intervals_m": metadata["dz_intervals_m"],
        "grid_shape": metadata["grid_shape"], "field_shape": metadata["field_shape"],
        "field_dtype": metadata["field_dtype"], "field_units": metadata["field_units"],
        "f_rep_hz": metadata["f_rep_hz"], "dt_hydro_s": metadata["dt_hydro_s"],
        "n_steps": metadata["n_steps"],
        "effective_hr4_parameters": metadata["effective_hr4_parameters"],
        "source_canonical_sha256": metadata["source_canonical_sha256"],
        "pre0_canonical_sha256": metadata["pre0_canonical_sha256"],
        "schedule_identity_sha256": metadata["schedule_identity_sha256"]}


def _activity_summary(before: dict, after: dict) -> dict:
    import numpy as np
    result = {}
    for name in FIELDS:
        left, right = np.asarray(before[name]), np.asarray(after[name])
        difference = np.abs(right - left)
        if not np.all(np.isfinite(difference)):
            raise Q3Error(f"HR4 activity difference for {name} is not finite")
        result[name] = {"max_abs_change": float(np.max(difference)),
            "nonzero_changed_cells": int(np.count_nonzero(difference))}
    return result


def _require_finite_json(value, label: str) -> None:
    if value is None:
        raise Q3Error(f"{label} is null")
    if isinstance(value, bool) or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise Q3Error(f"{label} is non-finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, f"{label}.{key}")
        return
    raise Q3Error(f"{label} has unsupported JSON type {type(value).__name__}")


def _hr4_observable_values(receipt: dict) -> dict:
    result = {}
    for path in ("screen_observables", "stability"):
        value = receipt.get(path)
        if not isinstance(value, dict):
            raise Q3Error(f"HR4 receipt is missing {path}")
        _require_finite_json(value, f"HR4.{path}")
        for key, item in value.items():
            result[f"{path}.{key}"] = item
    result["n_steps"] = receipt.get("n_steps")
    _require_finite_json(result["n_steps"], "HR4.n_steps")
    return result


def _hr4_metadata_values(*, checkpoint_metadata: dict, hydro: dict,
                         hr4_identity: dict) -> dict:
    screens = checkpoint_metadata.get("screen_z_order")
    post_hashes = checkpoint_metadata.get("post_field_canonical_sha256")
    if not isinstance(screens, list) or len(screens) != 8 or not isinstance(post_hashes, dict):
        raise Q3Error("HR4 metadata lacks eight screen identities or POST hashes")
    result = {}
    for ordinal, item in enumerate(screens):
        if item.get("ordinal") != ordinal or item.get("screen_id") != SCREEN_IDS[ordinal]:
            raise Q3Error("HR4 metadata screen ordering differs from the fixed eight-screen contract")
        prefix = f"{SCREEN_IDS[ordinal]}"
        for name in ("ordinal", "z_start_m", "z_end_m", "dz_m"):
            result[f"{prefix}.{name}"] = item[name]
        for name, field in (("post_delta_n_sha256", "delta_n"),
                            ("post_vx_sha256", "vx"), ("post_vy_sha256", "vy")):
            result[f"{prefix}.{name}"] = post_hashes[field]
    parameter_values = {
        "f_rep_hz": hydro["f_rep"], "dt_hydro_s": hydro["dt_hydro"],
        "n_steps": hydro["n_steps"], "dx_m": hydro["dx"], "dy_m": hydro["dy"],
        "n0_dimensionless": hydro["n0"], "chi_m2_s": hydro["chi"],
        "D_th_m2_s": hydro["chi"], "nu_m2_s": hydro["nu"],
        "gravity_x_m_s2": hydro["gravity_x"], "gravity_y_m_s2": hydro["gravity_y"],
        "cfl_limit": hydro["cfl_limit"], "x_min_m": hydro["x_min"],
        "x_max_m": hydro["x_max"], "y_min_m": hydro["y_min"],
        "y_max_m": hydro["y_max"],
        "advection_scheme": hydro["boundaries"]["advection"],
        "diffusion_scheme": hydro["boundaries"]["diffusion"],
        "time_integrator": hydro["boundaries"]["integrator"],
        "grid_layout": hydro["boundaries"]["grid_layout"],
        "boundary_delta_n": hydro["boundaries"]["delta_n"],
        "boundary_velocity": hydro["boundaries"]["velocity"],
        "identity_sha256": hr4_identity["identity_sha256"],
        "derived_config_sha256": hr4_identity["derived_config_raw_sha256"],
        "hr4_source_sha256": hr4_identity["hr4_source_canonical_lf_sha256"]}
    for key, value in parameter_values.items():
        result[f"hydro_parameters.{key}"] = value
    for key, value in result.items():
        _require_finite_json(value, f"HR4 metadata {key}")
    return result


def _write_hr4_cross_groups(*, output_root: Path, contract: dict,
                            group_rows: dict[str, list[dict]],
                            b_chain: dict, i_chain: dict) -> list[dict]:
    import numpy as np
    from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    branches = {"B": b_chain, "I": i_chain}
    arrays, observables, metadata = {}, {}, {}
    for branch, receipt in branches.items():
        chain = receipt
        hydro_receipt = chain["hr4"]
        workers = _validate_screen_ordinals(hydro_receipt.get("worker_rows", []))
        arrays[branch] = j3._load_npz(Path(chain["checkpoint_path"]),
            expected_raw_sha256=chain["next1"]["raw_sha256"],
            expected_field_sha256=chain["next1"]["field_canonical_sha256"],
            shape=(8, 351, 301), label=f"Q3 {branch} NEXT1 cross comparison")
        observables[branch] = [_hr4_observable_values(row["receipt"]) for row in workers]
        checkpoint_path = Path(chain["checkpoint_path"])
        receipt_path = checkpoint_path.with_name(checkpoint_path.stem + ".receipt.json")
        # The receipt is read from disk below; do not reuse the writer's in-memory object.
        disk_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (disk_receipt.get("status") != "CHECKPOINT_VALIDATED"
                or disk_receipt.get("raw_sha256") != chain["next1"]["raw_sha256"]
                or disk_receipt.get("field_canonical_sha256") != chain["next1"]["field_canonical_sha256"]):
            raise Q3Error(f"{branch} NEXT1 disk receipt identity changed")
        checkpoint_metadata = disk_receipt.get("metadata")
        metadata[branch] = _hr4_metadata_values(checkpoint_metadata=checkpoint_metadata,
            hydro=chain["hydro_parameters"], hr4_identity=chain["hr4_identity"])
    group_ids = ("I_H1:NEXT1_fields:B_H1", "I_H1:NEXT1_observables:B_H1",
                 "I_H1:NEXT1_metadata:B_H1")
    written = []
    for group_id in group_ids:
        group = contract["group_map"].get(group_id)
        if group is None:
            raise Q3Error(f"frozen Q3 contract lacks required HR4 cross group {group_id}")
        rows = []
        for spec in _expected_field_rows(group):
            name = spec["field_name"]
            if group["comparison_stage"] == "NEXT1_fields":
                _, screen, field = name.split(".")
                left, right = arrays["B"][field][int(screen[-2:])], arrays["I"][field][int(screen[-2:])]
                row = _array_row(field_name=name, reference=left, candidate=right,
                    diagnostic_source=spec["diagnostic_source"], identity=group["implementation"],
                    case_id=group["case_id"], reference_case_id=group["reference_case_id"],
                    comparison_stage=group["comparison_stage"])
            elif group["comparison_stage"] == "NEXT1_observables":
                parts = name.split(".")
                screen_index = int(parts[1][-2:])
                key = ".".join(parts[2:])
                left, right = observables["B"][screen_index][key], observables["I"][screen_index][key]
                if isinstance(left, (bool, str)) or isinstance(right, (bool, str)):
                    row = _metadata_row(field_name=name, scope=spec["scope"],
                        diagnostic_source=spec["diagnostic_source"], reference=left,
                        candidate=right, identity=group["implementation"], case_id=group["case_id"],
                        reference_case_id=group["reference_case_id"],
                        comparison_stage=group["comparison_stage"])
                else:
                    row = _array_row(field_name=name, reference=np.asarray(left),
                        candidate=np.asarray(right), diagnostic_source=spec["diagnostic_source"],
                        identity=group["implementation"], case_id=group["case_id"],
                        reference_case_id=group["reference_case_id"],
                        comparison_stage=group["comparison_stage"])
            else:
                key = name.removeprefix("NEXT1.")
                left, right = metadata["B"][key], metadata["I"][key]
                row = _metadata_row(field_name=name, scope=spec["scope"],
                    diagnostic_source=spec["diagnostic_source"], reference=left,
                    candidate=right, identity=group["implementation"], case_id=group["case_id"],
                    reference_case_id=group["reference_case_id"],
                    comparison_stage=group["comparison_stage"])
            row["scope"] = spec["scope"]
            row["diagnostic_source"] = spec["diagnostic_source"]
            rows.append(row)
        path = _group_path(output_root, group)
        _write_group_rows(path, group=group, rows=rows)
        written.append(_record_raw_group(output_root=output_root,
            group_id=group_id, contract=contract, group_rows=group_rows))
    return written


def _set_phase(phase_path: Path, *, phase: str, stage_id: str,
               identity: str, timeout_seconds: float,
               progress_path: Path | None = None,
               first_interval_timeout_seconds: float | None = None) -> None:
    _write_phase(phase_path, {"schema": SCHEMA, "phase": phase,
        "stage_id": stage_id, "implementation": identity,
        "progress_path": str(progress_path) if progress_path is not None else None,
        "phase_started_utc": time.time(),
        "phase_started_monotonic": time.monotonic(),
        "timeout_seconds": float(timeout_seconds),
        "first_interval_timeout_seconds": first_interval_timeout_seconds})


def _load_case_post(window: dict, context: dict, j3, *, case_root: Path) -> tuple[dict, dict, Path, str]:
    import numpy as np
    receipt = window["propagation_receipt"]
    post_path = Path(case_root) / "propagation" / "final_POST.npz"
    post_raw = j3.raw_sha256_file(post_path)
    if post_raw != receipt["final_POST_raw_sha256"]:
        raise Q3Error("Q3 POST1 raw file hash differs from the optical window receipt")
    with np.load(post_path, allow_pickle=False) as archive:
        if set(archive.files) != set(FIELDS):
            raise Q3Error("Q3 POST1 checkpoint does not contain exactly the three HR4 fields")
        fields = j3._validate_fields({name: np.array(archive[name], order="C", copy=True)
            for name in FIELDS}, shape=(8, int(context["grid"].Ny), int(context["grid"].Nx)),
            label="Q3 POST1")
    hashes = j3._field_hashes(fields)
    if hashes != receipt["final_POST_canonical_sha256"]:
        raise Q3Error("Q3 POST1 canonical fields differ from the optical receipt")
    return fields, hashes, post_path, post_raw


def _require_safe_next_stage(stage_root: Path, j3, label: str) -> dict:
    gate = j3._current_tree_rss_gate()
    latch = Path(stage_root) / "RSS80_LATCH.json"
    gate["supervisor_rss80_latch_exists"] = latch.is_file()
    if gate["stop_90_percent"]:
        raise Q3Error(f"{label}: process-tree RSS reached the 90% stop threshold")
    if gate["latch_80_percent"] or latch.is_file():
        raise Q3Error(f"{label}: RSS80 latch prevents opening another Q3 stage")
    predicted = EXISTING_HR4_WORKER_LIMIT_SECONDS if "HR4" in label else 0.0
    if time.time() + predicted > float(os.environ["Q3_WORK_DEADLINE_EPOCH"]):
        raise Q3Error(f"{label}: insufficient allocation time")
    return gate


def _write_pre2_group(*, output_root: Path, case_id: str, branch: str,
                      checkpoint_fields: dict, actual_fields: dict,
                      contract: dict, group_rows: dict[str, list[dict]]) -> dict:
    group_id = f"{case_id}:PRE2_entry:{branch}_H1"
    return _write_static_group(output_root=output_root, group_id=group_id,
        contract=contract, group_rows=group_rows,
        left_values=checkpoint_fields, right_values=actual_fields,
        array_prefix="PRE2.")


def _prepare_q3_context(args, expected_python):
    """Retain J3 validation but bind its private interpreter guard to locked Q3 Python."""
    from types import FunctionType
    from tools.cpu_o1 import q1_t0_real as q1
    from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    from tools.cpu_p1r import cpu_p1r_optical as p1r
    q1._check_expected_python(Path(expected_python))
    private_p1r = SimpleNamespace(**{**vars(p1r), "FIXED_PYTHON":sys.executable})
    namespace = dict(j3._prepare_real_context.__globals__)
    namespace["p1r"] = private_p1r
    private = FunctionType(j3._prepare_real_context.__code__, namespace)
    return private(args)


def _chain_child(args) -> int:
    """Run one branch's P1, eight real HR4 screens, and in-memory P2_CONT."""
    stage_root, output_root = Path(args.stage_root), Path(args.output_root)
    phase_path = Path(args.phase_path)
    identity = args.branch
    if identity not in BRANCHES:
        raise Q3Error("Q3 chain branch must be B or I")
    case_ids = (f"{identity}_P1", f"{identity}_P2_CONT")
    if any((output_root / case_id).exists() for case_id in case_ids):
        raise FileExistsError("Q3 optical case root already exists")
    group_rows: dict[str, list[dict]] = {}
    _set_phase(phase_path, phase="PREPARING", stage_id=f"{identity}_CHAIN_PREPARING",
        identity=identity, timeout_seconds=WINDOW_TIMEOUT_SECONDS)
    try:
        from tools.cpu_o1 import q1_t0_real as q1real
        from tools.cpu_p1 import cpu_p1_fft as common
        from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
        j3_args = SimpleNamespace(phase=j3.PHASE,
            bench_root=Path(args.bench_root).resolve(strict=True),
            input_root=Path(args.input_root).resolve(strict=True),
            vendor_dir=Path(args.vendor_dir).resolve(strict=True))
        prepared, allocation = _prepare_q3_context(j3_args, args.expected_python)
        context, hydro = prepared["context"], prepared["hydro"]
        expected_binding = q1real._load_expected_source_binding(args.expected_source_binding)
        if context.get("source_binding") != expected_binding:
            raise Q3Error("Q3 child source binding differs from the preflight-bound identity")
        hr4_identity = _validate_identity_and_hydro(j3_args.input_root, context, hydro)
        contract = load_static_contract(Path(args.contract_path))
        _require_safe_next_stage(stage_root, j3, f"{identity} before P1")

        pre0 = j3._initial_pre_fields(context)
        pre0_hashes = j3._field_hashes(pre0)
        references = ([ ("B_P1", output_root / "B_P1") ]
                      if identity == "I" else [])
        p1_case = _run_optical_window(args=args, context=context,
            identity=identity, case_id=case_ids[0], pre_fields=pre0,
            initial_pre0=True, expected_pre_hashes=pre0_hashes,
            output_root=output_root, phase_path=phase_path,
            comparison_references=references, contract=contract,
            group_rows=group_rows)
        post1, post_hashes, post_path, post_raw = _load_case_post(
            p1_case, context, j3, case_root=output_root / p1_case["case_id"])
        activity = {"post1_vs_pre0": _activity_summary(pre0, post1)}

        _require_safe_next_stage(stage_root, j3, f"{identity} before HR4")
        _set_phase(phase_path, phase="HYDRO", stage_id=f"{identity}_H1",
            identity=identity, timeout_seconds=HYDRO_OUTER_LIMIT_SECONDS)
        hydro_result = j3._process_hydro_workers(SimpleNamespace(
                bench_root=j3_args.bench_root, vendor_dir=j3_args.vendor_dir),
            child_root=stage_root, post_path=post_path,
            post_raw_sha256=post_raw, post_hashes=post_hashes,
            hydro_parameters=hydro, execution_scope="J3_REAL_INPUT")
        worker_rows = _validate_screen_ordinals(hydro_result["receipt"].get("worker_rows", []))
        if (hydro_result["receipt"].get("worker_processes_launched") != 8
                or hydro_result["receipt"].get("worker_processes_completed") != 8
                or any(row.get("status") != "DONE" for row in worker_rows)):
            raise Q3Error("Q3 HR4 did not complete exactly eight independent screen workers")
        for ordinal, row in enumerate(worker_rows):
            nested = row.get("receipt")
            if (not isinstance(nested, dict) or nested.get("ordinal") != ordinal
                    or nested.get("n_steps") != EXPECTED_HR4_STEPS
                    or nested.get("hr4_backend", {}).get("backend") != "numpy"):
                raise Q3Error(f"Q3 HR4 screen {ordinal} lacks its real 1000-step CPU receipt")
        next_fields = _validate_next1_fields(hydro_result["fields"],
            shape=(8, int(context["grid"].Ny), int(context["grid"].Nx)))
        next_hashes = j3._field_hashes(next_fields)
        if next_hashes != hydro_result["field_sha256"]:
            raise Q3Error("Q3 in-memory HR4 NEXT1 differs from worker result hashes")
        metadata = _checkpoint_metadata(identity=identity, context=context,
            hydro=hydro, post_hashes=post_hashes, hr4_identity=hr4_identity)
        checkpoint_path = stage_root / f"NEXT1_{identity}.npz"
        checkpoint = _publish_checkpoint(checkpoint_path, next_fields,
            metadata=metadata,
            shape=(8, int(context["grid"].Ny), int(context["grid"].Nx)), j3=j3)
        if checkpoint["field_sha256"] != next_hashes:
            raise Q3Error("Q3 immutable NEXT1 publication changed the in-memory hydro state")
        disk_fields, disk_checkpoint_receipt = _load_checkpoint(checkpoint_path,
            expected_branch=identity,
            expected_shape=(8, int(context["grid"].Ny), int(context["grid"].Nx)),
            expected_metadata=metadata, j3=j3)
        if j3._field_hashes(disk_fields) != next_hashes:
            raise Q3Error("Q3 published NEXT1 re-read differs from in-memory HR4 result")

        group_id = f"{identity}_H1:NEXT1_memory_vs_disk:{identity}_H1"
        _write_static_group(output_root=output_root, group_id=group_id,
            contract=contract, group_rows=group_rows,
            left_values=next_fields, right_values=disk_fields,
            array_prefix="NEXT1.")
        metadata_id = f"{identity}_H1:NEXT1_memory_vs_disk_metadata:{identity}_H1"
        metadata_values = _checkpoint_metadata_values(metadata)
        _write_static_group(output_root=output_root, group_id=metadata_id,
            contract=contract, group_rows=group_rows,
            left_values={}, right_values={}, left_metadata=metadata_values,
            right_metadata=_checkpoint_metadata_values(disk_checkpoint_receipt["metadata"]))
        activity["next1_vs_post1"] = _activity_summary(post1, next_fields)
        activity["next1_vs_pre0"] = _activity_summary(pre0, next_fields)
        hr4_rss = j3._current_tree_rss_gate()
        hr4_rss["worker_batch_rss80_latched"] = bool(
            hydro_result["receipt"].get("rss_80_percent_latched"))
        if (hr4_rss["stop_90_percent"] or hr4_rss["latch_80_percent"]
                or hr4_rss["worker_batch_rss80_latched"]
                or (stage_root / "RSS80_LATCH.json").is_file()):
            raise Q3Error("Q3 RSS gate prevents starting P2_CONT after HR4")

        # Persist the H1 evidence before any I P2 work; exact failure stops here.
        h1_receipt = {"stage_root": str(stage_root), "checkpoint_path": str(checkpoint_path),
            "next1": checkpoint["receipt"], "hr4": hydro_result["receipt"],
            "hydro_parameters": j3._json_safe(hydro), "hr4_identity": hr4_identity}
        _write_new_json(stage_root / "h1_receipt.json", h1_receipt)
        if identity == "I":
            b_h1 = json.loads((output_root / "stages/B_CHAIN/h1_receipt.json").read_text())
            _write_hr4_cross_groups(output_root=output_root, contract=contract,
                group_rows=group_rows, b_chain=b_h1, i_chain=h1_receipt)
        checkpoint_before_p2 = _sha256_file(checkpoint_path)
        p2_case = case_ids[1]
        p2_refs = ([ ("B_P2_CONT", output_root / "B_P2_CONT") ]
                   if identity == "I" else [])
        p2 = _run_optical_window(args=args, context=context,
            identity=identity, case_id=p2_case,
            pre_fields=next_fields, initial_pre0=False,
            expected_pre_hashes=next_hashes, output_root=output_root,
            phase_path=phase_path, comparison_references=p2_refs,
            contract=contract, group_rows=group_rows,
            capture_pre_for_comparison=True)
        actual_entry_fields = p2.pop("_actual_pre_fields_for_comparison")
        pre2_group = _write_pre2_group(output_root=output_root,
            case_id=p2_case, branch=identity, checkpoint_fields=next_fields,
            actual_fields=actual_entry_fields, contract=contract,
            group_rows=group_rows)
        if _sha256_file(checkpoint_path) != checkpoint_before_p2:
            raise Q3Error("NEXT1 changed during normal P2")
        _load_checkpoint(checkpoint_path, expected_branch=identity,
            expected_shape=tuple(next_fields["delta_n"].shape), expected_metadata=metadata)
        case_receipts = {case_ids[0]: p1_case, case_ids[1]: p2}
        _write_new_json(stage_root / "chain_receipt.json", {
            "schema": SCHEMA, "status": "CHAIN_COMPLETE_DIAGNOSTIC",
            "branch": identity, "pid": os.getpid(), "parent_pid": os.getppid(),
            "process_model": "ONE_PROCESS_P1_HR4_MEMORY_NEXT1_P2_CONT",
            "allocation": allocation, "source_binding": context["source_binding"],
            "source_canonical_sha256": context["source_hash"],
            "stage_root": str(stage_root), "checkpoint_path": str(checkpoint_path),
            "hydro_parameters": j3._json_safe(hydro), "hr4_identity": hr4_identity,
            "pre0_field_canonical_sha256": pre0_hashes,
            "p1": p1_case, "hr4": hydro_result["receipt"],
            "next1": checkpoint["receipt"], "p2_cont": p2,
            "p2_pre2_comparison": pre2_group, "activity": activity,
            "rss_gate_after_hr4": hr4_rss,
            "fresh_source_loads": 2, "optical_pulses": 2,
            "hydro_transitions": 1,
            "production_qualification": QUALIFICATION})
        return 0
    except BaseException as exc:
        try:
            _write_new_json(stage_root / "child_failure.json", {
                "schema": SCHEMA, "status": "FAIL", "branch": identity,
                "pid": os.getpid(), "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "production_qualification": QUALIFICATION})
        except Exception:
            pass
        traceback.print_exc()
        return 1


def _reload_child(args) -> int:
    """Run P2_RELOAD in a new interpreter using only the immutable NEXT1 disk files."""
    stage_root, output_root = Path(args.stage_root), Path(args.output_root)
    phase_path = Path(args.phase_path)
    identity = args.branch
    if identity not in BRANCHES:
        raise Q3Error("Q3 reload branch must be B or I")
    case_id = f"{identity}_P2_RELOAD"
    if (output_root / case_id).exists():
        raise FileExistsError("Q3 reload case root already exists")
    try:
        from tools.cpu_o1 import q1_t0_real as q1real
        from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
        j3_args = SimpleNamespace(phase=j3.PHASE,
            bench_root=Path(args.bench_root).resolve(strict=True),
            input_root=Path(args.input_root).resolve(strict=True),
            vendor_dir=Path(args.vendor_dir).resolve(strict=True))
        _set_phase(phase_path, phase="PREPARING", stage_id=f"{identity}_RELOAD_PREPARING",
            identity=identity, timeout_seconds=WINDOW_TIMEOUT_SECONDS)
        prepared, allocation = _prepare_q3_context(j3_args, args.expected_python)
        context, hydro = prepared["context"], prepared["hydro"]
        expected_binding = q1real._load_expected_source_binding(args.expected_source_binding)
        if context.get("source_binding") != expected_binding:
            raise Q3Error("Q3 reload source binding differs from the preflight-bound identity")
        hr4_identity = _validate_identity_and_hydro(j3_args.input_root, context, hydro)
        checkpoint_path = Path(args.checkpoint_path).resolve(strict=True)
        if os.getpid() == int(args.normal_chain_pid):
            raise Q3Error("Q3 P2_RELOAD did not run in a new interpreter process")
        sidecar_path = checkpoint_path.with_name(checkpoint_path.stem + ".receipt.json")
        preliminary = json.loads(sidecar_path.read_text(encoding="utf-8"))
        post_hashes = preliminary.get("metadata", {}).get("post_field_canonical_sha256")
        if not isinstance(post_hashes, dict):
            raise Q3Error("Q3 NEXT1 receipt lacks its POST1 scientific input identity")
        expected_metadata = _checkpoint_metadata(identity=identity,
            context=context, hydro=hydro, post_hashes=post_hashes,
            hr4_identity=hr4_identity)
        next_fields, checkpoint_receipt = _load_checkpoint(checkpoint_path,
            expected_branch=identity,
            expected_shape=(8, int(context["grid"].Ny), int(context["grid"].Nx)),
            expected_metadata=expected_metadata, j3=j3)
        checkpoint_raw_before = _sha256_file(checkpoint_path)
        checkpoint_hashes = j3._field_hashes(next_fields)
        _require_safe_next_stage(stage_root, j3, f"{identity} before P2_RELOAD")
        contract = load_static_contract(Path(args.contract_path))
        references = [(f"{identity}_P2_CONT", output_root / f"{identity}_P2_CONT")]
        group_rows: dict[str, list[dict]] = {}
        result = _run_optical_window(args=args, context=context,
            identity=identity, case_id=case_id,
            pre_fields=next_fields, initial_pre0=False,
            expected_pre_hashes=checkpoint_hashes, output_root=output_root,
            phase_path=phase_path, comparison_references=references,
            contract=contract, group_rows=group_rows,
            capture_pre_for_comparison=True)
        actual_entry_fields = result.pop("_actual_pre_fields_for_comparison")
        pre2_group = _write_pre2_group(output_root=output_root,
            case_id=case_id, branch=identity, checkpoint_fields=next_fields,
            actual_fields=actual_entry_fields, contract=contract,
            group_rows=group_rows)
        checkpoint_raw_after = _sha256_file(checkpoint_path)
        if checkpoint_raw_after != checkpoint_raw_before:
            raise Q3Error("Q3 immutable NEXT1 changed while the independent reload ran")
        _write_new_json(stage_root / "reload_receipt.json", {
            "schema": SCHEMA, "status": "RELOAD_COMPLETE_DIAGNOSTIC",
            "branch": identity, "pid": os.getpid(),
            "normal_chain_pid": int(args.normal_chain_pid),
            "process_model": "NEW_INTERPRETER_DISK_NEXT1_ONLY",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_raw_sha256": checkpoint_raw_before,
            "checkpoint_field_canonical_sha256": checkpoint_hashes,
            "checkpoint_metadata": checkpoint_receipt["metadata"],
            "fresh_source": result["fresh_source"],
            "actual_pre_fields_canonical_sha256": result[
                "actual_entry_pre_fields_canonical_sha256"],
            "p2_reload": result, "p2_pre2_comparison": pre2_group,
            "allocation": allocation,
            "production_qualification": QUALIFICATION})
        return 0
    except BaseException as exc:
        try:
            _write_new_json(stage_root / "child_failure.json", {
                "schema": SCHEMA, "status": "FAIL", "branch": identity,
                "pid": os.getpid(), "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "production_qualification": QUALIFICATION})
        except Exception:
            pass
        traceback.print_exc()
        return 1


def _supervise_q3_stage(command: list[str], *, stage_root: Path,
                        phase_path: Path, cwd: Path, env: dict,
                        allocation_deadline: float) -> dict:
    """Supervise one Q3 child while following its mutable optical/HR4 phase."""
    from tools.cpu_p1 import cpu_p1_fft as common
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    from tools.cpu_o1 import q1_t0_real as q1real

    stdout_path, stderr_path = stage_root / "stdout.log", stage_root / "stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        started = time.monotonic()
        proc = subprocess.Popen(command, cwd=str(cwd), env=env,
            stdout=stdout, stderr=stderr, start_new_session=True)
        peak_rss = peak_pss = 0
        samples = []
        first_interval_started = None
        first_boundary_arrived = False
        current_stage_id = None
        progress_path = None
        progress_offset = 0
        pending = b""
        latest_phase = {}
        stop = None
        rss_latched = False
        while proc.poll() is None:
            now = time.monotonic()
            if now >= allocation_deadline:
                stop = {"reason": "ALLOCATION_WORK_DEADLINE",
                    **j1._terminate_child_bounded(proc)}
                break
            try:
                latest_phase = json.loads(phase_path.read_text(encoding="utf-8"))
                stage_id = latest_phase.get("stage_id")
                if stage_id != current_stage_id:
                    current_stage_id = stage_id
                    first_interval_started = None
                    first_boundary_arrived = False
                    progress_offset = 0
                    pending = b""
                    progress_path = (Path(latest_phase["progress_path"])
                        if latest_phase.get("progress_path") else None)
                phase_started = latest_phase.get("phase_started_monotonic", started)
                timeout = float(latest_phase.get("timeout_seconds") or WINDOW_TIMEOUT_SECONDS)
                phase_name = latest_phase.get("phase")
                if progress_path is not None and progress_path.is_file():
                    with progress_path.open("rb") as stream:
                        stream.seek(progress_offset)
                        payload = stream.read()
                        progress_offset = stream.tell()
                    pending += payload
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        if not line.strip():
                            continue
                        event = json.loads(line.decode("utf-8"))
                        if event.get("event") == "first_interval_started":
                            first_interval_started = float(event["monotonic"])
                        elif event.get("event") == "first_boundary_arrived":
                            first_boundary_arrived = True
                tree = common.tree_snapshot(proc.pid)
                rss = int(tree["rss"]) if tree is not None else None
                pss_state = j1._pss_tree_snapshot(proc.pid)
                pss = pss_state.get("pss_bytes")
                cgroup = q1real._cgroup_memory_snapshot(proc.pid)
                memory = q1real._memory_fractions(rss_bytes=rss, pss_bytes=pss,
                    cgroup=cgroup, allocation_bytes=MEMORY_BYTES)
                if memory["max_enforced_fraction"] is None:
                    raise RuntimeError("Q3_MEMORY_TELEMETRY_UNAVAILABLE")
                if rss is not None:
                    peak_rss = max(peak_rss, rss)
                if pss is not None:
                    peak_pss = max(peak_pss, int(pss))
                fraction = memory["max_enforced_fraction"]
                if fraction >= MEMORY_WARN_FRACTION:
                    rss_latched = True
                    latch = stage_root / "RSS80_LATCH.json"
                    if not latch.exists():
                        _write_new_json(latch, {"reason": "MEMORY_80_PERCENT",
                            "stage_id": current_stage_id, "sample": memory,
                            "elapsed_seconds": now - started})
                samples.append({"elapsed_seconds": now-started,
                    "stage_id": current_stage_id, **memory,
                    "process_tree_cpu_seconds": float(tree.get("cpu", 0.0))
                    if tree is not None else None})
                reason = None
                if phase_name == "OPTICAL" and first_interval_started is not None and not first_boundary_arrived:
                    if now - first_interval_started >= FIRST_INTERVAL_SECONDS:
                        reason = "FIRST_INTERVAL_TIMEOUT"
                if now - float(phase_started) >= timeout:
                    reason = "STAGE_TIMEOUT"
                if fraction >= MEMORY_ABORT_FRACTION:
                    reason = "MEMORY_90_PERCENT"
                if reason:
                    stop = {"reason": reason, "stage_id": current_stage_id,
                        **j1._terminate_child_bounded(proc)}
                    break
            except BaseException as exc:
                stop = {"reason": "SUPERVISOR_FAILURE",
                    "error": f"{type(exc).__name__}: {exc}",
                    **j1._terminate_child_bounded(proc)}
                break
            time.sleep(max(0.0, MEMORY_SAMPLE_SECONDS - (time.monotonic() - now)))
        if proc.poll() is None and stop is not None:
            j1._terminate_child_bounded(proc)
        try:
            exit_code = proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            stop = {"reason": "POST_STOP_WAIT_EXPIRED",
                **j1._terminate_child_bounded(proc)}
            try:
                exit_code = proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                exit_code = None
    result = {"schema": SCHEMA, "exit_code": exit_code,
        "elapsed_wall_seconds": time.monotonic()-started,
        "stop": stop, "last_phase": latest_phase,
        "first_interval_started_monotonic": first_interval_started,
        "first_boundary_arrived": first_boundary_arrived,
        "memory_warn_fraction": MEMORY_WARN_FRACTION,
        "memory_abort_fraction": MEMORY_ABORT_FRACTION,
        "rss_80_percent_latched": rss_latched,
        "peak_process_tree_rss_bytes": peak_rss or None,
        "peak_process_tree_pss_bytes": peak_pss or None,
        "sample_interval_seconds": MEMORY_SAMPLE_SECONDS, "samples": samples,
        "logs": {"stdout": str(stdout_path), "stderr": str(stderr_path)}}
    _write_new_json(stage_root / "supervisor_receipt.json", result)
    return result


def _validate_caps(cap, files):
    if not (0 < int(cap) <= MAX_STORAGE_BYTES and 0 < int(files) <= MAX_FILES):
        raise Q3Error("Q3 storage cap must be within 16 GiB / 50000 files")


def _q3_dynamic_test_basetemp(storage_root: Path, output_root: Path) -> Path:
    """Keep pytest temporary files under the storage-accounted Q3 run scope."""
    run_scope = hashlib.sha256(os.fsencode(str(Path(output_root).resolve()))).hexdigest()
    return Path(storage_root).resolve() / "preflight" / f"q3_{run_scope}" / "pytest_tmp"


def _run_q3(args) -> dict:
    from types import FunctionType
    from tools.cpu_o1 import q1_t0_real as q1
    from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    # Only the private Q3 storage admission differs; Q1 module globals stay intact.
    namespace = dict(q1._coordinator_preflight.__globals__)
    namespace["_validate_storage_caps"] = _validate_caps
    private_preflight = FunctionType(q1._coordinator_preflight.__code__, namespace)
    started_epoch = time.time()
    os.environ["P1R_ALLOCATION_START_EPOCH"] = str(started_epoch)
    os.environ["P1R2_ALLOCATION_START_EPOCH"] = str(started_epoch)
    deadline_epoch = started_epoch + ALLOCATION_SECONDS - RESERVE_SECONDS
    os.environ["Q3_WORK_DEADLINE_EPOCH"] = str(deadline_epoch)
    contract = load_static_contract(args.contract_path)
    preflight = private_preflight(args)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    public_preflight = {k:v for k,v in preflight.items() if k != "held_native_thread_limit"}
    _write_new_json(root / "preflight.json", public_preflight)
    q1._write_environment_receipt(root, python_identity=preflight["python"],
        allocation=preflight["allocation"], linux_probe=preflight["linux_environment"],
        vendor_dir=args.vendor_dir)
    result = {"schema":SCHEMA,"status":"Q3_RUNNING","stages":{},
        "production_qualification":QUALIFICATION,"allocation_start_epoch":started_epoch}
    try:
        result["allocation_probe"] = q1._run_allocation_probe(root)
        result["wired_tiny"] = q1._run_wired_tiny(root, identities=("B","I"))
        if result["wired_tiny"].get("status") != "WIRED_TINY_ONLY":
            raise Q3Error("allocation B/I wiring gate failed")
        # Dynamic Linux regressions, including real tiny two-pulse/reload, run here only.
        pytest_basetemp = _q3_dynamic_test_basetemp(args.storage_root, root)
        pytest_basetemp.parent.mkdir(parents=True, exist_ok=False)
        command = [str(args.expected_python),"-s","-B","-m","pytest","-p","no:cacheprovider","-q",
            str(Path(__file__).with_name("test_cpu_o1.py")), str(Path(__file__).with_name("test_q3_t0_real.py")),
            "-k","q1_ or q2_ or q3_ or bounded_comparator","--basetemp",str(pytest_basetemp)]
        with (root/"dynamic_tests.stdout.log").open("xb") as so, (root/"dynamic_tests.stderr.log").open("xb") as se:
            test = subprocess.run(command,cwd=args.bench_root,stdout=so,stderr=se,timeout=900)
        _write_new_json(root/"DYNAMIC_TESTS.json",{"command":command,"exit_code":test.returncode})
        if test.returncode: raise Q3Error("Q3_DYNAMIC_TESTS_FAILED")
        env = q1._child_environment(args.bench_root,args.expected_python)
        for branch in BRANCHES:
            for mode in ("chain","reload"):
                stage_id = branch + ("_CHAIN" if mode=="chain" else "_RELOAD")
                q1._check_storage_growth(args.storage_root,growth_cap_bytes=args.storage_cap_bytes,max_files=args.max_files)
                if time.time() >= deadline_epoch: raise Q3Error("ALLOCATION_WORK_DEADLINE")
                stage = root/"stages"/stage_id
                stage.mkdir(parents=True,exist_ok=False)
                phase = stage/"active_phase.json"
                _set_phase(phase,phase="CHILD_STARTING",stage_id=stage_id,identity=branch,timeout_seconds=2700)
                cmd=[str(args.expected_python),"-s","-B","-m","tools.cpu_o1.q3_t0_real",
                    "--child-mode",mode,"--branch",branch,"--bench-root",str(args.bench_root),
                    "--input-root",str(args.input_root),"--vendor-dir",str(args.vendor_dir),
                    "--expected-source-binding",str(args.expected_source_binding),"--contract-path",str(args.contract_path),
                    "--output-root",str(root),"--stage-root",str(stage),"--phase-path",str(phase),
                    "--expected-python",str(args.expected_python)]
                if mode=="reload":
                    chain=result["stages"][branch+"_CHAIN"]["receipt"]
                    cmd += ["--checkpoint-path",chain["checkpoint_path"],"--normal-chain-pid",str(chain["pid"])]
                _write_new_json(stage/"COMMAND.json",{"command":cmd,"normal_process_exited_before_reload":mode=="reload"})
                supervisor=_supervise_q3_stage(cmd,stage_root=stage,phase_path=phase,cwd=args.bench_root,
                    env=env,allocation_deadline=time.monotonic()+max(0,deadline_epoch-time.time()))
                record={"supervisor":supervisor,"status":"FAIL"}
                result["stages"][stage_id]=record
                if supervisor["exit_code"] != 0 or supervisor["stop"]:
                    raise Q3Error("Q3 child failed: "+stage_id)
                receipt=json.loads((stage/("chain_receipt.json" if mode=="chain" else "reload_receipt.json")).read_text())
                expected="CHAIN_COMPLETE_DIAGNOSTIC" if mode=="chain" else "RELOAD_COMPLETE_DIAGNOSTIC"
                if receipt.get("status")!=expected: raise Q3Error("stage receipt incomplete")
                record.update(status="DONE",receipt=receipt)
                if supervisor["rss_80_percent_latched"]: raise Q3Error("RSS80 prevents next stage")
        raw={}
        for gid in contract["group_map"]:
            _record_raw_group(output_root=root,group_id=gid,contract=contract,group_rows=raw)
        resource=q1._check_storage_growth(args.storage_root,growth_cap_bytes=args.storage_cap_bytes,max_files=args.max_files)
        gate=build_q3_raw_once_gate(contract=contract,group_rows=raw,
            identity_checks={"six_windows_two_branches_completed":True,"dynamic_tests_passed":True,
                "independent_reload_processes":all(result["stages"][b+"_CHAIN"]["receipt"]["pid"] != result["stages"][b+"_RELOAD"]["receipt"]["pid"] for b in BRANCHES)},
            linux_environment_eligible=preflight["linux_environment"]["eligible"],
            resource_budget_respected=True,matrix_status="ALL_REQUIRED_STAGES_COMPLETE")
        _write_new_json(root/"NUMERIC_GATE.json",gate)
        if not gate.get("proceed_to_next_identity"): raise Q3Error("final raw numeric gate failed")
        result.update(status="Q3_READY_FOR_MANUAL_REVIEW",gate=gate,storage_final=resource,
            decision="Q3_SHORT_WINDOW_TWO_PULSE_RELOAD_EXACT_SUPPORTED")
    except BaseException as exc:
        result.update(status="Q3_PARTIAL_OR_FAILED",error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["wall_seconds"]=time.time()-started_epoch
        _write_new_json(root/"coordinator_receipt.json",result)
    return result


def _entry_smoke(args):
    from tools.cpu_o1 import q1_t0_real as q1
    q1._check_expected_python(args.expected_python)
    root=args.output_root;root.mkdir(parents=True,exist_ok=False)
    cmd=[str(args.expected_python),"-s","-B","-m","tools.cpu_o1.q3_t0_real",
        "--entry-smoke-child","--expected-python",str(args.expected_python)]
    env=os.environ.copy();env["PYTHONPATH"]=str(args.bench_root/"Filament_python")
    with (root/"stdout.log").open("xb") as so,(root/"stderr.log").open("xb") as se:
        proc=subprocess.run(cmd,cwd=args.bench_root,env=env,stdout=so,stderr=se,timeout=60)
    receipt={"command":cmd,"exit_code":proc.returncode,"scope":"NO_SCIENTIFIC_EXECUTION"}
    _write_new_json(root/"ENTRY_SMOKE.json",receipt)
    if proc.returncode:raise Q3Error("Q3 parent-child entry failed")
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("bench-root","provenance","input-root","vendor-dir","output-root","storage-root",
            "expected-source-binding","expected-python","stage-root","phase-path","checkpoint-path"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--contract-path",type=Path,default=CONTRACT_PATH)
    parser.add_argument("--storage-cap-bytes",type=int)
    parser.add_argument("--max-files",type=int,default=MAX_FILES)
    parser.add_argument("--branch",choices=BRANCHES)
    parser.add_argument("--normal-chain-pid",type=int)
    modes=parser.add_mutually_exclusive_group(required=True)
    for flag in ("run","dry-run","entry-smoke","entry-smoke-child"):
        modes.add_argument("--"+flag,action="store_true")
    modes.add_argument("--child-mode",choices=("chain","reload"))
    modes.add_argument("--tiny-mode",choices=("chain","reload"))
    args=parser.parse_args(argv)
    if args.tiny_mode:return _tiny_child(args)
    from tools.cpu_o1 import q1_t0_real as q1
    if args.expected_python is None:parser.error("--expected-python required")
    q1._check_expected_python(args.expected_python)
    if args.entry_smoke_child:
        load_static_contract(args.contract_path)
        print(json.dumps({"status":"ENTRY_SMOKE_CHILD_PASS","pid":os.getpid(),"python":sys.executable}))
        return 0
    if args.child_mode:
        return _chain_child(args) if args.child_mode=="chain" else _reload_child(args)
    for name in ("bench_root","provenance","input_root","vendor_dir","output_root","storage_root","expected_source_binding","storage_cap_bytes"):
        if getattr(args,name) is None:parser.error("missing "+name)
    _validate_caps(args.storage_cap_bytes,args.max_files)
    if args.dry_run:
        for name in ("bench_root","provenance","input_root","vendor_dir","storage_root","expected_source_binding"):
            if not getattr(args,name).exists():raise FileNotFoundError(str(getattr(args,name)))
        if args.output_root.exists():raise FileExistsError(str(args.output_root))
        print(json.dumps({"status":"DRY_RUN_ONLY","cases":WINDOW_CASES,"groups":len(load_static_contract(args.contract_path)["group_map"])}))
        return 0
    if args.entry_smoke:
        print(json.dumps(_entry_smoke(args)));return 0
    try:
        result=_run_q3(args)
        return 0 if result["status"]=="Q3_READY_FOR_MANUAL_REVIEW" else 3
    except BaseException as exc:
        failure={"status":"Q3_PREFLIGHT_BLOCKED","error":type(exc).__name__+": "+str(exc),"traceback":traceback.format_exc()}
        if args.storage_root and args.storage_root.is_dir():
            _write_new_json(args.storage_root/("Q3_COORDINATOR_FAILURE_"+str(os.getpid())+".json"),failure)
        print(json.dumps(failure),file=sys.stderr);return 1


def _tiny_child(args):
    """Real frozen operators on synthetic tiny arrays; no sealed real assets."""
    import numpy as np
    import numba
    import threadpoolctl
    from KHz_filament import hr4
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1, cpu_p1r2_j3 as j3
    from tools.cpu_o1.cpu_o1 import make_joint_propagator_factory
    limit=threadpoolctl.threadpool_limits(limits=1)
    root=args.output_root;root.mkdir(parents=True,exist_ok=False)
    context=j1._tiny_context();branch=args.branch
    if branch not in BRANCHES:raise Q3Error("tiny requires B/I")
    threads=min(24,numba.config.NUMBA_NUM_THREADS)
    source=context["source"];source_hash=j1._canonical_hash(source)
    shape=tuple(context["pre_fields"]["delta_n"].shape)
    metadata={"checkpoint_kind":"NEXT1","branch":branch,"completed_pulse":1,"next_pulse":2,
        "screen_ids":list(SCREEN_IDS),"scope":"TINY_REAL_OPERATORS_ONLY"}
    def window(name,fields,initial):
        fresh=np.array(source,copy=True,order="C")
        factory=make_joint_propagator_factory(branch,kernel_threads=threads,
            block_pixels=32,layout="NtB",temporal_workers=48,spatial_workers=24)
        result=j3._window(context,fields,output_root=root/name,stage_name=name,
            initial_pre0=initial,source=fresh,source_hash=source_hash,
            pre_hash=j1._canonical_hash(fields["delta_n"]) if initial else None,
            execution_scope="SOFTWARE_SMOKE_ONLY",propagator_factory=factory,
            state_origin=None if initial else "J3_NEXT1_MEMORY" if args.tiny_mode=="chain" else "J3_NEXT1_DISK_RELOADED")
        from tools.cpu_o1.q1_t0 import _check_identity_dispatch
        result["dispatch"]=_check_identity_dispatch(branch,factory,kernel_threads=threads,temporal_workers=48,spatial_workers=24)
        return result
    if args.tiny_mode=="chain":
        p1=window("P1",context["pre_fields"],True)
        with np.load(p1["final_POST_path"],allow_pickle=False) as z:post={k:np.array(z[k],copy=True) for k in FIELDS}
        params={"dx":context["grid"].Lx/context["grid"].Nx,"dy":context["grid"].Ly/context["grid"].Ny,
            "dt_hydro":1e-6,"chi":2.17e-5,"nu":1.5e-5,"n0":1.00027,"gravity_x":0.,"gravity_y":-9.81,
            "cfl_limit":1.,"n_steps":1000,"x_min":-context["grid"].Lx/2,"y_min":-context["grid"].Ly/2}
        next_fields={k:[] for k in FIELDS};calls=[]
        for ordinal in range(8):
            result=j3._invoke_hydro(hr4.advance_hr4_single_screen,{k:post[k][ordinal].copy() for k in FIELDS},params)
            calls.append({"ordinal":ordinal,"steps":result["steps"],"backend":result["performance"]["backend"]})
            for k in FIELDS:next_fields[k].append(result[k])
        next_fields={k:np.ascontiguousarray(np.stack(v),dtype=np.float64) for k,v in next_fields.items()}
        ck=_publish_checkpoint(root/"NEXT1.npz",next_fields,metadata=metadata,shape=shape)
        before=ck["raw_sha256"]
        p2=window("P2_CONT",next_fields,False)
        if _sha256_file(root/"NEXT1.npz")!=before:raise Q3Error("tiny checkpoint mutated")
        receipt={"pid":os.getpid(),"branch":branch,"p1":p1,"p2":p2,"hydro":calls,"source_hash":source_hash,
            "next_hashes":ck["field_sha256"],"checkpoint_path":str(root/"NEXT1.npz")}
    else:
        fields,ck=_load_checkpoint(args.checkpoint_path,expected_branch=branch,expected_shape=shape,expected_metadata=metadata)
        p2=window("P2_RELOAD",fields,False)
        receipt={"pid":os.getpid(),"branch":branch,"p2":p2,"source_hash":source_hash,"next_hashes":ck["field_canonical_sha256"]}
    _write_new_json(root/"tiny_receipt.json",receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

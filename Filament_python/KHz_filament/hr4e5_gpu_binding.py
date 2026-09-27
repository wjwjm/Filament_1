"""GPU role assignment and startup/final identity checks for HR-4E5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


GPU_BINDING_SCHEMA = "khz_filament.hr4e5_gpu_binding.v1"
GPU_ASSIGNMENT_SCHEMA = "khz_filament.hr4e5_gpu_assignment.v1"
GPU_COUNTS = (4, 6)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _roles_for_count(gpu_count: int) -> tuple[str, ...]:
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count not in GPU_COUNTS:
        raise ValueError("expected_gpu_count must be exactly 4 or 6")
    return ("R", "C_OPT", *(f"H{index}" for index in range(gpu_count - 2)))


def parse_cuda_visible_devices(value: str | None) -> tuple[str, ...]:
    """Parse a CUDA visibility list without normalizing away ambiguous tokens."""
    if not isinstance(value, str) or not value:
        raise ValueError("CUDA_VISIBLE_DEVICES is missing or empty")

    tokens = value.split(",")
    if any(not token or token != token.strip() or any(ch.isspace() for ch in token) for token in tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES contains an empty or whitespace-padded token")
    if any(token == "-1" for token in tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES hides all devices")

    def token_key(token: str) -> str:
        if token.isdecimal():
            return f"index:{int(token)}"
        return f"id:{token.casefold()}"

    keys = [token_key(token) for token in tokens]
    if len(set(keys)) != len(keys):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate device tokens")
    return tuple(tokens)


def assign_roles(cuda_visible_devices: str | None, expected_gpu_count: int) -> dict[str, str]:
    """Map coordinator-visible CUDA tokens to the explicit HR-4E-5 roles."""
    roles = _roles_for_count(expected_gpu_count)
    tokens = parse_cuda_visible_devices(cuda_visible_devices)
    if len(tokens) != expected_gpu_count:
        raise ValueError(
            f"expected {expected_gpu_count} CUDA_VISIBLE_DEVICES tokens, received {len(tokens)}"
        )
    return dict(zip(roles, tokens, strict=True))


def _worker_index(role: str) -> int | None:
    return int(role[1:]) if role.startswith("H") else None


def build_assignment_record(cuda_visible_devices: str | None, expected_gpu_count: int) -> dict[str, Any]:
    """Build the single schema-versioned role/token contract for an allocation."""
    role_to_token = assign_roles(cuda_visible_devices, expected_gpu_count)
    assignments = {
        role: {
            "token": token,
            "scheduler_index": index,
            "worker_index": _worker_index(role),
        }
        for index, (role, token) in enumerate(role_to_token.items())
    }
    return {
        "schema": GPU_ASSIGNMENT_SCHEMA,
        "status": "PASS",
        "expected_gpu_count": expected_gpu_count,
        "scheduler_visible_devices": cuda_visible_devices,
        "role_to_token": role_to_token,
        "assignments": assignments,
    }


def validate_assignment_record(
    record: Mapping[str, Any], expected_gpu_count: int | None = None
) -> bool:
    """Validate a persisted assignment dict before any child is launched."""
    if not isinstance(record, Mapping):
        raise ValueError("assignment record must be an object")
    if record.get("schema") != GPU_ASSIGNMENT_SCHEMA or record.get("status") != "PASS":
        raise ValueError("assignment record has an incompatible schema or status")
    count = record.get("expected_gpu_count")
    if isinstance(count, bool) or not isinstance(count, int) or count not in GPU_COUNTS:
        raise ValueError("assignment record must declare a 4- or 6-GPU topology")
    if expected_gpu_count is not None and count != expected_gpu_count:
        raise ValueError("assignment record GPU count does not match the requested topology")
    expected = build_assignment_record(record.get("scheduler_visible_devices"), count)
    role_to_token = record.get("role_to_token")
    assignments = record.get("assignments")
    if not isinstance(role_to_token, Mapping) or dict(role_to_token) != expected["role_to_token"]:
        raise ValueError("assignment role-to-token map does not match scheduler token order")
    if not isinstance(assignments, Mapping) or dict(assignments) != expected["assignments"]:
        raise ValueError("assignment entries do not match the role/token contract")
    return True


def build_cpu_schema_receipt(role: str) -> dict[str, Any]:
    """Return a schema-only local fixture with no invented GPU identity."""
    if role not in {"R", "C_OPT", "H0", "H1", "H2", "H3"}:
        raise ValueError(f"unknown GPU role: {role!r}")
    return {
        "schema": GPU_BINDING_SCHEMA,
        "receipt_type": "cpu_schema",
        "status": "SCHEMA_ONLY",
        "role": role,
        "identity": {
            "process": {
                "pid": None,
                "ppid": None,
                "start_token": None,
                "parent_start_token": None,
            },
            "assigned_cuda_visible_device": None,
            "gpu": {"uuid": None, "model": None, "total_memory_bytes": None},
        },
    }


def build_gpu_startup_receipt(
    *,
    campaign_id: str,
    scientific_pair_id: str,
    coordinator_epoch: str,
    implementation_identity_sha256: str,
    role: str,
    assigned_token: str,
    scheduler_visible_devices: str,
    visible_devices: str,
    local_index: int,
    pid: int,
    ppid: int,
    process_start_token: str,
    parent_start_token: str,
    gpu_uuid: str,
    gpu_model: str,
    gpu_total_memory_bytes: int,
    gpu_used_memory_bytes: int,
    cupy_version: str,
    cuda_runtime_version: int,
    representative_allocation_bytes: int,
    scheduler_job_id: str,
    hostname: str,
    started_utc: str,
) -> dict[str, Any]:
    """Build a startup receipt from values returned by the live GPU probe."""
    if role not in {"R", "C_OPT", "H0", "H1", "H2", "H3"}:
        raise ValueError(f"unknown GPU role: {role!r}")
    assigned = parse_cuda_visible_devices(assigned_token)
    visible = parse_cuda_visible_devices(visible_devices)
    if len(assigned) != 1 or visible != assigned:
        raise ValueError("child visibility must be exactly its assigned CUDA token")
    scheduler_tokens = parse_cuda_visible_devices(scheduler_visible_devices)
    if len(scheduler_tokens) not in GPU_COUNTS or assigned[0] not in scheduler_tokens:
        raise ValueError("assigned token is absent from the scheduler-visible allocation")
    if local_index != 0:
        raise ValueError("a single-token child mask must expose its GPU at local index 0")
    if any(not isinstance(value, str) or not value for value in (campaign_id, scientific_pair_id, coordinator_epoch)):
        raise ValueError("campaign, scientific pair, and coordinator epoch identifiers are required")
    if not isinstance(implementation_identity_sha256, str) or not _SHA256_RE.fullmatch(implementation_identity_sha256):
        raise ValueError("implementation_identity_sha256 must be 64 lowercase hexadecimal characters")
    if any(not isinstance(value, str) or not value for value in (process_start_token, parent_start_token, gpu_uuid, gpu_model,
                                                                 cupy_version, scheduler_job_id, hostname, started_utc)):
        raise ValueError("process start tokens and probed GPU identity must be non-empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (pid, ppid, gpu_total_memory_bytes)):
        raise ValueError("PID, PPID, and GPU VRAM must be positive integers")
    if (not isinstance(gpu_used_memory_bytes, int) or gpu_used_memory_bytes < 0
            or gpu_used_memory_bytes > gpu_total_memory_bytes
            or not isinstance(cuda_runtime_version, int) or cuda_runtime_version <= 0
            or not isinstance(representative_allocation_bytes, int) or representative_allocation_bytes <= 0):
        raise ValueError("GPU runtime memory/version probe is invalid")

    return {
        "schema": GPU_BINDING_SCHEMA,
        "receipt_type": "gpu_startup",
        "status": "PASS",
        "campaign_id": campaign_id,
        "scientific_pair_id": scientific_pair_id,
        "coordinator_epoch": coordinator_epoch,
        "implementation_identity_sha256": implementation_identity_sha256,
        "role": role,
        "worker_index": _worker_index(role),
        "scheduler_visible_devices": scheduler_visible_devices,
        "cuda_visible_devices": visible_devices,
        "scheduler_job_id": scheduler_job_id,
        "hostname": hostname,
        "started_utc": started_utc,
        "backend": "cupy/cuda",
        "cupy_import_status": "PASS",
        "cupy_version": cupy_version,
        "cuda_runtime_version": cuda_runtime_version,
        "representative_allocation_bytes": representative_allocation_bytes,
        "assignment": {
            "token": assigned[0],
            "scheduler_index": scheduler_tokens.index(assigned[0]),
            "local_index": local_index,
            "uuid": gpu_uuid,
        },
        "identity": {
            "process": {
                "pid": pid,
                "ppid": ppid,
                "start_token": process_start_token,
                "parent_start_token": parent_start_token,
            },
            "assigned_cuda_visible_device": assigned[0],
            "gpu": {
                "uuid": gpu_uuid,
                "model": gpu_model,
                "total_memory_bytes": gpu_total_memory_bytes,
                "used_memory_bytes": gpu_used_memory_bytes,
            },
        },
    }


def _as_receipt_sequence(value: Sequence[Mapping[str, Any]], label: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} receipts must be a sequence of receipt objects")
    receipts = tuple(value)
    if not receipts or any(not isinstance(receipt, Mapping) for receipt in receipts):
        raise ValueError(f"{label} receipts must be a non-empty sequence of objects")
    return receipts


def _index_gpu_receipts(
    receipts: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    seen_uuids: set[str] = set()
    seen_tokens: set[str] = set()

    for receipt in receipts:
        allowed_status = {"PASS"}
        allowed_type = {"gpu_startup"} if label == "startup" else {"gpu_final"}
        if receipt.get("schema") != GPU_BINDING_SCHEMA or receipt.get("status") not in allowed_status:
            raise ValueError(f"{label} contains an incompatible receipt status or schema")
        if receipt.get("receipt_type") not in allowed_type:
            raise ValueError(f"{label} contains an incompatible receipt type")
        if (receipt.get("backend") != "cupy/cuda"
                or receipt.get("cupy_import_status") != "PASS"
                or not receipt.get("cupy_version")
                or not isinstance(receipt.get("cuda_runtime_version"), int)
                or receipt.get("cuda_runtime_version") <= 0):
            raise ValueError(f"{label} receipt lacks a verified CUDA backend")
        role = receipt.get("role")
        if not isinstance(role, str) or role not in {"R", "C_OPT", "H0", "H1", "H2", "H3"}:
            raise ValueError(f"{label} contains an invalid role")
        if role in indexed:
            raise ValueError(f"{label} contains duplicate role {role}")
        if receipt.get("worker_index") != _worker_index(role):
            raise ValueError(f"{label} receipt for {role} has an invalid worker index")
        for field in ("campaign_id", "scientific_pair_id", "coordinator_epoch"):
            if not isinstance(receipt.get(field), str) or not receipt[field]:
                raise ValueError(f"{label} receipt for {role} is missing {field}")
        implementation_hash = receipt.get("implementation_identity_sha256")
        if not isinstance(implementation_hash, str) or not _SHA256_RE.fullmatch(implementation_hash):
            raise ValueError(f"{label} receipt for {role} has an invalid implementation identity hash")

        identity = receipt.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{label} receipt for {role} has no identity object")
        process = identity.get("process")
        gpu = identity.get("gpu")
        assignment = receipt.get("assignment")
        if not isinstance(process, Mapping) or not isinstance(gpu, Mapping) or not isinstance(assignment, Mapping):
            raise ValueError(f"{label} receipt for {role} has incomplete identity")
        required = (
            process.get("pid"),
            process.get("ppid"),
            process.get("start_token"),
            process.get("parent_start_token"),
            identity.get("assigned_cuda_visible_device"),
            gpu.get("uuid"),
            gpu.get("model"),
            gpu.get("total_memory_bytes"),
        )
        if any(value is None or value == "" for value in required):
            raise ValueError(f"{label} receipt for {role} has missing GPU/process identity")
        if (not isinstance(gpu.get("used_memory_bytes"), int)
                or gpu["used_memory_bytes"] < 0
                or gpu["used_memory_bytes"] > gpu["total_memory_bytes"]):
            raise ValueError(f"{label} receipt for {role} has invalid used VRAM")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in required[:2]):
            raise ValueError(f"{label} receipt for {role} has invalid PID/PPID")
        if isinstance(required[-1], bool) or not isinstance(required[-1], int) or required[-1] <= 0:
            raise ValueError(f"{label} receipt for {role} has invalid GPU VRAM")
        scheduler_devices = receipt.get("scheduler_visible_devices")
        scheduler_tokens = parse_cuda_visible_devices(scheduler_devices)
        if len(scheduler_tokens) not in GPU_COUNTS:
            raise ValueError(f"{label} receipt for {role} has an invalid scheduler allocation")
        role_to_token = assign_roles(scheduler_devices, len(scheduler_tokens))
        token = identity["assigned_cuda_visible_device"]
        token_list = parse_cuda_visible_devices(token)
        if len(token_list) != 1 or role_to_token.get(role) != token_list[0]:
            raise ValueError(f"{label} receipt for {role} disagrees with its scheduler role assignment")
        if receipt.get("scheduler_visible_devices") != scheduler_devices:
            raise ValueError(f"{label} receipt for {role} has inconsistent scheduler visibility")
        expected_assignment = {
            "token": token_list[0],
            "scheduler_index": scheduler_tokens.index(token_list[0]),
            "local_index": 0,
            "uuid": gpu.get("uuid"),
        }
        if dict(assignment) != expected_assignment:
            raise ValueError(f"{label} receipt for {role} has an inconsistent assignment identity")

        uuid_key = str(gpu["uuid"]).casefold()
        if uuid_key in seen_uuids:
            raise ValueError(f"{label} contains duplicate GPU UUIDs")
        seen_uuids.add(uuid_key)
        token_key = parse_cuda_visible_devices(str(identity["assigned_cuda_visible_device"]))[0]
        token_key = token_key.casefold()
        if token_key in seen_tokens:
            raise ValueError(f"{label} contains duplicate assigned CUDA tokens")
        seen_tokens.add(token_key)
        indexed[role] = receipt
    if indexed:
        first = next(iter(indexed.values()))
        shared_fields = (
            "campaign_id",
            "scientific_pair_id",
            "coordinator_epoch",
            "implementation_identity_sha256",
            "scheduler_visible_devices",
        )
        for receipt in indexed.values():
            if any(receipt[field] != first[field] for field in shared_fields):
                raise ValueError(f"{label} receipts disagree on the shared assignment contract")
    return indexed


def build_gpu_final_receipt(
    startup_receipt: Mapping[str, Any],
    *,
    exit_code: int,
    elapsed_seconds: float,
    error: Mapping[str, str] | None = None,
    final_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a final receipt that carries the startup process/device identity."""
    _index_gpu_receipts((startup_receipt,), "startup")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ValueError("exit_code must be an integer")
    if not isinstance(elapsed_seconds, (int, float)) or isinstance(elapsed_seconds, bool) or elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be a non-negative number")
    if error is not None and (not isinstance(error, Mapping) or any(not isinstance(value, str) for value in error.values())):
        raise ValueError("error must be an object containing string values")

    metadata = dict(final_metadata or {})
    for field in ("science_output_status", "writer_grant_status"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value or value == "NOT_REPORTED":
            metadata.setdefault(field, "NOT_REPORTED")
    writer_status = metadata.get("writer_grant_status")
    role = startup_receipt["role"]
    writer_closed = (writer_status == "NO_CHILD_GRANT_WRITERS_CLOSED" if role == "R" else
                     writer_status in {"CLOSED", "NO_CHILD_GRANT_WRITERS_CLOSED"} if role == "C_OPT" else
                     writer_status == "CLOSED")
    required_statuses_present = metadata.get("science_output_status") == "PASS" and writer_closed
    result = dict(startup_receipt)
    result.update(
        {
            "receipt_type": "gpu_final",
            "status": "PASS" if exit_code == 0 and required_statuses_present else "FAIL",
            "exit_code": exit_code,
            "elapsed_seconds": float(elapsed_seconds),
            "error": dict(error) if error is not None else None,
            "final_metadata": metadata,
        }
    )
    if not required_statuses_present:
        result["finalization_error"] = "science output or role-specific writer closure is not PASS"
    return result


def validate_startup_final_identities(
    startup_receipts: Sequence[Mapping[str, Any]],
    final_receipts: Sequence[Mapping[str, Any]],
) -> bool:
    """Require matching roles and unchanged process/device identity at finalization."""
    startups = _index_gpu_receipts(_as_receipt_sequence(startup_receipts, "startup"), "startup")
    finals = _index_gpu_receipts(_as_receipt_sequence(final_receipts, "final"), "final")
    if startups.keys() != finals.keys():
        raise ValueError("startup and final receipt role sets differ")

    for role, startup in startups.items():
        start_identity = startup["identity"]
        final_identity = finals[role]["identity"]
        start_gpu = start_identity["gpu"]
        final_gpu = final_identity["gpu"]
        fields = (
            (startup, finals[role], ("campaign_id", "scientific_pair_id", "coordinator_epoch", "implementation_identity_sha256", "worker_index", "scheduler_visible_devices", "backend", "cupy_version", "cuda_runtime_version", "scheduler_job_id", "hostname")),
            (startup["assignment"], finals[role]["assignment"], ("token", "scheduler_index", "local_index", "uuid")),
            (start_identity["process"], final_identity["process"], ("pid", "ppid", "start_token", "parent_start_token")),
            (start_identity, final_identity, ("assigned_cuda_visible_device",)),
            (start_gpu, final_gpu, ("uuid", "model", "total_memory_bytes")),
        )
        for start_values, final_values, keys in fields:
            for key in keys:
                first, second = start_values[key], final_values[key]
                if key == "uuid" and isinstance(first, str) and isinstance(second, str):
                    equal = first.casefold() == second.casefold()
                else:
                    equal = first == second
                if not equal:
                    raise ValueError(f"startup/final identity changed for role {role}: {key}")
    return True

"""Fail-closed, stdlib-first GPU child bootstrap for HR-4E-5 workers.

Example::

    python -m KHz_filament.hr4e5_gpu_bootstrap \
        --module KHz_filament.hr4e5_worker --role H0 --assigned-token 2 \
        --scheduler-visible-devices 0,1,2,3,4,5 --audit-root /scratch/audit \
        --campaign-root /scratch/campaign --campaign-id run-001 \
        --scientific-pair-id pair-001 --coordinator-epoch epoch-01 \
        --implementation-identity-sha256 <sha256> -- --worker-arg value

Only Python standard-library and the lightweight binding module are imported
before the single-token mask is checked. CuPy and KHz_filament.device are
loaded only after that check succeeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys
import tempfile
import time
from typing import Any

from KHz_filament.hr4e5_gpu_binding import (
    assign_roles,
    build_gpu_final_receipt,
    build_gpu_startup_receipt,
    parse_cuda_visible_devices,
)


_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_ROLE_NAMES = {"R", "C_OPT", "H0", "H1", "H2", "H3"}


def validate_child_device_assignment(
    role: str,
    visible_devices: str | None,
    assigned_token: str | None,
    scheduler_visible_devices: str | None,
) -> tuple[int, int]:
    """Validate child mask and coordinator contract; return scheduler/local indexes."""
    if role not in _ROLE_NAMES:
        raise ValueError(f"unknown GPU role: {role!r}")
    child_tokens = parse_cuda_visible_devices(visible_devices)
    assigned_tokens = parse_cuda_visible_devices(assigned_token)
    scheduler_tokens = parse_cuda_visible_devices(scheduler_visible_devices)
    if len(child_tokens) != 1 or len(assigned_tokens) != 1:
        raise ValueError("each child must receive exactly one CUDA_VISIBLE_DEVICES token")
    if child_tokens != assigned_tokens:
        raise ValueError("child CUDA_VISIBLE_DEVICES does not equal its assigned token")
    if len(scheduler_tokens) not in (4, 6):
        raise ValueError("scheduler allocation must contain exactly 4 or 6 visible devices")
    role_to_token = assign_roles(scheduler_visible_devices, len(scheduler_tokens))
    if role not in role_to_token or role_to_token[role] != assigned_tokens[0]:
        raise ValueError("child role/token pair disagrees with the coordinator assignment")
    return scheduler_tokens.index(assigned_tokens[0]), 0


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_receipt_paths(
    *,
    audit_root: str | os.PathLike[str],
    campaign_root: str | os.PathLike[str],
    role: str,
    startup_receipt: str | os.PathLike[str] | None = None,
    final_receipt: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve receipt paths and require an audit root disjoint from campaign storage."""
    if role not in _ROLE_NAMES:
        raise ValueError(f"unknown GPU role: {role!r}")
    audit = Path(audit_root).expanduser().resolve()
    campaign = Path(campaign_root).expanduser().resolve()
    if _is_within(audit, campaign) or _is_within(campaign, audit):
        raise ValueError("audit root and campaign root must be separate, non-nested paths")

    def resolve_one(value: str | os.PathLike[str] | None, suffix: str) -> Path:
        candidate = Path(value).expanduser() if value is not None else Path(f"{role}.{suffix}.json")
        if not candidate.is_absolute():
            candidate = audit / candidate
        result = candidate.resolve()
        if not _is_within(result, audit):
            raise ValueError("startup/final receipt must be located inside the audit root")
        return result

    startup_path = resolve_one(startup_receipt, "startup")
    final_path = resolve_one(final_receipt, "final")
    if startup_path == final_path:
        raise ValueError("startup and final receipts must use distinct paths")
    return audit, startup_path, final_path


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_durable_receipt(path: str | os.PathLike[str], receipt: dict[str, Any]) -> None:
    """Atomically persist one JSON receipt without overwriting existing evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing receipt: {target}") from exc
        temporary_path.unlink()
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _proc_start_token(pid: int) -> str:
    """Read Linux /proc starttime (field 22), handling spaces in comm."""
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read process start token for PID {pid}") from exc
    closing_paren = raw.rfind(")")
    if closing_paren < 0:
        raise RuntimeError(f"malformed /proc stat for PID {pid}")
    fields_after_comm = raw[closing_paren + 1 :].split()
    # The suffix starts at field 3 (state), so field 22 (starttime) is index 19.
    if len(fields_after_comm) <= 19:
        raise RuntimeError(f"/proc stat for PID {pid} lacks starttime")
    token = fields_after_comm[19]
    if not token.isdecimal():
        raise RuntimeError(f"/proc stat for PID {pid} has an invalid starttime")
    return token


def read_process_identity() -> dict[str, Any]:
    pid = os.getpid()
    ppid = os.getppid()
    if pid <= 0 or ppid <= 0:
        raise RuntimeError("cannot establish positive process PID/PPID")
    return {
        "pid": pid,
        "ppid": ppid,
        "process_start_token": _proc_start_token(pid),
        "parent_start_token": _proc_start_token(ppid),
    }


def _decode_text(value: Any, field: str) -> str:
    if isinstance(value, bytes):
        result = value.decode("utf-8", errors="strict").rstrip("\0")
    else:
        result = str(value).rstrip("\0")
    if not result:
        raise RuntimeError(f"GPU runtime returned an empty {field}")
    return result


def _format_gpu_uuid(value: Any) -> str:
    import uuid

    if isinstance(value, bytes):
        if len(value) == 16:
            return "GPU-" + str(uuid.UUID(bytes=value))
    if isinstance(value, (bytearray, list, tuple)) and len(value) == 16:
        return "GPU-" + str(uuid.UUID(bytes=bytes(value)))
    text = _decode_text(value, "GPU UUID")
    if text.startswith("GPU-"):
        return text
    try:
        return "GPU-" + str(uuid.UUID(text))
    except ValueError as exc:
        raise RuntimeError("CUDA runtime returned an unrecognized GPU UUID") from exc


def probe_gpu_runtime() -> dict[str, Any]:
    """Import CuPy/device after mask validation and fail closed on any CPU fallback."""
    import cupy as cp

    from KHz_filament import device

    if not getattr(device, "USE_GPU", False) or getattr(device, "xp", None) is not cp:
        raise RuntimeError("KHz_filament.device did not initialize the CuPy backend")
    runtime = cp.cuda.runtime
    if int(runtime.getDeviceCount()) != 1:
        raise RuntimeError("single-token CUDA mask did not expose exactly one GPU")
    cp.cuda.Device(0).use()
    if int(runtime.getDevice()) != 0:
        raise RuntimeError("CUDA runtime did not bind the child to local GPU index 0")

    properties = runtime.getDeviceProperties(0)
    if not isinstance(properties, dict):
        properties = dict(properties)
    gpu_model = _decode_text(properties.get("name"), "GPU model")
    memory_bytes = int(properties.get("totalGlobalMem", 0))
    if memory_bytes <= 0:
        raise RuntimeError("GPU runtime returned invalid total VRAM")
    probe = cp.empty((1024,), dtype=cp.uint8)
    runtime.deviceSynchronize()
    free_bytes, runtime_total_bytes = runtime.memGetInfo()
    if int(runtime_total_bytes) <= 0:
        raise RuntimeError("GPU runtime memory probe returned invalid total VRAM")
    del probe

    gpu_uuid_value = properties.get("uuid")
    if gpu_uuid_value is None:
        uuid_reader = getattr(runtime, "deviceGetUuid", None)
        if not callable(uuid_reader):
            raise RuntimeError("CuPy runtime does not expose a GPU UUID probe")
        gpu_uuid_value = uuid_reader(0)
    gpu_uuid = _format_gpu_uuid(gpu_uuid_value)
    return {
        "gpu_uuid": gpu_uuid,
        "gpu_model": gpu_model,
        "gpu_total_memory_bytes": memory_bytes,
        "gpu_used_memory_bytes": max(0, memory_bytes - int(free_bytes)),
        "cupy_version": str(cp.__version__),
        "cuda_runtime_version": int(runtime.runtimeGetVersion()),
        "representative_allocation_bytes": 1024,
        "local_index": 0,
    }


def _parse_final_metadata(raw_value: str | None) -> dict[str, Any]:
    if not raw_value:
        return {}
    result = json.loads(raw_value)
    if not isinstance(result, dict):
        raise ValueError("HR4E5_GPU_FINAL_METADATA_JSON must contain a JSON object")
    return result


def update_final_metadata(**fields: Any) -> None:
    """Update final receipt metadata from the worker running in this process."""
    current = _parse_final_metadata(os.environ.get("HR4E5_GPU_FINAL_METADATA_JSON"))
    current.update(fields)
    os.environ["HR4E5_GPU_FINAL_METADATA_JSON"] = json.dumps(
        current, sort_keys=True, ensure_ascii=False, allow_nan=False
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default=os.environ.get("HR4E5_GPU_MODULE"))
    parser.add_argument("--role", default=os.environ.get("HR4E5_GPU_ROLE"))
    parser.add_argument("--assigned-token", default=os.environ.get("HR4E5_GPU_ASSIGNED_TOKEN"))
    parser.add_argument("--scheduler-visible-devices", default=os.environ.get("HR4E5_SCHEDULER_VISIBLE_DEVICES"))
    parser.add_argument("--audit-root", default=os.environ.get("HR4E5_GPU_AUDIT_ROOT"))
    parser.add_argument("--campaign-root", default=os.environ.get("HR4E5_CAMPAIGN_ROOT"))
    parser.add_argument("--startup-receipt", default=os.environ.get("HR4E5_GPU_STARTUP_RECEIPT"))
    parser.add_argument("--final-receipt", default=os.environ.get("HR4E5_GPU_FINAL_RECEIPT"))
    parser.add_argument("--campaign-id", default=os.environ.get("HR4E5_CAMPAIGN_ID"))
    parser.add_argument("--scientific-pair-id", default=os.environ.get("HR4E5_SCIENTIFIC_PAIR_ID"))
    parser.add_argument("--coordinator-epoch", default=os.environ.get("HR4E5_COORDINATOR_EPOCH"))
    parser.add_argument(
        "--implementation-identity-sha256",
        default=os.environ.get("HR4E5_IMPLEMENTATION_IDENTITY_SHA256"),
    )
    parser.add_argument("--final-metadata-json", default=os.environ.get("HR4E5_GPU_FINAL_METADATA_JSON"))
    return parser


def _parse_invocation(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        separator = argv.index("--")
        options, module_argv = argv[:separator], argv[separator + 1 :]
    else:
        options, module_argv = argv, []
    return _parser().parse_args(options), module_argv


def _require_text(value: str | None, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is required")
    return value


def _exit_code(error: BaseException | None) -> int:
    if error is None:
        return 0
    if isinstance(error, SystemExit):
        code = error.code
        if code is None:
            return 0
        if isinstance(code, int) and not isinstance(code, bool):
            return code
    return 1


def main(argv: list[str] | None = None) -> int:
    options, module_argv = _parse_invocation(list(sys.argv[1:] if argv is None else argv))
    module_name = _require_text(options.module, "--module")
    if not _MODULE_RE.fullmatch(module_name) or module_name == __name__:
        raise ValueError("--module must be a qualified Python module name")
    role = _require_text(options.role, "--role")
    assigned_token = _require_text(options.assigned_token, "--assigned-token")
    scheduler_visible_devices = _require_text(options.scheduler_visible_devices, "--scheduler-visible-devices")
    campaign_id = _require_text(options.campaign_id, "--campaign-id")
    scientific_pair_id = _require_text(options.scientific_pair_id, "--scientific-pair-id")
    coordinator_epoch = _require_text(options.coordinator_epoch, "--coordinator-epoch")
    implementation_identity_sha256 = _require_text(
        options.implementation_identity_sha256, "--implementation-identity-sha256"
    )
    audit_root = _require_text(options.audit_root, "--audit-root")
    campaign_root = _require_text(options.campaign_root, "--campaign-root")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    scheduler_index, local_index = validate_child_device_assignment(
        role, visible_devices, assigned_token, scheduler_visible_devices
    )
    audit, startup_path, final_path = resolve_receipt_paths(
        audit_root=audit_root,
        campaign_root=campaign_root,
        role=role,
        startup_receipt=options.startup_receipt,
        final_receipt=options.final_receipt,
    )
    if os.environ.get("UPPE_USE_GPU", "1") != "1":
        raise RuntimeError("UPPE_USE_GPU must be enabled for GPU bootstrap")
    os.environ["UPPE_USE_GPU"] = "1"
    os.environ["HR4E5_GPU_FINAL_METADATA_JSON"] = options.final_metadata_json or "{}"
    _parse_final_metadata(os.environ["HR4E5_GPU_FINAL_METADATA_JSON"])

    gpu_identity = probe_gpu_runtime()
    if gpu_identity["local_index"] != local_index:
        raise RuntimeError("GPU runtime local index differs from the validated single-token mask")
    process_identity = read_process_identity()
    startup_receipt = build_gpu_startup_receipt(
        campaign_id=campaign_id,
        scientific_pair_id=scientific_pair_id,
        coordinator_epoch=coordinator_epoch,
        implementation_identity_sha256=implementation_identity_sha256,
        role=role,
        assigned_token=assigned_token,
        scheduler_visible_devices=scheduler_visible_devices,
        visible_devices=visible_devices or "",
        local_index=local_index,
        pid=process_identity["pid"],
        ppid=process_identity["ppid"],
        process_start_token=process_identity["process_start_token"],
        parent_start_token=process_identity["parent_start_token"],
        gpu_uuid=gpu_identity["gpu_uuid"],
        gpu_model=gpu_identity["gpu_model"],
        gpu_total_memory_bytes=gpu_identity["gpu_total_memory_bytes"],
        gpu_used_memory_bytes=gpu_identity["gpu_used_memory_bytes"],
        cupy_version=gpu_identity["cupy_version"],
        cuda_runtime_version=gpu_identity["cuda_runtime_version"],
        representative_allocation_bytes=gpu_identity["representative_allocation_bytes"],
        scheduler_job_id=_require_text(os.environ.get("SLURM_JOB_ID"), "SLURM_JOB_ID"),
        hostname=socket.gethostname(),
        started_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    startup_receipt["assignment"]["scheduler_index"] = scheduler_index
    write_durable_receipt(startup_path, startup_receipt)

    started = time.monotonic()
    pending_error: BaseException | None = None
    try:
        sys.argv = [module_name, *module_argv]
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    except BaseException as exc:
        pending_error = exc
    finally:
        exit_code = _exit_code(pending_error)
        error_record = None
        if pending_error is not None:
            error_record = {
                "type": type(pending_error).__name__,
                "message": str(pending_error),
            }
        try:
            final_metadata = _parse_final_metadata(os.environ.get("HR4E5_GPU_FINAL_METADATA_JSON"))
            final_receipt = build_gpu_final_receipt(
                startup_receipt,
                exit_code=exit_code,
                elapsed_seconds=time.monotonic() - started,
                error=error_record,
                final_metadata=final_metadata,
            )
        except Exception as exc:
            final_receipt = build_gpu_final_receipt(
                startup_receipt,
                exit_code=1,
                elapsed_seconds=time.monotonic() - started,
                error={"type": type(exc).__name__, "message": str(exc)},
                final_metadata={
                    "science_output_status": "NOT_REPORTED",
                    "writer_grant_status": "NOT_REPORTED",
                },
            )
            final_receipt["finalization_error"] = f"invalid final metadata: {exc}"
        write_durable_receipt(final_path, final_receipt)

    if pending_error is not None:
        raise pending_error
    if final_receipt["status"] != "PASS":
        raise RuntimeError("GPU final receipt did not meet the science/writer PASS gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

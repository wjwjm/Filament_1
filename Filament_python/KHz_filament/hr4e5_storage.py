"""Bounded, fail-closed storage bookkeeping for the E5-1A fixture.

This module owns no scientific state.  It provides a small durable byte ledger,
cross-process reservations, and an explicit whitelist based reclamation
transaction for files created below one campaign root.  The implementation is
intentionally local to the E5-1A glue; it is not a general storage service.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


HARD_CAP_BYTES = 300 * 1024**3
DEFAULT_FINAL_OUTPUT_BUDGET_BYTES = 64 * 1024**3
DEFAULT_SAFETY_MARGIN_BYTES = 8 * 1024**3
BUDGET_OBJECT_SCHEMA = "khz_filament.hr4e5.e5_1b.budget_objects.v2"
PROCESS_RECEIPT_SCHEMA = "khz_filament.hr4e5.e5_1b.wp2.process_receipt.v1"
STORAGE_SCHEMA = "khz_filament.hr4e5.e5_1a.storage.v1"
GC_SCHEMA = "khz_filament.hr4e5.e5_1a.gc.v1"
RECLAIM_PREREQUISITE_SCHEMA = "khz_filament.hr4e5.e5_1a.reclaim_prerequisites.v1"
INTENT_SCHEMA = "khz_filament.hr4e5.e5_1a.creation_intent.v1"
WRITER_SCHEMA = "khz_filament.hr4e5.e5_1a.writer.v1"
WRITER_RECEIPT_SCHEMA = "khz_filament.hr4e5.e5_1a.writer_receipt.v1"
CHILD_WRITER_GRANT_SCHEMA = "khz_filament.hr4e5.e5_1b.child_writer_grant.v1"
CHILD_GRANT_STATES = frozenset({
    "PENDING", "BOUND_ACTIVE", "CLOSED", "INTERRUPTED", "INTERRUPTED_STALE", "ABANDONED",
})
CHILD_GRANT_OPERATIONS = {
    "OPTICAL": ("OPTICAL_OUTPUT", "POST_WRITE"),
    "HYDRO": ("NEXT_COMMIT",),
}


def _host_identity() -> str:
    if sys.platform.startswith("linux"):
        boot = Path("/proc/sys/kernel/random/boot_id")
        if boot.is_file():
            return f"{socket.gethostname()}:{boot.read_text(encoding='ascii').strip()}"
    return socket.gethostname()


def _process_start_token(pid: int) -> str | None:
    """Return an OS process-instance token, not merely a reusable PID."""
    if pid <= 0:
        return None
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                                  ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                                  ctypes.POINTER(wintypes.FILETIME))
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            try:
                exit_code = wintypes.DWORD()
                if (not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                        or int(exit_code.value) != 259):  # STILL_ACTIVE
                    return None
                creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
                if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                                ctypes.byref(kernel), ctypes.byref(user)):
                    return None
                return str((int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime))
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii").split()
            return fields[21] if len(fields) > 21 else None
        except (OSError, ValueError):
            return None
    return None


def process_identity(pid: int | None = None) -> dict[str, Any]:
    value = int(os.getpid() if pid is None else pid)
    token = _process_start_token(value)
    if token is None:
        raise StorageIntegrityError("process start identity is unavailable")
    return {"pid": value, "host_identity": _host_identity(), "process_start_token": token}


def probe_process_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a recorded process as LIVE, DEAD, or UNKNOWN fail-closed."""
    if str(identity.get("host_identity", "")) != _host_identity():
        return {"status": "UNKNOWN", "reason": "different_host"}
    try:
        pid = int(identity["pid"]); expected = str(identity["process_start_token"])
    except (KeyError, TypeError, ValueError):
        return {"status": "UNKNOWN", "reason": "invalid_identity"}
    actual = _process_start_token(pid)
    if actual is None:
        return {"status": "DEAD", "reason": "pid_absent", "pid": pid}
    if str(actual) != expected:
        return {"status": "DEAD", "reason": "pid_reused", "pid": pid,
                "observed_process_start_token": str(actual)}
    return {"status": "LIVE", "reason": "identity_matches", "pid": pid}


def budget_object_table(*, n_pulses: int, k: int, ny: int, nx: int, nt: int,
                        safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
                        external_prefix_already_present: bool = False,
                        existing_full_source_bytes: int = 0,
                        canonical_source_artifact_bytes: int = 0,
                        canonical_source_manifest_allowance_bytes: int = 0,
                        canonical_source_write_temporary_bytes: int = 0,
                        frozen_lut_input_bytes: int = 0) -> dict[str, Any]:
    """Scalar, versioned objects used by every E5 campaign budget projection."""
    values = (n_pulses, k, ny, nx, nt)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("positive integer dimensions and pulse count required")
    margin = _nonnegative_int(safety_margin_bytes, "safety_margin_bytes")
    canonical_artifact = _nonnegative_int(
        canonical_source_artifact_bytes, "canonical_source_artifact_bytes")
    canonical_manifest = _nonnegative_int(
        canonical_source_manifest_allowance_bytes, "canonical_source_manifest_allowance_bytes")
    canonical_temporary = _nonnegative_int(
        canonical_source_write_temporary_bytes, "canonical_source_write_temporary_bytes")
    frozen_lut_inputs = _nonnegative_int(frozen_lut_input_bytes, "frozen_lut_input_bytes")
    if margin <= 0:
        raise StorageBudgetError("budget margin must be positive")
    if bool(canonical_artifact) != bool(canonical_manifest) or (
            canonical_artifact and canonical_temporary < max(canonical_artifact, canonical_manifest)):
        raise StorageBudgetError("canonical source storage ledger is incomplete")
    field = int(k) * int(ny) * int(nx) * 8
    generation = 3 * field
    optical = int(nt) * int(ny) * int(nx) * 16
    metadata = max(2 * 1024**3, (2 * 1024**3 * int(n_pulses) + 2) // 3)
    stages: dict[str, list[dict[str, Any]]] = {}

    def add(stage: str, name: str, count: int, each: int, *, owner: str,
            lifetime: str, kind: str = "campaign_disk") -> None:
        stages.setdefault(stage, []).append({
            "object": name, "owner": owner, "kind": kind,
            "count": int(count), "bytes_each": int(each), "bytes": int(count * each),
            "lifetime": lifetime, "capacity_domain": "campaign_filesystem",
        })

    # A historical R=7G/C=5G envelope already included six sinks per
    # trajectory (2G each).  Split the real containers before summing:
    # nonfinal R has 3G Streaming + 2G HR4C slots; C has 3G Streaming.
    # Both tracks together have 12 sink files = 4G, not another 12G.
    # Counts are independent physical containers: a Streaming generation is
    # K NPZ files, while each HR4C slot is three NPY files.  Namespace labels
    # in a manifest never create another counted copy.
    windows = [("initialize", (1, 0, 0, 1, 0, 0, 0, 0, 0), 0, 0, 0)]
    for pulse in range(n_pulses):
        first = 2 * pulse
        nonfinal = pulse < n_pulses - 1
        r_window = (1, 1, 1 if nonfinal else 0, 1, 0, 0, 2 if nonfinal else 0, 0, 0)
        pair_window = (1, 1, 1 if nonfinal else 0, 1, 1, 1 if nonfinal else 0, 2 if nonfinal else 0, 0, 0)
        windows.append((f"p{pulse}_r_optical_hydro", r_window, first + 1, 0, 1))
        windows.append((f"p{pulse}_c_streaming", pair_window, first + 2, 0, 2))
        windows.append((f"p{pulse}_pair_exact", pair_window, first + 2, 0, 2))
        if pulse < n_pulses - 1:
            windows.extend((
                (f"p{pulse}_r_successor_ready", (1, 1, 1, 1, 1, 1, 2, 1, 0), first + 2, 0, 2),
                (f"p{pulse}_failed_handoff_stop", (1, 1, 1, 1, 1, 1, 2, 1, 0), first + 2, 1, 2),
                (f"p{pulse}_r_gc_complete", (0, 0, 0, 1, 1, 1, 0, 1, 0), first + 2, 0, 1),
                (f"p{pulse}_c_successor_ready", (0, 0, 0, 1, 1, 1, 0, 1, 1), first + 2, 0, 1),
                (f"p{pulse}_c_gc_complete", (0, 0, 0, 0, 0, 0, 0, 1, 1), first + 2, 0, 0),
            ))
        else:
            windows.extend((
                ("terminal_ready", pair_window, first + 2, 0, 2),
                ("terminal_reference_reclaimed", (0, 0, 0, 1, 1, 0, 0, 0, 0), first + 2, 0, 1),
                ("final_retention", (0, 0, 0, 1, 1, 0, 0, 0, 0), first + 2, 0, 0),
                ("reopen", (0, 0, 0, 1, 1, 0, 0, 0, 0), first + 2, 0, 0),
            ))
    slow_roles = ("r_parent_current", "r_parent_post", "r_parent_next",
                  "c_parent_current", "c_parent_post", "c_parent_next",
                  "r_hr4c_slots", "r_child_current", "c_child_current")
    for stage, slow_counts, optical_count, failed_count, sink_outputs in windows:
        for role, count in zip(slow_roles, slow_counts):
            add(stage, role, count, generation, owner=role.split("_")[0].upper(),
                lifetime="exact_READY_quiescence_then_whitelist_GC")
        add(stage, "failed_generation_residue", failed_count, generation,
            owner="failed_handoff", lifetime="retained_on_failure")
        add(stage, "persistent_final_optical", optical_count, optical,
            owner="R_and_C_optical", lifetime="retained")
        add(stage, "full_interval_sink_arrays", 6 * sink_outputs, field,
            owner="R_and_C_optical", lifetime="paired_exact_then_explicit_gated_reclaim")
        add(stage, "external_delta_n_prefix", 1, field,
            owner="input_materialization", lifetime="through_terminal_review",
            kind="external_new_or_preexisting")
        add(stage, "ledger_metadata_and_containers", 1, metadata,
            owner="campaign", lifetime="retained_allowance", kind="allowance")
        if canonical_artifact:
            add(stage, "canonical_source_artifact", 1, canonical_artifact,
                owner="input_materialization", lifetime="retained_through_review",
                kind="external_campaign_input")
            add(stage, "canonical_source_sealed_manifest", 1, canonical_manifest,
                owner="input_materialization", lifetime="retained_through_review",
                kind="external_campaign_input_allowance")
    # Legacy CPU fixtures rebuild E_source in RAM.  GPU/formal WP2 inputs bind
    # one retained canonical artifact and load a private backend buffer per
    # pulse; that external file and its seal are charged below.  The writer
    # allowances remain disk headroom, not additional final optical files.
    for stage in stages:
        if stage != "final_retention":
            add(stage, "optical_writer_temporary", 1, optical,
                owner="optical_writer", lifetime="during_write_or_failed_partial", kind="allowance")
            # During Candidate pulse 0, one optical child is still committing
            # POST/sink data while hydro children commit NEXT files.  This is
            # one shared-reservation ownership sharding window, not one extra
            # full-z generation per worker.
            add(stage, "candidate_optical_hydro_overlap_temporary", 1, optical,
                owner="C_OPTICAL_AND_C_HYDRO", lifetime="candidate_overlap_window",
                kind="allowance")
            add(stage, "conservative_optical_overlap", 3, optical,
                owner="campaign", lifetime="peak_only", kind="allowance")
            add(stage, "safety_margin", 1, margin,
                owner="campaign", lifetime="unallocated_headroom", kind="allowance")
    if canonical_artifact:
        # Input generation precedes campaign writes.  Track that phase as its
        # own peak so the one atomic .npy temporary is not charged alongside
        # the later campaign output peak.
        add("canonical_source_materialization", "canonical_source_artifact", 1,
            canonical_artifact, owner="input_materialization",
            lifetime="retained_through_review", kind="external_campaign_input")
        add("canonical_source_materialization", "canonical_source_sealed_manifest", 1,
            canonical_manifest, owner="input_materialization",
            lifetime="retained_through_review", kind="external_campaign_input_allowance")
        add("canonical_source_materialization", "canonical_source_atomic_write_temporary", 1,
            canonical_temporary, owner="input_materialization",
            lifetime="single_atomic_publish_peak", kind="external_temporary")
    if frozen_lut_inputs:
        for stage in stages:
            add(stage, "frozen_lut_inputs_and_staging_receipt", 1, frozen_lut_inputs,
                owner="input_materialization", lifetime="retained_through_review",
                kind="preexisting_immutable_campaign_input")
    totals = {stage: sum(row["bytes"] for row in rows) for stage, rows in stages.items()}
    peak_stage = max((stage for stage in stages if stage != "final_retention"), key=totals.__getitem__)
    return {
        "schema": BUDGET_OBJECT_SCHEMA, "stages": stages, "stage_bytes": totals,
        "peak_stage": peak_stage, "peak_bytes": totals[peak_stage],
        "final_output_bytes": totals["final_retention"],
        "field_bytes": field, "generation_bytes": generation, "optical_bytes": optical,
        "metadata_bytes": metadata, "safety_margin_bytes": margin,
        "external_prefix_bytes": field,
        "external_prefix_already_present": bool(external_prefix_already_present),
        "file_templates": [
            {"object_id": "{side}:p{pulse}:{namespace}:screen_{index}",
             "path_template": "{side}/p{pulse}/state/{namespace}/screen_{index:06d}.npz",
             "container": "npz", "members": ["delta_n", "vx", "vy", "metadata_json"],
             "shape_each_field": [ny, nx], "dtype": "float64", "payload_bytes_each": 3*ny*nx*8,
             "count_per_generation": k, "header_and_metadata": "ledger_metadata_and_containers"},
            {"object_id": "R:p{pulse}:hr4c:{slot}:{field}",
             "path_template": "R/p{pulse}/batch/store.hr4c_{field}_{slot}.npy",
             "container": "npy", "shape": [k, ny, nx], "dtype": "float64",
             "payload_bytes_each": field, "count_per_nonfinal_r": 6},
            {"object_id": "{side}:p{pulse}:sink:{name}",
             "path_template": "{side}/p{pulse}/optical/{filename}",
             "container": "npy", "shape": [k, ny, nx], "dtype": "float64",
             "payload_bytes_each": field, "count_per_track_pulse": 6,
             "reclaim_prerequisite": "paired_actual_exact_and_ready_and_quiescence"},
            {"object_id": "{side}:p{pulse}:final_optical",
             "path_template": "{side}/p{pulse}/optical/final_optical_field.npy",
             "container": "npy", "shape": [nt, ny, nx], "dtype": "complex128",
             "payload_bytes_each": optical, "count_per_track_pulse": 1,
             "retained_final": True},
            {"object_id": "external_delta_n_prefix", "path_template": "bound PRE0 path",
             "container": "npy", "shape": [k, ny, nx], "dtype": "float64",
             "payload_bytes_each": field, "count": 1,
             "capacity_domain": "campaign_filesystem", "retained_final": True,
            "root_inside_charge": "actual root bytes", "root_outside_charge": "external_prefix_bytes"},
            *([{"object_id": "frozen_lut_inputs_and_staging_receipt",
                "path_template": "cache/rate_tables/*.npz and N5_LUT_STAGING_RECEIPT.json",
                "container": "npz/json", "file_bytes": frozen_lut_inputs,
                "capacity_domain": "campaign_filesystem", "retained_final": True}]
              if frozen_lut_inputs else []),
            {"object_id": "canonical_e_source", "path_template": "bound canonical E_source input path",
             "container": "npy", "shape": [nt, ny, nx], "dtype": "<c16",
             "payload_bytes_each": optical, "count": 1,
             "file_bytes": canonical_artifact, "manifest_allowance_bytes": canonical_manifest,
             "write_temporary_peak_bytes": canonical_temporary,
             "capacity_domain": "campaign_filesystem", "retained_final": True,
             "legacy_cpu_fixture": "not_materialized"},
            {"object_id": "{side}:p{pulse}:scientific_ledger",
             "path_template": "{side}/p{pulse}/optical/scientific_ledger.npz",
             "container": "npz", "members": 9, "shape_each_member": [k],
             "dtype": "float64", "payload_bytes_each": 9*k*8,
             "count_per_track_pulse": 1, "allowance": "ledger_metadata_and_containers",
             "retained_final": True},
            {"object_id": "exact_and_ready_gc_writer_evidence",
             "path_template": "exact/p{pulse}.json and per-track receipt paths",
             "container": "json/npy", "allowance": "ledger_metadata_and_containers",
             "retained_final": True},
            {"object_id": "candidate_optical_hydro_overlap_temporary",
             "path_template": "C/p{pulse}/optical and C/p{pulse}/state/next temporary objects",
             "container": "npy/npz", "payload_bytes_each": optical, "count_peak": 1,
             "owner": "C_OPTICAL_AND_C_HYDRO", "capacity_domain": "campaign_filesystem",
             "lifetime": "candidate_overlap_window", "retained_final": False,
             "multiprocess_shared_reservation": True},
            {"object_id": "atomic_and_incomplete_write", "path_template": "*.tmp",
             "container": "temporary", "allowance": "optical_writer_temporary",
             "retained_final": False},
            {"object_id": "failed_generation_residue", "path_template": "failed attempt files",
             "container": "npy/npz", "payload_bytes_each": generation,
             "count_peak": 1, "retained_final": False},
        ],
        "incremental_requirement_bytes": totals[peak_stage] - (
            (field if external_prefix_already_present else 0) + canonical_artifact + frozen_lut_inputs),
        "existing_full_source_bytes": _nonnegative_int(existing_full_source_bytes, "existing_full_source_bytes"),
        "existing_full_source_campaign_charge_bytes": 0,
        "optical_e_source_ram_bytes": optical,
        "legacy_cpu_source_rebuild": canonical_artifact == 0,
        "canonical_source_artifact_bytes": canonical_artifact,
        "canonical_source_manifest_allowance_bytes": canonical_manifest,
        "canonical_source_write_temporary_bytes": canonical_temporary,
        **({"frozen_lut_input_bytes": frozen_lut_inputs} if frozen_lut_inputs else {}),
        "capacity_domain": "campaign_filesystem",
    }


def plan_campaign_budget(*, n_pulses: int, k: int, ny: int, nx: int, nt: int,
                         max_campaign_live_bytes: int = HARD_CAP_BYTES,
                         final_output_budget_bytes: int = DEFAULT_FINAL_OUTPUT_BUDGET_BYTES,
                         safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
                         external_prefix_already_present: bool = False,
                         existing_full_source_bytes: int = 0,
                         canonical_source_artifact_bytes: int = 0,
                         canonical_source_manifest_allowance_bytes: int = 0,
                         canonical_source_write_temporary_bytes: int = 0,
                         frozen_lut_input_bytes: int = 0) -> dict[str, Any]:
    """Apply the unchanged policy caps to the shared object table."""
    cap = _nonnegative_int(max_campaign_live_bytes, "max_campaign_live_bytes")
    final_cap = _nonnegative_int(final_output_budget_bytes, "final_output_budget_bytes")
    if (cap <= 0 or cap > HARD_CAP_BYTES or final_cap <= 0
            or final_cap > DEFAULT_FINAL_OUTPUT_BUDGET_BYTES or final_cap > cap):
        raise StorageBudgetError("invalid campaign budget limits")
    objects = budget_object_table(
        n_pulses=n_pulses, k=k, ny=ny, nx=nx, nt=nt,
        safety_margin_bytes=safety_margin_bytes,
        external_prefix_already_present=external_prefix_already_present,
        existing_full_source_bytes=existing_full_source_bytes,
        canonical_source_artifact_bytes=canonical_source_artifact_bytes,
        canonical_source_manifest_allowance_bytes=canonical_source_manifest_allowance_bytes,
        canonical_source_write_temporary_bytes=canonical_source_write_temporary_bytes,
        frozen_lut_input_bytes=frozen_lut_input_bytes,
    )
    peak, final_bytes = objects["peak_bytes"], objects["final_output_bytes"]
    if peak > cap:
        raise StorageBudgetError(
            f"campaign allocation model exceeds cap: projected={peak} cap={cap}"
        )
    if final_bytes > final_cap:
        raise StorageBudgetError(
            f"campaign final retention model exceeds final budget: projected={final_bytes} cap={final_cap}"
        )
    return {
        **objects, "schema": "khz_filament.hr4e5.e5_1b.budget_plan.v1",
        "object_schema": BUDGET_OBJECT_SCHEMA,
        "n_pulses": int(n_pulses), "k": int(k), "ny": int(ny), "nx": int(nx), "nt": int(nt),
        "retained_optical_bytes": 2 * int(n_pulses) * objects["optical_bytes"],
        "canonical_source_retained_bytes": (
            objects["canonical_source_artifact_bytes"]
            + objects["canonical_source_manifest_allowance_bytes"]),
        "source_copy_bytes": 0,
        "failed_handoff_residue_bytes": objects["generation_bytes"] if n_pulses > 1 else 0,
        "peak_bytes": peak, "final_output_bytes": final_bytes,
        "max_campaign_live_bytes": cap, "final_output_budget_bytes": final_cap,
        "headroom_bytes": cap - peak, "final_headroom_bytes": final_cap - final_bytes,
        **({"frozen_lut_input_bytes": objects["frozen_lut_input_bytes"]}
           if objects.get("frozen_lut_input_bytes", 0) else {}),
        "k8048_arrays_created": False,
    }


derive_budget_plan = plan_campaign_budget


def plan_final_output(*, n_pulses: int, k: int, ny: int, nx: int, nt: int,
                      diagnostics: Sequence[str] = (), checkpoints: int = 0,
                      final_budget_bytes: int = DEFAULT_FINAL_OUTPUT_BUDGET_BYTES,
                      existing_full_source_bytes: int = 0) -> dict[str, Any]:
    """Explicit bounded output policy; scalar planning only, never allocations."""
    values = (n_pulses, k, ny, nx, nt)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError('positive integer dimensions and pulse count required')
    if checkpoints < 0 or len(set(diagnostics)) != len(diagnostics) or not set(diagnostics) <= {'ion','ib','raman','qthermal','increment','state_after'}:
        raise ValueError('invalid finite output selection')
    if not 0 < final_budget_bytes <= DEFAULT_FINAL_OUTPUT_BUDGET_BYTES:
        raise StorageBudgetError('invalid final output cap')
    objects = budget_object_table(n_pulses=n_pulses, k=k, ny=ny, nx=nx, nt=nt,
                                  existing_full_source_bytes=existing_full_source_bytes)
    field, generation, optical = objects["field_bytes"], objects["generation_bytes"], objects["optical_bytes"]
    total = (objects["final_output_bytes"] + 2*n_pulses*len(diagnostics)*field
             + checkpoints*generation)
    if total > final_budget_bytes:
        raise StorageBudgetError('declared final diagnostics/checkpoints exceed final output budget')
    return dict(final_output_bytes=total, final_output_budget_bytes=final_budget_bytes,
                candidate_final_state_bytes=2*generation, optical_bytes=2*n_pulses*optical,
                external_prefix_bytes=field, existing_full_source_bytes=existing_full_source_bytes,
                object_schema=BUDGET_OBJECT_SCHEMA,
                metadata_allowance_bytes=objects["metadata_bytes"],
                selected_diagnostics=list(diagnostics), checkpoints=checkpoints)


class StorageBudgetError(RuntimeError):
    """A write, reservation, or reclamation request was rejected."""


class StorageIntegrityError(StorageBudgetError):
    """A path or file identity cannot be proven safe."""


# Set FILAMENT_STORAGE_LOCK_TRACE_DIR to an absolute directory outside every
# campaign root to opt into process/thread-local JSONL lock diagnostics.
_LOCK_TRACE_DIR_ENV = "FILAMENT_STORAGE_LOCK_TRACE_DIR"
_LOCK_TRACE_LOCAL = threading.local()


def _trace_path_is_within(path: Path, root: Path) -> bool:
    candidate = os.path.normcase(str(path.resolve(strict=False)))
    boundary = os.path.normcase(str(root.resolve(strict=False)))
    try:
        return os.path.commonpath((candidate, boundary)) == boundary
    except ValueError:
        return False


def _lock_trace_context(
    *, lock_path: Path | None = None, campaign_root: Path | None = None,
    operation_name: str | None = None,
) -> dict[str, Any] | None:
    """Create diagnostic identity only when tracing is explicitly enabled."""
    raw_directory = os.environ.get(_LOCK_TRACE_DIR_ENV)
    if not raw_directory:
        return None
    try:
        directory = Path(raw_directory).expanduser()
        if not directory.is_absolute():
            raise ValueError(f"{_LOCK_TRACE_DIR_ENV} must be absolute")
        directory = directory.resolve(strict=False)
        root = None if campaign_root is None else Path(campaign_root).resolve(strict=False)
        if root is not None and _trace_path_is_within(directory, root):
            raise ValueError(f"{_LOCK_TRACE_DIR_ENV} must be outside the campaign root")

        pid = os.getpid()
        start_identity = _process_start_token(pid) or "unknown"
        host = _host_identity()
        hostname = socket.gethostname()
        thread_id = threading.get_ident()
        thread_name = threading.current_thread().name
        safe_component = lambda value: "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(value)
        )
        safe_host = safe_component(hostname)
        safe_start_identity = safe_component(start_identity)
        filename = f"lock-{safe_host}-{pid}-{safe_start_identity}-{thread_id}.jsonl"
        lock_kind = "unscoped"
        if lock_path is not None:
            name = Path(lock_path).name
            if name == ".E5_1A_CAMPAIGN.lock":
                lock_kind = "campaign_state"
            elif name == ".storage_budget.lock":
                lock_kind = "storage_ledger"
            else:
                lock_kind = "file"
        return {
            "trace_dir": directory,
            "trace_file": directory / filename,
            "campaign_root": root,
            "lock_path": None if lock_path is None else str(Path(lock_path)),
            "lock_kind": lock_kind,
            "operation_name": operation_name,
            "host_identity": host,
            "hostname": hostname,
            "pid": pid,
            "process_start_identity": start_identity,
            "thread_id": thread_id,
            "thread_name": thread_name,
            "lock_span_id": f"{pid}-{thread_id}-{time.monotonic_ns()}",
        }
    except Exception as error:
        _LOCK_TRACE_LOCAL.error_count = getattr(_LOCK_TRACE_LOCAL, "error_count", 0) + 1
        _LOCK_TRACE_LOCAL.last_error = f"{type(error).__name__}: {error}"
        return None


def _trace_utc(wall_time_ns: int) -> str:
    seconds, nanoseconds = divmod(int(wall_time_ns), 1_000_000_000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + f".{nanoseconds:09d}Z"


def _emit_lock_trace(
    event: str, context: dict[str, Any] | None, **fields: Any,
) -> None:
    """Append to a process/thread-specific file; tracing failures never escape."""
    if context is None:
        return
    started_ns = time.perf_counter_ns()
    try:
        wall_time_ns = time.time_ns()
        previous_ns = getattr(_LOCK_TRACE_LOCAL, "last_append_ns", None)
        record = {
            "schema": "khz_filament.hr4e5.storage_lock_trace.v1",
            "event": event,
            "wall_time_ns": wall_time_ns,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": _trace_utc(wall_time_ns),
            "host_identity": context["host_identity"],
            "hostname": context["hostname"],
            "pid": context["pid"],
            "process_start_identity": context["process_start_identity"],
            "thread_id": context["thread_id"],
            "thread_name": context["thread_name"],
            "operation_name": context.get("operation_name"),
            "lock_path": context.get("lock_path"),
            "lock_kind": context.get("lock_kind", "unscoped"),
            "lock_span_id": context.get("lock_span_id"),
            "campaign_root": (None if context.get("campaign_root") is None
                              else str(context["campaign_root"])),
            "previous_trace_append_duration_ns": previous_ns,
            **fields,
        }
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        trace_file = Path(context["trace_file"])
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(trace_file), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
        finally:
            os.close(descriptor)
        _LOCK_TRACE_LOCAL.write_count = getattr(_LOCK_TRACE_LOCAL, "write_count", 0) + 1
    except Exception as error:
        _LOCK_TRACE_LOCAL.error_count = getattr(_LOCK_TRACE_LOCAL, "error_count", 0) + 1
        _LOCK_TRACE_LOCAL.last_error = f"{type(error).__name__}: {error}"
    finally:
        try:
            duration_ns = time.perf_counter_ns() - started_ns
            _LOCK_TRACE_LOCAL.last_append_ns = duration_ns
            _LOCK_TRACE_LOCAL.write_attempt_count = (
                getattr(_LOCK_TRACE_LOCAL, "write_attempt_count", 0) + 1
            )
            _LOCK_TRACE_LOCAL.write_duration_ns = (
                getattr(_LOCK_TRACE_LOCAL, "write_duration_ns", 0) + duration_ns
            )
        except Exception:
            # Timing diagnostics are best-effort and cannot affect the lock.
            pass


def _trace_metric_add(name: str, value: int) -> None:
    """Accumulate per-thread work counters for the enclosing phase receipt."""
    metrics = getattr(_LOCK_TRACE_LOCAL, "phase_metrics", None)
    if metrics is not None:
        metrics[name] = int(metrics.get(name, 0)) + int(value)


@contextmanager
def _trace_phase(
    phase_name: str, *, campaign_root: Path | None = None,
    operation_name: str | None = None, **details: Any,
):
    """Measure a storage phase only when tracing is enabled."""
    context = getattr(_LOCK_TRACE_LOCAL, "lock_context", None)
    if context is None:
        context = _lock_trace_context(campaign_root=campaign_root, operation_name=operation_name)
    if context is None:
        yield None
        return
    metrics = getattr(_LOCK_TRACE_LOCAL, "phase_metrics", None)
    if metrics is None:
        metrics = {}
        _LOCK_TRACE_LOCAL.phase_metrics = metrics
    metrics_start = dict(metrics)
    wall_start = time.time_ns()
    monotonic_start = time.monotonic_ns()
    status = "ok"
    error_type = None
    try:
        yield details
    except Exception as error:
        status = "error"
        error_type = type(error).__name__
        raise
    finally:
        wall_end = time.time_ns()
        monotonic_end = time.monotonic_ns()
        metric_deltas = {
            name: int(value) - int(metrics_start.get(name, 0))
            for name, value in metrics.items()
        }
        _emit_lock_trace(
            "phase", context, phase_name=phase_name,
            start_wall_time_ns=wall_start, end_wall_time_ns=wall_end,
            start_monotonic_ns=monotonic_start, end_monotonic_ns=monotonic_end,
            duration_ns=max(0, monotonic_end - monotonic_start),
            status=status, error_type=error_type,
            phase_metrics=metric_deltas, **details,
        )


def _lock_trace_status() -> dict[str, Any]:
    """Expose per-thread trace status and measured append overhead to harnesses."""
    raw_directory = os.environ.get(_LOCK_TRACE_DIR_ENV)
    return {
        "enabled": bool(raw_directory),
        "environment_variable": _LOCK_TRACE_DIR_ENV,
        "configured_directory": raw_directory or None,
        "trace_file": getattr(_LOCK_TRACE_LOCAL, "trace_file", None),
        "last_append_duration_ns": getattr(_LOCK_TRACE_LOCAL, "last_append_ns", 0),
        "write_count": getattr(_LOCK_TRACE_LOCAL, "write_count", 0),
        "write_attempt_count": getattr(_LOCK_TRACE_LOCAL, "write_attempt_count", 0),
        "write_duration_ns": getattr(_LOCK_TRACE_LOCAL, "write_duration_ns", 0),
        "error_count": getattr(_LOCK_TRACE_LOCAL, "error_count", 0),
        "last_error": getattr(_LOCK_TRACE_LOCAL, "last_error", None),
    }


def _caller_operation_name() -> str:
    """Infer a stable caller name for imported campaign-lock uses."""
    try:
        caller = sys._getframe(2)
        return f"{caller.f_globals.get('__name__', 'unknown')}.{caller.f_code.co_qualname}"
    except Exception:
        return "unknown"


class _FileLock:
    """Small cross-platform advisory lock used for ledger transitions."""

    def __init__(
        self, path: Path, timeout_s: float = 10.0, *,
        operation_name: str | None = None, campaign_root: Path | None = None,
    ):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self._handle = None
        self.operation_name = operation_name
        self.campaign_root = Path(campaign_root) if campaign_root is not None else self.path.parent
        self._trace_context: dict[str, Any] | None = None
        self._trace_request_monotonic_ns: int | None = None
        self._trace_acquired_monotonic_ns: int | None = None
        self._trace_previous_context: dict[str, Any] | None = None
        self._trace_terminal_event = False

    def __enter__(self) -> "_FileLock":
        if os.environ.get(_LOCK_TRACE_DIR_ENV):
            operation_name = self.operation_name or _caller_operation_name()
            self._trace_context = _lock_trace_context(
                lock_path=self.path, campaign_root=self.campaign_root,
                operation_name=operation_name,
            )
            self._trace_request_monotonic_ns = time.monotonic_ns()
            if self._trace_context is not None:
                _LOCK_TRACE_LOCAL.trace_file = str(self._trace_context["trace_file"])
                _emit_lock_trace("requested", self._trace_context,
                                 lock_timeout_s=self.timeout_s)
        try:
            return self._acquire()
        except BaseException as error:
            # __exit__ is not called when __enter__ fails. In particular an
            # interrupt in post-acquire tracing must not strand the OS lock.
            try:
                if self._handle is not None:
                    if self._trace_acquired_monotonic_ns is not None:
                        self.__exit__(type(error), error, error.__traceback__)
                    else:
                        self._handle.close()
                        self._handle = None
            except BaseException:
                # Preserve the original process-control/acquisition exception.
                pass
            finally:
                if getattr(_LOCK_TRACE_LOCAL, "lock_context", None) is self._trace_context:
                    _LOCK_TRACE_LOCAL.lock_context = self._trace_previous_context
            if self._trace_context is not None and not self._trace_terminal_event:
                _emit_lock_trace(
                    "acquire_error", self._trace_context,
                    elapsed_ns=(None if self._trace_request_monotonic_ns is None else
                                max(0, time.monotonic_ns() - self._trace_request_monotonic_ns)),
                    error_type=type(error).__name__,
                )
            raise

    def _acquire(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        if self._handle.tell() == 0:
            self._handle.write(b"0")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    self._handle.seek(0)
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._trace_acquired_monotonic_ns = (
                    time.monotonic_ns() if self._trace_context is not None else None
                )
                if self._trace_context is not None:
                    self._trace_previous_context = getattr(_LOCK_TRACE_LOCAL, "lock_context", None)
                    _LOCK_TRACE_LOCAL.lock_context = self._trace_context
                    _emit_lock_trace(
                        "acquired", self._trace_context,
                        wait_duration_ns=max(
                            0, self._trace_acquired_monotonic_ns
                            - int(self._trace_request_monotonic_ns),
                        ),
                        lock_timeout_s=self.timeout_s,
                    )
                return self
            except OSError as error:
                if time.monotonic() >= deadline:
                    if self._trace_context is not None:
                        _emit_lock_trace(
                            "timeout", self._trace_context,
                            wait_duration_ns=max(
                                0, time.monotonic_ns() - int(self._trace_request_monotonic_ns),
                            ),
                            lock_timeout_s=self.timeout_s, error_type=type(error).__name__,
                        )
                        self._trace_terminal_event = True
                    self._handle.close()
                    self._handle = None
                    raise StorageBudgetError("timed out acquiring storage ledger lock") from error
                time.sleep(0.01)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is None:
            return
        release_error_type = None
        released_monotonic_ns = None
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except BaseException as error:
            release_error_type = type(error).__name__
            raise
        finally:
            try:
                self._handle.close()
            except BaseException as error:
                if release_error_type is None:
                    release_error_type = type(error).__name__
                raise
            finally:
                self._handle = None
                released_monotonic_ns = (
                    time.monotonic_ns() if self._trace_context is not None else None
                )
                if self._trace_context is not None:
                    if getattr(_LOCK_TRACE_LOCAL, "lock_context", None) is self._trace_context:
                        _LOCK_TRACE_LOCAL.lock_context = self._trace_previous_context
                    hold_duration_ns = (
                        None if self._trace_acquired_monotonic_ns is None
                        else max(0, int(released_monotonic_ns)
                                 - self._trace_acquired_monotonic_ns)
                    )
                    _emit_lock_trace(
                        "released" if release_error_type is None else "release_error",
                        self._trace_context,
                        wait_duration_ns=(
                            None if self._trace_request_monotonic_ns is None
                            or self._trace_acquired_monotonic_ns is None
                            else max(0, self._trace_acquired_monotonic_ns
                                     - self._trace_request_monotonic_ns)
                        ),
                        hold_duration_ns=hold_duration_ns,
                        released_monotonic_ns=released_monotonic_ns,
                        error_type=release_error_type,
                    )


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _checked_process_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StorageIntegrityError("child grant process identity is invalid")
    try:
        pid = value["pid"]
        if isinstance(pid, bool):
            raise ValueError
        pid = int(pid)
        host = str(value["host_identity"])
        token = str(value["process_start_token"])
    except (KeyError, TypeError, ValueError) as error:
        raise StorageIntegrityError("child grant process identity is incomplete") from error
    if pid <= 0 or not host or not token:
        raise StorageIntegrityError("child grant process identity is invalid")
    return {"pid": pid, "host_identity": host, "process_start_token": token}


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool = True) -> None:
    with _trace_phase(
        "atomic_json", campaign_root=path.parent, operation_name="_atomic_json",
        file_path=str(path), overwrite=bool(overwrite),
    ):
        _atomic_json_impl(path, value, overwrite=overwrite)


def _atomic_json_impl(path: Path, value: Mapping[str, Any], *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _trace_phase(
        "atomic_temp_create", campaign_root=path.parent,
        operation_name="_atomic_json", file_path=str(path),
    ):
        descriptor, temporary = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent,
        )
    try:
        with _trace_phase(
            "atomic_json_write_flush_fsync", campaign_root=path.parent,
            operation_name="_atomic_json", file_path=str(path),
        ):
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(_json_safe(dict(value)), handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(path)
        with _trace_phase(
            "atomic_replace", campaign_root=path.parent,
            operation_name="_atomic_json", file_path=str(path),
        ):
            os.replace(temporary, path)
        with _trace_phase(
            "atomic_directory_fsync", campaign_root=path.parent,
            operation_name="_atomic_json", file_path=str(path.parent),
        ):
            _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        return


def _sha256_file(path: Path) -> str:
    trace_active = getattr(_LOCK_TRACE_LOCAL, "lock_context", None) is not None
    started_ns = time.perf_counter_ns() if trace_active else None
    byte_count = 0
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
        return digest.hexdigest()
    finally:
        if trace_active:
            _trace_metric_add("sha256_file_count", 1)
            _trace_metric_add("sha256_bytes", byte_count)
            _trace_metric_add(
                "sha256_duration_ns", time.perf_counter_ns() - int(started_ns),
            )


def _relative(root: Path, path: str | Path) -> tuple[Path, str]:
    root_resolved = root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    if '..' in candidate.parts:
        raise StorageIntegrityError('parent traversal is not allowed')
    try:
        lexical = candidate.relative_to(root_resolved)
    except ValueError as error:
        raise StorageIntegrityError('path escapes campaign root') from error
    cursor = root_resolved
    for part in lexical.parts:
        cursor = cursor / part
        if _is_reparse(cursor):
            raise StorageIntegrityError('symlink or reparse path is not eligible')
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(root_resolved)
    except (OSError, ValueError) as error:
        raise StorageIntegrityError("path escapes campaign root") from error
    # ``resolve`` above intentionally rejects ``..`` and prefix collisions;
    # retaining a second textual check catches a path that does not exist yet.
    parts = Path(path).parts
    if ".." in parts:
        raise StorageIntegrityError("parent traversal is not allowed")
    if not relative.parts:
        raise StorageIntegrityError("campaign root itself is not an artifact")
    return resolved, relative.as_posix()


def _is_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        stat = path.lstat()
        flag = getattr(stat, "st_file_attributes", 0)
        return bool(flag & getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _identity(path: Path) -> dict[str, Any]:
    try:
        stat = path.lstat()
    except OSError as error:
        raise StorageIntegrityError(f"artifact cannot be stat'ed: {path}") from error
    if _is_reparse(path):
        raise StorageIntegrityError("symlink or reparse artifact is not eligible")
    if not path.is_file():
        raise StorageIntegrityError("artifact is not a regular file")
    return {
        "st_dev": int(getattr(stat, "st_dev", -1)),
        "st_ino": int(getattr(stat, "st_ino", -1)),
        "st_nlink": int(getattr(stat, "st_nlink", 1)),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", 0)),
    }


def _validated_external_canonical_source_identity(
    campaign_root: Path, admission_identity: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Rebind one external E_source artifact and seal to the admission identity."""
    if not isinstance(admission_identity, Mapping):
        return None
    source_identity = admission_identity.get("source_identity")
    if not isinstance(source_identity, Mapping):
        return None
    record = source_identity.get("canonical_source")
    if not isinstance(record, Mapping):
        return None

    from .hr4e5_canonical_source import (
        CANONICAL_SOURCE_HASH_SCOPE,
        CANONICAL_SOURCE_HASH_SCOPES,
        CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES,
        CANONICAL_SOURCE_MANIFEST_SCHEMA,
        CANONICAL_SOURCE_SEAL_HASH_SCOPES,
        CANONICAL_SOURCE_SEAL_SCHEMA,
        CANONICAL_SOURCE_SCHEMA,
    )

    def external_file(raw: Any, label: str, *, must_be_external: bool = True) -> tuple[Path, dict[str, Any]]:
        candidate = Path(str(raw))
        if ".." in candidate.parts:
            raise StorageIntegrityError(f"external canonical {label} path traverses a parent")
        for component in (candidate, *candidate.parents):
            if _is_reparse(component):
                raise StorageIntegrityError(f"external canonical {label} path is linked")
        try:
            path = candidate.resolve(strict=True)
            identity = _identity(path)
        except (OSError, StorageIntegrityError) as error:
            raise StorageIntegrityError(f"external canonical {label} is unavailable") from error
        if int(identity.get("st_nlink", 1)) != 1:
            raise StorageIntegrityError(f"external canonical {label} has an unknown hardlink")
        if must_be_external and path.is_relative_to(campaign_root):
            raise StorageIntegrityError(f"external canonical {label} is inside the campaign root")
        return path, identity

    try:
        manifest_path, manifest_identity = external_file(source_identity.get("path", ""), "manifest")
        config_path, _ = external_file(
            admission_identity.get("config_identity", {}).get("path", ""), "config",
            must_be_external=False,
        )
        expected_manifest_sha = str(source_identity.get("sha256", ""))
        if not _is_sha256(expected_manifest_sha) or _sha256_file(manifest_path) != expected_manifest_sha:
            raise StorageIntegrityError("external canonical source manifest differs from admission")
        if _sha256_file(config_path) != str(record.get("config_sha256", "")):
            raise StorageIntegrityError("external canonical source config differs from admission")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, Mapping)
                or manifest.get("schema") != CANONICAL_SOURCE_MANIFEST_SCHEMA
                or manifest.get("hash_scope") != CANONICAL_SOURCE_HASH_SCOPE
                or dict(manifest.get("canonical_source", {})) != dict(record)
                or manifest.get("canonical_array_hash") != record.get("canonical_array_hash")
                or manifest.get("canonical_source_artifact_path") != record.get("artifact_path")
                or manifest.get("canonical_source_sealed_manifest_path") != record.get("sealed_manifest_path")):
            raise StorageIntegrityError("external canonical source manifest bindings differ")
        if (record.get("schema") != CANONICAL_SOURCE_SCHEMA
                or record.get("hash_scopes") != CANONICAL_SOURCE_HASH_SCOPES):
            raise StorageIntegrityError("external canonical source schema or hash scopes are invalid")
        artifact_path, artifact_identity = external_file(record.get("artifact_path", ""), "artifact")
        seal_path, seal_identity = external_file(record.get("sealed_manifest_path", ""), "seal")
        if artifact_path.parent != manifest_path.parent or seal_path.parent != manifest_path.parent:
            raise StorageIntegrityError("external canonical source files must share one input directory")
        artifact_sha = _sha256_file(artifact_path)
        if (int(artifact_identity["size"]) != int(record.get("file_size_bytes", -1))
                or artifact_sha != str(record.get("file_sha256", ""))):
            raise StorageIntegrityError("external canonical source artifact differs from admission")
        seal_payload = json.loads(seal_path.read_text(encoding="utf-8"))
        admission_sha = str(admission_identity.get("identity_sha256", ""))
        implementation_sha = str(admission_identity.get("implementation_identity_sha256", ""))
        required_seal = {
            "schema": CANONICAL_SOURCE_SEAL_SCHEMA,
            "hash_scope": CANONICAL_SOURCE_HASH_SCOPE,
            "hash_scopes": CANONICAL_SOURCE_SEAL_HASH_SCOPES,
            "sealed": True,
            "source_logical_id": str(record.get("logical_id", "")),
            "artifact_path": str(artifact_path),
            "artifact_file_sha256": artifact_sha,
            "artifact_file_size_bytes": int(artifact_identity["size"]),
            "payload_bytes_sha256": str(record.get("payload_bytes_sha256", "")),
            "canonical_array_hash": str(record.get("canonical_array_hash", "")),
            "config_sha256": str(record.get("config_sha256", "")),
            "admission_identity_sha256": admission_sha,
            "implementation_identity_sha256": implementation_sha,
        }
        if (not isinstance(seal_payload, Mapping)
                or set(seal_payload) != set(required_seal) | {"created_utc"}
                or any(seal_payload.get(key) != value for key, value in required_seal.items())
                or not isinstance(seal_payload.get("created_utc"), str)
                or len(str(seal_payload.get("created_utc"))) < 20
                or (int(manifest_identity["size"]) + int(seal_identity["size"]))
                > CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES):
            raise StorageIntegrityError("external canonical source seal differs from admission")
        return {
            "schema": "khz_filament.hr4e5.e5_1b.external_canonical_source.v1",
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": expected_manifest_sha,
            "source_manifest_bytes": int(manifest_identity["size"]),
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "artifact_bytes": int(artifact_identity["size"]),
            "seal_path": str(seal_path),
            "seal_sha256": _sha256_file(seal_path),
            "seal_bytes": int(seal_identity["size"]),
            "canonical_array_hash": str(record.get("canonical_array_hash", "")),
            "payload_bytes_sha256": str(record.get("payload_bytes_sha256", "")),
            "admission_identity_sha256": admission_sha,
            "implementation_identity_sha256": implementation_sha,
        }
    except StorageIntegrityError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StorageIntegrityError("external canonical source identity cannot be reconstructed") from error


def _iter_regular_files(
    root: Path, *, phase_name: str = "directory_scan",
    campaign_root: Path | None = None,
) -> Iterable[Path]:
    with _trace_phase(
        phase_name, campaign_root=campaign_root,
        operation_name="_iter_regular_files", scan_root=str(root),
    ):
        yield from _iter_regular_files_impl(root)


def _iter_regular_files_impl(root: Path) -> Iterable[Path]:
    """Yield files without following directory links."""
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        safe_dirs = []
        for dirname in dirnames:
            path = base / dirname
            if _is_reparse(path):
                # A reparse point is an unbounded accounting boundary.  It is
                # safer to reject the whole budget operation than silently
                # omit a linked subtree from the campaign footprint.
                raise StorageIntegrityError(f"reparse directory is not account-safe: {path}")
            safe_dirs.append(dirname)
        dirnames[:] = safe_dirs
        for filename in filenames:
            path = base / filename
            if _is_reparse(path):
                raise StorageIntegrityError(f"reparse file is not account-safe: {path}")
            if path.is_file():
                yield path


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _relative_paths_overlap(left: str | Path, right: str | Path) -> bool:
    """Return whether two normalized campaign-relative paths overlap."""
    lhs, rhs = Path(left), Path(right)
    return lhs == rhs or lhs.is_relative_to(rhs) or rhs.is_relative_to(lhs)


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    bytes: int
    purpose: str
    owner: str
    created_utc: str


class StorageBudget:
    """Durable byte ledger and fail-closed whitelist GC coordinator."""

    @staticmethod
    def _checked_lock_timeout(value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("storage lock timeout must be finite and within (0, 180] seconds")
        result = float(value)
        if not math.isfinite(result) or not 0 < result <= 180:
            raise ValueError("storage lock timeout must be finite and within (0, 180] seconds")
        return result

    def __init__(
        self,
        root: str | Path,
        *,
        cap_bytes: int = HARD_CAP_BYTES,
        final_output_budget_bytes: int = DEFAULT_FINAL_OUTPUT_BUDGET_BYTES,
        safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
        provider: Any | None = None,
        require_quota: bool = True,
        campaign_id: str = "e5_1a_local",
        require_intents: bool = False,
        admission_hash: str | None = None,
        admission_identity: Mapping[str, Any] | None = None,
        external_prefix_bytes: int = 0,
        external_full_source_bytes: int = 0,
        frozen_lut_input_bytes: int | None = None,
        planned_final_output_bytes: int | None = None,
        lock_timeout_s: float = 10.0,
    ):
        # Only an explicitly bound campaign may opt into a measured finite
        # wait. This does not alter _FileLock or campaign-state lock defaults.
        self.lock_timeout_s = self._checked_lock_timeout(lock_timeout_s)
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "storage_ledger.json"
        self.lock_path = self.root / ".storage_budget.lock"
        cap = _nonnegative_int(cap_bytes, "cap_bytes")
        final_budget = _nonnegative_int(final_output_budget_bytes, "final_output_budget_bytes")
        margin = _nonnegative_int(safety_margin_bytes, "safety_margin_bytes")
        if cap > HARD_CAP_BYTES:
            raise StorageBudgetError(f"campaign cap cannot exceed hard limit {HARD_CAP_BYTES} bytes")
        if cap <= 0 or final_budget > cap or margin <= 0 or margin >= cap:
            raise StorageBudgetError("storage cap, final output budget, or safety margin is invalid")
        self.cap_bytes = cap
        self.final_output_budget_bytes = final_budget
        self.safety_margin_bytes = margin
        self.provider = provider
        self.require_quota = bool(require_quota)
        self.campaign_id = str(campaign_id)
        self.require_intents = bool(require_intents)
        self.external_prefix_bytes = _nonnegative_int(external_prefix_bytes, "external_prefix_bytes")
        bound_lut_input_bytes = _nonnegative_int(
            (admission_identity or {}).get("budget_identity", {}).get("frozen_lut_input_bytes", 0),
            "admitted frozen LUT input bytes",
        )
        if frozen_lut_input_bytes is None:
            frozen_lut_input_bytes = bound_lut_input_bytes
        else:
            frozen_lut_input_bytes = _nonnegative_int(frozen_lut_input_bytes, "frozen_lut_input_bytes")
            if admission_identity is not None and frozen_lut_input_bytes != bound_lut_input_bytes:
                raise StorageBudgetError("frozen LUT input bytes conflict with admission identity")
        self.frozen_lut_input_bytes = _nonnegative_int(
            frozen_lut_input_bytes, "frozen_lut_input_bytes")
        bound_final = (admission_identity or {}).get("budget_identity", {}).get("final_output_bytes", 0)
        if planned_final_output_bytes is None:
            planned_final_output_bytes = bound_final
        elif admission_identity is not None and int(planned_final_output_bytes) != int(bound_final):
            raise StorageBudgetError("planned final output conflicts with admission identity")
        self.planned_final_output_bytes = _nonnegative_int(planned_final_output_bytes, "planned_final_output_bytes")
        self.external_full_source_bytes = _nonnegative_int(external_full_source_bytes, "external_full_source_bytes")
        self.external_canonical_source_identity = _validated_external_canonical_source_identity(
            self.root, admission_identity,
        )
        self.external_canonical_source_bytes = (
            0 if self.external_canonical_source_identity is None else
            int(self.external_canonical_source_identity["source_manifest_bytes"])
            + int(self.external_canonical_source_identity["artifact_bytes"])
            + int(self.external_canonical_source_identity["seal_bytes"])
        )
        if admission_hash is None and admission_identity is not None:
            admission_hash = admission_identity.get("identity_sha256")
        self.admission_hash = None if admission_hash is None else str(admission_hash)
        self.admission_identity = None if admission_identity is None else dict(admission_identity)
        if not self.ledger_path.exists():
            _atomic_json(self.ledger_path, self._new_ledger(), overwrite=False)
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.__init__",
            campaign_root=self.root,
        ):
            self._validate(self._read())

    def _new_ledger(self) -> dict[str, Any]:
        return {
            "schema": STORAGE_SCHEMA,
            "campaign_id": self.campaign_id,
            "cap_bytes": self.cap_bytes,
            "final_output_budget_bytes": self.final_output_budget_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "external_prefix_bytes": self.external_prefix_bytes,
            **({"frozen_lut_input_bytes": self.frozen_lut_input_bytes}
               if self.frozen_lut_input_bytes else {}),
            "external_canonical_source_identity": self.external_canonical_source_identity,
            "planned_final_output_bytes": self.planned_final_output_bytes,
            "external_full_source_bytes": self.external_full_source_bytes,
            "require_quota": self.require_quota,
            "require_intents": self.require_intents,
            "admission_hash": self.admission_hash,
            "artifacts": {},
            "reservations": {},
            "intents": {},
            "writers": {},
            "child_writer_grants": {},
            "gc_plans": {},
            "events": [],
            "created_utc": _utc(),
        }

    def _read(self) -> dict[str, Any]:
        with _trace_phase(
            "ledger_json_read_decode", campaign_root=self.root,
            operation_name="StorageBudget._read", file_path=str(self.ledger_path),
        ):
            try:
                data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise StorageBudgetError("storage ledger is unreadable") from error
            if not isinstance(data, dict):
                raise StorageBudgetError("storage ledger must be an object")
            return data

    def _validate(self, data: Mapping[str, Any]) -> None:
        with _trace_phase(
            "ledger_validation", campaign_root=self.root,
            operation_name="StorageBudget._validate", file_path=str(self.ledger_path),
        ):
            self._validate_impl(data)

    def _validate_impl(self, data: Mapping[str, Any]) -> None:
        if data.get("schema") != STORAGE_SCHEMA:
            raise StorageBudgetError("storage ledger schema is invalid")
        if int(data.get("cap_bytes", -1)) != self.cap_bytes:
            raise StorageBudgetError("storage cap conflicts with durable ledger")
        if int(data.get("final_output_budget_bytes", -1)) != self.final_output_budget_bytes:
            raise StorageBudgetError("final output budget conflicts with durable ledger")
        if int(data.get("safety_margin_bytes", -1)) != self.safety_margin_bytes:
            raise StorageBudgetError("storage safety margin conflicts with durable ledger")
        if int(data.get("external_prefix_bytes", 0)) != self.external_prefix_bytes:
            raise StorageBudgetError("external prefix charge conflicts with durable ledger")
        if int(data.get("frozen_lut_input_bytes", 0)) != self.frozen_lut_input_bytes:
            raise StorageBudgetError("frozen LUT input charge conflicts with durable ledger")
        if int(data.get("planned_final_output_bytes", 0)) != self.planned_final_output_bytes:
            raise StorageBudgetError("planned final output conflicts with durable ledger")
        if int(data.get("external_full_source_bytes", 0)) != self.external_full_source_bytes:
            raise StorageBudgetError("external full source charge conflicts with durable ledger")
        if data.get("external_canonical_source_identity") != self.external_canonical_source_identity:
            raise StorageIntegrityError("external canonical source identity conflicts with durable ledger")
        if bool(data.get("require_quota", False)) != self.require_quota:
            raise StorageBudgetError("storage quota requirement conflicts with durable ledger")
        if bool(data.get("require_intents", False)) != self.require_intents:
            raise StorageBudgetError("storage intent requirement conflicts with durable ledger")
        saved_admission = data.get("admission_hash")
        if (None if saved_admission is None else str(saved_admission)) != self.admission_hash:
            raise StorageIntegrityError("storage admission identity conflicts with durable ledger")
        if str(data.get("campaign_id", "")) != self.campaign_id:
            raise StorageIntegrityError("storage ledger campaign identity conflicts with requested campaign")
        if (not isinstance(data.get("artifacts"), dict)
                or not isinstance(data.get("reservations"), dict)
                or not isinstance(data.get("intents", {}), dict)
                or not isinstance(data.get("writers", {}), dict)
                or not isinstance(data.get("child_writer_grants", {}), dict)):
            raise StorageBudgetError("storage ledger collections are invalid")
        for reservation_id, item in data.get("reservations", {}).items():
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"reservation record is invalid: {reservation_id}")
            paths = item.get("allocation_paths", [])
            if not isinstance(paths, list):
                raise StorageBudgetError(f"reservation allocation paths are invalid: {reservation_id}")
            normalized = [str(path).replace("\\", "/") for path in paths]
            for index, left in enumerate(normalized):
                for right in normalized[index + 1:]:
                    if _relative_paths_overlap(left, right):
                        raise StorageIntegrityError(f"reservation allocation paths overlap: {reservation_id}")
        for intent_id, item in data.get("intents", {}).items():
            if not isinstance(item, Mapping) or item.get("schema") != INTENT_SCHEMA:
                raise StorageBudgetError(f"creation intent record is invalid: {intent_id}")
            if str(item.get("campaign_id", "")) != self.campaign_id:
                raise StorageIntegrityError(f"creation intent campaign mismatch: {intent_id}")
            allowed = item.get("allowed_paths", [])
            if not isinstance(allowed, list) or not allowed:
                raise StorageBudgetError(f"creation intent paths are invalid: {intent_id}")
            for index, left in enumerate(str(path).replace("\\", "/") for path in allowed):
                for right in (str(path).replace("\\", "/") for path in allowed[index + 1:]):
                    if _relative_paths_overlap(left, right):
                        raise StorageIntegrityError(f"creation intent paths overlap: {intent_id}")
        for writer_id, item in data.get("writers", {}).items():
            if not isinstance(item, Mapping) or item.get("schema") != WRITER_SCHEMA:
                raise StorageBudgetError(f"writer record is invalid: {writer_id}")
            if str(item.get("campaign_id", "")) != self.campaign_id:
                raise StorageIntegrityError(f"writer campaign mismatch: {writer_id}")
        for grant_id, item in data.get("child_writer_grants", {}).items():
            if (not isinstance(item, Mapping)
                    or item.get("schema") != CHILD_WRITER_GRANT_SCHEMA
                    or str(item.get("grant_id", "")) != str(grant_id)):
                raise StorageBudgetError(f"child writer grant record is invalid: {grant_id}")
            if str(item.get("campaign_id", "")) != self.campaign_id:
                raise StorageIntegrityError(f"child writer grant campaign mismatch: {grant_id}")
            if item.get("state") not in CHILD_GRANT_STATES:
                raise StorageBudgetError(f"child writer grant state is invalid: {grant_id}")
            if (not isinstance(item.get("nonce_sha256"), str)
                    or len(str(item.get("nonce_sha256"))) != 64):
                raise StorageIntegrityError(f"child writer grant nonce hash is invalid: {grant_id}")
            scope = item.get("scope")
            if not isinstance(scope, Mapping):
                raise StorageBudgetError(f"child writer grant scope is invalid: {grant_id}")
            role = str(scope.get("role", ""))
            role_index = scope.get("role_index")
            if role == "OPTICAL":
                if role_index is not None:
                    raise StorageIntegrityError(f"OPTICAL grant has an index: {grant_id}")
            elif role == "HYDRO":
                if isinstance(role_index, bool) or not isinstance(role_index, int) or role_index < 0:
                    raise StorageIntegrityError(f"HYDRO grant index is invalid: {grant_id}")
            else:
                raise StorageIntegrityError(f"child writer grant role is invalid: {grant_id}")
            if tuple(scope.get("permitted_operations", ())) != CHILD_GRANT_OPERATIONS[role]:
                raise StorageIntegrityError(f"child writer grant operation scope is invalid: {grant_id}")
            if (not _is_sha256(scope.get("scientific_pair_id"))
                    or not _is_sha256(scope.get("admission_identity_sha256"))):
                raise StorageIntegrityError(f"child writer grant pair/admission identity is invalid: {grant_id}")
            implementation = scope.get("implementation_identity")
            if (not isinstance(implementation, Mapping)
                    or not str(implementation.get("schema", ""))
                    or not _is_sha256(implementation.get("identity_sha256"))):
                raise StorageIntegrityError(f"child writer grant implementation identity is invalid: {grant_id}")
            paths = scope.get("allowed_paths", [])
            if not isinstance(paths, list) or not paths:
                raise StorageIntegrityError(f"child writer grant paths are invalid: {grant_id}")
            normalized = [str(path).replace("\\", "/") for path in paths]
            for index, left in enumerate(normalized):
                for right in normalized[index + 1:]:
                    if _relative_paths_overlap(left, right):
                        raise StorageIntegrityError(f"child writer grant paths overlap: {grant_id}")
            try:
                _canonical_json(scope.get("implementation_identity"))
                _nonnegative_int(scope.get("expected_bytes"), "grant expected_bytes")
            except (TypeError, ValueError) as error:
                raise StorageIntegrityError(f"child writer grant identity or byte bound is invalid: {grant_id}") from error
            candidate = item.get("launch_candidate_identity")
            owner = item.get("process_identity")
            if (str(item.get("reservation_id")) != str(scope.get("reservation_id"))
                    or str(item.get("intent_id")) != str(scope.get("intent_id"))
                    or str(item.get("coordinator_epoch")) != str(scope.get("coordinator_epoch"))
                    or str(item.get("role_label")) != str(scope.get("role_label"))):
                raise StorageIntegrityError(f"child writer grant envelope differs from scope: {grant_id}")
            intent = data.get("intents", {}).get(str(item.get("intent_id")))
            reservation = data.get("reservations", {}).get(str(item.get("reservation_id")))
            if not isinstance(intent, Mapping) or not isinstance(reservation, Mapping):
                raise StorageIntegrityError(f"child writer grant references missing storage authority: {grant_id}")
            metadata = intent.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise StorageIntegrityError(f"child writer grant intent metadata is invalid: {grant_id}")
            intent_paths = [str(path) for path in intent.get("allowed_paths", [])]
            allocation_paths = [str(path) for path in reservation.get("allocation_paths", [])]
            if (str(intent.get("reservation_id")) != str(item.get("reservation_id"))
                    or str(intent.get("campaign_id")) != self.campaign_id
                    or str(intent.get("trajectory")) != str(scope.get("trajectory"))
                    or int(intent.get("pulse", -1)) != int(scope.get("pulse", -2))
                    or int(intent.get("attempt", -1)) != int(scope.get("logical_attempt", -2))
                    or str(intent.get("generation")) != str(scope.get("generation"))
                    or int(scope.get("expected_bytes", 0)) > int(intent.get("expected_bytes", 0))
                    or str(intent.get("admission_hash")) != str(scope.get("admission_identity_sha256"))
                    or str(metadata.get("scientific_pair_id", "")) != str(scope.get("scientific_pair_id"))
                    or _canonical_json(metadata.get("implementation_identity"))
                    != _canonical_json(implementation)):
                raise StorageIntegrityError(f"child writer grant differs from durable intent identity: {grant_id}")
            if (any(not any(self._path_is_allowed(path, [base]) for base in intent_paths)
                    for path in paths)
                    or (allocation_paths and any(not any(self._path_is_allowed(path, [base])
                                                        for base in allocation_paths) for path in paths))):
                raise StorageIntegrityError(f"child writer grant path exceeds durable allocation scope: {grant_id}")
            if candidate is not None:
                _checked_process_identity(candidate)
            if item.get("state") == "BOUND_ACTIVE" and owner is None:
                raise StorageIntegrityError(f"bound child writer grant lacks owner identity: {grant_id}")
            if owner is not None:
                _checked_process_identity(owner)
            if item.get("state") == "PENDING" and owner is not None:
                raise StorageIntegrityError(f"PENDING child writer grant already has a bound owner: {grant_id}")
            if item.get("state") in {"ABANDONED", "INTERRUPTED_STALE"}:
                evidence = item.get("liveness_evidence", {})
                if not isinstance(evidence, Mapping) or evidence.get("status") != "DEAD":
                    raise StorageIntegrityError(f"fenced child writer grant lacks DEAD evidence: {grant_id}")

    def _write(self, data: Mapping[str, Any]) -> None:
        with _trace_phase(
            "ledger_write", campaign_root=self.root,
            operation_name="StorageBudget._write", file_path=str(self.ledger_path),
        ):
            self._validate(data)
            _atomic_json(self.ledger_path, data)

    @contextmanager
    def _locked(self, *, operation_name: str = "StorageBudget._locked"):
        # Windows can reject os.replace transiently while another handle in a
        # freshly exited process is closing the same ledger.  The lock is held
        # here, so retrying the same durable replacement is safe and bounded.
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name=operation_name,
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            yield data
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    self._write(data)
                    return
                except PermissionError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)

    def _management_paths(self) -> set[Path]:
        return {self.ledger_path.resolve(), self.lock_path.resolve()}

    def actual_bytes(self) -> int:
        total = 0
        for path in _iter_regular_files(
            self.root, phase_name="root_allocation_scan", campaign_root=self.root,
        ):
            try:
                total += int(path.stat().st_size)
            except OSError as error:
                raise StorageBudgetError(f"cannot account file {path}") from error
        return total

    @property
    def used_bytes(self) -> int:
        return self.actual_bytes()

    def _reserved_bytes(self, data: Mapping[str, Any]) -> int:
        total = 0
        for item in data.get('reservations', {}).values():
            if item.get('status') != 'ACTIVE':
                continue
            materialized = max(0, self._allocation_bytes(item.get('allocation_paths', [])) - int(item.get('allocation_start_bytes', 0)))
            total += max(0, int(item['bytes']) - materialized)
        return total

    @staticmethod
    def _ensure_child_grants_quiescent(
        data: Mapping[str, Any], *, reservation_id: str | None = None,
        intent_id: str | None = None,
    ) -> None:
        unresolved = [str(grant_id) for grant_id, item in data.get("child_writer_grants", {}).items()
                      if item.get("state") in {"PENDING", "BOUND_ACTIVE"}
                      and (reservation_id is None or str(item.get("reservation_id")) == str(reservation_id))
                      and (intent_id is None or str(item.get("intent_id")) == str(intent_id))]
        if unresolved:
            raise StorageBudgetError(f"unresolved child writer grants block storage completion: {unresolved}")

    def _allocation_bytes(self, paths: Sequence[str]) -> int:
        total = 0
        for relative in paths:
            path, _ = _relative(self.root, relative)
            if path.is_dir():
                total += sum(
                    p.stat().st_size for p in _iter_regular_files(
                        path, phase_name="allocation_path_scan", campaign_root=self.root,
                    )
                )
            elif path.is_file():
                total += path.stat().st_size
        return total

    def _capacity_accounting(
        self, data: Mapping[str, Any], allocation: Sequence[str] = (),
    ) -> tuple[int, int, int]:
        """Count root and allocation bytes in one lock-local traversal.

        No state survives this operation and this is not an atomic filesystem
        snapshot. Existing path/reparse checks and every regular-file size
        remain authoritative. Intent content hashes stay under the ledger lock.
        """
        active = [item for item in data.get("reservations", {}).values()
                  if item.get("status") == "ACTIVE"]
        requested = [str(path) for item in active
                     for path in item.get("allocation_paths", [])] + list(allocation)
        scopes: dict[str, tuple[Path, bool, bool]] = {}
        for raw in requested:
            path, _ = _relative(self.root, raw)
            scopes[raw] = (path, path.is_dir(), path.is_file())
        totals = {raw: 0 for raw in scopes}
        used = 0
        with _trace_phase("capacity_accounting", campaign_root=self.root,
                          operation_name="StorageBudget._capacity_accounting",
                          allocation_scope_count=len(scopes)):
            for path in _iter_regular_files(
                self.root, phase_name="root_allocation_scan", campaign_root=self.root,
            ):
                try:
                    size = int(path.stat().st_size)
                except OSError as error:
                    raise StorageBudgetError(f"cannot account file {path}") from error
                used += size
                _trace_metric_add("accounting_file_count", 1)
                _trace_metric_add("accounting_logical_bytes", size)
                for raw, (scope, directory, regular) in scopes.items():
                    if (directory and path.is_relative_to(scope)) or (regular and path == scope):
                        totals[raw] += size
        reserved = 0
        for item in active:
            materialized = max(0, sum(totals[str(path)] for path in item.get("allocation_paths", []))
                               - int(item.get("allocation_start_bytes", 0)))
            reserved += max(0, int(item["bytes"]) - materialized)
        return used, reserved, sum(totals[path] for path in allocation)

    def _provider_free_bytes(self) -> int | None:
        provider = self.provider
        if provider is None:
            try:
                return int(shutil.disk_usage(self.root).free)
            except OSError:
                return None
        try:
            value = provider.free_bytes() if callable(getattr(provider, "free_bytes", None)) else getattr(provider, "free_bytes")
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        if value is None:
            return None
        return _nonnegative_int(value, "free_bytes")

    def _provider_quota_bytes(self) -> int | None:
        provider = self.provider
        if provider is None:
            return None
        try:
            value = provider.quota_bytes() if callable(getattr(provider, "quota_bytes", None)) else getattr(provider, "quota_bytes")
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        if value is None:
            return None
        return _nonnegative_int(value, "quota_bytes")

    def _require_quota_if_configured(self, quota: int | None) -> None:
        if self.require_quota and quota is None:
            raise StorageBudgetError(
                "quota is unknown; production storage admission is refused"
            )

    def check_capacity(self, additional_bytes: int = 0, *, peak_bytes: int = 0, purpose: str = "write") -> dict[str, Any]:
        additional = _nonnegative_int(additional_bytes, "additional_bytes")
        peak = _nonnegative_int(peak_bytes, "peak_bytes")
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.check_capacity",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            used, reserved, _ = self._capacity_accounting(data)
            projected = (used + self.external_prefix_bytes + self.external_canonical_source_bytes
                         + reserved + additional + peak + self.safety_margin_bytes)
            free = self._provider_free_bytes()
            quota = self._provider_quota_bytes()
            self._require_quota_if_configured(quota)
            if projected > self.cap_bytes:
                raise StorageBudgetError(
                    f"{purpose} exceeds campaign cap: projected={projected} cap={self.cap_bytes}"
                )
            if free is not None and reserved + additional + peak + self.safety_margin_bytes > free:
                raise StorageBudgetError(f"{purpose} exceeds current filesystem free space")
            if quota is not None and projected > quota:
                raise StorageBudgetError(f"{purpose} exceeds reported quota")
            return {
                "status": "PASS",
                "purpose": purpose,
                "actual_bytes": used,
                "external_prefix_bytes": self.external_prefix_bytes,
                **({"frozen_lut_input_bytes": self.frozen_lut_input_bytes}
                   if self.frozen_lut_input_bytes else {}),
                "external_canonical_source_bytes": self.external_canonical_source_bytes,
                "external_canonical_source_identity": self.external_canonical_source_identity,
                "external_full_source_bytes": self.external_full_source_bytes,
                "reserved_bytes": reserved,
                "additional_bytes": additional,
                "peak_bytes": peak,
                "safety_margin_bytes": self.safety_margin_bytes,
                "projected_bytes": projected,
                "cap_bytes": self.cap_bytes,
                "free_bytes": free,
                "quota_bytes": quota,
            }

    def reserve(
        self, bytes: int, *, purpose: str, owner: str | None = None,
        reservation_id: str | None = None, peak_bytes: int = 0,
        allocation_paths: Sequence[str | Path] = (),
    ) -> Reservation:
        amount = _nonnegative_int(bytes, "reservation bytes") + _nonnegative_int(peak_bytes, 'peak_bytes')
        allocation = [_relative(self.root, p)[1] for p in allocation_paths]
        for index, left in enumerate(allocation):
            for right in allocation[index + 1:]:
                if _relative_paths_overlap(left, right):
                    raise StorageBudgetError('overlapping allocation ownership')
        if not str(purpose):
            raise ValueError("reservation purpose is required")
        rid = str(reservation_id or f"res-{os.getpid()}-{time.time_ns()}")
        owner_name = str(owner or f"pid:{os.getpid()}")
        with self._locked(operation_name="StorageBudget.reserve") as data:
            reservations = data["reservations"]
            # Reject an already-owned scope before any expensive filesystem I/O.
            for previous in reservations.values():
                if previous.get('status') == 'ACTIVE':
                    for left in allocation:
                        for right in previous.get('allocation_paths', []):
                            if _relative_paths_overlap(left, right):
                                raise StorageBudgetError('overlapping allocation ownership')
            if rid in reservations and reservations[rid].get("status") == "ACTIVE":
                raise StorageBudgetError(f"reservation already active: {rid}")
            used, active, allocation_start = self._capacity_accounting(data, allocation)
            free = self._provider_free_bytes()
            quota = self._provider_quota_bytes()
            self._require_quota_if_configured(quota)
            projected = (used + self.external_prefix_bytes + self.external_canonical_source_bytes
                         + active + amount + self.safety_margin_bytes)
            if projected > self.cap_bytes:
                raise StorageBudgetError(f"reservation exceeds campaign cap: projected={projected} cap={self.cap_bytes}")
            if free is not None and active + amount + self.safety_margin_bytes > free:
                raise StorageBudgetError("reservation exceeds current filesystem free space")
            if quota is not None and projected > quota:
                raise StorageBudgetError("reservation exceeds reported quota")
            item = {"reservation_id": rid, "bytes": amount, "purpose": str(purpose), "owner": owner_name, "created_utc": _utc(), "status": "ACTIVE"}
            item.update(allocation_paths=allocation, allocation_start_bytes=allocation_start)
            reservations[rid] = item
            data["events"].append({"event": "RESERVE", **item})
        return Reservation(rid, amount, str(purpose), owner_name, item["created_utc"])

    reserve_bytes = reserve

    def release(self, reservation_id: str, *, status: str = "RELEASED") -> dict[str, Any]:
        rid = str(reservation_id)
        with self._locked(operation_name="StorageBudget.release") as data:
            item = data["reservations"].get(rid)
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"unknown reservation: {rid}")
            if item.get("status") != "ACTIVE":
                return dict(item)
            self._ensure_child_grants_quiescent(data, reservation_id=rid)
            if self.require_intents:
                allocation_paths = item.get("allocation_paths", [])
                current = self._allocation_bytes(allocation_paths)
                start = _nonnegative_int(item.get("allocation_start_bytes", 0), "allocation_start_bytes")
                if current > start:
                    raise StorageBudgetError("cannot release a reservation while residual files remain")
            mutable = dict(item)
            mutable["status"] = str(status)
            mutable["released_utc"] = _utc()
            data["reservations"][rid] = mutable
            data["events"].append({"event": "RELEASE", **mutable})
            return mutable

    def consume(self, reservation_id: str, *, actual_bytes: int | None = None) -> dict[str, Any]:
        rid = str(reservation_id)
        with self._locked(operation_name="StorageBudget.consume") as data:
            item = data["reservations"].get(rid)
            if not isinstance(item, Mapping) or item.get("status") != "ACTIVE":
                raise StorageBudgetError(f"reservation is not active: {rid}")
            self._ensure_child_grants_quiescent(data, reservation_id=rid)
            reserved_bytes = _nonnegative_int(item.get("bytes"), "reservation bytes")
            allocation_paths = item.get("allocation_paths", [])
            if not isinstance(allocation_paths, list):
                raise StorageBudgetError("reservation allocation paths are invalid")
            allocation_growth = 0
            if allocation_paths:
                current_bytes = self._allocation_bytes(allocation_paths)
                allocation_growth = max(
                    0,
                    current_bytes - _nonnegative_int(
                        item.get("allocation_start_bytes", 0), "allocation_start_bytes"
                    ),
                )
                if allocation_growth > reserved_bytes:
                    raise StorageBudgetError(
                        f"reservation allocation exceeded: growth={allocation_growth} reserved={reserved_bytes}"
                    )
            reported_bytes = (
                reserved_bytes
                if actual_bytes is None
                else _nonnegative_int(actual_bytes, "actual_bytes")
            )
            if reported_bytes > reserved_bytes:
                raise StorageBudgetError(
                    f"consumed bytes exceed reservation: consumed={reported_bytes} reserved={reserved_bytes}"
                )
            if allocation_paths and actual_bytes is not None and allocation_growth > reported_bytes:
                raise StorageBudgetError(
                    f"reported bytes understate allocation growth: growth={allocation_growth} reported={reported_bytes}"
                )
            mutable = dict(item)
            mutable["status"] = "CONSUMED"
            mutable["consumed_bytes"] = int(
                allocation_growth if actual_bytes is None and allocation_paths else reported_bytes
            )
            mutable["allocation_growth_bytes"] = int(allocation_growth)
            mutable["released_utc"] = _utc()
            data["reservations"][rid] = mutable
            data["events"].append({"event": "CONSUME", **mutable})
            return mutable

    def _intent_paths(self, paths: Sequence[str | Path]) -> list[str]:
        normalized = [_relative(self.root, path)[1] for path in paths]
        if not normalized:
            raise StorageIntegrityError("creation intent requires an allowed path")
        for index, left in enumerate(normalized):
            for right in normalized[index + 1:]:
                if _relative_paths_overlap(left, right):
                    raise StorageIntegrityError("creation intent paths overlap")
        return normalized

    def _path_is_allowed(self, relative: str, allowed_paths: Sequence[str]) -> bool:
        candidate = Path(str(relative).replace("\\", "/"))
        for raw in allowed_paths:
            base = Path(str(raw).replace("\\", "/"))
            if candidate == base or candidate.is_relative_to(base):
                return True
        return False

    def _snapshot_intent_files(self, allowed_paths: Sequence[str]) -> dict[str, dict[str, Any]]:
        with _trace_phase(
            "intent_snapshot_and_hash", campaign_root=self.root,
            operation_name="StorageBudget._snapshot_intent_files",
            allowed_path_count=len(allowed_paths),
        ):
            return self._snapshot_intent_files_impl(allowed_paths)

    def _snapshot_intent_files_impl(
        self, allowed_paths: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for raw in allowed_paths:
            path, relative = _relative(self.root, raw)
            if not path.exists():
                continue
            if path.is_file():
                snapshot[relative] = {"identity": _identity(path), "sha256": _sha256_file(path)}
            elif path.is_dir():
                for item in _iter_regular_files(
                    path, phase_name="intent_directory_scan", campaign_root=self.root,
                ):
                    _, item_relative = _relative(self.root, item)
                    snapshot[item_relative] = {"identity": _identity(item), "sha256": _sha256_file(item)}
        return snapshot

    def create_intent(
        self, *, reservation_id: str, trajectory: str, pulse: int, attempt: int,
        role: str, allowed_paths: Sequence[str | Path], expected_bytes: int | Mapping[str, int],
        admission_hash: str | None = None, intent_id: str | None = None,
        epoch: str | int | None = None, generation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        campaign_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a write-before-target creation intent.

        The intent is deliberately separate from a reservation: a reservation
        accounts bytes, while this record proves which trajectory/attempt is
        allowed to create which paths.  Existing files are snapshotted and can
        never become newly-owned artifacts through this operation.
        """
        rid = str(reservation_id)
        paths = self._intent_paths(allowed_paths)
        if isinstance(expected_bytes, Mapping):
            expected = {str(key).replace("\\", "/"): _nonnegative_int(value, "expected_bytes") for key, value in expected_bytes.items()}
            expected_total = sum(expected.values())
        else:
            expected_total = _nonnegative_int(expected_bytes, "expected_bytes")
            expected = {}
        if expected_total <= 0:
            raise StorageBudgetError("creation intent expected bytes must be positive")
        if generation is not None and not str(generation):
            raise StorageIntegrityError("creation intent generation is empty")
        bound_hash = self.admission_hash if admission_hash is None else str(admission_hash)
        if campaign_id is not None and str(campaign_id) != self.campaign_id:
            raise StorageIntegrityError("creation intent campaign identity conflicts with storage ledger")
        if self.require_intents and not bound_hash:
            raise StorageIntegrityError("formal creation intent requires an admission hash")
        iid = str(intent_id or f"intent-{os.getpid()}-{time.time_ns()}")
        with self._locked(operation_name="StorageBudget.create_intent") as data:
            reservation = data["reservations"].get(rid)
            if not isinstance(reservation, Mapping) or reservation.get("status") != "ACTIVE":
                raise StorageBudgetError("creation intent requires an active reservation")
            allocation_paths = [str(path) for path in reservation.get("allocation_paths", [])]
            if self.require_intents and allocation_paths:
                if any(not any(self._path_is_allowed(path, [allocation]) for allocation in allocation_paths) for path in paths):
                    raise StorageIntegrityError("creation intent is outside reservation allocation paths")
            if bound_hash and self.admission_hash and bound_hash != self.admission_hash:
                raise StorageIntegrityError("creation intent admission hash conflicts with storage ledger")
            if int(reservation.get("bytes", 0)) < expected_total:
                raise StorageBudgetError("creation intent exceeds its active reservation")
            if iid in data["intents"]:
                raise StorageBudgetError(f"creation intent already exists: {iid}")
            initial = self._snapshot_intent_files(paths)
            item = {
                "schema": INTENT_SCHEMA, "intent_id": iid, "campaign_id": self.campaign_id,
                "reservation_id": rid, "trajectory": str(trajectory), "pulse": int(pulse),
                "attempt": int(attempt), "role": str(role), "allowed_paths": paths,
                "expected_bytes": expected_total, "expected_file_bytes": expected,
                "admission_hash": bound_hash, "epoch": None if epoch is None else str(epoch),
                "generation": None if generation is None else str(generation),
                "initial_files": initial, "created_files": {}, "status": "ACTIVE",
                "created_utc": _utc(), "metadata": dict(metadata or {}),
            }
            data["intents"][iid] = item
            data["events"].append({"event": "INTENT_CREATE", "intent_id": iid, "reservation_id": rid, "timestamp_utc": _utc()})
        return dict(item)

    begin_creation_intent = create_intent
    create_reservation_intent = create_intent

    def open_writer(
        self, *, reservation_id: str, intent_id: str, coordinator_epoch: str | int,
        trajectory: str, pulse: int, attempt: int, generation: str,
        process: Mapping[str, Any] | None = None, writer_id: str | None = None,
        writer_kind: str = "LEGACY_WRITER",
    ) -> dict[str, Any]:
        """Register an epoch-bound writer after reservation and intent exist."""
        kind = str(writer_kind).upper()
        if kind not in {"LEGACY_WRITER", "COORDINATOR_RECONCILER"}:
            raise ValueError("writer_kind must be LEGACY_WRITER or COORDINATOR_RECONCILER")
        identity = dict(process or process_identity())
        wid = str(writer_id or f"writer-{identity['pid']}-{time.time_ns()}")
        with self._locked(operation_name="StorageBudget.open_writer") as data:
            reservation = data["reservations"].get(str(reservation_id))
            intent = data["intents"].get(str(intent_id))
            if not isinstance(reservation, Mapping) or reservation.get("status") != "ACTIVE":
                raise StorageBudgetError("writer requires an active reservation")
            if not isinstance(intent, Mapping) or intent.get("status") not in {"ACTIVE", "INTERRUPTED"}:
                raise StorageBudgetError("writer requires an active or resumable intent")
            if str(intent.get("reservation_id")) != str(reservation_id):
                raise StorageIntegrityError("writer reservation differs from intent")
            if intent.get("epoch") is not None and str(intent.get("epoch")) != str(coordinator_epoch):
                prior_writers = [item for item in data["writers"].values()
                                 if str(item.get("intent_id")) == str(intent_id)
                                 and str(item.get("coordinator_epoch")) == str(intent.get("epoch"))]
                fenced = any(item.get("status") == "INTERRUPTED_STALE" for item in prior_writers)
                closed_and_dead = (bool(prior_writers) and all(
                    item.get("status") in {"CLOSED", "INTERRUPTED", "INTERRUPTED_STALE"}
                    and probe_process_identity(item.get("process_identity", {}))["status"] == "DEAD"
                    for item in prior_writers))
                if not (fenced or closed_and_dead):
                    raise StorageIntegrityError("writer intent epoch differs from coordinator")
                rebound = dict(intent)
                rebound["epoch"] = str(coordinator_epoch)
                rebound["status"] = "ACTIVE"
                rebound["takeover_from_epoch"] = str(intent.get("epoch"))
                rebound["takeover_utc"] = _utc()
                data["intents"][str(intent_id)] = rebound
                intent = rebound
            elif intent.get("status") == "INTERRUPTED":
                if any(item.get("status") == "ACTIVE" and str(item.get("intent_id")) == str(intent_id)
                       for item in data["writers"].values()):
                    raise StorageBudgetError("interrupted intent still has an active writer")
                rebound = dict(intent); rebound["status"] = "ACTIVE"
                data["intents"][str(intent_id)] = rebound
                intent = rebound
            active_for_intent = [item for item in data["writers"].values()
                                 if item.get("status") == "ACTIVE"
                                 and str(item.get("intent_id")) == str(intent_id)]
            if active_for_intent:
                # E5-1B WP2 permits one shared Candidate OPTICAL reservation to
                # have one coordinator reconciler plus separate real optical and
                # hydro child writers.  This is ownership sharding, never a
                # second full-z scientific reservation.  All other intents stay
                # single-writer.
                allowed = (str(intent.get("role", "")) == "OPTICAL"
                           and str(intent.get("trajectory", "")) == "C")
                if not allowed:
                    raise StorageBudgetError("creation intent already has an active writer")
            item = {
                "schema": WRITER_SCHEMA, "writer_id": wid, "campaign_id": self.campaign_id,
                "admission_hash": self.admission_hash, "coordinator_epoch": str(coordinator_epoch),
                "process_identity": identity, "trajectory": str(trajectory), "pulse": int(pulse),
                "attempt": int(attempt), "generation": str(generation),
                "reservation_id": str(reservation_id), "intent_id": str(intent_id),
                "role": str(intent.get("role", "")),
                "writer_kind": kind,
                "manager_role": "RECONCILER" if kind == "COORDINATOR_RECONCILER" else None,
                "allowed_paths": list(intent.get("allowed_paths", [])), "status": "ACTIVE",
                "opened_utc": _utc(),
            }
            data["writers"][wid] = item
            data["events"].append({"event": "WRITER_OPEN", "writer_id": wid,
                                   "epoch": str(coordinator_epoch), "timestamp_utc": _utc()})
            return dict(item)

    def open_coordinator_reconciler(
        self, *, reservation_id: str, intent_id: str, coordinator_epoch: str | int,
        trajectory: str, pulse: int, attempt: int, generation: str,
        writer_id: str | None = None,
    ) -> dict[str, Any]:
        """Register the manager-side reconciler, distinct from child writers."""
        return self.open_writer(
            reservation_id=reservation_id, intent_id=intent_id,
            coordinator_epoch=coordinator_epoch, trajectory=trajectory,
            pulse=pulse, attempt=attempt, generation=generation,
            writer_id=writer_id, writer_kind="COORDINATOR_RECONCILER",
        )

    @staticmethod
    def _check_child_grant_nonce(item: Mapping[str, Any], nonce: str) -> None:
        supplied = hashlib.sha256(str(nonce).encode("utf-8")).hexdigest()
        if not hmac.compare_digest(supplied, str(item.get("nonce_sha256", ""))):
            raise StorageIntegrityError("child writer grant nonce is invalid")

    @staticmethod
    def _child_role(role: str, role_index: int | None) -> tuple[str, int | None, str]:
        value = str(role).strip().upper()
        if value.startswith("HYDRO[") and value.endswith("]"):
            if role_index is not None:
                raise StorageIntegrityError("HYDRO role index was supplied twice")
            raw_index = value[6:-1]
            if not raw_index.isdecimal():
                raise StorageIntegrityError("HYDRO grant role index is invalid")
            role_index = int(raw_index)
            value = "HYDRO"
        if value == "OPTICAL":
            if role_index is not None:
                raise StorageIntegrityError("OPTICAL child grant cannot have an index")
            return value, None, value
        if value == "HYDRO":
            if isinstance(role_index, bool) or not isinstance(role_index, int) or role_index < 0:
                raise StorageIntegrityError("HYDRO child grant requires a non-negative index")
            return value, int(role_index), f"HYDRO[{int(role_index)}]"
        if value == "RECONCILER":
            raise StorageIntegrityError("coordinator RECONCILER cannot receive a child writer grant")
        raise StorageIntegrityError("child writer grant role must be OPTICAL or HYDRO[index]")

    def rebind_intent_epoch(
        self, intent_id: str, *, reservation_id: str,
        previous_coordinator_epoch: str | int, coordinator_epoch: str | int,
        trajectory: str, pulse: int, logical_attempt: int, generation: str,
        allowed_paths: Sequence[str | Path], expected_bytes: int,
        admission_identity_sha256: str, scientific_pair_id: str,
        implementation_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Rebind the same intent to a fresh epoch after every old writer is quiescent."""
        iid, rid = str(intent_id), str(reservation_id)
        previous_epoch, new_epoch = str(previous_coordinator_epoch), str(coordinator_epoch)
        paths = self._intent_paths(allowed_paths)
        pulse_value = _nonnegative_int(pulse, "pulse")
        attempt_value = _nonnegative_int(logical_attempt, "logical_attempt")
        expected = _nonnegative_int(expected_bytes, "intent expected_bytes")
        admission = str(admission_identity_sha256)
        pair_id = str(scientific_pair_id)
        if not isinstance(implementation_identity, Mapping) or not implementation_identity:
            raise StorageIntegrityError("intent epoch rebound requires implementation identity")
        implementation = _json_safe(dict(implementation_identity))
        try:
            _canonical_json(implementation)
        except (TypeError, ValueError) as error:
            raise StorageIntegrityError("intent epoch rebound implementation identity is not canonical JSON") from error
        if new_epoch == previous_epoch:
            raise StorageIntegrityError("intent epoch rebound requires a fresh coordinator epoch")
        if (not _is_sha256(admission) or admission != str(self.admission_hash or "")
                or not _is_sha256(pair_id)
                or not str(implementation.get("schema", ""))
                or not _is_sha256(implementation.get("identity_sha256"))):
            raise StorageIntegrityError("intent epoch rebound identity is incomplete")
        with self._locked(operation_name="StorageBudget.rebind_intent_epoch") as data:
            intent = data["intents"].get(iid)
            reservation = data["reservations"].get(rid)
            if not isinstance(intent, Mapping) or intent.get("status") not in {"ACTIVE", "INTERRUPTED"}:
                raise StorageBudgetError("intent epoch rebound requires the original ACTIVE or INTERRUPTED intent")
            if not isinstance(reservation, Mapping) or reservation.get("status") != "ACTIVE":
                raise StorageBudgetError("intent epoch rebound requires the same active aggregate reservation")
            if str(intent.get("reservation_id")) != rid:
                raise StorageIntegrityError("intent epoch rebound reservation differs from original intent")
            if str(intent.get("epoch")) != previous_epoch:
                raise StorageIntegrityError("intent epoch rebound previous epoch differs from original intent")
            original_paths = [str(path) for path in intent.get("allowed_paths", [])]
            if (str(intent.get("campaign_id")) != self.campaign_id
                    or str(intent.get("trajectory")) != str(trajectory)
                    or int(intent.get("pulse", -1)) != pulse_value
                    or int(intent.get("attempt", -1)) != attempt_value
                    or str(intent.get("generation")) != str(generation)
                    or original_paths != paths
                    or int(intent.get("expected_bytes", -1)) != expected
                    or str(intent.get("admission_hash")) != admission):
                raise StorageIntegrityError("intent epoch rebound scope differs from original intent")
            metadata = intent.get("metadata", {})
            if (not isinstance(metadata, Mapping)
                    or str(metadata.get("scientific_pair_id", "")) != pair_id
                    or _canonical_json(metadata.get("implementation_identity")) != _canonical_json(implementation)):
                raise StorageIntegrityError("intent epoch rebound pair or implementation identity differs from persisted intent")
            child_grants = [item for item in data.get("child_writer_grants", {}).values()
                            if str(item.get("intent_id")) == iid]
            for grant in child_grants:
                if grant.get("state") in {"PENDING", "BOUND_ACTIVE"}:
                    raise StorageBudgetError("unresolved child writer grants block intent epoch rebound")
                scope = grant.get("scope", {})
                if (str(grant.get("reservation_id")) != rid
                        or str(scope.get("scientific_pair_id")) != pair_id
                        or _canonical_json(scope.get("implementation_identity")) != _canonical_json(implementation)
                        or str(scope.get("admission_identity_sha256")) != admission
                        or str(scope.get("trajectory")) != str(trajectory)
                        or int(scope.get("pulse", -1)) != pulse_value
                        or int(scope.get("logical_attempt", -1)) != attempt_value
                        or str(scope.get("generation")) != str(generation)):
                    raise StorageIntegrityError("old child writer grant identity differs from intent rebound")
            active_writers = [writer_id for writer_id, writer in data.get("writers", {}).items()
                              if writer.get("status") == "ACTIVE" and str(writer.get("intent_id")) == iid]
            if active_writers:
                raise StorageBudgetError(f"active intent writers block epoch rebound: {active_writers}")
            mutable = dict(intent)
            mutable["epoch"] = new_epoch
            mutable["status"] = "ACTIVE"
            mutable["takeover_from_epoch"] = previous_epoch
            mutable["takeover_utc"] = _utc()
            mutable["takeover_by_process_identity"] = process_identity()
            data["intents"][iid] = mutable
            data["events"].append({"event": "INTENT_EPOCH_REBOUND", "intent_id": iid,
                                   "reservation_id": rid, "previous_epoch": previous_epoch,
                                   "coordinator_epoch": new_epoch, "timestamp_utc": _utc()})
            return dict(mutable)

    def create_child_writer_grant(
        self, *, reservation_id: str, intent_id: str,
        coordinator_epoch: str | int, role: str, trajectory: str,
        pulse: int, logical_attempt: int, generation: str,
        scientific_pair_id: str, admission_identity_sha256: str,
        implementation_identity: Mapping[str, Any],
        allowed_paths: Sequence[str | Path], expected_bytes: int,
        execution_attempt: int = 0, role_index: int | None = None,
        launch_candidate_identity: Mapping[str, Any] | None = None,
        grant_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a bounded PENDING child grant without adding a reservation.

        The caller must pass the admitted implementation manifest and an exact
        path subset of the already-active creation intent.  The returned nonce
        is only returned to the launching coordinator; the ledger stores its
        digest.
        """
        normalized_role, normalized_index, role_label = self._child_role(role, role_index)
        if normalized_role == "HYDRO":
            admission_identity = getattr(self, "admission_identity", None)
            if not isinstance(admission_identity, Mapping):
                admission_identity = getattr(self, "_admission_identity", None)
            topology = admission_identity.get("hydro_topology", {}) if isinstance(admission_identity, Mapping) else {}
            try:
                admitted_hydro_workers = int(topology["candidate_hydro_workers_requested"])
            except (KeyError, TypeError, ValueError) as error:
                raise StorageIntegrityError("admission identity lacks the candidate hydro worker bound") from error
            if admitted_hydro_workers <= 0 or int(normalized_index) >= admitted_hydro_workers:
                raise StorageIntegrityError("HYDRO grant index exceeds admitted worker count")
        rid, iid = str(reservation_id), str(intent_id)
        epoch = str(coordinator_epoch)
        side = str(trajectory)
        pulse_value = _nonnegative_int(pulse, "pulse")
        attempt_value = _nonnegative_int(logical_attempt, "logical_attempt")
        execution_value = _nonnegative_int(execution_attempt, "execution_attempt")
        if not str(generation) or not _is_sha256(scientific_pair_id):
            raise StorageIntegrityError("child writer grant requires generation and scientific pair identity")
        admission = str(admission_identity_sha256)
        if not _is_sha256(admission) or admission != str(self.admission_hash or ""):
            raise StorageIntegrityError("child writer grant admission identity differs from storage ledger")
        if not isinstance(implementation_identity, Mapping) or not implementation_identity:
            raise StorageIntegrityError("child writer grant requires an implementation identity manifest")
        implementation = _json_safe(dict(implementation_identity))
        try:
            _canonical_json(implementation)
        except (TypeError, ValueError) as error:
            raise StorageIntegrityError("child writer grant implementation identity is not canonical JSON") from error
        if (not str(implementation.get("schema", ""))
                or not _is_sha256(implementation.get("identity_sha256"))):
            raise StorageIntegrityError("child writer grant implementation identity manifest is incomplete")
        paths = self._intent_paths(allowed_paths)
        expected = _nonnegative_int(expected_bytes, "grant expected_bytes")
        if expected <= 0:
            raise StorageBudgetError("child writer grant byte bound must be positive")
        candidate = None if launch_candidate_identity is None else _checked_process_identity(launch_candidate_identity)
        gid = str(grant_id or f"grant-{os.getpid()}-{time.time_ns()}-{secrets.token_hex(6)}")
        nonce = secrets.token_urlsafe(32)
        scope = {
            "campaign_id": self.campaign_id,
            "reservation_id": rid,
            "intent_id": iid,
            "coordinator_epoch": epoch,
            "role": normalized_role,
            "role_index": normalized_index,
            "role_label": role_label,
            "trajectory": side,
            "pulse": pulse_value,
            "logical_attempt": attempt_value,
            "execution_attempt": execution_value,
            "generation": str(generation),
            "scientific_pair_id": str(scientific_pair_id),
            "admission_identity_sha256": admission,
            "implementation_identity": implementation,
            "permitted_operations": list(CHILD_GRANT_OPERATIONS[normalized_role]),
            "allowed_paths": paths,
            "expected_bytes": expected,
        }
        with self._locked(operation_name="StorageBudget.create_child_writer_grant") as data:
            reservation = data["reservations"].get(rid)
            intent = data["intents"].get(iid)
            if not isinstance(reservation, Mapping) or reservation.get("status") != "ACTIVE":
                raise StorageBudgetError("child writer grant requires the single active aggregate reservation")
            if not isinstance(intent, Mapping) or intent.get("status") != "ACTIVE":
                raise StorageBudgetError("child writer grant requires an active creation intent")
            if str(intent.get("reservation_id")) != rid:
                raise StorageIntegrityError("child writer grant reservation differs from intent")
            if str(intent.get("campaign_id")) != self.campaign_id:
                raise StorageIntegrityError("child writer grant campaign differs from intent")
            if str(intent.get("epoch")) != epoch:
                raise StorageIntegrityError("child writer grant epoch differs from active intent")
            if (str(intent.get("trajectory")) != side
                    or int(intent.get("pulse", -1)) != pulse_value
                    or int(intent.get("attempt", -1)) != attempt_value
                    or str(intent.get("generation")) != str(generation)):
                raise StorageIntegrityError("child writer grant scope differs from creation intent")
            if str(intent.get("admission_hash")) != admission:
                raise StorageIntegrityError("child writer grant admission differs from creation intent")
            metadata = dict(intent.get("metadata", {}))
            prior_pair = metadata.get("scientific_pair_id")
            prior_implementation = metadata.get("implementation_identity")
            if prior_pair is None and prior_implementation is None:
                metadata["scientific_pair_id"] = str(scientific_pair_id)
                metadata["implementation_identity"] = implementation
                rebound_identity = dict(intent)
                rebound_identity["metadata"] = metadata
                data["intents"][iid] = rebound_identity
                intent = rebound_identity
            elif (str(prior_pair) != str(scientific_pair_id)
                  or _canonical_json(prior_implementation) != _canonical_json(implementation)):
                raise StorageIntegrityError("child writer grant pair or implementation identity differs from intent")
            if expected > int(intent.get("expected_bytes", 0)):
                raise StorageBudgetError("child writer grant byte bound exceeds creation intent")
            intent_paths = [str(path) for path in intent.get("allowed_paths", [])]
            if any(not any(self._path_is_allowed(path, [base]) for base in intent_paths) for path in paths):
                raise StorageIntegrityError("child writer grant path is outside creation intent")
            allocation_paths = [str(path) for path in reservation.get("allocation_paths", [])]
            if allocation_paths and any(not any(self._path_is_allowed(path, [base]) for base in allocation_paths)
                                        for path in paths):
                raise StorageIntegrityError("child writer grant path is outside aggregate reservation")
            grants = data.setdefault("child_writer_grants", {})
            if gid in grants:
                raise StorageBudgetError(f"child writer grant already exists: {gid}")
            for previous in grants.values():
                previous_scope = previous.get("scope", {})
                same_wave_slot = (
                    previous_scope.get("scientific_pair_id") == scope["scientific_pair_id"]
                    and previous_scope.get("coordinator_epoch") == epoch
                    and previous_scope.get("role") == normalized_role
                    and previous_scope.get("role_index") == normalized_index
                )
                if same_wave_slot and previous.get("state") in {"PENDING", "BOUND_ACTIVE"}:
                    raise StorageBudgetError("duplicate unresolved child writer grant role/scope")
                same_execution = (
                    previous_scope.get("scientific_pair_id") == scope["scientific_pair_id"]
                    and previous_scope.get("role") == normalized_role
                    and previous_scope.get("role_index") == normalized_index
                    and previous_scope.get("execution_attempt") == execution_value
                )
                if same_execution:
                    if _canonical_json(previous_scope) == _canonical_json(scope):
                        raise StorageIntegrityError("duplicate child writer grant scope")
                    raise StorageIntegrityError("child writer grant execution attempt cannot be reused with changed scope")
            item = {
                "schema": CHILD_WRITER_GRANT_SCHEMA,
                "grant_id": gid,
                "campaign_id": self.campaign_id,
                "reservation_id": rid,
                "intent_id": iid,
                "coordinator_epoch": epoch,
                "role_label": role_label,
                "scope": scope,
                "nonce_sha256": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
                "launch_candidate_identity": candidate,
                "process_identity": None,
                "state": "PENDING",
                "created_utc": _utc(),
            }
            grants[gid] = item
            data["events"].append({"event": "CHILD_WRITER_GRANT_CREATE", "grant_id": gid,
                                   "role": role_label, "epoch": epoch, "timestamp_utc": _utc()})
            return {**dict(item), "nonce": nonce}

    def record_child_writer_launch_candidate(
        self, grant_id: str, *, nonce: str, process: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attach the launched, waiting child's OS identity to a PENDING grant."""
        identity = _checked_process_identity(process)
        with self._locked(operation_name="StorageBudget.record_child_writer_launch_candidate") as data:
            item = data.setdefault("child_writer_grants", {}).get(str(grant_id))
            if not isinstance(item, Mapping) or item.get("state") != "PENDING":
                raise StorageBudgetError("child writer grant is not PENDING")
            self._check_child_grant_nonce(item, nonce)
            probe = probe_process_identity(identity)
            if probe.get("status") != "LIVE":
                if probe.get("status") == "UNKNOWN":
                    raise StorageIntegrityError("child writer launch candidate identity is UNKNOWN")
                raise StorageBudgetError("child writer launch candidate is already DEAD")
            prior = item.get("launch_candidate_identity")
            if prior is not None and _checked_process_identity(prior) != identity:
                raise StorageIntegrityError("child writer launch candidate identity changed")
            mutable = dict(item)
            mutable["launch_candidate_identity"] = identity
            mutable["launch_candidate_registered_utc"] = _utc()
            data["child_writer_grants"][str(grant_id)] = mutable
            data["events"].append({"event": "CHILD_WRITER_LAUNCH_CANDIDATE", "grant_id": str(grant_id),
                                   "pid": identity["pid"], "timestamp_utc": _utc()})
            return dict(mutable)

    def bind_child_writer_grant(
        self, grant_id: str, *, nonce: str, scope: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically bind PENDING to this process before it performs science writes."""
        if not isinstance(scope, Mapping):
            raise StorageIntegrityError("child writer bind requires the intended scope")
        with self._locked(operation_name="StorageBudget.bind_child_writer_grant") as data:
            item = data.setdefault("child_writer_grants", {}).get(str(grant_id))
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"child writer grant is missing: {grant_id}")
            if item.get("state") != "PENDING":
                raise StorageBudgetError("child writer grant is not PENDING or was already bound")
            self._check_child_grant_nonce(item, nonce)
            if _canonical_json(scope) != _canonical_json(item.get("scope", {})):
                raise StorageIntegrityError("child writer bind scope differs from pending grant")
            identity = _checked_process_identity(process_identity())
            candidate = item.get("launch_candidate_identity")
            if candidate is not None and _checked_process_identity(candidate) != identity:
                raise StorageIntegrityError("child process identity differs from launch candidate")
            probe = probe_process_identity(identity)
            if probe.get("status") != "LIVE":
                raise StorageIntegrityError("child process identity is not verifiably LIVE at bind")
            mutable = dict(item)
            mutable["state"] = "BOUND_ACTIVE"
            mutable["process_identity"] = identity
            mutable["launch_candidate_identity"] = identity
            mutable["bound_utc"] = _utc()
            data["child_writer_grants"][str(grant_id)] = mutable
            data["events"].append({"event": "CHILD_WRITER_GRANT_BIND", "grant_id": str(grant_id),
                                   "role": item.get("role_label"), "pid": identity["pid"],
                                   "timestamp_utc": _utc()})
            return dict(mutable)

    def authorize_child_writer_operation(
        self, grant_id: str, *, nonce: str, coordinator_epoch: str | int,
        operation: str, path: str | Path,
    ) -> dict[str, Any]:
        """Check the bound child's role, epoch, process, operation and path scope."""
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.authorize_child_writer_operation",
            campaign_root=self.root,
        ):
            data = self._read(); self._validate(data)
            item = data.get("child_writer_grants", {}).get(str(grant_id))
            if not isinstance(item, Mapping) or item.get("state") != "BOUND_ACTIVE":
                raise StorageBudgetError("child writer operation requires a BOUND_ACTIVE grant")
            self._check_child_grant_nonce(item, nonce)
            if str(item.get("coordinator_epoch")) != str(coordinator_epoch):
                raise StorageIntegrityError("stale coordinator epoch cannot use child writer grant")
            identity = _checked_process_identity(process_identity())
            if identity != _checked_process_identity(item.get("process_identity", {})):
                raise StorageIntegrityError("only the bound child process can use its writer grant")
            scope = item.get("scope", {})
            operation_name = str(operation).upper()
            if operation_name not in scope.get("permitted_operations", []):
                raise StorageIntegrityError("child writer operation is outside grant role")
            _, relative = _relative(self.root, path)
            if not self._path_is_allowed(relative, scope.get("allowed_paths", [])):
                raise StorageIntegrityError("child writer path is outside grant scope")
            return {
                "status": "PASS",
                "grant_id": str(grant_id),
                "role": item.get("role_label"),
                "coordinator_epoch": str(coordinator_epoch),
                "operation": operation_name,
                "relative_path": relative,
                "expected_bytes": int(scope.get("expected_bytes", 0)),
            }

    def close_child_writer_grant(
        self, grant_id: str, *, nonce: str, coordinator_epoch: str | int,
        status: str = "CLOSED", reason: str | None = None,
    ) -> dict[str, Any]:
        """Let only the bound child close its grant in the same coordinator epoch."""
        if status not in {"CLOSED", "INTERRUPTED"}:
            raise ValueError("child writer grant close status is invalid")
        with self._locked(operation_name="StorageBudget.close_child_writer_grant") as data:
            item = data.setdefault("child_writer_grants", {}).get(str(grant_id))
            if not isinstance(item, Mapping) or item.get("state") != "BOUND_ACTIVE":
                raise StorageBudgetError("child writer grant is not BOUND_ACTIVE")
            self._check_child_grant_nonce(item, nonce)
            if str(item.get("coordinator_epoch")) != str(coordinator_epoch):
                raise StorageIntegrityError("stale coordinator epoch cannot close child writer grant")
            identity = _checked_process_identity(process_identity())
            if identity != _checked_process_identity(item.get("process_identity", {})):
                raise StorageIntegrityError("only the bound child process can close its writer grant")
            mutable = dict(item)
            mutable["state"] = status
            mutable["closed_utc"] = _utc()
            mutable["closed_by_process_identity"] = identity
            if reason is not None:
                mutable["reason"] = str(reason)
            data["child_writer_grants"][str(grant_id)] = mutable
            data["events"].append({"event": "CHILD_WRITER_GRANT_CLOSE", "grant_id": str(grant_id),
                                   "status": status, "timestamp_utc": _utc()})
            return dict(mutable)

    def abandon_pending_writer_grant(self, grant_id: str, *, nonce: str) -> dict[str, Any]:
        """Abandon a pre-bind launch only when its recorded process is proved DEAD."""
        with self._locked(operation_name="StorageBudget.abandon_pending_writer_grant") as data:
            item = data.setdefault("child_writer_grants", {}).get(str(grant_id))
            if not isinstance(item, Mapping) or item.get("state") != "PENDING":
                raise StorageBudgetError("only a PENDING child writer grant can be abandoned")
            self._check_child_grant_nonce(item, nonce)
            candidate = item.get("launch_candidate_identity")
            if candidate is None:
                raise StorageIntegrityError("PENDING grant has no launch candidate death evidence")
            evidence = probe_process_identity(_checked_process_identity(candidate))
            if evidence.get("status") != "DEAD":
                if evidence.get("status") == "UNKNOWN":
                    raise StorageIntegrityError("PENDING grant launch candidate liveness is UNKNOWN")
                raise StorageBudgetError("PENDING grant launch candidate is still LIVE")
            mutable = dict(item)
            mutable["state"] = "ABANDONED"
            mutable["abandoned_utc"] = _utc()
            mutable["liveness_evidence"] = evidence
            data["child_writer_grants"][str(grant_id)] = mutable
            data["events"].append({"event": "CHILD_WRITER_GRANT_ABANDON", "grant_id": str(grant_id),
                                   "timestamp_utc": _utc()})
            return mutable

    def fence_dead_writer_grants(
        self, *, coordinator_epoch: str | int | None = None,
    ) -> dict[str, Any]:
        """Fence dead bound children and abandon dead pre-bind launches atomically."""
        with self._locked(operation_name="StorageBudget.fence_dead_writer_grants") as data:
            grants = data.setdefault("child_writer_grants", {})
            pending = [(gid, item) for gid, item in grants.items()
                       if item.get("state") == "PENDING"
                       and (coordinator_epoch is None or str(item.get("coordinator_epoch")) == str(coordinator_epoch))]
            bound = [(gid, item) for gid, item in grants.items()
                     if item.get("state") == "BOUND_ACTIVE"
                     and (coordinator_epoch is None or str(item.get("coordinator_epoch")) == str(coordinator_epoch))]
            evidence_by_id: dict[str, dict[str, Any]] = {}
            for grant_id, item in pending:
                identity = item.get("launch_candidate_identity")
                evidence_by_id[str(grant_id)] = (
                    {"status": "UNKNOWN", "reason": "launch_candidate_missing"}
                    if identity is None else probe_process_identity(_checked_process_identity(identity))
                )
            for grant_id, item in bound:
                identity = item.get("process_identity")
                evidence_by_id[str(grant_id)] = (
                    {"status": "UNKNOWN", "reason": "bound_process_identity_missing"}
                    if identity is None else probe_process_identity(_checked_process_identity(identity))
                )
            live = sorted(gid for gid, evidence in evidence_by_id.items() if evidence.get("status") == "LIVE")
            unknown = sorted(gid for gid, evidence in evidence_by_id.items() if evidence.get("status") != "DEAD")
            if live:
                raise StorageBudgetError(f"live child writer grants block takeover: {live}")
            if unknown:
                raise StorageIntegrityError(f"unverifiable child writer grants block takeover: {unknown}")
            abandoned: list[str] = []
            fenced: list[str] = []
            for grant_id, _item in pending:
                mutable = dict(grants[grant_id])
                mutable["state"] = "ABANDONED"
                mutable["abandoned_utc"] = _utc()
                mutable["liveness_evidence"] = evidence_by_id[grant_id]
                grants[grant_id] = mutable
                abandoned.append(str(grant_id))
            for grant_id, _item in bound:
                mutable = dict(grants[grant_id])
                mutable["state"] = "INTERRUPTED_STALE"
                mutable["stale_verified_utc"] = _utc()
                mutable["liveness_evidence"] = evidence_by_id[grant_id]
                grants[grant_id] = mutable
                fenced.append(str(grant_id))
            data["events"].append({
                "event": "CHILD_WRITER_GRANTS_FENCED",
                "coordinator_epoch": None if coordinator_epoch is None else str(coordinator_epoch),
                "abandoned_grant_ids": abandoned,
                "interrupted_stale_grant_ids": fenced,
                "timestamp_utc": _utc(),
            })
            return {"status": "PASS", "abandoned_grant_ids": abandoned,
                    "interrupted_stale_grant_ids": fenced, "probes": evidence_by_id}

    def writer_grants(self) -> dict[str, Any]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.writer_grants",
            campaign_root=self.root,
        ):
            data = self._read(); self._validate(data)
            return json.loads(json.dumps(data.get("child_writer_grants", {})))

    def unresolved_writer_grants(
        self, *, coordinator_epoch: str | int | None = None,
    ) -> list[dict[str, Any]]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.unresolved_writer_grants",
            campaign_root=self.root,
        ):
            data = self._read(); self._validate(data)
            return [json.loads(json.dumps(item)) for item in data.get("child_writer_grants", {}).values()
                    if item.get("state") in {"PENDING", "BOUND_ACTIVE"}
                    and (coordinator_epoch is None or str(item.get("coordinator_epoch")) == str(coordinator_epoch))]

    def pending_writer_grants(
        self, *, coordinator_epoch: str | int | None = None,
    ) -> list[dict[str, Any]]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.pending_writer_grants",
            campaign_root=self.root,
        ):
            data = self._read(); self._validate(data)
            return [json.loads(json.dumps(item)) for item in data.get("child_writer_grants", {}).values()
                    if item.get("state") == "PENDING"
                    and (coordinator_epoch is None or str(item.get("coordinator_epoch")) == str(coordinator_epoch))]

    def close_writer(self, writer_id: str, *, coordinator_epoch: str | int,
                     status: str = "CLOSED", reason: str | None = None) -> dict[str, Any]:
        if status not in {"CLOSED", "INTERRUPTED"}:
            raise ValueError("writer close status is invalid")
        with self._locked(operation_name="StorageBudget.close_writer") as data:
            item = data["writers"].get(str(writer_id))
            if not isinstance(item, Mapping) or item.get("status") != "ACTIVE":
                raise StorageBudgetError("writer is not active")
            if str(item.get("coordinator_epoch")) != str(coordinator_epoch):
                raise StorageIntegrityError("stale epoch cannot close writer")
            mutable = dict(item)
            mutable["status"] = status; mutable["closed_utc"] = _utc()
            if reason is not None:
                mutable["reason"] = str(reason)
            data["writers"][str(writer_id)] = mutable
            data["events"].append({"event": "WRITER_CLOSE", "writer_id": str(writer_id),
                                   "status": status, "timestamp_utc": _utc()})
            return mutable

    def active_writers(self, *, coordinator_epoch: str | int | None = None,
                       include_management: bool = True) -> list[dict[str, Any]]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.active_writers",
            campaign_root=self.root,
        ):
            data = self._read(); self._validate(data)
            management_roles = {"TERMINAL_EVIDENCE", "REPORT", "GC_EVIDENCE"}
            active = [dict(item) for item in data["writers"].values()
                      if item.get("status") == "ACTIVE" and
                      (include_management or (str(item.get("role", "")).upper() not in management_roles
                                               and item.get("writer_kind") != "COORDINATOR_RECONCILER")) and
                      (coordinator_epoch is None or str(item.get("coordinator_epoch")) == str(coordinator_epoch))]
            for grant_id, item in data.get("child_writer_grants", {}).items():
                if item.get("state") != "BOUND_ACTIVE":
                    continue
                if coordinator_epoch is not None and str(item.get("coordinator_epoch")) != str(coordinator_epoch):
                    continue
                scope = item.get("scope", {})
                active.append({
                    "schema": CHILD_WRITER_GRANT_SCHEMA,
                    "writer_id": f"child-grant:{grant_id}",
                    "grant_id": str(grant_id),
                    "writer_kind": "CHILD_GRANT",
                    "grant_state": "BOUND_ACTIVE",
                    "status": "ACTIVE",
                    "campaign_id": item.get("campaign_id"),
                    "reservation_id": item.get("reservation_id"),
                    "intent_id": item.get("intent_id"),
                    "coordinator_epoch": item.get("coordinator_epoch"),
                    "trajectory": scope.get("trajectory"),
                    "pulse": scope.get("pulse"),
                    "attempt": scope.get("logical_attempt"),
                    "generation": scope.get("generation"),
                    "role": item.get("role_label"),
                    "process_identity": item.get("process_identity"),
                    "allowed_paths": list(scope.get("allowed_paths", [])),
                })
            return active

    def interrupt_dead_writer(self, writer_id: str, *, probe: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Fence one verifiably dead writer without changing reserved bytes."""
        with self._locked(operation_name="StorageBudget.interrupt_dead_writer") as data:
            wid = str(writer_id)
            item = data["writers"].get(wid)
            if not isinstance(item, Mapping) or item.get("status") != "ACTIVE":
                raise StorageBudgetError(f"writer is not active: {wid}")
            evidence = dict(probe) if probe is not None else probe_process_identity(item.get("process_identity", {}))
            if evidence.get("status") != "DEAD":
                raise StorageBudgetError(f"writer is not verifiably dead: {wid}")
            mutable = dict(item)
            mutable["status"] = "INTERRUPTED_STALE"
            mutable["stale_verified_utc"] = _utc()
            mutable["liveness_evidence"] = evidence
            data["writers"][wid] = mutable
            intent_id = str(mutable.get("intent_id", ""))
            intent = data["intents"].get(intent_id)
            if isinstance(intent, Mapping) and intent.get("status") == "ACTIVE":
                interrupted = dict(intent)
                interrupted["status"] = "INTERRUPTED"
                interrupted["interrupted_utc"] = _utc()
                interrupted["interrupted_by_dead_writer"] = wid
                data["intents"][intent_id] = interrupted
            data["events"].append({
                "event": "WRITER_INTERRUPTED_STALE", "writer_id": wid,
                "single_writer": True, "timestamp_utc": _utc(),
            })
            return mutable

    def interrupt_stale_writers(self, *, coordinator_epoch: str | int) -> dict[str, Any]:
        """Verify dead writers and fence them without releasing reserved bytes."""
        with self._locked(operation_name="StorageBudget.interrupt_stale_writers") as data:
            active = [(wid, item) for wid, item in data["writers"].items()
                      if item.get("status") == "ACTIVE" and
                      str(item.get("coordinator_epoch")) == str(coordinator_epoch)]
            probes = [(wid, probe_process_identity(item.get("process_identity", {}))) for wid, item in active]
            live = [wid for wid, probe in probes if probe["status"] == "LIVE"]
            unknown = [wid for wid, probe in probes if probe["status"] == "UNKNOWN"]
            if live:
                raise StorageBudgetError(f"live ACTIVE writers block takeover: {live}")
            if unknown:
                raise StorageIntegrityError(f"unverifiable ACTIVE writers block takeover: {unknown}")
            changed = []
            for wid, probe in probes:
                mutable = dict(data["writers"][wid]); mutable["status"] = "INTERRUPTED_STALE"
                mutable["stale_verified_utc"] = _utc(); mutable["liveness_evidence"] = probe
                data["writers"][wid] = mutable; changed.append(wid)
                intent_id = str(mutable.get("intent_id", ""))
                intent = data["intents"].get(intent_id)
                if isinstance(intent, Mapping) and intent.get("status") == "ACTIVE":
                    interrupted = dict(intent)
                    interrupted["status"] = "INTERRUPTED"
                    interrupted["interrupted_utc"] = _utc()
                    interrupted["interrupted_by_stale_epoch"] = str(coordinator_epoch)
                    data["intents"][intent_id] = interrupted
            data["events"].append({"event": "WRITERS_INTERRUPTED_STALE", "epoch": str(coordinator_epoch),
                                   "writer_ids": changed, "timestamp_utc": _utc()})
            return {"status": "PASS", "coordinator_epoch": str(coordinator_epoch),
                    "interrupted_writer_ids": changed, "probes": dict(probes)}

    def write_quiescence_receipt(self, path: str | Path, *, coordinator_epoch: str | int,
                                 trajectory: str | None = None, pulse: int | None = None,
                                 attempt: int | None = None) -> dict[str, Any]:
        active = self.active_writers(coordinator_epoch=coordinator_epoch, include_management=True)
        if active:
            raise StorageBudgetError("writer registry is not quiescent")
        if self.pending_writer_grants(coordinator_epoch=coordinator_epoch):
            raise StorageBudgetError("pending child writer grants block quiescence")
        destination, _ = _relative(self.root, path)
        payload = {"schema": WRITER_RECEIPT_SCHEMA, "status": "PASS", "campaign_id": self.campaign_id,
                   "active_writers": [], "writer_epoch": str(coordinator_epoch),
                   "coordinator_process_id": str(coordinator_epoch),
                   "admission_identity_sha256": self.admission_hash,
                   "trajectory": trajectory, "pulse": pulse, "attempt": attempt,
                   "registry_backed": True, "created_utc": _utc()}
        _atomic_json(destination, payload, overwrite=False)
        return {"path": str(destination), "sha256": _sha256_file(destination), **payload}

    def _intent_file_list(self, item: Mapping[str, Any], files: Sequence[str | Path] | None) -> list[Path]:
        allowed = [str(path) for path in item.get("allowed_paths", [])]
        if files is None:
            values: list[Path] = []
            for raw in allowed:
                path, _ = _relative(self.root, raw)
                if path.is_file():
                    values.append(path)
                elif path.is_dir():
                    values.extend(_iter_regular_files(path))
        else:
            values = []
            for raw in files:
                path, relative = _relative(self.root, raw)
                if not self._path_is_allowed(relative, allowed):
                    raise StorageIntegrityError(f"creation target is outside intent paths: {relative}")
                values.append(path)
        result: list[Path] = []
        seen: set[str] = set()
        initial = item.get("initial_files", {})
        for path in values:
            _, relative = _relative(self.root, path)
            if relative in seen:
                continue
            seen.add(relative)
            if relative in initial:
                raise StorageIntegrityError(f"pre-existing file cannot be owned by creation intent: {relative}")
            if not path.is_file() or _is_reparse(path):
                raise StorageIntegrityError(f"creation target is missing or linked: {relative}")
            result.append(path)
        return result

    def complete_intent(
        self, intent_id: str, *, files: Sequence[str | Path] | None = None,
        expected_bytes: int | None = None, role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate files created under an intent and register ownership."""
        iid = str(intent_id)
        with self._locked(operation_name="StorageBudget.complete_intent") as data:
            item = data["intents"].get(iid)
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"unknown creation intent: {iid}")
            self._ensure_child_grants_quiescent(
                data, reservation_id=str(item.get("reservation_id", "")), intent_id=iid,
            )
            if item.get("status") == "COMPLETED":
                return dict(item)
            if item.get("status") != "ACTIVE":
                raise StorageBudgetError(f"creation intent is not active: {iid}")
            values = self._intent_file_list(item, files)
            if not values:
                raise StorageIntegrityError("creation intent completed without files")
            expected_total = _nonnegative_int(item.get("expected_bytes", 0), "intent expected bytes")
            actual_total = 0
            created: dict[str, Any] = {}
            for path in values:
                resolved, relative = _relative(self.root, path)
                identity = _identity(resolved)
                digest = _sha256_file(resolved)
                per_file = item.get("expected_file_bytes", {}).get(relative)
                if per_file is not None and identity["size"] != int(per_file):
                    raise StorageIntegrityError(f"creation target size differs from intent: {relative}")
                if relative in data["artifacts"]:
                    raise StorageIntegrityError(f"creation target already has ownership: {relative}")
                actual_total += identity["size"]
                created[relative] = {"identity": identity, "sha256": digest}
                data["artifacts"][relative] = {
                    "campaign_id": self.campaign_id,
                    "trajectory": str(item.get("trajectory")), "pulse": int(item.get("pulse")),
                    "attempt": int(item.get("attempt")),
                    "role": str(role or item.get("role")), "relative_path": relative,
                    "identity": identity, "expected_sha256": digest, "sha256": digest,
                    "reclaimable": bool((metadata or {}).get("reclaimable", False)),
                    "metadata": {**dict(item.get("metadata", {})), **dict(metadata or {})},
                    "intent_id": iid, "reservation_id": str(item.get("reservation_id")),
                    "admission_hash": item.get("admission_hash"),
                    "creation_record": {"intent_id": iid, "generation": item.get("generation"),
                                        "created_utc": item.get("created_utc")},
                    "registered_utc": _utc(),
                }
            if expected_bytes is not None and actual_total > _nonnegative_int(expected_bytes, "expected_bytes"):
                raise StorageBudgetError("created files exceed completion byte bound")
            if actual_total > expected_total:
                raise StorageBudgetError("created files exceed intent byte bound")
            mutable = dict(item)
            mutable["status"] = "COMPLETED"
            mutable["created_files"] = created
            mutable["actual_bytes"] = actual_total
            mutable["completed_utc"] = _utc()
            data["intents"][iid] = mutable
            data["events"].append({"event": "INTENT_COMPLETE", "intent_id": iid, "actual_bytes": actual_total, "timestamp_utc": _utc()})
            return dict(mutable)

    register_intent_files = complete_intent
    complete_creation_intent = complete_intent

    def interrupt_intent(self, intent_id: str, *, reason: str = "interrupted") -> dict[str, Any]:
        iid = str(intent_id)
        with self._locked(operation_name="StorageBudget.interrupt_intent") as data:
            item = data["intents"].get(iid)
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"unknown creation intent: {iid}")
            mutable = dict(item)
            if mutable.get("status") == "COMPLETED":
                return mutable
            mutable["status"] = "INTERRUPTED"
            mutable["interrupt_reason"] = str(reason)
            mutable["interrupted_utc"] = _utc()
            data["intents"][iid] = mutable
            data["events"].append({"event": "INTENT_INTERRUPTED", "intent_id": iid, "reason": str(reason), "timestamp_utc": _utc()})
            return mutable

    abort_intent = interrupt_intent
    interrupt_creation_intent = interrupt_intent

    def writer_receipt(self, writer_id: str) -> dict[str, Any]:
        """Return one writer registration for a controlled worker entry."""
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.writer_receipt",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            item = data.get("writers", {}).get(str(writer_id))
            if not isinstance(item, Mapping):
                raise StorageIntegrityError(f"writer receipt is missing: {writer_id}")
            return json.loads(json.dumps(item))

    def intent(self, intent_id: str) -> dict[str, Any]:
        """Return one creation intent for a controlled worker entry."""
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.intent",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            item = data.get("intents", {}).get(str(intent_id))
            if not isinstance(item, Mapping):
                raise StorageIntegrityError(f"creation intent is missing: {intent_id}")
            return json.loads(json.dumps(item))

    def admission_identity(self) -> dict[str, Any]:
        """Return the admission identity bound to the durable ledger."""
        if not isinstance(self._admission_identity, Mapping):
            raise StorageIntegrityError("storage ledger has no admission identity")
        return json.loads(json.dumps(self._admission_identity))

    @classmethod
    def open_for_worker(cls, root: str | Path, *, campaign_id: str,
                        admission_identity: Mapping[str, Any],
                        lock_timeout_s: float = 10.0) -> "StorageBudget":
        """Open a worker view that shares the coordinator ledger lock."""
        budget_identity = (admission_identity or {}).get("budget_identity", {})
        instance = cls.__new__(cls)
        instance.lock_timeout_s = cls._checked_lock_timeout(lock_timeout_s)
        instance.root = Path(root).resolve()
        instance.ledger_path = instance.root / "storage_ledger.json"
        instance.lock_path = instance.root / ".storage_budget.lock"
        instance.cap_bytes = int(budget_identity.get("max_campaign_live_bytes", HARD_CAP_BYTES))
        instance.final_output_budget_bytes = int(budget_identity.get("final_output_budget_bytes", DEFAULT_FINAL_OUTPUT_BUDGET_BYTES))
        instance.safety_margin_bytes = int(budget_identity.get("safety_margin_bytes", DEFAULT_SAFETY_MARGIN_BYTES))
        instance.external_prefix_bytes = int(budget_identity.get("external_prefix_bytes", 0))
        instance.frozen_lut_input_bytes = int(budget_identity.get("frozen_lut_input_bytes", 0))
        instance.planned_final_output_bytes = int(budget_identity.get("final_output_bytes", 0))
        instance.external_full_source_bytes = int(budget_identity.get("existing_full_source_bytes", 0))
        instance.external_canonical_source_identity = _validated_external_canonical_source_identity(
            instance.root, admission_identity,
        )
        instance.external_canonical_source_bytes = (
            0 if instance.external_canonical_source_identity is None else
            int(instance.external_canonical_source_identity["source_manifest_bytes"])
            + int(instance.external_canonical_source_identity["artifact_bytes"])
            + int(instance.external_canonical_source_identity["seal_bytes"])
        )
        instance.require_intents = True
        instance.campaign_id = str(campaign_id)
        instance._admission_identity = dict(admission_identity)
        instance.admission_hash = str((admission_identity or {}).get("identity_sha256", ""))
        instance.provider = MockQuotaProvider(free_bytes=None, quota_bytes=None)
        with _FileLock(
            instance.lock_path, timeout_s=instance.lock_timeout_s, operation_name="StorageBudget.open_for_worker",
            campaign_root=instance.root,
        ):
            data = instance._read()
            instance.require_quota = bool(data.get("require_quota", False))
            instance._validate(data)
        return instance


    def intents(self) -> dict[str, Any]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.intents",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            return json.loads(json.dumps(data.get("intents", {})))

    def reservations(self) -> dict[str, Any]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.reservations",
            campaign_root=self.root,
        ):
            data = self._read(); self._validate(data)
            return json.loads(json.dumps(data.get("reservations", {})))

    creation_intents = intents

    def validate_intent(self, intent_id: str, *, path: str | Path | None = None,
                        require_completed: bool = False, require_active: bool = False) -> dict[str, Any]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.validate_intent",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            item = data.get("intents", {}).get(str(intent_id))
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"unknown creation intent: {intent_id}")
            if require_completed and item.get("status") != "COMPLETED":
                raise StorageBudgetError("creation intent is not complete")
            if require_active:
                reservation_id = str(item.get("reservation_id", ""))
                reservation = data.get("reservations", {}).get(reservation_id)
                if not isinstance(reservation, Mapping) or reservation.get("status") != "ACTIVE":
                    raise StorageBudgetError("creation intent reservation is not active")
            if path is not None:
                _, relative = _relative(self.root, path)
                if not self._path_is_allowed(relative, item.get("allowed_paths", [])):
                    raise StorageIntegrityError("path is outside creation intent")
            return dict(item)

    validate_creation_intent = validate_intent

    def register_artifact(
        self, path: str | Path, *, role: str, trajectory: str | None = None,
        pulse: int | None = None, attempt: int | None = None,
        reclaimable: bool = False, expected_sha256: str | None = None,
        expected_bytes: int | None = None, metadata: Mapping[str, Any] | None = None,
        intent_id: str | None = None, reservation_id: str | None = None,
        admission_hash: str | None = None, legacy_test_only: bool = False,
    ) -> dict[str, Any]:
        resolved, relative = _relative(self.root, path)
        identity = _identity(resolved)
        if self.require_intents and legacy_test_only:
            raise StorageIntegrityError("formal artifact registration cannot use legacy_test_only")
        if self.require_intents and intent_id is None:
            raise StorageIntegrityError("formal artifact registration requires a creation intent")
        intent = None
        if intent_id is not None:
            intent = self.validate_intent(intent_id, path=resolved,
                                          require_completed=self.require_intents)
            if intent.get("status") not in {"ACTIVE", "COMPLETED"}:
                raise StorageIntegrityError("artifact intent is not active or complete")
            if self.require_intents:
                created = intent.get("created_files", {})
                if not isinstance(created, Mapping) or relative not in created:
                    raise StorageIntegrityError("formal artifact is not listed in a completed intent creation record")
                expected_created = created[relative]
                if not isinstance(expected_created, Mapping) or dict(expected_created.get("identity", {})) != identity:
                    raise StorageIntegrityError("formal artifact identity differs from completed creation record")
            if reservation_id is not None and str(reservation_id) != str(intent.get("reservation_id")):
                raise StorageIntegrityError("artifact reservation does not match creation intent")
            if admission_hash is not None and str(admission_hash) != str(intent.get("admission_hash")):
                raise StorageIntegrityError("artifact admission hash does not match creation intent")
            if trajectory is not None and str(trajectory) != str(intent.get("trajectory")):
                raise StorageIntegrityError("artifact trajectory does not match creation intent")
            if pulse is not None and int(pulse) != int(intent.get("pulse")):
                raise StorageIntegrityError("artifact pulse does not match creation intent")
            if attempt is not None and int(attempt) != int(intent.get("attempt")):
                raise StorageIntegrityError("artifact attempt does not match creation intent")
        if expected_bytes is not None and identity["size"] != _nonnegative_int(expected_bytes, "expected_bytes"):
            raise StorageIntegrityError("artifact size differs from expected size")
        digest = _sha256_file(resolved)
        if expected_sha256 is not None and digest != str(expected_sha256):
            raise StorageIntegrityError("artifact hash differs from expected hash")
        effective_admission_hash = None if intent is None else intent.get("admission_hash")
        record = {
            "campaign_id": self.campaign_id,
            "trajectory": None if trajectory is None else str(trajectory),
            "pulse": None if pulse is None else int(pulse),
            "attempt": None if attempt is None else int(attempt),
            "role": str(role),
            "relative_path": relative,
            "identity": identity,
            "expected_sha256": None if expected_sha256 is None else str(expected_sha256),
            "sha256": digest,
            "reclaimable": bool(reclaimable),
            "metadata": dict(metadata or {}),
            "intent_id": None if intent_id is None else str(intent_id),
            "reservation_id": None if reservation_id is None else str(reservation_id or (intent or {}).get("reservation_id")),
            "admission_hash": (effective_admission_hash if admission_hash is None else str(admission_hash)),
            "creation_record": None if intent is None else {"intent_id": str(intent_id),
                "generation": intent.get("generation"), "created_utc": intent.get("created_utc")},
            "registered_utc": _utc(),
        }
        key = relative
        with self._locked(operation_name="StorageBudget.register_artifact") as data:
            existing = data["artifacts"].get(key)
            if existing is not None and dict(existing.get("identity", {})) != identity:
                raise StorageIntegrityError("artifact identity changed for an existing ledger path")
            data["artifacts"][key] = record
            data["events"].append({"event": "REGISTER", "relative_path": relative, "role": str(role), "reclaimable": bool(reclaimable), "timestamp_utc": _utc()})
        return record

    def register_file(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        return self.register_artifact(path, **kwargs)

    def artifacts(self) -> dict[str, Any]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.artifacts",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            return json.loads(json.dumps(data["artifacts"]))

    def validate_paired_exact_ownership(self, bindings: Sequence[Mapping[str, Any]]) -> None:
        """Validate one exact report's durable ownership in one ledger snapshot.

        The caller has already checked every present array, file hash and file
        identity.  Keep the ledger lock for the whole ownership pass so that
        GC and intent transitions cannot mix records from different versions.
        """
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.validate_paired_exact_ownership",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            for binding in bindings:
                side = str(binding["side"])
                path, relative = _relative(self.root, binding["path"])
                descriptor = binding["descriptor"]
                creation = descriptor["creation_record"]
                artifact = data["artifacts"].get(relative)
                if not isinstance(artifact, Mapping):
                    raise ValueError(f"formal exact {side} object has no storage ownership record")
                if str(artifact.get("campaign_id", "")) != str(binding["campaign_id"]):
                    raise ValueError(f"formal exact {side} storage ownership campaign changed")
                intent_id = str(creation.get("intent_id"))
                if str(artifact.get("intent_id", "")) != intent_id:
                    raise ValueError(f"formal exact {side} storage ownership intent changed")
                digest = str(binding["file_sha256"])
                if str(artifact.get("sha256", artifact.get("expected_sha256", ""))) != digest:
                    raise ValueError(f"formal exact {side} storage ownership file hash changed")
                if dict(artifact.get("identity", {})) != dict(descriptor.get("file_identity", {})):
                    raise ValueError(f"formal exact {side} storage ownership file identity changed")
                intent = data["intents"].get(intent_id)
                if (not isinstance(intent, Mapping) or intent.get("status") != "COMPLETED"
                        or not self._path_is_allowed(relative, intent.get("allowed_paths", []))):
                    raise ValueError(f"formal exact {side} creation intent is not durably complete")
                artifact_creation = artifact.get("creation_record")
                if not isinstance(artifact_creation, Mapping):
                    raise ValueError(f"formal exact {side} artifact creation record is incomplete")
                generation = str(creation.get("generation", ""))
                if str(artifact_creation.get("generation", "")) != generation:
                    raise ValueError(f"formal exact {side} storage ownership generation changed")
                if str(intent.get("generation", "")) != generation:
                    raise ValueError(f"formal exact {side} completed intent generation changed")
                if str(intent.get("admission_hash", "")) != str(self.admission_hash or ""):
                    raise ValueError(f"formal exact {side} completed intent admission changed")
                for field in ("trajectory", "pulse", "attempt"):
                    if str(artifact.get(field)) != str(descriptor.get(field)):
                        raise ValueError(f"formal exact {side} storage ownership {field} changed")
                    if str(intent.get(field)) != str(descriptor.get(field)):
                        raise ValueError(f"formal exact {side} completed intent {field} changed")
                if not bool(binding["present"]):
                    if path.exists():
                        raise ValueError(f"formal exact {side} missing object reappeared")
                    reclaimed = False
                    for plan in data.get("gc_plans", {}).values():
                        if (not isinstance(plan, Mapping) or plan.get("schema") != GC_SCHEMA
                                or str(plan.get("campaign_id")) != str(binding["campaign_id"])
                                or str(plan.get("trajectory")) != side
                                or int(plan.get("pulse", -1)) != int(descriptor["pulse"])
                                or int(plan.get("attempt", -1)) != int(descriptor["attempt"])
                                or plan.get("status") == "ABORTED"):
                            continue
                        for target in plan.get("targets", []):
                            if (str(target.get("relative_path")) == relative
                                    and target.get("status") in {"DELETE_INTENT", "DELETED"}
                                    and dict(target.get("identity", {})) == dict(descriptor.get("file_identity", {}))
                                    and str(target.get("artifact_sha256", "")) == digest):
                                reclaimed = True
                                break
                        if reclaimed:
                            break
                    if not reclaimed:
                        raise ValueError(f"formal exact {side} missing object lacks durable GC evidence")

    def register_preexisting_prefix(self, path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
        """Account one immutable prefix already present inside the campaign root."""
        checked, relative = _relative(self.root, path)
        current = _identity(checked)
        if int(current.get("st_nlink", 1)) != 1:
            raise StorageIntegrityError("internal prefix is hard-linked")
        digest = _sha256_file(checked)
        if digest != str(expected_sha256):
            raise StorageIntegrityError("internal prefix raw identity changed")
        with self._locked(operation_name="StorageBudget.register_preexisting_prefix") as data:
            existing = data["artifacts"].get(relative)
            if existing is not None:
                if (existing.get("role") != "EXTERNAL_PREFIX" or existing.get("campaign_id") != self.campaign_id
                        or existing.get("sha256") != digest or existing.get("identity") != current
                        or existing.get("external_preexisting") is not True):
                    raise StorageIntegrityError("internal prefix ownership changed")
                return dict(existing)
            item = {"campaign_id": self.campaign_id, "trajectory": None, "pulse": None,
                    "attempt": 0, "role": "EXTERNAL_PREFIX", "relative_path": relative,
                    "identity": current, "sha256": digest, "expected_sha256": digest,
                    "external_preexisting": True, "reclaimable": False,
                    "registered_utc": _utc()}
            data["artifacts"][relative] = item
            data["events"].append({"event": "PREEXISTING_PREFIX_BIND", "path": relative,
                                   "sha256": digest, "timestamp_utc": _utc()})
            return dict(item)

    def register_preexisting_frozen_lut(
        self, path: str | Path, *, expected_sha256: str, expected_size_bytes: int,
        admission_identity_sha256: str,
    ) -> dict[str, Any]:
        """Bind one pre-staged immutable rate table or its staging receipt."""
        checked, relative = _relative(self.root, path)
        provenance = _identity(checked)
        stat_result = checked.stat()
        expected_size = _nonnegative_int(expected_size_bytes, "expected_size_bytes")
        uid = getattr(stat_result, "st_uid", None)
        effective_uid = getattr(os, "geteuid", None)
        if (callable(effective_uid) and uid is not None
                and int(uid) != int(effective_uid())):
            raise StorageIntegrityError("frozen LUT input is not owned by the current user")
        current = {"size": int(stat_result.st_size), "st_nlink": int(provenance["st_nlink"])}
        if uid is not None:
            current["st_uid"] = int(uid)
        gid = getattr(stat_result, "st_gid", None)
        if gid is not None:
            current["st_gid"] = int(gid)
        if expected_size <= 0 or current["st_nlink"] != 1:
            raise StorageIntegrityError("frozen LUT input is empty or hard-linked")
        if int(current["size"]) != expected_size:
            raise StorageIntegrityError("frozen LUT input size differs from admission")
        digest = _sha256_file(checked)
        if digest != str(expected_sha256):
            raise StorageIntegrityError("frozen LUT input hash differs from admission")
        admission_hash = str(admission_identity_sha256)
        if not admission_hash or admission_hash != str(self.admission_hash or ""):
            raise StorageIntegrityError("frozen LUT input admission binding differs")
        staging = (self.admission_identity or {}).get("lut_identity", {}).get("staging", {})
        registered_inputs = staging.get("registered_inputs", []) if isinstance(staging, Mapping) else []
        binding = next((item for item in registered_inputs
                        if isinstance(item, Mapping) and str(item.get("path", "")) == relative), None)
        if (not isinstance(binding, Mapping)
                or str(binding.get("sha256", "")) != digest
                or int(binding.get("size_bytes", -1)) != expected_size
                or str(binding.get("role", "")) != "INPUT_LUT"):
            raise StorageIntegrityError("frozen LUT input is not listed in admitted staging identity")
        input_kind = str(binding.get("input_kind", "frozen_lut"))
        with self._locked(operation_name="StorageBudget.register_preexisting_frozen_lut") as data:
            existing = data["artifacts"].get(relative)
            if existing is not None:
                if (existing.get("role") != "INPUT_LUT"
                        or existing.get("campaign_id") != self.campaign_id
                        or existing.get("relative_path") != relative
                        or existing.get("trajectory") is not None
                        or existing.get("pulse") is not None
                        or existing.get("attempt") != 0
                        or existing.get("sha256") != digest
                        or existing.get("identity") != current
                        or existing.get("external_preexisting") is not True
                        or existing.get("reclaimable") is not False
                        or existing.get("admission_hash") != admission_hash
                        or existing.get("metadata") != {"input_kind": input_kind}):
                    raise StorageIntegrityError("frozen LUT input ownership changed")
                return dict(existing)
            item = {
                "campaign_id": self.campaign_id, "trajectory": None, "pulse": None,
                "attempt": 0, "role": "INPUT_LUT", "relative_path": relative,
                "identity": current, "sha256": digest, "expected_sha256": digest,
                "external_preexisting": True, "reclaimable": False,
                "admission_hash": admission_hash,
                "metadata": {"input_kind": input_kind},
                "provenance": {
                    "st_dev": int(provenance["st_dev"]), "st_ino": int(provenance["st_ino"]),
                    "mtime_ns": int(provenance["mtime_ns"]),
                },
                "registered_utc": _utc(),
            }
            data["artifacts"][relative] = item
            data["events"].append({"event": "PREEXISTING_LUT_INPUT_BIND", "path": relative,
                                   "sha256": digest, "timestamp_utc": _utc()})
            return dict(item)

    def set_reclaimable(self, paths: Sequence[str | Path], *, value: bool = True) -> list[str]:
        """Set retention only for artifacts already owned by completed formal intents."""
        changed: list[str] = []
        with self._locked(operation_name="StorageBudget.set_reclaimable") as data:
            for raw in paths:
                _, relative = _relative(self.root, raw)
                item = data["artifacts"].get(relative)
                if not isinstance(item, Mapping) or not item.get("intent_id"):
                    raise StorageIntegrityError(f"retention target lacks creation ownership: {relative}")
                intent = data["intents"].get(str(item["intent_id"]))
                if not isinstance(intent, Mapping) or intent.get("status") != "COMPLETED":
                    raise StorageIntegrityError(f"retention target intent is not complete: {relative}")
                mutable = dict(item); mutable["reclaimable"] = bool(value)
                data["artifacts"][relative] = mutable; changed.append(relative)
            data["events"].append({"event": "RETENTION_SET", "paths": changed,
                                   "reclaimable": bool(value), "timestamp_utc": _utc()})
        return changed

    def final_output_bytes(self) -> int:
        if (self.root / "E5_1A_TERMINAL_ROLE_CONTRACT.json").is_file():
            # The terminal view charges the complete retained tree, including
            # ledger/lock and every durable receipt, exactly once.
            return (self.actual_bytes() + self.external_prefix_bytes
                    + self.external_canonical_source_bytes)
        total = self.external_prefix_bytes + self.external_canonical_source_bytes
        total += self.frozen_lut_input_bytes
        for relative, item in self.artifacts().items():
            if (str(item.get("role", "")).startswith("final")
                    or str(relative).endswith("/final_optical_field.npy")):
                path, _ = _relative(self.root, relative)
                if path.is_file():
                    total += int(path.stat().st_size)
        return total

    def check_final_output_budget(self, additional_bytes: int = 0) -> dict[str, Any]:
        forecast = self.final_output_bytes() + _nonnegative_int(additional_bytes, "additional_bytes")
        total = max(forecast, self.planned_final_output_bytes)
        if total > self.final_output_budget_bytes:
            raise StorageBudgetError("final output budget exceeded")
        return {"status": "PASS", "final_output_bytes": total, "final_output_budget_bytes": self.final_output_budget_bytes}

    def validate_terminal_inventory(
        self, expected_roles: Mapping[str, Sequence[str | Path]] | Sequence[str], *,
        root: str | Path | None = None, final: bool = True,
    ) -> dict[str, Any]:
        """Verify the durable role manifest against every actual file.

        Unknown NPY/NPZ files, orphan files, old attempts and unregistered
        terminal payloads are reported and fail closed.  The method never
        claims ownership or deletes an extra file.
        """
        base = self.root if root is None else Path(root).resolve()
        if not base.is_relative_to(self.root):
            raise StorageIntegrityError("terminal inventory root escapes campaign root")
        def resolve_expected(value: str | Path) -> str:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            return str(candidate.resolve())
        if isinstance(expected_roles, Mapping):
            expected_map = {
                str(role): {resolve_expected(path) for path in paths}
                for role, paths in expected_roles.items()
            }
        else:
            expected_map = {"final": {resolve_expected(path) for path in expected_roles}}
        expected_paths = set().union(*expected_map.values()) if expected_map else set()
        actual_paths = {
            str(path.resolve()) for path in _iter_regular_files(
                base, phase_name="terminal_root_inventory_scan", campaign_root=self.root,
            )
        }
        management = {str(path.resolve()) for path in self._management_paths()}
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.validate_terminal_inventory",
            campaign_root=self.root,
        ):
            ledger = self._read()
            self._validate(ledger)
        artifacts = json.loads(json.dumps(ledger["artifacts"]))
        active_reservations = sorted(
            str(reservation_id) for reservation_id, item in ledger["reservations"].items()
            if item.get("status") == "ACTIVE"
        )
        unfinished_intents = sorted(
            str(intent_id) for intent_id, item in ledger["intents"].items()
            if item.get("status") in {"ACTIVE", "INTERRUPTED"}
        )
        active_writers = sorted(
            str(writer_id) for writer_id, item in ledger["writers"].items()
            if item.get("status") == "ACTIVE"
        )
        if active_reservations or unfinished_intents or active_writers:
            raise StorageIntegrityError(
                "terminal storage ledger is not quiescent: "
                f"active_reservations={active_reservations} "
                f"unfinished_intents={unfinished_intents} active_writers={active_writers}"
            )
        registered = {str((self.root / relative).resolve()) for relative in artifacts}
        scientific_suffixes = {".npy", ".npz", ".mat", ".h5", ".hdf5"}
        suspicious_unregistered = {
            path for path in actual_paths - management - registered
            if Path(path).suffix.lower() in scientific_suffixes
        }
        # Durable orchestration manifests and locks are management state.  All
        # registered evidence remains in the contract, while any unregistered
        # scientific container is still an orphan and fails closed.
        actual_payload = (registered & actual_paths) | suspicious_unregistered
        missing = sorted(path for path in expected_paths if path not in actual_payload)
        extras = sorted(path for path in actual_payload if path not in expected_paths)
        unregistered = sorted(path for path in actual_payload if path not in registered)
        role_mismatches: list[str] = []
        for role, paths in expected_map.items():
            for path in paths & actual_payload:
                try:
                    relative = Path(path).relative_to(self.root).as_posix()
                except ValueError:
                    continue
                record = artifacts.get(relative)
                if record is None or str(record.get("role", "")) != str(role):
                    role_mismatches.append(relative)
        final_bytes = 0
        if final:
            final_bytes = (self.actual_bytes() + self.external_prefix_bytes
                           + self.external_canonical_source_bytes)
            if final_bytes > self.final_output_budget_bytes:
                raise StorageBudgetError("terminal final output exceeds final budget")
        ok = not missing and not extras and not unregistered and not role_mismatches
        if not ok:
            raise StorageIntegrityError(
                "terminal inventory mismatch: "
                f"missing={missing} extras={extras} unregistered={unregistered} role_mismatches={role_mismatches}"
            )
        return {
            "schema": "khz_filament.hr4e5.e5_1a.terminal_inventory.v1",
            "status": "PASS", "root": str(base), "expected_count": len(expected_paths),
            "actual_count": len(actual_payload), "final_output_bytes": final_bytes,
            "external_canonical_source_bytes": self.external_canonical_source_bytes,
            "external_canonical_source_identity": self.external_canonical_source_identity,
            "roles": {role: sorted(paths) for role, paths in expected_map.items()},
        }

    def _validate_gc_targets(self, data: Mapping[str, Any], targets: Sequence[str]) -> list[dict[str, Any]]:
        if not targets:
            raise StorageBudgetError("reclamation target list is empty")
        records = []
        for raw in targets:
            _, relative = _relative(self.root, raw)
            item = data["artifacts"].get(relative)
            if not isinstance(item, Mapping):
                raise StorageIntegrityError(f"unknown artifact ownership: {relative}")
            if item.get("campaign_id") != self.campaign_id or not bool(item.get("reclaimable")):
                raise StorageIntegrityError(f"artifact is outside the reclamation whitelist: {relative}")
            if self.require_intents:
                if not str(item.get("intent_id", "")) or not isinstance(item.get("creation_record"), Mapping):
                    raise StorageIntegrityError(f"artifact lacks durable creation ownership: {relative}")
                if str(item.get("admission_hash", "")) != str(self.admission_hash or ""):
                    raise StorageIntegrityError(f"artifact admission identity is not bound: {relative}")
            path = self.root / relative
            current = _identity(path)
            recorded = item.get("identity")
            if not isinstance(recorded, Mapping) or dict(recorded) != current:
                raise StorageIntegrityError(f"artifact identity changed: {relative}")
            if int(current.get("st_nlink", 1)) != 1:
                raise StorageIntegrityError(f"hard-linked artifact is protected: {relative}")
            if _sha256_file(path) != item.get('sha256'):
                raise StorageIntegrityError(f'artifact content changed: {relative}')
            records.append({
                "relative_path": relative, "bytes": int(current["size"]),
                "identity": dict(current), "artifact_sha256": str(item.get("sha256", "")),
                "role": item.get("role", ""),
            })
        return records

    def _read_writer_receipt(
        self, receipt: str | Path | Mapping[str, Any] | None, *, expected_sha256: str | None = None,
        expected_epoch: str | int | None = None,
        expected_trajectory: str | None = None, expected_pulse: int | None = None,
        expected_attempt: int | None = None, expected_admission_hash: str | None = None,
        _ledger_data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load a durable quiescence receipt and bind its file identity.

        A mapping alone is intentionally insufficient: it can be a caller's
        in-memory assertion and has no durability or file identity.  Callers
        must provide the receipt path either directly or as ``receipt_path``
        in a small mapping.
        """
        if isinstance(receipt, Mapping):
            receipt_path_value = receipt.get("receipt_path") or receipt.get("path")
            if receipt_path_value is None:
                raise StorageBudgetError("writer receipt path is required")
            receipt_path = Path(str(receipt_path_value))
        elif receipt is not None:
            receipt_path = Path(receipt)
        else:
            raise StorageBudgetError("writer quiescence receipt is required")
        if not receipt_path.is_absolute():
            receipt_path = self.root.parent / receipt_path
        if ".." in receipt_path.parts or any(_is_reparse(part) for part in (receipt_path, *receipt_path.parents)):
            raise StorageIntegrityError("writer quiescence receipt path is linked or traverses a parent")
        receipt_path = receipt_path.resolve()
        if not receipt_path.is_file() or _is_reparse(receipt_path):
            raise StorageIntegrityError("writer quiescence receipt is missing or linked")
        digest = _sha256_file(receipt_path)
        if expected_sha256 is not None and digest != str(expected_sha256):
            raise StorageIntegrityError("writer quiescence receipt hash changed")
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageIntegrityError("writer quiescence receipt is not valid JSON") from error
        if not isinstance(payload, Mapping) or str(payload.get("status", "")).upper() != "PASS":
            raise StorageBudgetError("writer quiescence receipt is not PASS")
        if self.require_intents and (payload.get("schema") != WRITER_RECEIPT_SCHEMA
                                     or payload.get("registry_backed") is not True):
            raise StorageIntegrityError("formal writer receipt is not registry-backed")
        campaign = payload.get("campaign_id")
        if campaign is not None and str(campaign) != self.campaign_id:
            raise StorageIntegrityError("writer quiescence receipt campaign mismatch")
        active = payload.get("active_writers", payload.get("active_writer_ids", []))
        if active not in ([], (), None):
            raise StorageBudgetError("writer quiescence receipt still has active writers")
        epoch = payload.get("writer_epoch", payload.get("epoch"))
        if epoch is None or isinstance(epoch, bool):
            raise StorageBudgetError("writer quiescence receipt lacks writer epoch")
        if expected_epoch is not None and str(epoch) != str(expected_epoch):
            raise StorageIntegrityError("writer quiescence receipt epoch changed")
        process_id = payload.get("coordinator_process_id", payload.get("process_id"))
        if process_id is None or not str(process_id):
            raise StorageBudgetError("writer quiescence receipt lacks coordinator identity")
        bindings = {
            "trajectory": expected_trajectory,
            "pulse": expected_pulse,
            "attempt": expected_attempt,
            "admission_identity_sha256": expected_admission_hash,
        }
        for name, expected in bindings.items():
            if expected is None:
                continue
            actual = payload.get(name)
            if actual is None:
                raise StorageBudgetError(f"writer quiescence receipt lacks {name} binding")
            if str(actual) != str(expected):
                raise StorageIntegrityError(f"writer quiescence receipt {name} changed")
        if self.require_intents:
            if _ledger_data is None:
                active_now = self.active_writers(coordinator_epoch=epoch, include_management=True)
            else:
                active_now = [item for item in _ledger_data.get("writers", {}).values()
                              if isinstance(item, Mapping) and item.get("status") == "ACTIVE"
                              and str(item.get("coordinator_epoch")) == str(epoch)]
            if active_now:
                raise StorageBudgetError("writer registry became active after quiescence receipt")
        return {
            "path": str(receipt_path), "sha256": digest, "writer_epoch": str(epoch),
            "coordinator_process_id": str(process_id), "status": "PASS",
            **{name: payload.get(name) for name in bindings if payload.get(name) is not None},
        }

    def _read_campaign_json(
        self, value: str | Path | Mapping[str, Any], *, label: str,
        expected_sha256: str | None = None, require_hash: bool = True,
    ) -> tuple[Path, str, dict[str, Any], str]:
        """Read a durable JSON evidence file below this campaign root.

        Reclamation authorization is evidence-driven.  A caller-provided
        mapping is not sufficient because it can be changed after planning;
        every evidence file is resolved, hashed, and parsed again by the
        storage coordinator.
        """
        if isinstance(value, Mapping):
            raw_path = value.get("path") or value.get("receipt_path")
            expected = value.get("sha256", expected_sha256)
        else:
            raw_path = value
            expected = expected_sha256
        if raw_path is None:
            raise StorageBudgetError(f"{label} path is required")
        try:
            path, relative = _relative(self.root, Path(str(raw_path)))
        except StorageIntegrityError as error:
            raise StorageIntegrityError(f"{label} is outside campaign root") from error
        if not path.is_file() or _is_reparse(path):
            raise StorageIntegrityError(f"{label} is missing or linked")
        digest = _sha256_file(path)
        if require_hash and (expected is None or not str(expected)):
            raise StorageIntegrityError(f"{label} hash is missing or changed")
        if expected is not None and (not str(expected) or digest != str(expected)):
            raise StorageIntegrityError(f"{label} hash is missing or changed")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageIntegrityError(f"{label} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise StorageIntegrityError(f"{label} must contain a JSON object")
        return path, relative, payload, digest

    def _read_evidence_descriptor(
        self, descriptor: Mapping[str, Any], *, label: str,
    ) -> tuple[dict[str, Any], str]:
        if not isinstance(descriptor, Mapping):
            raise StorageBudgetError(f"{label} descriptor is invalid")
        path, relative, payload, digest = self._read_campaign_json(
            descriptor, label=label, expected_sha256=str(descriptor.get("sha256", "")),
        )
        return {
            "path": str(path), "relative_path": relative, "sha256": digest,
            "status": str(payload.get("status", "")).upper(),
            "payload": payload,
        }, digest

    def _validate_successor_evidence(
        self, info: Mapping[str, Any], *, label: str,
    ) -> None:
        """Validate a self-contained successor receipt before any deletion.

        The production Streaming READY schema is checked with its existing
        payload validator, which re-reads every child CURRENT array.  A small
        independent reference fixture may use the generic self-contained
        shape below; its field files and exact binding are still hash-bound.
        """
        payload = info["payload"]
        status = str(payload.get("status", "")).upper()
        if status not in {"READY", "PASS"}:
            raise StorageBudgetError(f"{label} is not READY")
        schema = str(payload.get("schema", ""))
        ready_path = Path(str(info["path"]))
        if schema == "khz_filament.hr4e5.e5_1a.ready.v1":
            try:
                from .hr4e5_evidence import validate_ready_receipt

                validate_ready_receipt(ready_path.parent)
            except Exception as error:
                raise StorageIntegrityError(f"{label} payload validation failed") from error
            return
        if payload.get("terminal") == "POST_FINAL_READY":
            lifecycle_root_value = payload.get("lifecycle_root")
            retained = payload.get("retained_optical_hashes")
            if lifecycle_root_value is None or not isinstance(retained, Mapping) or len(retained) != 3:
                raise StorageBudgetError(f"{label} terminal evidence is incomplete")
            try:
                lifecycle_root, _ = _relative(self.root, Path(str(lifecycle_root_value)))
                if lifecycle_root.resolve() != ready_path.parent.resolve():
                    raise StorageIntegrityError(f"{label} lifecycle root identity changed")
                optical_paths = []
                for raw_path, expected_hash in retained.items():
                    optical_path, _ = _relative(self.root, Path(str(raw_path)))
                    if not optical_path.is_file() or _is_reparse(optical_path) or _sha256_file(optical_path) != str(expected_hash):
                        raise StorageIntegrityError(f"{label} retained optical evidence changed")
                    optical_paths.append(optical_path)
                # The terminal receipt intentionally retains only the final
                # optical field, ledger and run metadata.  Sink arrays may be
                # one of the approved GC targets, so revalidation must not
                # require those already-reclaimed intermediates.
                from .hr4e5s_streaming import StreamingLifecycle

                lifecycle = StreamingLifecycle.open(lifecycle_root)
                if lifecycle._authoritative_namespace != "CURRENT" or lifecycle.manifest.get("queue") or lifecycle.manifest.get("recovery_backlog"):
                    raise StorageIntegrityError(f"{label} terminal lifecycle is not a quiet CURRENT root")
                if lifecycle.manifest.get("barrier") is not None or lifecycle.manifest.get("promotion") is not None:
                    raise StorageIntegrityError(f"{label} terminal lifecycle has an active transition")
                if str(payload.get("current_generation")) != str(lifecycle.manifest.get("current_generation")) or str(payload.get("current_content_sha256")) != str(lifecycle.manifest.get("current_content_sha256")):
                    raise StorageIntegrityError(f"{label} terminal lifecycle identity changed")
                for record in lifecycle.manifest.get("records", []):
                    if record.get("state") != "POST_COMMITTED" or record.get("post") is None or record.get("next") is not None:
                        raise StorageIntegrityError(f"{label} terminal POST inventory is incomplete")
                    lifecycle._validate_record_provenance(record, require_post=True, require_next=False)
                lifecycle._assert_no_staged_or_orphaned_artifacts()
            except StorageBudgetError:
                raise
            except Exception as error:
                if isinstance(error, StorageIntegrityError):
                    raise
                raise StorageIntegrityError(f"{label} terminal evidence validation failed") from error
            return
        if payload.get("self_contained") is not True or payload.get("parent_payload_required") is not False:
            raise StorageBudgetError(f"{label} is not a self-contained recovery point")
        child_root_value = payload.get("child_root") or payload.get("root")
        if child_root_value is None:
            raise StorageBudgetError(f"{label} child root is missing")
        try:
            child_root, _ = _relative(self.root, Path(str(child_root_value)))
        except StorageIntegrityError as error:
            raise StorageIntegrityError(f"{label} child root escapes campaign root") from error
        if not child_root.is_dir():
            raise StorageIntegrityError(f"{label} child root is missing")
        fields = payload.get("fields")
        if not isinstance(fields, Mapping) or not fields:
            raise StorageBudgetError(f"{label} self-contained field inventory is missing")
        for name, descriptor in fields.items():
            if not isinstance(descriptor, Mapping):
                raise StorageIntegrityError(f"{label} field descriptor is invalid: {name}")
            field_path_value = descriptor.get("path")
            field_hash = descriptor.get("sha256")
            if field_path_value is None or not str(field_hash):
                raise StorageIntegrityError(f"{label} field identity is incomplete: {name}")
            try:
                field_path, _ = _relative(self.root, child_root / str(field_path_value))
            except StorageIntegrityError as error:
                raise StorageIntegrityError(f"{label} field escapes child root: {name}") from error
            if not field_path.is_file() or _is_reparse(field_path) or _sha256_file(field_path) != str(field_hash):
                raise StorageIntegrityError(f"{label} field is missing or changed: {name}")
        binding = payload.get("binding") or payload.get("binding_receipt")
        if binding is None:
            raise StorageBudgetError(f"{label} exact binding is missing")
        binding_info, _ = self._read_evidence_descriptor(binding, label=f"{label} exact binding")
        binding_payload = binding_info["payload"]
        if str(binding_payload.get("status", "")).upper() != "PASS":
            raise StorageBudgetError(f"{label} exact binding is not PASS")
        rows = binding_payload.get("rows")
        if not isinstance(rows, list) or not rows or any(
            not isinstance(row, Mapping) or row.get("status") != "PASS" for row in rows
        ):
            raise StorageIntegrityError(f"{label} exact binding rows are incomplete")

    def _read_reclaim_prerequisite_receipt(
        self, receipt: str | Path | Mapping[str, Any] | None, *,
        expected_sha256: str | None = None,
        expected_gates: Mapping[str, bool] | None = None,
        expected_targets: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Read and validate durable exact/READY/dependency prerequisites."""
        if receipt is None:
            raise StorageBudgetError("durable reclamation prerequisite receipt is required")
        path, relative, payload, digest = self._read_campaign_json(
            receipt, label="reclamation prerequisite receipt", expected_sha256=expected_sha256,
            require_hash=expected_sha256 is not None,
        )
        if payload.get("schema") != RECLAIM_PREREQUISITE_SCHEMA:
            raise StorageBudgetError("reclamation prerequisite receipt schema is invalid")
        if str(payload.get("status", "")).upper() != "PASS":
            raise StorageBudgetError("reclamation prerequisite receipt is not PASS")
        if str(payload.get("campaign_id", "")) != self.campaign_id:
            raise StorageIntegrityError("reclamation prerequisite campaign mismatch")
        required = (
            "exact_complete", "successor_ready", "no_active_writers",
            "no_future_dependency", "receipts_durable",
        )
        gates = payload.get("gates")
        if not isinstance(gates, Mapping) or any(gates.get(name) is not True for name in required):
            raise StorageBudgetError("reclamation prerequisite gates are incomplete")
        if expected_gates is not None:
            for name in required:
                if bool(expected_gates.get(name)) is not bool(gates.get(name)):
                    raise StorageIntegrityError(f"reclamation prerequisite gate changed: {name}")

        bindings = payload.get("target_bindings")
        if not isinstance(bindings, list) or not bindings:
            raise StorageBudgetError("reclamation prerequisite target bindings are missing")
        binding_map: dict[str, Mapping[str, Any]] = {}
        for item in bindings:
            if not isinstance(item, Mapping) or not str(item.get("relative_path", "")):
                raise StorageIntegrityError("reclamation prerequisite target binding is invalid")
            key = str(item["relative_path"]).replace("\\", "/")
            if key in binding_map:
                raise StorageIntegrityError(f"duplicate reclamation target binding: {key}")
            binding_map[key] = item
        if expected_targets is not None:
            expected_map = {str(item["relative_path"]): item for item in expected_targets}
            if set(binding_map) != set(expected_map):
                raise StorageIntegrityError("reclamation prerequisite target set changed")
            for relative_name, expected in expected_map.items():
                item = binding_map[relative_name]
                if str(item.get("sha256", "")) != str(expected.get("artifact_sha256", expected.get("sha256", ""))):
                    raise StorageIntegrityError(f"reclamation prerequisite target hash changed: {relative_name}")
                if dict(item.get("identity", {})) != dict(expected.get("identity", {})):
                    raise StorageIntegrityError(f"reclamation prerequisite target identity changed: {relative_name}")
                if str(expected.get("status", "")) not in {"DELETED", "DELETE_INTENT"}:
                    current_path, checked = _relative(self.root, relative_name)
                    if checked != relative_name or not current_path.is_file():
                        raise StorageIntegrityError(f"reclamation prerequisite target is missing: {relative_name}")
                    if dict(_identity(current_path)) != dict(item.get("identity", {})) or _sha256_file(current_path) != str(item.get("sha256", "")):
                        raise StorageIntegrityError(f"reclamation prerequisite target content changed: {relative_name}")
                elif str(expected.get("status", "")) == "DELETE_INTENT":
                    current_path, checked = _relative(self.root, relative_name)
                    if current_path.exists():
                        if dict(_identity(current_path)) != dict(item.get("identity", {})) or _sha256_file(current_path) != str(item.get("sha256", "")):
                            raise StorageIntegrityError(f"reclamation prerequisite target content changed: {relative_name}")

        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise StorageBudgetError("reclamation prerequisite evidence is missing")
        evidence_info: dict[str, Any] = {}
        for key in ("exact_complete", "successor_ready", "no_future_dependency"):
            descriptor = evidence.get(key)
            info, _ = self._read_evidence_descriptor(descriptor, label=f"reclamation {key} evidence")
            evidence_info[key] = info
        exact_payload = evidence_info["exact_complete"]["payload"]
        if evidence_info["exact_complete"]["status"] != "PASS" or int(exact_payload.get("mismatch_count", -1)) != 0:
            raise StorageBudgetError("reclamation exact evidence is not a complete PASS")
        if exact_payload.get("missing_reference") not in (None, [], ()) or exact_payload.get("missing_candidate") not in (None, [], ()):
            raise StorageBudgetError("reclamation exact evidence has missing objects")
        exact_rows = exact_payload.get("rows")
        if not isinstance(exact_rows, list) or not exact_rows or any(
            not isinstance(row, Mapping) or row.get("status") != "PASS" for row in exact_rows
        ):
            raise StorageBudgetError("reclamation exact evidence rows are incomplete")
        if self.require_intents:
            binding = exact_payload.get("object_set_binding")
            if not isinstance(binding, Mapping) or str(binding.get("status", "")).upper() != "PASS":
                raise StorageBudgetError("formal reclamation requires a validated exact object-set binding")
            if str(binding.get("campaign_id", self.campaign_id)) != self.campaign_id:
                raise StorageIntegrityError("formal exact object-set campaign mismatch")
            if not str(binding.get("binding_sha256", "")):
                raise StorageIntegrityError("formal exact object-set binding hash is missing")
            try:
                from .hr4e5_evidence import validate_durable_report
                reclaimed = any(
                    str(target.get("status")) in {"DELETE_INTENT", "DELETED"}
                    for plan in self._read().get("gc_plans", {}).values()
                    if isinstance(plan, Mapping)
                    for target in plan.get("targets", [])
                    if isinstance(target, Mapping)
                )
                validate_durable_report(
                    evidence_info["exact_complete"]["path"],
                    expected_sha256=evidence_info["exact_complete"]["sha256"],
                    campaign_id=self.campaign_id, root=self.root,
                    validate_objects=not reclaimed,
                )
            except Exception as error:
                if isinstance(error, StorageBudgetError):
                    raise
                raise StorageIntegrityError("formal exact object-set validation failed") from error
        self._validate_successor_evidence(evidence_info["successor_ready"], label="reclamation successor READY evidence")
        dependency_payload = evidence_info["no_future_dependency"]["payload"]
        if evidence_info["no_future_dependency"]["status"] != "PASS" or dependency_payload.get("no_future_dependency") is not True:
            raise StorageBudgetError("reclamation future-dependency evidence is not PASS")
        return {
            "path": str(path), "relative_path": relative, "sha256": digest,
            "status": "PASS", "gates": {name: True for name in required},
            "target_bindings": [dict(item) for item in bindings],
            # Keep the immutable reports separate; duplicating all exact rows
            # into every per-file GC journal rewrite multiplies metadata I/O.
            "evidence": {key: {name: value for name, value in info.items() if name != 'payload'}
                         for key, info in evidence_info.items()},
        }

    def plan_reclaim(
        self, targets: Sequence[str | Path], *, exact_complete: bool,
        successor_ready: bool, no_active_writers: bool,
        no_future_dependency: bool, receipts_durable: bool,
        plan_id: str | None = None, reason: str = "after_exact_successor_ready",
        writer_receipt: str | Path | Mapping[str, Any] | None = None,
        writer_receipt_sha256: str | None = None,
        writer_epoch: str | int | None = None,
        prerequisite_receipt: str | Path | Mapping[str, Any] | None = None,
        trajectory: str | None = None, pulse: int | None = None,
        attempt: int | None = None, admission_hash: str | None = None,
    ) -> dict[str, Any]:
        gates = {
            "exact_complete": bool(exact_complete),
            "successor_ready": bool(successor_ready),
            "no_active_writers": bool(no_active_writers),
            "no_future_dependency": bool(no_future_dependency),
            "receipts_durable": bool(receipts_durable),
        }
        if not all(gates.values()):
            raise StorageBudgetError("reclamation prerequisites are not all satisfied")
        if self.require_intents:
            if str(trajectory) not in {"R", "C"}:
                raise StorageBudgetError("formal reclamation requires an R/C trajectory binding")
            if pulse is None or isinstance(pulse, bool) or int(pulse) < 0:
                raise StorageBudgetError("formal reclamation requires a nonnegative pulse binding")
            if attempt is None or isinstance(attempt, bool) or int(attempt) < 0:
                raise StorageBudgetError("formal reclamation requires a nonnegative attempt binding")
            if admission_hash is None or str(admission_hash) != str(self.admission_hash or ""):
                raise StorageIntegrityError("formal reclamation admission identity changed")
        writer_info = self._read_writer_receipt(
            writer_receipt, expected_sha256=writer_receipt_sha256,
            expected_epoch=writer_epoch,
            expected_trajectory=trajectory if self.require_intents else None,
            expected_pulse=pulse if self.require_intents else None,
            expected_attempt=attempt if self.require_intents else None,
            expected_admission_hash=admission_hash if self.require_intents else None,
        )
        pid = str(plan_id or f"gc-{os.getpid()}-{time.time_ns()}")
        raw_targets = [str(item) for item in targets]
        with self._locked(operation_name="StorageBudget.plan_reclaim") as data:
            if pid in data["gc_plans"]:
                raise StorageBudgetError(f"reclamation plan already exists: {pid}")
            records = self._validate_gc_targets(data, raw_targets)
            prerequisite_info = self._read_reclaim_prerequisite_receipt(
                prerequisite_receipt, expected_gates=gates, expected_targets=records,
            )
            protected_paths = {
                prerequisite_info["relative_path"],
                *(str(item["relative_path"]) for item in prerequisite_info["evidence"].values()),
            }
            for record in records:
                if str(record["relative_path"]) in protected_paths:
                    raise StorageIntegrityError(
                        f"reclamation target is also prerequisite evidence: {record['relative_path']}"
                    )
            try:
                _, writer_relative = _relative(self.root, writer_info["path"])
            except StorageIntegrityError:
                writer_relative = None
            if writer_relative is not None and any(
                str(record["relative_path"]) == writer_relative for record in records
            ):
                raise StorageIntegrityError("reclamation target is also writer evidence")
            plan = {
                "schema": GC_SCHEMA, "plan_id": pid, "campaign_id": self.campaign_id,
                "status": "PLANNED", "reason": str(reason), "gates": gates,
                "trajectory": None if trajectory is None else str(trajectory),
                "pulse": None if pulse is None else int(pulse),
                "attempt": None if attempt is None else int(attempt),
                "admission_identity_sha256": None if admission_hash is None else str(admission_hash),
                "writer_receipt": writer_info,
                "prerequisite_receipt": prerequisite_info,
                "targets": records, "planned_bytes": sum(item["bytes"] for item in records),
                "deleted_bytes": 0, "created_utc": _utc(), "updated_utc": _utc(),
            }
            data["gc_plans"][pid] = plan
            data["events"].append({"event": "GC_PLAN", "plan_id": pid, "planned_bytes": plan["planned_bytes"], "timestamp_utc": _utc()})
        return plan

    create_gc_plan = plan_reclaim

    def _load_plan(self, data: Mapping[str, Any], plan_id: str) -> dict[str, Any]:
        plan = data.get("gc_plans", {}).get(str(plan_id))
        if not isinstance(plan, dict) or plan.get("schema") != GC_SCHEMA or plan.get("campaign_id") != self.campaign_id:
            raise StorageBudgetError("reclamation plan is missing or invalid")
        return plan

    def reclaim_plan(self, plan_id: str) -> dict[str, Any] | None:
        """Return a snapshot of an existing transaction without creating one."""
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.reclaim_plan",
            campaign_root=self.root,
        ):
            data = self._read(); self._validate(data)
            plan = data.get("gc_plans", {}).get(str(plan_id))
            return None if plan is None else json.loads(json.dumps(plan))

    def reclaimed_target(self, relative_path: str | Path, *, campaign_id: str,
                         trajectory: str, pulse: int, attempt: int,
                         identity: Mapping[str, Any], sha256: str) -> bool:
        """Prove a missing exact object has this transaction's durable deletion intent."""
        path, relative = _relative(self.root, relative_path)
        if path.exists():
            return False
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.reclaimed_target",
            campaign_root=self.root,
        ):
            data = self._read(); self._validate(data)
            artifact = data["artifacts"].get(relative)
            if (not isinstance(artifact, Mapping)
                    or str(artifact.get("campaign_id")) != str(campaign_id)
                    or dict(artifact.get("identity", {})) != dict(identity)
                    or str(artifact.get("sha256", "")) != str(sha256)):
                return False
            for plan in data.get("gc_plans", {}).values():
                if (not isinstance(plan, Mapping) or plan.get("schema") != GC_SCHEMA
                        or str(plan.get("campaign_id")) != str(campaign_id)
                        or str(plan.get("trajectory")) != str(trajectory)
                        or int(plan.get("pulse", -1)) != int(pulse)
                        or int(plan.get("attempt", -1)) != int(attempt)
                        or plan.get("status") == "ABORTED"):
                    continue
                for target in plan.get("targets", []):
                    if (str(target.get("relative_path")) == relative
                            and target.get("status") in {"DELETE_INTENT", "DELETED"}
                            and dict(target.get("identity", {})) == dict(identity)
                            and str(target.get("artifact_sha256", "")) == str(sha256)):
                        return True
        return False

    def _validate_reclaim_step_binding(
        self, plan: Mapping[str, Any], *, expected_trajectory: str | None,
        expected_pulse: int | None, expected_attempt: int | None,
        expected_admission_hash: str | None,
    ) -> None:
        bindings = {
            "trajectory": expected_trajectory,
            "pulse": expected_pulse,
            "attempt": expected_attempt,
            "admission_identity_sha256": expected_admission_hash,
        }
        if self.require_intents and any(value is None for value in bindings.values()):
            raise StorageIntegrityError("formal reclamation requires complete step bindings before unlink")
        for name, expected in bindings.items():
            if expected is not None and str(plan.get(name)) != str(expected):
                raise StorageIntegrityError(f"reclamation plan {name} binding changed")

    def validate_reclaim_plan(
        self, plan_id: str, *, gates: Mapping[str, bool] | None = None,
        writer_receipt: str | Path | Mapping[str, Any] | None = None,
        writer_epoch: str | int | None = None,
        prerequisite_receipt: str | Path | Mapping[str, Any] | None = None,
        expected_trajectory: str | None = None, expected_pulse: int | None = None,
        expected_attempt: int | None = None, expected_admission_hash: str | None = None,
    ) -> dict[str, Any]:
        with self._locked(operation_name="StorageBudget.validate_reclaim_plan") as data:
            plan = self._load_plan(data, plan_id)
            self._validate_reclaim_step_binding(
                plan, expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
            )
            if plan.get("status") in {"COMPLETED", "ABORTED"}:
                return dict(plan)
            if gates is not None and not all(bool(value) for value in gates.values()):
                raise StorageBudgetError("current reclamation gates are not satisfied")
            stored_writer = plan.get("writer_receipt")
            if not isinstance(stored_writer, Mapping):
                raise StorageBudgetError("reclamation plan lacks durable writer receipt")
            self._read_writer_receipt(
                writer_receipt or stored_writer.get("path"),
                expected_sha256=str(stored_writer.get("sha256", "")),
                expected_epoch=writer_epoch or stored_writer.get("writer_epoch"),
                expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
                _ledger_data=data,
            )
            stored_prerequisite = plan.get("prerequisite_receipt")
            if not isinstance(stored_prerequisite, Mapping):
                raise StorageBudgetError("reclamation plan lacks durable prerequisite receipt")
            self._read_reclaim_prerequisite_receipt(
                prerequisite_receipt or stored_prerequisite.get("path"),
                expected_sha256=str(stored_prerequisite.get("sha256", "")),
                expected_gates=plan.get("gates"), expected_targets=plan.get("targets", []),
            )
            for target in plan.get("targets", []):
                relative = str(target["relative_path"])
                path = self.root / relative
                if target.get("status") == "DELETED":
                    continue
                current = _identity(path)
                if dict(current) != dict(target.get("identity", {})):
                    raise StorageIntegrityError(f"reclamation target identity changed: {relative}")
                if int(current.get("st_nlink", 1)) != 1:
                    raise StorageIntegrityError(f"hard-linked reclamation target: {relative}")
                if _sha256_file(path) != str(target.get("artifact_sha256", "")):
                    raise StorageIntegrityError(f"reclamation target content changed: {relative}")
            return dict(plan)

    def apply_reclaim(
        self, plan_id: str, *, no_active_writers: bool = True,
        interrupt_after: int | None = None,
        writer_receipt: str | Path | Mapping[str, Any] | None = None,
        writer_epoch: str | int | None = None,
        prerequisite_receipt: str | Path | Mapping[str, Any] | None = None,
        fault_hook: Callable[[str, str], None] | None = None,
        expected_trajectory: str | None = None, expected_pulse: int | None = None,
        expected_attempt: int | None = None, expected_admission_hash: str | None = None,
    ) -> dict[str, Any]:
        if not bool(no_active_writers):
            raise StorageBudgetError("reclamation requires a quiescent coordinator")
        count = 0
        with self._locked(operation_name="StorageBudget.apply_reclaim") as data:
            plan = self._load_plan(data, plan_id)
            self._validate_reclaim_step_binding(
                plan, expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
            )
            if plan.get("status") == "COMPLETED":
                return dict(plan)
            if plan.get("status") == "ABORTED":
                raise StorageBudgetError("aborted reclamation plan cannot be resumed")
            stored_writer = plan.get("writer_receipt")
            if not isinstance(stored_writer, Mapping):
                raise StorageBudgetError("reclamation plan lacks durable writer receipt")
            self._read_writer_receipt(
                writer_receipt or stored_writer.get("path"),
                expected_sha256=str(stored_writer.get("sha256", "")),
                expected_epoch=writer_epoch or stored_writer.get("writer_epoch"),
                expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
                _ledger_data=data,
            )
            stored_prerequisite = plan.get("prerequisite_receipt")
            if not isinstance(stored_prerequisite, Mapping):
                raise StorageBudgetError("reclamation plan lacks durable prerequisite receipt")
            self._read_reclaim_prerequisite_receipt(
                prerequisite_receipt or stored_prerequisite.get("path"),
                expected_sha256=str(stored_prerequisite.get("sha256", "")),
                expected_gates=plan.get("gates"), expected_targets=plan.get("targets", []),
            )
            plan["status"] = "APPLYING"
            plan["updated_utc"] = _utc()
            # Persist the transaction state before any unlink.  A resumed
            # process can distinguish an intended deletion from an unknown
            # disappearance and can conservatively recover a post-unlink
            # interruption.
            self._write(data)
            for target in plan["targets"]:
                if target.get("status") == "DELETED":
                    continue
                if interrupt_after is not None and count >= int(interrupt_after):
                    plan["status"] = "INTERRUPTED"
                    plan["updated_utc"] = _utc()
                    data["events"].append({"event": "GC_INTERRUPTED", "plan_id": str(plan_id), "timestamp_utc": _utc()})
                    self._write(data)
                    return dict(plan)
                relative = str(target["relative_path"])
                path, checked_relative = _relative(self.root, relative)
                if checked_relative != relative:
                    raise StorageIntegrityError("reclamation path changed")
                if target.get("status") == "DELETE_INTENT" and not path.exists():
                    # The durable intent is the only evidence under which a
                    # missing file may be attributed to this transaction.
                    target["status"] = "DELETED"
                    target["deleted_utc"] = _utc()
                    target["recovered_after_interrupt"] = True
                    plan["deleted_bytes"] = int(plan.get("deleted_bytes", 0)) + int(target["bytes"])
                    count += 1
                    self._write(data)
                    continue
                current = _identity(path)
                if dict(current) != dict(target.get("identity", {})) or int(current.get("st_nlink", 1)) != 1:
                    raise StorageIntegrityError(f"reclamation identity or hardlink gate failed: {relative}")
                if _sha256_file(path) != data['artifacts'][relative].get('sha256'):
                    raise StorageIntegrityError(f'reclamation content changed: {relative}')
                target["status"] = "DELETE_INTENT"
                target["intent_utc"] = _utc()
                self._write(data)
                path.unlink()
                _fsync_directory(path.parent)
                if fault_hook is not None:
                    fault_hook('after_unlink', relative)
                target["status"] = "DELETED"
                target["deleted_utc"] = _utc()
                plan["deleted_bytes"] = int(plan.get("deleted_bytes", 0)) + int(target["bytes"])
                count += 1
                self._write(data)
            plan["status"] = "COMPLETED"
            plan["updated_utc"] = _utc()
            data["events"].append({"event": "GC_COMPLETED", "plan_id": str(plan_id), "deleted_bytes": plan["deleted_bytes"], "timestamp_utc": _utc()})
            self._write(data)
            return dict(plan)

    def resume_reclaim(
        self, plan_id: str, *, no_active_writers: bool = True,
        writer_receipt: str | Path | Mapping[str, Any] | None = None,
        writer_epoch: str | int | None = None,
        prerequisite_receipt: str | Path | Mapping[str, Any] | None = None,
        expected_trajectory: str | None = None, expected_pulse: int | None = None,
        expected_attempt: int | None = None, expected_admission_hash: str | None = None,
    ) -> dict[str, Any]:
        return self.apply_reclaim(
            plan_id, no_active_writers=no_active_writers,
            writer_receipt=writer_receipt, writer_epoch=writer_epoch,
            prerequisite_receipt=prerequisite_receipt,
            expected_trajectory=expected_trajectory, expected_pulse=expected_pulse,
            expected_attempt=expected_attempt,
            expected_admission_hash=expected_admission_hash,
        )

    def verify_reclaim(
        self, plan_id: str, *, expected_trajectory: str | None = None,
        expected_pulse: int | None = None, expected_attempt: int | None = None,
        expected_admission_hash: str | None = None,
    ) -> dict[str, Any]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.verify_reclaim",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            plan = self._load_plan(data, plan_id)
            self._validate_reclaim_step_binding(
                plan, expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
            )
            stored_writer = plan.get("writer_receipt")
            if not isinstance(stored_writer, Mapping):
                raise StorageIntegrityError("reclamation plan lacks durable writer receipt")
            self._read_writer_receipt(
                stored_writer.get("path"), expected_sha256=str(stored_writer.get("sha256", "")),
                expected_epoch=stored_writer.get("writer_epoch"),
                expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
                _ledger_data=data,
            )
            stored_prerequisite = plan.get("prerequisite_receipt")
            if not isinstance(stored_prerequisite, Mapping):
                raise StorageIntegrityError("reclamation plan lacks durable prerequisite receipt")
            self._read_reclaim_prerequisite_receipt(
                stored_prerequisite.get("path"),
                expected_sha256=str(stored_prerequisite.get("sha256", "")),
                expected_gates=plan.get("gates"), expected_targets=plan.get("targets", []),
            )
            missing = []
            for target in plan["targets"]:
                if target.get("status") == "DELETED":
                    if (self.root / str(target["relative_path"])).exists():
                        missing.append(str(target["relative_path"]) + ":still_present")
                elif not (self.root / str(target["relative_path"])).is_file():
                    missing.append(str(target["relative_path"]) + ":missing")
            status = "PASS" if not missing and plan.get("status") == "COMPLETED" else "FAIL"
            return {"schema": GC_SCHEMA, "plan_id": str(plan_id), "status": status, "missing": missing, "deleted_bytes": int(plan.get("deleted_bytes", 0))}

    def budget_report(self) -> dict[str, Any]:
        with _FileLock(
            self.lock_path, timeout_s=self.lock_timeout_s, operation_name="StorageBudget.budget_report",
            campaign_root=self.root,
        ):
            data = self._read()
            self._validate(data)
            used, reserved, _ = self._capacity_accounting(data)
            if (self.root / "E5_1A_TERMINAL_ROLE_CONTRACT.json").is_file():
                final_bytes = (used + self.external_prefix_bytes
                               + self.external_canonical_source_bytes)
            else:
                final_bytes = (self.external_prefix_bytes + self.external_canonical_source_bytes
                               + self.frozen_lut_input_bytes + sum(
                    int((self.root / relative).stat().st_size)
                    for relative, item in data.get("artifacts", {}).items()
                    if ((str(item.get("role", "")).startswith("final")
                         or str(relative).endswith("/final_optical_field.npy"))
                        and (self.root / relative).is_file())))
            return {
                "schema": STORAGE_SCHEMA,
                "campaign_id": self.campaign_id,
                "actual_bytes": used,
                "external_prefix_bytes": self.external_prefix_bytes,
                **({"frozen_lut_input_bytes": self.frozen_lut_input_bytes}
                   if self.frozen_lut_input_bytes else {}),
                "external_canonical_source_bytes": self.external_canonical_source_bytes,
                "external_canonical_source_identity": self.external_canonical_source_identity,
                "external_full_source_bytes": self.external_full_source_bytes,
                "reserved_bytes": reserved,
                "safety_margin_bytes": self.safety_margin_bytes,
                "projected_bytes": (used + self.external_prefix_bytes
                                    + self.external_canonical_source_bytes + reserved
                                    + self.safety_margin_bytes),
                "cap_bytes": self.cap_bytes,
                "final_output_bytes": final_bytes,
                "planned_final_output_bytes": self.planned_final_output_bytes,
                "final_output_budget_bytes": self.final_output_budget_bytes,
                "free_bytes": self._provider_free_bytes(),
                "quota_bytes": self._provider_quota_bytes(),
                "quota_required": self.require_quota,
                "require_intents": self.require_intents,
                "admission_hash": self.admission_hash,
                "intent_count": len(data.get("intents", {})),
            }


class MockQuotaProvider:
    """Deterministic provider used by low-cost fixture tests."""

    def __init__(self, *, free_bytes: int | None = None, quota_bytes: int | None = None):
        self._free_bytes = free_bytes
        self._quota_bytes = quota_bytes

    def free_bytes(self) -> int | None:
        return self._free_bytes

    def quota_bytes(self) -> int | None:
        return self._quota_bytes

    def set_free(self, value: int | None) -> None:
        self._free_bytes = value

    def set_quota(self, value: int | None) -> None:
        self._quota_bytes = value


__all__ = [
    "DEFAULT_FINAL_OUTPUT_BUDGET_BYTES", "DEFAULT_SAFETY_MARGIN_BYTES", "GC_SCHEMA",
    "HARD_CAP_BYTES", "INTENT_SCHEMA", "MockQuotaProvider", "RECLAIM_PREREQUISITE_SCHEMA", "Reservation", "STORAGE_SCHEMA",
    "derive_budget_plan", "plan_campaign_budget",
    "StorageBudget", "StorageBudgetError", "StorageIntegrityError",
    "PROCESS_RECEIPT_SCHEMA", "WRITER_SCHEMA", "WRITER_RECEIPT_SCHEMA", "process_identity", "probe_process_identity",
]

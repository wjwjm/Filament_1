"""Independent parent-side acceptance of a completed S4 case's outputs."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import time

from tools.cpu_o1.cpu_s4.storage_budget import MAX_PROFILE_RECEIPT_BYTES


SCHEMA = "filament.cpu_s4.output_inventory.v1"
CHUNK_BYTES = 64 * 1024**2
SINK_FILES = (
    "sinks.hr3a_qthermal_samples.npy",
    "sinks.hr3a_qion_samples.npy",
    "sinks.hr3a_qib_samples.npy",
    "sinks.hr3a_qraman_samples.npy",
    "sinks.hr3b_delta_n_increment_samples.npy",
    "sinks.hr3b_delta_n_state_after_update_samples.npy",
)
NAN_PLACEHOLDER_FIELDS = frozenset({
    "energy_step_start_J", "energy_after_linear_half1_J",
    "energy_after_raman_pre_J", "energy_after_nonraman_J",
    "energy_after_raman_post_J", "energy_after_linear_half2_J",
})
CONSTANT_DIAGNOSTIC_SHAPES = {
    "deposition_q_shape": (2,),
    "thermal_map_archive_shape": (3,),
    "hr3b_state_shape": (3,),
    "hr3b_map_archive_shape": (3,),
}


SCALAR_DTYPES = {'E_dep_accounted_authoritative_J': 'float64', 'E_dep_ib_pulse_J': 'float64', 'E_dep_ion_pulse_J': 'float64', 'E_dep_plasma_pulse_J': 'float64', 'E_dep_raman_operator_energy_pulse_residual_J': 'float64', 'E_dep_raman_operator_pulse_J': 'float64', 'E_dep_raman_pulse_J': 'float64', 'E_dep_raman_pulse_closure_residual_J': 'float64', 'E_dep_total_pulse_J': 'float64', 'E_field_energy_bookkeeping_relative_residual': 'float64', 'E_field_energy_bookkeeping_residual_J': 'float64', 'E_field_in_J': 'float64', 'E_field_loss_J': 'float64', 'E_field_out_J': 'float64', 'E_th_ib_pulse_J': 'float64', 'E_th_ion_pulse_J': 'float64', 'E_th_raman_pulse_J': 'float64', 'E_thermal_pulse_J': 'float64', 'authoritative_hr3a_thermal_source_available': 'bool', 'authoritative_hr_slow_state_update_active': 'bool', 'bk_nee_profile_enabled': 'bool', 'deposition_channel_count': 'int64', 'deposition_closure_relative_tolerance': 'float64', 'deposition_ib_active': 'bool', 'deposition_ib_authoritative': 'bool', 'deposition_ib_configured': 'bool', 'deposition_ion_active': 'bool', 'deposition_ion_authoritative': 'bool', 'deposition_ion_configured': 'bool', 'deposition_ion_feedback_applied': 'bool', 'deposition_level1_all_available_mechanism_closure_pass': 'bool', 'deposition_level2_all_available_mechanism_closure_pass': 'bool', 'deposition_n_intervals': 'int64', 'deposition_payload_allocated': 'bool', 'deposition_raman_active': 'bool', 'deposition_raman_authoritative': 'bool', 'deposition_raman_configured': 'bool', 'deposition_raman_feedback_applied': 'bool', 'diagnostic_validation_passed': 'bool', 'diagnostic_validation_trace_count': 'int64', 'f_R_ignored_rot_sinexp': 'bool', 'f_R_used_historical_fr_mixture': 'float64', 'field_energy_bookkeeping_authoritative': 'bool', 'historical_raman_Gamma_R_1_s': 'float64', 'historical_raman_omega_R_rad_s': 'float64', 'hr3b_authoritative': 'bool', 'hr3b_beta_th_m3_J': 'float64', 'hr3b_first_failed_interval': 'int64', 'hr3b_map_archive_complete': 'bool', 'hr3b_map_archive_enabled': 'bool', 'hr3b_state_disk_backed': 'bool', 'hr3b_state_interval_centered': 'bool', 'ionization_loss_on': 'bool', 'ionization_operator_correction_on': 'bool', 'ionization_solver_on': 'bool', 'legacy_slow_heat_compatibility_path_active': 'bool', 'longitudinal_z_end': 'float64', 'longitudinal_z_start': 'float64', 'n2_elec_used': 'float64', 'n_R_used': 'float64', 'n_intervals': 'int64', 'nonlinear_use_electronic_kerr': 'bool', 'nonlinear_use_ionization_loss': 'bool', 'nonlinear_use_ionization_solver': 'bool', 'nonlinear_use_plasma_phase': 'bool', 'nonlinear_use_raman_absorption': 'bool', 'nonlinear_use_raman_full_operator': 'bool', 'nonlinear_use_raman_phase': 'bool', 'nonlinear_use_self_steepening': 'bool', 'operator_energy_diagnostics_enabled': 'bool', 'optical_grid_Lx': 'float64', 'optical_grid_Ly': 'float64', 'optical_grid_Nx': 'int64', 'optical_grid_Ny': 'int64', 'optical_grid_dx': 'float64', 'optical_grid_dy': 'float64', 'performance_measurement_enabled': 'bool', 'raman_absorption_calculated': 'bool', 'raman_absorption_on': 'bool', 'raman_deposition_authoritative': 'bool', 'raman_operator_energy_cumulative_relative': 'float64', 'raman_operator_energy_cumulative_tolerance': 'float64', 'raman_operator_energy_step_p99': 'float64', 'raman_operator_energy_step_p99_tolerance': 'float64', 'raman_operator_feedback_enabled': 'bool', 'thermal_grid_Lx': 'float64', 'thermal_grid_Ly': 'float64', 'thermal_grid_Nx': 'int64', 'thermal_grid_Ny': 'int64', 'thermal_grid_dx': 'float64', 'thermal_grid_dy': 'float64', 'thermal_grid_matches_optical': 'bool', 'thermal_map_archive_complete': 'bool', 'thermal_map_archive_enabled': 'bool', 'thermalization_authoritative': 'bool', 'thermalization_complete_microscopic_approximation': 'bool', 'thermalization_first_failed_interval': 'int64', 'thermalization_max_abs_T2_residual_ib_J': 'float64', 'thermalization_max_abs_T2_residual_ion_J': 'float64', 'thermalization_max_abs_T2_residual_raman_J': 'float64', 'thermalization_max_abs_T3_residual_J': 'float64', 'total_deposition_authoritative': 'bool'}


class OutputInventoryError(ValueError):
    """The fixed S4 output contract was not independently satisfied."""


def _finite_json_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError('JSON floating-point value out of range')
    return value


def _strict_json(path: Path):
    def reject(token):
        raise ValueError(f"nonstandard JSON numeric token: {token}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject, parse_float=_finite_json_float)
    if not isinstance(value, dict):
        raise ValueError("JSON receipt must be an object")
    return value


def _strict_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"truncated JSONL line {number}")
            if not line.strip():
                raise ValueError(f"blank JSONL line {number}")
            value = json.loads(line.decode("utf-8"), parse_float=_finite_json_float, parse_constant=lambda token: (
                (_ for _ in ()).throw(ValueError(f"nonstandard JSON numeric token: {token}"))))
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("empty JSONL")
    return rows


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("required output is not a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _hash_stable(path: Path, raw_sha256_file) -> dict:
    before = _stat_signature(path)
    raw_before = raw_sha256_file(path)
    raw_after = raw_sha256_file(path)
    after = _stat_signature(path)
    if before != after or raw_before != raw_after:
        raise ValueError("output changed during independent acceptance")
    return {"path": path.as_posix(), "bytes": after[2], "raw_sha256": raw_after,
            "_signature": after, "stable_during_acceptance": True}


def _finite_bounded(array, *, np, label: str, permit_all_nan: bool = False) -> None:
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise ValueError(f"{label} has unsupported dtype {array.dtype}")
    flat = array.reshape(-1)
    step = max(1, CHUNK_BYTES // max(1, array.dtype.itemsize))
    all_nan = permit_all_nan and array.dtype.kind == "f" and flat.size > 0
    if all_nan:
        for start in range(0, flat.size, step):
            if not bool(np.isnan(flat[start:start + step]).all()):
                all_nan = False
                break
    if all_nan:
        return
    for start in range(0, flat.size, step):
        if not bool(np.isfinite(flat[start:start + step]).all()):
            raise ValueError(f"{label} contains nonfinite values")


def _npy(path: Path, *, np, raw_sha256_file, canonical_array_sha256,
         expected_shape: tuple[int, ...], expected_dtype: str,
         permit_all_nan: bool = False) -> dict:
    before = _stat_signature(path)
    entry = _hash_stable(path, raw_sha256_file)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError("NPY did not load as an array")
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"shape mismatch: expected {expected_shape}, found {array.shape}")
    if str(array.dtype) != expected_dtype:
        raise ValueError(f"dtype mismatch: expected {expected_dtype}, found {array.dtype}")
    _finite_bounded(array, np=np, label=path.name, permit_all_nan=permit_all_nan)
    canonical = canonical_array_sha256(array)
    if not isinstance(canonical, str) or len(canonical) != 64:
        raise ValueError("canonical array digest unavailable")
    entry.update(shape=list(array.shape), dtype=str(array.dtype),
                 canonical_array_sha256=canonical, bounded_array_scan_bytes=CHUNK_BYTES)
    # Re-hash after mapping and scanning to detect replacement or mutation.
    after = _stat_signature(path)
    if before != after or raw_sha256_file(path) != entry["raw_sha256"]:
        raise ValueError("NPY changed during validation")
    entry.pop("_signature", None)
    return entry


def _check_diag_shape(name: str, shape: tuple[int, ...], *, intervals: int,
                      ny: int, nx: int, nt: int) -> None:
    if name in {"z_edges", "thermalization_z_edges"}:
        expected = (intervals + 1,)
    elif name in CONSTANT_DIAGNOSTIC_SHAPES:
        expected = CONSTANT_DIAGNOSTIC_SHAPES[name]
    elif name == "Qacc_raman":
        expected = (ny, nx)
    elif name == "rho_onaxis_t_z":
        expected = (intervals, nt)
    else:
        expected = (intervals,)
    if shape != expected:
        raise ValueError(f"diagnostic {name} shape mismatch: expected {expected}, found {shape}")


def _npz(path: Path, *, np, raw_sha256_file, canonical_array_sha256,
         expected_hashes: dict[str, str], role: str, intervals: int,
         ny: int, nx: int, nt: int) -> tuple[dict, dict[str, dict]]:
    from zipfile import ZipFile, BadZipFile
    before = _stat_signature(path)
    entry = _hash_stable(path, raw_sha256_file)
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(not name.endswith(".npy") for name in names):
                raise ValueError("NPZ has duplicate or non-NPY members")
        loaded = np.load(path, allow_pickle=False)
        try:
            members = list(loaded.files)
            if len(members) != len(set(members)) or set(members) != set(expected_hashes):
                raise ValueError("NPZ member set differs from frozen receipt")
            details = {}
            for name in members:
                array = loaded[name]
                shape = tuple(int(x) for x in array.shape)
                if role == "POST":
                    expected_shape = (intervals, ny, nx)
                    if shape != expected_shape or str(array.dtype) != "float64":
                        raise ValueError(f"POST {name} shape/dtype mismatch")
                else:
                    _check_diag_shape(name, shape, intervals=intervals, ny=ny, nx=nx, nt=nt)
                    if array.dtype.kind not in "biufc" or array.dtype.hasobject:
                        raise ValueError(f"diagnostic {name} has unsupported dtype")
                permit_nan = role == "DIAGNOSTICS" and name in NAN_PLACEHOLDER_FIELDS
                _finite_bounded(array, np=np, label=f"{role}.{name}", permit_all_nan=permit_nan)
                digest = canonical_array_sha256(array)
                if digest != expected_hashes[name]:
                    raise ValueError(f"NPZ canonical digest mismatch: {name}")
                details[name] = {"shape": list(shape), "dtype": str(array.dtype),
                                 "canonical_array_sha256": digest}
        finally:
            loaded.close()
    except (OSError, ValueError, BadZipFile, EOFError, KeyError) as exc:
        raise ValueError(f"invalid {role} NPZ: {type(exc).__name__}: {exc}") from exc
    after = _stat_signature(path)
    if before != after or raw_sha256_file(path) != entry["raw_sha256"]:
        raise ValueError("NPZ changed during validation")
    entry.pop("_signature", None)
    entry["members"] = details
    return entry, details


def verify_output_inventory(case_root: str | os.PathLike, mode: str, *,
                            raw_sha256_file, canonical_array_sha256) -> dict:
    """Validate fixed outputs after child exit; return an explicit parent result.

    Both callback hashes are supplied by the caller from the repository's
    authoritative provenance and frozen scientific array-hash implementations.
    """
    started = time.perf_counter()
    root = Path(case_root)
    work = root / "work"
    propagation = work / "propagation"
    result = {"schema": SCHEMA, "status": "BLOCKED_OUTPUT_INVENTORY",
              "mode": mode, "started_monotonic": started, "entries": [],
              "failures": [], "hash_seconds": 0.0, "validation_seconds": 0.0,
              "inventory_seconds": 0.0}
    if mode not in {"OFF", "PROFILE"}:
        result["failures"].append("unsupported_mode")
    required = [
        root / "supervisor_receipt.json",
        work / "child_receipt.json",
        work / "window_receipt.json",
        work / "input_output_contract.json",
        work / "OUTPUT_SEAL.json",
        work / "scientific_Qacc.npy",
        work / "scientific_scalar_diagnostics.json",
        work / "candidate_events.jsonl",
        propagation / "child_receipt.json",
        propagation / "final_E.npy",
        propagation / "final_POST.npz",
        propagation / "diagnostics.npz",
        propagation / "observer_progress.jsonl",
        propagation / "linear_halfstep_profiles.json",
        *(propagation / name for name in SINK_FILES),
    ]
    required.append(propagation / 'function_fft_profile.json')
    hash_seconds = 0.0
    hash_calls = 0
    def measured_raw_hash(path):
        nonlocal hash_seconds, hash_calls
        hash_calls += 1
        hash_started = time.perf_counter()
        value = raw_sha256_file(path)
        hash_seconds += time.perf_counter() - hash_started
        return value
    canonical_seconds = 0.0
    canonical_calls = 0
    original_canonical_hash = canonical_array_sha256
    def measured_canonical_hash(array):
        nonlocal canonical_seconds, canonical_calls
        begin=time.perf_counter()
        try:
            return original_canonical_hash(array)
        finally:
            canonical_calls += 1
            canonical_seconds += time.perf_counter()-begin
    canonical_array_sha256=measured_canonical_hash
    parsed = {}
    for path in required:
        entry = {"path": path.relative_to(root).as_posix(), "status": "BLOCKED"}
        try:
            validate_started = time.perf_counter()
            signature = _stat_signature(path)
            if (mode == "PROFILE" and path == propagation / "function_fft_profile.json"
                    and signature[2] > MAX_PROFILE_RECEIPT_BYTES):
                raise OutputInventoryError("PROFILE_receipt_exceeds_512_MiB_cap")
            entry.update(_hash_stable(path, measured_raw_hash))
            entry["status"] = "PASS"
            result["entries"].append(entry)
            parsed[path.relative_to(root).as_posix()] = path
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            result["entries"].append(entry)
            result["failures"].append(f"{entry['path']}:{entry['error']}")
    result["hash_seconds"] = hash_seconds
    try:
        import numpy as np
        rel = lambda path: path.relative_to(root).as_posix()
        def required_json(path):
            if rel(path) not in parsed:
                raise OutputInventoryError(f"required_file_missing_or_unreadable:{rel(path)}")
            try:
                return _strict_json(path)
            except Exception as exc:
                raise OutputInventoryError(f"invalid_json:{rel(path)}:{exc}") from exc
        supervisor = required_json(root / "supervisor_receipt.json")
        child = required_json(work / "child_receipt.json")
        window = required_json(work / "window_receipt.json")
        contract = required_json(work / "input_output_contract.json")
        frozen = required_json(propagation / "child_receipt.json")
        scalars = required_json(work / "scientific_scalar_diagnostics.json")
        halfsteps = required_json(propagation / "linear_halfstep_profiles.json")
        if supervisor.get("status") != "COMPLETE" or supervisor.get("mode") != mode:
            raise OutputInventoryError("supervisor_receipt_status_or_mode_mismatch")
        if child.get("status") != "COMPLETE" or child.get("mode") != mode or child.get("implementation") != "I":
            raise OutputInventoryError("child_receipt_identity_or_mode_mismatch")
        if child.get("intervals") != list(range(8)) or child.get("candidate_event_count", 0) <= 0:
            raise OutputInventoryError("child_receipt_interval_or_candidate_contract_failed")
        if mode == "PROFILE" and child.get("profile_intervals") != list(range(8)):
            raise OutputInventoryError("PROFILE_interval_coverage_incomplete")
        if frozen.get("schema") != "filament.cpu_p1r2.j1_diagnostic.v1" or frozen.get("status") != "COMPLETE_DIAGNOSTIC":
            raise OutputInventoryError("frozen_window_receipt_schema_or_status_mismatch")
        if frozen.get("stage") != f"S4_I_{mode}" or frozen.get("interval_count") != 8:
            raise OutputInventoryError("frozen_window_receipt_stage_or_interval_mismatch")
        if (frozen.get("runtime_instrumentation") or {}).get("additional_profile_enabled") is not False:
            raise OutputInventoryError("frozen_window_profile_mode_mismatch")
        if window.get("status") != "COMPLETE" or window.get("mode") != mode:
            raise OutputInventoryError("window_receipt_status_or_mode_mismatch")
        interval_rows = window.get("interval_rows")
        if not isinstance(interval_rows, list) or [row.get("interval_index") for row in interval_rows] != list(range(8)):
            raise OutputInventoryError("window_receipt_interval_coverage_mismatch")
        if window.get("candidate_event_count") != child.get("candidate_event_count"):
            raise OutputInventoryError("candidate_event_count_mismatch")
        if contract.get("schema") != "filament.cpu_s4.input_output_contract.v1":
            raise OutputInventoryError("input_output_contract_schema_mismatch")
        if contract.get("implementation") != "I" or contract.get("mode") != mode or contract.get("interval_count") != 8:
            raise OutputInventoryError("input_output_contract_identity_mismatch")
        source_shape = contract.get("source_shape")
        post_shape = contract.get("post_shape")
        qacc_shape = contract.get("qacc_shape")
        if (not isinstance(source_shape, list) or len(source_shape) != 3
                or any(not isinstance(x, int) or x <= 0 for x in source_shape)
                or contract.get("source_dtype") != "complex128"):
            raise OutputInventoryError("input_output_contract_source_shape_or_dtype_invalid")
        nt, ny, nx = source_shape
        if post_shape != [8, ny, nx] or qacc_shape != [ny, nx] or contract.get("qacc_dtype") != "float64":
            raise OutputInventoryError("input_output_contract_output_shape_or_dtype_invalid")
        pre_shape = (frozen.get("PRE_metadata") or {}).get("shape")
        if pre_shape != post_shape:
            raise OutputInventoryError("frozen_PRE_shape_differs_from_independent_contract")
        if (child.get("source_hash") != frozen.get("source_canonical_sha256")
                or child.get("pre_hash") != frozen.get("input_PRE_fields_canonical_sha256")):
            raise OutputInventoryError("child_input_identity_differs_from_frozen_receipt")
        if scalars.get("schema") != "filament.cpu_q1.propagation_scalar_diagnostics.v1" or scalars.get("identity") != "I":
            raise OutputInventoryError("scientific_scalar_identity_mismatch")
        seal = required_json(work / 'OUTPUT_SEAL.json')
        if seal.get('schema') != 'filament.cpu_s4.output_seal.v1':
            raise OutputInventoryError('output_seal_schema')
        sealed_paths = [p for p in required if p not in (root/'supervisor_receipt.json', work/'child_receipt.json',work/'OUTPUT_SEAL.json')]
        if set(seal.get('files', {})) != {p.relative_to(work).as_posix() for p in sealed_paths}:
            raise OutputInventoryError('output_seal_members')
        if measured_raw_hash(work/'OUTPUT_SEAL.json') != child.get('output_seal_sha256'):
            raise OutputInventoryError('output_seal_binding')
        for p in sealed_paths:
            if measured_raw_hash(p) != seal['files'][p.relative_to(work).as_posix()]:
                raise OutputInventoryError('output_seal_hash:'+p.name)
        scalar_fields = scalars.get("fields")
        if not isinstance(scalar_fields, dict) or not scalar_fields:
            raise OutputInventoryError("scientific_scalar_fields_missing")
        if {k:v.get("dtype") for k,v in scalar_fields.items()} != SCALAR_DTYPES:
            raise OutputInventoryError("scientific_scalar_field_or_dtype_contract")
        for name, value in scalar_fields.items():
            if (not isinstance(name, str) or not isinstance(value, dict)
                    or value.get("shape") != [] or value.get("dtype") not in
                    {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"}
                    or "value" not in value):
                raise OutputInventoryError(f"invalid_scientific_scalar:{name}")
            encoded = value["value"]
            dtype = value["dtype"]
            if dtype == "bool":
                if type(encoded) is not bool:
                    raise OutputInventoryError(f"invalid_scalar_bool:{name}")
            elif dtype == "int64":
                if (not isinstance(encoded, str) or not encoded.lstrip("-").isascii() or not encoded.lstrip("-").isdigit()
                        or str(int(encoded)) != encoded or not -(2**63) <= int(encoded) < 2**63):
                    raise OutputInventoryError(f"invalid_scalar_int:{name}")
            else:
                if not isinstance(encoded, str) or not math.isfinite(float.fromhex(encoded)):
                    raise OutputInventoryError(f"invalid_scalar_float:{name}")
        expected_source = frozen.get("source_canonical_sha256")
        if child.get("source_hash") != expected_source:
            raise OutputInventoryError("source_hash_mismatch")
        event_path = work / "candidate_events.jsonl"
        events = _strict_jsonl(event_path)
        if len(events) != child["candidate_event_count"] or any(row.get("kind") != "ionization" for row in events):
            raise OutputInventoryError("ionization_candidate_events_incomplete")
        progress = _strict_jsonl(propagation / "observer_progress.jsonl")
        starts = [row for row in progress if row.get("event") == "first_interval_started"]
        completed = [row for row in progress if row.get("event") == "complete_interval"]
        if len(starts) != 1 or [row.get("interval_index") for row in completed] != list(range(8)):
            raise OutputInventoryError("observer_progress_interval_coverage_mismatch")
        start_time = starts[0].get("monotonic")
        if not isinstance(start_time, (int, float)) or not math.isfinite(start_time) or start_time <= 0:
            raise OutputInventoryError("observer_progress_start_timestamp_invalid")
        for index, row in enumerate(completed):
            boundary = row.get("boundary_monotonic")
            if (row.get("interval_index") != index or not isinstance(boundary, (int, float))
                    or not math.isfinite(boundary) or boundary < start_time):
                raise OutputInventoryError("observer_progress_boundary_invalid")
        if not isinstance(halfsteps.get("halfstep1_json"), list) or not isinstance(halfsteps.get("halfstep2_json"), list):
            raise OutputInventoryError("linear_halfstep_profiles_missing")
        common = halfsteps.get("common_instrumentation")
        if not isinstance(common, dict) or common.get("measure_performance") is not True or common.get("diag_bk_nee_profile") is not True:
            raise OutputInventoryError("linear_halfstep_instrumentation_contract_missing")
        if mode == "PROFILE":
            profile = required_json(propagation / "function_fft_profile.json")
            if (profile.get("schema") != "filament.cpu_s4.private_i_profile.v1"
                    or profile.get("mode") != "PROFILE" or profile.get("implementation") != "I"):
                raise OutputInventoryError("PROFILE_report_schema_mismatch")
            from tools.cpu_o1.cpu_s4.profile import _validate_profile_receipt
            _validate_profile_receipt(profile,kernel_threads=child.get('local_kernel_threads',24))
            if {row.get('interval_index') for row in profile['nested_timers']['records'] if row.get('calls',0)>0 and row.get('interval_index') is not None} != set(range(8)):
                raise OutputInventoryError('PROFILE_interval_timer_coverage')
        else:
            off_profile = required_json(propagation / 'function_fft_profile.json')
            if off_profile != frozen.get('profile') or 'nested_timers' in off_profile or 'mode' in off_profile:
                raise OutputInventoryError('OFF_FFT_dispatch_report_mismatch')
        dispatch = frozen.get("profile") or {}
        if mode == "PROFILE": dispatch = dispatch.get("fft_dispatch") or {}
        workers = dispatch.get("workers") or {}
        if workers.get("temporal_workers") != 48 or workers.get("spatial_workers") != 24:
            raise OutputInventoryError("frozen_FFT_worker_identity_mismatch")
        # The frozen receipt binds content hashes. Recompute them after mode,
        # shape, dtype, field-name and interval checks.
        fhash = frozen.get("final_E_raw_sha256")
        post_hash = frozen.get("final_POST_raw_sha256")
        diag_hash = frozen.get("diagnostics_raw_sha256")
        if not all(isinstance(x, str) and len(x) == 64 for x in (fhash, post_hash, diag_hash)):
            raise OutputInventoryError("frozen_receipt_expected_raw_hash_missing")
        final_path = propagation / "final_E.npy"
        entry = _npy(final_path, np=np, raw_sha256_file=measured_raw_hash,
            canonical_array_sha256=canonical_array_sha256,
            expected_shape=tuple(source_shape), expected_dtype="complex128")
        if entry["raw_sha256"] != fhash or entry["canonical_array_sha256"] != frozen.get("final_E_canonical_sha256"):
            raise OutputInventoryError("final_E_frozen_hash_mismatch")
        qacc_entry = _npy(work / "scientific_Qacc.npy", np=np,
            raw_sha256_file=measured_raw_hash, canonical_array_sha256=canonical_array_sha256,
            expected_shape=tuple(qacc_shape), expected_dtype="float64")
        if set(frozen.get("final_POST_canonical_sha256", {})) != {"delta_n", "vx", "vy"}:
            raise OutputInventoryError("POST_required_members")
        post_entry, _ = _npz(propagation / "final_POST.npz", np=np,
            raw_sha256_file=measured_raw_hash, canonical_array_sha256=canonical_array_sha256,
            expected_hashes=frozen.get("final_POST_canonical_sha256") or {}, role="POST",
            intervals=8, ny=ny, nx=nx, nt=nt)
        diag_entry, _ = _npz(propagation / "diagnostics.npz", np=np,
            raw_sha256_file=measured_raw_hash, canonical_array_sha256=canonical_array_sha256,
            expected_hashes=frozen.get("diagnostics_canonical_sha256") or {}, role="DIAGNOSTICS",
            intervals=8, ny=ny, nx=nx, nt=nt)
        if post_entry["raw_sha256"] != post_hash or diag_entry["raw_sha256"] != diag_hash:
            raise OutputInventoryError("frozen_npz_raw_hash_mismatch")
        checked_arrays = [entry, qacc_entry, post_entry, diag_entry]
        for name in SINK_FILES:
            checked_arrays.append(_npy(propagation / name, np=np,
                raw_sha256_file=measured_raw_hash,
                canonical_array_sha256=canonical_array_sha256,
                expected_shape=(8, ny, nx), expected_dtype="float64"))
        result["arrays"] = checked_arrays
        result["hash_seconds"] = hash_seconds
        result["validation_seconds"] = time.perf_counter() - started - result["hash_seconds"]
        for path, entry in zip(required, result['entries']):
            if entry.get('status') != 'PASS' or _stat_signature(path) != entry['_signature'] or measured_raw_hash(path) != entry['raw_sha256']:
                raise OutputInventoryError('output_changed_during_acceptance:'+path.name)
        if result['failures']:
            raise OutputInventoryError('earlier_inventory_failures')
        result["status"] = "PASS"
        result['ledger_closed'] = True
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    result["inventory_seconds"] = time.perf_counter() - started
    result["total_file_bytes"] = sum(int(row.get("bytes", 0)) for row in result["entries"])
    result["raw_hash_calls"] = hash_calls
    result["raw_hash_seconds"] = hash_seconds
    result["canonical_hash_seconds"] = canonical_seconds
    result["canonical_hash_calls"] = canonical_calls
    result["hash_seconds"] = hash_seconds + canonical_seconds
    result["validation_seconds"] = result["inventory_seconds"] - result["hash_seconds"]
    for entry in result["entries"]: entry.pop("_signature", None)
    return result

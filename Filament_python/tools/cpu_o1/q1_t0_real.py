"""Single-allocation CPU-Q1 B/I/R/IR real eight-interval screen.

The coordinator and its children are diagnostic-only adapters around the
reviewed P1-R2 window runner.  They do not alter the frozen propagator or
submit scheduler work themselves.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import FunctionType


IDENTITIES = ("B", "I", "R", "IR")
SCHEMA = "filament.cpu_q1.real8.v1"
ALLOCATION_SECONDS = 7200.0
WINDOW_SECONDS = 2700.0
FIRST_INTERVAL_SECONDS = 900.0
RESERVE_SECONDS = 900.0
MEMORY_WARN_FRACTION = 0.80
MEMORY_ABORT_FRACTION = 0.90
MEMORY_SAMPLE_SECONDS = 1.0
MAX_STORAGE_CAP_BYTES = 11 * 1024**3
MAX_STORAGE_FILES = 50000
EXPECTED_EXTERNAL_INPUT_COUNT = 14
EXPECTED_FFT_WORKERS = {"temporal": 48, "spatial": 24}
EXPECTED_KERNEL_THREADS = 24
EXPECTED_LAYOUT = "NtB"
EXPECTED_BLOCK_PIXELS = 32
BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


class Q1ControlledStop(RuntimeError):
    """A durable, reviewed gate intentionally stopped the identity matrix."""


def _strict_json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False, indent=2) + "\n").encode("utf-8")


def _write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_strict_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _resolved_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def _validate_external_input_set(manifest: dict, input_root: Path,
                                 expected_count: int = EXPECTED_EXTERNAL_INPUT_COUNT) -> list[str]:
    """Select the 14 frozen optical inputs from a fully validated manifest.

    A task manifest also contains external audit objects.  Those remain covered
    by ``pv.validate_manifest`` but are not members of the optical input set.
    """
    from tools.hpc_ops import provenance_v2 as pv

    root = input_root.resolve(strict=True)
    paths = []
    for record in pv.iter_records(manifest, require_hash_scope=True):
        if record.get("classification") != "external":
            continue
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            # Non-input audit externals are validated by provenance_v2; only
            # absolute paths can be members of this separately rooted set.
            continue
        candidate = Path(raw_path).resolve(strict=False)
        if not _resolved_under(candidate, root):
            continue
        path = candidate.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"optical input external is not a file: {raw_path}")
        paths.append(str(path))
    if len(paths) != int(expected_count):
        raise ValueError(
            f"expected {expected_count} external records under input-root; got {len(paths)}")
    if len(set(paths)) != expected_count:
        raise ValueError("external input paths under input-root are not unique")
    return sorted(paths)


def _storage_inventory(root: Path) -> dict:
    """Count task-root entries without following links or deleting anything.

    Symlinks contribute their own lstat size and one inode to the file count.
    Their targets remain ordinary walk entries, so a pytest ``current`` link
    does not double-count the numbered target directory.
    """
    root = root.resolve(strict=True)
    total_bytes = 0
    files = 0
    symlinks = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(dirnames):
            item = current_path / name
            if item.is_symlink():
                symlinks.append(str(item))
                try:
                    total_bytes += item.lstat().st_size
                except OSError as exc:
                    raise OSError(f"cannot lstat storage symlink {item}: {exc}") from exc
                files += 1
                dirnames.remove(name)
        for name in filenames:
            item = current_path / name
            if item.is_symlink():
                symlinks.append(str(item))
                try:
                    total_bytes += item.lstat().st_size
                except OSError as exc:
                    raise OSError(f"cannot lstat storage symlink {item}: {exc}") from exc
                files += 1
                continue
            try:
                total_bytes += item.stat().st_size
            except OSError as exc:
                raise OSError(f"cannot stat storage item {item}: {exc}") from exc
            files += 1
    return {"root": str(root), "bytes": int(total_bytes), "files": int(files),
            "symlinks": sorted(symlinks)}


def _check_storage_growth(root: Path, *, growth_cap_bytes: int,
                          max_files: int) -> dict:
    """Enforce total task-root bytes/files; do not grant startup bytes anew."""
    state = _storage_inventory(root)
    _validate_storage_symlinks(state, root)
    if int(state["bytes"]) > int(growth_cap_bytes):
        raise RuntimeError(f"total task-root bytes exceed cap: {state['bytes']} > {growth_cap_bytes}")
    if int(state["files"]) > int(max_files):
        raise RuntimeError(f"storage file count exceeds cap: {state['files']} > {max_files}")
    state["total_bytes_cap"] = int(growth_cap_bytes)
    state["growth_cap_bytes"] = int(growth_cap_bytes)
    state["max_files"] = int(max_files)
    return state


def _validate_storage_caps(storage_cap_bytes: int, max_files: int) -> None:
    if not (0 < int(storage_cap_bytes) <= MAX_STORAGE_CAP_BYTES) or not (
            0 < int(max_files) <= MAX_STORAGE_FILES):
        raise ValueError("storage caps must be within 11 GiB and 50000 files")


def _child_environment(bench_root: Path, expected_python: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str((bench_root / "Filament_python").resolve()) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONNOUSERSITE"] = "1"
    env["UPPE_USE_GPU"] = "0"
    env["NUMBA_THREADING_LAYER"] = "workqueue"
    env["NUMBA_CPU_NAME"] = "generic"
    env["NUMBA_NUM_THREADS"] = str(EXPECTED_KERNEL_THREADS)
    for name in BLAS_THREAD_VARIABLES:
        env[name] = "1"
    env["Q1_EXPECTED_PYTHON"] = str(expected_python.resolve())
    return env


def _child_command(*, python: Path, bench_root: Path, input_root: Path,
                   vendor_dir: Path, case_root: Path, identity: str,
                   expected_source_binding: Path,
                   baseline_boundary_root: Path | None) -> list[str]:
    command = [str(python), "-s", "-B", "-m", "tools.cpu_o1.q1_t0_real", "--child",
        "--bench-root", str(bench_root), "--input-root", str(input_root),
        "--vendor-dir", str(vendor_dir), "--case-root", str(case_root),
        "--identity", identity, "--expected-source-binding",
        str(expected_source_binding), "--expected-python", str(python)]
    if baseline_boundary_root is not None:
        command.extend(("--baseline-boundary-root", str(baseline_boundary_root)))
    return command


def _decode_scalar_payload(payload: dict) -> dict:
    import numpy as np

    def decode_real(encoded):
        if encoded == "NaN":
            return float("nan")
        if encoded == "+Infinity":
            return float("inf")
        if encoded == "-Infinity":
            return float("-inf")
        if not isinstance(encoded, str):
            raise ValueError("scalar float encoding must be a string")
        return float.fromhex(encoded)

    decoded = {}
    for name, item in payload.items():
        dtype = np.dtype(item["dtype"])
        if item.get("shape") != []:
            raise ValueError(f"scalar diagnostic has non-scalar shape: {name}")
        value = item["value"]
        if dtype.kind == "c":
            value = complex(decode_real(value["real"]), decode_real(value["imag"]))
        elif dtype.kind == "f":
            value = decode_real(value)
        elif dtype.kind == "b":
            value = bool(value)
        elif dtype.kind in "iu":
            value = int(value)
        else:
            raise TypeError(f"unsupported scalar diagnostic dtype: {dtype}")
        decoded[str(name)] = np.asarray(value, dtype=dtype)[()]
    return decoded


def _read_scalar_sidecar(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "filament.cpu_q1.propagation_scalar_diagnostics.v1":
        raise ValueError(f"unknown scalar sidecar schema: {path}")
    if not isinstance(value.get("fields"), dict):
        raise ValueError(f"scalar sidecar is missing fields: {path}")
    return _decode_scalar_payload(value["fields"])


def _cgroup_memory_snapshot(pid: int) -> dict:
    """Read v2/v1 cgroup memory counters for a supervised child, if exposed."""
    cgroup_file = Path("/proc") / str(pid) / "cgroup"
    try:
        rows = cgroup_file.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        return {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"}
    v2 = next((row.split("::", 1)[1] for row in rows if "::" in row), None)
    if v2 is not None:
        base = Path("/sys/fs/cgroup") / v2.lstrip("/")
        current_path, limit_path = base / "memory.current", base / "memory.max"
    else:
        memory = next((row.split(":", 2)[2] for row in rows
                       if row.split(":", 2)[1:2] == ["memory"]), None)
        if memory is None:
            return {"status": "UNAVAILABLE", "reason": "memory_controller_not_found"}
        base = Path("/sys/fs/cgroup/memory") / memory.lstrip("/")
        current_path, limit_path = base / "memory.usage_in_bytes", base / "memory.limit_in_bytes"
    try:
        current = int(current_path.read_text(encoding="ascii").strip())
        limit_text = limit_path.read_text(encoding="ascii").strip()
        limit = None if limit_text == "max" else int(limit_text)
    except (OSError, ValueError) as exc:
        return {"status": "UNAVAILABLE", "path": str(base),
                "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": "READ", "path": str(base), "current_bytes": current,
            "limit_bytes": limit, "fraction": (current / limit if limit else None)}


def _memory_fractions(*, rss_bytes: int | None, pss_bytes: int | None,
                      cgroup: dict, allocation_bytes: int) -> dict:
    rss_fraction = (rss_bytes / allocation_bytes if rss_bytes is not None else None)
    cgroup_fraction = cgroup.get("fraction") if cgroup.get("status") == "READ" else None
    monitored = [value for value in (rss_fraction, cgroup_fraction)
                 if isinstance(value, (int, float))]
    return {"tree_rss_bytes": rss_bytes, "tree_pss_bytes": pss_bytes,
            "allocation_fraction_rss": rss_fraction,
            "cgroup": cgroup,
            "max_enforced_fraction": max(monitored) if monitored else None,
            "pss_is_report_only": True}


def _deadline_reason(*, now: float, stage_started: float,
                     first_interval_started: float | None,
                     first_boundary_arrived: bool,
                     timeout_seconds: float) -> str | None:
    if (first_interval_started is not None and not first_boundary_arrived
            and now - first_interval_started >= FIRST_INTERVAL_SECONDS):
        return "FIRST_INTERVAL_TIMEOUT"
    if now - stage_started >= timeout_seconds:
        return "WINDOW_TIMEOUT"
    return None


def _private_run_window_with_observer(j1, observer_class):
    """Clone only the J1 wrapper globals; the frozen module remains untouched."""
    original = j1._run_window
    namespace = dict(original.__globals__)
    namespace["ScalarBoundaryObserver"] = observer_class
    return FunctionType(original.__code__, namespace, original.__name__,
                        original.__defaults__, original.__closure__)


def _frozen_diag_list_contract(source_path: Path) -> dict[str, str]:
    """Extract list-backed fields from the identity-bound frozen diag mapping."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next((node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == "propagate_one_pulse"), None)
    if function is None:
        raise RuntimeError("frozen propagation source has no propagate_one_pulse function")
    dictionaries = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "diag"
                for target in node.targets) and isinstance(node.value, ast.Dict):
            dictionaries.append(node.value)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "update"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "diag" and call.args
                    and isinstance(call.args[0], ast.Dict)):
                dictionaries.append(call.args[0])
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name) and target.value.id == "diag"
                for target in node.targets):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name) and target.value.id == "diag"
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "rho_onaxis_t_z"):
                    names = {item.id for item in ast.walk(node.value)
                             if isinstance(item, ast.Name) and item.id.endswith("_list")}
                    if names == {"rho_onaxis_time_list"}:
                        dictionaries.append(ast.Dict(
                            keys=[ast.Constant("rho_onaxis_t_z")], values=[node.value]))
    contract: dict[str, str] = {}
    for mapping in dictionaries:
        for key_node, value_node in zip(mapping.keys, mapping.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            names = sorted({item.id for item in ast.walk(value_node)
                            if isinstance(item, ast.Name) and item.id.endswith("_list")})
            if len(names) != 1:
                continue
            previous = contract.get(key_node.value)
            if previous is not None and previous != names[0]:
                raise RuntimeError(f"frozen diagnostic field has ambiguous list source: {key_node.value}")
            contract[key_node.value] = names[0]
    if not contract or "z_axis" not in contract:
        raise RuntimeError("frozen diagnostic list contract is empty or lacks z_axis")
    return dict(sorted(contract.items()))


def _interval_series(locals_map: dict, diagnostic_contract: dict[str, str]
                     ) -> tuple[dict, list[str]]:
    """Capture every list-backed field required at this accepted boundary."""
    import numpy as np
    arrays = {}
    missing = []
    for field_name, local_name in diagnostic_contract.items():
        sequence = locals_map.get(local_name)
        if not isinstance(sequence, (list, tuple)) or len(sequence) != 1:
            missing.append(field_name)
            continue
        try:
            value = np.asarray(sequence)
        except (TypeError, ValueError):
            missing.append(field_name)
            continue
        if value.dtype.kind not in "biufcSU":
            missing.append(field_name)
            continue
        arrays[field_name] = value
    return arrays, missing


def _first_boundary_snapshot(frame, *, identity: str, case_root: Path,
                             observer_started: float, boundary_monotonic: float,
                             diagnostic_contract: dict[str, str]) -> dict:
    import numpy as np
    from tools.cpu_p1r import cpu_p1r_optical as p1r
    from tools.cpu_o1 import q1_t0

    loc = frame.f_locals
    E = np.asarray(loc["E"])
    if E.dtype != np.dtype("complex128") or E.ndim != 3 or not E.flags.c_contiguous:
        raise RuntimeError("first accepted boundary E violates the frozen C-complex128 contract")
    slow_state = loc.get("thermal_slow_state")
    if slow_state is None or not callable(getattr(slow_state, "post_fields", None)):
        raise RuntimeError("first accepted boundary has no frozen PRE adapter")
    post = slow_state.post_fields(0)
    if set(post) != {"delta_n", "vx", "vy"}:
        raise RuntimeError("first accepted boundary did not expose all three POST fields")
    post_arrays = {key: np.asarray(value) for key, value in post.items()}
    if any(value.dtype != np.dtype("float64") or value.shape != E.shape[-2:]
           for value in post_arrays.values()):
        raise RuntimeError("first accepted boundary POST shape/dtype differs from frozen grid")

    diag_arrays, missing_diagnostic_fields = _interval_series(loc, diagnostic_contract)
    required_scalars = ("z", "interval_index", "dz_try", "U0_baseline", "U_now",
        "adaptive_rejection_count", "safety_mode_trigger_count")
    if any(key not in loc for key in required_scalars):
        raise RuntimeError("accepted boundary lacks live scientific/topology scalars")
    scalar_payload, scalar_values = q1_t0._numeric_scalar_diagnostics(
        {name: np.asarray(loc[name]) for name in required_scalars})
    diag_root = case_root / "first_boundary"
    diag_root.mkdir(parents=True, exist_ok=False)
    p1r._save_npy_new(diag_root / "E.npy", E)
    p1r._save_npz_new(diag_root / "POST.npz", post_arrays)
    qacc_valid = False
    if "Qacc" in loc:
        qacc = np.asarray(loc["Qacc"])
        if qacc.dtype == np.dtype("float64") and qacc.shape == E.shape[-2:]:
            p1r._save_npy_new(diag_root / "Qacc.npy", qacc)
            qacc_valid = True
    qacc_raman = np.asarray(loc.get("Qacc_raman"))
    qacc_raman_valid = (qacc_raman.dtype == np.dtype("float64")
        and qacc_raman.shape == E.shape[-2:])
    if qacc_raman_valid:
        p1r._save_npy_new(diag_root / "Qacc_raman.npy", qacc_raman)
    with (diag_root / "diagnostics.npz").open("xb") as stream:
        np.savez(stream, **diag_arrays)
        stream.flush()
        os.fsync(stream.fileno())
    q1_t0._write_scalar_diagnostics(
        diag_root / "scalar_diagnostics.json", identity, scalar_payload)

    interval_seconds = float(boundary_monotonic - observer_started)
    complete = bool(qacc_valid and qacc_raman_valid and not missing_diagnostic_fields
        and set(diag_arrays) == set(diagnostic_contract))
    return {"root": str(diag_root), "identity": identity,
        "interval_index": int(loc.get("interval_index", -1)),
        "interval_computation_seconds": interval_seconds,
        "snapshot_save_seconds": time.perf_counter() - boundary_monotonic,
        "E_shape": list(E.shape), "E_dtype": str(E.dtype),
        "POST_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for key, value in post_arrays.items()},
        "Qacc_available": (diag_root / "Qacc.npy").is_file(),
        "Qacc_valid": qacc_valid, "Qacc_raman_valid": qacc_raman_valid,
        "accumulator_scope": "through_interval_0",
        "diagnostic_array_fields": sorted(diag_arrays),
        "diagnostic_scalar_fields": sorted(scalar_values),
        "diagnostic_expected_fields": sorted(diagnostic_contract),
        "diagnostic_missing_fields": sorted(missing_diagnostic_fields),
        "complete_E_POST_diagnostics_snapshot": complete,
        "first_boundary_E_raw_sha256": p1r.raw_sha256_file(diag_root / "E.npy"),
        "first_boundary_POST_raw_sha256": p1r.raw_sha256_file(diag_root / "POST.npz"),
        "first_boundary_diagnostics_raw_sha256": p1r.raw_sha256_file(diag_root / "diagnostics.npz"),
        "first_boundary_scalars_raw_sha256": p1r.raw_sha256_file(diag_root / "scalar_diagnostics.json")}


def _first_boundary_rows(reference_root: Path, candidate_root: Path,
                         identity: str, output_path: Path) -> list[dict]:
    import numpy as np
    from tools.cpu_o1 import q1_t0

    rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        def emit(row):
            record = {"schema": "filament.cpu_q1.raw_comparison.v1",
                "record_type": "array_comparison", "identity": identity,
                "comparison_stage": "first_boundary", "scope": row.get("scope", "science"), **row}
            _append_jsonl_stream(stream, record)
            rows.append(record)

        try:
            for name in ("E.npy", "Qacc.npy", "Qacc_raman.npy"):
                left_path, right_path = reference_root / name, candidate_root / name
                if not left_path.is_file() and not right_path.is_file():
                    emit({"field_name": f"first_boundary.{name}", "scope": "science",
                        "metrics": _missing_array_metrics(False, False)})
                    continue
                if left_path.is_file() != right_path.is_file():
                    emit({"field_name": f"first_boundary.{name}", "scope": "science",
                        "metrics": _missing_array_metrics(left_path.is_file(), right_path.is_file())})
                elif left_path.is_file():
                    left = np.load(left_path, mmap_mode="r", allow_pickle=False)
                    right = np.load(right_path, mmap_mode="r", allow_pickle=False)
                    emit(q1_t0._array_comparison_row(
                        f"first_boundary.{name[:-4]}", left, right, scope="science"))

            with np.load(reference_root / "POST.npz", allow_pickle=False) as left, \
                    np.load(candidate_root / "POST.npz", allow_pickle=False) as right:
                if set(left.files) != set(right.files):
                    emit({"field_name": "first_boundary.POST_fields", "scope": "science",
                        "metrics": _missing_array_metrics(set(left.files), set(right.files))})
                else:
                    for name in sorted(left.files):
                        emit(q1_t0._array_comparison_row(
                            f"first_boundary.POST.{name}", left[name], right[name], scope="science"))

            with np.load(reference_root / "diagnostics.npz", allow_pickle=False) as left, \
                    np.load(candidate_root / "diagnostics.npz", allow_pickle=False) as right:
                if set(left.files) != set(right.files):
                    emit({"field_name": "first_boundary.diagnostic_arrays", "scope": "diagnostics",
                        "metrics": _missing_array_metrics(set(left.files), set(right.files))})
                else:
                    for name in sorted(left.files):
                        scope = "timing" if name in {
                            "raman_operator_walltime_step_s", "linear_walltime_step_s",
                            "ionization_walltime_step_s", "total_walltime_step_s"} else "diagnostics"
                        emit(q1_t0._array_comparison_row(
                            name, left[name], right[name], scope=scope))

            left_scalars = _read_scalar_sidecar(reference_root / "scalar_diagnostics.json")
            right_scalars = _read_scalar_sidecar(candidate_root / "scalar_diagnostics.json")
            if set(left_scalars) != set(right_scalars):
                emit({"field_name": "first_boundary.scalar_fields", "scope": "diagnostics",
                    "metrics": _missing_array_metrics(set(left_scalars), set(right_scalars))})
            else:
                for name in sorted(left_scalars):
                    scope = "timing" if name in {
                        "raman_operator_walltime_step_s", "linear_walltime_step_s",
                        "ionization_walltime_step_s", "total_walltime_step_s"} else "diagnostics"
                    emit(q1_t0._array_comparison_row(
                        name, np.asarray(left_scalars[name]),
                        np.asarray(right_scalars[name]), scope=scope))
        except BaseException as exc:
            _append_jsonl_stream(stream, {
                "schema": "filament.cpu_q1.raw_comparison.v1",
                "record_type": "comparison_error", "identity": identity,
                "error_type": type(exc).__name__, "error": str(exc),
                "rows_written": len(rows)})
            raise
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    return rows


def _append_jsonl_stream(stream, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _missing_array_metrics(left, right) -> dict:
    left_fields = sorted(left) if isinstance(left, (set, frozenset)) else None
    right_fields = sorted(right) if isinstance(right, (set, frozenset)) else None
    left_shape = [len(left_fields)] if left_fields is not None else [int(bool(left))]
    right_shape = [len(right_fields)] if right_fields is not None else [int(bool(right))]
    return {"reference_shape": left_shape, "candidate_shape": right_shape,
        "reference_dtype": "float64", "candidate_dtype": "float64",
        "dtype_equal": True, "shape_mismatch": True, "topology_equal": False,
        "canonical_exact": False, "bitwise_equal": False, "all_finite": False,
        "reference_fields": left_fields, "candidate_fields": right_fields}


def _compare_first_boundary(identity: str, candidate_root: Path,
                            baseline_root: Path | None) -> dict:
    from tools.cpu_o1.numeric_gate import build_numeric_gate, write_numeric_gate
    snapshot = json.loads((candidate_root / "snapshot.json").read_text())
    if identity != "B" and baseline_root is None:
        raise RuntimeError("candidate lacks B reference")
    reference = (snapshot if identity == "B" else
        json.loads((baseline_root / "snapshot.json").read_text()))
    checks = {"first_boundary_observed": snapshot.get("interval_index") == 0,
        "complete_E_POST_diagnostics_snapshot": bool(snapshot.get("complete_E_POST_diagnostics_snapshot")
            and reference.get("complete_E_POST_diagnostics_snapshot"))}
    started = time.perf_counter()
    rows = _first_boundary_rows(candidate_root, candidate_root, identity, candidate_root / "self_check.jsonl")
    self_seconds = time.perf_counter()-started
    own_gate = build_numeric_gate(identity_checks=checks, linux_environment_eligible=True, comparisons=rows)
    cross_seconds = 0.0
    if identity != "B" and own_gate["proceed_to_next_identity"]:
        started = time.perf_counter()
        rows += _first_boundary_rows(baseline_root, candidate_root, identity, candidate_root / "comparison.jsonl")
        cross_seconds = time.perf_counter()-started
    gate = build_numeric_gate(identity_checks=checks, linux_environment_eligible=True, comparisons=rows)
    gate.update(identity=identity, gate_scope="REAL_FIRST_ACCEPTED_INTERVAL", comparison_rows=len(rows),
        self_validation_seconds=self_seconds, cross_identity_comparison_seconds=cross_seconds,
        production_qualification="NOT_PRODUCTION_QUALIFIED")
    write_numeric_gate(candidate_root / "NUMERIC_GATE.json", gate)
    return gate


def _observer_class(j1, *, identity: str, case_root: Path,
                    baseline_root: Path | None,
                    diagnostic_contract: dict[str, str]):
    from tools.cpu_p1 import cpu_p1_fft as common

    class Q1Observer(j1.ScalarBoundaryObserver):
        last_instance = None

        def __init__(self, code, progress_path, *, b0_limit=j1.B0_SECONDS):
            super().__init__(code, progress_path, b0_limit=float("inf"))
            self.case_root = case_root
            self.identity = identity
            self.baseline_root = baseline_root
            self.q1_first_boundary = None
            self.q1_first_gate = None
            type(self).last_instance = self

        def _callback(self, code, line):
            frame = sys._getframe(1)
            while frame is not None and frame.f_code is not code:
                frame = frame.f_back
            if frame is None:
                raise j1.J1Error("Q1 observer could not recover frozen propagate frame")
            if line != j1.p1r.INTERVAL_END_LINE or int(frame.f_locals.get("interval_index", -1)) != 0:
                return super()._callback(code, line)

            boundary = time.perf_counter()
            start = self.starts.get(0)
            if start is None:
                raise j1.J1Error("Q1 first accepted boundary has no start marker")
            common_time = time.time()
            j1._append_event(self.progress_path, {
                "event": "first_boundary_arrived", "interval_index": 0,
                "monotonic": boundary, "utc": common_time,
                "interval_computation_seconds": boundary - start})
            # Preserve the frozen J1 interval row and B0 computation clock before
            # doing snapshot I/O or a cross-identity comparison.
            base_result = super()._callback(code, line)
            if not self.rows or self.rows[0].get("interval_index") != 0:
                raise j1.J1Error("Q1 observer did not preserve the frozen first interval row")
            snapshot_started = time.perf_counter()
            snapshot = _first_boundary_snapshot(frame, identity=self.identity,
                case_root=self.case_root, observer_started=start,
                boundary_monotonic=boundary,
                diagnostic_contract=diagnostic_contract)
            snapshot["observer_post_boundary_snapshot_seconds"] = (
                time.perf_counter() - snapshot_started)
            _write_new_json(Path(snapshot["root"]) / "snapshot.json", snapshot)
            gate_started = time.perf_counter()
            gate = _compare_first_boundary(self.identity,
                Path(snapshot["root"]), self.baseline_root)
            comparison_seconds = time.perf_counter() - gate_started
            snapshot["cross_identity_comparison_seconds"] = gate["cross_identity_comparison_seconds"]
            snapshot["self_validation_seconds"] = gate["self_validation_seconds"]
            snapshot["B0_status"] = ("DONE" if boundary - start <= j1.B0_SECONDS
                                      else "LIMIT_REACHED")
            snapshot["gate_classification"] = gate["classification"]
            snapshot["gate_proceed"] = gate["proceed_to_next_identity"]
            self.q1_first_boundary = snapshot
            self.q1_first_gate = gate
            j1._json_new(self.case_root / "first_boundary_receipt.json", snapshot)
            if snapshot["B0_status"] != "DONE":
                raise Q1ControlledStop("FIRST_INTERVAL_LIMIT_REACHED")
            if not gate["proceed_to_next_identity"]:
                raise Q1ControlledStop("FIRST_BOUNDARY_NUMERIC_GATE_BLOCKED")
            return base_result

    return Q1Observer


def _private_run_window(j1, observer_class):
    return _private_run_window_with_observer(j1, observer_class)


def _load_expected_source_binding(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("expected source binding must be a non-empty JSON object")
    return value


def _check_expected_python(expected_python: Path) -> dict:
    expected = expected_python.resolve(strict=True)
    actual = Path(sys.executable).resolve(strict=True)
    if actual != expected:
        raise RuntimeError(f"fixed Q1 Python mismatch: expected {expected}; got {actual}")
    if sys.version_info[:3] != (3, 12, 8):
        raise RuntimeError(f"Q1 Python must be 3.12.8, got {sys.version.split()[0]}")
    return {"expected": str(expected), "actual": str(actual),
            "version": sys.version.split()[0], "match": True}


def _child_main(args) -> int:
    case_root = args.case_root.resolve()
    started = time.perf_counter()
    capture = {}
    observer = None
    try:
        if args.identity not in IDENTITIES:
            raise ValueError(f"unknown Q1 identity: {args.identity}")
        if not case_root.is_dir():
            raise FileNotFoundError(f"coordinator did not create case directory: {case_root}")
        expected_python = _check_expected_python(args.expected_python)
        required_env = {
            "UPPE_USE_GPU": "0", "NUMBA_THREADING_LAYER": "workqueue",
            "NUMBA_CPU_NAME": "generic", "NUMBA_NUM_THREADS": str(EXPECTED_KERNEL_THREADS),
        }
        required_env.update({name: "1" for name in BLAS_THREAD_VARIABLES})
        mismatch = {key: {"expected": value, "actual": os.environ.get(key)}
                    for key, value in required_env.items()
                    if os.environ.get(key) != value}
        if mismatch:
            raise RuntimeError(f"Q1 child environment mismatch: {json.dumps(mismatch, sort_keys=True)}")
        from tools.cpu_p1 import cpu_p1_fft as common
        from tools.cpu_p1r import cpu_p1r_optical as p1r
        from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
        from tools.cpu_o1 import q1_t0
        from tools.cpu_o1.cpu_o1 import make_joint_propagator_factory
        from importlib import import_module
        propagate_module = import_module("KHz_filament.propagate")
        import numpy as np
        import numba
        import threadpoolctl

        allocation = p1r._check_allocation()
        common.add_vendor(str(args.vendor_dir.resolve(strict=True)))
        native_limit = threadpoolctl.threadpool_limits(limits=1)
        if getattr(threadpoolctl, "__version__", None) != "3.6.0":
            raise RuntimeError("locked threadpoolctl 3.6.0 is required")
        if any(int(row.get("num_threads", 1)) > 1
               for row in threadpoolctl.threadpool_info()):
            raise RuntimeError("BLAS/native thread pool exceeded one thread")
        if int(numba.config.NUMBA_NUM_THREADS) < EXPECTED_KERNEL_THREADS:
            raise RuntimeError("Numba thread ceiling is below the fixed 24-thread contract")
        numba.set_num_threads(EXPECTED_KERNEL_THREADS)
        if int(numba.get_num_threads()) != EXPECTED_KERNEL_THREADS:
            raise RuntimeError("Numba did not accept the fixed 24-thread setting")

        expected_binding = _load_expected_source_binding(args.expected_source_binding)
        actual_binding = p1r._source_binding(args.bench_root.resolve(strict=True))
        if actual_binding != expected_binding:
            raise RuntimeError("Q1 propagation source binding differs from the staged expected binding")
        context_started = time.perf_counter()
        context = p1r._load_context(args.input_root.resolve(strict=True),
                                    args.bench_root.resolve(strict=True),
                                    args.vendor_dir.resolve(strict=True))
        context_seconds = time.perf_counter() - context_started
        if context.get("source_binding") != expected_binding:
            raise RuntimeError("verified real-input context reports a different source binding")
        source_hash = str(context["source_hash"])
        pre0_hash = str(context["assets"]["pre0_canonical_sha256"])
        pre_fields = __import__("KHz_filament.hr4e5_formal_entry",
            fromlist=["pre0_fields_from_delta_n"]).pre0_fields_from_delta_n(
                context["assets"]["pre0"], expected_count=8)

        baseline_root = (args.baseline_boundary_root.resolve(strict=True)
                         if args.baseline_boundary_root is not None else None)
        diagnostic_contract = _frozen_diag_list_contract(
            Path(propagate_module.__file__).resolve(strict=True))
        observer_type = _observer_class(j1, identity=args.identity,
            case_root=case_root, baseline_root=baseline_root,
            diagnostic_contract=diagnostic_contract)
        private_window = _private_run_window(j1, observer_type)
        factory = make_joint_propagator_factory(args.identity,
            kernel_threads=EXPECTED_KERNEL_THREADS, block_pixels=EXPECTED_BLOCK_PIXELS,
            layout=EXPECTED_LAYOUT, temporal_workers=EXPECTED_FFT_WORKERS["temporal"],
            spatial_workers=EXPECTED_FFT_WORKERS["spatial"])
        scalar_capture = q1_t0._scalar_capture_factory(factory, capture)
        window_started = time.perf_counter()
        window_receipt = private_window(
            **{key: context[key] for key in (
                "source", "grid", "beam", "prop", "ion", "heat", "raman", "schedule")},
            pre_fields=pre_fields,
            output_dir=case_root / "propagation",
            source_label="sealed_pre0_first_8", initial_pre0=True,
            profile=False, source_hash=source_hash, pre0_hash=pre0_hash,
            stage_name=f"Q1_REAL_{args.identity}", execution_scope="J1_REAL_INPUT",
            observe=True, propagator_factory=scalar_capture)
        child_window_seconds = time.perf_counter() - window_started
        observer = observer_type.last_instance
        if observer is None or observer.q1_first_boundary is None or observer.q1_first_gate is None:
            raise RuntimeError("Q1 child completed without its accepted first-boundary gate")
        if any(key not in capture for key in ("qacc", "payload", "values")):
            raise RuntimeError("private propagation did not expose Qacc and scalar diagnostics")
        if observer.q1_first_gate.get("proceed_to_next_identity") is not True:
            raise Q1ControlledStop("FIRST_BOUNDARY_NUMERIC_GATE_BLOCKED")

        q1_t0._write_new_array(case_root / "scientific_Qacc.npy", capture["qacc"])
        scalar_path = case_root / "scientific_scalar_diagnostics.json"
        q1_t0._write_scalar_diagnostics(scalar_path, args.identity, capture["payload"])
        dispatch = q1_t0._check_identity_dispatch(args.identity, factory,
            kernel_threads=EXPECTED_KERNEL_THREADS,
            temporal_workers=EXPECTED_FFT_WORKERS["temporal"],
            spatial_workers=EXPECTED_FFT_WORKERS["spatial"])
        candidate_events = list(getattr(factory.last_propagator,
            "_cpu_o1_candidate_events", ()))
        candidate_thread_counts = sorted({int(row.get("kernel_threads", 0))
            for row in candidate_events if row.get("kernel_threads") is not None})
        if args.identity in {"I", "R", "IR"}:
            if candidate_thread_counts != [EXPECTED_KERNEL_THREADS]:
                raise RuntimeError(f"compiled candidate did not execute with exactly 24 workers: {candidate_thread_counts}")
            actual_layer = str(numba.threading_layer())
            if actual_layer != "workqueue":
                raise RuntimeError(f"actual Numba layer is not workqueue: {actual_layer}")
        else:
            actual_layer = "NOT_USED_BY_BASELINE"

        window_receipt.update({
            "source_raw_sha256": p1r.SOURCE_RAW_SHA256,
            "pre0_raw_sha256": p1r.PRE0_RAW_SHA256,
            "transfer_mapping_sha256": context["assets"]["mapping_sha256"],
            "source_binding": context["source_binding"],
            "source_validation": j1._json_safe(context["source_validation"]),
            "lut_validation": j1._json_safe(context["lut"]),
            "schedule": j1._json_safe(context["schedule"].as_metadata()),
            "runtime_config_delta": j1._json_safe(context["runtime_config_delta"]),
            "child_allocation": allocation,
            "native_threadpools": threadpoolctl.threadpool_info(),
        })
        j1._json_new(case_root / "propagation" / "child_provenance_receipt.json", {
            "schema": j1.SCHEMA, "phase": j1.PHASE,
            "source_canonical_sha256": window_receipt["source_canonical_sha256"],
            "pre0_canonical_sha256": window_receipt["pre0_canonical_sha256"],
            "execution_scope": "J1_REAL_INPUT",
            "source_raw_sha256": p1r.SOURCE_RAW_SHA256,
            "pre0_raw_sha256": p1r.PRE0_RAW_SHA256,
            "transfer_mapping_sha256": context["assets"]["mapping_sha256"],
            "source_binding": actual_binding,
        })
        timing_fields = {
            "case_wall_seconds_including_comparison": time.perf_counter() - started,
            "context_load_validation_seconds": context_seconds,
            "propagation_wall_seconds_including_observer": window_receipt["propagation_wall_seconds"],
            "first_boundary_cross_identity_comparison_seconds": (
                observer.q1_first_boundary["cross_identity_comparison_seconds"]),
            "propagation_wall_estimated_without_cross_identity_comparison_seconds": max(
                0.0, float(window_receipt["propagation_wall_seconds"])
                - float(observer.q1_first_boundary["cross_identity_comparison_seconds"])),
            "window_call_wall_seconds": child_window_seconds,
            "validation_output_seconds": window_receipt["validation_output_seconds"],
            "observer_snapshot_save_seconds": observer.q1_first_boundary[
                "observer_post_boundary_snapshot_seconds"],
            "jit_cache_state": "CACHE_PREWARMED_BY_PREFLIGHT; signature-level compile/cache data in candidate events",
        }
        q1_receipt = {"schema": SCHEMA, "status": "COMPLETE_DIAGNOSTIC",
            "production_qualification": "NOT_PRODUCTION_QUALIFIED",
            "identity": args.identity, "source_binding": actual_binding,
            "source_canonical_sha256": source_hash,
            "pre0_canonical_sha256": pre0_hash,
            "fresh_source_loaded": True, "dispatch": dispatch,
            "schedule": j1._json_safe(context["schedule"].as_metadata()),
            "actual_numba_layer": actual_layer,
            "numba_threads_configured": int(numba.get_num_threads()),
            "native_threadpools": threadpoolctl.threadpool_info(),
            "first_boundary_receipt": observer.q1_first_boundary,
            "first_boundary_gate": observer.q1_first_gate,
            "scalar_diagnostic_count": len(capture["payload"]),
            "public_return_Qacc": {"path": "scientific_Qacc.npy",
                "shape": list(np.asarray(capture["qacc"]).shape),
                "dtype": str(np.asarray(capture["qacc"]).dtype)},
            "timing": timing_fields,
            "propagation_receipt": window_receipt}
        _write_new_json(case_root / "q1_child_receipt.json", q1_receipt)
        del native_limit
        return 0
    except Q1ControlledStop as exc:
        if observer is None:
            observer_type = locals().get("observer_type")
            observer = getattr(observer_type, "last_instance", None)
        receipt = {"schema": SCHEMA, "status": "BLOCKED_CONTROLLED_STOP",
            "identity": args.identity, "reason": str(exc),
            "production_qualification": "NOT_PRODUCTION_QUALIFIED",
            "first_boundary_receipt": getattr(observer, "q1_first_boundary", None),
            "first_boundary_gate": getattr(observer, "q1_first_gate", None),
            "elapsed_wall_seconds": time.perf_counter() - started,
            "traceback": traceback.format_exc()}
        try:
            _write_new_json(case_root / "child_partial_receipt.json", receipt)
        except FileExistsError:
            pass
        return 3
    except BaseException as exc:
        if observer is None:
            observer_type = locals().get("observer_type")
            observer = getattr(observer_type, "last_instance", None)
        receipt = {"schema": SCHEMA, "status": "FAIL",
            "identity": getattr(args, "identity", None),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_wall_seconds": time.perf_counter() - started,
            "first_boundary_receipt": getattr(observer, "q1_first_boundary", None),
            "first_boundary_gate": getattr(observer, "q1_first_gate", None),
            "traceback": traceback.format_exc(),
            "production_qualification": "NOT_PRODUCTION_QUALIFIED"}
        try:
            _write_new_json(case_root / "child_failure.json", receipt)
        except FileExistsError:
            pass
        traceback.print_exc()
        return 1


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl_comparisons(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("record_type") == "comparison_error":
            raise RuntimeError(f"comparison JSONL contains a tool error at line {line_number}")
        if value.get("record_type") == "array_comparison":
            rows.append(value)
    return rows


def _allocation_environment() -> dict:
    from tools.cpu_p1 import cpu_p1_fft as common
    from tools.cpu_p1r import cpu_p1r_optical as p1r

    allocation = p1r._check_allocation()
    allowed = sorted(os.sched_getaffinity(0))
    topology = common.cpu_topology(allowed)
    if len(allowed) != 96 or topology.get("physical_core_count") != 96:
        raise RuntimeError("Q1 allocation must expose exactly 96 physical CPUs")
    return {"slurm": allocation, "allowed_cpus": allowed,
        "physical_core_topology": topology,
        "lscpu": common.optional_command("lscpu", []),
        "numactl_hardware": common.optional_command("numactl", ["--hardware"]),
        "proc_status": Path("/proc/self/status").read_text(encoding="utf-8", errors="replace"),
        "proc_cpuinfo": Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace"),
        "proc_meminfo": Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")}


def _write_environment_receipt(output_root: Path, *, python_identity: dict,
                               allocation: dict, linux_probe: dict,
                               vendor_dir: Path) -> None:
    import numpy as np
    import scipy
    import numba
    import llvmlite
    import threadpoolctl
    import psutil

    from tools.cpu_p1 import cpu_p1_fft as common

    receipt = {"schema": SCHEMA, "status": "PREFLIGHT_PASS",
        "production_qualification": "NOT_PRODUCTION_QUALIFIED",
        "python": python_identity, "versions": {
            "numpy": np.__version__, "scipy": scipy.__version__,
            "numba": numba.__version__, "llvmlite": llvmlite.__version__,
            "threadpoolctl": threadpoolctl.__version__, "psutil": psutil.__version__},
        "linux_probe": linux_probe, "allocation": allocation,
        "affinity_topology": common.cpu_topology(sorted(os.sched_getaffinity(0))),
        "native_threadpools": threadpoolctl.threadpool_info(),
        "vendor_dir": str(vendor_dir),
        "dispatch_contract": {"fft_workers": EXPECTED_FFT_WORKERS,
            "kernel_threads": EXPECTED_KERNEL_THREADS, "layout": EXPECTED_LAYOUT,
            "block_pixels": EXPECTED_BLOCK_PIXELS,
            "numba_layer": "workqueue", "numba_target": "generic",
            "blas_threads": 1, "jit_cache_state": "CACHE_PREWARMED_BY_PREFLIGHT"}}
    _write_new_json(output_root / "environment.json", receipt)
    for name, key in (("lscpu.txt", "lscpu"),
                      ("numactl_hardware.txt", "numactl_hardware"),
                      ("proc_status.txt", "proc_status"),
                      ("proc_cpuinfo.txt", "proc_cpuinfo"),
                      ("proc_meminfo.txt", "proc_meminfo")):
        value = allocation[key]
        if isinstance(value, dict):
            payload = str(value.get("stdout", "UNAVAILABLE\n")).encode("utf-8")
        else:
            payload = str(value).encode("utf-8")
        _write_new_bytes(output_root / name, payload)


def _run_allocation_probe(output_root: Path) -> dict:
    """Reuse the frozen bounded step lifecycle probe within this allocation."""
    from tools.cpu_p1r2 import allocation_probe
    result_code = allocation_probe.main(["--output", str(output_root / "allocation_probe")])
    receipt = json.loads((output_root / "allocation_probe" / "receipt.json").read_text())
    if result_code != 0 or receipt.get("status") != "PASS":
        raise RuntimeError("allocation-local startup/wait/cleanup probe failed")
    return receipt


def _run_wired_tiny(output_root: Path) -> dict:
    """Prove actual 48/24 FFT dispatch and 24-thread IR wiring in allocation."""
    import numpy as np
    import numba
    import threadpoolctl
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    from tools.cpu_o1.cpu_o1 import make_joint_propagator_factory
    from tools.cpu_o1 import q1_t0

    tiny_root = output_root / "wired_tiny"
    tiny_root.mkdir(exist_ok=False)
    context = j1._tiny_context()
    capture = {}
    factory = make_joint_propagator_factory("IR",
        kernel_threads=EXPECTED_KERNEL_THREADS,
        block_pixels=EXPECTED_BLOCK_PIXELS, layout=EXPECTED_LAYOUT,
        temporal_workers=EXPECTED_FFT_WORKERS["temporal"],
        spatial_workers=EXPECTED_FFT_WORKERS["spatial"])
    wrapped_factory = q1_t0._scalar_capture_factory(factory, capture)
    numba.set_num_threads(EXPECTED_KERNEL_THREADS)
    limits = threadpoolctl.threadpool_limits(limits=1)
    receipt = j1._run_window(**{key: context[key] for key in (
            "source", "pre_fields", "grid", "beam", "prop", "ion", "heat",
            "raman", "schedule")},
        output_dir=tiny_root / "propagation", source_label="synthetic_tiny_PRE0",
        initial_pre0=True, profile=False, source_hash=j1._canonical_hash(context["source"]),
        pre0_hash=j1._canonical_hash(context["pre_fields"]["delta_n"]),
        stage_name="Q1_ALLOCATION_WIRED_TINY", execution_scope="SOFTWARE_SMOKE_ONLY",
        observe=False, propagator_factory=wrapped_factory)
    if any(key not in capture for key in ("qacc", "payload", "values")):
        raise RuntimeError("allocation wired-tiny did not return Qacc/scalar diagnostics")
    dispatch = q1_t0._check_identity_dispatch("IR", factory,
        kernel_threads=EXPECTED_KERNEL_THREADS,
        temporal_workers=EXPECTED_FFT_WORKERS["temporal"],
        spatial_workers=EXPECTED_FFT_WORKERS["spatial"])
    events = list(getattr(factory.last_propagator, "_cpu_o1_candidate_events", ()))
    actual_threads = sorted({int(row.get("kernel_threads", 0)) for row in events
                             if row.get("kernel_threads") is not None})
    if actual_threads != [EXPECTED_KERNEL_THREADS]:
        raise RuntimeError(f"allocation wired-tiny candidate worker count differs: {actual_threads}")
    if str(numba.threading_layer()) != "workqueue" or int(numba.get_num_threads()) != EXPECTED_KERNEL_THREADS:
        raise RuntimeError("allocation wired-tiny did not initialize workqueue at 24 threads")
    if any(int(row.get("num_threads", 1)) > 1 for row in threadpoolctl.threadpool_info()):
        raise RuntimeError("allocation wired-tiny exceeded the one-thread native pool limit")
    q1_t0._write_new_array(tiny_root / "scientific_Qacc.npy", capture["qacc"])
    q1_t0._write_scalar_diagnostics(
        tiny_root / "scientific_scalar_diagnostics.json", "IR", capture["payload"])
    output = {"schema": SCHEMA, "status": "WIRED_TINY_ONLY",
        "identity": "IR", "production_qualification": "NOT_PRODUCTION_QUALIFIED",
        "synthetic_input_only": True,
        "dispatch": dispatch, "candidate_thread_counts": actual_threads,
        "numba_layer": str(numba.threading_layer()),
        "numba_threads": int(numba.get_num_threads()),
        "native_threadpools": threadpoolctl.threadpool_info(),
        "propagation_receipt": receipt}
    _write_new_json(tiny_root / "wired_tiny_receipt.json", output)
    del limits
    return output


def _supervise_child(command: list[str], *, case_root: Path, cwd: Path,
                     env: dict[str, str], timeout_seconds: float,
                     memory_bytes: int) -> dict:
    """One-second file-log, first-interval, process-tree and cgroup supervisor."""
    from tools.cpu_p1 import cpu_p1_fft as common
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    stdout_path, stderr_path = case_root / "stdout.log", case_root / "stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("Q1 child logs already exist")
    kwargs = {"cwd": str(cwd), "env": env,
        "stdout": stdout_path.open("xb"), "stderr": stderr_path.open("xb"),
        "start_new_session": True}
    started = time.monotonic()
    proc = None
    stop = None
    peak_rss = 0
    peak_pss = 0
    samples = []
    progress_path = case_root / "propagation" / "observer_progress.jsonl"
    progress_offset = 0
    pending_progress = b""
    first_start = None
    first_arrival = False
    rss_latched = False
    latest_cgroup = {"status": "UNAVAILABLE"}
    try:
        proc = subprocess.Popen(list(command), **kwargs)
    except BaseException:
        kwargs["stdout"].close()
        kwargs["stderr"].close()
        raise
    kwargs["stdout"].close()
    kwargs["stderr"].close()

    try:
        while proc.poll() is None:
            sample_started = time.monotonic()
            if progress_path.is_file():
                with progress_path.open("rb") as stream:
                    stream.seek(progress_offset)
                    payload = stream.read()
                    progress_offset = stream.tell()
                if payload:
                    pending_progress += payload
                    complete_lines, pending_progress = _complete_jsonl_lines(pending_progress)
                    for raw_line in complete_lines:
                        if not raw_line.strip():
                            continue
                        event = json.loads(raw_line.decode("utf-8"))
                        if event.get("event") == "first_interval_started":
                            first_start = float(event["monotonic"])
                        elif event.get("event") == "first_boundary_arrived":
                            first_arrival = True
    
            tree = common.tree_snapshot(proc.pid)
            rss = int(tree["rss"]) if tree is not None else None
            pss_state = j1._pss_tree_snapshot(proc.pid)
            pss = pss_state.get("pss_bytes")
            cgroup = _cgroup_memory_snapshot(proc.pid)
            latest_cgroup = cgroup
            if rss is not None:
                peak_rss = max(peak_rss, rss)
            if pss is not None:
                peak_pss = max(peak_pss, int(pss))
            memory = _memory_fractions(rss_bytes=rss, pss_bytes=pss,
                                       cgroup=cgroup, allocation_bytes=memory_bytes)
            fraction = memory["max_enforced_fraction"]
            if fraction is None and proc.poll() is None:
                raise RuntimeError("MEMORY_TELEMETRY_UNAVAILABLE")
            if isinstance(fraction, (int, float)) and fraction >= MEMORY_WARN_FRACTION:
                rss_latched = True
                latch_path = case_root / "RSS80_LATCH.json"
                if not latch_path.exists():
                    _write_new_json(latch_path, {"reason": "MEMORY_80_PERCENT",
                        "sample": memory, "monotonic_elapsed_seconds": sample_started - started})
            samples.append({"monotonic_elapsed_seconds": sample_started - started,
                            **memory, "process_tree_cpu_seconds": (
                                float(tree.get("cpu", 0.0)) if tree is not None else None)})
    
            reason = _deadline_reason(now=sample_started, stage_started=started,
                first_interval_started=first_start,
                first_boundary_arrived=first_arrival,
                timeout_seconds=timeout_seconds)
            if isinstance(fraction, (int, float)) and fraction >= MEMORY_ABORT_FRACTION:
                reason = "MEMORY_90_PERCENT"
            if reason is not None:
                stop = {"reason": reason, **j1._terminate_child_bounded(proc)}
                break
            sleep_for = MEMORY_SAMPLE_SECONDS - (time.monotonic() - sample_started)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except BaseException as exc:
        stop = {"reason": "SUPERVISOR_TOOL_FAILURE", "error": type(exc).__name__ + ": " + str(exc),
                "traceback": traceback.format_exc(), **j1._terminate_child_bounded(proc)}
    finally:
        if proc.poll() is None and stop is not None:
            j1._terminate_child_bounded(proc)

    try:
        exit_code = proc.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        stop = {"reason": "POST_STOP_WAIT_EXPIRED", **j1._terminate_child_bounded(proc)}
        try:
            exit_code = proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            exit_code = None
    elapsed = time.monotonic() - started
    result = {"schema": SCHEMA, "exit_code": exit_code,
        "elapsed_wall_seconds": elapsed, "timeout_seconds": float(timeout_seconds),
        "stop": stop, "first_interval_start_monotonic": first_start,
        "first_boundary_arrived": first_arrival,
        "first_interval_deadline_seconds": FIRST_INTERVAL_SECONDS,
        "memory_warn_fraction": MEMORY_WARN_FRACTION,
        "memory_abort_fraction": MEMORY_ABORT_FRACTION,
        "rss_80_percent_latched": rss_latched,
        "peak_process_tree_rss_bytes": peak_rss or None,
        "peak_process_tree_pss_bytes": peak_pss or None,
        "last_cgroup_sample": latest_cgroup,
        "sample_interval_seconds": MEMORY_SAMPLE_SECONDS,
        "samples": samples,
        "log_paths": {"stdout": str(stdout_path), "stderr": str(stderr_path),
                      "observer_progress": str(progress_path)}}
    _write_new_json(case_root / "supervisor_receipt.json", result)
    return result


def _coordinator_preflight(args) -> dict:
    """Validate code, 14 sealed inputs, interpreter, allocation and pinned Linux env."""
    from tools.hpc_ops import provenance_v2 as pv

    bench_root = args.bench_root.resolve(strict=True)
    input_root = args.input_root.resolve(strict=True)
    vendor_dir = args.vendor_dir.resolve(strict=True)
    provenance_path = args.provenance.resolve(strict=True)
    expected_binding_path = args.expected_source_binding.resolve(strict=True)
    storage_root = args.storage_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    expected_python = _check_expected_python(args.expected_python)
    if not (bench_root / "Filament_python").is_dir():
        raise FileNotFoundError("bench-root has no Filament_python directory")
    if output_root.exists():
        raise FileExistsError(f"Q1 output root must be new: {output_root}")
    if not _resolved_under(output_root.parent, storage_root):
        raise ValueError("output-root must be inside storage-root")
    _validate_storage_caps(args.storage_cap_bytes, args.max_files)
    storage_before = _storage_inventory(storage_root)
    _validate_storage_symlinks(storage_before, storage_root)
    if storage_before["files"] > int(args.max_files):
        raise ValueError("storage file count already exceeds the configured cap")

    manifest = pv.validate_manifest(bench_root, provenance_path,
        require_clean=True, require_hash_scope=True)
    external_paths = _validate_external_input_set(manifest, input_root)
    expected_binding = _load_expected_source_binding(expected_binding_path)

    incoming_gpu = os.environ.get("UPPE_USE_GPU")
    if incoming_gpu not in (None, "0", "false", "False"):
        raise RuntimeError(f"GPU environment must be disabled before Q1; got {incoming_gpu!r}")
    os.environ["UPPE_USE_GPU"] = "0"
    for name in BLAS_THREAD_VARIABLES:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be fixed to one before coordinator imports numerical libraries")
    if os.environ.get("NUMBA_THREADING_LAYER") != "workqueue":
        raise RuntimeError("NUMBA_THREADING_LAYER must be workqueue")
    if os.environ.get("NUMBA_CPU_NAME") != "generic":
        raise RuntimeError("NUMBA_CPU_NAME must be generic")
    if os.environ.get("NUMBA_NUM_THREADS") != str(EXPECTED_KERNEL_THREADS):
        raise RuntimeError("NUMBA_NUM_THREADS must be fixed to 24")

    from tools.cpu_p1 import cpu_p1_fft as common
    from tools.cpu_p1r import cpu_p1r_optical as p1r
    import threadpoolctl
    native_limit = threadpoolctl.threadpool_limits(limits=1)
    allocation = _allocation_environment()
    current_binding = p1r._source_binding(bench_root)
    if current_binding != expected_binding:
        raise RuntimeError("current frozen propagation source binding differs from expected-source-binding")
    linux_probe = __import__("tools.cpu_o1.linux_q1_preflight",
        fromlist=["run_linux_q1_preflight"]).run_linux_q1_preflight()
    if not linux_probe.get("eligible"):
        raise RuntimeError("pinned Linux environment gate blocked: " +
            json.dumps(linux_probe.get("reasons", []), ensure_ascii=False))

    context_started = time.perf_counter()
    context = p1r._load_context(input_root, bench_root, vendor_dir)
    context_seconds = time.perf_counter() - context_started
    if context.get("source_binding") != expected_binding:
        raise RuntimeError("validated optical input context carries a different source binding")
    if len(context["assets"].get("records", {})) not in (0, 11):
        raise RuntimeError("verified optical asset mapping has an unexpected record count")
    input_identity = {"source_canonical_sha256": context["source_hash"],
        "source_raw_sha256": p1r.SOURCE_RAW_SHA256,
        "pre0_canonical_sha256": context["assets"]["pre0_canonical_sha256"],
        "pre0_raw_sha256": p1r.PRE0_RAW_SHA256,
        "mapping_sha256": context["assets"]["mapping_sha256"],
        "lut_validation": context["lut"],
        "schedule": context["schedule"].as_metadata(),
        "runtime_config_delta": context["runtime_config_delta"]}
    del context
    import gc
    gc.collect()

    return {"bench_root": str(bench_root), "input_root": str(input_root),
        "vendor_dir": str(vendor_dir), "provenance_path": str(provenance_path),
        "provenance_sha256": p1r.raw_sha256_file(provenance_path),
        "provenance_head": manifest["head"],
        "external_input_paths": external_paths,
        "external_input_count": len(external_paths),
        "expected_source_binding": expected_binding,
        "source_binding": current_binding,
        "input_identity": input_identity,
        "input_context_validation_seconds": context_seconds,
        "python": expected_python,
        "allocation": allocation,
        "linux_environment": linux_probe,
        "storage_initial": storage_before,
        "storage_cap_bytes": int(args.storage_cap_bytes),
        "max_files": int(args.max_files),
        "native_threadpool_limit": "ONE",
        "native_threadpools": threadpoolctl.threadpool_info(),
        "git_common_path": common.__file__,
        "held_native_thread_limit": native_limit}


def _stage_endpoint_comparison(identity: str, baseline_root: Path,
                               case_root: Path) -> tuple[list[dict], dict]:
    from tools.cpu_o1 import q1_t0
    from tools.cpu_o1.numeric_gate import build_numeric_gate, write_numeric_gate
    candidate_scalar = _read_scalar_sidecar(case_root / "scientific_scalar_diagnostics.json")
    rows=[];self_seconds=0.0;cross_seconds=0.0
    try:
        started=time.perf_counter()
        rows=q1_t0._compare_window_outputs(case_root, case_root, identity,
            comparison_path=case_root / "endpoint_self_check.jsonl",
            scalar_reference=candidate_scalar,scalar_candidate=candidate_scalar)
        self_seconds=time.perf_counter()-started
        gate=build_numeric_gate(identity_checks={"endpoint_files_validated":True},
            linux_environment_eligible=True,comparisons=rows)
        if identity != "B" and gate["proceed_to_next_identity"]:
            baseline_scalar=_read_scalar_sidecar(baseline_root / "scientific_scalar_diagnostics.json")
            started=time.perf_counter()
            rows+=q1_t0._compare_window_outputs(baseline_root,case_root,identity,
                comparison_path=case_root / "endpoint_comparison.jsonl",
                scalar_reference=baseline_scalar,scalar_candidate=candidate_scalar)
            cross_seconds=time.perf_counter()-started
            gate=build_numeric_gate(identity_checks={"endpoint_files_validated":True},
                linux_environment_eligible=True,comparisons=rows)
    except BaseException as exc:
        gate=build_numeric_gate(identity_checks={"endpoint_files_validated":False},
            linux_environment_eligible=True,tool_error=type(exc).__name__+": "+str(exc))
    gate.update(identity=identity,gate_scope="COMPLETE_EIGHT_INTERVAL_ENDPOINT",comparison_rows=len(rows),
        self_validation_seconds=self_seconds,cross_identity_comparison_seconds=cross_seconds,
        production_qualification="NOT_PRODUCTION_QUALIFIED")
    write_numeric_gate(case_root / "NUMERIC_GATE_endpoint.json",gate)
    return rows,gate


def _check_stage_identity(identity: str, case_root: Path,
                          baseline_receipt: dict | None) -> dict:
    q1_receipt = json.loads((case_root / "q1_child_receipt.json").read_text(encoding="utf-8"))
    if q1_receipt.get("identity") != identity or q1_receipt.get("status") != "COMPLETE_DIAGNOSTIC":
        raise RuntimeError("Q1 child receipt does not identify the complete requested case")
    current = q1_receipt["propagation_receipt"]
    if (not q1_receipt.get("schedule") or current.get("execution_scope") != "J1_REAL_INPUT"
            or current.get("interval_count") != 8):
        raise RuntimeError("Q1 child receipt does not prove a real complete eight-interval window")
    if baseline_receipt is not None:
        baseline = baseline_receipt["propagation_receipt"]
        pairs = (("source_canonical_sha256", "source_canonical_sha256"),
                 ("pre0_canonical_sha256", "pre0_canonical_sha256"))
        if any(current.get(left) != baseline.get(right) for left, right in pairs):
            raise RuntimeError("source or PRE0 identity drifted between Q1 cases")
        if not q1_receipt.get("schedule") or q1_receipt["schedule"] != baseline_receipt.get("schedule"):
            raise RuntimeError("eight-interval schedule changed between Q1 cases")
    validated = __import__("tools.cpu_p1r2.cpu_p1r2_j1",
        fromlist=["_validate_stage"])._validate_stage(case_root)
    return {"q1_receipt": q1_receipt, "validation": validated}


def _run_coordinator(args) -> dict:
    from tools.cpu_o1.numeric_gate import build_numeric_gate, write_numeric_gate
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    from tools.cpu_p1r import cpu_p1r_optical as p1r
    from tools.cpu_o1 import q1_t0

    allocation_started = time.monotonic()
    output_root = args.output_root.resolve(strict=False)
    storage_root = args.storage_root.resolve(strict=True)
    preflight = _coordinator_preflight(args)
    output_root.mkdir(parents=True, exist_ok=False)
    storage_initial_bytes = int(preflight["storage_initial"]["bytes"])
    coordinator_receipt = {"schema": SCHEMA,
        "status": "Q1_RUNNING", "identity_order": list(IDENTITIES),
        "production_qualification": "NOT_PRODUCTION_QUALIFIED",
        "task_classification": "CPU_Q1_DIAGNOSTIC_ONLY",
        "allocation_started_monotonic": allocation_started,
        "allocation_budget_seconds": ALLOCATION_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "preflight": {key: value for key, value in preflight.items()
                      if key != "held_native_thread_limit"},
        "cases": []}
    _write_new_json(output_root / "preflight.json", coordinator_receipt["preflight"])
    _write_environment_receipt(output_root, python_identity=preflight["python"],
        allocation=preflight["allocation"], linux_probe=preflight["linux_environment"],
        vendor_dir=args.vendor_dir.resolve())
    job_env = {name: os.environ.get(name) for name in (
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_QOS", "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST")}
    _write_new_json(output_root / "scheduler_allocation.json", {
        "schema": SCHEMA, "allocation": job_env,
        "not_a_submission": True, "gpu": False,
        "production_qualification": "NOT_PRODUCTION_QUALIFIED"})
    memory_bytes = p1r.MEMORY_LIMIT_BYTES
    rss_latch = False
    baseline_receipt = None
    baseline_case = None
    previous_stage_wall = None
    terminal_status = None

    try:
        from tools.cpu_o1 import linux_q1_preflight
        import numba
        import threadpoolctl
        native_limit = threadpoolctl.threadpool_limits(limits=1)
        numba.set_num_threads(EXPECTED_KERNEL_THREADS)
        coordinator_receipt["allocation_probe"] = _run_allocation_probe(output_root)
        tiny = _run_wired_tiny(output_root)
        coordinator_receipt["allocation_wired_tiny"] = tiny
        _append_jsonl(output_root / "run_progress.jsonl", {
            "event": "allocation_wired_tiny_complete", "status": tiny["status"],
            "dispatch": tiny["dispatch"]})
        if not tiny.get("status") == "WIRED_TINY_ONLY":
            raise RuntimeError("allocation 48/24/24 wired-tiny did not pass")
        del native_limit

        for index, identity in enumerate(IDENTITIES):
            try:
                storage_state = _check_storage_growth(storage_root,
                    growth_cap_bytes=int(args.storage_cap_bytes),
                    max_files=int(args.max_files))
            except BaseException as exc:
                terminal_status = "Q1_PREFLIGHT_BLOCKED_STORAGE"
                coordinator_receipt["cases"].append({"identity": identity,
                    "status": terminal_status,
                    "error": f"{type(exc).__name__}: {exc}"})
                break
            elapsed_allocation = time.monotonic() - allocation_started
            remaining = ALLOCATION_SECONDS - elapsed_allocation
            if previous_stage_wall is None:
                predicted = WINDOW_SECONDS
            else:
                predicted = min(WINDOW_SECONDS,
                    max(300.0, float(previous_stage_wall) * 1.5))
            if remaining < predicted + RESERVE_SECONDS:
                terminal_status = "SKIPPED_INSUFFICIENT_TIME"
                coordinator_receipt["cases"].append({"identity": identity,
                    "status": terminal_status, "remaining_seconds": remaining,
                    "predicted_seconds": predicted,
                    "required_reserve_seconds": RESERVE_SECONDS})
                break
            case_root = output_root / "identities" / identity
            case_root.parent.mkdir(parents=True, exist_ok=True)
            case_root.mkdir(exist_ok=False)
            baseline_boundary_root = (
                baseline_case / "first_boundary" if identity != "B" and baseline_case else None)
            command = _child_command(python=args.expected_python,
                bench_root=args.bench_root, input_root=args.input_root,
                vendor_dir=args.vendor_dir, case_root=case_root, identity=identity,
                expected_source_binding=args.expected_source_binding,
                baseline_boundary_root=baseline_boundary_root)
            env = _child_environment(args.bench_root, args.expected_python)
            stage_timeout = min(WINDOW_SECONDS, max(1.0, remaining - RESERVE_SECONDS))
            _append_jsonl(output_root / "run_progress.jsonl", {
                "event": "identity_started", "identity": identity,
                "monotonic": time.monotonic(), "remaining_allocation_seconds": remaining,
                "predicted_case_seconds": predicted, "timeout_seconds": stage_timeout,
                "storage_before_identity": storage_state,
                "command": command})
            supervisor = _supervise_child(command, case_root=case_root,
                cwd=args.bench_root.resolve() / "Filament_python", env=env,
                timeout_seconds=stage_timeout, memory_bytes=memory_bytes)
            case = {"identity": identity, "status": "RUNNING_RESULT_PENDING_VALIDATION",
                "supervisor": supervisor,
                "child_wall_seconds_including_all_child_work": supervisor["elapsed_wall_seconds"],
                "command": command,
                "comparison_policy": "cross-identity comparisons are separately timed and excluded from uniform service estimate; all such costs remain in coordinator/allocation wall"}
            coordinator_receipt["cases"].append(case)
            if supervisor.get("rss_80_percent_latched"):
                rss_latch = True
            if supervisor.get("stop"):
                stop_reason = supervisor["stop"].get("reason")
                case["status"] = {
                    "FIRST_INTERVAL_TIMEOUT": "LIMIT_REACHED_FIRST_INTERVAL",
                    "WINDOW_TIMEOUT": "LIMIT_REACHED_WINDOW",
                    "MEMORY_90_PERCENT": "STOPPED_MEMORY_90_PERCENT",
                }.get(stop_reason, "STOPPED_RESOURCE_GATE")
                terminal_status = case["status"]
                _append_jsonl(output_root / "run_progress.jsonl", {
                    "event": "identity_stopped", "identity": identity,
                    "status": case["status"], "reason": stop_reason})
                break
            if supervisor.get("exit_code") == 3:
                partial_path = case_root / "child_partial_receipt.json"
                partial = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.is_file() else {}
                case["status"] = partial.get("reason", "BLOCKED_CONTROLLED_STOP")
                case["partial_receipt"] = partial
                terminal_status = case["status"]
                break
            if supervisor.get("exit_code") != 0:
                case["status"] = "FAIL_CHILD"
                failure_path = case_root / "child_failure.json"
                if failure_path.is_file():
                    case["child_failure"] = json.loads(failure_path.read_text(encoding="utf-8"))
                terminal_status = "FAIL_CHILD"
                break

            try:
                own_validation_started=time.perf_counter()
                validated = _check_stage_identity(identity, case_root,
                    baseline_receipt if identity != "B" else None)
                own_validation_seconds=time.perf_counter()-own_validation_started
                if identity == "B":
                    baseline_case = case_root
                    baseline_receipt = validated["q1_receipt"]
                    first_gate = json.loads((case_root / "first_boundary" / "NUMERIC_GATE.json").read_text(encoding="utf-8"))
                    if first_gate.get("proceed_to_next_identity") is not True:
                        raise Q1ControlledStop("B_FIRST_BOUNDARY_HARD_GATE_FAILED")
                validation_started = time.perf_counter()
                rows, endpoint_gate = _stage_endpoint_comparison(
                    identity, baseline_case, case_root)
                first_root = case_root / "first_boundary"
                first_jsonl = (first_root / "self_check.jsonl" if identity == "B"
                               else first_root / "comparison.jsonl")
                first_rows = _read_jsonl_comparisons(first_jsonl)
                from tools.cpu_o1.numeric_gate import build_numeric_gate
                final_gate = build_numeric_gate(identity_checks={
                    "linux_preflight": bool(preflight["linux_environment"].get("eligible")),
                    "allocation_96_physical_cores": len(preflight["allocation"]["allowed_cpus"]) == 96,
                    "source_and_PRE0_match_B": True,
                    "fixed_dispatch_exercised": bool(validated["q1_receipt"]["dispatch"]["fft_dispatch"].get("transform_counts")),
                    "first_boundary_gate": json.loads((first_root / "NUMERIC_GATE.json").read_text())["proceed_to_next_identity"] is True,
                    "endpoint_gate": endpoint_gate.get("proceed_to_next_identity") is True,
                }, linux_environment_eligible=True,
                    comparisons=first_rows + rows)
                final_gate.update({"identity": identity,
                    "gate_scope": "REAL_Q1_EIGHT_INTERVAL_WINDOW",
                    "first_boundary_gate": json.loads((first_root / "NUMERIC_GATE.json").read_text(encoding="utf-8")),
                    "endpoint_gate": endpoint_gate,
                    "first_boundary_comparison_rows": len(first_rows),
                    "endpoint_comparison_rows": len(rows),
                    "production_qualification": "NOT_PRODUCTION_QUALIFIED"})
                write_numeric_gate(case_root / "NUMERIC_GATE.json", final_gate)
                case["validation"] = validated
                case["endpoint_gate"] = endpoint_gate
                case["numeric_gate"] = final_gate
                case["service_seconds_uniform_boundary_estimate"] = max(0.0,
                    float(validated["q1_receipt"]["timing"]["case_wall_seconds_including_comparison"])
                    - float(validated["q1_receipt"]["timing"][
                        "first_boundary_cross_identity_comparison_seconds"])) + own_validation_seconds + endpoint_gate["self_validation_seconds"]
                case["endpoint_validation_and_comparison_seconds"] = time.perf_counter() - validation_started
                case["endpoint_cross_identity_comparison_seconds"] = (
                    endpoint_gate["cross_identity_comparison_seconds"])
                case["status"] = "DONE" if final_gate.get("proceed_to_next_identity") else final_gate["classification"]
                previous_stage_wall = supervisor["elapsed_wall_seconds"] + case["endpoint_validation_and_comparison_seconds"]
                _append_jsonl(output_root / "run_progress.jsonl", {
                    "event": "identity_completed", "identity": identity,
                    "status": case["status"],
                    "service_seconds_uniform_boundary_estimate": case["service_seconds_uniform_boundary_estimate"],
                    "endpoint_gate_classification": endpoint_gate.get("classification")})
                if not final_gate.get("proceed_to_next_identity"):
                    terminal_status = "BLOCKED_NUMERIC_GATE"
                    break
                if rss_latch:
                    terminal_status = "BLOCKED_RSS_80_PERCENT_LATCH"
                    break
            except Q1ControlledStop as exc:
                case["status"] = str(exc)
                case["validation_error"] = f"{type(exc).__name__}: {exc}"
                terminal_status = case["status"]
                break
            except BaseException as exc:
                case["status"] = "FAIL_OUTPUT_OR_COMPARISON_VALIDATION"
                case["validation_error"] = f"{type(exc).__name__}: {exc}"
                case["validation_traceback"] = traceback.format_exc()
                terminal_status = case["status"]
                break
            try:
                _check_storage_growth(storage_root,
                    growth_cap_bytes=int(args.storage_cap_bytes),
                    max_files=int(args.max_files))
            except BaseException as exc:
                terminal_status = "Q1_STORAGE_CAP_REACHED"
                case["storage_error"] = f"{type(exc).__name__}: {exc}"
                break

        if terminal_status is None:
            if len([row for row in coordinator_receipt["cases"]
                    if row.get("status") == "DONE"]) == len(IDENTITIES):
                terminal_status = "Q1_COMPLETE_FOUR_IDENTITY_SCREENING"
            else:
                terminal_status = "Q1_PARTIAL_MATRIX"
        coordinator_receipt["status"] = terminal_status
        coordinator_receipt["rss_80_percent_latched"] = rss_latch
        coordinator_receipt["allocation_elapsed_seconds"] = time.monotonic() - allocation_started
        coordinator_receipt["storage_final"] = _check_storage_growth(storage_root,
            growth_cap_bytes=int(args.storage_cap_bytes),
            max_files=int(args.max_files))
        coordinator_receipt["next_action"] = "MANUAL_REVIEW_ONLY"
        _write_new_json(output_root / "coordinator_receipt.json", coordinator_receipt)
        final_comparisons = []
        for row in coordinator_receipt["cases"]:
            final_comparisons.extend(row.get("numeric_gate", {}).get("comparisons", []))
        final_gate = build_numeric_gate(identity_checks={
            "all_four_identities_completed": terminal_status == "Q1_COMPLETE_FOUR_IDENTITY_SCREENING",
            "linux_environment_eligible": bool(preflight["linux_environment"].get("eligible")),
            "resource_budget_respected": not rss_latch,
        }, linux_environment_eligible=bool(preflight["linux_environment"].get("eligible")),
            comparisons=final_comparisons,
            tool_error=None)
        if terminal_status != "Q1_COMPLETE_FOUR_IDENTITY_SCREENING":
            # Partial scheduling/resource state is not itself a comparison-tool failure.
            final_gate["classification"] = "Q1_PARTIAL_MATRIX"
            for case in reversed(coordinator_receipt["cases"]):
                observed = (case.get("partial_receipt", {}).get("first_boundary_gate")
                    or case.get("endpoint_gate") or case.get("numeric_gate"))
                if observed and not observed.get("proceed_to_next_identity"):
                    final_gate["classification"] = observed["classification"]
                    final_gate["blocking_gate"] = observed
                    break
            final_gate["proceed_to_next_identity"] = False
        final_gate["matrix_status"] = terminal_status
        final_gate.update({"gate_scope": "REAL_Q1_MATRIX",
            "identities_completed": [row["identity"] for row in coordinator_receipt["cases"]
                                     if row.get("status") == "DONE"],
            "coordinator_receipt": "coordinator_receipt.json",
            "production_qualification": "NOT_PRODUCTION_QUALIFIED"})
        write_numeric_gate(output_root / "NUMERIC_GATE.json", final_gate)
        return coordinator_receipt
    except BaseException as exc:
        coordinator_receipt.update({"status": "Q1_PREFLIGHT_OR_COORDINATOR_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "allocation_elapsed_seconds": time.monotonic() - allocation_started,
            "next_action": "MANUAL_REVIEW_ONLY"})
        try:
            _write_new_json(output_root / "coordinator_failure.json", coordinator_receipt)
            failure_gate = build_numeric_gate(identity_checks={
                "coordinator_completed": False,
                "linux_environment_eligible": bool(preflight["linux_environment"].get("eligible")),
            }, linux_environment_eligible=bool(preflight["linux_environment"].get("eligible")),
                tool_error=f"{type(exc).__name__}: {exc}")
            failure_gate.update({"gate_scope": "REAL_Q1_MATRIX",
                "production_qualification": "NOT_PRODUCTION_QUALIFIED"})
            write_numeric_gate(output_root / "NUMERIC_GATE.json", failure_gate)
        except FileExistsError:
            pass
        raise






def _validate_storage_symlinks(state, root):
    root = Path(root).resolve(strict=True)
    allowed = root / "environment" / "venv"
    for text in state["symlinks"]:
        link = Path(text)
        try:
            link.relative_to(allowed)
            continue
        except ValueError:
            pass

        # pytest tmp_path leaves one convenience link beside its numbered
        # directory: preflight/<stage>/pytest_tmp/test_*current -> test_*<n>.
        # Admit only an existing real sibling directory with the matching
        # numeric instance name. Inventory does not traverse the link.
        try:
            relative = link.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("unexpected task-root symlink: " + text) from exc
        parts = relative.parts
        expected_shape = (len(parts) == 4 and parts[0] == "preflight"
                          and parts[1] not in ("", ".", "..")
                          and parts[2] == "pytest_tmp")
        suffix = "current"
        if (not expected_shape or not link.name.startswith("test_")
                or not link.name.endswith(suffix)):
            raise RuntimeError("unexpected task-root symlink: " + text)
        target_prefix = link.name[:-len(suffix)]
        try:
            raw_target = Path(os.readlink(link))
            target_entry = raw_target if raw_target.is_absolute() else link.parent / raw_target
            parent = link.parent.resolve(strict=True)
            if (target_entry.parent.resolve(strict=True) != parent
                    or target_entry.is_symlink()
                    or not target_entry.is_dir()):
                raise ValueError("pytest current target is not a real sibling directory")
            target = target_entry.resolve(strict=True)
            target_suffix = (target.name[len(target_prefix):]
                             if target.name.startswith(target_prefix) else "")
            if (target.parent != parent or not target_suffix
                    or not target_suffix.isascii() or not target_suffix.isdigit()):
                raise ValueError("pytest current target name is not the matching numbered instance")
        except (OSError, ValueError) as exc:
            raise RuntimeError("unexpected task-root symlink: " + text) from exc


def _complete_jsonl_lines(payload):
    parts = payload.split(b"\n")
    return parts[:-1], parts[-1]


def _entry_smoke(args):
    """Actual fresh interpreter and file-log round trip; no optical assets loaded."""
    _check_expected_python(args.expected_python)
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [str(args.expected_python), "-s", "-B", "-m", "tools.cpu_o1.q1_t0_real",
        "--entry-smoke-child", "--expected-python", str(args.expected_python)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.bench_root.resolve() / "Filament_python")
    env["NUMBA_NUM_THREADS"] = "1"
    started = time.perf_counter()
    with (out / "stdout.log").open("xb") as so, (out / "stderr.log").open("xb") as se:
        result = subprocess.run(command, env=env, cwd=args.bench_root, stdout=so, stderr=se, timeout=60)
    record = {"command": command, "returncode": result.returncode,
        "wall_seconds": time.perf_counter()-started, "real_input_used": False,
        "scope": "ACTUAL_PARENT_CHILD_ENTRY_ONLY"}
    _write_new_json(out / "ENTRY_SMOKE.json", record)
    if result.returncode:
        raise RuntimeError("entry smoke child failed")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bench-root", "provenance", "input-root", "vendor-dir", "output-root",
                 "storage-root", "expected-source-binding", "expected-python", "case-root",
                 "baseline-boundary-root"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--storage-cap-bytes", type=int)
    parser.add_argument("--max-files", type=int, default=50000)
    parser.add_argument("--identity", choices=IDENTITIES)
    mode = parser.add_mutually_exclusive_group()
    for name in ("child", "dry-run", "entry-smoke", "entry-smoke-child"):
        mode.add_argument("--"+name, action="store_true")
    args = parser.parse_args(argv)
    if args.expected_python is None:
        parser.error("--expected-python is required")
    if args.entry_smoke_child:
        _check_expected_python(args.expected_python)
        from tools.cpu_o1 import q1_t0, numeric_gate
        print(json.dumps({"status": "ENTRY_SMOKE_CHILD_PASS", "python": sys.executable,
                          "pid": os.getpid(), "real_input_used": False}))
        return 0
    required = ("bench_root", "input_root", "vendor_dir", "expected_source_binding")
    required += (("case_root", "identity") if args.child else
        ("output_root", "storage_root", "storage_cap_bytes", "provenance"))
    for name in required:
        if getattr(args, name) is None:
            parser.error("missing --"+name.replace("_", "-"))
    if args.dry_run:
        _check_expected_python(args.expected_python)
        for name in ("bench_root", "input_root", "vendor_dir", "expected_source_binding", "provenance", "storage_root"):
            if not getattr(args, name).exists():
                raise FileNotFoundError(str(getattr(args, name)))
        if args.output_root.exists():
            raise FileExistsError(str(args.output_root))
        _validate_storage_caps(args.storage_cap_bytes, args.max_files)
        storage = _check_storage_growth(args.storage_root,
            growth_cap_bytes=args.storage_cap_bytes, max_files=args.max_files)
        print(json.dumps({"status": "DRY_RUN_ONLY", "real_input_used": False,
            "identity_order": IDENTITIES, "fft": EXPECTED_FFT_WORKERS,
            "kernel_threads": EXPECTED_KERNEL_THREADS, "storage_cap_bytes": args.storage_cap_bytes,
            "storage_inventory": storage}, allow_nan=False))
        return 0
    if args.entry_smoke:
        print(json.dumps(_entry_smoke(args), allow_nan=False))
        return 0
    if args.child:
        return _child_main(args)
    try:
        result = _run_coordinator(args)
        return 0 if result["status"] == "Q1_COMPLETE_FOUR_IDENTITY_SCREENING" else 3
    except BaseException as exc:
        # Admission failures precede the scientific run; preserve outside its root.
        failure = {"status": "Q1_PREFLIGHT_BLOCKED", "error": type(exc).__name__+": "+str(exc),
                   "traceback": traceback.format_exc(), "next_action": "MANUAL_REVIEW_ONLY"}
        if args.storage_root and args.storage_root.is_dir():
            _write_new_json(args.storage_root / ("Q1_COORDINATOR_FAILURE_"+str(os.getpid())+".json"), failure)
        print(json.dumps(failure, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

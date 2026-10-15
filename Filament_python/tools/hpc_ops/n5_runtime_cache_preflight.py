#!/usr/bin/env python3
"""Fail-closed runtime cache validation and diagnostic-only CuPy Raman probe.

The probe evaluates one synthetic 384x2x2 Isaacs Raman stage. It never loads a
scientific source field or calls propagate_one_pulse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import stat
import sys
import tempfile
import time
from typing import Any

SCHEMA = "e5_n5.runtime_cache_preflight.v1"
ENV_NAMES = (
    "HOME",
    "XDG_CACHE_HOME",
    "CUPY_CACHE_DIR",
    "TMPDIR",
    "TMP",
    "TEMP",
)


def _canonical(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"path must be absolute: {path}")
    return path.resolve(strict=True)


def expected_environment(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root)
    if not root.is_absolute():
        raise ValueError("RUN_ROOT must be absolute")
    root_real = _canonical(root)
    if root_real != root:
        raise ValueError("RUN_ROOT must be canonical and contain no symlink components")
    entry = root.with_name(root.name + "_entry")
    runtime = entry / "runtime_env"
    return {
        "N5_ENTRY_ROOT": entry,
        "N5_RUNTIME_ENV_ROOT": runtime,
        "HOME": runtime / "home",
        "XDG_CACHE_HOME": runtime / "xdg_cache",
        "CUPY_CACHE_DIR": runtime / "cupy_cache",
        "TMPDIR": runtime / "tmp",
        "TMP": runtime / "tmp",
        "TEMP": runtime / "tmp",
    }


def _private_directory(path: Path, *, device_id: int) -> dict[str, Any]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"runtime path is not a non-symlink directory: {path}")
    if path.resolve(strict=True) != path:
        raise ValueError(f"runtime path does not resolve to its declared location: {path}")
    if int(info.st_dev) != int(device_id):
        raise ValueError(f"runtime path is on a different filesystem from RUN_ROOT: {path}")
    mode = stat.S_IMODE(info.st_mode)
    owner_uid = getattr(info, "st_uid", None)
    expected_uid = os.getuid() if hasattr(os, "getuid") else None
    if expected_uid is not None and owner_uid != expected_uid:
        raise ValueError(f"runtime directory is not owned by the current user: {path}")
    if os.name == "posix" and mode != 0o700:
        raise ValueError(f"runtime directory must have mode 700: {path} mode={mode:o}")
    return {
        "path": str(path),
        "mode": mode,
        "owner_uid": owner_uid,
        "device_id": int(info.st_dev),
    }


def _atomic_write_rename_probe(directory: Path) -> dict[str, Any]:
    nonce = secrets.token_hex(12)
    source = directory / (".n5-env-check-" + nonce + ".part")
    target = directory / (".n5-env-check-" + nonce + ".ready")
    payload = ("n5-runtime-check:" + nonce).encode("ascii")
    fd = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(source, target)
        with target.open("rb") as stream:
            actual = stream.read()
        if actual != payload:
            raise OSError(f"atomic rename/readback mismatch in {directory}")
        if os.name == "posix":
            dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        return {
            "directory": str(directory),
            "write": True,
            "fsync": True,
            "atomic_rename": True,
            "readback": True,
            "payload_sha256": hashlib.sha256(actual).hexdigest(),
        }
    finally:
        for candidate in (source, target):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def validate_runtime_environment(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    expected = expected_environment(root)
    if not root.is_dir():
        raise ValueError(f"RUN_ROOT must already exist: {root}")
    root_stat = root.stat()
    if stat.S_ISLNK(root.lstat().st_mode):
        raise ValueError("RUN_ROOT may not be a symlink")
    for key in ("N5_ENTRY_ROOT", "N5_RUNTIME_ENV_ROOT", *ENV_NAMES):
        actual = os.environ.get(key)
        expected_path = expected[key]
        if not actual:
            raise ValueError(f"required environment variable is unset: {key}")
        actual_path = Path(actual)
        if not actual_path.is_absolute():
            raise ValueError(f"environment path must be absolute: {key}")
        try:
            resolved = actual_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"environment path does not exist: {key}") from exc
        if actual_path != expected_path or resolved != expected_path:
            raise ValueError(
                f"{key} is not bound to the project runtime path; "
                f"expected={expected_path} actual={actual_path} resolved={resolved}"
            )

    checked = {
        "entry_root": _private_directory(expected["N5_ENTRY_ROOT"], device_id=root_stat.st_dev),
        "runtime_root": _private_directory(expected["N5_RUNTIME_ENV_ROOT"], device_id=root_stat.st_dev),
    }
    for name in ("HOME", "XDG_CACHE_HOME", "CUPY_CACHE_DIR", "TMPDIR"):
        checked[name] = _private_directory(expected[name], device_id=root_stat.st_dev)

    cwd = Path.cwd().resolve(strict=True)
    if cwd != root:
        raise ValueError(f"process cwd must be RUN_ROOT: expected={root} actual={cwd}")
    tempfile.tempdir = None
    tempdir = Path(tempfile.gettempdir()).resolve(strict=True)
    if tempdir != expected["TMPDIR"]:
        raise ValueError(f"Python temporary directory escaped the runtime contract: {tempdir}")
    if os.name == "posix" and Path.home().resolve(strict=True) != expected["HOME"]:
        raise ValueError("Python home directory escaped the runtime contract")

    probes = []
    for name in ("HOME", "XDG_CACHE_HOME", "CUPY_CACHE_DIR", "TMPDIR"):
        probes.append(_atomic_write_rename_probe(expected[name]))
    return {
        "run_root": str(root),
        "entry_root": str(expected["N5_ENTRY_ROOT"]),
        "runtime_root": str(expected["N5_RUNTIME_ENV_ROOT"]),
        "paths": {name: str(expected[name]) for name in ENV_NAMES},
        "cwd": str(cwd),
        "python_tempdir": str(tempdir),
        "same_filesystem_as_run_root": True,
        "directories": checked,
        "write_rename_probes": probes,
    }


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_inventory(cache_root: str | Path) -> dict[str, Any]:
    root = Path(cache_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"CuPy cache root must be a non-symlink directory: {root}")
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    for base, dirnames, filenames in os.walk(root, followlinks=False):
        base_path = Path(base)
        dirnames.sort()
        filenames.sort()
        for name in list(dirnames):
            path = base_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"non-directory or symlink in CuPy cache: {path}")
            directories.append(path.relative_to(root).as_posix())
        for name in filenames:
            path = base_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError(f"non-regular or symlink artifact in CuPy cache: {path}")
            digest = _raw_sha256(path)
            after = path.stat()
            if int(after.st_size) != int(info.st_size) or int(after.st_nlink) != int(info.st_nlink):
                raise ValueError(f"CuPy cache artifact changed while being inventoried: {path}")
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": int(after.st_size),
                    "nlink": int(after.st_nlink),
                    "sha256": digest,
                }
            )
    files.sort(key=lambda row: row["path"])
    directories.sort()
    canonical = json.dumps(
        {"directories": directories, "files": files},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "root": str(root.resolve(strict=True)),
        "directories": directories,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["size_bytes"] for row in files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _cache_signature(inventory: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(inventory["directories"]),
        tuple(
            (row["path"], row["size_bytes"], row["nlink"], row["sha256"])
            for row in inventory["files"]
        ),
    )


def validate_phase_preconditions(
    phase: str,
    before: dict[str, Any],
    *,
    cold_receipt: dict[str, Any] | None = None,
    current_pid: int | None = None,
    config_sha256: str | None = None,
    environment_paths: dict[str, str] | None = None,
    cuda_visible_devices: str | None = None,
) -> None:
    if phase == "cold":
        if before["file_count"] or before["directories"]:
            raise ValueError("cold CuPy cache must be empty before first compilation")
        return
    if phase != "warm":
        raise ValueError("phase must be cold or warm")
    if not isinstance(cold_receipt, dict) or cold_receipt.get("status") != "PASS":
        raise ValueError("warm phase requires a passing cold-process receipt")
    if cold_receipt.get("phase") != "cold":
        raise ValueError("warm receipt is not bound to a cold phase")
    if current_pid is not None and int(cold_receipt["process"]["pid"]) == int(current_pid):
        raise ValueError("warm phase must use a fresh process")
    if config_sha256 != cold_receipt.get("config_sha256"):
        raise ValueError("warm phase config differs from the cold phase")
    if environment_paths != cold_receipt["runtime_environment"]["paths"]:
        raise ValueError("warm phase runtime paths differ from cold phase")
    if cuda_visible_devices != cold_receipt["cuda_visible_devices"]:
        raise ValueError("warm phase CUDA-visible token differs from cold phase")
    cold_after = cold_receipt["cache_inventory_after"]
    if not cold_after["file_count"]:
        raise ValueError("cold phase produced no CuPy cache artifacts")
    if _cache_signature(before) != _cache_signature(cold_after):
        raise ValueError("warm phase did not start from the cold cache inventory")


def validate_phase_postconditions(
    phase: str,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    cold_receipt: dict[str, Any] | None = None,
) -> None:
    if phase == "cold":
        if not after["file_count"]:
            raise ValueError("cold Raman compile produced no CuPy cache artifacts")
        return
    if phase == "warm":
        if _cache_signature(before) != _cache_signature(after):
            raise ValueError("warm process changed the CuPy cache inventory")
        if cold_receipt is None or _cache_signature(after) != _cache_signature(
            cold_receipt["cache_inventory_after"]
        ):
            raise ValueError("warm cache differs from the cold-process receipt")
        return
    raise ValueError("phase must be cold or warm")


def _process_metadata() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd().resolve(strict=True)),
        "interpreter": str(Path(sys.executable).resolve(strict=True)),
        "started_unix_s": time.time(),
    }


def _load_cold_receipt(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    receipt_path = Path(path).resolve(strict=True)
    with receipt_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("cold receipt must be a JSON object")
    return data


def _load_formal_config(config_path: Path) -> tuple[Any, Any]:
    # confio imports constants/hr4, which intentionally import device/CuPy.
    # The caller must establish the cache boundary before entering this chain.
    from KHz_filament.confio import load_all

    loaded = load_all(str(config_path))
    grid, beam, _prop, _ion, _heat, _run, raman = loaded
    if int(grid.Nt) != 384:
        raise ValueError(f"formal N5 Raman compile probe requires Nt=384, got {grid.Nt}")
    if str(getattr(raman, "model", "")).lower() != "isaacs_rot_sinexp":
        raise ValueError("formal N5 Raman compile probe requires isaacs_rot_sinexp")
    if str(getattr(raman, "method", "iir")).lower() != "iir":
        raise ValueError("formal N5 Raman compile probe requires IIR")
    if str(getattr(raman, "iir_sampling", "")).lower() != "exact_piecewise_linear":
        raise ValueError("formal N5 Raman compile probe requires exact_piecewise_linear")
    for name in ("lam0", "n0"):
        value = float(getattr(beam, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"formal beam config has invalid {name}")
    for name in ("n_R", "omega_R", "Gamma_R"):
        value = float(getattr(raman, name))
        if not math.isfinite(value):
            raise ValueError(f"formal Raman config has nonfinite {name}")
    if float(raman.n_R) <= 0 or float(raman.omega_R) <= 0 or float(raman.Gamma_R) < 0:
        raise ValueError("formal Raman coefficients are outside the existing operator domain")
    return grid, (beam, raman)


def run_raman_probe(
    *,
    phase: str,
    run_root: Path,
    config_path: Path,
    output_path: Path,
    cold_receipt_path: str | None,
) -> dict[str, Any]:
    runtime_environment = validate_runtime_environment(run_root)
    config_path = config_path.resolve(strict=True)
    config_sha256 = _raw_sha256(config_path)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = visible.split(",")
    if os.environ.get("UPPE_USE_GPU") != "1":
        raise RuntimeError("Raman probe requires UPPE_USE_GPU=1")
    if len(tokens) != 1 or not tokens[0] or tokens[0].strip() != tokens[0] or tokens[0] == "-1":
        raise RuntimeError("Raman probe requires exactly one valid CUDA_VISIBLE_DEVICES token")

    cache_root = Path(os.environ["CUPY_CACHE_DIR"])
    before = cache_inventory(cache_root)
    cold_receipt = _load_cold_receipt(cold_receipt_path)
    validate_phase_preconditions(
        phase,
        before,
        cold_receipt=cold_receipt,
        current_pid=os.getpid(),
        config_sha256=config_sha256,
        environment_paths=runtime_environment["paths"],
        cuda_visible_devices=visible,
    )

    started = time.perf_counter()
    # Config loading follows the same import path as formal workers, but only
    # after inherited paths and the cold/warm cache precondition are proven.
    grid, (beam, raman) = _load_formal_config(config_path)
    from KHz_filament.hr4e5_gpu_bootstrap import probe_gpu_runtime

    gpu = probe_gpu_runtime()
    import cupy as cp
    from cupy.cuda.compiler import get_cache_dir

    if Path(get_cache_dir()).resolve(strict=True) != cache_root.resolve(strict=True):
        raise RuntimeError("CuPy compiler resolved a cache directory outside CUPY_CACHE_DIR")
    from KHz_filament.device import USE_GPU, xp
    from KHz_filament.grids import make_axes
    from KHz_filament.raman import isaacs_raman_stage
    from KHz_filament.constants import c0

    if not USE_GPU or xp is not cp:
        raise RuntimeError("KHz_filament.device silently selected a non-CuPy backend")

    axes = make_axes(2, 2, int(grid.Nt), float(grid.Lx), float(grid.Ly), float(grid.Twin))
    field = cp.full((int(grid.Nt), 2, 2), 1.0 + 0.125j, dtype=cp.complex128)
    omega0 = 2.0 * math.pi * float(c0) / float(beam.lam0)
    result = isaacs_raman_stage(
        field,
        Omega=axes.Omega,
        dt=axes.dt,
        omega0=omega0,
        n0=float(beam.n0),
        n_R=float(raman.n_R),
        omega_R=float(raman.omega_R),
        Gamma_R=float(raman.Gamma_R),
        method="iir",
        iir_sampling="exact_piecewise_linear",
        return_response=True,
        return_energy=True,
    )
    cp.cuda.runtime.deviceSynchronize()
    array_summary: dict[str, Any] = {}
    for key in ("rhs", "I_R", "u_R_signed", "q_R_positive"):
        value = result[key]
        if not bool(cp.isfinite(value).all().item()):
            raise FloatingPointError(f"Raman probe output is nonfinite: {key}")
        array_summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    scalar_summary = {
        key: float(result[key])
        for key in ("rhs_l2_norm", "IR_max", "walltime_s")
    }
    if not all(math.isfinite(value) for value in scalar_summary.values()):
        raise FloatingPointError("Raman probe returned a nonfinite scalar diagnostic")
    if int(result["convolution_count"]) != 1:
        raise RuntimeError("Raman probe did not execute exactly one convolution")
    cp.cuda.runtime.deviceSynchronize()
    if cold_receipt is not None and cold_receipt["gpu"]["gpu_uuid"] != gpu["gpu_uuid"]:
        raise RuntimeError("warm process did not reuse the same assigned GPU as cold process")
    after = cache_inventory(cache_root)
    validate_phase_postconditions(phase, before, after, cold_receipt=cold_receipt)

    record = {
        "schema": SCHEMA,
        "status": "PASS",
        "state": "completed",
        "kind": "cupy_raman_compile_reuse_probe",
        "phase": phase,
        "process": _process_metadata(),
        "cuda_visible_devices": visible,
        "gpu": gpu,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "runtime_environment": runtime_environment,
        "cupy_cache_directory": str(Path(get_cache_dir()).resolve(strict=True)),
        "cupy_version": str(cp.__version__),
        "formal_operator": {
            "model": "isaacs_rot_sinexp",
            "method": "iir",
            "iir_sampling": "exact_piecewise_linear",
            "Nt": int(grid.Nt),
            "space_shape": [2, 2],
            "field_dtype": str(field.dtype),
            "convolution_count": int(result["convolution_count"]),
        },
        "outputs": array_summary,
        "scalars": scalar_summary,
        "cache_inventory_before": before,
        "cache_inventory_after": after,
        "elapsed_s": float(time.perf_counter() - started),
        "propagation_called": False,
        "scientific_source_allocated": False,
    }
    if cold_receipt is not None:
        record["cold_process_pid"] = int(cold_receipt["process"]["pid"])
    return record


def write_receipt_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    if not output.is_absolute():
        raise ValueError("receipt output path must be absolute")
    parent = output.parent.resolve(strict=True)
    if output.parent.resolve(strict=True) != parent or output.parent.is_symlink():
        raise ValueError("receipt parent must be a resolved non-symlink directory")
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "posix":
            dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--run-root", required=True)
    validate.add_argument("--output", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--phase", choices=("cold", "warm"), required=True)
    probe.add_argument("--run-root", required=True)
    probe.add_argument("--config", required=True)
    probe.add_argument("--output", required=True)
    probe.add_argument("--cold-receipt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output)
    try:
        if args.command == "validate":
            environment = validate_runtime_environment(args.run_root)
            payload = {
                "schema": SCHEMA,
                "status": "PASS",
                "state": "completed",
                "kind": "runtime_environment",
                "process": _process_metadata(),
                "runtime_environment": environment,
                "cupy_imported": False,
            }
        else:
            payload = run_raman_probe(
                phase=args.phase,
                run_root=Path(args.run_root),
                config_path=Path(args.config),
                output_path=output,
                cold_receipt_path=args.cold_receipt,
            )
        write_receipt_exclusive(output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL",
            "state": "failed",
            "command": args.command,
            "phase": getattr(args, "phase", None),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "process": _process_metadata(),
            "propagation_called": False,
            "scientific_source_allocated": False,
        }
        try:
            write_receipt_exclusive(output, failure)
        except Exception:
            pass
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
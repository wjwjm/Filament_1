"""Single-thread Linux ABI/runtime probe for the CPU-Q1 diagnostic only."""
from __future__ import annotations

import argparse
import ctypes.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import sysconfig
import tempfile
import traceback


EXPECTED_VERSIONS = {
    "numpy": "2.2.5",
    "scipy": "1.15.2",
    "numba": "0.61.2",
    "llvmlite": "0.44.0",
    "pytest": "8.3.5",
    "threadpoolctl": "3.6.0",
    "psutil": "7.0.0",
}


def _loaded_shared_objects() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and ".so" in fields[5]:
            paths.add(fields[5].removesuffix(" (deleted)"))
    return sorted(paths)


def _version_map() -> dict[str, str | None]:
    versions = {}
    for package in EXPECTED_VERSIONS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _compile_and_initialize_workqueue(np, numba) -> dict:
    @numba.njit(parallel=True, cache=False, fastmath=False)
    def probe(values):
        result = np.empty_like(values)
        for index in numba.prange(values.size):
            result[index] = values[index] * 2.0 + 1.0
        return result

    numba.set_num_threads(1)
    source = np.arange(16, dtype=np.float64)
    actual = probe(source)
    expected = source * 2.0 + 1.0
    return {
        "active_threading_layer": str(numba.threading_layer()),
        "threads_current": int(numba.get_num_threads()),
        "probe_shape": list(actual.shape),
        "probe_dtype": str(actual.dtype),
        "probe_exact": bool(np.array_equal(actual, expected)),
    }


def run_linux_q1_preflight() -> dict:
    """Inspect package/ABI/backend identity and initialize actual workqueue."""
    import numpy as np
    import scipy
    import numba
    import psutil
    import threadpoolctl
    from scipy import fft as scipy_fft

    native_thread_limit = threadpoolctl.threadpool_limits(limits=1)
    versions = _version_map()
    reasons = []
    if platform.system() != "Linux":
        reasons.append("platform_not_linux")
    if platform.python_implementation() != "CPython" or platform.python_version() != "3.12.8":
        reasons.append("python_identity_mismatch")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        reasons.append("machine_not_x86_64")
    if sys.byteorder != "little":
        reasons.append("byteorder_not_little")
    if versions != EXPECTED_VERSIONS:
        reasons.append("locked_package_versions_mismatch")

    libc_name, libc_version = platform.libc_ver()
    if libc_name.lower() != "glibc" or not libc_version:
        reasons.append("glibc_identity_unavailable")
    soabi = sysconfig.get_config_var("SOABI")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not soabi or not ext_suffix:
        reasons.append("python_abi_identity_unavailable")

    target = str(numba.config.CPU_NAME)
    target_env = os.environ.get("NUMBA_CPU_NAME")
    layer_config = str(numba.config.THREADING_LAYER)
    layer_env = os.environ.get("NUMBA_THREADING_LAYER")
    if target != "generic" or target_env != "generic":
        reasons.append("numba_target_not_generic")
    if layer_config != "workqueue" or layer_env != "workqueue":
        reasons.append("numba_layer_not_configured_workqueue")

    cache_text = os.environ.get("NUMBA_CACHE_DIR")
    cache_path = Path(cache_text).expanduser() if cache_text else None
    cache_probe = {"path": str(cache_path) if cache_path is not None else None,
                   "writable": False, "error": None}
    if cache_path is not None:
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".cpu_q1_write_probe_",
                                             dir=cache_path, delete=True):
                pass
            cache_probe["writable"] = True
        except OSError as exc:
            cache_probe["error"] = f"{type(exc).__name__}: {exc}"
    if not cache_probe["writable"]:
        reasons.append("numba_cache_not_writable")

    fft_input = np.arange(16, dtype=np.complex128)
    numpy_fft = np.fft.fft(fft_input)
    scipy_fft_value = scipy_fft.fft(fft_input, workers=1)
    fft_smoke_finite = bool(np.isfinite(numpy_fft).all() and np.isfinite(scipy_fft_value).all())
    fft_smoke_max_abs_value = float(np.max(
        np.abs(numpy_fft - scipy_fft_value), initial=0.0))
    fft_smoke_max_abs = (fft_smoke_max_abs_value
                         if math.isfinite(fft_smoke_max_abs_value) else None)
    if not fft_smoke_finite:
        reasons.append("fft_smoke_nonfinite_output")

    workqueue = _compile_and_initialize_workqueue(np, numba)
    if workqueue["active_threading_layer"] != "workqueue":
        reasons.append("numba_active_workqueue_not_proven")
    pools = threadpoolctl.threadpool_info()
    if any(int(row.get("num_threads", 1)) > 1 for row in pools):
        reasons.append("native_threadpool_exceeds_single_thread_preflight")

    return {
        "schema": "filament.cpu_q1.linux_preflight.v1",
        "status": "LINUX_Q1_PREFLIGHT_PASS" if not reasons else "LINUX_Q1_PREFLIGHT_BLOCKED",
        "eligible": not reasons,
        "reasons": reasons,
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "soabi": soabi, "ext_suffix": ext_suffix},
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "byteorder": sys.byteorder,
                     "libc_name": libc_name, "libc_version": libc_version},
        "versions": versions,
        "expected_versions": EXPECTED_VERSIONS,
        "numba": {"cpu_target": target, "cpu_target_env": target_env,
                  "threading_layer_config": layer_config,
                  "threading_layer_env": layer_env,
                  "threading_layer_priority": list(numba.config.THREADING_LAYER_PRIORITY),
                  "workqueue_probe": workqueue},
        "native_threadpools": pools,
        "blas_and_runtime_libraries": sorted({
            str(row.get("filepath")) for row in pools if row.get("filepath")
        }),
        "loaded_shared_objects": _loaded_shared_objects(),
        "system_library_resolution": {
            "libm": ctypes.util.find_library("m"),
            "gomp": ctypes.util.find_library("gomp"),
            "openblas": ctypes.util.find_library("openblas"),
        },
        "numba_cache_probe": cache_probe,
        "fft_smoke": {"numpy_dtype": str(numpy_fft.dtype),
                      "scipy_dtype": str(scipy_fft_value.dtype),
                      "workers": 1, "finite": fft_smoke_finite,
                      "max_abs": fft_smoke_max_abs,
                      "max_abs_reason": (None if fft_smoke_max_abs is not None
                                         else "nonfinite_cross_backend_difference"),
                      "cross_backend_exactness_is_qualification_gate": False},
        "preflight_call_organization": (
            "single-thread compile probe; Q1 windows run sequentially with one active child"
        ),
        "production_qualification": "NOT_PRODUCTION_QUALIFIED",
    }


def _write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="new, non-overwriting JSON receipt path")
    args = parser.parse_args(argv)
    try:
        receipt = run_linux_q1_preflight()
    except BaseException as exc:
        receipt = {
            "schema": "filament.cpu_q1.linux_preflight.v1",
            "status": "LINUX_Q1_PREFLIGHT_TOOL_FAILURE",
            "eligible": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "production_qualification": "NOT_PRODUCTION_QUALIFIED",
        }
    _write_new_json(args.evidence, receipt)
    return 0 if receipt.get("eligible") else 1


if __name__ == "__main__":
    raise SystemExit(main())

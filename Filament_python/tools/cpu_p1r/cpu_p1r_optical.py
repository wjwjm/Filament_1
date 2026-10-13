#!/usr/bin/env python3
"""Eight-interval CPU optical diagnostic; never production-qualified."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

_INITIAL_GPU = os.environ.get("UPPE_USE_GPU")
os.environ["UPPE_USE_GPU"] = "0"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from tools.cpu_p1 import cpu_p1_fft as common
from tools.hpc_ops.provenance_v2 import raw_sha256_file, canonical_lf_bytes, canonical_lf_sha256, _git_blob_oid

SCHEMA = "filament.cpu_p1r.optical_diagnostic.v1"
CLASSIFICATION = "P1R_DIAGNOSTIC_ONLY"
QUALIFICATION = "NOT_PRODUCTION_QUALIFIED"
FIXED_PYTHON = "/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python"
MAPPING_SHA256 = "4ea5c7107d28b05f03518610ddb58a381ccf70869b50039ee1c510a0e1a929eb"
AUDIT_SHA256 = "b8840a2d584c5582bc1642b4d02c64ae083b2fe8655cdefdc11ced80a2be012a"
SOURCE_RAW_SHA256 = "3fcf21546fe61b0c9e3522816f7e03e004321b0ec91ddff1ab70b5c8a25309c7"
SOURCE_ARRAY_SHA256 = "52731713b0358c760af8605b7c7f39dd94df500722bdb4e299664c13c04af144"
PRE0_RAW_SHA256 = "b38a7a0efe9e43a10ae1520f063380828555228e4bf3efdacdda2434f21721c5"
PRE0_ARRAY_SHA256 = "66ec96c2babba7c9e277bd5e19bb5657fe7945f56062f077cc1853bd39c2dfbe"
PROPAGATE_BLOB = "f156111398415c2b92e4609ccc1da075fcf5d1e3"
PROPAGATE_CANONICAL_SHA256 = "909fbcf5ef05d2c50b8ed27506b450d40127b3973709d9b4b6a961799dc8d1f6"
INTERVAL_START_LINE = 628
INTERVAL_END_LINE = 1503
EXPECTED_SHAPE = (384, 351, 301)
EXPECTED_PRE0_SHAPE = (8, 351, 301)
MEMORY_LIMIT_BYTES = 320 * 1024**3
RSS_LIMIT_BYTES = int(MEMORY_LIMIT_BYTES * 0.80)
CASE_LIMIT_SECONDS = 2700
B0_LIMIT_SECONDS = 900
SLURM_FIELDS = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS",
                "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_QOS", "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST")


class OpticalError(RuntimeError):
    pass


def _jsonable(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def _json_new(path: Path, value) -> None:
    _write_new(path, (json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                                allow_nan=False, separators=(",", ":")) + "\n").encode())


def _append_event(path: Path, value) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False,
                                separators=(",", ":")) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _canonical_hash(array) -> str:
    from KHz_filament.hr4e5_evidence import sha256_array
    return str(sha256_array(array))


def _array_metrics(reference, candidate) -> dict:
    import numpy as np
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise OpticalError("comparison arrays have different shape/dtype or non-finite values")
    d = b - a
    max_abs = float(np.max(np.abs(d), initial=0.0))
    max_ref = float(np.max(np.abs(a), initial=0.0))
    tiny = np.finfo(np.float64).tiny
    return {"max_abs": max_abs, "max_rel": max_abs / max(max_ref, tiny),
            "l2_rel": float(np.linalg.norm(d.ravel()) / max(np.linalg.norm(a.ravel()), tiny))}


def verify_array_contract(array, *, shape, dtype, expected_hash=None, label="array") -> str:
    import numpy as np
    value = np.asarray(array)
    if value.shape != tuple(shape) or value.dtype != np.dtype(dtype) or not value.flags.c_contiguous:
        raise OpticalError(f"{label} shape/dtype/layout rejected: {value.shape}/{value.dtype}/C={value.flags.c_contiguous}")
    if not np.isfinite(value).all():
        raise OpticalError(f"{label} contains NaN/Inf")
    digest = _canonical_hash(value)
    if expected_hash is not None and digest != expected_hash:
        raise OpticalError(f"{label} canonical hash mismatch")
    return digest


def _safe_relative(root: Path, relative: str) -> Path:
    p = (root / relative).resolve(strict=False)
    if not p.is_relative_to(root.resolve(strict=True)):
        raise OpticalError(f"mapped path escapes input root: {relative}")
    return p


def verify_asset_mapping(input_root: Path) -> dict:
    import numpy as np
    mapping_path = input_root / "INPUT_TRANSFER_MAPPING.json"
    audit_path = input_root / "asset_audit_receipt.json"
    if raw_sha256_file(mapping_path) != MAPPING_SHA256 or raw_sha256_file(audit_path) != AUDIT_SHA256:
        raise OpticalError("optical transfer mapping/audit receipt hash mismatch")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    if mapping.get("schema") != "cpu_p1r.optical_input_transfer_mapping.v1" or mapping.get("hash_scope") != "classified_by_record":
        raise OpticalError("unsupported optical transfer mapping schema")
    found = {}
    for record in mapping.get("records", []):
        rel = record.get("relative")
        if not isinstance(rel, str) or rel in found:
            raise OpticalError("missing or duplicate relative asset record")
        path = _safe_relative(input_root, rel)
        if not path.is_file() or raw_sha256_file(path) != record.get("raw_sha256"):
            raise OpticalError(f"mapped raw asset mismatch: {rel}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise OpticalError(f"mapped asset size mismatch: {rel}")
        found[rel] = record
    if len(found) != 11:
        raise OpticalError(f"expected 11 frozen optical assets, found {len(found)}")
    pre_rec = found.get("pre0_first_8.npy")
    pre0 = np.load(input_root / "pre0_first_8.npy", allow_pickle=False)
    pre_hash = verify_array_contract(pre0, shape=EXPECTED_PRE0_SHAPE, dtype="float64",
                                     expected_hash=PRE0_ARRAY_SHA256, label="PRE0 first eight")
    if pre_rec.get("canonical_array_sha256") != pre_hash:
        raise OpticalError("PRE0 mapping canonical hash mismatch")
    return {"mapping": mapping, "records": found, "pre0": pre0,
            "mapping_sha256": MAPPING_SHA256, "asset_audit_sha256": AUDIT_SHA256,
            "pre0_canonical_sha256": pre_hash}


def validate_frozen_source(input_root: Path, assets: dict):
    import numpy as np
    from KHz_filament.hr4e5_canonical_source import validate_canonical_source, _expected_seal
    config_path = input_root / "derived_n5_execution_config.json"
    manifest_path = input_root / "SOURCE_MANIFEST.json"
    identity_path = input_root / "N5_FORMAL_ADMISSION_IDENTITY.json"
    seal_path = input_root / "E_source_canonical.sealed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    original = manifest["canonical_source"]
    if manifest.get("canonical_array_hash") != SOURCE_ARRAY_SHA256:
        raise OpticalError("frozen source manifest canonical hash mismatch")
    if original.get("file_sha256") != SOURCE_RAW_SHA256 or original.get("canonical_array_hash") != SOURCE_ARRAY_SHA256:
        raise OpticalError("frozen source manifest file/array identity mismatch")
    if identity.get("identity_sha256") is None or identity.get("source_identity", {}).get("canonical_source") != original:
        raise OpticalError("formal admission identity differs from frozen source manifest")
    mapped = dict(original)
    mapped["artifact_path"] = str(input_root / "E_source_canonical.npy")
    mapped["sealed_manifest_path"] = str(seal_path)
    validated = validate_canonical_source(mapped, config_path=config_path,
                                          source_manifest_path=manifest_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected = _expected_seal(original, identity,
                              {"identity_sha256": identity["implementation_identity_sha256"]})
    if any(seal.get(k) != v for k, v in expected.items() if k != "created_utc"):
        raise OpticalError("frozen source seal differs from source/admission identity")
    if seal.get("created_utc") is None or seal.get("admission_identity_sha256") != identity["identity_sha256"]:
        raise OpticalError("frozen source seal admission binding mismatch")
    source = np.load(input_root / "E_source_canonical.npy", allow_pickle=False)
    source_hash = verify_array_contract(source, shape=EXPECTED_SHAPE, dtype="complex128",
                                        expected_hash=SOURCE_ARRAY_SHA256, label="sealed E_source")
    if not source.flags.c_contiguous:
        raise OpticalError("sealed E_source is not C contiguous")
    return np.array(source, dtype=np.complex128, order="C", copy=True), source_hash, validated


class DiagnosticPREView:
    """In-memory copy of the formal read/update/post_fields contract."""
    def __init__(self, fields):
        import numpy as np
        self.shape = tuple(fields["delta_n"].shape[1:])
        self.fields = {k: np.array(v, dtype=np.float64, order="C", copy=True)
                       for k, v in fields.items()}
        self._read, self._updated, self._after = set(), set(), {}
        if any(np.signbit(self.fields[k]).any() for k in ("vx", "vy")):
            raise OpticalError("PRE0 velocities must use the sealed positive-zero recipe")
        if any(not np.equal(self.fields[k], 0.0).all() for k in ("vx", "vy")):
            raise OpticalError("PRE0 diagnostic velocities must be exact positive float64 zero")

    def read_interval(self, index):
        if index in self._read or index < 0 or index >= self.fields["delta_n"].shape[0]:
            raise ValueError("PRE interval must be read exactly once")
        self._read.add(index)
        return self.fields["delta_n"][index].copy()

    def update_interval(self, index, increment):
        import numpy as np
        if index not in self._read or index in self._updated:
            raise ValueError("PRE interval update requires one prior read")
        inc = np.asarray(increment, dtype=np.float64)
        before = self.fields["delta_n"][index]
        if inc.shape != self.shape or not np.isfinite(inc).all():
            raise ValueError("HR-3B increment shape/value mismatch")
        after = np.asarray(before + inc, dtype=np.float64)
        if not np.isfinite(after).all(): raise ValueError("POST is non-finite")
        self._after[index] = after.copy(); self._updated.add(index)
        return after

    def post_fields(self, index):
        import numpy as np
        if index not in self._updated: raise ValueError("POST requested before update")
        return {"delta_n": self._after[index].copy(),
                "vx": self.fields["vx"][index].copy(), "vy": self.fields["vy"][index].copy()}

    @property
    def complete(self):
        return len(self._read) == self.fields["delta_n"].shape[0] and self._updated == self._read

    def metadata(self):
        return {"hr3b_state_schema": "filament.cpu_p1r.diagnostic_pre_memory.v1",
            "hr3b_state_filename": "", "hr3b_state_dtype": "float64",
            "hr3b_state_shape": tuple(self.fields["delta_n"].shape),
            "hr3b_state_interval_centered": True, "hr3b_state_disk_backed": False,
            "schema": "filament.cpu_p1r.diagnostic_pre_memory.v1",
            "diagnostic_only": True, "state_origin": "sealed_pre0_first_8",
            "storage": "in_memory_copy", "disk_backed": False,
            "interval_centered": True, "shape": list(self.fields["delta_n"].shape),
            "dtype": "float64",
            "field_canonical_sha256": {key: _canonical_hash(value)
                for key, value in self.fields.items()}}

    def close(self): pass


class IntervalObserver:
    """sys.monitoring observer on the frozen loop-start and accepted-boundary lines."""
    def __init__(self, code, progress_path: Path, output_dir: Path, pre_view,
                 *, canonical_hash, b0_limit=B0_LIMIT_SECONDS):
        self.code, self.progress_path, self.output_dir = code, progress_path, output_dir
        self.pre_view, self.canonical_hash, self.b0_limit = pre_view, canonical_hash, b0_limit
        self.rows, self.starts, self.cpu_starts = [], {}, {}
        self.tool_id = None

    def _callback(self, code, line):
        frame = sys._getframe(1)
        while frame is not None and frame.f_code is not code:
            frame = frame.f_back
        if frame is None:
            raise OpticalError("sys.monitoring callback could not recover propagate frame")
        if line == INTERVAL_START_LINE:
            idx = int(frame.f_locals["interval_index"])
            self.starts[idx] = time.perf_counter()
            self.cpu_starts[idx] = time.process_time()
            if idx == 0:
                _append_event(self.progress_path, {"event": "first_interval_started",
                    "interval_index": idx, "monotonic": self.starts[idx], "utc": time.time()})
            return None
        if line == INTERVAL_END_LINE:
            boundary = time.perf_counter()  # capture before copying, hashing, or file I/O
            cpu_boundary = time.process_time()
            loc = frame.f_locals
            idx = int(loc["interval_index"])
            if idx not in self.starts or idx != len(self.rows):
                raise OpticalError("observer interval sequence/start anchor mismatch")
            E = loc["E"]
            post = self.pre_view.post_fields(idx)
            row = {"interval_index": idx, "z_m": float(loc["z"]),
                "dz_m": float(loc["dz_used_z_list"][-1]),
                "interval_wall_seconds": boundary - self.starts[idx],
                "interval_process_cpu_seconds": cpu_boundary - self.cpu_starts[idx],
                "E_canonical_sha256": self.canonical_hash(E),
                "post_field_canonical_sha256": {k: self.canonical_hash(v) for k, v in post.items()},
                "U_z_J": float(loc["U_z_list"][-1]), "I_max_W_m2": float(loc["I_max_z_list"][-1]),
                "energy_after_linear_half2_J": _finite_float(loc["energy_after_linear_half2_z_list"][-1]),
                "rho_N2_max_m3": float(loc["rho_N2_max_z_list"][-1]),
                "rho_O2_max_m3": float(loc["rho_O2_max_z_list"][-1]),
                "linear_walltime_step_seconds": float(loc["linear_walltime_step_z_list"][-1]),
                "propagation_step_wall_seconds": float(loc["total_walltime_step_z_list"][-1]),
                "adaptive_rejection_count": int(loc["adaptive_rejection_count_z_list"][-1]),
                "safety_mode_trigger_count": int(loc["safety_mode_trigger_count_z_list"][-1]),
                "raman_operator_substep_count": int(loc["raman_operator_substep_count_z_list"][-1]),
                "raman_convolution_count": int(loc["raman_convolution_count_step_z_list"][-1]),
                "bk_nee_profile_halfstep_1": _jsonable(loc["linear_profile_halfstep_1_z_list"][-1]),
                "bk_nee_profile_halfstep_2": _jsonable(loc["linear_profile_halfstep_2_z_list"][-1]),
                "boundary_monotonic": boundary, "boundary_utc": time.time()}
            if idx == 0:
                _save_npy_new(self.output_dir / "first_complete_interval_E.npy", E)
                _save_npz_new(self.output_dir / "first_complete_interval_POST.npz", post)
                row["B0_status"] = "DONE" if boundary - self.starts[idx] <= self.b0_limit else "LIMIT_REACHED"
            row["observer_overhead_seconds"] = time.perf_counter() - boundary
            self.rows.append(row)
            _append_event(self.progress_path, {"event": "complete_interval", **row})
            if idx == 0 and row["B0_status"] != "DONE":
                raise TimeoutError("B0 first complete interval exceeded 900-second limit")
            return None
        return sys.monitoring.DISABLE

    def __enter__(self):
        mon = sys.monitoring
        for candidate in range(6):
            if mon.get_tool(candidate) is None:
                self.tool_id = candidate; break
        if self.tool_id is None: raise OpticalError("no free sys.monitoring tool ID")
        mon.use_tool_id(self.tool_id, "filament_cpu_p1r_optical")
        try:
            mon.register_callback(self.tool_id, mon.events.LINE, self._callback)
            mon.set_local_events(self.tool_id, self.code, mon.events.LINE)
        except BaseException:
            try:
                mon.set_local_events(self.tool_id, self.code, 0)
                mon.set_events(self.tool_id, 0)
                mon.register_callback(self.tool_id, mon.events.LINE, None)
                mon.restart_events()
            finally:
                mon.free_tool_id(self.tool_id)
                self.tool_id = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.tool_id is not None:
            mon = sys.monitoring
            mon.set_local_events(self.tool_id, self.code, 0)
            mon.set_events(self.tool_id, 0)
            mon.register_callback(self.tool_id, mon.events.LINE, None)
            mon.free_tool_id(self.tool_id)
            mon.restart_events()
            self.tool_id = None
        return False


def _npy_bytes(array) -> bytes:
    import io, numpy as np
    stream = io.BytesIO(); np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _npz_bytes(fields) -> bytes:
    import io, numpy as np
    stream = io.BytesIO(); np.savez(stream, **{k: np.asarray(v) for k, v in fields.items()})
    return stream.getvalue()


def _save_npy_new(path: Path, array) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, np.asarray(array), allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())


def _save_npz_new(path: Path, fields) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **{k: np.asarray(v) for k, v in fields.items()})
        stream.flush(); os.fsync(stream.fileno())


def _source_binding(bench_root: Path) -> dict:
    path = bench_root / "Filament_python" / "KHz_filament" / "propagate.py"
    raw = path.read_bytes()
    canonical = canonical_lf_bytes(raw)
    git_blob = _git_blob_oid(bench_root, "Filament_python/KHz_filament/propagate.py")
    canonical_sha = canonical_lf_sha256(raw)
    lines = canonical.decode("utf-8").splitlines()
    if (git_blob != PROPAGATE_BLOB or canonical_sha != PROPAGATE_CANONICAL_SHA256
            or lines[INTERVAL_START_LINE - 1].strip() != "_performance_sync(measure_performance)"
            or lines[INTERVAL_END_LINE - 1].strip() != "pe = int(getattr(p, \"progress_every_z\", 0) or 0)"):
        raise OpticalError("frozen propagate.py blob/hash/observer anchors differ")
    return {"path": str(path), "git_blob": git_blob,
            "canonical_lf_sha256": canonical_sha,
            "interval_start_line": INTERVAL_START_LINE,
            "accepted_boundary_line": INTERVAL_END_LINE,
            "accepted_boundary_anchor": lines[INTERVAL_END_LINE - 1].strip()}


def _verify_lut(input_root: Path, assets: dict, ion, beam) -> dict:
    from KHz_filament.constants import c0
    from KHz_filament.ionization.lut import (_canonical_table_metadata,
        _ion_rate_table_defaults, _load_table_npz, _table_path, _table_signature)
    from KHz_filament.ionization.runtime import _resolve_rate
    lut_manifest = json.loads((input_root / "LUT_MANIFEST.json").read_text(encoding="utf-8"))
    if lut_manifest.get("schema") != "khz_filament.hr4e5.e5_1b.formal_lut.v1" or lut_manifest.get("mode") != "RATE_TABLE":
        raise OpticalError("frozen LUT manifest schema/mode mismatch")
    table_cfg = _ion_rate_table_defaults(ion)
    table_cfg["cache_dir"] = str(input_root / "cache" / "rate_tables")
    if not table_cfg.get("enabled") or not table_cfg.get("reuse_cache") or table_cfg.get("force_rebuild"):
        raise OpticalError("frozen LUT config does not permit read-only cache reuse")
    omega0 = 2.0 * 3.141592653589793 * c0 / float(beam.lam0)
    table_receipts = {}
    records_by_species = {}
    for species_params in ion.species:
        params = dict(species_params)
        rate = _resolve_rate(params, ion)
        params["rate"] = rate
        if rate not in ("ppt_talebpour_i_lut", "popruzhenko_atom_i_lut"):
            continue
        reference_model = str(params.get("reference_model", "")).lower()
        if not reference_model:
            reference_model = ("ppt_talebpour_i_full_reference" if rate == "ppt_talebpour_i_lut"
                               else "popruzhenko_atom_i_full_reference")
        metadata = _canonical_table_metadata(reference_model, str(params.get("name", "species")),
            params, omega0, float(beam.n0), table_cfg)
        signature = _table_signature(metadata)
        expected_name = Path(_table_path(table_cfg["cache_dir"], metadata["species_name"], signature)).name
        rec = next((r for r in lut_manifest.get("files", [])
                    if Path(r.get("path", "")).name == expected_name), None)
        if rec is None:
            raise OpticalError(f"frozen LUT manifest lacks configured cache file: {expected_name}")
        rel = next((name for name in assets["records"] if Path(name).name == expected_name), None)
        if rel is None:
            raise OpticalError("LUT manifest table is absent from transfer mapping")
        table = _load_table_npz(str(input_root / rel))
        species = str(table["species_name"])
        if (species != metadata["species_name"] or table["metadata"] != metadata
                or _table_signature(table["metadata"]) != signature
                or signature != lut_manifest.get("signature", {}).get(species)):
            raise OpticalError(f"frozen LUT cache is not an exact configured cache hit: {species}")
        records_by_species[species] = rec
        table_receipts[species] = {"relative_path": rel, "signature": signature,
            "array_names": rec.get("array_names"),
            "I_grid_shape": list(table["I_grid"].shape), "W_grid_shape": list(table["W_grid"].shape),
            "dtype_I_grid": str(table["I_grid"].dtype), "dtype_W_grid": str(table["W_grid"].dtype)}
    if set(table_receipts) != {"N2", "O2"}:
        raise OpticalError("frozen LUT cache does not contain N2 and O2")
    return {"manifest_schema": lut_manifest["schema"], "mode": lut_manifest["mode"],
            "config_sha256": lut_manifest.get("config_sha256"), "tables": table_receipts}


def _check_allocation() -> dict:
    required = {"SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "96", "SLURM_JOB_PARTITION": "v6_384",
        "SLURM_JOB_ACCOUNT": "bscc-t", "SLURM_JOB_QOS": "normal",
        "SLURM_MEM_PER_NODE": "327680", "UPPE_USE_GPU": "0"}
    bad = {k: {"expected": v, "actual": os.environ.get(k)}
           for k, v in required.items() if os.environ.get(k) != v}
    gpu_keys = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")
    gpu = {k: os.environ[k] for k in gpu_keys if os.environ.get(k) not in (None, "", "0")}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _INITIAL_GPU not in (None, "0", "false", "False", "") or gpu or (visible and visible not in ("NoDevFiles", "-1")):
        bad["gpu_allocation"] = {"requested_before_force": _INITIAL_GPU, **gpu,
                                  **({"CUDA_VISIBLE_DEVICES": visible} if visible else {})}
    if not hasattr(os, "sched_getaffinity"):
        bad["affinity"] = "Linux sched affinity required"
    else:
        allowed = sorted(os.sched_getaffinity(0))
        topology = common.cpu_topology(allowed)
        if (len(allowed) != 96 or topology.get("physical_core_count") != 96
                or not common.physical_topology_verified(topology)):
            bad["physical_core_topology"] = topology
    if bad:
        raise OpticalError("CPU optical allocation rejected: " + json.dumps(bad, sort_keys=True))
    return {k: os.environ.get(k) for k in (*SLURM_FIELDS, "CUDA_VISIBLE_DEVICES") if os.environ.get(k)}


def _environment_receipt(output_root: Path, *, vendor_dir: Path, allocation: dict) -> dict:
    import numpy as np
    import threadpoolctl
    if sys.executable != FIXED_PYTHON:
        raise OpticalError(f"fixed Python interpreter required: {FIXED_PYTHON}; got {sys.executable}")
    if getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise OpticalError("frozen vendor threadpoolctl 3.6.0 is required")
    pools = threadpoolctl.threadpool_info()
    if any(int(p.get("num_threads", 1)) > 1 for p in pools):
        raise OpticalError("thread pool exceeds the single-thread diagnostic limit")
    topo = common.cpu_topology(sorted(os.sched_getaffinity(0)))
    lscpu = common.optional_command("lscpu", [])
    numa = common.optional_command("numactl", ["--hardware"])
    try:
        import numpy as np
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer): np.__config__.show()
        numpy_config = buffer.getvalue()
    except Exception as exc:
        numpy_config = f"UNAVAILABLE: {type(exc).__name__}: {exc}\n"
    _write_new(output_root / "lscpu.txt", lscpu.get("stdout", "UNAVAILABLE\n").encode())
    _write_new(output_root / "numactl_hardware.txt", numa.get("stdout", "UNAVAILABLE\n").encode())
    _write_new(output_root / "numpy_config.txt", numpy_config.encode())
    for name in ("/proc/self/status", "/proc/cpuinfo", "/proc/meminfo"):
        path = Path(name)
        _write_new(output_root / (path.name + ".txt"), path.read_bytes() if path.is_file() else b"UNAVAILABLE\n")
    receipt = {"schema": SCHEMA, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION, "allocation": allocation,
        "python_executable": sys.executable, "python_version": sys.version,
        "numpy_version": np.__version__, "threadpoolctl_version": threadpoolctl.__version__,
        "threadpool_info": pools, "affinity_topology": topo,
        "lscpu_status": lscpu.get("status"), "numactl_status": numa.get("status"),
        "vendor_dir": str(vendor_dir), "backend": "NUMPY_CPU", "gpu": False}
    _json_new(output_root / "environment.json", receipt)
    return receipt


def _diagnostic_closure(diag: dict) -> dict:
    import numpy as np
    from KHz_filament.diagnostics import validate_nonlinear_diagnostics
    checked = validate_nonlinear_diagnostics(diag)
    statuses = {}
    for key, value in diag.items():
        if key.endswith("_closure_status") or key.startswith("hr3b_") or key == "deposition_level1_all_available_mechanism_closure_pass":
            statuses[key] = _jsonable(value)
    for key in ("deposition_ion_level1_closure_status", "deposition_ib_level1_closure_status",
                "deposition_raman_level1_closure_status", "deposition_raman_operator_energy_closure_status",
                "deposition_ion_level2_closure_status", "deposition_ib_level2_closure_status",
                "deposition_raman_level2_closure_status", "E_dep_total_level2_closure_status"):
        if key not in diag: continue
        vals = [str(v).upper() for v in np.asarray(diag[key]).reshape(-1).tolist() if v is not None]
        if any("FAIL" in v or "MISMATCH" in v for v in vals):
            raise OpticalError(f"existing scientific closure reports failure: {key}")
    all_pass = diag.get("deposition_level1_all_available_mechanism_closure_pass")
    if all_pass is not None:
        values = np.asarray(all_pass).reshape(-1).tolist()
        if not all(bool(v) for v in values):
            raise OpticalError("existing all-available deposition closure reports failure")
    if not bool(diag.get("hr3b_authoritative", False)):
        raise OpticalError("existing HR-3B authoritative source gate failed")
    for key in ("hr3b_source_authority_status", "hr3b_mapping_status", "hr3b_sign_status", "hr3b_thermodynamic_status"):
        if key in diag:
            vals = [str(v).upper() for v in np.asarray(diag[key]).reshape(-1).tolist()]
            if any(v not in ("PASS", "OK", "TRUE", "1") for v in vals):
                raise OpticalError(f"existing HR-3B diagnostic status failed: {key}")
    return {"validator": checked, "existing_statuses": statuses}


def _finite_float(value):
    import numpy as np
    result = float(value)
    return result if np.isfinite(result) else None


def _load_context(input_root: Path, bench_root: Path, vendor_dir: Path) -> dict:
    common.add_vendor(str(vendor_dir))
    import numpy as np
    from KHz_filament.device import USE_GPU, xp
    from KHz_filament.confio import load_all
    from KHz_filament.hr4e5_formal_entry import build_prefix_schedule
    from KHz_filament.longitudinal import build_longitudinal_schedule
    if USE_GPU or xp is not np:
        raise OpticalError("NumPy CPU backend required; GPU backend is disabled for this task")
    source_binding = _source_binding(bench_root)
    assets = verify_asset_mapping(input_root)
    source, source_hash, source_validation = validate_frozen_source(input_root, assets)
    grid, beam, prop, ion, heat, run, raman = load_all(str(input_root / "derived_n5_execution_config.json"))
    if (grid.Nt, grid.Ny, grid.Nx) != EXPECTED_SHAPE:
        raise OpticalError("frozen optical config grid differs from [384,351,301]")
    if not heat.hr3b_enabled or heat.hr3c_enabled:
        raise OpticalError("frozen HR-3B optical diagnostic requires HR-3B enabled and HR-3C disabled")
    if str(prop.linear_model).lower() != "bk_nee":
        raise OpticalError("accepted frozen optical configuration is not the BK-NEE path")
    lut = _verify_lut(input_root, assets, ion, beam)
    ion.rate_table = dict(ion.rate_table or {})
    old_cache = ion.rate_table.get("cache_dir", "cache/rate_tables")
    ion.rate_table["cache_dir"] = str(input_root / "cache" / "rate_tables")
    if bool(ion.rate_table.get("force_rebuild", False)):
        raise OpticalError("frozen ionization LUT configuration requests rebuild")
    prop_original = {"measure_performance": bool(prop.measure_performance),
                     "diag_bk_nee_profile": bool(prop.diag_bk_nee_profile),
                     "lut_cache_dir": str(old_cache)}
    prop.measure_performance = True
    prop.diag_bk_nee_profile = True
    full_schedule = build_longitudinal_schedule(
        dz=float(prop.dz), z_max=float(prop.z_max), z_start=0.0,
        focus_window_step=bool(prop.focus_window_step),
        focus_center_m=float(prop.focus_center_m),
        focus_halfwidth_m=float(prop.focus_halfwidth_m), dz_focus=float(prop.dz_focus))
    identity = json.loads((input_root / "N5_FORMAL_ADMISSION_IDENTITY.json").read_text(encoding="utf-8"))
    frozen_schedule = identity["schedule_identity"]
    if frozen_schedule["recipe"] != "build_full_then_prefix" or frozen_schedule["source_indices"] != [0, 8047]:
        raise OpticalError("frozen N5 prefix recipe/source indices differ")
    if json.loads(json.dumps(full_schedule.as_metadata())) != frozen_schedule["full_metadata"]:
        raise OpticalError("config-derived full schedule differs from frozen identity")
    formal_schedule = build_prefix_schedule(full_schedule, 8048)
    if json.loads(json.dumps(formal_schedule.as_metadata())) != frozen_schedule["metadata"]:
        raise OpticalError("formal 8048 prefix differs from frozen identity")
    schedule = build_prefix_schedule(formal_schedule, 8)
    if schedule.n_intervals != 8 or not np.array_equal(np.asarray(schedule.z_edges), np.asarray(full_schedule.z_edges[:9])):
        raise OpticalError("eight-interval prefix differs from frozen full schedule")
    if tuple(source.shape) != EXPECTED_SHAPE or tuple(assets["pre0"].shape) != EXPECTED_PRE0_SHAPE:
        raise OpticalError("sealed source/PRE0 dimensions differ from frozen config")
    return {"assets": assets, "source": source, "source_hash": source_hash,
        "source_validation": source_validation, "source_binding": source_binding,
        "grid": grid, "beam": beam, "prop": prop, "ion": ion, "heat": heat,
        "run": run, "raman": raman, "schedule": schedule, "lut": lut,
        "runtime_config_delta": {"measure_performance": True,
            "diag_bk_nee_profile": True, "original": prop_original,
            "mapped_lut_cache_dir": str(input_root / "cache" / "rate_tables")}}


def run_optical_window(*, source, pre0, grid, beam, prop, ion, heat, raman,
                       schedule, output_dir: Path, observe=True) -> dict:
    """Run the unchanged eight-interval propagator against an in-memory PRE view."""
    import numpy as np
    from KHz_filament.constants import N0_air, Ui_N2, c0, n2_air
    from KHz_filament.device import xp
    from KHz_filament.grids import make_axes
    from KHz_filament.longitudinal import build_deposition_contract
    from KHz_filament.propagate import propagate_one_pulse
    from KHz_filament.slow_state import HR3BDiagnosticSink, validate_hr3b_parameters
    from KHz_filament.thermalization import ThermalDiagnosticSink
    from KHz_filament.hr4e5_formal_entry import _all_interval_sample_plan, pre0_fields_from_delta_n
    from KHz_filament.hr4e5_evidence import sha256_array
    if schedule.n_intervals != 8:
        raise OpticalError("diagnostic propagation requires exactly eight complete intervals")
    if xp is not np:
        raise OpticalError("diagnostic propagation backend changed from NumPy")
    output_dir.mkdir(parents=True, exist_ok=False)
    original_source = np.asarray(source)
    if original_source.dtype != np.dtype("complex128") or not original_source.flags.c_contiguous:
        raise OpticalError("propagation source must be canonical C-contiguous complex128")
    E = np.array(original_source, dtype=np.complex128, order="C", copy=True)
    if np.shares_memory(E, original_source):
        raise OpticalError("fresh optical input aliases the sealed source")
    source_copy_hash = str(sha256_array(E))
    view = DiagnosticPREView(pre0_fields_from_delta_n(pre0, expected_count=8))
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    omega0 = 2.0 * xp.pi * c0 / float(beam.lam0)
    k0 = float(beam.n0) * float(omega0) / c0
    n2_used = float(getattr(prop, "n2", getattr(beam, "n2_air", n2_air)))
    beta = validate_hr3b_parameters(rho0=float(heat.rho0), Cv=float(heat.Cv),
                                     T0=float(prop.air_T), n0=float(beam.n0))
    hr3b_parameters = {"rho0": float(heat.rho0), "Cv": float(heat.Cv),
        "T0": float(prop.air_T), "n0": float(beam.n0), "beta_th": float(beta)}
    plan = _all_interval_sample_plan(schedule)
    thermal_sink = ThermalDiagnosticSink(plan=plan, output_path=str(output_dir / "sinks"),
        shape=(grid.Ny, grid.Nx), dtype=np.float64, enabled=True, mode="validation")
    hr3b_sink = HR3BDiagnosticSink(plan=plan, output_path=str(output_dir / "sinks"),
        shape=(grid.Ny, grid.Nx), dtype=np.float64, enabled=True)
    observer = IntervalObserver(propagate_one_pulse.__code__, output_dir / "observer_progress.jsonl",
        output_dir, view, canonical_hash=sha256_array) if observe else None
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    try:
        if observer is None:
            final_E, _, diag = propagate_one_pulse(
                E, kperp2=axes.kperp2, k0=k0, omega0=float(omega0),
                dz=float(schedule.dz_intervals[0]), z_max=float(schedule.z_end-schedule.z_start),
                n0=float(beam.n0), n2=n2_used, Ui=Ui_N2, N0=N0_air,
                ion_conf=ion, dn_gas=None, dt=axes.dt, axes=axes, prop_conf=prop,
                raman_conf=raman, record_onaxis_rho_time=True, record_every_z=1,
                longitudinal_schedule=schedule,
                deposition_contract=build_deposition_contract(schedule, axes=axes),
                thermal_sink=thermal_sink, thermal_slow_state=view,
                hr3b_parameters=hr3b_parameters, hr3b_sink=hr3b_sink)
        else:
            with observer:
                final_E, _, diag = propagate_one_pulse(
                    E, kperp2=axes.kperp2, k0=k0, omega0=float(omega0),
                    dz=float(schedule.dz_intervals[0]), z_max=float(schedule.z_end-schedule.z_start),
                    n0=float(beam.n0), n2=n2_used, Ui=Ui_N2, N0=N0_air,
                    ion_conf=ion, dn_gas=None, dt=axes.dt, axes=axes, prop_conf=prop,
                    raman_conf=raman, record_onaxis_rho_time=True, record_every_z=1,
                    longitudinal_schedule=schedule,
                    deposition_contract=build_deposition_contract(schedule, axes=axes),
                    thermal_sink=thermal_sink, thermal_slow_state=view,
                    hr3b_parameters=hr3b_parameters, hr3b_sink=hr3b_sink)
    except BaseException:
        try: thermal_sink.finalize()
        except Exception: pass
        try: hr3b_sink.finalize()
        except Exception: pass
        view.close()
        raise
    finally:
        view.close()
    propagation_wall = time.perf_counter() - start_wall
    propagation_cpu = time.process_time() - start_cpu
    output_started = time.perf_counter()
    if not view.complete:
        raise OpticalError("PRE adapter did not read/update every interval exactly once")
    if len(diag.get("z_axis", ())) != 8 or not np.allclose(
            np.asarray(diag["z_axis"], dtype=np.float64), np.asarray(schedule.z_edges[1:]),
            rtol=0.0, atol=1e-12):
        raise OpticalError("propagation returned an incomplete/non-matching eight-interval z history")
    closure = _diagnostic_closure(diag)
    verify_array_contract(final_E, shape=source.shape, dtype="complex128", label="final E")
    _save_npy_new(output_dir / "final_E.npy", final_E)
    final_post = {k: np.stack([view.post_fields(i)[k] for i in range(8)]) for k in ("delta_n", "vx", "vy")}
    _save_npz_new(output_dir / "final_POST.npz", final_post)
    numeric = {str(k): np.asarray(v) for k, v in diag.items()
               if isinstance(v, (np.ndarray, list, tuple)) and np.asarray(v).dtype.kind in "biufc"}
    _write_new(output_dir / "diagnostics.npz", _npz_bytes(numeric))
    profile = {"halfstep1_json": _jsonable(diag.get("bk_nee_profile_halfstep_1_json", ())),
               "halfstep2_json": _jsonable(diag.get("bk_nee_profile_halfstep_2_json", ()))}
    _json_new(output_dir / "linear_halfstep_profiles.json", profile)
    final_hash = str(sha256_array(np.asarray(final_E)))
    return {"schema": SCHEMA, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION, "status": "COMPLETE_DIAGNOSTIC",
        "interval_count": 8, "input_fresh_copy_canonical_sha256": source_copy_hash,
        "final_E_canonical_sha256": final_hash,
        "final_POST_canonical_sha256": {k: _canonical_hash(v) for k, v in final_post.items()},
        "final_POST_raw_sha256": raw_sha256_file(output_dir / "final_POST.npz"),
        "final_E_raw_sha256": raw_sha256_file(output_dir / "final_E.npy"),
        "interval_rows": observer.rows if observer is not None else [],
        "PRE_complete": view.complete, "diagnostic_closure": closure,
        "propagation_wall_seconds": propagation_wall,
        "propagation_process_cpu_seconds": propagation_cpu,
        "validation_output_seconds": time.perf_counter() - output_started,
        "observer_overhead_seconds": sum(r.get("observer_overhead_seconds", 0.0) for r in observer.rows) if observer else 0.0,
        "linear_profile_available": bool(prop.diag_bk_nee_profile),
        "profile_runtime_only": True,
        "observables": {k: _jsonable(diag.get(k)) for k in
            ("z_axis", "U_z", "I_max_z", "rho_N2_max_z", "rho_O2_max_z",
             "dz_used_z", "adaptive_rejection_count_z", "safety_mode_trigger_count_z")
            if k in diag}}


def _prepare_case(input_root: Path, bench_root: Path, vendor_dir: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    context = _load_context(input_root, bench_root, vendor_dir)
    load_seconds = time.perf_counter() - started
    run = run_optical_window(source=context["source"], pre0=context["assets"]["pre0"],
        grid=context["grid"], beam=context["beam"], prop=context["prop"],
        ion=context["ion"], heat=context["heat"], raman=context["raman"],
        schedule=context["schedule"], output_dir=output_dir, observe=True)
    run.update({"case_input_validation_seconds": load_seconds,
        "source_raw_sha256": SOURCE_RAW_SHA256,
        "source_canonical_sha256": context["source_hash"],
        "source_validation": _jsonable(context["source_validation"]),
        "source_binding": context["source_binding"],
        "pre0_raw_sha256": PRE0_RAW_SHA256,
        "pre0_canonical_sha256": context["assets"]["pre0_canonical_sha256"],
        "transfer_mapping_sha256": context["assets"]["mapping_sha256"],
        "lut_validation": context["lut"],
        "runtime_config_delta": context["runtime_config_delta"],
        "schedule": context["schedule"].as_metadata()})
    _json_new(output_dir / "child_receipt.json", run)
    return run


def _validate_case_outputs(case_dir: Path) -> dict:
    import numpy as np
    from KHz_filament.hr4e5_evidence import sha256_array
    child_root = case_dir / "propagation"
    receipt_path = child_root / "child_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETE_DIAGNOSTIC" or receipt.get("interval_count") != 8:
        raise OpticalError("child receipt does not show a complete eight-interval window")
    final_path = child_root / "final_E.npy"
    final = np.load(final_path, mmap_mode="r", allow_pickle=False)
    final_hash = verify_array_contract(final, shape=EXPECTED_SHAPE, dtype="complex128", label="parent final E")
    final_raw = raw_sha256_file(final_path)
    if final_hash != receipt.get("final_E_canonical_sha256") or final_raw != receipt.get("final_E_raw_sha256"):
        raise OpticalError("final optical field failed parent raw/canonical verification")
    first_path = child_root / "first_complete_interval_E.npy"
    first = np.load(first_path, mmap_mode="r", allow_pickle=False)
    first_hash = verify_array_contract(first, shape=EXPECTED_SHAPE, dtype="complex128", label="parent first E")
    if (first.shape != EXPECTED_SHAPE or first.dtype != np.dtype("complex128")
            or not first.flags.c_contiguous or first_hash != receipt["interval_rows"][0]["E_canonical_sha256"]):
        raise OpticalError("first complete interval E snapshot failed output validation")
    post_path = child_root / "first_complete_interval_POST.npz"
    with np.load(post_path, allow_pickle=False) as payload:
        if set(payload.files) != {"delta_n", "vx", "vy"}:
            raise OpticalError("first interval POST snapshot fields are incomplete")
        post_hashes = {key: str(sha256_array(payload[key])) for key in payload.files}
    if post_hashes != receipt["interval_rows"][0]["post_field_canonical_sha256"]:
        raise OpticalError("first interval POST snapshot failed canonical output validation")
    final_post_path = child_root / "final_POST.npz"
    if raw_sha256_file(final_post_path) != receipt["final_POST_raw_sha256"]:
        raise OpticalError("final POST raw hash mismatch")
    with np.load(final_post_path, allow_pickle=False) as payload:
        if set(payload.files) != {"delta_n", "vx", "vy"}:
            raise OpticalError("final POST field set mismatch")
        final_post_hashes = {k: verify_array_contract(payload[k], shape=EXPECTED_PRE0_SHAPE, dtype="float64", label="final POST") for k in payload.files}
    if final_post_hashes != receipt["final_POST_canonical_sha256"]:
        raise OpticalError("final POST canonical hash mismatch")
    rows = receipt.get("interval_rows", [])
    if len(rows) != 8 or [int(row["interval_index"]) for row in rows] != list(range(8)):
        raise OpticalError("observer receipt is missing full accepted interval coverage")
    return {"status": "DONE", "receipt": receipt,
        "final_path": str(final_path), "final_raw_sha256": final_raw,
        "final_canonical_sha256": final_hash,
        "first_E_raw_sha256": raw_sha256_file(first_path),
        "first_E_canonical_sha256": first_hash,
        "first_POST_raw_sha256": raw_sha256_file(post_path),
        "first_POST_canonical_sha256": post_hashes,
        "final_POST_canonical_sha256": final_post_hashes}


def _read_new_events(path: Path, offset: int):
    events, new_offset = [], offset
    if not path.is_file(): return events, offset
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        while True:
            line = stream.readline()
            if not line: break
            if not line.endswith("\n"): break
            new_offset = stream.tell()
            events.append(json.loads(line))
    return events, new_offset


def _stop_case(proc) -> dict:
    if proc.poll() is not None: return {"terminated": True, "method": "already_exited"}
    method = "p1_process_group_term_then_kill"
    common.kill_child(proc)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL) if os.name != "nt" else proc.kill()
        except OSError: pass
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: return {"terminated": False, "method": method, "bounded_wait_seconds": 10}
    return {"terminated": proc.poll() is not None, "method": method}


def _supervise_case(case: str, *, input_root: Path, bench_root: Path,
                    vendor_dir: Path, output_root: Path, case_limit=CASE_LIMIT_SECONDS) -> dict:
    case_dir = output_root / case
    case_dir.mkdir(exist_ok=False)
    log_path = case_dir / "child.log"
    propagation_dir = case_dir / "propagation"
    command = [sys.executable, str(Path(__file__).resolve()), "--child-case", case,
        "--input-root", str(input_root), "--bench-root", str(bench_root),
        "--vendor-dir", str(vendor_dir), "--output-root", str(propagation_dir)]
    env = dict(os.environ)
    env.update({"UPPE_USE_GPU": "0", "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    env["PYTHONPATH"] = os.pathsep.join([str(bench_root / "Filament_python"), str(vendor_dir), env.get("PYTHONPATH", "")])
    started = time.perf_counter()
    offset, first_start, peak_rss, sampled_cpu = 0, None, 0, 0.0
    stop_reason, stop_evidence = None, None
    with log_path.open("xb") as log:
        proc = subprocess.Popen(command, cwd=str(bench_root), env=env,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"))
        try:
            while proc.poll() is None:
                events, offset = _read_new_events(propagation_dir / "observer_progress.jsonl", offset)
                for event in events:
                    if event.get("event") == "first_interval_started":
                        first_start = float(event["monotonic"])
                    elif event.get("event") == "complete_interval" and event.get("interval_index") == 0:
                        first_start = None
                snap = common.tree_snapshot(proc.pid)
                if snap is None:
                    stop_reason = "PROCESS_TREE_RSS_UNAVAILABLE"; break
                peak_rss = max(peak_rss, int(snap["rss"]))
                sampled_cpu = max(sampled_cpu, float(snap["cpu"]))
                elapsed = time.perf_counter() - started
                if peak_rss >= RSS_LIMIT_BYTES:
                    stop_reason = "RSS_80_PERCENT_GATE"; break
                if first_start is not None and time.perf_counter() - first_start >= B0_LIMIT_SECONDS:
                    stop_reason = "B0_FIRST_INTERVAL_LIMIT_REACHED"; break
                if elapsed >= case_limit:
                    stop_reason = "O1_45_MINUTE_LIMIT_REACHED"; break
                time.sleep(0.5)
        except BaseException as exc:
            stop_reason = "MONITOR_ERROR: " + type(exc).__name__ + ": " + str(exc)
        finally:
            if proc.poll() is None:
                stop_evidence = _stop_case(proc)
        if stop_reason is not None:
            stop_evidence = _stop_case(proc)
        if proc.poll() is None:
            stop_evidence = _stop_case(proc)
        child_exit = proc.poll()
        if child_exit is None:
            status = "STOP_UNRESOLVED"
        elif stop_reason:
            status = "LIMIT_REACHED" if "LIMIT" in stop_reason else "STOPPED"
        else:
            status = "PROCESS_EXITED"
    child_wall = time.perf_counter() - started
    validated = None
    validation_error = None
    if status == "PROCESS_EXITED" and child_exit == 0:
        try:
            validated = _validate_case_outputs(case_dir)
            status = "DONE"
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            status = "OUTPUT_VALIDATION_FAIL"
    supervisor = {"schema": SCHEMA, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION, "case": case, "status": status,
        "child_command": command, "child_exit_code": child_exit,
        "stop_reason": stop_reason, "stop_evidence": stop_evidence,
        "child_wall_seconds": child_wall,
        "service_wall_seconds_including_output_validation": time.perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "sampled_process_tree_cpu_peak_not_cumulative_seconds": sampled_cpu,
        "validation": validated, "validation_error": validation_error,
        "stdout_stderr_file": str(log_path),
        "partial_progress_file": str(case_dir / "propagation" / "observer_progress.jsonl")}
    _json_new(case_dir / "supervisor_receipt.json", supervisor)
    return supervisor


def _fft_share(receipt: dict) -> dict:
    names = ("temporal_fft", "spatial_fft2", "inverse_spatial_fft2", "inverse_temporal_fft")
    fft_s, linear_s, compute_s = 0.0, 0.0, 0.0
    count = 0
    for row in receipt.get("interval_rows", []):
        for half in (row.get("bk_nee_profile_halfstep_1"), row.get("bk_nee_profile_halfstep_2")):
            if not isinstance(half, dict) or not isinstance(half.get("stages"), dict): continue
            count += 1
            fft_s += sum(float(half["stages"].get(name, {}).get("walltime_s", 0.0)) for name in names)
        linear_s += float(row.get("linear_walltime_step_seconds") or 0.0)
        compute_s += float(row.get("interval_wall_seconds") or 0.0)
    return {"profile_halfstep_count": count, "fft_exclusive_seconds": fft_s,
        "linear_step_wall_seconds": linear_s,
        "complete_interval_compute_seconds": compute_s,
        "fft_fraction_of_complete_interval_compute": (fft_s / compute_s if compute_s > 0 else None),
        "appendix_threshold_fraction": 0.50,
        "status": "ELIGIBLE_FOR_PARENT_DECISION" if count and compute_s > 0 and fft_s / compute_s >= 0.50
                 else ("SKIPPED_FFT_SHARE_BELOW_50_PERCENT" if count else "UNAVAILABLE_PROFILE")}


def _compare_cases(a: dict, b: dict) -> dict:
    import numpy as np
    final_a = np.load(Path(a["validation"]["final_path"]), mmap_mode="r", allow_pickle=False)
    final_b = np.load(Path(b["validation"]["final_path"]), mmap_mode="r", allow_pickle=False)
    final_metrics = _array_metrics(final_a, final_b)
    first_a = np.load(Path(a["validation"]["final_path"]).parent / "first_complete_interval_E.npy",
                      mmap_mode="r", allow_pickle=False)
    first_b = np.load(Path(b["validation"]["final_path"]).parent / "first_complete_interval_E.npy",
                      mmap_mode="r", allow_pickle=False)
    first_metrics = _array_metrics(first_a, first_b)
    post_a_path = Path(a["validation"]["final_path"]).parent / "first_complete_interval_POST.npz"
    post_b_path = Path(b["validation"]["final_path"]).parent / "first_complete_interval_POST.npz"
    post_metrics = {}
    with np.load(post_a_path, allow_pickle=False) as pa, np.load(post_b_path, allow_pickle=False) as pb:
        for key in ("delta_n", "vx", "vy"):
            post_metrics[key] = _array_metrics(pa[key], pb[key])
    final_post_metrics = {}
    with np.load(Path(a["validation"]["final_path"]).parent / "final_POST.npz", allow_pickle=False) as pa, np.load(Path(b["validation"]["final_path"]).parent / "final_POST.npz", allow_pickle=False) as pb:
        for key in ("delta_n", "vx", "vy"):
            final_post_metrics[key] = _array_metrics(pa[key], pb[key])
    ra, rb = a["validation"]["receipt"], b["validation"]["receipt"]
    observables_equal = ra["observables"] == rb["observables"]
    all_intervals_equal = all(x["E_canonical_sha256"] == y["E_canonical_sha256"] and x["post_field_canonical_sha256"] == y["post_field_canonical_sha256"] for x,y in zip(ra["interval_rows"], rb["interval_rows"], strict=True))
    exact = (observables_equal and all_intervals_equal and a["validation"]["final_POST_canonical_sha256"] == b["validation"]["final_POST_canonical_sha256"] and a["validation"]["final_canonical_sha256"] == b["validation"]["final_canonical_sha256"]
             and a["validation"]["first_E_canonical_sha256"] == b["validation"]["first_E_canonical_sha256"]
             and a["validation"]["first_POST_canonical_sha256"] == b["validation"]["first_POST_canonical_sha256"])
    return {"comparison_scope": "same sealed E_source/PRE0/config in independent fresh child processes",
        "exact_canonical_match": exact, "final_E": final_metrics,
        "final_POST_fields": final_post_metrics, "observables_equal": observables_equal, "all_interval_hashes_equal": all_intervals_equal,
        "first_complete_interval_E": first_metrics, "first_POST_fields": post_metrics,
        "classification": "BITEXACT_REPEAT" if exact else "NUMERICAL_DIFFERENCE"}


def _smoke_child(output_dir: Path) -> dict:
    """Synthetic tiny-grid software check only; never a benchmark observation."""
    import numpy as np
    from KHz_filament.config import BeamConfig, GridConfig, HeatConfig, IonizationConfig, PropagationConfig, RamanConfig
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.hr4e5_formal_entry import build_prefix_schedule
    grid = GridConfig(Nx=8, Ny=8, Nt=8, Lx=8e-4, Ly=8e-4, Twin=80e-15)
    beam = BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-10, focal_length=None)
    prop = PropagationConfig(z_max=8e-4, dz=1e-4, linear_model="bk_nee", auto_substep=False,
        focus_window_step=False, limit_focus_window=False, progress_every_z=0, energy_probe_every=0,
        diag_extra=True, diag_operator_energy=True, measure_performance=True, diag_bk_nee_profile=True,
        use_electronic_kerr=False, use_raman_phase=False, use_raman_absorption=False,
        use_plasma_phase=False, use_ionization_loss=False, use_ionization_solver=False)
    source = np.asarray((1 + np.arange(512).reshape(8,8,8)/512) * (1+0.03j), dtype=np.complex128)
    pre0 = np.full((8,8,8), -1e-9, dtype=np.float64)
    source_before, pre_before = _canonical_hash(source), _canonical_hash(pre0)
    kw = dict(source=source, pre0=pre0, grid=grid, beam=beam, prop=prop,
        ion=IonizationConfig(species=[]), heat=HeatConfig(hr3b_enabled=True),
        raman=RamanConfig(enabled=False, absorption=False),
        schedule=build_prefix_schedule(build_longitudinal_schedule(1e-4, 8e-4), 8))
    output_dir.mkdir(parents=True, exist_ok=False)
    receipts=[]
    for name, observe in (("plain",False),("observed_a",True),("observed_b",True)):
        receipts.append(run_optical_window(**kw, output_dir=output_dir/name, observe=observe))
    assert _canonical_hash(source)==source_before and _canonical_hash(pre0)==pre_before
    assert len({r["final_E_canonical_sha256"] for r in receipts})==1
    assert all(r["final_POST_canonical_sha256"]==receipts[0]["final_POST_canonical_sha256"] for r in receipts)
    assert all(r["observables"]==receipts[0]["observables"] for r in receipts)
    assert all(len(r["interval_rows"])==8 for r in receipts[1:])
    assert all(r["interval_rows"][0]["B0_status"]=="DONE" for r in receipts[1:])
    result={"status":"PASS", "scope":"SYNTHETIC_TINY_SOFTWARE_ONLY", "backend":"NUMPY_CPU",
        "observed_vs_unobserved_bitexact":True,"independent_repeat_bitexact":True,
        "python":sys.version,"receipts":receipts}
    _json_new(output_dir/"smoke_receipt.json",result)
    return result


def _entry_smoke(args) -> int:
    args.output_root.mkdir(parents=True, exist_ok=False)
    command=[sys.executable,"-s","-B",str(Path(__file__).resolve()),"--smoke-child",
        "--bench-root",str(args.bench_root),"--input-root",str(args.input_root),
        "--vendor-dir",str(args.vendor_dir),"--output-root",str(args.output_root/"child")]
    with (args.output_root/"child.log").open("xb") as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!="nt")
        try:
            code=child.wait(timeout=90)
        finally:
            if child.poll() is None: _stop_case(child)
    if code != 0: raise OpticalError("tiny parent-child entry failed; inspect child.log")
    result=json.loads((args.output_root/"child/smoke_receipt.json").read_text())
    assert result["status"]=="PASS"
    _json_new(args.output_root/"entry_receipt.json", {"status":"PASS","command":command,"scope":result["scope"]})
    return 0


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--child-case", choices=("A", "B"), help=argparse.SUPPRESS)
    parser.add_argument("--entry-smoke", action="store_true")
    parser.add_argument("--smoke-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--admission-only", action="store_true")
    return parser


def _child_main(args) -> int:
    try:
        if sys.version_info < (3, 12) or not hasattr(sys, "monitoring"):
            raise OpticalError("Python 3.12+ sys.monitoring is required")
        _check_allocation()
        common.add_vendor(str(args.vendor_dir))
        import threadpoolctl
        limit = threadpoolctl.threadpool_limits(limits=1)
        if any(int(p.get("num_threads", 1)) > 1 for p in threadpoolctl.threadpool_info()):
            raise OpticalError("child thread pool is not single-threaded")
        _prepare_case(args.input_root, args.bench_root, args.vendor_dir, args.output_root)
        del limit
        return 0
    except BaseException as exc:
        failure = {"schema": SCHEMA, "classification": CLASSIFICATION,
            "production_qualification": QUALIFICATION, "case": args.child_case,
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()}
        try: _json_new(args.output_root / "child_failure.json", failure)
        except Exception: pass
        traceback.print_exc(); return 1


def _coordinator(args) -> int:
    if sys.version_info < (3, 12) or not hasattr(sys, "monitoring"):
        raise OpticalError("Python 3.12+ sys.monitoring is required")
    allocation = _check_allocation()
    common.add_vendor(str(args.vendor_dir))
    import threadpoolctl
    thread_limit = threadpoolctl.threadpool_limits(limits=1)
    if getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise OpticalError("frozen vendor threadpoolctl 3.6.0 is required")
    preflight = _load_context(args.input_root, args.bench_root, args.vendor_dir)
    del preflight
    import gc
    gc.collect()
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(exist_ok=False)
    _environment_receipt(args.output_root, vendor_dir=args.vendor_dir, allocation=allocation)
    allocation_start = float(os.environ["P1R_ALLOCATION_START_EPOCH"])
    work_deadline = allocation_start + 7200 - 600
    stages = []
    for case in ("A", "B"):
        if stages and stages[-1].get("status") != "DONE":
            stages.append({"case": case, "status": "BLOCKED_AFTER_PRIOR_CASE_FAILURE"})
            continue
        remaining = work_deadline - time.time()
        predicted = stages[0]["service_wall_seconds_including_output_validation"] * 1.2 if stages else CASE_LIMIT_SECONDS
        if remaining < predicted:
            stages.append({"case": case, "status": "SKIPPED_INSUFFICIENT_TIME", "remaining_seconds": remaining, "required_seconds": predicted})
            continue
        result = _supervise_case(case, input_root=args.input_root,
            bench_root=args.bench_root, vendor_dir=args.vendor_dir,
            output_root=args.output_root, case_limit=min(CASE_LIMIT_SECONDS, remaining))
        stages.append(result)
        _append_event(args.output_root / "run_progress.jsonl", {"case": case,
            "status": result.get("status"),
            "service_wall_seconds": result.get("service_wall_seconds_including_output_validation")})
    completed = len(stages) == 2 and all(s.get("status") == "DONE" for s in stages)
    comparison = _compare_cases(stages[0], stages[1]) if completed else None
    fft = {s["case"]: _fft_share(s["validation"]["receipt"])
           for s in stages if s.get("status") == "DONE"}
    allocation_elapsed = None
    allocation_start = os.environ.get("P1R_ALLOCATION_START_EPOCH")
    if allocation_start:
        allocation_elapsed = max(0.0, time.time() - float(allocation_start))
    receipt = {"schema": SCHEMA, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION,
        "status": ("COMPLETE_DIAGNOSTIC" if completed and comparison["exact_canonical_match"]
                   else ("NUMERICAL_DIFFERENCE" if completed else "INCOMPLETE")),
        "execution_order": ["O1-A", "O1-B"], "execution_mode": "SERIAL",
        "allocation": allocation, "allocation_elapsed_seconds": allocation_elapsed,
        "allocation_wall_limit_seconds": 7200, "stages": stages,
        "repeat_comparison": comparison, "fft_profile_assessment": fft,
        "fft_appendix_threshold_fraction": 0.50,
        "optional_16_32_intervals": "SKIPPED_INPUT_ONLY8",
        "fft_candidate": "NOT_RUN_PENDING_MEASURED_EXCLUSIVE_SHARE_AND_PARENT_DECISION"}
    _json_new(args.output_root / "run_receipt.json", receipt)
    del thread_limit
    return 0 if receipt["status"] == "COMPLETE_DIAGNOSTIC" else 1


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.smoke_child:
            _source_binding(args.bench_root)
            _smoke_child(args.output_root)
            return 0
        if args.entry_smoke: return _entry_smoke(args)
        if args.admission_only:
            context = _load_context(args.input_root, args.bench_root, args.vendor_dir)
            _json_new(args.output_root, {"status":"PASS", "classification":CLASSIFICATION,
                "source_binding":context["source_binding"], "source_hash":context["source_hash"],
                "schedule":context["schedule"].as_metadata(), "lut":context["lut"],
                "runtime_config_delta":context["runtime_config_delta"]})
            return 0
        return _child_main(args) if args.child_case else _coordinator(args)
    except BaseException as exc:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

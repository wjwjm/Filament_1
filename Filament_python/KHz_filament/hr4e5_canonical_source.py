"""Immutable canonical E_source input for E5-1B WP2.

Only the CPU coordinator may materialize a source artifact. Optical workers
validate and load the sealed artifact; this module does not alter the source
field definition or its exact identity convention.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import device
from .confio import load_all
from .device import to_cpu, xp
from .grids import make_axes
from .hr4e5_evidence import sha256_array, sha256_file
from .runner import apply_thin_lens_achromatic, build_transverse_input_field
from .constants import c0


CANONICAL_SOURCE_SCHEMA = "khz_filament.hr4e5.wp2.canonical_source.v1"
CANONICAL_SOURCE_SEAL_SCHEMA = "khz_filament.hr4e5.wp2.canonical_source_seal.v1"
CANONICAL_SOURCE_MANIFEST_SCHEMA = "local-source.v1"
CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES = 64 * 1024
CANONICAL_SOURCE_HASH_SCOPE = "classified_by_record"
CANONICAL_SOURCE_HASH_SCOPES = {
    "config_sha256": "raw_bytes",
    "file_sha256": "raw_bytes",
    "payload_bytes_sha256": "c_order_numeric_array_payload_bytes",
    "canonical_array_hash": "sha256_array(dtype,shape,C-bytes)",
}
CANONICAL_SOURCE_SEAL_HASH_SCOPES = {
    "source_logical_id": "sha256(canonical_json(schema,config_sha256,canonical_array_hash))",
    "artifact_file_sha256": "raw_bytes",
    "payload_bytes_sha256": "c_order_numeric_array_payload_bytes",
    "canonical_array_hash": "sha256_array(dtype,shape,C-bytes)",
    "config_sha256": "raw_bytes",
    "admission_identity_sha256": "sha256(canonical_json(admission_identity_without_identity_sha256_and_created_utc))",
    "implementation_identity_sha256": "sha256(canonical_json(implementation_identity_payload))",
}
FROZEN_N2_K64_CONFIG_SHA256 = "509b156d3e332a57214750ec2121a960e3f284bc30523eccc532f91b6d42a58b"
_ARRAY_IDENTITY_N2_K64 = "595ed4eedac5438e0fe71af7897235c69f52158bc54e95c110192c624058a4a5"
_ARRAY_PAYLOAD_N2_K64 = "f45d4cedb31641c7a35d46f7fdeddb45dce35f6df4c76eaea88c512d1992a428"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _logical_id(config_sha256: str, array_sha256: str) -> str:
    payload = {"schema": CANONICAL_SOURCE_SCHEMA,
               "config_sha256": str(config_sha256),
               "canonical_array_hash": str(array_sha256)}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _require_frozen_n2_k64_source(config_sha256: str, array_sha256: str,
                                  payload_sha256: str) -> None:
    # The historical hash is an independent authority for this one frozen
    # qualification input. Other formal configs bind their own generated
    # artifact and hashes through the source manifest and admission seal.
    if str(config_sha256) != FROZEN_N2_K64_CONFIG_SHA256:
        return
    if str(array_sha256) != _ARRAY_IDENTITY_N2_K64:
        raise ValueError("canonical E_source array identity differs from the frozen N2/K64 source")
    if str(payload_sha256) != _ARRAY_PAYLOAD_N2_K64:
        raise ValueError("canonical E_source payload SHA256 differs from the frozen N2/K64 source")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create(path: Path, writer) -> None:
    """Publish one file atomically without replacing an existing path."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if the destination appeared.
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    _atomic_create(path, lambda handle: handle.write(payload))


def _checked_bound_file(raw_path: str | Path, *, input_root: Path, label: str) -> Path:
    candidate = Path(raw_path)
    if ".." in candidate.parts:
        raise ValueError(f"canonical source {label} path traverses a parent")
    lexical = Path(os.path.abspath(candidate if candidate.is_absolute() else input_root / candidate))
    root = input_root.resolve(strict=True)
    if not lexical.is_relative_to(root):
        raise ValueError(f"canonical source {label} is outside its input directory")
    current = lexical
    while current != root.parent:
        if current.is_symlink():
            raise ValueError(f"canonical source {label} path contains a symlink")
        if current == root:
            break
        current = current.parent
    resolved = lexical.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"canonical source {label} is not a regular input file")
    if int(resolved.stat().st_nlink) != 1:
        raise ValueError(f"canonical source {label} has an unknown hardlink")
    return resolved


def construct_canonical_source(config_path: str | Path) -> np.ndarray:
    """Use the existing frozen source constructor on the CPU coordinator."""
    if device.USE_GPU:
        raise RuntimeError("canonical E_source generation is restricted to the CPU coordinator")
    grid, beam, prop, *_ = load_all(str(config_path))
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    field, _ = build_transverse_input_field(axes, beam, xp.complex128)
    if getattr(beam, "focal_length", None):
        omega0 = 2.0 * np.pi * c0 / beam.lam0
        k0 = beam.n0 * omega0 / c0
        if str(getattr(prop, "linear_model", "uppe")).lower() == "uppe":
            field = apply_thin_lens_achromatic(
                field, axes, beam, prop, chunk_t=getattr(prop, "lens_chunk_t", 0),
            )
        else:
            X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
            field *= xp.exp(xp.asarray(
                -1j * k0 * (X ** 2 + Y ** 2) / (2.0 * float(beam.focal_length)),
                dtype=xp.complex128,
            ))
    host = np.asarray(to_cpu(field))
    if host.dtype.byteorder == ">":
        raise RuntimeError("canonical E_source requires a little-endian CPU coordinator")
    canonical = np.array(host, dtype=np.dtype("<c16"), order="C", copy=True)
    if canonical.ndim != 3 or not np.isfinite(canonical).all():
        raise ValueError("canonical E_source must be a finite (t,y,x) array")
    if sha256_array(host) != sha256_array(canonical):
        raise ValueError("little-endian canonicalization changed the existing array identity")
    return canonical


def _record_for_array(array: np.ndarray, *, config_path: Path,
                      artifact_path: Path, sealed_manifest_path: Path) -> dict[str, Any]:
    payload_sha = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    array_sha = sha256_array(array)
    config_sha = sha256_file(config_path)
    _require_frozen_n2_k64_source(config_sha, array_sha, payload_sha)
    return {
        "schema": CANONICAL_SOURCE_SCHEMA,
        "hash_scopes": dict(CANONICAL_SOURCE_HASH_SCOPES),
        "logical_id": _logical_id(config_sha, array_sha),
        "artifact_path": str(artifact_path.resolve(strict=False)),
        "sealed_manifest_path": str(sealed_manifest_path.resolve(strict=False)),
        "file_sha256": "",
        "file_size_bytes": 0,
        "payload_bytes_sha256": payload_sha,
        "canonical_array_hash": array_sha,
        "shape": list(array.shape),
        "dtype": "<c16",
        "byte_order": "little",
        "layout": "C",
        "axis_order": ["t", "y", "x"],
        "config_sha256": config_sha,
    }


def _validate_array_and_record(array: np.ndarray, record: Mapping[str, Any], *,
                               file_path: Path, config_path: Path) -> dict[str, Any]:
    if (array.dtype.str != "<c16" or array.dtype.kind != "c" or array.dtype.itemsize != 16
            or not array.flags.c_contiguous or array.ndim != 3
            or not np.isfinite(array).all()):
        raise ValueError("canonical E_source dtype, layout, shape, or finite-value check failed")
    config_sha = sha256_file(config_path)
    array_sha = sha256_array(array)
    payload_sha = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    _require_frozen_n2_k64_source(config_sha, array_sha, payload_sha)
    expected = {
        "schema": CANONICAL_SOURCE_SCHEMA,
        "hash_scopes": dict(CANONICAL_SOURCE_HASH_SCOPES),
        "logical_id": _logical_id(config_sha, array_sha),
        "file_sha256": sha256_file(file_path),
        "file_size_bytes": int(file_path.stat().st_size),
        "payload_bytes_sha256": payload_sha,
        "canonical_array_hash": array_sha,
        "shape": list(array.shape),
        "dtype": "<c16",
        "byte_order": "little",
        "layout": "C",
        "axis_order": ["t", "y", "x"],
        "config_sha256": config_sha,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"canonical E_source {key} differs from its manifest")
    grid, *_ = load_all(str(config_path))
    if tuple(array.shape) != (int(grid.Nt), int(grid.Ny), int(grid.Nx)):
        raise ValueError("canonical E_source shape differs from the bound config axes")
    return {**dict(record), **expected}


def validate_canonical_source(record: Mapping[str, Any], *, config_path: str | Path,
                              source_manifest_path: str | Path) -> dict[str, Any]:
    """Read and verify one canonical artifact without invoking its constructor."""
    config = Path(config_path).resolve(strict=True)
    manifest = Path(source_manifest_path).resolve(strict=False)
    if manifest.is_symlink():
        raise ValueError("canonical source manifest path is linked")
    root = manifest.parent.resolve(strict=True)
    artifact = _checked_bound_file(record.get("artifact_path", ""), input_root=root,
                                   label="artifact")
    raw_sealed = Path(str(record.get("sealed_manifest_path", "")))
    if ".." in raw_sealed.parts:
        raise ValueError("canonical source sealed manifest path traverses a parent")
    lexical_sealed = Path(os.path.abspath(raw_sealed if raw_sealed.is_absolute()
                                          else root / raw_sealed))
    if not lexical_sealed.is_relative_to(root):
        raise ValueError("canonical source sealed manifest is outside its input directory")
    sealed = (_checked_bound_file(lexical_sealed, input_root=root, label="sealed manifest")
              if lexical_sealed.exists() else lexical_sealed)
    if sealed.parent != root:
        raise ValueError("canonical source sealed manifest must share the input directory")
    loaded = np.load(artifact, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise ValueError("canonical E_source artifact is not an NPY array")
    validated = _validate_array_and_record(loaded, record, file_path=artifact,
                                           config_path=config)
    validated["artifact_path"] = str(artifact)
    validated["sealed_manifest_path"] = str(sealed)
    return validated


def prepare_canonical_source_inputs(config_path: str | Path,
                                    source_manifest_path: str | Path) -> dict[str, Any]:
    """Create or validate pre-campaign source inputs on the CPU coordinator."""
    if device.USE_GPU:
        raise RuntimeError("only the CPU coordinator may prepare canonical E_source inputs")
    config = Path(config_path).resolve(strict=True)
    manifest_path = Path(source_manifest_path).resolve(strict=False)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(existing, dict)
                or existing.get("hash_scope") != CANONICAL_SOURCE_HASH_SCOPE):
            raise ValueError("existing source manifest lacks classified hash_scope")
        record = existing.get("canonical_source") if isinstance(existing, dict) else None
        if not isinstance(record, Mapping):
            raise ValueError("existing source manifest lacks the canonical artifact; refusing migration")
        checked = validate_canonical_source(record, config_path=config,
                                            source_manifest_path=manifest_path)
        if str(existing.get("canonical_array_hash", "")) != checked["canonical_array_hash"]:
            raise ValueError("source manifest array identity differs from its canonical artifact")
        return {**existing, "canonical_source": checked}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = manifest_path.parent / "E_source_canonical.npy"
    sealed_path = manifest_path.parent / "E_source_canonical.sealed.json"
    if artifact_path.exists() or artifact_path.is_symlink() or sealed_path.exists() or sealed_path.is_symlink():
        raise FileExistsError("canonical source files exist without their source manifest")
    array = construct_canonical_source(config)
    record = _record_for_array(array, config_path=config, artifact_path=artifact_path,
                               sealed_manifest_path=sealed_path)
    _atomic_create(artifact_path, lambda handle: np.save(handle, array, allow_pickle=False))
    record["file_sha256"] = sha256_file(artifact_path)
    record["file_size_bytes"] = int(artifact_path.stat().st_size)
    checked = validate_canonical_source(record, config_path=config,
                                        source_manifest_path=manifest_path)
    result = {
        "schema": CANONICAL_SOURCE_MANIFEST_SCHEMA,
        "hash_scope": CANONICAL_SOURCE_HASH_SCOPE,
        "canonical_array_hash": checked["canonical_array_hash"],
        "canonical_source_artifact_path": checked["artifact_path"],
        "canonical_source_sealed_manifest_path": checked["sealed_manifest_path"],
        "canonical_source": checked,
    }
    _atomic_create_bytes(manifest_path, json.dumps(
        result, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8"))
    return result


def _expected_seal(record: Mapping[str, Any], admission_identity: Mapping[str, Any],
                   implementation_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CANONICAL_SOURCE_SEAL_SCHEMA,
        "hash_scope": CANONICAL_SOURCE_HASH_SCOPE,
        "hash_scopes": dict(CANONICAL_SOURCE_SEAL_HASH_SCOPES),
        "sealed": True,
        "source_logical_id": str(record["logical_id"]),
        "artifact_path": str(record["artifact_path"]),
        "artifact_file_sha256": str(record["file_sha256"]),
        "artifact_file_size_bytes": int(record["file_size_bytes"]),
        "payload_bytes_sha256": str(record["payload_bytes_sha256"]),
        "canonical_array_hash": str(record["canonical_array_hash"]),
        "config_sha256": str(record["config_sha256"]),
        "admission_identity_sha256": str(admission_identity["identity_sha256"]),
        "implementation_identity_sha256": str(implementation_identity["identity_sha256"]),
        "created_utc": _utc(),
    }


def seal_or_validate_canonical_source(record: Mapping[str, Any], *,
                                      admission_identity: Mapping[str, Any],
                                      implementation_identity: Mapping[str, Any],
                                      config_path: str | Path,
                                      source_manifest_path: str | Path,
                                      create_if_missing: bool) -> dict[str, Any]:
    """Seal once after identities exist, or verify the existing sealed binding."""
    checked = validate_canonical_source(record, config_path=config_path,
                                        source_manifest_path=source_manifest_path)
    path = Path(checked["sealed_manifest_path"])
    if not path.exists():
        if not create_if_missing:
            raise ValueError("reopened canonical source lacks its sealed manifest")
        payload = _expected_seal(checked, admission_identity, implementation_identity)
        encoded = json.dumps(payload, sort_keys=True, indent=2,
                             ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(encoded) > CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES:
            raise ValueError("canonical source sealed manifest exceeds its storage allowance")
        _atomic_create_bytes(path, encoded)
    else:
        if path.is_symlink() or int(path.stat().st_nlink) != 1:
            raise ValueError("canonical source sealed manifest is linked")
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = _expected_seal(checked, admission_identity, implementation_identity)
        if (not isinstance(payload, dict)
                or set(payload) != set(required)
                or any(payload.get(key) != value for key, value in required.items()
                       if key != "created_utc")
                or not isinstance(payload.get("created_utc"), str)
                or len(payload["created_utc"]) < 20):
            raise ValueError("canonical source sealed manifest binding differs")
    actual_bytes = int(path.stat().st_size)
    if actual_bytes > CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES:
        raise ValueError("canonical source sealed manifest exceeds its storage allowance")
    return {
        "schema": "khz_filament.hr4e5.wp2.canonical_source_seal_receipt.v1",
        "status": "PASS", "artifact_path": checked["artifact_path"],
        "artifact_file_sha256": checked["file_sha256"],
        "artifact_file_bytes": int(checked["file_size_bytes"]),
        "payload_bytes_sha256": checked["payload_bytes_sha256"],
        "canonical_array_hash": checked["canonical_array_hash"],
        "sealed_manifest_path": str(path),
        "sealed_manifest_file_bytes": actual_bytes,
        "sealed_manifest_allowance_bytes": CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES,
        "atomic_write_temporary_peak_bytes": max(
            int(checked["file_size_bytes"]), CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES),
        "admission_identity_sha256": str(admission_identity["identity_sha256"]),
        "implementation_identity_sha256": str(implementation_identity["identity_sha256"]),
    }


def load_canonical_source_for_pulse(record: Mapping[str, Any], *,
                                    admission_identity: Mapping[str, Any],
                                    config_path: str | Path,
                                    source_manifest_path: str | Path,
                                    trajectory: str, role: str, pulse: int,
                                    attempt: int) -> tuple[Any, dict[str, Any]]:
    """Load one pulse-private backend buffer and verify exact host roundtrip."""
    checked = validate_canonical_source(record, config_path=config_path,
                                        source_manifest_path=source_manifest_path)
    sealed_path = Path(checked["sealed_manifest_path"])
    if not sealed_path.is_file() or sealed_path.is_symlink() or int(sealed_path.stat().st_nlink) != 1:
        raise ValueError("formal optical entry requires a regular sealed canonical source manifest")
    seal = json.loads(sealed_path.read_text(encoding="utf-8"))
    expected_admission = str(admission_identity.get("identity_sha256", ""))
    expected_implementation = str(admission_identity.get("implementation_identity_sha256", ""))
    expected_seal = {
        "schema": CANONICAL_SOURCE_SEAL_SCHEMA,
        "hash_scope": CANONICAL_SOURCE_HASH_SCOPE,
        "hash_scopes": dict(CANONICAL_SOURCE_SEAL_HASH_SCOPES),
        "sealed": True,
        "source_logical_id": str(checked["logical_id"]),
        "artifact_path": str(checked["artifact_path"]),
        "artifact_file_sha256": str(checked["file_sha256"]),
        "artifact_file_size_bytes": int(checked["file_size_bytes"]),
        "payload_bytes_sha256": str(checked["payload_bytes_sha256"]),
        "canonical_array_hash": str(checked["canonical_array_hash"]),
        "config_sha256": str(checked["config_sha256"]),
        "admission_identity_sha256": expected_admission,
        "implementation_identity_sha256": expected_implementation,
    }
    if (not isinstance(seal, dict) or set(seal) != {*expected_seal, "created_utc"}
            or any(seal.get(key) != value for key, value in expected_seal.items())
            or not isinstance(seal.get("created_utc"), str)
            or len(seal["created_utc"]) < 20):
        raise ValueError("canonical source seal does not match optical admission")
    host = np.load(checked["artifact_path"], allow_pickle=False)
    private_host = np.array(host, dtype=np.dtype("<c16"), order="C", copy=True)
    private_backend = xp.array(private_host, dtype=xp.complex128, order="C", copy=True)
    roundtrip = np.asarray(to_cpu(private_backend))
    roundtrip_hash = sha256_array(roundtrip)
    if (roundtrip_hash != checked["canonical_array_hash"]
            or not np.array_equal(roundtrip, host)):
        raise ValueError("canonical E_source host-device-host exact roundtrip failed")
    if device.USE_GPU:
        pointer = int(private_backend.data.ptr)
        if pointer <= 0:
            raise ValueError("canonical E_source GPU buffer identity is unavailable")
        backend_buffer = {"backend": "cupy", "device_id": int(private_backend.device.id),
                          "buffer_pointer_hex": hex(pointer)}
        alias_checked = True
    else:
        if np.shares_memory(private_backend, private_host):
            raise ValueError("canonical E_source CPU pulse buffer aliases its host staging array")
        backend_buffer = {"backend": "numpy", "buffer_pointer_hex": hex(int(private_backend.ctypes.data))}
        alias_checked = True
    receipt = {
        "schema": "khz_filament.hr4e5.wp2.canonical_source_pulse_receipt.v1",
        "status": "PASS", "trajectory": str(trajectory), "role": str(role),
        "pulse": int(pulse),
        "attempt": int(attempt), "artifact_logical_id": checked["logical_id"],
        "artifact_path": checked["artifact_path"],
        "artifact_file_sha256": checked["file_sha256"],
        "artifact_file_size_bytes": int(checked["file_size_bytes"]),
        "payload_bytes_sha256": checked["payload_bytes_sha256"],
        "canonical_array_hash": checked["canonical_array_hash"],
        "runtime_roundtrip_array_hash": roundtrip_hash,
        "shape": list(private_backend.shape), "dtype": str(private_backend.dtype),
        "layout": "C", "private_backend_buffer": backend_buffer,
        "source_alias_checked": alias_checked, "host_device_host_exact": True,
        "admission_identity_sha256": expected_admission,
        "implementation_identity_sha256": expected_implementation,
    }
    return private_backend, receipt


__all__ = [
    "CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES", "CANONICAL_SOURCE_SCHEMA",
    "construct_canonical_source", "load_canonical_source_for_pulse",
    "prepare_canonical_source_inputs", "seal_or_validate_canonical_source",
    "validate_canonical_source",
]

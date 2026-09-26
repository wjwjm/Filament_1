"""Stable scientific-pair and source-byte identities for E5-1B WP2 recovery.

These identities are control-plane evidence.  They never change the optical,
hydro, Streaming, or exact scientific operators.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Mapping

from tools.hpc_ops.provenance_v2 import canonical_lf_sha256


IMPLEMENTATION_IDENTITY_SCHEMA = "khz_filament.hr4e5.wp2.implementation_identity.v1"
SCIENTIFIC_PAIR_IDENTITY_SCHEMA = "khz_filament.hr4e5.wp2.scientific_pair_identity.v1"
_REQUIRED_WORKERS = frozenset({
    "KHz_filament/hr4e5_production_worker.py",
    "KHz_filament/hr4e5_production_optical_worker.py",
})


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_implementation_identity(*, python_root: str | Path | None = None,
                                    include_runtime: bool = True) -> dict[str, Any]:
    """Bind actual source bytes, including dirty and untracked production files.

    Canonical LF is the existing provenance-v2 text hash convention.  Git HEAD,
    branches, timestamps and the tracked-file index do not define this value.
    """
    root = Path(python_root).resolve() if python_root is not None else Path(__file__).resolve().parents[1]
    package = root / "KHz_filament"
    helper = root / "tools" / "hpc_ops" / "provenance_v2.py"
    if not package.is_dir() or not helper.is_file() or helper.is_symlink():
        raise ValueError("production implementation source tree is incomplete")
    paths = sorted(package.rglob("*.py")) + [helper]
    files = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"implementation source is not a regular file: {path}")
        relative = path.relative_to(root).as_posix()
        files.append({
            "path": relative,
            "canonical_lf_sha256": canonical_lf_sha256(path.read_bytes()),
        })
    names = {item["path"] for item in files}
    if not _REQUIRED_WORKERS.issubset(names):
        raise ValueError("production implementation lacks WP2 subprocess modules")
    runtime = (
        {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": _installed_version("numpy"),
            "scipy": _installed_version("scipy"),
            "cupy": _installed_version("cupy"),
        }
        if include_runtime else {}
    )
    payload = {
        "schema": IMPLEMENTATION_IDENTITY_SCHEMA,
        "hash_scope": "canonical_lf_python_source+runtime_versions",
        "files": files,
        "runtime": runtime,
    }
    return {**payload, "identity_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest()}


def require_same_implementation(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Reject cross-epoch reuse when code or runtime identity differs."""
    if (str(expected.get("schema")) != IMPLEMENTATION_IDENTITY_SCHEMA
            or str(actual.get("schema")) != IMPLEMENTATION_IDENTITY_SCHEMA
            or not expected.get("identity_sha256")
            or expected.get("identity_sha256") != actual.get("identity_sha256")
            or dict(expected) != dict(actual)):
        raise ValueError("REUSE_REJECTED / IMPLEMENTATION_IDENTITY_MISMATCH")


def build_scientific_pair_identity(*, admission_identity: Mapping[str, Any],
                                   pulse: int, reference_generation: str,
                                   candidate_generation: str) -> dict[str, Any]:
    """Return a durable logical pair ID independent of coordinator epoch."""
    if int(pulse) < 0 or not reference_generation or not candidate_generation:
        raise ValueError("scientific pair scope is incomplete")
    campaign_id = str(admission_identity.get("campaign_id", ""))
    admission_hash = str(admission_identity.get("identity_sha256", ""))
    config = admission_identity.get("config_identity", {})
    config_hash = str(config.get("sha256", "")) if isinstance(config, Mapping) else ""
    if not campaign_id or not admission_hash or not config_hash:
        raise ValueError("scientific pair admission identity is incomplete")
    payload = {
        "schema": SCIENTIFIC_PAIR_IDENTITY_SCHEMA,
        "campaign_id": campaign_id,
        "admission_identity_sha256": admission_hash,
        "formal_config_sha256": config_hash,
        "pulse": int(pulse),
        "logical_scientific_attempt": 0,
        "pair_role": "R+C",
        "reference_source_generation": str(reference_generation),
        "candidate_source_generation": str(candidate_generation),
    }
    return {**payload, "scientific_pair_id": hashlib.sha256(_canonical_json(payload)).hexdigest()}


def validate_scientific_pair_identity(saved: Mapping[str, Any], *, admission_identity: Mapping[str, Any],
                                      pulse: int, reference_generation: str,
                                      candidate_generation: str) -> dict[str, Any]:
    expected = build_scientific_pair_identity(
        admission_identity=admission_identity, pulse=pulse,
        reference_generation=reference_generation,
        candidate_generation=candidate_generation,
    )
    if dict(saved) != expected:
        raise ValueError("scientific pair identity differs from durable admission/generation")
    return expected

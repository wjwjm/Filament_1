"""Single supported E5-1A production construction and execution surface.

The local qualification and a future site-qualified admission use this exact
runner.  The admission level changes evidence requirements, never the
scientific calls or orchestration implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import sys
from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .confio import load_all
from .grids import make_axes
from .hr4c_state import HR4CThreeFieldStore, evolve_hr4_full_z
from .hr4d_pulse_lifecycle import build_interpulse_step_schedule
from .hr4e5_evidence import (PAIRED_EXACT_SCHEMA, _file_identity, _load_locator, _paired_exact_keys,
                              atomic_json, bind_exact_report, build_expected_object_set,
                              compare_arrays_exact, sha256_array, sha256_file,
                              validate_durable_report, validate_paired_exact_report)
from .hr4e5_formal_entry import (BLOCK_SIZE, QUEUE_DEPTH, build_admission_identity,
                                  build_prefix_schedule, create_pre0_root, create_successor_root,
                                  optical_write_forecast, persist_admission_identity, run_streaming_optical_pulse,
                                  resume_final_post, resolve_storage_lock_timeout_s,
                                  validate_final_post, validate_storage_lock_timeout_s,
                                  load_admission_identity, STORAGE_LOCK_TIMEOUT_KEY)
from .hr4e5_paired_campaign import CAMPAIGN_STATE_FILENAME, _open_production_driver
from .hr4e5_storage import (DEFAULT_FINAL_OUTPUT_BUDGET_BYTES, DEFAULT_SAFETY_MARGIN_BYTES,
                              HARD_CAP_BYTES, INTENT_SCHEMA, PROCESS_RECEIPT_SCHEMA, RECLAIM_PREREQUISITE_SCHEMA,
                              Reservation, StorageBudgetError, StorageIntegrityError,
    _is_reparse, plan_campaign_budget, process_identity)
from .longitudinal import build_longitudinal_schedule
from . import device
from .hr4e5_canonical_source import (
    CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES,
    CANONICAL_SOURCE_MANIFEST_SCHEMA,
    CANONICAL_SOURCE_HASH_SCOPE,
    seal_or_validate_canonical_source,
    validate_canonical_source,
)
from .hr4e5_gpu_binding import (build_assignment_record, validate_assignment_record,
                                validate_startup_final_identities)
from .hr4e5s_streaming import FIELDS, StreamingLifecycle
from .hr4e5_recovery_identity import (build_scientific_pair_identity,
                                      capture_implementation_identity,
                                      require_same_implementation)
from .hr4e5_recovery_output import (qualify_output_attempt, activate_output_attempt,
                                    validate_active_output)
from .hr4e_timestep import sha256_array_streaming


LOCAL_ORCHESTRATION_QUALIFICATION = "LOCAL_ORCHESTRATION_QUALIFICATION"
FORMAL_SITE_ADMISSION = "FORMAL_SITE_ADMISSION"
_ADMISSION_LEVELS = frozenset({LOCAL_ORCHESTRATION_QUALIFICATION, FORMAL_SITE_ADMISSION})
_SINK_FILES = {
    "ion": "sinks.hr3a_qion_samples.npy",
    "ib": "sinks.hr3a_qib_samples.npy",
    "raman": "sinks.hr3a_qraman_samples.npy",
    "qthermal": "sinks.hr3a_qthermal_samples.npy",
    "increment": "sinks.hr3b_delta_n_increment_samples.npy",
    "state_after": "sinks.hr3b_delta_n_state_after_update_samples.npy",
}
_FROZEN_LUT_MANIFEST_SCHEMA = "khz_filament.hr4e5.e5_1b.formal_lut.v1"
_FROZEN_LUT_STAGING_RECEIPT_SCHEMA = "khz_filament.hr4e5.e5_1b.lut_staging.v1"
_FROZEN_LUT_ARRAY_NAMES = [
    "I_grid", "W_grid", "metadata_json", "model_name", "species_name",
    "build_timestamp", "schema_version", "validation_json",
]
_FROZEN_LUT_RATE_FIELDS = (
    "enabled", "reuse_cache", "force_rebuild", "I_min_SI", "I_max_SI", "n_samples",
    "spacing", "interp_mode", "popruzhenko_max_terms", "popruzhenko_sum_tol",
    "ref_cycle_avg_samples",
)


def _positive_science_child_timeout(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("science_child_timeout_s must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class E5AProductionSpec:
    root: str | Path
    campaign_id: str
    config_path: str | Path
    source_manifest_path: str | Path
    lut_manifest_path: str | Path
    pre0_delta_n_path: str | Path
    full_config_path: str | Path | None = None
    prefix_intervals: int | None = None
    slow_source_manifest_path: str | Path | None = None
    site_resource_manifest_path: str | Path | None = None
    n_pulses: int = 3
    candidate_hydro_workers: int = 1
    admission_level: str = LOCAL_ORCHESTRATION_QUALIFICATION
    require_gpu_backend: bool = False
    reference_subprocess: bool = False
    max_campaign_live_bytes: int = HARD_CAP_BYTES
    final_output_budget_bytes: int = DEFAULT_FINAL_OUTPUT_BUDGET_BYTES
    safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES
    science_child_timeout_s: int = 900
    lut_staging_receipt_path: str | Path | None = None
    storage_lock_timeout_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "science_child_timeout_s",
                           _positive_science_child_timeout(self.science_child_timeout_s))
        if self.storage_lock_timeout_s is not None:
            object.__setattr__(self, "storage_lock_timeout_s",
                               validate_storage_lock_timeout_s(self.storage_lock_timeout_s))


class _LocalPolicyCapacityProvider:
    """Local A evidence: real free space plus policy cap, never site quota."""
    scope = "LOCAL_POLICY_CAP_NOT_SITE_QUOTA"

    def __init__(self, root: Path, cap: int):
        self.root, self.cap = root, int(cap)

    def free_bytes(self) -> int:
        return int(shutil.disk_usage(self.root).free)

    def quota_bytes(self) -> int:
        return self.cap


class _ManifestCapacityProvider:
    def __init__(self, evidence: Mapping[str, Any]):
        self.evidence = dict(evidence)

    def free_bytes(self) -> int:
        return int(self.evidence["free_bytes"])

    def quota_bytes(self) -> int:
        return int(self.evidence["quota_bytes"])


def _read_manifest(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} manifest is missing or linked")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} manifest is invalid")
    return value


def _checked_input_path(raw: str | Path, *, label: str, directory: bool = False) -> Path:
    """Reject the caller's lexical link path before resolving its identity."""
    candidate = Path(raw)
    if ".." in candidate.parts:
        raise StorageIntegrityError(f"{label} path traverses a parent")
    lexical = Path(os.path.abspath(candidate))
    for component in (lexical, *lexical.parents):
        if _is_reparse(component):
            raise StorageIntegrityError(f"{label} path contains a symlink or reparse point")
    resolved = lexical.resolve(strict=True)
    if (not resolved.is_dir() if directory else not resolved.is_file()):
        raise ValueError(f"{label} path has the wrong file type")
    if not directory and int(resolved.stat().st_nlink) != 1:
        raise StorageIntegrityError(f"{label} has an unknown hardlink")
    return resolved


def _checked_campaign_root(raw: str | Path) -> Path:
    candidate = Path(raw)
    if ".." in candidate.parts:
        raise StorageIntegrityError("campaign root traverses a parent")
    lexical = Path(os.path.abspath(candidate))
    if any(_is_reparse(part) for part in (lexical, *lexical.parents)):
        raise StorageIntegrityError("campaign root contains a symlink or reparse point")
    return lexical.resolve(strict=False)


def _checked_owned_file(root: Path, path: Path, owned: Mapping[str, Any], *, label: str) -> None:
    if (not path.is_relative_to(root) or any(_is_reparse(part) for part in (path, *path.parents))
            or not path.is_file() or int(path.stat().st_nlink) != 1
            or _file_identity(path) != owned.get("identity")
            or sha256_file(path) != owned.get("sha256")):
        raise StorageIntegrityError(f"{label} changed or is linked")


def _lut_file_provenance(path: Path) -> dict[str, int]:
    if any(_is_reparse(part) for part in (path, *path.parents)) or not path.is_file():
        raise StorageIntegrityError(f"frozen LUT file is missing or linked: {path}")
    stat_result = path.stat()
    result = {
        "st_dev": int(getattr(stat_result, "st_dev", -1)),
        "st_ino": int(getattr(stat_result, "st_ino", -1)),
        "st_nlink": int(getattr(stat_result, "st_nlink", 1)),
        "size": int(stat_result.st_size),
        "mtime_ns": int(getattr(stat_result, "st_mtime_ns", 0)),
    }
    for name in ("st_uid", "st_gid"):
        value = getattr(stat_result, name, None)
        if value is not None:
            result[name] = int(value)
    if result["st_nlink"] != 1:
        raise StorageIntegrityError(f"frozen LUT file has a hardlink: {path}")
    effective_uid = getattr(os, "geteuid", None)
    if (callable(effective_uid) and "st_uid" in result
            and result["st_uid"] != int(effective_uid())):
        raise StorageIntegrityError(f"frozen LUT file is not owned by the current user: {path}")
    return result


def _lut_scalar_text(archive: Any, name: str) -> str:
    value = np.asarray(archive[name])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise StorageIntegrityError(f"frozen LUT NPZ field {name} is not a string scalar")
    scalar = value.item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def _inspect_frozen_lut_npz(path: Path, *, expected_metadata: Mapping[str, Any],
                            expected_species: str, expected_model: str) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(_FROZEN_LUT_ARRAY_NAMES):
                raise StorageIntegrityError("frozen LUT NPZ fields differ from the authority manifest")
            metadata = json.loads(_lut_scalar_text(archive, "metadata_json"))
            validation = json.loads(_lut_scalar_text(archive, "validation_json"))
            model_name = _lut_scalar_text(archive, "model_name")
            species_name = _lut_scalar_text(archive, "species_name")
            build_timestamp = _lut_scalar_text(archive, "build_timestamp")
            schema_version = _lut_scalar_text(archive, "schema_version")
            I_grid = np.asarray(archive["I_grid"])
            W_grid = np.asarray(archive["W_grid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        if isinstance(error, StorageIntegrityError):
            raise
        raise StorageIntegrityError(f"frozen LUT NPZ is unreadable: {path}") from error
    if (not isinstance(metadata, dict)
            or not _json_values_equal_with_types(metadata, dict(expected_metadata))
            or model_name != expected_model or species_name != expected_species
            or schema_version != str(expected_metadata.get("schema_version", ""))
            or not isinstance(validation, dict) or not build_timestamp):
        raise StorageIntegrityError(f"frozen LUT metadata differs for {expected_species}")
    if (I_grid.dtype != np.dtype(np.float64) or W_grid.dtype != np.dtype(np.float64)
            or I_grid.ndim != 1 or W_grid.shape != I_grid.shape
            or I_grid.size != int(expected_metadata["n_samples"])
            or not np.all(np.isfinite(I_grid)) or not np.all(np.isfinite(W_grid))
            or np.any(I_grid <= 0.0) or np.any(np.diff(I_grid) <= 0.0)
            or np.any(W_grid < 0.0)):
        raise StorageIntegrityError(f"frozen LUT grid or rate array is invalid for {expected_species}")
    return {
        "I_grid": I_grid, "W_grid": W_grid, "metadata": metadata,
        "model_name": model_name, "species_name": species_name,
        "build_timestamp": build_timestamp, "schema_version": schema_version,
        "validation": validation,
    }


def _manifest_path_value(raw: Any, *, manifest_path: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise StorageIntegrityError(f"frozen LUT manifest lacks {label}")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return _checked_input_path(path, label=label)


def _frozen_lut_contract(config_path: str | Path, lut_manifest_path: str | Path,
                         campaign_root: str | Path) -> dict[str, Any]:
    """Validate the immutable LUT manifest against current config and source files."""
    from .constants import c0
    from .ionization.lut import (_canonical_table_metadata, _ion_rate_table_defaults,
                                 _table_path, _table_signature)

    config = _checked_input_path(config_path, label="execution config")
    manifest_path = _checked_input_path(lut_manifest_path, label="LUT manifest")
    root = _checked_campaign_root(campaign_root)
    manifest = _read_manifest(manifest_path, label="LUT")
    if (manifest.get("schema") != _FROZEN_LUT_MANIFEST_SCHEMA
            or manifest.get("mode") != "RATE_TABLE"):
        raise StorageIntegrityError("formal frozen LUT manifest schema or mode is invalid")
    manifest_config = _manifest_path_value(
        manifest.get("config_path"), manifest_path=manifest_path, label="manifest config")
    if (manifest_config != config
            or str(manifest.get("config_sha256", "")) != sha256_file(config)):
        raise StorageIntegrityError("frozen LUT manifest is not bound to the execution config")

    grid, beam, _prop, ion, _heat, _run, _raman = load_all(str(config))
    table_cfg = _ion_rate_table_defaults(ion)
    if (table_cfg.get("enabled") is not True or table_cfg.get("reuse_cache") is not True
            or table_cfg.get("force_rebuild") is not False
            or str(table_cfg.get("cache_dir", "")) != "cache/rate_tables"):
        raise StorageIntegrityError("formal frozen LUT config must enable relative cache reuse without rebuild")
    rate_table = manifest.get("rate_table")
    expected_rate_table = {name: table_cfg[name] for name in _FROZEN_LUT_RATE_FIELDS}
    if (not isinstance(rate_table, Mapping)
            or not _json_values_equal_with_types(dict(rate_table), expected_rate_table)):
        raise StorageIntegrityError("frozen LUT manifest rate-table configuration differs")

    species_rows = getattr(ion, "species", None)
    if not isinstance(species_rows, (list, tuple)):
        raise StorageIntegrityError("formal frozen LUT config lacks a species list")
    omega0 = 2.0 * np.pi * float(c0) / float(beam.lam0)
    n0 = float(beam.n0)
    expected_tables: list[dict[str, Any]] = []
    for raw_species in species_rows:
        if not isinstance(raw_species, Mapping):
            raise StorageIntegrityError("formal LUT species entry is invalid")
        species = dict(raw_species)
        rate = str(species.get("rate", "")).lower()
        if rate not in {"ppt_talebpour_i_lut", "popruzhenko_atom_i_lut"}:
            continue
        defaults = {
            "ppt_talebpour_i_lut": "ppt_talebpour_i_full_reference",
            "popruzhenko_atom_i_lut": "popruzhenko_atom_i_full_reference",
        }
        model = str(species.get("reference_model", "") or defaults[rate]).lower()
        species_name = str(species.get("name", "species"))
        metadata = _canonical_table_metadata(model, species_name, species, omega0, n0, table_cfg)
        signature = _table_signature(metadata)
        relative_path = _table_path("cache/rate_tables", species_name, signature).replace("\\", "/")
        expected_tables.append({
            "species_name": species_name, "model_name": model,
            "metadata": metadata, "signature": signature,
            "relative_path": relative_path, "array_names": list(_FROZEN_LUT_ARRAY_NAMES),
        })
    if not expected_tables:
        raise StorageIntegrityError("formal RATE_TABLE config has no LUT species")
    names = [str(item["species_name"]) for item in expected_tables]
    if len(set(names)) != len(names):
        raise StorageIntegrityError("formal LUT species names are not unique")
    expected_signatures = {item["species_name"]: item["signature"] for item in expected_tables}
    if not _json_values_equal_with_types(manifest.get("signature"), expected_signatures):
        raise StorageIntegrityError("frozen LUT signatures differ from the execution config")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(expected_tables):
        raise StorageIntegrityError("frozen LUT manifest file inventory is incomplete")
    rows_by_name: dict[str, Mapping[str, Any]] = {}
    for row in files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise StorageIntegrityError("frozen LUT file inventory row is invalid")
        filename = Path(str(row["path"])).name
        if filename in rows_by_name:
            raise StorageIntegrityError("frozen LUT manifest contains duplicate file names")
        rows_by_name[filename] = row

    validated: list[dict[str, Any]] = []
    for expected in expected_tables:
        filename = Path(expected["relative_path"]).name
        row = rows_by_name.get(filename)
        if (not isinstance(row, Mapping)
                or row.get("array_names") != _FROZEN_LUT_ARRAY_NAMES):
            raise StorageIntegrityError(f"frozen LUT inventory differs for {expected['species_name']}")
        source_path = _manifest_path_value(row.get("path"), manifest_path=manifest_path,
                                           label=f"frozen LUT source {filename}")
        if source_path.is_relative_to(root):
            raise StorageIntegrityError("frozen LUT authority source must be outside the campaign root")
        raw_size = row.get("size_bytes")
        if isinstance(raw_size, bool) or not isinstance(raw_size, Integral):
            raise StorageIntegrityError(f"frozen LUT size is invalid for {filename}")
        expected_size = int(raw_size)
        expected_sha = str(row.get("raw_sha256", ""))
        if (expected_size <= 0
                or len(expected_sha) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha)):
            raise StorageIntegrityError(f"frozen LUT raw identity is invalid for {filename}")
        source_identity = _lut_file_provenance(source_path)
        if (source_identity["size"] != expected_size or sha256_file(source_path) != expected_sha):
            raise StorageIntegrityError(f"frozen LUT authority source differs for {filename}")
        _inspect_frozen_lut_npz(
            source_path, expected_metadata=expected["metadata"],
            expected_species=expected["species_name"], expected_model=expected["model_name"],
        )
        validated.append({**expected, "source_path": source_path,
                          "source_identity": source_identity,
                          "size_bytes": expected_size, "raw_sha256": expected_sha})
    cache_dir = root / "cache" / "rate_tables"
    return {
        "config_path": config, "config_sha256": sha256_file(config),
        "manifest_path": manifest_path, "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest, "root": root, "cache_dir": cache_dir,
        "files": validated,
    }


def _lut_receipt_path(root: Path, requested: str | Path | None = None) -> Path:
    expected = (root / "cache" / "N5_LUT_STAGING_RECEIPT.json").resolve(strict=False)
    if requested is None:
        return expected
    value = Path(requested)
    if not value.is_absolute():
        value = root / value
    resolved = value.resolve(strict=False)
    if resolved != expected:
        raise StorageIntegrityError("frozen LUT receipt must be cache/N5_LUT_STAGING_RECEIPT.json")
    return expected


def _preflight_lut_destination(root: Path, candidate: str | Path, *, label: str) -> Path:
    path = Path(candidate)
    if ".." in path.parts:
        raise StorageIntegrityError(f"{label} traverses a parent")
    if not path.is_absolute():
        path = root / path
    lexical = Path(os.path.abspath(path))
    root_lexical = Path(os.path.abspath(root))
    try:
        relative = lexical.relative_to(root_lexical)
    except ValueError as error:
        raise StorageIntegrityError(f"{label} escapes the campaign root") from error
    if not relative.parts:
        raise StorageIntegrityError(f"{label} cannot be the campaign root")
    if any(_is_reparse(part) for part in (lexical, *lexical.parents)):
        raise StorageIntegrityError(f"{label} contains a symlink or reparse point")
    try:
        resolved = lexical.resolve(strict=False)
        resolved_root = root_lexical.resolve(strict=False)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise StorageIntegrityError(f"{label} resolves outside the campaign root") from error
    if resolved == resolved_root:
        raise StorageIntegrityError(f"{label} cannot resolve to the campaign root")
    return lexical


def _portable_lut_staging_identity(receipt: Mapping[str, Any], *, receipt_sha256: str,
                                   receipt_size_bytes: int) -> dict[str, Any]:
    files = []
    registered_inputs = []
    staged_bytes = 0
    for row in receipt.get("files", []):
        portable = {key: row[key] for key in (
            "species_name", "relative_path", "size_bytes", "raw_sha256",
            "signature", "metadata", "array_names",
        )}
        files.append(portable)
        staged_bytes += int(portable["size_bytes"])
        registered_inputs.append({
            "path": str(portable["relative_path"]), "role": "INPUT_LUT",
            "sha256": str(portable["raw_sha256"]), "size_bytes": int(portable["size_bytes"]),
            "input_kind": "frozen_rate_table_npz",
        })
    receipt_relative = "cache/N5_LUT_STAGING_RECEIPT.json"
    registered_inputs.append({
        "path": receipt_relative, "role": "INPUT_LUT", "sha256": str(receipt_sha256),
        "size_bytes": int(receipt_size_bytes), "input_kind": "frozen_lut_staging_receipt",
    })
    registered_inputs.sort(key=lambda item: item["path"])
    return {
        "schema": _FROZEN_LUT_STAGING_RECEIPT_SCHEMA,
        "config_sha256": str(receipt["config"]["raw_sha256"]),
        "authority_manifest_sha256": str(receipt["authority_manifest"]["raw_sha256"]),
        "receipt_path": receipt_relative,
        "receipt_sha256": str(receipt_sha256),
        "receipt_size_bytes": int(receipt_size_bytes),
        "files": files,
        "registered_inputs": registered_inputs,
        "staged_file_bytes": staged_bytes,
        "frozen_lut_input_bytes": staged_bytes + int(receipt_size_bytes),
    }


def validate_staged_frozen_lut_cache_hit(
    config_path: str | Path, campaign_root: str | Path, *,
    lut_identity: Mapping[str, Any], staging_receipt_path: str | Path | None = None,
    admission_identity_sha256: str | None = None, require_campaign_cwd: bool = False,
) -> dict[str, Any]:
    """Verify frozen inputs and prime only the exact validated runtime cache keys."""
    from .ionization.common import _ION_LUT_MEMORY_CACHE
    from .ionization.lut import _table_path

    root = _checked_campaign_root(campaign_root)
    if require_campaign_cwd and Path.cwd().resolve() != root:
        raise StorageIntegrityError("frozen LUT science worker cwd differs from campaign root")
    manifest_path = _checked_input_path(str(lut_identity.get("path", "")), label="LUT manifest")
    manifest_sha = str(lut_identity.get("sha256", ""))
    if not manifest_sha or sha256_file(manifest_path) != manifest_sha:
        raise StorageIntegrityError("frozen LUT manifest identity differs from admission")
    contract = _frozen_lut_contract(config_path, manifest_path, root)
    receipt_path = _lut_receipt_path(root, staging_receipt_path)
    receipt_path = _checked_input_path(receipt_path, label="frozen LUT staging receipt")
    if not receipt_path.is_relative_to(root):
        raise StorageIntegrityError("frozen LUT staging receipt escapes the campaign root")
    receipt = _read_manifest(receipt_path, label="frozen LUT staging receipt")
    expected_receipt_header = {
        "schema": _FROZEN_LUT_STAGING_RECEIPT_SCHEMA,
        "campaign_root": str(root),
        "config": {"path": str(contract["config_path"]), "raw_sha256": contract["config_sha256"]},
        "authority_manifest": {
            "path": str(contract["manifest_path"]),
            "raw_sha256": contract["manifest_sha256"],
            "schema": _FROZEN_LUT_MANIFEST_SCHEMA,
        },
    }
    for key, expected_value in expected_receipt_header.items():
        if not _json_values_equal_with_types(receipt.get(key), expected_value):
            raise StorageIntegrityError(f"frozen LUT receipt {key} binding differs")

    table_dir = contract["cache_dir"]
    if any(_is_reparse(part) for part in (table_dir, *table_dir.parents)) or not table_dir.is_dir():
        raise StorageIntegrityError("frozen LUT cache directory is missing or linked")
    expected_names = {Path(row["relative_path"]).name for row in contract["files"]}
    if {path.name for path in table_dir.iterdir() if path.suffix.lower() == ".npz"} != expected_names:
        raise StorageIntegrityError("frozen LUT cache contains missing or extra NPZ files")

    receipt_files = receipt.get("files")
    if not isinstance(receipt_files, list) or len(receipt_files) != len(contract["files"]):
        raise StorageIntegrityError("frozen LUT receipt file inventory is incomplete")
    tables_to_prime: list[tuple[str, str, dict[str, Any]]] = []
    for expected, receipt_file in zip(contract["files"], receipt_files):
        target = root / Path(expected["relative_path"])
        target_identity = _lut_file_provenance(target)
        source_identity = _lut_file_provenance(expected["source_path"])
        if (target_identity["size"] != expected["size_bytes"]
                or sha256_file(target) != expected["raw_sha256"]
                or (target_identity["st_dev"], target_identity["st_ino"])
                == (source_identity["st_dev"], source_identity["st_ino"])):
            raise StorageIntegrityError(f"staged frozen LUT identity differs for {target.name}")
        if not isinstance(receipt_file, Mapping):
            raise StorageIntegrityError("frozen LUT receipt row is invalid")
        actual = {
            "species_name": expected["species_name"],
            "relative_path": expected["relative_path"],
            "filename": target.name,
            "source_path": str(expected["source_path"]),
            "source_identity": source_identity,
            "target_identity": target_identity,
            "size_bytes": int(expected["size_bytes"]),
            "raw_sha256": expected["raw_sha256"],
            "signature": expected["signature"],
            "metadata": expected["metadata"],
            "array_names": expected["array_names"],
        }
        portable_keys = (
            "species_name", "relative_path", "filename", "source_path", "size_bytes",
            "raw_sha256", "signature", "metadata", "array_names",
        )
        if (set(receipt_file) != {*portable_keys, "source_identity", "target_identity"}
                or any(not isinstance(receipt_file.get(name), Mapping)
                       for name in ("source_identity", "target_identity"))
                or not _json_values_equal_with_types(
                    {key: receipt_file.get(key) for key in portable_keys},
                    {key: actual[key] for key in portable_keys},
                )):
            raise StorageIntegrityError(f"frozen LUT receipt provenance differs for {target.name}")
        for field_name, expected_size in (("source_identity", expected["size_bytes"]),
                                          ("target_identity", expected["size_bytes"])):
            provenance = receipt_file[field_name]
            if (provenance.get("st_nlink") != 1
                    or provenance.get("size") != expected_size
                    or any(isinstance(provenance.get(key), bool)
                           or not isinstance(provenance.get(key), int)
                           for key in ("st_dev", "st_ino", "st_nlink", "size", "mtime_ns"))):
                raise StorageIntegrityError(f"frozen LUT receipt file provenance is invalid for {target.name}")
        table = _inspect_frozen_lut_npz(
            target, expected_metadata=expected["metadata"],
            expected_species=expected["species_name"], expected_model=expected["model_name"],
        )
        cache_key = (_table_path("cache/rate_tables", expected["species_name"], expected["signature"]),
                     expected["signature"])
        tables_to_prime.append((cache_key[0], cache_key[1], table))

    if (set(receipt) != {*expected_receipt_header, "files"}
            or not isinstance(receipt.get("files"), list)):
        raise StorageIntegrityError("frozen LUT staging receipt contains unbound fields")
    receipt_identity = _lut_file_provenance(receipt_path)
    receipt_sha = sha256_file(receipt_path)
    portable = _portable_lut_staging_identity(
        receipt, receipt_sha256=receipt_sha, receipt_size_bytes=receipt_identity["size"])
    expected_staging = lut_identity.get("staging")
    if (expected_staging is not None
            and not _json_values_equal_with_types(expected_staging, portable)):
        raise StorageIntegrityError("frozen LUT staging identity differs from admission")
    for cache_path, signature, table in tables_to_prime:
        _ION_LUT_MEMORY_CACHE[(cache_path, signature)] = table
    return {
        "portable_identity": portable,
        "registered_inputs": list(portable["registered_inputs"]),
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "receipt_size_bytes": int(receipt_identity["size"]),
        "staged_file_bytes": int(portable["staged_file_bytes"]),
        "frozen_lut_input_bytes": int(portable["frozen_lut_input_bytes"]),
        "admission_identity_sha256": admission_identity_sha256,
    }


def stage_frozen_lut_inputs(
    config_path: str | Path, lut_manifest_path: str | Path, campaign_root: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy frozen source tables into the campaign cache and bind a receipt."""
    contract = _frozen_lut_contract(config_path, lut_manifest_path, campaign_root)
    root = contract["root"]
    table_dir = contract["cache_dir"]
    receipt_candidate = root / "cache" / "N5_LUT_STAGING_RECEIPT.json"
    _preflight_lut_destination(root, table_dir, label="frozen LUT table directory")
    _preflight_lut_destination(root, receipt_candidate, label="frozen LUT receipt")
    if receipt_path is not None:
        requested_receipt = Path(receipt_path)
        if not requested_receipt.is_absolute():
            requested_receipt = root / requested_receipt
        _preflight_lut_destination(root, requested_receipt, label="requested frozen LUT receipt")
    target_receipt = _lut_receipt_path(root, receipt_path)
    _preflight_lut_destination(root, target_receipt, label="frozen LUT receipt")
    targets = []
    for expected in contract["files"]:
        target = root / Path(expected["relative_path"])
        _preflight_lut_destination(root, target, label=f"frozen LUT destination {target.name}")
        targets.append((expected, target))

    table_dir.mkdir(parents=True, exist_ok=True)
    for expected, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with expected["source_path"].open("rb") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
                target_handle.flush()
                os.fsync(target_handle.fileno())
        if (sha256_file(target) != expected["raw_sha256"]
                or int(target.stat().st_size) != expected["size_bytes"]):
            raise StorageIntegrityError(f"frozen LUT destination differs for {target.name}")
        _lut_file_provenance(target)

    expected_names = {Path(row["relative_path"]).name for row in contract["files"]}
    if {path.name for path in table_dir.iterdir() if path.suffix.lower() == ".npz"} != expected_names:
        raise StorageIntegrityError("frozen LUT cache contains missing or extra NPZ files")
    files = []
    for expected in contract["files"]:
        target = root / Path(expected["relative_path"])
        files.append({
            "species_name": expected["species_name"],
            "relative_path": expected["relative_path"], "filename": target.name,
            "source_path": str(expected["source_path"]),
            "source_identity": _lut_file_provenance(expected["source_path"]),
            "target_identity": _lut_file_provenance(target),
            "size_bytes": int(expected["size_bytes"]), "raw_sha256": expected["raw_sha256"],
            "signature": expected["signature"], "metadata": expected["metadata"],
            "array_names": expected["array_names"],
        })
    receipt = {
        "schema": _FROZEN_LUT_STAGING_RECEIPT_SCHEMA,
        "campaign_root": str(root),
        "config": {"path": str(contract["config_path"]), "raw_sha256": contract["config_sha256"]},
        "authority_manifest": {
            "path": str(contract["manifest_path"]), "raw_sha256": contract["manifest_sha256"],
            "schema": _FROZEN_LUT_MANIFEST_SCHEMA,
        },
        "files": files,
    }
    if any(_is_reparse(part) for part in (target_receipt, *target_receipt.parents)):
        raise StorageIntegrityError("frozen LUT receipt destination contains a symlink or reparse point")
    target_receipt.parent.mkdir(parents=True, exist_ok=True)
    base_identity = {
        "path": str(contract["manifest_path"]),
        "sha256": contract["manifest_sha256"],
        **contract["manifest"],
    }
    if target_receipt.exists():
        return validate_staged_frozen_lut_cache_hit(
            contract["config_path"], root, lut_identity=base_identity,
            staging_receipt_path=target_receipt,
        )
    else:
        atomic_json(target_receipt, receipt, overwrite=False)
    return validate_staged_frozen_lut_cache_hit(
        contract["config_path"], root, lut_identity=base_identity,
        staging_receipt_path=target_receipt,
    )


def _source_hash(config_path: Path) -> str:
    # Compatibility helper for legacy CPU-only fixtures.  GPU/formal paths
    # bind a pre-materialized canonical artifact instead of rebuilding here.
    from .hr4e5_canonical_source import construct_canonical_source
    return sha256_array(construct_canonical_source(config_path))


def _schedule_from_config(prop: Any) -> Any:
    return build_longitudinal_schedule(
        dz=float(prop.dz), z_max=float(prop.z_max),
        focus_window_step=bool(getattr(prop, "focus_window_step", False)),
        focus_center_m=getattr(prop, "focus_center_m", None),
        focus_halfwidth_m=float(getattr(prop, "focus_halfwidth_m", 0.0)),
        dz_focus=getattr(prop, "dz_focus", None),
    )


def _positive_integer_pulse_count(value: Any, *, label: str = "n_pulses") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _json_values_equal_with_types(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_values_equal_with_types(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_equal_with_types(a, b) for a, b in zip(left, right)
        )
    return left == right


def describe_e5_1b_input_contract(full_config_path: str | Path,
                                  *, prefix_intervals: int = 8048,
                                  n_pulses: int = 2) -> dict[str, Any]:
    """Describe the official input range without opening any scientific array."""
    requested_n_pulses = _positive_integer_pulse_count(n_pulses)
    full_config = _checked_input_path(full_config_path, label="full source config")
    grid, beam, prop, ion, heat, run, raman = load_all(str(full_config))
    if _positive_integer_pulse_count(
            getattr(run, "Npulses", None), label="full source config Npulses") != 1:
        raise ValueError("full source config must have Npulses=1")
    full = _schedule_from_config(prop)
    prefix = build_prefix_schedule(full, prefix_intervals)
    return {
        "schema": "khz_filament.hr4e5.e5_1b.input_contract.v1",
        "status": "INPUT_NOT_MATERIALIZED",
        "full_config_path": str(full_config), "full_config_raw_sha256": sha256_file(full_config),
        "derived_config_rule": f"change_run.Npulses_from_1_to_{requested_n_pulses}_only",
        "full_interval_count": full.n_intervals, "prefix_interval_count": prefix.n_intervals,
        "source_indices": [0, prefix.n_intervals - 1], "block_size": BLOCK_SIZE,
        "grid_shape": [int(grid.Ny), int(grid.Nx)],
        "full_edges_array_sha256": sha256_array(np.asarray(full.z_edges, dtype=np.float64)),
        "prefix_edges_array_sha256": sha256_array(np.asarray(prefix.z_edges, dtype=np.float64)),
        "full_midpoints_array_sha256": sha256_array(np.asarray(
            [(left + right) / 2 for left, right in zip(full.z_edges[:-1], full.z_edges[1:])],
            dtype=np.float64)),
        "prefix_end_m": float(prefix.z_end),
        "optical_source_recipe": "build_transverse_input_field_then_optional_thin_lens_fresh_each_pulse",
        "slow_prefix_recipe": "full_delta_n_indices_0_through_8047_no_rebuild",
        "velocity_recipe": "positive_float64_zero_runtime_only",
        "slow_prefix_raw_sha256": "NOT_MATERIALIZED",
        "slow_prefix_array_sha256": "NOT_MATERIALIZED",
        "pre0_r_root_sha256": "NOT_MATERIALIZED",
        "pre0_c_root_sha256": "NOT_MATERIALIZED",
        "formal_admission": "NOT_RUN", "formal_execution_authorized": False,
        "resource_execution_gate": "NOT_RELEASED",
    }


def derive_e5_1b_execution_config(full_config_path: str | Path,
                                   output_path: str | Path, *,
                                   n_pulses: int = 2) -> dict[str, Any]:
    """Write an execution config changing only Npulses from the full source."""
    requested_n_pulses = _positive_integer_pulse_count(n_pulses)
    source = _checked_input_path(full_config_path, label="full source config")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("run"), dict):
        raise ValueError("full source config lacks run settings")
    if _positive_integer_pulse_count(
            value["run"].get("Npulses", -1), label="full source config Npulses") != 1:
        raise ValueError("full source config must have Npulses=1")
    value["run"]["Npulses"] = requested_n_pulses
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _bound_configs(source, destination, requested_n_pulses)
    return {"parent_path": str(source), "parent_raw_sha256": sha256_file(source),
            "derived_path": str(destination), "derived_raw_sha256": sha256_file(destination),
            "changed_field": "run.Npulses", "value": requested_n_pulses}


def _bound_configs(full_config: Path, config: Path, n_pulses: int) -> dict[str, Any]:
    requested_n_pulses = _positive_integer_pulse_count(n_pulses)
    checked_full_config = _checked_input_path(full_config, label="full source config")
    checked_config = _checked_input_path(config, label="execution config")
    parent = json.loads(checked_full_config.read_text(encoding="utf-8"))
    derived = json.loads(checked_config.read_text(encoding="utf-8"))
    if not isinstance(parent, dict) or not isinstance(derived, dict):
        raise ValueError("production configs must be JSON objects")
    parent_run, derived_run = parent.get("run"), derived.get("run")
    if not isinstance(parent_run, dict) or not isinstance(derived_run, dict):
        raise ValueError("production configs lack run settings")
    parent_n_pulses = _positive_integer_pulse_count(
        parent_run.get("Npulses", -1), label="full source config Npulses")
    derived_n_pulses = _positive_integer_pulse_count(
        derived_run.get("Npulses", -1), label="derived config Npulses")
    if derived_n_pulses != requested_n_pulses:
        raise ValueError("derived config pulse count differs from production spec")
    if checked_full_config != checked_config:
        if parent_n_pulses != 1:
            raise ValueError("formal config derivation must change Npulses from 1 to requested value")
        expected = json.loads(json.dumps(parent))
        expected["run"]["Npulses"] = requested_n_pulses
        if not _json_values_equal_with_types(expected, derived):
            raise ValueError("derived config changes fields besides run.Npulses")
    return {"path": str(full_config), "sha256": sha256_file(checked_full_config)}


def _slow_prefix_identity(manifest_path: Path, pre0_path: Path, full_config: Path,
                          full_schedule: Any, schedule: Any, grid: Any) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path, label="slow source")
    if manifest.get("schema") != "khz_filament.hr4e5.e5_1b.slow_source.v1":
        raise ValueError("slow source manifest schema is invalid")
    source_path = _checked_input_path(str(manifest.get("source_path", "")), label="full slow source")
    if str(manifest.get("source_config_sha256", "")) != sha256_file(full_config):
        raise ValueError("slow source config identity differs from full config")
    full_edges_hash = sha256_array(np.asarray(full_schedule.z_edges, dtype=np.float64))
    full_midpoints_hash = sha256_array(np.asarray(
        [(left + right) / 2 for left, right in zip(
            full_schedule.z_edges[:-1], full_schedule.z_edges[1:])], dtype=np.float64))
    if (manifest.get("axis_order") != ["interval", "y", "x"]
            or manifest.get("source_indices") != [0, full_schedule.n_intervals - 1]
            or str(manifest.get("full_edges_array_sha256", "")) != full_edges_hash
            or str(manifest.get("full_midpoints_array_sha256", "")) != full_midpoints_hash):
        raise ValueError("slow source axes or coordinates differ from full schedule")
    if str(manifest.get("source_raw_sha256", "")) != sha256_file(source_path):
        raise ValueError("full slow source raw identity differs from manifest")
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    prefix = np.load(pre0_path, mmap_mode="r", allow_pickle=False)
    expected_tail = (int(grid.Ny), int(grid.Nx))
    if (source.ndim != 3 or source.dtype != np.float64 or
            source.shape != (full_schedule.n_intervals, *expected_tail)):
        raise ValueError("full slow source shape or dtype is invalid")
    if (prefix.shape != (schedule.n_intervals, *expected_tail) or
            prefix.dtype != np.float64):
        raise ValueError("slow prefix shape or dtype is invalid")
    if list(manifest.get("source_shape", [])) != list(source.shape):
        raise ValueError("full slow source shape differs from manifest")
    if str(manifest.get("source_array_sha256", "")) != sha256_array_streaming(source):
        raise ValueError("full slow source array identity differs from manifest")
    # Compare one screen at a time: the official prefix is never allocated here.
    for index in range(schedule.n_intervals):
        if not np.array_equal(source[index], prefix[index]):
            raise ValueError(f"slow prefix differs from full source at index {index}")
    return {
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "full_source_path": str(source_path),
        "full_source_raw_bytes": source_path.stat().st_size,
        "full_source_raw_sha256": str(manifest["source_raw_sha256"]),
        "full_source_array_sha256": str(manifest.get("source_array_sha256", "")),
        "source_shape": list(source.shape), "source_indices": [0, schedule.n_intervals - 1],
        "axis_order": ["interval", "y", "x"],
        "full_edges_array_sha256": full_edges_hash,
        "full_midpoints_array_sha256": full_midpoints_hash,
        "prefix_path": str(pre0_path), "prefix_raw_sha256": sha256_file(pre0_path),
        "prefix_array_sha256": sha256_array_streaming(prefix),
        "zero_velocity_recipe": "positive_float64_zero",
        "capacity_domain": str(manifest.get("capacity_domain", "campaign_filesystem")),
    }


def _build_admission(spec: E5AProductionSpec, *,
                     implementation_identity: Mapping[str, Any] | None = None
                     ) -> tuple[dict[str, Any], Any, tuple[Any, ...]]:
    level = str(spec.admission_level)
    if level not in _ADMISSION_LEVELS:
        raise ValueError("unsupported production admission level")
    requested_n_pulses = _positive_integer_pulse_count(spec.n_pulses)
    config = _checked_input_path(spec.config_path, label="execution config")
    full_config = (_checked_input_path(spec.full_config_path, label="full source config")
                   if spec.full_config_path is not None else config)
    if level == FORMAL_SITE_ADMISSION and full_config == config:
        raise ValueError("formal site admission requires distinct full source and derived execution configs")
    source_path = _checked_input_path(spec.source_manifest_path, label="optical source manifest")
    lut_path = _checked_input_path(spec.lut_manifest_path, label="LUT manifest")
    pre0_path = _checked_input_path(spec.pre0_delta_n_path, label="PRE0 source")
    full_config_identity = _bound_configs(full_config, config, requested_n_pulses)
    source = _read_manifest(source_path, label="source")
    lut = _read_manifest(lut_path, label="LUT")
    components = load_all(str(config))
    grid, beam, prop, ion, heat, run, raman = components
    full_components = load_all(str(full_config))
    full_schedule = _schedule_from_config(full_components[2])
    requested_k = full_schedule.n_intervals if spec.prefix_intervals is None else int(spec.prefix_intervals)
    schedule = build_prefix_schedule(full_schedule, requested_k)
    if level == FORMAL_SITE_ADMISSION and (
            full_schedule.n_intervals != 15000 or spec.prefix_intervals != 8048):
        raise ValueError("formal source and derived config require full K15000 and prefix K8048")
    if schedule.n_intervals % BLOCK_SIZE or schedule.n_intervals <= 0:
        raise ValueError("production schedule requires complete blocks of eight")
    pre0 = np.load(pre0_path, mmap_mode="r", allow_pickle=False)
    if pre0.shape != (schedule.n_intervals, grid.Ny, grid.Nx) or pre0.dtype != np.float64:
        raise ValueError("PRE0 delta_n shape or dtype differs from production config")
    canonical_source = source.get("canonical_source")
    requires_canonical_artifact = bool(
        spec.require_gpu_backend or level == FORMAL_SITE_ADMISSION)
    if isinstance(canonical_source, Mapping):
        if source.get("schema") != CANONICAL_SOURCE_MANIFEST_SCHEMA:
            raise ValueError("canonical source manifest schema is invalid")
        if source.get("hash_scope") != CANONICAL_SOURCE_HASH_SCOPE:
            raise ValueError("canonical source manifest hash_scope is invalid")
        if (source.get("canonical_source_artifact_path") != canonical_source.get("artifact_path")
                or source.get("canonical_source_sealed_manifest_path")
                != canonical_source.get("sealed_manifest_path")):
            raise ValueError("source.json canonical artifact path bindings differ")
        canonical_source = validate_canonical_source(
            canonical_source, config_path=config, source_manifest_path=source_path,
        )
        source_hash = str(canonical_source["canonical_array_hash"])
        if str(source.get("canonical_array_hash", "")) != source_hash:
            raise ValueError("source manifest array identity differs from canonical E_source")
        campaign_root = _checked_campaign_root(spec.root)
        canonical_input_paths = (
            source_path,
            Path(str(canonical_source["artifact_path"])).resolve(strict=False),
            Path(str(canonical_source["sealed_manifest_path"])).resolve(strict=False),
        )
        if any(path.is_relative_to(campaign_root) for path in canonical_input_paths):
            raise ValueError("canonical source manifest, artifact, and seal must be outside the campaign root")
    else:
        if requires_canonical_artifact:
            raise ValueError("GPU/formal admission requires the sealed canonical E_source artifact")
        source_hash = _source_hash(config)
        if str(source.get("canonical_array_hash", "")) != source_hash:
            raise ValueError("legacy CPU source manifest differs from reconstructed input field")
    lut_staging = None
    base_lut_identity = {"path": str(lut_path), "sha256": sha256_file(lut_path), **lut}
    frozen_lut_input_bytes = 0
    if lut.get("mode") == "DISABLED_BY_CONFIG":
        if getattr(ion, "species", None) not in ([], ()) or bool(getattr(prop, "use_ionization_solver", False)):
            raise ValueError("LUT cannot be disabled while ionization is enabled")
        if str(lut.get("config_sha256", "")) != sha256_file(config):
            raise ValueError("disabled LUT manifest is not bound to config")
    elif level == FORMAL_SITE_ADMISSION and lut.get("mode") != "RATE_TABLE":
        raise ValueError("formal admission requires a disabled or frozen RATE_TABLE LUT manifest")
    elif not lut.get("signature"):
        raise ValueError("production LUT manifest lacks a signature")
    elif level == FORMAL_SITE_ADMISSION:
        if spec.lut_staging_receipt_path is None:
            raise ValueError("formal enabled LUT admission requires its staging receipt")
        lut_staging = validate_staged_frozen_lut_cache_hit(
            config, spec.root, lut_identity=base_lut_identity,
            staging_receipt_path=spec.lut_staging_receipt_path,
        )
        frozen_lut_input_bytes = int(lut_staging["frozen_lut_input_bytes"])
    effective_n_pulses = _positive_integer_pulse_count(
        getattr(run, "Npulses", requested_n_pulses), label="execution config Npulses")
    if effective_n_pulses != requested_n_pulses:
        raise ValueError("production pulse count differs from effective config")
    if int(spec.candidate_hydro_workers) not in {1, 2, 4}:
        raise ValueError("Candidate hydro workers must be 1, 2, or 4")
    slow_source_identity = None
    if spec.slow_source_manifest_path is not None:
        slow_source_identity = _slow_prefix_identity(
            _checked_input_path(spec.slow_source_manifest_path, label="slow source manifest"), pre0_path,
            full_config, full_schedule, schedule, grid,
        )
    elif level == FORMAL_SITE_ADMISSION or schedule.n_intervals != full_schedule.n_intervals:
        raise ValueError("formal or shortened prefix requires bound full slow source")
    budget_plan = plan_campaign_budget(
        n_pulses=requested_n_pulses, k=schedule.n_intervals,
        ny=int(grid.Ny), nx=int(grid.Nx), nt=int(grid.Nt),
        max_campaign_live_bytes=int(spec.max_campaign_live_bytes),
        final_output_budget_bytes=int(spec.final_output_budget_bytes),
        safety_margin_bytes=int(spec.safety_margin_bytes),
        external_prefix_already_present=True,
        existing_full_source_bytes=int((slow_source_identity or {}).get("full_source_raw_bytes", 0)),
        canonical_source_artifact_bytes=(
            int(canonical_source["file_size_bytes"]) if isinstance(canonical_source, Mapping) else 0),
        canonical_source_manifest_allowance_bytes=(
            CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES
            if isinstance(canonical_source, Mapping) else 0),
        canonical_source_write_temporary_bytes=(
            max(int(canonical_source["file_size_bytes"]),
                CANONICAL_SOURCE_MANIFEST_ALLOWANCE_BYTES)
            if isinstance(canonical_source, Mapping) else 0),
        frozen_lut_input_bytes=frozen_lut_input_bytes,
    )
    if level == FORMAL_SITE_ADMISSION:
        if spec.site_resource_manifest_path is None:
            raise ValueError("formal site admission requires qualified resource evidence")
        site_path = _checked_input_path(spec.site_resource_manifest_path, label="site resource manifest")
        site = _read_manifest(site_path, label="site resource")
        if (site.get("scope") != "SITE_QUALIFIED_RESOURCE_EVIDENCE"
                or site.get("formal_input_authorized") is not True
                or site.get("site_resource_qualified") is not True
                or site.get("provenance_qualified") is not True):
            raise ValueError("formal site admission evidence is incomplete")
        for name in ("free_bytes", "quota_bytes"):
            if isinstance(site.get(name), bool) or int(site.get(name, 0)) <= 0:
                raise ValueError(f"formal site admission {name} is invalid")
        if (str(site.get("capacity_domain", "")) != "campaign_filesystem"
                or str((slow_source_identity or {}).get("capacity_domain", "")) != "campaign_filesystem"):
            raise ValueError("formal prefix and campaign capacity domains are not qualified together")
        if not isinstance(site.get("capacity_root"), str) or not site["capacity_root"]:
            raise ValueError("formal capacity root is not declared")
        capacity_root = _checked_input_path(site["capacity_root"], label="formal capacity root", directory=True)
        campaign_root = _checked_campaign_root(spec.root)
        if Path(str(slow_source_identity["full_source_path"])).is_relative_to(campaign_root):
            raise ValueError("pre-existing full slow source must be outside the campaign root")
        bound_paths = [campaign_root, pre0_path,
                       Path(str(slow_source_identity["full_source_path"]))]
        if isinstance(canonical_source, Mapping):
            bound_paths.extend((source_path, Path(str(canonical_source["artifact_path"])),
                                Path(str(canonical_source["sealed_manifest_path"]))))
        if any(not path.is_relative_to(capacity_root) for path in bound_paths):
            raise ValueError("formal inputs and campaign do not share the qualified capacity root")
        existing_parent = campaign_root
        while not existing_parent.exists():
            existing_parent = existing_parent.parent
        volume = capacity_root.stat().st_dev
        for path in (*bound_paths[1:], existing_parent):
            existing_path = path
            while not existing_path.exists():
                existing_path = existing_path.parent
            if existing_path.stat().st_dev != volume:
                raise ValueError("formal inputs and campaign occupy different filesystems")
        resource_evidence = {**site, "path": str(site_path), "sha256": sha256_file(site_path)}
    else:
        if spec.site_resource_manifest_path is not None:
            raise ValueError("local qualification cannot claim site resource evidence")
        resource_evidence = {"scope": _LocalPolicyCapacityProvider.scope}
    runtime_or_compatibility = {
        "module": "KHz_filament.hr4e5_production",
        "runner": "single-production-runner",
        "science_child_timeout_s": int(spec.science_child_timeout_s),
    }
    if spec.storage_lock_timeout_s is not None:
        runtime_or_compatibility[STORAGE_LOCK_TIMEOUT_KEY] = float(spec.storage_lock_timeout_s)
    identity = build_admission_identity(
        campaign_id=str(spec.campaign_id), execution_mode=level,
        runtime_or_compatibility=runtime_or_compatibility,
        config_identity={"path": str(config), "sha256": sha256_file(config),
                         "full_source_config": full_config_identity,
                         "derivation": f"run.Npulses={requested_n_pulses}_only"
                         if full_config != config else "same_config"},
        effective_params={"f_rep": float(heat.f_rep), "dt_hydro": float(getattr(heat, "dt_hydro", 1e-7))},
        source_identity={"path": str(source_path), "sha256": sha256_file(source_path),
                         "canonical_array_hash": source_hash,
                         **({"canonical_source": dict(canonical_source)}
                            if isinstance(canonical_source, Mapping) else {})},
        lut_identity={**base_lut_identity,
                      **({"staging": dict(lut_staging["portable_identity"])}
                         if lut_staging is not None else {})},
        schedule_identity={"k": schedule.n_intervals, "metadata": schedule.as_metadata(),
                           "full_metadata": full_schedule.as_metadata(),
                           "source_indices": [0, schedule.n_intervals - 1],
                           "recipe": "build_full_then_prefix"},
        grid_identity={"Nx": grid.Nx, "Ny": grid.Ny, "Nt": grid.Nt,
                       "shape": [grid.Ny, grid.Nx], "dtype": "complex128"},
        pre0_identity={"path": str(pre0_path), "sha256": sha256_file(pre0_path),
                       "canonical_array_hash": sha256_array_streaming(pre0),
                       "slow_source_identity": slow_source_identity or {"scope": "LOCAL_FULL_SCHEDULE_FIXTURE"}},
        n_pulses=requested_n_pulses, k=int(schedule.n_intervals), block_size=BLOCK_SIZE,
        queue_depth=QUEUE_DEPTH, f_rep=float(heat.f_rep),
        hydro_topology={"candidate_optical_gpus_requested": 1,
                         "candidate_hydro_workers_requested": int(spec.candidate_hydro_workers),
                         "candidate_hydro_processes_requested": int(spec.candidate_hydro_workers),
                         "candidate_hydro_gpus_requested": int(spec.candidate_hydro_workers),
                         "candidate_hydro_local_process_qualification": "PENDING",
                         "candidate_hydro_physical_gpu_binding_qualification": "PENDING_HPC",
                         "reference_gpus_requested": 1,
                         "reference_backend": "HR4CThreeFieldStore+evolve_hr4_full_z",
                         "primary_topology_requested": {"reference_gpus": 1, "candidate_optical_gpus": 1, "candidate_hydro_gpus": 4},
                         "fallback_topology_requested": {"reference_gpus": 1, "candidate_optical_gpus": 1, "candidate_hydro_gpus": 2},
                         "allowed_values": [1, 2, 4], "gpu_topology_qualified": False},
        dt_hydro=float(getattr(heat, "dt_hydro", 1e-7)),
        precision={"fields": "float64", "optical": "complex128"},
        r_roots={"root": "R"}, c_roots={"root": "C"},
        pulse_attempt_epoch={"pulse": 0, "attempt": 0, "epoch": "factory-assigned"},
        budget_identity={"max_campaign_live_bytes": int(spec.max_campaign_live_bytes),
                          "final_output_budget_bytes": int(spec.final_output_budget_bytes),
                          "safety_margin_bytes": int(spec.safety_margin_bytes),
                          "object_schema": budget_plan["object_schema"],
                          "peak_bytes": budget_plan["peak_bytes"],
                          "peak_stage": budget_plan["peak_stage"],
                          "final_output_bytes": budget_plan["final_output_bytes"],
                          "headroom_bytes": budget_plan["headroom_bytes"],
                          "final_headroom_bytes": budget_plan["final_headroom_bytes"],
                           "external_prefix_bytes": (0 if pre0_path.is_relative_to(_checked_campaign_root(spec.root))
                                                     else int(pre0_path.stat().st_size)),
                           "prefix_owned_bytes": int(pre0_path.stat().st_size),
                           "prefix_payload_bytes": budget_plan["external_prefix_bytes"],
                           "prefix_inside_campaign_root": pre0_path.is_relative_to(_checked_campaign_root(spec.root)),
                          "existing_full_source_bytes": budget_plan["existing_full_source_bytes"],
                          "canonical_source_artifact_bytes": budget_plan["canonical_source_artifact_bytes"],
                          "canonical_source_manifest_allowance_bytes": budget_plan[
                              "canonical_source_manifest_allowance_bytes"],
                          "canonical_source_write_temporary_bytes": budget_plan[
                              "canonical_source_write_temporary_bytes"],
                          **({"frozen_lut_input_bytes": frozen_lut_input_bytes}
                             if frozen_lut_input_bytes else {}),
                          "incremental_requirement_bytes": budget_plan["incremental_requirement_bytes"]},
        retention_identity={"policy": "E5_1A_MINIMUM_ENGINEERING_RETENTION_V1"},
        resource_evidence=resource_evidence,
        **({"implementation_identity_sha256": str(implementation_identity["identity_sha256"])}
           if implementation_identity is not None else {}),
    )
    return identity, schedule, components


class _E5AProductionRunner:
    """Concrete runner; no callbacks or alternate test implementation exist."""

    def __init__(self, spec: E5AProductionSpec, identity: Mapping[str, Any], schedule: Any,
                 components: tuple[Any, ...], implementation_identity: Mapping[str, Any]):
        self.spec, self.root = spec, _checked_campaign_root(spec.root)
        self.identity, self.schedule, self.components = dict(identity), schedule, components
        self.implementation_identity = dict(implementation_identity)
        self.driver = None
        self.require_gpu_backend = bool(spec.require_gpu_backend or spec.admission_level == FORMAL_SITE_ADMISSION)
        self._gpu_assignment_paths: dict[int, Path] = {}
        self._pair_cancel_event: threading.Event | None = None
        self._rc_failure_lock = threading.Lock()

    def set_pair_cancellation_event(self, event: threading.Event | None) -> None:
        self._pair_cancel_event = event

    def _record_concurrent_rc_failure(
        self, *, pulse: int, step: str, error: BaseException,
        details: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Persist a small failure receipt outside the storage ledger path."""
        try:
            epoch = str(self.driver.campaign.active_coordinator_epoch())
            folder = self.root.parent / f"{self.root.name}_failure_audit" / f"p{int(pulse)}" / f"epoch_{epoch}"
            path = folder / "RC_FAILURE.json"
            payload = {
                "schema": "khz_filament.hr4e5.rc_failure_audit.v1",
                "status": "FAIL",
                "campaign_id": str(self.identity.get("campaign_id", "")),
                "pulse": int(pulse),
                "trajectory": "R" if step == "reference" else "C",
                "step": str(step),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                "details": dict(details or {}),
                "coordinator_process_identity": process_identity(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "monotonic_s": time.monotonic(),
            }
            with self._rc_failure_lock:
                if path.exists():
                    return str(path)
                atomic_json(path, payload, overwrite=False)
            return str(path)
        except Exception as audit_error:
            try:
                error.add_note(f"failure audit could not be persisted: {type(audit_error).__name__}: {audit_error}")
            except Exception:
                pass
            return None

    def _validate_frozen_lut_inputs(self, *, require_campaign_cwd: bool) -> dict[str, Any] | None:
        lut_identity = self.identity.get("lut_identity", {})
        staging = lut_identity.get("staging") if isinstance(lut_identity, Mapping) else None
        mode = lut_identity.get("mode") if isinstance(lut_identity, Mapping) else None
        if not isinstance(staging, Mapping) and self.spec.lut_staging_receipt_path is None:
            if self.spec.admission_level == FORMAL_SITE_ADMISSION and mode == "RATE_TABLE":
                raise StorageIntegrityError("formal science worker lacks admitted frozen LUT inputs")
            return None
        if not isinstance(lut_identity, Mapping):
            raise StorageIntegrityError("admission identity lacks frozen LUT bindings")
        receipt_path = self.spec.lut_staging_receipt_path
        if receipt_path is None and isinstance(staging, Mapping):
            receipt_path = str(staging.get("receipt_path", ""))
        return validate_staged_frozen_lut_cache_hit(
            self.spec.config_path, self.root, lut_identity=lut_identity,
            staging_receipt_path=receipt_path,
            admission_identity_sha256=str(self.identity.get("identity_sha256", "")),
            require_campaign_cwd=require_campaign_cwd,
        )

    def _gpu_audit_root(self, pulse: int) -> Path:
        epoch = str(self.driver.campaign.active_coordinator_epoch())
        if not epoch.isdecimal():
            raise StorageIntegrityError("GPU audit path requires a numeric coordinator epoch")
        return self.root.parent / f"{self.root.name}_gpu_audit" / f"p{int(pulse)}" / f"epoch_{epoch}"

    def prepare_gpu_assignment(self, pulse: int) -> None:
        """Persist one scheduler-visible role contract before R/C launch."""
        if not self.require_gpu_backend:
            return
        from datetime import datetime, timezone

        count = 2 + int(self.spec.candidate_hydro_workers)
        contract = build_assignment_record(os.environ.get("CUDA_VISIBLE_DEVICES"), count)
        pair = self.driver.campaign._pair(int(pulse)).get("scientific_pair_identity")
        if not isinstance(pair, Mapping):
            raise StorageIntegrityError("GPU assignment lacks a bound scientific pair")
        contract.update({
            "campaign_id": self.identity["campaign_id"],
            "scientific_pair_id": str(pair["scientific_pair_id"]),
            "coordinator_epoch": str(self.driver.campaign.active_coordinator_epoch()),
            "implementation_identity_sha256": self.implementation_identity["identity_sha256"],
            "scheduler_job_id": str(os.environ["SLURM_JOB_ID"]),
            "assignment_source": "SLURM_CUDA_VISIBLE_DEVICES",
            "assigned_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })
        audit = self._gpu_audit_root(pulse)
        path = audit / "GPU_ROLE_ASSIGNMENT.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            validate_assignment_record(saved, count)
            for name in ("campaign_id", "scientific_pair_id", "coordinator_epoch",
                         "implementation_identity_sha256", "scheduler_job_id"):
                if saved.get(name) != contract[name]:
                    raise StorageIntegrityError("persisted GPU role assignment differs")
            contract = saved
        else:
            audit.mkdir(parents=True, exist_ok=True)
            atomic_json(path, contract, overwrite=False)
        self._gpu_assignment_paths[int(pulse)] = path

    def _gpu_role_environment(self, *, pulse: int, role: str) -> dict[str, str]:
        path = self._gpu_assignment_paths.get(int(pulse))
        if path is None or not path.is_file():
            raise StorageIntegrityError("GPU role assignment was not persisted before child launch")
        contract = json.loads(path.read_text(encoding="utf-8"))
        validate_assignment_record(contract, 2 + int(self.spec.candidate_hydro_workers))
        if (contract.get("coordinator_epoch") != str(self.driver.campaign.active_coordinator_epoch())
                or contract.get("implementation_identity_sha256") != self.implementation_identity["identity_sha256"]):
            raise StorageIntegrityError("GPU role assignment epoch or implementation changed")
        token = contract["role_to_token"].get(role)
        if not token:
            raise StorageIntegrityError("GPU role is not present in the assignment")
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": token,
            "UPPE_USE_GPU": "1",
            "HR4E5_GPU_ROLE": role,
            "HR4E5_GPU_ASSIGNED_TOKEN": token,
            "HR4E5_SCHEDULER_VISIBLE_DEVICES": contract["scheduler_visible_devices"],
            "HR4E5_GPU_AUDIT_ROOT": str(self._gpu_audit_root(pulse)),
            "HR4E5_CAMPAIGN_ROOT": str(self.root),
            "HR4E5_CAMPAIGN_ID": self.identity["campaign_id"],
            "HR4E5_SCIENTIFIC_PAIR_ID": contract["scientific_pair_id"],
            "HR4E5_COORDINATOR_EPOCH": contract["coordinator_epoch"],
            "HR4E5_IMPLEMENTATION_IDENTITY_SHA256": self.implementation_identity["identity_sha256"],
        })
        return environment

    def validate_gpu_pulse_receipts(self, pulse: int, *, saved_reference_epoch: str | int | None = None,
                                    saved_candidate_epoch: str | int | None = None) -> None:
        """Check new processes and explicitly qualified cross-epoch reuse."""
        if not self.require_gpu_backend:
            return
        contract_path = self._gpu_assignment_paths.get(int(pulse))
        if contract_path is None:
            raise StorageIntegrityError("GPU role assignment is missing at the pair barrier")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        validate_assignment_record(contract, 2 + int(self.spec.candidate_hydro_workers))
        pair_id = contract["scientific_pair_id"]
        epoch = contract["coordinator_epoch"]
        final_pulse = int(pulse) == int(self.spec.n_pulses) - 1
        expected_roles = ["R", "C_OPT"] + ([] if final_pulse else [
            f"H{index}" for index in range(int(self.spec.candidate_hydro_workers))])
        starts, finals = [], []
        reused: dict[str, dict[str, Any]] = {}
        audit = self._gpu_audit_root(pulse)
        audit_parent = audit.parent
        for role in expected_roles:
            startup_path, final_path = audit / f"{role}.startup.json", audit / f"{role}.final.json"
            if startup_path.is_file() or final_path.is_file():
                if not startup_path.is_file() or not final_path.is_file():
                    raise StorageIntegrityError(f"GPU process receipt pair is incomplete for {role}")
                starts.append(json.loads(startup_path.read_text(encoding="utf-8")))
                finals.append(json.loads(final_path.read_text(encoding="utf-8")))
                continue
            source_epoch = (saved_reference_epoch if role == "R" else saved_candidate_epoch)
            if role == "C_OPT" and not final_pulse:
                active = validate_active_output(
                    campaign_root=self.root,
                    state_root=self._track("C", pulse) / "state",
                    scientific_pair_id=pair_id,
                    implementation_identity_sha256=contract["implementation_identity_sha256"],
                )
                source_epoch = active["qualified_attempt"]["producer_coordinator_epoch"]
            if source_epoch is None or str(source_epoch) == epoch or not str(source_epoch).isdecimal():
                raise StorageIntegrityError(f"GPU process receipts are missing for {role}")
            folder = audit_parent / f"epoch_{source_epoch}"
            old_start, old_final = folder / f"{role}.startup.json", folder / f"{role}.final.json"
            old_assignment = folder / "GPU_ROLE_ASSIGNMENT.json"
            if not old_start.is_file() or not old_final.is_file() or not old_assignment.is_file():
                raise StorageIntegrityError(f"qualified prior GPU receipts are missing for reused {role}")
            prior_contract = json.loads(old_assignment.read_text(encoding="utf-8"))
            validate_assignment_record(prior_contract, 2 + int(self.spec.candidate_hydro_workers))
            prior_start = json.loads(old_start.read_text(encoding="utf-8"))
            prior_final = json.loads(old_final.read_text(encoding="utf-8"))
            if (str(prior_contract.get("coordinator_epoch")) != str(source_epoch)
                    or str(prior_start.get("coordinator_epoch")) != str(source_epoch)
                    or prior_contract.get("scientific_pair_id") != pair_id
                    or prior_contract.get("implementation_identity_sha256") != contract["implementation_identity_sha256"]
                    or prior_contract.get("campaign_id") != self.identity["campaign_id"]
                    or prior_start.get("role") != role
                    or prior_start.get("assignment", {}).get("token") != prior_contract["role_to_token"][role]):
                raise StorageIntegrityError(f"reused GPU role is not bound to source epoch {source_epoch}")
            try:
                validate_startup_final_identities([prior_start], [prior_final])
            except ValueError as exc:
                raise StorageIntegrityError(f"reused GPU process receipt failed for {role}: {exc}") from exc
            reused[role] = {"epoch": str(source_epoch),
                            "pid": prior_start["identity"]["process"]["pid"],
                            "gpu_uuid": prior_start["identity"]["gpu"]["uuid"],
                            "assignment_path": str(old_assignment),
                            "startup_path": str(old_start), "final_path": str(old_final)}
        try:
            if starts:
                validate_startup_final_identities(starts, finals)
        except ValueError as exc:
            raise StorageIntegrityError(f"GPU role/process/device evidence failed: {exc}") from exc
        for receipt in starts:
            if (receipt["identity"]["process"]["ppid"] != os.getpid()
                    or receipt["scientific_pair_id"] != contract["scientific_pair_id"]
                    or receipt["coordinator_epoch"] != contract["coordinator_epoch"]
                    or receipt["implementation_identity_sha256"] != contract["implementation_identity_sha256"]):
                raise StorageIntegrityError("GPU child process or assignment identity differs")
        current_roles = {receipt["role"] for receipt in starts}
        if current_roles | reused.keys() != set(expected_roles):
            raise StorageIntegrityError("GPU pulse role inventory is incomplete")
        atomic_json(audit / "GPU_ROLE_RECEIPTS_VALIDATED.json", {
            "schema": "khz_filament.hr4e5_gpu_role_receipts_validated.v1",
            "status": "PASS", "campaign_id": self.identity["campaign_id"],
            "scientific_pair_id": contract["scientific_pair_id"],
            "coordinator_epoch": contract["coordinator_epoch"],
            "roles": expected_roles, "current_roles": sorted(current_roles),
            "reused_roles": reused,
            "role_to_pid": {r["role"]: r["identity"]["process"]["pid"] for r in starts},
            "role_to_gpu_uuid": {r["role"]: r["identity"]["gpu"]["uuid"] for r in starts},
        }, overwrite=False)

    @property
    def budget(self):
        if self.driver is None:
            raise RuntimeError("production driver is not attached")
        return self.driver.campaign.storage

    def expected_terminal_roles(self, pulse: int) -> dict[str, list[str]]:
        contract = self.root / "E5_1A_TERMINAL_ROLE_CONTRACT.json"
        if not contract.is_file():
            raise StorageBudgetError("terminal role contract is missing")
        value = json.loads(contract.read_text(encoding="utf-8"))
        if int(value.get("terminal_pulse", -1)) != int(pulse):
            raise StorageBudgetError("terminal role contract pulse changed")
        return {str(role): [str(path) for path in paths] for role, paths in value["roles"].items()}

    def _track(self, side: str, pulse: int) -> Path:
        return self.root / side / f"p{int(pulse)}"

    def scientific_pair_identity(self, pulse: int) -> dict[str, Any]:
        reference = StreamingLifecycle.open(self._track("R", pulse) / "state")
        candidate = StreamingLifecycle.open(self._track("C", pulse) / "state")
        return build_scientific_pair_identity(
            admission_identity=self.identity, pulse=int(pulse),
            reference_generation=str(reference.manifest["current_generation"]),
            candidate_generation=str(candidate.manifest["current_generation"]),
        )

    def _qualify_optical_output(self, *, side: str, pulse: int, state: Path,
                                output: Path, generation: str,
                                producer_id: str) -> dict[str, Any]:
        pair = self.driver.campaign._pair(pulse).get("scientific_pair_identity")
        if not isinstance(pair, Mapping):
            raise StorageIntegrityError("optical qualification lacks durable scientific pair identity")
        attempt = qualify_output_attempt(
            campaign_root=self.root, state_root=state, output_dir=output,
            attempt_id=f"{side}_p{pulse}_{producer_id}",
            campaign_id=self.identity["campaign_id"],
            scientific_pair_id=str(pair["scientific_pair_id"]),
            trajectory=side, pulse=pulse, generation=generation,
            coordinator_epoch=self.driver.campaign.active_coordinator_epoch(),
            writer_grant_id=producer_id,
            admission_identity_sha256=self.identity["identity_sha256"],
            implementation_identity_sha256=self.implementation_identity["identity_sha256"],
        )
        return activate_output_attempt(
            campaign_root=self.root, state_root=state,
            record_path=attempt["record_path"],
            expected_record_sha256=attempt["record_sha256"],
        )

    def _hydro(self) -> dict[str, Any]:
        grid, beam, prop, ion, heat, run, raman = self.components
        schedule = build_interpulse_step_schedule(f_rep=float(heat.f_rep), dt_hydro=float(heat.dt_hydro))
        if schedule.remainder_s != 0.0:
            raise ValueError("production hydro worker does not support a remainder step")
        return {"dt_hydro": float(heat.dt_hydro), "n_hydro_steps": int(schedule.full_step_count),
                "chi": float(heat.chi), "nu": float(heat.nu), "n0": float(beam.n0),
                "gravity_x": float(getattr(heat, "gravity_x", 0.0)),
                "gravity_y": float(getattr(heat, "gravity_y", -9.81)), "cfl_limit": 1.0}

    def _intent(self, *, side: str, pulse: int, role: str, paths: list[Path],
                expected_bytes: int, generation: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        epoch = self.driver.campaign.active_coordinator_epoch()
        child_science = role == "OPTICAL" and side == "C" and pulse < self.spec.n_pulses - 1
        epoch_rebindable = child_science or (side == "R" and role in {"OPTICAL", "REFERENCE_HYDRO"})
        pair = (self.driver.campaign._pair(pulse).get("scientific_pair_identity")
                if epoch_rebindable else None)
        if epoch_rebindable and not isinstance(pair, Mapping):
            raise StorageIntegrityError("epoch-rebindable production intent lacks scientific pair identity")
        if epoch_rebindable or role in {"gc_evidence", "TERMINAL_READY", "TERMINAL_EVIDENCE", "exact_report"}:
            wanted_paths = {path.relative_to(self.root).as_posix() for path in paths}
            prior = [item for item in self.budget.intents().values()
                     if item.get("role") == role and item.get("trajectory") == side
                     and int(item.get("pulse", -1)) == int(pulse)
                     and item.get("generation") == generation
                     and item.get("status") in {"ACTIVE", "INTERRUPTED"}]
            if len(prior) > 1:
                raise StorageIntegrityError("multiple resumable production intents")
            if prior:
                intent = prior[0]
                if (set(intent.get("allowed_paths", [])) != wanted_paths
                        or int(intent.get("expected_bytes", -1)) != int(expected_bytes)
                        or int(intent.get("attempt", -1)) != 0
                        or str(intent.get("admission_hash")) != self.identity["identity_sha256"]):
                    raise StorageIntegrityError("resumable production intent scope changed")
                record = self.budget.reservations().get(str(intent["reservation_id"]))
                if not isinstance(record, Mapping) or record.get("status") != "ACTIVE":
                    raise StorageIntegrityError("resumable production reservation is missing")
                reservation = Reservation(str(record["reservation_id"]), int(record["bytes"]),
                                          str(record["purpose"]), str(record["owner"]),
                                          str(record["created_utc"]))
                if epoch_rebindable:
                    intent = self.budget.rebind_intent_epoch(
                        intent["intent_id"], reservation_id=reservation.reservation_id,
                        previous_coordinator_epoch=intent["epoch"], coordinator_epoch=epoch,
                        trajectory=side, pulse=pulse, logical_attempt=0,
                        generation=generation, allowed_paths=paths,
                        expected_bytes=expected_bytes,
                        admission_identity_sha256=self.identity["identity_sha256"],
                        scientific_pair_id=pair["scientific_pair_id"],
                        implementation_identity=self.implementation_identity,
                    )
                opener = self.budget.open_coordinator_reconciler if child_science else self.budget.open_writer
                writer = opener(
                    reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
                    coordinator_epoch=epoch, trajectory=side, pulse=int(pulse), attempt=0,
                    generation=generation,
                )
                return reservation, intent, writer
        reservation = self.budget.reserve(expected_bytes, purpose=f"{role}:{side}:p{pulse}",
                                          owner=str(epoch), allocation_paths=paths)
        intent_metadata = None
        if epoch_rebindable:
            intent_metadata = {
                "scientific_pair_id": pair["scientific_pair_id"],
                "implementation_identity": self.implementation_identity,
            }
            if side == "R" and role == "REFERENCE_HYDRO":
                intent_metadata["reservation_owner_epoch"] = str(epoch)
        intent = self.budget.create_intent(
            reservation_id=reservation.reservation_id, trajectory=side, pulse=int(pulse), attempt=0,
            role=role, allowed_paths=paths, expected_bytes=expected_bytes,
            admission_hash=self.identity["identity_sha256"], epoch=epoch, generation=generation,
            campaign_id=self.identity["campaign_id"],
            metadata=intent_metadata,
        )
        opener = self.budget.open_coordinator_reconciler if child_science else self.budget.open_writer
        writer = opener(
            reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
            coordinator_epoch=epoch, trajectory=side, pulse=int(pulse), attempt=0,
            generation=generation,
        )
        return reservation, intent, writer

    def _hydro_worker_command(self, state: Path, pulse: int, worker_index: int) -> list[str]:
        return [
            sys.executable, "-s", "-B", "-m", "KHz_filament.hr4e5_production_worker",
            "--lifecycle-root", str(state), "--pulse", str(int(pulse)),
            "--worker-index", str(int(worker_index)), "--hydro-workers", str(int(self.spec.candidate_hydro_workers)),
            "--config-path", str(self.spec.config_path),
            "--admission-identity-sha256", self.identity["identity_sha256"],
        ]

    def _launch_hydro_workers(self, state: Path, pulse: int) -> list[subprocess.Popen]:
        processes: list[subprocess.Popen] = []
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        environment["PYTHONNOUSERSITE"] = "1"
        environment["HR4E5S_WORKER_ID"] = "pending"
        for worker_index in range(int(self.spec.candidate_hydro_workers)):
            environment["HR4E5S_WORKER_ID"] = f"e5_1b_c_hydro_{pulse}_{worker_index}"
            process = subprocess.Popen(
                self._hydro_worker_command(state, pulse, worker_index),
                cwd=str(self.root), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            processes.append(process)
        return processes

    def _join_hydro_workers(self, processes: Sequence[subprocess.Popen]) -> None:
        if not processes:
            return
        failures: list[str] = []
        for process in processes:
            try:
                stdout, stderr = process.communicate(timeout=600)
            except subprocess.TimeoutExpired as error:
                process.kill(); stdout, stderr = process.communicate()
                failures.append(f"pid={process.pid}:hydro worker timeout")
                continue
            if process.returncode != 0:
                failures.append(f"pid={process.pid}:exit={process.returncode}:stderr={stderr.strip()}")
        if failures:
            raise RuntimeError("production hydro worker failed: " + "; ".join(failures))

    def _close_writer(self, writer: Mapping[str, Any], *, failed: BaseException | None = None) -> None:
        self.budget.close_writer(writer["writer_id"], coordinator_epoch=writer["coordinator_epoch"],
                                 status="INTERRUPTED" if failed else "CLOSED",
                                 reason=None if failed is None else f"{type(failed).__name__}: {failed}")

    def initialize(self) -> None:
        admission_report = self._track("R", 0) / "reports" / "admission.json"
        if not admission_report.is_file():
            capacity_snapshot = {
                "scope": self.identity["resource_evidence"]["scope"],
                "filesystem_free_bytes": int(shutil.disk_usage(self.root).free),
                "policy_cap_bytes": int(self.spec.max_campaign_live_bytes),
                "site_quota_measured": self.spec.admission_level == FORMAL_SITE_ADMISSION,
            }
            self._simple_report(
                side="R", pulse=0, name="admission",
                summary=np.asarray([self.spec.n_pulses, self.schedule.n_intervals], dtype=np.int64),
                extras={
                    "admission_identity_sha256": self.identity["identity_sha256"],
                    "config_identity": self.identity["config_identity"],
                    "source_identity": self.identity["source_identity"],
                    "lut_identity": self.identity["lut_identity"],
                    "resource_evidence": self.identity["resource_evidence"],
                    "capacity_snapshot": capacity_snapshot,
                },
            )
        grid = self.components[0]
        pre0_path = _checked_input_path(self.spec.pre0_delta_n_path, label="PRE0 source")
        if sha256_file(pre0_path) != self.identity["pre0_identity"]["sha256"]:
            raise ValueError("PRE0 raw file differs from admission identity")
        delta_n = np.load(pre0_path, allow_pickle=False)
        estimate = max(1024**2, int(delta_n.nbytes) * 4)
        for side in ("R", "C"):
            target = self._track(side, 0) / "state"
            if (target / "streaming_manifest.json").is_file():
                continue
            reservation, intent, writer = self._intent(
                side=side, pulse=0, role="PRE0", paths=[target], expected_bytes=estimate,
                generation=f"{side}:p0:PRE0",
            )
            error = None
            try:
                create_pre0_root(
                    root=target, delta_n=delta_n, schedule=self.schedule,
                    dx_m=float(grid.Lx / grid.Nx), dy_m=float(grid.Ly / grid.Ny),
                    current_generation=f"{side}:p0:PRE0", source_identity=self.identity["source_identity"],
                    admission_identity=self.identity, storage_budget=self.budget,
                    creation_intent=intent, trajectory=side, pulse=0, role="PRE0",
                )
            except BaseException as exc:
                error = exc; raise
            finally:
                self._close_writer(writer, failed=error)

    def _simple_report(self, *, side: str, pulse: int, name: str, summary: np.ndarray,
                       extras: Mapping[str, Any] | None = None) -> dict[str, Any]:
        folder = self._track(side, pulse) / "reports"
        array_path, report_path = folder / f"{name}_summary.npy", folder / f"{name}.json"
        if report_path.exists():
            relative = report_path.relative_to(self.root).as_posix()
            owned = self.budget.artifacts().get(relative)
            if not isinstance(owned, Mapping):
                raise StorageIntegrityError("existing production report has no completed ownership")
            _checked_owned_file(self.root, report_path, owned, label="existing production report")
            saved = validate_durable_report(
                report_path, expected_sha256=str(owned["sha256"]),
                campaign_id=self.identity["campaign_id"], root=self.root,
                pulse=int(pulse), trajectory=side,
            )
            if any(saved.get(key) != value for key, value in dict(extras or {}).items()):
                raise StorageIntegrityError("existing production report differs from retry")
            return saved
        reservation, intent, writer = self._intent(
            side=side, pulse=pulse, role=f"report_{name}", paths=[folder], expected_bytes=4*1024**2,
            generation=f"{side}:p{pulse}:{name}",
        )
        error = None
        try:
            folder.mkdir(parents=True, exist_ok=True); np.save(array_path, np.asarray(summary))
            rows = build_expected_object_set(
                {name: array_path}, campaign_id=self.identity["campaign_id"], trajectory=side,
                pulse=pulse, attempt=0, namespace="POST", root=self.root,
                source_indices={name: 0}, role=f"report_{name}",
            )
            result = bind_exact_report(report_path, {"status": "PASS", "mismatch_count": 0,
                                                          **dict(extras or {})}, rows,
                                       campaign_id=self.identity["campaign_id"], root=self.root)
            completed = self.budget.complete_intent(intent["intent_id"], files=[array_path, report_path],
                                                     metadata={"reclaimable": False})
            self.budget.consume(reservation.reservation_id, actual_bytes=completed["actual_bytes"])
            return result
        except BaseException as exc:
            error = exc
            try: self.budget.interrupt_intent(intent["intent_id"], reason=str(exc))
            except Exception: pass
            raise
        finally:
            self._close_writer(writer, failed=error)

    # Scientific step implementations are deliberately methods of this sealed
    # class.  They are filled by the production lifecycle below, never supplied
    # by a caller.
    def run_reference(self, pulse: int):
        if self.require_gpu_backend or self.spec.reference_subprocess:
            return self._run_reference_subprocess(pulse)
        return self._run_pulse("R", pulse)

    def _run_reference_subprocess(self, pulse: int, *, trajectory: str = "R") -> dict[str, Any]:
        pair_cancel_event = getattr(self, "_pair_cancel_event", None)
        if pair_cancel_event is not None and pair_cancel_event.is_set():
            raise RuntimeError(f"{trajectory} subprocess cancelled after paired worker failure")
        self._validate_frozen_lut_inputs(require_campaign_cwd=False)
        if trajectory not in {"R", "C"}:
            raise ValueError("independent science process trajectory must be R or C")
        if trajectory == "C" and int(pulse) != int(self.spec.n_pulses) - 1:
            raise ValueError("single-process Candidate path is only valid for the final pulse")
        role = "R" if trajectory == "R" else "C_OPT"
        label = "reference" if trajectory == "R" else "candidate_final"
        audit = self._gpu_audit_root(pulse) if self.require_gpu_backend else (
            self.root.parent / f"{self.root.name}_{label}_process_audit" / f"p{int(pulse)}"
            / f"epoch_{self.driver.campaign.active_coordinator_epoch()}"
        )
        audit.mkdir(parents=True, exist_ok=True)
        epoch = str(self.driver.campaign.active_coordinator_epoch())
        result_path = audit / f"{role}_epoch_{epoch}.result.json"
        spec_json = json.dumps(asdict(self.spec), default=str, sort_keys=True)
        command = [
            sys.executable, "-s", "-B", "-m", "KHz_filament.hr4e5_production_reference_worker",
            "--spec-json", spec_json, "--pulse", str(int(pulse)),
            "--trajectory", trajectory,
            "--result-path", str(result_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        environment["PYTHONNOUSERSITE"] = "1"
        if self.require_gpu_backend:
            environment.update(self._gpu_role_environment(pulse=pulse, role=role))
            command = self._gpu_bootstrap_command(command)
        else:
            environment["UPPE_USE_GPU"] = "0"
        stdout_path = audit / f"{role}.stdout.txt"
        stderr_path = audit / f"{role}.stderr.txt"
        with stdout_path.open("x", encoding="utf-8") as stdout_file, stderr_path.open("x", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                command, cwd=str(self.root), env=environment,
                stdout=stdout_file, stderr=stderr_file, text=True,
            )
        try:
            self._wait_science_child(process, role=role)
            stderr = stderr_path.read_text(encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"{role} subprocess failed: pid={process.pid}:exit={process.returncode}:stderr={stderr.strip()}")
            if not result_path.is_file():
                raise StorageIntegrityError(f"{role} subprocess produced no durable result")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (result.get("schema") != "khz_filament.hr4e5.reference_process_result.v1"
                    or result.get("campaign_id") != self.identity["campaign_id"]
                    or result.get("trajectory") != trajectory
                    or int(result.get("pulse", -1)) != int(pulse)
                    or result.get("coordinator_epoch") != epoch
                    or int(result.get("process_identity", {}).get("pid", -1)) != process.pid
                    or int(result.get("parent_pid", -1)) != os.getpid()):
                raise StorageIntegrityError(f"{role} subprocess result identity differs")
            return result["result"]
        except BaseException as error:
            self._record_concurrent_rc_failure(
                pulse=pulse, step="reference" if trajectory == "R" else "candidate",
                error=error,
                details={"role": role, "pid": int(process.pid), "stderr_path": str(stderr_path)},
            )
            raise

    def _wait_science_child(self, process: subprocess.Popen, *, role: str) -> None:
        timeout_s = int(self.spec.science_child_timeout_s)
        cancel_event = getattr(self, "_pair_cancel_event", None)

        def stop_child_and_raise(error: BaseException, *, graceful: bool = False,
                                 cause: BaseException | None = None) -> None:
            try:
                needs_kill = not graceful
                if graceful:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        # A natural exit won the race. Let the caller inspect
                        # its real return code and stderr instead of inventing cancellation.
                        process.wait(timeout=5)
                        return
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        needs_kill = True
                if needs_kill:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            except BaseException as cleanup_error:
                error.add_note(
                    f"owned child cleanup failed: pid={process.pid}: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise error from cause

        if cancel_event is None:
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                stop_child_and_raise(
                    TimeoutError(f"{role} subprocess exceeded {timeout_s} seconds"), cause=exc,
                )
            return
        deadline = time.monotonic() + timeout_s
        while process.poll() is None:
            if cancel_event.is_set():
                if process.poll() is not None:
                    return
                stop_child_and_raise(
                    RuntimeError(f"{role} subprocess cancelled after paired worker failure"),
                    graceful=True,
                )
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                stop_child_and_raise(TimeoutError(f"{role} subprocess exceeded {timeout_s} seconds"))
            try:
                process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue

    @staticmethod
    def _gpu_bootstrap_command(command: Sequence[str]) -> list[str]:
        if len(command) < 5 or command[0] != sys.executable or command[3] != "-m":
            raise ValueError("GPU bootstrap requires a controlled Python module command")
        return [
            sys.executable, "-s", "-B", "-m", "KHz_filament.hr4e5_gpu_bootstrap",
            "--module", str(command[4]), "--", *map(str, command[5:]),
        ]

    def run_candidate(self, pulse: int):
        if int(pulse) == int(self.spec.n_pulses) - 1 and (
                self.require_gpu_backend or self.spec.reference_subprocess):
            return self._run_reference_subprocess(pulse, trajectory="C")
        return self._run_pulse("C", pulse)

    def run_exact(self, pulse: int):
        return self._run_exact(pulse)

    def run_successor(self, pulse: int, trajectory: str):
        return self._run_successor(trajectory, pulse)

    def run_gc(self, pulse: int, trajectory: str):
        return self._run_gc(trajectory, pulse)

    def run_terminal(self, pulse: int):
        return self._run_terminal(pulse)

    def _hydro_worker_command(self, state: Path, pulse: int, worker_index: int,
                               *, intent_id: str | None = None,
                               reservation_id: str | None = None,
                               writer_id: str | None = None,
                               grant: Mapping[str, Any] | None = None,
                               scientific_pair_id: str | None = None,
                               producer_grant_id: str | None = None) -> list[str]:
        command = [
            sys.executable, "-s", "-B", "-m", "KHz_filament.hr4e5_production_worker",
            "--lifecycle-root", str(state), "--pulse", str(int(pulse)),
            "--worker-index", str(int(worker_index)), "--hydro-workers", str(int(self.spec.candidate_hydro_workers)),
            "--config-path", str(self.spec.config_path),
            "--admission-identity-sha256", self.identity["identity_sha256"],
        ]
        if intent_id is not None:
            command.extend(["--storage-root", str(self.root), "--intent-id", str(intent_id)])
        if reservation_id is not None:
            command.extend(["--reservation-id", str(reservation_id)])
        if writer_id is not None:
            command.extend(["--writer-id", str(writer_id)])
        if grant is not None:
            command.extend([
                "--grant-id", str(grant["grant_id"]), f"--grant-nonce={grant['nonce']}",
                "--scientific-pair-id", str(scientific_pair_id),
                "--producer-grant-id", str(producer_grant_id),
                "--implementation-identity-sha256", self.implementation_identity["identity_sha256"],
            ])
        return command

    def _launch_hydro_workers(self, state: Path, pulse: int, *, intent_id: str,
                              reservation_id: str, writer_ids: Sequence[str]) -> list[subprocess.Popen]:
        processes: list[subprocess.Popen] = []
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        environment["PYTHONNOUSERSITE"] = "1"
        for worker_index, writer_id in enumerate(writer_ids):
            environment["HR4E5S_WORKER_ID"] = f"e5_1b_c_hydro_{pulse}_{worker_index}"
            process = subprocess.Popen(
                self._hydro_worker_command(
                    state, pulse, worker_index, intent_id=intent_id,
                    reservation_id=reservation_id, writer_id=writer_id,
                ),
                cwd=str(self.root), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            processes.append(process)
        return processes

    def _start_logged_child(self, command: Sequence[str], *, state: Path,
                            grant_id: str, environment: Mapping[str, str]) -> tuple[subprocess.Popen, Path]:
        """Spool child output so a full pipe cannot hold a finished writer alive."""
        folder = state / "process_logs"
        folder.mkdir(parents=True, exist_ok=True)
        stdout_path = folder / f"{grant_id}.stdout.txt"
        stderr_path = folder / f"{grant_id}.stderr.txt"
        with stdout_path.open("x", encoding="utf-8") as stdout_file, stderr_path.open("x", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                list(command), cwd=str(self.root), env=dict(environment),
                stdout=stdout_file, stderr=stderr_file, text=True,
            )
        return process, stderr_path

    def _join_hydro_workers(self, processes: Sequence[subprocess.Popen],
                            stderr_paths: Mapping[int, Path] | None = None) -> None:
        if not processes:
            return
        timeout_s = int(self.spec.science_child_timeout_s)
        failures: list[str] = []
        for worker_index, process in enumerate(processes):
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                failures.append(f"H{worker_index}:pid={process.pid}:hydro worker timeout after {timeout_s} seconds")
                try:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                except BaseException as cleanup_error:
                    failures.append(f"H{worker_index}:pid={process.pid}:cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
                continue
            if process.returncode != 0:
                stderr_path = (stderr_paths or {}).get(process.pid)
                stderr = stderr_path.read_text(encoding="utf-8") if stderr_path is not None else ""
                failures.append(f"H{worker_index}:pid={process.pid}:exit={process.returncode}:stderr={stderr.strip()}")
        if failures:
            raise RuntimeError("production hydro worker failed: " + "; ".join(failures))

    @staticmethod
    def _stop_hydro_workers(processes: Sequence[subprocess.Popen]) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 10.0
        for process in processes:
            if process.poll() is None and time.monotonic() < deadline:
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
        kill_targets = []
        for process in processes:
            if process.poll() is None:
                process.kill()
                kill_targets.append(process)
        unreaped_pids = []
        for process in kill_targets:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                unreaped_pids.append(int(process.pid))
        if unreaped_pids:
            raise RuntimeError(
                "production hydro worker cleanup could not reap pending pids="
                f"{sorted(set(unreaped_pids))} within 5 seconds"
            )

    def _wait_for_candidate_hydro_workers(
            self, hydro_processes: Sequence[subprocess.Popen], *,
            optical_process: subprocess.Popen | None,
            optical_stderr_path: Path | None,
            hydro_stderr_paths: Mapping[int, Path] | None = None,
            pulse: int | None = None) -> None:
        timeout_s = int(self.spec.science_child_timeout_s)
        deadline = time.monotonic() + timeout_s
        while any(process.poll() is None for process in hydro_processes):
            for worker_index, process in enumerate(hydro_processes):
                if process.poll() is not None and process.returncode != 0:
                    process.wait(timeout=5)
                    stderr_path = (hydro_stderr_paths or {}).get(process.pid)
                    stderr = stderr_path.read_text(encoding="utf-8") if stderr_path is not None else ""
                    error = RuntimeError(
                        f"Candidate hydro H{worker_index} failed: pid={process.pid} "
                        f"exit={process.returncode}:stderr={stderr.strip()}"
                    )
                    if pulse is not None:
                        self._record_concurrent_rc_failure(
                            pulse=pulse, step="candidate", error=error,
                            details={"role": f"H{worker_index}", "pid": int(process.pid),
                                     "exit_code": int(process.returncode),
                                     "stderr_path": None if stderr_path is None else str(stderr_path),
                                     "stderr": stderr},
                        )
                    raise error
            if optical_process is not None and optical_process.poll() is not None and optical_process.returncode != 0:
                optical_process.wait(timeout=5)
                stderr = optical_stderr_path.read_text(encoding="utf-8") if optical_stderr_path else ""
                error = RuntimeError(
                    f"Candidate optical worker failed: exit={optical_process.returncode}:stderr={stderr.strip()}"
                )
                if pulse is not None:
                    self._record_concurrent_rc_failure(
                        pulse=pulse, step="candidate", error=error,
                        details={"role": "C_OPT", "pid": int(optical_process.pid),
                                 "exit_code": int(optical_process.returncode),
                                 "stderr_path": None if optical_stderr_path is None else str(optical_stderr_path),
                                 "stderr": stderr},
                    )
                raise error
            cancel_event = getattr(self, "_pair_cancel_event", None)
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Candidate hydro workers cancelled after paired worker failure")
            if time.monotonic() >= deadline:
                pending_pids = sorted({int(process.pid) for process in hydro_processes
                                       if process.poll() is None})
                raise TimeoutError(
                    f"Candidate hydro workers exceeded {timeout_s} seconds; "
                    f"pending_pids={pending_pids}"
                )
            time.sleep(0.02)

    def _process_receipt(self, *, trajectory: str, pulse: int, role: str,
                         worker_index: int | None, lifecycle_root: Path,
                         identity: Mapping[str, Any], started_monotonic_s: float,
                         ended_monotonic_s: float, exit_status: str,
                         created_files: Sequence[str | Path] = ()) -> dict[str, Any]:
        folder = self._track(trajectory, pulse) / "state" / "process_receipts"
        suffix = role if worker_index is None else f"{role}_{int(worker_index):03d}"
        path = folder / f"{suffix}.json"
        payload = {
            "schema": PROCESS_RECEIPT_SCHEMA,
            "campaign_id": self.identity["campaign_id"],
            "pulse": int(pulse), "trajectory": str(trajectory), "role": str(role),
            "worker_index": None if worker_index is None else int(worker_index),
            "coordinator_epoch": str(self.driver.campaign.active_coordinator_epoch()),
            "coordinator_process_identity": process_identity(),
            "admission_identity_sha256": self.identity["identity_sha256"],
            "lifecycle_root": str(lifecycle_root.resolve()),
            "started_monotonic_s": float(started_monotonic_s),
            "ended_monotonic_s": float(ended_monotonic_s),
            "exit_status": str(exit_status),
            "created_files": [str(Path(value).resolve()) for value in created_files],
            "writer_id": str(identity.get("writer_id", "")),
            "intent_id": str(identity.get("intent_id", "")),
            "reservation_id": str(identity.get("reservation_id", "")),
            "backend": "numpy/cpu",
            "physical_gpu_identity": "NOT_APPLICABLE_LOCAL_CPU",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(path, payload, overwrite=True)
        return {**payload, "report_path": str(path), "report_sha256": sha256_file(path)}

    def _validate_process_receipt(self, receipt: Mapping[str, Any], *, trajectory: str,
                                   pulse: int, role: str,
                                   worker_index: int | None = None) -> None:
        if not isinstance(receipt, MappingABC):
            raise StorageIntegrityError("process receipt is invalid")
        expected = {
            "schema": PROCESS_RECEIPT_SCHEMA, "campaign_id": self.identity["campaign_id"],
            "pulse": int(pulse), "trajectory": trajectory, "role": role,
            "worker_index": worker_index,
            "coordinator_epoch": str(self.driver.campaign.active_coordinator_epoch()),
            "admission_identity_sha256": self.identity["identity_sha256"],
            "backend": "numpy/cpu", "physical_gpu_identity": "NOT_APPLICABLE_LOCAL_CPU",
            "exit_status": "PASS",
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise StorageIntegrityError(f"{role} process receipt identity differs")
        for key in ("pid", "process_start_token", "started_monotonic_s", "ended_monotonic_s"):
            if not receipt.get(key):
                raise StorageIntegrityError(f"{role} process receipt lacks {key}")
        if not receipt.get("report_path") or not receipt.get("report_sha256"):
            raise StorageIntegrityError(f"{role} process receipt lacks durable evidence")
        if sha256_file(receipt["report_path"]) != receipt["report_sha256"]:
            raise StorageIntegrityError(f"{role} process receipt evidence changed")

    def _run_pulse(self, trajectory: str, pulse: int):
        self._validate_frozen_lut_inputs(
            require_campaign_cwd=self.spec.admission_level == FORMAL_SITE_ADMISSION,
        )
        state = self._track(trajectory, pulse) / "state"
        output = self._track(trajectory, pulse) / "optical"
        lifecycle = StreamingLifecycle.open(state)
        grid = self.components[0]
        final = int(pulse) == int(self.spec.n_pulses) - 1
        estimate = optical_write_forecast(
            k=self.schedule.n_intervals, ny=grid.Ny, nx=grid.Nx, nt=grid.Nt,
            final=(final or trajectory == "R"),
        )
        if trajectory == "R" and (state / "ACTIVE_OUTPUT.json").is_file():
            pair = self.driver.campaign._pair(pulse).get("scientific_pair_identity")
            if not isinstance(pair, Mapping):
                raise StorageIntegrityError("Reference output reuse lacks scientific pair identity")
            active = validate_active_output(
                campaign_root=self.root, state_root=state,
                scientific_pair_id=str(pair["scientific_pair_id"]),
                implementation_identity_sha256=self.implementation_identity["identity_sha256"],
            )
            if (active["trajectory"] != "R" or int(active["pulse"]) != int(pulse)
                    or active["generation"] != str(lifecycle.manifest["current_generation"])
                    or active["admission_identity_sha256"] != self.identity["identity_sha256"]
                    or not Path(active["output_dir"]).is_relative_to(output)):
                raise StorageIntegrityError("qualified Reference output reuse scope differs")
            if any(item.get("trajectory") == "R" and int(item.get("pulse", -1)) == int(pulse)
                   for item in self.budget.active_writers(include_management=False)):
                raise StorageIntegrityError("Reference output reuse has an active scientific writer")
            records = lifecycle.manifest["records"]
            promotion = lifecycle.manifest.get("promotion")
            if not final and promotion is None:
                pointer_path = state / "authoritative_generation.json"
                if pointer_path.is_file():
                    if (lifecycle.manifest.get("barrier", {}).get("status") != "PASS"
                            or any(row.get("state") != "BARRIER_VALIDATED" or row.get("next") is None
                                   for row in records)
                            or lifecycle.manifest.get("queue")
                            or lifecycle.manifest.get("recovery_backlog")):
                        raise StorageIntegrityError("Reference promotion pointer lacks a quiescent PASS barrier")
                    pointer_sha256 = sha256_file(pointer_path)
                    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                    recovered_pointer = lifecycle.promote_next_to_current(
                        actor="hr4c_reference_promotion_recovery",
                    )
                    if (sha256_file(pointer_path) != pointer_sha256
                            or recovered_pointer != pointer
                            or lifecycle.manifest.get("promotion") != pointer):
                        raise StorageIntegrityError("Reference promotion recovery changed its durable pointer")
                    promotion = lifecycle.manifest.get("promotion")
                if promotion is None:
                    if (any(row.get("state") != "POST_COMMITTED" or row.get("next") is not None
                            for row in records)
                            or lifecycle.manifest.get("queue") or lifecycle.manifest.get("recovery_backlog")):
                        raise StorageIntegrityError("partial Reference hydro requires separate recovery qualification")
                    self._run_reference_hydro(state=state, pulse=pulse, lifecycle=lifecycle, grid=grid)
                else:
                    self._reconcile_reference_hydro_intent(state=state, pulse=pulse,
                                                           lifecycle=lifecycle, grid=grid)
            elif not final and (not isinstance(promotion, Mapping)
                                or promotion.get("authoritative_namespace") != "NEXT"
                                or lifecycle.manifest.get("barrier", {}).get("status") != "PASS"):
                raise StorageIntegrityError("Reference hydro promotion is not qualified")
            elif not final:
                self._reconcile_reference_hydro_intent(state=state, pulse=pulse,
                                                       lifecycle=lifecycle, grid=grid)
            return self._simple_report(
                side=trajectory, pulse=pulse, name="pulse",
                summary=np.asarray([pulse, self.schedule.n_intervals], dtype=np.int64),
            )
        reservation, intent, writer = self._intent(
            side=trajectory, pulse=pulse, role="OPTICAL", paths=[output, state],
            expected_bytes=estimate, generation=str(lifecycle.manifest["current_generation"]),
        )
        if trajectory == "C" and not final:
            return self._run_candidate_multipulse(
                state=state, output=output, pulse=pulse, reservation=reservation,
                intent=intent, writer=writer,
            )

        error: BaseException | None = None
        try:
            result = run_streaming_optical_pulse(
                lifecycle_root=state, schedule=self.schedule, output_dir=output,
                config_path=self.spec.config_path, final=(final or trajectory == "R"),
                resume=(trajectory == "R" and output.exists()),
                storage_budget=self.budget, admission_identity=self.identity,
                creation_intent=intent, trajectory=trajectory, pulse=pulse, role="OPTICAL",
                defer_intent_completion=False,
            )
            self._qualify_optical_output(
                side=trajectory, pulse=pulse, state=state, output=Path(str(result["output_dir"])),
                generation=str(lifecycle.manifest["current_generation"]),
                producer_id=str(writer["writer_id"]),
            )
            if trajectory == "R" and not final:
                # Reopen after the optical child writes; R hydro runs only
                # after its optical POST artifacts are durable.
                lifecycle = StreamingLifecycle.open(state)
                self._run_reference_hydro(
                    state=state, pulse=pulse, lifecycle=lifecycle, grid=grid,
                )
            return self._simple_report(
                side=trajectory, pulse=pulse, name="pulse",
                summary=np.asarray([pulse, self.schedule.n_intervals], dtype=np.int64),
            )
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._close_writer(writer, failed=error)

    def _reference_hydro_scope(self, *, state: Path, pulse: int, grid: Any):
        batch_dir = self._track("R", pulse) / "batch"
        screen_bytes = int(grid.Nx) * int(grid.Ny) * 8
        expected_bytes = max(4*1024**2, 9*self.schedule.n_intervals*screen_bytes)
        paths = [state, batch_dir]
        return batch_dir, paths, expected_bytes, f"R:p{pulse}:NEXT"

    def _reference_hydro_created_files(self, *, state: Path, batch_dir: Path,
                                       lifecycle: StreamingLifecycle,
                                       intent: Mapping[str, Any],
                                       artifacts: Mapping[str, Any] | None = None) -> list[Path]:
        initial = intent.get("initial_files")
        if not isinstance(initial, Mapping):
            raise StorageIntegrityError("Reference hydro intent lacks its initial file snapshot")
        state_rel = state.relative_to(self.root).as_posix()
        mutable_manifest = f"{state_rel}/streaming_manifest.json"
        for relative, snapshot in initial.items():
            if relative == mutable_manifest:
                if not (self.root / relative).is_file():
                    raise StorageIntegrityError("Reference hydro initial Streaming manifest is missing")
                continue
            path = self.root / str(relative)
            if (not isinstance(snapshot, Mapping) or not path.is_file() or _is_reparse(path)
                    or _file_identity(path) != snapshot.get("identity")
                    or sha256_file(path) != str(snapshot.get("sha256", ""))):
                raise StorageIntegrityError(f"Reference hydro initial file changed: {relative}")

        initial_paths = {str(path) for path in initial}
        batch_rel = batch_dir.relative_to(self.root).as_posix()
        created: dict[str, Path] = {}
        valid_store_suffixes = {"state_manifest.json"} | {
            f"{field}_{slot}.npy" for field in FIELDS for slot in ("current", "next")
        }
        if batch_dir.exists():
            for path in batch_dir.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.root).as_posix()
                if relative in initial_paths:
                    continue
                name = path.name
                marker = ".hr4c_"
                prefix, separator, suffix = name.partition(marker)
                if (path.parent != batch_dir or not separator
                        or not (prefix == "store" or prefix.startswith("store_replay_"))
                        or suffix not in valid_store_suffixes):
                    raise StorageIntegrityError(f"unrecognized new Reference hydro file: {relative}")
                if _is_reparse(path):
                    raise StorageIntegrityError(f"Reference hydro output is linked: {relative}")
                if artifacts is not None and relative in artifacts:
                    raise StorageIntegrityError(f"Reference hydro output already has registered ownership: {relative}")
                created[relative] = path

        records = lifecycle.manifest.get("records")
        if not isinstance(records, list) or len(records) != self.schedule.n_intervals:
            raise StorageIntegrityError("promoted Reference hydro record count differs from schedule")
        for ordinal, record in enumerate(records):
            entry = record.get("next") if isinstance(record, Mapping) else None
            relative = f"next/screen_{ordinal:06d}.npz"
            artifact_relative = (str(entry.get("artifact", "")).replace("\\", "/")
                                 if isinstance(entry, Mapping) else "")
            if not isinstance(entry, Mapping) or artifact_relative != relative:
                raise StorageIntegrityError("promoted Reference hydro NEXT locator differs")
            path = state / relative
            if (not path.is_file() or _is_reparse(path)
                    or sha256_file(path) != str(entry.get("file_sha256", ""))):
                raise StorageIntegrityError(f"promoted Reference hydro NEXT file changed: {relative}")
            campaign_relative = f"{state_rel}/{relative}"
            if campaign_relative in initial_paths:
                raise StorageIntegrityError("promoted NEXT file predates its Reference hydro intent")
            if artifacts is not None and campaign_relative in artifacts:
                raise StorageIntegrityError(
                    f"Reference hydro NEXT file already has registered ownership: {campaign_relative}"
                )
            created[campaign_relative] = path

        pointer_path = state / "authoritative_generation.json"
        pointer_relative = pointer_path.relative_to(self.root).as_posix()
        promotion = lifecycle.manifest.get("promotion")
        if (pointer_relative in initial_paths or not pointer_path.is_file() or _is_reparse(pointer_path)
                or json.loads(pointer_path.read_text(encoding="utf-8")) != promotion):
            raise StorageIntegrityError("Reference hydro promotion pointer is not owned by this intent")
        if artifacts is not None and pointer_relative in artifacts:
            raise StorageIntegrityError("Reference hydro promotion pointer already has registered ownership")
        created[pointer_relative] = pointer_path
        if not any(relative.startswith(batch_rel + "/") for relative in created):
            raise StorageIntegrityError("Reference hydro intent has no newly created HR-4C store files")
        return [created[key] for key in sorted(created)]

    def _reconcile_reference_hydro_intent(self, *, state: Path, pulse: int,
                                          lifecycle: StreamingLifecycle, grid: Any) -> None:
        batch_dir, paths, expected_bytes, generation = self._reference_hydro_scope(
            state=state, pulse=pulse, grid=grid,
        )
        pair = self.driver.campaign._pair(pulse).get("scientific_pair_identity")
        if not isinstance(pair, Mapping):
            raise StorageIntegrityError("promoted Reference hydro lacks scientific pair identity")
        all_intents = self.budget.intents()
        candidates = [item for item in all_intents.values()
                     if item.get("role") == "REFERENCE_HYDRO" and item.get("trajectory") == "R"
                     and int(item.get("pulse", -1)) == int(pulse)
                     and item.get("generation") == generation]
        if len(candidates) != 1:
            raise StorageIntegrityError("promoted Reference hydro requires exactly one original intent")
        intent = candidates[0]
        reservation_id = str(intent.get("reservation_id", ""))
        reservation = self.budget.reservations().get(reservation_id)
        wanted_paths = [path.relative_to(self.root).as_posix() for path in paths]
        metadata = intent.get("metadata")
        reservation_owner_epoch = (
            metadata.get("reservation_owner_epoch") if isinstance(metadata, Mapping) else None
        )
        if reservation_owner_epoch is None:
            reservation_owner_epoch = intent.get("takeover_from_epoch", intent.get("epoch"))
        if (intent.get("schema") != INTENT_SCHEMA or not intent.get("intent_id")
                or intent.get("campaign_id") != self.identity["campaign_id"]
                or intent.get("trajectory") != "R" or int(intent.get("pulse", -1)) != int(pulse)
                or int(intent.get("attempt", -1)) != 0 or intent.get("role") != "REFERENCE_HYDRO"
                or intent.get("generation") != generation
                or list(intent.get("allowed_paths", [])) != wanted_paths
                or int(intent.get("expected_bytes", -1)) != expected_bytes
                or intent.get("admission_hash") != self.identity["identity_sha256"]
                or not isinstance(metadata, Mapping)
                or metadata.get("scientific_pair_id") != pair.get("scientific_pair_id")
                or metadata.get("implementation_identity") != self.implementation_identity):
            raise StorageIntegrityError("promoted Reference hydro intent identity differs")
        if (not isinstance(reservation, Mapping) or reservation.get("reservation_id") != reservation_id
                or int(reservation.get("bytes", -1)) != expected_bytes
                or reservation.get("purpose") != f"REFERENCE_HYDRO:R:p{pulse}"
                or reservation.get("owner") != str(reservation_owner_epoch)
                or list(reservation.get("allocation_paths", [])) != wanted_paths):
            raise StorageIntegrityError("promoted Reference hydro original reservation identity differs")

        artifacts = self.budget.artifacts()
        if intent.get("status") == "COMPLETED":
            created = intent.get("created_files")
            expected_files = self._reference_hydro_created_files(
                state=state, batch_dir=batch_dir, lifecycle=lifecycle, intent=intent,
            )
            expected_relatives = {path.relative_to(self.root).as_posix() for path in expected_files}
            if not isinstance(created, Mapping) or set(created) != expected_relatives:
                raise StorageIntegrityError("completed Reference hydro intent file set differs")
            for relative, record in created.items():
                path = self.root / relative
                artifact = artifacts.get(relative)
                if (not isinstance(record, Mapping) or not path.is_file() or _is_reparse(path)
                        or _file_identity(path) != record.get("identity")
                        or sha256_file(path) != str(record.get("sha256", ""))
                        or not isinstance(artifact, Mapping)
                        or artifact.get("intent_id") != intent.get("intent_id")
                        or artifact.get("reservation_id") != reservation_id
                        or artifact.get("identity") != record.get("identity")
                        or artifact.get("sha256") != record.get("sha256")):
                    raise StorageIntegrityError(f"completed Reference hydro ownership changed: {relative}")
            actual_bytes = sum(
                int(record.get("identity", {}).get("size", -1))
                for record in created.values() if isinstance(record, Mapping)
            )
            if actual_bytes != int(intent.get("actual_bytes", -1)):
                raise StorageIntegrityError("completed Reference hydro byte total differs")
            if reservation.get("status") == "ACTIVE":
                self.budget.consume(reservation_id)
            elif reservation.get("status") != "CONSUMED":
                raise StorageIntegrityError("completed Reference hydro reservation is not active or consumed")
            return
        if intent.get("status") not in {"ACTIVE", "INTERRUPTED"} or reservation.get("status") != "ACTIVE":
            raise StorageIntegrityError("incomplete Reference hydro intent lacks its active reservation")
        active_for_intent = [item for item in self.budget.active_writers(include_management=True)
                             if str(item.get("intent_id")) == str(intent["intent_id"])]
        if active_for_intent:
            raise StorageIntegrityError("Reference hydro intent still has an active writer")
        created_files = self._reference_hydro_created_files(
            state=state, batch_dir=batch_dir, lifecycle=lifecycle, intent=intent,
            artifacts=artifacts,
        )
        epoch = self.driver.campaign.active_coordinator_epoch()
        if str(intent.get("epoch")) != str(epoch):
            intent = self.budget.rebind_intent_epoch(
                intent["intent_id"], reservation_id=reservation_id,
                previous_coordinator_epoch=intent["epoch"], coordinator_epoch=epoch,
                trajectory="R", pulse=pulse, logical_attempt=0, generation=generation,
                allowed_paths=paths, expected_bytes=expected_bytes,
                admission_identity_sha256=self.identity["identity_sha256"],
                scientific_pair_id=pair["scientific_pair_id"],
                implementation_identity=self.implementation_identity,
            )
        writer = self.budget.open_writer(
            reservation_id=reservation_id, intent_id=intent["intent_id"],
            coordinator_epoch=epoch, trajectory="R", pulse=pulse, attempt=0,
            generation=generation,
        )
        error: BaseException | None = None
        try:
            completed = self.budget.complete_intent(
                intent["intent_id"], files=created_files, metadata={"reclaimable": True},
            )
            if completed.get("status") != "COMPLETED":
                raise StorageIntegrityError("Reference hydro intent did not complete")
            self.budget.consume(reservation_id)
        except BaseException as exc:
            error = exc
            try:
                self.budget.interrupt_intent(intent["intent_id"], reason=str(exc))
            except Exception:
                pass
            raise
        finally:
            self._close_writer(writer, failed=error)

    def _run_reference_hydro(self, *, state: Path, pulse: int, lifecycle: StreamingLifecycle,
                             grid: Any) -> None:
        batch_dir, paths, expected_bytes, generation = self._reference_hydro_scope(
            state=state, pulse=pulse, grid=grid,
        )
        hydro_res, hydro_intent, hydro_writer = self._intent(
            side="R", pulse=pulse, role="REFERENCE_HYDRO", paths=paths,
            expected_bytes=expected_bytes, generation=generation,
        )
        hydro_error: BaseException | None = None
        post = {
            field: np.stack([
                lifecycle._artifact_fields(
                    lifecycle.manifest["records"][i]["post"], namespace="POST"
                )[field]
                for i in range(self.schedule.n_intervals)
            ])
            for field in FIELDS
        }
        try:
            batch_dir.mkdir(parents=True, exist_ok=True)
            existing_store_files = list(batch_dir.glob("store.hr4c_*"))
            store_output = batch_dir / "store"
            if existing_store_files:
                epoch = str(self.driver.campaign.active_coordinator_epoch())
                epoch_tag = "".join(char if char.isalnum() or char in "-_" else "_" for char in epoch)
                while True:
                    store_output = batch_dir / f"store_replay_{epoch_tag}_{time.time_ns()}"
                    root = store_output.with_suffix("")
                    paths_to_check = [root.with_name(root.name + ".hr4c_state_manifest.json")]
                    paths_to_check.extend(
                        root.with_name(root.name + f".hr4c_{field}_{slot}.npy")
                        for field in FIELDS for slot in ("current", "next")
                    )
                    if not any(path.exists() for path in paths_to_check):
                        break
            store = HR4CThreeFieldStore(
                output_path=str(store_output), n_intervals=self.schedule.n_intervals,
                shape=(grid.Ny, grid.Nx), dtype=np.float64, z_edges=self.schedule.z_edges,
                dx=float(grid.Lx/grid.Nx), dy=float(grid.Ly/grid.Ny),
                authoritative_metadata={"admission_identity_sha256": self.identity["identity_sha256"]},
            )
            try:
                store.begin_staging(); store.write_staging_batch(0, post)
                store.commit_staging({"operation": "production_post", "batch_intervals": BLOCK_SIZE})
                evolve_hr4_full_z(store, batch_intervals=BLOCK_SIZE, **self._hydro())
                nxt = store.read_authoritative_batch(0, self.schedule.n_intervals)
                for block_start in range(0, self.schedule.n_intervals, BLOCK_SIZE):
                    block = list(range(block_start, block_start + BLOCK_SIZE))
                    for index in block:
                        lifecycle.enqueue_post(index, actor="hr4c_reference")
                    claimed = lifecycle.claim_block(actor="hr4c_reference")
                    if claimed != block:
                        raise ValueError("reference hydro claim differs from frozen block8 order")
                    for index in claimed:
                        lifecycle.begin_hydro_screen(index, actor="hr4c_reference")
                        lifecycle.commit_next(
                            index, {field: nxt[field][index] for field in FIELDS},
                            actor="hr4c_reference",
                        )
                lifecycle.validate_barrier(actor="hr4c_reference")
                lifecycle.promote_next_to_current(actor="hr4c_reference")
            finally:
                store.close()
            created = self._reference_hydro_created_files(
                state=state, batch_dir=batch_dir, lifecycle=lifecycle,
                intent=hydro_intent, artifacts=self.budget.artifacts(),
            )
            completed = self.budget.complete_intent(hydro_intent["intent_id"], files=created,
                                                    metadata={"reclaimable": True})
            self.budget.consume(hydro_res.reservation_id)
        except BaseException as exc:
            hydro_error = exc
            try: self.budget.interrupt_intent(hydro_intent["intent_id"], reason=str(exc))
            except Exception: pass
            raise
        finally:
            self._close_writer(hydro_writer, failed=hydro_error)

    def _run_candidate_multipulse(self, *, state: Path, output: Path, pulse: int,
                                  reservation: Reservation, intent: Mapping[str, Any],
                                  writer: Mapping[str, Any]) -> dict[str, Any]:
        grid = self.components[0]
        lifecycle = StreamingLifecycle.open(state)
        epoch = self.driver.campaign.active_coordinator_epoch()
        worker_count = int(self.spec.candidate_hydro_workers)
        pair = self.driver.campaign._pair(pulse).get("scientific_pair_identity")
        if not isinstance(pair, Mapping):
            raise StorageIntegrityError("Candidate child launch lacks durable scientific pair identity")
        pair_id = str(pair["scientific_pair_id"])
        generation = str(lifecycle.manifest["current_generation"])
        active_path = state / "ACTIVE_OUTPUT.json"
        previous_output = validate_active_output(
            campaign_root=self.root, state_root=state,
            scientific_pair_id=pair_id,
            implementation_identity_sha256=self.implementation_identity["identity_sha256"],
        ) if active_path.is_file() else None
        if previous_output is not None:
            attempt = previous_output["qualified_attempt"]
            if (attempt["trajectory"] != "C" or int(attempt["pulse"]) != pulse
                    or attempt["generation"] != generation
                    or attempt["admission_identity_sha256"] != self.identity["identity_sha256"]):
                raise StorageIntegrityError("reused Candidate output identity differs")
            producer_grant_id = str(attempt["writer_grant_id"])
            lifecycle.reconstruct_queue(actor=f"e5_1b_recovery_epoch_{epoch}")
            optical_grant = None
            optical_process = None
            optical_launch = None
        else:
            optical_grant = self.budget.create_child_writer_grant(
                reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
                coordinator_epoch=epoch, role="OPTICAL", trajectory="C", pulse=pulse,
                logical_attempt=0, execution_attempt=int(epoch),
                generation=generation, scientific_pair_id=pair_id,
                admission_identity_sha256=self.identity["identity_sha256"],
                implementation_identity=self.implementation_identity,
                allowed_paths=[output, state], expected_bytes=int(intent["expected_bytes"]),
            )
            producer_grant_id = str(optical_grant["grant_id"])
            optical_environment = os.environ.copy()
            optical_environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            optical_environment["PYTHONNOUSERSITE"] = "1"
            optical_environment["HR4E5S_WORKER_ID"] = f"e5_1b_c_optical_{pulse}"
            optical_command = [
                sys.executable, "-s", "-B", "-m", "KHz_filament.hr4e5_production_optical_worker",
                "--lifecycle-root", str(state), "--output-dir", str(output),
                "--pulse", str(int(pulse)), "--config-path", str(self.spec.config_path),
                "--prefix-intervals", str(int(self.schedule.n_intervals)),
                "--admission-identity-sha256", self.identity["identity_sha256"],
                "--storage-root", str(self.root), "--intent-id", str(intent["intent_id"]),
                "--reservation-id", str(reservation.reservation_id),
                "--writer-id", producer_grant_id,
                "--grant-id", producer_grant_id,
                f"--grant-nonce={optical_grant['nonce']}",
                "--scientific-pair-id", pair_id,
                "--implementation-identity-sha256", self.implementation_identity["identity_sha256"],
            ]
            if output.exists():
                optical_command.append("--resume-output")
            if self.require_gpu_backend:
                optical_environment.update(self._gpu_role_environment(pulse=pulse, role="C_OPT"))
                optical_command = self._gpu_bootstrap_command(optical_command)
            optical_process = None
            optical_launch = (optical_command, optical_environment)
        hydro_processes: list[subprocess.Popen] = []
        hydro_grants: list[str] = []
        hydro_stderr_paths: dict[int, Path] = {}
        optical_stderr_path: Path | None = None
        error: BaseException | None = None
        try:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            environment["PYTHONNOUSERSITE"] = "1"
            if previous_output is not None:
                marker_path = state / "PRODUCER_COMPLETE.json"
                if not marker_path.is_file():
                    atomic_json(marker_path, {
                        "schema": "khz_filament.hr4e5.e5_1b.producer_complete.v1",
                        "status": "PASS", "pulse": pulse,
                        "admission_identity_sha256": self.identity["identity_sha256"],
                        "scientific_pair_id": pair_id,
                        "producer_grant_id": producer_grant_id,
                        "producer_coordinator_epoch": previous_output["qualified_attempt"]["producer_coordinator_epoch"],
                        "recovered_from_active_output_sha256": previous_output["pointer_sha256"],
                    }, overwrite=False)
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if (marker.get("status") != "PASS"
                        or marker.get("scientific_pair_id") != pair_id
                        or marker.get("producer_grant_id") != producer_grant_id):
                    raise StorageIntegrityError("reused Candidate producer completion marker differs")
            for worker_index in range(worker_count):
                hydro_grant = self.budget.create_child_writer_grant(
                    reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
                    coordinator_epoch=epoch, role="HYDRO", role_index=worker_index,
                    trajectory="C", pulse=pulse, logical_attempt=0,
                    execution_attempt=int(epoch), generation=generation,
                    scientific_pair_id=pair_id,
                    admission_identity_sha256=self.identity["identity_sha256"],
                    implementation_identity=self.implementation_identity,
                    allowed_paths=[state], expected_bytes=int(intent["expected_bytes"]),
                )
                worker_environment = dict(environment)
                worker_environment["HR4E5S_WORKER_ID"] = f"e5_1b_c_hydro_{pulse}_{worker_index}"
                if self.require_gpu_backend:
                    worker_environment.update(self._gpu_role_environment(
                        pulse=pulse, role=f"H{worker_index}"))
                hydro_grants.append(str(hydro_grant["grant_id"]))
                hydro_command = self._hydro_worker_command(
                    state, pulse, worker_index, intent_id=intent["intent_id"],
                    reservation_id=reservation.reservation_id,
                    writer_id=hydro_grant["grant_id"], grant=hydro_grant,
                    scientific_pair_id=pair_id,
                    producer_grant_id=producer_grant_id,
                )
                if self.require_gpu_backend:
                    hydro_command = self._gpu_bootstrap_command(hydro_command)
                hydro_process, hydro_stderr = self._start_logged_child(
                    hydro_command, state=state, grant_id=hydro_grant["grant_id"],
                    environment=worker_environment,
                )
                hydro_processes.append(hydro_process)
                hydro_stderr_paths[hydro_process.pid] = hydro_stderr
            if optical_launch is not None:
                bind_deadline = time.monotonic() + 30.0
                while True:
                    if any(process.poll() is not None for process in hydro_processes):
                        worker_index, failed = next(
                            (index, process) for index, process in enumerate(hydro_processes)
                            if process.poll() is not None
                        )
                        failed.wait(timeout=5)
                        stderr = hydro_stderr_paths[failed.pid].read_text(encoding="utf-8")
                        failure = StorageIntegrityError(
                            f"hydro H{worker_index} child exited before grant bind: "
                            f"pid={failed.pid}:exit={failed.returncode}:stderr={stderr.strip()}"
                        )
                        self._record_concurrent_rc_failure(
                            pulse=pulse, step="candidate", error=failure,
                            details={"role": f"H{worker_index}", "pid": int(failed.pid),
                                     "exit_code": failed.returncode,
                                     "stderr_path": str(hydro_stderr_paths[failed.pid]), "stderr": stderr},
                        )
                        raise failure
                    cancel_event = getattr(self, "_pair_cancel_event", None)
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Candidate hydro workers cancelled after paired worker failure")
                    states = self.budget.writer_grants()
                    if all(states.get(grant_id, {}).get("state") == "BOUND_ACTIVE" for grant_id in hydro_grants):
                        break
                    if time.monotonic() >= bind_deadline:
                        raise TimeoutError("hydro child grant bind exceeded 30 seconds")
                    time.sleep(0.01)
                command, optical_environment = optical_launch
                optical_process, optical_stderr_path = self._start_logged_child(
                    command, state=state, grant_id=producer_grant_id,
                    environment=optical_environment,
                )
            self._wait_for_candidate_hydro_workers(
                hydro_processes, optical_process=optical_process,
                optical_stderr_path=optical_stderr_path,
                hydro_stderr_paths=hydro_stderr_paths, pulse=pulse,
            )
            self._join_hydro_workers(hydro_processes, hydro_stderr_paths)
            if optical_process is not None:
                self._wait_science_child(optical_process, role="Candidate optical worker")
                if optical_process.returncode != 0:
                    stderr = optical_stderr_path.read_text(encoding="utf-8") if optical_stderr_path else ""
                    raise RuntimeError(
                        f"Candidate optical worker failed: exit={optical_process.returncode}:stderr={stderr.strip()}"
                    )
            active_output = validate_active_output(
                campaign_root=self.root, state_root=state,
                scientific_pair_id=pair_id,
                implementation_identity_sha256=self.implementation_identity["identity_sha256"],
            )
            if active_output["qualified_attempt"]["writer_grant_id"] != producer_grant_id:
                raise StorageIntegrityError("Candidate optical output grant differs from launched child")
            lifecycle = StreamingLifecycle.open(state)
            lifecycle.validate_barrier(actor="e5_1b_production_barrier")
            lifecycle.promote_next_to_current(actor="e5_1b_production_promotion")
            initial = set(intent.get("initial_files", {}))
            created = [
                path for base in (output, state) for path in base.rglob("*")
                if path.is_file() and path.relative_to(self.root).as_posix() not in initial
            ]
            completed = self.budget.complete_intent(
                intent["intent_id"], files=created,
                metadata={"generation": str(lifecycle.manifest["current_generation"])},
            )
            self.budget.consume(reservation.reservation_id)
            return self._simple_report(
                side="C", pulse=pulse, name="pulse",
                summary=np.asarray([pulse, self.schedule.n_intervals], dtype=np.int64),
            )
        except BaseException as exc:
            error = exc
            self._record_concurrent_rc_failure(
                pulse=pulse, step="candidate", error=exc,
                details={"role": "C", "hydro_pids": [int(process.pid) for process in hydro_processes],
                         "optical_pid": None if optical_process is None else int(optical_process.pid)},
            )
            pair_cancel_event = getattr(self, "_pair_cancel_event", None)
            if pair_cancel_event is not None:
                pair_cancel_event.set()
            raise
        finally:
            if error is not None:
                cleanup_errors: list[BaseException] = []
                try:
                    self._stop_hydro_workers(hydro_processes)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                try:
                    if optical_process is not None and optical_process.poll() is None:
                        optical_process.terminate()
                        try:
                            optical_process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            optical_process.kill(); optical_process.wait(timeout=5)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                try:
                    self.budget.interrupt_intent(intent["intent_id"], reason=str(error))
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                try:
                    self._close_writer(writer, failed=error)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                if cleanup_errors:
                    messages = [f"{type(item).__name__}: {item}" for item in cleanup_errors]
                    try:
                        error.add_note("cleanup errors after first failure: " + " | ".join(messages))
                    except Exception:
                        setattr(error, "cleanup_errors", messages)
            else:
                self._close_writer(writer, failed=None)

    def _object_descriptor(self, side: str, pulse: int, key: str) -> dict[str, Any]:
        state = self._track(side, pulse) / "state"
        pair = self.driver.campaign._pair(pulse).get("scientific_pair_identity")
        if not isinstance(pair, Mapping):
            raise StorageIntegrityError("formal object read lacks scientific pair identity")
        active = validate_active_output(
            campaign_root=self.root, state_root=state,
            scientific_pair_id=str(pair["scientific_pair_id"]),
            implementation_identity_sha256=self.implementation_identity["identity_sha256"],
        )
        output = Path(active["output_dir"])
        lifecycle = StreamingLifecycle.open(state)
        parts = key.split(":"); locator: dict[str, Any] = {}
        if parts[0] == "screen":
            namespace, index, name = parts[1], int(parts[2]), parts[3]
            manifest_namespace = "current" if namespace == "pre" else namespace
            entry = lifecycle.manifest["records"][index][manifest_namespace]
            if not isinstance(entry, Mapping):
                raise ValueError(f"production screen object is missing: {key}")
            path = state / entry["artifact"]; locator = {"member": name}
            role, source_index = "screen", index
        elif parts[0] == "sink":
            name, source_index = parts[1], int(parts[2])
            path = output / _SINK_FILES[name]; locator = {"slice": [source_index]}; role = "sink"
            namespace = "sink"
        elif parts[0] == "ledger":
            name = parts[1]; source_index = None; namespace = "ledger"; role = "ledger"
            path = output / "scientific_ledger.npz"; locator = {"member": name}
        else:
            name = "final_optical_field"; source_index = None; namespace = "final"; role = "final_optical"
            path = output / "final_optical_field.npy"
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as loaded: array = np.asarray(loaded[locator["member"]])
        else:
            array = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False))
            if "slice" in locator: array = np.asarray(array[tuple(locator["slice"])])
            elif locator.get("member") is not None:
                with np.load(path, allow_pickle=False) as loaded: array = np.asarray(loaded[locator["member"]])
        relative = path.relative_to(self.root).as_posix()
        ownership = self.budget.artifacts().get(relative)
        if not isinstance(ownership, Mapping):
            raise ValueError(f"production object lacks creation ownership: {relative}")
        return {"trajectory": side, "campaign_id": self.identity["campaign_id"], "pulse": int(pulse),
                "attempt": 0, "path": relative, "relative_path": relative, "locator": locator,
                "shape": list(array.shape), "dtype": array.dtype.name, "finite": bool(np.isfinite(array).all()),
                "canonical_array_hash": sha256_array(array), "file_sha256": sha256_file(path),
                "file_identity": _file_identity(path), "creation_record": dict(ownership["creation_record"]),
                "role": role, "namespace": namespace, "source_index": source_index, "name": name}

    def _run_exact(self, pulse: int):
        folder = self.root / "exact"; path = folder / f"p{pulse}.json"
        owned = self.budget.artifacts().get(path.relative_to(self.root).as_posix())
        if path.is_file() and isinstance(owned, Mapping):
            _checked_owned_file(self.root, path, owned, label="existing paired exact report")
            return validate_paired_exact_report(
                path, admission_identity=self.identity, campaign_id=self.identity["campaign_id"],
                pulse=pulse, root=self.root, storage_budget=self.budget,
                expected_sha256=str(owned["sha256"]),
            )
        keys = sorted(_paired_exact_keys(k=self.schedule.n_intervals,
                                         n_pulses=int(self.spec.n_pulses), pulse=int(pulse)))
        rows = []
        for key in keys:
            left, right = self._object_descriptor("R", pulse, key), self._object_descriptor("C", pulse, key)
            la = {"path": self.root/left["relative_path"], **left["locator"]}
            ra = {"path": self.root/right["relative_path"], **right["locator"]}
            left_array = _load_locator(la)[0]; right_array = _load_locator(ra)[0]
            result = compare_arrays_exact(left_array, right_array, name=key,
                                          reference_path=self.root/left["relative_path"],
                                          candidate_path=self.root/right["relative_path"])
            if result["status"] != "PASS": raise ValueError(f"production exact mismatch: {key}")
            rows.append({"status": "PASS", "comparison_key": key, "reference": left, "candidate": right,
                         "reference_sha256_array": left["canonical_array_hash"],
                         "candidate_sha256_array": right["canonical_array_hash"]})
        reservation, intent, writer = self._intent(side="C", pulse=pulse, role="exact_report",
            paths=[path], expected_bytes=max(4*1024**2, len(rows)*8192), generation=f"exact:p{pulse}")
        error = None
        try:
            payload = {"schema": PAIRED_EXACT_SCHEMA, "status": "PASS",
                       "admission_identity_sha256": self.identity["identity_sha256"],
                       "campaign_id": self.identity["campaign_id"], "pulse": int(pulse), "attempt": 0,
                       "screen_count": (12 if pulse == self.spec.n_pulses-1 else 15)*self.schedule.n_intervals,
                       "ledger_count": 9, "optical_count": 1, "expected_object_count": len(rows),
                       "compared_object_count": len(rows), "mismatch_count": 0, "rows": rows}
            if path.is_file():
                saved = validate_paired_exact_report(
                    path, admission_identity=self.identity, campaign_id=self.identity["campaign_id"],
                    pulse=pulse, root=self.root, storage_budget=self.budget,
                    expected_sha256=sha256_file(path),
                )
                if [(row["comparison_key"], row["reference_sha256_array"], row["candidate_sha256_array"])
                        for row in saved["rows"]] != [
                            (row["comparison_key"], row["reference_sha256_array"], row["candidate_sha256_array"])
                            for row in rows]:
                    raise StorageIntegrityError("interrupted exact report differs from recomputed comparisons")
            else:
                atomic_json(path, payload, overwrite=False)
            completed = self.budget.complete_intent(intent["intent_id"], files=[path], metadata={"reclaimable": False})
            self.budget.consume(reservation.reservation_id, actual_bytes=completed["actual_bytes"])
            return {**payload, "report_path": str(path), "report_sha256": sha256_file(path)}
        except BaseException as exc:
            error = exc
            try: self.budget.interrupt_intent(intent["intent_id"], reason=str(exc))
            except Exception: pass
            raise
        finally: self._close_writer(writer, failed=error)

    def _run_successor(self, trajectory: str, pulse: int):
        parent = self._track(trajectory, pulse) / "state"
        child = self._track(trajectory, pulse+1) / "state"
        archive = parent / "E5_1A_ARCHIVED_AFTER_EXACT.json"
        child_generation = str(StreamingLifecycle.open(parent).manifest["next_generation"])
        estimate = max(4*1024**2, 4*3*self.schedule.n_intervals*self.components[0].Nx*self.components[0].Ny*8)
        reservation, intent, writer = self._intent(side=trajectory, pulse=pulse + 1, role="SUCCESSOR",
            paths=[child, archive], expected_bytes=estimate, generation=child_generation)
        error = None
        try:
            lifecycle, ready = create_successor_root(
                parent_root=parent, child_root=child, exact_report_path=child/"parent_next_child_current_exact.json",
                admission_identity=self.identity, storage_budget=self.budget, creation_intent=intent,
                trajectory=trajectory, pulse=pulse + 1, role="SUCCESSOR")
            return self._simple_report(side=trajectory, pulse=pulse, name="successor",
                summary=np.asarray([pulse, pulse+1], dtype=np.int64),
                extras={"child_root": str(child), "parent_root": str(parent),
                        "parent_generation": ready["parent_generation"]})
        except BaseException as exc: error = exc; raise
        finally: self._close_writer(writer, failed=error)

    def _run_gc(self, trajectory: str, pulse: int, *, terminal_ready_path: Path | None = None):
        final = pulse == self.spec.n_pulses - 1
        if (terminal_ready_path is None) == final or trajectory not in {"R", "C"}:
            raise ValueError("terminal reclaim requires the final POST_READY receipt")
        parent = self._track(trajectory, pulse)
        pair = self.driver.campaign._pair(pulse)
        lineage_receipt = pair.get("pair_lineage_receipt")
        if not isinstance(lineage_receipt, Mapping):
            raise StorageIntegrityError("production GC lacks qualified pair lineage")
        lineage_path = Path(str(lineage_receipt["path"])).resolve()
        if (not lineage_path.is_relative_to(self.root) or not lineage_path.is_file()
                or sha256_file(lineage_path) != lineage_receipt["sha256"]):
            raise StorageIntegrityError("production GC pair lineage hash changed")
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        output = Path(str(lineage["trajectories"][trajectory]["output_dir"])).resolve()
        if not output.is_relative_to(parent / "optical"):
            raise StorageIntegrityError("production GC output locator escapes trajectory")
        artifacts = self.budget.artifacts()
        targets: list[Path] = []
        for relative, item in artifacts.items():
            path = self.root / relative
            if (not (final and trajectory == "C") and path.is_relative_to(parent / "state")
                    and path.suffix in {".npy", ".npz"}):
                targets.append(path)
            elif (not (final and trajectory == "C") and path.is_relative_to(parent / "batch")
                   and path.suffix in {".npy", ".npz"}):
                targets.append(path)
            elif path.parent == output and path.name in _SINK_FILES.values():
                targets.append(path)
        targets.sort(key=lambda path: path.as_posix())
        sink_targets = {output / name for name in _SINK_FILES.values()}
        if set(targets) & sink_targets != sink_targets:
            raise StorageIntegrityError("production sink ownership is incomplete")
        if not targets:
            raise StorageBudgetError("production GC target set is empty")
        plan_id = f"production-{trajectory}-p{pulse}"
        saved_exact = self.driver._saved_receipt(pulse, "exact")
        if saved_exact is None:
            raise StorageIntegrityError("production reclaim lacks committed paired actual exact")
        exact_path = self.root / "exact" / f"p{pulse}.json"
        exact_owned = artifacts.get(exact_path.relative_to(self.root).as_posix())
        if (not isinstance(exact_owned, Mapping)
                or str(exact_owned.get("sha256")) != str(saved_exact.get("report_sha256"))):
            raise StorageIntegrityError("paired exact report ownership or hash changed")
        rows = {str(row.get("comparison_key")): row for row in saved_exact["rows"]}
        sink_keys: list[str] = []
        for name, filename in _SINK_FILES.items():
            target = output / filename
            relative = target.relative_to(self.root).as_posix()
            owned = artifacts[relative]
            for index in range(self.schedule.n_intervals):
                key = f"sink:{name}:{index}"
                row = rows.get(key)
                descriptor = row.get("reference" if trajectory == "R" else "candidate") if isinstance(row, Mapping) else None
                if (not isinstance(descriptor, Mapping) or row.get("status") != "PASS"
                        or descriptor.get("relative_path") != relative
                        or descriptor.get("locator") != {"slice": [index]}
                        or descriptor.get("role") != "sink"
                        or descriptor.get("source_index") != index
                        or descriptor.get("shape") != [self.components[0].Ny, self.components[0].Nx]
                        or descriptor.get("dtype") != "float64"
                        or descriptor.get("finite") is not True
                        or descriptor.get("file_sha256") != owned.get("sha256")
                        or descriptor.get("file_identity") != owned.get("identity")
                        or descriptor.get("creation_record") != owned.get("creation_record")):
                    raise StorageIntegrityError(f"sink has no matching executed exact row: {key}")
                sink_keys.append(key)
        existing = self.budget.reclaim_plan(plan_id)
        if existing is not None:
            if (set(str(item["relative_path"]) for item in existing["targets"])
                    != {path.relative_to(self.root).as_posix() for path in targets}):
                raise StorageIntegrityError("existing reclaim plan target set changed")
            result = self.budget.resume_reclaim(
                plan_id, expected_trajectory=trajectory, expected_pulse=pulse,
                expected_attempt=0, expected_admission_hash=self.identity["identity_sha256"],
            )
            verified = self.budget.verify_reclaim(
                plan_id, expected_trajectory=trajectory, expected_pulse=pulse,
                expected_attempt=0, expected_admission_hash=self.identity["identity_sha256"],
            )
            if result.get("status") != "COMPLETED" or verified.get("status") != "PASS":
                raise StorageIntegrityError("resumed production reclaim is incomplete")
            writer_receipt = existing["writer_receipt"]
            return self._simple_report(side=trajectory, pulse=pulse, name="gc",
                summary=np.asarray([pulse, len(targets)], dtype=np.int64),
                extras={"gc_plan_id": plan_id, "writer_receipt_path": writer_receipt["path"],
                        "writer_receipt_sha256": writer_receipt["sha256"]})
        self.budget.set_reclaimable(targets, value=True)
        evidence = parent / "gc_evidence"; files: list[Path] = []
        reservation, intent, writer = self._intent(side=trajectory, pulse=pulse, role="gc_evidence",
            paths=[evidence], expected_bytes=max(4*1024**2, len(targets)*8192), generation=f"gc:{trajectory}:p{pulse}")
        error = None
        try:
            evidence.mkdir(parents=True, exist_ok=True)
            exact = evidence / "targets_exact.json"
            target_objects: dict[str, Any] = {}
            for index, target in enumerate(targets):
                if target.suffix == ".npz":
                    with np.load(target, allow_pickle=False) as loaded:
                        members = [name for name in loaded.files if name in FIELDS]
                    for member in members:
                        target_objects[f"target_{index}_{member}"] = {
                            "path": target, "member": member,
                        }
                else:
                    target_objects[f"target_{index}"] = target
            rows = build_expected_object_set(
                target_objects,
                campaign_id=self.identity["campaign_id"], trajectory=trajectory, pulse=pulse,
                attempt=0, namespace="gc_target", root=self.root,
            )
            bind_exact_report(exact, {"status": "PASS", "mismatch_count": 0,
                                      "missing_reference": [], "missing_candidate": []}, rows,
                              campaign_id=self.identity["campaign_id"], root=self.root)
            ready = (terminal_ready_path if terminal_ready_path is not None else
                     self._track(trajectory, pulse+1) / "state" / "E5_1A_READY.json")
            dependency = evidence / "dependency.json"
            atomic_json(dependency, {"status": "PASS", "no_future_dependency": True,
                                     "paired_exact": str(exact_path),
                                     "paired_exact_sha256": saved_exact["report_sha256"],
                                     "sink_comparison_keys": sink_keys,
                                     "terminal_release": final}, overwrite=False)
            owned = self.budget.artifacts(); bindings = []
            for target in targets:
                relative = target.relative_to(self.root).as_posix(); item = owned[relative]
                bindings.append({"relative_path": relative, "identity": dict(item["identity"]),
                                 "sha256": item["sha256"]})
            prerequisite = evidence / "prerequisites.json"
            atomic_json(prerequisite, {"schema": RECLAIM_PREREQUISITE_SCHEMA, "status": "PASS",
                "campaign_id": self.identity["campaign_id"], "gates": {"exact_complete": True,
                "successor_ready": True, "no_active_writers": True, "no_future_dependency": True,
                "receipts_durable": True}, "target_bindings": bindings,
                "evidence": {"exact_complete": {"path": str(exact), "sha256": sha256_file(exact)},
                "successor_ready": {"path": str(ready), "sha256": sha256_file(ready)},
                "no_future_dependency": {"path": str(dependency), "sha256": sha256_file(dependency)}}}, overwrite=False)
            files = [exact, dependency, prerequisite]
            completed = self.budget.complete_intent(intent["intent_id"], files=files,
                                                     metadata={"reclaimable": False})
            self.budget.consume(reservation.reservation_id, actual_bytes=completed["actual_bytes"])
        except BaseException as exc:
            error = exc
            try: self.budget.interrupt_intent(intent["intent_id"], reason=str(exc))
            except Exception: pass
            raise
        finally: self._close_writer(writer, failed=error)
        epoch = self.driver.campaign.active_coordinator_epoch()
        writer_receipt = self.budget.write_quiescence_receipt(
            evidence / "writer_quiescent.json", coordinator_epoch=epoch,
            trajectory=trajectory, pulse=pulse, attempt=0)
        self.budget.plan_reclaim(targets, exact_complete=True, successor_ready=True,
            no_active_writers=True, no_future_dependency=True, receipts_durable=True,
            writer_receipt=writer_receipt, writer_epoch=epoch, prerequisite_receipt=prerequisite,
            plan_id=plan_id, trajectory=trajectory, pulse=pulse, attempt=0,
            admission_hash=self.identity["identity_sha256"],
            reason="after_final_post_ready" if terminal_ready_path is not None else "after_exact_successor_ready")
        self.budget.apply_reclaim(plan_id, expected_trajectory=trajectory, expected_pulse=pulse,
            expected_attempt=0, expected_admission_hash=self.identity["identity_sha256"])
        self.budget.verify_reclaim(plan_id, expected_trajectory=trajectory, expected_pulse=pulse,
            expected_attempt=0, expected_admission_hash=self.identity["identity_sha256"])
        return self._simple_report(side=trajectory, pulse=pulse, name="gc",
            summary=np.asarray([pulse, len(targets)], dtype=np.int64),
            extras={"gc_plan_id": plan_id, "writer_receipt_path": writer_receipt["path"],
                    "writer_receipt_sha256": writer_receipt["sha256"]})

    def _run_terminal(self, pulse: int):
        state = self._track("C", pulse) / "state"; output = self._track("C", pulse) / "optical"
        folder = self.root / "terminal"; ready = state / "POST_FINAL_READY.json"
        writer_path = state / "writer_quiescent.json"
        contract = self.root / "E5_1A_TERMINAL_ROLE_CONTRACT.json"
        ready_owned = self.budget.artifacts().get(ready.relative_to(self.root).as_posix())
        if ready.is_file() and isinstance(ready_owned, Mapping):
            _checked_owned_file(self.root, ready, ready_owned, label="terminal READY")
            saved = json.loads(ready.read_text(encoding="utf-8"))
            receipt = dict(saved.get("writer_receipt", {}))
            if not receipt.get("path") or not receipt.get("sha256"):
                raise StorageIntegrityError("terminal READY lacks its writer receipt")
            resumed = resume_final_post(lifecycle_root=state, receipt_path=ready,
                                        writer_receipt=receipt, fixture_only=False)
            if resumed.get("status") != "PASS":
                raise StorageIntegrityError("terminal READY cannot be reopened")
        else:
            ready_res, ready_intent, ready_writer = self._intent(
                side="C", pulse=pulse, role="TERMINAL_READY",
                paths=[ready, writer_path], expected_bytes=4*1024**2,
                generation=f"terminal-ready:p{pulse}")
            ready_error = None
            try:
                # Quiescence is sampled with the management writer closed.
                self._close_writer(ready_writer)
                ready_writer = None
                if writer_path.is_file():
                    receipt = self.budget._read_writer_receipt(
                        writer_path, expected_trajectory="C", expected_pulse=pulse,
                        expected_attempt=0, expected_admission_hash=self.identity["identity_sha256"])
                else:
                    receipt = self.budget.write_quiescence_receipt(
                        writer_path, coordinator_epoch=self.driver.campaign.active_coordinator_epoch(),
                        trajectory="C", pulse=pulse, attempt=0)
                if not ready.is_file():
                    ready_writer = self.budget.open_writer(
                        reservation_id=ready_res.reservation_id, intent_id=ready_intent["intent_id"],
                        coordinator_epoch=self.driver.campaign.active_coordinator_epoch(),
                        trajectory="C", pulse=pulse, attempt=0,
                        generation=f"terminal-ready:p{pulse}")
                    result = validate_final_post(
                        lifecycle_root=state, receipt_path=ready, writer_quiescent=False,
                        fixture_only=False, expected_optical_dir=output, writer_receipt=receipt)
                    if result.get("status") != "PASS":
                        raise StorageIntegrityError("terminal POST failed initial validation")
                else:
                    resume_final_post(lifecycle_root=state, receipt_path=ready,
                                      writer_receipt=receipt, fixture_only=False)
                completed = self.budget.complete_intent(
                    ready_intent["intent_id"], files=[writer_path, ready],
                    metadata={"reclaimable": False})
                self.budget.consume(ready_res.reservation_id, actual_bytes=completed["actual_bytes"])
            except BaseException as exc:
                ready_error = exc
                try: self.budget.interrupt_intent(ready_intent["intent_id"], reason=str(exc))
                except Exception: pass
                raise
            finally:
                if ready_writer is not None:
                    self._close_writer(ready_writer, failed=ready_error)
        # Both tracks are quiescent before either final payload is released.
        for side in ("R", "C"):
            plan_id = f"production-{side}-p{pulse}"
            # Reenter even after the reclaim plan completed: a process can
            # stop before its separate durable GC report was written.
            self._run_gc(side, pulse, terminal_ready_path=ready)
            verified = self.budget.verify_reclaim(
                plan_id, expected_trajectory=side, expected_pulse=pulse,
                expected_attempt=0, expected_admission_hash=self.identity["identity_sha256"])
            if verified.get("status") != "PASS":
                raise StorageIntegrityError("terminal reclaim remains incomplete")
        report = folder / "terminal.json"; summary = folder / "terminal_summary.npy"
        if contract.is_file():
            owned = self.budget.artifacts().get(contract.relative_to(self.root).as_posix())
            if not isinstance(owned, Mapping):
                raise StorageIntegrityError("terminal role contract changed or is unowned")
            _checked_owned_file(self.root, contract, owned, label="terminal role contract")
            return {**validate_durable_report(report, expected_sha256=self.budget.artifacts()[
                report.relative_to(self.root).as_posix()]["sha256"],
                campaign_id=self.identity["campaign_id"], root=self.root, pulse=pulse, trajectory="C"),
                "writer_receipt_path": receipt["path"], "writer_receipt_sha256": receipt["sha256"]}
        reservation, intent, writer = self._intent(side="C", pulse=pulse, role="TERMINAL_EVIDENCE",
            paths=[folder, contract], expected_bytes=16*1024**2,
            generation=f"terminal:p{pulse}")
        error = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            np.save(summary, np.asarray([pulse, self.schedule.n_intervals], dtype=np.int64))
            rows = build_expected_object_set(
                {"terminal": summary}, campaign_id=self.identity["campaign_id"], trajectory="C",
                pulse=pulse, attempt=0, namespace="terminal", root=self.root,
                source_indices={"terminal": 0}, role="TERMINAL_EVIDENCE")
            result = bind_exact_report(report, {"status": "PASS", "mismatch_count": 0,
                "writer_receipt_path": receipt["path"], "writer_receipt_sha256": receipt["sha256"]},
                rows, campaign_id=self.identity["campaign_id"], root=self.root)
            role_map: dict[str, list[str]] = {}
            artifacts = self.budget.artifacts()
            if any(int(item.get("attempt", -1)) != 0 for item in artifacts.values()):
                raise StorageIntegrityError("terminal inventory contains an old or unknown attempt")
            for relative, item in artifacts.items():
                if (self.root/relative).is_file(): role_map.setdefault(str(item["role"]), []).append(relative)
            required_role_counts = {
                "report_admission": 2,
                "OPTICAL": 4 * int(self.spec.n_pulses),
                "report_pulse": 4 * int(self.spec.n_pulses),
                "exact_report": int(self.spec.n_pulses),
                "report_successor": 4 * (int(self.spec.n_pulses) - 1),
                "gc_evidence": 6 * int(self.spec.n_pulses),
                "report_gc": 4 * int(self.spec.n_pulses),
                "TERMINAL_READY": 2,
            }
            missing_roles = {
                role: minimum for role, minimum in required_role_counts.items()
                if len(role_map.get(role, [])) < minimum
            }
            if missing_roles:
                raise StorageIntegrityError(
                    f"terminal production role contract is incomplete: {missing_roles}"
                )
            terminal_files = [summary, report, contract]
            role_map.setdefault("TERMINAL_EVIDENCE", []).extend(
                path.relative_to(self.root).as_posix() for path in terminal_files)
            atomic_json(contract, {"schema": "khz_filament.hr4e5.e5_1a.terminal_roles.v1",
                "admission_identity_sha256": self.identity["identity_sha256"],
                "terminal_pulse": int(pulse), "roles": role_map}, overwrite=False)
            completed = self.budget.complete_intent(intent["intent_id"], files=terminal_files,
                                                     metadata={"reclaimable": False})
            self.budget.consume(reservation.reservation_id, actual_bytes=completed["actual_bytes"])
            return {**result, "writer_receipt_path": receipt["path"],
                    "writer_receipt_sha256": receipt["sha256"]}
        except BaseException as exc:
            error = exc
            try: self.budget.interrupt_intent(intent["intent_id"], reason=str(exc))
            except Exception: pass
            raise
        finally: self._close_writer(writer, failed=error)


class E5AProductionCampaign:
    def __init__(self, runner: _E5AProductionRunner, driver: Any):
        self._runner, self._driver = runner, driver

    def run(self, *, stop_after: int | None = None):
        result = self._driver.run(stop_after=stop_after)
        if result.get("status") == "COMPLETE":
            roles = self._runner.expected_terminal_roles(self._runner.spec.n_pulses - 1)
            inventory = self._runner.budget.validate_terminal_inventory(
                roles, root=self._runner.root, final=True)
            measured = self._runner.budget.final_output_bytes()
            if measured != inventory["final_output_bytes"]:
                raise StorageIntegrityError("final byte views differ after campaign completion")
            result = {**result, "terminal_inventory_after_close": inventory,
                      "storage_after_close": self._runner.budget.budget_report()}
        return {**result, "admission_identity": dict(self._runner.identity)}
    def pause(self, *, reason: str = "operator_pause"): return self._driver.pause(reason=reason)
    def close(self): return self._driver.close()
    def report(self):
        return {**self._driver.report(), "admission_identity": dict(self._runner.identity)}


def open_e5_1a_production_campaign(spec: E5AProductionSpec) -> E5AProductionCampaign:
    """The only supported creator/resumer for a production-mode E5-1A run."""
    if not isinstance(spec, E5AProductionSpec):
        raise TypeError("E5AProductionSpec is required")
    checked_root = _checked_campaign_root(spec.root)
    persisted_admission_path = checked_root / "E5_1A_ADMISSION_IDENTITY.json"
    if persisted_admission_path.is_file():
        persisted_identity = load_admission_identity(persisted_admission_path, formal=True)
        admitted_timeout = resolve_storage_lock_timeout_s(
            persisted_identity, explicit=spec.storage_lock_timeout_s,
        )
        runtime = persisted_identity["runtime_or_compatibility"]
        if spec.storage_lock_timeout_s is None and STORAGE_LOCK_TIMEOUT_KEY in runtime:
            # Reconstructing the identity and child spec must preserve the
            # policy already bound to this campaign's admission hash.
            spec = replace(spec, storage_lock_timeout_s=admitted_timeout)
    if spec.require_gpu_backend or spec.admission_level == FORMAL_SITE_ADMISSION:
        if os.environ.get("UPPE_USE_GPU") != "0" or device.USE_GPU:
            raise RuntimeError("formal GPU coordinator must start with UPPE_USE_GPU=0")
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("formal GPU admission requires a Slurm job allocation")
        build_assignment_record(
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            2 + int(spec.candidate_hydro_workers),
        )
    current_implementation = capture_implementation_identity()
    identity, schedule, components = _build_admission(
        spec, implementation_identity=current_implementation,
    )
    root = checked_root; root.mkdir(parents=True, exist_ok=True)
    reopening = (root / CAMPAIGN_STATE_FILENAME).is_file()
    admission_path = root / "E5_1A_ADMISSION_IDENTITY.json"
    implementation_path = root / "E5_1B_IMPLEMENTATION_IDENTITY.json"
    seal_path = root / "E5_1B_CANONICAL_SOURCE_SEAL_RECEIPT.json"
    if admission_path.is_file():
        persist_admission_identity(admission_path, identity)
    if implementation_path.is_file():
        require_same_implementation(
            json.loads(implementation_path.read_text(encoding="utf-8")),
            current_implementation,
        )
    elif reopening:
        raise StorageIntegrityError("reopened WP2 campaign lacks implementation identity")
    if reopening and isinstance(identity.get("source_identity", {}).get("canonical_source"), Mapping):
        if not seal_path.is_file():
            raise StorageIntegrityError("reopened WP2 campaign lacks canonical source seal receipt")
        prior_seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if (prior_seal.get("admission_identity_sha256") != identity["identity_sha256"]
                or prior_seal.get("implementation_identity_sha256")
                != current_implementation["identity_sha256"]):
            raise StorageIntegrityError("reopened canonical source seal receipt binding differs")
    source_identity = identity.get("source_identity", {})
    canonical_source = source_identity.get("canonical_source") if isinstance(source_identity, Mapping) else None
    seal_receipt = None
    if isinstance(canonical_source, Mapping):
        seal_receipt = seal_or_validate_canonical_source(
            canonical_source, admission_identity=identity,
            implementation_identity=current_implementation,
            config_path=spec.config_path, source_manifest_path=spec.source_manifest_path,
            create_if_missing=not reopening,
        )
    persist_admission_identity(admission_path, identity)
    if implementation_path.is_file():
        persisted_implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
        require_same_implementation(persisted_implementation, current_implementation)
    elif reopening:
        raise StorageIntegrityError("reopened WP2 campaign lacks implementation identity")
    else:
        atomic_json(implementation_path, current_implementation, overwrite=False)
    if seal_receipt is not None:
        if seal_path.is_file():
            prior_seal_receipt = json.loads(seal_path.read_text(encoding="utf-8"))
            if prior_seal_receipt != seal_receipt:
                raise StorageIntegrityError("reopened canonical source seal receipt differs")
        else:
            if reopening:
                raise StorageIntegrityError("reopened WP2 campaign lacks canonical source seal receipt")
            atomic_json(seal_path, seal_receipt, overwrite=False)
    runner = _E5AProductionRunner(spec, identity, schedule, components, current_implementation)
    driver = _open_production_driver(
        root, admission_identity=identity, runner=runner, n_pulses=int(spec.n_pulses),
        campaign_id=str(spec.campaign_id), max_campaign_live_bytes=int(spec.max_campaign_live_bytes),
        final_output_budget_bytes=int(spec.final_output_budget_bytes),
        safety_margin_bytes=int(spec.safety_margin_bytes), require_quota=True,
        storage_lock_timeout_s=spec.storage_lock_timeout_s,
        takeover=reopening,
    )
    runner.driver = driver
    if identity["budget_identity"].get("prefix_inside_campaign_root"):
        driver.campaign.storage.register_preexisting_prefix(
            identity["pre0_identity"]["path"],
            expected_sha256=identity["pre0_identity"]["sha256"],
        )
    lut_staging = identity.get("lut_identity", {}).get("staging", {})
    for input_record in lut_staging.get("registered_inputs", []) if isinstance(lut_staging, Mapping) else []:
        driver.campaign.storage.register_preexisting_frozen_lut(
            root / Path(str(input_record["path"])),
            expected_sha256=str(input_record["sha256"]),
            expected_size_bytes=int(input_record["size_bytes"]),
            admission_identity_sha256=str(identity["identity_sha256"]),
        )
    if spec.admission_level == LOCAL_ORCHESTRATION_QUALIFICATION:
        driver.campaign.storage.provider = _LocalPolicyCapacityProvider(root, spec.max_campaign_live_bytes)
    else:
        driver.campaign.storage.provider = _ManifestCapacityProvider(
            _read_manifest(_checked_input_path(spec.site_resource_manifest_path,
                label="site resource manifest"), label="site resource")
        )
    runner.initialize()
    return E5AProductionCampaign(runner, driver)


__all__ = ["E5AProductionCampaign", "E5AProductionSpec", "FORMAL_SITE_ADMISSION",
           "LOCAL_ORCHESTRATION_QUALIFICATION", "open_e5_1a_production_campaign",
           "stage_frozen_lut_inputs", "validate_staged_frozen_lut_cache_hit"]

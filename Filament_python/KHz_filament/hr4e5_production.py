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
import time
import sys
from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import dataclass
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
                                  resume_final_post, validate_final_post)
from .hr4e5_paired_campaign import CAMPAIGN_STATE_FILENAME, _open_production_driver
from .hr4e5_storage import (DEFAULT_FINAL_OUTPUT_BUDGET_BYTES, DEFAULT_SAFETY_MARGIN_BYTES,
                              HARD_CAP_BYTES, INTENT_SCHEMA, PROCESS_RECEIPT_SCHEMA, RECLAIM_PREREQUISITE_SCHEMA,
                              Reservation, StorageBudgetError, StorageIntegrityError,
                              _is_reparse, plan_campaign_budget, process_identity)
from .longitudinal import build_longitudinal_schedule
from .runner import apply_thin_lens_achromatic, build_transverse_input_field
from .constants import c0
from .device import to_cpu, xp
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
    max_campaign_live_bytes: int = HARD_CAP_BYTES
    final_output_budget_bytes: int = DEFAULT_FINAL_OUTPUT_BUDGET_BYTES
    safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES


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


def _source_hash(config_path: Path) -> str:
    grid, beam, prop, *_ = load_all(str(config_path))
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    field, _ = build_transverse_input_field(axes, beam, xp.complex128)
    if getattr(beam, "focal_length", None):
        omega0 = 2.0 * np.pi * c0 / beam.lam0
        k0 = beam.n0 * omega0 / c0
        if str(getattr(prop, "linear_model", "uppe")).lower() == "uppe":
            field = apply_thin_lens_achromatic(field, axes, beam, prop,
                                                chunk_t=getattr(prop, "lens_chunk_t", 0))
        else:
            X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
            field *= xp.exp(xp.asarray(-1j * k0 * (X ** 2 + Y ** 2) /
                                       (2.0 * float(beam.focal_length)), dtype=xp.complex128))
    return sha256_array(np.asarray(to_cpu(field)))


def _schedule_from_config(prop: Any) -> Any:
    return build_longitudinal_schedule(
        dz=float(prop.dz), z_max=float(prop.z_max),
        focus_window_step=bool(getattr(prop, "focus_window_step", False)),
        focus_center_m=getattr(prop, "focus_center_m", None),
        focus_halfwidth_m=float(getattr(prop, "focus_halfwidth_m", 0.0)),
        dz_focus=getattr(prop, "dz_focus", None),
    )


def describe_e5_1b_input_contract(full_config_path: str | Path,
                                  *, prefix_intervals: int = 8048) -> dict[str, Any]:
    """Describe the official input range without opening any scientific array."""
    full_config = _checked_input_path(full_config_path, label="full source config")
    grid, beam, prop, ion, heat, run, raman = load_all(str(full_config))
    full = _schedule_from_config(prop)
    prefix = build_prefix_schedule(full, prefix_intervals)
    return {
        "schema": "khz_filament.hr4e5.e5_1b.input_contract.v1",
        "status": "INPUT_NOT_MATERIALIZED",
        "full_config_path": str(full_config), "full_config_raw_sha256": sha256_file(full_config),
        "derived_config_rule": "change_run.Npulses_from_1_to_2_only",
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
                                   output_path: str | Path) -> dict[str, Any]:
    """Write the sole permitted N=2 config derivation without changing its parent."""
    source = _checked_input_path(full_config_path, label="full source config")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("run"), dict):
        raise ValueError("full source config lacks run settings")
    if int(value["run"].get("Npulses", -1)) != 1:
        raise ValueError("full source config must have Npulses=1")
    value["run"]["Npulses"] = 2
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _bound_configs(source, destination, 2)
    return {"parent_path": str(source), "parent_raw_sha256": sha256_file(source),
            "derived_path": str(destination), "derived_raw_sha256": sha256_file(destination),
            "changed_field": "run.Npulses", "value": 2}


def _bound_configs(full_config: Path, config: Path, n_pulses: int) -> dict[str, Any]:
    _checked_input_path(full_config, label="full source config")
    parent = json.loads(full_config.read_text(encoding="utf-8"))
    derived = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(parent, dict) or not isinstance(derived, dict):
        raise ValueError("production configs must be JSON objects")
    parent_run, derived_run = parent.get("run"), derived.get("run")
    if not isinstance(parent_run, dict) or not isinstance(derived_run, dict):
        raise ValueError("production configs lack run settings")
    if int(derived_run.get("Npulses", -1)) != int(n_pulses):
        raise ValueError("derived config pulse count differs from production spec")
    if full_config != config:
        if int(parent_run.get("Npulses", -1)) != 1 or int(n_pulses) != 2:
            raise ValueError("formal config derivation must change Npulses from 1 to 2")
        expected = json.loads(json.dumps(parent))
        expected["run"]["Npulses"] = 2
        if expected != derived:
            raise ValueError("derived config changes fields besides run.Npulses")
    return {"path": str(full_config), "sha256": sha256_file(full_config)}


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


def _build_admission(spec: E5AProductionSpec) -> tuple[dict[str, Any], Any, tuple[Any, ...]]:
    level = str(spec.admission_level)
    if level not in _ADMISSION_LEVELS:
        raise ValueError("unsupported production admission level")
    config = _checked_input_path(spec.config_path, label="execution config")
    source_path = _checked_input_path(spec.source_manifest_path, label="optical source manifest")
    lut_path = _checked_input_path(spec.lut_manifest_path, label="LUT manifest")
    pre0_path = _checked_input_path(spec.pre0_delta_n_path, label="PRE0 source")
    full_config = (_checked_input_path(spec.full_config_path, label="full source config")
                   if spec.full_config_path is not None else config)
    full_config_identity = _bound_configs(full_config, config, spec.n_pulses)
    source = _read_manifest(source_path, label="source")
    lut = _read_manifest(lut_path, label="LUT")
    components = load_all(str(config))
    grid, beam, prop, ion, heat, run, raman = components
    full_components = load_all(str(full_config))
    full_schedule = _schedule_from_config(full_components[2])
    requested_k = full_schedule.n_intervals if spec.prefix_intervals is None else int(spec.prefix_intervals)
    schedule = build_prefix_schedule(full_schedule, requested_k)
    if level == FORMAL_SITE_ADMISSION and (
            full_config == config or full_schedule.n_intervals != 15000
            or spec.prefix_intervals != 8048):
        raise ValueError("formal source and derived config require full K15000 and prefix K8048")
    if schedule.n_intervals % BLOCK_SIZE or schedule.n_intervals <= 0:
        raise ValueError("production schedule requires complete blocks of eight")
    pre0 = np.load(pre0_path, mmap_mode="r", allow_pickle=False)
    if pre0.shape != (schedule.n_intervals, grid.Ny, grid.Nx) or pre0.dtype != np.float64:
        raise ValueError("PRE0 delta_n shape or dtype differs from production config")
    source_hash = _source_hash(config)
    if str(source.get("canonical_array_hash", "")) != source_hash:
        raise ValueError("analytic source manifest differs from reconstructed input field")
    if lut.get("mode") == "DISABLED_BY_CONFIG":
        if getattr(ion, "species", None) not in ([], ()) or bool(getattr(prop, "use_ionization_solver", False)):
            raise ValueError("LUT cannot be disabled while ionization is enabled")
        if str(lut.get("config_sha256", "")) != sha256_file(config):
            raise ValueError("disabled LUT manifest is not bound to config")
    elif not lut.get("signature"):
        raise ValueError("production LUT manifest lacks a signature")
    if int(getattr(run, "Npulses", spec.n_pulses)) != int(spec.n_pulses):
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
        n_pulses=int(spec.n_pulses), k=schedule.n_intervals,
        ny=int(grid.Ny), nx=int(grid.Nx), nt=int(grid.Nt),
        max_campaign_live_bytes=int(spec.max_campaign_live_bytes),
        final_output_budget_bytes=int(spec.final_output_budget_bytes),
        safety_margin_bytes=int(spec.safety_margin_bytes),
        external_prefix_already_present=True,
        existing_full_source_bytes=int((slow_source_identity or {}).get("full_source_raw_bytes", 0)),
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
        bound_paths = (campaign_root, pre0_path,
                       Path(str(slow_source_identity["full_source_path"])))
        if any(not path.is_relative_to(capacity_root) for path in bound_paths):
            raise ValueError("formal inputs and campaign do not share the qualified capacity root")
        existing_parent = campaign_root
        while not existing_parent.exists():
            existing_parent = existing_parent.parent
        volume = capacity_root.stat().st_dev
        if any(path.stat().st_dev != volume for path in (pre0_path, bound_paths[2], existing_parent)):
            raise ValueError("formal inputs and campaign occupy different filesystems")
        resource_evidence = {**site, "path": str(site_path), "sha256": sha256_file(site_path)}
    else:
        if spec.site_resource_manifest_path is not None:
            raise ValueError("local qualification cannot claim site resource evidence")
        resource_evidence = {"scope": _LocalPolicyCapacityProvider.scope}
    identity = build_admission_identity(
        campaign_id=str(spec.campaign_id), execution_mode=level,
        runtime_or_compatibility={"module": "KHz_filament.hr4e5_production", "runner": "single-production-runner"},
        config_identity={"path": str(config), "sha256": sha256_file(config),
                         "full_source_config": full_config_identity,
                         "derivation": "run.Npulses=2_only" if full_config != config else "same_config"},
        effective_params={"f_rep": float(heat.f_rep), "dt_hydro": float(getattr(heat, "dt_hydro", 1e-7))},
        source_identity={"path": str(source_path), "sha256": sha256_file(source_path),
                         "canonical_array_hash": source_hash},
        lut_identity={"path": str(lut_path), "sha256": sha256_file(lut_path), **lut},
        schedule_identity={"k": schedule.n_intervals, "metadata": schedule.as_metadata(),
                           "full_metadata": full_schedule.as_metadata(),
                           "source_indices": [0, schedule.n_intervals - 1],
                           "recipe": "build_full_then_prefix"},
        grid_identity={"Nx": grid.Nx, "Ny": grid.Ny, "Nt": grid.Nt,
                       "shape": [grid.Ny, grid.Nx], "dtype": "complex128"},
        pre0_identity={"path": str(pre0_path), "sha256": sha256_file(pre0_path),
                       "canonical_array_hash": sha256_array_streaming(pre0),
                       "slow_source_identity": slow_source_identity or {"scope": "LOCAL_FULL_SCHEDULE_FIXTURE"}},
        n_pulses=int(spec.n_pulses), k=int(schedule.n_intervals), block_size=BLOCK_SIZE,
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
                          "final_output_bytes": budget_plan["final_output_bytes"],
                           "external_prefix_bytes": (0 if pre0_path.is_relative_to(_checked_campaign_root(spec.root))
                                                     else int(pre0_path.stat().st_size)),
                           "prefix_owned_bytes": int(pre0_path.stat().st_size),
                           "prefix_payload_bytes": budget_plan["external_prefix_bytes"],
                           "prefix_inside_campaign_root": pre0_path.is_relative_to(_checked_campaign_root(spec.root)),
                          "existing_full_source_bytes": budget_plan["existing_full_source_bytes"],
                          "incremental_requirement_bytes": budget_plan["incremental_requirement_bytes"]},
        retention_identity={"policy": "E5_1A_MINIMUM_ENGINEERING_RETENTION_V1"},
        resource_evidence=resource_evidence,
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
        return self._run_pulse("R", pulse)

    def run_candidate(self, pulse: int):
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

    @staticmethod
    def _join_hydro_workers(processes: Sequence[subprocess.Popen],
                            stderr_paths: Mapping[int, Path] | None = None) -> None:
        if not processes:
            return
        failures: list[str] = []
        for process in processes:
            try:
                process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
                failures.append(f"pid={process.pid}:hydro worker timeout")
                continue
            if process.returncode != 0:
                stderr_path = (stderr_paths or {}).get(process.pid)
                stderr = stderr_path.read_text(encoding="utf-8") if stderr_path is not None else ""
                failures.append(f"pid={process.pid}:exit={process.returncode}:stderr={stderr.strip()}")
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
        for process in processes:
            if process.poll() is None:
                process.kill()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

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
                environment["HR4E5S_WORKER_ID"] = f"e5_1b_c_hydro_{pulse}_{worker_index}"
                hydro_grants.append(str(hydro_grant["grant_id"]))
                hydro_process, hydro_stderr = self._start_logged_child(
                    self._hydro_worker_command(
                        state, pulse, worker_index, intent_id=intent["intent_id"],
                        reservation_id=reservation.reservation_id,
                        writer_id=hydro_grant["grant_id"], grant=hydro_grant,
                        scientific_pair_id=pair_id,
                        producer_grant_id=producer_grant_id,
                    ), state=state, grant_id=hydro_grant["grant_id"],
                    environment=environment,
                )
                hydro_processes.append(hydro_process)
                hydro_stderr_paths[hydro_process.pid] = hydro_stderr
            if optical_launch is not None:
                bind_deadline = time.monotonic() + 30.0
                while True:
                    states = self.budget.writer_grants()
                    if all(states.get(grant_id, {}).get("state") == "BOUND_ACTIVE" for grant_id in hydro_grants):
                        break
                    if any(process.poll() is not None for process in hydro_processes):
                        failed = next(process for process in hydro_processes if process.poll() is not None)
                        failed.wait(timeout=5)
                        stderr = hydro_stderr_paths[failed.pid].read_text(encoding="utf-8")
                        raise StorageIntegrityError(
                            f"hydro child exited before grant bind: exit={failed.returncode}:stderr={stderr.strip()}"
                        )
                    if time.monotonic() >= bind_deadline:
                        raise TimeoutError("hydro child grant bind exceeded 30 seconds")
                    time.sleep(0.01)
                command, optical_environment = optical_launch
                optical_process, optical_stderr_path = self._start_logged_child(
                    command, state=state, grant_id=producer_grant_id,
                    environment=optical_environment,
                )
            deadline = time.monotonic() + 900.0
            while any(process.poll() is None for process in hydro_processes):
                if optical_process is not None and optical_process.poll() is not None and optical_process.returncode != 0:
                    optical_process.wait(timeout=5)
                    stderr = optical_stderr_path.read_text(encoding="utf-8") if optical_stderr_path else ""
                    raise RuntimeError(
                        f"Candidate optical worker failed: exit={optical_process.returncode}:stderr={stderr.strip()}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Candidate hydro workers exceeded 900 seconds")
                time.sleep(0.02)
            self._join_hydro_workers(hydro_processes, hydro_stderr_paths)
            if optical_process is not None:
                optical_process.wait(timeout=900)
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
            raise
        finally:
            if error is not None:
                self._stop_hydro_workers(hydro_processes)
                if optical_process is not None and optical_process.poll() is None:
                    optical_process.terminate()
                    try:
                        optical_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        optical_process.kill(); optical_process.wait()
                self.budget.interrupt_intent(intent["intent_id"], reason=str(error))
            self._close_writer(writer, failed=error)

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
    identity, schedule, components = _build_admission(spec)
    root = _checked_campaign_root(spec.root); root.mkdir(parents=True, exist_ok=True)
    reopening = (root / CAMPAIGN_STATE_FILENAME).is_file()
    persist_admission_identity(root / "E5_1A_ADMISSION_IDENTITY.json", identity)
    current_implementation = capture_implementation_identity()
    implementation_path = root / "E5_1B_IMPLEMENTATION_IDENTITY.json"
    if implementation_path.is_file():
        persisted_implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
        require_same_implementation(persisted_implementation, current_implementation)
    elif reopening:
        raise StorageIntegrityError("reopened WP2 campaign lacks implementation identity")
    else:
        atomic_json(implementation_path, current_implementation, overwrite=False)
    runner = _E5AProductionRunner(spec, identity, schedule, components, current_implementation)
    driver = _open_production_driver(
        root, admission_identity=identity, runner=runner, n_pulses=int(spec.n_pulses),
        campaign_id=str(spec.campaign_id), max_campaign_live_bytes=int(spec.max_campaign_live_bytes),
        final_output_budget_bytes=int(spec.final_output_budget_bytes),
        safety_margin_bytes=int(spec.safety_margin_bytes), require_quota=True,
        takeover=reopening,
    )
    runner.driver = driver
    if identity["budget_identity"].get("prefix_inside_campaign_root"):
        driver.campaign.storage.register_preexisting_prefix(
            identity["pre0_identity"]["path"],
            expected_sha256=identity["pre0_identity"]["sha256"],
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
           "LOCAL_ORCHESTRATION_QUALIFICATION", "open_e5_1a_production_campaign"]

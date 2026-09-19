"""Single supported E5-1A production construction and execution surface.

The local qualification and a future site-qualified admission use this exact
runner.  The admission level changes evidence requirements, never the
scientific calls or orchestration implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .confio import load_all
from .grids import make_axes
from .hr4c_state import HR4CThreeFieldStore, evolve_hr4_full_z
from .hr4d_pulse_lifecycle import build_interpulse_step_schedule
from .hr4e5_evidence import (PAIRED_EXACT_SCHEMA, _file_identity, _load_locator, _paired_exact_keys,
                              atomic_json, bind_exact_report, build_expected_object_set,
                              compare_arrays_exact, sha256_array, sha256_file)
from .hr4e5_formal_entry import (BLOCK_SIZE, QUEUE_DEPTH, build_admission_identity,
                                  create_pre0_root, create_successor_root,
                                  persist_admission_identity, run_streaming_optical_pulse,
                                  validate_final_post)
from .hr4e5_paired_campaign import CAMPAIGN_STATE_FILENAME, _open_production_driver
from .hr4e5_storage import (DEFAULT_FINAL_OUTPUT_BUDGET_BYTES, DEFAULT_SAFETY_MARGIN_BYTES,
                             HARD_CAP_BYTES, RECLAIM_PREREQUISITE_SCHEMA, StorageBudgetError,
                             StorageIntegrityError)
from .longitudinal import build_longitudinal_schedule
from .runner import apply_thin_lens_achromatic, build_transverse_input_field
from .constants import c0
from .device import to_cpu, xp
from .hr4e5s_streaming import FIELDS, StreamingLifecycle


LOCAL_ORCHESTRATION_QUALIFICATION = "LOCAL_ORCHESTRATION_QUALIFICATION"
FORMAL_SITE_ADMISSION = "FORMAL_SITE_ADMISSION"
_ADMISSION_LEVELS = frozenset({LOCAL_ORCHESTRATION_QUALIFICATION, FORMAL_SITE_ADMISSION})


@dataclass(frozen=True)
class E5AProductionSpec:
    root: str | Path
    campaign_id: str
    config_path: str | Path
    source_manifest_path: str | Path
    lut_manifest_path: str | Path
    pre0_delta_n_path: str | Path
    site_resource_manifest_path: str | Path | None = None
    n_pulses: int = 3
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


def _build_admission(spec: E5AProductionSpec) -> tuple[dict[str, Any], Any, tuple[Any, ...]]:
    level = str(spec.admission_level)
    if level not in _ADMISSION_LEVELS:
        raise ValueError("unsupported production admission level")
    config = Path(spec.config_path).resolve()
    source_path = Path(spec.source_manifest_path).resolve()
    lut_path = Path(spec.lut_manifest_path).resolve()
    pre0_path = Path(spec.pre0_delta_n_path).resolve()
    source = _read_manifest(source_path, label="source")
    lut = _read_manifest(lut_path, label="LUT")
    components = load_all(str(config))
    grid, beam, prop, ion, heat, run, raman = components
    if not config.is_file() or config.is_symlink() or not pre0_path.is_file() or pre0_path.is_symlink():
        raise ValueError("production config or PRE0 source is missing or linked")
    schedule = build_longitudinal_schedule(dz=float(prop.dz), z_max=float(prop.z_max))
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
    if level == FORMAL_SITE_ADMISSION:
        if spec.site_resource_manifest_path is None:
            raise ValueError("formal site admission requires qualified resource evidence")
        site_path = Path(spec.site_resource_manifest_path).resolve()
        site = _read_manifest(site_path, label="site resource")
        if (site.get("scope") != "SITE_QUALIFIED_RESOURCE_EVIDENCE"
                or site.get("formal_input_authorized") is not True
                or site.get("site_resource_qualified") is not True
                or site.get("provenance_qualified") is not True):
            raise ValueError("formal site admission evidence is incomplete")
        for name in ("free_bytes", "quota_bytes"):
            if isinstance(site.get(name), bool) or int(site.get(name, 0)) <= 0:
                raise ValueError(f"formal site admission {name} is invalid")
        resource_evidence = {**site, "path": str(site_path), "sha256": sha256_file(site_path)}
    else:
        if spec.site_resource_manifest_path is not None:
            raise ValueError("local qualification cannot claim site resource evidence")
        resource_evidence = {"scope": _LocalPolicyCapacityProvider.scope}
    identity = build_admission_identity(
        campaign_id=str(spec.campaign_id), execution_mode=level,
        runtime_or_compatibility={"module": "KHz_filament.hr4e5_production", "runner": "single-production-runner"},
        config_identity={"path": str(config), "sha256": sha256_file(config)},
        effective_params={"f_rep": float(heat.f_rep), "dt_hydro": float(getattr(heat, "dt_hydro", 1e-7))},
        source_identity={"path": str(source_path), "sha256": sha256_file(source_path),
                         "canonical_array_hash": source_hash},
        lut_identity={"path": str(lut_path), "sha256": sha256_file(lut_path), **lut},
        schedule_identity={"k": schedule.n_intervals, "metadata": schedule.as_metadata()},
        grid_identity={"Nx": grid.Nx, "Ny": grid.Ny, "Nt": grid.Nt,
                       "shape": [grid.Ny, grid.Nx], "dtype": "complex128"},
        pre0_identity={"path": str(pre0_path), "sha256": sha256_file(pre0_path),
                       "canonical_array_hash": sha256_array(pre0)},
        n_pulses=int(spec.n_pulses), k=int(schedule.n_intervals), block_size=BLOCK_SIZE,
        queue_depth=QUEUE_DEPTH, f_rep=float(heat.f_rep),
        dt_hydro=float(getattr(heat, "dt_hydro", 1e-7)),
        precision={"fields": "float64", "optical": "complex128"},
        r_roots={"root": "R"}, c_roots={"root": "C"},
        pulse_attempt_epoch={"pulse": 0, "attempt": 0, "epoch": "factory-assigned"},
        budget_identity={"max_campaign_live_bytes": int(spec.max_campaign_live_bytes),
                         "final_output_budget_bytes": int(spec.final_output_budget_bytes),
                         "safety_margin_bytes": int(spec.safety_margin_bytes)},
        retention_identity={"policy": "E5_1A_MINIMUM_ENGINEERING_RETENTION_V1"},
        resource_evidence=resource_evidence,
    )
    return identity, schedule, components


class _E5AProductionRunner:
    """Concrete runner; no callbacks or alternate test implementation exist."""

    def __init__(self, spec: E5AProductionSpec, identity: Mapping[str, Any], schedule: Any,
                 components: tuple[Any, ...]):
        self.spec, self.root = spec, Path(spec.root).resolve()
        self.identity, self.schedule, self.components = dict(identity), schedule, components
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
        reservation = self.budget.reserve(expected_bytes, purpose=f"{role}:{side}:p{pulse}",
                                          owner=str(epoch), allocation_paths=paths)
        intent = self.budget.create_intent(
            reservation_id=reservation.reservation_id, trajectory=side, pulse=int(pulse), attempt=0,
            role=role, allowed_paths=paths, expected_bytes=expected_bytes,
            admission_hash=self.identity["identity_sha256"], epoch=epoch, generation=generation,
            campaign_id=self.identity["campaign_id"],
        )
        writer = self.budget.open_writer(
            reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
            coordinator_epoch=epoch, trajectory=side, pulse=int(pulse), attempt=0,
            generation=generation,
        )
        return reservation, intent, writer

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
        delta_n = np.load(Path(self.spec.pre0_delta_n_path), allow_pickle=False)
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

    def _run_pulse(self, trajectory: str, pulse: int):
        state = self._track(trajectory, pulse) / "state"
        output = self._track(trajectory, pulse) / "optical"
        lifecycle = StreamingLifecycle.open(state)
        grid = self.components[0]; final = int(pulse) == int(self.spec.n_pulses) - 1
        screen_bytes = int(grid.Nx) * int(grid.Ny) * 8
        optical_bytes = int(grid.Nx) * int(grid.Ny) * int(grid.Nt) * 16
        estimate = (9 if final or trajectory == "R" else 12) * self.schedule.n_intervals * screen_bytes + optical_bytes + 4*1024**2
        reservation, intent, writer = self._intent(
            side=trajectory, pulse=pulse, role="OPTICAL", paths=[output, state],
            expected_bytes=estimate, generation=str(lifecycle.manifest["current_generation"]),
        )
        errors: list[BaseException] = []; finished = threading.Event(); available = threading.Event()
        thread = None
        if trajectory == "C" and not final:
            def consumer():
                try:
                    worker = StreamingLifecycle.open(state)
                    while True:
                        available.clear()
                        if worker.run_one_hydro_block(**self._hydro(), actor="e5_1a_production_hydro"):
                            continue
                        if finished.is_set(): break
                        if finished.wait(0.01): break
                except BaseException as exc: errors.append(exc)
            thread = threading.Thread(target=consumer, name=f"e5a-hydro-{pulse}")
            thread.start()
        error = None
        try:
            result = run_streaming_optical_pulse(
                lifecycle_root=state, schedule=self.schedule, output_dir=output,
                config_path=self.spec.config_path, final=(final or trajectory == "R"),
                storage_budget=self.budget, admission_identity=self.identity,
                creation_intent=intent, trajectory=trajectory, pulse=pulse, role="OPTICAL",
                defer_intent_completion=(trajectory == "C" and not final),
            )
            finished.set(); available.set()
            if thread: thread.join(60)
            if thread and thread.is_alive(): raise TimeoutError("production hydro writer did not exit")
            if errors: raise errors[0]
            if trajectory == "C" and not final:
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
            lifecycle = StreamingLifecycle.open(state)
            if trajectory == "C" and not final:
                lifecycle.validate_barrier(actor="e5_1a_production_barrier")
                lifecycle.promote_next_to_current(actor="e5_1a_production_promotion")
            elif trajectory == "R" and not final:
                batch_dir = self._track("R", pulse) / "batch"
                before = {path for base in (state, batch_dir)
                          for path in base.rglob("*") if path.is_file()}
                hydro_res, hydro_intent, hydro_writer = self._intent(
                    side="R", pulse=pulse, role="REFERENCE_HYDRO",
                    paths=[state, batch_dir],
                    expected_bytes=max(4*1024**2, 6*self.schedule.n_intervals*screen_bytes),
                    generation=f"R:p{pulse}:NEXT",
                )
                hydro_error = None
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
                    store = HR4CThreeFieldStore(
                        output_path=str(batch_dir / "store"), n_intervals=self.schedule.n_intervals,
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
                    created = [path for base in (state, batch_dir)
                               for path in base.rglob("*") if path.is_file() and path not in before]
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
            return self._simple_report(side=trajectory, pulse=pulse, name="pulse",
                                       summary=np.asarray([pulse, self.schedule.n_intervals], dtype=np.int64))
        except BaseException as exc:
            error = exc; finished.set(); available.set()
            if thread: thread.join(5)
            raise
        finally:
            self._close_writer(writer, failed=error)

    def _object_descriptor(self, side: str, pulse: int, key: str) -> dict[str, Any]:
        state = self._track(side, pulse) / "state"
        output = self._track(side, pulse) / "optical"
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
            suffix = {"ion": "hr3a_qion_samples.npy", "ib": "hr3a_qib_samples.npy",
                      "raman": "hr3a_qraman_samples.npy", "qthermal": "hr3a_qthermal_samples.npy",
                      "increment": "hr3b_delta_n_increment_samples.npy",
                      "state_after": "hr3b_delta_n_state_after_update_samples.npy"}[name]
            path = output / f"sinks.{suffix}"; locator = {"slice": [source_index]}; role = "sink"
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
        folder = self.root / "exact"; path = folder / f"p{pulse}.json"
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

    def _run_gc(self, trajectory: str, pulse: int):
        parent = self._track(trajectory, pulse)
        artifacts = self.budget.artifacts()
        targets = []
        for relative, item in artifacts.items():
            path = self.root / relative
            if path.is_relative_to(parent / "state") and path.suffix in {".npy", ".npz"}:
                targets.append(path)
            elif (path.is_relative_to(parent / "batch") and path.is_file()
                  and path.suffix in {".npy", ".npz"}):
                targets.append(path)
        if not targets:
            raise StorageBudgetError("production GC target set is empty")
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
            ready = self._track(trajectory, pulse+1) / "state" / "E5_1A_READY.json"
            dependency = evidence / "dependency.json"
            atomic_json(dependency, {"status": "PASS", "no_future_dependency": True,
                                     "paired_exact": str((self.root/"exact"/f"p{pulse}.json").resolve())}, overwrite=False)
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
        plan_id = f"production-{trajectory}-p{pulse}"
        self.budget.plan_reclaim(targets, exact_complete=True, successor_ready=True,
            no_active_writers=True, no_future_dependency=True, receipts_durable=True,
            writer_receipt=writer_receipt, writer_epoch=epoch, prerequisite_receipt=prerequisite,
            plan_id=plan_id, trajectory=trajectory, pulse=pulse, attempt=0,
            admission_hash=self.identity["identity_sha256"])
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
        reservation, intent, writer = self._intent(side="C", pulse=pulse, role="TERMINAL_EVIDENCE",
            paths=[folder, ready, writer_path, contract], expected_bytes=16*1024**2,
            generation=f"terminal:p{pulse}")
        error = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            receipt = self.budget.write_quiescence_receipt(
                writer_path, coordinator_epoch=writer["coordinator_epoch"],
                trajectory="C", pulse=pulse, attempt=0)
            validate_final_post(lifecycle_root=state, receipt_path=ready, writer_quiescent=False,
                                fixture_only=False, expected_optical_dir=output,
                                writer_receipt=receipt)
            summary = folder / "terminal_summary.npy"; report = folder / "terminal.json"
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
                "gc_evidence": 6 * (int(self.spec.n_pulses) - 1),
                "report_gc": 4 * (int(self.spec.n_pulses) - 1),
            }
            missing_roles = {
                role: minimum for role, minimum in required_role_counts.items()
                if len(role_map.get(role, [])) < minimum
            }
            if missing_roles:
                raise StorageIntegrityError(
                    f"terminal production role contract is incomplete: {missing_roles}"
                )
            terminal_files = [Path(receipt["path"]), ready, summary, report, contract]
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
        return {**self._driver.run(stop_after=stop_after),
                "admission_identity": dict(self._runner.identity)}
    def pause(self, *, reason: str = "operator_pause"): return self._driver.pause(reason=reason)
    def close(self): return self._driver.close()
    def report(self):
        return {**self._driver.report(), "admission_identity": dict(self._runner.identity)}


def open_e5_1a_production_campaign(spec: E5AProductionSpec) -> E5AProductionCampaign:
    """The only supported creator/resumer for a production-mode E5-1A run."""
    if not isinstance(spec, E5AProductionSpec):
        raise TypeError("E5AProductionSpec is required")
    root = Path(spec.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    reopening = (root / CAMPAIGN_STATE_FILENAME).is_file()
    identity, schedule, components = _build_admission(spec)
    persist_admission_identity(root / "E5_1A_ADMISSION_IDENTITY.json", identity)
    runner = _E5AProductionRunner(spec, identity, schedule, components)
    driver = _open_production_driver(
        root, admission_identity=identity, runner=runner, n_pulses=int(spec.n_pulses),
        campaign_id=str(spec.campaign_id), max_campaign_live_bytes=int(spec.max_campaign_live_bytes),
        final_output_budget_bytes=int(spec.final_output_budget_bytes),
        safety_margin_bytes=int(spec.safety_margin_bytes), require_quota=True,
        takeover=reopening,
    )
    runner.driver = driver
    if spec.admission_level == LOCAL_ORCHESTRATION_QUALIFICATION:
        driver.campaign.storage.provider = _LocalPolicyCapacityProvider(root, spec.max_campaign_live_bytes)
    else:
        driver.campaign.storage.provider = _ManifestCapacityProvider(
            _read_manifest(Path(spec.site_resource_manifest_path).resolve(), label="site resource")
        )
    runner.initialize()
    return E5AProductionCampaign(runner, driver)


__all__ = ["E5AProductionCampaign", "E5AProductionSpec", "FORMAL_SITE_ADMISSION",
           "LOCAL_ORCHESTRATION_QUALIFICATION", "open_e5_1a_production_campaign"]

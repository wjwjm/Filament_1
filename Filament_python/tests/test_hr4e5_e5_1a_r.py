"""Focused E5-1A-R admission, intent, evidence and coordinator checks."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def _formal_identity(campaign_id: str = "formal-r", *, n_pulses: int = 1):
    from KHz_filament.hr4e5_formal_entry import build_admission_identity

    return build_admission_identity(
        campaign_id=campaign_id, runtime_or_compatibility={"runtime": "local-test", "sha": "runtime"},
        config_identity={"path": "config.json", "sha256": "config"},
        effective_params={"dz": 1e-4, "z_max": 8e-4}, source_identity={"path": "source.npy", "sha256": "source"},
        lut_identity={"path": "lut", "sha256": "lut"}, schedule_identity={"edges": [0, 1], "sha256": "schedule"},
        grid_identity={"Nx": 4, "Ny": 4, "Nt": 4, "dtype": "complex128"},
        pre0_identity={"parent_sha256": "pre0", "derived_sha256": "prefix"}, n_pulses=n_pulses, k=8,
        f_rep=5e6, dt_hydro=1e-7, precision={"fields": "float64", "optical": "complex128"},
        r_roots={"root": "R"}, c_roots={"root": "C"},
        pulse_attempt_epoch={"pulse": 0, "attempt": 0, "epoch": "epoch-0"},
        budget_identity={"max_campaign_live_bytes": 322122547200, "final_output_budget_bytes": 68719476736},
        retention_identity={"roles": ["final", "ledger"]},
    )


def _production_runner_shell():
    """Return a typed, non-running shell for private formal-gate unit tests."""
    from KHz_filament.hr4e5_production import _E5AProductionRunner

    return _E5AProductionRunner.__new__(_E5AProductionRunner)


def test_r01_formal_identity_rejects_missing_and_tamper():
    from KHz_filament.hr4e5_formal_entry import validate_admission_identity

    with pytest.raises(ValueError, match="missing|placeholder"):
        validate_admission_identity({"schema": "khz_filament.hr4e5.e5_1a.admission.v1"}, formal=True)
    identity = _formal_identity()
    changed = dict(identity, k=16)
    with pytest.raises(ValueError, match="hash"):
        validate_admission_identity(changed, formal=True)
    fixture = dict(identity, execution_mode="TEST_FIXTURE_ONLY", scope="TEST_FIXTURE_ONLY")
    fixture["identity_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash|fixture"):
        validate_admission_identity(fixture, formal=True)


def test_r01_same_driver_reaches_real_cpu_optical_path(tmp_path):
    """The controlled driver reaches the real optical call; quota stays fixture-only."""
    from test_hr4e5_e5_1a import _fields, _records, _schedule
    from KHz_filament.config import (
        BeamConfig, GridConfig, HeatConfig, IonizationConfig, PropagationConfig,
        RamanConfig, RunConfig,
    )
    from KHz_filament.hr4e5_formal_entry import (
        build_fixture_admission_identity, run_streaming_optical_pulse,
    )
    from KHz_filament.hr4e5_paired_campaign import FormalPairedDriver
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    root = tmp_path / "controlled-real-optical"
    identity = build_fixture_admission_identity(
        campaign_id="real-optical-driver-fixture", n_pulses=1, k=8, shape=(8, 8),
    )
    grid = GridConfig(Nx=8, Ny=8, Nt=8, Lx=8e-4, Ly=8e-4, Twin=80e-15)
    components = {
        "grid": grid,
        "beam": BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-10, focal_length=None),
        "prop": PropagationConfig(
            z_max=8e-4, dz=1e-4, linear_model="paraxial", auto_substep=False,
            focus_window_step=False, limit_focus_window=False, progress_every_z=0,
            energy_probe_every=0, diag_extra=False, use_electronic_kerr=False,
            use_raman_phase=False, use_raman_absorption=False, use_plasma_phase=False,
            use_ionization_loss=False, use_ionization_solver=False,
        ),
        "ion": IonizationConfig(species=[]), "heat": HeatConfig(hr3b_enabled=True),
        "run": RunConfig(Npulses=1), "raman": RamanConfig(enabled=False, absorption=False),
    }
    lifecycle = StreamingLifecycle.create(
        root=root / "candidate_state", current=_fields(shape=(8, 8)),
        screen_records=_records(), current_generation="fixture:pre",
        dx_m=grid.Lx / grid.Nx, dy_m=grid.Ly / grid.Ny,
    )

    class Runner:
        def __init__(self):
            self.budget = None
            self.steps = []

        def _report(self, name):
            self.steps.append(name)
            return _write_formal_simple_report(
                root, root / f"{name}.json", identity,
                "R" if name == "R" else "C",
            )

        def run_reference(self, pulse):
            return self._report("R")

        def run_candidate(self, pulse):
            result = run_streaming_optical_pulse(
                lifecycle_root=lifecycle.root, schedule=_schedule(),
                output_dir=root / "candidate_optical", components=components,
                final=True, storage_budget=self.budget,
                admission_identity=identity, fixture_only=True,
                trajectory="C", pulse=int(pulse),
            )
            assert result["status"] == "PASS" and result["schedule_intervals"] == 8
            return self._report("C")

        def run_exact(self, pulse):
            return self._report("exact")

        def run_terminal(self, pulse):
            return self._report("terminal")

    runner = Runner()
    driver = FormalPairedDriver(
        root, admission_identity=identity, runner=runner, n_pulses=1,
        campaign_id=identity["campaign_id"], fixture_only=True,
        require_quota=False, safety_margin_bytes=1024**2,
    )
    runner.budget = driver.campaign.storage
    result = driver.run()
    assert result["status"] == "COMPLETE"
    assert runner.steps == ["R", "C", "exact", "terminal"]
    assert (root / "candidate_optical" / "final_optical_field.npy").is_file()
    reopened = StreamingLifecycle.open(lifecycle.root)
    assert all(record["state"] == "POST_COMMITTED" for record in reopened.manifest["records"])
    assert not list((lifecycle.root / "next").glob("*.npz"))


def test_r03_intent_preexisting_file_is_not_retroactively_owned(tmp_path):
    from KHz_filament.hr4e5_storage import MockQuotaProvider, StorageBudget, StorageIntegrityError

    identity = _formal_identity()
    root = tmp_path / "campaign"
    target = root / "payload"
    target.mkdir(parents=True)
    old = target / "old.npy"
    np.save(old, np.ones(2, dtype=np.float64))
    budget = StorageBudget(root, campaign_id=identity["campaign_id"], require_intents=True,
                           admission_hash=identity["identity_sha256"], require_quota=True,
                           provider=MockQuotaProvider(free_bytes=2**30, quota_bytes=2**30), safety_margin_bytes=128)
    reservation = budget.reserve(4096, purpose="payload", allocation_paths=[target])
    intent = budget.create_intent(reservation_id=reservation.reservation_id, trajectory="R", pulse=0,
                                  attempt=0, role="PRE0", allowed_paths=[target], expected_bytes=4096,
                                  admission_hash=identity["identity_sha256"])
    new = target / "new.npy"
    np.save(new, np.zeros(2, dtype=np.float64))
    with pytest.raises(StorageIntegrityError, match="pre-existing"):
        budget.complete_intent(intent["intent_id"])
    assert budget.intents()[intent["intent_id"]]["status"] == "ACTIVE"


def test_r03_terminal_inventory_requires_quiescent_storage_ledger(tmp_path):
    from KHz_filament.hr4e5_storage import (
        MockQuotaProvider, StorageBudget, StorageIntegrityError,
    )

    identity = _formal_identity(campaign_id="terminal-quiescence-r")
    root = tmp_path / "terminal-quiescence"
    target = root / "payload.npy"
    budget = StorageBudget(
        root, campaign_id=identity["campaign_id"], require_intents=True,
        admission_hash=identity["identity_sha256"], require_quota=True,
        provider=MockQuotaProvider(free_bytes=2**30, quota_bytes=2**30),
        safety_margin_bytes=128,
    )
    reservation = budget.reserve(4096, purpose="unfinished", allocation_paths=[target])
    intent = budget.create_intent(
        reservation_id=reservation.reservation_id, trajectory="C", pulse=0,
        attempt=0, role="FINAL", allowed_paths=[target], expected_bytes=4096,
        admission_hash=identity["identity_sha256"], epoch="epoch-0",
        generation="terminal:test",
    )
    writer = budget.open_writer(
        reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
        coordinator_epoch="epoch-0", trajectory="C", pulse=0, attempt=0,
        generation="terminal:test",
    )
    with pytest.raises(StorageIntegrityError, match="not quiescent") as active:
        budget.validate_terminal_inventory({})
    assert "active_reservations" in str(active.value)
    assert "unfinished_intents" in str(active.value)
    assert "active_writers" in str(active.value)
    budget.close_writer(writer["writer_id"], coordinator_epoch="epoch-0", status="INTERRUPTED")
    budget.interrupt_intent(intent["intent_id"], reason="test interruption")
    with pytest.raises(StorageIntegrityError, match="unfinished_intents"):
        budget.validate_terminal_inventory({})


def test_r03_formal_pre0_refuses_without_intent_and_fixture_is_explicit(tmp_path):
    from KHz_filament.hr4e5_formal_entry import build_fixture_admission_identity, create_pre0_root
    from KHz_filament.hr4e5_storage import StorageBudget
    from KHz_filament.longitudinal import build_longitudinal_schedule

    schedule = build_longitudinal_schedule(dz=1e-4, z_max=8e-4)
    fields = np.zeros((8, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="admission identity"):
        create_pre0_root(root=tmp_path / "formal", delta_n=fields, schedule=schedule, dx_m=1e-4, dy_m=1e-4)
    fixture = build_fixture_admission_identity(k=8, shape=(2, 2))
    lifecycle = create_pre0_root(root=tmp_path / "fixture", delta_n=fields, schedule=schedule,
                                 dx_m=1e-4, dy_m=1e-4, fixture_only=True,
                                 admission_identity=fixture)
    metadata = json.loads((lifecycle.root / "E5_1A_ROOT_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["scope"] == "TEST_FIXTURE_ONLY"

    identity = _formal_identity(campaign_id="unknown-quota-r")
    campaign_root = tmp_path / "formal_unknown_quota"
    target = campaign_root / "R" / "PRE0"
    budget = StorageBudget(
        campaign_root,
        campaign_id=identity["campaign_id"],
        require_intents=True,
        admission_hash=identity["identity_sha256"],
        require_quota=False,
        safety_margin_bytes=128,
    )
    reservation = budget.reserve(1024**2, purpose="PRE0", allocation_paths=[target])
    intent = budget.create_intent(
        reservation_id=reservation.reservation_id,
        trajectory="R",
        pulse=0,
        attempt=0,
        role="PRE0",
        allowed_paths=[target],
        expected_bytes=1024**2,
        admission_hash=identity["identity_sha256"],
        generation="E5:E5_1A:PRE0",
        campaign_id=identity["campaign_id"],
    )
    with pytest.raises(ValueError, match="quota"):
        create_pre0_root(
            root=target,
            delta_n=fields,
            schedule=schedule,
            dx_m=1e-4,
            dy_m=1e-4,
            admission_identity=identity,
            storage_budget=budget,
            creation_intent=intent,
        )
    assert not target.exists()


def test_r02_object_binding_rejects_replaced_array(tmp_path):
    from KHz_filament.hr4e5_evidence import build_expected_object_set, validate_object_set

    root = tmp_path / "evidence"
    root.mkdir()
    path = root / "array.npy"
    np.save(path, np.arange(4, dtype=np.float64))
    rows = build_expected_object_set({"a": path}, campaign_id="c", trajectory="R", pulse=0,
                                    attempt=0, namespace="POST", root=root, source_indices={"a": 0})
    assert validate_object_set(rows, campaign_id="c", root=root)["status"] == "PASS"
    np.save(path, np.arange(4, dtype=np.float64) + 1)
    with pytest.raises(ValueError, match="identity|hash"):
        validate_object_set(rows, campaign_id="c", root=root)


def test_r10_budget_plan_is_scalar_and_refuses_unbudgeted_full_sinks():
    from KHz_filament.hr4e5_storage import (StorageBudgetError, budget_object_table,
                                            plan_campaign_budget, plan_final_output)

    args = dict(n_pulses=3, k=8048, ny=351, nx=301, nt=384)
    objects = budget_object_table(**args)
    assert objects["stage_bytes"][objects["peak_stage"]] == sum(
        row["bytes"] for row in objects["stages"][objects["peak_stage"]])
    assert objects["stage_bytes"]["final_retention"] < 68719476736
    assert objects["peak_bytes"] < 322122547200
    assert objects["external_prefix_bytes"] == 6802233984
    plan = plan_campaign_budget(**args)
    assert plan["peak_bytes"] == objects["peak_bytes"]
    assert plan["final_output_bytes"] == objects["final_output_bytes"]
    assert plan_campaign_budget(**args, max_campaign_live_bytes=plan["peak_bytes"])["headroom_bytes"] == 0
    with pytest.raises(StorageBudgetError, match="exceeds cap"):
        plan_campaign_budget(**args, max_campaign_live_bytes=plan["peak_bytes"] - 1)
    assert objects["final_output_bytes"] + 36 * objects["field_bytes"] > 68719476736
    with pytest.raises(StorageBudgetError, match="exceed"):
        plan_final_output(**args, diagnostics=("ion", "ib", "raman", "qthermal", "increment", "state_after"))


def _write_formal_simple_report(root, path, identity, trajectory, *, writer=None):
    from KHz_filament.hr4e5_evidence import bind_exact_report, build_expected_object_set, write_exact_report

    array = root / f"{Path(path).stem}_{trajectory.lower()}_summary.npy"
    np.save(array, np.arange(4, dtype=np.float64))
    rows = build_expected_object_set(
        {"a": array}, campaign_id=identity["campaign_id"], trajectory=trajectory,
        pulse=0, attempt=0, namespace="POST", root=root, source_indices={"a": 0},
    )
    result = bind_exact_report(path, {"status": "PASS", "mismatch_count": 0}, rows,
                               campaign_id=identity["campaign_id"], root=root)
    if writer is not None:
        payload = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
        payload.update(writer)
        result = write_exact_report(result["report_path"], payload)
    return result


def _formal_pair_report(root, path, identity, budget):
    from KHz_filament.hr4e5_evidence import (
        PAIRED_EXACT_SCHEMA, _file_identity, _paired_exact_keys, atomic_json,
        sha256_array, sha256_file,
    )

    n_pulses = int(identity["n_pulses"])
    expected_keys = sorted(_paired_exact_keys(k=int(identity["k"]), n_pulses=n_pulses, pulse=0))
    descriptors = {"R": {}, "C": {}}
    for side in ("R", "C"):
        object_root = root / "objects" / side
        reservation = budget.reserve(
            len(expected_keys) * 1024,
            purpose=f"formal exact {side} objects",
            allocation_paths=[object_root],
        )
        intent = budget.create_intent(
            reservation_id=reservation.reservation_id,
            trajectory=side,
            pulse=0,
            attempt=0,
            role="exact_object",
            allowed_paths=[object_root],
            expected_bytes=len(expected_keys) * 1024,
            admission_hash=identity["identity_sha256"],
            generation="generation-0",
            campaign_id=identity["campaign_id"],
        )
        created = []
        for key in expected_keys:
            target = object_root / (key.replace(":", "_") + ".npy")
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, np.asarray([1.0], dtype=np.float64))
            created.append(target)
        completed = budget.complete_intent(
            intent["intent_id"], files=created, role="exact_object",
            metadata={"reclaimable": True},
        )
        budget.consume(reservation.reservation_id, actual_bytes=int(completed["actual_bytes"]))
        artifacts = budget.artifacts()
        for key, target in zip(expected_keys, created):
            value = np.load(target, allow_pickle=False)
            ownership = artifacts[target.relative_to(root).as_posix()]
            descriptors[side][key] = {
                "trajectory": side, "campaign_id": identity["campaign_id"], "pulse": 0,
                "attempt": 0, "path": target.relative_to(root).as_posix(),
                "relative_path": target.relative_to(root).as_posix(), "locator": {},
                "shape": list(value.shape), "dtype": value.dtype.name, "finite": True,
                "canonical_array_hash": sha256_array(value), "file_sha256": sha256_file(target),
                "file_identity": _file_identity(target),
                "creation_record": dict(ownership["creation_record"]),
            }

    rows = []
    for key in expected_keys:
        sides = {}
        key_parts = key.split(":")
        if key_parts[0] == "screen":
            descriptor_fields = {"role": "screen", "namespace": key_parts[1], "source_index": int(key_parts[2]), "name": key_parts[3]}
        elif key_parts[0] == "sink":
            descriptor_fields = {"role": "sink", "namespace": "sink", "source_index": int(key_parts[2]), "name": key_parts[1]}
        elif key_parts[0] == "ledger":
            descriptor_fields = {"role": "ledger", "namespace": "ledger", "source_index": None, "name": key_parts[1]}
        else:
            descriptor_fields = {"role": "final_optical", "namespace": "final", "source_index": None, "name": "final_optical_field"}
        for side in ("R", "C"):
            sides[side] = {**descriptors[side][key], **descriptor_fields}
        rows.append({"status": "PASS", "comparison_key": key,
                     "reference": sides["R"], "candidate": sides["C"],
                     "reference_sha256_array": sides["R"]["canonical_array_hash"],
                     "candidate_sha256_array": sides["C"]["canonical_array_hash"]})
    payload = {
        "schema": PAIRED_EXACT_SCHEMA, "status": "PASS",
        "admission_identity_sha256": identity["identity_sha256"],
        "campaign_id": identity["campaign_id"], "pulse": 0, "attempt": 0,
        "screen_count": (12 if n_pulses == 1 else 15) * int(identity["k"]), "ledger_count": 9, "optical_count": 1,
        "expected_object_count": len(rows), "compared_object_count": len(rows),
        "mismatch_count": 0, "rows": rows,
    }
    atomic_json(path, payload)
    return {**payload, "report_path": str(path), "report_sha256": sha256_file(path)}


def _complete_formal_gc(root, budget, identity, *, side, pulse, targets, writer_epoch):
    from KHz_filament.hr4e5_evidence import (
        atomic_json, bind_exact_report, build_expected_object_set, sha256_file,
    )
    from KHz_filament.hr4e5_storage import (
        RECLAIM_PREREQUISITE_SCHEMA, WRITER_RECEIPT_SCHEMA, StorageIntegrityError,
    )

    writer = root / f"writer_{side}{pulse}.json"
    reservation = budget.reserve(
        1024**2, purpose=f"gc-writer-{side}-{pulse}",
        allocation_paths=[writer],
    )
    intent = budget.create_intent(
        reservation_id=reservation.reservation_id, trajectory=side,
        pulse=int(pulse), attempt=0, role="GC_EVIDENCE",
        allowed_paths=[writer], expected_bytes=1024**2,
        admission_hash=identity["identity_sha256"], epoch=writer_epoch,
        generation=f"gc:{side}:p{pulse}",
    )
    writer_record = budget.open_writer(
        reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
        coordinator_epoch=writer_epoch, trajectory=side, pulse=int(pulse),
        attempt=0, generation=f"gc:{side}:p{pulse}",
    )
    budget.close_writer(writer_record["writer_id"], coordinator_epoch=writer_epoch)
    writer_receipt = budget.write_quiescence_receipt(
        writer, coordinator_epoch=writer_epoch, trajectory=side,
        pulse=int(pulse), attempt=0,
    )
    assert writer_receipt["schema"] == WRITER_RECEIPT_SCHEMA
    assert writer_receipt["registry_backed"] is True
    completed_writer = budget.complete_intent(intent["intent_id"], files=[writer])
    budget.consume(reservation.reservation_id, actual_bytes=completed_writer["actual_bytes"])
    assert budget.active_writers(coordinator_epoch=writer_epoch, include_management=False) == []
    evidence = root / f"evidence_formal_{side}{pulse}"
    evidence.mkdir(parents=True, exist_ok=True)
    exact = evidence / "exact.json"
    object_rows = build_expected_object_set(
        {f"target_{index}": target for index, target in enumerate(targets)},
        campaign_id=identity["campaign_id"], trajectory=side,
        pulse=int(pulse), attempt=0, namespace="gc_target", root=root,
    )
    bind_exact_report(
        exact, {
            "status": "PASS", "mismatch_count": 0,
            "missing_reference": [], "missing_candidate": [],
        }, object_rows, campaign_id=identity["campaign_id"], root=root,
    )
    child = evidence / "child"
    child.mkdir(parents=True, exist_ok=True)
    child_field = child / "field.bin"
    child_field.write_bytes(b"self-contained-successor")
    child_binding = child / "binding.json"
    atomic_json(child_binding, {"status": "PASS", "rows": [{"status": "PASS"}]})
    ready = evidence / "READY.json"
    atomic_json(ready, {
        "status": "READY", "self_contained": True,
        "parent_payload_required": False,
        "child_root": str(child),
        "fields": {"field": {"path": "field.bin", "sha256": sha256_file(child_field)}},
        "binding": {"path": str(child_binding), "sha256": sha256_file(child_binding)},
    })
    dependency = evidence / "dependency.json"
    atomic_json(dependency, {"status": "PASS", "no_future_dependency": True})
    artifacts = budget.artifacts()
    target_bindings = []
    for target in targets:
        relative = target.resolve().relative_to(root).as_posix()
        item = artifacts[relative]
        target_bindings.append({
            "relative_path": relative, "identity": dict(item["identity"]),
            "sha256": item["sha256"],
        })
    prerequisite = evidence / "prerequisites.json"
    atomic_json(prerequisite, {
        "schema": RECLAIM_PREREQUISITE_SCHEMA, "status": "PASS",
        "campaign_id": identity["campaign_id"],
        "gates": {
            "exact_complete": True, "successor_ready": True,
            "no_active_writers": True, "no_future_dependency": True,
            "receipts_durable": True,
        },
        "target_bindings": target_bindings,
        "evidence": {
            "exact_complete": {"path": str(exact), "sha256": sha256_file(exact)},
            "successor_ready": {"path": str(ready), "sha256": sha256_file(ready)},
            "no_future_dependency": {
                "path": str(dependency), "sha256": sha256_file(dependency),
            },
        },
    })
    plan = budget.plan_reclaim(
        targets, exact_complete=True, successor_ready=True,
        no_active_writers=True, no_future_dependency=True,
        receipts_durable=True, writer_receipt=writer_receipt,
        prerequisite_receipt=prerequisite, plan_id=f"formal-{side}{pulse}",
        trajectory=side, pulse=int(pulse), attempt=0,
        admission_hash=identity["identity_sha256"],
    )
    wrong_side = "C" if side == "R" else "R"
    with pytest.raises(StorageIntegrityError, match="trajectory"):
        budget.apply_reclaim(
            plan["plan_id"], expected_trajectory=wrong_side,
            expected_pulse=int(pulse), expected_attempt=0,
            expected_admission_hash=identity["identity_sha256"],
        )
    assert all(target.is_file() for target in targets)
    completed = budget.apply_reclaim(
        plan["plan_id"], expected_trajectory=side,
        expected_pulse=int(pulse), expected_attempt=0,
        expected_admission_hash=identity["identity_sha256"],
    )
    assert completed["status"] == "COMPLETED"
    verified = budget.verify_reclaim(
        plan["plan_id"], expected_trajectory=side, expected_pulse=int(pulse),
        expected_attempt=0, expected_admission_hash=identity["identity_sha256"],
    )
    assert verified["status"] == "PASS"
    report = root / f"gc_report_{side}{pulse}.json"
    atomic_json(report, {
        "status": "PASS", "campaign_id": identity["campaign_id"],
        "gc_plan_id": plan["plan_id"],
        "writer_receipt_path": str(writer),
        "writer_receipt_sha256": sha256_file(writer),
    })
    return {
        "status": "PASS", "report_path": str(report),
        "report_sha256": sha256_file(report),
        "gc_plan_id": plan["plan_id"],
        "writer_receipt_path": str(writer),
        "writer_receipt_sha256": sha256_file(writer),
    }


def test_r06_formal_exact_rejects_r_only_and_accepts_a_complete_paired_report(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import _open_production_driver

    identity = _formal_identity()

    def run_case(root):
        root.mkdir(exist_ok=True)
        driver = _open_production_driver(root, admission_identity=identity, runner=_production_runner_shell(), n_pulses=1,
                                         campaign_id=identity["campaign_id"])
        from KHz_filament.hr4e5_storage import MockQuotaProvider
        driver.campaign.storage.provider = MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40)
        exact = _formal_pair_report(root, root / "paired_exact.json", identity, driver.campaign.storage)
        result = driver._validate_receipt(
            {"status": "PASS", "report_path": exact["report_path"]},
            pulse=0, step="exact",
        )
        driver.close()
        return result

    bad_root = tmp_path / "r_only"
    bad_root.mkdir()
    report_r = _write_formal_simple_report(bad_root, bad_root / "reference.json", identity, "R")
    report_c = _write_formal_simple_report(bad_root, bad_root / "candidate.json", identity, "C")

    class BadRunner:
        def run_reference(self, pulse): return {"status": "PASS", "report_path": report_r["report_path"]}
        def run_candidate(self, pulse): return {"status": "PASS", "report_path": report_c["report_path"]}
        def run_exact(self, pulse): return {"status": "PASS", "report_path": report_r["report_path"]}

    bad_runner = _production_runner_shell()
    bad = BadRunner()
    bad_runner.run_reference = bad.run_reference
    bad_runner.run_candidate = bad.run_candidate
    bad_runner.run_exact = bad.run_exact
    with pytest.raises(Exception, match="formal exact|schema|object"):
        _open_production_driver(bad_root, admission_identity=identity, runner=bad_runner, n_pulses=1,
                           campaign_id=identity["campaign_id"]).run()

    good_root = tmp_path / "paired"
    result = run_case(good_root)
    assert result["status"] == "PASS"
    assert result["validated_object_count"] == 12 * 8 + 9 + 1


def test_r07_crash_takeover_requires_hash_bound_stale_exit_receipt(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import PairedCampaign
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    identity = _formal_identity(campaign_id="takeover-r")
    root = tmp_path / "takeover"
    campaign = PairedCampaign.create(root, n_pulses=1, campaign_id=identity["campaign_id"],
                                     admission_identity=identity, formal=True,
                                     require_intents=True, require_quota=False)
    campaign.register_process_start(process_id="old-coordinator", source="test")
    reopened = PairedCampaign.open(root, n_pulses=1, campaign_id=identity["campaign_id"],
                                   admission_identity=identity, formal=True,
                                   require_intents=True, require_quota=False)
    with pytest.raises(StorageIntegrityError, match="live ACTIVE coordinator"):
        reopened.register_process_start(process_id="new-coordinator", source="takeover", takeover=True)
    with pytest.raises(StorageIntegrityError, match="external crash takeover receipts are forbidden"):
        reopened.register_process_start(
            process_id="new-coordinator", source="takeover", takeover=True,
            takeover_receipt={"path": str(root / "forged.json"), "sha256": "0" * 64},
        )
    campaign.register_process_exit(process_id="old-coordinator")


def test_r07_independent_process_crash_stale_takeover_and_resume(tmp_path):
    identity = _formal_identity(campaign_id="takeover-subprocess-r")
    root = tmp_path / "subprocess-takeover"
    identity_path = tmp_path / "admission.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    child_one = textwrap.dedent(
        """
        import json, os, sys
        from pathlib import Path
        from KHz_filament.hr4e5_paired_campaign import _open_production_driver
        from KHz_filament.hr4e5_evidence import atomic_json
        from KHz_filament.hr4e5_production import _E5AProductionRunner

        root = Path(sys.argv[1]).resolve()
        identity = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        runner = _E5AProductionRunner.__new__(_E5AProductionRunner)
        driver = _open_production_driver(root, admission_identity=identity, runner=runner,
                                    n_pulses=1, campaign_id=identity["campaign_id"])
        atomic_json(root / "child_one_started.json", {
            "process_id": driver.epoch, "campaign_id": identity["campaign_id"],
            "next_pair_index": driver.campaign.next_pair_index,
        })
        os._exit(17)
        """
    )
    first = subprocess.run(
        [sys.executable, "-s", "-B", "-c", child_one, str(root), str(identity_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert first.returncode == 17, first.stdout + "\n" + first.stderr
    state_path = root / "E5_1A_CAMPAIGN_STATE.json"
    state_before = json.loads(state_path.read_text(encoding="utf-8"))
    active = state_before.get("active_coordinator")
    assert isinstance(active, dict) and active.get("status") == "ACTIVE"
    assert state_before["admission_hash"] == identity["identity_sha256"]
    assert state_before["next_pair_index"] == 0
    assert state_before["pairs"][0]["pulse_index"] == 0
    assert state_before["pairs"][0].get("formal_steps") == {}

    child_two = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from KHz_filament.hr4e5_paired_campaign import _open_production_driver
        from KHz_filament.hr4e5_evidence import atomic_json
        from KHz_filament.hr4e5_production import _E5AProductionRunner

        root = Path(sys.argv[1]).resolve()
        identity = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        runner = _E5AProductionRunner.__new__(_E5AProductionRunner)
        driver = _open_production_driver(root, admission_identity=identity, runner=runner,
                                    n_pulses=1, campaign_id=identity["campaign_id"],
                                    takeover=True)
        new_epoch = driver.epoch
        report = driver.resume(run=False)
        assert report["admission_hash"] == identity["identity_sha256"]
        assert report["next_pair_index"] == 0
        assert report["pairs"][0]["pulse_index"] == 0
        assert report["pairs"][0].get("formal_steps") == {}
        driver.close()
        atomic_json(root / "child_two_resumed.json", {
            "process_id": new_epoch, "campaign_id": identity["campaign_id"],
            "admission_hash": report["admission_hash"],
            "next_pair_index": report["next_pair_index"],
            "pulse_index": report["pairs"][0]["pulse_index"],
        })
        """
    )
    second = subprocess.run(
        [sys.executable, "-s", "-B", "-c", child_two, str(root), str(identity_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert second.returncode == 0, second.stdout + "\n" + second.stderr
    resumed = json.loads((root / "child_two_resumed.json").read_text(encoding="utf-8"))
    state_after = json.loads(state_path.read_text(encoding="utf-8"))
    epochs = state_after["process_epochs"]
    assert len(epochs) == 2
    assert len({str(item["process_id"]) for item in epochs}) == 2
    assert epochs[0]["status"] == "EXITED_STALE"
    assert epochs[0]["exit_reason"] == "durable_crash_takeover_receipt"
    takeover_path = Path(epochs[0]["exit_receipt"]["path"])
    assert takeover_path == root / ".takeover" / f"epoch_{active['epoch']}.json"
    takeover = json.loads(takeover_path.read_text(encoding="utf-8"))
    assert takeover["registry_backed"] is True
    assert takeover["stale"] is True and takeover["exit_observed"] is True
    assert takeover["liveness_evidence"]["status"] == "DEAD"
    assert epochs[1]["status"] == "EXITED"
    assert resumed["process_id"] == epochs[1]["process_id"]
    assert resumed["process_id"] != epochs[0]["process_id"]
    assert state_after["active_coordinator"] is None
    assert state_after["admission_hash"] == identity["identity_sha256"]
    assert state_after["next_pair_index"] == 0
    assert state_after["pairs"][0]["pulse_index"] == 0
    assert state_after["pairs"][0].get("formal_steps") == {}


def test_r08_formal_old_epoch_cannot_record_or_commit(tmp_path):
    from KHz_filament.hr4e5_evidence import atomic_json, sha256_file
    from KHz_filament.hr4e5_paired_campaign import PairedCampaign
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    identity = _formal_identity(campaign_id="stale-epoch-r")
    root = tmp_path / "stale-epoch"
    campaign = PairedCampaign.create(
        root, n_pulses=1, campaign_id=identity["campaign_id"],
        admission_identity=identity, formal=True, require_intents=True, require_quota=False,
    )
    old = campaign.register_process_start(process_id="old-epoch", source="test")
    receipt = root / "step.json"
    atomic_json(receipt, {"status": "PASS", "campaign_id": identity["campaign_id"]})
    durable = {"path": str(receipt), "sha256": sha256_file(receipt)}
    campaign.formal_step(0, "reference", status="COMMITTED", receipt=durable, epoch=old["epoch"])
    campaign.register_process_exit(process_id="old-epoch")
    new = campaign.register_process_start(process_id="new-epoch", source="resume")
    with pytest.raises(StorageIntegrityError, match="current coordinator epoch"):
        campaign.formal_step(0, "candidate", status="IN_PROGRESS")
    with pytest.raises(StorageIntegrityError, match="stale coordinator epoch"):
        campaign.formal_step(0, "candidate", status="IN_PROGRESS", epoch=old["epoch"])
    with pytest.raises(StorageIntegrityError, match="stale coordinator epoch"):
        campaign.commit_formal_pair(0, epoch=old["epoch"])
    assert new["epoch"] != old["epoch"]
    campaign.register_process_exit(process_id="new-epoch")


def test_r08_exact_resume_after_gc_uses_metadata_only_pair_validator(tmp_path):
    from KHz_filament.hr4e5_evidence import sha256_file
    from KHz_filament.hr4e5_paired_campaign import _open_production_driver
    from KHz_filament.hr4e5_storage import MockQuotaProvider, StorageIntegrityError

    identity = _formal_identity(campaign_id="exact-after-gc-r", n_pulses=2)
    root = tmp_path / "exact-after-gc"
    driver = _open_production_driver(
        root, admission_identity=identity, runner=_production_runner_shell(), n_pulses=2,
        campaign_id=identity["campaign_id"], safety_margin_bytes=1024**2,
    )
    driver.campaign.storage.provider = MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40)
    report = _formal_pair_report(root, root / "paired_exact.json", identity, driver.campaign.storage)
    durable = {"path": report["report_path"], "sha256": sha256_file(report["report_path"])}
    epoch = driver.campaign.active_coordinator_epoch()
    driver.campaign.formal_step(0, "exact", status="COMMITTED", receipt=durable, epoch=epoch)
    artifacts = driver.campaign.storage.artifacts()
    targets = {
        side: [root / relative for relative in artifacts if relative.startswith(f"objects/{side}/")]
        for side in ("R", "C")
    }
    reference_gc = _complete_formal_gc(
        root, driver.campaign.storage, identity, side="R", pulse=0,
        targets=targets["R"], writer_epoch=epoch,
    )
    driver.campaign.formal_step(
        0, "reference_gc", status="COMMITTED", receipt=reference_gc, epoch=epoch,
    )
    with pytest.raises(StorageIntegrityError, match="trajectory"):
        driver.campaign.formal_step(
            0, "candidate_gc", status="COMMITTED", receipt=reference_gc, epoch=epoch,
        )
    candidate_gc = _complete_formal_gc(
        root, driver.campaign.storage, identity, side="C", pulse=0,
        targets=targets["C"], writer_epoch=epoch,
    )
    driver.campaign.formal_step(
        0, "candidate_gc", status="COMMITTED", receipt=candidate_gc, epoch=epoch,
    )
    resumed = driver._saved_receipt(0, "exact")
    assert resumed["status"] == "PASS"
    assert resumed["metadata_only"] is True
    assert all(row.get("metadata_only") is True for row in resumed["rows"])
    driver.close()


def test_r09_formal_successor_requires_ready_receipt(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import _open_production_driver
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    identity = _formal_identity(campaign_id="missing-ready-r", n_pulses=2)
    root = tmp_path / "missing-ready"
    root.mkdir()
    report = _write_formal_simple_report(root, root / "successor_report.json", identity, "R")
    parent = root / "parent"
    parent.mkdir(parents=True)
    child = root / "child"
    child.mkdir(parents=True)
    driver = _open_production_driver(
        root, admission_identity=identity, runner=_production_runner_shell(), n_pulses=2,
        campaign_id=identity["campaign_id"], safety_margin_bytes=1024**2,
    )
    with pytest.raises(StorageIntegrityError, match="READY"):
        driver._validate_receipt(
            {
                "status": "PASS",
                "report_path": report["report_path"],
                "child_root": str(child),
                "parent_root": str(parent),
                "parent_generation": "generation-0",
            },
            pulse=0, step="reference_successor",
        )
    driver.close()


def test_r09_formal_terminal_requires_inventory_contract(tmp_path):
    from KHz_filament.hr4e5_evidence import sha256_file
    from KHz_filament.hr4e5_paired_campaign import _open_production_driver
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    identity = _formal_identity(campaign_id="missing-terminal-inventory-r")
    root = tmp_path / "missing-terminal-inventory"
    writer_path = root / "writer.json"
    writer_path.parent.mkdir(parents=True)
    writer_path.write_text(json.dumps({
        "status": "PASS", "campaign_id": identity["campaign_id"],
        "active_writers": [], "writer_epoch": "writer", "coordinator_process_id": "runner",
    }), encoding="utf-8")
    report = _write_formal_simple_report(
        root, root / "terminal.json", identity, "C",
        writer={"writer_receipt_path": str(writer_path), "writer_receipt_sha256": sha256_file(writer_path)},
    )
    class Runner:
        def expected_terminal_roles(self, pulse): return {"final": []}
    runner = _production_runner_shell()
    runner.expected_terminal_roles = Runner().expected_terminal_roles
    driver = _open_production_driver(
        root, admission_identity=identity, runner=runner, n_pulses=1,
        campaign_id=identity["campaign_id"], safety_margin_bytes=1024**2,
    )
    value = {
        "status": "PASS", "report_path": report["report_path"],
        "writer_receipt_path": str(writer_path), "writer_receipt_sha256": sha256_file(writer_path),
    }
    with pytest.raises(StorageIntegrityError, match="registry-backed|writer receipt"):
        driver._validate_receipt(value, pulse=0, step="terminal")
    driver.close()


def _local_production_spec(tmp_path, *, campaign_id="local-production", n_pulses=1, k=8,
                             candidate_hydro_workers=1, max_campaign_live_bytes=300*1024**3,
                             final_output_budget_bytes=64*1024**3,
                             safety_margin_bytes=1024**2):
    from KHz_filament.hr4e5_evidence import sha256_file
    from KHz_filament.hr4e5_production import E5AProductionSpec, _source_hash

    inputs = tmp_path / "inputs"; inputs.mkdir(parents=True, exist_ok=True)
    config = inputs / "config.json"
    value = {
        "grid": {"Nx": 8, "Ny": 8, "Nt": 8, "Lx": 8e-4, "Ly": 8e-4, "Twin": 80e-15},
        "beam": {"w0": 1.5e-4, "tau_fwhm": 40e-15, "energy_J": 1e-10,
                 "focal_length": None},
        "propagation": {"z_max": k*1e-4, "dz": 1e-4, "linear_model": "paraxial",
            "auto_substep": False, "focus_window_step": False, "limit_focus_window": False,
            "progress_every_z": 0, "energy_probe_every": 0, "diag_extra": False,
            "use_electronic_kerr": False, "use_raman_phase": False,
            "use_raman_absorption": False, "use_plasma_phase": False,
            "use_ionization_loss": False, "use_ionization_solver": False},
        "ionization": {"species": []},
        "heat": {"hr3b_enabled": True, "f_rep": 5e6, "dt_hydro": 1e-7,
                 "chi": 2.17e-5, "D_th": 2.17e-5, "nu": 1.5e-5,
                 "gravity_x": 0.0, "gravity_y": -9.81},
        "run": {"Npulses": n_pulses},
        "raman": {"enabled": False, "absorption": False},
    }
    config.write_text(json.dumps(value), encoding="utf-8")
    source = inputs / "source.json"
    source.write_text(json.dumps({"schema": "local-source.v1",
                                  "canonical_array_hash": _source_hash(config)}), encoding="utf-8")
    lut = inputs / "lut.json"
    lut.write_text(json.dumps({"schema": "local-lut.v1", "mode": "DISABLED_BY_CONFIG",
                               "config_sha256": sha256_file(config)}), encoding="utf-8")
    pre0 = inputs / "pre0.npy"; np.save(pre0, np.zeros((k, 8, 8), dtype=np.float64))
    return E5AProductionSpec(root=tmp_path/"campaign", campaign_id=campaign_id,
        config_path=config, source_manifest_path=source, lut_manifest_path=lut,
        pre0_delta_n_path=pre0, n_pulses=n_pulses,
        candidate_hydro_workers=candidate_hydro_workers,
        max_campaign_live_bytes=max_campaign_live_bytes,
        final_output_budget_bytes=final_output_budget_bytes,
        safety_margin_bytes=safety_margin_bytes)


def test_production_factory_is_unique_and_n1_real_cpu_path(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import FormalPairedDriver, _open_production_driver
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    with pytest.raises(StorageIntegrityError, match="production factory"):
        FormalPairedDriver(tmp_path/"forbidden", admission_identity=_formal_identity(), runner=object(), n_pulses=1)
    with pytest.raises(StorageIntegrityError, match="fixed E5-1A production runner"):
        _open_production_driver(
            tmp_path/"forbidden-private", admission_identity=_formal_identity(),
            runner=object(), n_pulses=1,
        )
    campaign = open_e5_1a_production_campaign(_local_production_spec(tmp_path))
    result = campaign.run()
    assert result["status"] == "COMPLETE"
    assert result["admission_identity"]["execution_mode"] == "LOCAL_ORCHESTRATION_QUALIFICATION"


def test_production_reference_subprocess_has_distinct_pid_and_exact_output(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    spec = replace(_local_production_spec(tmp_path), reference_subprocess=True)
    result = open_e5_1a_production_campaign(spec).run()
    assert result["status"] == "COMPLETE"
    receipt_path = (tmp_path / "campaign_reference_process_audit" / "p0" / "epoch_0"
                    / "R_epoch_0.result.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["process_identity"]["pid"] != os.getpid()
    assert receipt["parent_pid"] == os.getpid()
    candidate_path = (tmp_path / "campaign_candidate_final_process_audit" / "p0" / "epoch_0"
                      / "C_OPT_epoch_0.result.json")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["trajectory"] == "C"
    assert candidate["process_identity"]["pid"] not in {os.getpid(), receipt["process_identity"]["pid"]}
    assert candidate["parent_pid"] == os.getpid()
    exact = json.loads((spec.root / "exact" / "p0.json").read_text(encoding="utf-8"))
    assert exact["status"] == "PASS" and exact["mismatch_count"] == 0


def test_production_n2_reference_and_terminal_candidate_use_independent_processes(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    spec = replace(_local_production_spec(
        tmp_path, n_pulses=2, k=8, candidate_hydro_workers=2),
        reference_subprocess=True,
    )
    result = open_e5_1a_production_campaign(spec).run()
    assert result["status"] == "COMPLETE"
    final_candidate = (tmp_path / "campaign_candidate_final_process_audit" / "p1" / "epoch_0"
                       / "C_OPT_epoch_0.result.json")
    payload = json.loads(final_candidate.read_text(encoding="utf-8"))
    assert payload["trajectory"] == "C" and payload["process_identity"]["pid"] != os.getpid()
    for pulse in range(2):
        exact = json.loads((spec.root / "exact" / f"p{pulse}.json").read_text(encoding="utf-8"))
        assert exact["status"] == "PASS" and exact["mismatch_count"] == 0


def test_production_admission_recomputes_source_lut_and_budget_identity(tmp_path):
    from dataclasses import replace
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    source_spec = _local_production_spec(tmp_path / "source", campaign_id="tamper-source")
    Path(source_spec.source_manifest_path).write_text(
        json.dumps({"schema": "local-source.v1", "canonical_array_hash": "bad"}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source manifest differs"):
        open_e5_1a_production_campaign(source_spec)

    lut_spec = _local_production_spec(tmp_path / "lut", campaign_id="tamper-lut")
    Path(lut_spec.lut_manifest_path).write_text(
        json.dumps({"schema": "local-lut.v1", "mode": "DISABLED_BY_CONFIG",
                    "config_sha256": "bad"}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disabled LUT manifest is not bound"):
        open_e5_1a_production_campaign(lut_spec)

    budget_spec = _local_production_spec(tmp_path / "budget", campaign_id="tamper-budget")
    campaign = open_e5_1a_production_campaign(budget_spec)
    campaign.close()
    with pytest.raises((ValueError, RuntimeError), match="identity|safety margin|conflicts"):
        open_e5_1a_production_campaign(
            replace(budget_spec, safety_margin_bytes=2 * 1024**2)
        )


def test_production_n2_k16_multiblock_overlap_happy_path(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import validate_overlap_events
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    campaign = open_e5_1a_production_campaign(
        _local_production_spec(tmp_path, campaign_id="local-production-k16", n_pulses=2, k=16)
    )
    result = campaign.run()
    assert result["status"] == "COMPLETE"
    assert result["next_pair_index"] == 2
    exact0 = json.loads((tmp_path / "campaign" / "exact" / "p0.json").read_text(encoding="utf-8"))
    exact1 = json.loads((tmp_path / "campaign" / "exact" / "p1.json").read_text(encoding="utf-8"))
    assert exact0["screen_count"] == 15 * 16
    assert exact1["screen_count"] == 12 * 16
    state0 = json.loads((tmp_path / "campaign" / "C" / "p0" / "state" / "streaming_manifest.json").read_text(encoding="utf-8"))
    assert state0["barrier"]["status"] == "PASS"
    assert state0["promotion"]["authoritative_namespace"] == "NEXT"
    assert len(state0["records"]) == 16
    events = state0["telemetry_events"]
    overlap = validate_overlap_events(events)
    assert overlap["status"] == "PASS"
    first_hydro_index, first_hydro = next(
        (index, event) for index, event in enumerate(events)
        if event["event"] == "HYDRO_BLOCK_START"
    )
    optical_complete_index, optical_complete = next(
        (index, event) for index, event in enumerate(events)
        if event["event"] == "OPTICAL_COMPLETE"
    )
    assert first_hydro["block"] == list(range(8))
    assert first_hydro_index < optical_complete_index
    assert first_hydro["monotonic_s"] < optical_complete["monotonic_s"]
    ledger = json.loads((tmp_path / "campaign" / "storage_ledger.json").read_text(encoding="utf-8"))
    for side in ("R", "C"):
        relative = f"{side}/p0/state/E5_1A_ARCHIVED_AFTER_EXACT.json"
        assert relative in ledger["artifacts"]
        assert ledger["artifacts"][relative]["role"] == "SUCCESSOR"


def test_production_n2_k64_four_hydro_workers_matches_single_worker(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    multi_spec = _local_production_spec(
        tmp_path / "multi", campaign_id="local-production-k64-4-workers",
        n_pulses=2, k=64, candidate_hydro_workers=4,
    )
    too_small_spec = _local_production_spec(
        tmp_path / "too-small", campaign_id="local-production-too-small",
        n_pulses=2, k=64, candidate_hydro_workers=4,
        max_campaign_live_bytes=2*1024**3,
        final_output_budget_bytes=1024*1024,
        safety_margin_bytes=1024*1024,
    )
    with pytest.raises(RuntimeError, match="exceeds cap"):
        open_e5_1a_production_campaign(too_small_spec)
    multi_result = open_e5_1a_production_campaign(multi_spec).run()
    assert multi_result["status"] == "COMPLETE" and multi_result["next_pair_index"] == 2

    single_spec = _local_production_spec(
        tmp_path / "single", campaign_id="local-production-k64-single-worker",
        n_pulses=2, k=64, candidate_hydro_workers=1,
    )
    single_result = open_e5_1a_production_campaign(single_spec).run()
    assert single_result["status"] == "COMPLETE" and single_result["next_pair_index"] == 2

    def exact_signature(root):
        reports = [json.loads((Path(root) / "exact" / f"p{pulse}.json").read_text(encoding="utf-8"))
                   for pulse in range(2)]
        return [row["comparison_key"] for report in reports for row in report["rows"]], [
            (row["comparison_key"], row["reference_sha256_array"], row["candidate_sha256_array"])
            for report in reports for row in report["rows"]
        ]

    multi_keys, multi_values = exact_signature(multi_spec.root)
    single_keys, single_values = exact_signature(single_spec.root)
    assert multi_keys == single_keys and multi_values == single_values

    manifest = json.loads((Path(multi_spec.root) / "C" / "p0" / "state" /
                           "streaming_manifest.json").read_text(encoding="utf-8"))
    hydro_receipts = [json.loads((Path(multi_spec.root) / "C" / "p0" / "state" /
                                  "process_receipts" / f"C_HYDRO_{index:03d}.json").read_text(encoding="utf-8"))
                      for index in range(4)]
    assert {receipt["role"] for receipt in hydro_receipts} == {"C_HYDRO"}
    assert len({receipt["pid"] for receipt in hydro_receipts}) == 4
    assert {receipt["worker_index"] for receipt in hydro_receipts} == set(range(4))
    assert {receipt["exit_status"] for receipt in hydro_receipts} == {"PASS"}
    claimed = [ordinal for receipt in hydro_receipts for ordinal in receipt["claimed_ordinals"]]
    assert sorted(claimed) == list(range(64)) and len(claimed) == len(set(claimed))
    actors = {str(event.get("actor")) for event in manifest["telemetry_events"]
              if str(event.get("event")) == "HYDRO_CLAIM"}
    expected = {f"e5_1b_c_hydro_0_{index}" for index in range(4)}
    assert actors <= expected
    assert manifest["barrier"]["status"] == "PASS"
    assert manifest["promotion"]["authoritative_namespace"] == "NEXT"
    admission = json.loads((Path(multi_spec.root) / "R" / "p0" / "reports" /
                            "admission.json").read_text(encoding="utf-8"))
    assert admission["admission_identity_sha256"] == multi_result["admission_identity"]["identity_sha256"]
    topology = multi_result["admission_identity"]["hydro_topology"]
    assert topology["candidate_hydro_workers_requested"] == 4
    assert topology["candidate_hydro_processes_requested"] == 4
    assert topology["candidate_hydro_gpus_requested"] == 4
    assert topology["candidate_hydro_local_process_qualification"] in {"PENDING", "PASS"}
    assert topology["candidate_hydro_physical_gpu_binding_qualification"] == "PENDING_HPC"
    assert topology["gpu_topology_qualified"] is False



def test_production_n2_k64_h2_fallback_multiprocess(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    spec = _local_production_spec(
        tmp_path, campaign_id="local-production-k64-h2-fallback", n_pulses=2, k=64,
        candidate_hydro_workers=2,
    )
    result = open_e5_1a_production_campaign(spec).run()
    assert result["status"] == "COMPLETE" and result["next_pair_index"] == 2
    receipts = [json.loads((Path(spec.root) / "C" / "p0" / "state" / "process_receipts" /
                            f"C_HYDRO_{index:03d}.json").read_text(encoding="utf-8"))
                for index in range(2)]
    assert len({receipt["pid"] for receipt in receipts}) == 2
    assert {receipt["exit_status"] for receipt in receipts} == {"PASS"}
    assert all(receipt["claimed_ordinals"] for receipt in receipts)
    claimed = [ordinal for receipt in receipts for ordinal in receipt["claimed_ordinals"]]
    assert sorted(claimed) == list(range(64)) and len(claimed) == len(set(claimed))
    for pulse in range(2):
        exact = json.loads((Path(spec.root) / "exact" / f"p{pulse}.json").read_text(encoding="utf-8"))
        assert exact["status"] == "PASS" and exact["mismatch_count"] == 0
    manifest = json.loads((Path(spec.root) / "C" / "p0" / "state" /
                           "streaming_manifest.json").read_text(encoding="utf-8"))
    assert manifest["barrier"]["status"] == "PASS"
    assert manifest["promotion"]["authoritative_namespace"] == "NEXT"
    assert not manifest["queue"] and not manifest["recovery_backlog"]
    ledger = json.loads((Path(spec.root) / "storage_ledger.json").read_text(encoding="utf-8"))
    assert not [item for item in ledger["writers"].values() if item["status"] == "ACTIVE"]


@pytest.mark.parametrize("k,extra_takeover", [(16, False), (16, True), (64, False)],
                         ids=["k16", "k16-repeated", "k64"])
def test_production_n2_k64_h4_dead_hydro_worker_fresh_coordinator_recovery(tmp_path, k, extra_takeover):
    from KHz_filament.hr4e5_storage import probe_process_identity
    from KHz_filament.hr4e5_evidence import sha256_file

    spec = _local_production_spec(
        tmp_path, campaign_id="local-production-k64-h4-dead-worker-recovery",
        n_pulses=2, k=k, candidate_hydro_workers=4,
    )
    attempts = tmp_path / "attempt_evidence"
    first_attempt = attempts / "initial"
    replacement_attempt = attempts / "replacement"
    first_attempt.mkdir(parents=True)
    replacement_attempt.mkdir(parents=True)
    claim_marker = first_attempt / "paused_hydro_claim.json"
    replacement_result_path = replacement_attempt / "result.json"
    python_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(python_root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONNOUSERSITE"] = "1"

    first_coordinator = textwrap.dedent(
        """
        import sys, textwrap
        from pathlib import Path
        from KHz_filament.hr4e5_production import (
            E5AProductionSpec, _E5AProductionRunner, open_e5_1a_production_campaign,
        )

        root, campaign_id, config, source, lut, pre0, marker = sys.argv[1:]
        spec = E5AProductionSpec(
            root=Path(root), campaign_id=campaign_id, config_path=Path(config),
            source_manifest_path=Path(source), lut_manifest_path=Path(lut),
            pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=4,
            safety_margin_bytes=1024**2,
        )
        worker_wrapper = textwrap.dedent(r'''
            import json, os, sys, time
            from pathlib import Path
            from KHz_filament.hr4e5_production_worker import main
            from KHz_filament.hr4e5_storage import process_identity
            from KHz_filament.hr4e5s_streaming import StreamingLifecycle

            marker = Path(sys.argv[1])
            worker_args = sys.argv[2:]
            worker_index = int(worker_args[worker_args.index("--worker-index") + 1])
            sys.argv = [sys.argv[0], *worker_args]
            original_claim = StreamingLifecycle.claim_block

            def pause_after_durable_claim(self, *, actor="hydro"):
                block = original_claim(self, actor=actor)
                if block and not marker.exists():
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        owner_fd = os.open(str(marker) + ".owner", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                    except FileExistsError:
                        owner_fd = None
                    if owner_fd is not None:
                        os.close(owner_fd)
                        temporary_marker = marker.with_name(marker.name + f".{os.getpid()}.tmp")
                        temporary_marker.write_text(json.dumps({
                            "process_identity": process_identity(), "worker_index": worker_index,
                            "actor": actor, "block": block,
                            "lifecycle_root": str(self.root),
                            "writer_id": worker_args[worker_args.index("--writer-id") + 1],
                        }), encoding="utf-8")
                        temporary_marker.replace(marker)
                        deadline = time.monotonic() + 120.0
                        while time.monotonic() < deadline:
                            time.sleep(0.05)
                return block

            StreamingLifecycle.claim_block = pause_after_durable_claim
            raise SystemExit(main())
        ''')
        original_command = _E5AProductionRunner._hydro_worker_command
        def paused_worker_command(self, state, pulse, worker_index, **kwargs):
            command = original_command(self, state, pulse, worker_index, **kwargs)
            return [*command[:3], "-c", worker_wrapper, marker, *command[5:]]
        _E5AProductionRunner._hydro_worker_command = paused_worker_command
        open_e5_1a_production_campaign(spec).run()
        """
    )
    first_stdout = first_attempt / "coordinator.stdout.txt"
    first_stderr = first_attempt / "coordinator.stderr.txt"
    with first_stdout.open("w", encoding="utf-8") as stdout_file, first_stderr.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        coordinator = subprocess.Popen(
            [sys.executable, "-s", "-B", "-c", first_coordinator,
             str(spec.root), spec.campaign_id, str(spec.config_path),
             str(spec.source_manifest_path), str(spec.lut_manifest_path),
             str(spec.pre0_delta_n_path), str(claim_marker)],
            cwd=python_root, env=environment, stdout=stdout_file, stderr=stderr_file, text=True,
        )
        try:
            claim_deadline = time.monotonic() + 90.0
            while not claim_marker.is_file():
                if coordinator.poll() is not None:
                    pytest.fail(
                        "initial coordinator exited before the hydro claim pause; "
                        f"exit={coordinator.returncode}, stdout={first_stdout.read_text(encoding='utf-8')}, "
                        f"stderr={first_stderr.read_text(encoding='utf-8')}"
                    )
                if time.monotonic() >= claim_deadline:
                    pytest.fail("no hydro child reached its durable claim pause within 90 seconds")
                time.sleep(0.05)
            marker = json.loads(claim_marker.read_text(encoding="utf-8"))
            worker_identity = marker["process_identity"]
            assert coordinator.pid != os.getpid() and coordinator.poll() is None
            assert marker["worker_index"] in range(4)
            assert marker["actor"].endswith(f"_{marker['worker_index']}")
            assert int(worker_identity["pid"]) != coordinator.pid
            assert probe_process_identity(worker_identity)["status"] == "LIVE"

            lifecycle_root = Path(marker["lifecycle_root"])
            manifest = json.loads((lifecycle_root / "streaming_manifest.json").read_text(encoding="utf-8"))
            claimed = [int(value) for value in marker["block"]]
            claim_events = [event for event in manifest["telemetry_events"]
                            if event.get("event") == "HYDRO_CLAIM" and event.get("actor") == marker["actor"]]
            assert claim_events and claim_events[-1]["block"] == claimed
            for ordinal in claimed:
                record = manifest["records"][ordinal]
                assert record["state"] == "HYDRO_RUNNING" and record["next"] is None
                assert not (lifecycle_root / "next" / f"screen_{ordinal:06d}.npz").exists()
            (first_attempt / "claim_observation.json").write_text(json.dumps({
                "coordinator_pid": coordinator.pid, "worker_process_identity": worker_identity,
                "worker_liveness": "LIVE", "worker_index": marker["worker_index"],
                "actor": marker["actor"], "claimed_ordinals": claimed,
                "all_claimed_uncommitted": True,
            }, indent=2), encoding="utf-8")

            # The worker identity was probed LIVE immediately before this exact PID is killed.
            os.kill(int(worker_identity["pid"]), signal.SIGTERM)
            worker_deadline = time.monotonic() + 15.0
            worker_probe = probe_process_identity(worker_identity)
            while worker_probe["status"] == "LIVE" and time.monotonic() < worker_deadline:
                time.sleep(0.05)
                worker_probe = probe_process_identity(worker_identity)
            assert worker_probe["status"] == "DEAD", worker_probe
            coordinator.wait(timeout=600)
            (first_attempt / "exit_observation.json").write_text(json.dumps({
                "worker_liveness_after_kill": worker_probe,
                "coordinator_returncode": coordinator.returncode,
            }, indent=2), encoding="utf-8")
            assert coordinator.returncode != 0, first_stdout.read_text(encoding="utf-8")
        except BaseException:
            if claim_marker.is_file():
                failed_marker = json.loads(claim_marker.read_text(encoding="utf-8"))
                failed_worker = failed_marker["process_identity"]
                if probe_process_identity(failed_worker)["status"] == "LIVE":
                    os.kill(int(failed_worker["pid"]), signal.SIGTERM)
            if coordinator.poll() is None:
                try:
                    coordinator.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    coordinator.terminate()
                    try:
                        coordinator.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        coordinator.kill()
                        coordinator.wait(timeout=10)
            raise

    active_locator = Path(spec.root) / "C" / "p0" / "state" / "ACTIVE_OUTPUT.json"
    original_locator_sha256 = sha256_file(active_locator)
    if extra_takeover:
        intermediate = textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
            root, campaign_id, config, source, lut, pre0 = sys.argv[1:]
            spec = E5AProductionSpec(
                root=Path(root), campaign_id=campaign_id, config_path=Path(config),
                source_manifest_path=Path(source), lut_manifest_path=Path(lut),
                pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=4,
                safety_margin_bytes=1024**2,
            )
            open_e5_1a_production_campaign(spec).close()
            """
        )
        interrupted = subprocess.run(
            [sys.executable, "-s", "-B", "-c", intermediate,
             str(spec.root), spec.campaign_id, str(spec.config_path),
             str(spec.source_manifest_path), str(spec.lut_manifest_path),
             str(spec.pre0_delta_n_path)],
            cwd=python_root, env=environment, capture_output=True, text=True, timeout=120,
        )
        assert interrupted.returncode == 0, interrupted.stderr

    replacement_coordinator = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from KHz_filament.hr4e5_evidence import atomic_json
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign

        root, campaign_id, config, source, lut, pre0, result_path = sys.argv[1:]
        spec = E5AProductionSpec(
            root=Path(root), campaign_id=campaign_id, config_path=Path(config),
            source_manifest_path=Path(source), lut_manifest_path=Path(lut),
            pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=4,
            safety_margin_bytes=1024**2,
        )
        result = open_e5_1a_production_campaign(spec).run()
        atomic_json(result_path, result)
        """
    )
    replacement_stdout = replacement_attempt / "coordinator.stdout.txt"
    replacement_stderr = replacement_attempt / "coordinator.stderr.txt"
    with replacement_stdout.open("w", encoding="utf-8") as stdout_file, replacement_stderr.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        replacement = subprocess.run(
            [sys.executable, "-s", "-B", "-c", replacement_coordinator,
             str(spec.root), spec.campaign_id, str(spec.config_path),
             str(spec.source_manifest_path), str(spec.lut_manifest_path),
             str(spec.pre0_delta_n_path), str(replacement_result_path)],
            cwd=python_root, env=environment, stdout=stdout_file, stderr=stderr_file,
            text=True, timeout=1800 if k == 64 else 900,
        )
    assert replacement.returncode == 0, (
        replacement_stdout.read_text(encoding="utf-8") + "\n" +
        replacement_stderr.read_text(encoding="utf-8")
    )
    result = json.loads(replacement_result_path.read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETE" and result["next_pair_index"] == 2
    assert result["terminal_inventory_after_close"]["status"] == "PASS"
    assert sha256_file(active_locator) == original_locator_sha256

    root = Path(spec.root)
    for pulse in range(2):
        exact = json.loads((root / "exact" / f"p{pulse}.json").read_text(encoding="utf-8"))
        assert exact["status"] == "PASS" and exact["mismatch_count"] == 0
        assert all(
            row["status"] == "PASS"
            and row["reference_sha256_array"] == row["candidate_sha256_array"]
            for row in exact["rows"]
        )
    receipt_paths = sorted((root / "C" / "p0" / "state" / "process_receipts").glob("C_HYDRO_*.json"))
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths]
    assert {receipt["exit_status"] for receipt in receipts} == {"PASS"}
    assert len({receipt["pid"] for receipt in receipts}) >= 4
    covered = [int(ordinal) for receipt in receipts for ordinal in receipt["claimed_ordinals"]]
    assert sorted(covered) == list(range(k)) and len(covered) == len(set(covered))

    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    assert sum(item["purpose"] == "OPTICAL:C:p0" for item in ledger["reservations"].values()) == 1
    stale_writer = ledger["child_writer_grants"][marker["writer_id"]]
    assert stale_writer["state"] == "INTERRUPTED_STALE"
    assert stale_writer["liveness_evidence"]["status"] == "DEAD"
    assert not [item for item in ledger["writers"].values() if item["status"] == "ACTIVE"]
    assert not [item for item in ledger["child_writer_grants"].values()
                if item["state"] in {"PENDING", "BOUND_ACTIVE"}]
    lineage = json.loads((root / "pair_lineage" / "p0.json").read_text(encoding="utf-8"))
    assert lineage["cross_epoch"] is True
    assert lineage["trajectories"]["R"]["trajectory_commit_epoch"] != lineage["trajectories"]["C"]["trajectory_commit_epoch"]
    assert lineage["trajectories"]["C"]["trajectory_commit_epoch"] == ("2" if extra_takeover else "1")
    manifest = json.loads((root / "C" / "p0" / "state" /
                           "streaming_manifest.json").read_text(encoding="utf-8"))
    assert manifest["barrier"]["status"] == "PASS"
    assert manifest["promotion"]["authoritative_namespace"] == "NEXT"
    assert not manifest["queue"] and not manifest["recovery_backlog"]


@pytest.mark.parametrize("interrupt_hydro", [False, True], ids=["c0-r1", "c0-r2-hydro-restart"])
def test_production_reference_intents_rebind_and_completed_output_replays_across_epochs(tmp_path, interrupt_hydro):
    from KHz_filament.hr4e5_evidence import sha256_file

    spec = _local_production_spec(
        tmp_path, campaign_id="local-production-reference-recovery", n_pulses=2, k=8,
        candidate_hydro_workers=2,
    )
    python_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(python_root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONNOUSERSITE"] = "1"

    def run_coordinator(source, *arguments):
        completed = subprocess.run(
            [sys.executable, "-s", "-B", "-c", textwrap.dedent(source), *map(str, arguments)],
            cwd=python_root, env=environment, capture_output=True, text=True, timeout=300,
        )
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    first = """
        import sys
        from pathlib import Path
        import KHz_filament.hr4e5_production as production

        root, campaign_id, config, source, lut, pre0 = sys.argv[1:]
        spec = production.E5AProductionSpec(
            root=Path(root), campaign_id=campaign_id, config_path=Path(config),
            source_manifest_path=Path(source), lut_manifest_path=Path(lut),
            pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=2,
            safety_margin_bytes=1024**2,
        )
        def interrupt_reference_optical(**kwargs):
            if kwargs.get("trajectory") == "R" and int(kwargs.get("pulse", -1)) == 0:
                failed_output = Path(kwargs["output_dir"])
                failed_output.mkdir(parents=True, exist_ok=False)
                (failed_output / "failed_attempt.txt").write_text("retained failed attempt", encoding="utf-8")
                kwargs["storage_budget"].interrupt_intent(
                    kwargs["creation_intent"]["intent_id"], reason="injected reference optical interruption")
                raise RuntimeError("injected reference optical interruption")
            return original_optical(**kwargs)
        original_optical = production.run_streaming_optical_pulse
        production.run_streaming_optical_pulse = interrupt_reference_optical
        try:
            production.open_e5_1a_production_campaign(spec).run()
        except RuntimeError as error:
            if "injected reference optical interruption" not in str(error):
                raise
        else:
            raise AssertionError("reference optical interruption did not occur")
    """
    run_coordinator(first, spec.root, spec.campaign_id, spec.config_path,
                    spec.source_manifest_path, spec.lut_manifest_path, spec.pre0_delta_n_path)

    root = Path(spec.root)
    state_path = root / "E5_1A_CAMPAIGN_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = state["pairs"][0]["formal_steps"]
    assert steps["candidate"]["status"] == "COMMITTED"
    assert steps["candidate"]["epoch"] == 0
    assert steps["reference"]["status"] == "INTERRUPTED"
    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    reference_optical = [item for item in ledger["intents"].values()
                         if item["trajectory"] == "R" and item["pulse"] == 0
                         and item["role"] == "OPTICAL"]
    assert len(reference_optical) == 1
    optical_intent_id = reference_optical[0]["intent_id"]
    optical_reservation_id = reference_optical[0]["reservation_id"]
    assert reference_optical[0]["status"] == "INTERRUPTED"
    assert ledger["reservations"][optical_reservation_id]["status"] == "ACTIVE"
    failed_attempt = root / "R" / "p0" / "optical" / "failed_attempt.txt"
    failed_attempt_sha256 = sha256_file(failed_attempt)

    if not interrupt_hydro:
        result_path = tmp_path / "two_epoch_result.json"
        second = """
            import sys
            from pathlib import Path
            from KHz_filament.hr4e5_evidence import atomic_json
            from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign

            root, campaign_id, config, source, lut, pre0, result_path = sys.argv[1:]
            spec = E5AProductionSpec(
                root=Path(root), campaign_id=campaign_id, config_path=Path(config),
                source_manifest_path=Path(source), lut_manifest_path=Path(lut),
                pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=2,
                safety_margin_bytes=1024**2,
            )
            atomic_json(result_path, open_e5_1a_production_campaign(spec).run())
        """
        run_coordinator(second, spec.root, spec.campaign_id, spec.config_path,
                        spec.source_manifest_path, spec.lut_manifest_path, spec.pre0_delta_n_path,
                        result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["status"] == "COMPLETE"
        assert result["terminal_inventory_after_close"]["status"] == "PASS"
        assert sha256_file(failed_attempt) == failed_attempt_sha256
        ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
        rebound = ledger["intents"][optical_intent_id]
        assert rebound["status"] == "COMPLETED"
        assert rebound["reservation_id"] == optical_reservation_id
        assert str(rebound["takeover_from_epoch"]) == "0"
        assert sum(item["purpose"] == "OPTICAL:R:p0" for item in ledger["reservations"].values()) == 1
        locator = json.loads((root / "R" / "p0" / "state" / "ACTIVE_OUTPUT.json").read_text(encoding="utf-8"))
        active_record = json.loads((root / locator["record_path"]).read_text(encoding="utf-8"))
        assert Path(active_record["output_dir"]).parts[-2] == "optical"
        assert Path(active_record["output_dir"]).name.startswith("replay_")
        lineage = json.loads((root / "pair_lineage" / "p0.json").read_text(encoding="utf-8"))
        assert lineage["cross_epoch"] is True
        assert lineage["trajectories"]["C"]["trajectory_commit_epoch"] == "0"
        assert lineage["trajectories"]["R"]["trajectory_commit_epoch"] == "1"
        exact = json.loads((root / "exact" / "p0.json").read_text(encoding="utf-8"))
        assert exact["status"] == "PASS" and exact["mismatch_count"] == 0
        return

    second = """
        import sys
        from pathlib import Path
        import KHz_filament.hr4e5_production as production

        root, campaign_id, config, source, lut, pre0 = sys.argv[1:]
        spec = production.E5AProductionSpec(
            root=Path(root), campaign_id=campaign_id, config_path=Path(config),
            source_manifest_path=Path(source), lut_manifest_path=Path(lut),
            pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=2,
            safety_margin_bytes=1024**2,
        )
        def interrupt_after_reference_hydro_store(store, **kwargs):
            if not Path(store.manifest_path).is_file():
                raise AssertionError("injected boundary did not follow HR-4C store creation")
            raise RuntimeError("injected reference hydro interruption after store creation")
        production.evolve_hr4_full_z = interrupt_after_reference_hydro_store
        try:
            production.open_e5_1a_production_campaign(spec).run()
        except RuntimeError as error:
            if "injected reference hydro interruption after store creation" not in str(error):
                raise
        else:
            raise AssertionError("reference hydro interruption did not occur")
    """
    run_coordinator(second, spec.root, spec.campaign_id, spec.config_path,
                    spec.source_manifest_path, spec.lut_manifest_path, spec.pre0_delta_n_path)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = state["pairs"][0]["formal_steps"]
    assert steps["candidate"]["status"] == "COMMITTED" and steps["candidate"]["epoch"] == 0
    assert steps["reference"]["status"] == "INTERRUPTED" and steps["reference"]["epoch"] == 1
    pair_id = state["pairs"][0]["scientific_pair_identity"]["scientific_pair_id"]
    implementation = json.loads((root / "E5_1B_IMPLEMENTATION_IDENTITY.json").read_text(encoding="utf-8"))
    implementation_sha256 = implementation["identity_sha256"]
    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    rebound_optical = ledger["intents"][optical_intent_id]
    assert rebound_optical["status"] == "COMPLETED"
    assert rebound_optical["reservation_id"] == optical_reservation_id
    assert str(rebound_optical["takeover_from_epoch"]) == "0"
    assert rebound_optical["metadata"]["scientific_pair_id"] == pair_id
    assert rebound_optical["metadata"]["implementation_identity"]["identity_sha256"] == implementation_sha256
    assert sum(item["purpose"] == "OPTICAL:R:p0" for item in ledger["reservations"].values()) == 1
    reference_hydro = [item for item in ledger["intents"].values()
                       if item["trajectory"] == "R" and item["pulse"] == 0
                       and item["role"] == "REFERENCE_HYDRO"]
    assert len(reference_hydro) == 1 and reference_hydro[0]["status"] == "INTERRUPTED"
    hydro_intent_id = reference_hydro[0]["intent_id"]
    hydro_reservation_id = reference_hydro[0]["reservation_id"]
    assert ledger["reservations"][hydro_reservation_id]["status"] == "ACTIVE"
    assert reference_hydro[0]["metadata"]["scientific_pair_id"] == pair_id
    assert reference_hydro[0]["metadata"]["implementation_identity"]["identity_sha256"] == implementation_sha256

    batch_dir = root / "R" / "p0" / "batch"
    old_hr4c_files = sorted(batch_dir.glob("store.hr4c_*"))
    assert old_hr4c_files
    old_hr4c_hashes = {path.name: sha256_file(path) for path in old_hr4c_files}
    incomplete_lifecycle = json.loads((root / "R" / "p0" / "state" / "streaming_manifest.json").read_text(encoding="utf-8"))
    assert incomplete_lifecycle.get("promotion") is None
    assert all(row.get("next") is None for row in incomplete_lifecycle["records"])

    old_output = root / "R" / "p0" / "optical"
    old_output_hashes = {
        path.relative_to(old_output).as_posix(): sha256_file(path)
        for path in old_output.rglob("*") if path.is_file()
    }
    assert old_output_hashes
    active_locator = root / "R" / "p0" / "state" / "ACTIVE_OUTPUT.json"
    old_locator_sha256 = sha256_file(active_locator)
    replay_evidence = tmp_path / "reference_replay_evidence.json"
    result_path = tmp_path / "recovered_result.json"
    third = """
        import json, sys
        from pathlib import Path
        from KHz_filament.hr4e5_evidence import atomic_json, sha256_file
        from KHz_filament.hr4e5_recovery_output import validate_active_output
        import KHz_filament.hr4e5_production as production

        (root, campaign_id, config, source, lut, pre0, old_output, old_hashes_json, old_hr4c_hashes_json,
         old_locator_sha256, evidence_path, result_path) = sys.argv[1:]
        root, old_output = Path(root), Path(old_output)
        old_hashes = json.loads(old_hashes_json)
        old_hr4c_hashes = json.loads(old_hr4c_hashes_json)
        spec = production.E5AProductionSpec(
            root=root, campaign_id=campaign_id, config_path=Path(config),
            source_manifest_path=Path(source), lut_manifest_path=Path(lut),
            pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=2,
            safety_margin_bytes=1024**2,
        )
        original_hydro = production._E5AProductionRunner._run_reference_hydro
        def verify_replayed_reference_output(self, *, state, pulse, lifecycle, grid):
            result = original_hydro(self, state=state, pulse=pulse, lifecycle=lifecycle, grid=grid)
            if int(pulse) == 0:
                identity = self.driver.campaign._pair(pulse)["scientific_pair_identity"]
                active = validate_active_output(
                    campaign_root=self.root, state_root=state,
                    scientific_pair_id=identity["scientific_pair_id"],
                    implementation_identity_sha256=self.implementation_identity["identity_sha256"],
                )
                active_output = Path(active["output_dir"]).resolve()
                assert active_output.is_relative_to(old_output.resolve())
                assert sha256_file(state / "ACTIVE_OUTPUT.json") == old_locator_sha256
                for relative, expected in old_hashes.items():
                    assert sha256_file(old_output / relative) == expected
                batch_dir = self._track("R", pulse) / "batch"
                for name, expected in old_hr4c_hashes.items():
                    assert sha256_file(batch_dir / name) == expected
                replay_manifests = sorted(batch_dir.glob("store_replay_*.hr4c_state_manifest.json"))
                assert replay_manifests
                atomic_json(evidence_path, {
                    "status": "PASS", "old_output_dir": str(old_output.resolve()),
                    "active_output_dir": str(active_output),
                    "old_output_files_preserved": len(old_hashes),
                    "old_hr4c_files_preserved": len(old_hr4c_hashes),
                    "replay_hr4c_store_count": len(replay_manifests),
                    "active_locator_sha256": sha256_file(state / "ACTIVE_OUTPUT.json"),
                })
            return result
        production._E5AProductionRunner._run_reference_hydro = verify_replayed_reference_output
        result = production.open_e5_1a_production_campaign(spec).run()
        atomic_json(result_path, result)
    """
    run_coordinator(third, spec.root, spec.campaign_id, spec.config_path,
                    spec.source_manifest_path, spec.lut_manifest_path, spec.pre0_delta_n_path,
                    old_output, json.dumps(old_output_hashes, sort_keys=True),
                    json.dumps(old_hr4c_hashes, sort_keys=True), old_locator_sha256,
                    replay_evidence, result_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETE" and result["next_pair_index"] == 2
    assert result["terminal_inventory_after_close"]["status"] == "PASS"
    replay = json.loads(replay_evidence.read_text(encoding="utf-8"))
    assert replay["status"] == "PASS" and replay["old_output_files_preserved"] == len(old_output_hashes)
    assert replay["old_hr4c_files_preserved"] == len(old_hr4c_hashes)
    assert replay["replay_hr4c_store_count"] >= 1
    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    assert ledger["intents"][hydro_intent_id]["status"] == "COMPLETED"
    assert ledger["intents"][hydro_intent_id]["reservation_id"] == hydro_reservation_id
    assert str(ledger["intents"][hydro_intent_id]["takeover_from_epoch"]) == "1"
    assert sum(item["purpose"] == "REFERENCE_HYDRO:R:p0" for item in ledger["reservations"].values()) == 1
    assert sum(item["purpose"] == "OPTICAL:R:p0" for item in ledger["reservations"].values()) == 1
    pair_steps = json.loads(state_path.read_text(encoding="utf-8"))["pairs"][0]["formal_steps"]
    assert pair_steps["candidate"]["epoch"] == 0 and pair_steps["reference"]["epoch"] == 2
    lineage = json.loads((root / "pair_lineage" / "p0.json").read_text(encoding="utf-8"))
    assert lineage["cross_epoch"] is True
    assert lineage["trajectories"]["C"]["trajectory_commit_epoch"] == "0"
    assert lineage["trajectories"]["R"]["trajectory_commit_epoch"] == "2"
    exact = json.loads((root / "exact" / "p0.json").read_text(encoding="utf-8"))
    assert exact["status"] == "PASS" and exact["mismatch_count"] == 0


@pytest.mark.parametrize("failure_point", ["before_complete", "before_consume", "after_rebind", "f08"])
def test_production_reference_hydro_promotion_recovers_original_intent_and_reservation(
    tmp_path, failure_point,
):
    from KHz_filament.hr4e5_evidence import sha256_file

    spec = _local_production_spec(
        tmp_path, campaign_id=f"local-production-reference-promotion-{failure_point}",
        n_pulses=2, k=8, candidate_hydro_workers=2,
    )
    python_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(python_root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONNOUSERSITE"] = "1"

    def run_coordinator(source, *arguments):
        completed = subprocess.run(
            [sys.executable, "-s", "-B", "-c", textwrap.dedent(source), *map(str, arguments)],
            cwd=python_root, env=environment, capture_output=True, text=True, timeout=300,
        )
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    first = """
        import sys
        from pathlib import Path
        from KHz_filament.hr4e5_storage import StorageBudget
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign

        root, campaign_id, config, source, lut, pre0, failure_point = sys.argv[1:]
        spec = E5AProductionSpec(
            root=Path(root), campaign_id=campaign_id, config_path=Path(config),
            source_manifest_path=Path(source), lut_manifest_path=Path(lut),
            pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=2,
            safety_margin_bytes=1024**2,
        )
        injected = {"done": False}
        if failure_point in {"before_complete", "after_rebind"}:
            original = StorageBudget.complete_intent
            def fail_before_complete(self, intent_id, **kwargs):
                item = self.intent(intent_id)
                if item.get("role") == "REFERENCE_HYDRO" and not injected["done"]:
                    injected["done"] = True
                    raise RuntimeError("injected after promotion before hydro intent completion")
                return original(self, intent_id, **kwargs)
            StorageBudget.complete_intent = fail_before_complete
        elif failure_point == "before_consume":
            original = StorageBudget.consume
            def fail_before_consume(self, reservation_id, **kwargs):
                item = self.reservations().get(str(reservation_id), {})
                if item.get("purpose") == "REFERENCE_HYDRO:R:p0" and not injected["done"]:
                    injected["done"] = True
                    raise RuntimeError("injected after hydro intent completion before reservation consume")
                return original(self, reservation_id, **kwargs)
            StorageBudget.consume = fail_before_consume
        else:
            from KHz_filament.hr4e5s_streaming import StreamingLifecycle
            original = StreamingLifecycle.promote_next_to_current
            reference_state = (Path(root) / "R" / "p0" / "state").resolve()
            injected = {"done": False}
            def inject_reference_f08(self, actor="barrier"):
                if Path(self.root).resolve() == reference_state and not injected["done"]:
                    self._s5_fault = {
                        "fault_id": "F08_PROMOTION_TRANSACTION_BOUNDARY",
                        "target_screen": str(self.manifest["records"][-1]["screen_id"]),
                        "once": True,
                    }
                try:
                    return original(self, actor=actor)
                except RuntimeError as error:
                    if "F08_PROMOTION_TRANSACTION_BOUNDARY" in str(error):
                        injected["done"] = True
                    raise
            StreamingLifecycle.promote_next_to_current = inject_reference_f08
        try:
            open_e5_1a_production_campaign(spec).run()
        except RuntimeError as error:
            if failure_point == "f08":
                expected = "F08_PROMOTION_TRANSACTION_BOUNDARY"
            elif failure_point in {"before_complete", "after_rebind"}:
                expected = "injected after promotion before hydro intent completion"
            else:
                expected = "injected after hydro intent completion before reservation consume"
            if expected not in str(error) or not injected["done"]:
                raise
        else:
            raise AssertionError("promotion bookkeeping fault did not occur")
    """
    run_coordinator(first, spec.root, spec.campaign_id, spec.config_path,
                    spec.source_manifest_path, spec.lut_manifest_path,
                    spec.pre0_delta_n_path, failure_point)

    root = Path(spec.root)
    state_path = root / "E5_1A_CAMPAIGN_STATE.json"
    lifecycle_path = root / "R" / "p0" / "state" / "streaming_manifest.json"
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["barrier"]["status"] == "PASS"
    pointer_path = root / "R" / "p0" / "state" / "authoritative_generation.json"
    if failure_point == "f08":
        assert lifecycle["promotion"] is None
        assert pointer_path.is_file()
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        assert pointer["authoritative_namespace"] == "NEXT"
        pointer_sha256 = sha256_file(pointer_path)
    else:
        assert lifecycle["promotion"]["authoritative_namespace"] == "NEXT"
        pointer_sha256 = None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["pairs"][0]["formal_steps"]["candidate"]["status"] == "COMMITTED"
    assert state["pairs"][0]["formal_steps"]["reference"]["status"] == "INTERRUPTED"
    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    intents = [item for item in ledger["intents"].values()
               if item.get("role") == "REFERENCE_HYDRO" and item.get("trajectory") == "R"
               and int(item.get("pulse", -1)) == 0]
    assert len(intents) == 1
    intent_id = intents[0]["intent_id"]
    reservation_id = intents[0]["reservation_id"]
    assert intents[0]["status"] == ("COMPLETED" if failure_point == "before_consume" else "INTERRUPTED")
    assert intents[0]["metadata"]["reservation_owner_epoch"] == "0"
    assert ledger["reservations"][reservation_id]["status"] == "ACTIVE"

    if failure_point == "after_rebind":
        interrupted_recovery = """
            import sys
            from pathlib import Path
            from KHz_filament.hr4e5_storage import StorageBudget
            from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign

            root, campaign_id, config, source, lut, pre0 = sys.argv[1:]
            spec = E5AProductionSpec(
                root=Path(root), campaign_id=campaign_id, config_path=Path(config),
                source_manifest_path=Path(source), lut_manifest_path=Path(lut),
                pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=2,
                safety_margin_bytes=1024**2,
            )
            original = StorageBudget.complete_intent
            injected = {"done": False}
            def fail_after_rebind(self, intent_id, **kwargs):
                item = self.intent(intent_id)
                if item.get("role") == "REFERENCE_HYDRO" and not injected["done"]:
                    injected["done"] = True
                    raise RuntimeError("injected after hydro intent epoch rebind")
                return original(self, intent_id, **kwargs)
            StorageBudget.complete_intent = fail_after_rebind
            try:
                open_e5_1a_production_campaign(spec).run()
            except RuntimeError as error:
                if "injected after hydro intent epoch rebind" not in str(error) or not injected["done"]:
                    raise
            else:
                raise AssertionError("post-rebind fault did not occur")
        """
        run_coordinator(interrupted_recovery, spec.root, spec.campaign_id, spec.config_path,
                        spec.source_manifest_path, spec.lut_manifest_path, spec.pre0_delta_n_path)
        ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
        rebound = ledger["intents"][intent_id]
        assert rebound["status"] == "INTERRUPTED" and rebound["epoch"] == "1"
        assert rebound["takeover_from_epoch"] == "0"
        assert ledger["reservations"][reservation_id]["status"] == "ACTIVE"
        assert ledger["reservations"][reservation_id]["owner"] == "0"

    second = """
        import sys
        from pathlib import Path
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign

        root, campaign_id, config, source, lut, pre0 = sys.argv[1:]
        spec = E5AProductionSpec(
            root=Path(root), campaign_id=campaign_id, config_path=Path(config),
            source_manifest_path=Path(source), lut_manifest_path=Path(lut),
            pre0_delta_n_path=Path(pre0), n_pulses=2, candidate_hydro_workers=2,
            safety_margin_bytes=1024**2,
        )
        result = open_e5_1a_production_campaign(spec).run()
        assert result["status"] == "COMPLETE"
        assert result["terminal_inventory_after_close"]["status"] == "PASS"
    """
    run_coordinator(second, spec.root, spec.campaign_id, spec.config_path,
                    spec.source_manifest_path, spec.lut_manifest_path, spec.pre0_delta_n_path)
    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    assert ledger["intents"][intent_id]["status"] == "COMPLETED"
    assert ledger["intents"][intent_id]["reservation_id"] == reservation_id
    assert ledger["reservations"][reservation_id]["status"] == "CONSUMED"
    if failure_point == "after_rebind":
        assert ledger["intents"][intent_id]["takeover_from_epoch"] == "1"
        assert ledger["reservations"][reservation_id]["owner"] == "0"
    assert sum(item.get("purpose") == "REFERENCE_HYDRO:R:p0"
               for item in ledger["reservations"].values()) == 1
    assert sum(item.get("role") == "REFERENCE_HYDRO" and item.get("trajectory") == "R"
               and int(item.get("pulse", -1)) == 0 for item in ledger["intents"].values()) == 1
    if failure_point == "f08":
        recovered_lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        assert recovered_lifecycle["promotion"] == json.loads(pointer_path.read_text(encoding="utf-8"))
        assert sha256_file(pointer_path) == pointer_sha256
    exact = json.loads((root / "exact" / "p0.json").read_text(encoding="utf-8"))
    assert exact["status"] == "PASS" and exact["mismatch_count"] == 0


def test_production_rc_and_candidate_internal_overlap_evidence(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    spec = _local_production_spec(
        tmp_path, campaign_id="local-production-rc-overlap", n_pulses=2, k=16,
        candidate_hydro_workers=4,
    )
    result = open_e5_1a_production_campaign(spec).run()
    assert result["status"] == "COMPLETE"
    state = json.loads((Path(spec.root) / "E5_1A_CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
    for pulse in range(2):
        events = [item for item in state["pairs"][pulse]["events"]
                  if item.get("event") in {"R_START", "REFERENCE_END", "C_START", "CANDIDATE_END"}]
        rows = {item["event"]: float(item["monotonic_s"]) for item in events}
        assert set(rows) == {"R_START", "REFERENCE_END", "C_START", "CANDIDATE_END"}
        assert max(rows["R_START"], rows["C_START"]) < min(rows["REFERENCE_END"], rows["CANDIDATE_END"])
    manifest = json.loads((Path(spec.root) / "C" / "p0" / "state" /
                           "streaming_manifest.json").read_text(encoding="utf-8"))
    first_hydro = min(float(item["monotonic_s"]) for item in manifest["telemetry_events"]
                      if item.get("event") == "HYDRO_BLOCK_START")
    optical_end = max(float(item["monotonic_s"]) for item in manifest["telemetry_events"]
                      if item.get("event") == "OPTICAL_COMPLETE")
    assert first_hydro < optical_end
    assert result["status"] == "COMPLETE" and result["next_pair_index"] == 2

def test_production_rejects_unsupported_candidate_hydro_workers(tmp_path):
    from dataclasses import replace
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    spec = _local_production_spec(tmp_path, campaign_id="bad-hydro-workers")
    with pytest.raises(ValueError, match="Candidate hydro workers"):
        open_e5_1a_production_campaign(replace(spec, candidate_hydro_workers=3))


def test_formal_n2_derivation_changes_only_pulse_count(tmp_path):
    import json
    from KHz_filament.hr4e5_production import derive_e5_1b_execution_config

    inputs = tmp_path / "inputs"; inputs.mkdir(parents=True)
    value = {
        "grid": {"Nx": 8, "Ny": 8, "Nt": 8, "Lx": 8e-4, "Ly": 8e-4, "Twin": 80e-15},
        "beam": {"w0": 1.5e-4, "tau_fwhm": 40e-15, "energy_J": 1e-10, "focal_length": None},
        "propagation": {"z_max": 8e-4, "dz": 1e-4, "linear_model": "paraxial"},
        "ionization": {"species": []},
        "heat": {"hr3b_enabled": True, "f_rep": 5e6, "dt_hydro": 1e-7,
                 "chi": 2.17e-5, "D_th": 2.17e-5, "nu": 1.5e-5,
                 "gravity_x": 0.0, "gravity_y": -9.81},
        "run": {"Npulses": 1},
        "raman": {"enabled": False, "absorption": False},
    }
    parent = inputs / "full.json"; parent.write_text(json.dumps(value), encoding="utf-8")
    derived = inputs / "execution.json"
    receipt = derive_e5_1b_execution_config(parent, derived)
    result = json.loads(derived.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(value)); expected["run"]["Npulses"] = 2
    assert result == expected and receipt["value"] == 2


def test_production_n3_k8_split_process_matches_uninterrupted(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament.hr4e5_storage import plan_campaign_budget

    split_base = tmp_path / "split"
    split_spec = _local_production_spec(
        split_base, campaign_id="local-production-n3-split", n_pulses=3, k=8,
    )
    child = textwrap.dedent(
        """
        import sys
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
        spec = E5AProductionSpec(
            root=sys.argv[1], campaign_id=sys.argv[2], config_path=sys.argv[3],
            source_manifest_path=sys.argv[4], lut_manifest_path=sys.argv[5],
            pre0_delta_n_path=sys.argv[6], n_pulses=3, safety_margin_bytes=1024**2,
        )
        result = open_e5_1a_production_campaign(spec).run(stop_after=1)
        assert result["next_pair_index"] == 1
        """
    )
    first = subprocess.run(
        [sys.executable, "-s", "-B", "-c", child, str(split_spec.root),
         split_spec.campaign_id, str(split_spec.config_path), str(split_spec.source_manifest_path),
         str(split_spec.lut_manifest_path), str(split_spec.pre0_delta_n_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert first.returncode == 0, first.stdout + "\n" + first.stderr
    split_result = open_e5_1a_production_campaign(split_spec).run()
    assert split_result["status"] == "COMPLETE" and split_result["next_pair_index"] == 3

    baseline_base = tmp_path / "baseline"
    baseline_seed = _local_production_spec(
        baseline_base, campaign_id="local-production-n3-baseline", n_pulses=3, k=8,
    )
    baseline_result = open_e5_1a_production_campaign(baseline_seed).run()
    assert baseline_result["status"] == "COMPLETE"

    def exact_signature(root):
        reports = [json.loads((root / "exact" / f"p{pulse}.json").read_text(encoding="utf-8"))
                   for pulse in range(3)]
        return reports, [
            (row["comparison_key"], row["reference_sha256_array"], row["candidate_sha256_array"])
            for report in reports for row in report["rows"]
        ]

    split_root = Path(split_spec.root)
    baseline_root = Path(baseline_seed.root)
    split_exact, split_signature = exact_signature(split_root)
    baseline_exact, baseline_signature = exact_signature(baseline_root)
    assert [item["screen_count"] for item in split_exact] == [120, 120, 96]
    assert sum(item["screen_count"] for item in split_exact) == 336
    assert sum(item["ledger_count"] for item in split_exact) == 27
    assert sum(item["optical_count"] for item in split_exact) == 3
    assert split_signature == baseline_signature
    for root in (split_root, baseline_root):
        assert not list(root.glob("*/p*/optical/sinks.*.npy"))
        assert len(list(root.glob("*/p*/optical/final_optical_field.npy"))) == 6
        assert len(list(root.glob("*/p*/optical/scientific_ledger.npz"))) == 6
        storage = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
        assert len(storage["gc_plans"]) == 6
        assert all(plan["status"] == "COMPLETED" for plan in storage["gc_plans"].values())

    def handoff_count(root):
        total = 0
        for side in ("R", "C"):
            for pulse in (1, 2):
                report = json.loads((root / side / f"p{pulse}" / "state" /
                                     "parent_next_child_current_exact.json").read_text(encoding="utf-8"))
                total += int(report["compared_object_count"])
        return total

    assert handoff_count(split_root) == handoff_count(baseline_root) == 96

    def durable_signature(root):
        ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
        state = json.loads((root / "E5_1A_CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
        roles = json.loads((root / "E5_1A_TERMINAL_ROLE_CONTRACT.json").read_text(encoding="utf-8"))["roles"]
        return {
            "gc": sorted((plan["trajectory"], plan["pulse"], plan["status"], len(plan["targets"]))
                         for plan in ledger["gc_plans"].values()),
            "writers": sorted((item["role"], item["status"], item["trajectory"], item["pulse"])
                              for item in ledger["writers"].values()),
            "reservations": sorted(item["status"] for item in ledger["reservations"].values()),
            "roles": {key: len(value) for key, value in roles.items()},
            "pairs": [(pair["pulse_index"], pair["status"]) for pair in state["pairs"]],
        }, state

    split_durable, split_state = durable_signature(split_root)
    baseline_durable, baseline_state = durable_signature(baseline_root)
    assert split_durable == baseline_durable
    assert set(split_durable["reservations"]) == {"CONSUMED"}
    assert len(split_state["process_epochs"]) == 2
    assert len(baseline_state["process_epochs"]) == 1
    assert all(item["status"] == "EXITED" for item in split_state["process_epochs"])
    local_plan = plan_campaign_budget(n_pulses=3, k=8, ny=8, nx=8, nt=8,
                                      safety_margin_bytes=1024**2)
    assert local_plan["peak_bytes"] < local_plan["max_campaign_live_bytes"]
    admission_report = json.loads(
        (split_root / "R" / "p0" / "reports" / "admission.json").read_text(encoding="utf-8")
    )
    assert admission_report["capacity_snapshot"]["scope"] == "LOCAL_POLICY_CAP_NOT_SITE_QUOTA"
    assert admission_report["capacity_snapshot"]["filesystem_free_bytes"] > 0
    assert admission_report["capacity_snapshot"]["site_quota_measured"] is False

    readonly = textwrap.dedent(
        """
        import sys
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
        spec = E5AProductionSpec(root=sys.argv[1], campaign_id=sys.argv[2], config_path=sys.argv[3],
            source_manifest_path=sys.argv[4], lut_manifest_path=sys.argv[5], pre0_delta_n_path=sys.argv[6],
            n_pulses=3, safety_margin_bytes=1024**2)
        campaign = open_e5_1a_production_campaign(spec)
        assert campaign.report()["status"] == "COMPLETE"
        campaign.close()
        """
    )
    third = subprocess.run(
        [sys.executable, "-s", "-B", "-c", readonly, str(split_spec.root), split_spec.campaign_id,
         str(split_spec.config_path), str(split_spec.source_manifest_path), str(split_spec.lut_manifest_path),
         str(split_spec.pre0_delta_n_path)], capture_output=True, text=True, timeout=30,
    )
    assert third.returncode == 0, third.stdout + "\n" + third.stderr
    assert not (split_root / "R" / "p3").exists()
    assert not (split_root / "C" / "p3").exists()


@pytest.mark.parametrize("fault", [
    "after_exact", "after_ready", "partial_sink", "after_unlink", "after_gc_plan",
    "after_reference_gc", "after_both_gc", "after_complete",
])
def test_storage_loop_terminal_new_process_recovery(tmp_path, fault):
    """A real K8 optical run survives each durable terminal boundary."""
    from KHz_filament.hr4e5_evidence import sha256_file
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament import hr4e5_production as production

    spec = _local_production_spec(tmp_path / "interrupted", campaign_id=f"terminal-{fault}")
    child = textwrap.dedent(r"""
        import os, sys
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
        from KHz_filament import hr4e5_production as production
        from KHz_filament.hr4e5_storage import StorageBudget
        fault = sys.argv[7]
        if fault == 'after_exact':
            original = production._E5AProductionRunner._run_exact
            def injected(self, pulse):
                original(self, pulse)
                os._exit(79)
            production._E5AProductionRunner._run_exact = injected
        elif fault == 'after_ready':
            original = production.validate_final_post
            def injected(**kw):
                result = original(**kw)
                if result['status'] == 'PASS': os._exit(79)
                return result
            production.validate_final_post = injected
        elif fault in {'partial_sink', 'after_unlink', 'after_gc_plan'}:
            original = StorageBudget.apply_reclaim
            def injected(self, plan_id, **kw):
                if plan_id.endswith('-R-p0'):
                    if fault == 'partial_sink':
                        original(self, plan_id, interrupt_after=1, **kw)
                        os._exit(79)
                    if fault == 'after_gc_plan':
                        result = original(self, plan_id, **kw)
                        os._exit(79)
                    def exit_after_unlink(stage, relative):
                        if stage == 'after_unlink': os._exit(79)
                    kw['fault_hook'] = exit_after_unlink
                return original(self, plan_id, **kw)
            StorageBudget.apply_reclaim = injected
        elif fault in {'after_reference_gc', 'after_both_gc'}:
            original = production._E5AProductionRunner._run_gc
            def injected(self, trajectory, pulse, **kw):
                result = original(self, trajectory, pulse, **kw)
                if kw.get('terminal_ready_path') is not None and (
                    trajectory == ('R' if fault == 'after_reference_gc' else 'C')):
                    os._exit(79)
                return result
            production._E5AProductionRunner._run_gc = injected
        spec = E5AProductionSpec(root=sys.argv[1], campaign_id=sys.argv[2],
            config_path=sys.argv[3], source_manifest_path=sys.argv[4],
            lut_manifest_path=sys.argv[5], pre0_delta_n_path=sys.argv[6],
            n_pulses=1, safety_margin_bytes=1024**2)
        assert open_e5_1a_production_campaign(spec).run()['status'] == 'COMPLETE'
    """)
    args = [sys.executable, "-s", "-B", "-c", child, str(spec.root), spec.campaign_id,
            str(spec.config_path), str(spec.source_manifest_path), str(spec.lut_manifest_path),
            str(spec.pre0_delta_n_path), fault]
    first = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert first.returncode == (0 if fault == "after_complete" else 79), first.stdout + first.stderr
    before = Path(spec.root) / "exact" / "p0.json"
    exact_hash = sha256_file(before)
    resume_child = textwrap.dedent(r"""
        import sys
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
        spec = E5AProductionSpec(root=sys.argv[1], campaign_id=sys.argv[2],
            config_path=sys.argv[3], source_manifest_path=sys.argv[4],
            lut_manifest_path=sys.argv[5], pre0_delta_n_path=sys.argv[6],
            n_pulses=1, safety_margin_bytes=1024**2)
        assert open_e5_1a_production_campaign(spec).run()['status'] == 'COMPLETE'
    """)
    resumed = subprocess.run(args[:4] + [resume_child] + args[5:-1],
                             capture_output=True, text=True, timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    root = Path(spec.root)
    assert sha256_file(before) == exact_hash


    exact = json.loads(before.read_text(encoding="utf-8"))
    assert (exact["screen_count"], exact["ledger_count"], exact["optical_count"]) == (96, 9, 1)
    assert len(exact["rows"]) == 106
    assert not list(root.glob("*/p0/optical/sinks.*.npy"))
    assert len(list((root / "C" / "p0" / "state").rglob("screen_*.npz"))) == 16
    assert not list((root / "R" / "p0" / "state").rglob("screen_*.npz"))
    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    assert set(ledger["gc_plans"]) == {"production-R-p0", "production-C-p0"}
    assert all(plan["status"] == "COMPLETED" for plan in ledger["gc_plans"].values())
    assert sum(sum(target["relative_path"].endswith(name) for target in plan["targets"]
                   for name in production._SINK_FILES.values())
               for plan in ledger["gc_plans"].values()) == 12
    baseline = _local_production_spec(tmp_path / "baseline", campaign_id=f"baseline-{fault}")
    assert open_e5_1a_production_campaign(baseline).run()["status"] == "COMPLETE"
    for side in ("R", "C"):
        assert np.array_equal(np.load(root / side / "p0" / "optical" / "final_optical_field.npy"),
                              np.load(Path(baseline.root) / side / "p0" / "optical" / "final_optical_field.npy"))
    for namespace in ("current", "post"):
        for index in range(8):
            relative = Path("C") / "p0" / "state" / namespace / f"screen_{index:06d}.npz"
            with np.load(root / relative, allow_pickle=False) as actual, np.load(Path(baseline.root) / relative, allow_pickle=False) as expected:
                for field in ("delta_n", "vx", "vy"):
                    assert np.array_equal(actual[field], expected[field])
    again = subprocess.run(args[:4] + [resume_child] + args[5:-1],
                           capture_output=True, text=True, timeout=120)
    assert again.returncode == 0, again.stdout + again.stderr
    assert sha256_file(before) == exact_hash


def test_storage_loop_internal_prefix_charged_once_and_terminal_bytes_agree(tmp_path):
    from dataclasses import replace
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament.hr4e5_storage import plan_campaign_budget

    spec = _local_production_spec(tmp_path, campaign_id="internal-prefix")
    internal = Path(spec.root) / "delta_n_prefix.npy"
    internal.parent.mkdir(parents=True, exist_ok=True)
    internal.write_bytes(Path(spec.pre0_delta_n_path).read_bytes())
    campaign = open_e5_1a_production_campaign(replace(spec, pre0_delta_n_path=internal))
    assert campaign.run()["status"] == "COMPLETE"
    budget = campaign._runner.budget
    roles = campaign._runner.expected_terminal_roles(0)
    inventory = budget.validate_terminal_inventory(roles, root=spec.root, final=True)
    assert roles["EXTERNAL_PREFIX"] == ["delta_n_prefix.npy"]
    assert budget.external_prefix_bytes == 0
    assert budget.final_output_bytes() == inventory["final_output_bytes"]
    assert budget.final_output_bytes() == budget.actual_bytes()
    args = {"n_pulses": 3, "k": 8, "ny": 8, "nx": 8, "nt": 8,
            "safety_margin_bytes": 1024**2}
    existing = plan_campaign_budget(**args, external_prefix_already_present=True)
    new = plan_campaign_budget(**args, external_prefix_already_present=False)
    assert existing["peak_bytes"] == new["peak_bytes"]
    assert existing["final_output_bytes"] == new["final_output_bytes"]
    assert new["incremental_requirement_bytes"] - existing["incremental_requirement_bytes"] == existing["field_bytes"]


@pytest.mark.parametrize("damage", ["changed", "unknown_missing"])
def test_storage_loop_rejects_sink_changed_before_reclaim(tmp_path, monkeypatch, damage):
    from KHz_filament import hr4e5_production as production
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    spec = _local_production_spec(tmp_path, campaign_id=f"sink-{damage}")
    original = production._E5AProductionRunner._run_gc
    def damaged_gc(self, trajectory, pulse, **kw):
        if trajectory == "R" and kw.get("terminal_ready_path") is not None:
            sink = self._track("R", pulse) / "optical" / production._SINK_FILES["ion"]
            if damage == "changed":
                values = np.load(sink, allow_pickle=False)
                values[0, 0, 0] += 1.0
                np.save(sink, values)
            else:
                sink.unlink()
        return original(self, trajectory, pulse, **kw)
    monkeypatch.setattr(production._E5AProductionRunner, "_run_gc", damaged_gc)
    with pytest.raises((ValueError, RuntimeError), match="exact|missing|changed"):
        open_e5_1a_production_campaign(spec).run()
    root = Path(spec.root)
    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    assert ledger["gc_plans"] == {}
    assert (root / "R" / "p0" / "optical" / production._SINK_FILES["ib"]).is_file()


@pytest.mark.parametrize("component", ["config", "parent", "root"])
def test_storage_loop_rejects_lexical_reparse_before_root_creation(tmp_path, monkeypatch, component):
    from KHz_filament import hr4e5_production as production
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    spec = _local_production_spec(tmp_path, campaign_id=f"reparse-{component}")
    target = {"config": Path(spec.config_path),
              "parent": Path(spec.config_path).parent,
              "root": Path(spec.root)}[component]
    actual = production._is_reparse
    monkeypatch.setattr(production, "_is_reparse", lambda path: Path(path) == target or actual(path))
    with pytest.raises(StorageIntegrityError, match="symlink|reparse"):
        open_e5_1a_production_campaign(spec)
    assert not Path(spec.root).exists()


def _bound_prefix_spec(tmp_path):
    """A full K16 source with a distinct K8 prefix and N3 execution config."""
    from dataclasses import replace
    from KHz_filament.hr4e5_evidence import sha256_file
    from KHz_filament.hr4e5_production import (
        _source_hash, derive_e5_1b_execution_config, describe_e5_1b_input_contract,
    )
    from KHz_filament.hr4e_timestep import sha256_array_streaming

    seed = _local_production_spec(tmp_path, campaign_id="bound-prefix", n_pulses=2, k=8)
    inputs = Path(seed.config_path).parent
    full_config = inputs / "full_config.json"
    full = json.loads(Path(seed.config_path).read_text(encoding="utf-8"))
    full["run"]["Npulses"] = 1
    full["propagation"]["z_max"] = 16e-4
    full_config.write_text(json.dumps(full), encoding="utf-8")
    derived = inputs / "execution_config.json"
    derive_e5_1b_execution_config(full_config, derived)
    Path(seed.source_manifest_path).write_text(json.dumps({
        "schema": "local-source.v1", "canonical_array_hash": _source_hash(derived),
    }), encoding="utf-8")
    Path(seed.lut_manifest_path).write_text(json.dumps({
        "schema": "local-lut.v1", "mode": "DISABLED_BY_CONFIG",
        "config_sha256": sha256_file(derived),
    }), encoding="utf-8")
    full_slow = inputs / "full_slow.npy"
    values = np.arange(16 * 8 * 8, dtype=np.float64).reshape(16, 8, 8) * 1e-12
    np.save(full_slow, values)
    np.save(seed.pre0_delta_n_path, values[:8])
    slow_manifest = inputs / "slow_source.json"
    contract = describe_e5_1b_input_contract(full_config, prefix_intervals=8)
    slow_manifest.write_text(json.dumps({
        "schema": "khz_filament.hr4e5.e5_1b.slow_source.v1",
        "source_path": str(full_slow), "source_raw_sha256": sha256_file(full_slow),
        "source_array_sha256": sha256_array_streaming(values),
        "source_shape": list(values.shape),
        "source_config_sha256": sha256_file(full_config),
        "axis_order": ["interval", "y", "x"],
        "source_indices": [0, 15],
        "full_edges_array_sha256": contract["full_edges_array_sha256"],
        "full_midpoints_array_sha256": contract["full_midpoints_array_sha256"],
        "capacity_domain": "campaign_filesystem",
    }), encoding="utf-8")
    return replace(seed, config_path=derived, full_config_path=full_config,
                   prefix_intervals=8, slow_source_manifest_path=slow_manifest)


def test_e5_1b_full_prefix_runs_real_optical_and_retains_two_sources(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    spec = _bound_prefix_spec(tmp_path)
    campaign = open_e5_1a_production_campaign(spec)
    identity = campaign.report()["admission_identity"]
    assert identity["schedule_identity"]["full_metadata"]["n_intervals"] == 16
    assert identity["schedule_identity"]["metadata"]["n_intervals"] == 8
    assert identity["schedule_identity"]["source_indices"] == [0, 7]
    assert identity["config_identity"]["full_source_config"]["sha256"] != identity["config_identity"]["sha256"]
    slow = identity["pre0_identity"]["slow_source_identity"]
    assert slow["full_source_array_sha256"] != slow["prefix_array_sha256"]
    result = campaign.run()
    assert result["status"] == "COMPLETE"
    root = Path(spec.root)
    for side in ("R", "C"):
        for pulse in range(2):
            output = root / side / f"p{pulse}" / "optical"
            assert (output / "final_optical_field.npy").is_file()
            assert (output / "source_identity.json").is_file()
    assert (root / "R" / "p1" / "reports" / "gc.json").is_file()


@pytest.mark.parametrize("fault", ["prefix_value", "axis_order", "dtype", "shape", "source_value", "config_change"])
def test_e5_1b_bound_prefix_rejects_tamper_before_campaign_root(tmp_path, fault):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    spec = _bound_prefix_spec(tmp_path)
    prefix = Path(spec.pre0_delta_n_path)
    full_source = Path(json.loads(Path(spec.slow_source_manifest_path).read_text(encoding="utf-8"))["source_path"])
    if fault in {"prefix_value", "axis_order", "dtype", "shape"}:
        values = np.load(prefix)
        if fault == "prefix_value": values[0, 0, 0] += 1e-9
        if fault == "axis_order": values = values.transpose(0, 2, 1)
        if fault == "dtype": values = values.astype(np.float32)
        if fault == "shape": values = values[:-1]
        np.save(prefix, values)
    elif fault == "source_value":
        values = np.load(full_source); values[0, 0, 0] += 1e-9; np.save(full_source, values)
    else:
        config = json.loads(Path(spec.full_config_path).read_text(encoding="utf-8"))
        config["propagation"]["dz"] *= 2
        Path(spec.full_config_path).write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        open_e5_1a_production_campaign(spec)
    assert not Path(spec.root).exists()


def test_e5_1b_budget_objects_reconcile_and_reservation_is_shared():
    from KHz_filament.hr4e5_formal_entry import optical_write_forecast
    from KHz_filament.hr4e5_storage import StorageBudgetError, budget_object_table, plan_campaign_budget, plan_final_output

    args = dict(n_pulses=3, k=8048, ny=351, nx=301, nt=384)
    table = budget_object_table(**args, external_prefix_already_present=True)
    for stage, rows in table["stages"].items():
        assert sum(row["bytes"] for row in rows) == table["stage_bytes"][stage]
        assert len({row["object"] for row in rows}) == len(rows)
    plan = plan_campaign_budget(**args, external_prefix_already_present=True)
    assert plan["peak_bytes"] == table["peak_bytes"]
    assert plan["final_output_bytes"] == table["final_output_bytes"]
    assert plan_final_output(**args)["final_output_bytes"] == table["final_output_bytes"]
    with pytest.raises(StorageBudgetError, match="exceeds cap"):
        plan_campaign_budget(**args, max_campaign_live_bytes=plan["peak_bytes"] - 1)
    with pytest.raises(StorageBudgetError, match="exceeds final budget"):
        plan_campaign_budget(**args, final_output_budget_bytes=plan["final_output_bytes"] - 1)
    with pytest.raises(StorageBudgetError, match="exceed"):
        plan_final_output(**args, diagnostics=("ion", "ib", "raman", "qthermal", "increment", "state_after"))
    assert table["incremental_requirement_bytes"] == table["peak_bytes"] - table["external_prefix_bytes"]
    assert table["stage_bytes"]["final_retention"] < 68719476736
    assert next(row for row in table["stages"]["final_retention"] if row["object"] == "full_interval_sink_arrays")["bytes"] == 0
    assert optical_write_forecast(k=8048, ny=351, nx=301, nt=384, final=True) > 0


def test_production_writer_live_refusal_and_dead_stale_takeover(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament.hr4e5_storage import StorageBudgetError, StorageIntegrityError

    spec = _local_production_spec(
        tmp_path, campaign_id="local-production-writer-takeover", n_pulses=2, k=8,
    )
    coordinator = textwrap.dedent(
        """
        import os, sys
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
        spec = E5AProductionSpec(root=sys.argv[1], campaign_id=sys.argv[2], config_path=sys.argv[3],
            source_manifest_path=sys.argv[4], lut_manifest_path=sys.argv[5], pre0_delta_n_path=sys.argv[6],
            n_pulses=2, safety_margin_bytes=1024**2)
        open_e5_1a_production_campaign(spec)
        os._exit(17)
        """
    )
    first = subprocess.run(
        [sys.executable, "-s", "-B", "-c", coordinator, str(spec.root), spec.campaign_id,
         str(spec.config_path), str(spec.source_manifest_path), str(spec.lut_manifest_path),
         str(spec.pre0_delta_n_path)], capture_output=True, text=True, timeout=30,
    )
    assert first.returncode == 17

    writer_script = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from KHz_filament.hr4e5_storage import StorageBudget, MockQuotaProvider
        root = Path(sys.argv[1])
        identity = json.loads((root / "E5_1A_ADMISSION_IDENTITY.json").read_text(encoding="utf-8"))
        state = json.loads((root / "E5_1A_CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
        epoch = state["active_coordinator"]["epoch"]
        budget = StorageBudget(root, safety_margin_bytes=1024**2,
            provider=MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40), require_quota=True,
            campaign_id=identity["campaign_id"], require_intents=True,
            admission_hash=identity["identity_sha256"], admission_identity=identity,
            external_prefix_bytes=identity["budget_identity"]["external_prefix_bytes"],
            external_full_source_bytes=identity["budget_identity"]["existing_full_source_bytes"])
        residual = root / "crash_writer" / "residual.bin"
        reservation = budget.reserve(1024**2, purpose="crash-writer", allocation_paths=[residual])
        intent = budget.create_intent(reservation_id=reservation.reservation_id, trajectory="C",
            pulse=0, attempt=0, role="CRASH_TEST", allowed_paths=[residual], expected_bytes=1024**2,
            admission_hash=identity["identity_sha256"], epoch=epoch, generation="crash:test")
        writer = budget.open_writer(reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
            coordinator_epoch=epoch, trajectory="C", pulse=0, attempt=0, generation="crash:test",
            writer_id="crash-writer")
        residual.parent.mkdir(parents=True, exist_ok=True)
        residual.write_bytes(b"stale-residual")
        (root / "writer_started.json").write_text(json.dumps({"reservation_id": reservation.reservation_id,
            "intent_id": intent["intent_id"], "writer_id": writer["writer_id"], "epoch": epoch}), encoding="utf-8")
        sys.stdin.read()
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-s", "-B", "-c", writer_script, str(spec.root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    marker = Path(spec.root) / "writer_started.json"
    for _ in range(100):
        if marker.is_file():
            break
        time.sleep(0.05)
    assert marker.is_file()
    with pytest.raises(StorageBudgetError, match="live ACTIVE writers block takeover"):
        open_e5_1a_production_campaign(spec)
    process.terminate()
    process.wait(timeout=15)

    recovered = open_e5_1a_production_campaign(spec)
    storage = recovered._driver.campaign.storage
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    ledger = json.loads((Path(spec.root) / "storage_ledger.json").read_text(encoding="utf-8"))
    stale = ledger["writers"][marker_value["writer_id"]]
    assert stale["status"] == "INTERRUPTED_STALE"
    assert stale["liveness_evidence"]["status"] == "DEAD"
    assert ledger["reservations"][marker_value["reservation_id"]]["status"] == "ACTIVE"
    assert (Path(spec.root) / "crash_writer" / "residual.bin").is_file()

    new_epoch = recovered._driver.coordinator_epoch
    replacement = storage.open_writer(
        reservation_id=marker_value["reservation_id"], intent_id=marker_value["intent_id"],
        coordinator_epoch=new_epoch, trajectory="C", pulse=0, attempt=0,
        generation="crash:test", writer_id="replacement-writer",
    )
    with pytest.raises(StorageIntegrityError, match="stale epoch"):
        storage.close_writer(replacement["writer_id"], coordinator_epoch=marker_value["epoch"])
    completed = storage.complete_intent(
        marker_value["intent_id"], files=[Path(spec.root) / "crash_writer" / "residual.bin"],
    )
    storage.consume(marker_value["reservation_id"], actual_bytes=completed["actual_bytes"])
    storage.close_writer(replacement["writer_id"], coordinator_epoch=new_epoch)
    recovered.close()

    identity = storage.active_writers()
    assert identity == []


def test_writer_pid_reuse_and_unknown_liveness_fail_closed(tmp_path):
    from KHz_filament.hr4e5_storage import (
        MockQuotaProvider, StorageBudget, StorageIntegrityError, process_identity,
    )

    def opened(root, process, writer_id):
        budget = StorageBudget(
            root, provider=MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40),
            require_quota=True, campaign_id=writer_id, require_intents=True,
            admission_hash="a" * 64,
        )
        target = root / "residual.bin"
        reservation = budget.reserve(1024**2, purpose=writer_id, allocation_paths=[target])
        intent = budget.create_intent(
            reservation_id=reservation.reservation_id, trajectory="C", pulse=0, attempt=0,
            role="CRASH_TEST", allowed_paths=[target], expected_bytes=1024**2,
            admission_hash="a" * 64, epoch=0, generation="reuse:test",
        )
        budget.open_writer(
            reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
            coordinator_epoch=0, trajectory="C", pulse=0, attempt=0,
            generation="reuse:test", process=process, writer_id=writer_id,
        )
        target.write_bytes(b"residual")
        return budget, reservation

    reused = process_identity()
    reused["process_start_token"] += "-reused"
    reused_budget, reused_reservation = opened(tmp_path / "reused", reused, "pid-reused")
    result = reused_budget.interrupt_stale_writers(coordinator_epoch=0)
    assert result["probes"]["pid-reused"]["reason"] == "pid_reused"
    reused_ledger = json.loads((tmp_path / "reused" / "storage_ledger.json").read_text(encoding="utf-8"))
    assert reused_ledger["writers"]["pid-reused"]["status"] == "INTERRUPTED_STALE"
    assert reused_ledger["reservations"][reused_reservation.reservation_id]["status"] == "ACTIVE"

    unknown = process_identity()
    unknown["host_identity"] = "unknown-remote-host"
    unknown_budget, unknown_reservation = opened(tmp_path / "unknown", unknown, "unknown-host")
    with pytest.raises(StorageIntegrityError, match="unverifiable ACTIVE writers block takeover"):
        unknown_budget.interrupt_stale_writers(coordinator_epoch=0)
    unknown_ledger = json.loads((tmp_path / "unknown" / "storage_ledger.json").read_text(encoding="utf-8"))
    assert unknown_ledger["writers"]["unknown-host"]["status"] == "ACTIVE"
    assert unknown_ledger["reservations"][unknown_reservation.reservation_id]["status"] == "ACTIVE"


def _child_grant_fixture(tmp_path, *, launch_candidate_identity=None):
    from KHz_filament.hr4e5_storage import MockQuotaProvider, StorageBudget

    root = tmp_path / "child-grants"
    budget = StorageBudget(
        root, safety_margin_bytes=1024**2,
        provider=MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40),
        require_quota=True, campaign_id="child-grant-test", require_intents=True,
        admission_hash="a" * 64,
        admission_identity={"identity_sha256": "a" * 64,
                           "hydro_topology": {"candidate_hydro_workers_requested": 2}},
    )
    allocation = root / "pulse0"
    target = allocation / "optical"
    reservation = budget.reserve(8192, purpose="single-aggregate", allocation_paths=[allocation])
    intent = budget.create_intent(
        reservation_id=reservation.reservation_id, trajectory="C", pulse=0,
        attempt=0, role="OPTICAL", allowed_paths=[target], expected_bytes=4096,
        admission_hash="a" * 64, epoch="epoch-0", generation="current:g0",
    )
    grant = budget.create_child_writer_grant(
        reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
        coordinator_epoch="epoch-0", role="OPTICAL", trajectory="C", pulse=0,
        logical_attempt=0, generation="current:g0", scientific_pair_id="c" * 64,
        admission_identity_sha256="a" * 64,
        implementation_identity={"schema": "test-implementation.v1", "identity_sha256": "b" * 64},
        allowed_paths=[target], expected_bytes=4096,
        launch_candidate_identity=launch_candidate_identity,
    )
    return budget, reservation, intent, grant


def test_child_writer_grant_scope_bind_and_close_are_atomic(tmp_path):
    from KHz_filament.hr4e5_storage import StorageBudgetError, StorageIntegrityError, process_identity

    budget, reservation, intent, grant = _child_grant_fixture(tmp_path, launch_candidate_identity=process_identity())
    grant_id, nonce, scope = grant["grant_id"], grant["nonce"], grant["scope"]
    assert len(budget.pending_writer_grants(coordinator_epoch="epoch-0")) == 1
    assert budget.active_writers(coordinator_epoch="epoch-0") == []
    persisted = json.loads((budget.root / "storage_ledger.json").read_text(encoding="utf-8"))
    assert nonce not in (budget.root / "storage_ledger.json").read_text(encoding="utf-8")
    assert persisted["reservations"][reservation.reservation_id]["status"] == "ACTIVE"
    assert len(budget.reservations()) == 1
    with pytest.raises(StorageBudgetError, match="unresolved child writer grants"):
        budget.complete_intent(intent["intent_id"])

    with pytest.raises(StorageBudgetError, match="duplicate unresolved"):
        budget.create_child_writer_grant(
            reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
            coordinator_epoch="epoch-0", role="OPTICAL", trajectory="C", pulse=0,
            logical_attempt=0, generation="current:g0", scientific_pair_id="c" * 64,
            admission_identity_sha256="a" * 64,
            implementation_identity={"schema": "test-implementation.v1", "identity_sha256": "b" * 64},
            allowed_paths=[budget.root / "pulse0" / "optical"], expected_bytes=4096,
        )
    with pytest.raises(StorageIntegrityError, match="nonce"):
        budget.bind_child_writer_grant(grant_id, nonce="wrong", scope=scope)
    changed_scope = dict(scope, expected_bytes=4095)
    with pytest.raises(StorageIntegrityError, match="scope"):
        budget.bind_child_writer_grant(grant_id, nonce=nonce, scope=changed_scope)
    assert budget.pending_writer_grants(coordinator_epoch="epoch-0")[0]["state"] == "PENDING"

    bound = budget.bind_child_writer_grant(grant_id, nonce=nonce, scope=scope)
    assert bound["state"] == "BOUND_ACTIVE"
    assert bound["process_identity"] == process_identity()
    active = budget.active_writers(coordinator_epoch="epoch-0")
    assert len(active) == 1 and active[0]["writer_kind"] == "CHILD_GRANT"
    permitted_path = budget.root / "pulse0" / "optical" / "payload.bin"
    authorization = budget.authorize_child_writer_operation(
        grant_id, nonce=nonce, coordinator_epoch="epoch-0",
        operation="optical_output", path=permitted_path,
    )
    assert authorization["status"] == "PASS"
    with pytest.raises(StorageIntegrityError, match="outside grant role"):
        budget.authorize_child_writer_operation(
            grant_id, nonce=nonce, coordinator_epoch="epoch-0",
            operation="NEXT_COMMIT", path=permitted_path,
        )
    with pytest.raises(StorageIntegrityError, match="outside grant scope"):
        budget.authorize_child_writer_operation(
            grant_id, nonce=nonce, coordinator_epoch="epoch-0",
            operation="OPTICAL_OUTPUT", path=budget.root / "pulse0" / "other" / "payload.bin",
        )
    with pytest.raises(StorageBudgetError, match="PENDING or was already bound"):
        budget.bind_child_writer_grant(grant_id, nonce=nonce, scope=scope)
    with pytest.raises(StorageIntegrityError, match="stale coordinator epoch"):
        budget.close_child_writer_grant(grant_id, nonce=nonce, coordinator_epoch="epoch-old")
    closed = budget.close_child_writer_grant(grant_id, nonce=nonce, coordinator_epoch="epoch-0")
    assert closed["state"] == "CLOSED"
    assert budget.active_writers(coordinator_epoch="epoch-0") == []
    assert budget.unresolved_writer_grants(coordinator_epoch="epoch-0") == []
    reconciler = budget.open_coordinator_reconciler(
        reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
        coordinator_epoch="epoch-0", trajectory="C", pulse=0, attempt=0,
        generation="current:g0", writer_id="coordinator-reconciler",
    )
    assert reconciler["writer_kind"] == "COORDINATOR_RECONCILER"
    assert reconciler["manager_role"] == "RECONCILER"
    assert budget.active_writers(coordinator_epoch="epoch-0", include_management=False) == []
    budget.close_writer(reconciler["writer_id"], coordinator_epoch="epoch-0")
    with pytest.raises(StorageIntegrityError, match="cannot receive a child writer grant"):
        budget._child_role("RECONCILER", None)
    permitted_path.parent.mkdir(parents=True, exist_ok=True)
    permitted_path.write_bytes(b"payload")
    completed = budget.complete_intent(intent["intent_id"], files=[permitted_path])
    consumed = budget.consume(reservation.reservation_id, actual_bytes=completed["actual_bytes"])
    assert consumed["status"] == "CONSUMED"
    assert len(budget.reservations()) == 1


def test_pending_child_grant_requires_dead_launch_evidence_for_abandonment(tmp_path, monkeypatch):
    from KHz_filament import hr4e5_storage as storage_module
    from KHz_filament.hr4e5_storage import StorageBudgetError, StorageIntegrityError, process_identity

    budget, _reservation, _intent, grant = _child_grant_fixture(tmp_path, launch_candidate_identity=process_identity())
    with pytest.raises(StorageBudgetError, match="still LIVE"):
        budget.abandon_pending_writer_grant(grant["grant_id"], nonce=grant["nonce"])
    with pytest.raises(StorageBudgetError, match="live child writer grants"):
        budget.fence_dead_writer_grants(coordinator_epoch="epoch-0")
    assert budget.pending_writer_grants(coordinator_epoch="epoch-0")[0]["state"] == "PENDING"

    monkeypatch.setattr(storage_module, "probe_process_identity",
                        lambda _identity: {"status": "UNKNOWN", "reason": "test-unknown"})
    with pytest.raises(StorageIntegrityError, match="UNKNOWN"):
        budget.abandon_pending_writer_grant(grant["grant_id"], nonce=grant["nonce"])
    with pytest.raises(StorageIntegrityError, match="unverifiable child writer grants"):
        budget.fence_dead_writer_grants(coordinator_epoch="epoch-0")
    assert budget.pending_writer_grants(coordinator_epoch="epoch-0")[0]["state"] == "PENDING"

    monkeypatch.setattr(storage_module, "probe_process_identity",
                        lambda _identity: {"status": "DEAD", "reason": "test-dead"})
    abandoned = budget.abandon_pending_writer_grant(grant["grant_id"], nonce=grant["nonce"])
    assert abandoned["state"] == "ABANDONED"
    assert abandoned["liveness_evidence"]["status"] == "DEAD"
    assert budget.pending_writer_grants(coordinator_epoch="epoch-0") == []


def test_bound_child_grant_takeover_refuses_live_and_unknown_then_fences_dead(tmp_path, monkeypatch):
    from KHz_filament import hr4e5_storage as storage_module
    from KHz_filament.hr4e5_storage import StorageBudgetError, StorageIntegrityError

    budget, _reservation, _intent, grant = _child_grant_fixture(tmp_path)
    bound = budget.bind_child_writer_grant(
        grant["grant_id"], nonce=grant["nonce"], scope=grant["scope"],
    )
    assert bound["state"] == "BOUND_ACTIVE"
    with pytest.raises(StorageBudgetError, match="live child writer grants"):
        budget.fence_dead_writer_grants(coordinator_epoch="epoch-0")
    assert budget.active_writers(coordinator_epoch="epoch-0")[0]["grant_state"] == "BOUND_ACTIVE"

    monkeypatch.setattr(storage_module, "probe_process_identity",
                        lambda _identity: {"status": "UNKNOWN", "reason": "test-unknown"})
    with pytest.raises(StorageIntegrityError, match="unverifiable child writer grants"):
        budget.fence_dead_writer_grants(coordinator_epoch="epoch-0")
    assert budget.active_writers(coordinator_epoch="epoch-0")[0]["grant_state"] == "BOUND_ACTIVE"

    monkeypatch.setattr(storage_module, "probe_process_identity",
                        lambda _identity: {"status": "DEAD", "reason": "test-dead"})
    fenced = budget.fence_dead_writer_grants(coordinator_epoch="epoch-0")
    assert fenced["interrupted_stale_grant_ids"] == [grant["grant_id"]]
    assert fenced["probes"][grant["grant_id"]]["status"] == "DEAD"
    assert budget.active_writers(coordinator_epoch="epoch-0") == []


def test_intent_epoch_rebind_requires_quiescent_matching_grants_and_reuses_reservation(tmp_path, monkeypatch):
    from KHz_filament import hr4e5_storage as storage_module
    from KHz_filament.hr4e5_storage import StorageBudgetError, StorageIntegrityError, process_identity

    budget, reservation, intent, grant = _child_grant_fixture(tmp_path, launch_candidate_identity=process_identity())
    rebind = {
        "reservation_id": reservation.reservation_id,
        "previous_coordinator_epoch": "epoch-0",
        "coordinator_epoch": "epoch-1",
        "trajectory": "C",
        "pulse": 0,
        "logical_attempt": 0,
        "generation": "current:g0",
        "allowed_paths": [budget.root / "pulse0" / "optical"],
        "expected_bytes": 4096,
        "admission_identity_sha256": "a" * 64,
        "scientific_pair_id": "c" * 64,
        "implementation_identity": {"schema": "test-implementation.v1", "identity_sha256": "b" * 64},
    }
    with pytest.raises(StorageBudgetError, match="unresolved child writer grants"):
        budget.rebind_intent_epoch(intent["intent_id"], **rebind)

    monkeypatch.setattr(storage_module, "probe_process_identity",
                        lambda _identity: {"status": "DEAD", "reason": "test-dead"})
    abandoned = budget.abandon_pending_writer_grant(grant["grant_id"], nonce=grant["nonce"])
    assert abandoned["state"] == "ABANDONED"
    rebound = budget.rebind_intent_epoch(intent["intent_id"], **rebind)
    assert rebound["intent_id"] == intent["intent_id"]
    assert rebound["epoch"] == "epoch-1"
    assert rebound["takeover_from_epoch"] == "epoch-0"
    assert len(budget.reservations()) == 1
    assert budget.reservations()[reservation.reservation_id]["status"] == "ACTIVE"

    replay_grant = budget.create_child_writer_grant(
        reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
        coordinator_epoch="epoch-1", role="HYDRO[0]", trajectory="C", pulse=0,
        logical_attempt=0, generation="current:g0", scientific_pair_id="c" * 64,
        admission_identity_sha256="a" * 64,
        implementation_identity={"schema": "test-implementation.v1", "identity_sha256": "b" * 64},
        allowed_paths=[budget.root / "pulse0" / "optical"], expected_bytes=4096,
        execution_attempt=1,
    )
    assert replay_grant["scope"]["role_label"] == "HYDRO[0]"
    assert len(budget.reservations()) == 1

    with pytest.raises(StorageIntegrityError, match="scope differs from original intent"):
        budget.rebind_intent_epoch(
            intent["intent_id"], **{**rebind, "previous_coordinator_epoch": "epoch-1",
                                      "coordinator_epoch": "epoch-2", "pulse": 1}
        )

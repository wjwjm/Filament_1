"""Targeted real-producer baseline, one fault, and exact recovery checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from .adapter import build_role
from .scientific_context import context, frozen_tools


def setup_case(tmp_path):
    import numba
    report = Path(os.environ["S43SG_REPORT_ROOT"])
    cases, exact = frozen_tools(report)
    ctx = context(report, shape=(32, 17, 19))
    prop, dispatch, clock, binding, audit, lease = build_role("G3_SMALL", "PURE_OFF_SMALLGRID", capture_root=tmp_path / "capture", local_engineering=True, pool_override=1)
    state = dispatch.cpu_s4_3g_state

    def run(label):
        directory = tmp_path / label
        field, kwargs, view = cases.prepare(ctx, directory, intervals=2)
        try:
            result = prop(field, **kwargs)
            return cases.finish(result, view, directory, intervals=2)
        finally:
            view.close()

    baseline = run("baseline")
    oracle = exact.save_tree(tmp_path / "oracle", baseline, compressed=True)
    previous = np.array(baseline["returned"][0], copy=True)

    def recover(label):
        actual = run(label)
        comparison = exact.compare(Path(oracle), actual, tmp_path / (label + ".jsonl"), disk=True)
        assert comparison["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        assert np.array_equal(baseline["returned"][0].view(np.uint64), previous.view(np.uint64))
        assert not state._g_propagation_active and state._g_pending_cap is None
        assert getattr(state._local, "context", None) is None
        assert not state.metrics()["active_lifecycle_identity"]
        assert numba.get_num_threads() == 1
        return comparison

    return ctx, prop, state, run, recover


@pytest.mark.parametrize("fault,reason", (
    ("owner", "LIFETIME_IDENTITY_OWNER_INVALID"),
    ("capability", "LIFETIME_IDENTITY_CAPABILITY_OWNER_INVALID"),
    ("generation", "LIFETIME_IDENTITY_HEUN_GENERATION_MISMATCH"),
    ("scope", "LIFETIME_IDENTITY_SCOPE_MISMATCH"),
    ("consumed", "LIFETIME_IDENTITY_SCHEMA_OR_CONSUMPTION_INVALID"),
    ("schema", "LIFETIME_IDENTITY_TOKEN_SCHEMA_OR_METADATA_CHANGED"),
    ("legacy_member", "LIFETIME_IDENTITY_TOKEN_SCHEMA_OR_METADATA_CHANGED"),
    ("across_half", "LIFETIME_IDENTITY_SCHEMA_OR_CONSUMPTION_INVALID"),
    ("across_z", "LIFETIME_IDENTITY_SCHEMA_OR_CONSUMPTION_INVALID"),
    ("Omega_content", "LIFETIME_IDENTITY_PARAMETER_MISMATCH"),
    ("stage_exception", "SG_INJECTED_STAGE1"),
    ("stage2_exception", "SG_INJECTED_STAGE2"),
    ("field_object", "LIFETIME_IDENTITY_FIELD_OBJECT_MISMATCH"),
))
def test_smallgrid_real_fault_retire_and_exact_recovery(fault, reason, tmp_path, monkeypatch):
    ctx, prop, state, run, recover = setup_case(tmp_path)
    original = state._internal_stage
    stored = []
    count = [0]
    omega = None
    omega_bytes = None

    def stage(call_context, current, params, *, stage2):
        nonlocal omega, omega_bytes
        if not stage2 and fault == "stage_exception":
            raise RuntimeError("SG_INJECTED_STAGE1")
        if stage2 and fault == "stage2_exception":
            raise RuntimeError("SG_INJECTED_STAGE2")
        result = original(call_context, current, params, stage2=stage2)
        if stage2:
            return result
        count[0] += 1
        token = call_context.field_token
        identity = token["lifetime_identity"]
        if fault.startswith("across_"):
            if not stored:
                stored.append(token)
            elif count[0] == (2 if fault == "across_half" else 3):
                call_context.field_token = stored[0]
            return result
        if count[0] != 1:
            return result
        if fault == "owner": identity.owner = object()
        elif fault == "capability": identity.capability = object()
        elif fault == "generation": identity.heun_generation -= 1
        elif fault == "scope": identity.scope = "post"
        elif fault == "consumed": identity.consumed = True
        elif fault == "field_object": identity.field_object_id += 1
        elif fault == "schema": token["identity_kind"] = "legacy_content_hash_v1"
        elif fault == "legacy_member": token["content_raw_sha256"] = "unexpected_legacy_member"
        elif fault == "Omega_content":
            omega = params["Omega"]
            omega_bytes = omega.copy()
            omega[0] += 1.0
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(state, "_internal_stage", stage)
            with pytest.raises((RuntimeError, ValueError), match=reason) as rejected:
                run("specified_fault")
    finally:
        if omega is not None:
            omega[:] = omega_bytes
    assert not state._g_propagation_active and state._g_pending_cap is None
    assert getattr(state._local, "context", None) is None
    assert not state.metrics()["active_lifecycle_identity"]
    recovery = recover("recovered")
    (tmp_path / "FAULT_RECEIPT.json").write_text(json.dumps(dict(fault=fault, baseline="PASS_REAL_PRODUCER", refusal=str(rejected.value), recovery=recovery, real_os_thread=threading.get_ident(), mask_restored=True), indent=2) + "\n", encoding="utf-8")


def test_smallgrid_real_os_thread_reject_recover(tmp_path):
    ctx, prop, state, run, recover = setup_case(tmp_path)
    errors = []
    thread_ids = []
    def other_thread():
        thread_ids.append(threading.get_ident())
        try:
            run("wrong_actual_thread")
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive() and len(errors) == 1 and "THREAD" in str(errors[0])
    assert thread_ids[0] != threading.get_ident()
    recover("thread_recovered")


def test_smallgrid_real_nested_reject_keeps_outer_lease(tmp_path, monkeypatch):
    ctx, prop, state, run, recover = setup_case(tmp_path)
    original = state._internal_stage
    nested = []
    def stage(call_context, field, params, *, stage2):
        result = original(call_context, field, params, stage2=stage2)
        if not stage2 and not nested:
            identity = call_context.field_token["lifetime_identity"]
            with pytest.raises(RuntimeError, match="CAPABILITY_REPLAY_OR_RETIRED") as rejected:
                state._g_controlled_full_heun(field, 1e-5, params, state._g_pending_cap)
            nested.append(str(rejected.value))
            assert identity.valid and state._g_lifecycle_identity is identity
        return result
    with monkeypatch.context() as patch:
        patch.setattr(state, "_internal_stage", stage)
        recover("outer_completes_after_nested_refusal")
    assert len(nested) == 1
    recover("nested_recovered")


def test_smallgrid_real_stale_predictor_reject_recover(tmp_path, monkeypatch):
    ctx, prop, state, run, recover = setup_case(tmp_path)
    state._g_set_qualification_capture(True)
    run("real_predictor_baseline")
    captures = state._g_take_qualification_capture()
    old_predictor = captures[0]["predictor"]
    state._g_set_qualification_capture(False)
    namespace = state._internal_apply_inner.__globals__
    original = namespace["_s43c_predictor"]
    invoked = []
    def stale(*args, **kwargs):
        invoked.append(True)
        output = original(*args, **kwargs)
        state._local.context.predictor = old_predictor
        return output
    with monkeypatch.context() as patch:
        patch.setitem(namespace, "_s43c_predictor", stale)
        with pytest.raises(RuntimeError, match="PREDICTOR_BINDING_MISMATCH"):
            run("stale_predictor")
    assert invoked
    recover("predictor_recovered")


def test_smallgrid_real_role_schema_reject_recover(tmp_path, monkeypatch):
    ctx, prop, state, run, recover = setup_case(tmp_path)
    original = state._set_controlled_hash_stats
    injected = []
    def wrong_source_role():
        injected.append(True)
        state._hash_records[0]["source_role"] = "E"
        return original()
    with monkeypatch.context() as patch:
        patch.setattr(state, "_set_controlled_hash_stats", wrong_source_role)
        with pytest.raises(RuntimeError, match="CONTROLLED_HASH_SOURCE_ROLE_MISMATCH"):
            run("mixed_hash_role_schema")
    assert injected
    recover("role_schema_recovered")


def test_smallgrid_optimized_identity_explicit_rejection(tmp_path):
    from .candidate import source_identity
    source_identity()
    command = [sys.executable, "-s", "-B", "-O", "-c", "from tools.cpu_s4_3g3_smallgrid.candidate import source_identity; source_identity()"]
    flag = subprocess.run(command, capture_output=True, env=os.environ.copy())
    assert flag.returncode != 0 and b"SMALLGRID_OPTIMIZATION_FORBIDDEN" in flag.stderr
    env = os.environ.copy()
    env["PYTHONOPTIMIZE"] = "1"
    variable = subprocess.run([sys.executable, "-s", "-B", "-c", command[-1]], capture_output=True, env=env)
    assert variable.returncode != 0 and b"SMALLGRID_OPTIMIZATION_FORBIDDEN" in variable.stderr
    source_identity()
    (tmp_path / "OPTIMIZATION_FLAG.stderr").write_bytes(flag.stderr)
    (tmp_path / "OPTIMIZATION_ENV.stderr").write_bytes(variable.stderr)

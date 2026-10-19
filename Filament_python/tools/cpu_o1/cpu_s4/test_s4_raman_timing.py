from __future__ import annotations

import ast
import gc
import inspect
from pathlib import Path
from types import FunctionType
import weakref

import numpy as np
import pytest

import KHz_filament.raman as raman
from tools.cpu_o1.cpu_s4.profile import NestedTimer
from tools.cpu_o1.cpu_s4.timing_instrumentation import (
    InstrumentationError,
    _function_node,
    _target_matches,
    instrument_active_raman_function,
)


def _private_function(name: str):
    original = getattr(raman, name)
    namespace = dict(raman.__dict__)
    private = FunctionType(original.__code__, namespace, name,
                           original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    private.__annotations__ = dict(original.__annotations__)
    namespace[name] = private
    return private, namespace


def test_heun_corrector_anchor_excludes_complex64_projection_assignment():
    function, _ = _private_function("apply_isaacs_raman_operator_step")
    source = inspect.getsourcefile(function)
    tree = ast.parse(Path(source).read_text(encoding="utf-8"), filename=source)
    node = _function_node(tree, function)
    matches = [statement for statement in ast.walk(node)
               if isinstance(statement, ast.stmt)
               and _target_matches(function.__name__, "result", statement)]

    assert len(matches) == 1
    assert isinstance(matches[0], ast.Assign)
    assert len(matches[0].targets) == 1
    assert isinstance(matches[0].targets[0], ast.Name)
    assert matches[0].targets[0].id == "result"
    assert isinstance(matches[0].value, ast.Call)
    assert isinstance(matches[0].value.func, ast.Attribute)
    assert matches[0].value.func.attr == "astype"
    assert any(isinstance(part, ast.Subscript)
               and isinstance(part.value, ast.Name)
               and part.value.id == "stage2" for part in ast.walk(matches[0].value))
    complete, _ = _private_function("isaacs_complete_eq27_stage")
    with pytest.raises(InstrumentationError, match="only source-audited"):
        instrument_active_raman_function(complete, NestedTimer())


def test_heun_profile_reuses_stage1_and_recomputes_stage2_from_predictor():
    function, namespace = _private_function("apply_isaacs_raman_operator_step")
    calls = []

    def fake_stage(field, **kwargs):
        calls.append(np.array(field, copy=True))
        return {"rhs": np.full_like(field, 3.0 + 1.0j),
                "q_R_positive": np.zeros(field.shape[1:], dtype=np.float64)}

    namespace["isaacs_raman_stage"] = fake_stage
    timer = NestedTimer(array_threshold_bytes=1)
    profiled, binding = instrument_active_raman_function(function, timer)
    field = np.full((2, 3, 4), 2.0 + 0.5j, dtype=np.complex128)
    k1 = np.full_like(field, 0.25 - 0.5j)
    stage1 = {"rhs": k1,
              "q_R_positive": np.zeros(field.shape[1:], dtype=np.float64)}
    dz = 0.125

    actual = profiled(field, dz, Omega=np.ones(2), dt=1.0, omega0=1.0,
                      n0=1.0, n_R=1.0, omega_R=1.0, Gamma_R=1.0,
                      stage1=stage1, integrator="heun")

    expected_predictor = field + dz * k1
    expected = (field + 0.5 * dz * (k1 + (3.0 + 1.0j))).astype(
        field.dtype, copy=False)
    assert np.array_equal(actual, expected)
    assert len(calls) == 1
    assert np.array_equal(calls[0], expected_predictor)
    assert binding["function"] == "apply_isaacs_raman_operator_step"
    rows = {row["label"]: row for row in timer.report()["records"]}
    assert rows["raman.heun.predictor_update"]["calls"] == 1
    assert rows["raman.heun.stage2_recomputed"]["calls"] == 1
    assert rows["raman.heun.corrector_update"]["calls"] == 1


def test_stable_energy_float64_casts_report_copy_and_view_storage():
    expected = {
        "raman.energy.before_real.float64_cast",
        "raman.energy.after_real.float64_cast",
        "raman.energy.before_imag.float64_cast",
        "raman.energy.after_imag.float64_cast",
    }
    for dtype, expected_storage in (
            (np.complex64, "distinct_output_storage_observed"),
            (np.complex128, "shared_or_view_storage_observed")):
        function, _ = _private_function("stable_field_fluence_and_loss")
        timer = NestedTimer(array_threshold_bytes=1)
        profiled, binding = instrument_active_raman_function(function, timer)
        before = np.ones((5, 3, 4), dtype=dtype) * (1.0 + 2.0j)
        after = np.ones_like(before) * (0.75 + 1.5j)

        profiled(before, after, dt=1e-15, n0=1e25, chunk_t=2)

        report = timer.report()
        assert binding["function"] == "stable_field_fluence_and_loss"
        assert {row["label"] for row in report["records"]} >= expected
        operations = report["array_accounting"]["operations"]
        assert {row["operation"] for row in operations} >= expected
        for operation in expected:
            rows = [row for row in operations if row["operation"] == operation]
            assert sum(row["calls"] for row in rows) == 3
            assert {row["storage_observation"] for row in rows} == {expected_storage}
            assert {row["timing_scope"] for row in rows} == {"whole_statement"}
            assert {row["aggregation_warning"] for row in rows} == {
                "alternative_view_do_not_sum_with_nested_timers"}
            for row in rows:
                assert row["inclusive_wall_seconds"] >= row["exclusive_wall_seconds"] >= 0
                assert row["inclusive_process_cpu_seconds"] >= row["exclusive_process_cpu_seconds"] >= 0
            assert all(array["dtype"] == "float64"
                       for row in rows for array in row["output_arrays"])


@pytest.mark.parametrize("fail_inside", [False, True])
def test_timer_releases_array_references_after_scope_even_on_exception(fail_inside):
    timer = NestedTimer(array_threshold_bytes=1)
    scope = timer.statement("weakref_fixture", callsite="test:1:0",
                            stage_role="test", parent_stage="test")
    source = np.arange(32, dtype=np.float64)
    output = source[::2]
    source_ref, output_ref = weakref.ref(source), weakref.ref(output)

    if fail_inside:
        with pytest.raises(LookupError, match="fixture"):
            with scope:
                scope.observe_arrays(operation="test.view", inputs=(source,),
                                     output=output, kind="view")
                del source, output
                raise LookupError("fixture")
    else:
        with scope:
            scope.observe_arrays(operation="test.view", inputs=(source,),
                                 output=output, kind="view")
            del source, output

    gc.collect()
    assert scope._operations == []
    assert source_ref() is None
    assert output_ref() is None


def test_storage_sharing_failure_is_reported_as_indeterminate(monkeypatch):
    def cannot_determine(*args, **kwargs):
        raise RuntimeError("bounded sharing check failed")

    monkeypatch.setattr(np, "shares_memory", cannot_determine)
    timer = NestedTimer(array_threshold_bytes=1)
    source = np.arange(16, dtype=np.float64)
    output = source[::2]
    scope = timer.statement("sharing_fixture", callsite="test:2:0",
                            stage_role="test", parent_stage="test")
    with scope:
        scope.observe_arrays(operation="test.unknown", inputs=(source,),
                             output=output, kind="view")

    row = timer.report()["array_accounting"]["operations"][0]
    assert row["storage_observation"] == "storage_sharing_indeterminate"


def test_s4_timing_extended_tiny_contract(tmp_path):
    import json
    from tools.cpu_o1.cpu_s4.profile import run_validated_context_pair, _validate_profile_receipt
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    context = j1._tiny_context(operator_mode="full_isaacs_eq27")
    context["_s4_array_threshold_bytes"] = 1
    context["source_hash"] = j1._canonical_hash(context["source"])
    context["assets"] = {"pre0": context["pre0"],
        "pre0_canonical_sha256": j1._canonical_hash(context["pre0"])}
    source_before = j1._canonical_hash(context["source"])
    pre_before = {key: j1._canonical_hash(value)
                  for key, value in context["pre_fields"].items()}

    result = run_validated_context_pair(context, tmp_path / "s4-tiny")

    assert result["status"] == "LOCAL_PASS"
    summary = json.loads((tmp_path / "s4-tiny" / "S4_PROFILE_SUMMARY.json").read_text())
    assert summary["production_qualification"] == "NOT_PRODUCTION_QUALIFIED"
    assert summary["hpc_submission"] == "NOT_SUBMITTED"
    assert summary["comparison_gate"]["status"] == "EXACT"
    assert summary["comparison_gate"]["raw_row_count"] == 235
    assert summary["modes"]["off"]["candidate_event_count"] > 0
    assert summary["modes"]["profile"]["candidate_event_count"] > 0
    assert len(summary["modes"]["off"]["interval_rows"]) == 8
    assert len(summary["modes"]["profile"]["interval_rows"]) == 8
    assert all(r["profiled_call_count"] > 0 for r in summary["modes"]["profile"]["interval_rows"])
    profile = json.loads((tmp_path / "s4-tiny" / "profile" / "propagation" /
                          "function_fft_profile.json").read_text())
    assert profile["mode"] == "PROFILE"
    assert profile["nested_timers"]["calls"] > 0
    assert sum(r["calls"] for r in profile["nested_timers"]["records"]
               if r["label"] == "ionization.full_time_rk4_candidate") == summary["modes"]["profile"]["candidate_event_count"]
    timing_contract = profile["timing_contract"]
    assert timing_contract["raman_mode"] == "full_isaacs_eq27"
    assert timing_contract["active_path"] == [
        "isaacs_raman_stage", "apply_isaacs_raman_operator_step"]
    assert timing_contract["inactive_complete_path"] == (
        "isaacs_complete_eq27_stage_not_instrumented")
    assert "isaacs_complete_eq27_stage" not in timing_contract["active_function_call_counts"]
    assert timing_contract["inactive_complete_runtime_count"] is None
    assert timing_contract["inactive_complete_evidence"] == "excluded_by_frozen_mode_and_source_not_timed"
    assert timing_contract["active_support_function_call_counts"][
        "stable_field_fluence_and_loss"] > 0
    assert timing_contract["stage_call_roles"]["stage1_supplied"] > 0
    assert timing_contract["stage_call_roles"]["stage2_recomputed"] > 0
    assert {row["function"] for row in timing_contract["source_bindings"]} == {
        "isaacs_raman_stage", "apply_isaacs_raman_operator_step",
        "stable_field_fluence_and_loss"}
    labels = {row["label"] for row in profile["nested_timers"]["records"]
              if row["calls"] > 0}
    assert {"raman.heun.predictor_update", "raman.heun.stage2_recomputed",
            "raman.heun.corrector_update",
            "raman.energy.before_real.float64_cast",
            "raman.energy.after_real.float64_cast",
            "raman.energy.before_imag.float64_cast",
            "raman.energy.after_imag.float64_cast"} <= labels
    array_operations = profile["nested_timers"]["array_accounting"]["operations"]
    assert {row["operation"] for row in array_operations} >= {
        "raman.energy.before_real.float64_cast",
        "raman.energy.after_real.float64_cast",
        "raman.energy.before_imag.float64_cast",
        "raman.energy.after_imag.float64_cast"}
    assert profile["global_science_modules_modified"] is False
    assert profile["full_arrays_retained"] is False
    assert j1._canonical_hash(context["source"]) == source_before
    assert {key: j1._canonical_hash(value) for key, value in context["pre_fields"].items()} == pre_before

    workspace_rows = [row for row in array_operations
                      if row['operation'] == 'ionization.rate_workspace.acquire']
    assert sum(row['calls'] for row in workspace_rows) == summary['modes']['profile']['candidate_event_count']
    assert {row['storage_observation'] for row in workspace_rows} == {'allocated', 'reused'}
    assert sum(row['calls'] for row in workspace_rows if row['storage_observation']=='allocated') == 1
    import copy
    for missing in ('timing_contract',):
        broken = copy.deepcopy(profile); broken.pop(missing)
        with pytest.raises(RuntimeError):
            _validate_profile_receipt(broken)
    broken = copy.deepcopy(profile)
    broken['timing_contract']['raman_mode'] = 'full_isaacs_eq27_complete'
    with pytest.raises(RuntimeError):
        _validate_profile_receipt(broken)


def test_s4_timing_rejects_global_module_and_source_drift():
    from tools.cpu_o1.cpu_s4.timing_instrumentation import _source_binding
    with pytest.raises(InstrumentationError, match='global science-module'):
        instrument_active_raman_function(raman.isaacs_raman_stage, NestedTimer())
    function, namespace = _private_function('isaacs_raman_stage')
    path = Path(inspect.getsourcefile(function)).resolve()
    original = namespace['isaacs_raman_stage']
    with pytest.raises(InstrumentationError, match='frozen P1-R2 blob'):
        _source_binding(function,path,path.read_bytes()+b'\n# drift\n',[])
    assert namespace['isaacs_raman_stage'] is original


def test_s4_timing_ast_restoration_guard(monkeypatch):
    from tools.cpu_o1.cpu_s4 import timing_instrumentation as module
    function, namespace = _private_function('isaacs_raman_stage')
    strip = module.strip_s4_timing_nodes
    def corrupt(node):
        restored = strip(node)
        restored.body.append(ast.Pass())
        return restored
    monkeypatch.setattr(module,'strip_s4_timing_nodes',corrupt)
    with pytest.raises(InstrumentationError,match='stripped PROFILE AST differs'):
        module.instrument_active_raman_function(function,NestedTimer())
    assert namespace['isaacs_raman_stage'] is function

@pytest.mark.parametrize('offset',[-1000.,1000.])
def test_s4_timing_supervisor_uses_observer_clock(tmp_path,monkeypatch,offset):
    import os,sys,time
    from tools.cpu_o1.cpu_s4 import supervisor
    real_counter=time.perf_counter
    monkeypatch.setattr(supervisor.time,'monotonic',lambda:real_counter()+offset)
    monkeypatch.setattr(supervisor,'progress_state',lambda path:(real_counter(),set(range(8))))
    monkeypatch.setattr(supervisor,'resource_snapshot',lambda pid:{'available':False})
    row=supervisor.supervise_child([sys.executable,'-s','-B','-c','import time;time.sleep(.05)'],
        tmp_path/'case',env=os.environ.copy(),sample_seconds=.01,progress_clock='perf_counter')
    assert row['status']=='COMPLETE',row
    assert row['progress_clock']=='perf_counter'

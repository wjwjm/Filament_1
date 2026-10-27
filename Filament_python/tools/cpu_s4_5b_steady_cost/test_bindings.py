"""AST/source-binding checks that do not run a propagation."""

from __future__ import annotations

import ast

import pytest

from .adapter import _merge_snapshots
from .instrument import (
    INSTRUMENT_SCHEMA,
    instrument_eq10_stage,
    instrument_propagation_ast,
    strip_generated_nodes,
    _same_fixed_backend_callable,
)


def test_fixed_fresh_closure_identity_and_rejection() -> None:
    fixed = object()
    def make(value):
        def function():return value
        return function
    first,second=make(fixed),make(fixed)
    assert first is not second and _same_fixed_backend_callable(first,second)
    assert not _same_fixed_backend_callable(first,make(object()))
    second.__code__=second.__code__.replace(co_name='wrong_backend')
    assert not _same_fixed_backend_callable(first,second)


def _propagation_fixture() -> ast.FunctionDef:
    source = """
def propagate_one_pulse(schedule):
    setup = 1
    for interval_index, interval in enumerate(schedule.intervals):
        step_started = 0
        if use_uppe:
            linear_result = 1
        prep = 1
        operator_energy_after_nonraman = 0
        energy = 1
        operator_energy_after_raman_post = 0
        if post_commit_hook is not None:
            post_commit_hook()
        operator_energy_after_linear_half2 = 0
        save_count += 1
        if (save_count % save_every) == 0 or interval_index == 7:
            saved = 1
        z += dz_try
    final = 1
    return final
"""
    return ast.parse(source).body[0]


def _eq10_fixture() -> ast.FunctionDef:
    source = """
def isaacs_raman_stage(intensity, response, Omega, rdtype, dt, n_R, c0, result, return_energy):
    if return_energy:
        intensity_scale = float(xp.max(xp.abs(intensity)))
        if intensity_scale > 0.0:
            derivative_increment_norm = xp.fft.ifft(
                (1j * xp.asarray(Omega, dtype=rdtype) * float(dt))[:, None, None]
                * xp.fft.fft(intensity, axis=0), axis=0).real / intensity_scale
            response_norm = response / intensity_scale
            energy_scale = (float(n_R) / float(c0)) * intensity_scale * intensity_scale
            u_signed = energy_scale * xp.sum(response_norm * derivative_increment_norm, axis=0)
        else:
            u_signed = xp.zeros(intensity.shape[1:], dtype=rdtype)
        result["u_R_signed"] = u_signed.astype(rdtype, copy=False)
        result["q_R_positive"] = xp.maximum(-u_signed, 0.0).astype(rdtype, copy=False)
    return result
"""
    return ast.parse(source).body[0]


def test_propagation_scope_strip_roundtrip() -> None:
    function = _propagation_fixture()
    baseline = ast.dump(function, include_attributes=False)
    receipt = instrument_propagation_ast(function)
    assert receipt["schema"].startswith(INSTRUMENT_SCHEMA)
    assert receipt["anchor_counts"]["interval_for"] == 1
    assert {row["label"] for row in receipt["scope_sites"]} == {
        "PROP_ENTRY_INIT",
        "INTERVAL_BODY",
        "INTERVAL_PREP_AND_OTHER_ARRAYS",
        "INTERVAL_ENERGY_BOOKKEEPING",
        "INTERVAL_DEPOSITION_AND_QACC",
        "INTERVAL_POST_COMMIT",
        "INTERVAL_SAVED_DIAGNOSTICS",
        "PROP_FINALIZE_EXIT",
    }
    assert ast.dump(strip_generated_nodes(function), include_attributes=False) == baseline


def test_eq10_scope_strip_roundtrip() -> None:
    function = _eq10_fixture()
    baseline = ast.dump(function, include_attributes=False)
    receipt = instrument_eq10_stage(function)
    assert receipt["fft_calls"] == 1
    assert receipt["ifft_calls"] == 1
    assert receipt["sum_calls"] == 1
    assert receipt["zero_markers"] == 1
    assert ast.dump(strip_generated_nodes(function), include_attributes=False) == baseline


def _snapshot(owner: int, records: list[dict], *, profile: bool = True) -> dict:
    return {
        "schema": "synthetic",
        "profile": profile,
        "enabled": profile,
        "owner_thread_id": owner,
        "clock_domain": "perf_counter_ns/process_time_ns",
        "events_dropped": 0,
        "incomplete_scopes": [],
        "records": records,
    }


def _row(token: int, start: int, end: int, parent: int | None = None) -> dict:
    return {
        "token": token,
        "label": "APPLICATION",
        "context": {
            "interval_index": 0,
            "ordinal": 0,
            "ordinal_1based": 1,
            "attempt": 1,
            "phase": "none",
            "stage": "none",
            "occurrence_class": "WINDOW_ONCE",
            "source_site": "TEST@1",
        },
        "parent_token": parent,
        "start_wall_ns": start,
        "end_wall_ns": end,
        "start_cpu_ns": start,
        "end_cpu_ns": end,
        "failed": False,
        "status": "complete",
    }


def test_merge_preserves_raw_parent_and_recomputes_exclusive() -> None:
    merged = _merge_snapshots(
        _snapshot(7, [_row(1, 10, 20)]),
        _snapshot(7, [_row(1, 0, 40)]),
        profile=True,
    )
    parent = next(row for row in merged["records"] if row["origin"] == "A")
    child = next(row for row in merged["records"] if row["origin"] == "B")
    assert child["raw_parent_token"] is None
    assert child["merged_parent_token"] == parent["merged_token"]
    assert parent["wall_exclusive_ns"] == 30
    assert parent["cpu_exclusive_ns"] == 30
    assert merged["all_raw_parent_edges_preserved"] is True


def test_merge_rejects_crossing_owner_and_equal_spans() -> None:
    with pytest.raises(RuntimeError, match="CROSSING"):
        _merge_snapshots(
            _snapshot(7, [_row(1, 0, 30)]),
            _snapshot(7, [_row(1, 20, 40)]),
            profile=True,
        )
    with pytest.raises(RuntimeError, match="OWNER"):
        _merge_snapshots(
            _snapshot(7, [_row(1, 0, 10)]),
            _snapshot(8, [_row(1, 20, 30)]),
            profile=True,
        )
    with pytest.raises(RuntimeError, match="EQUAL"):
        _merge_snapshots(
            _snapshot(7, [_row(1, 0, 10)]),
            _snapshot(7, [_row(1, 0, 10)]),
            profile=True,
        )


def test_actual_body_ordinal_overrides_only_normalized_old_counter() -> None:
    body = _row(1, 0, 100)
    body['label'] = 'INTERVAL_BODY'
    body['context'].update(interval_index=3, ordinal=3, ordinal_1based=4)
    old = _row(1, 20, 40)
    old['context'] = dict(interval=99, phase='post', stage='stage2')
    merged = _merge_snapshots(_snapshot(7, [body]), _snapshot(7, [old]), profile=True)
    actual = next(row for row in merged['records'] if row['origin'] == 'A')
    assert actual['raw_context'] == old['context']
    assert actual['normalized_interval']['interval_index'] == 3
    assert actual['normalized_interval']['ordinal_1based'] == 4


def test_merge_accepts_zero_cpu_and_interleaved_raw_parent_ancestor() -> None:
    merged = _merge_snapshots(
        _snapshot(7, [_row(1, 10, 90)]),
        _snapshot(7, [_row(1, 0, 100), _row(2, 20, 30, parent=1)]),
        profile=True,
    )
    a_parent = next(
        row for row in merged["records"] if row["origin"] == "A" and row["raw_token"] == 1
    )
    b_middle = next(row for row in merged["records"] if row["origin"] == "B")
    a_child = next(
        row for row in merged["records"] if row["origin"] == "A" and row["raw_token"] == 2
    )
    assert a_parent["merged_parent_token"] is None
    assert b_middle["merged_parent_token"] == a_parent["merged_token"]
    assert a_child["merged_parent_token"] == b_middle["merged_token"]
    assert a_child["raw_parent_token"] == 1
    assert a_parent["wall_exclusive_ns"] == 20
    assert b_middle["wall_exclusive_ns"] == 70
    assert a_child["wall_exclusive_ns"] == 10
    assert all(row["cpu_exclusive_ns"] >= 0 for row in merged["records"])

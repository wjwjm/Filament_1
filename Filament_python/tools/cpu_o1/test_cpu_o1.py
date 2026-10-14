from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.cpu_o1.cpu_o1 import (
    CPUExecutionPlan,
    CPUExecutor,
    WorkspaceStamp,
    available_numba_threads,
    compare_numeric,
    compare_numeric_bounded,
    evolve_rho_time_cpu_o1,
    local_environment_fingerprint,
    make_joint_propagator_factory,
    workspace_matches,
)
from tools.cpu_o1.numeric_gate import build_numeric_gate, classify_comparison
from tools.cpu_o1 import q1_t0
from tools.cpu_o1.q1_t0 import _array_comparison_row, main as q1_t0_main


class _ToyWfunc:
    def __init__(self, *, integrator="rk4", time_mode="full", high_rate=False):
        self._integrator = integrator
        self._time_mode = time_mode
        if high_rate:
            rate_a = lambda value: np.full(value.shape, 8.0, dtype=np.float64)
            rate_b = lambda value: np.full(value.shape, 11.0, dtype=np.float64)
        else:
            def rate_a(value):
                temporal = np.arange(value.shape[0], dtype=np.float64)[:, None, None]
                return 1.4 + 0.08 * np.abs(value) + 0.03 * temporal

            def rate_b(value):
                return 0.55 + 0.04 * np.abs(value)

        self._species_entries = (
            {
                "name": "toy-a", "fraction": 0.65, "Ui_J": 0.4,
                "W_s": rate_a, "W_runtime": rate_a,
            },
            {
                "name": "toy-b", "fraction": 0.35, "Ui_J": 1.3,
                "W_s": rate_b, "W_runtime": rate_b,
            },
        )


def _tiny_field(nt=9, ny=3, nx=5):
    values = np.linspace(0.05, 1.2, nt * ny * nx, dtype=np.float64).reshape(nt, ny, nx)
    return (values + 0.2j * values[::-1]).astype(np.complex128)


def _args(field, *, high_rate=False):
    return {
        "input_array": field,
        "dt": 1.0 if high_rate else 0.2,
        "N0": 20.0 if high_rate else 25.0,
        "beta_rec": 0.01 if high_rate else 0.04,
        "Wfunc": _ToyWfunc(high_rate=high_rate),
        "return_species_terms": True,
        "return_species_densities": True,
        "diagnose_integrator_stability": True,
    }


def _assert_exact_tree(reference, candidate, path="root"):
    if isinstance(reference, np.ndarray):
        metric = compare_numeric(reference, candidate)
        assert metric["canonical_exact"], f"{path}: {metric['max_abs']=}"
        assert metric["bitwise_equal"], f"{path}: dtype/bit pattern differs"
        assert reference.dtype == candidate.dtype
        return
    if isinstance(reference, dict):
        assert list(reference) == list(candidate), path
        for key in reference:
            _assert_exact_tree(reference[key], candidate[key], f"{path}.{key}")
        return
    if isinstance(reference, tuple):
        assert len(reference) == len(candidate), path
        for index, (left, right) in enumerate(zip(reference, candidate)):
            _assert_exact_tree(left, right, f"{path}[{index}]")
        return
    assert reference == candidate, path


def _assert_contract_tree(reference, candidate, path="root"):
    """Enforce the accepted fixed-environment exact return contract."""
    if isinstance(reference, np.ndarray):
        assert isinstance(candidate, np.ndarray), path
        assert reference.shape == candidate.shape, path
        assert reference.dtype == candidate.dtype, path
        metric = compare_numeric(reference, candidate)
        assert metric["canonical_exact"] and metric["bitwise_equal"], path
        return
    if isinstance(reference, dict):
        assert list(reference) == list(candidate), path
        for key in reference:
            _assert_contract_tree(reference[key], candidate[key], f"{path}.{key}")
        return
    if isinstance(reference, tuple):
        assert len(reference) == len(candidate), path
        for index, (left, right) in enumerate(zip(reference, candidate)):
            _assert_contract_tree(left, right, f"{path}[{index}]")
        return
    if isinstance(reference, (bool, int, float, complex, np.number)):
        assert type(reference) is type(candidate), path
        left = np.asarray(reference)
        right = np.asarray(candidate)
        assert left.dtype == right.dtype, path
        assert left.tobytes() == right.tobytes(), path
        return
    assert type(reference) is type(candidate), path
    assert reference == candidate, path


def _persist_metrics(record):
    root = os.environ.get("CPU_O1_EVIDENCE_DIR")
    if not root:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "cpu_o1_local_comparisons.jsonl"
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _scalar_token(value):
    if isinstance(value, np.generic):
        return {
            "type": type(value).__name__,
            "dtype": str(np.asarray(value).dtype),
            "bytes_hex": np.asarray(value).tobytes().hex(),
            "repr": repr(value.item()),
        }
    if isinstance(value, float):
        return {"type": "float", "bytes_hex": np.asarray(value).tobytes().hex(), "repr": repr(value)}
    if isinstance(value, complex):
        return {"type": "complex", "bytes_hex": np.asarray(value).tobytes().hex(), "repr": repr(value)}
    return {"type": type(value).__name__, "value": value}


def _metric_record(reference, candidate, *, label, settings, measurement=None, require_exact=True):
    result = {}
    scalar_metrics = {}
    def visit(ref_value, candidate_value, key):
        if isinstance(ref_value, np.ndarray):
            metric = compare_numeric(ref_value, candidate_value)
            result[key] = _metric_fields(metric)
            if not (metric["canonical_exact"] and metric["bitwise_equal"]):
                mismatches.append(key)
        elif isinstance(ref_value, dict):
            assert list(ref_value) == list(candidate_value), key
            for name, nested in ref_value.items():
                visit(nested, candidate_value[name], f"{key}.{name}")
        elif isinstance(ref_value, tuple):
            assert len(ref_value) == len(candidate_value), key
            for index, (left, right) in enumerate(zip(ref_value, candidate_value)):
                visit(left, right, f"{key}[{index}]")
        elif isinstance(ref_value, (bool, int, float, complex, np.number)):
            exact = type(ref_value) is type(candidate_value)
            if exact:
                left = np.asarray(ref_value)
                right = np.asarray(candidate_value)
                exact = left.dtype == right.dtype and left.tobytes() == right.tobytes()
            scalar_metrics[key] = {
                "reference": _scalar_token(ref_value),
                "candidate": _scalar_token(candidate_value),
                "exact": bool(exact),
            }
            if not exact:
                mismatches.append(key)
        else:
            assert type(ref_value) is type(candidate_value), key
            assert ref_value == candidate_value, key

    mismatches = []
    visit(reference, candidate, "result")
    record = {
        "label": label,
        "settings": settings,
        "measurement": measurement,
        "array_metrics": result,
        "scalar_metrics": scalar_metrics,
        "exact_gate": "required" if require_exact else "diagnostic_only",
        "mismatches": mismatches,
        "numerical_classification": (
            "NUMERIC_DIAGNOSTIC_ONLY / ACCEPTANCE_TOLERANCE_NOT_AUTHORIZED"
            if mismatches else "EXACT_LOCAL_COMPARISON"
        ),
    }
    _persist_metrics(record)
    if require_exact and mismatches:
        raise AssertionError(f"{label}: exact local regression mismatch: {mismatches}")


def _metric_fields(metric):
    fields = {
        name: metric[name]
        for name in (
            "shape", "reference_shape", "candidate_shape",
            "reference_dtype", "candidate_dtype",
            "canonical_exact", "value_equal", "bitwise_equal", "all_finite", "max_abs",
            "reference_canonical_sha256", "candidate_canonical_sha256",
            "reference_max_abs", "normalized_linf", "normalized_linf_reason",
            "relative_l2", "relative_l2_reason",
            "zero_reference_count",
        )
    }
    relative = np.asarray(metric["elementwise_relative_error"], dtype=np.float64)
    fields["elementwise_relative_error"] = [
        float(value) if np.isfinite(value) else None for value in relative.ravel()
    ]
    fields["elementwise_relative_error_shape"] = list(relative.shape)
    fields["zero_reference_mask"] = np.asarray(
        metric["zero_reference_mask"], dtype=bool,
    ).ravel().tolist()
    fields["zero_reference_mask_shape"] = list(relative.shape)
    return fields


def test_exact_regression_environment_is_the_pinned_local_toolchain():
    fingerprint = local_environment_fingerprint()
    assert fingerprint["python"] == "3.12.8"
    assert fingerprint["python_implementation"] == "CPython"
    assert fingerprint["versions"] == {
        "numpy": "2.2.5",
        "scipy": "1.15.2",
        "numba": "0.61.2",
        "llvmlite": "0.44.0",
        "pytest": "8.3.5",
    }
    assert fingerprint["byteorder"] == "little"
    assert fingerprint["exact_regression_qualification"] == {
        "eligible": True,
        "status": "PINNED_WINDOWS_AMD64_GENERIC_WORKQUEUE",
        "reasons": [],
    }


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [("system", "Linux", "platform_not_windows"),
     ("machine", "ARM64", "machine_not_amd64")],
)
def test_exact_regression_qualification_rejects_platform_drift(
    monkeypatch, attribute, value, reason,
):
    import tools.cpu_o1.cpu_o1 as cpu_o1_module

    monkeypatch.setattr(cpu_o1_module.platform, attribute, lambda: value)
    qualification = local_environment_fingerprint()["exact_regression_qualification"]
    assert qualification["eligible"] is False
    assert qualification["status"] == "UNQUALIFIED_ENVIRONMENT"
    assert reason in qualification["reasons"]
    linux = local_environment_fingerprint()["linux_q1_qualification"]
    assert linux["eligible"] is False
    assert "platform_not_linux" not in linux["reasons"] if value == "Linux" else True


def test_exact_regression_qualification_rejects_cpu_target_drift(monkeypatch):
    monkeypatch.setenv("NUMBA_CPU_NAME", "native")
    qualification = local_environment_fingerprint()["exact_regression_qualification"]
    assert qualification["eligible"] is False
    assert qualification["status"] == "UNQUALIFIED_ENVIRONMENT"
    assert "numba_target_not_generic" in qualification["reasons"]


def test_environment_fingerprint_separates_layer_policy_from_parallel_execution():
    runtime = local_environment_fingerprint()["runtime"]
    assert "nested_parallelism_enabled" not in runtime
    assert isinstance(runtime["numba_threading_layer_priority"], list)
    assert runtime["parallel_call_organization"] == "NOT_OBSERVED_BY_ENVIRONMENT_FINGERPRINT"
    assert "one active child" in runtime["parallel_call_policy"]


def test_comparison_tree_rejects_scalar_dtype_and_array_changes():
    reference = {
        "step_clip_count": 0,
        "preclip_step_min": np.float64(-0.0),
        "field": np.asarray([1.0, 2.0], dtype=np.float64),
    }
    with pytest.raises(AssertionError):
        _assert_contract_tree(reference, {**reference, "step_clip_count": 999})
    with pytest.raises(AssertionError):
        _assert_contract_tree(reference, {**reference, "preclip_step_min": np.float64(0.0)})
    with pytest.raises(AssertionError):
        _assert_contract_tree(reference, {**reference, "field": reference["field"].astype(np.float32)})
    changed_field = reference["field"].copy()
    changed_field[1] = 9.0
    with pytest.raises(AssertionError):
        _assert_contract_tree(reference, {**reference, "field": changed_field})


def test_metric_record_captures_scalar_leaves_and_array_dtypes(tmp_path, monkeypatch):
    monkeypatch.setenv("CPU_O1_EVIDENCE_DIR", str(tmp_path))
    reference = {"clip": 2, "extreme": np.float64(-0.25), "field": np.ones((2,), dtype=np.float64)}
    _metric_record(reference, reference, label="scalar_dtype_probe", settings={})
    record = json.loads((tmp_path / "cpu_o1_local_comparisons.jsonl").read_text(encoding="utf-8"))
    assert record["scalar_metrics"]["result.clip"]["exact"]
    assert record["scalar_metrics"]["result.extreme"]["exact"]
    assert record["array_metrics"]["result.field"]["reference_dtype"] == "float64"
    assert record["array_metrics"]["result.field"]["candidate_dtype"] == "float64"
    assert record["array_metrics"]["result.field"]["reference_shape"] == [2]
    assert record["array_metrics"]["result.field"]["candidate_shape"] == [2]


def test_bounded_comparator_streams_and_caps_difference_examples():
    import tracemalloc

    reference = np.arange(4096, dtype=np.float64).reshape(64, 64)
    candidate = reference.copy()
    candidate[0, 0] = 2.0
    candidate[63, 63] += 1.0
    tracing_was_active = tracemalloc.is_tracing()
    metric = compare_numeric_bounded(
        reference, candidate, max_working_bytes=4096, max_examples=1,
    )
    assert tracemalloc.is_tracing() is tracing_was_active
    assert metric["shape"] == [64, 64]
    assert metric["reference_dtype"] == metric["candidate_dtype"] == "float64"
    assert metric["max_abs"] == 2.0
    assert len(metric["worst_differences"]) <= 1
    assert metric["comparison_chunk_limit_bytes"] == 4096
    assert metric["zero_reference_count"] == 1
    assert metric["zero_reference_mask_zlib_base64"]
    assert metric["canonical_hash_traced_memory_scope"] == (
        "tracemalloc-tracked allocations; not process RSS or all native allocations"
    )
    assert metric["canonical_hash_process_rss_scope"] == (
        "whole-process RSS endpoints; not isolated to hash allocations and not a peak"
    )
    if tracing_was_active:
        assert metric["canonical_hash_tracemalloc_status"] == "UNMEASURED_EXTERNAL_TRACEMALLOC_ACTIVE"
        assert metric["canonical_hash_traced_peak_bytes"] is None
        assert metric["canonical_hash_tracemalloc_unmeasured_reason"]
    else:
        assert metric["canonical_hash_tracemalloc_status"] == "MEASURED_ISOLATED_TRACED_PEAK"
        assert isinstance(metric["canonical_hash_traced_peak_bytes"], int)
        assert metric["canonical_hash_traced_peak_bytes"] >= 0
        assert metric["canonical_hash_traced_current_delta_bytes"] >= 0
    if metric["canonical_hash_process_rss_status"] == "ENDPOINTS_SAMPLED_NOT_HASH_PEAK":
        assert isinstance(metric["canonical_hash_process_rss_before_bytes"], int)
        assert isinstance(metric["canonical_hash_process_rss_after_bytes"], int)
        assert metric["canonical_hash_process_rss_delta_bytes"] == (
            metric["canonical_hash_process_rss_after_bytes"]
            - metric["canonical_hash_process_rss_before_bytes"]
        )
    memory_fields = (
        "canonical_hash_seconds", "canonical_hash_input_bytes",
        "canonical_hash_tracemalloc_status", "canonical_hash_tracemalloc_unmeasured_reason",
        "canonical_hash_traced_peak_bytes", "canonical_hash_traced_current_delta_bytes",
        "canonical_hash_traced_memory_scope", "canonical_hash_process_rss_before_bytes",
        "canonical_hash_process_rss_after_bytes", "canonical_hash_process_rss_delta_bytes",
        "canonical_hash_process_rss_status", "canonical_hash_process_rss_unavailable_reason",
        "canonical_hash_process_rss_scope", "comparison_working_memory_scope",
    )
    _persist_metrics({
        "label": "bounded_comparison_memory",
        "settings": {"comparison_chunk_limit_bytes": 4096},
        "measurement": {name: metric[name] for name in memory_fields},
        "comparison": {
            "shape": metric["shape"], "reference_dtype": metric["reference_dtype"],
            "candidate_dtype": metric["candidate_dtype"],
            "canonical_exact": metric["canonical_exact"],
        },
        "exact_gate": "MEMORY_DIAGNOSTIC_ONLY",
    })


def test_bounded_comparator_preserves_external_tracemalloc_state_and_peak():
    import gc
    import tracemalloc

    reference = np.arange(1024, dtype=np.float64)
    candidate = reference.copy()
    tracing_was_active = tracemalloc.is_tracing()
    started_here = not tracing_was_active
    if started_here:
        tracemalloc.start()
    try:
        if started_here:
            prior_peak = bytearray(2 * 1024 * 1024)
            del prior_peak
            gc.collect()
        _, peak_before = tracemalloc.get_traced_memory()
        metric = compare_numeric_bounded(reference, candidate, max_working_bytes=4096)
        _, peak_after = tracemalloc.get_traced_memory()
        assert tracemalloc.is_tracing()
        assert peak_after >= peak_before
        assert metric["canonical_hash_tracemalloc_status"] == "UNMEASURED_EXTERNAL_TRACEMALLOC_ACTIVE"
        assert metric["canonical_hash_traced_peak_bytes"] is None
        assert metric["canonical_hash_tracemalloc_unmeasured_reason"] == (
            "tracemalloc was already active; its caller-owned peak was not reset or read"
        )
    finally:
        if started_here and tracemalloc.is_tracing():
            tracemalloc.stop()
    assert tracemalloc.is_tracing() is tracing_was_active


def test_bounded_comparator_sidecar_roundtrip_for_strided_multichunk_input(tmp_path):
    import hashlib
    import zlib

    base = np.arange(64 * 48, dtype=np.float64).reshape(64, 48)
    base[::7, ::5] = 0.0
    reference = base[::2, 1::3].T
    assert not reference.flags.c_contiguous
    candidate = reference.copy()
    candidate[3, 4] += 0.5
    sidecar = tmp_path / "zero_mask.bin"

    metric = compare_numeric_bounded(
        reference, candidate, max_working_bytes=1024,
        max_examples=128, zero_mask_path=sidecar,
    )
    expected_packed = np.packbits(
        (reference == 0).ravel(order="C"), bitorder="little",
    ).tobytes()
    assert zlib.decompress(sidecar.read_bytes()) == expected_packed
    assert metric["zero_reference_mask_sha256"] == hashlib.sha256(expected_packed).hexdigest()
    assert metric["zero_reference_mask_status"] == "SIDECAR_WRITTEN"
    assert len(metric["worst_differences"]) <= 128


def test_bounded_comparator_rejects_out_of_contract_limits_and_integer_arrays():
    floating = np.ones((4,), dtype=np.float64)
    with pytest.raises(ValueError, match="64 MiB"):
        compare_numeric_bounded(
            floating, floating, max_working_bytes=64 * 1024 * 1024 + 1,
        )
    with pytest.raises(ValueError, match="128"):
        compare_numeric_bounded(floating, floating, max_examples=129)
    with pytest.raises(TypeError, match="floating or complex"):
        compare_numeric_bounded(np.arange(4), np.arange(4))
    with pytest.raises(TypeError, match="floating or complex"):
        compare_numeric_bounded(np.ones(4, dtype=bool), np.ones(4, dtype=bool))


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        (np.asarray([1.0e308]), np.asarray([-1.0e308]), "finite_subtraction_overflow_count"),
        (np.asarray([1.7e308 + 1.7e308j]), np.asarray([1.7e308 + 1.7e308j]), "reference_magnitude_overflow_count"),
        (np.asarray([np.nextafter(0.0, 1.0)]), np.asarray([1.0]), "normalization_overflow_count"),
    ],
)
def test_bounded_comparator_reports_overflow_without_nonstandard_json(
    reference, candidate, expected,
):
    metrics = compare_numeric_bounded(reference, candidate, max_working_bytes=1024)
    assert metrics[expected] > 0
    assert metrics["max_abs"] is None or np.isfinite(metrics["max_abs"])
    assert metrics["normalized_linf"] is None or np.isfinite(metrics["normalized_linf"])
    assert metrics["relative_l2"] is None or np.isfinite(metrics["relative_l2"])
    json.dumps(metrics, allow_nan=False)


def test_bounded_comparator_distinguishes_matching_and_mismatched_nonfinite_values():
    matching = compare_numeric_bounded(
        np.asarray([np.nan, np.inf, -np.inf]),
        np.asarray([np.nan, np.inf, -np.inf]),
        max_working_bytes=1024,
    )
    assert matching["same_nonfinite_value_count"] == 3
    assert matching["nonfinite_mismatch_count"] == 0
    assert matching["canonical_exact"]
    assert matching["max_abs"] is None
    assert matching["max_abs_reason"] == "nonfinite_operands_present"
    json.dumps(matching, allow_nan=False)

    mismatch = compare_numeric_bounded(
        np.asarray([1.0]), np.asarray([np.nan]), max_working_bytes=1024,
    )
    assert mismatch["reference_operand_nonfinite_count"] == 0
    assert mismatch["candidate_operand_nonfinite_count"] == 1
    assert mismatch["nonfinite_mismatch_count"] == 1
    assert mismatch["max_abs"] is None
    json.dumps(mismatch, allow_nan=False)


def test_bounded_comparator_surfaces_dtype_shape_and_zero_scale_gates():
    mixed = compare_numeric_bounded(
        np.asarray([1.0], dtype=np.float64),
        np.asarray([1.0], dtype=np.float32), max_working_bytes=1024,
    )
    assert mixed["dtype_equal"] is False
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_numeric_bounded(np.ones((2,), dtype=np.float64),
                                np.ones((1,), dtype=np.float64))
    zero = compare_numeric_bounded(
        np.zeros(1, dtype=np.float64), np.ones(1, dtype=np.float64),
        max_working_bytes=1024,
    )
    assert zero["zero_reference_nonzero_difference_count"] == 1
    assert zero["normalized_linf"] is None
    assert zero["normalized_linf_reason"] == "zero_reference_scale_with_nonzero_difference"
    json.dumps(zero, allow_nan=False)


def _q1_metric_row(name, metrics, *, scope="science"):
    return {"field_name": name, "scope": scope, "metrics": metrics}


def test_q1_numeric_gate_exact_finite_diff_signed_zero_and_hard_failures():
    base = {
        "reference_shape": [1], "candidate_shape": [1], "dtype_equal": True,
        "reference_dtype": "float64", "candidate_dtype": "float64",
        "all_finite": True, "canonical_exact": True, "bitwise_equal": True,
        "value_equal": True, "max_abs": 0.0,
    }
    exact = classify_comparison(_q1_metric_row("E", {**base}))
    assert exact["classification"] == "CURRENT_LINUX_LOCAL_EXACT"
    signed_zero = classify_comparison(_q1_metric_row("E", {
        **base, "canonical_exact": False, "bitwise_equal": False,
    }))
    assert signed_zero["classification"] == "FINITE_NONEXACT_DIFFERENCE"
    numeric_diff = classify_comparison(_q1_metric_row("E", {
        **base, "canonical_exact": False, "bitwise_equal": False,
        "value_equal": False, "max_abs": 1e-8,
    }))
    assert numeric_diff["classification"] == "FINITE_NONZERO_DIFFERENCE"
    dtype_fail = classify_comparison(_q1_metric_row("E", {
        **base, "candidate_dtype": "float32", "dtype_equal": False,
    }))
    assert dtype_fail["classification"] == "NONFINITE_OR_TOPOLOGY_FAILURE"
    inconsistent_dtype = classify_comparison(_q1_metric_row("E", {
        **base, "dtype_equal": False,
    }))
    assert inconsistent_dtype["classification"] == "TOOL_FAILURE"


def test_q1_numeric_gate_allows_only_named_exact_nan_placeholders_and_tool_errors():
    exact_nan = compare_numeric_bounded(
        np.asarray([np.nan, np.nan], dtype=np.float64),
        np.asarray([np.nan, np.nan], dtype=np.float64), max_working_bytes=1024,
    )
    allowed = classify_comparison(_q1_metric_row(
        "energy_step_start_J", exact_nan, scope="diagnostics",
    ))
    assert allowed["proceed"] is True
    denied = classify_comparison(_q1_metric_row(
        "unlisted", exact_nan, scope="diagnostics",
    ))
    assert denied["classification"] == "NONFINITE_OR_TOPOLOGY_FAILURE"
    for invalid_complex in (
        np.asarray([complex(np.nan, np.inf)]),
        np.asarray([complex(np.nan, np.nan)]),
    ):
        complex_metrics = compare_numeric_bounded(
            invalid_complex, invalid_complex.copy(), max_working_bytes=1024,
        )
        rejected = classify_comparison(_q1_metric_row(
            "energy_step_start_J", complex_metrics, scope="diagnostics",
        ))
        assert rejected["classification"] == "NONFINITE_OR_TOPOLOGY_FAILURE"
        assert rejected["proceed"] is False

    infinity_metrics = compare_numeric_bounded(
        np.asarray([np.inf, np.inf], dtype=np.float64),
        np.asarray([np.inf, np.inf], dtype=np.float64), max_working_bytes=1024,
    )
    assert classify_comparison(_q1_metric_row(
        "energy_step_start_J", infinity_metrics, scope="diagnostics",
    ))["proceed"] is False
    tool_failure = build_numeric_gate(
        identity_checks={"source": True}, linux_environment_eligible=True,
        tool_error="RuntimeError: comparator failed",
    )
    assert tool_failure["classification"] == "TOOL_FAILURE"
    with pytest.raises(ValueError):
        json.dumps(tool_failure | {"bad": float("inf")}, allow_nan=False)
    json.dumps(tool_failure, allow_nan=False)


def test_q1_timing_only_rows_do_not_claim_a_numerical_exact_gate():
    receipt = build_numeric_gate(
        identity_checks={"source": True}, linux_environment_eligible=True,
        comparisons=[_q1_metric_row("linear_walltime_step_s", {}, scope="timing")],
    )
    assert receipt["classification"] == "IDENTITY_PASS"
    assert receipt["proceed_to_next_identity"] is True
    unknown_timing = classify_comparison(_q1_metric_row(
        "unregistered_field", {}, scope="timing",
    ))
    assert unknown_timing["classification"] == "TOOL_FAILURE"
    assert unknown_timing["proceed"] is False


@pytest.mark.parametrize("missing", [
    "reference_shape", "candidate_shape", "reference_dtype", "candidate_dtype",
    "dtype_equal", "canonical_exact", "bitwise_equal", "all_finite",
])
def test_q1_numeric_gate_rejects_missing_comparison_identity_metadata(missing):
    metrics = {
        "reference_shape": [1], "candidate_shape": [1],
        "reference_dtype": "float64", "candidate_dtype": "float64",
        "dtype_equal": True, "canonical_exact": True, "bitwise_equal": True,
        "all_finite": True,
    }
    metrics.pop(missing)
    result = classify_comparison(_q1_metric_row("E", metrics))
    assert result["classification"] == "TOOL_FAILURE"
    assert result["proceed"] is False


def test_q1_tiny_gate_array_adapter_uses_bounded_comparator_and_hard_contracts():
    exact = _array_comparison_row(
        "tiny_E", np.arange(32, dtype=np.float64), np.arange(32, dtype=np.float64),
        scope="science",
    )
    assert exact["metrics"]["canonical_exact"] is True
    assert exact["metrics"]["bitwise_equal"] is True
    json.dumps(exact, allow_nan=False)

    dtype_mismatch = _array_comparison_row(
        "tiny_E", np.ones(2, dtype=np.float64), np.ones(2, dtype=np.float32),
        scope="science",
    )
    assert classify_comparison(dtype_mismatch)["classification"] == "NONFINITE_OR_TOPOLOGY_FAILURE"
    shape_mismatch = _array_comparison_row(
        "tiny_POST", np.ones((2, 3), dtype=np.float64),
        np.ones((3, 2), dtype=np.float64), scope="science",
    )
    assert classify_comparison(shape_mismatch)["classification"] == "NONFINITE_OR_TOPOLOGY_FAILURE"


def _make_q1_comparison_fixture(root: Path) -> tuple[Path, Path]:
    reference, candidate = root / "reference", root / "candidate"
    for side in (reference, candidate):
        propagation = side / "propagation"
        propagation.mkdir(parents=True)
        np.save(propagation / "final_E.npy", np.arange(8, dtype=np.float64).reshape(2, 2, 2))
        np.save(side / "scientific_Qacc.npy", np.asarray([0.0, 2.0], dtype=np.float64))
        np.savez(propagation / "final_POST.npz", delta_n=np.asarray([1.0, 2.0]))
        np.savez(propagation / "diagnostics.npz", U_z=np.asarray([3.0, 4.0]))
    return reference, candidate


def test_q1_raw_comparison_jsonl_preserves_rows_before_later_comparison_error(
    tmp_path, monkeypatch,
):
    reference, candidate = _make_q1_comparison_fixture(tmp_path)
    comparison_path = candidate / "comparison.jsonl"
    original = q1_t0._array_comparison_row

    def fail_on_post(field_name, *args, **kwargs):
        if field_name == "final_POST.delta_n":
            raise ArithmeticError("injected POST comparator failure")
        return original(field_name, *args, **kwargs)

    monkeypatch.setattr(q1_t0, "_array_comparison_row", fail_on_post)
    with pytest.raises(ArithmeticError, match="injected POST comparator"):
        q1_t0._compare_window_outputs(
            reference, candidate, "I", comparison_path=comparison_path,
            scalar_reference={}, scalar_candidate={},
        )

    records = [json.loads(line) for line in comparison_path.read_text(encoding="utf-8").splitlines()]
    assert [row["record_type"] for row in records] == [
        "array_comparison", "array_comparison", "comparison_error",
    ]
    assert records[0]["identity"] == "I"
    assert records[0]["field_name"] == "final_E"
    assert records[0]["metrics"]["reference_canonical_sha256"]
    assert records[0]["metrics"]["zero_reference_mask_sha256"]
    assert records[0]["metrics"]["max_abs"] == 0.0
    assert records[1]["field_name"] == "public_return_Qacc"
    assert records[1]["metrics"]["reference_canonical_sha256"]
    assert records[2]["rows_written"] == 2
    assert records[2]["error"] == "injected POST comparator failure"
    for row in records:
        json.dumps(row, allow_nan=False)


def test_q1_tiny_gate_compares_and_persists_private_python_scalar_diagnostics(tmp_path):
    reference, candidate = _make_q1_comparison_fixture(tmp_path)
    payload, reference_scalars = q1_t0._numeric_scalar_diagnostics({
        "n_intervals": np.int64(8), "z_start": np.float64(-0.0),
        "nested": {"deposition_channel": np.float64(2.5)},
        "array_field": np.asarray([1.0]),
    })
    candidate_payload, candidate_scalars = q1_t0._numeric_scalar_diagnostics({
        "n_intervals": np.int64(8), "z_start": np.float64(-0.0),
        "nested": {"deposition_channel": np.float64(2.5)},
        "array_field": np.asarray([1.0]),
    })
    assert payload == candidate_payload
    assert set(reference_scalars) == {"n_intervals", "nested.deposition_channel", "z_start"}
    scalar_path = candidate / "scientific_scalar_diagnostics.json"
    q1_t0._write_scalar_diagnostics(scalar_path, "I", payload)
    assert json.loads(scalar_path.read_text(encoding="utf-8"))["fields"] == payload

    comparison_path = candidate / "comparison.jsonl"
    rows = q1_t0._compare_window_outputs(
        reference, candidate, "I", comparison_path=comparison_path,
        scalar_reference=reference_scalars, scalar_candidate=candidate_scalars,
    )
    scalar_rows = [row for row in rows if row.get("diagnostic_source") ==
                   "private_propagator_return_diag_scalar"]
    assert {row["field_name"] for row in scalar_rows} == set(reference_scalars)
    assert all(row["metrics"]["canonical_exact"] for row in scalar_rows)
    persisted = [json.loads(line) for line in comparison_path.read_text(encoding="utf-8").splitlines()]
    assert len(persisted) == len(rows)
    assert all(record["identity"] == "I" for record in persisted)
    assert all(record["metrics"]["reference_shape"] == [] for record in persisted
               if record.get("diagnostic_source") == "private_propagator_return_diag_scalar")
    for record in persisted:
        json.dumps(record, allow_nan=False)


def test_q1_public_qacc_return_difference_is_bounded_and_blocks_gate(tmp_path):
    reference, candidate = _make_q1_comparison_fixture(tmp_path)
    np.save(candidate / "scientific_Qacc.npy",
            np.asarray([0.0, 2.25], dtype=np.float64))
    comparison_path = candidate / "comparison.jsonl"
    rows = q1_t0._compare_window_outputs(
        reference, candidate, "I", comparison_path=comparison_path,
        scalar_reference={}, scalar_candidate={},
    )
    qacc_row = next(row for row in rows if row["field_name"] == "public_return_Qacc")
    assert qacc_row["metrics"]["max_abs"] == 0.25
    assert qacc_row["metrics"]["canonical_exact"] is False
    decision = classify_comparison(qacc_row)
    assert decision["classification"] == "FINITE_NONZERO_DIFFERENCE"
    assert decision["proceed"] is False

    gate = build_numeric_gate(
        identity_checks={"environment": True, "source": True},
        linux_environment_eligible=True, comparisons=rows,
    )
    assert gate["classification"] == "FINITE_NONZERO_DIFFERENCE"
    assert gate["proceed_to_next_identity"] is False
    persisted = [json.loads(line) for line in comparison_path.read_text(encoding="utf-8").splitlines()]
    persisted_qacc = next(row for row in persisted if row["field_name"] == "public_return_Qacc")
    assert persisted_qacc["identity"] == "I"
    assert persisted_qacc["metrics"]["candidate_canonical_sha256"]
    json.dumps(persisted_qacc, allow_nan=False)


def test_q1_joint_factory_can_explicitly_configure_single_thread_preflight_dispatch():
    factory = make_joint_propagator_factory(
        "B", kernel_threads=1, block_pixels=32, layout="NtB",
        temporal_workers=1, spatial_workers=1,
    )
    assert callable(factory)
    with pytest.raises(ValueError, match="temporal_workers"):
        make_joint_propagator_factory("B", temporal_workers=0)


def test_q1_linux_tiny_gate_cli_persists_strict_tool_failure_receipt(
    tmp_path, monkeypatch,
):
    def fail_before_probe(*args, **kwargs):
        raise RuntimeError("pinned Linux dependency unavailable")

    monkeypatch.setattr("tools.cpu_o1.q1_t0.run_linux_tiny_science_gate", fail_before_probe)
    output = tmp_path / "linux_tiny"
    assert q1_t0_main(["--linux-tiny-gate", "--output-root", str(output)]) == 1
    receipt = json.loads((output / "NUMERIC_GATE.json").read_text(encoding="utf-8"))
    assert receipt["classification"] == "TOOL_FAILURE"
    assert receipt["proceed_to_next_identity"] is False
    json.dumps(receipt, allow_nan=False)


def test_plan_rejects_ambiguous_parallel_and_layout_settings():
    with pytest.raises(ValueError, match="kernel_threads=1"):
        CPUExecutionPlan(implementation="numba_serial", kernel_threads=2)
    with pytest.raises(ValueError, match="layout"):
        CPUExecutionPlan(layout="xyz")
    with pytest.raises(ValueError, match="temporal_workers"):
        CPUExecutionPlan(temporal_workers=True)
    with pytest.raises(ValueError, match="spatial_workers"):
        CPUExecutionPlan(spatial_workers=3)
    with pytest.raises(ValueError, match="workspace"):
        CPUExecutionPlan(workspace_owner="global")
    base = CPUExecutionPlan(temporal_workers=48, spatial_workers=24)
    assert base.as_dict()["temporal_workers"] == 48
    assert base.as_dict()["spatial_workers"] == 24


def test_numeric_metrics_keep_zero_reference_mask_and_undefined_scales_explicit():
    reference = np.asarray([0.0, -0.0, 2.0], dtype=np.float64)
    candidate = np.asarray([0.0, 0.0, 2.5], dtype=np.float64)
    metrics = compare_numeric(reference, candidate)
    assert not metrics["canonical_exact"]
    assert not metrics["bitwise_equal"]
    assert metrics["zero_reference_count"] == 2
    assert np.array_equal(metrics["zero_reference_mask"], [True, True, False])
    assert np.isnan(metrics["elementwise_relative_error"][:2]).all()
    assert metrics["normalized_linf"] == 0.25
    zero = compare_numeric(np.zeros(3), np.zeros(3))
    assert zero["normalized_linf"] == 0.0
    assert zero["normalized_linf_reason"] is None
    assert zero["relative_l2"] == 0.0
    assert zero["relative_l2_reason"] is None
    different = compare_numeric(np.zeros(3), np.asarray([0.0, 1.0, 0.0]))
    assert different["normalized_linf"] is None
    assert different["normalized_linf_reason"] == "zero_reference_scale_with_nonzero_difference"
    assert different["relative_l2"] is None
    assert different["relative_l2_reason"] == "zero_reference_scale_with_nonzero_difference"


def test_canonical_hash_and_value_equality_distinguish_signed_zero_contracts():
    reference = np.asarray([0.0, -0.0], dtype=np.float64)
    candidate = np.asarray([-0.0, -0.0], dtype=np.float64)
    metrics = compare_numeric(reference, candidate)
    assert metrics["value_equal"]
    assert not metrics["canonical_exact"]
    assert not metrics["bitwise_equal"]
    assert metrics["reference_canonical_sha256"] != metrics["candidate_canonical_sha256"]


@pytest.mark.parametrize(
    "values, lower, upper",
    [
        ([-0.0, 0.0, -1.0, 1.0, 2.0], 0.0, 1.0),
        ([-0.0, 0.0, -1.0, 1.0, 2.0], -1.0, 0.0),
        ([-2.0, -0.0, 0.0, 2.0], 1.0, -1.0),
    ],
)
def test_numba_photo_clip_scalar_matches_numpy_clip_signed_zero_and_bounds(values, lower, upper):
    from tools.cpu_o1 import cpu_o1

    _load = cpu_o1._load_numba_kernels
    _load()
    compiled_clip = cpu_o1._NUMBA_PHOTO_CLIP
    actual = np.asarray([compiled_clip(float(value), lower, upper) for value in values])
    expected = np.clip(np.asarray(values, dtype=np.float64), lower, upper)
    metrics = compare_numeric(expected, actual)
    assert metrics["canonical_exact"]
    assert metrics["bitwise_equal"]
    _persist_metrics({
        "label": "numba_photo_clip_scalar_vs_numpy_clip",
        "settings": {"lower": lower, "upper": upper, "values": values},
        "measurement": {"implementation": "shared_njit_photo_clip_helper"},
        "array_metrics": {"clip_result": _metric_fields(metrics)},
    })


def test_reference_adapter_preserves_frozen_tuple_and_isolation():
    from KHz_filament.ionization import runtime

    field = _tiny_field()
    before = field.copy()
    xp_before = runtime.xp
    fft_before = np.fft
    args = _args(field)
    direct = runtime.evolve_rho_time(
        args["input_array"], args["dt"], args["N0"], args["beta_rec"], args["Wfunc"],
        return_species_terms=True, return_species_densities=True,
        diagnose_integrator_stability=True,
    )
    adapted = evolve_rho_time_cpu_o1(**args, plan=CPUExecutionPlan())
    _assert_exact_tree(direct, adapted)
    assert np.array_equal(field, before)
    assert runtime.xp is xp_before
    assert np.fft is fft_before


def test_full_time_mode_overrides_quasi_static_argument_like_reference():
    field = _tiny_field(nt=7, ny=2, nx=3)
    args = _args(field)
    from KHz_filament.ionization import runtime

    reference = runtime.evolve_rho_time(
        args["input_array"], args["dt"], args["N0"], args["beta_rec"], args["Wfunc"],
        quasi_static_time=True, return_species_terms=True,
        return_species_densities=True, diagnose_integrator_stability=True,
    )
    candidate = evolve_rho_time_cpu_o1(
        **args, quasi_static_time=True,
        plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
    )
    _assert_contract_tree(reference, candidate)


def test_cpu_executor_is_a_thin_plan_bound_adapter():
    field = _tiny_field(nt=7, ny=2, nx=3)
    args = _args(field)
    executor = CPUExecutor(CPUExecutionPlan())
    direct = evolve_rho_time_cpu_o1(**args, plan=CPUExecutionPlan())
    adapted = executor.evolve_rho_time(**args)
    _assert_exact_tree(direct, adapted)
    with pytest.raises(TypeError, match="binds its plan"):
        executor.evolve_rho_time(**args, plan=CPUExecutionPlan())


@pytest.mark.parametrize("layout", ["NtB", "BNt"])
@pytest.mark.parametrize("block_pixels", [1, 7, 32])
def test_serial_kernel_matches_reference_across_layouts_and_tails(layout, block_pixels):
    field = _tiny_field(nt=11, ny=3, nx=5)
    before = field.copy()
    args = _args(field)
    reference = evolve_rho_time_cpu_o1(**args, plan=CPUExecutionPlan())
    metrics = {}
    candidate = evolve_rho_time_cpu_o1(
        **args,
        plan=CPUExecutionPlan(
            implementation="numba_serial", layout=layout, block_pixels=block_pixels,
        ),
        metrics_out=metrics,
    )
    _assert_contract_tree(reference, candidate)
    _metric_record(
        reference, candidate, label="serial", settings=metrics.get("settings"),
        measurement=metrics,
    )
    repeated = evolve_rho_time_cpu_o1(
        **args,
        plan=CPUExecutionPlan(
            implementation="numba_serial", layout=layout, block_pixels=block_pixels,
        ),
    )
    _assert_exact_tree(candidate, repeated)
    assert metrics["implementation"] == "numba_serial"
    assert np.array_equal(field, before)
    assert all(np.all(np.isfinite(value)) for value in candidate[2].values() if isinstance(value, np.ndarray))
    net = candidate[2]["drho_dt_u_sum"]
    photo = candidate[2]["photoionization_energy_rate"]
    assert np.any(net < photo), "the diagnostic net derivative must include recombination"


@pytest.mark.parametrize("layout", ["NtB", "BNt"])
@pytest.mark.parametrize("block_pixels", [1, 7, 32])
def test_parallel_kernel_matches_reference_and_restores_thread_mask(layout, block_pixels):
    field = _tiny_field(nt=11, ny=3, nx=5)
    args = _args(field)
    reference = evolve_rho_time_cpu_o1(**args, plan=CPUExecutionPlan())
    serial = evolve_rho_time_cpu_o1(
        **args,
        plan=CPUExecutionPlan(
            implementation="numba_serial", layout=layout, block_pixels=block_pixels,
        ),
    )
    max_threads = available_numba_threads()
    thread_counts = sorted({min(value, max_threads) for value in (1, 2, 4)})
    from tools.cpu_o1 import cpu_o1

    numba, _, _ = cpu_o1._load_numba_kernels()
    original_threads = int(numba.get_num_threads())
    for threads in thread_counts:
        metrics = {}
        candidate = evolve_rho_time_cpu_o1(
            **args,
            plan=CPUExecutionPlan(
                implementation="numba_parallel", kernel_threads=threads,
                layout=layout, block_pixels=block_pixels,
            ),
            metrics_out=metrics,
        )
        _assert_contract_tree(reference, candidate)
        _assert_exact_tree(serial, candidate)
        repeated = evolve_rho_time_cpu_o1(
            **args,
            plan=CPUExecutionPlan(
                implementation="numba_parallel", kernel_threads=threads,
                layout=layout, block_pixels=block_pixels,
            ),
        )
        _assert_exact_tree(candidate, repeated)
        _metric_record(
            reference, candidate, label="parallel", settings=metrics.get("settings"),
            measurement=metrics,
        )
        assert int(numba.get_num_threads()) == original_threads
        assert metrics["numba_threading_layer"] == "workqueue"
        assert metrics["numba_dependency_load_s"] >= 0.0
        assert metrics["kernel_compilation_or_cache_load_s"] >= 0.0
        assert metrics["kernel_s"] >= 0.0


def test_nt384_complete_time_chain_accepts_readonly_broadcast_view():
    source = _tiny_field(nt=384, ny=5, nx=1)
    field = np.broadcast_to(source, (384, 5, 7))
    assert not field.flags.writeable
    assert not field.flags.c_contiguous
    before = np.array(field, copy=True)
    args = _args(field)

    reference = evolve_rho_time_cpu_o1(**args, plan=CPUExecutionPlan())
    serial_metrics = {}
    serial = evolve_rho_time_cpu_o1(
        **args,
        plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
        metrics_out=serial_metrics,
    )
    parallel_metrics = {}
    parallel = evolve_rho_time_cpu_o1(
        **args,
        plan=CPUExecutionPlan(
            implementation="numba_parallel",
            kernel_threads=min(2, available_numba_threads()),
            block_pixels=7,
        ),
        metrics_out=parallel_metrics,
    )

    _assert_contract_tree(reference, serial)
    _assert_contract_tree(reference, parallel)
    _assert_exact_tree(serial, parallel)
    assert np.array_equal(field, before)
    assert not field.flags.writeable
    _metric_record(
        reference, serial, label="nt384_readonly_broadcast_serial",
        settings=serial_metrics.get("settings"), measurement=serial_metrics,
    )
    _metric_record(
        reference, parallel, label="nt384_readonly_broadcast_parallel",
        settings=parallel_metrics.get("settings"), measurement=parallel_metrics,
    )


def test_final_fraction_clip_and_stability_return_match_reference():
    field = _tiny_field(nt=7, ny=2, nx=3)
    args = _args(field, high_rate=True)
    reference = evolve_rho_time_cpu_o1(**args, plan=CPUExecutionPlan())
    metrics = {}
    candidate = evolve_rho_time_cpu_o1(
        **args,
        plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
        metrics_out=metrics,
    )
    _assert_contract_tree(reference, candidate)
    _metric_record(
        reference, candidate, label="clip_stability", settings=metrics.get("settings"),
        measurement=metrics,
    )
    repeated = evolve_rho_time_cpu_o1(
        **args,
        plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
    )
    _assert_exact_tree(candidate, repeated)
    clip_count = sum(item["step_clip_count"] for item in candidate[3]["species"].values())
    intermediate_count = sum(
        item["intermediate_violation_count"] for item in candidate[3]["species"].values()
    )
    assert clip_count > 0
    assert intermediate_count > 0


@pytest.mark.parametrize("return_terms", [False, True])
@pytest.mark.parametrize("return_densities", [False, True])
@pytest.mark.parametrize("stability", [False, True])
def test_compiled_adapter_preserves_all_return_flag_combinations(
    return_terms, return_densities, stability,
):
    from KHz_filament.ionization import runtime

    field = _tiny_field(nt=4, ny=2, nx=3)
    wfunc = _ToyWfunc()
    kwargs = {
        "return_species_terms": return_terms,
        "return_species_densities": return_densities,
        "diagnose_integrator_stability": stability,
    }
    reference = runtime.evolve_rho_time(field, 0.2, 25.0, 0.04, wfunc, **kwargs)
    candidate = evolve_rho_time_cpu_o1(
        field, 0.2, 25.0, 0.04, wfunc, **kwargs,
        plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
    )
    _assert_contract_tree(reference, candidate)


@pytest.mark.parametrize(
    "wfunc, quasi_static",
    [(_ToyWfunc(integrator="euler"), False), (_ToyWfunc(time_mode="qs_peak"), False)],
)
def test_compiled_adapter_rejects_unsupported_integrator_and_time_mode(wfunc, quasi_static):
    with pytest.raises(ValueError, match="only supports"):
        evolve_rho_time_cpu_o1(
            _tiny_field(), 0.1, 10.0, 0.01, wfunc,
            plan=CPUExecutionPlan(implementation="numba_serial"),
            quasi_static_time=quasi_static,
        )


@pytest.mark.parametrize("field_name,value", [
    ("dt", np.inf), ("N0", np.nan), ("beta_rec", -np.inf),
])
def test_compiled_adapter_rejects_nonfinite_scalar_inputs(field_name, value):
    kwargs = _args(_tiny_field())
    kwargs[field_name] = value
    with pytest.raises(ValueError, match="finite"):
        evolve_rho_time_cpu_o1(
            **kwargs, plan=CPUExecutionPlan(implementation="numba_serial"),
        )


def test_reference_plan_keeps_invalid_input_behavior_owned_by_frozen_runtime():
    field = _tiny_field(nt=4, ny=2, nx=3)
    args = _args(field)
    args["dt"] = np.inf
    from KHz_filament.ionization import runtime

    expected = runtime.evolve_rho_time(
        args["input_array"], args["dt"], args["N0"], args["beta_rec"], args["Wfunc"],
        return_species_terms=True, return_species_densities=True,
        diagnose_integrator_stability=True,
    )
    actual = evolve_rho_time_cpu_o1(**args, plan=CPUExecutionPlan())
    _assert_exact_tree(expected, actual)


def test_compiled_adapter_rejects_nonfinite_input_rates_and_ionization_energy():
    field = _tiny_field()
    invalid_field = field.copy()
    invalid_field[0, 0, 0] = np.nan + 0j
    with pytest.raises(ValueError, match="input must be finite"):
        evolve_rho_time_cpu_o1(
            **_args(invalid_field), plan=CPUExecutionPlan(implementation="numba_serial"),
        )

    invalid_rates = _ToyWfunc()
    invalid_rates._species_entries = tuple(
        {**entry, "W_runtime": lambda value: np.full(value.shape, np.inf)}
        if index == 0 else entry
        for index, entry in enumerate(invalid_rates._species_entries)
    )
    with pytest.raises(ValueError, match="W_runtime output"):
        evolve_rho_time_cpu_o1(
            field, 0.2, 25.0, 0.04, invalid_rates,
            plan=CPUExecutionPlan(implementation="numba_serial"),
        )

    invalid_energy = _ToyWfunc()
    invalid_energy._species_entries = (
        {**invalid_energy._species_entries[0], "Ui_J": np.inf},
        invalid_energy._species_entries[1],
    )
    with pytest.raises(ValueError, match="ionization energies"):
        evolve_rho_time_cpu_o1(
            field, 0.2, 25.0, 0.04, invalid_energy,
            plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
            return_species_terms=True,
        )


def test_compiled_adapter_rejects_nonfinite_derived_constants_and_output_arrays():
    field = _tiny_field(nt=4, ny=2, nx=3)
    args = _args(field)
    with pytest.raises(ValueError, match="derived species constants"):
        evolve_rho_time_cpu_o1(
            **{**args, "N0": 1.0e308, "beta_rec": 1.0e308},
            plan=CPUExecutionPlan(implementation="numba_serial"),
        )

    wfunc = _ToyWfunc()
    wfunc._species_entries = tuple(
        {**entry, "Ui_J": 1.0e308,
         "W_s": lambda value: np.full(value.shape, 1.0e308),
         "W_runtime": lambda value: np.full(value.shape, 1.0e308)}
        for entry in wfunc._species_entries
    )
    with pytest.raises(ValueError, match="non-finite output array"):
        evolve_rho_time_cpu_o1(
            field, 1.0e-308, 1.0e308, 0.0, wfunc,
            plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
            return_species_terms=True,
        )


def test_fraction_normalization_preserves_reference_zero_and_infinite_sum_fallback():
    field = _tiny_field(nt=7, ny=2, nx=3)
    for fractions in ((0.0, 0.0), (np.inf, 0.0)):
        wfunc = _ToyWfunc()
        wfunc._species_entries = tuple(
            {**entry, "fraction": fractions[index]}
            for index, entry in enumerate(wfunc._species_entries)
        )
        from KHz_filament.ionization import runtime

        reference = runtime.evolve_rho_time(
            field, 0.2, 25.0, 0.04, wfunc,
            return_species_terms=True, return_species_densities=True,
        )
        candidate = evolve_rho_time_cpu_o1(
            field, 0.2, 25.0, 0.04, wfunc,
            plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
            return_species_terms=True, return_species_densities=True,
        )
        _assert_exact_tree(reference, candidate)


def test_single_species_density_return_preserves_frozen_alias():
    from KHz_filament.ionization import runtime

    field = _tiny_field(nt=7, ny=2, nx=3)
    wfunc = _ToyWfunc()
    wfunc._species_entries = (wfunc._species_entries[0],)
    reference = runtime.evolve_rho_time(
        field, 0.2, 25.0, 0.04, wfunc,
        return_species_terms=True, return_species_densities=True,
    )
    candidate = evolve_rho_time_cpu_o1(
        field, 0.2, 25.0, 0.04, wfunc,
        plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
        return_species_terms=True, return_species_densities=True,
    )
    _assert_contract_tree(reference, candidate)
    assert reference[0] is reference[2]["rho_by_species"]["toy-a"]
    assert candidate[0] is candidate[2]["rho_by_species"]["toy-a"]

    multi = _args(field)
    candidate_multi = evolve_rho_time_cpu_o1(
        **multi, plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
    )
    assert candidate_multi[0] is not candidate_multi[2]["rho_by_species"]["toy-a"]


def test_signed_zero_in_rk4_input_and_rates_is_recorded_without_tolerance():
    field = _tiny_field(nt=7, ny=2, nx=3)
    field[0, 0, 0] = complex(-0.0, 0.0)
    assert np.signbit(field.real[0, 0, 0])
    observed_input_signs = []

    def negative_zero_rate(value):
        observed_input_signs.append(bool(np.signbit(value[0, 0, 0])))
        return np.full(value.shape, -0.0, dtype=np.float64)

    wfunc = _ToyWfunc()
    wfunc._species_entries = tuple(
        {**entry, "W_s": negative_zero_rate, "W_runtime": negative_zero_rate}
        for entry in wfunc._species_entries
    )
    args = {
        "input_array": field,
        "dt": 0.2,
        "N0": 25.0,
        "beta_rec": 0.04,
        "Wfunc": wfunc,
        "return_species_terms": True,
        "return_species_densities": True,
        "diagnose_integrator_stability": True,
    }
    from KHz_filament.ionization import runtime

    reference = runtime.evolve_rho_time(**args)
    reference_sign_observations = len(observed_input_signs)
    serial_metrics = {}
    serial = evolve_rho_time_cpu_o1(
        **args, plan=CPUExecutionPlan(implementation="numba_serial", block_pixels=7),
        metrics_out=serial_metrics,
    )
    assert reference_sign_observations > 0
    assert all(observed_input_signs)
    _assert_contract_tree(reference, serial)
    _metric_record(
        reference, serial, label="signed_zero_rk4",
        settings=serial_metrics.get("settings"), measurement=serial_metrics,
    )


def test_workspace_stamp_requires_same_field_version_shape_dtype_and_layout():
    stamp = WorkspaceStamp(field_version=4, shape=(9, 3, 5), dtype="complex128", layout="NtB")
    assert workspace_matches(
        stamp, field_version=4, shape=(9, 3, 5), dtype="complex128", layout="NtB",
    )
    assert not workspace_matches(
        stamp, field_version=5, shape=(9, 3, 5), dtype="complex128", layout="NtB",
    )
    assert not workspace_matches(
        stamp, field_version=4, shape=(9, 3, 5), dtype="complex64", layout="NtB",
    )
    assert not workspace_matches(
        stamp, field_version=4, shape=(9, 3, 5), dtype="complex128", layout="BNt",
    )


def test_raman_reference_stage_heun_reload_and_iir_causality():
    from KHz_filament.constants import c0, eps0
    from KHz_filament.raman import (
        apply_isaacs_raman_operator_step,
        isaacs_raman_stage,
        raman_convolve_intensity,
        stable_field_fluence_and_loss,
    )

    nt, ny, nx = 8, 2, 3
    dt = 1.0e-15
    tau = np.arange(nt, dtype=np.float64)[:, None, None]
    envelope = np.exp(-((tau - 3.0) / 2.0) ** 2)
    spatial = np.linspace(0.8, 1.1, ny * nx, dtype=np.float64).reshape(1, ny, nx)
    field = (2.0e7 * envelope * spatial * (1.0 + 0.15j)).astype(np.complex128)
    omega = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
    kwargs = {
        "Omega": omega, "dt": dt, "omega0": 2.0e15, "n0": 1.0,
        "n_R": 1.0e-7, "omega_R": 2.0e14, "Gamma_R": 2.0e12,
        "method": "iir", "iir_sampling": "exact_piecewise_linear",
    }

    stage1 = isaacs_raman_stage(field, **kwargs)
    direct_repeat = isaacs_raman_stage(field.copy(), **kwargs)
    _assert_exact_tree(
        {key: value for key, value in stage1.items() if key != "walltime_s"},
        {key: value for key, value in direct_repeat.items() if key != "walltime_s"},
    )
    # Recompose the existing whole-field energy expression by spatial blocks.
    # The global scale is prepared once from the full intensity field and then
    # shared by every block, as required by the reference stage.
    intensity = (
        0.5 * float(eps0) * float(c0) * kwargs["n0"] * np.abs(field) ** 2
    ).astype(np.float64, copy=False)
    intensity_scale = float(np.max(np.abs(intensity)))
    derivative_increment_norm = np.fft.ifft(
        (1j * np.asarray(omega, dtype=np.float64) * dt)[:, None, None]
        * np.fft.fft(intensity, axis=0), axis=0,
    ).real / intensity_scale
    response_norm = stage1["I_R"] / intensity_scale
    energy_scale = (kwargs["n_R"] / float(c0)) * intensity_scale * intensity_scale
    response_blocks = response_norm.reshape(nt, ny * nx)
    derivative_blocks = derivative_increment_norm.reshape(nt, ny * nx)
    tiled_u = np.empty((ny * nx,), dtype=np.float64)
    for start in range(0, ny * nx, 2):
        stop = min(start + 2, ny * nx)
        tiled_u[start:stop] = energy_scale * np.sum(
            response_blocks[:, start:stop] * derivative_blocks[:, start:stop],
            axis=0,
        )
    tiled_u = tiled_u.reshape(ny, nx)
    _assert_exact_tree(stage1["u_R_signed"], tiled_u)
    _metric_record(
        stage1["u_R_signed"], tiled_u, label="raman_global_scale_block_reassembly",
        settings={"block_pixels": 2, "scale_scope": "full_field"},
        measurement={"intensity_scale": intensity_scale, "energy_scale": energy_scale},
    )
    predictor = (field + 2.0e-13 * stage1["rhs"]).astype(field.dtype, copy=False)
    stage2 = isaacs_raman_stage(predictor, **kwargs)
    assert not np.array_equal(stage1["I_R"], stage2["I_R"])
    result = apply_isaacs_raman_operator_step(
        field, 2.0e-13, **kwargs, integrator="heun", return_diagnostics=False,
    )
    expected = (field + 0.5 * 2.0e-13 * (stage1["rhs"] + stage2["rhs"])).astype(
        field.dtype, copy=False
    )
    _assert_exact_tree(expected, result)
    assert stage1["convolution_count"] == 1
    assert stage2["convolution_count"] == 1
    assert stage1["u_R_signed"].shape == (ny, nx)
    np.testing.assert_array_equal(stage1["q_R_positive"], np.maximum(-stage1["u_R_signed"], 0.0))
    before_fluence, equal_field_loss = stable_field_fluence_and_loss(
        field, field.copy(), dt=dt, n0=1.0,
    )
    assert np.all(np.isfinite(before_fluence))
    assert np.array_equal(equal_field_loss, np.zeros_like(equal_field_loss))

    impulse_at_end = np.zeros((nt, 1, 1), dtype=np.float64)
    impulse_at_end[-1, 0, 0] = 1.0
    causal = raman_convolve_intensity(
        impulse_at_end, method="iir", dt=dt, omega_R=2.0e14,
        Gamma_R=2.0e12, iir_sampling="exact_piecewise_linear",
    )
    assert np.array_equal(causal[:-1], np.zeros_like(causal[:-1]))


def _raman_candidate_fixture(nt=384, ny=5, nx=7, *, zero=False):
    t = np.arange(nt, dtype=np.float64)[:, None, None]
    scale = np.geomspace(1.0e-3, 1.0e3, ny * nx).reshape(1, ny, nx)
    temporal = np.exp(-((t - 0.38 * nt) / (0.19 * nt)) ** 2)
    phase = 0.7 * np.sin(0.071 * t) + 0.2 * np.cos(0.037 * t)
    field = (1.0e7 * temporal * scale * (np.cos(phase) + 1j * np.sin(phase)))
    field = np.ascontiguousarray(field, dtype=np.complex128)
    if zero:
        field.fill(0.0)
    dt = 1.0e-15
    kwargs = {
        "Omega": 2.0 * np.pi * np.fft.fftfreq(nt, d=dt),
        "dt": dt,
        "omega0": 2.0e15,
        "n0": 1.0,
        "n_R": 2.3e-23,
        "omega_R": 1.6e13,
        "Gamma_R": 1.3e13,
        "method": "iir",
        "iir_sampling": "exact_piecewise_linear",
    }
    return field, kwargs


@pytest.mark.parametrize("kernel_threads", [1, 2, 4])
def test_raman_candidate_stage_heun_energy_and_field_version(kernel_threads):
    from KHz_filament.propagate import propagate_one_pulse
    from KHz_filament.raman import stable_field_fluence_and_loss
    from tools.cpu_o1.cpu_o1 import (
        _stable_field_fluence_and_loss_cpu_o1,
        apply_isaacs_raman_step_cpu_o1,
        isaacs_raman_stage_cpu_o1,
    )

    if kernel_threads > available_numba_threads():
        pytest.skip(f"Numba exposes only {available_numba_threads()} local threads")
    implementation = "numba_serial" if kernel_threads == 1 else "numba_parallel"
    plan = CPUExecutionPlan(implementation=implementation, kernel_threads=kernel_threads)
    field, kwargs = _raman_candidate_fixture()
    before = field.copy()
    factory = make_joint_propagator_factory("R", kernel_threads=kernel_threads, block_pixels=32)
    _, dispatch = factory(propagate_one_pulse)
    namespace = dispatch.cpu_o1_raman_namespace
    reference_stage_function = namespace["isaacs_raman_stage"]
    reference_apply_function = namespace["apply_isaacs_raman_operator_step"]
    reference_stage = reference_stage_function(field, **kwargs)
    dz = 1.0e-5
    reference_result, reference_diagnostics = reference_apply_function(
        field, dz, **kwargs, integrator="heun", return_diagnostics=True,
        stage1=reference_stage,
    )
    reference_transform_counts = dispatch.report()["transform_counts"]
    assert reference_transform_counts.get("temporal:fft", 0) > 0
    assert reference_transform_counts.get("temporal:ifft", 0) > 0
    dispatch.counts.clear()
    candidate_stage_metrics = {}
    candidate_stage = isaacs_raman_stage_cpu_o1(
        field, raman_namespace=namespace, plan=plan,
        metrics_out=candidate_stage_metrics, **kwargs,
    )
    assert candidate_stage_metrics["implementation"] == "cpu_o1_raman_pointwise_numba"
    assert candidate_stage_metrics["product"]["kernel"] == "product"
    assert candidate_stage_metrics["rhs"]["kernel"] == "rhs"
    assert candidate_stage["I_R"].shape == (384, 5, 7)
    assert not np.shares_memory(field, candidate_stage["rhs"])
    assert not np.shares_memory(field, candidate_stage["I_R"])
    assert np.array_equal(field, before)
    _metric_record(
        {key: value for key, value in reference_stage.items() if key != "walltime_s"},
        {key: value for key, value in candidate_stage.items() if key != "walltime_s"},
        label=f"raman_candidate_stage_nt384_threads{kernel_threads}",
        settings=plan.as_dict(), measurement=candidate_stage_metrics,
        require_exact=False,
    )

    stage2_calls = []

    def candidate_stage_function(current, **stage_kwargs):
        metrics = {}
        value = isaacs_raman_stage_cpu_o1(
            current, raman_namespace=namespace, plan=plan,
            metrics_out=metrics, **stage_kwargs,
        )
        stage2_calls.append((current.copy(), value, metrics))
        return value

    candidate_apply_metrics = {}
    candidate_result, candidate_diagnostics = apply_isaacs_raman_step_cpu_o1(
        field, dz, raman_namespace=namespace, plan=plan,
        stage_function=candidate_stage_function, metrics_out=candidate_apply_metrics,
        **kwargs, integrator="heun", return_diagnostics=True,
        stage1=candidate_stage,
    )
    assert len(stage2_calls) == 1
    assert not np.array_equal(candidate_stage["I_R"], candidate_diagnostics["I_R_stage2"])
    assert candidate_apply_metrics["implementation"] == "cpu_o1_raman_heun_numba"
    assert candidate_diagnostics["convolution_count"] == 2
    assert candidate_diagnostics["finite"] is True
    assert candidate_diagnostics["actual_loss_evaluation"] == "stable_component_difference_float64"
    candidate_transform_counts = dispatch.report()["transform_counts"]
    assert candidate_transform_counts == reference_transform_counts
    assert candidate_transform_counts.get("temporal:fft", 0) > 0
    assert candidate_transform_counts.get("temporal:ifft", 0) > 0
    assert np.array_equal(field, before)
    assert np.isfinite(candidate_result).all()
    ref_ledger = {key: value for key, value in reference_diagnostics.items()
                  if key != "operator_walltime_s"}
    candidate_ledger = {key: value for key, value in candidate_diagnostics.items()
                        if key != "operator_walltime_s"}
    _metric_record(
        {"field": reference_result, "ledger": ref_ledger},
        {"field": candidate_result, "ledger": candidate_ledger},
        label=f"raman_candidate_heun_ledger_nt384_threads{kernel_threads}",
        settings=plan.as_dict(), measurement=candidate_apply_metrics,
        require_exact=False,
    )

    changed = field.copy()
    changed[::2, 1::2, 1::2] *= (1.0 - 1.0e-12 + 2.0e-12j)
    reference_energy = stable_field_fluence_and_loss(field, changed, dt=kwargs["dt"], n0=kwargs["n0"])
    energy_metrics = {}
    candidate_energy = _stable_field_fluence_and_loss_cpu_o1(
        field, changed, dt=kwargs["dt"], n0=kwargs["n0"],
        raman_namespace=namespace, plan=plan, metrics_out=energy_metrics, chunk_t=32,
    )
    assert np.isfinite(candidate_energy[0]).all() and np.isfinite(candidate_energy[1]).all()
    assert np.any(candidate_energy[1] != 0.0)
    _metric_record(
        reference_energy, candidate_energy,
        label=f"raman_candidate_stable_energy_tiny_change_threads{kernel_threads}",
        settings={**plan.as_dict(), "chunk_t": 32}, measurement=energy_metrics,
        require_exact=False,
    )


def test_raman_candidate_zero_field_keeps_complete_heun_diagnostics_exact():
    from KHz_filament import raman as raman_module
    from KHz_filament.raman import apply_isaacs_raman_operator_step, isaacs_raman_stage
    from tools.cpu_o1.cpu_o1 import apply_isaacs_raman_step_cpu_o1, isaacs_raman_stage_cpu_o1

    field, kwargs = _raman_candidate_fixture(nt=16, ny=2, nx=3, zero=True)
    namespace = vars(raman_module)
    plan = CPUExecutionPlan(implementation="numba_serial")
    reference_stage = isaacs_raman_stage(field, **kwargs)
    candidate_stage = isaacs_raman_stage_cpu_o1(
        field, raman_namespace=namespace, plan=plan, **kwargs,
    )
    _metric_record(
        {key: value for key, value in reference_stage.items() if key != "walltime_s"},
        {key: value for key, value in candidate_stage.items() if key != "walltime_s"},
        label="raman_candidate_zero_stage", settings=plan.as_dict(),
    )

    def candidate_stage_function(current, **stage_kwargs):
        return isaacs_raman_stage_cpu_o1(
            current, raman_namespace=namespace, plan=plan, **stage_kwargs,
        )

    reference_result, reference_ledger = apply_isaacs_raman_operator_step(
        field, 1.0e-5, **kwargs, integrator="heun", return_diagnostics=True,
        stage1=reference_stage,
    )
    candidate_result, candidate_ledger = apply_isaacs_raman_step_cpu_o1(
        field, 1.0e-5, raman_namespace=namespace, plan=plan,
        stage_function=candidate_stage_function, **kwargs,
        integrator="heun", return_diagnostics=True, stage1=candidate_stage,
    )
    assert candidate_ledger["finite"] is True
    assert candidate_ledger["convolution_count"] == 2
    assert np.array_equal(candidate_result, np.zeros_like(field))
    _metric_record(
        {"field": reference_result,
         "ledger": {key: value for key, value in reference_ledger.items()
                    if key != "operator_walltime_s"}},
        {"field": candidate_result,
         "ledger": {key: value for key, value in candidate_ledger.items()
                    if key != "operator_walltime_s"}},
        label="raman_candidate_zero_heun_full_ledger", settings=plan.as_dict(),
    )


def test_joint_factory_scratch_and_event_state_are_per_propagator():
    from KHz_filament.propagate import propagate_one_pulse

    reference_globals = propagate_one_pulse.__globals__
    source_evolve = reference_globals["evolve_rho_time"]
    source_stage = reference_globals["isaacs_raman_stage"]
    factory = make_joint_propagator_factory("I", kernel_threads=1)
    first, first_dispatch = factory(propagate_one_pulse)
    second, second_dispatch = factory(propagate_one_pulse)
    assert first_dispatch.cpu_o1_rate_workspace is not second_dispatch.cpu_o1_rate_workspace
    assert first._cpu_o1_candidate_events is first_dispatch.cpu_o1_candidate_events
    assert second._cpu_o1_candidate_events is second_dispatch.cpu_o1_candidate_events
    assert first._cpu_o1_candidate_events is not second._cpu_o1_candidate_events
    assert first._cpu_o1_execution["workspace_owner"] == "per_factory"
    assert first._cpu_o1_execution["temporal_workers"] == 48
    assert first._cpu_o1_execution["spatial_workers"] == 24
    assert reference_globals["evolve_rho_time"] is source_evolve
    assert reference_globals["isaacs_raman_stage"] is source_stage


def test_joint_b_i_r_ir_tiny_propagations_wire_candidates_and_preserve_topology(tmp_path):
    import importlib
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    from tools.cpu_p1r2 import cpu_p1r2_j2 as j2

    propagate_module = importlib.import_module("KHz_filament.propagate")
    outputs = {}
    factories = {}
    original_propagate = propagate_module.propagate_one_pulse
    original_stage = original_propagate.__globals__["isaacs_raman_stage"]
    for identity in ("B", "I", "R", "IR"):
        context = j1._tiny_context(operator_mode="full_isaacs_eq27")
        from KHz_filament.config import GridConfig
        from KHz_filament.grids import make_axes
        from KHz_filament.hr4e5_formal_entry import pre0_fields_from_delta_n
        from KHz_filament.runner import build_transverse_input_field

        grid = GridConfig(
            Nx=7, Ny=5, Nt=64, Lx=context["grid"].Lx,
            Ly=context["grid"].Ly, Twin=context["grid"].Twin,
        )
        axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
        source, _ = build_transverse_input_field(axes, context["beam"], np.complex128)
        context["grid"] = grid
        context["source"] = np.ascontiguousarray(source, dtype=np.complex128)
        pre0 = np.zeros((8, grid.Ny, grid.Nx), dtype=np.float64, order="C")
        context["pre_fields"] = pre0_fields_from_delta_n(pre0, expected_count=8)
        assert context["source"].shape == (64, 5, 7)
        assert context["source"].shape[0] <= 384
        assert context["source"].shape[1] <= 5
        assert context["source"].shape[2] <= 7
        case_root = tmp_path / identity
        factory = make_joint_propagator_factory(identity, kernel_threads=1, block_pixels=32)
        receipt = j1._run_window(
            source=context["source"], pre_fields=context["pre_fields"],
            grid=context["grid"], beam=context["beam"], prop=context["prop"],
            ion=context["ion"], heat=context["heat"], raman=context["raman"],
            schedule=context["schedule"], output_dir=case_root,
            source_label="synthetic_tiny_PRE0", initial_pre0=True, profile=False,
            stage_name=f"cpu_o1_{identity}", execution_scope="SOFTWARE_SMOKE_ONLY",
            observe=False, propagator_factory=factory,
        )
        assert receipt["status"] == "COMPLETE_DIAGNOSTIC"
        assert receipt["interval_count"] == 8
        dispatch = factory.last_dispatch
        assert dispatch.cpu_o1_execution["identity"] == identity
        assert dispatch.cpu_o1_execution["temporal_workers"] == 48
        assert dispatch.cpu_o1_execution["spatial_workers"] == 24
        assert factory.last_propagator._cpu_o1_candidate_events is dispatch.cpu_o1_candidate_events
        kinds = [event["kind"] for event in dispatch.cpu_o1_candidate_events]
        if identity in {"I", "IR"}:
            assert "ionization" in kinds
        if identity in {"R", "IR"}:
            assert "raman_stage" in kinds and "raman_apply" in kinds
        if identity == "B":
            assert kinds == []
        outputs[identity] = {
            "receipt": receipt,
            "root": case_root,
            "events": kinds,
            "execution": dispatch.cpu_o1_execution,
            "transform_counts": dispatch.report()["transform_counts"],
        }
        factories[identity] = factory

    baseline_raman_counts = outputs["B"]["transform_counts"]
    assert baseline_raman_counts.get("temporal:fft", 0) > 0
    assert baseline_raman_counts.get("temporal:ifft", 0) > 0
    for identity in ("I", "R", "IR"):
        assert outputs[identity]["transform_counts"] == baseline_raman_counts

    for identity in ("I", "R", "IR"):
        baseline = outputs["B"]
        candidate = outputs[identity]
        comparison = j2._compare_stage_files(
            {
                "receipt": baseline["receipt"],
                "final_E_path": baseline["root"] / "final_E.npy",
                "final_POST_path": baseline["root"] / "final_POST.npz",
                "diagnostics_path": baseline["root"] / "diagnostics.npz",
            },
            {
                "receipt": candidate["receipt"],
                "final_E_path": candidate["root"] / "final_E.npy",
                "final_POST_path": candidate["root"] / "final_POST.npz",
                "diagnostics_path": candidate["root"] / "diagnostics.npz",
            },
            label=f"B_vs_{identity}_tiny_full_operator",
        )
        array_sets = {}
        for side, item in (("reference", baseline), ("candidate", candidate)):
            array_sets[side] = {
                "final_E": {"E": np.load(item["root"] / "final_E.npy", allow_pickle=False)},
            }
            for group, filename in (("final_POST", "final_POST.npz"),
                                    ("diagnostics", "diagnostics.npz")):
                with np.load(item["root"] / filename, allow_pickle=False) as archive:
                    array_sets[side][group] = {name: archive[name].copy() for name in archive.files}
        for group, rows in comparison["metrics"].items():
            for name, metrics in rows.items():
                reference_array = array_sets["reference"].get(group, {}).get(name)
                candidate_array = array_sets["candidate"].get(group, {}).get(name)
                if reference_array is not None and candidate_array is not None:
                    metrics.update({
                        "reference_shape": list(reference_array.shape),
                        "candidate_shape": list(candidate_array.shape),
                        "reference_dtype": str(reference_array.dtype),
                        "candidate_dtype": str(candidate_array.dtype),
                    })
        assert comparison["hard_gate_pass"], comparison["hard_gate_errors"]
        _persist_metrics({
            "label": f"joint_tiny_propagation_B_vs_{identity}",
            "settings": {"identity": identity, "events": candidate["events"],
                         "execution": candidate["execution"],
                         "reference_transform_counts": baseline["transform_counts"],
                         "candidate_transform_counts": candidate["transform_counts"]},
            "measurement": None,
            "comparison": comparison,
            "exact_gate": "frozen_topology_and_diagnostic_hard_gates",
            "numerical_classification": (
                "EXACT_DIAGNOSTIC_MATCH" if comparison["canonical_exact_match"]
                else "NUMERIC_DIAGNOSTIC_ONLY / ACCEPTANCE_TOLERANCE_NOT_AUTHORIZED"
            ),
        })

    workspaces = [factories[name].last_dispatch.cpu_o1_rate_workspace for name in ("I", "IR")]
    assert workspaces[0] is not workspaces[1]
    assert propagate_module.propagate_one_pulse is original_propagate
    assert original_propagate.__globals__["isaacs_raman_stage"] is original_stage

# Q1 real adapter tests use bounded synthetic fixtures and never submit jobs.
def test_q1_real_progress_framing_and_deadlines():
    from tools.cpu_o1 import q1_t0_real as real
    rows, tail = real._complete_jsonl_lines(b'{"event":1}\n{"even')
    assert rows == [b'{"event":1}'] and tail == b'{"even'
    rows, tail = real._complete_jsonl_lines(tail + b't":2}\n')
    assert json.loads(rows[0]) == {"event":2} and tail == b''
    assert real._deadline_reason(now=901, stage_started=0, first_interval_started=0,
        first_boundary_arrived=False, timeout_seconds=2700) == 'FIRST_INTERVAL_TIMEOUT'
    assert real._deadline_reason(now=901, stage_started=0, first_interval_started=0,
        first_boundary_arrived=True, timeout_seconds=2700) is None
    assert real._deadline_reason(now=2701, stage_started=0, first_interval_started=None,
        first_boundary_arrived=False, timeout_seconds=2700) == 'WINDOW_TIMEOUT'


def test_q1_real_memory_missing_stops_and_reaps(tmp_path, monkeypatch):
    from tools.cpu_o1 import q1_t0_real as real
    from tools.cpu_p1 import cpu_p1_fft as common
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    class FakeProcess:
        pid = 123
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout): return self.returncode
    process = FakeProcess()
    monkeypatch.setattr(real.subprocess, 'Popen', lambda *a, **kw: process)
    monkeypatch.setattr(common, 'tree_snapshot', lambda pid: None)
    monkeypatch.setattr(j1, '_pss_tree_snapshot', lambda pid: {})
    monkeypatch.setattr(real, '_cgroup_memory_snapshot', lambda pid: {'status':'UNAVAILABLE'})
    def terminate(proc):
        proc.returncode = -15
        return {'test_terminated': True}
    monkeypatch.setattr(j1, '_terminate_child_bounded', terminate)
    result = real._supervise_child(['unused'], case_root=tmp_path, cwd=tmp_path,
        env={}, timeout_seconds=10, memory_bytes=1024)
    assert result['stop']['reason'] == 'SUPERVISOR_TOOL_FAILURE'
    assert 'MEMORY_TELEMETRY_UNAVAILABLE' in result['stop']['error']
    assert result['exit_code'] == -15 and (tmp_path/'supervisor_receipt.json').is_file()


def test_q1_real_storage_total_cap_and_symlink_scope(tmp_path):
    from tools.cpu_o1 import q1_t0_real as real
    (tmp_path/'existing').write_bytes(b'1234')
    with pytest.raises(RuntimeError, match='total task-root'):
        real._check_storage_growth(tmp_path, growth_cap_bytes=3, max_files=20)
    assert real._check_storage_growth(tmp_path, growth_cap_bytes=4, max_files=20)['bytes']==4
    real._validate_storage_symlinks({'symlinks':[str(tmp_path/'environment/venv/bin/python')]}, tmp_path)
    with pytest.raises(RuntimeError, match='unexpected'):
        real._validate_storage_symlinks({'symlinks':[str(tmp_path/'outputs/link')]}, tmp_path)


def test_q1_real_first_boundary_completeness_and_nonexact_stop(tmp_path):
    from tools.cpu_o1 import q1_t0_real as real
    from types import SimpleNamespace
    import shutil
    case=tmp_path/'B';case.mkdir()
    loc={'E':np.ones((4,2,3),dtype=np.complex128), 'Qacc':np.zeros((2,3)),
        'Qacc_raman':np.zeros((2,3)), 'z':0.1,'interval_index':0,'dz_try':0.1,
        'U0_baseline':1.0,'U_now':1.0,'adaptive_rejection_count':0,'safety_mode_trigger_count':0,
        'z_list':[0.1], 'thermal_slow_state':SimpleNamespace(post_fields=lambda index:
            {k:np.zeros((2,3)) for k in ('delta_n','vx','vy')})}
    snapshot=real._first_boundary_snapshot(SimpleNamespace(f_locals=loc), identity='B',
        case_root=case,observer_started=0,boundary_monotonic=1,
        diagnostic_contract={'z_axis':'z_list'})
    root=case/'first_boundary';real._write_new_json(root/'snapshot.json',snapshot)
    assert real._compare_first_boundary('B',root,None)['proceed_to_next_identity']
    candidate=tmp_path/'I';candidate.mkdir()
    for name in ('E.npy','Qacc.npy','Qacc_raman.npy','POST.npz','diagnostics.npz','scalar_diagnostics.json','snapshot.json'):
        shutil.copyfile(root/name,candidate/name)
    field=np.load(candidate/'E.npy');field[0,0,0]+=0.25;np.save(candidate/'E.npy',field)
    gate=real._compare_first_boundary('I',candidate,root)
    assert not gate['proceed_to_next_identity']
    assert gate['classification']=='FINITE_NONZERO_DIFFERENCE'
    broken=tmp_path/'missing';broken.mkdir()
    bad=dict(loc);bad.pop('Qacc')
    receipt=real._first_boundary_snapshot(SimpleNamespace(f_locals=bad), identity='B',
        case_root=broken,observer_started=0,boundary_monotonic=1,
        diagnostic_contract={'z_axis':'z_list','missing':'missing_list'})
    assert not receipt['complete_E_POST_diagnostics_snapshot']
    assert receipt['diagnostic_missing_fields']==['missing']


def test_q1_real_full_observer_preserves_frozen_tiny_result(tmp_path, monkeypatch):
    from tools.cpu_o1 import q1_t0_real as real
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    from importlib import import_module
    propagation_module = import_module("KHz_filament.propagate")
    context=j1._tiny_context()
    case=tmp_path/'observed';case.mkdir()
    contract=real._frozen_diag_list_contract(Path(propagation_module.__file__))
    observer=real._observer_class(j1,identity='B',case_root=case,
        baseline_root=None,diagnostic_contract=contract)
    run=real._private_run_window(j1,observer)
    assert j1._run_window.__globals__['ScalarBoundaryObserver'] is j1.ScalarBoundaryObserver
    def execute(call, target, observe, identity="B"):
        factory=make_joint_propagator_factory(identity,kernel_threads=1,block_pixels=32,layout='NtB',
            temporal_workers=1,spatial_workers=1)
        return call(**{k:context[k] for k in ('source','pre_fields','grid','beam','prop','ion','heat','raman','schedule')},
            output_dir=target,source_label='SYNTHETIC_TEST_ONLY',initial_pre0=True,profile=False,
            source_hash=j1._canonical_hash(context['source']),pre0_hash=j1._canonical_hash(context['pre_fields']['delta_n']),
            stage_name='Q1_OBSERVER_TEST',execution_scope='SOFTWARE_SMOKE_ONLY',observe=observe,propagator_factory=factory)
    execute(run,case/'propagation',True)
    execute(j1._run_window,tmp_path/'unobserved',False)
    assert observer.last_instance.q1_first_gate['proceed_to_next_identity'] is True
    left=np.load(case/'propagation/final_E.npy');right=np.load(tmp_path/'unobserved/final_E.npy')
    assert compare_numeric_bounded(left,right)['bitwise_equal']
    assert len(observer.last_instance.rows)==8

    # Exercise each private candidate and reject at the first complete boundary.
    for identity in ("I", "R", "IR"):
        target=tmp_path/identity;target.mkdir()
        candidate_observer=real._observer_class(j1,identity=identity,case_root=target,
            baseline_root=case/"first_boundary",diagnostic_contract=contract)
        execute(real._private_run_window(j1,candidate_observer),target/"propagation",True,identity)
        assert candidate_observer.last_instance.q1_first_gate["proceed_to_next_identity"]
        assert len(candidate_observer.last_instance.rows)==8
    original_snapshot=real._first_boundary_snapshot
    def perturbed(*args,**kwargs):
        result=original_snapshot(*args,**kwargs)
        field_path=Path(result["root"])/"E.npy"
        field=np.load(field_path);field.flat[0]+=1.0;np.save(field_path,field)
        return result
    monkeypatch.setattr(real,"_first_boundary_snapshot",perturbed)
    stopped=tmp_path/"stopped";stopped.mkdir()
    rejecting=real._observer_class(j1,identity="I",case_root=stopped,
        baseline_root=case/"first_boundary",diagnostic_contract=contract)
    with pytest.raises(real.Q1ControlledStop):
        execute(real._private_run_window(j1,rejecting),stopped/"propagation",True,"I")
    assert len(rejecting.last_instance.rows)==1
    assert not (stopped/"propagation/final_E.npy").exists()
    assert rejecting.last_instance.q1_first_gate["classification"]=="FINITE_NONZERO_DIFFERENCE"
    assert j1._run_window.__globals__["ScalarBoundaryObserver"] is j1.ScalarBoundaryObserver


def test_q1_real_allocation_probe_requires_complete_lifecycle(tmp_path,monkeypatch):
    from tools.cpu_o1 import q1_t0_real as real
    from tools.cpu_p1r2 import allocation_probe
    def probe(argv):
        target=Path(argv[1]);target.mkdir()
        (target/"receipt.json").write_text(json.dumps({"status":"PASS"}))
        return 0
    monkeypatch.setattr(allocation_probe,"main",probe)
    assert real._run_allocation_probe(tmp_path)["status"]=="PASS"
    def failed(argv):
        target=Path(argv[1]);target.mkdir()
        (target/"receipt.json").write_text(json.dumps({"status":"FAIL"}))
        return 1
    monkeypatch.setattr(allocation_probe,"main",failed)
    another=tmp_path/"failure";another.mkdir()
    with pytest.raises(RuntimeError,match="startup/wait/cleanup"):
        real._run_allocation_probe(another)

def test_q1_real_stage_identity_requires_explicit_schedule(tmp_path,monkeypatch):
    from tools.cpu_o1 import q1_t0_real as real
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    monkeypatch.setattr(j1,"_validate_stage",lambda path:{"status":"VALIDATED"})
    value={"identity":"B","status":"COMPLETE_DIAGNOSTIC","propagation_receipt":{
        "execution_scope":"J1_REAL_INPUT","interval_count":8,"source_canonical_sha256":"source","pre0_canonical_sha256":"pre"}}
    path=tmp_path/"q1_child_receipt.json"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match="real complete"):
        real._check_stage_identity("B",tmp_path,None)
    value["schedule"]={"z_edges":[0,1,2,3,4,5,6,7,8]}
    path.write_text(json.dumps(value))
    assert real._check_stage_identity("B",tmp_path,None)["validation"]["status"]=="VALIDATED"
    baseline=dict(value);baseline["schedule"]={"z_edges":[0,2,4,6,8,10,12,14,16]}
    with pytest.raises(RuntimeError,match="schedule changed"):
        real._check_stage_identity("B",tmp_path,baseline)

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
    evolve_rho_time_cpu_o1,
    workspace_matches,
)


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
    """Check return shape/keys without imposing a cross-implementation tolerance."""
    if isinstance(reference, np.ndarray):
        assert isinstance(candidate, np.ndarray), path
        assert reference.shape == candidate.shape, path
        assert reference.dtype == candidate.dtype, path
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
    if isinstance(reference, (int, float, np.number)):
        assert isinstance(candidate, (int, float, np.number)), path
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


def _metric_record(reference, candidate, *, label, settings, measurement=None):
    result = {}
    def visit(ref_value, candidate_value, key):
        if isinstance(ref_value, np.ndarray):
            result[key] = _metric_fields(compare_numeric(ref_value, candidate_value))
        elif isinstance(ref_value, dict):
            for name, nested in ref_value.items():
                visit(nested, candidate_value[name], f"{key}.{name}")
        elif isinstance(ref_value, tuple):
            for index, (left, right) in enumerate(zip(ref_value, candidate_value)):
                visit(left, right, f"{key}[{index}]")

    visit(reference, candidate, "result")
    _persist_metrics({
        "label": label,
        "settings": settings,
        "measurement": measurement,
        "array_metrics": result,
    })


def _metric_fields(metric):
    fields = {
        name: metric[name]
        for name in (
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

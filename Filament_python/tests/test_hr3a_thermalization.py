from __future__ import annotations

import numpy as np
import pytest


def _interval_energy(q, dx, dy, dz):
    return np.sum(q, axis=(1, 2)) * dx * dy * np.asarray(dz)


def _inputs(*, raman_authoritative=True, ib_active=False):
    q_ion = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    q_ib = np.zeros_like(q_ion)
    q_raman = np.full_like(q_ion, 0.5)
    dz = np.array([0.25, 0.5])
    dx, dy = 0.2, 0.3
    return dict(
        q_ion=q_ion,
        q_ib=q_ib,
        q_raman=q_raman,
        z_edges=np.array([0.0, 0.25, 0.75]),
        dz_intervals=dz,
        dx=dx,
        dy=dy,
        deposition_mechanisms={
            "ion": {"active": True, "authoritative": True, "source": "positive_photoionization_energy_rate"},
            "ib": {"active": ib_active, "authoritative": True, "source": "alpha_ib_times_intensity" if ib_active else "off"},
            "raman": {"active": True, "authoritative": raman_authoritative, "source": "eq10_heun_positive_rotational_energy" if raman_authoritative else "legacy_unavailable"},
        },
        deposition_interval_J={
            "ion": _interval_energy(q_ion, dx, dy, dz),
            "ib": _interval_energy(q_ib, dx, dy, dz),
            "raman": _interval_energy(q_raman, dx, dy, dz),
        },
    )


def test_complete_thermalization_is_channel_identity_and_closes_reductions():
    from KHz_filament.thermalization import build_complete_thermalization_ledger

    inputs = _inputs()
    ledger = build_complete_thermalization_ledger(**inputs)

    assert ledger["authoritative"]
    assert ledger["unavailable_reason"] == ""
    np.testing.assert_array_equal(ledger["q_th_ion"], inputs["q_ion"])
    np.testing.assert_array_equal(ledger["q_th_ib"], inputs["q_ib"])
    np.testing.assert_array_equal(ledger["q_th_raman"], inputs["q_raman"])
    np.testing.assert_array_equal(
        ledger["q_thermal"], inputs["q_ion"] + inputs["q_ib"] + inputs["q_raman"]
    )
    assert set(ledger["level_t1"].values()) == {"pass"}
    assert set(ledger["level_t2"].values()) == {"pass"}
    assert ledger["level_t3"] == "pass"
    assert ledger["thermalization_t1_status"] == "pass"
    assert ledger["thermalization_t2_status"] == "pass"
    assert ledger["thermalization_t3_status"] == "pass"
    assert ledger["thermalization_first_failed_level"] == ""
    assert ledger["E_thermal_pulse_J"] == pytest.approx(
        ledger["E_th_ion_pulse_J"] + ledger["E_th_ib_pulse_J"] + ledger["E_th_raman_pulse_J"]
    )


def test_inactive_ib_stays_exact_zero():
    from KHz_filament.thermalization import build_complete_thermalization_ledger

    ledger = build_complete_thermalization_ledger(**_inputs(ib_active=False))
    assert not ledger["mechanisms"]["ib"]["active"]
    assert ledger["zero_channel_pass"]
    assert np.array_equal(ledger["q_th_ib"], np.zeros_like(ledger["q_th_ib"]))
    assert np.array_equal(ledger["E_th_ib_interval_J"], np.zeros(2))
    assert ledger["E_th_ib_pulse_J"] == 0.0
    assert ledger["level_t1"]["ib"] == "pass"
    assert ledger["level_t2"]["ib"] == "pass"


def test_non_authoritative_active_channel_is_marked_unavailable_without_fallback():
    from KHz_filament.thermalization import build_complete_thermalization_ledger

    ledger = build_complete_thermalization_ledger(**_inputs(raman_authoritative=False))
    assert not ledger["authoritative"]
    assert "raman:legacy_unavailable" in ledger["unavailable_reason"]
    assert np.isnan(ledger["q_thermal"]).all()
    assert ledger["level_t1"]["raman"] == "unavailable"
    assert ledger["level_t3"] == "unavailable"
    assert ledger["thermalization_t1_status"] == "unavailable"
    assert ledger["thermalization_t2_status"] == "unavailable"
    assert ledger["thermalization_t3_status"] == "unavailable"
    assert ledger["thermalization_first_failed_level"] == "T1"
    assert np.isnan(ledger["E_thermal_pulse_J"])


@pytest.mark.parametrize("field", ["q_ion", "q_ib", "q_raman"])
def test_missing_or_nonfinite_authoritative_input_is_rejected(field):
    from KHz_filament.thermalization import build_complete_thermalization_ledger

    missing = _inputs()
    missing[field] = None
    with pytest.raises(ValueError, match="required"):
        build_complete_thermalization_ledger(**missing)

    invalid = _inputs()
    invalid[field] = invalid[field].copy()
    invalid[field][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_complete_thermalization_ledger(**invalid)


def test_schedule_mismatch_and_inactive_nonzero_input_are_rejected():
    from KHz_filament.thermalization import build_complete_thermalization_ledger

    mismatch = _inputs()
    mismatch["z_edges"] = np.array([0.0, 0.3, 0.75])
    with pytest.raises(ValueError, match="interval geometry"):
        build_complete_thermalization_ledger(**mismatch)

    inactive_nonzero = _inputs()
    inactive_nonzero["q_ib"] = np.ones_like(inactive_nonzero["q_ib"])
    inactive_nonzero["deposition_interval_J"]["ib"] = _interval_energy(
        inactive_nonzero["q_ib"],
        inactive_nonzero["dx"],
        inactive_nonzero["dy"],
        inactive_nonzero["dz_intervals"],
    )
    with pytest.raises(ValueError, match="inactive ib"):
        build_complete_thermalization_ledger(**inactive_nonzero)


def test_failed_reduction_closure_is_not_authoritative():
    from KHz_filament.thermalization import build_complete_thermalization_ledger

    inputs = _inputs()
    inputs["deposition_interval_J"]["ion"] = np.array([0.0, 0.0])
    ledger = build_complete_thermalization_ledger(**inputs)
    assert not ledger["authoritative"]
    assert ledger["unavailable_reason"] == "t2_reduction_closure_failed"
    assert ledger["thermalization_t1_status"] == "pass"
    assert ledger["thermalization_t2_status"] == "failed"
    assert ledger["thermalization_t3_status"] == "pass"
    assert ledger["thermalization_first_failed_level"] == "T2"
    assert ledger["level_t2"]["ion"] == "failed"
    assert np.isfinite(ledger["q_thermal"]).all()
    np.testing.assert_allclose(
        ledger["E_thermal_interval_J"],
        ledger["E_th_ion_interval_J"]
        + ledger["E_th_ib_interval_J"]
        + ledger["E_th_raman_interval_J"],
    )


def test_scalar_ledger_keeps_t3_independent_of_synthetic_t3_failure():
    from KHz_filament.thermalization import (
        ThermalScalarLedger,
        thermalize_interval,
    )

    inputs = _inputs()
    result = thermalize_interval(
        q_ion=inputs["q_ion"][0], q_ib=inputs["q_ib"][0],
        q_raman=inputs["q_raman"][0], dz=inputs["dz_intervals"][0],
        dx=inputs["dx"], dy=inputs["dy"],
        mechanisms=inputs["deposition_mechanisms"],
        reference_interval_J={
            name: inputs["deposition_interval_J"][name][0]
            for name in ("ion", "ib", "raman")
        },
    )
    synthetic = dict(result)
    synthetic.update(
        authoritative=False,
        reason="t3_channel_sum_closure_failed",
        t3_ok=False,
        t3_status="failed",
        t3=1.0,
    )
    ledger = ThermalScalarLedger(inputs["deposition_mechanisms"])
    ledger.append(0, synthetic)
    summary = ledger.as_dict()

    assert summary["thermalization_t1_status"] == "pass"
    assert summary["thermalization_t2_status"] == "pass"
    assert summary["thermalization_t3_status"] == "failed"
    assert not summary["thermalization_authoritative"]
    assert summary["thermalization_first_failed_level"] == "T3"


def test_physical_sample_plan_uses_midpoints_and_deduplicates_intervals():
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.thermalization import build_physical_sample_plan

    schedule = build_longitudinal_schedule(
        0.003, 0.020, focus_window_step=True,
        focus_center_m=0.010, focus_halfwidth_m=0.003, dz_focus=0.001,
    )
    plan = build_physical_sample_plan(
        schedule, focus_center_m=0.010, focus_halfwidth_m=0.003,
        focus_enabled=True, focal_plane_m=0.010,
    )
    assert len(plan.interval_index) == len(np.unique(plan.interval_index))
    assert plan.interval_index[0] == 0
    assert plan.interval_index[-1] == schedule.n_intervals - 1
    assert any("focus" in value for value in plan.region)
    np.testing.assert_allclose(
        plan.z_mid_m, 0.5 * (plan.z_left_m + plan.z_right_m)
    )


def test_memmap_sink_streams_each_slot_once_and_reopens(tmp_path):
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.thermalization import ThermalDiagnosticSink, build_physical_sample_plan

    schedule = build_longitudinal_schedule(0.005, 0.010)
    plan = build_physical_sample_plan(
        schedule, focus_center_m=None, focus_halfwidth_m=0.0,
        focus_enabled=False, focal_plane_m=None,
    )
    sink = ThermalDiagnosticSink(
        plan=plan, output_path=str(tmp_path / "tiny.npz"), shape=(2, 3),
        dtype=np.float32, enabled=True,
    )
    for interval in plan.interval_index:
        sink.record_sample(int(interval), np.full((2, 3), interval, dtype=np.float32))
    with pytest.raises(ValueError, match="only once"):
        sink.record_sample(int(plan.interval_index[0]), np.zeros((2, 3), dtype=np.float32))
    meta = sink.finalize()
    archive = tmp_path / meta["thermal_map_archive_filename"]
    reopened = np.lib.format.open_memmap(archive, mode="r")
    assert meta["thermal_map_archive_complete"]
    assert reopened.shape == (plan.count, 2, 3)
    assert reopened.dtype == np.float32

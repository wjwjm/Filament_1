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


def test_non_authoritative_active_channel_is_marked_unavailable_without_fallback():
    from KHz_filament.thermalization import build_complete_thermalization_ledger

    ledger = build_complete_thermalization_ledger(**_inputs(raman_authoritative=False))
    assert not ledger["authoritative"]
    assert "raman:legacy_unavailable" in ledger["unavailable_reason"]
    assert np.isnan(ledger["q_thermal"]).all()
    assert ledger["level_t1"]["raman"] == "unavailable"
    assert ledger["level_t3"] == "unavailable"


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
    assert ledger["unavailable_reason"] == "thermalization_closure_failed"
    assert ledger["level_t2"]["ion"] == "failed"

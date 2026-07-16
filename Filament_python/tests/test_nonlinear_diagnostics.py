from __future__ import annotations

import json

import numpy as np


def test_nonlinear_diagnostic_validation_rejects_bad_trace_length():
    from KHz_filament.diagnostics import Z_HISTORY_TRACE_KEYS, validate_nonlinear_diagnostics

    diag = {"z_axis": np.array([1.0, 2.0])}
    for key in Z_HISTORY_TRACE_KEYS:
        diag[key] = np.zeros(2)
    diag["U_z"] = np.array([1.0, 1.0])
    diag["I_max_z"] = np.ones(2)
    diag["IR_max_z"] = np.zeros(1)

    try:
        validate_nonlinear_diagnostics(diag)
    except ValueError as exc:
        assert "IR_max_z" in str(exc)
    else:
        raise AssertionError("length mismatch must fail diagnostic validation")


def test_minimal_run_saves_complete_nonlinear_diagnostics(tmp_path):
    from KHz_filament.config import (
        BeamConfig,
        GridConfig,
        HeatConfig,
        IonizationConfig,
        PropagationConfig,
        RamanConfig,
        RunConfig,
    )
    from KHz_filament.diagnostics import Z_HISTORY_TRACE_KEYS
    from KHz_filament.runner import run_demo

    out_path = tmp_path / "minimal.npz"
    run_demo(
        grid=GridConfig(Nx=8, Ny=8, Nt=16, Lx=8e-4, Ly=8e-4, Twin=160e-15),
        beam=BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-9, focal_length=None),
        prop=PropagationConfig(
            z_max=2e-4,
            dz=1e-4,
            linear_model="paraxial",
            auto_substep=False,
            focus_window_step=False,
            limit_focus_window=False,
            progress_every_z=0,
            diag_extra=False,
            energy_probe_every=0,
        ),
        ion=IonizationConfig(species=[{"name": "test", "rate": "mpa_fact", "ell": 2, "I_mp": 1e18, "Ip_eV": 15.0, "fraction": 1.0}]),
        heat=HeatConfig(f_rep=1e3),
        run=RunConfig(Npulses=1),
        raman=RamanConfig(enabled=True, absorption=True, absorption_model="closed_form"),
        out_path=str(out_path),
        dtype="fp32",
    )

    assert out_path.is_file()
    report_path = out_path.with_suffix(".diagnostic_report.json")
    assert report_path.is_file()
    with np.load(out_path, allow_pickle=False) as data:
        n_records = data["z_axis"].size
        assert n_records == 2
        for key in Z_HISTORY_TRACE_KEYS:
            values = data[key]
            assert values.shape == (n_records,)
            assert np.all(np.isfinite(values))
        assert bool(data["diagnostic_validation_passed"])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["validation"]["passed"] is True
    assert report["validation"]["z_records"] == 2
    assert "delta_n_elec_max_z" in report["variables"]
    zero_traces = set(report["validation"]["all_zero_traces"])
    # This smoke configuration enables electronic Kerr, Raman convolution and
    # an MPA ionization channel; their histories must not disappear into a
    # spurious all-zero trace.  IB remains physically zero because sigma_ib=0.
    assert not {
        "delta_n_elec_max_z",
        "delta_n_rot_max_z",
        "IR_max_z",
        "dphi_kerr_max_abs_z",
        "dphi_plasma_max_abs_z",
        "alpha_ion_corr_max_z",
        "alpha_R_eff_z",
    }.intersection(zero_traces)

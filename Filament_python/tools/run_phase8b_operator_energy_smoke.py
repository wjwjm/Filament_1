#!/usr/bin/env python3
"""Run a local one-step full-operator smoke for opt-in energy diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.config import (  # noqa: E402
    BeamConfig, GridConfig, HeatConfig, IonizationConfig, PropagationConfig,
    RamanConfig, RunConfig,
)
from KHz_filament.runner import run_demo  # noqa: E402


def run_smoke(output: Path, scratch_npz: Path) -> dict:
    scratch_npz.parent.mkdir(parents=True, exist_ok=True)
    run_demo(
        grid=GridConfig(Nx=8, Ny=8, Nt=64, Lx=8e-4, Ly=8e-4, Twin=640e-15),
        beam=BeamConfig(w0=1.5e-4, tau_fwhm=120e-15, energy_J=1e-8, P0_peak=None, focal_length=None),
        prop=PropagationConfig(
            z_max=1e-5, dz=1e-5, linear_model="paraxial", auto_substep=False,
            focus_window_step=False, limit_focus_window=False, progress_every_z=0,
            energy_probe_every=0, diag_extra=False, measure_performance=False,
            diag_operator_energy=True, use_self_steepening=False,
            use_electronic_kerr=False, use_raman_phase=False,
            use_raman_full_operator=True, use_raman_absorption=False,
            use_plasma_phase=False, use_ionization_loss=False,
            use_ionization_solver=False,
        ),
        ion=IonizationConfig(species=[]), heat=HeatConfig(), run=RunConfig(Npulses=1),
        raman=RamanConfig(
            enabled=True, model="isaacs_rot_sinexp", f_R=None, n_R=2.3e-23,
            omega_R=1.6e13, Gamma_R=1.3e13, T_R=None, T2=None,
            operator_mode="full_isaacs_eq27", operator_convention="isaacs_eq27",
            iir_sampling="exact_piecewise_linear", operator_integrator="heun",
            absorption=False, nonlinear_split_order="strang",
        ),
        out_path=str(scratch_npz), dtype="fp64",
    )
    with np.load(scratch_npz, allow_pickle=False) as data:
        keys = (
            "energy_step_start_J", "energy_after_linear_half1_J",
            "energy_after_raman_pre_J", "energy_after_nonraman_J",
            "energy_after_raman_post_J", "energy_after_linear_half2_J",
        )
        values = {key: np.asarray(data[key], dtype=np.float64) for key in keys}
        delta_sum = (
            (values["energy_after_linear_half1_J"] - values["energy_step_start_J"])
            + (values["energy_after_raman_pre_J"] - values["energy_after_linear_half1_J"])
            + (values["energy_after_nonraman_J"] - values["energy_after_raman_pre_J"])
            + (values["energy_after_raman_post_J"] - values["energy_after_nonraman_J"])
            + (values["energy_after_linear_half2_J"] - values["energy_after_raman_post_J"])
        )
        net = values["energy_after_linear_half2_J"] - values["energy_step_start_J"]
        u = np.asarray(data["U_z"], dtype=np.float64)
        u_step = np.asarray(data["U_step_change_z"], dtype=np.float64)
        u0 = float(u[0] - u_step[0])
        total_closure = abs((u0 - u[-1]) - float(np.asarray(data["E_dep_cumulative_z"], dtype=np.float64)[-1])) / u0
        result = {
            "schema": "khz_filament.phase8b_r.operator_energy_smoke.v1",
            "execution": {"backend": "numpy", "full_job_submitted": False, "job2_prepared": False, "job2_submitted": False},
            "configuration": {"operator_mode": str(data["raman_operator_mode"].item()), "split_order": "strang", "integrator": "heun"},
            "checks": {
                "operator_energy_diagnostics_enabled": bool(data["operator_energy_diagnostics_enabled"].item()),
                "operator_energy_histories_finite": bool(all(np.all(np.isfinite(value)) for value in values.values())),
                "operator_energy_telescope_max_abs_J": float(np.max(np.abs(delta_sum - net))),
                "total_energy_closure": total_closure,
                "total_energy_closure_threshold": 0.01,
                "raman_step_closure_p99": float(np.percentile(np.asarray(data["raman_closure_residual_step"], dtype=float), 99)),
                "raman_cumulative_closure": float(np.asarray(data["raman_cumulative_closure_residual"], dtype=float)[-1]),
                "legacy_alpha_R_max": float(np.max(np.abs(np.asarray(data["alpha_R_applied_max_z"], dtype=float)))),
                "raman_substeps": sorted(set(np.asarray(data["raman_operator_substep_count"], dtype=int).tolist())),
                "raman_convolutions": sorted(set(np.asarray(data["raman_convolution_count_step"], dtype=int).tolist())),
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "isaacs_raman_closure" / "phase8b_controlled_propagation" / "r3_operator_energy_smoke.json")
    parser.add_argument("--scratch-npz", type=Path, default=ROOT / ".." / "tmp" / "phase8b_r3_operator_energy_smoke.npz")
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(args.output, args.scratch_npz), indent=2))


if __name__ == "__main__":
    main()

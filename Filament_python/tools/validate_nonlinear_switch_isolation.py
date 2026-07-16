#!/usr/bin/env python3
"""Run CPU smoke checks proving each Phase-2 nonlinear switch is isolated."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
if str(FILAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(FILAMENT_ROOT))

from KHz_filament.config import (  # noqa: E402
    BeamConfig,
    GridConfig,
    HeatConfig,
    IonizationConfig,
    PropagationConfig,
    RamanConfig,
    RunConfig,
)
from KHz_filament.runner import run_demo  # noqa: E402


def _components(*, self_steepening: bool = False, **switches: bool) -> dict[str, Any]:
    return {
        "grid": GridConfig(Nx=8, Ny=8, Nt=16, Lx=8e-4, Ly=8e-4, Twin=160e-15),
        "beam": BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-9, focal_length=None),
        "prop": PropagationConfig(
            z_max=2e-4,
            dz=1e-4,
            linear_model="paraxial",
            auto_substep=False,
            focus_window_step=False,
            limit_focus_window=False,
            progress_every_z=0,
            diag_extra=False,
            energy_probe_every=0,
            use_self_steepening=self_steepening,
            **switches,
        ),
        "ion": IonizationConfig(species=[{
            "name": "test",
            "rate": "mpa_fact",
            "ell": 2,
            "I_mp": 1e18,
            "Ip_eV": 15.0,
            "fraction": 1.0,
        }]),
        "heat": HeatConfig(f_rep=1e3),
        "run": RunConfig(Npulses=1),
        "raman": RamanConfig(enabled=True, absorption=True, absorption_model="closed_form"),
    }


def _run_case(directory: Path, name: str, *, self_steepening: bool = False, **switches: bool) -> dict[str, np.ndarray]:
    out_path = directory / f"{name}.npz"
    run_demo(**_components(self_steepening=self_steepening, **switches), out_path=str(out_path), dtype="fp32")
    with np.load(out_path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _max(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _check(name: str, passed: bool, **metrics: float | bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "metrics": metrics}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=FILAMENT_ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()


def run_switch_isolation_smoke() -> dict[str, Any]:
    """Return a lightweight, auditable report without retaining raw NPZ files."""
    with tempfile.TemporaryDirectory(prefix="khzfil_switch_isolation_") as temp:
        tmp = Path(temp)
        baseline = _run_case(tmp, "legacy_default", self_steepening=True)
        explicit_full = _run_case(
            tmp,
            "explicit_full",
            self_steepening=True,
            use_electronic_kerr=True,
            use_raman_phase=True,
            use_plasma_phase=True,
            use_ionization_loss=True,
            use_raman_absorption=True,
            use_ionization_solver=True,
        )
        default_match = all(
            np.allclose(baseline[key], explicit_full[key], rtol=2e-6, atol=1e-12)
            for key in ("I_out_center_t", "U_z", "rho_max_z", "dphi_kerr_max_abs_z", "alpha_total_max_z")
        )

        electronic_off = _run_case(tmp, "electronic_off", use_electronic_kerr=False)
        raman_phase_off = _run_case(tmp, "raman_phase_off", use_raman_phase=False, use_raman_absorption=True)
        raman_absorption_off = _run_case(tmp, "raman_absorption_off", use_raman_phase=True, use_raman_absorption=False)
        plasma_off = _run_case(tmp, "plasma_off", use_plasma_phase=False)
        ionization_loss_off = _run_case(tmp, "ionization_loss_off", use_ionization_loss=False)

    checks = [
        _check("default_full_model_regression", default_match),
        _check(
            "electronic_kerr_off",
            _max(electronic_off["delta_n_elec_max_z"]) > 0.0
            and _max(electronic_off["delta_n_elec_applied_max_z"]) == 0.0
            and _max(electronic_off["dphi_elec_applied_max_abs_z"]) == 0.0,
            delta_n_elec_raw_max=_max(electronic_off["delta_n_elec_max_z"]),
            delta_n_elec_applied_max=_max(electronic_off["delta_n_elec_applied_max_z"]),
        ),
        _check(
            "raman_phase_off_with_absorption_on",
            _max(raman_phase_off["IR_abs_max_z"]) > 0.0
            and _max(raman_phase_off["delta_n_rot_max_z"]) > 0.0
            and _max(raman_phase_off["delta_n_rot_applied_max_z"]) == 0.0
            and _max(raman_phase_off["alpha_R_applied_max_z"]) > 0.0,
            raman_convolution_max=_max(raman_phase_off["IR_abs_max_z"]),
            raman_phase_applied_max=_max(raman_phase_off["delta_n_rot_applied_max_z"]),
            raman_absorption_applied_max=_max(raman_phase_off["alpha_R_applied_max_z"]),
        ),
        _check(
            "raman_absorption_off_with_phase_on",
            _max(raman_absorption_off["delta_n_rot_applied_max_z"]) > 0.0
            and _max(raman_absorption_off["alpha_R_raw_max_z"]) > 0.0
            and _max(raman_absorption_off["alpha_R_applied_max_z"]) == 0.0,
            raman_phase_applied_max=_max(raman_absorption_off["delta_n_rot_applied_max_z"]),
            raman_absorption_raw_max=_max(raman_absorption_off["alpha_R_raw_max_z"]),
            raman_absorption_applied_max=_max(raman_absorption_off["alpha_R_applied_max_z"]),
        ),
        _check(
            "plasma_phase_off",
            _max(plasma_off["rho_max_z"]) > 0.0
            and _max(plasma_off["dphi_plasma_raw_max_abs_z"]) > 0.0
            and _max(plasma_off["dphi_plasma_applied_max_abs_z"]) == 0.0,
            rho_max=_max(plasma_off["rho_max_z"]),
            plasma_phase_raw_max=_max(plasma_off["dphi_plasma_raw_max_abs_z"]),
            plasma_phase_applied_max=_max(plasma_off["dphi_plasma_applied_max_abs_z"]),
        ),
        _check(
            "ionization_loss_off",
            _max(ionization_loss_off["rho_max_z"]) > 0.0
            and _max(ionization_loss_off["alpha_ion_raw_max_z"]) > 0.0
            and _max(ionization_loss_off["alpha_ion_applied_max_z"]) == 0.0,
            rho_max=_max(ionization_loss_off["rho_max_z"]),
            alpha_ion_raw_max=_max(ionization_loss_off["alpha_ion_raw_max_z"]),
            alpha_ion_applied_max=_max(ionization_loss_off["alpha_ion_applied_max_z"]),
        ),
    ]
    return {
        "schema": "khz_filament.nonlinear_switch_isolation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_sha": _git_sha(),
        "execution": {"backend": "cpu", "grid": {"Nx": 8, "Ny": 8, "Nt": 16}, "z_max_m": 2e-4, "saved_raw_npz": False},
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output JSON report; no NPZ is retained")
    args = parser.parse_args()
    report = run_switch_isolation_smoke()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

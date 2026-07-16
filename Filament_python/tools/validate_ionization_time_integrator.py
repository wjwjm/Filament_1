#!/usr/bin/env python3
"""0D harness for production ionization-rate and time-integrator validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
if str(FILAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(FILAMENT_ROOT))

from KHz_filament.config import IonizationConfig  # noqa: E402
from KHz_filament.confio import load_all  # noqa: E402
from KHz_filament.constants import N0_air, c0  # noqa: E402
from KHz_filament.device import to_cpu  # noqa: E402
from KHz_filament.grids import make_axes  # noqa: E402
from KHz_filament.ionization import evolve_rho_time, make_Wfunc  # noqa: E402
from KHz_filament.utils import gaussian_pulse_t  # noqa: E402


DEFAULT_CONFIGS = (
    FILAMENT_ROOT / "configs" / "profile_validation" / "flat_top_90_40fs.json",
    FILAMENT_ROOT / "configs" / "profile_validation" / "flat_top_90_120fs.json",
)
DEFAULT_INTENSITIES_W_M2 = (1e15, 3e15, 1e16, 3e16, 1e17, 3e17, 1e18)


@dataclass(frozen=True)
class IonizationCase:
    config_path: Path
    tau_fwhm_s: float
    case_label: str
    t_s: np.ndarray
    dt_s: float
    I_peak_W_m2: float
    I_t_W_m2: np.ndarray
    W_by_species_s: dict[str, np.ndarray]
    rho_by_species_m3: dict[str, np.ndarray]
    rho_total_m3: np.ndarray
    species_fractions: dict[str, float]
    species_rates: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=FILAMENT_ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()


def _case_label(config_path: Path, tau_fwhm_s: float, intensity: float) -> str:
    tau_fs = round(tau_fwhm_s * 1e15)
    return f"tau{tau_fs:d}fs_I{intensity:.3e}".replace("+", "p").replace("-", "m").replace(".", "p")


def _species_fraction(species: list[dict[str, Any]]) -> list[float]:
    raw = [max(0.0, float(item.get("fraction", 1.0))) for item in species]
    total = sum(raw)
    if total <= 0.0:
        raise ValueError("ionization species fractions must sum to a positive value")
    return [value / total for value in raw]


def _single_species_ionization_config(ion: IonizationConfig, species: dict[str, Any]) -> IonizationConfig:
    """Create a production-config-compatible one-species view for reporting."""
    result = deepcopy(ion)
    one_species = deepcopy(species)
    one_species["fraction"] = 1.0
    result.species = [one_species]
    return result


def build_intensity_envelope(t_s: np.ndarray, tau_fwhm_s: float, I_peak_W_m2: float) -> np.ndarray:
    """Return the production Gaussian intensity envelope in W/m^2.

    ``gaussian_pulse_t`` returns the field envelope.  Squaring its magnitude
    gives the intensity provided to the ionization evaluator.  With the
    current implementation the input ``tau_fwhm`` is the *intensity* FWHM.
    """
    field_envelope = np.asarray(to_cpu(gaussian_pulse_t(t_s, tau_fwhm_s)))[:, 0, 0]
    return float(I_peak_W_m2) * np.abs(field_envelope) ** 2


def run_production_0d_case(config_path: Path, I_peak_W_m2: float) -> IonizationCase:
    """Run the production RK4/evaluator on one temporal point (no 3D propagation)."""
    grid, beam, _prop, ion, _heat, _run, _raman = load_all(str(config_path))
    if str(ion.time_mode).lower() != "full":
        raise ValueError(f"0D integrator validation requires ionization.time_mode='full': {config_path}")
    if not ion.species:
        raise ValueError(f"0D integrator validation requires non-empty ionization.species: {config_path}")
    if float(I_peak_W_m2) <= 0.0:
        raise ValueError("I_peak_W_m2 must be positive")

    axes = make_axes(1, 1, int(grid.Nt), 1.0, 1.0, float(grid.Twin))
    t_s = np.asarray(to_cpu(axes.t), dtype=np.float64)
    I_t = build_intensity_envelope(t_s, float(beam.tau_fwhm), float(I_peak_W_m2))
    I_3d = I_t[:, None, None]
    omega0 = 2.0 * math.pi * float(c0) / float(beam.lam0)
    Wfunc = make_Wfunc("production_0d", ion, omega0, float(beam.n0))
    rho_total, _Wt = evolve_rho_time(I_3d, axes.dt, N0_air, float(ion.beta_rec), Wfunc)

    species = [dict(item) for item in ion.species]
    fractions = _species_fraction(species)
    W_by_species: dict[str, np.ndarray] = {}
    rho_by_species: dict[str, np.ndarray] = {}
    species_fractions: dict[str, float] = {}
    species_rates: dict[str, str] = {}
    for item, fraction in zip(species, fractions):
        name = str(item.get("name", f"species_{len(W_by_species)}"))
        species_rates[name] = str(item.get("rate", ""))
        species_fractions[name] = fraction
        single_ion = _single_species_ionization_config(ion, item)
        single_wfunc = make_Wfunc("production_0d_species", single_ion, omega0, float(beam.n0))
        W_by_species[name] = np.asarray(to_cpu(single_wfunc(I_3d)))[:, 0, 0].astype(np.float64, copy=False)
        rho_j, _ = evolve_rho_time(I_3d, axes.dt, N0_air * fraction, float(single_ion.beta_rec), single_wfunc)
        rho_by_species[name] = np.asarray(to_cpu(rho_j))[:, 0, 0].astype(np.float64, copy=False)

    return IonizationCase(
        config_path=config_path.resolve(),
        tau_fwhm_s=float(beam.tau_fwhm),
        case_label=_case_label(config_path, float(beam.tau_fwhm), float(I_peak_W_m2)),
        t_s=t_s,
        dt_s=float(axes.dt),
        I_peak_W_m2=float(I_peak_W_m2),
        I_t_W_m2=I_t,
        W_by_species_s=W_by_species,
        rho_by_species_m3=rho_by_species,
        rho_total_m3=np.asarray(to_cpu(rho_total))[:, 0, 0].astype(np.float64, copy=False),
        species_fractions=species_fractions,
        species_rates=species_rates,
    )


def case_summary_row(case: IonizationCase) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_label": case.case_label,
        "config_path": str(case.config_path),
        "tau_fwhm_fs": case.tau_fwhm_s * 1e15,
        "I_peak_W_m2": case.I_peak_W_m2,
        "Nt": int(case.t_s.size),
        "Twin_fs": (case.t_s[-1] - case.t_s[0] + case.dt_s) * 1e15,
        "dt_fs": case.dt_s * 1e15,
        "rho_total_peak_m3": float(np.max(case.rho_total_m3)),
        "rho_total_final_m3": float(case.rho_total_m3[-1]),
        "final_ionization_fraction": float(case.rho_total_m3[-1] / N0_air),
    }
    for name, W in case.W_by_species_s.items():
        row[f"W_{name}_max_s-1"] = float(np.max(W))
        row[f"max_W_{name}_dt"] = float(np.max(W) * case.dt_s)
    for name, rho in case.rho_by_species_m3.items():
        row[f"rho_{name}_peak_m3"] = float(np.max(rho))
        row[f"rho_{name}_final_m3"] = float(rho[-1])
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_timeseries(path: Path, cases: Iterable[IonizationCase]) -> None:
    arrays: dict[str, np.ndarray] = {}
    case_ids: list[str] = []
    for case in cases:
        prefix = case.case_label
        case_ids.append(prefix)
        arrays[f"{prefix}__t_s"] = case.t_s
        arrays[f"{prefix}__I_W_m2"] = case.I_t_W_m2
        arrays[f"{prefix}__rho_total_m3"] = case.rho_total_m3
        for name, values in case.W_by_species_s.items():
            arrays[f"{prefix}__W_{name}_s-1"] = values
        for name, values in case.rho_by_species_m3.items():
            arrays[f"{prefix}__rho_{name}_m3"] = values
    arrays["case_ids"] = np.asarray(case_ids, dtype="U96")
    np.savez_compressed(path, **arrays)


def run_0d_ionization_harness(config_paths: Iterable[Path], intensities_W_m2: Iterable[float], out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    cases = [run_production_0d_case(Path(path), float(intensity)) for path in config_paths for intensity in intensities_W_m2]
    rows = [case_summary_row(case) for case in cases]
    _write_csv(out_dir / "ionization_integrator_cases.csv", rows)
    _write_timeseries(out_dir / "ionization_integrator_timeseries.npz", cases)
    metadata = {
        "schema": "khz_filament.ionization_time_harness.v1",
        "code_commit_sha": _git_sha(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rate_evaluator": "production make_Wfunc runtime evaluator (LUT/reference according to each supplied config)",
        "temporal_convention": {
            "tau_fwhm_interpretation": "intensity FWHM",
            "gaussian_pulse_t_output": "field envelope",
            "ionization_input": "I(t) = I_peak * abs(gaussian_pulse_t(t, tau_fwhm))**2 [W/m^2]",
        },
        "cases": [{
            "case_label": case.case_label,
            "config_path": str(case.config_path),
            "config_sha256": _sha256(case.config_path),
            "tau_fwhm_fs": case.tau_fwhm_s * 1e15,
            "I_peak_W_m2": case.I_peak_W_m2,
            "species_fractions": case.species_fractions,
            "species_rates": case.species_rates,
        } for case in cases],
    }
    (out_dir / "ionization_time_harness_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", help="production config; repeat for 40 fs and 120 fs")
    parser.add_argument("--intensity-W-m2", type=float, nargs="+", default=list(DEFAULT_INTENSITIES_W_M2))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    configs = args.config if args.config else list(DEFAULT_CONFIGS)
    metadata = run_0d_ionization_harness(configs, args.intensity_W_m2, args.out_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

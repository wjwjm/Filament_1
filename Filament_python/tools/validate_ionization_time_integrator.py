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
    stability_by_species: dict[str, dict[str, Any]] | None


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


def run_production_0d_case(config_path: Path, I_peak_W_m2: float, *, time_refinement: int = 1,
                           diagnose_integrator_stability: bool = False) -> IonizationCase:
    """Run the production RK4/evaluator on one temporal point (no 3D propagation)."""
    grid, beam, _prop, ion, _heat, _run, _raman = load_all(str(config_path))
    if str(ion.time_mode).lower() != "full":
        raise ValueError(f"0D integrator validation requires ionization.time_mode='full': {config_path}")
    if not ion.species:
        raise ValueError(f"0D integrator validation requires non-empty ionization.species: {config_path}")
    if float(I_peak_W_m2) <= 0.0:
        raise ValueError("I_peak_W_m2 must be positive")

    if int(time_refinement) < 1:
        raise ValueError("time_refinement must be >= 1")
    axes = make_axes(1, 1, int(grid.Nt) * int(time_refinement), 1.0, 1.0, float(grid.Twin))
    t_s = np.asarray(to_cpu(axes.t), dtype=np.float64)
    I_t = build_intensity_envelope(t_s, float(beam.tau_fwhm), float(I_peak_W_m2))
    I_3d = I_t[:, None, None]
    omega0 = 2.0 * math.pi * float(c0) / float(beam.lam0)
    Wfunc = make_Wfunc("production_0d", ion, omega0, float(beam.n0))
    rho_output = evolve_rho_time(
        I_3d,
        axes.dt,
        N0_air,
        float(ion.beta_rec),
        Wfunc,
        diagnose_integrator_stability=diagnose_integrator_stability,
    )
    if diagnose_integrator_stability:
        rho_total, _Wt, stability = rho_output
    else:
        rho_total, _Wt = rho_output
        stability = None

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
        stability_by_species=stability["species"] if stability is not None else None,
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


def cumulative_rate_reference(W_s: np.ndarray, t_s: np.ndarray, N0_species_m3: float) -> np.ndarray:
    """No-recombination reference rho=N0*(1-exp(-integral W dt)) using trapezoids."""
    W_s = np.asarray(W_s, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    cumulative = np.zeros_like(W_s)
    if W_s.size > 1:
        cumulative[1:] = np.cumsum(0.5 * (W_s[1:] + W_s[:-1]) * np.diff(t_s))
    return float(N0_species_m3) * (-np.expm1(-cumulative))


def exponential_average_update(W_s: np.ndarray, t_s: np.ndarray, N0_species_m3: float) -> np.ndarray:
    """Optional candidate update; not used by production propagation."""
    W_s = np.asarray(W_s, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    u = np.zeros_like(W_s)
    for index in range(W_s.size - 1):
        W_mean = 0.5 * (W_s[index] + W_s[index + 1])
        u[index + 1] = 1.0 - (1.0 - u[index]) * math.exp(-W_mean * (t_s[index + 1] - t_s[index]))
    return float(N0_species_m3) * u


def _fixed_threshold_time(t_s: np.ndarray, rho_m3: np.ndarray, threshold_m3: float) -> float:
    hit = np.flatnonzero(np.asarray(rho_m3) >= float(threshold_m3))
    if hit.size == 0:
        return float("nan")
    index = int(hit[0])
    if index == 0:
        return float(t_s[0])
    y0, y1 = float(rho_m3[index - 1]), float(rho_m3[index])
    if y1 <= y0:
        return float(t_s[index])
    fraction = (float(threshold_m3) - y0) / (y1 - y0)
    return float(t_s[index - 1] + fraction * (t_s[index] - t_s[index - 1]))


def _error_metrics(rho_rk4: np.ndarray, rho_ref: np.ndarray, t_s: np.ndarray, *, rho_floor_m3: float,
                   rise_threshold_m3: float) -> dict[str, float]:
    denominator = max(float(np.max(rho_ref)), float(rho_floor_m3))
    t_rk4 = _fixed_threshold_time(t_s, rho_rk4, rise_threshold_m3)
    t_ref = _fixed_threshold_time(t_s, rho_ref, rise_threshold_m3)
    return {
        "rho_final_rel_error": float(abs(rho_rk4[-1] - rho_ref[-1]) / max(abs(rho_ref[-1]), rho_floor_m3)),
        "rho_peak_rel_error": float(abs(np.max(rho_rk4) - np.max(rho_ref)) / denominator),
        "rho_time_max_rel_error": float(np.max(np.abs(rho_rk4 - rho_ref)) / denominator),
        "rise_time_error_fs": float((t_rk4 - t_ref) * 1e15) if np.isfinite(t_rk4) and np.isfinite(t_ref) else float("nan"),
        "rk4_rise_time_fs": float(t_rk4 * 1e15) if np.isfinite(t_rk4) else float("nan"),
        "reference_rise_time_fs": float(t_ref * 1e15) if np.isfinite(t_ref) else float("nan"),
    }


def _interpolate_to(t_target: np.ndarray, t_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(t_target, dtype=np.float64), np.asarray(t_source, dtype=np.float64), np.asarray(values, dtype=np.float64))


def _stability_metrics(case: IonizationCase, species_name: str) -> dict[str, float | int]:
    raw = (case.stability_by_species or {}).get(species_name, {})
    return {
        "preclip_step_min": float(raw.get("preclip_step_min", float("nan"))),
        "preclip_step_max": float(raw.get("preclip_step_max", float("nan"))),
        "preclip_intermediate_min": float(raw.get("preclip_intermediate_min", float("nan"))),
        "preclip_intermediate_max": float(raw.get("preclip_intermediate_max", float("nan"))),
        "step_clip_count": int(raw.get("step_clip_count", 0)),
        "intermediate_violation_count": int(raw.get("intermediate_violation_count", 0)),
    }


def run_integrator_comparison(config_paths: Iterable[Path], intensities_W_m2: Iterable[float], out_dir: Path, *,
                              refinements: tuple[int, ...] = (1, 2, 4, 8), rho_floor_m3: float = 1e10,
                              rise_threshold_m3: float = 1e20) -> dict[str, Any]:
    """Compare production RK4 with recomputed refined grids and a trapezoid reference.

    Each refinement rebuilds the temporal grid, Gaussian intensity, and rate
    histories through production code.  Only the final reference is interpolated
    down to a coarser grid for like-for-like error reporting.
    """
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    if tuple(refinements) != tuple(sorted(set(refinements))) or int(refinements[0]) != 1:
        raise ValueError("refinements must be an increasing unique sequence beginning with 1")
    out_dir.mkdir(parents=True)
    error_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    metadata_cases: list[dict[str, Any]] = []

    for config_path in config_paths:
        for intensity in intensities_W_m2:
            config_grid, _config_beam, _config_prop, config_ion, _config_heat, _config_run, _config_raman = load_all(str(config_path))
            solutions = {
                factor: run_production_0d_case(
                    Path(config_path), float(intensity), time_refinement=int(factor), diagnose_integrator_stability=True
                )
                for factor in refinements
            }
            coarse = solutions[1]
            finest = solutions[refinements[-1]]
            if not math.isclose(float(config_ion.beta_rec), 0.0, abs_tol=0.0):
                raise ValueError("cumulative-rate reference is only valid when ionization.beta_rec=0")

            case_rows.append(case_summary_row(coarse))
            case_key = coarse.case_label
            arrays[f"{case_key}__t_s"] = coarse.t_s
            arrays[f"{case_key}__I_W_m2"] = coarse.I_t_W_m2
            arrays[f"{case_key}__rho_total_rk4_f1_m3"] = coarse.rho_total_m3
            arrays[f"{case_key}__rho_total_reference_f{refinements[-1]}_m3"] = np.zeros_like(coarse.rho_total_m3)

            species_names = tuple(coarse.rho_by_species_m3)

            def _total_stability(solution: IonizationCase) -> dict[str, float | int]:
                values = [_stability_metrics(solution, name) for name in species_names]
                return {
                    "preclip_step_min": min(item["preclip_step_min"] for item in values),
                    "preclip_step_max": max(item["preclip_step_max"] for item in values),
                    "preclip_intermediate_min": min(item["preclip_intermediate_min"] for item in values),
                    "preclip_intermediate_max": max(item["preclip_intermediate_max"] for item in values),
                    "step_clip_count": sum(item["step_clip_count"] for item in values),
                    "intermediate_violation_count": sum(item["intermediate_violation_count"] for item in values),
                }

            for name in species_names:
                N0_species = N0_air * coarse.species_fractions[name]
                fine_ref = cumulative_rate_reference(finest.W_by_species_s[name], finest.t_s, N0_species)
                coarse_ref = _interpolate_to(coarse.t_s, finest.t_s, fine_ref)
                arrays[f"{case_key}__W_{name}_f1_s-1"] = coarse.W_by_species_s[name]
                arrays[f"{case_key}__rho_{name}_rk4_f1_m3"] = coarse.rho_by_species_m3[name]
                arrays[f"{case_key}__rho_{name}_reference_f{refinements[-1]}_m3"] = coarse_ref
                arrays[f"{case_key}__rho_{name}_exponential_f1_m3"] = exponential_average_update(
                    coarse.W_by_species_s[name], coarse.t_s, N0_species
                )
                row = {
                    "case_label": case_key,
                    "config_path": str(coarse.config_path),
                    "tau_fwhm_fs": coarse.tau_fwhm_s * 1e15,
                    "I_peak_W_m2": coarse.I_peak_W_m2,
                    "species": name,
                    "refinement_factor": 1,
                    "dt_fs": coarse.dt_s * 1e15,
                    "max_W_dt": float(np.max(coarse.W_by_species_s[name]) * coarse.dt_s),
                    **_error_metrics(
                        coarse.rho_by_species_m3[name], coarse_ref, coarse.t_s,
                        rho_floor_m3=rho_floor_m3, rise_threshold_m3=rise_threshold_m3,
                    ),
                    **_stability_metrics(coarse, name),
                }
                error_rows.append(row)

            fine_total_ref = sum(
                cumulative_rate_reference(finest.W_by_species_s[name], finest.t_s, N0_air * finest.species_fractions[name])
                for name in species_names
            )
            coarse_total_ref = _interpolate_to(coarse.t_s, finest.t_s, fine_total_ref)
            arrays[f"{case_key}__rho_total_reference_f{refinements[-1]}_m3"] = coarse_total_ref
            total_stability = _total_stability(coarse)
            error_rows.append({
                "case_label": case_key,
                "config_path": str(coarse.config_path),
                "tau_fwhm_fs": coarse.tau_fwhm_s * 1e15,
                "I_peak_W_m2": coarse.I_peak_W_m2,
                "species": "total",
                "refinement_factor": 1,
                "dt_fs": coarse.dt_s * 1e15,
                "max_W_dt": max(float(np.max(coarse.W_by_species_s[name]) * coarse.dt_s) for name in species_names),
                **_error_metrics(coarse.rho_total_m3, coarse_total_ref, coarse.t_s, rho_floor_m3=rho_floor_m3, rise_threshold_m3=rise_threshold_m3),
                **total_stability,
            })
            for factor in refinements[1:]:
                refined = solutions[factor]
                arrays[f"{case_key}__t_f{factor}_s"] = refined.t_s
                arrays[f"{case_key}__I_f{factor}_W_m2"] = refined.I_t_W_m2
                for name in species_names:
                    N0_species = N0_air * refined.species_fractions[name]
                    fine_ref = cumulative_rate_reference(finest.W_by_species_s[name], finest.t_s, N0_species)
                    refined_ref = _interpolate_to(refined.t_s, finest.t_s, fine_ref)
                    arrays[f"{case_key}__rho_{name}_rk4_f{factor}_m3"] = refined.rho_by_species_m3[name]
                    arrays[f"{case_key}__rho_{name}_reference_f{refinements[-1]}_on_f{factor}_m3"] = refined_ref
                    error_rows.append({
                        "case_label": case_key,
                        "config_path": str(refined.config_path),
                        "tau_fwhm_fs": refined.tau_fwhm_s * 1e15,
                        "I_peak_W_m2": refined.I_peak_W_m2,
                        "species": name,
                        "refinement_factor": factor,
                        "dt_fs": refined.dt_s * 1e15,
                        "max_W_dt": float(np.max(refined.W_by_species_s[name]) * refined.dt_s),
                        **_error_metrics(refined.rho_by_species_m3[name], refined_ref, refined.t_s, rho_floor_m3=rho_floor_m3, rise_threshold_m3=rise_threshold_m3),
                        **_stability_metrics(refined, name),
                    })
                refined_total_ref = _interpolate_to(refined.t_s, finest.t_s, fine_total_ref)
                arrays[f"{case_key}__rho_total_rk4_f{factor}_m3"] = refined.rho_total_m3
                arrays[f"{case_key}__rho_total_reference_f{refinements[-1]}_on_f{factor}_m3"] = refined_total_ref
                error_rows.append({
                    "case_label": case_key,
                    "config_path": str(refined.config_path),
                    "tau_fwhm_fs": refined.tau_fwhm_s * 1e15,
                    "I_peak_W_m2": refined.I_peak_W_m2,
                    "species": "total",
                    "refinement_factor": factor,
                    "dt_fs": refined.dt_s * 1e15,
                    "max_W_dt": max(float(np.max(refined.W_by_species_s[name]) * refined.dt_s) for name in species_names),
                    **_error_metrics(refined.rho_total_m3, refined_total_ref, refined.t_s, rho_floor_m3=rho_floor_m3, rise_threshold_m3=rise_threshold_m3),
                    **_total_stability(refined),
                })
            metadata_cases.append({
                "case_label": case_key,
                "config_path": str(coarse.config_path),
                "tau_fwhm_fs": coarse.tau_fwhm_s * 1e15,
                "I_peak_W_m2": coarse.I_peak_W_m2,
                "refinements": list(refinements),
                "species_rates": coarse.species_rates,
                "species_fractions": coarse.species_fractions,
                "production_grid": {
                    "Nt": int(config_grid.Nt),
                    "Twin_fs": float(config_grid.Twin) * 1e15,
                    "dt_fs": coarse.dt_s * 1e15,
                },
                "species_parameters": [
                    {
                        key: item[key]
                        for key in ("name", "rate", "reference_model", "Ip_eV", "Ip_eV_eff", "Z", "Zeff", "l", "m", "fraction")
                        if key in item
                    }
                    for item in (config_ion.species or [])
                ],
            })

    _write_csv(out_dir / "ionization_integrator_cases.csv", case_rows)
    _write_csv(out_dir / "ionization_integrator_error_summary.csv", error_rows)
    np.savez_compressed(out_dir / "ionization_integrator_timeseries.npz", **arrays)
    metadata = {
        "schema": "khz_filament.ionization_integrator_comparison.v1",
        "code_commit_sha": _git_sha(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "refinements": list(refinements),
        "reference": "per-species no-recombination cumulative trapezoid integral of production W(t)",
        "candidate_exponential_update": "reported only; production RK4 is not replaced",
        "rho_floor_m3": rho_floor_m3,
        "rise_threshold_m3": rise_threshold_m3,
        "cases": metadata_cases,
    }
    (out_dir / "ionization_integrator_comparison_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def classify_integrator_evidence(case_rows: list[dict[str, Any]], error_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the production-dt evidence using the Phase-3 quantitative gates."""
    final_fraction = {row["case_label"]: _as_float(row, "final_ionization_fraction") for row in case_rows}
    production_rows = [row for row in error_rows if int(row.get("refinement_factor", 0)) == 1]
    total_rows = [row for row in production_rows if row.get("species") == "total"]
    relevant = [row for row in total_rows if final_fraction.get(row["case_label"], 0.0) >= 1e-6]
    nonsaturated = [row for row in relevant if final_fraction.get(row["case_label"], 0.0) < 0.95]
    clips = [row for row in relevant if int(float(row.get("step_clip_count", 0))) > 0 or int(float(row.get("intermediate_violation_count", 0))) > 0]
    severe = [
        row for row in nonsaturated
        if _as_float(row, "rho_final_rel_error") > 0.05
        or _as_float(row, "rho_time_max_rel_error") > 0.05
        or (np.isfinite(_as_float(row, "rise_time_error_fs")) and abs(_as_float(row, "rise_time_error_fs")) > 0.5)
    ]
    passed = [
        row for row in nonsaturated
        if _as_float(row, "rho_final_rel_error") < 0.01
        and _as_float(row, "rho_time_max_rel_error") < 0.01
        and (not np.isfinite(_as_float(row, "rise_time_error_fs")) or abs(_as_float(row, "rise_time_error_fs")) < 0.5)
    ]
    if not nonsaturated:
        classification = "inconclusive"
        rationale = "No non-saturated case reached the automatic meaningful-ionization threshold."
    elif clips or severe:
        classification = "supported"
        rationale = "At least one meaningful non-saturated case exceeded the 5%/0.5 fs gate or required clipping."
    elif len(passed) == len(nonsaturated):
        classification = "not_supported"
        rationale = "All meaningful non-saturated production-dt cases satisfy the 1%/0.5 fs gates without clipping."
    else:
        classification = "inconclusive"
        rationale = "No severe failure occurred, but one or more meaningful cases fall in the 1%–5% warning band."
    return {
        "classification": classification,
        "rationale": rationale,
        "meaningful_ionization_fraction_floor": 1e-6,
        "saturation_fraction_threshold": 0.95,
        "relevant_case_count": len(relevant),
        "nonsaturated_case_count": len(nonsaturated),
        "clip_case_count": len(clips),
        "severe_case_count": len(severe),
        "passed_case_count": len(passed),
        "relevant_intensity_range_W_m2": [
            min((_as_float(row, "I_peak_W_m2") for row in relevant), default=float("nan")),
            max((_as_float(row, "I_peak_W_m2") for row in relevant), default=float("nan")),
        ],
    }


def write_integrator_validation_report(out_dir: Path) -> dict[str, Any]:
    """Build the Task-3 Markdown conclusion from already-auditable CSV outputs."""
    out_dir = Path(out_dir)
    with (out_dir / "ionization_integrator_cases.csv").open(encoding="utf-8", newline="") as handle:
        case_rows = list(csv.DictReader(handle))
    with (out_dir / "ionization_integrator_error_summary.csv").open(encoding="utf-8", newline="") as handle:
        error_rows = list(csv.DictReader(handle))
    metadata = json.loads((out_dir / "ionization_integrator_comparison_metadata.json").read_text(encoding="utf-8"))
    classification = classify_integrator_evidence(case_rows, error_rows)
    production_rows = [row for row in error_rows if int(row.get("refinement_factor", 0)) == 1]
    total_rows = [row for row in error_rows if row.get("species") == "total" and int(row.get("refinement_factor", 0)) == 1]
    lines = [
        "# Ionization time-integrator validation report",
        "",
        f"- Code SHA: `{metadata['code_commit_sha']}`",
        f"- Generated (UTC): {metadata['generated_at_utc']}",
        "- Production path: `make_Wfunc` → `evolve_rho_time` with the supplied configuration evaluator (LUT/reference as configured).",
        f"- Temporal refinements: {', '.join(str(value) for value in metadata['refinements'])}",
        f"- Fixed rise threshold: {metadata['rise_threshold_m3']:.3e} m^-3",
        "- `tau_fwhm` is the intensity FWHM; `gaussian_pulse_t` produces the field envelope and the rate receives `I(t)` in W/m².",
        "",
        "## Test configuration",
        "",
    ]
    config_summaries: dict[str, dict[str, Any]] = {}
    for item in metadata["cases"]:
        config_summaries.setdefault(item["config_path"], item)
    for item in config_summaries.values():
        grid = item["production_grid"]
        lines += [
            f"- {item['tau_fwhm_fs']:.0f} fs: `{item['config_path']}`; Nt={grid['Nt']}, Twin={grid['Twin_fs']:.1f} fs, dt={grid['dt_fs']:.3f} fs; species={json.dumps(item['species_parameters'], ensure_ascii=False)}",
        ]
    lines += [
        "",
        "## Production-dt density errors",
        "",
        "| tau (fs) | Ipeak (W/m²) | species | final rho error | time-max rho error | rise error (fs) | max(Wdt) | step clips | intermediate violations |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    species_order = {"N2": 0, "O2": 1, "total": 2}
    for row in sorted(production_rows, key=lambda item: (_as_float(item, "tau_fwhm_fs"), _as_float(item, "I_peak_W_m2"), species_order.get(item["species"], 99))):
        lines.append(
            f"| {_as_float(row, 'tau_fwhm_fs'):.0f} | {_as_float(row, 'I_peak_W_m2'):.3e} | "
            f"{row['species']} | {_as_float(row, 'rho_final_rel_error'):.3e} | {_as_float(row, 'rho_time_max_rel_error'):.3e} | "
            f"{_as_float(row, 'rise_time_error_fs'):.3e} | {_as_float(row, 'max_W_dt'):.3e} | "
            f"{int(float(row['step_clip_count']))} | {int(float(row['intermediate_violation_count']))} |"
        )
    lines += [
        "",
        "## Decision gates",
        "",
        "- `not_supported`: every meaningful non-saturated production-dt case has final and time-history errors below 1%, fixed-threshold rise-time error below 0.5 fs (when crossed), and no pre-clip violation.",
        "- `inconclusive`: no severe failure, but a meaningful case is in the 1%–5% warning band or lacks a decisive threshold crossing.",
        "- `supported`: a meaningful non-saturated case exceeds 5%, exceeds 0.5 fs in rise time, or shows a pre-clip violation.",
        "",
        "## Causal conclusion",
        "",
        f"**{classification['classification']}** — {classification['rationale']}",
        "",
        f"Automatic meaningful intensity interval: {classification['relevant_intensity_range_W_m2'][0]:.3e} to {classification['relevant_intensity_range_W_m2'][1]:.3e} W/m²; "
        f"non-saturated cases: {classification['nonsaturated_case_count']}; clip cases: {classification['clip_case_count']}.",
        "",
        "This conclusion concerns whether the current fixed-step RK4 can be the principal cause of the observed filament electron-density onset/peak/tail discrepancy. It does not change the production integrator.",
    ]
    report_path = out_dir / "ionization_integrator_validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"report_path": str(report_path), **classification}
    (out_dir / "ionization_integrator_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", help="production config; repeat for 40 fs and 120 fs")
    parser.add_argument("--intensity-W-m2", type=float, nargs="+", default=list(DEFAULT_INTENSITIES_W_M2))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--compare-refinements", action="store_true", help="run RK4/refinement/reference comparison instead of Task-1 harness-only output")
    parser.add_argument("--write-final-report", action="store_true", help="write the Task-3 Markdown classification after comparison")
    parser.add_argument("--rise-threshold-m3", type=float, default=1e20)
    args = parser.parse_args()
    configs = args.config if args.config else list(DEFAULT_CONFIGS)
    metadata = (
        run_integrator_comparison(configs, args.intensity_W_m2, args.out_dir, rise_threshold_m3=args.rise_threshold_m3)
        if args.compare_refinements
        else run_0d_ionization_harness(configs, args.intensity_W_m2, args.out_dir)
    )
    if args.write_final_report:
        if not args.compare_refinements:
            raise ValueError("--write-final-report requires --compare-refinements")
        metadata["validation_report"] = write_integrator_validation_report(args.out_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

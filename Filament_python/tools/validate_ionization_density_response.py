#!/usr/bin/env python3
"""Map Popruzhenko/Talebpour local 0D ionization responses to density thresholds.

The physical-model comparison uses the production full-reference evaluators.
The no-recombination cumulative solution is primary; the existing production
RK4 path is calculated as a consistency check and is not modified here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
if str(FILAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(FILAMENT_ROOT))

from KHz_filament.confio import load_all  # noqa: E402
from KHz_filament.constants import N0_air  # noqa: E402
from KHz_filament.device import to_cpu  # noqa: E402
from KHz_filament.grids import make_axes  # noqa: E402
from KHz_filament.ionization import evolve_rho_time  # noqa: E402
from validate_ionization_rate_models import (  # noqa: E402
    DEFAULT_CONFIG,
    FILAMENT_ROOT as RATE_FILAMENT_ROOT,
    RATE_MODELS,
    _production_species,
    _timestamped_output_dir,
    make_rate_evaluator,
    repo_relative,
)
from validate_ionization_time_integrator import build_intensity_envelope  # noqa: E402


DEFAULT_CONFIGS = (
    FILAMENT_ROOT / "configs" / "profile_validation" / "flat_top_90_40fs.json",
    FILAMENT_ROOT / "configs" / "profile_validation" / "flat_top_90_120fs.json",
)
DEFAULT_THRESHOLDS_M3 = (1e19, 1e20, 1e21, 1e22)


def _to_numpy(values: Any) -> np.ndarray:
    return np.asarray(to_cpu(values), dtype=np.float64)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(value, float) and not math.isfinite(value) else value for key, value in row.items()})


def _fractions(species: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = {str(item["name"]): max(0.0, float(item.get("fraction", 1.0))) for item in species}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("species fractions must have a positive sum")
    return {name: value / total for name, value in values.items()}


def cumulative_reference_batch(W_s: np.ndarray, t_s: np.ndarray, N0_species_m3: float) -> np.ndarray:
    """Vectorized Phase-3 trapezoid cumulative reference for [Ipeak, time]."""
    W_s = np.asarray(W_s, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    if W_s.ndim != 2 or W_s.shape[1] != t_s.size:
        raise ValueError("W_s must be [n_intensity, n_time]")
    dose = np.zeros_like(W_s)
    if t_s.size > 1:
        dose[:, 1:] = np.cumsum(0.5 * (W_s[:, 1:] + W_s[:, :-1]) * np.diff(t_s)[None, :], axis=1)
    return float(N0_species_m3) * (-np.expm1(-dose))


def threshold_intensity_log_interpolation(I_W_m2: np.ndarray, rho_m3: np.ndarray, threshold_m3: float) -> dict[str, Any]:
    """Return an in-range threshold crossing using log-density/log-intensity interpolation."""
    I = np.asarray(I_W_m2, dtype=np.float64)
    rho = np.asarray(rho_m3, dtype=np.float64)
    if I.ndim != 1 or rho.shape != I.shape or not np.all(np.diff(I) > 0.0):
        raise ValueError("I and rho must be monotonic one-dimensional arrays")
    hit = np.flatnonzero(rho >= float(threshold_m3))
    if hit.size == 0:
        return {"status": "not_crossed", "I_threshold_W_m2": None}
    index = int(hit[0])
    if index == 0:
        return {"status": "crossed_at_scan_min", "I_threshold_W_m2": float(I[0])}
    lo, hi = index - 1, index
    if rho[lo] <= 0.0 or rho[hi] <= rho[lo]:
        return {"status": "crossed_grid_point", "I_threshold_W_m2": float(I[hi])}
    fraction = (math.log10(float(threshold_m3)) - math.log10(float(rho[lo]))) / (math.log10(float(rho[hi])) - math.log10(float(rho[lo])))
    logI = math.log10(float(I[lo])) + min(1.0, max(0.0, fraction)) * (math.log10(float(I[hi])) - math.log10(float(I[lo])))
    return {"status": "crossed_interpolated", "I_threshold_W_m2": float(10.0 ** logI)}


def _interpolate_logI(I_W_m2: np.ndarray, values: np.ndarray, I_target_W_m2: float) -> float:
    return float(np.interp(math.log10(float(I_target_W_m2)), np.log10(np.asarray(I_W_m2, dtype=np.float64)), np.asarray(values, dtype=np.float64)))


def load_rate_interpolators(rate_data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load Task-1 production-reference curves as log-log rate interpolators."""
    result: dict[str, dict[str, Any]] = {}
    source: dict[str, Any] = {"rate_data_dir": repo_relative(rate_data_dir)}
    for name in ("N2", "O2"):
        path = rate_data_dir / f"ionization_rate_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Task 2 requires Task 1 output: {repo_relative(path)}")
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        I = np.asarray([float(row["I_W_m2"]) for row in rows], dtype=np.float64)
        if I.size < 500 or not np.all(np.diff(I) > 0.0):
            raise ValueError(f"Task 1 rate grid must contain >=500 increasing points: {repo_relative(path)}")
        result[name] = {"I_W_m2": I}
        for family in ("popruzhenko", "talebpour"):
            W = np.asarray([float(row[f"W_{family}_reference_s-1"]) for row in rows], dtype=np.float64)
            logI = np.log10(I)
            logW = np.log10(np.maximum(W, np.finfo(np.float64).tiny))

            def evaluate(values: np.ndarray, *, _I=I, _logI=logI, _logW=logW, _W=W) -> np.ndarray:
                array = np.asarray(values, dtype=np.float64)
                log_values = np.log10(np.maximum(array, np.finfo(np.float64).tiny))
                interpolated = np.power(10.0, np.interp(log_values, _logI, _logW))
                return np.where(array < _I[0], 0.0, np.where(array > _I[-1], _W[-1], interpolated))

            result[name][family] = evaluate
        source[name] = {"path": repo_relative(path), "I_min_W_m2": float(I[0]), "I_max_W_m2": float(I[-1]), "n_points": int(I.size)}
    return result, source


def _run_model_response(config_path: Path, I_peak_W_m2: np.ndarray, family: str, rate_interpolators: dict[str, dict[str, Any]], *,
                        time_refinement: int = 1) -> dict[str, Any]:
    grid, beam, _prop, _production_ion, _heat, _run, _raman = load_all(str(config_path))
    _beam_check, _ion_check, production_species_list = _production_species(config_path)
    species_by_name = {str(item["name"]): item for item in production_species_list}
    fractions = _fractions(production_species_list)
    if int(time_refinement) < 1:
        raise ValueError("time_refinement must be >= 1")
    axes = make_axes(1, 1, int(grid.Nt) * int(time_refinement), 1.0, 1.0, float(grid.Twin))
    t_s = _to_numpy(axes.t)
    envelope = build_intensity_envelope(t_s, float(beam.tau_fwhm), 1.0)
    intensity_history = np.asarray(I_peak_W_m2, dtype=np.float64)[:, None] * envelope[None, :]
    rho_reference: dict[str, np.ndarray] = {}
    W_by_species: dict[str, np.ndarray] = {}
    for name in ("N2", "O2"):
        W = rate_interpolators[name][family](intensity_history)
        if not np.all(np.isfinite(W)):
            raise ValueError(f"non-finite {family} {name} rates")
        N0_species = N0_air * fractions[name]
        rho_reference[name] = cumulative_reference_batch(W, t_s, N0_species)
        W_by_species[name] = W
    rho_reference_air = rho_reference["N2"] + rho_reference["O2"]
    return {
        "tau_fwhm_fs": float(beam.tau_fwhm) * 1e15,
        "config_path": Path(config_path).resolve(),
        "t_s": t_s,
        "dt_s": float(axes.dt),
        "I_peak_W_m2": np.asarray(I_peak_W_m2, dtype=np.float64),
        "I_t_W_m2": intensity_history,
        "W_by_species_s-1": W_by_species,
        "rho_reference_by_species_m3": rho_reference,
        "rho_reference_air_m3": rho_reference_air,
        "fractions": fractions,
        "time_refinement": int(time_refinement),
    }


def reference_time_convergence(config_path: Path, family: str, rate_interpolators: dict[str, dict[str, Any]], *,
                               probe_I_W_m2: Iterable[float] = (1e17, 3e17, 1e18), refinement: int = 8) -> list[dict[str, Any]]:
    """Check the cumulative reference on production and refined temporal grids."""
    probe = np.asarray(tuple(float(item) for item in probe_I_W_m2), dtype=np.float64)
    coarse = _run_model_response(config_path, probe, family, rate_interpolators, time_refinement=1)
    fine = _run_model_response(config_path, probe, family, rate_interpolators, time_refinement=refinement)
    rows: list[dict[str, Any]] = []
    for name, coarse_values, fine_values in (
        ("N2", coarse["rho_reference_by_species_m3"]["N2"], fine["rho_reference_by_species_m3"]["N2"]),
        ("O2", coarse["rho_reference_by_species_m3"]["O2"], fine["rho_reference_by_species_m3"]["O2"]),
        ("air", coarse["rho_reference_air_m3"], fine["rho_reference_air_m3"]),
    ):
        for index, peak in enumerate(probe):
            value_coarse = float(coarse_values[index, -1])
            value_fine = float(fine_values[index, -1])
            rows.append({
                "tau_fwhm_fs": coarse["tau_fwhm_fs"], "model": family, "species": name,
                "I_peak_W_m2": float(peak), "coarse_dt_fs": coarse["dt_s"] * 1e15,
                "reference_refinement": int(refinement), "refined_dt_fs": fine["dt_s"] * 1e15,
                "coarse_final_m3": value_coarse, "refined_final_m3": value_fine,
                "final_relative_difference": abs(value_coarse - value_fine) / max(abs(value_fine), 1e10),
            })
    return rows


def rk4_consistency_probes(config_path: Path, family: str, cache_dir: Path, *,
                           probe_I_W_m2: Iterable[float] = (1e17, 3e17, 1e18)) -> list[dict[str, Any]]:
    """Run the unmodified production RK4/evaluator on a small relevant set."""
    grid, beam, _prop, production_ion, _heat, _run, _raman = load_all(str(config_path))
    _beam_check, _ion_check, production_species_list = _production_species(config_path)
    species_by_name = {str(item["name"]): item for item in production_species_list}
    fractions = _fractions(production_species_list)
    axes = make_axes(1, 1, int(grid.Nt), 1.0, 1.0, float(grid.Twin))
    t_s = _to_numpy(axes.t)
    probe = np.asarray(tuple(float(item) for item in probe_I_W_m2), dtype=np.float64)
    intensity_history = probe[:, None] * build_intensity_envelope(t_s, float(beam.tau_fwhm), 1.0)[None, :]
    rows: list[dict[str, Any]] = []
    for name in ("N2", "O2"):
        evaluator, _metadata = make_rate_evaluator(
            production_ion, beam, species_by_name[name], RATE_MODELS[family][0], cache_dir=cache_dir
        )
        W = evaluator(intensity_history)
        reference = cumulative_reference_batch(W, t_s, N0_air * fractions[name])
        Wfunc = evaluator.production_Wfunc  # type: ignore[attr-defined]
        rho_rk4, _ = evolve_rho_time(intensity_history.T[:, None, :], float(axes.dt), N0_air * fractions[name], float(production_ion.beta_rec), Wfunc)
        rk4 = _to_numpy(rho_rk4)[:, 0, :].T
        for index, peak in enumerate(probe):
            rows.append({
                "tau_fwhm_fs": float(beam.tau_fwhm) * 1e15, "model": family, "species": name,
                "I_peak_W_m2": float(peak), "dt_fs": float(axes.dt) * 1e15,
                "reference_final_m3": float(reference[index, -1]), "rk4_final_m3": float(rk4[index, -1]),
                "final_relative_error": abs(float(rk4[index, -1]) - float(reference[index, -1])) / max(abs(float(reference[index, -1])), 1e10),
            })
    return rows


def _propagation_intensity_range(npz_path: Path | None) -> dict[str, Any] | None:
    if npz_path is None:
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        key = next((candidate for candidate in ("I_onaxis_max_z", "I_max_z") if candidate in data.files), None)
        if key is None:
            return {"path": repo_relative(npz_path), "status": "missing_intensity_key", "available_keys": sorted(data.files)}
        values = np.asarray(data[key], dtype=np.float64)
        return {"path": repo_relative(npz_path), "status": "ok", "source_key": key, "I_min_W_m2": float(np.min(values)), "I_max_W_m2": float(np.max(values))}


def run_density_response(config_paths: Iterable[Path] = DEFAULT_CONFIGS, *, I_min_W_m2: float = 1e14, I_max_W_m2: float = 1e19,
                         n_points: int = 501, thresholds_m3: Iterable[float] = DEFAULT_THRESHOLDS_M3,
                         out_dir: Path | None = None, propagation_npz: Path | None = None,
                         rate_data_dir: Path | None = None) -> dict[str, Any]:
    """Create 40 fs/120 fs local density response and fixed-threshold maps."""
    if I_min_W_m2 <= 0.0 or I_max_W_m2 <= I_min_W_m2 or int(n_points) < 2:
        raise ValueError("require 0 < I_min_W_m2 < I_max_W_m2 and n_points >= 2")
    I = np.logspace(math.log10(float(I_min_W_m2)), math.log10(float(I_max_W_m2)), int(n_points))
    if out_dir is None:
        out_dir = _timestamped_output_dir(RATE_FILAMENT_ROOT / "results" / "ionization_rate_model_validation")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rate_data_dir = (out_dir if rate_data_dir is None else Path(rate_data_dir).resolve())
    rate_interpolators, rate_curve_source = load_rate_interpolators(rate_data_dir)
    cache_dir = out_dir / "lut_cache"
    responses: dict[str, dict[str, Any]] = {}
    response_rows_by_tau: dict[str, list[dict[str, Any]]] = {}
    threshold_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"I_peak_W_m2": I}
    rk4_summary: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []

    for config_path in config_paths:
        config_path = Path(config_path).resolve()
        grid, beam, *_ = load_all(str(config_path))
        tau_key = f"{round(float(beam.tau_fwhm) * 1e15):d}fs"
        pop = _run_model_response(config_path, I, "popruzhenko", rate_interpolators)
        tal = _run_model_response(config_path, I, "talebpour", rate_interpolators)
        convergence_rows.extend(reference_time_convergence(config_path, "popruzhenko", rate_interpolators))
        convergence_rows.extend(reference_time_convergence(config_path, "talebpour", rate_interpolators))
        rk4_summary.extend(rk4_consistency_probes(config_path, "popruzhenko", cache_dir))
        rk4_summary.extend(rk4_consistency_probes(config_path, "talebpour", cache_dir))
        responses[tau_key] = {"popruzhenko": pop, "talebpour": tal}
        rows: list[dict[str, Any]] = []
        for index, peak in enumerate(I):
            pop_air = float(pop["rho_reference_air_m3"][index, -1])
            tal_air = float(tal["rho_reference_air_m3"][index, -1])
            rows.append({
                "I_peak_W_m2": float(peak),
                "rho_N2_final_popruzhenko_m3": float(pop["rho_reference_by_species_m3"]["N2"][index, -1]),
                "rho_O2_final_popruzhenko_m3": float(pop["rho_reference_by_species_m3"]["O2"][index, -1]),
                "rho_air_final_popruzhenko_m3": pop_air,
                "rho_N2_final_talebpour_m3": float(tal["rho_reference_by_species_m3"]["N2"][index, -1]),
                "rho_O2_final_talebpour_m3": float(tal["rho_reference_by_species_m3"]["O2"][index, -1]),
                "rho_air_final_talebpour_m3": tal_air,
                "rho_pop_over_talebpour": pop_air / max(tal_air, np.finfo(np.float64).tiny),
            })
        response_rows_by_tau[tau_key] = rows
        _write_csv(out_dir / f"ionization_density_response_{tau_key}.csv", rows)
        for family, response in (("popruzhenko", pop), ("talebpour", tal)):
            arrays[f"{tau_key}__{family}__t_s"] = response["t_s"]
            arrays[f"{tau_key}__{family}__I_t_W_m2"] = response["I_t_W_m2"]
            arrays[f"{tau_key}__{family}__rho_N2_reference_m3"] = response["rho_reference_by_species_m3"]["N2"]
            arrays[f"{tau_key}__{family}__rho_O2_reference_m3"] = response["rho_reference_by_species_m3"]["O2"]
            arrays[f"{tau_key}__{family}__rho_air_reference_m3"] = response["rho_reference_air_m3"]

        for threshold in thresholds_m3:
            pop_threshold = threshold_intensity_log_interpolation(I, pop["rho_reference_air_m3"][:, -1], float(threshold))
            tal_threshold = threshold_intensity_log_interpolation(I, tal["rho_reference_air_m3"][:, -1], float(threshold))
            can_compare = pop_threshold["I_threshold_W_m2"] is not None and tal_threshold["I_threshold_W_m2"] is not None
            row = {
                "tau_fwhm_fs": pop["tau_fwhm_fs"], "density_threshold_m3": float(threshold),
                "popruzhenko_status": pop_threshold["status"], "talebpour_status": tal_threshold["status"],
                "I_threshold_popruzhenko_W_m2": pop_threshold["I_threshold_W_m2"],
                "I_threshold_talebpour_W_m2": tal_threshold["I_threshold_W_m2"],
                "I_threshold_ratio_pop_over_tal": None, "delta_log10_I_pop_minus_tal": None,
            }
            if can_compare:
                I_pop = float(pop_threshold["I_threshold_W_m2"])
                I_tal = float(tal_threshold["I_threshold_W_m2"])
                row["I_threshold_ratio_pop_over_tal"] = I_pop / I_tal
                row["delta_log10_I_pop_minus_tal"] = math.log10(I_pop) - math.log10(I_tal)
                for family, response, crossing in (("popruzhenko", pop, I_pop), ("talebpour", tal, I_tal)):
                    n2 = _interpolate_logI(I, response["rho_reference_by_species_m3"]["N2"][:, -1], crossing)
                    o2 = _interpolate_logI(I, response["rho_reference_by_species_m3"]["O2"][:, -1], crossing)
                    total = max(n2 + o2, np.finfo(np.float64).tiny)
                    contribution_rows.append({
                        "tau_fwhm_fs": response["tau_fwhm_fs"], "model": family, "density_threshold_m3": float(threshold),
                        "I_threshold_W_m2": crossing, "rho_N2_fraction": n2 / total, "rho_O2_fraction": o2 / total,
                    })
            threshold_rows.append(row)
    _write_csv(out_dir / "ionization_density_thresholds.csv", threshold_rows)
    _write_csv(out_dir / "ionization_species_contribution.csv", contribution_rows)
    _write_csv(out_dir / "ionization_density_reference_convergence.csv", convergence_rows)
    np.savez_compressed(out_dir / "ionization_density_response_timeseries.npz", **arrays)
    metadata = {
        "schema": "khz_filament.ionization_density_response.v1",
        "configs": [repo_relative(Path(path)) for path in config_paths],
        "intensity_unit": "W/m^2",
        "I_min_W_m2": float(I_min_W_m2), "I_max_W_m2": float(I_max_W_m2), "n_points": int(n_points),
        "thresholds_m3": [float(item) for item in thresholds_m3],
        "primary_integrator": "cumulative_rate_reference_trapezoid_no_recombination",
        "rate_evaluation": "Task-1 production full-reference curves, log-log interpolated over the 0D temporal envelope",
        "rate_curve_source": rate_curve_source,
        "production_consistency_check": "production evolve_rho_time RK4",
        "rk4_consistency": rk4_summary,
        "reference_time_convergence": convergence_rows,
        "propagation_intensity_marker": _propagation_intensity_range(Path(propagation_npz).resolve()) if propagation_npz else None,
        "outputs": ["ionization_density_response_40fs.csv", "ionization_density_response_120fs.csv", "ionization_density_thresholds.csv", "ionization_species_contribution.csv", "ionization_density_reference_convergence.csv", "ionization_density_response_timeseries.npz"],
    }
    (out_dir / "ionization_density_response_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"out_dir": out_dir, "responses": responses, "threshold_rows": threshold_rows, "contribution_rows": contribution_rows, "metadata": metadata}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--I-min", dest="I_min_W_m2", type=float, default=1e14)
    parser.add_argument("--I-max", dest="I_max_W_m2", type=float, default=1e19)
    parser.add_argument("--n-points", type=int, default=501)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--propagation-npz", type=Path, default=None)
    parser.add_argument("--rate-data-dir", type=Path, default=None, help="Task-1 result directory; defaults to --out-dir")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_density_response(I_min_W_m2=args.I_min_W_m2, I_max_W_m2=args.I_max_W_m2, n_points=args.n_points,
                                  out_dir=args.out_dir, propagation_npz=args.propagation_npz, rate_data_dir=args.rate_data_dir)
    print(f"ionization density-response validation written to {result['out_dir']}")


if __name__ == "__main__":
    main()

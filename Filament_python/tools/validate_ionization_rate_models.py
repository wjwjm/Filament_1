#!/usr/bin/env python3
"""Compare the production N2/O2 ionization-rate evaluators on a CPU grid.

This is a validation harness, not a replacement ionization implementation.  It
constructs all four evaluators through :func:`KHz_filament.ionization.make_Wfunc`
and therefore follows the production runtime's parameter defaults and LUT path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = FILAMENT_ROOT.parent
if str(FILAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(FILAMENT_ROOT))

from KHz_filament.config import IonizationConfig  # noqa: E402
from KHz_filament.confio import load_all  # noqa: E402
from KHz_filament.constants import c0  # noqa: E402
from KHz_filament.device import to_cpu  # noqa: E402
from KHz_filament.ionization import make_Wfunc  # noqa: E402
from KHz_filament.ionization.lut import _ion_rate_table_defaults, _table_signature  # noqa: E402
from KHz_filament.ionization.runtime import _talebpour_defaults  # noqa: E402


DEFAULT_CONFIG = FILAMENT_ROOT / "configs" / "profile_validation" / "flat_top_90_40fs.json"
RATE_MODELS = {
    "popruzhenko": ("popruzhenko_atom_i_full_reference", "popruzhenko_atom_i_lut"),
    "talebpour": ("ppt_talebpour_i_full_reference", "ppt_talebpour_i_lut"),
}
RELEVANT_I_MIN_W_M2 = 1e16
RELEVANT_I_MAX_W_M2 = 1e18


def repo_relative(path: Path) -> str:
    """Return a repository-relative path suitable for durable result metadata."""
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _timestamped_output_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"ionization_rate_model_validation_{stamp}"


def _as_numpy(values: Any) -> np.ndarray:
    return np.asarray(to_cpu(values), dtype=np.float64)


def _finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _production_species(config_path: Path) -> tuple[Any, Any, list[dict[str, Any]]]:
    _grid, beam, _prop, ion, _heat, _run, _raman = load_all(str(config_path))
    species = [deepcopy(dict(item)) for item in (ion.species or [])]
    names = {str(item.get("name", "")).upper() for item in species}
    if names != {"N2", "O2"}:
        raise ValueError(f"expected FT90 N2/O2 production species, got {sorted(names)}")
    return beam, ion, species


def model_species_parameters(production_species: dict[str, Any], family: str) -> dict[str, Any]:
    """Build one species dictionary using production runtime parameter rules."""
    item = deepcopy(production_species)
    name = str(item["name"]).upper()
    item["fraction"] = 1.0
    if family == "popruzhenko":
        item.pop("Zeff", None)
        item["Ip_eV"] = float(item["Ip_eV"])
        item["Z"] = int(item.get("Z", 1))
    elif family == "talebpour":
        # Do not fit Talebpour parameters.  Calling the runtime helper keeps
        # the N2/O2 defaults exactly aligned with make_Wfunc.
        Ip_use, Zeff_use = _talebpour_defaults(
            name=name,
            Ip_eV=item.get("Ip_eV"),
            Ip_eV_eff=item.get("Ip_eV_eff"),
            Zeff=item.get("Zeff"),
        )
        item["Ip_eV_eff"] = float(Ip_use)
        item["Zeff"] = float(Zeff_use)
        item.setdefault("Ip_eV", float(production_species["Ip_eV"]))
        item.pop("Z", None)
    else:
        raise ValueError(f"unknown rate family: {family}")
    item.setdefault("l", 0)
    item.setdefault("m", 0)
    return item


def _one_species_ionization_config(production_ion: Any, species: dict[str, Any], rate: str, *, cache_dir: Path,
                                   cycle_avg_samples: int | None = None) -> IonizationConfig:
    result = IonizationConfig(
        species=[deepcopy(species)],
        time_mode=str(production_ion.time_mode),
        integrator=str(production_ion.integrator),
        cycle_avg_samples=int(production_ion.cycle_avg_samples if cycle_avg_samples is None else cycle_avg_samples),
        mean_clip_frac=float(production_ion.mean_clip_frac),
        beta_rec=float(production_ion.beta_rec),
        sigma_ib=float(production_ion.sigma_ib),
        nu_ei_const=production_ion.nu_ei_const,
        I_cap=float(production_ion.I_cap),
        W_cap=float(production_ion.W_cap),
    )
    result.species[0]["rate"] = rate
    result.species[0]["reference_model"] = RATE_MODELS["talebpour" if rate.startswith("ppt_") else "popruzhenko"][0]
    table = deepcopy(_ion_rate_table_defaults(production_ion))
    table["cache_dir"] = str(cache_dir)
    table["save_tables"] = True
    result.rate_table = table
    return result


def make_rate_evaluator(production_ion: Any, beam: Any, production_species: dict[str, Any], rate: str, *, cache_dir: Path,
                        cycle_avg_samples: int | None = None) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Return a production-runtime evaluator and its resolved metadata."""
    family = "talebpour" if rate.startswith("ppt_") else "popruzhenko"
    species = model_species_parameters(production_species, family)
    species["rate"] = rate
    species["reference_model"] = RATE_MODELS[family][0]
    ion = _one_species_ionization_config(
        production_ion, species, rate, cache_dir=cache_dir, cycle_avg_samples=cycle_avg_samples
    )
    omega0 = 2.0 * math.pi * float(c0) / float(beam.lam0)
    Wfunc = make_Wfunc("ionization_rate_model_validation", ion, omega0, float(beam.n0))

    def evaluator(I_W_m2: np.ndarray) -> np.ndarray:
        return _as_numpy(Wfunc(np.asarray(I_W_m2, dtype=np.float64)))

    table_cfg = _ion_rate_table_defaults(ion)
    table_cfg_metadata = deepcopy(table_cfg)
    cache_path = Path(str(table_cfg_metadata["cache_dir"])).resolve()
    try:
        table_cfg_metadata["cache_dir"] = repo_relative(cache_path)
    except ValueError:
        table_cfg_metadata["cache_dir"] = "external_cache_path"
    table_signature = None
    entry = Wfunc._species_entries[0]
    if rate.endswith("_lut"):
        runtime = entry["W_runtime"]
        # Evaluating once forces the production LUT construction/cache lookup.
        _ = runtime(np.asarray([max(float(table_cfg["I_min_SI"]), 1.0)], dtype=np.float64))
        table = runtime.__defaults__[0] if runtime.__defaults__ else None
        if isinstance(table, dict) and "metadata" in table:
            table_signature = _table_signature(table["metadata"])
    return evaluator, {
        "rate": rate,
        "family": family,
        "species": species,
        "cycle_avg_samples": int(ion.cycle_avg_samples),
        "W_cap_s-1": float(ion.W_cap),
        "rate_table": table_cfg_metadata,
        "lut_signature": table_signature,
    }


def meaningful_relative_error(reference: np.ndarray, candidate: np.ndarray, floor_s_1: float) -> np.ndarray:
    """Relative error only where the reference rate is physically meaningful."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    result = np.full(reference.shape, np.nan, dtype=np.float64)
    mask = reference >= float(floor_s_1)
    result[mask] = np.abs(candidate[mask] - reference[mask]) / reference[mask]
    return result


def lut_error_statistics(I_W_m2: np.ndarray, reference: np.ndarray, candidate: np.ndarray, *, meaningful_floor_s_1: float,
                         relevant_I_min_W_m2: float = RELEVANT_I_MIN_W_M2, relevant_I_max_W_m2: float = RELEVANT_I_MAX_W_M2) -> dict[str, Any]:
    """Compute denominator-safe LUT accuracy statistics for all and relevant ranges."""
    I = np.asarray(I_W_m2, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    lut = np.asarray(candidate, dtype=np.float64)
    rel = meaningful_relative_error(ref, lut, meaningful_floor_s_1)

    def stats(mask: np.ndarray) -> dict[str, Any]:
        valid = mask & np.isfinite(rel)
        if not np.any(valid):
            return {"sample_count": 0, "max_relative_error": None, "median_relative_error": None, "max_error_I_W_m2": None}
        idx = np.flatnonzero(valid)
        best = int(idx[np.argmax(rel[valid])])
        return {
            "sample_count": int(idx.size),
            "max_relative_error": float(np.max(rel[valid])),
            "median_relative_error": float(np.median(rel[valid])),
            "max_error_I_W_m2": float(I[best]),
        }

    return {
        "meaningful_floor_s-1": float(meaningful_floor_s_1),
        "full_scan": stats(np.ones(I.shape, dtype=bool)),
        "relevant_interval": stats((I >= relevant_I_min_W_m2) & (I <= relevant_I_max_W_m2)),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in rows])


def _maybe_write_plots(out_dir: Path, curves: dict[str, dict[str, np.ndarray]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: list[str] = []
    for name, values in curves.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        I = values["I_W_m2"]
        ax.loglog(I, values["W_popruzhenko_s-1"], label="Popruzhenko reference")
        ax.loglog(I, values["W_talebpour_s-1"], label="Talebpour reference")
        ax.set_xlabel("Peak intensity (W/m²)")
        ax.set_ylabel("Ionization rate (s⁻¹)")
        ax.set_title(name)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"ionization_rate_{name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path.name)
    return written


def run_rate_model_comparison(config_path: Path = DEFAULT_CONFIG, *, I_min_W_m2: float = 1e14, I_max_W_m2: float = 1e19,
                              n_points: int = 501, meaningful_floor_s_1: float = 1.0, out_dir: Path | None = None,
                              make_plots: bool = False) -> dict[str, Any]:
    """Evaluate Popruzhenko and Talebpour reference/LUT curves for N2 and O2."""
    if I_min_W_m2 <= 0.0 or I_max_W_m2 <= I_min_W_m2 or int(n_points) < 2:
        raise ValueError("require 0 < I_min_W_m2 < I_max_W_m2 and n_points >= 2")
    config_path = Path(config_path).resolve()
    if out_dir is None:
        out_dir = _timestamped_output_dir(FILAMENT_ROOT / "results" / "ionization_rate_model_validation")
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    lut_cache_dir = out_dir / "lut_cache"
    I = np.logspace(math.log10(float(I_min_W_m2)), math.log10(float(I_max_W_m2)), int(n_points))
    beam, production_ion, production_species_list = _production_species(config_path)
    production_species = {str(item["name"]).upper(): item for item in production_species_list}
    curves: dict[str, dict[str, np.ndarray]] = {}
    metadata_models: dict[str, Any] = {}
    validation_rows: list[dict[str, Any]] = []

    for name in ("N2", "O2"):
        species = production_species[name]
        pop_ref, pop_ref_meta = make_rate_evaluator(production_ion, beam, species, RATE_MODELS["popruzhenko"][0], cache_dir=lut_cache_dir)
        pop_lut, pop_lut_meta = make_rate_evaluator(production_ion, beam, species, RATE_MODELS["popruzhenko"][1], cache_dir=lut_cache_dir)
        tal_ref, tal_ref_meta = make_rate_evaluator(production_ion, beam, species, RATE_MODELS["talebpour"][0], cache_dir=lut_cache_dir)
        tal_lut, tal_lut_meta = make_rate_evaluator(production_ion, beam, species, RATE_MODELS["talebpour"][1], cache_dir=lut_cache_dir)
        lut_reference_samples = int(pop_lut_meta["rate_table"]["ref_cycle_avg_samples"])
        pop_lut_reference, pop_lut_reference_meta = make_rate_evaluator(
            production_ion, beam, species, RATE_MODELS["popruzhenko"][0], cache_dir=lut_cache_dir,
            cycle_avg_samples=lut_reference_samples,
        )
        tal_lut_reference, tal_lut_reference_meta = make_rate_evaluator(
            production_ion, beam, species, RATE_MODELS["talebpour"][0], cache_dir=lut_cache_dir,
            cycle_avg_samples=lut_reference_samples,
        )
        W_pop_ref, W_pop_lut, W_pop_lut_reference = pop_ref(I), pop_lut(I), pop_lut_reference(I)
        W_tal_ref, W_tal_lut, W_tal_lut_reference = tal_ref(I), tal_lut(I), tal_lut_reference(I)
        for label, values in {
            "W_pop_ref": W_pop_ref, "W_pop_lut": W_pop_lut, "W_pop_lut_reference": W_pop_lut_reference,
            "W_tal_ref": W_tal_ref, "W_tal_lut": W_tal_lut, "W_tal_lut_reference": W_tal_lut_reference,
        }.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite values from {name} {label}")
        # LUT accuracy is measured against the exact evaluator used to build
        # the table (normally ref_cycle_avg_samples=64), not against the
        # production curve's lower phase-sampling count.  The latter remains
        # in the CSV as a separate runtime-comparability quantity.
        pop_rel = meaningful_relative_error(W_pop_lut_reference, W_pop_lut, meaningful_floor_s_1)
        tal_rel = meaningful_relative_error(W_tal_lut_reference, W_tal_lut, meaningful_floor_s_1)
        log_floor = max(float(meaningful_floor_s_1), np.finfo(np.float64).tiny)
        curve = {
            "I_W_m2": I,
            "W_popruzhenko_s-1": W_pop_ref,
            "W_talebpour_s-1": W_tal_ref,
            "W_popruzhenko_reference_s-1": W_pop_ref,
            "W_popruzhenko_lut_s-1": W_pop_lut,
            "W_popruzhenko_lut_validation_reference_s-1": W_pop_lut_reference,
            "W_talebpour_reference_s-1": W_tal_ref,
            "W_talebpour_lut_s-1": W_tal_lut,
            "W_talebpour_lut_validation_reference_s-1": W_tal_lut_reference,
            "popruzhenko_lut_relative_error": pop_rel,
            "talebpour_lut_relative_error": tal_rel,
            "popruzhenko_lut_abs_log10_rate_difference": np.abs(np.log10(np.maximum(W_pop_lut, log_floor)) - np.log10(np.maximum(W_pop_ref, log_floor))),
            "talebpour_lut_abs_log10_rate_difference": np.abs(np.log10(np.maximum(W_tal_lut, log_floor)) - np.log10(np.maximum(W_tal_ref, log_floor))),
            "log10_W_pop_minus_talebpour": np.log10(np.maximum(W_pop_ref, log_floor)) - np.log10(np.maximum(W_tal_ref, log_floor)),
            "W_pop_over_talebpour": W_pop_ref / np.maximum(W_tal_ref, log_floor),
        }
        curves[name] = curve
        _write_csv(out_dir / f"ionization_rate_{name}.csv", [
            {key: float(values[index]) for key, values in curve.items()} for index in range(I.size)
        ])
        for family, ref_values, lut_values, model_meta in (
            ("popruzhenko", W_pop_lut_reference, W_pop_lut, pop_lut_meta),
            ("talebpour", W_tal_lut_reference, W_tal_lut, tal_lut_meta),
        ):
            stats = lut_error_statistics(I, ref_values, lut_values, meaningful_floor_s_1=meaningful_floor_s_1)
            for scope, values in stats.items():
                if scope == "meaningful_floor_s-1":
                    continue
                validation_rows.append({
                    "species": name,
                    "family": family,
                    "scope": scope,
                    "meaningful_floor_s-1": meaningful_floor_s_1,
                    "max_relative_error": values["max_relative_error"],
                    "median_relative_error": values["median_relative_error"],
                    "max_error_I_W_m2": values["max_error_I_W_m2"],
                    "meaningful_sample_count": values["sample_count"],
                    "accepted_max_relative_error": 0.03,
                    "lut_pass": bool(values["max_relative_error"] is not None and values["max_relative_error"] <= 0.03),
                    "lut_signature": model_meta["lut_signature"],
                })
        metadata_models[name] = {
            "production_species": production_species[name],
            "popruzhenko": {"reference": pop_ref_meta, "lut": pop_lut_meta},
            "talebpour": {"reference": tal_ref_meta, "lut": tal_lut_meta},
            "lut_validation_references": {
                "cycle_avg_samples": lut_reference_samples,
                "popruzhenko": pop_lut_reference_meta,
                "talebpour": tal_lut_reference_meta,
            },
        }

    _write_csv(out_dir / "ionization_rate_lut_validation.csv", validation_rows)
    metadata = {
        "schema": "khz_filament.ionization_rate_model_validation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_sha": _git_sha(),
        "config_path": repo_relative(config_path),
        "config_sha256": _sha256(config_path),
        "intensity_unit": "W/m^2",
        "intensity_scan": {"I_min_W_m2": float(I_min_W_m2), "I_max_W_m2": float(I_max_W_m2), "n_points": int(n_points)},
        "relevant_intensity_range_W_m2": [RELEVANT_I_MIN_W_M2, RELEVANT_I_MAX_W_M2],
        "W_meaningful_floor_s-1": float(meaningful_floor_s_1),
        "lut_acceptance_max_relative_error": 0.03,
        "models": metadata_models,
        "outputs": ["ionization_rate_N2.csv", "ionization_rate_O2.csv", "ionization_rate_lut_validation.csv"],
    }
    metadata_path = out_dir / "ionization_rate_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if make_plots:
        metadata["outputs"].extend(_maybe_write_plots(out_dir, curves))
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"out_dir": out_dir, "metadata": metadata, "curves": curves, "lut_validation_rows": validation_rows}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--I-min", dest="I_min_W_m2", type=float, default=1e14)
    parser.add_argument("--I-max", dest="I_max_W_m2", type=float, default=1e19)
    parser.add_argument("--n-points", type=int, default=501)
    parser.add_argument("--W-meaningful-floor-s-1", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--plots", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_rate_model_comparison(
        args.config,
        I_min_W_m2=args.I_min_W_m2,
        I_max_W_m2=args.I_max_W_m2,
        n_points=args.n_points,
        meaningful_floor_s_1=args.W_meaningful_floor_s_1,
        out_dir=args.out_dir,
        make_plots=args.plots,
    )
    print(f"ionization rate-model validation written to {result['out_dir']}")


if __name__ == "__main__":
    main()

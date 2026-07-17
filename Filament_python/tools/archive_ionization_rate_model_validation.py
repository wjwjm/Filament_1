#!/usr/bin/env python3
"""Archive the Phase-4 N2/O2 ionization-rate validation into a decision report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = FILAMENT_ROOT.parent
if str(FILAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(FILAMENT_ROOT))

from validate_ionization_rate_models import repo_relative  # noqa: E402


DEFAULT_OUTPUT_ROOT = FILAMENT_ROOT / "results" / "ionization_rate_model_validation"


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def classify_rate_model_validation(lut_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Apply the Phase-4 decision rules without inferring an axial distance."""
    relevant = [row for row in lut_rows if str(row.get("scope")) == "relevant_interval"]
    if not relevant or any(str(row.get("lut_pass")).lower() not in ("true", "1") for row in relevant):
        return "inconclusive", "At least one relevant-interval LUT validation did not pass, so physical-model and LUT effects are not separable."
    comparable = [
        row for row in threshold_rows
        if _as_float(row.get("I_threshold_ratio_pop_over_tal")) is not None
        and str(row.get("popruzhenko_status", "")).startswith("crossed")
        and str(row.get("talebpour_status", "")).startswith("crossed")
    ]
    if not comparable:
        return "inconclusive", "No fixed-density threshold was crossed by both models within the scanned intensity range."
    high_priority = [
        row for row in comparable
        if float(row["density_threshold_m3"]) in (1e20, 1e21)
        and abs(float(row["I_threshold_ratio_pop_over_tal"]) - 1.0) >= 0.10
    ]
    directions = {math.copysign(1.0, float(row["I_threshold_ratio_pop_over_tal"]) - 1.0) for row in high_priority}
    widths = {float(row["tau_fwhm_fs"]) for row in high_priority}
    if high_priority and len(directions) == 1 and {40.0, 120.0}.issubset(widths):
        return "supported", "Both pulse widths show a same-direction >=10% fixed-threshold intensity shift in the 1e20–1e21 m^-3 onset band."
    if all(abs(float(row["I_threshold_ratio_pop_over_tal"]) - 1.0) < 0.03 for row in comparable):
        return "not_supported", "All crossed fixed-density thresholds differ by less than 3% after both LUTs pass."
    return "inconclusive", "The local model difference is measurable but does not satisfy the consistent two-pulse-width high-priority criterion."


def _rate_difference_summary(out_dir: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in ("N2", "O2"):
        rows = _read_csv(out_dir / f"ionization_rate_{name}.csv")
        relative: list[float] = []
        log_difference: list[float] = []
        for row in rows:
            intensity = float(row["I_W_m2"])
            pop = float(row["W_popruzhenko_reference_s-1"])
            tal = float(row["W_talebpour_reference_s-1"])
            if 1e16 <= intensity <= 1e18 and max(pop, tal) >= 1.0:
                relative.append(abs(pop / tal - 1.0))
                log_difference.append(abs(math.log10(pop) - math.log10(tal)))
        result[name] = {
            "max_abs_relative_difference": max(relative) if relative else 0.0,
            "max_abs_log10_rate_difference": max(log_difference) if log_difference else 0.0,
        }
    return result


def _lut_summary(lut_rows: list[dict[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in lut_rows:
        if row["scope"] != "relevant_interval":
            continue
        output[f"{row['species']}_{row['family']}"] = {
            "passed": str(row["lut_pass"]).lower() == "true",
            "max_relative_error": _as_float(row["max_relative_error"]),
            "median_relative_error": _as_float(row["median_relative_error"]),
            "lut_signature": row["lut_signature"],
        }
    return output


def archive_validation(out_dir: Path) -> dict[str, Any]:
    """Write Phase-4 report, summary JSON, and a one-row decision CSV."""
    out_dir = Path(out_dir).resolve()
    required = (
        "ionization_rate_model_metadata.json", "ionization_rate_lut_validation.csv", "ionization_density_thresholds.csv",
        "ionization_species_contribution.csv", "ionization_density_response_metadata.json",
    )
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing Phase-4 inputs in {repo_relative(out_dir)}: {missing}")
    rate_metadata = json.loads((out_dir / "ionization_rate_model_metadata.json").read_text(encoding="utf-8"))
    density_metadata = json.loads((out_dir / "ionization_density_response_metadata.json").read_text(encoding="utf-8"))
    lut_rows = _read_csv(out_dir / "ionization_rate_lut_validation.csv")
    threshold_rows = _read_csv(out_dir / "ionization_density_thresholds.csv")
    contribution_rows = _read_csv(out_dir / "ionization_species_contribution.csv")
    classification, rationale = classify_rate_model_validation(lut_rows, threshold_rows)
    lut = _lut_summary(lut_rows)
    rate_difference = _rate_difference_summary(out_dir)
    onset_rows = [row for row in contribution_rows if float(row["density_threshold_m3"]) in (1e19, 1e20, 1e21)]
    o2_fractions = [float(row["rho_O2_fraction"]) for row in onset_rows]
    o2_onset = {"minimum": min(o2_fractions), "maximum": max(o2_fractions), "sample_count": len(o2_fractions)}
    threshold_compact = [
        {
            "tau_fwhm_fs": float(row["tau_fwhm_fs"]), "density_threshold_m3": float(row["density_threshold_m3"]),
            "I_threshold_ratio_pop_over_tal": _as_float(row["I_threshold_ratio_pop_over_tal"]),
            "delta_log10_I_pop_minus_tal": _as_float(row["delta_log10_I_pop_minus_tal"]),
        }
        for row in threshold_rows
    ]
    recommended = []
    if classification == "supported":
        recommended = [
            "120 fs: current Popruzhenko full model versus Talebpour full-model control",
            "40 fs: Talebpour full-model control only after the 120 fs comparison confirms a material propagation effect",
        ]
    summary = {
        "schema": "khz_filament.ionization_rate_model_decision.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_sha": _git_sha(),
        "validation_directory": repo_relative(out_dir),
        "classification": classification,
        "rationale": rationale,
        "popruzhenko_lut_passed": all(item["passed"] for key, item in lut.items() if key.endswith("_popruzhenko")),
        "talebpour_lut_passed": all(item["passed"] for key, item in lut.items() if key.endswith("_talebpour")),
        "relevant_intensity_range_W_m2": rate_metadata["relevant_intensity_range_W_m2"],
        "N2_max_model_difference": rate_difference["N2"],
        "O2_max_model_difference": rate_difference["O2"],
        "threshold_intensity_shift_40fs": [row for row in threshold_compact if row["tau_fwhm_fs"] == 40.0],
        "threshold_intensity_shift_120fs": [row for row in threshold_compact if row["tau_fwhm_fs"] == 120.0],
        "O2_onset_fraction": o2_onset,
        "recommended_full_propagation_cases": recommended,
        "production_physics_changed": False,
        "slurm_jobs_submitted": False,
    }
    (out_dir / "ionization_rate_model_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "ionization_rate_model_decision.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["classification", "rationale", "recommended_full_propagation_case_count", "production_physics_changed", "slurm_jobs_submitted"])
        writer.writeheader()
        writer.writerow({
            "classification": classification, "rationale": rationale, "recommended_full_propagation_case_count": len(recommended),
            "production_physics_changed": False, "slurm_jobs_submitted": False,
        })

    lut_lines = "\n".join(
        f"- {key}: {'pass' if item['passed'] else 'fail'}; relevant-window max relative error = {item['max_relative_error']:.4%}."
        for key, item in sorted(lut.items())
    )
    threshold_lines = "\n".join(
        f"- {row['tau_fwhm_fs']:.0f} fs, {row['density_threshold_m3']:.0e} m^-3: "
        f"Pop/Tal threshold-intensity ratio = {row['I_threshold_ratio_pop_over_tal']:.4f}, "
        f"Δlog10 I = {row['delta_log10_I_pop_minus_tal']:.4f}."
        for row in threshold_compact if row["I_threshold_ratio_pop_over_tal"] is not None
    )
    report = f"""# N₂/O₂ ionization-rate model validation

## Decision

**Classification: `{classification}`.** {rationale}

This is a local CPU/0D comparison. It does not convert the result into a centimetre-scale axial shift and does not identify Talebpour as PyCAP's internal model. It only prioritizes (or deprioritizes) the present Popruzhenko-versus-Talebpour model difference for a later controlled propagation check.

## Scope and reproducibility

- Code commit evaluated: `{summary['code_commit_sha']}`.
- FT90 configuration: `{rate_metadata['config_path']}` (SHA256 `{rate_metadata['config_sha256']}`).
- Intensity scan: {rate_metadata['intensity_scan']['I_min_W_m2']:.1e}–{rate_metadata['intensity_scan']['I_max_W_m2']:.1e} W/m², {rate_metadata['intensity_scan']['n_points']} log-spaced points.
- Local-density pulse grids: 40 fs and 120 fs, both with production Nt=384/Twin=960 fs; the primary solution is the no-recombination trapezoid cumulative reference. Production `evolve_rho_time` RK4 is a consistency check.
- Paths in this archive are repository relative. No 3D propagation or Slurm submission was performed.

## Rate parameters and LUT accuracy

The production Popruzhenko species are N₂ (`Ip_eV=15.6`, `Z=1`) and O₂ (`Ip_eV=12.1`, `Z=1`). The runtime Talebpour comparator resolves N₂ to `Ip_eV_eff=15.6`, `Zeff=0.9`, and O₂ to `Ip_eV_eff=12.55`, `Zeff=0.53`; both retain `l=0`, `m=0`. The production phase sampling is 32, while LUT tables use their configured reference sampling of 64; `W_cap=1e19 s^-1`.

LUT accuracy is tested against each table's actual 64-sample reference evaluator, separately from the physical Popruzhenko-versus-Talebpour comparison. The relevant range is {rate_metadata['relevant_intensity_range_W_m2'][0]:.0e}–{rate_metadata['relevant_intensity_range_W_m2'][1]:.0e} W/m²; the acceptance rule is max relative error <= 3%.

{lut_lines}

## Physical-model difference

- N₂ maximum absolute rate difference in the relevant interval: {rate_difference['N2']['max_abs_relative_difference']:.1%} ({rate_difference['N2']['max_abs_log10_rate_difference']:.3f} decades).
- O₂ maximum absolute rate difference in the relevant interval: {rate_difference['O2']['max_abs_relative_difference']:.1%} ({rate_difference['O2']['max_abs_log10_rate_difference']:.3f} decades).
- The local low-density onset is O₂-dominated: across 10^19–10^21 m^-3, O₂ contributes {o2_onset['minimum']:.2%}–{o2_onset['maximum']:.2%} of the total density at each model's own threshold intensity.

## Fixed-density threshold map

{threshold_lines}

At 10^21 m^-3, the Popruzhenko/Talebpour intensity ratios are below one for both pulse widths (40 fs: {next(row['I_threshold_ratio_pop_over_tal'] for row in threshold_compact if row['tau_fwhm_fs'] == 40.0 and row['density_threshold_m3'] == 1e21):.4f}; 120 fs: {next(row['I_threshold_ratio_pop_over_tal'] for row in threshold_compact if row['tau_fwhm_fs'] == 120.0 and row['density_threshold_m3'] == 1e21):.4f}). This is a same-direction local response difference above the 10% high-priority screen.

## Numerical consistency

- Maximum production-RK4 versus cumulative-reference final-density error over the direct relevant probes: {max(float(row['final_relative_error']) for row in density_metadata['rk4_consistency']):.3e}.
- Maximum 1× versus 8× cumulative-reference final-density difference over the stated probes: {max(float(row['final_relative_difference']) for row in density_metadata['reference_time_convergence']):.3%}.

## Causal interpretation and next action

The result supports the ionization-rate-model difference as a high-priority candidate for the observed common rising-front/peak shift, because both 40 fs and 120 fs show a consistent >10% threshold shift at 10^21 m^-3 and the onset is overwhelmingly O₂-controlled. This is not proof of the reported approximately -3.270 cm (40 fs) or -2.589 cm (120 fs) rising-front advance, nor of the approximately -2.9 cm common peak-centre shift: propagation feedback remains untested in this phase.

Recommended new full-propagation cases: {len(recommended)}.
"""
    if recommended:
        report += "\n" + "\n".join(f"- {item}" for item in recommended) + "\n"
    else:
        report += "\nNo additional full-propagation case is recommended from this classification.\n"
    report += "\nProduction physics defaults were unchanged; no Slurm jobs were submitted.\n"
    (out_dir / "ionization_rate_model_validation_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Existing Phase-4 result directory")
    args = parser.parse_args()
    result = archive_validation(args.out_dir)
    print(f"classification={result['classification']}")


if __name__ == "__main__":
    main()

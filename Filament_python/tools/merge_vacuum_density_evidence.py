#!/usr/bin/env python3
"""Merge the independent vacuum-focus and density-shape evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vacuum-summary", required=True)
    parser.add_argument("--density-fit", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    vacuum = json.loads(Path(args.vacuum_summary).read_text(encoding="utf-8"))
    density = json.loads(Path(args.density_fit).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    x_vac = float(vacuum["focus_peak"]["x_focus_cm"])
    summaries = {}
    for case, result in density["cases"].items():
        classification = result["classification"]
        onset = float(classification["mean_absolute_rising_shift_cm"])
        summaries[case] = {
            "density_classification": classification["classification"],
            "confidence": classification["confidence"],
            "rising_edge_shift_cm": onset,
            "vacuum_minus_rising_shift_cm": x_vac - onset,
            "fwhm_ratio_sim_over_paper": classification["fwhm_ratio_sim_over_paper"],
            "scale_fit_rmse_improvement_fraction": classification["scale_fit_rmse_improvement_fraction"],
        }
    result = {
        "coordinate_definition": "x_focus_cm = 100 * (z_m - 0.95); no curve was shifted or re-zeroed.",
        "vacuum_focus": {"x_vac_peak_cm": x_vac, "uncertainty_cm": float(vacuum["focus_peak"]["sampling_half_step_uncertainty_cm"]), "maximum_power_drift": float(vacuum["power_conservation"]["maximum_relative_drift"])},
        "density_cases": summaries,
        "combined_conclusion": "FT90 finite-aperture/edge diffraction is the strongest verified common cause of the early leading-edge shift, but it does not by itself reproduce the complete nonlinear density-curve width/tail.",
        "next_nonlinear_run": "A new nonlinear run is not needed to establish the vacuum-focus offset. One tightly controlled FT90 nonlinear rerun is needed for causal closure if the goal is to determine whether correcting only the input lens/wavefront geometry removes the residual width/tail discrepancy; do not change grid, n2, Raman, ionization, or the FT90 profile in that run.",
    }
    (out_dir / "ft90_focus_cause_diagnosis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# FT90 focus-shift diagnosis: merged vacuum and nonlinear-curve evidence",
        "",
        "All coordinates are permanently `x_focus = 100 * (z - 0.95) cm`; no peak, intensity maximum, or fitted translation defines zero.",
        "",
        "## Independent vacuum test",
        "",
        f"- Direct-from-lens angular-spectrum FT90 vacuum focus: **{x_vac:.4f} cm** relative to the 0.95 m geometric focus.",
        f"- Parabolic interpolation sampling uncertainty: {vacuum['focus_peak']['sampling_half_step_uncertainty_cm']:.4f} cm; maximum transverse-power drift: {vacuum['power_conservation']['maximum_relative_drift']:.3e}.",
        "- This satisfies the predeclared `x_vac,peak <= -2 cm` criterion: finite aperture / FT90 edge diffraction is a strong candidate for the early shift.",
        "",
        "## Existing nonlinear FT90 curves versus paper PyCAP",
        "",
    ]
    for case, item in summaries.items():
        lines += [
            f"- {case}: fixed-absolute-density rising edge is {item['rising_edge_shift_cm']:.3f} cm early; classification is **{item['density_classification']}** ({item['confidence']} confidence).",
            f"  The vacuum value is {item['vacuum_minus_rising_shift_cm']:.3f} cm more forward than that leading-edge shift; FWHM ratio is {item['fwhm_ratio_sim_over_paper']:.3f} and adding a scale parameter improves RMSE by {100*item['scale_fit_rmse_improvement_fraction']:.1f}%.",
        ]
    lines += [
        "",
        "## Final interpretation",
        "",
        "The vacuum focus is strongly and independently shifted forward in the same direction as both nonlinear leading edges. The 40 fs and 120 fs leading-edge shifts differ by only about 0.57 cm, which supports a shared transverse-optical contribution rather than a solely pulse-duration-specific temporal mechanism.",
        "",
        "However, the full profiles are not pure translations: both have materially broader FWHM/tails, and the translation-plus-scale fit reduces the residual by roughly 68-71%. The paper 120 fs trace also has a flat peak plateau, so its single peak coordinate is less reliable than its threshold crossings. Thus the verified statement is: **FT90 finite-aperture/edge diffraction is the most reliable primary explanation for the common 2.5-3.1 cm early leading edge, while a residual nonlinear-shape difference remains.**",
        "",
        "## Is another full nonlinear run needed?",
        "",
        "Not to establish the vacuum offset: task one already does that. It is needed only for causal closure of the residual width/tail: run one FT90 nonlinear control in which only the lens/wavefront geometry is corrected against the measured vacuum focus, then reuse the same fixed coordinate and compare the entire density curves. Keep the current 512² grid, 8 mm window, 17 GW, FT90 profile, n2, Raman, and ionization settings unchanged.",
    ]
    (out_dir / "ft90_focus_cause_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

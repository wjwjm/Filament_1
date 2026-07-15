#!/usr/bin/env python3
"""Quantitatively distinguish longitudinal translation from broadening.

Both the paper PyCAP trace and the current FT90 result remain in the fixed
geometric-focus coordinate x_focus = 100 * (z - 0.95) cm.  No feature is
aligned before extracting landmarks or fitting the models.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ABSOLUTE_LEVELS = (0.1, 0.2, 0.5, 1.0, 2.0)  # 1e16 cm^-3
RELATIVE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass
class Curve:
    name: str
    x: np.ndarray
    rho: np.ndarray


def load_csv_curve(path: Path, name: str) -> Curve:
    values = np.genfromtxt(path, delimiter=",", names=True)
    return Curve(name, np.asarray(values["x_focus_cm"], dtype=float), np.asarray(values["rho_1e16_cm3"], dtype=float))


def load_sim_curve(path: Path, name: str) -> Curve:
    with np.load(path, allow_pickle=False) as data:
        x = 100.0 * (np.asarray(data["z_axis"], dtype=float) - 0.95)
        rho = np.asarray(data["rho_max_z"], dtype=float) / 1e22
    in_window = (x >= -20.0) & (x <= 20.0)
    return Curve(name, x[in_window], rho[in_window])


def _sorted_unique(curve: Curve) -> Curve:
    order = np.argsort(curve.x)
    x, rho = curve.x[order], np.maximum(curve.rho[order], 0.0)
    unique, inverse = np.unique(x, return_inverse=True)
    if unique.size == x.size:
        return Curve(curve.name, x, rho)
    summed = np.zeros_like(unique)
    count = np.zeros_like(unique)
    np.add.at(summed, inverse, rho)
    np.add.at(count, inverse, 1)
    return Curve(curve.name, unique, summed / count)


def evaluate(curve: Curve, x: np.ndarray) -> np.ndarray:
    curve = _sorted_unique(curve)
    return np.interp(x, curve.x, curve.rho, left=0.0, right=0.0)


def crossing(curve: Curve, level: float, direction: str) -> float | None:
    curve = _sorted_unique(curve)
    y = curve.rho - level
    if direction == "rising":
        hits = np.where((y[:-1] < 0.0) & (y[1:] >= 0.0))[0]
        index = int(hits[0]) if hits.size else None
    elif direction == "falling":
        hits = np.where((y[:-1] >= 0.0) & (y[1:] < 0.0))[0]
        index = int(hits[-1]) if hits.size else None
    else:
        raise ValueError(direction)
    if index is None:
        return None
    x0, x1 = curve.x[index:index + 2]
    y0, y1 = curve.rho[index:index + 2]
    if y1 == y0:
        return float(x0)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def curve_features(curve: Curve) -> dict:
    curve = _sorted_unique(curve)
    peak_index = int(np.argmax(curve.rho))
    peak = float(curve.rho[peak_index])
    output = {"peak_rho_1e16_cm3": peak, "x_peak_cm": float(curve.x[peak_index]), "area_1e16_cm3_cm": float(np.trapezoid(curve.rho, curve.x))}
    for level in ABSOLUTE_LEVELS:
        output[f"absolute_{level:g}_rising_cm"] = crossing(curve, level, "rising") if peak >= level else None
    for fraction in RELATIVE_LEVELS:
        level = fraction * peak
        output[f"relative_{fraction:g}_rising_cm"] = crossing(curve, level, "rising")
        output[f"relative_{fraction:g}_falling_cm"] = crossing(curve, level, "falling")
    x10 = output["relative_0.1_rising_cm"]
    x90 = output["relative_0.9_rising_cm"]
    x50_up = output["relative_0.5_rising_cm"]
    x50_down = output["relative_0.5_falling_cm"]
    x10_down = output["relative_0.1_falling_cm"]
    output["rise_10_90_cm"] = None if x10 is None or x90 is None else x90 - x10
    output["fwhm_cm"] = None if x50_up is None or x50_down is None else x50_down - x50_up
    output["post_peak_decay_to_10pct_cm"] = None if x10_down is None else x10_down - output["x_peak_cm"]
    return output


def _fit_models(paper: Curve, sim: Curve) -> dict:
    from scipy.optimize import least_squares

    x = np.linspace(-20.0, 20.0, 1601)
    target = evaluate(sim, x)

    def paper_at(argument):
        return evaluate(paper, np.asarray(argument, dtype=float))

    def fit_translation(params):
        amplitude, delta_x = params
        return amplitude * paper_at(x - delta_x) - target

    def fit_scale(params):
        amplitude, x_c, scale = params
        return amplitude * paper_at((x - x_c) / scale) - target

    translation = least_squares(fit_translation, x0=(1.0, -3.0), bounds=([0.0, -12.0], [3.0, 12.0]))
    scaled = least_squares(fit_scale, x0=(1.0, -3.0, 1.0), bounds=([0.0, -12.0, 0.4], [3.0, 12.0, 2.5]))

    def record(result, k: int, names):
        residual = result.fun
        rss = float(np.sum(residual ** 2))
        n = residual.size
        rmse = float(np.sqrt(rss / n))
        return {"parameters": {name: float(value) for name, value in zip(names, result.x)}, "rmse_1e16_cm3": rmse, "rss": rss, "aic": float(n * np.log(max(rss / n, 1e-300)) + 2 * k), "bic": float(n * np.log(max(rss / n, 1e-300)) + k * np.log(n)), "fitted_rho_1e16_cm3": (target + residual).tolist(), "residual_1e16_cm3": residual.tolist()}

    first = record(translation, 2, ("amplitude_A", "delta_x_cm"))
    second = record(scaled, 3, ("amplitude_A", "x_c_cm", "scale_s"))
    return {"x_focus_cm": x.tolist(), "sim_rho_1e16_cm3": target.tolist(), "pure_translation": first, "translation_plus_scale": second, "rmse_improvement_fraction": float((first["rmse_1e16_cm3"] - second["rmse_1e16_cm3"]) / max(first["rmse_1e16_cm3"], 1e-300))}


def _bootstrap(paper: Curve, sim: Curve, *, samples: int, seed: int, sigma_x: float, sigma_y: float) -> dict:
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    x_grid = np.linspace(-20.0, 20.0, 801)
    target = evaluate(sim, x_grid)
    values = []
    source = _sorted_unique(paper)
    for _ in range(samples):
        x_noisy = source.x + rng.normal(0.0, sigma_x, source.x.size)
        y_noisy = np.clip(source.rho + rng.normal(0.0, sigma_y, source.rho.size), 0.0, None)
        boot = _sorted_unique(Curve("bootstrap", x_noisy, y_noisy))

        def p(argument):
            return evaluate(boot, np.asarray(argument, dtype=float))

        first = least_squares(lambda q: q[0] * p(x_grid - q[1]) - target, x0=(1.0, -3.0), bounds=([0.0, -12.0], [3.0, 12.0]))
        second = least_squares(lambda q: q[0] * p((x_grid - q[1]) / q[2]) - target, x0=(1.0, -3.0, 1.0), bounds=([0.0, -12.0, 0.4], [3.0, 12.0, 2.5]))
        values.append((first.x[1], second.x[1], second.x[2]))
    array = np.asarray(values)
    labels = ("translation_delta_x_cm", "scale_x_c_cm", "scale_s")
    return {label: {"median": float(np.median(array[:, i])), "ci95": [float(np.quantile(array[:, i], 0.025)), float(np.quantile(array[:, i], 0.975))]} for i, label in enumerate(labels)}


def _classification(paper_features: dict, sim_features: dict, fit: dict, bootstrap: dict) -> dict:
    shifts = []
    for level in ABSOLUTE_LEVELS:
        key = f"absolute_{level:g}_rising_cm"
        p, s = paper_features.get(key), sim_features.get(key)
        if p is not None and s is not None:
            shifts.append(float(s - p))
    peak_shift = float(sim_features["x_peak_cm"] - paper_features["x_peak_cm"])
    spread = float(max(shifts) - min(shifts)) if len(shifts) >= 2 else float("nan")
    shift_mean = float(np.mean(shifts)) if shifts else float("nan")
    fwhm_ratio = None
    if paper_features.get("fwhm_cm") and sim_features.get("fwhm_cm"):
        fwhm_ratio = float(sim_features["fwhm_cm"] / paper_features["fwhm_cm"])
    rise_ratio = None
    if paper_features.get("rise_10_90_cm") and sim_features.get("rise_10_90_cm"):
        rise_ratio = float(sim_features["rise_10_90_cm"] / paper_features["rise_10_90_cm"])
    improvement = float(fit["rmse_improvement_fraction"])
    low_extra = None
    low_key, high_key = "absolute_0.1_rising_cm", "absolute_2_rising_cm"
    if paper_features.get(low_key) is not None and sim_features.get(low_key) is not None and paper_features.get(high_key) is not None and sim_features.get(high_key) is not None:
        low_extra = abs((sim_features[low_key] - paper_features[low_key]) - (sim_features[high_key] - paper_features[high_key]))
    translation = bool(np.isfinite(spread) and spread <= 1.0 and abs(peak_shift - shift_mean) <= 1.0 and fwhm_ratio is not None and 0.9 <= fwhm_ratio <= 1.1 and improvement < 0.20)
    widening = bool(low_extra is not None and low_extra >= 1.0 and rise_ratio is not None and rise_ratio > 1.10 and improvement >= 0.20)
    if translation:
        label = "mainly_translation"
    elif widening and not (np.isfinite(spread) and spread <= 1.0):
        label = "mainly_broadening"
    else:
        label = "translation_plus_broadening"
    ci = bootstrap["translation_delta_x_cm"]["ci95"]
    confidence = "high" if ci[0] < -1.0 and ci[1] < -1.0 and (translation or widening) else "medium" if ci[0] < 0.0 else "low"
    return {"classification": label, "confidence": confidence, "absolute_rising_shifts_cm": shifts, "mean_absolute_rising_shift_cm": shift_mean, "absolute_rising_shift_spread_cm": spread, "peak_shift_cm": peak_shift, "fwhm_ratio_sim_over_paper": fwhm_ratio, "rise_10_90_ratio_sim_over_paper": rise_ratio, "low_vs_high_threshold_extra_shift_cm": low_extra, "scale_fit_rmse_improvement_fraction": improvement}


def _landmark_rows(case: str, source: str, features: dict):
    for name, value in features.items():
        yield {"case": case, "source": source, "feature": name, "value": "" if value is None else value, "unit": "cm" if name.endswith("_cm") else ("1e16 cm^-3" if "rho" in name else "1e16 cm^-3 cm")}


def _plot_case(out_dir: Path, case: str, paper: Curve, sim: Curve, fit: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(fit["x_focus_cm"])
    target = np.asarray(fit["sim_rho_1e16_cm3"])
    trans = np.asarray(fit["pure_translation"]["fitted_rho_1e16_cm3"])
    scaled = np.asarray(fit["translation_plus_scale"]["fitted_rho_1e16_cm3"])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(sim.x, sim.rho, color="#c62828", lw=1.7, label=f"current FT90 {case}")
    axes[0].plot(paper.x, paper.rho, color="black", lw=1.4, ls="--", label=f"paper PyCAP {case} (digitized)")
    axes[0].axvline(0.0, color="0.45", lw=1.0, ls=":", label="geometric focus")
    axes[0].set(xlim=(-20, 20), ylim=(0, max(7.0, 1.08 * max(float(sim.rho.max()), float(paper.rho.max())))), ylabel=r"peak electron density ($10^{16}$ cm$^{-3}$)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(f"{case}: fixed geometric-focus coordinate; no pre-alignment")
    axes[1].plot(x, target - trans, color="#1f4e79", lw=1.2, label="pure translation residual")
    axes[1].plot(x, target - scaled, color="#c62828", lw=1.2, label="translation + scale residual")
    axes[1].axhline(0.0, color="0.45", lw=0.9)
    axes[1].set(xlim=(-20, 20), xlabel=r"$x_{\rm focus}=100(z-0.95)$ (cm)", ylabel=r"residual ($10^{16}$ cm$^{-3}$)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / f"density_translation_width_{case.replace(' ', '')}.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare paper PyCAP and current FT90 density-curve translation/broadening")
    parser.add_argument("--paper-120", required=True)
    parser.add_argument("--paper-40", required=True)
    parser.add_argument("--sim-120", required=True)
    parser.add_argument("--sim-40", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "120 fs": (load_csv_curve(Path(args.paper_120), "paper PyCAP 120 fs"), load_sim_curve(Path(args.sim_120), "current FT90 120 fs")),
        "40 fs": (load_csv_curve(Path(args.paper_40), "paper PyCAP 40 fs"), load_sim_curve(Path(args.sim_40), "current FT90 40 fs")),
    }
    all_rows, report_cases = [], {}
    for index, (case, (paper, sim)) in enumerate(cases.items()):
        paper_features, sim_features = curve_features(paper), curve_features(sim)
        fit = _fit_models(paper, sim)
        bootstrap = _bootstrap(paper, sim, samples=args.bootstrap_samples, seed=20260715 + index, sigma_x=0.15, sigma_y=0.05)
        classification = _classification(paper_features, sim_features, fit, bootstrap)
        report_cases[case] = {"paper_features": paper_features, "simulation_features": sim_features, "fit": fit, "bootstrap": bootstrap, "classification": classification}
        all_rows.extend(_landmark_rows(case, "paper_pycap_digitized", paper_features))
        all_rows.extend(_landmark_rows(case, "current_ft90_npz", sim_features))
        _plot_case(out_dir, case, paper, sim, fit)

    with (out_dir / "density_landmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "source", "feature", "value", "unit"))
        writer.writeheader(); writer.writerows(all_rows)
    output = {"coordinate_definition": "x_focus_cm = 100 * (z_m - 0.95); the geometric focus remains zero for all curves and fits.", "density_definition": "rho_plot = rho_e[m^-3] / 1e22 in units of 1e16 cm^-3", "fit_models": {"pure_translation": "rho_sim(x) = A * rho_paper(x - delta_x)", "translation_plus_scale": "rho_sim(x) = A * rho_paper((x - x_c) / s)"}, "digitization_uncertainty": {"x_cm": 0.15, "rho_1e16_cm3": 0.05}, "cases": report_cases}
    (out_dir / "translation_width_fit.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# FT90 density curve: translation versus broadening", "", "All curves use the permanent coordinate `x_focus = 100 * (z - 0.95) cm`; neither peak nor onset was aligned.", "", "## Evidence sources", "", "- Paper: Isaacs et al. (2022), Fig. 5(b), digitized PyCAP traces with the saved pixel calibration and colour-selection metadata.", "- Simulation: downloaded FT90 `rho_max_z` NPZ files, converted as `rho_e / 1e22` to `10^16 cm^-3`.", ""]
    for case, result in report_cases.items():
        c = result["classification"]
        fit = result["fit"]
        lines += [f"## {case}", "", f"- Classification: **{c['classification']}** (confidence: **{c['confidence']}**).", f"- Mean absolute rising-edge shift: {c['mean_absolute_rising_shift_cm']:.3f} cm; spread across available levels: {c['absolute_rising_shift_spread_cm']:.3f} cm.", f"- Peak shift (current FT90 minus paper PyCAP): {c['peak_shift_cm']:.3f} cm.", f"- FWHM ratio (current/paper): {c['fwhm_ratio_sim_over_paper'] if c['fwhm_ratio_sim_over_paper'] is not None else 'unavailable'}; 10-90% rising-width ratio: {c['rise_10_90_ratio_sim_over_paper'] if c['rise_10_90_ratio_sim_over_paper'] is not None else 'unavailable'}.", f"- Translation-only fit: Δx = {fit['pure_translation']['parameters']['delta_x_cm']:.3f} cm, RMSE = {fit['pure_translation']['rmse_1e16_cm3']:.3f}.", f"- Translation+scale fit: x_c = {fit['translation_plus_scale']['parameters']['x_c_cm']:.3f} cm, s = {fit['translation_plus_scale']['parameters']['scale_s']:.3f}, RMSE improvement = {100*fit['rmse_improvement_fraction']:.1f}%.", f"- Bootstrap 95% CI for translation Δx: [{result['bootstrap']['translation_delta_x_cm']['ci95'][0]:.3f}, {result['bootstrap']['translation_delta_x_cm']['ci95'][1]:.3f}] cm.", ""]
        if case == "120 fs":
            lines += ["The paper 120 fs trace has a visibly flat high-density plateau. Its single `argmax` is therefore digitization-sensitive; the fixed absolute rising-edge shifts are the more stable translation evidence.", ""]
    lines += ["For the translation+scale model, `x_c` is a model coordinate parameter. When `s != 1`, it is not directly interchangeable with a feature-by-feature translation; use the reported fixed-threshold shifts for that comparison.", ""]
    lines += ["## Provisional physical implication", "", "This report classifies the existing nonlinear curves only. The vacuum-focus job is the independent optical test: its result decides whether a measured translation can be attributed primarily to FT90 finite-aperture/edge diffraction or instead leaves nonlinear self-focusing and ionization-tail mechanisms as the leading candidates.", "", "## Digitization caveat", "", "The paper curves are raster-digitized, not author-supplied data. The stored bootstrap includes ±0.15 cm horizontal and ±0.05 in `10^16 cm^-3` vertical reading uncertainty; conclusions should therefore be treated as quantitative with this image-resolution caveat."]
    (out_dir / "density_translation_width_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

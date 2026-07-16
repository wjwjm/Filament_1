#!/usr/bin/env python3
"""Compare FT90 density traces with digitized PyCAP curves without re-zeroing x."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ABSOLUTE_LEVELS = (0.1, 0.2, 0.5, 1.0, 2.0)
RELATIVE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass
class Curve:
    name: str
    x: np.ndarray
    rho: np.ndarray


def _unique(curve: Curve) -> Curve:
    order = np.argsort(curve.x)
    x, rho = curve.x[order], np.clip(curve.rho[order], 0.0, None)
    x_unique, inverse = np.unique(x, return_inverse=True)
    if x_unique.size == x.size:
        return Curve(curve.name, x, rho)
    values, counts = np.zeros_like(x_unique), np.zeros_like(x_unique)
    np.add.at(values, inverse, rho); np.add.at(counts, inverse, 1)
    return Curve(curve.name, x_unique, values / counts)


def load_csv_curve(path: Path, name: str) -> Curve:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return _unique(Curve(name, np.asarray(data["x_focus_cm"], float), np.asarray(data["rho_1e16_cm3"], float)))


def load_sim_curve(path: Path, name: str) -> Curve:
    with np.load(path, allow_pickle=False) as data:
        x = 100.0 * (np.asarray(data["z_axis"], float) - 0.95)
        rho = np.asarray(data["rho_max_z"], float) / 1e22
    select = (x >= -20.0) & (x <= 20.0)
    return _unique(Curve(name, x[select], rho[select]))


def evaluate(curve: Curve, x: np.ndarray) -> np.ndarray:
    curve = _unique(curve)
    return np.interp(x, curve.x, curve.rho, left=0.0, right=0.0)


def crossing(curve: Curve, level: float, direction: str) -> float | None:
    curve = _unique(curve)
    sign = curve.rho - level
    if direction == "rising":
        hits = np.where((sign[:-1] < 0.0) & (sign[1:] >= 0.0))[0]
        index = int(hits[0]) if hits.size else None
    elif direction == "falling":
        hits = np.where((sign[:-1] >= 0.0) & (sign[1:] < 0.0))[0]
        index = int(hits[-1]) if hits.size else None
    else:
        raise ValueError(direction)
    if index is None:
        return None
    x0, x1 = curve.x[index:index + 2]; y0, y1 = curve.rho[index:index + 2]
    return float(x0 if y1 == y0 else x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def peak_interval(curve: Curve, fraction: float = 0.99) -> dict[str, float]:
    curve = _unique(curve)
    max_index = int(np.argmax(curve.rho)); peak = float(curve.rho[max_index])
    mask = curve.rho >= fraction * peak
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise RuntimeError("peak interval unexpectedly empty")
    # Retain only the contiguous above-threshold block containing argmax.
    left = max_index
    while left > 0 and mask[left - 1]: left -= 1
    right = max_index
    while right < curve.x.size - 1 and mask[right + 1]: right += 1
    return {"peak_rho_1e16_cm3": peak, "x_argmax_cm": float(curve.x[max_index]), "peak_interval_left_cm": float(curve.x[left]), "peak_interval_right_cm": float(curve.x[right]), "peak_interval_center_cm": float(0.5 * (curve.x[left] + curve.x[right])), "peak_interval_width_cm": float(curve.x[right] - curve.x[left])}


def curve_features(curve: Curve) -> dict:
    curve = _unique(curve)
    output = peak_interval(curve)
    output["area_1e16_cm3_cm"] = float(np.trapezoid(curve.rho, curve.x))
    peak = output["peak_rho_1e16_cm3"]
    for level in ABSOLUTE_LEVELS:
        output[f"absolute_{level:g}_rising_cm"] = crossing(curve, level, "rising") if peak >= level else None
    for fraction in RELATIVE_LEVELS:
        level = fraction * peak
        output[f"relative_{fraction:g}_rising_cm"] = crossing(curve, level, "rising")
        output[f"relative_{fraction:g}_falling_cm"] = crossing(curve, level, "falling")
    x10, x90 = output["relative_0.1_rising_cm"], output["relative_0.9_rising_cm"]
    x50_up, x50_down = output["relative_0.5_rising_cm"], output["relative_0.5_falling_cm"]
    x90_down, x10_down = output["relative_0.9_falling_cm"], output["relative_0.1_falling_cm"]
    centre = output["peak_interval_center_cm"]
    output["rise_10_90_cm"] = None if x10 is None or x90 is None else x90 - x10
    output["fwhm_cm"] = None if x50_up is None or x50_down is None else x50_down - x50_up
    output["peak_to_50pct_falling_cm"] = None if x50_down is None else x50_down - centre
    output["peak_to_10pct_falling_cm"] = None if x10_down is None else x10_down - centre
    output["fall_10_90_width_cm"] = None if x90_down is None or x10_down is None else x10_down - x90_down
    tail = curve.x >= centre
    output["post_focus_tail_area_1e16_cm3_cm"] = float(np.trapezoid(curve.rho[tail], curve.x[tail]))
    return output


def _fit_models(paper: Curve, sim: Curve, x_ref: float) -> dict:
    from scipy.optimize import least_squares
    x = np.linspace(-20.0, 20.0, 1601)
    target = evaluate(sim, x)

    def p(argument): return evaluate(paper, np.asarray(argument, float))
    pure = least_squares(lambda q: q[0] * p(x - q[1]) - target, x0=(1.0, -3.0), bounds=([0.0, -12.0], [3.0, 12.0]))
    scaled = least_squares(lambda q: q[0] * p(x_ref + (x - x_ref - q[1]) / q[2]) - target, x0=(1.0, -3.0, 1.0), bounds=([0.0, -12.0, 0.4], [3.0, 12.0, 2.5]))

    def pack(result, labels):
        residual = result.fun; rss = float(np.sum(residual ** 2)); n = residual.size; k = len(labels)
        return {"parameters": {key: float(value) for key, value in zip(labels, result.x)}, "rmse_1e16_cm3": float(np.sqrt(rss / n)), "rss": rss, "aic": float(n * np.log(max(rss / n, 1e-300)) + 2 * k), "bic": float(n * np.log(max(rss / n, 1e-300)) + k * np.log(n)), "fitted_rho_1e16_cm3": (target + residual).tolist(), "residual_1e16_cm3": residual.tolist()}
    a, b = pack(pure, ("amplitude_A", "delta_x_cm")), pack(scaled, ("amplitude_A", "delta_x_cm", "scale_s"))
    return {"x_focus_cm": x.tolist(), "sim_rho_1e16_cm3": target.tolist(), "x_ref_cm": float(x_ref), "pure_translation": a, "translation_plus_scale": b, "rmse_improvement_fraction": float((a["rmse_1e16_cm3"] - b["rmse_1e16_cm3"]) / max(a["rmse_1e16_cm3"], 1e-300))}


def _bootstrap(paper: Curve, sim: Curve, x_ref: float, samples: int, seed: int) -> dict:
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed); grid = np.linspace(-20.0, 20.0, 801); target = evaluate(sim, grid); source = _unique(paper)
    values = []
    for _ in range(samples):
        boot = _unique(Curve("bootstrap", source.x + rng.normal(0.0, 0.15, source.x.size), np.clip(source.rho + rng.normal(0.0, 0.05, source.rho.size), 0.0, None)))
        def p(argument): return evaluate(boot, np.asarray(argument, float))
        pure = least_squares(lambda q: q[0] * p(grid - q[1]) - target, (1.0, -3.0), bounds=([0.0, -12.0], [3.0, 12.0]))
        scale = least_squares(lambda q: q[0] * p(x_ref + (grid - x_ref - q[1]) / q[2]) - target, (1.0, -3.0, 1.0), bounds=([0.0, -12.0, 0.4], [3.0, 12.0, 2.5]))
        values.append((pure.x[0], pure.x[1], scale.x[0], scale.x[1], scale.x[2]))
    values = np.asarray(values); labels = ("pure_A", "pure_delta_x_cm", "scaled_A", "scaled_delta_x_cm", "scaled_s")
    return {key: {"median": float(np.median(values[:, i])), "ci95": [float(np.quantile(values[:, i], .025)), float(np.quantile(values[:, i], .975))]} for i, key in enumerate(labels)}


def _classification(paper: dict, sim: dict, fit: dict, bootstrap: dict) -> dict:
    shifts = [float(sim[f"absolute_{v:g}_rising_cm"] - paper[f"absolute_{v:g}_rising_cm"]) for v in ABSOLUTE_LEVELS if sim.get(f"absolute_{v:g}_rising_cm") is not None and paper.get(f"absolute_{v:g}_rising_cm") is not None]
    mean = float(np.mean(shifts)); spread = float(np.ptp(shifts))
    peak_shift = float(sim["peak_interval_center_cm"] - paper["peak_interval_center_cm"])
    fwhm_ratio = float(sim["fwhm_cm"] / paper["fwhm_cm"])
    post50_ratio = float(sim["peak_to_50pct_falling_cm"] / paper["peak_to_50pct_falling_cm"])
    scale_gain = float(fit["rmse_improvement_fraction"])
    rigid = spread <= 1.0 and abs(peak_shift - mean) <= 1.0
    post_broad = fwhm_ratio > 1.10 and post50_ratio > 1.10 and scale_gain >= .20
    label = "mainly_translation" if rigid and not post_broad else ("mainly_broadening" if post_broad and not rigid else "translation_plus_broadening")
    ci = bootstrap["pure_delta_x_cm"]["ci95"]
    return {"classification": label, "confidence": "high" if ci[1] < -1 else "medium" if ci[1] < 0 else "low", "absolute_rising_shifts_cm": shifts, "mean_absolute_rising_shift_cm": mean, "absolute_rising_shift_spread_cm": spread, "peak_interval_center_shift_cm": peak_shift, "fwhm_ratio_sim_over_paper": fwhm_ratio, "post_peak_50pct_ratio_sim_over_paper": post50_ratio, "scale_fit_rmse_improvement_fraction": scale_gain, "rising_edge_approximately_rigid": rigid, "post_peak_broadening": post_broad}


def _write_landmarks(path: Path, cases: dict[str, tuple[dict, dict]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "source", "feature", "value")); writer.writeheader()
        for case, pair in cases.items():
            for source, item in zip(("paper_pycap_digitized", "current_ft90_npz"), pair):
                writer.writerows({"case": case, "source": source, "feature": key, "value": "" if value is None else value} for key, value in item.items())


def _plot(out: Path, label: str, paper: Curve, sim: Curve, fit: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.asarray(fit["x_focus_cm"]); y = np.asarray(fit["sim_rho_1e16_cm3"])
    pure = np.asarray(fit["pure_translation"]["fitted_rho_1e16_cm3"]); scaled = np.asarray(fit["translation_plus_scale"]["fitted_rho_1e16_cm3"])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7), sharex=True, constrained_layout=True)
    axes[0].plot(sim.x, sim.rho, color="#c62828", lw=1.8, label=f"current FT90 {label}")
    axes[0].plot(paper.x, paper.rho, color="black", lw=1.5, ls="--", label=f"paper PyCAP {label}")
    axes[0].axvline(0, color="0.5", ls=":", lw=1.0, label="geometric focus")
    axes[0].set(xlim=(-20,20), ylim=(0, max(7, 1.08*max(paper.rho.max(),sim.rho.max()))), ylabel=r"peak electron density ($10^{16}$ cm$^{-3}$)", title=f"{label}: fixed geometric-focus coordinate; no pre-alignment")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, y-pure, color="#1f4e79", label="pure translation residual")
    axes[1].plot(x, y-scaled, color="#c62828", label="translation + scale residual")
    axes[1].axhline(0, color="0.5", lw=.8); axes[1].set(xlim=(-20,20), xlabel=r"$x_{\rm focus}=100(z-0.95)$ (cm)", ylabel=r"residual ($10^{16}$ cm$^{-3}$)")
    axes[1].legend(frameon=False, fontsize=8); fig.savefig(out / f"density_translation_width_{label.replace(' ','')}.png", dpi=220); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--paper-120", required=True); parser.add_argument("--paper-40", required=True); parser.add_argument("--sim-120", required=True); parser.add_argument("--sim-40", required=True); parser.add_argument("--out-dir", required=True); parser.add_argument("--bootstrap-samples", type=int, default=200); args = parser.parse_args()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    sources = {"120 fs": (load_csv_curve(Path(args.paper_120), "paper 120 fs"), load_sim_curve(Path(args.sim_120), "FT90 120 fs")), "40 fs": (load_csv_curve(Path(args.paper_40), "paper 40 fs"), load_sim_curve(Path(args.sim_40), "FT90 40 fs"))}
    results, landmarks = {}, {}
    for n, (label, (paper_curve, sim_curve)) in enumerate(sources.items()):
        paper, sim = curve_features(paper_curve), curve_features(sim_curve); x_ref = paper["peak_interval_center_cm"]
        fit = _fit_models(paper_curve, sim_curve, x_ref); boot = _bootstrap(paper_curve, sim_curve, x_ref, args.bootstrap_samples, 20260716+n)
        results[label] = {"paper_features": paper, "simulation_features": sim, "fit": fit, "bootstrap": boot, "classification": _classification(paper, sim, fit, boot)}; landmarks[label] = (paper, sim); _plot(out, label, paper_curve, sim_curve, fit)
    _write_landmarks(out / "density_landmarks.csv", landmarks)
    summary = {"coordinate_definition": "x_focus_cm = 100*(z_m-0.95); all comparisons retain the geometric-focus zero.", "density_definition": "rho_e[m^-3]/1e22, in 1e16 cm^-3", "fit_models": {"pure": "A*rho_paper(x-delta_x)", "scaled": "A*rho_paper[x_ref+(x-x_ref-delta_x)/s]"}, "cases": results}
    (out / "translation_width_fit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=["# FT90 density translation and post-peak-width analysis", "", "All x coordinates use `x_focus = 100*(z-0.95) cm`; no peak alignment or alternate zero was used.", ""]
    for label, item in results.items():
        c=item["classification"]; p=item["paper_features"]; s=item["simulation_features"]; f=item["fit"]
        lines += [f"## {label}", "", f"- Classification: **{c['classification']}** ({c['confidence']} confidence).", f"- Mean fixed-threshold rising shift: {c['mean_absolute_rising_shift_cm']:.3f} cm; threshold spread: {c['absolute_rising_shift_spread_cm']:.3f} cm.", f"- Paper peak interval: [{p['peak_interval_left_cm']:.3f}, {p['peak_interval_right_cm']:.3f}] cm; centre {p['peak_interval_center_cm']:.3f} cm.", f"- Current FT90 peak interval: [{s['peak_interval_left_cm']:.3f}, {s['peak_interval_right_cm']:.3f}] cm; centre {s['peak_interval_center_cm']:.3f} cm.", f"- FWHM ratio current/paper: {c['fwhm_ratio_sim_over_paper']:.3f}; post-peak 50% distance ratio: {c['post_peak_50pct_ratio_sim_over_paper']:.3f}.", f"- Pivoted scale fit: A={f['translation_plus_scale']['parameters']['amplitude_A']:.3f}, Δx={f['translation_plus_scale']['parameters']['delta_x_cm']:.3f} cm, s={f['translation_plus_scale']['parameters']['scale_s']:.3f}, RMSE improvement={100*f['rmse_improvement_fraction']:.1f}%.", ""]
    lines += ["## Separation of effects", "", "The fixed-threshold result assesses the rising edge; FWHM, falling landmarks, and post-focus tail area assess width after the peak. A broader post-peak tail cannot itself cause an earlier rising crossing; it is therefore reported separately from the onset translation."]
    (out / "density_translation_width_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())

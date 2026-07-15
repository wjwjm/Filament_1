#!/usr/bin/env python3
"""Create report and fixed-geometric-focus figures for a vacuum focus scan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(args.out_dir).resolve()
    summary = json.loads((out_dir / "vacuum_focus_summary.json").read_text(encoding="utf-8"))
    with (out_dir / "vacuum_focus_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("vacuum_focus_metrics.csv has no rows")
    x_focus = np.asarray([float(row["x_focus_cm"]) for row in rows])
    i_onaxis = np.asarray([float(row["I_onaxis_W_m2"]) for row in rows])
    i_max = np.asarray([float(row["I_max_W_m2"]) for row in rows])
    w_mom = np.asarray([float(row["w_second_moment_m"]) for row in rows])
    plane = np.load(out_dir / "vacuum_focus_plane.npz")
    x_mm = np.asarray(plane["x_m"]) * 1e3
    y_mm = np.asarray(plane["y_m"]) * 1e3
    intensity = np.asarray(plane["intensity_W_m2"])
    peak = summary["focus_peak"]
    peak_x = float(peak["x_focus_cm"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    axes[0, 0].plot(x_focus, i_onaxis / i_onaxis.max(), color="black", lw=1.7, label=r"$I(0,0,z)/\max I(0,0,z)$")
    axes[0, 0].axvline(0.0, color="0.4", ls="--", lw=1.1, label="geometric focus")
    axes[0, 0].axvline(peak_x, color="#c62828", ls="-", lw=1.3, label=f"vacuum peak = {peak_x:.3f} cm")
    axes[0, 0].set(xlim=(-20, 20), ylim=(0, 1.05), xlabel=r"$x_{\rm focus}=100(z-0.95)$ (cm)", ylabel="normalized on-axis intensity")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(x_focus, w_mom * 1e3, color="#1f4e79", lw=1.7)
    axes[0, 1].axvline(0.0, color="0.4", ls="--", lw=1.1)
    axes[0, 1].axvline(peak_x, color="#c62828", lw=1.3)
    axes[0, 1].set(xlim=(-20, 20), xlabel=r"$x_{\rm focus}=100(z-0.95)$ (cm)", ylabel="second-moment radius (mm)")

    image = axes[1, 0].imshow(intensity / 1e16, extent=(x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()), origin="lower", cmap="magma", aspect="equal")
    axes[1, 0].plot(0, 0, marker="+", color="cyan", ms=8, mew=1.3)
    axes[1, 0].set(xlabel="x (mm)", ylabel="y (mm)", title=f"nearest sampled focus plane: z={float(plane['z_discrete_m']):.6f} m")
    fig.colorbar(image, ax=axes[1, 0], label=r"intensity ($10^{16}$ W m$^{-2}$)")

    axes[1, 1].plot(x_focus, i_max / 1e16, color="#c62828", lw=1.7, label=r"$I_{\max}$")
    axes[1, 1].plot(x_focus, i_onaxis / 1e16, color="black", lw=1.0, alpha=0.75, label=r"$I(0,0)$")
    axes[1, 1].axvline(0.0, color="0.4", ls="--", lw=1.1, label="geometric focus")
    axes[1, 1].axvline(peak_x, color="#c62828", lw=1.3, label="parabolic Imax peak")
    axes[1, 1].set(xlim=(-20, 20), xlabel=r"$x_{\rm focus}=100(z-0.95)$ (cm)", ylabel=r"intensity ($10^{16}$ W m$^{-2}$)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "vacuum_focus_diagnosis.png", dpi=220)
    plt.close(fig)

    x_value = peak_x
    if x_value <= -2.0:
        judgment = "FT90 finite aperture / edge diffraction is a strong candidate for a material forward focus shift."
    elif -2.0 < x_value < -0.5:
        judgment = "The vacuum shift can explain only part of a 3-5 cm early filament onset."
    elif x_value <= 0.5:
        judgment = "The vacuum focus shift cannot explain a 3-5 cm early filament onset."
    else:
        judgment = "The vacuum shift is downstream, opposite to the observed early-onset direction."
    report = f"""# FT90 vacuum-focus validation

Coordinate convention is fixed for every axial quantity:

`x_focus = 100 * (z - 0.95) cm`.

No intensity maximum, density maximum, or post-hoc translation defines zero.

## Result

- Parabolically interpolated `I_max` focus: `{peak['z_parabolic_m']:.9f} m`
- Relative to the 0.95 m geometric focus: `{x_value:.4f} cm`
- Axial sampling half-step uncertainty: `{peak['sampling_half_step_uncertainty_cm']:.4f} cm`
- Input sampled peak power: `{summary['input']['discrete_peak_power_W']:.9g} W`
- Maximum relative transverse-power drift: `{summary['power_conservation']['maximum_relative_drift']:.3e}`

## Interpretation

{judgment}

The propagated field is direct-from-lens angular-spectrum propagation, so the result has no axial stepping accumulation. All nonlinear, plasma, gas-dispersion, Raman, ionization, collision, recombination, absorption, and self-steepening terms are explicitly absent from this driver.
"""
    (out_dir / "vacuum_focus_validation_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Digitize the two PyCAP curves in Isaacs et al. (2022), Fig. 5(b).

The calibration deliberately retains the published `Distance from vacuum
focus` zero.  It does not align either curve to a maximum or onset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# Pixel calibration of Fig. 5(b) rendered from page 11 at 220 dpi.  The two
# x-axis points are the panel-box vertical axes.  The y calibration uses the
# centres of the labelled 0 and 6 tick levels (the outer box extends below 0).
CALIBRATION = {
    "source": "Isaacs et al., Optics Express 30, 22316 (2022), Fig. 5(b), PyCAP curves",
    "render_page": 11,
    "render_dpi": 220,
    "panel": "Fig. 5(b)",
    "x_pixels": [[679, -20.0], [1252, 20.0]],
    "y_pixels": [[727, 0.0], [579, 6.0]],
    "panel_bounds_px": {"left": 679, "right": 1252, "top": 578, "bottom": 735},
    "digitization_uncertainty": {"x_focus_cm": 0.15, "rho_1e16_cm3": 0.05},
}


def _map_linear(pixel, anchors):
    (p0, value0), (p1, value1) = anchors
    return value0 + (pixel - p0) * (value1 - value0) / (p1 - p0)


def _extract(image, *, colour: str):
    import numpy as np

    left = CALIBRATION["panel_bounds_px"]["left"]
    right = CALIBRATION["panel_bounds_px"]["right"]
    top = CALIBRATION["panel_bounds_px"]["top"]
    bottom = CALIBRATION["panel_bounds_px"]["bottom"]
    rgb = np.asarray(image, dtype=np.uint8)
    if colour == "red":
        mask = (rgb[:, :, 0] >= 170) & (rgb[:, :, 1] <= 145) & (rgb[:, :, 2] <= 145)
        x_start, x_stop = left + 5, right - 5
        colour_rule = "R >= 170, G <= 145, B <= 145"
    elif colour == "black":
        mask = rgb.max(axis=2) <= 120
        # Limit this extraction to the plotted black trace, excluding its
        # legend at the upper right and the panel '(b)' annotation.
        x_start, x_stop = left + 55, left + 365
        colour_rule = "max(R,G,B) <= 120; upper-right legend and panel label excluded"
    else:
        raise ValueError(colour)

    points = []
    for x_pixel in range(x_start, x_stop):
        y_pixels = np.where(mask[top + 2:bottom - 1, x_pixel])[0] + top + 2
        if colour == "black":
            y_pixels = y_pixels[~((x_pixel >= left + 350) & (y_pixels <= top + 45))]
        if y_pixels.size == 0:
            continue
        # The line thickness/anti-aliasing normally leaves a compact cluster.
        # A median is stable against isolated compression or raster artefacts.
        y_pixel = float(np.median(y_pixels))
        x_value = float(_map_linear(x_pixel, CALIBRATION["x_pixels"]))
        rho_value = float(_map_linear(y_pixel, CALIBRATION["y_pixels"]))
        if rho_value >= 0.08:  # zero-level trace overlaps the rasterized x-axis
            points.append((x_value, rho_value))
    if not points:
        raise RuntimeError(f"no {colour} PyCAP pixels found with the declared calibration")
    points = np.asarray(points, dtype=float)
    return points[:, 0], points[:, 1], colour_rule


def _write_csv(path: Path, x, rho) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_focus_cm", "rho_1e16_cm3"])
        writer.writerows(zip(x, rho))


def main() -> int:
    parser = argparse.ArgumentParser(description="Digitize Fig. 5(b) PyCAP density curves")
    parser.add_argument("--image", required=True, help="rendered page-11 PNG at 220 dpi")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    image_path = Path(args.image).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    x_120, rho_120, rule_120 = _extract(image, colour="black")
    x_40, rho_40, rule_40 = _extract(image, colour="red")
    _write_csv(out_dir / "paper_pycap_120fs.csv", x_120, rho_120)
    _write_csv(out_dir / "paper_pycap_40fs.csv", x_40, rho_40)

    metadata = {
        **CALIBRATION,
        "image_path": str(image_path),
        "curve_rules": {"120fs_black": rule_120, "40fs_red": rule_40},
        "subthreshold_handling": "points below 0.08 in 1e16 cm^-3 are treated as zero-level/axis-overlap and omitted; later interpolation uses zero outside the retained trace.",
        "output_units": {"x": "cm relative to the published vacuum focus", "rho": "1e16 cm^-3"},
        "point_counts": {"120fs": int(x_120.size), "40fs": int(x_40.size)},
    }
    (out_dir / "paper_digitization_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    axes[0].imshow(np.asarray(image))
    bounds = CALIBRATION["panel_bounds_px"]
    axes[0].set(xlim=(bounds["left"] - 30, bounds["right"] + 30), ylim=(bounds["bottom"] + 35, bounds["top"] - 25), title="source panel and fixed calibration")
    axes[0].axis("off")
    axes[1].plot(x_120, rho_120, color="black", lw=1.5, label="PyCAP 120 fs (digitized)")
    axes[1].plot(x_40, rho_40, color="red", lw=1.5, label="PyCAP 40 fs (digitized)")
    axes[1].set(xlim=(-20, 20), ylim=(0, 7), xlabel=r"$x_{\rm focus}=100(z-0.95)$ (cm)", ylabel=r"peak electron density ($10^{16}$ cm$^{-3}$)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "paper_digitization_overlay.png", dpi=220)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

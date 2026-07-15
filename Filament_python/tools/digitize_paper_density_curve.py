#!/usr/bin/env python3
"""Trace the PyCAP curves in Isaacs et al. (2022), Fig. 5(b).

The script works on a rendered page image, keeps the paper's geometric-focus
coordinate untouched, and records the raster calibration and tracing choices.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


# Coordinates measured from the full 220-dpi rendering.  The box and tick
# locations were cross-checked against the vector PDF axes.  In particular,
# top=532 is the actual panel-box top; the y=6 tick is at y=557 and is not a
# clipping boundary.
PANEL = {"left": 679, "right": 1252, "top": 532, "bottom": 735}
X_TICKS = [(679.0, -20.0), (822.0, -10.0), (966.0, 0.0), (1109.0, 10.0), (1252.0, 20.0)]
Y_TICKS = [(726.0, 0.0), (670.0, 2.0), (614.0, 4.0), (557.0, 6.0)]
EXCLUSIONS = {
    "panel_label": {"x0": 683, "x1": 724, "y0": 536, "y1": 579},
    "legend": {"x0": 1044, "x1": 1247, "y0": 540, "y1": 588},
}


def fit_pixel_calibration(ticks: list[tuple[float, float]]) -> dict[str, Any]:
    pixels = np.asarray([item[0] for item in ticks], dtype=float)
    values = np.asarray([item[1] for item in ticks], dtype=float)
    slope, intercept = np.polyfit(pixels, values, deg=1)
    residuals = values - (slope * pixels + intercept)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "ticks": [{"pixel": float(pixel), "value": float(value), "residual": float(residual)} for pixel, value, residual in zip(pixels, values, residuals)],
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def _inside(rect: dict[str, int], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x >= rect["x0"]) & (x <= rect["x1"]) & (y >= rect["y0"]) & (y <= rect["y1"])


def _candidate_mask(rgb: np.ndarray, colour: str) -> np.ndarray:
    if colour == "red":
        return (rgb[:, :, 0] >= 150) & ((rgb[:, :, 0].astype(int) - rgb[:, :, 1].astype(int)) >= 65) & ((rgb[:, :, 0].astype(int) - rgb[:, :, 2].astype(int)) >= 65)
    if colour == "black":
        chroma = rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int)
        return (rgb.max(axis=2) <= 220) & (chroma <= 24)
    raise ValueError(colour)


def _component_path(mask: np.ndarray, *, max_jump_px: float = 12.0, max_gap_px: int = 4) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select curve-like connected components and make a column-continuous path.

    Small raster breaks (<=4 columns, <=0.28 cm) are linearly bridged.  Longer
    gaps are retained as missing evidence and returned in diagnostics.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    keep: list[int] = []
    for label in range(1, count + 1):
        y, x = np.where(labels == label)
        if x.size == 0:
            continue
        span_x, span_y = int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)
        # Curves are the only retained components with nontrivial horizontal
        # extent after axes, panel label, and legend are excluded.
        if span_x >= 5 and span_y >= 2:
            keep.append(label)
    if not keep:
        raise RuntimeError("no curve-like connected components remain after exclusions")
    selected = np.isin(labels, keep)
    x_values, y_values = [], []
    for x in range(selected.shape[1]):
        y = np.where(selected[:, x])[0]
        if y.size:
            x_values.append(x)
            y_values.append(float(np.median(y)))
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.size < 10:
        raise RuntimeError("too few traced columns")

    # Split at implausible vertical jumps.  This prevents a nearby text glyph
    # from being silently joined to a physical trace.
    good = np.r_[True, np.abs(np.diff(y)) <= max_jump_px]
    x, y = x[good], y[good]
    inserted_x: list[float] = []
    inserted_y: list[float] = []
    long_gaps: list[dict[str, float]] = []
    for i in range(x.size - 1):
        inserted_x.append(float(x[i])); inserted_y.append(float(y[i]))
        gap = int(round(x[i + 1] - x[i] - 1))
        if 0 < gap <= max_gap_px:
            for step in range(1, gap + 1):
                fraction = step / (gap + 1)
                inserted_x.append(float(x[i] + step))
                inserted_y.append(float(y[i] + fraction * (y[i + 1] - y[i])))
        elif gap > max_gap_px:
            long_gaps.append({"x_start_px": float(x[i]), "x_end_px": float(x[i + 1]), "gap_columns": gap})
    inserted_x.append(float(x[-1])); inserted_y.append(float(y[-1]))
    return np.asarray(inserted_x), np.asarray(inserted_y), {"components_total": int(count), "components_retained": [int(value) for value in keep], "long_gaps": long_gaps, "max_jump_px": max_jump_px, "max_interpolated_gap_columns": max_gap_px}


def trace_curve(image: np.ndarray, colour: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    top, bottom = PANEL["top"], PANEL["bottom"]
    left, right = PANEL["left"], PANEL["right"]
    # Remove the four box edges before candidate/component detection.
    # The horizontal coordinate axis is centred at y=726 px.  Stop three
    # pixels above it so dark/red anti-aliasing from the axis cannot become a
    # false low-density continuation of either curve.
    crop = image[top + 3:bottom - 11, left + 3:right - 3]
    candidate = _candidate_mask(crop, colour)
    yy, xx = np.indices(candidate.shape)
    for rect in EXCLUSIONS.values():
        local = {key: rect[key] - (left + 3 if key.startswith("x") else top + 3) for key in rect}
        candidate[_inside(local, xx, yy)] = False
    x_local, y_local, trace = _component_path(candidate)
    x_pixel = x_local + left + 3
    y_pixel = y_local + top + 3
    return x_pixel, y_pixel, trace


def _write_csv(path: Path, x: np.ndarray, rho: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_focus_cm", "rho_1e16_cm3"])
        writer.writerows(zip(x, rho))


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace the two PyCAP density curves in Fig. 5(b)")
    parser.add_argument("--image", required=True, help="page-11 rendering at 220 dpi")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    image_path = Path(args.image).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    x_cal, y_cal = fit_pixel_calibration(X_TICKS), fit_pixel_calibration(Y_TICKS)
    if y_cal["max_abs_residual"] > 0.03:
        raise RuntimeError(f"vertical calibration residual exceeds gate: {y_cal['max_abs_residual']:.4f}")

    traces: dict[str, dict[str, Any]] = {}
    for label, colour in (("120fs", "black"), ("40fs", "red")):
        xp, yp, trace = trace_curve(image, colour)
        x = x_cal["slope"] * xp + x_cal["intercept"]
        rho = np.clip(y_cal["slope"] * yp + y_cal["intercept"], 0.0, None)
        order = np.argsort(x)
        x, rho, xp, yp = x[order], rho[order], xp[order], yp[order]
        trace["max_gap_cm"] = max((gap["gap_columns"] * abs(x_cal["slope"]) for gap in trace["long_gaps"]), default=0.0)
        if trace["max_gap_cm"] > 0.3:
            raise RuntimeError(f"{label} contains an unsupported tracing gap of {trace['max_gap_cm']:.3f} cm")
        if label == "120fs" and float(np.max(rho)) <= 6.0:
            raise RuntimeError("120 fs trace does not retain the published density above 6e16 cm^-3")
        traces[label] = {"x": x, "rho": rho, "x_pixel": xp, "y_pixel": yp, "trace": trace}
        _write_csv(out_dir / f"paper_pycap_{label}.csv", x, rho)

    metadata = {
        "source": "Isaacs et al., Optics Express 30, 22316 (2022), Fig. 5(b), rendered page 11",
        "render_dpi": 220,
        "coordinate_definition": "x_focus_cm = 100 * (z - 0.95); zero remains the published vacuum/geometric focus.",
        "panel_bounds_px": PANEL,
        "x_calibration": x_cal,
        "y_calibration": y_cal,
        "excluded_regions_px": EXCLUSIONS,
        "curve_rules": {"120fs": "near-neutral dark candidate pixels, connected-component selection, bounded-jump path", "40fs": "red-dominant candidate pixels, connected-component selection, bounded-jump path"},
        "zero_overlap_rule": "Pixels on the x-axis/border are excluded before tracing. The CSV starts at the lowest unambiguous above-axis curve pixel; downstream interpolation is zero outside the retained trace.",
        "trace_diagnostics": {key: value["trace"] for key, value in traces.items()},
        "quality_gates": {"vertical_calibration_max_abs_residual": y_cal["max_abs_residual"], "120fs_peak_above_6": float(np.max(traces["120fs"]["rho"])) > 6.0},
    }
    (out_dir / "paper_digitization_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Direct pixel overlay: no re-plotted surrogate is substituted for the
    # visual check requested by the task.
    fig, ax = plt.subplots(figsize=(9.8, 5.5), constrained_layout=True)
    ax.imshow(image)
    ax.scatter(traces["120fs"]["x_pixel"], traces["120fs"]["y_pixel"], s=8, color="cyan", alpha=0.72, label="accepted 120 fs pixels")
    ax.scatter(traces["40fs"]["x_pixel"], traces["40fs"]["y_pixel"], s=8, color="yellow", alpha=0.72, label="accepted 40 fs pixels")
    ax.add_patch(Rectangle((PANEL["left"], PANEL["top"]), PANEL["right"] - PANEL["left"], PANEL["bottom"] - PANEL["top"], fill=False, ec="lime", lw=1.0, label="coordinate box"))
    for name, rect in EXCLUSIONS.items():
        ax.add_patch(Rectangle((rect["x0"], rect["y0"]), rect["x1"] - rect["x0"], rect["y1"] - rect["y0"], fill=True, fc="magenta", alpha=0.17, ec="magenta", lw=0.8, label=f"excluded: {name}"))
    ax.scatter([item[0] for item in X_TICKS], [PANEL["bottom"]] * len(X_TICKS), color="lime", marker="x", s=26, label="x calibration")
    ax.scatter([PANEL["left"]] * len(Y_TICKS), [item[0] for item in Y_TICKS], color="lime", marker="x", s=26, label="y calibration")
    ax.set(xlim=(PANEL["left"] - 18, PANEL["right"] + 18), ylim=(PANEL["bottom"] + 18, PANEL["top"] - 18), title="Fig. 5(b): calibrated pixel-level trace overlay")
    ax.legend(loc="upper left", fontsize=7, frameon=True, ncol=2)
    fig.savefig(out_dir / "paper_digitization_pixel_overlay.png", dpi=220)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

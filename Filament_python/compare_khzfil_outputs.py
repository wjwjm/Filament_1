#!/usr/bin/env python3
"""Compare KHz-filament NPZ or MAT outputs without resampling their z axes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


PLOT_SPECS = {
    "I_max_z": ("Intensity (W/m²)", True, 1.0),
    "I_onaxis_max_z": ("Intensity (W/m²)", True, 1.0),
    "rho_onaxis_max_z": ("Electron density (m⁻³)", True, 1.0),
    "rho_max_z": ("Electron density (m⁻³)", True, 1.0),
    "w_mom_z": ("w_mom (mm)", False, 1e3),
    "U_z": ("Pulse energy (J)", False, 1.0),
    "fwhm_plasma_z": ("FWHM (µm)", False, 1e6),
    "fwhm_fluence_z": ("FWHM (µm)", False, 1e6),
}


def _vector(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim > 2 or (value.ndim == 2 and 1 not in value.shape):
        raise ValueError(f"{name} must be one-dimensional, got {value.shape}")
    return np.asarray(value, dtype=float).reshape(-1)


def load_result_file(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            return {name: source[name] for name in source.files}
    if path.suffix.lower() == ".mat":
        return {name: value for name, value in scipy.io.loadmat(path).items() if not name.startswith("__")}
    raise ValueError(f"unsupported result file: {path}")


def validate_dataset(dataset: dict[str, np.ndarray], label: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if "z_axis" not in dataset:
        raise ValueError(f"{label}: missing z_axis")
    z = _vector(dataset["z_axis"], "z_axis")
    if z.size == 0:
        raise ValueError(f"{label}: z_axis is empty")
    fields: dict[str, np.ndarray] = {}
    for field in PLOT_SPECS:
        if field not in dataset:
            continue
        values = _vector(dataset[field], field)
        if values.size != z.size:
            raise ValueError(f"{label}: {field} length does not match z_axis")
        fields[field] = values
    return z, fields


def get_field_plot_spec(field: str) -> tuple[str, bool, float]:
    return PLOT_SPECS[field]


def _positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values) | (values <= 0)] = np.nan
    return values


def _case_metrics(case_id: str, label: str, z: np.ndarray, fields: dict[str, np.ndarray]) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case_id, "case_label": label, "z_points": int(z.size)}
    for field in ("I_max_z", "rho_onaxis_max_z"):
        if field in fields and np.any(np.isfinite(fields[field])):
            index = int(np.nanargmax(fields[field]))
            row[f"{field}_peak"] = float(fields[field][index])
            row[f"z_{field}_peak_m"] = float(z[index])
    if "w_mom_z" in fields and np.any(np.isfinite(fields["w_mom_z"])):
        index = int(np.nanargmin(fields["w_mom_z"]))
        row["w_mom_min_m"] = float(fields["w_mom_z"][index])
        row["z_w_mom_min_m"] = float(z[index])
    if "U_z" in fields:
        valid = np.flatnonzero(np.isfinite(fields["U_z"]))
        if valid.size:
            u0, uend = fields["U_z"][valid[0]], fields["U_z"][valid[-1]]
            row["U0_J"] = float(u0)
            row["U_end_J"] = float(uend)
            if u0 != 0:
                row["U_drift_pct"] = float((uend / u0 - 1.0) * 100.0)
    return row


def generate_comparison_figures(
    results: list[tuple[str, str, dict[str, np.ndarray]]],
    output_dir: str | Path,
    fields: list[str],
    dpi: int = 200,
    z_shift_cm: float = 0.0,
    stage_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [(case_id, label, *validate_dataset(data, label)) for case_id, label, data in results]
    generated: list[str] = []
    skipped: dict[str, str] = {}
    overview_fields: list[str] = []

    for field in fields:
        if field not in PLOT_SPECS:
            skipped[field] = "unknown comparison field"
            continue
        present = [(case_id, label, z, data[field]) for case_id, label, z, data in datasets if field in data]
        if len(present) != len(datasets):
            skipped[field] = "field missing from one or more cases"
            continue
        ylabel, log_scale, scale = get_field_plot_spec(field)
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for _, label, z, values in present:
            values = _positive(values) if log_scale else values
            if log_scale:
                ax.semilogy(z * 100.0 + z_shift_cm, values * scale, linewidth=1.6, label=label)
            else:
                ax.plot(z * 100.0 + z_shift_cm, values * scale, linewidth=1.6, label=label)
        ax.set(xlabel="z (cm)" if z_shift_cm == 0 else f"z (cm), shifted {z_shift_cm:+g} cm", ylabel=ylabel, title=f"{field}: 40 fs vs 120 fs")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        path = output_dir / f"compare_{field}.png"
        fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        generated.append(path.name); overview_fields.append(field)

    if overview_fields:
        columns = 2
        rows = int(np.ceil(len(overview_fields) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(12, 4.0 * rows), squeeze=False)
        for ax, field in zip(axes.flat, overview_fields):
            ylabel, log_scale, scale = get_field_plot_spec(field)
            for _, label, z, data in datasets:
                values = _positive(data[field]) if log_scale else data[field]
                (ax.semilogy if log_scale else ax.plot)(z * 100.0 + z_shift_cm, values * scale, linewidth=1.2, label=label)
            ax.set_title(field); ax.set_xlabel("z (cm)"); ax.set_ylabel(ylabel); ax.grid(True, which="both", alpha=0.25)
        for ax in axes.flat[len(overview_fields):]: ax.axis("off")
        axes.flat[0].legend(loc="best")
        path = output_dir / "comparison_overview.png"
        fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        generated.append(path.name)

    metrics = [_case_metrics(case_id, label, z, data) for case_id, label, z, data in datasets]
    csv_path = output_dir / "comparison_metrics.csv"
    keys = sorted({key for row in metrics for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(metrics)
    summary: dict[str, Any] = {
        "cases": [case_id for case_id, _, _, _ in datasets], "generated_figures": generated,
        "skipped_fields": skipped, "comparison_metrics_csv": csv_path.name, "metrics": metrics,
    }
    if stage_metadata: summary.update(stage_metadata)
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare KHz-filament output files")
    parser.add_argument("--inputs", nargs=2, required=True)
    parser.add_argument("--labels", nargs=2, default=["40 fs", "120 fs"])
    parser.add_argument("--case-ids", nargs=2, default=["40fs", "120fs"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fields", default=",".join(PLOT_SPECS))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--z-shift-cm", type=float, default=0.0)
    parser.add_argument("--stage-spec", default=None)
    args = parser.parse_args()
    metadata = None
    if args.stage_spec:
        spec = json.loads(Path(args.stage_spec).read_text(encoding="utf-8"))
        metadata = {"stage_id": spec["stage_id"], "stage_name": spec["stage_name"], "comparison_mode": spec["comparison_mode"], "fixed_peak_power_W": spec["required_invariants"]["beam.P0_peak"], "cases": args.case_ids}
    results = [(case_id, label, load_result_file(path)) for case_id, label, path in zip(args.case_ids, args.labels, args.inputs)]
    generate_comparison_figures(results, args.out_dir, [x for x in args.fields.split(",") if x], args.dpi, args.z_shift_cm, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

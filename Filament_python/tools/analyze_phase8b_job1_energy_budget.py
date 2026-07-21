#!/usr/bin/env python3
"""Reconstruct the signed Phase 8B Job 1 energy budget from a saved archive.

This is a post-processing tool only.  It never imports or changes the
propagation implementation and performs every reduction in float64.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-30


def _array(data: Mapping, key: str) -> np.ndarray:
    return np.asarray(data[key], dtype=np.float64)


def _require_same_length(n: int, **arrays: np.ndarray) -> None:
    bad = {name: values.shape for name, values in arrays.items() if values.ndim != 1 or values.size != n}
    if bad:
        raise ValueError(f"energy-budget history shape mismatch: {bad}")


def _step_relative(signed_step: np.ndarray, field_loss_step: np.ndarray, accounted_step: np.ndarray, u0: float) -> np.ndarray:
    denominator = np.maximum.reduce((np.abs(field_loss_step), np.abs(accounted_step), np.full_like(signed_step, abs(u0) * 1e-15)))
    return signed_step / denominator


def build_energy_budget(data: Mapping) -> dict:
    """Return aligned signed energy-accounting histories and summary metadata.

    All histories are step-end values.  ``field_loss_step_J`` is positive when
    the optical field loses energy; a positive signed residual therefore means
    field loss exceeds the recorded material-energy channels.
    """
    z = _array(data, "z_axis")
    u = _array(data, "U_z")
    u_step = _array(data, "U_step_change_z")
    ion_ib_step = _array(data, "E_dep_z")
    raman_actual_step = _array(data, "raman_actual_loss_step_J")
    raman_target_step = _array(data, "raman_target_loss_step_J")
    raman_target_cumulative = _array(data, "raman_target_loss_cumulative_J")
    raman_actual_cumulative = _array(data, "raman_actual_loss_cumulative_J")
    deposited_total_step = _array(data, "E_dep_total_z")
    deposited_total_cumulative = _array(data, "E_dep_cumulative_z")
    legacy_alpha = _array(data, "alpha_R_applied_max_z")
    alpha_ib = _array(data, "alpha_ib_max_z")
    n = z.size
    _require_same_length(
        n, U_z=u, U_step_change_z=u_step, E_dep_z=ion_ib_step,
        raman_actual_loss_step_J=raman_actual_step,
        raman_target_loss_step_J=raman_target_step,
        raman_target_loss_cumulative_J=raman_target_cumulative,
        raman_actual_loss_cumulative_J=raman_actual_cumulative,
        E_dep_total_z=deposited_total_step,
        E_dep_cumulative_z=deposited_total_cumulative,
        alpha_R_applied_max_z=legacy_alpha, alpha_ib_max_z=alpha_ib,
    )
    if n == 0:
        raise ValueError("energy-budget archive is empty")

    u0 = float(u[0] - u_step[0])
    if not np.isfinite(u0) or u0 <= 0.0:
        raise ValueError(f"invalid reconstructed initial field energy {u0}")
    u_start = np.concatenate(([u0], u[:-1]))
    field_loss_step = u_start - u
    field_loss_cumulative = u0 - u

    ib_disabled = bool(np.all(alpha_ib == 0.0))
    legacy_disabled = bool(np.all(legacy_alpha == 0.0))
    ionization_step = ion_ib_step.copy() if ib_disabled else np.full(n, np.nan)
    ib_step = np.zeros(n, dtype=np.float64) if ib_disabled else np.full(n, np.nan)
    legacy_alpha_step = np.zeros(n, dtype=np.float64) if legacy_disabled else np.full(n, np.nan)

    accounted_step_from_channels = ion_ib_step + raman_actual_step
    accounted_cumulative_from_channels = np.cumsum(accounted_step_from_channels, dtype=np.float64)
    stored_total_step_mismatch = deposited_total_step - accounted_step_from_channels
    stored_total_cumulative_mismatch = deposited_total_cumulative - accounted_cumulative_from_channels
    signed_residual_step = field_loss_step - deposited_total_step
    signed_residual_cumulative = field_loss_cumulative - deposited_total_cumulative
    relative_closure_signed = signed_residual_cumulative / u0
    relative_closure_abs = np.abs(relative_closure_signed)
    relative_step_signed = _step_relative(signed_residual_step, field_loss_step, deposited_total_step, u0)

    return {
        "z_m": z,
        "field_energy_J": u,
        "initial_field_energy_J": np.full(n, u0),
        "field_loss_step_J": field_loss_step,
        "field_loss_cumulative_J": field_loss_cumulative,
        "raman_target_step_J": raman_target_step,
        "raman_target_cumulative_J": raman_target_cumulative,
        "raman_actual_step_J": raman_actual_step,
        "raman_actual_cumulative_J": raman_actual_cumulative,
        "ionization_plus_ib_step_J": ion_ib_step,
        "ionization_step_J": ionization_step,
        "plasma_ib_step_J": ib_step,
        "legacy_alpha_energy_step_J": legacy_alpha_step,
        "total_accounted_step_J": deposited_total_step,
        "total_accounted_cumulative_J": deposited_total_cumulative,
        "total_accounted_step_from_channels_J": accounted_step_from_channels,
        "total_accounted_cumulative_from_channels_J": accounted_cumulative_from_channels,
        "stored_total_step_mismatch_J": stored_total_step_mismatch,
        "stored_total_cumulative_mismatch_J": stored_total_cumulative_mismatch,
        "unaccounted_boundary_filter_or_numerical_step_J": signed_residual_step,
        "unaccounted_boundary_filter_or_numerical_cumulative_J": signed_residual_cumulative,
        "relative_closure_signed": relative_closure_signed,
        "relative_closure_abs": relative_closure_abs,
        "relative_step_residual_signed": relative_step_signed,
        "legacy_alpha_disabled": legacy_disabled,
        "ib_disabled": ib_disabled,
        "initial_energy_J": u0,
    }


def _segment_rows(budget: dict) -> list[dict]:
    z = budget["z_m"]
    boundaries = (("pre_focus", -np.inf, 0.85), ("near_focus", 0.85, 1.05), ("post_focus", 1.05, np.inf))
    rows = []
    for name, lo, hi in boundaries:
        mask = (z >= lo) & (z < hi) if np.isfinite(hi) else z >= lo
        indices = np.flatnonzero(mask)
        if not indices.size:
            continue
        fields = (
            "field_loss_step_J", "ionization_plus_ib_step_J", "raman_actual_step_J",
            "raman_target_step_J", "total_accounted_step_J",
            "unaccounted_boundary_filter_or_numerical_step_J",
        )
        row = {"segment": name, "z_start_m": float(z[indices[0]]), "z_end_m": float(z[indices[-1]]), "records": int(indices.size)}
        for field in fields:
            row[field.removesuffix("_step_J") + "_increment_J"] = float(np.nansum(budget[field][indices], dtype=np.float64))
        row["closure_abs_start"] = float(budget["relative_closure_abs"][indices[0]])
        row["closure_abs_end"] = float(budget["relative_closure_abs"][indices[-1]])
        row["closure_abs_max"] = float(np.nanmax(budget["relative_closure_abs"][indices]))
        rows.append(row)
    return rows


def summarize_budget(budget: dict) -> dict:
    z = budget["z_m"]
    closure = budget["relative_closure_abs"]
    first = np.flatnonzero(closure > 0.01)
    near = (z >= 0.85) & (z <= 1.05)
    near_index = int(np.flatnonzero(near)[np.argmax(closure[near])]) if np.any(near) else -1
    step_abs = np.abs(budget["relative_step_residual_signed"])
    max_step = int(np.argmax(step_abs))
    return {
        "schema": "khz_filament.phase8b_r.job1_signed_energy_budget.v1",
        "sign_convention": {
            "field_loss_positive": "U_start - U_end > 0 means optical field energy lost in the accepted step",
            "accounted_energy_positive": "positive deposited energy is transferred from the field to matter",
            "signed_residual": "field_loss - total_accounted; positive means an unaccounted field-energy loss",
            "all_histories": "step-end aligned; step channels correspond to the accepted step ending at z_m",
            "cumulative_denominator": "initial field energy U0 reconstructed as U_z[0] - U_step_change_z[0]",
            "step_denominator": "max(abs(field_loss_step), abs(total_accounted_step), U0*1e-15)",
        },
        "units": {
            "energy_channels": "J",
            "z_m": "m",
            "relative_closure": "1",
        },
        "initial_field_energy_J": float(budget["initial_energy_J"]),
        "ib_channel": "zero because alpha_ib_max_z is exactly zero" if budget["ib_disabled"] else "not separately observable in E_dep_z",
        "legacy_alpha_channel": "zero because alpha_R_applied_max_z is exactly zero" if budget["legacy_alpha_disabled"] else "not separately observable as an energy channel",
        "boundary_filter_numerical_channel": "not independently recorded; reported only as the signed unaccounted residual",
        "final": {
            "field_loss_J": float(budget["field_loss_cumulative_J"][-1]),
            "total_accounted_J": float(budget["total_accounted_cumulative_J"][-1]),
            "raman_target_J": float(budget["raman_target_cumulative_J"][-1]),
            "raman_actual_J": float(budget["raman_actual_cumulative_J"][-1]),
            "signed_residual_J": float(budget["unaccounted_boundary_filter_or_numerical_cumulative_J"][-1]),
            "relative_closure": float(budget["relative_closure_abs"][-1]),
        },
        "near_focus": {
            "max_relative_closure": float(closure[near_index]) if near_index >= 0 else float("nan"),
            "z_at_max_m": float(z[near_index]) if near_index >= 0 else float("nan"),
        },
        "post_focus_from_near_focus_max": {
            "relative_closure_change": float(closure[-1] - closure[near_index]) if near_index >= 0 else float("nan"),
            "percentage_point_change": float(100.0 * (closure[-1] - closure[near_index])) if near_index >= 0 else float("nan"),
            "field_loss_increment_J": float(budget["field_loss_cumulative_J"][-1] - budget["field_loss_cumulative_J"][near_index]) if near_index >= 0 else float("nan"),
            "accounted_increment_J": float(budget["total_accounted_cumulative_J"][-1] - budget["total_accounted_cumulative_J"][near_index]) if near_index >= 0 else float("nan"),
            "raman_actual_increment_J": float(budget["raman_actual_cumulative_J"][-1] - budget["raman_actual_cumulative_J"][near_index]) if near_index >= 0 else float("nan"),
        },
        "first_relative_closure_above_1_percent": None if not first.size else {"z_m": float(z[first[0]]), "relative_closure": float(closure[first[0]])},
        "step_residual_statistics": {
            "median_abs_relative": float(np.median(step_abs)),
            "p95_abs_relative": float(np.percentile(step_abs, 95)),
            "p99_abs_relative": float(np.percentile(step_abs, 99)),
            "max_abs_relative": float(step_abs[max_step]),
            "z_at_max_m": float(z[max_step]),
            "signed_residual_J_at_max": float(budget["unaccounted_boundary_filter_or_numerical_step_J"][max_step]),
        },
        "stored_channel_consistency": {
            "max_total_step_mismatch_J": float(np.max(np.abs(budget["stored_total_step_mismatch_J"]))),
            "max_total_cumulative_mismatch_J": float(np.max(np.abs(budget["stored_total_cumulative_mismatch_J"]))),
        },
        "segments": _segment_rows(budget),
    }


def _write_csv(path: Path, budget: dict) -> None:
    fields = [key for key, value in budget.items() if isinstance(value, np.ndarray) and value.ndim == 1]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(budget["z_m"].size):
            writer.writerow({field: float(budget[field][index]) for field in fields})


def _write_segments(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(out_dir: Path, budget: dict) -> None:
    z = budget["z_m"]
    plt.figure(figsize=(7.2, 4.3))
    plt.plot(z, 100.0 * budget["relative_closure_abs"], label="|total closure|")
    plt.axhline(1.0, color="tab:red", linestyle="--", label="1% contract")
    plt.axvspan(0.85, 1.05, color="0.9", label="near focus")
    plt.xlabel("z (m)")
    plt.ylabel("closure (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "job1_energy_closure_vs_z.png", dpi=180)
    plt.close()

    scale = 1e6
    plt.figure(figsize=(7.2, 4.3))
    plt.plot(z, scale * budget["field_loss_cumulative_J"], label="field loss")
    plt.plot(z, scale * budget["total_accounted_cumulative_J"], label="accounted deposition")
    plt.plot(z, scale * budget["raman_actual_cumulative_J"], label="Raman actual")
    plt.plot(z, scale * budget["unaccounted_boundary_filter_or_numerical_cumulative_J"], label="signed residual")
    plt.xlabel("z (m)")
    plt.ylabel("cumulative energy (uJ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "job1_energy_channels_vs_z.png", dpi=180)
    plt.close()


def analyze(npz_path: Path, out_dir: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        budget = build_energy_budget(data)
    summary = summarize_budget(budget)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "job1_energy_closure_vs_z.csv", budget)
    _write_segments(out_dir / "job1_energy_budget_segments.csv", summary["segments"])
    (out_dir / "job1_energy_budget_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot(out_dir, budget)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "isaacs_raman_closure" / "phase8b_controlled_propagation")
    args = parser.parse_args(argv)
    summary = analyze(args.npz, args.out_dir)
    print(json.dumps(summary["final"], indent=2))


if __name__ == "__main__":
    main()

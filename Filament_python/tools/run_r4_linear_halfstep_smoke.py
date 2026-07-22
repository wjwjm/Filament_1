#!/usr/bin/env python3
"""Run a bounded R4 BK-NEE linear-half-step smoke; never a full Job 1."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.confio import load_all  # noqa: E402
from KHz_filament.device import xp  # noqa: E402
from KHz_filament.runner import run_demo  # noqa: E402


def analyze(data: dict[str, np.ndarray], *, mode: str) -> dict:
    u0 = float(data["linear_halfstep_1_energy_before_J"][0])
    rows = []
    cumulative_residual = 0.0
    p99_values = []
    for half in (1, 2):
        before = np.asarray(data[f"linear_halfstep_{half}_energy_before_J"], dtype=np.float64)
        residual = np.asarray(data[f"linear_halfstep_{half}_unaccounted_residual_J"], dtype=np.float64)
        explicit = sum(
            np.asarray(data[f"linear_halfstep_{half}_{key}"], dtype=np.float64)
            for key in (
                "explicit_boundary_loss_J", "explicit_spectral_filter_loss_J",
                "explicit_crop_loss_J", "explicit_evanescent_loss_J", "explicit_other_loss_J",
            )
        )
        relative = np.abs(residual) / np.maximum(np.abs(before), u0 * 1e-15)
        cumulative_residual += float(np.sum(residual, dtype=np.float64))
        p99_values.extend(relative.tolist())
        rows.append({
            "halfstep": half,
            "field_delta_sum_J": float(np.sum(data[f"linear_halfstep_{half}_field_delta_J"], dtype=np.float64)),
            "explicit_loss_sum_J": float(np.sum(explicit, dtype=np.float64)),
            "unaccounted_residual_sum_J": float(np.sum(residual, dtype=np.float64)),
            "relative_residual_p99": float(np.percentile(relative, 99)),
        })
    cumulative_relative = abs(cumulative_residual) / max(abs(u0), 1e-300)
    result = {
        "schema": "khz_filament.phase8b_r.r4_linear_halfstep_smoke.v1",
        "mode": mode,
        "backend": xp.__name__,
        "steps": int(np.asarray(data["z_axis"]).size),
        "linear_halfsteps": rows,
        "linear_per_halfstep_residual_p99": float(np.percentile(p99_values, 99)),
        "linear_cumulative_unaccounted_residual_J": cumulative_residual,
        "linear_cumulative_unaccounted_relative": cumulative_relative,
        "pure_lossless_contract": {
            "per_halfstep_p99_lt": 1e-6,
            "cumulative_lt": 1e-5,
            "passed": bool(np.percentile(p99_values, 99) < 1e-6 and cumulative_relative < 1e-5),
        },
    }
    if mode == "full_physics":
        result["raman_step_closure_p99"] = float(np.percentile(data["raman_closure_residual_step"], 99))
        result["raman_cumulative_closure"] = float(data["raman_cumulative_closure_residual"][-1])
    return result


def run(*, config: Path, output_npz: Path, report: Path, steps: int, mode: str, dtype: str) -> dict:
    if xp.__name__ != "cupy":
        raise RuntimeError("R4 full-size smoke requires the CuPy backend; NumPy fallback is forbidden")
    grid, beam, prop, ion, heat, run_cfg, raman = load_all(str(config))
    prop = replace(
        prop, z_max=float(steps) * float(prop.dz),
        focus_window_step=False, limit_focus_window=False,
        progress_every_z=0, energy_probe_every=0, diag_extra=False,
        diag_operator_energy=True, diag_linear_halfstep_energy=True,
    )
    if mode == "pure_linear":
        prop = replace(
            prop, use_self_steepening=False, use_electronic_kerr=False,
            use_raman_phase=False, use_raman_full_operator=False,
            use_plasma_phase=False, use_ionization_loss=False,
            use_raman_absorption=False, use_ionization_solver=False,
        )
        raman = replace(raman, enabled=False)
    elif mode != "full_physics":
        raise ValueError("mode must be pure_linear or full_physics")
    run_demo(grid=grid, beam=beam, prop=prop, ion=ion, heat=heat,
             run=replace(run_cfg, Npulses=1), raman=raman,
             out_path=str(output_npz), dtype=dtype)
    with np.load(output_npz, allow_pickle=False) as source:
        result = analyze({key: source[key].copy() for key in source.files}, mode=mode)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on_energy_audit.json")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--mode", choices=("pure_linear", "full_physics"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default="fp32")
    args = parser.parse_args(argv)
    print(json.dumps(run(config=args.config, output_npz=args.output_npz, report=args.report,
                         steps=args.steps, mode=args.mode, dtype=args.dtype), indent=2))


if __name__ == "__main__":
    main()

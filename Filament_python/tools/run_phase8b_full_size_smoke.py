#!/usr/bin/env python3
"""Run and audit a short full-grid Phase 8B smoke case on one GPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("on", "off"), required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--production-steps", type=int, default=15000)
    parser.add_argument("--production-slurm-limit-s", type=float, default=8*3600)
    parser.add_argument("--legacy-full-runtime-s", type=float, default=2*3600+4*60+48)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    smoke = deepcopy(raw)
    dz = float(smoke["propagation"]["dz"])
    smoke["propagation"].update({
        "z_max": args.steps * dz,
        "auto_substep": False,
        "focus_window_step": False,
        "limit_focus_window": False,
        "progress_every_z": 1,
        "energy_probe_every": 1,
        "diag_extra": False,
        "measure_performance": True,
    })
    smoke_config = args.out_dir / f"phase8b_full_size_smoke_{args.case}.json"
    smoke_config.write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / f"phase8b_full_size_smoke_{args.case}_config_audit.json").write_text(
        json.dumps({
            "source_config": str(args.config),
            "smoke_config": str(smoke_config),
            "authorized_smoke_overrides": {
                "propagation.z_max": smoke["propagation"]["z_max"],
                "propagation.auto_substep": False,
                "propagation.focus_window_step": False,
                "propagation.limit_focus_window": False,
                "propagation.progress_every_z": 1,
                "propagation.energy_probe_every": 1,
                "propagation.diag_extra": False,
                "propagation.measure_performance": True,
            },
            "grid_unchanged": smoke["grid"] == raw["grid"],
            "raman_unchanged": smoke["raman"] == raw["raman"],
            "ionization_unchanged": smoke["ionization"] == raw["ionization"],
            "beam_unchanged": smoke["beam"] == raw["beam"],
        }, indent=2) + "\n", encoding="utf-8")

    import cupy as cp
    from KHz_filament.runner import run_from_file

    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    free_before, total_memory = cp.cuda.runtime.memGetInfo()
    cp.get_default_memory_pool().free_all_blocks()
    result_path = args.out_dir / f"phase8b_full_size_smoke_{args.case}.npz"
    started_iso = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    run_from_file(str(smoke_config), out_path=str(result_path), dtype="fp32")
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    ended_iso = datetime.now(timezone.utc).isoformat()

    with np.load(result_path, allow_pickle=False) as data:
        step_wall = np.asarray(data["total_walltime_step_s"], dtype=float)
        linear_wall = np.asarray(data["linear_walltime_step_s"], dtype=float)
        ion_wall = np.asarray(data["ionization_walltime_step_s"], dtype=float)
        raman_wall = np.asarray(data["raman_operator_walltime_step_s"], dtype=float)
        allocated = np.asarray(data["gpu_allocated_step_bytes"], dtype=np.int64)
        reserved = np.asarray(data["gpu_reserved_step_bytes"], dtype=np.int64)
        conv = np.asarray(data["raman_convolution_count_step"], dtype=np.int64)
        substeps = np.asarray(data["raman_operator_substep_count"], dtype=np.int64)
        rhs = np.asarray(data["raman_rhs_l2_norm"], dtype=float)
        ir_raw = np.asarray(data["raman_IR_max_raw"], dtype=float)
        target = np.asarray(data["raman_target_loss_step_J"], dtype=float)
        actual = np.asarray(data["raman_actual_loss_step_J"], dtype=float)
        closure = np.asarray(data["raman_closure_residual_step"], dtype=float)
        cumulative_closure = np.asarray(data["raman_cumulative_closure_residual"], dtype=float)
        legacy_alpha = np.asarray(data["alpha_R_applied_max_z"], dtype=float)
        finite = all(np.all(np.isfinite(data[key])) for key in (
            "U_z", "I_max_z", "rho_max_z", "raman_IR_max_raw",
            "raman_target_loss_step_J", "raman_actual_loss_step_J"))

    mean_step = float(np.mean(step_wall))
    estimated_full = mean_step * args.production_steps
    legacy_mean_step = args.legacy_full_runtime_s / args.production_steps
    scheduler_memory_mb = int(
        os.environ.get("SLURM_MEM_PER_NODE")
        or os.environ.get("PHASE8B_SCHEDULER_MEMORY_MB")
        or 0
    )
    metrics = {
        "schema": "khz_filament.phase8b.full_size_smoke.v1",
        "case": args.case,
        "started_utc": started_iso,
        "ended_utc": ended_iso,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "resources": {
            "partition": os.environ.get("SLURM_JOB_PARTITION", "gpu"),
            "gpu_count": int(os.environ.get("SLURM_GPUS_ON_NODE", "1").split("(")[0] or 1),
            "cpu_threads": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
            "requested_memory": f"{scheduler_memory_mb}M" if scheduler_memory_mb else "scheduler_default_per_gpu",
            "requested_memory_mb": scheduler_memory_mb,
            "memory_request_mode": "scheduler_default_per_gpu",
            "smoke_time_limit_s": 1800,
        },
        "gpu": {
            "type": gpu_name,
            "device_id": int(device.id),
            "total_memory_bytes": int(total_memory),
            "free_memory_before_bytes": int(free_before),
            "peak_allocated_bytes": int(np.max(allocated)),
            "peak_reserved_bytes": int(np.max(reserved)),
            "peak_reserved_fraction": float(np.max(reserved) / total_memory),
        },
        "grid": smoke["grid"],
        "steps_requested": args.steps,
        "steps_recorded": int(step_wall.size),
        "elapsed_walltime_s": elapsed,
        "mean_step_walltime_s": mean_step,
        "p95_step_walltime_s": percentile(step_wall, 95),
        "mean_raman_walltime_s": float(np.mean(raman_wall)),
        "mean_ionization_walltime_s": float(np.mean(ion_wall)),
        "mean_linear_walltime_s": float(np.mean(linear_wall)),
        "total_convolution_count": int(np.sum(conv)),
        "convolution_count_per_z_step": float(np.mean(conv)),
        "convolution_count_per_operator_substep": float(np.sum(conv)/max(np.sum(substeps), 1)),
        "operator_substeps_per_z_step": float(np.mean(substeps)),
        "estimated_15000_step_walltime_s": estimated_full,
        "production_slurm_time_limit_s": args.production_slurm_limit_s,
        "estimated_fraction_of_time_limit": estimated_full / args.production_slurm_limit_s,
        "legacy_reference_mean_step_s": legacy_mean_step,
        "full_operator_slowdown_vs_legacy": mean_step / legacy_mean_step,
        "finite": bool(finite),
        "raman_rhs_l2_norm_max": float(np.max(rhs)),
        "raman_IR_max_raw": float(np.max(ir_raw)),
        "raman_target_loss_total_J": float(np.sum(target)),
        "raman_actual_loss_total_J": float(np.sum(actual)),
        "raman_step_closure_p99": percentile(closure, 99),
        "raman_cumulative_closure_final": float(cumulative_closure[-1]),
        "legacy_alpha_R_max": float(np.max(np.abs(legacy_alpha))),
    }
    metrics["gates"] = {
        "steps_complete": metrics["steps_recorded"] == metrics["steps_requested"],
        "finite": metrics["finite"],
        "energy_closure_finite": bool(
            np.isfinite(metrics["raman_step_closure_p99"])
            and np.isfinite(metrics["raman_cumulative_closure_final"])
        ),
        "memory_below_85_percent": metrics["gpu"]["peak_reserved_fraction"] < .85,
        "runtime_below_80_percent": metrics["estimated_fraction_of_time_limit"] < .8,
        "slowdown_at_most_3": metrics["full_operator_slowdown_vs_legacy"] <= 3.0 if args.case == "on" else True,
        "legacy_alpha_zero": metrics["legacy_alpha_R_max"] == 0.0,
        "raw_IR_nonzero": metrics["raman_IR_max_raw"] > 0.0,
        "target_loss_nonzero": metrics["raman_target_loss_total_J"] > 0.0,
        "applied_rhs_expected": metrics["raman_rhs_l2_norm_max"] > 0.0 if args.case == "on" else metrics["raman_rhs_l2_norm_max"] == 0.0,
        "actual_loss_expected": metrics["raman_actual_loss_total_J"] > 0.0 if args.case == "on" else metrics["raman_actual_loss_total_J"] == 0.0,
        "convolution_reuse": metrics["convolution_count_per_operator_substep"] == 2.0 if args.case == "on" else metrics["convolution_count_per_z_step"] == 1.0,
    }
    (args.out_dir / f"phase8b_full_size_smoke_{args.case}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

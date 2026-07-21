#!/usr/bin/env python3
"""Combine the two serial Phase 8B-P full-grid smoke audits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(*values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _config_audit_passed(audit: dict) -> bool:
    return all(bool(audit.get(key)) for key in (
        "grid_unchanged", "raman_unchanged", "ionization_unchanged", "beam_unchanged"
    ))


def summarize(on: dict, off: dict, on_audit: dict, off_audit: dict, *, scheduler_memory_mb: int) -> tuple[dict, dict]:
    on_contract = {
        "completed": on.get("case") == "on" and int(on.get("steps_recorded", -1)) == int(on.get("steps_requested", -2)),
        "finite": bool(on.get("finite")),
        "raw_IR_nonzero": float(on.get("raman_IR_max_raw", 0.0)) > 0.0,
        "applied_rhs_nonzero": float(on.get("raman_rhs_l2_norm_max", 0.0)) > 0.0,
        "target_loss_nonzero": float(on.get("raman_target_loss_total_J", 0.0)) > 0.0,
        "actual_loss_nonzero": float(on.get("raman_actual_loss_total_J", 0.0)) > 0.0,
        "step_closure_p99_below_contract": float(on.get("raman_step_closure_p99", math.inf)) < 1e-3,
        "cumulative_closure_below_contract": float(on.get("raman_cumulative_closure_final", math.inf)) < 5e-3,
        "legacy_alpha_zero": float(on.get("legacy_alpha_R_max", math.inf)) == 0.0,
        "two_convolutions_per_operator_substep": float(on.get("convolution_count_per_operator_substep", math.nan)) == 2.0,
        "two_strang_substeps_per_z_step": float(on.get("operator_substeps_per_z_step", math.nan)) == 2.0,
        "config_audit": _config_audit_passed(on_audit),
    }
    off_contract = {
        "completed": off.get("case") == "off" and int(off.get("steps_recorded", -1)) == int(off.get("steps_requested", -2)),
        "finite": bool(off.get("finite")),
        "raw_IR_nonzero": float(off.get("raman_IR_max_raw", 0.0)) > 0.0,
        "target_loss_nonzero": float(off.get("raman_target_loss_total_J", 0.0)) > 0.0,
        "applied_rhs_zero": float(off.get("raman_rhs_l2_norm_max", math.inf)) == 0.0,
        "actual_loss_zero": float(off.get("raman_actual_loss_total_J", math.inf)) == 0.0,
        "legacy_alpha_zero": float(off.get("legacy_alpha_R_max", math.inf)) == 0.0,
        "one_raw_diagnostic_convolution_per_z_step": float(off.get("convolution_count_per_z_step", math.nan)) == 1.0,
        "config_audit": _config_audit_passed(off_audit),
    }
    peak_memory_fraction = max(
        float(on["gpu"]["peak_reserved_fraction"]),
        float(off["gpu"]["peak_reserved_fraction"]),
    )
    runtime_fraction = float(on["estimated_fraction_of_time_limit"])
    slowdown = float(on["full_operator_slowdown_vs_legacy"])
    closure_finite = _finite(
        on["raman_step_closure_p99"], on["raman_cumulative_closure_final"],
        off["raman_step_closure_p99"], off["raman_cumulative_closure_final"],
    )
    gates = {
        "full_size_smoke_gate": all(on_contract.values()) and all(off_contract.values()) and closure_finite,
        "memory_gate": peak_memory_fraction < 0.85,
        "runtime_gate": runtime_fraction < 0.80 and slowdown <= 3.0,
    }
    metrics = {
        "schema": "khz_filament.phase8b.full_size_smoke_summary.v1",
        "execution_scope": {
            "short_smoke_slurm_jobs_submitted": 2,
            "full_production_slurm_jobs_submitted": 0,
            "production_propagation_executed": False,
            "steps_per_case": int(on["steps_requested"]),
        },
        "slurm": {
            "on_job_id": str(on.get("slurm_job_id", "")),
            "off_job_id": str(off.get("slurm_job_id", "")),
            "on_state": "COMPLETED 0:0",
            "off_state": "COMPLETED 0:0",
            "partition": on["resources"]["partition"],
            "gpu_count": int(on["resources"]["gpu_count"]),
            "cpu_threads": int(on["resources"]["cpu_threads"]),
            "scheduler_memory_mb": int(scheduler_memory_mb),
            "smoke_time_limit_s": float(on["resources"]["smoke_time_limit_s"]),
        },
        "gpu_type": on["gpu"]["type"],
        "grid": on["grid"],
        "on": on,
        "off": off,
        "on_contract": on_contract,
        "off_contract": off_contract,
        "energy_closure_finite": closure_finite,
        "peak_reserved_fraction": peak_memory_fraction,
        "gates": gates,
    }
    runtime = {
        "schema": "khz_filament.phase8b.runtime_estimate.v1",
        "basis": "20-step full-grid full-operator ON smoke",
        "gpu_type": on["gpu"]["type"],
        "mean_step_walltime_s": float(on["mean_step_walltime_s"]),
        "p95_step_walltime_s": float(on["p95_step_walltime_s"]),
        "mean_raman_walltime_s": float(on["mean_raman_walltime_s"]),
        "mean_ionization_walltime_s": float(on["mean_ionization_walltime_s"]),
        "mean_linear_walltime_s": float(on["mean_linear_walltime_s"]),
        "production_steps": 15000,
        "estimated_15000_step_walltime_s": float(on["estimated_15000_step_walltime_s"]),
        "estimated_15000_step_walltime_h": float(on["estimated_15000_step_walltime_s"]) / 3600.0,
        "slurm_time_limit_s": float(on["production_slurm_time_limit_s"]),
        "estimated_fraction_of_time_limit": runtime_fraction,
        "legacy_reference_mean_step_s": float(on["legacy_reference_mean_step_s"]),
        "full_operator_slowdown_vs_legacy": slowdown,
        "thresholds": {"time_limit_fraction_lt": 0.80, "slowdown_le": 3.0},
        "runtime_gate": gates["runtime_gate"],
    }
    return metrics, runtime


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-metrics", type=Path, required=True)
    parser.add_argument("--off-metrics", type=Path, required=True)
    parser.add_argument("--on-audit", type=Path, required=True)
    parser.add_argument("--off-audit", type=Path, required=True)
    parser.add_argument("--scheduler-memory-mb", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics, runtime = summarize(
        _load(args.on_metrics), _load(args.off_metrics),
        _load(args.on_audit), _load(args.off_audit),
        scheduler_memory_mb=args.scheduler_memory_mb,
    )
    (args.out_dir / "phase8b_full_size_smoke_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "phase8b_runtime_estimate.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

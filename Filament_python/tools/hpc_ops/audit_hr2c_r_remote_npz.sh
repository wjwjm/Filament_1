#!/usr/bin/env bash
# Read-only scalar audit for the completed HR-2E 120 fs NPZ files.
#
# The native-SSH wrapper stages this script transiently and expects exactly
# one receipt JSON object on stdout.  The script never writes a run directory
# and never emits a field/map payload.
set -euo pipefail

python_bin='/data/home/scvi806/.conda/envs/Filament_python/bin/python'
exec "$python_bin" - "$@" <<'PY'
import json
import math
import sys

import numpy as np


def scalar(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected scalar")
    return array.reshape(()).item()


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "size": int(array.size),
        "finite": bool(np.all(np.isfinite(array))),
        "minimum": float(np.min(array)) if array.size else math.nan,
        "maximum": float(np.max(array)) if array.size else math.nan,
        "sum": float(np.sum(array, dtype=np.float64)),
    }


def residual_summary(values):
    result = summary(values)
    array = np.asarray(values, dtype=np.float64)
    result.update({
        "p99": float(np.quantile(array, 0.99)) if array.size else math.nan,
        "p999": float(np.quantile(array, 0.999)) if array.size else math.nan,
        "count_gt_1e-3": int(np.count_nonzero(array > 1.0e-3)),
    })
    return result


def optional_summary(data, key):
    if key not in data:
        return {"present": False}
    result = summary(data[key])
    result["present"] = True
    return result


def inspect(path):
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        required = (
            "E_dep_raman_interval_J",
            "E_dep_raman_interval_operator_J",
            "E_dep_raman_interval_closure_residual_J",
            "E_dep_raman_pulse_J",
            "E_dep_raman_operator_pulse_J",
            "E_dep_raman_pulse_closure_residual_J",
            "n_intervals",
            "raman_actual_local_negative_min_J_m2",
            "raman_operator_applied",
            "raman_target_loss_step_J",
            "raman_actual_loss_step_J",
            "raman_cumulative_closure_residual",
            "raman_closure_residual_step",
            "raman_energy_projection_iterations",
            "raman_energy_projection_scale_deviation",
            "raman_energy_projection_initial_residual",
        )
        missing = sorted(set(required) - keys)
        if missing:
            raise ValueError("missing required keys: " + ",".join(missing))
        interval = np.asarray(data["E_dep_raman_interval_J"], dtype=np.float64)
        operator = np.asarray(
            data["E_dep_raman_interval_operator_J"], dtype=np.float64
        )
        stored = np.asarray(
            data["E_dep_raman_interval_closure_residual_J"], dtype=np.float64
        )
        difference = interval - operator
        denominator = np.maximum.reduce((
            np.abs(interval), np.abs(operator), np.full(interval.size, 1e-30),
        ))
        relative = np.abs(difference) / denominator
        target = np.asarray(data["raman_target_loss_step_J"], dtype=np.float64)
        actual = np.asarray(data["raman_actual_loss_step_J"], dtype=np.float64)
        target_actual = None
        if target.size == actual.size:
            target_actual = summary(target - actual)
            target_actual["max_relative"] = float(np.max(
                np.abs(target - actual)
                / np.maximum.reduce((np.abs(target), np.abs(actual), np.full(target.size, 1e-30)))
            )) if target.size else 0.0
        return {
            "path": path,
            "n_intervals": int(scalar(data["n_intervals"])),
            "canonical_interval": summary(interval),
            "operator_signed_interval": summary(operator),
            "stored_residual": summary(stored),
            "recomputed_residual": summary(difference),
            "stored_residual_matches_recomputed": bool(np.allclose(
                stored, difference, rtol=2e-12, atol=1e-30
            )),
            "clipping_penalty_nonnegative": bool(np.all(difference >= -1e-30)),
            "max_relative_old_level1_difference": float(np.max(relative)) if relative.size else 0.0,
            "negative_local_min": summary(data["raman_actual_local_negative_min_J_m2"]),
            "operator_applied_all": bool(np.all(np.asarray(data["raman_operator_applied"], dtype=bool))),
            "target_step": summary(target),
            "actual_step": summary(actual),
            "target_actual_step_difference": target_actual,
            "target_step_matches_interval_count": bool(target.size == interval.size),
            "actual_step_matches_interval_count": bool(actual.size == interval.size),
            "cumulative_operator_closure": optional_summary(
                data, "raman_cumulative_closure_residual"
            ),
            "operator_step_closure": residual_summary(
                data["raman_closure_residual_step"]
            ),
            "projection_iterations": summary(data["raman_energy_projection_iterations"]),
            "projection_scale_deviation": summary(
                data["raman_energy_projection_scale_deviation"]
            ),
            "projection_initial_residual": summary(
                data["raman_energy_projection_initial_residual"]
            ),
            "pulse": {
                "canonical_J": float(scalar(data["E_dep_raman_pulse_J"])),
                "operator_signed_J": float(scalar(data["E_dep_raman_operator_pulse_J"])),
                "residual_J": float(scalar(data["E_dep_raman_pulse_closure_residual_J"])),
            },
        }


try:
    if len(sys.argv) != 4:
        raise ValueError("expected coarse, candidate, and fine NPZ paths")
    report = {
        "schema": "khz_filament.hr2c_r.remote_npz_audit.v1",
        "cases": {
            "coarse": inspect(sys.argv[1]),
            "candidate": inspect(sys.argv[2]),
            "fine": inspect(sys.argv[3]),
        },
    }
    print(json.dumps({
        "schema": "filament.hpc_ops.remote_exec.v1",
        "ok": True,
        "state": "completed",
        "audit": report,
    }, sort_keys=True, allow_nan=False))
except Exception:
    print(json.dumps({
        "schema": "filament.hpc_ops.remote_exec.v1",
        "ok": False,
        "state": "failed",
    }, sort_keys=True))
    raise
PY

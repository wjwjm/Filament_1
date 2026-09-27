#!/usr/bin/env python3
"""Bounded N2/K64 GPU binding qualification; this is not formal E5 science."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
from datetime import datetime, timezone

import numpy as np

from KHz_filament.hr4e5_evidence import sha256_file
from KHz_filament.hr4e5_production import (
    E5AProductionSpec, LOCAL_ORCHESTRATION_QUALIFICATION,
    _source_hash, open_e5_1a_production_campaign,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _prepare_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    inputs = root / "inputs"
    inputs.mkdir(mode=0o700)
    config = inputs / "config.json"
    value = {
        "grid": {"Nx": 8, "Ny": 8, "Nt": 8, "Lx": 8e-4, "Ly": 8e-4, "Twin": 80e-15},
        "beam": {"w0": 1.5e-4, "tau_fwhm": 40e-15, "energy_J": 1e-10,
                 "focal_length": None},
        "propagation": {"z_max": 64e-4, "dz": 1e-4, "linear_model": "paraxial",
            "auto_substep": False, "focus_window_step": False,
            "limit_focus_window": False, "progress_every_z": 0,
            "energy_probe_every": 0, "diag_extra": False,
            "use_electronic_kerr": False, "use_raman_phase": False,
            "use_raman_absorption": False, "use_plasma_phase": False,
            "use_ionization_loss": False, "use_ionization_solver": False},
        "ionization": {"species": []},
        "heat": {"hr3b_enabled": True, "f_rep": 5e6, "dt_hydro": 1e-7,
                 "chi": 2.17e-5, "D_th": 2.17e-5, "nu": 1.5e-5,
                 "gravity_x": 0.0, "gravity_y": -9.81},
        "run": {"Npulses": 2},
        "raman": {"enabled": False, "absorption": False},
    }
    config.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    source = inputs / "source.json"
    source.write_text(json.dumps({"schema": "local-source.v1",
        "canonical_array_hash": _source_hash(config)}, sort_keys=True), encoding="utf-8")
    lut = inputs / "lut.json"
    lut.write_text(json.dumps({"schema": "local-lut.v1", "mode": "DISABLED_BY_CONFIG",
        "config_sha256": sha256_file(config)}, sort_keys=True), encoding="utf-8")
    pre0 = inputs / "pre0.npy"
    np.save(pre0, np.zeros((64, 8, 8), dtype=np.float64))
    return config, source, lut, pre0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--hydro-workers", type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    if os.environ.get("UPPE_USE_GPU") != "0" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU preflight needs a CPU coordinator inside Slurm")
    root = args.run_root.resolve(strict=True)
    if any(root.iterdir()):
        raise RuntimeError("GPU preflight run root must be empty before execution")
    config, source, lut, pre0 = _prepare_inputs(root)
    started = _utc()
    spec = E5AProductionSpec(
        root=root / "campaign", campaign_id=args.campaign_id,
        config_path=config, source_manifest_path=source,
        lut_manifest_path=lut, pre0_delta_n_path=pre0,
        n_pulses=2, candidate_hydro_workers=args.hydro_workers,
        admission_level=LOCAL_ORCHESTRATION_QUALIFICATION,
        require_gpu_backend=True, reference_subprocess=True,
        safety_margin_bytes=1024**2,
    )
    result = open_e5_1a_production_campaign(spec).run()
    summary = {
        "schema": "khz_filament.e5_1b_wp2.gpu_preflight.v1",
        "qualification": "LOCAL_ORCHESTRATION_QUALIFICATION_ON_HPC_GPU",
        "started_utc": started, "ended_utc": _utc(),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hydro_workers": args.hydro_workers,
        "scheduler_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "status": result["status"],
        "next_pair_index": result["next_pair_index"],
        "implementation_identity_sha256": json.loads(
            (spec.root / "E5_1B_IMPLEMENTATION_IDENTITY.json").read_text(encoding="utf-8")
        )["identity_sha256"],
        "terminal_inventory_status": result["terminal_inventory_after_close"]["status"],
        "exact": [],
    }
    for pulse in range(2):
        report = json.loads((spec.root / "exact" / f"p{pulse}.json").read_text(encoding="utf-8"))
        summary["exact"].append({"pulse": pulse, "status": report["status"],
                                 "mismatch_count": report["mismatch_count"]})
    path = root / "gpu_preflight_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if (summary["status"] != "COMPLETE" or summary["next_pair_index"] != 2
            or summary["terminal_inventory_status"] != "PASS"
            or any(row["status"] != "PASS" or row["mismatch_count"] != 0
                   for row in summary["exact"])):
        raise RuntimeError("bounded GPU preflight terminal or exact check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

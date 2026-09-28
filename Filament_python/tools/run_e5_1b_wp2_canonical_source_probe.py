#!/usr/bin/env python3
"""One-GPU canonical E_source load qualification without propagation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import uuid


EXPECTED_ARRAY_SHA = "595ed4eedac5438e0fe71af7897235c69f52158bc54e95c110192c624058a4a5"


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _gpu_uuid(raw: object) -> str:
    if isinstance(raw, (bytes, bytearray, list, tuple)) and len(raw) == 16:
        return "GPU-" + str(uuid.UUID(bytes=bytes(raw)))
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    value = str(raw).rstrip("\x00")
    return value if value.startswith("GPU-") else "GPU-" + str(uuid.UUID(value))


def _child(args: argparse.Namespace) -> int:
    if os.environ.get("UPPE_USE_GPU") != "1" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("source probe child requires Slurm GPU runtime")
    import cupy as cp
    from KHz_filament.hr4e5_canonical_source import load_canonical_source_for_pulse
    from KHz_filament.hr4e5_recovery_identity import capture_implementation_identity

    identity = json.loads(args.admission.read_text(encoding="utf-8"))
    if capture_implementation_identity()["identity_sha256"] != identity["implementation_identity_sha256"]:
        raise RuntimeError("HPC implementation identity changed between coordinator and child")
    if int(cp.cuda.runtime.getDeviceCount()) != 1:
        raise RuntimeError("CuPy must see exactly one scheduler GPU")
    cp.cuda.Device(0).use()
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_uuid = props.get("uuid") if isinstance(props, dict) else None
    if raw_uuid is None:
        raw_uuid = cp.cuda.runtime.deviceGetUuid(0)
    gpu_uuid = _gpu_uuid(raw_uuid)
    trial = cp.arange(4096, dtype=cp.int32)
    if not bool(cp.all(trial + trial == 2 * trial).get()):
        raise RuntimeError("deterministic CuPy compute failed")
    cp.cuda.runtime.deviceSynchronize()

    record = identity["source_identity"]["canonical_source"]
    if record["canonical_array_hash"] != EXPECTED_ARRAY_SHA:
        raise RuntimeError("canonical source authority differs from frozen admission")
    source, receipt = load_canonical_source_for_pulse(
        record, admission_identity=identity, config_path=args.config,
        source_manifest_path=args.source_manifest, trajectory=args.role,
        role="OPTICAL", pulse=0, attempt=args.repeat,
    )
    if (receipt["runtime_roundtrip_array_hash"] != EXPECTED_ARRAY_SHA
            or receipt["private_backend_buffer"]["backend"] != "cupy"
            or not receipt["host_device_host_exact"]):
        raise RuntimeError("canonical host-device-host identity failed")
    receipt.update({"pid": os.getpid(), "gpu_uuid": gpu_uuid,
                    "node": socket.gethostname().split(".", 1)[0],
                    "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "probe_repeat": args.repeat, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    _write_json(args.output, receipt)
    del source
    return 0


def _parent(args: argparse.Namespace) -> int:
    from KHz_filament.hr4e5_canonical_source import seal_or_validate_canonical_source
    from KHz_filament.hr4e5_production import (
        E5AProductionSpec, LOCAL_ORCHESTRATION_QUALIFICATION, _build_admission,
    )
    from KHz_filament.hr4e5_recovery_identity import capture_implementation_identity
    from tools.run_e5_1b_wp2_gpu_preflight import _prepare_inputs

    if os.environ.get("UPPE_USE_GPU") != "0":
        raise RuntimeError("CPU coordinator must set UPPE_USE_GPU=0")
    if not args.admission_only:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("GPU source probe requires Slurm")
        mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if len(mask.split(",")) != 1 or not mask or mask == "-1":
            raise RuntimeError("source probe requires one scheduler-visible GPU")
        if os.environ.get("SLURM_GPUS_ON_NODE") != "1":
            raise RuntimeError("source probe requires exactly one allocated GPU")
        if socket.gethostname().split(".", 1)[0] == "g0603":
            raise RuntimeError("g0603 is temporarily excluded for E5-1B/E5-1C")
    root = args.run_root.resolve(strict=True)
    if any(root.iterdir()):
        raise RuntimeError("source probe run root must be empty")
    config, source_manifest, lut, pre0 = _prepare_inputs(root)
    spec = E5AProductionSpec(
        root=root / "campaign", campaign_id=args.campaign_id,
        config_path=config, source_manifest_path=source_manifest,
        lut_manifest_path=lut, pre0_delta_n_path=pre0,
        n_pulses=2, candidate_hydro_workers=2,
        admission_level=LOCAL_ORCHESTRATION_QUALIFICATION,
        require_gpu_backend=True, reference_subprocess=True,
        safety_margin_bytes=1024**2,
    )
    implementation = capture_implementation_identity()
    identity, _, _ = _build_admission(spec, implementation_identity=implementation)
    record = identity["source_identity"]["canonical_source"]
    if record["canonical_array_hash"] != EXPECTED_ARRAY_SHA:
        raise RuntimeError("canonical artifact differs from frozen admission")
    seal = seal_or_validate_canonical_source(
        record, admission_identity=identity, implementation_identity=implementation,
        config_path=config, source_manifest_path=source_manifest,
        create_if_missing=True,
    )
    _write_json(root / "admission_identity.json", identity)
    _write_json(root / "implementation_identity.json", implementation)
    _write_json(root / "source_seal_receipt.json", seal)
    if args.admission_only:
        _write_json(root / "probe_summary.json", {"status": "LOCAL_ADMISSION_ONLY",
            "implementation_identity_sha256": implementation["identity_sha256"],
            "canonical_array_hash": record["canonical_array_hash"],
            "propagation": False})
        return 0

    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30, check=False)
    (root / "nvidia_smi.stdout.txt").write_bytes(smi.stdout)
    (root / "nvidia_smi.stderr.txt").write_bytes(smi.stderr)
    if smi.returncode != 0:
        raise RuntimeError("nvidia-smi health gate failed")
    receipts = []
    for role in ("R", "C"):
        for repeat in range(1, 4):
            name = f"{role.lower()}_{repeat}"
            output = root / f"{name}.json"
            child_env = os.environ.copy()
            child_env["UPPE_USE_GPU"] = "1"
            cmd = [sys.executable, "-s", "-B", str(Path(__file__).resolve()),
                   "--child", "--admission", str(root / "admission_identity.json"),
                   "--config", str(config), "--source-manifest", str(source_manifest),
                   "--role", role, "--repeat", str(repeat), "--output", str(output)]
            result = subprocess.run(cmd, env=child_env, capture_output=True, timeout=300, check=False)
            (root / f"{name}.stdout.txt").write_bytes(result.stdout)
            (root / f"{name}.stderr.txt").write_bytes(result.stderr)
            if result.returncode != 0:
                _write_json(root / "probe_failure.json", {"status": "FAIL", "role": role,
                    "repeat": repeat, "exit_code": result.returncode})
                return 1
            receipts.append(json.loads(output.read_text(encoding="utf-8")))
    pids = [int(row["pid"]) for row in receipts]
    uuids = [str(row["gpu_uuid"]) for row in receipts]
    if (len(set(pids)) != 6 or len(set(uuids)) != 1
            or uuids[0].encode("ascii") not in smi.stdout
            or any(row["runtime_roundtrip_array_hash"] != EXPECTED_ARRAY_SHA for row in receipts)
            or any(row["artifact_logical_id"] != record["logical_id"] for row in receipts)
            or any(row["artifact_file_sha256"] != record["file_sha256"] for row in receipts)
            or any(row["payload_bytes_sha256"] != record["payload_bytes_sha256"] for row in receipts)
            or any(row["source_alias_checked"] is not True for row in receipts)
            or any(row["implementation_identity_sha256"] != implementation["identity_sha256"] for row in receipts)):
        raise RuntimeError("source-only process, GPU UUID, or identity matrix failed")
    _write_json(root / "probe_summary.json", {
        "schema": "khz_filament.e5_1b_wp2.canonical_source_gpu_probe.v1",
        "status": "PASS", "propagation": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "node": socket.gethostname().split(".", 1)[0],
        "gpu_uuid": uuids[0], "child_pids": pids,
        "R_count": 3, "C_count": 3,
        "canonical_array_hash": EXPECTED_ARRAY_SHA,
        "artifact_file_sha256": record["file_sha256"],
        "payload_bytes_sha256": record["payload_bytes_sha256"],
        "implementation_identity_sha256": implementation["identity_sha256"],
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--role", choices=("R", "C"))
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        if any(value is None for value in (args.admission, args.config, args.source_manifest,
                                           args.role, args.repeat, args.output)):
            parser.error("child arguments are incomplete")
        return _child(args)
    if args.run_root is None or args.campaign_id is None:
        parser.error("parent arguments are incomplete")
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

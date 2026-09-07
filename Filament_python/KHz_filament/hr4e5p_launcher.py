"""Fail-closed P3 submission planning for the HR-4E-5P executor.

This module is orchestration-only: it does not import or invoke the HR-4
solver. A final receipt is created only after every sbatch call returns a
numeric job identifier.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .hr4e_timestep import json_safe, sha256_file


P3_INDICES = (0, 1, 2500, 5000, 7827, 8022, 9000, 10338, 12000, 14997, 14998, 14999)
P3_SOURCE_FILE_SHA256 = "70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467"
P3_CASES = (("p3_serial", 1, "serial"), ("p3_parallel_g1", 1, "parallel"), ("p3_parallel_g2", 2, "parallel"), ("p3_parallel_g4", 4, "parallel"), ("p3_parallel_g7", 7, "parallel"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(dict(value)), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _case_records(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = value.get("screen_records")
    if not isinstance(records, list) or len(records) != len(P3_INDICES):
        raise ValueError("P3 payload must contain exactly the frozen 12 screens")
    result = []
    for ordinal, (record, source_index) in enumerate(zip(records, P3_INDICES, strict=True)):
        if not isinstance(record, Mapping) or int(record.get("ordinal", -1)) != ordinal or int(record.get("source_index", -1)) != source_index:
            raise ValueError("P3 screen identity/order differs from the frozen validation set")
        screen_id = f"source_index_{source_index:05d}"
        if str(record.get("screen_id", "")) != screen_id or not isinstance(record.get("source_array_sha256"), str):
            raise ValueError("P3 screen identity or source hash is invalid")
        result.append({"ordinal": ordinal, "screen_id": screen_id, "source_index": source_index, "z_m": float(record["z_m"]), "source_array_sha256": str(record["source_array_sha256"])})
    return result


def validate_p3_case_payloads(case_inputs: Mapping[str, str | Path]) -> dict[str, Any]:
    """Ensure all serial/parallel cases retain identical frozen input identity."""
    expected_names = [item[0] for item in P3_CASES]
    if list(case_inputs) != expected_names:
        raise ValueError("P3 cases must be the fixed serial plus 1/2/4/7-GPU set in order")
    baseline: dict[str, Any] | None = None
    reports = []
    for name, gpu_count, mode in P3_CASES:
        path = Path(case_inputs[name])
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _read_json(path)
        if payload.get("schema") != "khz_filament.hr4e5p.validation_input.v1":
            raise ValueError("P3 validation-input schema is invalid")
        identity = {"source_manifest_sha256": str(payload.get("source_manifest_sha256", "")), "source_state_file_sha256": str(payload.get("source_state_file_sha256", "")), "source_state_array_sha256": str(payload.get("source_state_array_sha256", "")), "dtype": str(payload.get("dtype", "")), "geometry": payload.get("geometry"), "screen_records": _case_records(payload)}
        if identity["source_state_file_sha256"] != P3_SOURCE_FILE_SHA256:
            raise ValueError("P3 source-state provenance does not match the strict preflight source")
        if identity["dtype"] != "float64" or not isinstance(identity["geometry"], Mapping):
            raise ValueError("P3 payload dtype or geometry is invalid")
        state_name = "serial_state" if mode == "serial" else "parallel_state"
        state = payload.get(state_name)
        if not isinstance(state, Mapping) or not Path(str(state.get("output_path", ""))).parent.is_dir():
            raise ValueError("P3 case state is incomplete or missing")
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise ValueError("P3 serial/parallel payload scientific identity mismatch")
        reports.append({"case_id": name, "mode": mode, "gpu_count": gpu_count, "validation_input": str(path), "validation_input_sha256": sha256_file(path), "state_json": str(path.with_name(state_name + ".json")), "partition_json": str(path.with_name("partition.json"))})
    assert baseline is not None
    return {"schema": "khz_filament.hr4e5p.p3_payload_validation.v1", "status": "PASS", "frozen_source_indices": list(P3_INDICES), "scientific_identity": baseline, "cases": reports}


def build_submission_plan(*, payload_validation: Mapping[str, Any], expected_git_sha: str, repo: str | Path, batch: str | Path) -> dict[str, Any]:
    batch_path = Path(batch)
    if not batch_path.is_file() or not re.fullmatch(r"[0-9a-f]{40}", expected_git_sha):
        raise ValueError("P3 batch entry or expected Git SHA is invalid")
    cases = []
    for case in payload_validation.get("cases", []):
        state, partition = Path(str(case["state_json"])), Path(str(case["partition_json"]))
        if not state.is_file() or not partition.is_file():
            raise FileNotFoundError("P3 state or partition artifact is missing")
        cases.append({**dict(case), "state_json": str(state), "partition_json": str(partition)})
    if len(cases) != len(P3_CASES):
        raise ValueError("P3 plan lacks a required case")
    return {"schema": "khz_filament.hr4e5p.p3_submission_plan.v1", "status": "READY_FOR_NO_SUBMIT_OR_SUBMIT", "expected_git_sha": expected_git_sha, "repo": str(Path(repo)), "batch": str(batch_path), "batch_sha256": sha256_file(batch_path), "requested_cpus_per_task": 8, "requested_walltime": "04:00:00", "payload_validation": dict(payload_validation), "cases": cases}


def _submission_command(plan: Mapping[str, Any], case: Mapping[str, Any]) -> list[str]:
    state = str(case["state_json"])
    case_dir = str(Path(state).parent)
    exports = ["ALL", f"EXPECTED_GIT_SHA={plan['expected_git_sha']}", f"REPO_DIR={plan['repo']}", f"CASE_DIR={case_dir}", f"MODE={case['mode']}", f"STATE_JSON={state}", f"PARTITION_JSON={case['partition_json']}", f"GPU_COUNT={case['gpu_count']}", "N_HYDRO_STEPS=1000", "BATCH_INTERVALS=1"]
    return ["sbatch", "--parsable", f"--job-name=e5p-{case['case_id']}", f"--gres=gpu:{case['gpu_count']}", f"--ntasks={case['gpu_count']}", f"--output={case_dir}/slurm-%j.out", f"--error={case_dir}/slurm-%j.err", "--export=" + ",".join(exports), str(plan["batch"])]


def submit_submission_plan(plan: Mapping[str, Any], *, attempt_path: str | Path, receipt_path: str | Path, submitter: Callable[[Sequence[str]], str] | None = None) -> dict[str, Any]:
    """Submit a validated plan; leave only a failed attempt record on failure."""
    if plan.get("schema") != "khz_filament.hr4e5p.p3_submission_plan.v1":
        raise ValueError("P3 submission plan schema is invalid")
    attempt, receipt = Path(attempt_path), Path(receipt_path)
    if receipt.exists():
        raise FileExistsError(receipt)
    jobs: list[dict[str, Any]] = []
    _atomic_json(attempt, {"schema": "khz_filament.hr4e5p.p3_submission_attempt.v1", "status": "SUBMITTING", "jobs": jobs})
    invoke = submitter or (lambda command: subprocess.run(command, check=True, text=True, capture_output=True).stdout)
    try:
        for case in plan["cases"]:
            output = str(invoke(_submission_command(plan, case))).strip()
            job_id = output.split(";", 1)[0]
            if not re.fullmatch(r"[0-9]+", job_id):
                raise RuntimeError("sbatch did not return a numeric job identifier")
            jobs.append({"case_id": case["case_id"], "job_id": job_id, "gpu_count": case["gpu_count"], "mode": case["mode"]})
            _atomic_json(attempt, {"schema": "khz_filament.hr4e5p.p3_submission_attempt.v1", "status": "SUBMITTING", "jobs": jobs})
    except Exception as error:
        _atomic_json(attempt, {"schema": "khz_filament.hr4e5p.p3_submission_attempt.v1", "status": "FAILED", "jobs": jobs, "failure_class": type(error).__name__})
        raise
    temporary = receipt.with_name(receipt.name + ".tmp")
    temporary.write_text("case_id\tjob_id\tgpu_count\tmode\n" + "".join(f"{job['case_id']}\t{job['job_id']}\t{job['gpu_count']}\t{job['mode']}\n" for job in jobs), encoding="utf-8", newline="\n")
    os.replace(temporary, receipt)
    _atomic_json(attempt, {"schema": "khz_filament.hr4e5p.p3_submission_attempt.v1", "status": "COMPLETED", "jobs": jobs, "receipt": str(receipt)})
    return {"jobs": jobs, "receipt": str(receipt)}

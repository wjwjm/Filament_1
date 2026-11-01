"""Diagnostic-only wrapper around the frozen S4-6F Eq10 controller."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

from tools.hpc_ops import provenance_v2 as pv

from . import DIAGNOSTIC_SCHEMA, DIAGNOSTIC_TASK
from .observer import SCHEMA as TELEMETRY_SCHEMA
from .observer import ProcessObserver, probe_current_pid


class ObserverWindowFailure(RuntimeError):
    """An observer failure raised only after the frozen supervisor returns."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _module_manifest_paths(repo: Path) -> set[str]:
    package = Path(__file__).resolve().parent
    return {
        path.relative_to(repo).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sbatch"}
    }


def _validate_source_binding(report: Path, execution: dict[str, Any]) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    manifest_path = report / "EXECUTION_SOURCE_MANIFEST.json"
    try:
        manifest = pv.validate_manifest(repo, manifest_path, require_clean=True, require_hash_scope=True)
        records = pv.iter_records(manifest, require_hash_scope=True)
    except Exception as exc:
        raise RuntimeError("S46G_SOURCE_MANIFEST_INVALID") from exc
    paths = {str(row.get("path")) for row in records if isinstance(row, dict)}
    missing = sorted(_module_manifest_paths(repo) - paths)
    if missing:
        raise RuntimeError("S46G_SOURCE_MANIFEST_MODULE_MISSING:" + ",".join(missing))
    execution_commit = execution.get("execution_commit")
    if not isinstance(execution_commit, str) or manifest.get("head") != execution_commit:
        raise RuntimeError("S46G_EXECUTION_SOURCE_COMMIT")
    try:
        current_head = pv._git(repo, "rev-parse", "HEAD")
    except Exception as exc:
        raise RuntimeError("S46G_CURRENT_HEAD_UNAVAILABLE") from exc
    if current_head != execution_commit:
        raise RuntimeError("S46G_EXECUTION_CURRENT_HEAD")
    source_sha = execution.get("source_manifest_raw_sha256")
    if not isinstance(source_sha, str) or pv.raw_sha256_file(manifest_path) != source_sha:
        raise RuntimeError("S46G_EXECUTION_SOURCE_IDENTITY")
    return {"manifest": manifest_path, "head": current_head, "module_paths": sorted(_module_manifest_paths(repo))}


def _validate_policy_plan_binding(report: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    from .policy_analysis import policy_plan

    plan_path = report / "POLICY_PLAN.json"
    if not plan_path.is_file():
        raise RuntimeError("S46G_POLICY_PLAN_MISSING")
    if context.get("diagnostic_kind") != "MEMORY_POLICY_CROSSOVER":
        raise RuntimeError("S46G_POLICY_DIAGNOSTIC_KIND")
    if pv.raw_sha256_file(plan_path) != context.get("policy_plan_raw_sha256"):
        raise RuntimeError("S46G_POLICY_PLAN_RAW_SHA256")
    if _read(plan_path) != policy_plan():
        raise RuntimeError("S46G_POLICY_PLAN_SOURCE_BINDING")
    return _read(plan_path)


def validate_context(report: str | Path, *, execution_commit: str | None = None) -> dict[str, Any]:
    report = Path(report).resolve(strict=True)
    path = report / "DIAGNOSTIC_CONTEXT.json"
    if not path.is_file():
        raise RuntimeError("S46G_DIAGNOSTIC_CONTEXT_MISSING")
    value = _read(path)
    required = {
        "schema": DIAGNOSTIC_SCHEMA,
        "task": DIAGNOSTIC_TASK,
        "candidate": "EQ10_P48",
        "diagnostic_only": True,
        "observer_period_s": 5.0,
        "numa_every_samples": 3,
        "science_arrays_retained": False,
        "old_results_superseded": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeError(f"S46G_CONTEXT_BINDING:{key}")
    commit = execution_commit or value.get("execution_commit")
    if not isinstance(commit, str) or value.get("execution_commit") != commit:
        raise RuntimeError("S46G_CONTEXT_EXECUTION_COMMIT")
    if value.get("observer_thread_cap") != 512 or value.get("sample_cap") != 1450 or value.get("output_cap_bytes") != 64 * 1024 * 1024:
        raise RuntimeError("S46G_CONTEXT_BOUNDS")
    context_sha = pv.raw_sha256_file(path)
    expected_sha = os.environ.get("S46G_CONTEXT_SHA256")
    if not expected_sha:
        raise RuntimeError("S46G_CONTEXT_RAW_SHA256_MISSING")
    if context_sha != expected_sha.lower():
        raise RuntimeError("S46G_CONTEXT_RAW_SHA256")
    execution_path = report / "EXECUTION.json"
    if not execution_path.is_file():
        raise RuntimeError("S46G_EXECUTION_MISSING")
    execution = _read(execution_path)
    if execution.get("diagnostic_context_raw_sha256") != context_sha:
        raise RuntimeError("S46G_EXECUTION_CONTEXT_SHA256")
    if not isinstance(execution.get("execution_commit"), str) or value.get("execution_commit") != execution.get("execution_commit"):
        raise RuntimeError("S46G_CONTEXT_EXECUTION_BINDING")
    _validate_source_binding(report, execution)
    if value.get("diagnostic_kind") == "MEMORY_POLICY_CROSSOVER":
        _validate_policy_plan_binding(report, value)
    value = dict(value)
    value["context_raw_sha256"] = context_sha
    value["execution"] = execution
    return value


def _diagnostic_receipt(report: Path, *, status: str, result: Any = None,
                        observations: list[dict[str, Any]] | None = None,
                        error: BaseException | None = None,
                        error_traceback: str | None = None,
                        observations_valid: bool = False,
                        science_exact: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "filament.cpu_s46g.diagnostic_result.v1",
        "status": status,
        "classification": "DIAGNOSTIC_ONLY",
        "diagnostic_only": True,
        "candidate": "EQ10_P48",
        "report": str(report),
        "observation_windows": len(observations or []),
        "observation_receipts": observations or [],
        "observation_receipts_valid": observations_valid,
        "science_exact": science_exact,
        "complete": status == "DIAGNOSTIC_COMPLETE",
        "diagnostic_classification": status,
        "underlying_result_status": result.get("status") if isinstance(result, dict) else None,
        "underlying_result_classification": result.get("classification") if isinstance(result, dict) else None,
        "old_results_superseded": False,
        "science_arrays_retained": False,
        "default": "I0",
        "production_qualified": False,
    }
    if error is not None:
        payload["error"] = repr(error)
        payload["traceback"] = error_traceback or ""
    return payload


def _expected_cases() -> tuple[str, ...]:
    protocol = importlib.import_module("tools.cpu_s4_6f_fullsize_parallel.protocol")
    return tuple(row["case"] for row in protocol.matrix("EQ10_P48"))


def _valid_observations(observations: list[dict[str, Any]]) -> bool:
    if len(observations) != 14:
        return False
    if {row.get("case") for row in observations} != set(_expected_cases()):
        return False
    for row in observations:
        receipt = row.get("receipt")
        if not isinstance(receipt, dict) or receipt.get("schema") != TELEMETRY_SCHEMA + ".receipt.v1":
            return False
        if receipt.get("status") not in {"STOPPED", "PID_EXITED"} or receipt.get("errors"):
            return False
        if type(receipt.get("samples")) is not int or receipt["samples"] < 1 or type(receipt.get("bytes")) is not int or receipt["bytes"] < 1:
            return False
        if receipt.get("science_arrays_retained") is not False or receipt.get("short_lived_children_observed") is not False:
            return False
        if receipt.get("last_periodic_sample_only") is not True or receipt.get("end_state_snapshot") is not False:
            return False
        if (receipt.get("max_samples"), receipt.get("max_threads"), receipt.get("max_output_bytes")) != (1450, 512, 64 * 1024 * 1024):
            return False
        if not isinstance(receipt.get("collection_wall_ns"), int) or receipt["collection_wall_ns"] < 0:
            return False
        if not isinstance(receipt.get("clock"), dict) or not isinstance(receipt.get("units"), dict):
            return False
    return True


def _scientific_result_complete(result: Any) -> bool:
    if not isinstance(result, dict) or not isinstance(result.get("status"), str) or result.get("status") == "SCIENTIFIC_NONEXACT":
        return False
    gates = result.get("gates")
    return (
        isinstance(gates, dict)
        and gates.get("science_exact") is True
        and result.get("formal_windows") == 8
        and result.get("qualification_windows") == 2
        and result.get("reference_control_windows") == 4
    )


def preflight(report: str | Path, candidate: str = "EQ10_P48") -> dict[str, Any]:
    """Run the frozen 6F read-only preflight with the diagnostic context bound."""

    report = Path(report).resolve(strict=True)
    if str(candidate).upper() != "EQ10_P48":
        raise RuntimeError("S46G_EQ10_ONLY")
    context = validate_context(report)
    observer_probe = probe_current_pid()
    frozen_hpc = importlib.import_module("tools.cpu_s4_6f_fullsize_parallel.hpc")
    result = frozen_hpc.preflight(report, "EQ10_P48")
    if result.get("execution_commit") != context.get("execution_commit"):
        raise RuntimeError("S46G_CONTEXT_EXECUTION_COMMIT_MISMATCH")
    result = dict(result)
    result.update(
        diagnostic_only=True,
        diagnostic_context_raw_sha256=pv.raw_sha256_file(report / "DIAGNOSTIC_CONTEXT.json"),
        observer_period_s=context["observer_period_s"],
        numa_every_samples=context["numa_every_samples"],
        observer_capability=observer_probe,
    )
    return result


def run(report: str | Path, candidate: str = "EQ10_P48") -> dict[str, Any]:
    report = Path(report).resolve(strict=True)
    if str(candidate).upper() != "EQ10_P48":
        raise RuntimeError("S46G_EQ10_ONLY")
    frozen_hpc = importlib.import_module("tools.cpu_s4_6f_fullsize_parallel.hpc")

    context = validate_context(report)
    if context.get("diagnostic_kind") == "MEMORY_POLICY_CROSSOVER":
        from .policy_diagnostic import run_policy

        return run_policy(report, candidate, context=context)
    original_supervisor = frozen_hpc._supervise_process
    observations: list[dict[str, Any]] = []
    observer_errors: list[dict[str, str]] = []

    def observed_supervisor(process, *, node, deadline, receipt_path):
        output_dir = Path(receipt_path).parent / "TELEMETRY"
        case = output_dir.parent.name
        observer: ProcessObserver | None = None
        start_error: BaseException | None = None
        observer_failure: str | None = None
        supervisor_result: Any = None
        supervisor_error: BaseException | None = None
        supervisor_traceback = None
        try:
            observer = ProcessObserver(process.pid, output_dir,
                                       period_s=float(context["observer_period_s"]),
                                       numa_every_samples=int(context["numa_every_samples"]))
            observer.start()
        except BaseException as exc:
            start_error = exc
            observer_failure = f"OBSERVER_START:{exc!r}"
            observer_errors.append({"kind": "OBSERVER_START", "message": repr(exc), "case": case})
        try:
            supervisor_result = original_supervisor(process, node=node, deadline=deadline, receipt_path=receipt_path)
        except BaseException as exc:
            supervisor_error = exc
            supervisor_traceback = exc.__traceback__
        finally:
            if observer is not None:
                try:
                    receipt = observer.stop()
                    observations.append({"case": case, "path": str(output_dir), "receipt": receipt})
                    if not isinstance(receipt, dict) or receipt.get("status") not in {"STOPPED", "PID_EXITED"} or receipt.get("errors"):
                        observer_failure = observer_failure or f"OBSERVER_FAILED:{receipt!r}"
                        observer_errors.append({"kind": "OBSERVER_FAILED", "message": repr(receipt.get("errors") if isinstance(receipt, dict) else receipt), "case": case})
                except BaseException as exc:
                    observer_failure = observer_failure or f"OBSERVER_STOP:{exc!r}"
                    observer_errors.append({"kind": "OBSERVER_STOP", "message": repr(exc), "case": case})
            if start_error is not None:
                observer_failure = observer_failure or f"OBSERVER_START:{start_error!r}"
        if supervisor_error is not None:
            # The frozen supervisor remains authoritative for child failures;
            # observer errors never mask its exception after reaping.
            raise supervisor_error.with_traceback(supervisor_traceback)
        if observer_failure is not None:
            # This is the window boundary: the current child was reaped above,
            # and the frozen run aborts before launching another window.
            raise ObserverWindowFailure(f"{case}:{observer_failure}")
        return supervisor_result

    frozen_hpc._supervise_process = observed_supervisor
    result: Any = None
    controller_error: BaseException | None = None
    controller_traceback: str | None = None
    try:
        result = frozen_hpc.run(report, "EQ10_P48")
    except BaseException as exc:
        controller_error = exc
        controller_traceback = traceback.format_exc()
    finally:
        frozen_hpc._supervise_process = original_supervisor

    science_exact = _scientific_result_complete(result)
    observations_valid = controller_error is None and _valid_observations(observations)
    completion_error: BaseException | None = None
    if controller_error is not None:
        status = "DIAGNOSTIC_OBSERVER_FAILURE" if isinstance(controller_error, ObserverWindowFailure) else "DIAGNOSTIC_CONTROLLER_FAILURE"
    elif isinstance(result, dict) and result.get("status") == "SCIENTIFIC_NONEXACT":
        status = "DIAGNOSTIC_SCIENTIFIC_NONEXACT"
        completion_error = RuntimeError("S46G_SCIENTIFIC_NONEXACT")
    elif not science_exact or not observations_valid or observer_errors:
        status = "DIAGNOSTIC_INCOMPLETE"
        completion_error = RuntimeError("S46G_DIAGNOSTIC_INCOMPLETE")
    else:
        status = "DIAGNOSTIC_COMPLETE"
    payload = _diagnostic_receipt(report, status=status, result=result, observations=observations,
                                  error=controller_error or completion_error, error_traceback=controller_traceback,
                                  observations_valid=observations_valid, science_exact=science_exact)
    payload["observer_errors"] = observer_errors
    payload["context_raw_sha256"] = context["context_raw_sha256"]
    out = report / "DIAGNOSTIC_RESULT.json"
    if out.exists():
        raise RuntimeError("S46G_NON_OVERWRITING_DIAGNOSTIC_RESULT")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if controller_error is not None:
        raise controller_error
    if completion_error is not None:
        raise completion_error
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", default="EQ10_P48", choices=("EQ10_P48",))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    result = preflight(args.report, args.candidate) if args.preflight else run(args.report, args.candidate)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

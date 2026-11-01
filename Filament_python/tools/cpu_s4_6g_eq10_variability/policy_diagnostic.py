"""Private counterbalanced D/I memory-policy driver for the frozen 6F entry."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import sys
import time
import traceback
from typing import Any, Mapping

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv

from .observer import ProcessObserver, parse_allowed_list, parse_numa_policy_maps, parse_status
from .policy_analysis import analyze_policy, policy_plan


WORK_DEADLINE_S = 7080.0


class PolicyEngineeringFailure(RuntimeError):
    """A policy/observer/launch contract failure after child reaping."""


class PolicyScientificFailure(RuntimeError):
    """A frozen entry scientific exactness failure."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _require(ok: bool, reason: str, detail: Any = None) -> None:
    if not ok:
        raise PolicyEngineeringFailure(reason if detail is None else f"{reason}:{detail}")


def _deadline_from_start(started: float) -> float:
    return float(started) + WORK_DEADLINE_S


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    """Refresh one newly-created diagnostic receipt after late evidence is known."""

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        stream.write("\n")


def _validate_plan_binding(report: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    plan_path = report / "POLICY_PLAN.json"
    _require(plan_path.is_file(), "POLICY_PLAN_MISSING")
    expected = policy_plan()
    actual = _read(plan_path)
    _require(actual == expected, "POLICY_PLAN_SOURCE_MISMATCH")
    _require(context.get("policy_plan_raw_sha256") == pv.raw_sha256_file(plan_path), "POLICY_PLAN_RAW_BINDING")
    _require(context.get("diagnostic_kind") == "MEMORY_POLICY_CROSSOVER", "POLICY_DIAGNOSTIC_KIND")
    return expected


def _numactl_probe(expected_mems: set[int]) -> dict[str, Any]:
    path = Path("/usr/bin/numactl")
    _require(path.is_file() and os.access(path, os.X_OK), "POLICY_NUMACTL_MISSING")
    commands = ([str(path), "--show"], [str(path), "--interleave=all", "--show"])
    receipts = []
    for argv in commands:
        try:
            result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PolicyEngineeringFailure("POLICY_NUMACTL_PROBE_FAILED") from exc
        receipts.append(dict(argv=argv, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
        _require(result.returncode == 0, "POLICY_NUMACTL_PROBE_FAILED", result.stderr[-512:])
    return dict(status="PASS_NODE_NUMACTL_PROBE", expected_mems=sorted(expected_mems), commands=receipts)


def _compute_node_policy_probe(expected_mems: set[int]) -> dict[str, Any]:
    """Prove D/I on this allocation with a tiny non-scientific child process."""

    proc_root = Path("/proc")
    _require(proc_root.is_dir(), "POLICY_PROCFS_MISSING")
    try:
        expected_cpus = set(int(value) for value in os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise PolicyEngineeringFailure("POLICY_CPU_AFFINITY_UNAVAILABLE") from exc
    _require(expected_cpus, "POLICY_CPU_AFFINITY_EMPTY")
    numactl = "/usr/bin/numactl"
    child_code = (
        "import mmap,time; m=mmap.mmap(-1,16384); m.write(b'x'*16384); "
        "print('S46G_POLICY_PROBE_READY', flush=True); time.sleep(1.0)"
    )
    records: dict[str, Any] = {}
    for policy, prefix in (("D", []), ("I", [numactl, "--interleave=all"])):
        argv = [*prefix, sys.executable, "-s", "-B", "-c", child_code]
        try:
            child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PolicyEngineeringFailure("POLICY_CHILD_PROBE_START_FAILED:" + policy) from exc
        evidence = None
        try:
            until = time.monotonic() + 10.0
            while time.monotonic() < until and child.poll() is None:
                root = proc_root / str(child.pid)
                try:
                    status_text = (root / "status").read_text(encoding="utf-8", errors="replace")
                    maps_text = (root / "numa_maps").read_text(encoding="utf-8", errors="replace")
                    status = parse_status(status_text, {"Name", "Cpus_allowed_list", "Mems_allowed_list"})
                    parsed = parse_numa_policy_maps(maps_text)
                    cpus = parse_allowed_list(status.get("Cpus_allowed_list"))
                    mems = parse_allowed_list(status.get("Mems_allowed_list"))
                    expected_label = "default" if policy == "D" else "interleave"
                    labels = parsed.get("policy_labels", {}) if parsed.get("status") == "MEASURED" else {}
                    node_sets = parsed.get("policy_node_sets", {}).get(expected_label, []) if labels else []
                    label_ok = (
                        set(labels) == {expected_label}
                        if policy == "D"
                        else expected_label in labels and set(labels).issubset({"default", expected_label})
                    )
                    node_ok = policy == "D" or bool(node_sets) and all(set(nodes) == expected_mems for nodes in node_sets)
                    if cpus == expected_cpus and mems == expected_mems and label_ok and node_ok:
                        evidence = {
                            "policy": policy, "pid": child.pid, "argv": argv,
                            "cpus_allowed": sorted(cpus), "mems_allowed": sorted(mems),
                            "policy_labels": labels, "policy_node_sets": node_sets,
                            "pages_by_node": parsed.get("pages_by_node"),
                        }
                        break
                except (FileNotFoundError, PermissionError, OSError):
                    pass
                time.sleep(0.05)
            stdout, stderr = child.communicate(timeout=5)
        except subprocess.TimeoutExpired as exc:
            child.kill()
            stdout, stderr = child.communicate(timeout=5)
            raise PolicyEngineeringFailure("POLICY_CHILD_PROBE_TIMEOUT:" + policy) from exc
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
        _require(child.returncode == 0 and evidence is not None, "POLICY_CHILD_PROBE_FAILED", {"policy": policy, "returncode": child.returncode, "stdout": stdout[-512:], "stderr": stderr[-512:]})
        records[policy] = evidence
    return {"status": "PASS_NODE_POLICY_PROBE", "expected_mems": sorted(expected_mems), "expected_cpus": sorted(expected_cpus), "records": records}


def _base_argv(report: Path, work: Path, row: Mapping[str, Any], reference: Mapping[str, Any] | None) -> list[str]:
    argv = [
        sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.entry",
        "--report", str(report), "--work", str(work), "--candidate", "EQ10_P48",
        "--role", str(row["role"]), "--intervals", str(row["intervals"]),
        "--input-manifest", str(report / "FULL_INPUT_MANIFEST.json"),
    ]
    if reference:
        if reference.get("output") and (int(row["intervals"]) == 8 or row["role"] == "X"):
            argv.extend(("--reference-output", str(reference["output"])))
        if int(row["intervals"]) == 2:
            for flag, key in (("--reference-boundary", "boundary_path"), ("--reference-stages", "stages")):
                if reference.get(key):
                    argv.extend((flag, str(reference[key])))
    return argv


def _reference(row: Mapping[str, Any], cases: Mapping[str, Any], historical: Mapping[str, Any]) -> Mapping[str, Any] | None:
    case = row["case"]
    if case == "QUAL_R2":
        return {"boundary_path": historical.get("boundary_path"), "stages": historical.get("stages")}
    if case == "QUAL_X2":
        return cases["QUAL_R2"]
    if case == "CTRL_EARLY_R1":
        return historical
    return cases["CTRL_EARLY_R1"]


def _clock_metrics(work: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    raw = _read(work / "RAW_TIMING.json")
    _require(not raw["a"]["enabled"] and not raw["b"]["enabled"], "POLICY_FINE_TIMERS_NOT_OFF")
    _require(raw["merged"]["clock_closure"]["pass"] is True, "POLICY_CLOCK_CLOSURE")
    rows = raw["merged"]["records"]

    def one(label: str) -> Mapping[str, Any]:
        matches = [row for row in rows if row["label"] == label]
        _require(len(matches) == 1 and matches[0]["status"] == "complete" and not matches[0]["failed"], "POLICY_ONE_COMPLETE_CLOCK", label)
        return matches[0]

    propagation = one("PROPAGATION")
    bodies = sorted((row for row in rows if row["label"] == "INTERVAL_BODY"), key=lambda row: row["raw_context"]["interval_index"])
    _require(len(bodies) == int(result["intervals"]), "POLICY_INTERVAL_BODY_COUNT")
    for index, row in enumerate(bodies):
        _require(row["raw_context"]["interval_index"] == index and row["status"] == "complete" and not row["failed"], "POLICY_INTERVAL_BODY_IDENTITY")
        _require(row["merged_parent_token"] == propagation["merged_token"], "POLICY_BODY_PARENT_IDENTITY")
        _require(propagation["start_wall_ns"] <= row["start_wall_ns"] < row["end_wall_ns"] <= propagation["end_wall_ns"], "POLICY_BODY_INSIDE_PROPAGATION")
    body_ns = sum(row["wall_inclusive_ns"] for row in bodies)
    _require(propagation["direct_child_wall_inclusive_ns"] == body_ns, "POLICY_PROPAGATION_BODY_DECOMPOSITION")
    _require(propagation["wall_inclusive_ns"] == body_ns + propagation["wall_exclusive_ns"], "POLICY_PROPAGATION_INCLUSIVE_DECOMPOSITION")
    recorded_body = _read(work / "INTERVAL_CLOCKS.json")["body_s"]
    _require(
        [round(value * 1e9) for value in recorded_body] == [row["wall_inclusive_ns"] for row in bodies]
        and recorded_body == result["body_s"],
        "POLICY_RECORDED_BODY_NANOSECONDS",
    )
    _require(float(result["later_body_s"]) == sum(recorded_body[1:]), "POLICY_MAIN_BODY_DENOMINATOR")
    init = _read(work / "INITIALIZATION_CLOCKS.json")
    pacing = _read(work / "OPTICAL_PACING_SIDECAR.json")
    posts = pacing["POST_COMMITTED"]
    _require(len(posts) == int(result["intervals"]), "POLICY_POST_COUNT")
    app_start = float(init["application_start_monotonic_s"])
    init_total = float(result["initialization_total_before_propagation_s"])
    _require(math.isclose(init_total, propagation["start_wall_ns"] * 1e-9 - app_start, abs_tol=2e-6), "POLICY_TOTAL_INITIALIZATION_CLOCK")
    _require(float(result["first_POST_latency_s"]) >= init_total >= float(result["source_factory_preparation_s"]), "POLICY_INITIALIZATION_ORDER")
    _require(math.isclose(float(result["first_POST_latency_s"]), float(posts[0]["time_s"]) - app_start, abs_tol=2e-6), "POLICY_FIRST_POST_CLOCK")
    post_gaps = [float(posts[index]["time_s"]) - float(posts[index - 1]["time_s"]) for index in range(1, len(posts))]
    _require(math.isclose(float(result["post_gap_median_s"]), statistics.median(post_gaps), abs_tol=2e-6), "POLICY_POST_GAP_CLOCK")
    pool = result["pointwise_pool"]["pool"]
    source_validation = one("SOURCE_VALIDATION")
    factory = one("L1_FACTORY")
    return {
        "propagation_s": propagation["wall_inclusive_ns"] / 1e9,
        "body_all_s": body_ns / 1e9,
        "body_later_s": float(result["later_body_s"]),
        "application_s": float(result["application_service_s"]),
        "first_body_s": float(result["first_body_s"]),
        "first_POST_s": float(result["first_POST_latency_s"]),
        "initialization_total_s": init_total,
        "canonical_setup_s": float(result["canonical_setup_s"]),
        "source_validation_s": source_validation["wall_inclusive_ns"] / 1e9,
        "factory_s": factory["wall_inclusive_ns"] / 1e9,
        "pool_creation_s": float(pool["initialization_ns"]) / 1e9,
        "pool_close_s": float(pool["close_ns"]) / 1e9,
        "propagation_start_wall_ns": propagation["start_wall_ns"],
        "propagation_end_wall_ns": propagation["end_wall_ns"],
        "body_intervals": [{"start_wall_ns": row["start_wall_ns"], "end_wall_ns": row["end_wall_ns"]} for row in bodies],
        "clock_closure": True,
    }


def _policy_body_evidence(telemetry: Path, clock_metrics: Mapping[str, Any]) -> dict[str, Any]:
    samples_path = telemetry / "samples.jsonl"
    samples = []
    if samples_path.is_file():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(json.loads(line))
    valid = [row for row in samples if row.get("policy_snapshot", {}).get("status") == "MEASURED" and row.get("policy_snapshot", {}).get("phase") == "BODY_POLICY"]
    covered = []
    for interval in clock_metrics["body_intervals"]:
        def span_inside(row: Mapping[str, Any]) -> bool:
            start = row.get("collection_start_perf_counter_ns")
            end = row.get("collection_end_perf_counter_ns")
            if type(start) is not int or type(end) is not int or start > end:
                return False
            return interval["start_wall_ns"] <= start <= end <= interval["end_wall_ns"]

        matches = [row for row in valid if span_inside(row)]
        covered.append(bool(matches))
    _require(all(covered), "POLICY_BODY_INTERVAL_SNAPSHOT_MISSING", covered)
    return {
        "status": "PASS_POLICY_BODY_SNAPSHOTS",
        "valid_snapshot_count": len(valid),
        "body_interval_coverage": covered,
        "valid_body_snapshot": True,
        "transition_snapshot_count": sum(1 for row in samples if row.get("policy_snapshot", {}).get("status") == "EXEC_TRANSITION"),
    }


def _launch(
    *, report: Path, row: Mapping[str, Any], reference: Mapping[str, Any] | None,
    cases: Mapping[str, Any], node: Mapping[str, Any], plan: Mapping[str, Any],
    deadline: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case = str(row["case"])
    policy = str(row["policy"])
    _require(time.monotonic() < deadline, "POLICY_DEADLINE_BEFORE_NEXT_WINDOW", case)
    folder = report / "runs" / "S4G_POLICY" / "EQ10_P48" / case
    folder.mkdir(parents=True, exist_ok=False)
    work = folder / "work"
    base_argv = _base_argv(report, work, row, reference)
    prefix = list(plan["policies"][policy]["argv_prefix"])
    argv = [*prefix, *base_argv]
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["NUMBA_CACHE_DIR"] = str(report / "cache" / "POLICY_EQ10_P48" / str(row["role"]))
    launch_start = time.perf_counter_ns()
    sg.write(folder / "LAUNCH.json", dict(argv=argv, base_argv=base_argv, prefix=prefix, case=case, policy=policy, role=row["role"], intervals=row["intervals"], kind=row["kind"], UTC=dt.datetime.now(dt.timezone.utc).isoformat(), warmup_intervals=0, factor_cache="EMPTY_PRIVATE_NEW_PROCESS", JIT_cache=env["NUMBA_CACHE_DIR"]))
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    popen_return = time.perf_counter_ns()
    telemetry = folder / "TELEMETRY"
    observer = None
    observer_start_error = None
    observer_stop_error = None
    observer_receipt = None
    supervisor_error = None
    supervisor_traceback = None
    supervisor = None
    try:
        observer = ProcessObserver(
            process.pid, telemetry, period_s=5.0, numa_every_samples=3,
            expected_policy=policy, expected_mems=set(int(value) for value in plan["expected_mems"]),
            expected_cpus=set(int(value) for value in node["affinity"]),
        )
        observer.start()
    except BaseException as exc:
        observer_start_error = exc
    try:
        import tools.cpu_s4_6f_fullsize_parallel.hpc as frozen_hpc
        supervisor = frozen_hpc._supervise_process(process, node=node, deadline=deadline, receipt_path=folder / "SUPERVISOR_RECEIPT.json")
    except BaseException as exc:
        supervisor_error = exc
        supervisor_traceback = exc.__traceback__
    supervisor_end = time.perf_counter_ns()
    try:
        if observer is not None:
            observer_receipt = observer.stop()
    except BaseException as exc:
        observer_stop_error = exc
    observer_stop = time.perf_counter_ns()
    launch_receipt: dict[str, Any] = {
        "schema": "filament.cpu_s46g.policy_launch_receipt.v1",
        "case": case, "role": row["role"], "intervals": row["intervals"], "policy": policy,
        "base_argv": base_argv, "argv": argv, "prefix": prefix,
        "launch_start_perf_counter_ns": launch_start,
        "popen_return_perf_counter_ns": popen_return,
        "supervisor_end_perf_counter_ns": supervisor_end,
        "observer_stop_perf_counter_ns": observer_stop,
        "launch_scope_s": (observer_stop - launch_start) / 1e9,
        "popen_scope_s": (popen_return - launch_start) / 1e9,
        "post_popen_to_supervisor_end_s": (supervisor_end - popen_return) / 1e9,
        "supervisor_to_observer_stop_s": (observer_stop - supervisor_end) / 1e9,
        "process_service_s": None if not isinstance(supervisor, Mapping) else supervisor.get("process_service_s"),
        "supervisor": supervisor,
        "observer_receipt": observer_receipt,
        "observer_start_error": repr(observer_start_error) if observer_start_error else None,
        "observer_stop_error": repr(observer_stop_error) if observer_stop_error else None,
        "clock_domain": "time.perf_counter_ns; same Linux monotonic domain is recorded by observer",
    }
    sg.write(folder / "POLICY_LAUNCH_RECEIPT.json", launch_receipt)
    if supervisor_error is not None:
        raise supervisor_error.with_traceback(supervisor_traceback)
    if observer_start_error is not None or observer_stop_error is not None:
        raise PolicyEngineeringFailure(f"POLICY_OBSERVER_LIFECYCLE:{case}")
    if not isinstance(observer_receipt, Mapping) or observer_receipt.get("status") not in {"STOPPED", "PID_EXITED"} or observer_receipt.get("errors"):
        raise PolicyEngineeringFailure(f"POLICY_OBSERVER_FAILED:{case}")
    work_failure = work / "ENTRY_FAILURE.json"
    failure = _read(work_failure) if work_failure.is_file() else None
    import tools.cpu_s4_6f_fullsize_parallel.hpc as frozen_hpc
    classification = frozen_hpc._child_failure_classification(supervisor, failure)
    if classification == "SCIENTIFIC_NONEXACT":
        raise PolicyScientificFailure(f"POLICY_SCIENTIFIC_NONEXACT:{case}")
    if classification == "ENGINEERING_OR_RESOURCE_FAILURE":
        raise PolicyEngineeringFailure(f"S46F_STOP_CHILD_FAILURE:{case}")
    case_path = work / "CASE_RECEIPT.json"
    _require(case_path.is_file(), "POLICY_CASE_RECEIPT_MISSING", case)
    result = _read(case_path)
    result.update(case_id=case, kind=row["kind"], supervisor=supervisor)
    try:
        if case == "QUAL_R2":
            frozen_hpc._exact(result, terminal=False, boundary=True, stages=True, candidate="EQ10_P48")
        elif case == "QUAL_X2":
            frozen_hpc._exact(result, terminal=True, boundary=True, stages=True, candidate="EQ10_P48")
        else:
            frozen_hpc._exact(result, terminal=True, boundary=False, stages=False, candidate="EQ10_P48")
    except RuntimeError as exc:
        if any(tag in str(exc) for tag in ("S46F_STOP_TERMINAL_NONEXACT", "S46F_STOP_BOUNDARY_NONEXACT", "S46F_STOP_STAGES_NONEXACT")):
            raise PolicyScientificFailure(f"POLICY_SCIENTIFIC_NONEXACT:{case}") from exc
        raise PolicyEngineeringFailure(f"POLICY_EXACT_CONTRACT:{case}") from exc
    clock_metrics = _clock_metrics(work, result)
    policy_evidence = _policy_body_evidence(telemetry, clock_metrics)
    launch_receipt["clock_metrics"] = dict(clock_metrics, launch_scope_s=launch_receipt["launch_scope_s"], process_service_s=launch_receipt["process_service_s"])
    launch_receipt["policy_evidence"] = policy_evidence
    _replace_json(folder / "POLICY_LAUNCH_RECEIPT.json", launch_receipt)
    return result, launch_receipt


def _diagnostic_payload(report: Path, *, status: str, context: Mapping[str, Any], plan: Mapping[str, Any], accepted: Mapping[str, Any], launch_receipts: Mapping[str, Any], result: Any = None, error: BaseException | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "filament.cpu_s46g.policy_diagnostic_result.v1",
        "status": status,
        "classification": "DIAGNOSTIC_ONLY",
        "diagnostic_kind": "MEMORY_POLICY_CROSSOVER",
        "diagnostic_only": True,
        "candidate": "EQ10_P48",
        "default": "I0",
        "production_qualified": False,
        "context_raw_sha256": context.get("context_raw_sha256"),
        "policy_plan_raw_sha256": context.get("policy_plan_raw_sha256"),
        "policy_plan": plan,
        "accepted_case_count": len(accepted),
        "launch_receipt_count": len(launch_receipts),
        "accepted_cases": accepted,
        "launch_receipts": launch_receipts,
        "result": result,
    }
    if error is not None:
        payload["error"] = repr(error)
        payload["traceback"] = traceback.format_exc()
    return payload


def run_policy(report: str | Path, candidate: str = "EQ10_P48", *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    report = Path(report).resolve(strict=True)
    _require(str(candidate).upper() == "EQ10_P48", "POLICY_EQ10_ONLY")
    from .controller import validate_context

    context = validate_context(report) if context is None else dict(context)
    plan = _validate_plan_binding(report, context)
    import tools.cpu_s4_6f_fullsize_parallel.hpc as frozen_hpc
    started = time.monotonic()
    deadline = _deadline_from_start(started)
    accepted: dict[str, Any] = {}
    launch_receipts: dict[str, Any] = {}
    result: Any = None
    status = "DIAGNOSTIC_COMPLETE"
    error: BaseException | None = None
    try:
        node = frozen_hpc.node_gate(report, "EQ10_P48")
        expected_mems = set(int(value) for value in plan["expected_mems"])
        _require(set(int(value) for value in node.get("numa_nodes", ())) == expected_mems, "POLICY_NODE_NUMA_SET_MISMATCH")
        capability = _numactl_probe(expected_mems)
        capability["child_policy_probe"] = _compute_node_policy_probe(expected_mems)
        historical_contract = _read(report / "REFERENCE_INPUT_RAW_RECORDS.json")
        historical = historical_contract.get("reference", historical_contract)
        manifest = _read(report / "FULL_INPUT_MANIFEST.json")
        contract = _read(report / "INPUT_CONTRACT.json")
        _require(manifest.get("shape") == [384, 351, 301] and manifest.get("fingerprints") == contract.get("expected_input_fingerprints"), "POLICY_FROZEN_FULLSIZE_INPUTS")
        for row in plan["matrix"]:
            reference = _reference(row, accepted, historical)
            case_result, launch = _launch(report=report, row=row, reference=reference, cases=accepted, node=node, plan=plan, deadline=deadline)
            accepted[row["case"]] = case_result
            launch["numactl_capability"] = capability
            launch_receipts[row["case"]] = launch
            _replace_json(report / "runs" / "S4G_POLICY" / "EQ10_P48" / row["case"] / "POLICY_LAUNCH_RECEIPT.json", launch)
            sg.write(report / "runs" / "S4G_POLICY" / "EQ10_P48" / row["case"] / "ACCEPTED_CASE.json", case_result)
        result = analyze_policy(accepted, launch_receipts)
        result["node_gate_status"] = node.get("status")
        result["resource_service_s"] = time.monotonic() - started
        sg.write(report / "POLICY_ACCEPTED_CASES.json", accepted)
        sg.write(report / "POLICY_LAUNCH_RECEIPTS.json", launch_receipts)
        sg.write(report / "POLICY_RESULT.json", result)
    except BaseException as exc:
        error = exc
        status = "DIAGNOSTIC_SCIENTIFIC_NONEXACT" if isinstance(exc, PolicyScientificFailure) else "DIAGNOSTIC_POLICY_FAILURE"
        if isinstance(exc, PolicyScientificFailure):
            status = "DIAGNOSTIC_SCIENTIFIC_NONEXACT"
        payload = _diagnostic_payload(report, status=status, context=context, plan=plan, accepted=accepted, launch_receipts=launch_receipts, result=result, error=error)
        out = report / "DIAGNOSTIC_RESULT.json"
        if not out.exists():
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        raise
    payload = _diagnostic_payload(report, status=status, context=context, plan=plan, accepted=accepted, launch_receipts=launch_receipts, result=result)
    out = report / "DIAGNOSTIC_RESULT.json"
    _require(not out.exists(), "POLICY_DIAGNOSTIC_RESULT_PREEXISTS")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", default="EQ10_P48", choices=("EQ10_P48",))
    args = parser.parse_args(argv)
    print(json.dumps(run_policy(args.report, args.candidate), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

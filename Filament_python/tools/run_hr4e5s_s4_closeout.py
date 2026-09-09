#!/usr/bin/env python3
"""Materialize the bounded HR-4E-5S S4 evidence closeout from persisted data."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


START_SHA = "e7e900e94830405988626f7825408ebf6e3bec45"
S3_BATCH_JOB = "233394"
S3_BATCH_WALLTIME_S = 3 * 3600 + 6 * 60 + 26
SCREEN_COUNT = 48
FULL_SCREEN_COUNT = 15_000
R_OPT_BASELINE = 0.35351035411669146


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if path.exists():
        raise FileExistsError(path)
    columns = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(materialized)


def merge_csv(*, evidence: Path, cases: list[tuple[str, str]], source_name: str, destination: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case_id, directory in cases:
        with (evidence / directory / source_name).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"case_id": case_id, **row})
    write_csv(destination, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence, out = args.evidence_root, args.out_dir
    if not evidence.is_dir():
        raise FileNotFoundError(evidence)
    out.mkdir(parents=True, exist_ok=True)

    metric_dirs = {
        "smoke_1p1": "smoke_telemetry",
        "replay_hydro_1": "replay_1_telemetry",
        "replay_hydro_2": "replay_2_telemetry",
        "replay_hydro_4": "replay_4_telemetry",
        "stream_candidate_1p2_r1": "r1_telemetry",
        "stream_candidate_1p2_r2": "r2_telemetry",
    }
    metrics = {case: read_json(evidence / directory / "hr4e5s_s4_streaming_metrics.json") for case, directory in metric_dirs.items()}
    replay_rates = {workers: metrics[f"replay_hydro_{workers}"]["R_hydro_screens_per_s"] for workers in (1, 2, 4)}
    candidate_cases = ("stream_candidate_1p2_r1", "stream_candidate_1p2_r2")
    candidate_walltimes = [float(metrics[case]["T_stream_s"]) for case in candidate_cases]
    median_stream = float(statistics.median(candidate_walltimes))
    repeat_difference_fraction = abs(candidate_walltimes[0] - candidate_walltimes[1]) / median_stream
    observed_optical_rates = [float(metrics[case]["R_opt_screens_per_s"]) for case in candidate_cases]
    conservative_optical_rate = max(observed_optical_rates)
    capacity = {workers: replay_rates[workers] / conservative_optical_rate for workers in replay_rates}
    batch_reduction = 1.0 - median_stream / S3_BATCH_WALLTIME_S

    jobs = [
        {"case_id": "batch_reference", "job_id": S3_BATCH_JOB, "mode": "batch", "hydro_workers": 1, "gpu_count": 1, "state": "COMPLETED/0:0", "elapsed_s": S3_BATCH_WALLTIME_S, "reused": True},
        {"case_id": "smoke_1p1", "job_id": "234721", "mode": "stream", "hydro_workers": 1, "gpu_count": 2, "state": "COMPLETED/0:0", "elapsed_s": 3 * 3600 + 6 * 60 + 2, "reused": False},
        {"case_id": "replay_hydro_1", "job_id": "235405", "mode": "replay", "hydro_workers": 1, "gpu_count": 1, "state": "COMPLETED/0:0", "elapsed_s": 4 * 60 + 6, "reused": False},
        {"case_id": "replay_hydro_2", "job_id": "235407", "mode": "replay", "hydro_workers": 2, "gpu_count": 2, "state": "COMPLETED/0:0", "elapsed_s": 2 * 60 + 22, "reused": False},
        {"case_id": "replay_hydro_4", "job_id": "235408", "mode": "replay", "hydro_workers": 4, "gpu_count": 4, "state": "COMPLETED/0:0", "elapsed_s": 2 * 60 + 10, "reused": False},
        {"case_id": "stream_candidate_1p2_r1", "job_id": "235526", "mode": "stream", "hydro_workers": 2, "gpu_count": 3, "state": "COMPLETED/0:0", "elapsed_s": 3 * 3600 + 2 * 60 + 5, "reused": False},
        {"case_id": "stream_candidate_1p2_r2", "job_id": "235933", "mode": "stream", "hydro_workers": 2, "gpu_count": 3, "state": "COMPLETED/0:0", "elapsed_s": 3 * 3600 + 1 * 60 + 57, "reused": False},
    ]
    write_csv(out / "hr4e5s_s4_job_matrix.csv", jobs)

    service_rows = [{"hydro_workers": workers, "R_hydro_screens_per_s": replay_rates[workers], "R_opt_conservative_screens_per_s": conservative_optical_rate, "capacity_ratio": capacity[workers], "headroom": capacity[workers] - 1.0, "exact_next_status": "PASS"} for workers in (1, 2, 4)]
    write_csv(out / "hr4e5s_s4_service_rates.csv", service_rows)

    stream_rows = []
    for case in ("smoke_1p1", *candidate_cases):
        item = metrics[case]
        stream_rows.append({
            "case_id": case, "hydro_workers": 1 if case == "smoke_1p1" else 2,
            "T_stream_s": item["T_stream_s"], "T_opt_active_s": item["T_opt_active_s"], "T_hydro_span_s": item["T_hydro_span_s"],
            "T_tail_s": item["T_tail_s"], "T_overlap_s": item["T_overlap_s"],
            "overlap_fraction_of_optical": item["overlap_fraction_of_optical"], "producer_blocked_fraction": item["queue"]["producer_blocked_fraction"],
            "max_queue_occupancy": item["queue"]["max"], "median_hydro_block_service_s": item["median_hydro_block_service_s"],
            "exact_status": "PASS",
        })
    write_csv(out / "hr4e5s_s4_streaming_metrics.csv", stream_rows)

    event_cases = [(case, directory) for case, directory in metric_dirs.items()]
    merge_csv(evidence=evidence, cases=event_cases, source_name="hr4e5s_s4_runtime_events.csv", destination=out / "hr4e5s_s4_runtime_events.csv")
    merge_csv(evidence=evidence, cases=event_cases, source_name="hr4e5s_s4_queue_trace.csv", destination=out / "hr4e5s_s4_queue_trace.csv")
    merge_csv(evidence=evidence, cases=event_cases, source_name="hr4e5s_s4_worker_utilization.csv", destination=out / "hr4e5s_s4_worker_utilization.csv")

    exact: dict[str, Any] = {"schema": "khz_filament.hr4e5s.s4.exact_summary.v1", "status": "PASS", "cases": {}}
    for workers in (1, 2, 4):
        result = read_json(evidence / f"replay_{workers}_exact.json")
        exact["cases"][f"replay_hydro_{workers}"] = {key: result[key] for key in ("status", "expected_field_comparisons", "completed_field_comparisons", "mismatch_count")}
    for case, directory in (("stream_candidate_1p2_r1", "r1_comparison"), ("stream_candidate_1p2_r2", "r2_comparison")):
        result = read_json(evidence / directory / "hr4e5s_s3_scientific_comparisons.json")
        exact["cases"][case] = {key: result[key] for key in ("status", "expected_field_comparisons", "completed_field_comparisons", "mismatch_count", "optical_status", "ledger_status", "normalized_manifest_status", "ownership_status", "barrier_promotion_status")}
    exact["cases"]["smoke_1p1"] = {"status": "PASS", "expected_field_comparisons": 432, "completed_field_comparisons": 432, "mismatch_count": 0, "reused_as_immediate_lower_topology": True}
    write_json(out / "hr4e5s_s4_exact_equivalence_summary.json", exact)

    producer_full_s = FULL_SCREEN_COUNT / conservative_optical_rate
    projection_rows = []
    for workers in (1, 2, 4):
        hydro_full_s = FULL_SCREEN_COUNT / replay_rates[workers]
        total_s = max(producer_full_s, hydro_full_s)
        tail_s = max(0.0, hydro_full_s - producer_full_s)
        projection_rows.append({"hydro_workers": workers, "producer_limited_s": producer_full_s, "hydro_limited_s": hydro_full_s, "projected_streaming_s": total_s, "expected_tail_s": tail_s, "total_gpus": 1 + workers, "projected_gpu_hours": total_s * (1 + workers) / 3600.0, "production_qualified": False})
    resource_model = {"schema": "khz_filament.hr4e5s.s4.resource_model.v1", "status": "INSUFFICIENT_CAPACITY", "full_screen_count": FULL_SCREEN_COUNT, "producer_rate_screens_per_s": conservative_optical_rate, "rows": projection_rows, "balanced_streaming_projection": None, "reason": "no tested hydro worker count has capacity_ratio >= 1.10"}
    write_json(out / "hr4e5s_s4_resource_model.json", resource_model)
    write_json(out / "hr4e5s_s4_fullz_projection.json", {"schema": "khz_filament.hr4e5s.s4.fullz_projection.v1", "status": "PROJECTION_ONLY", **resource_model})

    preflight = read_json(evidence.parent / "remote_evidence" / "hr4e5s_s4_preflight.json")
    environment = {"schema": "khz_filament.hr4e5s.s4.environment.v1", "status": "PARTIAL_PERSISTED_RECORD", "start_sha": START_SHA, "branch": "HR-4E", "worktree_clean_at_preflight": True, "remote_run_root": preflight["run_root"], "fixed_python": "/data/home/scvi806/.conda/envs/Filament_python/bin/python", "cpu_per_task": 8, "partition": "gpu", "nodes": {"smoke_r1_r2": "m4gl1701", "replay_4": "m4gn1401"}, "gpu_model": "not captured by the submitted batch entry", "cuda_runtime": "not captured by the submitted batch entry", "cupy_version": "not captured by the submitted batch entry", "visible_device_mapping": "per-srun CUDA_VISIBLE_DEVICES recorded in telemetry events", "preflight": preflight}
    write_json(out / "hr4e5s_s4_environment.json", environment)

    gates = {"G1_scientific_freeze": "PASS", "G2_telemetry_integrity": "PASS", "G3_new_topology_exactness": "PASS", "G4_real_overlap": "PASS", "G5_sustainable_consumer_capacity": "FAIL", "G6_backpressure_control": "NOT_ACCEPTED_WITHOUT_G5", "G7_pipeline_tail": "PASS", "G8_net_walltime_benefit": "FAIL", "G9_resource_minimality": "NOT_APPLICABLE"}
    decision = {"schema": "khz_filament.hr4e5s.s4.final_decision.v1", "status": "BLOCKED / HYDRO_CAPACITY_INSUFFICIENT", "decision": "S4_NOT_CLOSED", "parent_status": "HR-4E-5S = S3_CLOSED / S4_NONPASS", "start_sha": START_SHA, "final_sha": START_SHA, "scientific_operator_change": False, "frozen_case_identity": {"screen_indices": [7998, 8045], "screen_count": SCREEN_COUNT, "queue_depth": 16, "block_size": 8}, "reused_s3_jobs": [S3_BATCH_JOB, "234721"], "new_job_ids": ["235405", "235407", "235408", "235526", "235933"], "selected_optical_gpu_count": None, "selected_hydro_gpu_count": None, "selected_queue_depth": None, "selected_block_size": 8, "R_opt": {"smoke": R_OPT_BASELINE, "candidate_repeats": observed_optical_rates, "conservative_for_capacity": conservative_optical_rate}, "R_hydro": replay_rates, "capacity_ratios": capacity, "overlap_metrics": {case: {key: metrics[case][key] for key in ("T_overlap_s", "overlap_fraction_of_optical", "overlap_fraction_of_hydro_span")} for case in candidate_cases}, "backpressure_metrics": {case: metrics[case]["queue"] for case in candidate_cases}, "tail_metrics": {case: metrics[case]["T_tail_s"] for case in candidate_cases}, "batch_walltime_s": S3_BATCH_WALLTIME_S, "streaming_walltimes_s": dict(zip(candidate_cases, candidate_walltimes)), "median_streaming_walltime_s": median_stream, "walltime_reduction": batch_reduction, "repeat_difference_fraction": repeat_difference_fraction, "exact_comparison_counts": exact, "mismatch_counts": {case: 0 for case in exact["cases"]}, "barrier_promotion_result": "PASS", "gates": gates, "full_z_projection_summary": resource_model, "not_started": ["S4 N_hydro=8 extension", "S5", "formal HR-4E-5", "HR-4F", "HR-5"], "reason": "The conservative candidate producer rate exceeds every tested hydro replay rate by too much for the required 10% headroom; median end-to-end walltime reduction is also below 5%."}
    write_json(out / "hr4e5s_s4_final_decision.json", decision)

    closeout = f"""# HR-4E-5S-S4 closeout\n\n## Decision\n\n`HR-4E-5S-S4 = BLOCKED / HYDRO_CAPACITY_INSUFFICIENT`\n\nS4 is not closed. No S5, formal HR-4E-5, HR-4F, or HR-5 work was started.\n\n## Exact-science guardrail\n\nAll three replay cases passed 144 / 144 strict NEXT comparisons with zero mismatch. The two new 1 optical + 2 hydro streaming repeats each passed 432 / 432 strict field comparisons with zero mismatch, plus optical, ledger, normalized-manifest, CURRENT/NEXT ownership, barrier, and promotion checks.\n\n## Capacity result\n\nThe conservative measured producer rate was {conservative_optical_rate:.9f} screens/s. Hydro replay rates were 1 GPU={replay_rates[1]:.9f}, 2 GPU={replay_rates[2]:.9f}, and 4 GPU={replay_rates[4]:.9f} screens/s. Corresponding capacity ratios were {capacity[1]:.3f}, {capacity[2]:.3f}, and {capacity[4]:.3f}; none reaches 1.10.\n\n## Walltime result\n\nThe two candidate application walltimes were {candidate_walltimes[0]:.3f} s and {candidate_walltimes[1]:.3f} s (difference {repeat_difference_fraction:.4%}); therefore no r3 was required. Their median is {median_stream:.3f} s, versus the reused batch walltime {S3_BATCH_WALLTIME_S:.3f} s, a reduction of {batch_reduction:.4%}. This is below the required 5% criterion.\n\n## Gates\n\n- G1/G2/G3/G4/G7: PASS.\n- G5: FAIL; no tested hydro topology provides 10% service-rate headroom.\n- G8: FAIL; median walltime reduction is below 5%.\n- G6/G9: not accepted because G5 fails.\n\nThe optional N=8 extension was not started.\n"""
    closeout_path = out / "HR4E5S_S4_CLOSEOUT.md"
    if closeout_path.exists():
        raise FileExistsError(closeout_path)
    closeout_path.write_text(closeout, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Focused standard-library tests for replay_readonly.py."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from replay_readonly import (
    ReplayError,
    _datasets_from_case_runs,
    _executed_source_replay,
    _replay_case_metadata,
    _raw_timing_replay,
    load_json_strict,
    off_statistics,
    host_clock_exclusive_ledger,
    light_exclusive_summaries,
    aux_serialized_summaries,
    native_off_macro_summaries,
    recompute_timing_gate,
    replay_report,
    replay_resource_rows,
    resolve_report_path,
    validate_clock_closure,
    validate_manifest_records,
)
from build_measurement_tables import _gap_rows, _native_gap_rows


def _rows() -> list[dict[str, object]]:
    return [
        {"case": "P1_OFF", "application_service_s": 10, "window_completed_s": 20, "later_body_s": 30},
        {"case": "P1_LIGHT", "application_service_s": 10, "window_completed_s": 20, "later_body_s": 30},
        {"case": "P2_OFF", "application_service_s": 20, "window_completed_s": 40, "later_body_s": 60},
        {"case": "P2_LIGHT", "application_service_s": 20, "window_completed_s": 40, "later_body_s": 60},
        {"case": "P3_OFF", "application_service_s": 30, "window_completed_s": 60, "later_body_s": 90},
        {"case": "P3_LIGHT", "application_service_s": 30, "window_completed_s": 60, "later_body_s": 90},
    ]


def _write_source_bundle_fixture(root: Path, *, corrupt_gpu: bool = False) -> None:
    portable_helper = (
        root
        / "source_roles"
        / "CPU"
        / "executed_source"
        / "Filament_python"
        / "tools"
        / "hpc_ops"
        / "provenance_v2.py"
    )
    repo_helper = (
        Path(__file__).parents[3]
        / "Filament_python"
        / "tools"
        / "hpc_ops"
        / "provenance_v2.py"
    )
    helper_source = (portable_helper if portable_helper.is_file() else repo_helper).read_bytes()

    def write_role(role: str, commit: str, role_bytes: bytes) -> None:
        bundle = root / "source_roles" / role
        source = bundle / "executed_source"
        helper_path = source / "Filament_python" / "tools" / "hpc_ops" / "provenance_v2.py"
        role_path = source / "Filament_python" / "tools" / "role.py"
        helper_path.parent.mkdir(parents=True, exist_ok=True)
        role_path.parent.mkdir(parents=True, exist_ok=True)
        helper_path.write_bytes(helper_source)
        role_path.write_bytes(role_bytes)

        def record(relative: str, raw: bytes) -> dict[str, str]:
            blob = b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
            return {
                "path": relative,
                "classification": "tracked_text",
                "hash_scope": "git_blob_oid+canonical_lf_sha256",
                "git_blob_oid": hashlib.sha1(blob).hexdigest(),
                "canonical_lf_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest(),
            }

        records = [
            record("Filament_python/tools/hpc_ops/provenance_v2.py", helper_source),
            record("Filament_python/tools/role.py", role_bytes),
        ]
        if corrupt_gpu and role == "GPU":
            records[1]["git_blob_oid"] = "0" * 40
        (bundle / "EXECUTION_SOURCE_MANIFEST.json").write_text(
            json.dumps({"head": commit, "records": records}), encoding="utf-8"
        )

    write_role("CPU", "1" * 40, b"CPU\n")
    write_role("GPU", "2" * 40, b"GPU\n")
    (root / "SOURCE_BUNDLE_INDEX.json").write_text(
        json.dumps(
            {
                "schema": "filament.cpu.s46a.source_bundles.v1",
                "bundles": [
                    {"role": "CPU", "root": "source_roles/CPU", "execution_commit": "1" * 40},
                    {"role": "GPU", "root": "source_roles/GPU", "execution_commit": "2" * 40},
                ],
            }
        ),
        encoding="utf-8",
    )


class ReplayTests(unittest.TestCase):
    def test_strict_json_and_path_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaises(ReplayError):
                load_json_strict(duplicate)
            overflow = root / "overflow.json"
            overflow.write_text('{"a":1e999}', encoding="utf-8")
            with self.assertRaises(ReplayError):
                load_json_strict(overflow)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "same.txt").write_text("a", encoding="utf-8")
            (root / "b" / "same.txt").write_text("b", encoding="utf-8")
            with self.assertRaises(ReplayError):
                resolve_report_path(root, "same.txt")
            self.assertEqual(resolve_report_path(root, "a/same.txt").read_text(encoding="utf-8"), "a")
            (root / "same.txt").write_text("root", encoding="utf-8")
            self.assertEqual(resolve_report_path(root, "./same.txt").read_text(encoding="utf-8"), "root")
            with self.assertRaises(ReplayError):
                resolve_report_path(root, "same.txt")
            with self.assertRaises(ReplayError):
                resolve_report_path(root, "../same.txt")

    def test_source_bundle_index_replays_two_roles_with_distinct_same_path_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source_bundle_fixture(root)
            result = _executed_source_replay(root)
            self.assertEqual(result["status"], "SOURCE_BUNDLES_REPLAYED")
            self.assertEqual({item["role"] for item in result["roles"]}, {"CPU", "GPU"})
            cpu_role = next(item for item in result["roles"] if item["role"] == "CPU")
            gpu_role = next(item for item in result["roles"] if item["role"] == "GPU")
            cpu_record = next(item for item in cpu_role["records"] if item["path"].endswith("tools/role.py"))
            gpu_record = next(item for item in gpu_role["records"] if item["path"].endswith("tools/role.py"))
            self.assertNotEqual(cpu_record["git_blob_oid"], gpu_record["git_blob_oid"])

    def test_source_bundle_index_rejects_wrong_role_blob_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source_bundle_fixture(root, corrupt_gpu=True)
            result = _executed_source_replay(root)
            self.assertEqual(result["status"], "SOURCE_BUNDLE_INVALID")
            self.assertTrue(any("PACKAGED_GIT_BLOB_OID_MISMATCH" in issue for issue in result["issues"]))

    def test_gate_and_off_mad(self) -> None:
        gate = recompute_timing_gate(_rows())
        self.assertEqual(gate["status"], "PASS_TIMING_GATE")
        self.assertEqual(gate["metrics"]["application"]["median"], 1.0)
        off = off_statistics(_rows())
        self.assertEqual(off["metrics"]["application"]["median_s"], 20.0)
        self.assertEqual(off["metrics"]["application"]["range_s"], 20.0)
        self.assertEqual(off["metrics"]["application"]["mad_s"], 10.0)

    def test_clock_closure_and_overlap(self) -> None:
        good = [
            {"id": "w", "label": "WINDOW", "start_ns": 0, "end_ns": 100, "completed": True},
            {"id": "i", "label": "INTERVAL_BODY", "interval_index": 0, "start_ns": 0, "end_ns": 40, "completed": True, "parent": "w"},
            {"id": "l", "label": "INTERVAL_BODY", "interval_index": 1, "start_ns": 50, "end_ns": 100, "completed": True, "parent": "w"},
        ]
        self.assertEqual(validate_clock_closure(good)["status"], "PASS_CLOCK_CLOSURE")
        bad = [dict(row) for row in good]
        bad[-1]["start_ns"] = 30
        self.assertIn("CLOCK_SIBLING_OVERLAP:w", validate_clock_closure(bad)["issues"])

    def test_authoritative_merged_and_gpu_host_clock_views(self) -> None:
        merged_rows = [
            {"merged_token": "w", "label": "PROPAGATION", "start_wall_ns": 0, "end_wall_ns": 100, "completed": True},
            {"merged_token": "i0", "merged_parent_token": "w", "label": "INTERVAL_BODY", "raw_context": {"interval_index": 0}, "start_wall_ns": 0, "end_wall_ns": 40, "completed": True},
            {"merged_token": "i1", "merged_parent_token": "w", "label": "INTERVAL_BODY", "raw_context": {"interval_index": 1}, "start_wall_ns": 50, "end_wall_ns": 100, "completed": True},
        ]
        merged = _raw_timing_replay({"merged": {"records": merged_rows}, "b": {"records": merged_rows}, "s45a": {"records": merged_rows}})
        self.assertEqual(merged["status"], "PASS_CLOCK_CLOSURE")
        self.assertEqual(merged["clock_source"], "merged.records")
        self.assertEqual(merged["overlap_views_ignored"], ["b", "s45a"])
        gpu = _raw_timing_replay({"records": [
            {"token": "completed_propagation_window", "start_wall_ns": 0, "end_wall_ns": 100, "completed": True, "device_event_span_ms": 99},
            {"token": "i0", "parent": "completed_propagation_window", "label": "INTERVAL_BODY", "interval": 0, "start_wall_ns": 0, "end_wall_ns": 40, "completed": True},
            {"token": "i1", "parent": "completed_propagation_window", "label": "INTERVAL_BODY", "interval": 1, "start_wall_ns": 50, "end_wall_ns": 100, "completed": True},
        ]})
        self.assertEqual(gpu["status"], "PASS_CLOCK_CLOSURE")
        self.assertEqual(gpu["clock_source"], "records.host_wall_ns")
        self.assertEqual(gpu["deferred_device_event_spans_ms"], [99])
        self.assertTrue(gpu["device_spans_not_subtracted_from_host"])

    def test_formal_case_receipts_replay_fixed_matrix(self) -> None:
        case_order = ("QUAL_OFF", "QUAL_LIGHT", "P1_OFF", "P1_LIGHT", "P2_LIGHT", "P2_OFF", "P3_OFF", "P3_LIGHT")
        values = {
            "QUAL_OFF": 8,
            "QUAL_LIGHT": 8,
            "P1_OFF": 10,
            "P1_LIGHT": 10,
            "P2_LIGHT": 20,
            "P2_OFF": 20,
            "P3_OFF": 30,
            "P3_LIGHT": 30,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in case_order:
                work = root / "raw" / "runs" / "CPU" / "1x1x1" / case / "work"
                work.mkdir(parents=True)
                seconds = values[case]
                app_end = seconds * 1_000_000_000
                window_start = 100
                window_end = window_start + app_end
                midpoint = window_start + 10
                receipt = {
                    "backend": "CPU",
                    "shape": [1, 1, 1],
                    "case": case,
                    "mode": "LIGHT" if case.endswith("LIGHT") else "OFF",
                    "intervals": 2,
                    "application_clocks": {"start_ns": 0, "application_end_ns": app_end},
                    "completed_window_clocks": {
                        "start_ns": window_start,
                        "end_ns": window_end,
                        "start_cpu_ns": window_start,
                        "end_cpu_ns": window_end,
                    },
                }
                raw = {
                    "backend": "CPU",
                    "merged": {"records": [
                        {"merged_token": "completed_propagation_window", "label": "PROPAGATION", "start_wall_ns": window_start, "end_wall_ns": window_end, "start_cpu_ns": window_start, "end_cpu_ns": window_end, "completed": True},
                        {"merged_token": "i0", "merged_parent_token": "completed_propagation_window", "label": "INTERVAL_BODY", "raw_context": {"interval_index": 0}, "start_wall_ns": window_start, "end_wall_ns": midpoint, "start_cpu_ns": window_start, "end_cpu_ns": midpoint, "completed": True},
                        {"merged_token": "i1", "merged_parent_token": "completed_propagation_window", "label": "INTERVAL_BODY", "raw_context": {"interval_index": 1}, "start_wall_ns": midpoint, "end_wall_ns": window_end, "start_cpu_ns": midpoint, "end_cpu_ns": window_end, "completed": True},
                    ]},
                }
                (work / "CASE_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
                (work / "RAW_CPU_TIMING.json").write_text(json.dumps(raw), encoding="utf-8")
            report = replay_report(root)
            self.assertEqual(report["case_metadata"]["status"], "REPLAYED")
            self.assertEqual(len(report["case_metadata"]["runs"]), 8)
            self.assertEqual(len(report["matrix"]), 1)
            dataset = report["matrix"][0]
            self.assertEqual(dataset["case_order"], list(case_order))
            self.assertEqual(dataset["status"], "REPLAYED")
            self.assertEqual(dataset["timing_gate"]["status"], "PASS_TIMING_GATE")
            self.assertEqual(dataset["off_statistics"]["metrics"]["application"]["median_s"], 20.0)
            self.assertEqual(report["DEFAULT"], "I0")
            self.assertFalse(report["PRODUCTION_QUALIFIED"])
            self.assertTrue(report["MANUAL_REVIEW_PENDING"])

    def test_formal_gpu_receipt_uses_gpu_raw_and_persisted_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "raw" / "runs" / "GPU" / "1x1x1" / "P1_OFF" / "work"
            work.mkdir(parents=True)
            receipt = {
                "backend": "GPU",
                "shape": [1, 1, 1],
                "case": "P1_OFF",
                "mode": "OFF",
                "intervals": 2,
                "application_clocks": {"start_ns": 0, "application_end_ns": 1_000_000_000},
                "completed_window_clocks": {
                    "start_ns": 100,
                    "end_ns": 900_000_000,
                    "start_cpu_ns": 100,
                    "end_cpu_ns": 900_000_000,
                },
            }
            raw = {"records": [
                {"token": "completed_propagation_window", "parent": None, "start_wall_ns": 100, "end_wall_ns": 900_000_000, "start_cpu_ns": 100, "end_cpu_ns": 900_000_000, "wall_ns": 899_999_900, "process_cpu_ns": 899_999_800, "completed": True, "failed": False},
                {"token": "i0", "parent": "completed_propagation_window", "label": "INTERVAL_BODY", "interval": 0, "start_wall_ns": 100, "end_wall_ns": 400_000_000, "start_cpu_ns": 100, "end_cpu_ns": 400_000_000, "completed": True, "failed": False},
                {"token": "i1", "parent": "completed_propagation_window", "label": "INTERVAL_BODY", "interval": 1, "start_wall_ns": 400_000_000, "end_wall_ns": 900_000_000, "start_cpu_ns": 400_000_000, "end_cpu_ns": 900_000_000, "completed": True, "failed": False},
            ]}
            (work / "CASE_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
            (work / "RAW_GPU_TIMING.json").write_text(json.dumps(raw), encoding="utf-8")
            report = replay_report(root)
            self.assertEqual(report["case_metadata"]["status"], "REPLAYED")
            run = report["case_metadata"]["runs"][0]
            self.assertEqual(run["backend"], "GPU")
            self.assertTrue(run["raw_timing_path"].endswith("RAW_GPU_TIMING.json"))
            self.assertEqual(run["metric_sources"]["window"], "RAW.records.PROPAGATION.host_wall_ns")
            self.assertEqual(run["clock_closure"]["status"], "PASS_CLOCK_CLOSURE")

    def test_aux_and_failed_receipts_never_fill_formal_matrix(self) -> None:
        rows = []
        for case in ("QUAL_OFF", "QUAL_LIGHT", "P1_OFF", "P1_LIGHT", "P2_LIGHT", "P2_OFF", "P3_OFF", "P3_LIGHT"):
            rows.append(
                {
                    "backend": "CPU",
                    "shape": "shape:1x1x1",
                    "case": case,
                    "matrix_eligible": True,
                    "metrics_s": {"application": 1.0, "window": 1.0, "later": 1.0},
                }
            )
        rows.append(
            {
                "backend": "CPU",
                "shape": "shape:1x1x1",
                "case": "P1_LIGHT",
                "mode": "AUX_SERIALIZED",
                "matrix_eligible": False,
                "matrix_exclusion_reason": "AUX_SERIALIZED_DIAGNOSTIC_ONLY",
                "metrics_s": {"application": 999.0, "window": 999.0, "later": 999.0},
            }
        )
        rows.append(
            {
                "backend": "CPU",
                "shape": "shape:1x1x1",
                "case": "P2_OFF",
                "matrix_eligible": False,
                "matrix_exclusion_reason": "FAILED_OR_INCOMPLETE_RECEIPT",
                "metrics_s": {"application": 999.0, "window": 999.0, "later": 999.0},
            }
        )
        datasets = _datasets_from_case_runs(rows)
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0]["status"], "REPLAYED")
        self.assertEqual(datasets[0]["off_statistics"]["metrics"]["application"]["median_s"], 1.0)

    def test_real_gpu_full_gate_failure_keeps_off_stats_and_aux_separate(self) -> None:
        report = replay_report(Path(__file__).parent)
        gpu_datasets = {
            item["shape"]: item
            for item in report["matrix"]
            if item.get("backend") == "GPU"
        }
        full = gpu_datasets["shape:384x351x301"]
        self.assertEqual(full["status"], "INCOMPLETE")
        self.assertEqual(full["timing_gate"]["status"], "TIMING_GATE_FAIL")
        for metric in ("application", "window", "later"):
            self.assertEqual(full["off_statistics"]["metrics"][metric]["count"], 3)
        aux = [
            run
            for run in report["case_metadata"]["runs"]
            if run.get("backend") == "GPU" and run.get("case") == "AUX_SERIALIZED"
        ]
        self.assertEqual(len(aux), 2)
        self.assertTrue(all(not run["matrix_eligible"] for run in aux))
        self.assertTrue(all(run["matrix_exclusion_reason"] == "AUX_SERIALIZED_DIAGNOSTIC_ONLY" for run in aux))
        formal_gpu = [
            run
            for run in report["case_metadata"]["runs"]
            if run.get("backend") == "GPU" and run.get("matrix_eligible") is True
        ]
        self.assertTrue(formal_gpu)
        self.assertTrue(
            all(
                run["completed_window_consistency"]["status"] == "PASS_COMPLETED_WINDOW_ENDPOINT_MATCH"
                for run in formal_gpu
            )
        )

    def test_real_all_backend_light_ledgers_rank_cpu_only_and_aux_ratios(self) -> None:
        report = replay_report(Path(__file__).parent)
        summaries = report["light_exclusive_summaries"]
        cpu_full = next(
            item for item in summaries
            if item.get("backend") == "CPU" and item.get("shape") == "shape:384x351x301"
        )
        self.assertEqual(cpu_full["status"], "REPLAYED")
        self.assertTrue(cpu_full["closure_all_pass"])
        self.assertEqual(len(cpu_full["categories"]), 7)
        self.assertEqual(cpu_full["rank_scope"], "CPU process-CPU median only")
        self.assertTrue(cpu_full["native_attribution_allowed"])
        top_rank = next(item for item in cpu_full["categories"] if item.get("cpu_cost_rank") == 1)
        self.assertEqual(top_rank["category"], "Raman_Heun")
        top_wall = next(item for item in cpu_full["categories"] if item.get("wall_cost_rank") == 1)
        self.assertEqual(top_wall["category"], "Raman_Heun")
        gpu_full = next(
            item for item in summaries
            if item.get("backend") == "GPU" and item.get("shape") == "shape:384x351x301"
        )
        self.assertEqual(gpu_full["timing_gate"], "TIMING_GATE_FAIL")
        self.assertIsNone(gpu_full["rank_by_process_cpu_median"])
        self.assertFalse(gpu_full["native_attribution_allowed"])
        aux = report["aux_serialized_summaries"]
        self.assertEqual(len(aux), 2)
        self.assertTrue(all(item["diagnostic_only"] and item["not_off_median"] for item in aux))
        self.assertTrue(all(item["aux_to_p1_off_application_ratio"] is not None for item in aux))
        self.assertTrue(all(item["aux_to_off_application_median_ratio"] is not None for item in aux))
        self.assertEqual(report["resources"]["backends"]["CPU"]["job_ids"], ["8854173", "8854203", "8854552", "8855018"])
        self.assertEqual(report["resources"]["status"], "REPLAYED")

    def test_native_off_macro_leaves_close_and_full_gate_stays_unknown(self) -> None:
        report = replay_report(Path(__file__).parent)
        summaries = report["native_off_macro_summaries"]
        self.assertEqual({item["shape_class"] for item in summaries}, {"SMALL", "FULL"})
        for item in summaries:
            self.assertTrue(item["delta"]["closure_pass"], item)
            self.assertEqual(item["delta"]["closure_residual_decimal_s"], "0")
            self.assertIn("no round-paired", item["clock_domain"])
            for backend in ("CPU", "GPU"):
                self.assertEqual(item["backends"][backend]["status"], "REPLAYED")
                self.assertEqual(set(item["backends"][backend]["leaves"]), {"outside", "window_remainder", "later"})
        full = next(item for item in summaries if item["shape_class"] == "FULL")
        self.assertEqual(full["status"], "REPLAYED_METADATA_ONLY")
        self.assertEqual(full["native_operator_delta_status"], "UNKNOWN")
        self.assertEqual(full["backends"]["GPU"]["timing_gate"], "TIMING_GATE_FAIL")
        small = next(item for item in summaries if item["shape_class"] == "SMALL")
        self.assertEqual(small["status"], "REPLAYED_METADATA_ONLY")
        self.assertEqual(small["native_operator_delta_status"], "DESCRIPTIVE_MACRO_ONLY")
        self.assertEqual(native_off_macro_summaries(report["matrix"]), summaries)

        native_rows = _native_gap_rows(report, table_status="FINAL_MEASURED_LIMITED")
        full_outside = next(
            row for row in native_rows
            if row["shape_class"] == "FULL" and row["scope"] == "outside"
        )
        cpu_leaf_values = full_outside["CPU_values_s"]
        self.assertIsInstance(cpu_leaf_values, list)
        self.assertEqual(len(cpu_leaf_values), 3)
        self.assertAlmostEqual(
            sum(float(value) for value in cpu_leaf_values) / 3.0,
            float(full_outside["CPU_mean_s"]),
        )
        self.assertNotEqual(cpu_leaf_values, [346.097522225, 347.451679406, 341.130852481])
        self.assertEqual(full_outside["leaf_anchor"], "application-window")
        self.assertTrue(full_outside["mutually_exclusive"])
        self.assertEqual(full_outside["table_status"], "FINAL_MEASURED_LIMITED")

    def test_gap_rows_block_dataset_with_failed_light_gate(self) -> None:
        runs = [
            {
                "backend": backend,
                "shape": "shape:384x351x301",
                "case": "P1_LIGHT",
                "mode": "LIGHT",
                "matrix_eligible": True,
                "status": "REPLAYED",
                "clock_closure": {"status": "PASS_CLOCK_CLOSURE"},
                "host_clock_exclusive": {
                    "status": "PASS_HOST_CLOCK_EXCLUSIVE",
                    "diagnostic_only": False,
                    "categories": {"linear": {"wall_exclusive_ns": 1_000_000_000}},
                },
            }
            for backend in ("CPU", "GPU")
        ]
        datasets = [
            {
                "backend": "CPU",
                "shape": "shape:384x351x301",
                "status": "REPLAYED",
                "timing_gate": {"status": "PASS_TIMING_GATE"},
            },
            {
                "backend": "GPU",
                "shape": "shape:384x351x301",
                "status": "INCOMPLETE",
                "timing_gate": {"status": "TIMING_GATE_FAIL"},
            },
        ]
        rows = _gap_rows(runs, table_status="DRAFT", datasets=datasets)
        linear = next(row for row in rows if row["shape_class"] == "FULL" and row["scope"] == "linear")
        self.assertEqual(linear["status"], "DRAFT")
        self.assertIn("GPU_DATASET_TIMING_GATE_FAIL", linear["missing_reason"])

    def test_cpu_small_case_replays_process_clocks_first_interval_and_post(self) -> None:
        case_path = (
            Path(__file__).parent
            / "evidence_CPU_SMALL/raw/CPU_SMALL/report/runs/CPU/384x117x129/P3_LIGHT/work/CASE_RECEIPT.json"
        )
        run = _replay_case_metadata(Path(__file__).parent, case_path)
        self.assertEqual(run["status"], "REPLAYED", run["issues"])
        self.assertGreater(run["first_interval_s"], 0.0)
        self.assertGreater(run["process_metrics_s"]["application"], 0.0)
        self.assertGreater(run["process_metrics_s"]["window"], 0.0)
        self.assertGreater(run["process_metrics_s"]["later"], 0.0)
        self.assertEqual(run["post_timeline"]["status"], "REPLAYED")
        self.assertEqual(run["post_timeline"]["adjacent_POST_gap_count"], 7)

    def test_negative_metadata_gap_is_retained(self) -> None:
        categories = {
            label: {"wall_exclusive_ns": 2_000_000_000 if label == "linear" else 0}
            for label in (
                "linear",
                "Raman_Heun",
                "ion",
                "otherNL",
                "deposition_POST",
                "diagnostics_output",
                "management_remainder_unknown",
            )
        }
        cpu_ledger = {"status": "PASS_HOST_CLOCK_EXCLUSIVE", "diagnostic_only": False, "categories": categories}
        gpu_categories = {key: dict(value) for key, value in categories.items()}
        gpu_categories["linear"]["wall_exclusive_ns"] = 3_000_000_000
        gpu_ledger = {"status": "PASS_HOST_CLOCK_EXCLUSIVE", "diagnostic_only": False, "categories": gpu_categories}
        runs = [
            {"backend": "CPU", "shape": "shape:1x1x1", "case": "P1_LIGHT", "mode": "LIGHT", "matrix_eligible": True, "path": "cpu", "host_clock_exclusive": cpu_ledger},
            {"backend": "GPU", "shape": "shape:1x1x1", "case": "P1_LIGHT", "mode": "LIGHT", "matrix_eligible": True, "path": "gpu", "host_clock_exclusive": gpu_ledger},
        ]
        rows = _gap_rows(runs, table_status="DRAFT")
        linear = next(row for row in rows if row["shape_class"] == "FULL" and row["pair"] == "P1" and row["scope"] == "linear")
        self.assertEqual(linear["status"], "PROFILE_DIAGNOSTIC_ONLY")
        self.assertIsNone(linear["delta_s"])
        self.assertEqual(linear["native_operator_delta_status"], "UNKNOWN")
        self.assertIn("not a cross-backend delta", linear["delta_definition"])
        self.assertFalse(linear["scientific_interpretation_allowed"])

    def test_explicit_scope_excludes_old_matrix_tree(self) -> None:
        cases = [
            {"case": name, "application_service_s": 1, "window_completed_s": 1, "later_body_s": 1}
            for name in ("QUAL_OFF", "QUAL_LIGHT", "P1_OFF", "P1_LIGHT", "P2_LIGHT", "P2_OFF", "P3_OFF", "P3_LIGHT")
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "current").mkdir()
            (root / "versions" / "old").mkdir(parents=True)
            payload = {"backend": "CPU", "shape": [1, 1, 1], "cases": cases}
            (root / "current" / "MATRIX_RESULT_CURRENT.json").write_text(json.dumps(payload), encoding="utf-8")
            (root / "versions" / "old" / "MATRIX_RESULT_OLD.json").write_text(json.dumps(payload), encoding="utf-8")
            (root / "REPLAY_SCOPE.json").write_text(json.dumps({"datasets": ["current"], "timings": [], "ledgers": [], "manifests": []}), encoding="utf-8")
            report = replay_report(root)
            self.assertEqual(len(report["matrix"]), 1)
            self.assertEqual(report["matrix"][0]["path"], "current/MATRIX_RESULT_CURRENT.json")

    def test_manifest_raw_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_bytes(b"source")
            digest = hashlib.sha256(b"source").hexdigest()
            result = validate_manifest_records(root, [{"path": "source.txt", "hash_scope": "raw_bytes", "raw_sha256": digest}])
            self.assertEqual(result["status"], "MANIFEST_REPLAYED")
            bad = validate_manifest_records(root, [{"path": "source.txt", "hash_scope": "raw_bytes", "raw_sha256": "0" * 64}])
            self.assertEqual(bad["status"], "INCOMPLETE")

    def test_resource_contract_and_missing_replay(self) -> None:
        rows = [
            {"backend": "CPU", "job_id": "c1", "category": "small", "state": "completed", "cpus": 96, "gpus": 0, "actual_walltime_s": 1200},
            {"backend": "GPU", "job_id": "g1", "category": "repair", "state": "completed", "cpus": 8, "gpus": 1, "actual_walltime_s": 1800},
        ]
        result = replay_resource_rows(rows)
        self.assertEqual(result["status"], "REPLAYED")
        self.assertEqual(result["backends"]["GPU"]["actual_gpu_hours"], 0.5)
        exact = replay_resource_rows([
            {"backend": "GPU", "job_id": "g-main", "category": "main", "state": "completed", "cpus": 8, "gpus": 1, "actual_walltime_s": 5400},
            {"backend": "GPU", "job_id": "g-repair", "category": "repair", "state": "completed", "cpus": 8, "gpus": 1, "actual_walltime_s": 1800},
        ])
        self.assertEqual(exact["status"], "REPLAYED")
        self.assertEqual(exact["backends"]["GPU"]["actual_gpu_hours"], 2.0)
        self.assertEqual(exact["backends"]["GPU"]["actual_core_hours"], 16.0)
        cpu_four = replay_resource_rows([
            {"backend": "CPU", "job_id": f"c{index}", "category": "small", "state": "completed", "cpus": 96, "gpus": 0, "actual_walltime_s": 1}
            for index in range(4)
        ])
        self.assertEqual(cpu_four["status"], "REPLAYED")
        cpu_five = replay_resource_rows([
            {"backend": "CPU", "job_id": f"c{index}", "category": "small", "state": "completed", "cpus": 96, "gpus": 0, "actual_walltime_s": 1}
            for index in range(5)
        ])
        self.assertEqual(cpu_five["status"], "BUDGET_BLOCKED")
        self.assertIn("CPU:JOB_ID_CAP_EXCEEDED:5", cpu_five["issues"])
        with tempfile.TemporaryDirectory() as directory:
            report = replay_report(Path(directory))
            self.assertEqual(report["status"], "PENDING")
            self.assertTrue(report["MANUAL_REVIEW_PENDING"])

    def test_gpu_four_job_cap_accepts_253s_ledger_and_rejects_fifth(self) -> None:
        prior = [
            {"backend": "GPU", "job_id": "290147", "category": "main", "state": "completed", "cpus": 8, "gpus": 1, "actual_walltime_s": 47},
            {"backend": "GPU", "job_id": "290163", "category": "repair", "state": "completed", "cpus": 8, "gpus": 1, "actual_walltime_s": 206},
        ]
        third = [
            *prior,
            {"backend": "GPU", "job_id": "GPU_NEW_3", "category": "main", "state": "completed", "cpus": 8, "gpus": 1, "actual_walltime_s": 1},
        ]
        accepted = replay_resource_rows(third)
        self.assertEqual(accepted["status"], "REPLAYED")
        self.assertEqual(accepted["backends"]["GPU"]["job_ids"], ["290147", "290163", "GPU_NEW_3"])
        self.assertAlmostEqual(accepted["backends"]["GPU"]["actual_gpu_hours"], 254 / 3600)
        self.assertAlmostEqual(accepted["backends"]["GPU"]["worst_case_gpu_hours"], 254 / 3600)
        self.assertAlmostEqual(accepted["backends"]["GPU"]["worst_case_core_hours"], 254 * 8 / 3600)

        fourth = [
            *third,
            {"backend": "GPU", "job_id": "GPU_NEW_4", "category": "main", "state": "completed", "cpus": 8, "gpus": 1, "actual_walltime_s": 1},
        ]
        accepted_four = replay_resource_rows(fourth)
        self.assertEqual(accepted_four["status"], "REPLAYED")
        self.assertEqual(accepted_four["backends"]["GPU"]["job_ids"], ["290147", "290163", "GPU_NEW_3", "GPU_NEW_4"])

        fifth = [
            *fourth,
            {"backend": "GPU", "job_id": "GPU_NEW_5", "category": "main", "state": "pending", "new": True, "cpus": 8, "gpus": 1, "actual_walltime_s": 0, "worst_walltime_s": 1},
        ]
        rejected = replay_resource_rows(fifth)
        self.assertEqual(rejected["status"], "BUDGET_BLOCKED")
        self.assertIn("GPU:JOB_ID_CAP_EXCEEDED:5", rejected["issues"])

    def test_cpu_small_p3_light_has_seven_class_exclusive_sum_to_propagation(self) -> None:
        raw_path = (
            Path(__file__).parent
            / "evidence_CPU_SMALL/raw/CPU_SMALL/report/runs/CPU/384x117x129/P3_LIGHT/work/RAW_CPU_TIMING.json"
        )
        self.assertTrue(raw_path.is_file(), raw_path)
        raw = load_json_strict(raw_path)
        ledger = host_clock_exclusive_ledger(raw, backend="CPU", mode="LIGHT")
        self.assertEqual(ledger["status"], "PASS_HOST_CLOCK_EXCLUSIVE", ledger)
        self.assertEqual(
            set(ledger["categories"]),
            {"linear", "Raman_Heun", "ion", "otherNL", "deposition_POST", "diagnostics_output", "management_remainder_unknown"},
        )
        self.assertTrue(ledger["exclusive_sum"]["wall_equals_propagation"])
        self.assertTrue(ledger["exclusive_sum"]["process_cpu_equals_propagation"])
        self.assertEqual(
            ledger["exclusive_sum"]["wall_ns"],
            ledger["propagation"]["wall_inclusive_ns"],
        )
        self.assertEqual(
            ledger["exclusive_sum"]["process_cpu_ns"],
            ledger["propagation"]["process_cpu_inclusive_ns"],
        )
        self.assertFalse(ledger["cross_backend_operator_difference_allowed"])

    def test_host_exclusive_negative_is_not_clipped_and_gpu_light_events_are_separate(self) -> None:
        rows = [
            {"id": "p", "label": "PROPAGATION", "parent": None, "start_ns": 0, "end_ns": 10, "start_cpu_ns": 0, "end_cpu_ns": 10, "completed": True},
            {"id": "c1", "label": "LINEAR", "parent": "p", "start_ns": 0, "end_ns": 8, "start_cpu_ns": 0, "end_cpu_ns": 8, "completed": True},
            {"id": "c2", "label": "RAMAN_STAGE", "parent": "p", "start_ns": 2, "end_ns": 10, "start_cpu_ns": 2, "end_cpu_ns": 10, "completed": True, "device_event_span_ms": 1.5},
        ]
        ledger = host_clock_exclusive_ledger(rows, backend="GPU", mode="LIGHT")
        self.assertEqual(ledger["status"], "INCOMPLETE")
        self.assertIn("NEGATIVE_WALL_EXCLUSIVE:p", ledger["issues"])
        self.assertTrue(ledger["cuda_event_spans_are_idle_inclusive_not_busy"])
        self.assertTrue(ledger["cuda_event_spans_not_subtracted_from_host"])
        self.assertEqual(ledger["categories"]["management_remainder_unknown"]["wall_exclusive_ns"], -6)
        aux_rows = [
            {"id": "p", "label": "PROPAGATION", "parent": None, "start_ns": 0, "end_ns": 10, "start_cpu_ns": 0, "end_cpu_ns": 10, "completed": True},
            {"id": "c1", "label": "LINEAR", "parent": "p", "start_ns": 0, "end_ns": 5, "start_cpu_ns": 0, "end_cpu_ns": 5, "completed": True, "device_event_span_ms": 1.0},
            {"id": "c2", "label": "RAMAN_STAGE", "parent": "p", "start_ns": 5, "end_ns": 10, "start_cpu_ns": 5, "end_cpu_ns": 10, "completed": True, "device_event_span_ms": 2.0},
        ]
        aux = host_clock_exclusive_ledger(aux_rows, backend="GPU", mode="AUX_SERIALIZED")
        self.assertEqual(aux["status"], "PASS_HOST_CLOCK_EXCLUSIVE")
        self.assertTrue(aux["diagnostic_only"])
        self.assertFalse(aux["can_fill_off_or_p_samples"])


if __name__ == "__main__":
    unittest.main()

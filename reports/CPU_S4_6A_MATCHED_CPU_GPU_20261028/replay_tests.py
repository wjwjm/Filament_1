"""Focused standard-library tests for replay_readonly.py."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from replay_readonly import (
    ReplayError,
    _raw_timing_replay,
    load_json_strict,
    off_statistics,
    recompute_timing_gate,
    replay_report,
    replay_resource_rows,
    resolve_report_path,
    validate_clock_closure,
    validate_manifest_records,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"case": "P1_OFF", "application_service_s": 10, "window_completed_s": 20, "later_body_s": 30},
        {"case": "P1_LIGHT", "application_service_s": 10, "window_completed_s": 20, "later_body_s": 30},
        {"case": "P2_OFF", "application_service_s": 20, "window_completed_s": 40, "later_body_s": 60},
        {"case": "P2_LIGHT", "application_service_s": 20, "window_completed_s": 40, "later_body_s": 60},
        {"case": "P3_OFF", "application_service_s": 30, "window_completed_s": 60, "later_body_s": 90},
        {"case": "P3_LIGHT", "application_service_s": 30, "window_completed_s": 60, "later_body_s": 90},
    ]


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
            {"token": "w", "label": "PROPAGATION", "start_wall_ns": 0, "end_wall_ns": 100, "completed": True, "device_event_span_ms": 99},
            {"token": "i0", "parent": "w", "label": "INTERVAL_BODY", "interval": 0, "start_wall_ns": 0, "end_wall_ns": 40, "completed": True},
            {"token": "i1", "parent": "w", "label": "INTERVAL_BODY", "interval": 1, "start_wall_ns": 50, "end_wall_ns": 100, "completed": True},
        ]})
        self.assertEqual(gpu["status"], "PASS_CLOCK_CLOSURE")
        self.assertEqual(gpu["clock_source"], "records.host_wall_ns")
        self.assertEqual(gpu["deferred_device_event_spans_ms"], [99])
        self.assertTrue(gpu["device_spans_not_subtracted_from_host"])

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
        with tempfile.TemporaryDirectory() as directory:
            report = replay_report(Path(directory))
            self.assertEqual(report["status"], "PENDING")
            self.assertTrue(report["MANUAL_REVIEW_PENDING"])


if __name__ == "__main__":
    unittest.main()

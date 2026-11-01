"""Private-driver lifecycle tests with a fake child and no propagation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import policy_diagnostic as driver
from .policy_analysis import policy_plan


class _FakeProcess:
    pid = 9173


def _case_receipt() -> dict:
    return {
        "intervals": 8,
        "later_body_s": 10.0,
        "application_service_s": 12.0,
        "body_s": [1.0] * 8,
    }


class PolicyDriverTests(unittest.TestCase):
    def _run_launch(self, *, policy: str, fail: bool = False):
        plan = policy_plan()
        row = next(item for item in plan["matrix"] if item["case"] == ("K1_X2" if policy == "I" else "K1_X1"))
        captured = []

        def fake_popen(argv, **kwargs):
            captured.append(list(argv))
            return _FakeProcess()

        def fake_supervise(process, *, node, deadline, receipt_path):
            work = Path(receipt_path).parent / "work"
            work.mkdir(parents=True, exist_ok=True)
            (work / "CASE_RECEIPT.json").write_text(json.dumps(_case_receipt()), encoding="utf-8")
            if fail:
                raise ValueError("frozen supervisor failure")
            return {"exit_code": 0, "stop_reason": None, "process_service_s": 3.0}

        observer = mock.MagicMock()
        observer.start.return_value = observer
        observer.stop.return_value = {"status": "STOPPED", "errors": []}

        with tempfile.TemporaryDirectory() as root:
            report = Path(root)
            hpc = __import__("tools.cpu_s4_6f_fullsize_parallel.hpc", fromlist=["*"])
            with mock.patch.object(driver.subprocess, "Popen", side_effect=fake_popen), mock.patch.object(
                driver, "ProcessObserver", return_value=observer
            ), mock.patch.object(hpc, "_supervise_process", side_effect=fake_supervise), mock.patch.object(
                hpc, "_child_failure_classification", return_value=None
            ), mock.patch.object(hpc, "_exact"), mock.patch.object(
                driver, "_clock_metrics", return_value={"propagation_s": 5.0, "body_all_s": 8.0, "body_later_s": 7.0, "application_s": 12.0, "body_intervals": []}
            ), mock.patch.object(driver, "_policy_body_evidence", return_value={"valid_body_snapshot": True}):
                if fail:
                    with self.assertRaisesRegex(ValueError, "frozen supervisor failure"):
                        driver._launch(
                            report=report, row=row, reference=None, cases={}, node={"affinity": [0, 1]},
                            plan=plan, deadline=driver.time.monotonic() + 60,
                        )
                else:
                    result, receipt = driver._launch(
                        report=report, row=row, reference=None, cases={}, node={"affinity": [0, 1]},
                        plan=plan, deadline=driver.time.monotonic() + 60,
                    )
                    self.assertEqual(result["case_id"], row["case"])
                    self.assertEqual(receipt["policy"], policy)
                    receipt_path = report / "runs" / "S4G_POLICY" / "EQ10_P48" / row["case"] / "POLICY_LAUNCH_RECEIPT.json"
                    self.assertTrue(receipt_path.is_file())
                    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
                    self.assertIn("popen_return_perf_counter_ns", saved)
                    self.assertIn("clock_metrics", saved)
            self.assertEqual(observer.start.call_count, 1)
            self.assertEqual(observer.stop.call_count, 1)
            prefix = plan["policies"][policy]["argv_prefix"]
            if prefix:
                self.assertEqual(captured[0][:len(prefix)], prefix)
            else:
                self.assertNotEqual(captured[0][0], "/usr/bin/numactl")

    def test_launch_records_anchors_and_policy_prefix(self):
        self._run_launch(policy="I")

    def test_supervisor_exception_is_transparent_and_observer_stops(self):
        self._run_launch(policy="D", fail=True)

    def test_clock_metrics_keeps_propagation_scope_and_later_body_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            work = Path(root)
            bodies = []
            for index in range(8):
                start = 100_000_000 + index * 40_000_000
                bodies.append({
                    "label": "INTERVAL_BODY", "raw_context": {"interval_index": index},
                    "merged_parent_token": 1, "merged_token": index + 2,
                    "start_wall_ns": start, "end_wall_ns": start + 40_000_000,
                    "wall_inclusive_ns": 40_000_000, "wall_exclusive_ns": 40_000_000,
                    "status": "complete", "failed": False,
                })
            records = [{
                "label": "PROPAGATION", "merged_token": 1,
                "start_wall_ns": 100_000_000, "end_wall_ns": 500_000_000,
                "wall_inclusive_ns": 400_000_000, "wall_exclusive_ns": 80_000_000,
                "direct_child_wall_inclusive_ns": 320_000_000,
                "status": "complete", "failed": False,
            }, *bodies, {
                "label": "SOURCE_VALIDATION", "wall_inclusive_ns": 10,
                "status": "complete", "failed": False,
            }, {
                "label": "L1_FACTORY", "wall_inclusive_ns": 20,
                "status": "complete", "failed": False,
            }]
            (work / "RAW_TIMING.json").write_text(json.dumps({
                "a": {"enabled": False}, "b": {"enabled": False},
                "merged": {"clock_closure": {"pass": True}, "records": records},
            }), encoding="utf-8")
            (work / "INTERVAL_CLOCKS.json").write_text(json.dumps({"body_s": [0.04] * 8}), encoding="utf-8")
            (work / "INITIALIZATION_CLOCKS.json").write_text(json.dumps({
                "application_start_monotonic_s": 0.0999999,
            }), encoding="utf-8")
            (work / "OPTICAL_PACING_SIDECAR.json").write_text(json.dumps({
                "POST_COMMITTED": [{"time_s": 0.0999999 + 1e-6 + index * 1e-6} for index in range(8)],
            }), encoding="utf-8")
            result = {
                "intervals": 8, "body_s": [0.04] * 8, "later_body_s": 0.28,
                "application_service_s": 1.0, "first_body_s": 0.04,
                "initialization_total_before_propagation_s": 1e-7,
                "source_factory_preparation_s": 1e-8,
                "first_POST_latency_s": 1e-6, "post_gap_median_s": 1e-6,
                "canonical_setup_s": 0.0,
                "pointwise_pool": {"pool": {"initialization_ns": 4, "close_ns": 5}},
            }
            clocks = driver._clock_metrics(work, result)
        self.assertEqual(clocks["propagation_s"], 0.4)
        self.assertEqual(clocks["body_all_s"], 0.32)
        self.assertEqual(clocks["body_later_s"], 0.28)

    def test_policy_snapshot_span_crossing_body_end_does_not_cover(self):
        with tempfile.TemporaryDirectory() as root:
            telemetry = Path(root)
            (telemetry / "samples.jsonl").write_text(json.dumps({
                "policy_snapshot": {"status": "MEASURED", "phase": "BODY_POLICY"},
                "collection_start_perf_counter_ns": 150,
                "collection_end_perf_counter_ns": 250,
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(driver.PolicyEngineeringFailure, "POLICY_BODY_INTERVAL_SNAPSHOT_MISSING"):
                driver._policy_body_evidence(
                    telemetry,
                    {"body_intervals": [{"start_wall_ns": 100, "end_wall_ns": 200}]},
                )

    def test_deadline_is_anchored_before_slow_node_gate(self):
        plan = policy_plan()
        deadlines = []
        now = [100.0]

        def fake_gate(report, candidate):
            now[0] = 9000.0
            return {"numa_nodes": [0, 1, 2, 3], "status": "PASS"}

        def fake_launch(*, row, deadline, **kwargs):
            deadlines.append(deadline)
            return (
                {"role": row["role"], "intervals": row["intervals"], "exact_gate_pass": True},
                {"policy": row["policy"], "policy_evidence": {"valid_body_snapshot": True}, "clock_metrics": {"propagation_s": 1.0}, "launch_scope_s": 1.0, "process_service_s": 1.0},
            )

        with tempfile.TemporaryDirectory() as root:
            report = Path(root)
            (report / "REFERENCE_INPUT_RAW_RECORDS.json").write_text(json.dumps({"reference": {}}), encoding="utf-8")
            (report / "FULL_INPUT_MANIFEST.json").write_text(json.dumps({"shape": [384, 351, 301], "fingerprints": {"x": "y"}}), encoding="utf-8")
            (report / "INPUT_CONTRACT.json").write_text(json.dumps({"expected_input_fingerprints": {"x": "y"}}), encoding="utf-8")
            context = {"diagnostic_kind": "MEMORY_POLICY_CROSSOVER", "context_raw_sha256": "context"}
            # Patch the frozen module after import; no child is started.
            hpc = __import__("tools.cpu_s4_6f_fullsize_parallel.hpc", fromlist=["*"])
            with mock.patch.object(driver.time, "monotonic", side_effect=lambda: now[0]), mock.patch.object(
                driver, "_validate_plan_binding", return_value=plan
            ), mock.patch.object(hpc, "node_gate", side_effect=fake_gate), mock.patch.object(
                driver, "_numactl_probe", return_value={"status": "PASS"}
            ), mock.patch.object(driver, "_compute_node_policy_probe", return_value={"status": "PASS"}), mock.patch.object(
                driver, "_launch", side_effect=fake_launch
            ), mock.patch.object(driver, "analyze_policy", return_value={"classification": "NO_DETERMINATE_POLICY_SIGNAL"}), mock.patch.object(
                driver, "_replace_json"
            ), mock.patch.object(driver.sg, "write"):
                driver.run_policy(report, context=context)
        self.assertEqual(deadlines, [100.0 + driver.WORK_DEADLINE_S] * len(plan["matrix"]))
        self.assertGreater(now[0], deadlines[0])


if __name__ == "__main__":
    unittest.main()

"""Controller tests use a fake supervisor and never start propagation."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from . import DIAGNOSTIC_SCHEMA, DIAGNOSTIC_TASK
from . import controller


def _context(commit="abc"):
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "task": DIAGNOSTIC_TASK,
        "candidate": "EQ10_P48",
        "diagnostic_only": True,
        "execution_commit": commit,
        "observer_period_s": 5.0,
        "numa_every_samples": 3,
        "science_arrays_retained": False,
        "old_results_superseded": False,
        "observer_thread_cap": 512,
        "sample_cap": 1450,
        "output_cap_bytes": 64 * 1024 * 1024,
    }


def _write_context_report(root, *, commit="abc"):
    report = Path(root)
    context_path = report / "DIAGNOSTIC_CONTEXT.json"
    context_path.write_text(json.dumps(_context(commit), sort_keys=True), encoding="utf-8")
    context_sha = hashlib.sha256(context_path.read_bytes()).hexdigest()
    (report / "EXECUTION.json").write_text(json.dumps({
        "execution_commit": commit,
        "diagnostic_context_raw_sha256": context_sha,
        "source_manifest_raw_sha256": "source",
    }), encoding="utf-8")
    return report, context_sha


class ContextTests(unittest.TestCase):
    def test_context_rejects_candidate_or_mode_change(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "DIAGNOSTIC_CONTEXT.json"
            value = _context()
            value["candidate"] = "INTENSITY_P24"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "candidate"):
                controller.validate_context(root, execution_commit="abc")

    def test_context_accepts_bound_contract_without_hpc(self):
        with tempfile.TemporaryDirectory() as root:
            report, context_sha = _write_context_report(root)
            with mock.patch.dict(os.environ, {"S46G_CONTEXT_SHA256": context_sha}, clear=False), mock.patch.object(controller, "_validate_source_binding"):
                self.assertEqual(controller.validate_context(report, execution_commit="abc")["task"], DIAGNOSTIC_TASK)

    def test_context_requires_environment_sha(self):
        with tempfile.TemporaryDirectory() as root:
            report, _ = _write_context_report(root)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(controller, "_validate_source_binding"):
                with self.assertRaisesRegex(RuntimeError, "RAW_SHA256_MISSING"):
                    controller.validate_context(report, execution_commit="abc")

    def test_execution_context_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            report, context_sha = _write_context_report(root)
            execution = json.loads((report / "EXECUTION.json").read_text(encoding="utf-8"))
            execution["diagnostic_context_raw_sha256"] = "0" * 64
            (report / "EXECUTION.json").write_text(json.dumps(execution), encoding="utf-8")
            with mock.patch.dict(os.environ, {"S46G_CONTEXT_SHA256": context_sha}, clear=False), mock.patch.object(controller, "_validate_source_binding"):
                with self.assertRaisesRegex(RuntimeError, "EXECUTION_CONTEXT_SHA256"):
                    controller.validate_context(report, execution_commit="abc")


class WrapperTests(unittest.TestCase):
    def test_zero_sample_window_is_incomplete(self):
        receipt = dict(schema=controller.TELEMETRY_SCHEMA + ".receipt.v1", status="PID_EXITED", errors=[],
            samples=1, bytes=1, science_arrays_retained=False, short_lived_children_observed=False,
            last_periodic_sample_only=True, end_state_snapshot=False, max_samples=1450, max_threads=512,
            max_output_bytes=64*1024*1024, collection_wall_ns=1, clock={}, units={})
        rows = [dict(case=case, receipt=dict(receipt)) for case in controller._expected_cases()]
        self.assertTrue(controller._valid_observations(rows))
        rows[0]["receipt"]["samples"] = 0
        self.assertFalse(controller._valid_observations(rows))
        rows[0]["receipt"]["samples"] = 1
        rows[0]["receipt"]["bytes"] = 0
        self.assertFalse(controller._valid_observations(rows))

    def test_fake_supervisor_result_and_exception_are_preserved(self):
        class FakeProcess:
            pid = 123

        class FakeHpc:
            def __init__(self, fail=False):
                self.fail = fail
                self._supervise_process = self.supervise
                self.restored = False

            def supervise(self, process, *, node, deadline, receipt_path):
                if self.fail:
                    raise ValueError("child failure")
                return {"exit_code": 0, "receipt_path": str(receipt_path)}

            def run(self, report, candidate):
                return self.module._supervise_process(FakeProcess(), node={}, deadline=0, receipt_path=Path(report) / "case" / "SUPERVISOR_RECEIPT.json")

        # The production wrapper is exercised indirectly by checking its
        # source-level contract; observer construction is patched to avoid a
        # real PID and the fake run is used only for lifecycle propagation.
        context = _context()
        with tempfile.TemporaryDirectory() as root:
            report, context_sha = _write_context_report(root)
            fake = FakeHpc()
            observer = mock.MagicMock()
            observer.start.return_value = observer
            observer.stop.return_value = {"status": "STOPPED", "errors": []}
            fake_module = types.SimpleNamespace(_supervise_process=fake.supervise, run=fake.run)
            fake.module = fake_module
            real_import = controller.importlib.import_module
            def import_module(name):
                if name == "tools.cpu_s4_6f_fullsize_parallel.hpc":
                    return fake_module
                return real_import(name)
            with mock.patch.object(controller.importlib, "import_module", side_effect=import_module), mock.patch.object(
                controller, "ProcessObserver", return_value=observer
            ), mock.patch.object(controller, "pv") as pv, mock.patch.object(controller, "_validate_source_binding"):
                pv.raw_sha256_file.return_value = context_sha
                with mock.patch.dict(os.environ, {"S46G_CONTEXT_SHA256": context_sha}, clear=False), self.assertRaisesRegex(RuntimeError, "DIAGNOSTIC_INCOMPLETE"):
                    controller.run(report)
            result = json.loads((report / "DIAGNOSTIC_RESULT.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "DIAGNOSTIC_INCOMPLETE")
            self.assertTrue(observer.start.called)
            self.assertTrue(observer.stop.called)
            self.assertIs(fake_module._supervise_process.__func__, fake.supervise.__func__)

    def test_sampler_failure_reaps_window_and_stops_matrix(self):
        class FakeProcess:
            pid = 123

        class FakeHpc:
            def __init__(self, fail=False):
                self.fail = fail
                self.calls = 0

            def supervise(self, process, *, node, deadline, receipt_path):
                self.calls += 1
                if self.fail:
                    raise ValueError("child failure")
                return {"exit_code": 0}

            def run(self, report, candidate):
                for index in range(2):
                    self.module._supervise_process(
                        FakeProcess(), node={}, deadline=0,
                        receipt_path=Path(report) / f"case{index}" / "SUPERVISOR_RECEIPT.json",
                    )
                return {"status": "unexpected"}

        with tempfile.TemporaryDirectory() as root:
            report, context_sha = _write_context_report(root)
            fake = FakeHpc()
            observer = mock.MagicMock()
            observer.start.return_value = observer
            observer.stop.return_value = {
                "schema": controller.TELEMETRY_SCHEMA + ".receipt.v1",
                "status": "FAILED", "errors": [{"kind": "SAMPLE_ERROR"}],
            }
            fake_module = types.SimpleNamespace(_supervise_process=fake.supervise, run=fake.run)
            fake.module = fake_module
            real_import = controller.importlib.import_module

            def import_module(name):
                if name == "tools.cpu_s4_6f_fullsize_parallel.hpc":
                    return fake_module
                return real_import(name)

            with mock.patch.object(controller.importlib, "import_module", side_effect=import_module), mock.patch.object(
                controller, "ProcessObserver", return_value=observer
            ), mock.patch.object(controller, "pv") as pv, mock.patch.object(controller, "_validate_source_binding"):
                pv.raw_sha256_file.return_value = context_sha
                with mock.patch.dict(os.environ, {"S46G_CONTEXT_SHA256": context_sha}, clear=False), self.assertRaisesRegex(controller.ObserverWindowFailure, "case0"):
                    controller.run(report)
            result = json.loads((report / "DIAGNOSTIC_RESULT.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "DIAGNOSTIC_OBSERVER_FAILURE")
            self.assertEqual(fake.calls, 1)
            self.assertIs(fake_module._supervise_process.__func__, fake.supervise.__func__)

        with tempfile.TemporaryDirectory() as root:
            report, context_sha = _write_context_report(root)
            fake = FakeHpc(fail=True)
            observer = mock.MagicMock()
            observer.start.return_value = observer
            observer.stop.return_value = {"status": "STOPPED", "errors": []}
            fake_module = types.SimpleNamespace(_supervise_process=fake.supervise, run=fake.run)
            fake.module = fake_module
            real_import = controller.importlib.import_module

            def import_module(name):
                if name == "tools.cpu_s4_6f_fullsize_parallel.hpc":
                    return fake_module
                return real_import(name)

            with mock.patch.object(controller.importlib, "import_module", side_effect=import_module), mock.patch.object(
                controller, "ProcessObserver", return_value=observer
            ), mock.patch.object(controller, "pv") as pv, mock.patch.object(controller, "_validate_source_binding"):
                pv.raw_sha256_file.return_value = context_sha
                with mock.patch.dict(os.environ, {"S46G_CONTEXT_SHA256": context_sha}, clear=False), self.assertRaisesRegex(ValueError, "child failure"):
                    controller.run(report)
            self.assertTrue(observer.stop.called)
            self.assertIs(fake_module._supervise_process.__func__, fake.supervise.__func__)


if __name__ == "__main__":
    unittest.main()

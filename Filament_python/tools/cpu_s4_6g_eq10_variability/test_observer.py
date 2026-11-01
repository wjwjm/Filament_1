"""Focused, array-free tests for the S4-6G procfs observer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import observer as observer_module
from .observer import (
    MAX_OUTPUT_BYTES,
    MAX_SAMPLES,
    MAX_THREADS,
    ObservationError,
    ProcessObserver,
    parse_cpu_stat,
    parse_allowed_list,
    parse_numa_policy_maps,
    parse_proc_stat,
    parse_schedstat,
    parse_status,
    probe_current_pid,
    summarize_numa_maps,
    validate_policy_snapshot,
    _node_sample,
    _safe_file,
    _thread_ids_with_bound,
)


class ObserverParserTests(unittest.TestCase):
    def test_proc_stat_keeps_parenthesized_name_and_field_positions(self):
        # field 2 may contain spaces and a closing parenthesis.
        tail = ["S"] + [str(i) for i in range(4, 53)]
        value = parse_proc_stat("17 (python worker) weird) " + " ".join(tail))
        self.assertEqual(value["pid"], 17)
        self.assertEqual(value["utime_ticks"], 14)
        self.assertEqual(value["stime_ticks"], 15)
        self.assertEqual(value["minflt"], 10)
        self.assertEqual(value["majflt"], 12)
        self.assertEqual(value["starttime_ticks"], 22)
        self.assertEqual(value["rss_pages"], 24)
        self.assertEqual(value["num_threads"], 20)
        self.assertEqual(value["last_cpu"], 39)

    def test_proc_status_and_schedstat(self):
        status = parse_status("Uid:\t10\t10\t10\t10\nVmRSS: 42 kB\nCpus_allowed_list: 0-3\n", {"Uid", "VmRSS", "Cpus_allowed_list", "Mems_allowed_list"})
        self.assertEqual(status["Uid"], "10\t10\t10\t10")
        self.assertEqual(status["VmRSS"], 42)
        self.assertEqual(status["Cpus_allowed_list"], "0-3")
        self.assertEqual(status["Mems_allowed_list"]["status"], "UNAVAILABLE")
        self.assertEqual(parse_schedstat("100 20 3"), {"runtime_ns": 100, "runqueue_wait_ns": 20, "timeslices": 3})

    def test_numa_summary_drops_mapping_identity(self):
        value = summarize_numa_maps("00400000 default file=/x N0=3 N1=2\n00401000 N1=4 anon=1\n")
        self.assertEqual(value, {"N0": 3, "N1": 6})
        self.assertNotIn("file", value)

    def test_numa_summary_without_node_tokens_is_unavailable(self):
        self.assertEqual(summarize_numa_maps("00400000 default file=/x\n"), {
            "status": "UNAVAILABLE", "reason": "NUMA_NODE_TOKENS_MISSING"
        })

    def test_policy_parser_and_allowed_sets(self):
        text = "00400000 default file=/x N0=1\n00401000 interleave:0-3 anon=4 N0=1 N1=1 N2=1 N3=1\n"
        parsed = parse_numa_policy_maps(text)
        self.assertEqual(parsed["policy_labels"], {"default": 1, "interleave": 1})
        self.assertEqual(parse_allowed_list("0-3"), {0, 1, 2, 3})
        valid = validate_policy_snapshot(
            text, expected_policy="I", expected_mems={0, 1, 2, 3}, expected_cpus={0, 1},
            cpus_allowed="0-1", mems_allowed="0-3",
        )
        self.assertEqual(valid["phase"], "BODY_POLICY")

    def test_default_policy_does_not_require_interleave_suffix(self):
        valid = validate_policy_snapshot(
            "00400000 default anon=4 N0=4\n", expected_policy="D",
            expected_mems={0, 1, 2, 3}, expected_cpus={0},
            cpus_allowed="0", mems_allowed="0-3",
        )
        self.assertEqual(valid["policy"], "D")

    @unittest.skipUnless(Path("/proc/self/stat").is_file(), "Linux procfs required")
    def test_current_pid_probe_is_read_only_capability_record(self):
        value = probe_current_pid()
        self.assertEqual(value["pid"], os.getpid())
        self.assertTrue(value["identity_verified"])
        self.assertEqual(value["required"]["main_stat"], "AVAILABLE")
        self.assertIn("clock", value)
        self.assertIn("units", value)

    def test_fixed_bounds_are_explicit(self):
        self.assertEqual((MAX_SAMPLES, MAX_THREADS, MAX_OUTPUT_BYTES), (1450, 512, 64 * 1024 * 1024))
        with self.assertRaises(ObservationError):
            parse_proc_stat("bad")

    def test_missing_optional_proc_file_is_unavailable(self):
        with tempfile.TemporaryDirectory() as root:
            value = _safe_file(Path(root) / "missing", parse_cpu_stat)
        self.assertEqual(value, {"status": "UNAVAILABLE", "reason": "FILE_MISSING"})

    def test_missing_thread_directory_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            tids, total, capped, error = _thread_ids_with_bound(Path(root) / "task")
        self.assertEqual((tids, total, capped, error), ([], 0, False, "FILE_MISSING"))

    def test_cpu_mhz_missing_is_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as root:
            proc = Path(root)
            (proc / "cpuinfo").write_text("processor\t: 0\n", encoding="utf-8")
            value = _node_sample(proc)
        self.assertEqual(value["cpu_mhz"]["status"], "UNAVAILABLE")
        self.assertEqual(value["cpu_mhz"]["processor_count"], 1)

    def test_pid_identity_change_is_rejected(self):
        observer = ProcessObserver(123, tempfile.gettempdir())
        observer._identity = (1.0, 1000, 10)
        with mock.patch.object(observer_module, "_pid_identity", return_value=(2.0, 1000, 11)):
            with self.assertRaisesRegex(ObservationError, "PID_IDENTITY_CHANGED"):
                observer._sample(0)

    def test_stop_and_join_are_repeatable(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(observer_module, "_pid_identity", return_value=(1.0, 1000, 10)):
                with mock.patch.object(ProcessObserver, "_sample", return_value={"scalar": 1}):
                    observer = ProcessObserver(123, Path(root) / "telemetry", period_s=60.0)
                    observer.start()
                    first = observer.stop()
                    second = observer.stop()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "STOPPED")


if __name__ == "__main__":
    unittest.main()

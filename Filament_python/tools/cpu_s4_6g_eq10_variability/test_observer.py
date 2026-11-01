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
    @staticmethod
    def _reservation_fixture(*, local_count=2, size_hex="02000000", local_policy="local", resident="N0=0", special=False):
        maps = []
        numa = []
        base = 0x00400000
        for index in range(local_count):
            start = base + index * int(size_hex, 16)
            end = start + int(size_hex, 16)
            path = " [heap]" if special and index == 0 else ""
            maps.append(f"{start:08x}-{end:08x} rw-p 00000000 00:00 0{path}")
            numa.append(f"{start:08x} {local_policy} anon {resident}")
        start = base + local_count * int(size_hex, 16)
        numa.append(f"{start:08x} default anon N0=1")
        maps.append(f"{start:08x}-{start + 0x1000:08x} rw-p 00000000 00:00 0")
        return "\n".join(maps) + "\n", "\n".join(numa) + "\n"

    def test_verified_local_reservations_are_excluded_for_default_and_interleave(self):
        maps, numa = self._reservation_fixture()
        value = validate_policy_snapshot(
            numa, expected_policy="D", expected_mems={0, 1, 2, 3}, expected_cpus={0},
            cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
        )
        self.assertEqual(value["policy_labels"], {"default": 1})
        self.assertEqual(value["local_reservation_summary"]["count"], 2)
        interleave = numa.replace("default anon N0=1", "interleave:0-3 anon N0=1 N1=1 N2=1 N3=1")
        value = validate_policy_snapshot(
            interleave, expected_policy="I", expected_mems={0, 1, 2, 3}, expected_cpus={0},
            cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
        )
        self.assertEqual(value["policy_labels"], {"interleave": 1})

    def test_zero_resident_local_reservation_without_node_tokens_is_valid(self):
        maps, numa = self._reservation_fixture(resident="anon=0")
        value = validate_policy_snapshot(
            numa, expected_policy="D", expected_mems={0, 1, 2, 3}, expected_cpus={0},
            cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
        )
        self.assertEqual(value["local_reservation_summary"]["count"], 2)
        self.assertEqual(value["local_reservations"][0]["resident_pages_by_node"], {})

    def test_positive_page_counter_without_node_tokens_is_rejected(self):
        maps, numa = self._reservation_fixture(resident="anon=1")
        with self.assertRaisesRegex(ObservationError, "LOCAL_RESERVATION_COUNTERS_WITHOUT_NODES"):
            validate_policy_snapshot(
                numa, expected_policy="D", expected_mems={0, 1, 2, 3}, expected_cpus={0},
                cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
            )

    def test_local_reservation_resident_size_count_special_and_association_are_rejected(self):
        cases = (
            self._reservation_fixture(resident="N0=1"),
            self._reservation_fixture(size_hex="00100000"),
            self._reservation_fixture(local_count=3),
            self._reservation_fixture(special=True),
        )
        for maps, numa in cases:
            with self.assertRaises(ObservationError):
                validate_policy_snapshot(
                    numa, expected_policy="D", expected_mems={0, 1, 2, 3}, expected_cpus={0},
                    cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
                )
        maps, numa = self._reservation_fixture()
        with self.assertRaisesRegex(ObservationError, "LOCAL_RESERVATION_MAP_ASSOCIATION_MISSING"):
            validate_policy_snapshot(
                numa.replace("00400000", "0bad0000", 1), expected_policy="D",
                expected_mems={0, 1, 2, 3}, expected_cpus={0}, cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
            )
        maps, numa = self._reservation_fixture(local_count=3)
        with self.assertRaises(ObservationError) as raised:
            validate_policy_snapshot(
                numa, expected_policy="D", expected_mems={0, 1, 2, 3}, expected_cpus={0},
                cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
            )
        self.assertEqual(raised.exception.details["local_reservation_count"], 3)
        self.assertLessEqual(len(raised.exception.details["local_reservations"]), 2)

    def test_malformed_n_field_and_failure_summary_do_not_retain_address_or_path(self):
        maps, numa = self._reservation_fixture()
        with self.assertRaises(ObservationError) as raised:
            validate_policy_snapshot(
                numa.replace("N0=0", "Nbad=0", 1), expected_policy="D", expected_mems={0, 1, 2, 3},
                expected_cpus={0}, cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
            )
        details = raised.exception.details
        self.assertIn("actual_policy_labels", details)
        self.assertIn("local_reservations", details)
        self.assertNotIn("00400000", repr(details))
        self.assertNotIn("/", repr(details))

    def test_duplicate_n_field_and_unknown_policy_are_rejected_with_scalar_details(self):
        maps, numa = self._reservation_fixture()
        with self.assertRaisesRegex(ObservationError, "NUMA_NODE_FIELD_DUPLICATE") as duplicate:
            validate_policy_snapshot(
                numa.replace("N0=0", "N0=0 N0=0", 1), expected_policy="D",
                expected_mems={0, 1, 2, 3}, expected_cpus={0},
                cpus_allowed="0", mems_allowed="0-3", maps_text=maps,
            )
        self.assertNotIn("00400000", repr(duplicate.exception.details))
        with self.assertRaisesRegex(ObservationError, "NUMA_POLICY_UNKNOWN") as unknown:
            validate_policy_snapshot(
                "00400000 mystery anon N0=1\n00401000 default anon N0=1\n",
                expected_policy="D", expected_mems={0, 1, 2, 3}, expected_cpus={0},
                cpus_allowed="0", mems_allowed="0-3",
            )
        self.assertEqual(unknown.exception.details["actual_policy_labels"]["mystery"], 1)

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

    def test_failed_policy_sample_preserves_address_free_scalar_summary(self):
        details = {
            "actual_policy_labels": {"local": 2, "default": 1},
            "local_reservation_count": 2,
            "local_reservations": [{"index": 0, "kind": "anonymous", "size_bytes": 33554432, "resident_pages_total": 0}],
        }
        with tempfile.TemporaryDirectory() as root:
            telemetry = Path(root) / "telemetry"
            with mock.patch.object(observer_module, "_pid_identity", return_value=(1.0, 1000, 10)), mock.patch.object(
                ProcessObserver, "_sample", side_effect=ObservationError("LOCAL_RESERVATION_SIZE_INVALID", details)
            ):
                observer = ProcessObserver(123, telemetry, period_s=60.0)
                observer.start()
                assert observer._thread is not None
                observer._thread.join(1.0)
                receipt = observer.stop()
            rows = [json.loads(line) for line in (telemetry / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(receipt["status"], "FAILED")
        self.assertEqual(rows[0]["sample_failure"]["reason"], "LOCAL_RESERVATION_SIZE_INVALID")
        self.assertNotIn("00400000", json.dumps(rows[0]))
        self.assertNotIn("/tmp", json.dumps(rows[0]))


if __name__ == "__main__":
    unittest.main()

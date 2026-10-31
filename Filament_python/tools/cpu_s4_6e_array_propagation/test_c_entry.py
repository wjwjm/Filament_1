"""C entry contract and explicitly opt-in fresh microgrid runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.cpu_s4_3g3_smallgrid import benchmark as sg

from KHz_filament.nonlinear import apply_nonlinear

from .c_capture import CaptureStore
from . import c_hpc
from .c_entry import _load_capture, _mechanism, _qualify_boundary, _save_capture, _write_entry_failure, admission, input_domain
from .c_protocol import MICRO_SHAPE, SHAPE, plan


def test_s46e_c_entry_domain_and_plan_contracts(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    admission(local_engineering=True, shape=MICRO_SHAPE, intervals=2, mode="QUALIFY_BOUNDARY", role="R")
    admission(local_engineering=True, shape=MICRO_SHAPE, intervals=8, mode="PURE_OFF_SMALLGRID", role="C")
    with pytest.raises(RuntimeError, match="LOCAL_NT384"):
        admission(local_engineering=True, shape=SHAPE, intervals=2, mode="QUALIFY_BOUNDARY", role="R")
    with pytest.raises(RuntimeError, match="CAPTURE_ONLY_R8"):
        admission(local_engineering=True, shape=MICRO_SHAPE, intervals=2, mode="QUALIFY_BOUNDARY", role="R", capture=True)
    assert plan()["mechanism"]["formal_calls"] == 90
    assert plan()["mechanism"]["warmup_calls"] == 36


def test_s46e_c_input_domain_rejects_main_grid_in_local_mode():
    input_domain({"shape": list(MICRO_SHAPE)}, local_engineering=True)
    with pytest.raises(RuntimeError, match="MANIFEST_SHAPE"):
        input_domain({"shape": list(SHAPE)}, local_engineering=True)


def test_s46e_c_capture_package_roundtrip_is_npy_only(tmp_path):
    store = CaptureStore()
    store.begin("G1", source_role="R")
    for interval in range(8):
        E = np.ones((2, 2, 2), dtype=np.complex128)
        phase = np.full(E.shape, interval, dtype=np.float64)
        alpha = np.full(E.shape, 0.01, dtype=np.float64)
        dn = np.ones(E.shape[1:], dtype=np.float64)
        args = (E, phase, alpha, 0.01)
        kwargs = {"dn_gas": dn, "k0": 0.2}
        store.before_call(interval, args, kwargs)
        apply_nonlinear(*args, **kwargs)
        store.after_call(interval, E, args, kwargs)
    package = _save_capture(
        store,
        tmp_path / "capture",
        provenance={
            "execution_json_sha256": "execution",
            "input_manifest_sha256": "inputs",
            "source_identity_digest": "source",
            "source_binding": {"role": "R", "source": {"canonical_lf_sha256": "source-sha"}},
            "case_source": {"case": "TEST", "role": "R", "intervals": 8, "shape": [2, 2, 2], "capture": True},
        },
    )
    assert package["array_storage"] == "NPY_ALLOW_PICKLE_FALSE"
    assert package["live_original_array_refs_retained"] is False
    assert package["captured_owned_arrays"] is True
    assert all(row["classification"] == "external" and row["hash_scope"] == "raw_bytes" and row["raw_sha256"] for row in package["files"])
    assert len(list((tmp_path / "capture").glob("*.npy"))) == 15
    loaded = _load_capture(tmp_path / "capture", expected_shape=(2, 2, 2))
    assert loaded["metadata"]["selected_interval_indices"] == [0, 3, 7]
    assert loaded["metadata"]["expected_total_calls"] == 8
    assert set(loaded["records"]) == {0, 3, 7}
    store.release_owned_arrays()
    assert store.records() == {}


def test_s46e_c_capture_load_rejects_source_case_tamper(tmp_path):
    package_root = tmp_path / "capture"
    package = {
        "schema": "filament.cpu_s4_6e.c_capture.package.v1",
        "metadata": {},
        "records": {},
        "files": [],
        "provenance": {
            "execution_json_sha256": "execution",
            "input_manifest_sha256": "inputs",
            "source_identity_digest": "source",
            "source_binding": {"role": "C", "source": {"canonical_lf_sha256": "source-sha"}},
            "case_source": {"role": "R", "intervals": 8, "shape": [2, 2, 2], "capture": True},
        },
        "live_original_array_refs_retained": False,
        "captured_owned_arrays": True,
    }
    package_root.mkdir()
    (package_root / "CAPTURE.json").write_text(json.dumps(package), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SOURCE_ROLE"):
        _load_capture(package_root, expected_shape=(2, 2, 2))


def test_s46e_c_hpc_help_and_p24_hpc_only(tmp_path):
    command = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6e_array_propagation.c_hpc", "--help"]
    completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[3]), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    from types import SimpleNamespace

    with pytest.raises(RuntimeError, match="P24_HPC_ONLY"):
        _mechanism(SimpleNamespace(local_engineering=True, allow_p24=True, report=tmp_path))


def test_s46e_c_entry_failure_preserves_science_index_and_exact_paths(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    exact = work / "formal" / "BOUNDARY_EXACT.jsonl"
    exact.parent.mkdir()
    exact.write_text("{}\n", encoding="utf-8")
    error = RuntimeError("STOP_FIRST_SCIENCE_NONEXACT interval=7")
    error.failure = {"path": "formal/BOUNDARY_EXACT.jsonl", "interval_index": 7}
    payload = _write_entry_failure(work, error)
    assert payload["classification"] == "SCIENTIFIC_NONEXACT"
    assert payload["stop_reason"] == "STOP_FIRST_SCIENCE_NONEXACT"
    assert payload["failure_index"]["interval_index"] == 7
    assert "formal/BOUNDARY_EXACT.jsonl" in payload["exact_comparison_paths"]
    assert Path(payload["traceback_path"]).is_file()


def test_s46e_c_hpc_generates_and_validates_input_manifest_before_launch(tmp_path, monkeypatch):
    calls = []

    def fake_node_gate(report):
        return {"job_id": "mock-job"}

    def fake_input_manifest(report, *, shape):
        calls.append(tuple(shape))
        return {"fingerprints": {"source": "wrong"}}

    monkeypatch.setattr(c_hpc, "node_gate", fake_node_gate)
    monkeypatch.setattr(c_hpc.sg, "input_manifest", fake_input_manifest)
    monkeypatch.setattr(
        c_hpc.c_entry,
        "read",
        lambda path: {"expected_input_fingerprints": {"source": "frozen"}},
    )
    monkeypatch.setattr(c_hpc, "_launch", lambda *args, **kwargs: pytest.fail("launch must follow the input gate"))
    with pytest.raises(RuntimeError, match="FROZEN_DIRECT_SMALLGRID_INPUTS"):
        c_hpc.run(tmp_path)
    assert calls == [SHAPE]


def test_s46e_c_two_interval_boundary_is_disk_checked_and_reported(tmp_path):
    from types import SimpleNamespace

    work = tmp_path / "case"
    boundary = work / "formal" / "BOUNDARY_RECEIPT.json"
    reference = tmp_path / "reference_boundary.json"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("{}", encoding="utf-8")
    reference.write_text("{}", encoding="utf-8")

    class Exact:
        def inspect_boundary_metadata(self, path, *, require_members, header_only):
            assert path == boundary
            assert require_members is True and header_only is True
            return {"status": "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"}

        def compare_current_boundary_pair(self, old, current, output):
            assert old == reference and current == boundary
            assert output == work / "formal" / "BOUNDARY_EXACT.jsonl"
            return {"status": "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"}

    class Lease:
        def transparency(self):
            return {"retired": True}

    args = SimpleNamespace(intervals=2, reference_boundary=reference)
    result = {"boundary": {"retained_live_array_refs": 0}, "comparisons": {}}
    path = _qualify_boundary(args, work, result, [None, None, None, None, None, Lease()], lambda name: Exact())
    assert path == str(boundary)
    assert result["boundary_path"] == str(boundary)
    assert result["comparisons"]["boundary"]["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
    assert (work / "DISK_BOUNDARY_ROUNDTRIP.json").is_file()

    result8 = {}
    assert _qualify_boundary(SimpleNamespace(intervals=8), work, result8, [None] * 6, lambda name: Exact()) is None
    assert result8["boundary_path"] is None


@pytest.mark.skipif(not os.environ.get("S46E_C_REAL_ENTRY_REPORT"), reason="Explicit clean-source C micro entry phase only")
def test_s46e_c_real_entry_micro_fresh_r2_c2_r8_capture_c8():
    report = Path(os.environ["S46E_C_REAL_ENTRY_REPORT"]).resolve(strict=True)
    sg.input_manifest(report, shape=MICRO_SHAPE)
    root = report / "local_c"
    root.mkdir(exist_ok=False)

    def run_case(case, role, intervals, *, capture=False, reference=None):
        folder = root / case
        folder.mkdir(exist_ok=False)
        work = folder / "work"
        argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6e_array_propagation.c_entry", "--report", str(report), "--work", str(work), "--role", role, "--intervals", str(intervals), "--local-engineering", "--input-manifest", str(report / "BENCHMARK_INPUT_MANIFEST.json")]
        if capture:
            argv.extend(("--capture", "--capture-root", str(folder / "CAPTURE")))
        if reference is not None:
            argv.extend(("--reference-output", reference["output"]))
            if intervals == 2:
                argv.extend(("--reference-boundary", reference["boundary_path"], "--reference-stages", reference["stages"]))
        env = os.environ.copy()
        env["NUMBA_NUM_THREADS"] = "1"
        subprocess.run(argv, check=True, env=env, cwd=str(Path(__file__).resolve().parents[3]))
        return json.loads((work / "CASE_RECEIPT.json").read_text(encoding="utf-8"))

    r2 = run_case("R2", "R", 2)
    c2 = run_case("C2", "C", 2, reference=r2)
    assert c2["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    assert c2["comparisons"]["boundary"]["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
    assert c2["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    r8 = run_case("R8_UNCAPTURED", "R", 8)
    r8c = run_case("R8_CAPTURED", "R", 8, capture=True, reference=r8)
    assert r8c["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    c8 = run_case("C8", "C", 8, reference=r8)
    assert c8["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    assert Path(r8c["capture_package"]).joinpath("CAPTURE.json").is_file()

"""Micro-only control-flow checks for the FULLSIZE confirmation namespace."""
from __future__ import annotations

import os
import json
from pathlib import Path
import time

import pytest

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_3g3_smallgrid.adapter import build_role
from tools.cpu_s4_checkpoint_support import support_module
from . import entry


def _data_root() -> Path:
    value = os.environ.get("CPU_S4_DATA_ROOT")
    if not value:
        pytest.skip("CPU_S4_DATA_ROOT is required for the real micro control-flow check")
    root = Path(value).resolve(strict=True)
    for path in (root / "SMALLGRID_CONFIG.json", root / "raw" / "FROZEN_LUT_RETRIEVAL.json",
                 root / "inputs" / "LUT_MANIFEST.json"):
        if not path.is_file():
            pytest.skip("sealed CPU_S4_DATA_ROOT is incomplete")
    return root


def test_full_entry_micro_two_interval_g2_g3_exact_and_clock(tmp_path, monkeypatch):
    _data_root()
    monkeypatch.setenv("NUMBA_NUM_THREADS", "1")
    numba = pytest.importorskip("numba")
    if int(numba.config.NUMBA_NUM_THREADS) != 1:
        pytest.skip("micro control-flow check requires the admitted local pool=1")
    report = tmp_path / "report"
    report.mkdir()
    context = entry.micro_context(report, shape=(32, 17, 19))
    fingerprints = sg.fingerprints(context)
    manifest = report / "MICRO_INPUT_MANIFEST.json"
    sg.write(manifest, {"schema": "filament.fullsize.micro_control.v1",
                        "shape": [32, 17, 19], "fingerprints": fingerprints,
                        "execution_domain": "MICRO_CONTROL_FLOW_ONLY"})
    full_window = entry.full_window(entry.micro_context)
    exact = support_module("g1_exact")
    assert Path(entry.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[3])
    assert Path(exact.__file__).resolve().parent == Path(support_module("g1_exact").__file__).resolve().parent
    previous = None
    for role in ("G2_SMALL", "G3_SMALL"):
        work = tmp_path / role
        started=time.perf_counter()
        built = build_role(role, "QUALIFY_BOUNDARY", capture_root=work / "BOUNDARY",
                           local_engineering=True, pool_override=1)
        args = type("Args", (), {
            "report": report, "work": work, "role": role, "reported_role": role,
            "mode": "QUALIFY_BOUNDARY", "intervals": 2, "local_engineering": True,
            "input_manifest": manifest, "reference_output": None if previous is None else previous["output"],
            "reference_boundary": None if previous is None else previous["boundary_path"],
            "reference_stages": None if previous is None else previous["stages"],
            "execution_domain": "MICRO_CONTROL_FLOW_ONLY",
        })()
        result = full_window(args, built, label="formal", intervals=2, capture=True,
                             application_started=started)
        assert result["scope"] == "LOCAL_ENGINEERING_ONLY"
        assert result["fullsize_new_candidate"] == "NOT_RUN"
        assert result["route"]["nH"] >= 4
        assert result["stage_activity"] and all(value["I_R_max"] > 0 and value["rhs_max"] > 0
                                                  for value in result["stage_activity"].values())
        assert result["optical_application_window_s"] >= result["optical_compute_window_s"]
        previous = dict(result)
        previous["boundary_path"] = work / "formal" / "BOUNDARY_RECEIPT.json"
    assert previous["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    assert previous["comparisons"]["boundary"]["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
    assert previous["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    assert previous["scope"] == "LOCAL_ENGINEERING_ONLY"
    terminal_tree = json.loads((Path(previous["output"]) / "TREE.json").read_text(encoding="utf-8"))
    stage_tree = json.loads((Path(previous["stages"]) / "TREE.json").read_text(encoding="utf-8"))
    assert "POST" in json.dumps(terminal_tree)
    assert all(name in json.dumps(stage_tree) for name in ("field", "stage1", "predictor", "stage2"))

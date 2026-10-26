"""Real micro collector-to-disk-to-reload boundary regression.

This deliberately uses the local engineering contract and a tiny transverse
space. It exercises the same ``entry.full_window`` namespace as the fullsize
controller, while keeping the scientific full grid and HPC allocation out of
the test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.cpu_s4_3g3_smallgrid.adapter import build_role
from tools.cpu_s4_checkpoint_support import support_module
from . import entry


def _data_root() -> Path:
    configured = os.environ.get("CPU_S4_DATA_ROOT")
    if not configured:
        pytest.skip("CPU_S4_DATA_ROOT is required for the real boundary micro check")
    root = Path(configured).resolve(strict=True)
    for path in (root / "SMALLGRID_CONFIG.json", root / "raw" / "FROZEN_LUT_RETRIEVAL.json",
                 root / "inputs" / "LUT_MANIFEST.json"):
        if not path.is_file():
            pytest.skip("sealed CPU_S4_DATA_ROOT is incomplete")
    return root


def _require(condition: bool, reason: str, context=None) -> None:
    if not condition:
        raise RuntimeError({"gate": reason, "context": context})


@pytest.mark.parametrize("intervals", (2, 8))
def test_real_collector_close_disk_reload_and_original_compare(intervals: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _data_root()
    monkeypatch.setenv("NUMBA_NUM_THREADS", "1")
    numba = pytest.importorskip("numba")
    if int(numba.config.NUMBA_NUM_THREADS) != 1:
        pytest.skip("real boundary micro check requires the admitted local pool=1")
    report = tmp_path / "report"
    report.mkdir()
    shape = (384, 17, 19)
    context = entry.micro_context(report, shape=shape)
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg

    manifest = report / "MICRO_INPUT_MANIFEST.json"
    sg.write(manifest, {
        "schema": "filament.fullsize.micro_boundary_roundtrip.v1",
        "shape": list(shape),
        "fingerprints": sg.fingerprints(context),
        "execution_domain": "MICRO_CONTROL_FLOW_ONLY",
        "intervals": intervals,
        "Nt": shape[0],
    })
    full_window = entry.full_window(entry.micro_context)
    exact = support_module("g2_hpc_exact")
    previous = None
    records = []
    for role in ("G2_SMALL", "G3_SMALL"):
        work = tmp_path / f"{role}_{intervals}"
        built = build_role(role, "QUALIFY_BOUNDARY", capture_root=work / "BOUNDARY",
                           local_engineering=True, pool_override=1)
        args = type("Args", (), {
            "report": report,
            "work": work,
            "role": role,
            "reported_role": role,
            "mode": "QUALIFY_BOUNDARY",
            "intervals": intervals,
            "local_engineering": True,
            "input_manifest": manifest,
            "reference_output": None if previous is None else previous["output"],
            "reference_boundary": None if previous is None else previous["boundary_path"],
            "reference_stages": None if previous is None else previous["stages"],
            "execution_domain": "MICRO_CONTROL_FLOW_ONLY",
        })()
        result = full_window(args, built, label="formal", intervals=intervals, capture=True,
                             application_started=None)
        boundary_path = work / "formal" / "BOUNDARY_RECEIPT.json"
        _require(boundary_path.is_file(), "REAL_BOUNDARY_RECEIPT_PATH")
        _require(result["boundary"]["field_count"] == 112, "REAL_BOUNDARY_FIELD_COUNT")
        _require(result["boundary"]["retained_live_array_refs"] == 0, "REAL_BOUNDARY_LIVE_REFERENCE_RELEASE")
        _require(built[5].transparency().get("retired") is True, "REAL_BOUNDARY_LEASE_RETIRED")
        metadata = exact.inspect_boundary_metadata(boundary_path, require_members=True, header_only=True)
        _require(metadata["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY", "REAL_BOUNDARY_METADATA_GATE")
        _require(metadata["scientific_comparison_status"] == "NOT_RUN_METADATA_ONLY", "METADATA_GATE_SCOPE")
        if previous is not None:
            disk_comparison = exact.compare_current_boundary_pair(
                previous["boundary_path"], boundary_path,
                work / "DISK_BOUNDARY_EXACT.jsonl",
            )
            _require(disk_comparison["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT",
                     "REAL_DISK_BOUNDARY_COMPARE", disk_comparison)
            _require(result["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS",
                     "REAL_TERMINAL_COMPARE", result["comparisons"])
            _require(result["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS",
                     "REAL_STAGE_COMPARE", result["comparisons"])
        tree = json.loads((Path(result["output"]) / "TREE.json").read_text(encoding="utf-8"))
        _require("POST" in json.dumps(tree), "REAL_POST_TREE_PRESENT")
        _require(result["route"]["nH"] > 0 and result["route"]["fallback"] == 0,
                 "REAL_ROUTE_CONTRACT", result["route"])
        previous = dict(result)
        previous["boundary_path"] = boundary_path
        records.append(dict(role=role,intervals=intervals,shape=list(shape),
                            boundary_path=str(boundary_path),compatibility=metadata,
                            route=result["route"],comparisons=result["comparisons"],
                            retained_live_array_refs=result["boundary"]["retained_live_array_refs"],
                            view_closed_by_original_window=True))
    _require(previous["shape"] == list(shape), "REAL_MICRO_NT384_SHAPE")
    _require(previous["intervals"] == intervals, "REAL_MICRO_INTERVAL_COUNT")
    exact.write_json(tmp_path / "REAL_DISK_ROUNDTRIP_RECEIPT.json", dict(
        status="PASS_LOCAL_REAL_COLLECTOR_DISK_ROUNDTRIP",local_worker_pool=1,
        intervals=intervals,Nt=384,records=records,
        fullsize_qualification="NOT_RUN",W48_qualification="NOT_RUN"))

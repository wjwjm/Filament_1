"""The final entry, normal outputs and real warmup on the approved micro grid."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest

from .benchmark import input_manifest, run


def test_smallgrid_entry_rejects_preloaded_wrong_adapter(tmp_path, monkeypatch):
    from .adapter import build_role
    from .scientific_context import frozen_tools
    report = Path(os.environ["S43SG_REPORT_ROOT"])
    frozen_tools(report)
    build_role("G3_SMALL", "PURE_OFF_SMALLGRID", capture_root=tmp_path / "baseline", local_engineering=True, pool_override=1)
    wrong = ModuleType("g2_adapter")
    wrong.__file__ = str(tmp_path / "wrong_report_adapter.py")
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "g2_adapter", wrong)
        with pytest.raises(RuntimeError, match="SMALLGRID_FROZEN_ADAPTER_ACTUAL_MODULE_MISMATCH"):
            build_role("G3_SMALL", "PURE_OFF_SMALLGRID", capture_root=tmp_path / "bad", local_engineering=True, pool_override=1)
    build_role("G3_SMALL", "PURE_OFF_SMALLGRID", capture_root=tmp_path / "recovered", local_engineering=True, pool_override=1)


def test_smallgrid_entry_normal_io_warmup_exact(tmp_path):
    report = Path(os.environ["S43SG_REPORT_ROOT"])
    suffix = "_MICRO_ENTRY_" + os.environ["S43SG_STAGE"]
    input_manifest(report, suffix=suffix, shape=(32, 17, 19))
    manifest = report / ("BENCHMARK_INPUT_MANIFEST" + suffix + ".json")
    reference = None
    boundary = None
    stages = None
    for role, mode in (("I0_SMALL", "QUALIFY_BOUNDARY"), ("G2_SMALL", "QUALIFY_BOUNDARY"), ("G3_SMALL", "QUALIFY_BOUNDARY"), ("G3_SMALL", "PURE_OFF_SMALLGRID")):
        args = SimpleNamespace(report=report, work=tmp_path / (role + "_" + mode), role=role, mode=mode, intervals=2,
            local_engineering=True, warmup=mode == "PURE_OFF_SMALLGRID", input_manifest=manifest,
            reference_output=reference, reference_boundary=boundary if mode == "QUALIFY_BOUNDARY" else None,
            reference_stages=stages if role == "G3_SMALL" and mode == "QUALIFY_BOUNDARY" else None)
        receipt = run(args)
        assert receipt["normal_outputs"]["scalar_diagnostic_field_count"] > 0
        if role == "I0_SMALL":
            reference = Path(receipt["output"])
            boundary = receipt["boundary"]
        else:
            assert receipt["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        if role == "G2_SMALL":
            stages = Path(receipt["stages"])
        if mode == "PURE_OFF_SMALLGRID":
            assert receipt["boundary"] is None and receipt["warmup"] is not None

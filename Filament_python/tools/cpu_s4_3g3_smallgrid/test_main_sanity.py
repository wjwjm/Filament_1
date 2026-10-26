"""One local main-grid first-boundary sanity through the actual new entry."""
import os
from pathlib import Path
from types import SimpleNamespace
from .benchmark import input_manifest, run


def test_smallgrid_main_first_boundary_sanity(tmp_path):
    report = Path(os.environ["S43SG_REPORT_ROOT"])
    input_manifest(report, suffix="_WINDOWS_LOCAL")
    oracle = None
    boundary = None
    stages = None
    for role in ("I0_SMALL", "G2_SMALL", "G3_SMALL"):
        args = SimpleNamespace(report=report, work=tmp_path / role, role=role, mode="QUALIFY_BOUNDARY",
            intervals=1, local_engineering=True, warmup=False, input_manifest=report / "BENCHMARK_INPUT_MANIFEST_WINDOWS_LOCAL.json",
            reference_output=oracle, reference_boundary=boundary, reference_stages=stages)
        receipt = run(args)
        assert receipt["shape"] == [384, 117, 129] and receipt["intervals"] == 1
        if oracle is None:
            oracle = Path(receipt["output"])
            boundary = receipt["boundary"]
        else:
            assert receipt["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        if role == "G2_SMALL":
            stages = Path(receipt["stages"])
        if role == "G3_SMALL":
            assert receipt["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"

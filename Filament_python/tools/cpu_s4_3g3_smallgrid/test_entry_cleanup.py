"""Real micro propagation followed by a writer fault retires capture refs."""
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from .adapter import build_role
from . import benchmark

def test_checkpoint_real_writer_failure_retires_capture_and_recovers(tmp_path, monkeypatch):
    report=Path(os.environ["S43SG_REPORT_ROOT"])
    suffix="_CLEANUP_"+os.environ["S43SG_STAGE"]
    benchmark.input_manifest(report,suffix=suffix,shape=(32,17,19))
    manifest=report/("BENCHMARK_INPUT_MANIFEST"+suffix+".json")
    args=SimpleNamespace(report=report,work=tmp_path/"baseline",role="G3_SMALL",mode="QUALIFY_BOUNDARY",intervals=2,
        local_engineering=True,input_manifest=manifest,reference_output=None,reference_boundary=None,reference_stages=None)
    built=build_role(args.role,args.mode,capture_root=tmp_path/"baseline_boundary",local_engineering=True,pool_override=1)
    baseline=benchmark.run_window(args,built,label="formal",intervals=2,capture=True)
    state=built[1].cpu_s4_3g_state
    assert not state._g_qualification_capture_enabled and not state._g_qualification_capture_records
    original_writer=benchmark.normal_writer
    def failing_writer(report,intervals):
        writer=original_writer(report,intervals)
        def write(*a,**k):
            assert state._g_qualification_capture_enabled and state._g_qualification_capture_records
            raise RuntimeError("INJECTED_WRITER_FAILURE_AFTER_REAL_PROPAGATION")
        return write
    args.work=tmp_path/"failed"
    built=build_role(args.role,args.mode,capture_root=tmp_path/"failed_boundary",local_engineering=True,pool_override=1)
    state=built[1].cpu_s4_3g_state
    with monkeypatch.context() as patch:
        patch.setattr(benchmark,"normal_writer",failing_writer)
        with pytest.raises(RuntimeError,match="INJECTED_WRITER_FAILURE"):
            benchmark.run_window(args,built,label="formal",intervals=2,capture=True)
    assert not state._g_qualification_capture_enabled and not state._g_qualification_capture_records
    assert not state._g_propagation_active and state._g_pending_cap is None
    # The first-boundary collector is single-use, so recovery uses a fresh role.
    args.work=tmp_path/"recovered";args.reference_output=baseline["output"]
    recovered=build_role(args.role,args.mode,capture_root=tmp_path/"recovered_boundary",local_engineering=True,pool_override=1)
    result=benchmark.run_window(args,recovered,label="formal",intervals=2,capture=True)
    assert result["comparisons"]["terminal"]["status"]=="PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"

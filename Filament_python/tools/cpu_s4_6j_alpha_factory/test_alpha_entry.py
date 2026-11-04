"""Alpha protocol checks and opt-in actual W1 qualification/capture entry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from . import alpha_entry as entry
from .alpha_protocol import CANDIDATE, MICRO_SHAPE, SHAPE, plan, matrix, pairings

def _four_pool_case():
    from . import alpha_capture_hpc as hpc
    from tools.cpu_s4_6i_residual_parallel.selected_measurement import POWER_LABELS
    from tools.cpu_s4_6i_residual_parallel.selected_protocol import CONTEXTS
    n=8
    components={}
    activity=[]
    for name,key in hpc.KEYS.items():
        labels=hpc.parent.POINTWISE_LABELS[name]
        count=n if name=="PHASE_ABSORB_P24" else 4*n
        workers=48 if name=="EQ10_P48" else 24
        components[key]=dict(closed=True,active_thread_ids=[],pending_scalar_records=0,
            close_error=None,operation_counts={label:count for label in labels})
        activity.extend(dict(label=label,requested_workers=workers,actual_workers=workers,
            workers=[dict(items=384//workers)]*workers) for _ in range(count) for label in labels)
    components["power_volume"]=dict(closed=True,active_thread_ids=[],pending_scalar_records=0,
        close_error=None,operation_counts={label:4*n for label in POWER_LABELS})
    activity.extend(dict(label=label,context=context,interval_index=i,
        requested_workers=24,actual_workers=24,workers=[dict(items=16)]*24)
        for i in range(n) for context in CONTEXTS for label in POWER_LABELS)
    return dict(intervals=n,role="R",pool_role="X",cache_fallback=0,
        comparisons=dict(terminal=dict(status="PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS")),
        route=dict(nH=16,fresh_responses=32,math_dispatches=128,fallback=0),
        cache=dict(stats=dict(factor_requests=6144,plane_builds=384,hits=5760)),
        pointwise_pool=dict(pool=dict(components=components,alpha=None,
            all_pools_retired_before_output=True),activity=activity))

def test_alpha_controller_checks_whole_terminal_and_fourth_pool():
    from . import alpha_capture_hpc as hpc
    row=_four_pool_case()
    hpc._exact(row,terminal=True,boundary=False,stages=False)
    assert row["exact_gate_pass"]
    row["comparisons"]["terminal"]["status"]="NONEXACT"
    with pytest.raises(RuntimeError,match="TERMINAL_NONEXACT"):
        hpc._exact(row,terminal=True,boundary=False,stages=False)
    row=_four_pool_case()
    label=next(iter(row["pointwise_pool"]["pool"]["components"]["power_volume"]["operation_counts"]))
    row["pointwise_pool"]["pool"]["components"]["power_volume"]["operation_counts"][label]=0
    with pytest.raises(RuntimeError,match="POOL_COUNTS"):
        hpc._exact(row,terminal=True,boundary=False,stages=False)

def test_alpha_controller_retains_scientific_child_failure(tmp_path, monkeypatch):
    from . import alpha_capture_hpc as hpc
    values={"FULL_INPUT_MANIFEST.json":dict(shape=list(SHAPE),fingerprints={}),
        "INPUT_CONTRACT.json":dict(expected_input_fingerprints={}),
        "REFERENCE_INPUT_RAW_RECORDS.json":dict(reference={})}
    for name,value in values.items():
        (tmp_path/name).write_text(json.dumps(value),encoding="utf-8")
    monkeypatch.setattr(hpc,"_require_fresh_controller",lambda path:None)
    monkeypatch.setattr(hpc,"node_gate",lambda *args:dict(job_id="synthetic"))
    failed=dict(status="SCIENTIFIC_NONEXACT",entry_failure=dict(error="probe"))
    monkeypatch.setattr(hpc,"launch",lambda *args,**kwargs:failed)
    with pytest.raises(RuntimeError,match="SCIENTIFIC_NONEXACT"):
        hpc.run(tmp_path)
    receipt=json.loads((tmp_path/"JOB_FAILURE.json").read_text())
    assert receipt["active_case"]=="R8_UNCAPTURED"
    assert receipt["failed_case"]==failed
    assert receipt["classification"]=="SCIENTIFIC_NONEXACT"

def test_alpha_protocol_keeps_original_size_and_fixed_orders():
    rows = matrix()
    assert len(rows) == 14
    assert "".join(r["role"] for r in rows) == "RXRRRXXRXRRXRR"
    assert rows[0]["case"] == "QUAL_R2" and rows[1]["case"] == "QUAL_X2"
    assert len(pairings()) == 4
    assert plan()["shape"] == [384,351,301]
    assert plan()["resources"]["physical_cpus"] == 96
    assert plan()["resources"]["memory_GiB"] == 128
    assert plan()["resources"]["work_deadline_s"] == 7080
    assert plan()["acceptance"]["paired_body_median_gain_strictly_greater_than"] == 0
    with pytest.raises(ValueError):
        plan("FOUR_COMPONENT_FACTORY_COST")

@pytest.mark.skipif(not os.environ.get("S46J_ALPHA_ENTRY_REPORT"), reason="opt-in actual W1 only")
def test_alpha_real_micro_entry():
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    report = Path(os.environ["S46J_ALPHA_ENTRY_REPORT"]).resolve(strict=True)
    work = Path(os.environ["S46J_ALPHA_ENTRY_WORK"]).resolve()
    work.mkdir(parents=True, exist_ok=False)
    manifest = report / "BENCHMARK_INPUT_MANIFEST.json"
    if not manifest.exists():
        sg.input_manifest(report, shape=MICRO_SHAPE)
    cases = {}
    for name,role,n,capture in (("R2","R",2,False),("X2","X",2,False),
            ("R8","R",8,False),("X8","X",8,False),("CAPTURE","R",8,True)):
        target = work / name
        argv = [sys.executable,"-s","-B","-m","tools.cpu_s4_6j_alpha_factory.alpha_entry",
            "--report",str(report),"--work",str(target),"--candidate",CANDIDATE,
            "--role",role,"--intervals",str(n),"--local-engineering","--input-manifest",str(manifest)]
        reference = cases.get("R2" if n == 2 else "R8")
        if reference:
            argv += ["--reference-output",reference["output"]]
            if n == 2:
                argv += ["--reference-boundary",reference["boundary_path"],"--reference-stages",reference["stages"]]
        if capture:
            argv += ["--capture"]
        with (work/(name+".stdout.log")).open("xb") as out, (work/(name+".stderr.log")).open("xb") as err:
            process = subprocess.run(argv, env=os.environ.copy(), stdout=out, stderr=err)
        assert process.returncode == 0, f"{name} failed; see {work}"
        row = json.loads((target/"ALPHA_CASE_RECEIPT.json").read_text(encoding="utf-8"))
        assert row["execution_domain"] == "S46J_LOCAL_MICRO"
        assert row["pointwise_calls"] == 38*n
        assert row["alpha_rounds"] == (5*n if role == "X" else 0)
        assert row["route"]["math_dispatches"] == 16*n and row["route"]["fallback"] == 0
        assert row["pointwise_pool"]["pool"]["closed"]
        assert not row["pointwise_pool"]["pool"]["active_thread_ids"]
        if reference:
            assert row["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
            if n == 2:
                assert row["comparisons"]["boundary"]["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
                assert row["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        cases[name] = row
    argv = [sys.executable,"-s","-B","-m","tools.cpu_s4_6j_alpha_factory.alpha_entry",
        "--report",str(report),"--work",str(work/"MECHANISM"),"--candidate",CANDIDATE,
        "--role","R","--intervals","8","--local-engineering","--mechanism",
        "--capture-root",cases["CAPTURE"]["capture_package"]]
    with (work/"MECHANISM.stdout.log").open("xb") as out, (work/"MECHANISM.stderr.log").open("xb") as err:
        process = subprocess.run(argv,env=os.environ.copy(),stdout=out,stderr=err)
    assert process.returncode == 0, f"MECHANISM failed; see {work}"
    mechanism = json.loads((work/"MECHANISM/MECHANISM_RESULT.json").read_text(encoding="utf-8"))
    assert mechanism["all_exact"] and mechanism["inputs"] == 3
    assert mechanism["formal_calls"] == 90 and mechanism["warmup_calls"] == 36
    assert mechanism["formal14_allowed"] is False
    sg.write(work/"LOCAL_ALPHA_ACCEPTANCE.json", dict(status="PASS_W1_ENGINEERING",
        shape=list(MICRO_SHAPE),cases=list(cases),mechanism_exact=True,
        formal14_authorized_by_local_test=False,scientific_qualification=False))

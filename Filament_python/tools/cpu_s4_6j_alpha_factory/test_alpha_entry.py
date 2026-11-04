"""Alpha protocol checks and opt-in actual W1 qualification/capture entry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from . import alpha_entry as entry
from .alpha_protocol import CANDIDATE, MICRO_SHAPE, SHAPE, plan, matrix, pairings

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

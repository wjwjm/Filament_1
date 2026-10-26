"""Same G2/G3 mathematics, sealed original inputs and FULLSIZE_CONFIRMATION."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import sysconfig
import time
from types import FunctionType, SimpleNamespace

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_3g3_smallgrid.adapter import build_role
from tools.cpu_s4_3g3_smallgrid.candidate import unoptimized
from tools.cpu_s4_3g3_smallgrid.scientific_context import context as micro_context
from tools.cpu_s4_checkpoint_support import support_module
from tools.hpc_ops import provenance_v2 as pv

FULL_SHAPE=(384,351,301)
ROLES={"G2_FULL":"G2_SMALL","G3_FULL":"G3_SMALL"}
SOURCE_SHA="52731713b0358c760af8605b7c7f39dd94df500722bdb4e299664c13c04af144"
PRE_SHA={"delta_n":"66ec96c2babba7c9e277bd5e19bb5657fe7945f56062f077cc1853bd39c2dfbe","vx":"054c5f21d0b7b382ba8da3d1d48c51e954d72e1e7fd0b0ba15f7870f4e0ac06b","vy":"054c5f21d0b7b382ba8da3d1d48c51e954d72e1e7fd0b0ba15f7870f4e0ac06b"}

def require(ok,why):
    if not ok:raise RuntimeError(why)

def verify_node_gate_receipt(node, job_id):
    """Validate the R1 node receipt without changing execution or science."""
    require(isinstance(node, dict)
            and node.get("status")=="PASS_FULLSIZE_NODE_SOURCE_INPUT_AND_REFERENCE_METADATA_GATES"
            and node.get("job_id")==job_id,"ACTUAL_FULLSIZE_NODE_GATE_REQUIRED")

def verify_execution(report):
    unoptimized()
    require(os.environ.get("SLURM_JOB_ID","").isdigit(),"FULLSIZE_REQUIRES_ACTUAL_SLURM")
    plan=sg.read(Path(report)/"FULLSIZE_CONFIRMATION_PLAN.json")
    require(plan["execution_domain"]=="FULLSIZE_CONFIRMATION" and plan["shape"]==list(FULL_SHAPE) and plan["intervals"]==8,"EXPLICIT_FULLSIZE_ENTRY8_ONLY")
    require(pv.raw_sha256_file(Path(report)/"FULLSIZE_CONFIRMATION_PLAN.json")==os.environ.get("CPU_G3_FULL_PLAN_SHA256"),"FULL_PLAN_RAW_IDENTITY")
    node=sg.read(Path(report)/"NODE_GATE.json")
    verify_node_gate_receipt(node,os.environ["SLURM_JOB_ID"])
    from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool
    verify_runtime_query_tool(plan)
    repo=Path(__file__).resolve().parents[3]
    require(pv._git(repo,"rev-parse","HEAD")==plan["execution_commit"],"REAL_CHECKPOINT_HEAD")
    require(pv._git(repo,"rev-parse","HEAD:Filament_python/KHz_filament")=="821fec5bdacd832b0347e1f2e84285cb54172e57","FROZEN_SCIENCE_TREE")
    for row in plan["code_records"]:
        path=repo/row["path"]
        require(pv.canonical_lf_sha256(path.read_bytes())==row["canonical_lf_sha256"] and pv._git(repo,"rev-parse","HEAD:"+row["path"])==row["git_blob_oid"],"FULL_COMMITTED_SOURCE_IDENTITY")
    return plan

def full_context(report,*,shape=FULL_SHAPE):
    """Original frozen loader body; only source-metadata binding is relocated."""
    plan=verify_execution(report)
    require(tuple(shape)==FULL_SHAPE,"NO_OTHER_FULLSIZE_SHAPE")
    from KHz_filament.hr4e5_formal_entry import pre0_fields_from_delta_n
    from tools.cpu_p1r import cpu_p1r_optical as original
    root=Path(__file__).resolve().parents[3]
    def current_source_binding(bench_root):
        require(Path(bench_root).resolve()==root,"FULL_SOURCE_ROOT")
        require(pv._git(root,"rev-parse","HEAD:Filament_python/KHz_filament/propagate.py")==original.PROPAGATE_BLOB,"ORIGINAL_PROPAGATE_BLOB")
        return dict(schema="filament.fullsize_checkpoint.source.v1",actual_checkpoint=plan["execution_commit"],science_tree=plan["science_tree"],historical_deployment_head=plan["historical_deployment_head"],metadata_relocation="Real current Git commit; original source loader mathematics unchanged")
    namespace=dict(original._load_context.__globals__)
    namespace["_source_binding"]=current_source_binding
    loader=FunctionType(original._load_context.__code__,namespace,original._load_context.__name__,original._load_context.__defaults__,original._load_context.__closure__)
    ctx=loader(Path(plan["full_input_root"]).resolve(strict=True),root,Path(sysconfig.get_paths()["purelib"]))
    ctx["pre0"]=ctx["assets"]["pre0"]
    ctx["pre_fields"]=pre0_fields_from_delta_n(ctx["pre0"],expected_count=8)
    ctx["source"].setflags(write=False)
    for value in ctx["pre_fields"].values():value.setflags(write=False)
    require(sg.fingerprints(ctx)==dict(source=SOURCE_SHA,PRE=PRE_SHA),"SEALED_ORIGINAL_G2_SOURCE_PRE_CONTENT")
    ctx["benchmark_input_kind"]="ORIGINAL_SEALED_G2_FULLSIZE_SOURCE_AND_PRE"
    return ctx

def full_window(context_function=full_context):
    """Private construction namespace; fixed SG guard is not edited globally."""
    namespace=dict(vars(sg));namespace["context"]=context_function;namespace["SHAPE"]=FULL_SHAPE
    namespace["_run_window"]=FunctionType(sg._run_window.__code__,namespace,sg._run_window.__name__,sg._run_window.__defaults__,sg._run_window.__closure__)
    namespace["_run_window"].__kwdefaults__=sg._run_window.__kwdefaults__
    result=FunctionType(sg.run_window.__code__,namespace,sg.run_window.__name__,sg.run_window.__defaults__,sg.run_window.__closure__)
    result.__kwdefaults__=sg.run_window.__kwdefaults__
    return result

def run(args):
    service_started=time.perf_counter()
    app_started=time.perf_counter()
    unoptimized();require(args.role in ROLES,"TWO_FULLSIZE_ROLES_ONLY")
    plan=verify_execution(args.report)
    require(plan["warmup_intervals_per_formal_case"]==0,"PREREGISTERED_SYMMETRIC_NO_EXTRA_WARMUP")
    report=Path(args.report).resolve(strict=True);work=Path(args.work).resolve()
    require(not work.exists(),"NEW_FULL_CASE_REQUIRED");work.mkdir(parents=True)
    support_module("g2_hpc_entry")._quiet_runtime()
    internal=ROLES[args.role]
    built=build_role(internal,args.mode,capture_root=work/"BOUNDARY",local_engineering=False)
    factory_finished=time.perf_counter()
    request=SimpleNamespace(report=report,work=work,role=internal,mode=args.mode,intervals=8,local_engineering=False,
        input_manifest=report/"FULL_INPUT_MANIFEST.json",reference_output=args.reference_output,
        reference_boundary=args.reference_boundary,reference_stages=args.reference_stages,
        execution_domain="FULLSIZE_CONFIRMATION",reported_role=args.role)
    result=full_window()(request,built,label="formal",intervals=8,capture=args.mode=="QUALIFY_BOUNDARY",application_started=app_started)
    result.update(role=args.role,execution_domain="FULLSIZE_CONFIRMATION",factory_role=internal,
        application_service_s=result["optical_application_window_s"],initialization_within_application_s=factory_finished-app_started,
        research_service_s=time.perf_counter()-service_started,execution_commit=plan["execution_commit"],warmup_intervals=0,
        fullsize_new_candidate="EXECUTED_ENTRY8" if args.role=="G3_FULL" else "G2_REFERENCE_EXECUTED_ENTRY8",
        scope="FULLSIZE_ENTRY_EIGHT_ONLY",production_qualified=False,default="I0",review_gate="MANUAL_REVIEW_PENDING")
    sg.write(work/"CASE_RECEIPT.json",result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True);parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--role",choices=tuple(ROLES),required=True)
    parser.add_argument("--mode",choices=("QUALIFY_BOUNDARY","PURE_OFF_SMALLGRID"),required=True)
    for flag in ("reference-output","reference-boundary","reference-stages"):parser.add_argument("--"+flag,type=Path)
    result=run(parser.parse_args());print(json.dumps(dict(status=result["status"],role=result["role"],application_service_s=result["application_service_s"])))
    return 0

if __name__=="__main__":raise SystemExit(main())

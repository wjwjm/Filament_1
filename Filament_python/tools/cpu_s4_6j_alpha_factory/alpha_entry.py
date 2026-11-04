"""Source-bound alpha R0/X windows and the separate capture mechanism process."""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path
from types import FunctionType

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from . import entry as cost_entry
from . import alpha_role_spec as roles
from .alpha_capture import AlphaCaptureStore, capture_source_binding, load_capture
from .alpha_measurement import run_window
from .alpha_protocol import BASE, TASK, SCIENCE_TREE, SHAPE, MICRO_SHAPE, CANDIDATE, candidate_spec, plan, exception_classification

read, require = cost_entry.read, cost_entry.require
FULL_GATE_STATUS = "PASS_S46J_ALPHA_FULLSIZE_SOURCE_RESOURCE_INPUT_GATES"

def verify_source(*args, **kwargs):
    namespace = dict(cost_entry.verify_source.__globals__, BASE=BASE, TASK=TASK,
        SCIENCE_TREE=SCIENCE_TREE, SHAPE=SHAPE, MICRO_SHAPE=MICRO_SHAPE,
        candidate_spec=candidate_spec, plan=plan, FULL_GATE_STATUS=FULL_GATE_STATUS)
    binding = FunctionType(cost_entry.verify_source.__code__, namespace)(*args, **kwargs)
    report = Path(args[0] if args else kwargs["report"]).resolve(strict=True)
    selection = read(report / "REFERENCE_SELECTION.json")
    require(selection["status"] == "R0_SELECTED_NO_QUALIFIED_FACTORY_REPAIR"
        and selection["factory_repair_selected"] is False, "S46J_ALPHA_R0_REFERENCE")
    require(pv.raw_sha256_file(report / "REFERENCE_SELECTION.json") ==
        binding.get("reference_selection_raw_sha256"), "S46J_ALPHA_REFERENCE_IDENTITY")
    return binding

def capture_binding(binding):
    return dict(execution_commit=binding["execution_commit"],
        source_manifest_raw_sha256=binding["source_manifest_raw_sha256"],
        input_contract_raw_sha256=binding["input_contract_raw_sha256"],
        reference_selection_raw_sha256=binding["reference_selection_raw_sha256"],
        capture_source=capture_source_binding())

def run(args):
    args.cost_mode = "OFF"
    args.shape = MICRO_SHAPE if args.local_engineering else SHAPE
    candidate_spec(args.candidate)
    capture = bool(getattr(args, "capture", False))
    mechanism = bool(getattr(args, "mechanism", False))
    require(not (capture and mechanism), "S46J_CAPTURE_MECHANISM_EXCLUSIVE")
    require(not (capture or mechanism) or (args.role == "R" and args.intervals == 8),
        "S46J_ALPHA_CAPTURE_R8_ONLY")
    if mechanism:
        require(args.capture_root is not None, "S46J_ALPHA_CAPTURE_INPUT_REQUIRED")
        binding = verify_source(args.report, candidate=CANDIDATE, local_engineering=args.local_engineering,
            shape=args.shape, intervals=8, mode="PURE_OFF_FULLSIZE")
        require(not args.work.exists(), "S46J_NON_OVERWRITING_MECHANISM")
        args.work.mkdir(parents=True)
        from .alpha_mechanism import run as mechanism_run
        return mechanism_run(args.capture_root, args.work / "MECHANISM_RESULT.json",
            shape=args.shape, execution_commit=binding["execution_commit"],
            local_engineering=args.local_engineering, expected_binding=capture_binding(binding))
    require(capture or args.capture_root is None, "S46J_UNUSED_CAPTURE_ROOT")
    # Defer capture directory creation until after source validation/work creation.
    capture_holder = {}
    def build(semantic, mode, **kwargs):
        if capture:
            root = args.capture_root or args.work / "ALPHA_CAPTURE"
            store = AlphaCaptureStore(root, shape=args.shape)
            capture_holder["store"] = store
            capture_holder["root"] = Path(root)
            kwargs["capture_store"] = store
        return roles.build_role(semantic, mode, **kwargs)
    namespace = dict(cost_entry.run.__globals__, candidate_spec=candidate_spec,
        verify_source=verify_source, build_role=build, verify_role=roles.verify_role,
        close_role=roles.close_role, run_window=run_window)
    namespace["admission"] = FunctionType(cost_entry.admission.__code__, namespace)
    result = FunctionType(cost_entry.run.__code__, namespace)(args)
    result.update(scope="LOCAL_W1_ENGINEERING_ONLY" if args.local_engineering else "ORIGINAL_SIZE_EIGHT_INTERVAL_STUDY",
        capture_enabled=capture, alpha_enabled=args.role == "X")
    if capture:
        # SOURCE_VALIDATION already verified the complete execution before work.
        binding = read(Path(args.report) / "EXECUTION.json")
        package = capture_holder["store"].finish(binding=capture_binding(binding))
        result.update(capture_package=str(capture_holder["root"]),
            capture_states=package["captured_states"], capture_arrays=package["array_file_count"],
            capture_manifest_raw_sha256=pv.raw_sha256_file(capture_holder["root"] / "CAPTURE.json"),
            capture_source_binding=package["source_binding"])
    # Distinct receipt preserves the inherited window receipt and its clock.
    sg.write(args.work / "ALPHA_CASE_RECEIPT.json", result)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--candidate", choices=(CANDIDATE,), default=CANDIDATE)
    parser.add_argument("--role", choices=("R","X"), required=True)
    parser.add_argument("--intervals", type=int, choices=(2,8), required=True)
    parser.add_argument("--local-engineering", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    for flag in ("reference-output","reference-boundary","reference-stages","input-manifest"):
        parser.add_argument("--"+flag, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(dict(status=result["status"], candidate=CANDIDATE)))
        return 0
    except BaseException as error:
        if args.work.is_dir() and not (args.work / "ENTRY_FAILURE.json").exists():
            sg.write(args.work / "ENTRY_FAILURE.json", dict(classification=exception_classification(error),
                error=repr(error), traceback=traceback.format_exc(), notes=list(getattr(error,"__notes__",()))))
        raise

if __name__ == "__main__":
    raise SystemExit(main())

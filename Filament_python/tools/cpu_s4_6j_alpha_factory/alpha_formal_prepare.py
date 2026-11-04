"""Prepare a fresh formal report bound to the admitted alpha mechanism."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import FunctionType
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv

from . import alpha_mechanism_admission, alpha_prepare
from .alpha_formal_protocol import (
    BASE,
    CANDIDATE,
    FORMAL_SCHEMA,
    TASK,
    candidate_spec,
    plan,
)


SCHEMA = FORMAL_SCHEMA.replace("formal_protocol", "formal_prepare")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("S46J_FORMAL_PREPARE_OBJECT:" + path.name)
    return value


def _bound_alpha_prepare() -> FunctionType:
    """Bind the existing 6J packager to this formal protocol.

    The underlying packager still owns the PV2 preparation and candidate-path
    closure.  Only its protocol globals are rebound here, matching the 6I
    formal-preparation pattern; no generic cache or alternate hash semantic is
    introduced.
    """

    namespace = dict(alpha_prepare.prepare.__globals__)
    namespace.update(
        __file__=__file__,
        BASE=BASE,
        TASK=TASK,
        candidate_spec=candidate_spec,
        plan=plan,
    )
    function = FunctionType(
        alpha_prepare.prepare.__code__,
        namespace,
        name="formal_alpha_prepare",
        argdefs=alpha_prepare.prepare.__defaults__,
        closure=alpha_prepare.prepare.__closure__,
    )
    function.__kwdefaults__ = alpha_prepare.prepare.__kwdefaults__
    return function


def prepare(
    report: str | Path,
    *,
    mechanism_admission_path: str | Path,
    reference_selection: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    requested_candidate = str(kwargs.pop("candidate", CANDIDATE)).upper()
    if requested_candidate != CANDIDATE:
        raise ValueError("S46J_FORMAL_CANDIDATE_ONLY")
    local_engineering = bool(kwargs.pop("local_engineering", False))
    report_path = Path(report).resolve()
    if report_path.exists():
        raise FileExistsError("S46J_FORMAL_NON_OVERWRITING_PREPARATION")
    admission_path = Path(mechanism_admission_path).resolve(strict=True)
    selection_path = Path(reference_selection).resolve(strict=True)
    if admission_path.name != "MECHANISM_ADMISSION.json":
        raise ValueError("S46J_FORMAL_ADMISSION_BASENAME")
    if selection_path.name != "REFERENCE_SELECTION.json":
        raise ValueError("S46J_FORMAL_REFERENCE_SELECTION_BASENAME")

    # alpha_prepare's rebound 6J packager supplies candidate=CANDIDATE to its
    # underlying 6F function.  Do not pass candidate or local_engineering a
    # second time into that call; the latter is an admission-only switch.
    binding = _bound_alpha_prepare()(
        report_path,
        reference_selection=selection_path,
        **dict(kwargs),
    )
    target_admission = report_path / "MECHANISM_ADMISSION.json"
    if target_admission.exists():
        raise FileExistsError("S46J_FORMAL_ADMISSION_TARGET_EXISTS")
    shutil.copyfile(admission_path, target_admission)
    admission_raw_sha256 = pv.raw_sha256_file(target_admission)
    admission = _read_object(target_admission)
    selection = _read_object(report_path / "REFERENCE_SELECTION.json")
    if selection.get("status") != "R0_SELECTED_NO_QUALIFIED_FACTORY_REPAIR" or selection.get("factory_repair_selected") is not False:
        raise RuntimeError("S46J_FORMAL_R0_REFERENCE_SELECTION")

    plan_path = report_path / "HPC_EXECUTION_PLAN.json"
    hpc_plan = _read_object(plan_path)
    hpc_plan["work_deadline_s"] = 7080
    hpc_plan["walltime_s"] = 7200
    hpc_plan["formal_plan_schema"] = plan()["schema"]
    plan_path.write_text(json.dumps(hpc_plan, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    hpc_plan_sha256 = pv.raw_sha256_file(plan_path)

    execution_path = report_path / "EXECUTION.json"
    execution = _read_object(execution_path)
    execution.update(
        {
            "formal_phase": "FORMAL_ALPHA_FOURTEEN",
            "formal_plan_schema": plan()["schema"],
            "baseline_commit": BASE,
            "candidate": CANDIDATE,
            "mechanism_admission_raw_sha256": admission_raw_sha256,
            "mechanism_admission_path": "MECHANISM_ADMISSION.json",
            "capture_execution_commit": admission.get("capture_execution_commit"),
            "capture_result_raw_sha256": admission.get("capture_result_raw_sha256"),
            "capture_manifest_raw_sha256": admission.get("capture_manifest_raw_sha256"),
            "capture_source_manifest_raw_sha256": admission.get("capture_source_manifest_raw_sha256"),
            "input_contract_raw_sha256": admission.get("input_contract_raw_sha256"),
            "reference_selection_raw_sha256": pv.raw_sha256_file(report_path / "REFERENCE_SELECTION.json"),
            "hpc_plan_raw_sha256": hpc_plan_sha256,
            "work_deadline_s": 7080,
            "production_qualified": False,
        }
    )
    execution_path.write_text(json.dumps(execution, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    binding = dict(binding)
    binding.update(execution)
    binding["execution_raw_sha256"] = pv.raw_sha256_file(execution_path)
    admission_receipt = alpha_mechanism_admission.verify(
        report_path,
        repo=Path(__file__).resolve().parents[3],
        local_engineering=local_engineering,
    )
    binding["mechanism_admission"] = admission_receipt
    sg.write(
        report_path / "FORMAL_PREPARE_RECEIPT.json",
        {
            "schema": SCHEMA + ".receipt.v1",
            "task": TASK,
            "candidate": CANDIDATE,
            "formal_phase": "FORMAL_ALPHA_FOURTEEN",
            "mechanism_admission_raw_sha256": admission_raw_sha256,
            "capture_manifest_raw_sha256": admission.get("capture_manifest_raw_sha256"),
            "hpc_plan_raw_sha256": hpc_plan_sha256,
            "mechanism_admission": admission_receipt,
        },
    )
    return binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mechanism-admission", dest="mechanism_admission_path", type=Path, required=True)
    parser.add_argument("--reference-selection", type=Path, required=True)
    parser.add_argument("--inherited-manifest", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--full-input-root", type=str, required=True)
    parser.add_argument("--canonical-input-manifest", type=str, required=True)
    parser.add_argument("--reference-output", type=str, required=True)
    parser.add_argument("--reference-boundary", type=str)
    parser.add_argument("--reference-stages", type=str)
    parser.add_argument("--canonical-manifest-sha256")
    parser.add_argument("--local-engineering", action="store_true")
    args = parser.parse_args(argv)
    prepare(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "main", "prepare"]

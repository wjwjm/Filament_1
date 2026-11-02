"""Formal fourteen-window entry bound to the frozen selected entry."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import FunctionType
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg

from . import mechanism_admission, selected_entry
from .formal_protocol import (
    BASE,
    CANDIDATE,
    FORMAL_SCHEMA,
    MICRO_SHAPE,
    PHASE,
    SCIENCE_TREE,
    SHAPE,
    TASK,
    candidate_spec,
    exception_classification,
    plan,
)


SCHEMA = FORMAL_SCHEMA.replace("formal_protocol", "formal_entry")


def _clone(function: Any, namespace: dict[str, Any], name: str | None = None) -> FunctionType:
    clone = FunctionType(
        function.__code__,
        namespace,
        name=name or function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    clone.__doc__ = function.__doc__
    return clone


def _bound_verify_source() -> FunctionType:
    namespace = dict(selected_entry.verify_source.__globals__)
    namespace.update(
        BASE=BASE,
        TASK=TASK,
        SCIENCE_TREE=SCIENCE_TREE,
        SHAPE=SHAPE,
        MICRO_SHAPE=MICRO_SHAPE,
        CANDIDATE=CANDIDATE,
        candidate_spec=candidate_spec,
        plan=plan,
    )
    return _clone(selected_entry.verify_source, namespace, "formal_verify_source_selected")


def verify_source(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run selected source verification, then require mechanism admission."""

    bound = _bound_verify_source()
    binding = bound(*args, **kwargs)
    report = kwargs.get("report") if "report" in kwargs else (args[0] if args else None)
    local_engineering = bool(kwargs.get("local_engineering", False))
    admission = mechanism_admission.verify(
        report,
        repo=Path(__file__).resolve().parents[3],
        local_engineering=local_engineering,
    )
    result = dict(binding)
    result["formal_phase"] = PHASE
    result["mechanism_admission"] = admission
    result["mechanism_admission_raw_sha256"] = admission["admission_raw_sha256"]
    return result


def _formal_run_function() -> FunctionType:
    namespace = dict(selected_entry.run.__globals__)
    namespace.update(
        BASE=BASE,
        TASK=TASK,
        CANDIDATE=CANDIDATE,
        candidate_spec=candidate_spec,
        plan=plan,
        verify_source=verify_source,
    )
    # ``args.capture`` is forced off by ``run`` below, so no selected
    # mechanism CaptureStore is constructed.  Leave the inherited R2
    # qualification boundary argument intact: it is the original window
    # contract, not the prohibited mechanism capture package.
    return _clone(selected_entry.run, namespace, "formal_selected_run")


def run(args: Any) -> dict[str, Any]:
    """Run one formal R/X window through a private rebound selected entry."""

    args.capture = False
    args.mechanism = False
    result = _formal_run_function()(args)
    formal = dict(result)
    formal.update(
        {
            "schema": SCHEMA + ".case.v1",
            "formal_phase": PHASE,
            "task": TASK,
            "baseline_commit": BASE,
            "candidate": CANDIDATE,
            "capture_enabled": False,
            "mechanism_enabled": False,
            "formal_plan_schema": plan()["schema"],
            "selected_operation_groups_contract": {"2": 16, "8": 64},
            "original_component_groups": {"2": 60, "8": 240},
            "source_math_changed": False,
        }
    )
    sg.write(Path(args.work) / "FORMAL_CASE_RECEIPT.json", formal)
    return formal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--candidate", choices=(CANDIDATE,), default=CANDIDATE)
    parser.add_argument("--role", choices=("R", "X"), required=True)
    parser.add_argument("--intervals", type=int, choices=(2, 8), required=True)
    parser.add_argument("--local-engineering", action="store_true")
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--reference-boundary", type=Path)
    parser.add_argument("--reference-stages", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args(argv)
    args.shape = MICRO_SHAPE if args.local_engineering else SHAPE
    args.capture = False
    args.mechanism = False
    try:
        result = run(args)
        print(json.dumps({key: result.get(key) for key in ("status", "candidate", "role", "intervals", "capture_enabled")}, sort_keys=True))
        return 0
    except BaseException as error:
        if args.work.is_dir() and not (args.work / "FORMAL_ENTRY_FAILURE.json").exists():
            sg.write(
                args.work / "FORMAL_ENTRY_FAILURE.json",
                {
                    "schema": SCHEMA + ".failure.v1",
                    "classification": exception_classification(error),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "main", "run", "verify_source"]

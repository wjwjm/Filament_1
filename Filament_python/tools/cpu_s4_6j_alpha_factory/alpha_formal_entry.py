"""Formal fourteen-window entry after independently admitted alpha mechanism."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import FunctionType
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg

from . import alpha_entry, alpha_mechanism_admission
from .alpha_formal_protocol import (
    BASE,
    CANDIDATE,
    FORMAL_SCHEMA,
    MICRO_SHAPE,
    PHASE,
    SHAPE,
    TASK,
    candidate_spec,
    exception_classification,
    plan,
)


SCHEMA = FORMAL_SCHEMA.replace("formal_protocol", "formal_entry")
FULL_GATE_STATUS = "PASS_S46J_ALPHA_FORMAL_SOURCE_RESOURCE_INPUT_GATES"
read = alpha_entry.read


def _bind(function: Any, **overrides: Any) -> FunctionType:
    namespace = dict(function.__globals__, BASE=BASE, TASK=TASK,
        SHAPE=SHAPE, MICRO_SHAPE=MICRO_SHAPE, CANDIDATE=CANDIDATE,
        plan=plan, candidate_spec=candidate_spec, FULL_GATE_STATUS=FULL_GATE_STATUS)
    namespace.update(overrides)
    bound = FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def verify_source(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = kwargs.get("report") if "report" in kwargs else (args[0] if args else None)
    local_engineering = bool(kwargs.get("local_engineering", False))
    binding = _bind(alpha_entry.verify_source)(*args, **kwargs)
    admission = alpha_mechanism_admission.verify(
        report,
        repo=Path(__file__).resolve().parents[3],
        local_engineering=local_engineering,
    )
    result = dict(binding)
    result.update(
        formal_phase=PHASE,
        formal_plan_schema=plan()["schema"],
        mechanism_admission=admission,
        mechanism_admission_raw_sha256=admission["admission_raw_sha256"],
    )
    return result


def run(args: Any) -> dict[str, Any]:
    candidate_spec(args.candidate)
    args.capture = False
    args.mechanism = False
    args.capture_root = None
    verified: dict[str, Any] = {}

    def verify_current(*verify_args: Any, **verify_kwargs: Any) -> dict[str, Any]:
        binding = verify_source(*verify_args, **verify_kwargs)
        verified.update(binding)
        return binding

    # Rebind privately so formal admission runs at the original source boundary
    # inside APPLICATION, with no extra pre-clock validation or cached PASS.
    result = _bind(alpha_entry.run, verify_source=verify_current)(args)
    binding = verified
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
            "source_math_changed": False,
            "production_qualified": False,
            "mechanism_admission": binding["mechanism_admission"],
            "formal_execution_commit": binding["execution_commit"],
            "capture_execution_commit": binding["mechanism_admission"]["capture_execution_commit"],
            "mechanism_admission_raw_sha256": binding["mechanism_admission_raw_sha256"],
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
    args.capture_root = None
    try:
        result = run(args)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in ("status", "candidate", "role", "intervals", "capture_enabled")
                },
                sort_keys=True,
            )
        )
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

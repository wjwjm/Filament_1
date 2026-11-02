"""Create a non-submitting S4-6I formal fourteen-window package.

The selected capture package remains the source of the preparation mechanics;
this module only rebinds its protocol/source closure and adds the externally
produced mechanism-admission receipt.  No capture or scheduler side effect is
performed here.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import FunctionType
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv

from . import mechanism_admission as mechanism_admission_module, selected_prepare
from .formal_protocol import (
    BASE,
    CANDIDATE,
    FORMAL_SCHEMA,
    SCIENCE_TREE,
    SHAPE,
    TASK,
    candidate_spec,
    plan,
)


SCHEMA = FORMAL_SCHEMA.replace("formal_protocol", "formal_prepare")


def _candidate_paths(repo: Path) -> set[str]:
    """Extend the selected closure with every formal module in this package."""

    names = set(selected_prepare._candidate_paths(repo))
    package = Path(__file__).resolve().parent
    names.update(
        path.relative_to(repo).as_posix()
        for path in package.glob("*.py")
        if path.is_file()
    )
    return names


def _bound_selected_prepare() -> FunctionType:
    namespace = dict(selected_prepare.prepare.__globals__)
    namespace.update(
        __file__=__file__,
        BASE=BASE,
        TASK=TASK,
        SCIENCE_TREE=SCIENCE_TREE,
        SHAPE=SHAPE,
        candidate_spec=candidate_spec,
        plan=plan,
        _candidate_paths=_candidate_paths,
    )
    function = FunctionType(
        selected_prepare.prepare.__code__,
        namespace,
        name="formal_selected_prepare",
        argdefs=selected_prepare.prepare.__defaults__,
        closure=selected_prepare.prepare.__closure__,
    )
    function.__kwdefaults__ = selected_prepare.prepare.__kwdefaults__
    return function


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"S46I_FORMAL_PREPARE_OBJECT:{path.name}")
    return value


def prepare(
    report: str | Path,
    *,
    mechanism_admission_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare a fresh report and bind a verified admission artifact."""

    source = Path(mechanism_admission_path).resolve(strict=True)
    if source.name != "MECHANISM_ADMISSION.json":
        raise ValueError("S46I_FORMAL_MECHANISM_ADMISSION_BASENAME")

    repo = Path(__file__).resolve().parents[3]
    report_path = Path(report).resolve()
    if report_path.exists():
        raise FileExistsError("S46I_FORMAL_NON_OVERWRITING_PREPARATION")

    requested_candidate = str(kwargs.pop("candidate", CANDIDATE)).upper()
    if requested_candidate != CANDIDATE:
        raise ValueError("S46I_FORMAL_CANDIDATE_ONLY")
    local_engineering = bool(kwargs.pop("local_engineering", False))
    # selected_prepare itself supplies candidate=CANDIDATE to its rebound 6F
    # packager; passing a second candidate would create a duplicate keyword.
    binding = _bound_selected_prepare()(report_path, **dict(kwargs))

    target = report_path / "MECHANISM_ADMISSION.json"
    if target.exists():
        raise FileExistsError("S46I_FORMAL_ADMISSION_TARGET_EXISTS")
    shutil.copyfile(source, target)

    admission_raw_sha256 = pv.raw_sha256_file(target)
    admission = _read_object(target)
    execution_path = report_path / "EXECUTION.json"
    execution = _read_object(execution_path)
    execution.update(
        {
            "formal_phase": "FORMAL_FOURTEEN",
            "formal_plan_schema": plan()["schema"],
            "mechanism_admission_raw_sha256": admission_raw_sha256,
            "mechanism_admission_path": "MECHANISM_ADMISSION.json",
            "mechanism_admission_status": admission.get("status"),
            "capture_source_manifest_raw_sha256": admission.get("capture_source_manifest_raw_sha256"),
            "input_contract_raw_sha256": admission.get("input_contract_raw_sha256"),
        }
    )
    execution_path.write_text(
        json.dumps(execution, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    binding = dict(binding)
    binding.update(execution)
    binding["execution_raw_sha256"] = pv.raw_sha256_file(execution_path)
    # Recheck the newly bound package, including PV2 source identities.  The
    # verifier intentionally performs no remote reads or array loading.
    receipt = mechanism_admission_module.verify(report_path, repo=repo, local_engineering=local_engineering)
    binding["mechanism_admission"] = receipt
    sg.write(report_path / "FORMAL_PREPARE_RECEIPT.json", {
        "schema": SCHEMA + ".receipt.v1",
        "task": TASK,
        "candidate": CANDIDATE,
        "formal_phase": "FORMAL_FOURTEEN",
        "mechanism_admission_raw_sha256": admission_raw_sha256,
        "mechanism_admission": receipt,
    })
    return binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mechanism-admission", dest="mechanism_admission_path", type=Path, required=True)
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

"""Prepare a C14 propagation report from an independently admitted mechanism."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv

from .c_formal14_evidence import copy_mechanism_export
from .c_formal14_protocol import BASE, EXECUTION_SCHEMA, HPC_EXECUTION_SCHEMA, PLAN_SCHEMA, REMOTE_PYTHON, RESOURCES, SCIENCE_TREE, TASK, plan


def _require(ok: bool, reason: str, detail=None) -> None:
    if not ok:
        raise RuntimeError(reason if detail is None else f"{reason}:{detail}")


def _source_paths(repo: Path, inherited_manifest: Path) -> list[str]:
    inherited = sg.read(inherited_manifest)
    paths = {
        str(row["path"])
        for row in pv.iter_records(inherited, require_hash_scope=True)
        if row.get("classification") == "tracked_text"
    }
    owned = Path(__file__).parent
    paths.update(
        path.relative_to(repo).as_posix()
        for path in owned.iterdir()
        if path.suffix in (".py", ".sbatch", ".md")
    )
    paths.update(("Filament_python/tools/run_local_tests.ps1", "Filament_python/tools/hpc_ops/audit_batch_entry.py"))
    return sorted(paths)


def _write_hpc_plan(report: Path, *, execution_commit: str, source_root: str, stage_root: str, query_git_launcher: str | None) -> dict:
    value = {
        "schema": HPC_EXECUTION_SCHEMA,
        "source_root": source_root,
        "python": REMOTE_PYTHON,
        "system_git": "/usr/bin/git",
        "query_git_launcher": query_git_launcher or "/usr/bin/git",
        "execution_commit": execution_commit,
        "stage_root": stage_root,
        "report_root": str(report),
        "resources": copy.deepcopy(RESOURCES),
        "submission_authorized": False,
    }
    sg.write(report / "HPC_EXECUTION_PLAN.json", value)
    return value


def prepare(
    report: str | Path,
    *,
    mechanism_evidence: str | Path,
    inherited_manifest: str | Path,
    source_root: str = "",
    stage_root: str = "",
    query_git_launcher: str | None = None,
) -> dict:
    repo = Path(__file__).resolve().parents[3]
    pv._ensure_clean(repo)
    _require(pv._git(repo, "merge-base", BASE, "HEAD") == BASE, "C14_DESCENDANT_OF_MECHANISM_EXECUTION")
    _require(pv._git(repo, "rev-parse", "HEAD:Filament_python/KHz_filament") == SCIENCE_TREE, "C14_FROZEN_SCIENCE")
    report = Path(report).resolve()
    _require(not report.exists(), "C14_NON_OVERWRITING_PREPARATION")
    evidence = Path(mechanism_evidence).resolve(strict=True)
    report.mkdir(parents=True, exist_ok=False)
    admission = copy_mechanism_export(evidence, report / "evidence_C_mechanism")
    for name in ("INPUT_CONTRACT.json", "REFERENCE_INPUT_RAW_RECORDS.json"):
        shutil.copyfile(evidence / name, report / name)
    sg.write(report / "MECHANISM_ADMISSION.json", admission)
    sg.write(report / "PLAN.json", plan())
    source_manifest = repo.parent / (repo.name + "-" + report.name + "-SOURCE_MANIFEST.json")
    _require(not source_manifest.exists(), "C14_NEW_EXTERNAL_MANIFEST_TARGET")
    manifest = pv.create_manifest(repo, source_manifest, _source_paths(repo, Path(inherited_manifest).resolve(strict=True)), [])
    shutil.copyfile(source_manifest, report / "EXECUTION_SOURCE_MANIFEST.json")
    pv.validate_manifest(repo, report / "EXECUTION_SOURCE_MANIFEST.json", require_clean=True, require_hash_scope=True)
    input_contract = sg.read(report / "INPUT_CONTRACT.json")
    reference_contract = sg.read(report / "REFERENCE_INPUT_RAW_RECORDS.json")
    hpc_plan = _write_hpc_plan(
        report,
        execution_commit=manifest["head"],
        source_root=source_root or str(repo.parent),
        stage_root=stage_root or str(repo.parent),
        query_git_launcher=query_git_launcher,
    )
    execution = dict(
        schema=EXECUTION_SCHEMA,
        task=TASK,
        baseline_commit=BASE,
        mechanism_execution_commit=BASE,
        science_tree=SCIENCE_TREE,
        execution_commit=manifest["head"],
        execution_branch=manifest["branch"],
        plan_raw_sha256=pv.raw_sha256_file(report / "PLAN.json"),
        source_manifest_raw_sha256=pv.raw_sha256_file(report / "EXECUTION_SOURCE_MANIFEST.json"),
        inherited_manifest_raw_sha256=pv.raw_sha256_file(Path(inherited_manifest).resolve(strict=True)),
        input_contract_raw_sha256=pv.raw_sha256_file(report / "INPUT_CONTRACT.json"),
        reference_contract_raw_sha256=pv.raw_sha256_file(report / "REFERENCE_INPUT_RAW_RECORDS.json"),
        hpc_plan_raw_sha256=pv.raw_sha256_file(report / "HPC_EXECUTION_PLAN.json"),
        mechanism_admission_raw_sha256=pv.raw_sha256_file(report / "MECHANISM_ADMISSION.json"),
        mechanism_result_raw_sha256=admission["mechanism_result_raw_sha256"],
        mechanism_source_manifest_raw_sha256=admission["mechanism_source_manifest_raw_sha256"],
        submission_authorized=False,
        default="I0",
        production_qualified=False,
        review_gate="MANUAL_REVIEW_PENDING",
    )
    sg.write(report / "EXECUTION.json", execution)
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mechanism-evidence", type=Path, required=True)
    parser.add_argument("--inherited-manifest", type=Path, required=True)
    parser.add_argument("--source-root", default="")
    parser.add_argument("--stage-root", default="")
    parser.add_argument("--query-git-launcher")
    args = parser.parse_args()
    result = prepare(
        args.report,
        mechanism_evidence=args.mechanism_evidence,
        inherited_manifest=args.inherited_manifest,
        source_root=args.source_root,
        stage_root=args.stage_root,
        query_git_launcher=args.query_git_launcher,
    )
    print(result["execution_commit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

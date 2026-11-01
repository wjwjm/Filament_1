"""Create a non-submitting CPU-S4-6F source/input execution package."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv

from .entry import read, require
from .protocol import BASE, SCIENCE_TREE, SHAPE, TASK, candidate_spec, plan


def _external_path(value: str | Path) -> str:
    # Keep remote POSIX paths byte-for-byte. The CLI deliberately passes
    # these values as strings: on Windows, Path('/publicfs/...') first
    # becomes '\\publicfs\\...' and would otherwise be rebased locally.
    text = value if isinstance(value, str) else str(value)
    if text.startswith("/"):
        return text
    if isinstance(value, Path) and text.startswith("\\"):
        raise ValueError("S46F_EXTERNAL_PATH_LOST_POSIX_ROOT")
    return str(Path(text).resolve())


def _candidate_paths(repo: Path) -> set[str]:
    names = {
        "Filament_python/tools/cpu_s4_6a/entry.py",
        "Filament_python/tools/cpu_s4_6a/inputs.py",
        "Filament_python/tools/cpu_s4_6a/records.py",
        "Filament_python/tools/cpu_s4_4b_fullsize_linear_cache/candidate.py",
        "Filament_python/tools/cpu_s4_4b_fullsize_linear_cache/cache.py",
        "Filament_python/tools/cpu_s4_4b_fullsize_linear_cache/contract.py",
        "Filament_python/tools/cpu_s4_6d_eq10_propagation/candidate.py",
        "Filament_python/tools/cpu_s4_6d_eq10_propagation/instrument.py",
        "Filament_python/tools/cpu_s4_6d_eq10_propagation/measurement.py",
        "Filament_python/tools/cpu_s4_6e_array_propagation/candidate.py",
        "Filament_python/tools/cpu_s4_6e_array_propagation/instrument.py",
        "Filament_python/tools/cpu_s4_6e_array_propagation/measurement.py",
        "Filament_python/tools/cpu_s4_6e_array_propagation/c_candidate.py",
        "Filament_python/tools/cpu_s4_6e_array_propagation/c_instrument.py",
        "Filament_python/tools/cpu_s4_6e_array_propagation/c_pool.py",
        "Filament_python/tools/cpu_s4_6e_array_propagation/capture.py",
        "Filament_python/tools/run_local_tests.ps1",
        "Filament_python/tools/hpc_ops/audit_batch_entry.py",
    }
    return {name for name in names if (repo / name).is_file()}


def prepare(
    report: str | Path,
    *,
    candidate: str,
    inherited_manifest: str | Path,
    reference_contract: str | Path,
    input_plan: str | Path,
    full_input_root: str | Path,
    canonical_input_manifest: str | Path,
    reference_output: str | Path,
    reference_boundary: str | Path | None = None,
    reference_stages: str | Path | None = None,
    canonical_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    spec = candidate_spec(candidate)
    pv._ensure_clean(repo)
    require(pv._git(repo, "rev-parse", "HEAD:Filament_python/KHz_filament") == SCIENCE_TREE, "S46F_FROZEN_SCIENCE")
    require(pv._git(repo, "merge-base", BASE, "HEAD") == BASE, "S46F_DESCENDANT_OF_APPROVED_BASE")
    report = Path(report).resolve()
    require(not report.exists(), "S46F_NON_OVERWRITING_PREPARATION")
    inherited = read(inherited_manifest)
    paths = {row["path"] for row in pv.iter_records(inherited, require_hash_scope=True) if row["classification"] == "tracked_text"}
    paths.update(_candidate_paths(repo))
    paths.update(path.relative_to(repo).as_posix() for path in Path(__file__).parent.iterdir() if path.suffix in (".py", ".sbatch", ".md"))
    require(len(paths) >= 252, "S46F_INHERITED_SOURCE_CLOSURE")
    report.mkdir(parents=True)
    sg.write(report / "PLAN.json", plan(candidate))
    source_manifest = repo.parent / (repo.name + "-" + report.name + "-SOURCE_MANIFEST.json")
    require(not source_manifest.exists(), "S46F_NEW_EXTERNAL_MANIFEST_TARGET")
    manifest = pv.create_manifest(repo, source_manifest, sorted(paths), [])
    shutil.copyfile(source_manifest, report / "EXECUTION_SOURCE_MANIFEST.json")
    pv.validate_manifest(repo, report / "EXECUTION_SOURCE_MANIFEST.json", require_clean=True, require_hash_scope=True)
    input_source = read(input_plan)
    canonical_root = _external_path(full_input_root)
    canonical_manifest_hash = canonical_manifest_sha256 or pv.raw_sha256_file(canonical_input_manifest)
    sg.write(report / "INPUT_CONTRACT.json", dict(schema="filament.cpu_s4_6f.input_contract.v1", input_contract=input_source.get("input_contract", {}), expected_input_fingerprints=input_source.get("expected_input_fingerprints", {}), canonical_root=canonical_root, canonical_manifest_raw_sha256=canonical_manifest_hash, canonical_input_manifest=str(Path(canonical_input_manifest).resolve()), canonical_input_manifest_raw_sha256=canonical_manifest_hash, full_input_root=canonical_root, source=str(Path(input_plan).resolve()), classification="external", hash_scope="raw_bytes", source_raw_sha256=pv.raw_sha256_file(input_plan), shape=list(SHAPE), dtype="complex128", native_layout="C"))
    sg.write(report / "FULL_INPUT_MANIFEST.json", dict(schema="filament.cpu_s4_6f.full_input_manifest.v1", classification="external", hash_scope="raw_bytes", shape=list(SHAPE), dtype="complex128", native_layout="C", fingerprints=input_source.get("expected_input_fingerprints", {}), canonical_root=canonical_root, canonical_manifest_raw_sha256=canonical_manifest_hash, input_plan_raw_sha256=pv.raw_sha256_file(input_plan), source=str(Path(input_plan).resolve()), source_raw_sha256=pv.raw_sha256_file(input_plan)))
    full_input_manifest_raw_sha256 = pv.raw_sha256_file(report / "FULL_INPUT_MANIFEST.json")
    shutil.copyfile(reference_contract, report / "REFERENCE_INPUT_RAW_RECORDS.json")
    hpc_plan = dict(schema="filament.cpu_s4_6f.hpc_execution.v1", task=TASK, candidate=spec["semantic_id"], old_role=spec["old_role"], execution_commit=manifest["head"], shape=list(SHAPE), intervals=(2, 8), full_input_root=canonical_root, canonical_root=canonical_root, canonical_input_manifest=_external_path(canonical_input_manifest), canonical_manifest_raw_sha256=canonical_manifest_hash, canonical_input_manifest_raw_sha256=canonical_manifest_hash, reference_output=_external_path(reference_output), reference_boundary=None if reference_boundary is None else _external_path(reference_boundary), reference_stages=None if reference_stages is None else _external_path(reference_stages), resources=plan(candidate)["resources"], thread_environment=plan(candidate)["thread_environment"], work_deadline_s=7080, memory_guard_fraction=0.8, requeue=False, submission_authorized=False)
    sg.write(report / "HPC_EXECUTION_PLAN.json", hpc_plan)
    binding = dict(schema="filament.cpu_s4_6f.execution.v1", task=TASK, candidate=spec["semantic_id"], old_role=spec["old_role"], baseline_commit=BASE, science_tree=SCIENCE_TREE, execution_commit=manifest["head"], execution_branch=manifest["branch"], plan_raw_sha256=pv.raw_sha256_file(report / "PLAN.json"), source_manifest_raw_sha256=pv.raw_sha256_file(report / "EXECUTION_SOURCE_MANIFEST.json"), inherited_manifest_raw_sha256=pv.raw_sha256_file(inherited_manifest), input_contract_raw_sha256=pv.raw_sha256_file(report / "INPUT_CONTRACT.json"), reference_contract_raw_sha256=pv.raw_sha256_file(report / "REFERENCE_INPUT_RAW_RECORDS.json"), full_input_manifest_raw_sha256=full_input_manifest_raw_sha256, hpc_plan_raw_sha256=pv.raw_sha256_file(report / "HPC_EXECUTION_PLAN.json"), full_input_root=canonical_root, canonical_root=canonical_root, canonical_input_manifest=_external_path(canonical_input_manifest), canonical_manifest_raw_sha256=canonical_manifest_hash, reference_shape=list(SHAPE), reference_intervals=8, reference_role="L1_LINEAR_FACTOR_CACHE", default="I0", production_qualified=False, submission_authorized=False)
    sg.write(report / "EXECUTION.json", binding)
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(sorted(("EQ10_P48", "INTENSITY_P24", "PHASE_ABSORB_P24"))), required=True)
    parser.add_argument("--inherited-manifest", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    # These are external paths and may be POSIX paths on the remote node.
    # Keep them as strings until _external_path records their provenance.
    parser.add_argument("--full-input-root", type=str, required=True)
    parser.add_argument("--canonical-input-manifest", type=str, required=True)
    parser.add_argument("--reference-output", type=str, required=True)
    parser.add_argument("--reference-boundary", type=str)
    parser.add_argument("--reference-stages", type=str)
    parser.add_argument("--canonical-manifest-sha256")
    args = parser.parse_args(argv)
    result = prepare(**vars(args))
    print(result["execution_commit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

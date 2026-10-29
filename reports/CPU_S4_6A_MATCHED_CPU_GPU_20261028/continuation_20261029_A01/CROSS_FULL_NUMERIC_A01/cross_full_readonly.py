"""Acquisition-only FULL QUAL_OFF cross-backend comparison adapter.

The numerical comparison body is restored from the frozen ``cross_small``
function in ``Filament_python/tools/cpu_s4_6a/matrix.py`` with only the
function name and four small-grid path literals changed.  This module does
not implement a second comparator and does not open scientific arrays during
import or self-test.  The actual invocation is intended for the GPU staging
host after the CPU snapshot has been relayed into this stage.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


STATUS = "CROSS_BACKEND_NUMERICAL_EQUIVALENCE_NOT_QUALIFIED"
FULL_SHAPE = (384, 351, 301)
CPU_SOURCE_HEAD = "ed8754041625549a80959db7a8abd804d5d219ae"
GPU_SOURCE_HEAD = "6440ace43e07523ef924c790055b3cc21e7bc1fc"
SCIENTIFIC_TREE_HEAD = "821fec5bdacd832b0347e1f2e84285cb54172e57"
GPU_CODE_ROOT = "/data/run01/scvi806/user_Wangjimin/projects/filament_cpu_s46a_20261029_r6/code"
STAGE_NAME = "CROSS_FULL_NUMERIC_A01"

_REPLACEMENTS = {
    "CPU_SMALLGRID_TRANSPARENCY.json": "CPU_FULLGRID_TRANSPARENCY.json",
    "CPU_SMALLGRID_OUTPUT": "CPU_FULLGRID_OUTPUT",
    "CPU_SMALLGRID_FIRST_BOUNDARY": "CPU_FULLGRID_FIRST_BOUNDARY",
    "CROSS_BACKEND_NUMERIC_DIFFERENCES_SMALL.json": "CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json",
}
_REQUIRED_COMPARISON_TEXT = (
    "np.array_equal",
    "np.isnan",
    "np.isinf",
    "np.isfinite",
    ".shape",
    ".dtype",
    "np.linalg.norm",
    "L2(CPU finite elements)",
    "MATCHED_ACCEPTED_SCHEDULE_AND_BRANCHES",
    "SAME_WORK_BRANCH",
)


class CrossFullError(RuntimeError):
    """Fail-closed error for the fixed FULL acquisition stage."""


def _repo_root() -> Path:
    # .../cpu_optimization/reports/.../continuation/CROSS_FULL_NUMERIC_A01/file
    local = Path(__file__).resolve().parents[4]
    if (local / "Filament_python").is_dir():
        return local
    fixed = Path(GPU_CODE_ROOT)
    if (fixed / "Filament_python").is_dir():
        return fixed
    return local


def _matrix_source(repo: Path | None = None) -> Path:
    root = repo or _repo_root()
    path = root / "Filament_python/tools/cpu_s4_6a/matrix.py"
    if not path.is_file() or path.is_symlink():
        raise CrossFullError(f"MATRIX_SOURCE_MISSING:{path}")
    return path


def _read_source(path: Path) -> tuple[str, bytes]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CrossFullError(f"MATRIX_SOURCE_UTF8_REQUIRED:{path}") from exc
    return text, raw


def _find_cross_small(tree: ast.AST) -> ast.FunctionDef:
    for node in getattr(tree, "body", ()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "cross_small":
            if isinstance(node, ast.AsyncFunctionDef):
                raise CrossFullError("CROSS_SMALL_ASYNC_UNEXPECTED")
            return node
    raise CrossFullError("CROSS_SMALL_FUNCTION_MISSING")


class _FullLiteralTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.counts = {key: 0 for key in _REPLACEMENTS}

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value = node.value
        if isinstance(value, str) and value in _REPLACEMENTS:
            self.counts[value] += 1
            return ast.copy_location(ast.Constant(value=_REPLACEMENTS[value]), node)
        return node


def _restored_ast(repo: Path | None = None) -> tuple[Path, str, bytes, ast.FunctionDef, ast.FunctionDef, dict[str, int]]:
    source_path = _matrix_source(repo)
    text, raw = _read_source(source_path)
    try:
        tree = ast.parse(text, filename=str(source_path))
    except SyntaxError as exc:
        raise CrossFullError(f"MATRIX_SOURCE_PARSE_FAILED:{source_path}") from exc
    original = _find_cross_small(tree)
    restored = ast.parse(ast.unparse(original)).body[0]
    if not isinstance(restored, ast.FunctionDef):
        raise CrossFullError("CROSS_FULL_AST_FUNCTION_REQUIRED")
    restored.name = "cross_full"
    transformer = _FullLiteralTransformer()
    restored = transformer.visit(restored)
    restored = ast.fix_missing_locations(restored)
    if not isinstance(restored, ast.FunctionDef):
        raise CrossFullError("CROSS_FULL_AST_FUNCTION_REQUIRED")
    for small, count in transformer.counts.items():
        if count != 1:
            raise CrossFullError(f"CROSS_FULL_LITERAL_COUNT:{small}:{count}")
    return source_path, text, raw, original, restored, dict(transformer.counts)


def _function_text(node: ast.FunctionDef) -> str:
    return ast.unparse(node)


def _source_proof(repo: Path | None = None) -> dict[str, Any]:
    source_path, text, raw, original, restored, counts = _restored_ast(repo)
    original_text = _function_text(original)
    restored_text = _function_text(restored)
    for required in _REQUIRED_COMPARISON_TEXT:
        if required not in original_text:
            raise CrossFullError(f"CROSS_SOURCE_CONTRACT_MISSING:{required}")
    if "np.allclose" in original_text or "np.isclose" in original_text:
        raise CrossFullError("CROSS_SOURCE_UNAUTHORIZED_TOLERANCE")
    normalized_original = original_text
    for small, full in _REPLACEMENTS.items():
        normalized_original = normalized_original.replace(small, full)
    normalized_original = normalized_original.replace("def cross_small", "def cross_full", 1)
    ast_equivalent = normalized_original == restored_text
    if not ast_equivalent:
        raise CrossFullError("CROSS_FULL_AST_RESTORE_MISMATCH")
    return {
        "source_path": str(source_path.relative_to(repo or _repo_root()).as_posix()),
        "function_source": "cross_small",
        "restored_function": "cross_full",
        "allowed_replacements": dict(_REPLACEMENTS),
        "replacement_counts": counts,
        "ast_restore_status": "PASS_AST_RESTORED_EXACT_BODY",
        "ast_body_equivalent_after_allowed_replacements": True,
        "comparison_contract": {
            "schedule_branch_checks": True,
            "shape_dtype_checks": True,
            "nan_inf_finite_checks": True,
            "finite_element_difference": True,
            "relative_l2_denominator": "L2(CPU finite elements)",
            "zero_denominator_branch": True,
            "new_epsilon": None,
        },
        "source_identity_pv2": _pv2_identity(source_path, repo or _repo_root(), tracked=True),
        "source_text_bytes": len(raw),
        "source_text_line_count": text.count("\n") + 1,
    }


def _strict_load(path: Path, repo: Path | None = None) -> Any:
    repo = repo or _repo_root()
    source_root = repo / "Filament_python"
    inserted = False
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
        inserted = True
    try:
        from tools.cpu_s4_6a.records import load_json_strict

        return load_json_strict(path)
    finally:
        if inserted:
            try:
                sys.path.remove(source_text)
            except ValueError:
                pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CrossFullError(reason)


def _case0(payload: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    cases = payload.get("cases")
    _require(isinstance(cases, list) and cases, f"{side}_CASES_REQUIRED")
    case = cases[0]
    _require(isinstance(case, Mapping), f"{side}_CASE0_OBJECT_REQUIRED")
    _require(case.get("case") == "QUAL_OFF", f"{side}_FIRST_CASE_QUAL_OFF_REQUIRED")
    shape = case.get("shape", payload.get("shape"))
    _require(tuple(shape or ()) == FULL_SHAPE, f"{side}_FULL_SHAPE_REQUIRED")
    return case


def _validate_stage(stage: Path, cpu: Mapping[str, Any], gpu: Mapping[str, Any]) -> dict[str, Any]:
    _require(stage.name == STAGE_NAME, f"FULL_STAGE_NAME_REQUIRED:{STAGE_NAME}")
    _require(stage.is_dir() and not stage.is_symlink(), "FULL_STAGE_DIRECTORY_REQUIRED")
    cpu_case = _case0(cpu, "CPU")
    gpu_case = _case0(gpu, "GPU")
    _require(str(cpu.get("backend", "CPU")).upper() == "CPU", "CPU_FULL_BACKEND_REQUIRED")
    _require(str(gpu.get("backend", "GPU")).upper() == "GPU", "GPU_FULL_BACKEND_REQUIRED")
    _require(cpu_case.get("input_identity") == gpu_case.get("input_identity"), "CROSS_BACKEND_IDENTICAL_COMPLETE_INPUT")
    _require(cpu_case.get("source_binding", {}).get("head") == CPU_SOURCE_HEAD, "CPU_FULL_SOURCE_HEAD_MISMATCH")
    _require(gpu_case.get("source_binding", {}).get("head") == GPU_SOURCE_HEAD, "GPU_FULL_SOURCE_HEAD_MISMATCH")
    required_snapshot_files: dict[str, list[str]] = {}
    for name in ("CPU_FULLGRID_OUTPUT", "CPU_FULLGRID_FIRST_BOUNDARY"):
        root = stage / name
        _require(root.is_dir() and not root.is_symlink(), f"{name}_DIRECTORY_REQUIRED")
        files = []
        for filename in ("TREE.json", "ARRAYS.zip", "ALIASES.json", "FINGERPRINTS.json"):
            path = root / filename
            _require(path.is_file() and not path.is_symlink(), f"{name}_{filename}_REQUIRED")
            files.append(filename)
        required_snapshot_files[name] = files
    return {
        "stage": stage.name,
        "shape": list(FULL_SHAPE),
        "first_case": "QUAL_OFF",
        "input_identity_match": True,
        "cpu_source_head": cpu_case["source_binding"]["head"],
        "gpu_source_head": gpu_case["source_binding"]["head"],
        "gpu_code_root": GPU_CODE_ROOT,
        "scientific_tree_head": SCIENTIFIC_TREE_HEAD,
        "required_cpu_snapshots": required_snapshot_files,
    }


@contextmanager
def _host_comparison_environment() -> Any:
    names = ("UPPE_USE_GPU", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names if name != "UPPE_USE_GPU"})
    os.environ["UPPE_USE_GPU"] = "0"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _runtime_globals(repo: Path) -> dict[str, Any]:
    source_root = repo / "Filament_python"
    source_text = str(source_root)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        from tools.cpu_s4_6a.inputs import require, write
        from tools.cpu_s4_6a.records import load_json_strict
    except Exception:
        if inserted:
            sys.path.remove(source_text)
        raise
    return {
        "__name__": "tools.cpu_s4_6a.cross_full_restored",
        "__package__": "tools.cpu_s4_6a",
        "load_json_strict": load_json_strict,
        "require": require,
        "write": write,
    }


def _restored_cross_full(repo: Path | None = None) -> tuple[Any, dict[str, Any]]:
    root = repo or _repo_root()
    _, _, _, _, restored, _ = _restored_ast(root)
    namespace = _runtime_globals(root)
    exec(compile(ast.Module(body=[restored], type_ignores=[]), filename="matrix.py::cross_full", mode="exec"), namespace)
    return namespace["cross_full"], _source_proof(root)


def _pv2_identity(path: Path, repo: Path, *, tracked: bool = False) -> dict[str, Any]:
    source_text = str(repo / "Filament_python")
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        from tools.hpc_ops import provenance_v2 as pv

        if tracked:
            relative = path.resolve().relative_to(repo.resolve()).as_posix()
            oid = pv._git(repo, "rev-parse", "HEAD:" + relative)
            blob = pv._read_git_blob(repo, oid)
            _require(pv.canonical_lf_bytes(path.read_bytes()) == pv.canonical_lf_bytes(blob),
                     "TRACKED_SOURCE_CANONICAL_BINDING")
            return dict(implementation="Filament_python/tools/hpc_ops/provenance_v2.py",
                        path=relative, classification="tracked_text", hash_scope=pv.TRACKED_HASH_SCOPE,
                        git_blob_oid=oid, canonical_lf_sha256=pv.canonical_lf_sha256(blob))
        return dict(implementation="Filament_python/tools/hpc_ops/provenance_v2.py",
                    path=str(path), classification="external", hash_scope="raw_bytes",
                    raw_sha256=pv.raw_sha256_file(path), size_bytes=path.stat().st_size)
    finally:
        if inserted:
            try:
                sys.path.remove(source_text)
            except ValueError:
                pass


def _peak_rss_bytes() -> dict[str, Any]:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            value *= 1024
        return {"status": "RECORDED", "bytes": value, "source": "resource.RUSAGE_SELF.ru_maxrss", "scope": "acquisition_only_comparator_process"}
    except (ImportError, OSError, AttributeError):
        return {"status": "UNAVAILABLE", "bytes": None, "source": "resource.RUSAGE_SELF.ru_maxrss", "scope": "acquisition_only_comparator_process"}


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_full_cross(
    stage: os.PathLike[str] | str,
    gpu_result_path: os.PathLike[str] | str,
    *,
    code_root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    stage_path = Path(stage).resolve(strict=True)
    gpu_path = Path(gpu_result_path).resolve(strict=True)
    repo = Path(code_root).resolve(strict=True) if code_root is not None else _repo_root()
    _require((repo / "Filament_python").is_dir(), "FILAMENT_CODE_ROOT_REQUIRED")
    _require(repo == Path(GPU_CODE_ROOT).resolve(strict=True), "FIXED_GPU_CODE_ROOT_REQUIRED")
    _require(stage_path == repo.parent / STAGE_NAME, "FIXED_FULL_STAGE_ROOT_REQUIRED")
    sys.path.insert(0, str(repo / "Filament_python"))
    from tools.hpc_ops import provenance_v2 as pv
    _require(pv._git(repo, "rev-parse", "HEAD") == GPU_SOURCE_HEAD, "ACTUAL_GPU_CODE_HEAD_REQUIRED")
    _require(pv._git(repo, "rev-parse", "HEAD:Filament_python/KHz_filament") == SCIENTIFIC_TREE_HEAD,
             "ACTUAL_FROZEN_SCIENTIFIC_TREE_REQUIRED")
    pv._ensure_clean(repo)
    _require(gpu_path.is_file() and not gpu_path.is_symlink(), "GPU_FULL_MATRIX_FILE_REQUIRED")
    cpu_path = stage_path / "CPU_FULLGRID_TRANSPARENCY.json"
    _require(cpu_path.is_file() and not cpu_path.is_symlink(), "CPU_FULLGRID_TRANSPARENCY_REQUIRED")
    cpu_result = _strict_load(cpu_path, repo)
    gpu_result = _strict_load(gpu_path, repo)
    _require(isinstance(cpu_result, Mapping), "CPU_FULL_MATRIX_OBJECT_REQUIRED")
    _require(isinstance(gpu_result, Mapping), "GPU_FULL_MATRIX_OBJECT_REQUIRED")
    stage_identity = _validate_stage(stage_path, cpu_result, gpu_result)
    source_proof = _source_proof(repo)
    cross_full, _ = _restored_cross_full(repo)
    started_wall = time.perf_counter_ns()
    started_cpu = time.process_time_ns()
    with _host_comparison_environment():
        # The host service preload is deliberately before the restored
        # comparator and outside every scientific timing window.
        source_text = str(repo / "Filament_python")
        inserted = source_text not in sys.path
        if inserted:
            sys.path.insert(0, source_text)
        try:
            from tools.cpu_s4_6a import gpu as gpu_module

            preload = getattr(gpu_module, "_preload_host_services", None)
            _require(callable(preload), "GPU_HOST_SERVICE_PRELOAD_MISSING")
            preload()
            result = cross_full(stage_path, gpu_result)
        finally:
            if inserted:
                try:
                    sys.path.remove(source_text)
                except ValueError:
                    pass
    finished_cpu = time.process_time_ns()
    finished_wall = time.perf_counter_ns()
    _require(isinstance(result, Mapping), "CROSS_FULL_RESULT_OBJECT_REQUIRED")
    _require(result.get("status") == STATUS, "CROSS_FULL_STATUS_NOT_PRESERVED")
    sidecar = {
        "schema": "filament.cpu.s4_6a.cross_full_acquisition.v1",
        "status": STATUS,
        "stage": stage_identity,
        "matrix_result": {
            "path": str(gpu_path),
            "pv2": _pv2_identity(gpu_path, repo),
            "backend": gpu_result.get("backend"),
            "shape": gpu_result.get("shape"),
            "first_case": gpu_result.get("cases", [{}])[0].get("case") if isinstance(gpu_result.get("cases"), list) and gpu_result.get("cases") else None,
        },
        "source": {
            "cpu_source_head": CPU_SOURCE_HEAD,
            "gpu_source_head": GPU_SOURCE_HEAD,
            "scientific_tree_head": SCIENTIFIC_TREE_HEAD,
            "gpu_code_root": GPU_CODE_ROOT,
            "matrix_source_proof": source_proof,
            "matrix_source_pv2": _pv2_identity(_matrix_source(repo), repo, tracked=True),
            "analysis_adapter_pv2": _pv2_identity(Path(__file__), repo),
            "cpu_transparency_pv2": _pv2_identity(cpu_path, repo),
        },
        "timing": {
            "scope": "acquisition_only_comparator_process",
            "clock": "perf_counter_ns/process_time_ns",
            "start_wall_ns": started_wall,
            "end_wall_ns": finished_wall,
            "wall_s": (finished_wall - started_wall) / 1e9,
            "process_cpu_s": (finished_cpu - started_cpu) / 1e9,
            "scientific_timing_window_modified": False,
        },
        "peak_rss": _peak_rss_bytes(),
        "output": "CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json",
        "result_status": result.get("status"),
        "field_count": len(result.get("fields", [])) if isinstance(result.get("fields"), list) else None,
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
    }
    _write_new_json(stage_path / "CROSS_FULL_NUMERIC_A01_SIDECAR.json", sidecar)
    return {
        "status": STATUS,
        "stage": stage_identity["stage"],
        "output": str(stage_path / "CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json"),
        "sidecar": str(stage_path / "CROSS_FULL_NUMERIC_A01_SIDECAR.json"),
        "field_count": sidecar["field_count"],
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
    }


def self_test(repo: Path | None = None) -> dict[str, Any]:
    proof = _source_proof(repo or _repo_root())
    _require(proof["ast_restore_status"] == "PASS_AST_RESTORED_EXACT_BODY", "SELF_TEST_AST_RESTORE")
    _require(proof["ast_body_equivalent_after_allowed_replacements"], "SELF_TEST_AST_BODY")
    return {
        "schema": "filament.cpu.s4_6a.cross_full_acquisition.v1",
        "status": "SELF_TEST_PASS",
        "ast_restore": proof,
        "arrays_opened": False,
        "propagation_run": False,
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S46A FULL QUAL_OFF read-only cross-backend adapter")
    parser.add_argument("--stage", default=None, help="CROSS_FULL_NUMERIC_A01 staging directory")
    parser.add_argument("--gpu-result", default=None, help="GPU MATRIX_RESULT_GPU_351.json")
    parser.add_argument("--code-root", default=None, help="GPU code root containing Filament_python")
    parser.add_argument("--self-test", action="store_true", help="AST-only source restoration check")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
        return 0
    if not args.stage or not args.gpu_result:
        parser.error("--stage and --gpu-result are required unless --self-test is used")
    with _host_comparison_environment():
        result = run_full_cross(args.stage, args.gpu_result, code_root=args.code_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

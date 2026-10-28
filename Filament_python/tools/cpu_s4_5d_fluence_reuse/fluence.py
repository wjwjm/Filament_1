"""Private post-processing for the S4-5D saved-fluence seam.

The public :func:`KHz_filament.diagnostics.second_moment_radius` is the
reference oracle and remains untouched.  ``second_moment_radius_from_f2d``
is its post-integration body copied in source order; the only removed
operation is the time trapezoid itself.  It deliberately does not call the
existing public 2-D helper, whose threshold-index convention is different.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as _np

from KHz_filament.device import xp
from tools.hpc_ops import provenance_v2 as _provenance


def second_moment_radius_from_f2d(
    F2D,
    x,
    y,
    *,
    frac_keep: float = 0.999,
    rel_floor: float = 1e-8,
) -> float:
    """Evaluate the original 3-D radius post-processing on an existing F2D.

    ``F2D`` is the native result of the original saved-branch trapezoid.  The
    cleaning, ordering, cumulative-threshold, reduction, and square-root
    operations below intentionally follow the original 3-D function after its
    time integration, including the ``count_nonzero(csum < target) + 1``
    threshold rule.  Every cleaning operation creates a new array; the input
    fluence is never written in place.
    """

    # The admitted execution domain is the CPU complex128 propagation path:
    # intensity/F2D is native NumPy float64.  Do not silently fall back to a
    # different backend or dtype when the source-bound candidate is used.
    if getattr(xp, "__name__", "numpy") != "numpy":
        raise TypeError("S45D_CPU_NUMPY_BACKEND_REQUIRED")
    if not hasattr(F2D, "dtype") or _np.dtype(F2D.dtype) != _np.dtype(_np.float64):
        raise TypeError("S45D_F2D_FLOAT64_REQUIRED")

    # ---- 1) The original function has already integrated along t. ----
    F = xp.asarray(F2D)
    F = xp.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    F = xp.maximum(F, 0.0)

    # ---- 2) 基础过滤：相对门限 ----
    fmax = float(F.max())
    if not math.isfinite(fmax) or fmax <= 0.0:
        return 0.0
    if rel_floor > 0.0:
        F = xp.where(F >= rel_floor * fmax, F, 0.0)

    Ny, Nx = F.shape
    x = xp.asarray(x)
    y = xp.asarray(y)
    dx = float(x[1] - x[0]) if Nx > 1 else 1.0
    dy = float(y[1] - y[0]) if Ny > 1 else 1.0

    # ---- 3) 主能量区选择（避免 CuPy 的 searchsorted 限制）----
    # 扁平化并按强度降序
    if 0.0 < frac_keep < 1.0:
        flat = F.ravel()
        order = xp.argsort(flat)[::-1]
        flat_sorted = flat[order]
        csum = xp.cumsum(flat_sorted, dtype=xp.float64)
        total = float(csum[-1])
        if total <= 0.0:
            return 0.0
        target = frac_keep * total
        # CuPy 对标量 searchsorted 有兼容问题；用计数实现等价功能
        k = int(xp.count_nonzero(csum < target)) + 1  # 第一个使累计≥target的位置（1基）
        k = max(1, min(k, flat_sorted.size))
        thr = float(flat_sorted[k - 1])
        F = xp.where(F >= thr, F, 0.0)

    # ---- 4) 二阶矩计算（矢量化，避免大 meshgrid）----
    S = float(xp.sum(F, dtype=xp.float64) * dx * dy)
    if S <= 0.0 or not math.isfinite(S):
        return 0.0

    x2 = x * x                  # [Nx]
    y2 = y * y                  # [Ny]
    Fx_sum_y = xp.sum(F, axis=0, dtype=xp.float64)  # [Nx]
    Fy_sum_x = xp.sum(F, axis=1, dtype=xp.float64)  # [Ny]
    mom_x2 = float(xp.sum(x2 * Fx_sum_y) * dx * dy)
    mom_y2 = float(xp.sum(y2 * Fy_sum_x) * dx * dy)

    mean_x2 = mom_x2 / S
    mean_y2 = mom_y2 / S

    # 高斯关系：<x^2>=w^2/4, <y^2>=w^2/4 => w = sqrt(2*(<x^2>+<y^2>))
    w_sq = 2.0 * (mean_x2 + mean_y2)
    if w_sq <= 0.0 or not math.isfinite(w_sq):
        return 0.0
    return math.sqrt(w_sq)


def _function_node(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"S45D_POST_AST_FUNCTION_COUNT:{name}:{len(matches)}")
    return matches[0]


def _is_original_integral(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    value = node.value
    if not isinstance(target, ast.Name) or target.id != "F" or not isinstance(value, ast.Call):
        return False
    if not any(keyword.arg == "axis" for keyword in value.keywords):
        return False
    axis = next(keyword.value for keyword in value.keywords if keyword.arg == "axis")
    return isinstance(axis, ast.Constant) and axis.value == 0


def _is_private_f2d_seed(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    value = node.value
    return bool(
        isinstance(target, ast.Name)
        and target.id == "F"
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "xp"
        and value.func.attr == "asarray"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "F2D"
        and not value.keywords
    )


def postprocess_source_proof(diagnostics_path: str | Path | None = None) -> dict[str, object]:
    """Prove the private body matches the bound 3-D post-integral AST.

    The original function's first ``F`` assignment is its time integral.  The
    comparison starts with the following statement.  The private function's
    ``F = xp.asarray(F2D)`` seed is skipped, as it is the explicit bound input
    replacement for that removed integral.  CPU/dtype guards and docstrings
    are outside the compared region.  AST hashes are structural and do not
    retain field values or arrays.
    """

    if diagnostics_path is None:
        repo = Path(__file__).resolve().parents[3]
        diagnostics = repo / "Filament_python/KHz_filament/diagnostics.py"
    else:
        diagnostics = Path(diagnostics_path).resolve(strict=True)
    private_path = Path(__file__).resolve(strict=True)
    original_module = ast.parse(diagnostics.read_text(encoding="utf-8"), filename=str(diagnostics))
    private_module = ast.parse(private_path.read_text(encoding="utf-8"), filename=str(private_path))
    original = _function_node(original_module, "second_moment_radius")
    private = _function_node(private_module, "second_moment_radius_from_f2d")
    original_indices = [
        index for index, statement in enumerate(original.body) if _is_original_integral(statement)
    ]
    private_indices = [
        index for index, statement in enumerate(private.body) if _is_private_f2d_seed(statement)
    ]
    if len(original_indices) != 1 or len(private_indices) != 1:
        raise ValueError(
            f"S45D_POST_AST_ANCHOR_COUNT:{len(original_indices)}:{len(private_indices)}"
        )
    original_post = original.body[original_indices[0] + 1 :]
    private_post = private.body[private_indices[0] + 1 :]
    original_tree = ast.Module(body=copy_ast_nodes(original_post), type_ignores=[])
    private_tree = ast.Module(body=copy_ast_nodes(private_post), type_ignores=[])
    original_dump = ast.dump(original_tree, include_attributes=False)
    private_dump = ast.dump(private_tree, include_attributes=False)
    if original_dump != private_dump:
        raise ValueError("S45D_PRIVATE_POSTPROCESS_AST_MISMATCH")
    public_2d_refs = sum(
        1
        for node in ast.walk(private)
        if isinstance(node, ast.Name) and node.id == "second_moment_radius_from_2d"
    )
    if public_2d_refs:
        raise ValueError("S45D_PUBLIC_2D_HELPER_REFERENCE_FORBIDDEN")
    return {
        "schema": "filament.cpu_s4_5d.fluence_post_ast_proof.v1",
        "diagnostics_path": str(diagnostics),
        "private_helper_path": str(private_path),
        "original_integral_statement_index": int(original_indices[0]),
        "private_f2d_seed_statement_index": int(private_indices[0]),
        "original_post_ast_sha256": _provenance.canonical_lf_sha256(original_dump.encode("utf-8")),
        "private_post_ast_sha256": _provenance.canonical_lf_sha256(private_dump.encode("utf-8")),
        "post_ast_equal": True,
        "public_2d_helper_references": int(public_2d_refs),
        "guards_excluded_from_comparison": True,
        "arrays_retained": False,
    }


def copy_ast_nodes(nodes: list[ast.stmt]) -> list[ast.stmt]:
    # Kept as a tiny local utility so the proof never aliases source AST nodes.
    import copy

    return copy.deepcopy(nodes)


__all__ = ["postprocess_source_proof", "second_moment_radius_from_f2d"]

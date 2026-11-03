"""Offline proof that S4-6J timing wrappers strip back to the frozen AST.

This is an analysis utility, not a runtime hook.  It reads the requested base
blob with ``git show`` and compares normalized ASTs; it never writes source or
result files.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import copy
from collections import Counter
from pathlib import Path
from typing import Any


BASE = "04aafc1386a21412fdfc8139515a6cc64da8a1a2"
SOURCE_FILES = (
    "Filament_python/tools/cpu_s4_6i_residual_parallel/power_hook.py",
    "Filament_python/tools/cpu_s4_6i_residual_parallel/selected_factory.py",
    "Filament_python/tools/cpu_s4_6h_combined_parallel/combined_candidate.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_candidate.py",
)


class ProofError(RuntimeError):
    pass


def validate_static_inventory(tree, relative):
    from .scope_inventory import STATIC_SCOPE_COUNTS
    counts = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'factory_scope':
            keywords = [item for item in node.keywords if item.arg == 'site']
            if len(keywords) != 1 or not isinstance(keywords[0].value, ast.Constant):
                raise ProofError('UNEXPECTED_TIMING_SITE')
            counts[keywords[0].value.value] += 1
    if dict(counts) != STATIC_SCOPE_COUNTS[relative]:
        raise ProofError('STATIC_SCOPE_INVENTORY_CHANGED:'+relative)


class _StripFactoryTiming(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.module == "tools.cpu_s4_6j_alpha_factory.factory_timing":
            if len(node.names) != 1 or node.names[0].name != 'scope' or node.names[0].asname != 'factory_scope':
                raise ProofError('UNEXPECTED_TIMING_IMPORT')
            return None
        return self.generic_visit(node)

    def visit_With(self, node: ast.With) -> Any:
        node = self.generic_visit(node)
        if len(node.items) == 1:
            expression = node.items[0].context_expr
            if (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id == "factory_scope"
            ):
                from .factory_timing import FACTORY_TIMING_SITES
                if (node.items[0].optional_vars is not None or len(expression.args) != 1
                    or not isinstance(expression.args[0], ast.Constant) or len(expression.keywords) != 1
                    or expression.keywords[0].arg != 'site' or not isinstance(expression.keywords[0].value, ast.Constant)):
                    raise ProofError('UNEXPECTED_TIMING_SCOPE_SHAPE')
                site = expression.keywords[0].value.value
                if site not in FACTORY_TIMING_SITES or site.split('.',1)[0] != expression.args[0].value:
                    raise ProofError('UNREGISTERED_TIMING_SCOPE')
                return node.body
        return node


def _git_blob(repo: Path, base: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{base}:{relative}"],
        cwd=str(repo),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ProofError(f"GIT_SHOW_FAILED:{relative}:{completed.returncode}")
    return completed.stdout.decode("utf-8")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ProofError(f"FUNCTION_COUNT:{name}:{len(matches)}")
    return matches[0]


def _normalize_source_function(current: ast.Module, frozen: ast.Module) -> None:
    """Inline only the added temporary; retain every other current AST node."""
    current_node = _function(current, "_source_function")
    current_try = current_node.body[0]
    if not isinstance(current_try, ast.Try):
        raise ProofError("SOURCE_FUNCTION_TRY_MISSING")
    if len(current_try.body) < 2:
        raise ProofError("SOURCE_FUNCTION_READ_PARSE_SPLIT_MISSING")
    source_assign, tree_assign = current_try.body[:2]
    if not isinstance(source_assign, ast.Assign) or not isinstance(tree_assign, ast.Assign):
        raise ProofError("SOURCE_FUNCTION_ASSIGNMENT_SHAPE_CHANGED")
    if not isinstance(source_assign.targets[0], ast.Name) or source_assign.targets[0].id != "source":
        raise ProofError("SOURCE_FUNCTION_SOURCE_NAME_CHANGED")
    if not isinstance(tree_assign.targets[0], ast.Name) or tree_assign.targets[0].id != "tree":
        raise ProofError("SOURCE_FUNCTION_TREE_NAME_CHANGED")
    if not (
        isinstance(tree_assign.value, ast.Call)
        and isinstance(tree_assign.value.func, ast.Attribute)
        and tree_assign.value.func.attr == "parse"
        and tree_assign.value.args
        and isinstance(tree_assign.value.args[0], ast.Name)
        and tree_assign.value.args[0].id == "source"
    ):
        raise ProofError("SOURCE_FUNCTION_PARSE_INPUT_CHANGED")
    if len(source_assign.targets) != 1 or len(tree_assign.targets) != 1:
        raise ProofError('SOURCE_FUNCTION_EXTRA_ASSIGNMENT')
    tree_assign.value.args[0] = copy.deepcopy(source_assign.value)
    current_try.body = [tree_assign] + current_try.body[2:]
    # The complete current function is compared below, including read call,
    # parse arguments, handlers, later statements, return and side effects.


def prove(*, repo: Path, base: str = BASE) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "filament.cpu.s4_6j.reverse_ast_proof.v1",
        "base": base,
        "files": {},
    }
    for relative in SOURCE_FILES:
        path = repo / relative
        current = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        validate_static_inventory(current, relative)
        frozen = ast.parse(_git_blob(repo, base, relative), filename=relative)
        current = _StripFactoryTiming().visit(current)
        ast.fix_missing_locations(current)
        if relative.endswith("power_hook.py"):
            _normalize_source_function(current, frozen)
        if ast.dump(current, include_attributes=False) != ast.dump(frozen, include_attributes=False):
            raise ProofError(f"NORMALIZED_AST_MISMATCH:{relative}")
        result["files"][relative] = {"status": "PASS", "normalized_ast_equal": True}
    result["status"] = "PASS"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args(argv)
    print(json.dumps(prove(repo=args.repo.resolve(), base=args.base), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

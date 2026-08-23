#!/usr/bin/env python3
"""Build repository_inventory.md + repository_inventory.json for Filament_1.

Read-only classifier. It does not move, edit, or delete any file, and it never
imports the KHz_filament package, so it cannot alter production behavior.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}

HIST_TOKENS = (
    "phase",
    "isaacs",
    "raman",
    "historical_fr",
    "ionization_rate_model_validation",
    "ionization_model_propagation",
    "ionization_integrator_validation",
    "density_translation",
    "paper_curve",
    "vacuum_focus",
    "transverse_profile_validation",
    "profile_validation",
    "stage1",
    "submit_stage",
    "finalize",
    "linear_checkpoint",
    "linear_transfer_kernel_audit",
    "current_observability",
    "feedback_energy",
    "filament_effect_ledger",
    "ionization_time_harness",
    "time_derivative_convention",
    "nonlinear_ablation",
    "nonlinear_switch_isolation_report",
    "nonlinear_switch_isolation",
    "eq10_eq11",
    "gate_computation",
)

GENERIC_FORCE_TOOLS = {
    "build_ion_lut_cache.py",
    "validate_ion_lut.py",
    "validate_ion_lut_runtime.py",
}

GENERIC_TOPLEVEL = {
    "test_run.py",
    "plot_khzfil_out.py",
    "npz2mat.py",
    "compare_khzfil_outputs.py",
    "submit_stage.py",
}

PRODUCTION_CONFIGS = {
    "config_ref.json",
    "khz_config.json",
    "khz_config_lut.json",
}


def relpath_forward(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_phase_branded(name: str) -> bool:
    return any(tok in name for tok in HIST_TOKENS)


def classify(path: Path) -> tuple[str, str]:
    """Return (class_one, kind)."""
    rel = relpath_forward(path)
    name = path.name.lower()

    if rel.startswith("Filament_python/KHz_filament/"):
        if path.suffix == ".py":
            return "production_runtime", "runtime_source"
        return "production_runtime", "runtime_doc"

    if path.name in PRODUCTION_CONFIGS and rel.startswith("Filament_python/"):
        return "production_config", "default_config"

    if rel.startswith("Filament_python/tests/"):
        if path.name in ("conftest.py", "README.md"):
            return "generic_tests_tools", "test_meta"
        if is_phase_branded(name):
            return "historical_experiments_audits", "phase_test"
        if path.suffix == ".py":
            return "generic_tests_tools", "test"
        return "results_documentation_evidence", "test_doc"

    if rel.startswith("Filament_python/matlab/"):
        return "generic_tests_tools", "postprocess_matlab"

    if rel.startswith("Filament_python/tools/"):
        if rel.startswith("Filament_python/tools/hpc_ops/"):
            return "generic_tests_tools", "hpc_ops"
        if rel.startswith("Filament_python/tools/audit/"):
            return "generic_tests_tools", "audit_tool"
        if path.name == "README.md":
            return "results_documentation_evidence", "tool_doc"
        if path.name in GENERIC_FORCE_TOOLS:
            return "generic_tests_tools", "tool"
        if is_phase_branded(name):
            return "historical_experiments_audits", "phase_tool"
        if path.suffix == ".py":
            return "generic_tests_tools", "tool"
        if path.suffix in (".sh", ".sbatch"):
            return "historical_experiments_audits", "phase_launcher"
        return "historical_experiments_audits", "phase_artifact"

    if rel.startswith("Filament_python/"):
        top = path.name
        if top in GENERIC_TOPLEVEL:
            return "generic_tests_tools", "runner_or_postprocess"
        if path.suffix == ".py":
            if is_phase_branded(name):
                return "historical_experiments_audits", "phase_tool"
            return "generic_tests_tools", "runner_or_postprocess"
        if path.suffix == ".sh":
            if is_phase_branded(name):
                return "historical_experiments_audits", "phase_launcher"
            return "generic_tests_tools", "run_launcher"
        if rel.startswith("Filament_python/configs/"):
            if path.suffix == ".json":
                return "historical_experiments_audits", "experiment_config"
            return "results_documentation_evidence", "config_doc"
        if rel.startswith("Filament_python/stages/"):
            return "historical_experiments_audits", "stage_spec"
        if rel.startswith("Filament_python/docs/"):
            return "historical_experiments_audits", "phase_doc"
        if rel.startswith("Filament_python/results/"):
            return "results_documentation_evidence", "result_evidence"
        if rel.startswith("Filament_python/tmp/"):
            return "results_documentation_evidence", "temporary_evidence"
        if path.name in ("README.md", "requirements.txt"):
            return "results_documentation_evidence", "repo_doc"

    if rel.startswith(("references/", "修改记录/", "docs/")):
        return "results_documentation_evidence", "reference_or_doc"
    if rel.startswith((".codex/", ".workbuddy/")):
        return "results_documentation_evidence", "repo_meta"
    if path.name in ("AGENTS.md", "README.md", ".gitignore", ".gitattributes"):
        return "results_documentation_evidence", "repo_doc"

    return "unclassified", "unknown"


def git_tracked() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    return {p.replace("\\", "/") for p in out.decode("utf-8").split("\0") if p}


def main() -> int:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, check=True
    ).stdout.decode().strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=REPO, capture_output=True, check=True
    ).stdout.decode().strip()

    tracked = git_tracked()
    entries: list[dict] = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        rel = relpath_forward(path)
        class_one, kind = classify(path)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        entries.append(
            {
                "path": rel,
                "size": size,
                "class_one": class_one,
                "kind": kind,
                "git_tracked": rel in tracked,
                "sha256": sha256(path) if size < (64 << 20) else None,
            }
        )

    summary: dict[str, int] = {}
    for e in entries:
        summary[e["class_one"]] = summary.get(e["class_one"], 0) + 1

    manifest = {
        "schema": "filament_1.repository_inventory.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_head_sha": head,
        "branch": branch,
        "total_files": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "class_summary": summary,
        "files": entries,
    }

    out_dir = REPO / "docs" / "repo_layout"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repository_inventory.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = ["# Filament_1 仓库结构 inventory", ""]
    md_lines.append(f"- 生成时间(UTC): {manifest['generated_at']}")
    md_lines.append(f"- 仓库 HEAD SHA: `{head}`")
    md_lines.append(f"- 分支: `{branch}`")
    md_lines.append(f"- 文件总数: {manifest['total_files']}")
    md_lines.append(f"- 总字节数: {manifest['total_bytes']}")
    md_lines.append("")
    md_lines.append("## 五类统计")
    for k in (
        "production_runtime",
        "production_config",
        "generic_tests_tools",
        "historical_experiments_audits",
        "results_documentation_evidence",
        "unclassified",
    ):
        if summary.get(k):
            md_lines.append(f"- `{k}`: {summary[k]}")
    md_lines.append("")
    md_lines.append("## 分类明细（按 path 排序，完整清单见 repository_inventory.json）")
    for e in entries:
        md_lines.append(f"- `{e['path']}` — {e['class_one']} / {e['kind']} ({e['size']} B)")
    md_lines.append("")
    (out_dir / "repository_inventory.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    print(f"wrote {relpath_forward(out_dir / 'repository_inventory.json')}")
    print(f"wrote {relpath_forward(out_dir / 'repository_inventory.md')}")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

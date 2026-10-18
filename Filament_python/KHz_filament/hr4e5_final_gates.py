"""Final, evidence-backed admission gates for the N3/K8048/H4 campaign.

This module validates external evidence before the production campaign root is
created.  It does not change the scientific operators or the streaming journal
protocol.  Legacy callers may omit the gate manifest; the N3 launcher must
require one explicitly.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from tools.hpc_ops.provenance_v2 import canonical_lf_sha256, raw_sha256_file


FINAL_GATES_SCHEMA = "khz_filament.hr4e5.final_gates.v1"
N3_FORMAL_SCOPE = {
    "n_pulses": 3,
    "k": 8048,
    "hydro_workers": 4,
    "allocation_seconds": 432000,
    "science_child_timeout_s": 432000,
    "storage_lock_timeout_s": 163,
    "lifecycle_lock_timeout_s": 10,
    "max_campaign_live_bytes": 300 * 1024**3,
    "final_output_budget_bytes": 64 * 1024**3,
    "safety_margin_bytes": 8 * 1024**3,
}
PERFORMANCE_CHECKS = frozenset({
    "all_case_correctness",
    "matched_scope",
    "same_transition_count",
    "p95_hold_at_least_3x",
    "metadata_reduction_at_least_90_percent",
    "serial_throughput_at_least_2x",
    "no_new_high_frequency_checkpoint_rewrite",
    "new_concurrent_target_wait_below_5s",
    "new_concurrent_cycles_complete",
    "new_concurrent_transition_count",
    "no_lock_timeout",
    "observation_overhead_within_limit",
})
PERFORMANCE_HOT_PATHS = frozenset({
    "KHz_filament/hr4e5_streaming_journal.py",
    "KHz_filament/hr4e5s_streaming.py",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require_final_gate_path(value: str | os.PathLike[str] | None) -> Path:
    """Require an explicit final-gate path at a new formal entry point."""
    if value is None or not str(value).strip():
        raise ValueError("N3 formal entry requires final_gate_manifest_path")
    return Path(value)


def _external_regular_file(
    raw_path: Any, *, label: str, campaign_root: Path,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} path is unavailable") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    root = campaign_root.resolve(strict=False)
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError(f"{label} must be outside the campaign root")
    return resolved


def _raw_sha256(path: Path) -> str:
    return raw_sha256_file(path)


def _read_hashed_json(
    raw_path: Any, expected_sha256: Any, *, label: str, campaign_root: Path,
) -> tuple[Path, dict[str, Any], str]:
    path = _external_regular_file(raw_path, label=label, campaign_root=campaign_root)
    expected = str(expected_sha256 or "")
    if not _SHA256.fullmatch(expected):
        raise ValueError(f"{label} raw SHA256 is invalid")
    actual = _raw_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} raw SHA256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return path, value, actual


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _close(actual: Any, expected: float, *, label: str) -> float:
    value = _finite_number(actual, label=label, positive=True)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(f"{label} does not match recomputed value")
    return value


def _validate_performance(
    raw: Any, *, package_root: Path, campaign_root: Path,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("status") != "PASS":
        raise ValueError("site performance qualification is not PASS")
    evidence_path, evidence, evidence_sha = _read_hashed_json(
        raw.get("evidence_path"), raw.get("evidence_raw_sha256"),
        label="site performance evidence", campaign_root=campaign_root,
    )
    if evidence.get("status") != "PASS":
        raise ValueError("site performance evidence status is not PASS")
    checks = evidence.get("checks")
    if not isinstance(checks, Mapping) or not PERFORMANCE_CHECKS.issubset(checks):
        raise ValueError("site performance evidence lacks required checks")
    if any(value is not True for value in checks.values()):
        raise ValueError("site performance evidence has a failed check")
    ratios = evidence.get("ratios")
    if not isinstance(ratios, Mapping):
        raise ValueError("site performance ratios are missing")
    if _finite_number(ratios.get("p95_hold_speedup"), label="p95 hold speedup") < 3.0:
        raise ValueError("site performance p95 hold gate failed")
    if _finite_number(ratios.get("metadata_write_reduction"), label="metadata reduction") < 0.90:
        raise ValueError("site performance metadata reduction gate failed")
    if _finite_number(ratios.get("matched_serial_throughput_speedup"),
                       label="serial throughput speedup") < 2.0:
        raise ValueError("site performance serial throughput gate failed")
    if str(raw.get("baseline_head", "")) != "46b6a446754e44b8b126911e759954720fe9438c":
        raise ValueError("site performance baseline HEAD is not the frozen R1 baseline")
    metrics = evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("site performance metrics are missing")
    for case in ("new_serial", "new_concurrent"):
        item = metrics.get(case)
        if not isinstance(item, Mapping):
            raise ValueError(f"site performance {case} metrics are missing")
        if _finite_number(item.get("observer_fraction"), label=f"{case} observer fraction") > 0.10:
            raise ValueError(f"site performance {case} observer overhead exceeds 10 percent")
        if int(item.get("full_manifest_writes", -1)) != 0:
            raise ValueError(f"site performance {case} rewrites a full checkpoint")
    concurrent = metrics["new_concurrent"]
    if _finite_number(concurrent.get("wait_max_s"), label="new concurrent max wait") >= 5.0:
        raise ValueError("site performance concurrent wait gate failed")
    unchanged = raw.get("unchanged_hot_paths")
    if not isinstance(unchanged, list):
        raise ValueError("site performance hot-path identity list is missing")
    seen: dict[str, str] = {}
    for item in unchanged:
        if not isinstance(item, Mapping):
            raise ValueError("site performance hot-path identity is invalid")
        relative = str(item.get("path", ""))
        digest = str(item.get("canonical_lf_sha256", ""))
        if relative not in PERFORMANCE_HOT_PATHS or not _SHA256.fullmatch(digest) or relative in seen:
            raise ValueError("site performance hot-path identity is invalid")
        source = (package_root / Path(*relative.split("/"))).resolve(strict=True)
        if not source.is_file() or source.is_symlink():
            raise ValueError("site performance hot-path source is not a regular file")
        if canonical_lf_sha256(source.read_bytes()) != digest:
            raise ValueError(f"site performance hot-path source changed: {relative}")
        seen[relative] = digest
    if set(seen) != PERFORMANCE_HOT_PATHS:
        raise ValueError("site performance evidence does not bind both streaming hot-path modules")
    return {"path": str(evidence_path), "raw_sha256": evidence_sha,
            "baseline_head": str(raw["baseline_head"]), "unchanged_hot_paths": seen}


def _validate_runtime_budget(
    raw: Any, *, campaign_root: Path,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("status") != "PASS_WITH_DOCUMENTED_ESTIMATION":
        raise ValueError("N3 runtime estimate status is not admissible")
    budget_path, budget, budget_sha = _read_hashed_json(
        raw.get("budget_path"), raw.get("budget_raw_sha256"),
        label="N3 runtime budget", campaign_root=campaign_root,
    )
    if budget.get("schema") != "khz_filament.hr4e5.n3_runtime_budget.v1":
        raise ValueError("N3 runtime budget schema is invalid")
    for key, expected in (("status", "PASS_WITH_DOCUMENTED_ESTIMATION"),
                          ("n_pulses", 3), ("k", 8048),
                          ("allocation_seconds", N3_FORMAL_SCOPE["allocation_seconds"]),
                          ("science_child_timeout_s", N3_FORMAL_SCOPE["science_child_timeout_s"]),
                          ("exact_field_pairs", 338016), ("metadata_keys", 30),
                          ("nonfinal_pairs", 2), ("terminal_pairs", 1)):
        if budget.get(key) != expected:
            raise ValueError(f"N3 runtime budget {key} differs from the formal scope")
    if budget.get("estimate_not_completion_guarantee") is not True:
        raise ValueError("N3 runtime estimate must retain its completion limitation")
    scenarios = budget.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("N3 runtime scenarios are missing")
    by_name = {str(row.get("name", "")): row for row in scenarios if isinstance(row, Mapping)}
    if set(by_name) != {"base", "conservative", "stress"}:
        raise ValueError("N3 runtime budget must include base, conservative and stress scenarios")
    conservative_total = None
    for name, row in by_name.items():
        components = row.get("components_s")
        if not isinstance(components, Mapping) or set(components) != {
                "pair_critical_s", "exact_s", "successor_reclaim_s", "init_terminal_s", "margin_s"}:
            raise ValueError(f"N3 runtime {name} components are incomplete")
        r_nonfinal = _finite_number(row.get("R_nonfinal_child_s"),
                                    label=f"{name} R nonfinal child", positive=True)
        c_critical = _finite_number(row.get("C_critical_each_s"),
                                    label=f"{name} C critical", positive=True)
        r_terminal = _finite_number(row.get("R_optical_each_s"),
                                    label=f"{name} terminal R optical", positive=True)
        pair = 2 * max(r_nonfinal, c_critical) + r_terminal
        _close(components.get("pair_critical_s"), pair, label=f"{name} pair critical time")
        exact = _finite_number(components.get("exact_s"), label=f"{name} exact time", positive=True)
        successor = _finite_number(components.get("successor_reclaim_s"),
                                   label=f"{name} successor/reclaim time", positive=True)
        init_terminal = _finite_number(components.get("init_terminal_s"),
                                       label=f"{name} init/terminal time", positive=True)
        margin = _finite_number(components.get("margin_s"), label=f"{name} margin", positive=True)
        fraction = _finite_number(row.get("margin_fraction"), label=f"{name} margin fraction", positive=True)
        _close(margin, (pair + exact + successor + init_terminal) * fraction,
               label=f"{name} margin")
        total = pair + exact + successor + init_terminal + margin
        _close(row.get("total_s"), total, label=f"{name} total")
        _close(row.get("total_h"), total / 3600.0, label=f"{name} hours")
        if name == "conservative":
            conservative_total = total
    if conservative_total is None:
        raise ValueError("conservative N3 runtime scenario is missing")
    _close(budget.get("conservative_total_s"), conservative_total,
           label="N3 conservative total")
    _close(raw.get("conservative_total_s"), conservative_total,
           label="manifest conservative total")
    if conservative_total > N3_FORMAL_SCOPE["allocation_seconds"]:
        raise ValueError("N3 conservative runtime estimate exceeds the 120-hour allocation")
    source = budget.get("source")
    if not isinstance(source, Mapping) or source.get("hash_scope") != "raw_bytes":
        raise ValueError("N3 runtime budget lacks its source identity")
    source_path = _external_regular_file(source.get("path"), label="N3 runtime source",
                                         campaign_root=campaign_root)
    source_digest = str(source.get("raw_sha256", ""))
    if not _SHA256.fullmatch(source_digest) or _raw_sha256(source_path) != source_digest:
        raise ValueError("N3 runtime source identity mismatch")
    source_budget = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = {row["name"]: row for row in source_budget["scenarios"]}
    for name, row in by_name.items():
        prior = source_rows[name]
        for key in ("R_nonfinal_child_s", "C_critical_each_s", "R_optical_each_s", "margin_fraction"):
            _close(row[key], prior[key], label=f"{name} source {key}")
        expected = {
            "exact_s": (prior["exact_total_s"]-prior["exact_metadata_s"])*338016/579456
                       + prior["exact_metadata_s"]*30/50,
            "successor_reclaim_s": prior["successor_reclaim_s"]/2,
            "init_terminal_s": prior["init_terminal_s"],
        }
        for key, value in expected.items():
            _close(row["components_s"][key], value, label=f"{name} source {key}")
    return {"path": str(budget_path), "raw_sha256": budget_sha,
            "conservative_total_s": conservative_total,
            "source_path": str(source_path), "source_raw_sha256": source_digest}


def validate_final_gate_manifest(
    manifest_path: str | os.PathLike[str], *,
    campaign_root: str | os.PathLike[str],
    package_root: str | os.PathLike[str],
    campaign_id: str,
    expected_inputs: Mapping[str, Mapping[str, str]],
    implementation_identity_sha256: str,
    journal_capacity_bound: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the bound evidence bundle for the N3 entry."""
    root = Path(campaign_root).resolve(strict=False)
    path = _external_regular_file(str(manifest_path), label="final gate manifest",
                                  campaign_root=root)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("final gate manifest JSON is unreadable") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != FINAL_GATES_SCHEMA:
        raise ValueError("final gate manifest schema is invalid")
    if manifest.get("campaign_id") != str(campaign_id):
        raise ValueError("final gate manifest campaign ID differs")
    if manifest.get("scope") != N3_FORMAL_SCOPE:
        raise ValueError("final gate manifest scope differs from the fixed N3 contract")
    if str(manifest.get("implementation_identity_sha256", "")) != str(implementation_identity_sha256):
        raise ValueError("final gate implementation identity differs from the running source")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or dict(inputs) != dict(expected_inputs):
        raise ValueError("final gate input identities differ from the production inputs")

    performance = _validate_performance(
        manifest.get("performance"), package_root=Path(package_root).resolve(strict=True),
        campaign_root=root,
    )
    runtime = _validate_runtime_budget(manifest.get("runtime"), campaign_root=root)

    storage = manifest.get("storage")
    if not isinstance(storage, Mapping) or storage.get("status") != "PASS":
        raise ValueError("journal storage capacity gate is not PASS")
    if not isinstance(storage.get("bound"), Mapping) or dict(storage["bound"]) != dict(journal_capacity_bound):
        raise ValueError("journal storage bound differs from the admission recomputation")
    capacity_path, capacity_report, capacity_sha = _read_hashed_json(
        storage.get("capacity_report_path"), storage.get("capacity_report_raw_sha256"),
        label="journal capacity report", campaign_root=root,
    )
    if capacity_report.get("status") != "PASS" or capacity_report.get("bound") != dict(journal_capacity_bound):
        raise ValueError("journal capacity report differs from the admission recomputation")
    return {
        "schema": FINAL_GATES_SCHEMA,
        "path": str(path),
        "raw_sha256": _raw_sha256(path),
        "campaign_id": str(campaign_id),
        "scope": dict(N3_FORMAL_SCOPE),
        "implementation_identity_sha256": str(implementation_identity_sha256),
        "inputs": {str(key): dict(value) for key, value in expected_inputs.items()},
        "performance": performance,
        "runtime": runtime,
        "storage": {"path": str(capacity_path), "raw_sha256": capacity_sha,
                    "bound": dict(journal_capacity_bound)},
    }


__all__ = [
    "FINAL_GATES_SCHEMA", "N3_FORMAL_SCOPE", "require_final_gate_path",
    "validate_final_gate_manifest",
]

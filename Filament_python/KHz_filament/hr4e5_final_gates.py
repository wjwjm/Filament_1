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

from tools.hpc_ops.provenance_v2 import (
    TRACKED_HASH_SCOPE, canonical_lf_sha256, raw_sha256_file, validate_record,
)


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
PERFORMANCE_REQUALIFICATION_SCHEMA = "n3.performance_requalification.v1"
PERFORMANCE_REQUALIFICATION_TOOLS = frozenset({
    "site_perf_requal.sbatch",
    "site_control_driver_n3.py",
    "site_fixture_requal.py",
    "control_benchmark_v9.py",
    "analyze_control_v4.py",
    "evaluate_performance.py",
    "site_control_driver_v7_ref.py",
    "baseline_journal.py",
})
PERFORMANCE_REQUALIFICATION_SOURCES = frozenset({
    "Filament_python/KHz_filament/hr4e5_streaming_journal.py",
    "Filament_python/KHz_filament/hr4e5s_streaming.py",
})
NORMAL_MAX_JOURNAL_BYTES = 211863864
TERMINAL_MAX_JOURNAL_BYTES = 278972728
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_BLOB = re.compile(r"^[0-9a-f]{40}$")


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


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


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
    has_revalidation_path = "revalidation_path" in raw
    has_revalidation_sha = "revalidation_raw_sha256" in raw
    if has_revalidation_path != has_revalidation_sha:
        raise ValueError("performance revalidation path and raw SHA256 must be supplied together")
    if has_revalidation_path:
        revalidation_path = raw.get("revalidation_path")
        revalidation_sha = raw.get("revalidation_raw_sha256")
        if revalidation_path is None or revalidation_sha is None:
            raise ValueError("performance revalidation path and raw SHA256 must be supplied together")
        revalidation = _validate_performance_requalification(
            revalidation_path, revalidation_sha, evidence=evidence,
            evidence_path=evidence_path, evidence_sha=evidence_sha,
            package_root=package_root, campaign_root=campaign_root,
        )
        return {"path": str(evidence_path), "raw_sha256": evidence_sha,
                "baseline_head": str(raw["baseline_head"]),
                "revalidation": revalidation}

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


def _validate_performance_requalification(
    raw_path: Any, expected_sha256: Any, *, evidence: Mapping[str, Any],
    evidence_path: Path, evidence_sha: str, package_root: Path, campaign_root: Path,
) -> dict[str, Any]:
    path, receipt, receipt_sha = _read_hashed_json(
        raw_path, expected_sha256, label="performance requalification receipt",
        campaign_root=campaign_root,
    )
    if receipt.get("schema") != PERFORMANCE_REQUALIFICATION_SCHEMA:
        raise ValueError("performance requalification receipt schema is invalid")
    if receipt.get("status") != "PASS_PERFORMANCE_REQUALIFICATION_ONLY":
        raise ValueError("performance requalification did not pass")
    if receipt.get("performance_allocation_only") is not True:
        raise ValueError("performance requalification is not allocation-only")
    if _nonnegative_integer(receipt.get("science_calls"), label="requalification science_calls") != 0:
        raise ValueError("performance requalification includes a science call")
    if _nonnegative_integer(receipt.get("formal_submissions"), label="requalification formal_submissions") != 0:
        raise ValueError("performance requalification includes a formal submission")

    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("performance requalification runtime identity is missing")
    job_id = runtime.get("slurm_job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("performance requalification Slurm job ID is not numeric")
    if _nonnegative_integer(runtime.get("slurm_cpus_per_task"),
                            label="requalification Slurm CPUs") != 8:
        raise ValueError("performance requalification did not use 8 CPUs")
    if _nonnegative_integer(runtime.get("allocated_gpu_count"),
                            label="requalification allocated GPUs") != 1:
        raise ValueError("performance requalification did not use exactly 1 GPU")
    if runtime.get("UPPE_USE_GPU") != "0":
        raise ValueError("performance requalification did not disable UPPE GPU execution")

    gate = receipt.get("performance_gate")
    if not isinstance(gate, Mapping) or gate.get("status") != "PASS":
        raise ValueError("requalification receipt does not bind a passing performance gate")
    if gate.get("raw_sha256") != evidence_sha:
        raise ValueError("requalification performance gate hash differs from the hashed gate")
    gate_checks = gate.get("checks")
    gate_ratios = gate.get("ratios")
    if gate_checks != evidence.get("checks") or gate_ratios != evidence.get("ratios"):
        raise ValueError("requalification performance gate contents differ from the hashed gate")
    gate_path = _external_regular_file(
        gate.get("path"), label="requalification performance gate", campaign_root=campaign_root,
    )
    if _raw_sha256(gate_path) != evidence_sha:
        raise ValueError("requalification performance gate file hash mismatch")

    source_identities = receipt.get("source_identities")
    runtime_manifest = source_identities.get("runtime_manifest") if isinstance(source_identities, Mapping) else None
    sources = runtime_manifest.get("journal_and_lifecycle") if isinstance(runtime_manifest, Mapping) else None
    source_head = runtime_manifest.get("head") if isinstance(runtime_manifest, Mapping) else None
    if not isinstance(source_head, str) or not _GIT_BLOB.fullmatch(source_head):
        raise ValueError("requalification runtime source HEAD is invalid")
    if not isinstance(sources, Mapping) or set(sources) != PERFORMANCE_REQUALIFICATION_SOURCES:
        raise ValueError("requalification receipt lacks journal and lifecycle source identities")
    source_hashes: dict[str, str] = {}
    for relative, source_record in sources.items():
        if not isinstance(source_record, Mapping):
            raise ValueError("requalification source identity is invalid")
        digest = str(source_record.get("canonical_lf_sha256", ""))
        if not _SHA256.fullmatch(digest) or source_record.get("hash_scope") != "git_blob_oid+canonical_lf_sha256":
            raise ValueError(f"requalification canonical source identity is invalid: {relative}")
        if not _GIT_BLOB.fullmatch(str(source_record.get("git_blob_oid", ""))):
            raise ValueError(f"requalification Git blob identity is invalid: {relative}")
        module_relative = relative.removeprefix("Filament_python/")
        source = package_root
        for part in module_relative.split("/"):
            source = source / part
            if source.is_symlink():
                raise ValueError(f"requalification source path contains a symlink: {relative}")
        source = source.resolve(strict=True)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"requalification source is not a regular module: {relative}")
        if canonical_lf_sha256(source.read_bytes()) != digest:
            raise ValueError(f"requalification source differs from the running module: {relative}")
        try:
            validate_record(
                package_root.parent,
                {"path": relative, "classification": "tracked_text",
                 "hash_scope": TRACKED_HASH_SCOPE,
                 "git_blob_oid": source_record["git_blob_oid"],
                 "canonical_lf_sha256": digest},
                require_hash_scope=True,
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"requalification Git blob identity differs from the running source: {relative}") from error
        source_hashes[relative] = digest

    fixture_info = receipt.get("journal_fixture")
    if not isinstance(fixture_info, Mapping):
        raise ValueError("requalification site fixture identity is missing")
    fixture_path, fixture, fixture_sha = _read_hashed_json(
        fixture_info.get("path"), fixture_info.get("raw_sha256"),
        label="journal requalification fixture", campaign_root=campaign_root,
    )
    if fixture_info.get("status") != "PASS_SITE_FIXTURE_ONLY" or fixture.get("status") != "PASS_SITE_FIXTURE_ONLY":
        raise ValueError("journal requalification fixture did not pass")
    if fixture.get("schema") != "n3.journal_performance_requalification_fixture.v2":
        raise ValueError("journal requalification fixture schema is invalid")
    if _nonnegative_integer(fixture.get("science_calls"), label="fixture science_calls") != 0 \
            or _nonnegative_integer(fixture.get("formal_submissions"),
                                    label="fixture formal_submissions") != 0:
        raise ValueError("journal requalification fixture includes a science call or formal submission")
    if _nonnegative_integer(fixture.get("normal_max_journal_bytes"),
                            label="normal fixture journal bytes") != NORMAL_MAX_JOURNAL_BYTES:
        raise ValueError("journal fixture normal capacity size differs from the N3 bound")
    if _nonnegative_integer(fixture.get("terminal_max_journal_bytes"),
                            label="terminal fixture journal bytes") != TERMINAL_MAX_JOURNAL_BYTES:
        raise ValueError("journal fixture terminal capacity size differs from the N3 bound")
    authority_cases = fixture.get("metadata_authority_cases")
    if not isinstance(authority_cases, Mapping) \
            or not isinstance(authority_cases.get("pre_repair"), Mapping) \
            or authority_cases["pre_repair"].get("status") != "PASS_EXPECTED_REJECT" \
            or not isinstance(authority_cases.get("repaired"), Mapping) \
            or authority_cases["repaired"].get("status") != "PASS_ACCEPT_AND_REHASH":
        raise ValueError("journal fixture lacks timestamp-only metadata authority regression")
    corruption = fixture.get("same_length_corruption")
    if not isinstance(corruption, Mapping) or corruption.get("status") != "PASS_REJECT_SAME_LENGTH_CORRUPTION":
        raise ValueError("journal fixture lacks same-length corruption rejection")
    capacity_rows = fixture.get("capacity_sized_valid_journal_rechecks")
    if not isinstance(capacity_rows, list) or len(capacity_rows) != 2:
        raise ValueError("journal fixture capacity rechecks are incomplete")
    expected_rows = {
        "normal_max_C_p0_journal": NORMAL_MAX_JOURNAL_BYTES,
        "terminal_max_single_journal": TERMINAL_MAX_JOURNAL_BYTES,
    }
    measured: dict[str, float] = {}
    for row in capacity_rows:
        if not isinstance(row, Mapping):
            raise ValueError("journal fixture capacity recheck is invalid")
        label = row.get("label")
        if label not in expected_rows or label in measured:
            raise ValueError("journal fixture capacity recheck labels are invalid")
        if _nonnegative_integer(row.get("journal_bytes"),
                                label=f"{label} journal bytes") != expected_rows[label]:
            raise ValueError(f"journal fixture size differs from the formal bound: {label}")
        if row.get("status") != "PASS_VALID_SYNTHETIC_AUTHORITY_BOUNDARY":
            raise ValueError(f"journal fixture capacity recheck did not pass: {label}")
        if _nonnegative_integer(row.get("metadata_refresh_digest_recheck_count"),
                                label=f"{label} digest recheck count") != 1:
            raise ValueError(f"journal fixture did not perform one measured digest recheck: {label}")
        measured[label] = _finite_number(
            row.get("metadata_refresh_elapsed_s"), label=f"{label} recheck time", positive=True,
        )
    if set(measured) != set(expected_rows):
        raise ValueError("journal fixture normal or terminal capacity class is missing")
    measured_normal = measured["normal_max_C_p0_journal"]
    measured_terminal = measured["terminal_max_single_journal"]
    measured_max = max(measured_normal, measured_terminal)
    candidate_module = fixture.get("candidate_module")
    if not isinstance(candidate_module, Mapping):
        raise ValueError("journal fixture candidate module identity is missing")
    journal_relative = "Filament_python/KHz_filament/hr4e5_streaming_journal.py"
    candidate_path = _external_regular_file(
        candidate_module.get("path"), label="journal fixture candidate module",
        campaign_root=campaign_root,
    )
    expected_candidate_path = (path.parent / "source" / Path(*journal_relative.split("/"))).resolve(strict=True)
    if candidate_path != expected_candidate_path:
        raise ValueError("journal fixture candidate path differs from the requalified source")
    candidate_sha = str(candidate_module.get("raw_sha256", ""))
    if not _SHA256.fullmatch(candidate_sha) or candidate_sha != source_hashes[journal_relative]:
        raise ValueError("journal fixture candidate raw hash differs from journal canonical-LF identity")
    if _raw_sha256(candidate_path) != candidate_sha:
        raise ValueError("journal fixture candidate module raw hash mismatch")
    if not math.isclose(
            _finite_number(fixture_info.get("max_single_recheck_s"),
                           label="reported maximum recheck time", positive=True),
            measured_max, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("requalification maximum recheck time differs from its fixture")
    if fixture_info.get("raw_sha256") != fixture_sha:
        raise ValueError("journal fixture hash changed during validation")

    tool_hashes = receipt.get("tool_raw_sha256")
    if not isinstance(tool_hashes, Mapping) or set(tool_hashes) != PERFORMANCE_REQUALIFICATION_TOOLS:
        raise ValueError("requalification receipt tool hash inventory is incomplete")
    tool_identity: dict[str, str] = {}
    for name in sorted(PERFORMANCE_REQUALIFICATION_TOOLS):
        expected = str(tool_hashes.get(name, ""))
        if not _SHA256.fullmatch(expected):
            raise ValueError(f"requalification tool raw SHA256 is invalid: {name}")
        tool_path = _external_regular_file(
            str(path.parent / name), label=f"performance requalification tool {name}",
            campaign_root=campaign_root,
        )
        if _raw_sha256(tool_path) != expected:
            raise ValueError(f"performance requalification tool hash mismatch: {name}")
        tool_identity[name] = expected

    baseline_module = fixture.get("baseline_module")
    if not isinstance(baseline_module, Mapping):
        raise ValueError("journal fixture baseline module identity is missing")
    baseline_path = _external_regular_file(
        baseline_module.get("path"), label="journal fixture baseline module",
        campaign_root=campaign_root,
    )
    expected_baseline_path = (path.parent / "baseline_journal.py").resolve(strict=True)
    baseline_sha = str(baseline_module.get("raw_sha256", ""))
    if baseline_path != expected_baseline_path or baseline_sha != tool_identity["baseline_journal.py"]:
        raise ValueError("journal fixture baseline module differs from its hashed baseline tool")
    if _raw_sha256(baseline_path) != baseline_sha:
        raise ValueError("journal fixture baseline module raw hash mismatch")

    return {
        "path": str(path), "raw_sha256": receipt_sha, "slurm_job_id": job_id,
        "source_head": source_head,
        "performance_gate_raw_sha256": evidence_sha,
        "fixture": {"path": str(fixture_path), "raw_sha256": fixture_sha,
                    "status": fixture["status"],
                    "normal_max_recheck_s": measured_normal,
                    "terminal_max_recheck_s": measured_terminal},
        "journal_and_lifecycle": source_hashes,
        "tool_raw_sha256": tool_identity,
    }


def _validate_runtime_budget(
    raw: Any, *, campaign_root: Path,
    performance_revalidation: Mapping[str, Any] | None = None,
    journal_capacity_bound: Mapping[str, Any] | None = None,
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
    journal_revalidation_raw = budget.get("journal_revalidation")
    if performance_revalidation is None:
        if journal_revalidation_raw is not None:
            raise ValueError("journal revalidation budget lacks matching performance requalification")
        journal_revalidation = None
    else:
        if not isinstance(journal_revalidation_raw, Mapping):
            raise ValueError("N3 runtime budget lacks journal revalidation overhead")
        if not isinstance(journal_capacity_bound, Mapping):
            raise ValueError("N3 journal revalidation counts lack the recomputed capacity bound")
        from .hr4e5_journal_capacity import (
            N3_BACKPRESSURE_TRANSACTIONS, N3_BASE_TRANSACTIONS, N3_K,
            N3_PULSES, N3_ROOT_COUNT,
        )
        if (journal_capacity_bound.get("n_pulses") != N3_PULSES
                or journal_capacity_bound.get("k") != N3_K
                or journal_capacity_bound.get("lifecycle_roots") != N3_ROOT_COUNT):
            raise ValueError("journal revalidation counts require the fixed N3/K8048 six-root capacity bound")
        capacity_counts = journal_capacity_bound.get("transaction_counts")
        if not isinstance(capacity_counts, Mapping):
            raise ValueError("N3 capacity transaction counts are missing")
        base_count = _nonnegative_integer(capacity_counts.get("base"), label="N3 base capacity transactions")
        backpressure_count = _nonnegative_integer(
            capacity_counts.get("backpressure_max"), label="N3 backpressure capacity transactions")
        if base_count != N3_BASE_TRANSACTIONS or backpressure_count != N3_BACKPRESSURE_TRANSACTIONS:
            raise ValueError("N3 capacity transaction counts differ from the fixed lifecycle derivation")
        expected_recheck_counts = {
            "base": base_count,
            "conservative": backpressure_count,
            "stress": 2 * backpressure_count,
        }
        required_revalidation_fields = {
            "revalidation_path", "revalidation_raw_sha256", "normal_max_recheck_s",
            "terminal_max_recheck_s", "scenario_recheck_counts",
            "terminal_recheck_count", "planning_basis",
        }
        if set(journal_revalidation_raw) != required_revalidation_fields:
            raise ValueError("N3 journal revalidation budget fields are incomplete or unexpected")
        revalidation_path = _external_regular_file(
            journal_revalidation_raw.get("revalidation_path"),
            label="N3 runtime journal revalidation receipt", campaign_root=campaign_root,
        )
        if str(revalidation_path) != str(performance_revalidation.get("path")):
            raise ValueError("N3 runtime budget references a different performance requalification receipt")
        revalidation_sha = str(journal_revalidation_raw.get("revalidation_raw_sha256", ""))
        if revalidation_sha != performance_revalidation.get("raw_sha256"):
            raise ValueError("N3 runtime budget requalification receipt hash differs from performance gate")
        normal_max = _finite_number(
            journal_revalidation_raw.get("normal_max_recheck_s"),
            label="N3 normal maximum journal recheck time", positive=True,
        )
        terminal_max = _finite_number(
            journal_revalidation_raw.get("terminal_max_recheck_s"),
            label="N3 terminal maximum journal recheck time", positive=True,
        )
        fixture = performance_revalidation.get("fixture")
        if not isinstance(fixture, Mapping):
            raise ValueError("performance requalification lacks measured fixture times")
        if normal_max < _finite_number(
                fixture.get("normal_max_recheck_s"), label="measured normal recheck time", positive=True):
            raise ValueError("N3 normal recheck budget underestimates the measured maximum")
        if terminal_max < _finite_number(
                fixture.get("terminal_max_recheck_s"), label="measured terminal recheck time", positive=True):
            raise ValueError("N3 terminal recheck budget underestimates the measured maximum")
        counts = journal_revalidation_raw.get("scenario_recheck_counts")
        if not isinstance(counts, Mapping) or set(counts) != {"base", "conservative", "stress"}:
            raise ValueError("N3 journal revalidation scenario counts are incomplete")
        normalized_counts = {
            name: _nonnegative_integer(counts[name], label=f"{name} normal journal rechecks")
            for name in ("base", "conservative", "stress")
        }
        if normalized_counts != expected_recheck_counts:
            raise ValueError("N3 journal recheck counts understate or differ from the selected lifecycle budget")
        terminal_count = _nonnegative_integer(
            journal_revalidation_raw.get("terminal_recheck_count"),
            label="terminal journal rechecks",
        )
        if terminal_count != 6:
            raise ValueError("N3 terminal journal recheck count must cover all six lifecycle roots")
        planning_basis = journal_revalidation_raw.get("planning_basis")
        if not isinstance(planning_basis, Mapping) or set(planning_basis) != {"base", "conservative", "stress"}:
            raise ValueError("N3 journal revalidation planning basis is incomplete")
        normalized_basis: dict[str, str] = {}
        for name in ("base", "conservative", "stress"):
            basis = planning_basis.get(name)
            if not isinstance(basis, str) or not basis.strip():
                raise ValueError(f"N3 {name} journal revalidation planning basis is undocumented")
            normalized_basis[name] = basis.strip()
        journal_revalidation = {
            "path": str(revalidation_path), "raw_sha256": revalidation_sha,
            "normal_max_recheck_s": normal_max,
            "terminal_max_recheck_s": terminal_max,
            "scenario_recheck_counts": normalized_counts,
            "capacity_transaction_counts": expected_recheck_counts,
            "capacity_scope": {"n_pulses": N3_PULSES, "k": N3_K,
                               "lifecycle_roots": N3_ROOT_COUNT},
            "terminal_recheck_count": terminal_count,
            "planning_basis": normalized_basis,
        }
    conservative_total = None
    journal_revalidation_totals: dict[str, float] = {}
    for name, row in by_name.items():
        components = row.get("components_s")
        expected_components = {
            "pair_critical_s", "exact_s", "successor_reclaim_s", "init_terminal_s", "margin_s"
        }
        if journal_revalidation is not None:
            expected_components.add("journal_revalidation_s")
        if not isinstance(components, Mapping) or set(components) != expected_components:
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
        revalidation_overhead = 0.0
        if journal_revalidation is not None:
            revalidation_overhead = (
                journal_revalidation["normal_max_recheck_s"]
                * journal_revalidation["scenario_recheck_counts"][name]
                + journal_revalidation["terminal_max_recheck_s"]
                * journal_revalidation["terminal_recheck_count"]
            )
            _close(components.get("journal_revalidation_s"), revalidation_overhead,
                   label=f"{name} journal revalidation overhead")
            journal_revalidation_totals[name] = revalidation_overhead
        margin = _finite_number(components.get("margin_s"), label=f"{name} margin", positive=True)
        fraction = _finite_number(row.get("margin_fraction"), label=f"{name} margin fraction", positive=True)
        _close(margin, (pair + exact + successor + init_terminal + revalidation_overhead) * fraction,
               label=f"{name} margin")
        total = pair + exact + successor + init_terminal + revalidation_overhead + margin
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
    normalized_runtime = {"path": str(budget_path), "raw_sha256": budget_sha,
                          "conservative_total_s": conservative_total,
                          "source_path": str(source_path), "source_raw_sha256": source_digest}
    if journal_revalidation is not None:
        normalized_runtime["journal_revalidation"] = {
            **journal_revalidation,
            "scenario_overhead_s": journal_revalidation_totals,
        }
    return normalized_runtime


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
    performance_revalidation = performance.get("revalidation")
    runtime = _validate_runtime_budget(
        manifest.get("runtime"), campaign_root=root,
        performance_revalidation=performance_revalidation,
        journal_capacity_bound=journal_capacity_bound,
    )

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

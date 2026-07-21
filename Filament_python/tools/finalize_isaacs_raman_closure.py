#!/usr/bin/env python3
"""Gate helpers and finalizer for Isaacs Raman closure audits.

Numerical gates are derived from their own evidence files.  A missing file,
missing/invalid metric, NaN, or Inf can never become a passing gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "isaacs_raman_closure" / "phase8a_static_closure"
VALID_GATE_STATES = {"passed", "failed", "inconclusive", "not_applicable"}


class MetricSchemaError(ValueError):
    """Raised when an evidence CSV exists but does not match its contract."""


@dataclass(frozen=True)
class MetricResult:
    value: float | None
    reason: str
    sample_count: int = 0


def gate(status, evidence, numerical_result, threshold, comparison_operator,
         physical_impact, production_impact, required_action):
    if status not in VALID_GATE_STATES:
        raise ValueError(f"invalid gate status: {status}")
    return {
        "status": status,
        "evidence": evidence,
        "numerical_result": numerical_result,
        "threshold": threshold,
        "comparison_operator": comparison_operator,
        "physical_impact": physical_impact,
        "production_impact": production_impact,
        "required_action": required_action,
    }


def threshold_gate(value, threshold, *, mode="lt"):
    """Return an automatically derived status and comparison record."""
    try:
        numeric = float(value)
        limit = float(threshold)
    except (TypeError, ValueError):
        return {"status": "inconclusive", "value": value, "threshold": threshold,
                "comparison_operator": mode, "comparison_result": None}
    if not math.isfinite(numeric) or not math.isfinite(limit):
        return {"status": "inconclusive", "value": numeric, "threshold": limit,
                "comparison_operator": mode, "comparison_result": None}
    operators: Mapping[str, Callable[[float, float], bool]] = {
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
    }
    if mode not in operators:
        raise ValueError(f"unsupported comparison mode: {mode}")
    result = bool(operators[mode](numeric, limit))
    return {"status": "passed" if result else "failed", "value": numeric,
            "threshold": limit, "comparison_operator": mode,
            "comparison_result": result}


def read_metric(path: Path, value_column: str, *, filters: Mapping[str, str] | None = None,
                reducer: Callable[[Iterable[float]], float] = max) -> MetricResult:
    """Read one metric contract; missing files are inconclusive, bad schemas fail loudly."""
    path = Path(path)
    if not path.is_file():
        return MetricResult(None, "missing_file", 0)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {value_column, *(filters or {}).keys()}
        missing = sorted(required - fields)
        if missing:
            raise MetricSchemaError(f"{path.name} missing required columns: {missing}")
        values = []
        for row in reader:
            if filters and any(str(row[key]) != str(expected) for key, expected in filters.items()):
                continue
            try:
                values.append(float(row[value_column]))
            except (TypeError, ValueError):
                return MetricResult(None, f"invalid_value:{value_column}", len(values))
    if not values:
        return MetricResult(None, "no_matching_rows", 0)
    value = float(reducer(values))
    if not math.isfinite(value):
        return MetricResult(None, "non_finite", len(values))
    return MetricResult(value, "ok", len(values))


def metric_gate(path: Path, column: str, threshold: float, *, filters=None, mode="lt",
                evidence_label=None, physical_impact="", production_impact="",
                required_action="inspect evidence"):
    try:
        metric = read_metric(path, column, filters=filters)
    except MetricSchemaError as exc:
        metric = MetricResult(None, f"schema_error:{exc}", 0)
    comparison = threshold_gate(metric.value, threshold, mode=mode)
    return gate(
        comparison["status"], evidence_label or str(path),
        {"value": metric.value, "reason": metric.reason, "sample_count": metric.sample_count,
         "comparison_result": comparison["comparison_result"]},
        threshold, mode, physical_impact, production_impact, required_action,
    )


def build_numeric_gates(out_dir: Path):
    """Build independent gates without reusing semantically unrelated columns."""
    return {
        "fft_linear_convolution_gate": metric_gate(
            out_dir / "raman_fft_direct_comparison.csv", "relative_linf_error", 1e-10,
            filters={"dtype": "float64"}, evidence_label="raman_fft_direct_comparison.csv",
            physical_impact="causal convolution accuracy", production_impact="FFT Raman path",
            required_action="repair FFT convolution evidence if failed/inconclusive"),
        "eq11_analytic_recovery_gate": metric_gate(
            out_dir / "eq10_eq11_validation.csv", "direct_vs_eq11_error", .01,
            evidence_label="eq10_eq11_validation.csv", physical_impact="Eq.10/Eq.11 closure",
            production_impact="Raman energy reference", required_action="refine Eq.10/Eq.11 audit"),
        "iir_convergence_gate": metric_gate(
            out_dir / "raman_iir_direct_convergence.csv", "iir_vs_direct_error", .01,
            evidence_label="raman_iir_direct_convergence.csv", physical_impact="IIR response accuracy",
            production_impact="legacy/current IIR convolution", required_action="repair or reject IIR"),
        "production_split_comparison_gate": metric_gate(
            out_dir / "production_split_vs_full_operator.csv", "gate_error", .02,
            evidence_label="production_split_vs_full_operator.csv", physical_impact="actual split/full equivalence",
            production_impact="candidate architecture selection", required_action="select full operator if failed"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gates = build_numeric_gates(args.out_dir)
    required = ("fft_linear_convolution_gate", "eq11_analytic_recovery_gate",
                "iir_convergence_gate", "production_split_comparison_gate")
    admission = "passed" if all(gates[name]["status"] == "passed" for name in required) else "failed"
    gates["propagation_admission_gate"] = gate(
        admission, "aggregate corrected numerical gates",
        {name: gates[name]["status"] for name in required}, "all required gates passed", "all",
        "controls Phase 8B admission", "blocks or permits production propagation",
        "resolve every failed or inconclusive prerequisite" if admission != "passed" else "none")
    (args.out_dir / "phase8a_gate_summary.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    return gates


if __name__ == "__main__":
    main()

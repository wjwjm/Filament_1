#!/usr/bin/env python3
"""Deterministic HR-4E-1 case comparison and tolerance classification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_timestep import (  # noqa: E402
    E1A_CENTROID_TOLERANCE_M,
    E1A_EXTREME_RELATIVE_TOLERANCE,
    E1A_PRIMARY_HORIZONS_US,
    E1A_WIDTH_RELATIVE_TOLERANCE,
    E1B_PRIMARY_HORIZONS_US,
    E1_BOUNDARY_FIRST_RING_RATIO_LIMIT,
    E1_BOUNDARY_SIGMA_Y_TOP_CLEARANCE_LIMIT,
    json_safe,
)


OBSERVABLES = (
    "yc_m",
    "sigma_x_m",
    "sigma_y_m",
    "min_delta_n",
    "max_abs_vy_m_s",
    "max_abs_v_m_s",
)
PRIMARY_OBSERVABLES = ("yc_m", "sigma_x_m", "sigma_y_m")
EXTREME_OBSERVABLES = ("min_delta_n", "max_abs_vy_m_s", "max_abs_v_m_s")
TOLERANCE_KIND = {
    "yc_m": "absolute",
    "sigma_x_m": "relative",
    "sigma_y_m": "relative",
    "min_delta_n": "relative",
    "max_abs_vy_m_s": "relative",
    "max_abs_v_m_s": "relative",
}
DT_KEYS = {"dt_hydro_s", "dt_hydro", "dt_hydro_us"}
SCREEN_KEYS = {
    "screen",
    "screen_id",
    "screen_index",
    "screen_z_m",
    "screen_z_position_m",
    "screen_identity",
}


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: [_plain(item) for item in value]}
    return {prefix: value}


def _plain(value: Any) -> Any:
    """Return a deterministic JSON-like value for list-valued config leaves."""
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return bool(math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15))
        except (TypeError, ValueError):
            return left == right
    return left == right


def _allowed_path(path: str, *, allow_screen_identity: bool) -> bool:
    leaf = path.rsplit(".", 1)[-1]
    if leaf in DT_KEYS or leaf.endswith("dt_hydro_s") or leaf.endswith("dt_hydro_us"):
        return True
    # ``allow_screen_identity`` is retained for callers of the initial draft,
    # but a convergence family must never compare different source screens.
    # Screen identity is therefore an ordinary guarded configuration field.
    return False


def config_diff(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    allow_screen_identity: bool = False,
) -> list[dict[str, Any]]:
    """Return only differences, retaining whether each is an allowed dt/screen field."""
    left = _flatten(reference)
    right = _flatten(candidate)
    rows: list[dict[str, Any]] = []
    for path in sorted(set(left) | set(right)):
        before, after = left.get(path), right.get(path)
        if _same_value(before, after):
            continue
        rows.append(
            {
                "path": path,
                "before": before,
                "after": after,
                "allowed": _allowed_path(path, allow_screen_identity=allow_screen_identity),
            }
        )
    return rows


def _case_items(cases: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    if isinstance(cases, Mapping):
        return [(str(key), value) for key, value in cases.items()]
    result = []
    for index, case in enumerate(cases):
        label = str(case.get("case_id", case.get("id", index)))
        result.append((label, case))
    return result


def _family_signature(case: Mapping[str, Any]) -> dict[str, Any]:
    """Extract identity that must remain fixed across one dt family."""
    configuration = case.get("configuration", case.get("config", {}))
    if not isinstance(configuration, Mapping):
        configuration = {}
    execution = configuration.get("execution", {})
    if not isinstance(execution, Mapping):
        execution = {}
    initial_state = case.get("initial_state", {})
    if not isinstance(initial_state, Mapping):
        initial_state = {}
    configured_initial = configuration.get("initial_state", {})
    if not isinstance(configured_initial, Mapping):
        configured_initial = {}

    def first(*values: Any) -> Any:
        for value in values:
            if value not in (None, ""):
                return value
        return None

    benchmark = first(case.get("benchmark"), configuration.get("benchmark"))
    signature = {
        "benchmark": benchmark,
        "screen_identity": _plain(
            first(configuration.get("screen_identity"), initial_state.get("screen_identity"))
        ),
        "initial_state_sha256": first(
            case.get("initial_state_sha256"), initial_state.get("delta_n_sha256")
        ),
        "source_file_sha256": first(
            initial_state.get("source_file_sha256"),
            initial_state.get("source_sha256"),
            configured_initial.get("source_file_sha256"),
        ),
        "source_array_sha256": first(
            initial_state.get("source_array_sha256"),
            configured_initial.get("source_array_sha256"),
        ),
        "source_state_file_sha256": first(
            initial_state.get("source_state_file_sha256"),
            configured_initial.get("source_state_file_sha256"),
        ),
        "source_state_array_sha256": first(
            initial_state.get("source_state_array_sha256"),
            configured_initial.get("source_state_array_sha256"),
        ),
        "dtype": first(case.get("dtype"), execution.get("dtype"), initial_state.get("dtype")),
        "backend": first(case.get("backend"), execution.get("backend")),
        "git_sha": first(case.get("git_sha"), execution.get("git_sha")),
    }
    return {key: _plain(value) for key, value in signature.items()}


def _signature_required_fields(signature: Mapping[str, Any]) -> tuple[str, ...]:
    required = ["benchmark", "initial_state_sha256", "dtype", "backend", "git_sha"]
    if str(signature.get("benchmark", "")).upper().startswith("E1-B"):
        required.extend(
            [
                "screen_identity",
                "source_file_sha256",
                "source_array_sha256",
                "source_state_file_sha256",
                "source_state_array_sha256",
            ]
        )
    return tuple(required)


def config_diff_guard(
    cases: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    allow_screen_identity: bool = False,
) -> dict[str, Any]:
    """Guard a convergence family against non-timestep or source drift.

    Only the hydrodynamic timestep may differ.  Benchmark, screen/source
    identity, dtype, backend, Git SHA, and all other configuration metadata are
    fixed family identity; ``allow_screen_identity`` cannot relax this rule.
    """
    items = _case_items(cases)
    if not items:
        return {"pass": False, "reason": "no cases", "comparisons": []}
    reference_name, reference_case = items[0]
    reference_config = reference_case.get("configuration", reference_case.get("config"))
    if not isinstance(reference_config, Mapping):
        return {"pass": False, "reason": f"{reference_name} has no configuration", "comparisons": []}
    reference_signature = _family_signature(reference_case)
    reference_required = _signature_required_fields(reference_signature)
    missing_reference = [
        key for key in reference_required if reference_signature.get(key) in (None, "")
    ]
    comparisons = []
    all_pass = not missing_reference
    for name, case in items[1:]:
        config = case.get("configuration", case.get("config"))
        if not isinstance(config, Mapping):
            differences = [{"path": "configuration", "before": "mapping", "after": None, "allowed": False}]
        else:
            differences = config_diff(
                reference_config, config, allow_screen_identity=allow_screen_identity
            )
        candidate_signature = _family_signature(case)
        required_fields = tuple(
            sorted(set(reference_required) | set(_signature_required_fields(candidate_signature)))
        )
        for key in required_fields:
            before = reference_signature.get(key)
            after = candidate_signature.get(key)
            if not _same_value(before, after):
                differences.append(
                    {
                        "path": f"family_identity.{key}",
                        "before": before,
                        "after": after,
                        "allowed": False,
                    }
                )
        missing_candidate = [
            key for key in _signature_required_fields(candidate_signature)
            if candidate_signature.get(key) in (None, "")
        ]
        if missing_reference:
            differences.extend(
                {
                    "path": f"family_identity.reference.{key}",
                    "before": None,
                    "after": None,
                    "allowed": False,
                }
                for key in missing_reference
            )
        for key in missing_candidate:
            differences.append(
                {
                    "path": f"family_identity.candidate.{key}",
                    "before": None,
                    "after": None,
                    "allowed": False,
                }
            )
        unexpected = [row for row in differences if not row["allowed"]]
        passed = not unexpected
        all_pass = all_pass and passed
        comparisons.append(
            {
                "reference_case": reference_name,
                "candidate_case": name,
                "pass": passed,
                "differences": differences,
                "unexpected_differences": unexpected,
            }
        )
    return {"pass": all_pass, "reference_case": reference_name, "comparisons": comparisons}


def assert_config_diff_guard(
    cases: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    allow_screen_identity: bool = False,
) -> dict[str, Any]:
    result = config_diff_guard(cases, allow_screen_identity=allow_screen_identity)
    if not result.get("pass", False):
        raise ValueError("HR-4E-1 configuration guard failed: " + json.dumps(result, sort_keys=True))
    return result


# Compatibility aliases for small external audit scripts.
guard_config_diff = config_diff_guard
compare_config_diff = config_diff


def _case_dt_s(case: Mapping[str, Any]) -> float:
    config = case.get("configuration", case.get("config", {}))
    for source in (case, config):
        if not isinstance(source, Mapping):
            continue
        for key in ("dt_hydro_s", "dt_hydro"):
            if key in source:
                return _finite_float(source[key], key)
        if "dt_hydro_us" in source:
            return _finite_float(source["dt_hydro_us"], "dt_hydro_us") * 1.0e-6
    raise ValueError("case has no dt_hydro value")


def _dt_label(dt_s: float) -> str:
    return f"{dt_s * 1.0e6:.12g}us"


def _snapshot_map(case: Mapping[str, Any]) -> dict[float, Mapping[str, Any]]:
    snapshots = case.get("snapshots", case.get("observables", ()))
    if isinstance(snapshots, Mapping):
        snapshots = list(snapshots.values())
    result: dict[float, Mapping[str, Any]] = {}
    for item in snapshots or ():
        if not isinstance(item, Mapping):
            continue
        if "time_us" in item:
            time_us = _finite_float(item["time_us"], "time_us")
        elif "time_s" in item:
            time_us = _finite_float(item["time_s"], "time_s") * 1.0e6
        elif "t_us" in item:
            time_us = _finite_float(item["t_us"], "t_us")
        else:
            continue
        result[round(time_us, 9)] = item
    return result


def _value_at(case: Mapping[str, Any], horizon_us: float, observable: str) -> float | None:
    item = _snapshot_map(case).get(round(float(horizon_us), 9))
    if item is None:
        return None
    value = item.get(observable)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def relative_difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    numerator = abs(float(value) - float(reference))
    denominator = abs(float(reference))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return abs(float(value) - float(reference))


def _trend(d1: float | None, d2: float | None) -> bool:
    if d1 is None or d2 is None or not math.isfinite(d1) or not math.isfinite(d2):
        return False
    # An observable that is identical at all three timesteps has no measured
    # temporal error; treat that zero-error limit as converged even though the
    # strict refinement inequality is not informative there.
    return d2 < d1 or (d1 == 0.0 and d2 == 0.0)


def _contaminated(case: Mapping[str, Any], horizon_us: float) -> bool:
    for key in ("boundary_contaminated_horizons_us", "boundary_contaminated_horizons"):
        values = case.get(key, ())
        if any(math.isclose(float(value), horizon_us, rel_tol=0.0, abs_tol=1.0e-6) for value in values):
            return True
    item = _snapshot_map(case).get(round(float(horizon_us), 9))
    return bool(item and item.get("boundary_contaminated", False))


def convergence_rows(
    cases: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    horizons_us: Sequence[float] = E1A_PRIMARY_HORIZONS_US,
    allow_screen_identity: bool = False,
) -> dict[str, Any]:
    """Calculate D1/D2, D1/D2 ratios, and candidate/reference errors."""
    items = _case_items(cases)
    guard = config_diff_guard(cases, allow_screen_identity=allow_screen_identity)
    by_dt: dict[float, tuple[str, Mapping[str, Any]]] = {}
    for name, case in items:
        by_dt[_case_dt_s(case)] = (name, case)
    required = ((1.0, "1.0us"), (0.5, "0.5us"), (0.25, "0.25us"))
    resolved: dict[str, tuple[str, Mapping[str, Any]] | None] = {}
    for dt_us, label in required:
        match = next(
            ((name, case) for dt_s, (name, case) in by_dt.items() if math.isclose(dt_s * 1.0e6, dt_us, rel_tol=0.0, abs_tol=1.0e-9)),
            None,
        )
        resolved[label] = match

    rows: list[dict[str, Any]] = []
    for horizon_us in horizons_us:
        contaminated = any(
            case is not None and _contaminated(case[1], float(horizon_us)) for case in resolved.values()
        )
        for observable in OBSERVABLES:
            q1 = _value_at(resolved["1.0us"][1], horizon_us, observable) if resolved["1.0us"] else None
            q05 = _value_at(resolved["0.5us"][1], horizon_us, observable) if resolved["0.5us"] else None
            q025 = _value_at(resolved["0.25us"][1], horizon_us, observable) if resolved["0.25us"] else None
            d1 = _difference(q1, q05)
            d2 = _difference(q05, q025)
            d_ref_1 = _difference(q1, q025)
            ratio = (d1 / d2) if d1 is not None and d2 is not None and d2 > 0.0 else None
            row = {
                "horizon_us": float(horizon_us),
                "observable": observable,
                "boundary_contaminated": contaminated,
                "Q_1p0": q1,
                "Q_0p5": q05,
                "Q_0p25": q025,
                "D1_1p0_vs_0p5": d1,
                "D2_0p5_vs_0p25": d2,
                "D_1p0_vs_0p25": d_ref_1,
                "D1_over_D2": ratio,
                "trend_D2_lt_D1": _trend(d1, d2),
            }
            for label, q in (("1p0", q1), ("0p5", q05)):
                difference = _difference(q, q025)
                relative = relative_difference(q, q025)
                if TOLERANCE_KIND[observable] == "absolute":
                    tolerance = E1A_CENTROID_TOLERANCE_M
                    passed = difference is not None and difference <= tolerance
                    value = difference
                else:
                    tolerance = (
                        E1A_WIDTH_RELATIVE_TOLERANCE
                        if observable in ("sigma_x_m", "sigma_y_m")
                        else E1A_EXTREME_RELATIVE_TOLERANCE
                    )
                    passed = relative is not None and relative <= tolerance
                    value = relative
                row[f"{label}_vs_ref_value"] = value
                row[f"{label}_vs_ref_tolerance"] = tolerance
                row[f"{label}_vs_ref_pass"] = bool(passed and not contaminated)
            rows.append(row)

    def _case_stability(label: str) -> bool:
        item = resolved[label]
        if item is None:
            return False
        case = item[1]
        return bool(case.get("status") == "PASS" and case.get("stability", {}).get("overall_pass", False))

    candidate_checks: dict[str, dict[str, Any]] = {}
    for label in ("1p0", "0p5"):
        case_label = "1.0us" if label == "1p0" else "0.5us"
        relevant = [row for row in rows if not row["boundary_contaminated"]]
        tolerance_rows = [row[f"{label}_vs_ref_pass"] for row in relevant]
        trend_rows = [row["trend_D2_lt_D1"] for row in relevant if row["observable"] in PRIMARY_OBSERVABLES]
        candidate_checks[label] = {
            "case_id": resolved[case_label][0] if resolved[case_label] else None,
            "stability_pass": _case_stability(case_label),
            "tolerance_pass": bool(tolerance_rows) and all(tolerance_rows),
            "trend_pass": bool(trend_rows) and all(trend_rows),
            "pass": bool(_case_stability(case_label) and tolerance_rows and all(tolerance_rows) and trend_rows and all(trend_rows)),
            "boundary_free_horizons_us": sorted({row["horizon_us"] for row in relevant}),
        }

    relevant_rows = [row for row in rows if not row["boundary_contaminated"]]
    any_failed_case = any(case.get("status") != "PASS" for _, case in items)
    if not guard.get("pass", False):
        classification, status = "D", "FAIL_CONFIG"
    elif not all(resolved.values()):
        classification, status = "D", "NOT_RUN"
    elif any_failed_case:
        classification, status = "D", "FAIL_CASE"
    elif not relevant_rows:
        classification, status = "D", "FAIL_BOUNDARY"
    elif not candidate_checks["1p0"]["pass"] and candidate_checks["0p5"]["pass"]:
        classification, status = "B", "PASS"
    elif candidate_checks["1p0"]["pass"]:
        classification, status = "A", "PASS"
    else:
        classification, status = "C", "FAIL_CONVERGENCE"

    return {
        "schema": "khz_filament.hr4e1.convergence_report.v1",
        "status": status,
        "classification": classification,
        "decision": {
            "A": "1.0 us acceptable candidate at 10 um grid",
            "B": "1.0 us fails; 0.5 us acceptable candidate at 10 um grid",
            "C": "0.5 us is not converged; 0.125 us refinement required",
            "D": "benchmark invalid because of identified contamination / implementation issue",
        }[classification],
        "config_guard": guard,
        "required_cases": {
            label: (item[0] if item is not None else None) for label, item in resolved.items()
        },
        "rows": rows,
        "candidate_checks": candidate_checks,
        "tolerances": {
            "yc_abs_m": E1A_CENTROID_TOLERANCE_M,
            "width_relative": E1A_WIDTH_RELATIVE_TOLERANCE,
            "extreme_relative": E1A_EXTREME_RELATIVE_TOLERANCE,
        },
        "formal_edge_metric": "formal_edge_boundary_ratio",
        "first_interior_ring_proxy": "first_interior_ring_ratio",
    }


calculate_convergence = convergence_rows
classify_timestep_convergence = convergence_rows


def load_case(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_dir():
        choices = [candidate / "manifest.json", candidate / "case_manifest.json"]
        choices.extend(sorted(candidate.glob("*.json")))
        candidate = next((item for item in choices if item.is_file()), candidate)
    return json.loads(candidate.read_text(encoding="utf-8"))


def write_report(report: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "convergence_report.json"
    csv_path = destination / "convergence_observables.csv"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_safe(dict(report)), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    fields = [
        "horizon_us", "observable", "boundary_contaminated", "Q_1p0", "Q_0p5", "Q_0p25",
        "D1_1p0_vs_0p5", "D2_0p5_vs_0p25", "D_1p0_vs_0p25", "D1_over_D2",
        "trend_D2_lt_D1", "1p0_vs_ref_value", "1p0_vs_ref_tolerance", "1p0_vs_ref_pass",
        "0p5_vs_ref_value", "0p5_vs_ref_tolerance", "0p5_vs_ref_pass",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report.get("rows", ()):
            writer.writerow({key: json_safe(row.get(key)) for key in fields})
    return json_path, csv_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--case", dest="case_paths", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--horizon-us", action="append", type=float, default=[],
        help="override formal horizons; default is 100/1000 us for E1-A and 100 us for E1-B",
    )
    args = parser.parse_args(argv)
    paths = [*args.paths, *args.case_paths]
    if not paths:
        parser.error("at least one case manifest is required")
    cases = []
    for path in paths:
        cases.append(load_case(path))
    benchmarks = {str(case.get("benchmark", case.get("configuration", {}).get("benchmark", ""))) for case in cases}
    if args.horizon_us:
        horizons_us = tuple(args.horizon_us)
    elif benchmarks == {"E1-B"}:
        horizons_us = E1B_PRIMARY_HORIZONS_US
    else:
        horizons_us = E1A_PRIMARY_HORIZONS_US
    report = convergence_rows(cases, horizons_us=horizons_us)
    json_path, csv_path = write_report(report, args.out_dir)
    print(json.dumps({"status": report["status"], "classification": report["classification"], "json": str(json_path), "csv": str(csv_path)}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

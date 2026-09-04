"""Deterministic report-only adjudication of E2-C mapping versus evolution."""
from __future__ import annotations

import math
import sys
from typing import Any, Mapping, Sequence

from .hr4e_spatial import (
    E2_CENTROID_TOLERANCE_M,
    E2_EXTREME_RELATIVE_TOLERANCE,
    E2_M0_RELATIVE_TOLERANCE,
    E2_WIDTH_RELATIVE_TOLERANCE,
)


SPACINGS = (20.0e-6, 10.0e-6, 5.0e-6)
OBSERVABLES = (
    "xc_m", "yc_m", "sigma_x_m", "sigma_y_m", "min_delta_n",
    "max_abs_vx_m_s", "max_abs_vy_m_s", "max_abs_v_m_s",
    "M0_negative_index_m2",
)
WIDTHS = {"sigma_x_m", "sigma_y_m"}
EXTREMES = {"min_delta_n", "max_abs_vx_m_s", "max_abs_vy_m_s", "max_abs_v_m_s"}
NUMERIC_NEGLIGIBLE_TOLERANCE_FRACTION = 0.01


def tolerance(observable: str) -> tuple[str, float]:
    if observable in {"xc_m", "yc_m"}:
        return "absolute", E2_CENTROID_TOLERANCE_M
    if observable in WIDTHS:
        return "relative", E2_WIDTH_RELATIVE_TOLERANCE
    if observable == "M0_negative_index_m2":
        return "relative", E2_M0_RELATIVE_TOLERANCE
    if observable in EXTREMES:
        return "relative", E2_EXTREME_RELATIVE_TOLERANCE
    raise ValueError(f"unsupported E2-C observable: {observable}")


def relative_difference(left: float, right: float) -> float:
    if left == right == 0.0:
        return 0.0
    if right == 0.0:
        return float("inf")
    return abs(left - right) / abs(right)


def _finite(values: Sequence[Any], name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in values)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite values")
    return result  # type: ignore[return-value]


def _machine_floor(values: Sequence[float]) -> float:
    return 64.0 * sys.float_info.epsilon * max(1.0e-30, *(abs(item) for item in values))


def _difference_value(values: Sequence[float], kind: str) -> tuple[float, float]:
    q20, q10, q5 = values
    if kind == "absolute":
        return abs(q20 - q10), abs(q10 - q5)
    return relative_difference(q20, q10), relative_difference(q10, q5)


def _trend(d20_10: float, d10_5: float, *, near_zero: bool) -> tuple[bool | None, float | None]:
    if near_zero:
        return None, None
    if d20_10 == 0.0 or d10_5 == 0.0:
        return d10_5 < d20_10, None
    return d10_5 < d20_10, math.log2(d20_10 / d10_5)


def _initial_near_zero(values: Sequence[float], kind: str, limit: float) -> bool:
    return (max(abs(item) for item in values) <= limit) if kind == "absolute" else all(item == 0.0 for item in values)


def _evolution_near_zero(values: Sequence[float], d20_10: float, d10_5: float, kind: str, limit: float) -> bool:
    if kind == "absolute":
        return max(abs(d20_10), abs(d10_5)) <= limit
    return max(abs(d20_10), abs(d10_5)) <= _machine_floor(values)


def _initial_status(trend: bool | None, fine_fraction: float) -> str:
    if trend is None:
        return "INIT_NEAR_ZERO_NA"
    if trend:
        return "INIT_MONOTONIC"
    return "INIT_NONMONOTONIC_WITHIN_TOLERANCE" if fine_fraction <= 1.0 else "INIT_NONMONOTONIC_MATERIAL"


def _adjudication_category(
    *, final_trend: bool | None, initial_status: str, evolution_trend: bool | None,
    initial_fraction: float, evolution_fraction: float, final_fraction: float,
) -> tuple[str, str]:
    if final_trend is not False:
        return "NO_FINAL_WARNING", "No resolved final-state non-monotonicity requires adjudication."
    if max(initial_fraction, evolution_fraction, final_fraction) <= NUMERIC_NEGLIGIBLE_TOLERANCE_FRACTION:
        return "WARNING_NUMERICALLY_NEGLIGIBLE", "All compared discrepancies use at most 1% of the frozen tolerance."
    initial_nonmonotonic = initial_status.startswith("INIT_NONMONOTONIC")
    if initial_nonmonotonic and evolution_trend in {True, None}:
        return "WARNING_MAPPING_DOMINATED", "The non-monotonic pattern is already present at t=0 while evolution increments are monotonic or near-zero."
    if initial_nonmonotonic and evolution_trend is False:
        return "WARNING_MIXED_OR_AMBIGUOUS", "Both the validation-state mapping and the hydro evolution increments are non-monotonic."
    if initial_status in {"INIT_MONOTONIC", "INIT_NEAR_ZERO_NA"} and evolution_trend is False:
        return "WARNING_HYDRO_EVOLUTION_DOMINATED", "The t=0 state is monotonic or near-zero, but the hydro evolution increments are non-monotonic."
    return "WARNING_MIXED_OR_AMBIGUOUS", "The available t=0 and evolution evidence does not separate the two contributions robustly."


def adjudicate_observable(observable: str, initial: Sequence[Any], final: Sequence[Any]) -> dict[str, Any]:
    """Compare one observable across 20/10/5 um at t=0 and t=100 us."""
    initial_values = _finite(initial, f"{observable} initial")
    final_values = _finite(final, f"{observable} final")
    kind, limit = tolerance(observable)
    d20_init_raw, d10_init_raw = abs(initial_values[0] - initial_values[1]), abs(initial_values[1] - initial_values[2])
    d20_final_raw, d10_final_raw = abs(final_values[0] - final_values[1]), abs(final_values[1] - final_values[2])
    d20_init, d10_init = _difference_value(initial_values, kind)
    d20_final, d10_final = _difference_value(final_values, kind)
    increments = tuple(f - i for i, f in zip(initial_values, final_values, strict=True))
    d20_evol_raw, d10_evol_raw = abs(increments[0] - increments[1]), abs(increments[1] - increments[2])
    # Delta Q may be close to zero even when Q itself is not.  Relative
    # normalization by Delta Q5 would therefore manufacture an ill-conditioned
    # evolution discrepancy.  Relative tolerances are anchored to the stored
    # physical Q scale, while the required evolution D and p remain absolute.
    evolution_scale = max(abs(initial_values[2]), abs(final_values[2]), _machine_floor(initial_values + final_values))
    d20_evol = d20_evol_raw if kind == "absolute" else d20_evol_raw / evolution_scale
    d10_evol = d10_evol_raw if kind == "absolute" else d10_evol_raw / evolution_scale
    initial_near_zero = _initial_near_zero(initial_values, kind, limit)
    evolution_near_zero = _evolution_near_zero(increments, d20_evol, d10_evol, kind, limit)
    final_near_zero = _initial_near_zero(final_values, kind, limit)
    initial_trend, p_init = _trend(d20_init_raw, d10_init_raw, near_zero=initial_near_zero)
    evolution_trend, p_evol = _trend(d20_evol_raw, d10_evol_raw, near_zero=evolution_near_zero)
    final_trend, p_final = _trend(d20_final_raw, d10_final_raw, near_zero=final_near_zero)
    initial_fraction = d10_init / limit
    evolution_fraction = d10_evol / limit
    final_fraction = d10_final / limit
    initial_state = _initial_status(initial_trend, initial_fraction)
    category, note = _adjudication_category(
        final_trend=final_trend, initial_status=initial_state, evolution_trend=evolution_trend,
        initial_fraction=initial_fraction, evolution_fraction=evolution_fraction, final_fraction=final_fraction,
    )
    return {
        "observable": observable, "tolerance_kind": kind, "frozen_10_vs_5_tolerance": limit,
        "Q_20um_0": initial_values[0], "Q_10um_0": initial_values[1], "Q_5um_0": initial_values[2],
        "Q_20um_100us": final_values[0], "Q_10um_100us": final_values[1], "Q_5um_100us": final_values[2],
        "D20_10_init": d20_init_raw, "D10_5_init": d10_init_raw,
        "D20_10_init_metric": d20_init, "D10_5_init_metric": d10_init,
        "p_init": p_init, "initial_near_zero": initial_near_zero, "initial_status": initial_state,
        "Delta_Q20": increments[0], "Delta_Q10": increments[1], "Delta_Q5": increments[2],
        "D20_10_evol": d20_evol_raw, "D10_5_evol": d10_evol_raw,
        "D20_10_evol_metric": d20_evol, "D10_5_evol_metric": d10_evol,
        "p_evol": p_evol, "evolution_near_zero": evolution_near_zero,
        "evolution_trend_status": "N/A_NEAR_ZERO" if evolution_trend is None else ("PASS" if evolution_trend else "WARNING"),
        "D20_10_final": d20_final_raw, "D10_5_final": d10_final_raw,
        "D20_10_final_metric": d20_final, "D10_5_final_metric": d10_final,
        "p_final": p_final, "final_near_zero": final_near_zero,
        "final_trend_status": "N/A_NEAR_ZERO" if final_trend is None else ("PASS" if final_trend else "WARNING"),
        "fraction_of_tolerance_init": initial_fraction,
        "fraction_of_tolerance_evol": evolution_fraction,
        "fraction_of_tolerance_final": final_fraction,
        "hard_10_vs_5_pass": final_fraction <= 1.0,
        "adjudication_category": category, "diagnostic_note": note,
    }


def _snapshot(case: Mapping[str, Any], time_us: float) -> Mapping[str, Any]:
    matches = [item for item in case.get("snapshots", []) if math.isclose(float(item.get("time_us", float("nan"))), time_us, rel_tol=0.0, abs_tol=1.0e-8)]
    if len(matches) != 1:
        raise ValueError(f"{case.get('case_id')} lacks one {time_us:g} us snapshot")
    return matches[0]


def _screen_id(case: Mapping[str, Any]) -> str:
    return str(case["source_provenance"]["screen_identity"]["screen_id"])


def adjudicate_screen(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected: list[Mapping[str, Any]] = []
    for spacing in SPACINGS:
        found = [case for case in cases if math.isclose(float(case["configuration"]["grid"]["dx_m"]), spacing, rel_tol=0.0, abs_tol=1.0e-15)]
        if len(found) != 1:
            raise ValueError("each screen requires exactly one 20/10/5 um case")
        selected.append(found[0])
    reference = selected[0]
    provenance, representation = reference["source_provenance"], reference["validation_representation"]
    config_pass = all(
        case.get("status") == "PASS" and case.get("stability", {}).get("overall_pass")
        and case.get("source_provenance") == provenance
        and case.get("validation_representation") == representation
        and case["configuration"].get("dt_hydro_s") == reference["configuration"].get("dt_hydro_s")
        and case["configuration"].get("operator") == reference["configuration"].get("operator")
        for case in selected
    )
    snapshots_0 = [_snapshot(case, 0.0) for case in selected]
    snapshots_100 = [_snapshot(case, 100.0) for case in selected]
    boundary_clean = all(not snapshot.get("boundary_contaminated", True) for snapshot in snapshots_0 + snapshots_100)
    rows = [adjudicate_observable(observable, [snapshot[observable] for snapshot in snapshots_0], [snapshot[observable] for snapshot in snapshots_100]) for observable in OBSERVABLES]
    categories = [row["adjudication_category"] for row in rows]
    if not config_pass or not boundary_clean or not all(row["hard_10_vs_5_pass"] for row in rows):
        status = "INVALID"
    elif "WARNING_HYDRO_EVOLUTION_DOMINATED" in categories or "WARNING_MIXED_OR_AMBIGUOUS" in categories:
        status = "WARNING"
    else:
        status = "PASS"
    return {
        "screen_identity": provenance["screen_identity"], "source_provenance": provenance,
        "validation_representation": representation, "case_ids": [case["case_id"] for case in selected],
        "case_git_shas": sorted({str(case.get("git_sha")) for case in selected}),
        "configuration_guard_pass": config_pass, "boundary_contamination": not boundary_clean,
        "status": status, "rows": rows,
    }


def adjudicate(cases: Sequence[Mapping[str, Any]], *, e2a: Mapping[str, Any], e2b: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(_screen_id(case), []).append(case)
    screens = [adjudicate_screen(grouped[name]) for name in sorted(grouped)]
    rows = [row for screen in screens for row in screen["rows"]]
    hard_pass = bool(rows) and all(row["hard_10_vs_5_pass"] for row in rows)
    clean = all(screen["configuration_guard_pass"] and not screen["boundary_contamination"] for screen in screens)
    adverse = {"WARNING_HYDRO_EVOLUTION_DOMINATED", "WARNING_MIXED_OR_AMBIGUOUS"}
    has_adverse = any(row["adjudication_category"] in adverse for row in rows)
    near_limit = any(row["fraction_of_tolerance_evol"] >= 0.75 or row["fraction_of_tolerance_final"] >= 0.75 for row in rows)
    e2a_pass = e2a.get("status") == "PASS" and e2a.get("validity") == "VALID"
    e2b_pass = e2b.get("status") == "PASS" and e2b.get("classification") == "B1"
    if e2a_pass and e2b_pass and hard_pass and clean and not has_adverse:
        decision, status = "A1_SCOPED_CLASS_A_GRANTED", "PASS"
        statement = "HR-4E-2 is accepted as scoped Class A for the HR-4 hydrodynamic solver operating on the current HR-3B POST-state interface. dx = dy = 10 um is accepted as the production spatial candidate for this hydro solver. The previously qualified dt_hydro = 1.0 us remains the production timestep candidate for that 10 um hydro grid."
    elif near_limit:
        decision, status = "A3_SUPPLEMENTARY_SPATIAL_LEVEL_REQUIRED", "WARNING"
        statement = None
    else:
        decision, status = "A2_WARNING_A_NOT_GRANTED", "WARNING"
        statement = None
    return {
        "schema": "khz_filament.hr4e2c.mapping_vs_evolution_adjudication.v1",
        "status": status, "decision": decision,
        "e2a_status": e2a.get("status"), "e2b_status": e2b.get("status"),
        "hard_10_vs_5_tolerances_pass": hard_pass, "configuration_and_boundary_guards_pass": clean,
        "has_material_hydro_or_mixed_warning": has_adverse, "near_limit": near_limit,
        "production_candidate_statement": statement,
        "scope_is_hydro_only_validation": True,
        "full_chain_transverse_convergence_claimed": False,
        "production_multigrid_mapping_modified": False,
        "validation_only_statement": "The 5 um initial field adds no physical information beyond the frozen native 10 um POST morphology; the bilinear representation is validation-only and not a production HR-3B-to-HR-4 mapper.",
        "execution_sha_note": "The manifest Git SHAs are recorded per screen. Any differing SHA must be audited independently; this adjudication does not treat a submission-only wrapper difference as a frozen HR-4 operator change.",
        "screens": screens,
    }

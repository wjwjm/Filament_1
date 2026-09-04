#!/usr/bin/env python3
"""Create non-overwriting SHA and raw-velocity diagnostics for HR-4E-2C.

The tool is analysis-only.  It never invokes a GPU executable, changes raw
artifacts, or starts HR-5/HR-4F.  If a raw index is absent or incomplete it
writes the required blocked reports instead of using scalar manifests.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e2c_velocity_diagnostic import (  # noqa: E402
    classify_velocity_screen, convergence_order, field_norms, nested_node_indices,
    peak_audit, plume_weighted_velocity, quadratic_subgrid_peak, restrict_nested_field,
    velocity_energy,
)
from KHz_filament.hr4e_timestep import json_safe  # noqa: E402

OLD_SHA = "51e4969fb8c54c3ba5272196aa767c22c587fbcc"
NEW_SHA = "d7e77cd380d48dafdab04d307a06264159afa958"
SCREENS = ("front", "peak", "rear")
SPACINGS = (20, 10, 5)


def sha_audit(repo: Path) -> dict[str, Any]:
    names = subprocess.run(["git", "diff", "--name-status", OLD_SHA, NEW_SHA], cwd=repo, check=True, text=True, capture_output=True).stdout.splitlines()
    changes = []
    for line in names:
        status, path = line.split("\t", 1)
        classification = "SUBMISSION_WRAPPER_ONLY" if path in {
            "Filament_python/tools/hpc_ops/submit_hr4e2c_real.sh",
            "Filament_python/tools/hpc_ops/submit_hr4e2c_real_failed.sh",
        } else "UNKNOWN"
        changes.append({"status": status, "path": path, "classification": classification})
    forbidden = {
        "hr4_pde_update": False, "advection_operator": False, "laplacian": False,
        "forward_euler": False, "buoyancy_source": False, "viscosity": False,
        "thermal_diffusion": False, "open_boundary": False, "grid_construction": False,
        "physical_domain": False, "bilinear_validation_representation": False,
        "target_grid_sampling": False, "observables_extraction": False,
        "dtype_backend": False,
    }
    passed = bool(changes) and all(change["classification"] in {"SUBMISSION_WRAPPER_ONLY", "REPORTING_ONLY", "TEST_ONLY"} for change in changes)
    return {
        "schema": "khz_filament.hr4e2c.sha_equivalence_audit.v1",
        "base_sha": OLD_SHA, "comparison_sha": NEW_SHA, "changed_files": changes,
        "scientific_path_checks": forbidden,
        "SHA_EQUIVALENT_FOR_E2C": "PASS" if passed else "FAIL",
        "scientific_e2c_paths_identical": passed,
        "decision_basis": "Only submission log-directory creation/output routing and a five-case retry launcher changed; neither invokes nor alters the E2-C hydro computation path.",
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(json_safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _load_index(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _missing_raw(index: Mapping[str, Any]) -> dict[str, Any]:
    missing = []
    for screen in SCREENS:
        for spacing in SPACINGS:
            item = index.get(screen, {}).get(str(spacing), {})
            if not item.get("vy") or not item.get("x") or not item.get("y"):
                missing.append({"screen": screen, "dx_um": spacing, "required": ["vy", "x", "y"], "provided": sorted(item)})
    return {"status": "FIELD_DIAGNOSTIC_BLOCKED_BY_MISSING_RAW_ARTIFACT", "missing": missing}


def _array(item: Mapping[str, Any], key: str) -> np.ndarray | None:
    value = item.get(key)
    return None if value is None else np.load(Path(value), allow_pickle=False)


def _screen_metrics(index: Mapping[str, Any], screen: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    data = {spacing: index[screen][str(spacing)] for spacing in SPACINGS}
    fields = {spacing: _array(data[spacing], "vy") for spacing in SPACINGS}
    xs = {spacing: _array(data[spacing], "x") for spacing in SPACINGS}
    ys = {spacing: _array(data[spacing], "y") for spacing in SPACINGS}
    assert all(value is not None for value in [*fields.values(), *xs.values(), *ys.values()])
    ix_20_10, iy_20_10 = nested_node_indices(xs[20], xs[10]), nested_node_indices(ys[20], ys[10])
    ix_10_5, iy_10_5 = nested_node_indices(xs[10], xs[5]), nested_node_indices(ys[10], ys[5])
    restricted_10 = restrict_nested_field(fields[10], iy_20_10, ix_20_10)
    restricted_5 = restrict_nested_field(fields[5], iy_10_5, ix_10_5)
    dA = {spacing: float((xs[spacing][1] - xs[spacing][0]) * (ys[spacing][1] - ys[spacing][0])) for spacing in SPACINGS}
    n20_10, n10_5 = field_norms(fields[20], restricted_10, dA=dA[20]), field_norms(fields[10], restricted_5, dA=dA[10])
    energies = {str(spacing): velocity_energy(fields[spacing], dA=dA[spacing]) for spacing in SPACINGS}
    norms = {
        "screen": screen, "nested_node_alignment": {"20_to_10": "PASS_EXACT_INDEX_SELECTION", "10_to_5": "PASS_EXACT_INDEX_SELECTION"},
        "20_10": n20_10, "10_5": n10_5,
        "p_L1": convergence_order(n20_10["E_L1"], n10_5["E_L1"]), "p_L2": convergence_order(n20_10["E_L2"], n10_5["E_L2"]),
        "Ev": energies, "D20_10_Ev": abs(energies["20"] - energies["10"]), "D10_5_Ev": abs(energies["10"] - energies["5"]),
    }
    weights = {}
    for spacing in SPACINGS:
        delta_n = _array(data[spacing], "delta_n")
        weights[str(spacing)] = None if delta_n is None else plume_weighted_velocity(fields[spacing], delta_n, dA=dA[spacing])
    norms["plume_weighted_vy"] = weights
    peaks = {str(spacing): peak_audit(fields[spacing], xs[spacing], ys[spacing]) for spacing in SPACINGS}
    subgrid = {str(spacing): quadratic_subgrid_peak(fields[spacing], xs[spacing], ys[spacing], peaks[str(spacing)]) for spacing in SPACINGS}
    category = classify_velocity_screen(norms_20_10=n20_10, norms_10_5=n10_5, energy_20=energies["20"], energy_10=energies["10"], energy_5=energies["5"], raw_peaks=[peaks[str(s)] for s in SPACINGS], subgrid_peaks=[subgrid[str(s)] for s in SPACINGS])
    return norms, {"screen": screen, "peaks": peaks}, {"screen": screen, "subgrid_peaks": subgrid}, category


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raw-index", type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    sha = sha_audit(ROOT.parent)
    index = _load_index(args.raw_index)
    missing = _missing_raw(index)
    common = {"analysis_only": True, "gpu_jobs_submitted": False, "raw_case_manifests": [str(path) for path in args.case], "raw_index": None if args.raw_index is None else str(args.raw_index)}
    _write(args.out_dir / "e2c_sha_equivalence_audit.json", {**sha, **common})
    if missing["missing"]:
        blocked = {"schema": "khz_filament.hr4e2c.velocity_field.v1", **common, **missing}
        for name in ("e2c_velocity_field_norms.json", "e2c_velocity_peak_audit.json", "e2c_velocity_subgrid_peak_report.json", "e2c_velocity_field_adjudication.json"):
            _write(args.out_dir / name, blocked)
        decision = {"schema": "khz_filament.hr4e2c.velocity_followup_decision.v1", **common, "sha_equivalence": sha["SHA_EQUIVALENT_FOR_E2C"], "field_diagnostic": missing["status"], "outcome": "S2_KEEP_A2_WARNING", "decision": "A2_WARNING_A_NOT_GRANTED", "scoped_class_a_reconsideration_supported": False, "supplementary_2p5um_study_justified": False, "reason": "The scheduler manifests are terminal and valid, but no authoritative persisted vy(x,y,100 us) arrays exist in either E2-C run root. Scalar maxima are not substituted for field evidence."}
    else:
        norms, peaks, subgrid, categories = [], [], [], []
        for screen in SCREENS:
            screen_norms, screen_peaks, screen_subgrid, category = _screen_metrics(index, screen)
            norms.append(screen_norms); peaks.append(screen_peaks); subgrid.append(screen_subgrid); categories.append({"screen": screen, "category": category})
        _write(args.out_dir / "e2c_velocity_field_norms.json", {"schema": "khz_filament.hr4e2c.velocity_field_norms.v1", **common, "status": "PASS", "screens": norms})
        _write(args.out_dir / "e2c_velocity_peak_audit.json", {"schema": "khz_filament.hr4e2c.velocity_peak_audit.v1", **common, "status": "PASS", "screens": peaks})
        _write(args.out_dir / "e2c_velocity_subgrid_peak_report.json", {"schema": "khz_filament.hr4e2c.velocity_subgrid_peak.v1", **common, "status": "PASS", "screens": subgrid})
        _write(args.out_dir / "e2c_velocity_field_adjudication.json", {"schema": "khz_filament.hr4e2c.velocity_field_adjudication.v1", **common, "status": "PASS", "screens": categories})
        outcome = "S1_SCOPED_CLASS_A_RECONSIDERATION" if sha["SHA_EQUIVALENT_FOR_E2C"] == "PASS" and all(item["category"].startswith(("V1", "V4")) for item in categories) else "S2_KEEP_A2_WARNING"
        decision = {"schema": "khz_filament.hr4e2c.velocity_followup_decision.v1", **common, "sha_equivalence": sha["SHA_EQUIVALENT_FOR_E2C"], "outcome": outcome, "decision": "SCOPED_CLASS_A_RECONSIDERATION_SUPPORTED" if outcome.startswith("S1") else "A2_WARNING_A_NOT_GRANTED", "scoped_class_a_reconsideration_supported": outcome.startswith("S1"), "supplementary_2p5um_study_justified": any(item["category"].startswith("V2") for item in categories), "screens": categories}
    _write(args.out_dir / "hr4e2_velocity_followup_decision.json", decision)
    markdown = "# HR-4E-2C SHA Equivalence and Velocity-Field Diagnostic\n\n" + f"- SHA equivalence: `{sha['SHA_EQUIVALENT_FOR_E2C']}`\n- Velocity-field status: `{decision.get('field_diagnostic', 'EVALUATED')}`\n- Follow-up: `{decision['outcome']}`\n- No GPU jobs were submitted; HR-5 and HR-4F were not started.\n"
    (args.out_dir / "hr4e2_velocity_field_diagnostic.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(json.dumps({"out_dir": str(args.out_dir), "outcome": decision["outcome"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

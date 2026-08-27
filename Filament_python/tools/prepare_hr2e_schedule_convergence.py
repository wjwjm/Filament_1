"""Prepare the fixed HR-2E coarse/candidate/fine convergence inputs.

This tool is CPU-only.  It creates no remote state and never submits jobs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from KHz_filament.longitudinal import build_longitudinal_schedule  # noqa: E402
from hr2e_schedule_convergence import historical_proposal, schedule_summary  # noqa: E402


BASELINES = {
    "40fs": ROOT / "configs" / "ionization_model_propagation" / "40fs_talebpour_full_model.json",
    "120fs": ROOT / "configs" / "ionization_model_propagation" / "120fs_talebpour_full_model.json",
}
HISTORICAL = {
    "40fs": ROOT / "results" / "ionization_model_propagation" / "talebpour_40fs_20260717T151324Z" / "baseline_axial_diagnostics.csv",
    "120fs": ROOT / "results" / "ionization_model_propagation" / "talebpour_120fs_20260717T114321Z" / "baseline_axial_diagnostics.csv",
}
FULL_RAMAN_SOURCE = (
    ROOT / "configs" / "isaacs_raman_closure"
    / "120fs_talebpour_isaacs_full_operator_on_energy_audit.json"
)
DEFAULT_CONFIG_DIR = ROOT / "configs" / "hr2e_schedule_convergence"
DEFAULT_RESULT_DIR = ROOT / "results" / "hr2e_schedule_convergence" / "stage1_preflight"

SCHEDULES = {
    "coarse": {"dz": 2.0e-4, "dz_focus": 1.0e-4},
    "candidate": {"dz": 1.0e-4, "dz_focus": 5.0e-5},
    "fine": {"dz": 5.0e-5, "dz_focus": 2.5e-5},
}
FOCUS_CENTER_M = 0.90
FOCUS_HALFWIDTH_M = 0.15

FULL_RAMAN_PROPAGATION_KEYS = {
    "diag_operator_energy",
    "use_raman_absorption",
    "use_raman_full_operator",
    "use_raman_phase",
}
SCHEDULE_PATHS = {
    "propagation.dz",
    "propagation.dz_focus",
    "propagation.focus_center_m",
    "propagation.focus_halfwidth_m",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        result.update(_flatten(item, path))
    return result


def _diff(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    left = _flatten(before)
    right = _flatten(after)
    return [
        {"path": path, "before": left.get(path), "after": right.get(path)}
        for path in sorted(set(left) | set(right))
        if left.get(path) != right.get(path)
    ]


def _assert_source_contract(
    baseline_40: dict[str, Any], baseline_120: dict[str, Any], full_120: dict[str, Any]
) -> None:
    baseline_diff = _diff(baseline_40, baseline_120)
    if [row["path"] for row in baseline_diff] != ["beam.tau_fwhm"]:
        raise ValueError("40/120 fs production baselines differ by more than beam.tau_fwhm")
    if full_120["raman"].get("operator_mode") != "full_isaacs_eq27":
        raise ValueError("full Raman source is not full_isaacs_eq27")
    if full_120["raman"].get("operator_integrator") != "heun":
        raise ValueError("full Raman source does not use Heun")
    if full_120["raman"].get("nonlinear_split_order") != "strang":
        raise ValueError("full Raman source does not use Strang splitting")
    if full_120["raman"].get("iir_sampling") != "exact_piecewise_linear":
        raise ValueError("full Raman source does not use exact_piecewise_linear sampling")
    allowed_source = {f"raman.{key}" for key in set(baseline_120["raman"]) | set(full_120["raman"])}
    allowed_source |= {f"propagation.{key}" for key in FULL_RAMAN_PROPAGATION_KEYS}
    unexpected = [row["path"] for row in _diff(baseline_120, full_120) if row["path"] not in allowed_source]
    if unexpected:
        raise ValueError(f"full Raman source changes unrelated fields: {unexpected}")


def build_config(
    baseline: dict[str, Any], full_source: dict[str, Any], *, schedule_name: str
) -> dict[str, Any]:
    values = SCHEDULES[schedule_name]
    config = copy.deepcopy(baseline)
    config["raman"] = copy.deepcopy(full_source["raman"])
    for key in FULL_RAMAN_PROPAGATION_KEYS:
        if key in full_source["propagation"]:
            config["propagation"][key] = copy.deepcopy(full_source["propagation"][key])
        else:
            config["propagation"].pop(key, None)
    prop = config["propagation"]
    prop["z_max"] = 1.3
    prop["dz"] = values["dz"]
    prop["focus_window_step"] = True
    prop["focus_center_m"] = FOCUS_CENTER_M
    prop["focus_halfwidth_m"] = FOCUS_HALFWIDTH_M
    prop["dz_focus"] = values["dz_focus"]
    config["run"]["Npulses"] = 1
    return config


def prepare(config_dir: Path, result_dir: Path) -> dict[str, Any]:
    baseline = {name: _load(path) for name, path in BASELINES.items()}
    full_source = _load(FULL_RAMAN_SOURCE)
    _assert_source_contract(baseline["40fs"], baseline["120fs"], full_source)
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    historical = {
        pulse: historical_proposal(path)
        for pulse, path in HISTORICAL.items()
    }
    for record in historical.values():
        record["source_csv"] = Path(record["source_csv"]).relative_to(ROOT).as_posix()
    worst_case = max(
        historical,
        key=lambda pulse: historical[pulse]["max_normalized_gradient_per_m"],
    )
    cases: list[dict[str, Any]] = []
    allowed_prefixes = {
        "raman",
        *(f"propagation.{key}" for key in FULL_RAMAN_PROPAGATION_KEYS),
        *SCHEDULE_PATHS,
    }
    for pulse in ("40fs", "120fs"):
        for schedule_name, spacing in SCHEDULES.items():
            config = build_config(baseline[pulse], full_source, schedule_name=schedule_name)
            config_name = f"{pulse}_full_isaacs_{schedule_name}.json"
            config_path = config_dir / config_name
            _write_text_lf(config_path, json.dumps(config, indent=2) + "\n")
            config_diff = _diff(baseline[pulse], config)
            unexpected = [
                row["path"] for row in config_diff
                if not any(row["path"] == prefix or row["path"].startswith(prefix + ".") for prefix in allowed_prefixes)
            ]
            if unexpected:
                raise ValueError(f"{config_name} changes unrelated fields: {unexpected}")
            prop = config["propagation"]
            schedule = build_longitudinal_schedule(
                prop["dz"], prop["z_max"],
                focus_window_step=prop["focus_window_step"],
                focus_center_m=prop["focus_center_m"],
                focus_halfwidth_m=prop["focus_halfwidth_m"],
                dz_focus=prop["dz_focus"],
            )
            case_id = f"hr2e_{pulse}_{schedule_name}"
            cases.append({
                "case_id": case_id,
                "pulse_width": pulse,
                "schedule": schedule_name,
                "config_path": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": _sha256(config_path),
                "dtype": "fp32",
                "raman_mode": "full_isaacs_eq27",
                "strict_diff_from_production_baseline": config_diff,
                "schedule_metadata": schedule_summary(
                    schedule.z_edges, schedule.dz_intervals,
                    base_dz=spacing["dz"], focus_dz=spacing["dz_focus"],
                ),
            })

    stage2 = [case["case_id"] for case in cases if case["pulse_width"] == worst_case]
    other = "40fs" if worst_case == "120fs" else "120fs"
    stage3 = [
        case["case_id"] for case in cases
        if case["pulse_width"] == other and case["schedule"] in {"candidate", "fine"}
    ]
    manifest = {
        "schema": "khz_filament.hr2e.stage1_preflight.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generation_git_sha": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "proposal_evidence_only": True,
        "historical_inputs": historical,
        "candidate_window": {
            "focus_center_m": FOCUS_CENTER_M,
            "focus_halfwidth_m": FOCUS_HALFWIDTH_M,
            "range_m": [FOCUS_CENTER_M - FOCUS_HALFWIDTH_M, FOCUS_CENTER_M + FOCUS_HALFWIDTH_M],
        },
        "full_raman_source_config": FULL_RAMAN_SOURCE.relative_to(ROOT).as_posix(),
        "full_raman_source_sha256": _sha256(FULL_RAMAN_SOURCE),
        "full_raman_source_contract": {
            "operator_mode": "full_isaacs_eq27",
            "operator_integrator": "heun",
            "nonlinear_split_order": "strang",
            "iir_sampling": "exact_piecewise_linear",
            "diagnostic_only_not_production_default_change": True,
        },
        "precision_strategy": "fp32 for every convergence run",
        "cases": cases,
        "worst_case_pulse_width": worst_case,
        "stage2_parallel_jobs": stage2,
        "stage2_job_count": len(stage2),
        "stage3_conditional_jobs": stage3,
        "stage3_job_count": len(stage3),
        "normal_expected_new_jobs": len(stage2) + len(stage3),
        "repeated_identical_jobs": 0,
        "production_config_changed": False,
        "full_pytest_planned": False,
    }
    manifest_path = result_dir / "hr2e_stage1_preflight_manifest.json"
    _write_text_lf(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args(argv)
    manifest = prepare(args.config_dir, args.result_dir)
    print(json.dumps({
        "schema": manifest["schema"],
        "worst_case_pulse_width": manifest["worst_case_pulse_width"],
        "stage2_job_count": manifest["stage2_job_count"],
        "stage3_job_count": manifest["stage3_job_count"],
        "case_count": len(manifest["cases"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

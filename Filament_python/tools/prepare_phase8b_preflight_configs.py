#!/usr/bin/env python3
"""Create Phase 8B-P production configs from the locked Phase 6 baseline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "ionization_model_propagation" / "120fs_talebpour_full_model.json"
CONFIG_DIR = ROOT / "configs" / "isaacs_raman_closure"
RESULT_DIR = ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight"
MISSING = "__missing__"

AUTHORIZED_BASELINE_DIFFS = {
    "raman.model",
    "raman.operator_mode",
    "raman.operator_convention",
    "raman.iir_sampling",
    "raman.operator_integrator",
    "raman.f_R",
    "raman.T_R",
    "raman.T2",
    "raman.absorption_model",
    "raman.absorption",
    "raman.abs_mask_frac",
    "raman.max_alpha_dz",
    "propagation.use_raman_phase",
    "propagation.use_raman_full_operator",
    "propagation.use_raman_absorption",
}
AUTHORIZED_ON_OFF_DIFFS = {"propagation.use_raman_full_operator"}


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            result.update(_flatten(child, path))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def diff_records(left, right):
    a, b = _flatten(left), _flatten(right)
    records = []
    for path in sorted(set(a) | set(b)):
        old, new = a.get(path, MISSING), b.get(path, MISSING)
        if old != new:
            records.append({"path": path, "before": old, "after": new})
    return records


def make_full_operator_config(baseline, enabled):
    config = deepcopy(baseline)
    propagation = config["propagation"]
    propagation["use_raman_phase"] = False
    propagation["use_raman_full_operator"] = bool(enabled)
    propagation["use_raman_absorption"] = False
    raman = config["raman"]
    for field in (
        "f_R", "T_R", "T2", "absorption_model", "absorption",
        "abs_mask_frac", "max_alpha_dz",
    ):
        raman.pop(field, None)
    raman.update({
        "model": "isaacs_rot_sinexp",
        "operator_mode": "full_isaacs_eq27",
        "operator_convention": "isaacs_eq27",
        "iir_sampling": "exact_piecewise_linear",
        "operator_integrator": "heun",
    })
    return config


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    on = make_full_operator_config(baseline, True)
    off = make_full_operator_config(baseline, False)
    on_path = args.config_dir / "120fs_talebpour_isaacs_full_operator_on.json"
    off_path = args.config_dir / "120fs_talebpour_isaacs_full_operator_feedback_off.json"
    _write_json(on_path, on)
    _write_json(off_path, off)

    baseline_diff = diff_records(baseline, on)
    on_off_diff = diff_records(on, off)
    _write_json(args.out_dir / "baseline_to_full_operator_config_diff.json", {
        "baseline": str(args.baseline),
        "candidate": str(on_path),
        "differences": baseline_diff,
        "authorized_paths": sorted(AUTHORIZED_BASELINE_DIFFS),
        "unexpected_paths": sorted({row["path"] for row in baseline_diff} - AUTHORIZED_BASELINE_DIFFS),
        "status": "passed" if {row["path"] for row in baseline_diff} <= AUTHORIZED_BASELINE_DIFFS else "failed",
    })
    _write_json(args.out_dir / "full_operator_on_vs_off_config_diff.json", {
        "on": str(on_path),
        "off": str(off_path),
        "differences": on_off_diff,
        "authorized_paths": sorted(AUTHORIZED_ON_OFF_DIFFS),
        "unexpected_paths": sorted({row["path"] for row in on_off_diff} - AUTHORIZED_ON_OFF_DIFFS),
        "status": "passed" if {row["path"] for row in on_off_diff} == AUTHORIZED_ON_OFF_DIFFS else "failed",
    })


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only static audit: pre-April f_R-mixture Raman vs frozen production.

Compares the Raman phase semantics across four reference points:
  - c34c3a267cae47b4cf60f7aaca4c60be86c7d9db  (2026-03-13, fixed IIR params)
  - 4c330ac8c5e9ff71a41b2ecc4e29b90cac139650  (2026-03-18, stable pre-April snapshot)
  - 037ead0b31305b0dd1862dc5f665bde7891b7995  (2026-04-02, implementation boundary)
  - e11d13f103c484953c0f733aa9b410bff385b2b5   (frozen physical baseline)

The audit only inspects Git objects; it never checks out a commit and never
mutates the working tree.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

SHAS = {
    "c34c3a": "c34c3a267cae47b4cf60f7aaca4c60be86c7d9db",
    "4c330ac": "4c330ac8c5e9ff71a41b2ecc4e29b90cac139650",
    "037ead0": "037ead0b31305b0dd1862dc5f665bde7891b7995",
    "e11d13f": "e11d13f103c484953c0f733aa9b410bff385b2b5",
}

FILES = {
    "raman.py": "Filament_python/KHz_filament/raman.py",
    "propagate.py": "Filament_python/KHz_filament/propagate.py",
    "nonlinear.py": "Filament_python/KHz_filament/nonlinear.py",
    "config.py": "Filament_python/KHz_filament/config.py",
}


def git_show(sha: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{sha}:{path}"],
        text=True,
    )


def markers(content: str) -> dict[str, bool]:
    def has(needle: str) -> bool:
        return needle in content

    return {
        # raman.py markers
        "kernel_area_normalization": has("h_raw / area") or has("h = h_raw / area"),
        "kernel_analytic_prefactor_only": has("return h.astype(xp.float64)") and not has("h_raw / area"),
        "iir_legacy_loop": has("S = r * S + c * I2[n]"),
        "iir_imag_kS": has("IR[n] = xp.imag(k * S)"),
        "resolve_prefers_omega_gamma": has("omega_R/Gamma_R"),
        # propagate.py markers
        "phase_fr_mixture": has("I_nl = (1.0 - fR) * I + fR * IR"),
        "phase_explicit_two_coefficient": has("delta_n_rot = n_R * IR"),
        "phase_legacy_split_named": has("legacy_split"),
        "shock_on_mixture_before_n2": has("I_kerr = shock_intensity(I_nl"),
        "shock_on_summed_delta_n": has("shock_intensity(\n                delta_n_kerr") or has("shock_intensity(delta_n_kerr"),
        "absorption_conv_deriv_n0_factor": has("(n0 / c0) * n_R * IR * dIdt"),
        "absorption_conv_deriv_no_n0_factor": has("(n_R / c0) * IR * dIdt"),
        "absorption_closed_form_present": has("closed_form"),
        # nonlinear.py markers
        "kerr_phase_used": has("kerr_phase("),
        "kerr_phase_from_deltan_used": has("kerr_phase_from_deltan("),
        "shock_intensity_exists": has("def shock_intensity("),
        # config.py markers
        "raman_f_R_field": has("f_R: float"),
        "raman_n_R_field": has("n_R: float"),
        "raman_operator_mode_field": has("operator_mode:"),
    }


def audit_point(label: str, sha: str) -> dict[str, Any]:
    files = {}
    for name, path in FILES.items():
        try:
            files[name] = markers(git_show(sha, path))
        except subprocess.CalledProcessError:
            files[name] = {"missing": True}
    return {"label": label, "sha": sha, "files": files}


def classify_phase(point: dict[str, Any]) -> str:
    prop = point["files"]["propagate.py"]
    if prop.get("phase_fr_mixture"):
        return "historical_fr_mixture"
    if prop.get("phase_explicit_two_coefficient"):
        return "explicit_n2_elec_I_plus_n_R_IR"
    return "unresolved"


def build_report(points: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    rows = []
    for point in points:
        raman = point["files"]["raman.py"]
        prop = point["files"]["propagate.py"]
        nlin = point["files"]["nonlinear.py"]
        cfg = point["files"]["config.py"]
        rows.append({
            "label": point["label"],
            "sha": point["sha"],
            "phase_semantics": classify_phase(point),
            "kernel": "area_normalized" if raman.get("kernel_area_normalization")
                       else "analytic_prefactor" if raman.get("kernel_analytic_prefactor_only")
                       else "unknown",
            "iir_recursion": "S=rS+cI[n]; IR=Im(kS)" if raman.get("iir_legacy_loop") else "other",
            "f_R_in_phase": bool(prop.get("phase_fr_mixture")),
            "n_R_in_phase": bool(prop.get("phase_explicit_two_coefficient")),
            "self_steepening_order": "shock(I_nl) then *n2" if prop.get("shock_on_mixture_before_n2")
                                      else "shock(summed delta_n)" if prop.get("shock_on_summed_delta_n")
                                      else "n/a",
            "absorption_wR_factor": "n0/c0" if prop.get("absorption_conv_deriv_n0_factor")
                                     else "n_R/c0" if prop.get("absorption_conv_deriv_no_n0_factor")
                                     else "n/a",
            "kerr_phase_api": "kerr_phase" if nlin.get("kerr_phase_used") else "kerr_phase_from_deltan",
        })

    assertions = {
        "legacy_split_is_explicit_nR_IR_not_fr_mixture": (
            rows[-1]["phase_semantics"] == "explicit_n2_elec_I_plus_n_R_IR"
            and not rows[-1]["f_R_in_phase"]
        ),
        "4c330ac_is_fr_mixture": (
            next(r for r in rows if r["label"] == "4c330ac")["phase_semantics"]
            == "historical_fr_mixture"
        ),
        "037ead0_is_boundary": (
            next(r for r in rows if r["label"] == "037ead0")["phase_semantics"]
            == "explicit_n2_elec_I_plus_n_R_IR"
        ),
        "historical_absorption_not_to_restore": (
            next(r for r in rows if r["label"] == "4c330ac")["absorption_wR_factor"] == "n0/c0"
            and rows[-1]["absorption_wR_factor"] == "n_R/c0"
        ),
    }
    summary = {"rows": rows, "assertions": assertions}

    lines = [
        "# Historical f_R-mixture Raman static audit",
        "",
        "Phase semantics and operator path across the four reference points:",
        "",
        "| point | phase semantics | kernel | IIR | f_R in phase | n_R in phase | self-steepening order | absorption w_R |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['phase_semantics']} | {r['kernel']} | {r['iir_recursion']} "
            f"| {r['f_R_in_phase']} | {r['n_R_in_phase']} | {r['self_steepening_order']} | {r['absorption_wR_factor']} |"
        )
    lines += [
        "",
        "## Assertions",
        "",
    ]
    for key, value in assertions.items():
        lines.append(f"- `{key}`: **{value}**")
    lines += [
        "",
        "## Boundary",
        "",
        "- Current `legacy_split` is the explicit two-coefficient form `Δn = n2_elec*I + n_R*I_R`; it is NOT the pre-April f_R mixture.",
        "- The pre-April f_R mixture (4c330ac) uses `I_nl=(1-f_R)I + f_R*I_R`, `Δn=n2*I_nl`, with `f_R=0.15, T2=80ps, T_R=8.4ps, method=iir`.",
        "- 037ead0 (2026-04-02) removed the mixture and introduced `Δn=n2_elec*I + n_R*I_R`.",
        "- Self-steepening: historical applies `shock_intensity` to `I_nl` then multiplies by n2; frozen applies it to the summed `delta_n`. Both are equivalent for the linear tdiff/fft shock operator.",
        "- Raman absorption is intentionally NOT restored: historical conv_deriv uses `(n0/c0)*n_R*IR*dI/dt`; frozen uses `(n_R/c0)*IR*dI/dt`.",
        "",
    ]
    return summary, "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results" / "historical_fr_mixture_causality" / "static_audit")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = [audit_point(label, sha) for label, sha in SHAS.items()]
    summary, markdown = build_report(points)
    (args.out_dir / "raman_historical_vs_frozen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "raman_historical_vs_frozen.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary["assertions"], indent=2))


if __name__ == "__main__":
    main()

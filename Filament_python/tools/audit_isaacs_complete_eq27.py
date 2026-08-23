#!/usr/bin/env python3
"""CPU-only C1 closure audit for the complete Isaacs Eq. (27) operator.

This script evaluates fixed small arrays only.  It does not call the
propagation runner, submit a Slurm job, alter Raman parameters, or write
production NPZ/MAT results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.config import PropagationConfig, RamanConfig  # noqa: E402
from KHz_filament.config_normalize import normalize_config  # noqa: E402
from KHz_filament.constants import c0, eps0  # noqa: E402
from KHz_filament.raman import (  # noqa: E402
    apply_isaacs_complete_eq27_operator_step,
    isaacs_complete_eq27_stage,
    isaacs_raman_stage,
)


BASE_SHA = "c9d9b952c4c23d6839374bdc5de184f0cd389eb3"
N0 = 1.00027
N2 = 7.8e-24
N_R = 2.3e-23
OMEGA_R = 1.6e13
GAMMA_R = 1.3e13
OMEGA0 = 2.0 * np.pi * c0 / 800e-9
# C1 closure window: 384 samples at 2.5 fs = 960 fs.  The earlier 256 x
# 0.5 fs (128 fs) probe left a substantial pulse amplitude at the edges and
# contaminated the field-vs-Eq. (10) energy comparison.
DT = 2.5e-15
NT = 384


def _field(dtype=np.complex128):
    t = (np.arange(NT, dtype=np.float64) - NT // 2) * DT
    intensity = 5.0e17 * np.exp(-4.0 * np.log(2.0) * (t / 120e-15) ** 2)
    phase = 0.15 * np.sin(2.0 * np.pi * t / 85e-15) + 1.2e27 * t * t
    amplitude = np.sqrt(2.0 * intensity / (eps0 * c0 * N0))
    return (amplitude * np.exp(1j * phase)).astype(dtype)[:, None, None]


def _omega():
    return 2.0 * np.pi * np.fft.fftfreq(NT, d=DT)


def _reference_response(intensity):
    """Direct fp64 exact-PWL recurrence, independent of the production call."""
    a = GAMMA_R - 1j * OMEGA_R
    r = np.exp(-a * DT)
    c = (1.0 - r) / a
    c1 = c - (1.0 - r * (1.0 + a * DT)) / (a * a * DT)
    c0_ = c - c1
    k = 1.0 / np.imag(1.0 / a)
    values = np.asarray(intensity, dtype=np.float64).reshape(NT, -1)
    response = np.zeros_like(values, dtype=np.float64)
    state = np.zeros(values.shape[1], dtype=np.complex128)
    for index in range(1, NT):
        state = r * state + c0_ * values[index - 1] + c1 * values[index]
        response[index] = np.imag(k * state)
    return response.reshape(np.asarray(intensity).shape)


def _reference_rhs(field, *, n2, n_R):
    intensity = 0.5 * eps0 * c0 * N0 * np.abs(field) ** 2
    response = _reference_response(intensity)
    source = (float(n2) * intensity + float(n_R) * response) * field
    derivative = np.fft.ifft(
        (1j * _omega())[:, None, None] * np.fft.fft(source, axis=0), axis=0
    )
    return 1j * (OMEGA0 / c0) * source - derivative / c0


def _rhs_kwargs():
    return dict(
        Omega=_omega(), dt=DT, omega0=OMEGA0, n0=N0,
        n2=N2, n_R=N_R, omega_R=OMEGA_R, Gamma_R=GAMMA_R,
        method="iir", iir_sampling="exact_piecewise_linear",
    )


def _relative_l2(actual, expected):
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)) / max(
        np.linalg.norm(np.asarray(expected)), 1e-300
    ))


def _edge_amplitude_ratio(field):
    amplitude = np.abs(np.asarray(field))
    peak = float(np.max(amplitude))
    edge = max(float(np.max(amplitude[0])), float(np.max(amplitude[-1])))
    return edge / max(peak, 1e-300)


def _step(field, dz):
    return apply_isaacs_complete_eq27_operator_step(
        field, dz, integrator="heun", **_rhs_kwargs()
    )


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True,
            capture_output=True, text=True,
        ).stdout.rstrip()
    except Exception:
        return ""


def _git_bytes(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True,
            capture_output=True,
        ).stdout
    except Exception:
        return b""


def _changed_paths():
    allowed = {
        "Filament_python/KHz_filament/Config_explain.md",
        "Filament_python/KHz_filament/README.md",
        "Filament_python/KHz_filament/config.py",
        "Filament_python/KHz_filament/config_normalize.py",
        "Filament_python/KHz_filament/diagnostics.py",
        "Filament_python/KHz_filament/propagate.py",
        "Filament_python/KHz_filament/raman.py",
        "Filament_python/tests/test_isaacs_complete_eq27.py",
        "Filament_python/tools/audit_isaacs_complete_eq27.py",
    }
    status = _git("status", "--short", "--untracked-files=all")
    paths = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    unrelated = [
        path for path in paths
        if path not in allowed and not path.startswith("Filament_python/results/isaacs_complete_eq27/")
    ]
    paths = sorted(set(paths))
    unrelated = sorted(set(unrelated))
    return paths, unrelated


def _provenance(changed_paths):
    """Return dirty-state and a stable hash of the current implementation diff.

    ``HEAD`` alone is insufficient while this audit is run in a dirty
    worktree.  The hash includes the tracked patch and bytes of untracked
    implementation files, with generated C1 report contents excluded so
    rerunning the audit does not hash its own output.
    """
    status = _git_bytes("status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status.strip())
    hasher = hashlib.sha256()
    hasher.update(b"khz-filament-c1-diff-v1\0")
    hasher.update(status.replace(b"\\", b"/"))
    hasher.update(b"\0")
    hasher.update(_git_bytes("diff", "HEAD", "--binary", "--no-ext-diff"))
    result_prefix = "Filament_python/results/isaacs_complete_eq27/"
    for path in sorted(changed_paths):
        if path.startswith(result_prefix):
            continue
        candidate = REPO / Path(path)
        if candidate.is_file() and path.startswith("Filament_python/"):
            hasher.update(b"UNTRACKED_OR_WORKTREE\0")
            hasher.update(path.encode("utf-8"))
            hasher.update(b"\0")
            # This additional record makes untracked test/audit files part of
            # the identity; tracked files are already represented by git diff.
            status_line = status.decode("utf-8", errors="replace")
            if any(
                line[3:].replace("\\", "/") == path
                and line.startswith(("??", "A "))
                for line in status_line.splitlines()
                if len(line) >= 4
            ):
                hasher.update(candidate.read_bytes())
    return {
        "git_dirty": dirty,
        "changed_paths": list(changed_paths),
        "diff_hash_algorithm": "sha256",
        "diff_hash": hasher.hexdigest(),
    }


def run_audit(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    field = _field()
    stage = isaacs_complete_eq27_stage(
        field, return_components=True, **_rhs_kwargs()
    )
    electronic_reference = _reference_rhs(field, n2=N2, n_R=0.0)
    rotational_reference = _reference_rhs(field, n2=0.0, n_R=N_R)
    combined_reference = _reference_rhs(field, n2=N2, n_R=N_R)
    electronic_error = _relative_l2(stage["rhs_electronic"], electronic_reference)
    rotational_error = _relative_l2(stage["rhs_rotational"], rotational_reference)
    combined_error = _relative_l2(stage["rhs"], combined_reference)
    component_error = _relative_l2(
        stage["rhs"], stage["rhs_electronic"] + stage["rhs_rotational"]
    )

    electronic_only = isaacs_complete_eq27_stage(
        field, return_components=True, **{**_rhs_kwargs(), "n_R": 0.0}
    )
    rotational_only = isaacs_complete_eq27_stage(
        field, return_components=True, **{**_rhs_kwargs(), "n2": 0.0}
    )
    doubled_electronic = isaacs_complete_eq27_stage(
        field, return_components=True,
        **{**_rhs_kwargs(), "n2": 2.0 * N2, "n_R": 0.0},
    )
    doubled_rotational = isaacs_complete_eq27_stage(
        field, return_components=True,
        **{**_rhs_kwargs(), "n2": 0.0, "n_R": 2.0 * N_R},
    )
    coefficient_checks = {
        "electronic_zero": float(np.linalg.norm(electronic_only["rhs_rotational"])),
        "rotational_zero": float(np.linalg.norm(rotational_only["rhs_electronic"])),
        "electronic_single_count_rel_l2": _relative_l2(
            electronic_only["rhs"], stage["rhs_electronic"]
        ),
        "rotational_single_count_rel_l2": _relative_l2(
            rotational_only["rhs"], stage["rhs_rotational"]
        ),
        "electronic_double_scale_rel_l2": _relative_l2(
            doubled_electronic["rhs"], 2.0 * stage["rhs_electronic"]
        ),
        "rotational_double_scale_rel_l2": _relative_l2(
            doubled_rotational["rhs"], 2.0 * stage["rhs_rotational"]
        ),
    }

    source = (
        N2 * (0.5 * eps0 * c0 * N0 * np.abs(field) ** 2)
        + N_R * _reference_response(0.5 * eps0 * c0 * N0 * np.abs(field) ** 2)
    ) * field
    derivative = np.fft.ifft(
        (1j * _omega())[:, None, None] * np.fft.fft(source, axis=0), axis=0
    )
    sign_prefactor_error = _relative_l2(
        stage["rhs"], 1j * (OMEGA0 / c0) * source - derivative / c0
    )
    wrong_medium_prefactor_error = _relative_l2(
        stage["rhs"], 1j * (N0 * OMEGA0 / c0) * source - derivative / c0
    )

    dz = 1.0e-5
    whole = _step(field, dz)
    half = _step(field, dz / 2.0)
    half = _step(half, dz / 2.0)
    quarter = _step(field, dz / 4.0)
    for _ in range(3):
        quarter = _step(quarter, dz / 4.0)
    heun_error = float(np.linalg.norm(whole - half))
    finer_error = float(np.linalg.norm(half - quarter))
    heun_ratio = heun_error / max(finer_error, 1e-300)

    edge_amplitude_ratio = _edge_amplitude_ratio(field)
    _, pure_complex128_energy = apply_isaacs_complete_eq27_operator_step(
        field, dz, integrator="heun", return_diagnostics=True,
        diagnose_projection_difference=True, **_rhs_kwargs()
    )
    pure_complex128_energy_closure = {
        "actual_global_energy_loss_J": pure_complex128_energy[
            "actual_global_energy_loss_J"
        ],
        "target_global_energy_loss_J": pure_complex128_energy[
            "target_global_energy_loss_J"
        ],
        "relative_residual": pure_complex128_energy["global_closure_residual"],
        "threshold": 1e-6,
        "passed": bool(pure_complex128_energy["global_closure_residual"] < 1e-6),
    }

    _, projection = apply_isaacs_complete_eq27_operator_step(
        _field(np.complex64), dz, integrator="heun",
        return_diagnostics=True, diagnose_projection_difference=True,
        **_rhs_kwargs()
    )
    changed_paths, unrelated_paths = _changed_paths()
    provenance = _provenance(changed_paths)
    default_cfg = normalize_config({})
    old_stage = isaacs_raman_stage(
        field, Omega=_omega(), dt=DT, omega0=OMEGA0, n0=N0,
        n_R=N_R, omega_R=OMEGA_R, Gamma_R=GAMMA_R,
        method="iir", iir_sampling="exact_piecewise_linear",
    )
    old_error = _relative_l2(old_stage["rhs"], rotational_reference)

    gates = {
        "electronic_D_IA_closure": {
            "passed": bool(electronic_error < 1e-6),
            "relative_l2": electronic_error,
            "threshold": 1e-6,
        },
        "rotational_D_IRA_closure": {
            "passed": bool(rotational_error < 1e-6),
            "relative_l2": rotational_error,
            "threshold": 1e-6,
        },
        "combined_Eq27_closure": {
            "passed": bool(
                combined_error < 1e-6
                and component_error < 1e-12
                and edge_amplitude_ratio < 1e-6
                and pure_complex128_energy_closure["passed"]
            ),
            "relative_l2": combined_error,
            "combined_minus_components_rel_l2": component_error,
            "threshold": 1e-6,
            "pure_complex128_field_vs_Eq10_energy_closure": pure_complex128_energy_closure,
            "edge_amplitude_ratio": edge_amplitude_ratio,
            "edge_amplitude_ratio_threshold": 1e-6,
        },
        "coefficient_single_count_audit": {
            "passed": bool(
                coefficient_checks["electronic_zero"] == 0.0
                and coefficient_checks["rotational_zero"] == 0.0
                and coefficient_checks["electronic_single_count_rel_l2"] < 1e-12
                and coefficient_checks["rotational_single_count_rel_l2"] < 1e-12
                and coefficient_checks["electronic_double_scale_rel_l2"] < 1e-12
                and coefficient_checks["rotational_double_scale_rel_l2"] < 1e-12
            ),
            "checks": coefficient_checks,
        },
        "Heun_convergence": {
            "passed": bool(heun_ratio > 3.5),
            "single_vs_two_half_error": heun_error,
            "two_half_vs_four_quarter_error": finer_error,
            "error_ratio": heun_ratio,
            "expected_minimum_ratio": 3.5,
        },
        "production_default_unchanged": {
            "passed": bool(
                default_cfg["raman"]["operator_mode"] == "legacy_split"
                and default_cfg["raman"]["operator_convention"] == "legacy"
                and RamanConfig().operator_mode == "legacy_split"
                and PropagationConfig().use_raman_full_operator is None
                and old_error < 1e-6
            ),
            "default_operator_mode": default_cfg["raman"]["operator_mode"],
            "default_operator_convention": default_cfg["raman"]["operator_convention"],
            "old_rotational_reference_rel_l2": old_error,
        },
        "no_Raman_parameter_change": {
            "passed": bool(
                N2 == 7.8e-24 and N_R == 2.3e-23
                and OMEGA_R == 1.6e13 and GAMMA_R == 1.3e13
            ),
            "n2": N2, "n_R": N_R,
            "omega_R": OMEGA_R, "Gamma_R": GAMMA_R,
        },
        "no_unrelated_physics_modification": {
            "passed": not unrelated_paths,
            "allowed_diff_scope": sorted(set(changed_paths) - set(unrelated_paths)),
            "unrelated_paths": unrelated_paths,
        },
    }
    overall_passed = all(bool(item["passed"]) for item in gates.values())
    payload = {
        "schema": "khz_filament.isaacs_complete_eq27.c1.v1",
        "overall": "PASS" if overall_passed else "STOP",
        "next_action": "C1 audit complete; parent must decide whether C2 is admissible" if overall_passed else "STOP; do not submit C2 Slurm propagation",
        "base_sha": BASE_SHA,
        "current_sha": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "git_dirty": provenance["git_dirty"],
        "changed_paths": provenance["changed_paths"],
        "diff_hash_algorithm": provenance["diff_hash_algorithm"],
        "diff_hash": provenance["diff_hash"],
        "full_propagation_run": False,
        "slurm_jobs_submitted": 0,
        "raman_parameters_changed": False,
        "production_default_changed": False,
        "gates": gates,
        "operator": {
            "mode": "full_isaacs_eq27_complete",
            "source_definition": "S=(n2*I+n_R*I_R)*A",
            "intensity_definition": "I=0.5*eps0*c0*n0*abs(A)^2",
            "rhs_definition": "i*(omega0/c0)*S-(1/c0)*d_tau(S)",
            "prefactor": "vacuum omega0/c0",
            "iir_sampling": "exact_piecewise_linear",
            "electronic_coefficient_count": 1,
            "rotational_coefficient_count": 1,
        },
        "projection": {
            "energy_projection_scale": projection["energy_projection_scale"],
            "single_step_field_relative_diff": projection["projection_field_relative_l2"],
            "single_step_energy_relative_diff": projection["projection_energy_difference_relative"],
            "status": projection["projection_status"],
            "not_primary_for_C2": projection["projection_status"] == "not_primary_for_C2",
        },
        "energy_closure_supporting": {
            "window_samples": NT,
            "window_duration_s": NT * DT,
            "dt_s": DT,
            "edge_amplitude_ratio": edge_amplitude_ratio,
            "edge_amplitude_ratio_threshold": 1e-6,
            "pure_complex128_field_vs_Eq10_relative_residual": pure_complex128_energy_closure[
                "relative_residual"
            ],
            "pure_complex128_field_vs_Eq10_threshold": 1e-6,
            "pure_complex128_field_vs_Eq10_passed": pure_complex128_energy_closure[
                "passed"
            ],
            "old_128fs_window_edge_amplitude_ratio": 0.678285838808053,
            "old_128fs_window_spurious_flux_fraction": 0.08115089017737817,
        },
        "allowed_diff_scope": sorted(set(changed_paths) - set(unrelated_paths)),
    }
    (out_dir / "c1_closure_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate_lines = [
        "# Isaacs complete Eq.27 C1 operator report", "",
        f"Overall: **{payload['overall']}**.", "",
        f"Base SHA: `{BASE_SHA}`; current HEAD: `{payload['current_sha']}`.",
        f"Git dirty: **{payload['git_dirty']}**; implementation diff hash ({payload['diff_hash_algorithm']}): `{payload['diff_hash']}`.",
        "Implementation identity is the base/current SHA pair plus the dirty-worktree diff hash; HEAD alone is not sufficient.",
        "Changed paths:",
        *[f"  - `{path}`" for path in payload["changed_paths"]],
        "No propagation or Slurm job was run.", "",
        "## Gates", "",
    ]
    for name, gate in gates.items():
        gate_lines.append(f"- `{name}`: **{'PASS' if gate['passed'] else 'FAIL'}**")
    gate_lines.extend([
        "", "## Closure metrics", "",
        f"- electronic D[I A] relative L2: `{electronic_error:.6e}`",
        f"- rotational D[I_R A] relative L2: `{rotational_error:.6e}`",
        f"- combined Eq.27 relative L2: `{combined_error:.6e}`",
        f"- combined-minus-components relative L2: `{component_error:.6e}`",
        f"- Heun dz-halving error ratio: `{heun_ratio:.6e}`",
        f"- vacuum-prefactor/sign relative L2: `{sign_prefactor_error:.6e}`",
        f"- wrong n0*omega0 prefactor separation: `{wrong_medium_prefactor_error:.6e}`",
        f"- 960 fs edge amplitude ratio: `{edge_amplitude_ratio:.6e}`",
        f"- pure complex128 field-vs-Eq.10 energy residual: `{pure_complex128_energy_closure['relative_residual']:.6e}`",
        "- The old 128 fs window had edge amplitude ratio `0.678`; its truncated-tail comparison showed `8.1%` spurious flux.",
        "", "## complex64 projection", "",
        f"- scale: `{projection['energy_projection_scale']:.12g}`",
        f"- single-step field relative difference: `{projection['projection_field_relative_l2']:.6e}`",
        f"- single-step energy relative difference: `{projection['projection_energy_difference_relative']:.6e}`",
        f"- status: **{projection['projection_status']}**",
        "", "## Scope", "",
        "- Raman parameters are fixed at the C1 audit values; no f_R or historical mixture is used.",
        "- Existing `full_isaacs_eq27` remains rotational-only plus scalar electronic; only the new complete mode uses combined D[(n2 I+n_R I_R)A].",
        "- Ionization, plasma, BK-NEE, self-steepening coefficients, defaults, and production results were not changed.",
        "- Complete-mode scalar `dphi_kerr` is not applicable; self-steepening is represented by the full product derivative.",
        "- A failing gate means STOP and no C2 submission.",
    ])
    (out_dir / "c1_operator_report.md").write_text(
        "\n".join(gate_lines) + "\n", encoding="utf-8"
    )
    state_lines = [
        "# C1 project state", "",
        f"- Baseline SHA: `{BASE_SHA}`",
        f"- Audit current HEAD: `{payload['current_sha']}`",
        f"- Branch: `{payload['branch']}`",
        f"- Git dirty: `{payload['git_dirty']}`",
        f"- Implementation diff hash ({payload['diff_hash_algorithm']}): `{payload['diff_hash']}`",
        "- Changed paths:",
        *[f"  - `{path}`" for path in payload["changed_paths"]],
        "- Scope: complete Eq.27 electronic+rotational operator closure only",
        "- Raman parameter change: none",
        "- Production/default change: none",
        "- Full propagation: not run",
        "- Slurm submission: none",
        f"- Overall C1 gate: **{payload['overall']}**",
        "- C2 status: parent scientific decision required; this audit does not authorize submission",
    ]
    (out_dir / "PROJECT_STATE.md").write_text(
        "\n".join(state_lines) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "isaacs_complete_eq27",
    )
    args = parser.parse_args()
    payload = run_audit(args.out_dir)
    print(json.dumps({
        "overall": payload["overall"],
        "out_dir": str(args.out_dir),
        "electronic_rel_l2": payload["gates"]["electronic_D_IA_closure"]["relative_l2"],
        "combined_rel_l2": payload["gates"]["combined_Eq27_closure"]["relative_l2"],
        "energy_closure_residual": payload["energy_closure_supporting"][
            "pure_complex128_field_vs_Eq10_relative_residual"
        ],
        "git_dirty": payload["git_dirty"],
        "diff_hash": payload["diff_hash"],
        "projection_status": payload["projection"]["status"],
    }, indent=2))
    return 0 if payload["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

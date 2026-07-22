#!/usr/bin/env python3
"""Audit the actual BK-NEE production transfer multiplier without replacing it.

The implementation mirrors the phase construction in
``linear.step_linear_bk_nee_factorized``.  It does not construct an alternate
physics kernel and deliberately records unavailable checkpoint-bin fractions
when an archive has no saved complex fields.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def bk_kernel_stats(*, omega: np.ndarray, kperp2: np.ndarray, k0: float,
                    omega0: float, dz_eff: float, beta2: float,
                    denom_floor: float, dtype: np.dtype) -> dict:
    rdtype = np.float32 if dtype == np.complex64 else np.float64
    omega = np.asarray(omega, dtype=rdtype)
    kperp2 = np.asarray(kperp2, dtype=rdtype)
    denom = 1.0 + omega / float(omega0)
    denom = np.where(denom >= 0.0, 1.0, -1.0) * np.maximum(np.abs(denom), denom_floor)
    coeff_diff = -1.0 / (2.0 * float(k0) * denom)
    coeff_gvd = 0.5 * float(beta2) * omega**2
    total = int(omega.size * kperp2.size)
    values = []
    zero = 0
    below = {tol: 0 for tol in (1e-7, 1e-6, 1e-5)}
    above = {tol: 0 for tol in (1e-7, 1e-6, 1e-5)}
    for i in range(omega.size):
        phase = (coeff_diff[i] * kperp2 + coeff_gvd[i]) * float(dz_eff)
        h = np.exp((1j * phase).astype(dtype, copy=False)).astype(dtype, copy=False)
        modulus = np.abs(h).astype(np.float64, copy=False)
        values.append((float(np.min(modulus)), float(np.max(modulus)), float(np.max(np.abs(modulus - 1.0)))))
        zero += int(np.count_nonzero(h == 0))
        for tol in below:
            below[tol] += int(np.count_nonzero(modulus < 1.0 - tol))
            above[tol] += int(np.count_nonzero(modulus > 1.0 + tol))
    return {
        "dtype": np.dtype(dtype).name,
        "total_bins": total,
        "min_abs_H": min(item[0] for item in values),
        "max_abs_H": max(item[1] for item in values),
        "max_abs_abs_H_minus_1": max(item[2] for item in values),
        "zero_bins": zero,
        "zero_fraction": zero / total,
        "below_unity": {str(tol): {"count": count, "fraction": count / total} for tol, count in below.items()},
        "above_unity": {str(tol): {"count": count, "fraction": count / total} for tol, count in above.items()},
    }


def audit(config_path: Path) -> dict:
    import sys
    sys.path.insert(0, str(ROOT))
    from KHz_filament.constants import c0
    from KHz_filament.grids import make_axes

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    grid, beam, prop = cfg["grid"], cfg["beam"], cfg["propagation"]
    axes = make_axes(grid["Nx"], grid["Ny"], grid["Nt"], grid["Lx"], grid["Ly"], grid["Twin"])
    omega0 = 2.0 * math.pi * c0 / float(beam["lam0"])
    k0 = float(beam["n0"]) * omega0 / c0
    dz_eff = float(prop["dz"]) / 2.0
    beta2 = float(prop.get("nee_beta2", 0.0))
    denom_floor = float(prop.get("nee_denom_floor", 1e-4))
    omega = np.asarray(axes.Omega)
    kperp2 = np.asarray(axes.kperp2)
    return {
        "schema": "khz_filament.phase8b_r.linear_transfer_kernel_audit.v1",
        "config": str(config_path).replace("\\", "/"),
        "selected_production_branch": "bk_nee -> step_linear_bk_nee_factorized",
        "grid": {"Nt": int(grid["Nt"]), "Ny": int(grid["Ny"]), "Nx": int(grid["Nx"]), "total_3d_bins": int(omega.size * kperp2.size)},
        "parameters": {"k0_m_inverse": k0, "omega0_rad_s": omega0, "dz_eff_m": dz_eff, "nee_beta2": beta2, "nee_denom_floor": denom_floor},
        "pure_phase_design": True,
        "explicit_linear_loss_operations": [],
        "actual_complex64": bk_kernel_stats(omega=omega, kperp2=kperp2, k0=k0, omega0=omega0, dz_eff=dz_eff, beta2=beta2, denom_floor=denom_floor, dtype=np.complex64),
        "float64_reference": bk_kernel_stats(omega=omega, kperp2=kperp2, k0=k0, omega0=omega0, dz_eff=dz_eff, beta2=beta2, denom_floor=denom_floor, dtype=np.complex128),
        "checkpoint_energy_fraction_in_attenuated_bins": "unavailable: Job 179988 saved no complex field checkpoints",
        "predicted_loss_from_kernel": "not inferable without field spectra; pure-phase design predicts zero exact-arithmetic loss",
    }


def write_artifacts(config_path: Path, output_dir: Path) -> dict:
    payload = audit(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "linear_transfer_kernel_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    actual = payload["actual_complex64"]
    reference = payload["float64_reference"]
    lines = [
        "# Linear transfer kernel audit",
        "",
        f"- Production branch: `{payload['selected_production_branch']}`.",
        "- The selected BK-NEE transfer is a pure phase by design; it has no physical linear absorption, high-k filter, evanescent deletion, mask, crop, or padding operation.",
        f"- complex64 `max(abs(abs(H)-1))`: `{actual['max_abs_abs_H_minus_1']}`.",
        f"- float64 reference `max(abs(abs(H)-1))`: `{reference['max_abs_abs_H_minus_1']}`.",
        f"- complex64 zero bins: `{actual['zero_bins']}` of `{actual['total_bins']}`.",
        "- Per-checkpoint attenuated-bin energy fractions cannot be reconstructed from Job 179988 because it contains no complex field checkpoints.",
        "",
    ]
    (output_dir / "linear_transfer_kernel_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on_energy_audit.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_artifacts(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

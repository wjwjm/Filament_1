#!/usr/bin/env python3
"""Audit whether an archive contains fields required for linear-step replay.

This deliberately refuses to synthesize an input field from scalar energy
histories.  When field checkpoints are absent, it exports the observed
half-step energy changes with an explicit inconclusive status instead.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELD_PREFIXES = (
    "linear_checkpoint_field_",
    "field_checkpoint_",
    "E_checkpoint_",
)


def _select_indices(data: np.lib.npyio.NpzFile) -> list[tuple[str, int]]:
    z = np.asarray(data["z_axis"], dtype=np.float64)
    imax = np.asarray(data["I_max_z"], dtype=np.float64)
    peak = int(np.argmax(imax))
    rising = np.flatnonzero((np.arange(z.size) < peak) & (imax >= 0.1 * imax[peak]))
    pre_rise = int(rising[0]) if rising.size else 0
    near_focus = int(np.argmin(np.abs(z - 0.85)))
    post_peak = int(np.argmin(np.abs(z - 1.05)))
    return [
        ("early", 0),
        ("pre_rise_10pct_peak", pre_rise),
        ("near_focus_window", near_focus),
        ("intensity_peak", peak),
        ("post_peak", post_peak),
        ("final", z.size - 1),
    ]


def audit(npz_path: Path) -> tuple[list[dict], dict]:
    with np.load(npz_path, allow_pickle=False) as data:
        present_fields = [key for key in data.files if key.startswith(FIELD_PREFIXES)]
        z = np.asarray(data["z_axis"], dtype=np.float64)
        e0 = np.asarray(data["energy_step_start_J"], dtype=np.float64)
        e1 = np.asarray(data["energy_after_linear_half1_J"], dtype=np.float64)
        e4 = np.asarray(data["energy_after_raman_post_J"], dtype=np.float64)
        e5 = np.asarray(data["energy_after_linear_half2_J"], dtype=np.float64)
        rows = []
        for label, index in _select_indices(data):
            for half, before, after in (
                ("linear_halfstep_1", e0[index], e1[index]),
                ("linear_halfstep_2", e4[index], e5[index]),
            ):
                delta = float(after - before)
                rows.append({
                    "checkpoint_label": label,
                    "checkpoint_index": index,
                    "z_m": float(z[index]),
                    "halfstep": half,
                    "U_input_J": float(before),
                    "U_output_J": float(after),
                    "absolute_delta_J": delta,
                    "relative_delta": delta / max(abs(float(before)), 1e-300),
                    "U_after_forward_fft_J": None,
                    "U_after_transfer_J": None,
                    "U_after_inverse_fft_J": None,
                    "U_after_mask_J": None,
                    "U_after_filter_J": None,
                    "U_after_crop_J": None,
                    "replay_status": "inconclusive_missing_field_checkpoint",
                    "replay_reason": "archive stores scalar checkpoint energies but no complex field checkpoint",
                })
    summary = {
        "schema": "khz_filament.phase8b_r.linear_checkpoint_replay.v1",
        "status": "inconclusive_missing_field_checkpoints",
        "archive": str(npz_path),
        "field_checkpoint_keys": present_fields,
        "field_replay_performed": False,
        "field_replay_forbidden_without_saved_field": True,
        "checkpoint_count": 6,
        "halfstep_rows": len(rows),
        "observed_energy_only": True,
    }
    return rows, summary


def write_artifacts(npz_path: Path, output_dir: Path) -> dict:
    rows, summary = audit(npz_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "linear_checkpoint_replay.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "linear_checkpoint_replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = "\n".join([
        "# Linear checkpoint replay audit",
        "",
        "- Status: **inconclusive_missing_field_checkpoints**.",
        "- The archive has six scalar energy histories but no complex field checkpoint array.",
        "- No field is reconstructed from energy scalars and no theoretical substitute kernel is used.",
        "- The CSV reports actual archived half-step energy changes only.",
        "- Internal FFT/transfer/inverse-FFT losses require the opt-in R4 linear diagnostics on a controlled short smoke.",
        "",
    ])
    (output_dir / "linear_checkpoint_replay_report.md").write_text(report, encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_artifacts(args.npz, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

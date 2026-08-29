"""Localize HR-2E longitudinal deposition differences without propagation.

The tool reads only canonical scalar interval ledgers.  It rebuilds the
HR-2C-R Class-A Raman ledger when needed, uses conservative overlap reductions
at the fixed 0.75 m and 1.05 m boundaries, and never writes an NPZ.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from hr2e_schedule_convergence import (  # noqa: E402
    conservative_remap,
    load_canonical_npz,
    union_edges,
)


CHANNELS = ("ion", "raman", "total")
REGIONS = (
    ("pre_focus", 0.0, 0.75),
    ("focus", 0.75, 1.05),
    ("post_focus", 1.05, None),
)


def _extended_common_edges(first: Iterable[float], second: Iterable[float]) -> np.ndarray:
    """Return the conservative comparison grid split at the focus boundaries."""
    common = union_edges(first, second)
    values = np.unique(np.concatenate((common, np.asarray((0.75, 1.05)))))
    values[0] = common[0]
    values[-1] = common[-1]
    return values


def _region_energy(edges: np.ndarray, energy: np.ndarray, left: float, right: float) -> float:
    """Conservatively reduce piecewise-constant interval energy over one span."""
    widths = np.diff(edges)
    density = energy / widths
    overlap = np.maximum(0.0, np.minimum(edges[1:], right) - np.maximum(edges[:-1], left))
    return float(np.sum(density * overlap))


def _safe_share(value: float, total: float) -> float | None:
    if abs(total) <= 1.0e-30:
        return None
    return float(value / total)


def compare_channel(
    left: dict[str, Any], right: dict[str, Any], channel: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return segment and cumulative signed-energy evidence for one channel."""
    left_edges = np.asarray(left["z_edges"], dtype=np.float64)
    right_edges = np.asarray(right["z_edges"], dtype=np.float64)
    left_energy = np.asarray(left["channels"][channel], dtype=np.float64)
    right_energy = np.asarray(right["channels"][channel], dtype=np.float64)
    common = _extended_common_edges(left_edges, right_edges)
    left_common = conservative_remap(left_edges, left_energy, common)
    right_common = conservative_remap(right_edges, right_energy, common)
    delta_interval = left_common - right_common
    cumulative = np.concatenate(([0.0], np.cumsum(delta_interval)))
    total_delta = float(cumulative[-1])
    segments = []
    for name, start, stop in REGIONS:
        end = float(common[-1]) if stop is None else stop
        left_value = _region_energy(left_edges, left_energy, start, end)
        right_value = _region_energy(right_edges, right_energy, start, end)
        difference = left_value - right_value
        segments.append({
            "region": name,
            "z_start_m": start,
            "z_end_m": end,
            "left_energy_J": left_value,
            "right_energy_J": right_value,
            "left_minus_right_J": difference,
            "absolute_difference_J": abs(difference),
            "share_of_full_signed_difference": _safe_share(difference, total_delta),
            "absolute_share_of_full_signed_difference": _safe_share(abs(difference), abs(total_delta)),
        })
    boundary = {float(value): int(np.where(np.isclose(common, value, rtol=0.0, atol=1e-12))[0][0]) for value in (0.75, 1.05)}
    max_index = int(np.argmax(np.abs(cumulative)))
    report = {
        "channel": channel,
        "left_pulse_energy_J": float(left_energy.sum()),
        "right_pulse_energy_J": float(right_energy.sum()),
        "full_left_minus_right_J": total_delta,
        "full_absolute_difference_J": abs(total_delta),
        "segments": segments,
        "cumulative": {
            "delta_at_0_75_m_J": float(cumulative[boundary[0.75]]),
            "delta_at_1_05_m_J": float(cumulative[boundary[1.05]]),
            "delta_at_end_J": total_delta,
            "change_pre_focus_J": float(cumulative[boundary[0.75]]),
            "change_focus_J": float(cumulative[boundary[1.05]] - cumulative[boundary[0.75]]),
            "change_post_focus_J": float(cumulative[-1] - cumulative[boundary[1.05]]),
            "max_absolute_delta_J": float(abs(cumulative[max_index])),
            "max_absolute_delta_z_m": float(common[max_index]),
        },
    }
    return report, {"edges": common, "cumulative": cumulative}


def build_report(coarse: dict[str, Any], candidate: dict[str, Any], fine: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    pairs = {
        "coarse_to_candidate": (coarse, candidate),
        "candidate_to_fine": (candidate, fine),
    }
    report: dict[str, Any] = {
        "schema": "khz_filament.hr2e.error_localization.v1",
        "focus_window_m": [0.75, 1.05],
        "channels": list(CHANNELS),
        "comparisons": {},
    }
    curves: dict[str, dict[str, np.ndarray]] = {}
    for name, (left, right) in pairs.items():
        comparison: dict[str, Any] = {
            "left_label": left["label"],
            "right_label": right["label"],
            "channels": {},
        }
        pair_curves: dict[str, np.ndarray] = {}
        for channel in CHANNELS:
            channel_report, curve = compare_channel(left, right, channel)
            comparison["channels"][channel] = channel_report
            pair_curves[channel] = curve
        report["comparisons"][name] = comparison
        curves[name] = pair_curves
    return report, curves


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "comparison", "channel", "region", "z_start_m", "z_end_m",
            "left_energy_J", "right_energy_J", "left_minus_right_J",
            "absolute_difference_J", "share_of_full_signed_difference",
            "absolute_share_of_full_signed_difference",
        ))
        writer.writeheader()
        for comparison_name, comparison in report["comparisons"].items():
            for channel, values in comparison["channels"].items():
                for row in values["segments"]:
                    writer.writerow({"comparison": comparison_name, "channel": channel, **row})


def _write_plot(path: Path, curves: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    colors = {"ion": "#2563eb", "raman": "#dc2626", "total": "#111827"}
    for channel in CHANNELS:
        curve = curves["candidate_to_fine"][channel]
        axis.plot(
            curve["edges"], curve["cumulative"], label=channel,
            color=colors[channel], linewidth=1.6,
        )
    for boundary in (0.75, 1.05):
        axis.axvline(boundary, color="#6b7280", linewidth=0.9, linestyle="--")
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8)
    axis.set(xlabel="z [m]", ylabel="candidate cumulative E - fine cumulative E [J]",
             title="HR-2E 120 fs candidate-to-fine cumulative deposition difference")
    axis.legend(title="channel")
    axis.grid(alpha=0.22)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--fine", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    runs = {
        "coarse": load_canonical_npz(args.coarse, label="coarse"),
        "candidate": load_canonical_npz(args.candidate, label="candidate"),
        "fine": load_canonical_npz(args.fine, label="fine"),
    }
    report, curves = build_report(runs["coarse"], runs["candidate"], runs["fine"])
    report["input_paths"] = {name: str(path) for name, path in (
        ("coarse", args.coarse), ("candidate", args.candidate), ("fine", args.fine)
    )}
    report["hr2c_r_contract_reconstructed"] = {
        name: bool(run["hr2c_r_contract_reconstructed"])
        for name, run in runs.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "error_localization_summary.json", report)
    _write_csv(args.output_dir / "error_localization_segments.csv", report)
    _write_plot(args.output_dir / "cumulative_delta_energy_candidate_vs_fine.png", curves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

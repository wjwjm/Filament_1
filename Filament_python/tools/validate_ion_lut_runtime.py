#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate ionization LUT runtime evaluator against reference evaluator (species-wise)."
    )
    ap.add_argument("--config", required=True, help="Path to config json/yaml/toml")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--species", nargs="+", default=None, help="Optional species names to include")
    ap.add_argument("--backend", choices=("numpy", "cupy"), default="numpy", help="Backend preference")

    ap.add_argument("--num-points", type=int, default=3000, help="Number of log-spaced intensity samples")
    ap.add_argument("--onset-W-min", type=float, default=1e4, help="Auto onset lower bound on W_reference [1/s]")
    ap.add_argument("--onset-W-max", type=float, default=1e10, help="Auto onset upper bound on W_reference [1/s]")
    ap.add_argument("--onset-I-min", type=float, default=None, help="Manual onset window lower intensity [W/m^2]")
    ap.add_argument("--onset-I-max", type=float, default=None, help="Manual onset window upper intensity [W/m^2]")
    ap.add_argument("--fallback-zoom-I-min", type=float, default=1e13, help="Fallback onset zoom min when onset mask is empty")
    ap.add_argument("--fallback-zoom-I-max", type=float, default=1e16, help="Fallback onset zoom max when onset mask is empty")

    ap.add_argument("--rel-floor-abs", type=float, default=1e-300, help="Absolute floor lower bound for relative error denominator")
    ap.add_argument("--rel-floor-scale", type=float, default=1e-30, help="Relative floor scaling: floor=max(abs, scale*max(W_ref))")
    ap.add_argument("--min-onset-points", type=int, default=16, help="Warn if onset window has fewer points than this")
    return ap.parse_args()


def _setup_backend(backend: str) -> None:
    os.environ["UPPE_USE_GPU"] = "1" if backend == "cupy" else "0"


def _ensure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    filament_root = repo_root / "Filament_python"
    import sys

    for p in (str(repo_root), str(filament_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_reference_and_lut_rate(resolved_rate: str) -> Tuple[str | None, str | None]:
    r = str(resolved_rate).lower()
    if r in ("ppt_talebpour_i_lut", "ppt_talebpour_i_full_reference", "ppt_talebpour_i_legacy"):
        return "ppt_talebpour_i_full_reference", "ppt_talebpour_i_lut"
    if r in ("popruzhenko_atom_i_lut", "popruzhenko_atom_i_full_reference", "popruzhenko_atom_i_legacy"):
        return "popruzhenko_atom_i_full_reference", "popruzhenko_atom_i_lut"
    return None, None


def _onset_mask(
    I_grid: np.ndarray,
    W_ref: np.ndarray,
    *,
    onset_i_min: float | None,
    onset_i_max: float | None,
    onset_w_min: float,
    onset_w_max: float,
    fallback_i_min: float,
    fallback_i_max: float,
) -> Tuple[np.ndarray, str, List[str]]:
    notes: List[str] = []
    if onset_i_min is not None or onset_i_max is not None:
        if onset_i_min is None or onset_i_max is None:
            raise ValueError("Manual onset intensity window requires both --onset-I-min and --onset-I-max")
        lo, hi = sorted((float(onset_i_min), float(onset_i_max)))
        mask = (I_grid >= lo) & (I_grid <= hi)
        source = "manual"
        if not np.any(mask):
            notes.append("manual onset intensity window has 0 points; fallback to intensity zoom window")
            zlo, zhi = sorted((float(fallback_i_min), float(fallback_i_max)))
            mask = (I_grid >= zlo) & (I_grid <= zhi)
            source = "manual_fallback_zoom"
        return mask, source, notes

    lo_w, hi_w = sorted((float(onset_w_min), float(onset_w_max)))
    mask = (W_ref >= lo_w) & (W_ref <= hi_w)
    source = "auto"
    if not np.any(mask):
        notes.append("auto onset rate window has 0 points; fallback to intensity zoom window")
        zlo, zhi = sorted((float(fallback_i_min), float(fallback_i_max)))
        mask = (I_grid >= zlo) & (I_grid <= zhi)
        source = "auto_fallback_zoom"
    return mask, source, notes


def _percentiles(arr: np.ndarray) -> Dict[str, float]:
    return {
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _plot_species(
    outdir: Path,
    species_name: str,
    model_name: str,
    onset_source: str,
    I_grid: np.ndarray,
    W_ref: np.ndarray,
    W_lut: np.ndarray,
    rel_err: np.ndarray,
    onset_mask: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title_suffix = f"species={species_name} model={model_name} onset={onset_source}"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(I_grid, W_ref, label="W_reference")
    ax.plot(I_grid, W_lut, "--", label="W_LUT")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("I [W/m^2]")
    ax.set_ylabel("W [1/s]")
    ax.set_title(f"Rate compare ({title_suffix})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{species_name}_rate_compare_full.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(I_grid, rel_err)
    ax.set_xscale("log")
    ax.set_xlabel("I [W/m^2]")
    ax.set_ylabel("relative error")
    ax.set_title(f"Relative error ({title_suffix})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{species_name}_relerr_full.png", dpi=180)
    plt.close(fig)

    Ion = I_grid[onset_mask]
    Wr_on = W_ref[onset_mask]
    Wl_on = W_lut[onset_mask]
    Eon = rel_err[onset_mask]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(Ion, Wr_on, label="W_reference")
    ax.plot(Ion, Wl_on, "--", label="W_LUT")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("I [W/m^2]")
    ax.set_ylabel("W [1/s]")
    ax.set_title(f"Onset rate compare ({title_suffix})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{species_name}_rate_compare_onset.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(Ion, Eon)
    ax.set_xscale("log")
    ax.set_xlabel("I [W/m^2]")
    ax.set_ylabel("relative error")
    ax.set_title(f"Onset relative error ({title_suffix})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{species_name}_relerr_onset.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    _setup_backend(args.backend)
    _ensure_import_path()

    from Filament_python.KHz_filament.confio import load_all
    from Filament_python.KHz_filament.constants import c0
    from Filament_python.KHz_filament.device import debug_backend
    from Filament_python.KHz_filament.ionization import (
        _canonical_table_metadata,
        _get_reference_evaluator,
        _ion_rate_table_defaults,
        _resolve_rate,
        _table_path,
        _table_signature,
        eval_rate_from_table,
        prepare_ionization_lut_for_species,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _grid, beam, _prop, ion, _heat, _run, _raman = load_all(args.config)
    omega0 = 2.0 * math.pi * float(c0) / float(beam.lam0)
    n0 = float(beam.n0)
    table_cfg = _ion_rate_table_defaults(ion)

    species_all = list(getattr(ion, "species", None) or [])
    if args.species:
        selected = {s.lower() for s in args.species}
        species_all = [sp for sp in species_all if str(sp.get("name", "")).lower() in selected]

    if not species_all:
        raise ValueError("No species selected/found in config.")

    print(f"[ion-lut-validate] config={args.config}")
    print(f"[ion-lut-validate] backend={debug_backend()}")
    print(f"[ion-lut-validate] outdir={outdir}")

    summary_rows: List[Dict[str, object]] = []
    weighted_ref: List[np.ndarray] = []
    weighted_lut: List[np.ndarray] = []
    weighted_frac: List[float] = []

    for sp in species_all:
        sp_name = str(sp.get("name", "species"))
        rate_resolved = _resolve_rate(sp, ion)
        ref_model, lut_rate = _resolve_reference_and_lut_rate(rate_resolved)
        if ref_model is None or lut_rate is None:
            print(f"[ion-lut-validate] skip species={sp_name} rate={rate_resolved}: unsupported for LUT/reference compare")
            continue

        sp_lut = dict(sp)
        sp_lut["rate"] = lut_rate
        sp_lut.setdefault("reference_model", ref_model)

        metadata = _canonical_table_metadata(
            model_name=sp_lut["reference_model"],
            species_name=sp_name,
            species_params=sp_lut,
            omega0_SI=omega0,
            n0=n0,
            table_cfg=table_cfg,
        )
        signature = _table_signature(metadata)
        lut_path = _table_path(str(table_cfg.get("cache_dir", "cache/rate_tables")), sp_name, signature)
        existed_before = os.path.exists(lut_path)

        table = prepare_ionization_lut_for_species(
            species_params=sp_lut,
            omega0_SI=omega0,
            n0=n0,
            rate_table_cfg=table_cfg,
        )

        lut_source = "cache" if existed_before else "generated"
        print(f"[ion-lut-validate] species={sp_name} lut_source={lut_source} lut_file={lut_path}")

        i_min = float(table_cfg["I_min_SI"])
        i_max = float(table_cfg["I_max_SI"])
        I_grid = np.logspace(math.log10(i_min), math.log10(i_max), int(max(32, args.num_points)))

        ref_opts = {
            "cycle_avg_samples_ref": int(table_cfg.get("ref_cycle_avg_samples", 64)),
            "sum_rel_tol": float(table_cfg.get("popruzhenko_sum_tol", 1e-6)),
            "max_terms": int(table_cfg.get("popruzhenko_max_terms", 256)),
            "W_cap": float(getattr(ion, "W_cap", 1e16)),
        }
        ref_eval = _get_reference_evaluator(sp_lut["reference_model"], sp_lut, omega0, n0, ref_opts)

        W_ref = np.asarray(ref_eval(I_grid), dtype=np.float64)
        W_lut = np.asarray(eval_rate_from_table(I_grid, table, method=table_cfg.get("interp_mode", "loglog")), dtype=np.float64)

        floor = max(float(args.rel_floor_abs), float(args.rel_floor_scale) * float(np.max(W_ref)))
        rel_err = np.abs(W_lut - W_ref) / np.maximum(W_ref, floor)
        print(f"[ion-lut-validate] species={sp_name} rel_err_floor={floor:.3e}")

        onset_mask, onset_source, notes = _onset_mask(
            I_grid,
            W_ref,
            onset_i_min=args.onset_I_min,
            onset_i_max=args.onset_I_max,
            onset_w_min=args.onset_W_min,
            onset_w_max=args.onset_W_max,
            fallback_i_min=args.fallback_zoom_I_min,
            fallback_i_max=args.fallback_zoom_I_max,
        )
        if not np.any(onset_mask):
            raise RuntimeError(f"species={sp_name}: onset mask still empty after fallback")
        if np.count_nonzero(onset_mask) < int(args.min_onset_points):
            notes.append(
                f"onset window has few points: {int(np.count_nonzero(onset_mask))} < min_onset_points={int(args.min_onset_points)}"
            )

        full_stats = _percentiles(rel_err)
        onset_rel = rel_err[onset_mask]
        onset_stats = _percentiles(onset_rel)
        onset_I = I_grid[onset_mask]
        onset_max_idx = int(np.argmax(onset_rel))

        data_file = outdir / f"{sp_name}_lut_validation_data.npz"
        np.savez_compressed(
            data_file,
            I_grid=I_grid,
            W_reference=W_ref,
            W_LUT=W_lut,
            rel_err=rel_err,
            onset_mask=onset_mask.astype(np.uint8),
            onset_I_grid=onset_I,
            onset_W_reference=W_ref[onset_mask],
            onset_W_LUT=W_lut[onset_mask],
            onset_rel_err=onset_rel,
        )

        _plot_species(outdir, sp_name, str(sp_lut["reference_model"]), onset_source, I_grid, W_ref, W_lut, rel_err, onset_mask)

        frac = float(sp.get("fraction", 1.0))
        weighted_ref.append(W_ref)
        weighted_lut.append(W_lut)
        weighted_frac.append(max(0.0, frac))

        row = {
            "species": sp_name,
            "reference_model": str(sp_lut["reference_model"]),
            "resolved_rate": rate_resolved,
            "lut_file": lut_path,
            "lut_source": lut_source,
            "rel_floor": floor,
            "onset_source": onset_source,
            "onset_I_min": float(onset_I[0]),
            "onset_I_max": float(onset_I[-1]),
            "onset_points": int(np.count_nonzero(onset_mask)),
            "full_max_rel_err": full_stats["max"],
            "full_median_rel_err": full_stats["median"],
            "full_p95_rel_err": full_stats["p95"],
            "onset_max_rel_err": onset_stats["max"],
            "onset_median_rel_err": onset_stats["median"],
            "onset_p95_rel_err": onset_stats["p95"],
            "onset_max_rel_err_I": float(onset_I[onset_max_idx]),
            "notes": "; ".join(notes),
            "data_file": str(data_file),
        }
        summary_rows.append(row)

        print(
            f"[ion-lut-validate] {sp_name}: full(max/med/p95)=({full_stats['max']:.3e}/{full_stats['median']:.3e}/{full_stats['p95']:.3e}) "
            f"onset(max/med/p95)=({onset_stats['max']:.3e}/{onset_stats['median']:.3e}/{onset_stats['p95']:.3e}) "
            f"onset_I=[{onset_I[0]:.3e},{onset_I[-1]:.3e}]"
        )
        if notes:
            print(f"[ion-lut-validate] {sp_name} notes: {' | '.join(notes)}")

    if not summary_rows:
        raise RuntimeError("No species produced LUT/reference comparison results.")

    if len(weighted_ref) >= 2 and sum(weighted_frac) > 0.0:
        frac_arr = np.asarray(weighted_frac, dtype=np.float64)
        frac_arr = frac_arr / np.sum(frac_arr)
        W_air_ref = np.zeros_like(weighted_ref[0])
        W_air_lut = np.zeros_like(weighted_lut[0])
        for fj, wr, wl in zip(frac_arr, weighted_ref, weighted_lut):
            W_air_ref += fj * wr
            W_air_lut += fj * wl

        floor_air = max(float(args.rel_floor_abs), float(args.rel_floor_scale) * float(np.max(W_air_ref)))
        rel_air = np.abs(W_air_lut - W_air_ref) / np.maximum(W_air_ref, floor_air)
        onset_mask_air, onset_source_air, notes_air = _onset_mask(
            I_grid,
            W_air_ref,
            onset_i_min=args.onset_I_min,
            onset_i_max=args.onset_I_max,
            onset_w_min=args.onset_W_min,
            onset_w_max=args.onset_W_max,
            fallback_i_min=args.fallback_zoom_I_min,
            fallback_i_max=args.fallback_zoom_I_max,
        )
        if np.count_nonzero(onset_mask_air) < int(args.min_onset_points):
            notes_air.append(
                f"onset window has few points: {int(np.count_nonzero(onset_mask_air))} < min_onset_points={int(args.min_onset_points)}"
            )

        np.savez_compressed(
            outdir / "air_rate_compare.npz",
            I_grid=I_grid,
            W_air_reference=W_air_ref,
            W_air_LUT=W_air_lut,
            rel_err_air=rel_air,
            onset_mask=onset_mask_air.astype(np.uint8),
            onset_I_grid=I_grid[onset_mask_air],
            onset_W_air_reference=W_air_ref[onset_mask_air],
            onset_W_air_LUT=W_air_lut[onset_mask_air],
            onset_rel_err_air=rel_air[onset_mask_air],
            fractions=frac_arr,
        )

        _plot_species(outdir, "air", "fraction_weighted_mix", onset_source_air, I_grid, W_air_ref, W_air_lut, rel_air, onset_mask_air)

        onset_I_air = I_grid[onset_mask_air]
        onset_rel_air = rel_air[onset_mask_air]
        idx_air = int(np.argmax(onset_rel_air))
        stats_full_air = _percentiles(rel_air)
        stats_on_air = _percentiles(onset_rel_air)
        summary_rows.append(
            {
                "species": "air_mixed",
                "reference_model": "fraction_weighted_mix",
                "resolved_rate": "weighted_sum",
                "lut_file": "mixed_from_species",
                "lut_source": "derived",
                "rel_floor": floor_air,
                "onset_source": onset_source_air,
                "onset_I_min": float(onset_I_air[0]),
                "onset_I_max": float(onset_I_air[-1]),
                "onset_points": int(np.count_nonzero(onset_mask_air)),
                "full_max_rel_err": stats_full_air["max"],
                "full_median_rel_err": stats_full_air["median"],
                "full_p95_rel_err": stats_full_air["p95"],
                "onset_max_rel_err": stats_on_air["max"],
                "onset_median_rel_err": stats_on_air["median"],
                "onset_p95_rel_err": stats_on_air["p95"],
                "onset_max_rel_err_I": float(onset_I_air[idx_air]),
                "notes": "; ".join(notes_air),
                "data_file": str(outdir / "air_rate_compare.npz"),
            }
        )

    summary_json = outdir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    summary_csv = outdir / "summary.csv"
    headers = list(summary_rows[0].keys())
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[ion-lut-validate] wrote {summary_json}")
    print(f"[ion-lut-validate] wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

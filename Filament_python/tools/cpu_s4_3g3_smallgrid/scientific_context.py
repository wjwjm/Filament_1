"""Directly generated small fields with the unchanged N5 scientific settings."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


SHAPE = (384, 117, 129)
MICRO_SHAPES = ((32, 17, 19), (384, 17, 19))


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def data_root(_report=None) -> Path:
    """Resolve the external input root without retaining a private path."""
    configured = os.environ.get("CPU_S4_DATA_ROOT")
    require(configured is not None and Path(configured).is_absolute(), "CPU_S4_DATA_ROOT_ABSOLUTE_REQUIRED")
    root = Path(configured).resolve(strict=True)
    required = (
        root / "SMALLGRID_CONFIG.json",
        root / "raw" / "FROZEN_LUT_RETRIEVAL.json",
        root / "inputs" / "LUT_MANIFEST.json",
    )
    for path in required:
        require(path.is_file(), "CPU_S4_DATA_REQUIRED:" + path.name)
    return root


def frozen_tools(_report=None):
    from tools.cpu_s4_checkpoint_support import support_module
    modules = tuple(support_module(name) for name in ("g2_cases", "g1_exact"))
    return modules


def context(report, *, shape=SHAPE):
    from .candidate import unoptimized
    unoptimized()
    import numpy as np
    from KHz_filament.confio import load_all
    from KHz_filament.constants import c0
    from KHz_filament.device import USE_GPU, xp
    from KHz_filament.grids import make_axes
    from KHz_filament.hr4e5_formal_entry import build_prefix_schedule, pre0_fields_from_delta_n
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.runner import build_transverse_input_field
    from tools.cpu_p1r import cpu_p1r_optical as p1r

    inputs = data_root(report)
    shape = tuple(int(value) for value in shape)
    require(shape in (SHAPE, *MICRO_SHAPES), "FIXED_SMALLGRID_OR_DECLARED_MICRO_ONLY")
    require(not USE_GPU and xp is np, "NUMPY_CPU_ONLY")
    grid, beam, prop, ion, heat, run, raman = load_all(str(inputs / "SMALLGRID_CONFIG.json"))
    # No field or axes exists until all three dimensions have been selected.
    grid.Nt, grid.Ny, grid.Nx = shape
    require(heat.hr3b_enabled and not heat.hr3c_enabled, "ORIGINAL_HR3B_OPTICAL_ONLY")
    require(prop.linear_model == "bk_nee", "REAL_BK_NEE_REQUIRED")
    require(getattr(raman, "method", "iir") == "iir" and raman.iir_sampling == "exact_piecewise_linear" and raman.operator_integrator == "heun" and raman.operator_mode == "full_isaacs_eq27", "FROZEN_RAMAN_PATH")
    lut_rows = json.loads((inputs / "raw" / "FROZEN_LUT_RETRIEVAL.json").read_text(encoding="utf-8-sig"))
    lut = p1r._verify_lut(inputs / "inputs", {"records": {row["path"]: row for row in lut_rows}}, ion, beam)
    ion.rate_table = dict(ion.rate_table)
    ion.rate_table["cache_dir"] = str(inputs / "inputs" / "cache" / "rate_tables")
    prop.measure_performance = True
    prop.diag_bk_nee_profile = True
    full = build_longitudinal_schedule(dz=float(prop.dz), z_max=float(prop.z_max), z_start=0.0,
        focus_window_step=bool(prop.focus_window_step), focus_center_m=float(prop.focus_center_m),
        focus_halfwidth_m=float(prop.focus_halfwidth_m), dz_focus=float(prop.dz_focus))
    schedule = build_prefix_schedule(full, 8)
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    source, profile = build_transverse_input_field(axes, beam, np.complex128)
    # The original BK source's thin-lens phase, sampled directly on this grid.
    if getattr(beam, "focal_length", None):
        omega0 = 2.0 * np.pi * c0 / beam.lam0
        k0 = beam.n0 * omega0 / c0
        X, Y = np.meshgrid(axes.x, axes.y, indexing="xy")
        source *= np.exp(np.asarray(-1j * k0 * (X ** 2 + Y ** 2) / (2.0 * float(beam.focal_length)), dtype=np.complex128))
    source = np.ascontiguousarray(source, dtype=np.complex128)
    pre0 = np.zeros((8, grid.Ny, grid.Nx), dtype=np.float64, order="C")
    pre_fields = pre0_fields_from_delta_n(pre0, expected_count=8)
    source.setflags(write=False)
    for value in pre_fields.values():
        value.setflags(write=False)
    return dict(source=source, pre0=pre0, pre_fields=pre_fields, grid=grid, beam=beam, prop=prop, ion=ion,
                heat=heat, run=run, raman=raman, schedule=schedule, lut=lut,
                benchmark_input_kind="DIRECT_SMALLGRID_SOURCE_AND_FIXED_AMBIENT_PRE0", input_profile=profile)

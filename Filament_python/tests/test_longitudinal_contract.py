from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from KHz_filament.longitudinal import (
    DEPOSITION_CHANNELS,
    DepositionContract,
    GridMetadata,
    LongitudinalSchedule,
    build_deposition_contract,
    build_longitudinal_schedule,
    build_transverse_grid_metadata,
)


def test_schedule_endpoints_diff_and_replay_are_deterministic():
    kwargs = dict(
        dz=0.4,
        z_max=1.1,
        z_start=0.2,
        focus_window_step=True,
        focus_center_m=0.8,
        focus_halfwidth_m=0.25,
        dz_focus=0.1,
    )
    first = build_longitudinal_schedule(**kwargs)
    second = build_longitudinal_schedule(**kwargs)

    assert first.z_edges[0] == pytest.approx(0.2)
    assert first.z_edges[-1] == pytest.approx(1.1)
    assert first.n_intervals == len(first.dz_intervals) == len(first.intervals)
    np.testing.assert_allclose(first.dz_intervals, np.diff(first.z_edges))
    assert first.z_edges == second.z_edges
    assert first.dz_intervals == second.dz_intervals
    assert first.intervals == second.intervals

    # Midpoint focus selection is evaluated in the absolute z frame.  The
    # final interval is the un-snapped min(candidate, z_max-z) remainder.
    np.testing.assert_allclose(first.dz_intervals, [0.4, 0.1, 0.1, 0.1, 0.2])
    with pytest.raises(FrozenInstanceError):
        first.z_edges = ()


def test_schedule_validation_rejects_inconsistent_interval_metadata():
    with pytest.raises(ValueError, match="dz_intervals"):
        LongitudinalSchedule(
            z_edges=(0.0, 0.5),
            dz_intervals=(0.4,),
            intervals=(),
            z_start=0.0,
            z_end=0.5,
        )


def test_deposition_contract_is_metadata_only_and_grid_pair_is_explicit():
    axes = SimpleNamespace(
        x=np.arange(8, dtype=float),
        y=np.arange(6, dtype=float),
        dx=2.5e-5,
        dy=3.5e-5,
    )
    grid = build_transverse_grid_metadata(axes)
    assert grid.optical_grid.Nx == 8
    assert grid.optical_grid.Ny == 6
    assert grid.optical_grid.Lx == pytest.approx(8 * axes.dx)
    assert grid.optical_grid.Ly == pytest.approx(6 * axes.dy)
    assert grid.thermal_grid == grid.optical_grid
    assert grid.same_grid

    schedule = build_longitudinal_schedule(0.1, 0.35)
    contract = build_deposition_contract(schedule, axes=axes)
    assert isinstance(contract, DepositionContract)
    assert contract.channels == DEPOSITION_CHANNELS
    assert contract.n_intervals == schedule.n_intervals
    assert contract.q_shape == (6, 8)
    assert contract.payload_allocated is False
    assert contract.transverse_grid.optical_grid == contract.transverse_grid.thermal_grid
    assert all(
        not isinstance(value, np.ndarray)
        for value in (*contract.schedule.__dict__.values(), *contract.__dict__.values())
    )
    assert not any(key.startswith("q_") for key in contract.__dict__)


def test_grid_metadata_can_describe_a_future_distinct_thermal_grid():
    optical = GridMetadata(Nx=8, Ny=6, dx=1e-5, dy=2e-5, Lx=8e-5, Ly=12e-5)
    thermal = GridMetadata(Nx=4, Ny=3, dx=2e-5, dy=4e-5, Lx=8e-5, Ly=12e-5)
    axes = SimpleNamespace(x=np.arange(8), y=np.arange(6), dx=1e-5, dy=2e-5)
    derived = build_transverse_grid_metadata(axes, thermal_grid=thermal)
    assert derived.optical_grid.Nx == optical.Nx
    assert derived.optical_grid.Ny == optical.Ny
    assert derived.optical_grid.dx == pytest.approx(optical.dx)
    assert derived.optical_grid.dy == pytest.approx(optical.dy)
    assert derived.optical_grid.Lx == pytest.approx(optical.Lx)
    assert derived.optical_grid.Ly == pytest.approx(optical.Ly)
    assert derived.thermal_grid == thermal
    assert derived.remapping_required

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from KHz_filament.config import BeamConfig
from KHz_filament.config_normalize import normalize_config
from KHz_filament.constants import c0, eps0
from KHz_filament.diagnostics import intensity, pulse_energy
from KHz_filament.grids import make_axes
from KHz_filament.runner import build_transverse_input_field
from KHz_filament.utils import gaussian_beam_xy, gaussian_pulse_t, transverse_intensity_profile


ROOT = Path(__file__).resolve().parents[1]
RADIUS = 1.979e-3


def test_gaussian_profile_matches_legacy_field_squared() -> None:
    x = np.linspace(-2.0e-3, 2.0e-3, 65)
    y = np.linspace(-2.0e-3, 2.0e-3, 65)
    profile = transverse_intensity_profile(x, y, {"type": "gaussian", "radius_m": RADIUS}, RADIUS)
    legacy = gaussian_beam_xy(x, y, RADIUS) ** 2
    assert np.allclose(profile, legacy, rtol=0.0, atol=1e-14)
    assert profile[32, 32] == pytest.approx(1.0)


def test_flat_top_cosine_definition_and_symmetry() -> None:
    x = np.array([0.0, 0.89 * RADIUS, 0.95 * RADIUS, RADIUS, 1.01 * RADIUS])
    profile = transverse_intensity_profile(x, np.array([0.0]), {"type": "flat_top_cosine", "radius_m": RADIUS, "edge_start_fraction": 0.9}, RADIUS)[0]
    assert profile[0] == pytest.approx(1.0)
    assert profile[1] == pytest.approx(1.0)
    assert profile[2] == pytest.approx(0.5, abs=1e-12)
    assert profile[3] == pytest.approx(0.0, abs=1e-12)
    assert profile[4] == pytest.approx(0.0, abs=1e-12)

    xx = np.linspace(-2.1e-3, 2.1e-3, 101)
    field = transverse_intensity_profile(xx, xx, {"type": "flat_top_cosine", "radius_m": RADIUS, "edge_start_fraction": 0.9}, RADIUS)
    assert np.all(np.isfinite(field)) and np.all(field >= 0.0)
    assert np.allclose(field, field[::-1, :]) and np.allclose(field, field[:, ::-1])


def _profile_beam(profile: dict) -> BeamConfig:
    return BeamConfig(
        lam0=800e-9,
        n0=1.00027,
        w0=RADIUS,
        tau_fwhm=120e-15,
        E0_peak=0.0,
        energy_J=None,
        P0_peak=17e9,
        focal_length=0.95,
        transverse_profile=profile,
    )


def test_discrete_peak_power_and_ft90_peak_intensity_ratio() -> None:
    axes = make_axes(512, 512, 4, 8e-3, 8e-3, 0.96e-12)
    _, gaussian = build_transverse_input_field(axes, _profile_beam({"type": "gaussian", "radius_m": RADIUS}), np.complex64)
    _, ft90 = build_transverse_input_field(axes, _profile_beam({"type": "flat_top_cosine", "radius_m": RADIUS, "edge_start_fraction": 0.9}), np.complex64)
    for data in (gaussian, ft90):
        assert abs(float(data["input_peak_power_W"]) - 17e9) / 17e9 < 1e-4
        assert float(data["input_effective_area_m2"]) > 0.0
        assert 0.0 <= float(data["input_boundary_I_fraction"]) < 1e-3
    ratio = float(ft90["input_peak_intensity_W_m2"]) / float(gaussian["input_peak_intensity_W_m2"])
    assert ratio == pytest.approx(0.554, rel=0.015)


def test_gaussian_discrete_normalization_tracks_analytic_power_and_energy() -> None:
    axes = make_axes(512, 512, 4, 8e-3, 8e-3, 0.96e-12)
    beam = _profile_beam({"type": "gaussian", "radius_m": RADIUS})
    field, diagnostics = build_transverse_input_field(axes, beam, np.complex64)
    analytic_area = np.pi * RADIUS ** 2 / 2.0
    pref = 0.5 * eps0 * c0 * beam.n0
    legacy_e0 = np.sqrt(beam.P0_peak / (pref * analytic_area))
    legacy_power = pref * legacy_e0 ** 2 * float(diagnostics["input_effective_area_m2"])
    assert abs(legacy_power - beam.P0_peak) / beam.P0_peak < 1e-3
    legacy_field = (legacy_e0 * gaussian_pulse_t(axes.t, beam.tau_fwhm) * gaussian_beam_xy(axes.x, axes.y, RADIUS)[None, ...]).astype(np.complex64)
    new_energy = pulse_energy(intensity(field, beam.n0), axes.dt, axes.dx, axes.dy)
    legacy_energy = pulse_energy(intensity(legacy_field, beam.n0), axes.dt, axes.dx, axes.dy)
    assert abs(new_energy - legacy_energy) / legacy_energy < 1e-3


@pytest.mark.parametrize(
    "profile, message",
    [
        ({"type": "unknown", "radius_m": RADIUS}, "unknown beam.transverse_profile.type"),
        ({"type": "gaussian", "radius_m": 0.0}, "radius_m must be positive"),
        ({"type": "flat_top_cosine", "radius_m": RADIUS, "edge_start_fraction": 0.0}, "edge_start_fraction"),
        ({"type": "flat_top_cosine", "radius_m": RADIUS, "edge_start_fraction": 1.0}, "edge_start_fraction"),
    ],
)
def test_invalid_transverse_profile_configurations_are_rejected(profile: dict, message: str) -> None:
    raw = {"grid": {}, "beam": {"w0": RADIUS, "tau_fwhm": 120e-15, "n0": 1.00027, "E0_peak": 0.0, "energy_J": None, "P0_peak": 17e9, "transverse_profile": profile}, "ionization": {}}
    with pytest.raises(ValueError, match=message):
        normalize_config(raw)


def test_energy_and_peak_power_remain_mutually_exclusive() -> None:
    raw = {"grid": {}, "beam": {"w0": RADIUS, "tau_fwhm": 120e-15, "n0": 1.00027, "E0_peak": 0.0, "energy_J": 1e-3, "P0_peak": 17e9}, "ionization": {}}
    with pytest.raises(ValueError, match="mutually exclusive"):
        normalize_config(raw)


def test_profile_validation_configs_only_differ_in_profile() -> None:
    gaussian = json.loads((ROOT / "configs" / "profile_validation" / "gaussian_120fs.json").read_text(encoding="utf-8"))
    ft90 = json.loads((ROOT / "configs" / "profile_validation" / "flat_top_90_120fs.json").read_text(encoding="utf-8"))
    g_profile = gaussian["beam"].pop("transverse_profile")
    ft_profile = ft90["beam"].pop("transverse_profile")
    assert gaussian == ft90
    assert g_profile["type"] == "gaussian"
    assert ft_profile == {"type": "flat_top_cosine", "radius_m": RADIUS, "edge_start_fraction": 0.9}

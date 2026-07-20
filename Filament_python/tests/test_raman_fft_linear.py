from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.raman import raman_convolve_intensity_fft_linear


def _kernel(t):
    omega, gamma = 1.6e13, 1.3e13
    return ((omega * omega + gamma * gamma) / omega) * np.exp(-gamma * t) * np.sin(omega * t)


@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 1e-10), (np.float32, 1e-5)])
@pytest.mark.parametrize("case", ["40fs", "120fs", "chirped", "tail", "impulse", "constant"])
def test_fft_linear_matches_direct_causal_convolution(dtype, tolerance, case):
    dt = 0.625e-15
    t = np.arange(1024, dtype=dtype) * dtype(dt)
    x = t - dtype(300 * dt)
    if case == "40fs":
        intensity = np.exp(-4 * np.log(2) * (x / dtype(40e-15)) ** 2)
    elif case == "120fs":
        intensity = np.exp(-4 * np.log(2) * (x / dtype(120e-15)) ** 2)
    elif case == "chirped":
        intensity = np.exp(-4 * np.log(2) * (x / dtype(120e-15)) ** 2) * (1 + .25 * x / dtype(120e-15))
    elif case == "tail":
        intensity = np.exp(-4 * np.log(2) * (x / dtype(120e-15)) ** 2) + .2 * np.exp(-((x - dtype(120e-15)) / dtype(80e-15)) ** 2)
    elif case == "impulse":
        intensity = np.zeros_like(t); intensity[300] = 1 / dtype(dt)
    else:
        intensity = np.ones_like(t)
    kernel = _kernel(t).astype(dtype)
    expected = np.convolve(intensity, kernel, mode="full")[: len(t)] * dt
    actual = np.asarray(raman_convolve_intensity_fft_linear(intensity[:, None, None], kernel, dt=dt))[:, 0, 0]
    assert np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-30) < tolerance


def test_fft_linear_has_no_wraparound_before_a_late_impulse():
    dt = 1e-15
    t = np.arange(256) * dt
    intensity = np.zeros_like(t); intensity[-4] = 1.0
    out = np.asarray(raman_convolve_intensity_fft_linear(intensity[:, None, None], _kernel(t), dt=dt))[:, 0, 0]
    assert np.allclose(out[:-4], 0.0, atol=1e-14)

from __future__ import annotations

import copy
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.config_normalize import normalize_config
from KHz_filament.raman import make_raman_kernel


def _isaacs():
    return {"raman": {"model": "isaacs_rot_sinexp", "n_R": 2.3e-23, "omega_R": 1.6e13, "Gamma_R": 1.3e13}}


def test_isaacs_strict_config_accepts_explicit_parameters():
    cfg = normalize_config(_isaacs())
    assert cfg["raman"]["n_R"] == 2.3e-23
    assert cfg["raman"]["omega_R"] == 1.6e13


@pytest.mark.parametrize("field,value", [("f_R", 0.15), ("T_R", 8.4e-12), ("T2", 80e-12)])
def test_isaacs_strict_config_rejects_legacy_parameters(field, value):
    cfg = _isaacs()
    cfg["raman"][field] = value
    with pytest.raises(ValueError, match="f_R/T_R/T2/Omega_R/tau2"):
        normalize_config(cfg)


def test_legacy_rot_sinexp_kernel_remains_reproducible():
    import numpy as np
    t = np.arange(64) * 1e-15
    legacy = {"model": "rot_sinexp", "omega_R": 1.6e13, "Gamma_R": 1.3e13, "T_R": 8.4e-12, "T2": 80e-12}
    before = make_raman_kernel(t, copy.deepcopy(legacy))
    after = make_raman_kernel(t, legacy)
    assert np.array_equal(np.asarray(before), np.asarray(after))

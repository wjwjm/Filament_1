from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from KHz_filament.config import IonizationConfig, PropagationConfig, RamanConfig
from KHz_filament.confio import load_all
from KHz_filament.constants import N0_air, Ui_N2, c0
from KHz_filament.grids import make_axes
from KHz_filament.propagate import propagate_one_pulse


ROOT = Path(__file__).resolve().parents[1]


def load_prepare():
    path = ROOT / "tools" / "prepare_raman_off_kerr085_job.py"
    spec = importlib.util.spec_from_file_location("prepare_raman_off_kerr085_job", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_is_exactly_one_field_and_loadable(tmp_path):
    module = load_prepare()
    base = json.loads(module.BASE.read_text(encoding="utf-8"))
    assert module.executed_parent_crlf_sha256(module.BASE) == module.EXECUTED_PARENT_SHA256
    derived, diff = module.build(base)
    assert diff == [{
        "path": "beam.n2_air",
        "raman_phase_off": 7.8e-24,
        "raman_off_kerr085": 6.63e-24,
    }]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(derived), encoding="utf-8")
    _, beam, prop, _, _, _, raman = load_all(str(path))
    assert beam.n2_air == 6.63e-24
    assert prop.use_electronic_kerr is True
    assert prop.use_raman_phase is False
    assert prop.use_raman_absorption is True
    assert raman.absorption is True and raman.absorption_model == "conv_deriv"


def test_small_grid_executes_candidate_semantics():
    nx = ny = 16
    nt = 64
    axes = make_axes(nx, ny, nt, Lx=1.6e-3, Ly=1.6e-3, Twin=400e-15)
    lam0, n0 = 800e-9, 1.00027
    omega0 = 2.0 * np.pi * c0 / lam0
    k0 = n0 * omega0 / c0
    t, x, y = np.asarray(axes.t), np.asarray(axes.x), np.asarray(axes.y)
    env_t = np.exp(-2.0 * np.log(2.0) * (t / (120e-15 / 2.0)) ** 2)
    env_x = np.exp(-(x / 0.35e-3) ** 2)
    env_y = np.exp(-(y / 0.35e-3) ** 2)
    field = (1e9 * env_t[:, None, None] * env_y[None, :, None] * env_x[None, None, :]).astype(np.complex64)
    prop = PropagationConfig(
        linear_model="bk_nee", use_self_steepening=True, use_electronic_kerr=True,
        use_raman_phase=False, use_raman_absorption=True, use_plasma_phase=False,
        use_ionization_loss=False, use_ionization_solver=False, focus_window_step=False,
    )
    raman = RamanConfig(
        enabled=True, model="rot_sinexp", absorption=True,
        absorption_model="conv_deriv", omega_R=1.6e13, Gamma_R=1.3e13,
        n_R=2.3e-23, T2=80e-12, T_R=8.4e-12,
    )
    _, _, diag = propagate_one_pulse(
        field, kperp2=axes.kperp2, k0=k0, omega0=omega0, dz=2e-4, z_max=4e-4,
        n0=n0, n2=6.63e-24, Ui=Ui_N2, N0=N0_air, ion_conf=IonizationConfig(species=None),
        dn_gas=np.zeros((ny, nx), dtype=np.float32), dt=float(axes.dt), axes=axes,
        prop_conf=prop, raman_conf=raman, record_every_z=1,
    )
    assert diag["n2_elec_used"] == 6.63e-24
    assert np.allclose(diag["delta_n_rot_applied_max_z"], 0.0, atol=1e-30)
    assert np.max(diag["IR_abs_max_z"]) > 0.0
    assert np.max(diag["alpha_R_applied_max_z"]) > 0.0
    ratio = np.asarray(diag["delta_n_elec_max_z"]) / np.asarray(diag["I_max_z"])
    assert np.allclose(ratio, 6.63e-24, rtol=2e-5, atol=0.0)


def test_submission_scripts_do_not_pin_a_node():
    sbatch = (ROOT / "tools" / "raman_off_kerr085_full.sbatch").read_text(encoding="utf-8")
    submit = (ROOT / "tools" / "submit_raman_off_kerr085_job.sh").read_text(encoding="utf-8")
    assert "--nodelist" not in sbatch and "--nodelist" not in submit
    assert "--chdir=" in submit and "--output=" in submit and "--error=" in submit

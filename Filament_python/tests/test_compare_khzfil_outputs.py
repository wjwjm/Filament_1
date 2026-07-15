from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from compare_khzfil_outputs import generate_comparison_figures, load_result_file


def _data(scale: float) -> dict[str, np.ndarray]:
    z = np.array([0.0, 0.01, 0.02])
    return {"z_axis": z, "I_max_z": scale * np.array([1e12, 2e12, 1e12]), "I_onaxis_max_z": scale * np.array([1e11, 2e11, 1e11]), "rho_onaxis_max_z": scale * np.array([1e18, 2e18, 1e18]), "rho_max_z": scale * np.array([2e18, 3e18, 2e18]), "w_mom_z": np.array([2e-3, 1e-3, 2e-3]), "U_z": np.array([1e-3, 0.99e-3, 0.98e-3]), "fwhm_plasma_z": np.array([100e-6, 90e-6, 100e-6]), "fwhm_fluence_z": np.array([120e-6, 110e-6, 120e-6])}


def test_compare_npz_and_mat_without_resampling(tmp_path: Path) -> None:
    npz = tmp_path / "40.npz"; mat = tmp_path / "120.mat"
    np.savez(npz, **_data(1.0)); scipy.io.savemat(mat, _data(2.0))
    summary = generate_comparison_figures([("40fs", "40 fs", load_result_file(npz)), ("120fs", "120 fs", load_result_file(mat))], tmp_path / "comparison", list(_data(1.0).keys())[1:], dpi=60)
    assert "comparison_overview.png" in summary["generated_figures"]
    assert (tmp_path / "comparison" / "comparison_metrics.csv").is_file()
    assert not summary["skipped_fields"]


def test_missing_field_is_skipped(tmp_path: Path) -> None:
    first, second = _data(1.0), _data(2.0); second.pop("rho_max_z")
    summary = generate_comparison_figures([("40fs", "40 fs", first), ("120fs", "120 fs", second)], tmp_path, ["rho_max_z"], dpi=60)
    assert "rho_max_z" in summary["skipped_fields"]


def test_rho_max_near_focus_plot_uses_linear_1e16_cm3_window(tmp_path: Path) -> None:
    z = np.array([0.7, 0.75, 0.95, 1.15, 1.2])
    first = {"z_axis": z, "rho_max_z": np.array([1e20, 2e21, 4e22, 2e21, 1e20])}
    second = {"z_axis": z.copy(), "rho_max_z": np.array([2e20, 4e21, 6e22, 4e21, 2e20])}
    summary = generate_comparison_figures(
        [("gaussian", "Gaussian, 120 fs", first), ("ft90", "FT90, 120 fs", second)],
        tmp_path, ["rho_max_z"], dpi=60,
        rho_max_plot={"geometric_focus_m": 0.95, "half_window_m": 0.2},
    )
    assert (tmp_path / "compare_rho_max_z.png").is_file()
    assert summary["rho_max_plot"] == {
        "geometric_focus_m": 0.95,
        "half_window_m": 0.2,
        "density_unit": "1e16 cm^-3",
    }

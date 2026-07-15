from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


FILAMENT_PYTHON = Path(__file__).resolve().parents[1]
if str(FILAMENT_PYTHON) not in sys.path:
    sys.path.insert(0, str(FILAMENT_PYTHON))

from plot_khzfil_out import FIGURE_SPECS, generate_figures


def _write_synthetic_npz(path: Path, rho_shape: str = "zt", **overrides: np.ndarray) -> Path:
    z = np.linspace(0.0, 0.04, 5)
    t = np.linspace(-50e-15, 50e-15, 4)
    rho_zt = np.arange(z.size * t.size, dtype=float).reshape(z.size, t.size) + 1e18
    values: dict[str, np.ndarray] = {
        "z_axis": z,
        "t_axis": t,
        "I_max_z": np.array([1e12, 2e12, 5e12, 3e12, 2e12]),
        "I_onaxis_max_z": np.array([0.9e12, 1.8e12, 4e12, 2.5e12, 1.5e12]),
        "I_center_t0_z": np.array([0.8e12, 1.5e12, 3e12, 2e12, 1e12]),
        "rho_onaxis_max_z": np.array([1e18, 2e18, 5e18, 3e18, 2e18]),
        "rho_max_z": np.array([2e18, 3e18, 6e18, 4e18, 3e18]),
        "w_mom_z": np.array([1.5e-3, 1.0e-3, 0.5e-3, 0.8e-3, 1.2e-3]),
        "U_z": np.array([1.0e-3, 0.995e-3, 0.99e-3, 0.985e-3, 0.98e-3]),
        "fwhm_plasma_z": np.array([120e-6, 110e-6, 100e-6, 105e-6, 115e-6]),
        "fwhm_fluence_z": np.array([150e-6, 140e-6, 130e-6, 135e-6, 145e-6]),
        "rho_onaxis_t_z": rho_zt if rho_shape == "zt" else rho_zt.T,
    }
    values.update(overrides)
    np.savez(path, **values)
    return path


def test_generate_all_figures_and_summary(tmp_path: Path) -> None:
    npz_path = _write_synthetic_npz(tmp_path / "run.npz")
    figure_dir = tmp_path / "figures"

    summary = generate_figures(npz_path, figure_dir, z_shift_cm=-20.0, dpi=80, metadata={"stage_id": "stage1", "case_id": "40fs", "pulse_width_fs": 40.0})

    assert set(summary["generated_figures"]) == set(FIGURE_SPECS.values())
    for name in FIGURE_SPECS.values():
        assert (figure_dir / name).stat().st_size > 0
    saved = json.loads((figure_dir / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["I_max_peak_W_m2"] == pytest.approx(5e12)
    assert saved["metrics"]["z_I_max_peak_m"] == pytest.approx(0.02)
    assert saved["metrics"]["U_drift_pct"] == pytest.approx(-2.0)
    assert saved["z_shift_cm"] == -20.0
    assert saved["stage_id"] == "stage1" and saved["case_id"] == "40fs"
    assert saved["quality_observations"]["z_strictly_increasing"] is True


def test_missing_optional_fields_skips_only_affected_figures(tmp_path: Path) -> None:
    npz_path = tmp_path / "partial.npz"
    np.savez(npz_path, z_axis=np.arange(4.0), I_max_z=np.array([1.0, 2.0, 3.0, 2.0]), U_z=np.ones(4))

    summary = generate_figures(npz_path, tmp_path / "figures", dpi=80)

    assert "01_intensity_vs_z.png" in summary["generated_figures"]
    assert "04_energy_vs_z.png" in summary["generated_figures"]
    assert "02_plasma_density_vs_z.png" in summary["skipped_figures"]
    assert "06_rho_onaxis_t_z.png" in summary["skipped_figures"]


def test_missing_z_axis_or_mismatched_z_series_fails_clearly(tmp_path: Path) -> None:
    missing_z = tmp_path / "missing_z.npz"
    np.savez(missing_z, I_max_z=np.ones(3))
    with pytest.raises(ValueError, match="z_axis"):
        generate_figures(missing_z, tmp_path / "figures_missing")

    mismatch = tmp_path / "mismatch.npz"
    np.savez(mismatch, z_axis=np.arange(4.0), I_max_z=np.ones(3))
    with pytest.raises(ValueError, match="I_max_z length .*z_axis length"):
        generate_figures(mismatch, tmp_path / "figures_mismatch")


@pytest.mark.parametrize("rho_shape", ["zt", "tz"])
def test_rho_onaxis_time_map_accepts_both_axis_orders(tmp_path: Path, rho_shape: str) -> None:
    npz_path = _write_synthetic_npz(tmp_path / f"{rho_shape}.npz", rho_shape=rho_shape)
    summary = generate_figures(npz_path, tmp_path / f"figures_{rho_shape}", selected_figures="rho_tz", dpi=80)

    assert summary["generated_figures"] == ["06_rho_onaxis_t_z.png"]


def test_plot_failure_keeps_npz_for_later_postprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import plot_khzfil_out
    import test_run

    npz_path = _write_synthetic_npz(tmp_path / "source.npz")

    def fail_generate(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("simulated plotting failure")

    monkeypatch.setattr(plot_khzfil_out, "generate_figures", fail_generate)
    args = SimpleNamespace(
        out=str(npz_path),
        fig_dir=str(tmp_path / "figures"),
        no_plots=False,
        fig_select="all",
        z_shift_cm=0.0,
        fig_dpi=80,
        mat_dir=str(tmp_path / "matlab保存数据"),
        mat_name=None,
        remove_npz=True,
    )

    with pytest.raises(RuntimeError, match="simulated plotting failure"):
        test_run._postprocess_output(args)
    assert npz_path.exists()


def test_successful_postprocessing_writes_mat_png_json_then_removes_npz(tmp_path: Path) -> None:
    import test_run

    npz_path = _write_synthetic_npz(tmp_path / "source.npz")
    figure_dir = tmp_path / "figures" / "run_001"
    mat_dir = tmp_path / "matlab保存数据"
    args = SimpleNamespace(
        out=str(npz_path),
        fig_dir=str(figure_dir),
        no_plots=False,
        fig_select="all",
        z_shift_cm=0.0,
        fig_dpi=80,
        mat_dir=str(mat_dir),
        mat_name=None,
        remove_npz=True,
    )

    test_run._postprocess_output(args)

    assert not npz_path.exists()
    assert (mat_dir / "source.mat").stat().st_size > 0
    assert (figure_dir / "diagnostic_summary.json").stat().st_size > 0
    assert all((figure_dir / name).is_file() for name in FIGURE_SPECS.values())


def test_mat_conversion_failure_keeps_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npz2mat
    import test_run

    npz_path = _write_synthetic_npz(tmp_path / "source.npz")

    def fail_conversion(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("simulated MAT conversion failure")

    monkeypatch.setattr(npz2mat, "convert_npz_to_mat", fail_conversion)
    args = SimpleNamespace(
        out=str(npz_path),
        fig_dir=None,
        no_plots=True,
        fig_select="all",
        z_shift_cm=0.0,
        fig_dpi=80,
        mat_dir=str(tmp_path / "matlab保存数据"),
        mat_name=None,
        remove_npz=True,
    )

    with pytest.raises(RuntimeError, match="simulated MAT conversion failure"):
        test_run._postprocess_output(args)
    assert npz_path.exists()

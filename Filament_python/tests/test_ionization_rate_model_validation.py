from __future__ import annotations

import copy
import math
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from KHz_filament.confio import load_all
from archive_ionization_rate_model_validation import classify_rate_model_validation
from validate_ionization_density_response import cumulative_reference_batch, threshold_intensity_log_interpolation
from validate_ionization_rate_models import (
    DEFAULT_CONFIG,
    RATE_MODELS,
    _production_species,
    lut_error_statistics,
    make_rate_evaluator,
    model_species_parameters,
    repo_relative,
)


def test_reference_and_lut_evaluators_call_with_runtime_parameters(tmp_path):
    beam, ion, species = _production_species(DEFAULT_CONFIG)
    ion_small = copy.deepcopy(ion)
    ion_small.rate_table = dict(ion.rate_table)
    ion_small.rate_table.update({"cache_dir": str(tmp_path / "lut"), "n_samples": 32, "ref_cycle_avg_samples": 16, "save_tables": False})
    species_by_name = {item["name"]: item for item in species}
    for family, name in (("popruzhenko", "N2"), ("talebpour", "O2")):
        reference, _ = make_rate_evaluator(ion_small, beam, species_by_name[name], RATE_MODELS[family][0], cache_dir=tmp_path / "lut")
        lut, _ = make_rate_evaluator(ion_small, beam, species_by_name[name], RATE_MODELS[family][1], cache_dir=tmp_path / "lut")
        I = np.asarray([1e16, 1e17, 1e18])
        assert np.all(np.isfinite(reference(I)))
        assert np.all(np.isfinite(lut(I)))


def test_n2_o2_runtime_parameter_resolution():
    _beam, _ion, species = _production_species(DEFAULT_CONFIG)
    by_name = {item["name"]: item for item in species}
    assert model_species_parameters(by_name["N2"], "popruzhenko")["Ip_eV"] == 15.6
    assert model_species_parameters(by_name["O2"], "popruzhenko")["Z"] == 1
    o2_tal = model_species_parameters(by_name["O2"], "talebpour")
    assert o2_tal["Ip_eV_eff"] == 12.55
    assert o2_tal["Zeff"] == 0.53


def test_lut_error_statistics_masks_zero_rate_denominator():
    stats = lut_error_statistics(np.asarray([1e15, 1e16, 1e17]), np.asarray([0.0, 2.0, 4.0]), np.asarray([1.0, 2.1, 4.4]), meaningful_floor_s_1=1.0)
    assert stats["full_scan"]["sample_count"] == 2
    assert math.isclose(stats["full_scan"]["max_relative_error"], 0.1)


def test_fixed_density_threshold_interpolation_and_not_crossed():
    I = np.asarray([1e16, 1e17, 1e18])
    rho = np.asarray([1e18, 1e20, 1e22])
    crossed = threshold_intensity_log_interpolation(I, rho, 1e21)
    assert crossed["status"] == "crossed_interpolated"
    assert math.isclose(crossed["I_threshold_W_m2"], 10 ** 17.5)
    assert threshold_intensity_log_interpolation(I, rho, 1e23)["status"] == "not_crossed"


def test_species_density_addition_and_both_ft90_widths_read():
    t = np.asarray([0.0, 1.0])
    n2 = cumulative_reference_batch(np.asarray([[0.0, 1.0]]), t, 0.8)[0, -1]
    o2 = cumulative_reference_batch(np.asarray([[0.0, 2.0]]), t, 0.2)[0, -1]
    assert math.isclose(n2 + o2, cumulative_reference_batch(np.asarray([[0.0, 1.0]]), t, 0.8)[0, -1] + cumulative_reference_batch(np.asarray([[0.0, 2.0]]), t, 0.2)[0, -1])
    paths = [ROOT / "configs" / "profile_validation" / name for name in ("flat_top_90_40fs.json", "flat_top_90_120fs.json")]
    assert [round(load_all(str(path))[1].tau_fwhm * 1e15) for path in paths] == [40, 120]


def test_repository_relative_paths_and_synthetic_classification():
    assert not pathlib.PurePath(repo_relative(DEFAULT_CONFIG)).is_absolute()
    lut_pass = [{"scope": "relevant_interval", "lut_pass": True}]
    supported_rows = [
        {"density_threshold_m3": 1e21, "I_threshold_ratio_pop_over_tal": 0.85, "popruzhenko_status": "crossed_interpolated", "talebpour_status": "crossed_interpolated", "tau_fwhm_fs": 40.0},
        {"density_threshold_m3": 1e21, "I_threshold_ratio_pop_over_tal": 0.86, "popruzhenko_status": "crossed_interpolated", "talebpour_status": "crossed_interpolated", "tau_fwhm_fs": 120.0},
    ]
    assert classify_rate_model_validation(lut_pass, supported_rows)[0] == "supported"
    not_supported_rows = [{**supported_rows[0], "I_threshold_ratio_pop_over_tal": 1.01}, {**supported_rows[1], "I_threshold_ratio_pop_over_tal": 0.99}]
    assert classify_rate_model_validation(lut_pass, not_supported_rows)[0] == "not_supported"
    assert classify_rate_model_validation([{"scope": "relevant_interval", "lut_pass": False}], supported_rows)[0] == "inconclusive"

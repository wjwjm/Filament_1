from __future__ import annotations

import pytest

from KHz_filament.hr4e2c_adjudication import adjudicate_observable


def test_delta_q_and_exact_zero_velocity_are_reported_without_ratio_division():
    row = adjudicate_observable("max_abs_v_m_s", (0.0, 0.0, 0.0), (4.0e-5, 2.2e-5, 1.4e-5))
    assert row["Delta_Q20"] == pytest.approx(4.0e-5)
    assert row["Delta_Q10"] == pytest.approx(2.2e-5)
    assert row["Delta_Q5"] == pytest.approx(1.4e-5)
    assert row["initial_status"] == "INIT_NEAR_ZERO_NA"
    assert row["evolution_trend_status"] == "PASS"


def test_near_zero_centroid_evolution_is_not_forced_to_observed_order():
    row = adjudicate_observable("xc_m", (0.0, 0.0, 0.0), (1.0e-13, 2.0e-13, 1.0e-13))
    assert row["evolution_trend_status"] == "N/A_NEAR_ZERO"
    assert row["p_evol"] is None


def test_mapping_dominated_category_requires_nonmonotonic_initial_and_monotonic_evolution():
    row = adjudicate_observable("sigma_x_m", (80e-6, 80.1e-6, 80.7e-6), (100e-6, 100.5e-6, 101.3e-6))
    assert row["initial_status"] == "INIT_NONMONOTONIC_WITHIN_TOLERANCE"
    assert row["evolution_trend_status"] == "PASS"
    assert row["adjudication_category"] == "WARNING_MAPPING_DOMINATED"


def test_hydro_evolution_dominated_category_requires_monotonic_initial_and_nonmonotonic_evolution():
    row = adjudicate_observable("sigma_y_m", (80e-6, 80.4e-6, 80.5e-6), (100e-6, 100.2e-6, 101.0e-6))
    assert row["initial_status"] == "INIT_MONOTONIC"
    assert row["evolution_trend_status"] == "WARNING"
    assert row["adjudication_category"] == "WARNING_HYDRO_EVOLUTION_DOMINATED"


def test_mixed_and_numerically_negligible_categories_are_distinguished():
    mixed = adjudicate_observable("min_delta_n", (-10e-6, -10.1e-6, -10.7e-6), (-8e-6, -8.2e-6, -9.0e-6))
    negligible = adjudicate_observable("sigma_x_m", (80e-6, 80.0000001e-6, 80.0000003e-6), (100e-6, 100.0000001e-6, 100.0000003e-6))
    assert mixed["adjudication_category"] == "WARNING_MIXED_OR_AMBIGUOUS"
    assert negligible["adjudication_category"] == "WARNING_NUMERICALLY_NEGLIGIBLE"


def test_relative_evolution_fraction_is_anchored_to_physical_q_not_near_zero_delta_q():
    row = adjudicate_observable("M0_negative_index_m2", (1.0e-13, 1.01e-13, 1.02e-13), (1.0e-13, 1.01e-13, 1.020001e-13))
    assert row["fraction_of_tolerance_evol"] < 1.0


def test_frozen_width_tolerance_is_preserved_in_every_adjudication_row():
    row = adjudicate_observable("sigma_x_m", (80e-6, 80.2e-6, 80.3e-6), (100e-6, 100.2e-6, 100.3e-6))
    assert row["frozen_10_vs_5_tolerance"] == pytest.approx(0.01)

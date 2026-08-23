from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_audit():
    path = ROOT / "tools" / "audit_isaacs_raman_reclosure.py"
    spec = importlib.util.spec_from_file_location("audit_isaacs_raman_reclosure", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_iir_equivalent_kernel_has_unit_dc_gain():
    audit = load_audit()
    weights = audit.exact_piecewise_linear_kernel(audit.DT_PRODUCTION, 8192)
    assert np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=2e-13)


def test_production_iir_matches_direct_120fs_gaussian():
    pytest.importorskip("scipy")
    audit = load_audit()
    t = (np.arange(audit.NT_PRODUCTION) - audit.NT_PRODUCTION // 2) * audit.DT_PRODUCTION
    intensity = audit.gaussian_intensity(t)
    reference = audit.continuous_response(t)
    response = audit.production_iir(intensity)
    error = np.max(np.abs(response - reference)) / np.max(reference)
    assert error < 5e-4


def test_rotational_eq27_rhs_contains_full_product_derivative():
    audit = load_audit()
    t = (np.arange(audit.NT_PRODUCTION) - audit.NT_PRODUCTION // 2) * audit.DT_PRODUCTION
    intensity = audit.gaussian_intensity(t)
    response = audit.production_iir(intensity)
    amplitude = np.sqrt(2.0 * intensity / (audit.eps0 * audit.c0 * audit.N0))
    field = amplitude * np.exp(1j * 2.5e27 * t * t)
    current = audit.stage_rhs(field, audit.DT_PRODUCTION)
    direct = audit.direct_rotational_rhs(field, response, audit.DT_PRODUCTION)
    omitted = audit.incomplete_rotational_rhs(field, response, audit.DT_PRODUCTION)
    assert audit.relative_l2(current, direct) < 1e-12
    assert audit.relative_l2(omitted, direct) > 1e-3


def test_total_eq27_boundary_detects_scalar_electronic_approximation():
    audit = load_audit()
    t = (np.arange(audit.NT_PRODUCTION) - audit.NT_PRODUCTION // 2) * audit.DT_PRODUCTION
    intensity = audit.gaussian_intensity(t)
    amplitude = np.sqrt(2.0 * intensity / (audit.eps0 * audit.c0 * audit.N0))
    field = amplitude * np.exp(1j * 2.5e27 * t * t)
    full = audit.full_electronic_rhs(field, intensity, audit.DT_PRODUCTION)
    scalar = audit.scalar_split_electronic_rhs(field, intensity, audit.DT_PRODUCTION)
    assert audit.relative_l2(scalar, full) > 1e-2

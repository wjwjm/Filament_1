from __future__ import annotations

import csv
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from finalize_isaacs_raman_closure import (
    MetricSchemaError,
    build_numeric_gates,
    read_metric,
    threshold_gate,
)


def test_threshold_gate_fails_large_error():
    assert threshold_gate(.049, 1e-10)["status"] == "failed"


def test_threshold_gate_passes_small_error():
    assert threshold_gate(.005, .01)["status"] == "passed"


def test_threshold_gate_marks_nan_inconclusive():
    assert threshold_gate(math.nan, .01)["status"] == "inconclusive"


def test_missing_file_is_inconclusive_metric(tmp_path):
    result = read_metric(tmp_path / "missing.csv", "error")
    assert result.value is None and result.reason == "missing_file"


def test_incorrect_column_name_is_rejected(tmp_path):
    path = tmp_path / "metric.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual_error"])
        writer.writeheader(); writer.writerow({"actual_error": 0.0})
    with pytest.raises(MetricSchemaError, match="missing required columns"):
        read_metric(path, "wrong_error")


def test_fft_metric_is_not_reused_as_eq10_eq11_metric(tmp_path):
    with (tmp_path / "raman_fft_direct_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["dtype", "relative_linf_error"])
        writer.writeheader()
        writer.writerow({"dtype": "float64", "relative_linf_error": 0.049})
    with (tmp_path / "eq10_eq11_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["direct_vs_eq11_error"])
        writer.writeheader()
        writer.writerow({"direct_vs_eq11_error": 0.005})

    gates = build_numeric_gates(tmp_path)
    assert gates["fft_linear_convolution_gate"]["status"] == "failed"
    assert gates["eq11_analytic_recovery_gate"]["status"] == "passed"

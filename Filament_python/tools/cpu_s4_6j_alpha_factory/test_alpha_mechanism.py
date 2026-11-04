"""Bounded synthetic checks for the frozen alpha mechanism comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import statistics

from tools.cpu_s4_6j_alpha_factory.alpha_capture import AlphaCaptureStore
from tools.cpu_s4_6j_alpha_factory.alpha_mechanism import (
    EXPECTED_LABELS,
    AlphaMechanism,
    AlphaMechanismError,
    _same_data,
    run,
)
from tools.cpu_s4_6j_alpha_factory.alpha_pool import AlphaPool
from tools.cpu_s4_6j_alpha_factory.test_alpha_capture import (
    SHAPE,
    _assert_ieee_equal,
    _populate,
    _source,
)


def test_compiled_reference_and_p1_are_exact_and_source_bound() -> None:
    mechanism = AlphaMechanism()
    binding = mechanism.verify_binding()
    assert binding["inverse_proof"]["replacement_count"] == 4
    assert binding["inverse_proof"]["inverse_ast_equal"] is True
    assert binding["pool_methods_bound"] is True

    source = _source(0)
    intensity = np.ascontiguousarray(np.abs(source.copy()) + 2.0)
    candidate_source = source.copy()
    candidate_intensity = intensity.copy()
    pool = AlphaPool(1)
    try:
        with np.errstate(all="ignore"):
            ref_clean, ref_alpha = mechanism.call(
                "REF", source, intensity, 0.5, None, interval_index=0
            )
            p1_clean, p1_alpha = mechanism.call(
                "P1", candidate_source, candidate_intensity, 0.5, pool, interval_index=0
            )
        assert _same_data(p1_clean, ref_clean)
        assert _same_data(p1_alpha, ref_alpha)
        _assert_ieee_equal(source, _source(0))
        _assert_ieee_equal(candidate_source, _source(0))
        _assert_ieee_equal(candidate_intensity, intensity)
        rows = mechanism.drain(pool, interval_index=0)
        assert tuple(row["label"] for row in rows) == EXPECTED_LABELS
        assert [row["context"] for row in rows] == [
            "clean_source",
            "ratio",
            "ratio",
            "clean_alpha",
            "clip_alpha",
        ]
    finally:
        pool.close()


def test_pool_instance_method_binding_tamper_is_rejected() -> None:
    mechanism = AlphaMechanism()
    pool = AlphaPool(1)
    original = pool.run_round
    try:
        mechanism.verify_pool(pool)
        pool.run_round = lambda *args, **kwargs: None  # type: ignore[method-assign]
        with pytest.raises(AlphaMechanismError, match="INSTANCE_METHOD"):
            mechanism.verify_pool(pool)
        pool.run_round = original  # type: ignore[method-assign]
        mechanism.verify_pool(pool)
    finally:
        pool.close()


def test_nonexact_result_is_rejected_and_partial_evidence_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "capture"
    store = AlphaCaptureStore(capture_root, shape=SHAPE)
    _populate(store)
    store.finish(binding={"execution_commit": "synthetic"})

    original_call = AlphaMechanism.call

    def nonexact_call(self, config, source, intensity, I_floor, pool, interval_index):
        cleaned, alpha = original_call(
            self, config, source, intensity, I_floor, pool, interval_index
        )
        broken = np.array(cleaned, copy=True, order="C")
        broken.flat[0] = np.nextafter(broken.flat[0], np.inf)
        return broken, alpha

    monkeypatch.setattr(AlphaMechanism, "call", nonexact_call)
    output = tmp_path / "mechanism" / "RESULT.json"
    with pytest.raises(AlphaMechanismError, match="SCIENTIFIC_NONEXACT"):
        run(
            capture_root,
            output,
            shape=SHAPE,
            execution_commit="synthetic",
            local_engineering=True,
        )
    assert (output.parent / "MECHANISM_PARTIAL.json").is_file()
    assert (output.parent / "MECHANISM_POOLS.json").is_file()
    assert list(output.parent.glob("NONEXACT_*_cleaned.npy"))
    assert list(output.parent.glob("NONEXACT_*_alpha.npy"))


def test_exact_gate_rejects_finite_and_signed_zero_differences() -> None:
    expected = np.zeros((5, 2, 3), dtype=np.float64)
    changed = expected.copy()
    changed.flat[0] = np.nextafter(0.0, 1.0)
    assert not _same_data(changed, expected)
    signed = expected.copy()
    signed.flat[0] = -0.0
    assert not _same_data(signed, expected)


def test_complete_protocol_recomputes_each_configuration_median(tmp_path):
    store = AlphaCaptureStore(tmp_path / "capture", shape=SHAPE)
    _populate(store)
    store.finish(binding={"execution_commit":"synthetic"})
    result = run(tmp_path/"capture", tmp_path/"mechanism/RESULT.json",
        shape=SHAPE, execution_commit="synthetic", local_engineering=True)
    assert result["inputs"] == 3
    assert result["formal_calls"] == 90 and result["warmup_calls"] == 36
    assert len(result["blocks"]) == 6
    assert result["formal14_allowed"] is False
    for block in result["blocks"]:
        for config in ("REF","P1","P24"):
            rows = [r["wall_s"] for r in result["rows"] if not r["warmup"]
                and r["group"] == block["group"] and r["key"] == block["key"] and r["config"] == config]
            assert len(rows) == 5
            assert block["medians_s"][config] == statistics.median(rows)
        assert block["p24_gain"] == 1-block["medians_s"]["P24"]/block["medians_s"]["REF"]


def test_old_output_mutation_is_detected(tmp_path, monkeypatch):
    store = AlphaCaptureStore(tmp_path / "capture", shape=SHAPE)
    _populate(store)
    store.finish(binding={"execution_commit":"synthetic"})
    original = AlphaMechanism.call
    previous = []
    def mutate_old(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if previous:
            previous[0][0].flat[1] = 123.0
        previous[:] = [value]
        return value
    monkeypatch.setattr(AlphaMechanism, "call", mutate_old)
    with pytest.raises(AlphaMechanismError, match="SCIENTIFIC_NONEXACT"):
        run(tmp_path/"capture", tmp_path/"mechanism/RESULT.json",
            shape=SHAPE, execution_commit="synthetic", local_engineering=True)

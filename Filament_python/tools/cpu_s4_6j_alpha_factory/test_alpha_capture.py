"""Lightweight synthetic tests for the S4-6J alpha capture contract.

The fixture exercises the complete eight-chain callback order on a small
non-divisible axis-0 shape.  It never starts a propagation or writes outside
the pytest temporary directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import json
import os
import subprocess
import sys

from tools.cpu_s4_6j_alpha_factory.alpha_capture import (
    AlphaCaptureError,
    AlphaCaptureStore,
    CAPTURE_CONTEXTS,
    CAPTURE_INTERVALS,
    STATE_FIELDS,
    load_capture,
    load_state_arrays,
)
from tools.cpu_s4_6j_alpha_factory.alpha_hook import (
    ALPHA_CLEAN_RESULT_HELPER,
    ALPHA_CLEAN_SOURCE_HELPER,
    ALPHA_CLIP_HELPER,
    ALPHA_RATIO_HELPER,
    ALPHA_SOURCE_SITE,
)


SHAPE = (5, 2, 3)

def test_capture_binding_is_stable_across_hash_seed_processes():
    script = (
        "import json; from tools.cpu_s4_6j_alpha_factory.alpha_capture import capture_source_binding; "
        "print(json.dumps(capture_source_binding(),sort_keys=True))"
    )
    values=[]
    for seed in ("0","1"):
        env=dict(os.environ,PYTHONHASHSEED=seed)
        proc=subprocess.run([sys.executable,"-s","-B","-c",script],env=env,
            cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,check=True)
        values.append(json.loads(proc.stdout))
    assert values[0]==values[1]


def _source(seed: int) -> np.ndarray:
    value = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE) + float(seed)
    value[0, 0, 0] = np.nan
    value[0, 0, 1] = np.inf
    value[0, 0, 2] = -np.inf
    value[0, 1, 0] = -0.0
    return np.ascontiguousarray(value)


def _assert_scalar_tree(value: Any) -> None:
    assert not isinstance(value, np.ndarray)
    if isinstance(value, dict):
        for item in value.values():
            _assert_scalar_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_scalar_tree(item)


def _assert_ieee_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    assert np.array_equal(actual_nan, expected_nan)
    actual_inf = np.isinf(actual)
    expected_inf = np.isinf(expected)
    assert np.array_equal(actual_inf, expected_inf)
    finite = ~(actual_nan | actual_inf)
    np.testing.assert_array_equal(actual[finite], expected[finite])
    both_inf = actual_inf & expected_inf
    assert np.array_equal(np.signbit(actual[both_inf]), np.signbit(expected[both_inf]))
    both_zero = finite & (actual == 0.0) & (expected == 0.0)
    if np.any(both_zero):
        np.testing.assert_array_equal(
            actual[both_zero].view(np.uint64), expected[both_zero].view(np.uint64)
        )


def _populate(store: AlphaCaptureStore) -> dict[int, dict[str, np.ndarray | float]]:
    states: dict[int, dict[str, np.ndarray | float]] = {}
    for interval in range(8):
        source = _source(interval)
        intensity = np.ascontiguousarray(np.abs(source.copy()) + 2.0)
        intensity[0, 0, 0] = 1.0
        floor = float(0.25 + interval / 100.0)
        source_before = source.copy()
        clean_token = store.before_call(
            context=ALPHA_CLEAN_SOURCE_HELPER,
            interval_index=interval,
            source=source,
            source_site=ALPHA_SOURCE_SITE,
            capture_inputs={"source_input": source},
        )
        cleaned = np.nan_to_num(
            source, nan=0.0, posinf=0.0, neginf=0.0, copy=True
        )
        store.after_call(
            clean_token,
            source=source,
            output=cleaned,
            capture_outputs={"cleaned_source_output": cleaned},
        )
        _assert_ieee_equal(source, source_before)

        ratio_token = store.before_call(
            context=ALPHA_RATIO_HELPER,
            interval_index=interval,
            source=cleaned,
            source_site=ALPHA_SOURCE_SITE,
            capture_inputs={"intensity_input": intensity, "I_floor": floor},
        )
        with np.errstate(all="ignore"):
            quotient = np.divide(cleaned, np.add(intensity, floor))
        store.after_call(
            ratio_token,
            source=cleaned,
            output=quotient,
            capture_inputs={"intensity_input": intensity},
        )

        clean_alpha_token = store.before_call(
            context=ALPHA_CLEAN_RESULT_HELPER,
            interval_index=interval,
            source=quotient,
            source_site=ALPHA_SOURCE_SITE,
        )
        cleaned_alpha = np.nan_to_num(
            quotient, nan=0.0, posinf=0.0, neginf=0.0, copy=True
        )
        store.after_call(
            clean_alpha_token,
            source=quotient,
            output=cleaned_alpha,
        )

        clip_token = store.before_call(
            context=ALPHA_CLIP_HELPER,
            interval_index=interval,
            source=cleaned_alpha,
            source_site=ALPHA_SOURCE_SITE,
        )
        external_view = cleaned_alpha.view()
        np.maximum(cleaned_alpha, 0.0, out=cleaned_alpha)
        store.after_call(
            clip_token,
            source=cleaned_alpha,
            output=cleaned_alpha,
            capture_outputs={"final_alpha": cleaned_alpha, "inplace": True},
        )
        assert np.shares_memory(external_view, cleaned_alpha)
        if interval in CAPTURE_INTERVALS:
            states[interval] = {
                "source": source_before,
                "intensity": intensity.copy(),
                "cleaned": cleaned.copy(),
                "alpha": cleaned_alpha.copy(),
                "floor": floor,
            }
    return states


def test_capture_writes_three_states_and_loads_all_persisted_identity(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    store = AlphaCaptureStore(root, shape=SHAPE)
    expected_states = _populate(store)
    receipt = store.finish(binding={"execution_commit": "synthetic"})
    assert receipt["total_calls"] == 32
    assert receipt["array_file_count"] == 12
    assert receipt["source_binding"]["execution_commit"] == "synthetic"
    assert receipt["retained_array_references"] == 0
    _assert_scalar_tree(receipt)

    loaded = load_capture(root, shape=SHAPE, expected_binding={"execution_commit": "synthetic"})
    assert [row["interval_index"] for row in loaded["records"]] == [0, 3, 7]
    assert sorted(path.name for path in root.glob("*.npy"))
    for row in loaded["records"]:
        arrays = load_state_arrays(root, row)
        expected = expected_states[int(row["interval_index"])]
        for field, key in (
            ("source_input", "source"),
            ("intensity_input", "intensity"),
            ("cleaned_source_output", "cleaned"),
            ("alpha_output", "alpha"),
        ):
            _assert_ieee_equal(arrays[field], expected[key])
        assert type(row["I_floor"]) is float
        assert row["clip_inplace"] is True


def test_capture_rejects_wrong_order_alias_and_out_of_place_clip(tmp_path: Path) -> None:
    store = AlphaCaptureStore(tmp_path / "bad", shape=SHAPE)
    source = _source(0)
    with pytest.raises(AlphaCaptureError, match="CALL_ORDER"):
        store.before_call(
            context=ALPHA_RATIO_HELPER,
            interval_index=0,
            source=source,
            source_site=ALPHA_SOURCE_SITE,
            capture_inputs={"intensity_input": source, "I_floor": 0.5},
        )

    token = store.before_call(
        context=ALPHA_CLEAN_SOURCE_HELPER,
        interval_index=0,
        source=source,
        source_site=ALPHA_SOURCE_SITE,
        capture_inputs={"source_input": source},
    )
    cleaned = np.nan_to_num(source, copy=True)
    store.after_call(token, source=source, output=cleaned)
    with pytest.raises(AlphaCaptureError, match="SOURCE_INTENSITY_ALIAS"):
        store.before_call(
            context=ALPHA_RATIO_HELPER,
            interval_index=0,
            source=cleaned,
            source_site=ALPHA_SOURCE_SITE,
            capture_inputs={"intensity_input": cleaned, "I_floor": 0.5},
        )

    # Close this incomplete synthetic transaction without using its receipt;
    # the failed call must not silently advance the four-helper order.
    store.pending = None
    with pytest.raises(AlphaCaptureError, match="COMPLETENESS"):
        store.finish()


def test_capture_rejects_changed_nonclip_input_and_bad_file_identity(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    store = AlphaCaptureStore(root, shape=SHAPE)
    source = _source(0)
    token = store.before_call(
        context=ALPHA_CLEAN_SOURCE_HELPER,
        interval_index=0,
        source=source,
        source_site=ALPHA_SOURCE_SITE,
        capture_inputs={"source_input": source},
    )
    source[0, 1, 1] = 9.0
    with pytest.raises(AlphaCaptureError, match="INPUT_CHANGED"):
        store.after_call(token, source=source, output=np.nan_to_num(source, copy=True))

    # The store has no valid receipt after the failed callback.  A complete
    # package is created separately, then its NPY bytes are changed so the
    # offline loader must reject the recorded raw identity.
    complete_root = tmp_path / "complete"
    complete = AlphaCaptureStore(complete_root, shape=SHAPE)
    _populate(complete)
    complete.finish(binding={"execution_commit": "synthetic"})
    target = next(complete_root.glob("i1_source_input.npy"))
    with target.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(AlphaCaptureError, match="FILE_IDENTITY"):
        load_capture(complete_root, shape=SHAPE)


__all__ = ["_populate"]

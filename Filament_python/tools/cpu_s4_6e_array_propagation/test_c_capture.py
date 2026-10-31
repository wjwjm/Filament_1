"""Focused C capture and local mechanism checks."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from KHz_filament.nonlinear import apply_nonlinear

from .c_capture import CaptureStore
from .c_mechanism import MechanismError, run_local_mechanism


def _store(group: str, seed: int) -> CaptureStore:
    rng = np.random.default_rng(seed)
    store = CaptureStore()
    store.begin(group)
    for interval in range(8):
        E = np.asarray(rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2)), dtype=np.complex128, order="C")
        phase = np.asarray(rng.normal(size=E.shape), dtype=np.float64, order="C")
        alpha = np.asarray(rng.normal(size=E.shape), dtype=np.float64, order="C")
        dn = np.asarray(rng.normal(size=E.shape[1:]), dtype=np.float64, order="C")
        store.before_call(interval, (E, phase, alpha, 0.01), {"dn_gas": dn, "k0": 0.2})
        apply_nonlinear(E, phase, alpha, 0.01, dn_gas=dn, k0=0.2)
        store.after_call(interval, E, (E, phase, alpha, 0.01), {"dn_gas": dn, "k0": 0.2})
    return store


def _bundle(store: CaptureStore, group: str) -> dict[str, object]:
    metadata = store.metadata()
    metadata["group"] = group
    records = store.records()
    for row in records.values():
        row["group"] = group
    return {"metadata": metadata, "records": records}


def test_s46e_c_capture_selected_intervals_are_copied_and_fingerprinted():
    store = _store("G1", 17)
    metadata = store.metadata()
    assert metadata["selected_interval_ordinals_1based"] == [1, 4, 8]
    assert metadata["expected_total_calls"] == 8
    assert metadata["interval_sequence"] == list(range(8))
    assert metadata["source_role"] == "R"
    assert metadata["record_count"] == 3
    rows = store.records()
    assert all(row["output"] is not None for row in rows.values())
    assert all(row["live_E_id"] != id(row["output"]) for row in rows.values())
    assert all(row["input_fingerprints"]["E"]["sha256"] for row in rows.values())


def test_s46e_c_local_mechanism_uses_reversed_blocks_and_90_formal_calls():
    shared_store = _store("G1", 17)
    shared = _bundle(shared_store, "G1")
    shared_g2 = copy.deepcopy(shared)
    shared_g2["metadata"]["group"] = "G2"
    for row in shared_g2["records"].values():
        row["group"] = "G2"
    groups = {
        "G1": shared,
        "G2": shared_g2,
    }
    result = run_local_mechanism(groups)
    assert result["warmup_calls"] == 36
    assert result["formal_calls"] == 90
    assert result["config_order"]["G1"] == ("REF", "P1", "P24")
    assert result["config_order"]["G2"] == ("P24", "P1", "REF")
    assert result["exact_by_config"] == {"REF": True, "P1": True, "P24": True}
    assert len(result["formal_rows"]) == 90
    assert set(result["p24_block_gains"]) == {"G1:0", "G1:3", "G1:7", "G2:0", "G2:3", "G2:7"}
    assert "p1_block_gate" not in result
    assert all(row["restore_outside_clock"] and row["post_checks_outside_clock"] for row in result["formal_rows"])
    assert result["production_qualified"] is False


def test_s46e_c_nonexact_warmup_preserves_failure_index_and_stops():
    g1 = _bundle(_store("G1", 17), "G1")
    g1["records"][0]["output"][0, 0, 0] += 1.0
    g2 = copy.deepcopy(g1)
    g2["metadata"]["group"] = "G2"
    for row in g2["records"].values():
        row["group"] = "G2"
    groups = {"G1": g1, "G2": g2}
    result = run_local_mechanism(groups)
    assert result["status"] == "SCIENTIFIC_NONEXACT"
    assert result["failure"]["group"] == "G1"
    assert result["failure"]["config"] == "REF"
    assert result["failure"]["interval_index"] == 0
    assert result["failure"]["phase"] == "warmup"
    assert result["failure"]["failure"] == "NONEXACT"
    evidence = result["failure"]["evidence"]
    assert evidence["actual"]["shape"] == evidence["reference"]["shape"]
    assert evidence["actual"]["dtype"] == evidence["reference"]["dtype"] == "complex128"
    assert evidence["actual"]["sha256"] != evidence["reference"]["sha256"]
    assert evidence["first_difference_index"] == [0, 0, 0]
    assert evidence["difference_count"] >= 1
    assert result["formal_calls"] == 0
    assert result["stopped_before_remaining"] is True
    assert result["exact_by_config"]["REF"] is False


def test_s46e_c_mechanism_rejects_different_block_capture_states():
    groups = {
        "G1": _bundle(_store("G1", 17), "G1"),
        "G2": _bundle(_store("G2", 18), "G2"),
    }
    with pytest.raises(MechanismError, match="STATE_MISMATCH"):
        run_local_mechanism(groups)

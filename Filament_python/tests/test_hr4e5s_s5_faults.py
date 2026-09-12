from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from KHz_filament.hr4e5s_s3 import _SelectedStreamingHook, finalize_streaming
from KHz_filament.hr4e5s_s5 import compare_clean_reference, inspect_lifecycle
from KHz_filament.hr4e5s_streaming import S5FaultInjectedError, S5_FAULT_SPECS, StreamingLifecycle


def _records(count: int = 8) -> list[dict[str, object]]:
    return [{"ordinal": index, "screen_id": f"z{index:05d}", "z_m": index * 1.0e-4} for index in range(count)]


def _lifecycle(tmp_path, name: str, *, count: int = 8) -> StreamingLifecycle:
    coordinate = np.linspace(-1.0, 1.0, 8)
    mode = np.outer(1.0 - coordinate**2, 1.0 - coordinate**2)
    current = {
        "delta_n": np.stack([-1.0e-6 * (index + 1) * mode for index in range(count)]).astype(np.float64),
        "vx": np.zeros((count, 8, 8), dtype=np.float64),
        "vy": np.zeros((count, 8, 8), dtype=np.float64),
    }
    return StreamingLifecycle.create(
        root=tmp_path / name,
        current=current,
        screen_records=_records(count),
        current_generation="s5-current",
        dx_m=1.0e-4,
        dy_m=1.0e-4,
        queue_depth=max(8, count),
        actor="s5_test",
    )


def _post_fields(lifecycle: StreamingLifecycle, ordinal: int) -> dict[str, np.ndarray]:
    return lifecycle._artifact_fields(lifecycle.manifest["records"][ordinal]["post"], namespace="POST")


def _commit_post(lifecycle: StreamingLifecycle, ordinal: int) -> None:
    lifecycle.deposition_finalized(ordinal, actor="s5_test")
    lifecycle.commit_post_from_delta_n(ordinal, lifecycle.current_fields(ordinal)["delta_n"] - 1.0e-8, actor="s5_test")
    lifecycle.enqueue_post(ordinal, actor="s5_test")


def _prepare_posts(lifecycle: StreamingLifecycle) -> None:
    for ordinal in range(int(lifecycle.manifest["expected_screen_count"])):
        _commit_post(lifecycle, ordinal)


def _commit_all_next(lifecycle: StreamingLifecycle) -> None:
    while True:
        block = lifecycle.claim_block(actor="s5_test")
        if not block:
            return
        for ordinal in block:
            lifecycle.commit_next(ordinal, _post_fields(lifecycle, ordinal), actor="s5_test")


def _finish(lifecycle: StreamingLifecycle) -> StreamingLifecycle:
    _commit_all_next(lifecycle)
    assert lifecycle.validate_barrier(actor="s5_test")["status"] == "PASS"
    assert lifecycle.promote_next_to_current(actor="s5_test")["authoritative_namespace"] == "NEXT"
    return StreamingLifecycle.open(lifecycle.root)


def _exact_lifecycle_view(lifecycle: StreamingLifecycle) -> dict[str, object]:
    return {
        "current_generation": lifecycle.manifest["current_generation"],
        "next_generation": lifecycle.manifest["next_generation"],
        "records": [
            {
                "ordinal": record["ordinal"],
                "screen_id": record["screen_id"],
                "current": record["current"]["field_sha256"],
                "post": record["post"]["field_sha256"],
                "next": record["next"]["field_sha256"],
            }
            for record in lifecycle.manifest["records"]
        ],
        "barrier": {
            "status": lifecycle.manifest["barrier"]["status"],
            "current_generation": lifecycle.manifest["barrier"]["current_generation"],
            "next_generation": lifecycle.manifest["barrier"]["next_generation"],
            "current_content_sha256": lifecycle.manifest["barrier"]["current_content_sha256"],
        },
        "promotion": {
            "authoritative_namespace": lifecycle.manifest["promotion"]["authoritative_namespace"],
            "source_current_generation": lifecycle.manifest["promotion"]["source_current_generation"],
            "authoritative_generation": lifecycle.manifest["promotion"]["authoritative_generation"],
        },
    }


def _clean_reference(tmp_path) -> StreamingLifecycle:
    lifecycle = _lifecycle(tmp_path, "clean")
    _prepare_posts(lifecycle)
    return _finish(lifecycle)


def _enable(monkeypatch, fault_id: str, screen: str = "z00000") -> None:
    monkeypatch.setenv("HR4_S5_FAULT_ID", fault_id)
    monkeypatch.setenv("HR4_S5_FAULT_SCREEN", screen)
    monkeypatch.setenv("HR4_S5_FAULT_ONCE", "1")


def test_s5_contract_fault_ids_are_unique_and_invalid_id_is_rejected(tmp_path, monkeypatch):
    required = {
        "F01_OPTICAL_PRE_POST_COMMIT",
        "F02_POST_COMMITTED_PRE_HYDRO",
        "F03_HYDRO_PRE_NEXT_COMMIT",
        "F04_NEXT_TEMP_PRE_ATOMIC_RENAME",
        "F05_NEXT_COMMITTED_PRE_BARRIER",
        "F06_BARRIER_PASS_PRE_PROMOTION",
    }
    assert required <= set(S5_FAULT_SPECS)
    assert len(S5_FAULT_SPECS) == len(set(S5_FAULT_SPECS)) == 9
    monkeypatch.setenv("HR4_S5_FAULT_ID", "F99_NOT_A_CONTRACT_BOUNDARY")
    monkeypatch.setenv("HR4_S5_FAULT_SCREEN", "z00000")
    with pytest.raises(ValueError, match="unknown HR4 S5 fault ID"):
        _lifecycle(tmp_path, "invalid")
    monkeypatch.setenv("HR4_S5_FAULT_ID", "F01_OPTICAL_PRE_POST_COMMIT")
    monkeypatch.setenv("HR4_S5_FAULT_SCREEN", "not-an-authoritative-screen")
    with pytest.raises(ValueError, match="not an authoritative screen"):
        _lifecycle(tmp_path, "invalid-target")


def test_s5_fault_off_preserves_normal_lifecycle(tmp_path, monkeypatch):
    monkeypatch.delenv("HR4_S5_FAULT_ID", raising=False)
    monkeypatch.delenv("HR4_S5_FAULT_SCREEN", raising=False)
    lifecycle = _lifecycle(tmp_path, "off")
    _prepare_posts(lifecycle)
    finished = _finish(lifecycle)
    assert not (finished.root / "s5_fault_provenance.json").exists()
    assert finished._authoritative_namespace == "NEXT"


def test_f01_replays_missing_post_once_and_matches_clean_reference(tmp_path, monkeypatch):
    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, "F01_OPTICAL_PRE_POST_COMMIT")
    lifecycle = _lifecycle(tmp_path, "f01")
    lifecycle.deposition_finalized(0, actor="s5_test")
    with pytest.raises(S5FaultInjectedError, match="F01_OPTICAL_PRE_POST_COMMIT"):
        lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"] - 1.0e-8, actor="s5_test")
    assert lifecycle.manifest["records"][0]["post"] is None
    assert lifecycle.manifest["records"][0]["state"] == "DEPOSITION_FINALIZED"
    assert json.loads((lifecycle.root / "s5_fault_provenance.json").read_text(encoding="utf-8"))["fault_once"] is True
    lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"] - 1.0e-8, actor="s5_test")
    lifecycle.enqueue_post(0, actor="s5_test")
    for ordinal in range(1, 8):
        _commit_post(lifecycle, ordinal)
    assert _exact_lifecycle_view(_finish(lifecycle)) == _exact_lifecycle_view(clean)


def test_f01_resume_hook_replays_only_missing_post(tmp_path, monkeypatch):
    _enable(monkeypatch, "F01_OPTICAL_PRE_POST_COMMIT")
    lifecycle = _lifecycle(tmp_path, "f01-resume")
    lifecycle.deposition_finalized(0, actor="s5_test")
    state_after = lifecycle.current_fields(0)["delta_n"] - 1.0e-8
    with pytest.raises(S5FaultInjectedError, match="F01_OPTICAL_PRE_POST_COMMIT"):
        lifecycle.commit_post_from_delta_n(0, state_after, actor="s5_test")
    hook = _SelectedStreamingHook(lifecycle, [{"source_index": 77, "ordinal": 0}], resume=True)
    hook(interval=SimpleNamespace(index=77), state_after=state_after, hr3a_authoritative=True, hr3b_authoritative=True)
    assert lifecycle.has_authoritative_post(0)
    assert lifecycle.manifest["records"][0]["state"] == "HYDRO_QUEUED"


def test_f02_retains_post_and_reconstructs_hydro_queue(tmp_path, monkeypatch):
    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, "F02_POST_COMMITTED_PRE_HYDRO")
    lifecycle = _lifecycle(tmp_path, "f02")
    lifecycle.deposition_finalized(0, actor="s5_test")
    with pytest.raises(S5FaultInjectedError, match="F02_POST_COMMITTED_PRE_HYDRO"):
        lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"] - 1.0e-8, actor="s5_test")
    assert lifecycle.manifest["records"][0]["post"] is not None
    recovered = StreamingLifecycle.open(lifecycle.root)
    assert recovered.reconstruct_queue(actor="s5_restart") == [0]
    for ordinal in range(1, 8):
        _commit_post(recovered, ordinal)
    assert _exact_lifecycle_view(_finish(recovered)) == _exact_lifecycle_view(clean)


def test_f03_requeues_hydro_running_work_without_rewriting_post(tmp_path, monkeypatch):
    import KHz_filament.hr4e5s_streaming as streaming

    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, "F03_HYDRO_PRE_NEXT_COMMIT")
    lifecycle = _lifecycle(tmp_path, "f03")
    _prepare_posts(lifecycle)
    monkeypatch.setattr(
        streaming,
        "advance_hr4_single_screen",
        lambda delta_n, vx, vy, **kwargs: {"delta_n": delta_n, "vx": vx, "vy": vy},
    )
    with pytest.raises(S5FaultInjectedError, match="F03_HYDRO_PRE_NEXT_COMMIT"):
        lifecycle.run_one_hydro_block(dt_hydro=1.0e-6, n_hydro_steps=1, chi=0.0, nu=0.0, n0=1.0, gravity_y=0.0)
    assert all(record["next"] is None for record in lifecycle.manifest["records"])
    recovered = StreamingLifecycle.open(lifecycle.root)
    assert recovered.reconstruct_queue(actor="s5_restart") == list(range(8))
    assert _exact_lifecycle_view(_finish(recovered)) == _exact_lifecycle_view(clean)


def test_f04_discards_only_staged_next_and_matches_clean_reference(tmp_path, monkeypatch):
    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, "F04_NEXT_TEMP_PRE_ATOMIC_RENAME")
    lifecycle = _lifecycle(tmp_path, "f04")
    _prepare_posts(lifecycle)
    assert lifecycle.claim_block(actor="s5_test") == list(range(8))
    with pytest.raises(S5FaultInjectedError, match="F04_NEXT_TEMP_PRE_ATOMIC_RENAME"):
        lifecycle.commit_next(0, _post_fields(lifecycle, 0), actor="s5_test")
    assert lifecycle.manifest["records"][0]["next"] is None
    assert list((lifecycle.root / "next").glob("screen_*.npz.*.tmp"))
    recovered = StreamingLifecycle.open(lifecycle.root)
    assert recovered.reconstruct_queue(actor="s5_restart") == list(range(8))
    assert not list((recovered.root / "next").glob("screen_*.npz.*.tmp"))
    assert _exact_lifecycle_view(_finish(recovered)) == _exact_lifecycle_view(clean)


def test_f05_retains_next_until_barrier_then_matches_clean_reference(tmp_path, monkeypatch):
    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, "F05_NEXT_COMMITTED_PRE_BARRIER", screen="z00007")
    lifecycle = _lifecycle(tmp_path, "f05")
    _prepare_posts(lifecycle)
    assert lifecycle.claim_block(actor="s5_test") == list(range(8))
    for ordinal in range(7):
        lifecycle.commit_next(ordinal, _post_fields(lifecycle, ordinal), actor="s5_test")
    with pytest.raises(S5FaultInjectedError, match="F05_NEXT_COMMITTED_PRE_BARRIER"):
        lifecycle.commit_next(7, _post_fields(lifecycle, 7), actor="s5_test")
    assert lifecycle.manifest["records"][7]["next"] is not None
    assert lifecycle.manifest["barrier"] is None
    assert _exact_lifecycle_view(_finish(lifecycle)) == _exact_lifecycle_view(clean)


def test_f07_discards_orphaned_renamed_next_and_matches_clean_reference(tmp_path, monkeypatch):
    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, "F07_NEXT_RENAMED_PRE_MANIFEST")
    lifecycle = _lifecycle(tmp_path, "f07")
    _prepare_posts(lifecycle)
    assert lifecycle.claim_block(actor="s5_test") == list(range(8))
    with pytest.raises(S5FaultInjectedError, match="F07_NEXT_RENAMED_PRE_MANIFEST"):
        lifecycle.commit_next(0, _post_fields(lifecycle, 0), actor="s5_test")
    assert lifecycle.manifest["records"][0]["next"] is None
    assert (lifecycle.root / "next" / "screen_000000.npz").is_file()
    recovered = StreamingLifecycle.open(lifecycle.root)
    assert recovered.reconstruct_queue(actor="s5_restart") == list(range(8))
    assert not (recovered.root / "next" / "screen_000000.npz").exists()
    assert _exact_lifecycle_view(_finish(recovered)) == _exact_lifecycle_view(clean)


def test_f09_discards_orphaned_renamed_post_and_matches_clean_reference(tmp_path, monkeypatch):
    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, "F09_POST_RENAMED_PRE_MANIFEST")
    lifecycle = _lifecycle(tmp_path, "f09")
    lifecycle.deposition_finalized(0, actor="s5_test")
    with pytest.raises(S5FaultInjectedError, match="F09_POST_RENAMED_PRE_MANIFEST"):
        lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"] - 1.0e-8, actor="s5_test")
    assert lifecycle.manifest["records"][0]["post"] is None
    assert (lifecycle.root / "post" / "screen_000000.npz").is_file()
    recovered = StreamingLifecycle.open(lifecycle.root)
    assert recovered.reconstruct_queue(actor="s5_restart") == []
    assert not (recovered.root / "post" / "screen_000000.npz").exists()
    _commit_post(recovered, 0)
    for ordinal in range(1, 8):
        _commit_post(recovered, ordinal)
    assert _exact_lifecycle_view(_finish(recovered)) == _exact_lifecycle_view(clean)


def test_f06_reuses_pass_barrier_for_one_promotion(tmp_path, monkeypatch):
    _enable(monkeypatch, "F06_BARRIER_PASS_PRE_PROMOTION", screen="z00007")
    lifecycle = _lifecycle(tmp_path, "f06")
    _prepare_posts(lifecycle)
    _commit_all_next(lifecycle)
    with pytest.raises(S5FaultInjectedError, match="F06_BARRIER_PASS_PRE_PROMOTION"):
        finalize_streaming(lifecycle_root=lifecycle.root, out_path=tmp_path / "f06-final.json")
    failed = StreamingLifecycle.open(lifecycle.root)
    assert failed.manifest["barrier"]["status"] == "PASS"
    assert not (failed.root / "authoritative_generation.json").exists()
    result = finalize_streaming(lifecycle_root=failed.root, out_path=tmp_path / "f06-final.json")
    assert result["promotion"]["authoritative_namespace"] == "NEXT"
    reopened = StreamingLifecycle.open(failed.root)
    first_pointer = json.loads((reopened.root / "authoritative_generation.json").read_text(encoding="utf-8"))
    assert reopened.promote_next_to_current(actor="s5_restart") == first_pointer
    assert json.loads((reopened.root / "authoritative_generation.json").read_text(encoding="utf-8")) == first_pointer


def test_f08_finishes_existing_promotion_pointer_without_rewrite(tmp_path, monkeypatch):
    _enable(monkeypatch, "F08_PROMOTION_TRANSACTION_BOUNDARY", screen="z00007")
    lifecycle = _lifecycle(tmp_path, "f08")
    _prepare_posts(lifecycle)
    _commit_all_next(lifecycle)
    assert lifecycle.validate_barrier(actor="s5_test")["status"] == "PASS"
    with pytest.raises(S5FaultInjectedError, match="F08_PROMOTION_TRANSACTION_BOUNDARY"):
        lifecycle.promote_next_to_current(actor="s5_test")
    pointer_path = lifecycle.root / "authoritative_generation.json"
    pointer_before = pointer_path.read_bytes()
    failed = StreamingLifecycle.open(lifecycle.root)
    assert failed.manifest["promotion"] is None
    assert failed.promote_next_to_current(actor="s5_restart") == json.loads(pointer_before.decode("utf-8"))
    assert pointer_path.read_bytes() == pointer_before


def test_restart_discards_only_atomic_json_temporary_siblings(tmp_path, monkeypatch):
    monkeypatch.delenv("HR4_S5_FAULT_ID", raising=False)
    lifecycle = _lifecycle(tmp_path, "json-temp")
    manifest_temp = lifecycle.root / "streaming_manifest.json.crash.tmp"
    pointer_temp = lifecycle.root / "authoritative_generation.json.crash.tmp"
    manifest_temp.write_text("incomplete", encoding="utf-8")
    pointer_temp.write_text("incomplete", encoding="utf-8")
    recovered = StreamingLifecycle.open(lifecycle.root)
    assert recovered.reconstruct_queue(actor="s5_restart") == []
    assert not manifest_temp.exists() and not pointer_temp.exists()


def test_restart_rejects_unqualified_staged_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("HR4_S5_FAULT_ID", raising=False)
    lifecycle = _lifecycle(tmp_path, "unknown-temp")
    unknown = lifecycle.root / "current" / "screen_000000.npz.crash.tmp"
    unknown.write_text("incomplete", encoding="utf-8")
    recovered = StreamingLifecycle.open(lifecycle.root)
    with pytest.raises(Exception, match="staged artifact"):
        recovered.reconstruct_queue(actor="s5_restart")
    assert unknown.exists()


def _s5_optical_stub(directory, count):
    directory.mkdir()
    for name in ("ion", "ib", "raman"):
        np.save(directory / f"s3_optical.hr3a_q{name}_samples.npy", np.zeros((count, 2, 2), dtype=np.float64))
    np.save(directory / "final.npy", np.ones((2, 2), dtype=np.complex128))
    np.savez(directory / "ledger.npz", ion=np.arange(count, dtype=np.float64))
    (directory / "optical_run.json").write_text(json.dumps({
        "final_optical_field": "final.npy", "ledger": "ledger.npz",
        "optical_current_ownership_before": {"authoritative_namespace": "CURRENT"},
        "optical_current_ownership_after": {"authoritative_namespace": "CURRENT"},
    }), encoding="utf-8")


def test_s5_snapshot_and_comparator_exactly_compares_real_window_cardinality(tmp_path, monkeypatch):
    monkeypatch.delenv("HR4_S5_FAULT_ID", raising=False)
    reference = _lifecycle(tmp_path, "reference", count=48)
    _prepare_posts(reference)
    reference = _finish(reference)
    candidate = _lifecycle(tmp_path, "candidate", count=48)
    _prepare_posts(candidate)
    candidate = _finish(candidate)
    reference_optical, candidate_optical = tmp_path / "reference-optical", tmp_path / "candidate-optical"
    _s5_optical_stub(reference_optical, 48)
    _s5_optical_stub(candidate_optical, 48)
    snapshot = inspect_lifecycle(lifecycle_root=reference.root, out_path=tmp_path / "snapshot.json")
    assert snapshot["pointer_present"] is True
    result = compare_clean_reference(
        reference_lifecycle_root=reference.root, reference_optical_dir=reference_optical,
        candidate_lifecycle_root=candidate.root, candidate_optical_dir=candidate_optical,
        out_dir=tmp_path / "comparison",
    )
    assert result["status"] == "PASS"
    assert result["completed_field_comparisons"] == result["expected_field_comparisons"] == 48 * 9

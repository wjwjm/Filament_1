from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from KHz_filament.hr4e5s_s3 import _SelectedStreamingHook, bootstrap_recovery, finalize_streaming, validate_recovery_bootstrap_receipt
from KHz_filament.hr4e5s_s5 import compare_clean_reference, inspect_lifecycle, validate_fault_contract, validate_recovery_provenance
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


def _commit_durable_posts_without_queue(lifecycle: StreamingLifecycle) -> None:
    """Build persisted POST candidates for serial-bootstrap tests only."""
    for ordinal in range(int(lifecycle.manifest["expected_screen_count"])):
        lifecycle.deposition_finalized(ordinal, actor="s5_test")
        lifecycle.commit_post_from_delta_n(ordinal, lifecycle.current_fields(ordinal)["delta_n"] - 1.0e-8, actor="s5_test")


def _recovery_lifecycle(tmp_path, name: str, *, count: int, queue_depth: int) -> StreamingLifecycle:
    coordinate = np.linspace(-1.0, 1.0, 8)
    mode = np.outer(1.0 - coordinate**2, 1.0 - coordinate**2)
    current = {
        "delta_n": np.stack([-1.0e-6 * (index + 1) * mode for index in range(count)]).astype(np.float64),
        "vx": np.zeros((count, 8, 8), dtype=np.float64),
        "vy": np.zeros((count, 8, 8), dtype=np.float64),
    }
    return StreamingLifecycle.create(
        root=tmp_path / name, current=current, screen_records=_records(count), current_generation="s5-current",
        dx_m=1.0e-4, dy_m=1.0e-4, queue_depth=queue_depth, actor="s5_test",
    )


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


def _faulted_lifecycle(tmp_path, monkeypatch, fault_id: str):
    """Build one persisted crash state without using recovery/reconstruction."""
    target = "z00007" if fault_id in {"F05_NEXT_COMMITTED_PRE_BARRIER", "F06_BARRIER_PASS_PRE_PROMOTION"} else "z00000"
    _enable(monkeypatch, fault_id, target)
    lifecycle = _lifecycle(tmp_path, f"contract-{fault_id}-{len(list(tmp_path.iterdir()))}")
    if fault_id == "F01_OPTICAL_PRE_POST_COMMIT":
        lifecycle.deposition_finalized(0, actor="s5_contract_test")
        with pytest.raises(S5FaultInjectedError):
            lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"] - 1.0e-8, actor="s5_contract_test")
    elif fault_id == "F02_POST_COMMITTED_PRE_HYDRO":
        lifecycle.deposition_finalized(0, actor="s5_contract_test")
        with pytest.raises(S5FaultInjectedError):
            lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"] - 1.0e-8, actor="s5_contract_test")
    elif fault_id == "F03_HYDRO_PRE_NEXT_COMMIT":
        import KHz_filament.hr4e5s_streaming as streaming

        _prepare_posts(lifecycle)
        monkeypatch.setattr(streaming, "advance_hr4_single_screen", lambda delta_n, vx, vy, **kwargs: {"delta_n": delta_n, "vx": vx, "vy": vy})
        with pytest.raises(S5FaultInjectedError):
            lifecycle.run_one_hydro_block(dt_hydro=1.0e-6, n_hydro_steps=1, chi=0.0, nu=0.0, n0=1.0, gravity_y=0.0)
    elif fault_id == "F04_NEXT_TEMP_PRE_ATOMIC_RENAME":
        _prepare_posts(lifecycle)
        assert lifecycle.claim_block(actor="s5_contract_test") == list(range(8))
        with pytest.raises(S5FaultInjectedError):
            lifecycle.commit_next(0, _post_fields(lifecycle, 0), actor="s5_contract_test")
    elif fault_id == "F05_NEXT_COMMITTED_PRE_BARRIER":
        _prepare_posts(lifecycle)
        assert lifecycle.claim_block(actor="s5_contract_test") == list(range(8))
        for ordinal in range(7):
            lifecycle.commit_next(ordinal, _post_fields(lifecycle, ordinal), actor="s5_contract_test")
        with pytest.raises(S5FaultInjectedError):
            lifecycle.commit_next(7, _post_fields(lifecycle, 7), actor="s5_contract_test")
    elif fault_id == "F06_BARRIER_PASS_PRE_PROMOTION":
        _prepare_posts(lifecycle)
        _commit_all_next(lifecycle)
        with pytest.raises(S5FaultInjectedError):
            finalize_streaming(lifecycle_root=lifecycle.root, out_path=tmp_path / f"{fault_id}.json")
    else:
        raise AssertionError(fault_id)
    return lifecycle.root, target


def _contract_failure(tmp_path, monkeypatch, fault_id: str, mutate) -> None:
    root, target = _faulted_lifecycle(tmp_path, monkeypatch, fault_id)
    monkeypatch.delenv("HR4_S5_FAULT_ID", raising=False)
    monkeypatch.delenv("HR4_S5_FAULT_SCREEN", raising=False)
    monkeypatch.delenv("HR4_S5_FAULT_ONCE", raising=False)
    mutate(root)
    result = validate_fault_contract(lifecycle_root=root, fault_id=fault_id, target_screen=target)
    assert result["status"] == "FAIL", result


def _monitor_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "monitor_hr4e5s_s5_fault_matrix.py"
    spec = importlib.util.spec_from_file_location("s5_fault_matrix_monitor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.mark.parametrize("fault_id", [
    "F01_OPTICAL_PRE_POST_COMMIT",
    "F02_POST_COMMITTED_PRE_HYDRO",
    "F03_HYDRO_PRE_NEXT_COMMIT",
    "F04_NEXT_TEMP_PRE_ATOMIC_RENAME",
    "F05_NEXT_COMMITTED_PRE_BARRIER",
    "F06_BARRIER_PASS_PRE_PROMOTION",
])
def test_s5_contract_gate_accepts_each_declared_persisted_crash_state(tmp_path, monkeypatch, fault_id):
    root, target = _faulted_lifecycle(tmp_path, monkeypatch, fault_id)
    out = root.parent / "contract_check.json"
    result = validate_fault_contract(lifecycle_root=root, fault_id=fault_id, target_screen=target, out_path=out)
    assert result["status"] == "PASS", result
    assert result["contract_match"] is True
    assert json.loads(out.read_text(encoding="utf-8"))["checks"] == result["checks"]


def test_s5_contract_gate_rejects_required_authoritative_state_and_provenance_violations(tmp_path, monkeypatch):
    def f01_post(root):
        lifecycle = StreamingLifecycle.open(root)
        lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"] - 1.0e-8, actor="negative")

    _contract_failure(tmp_path, monkeypatch, "F01_OPTICAL_PRE_POST_COMMIT", f01_post)

    def f02_missing_post(root):
        lifecycle = StreamingLifecycle.open(root)
        record = lifecycle.manifest["records"][0]
        record["post"] = None
        record["state"] = "DEPOSITION_FINALIZED"
        lifecycle._save()

    _contract_failure(tmp_path, monkeypatch, "F02_POST_COMMITTED_PRE_HYDRO", f02_missing_post)

    def f03_next_present(root):
        lifecycle = StreamingLifecycle.open(root)
        lifecycle.commit_next(0, _post_fields(lifecycle, 0), actor="negative")

    _contract_failure(tmp_path, monkeypatch, "F03_HYDRO_PRE_NEXT_COMMIT", f03_next_present)

    def f04_missing_temp(root):
        provenance = json.loads((root / "s5_fault_provenance.json").read_text(encoding="utf-8"))
        (root / provenance["temporary_artifact"]).unlink()

    _contract_failure(tmp_path, monkeypatch, "F04_NEXT_TEMP_PRE_ATOMIC_RENAME", f04_missing_temp)

    def f04_extra_temp(root):
        (root / "next" / "screen_000001.npz.unrelated.tmp").write_text("not-authoritative", encoding="utf-8")

    _contract_failure(tmp_path, monkeypatch, "F04_NEXT_TEMP_PRE_ATOMIC_RENAME", f04_extra_temp)

    def f05_barrier_pass(root):
        StreamingLifecycle.open(root).validate_barrier(actor="negative")

    _contract_failure(tmp_path, monkeypatch, "F05_NEXT_COMMITTED_PRE_BARRIER", f05_barrier_pass)

    def f05_promotion_present(root):
        lifecycle = StreamingLifecycle.open(root)
        lifecycle.validate_barrier(actor="negative")
        lifecycle.promote_next_to_current(actor="negative")

    _contract_failure(tmp_path, monkeypatch, "F05_NEXT_COMMITTED_PRE_BARRIER", f05_promotion_present)

    def f06_promotion_present(root):
        finalize_streaming(lifecycle_root=root, out_path=root.parent / "f06-recovered.json")

    _contract_failure(tmp_path, monkeypatch, "F06_BARRIER_PASS_PRE_PROMOTION", f06_promotion_present)

    def provenance_mismatch(root):
        marker = root / ".s5_fault_consumed.json"
        value = json.loads(marker.read_text(encoding="utf-8"))
        value["fault_id"] = "F99_TAMPERED"
        marker.write_text(json.dumps(value), encoding="utf-8")

    _contract_failure(tmp_path, monkeypatch, "F01_OPTICAL_PRE_POST_COMMIT", provenance_mismatch)
    root, _ = _faulted_lifecycle(tmp_path, monkeypatch, "F01_OPTICAL_PRE_POST_COMMIT")
    assert validate_fault_contract(lifecycle_root=root, fault_id="F01_OPTICAL_PRE_POST_COMMIT", target_screen="z00001")["status"] == "FAIL"


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


@pytest.mark.parametrize("fault_id", ["F03_HYDRO_PRE_NEXT_COMMIT", "F04_NEXT_TEMP_PRE_ATOMIC_RENAME"])
def test_recovery_provenance_requires_exact_target_retry_for_f03_and_f04(tmp_path, monkeypatch, fault_id):
    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, fault_id)
    lifecycle = _lifecycle(tmp_path, f"provenance-{fault_id}")
    _prepare_posts(lifecycle)
    if fault_id == "F03_HYDRO_PRE_NEXT_COMMIT":
        import KHz_filament.hr4e5s_streaming as streaming

        monkeypatch.setattr(
            streaming,
            "advance_hr4_single_screen",
            lambda delta_n, vx, vy, **kwargs: {"delta_n": delta_n, "vx": vx, "vy": vy},
        )
        with pytest.raises(S5FaultInjectedError, match=fault_id):
            lifecycle.run_one_hydro_block(dt_hydro=1.0e-6, n_hydro_steps=1, chi=0.0, nu=0.0, n0=1.0, gravity_y=0.0)
    else:
        assert lifecycle.claim_block(actor="provenance") == list(range(8))
        with pytest.raises(S5FaultInjectedError, match=fault_id):
            lifecycle.commit_next(0, _post_fields(lifecycle, 0), actor="provenance")
    recovered = StreamingLifecycle.open(lifecycle.root)
    assert recovered.reconstruct_queue(actor="provenance") == list(range(8))
    finished = _finish(recovered)
    result = validate_recovery_provenance(
        reference_lifecycle_root=clean.root,
        candidate_lifecycle_root=finished.root,
        fault_id=fault_id,
        target_screen="z00000",
    )
    assert result["status"] == "PASS", result
    assert result["retry_deltas"] == {ordinal: (1 if ordinal == 0 else 0) for ordinal in range(8)}
    assert [attempt["ordinal"] for attempt in result["recovery_attempts"]] == [0]


def test_recovery_provenance_rejects_unrelated_or_wrong_retry_count(tmp_path, monkeypatch):
    clean = _clean_reference(tmp_path)
    _enable(monkeypatch, "F03_HYDRO_PRE_NEXT_COMMIT")
    lifecycle = _lifecycle(tmp_path, "provenance-negative")
    _prepare_posts(lifecycle)
    import KHz_filament.hr4e5s_streaming as streaming

    monkeypatch.setattr(
        streaming,
        "advance_hr4_single_screen",
        lambda delta_n, vx, vy, **kwargs: {"delta_n": delta_n, "vx": vx, "vy": vy},
    )
    with pytest.raises(S5FaultInjectedError):
        lifecycle.run_one_hydro_block(dt_hydro=1.0e-6, n_hydro_steps=1, chi=0.0, nu=0.0, n0=1.0, gravity_y=0.0)
    recovered = StreamingLifecycle.open(lifecycle.root)
    recovered.reconstruct_queue(actor="provenance")
    finished = _finish(recovered)
    finished.manifest["records"][1]["retry_count"] = 1
    finished._save()
    unrelated = validate_recovery_provenance(
        reference_lifecycle_root=clean.root,
        candidate_lifecycle_root=finished.root,
        fault_id="F03_HYDRO_PRE_NEXT_COMMIT",
        target_screen="z00000",
    )
    assert unrelated["status"] == "FAIL"
    finished.manifest["records"][1]["retry_count"] = 0
    finished.manifest["records"][0]["retry_count"] = 2
    finished._save()
    wrong_target_count = validate_recovery_provenance(
        reference_lifecycle_root=clean.root,
        candidate_lifecycle_root=finished.root,
        fault_id="F03_HYDRO_PRE_NEXT_COMMIT",
        target_screen="z00000",
    )
    assert wrong_target_count["status"] == "FAIL"


@pytest.mark.parametrize("fault_id", [
    "F01_OPTICAL_PRE_POST_COMMIT", "F02_POST_COMMITTED_PRE_HYDRO",
    "F05_NEXT_COMMITTED_PRE_BARRIER", "F06_BARRIER_PASS_PRE_PROMOTION",
])
def test_recovery_provenance_rejects_retry_for_zero_retry_faults(tmp_path, monkeypatch, fault_id):
    monkeypatch.delenv("HR4_S5_FAULT_ID", raising=False)
    clean = _clean_reference(tmp_path / "reference")
    candidate = _clean_reference(tmp_path / "candidate")
    candidate.manifest["records"][0]["retry_count"] = 1
    candidate._save()
    result = validate_recovery_provenance(
        reference_lifecycle_root=clean.root,
        candidate_lifecycle_root=candidate.root,
        fault_id=fault_id,
        target_screen="z00000",
    )
    assert result["status"] == "FAIL"


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
    assert result["authoritative_manifest_exact"] is True
    assert result["recovery_provenance"]["status"] == "PASS"


@pytest.mark.parametrize("count", [15, 16, 17, 33])
def test_serial_recovery_bootstrap_refills_all_pending_posts_before_any_claim(tmp_path, count):
    lifecycle = _recovery_lifecycle(tmp_path, f"bootstrap-{count}", count=count, queue_depth=16)
    _commit_durable_posts_without_queue(lifecycle)
    assert lifecycle.reconstruct_queue(actor="precrash") == list(range(min(16, count)))
    assert lifecycle.claim_block(actor="hydro_consumer") == list(range(8))
    lifecycle.begin_hydro_screen(0, actor="hydro_consumer")

    receipt_path = tmp_path / f"bootstrap-{count}.json"
    receipt = bootstrap_recovery(
        lifecycle_root=lifecycle.root, out_path=receipt_path,
        runtime_sha="a" * 40, case_id="F03",
    )
    assert receipt["pending_post_count"] == count
    assert receipt["queue_size"] == min(16, count)
    assert receipt["backlog_size"] == max(0, count - 16)
    assert receipt["stale_hydro_running_count_before"] == 1
    assert receipt["reconstructed_stale_hydro_running_count"] == 1
    assert receipt["pre_bootstrap_hydro_claim_count"] == 1
    validate_recovery_bootstrap_receipt(
        receipt_path=receipt_path, lifecycle_root=lifecycle.root,
        runtime_sha="a" * 40, case_id="F03",
    )

    recovered = StreamingLifecycle.open(lifecycle.root)
    events = recovered.manifest["telemetry_events"]
    bootstrap_index = receipt["telemetry_event_index"]
    assert events[bootstrap_index]["event"] == "RESTART_RECONSTRUCTED"
    claimed = recovered.claim_block(actor="hydro_consumer")
    assert claimed == list(range(min(8, count)))
    claimed_events = [
        index for index, event in enumerate(recovered.manifest["telemetry_events"])
        if event["event"] == "HYDRO_CLAIM" and event["actor"] == "hydro_consumer"
    ]
    assert len(claimed_events) == 2
    assert claimed_events[0] < bootstrap_index < claimed_events[1]
    with pytest.raises(ValueError, match="telemetry length"):
        validate_recovery_bootstrap_receipt(
            receipt_path=receipt_path, lifecycle_root=lifecycle.root,
            runtime_sha="a" * 40, case_id="F03",
        )
    validate_recovery_bootstrap_receipt(
        receipt_path=receipt_path, lifecycle_root=lifecycle.root,
        runtime_sha="a" * 40, case_id="F03",
        require_current_telemetry_count=False,
    )
    with pytest.raises(ValueError, match="already recorded"):
        bootstrap_recovery(
            lifecycle_root=lifecycle.root, out_path=tmp_path / f"second-{count}.json",
            runtime_sha="a" * 40, case_id="F03",
        )


def test_serial_bootstrap_preserves_f05_target_next_and_never_retries_it(tmp_path, monkeypatch):
    _enable(monkeypatch, "F05_NEXT_COMMITTED_PRE_BARRIER", "z00007")
    lifecycle = _recovery_lifecycle(tmp_path, "bootstrap-f05", count=17, queue_depth=16)
    _commit_durable_posts_without_queue(lifecycle)
    assert lifecycle.reconstruct_queue(actor="fault") == list(range(16))
    assert lifecycle.claim_block(actor="fault") == list(range(8))
    for ordinal in range(7):
        lifecycle.commit_next(ordinal, _post_fields(lifecycle, ordinal), actor="fault")
    with pytest.raises(S5FaultInjectedError, match="F05_NEXT_COMMITTED_PRE_BARRIER"):
        lifecycle.commit_next(7, _post_fields(lifecycle, 7), actor="fault")

    receipt = bootstrap_recovery(
        lifecycle_root=lifecycle.root, out_path=tmp_path / "f05-bootstrap.json",
        runtime_sha="b" * 40, case_id="F05",
    )
    recovered = StreamingLifecycle.open(lifecycle.root)
    target = recovered.manifest["records"][7]
    assert target["state"] == "NEXT_COMMITTED" and target["next"] is not None and target["retry_count"] == 0
    assert 7 not in recovered.manifest["queue"] and 7 not in recovered.manifest["recovery_backlog"]
    assert receipt["stale_hydro_running_count_before"] == 0


def test_monitor_distinguishes_fault_history_from_recovery_claims(tmp_path):
    monitor = _monitor_module()
    case_root = tmp_path / "F03"
    lifecycle_root = case_root / "injected" / "lifecycle"
    lifecycle_root.mkdir(parents=True)
    receipt_path = case_root / "recovery" / "restart_reconstructed.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps({
        "schema": "khz_filament.hr4e5s.s5.recovery_bootstrap.v1", "status": "PASS",
        "bootstrap_event": "RESTART_RECONSTRUCTED", "runtime_sha": "c" * 40, "case_id": "F03",
        "telemetry_event_index": 1, "pre_bootstrap_hydro_claim_count": 1,
    }), encoding="utf-8")
    (lifecycle_root / "streaming_manifest.json").write_text(json.dumps({"telemetry_events": [
        {"event": "HYDRO_CLAIM", "actor": "hydro_consumer"},
        {"event": "RESTART_RECONSTRUCTED", "actor": "s5_restart", "bootstrap": True},
    ]}), encoding="utf-8")
    passed, detail = monitor._recovery_bootstrap_order(case_root, expected_sha="c" * 40, case_id="F03")
    assert not passed and detail["reason"] == "RECOVERY_BOOTSTRAP_ORDER_VIOLATION"
    (lifecycle_root / "streaming_manifest.json").write_text(json.dumps({"telemetry_events": [
        {"event": "HYDRO_CLAIM", "actor": "hydro_consumer"},
        {"event": "RESTART_RECONSTRUCTED", "actor": "s5_restart", "bootstrap": True},
        {"event": "HYDRO_CLAIM", "actor": "hydro_consumer"},
    ]}), encoding="utf-8")
    passed, detail = monitor._recovery_bootstrap_order(case_root, expected_sha="c" * 40, case_id="F03")
    assert passed and detail["historical_hydro_claim_count"] == 1
    assert detail["first_recovery_hydro_claim_index"] == 2
    assert detail["bootstrap_before_first_recovery_claim"] is True


def test_s5_submit_wrapper_pins_batch_workdir_to_run_root():
    submit = (Path(__file__).resolve().parents[1] / "tools" / "hpc_ops" / "submit_hr4e5s_s5.sh").read_text(encoding="utf-8")
    assert '--chdir="$RUN_ROOT"' in submit
    assert '"$RUN_ROOT/${CASE_ID}_fault_submission_receipt.tsv"' in submit
    assert '"$RUN_ROOT/${CASE_ID}_recovery_submission_receipt.tsv"' in submit
    assert 'CASE_MODE" == recovery' in submit
    assert 'reference_receipt="$(dirname -- "$REFERENCE_CASE_ROOT")/clean_submission_receipt.tsv"' in submit


def test_s5_recovery_creates_parent_once_before_consumer_and_never_uses_mkdir_p():
    batch = (Path(__file__).resolve().parents[1] / "tools" / "hr4e5s_s5.sbatch").read_text(encoding="utf-8")
    recovery = batch.index('if [[ "$CASE_MODE" == recovery ]]')
    parent = batch.index('mkdir -m 700 -- "$CASE_ROOT/recovery"', recovery)
    bootstrap = batch.index('bootstrap-recovery --stream-root "$CASE_ROOT/injected/lifecycle"', recovery)
    validate = batch.index('validate-bootstrap --stream-root "$CASE_ROOT/injected/lifecycle"', recovery)
    pair = batch.index('pair_run "$CASE_ROOT/injected/lifecycle" "$CASE_ROOT/recovery/optical"', recovery)
    contract = batch.index("check['contract_match'] is True", recovery)
    consumer = batch.index('mkdir -m 700 -- "$consumer"')
    assert contract < parent
    assert parent < bootstrap < validate < pair
    assert consumer < pair
    assert "\n  mkdir -p" not in batch
    assert '--bootstrap-receipt "$bootstrap_receipt"' in batch


def _write_monitor_manifest(tmp_path, monitor, *, case_ids=None):
    run_root = tmp_path / "rerun"
    run_root.mkdir()
    cases = [
        {"case_id": f"F0{index}", "fault_id": fault_id, "fault_screen": "z00000" if index < 5 else "z00007"}
        for index, fault_id in enumerate((
            "F01_OPTICAL_PRE_POST_COMMIT", "F02_POST_COMMITTED_PRE_HYDRO", "F03_HYDRO_PRE_NEXT_COMMIT",
            "F04_NEXT_TEMP_PRE_ATOMIC_RENAME", "F05_NEXT_COMMITTED_PRE_BARRIER",
            "F06_BARRIER_PASS_PRE_PROMOTION",
        ), start=1)
    ]
    if case_ids is not None:
        cases = [case for case in cases if case["case_id"] in set(case_ids)]
    manifest = {
        "schema": monitor.SCHEMA, "run_root": str(run_root), "repo": "/repo", "expected_sha": "a" * 40,
        "preflight": str(run_root / "preflight.json"), "reference_case_root": str(run_root / "clean"),
        "submit_script": "/submit_hr4e5s_s5.sh", "cases": cases,
        "clean_reference_job": "7000", "historical_clean_reference_job": "238465",
    }
    path = run_root / "matrix_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_s5_monitor_allows_only_the_ordered_f03_f05_repair_subset(tmp_path):
    monitor = _monitor_module()
    path, manifest = _write_monitor_manifest(tmp_path, monitor, case_ids=("F03", "F04", "F05"))
    assert tuple(case["case_id"] for case in monitor.load_manifest(path)["cases"]) == ("F03", "F04", "F05")
    state = monitor._fresh_state(path, manifest)
    monitor._atomic_json(Path(manifest["run_root"]) / "monitor_state.json", state)
    assert set(monitor.load_or_create_state(path, manifest)["cases"]) == {"F03", "F04", "F05"}


def _monitor_receipt(path, case_id, stage, sha, job_id):
    path.write_text(
        "case_id\tcase_mode\tfault_id\tfault_screen\tjob_id\toptical_gpus\thydro_gpus\texecution_sha\n"
        f"{case_id}\t{stage}\t\t\t{job_id}\t1\t1\t{sha}\n",
        encoding="utf-8",
    )


def test_s5_monitor_resume_reuses_known_fault_and_submits_recovery_only_after_contract_pass(tmp_path):
    monitor = _monitor_module()
    manifest_path, manifest = _write_monitor_manifest(tmp_path, monitor)
    run_root = Path(manifest["run_root"])
    _monitor_receipt(run_root / "F01_fault_submission_receipt.tsv", "F01", "fault", manifest["expected_sha"], "7001")
    state = monitor._fresh_state(manifest_path, manifest)
    for case_id in ("F02", "F03", "F04", "F05", "F06"):
        state["cases"][case_id]["state"] = "PASS"
    monitor._atomic_json(run_root / "monitor_state.json", state)
    calls = []

    def submitter(argv, cwd):
        calls.append(list(argv))
        case_id, stage = argv[5], argv[6]
        _monitor_receipt(monitor._receipt_path(cwd, case_id, stage), case_id, stage, manifest["expected_sha"], "7002")
        return subprocess.CompletedProcess(argv, 0, "submitted", "")

    monitor.advance_matrix(manifest_path, submitter=submitter)
    assert calls == []
    resumed = monitor._read_json(run_root / "monitor_state.json")
    assert resumed["cases"]["F01"]["state"] == "SUBMITTED_FAULT"

    injected = run_root / "F01" / "injected"
    injected.mkdir(parents=True)
    (injected / "disk_state_audit.json").write_text(json.dumps({"fault_provenance": []}), encoding="utf-8")
    (run_root / "F01" / "contract_check.json").write_text(json.dumps({
        "status": "PASS", "contract_match": True, "fault_id": manifest["cases"][0]["fault_id"], "target_screen": "z00000",
    }), encoding="utf-8")
    resumed["cases"]["F01"]["state"] = "FAULT_AUDIT_PENDING"
    monitor._atomic_json(run_root / "monitor_state.json", resumed)
    monitor.advance_matrix(manifest_path, submitter=submitter)
    assert calls == []
    monitor.advance_matrix(manifest_path, submitter=submitter)
    assert [call[6] for call in calls] == ["recovery"]
    monitor.advance_matrix(
        manifest_path,
        submitter=submitter,
        scheduler_query=lambda job: {"state": "RUNNING", "terminal": False, "job_id": job},
    )
    assert [call[6] for call in calls] == ["recovery"]

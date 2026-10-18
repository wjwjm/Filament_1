"""Negative tests for v2 ownership, cursor and transition authority bindings."""
import hashlib
import json

import numpy as np
import pytest


def fixture(root, count=8):
    from KHz_filament.hr4e5s_streaming import FIELDS, StreamingLifecycle
    arrays = {name: np.zeros((count, 2, 2), dtype=np.float64) for name in FIELDS}
    life = StreamingLifecycle.create(root=root, current=arrays,
        screen_records=[dict(ordinal=i, screen_id=f"s{i}", z_m=i*1e-4) for i in range(count)],
        current_generation="fencing:p0", dx_m=1e-4, dy_m=1e-4)
    for i in range(count):
        life.deposition_finalized(i)
        life.commit_post(i, {name: value[i] for name, value in arrays.items()})
    return life, {name: value[0] for name, value in arrays.items()}


def rewrite_last(root, mutate):
    journal = root / "streaming_manifest.journal.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    tx = json.loads(lines[-1]); mutate(tx); tx.pop("transaction_sha256")
    canonical = lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    tx["transaction_sha256"] = hashlib.sha256(canonical(tx)).hexdigest()
    lines[-1] = canonical(tx) + b"\n"; journal.write_bytes(b"".join(lines))
    headpath = root / "streaming_manifest.head.json"; head = json.loads(headpath.read_text())
    head.update(transaction_sha256=tx["transaction_sha256"], committed_offset=journal.stat().st_size)
    headpath.write_text(json.dumps(head), encoding="utf-8")


def test_same_role_reopen_and_old_claim_after_recovery_are_fenced(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError
    root = tmp_path / "life"; old, fields = fixture(root)
    for i in range(8): old.enqueue_post(i)
    assert old.claim_block(actor="H0") == list(range(8))
    fresh = StreamingLifecycle.open(root)
    with pytest.raises(StreamingLifecycleError, match="durable claim"):
        fresh.begin_hydro_screen(0, actor="H0")
    fresh.reconstruct_queue()
    assert fresh.claim_block(actor="H0") == list(range(8))
    with pytest.raises(StreamingLifecycleError, match="durable claim"):
        old.commit_next(0, fields, actor="H0")
    fresh.commit_next(0, fields, actor="H0")
    assert 0 not in fresh._held_claims
    assert StreamingLifecycle.open(root).manifest["records"][0]["state"] == "NEXT_COMMITTED"


def test_process_identity_reuse_is_fenced(tmp_path, monkeypatch):
    import KHz_filament.hr4e5s_streaming as module
    life, _ = fixture(tmp_path / "life")
    for i in range(8): life.enqueue_post(i)
    life.claim_block(actor="H0")
    original = module.process_identity()
    monkeypatch.setattr(module, "process_identity", lambda: {**original, "process_start_token": "different-start"})
    with pytest.raises(module.StreamingLifecycleError, match="durable claim"):
        life.begin_hydro_screen(0, actor="H0")


def test_cursor_jump_rejected_before_durable_publication(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError
    root = tmp_path / "life"; life, _ = fixture(root, count=32)
    life.reconstruct_queue()
    before = life.authority_identity()
    with pytest.raises(StreamingLifecycleError, match="cursor skipped"):
        with life._locked_manifest("claim_block"):
            life.manifest["recovery_backlog_cursor"] += 8
            life._save()
    assert StreamingLifecycle.open(root).authority_identity() == before


def test_rehashed_cursor_skip_rejected_on_fresh_replay(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError
    root = tmp_path / "life"; life, _ = fixture(root, count=32)
    life.reconstruct_queue(); life.claim_block(actor="H0")
    def mutate(tx):
        cursor_ops = [op for op in tx["operations"] if op["path"] == ["recovery_backlog_cursor"]]
        assert cursor_ops[-1]["value"] == 24
        cursor_ops[-1]["value"] = 32
    rewrite_last(root, mutate)
    with pytest.raises(StreamingLifecycleError, match="cursor skipped"):
        StreamingLifecycle.open(root)


def test_rehashed_queue_removal_cannot_orphan_an_unclaimed_screen(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError
    root = tmp_path / "life"; life, _ = fixture(root, count=16)
    for i in range(16): life.enqueue_post(i)
    life.claim_block(actor="H0")
    def mutate(tx):
        op = next(op for op in tx["operations"] if op["path"] == ["queue"] and op["op"] == "set")
        assert op["value"] == list(range(8,16))
        op["value"] = list(range(9,16))
    rewrite_last(root, mutate)
    with pytest.raises(StreamingLifecycleError, match="lost a queued"):
        StreamingLifecycle.open(root)


def test_bound_pre_read_serializes_generation_check_with_promotion(tmp_path, monkeypatch):
    import threading
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError
    from KHz_filament.hr4e5_formal_entry import StreamingPulseReadView
    root = tmp_path / "life"; life, fields = fixture(root)
    view = StreamingPulseReadView(root)
    for i in range(8): life.enqueue_post(i)
    life.claim_block(actor="H0")
    for i in range(8): life.commit_next(i, {key:value+50.0 for key,value in fields.items()}, actor="H0")
    life.validate_barrier()
    reading = threading.Event(); promoted = threading.Event(); errors=[]
    original = view._lifecycle._current_fields
    def guarded_read(index):
        reading.set()
        assert not promoted.wait(.05), 'promotion passed a held PRE read transaction'
        return original(index)
    monkeypatch.setattr(view._lifecycle, '_current_fields', guarded_read)
    def promote():
        try:
            assert reading.wait(2)
            StreamingLifecycle.open(root).promote_next_to_current()
            promoted.set()
        except BaseException as error: errors.append(error)
    thread=threading.Thread(target=promote); thread.start()
    np.testing.assert_array_equal(view.read_interval(0),fields['delta_n'])
    thread.join(3); assert not thread.is_alive() and not errors and promoted.is_set()
    with pytest.raises(StreamingLifecycleError, match='promoted'):
        view.read_interval(1)


def test_legacy_pre_view_rereads_identity_changes(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError
    from KHz_filament.hr4e5_formal_entry import StreamingPulseReadView
    root=tmp_path/'life'; life,_=fixture(root)
    snapshot=life.authority_snapshot()['manifest']; snapshot.pop('persistence_format')
    checkpoint=root/'streaming_manifest.json'; checkpoint.chmod(0o600)
    checkpoint.write_text(json.dumps(snapshot),encoding='utf-8')
    for name in ('streaming_manifest.head.json','streaming_manifest.journal.jsonl'):(root/name).unlink()
    legacy=StreamingLifecycle.open(root); view=StreamingPulseReadView(legacy)
    snapshot['current_generation']='different-generation'; checkpoint.write_text(json.dumps(snapshot),encoding='utf-8')
    with pytest.raises(StreamingLifecycleError,match='identity changed'):
        view.read_interval(0)


def test_bound_pre_read_uses_one_lifecycle_lock(tmp_path, monkeypatch):
    import KHz_filament.hr4e5s_streaming as module
    from KHz_filament.hr4e5_formal_entry import StreamingPulseReadView
    life,_=fixture(tmp_path/'life'); view=StreamingPulseReadView(life)
    entered=[]; original=module._ManifestFileLock.__enter__
    def enter(lock):
        entered.append(lock.operation); return original(lock)
    monkeypatch.setattr(module._ManifestFileLock,'__enter__',enter)
    view.read_interval(0)
    assert entered == ['bound_current_fields']


@pytest.mark.parametrize("field,value", [("ordinal", 7), ("screen_id", "wrong"),
    ("current_generation", "wrong"), ("status", "FAIL"), ("source_file_sha256", "f"*64), ("actor", "H3")])
def test_rehashed_transition_binding_rejected(tmp_path, field, value):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError
    root = tmp_path / "life"; life, _ = fixture(root)
    for i in range(8): life.enqueue_post(i)
    life.claim_block(actor="H0")
    def mutate(tx):
        event = next(op["value"] for op in tx["operations"] if op["path"] == ["records", 0, "transitions"])
        event[field] = value
    rewrite_last(root, mutate)
    with pytest.raises(StreamingLifecycleError, match="transition"):
        StreamingLifecycle.open(root)

"""Adversarial checks for the v2 streaming journal's durable boundaries.

These tests exercise the durable authority and public lifecycle API.  They do
not run optical or hydro solvers; tiny float64 artifacts are used only to
exercise persisted POST/NEXT contracts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _checkpoint() -> dict:
    return {
        "schema": "khz_filament.hr4e5s.streaming.v1",
        "expected_screen_count": 1,
        "recovery_backlog_cursor": 0,
        "records": [{"ordinal": 0, "screen_id": "s0", "transitions": []}],
        "recovery_backlog": [],
        "recovery_attempts": [],
        "rate_events": [],
        "telemetry_events": [],
        "s5_recovery_events": [],
    }


def _initialize_journal(tmp_path: Path):
    from KHz_filament.hr4e5_streaming_journal import JournalAuthority

    root = tmp_path / "journal"
    root.mkdir(parents=True)
    checkpoint = root / "streaming_manifest.json"
    raw = (json.dumps(_checkpoint(), sort_keys=True, indent=2) + "\n").encode("utf-8")
    checkpoint.write_bytes(raw)
    authority = JournalAuthority.initialize(root, raw)
    return root, authority


def _commit_event(authority, event: str):
    from KHz_filament.hr4e5_streaming_journal import _ChangeSet

    changes = _ChangeSet()
    document, head, _ = authority.load(changes=changes)
    document["telemetry_events"].append({"event": event})
    return authority.commit(
        changes.ops,
        previous=head,
        operation="record_telemetry",
        actor="adversarial-test",
        process={"pid": os.getpid()},
    )


def _head(root: Path) -> dict:
    return json.loads((root / "streaming_manifest.head.json").read_text(encoding="utf-8"))


def _write_head(root: Path, value: dict) -> None:
    (root / "streaming_manifest.head.json").write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def test_failure_before_journal_append_keeps_previous_head_usable(tmp_path, monkeypatch):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _initialize_journal(tmp_path)
    journal_path = root / "streaming_manifest.journal.jsonl"
    before = _head(root)
    original_open = journal_module.os.open

    def fail_append(path, flags, *args, **kwargs):
        if Path(path) == journal_path and flags & os.O_APPEND:
            raise OSError("injected failure before append")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(journal_module.os, "open", fail_append)
    with pytest.raises(journal_module.JournalError, match="durable journal append failed"):
        _commit_event(authority, "must-not-commit")

    assert journal_path.stat().st_size == 0
    assert _head(root) == before
    replayed, replayed_head, _ = journal_module.JournalAuthority(root).load()
    assert replayed_head.sequence == 0
    assert replayed["telemetry_events"] == []


def test_crash_after_append_fsync_before_head_fails_closed(tmp_path, monkeypatch):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _initialize_journal(tmp_path)

    def crash_before_head(*_args, **_kwargs):
        raise RuntimeError("simulated crash after journal fsync, before HEAD")

    monkeypatch.setattr(journal_module, "_atomic_head", crash_before_head)
    with pytest.raises(RuntimeError, match="after journal fsync"):
        _commit_event(authority, "unacknowledged")

    assert (root / "streaming_manifest.journal.jsonl").stat().st_size > 0
    assert _head(root)["sequence"] == 0
    with pytest.raises(journal_module.JournalError, match="size differs from committed HEAD"):
        journal_module.JournalAuthority(root).load()


def test_error_after_head_replace_is_commit_uncertain_but_replayable(tmp_path, monkeypatch):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _initialize_journal(tmp_path)
    original_fsync_dir = journal_module._fsync_dir_strict

    def fail_directory_fsync(path):
        raise OSError("simulated failure after HEAD replace, before directory fsync")

    monkeypatch.setattr(journal_module, "_fsync_dir_strict", fail_directory_fsync)
    with pytest.raises(OSError, match="before directory fsync"):
        _commit_event(authority, "commit-uncertain")

    # HEAD replacement is the visibility boundary.  A caller that saw the
    # fsync error must reconcile from this authority before it can continue.
    replayed, head, _ = journal_module.JournalAuthority(root).load()
    assert head.sequence == 1
    assert replayed["telemetry_events"] == [{"event": "commit-uncertain"}]
    monkeypatch.setattr(journal_module, "_fsync_dir_strict", original_fsync_dir)


@pytest.mark.parametrize(
    ("operation", "path", "value", "expected"),
    [
        ("record_telemetry", ["records", 0, "transitions"], [], "record transitions are append-only"),
        ("record_telemetry", ["telemetry_events"], [], "append-only lifecycle history cannot be replaced"),
        ("record_telemetry", ["rate_events"], [], "append-only lifecycle history cannot be replaced"),
        ("record_telemetry", ["recovery_attempts"], [], "append-only lifecycle history cannot be replaced"),
        ("record_telemetry", ["s5_recovery_events"], [], "append-only lifecycle history cannot be replaced"),
    ],
)
def test_forged_history_replacement_is_rejected_before_head_commit(
    tmp_path, operation, path, value, expected
):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _initialize_journal(tmp_path)
    changes = journal_module._ChangeSet()
    document, previous, _ = authority.load(changes=changes)
    parent = document
    for segment in path[:-1]:
        parent = parent[segment]
    final = path[-1]
    existed = final in parent if isinstance(parent, dict) else 0 <= final < len(parent)
    predecessor = parent[final] if existed else None
    forged = {
        "op": "set",
        "path": path,
        "value": value,
        "existed": existed,
        "previous": journal_module._plain(predecessor) if existed else None,
    }

    with pytest.raises(journal_module.JournalError, match=expected):
        authority.commit(
            [forged], previous=previous, operation=operation,
            actor="forged-test", process={"pid": os.getpid()},
        )

    assert _head(root)["sequence"] == 0
    assert (root / "streaming_manifest.journal.jsonl").stat().st_size == 0


@pytest.mark.parametrize("tamper", ["sequence", "lineage"])
def test_rehashed_transaction_with_wrong_sequence_or_lineage_is_rejected(tmp_path, tamper):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _initialize_journal(tmp_path)
    _commit_event(authority, "valid")
    journal_path = root / "streaming_manifest.journal.jsonl"
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction.pop("transaction_sha256")
    if tamper == "sequence":
        transaction["sequence"] = 2
        expected = "sequence, lineage, or fields are invalid"
    else:
        transaction["lineage_id"] = "f" * 32
        expected = "sequence, lineage, or fields are invalid"
    digest = hashlib.sha256(_canonical(transaction)).hexdigest()
    transaction["transaction_sha256"] = digest
    line = _canonical(transaction) + b"\n"
    journal_path.write_bytes(line)
    head = _head(root)
    head["committed_offset"] = len(line)
    head["transaction_sha256"] = digest
    if tamper == "sequence":
        head["sequence"] = 2
    _write_head(root, head)

    with pytest.raises(journal_module.JournalError, match=expected):
        journal_module.JournalAuthority(root).load()


def test_torn_committed_record_and_same_size_checksum_tamper_fail_closed(tmp_path):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _initialize_journal(tmp_path)
    _commit_event(authority, "payload")
    journal_path = root / "streaming_manifest.journal.jsonl"
    original = journal_path.read_bytes()
    torn = original[:-1]
    journal_path.write_bytes(torn)
    head = _head(root)
    head["committed_offset"] = len(torn)
    _write_head(root, head)
    with pytest.raises(journal_module.JournalError, match="oversized, truncated, or malformed transaction"):
        journal_module.JournalAuthority(root).load()

    # A separate root keeps this case independent of the truncated evidence.
    other, other_authority = _initialize_journal(tmp_path / "checksum")
    _commit_event(other_authority, "payload")
    other_journal = other / "streaming_manifest.journal.jsonl"
    line = other_journal.read_bytes()
    assert b"payload" in line
    changed = line.replace(b"payload", b"payloae", 1)
    assert len(changed) == len(line)
    other_journal.write_bytes(changed)
    with pytest.raises(journal_module.JournalError, match="checksum mismatch"):
        journal_module.JournalAuthority(other).load()


def test_stale_writer_and_same_size_journal_replacement_are_rejected(tmp_path):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, first = _initialize_journal(tmp_path)
    first_changes = journal_module._ChangeSet()
    _, stale_head, _ = first.load(changes=first_changes)
    second = journal_module.JournalAuthority(root)
    _commit_event(second, "winner")
    with pytest.raises(journal_module.JournalError, match="stale lifecycle HEAD"):
        first.commit(
            [{"op": "append", "path": ["telemetry_events"], "value": {"event": "loser"}}],
            previous=stale_head,
            operation="record_telemetry",
            actor="stale-writer",
            process={"pid": os.getpid()},
        )

    other, other_authority = _initialize_journal(tmp_path / "replacement")
    _commit_event(other_authority, "one-transaction")
    observer = journal_module.JournalAuthority(other)
    changes = journal_module._ChangeSet()
    document, head, _ = observer.load(changes=changes)
    journal = other / "streaming_manifest.journal.jsonl"
    replacement = other / "replacement.bytes"
    original = journal.read_bytes()
    replacement.write_bytes(original.replace(b"one-transaction", b"two-transaction", 1))
    assert replacement.stat().st_size == journal.stat().st_size
    os.replace(replacement, journal)
    with pytest.raises(journal_module.JournalError, match="inode changed"):
        observer.refresh(document, head, changes)


def test_checkpoint_rewrite_and_missing_checkpoint_do_not_fall_back(tmp_path):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _initialize_journal(tmp_path)
    checkpoint = root / "streaming_manifest.json"
    before = checkpoint.stat()
    original = checkpoint.read_bytes()
    altered = original.replace(b"s0", b"x0", 1)
    assert len(altered) == len(original)
    checkpoint.chmod(0o600)
    checkpoint.write_bytes(altered)
    os.utime(checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(journal_module.JournalError, match="checkpoint identity changed|checkpoint hash"):
        authority.load()

    other, other_authority = _initialize_journal(tmp_path / "missing")
    checkpoint = other / "streaming_manifest.json"
    checkpoint.chmod(0o600)
    checkpoint.unlink()
    with pytest.raises(journal_module.JournalError, match="checkpoint is missing|checkpoint identity changed"):
        other_authority.load()


def test_authority_symlink_and_hardlink_substitution_are_rejected(tmp_path):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, _ = _initialize_journal(tmp_path)
    outside = tmp_path / "outside-head.json"
    outside.write_bytes((root / "streaming_manifest.head.json").read_bytes())
    head = root / "streaming_manifest.head.json"
    head.unlink()
    try:
        head.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(journal_module.JournalError, match="private regular file"):
        journal_module.JournalAuthority(root).load()

    other, _ = _initialize_journal(tmp_path / "hardlink")
    (tmp_path / "outside-journal.jsonl").hardlink_to(other / "streaming_manifest.journal.jsonl")
    with pytest.raises(journal_module.JournalError, match="private regular file"):
        journal_module.JournalAuthority(other).load()


def test_wrong_owner_authority_file_is_rejected(tmp_path, monkeypatch):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, _ = _initialize_journal(tmp_path)
    head = root / "streaming_manifest.head.json"
    original_lstat = Path.lstat
    real_info = head.lstat()
    process_uid = getattr(os, "getuid", lambda: 1000)()

    class StatWithWrongOwner:
        def __getattr__(self, name):
            return getattr(real_info, name)

        @property
        def st_uid(self):
            return int(process_uid) + 1

    def forged_lstat(path):
        if path == head:
            return StatWithWrongOwner()
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", forged_lstat)
    monkeypatch.setattr(journal_module.os, "getuid", lambda: process_uid, raising=False)
    with pytest.raises(journal_module.JournalError, match="owner"):
        journal_module.JournalAuthority(root).load()


def _small_lifecycle(root: Path, count: int = 8):
    from KHz_filament.hr4e5s_streaming import FIELDS, StreamingLifecycle

    base = np.arange(count * 4, dtype=np.float64).reshape(count, 2, 2)
    current = {"delta_n": base.copy(), "vx": base + 10.0, "vy": base + 20.0}
    records = [
        {"ordinal": index, "screen_id": f"s{index}", "z_m": index * 1.0e-3}
        for index in range(count)
    ]
    life = StreamingLifecycle.create(
        root=root,
        current=current,
        screen_records=records,
        current_generation="adversarial:current",
        dx_m=1.0e-4,
        dy_m=1.0e-4,
        queue_depth=max(8, count),
        actor="adversarial-test",
    )
    return life, current


def _publish_all_posts(life, current, count: int = 8):
    for ordinal in range(count):
        life.deposition_finalized(ordinal)
        payload = {key: current[key][ordinal] for key in current}
        life.commit_post(ordinal, payload)
        life.enqueue_post(ordinal)


def test_fresh_replay_serializes_claim_and_prevents_duplicate_next(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError

    root = tmp_path / "lifecycle"
    writer, current = _small_lifecycle(root)
    _publish_all_posts(writer, current)

    stale = StreamingLifecycle.open(root)
    assert writer.claim_block(actor="worker-a") == list(range(8))
    assert stale.claim_block(actor="worker-b") == []

    next_fields = {
        "delta_n": current["delta_n"] + 100.0,
        "vx": current["vx"] + 100.0,
        "vy": current["vy"] + 100.0,
    }
    for ordinal in range(8):
        writer.begin_hydro_screen(ordinal, actor="worker-a")
        writer.commit_next(
            ordinal,
            {key: next_fields[key][ordinal] for key in next_fields},
            actor="worker-a",
        )

    fresh = StreamingLifecycle.open(root)
    assert fresh.claim_block(actor="worker-c") == []
    before = fresh.authority_identity(refresh=True)
    with pytest.raises(StreamingLifecycleError):
        fresh.commit_next(0, {key: next_fields[key][0] for key in next_fields}, actor="worker-c")
    assert fresh.authority_identity(refresh=True) == before

    assert fresh.validate_barrier()["status"] == "PASS"
    fresh.promote_next_to_current()
    np.testing.assert_array_equal(
        stale.current_fields(0)["delta_n"], next_fields["delta_n"][0]
    )
    snapshot = stale.authority_snapshot()
    assert snapshot["manifest"]["records"][0]["next"] is not None


def test_lifecycle_refresh_valid_then_corrupt_transaction_poisons_stale_object(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError

    root = tmp_path / "lifecycle"
    seed, _ = _small_lifecycle(root, count=1)
    stale = StreamingLifecycle.open(root)
    writer = StreamingLifecycle.open(root)
    writer.record_telemetry("valid-first", actor="writer")
    writer.record_telemetry("corrupt-second", actor="writer")

    journal = root / "streaming_manifest.journal.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    assert b"corrupt-second" in lines[1]
    lines[1] = lines[1].replace(b"corrupt-second", b"corrupt-seconD", 1)
    assert len(lines[1]) == len(journal.read_bytes().splitlines(keepends=True)[1])
    journal.write_bytes(b"".join(lines))

    with pytest.raises(StreamingLifecycleError, match="checksum mismatch"):
        stale.record_telemetry("must-not-append", actor="stale")
    assert stale._poisoned is True
    head_after_failure = stale._journal_head
    with pytest.raises(StreamingLifecycleError, match="poison"):
        stale.record_telemetry("must-stay-blocked", actor="stale")
    assert stale._journal_head == head_after_failure
    assert seed._journal_head.sequence == 0


def test_multisave_failure_preserves_first_durable_hydro_boundary(tmp_path, monkeypatch):
    import KHz_filament.hr4e5_streaming_journal as journal_module
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError

    root = tmp_path / "lifecycle"
    life, current = _small_lifecycle(root)
    _publish_all_posts(life, current)
    assert life.claim_block(actor="hydro") == list(range(8))

    original_open = journal_module.os.open
    append_count = 0
    journal_path = root / "streaming_manifest.journal.jsonl"

    def fail_second_append(path, flags, *args, **kwargs):
        nonlocal append_count
        if Path(path) == journal_path and flags & os.O_APPEND:
            append_count += 1
            if append_count == 2:
                raise OSError("injected before second save append")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(journal_module.os, "open", fail_second_append)
    payload = {key: current[key][0] + 50.0 for key in current}
    with pytest.raises(StreamingLifecycleError):
        life.commit_next(0, payload, actor="hydro")
    assert append_count == 2

    # The first save (hydro attempt started) is durable.  The second save did
    # not append, so it cannot replace that state with NEXT_COMMITTED.
    fresh = StreamingLifecycle.open(root)
    record = fresh.manifest["records"][0]
    assert record["state"] == "HYDRO_RUNNING"
    assert record["hydro_attempt_started"] is True
    assert record["next"] is None
    assert (root / "next" / "screen_000000.npz").is_file()

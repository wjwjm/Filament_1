from __future__ import annotations

import json
import hashlib
import os

import pytest


def _checkpoint() -> dict:
    return {
        "schema": "khz_filament.hr4e5s.streaming.v1",
        "expected_screen_count": 1,
        "recovery_backlog_cursor": 0,
        "records": [{"ordinal": 0, "screen_id": "s0"}],
        "telemetry_events": [],
    }


def _new_root(tmp_path):
    from KHz_filament.hr4e5_streaming_journal import JournalAuthority

    root = tmp_path / "journal"
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    checkpoint = root / "streaming_manifest.json"
    raw = (json.dumps(_checkpoint(), sort_keys=True, indent=2) + "\n").encode("utf-8")
    checkpoint.write_bytes(raw)
    authority = JournalAuthority.initialize(root, raw)
    return root, authority


def test_head_commit_follows_durable_append_and_replays(tmp_path, monkeypatch):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _new_root(tmp_path)
    changes = journal_module._ChangeSet()
    document, initial_head, _ = authority.load(changes=changes)
    document["telemetry_events"].append({"event": "committed", "ordinal": 0})
    original = journal_module._atomic_head
    observed = {}

    def checked_head(path, head):
        observed["journal_size_before_head"] = (root / "streaming_manifest.journal.jsonl").stat().st_size
        observed["head_offset"] = head["committed_offset"]
        return original(path, head)

    monkeypatch.setattr(journal_module, "_atomic_head", checked_head)
    new_head, io = authority.commit(
        changes.ops,
        previous=initial_head,
        operation="record_telemetry",
        actor="test",
        process={"pid": 1},
    )
    assert observed["journal_size_before_head"] > 0
    assert observed["journal_size_before_head"] == observed["head_offset"]
    assert io["journal_bytes_written"] == observed["head_offset"]
    replayed, reopened_head, _ = journal_module.JournalAuthority(root).load()
    assert reopened_head == new_head
    assert replayed["telemetry_events"] == [{"event": "committed", "ordinal": 0}]


def test_uncommitted_or_truncated_journal_tail_fails_closed(tmp_path):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, _ = _new_root(tmp_path)
    with (root / "streaming_manifest.journal.jsonl").open("ab") as handle:
        handle.write(b'{"uncommitted":true}\n')
        handle.flush()
    with pytest.raises(journal_module.JournalError, match="size differs from committed HEAD"):
        journal_module.JournalAuthority(root).load()


def test_initial_checkpoint_is_hash_bound_and_hardlinks_are_rejected(tmp_path):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, _ = _new_root(tmp_path)
    checkpoint = root / "streaming_manifest.json"
    checkpoint.chmod(0o600)
    checkpoint.write_text(json.dumps(_checkpoint()), encoding="utf-8")
    with pytest.raises(journal_module.JournalError, match="checkpoint hash"):
        journal_module.JournalAuthority(root).load()

    # Restore an independent root and prove a second directory entry for the
    # append-only authority is rejected before replay.
    other, _ = _new_root(tmp_path / "other")
    (tmp_path / "journal-hardlink.jsonl").hardlink_to(other / "streaming_manifest.journal.jsonl")
    with pytest.raises(journal_module.JournalError, match="private regular file"):
        journal_module.JournalAuthority(other).load()


def _commit_telemetry(authority, event: str):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    changes = journal_module._ChangeSet()
    document, head, _ = authority.load(changes=changes)
    document["telemetry_events"].append({"event": event})
    new_head, _ = authority.commit(
        changes.ops,
        previous=head,
        operation="record_telemetry",
        actor="journal-digest-test",
        process={"pid": os.getpid()},
    )
    return document, new_head, changes


def test_timestamp_only_change_is_accepted_after_full_digest_verification(tmp_path, caplog):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    caplog.set_level("WARNING", logger=journal_module.__name__)
    root, authority = _new_root(tmp_path)
    document, head, changes = _commit_telemetry(authority, "unchanged")
    journal = root / "streaming_manifest.journal.jsonl"
    expected = hashlib.sha256(journal.read_bytes()).hexdigest()
    before = journal.stat()
    os.utime(journal, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000))

    refreshed, io = authority.refresh(document, head, changes)

    assert refreshed == head
    assert io["journal_bytes_replayed"] == 0
    assert authority._journal_digest.hexdigest() == expected
    assert authority.journal_digest_recheck_count == 1
    assert authority.journal_digest_recheck_elapsed_s >= 0
    assert authority._journal_verified_head == head
    assert authority._journal_digest_offset == head.committed_offset
    marker_records = [record for record in caplog.records if "JOURNAL_METADATA_RECHECK" in record.getMessage()]
    assert len(marker_records) == 1
    payload = json.loads(marker_records[0].getMessage().split(" ", 1)[1])
    assert set(payload) == {
        "pid", "verified_sequence", "verified_offset", "reason", "count", "elapsed_s", "total_elapsed_s"
    }
    assert payload["reason"] == "metadata_changed_without_HEAD_advance"
    assert payload["verified_sequence"] == head.sequence
    assert payload["verified_offset"] == head.committed_offset


def test_same_head_same_size_content_corruption_fails_with_digest_context(tmp_path):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _new_root(tmp_path)
    document, head, changes = _commit_telemetry(authority, "integrity")
    journal = root / "streaming_manifest.journal.jsonl"
    original = journal.read_bytes()
    corrupted = original.replace(b"integrity", b"integriqy", 1)
    assert len(corrupted) == len(original)
    journal.write_bytes(corrupted)

    with pytest.raises(journal_module.JournalError) as caught:
        authority.refresh(document, head, changes)

    assert caught.value.category == "journal_digest_mismatch"
    assert caught.value.details["reason"] == "metadata_changed_without_HEAD_advance"
    assert caught.value.details["expected_sha256"] == hashlib.sha256(original).hexdigest()
    assert caught.value.details["actual_sha256"] == hashlib.sha256(corrupted).hexdigest()
    assert caught.value.details["head_before"] == caught.value.details["head_after"]


def test_metadata_recheck_before_commit_emits_one_fixed_warning(tmp_path, caplog):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    caplog.set_level("WARNING", logger=journal_module.__name__)
    root, authority = _new_root(tmp_path)
    changes = journal_module._ChangeSet()
    document, head, _ = authority.load(changes=changes)
    journal = root / "streaming_manifest.journal.jsonl"
    before = journal.stat()
    os.utime(journal, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000))
    document["telemetry_events"].append({"event": "commit-after-metadata-drift"})

    committed_head, _ = authority.commit(
        changes.ops,
        previous=head,
        operation="record_telemetry",
        actor="journal-digest-test",
        process={"pid": os.getpid()},
    )

    assert committed_head.sequence == head.sequence + 1
    marker_records = [record for record in caplog.records if "JOURNAL_METADATA_RECHECK" in record.getMessage()]
    assert len(marker_records) == 1
    payload = json.loads(marker_records[0].getMessage().split(" ", 1)[1])
    assert payload["reason"] == "metadata_changed_before_commit"
    assert payload["verified_sequence"] == head.sequence
    assert payload["verified_offset"] == head.committed_offset
    assert payload["count"] == 1


def test_load_accepts_metadata_drift_only_after_candidate_digest_confirmation(tmp_path, monkeypatch, caplog):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    caplog.set_level("WARNING", logger=journal_module.__name__)
    root, writer = _new_root(tmp_path)
    _, committed_head, _ = _commit_telemetry(writer, "load-drift")
    journal = root / "streaming_manifest.journal.jsonl"
    reopened = journal_module.JournalAuthority(root)
    original_decode = reopened._decode_transaction
    drifted = False

    def decode_then_drift(*args, **kwargs):
        nonlocal drifted
        transaction = original_decode(*args, **kwargs)
        if not drifted:
            before = journal.stat()
            os.utime(journal, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000))
            drifted = True
        return transaction

    monkeypatch.setattr(reopened, "_decode_transaction", decode_then_drift)
    _, loaded_head, _ = reopened.load()

    assert drifted
    assert loaded_head == committed_head
    assert reopened._journal_digest.hexdigest() == hashlib.sha256(journal.read_bytes()).hexdigest()
    marker_records = [record for record in caplog.records if "JOURNAL_METADATA_RECHECK" in record.getMessage()]
    assert len(marker_records) == 1
    assert json.loads(marker_records[0].getMessage().split(" ", 1)[1])["reason"] == "metadata_changed_during_load"


def test_delta_refresh_accepts_metadata_drift_only_after_candidate_digest_confirmation(tmp_path, monkeypatch, caplog):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    caplog.set_level("WARNING", logger=journal_module.__name__)
    root, observer = _new_root(tmp_path)
    document, first_head, changes = _commit_telemetry(observer, "first")
    writer = journal_module.JournalAuthority(root)
    _, second_head, _ = _commit_telemetry(writer, "second")
    journal = root / "streaming_manifest.journal.jsonl"
    original_decode = observer._decode_transaction
    drifted = False

    def decode_then_drift(*args, **kwargs):
        nonlocal drifted
        transaction = original_decode(*args, **kwargs)
        if not drifted:
            before = journal.stat()
            os.utime(journal, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000))
            drifted = True
        return transaction

    monkeypatch.setattr(observer, "_decode_transaction", decode_then_drift)
    refreshed_head, _ = observer.refresh(document, first_head, changes)

    assert drifted
    assert refreshed_head == second_head
    assert observer._journal_digest.hexdigest() == hashlib.sha256(journal.read_bytes()).hexdigest()
    marker_records = [record for record in caplog.records if "JOURNAL_METADATA_RECHECK" in record.getMessage()]
    assert len(marker_records) == 1
    assert json.loads(marker_records[0].getMessage().split(" ", 1)[1])["reason"] == "metadata_changed_during_delta_refresh"


def test_old_committed_prefix_corruption_at_same_head_is_rejected(tmp_path):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _new_root(tmp_path)
    _commit_telemetry(authority, "first")
    document, head, changes = _commit_telemetry(authority, "second")

    journal = root / "streaming_manifest.journal.jsonl"
    original = journal.read_bytes()
    corrupted = original.replace(b"first", b"firsx", 1)
    assert len(corrupted) == len(original)
    journal.write_bytes(corrupted)

    with pytest.raises(journal_module.JournalError) as caught:
        authority.refresh(document, head, changes)

    assert caught.value.category == "journal_digest_mismatch"
    assert caught.value.details["reason"] == "metadata_changed_without_HEAD_advance"
    assert caught.value.details["expected_sha256"] == hashlib.sha256(original).hexdigest()
    assert caught.value.details["actual_sha256"] == hashlib.sha256(corrupted).hexdigest()


def test_raw_digest_tracks_initialize_load_local_commit_external_refresh_and_reopen(tmp_path, caplog):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    caplog.set_level("WARNING", logger=journal_module.__name__)
    root, authority = _new_root(tmp_path)
    journal = root / "streaming_manifest.journal.jsonl"
    assert authority._journal_digest.hexdigest() == hashlib.sha256(b"").hexdigest()
    document, first_head, changes = _commit_telemetry(authority, "first")
    assert authority._journal_digest.hexdigest() == hashlib.sha256(journal.read_bytes()).hexdigest()

    writer = journal_module.JournalAuthority(root)
    _, second_head, _ = _commit_telemetry(writer, "second")
    refreshed_head, _ = authority.refresh(document, first_head, changes)
    assert refreshed_head == second_head
    assert authority._journal_digest.hexdigest() == hashlib.sha256(journal.read_bytes()).hexdigest()
    assert authority._journal_digest_offset == second_head.committed_offset
    assert authority._journal_verified_head == second_head

    reopened = journal_module.JournalAuthority(root)
    _, loaded_head, _ = reopened.load()
    assert loaded_head == second_head
    assert reopened._journal_digest.hexdigest() == hashlib.sha256(journal.read_bytes()).hexdigest()
    assert not any("JOURNAL_METADATA_RECHECK" in record.getMessage() for record in caplog.records)


def test_unstable_digest_snapshot_is_retried_once_then_rejected(tmp_path, monkeypatch):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _new_root(tmp_path)
    document, head, changes = _commit_telemetry(authority, "unstable")
    journal = root / "streaming_manifest.journal.jsonl"
    before = journal.stat()
    os.utime(journal, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000))
    original_strict_head = journal_module._strict_head
    calls = 0

    def fluctuate_head(path, parent):
        nonlocal calls
        result = original_strict_head(path, parent)
        calls += 1
        if calls in {3, 5}:
            actual, info, byte_count = result
            unstable = journal_module.JournalHead(
                actual.lineage_id,
                actual.checkpoint_sha256,
                actual.sequence + 1,
                actual.committed_offset,
                actual.transaction_sha256,
            )
            return unstable, info, byte_count
        return result

    monkeypatch.setattr(journal_module, "_strict_head", fluctuate_head)
    with pytest.raises(journal_module.JournalError) as caught:
        authority.refresh(document, head, changes)

    assert calls == 5  # outer snapshot plus two before/after verifier samples
    assert caught.value.category == "journal_snapshot_unstable"
    assert caught.value.details["attempt"] == 2
    assert caught.value.details["attempt_limit"] == 2


def test_content_change_after_first_hash_chunk_is_retried_and_rejected(tmp_path, monkeypatch):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, authority = _new_root(tmp_path)
    document, head, changes = _commit_telemetry(authority, "A" * 1_100_000)
    journal = root / "streaming_manifest.journal.jsonl"
    before = journal.stat()
    os.utime(journal, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000))
    original_read = journal_module.os.read
    mutated = False
    mutation_offset = 1024 * 1024 + 128

    def read_then_mutate(fd, count):
        nonlocal mutated
        data = original_read(fd, count)
        if data and not mutated and count >= 1024 * 1024:
            with journal.open("r+b") as handle:
                handle.seek(mutation_offset)
                old = handle.read(1)
                assert old == b"A"
                handle.seek(mutation_offset)
                handle.write(b"B")
                handle.flush()
            mutated = True
        return data

    monkeypatch.setattr(journal_module.os, "read", read_then_mutate)
    with pytest.raises(journal_module.JournalError) as caught:
        authority.refresh(document, head, changes)

    assert mutated
    assert caught.value.category == "journal_digest_mismatch"
    assert caught.value.details["attempt"] == 2
    assert caught.value.details["expected_sha256"] != caught.value.details["actual_sha256"]


@pytest.mark.parametrize(
    ("stage", "marker", "replacement", "expected_reason"),
    [
        ("load", b"load-window", b"load-windox", "metadata_changed_during_load"),
        ("delta", b"delta-window", b"delta-windox", "metadata_changed_during_delta_refresh"),
    ],
)
def test_replay_window_same_length_content_change_is_rejected_without_digest_publish(
    tmp_path, monkeypatch, stage, marker, replacement, expected_reason
):
    import KHz_filament.hr4e5_streaming_journal as journal_module

    root, initializer = _new_root(tmp_path)
    if stage == "load":
        _commit_telemetry(initializer, "load-window")
        authority = journal_module.JournalAuthority(root)
        invoke = lambda: authority.load()
        cached_digest = None
        cached_head = None
        cached_stat = None
    else:
        document, cached_head, changes = _commit_telemetry(initializer, "first")
        writer = journal_module.JournalAuthority(root)
        _commit_telemetry(writer, "delta-window")
        authority = initializer
        cached_digest = authority._journal_digest
        cached_stat = authority._journal_stat
        invoke = lambda: authority.refresh(document, cached_head, changes)

    journal = root / "streaming_manifest.journal.jsonl"
    original_bytes = journal.read_bytes()
    assert marker in original_bytes
    assert len(marker) == len(replacement)
    original_decode = authority._decode_transaction
    mutated = False

    def decode_then_mutate_same_size(*args, **kwargs):
        nonlocal mutated
        transaction = original_decode(*args, **kwargs)
        if not mutated:
            current_bytes = journal.read_bytes()
            assert marker in current_bytes
            changed_bytes = current_bytes.replace(marker, replacement, 1)
            assert len(changed_bytes) == len(current_bytes)
            marker_offset = current_bytes.index(marker)
            with journal.open("r+b") as handle:
                handle.seek(marker_offset)
                handle.write(replacement)
                handle.flush()
            mutated = True
        return transaction

    monkeypatch.setattr(authority, "_decode_transaction", decode_then_mutate_same_size)
    with pytest.raises(journal_module.JournalError) as caught:
        invoke()

    assert mutated
    assert caught.value.category == "journal_digest_mismatch"
    assert caught.value.details["reason"] == expected_reason
    assert caught.value.details["expected_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert caught.value.details["actual_sha256"] == hashlib.sha256(journal.read_bytes()).hexdigest()
    if stage == "load":
        assert authority._journal_digest is None
        assert authority._journal_digest_offset is None
        assert authority._journal_verified_head is None
        assert authority._journal_stat is None
    else:
        assert authority._journal_digest is cached_digest
        assert authority._journal_verified_head == cached_head
        assert authority._journal_digest_offset == cached_head.committed_offset
        assert authority._journal_stat == cached_stat

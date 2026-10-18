from __future__ import annotations

import json

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

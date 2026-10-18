"""Append-only durable authority for a StreamingLifecycle root.

The initial ``streaming_manifest.json`` is an immutable checkpoint.  A
transaction becomes committed only when the small HEAD file is atomically
replaced after the journal record has been fsynced.  A journal tail beyond
HEAD is deliberately rejected: callers must preserve and diagnose it rather
than guessing whether it was committed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


JOURNAL_SCHEMA = "khz_filament.hr4e5s.streaming.journal.v2"
HEAD_SCHEMA = "khz_filament.hr4e5s.streaming.journal_head.v2"
MAX_HEAD_BYTES = 64 * 1024
MAX_TRANSACTION_BYTES = 64 * 1024 * 1024
_MUTABLE_RECORD_FIELDS = frozenset({"state", "retry_count", "hydro_attempt_started", "post", "next", "transitions", "hydro_owner"})
_APPEND_ONLY_TOP_LISTS = frozenset({"recovery_attempts", "rate_events", "telemetry_events", "s5_recovery_events"})
_SETTABLE_TOP_FIELDS = frozenset({"queue", "recovery_active", "recovery_backlog", "recovery_backlog_cursor", "barrier", "promotion"})
_LOGGER = logging.getLogger(__name__)


class JournalError(RuntimeError):
    """A durable lifecycle root cannot be read or committed safely."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None):
        self.details = dict(details or {})
        self.category = self.details.get("category")
        if self.details:
            context = json.dumps(self.details, sort_keys=True, separators=(",", ":"), allow_nan=False)
            message = f"{message} [context={context}]"
        super().__init__(message)


class _Cursor:
    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = int(value)


class _ChangeSet:
    __slots__ = ("ops", "suspended", "cursor")

    def __init__(self, cursor: int = 0):
        self.ops: list[dict[str, Any]] = []
        self.suspended = False
        self.cursor = _Cursor(cursor)

    def add(self, op: Mapping[str, Any]) -> None:
        if not self.suspended:
            self.ops.append(dict(op))


def _plain(value: Any, *, raw_backlog: bool = True) -> Any:
    if isinstance(value, RecoveryBacklogList):
        items = list.__iter__(value) if raw_backlog else iter(value)
        return [_plain(item, raw_backlog=raw_backlog) for item in items]
    if isinstance(value, dict):
        return {str(key): _plain(item, raw_backlog=raw_backlog) for key, item in dict.items(value)}
    if isinstance(value, list):
        return [_plain(item, raw_backlog=raw_backlog) for item in list.__iter__(value)]
    if isinstance(value, tuple):
        return [_plain(item, raw_backlog=raw_backlog) for item in value]
    return value


def _wrap(value: Any, path: tuple[str | int, ...], changes: _ChangeSet) -> Any:
    if isinstance(value, dict):
        return TrackedDict(value, path, changes)
    if isinstance(value, list):
        if path == ("recovery_backlog",):
            return RecoveryBacklogList(value, path, changes)
        return TrackedList(value, path, changes)
    return value


class TrackedDict(dict):
    """JSON dictionary that emits explicit path operations on mutation."""

    def __init__(self, source: Mapping[str, Any], path: tuple[str | int, ...], changes: _ChangeSet):
        self._path = path
        self._changes = changes
        dict.__init__(self)
        for key, value in source.items():
            dict.__setitem__(self, key, _wrap(value, path + (str(key),), changes))

    def __setitem__(self, key: str, value: Any) -> None:
        key = str(key)
        existed = key in self
        previous = _plain(dict.__getitem__(self, key)) if existed else None
        wrapped = _wrap(value, self._path + (key,), self._changes)
        dict.__setitem__(self, key, wrapped)
        if self._path == () and key == "recovery_backlog_cursor":
            self._changes.cursor.value = int(value)
        self._changes.add({"op": "set", "path": list(self._path + (key,)), "value": _plain(value), "existed": existed, "previous": previous})

    def __delitem__(self, key: str) -> None:
        key = str(key)
        dict.__delitem__(self, key)
        self._changes.add({"op": "delete", "path": list(self._path + (key,))})

    def clear(self) -> None:
        for key in list(dict.keys(self)):
            del self[key]

    def pop(self, key: str, *default: Any) -> Any:
        key = str(key)
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = self[key]
        del self[key]
        return value

    def popitem(self) -> tuple[str, Any]:
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(self))
        return key, self.pop(key)

    def setdefault(self, key: str, default: Any = None) -> Any:
        key = str(key)
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, other=(), /, **kwargs: Any) -> None:
        values = dict(other, **kwargs)
        for key, value in values.items():
            self[key] = value


class TrackedList(list):
    """JSON list that records append deltas and bounded replacements."""

    def __init__(self, source: Iterable[Any], path: tuple[str | int, ...], changes: _ChangeSet):
        self._path = path
        self._changes = changes
        list.__init__(self, (_wrap(value, path + (index,), changes) for index, value in enumerate(source)))

    def _replace_op(self, previous=None) -> None:
        op = {"op": "set", "path": list(self._path), "value": _plain(self)}
        if self._path == ("queue",):
            if previous is None:
                raise JournalError("bounded queue replacement lacks its predecessor")
            op.update(existed=True, previous=previous)
        self._changes.add(op)

    def __setitem__(self, index, value: Any) -> None:
        if isinstance(index, slice):
            wrapped = [_wrap(item, self._path + (i,), self._changes) for i, item in enumerate(value)]
            list.__setitem__(self, index, wrapped)
            self._replace_op()
            return
        normalized = index if index >= 0 else len(self) + index
        wrapped = _wrap(value, self._path + (normalized,), self._changes)
        list.__setitem__(self, index, wrapped)
        self._changes.add({"op": "set", "path": list(self._path + (normalized,)), "value": _plain(value)})

    def __delitem__(self, index) -> None:
        list.__delitem__(self, index)
        self._replace_op()

    def append(self, value: Any) -> None:
        wrapped = _wrap(value, self._path + (len(self),), self._changes)
        list.append(self, wrapped)
        self._changes.add({"op": "append", "path": list(self._path), "value": _plain(value)})

    def extend(self, values: Iterable[Any]) -> None:
        for value in values:
            self.append(value)

    def insert(self, index: int, value: Any) -> None:
        list.insert(self, index, _wrap(value, self._path + (index,), self._changes))
        self._replace_op()

    def pop(self, index: int = -1) -> Any:
        value = list.pop(self, index)
        self._replace_op()
        return value

    def remove(self, value: Any) -> None:
        list.remove(self, value)
        self._replace_op()

    def clear(self) -> None:
        list.clear(self)
        self._replace_op()

    def reverse(self) -> None:
        list.reverse(self)
        self._replace_op()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        previous = _plain(self) if self._path == ("queue",) else None
        list.sort(self, *args, **kwargs)
        self._replace_op(previous)

    def __iadd__(self, values):
        self.extend(values)
        return self


class RecoveryBacklogList(TrackedList):
    """A pending-work view over an immutable sorted backlog plus a cursor.

    The underlying list remains stable as entries move to the bounded queue;
    advancing the cursor costs O(1).  Iteration and length expose only pending
    entries to existing readers.
    """

    def __len__(self) -> int:
        return max(0, list.__len__(self) - self._changes.cursor.value)

    def __eq__(self, other) -> bool:
        return list(self) == other

    def __iter__(self):
        cursor = self._changes.cursor.value
        for index in range(cursor, list.__len__(self)):
            yield list.__getitem__(self, index)

    def __getitem__(self, index):
        cursor = self._changes.cursor.value
        if isinstance(index, slice):
            start, stop, step = index.indices(max(0, list.__len__(self) - cursor))
            return [list.__getitem__(self, cursor + offset) for offset in range(start, stop, step)]
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("recovery backlog index out of range")
        return list.__getitem__(self, cursor + normalized)

    def append(self, value: Any) -> None:
        index = list.__len__(self)
        list.append(self, _wrap(value, self._path + (index,), self._changes))
        self._changes.add({"op": "append", "path": list(self._path), "value": _plain(value)})


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise JournalError(f"journal payload is not canonical JSON: {error}") from error


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class JournalHead:
    lineage_id: str
    checkpoint_sha256: str
    sequence: int
    committed_offset: int
    transaction_sha256: str

    def public(self) -> dict[str, Any]:
        return {
            "schema": HEAD_SCHEMA,
            "lineage_id": self.lineage_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "sequence": self.sequence,
            "committed_offset": self.committed_offset,
            "transaction_sha256": self.transaction_sha256,
        }


def _safe_regular(path: Path, *, expected_parent: Path, label: str) -> os.stat_result:
    try:
        parent = path.parent.resolve(strict=True)
        expected = expected_parent.resolve(strict=True)
        info = path.lstat()
    except OSError as error:
        raise JournalError(f"{label} is missing or inaccessible: {error}") from error
    if parent != expected or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise JournalError(f"{label} path is not a private regular file in the lifecycle root")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise JournalError(f"{label} owner differs from the current process")
    return info


def _safe_root(root: Path) -> Path:
    try:
        info = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise JournalError(f"journal lifecycle root is missing: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or root.absolute() != resolved:
        raise JournalError("journal lifecycle root must be a non-symlink directory")
    return resolved


def _fsync_dir_strict(path: Path) -> None:
    if os.name == "nt":
        # Windows does not expose a portable directory FlushFileBuffers via
        # Python.  File fsync and atomic ReplaceFile remain required there.
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_head(path: Path, head: Mapping[str, Any]) -> int:
    data = json.dumps(head, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir_strict(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return len(data)


def _strict_head(path: Path, root: Path) -> tuple[JournalHead, os.stat_result, int]:
    info = _safe_regular(path, expected_parent=root, label="journal HEAD")
    if info.st_size <= 0 or info.st_size > MAX_HEAD_BYTES:
        raise JournalError("journal HEAD has an invalid size")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise JournalError(f"journal HEAD is unreadable: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != HEAD_SCHEMA:
        raise JournalError("journal HEAD schema is invalid")
    if set(value) != {"schema", "lineage_id", "checkpoint_sha256", "sequence", "committed_offset", "transaction_sha256"}:
        raise JournalError("journal HEAD contains unsupported authority fields")
    lineage = value.get("lineage_id")
    checkpoint = value.get("checkpoint_sha256")
    tx_hash = value.get("transaction_sha256")
    sequence = value.get("sequence")
    offset = value.get("committed_offset")
    if (
        not isinstance(lineage, str) or len(lineage) != 32
        or not isinstance(checkpoint, str) or len(checkpoint) != 64
        or not isinstance(tx_hash, str) or len(tx_hash) != 64
        or not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0
        or not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
    ):
        raise JournalError("journal HEAD fields are invalid")
    return JournalHead(lineage, checkpoint, sequence, offset, tx_hash), info, len(raw)


def _journal_stat_tuple(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _journal_identity_size(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size)


def _journal_stat_context(info: os.stat_result | None) -> dict[str, int] | None:
    if info is None:
        return None
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _journal_stat_tuple_context(value: tuple[int, int, int, int, int] | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "device": value[0],
        "inode": value[1],
        "size": value[2],
        "mtime_ns": value[3],
        "ctime_ns": value[4],
    }


def _head_context(head: JournalHead | None) -> dict[str, Any] | None:
    return head.public() if head is not None else None


def _stream_journal_prefix(path: Path, root: Path, length: int) -> tuple[Any, os.stat_result, os.stat_result, os.stat_result]:
    """Hash exactly ``length`` raw journal bytes without buffering the file."""
    before = _safe_regular(path, expected_parent=root, label="manifest journal")
    if length < 0 or before.st_size < length:
        raise JournalError("journal is shorter than the requested verified prefix")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise JournalError("journal inode changed before digest verification")
        digest = hashlib.sha256()
        remaining = length
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise JournalError("journal ended during digest verification")
            digest.update(chunk)
            remaining -= len(chunk)
        fd_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = _safe_regular(path, expected_parent=root, label="manifest journal")
    return digest, before, fd_after, path_after


def _validate_path_op(op: Mapping[str, Any], record_count: int, *, operation: str, allow_backlog_rebuild: bool) -> None:
    kind = op.get("op")
    path = op.get("path")
    if kind not in {"set", "append", "delete"} or not isinstance(path, list) or not path:
        raise JournalError("journal mutation operation is malformed")
    if any(not isinstance(item, (str, int)) or isinstance(item, bool) for item in path):
        raise JournalError("journal mutation path has an invalid segment")
    if path[0] == "records":
        if len(path) != 3 or not isinstance(path[1], int) or not 0 <= path[1] < record_count or path[2] not in _MUTABLE_RECORD_FIELDS:
            raise JournalError("journal attempted to change immutable screen identity or unsupported field")
        if kind == "append" and path[2] != "transitions":
            raise JournalError("journal append is only valid for record transitions")
        if kind == "delete":
            raise JournalError("journal cannot delete record state")
        if path[2] == "transitions" and kind != "append":
            raise JournalError("record transitions are append-only")
        if kind == "set":
            if "previous" not in op or type(op.get("existed")) is not bool:
                raise JournalError("record mutation lacks a durable predecessor")
            field, value, old = path[2], op.get("value"), op["previous"]
            if field == "state":
                edges = {
                    "begin_optical": {("CURRENT_READY", "OPTICAL_IN_PROGRESS")},
                    "deposition_finalized": {("CURRENT_READY", "OPTICAL_IN_PROGRESS"), ("OPTICAL_IN_PROGRESS", "DEPOSITION_FINALIZED")},
                    "commit_post": {("DEPOSITION_FINALIZED", "POST_COMMITTED")},
                    "enqueue_post": {("POST_COMMITTED", "HYDRO_QUEUED")},
                    "claim_block": {("HYDRO_QUEUED", "HYDRO_RUNNING"), ("POST_COMMITTED", "HYDRO_QUEUED")},
                    "begin_hydro_screen": {("HYDRO_RUNNING", "HYDRO_RUNNING")},
                    "commit_next": {("HYDRO_RUNNING", "HYDRO_RUNNING"), ("HYDRO_RUNNING", "NEXT_COMMITTED")},
                    "validate_barrier": {("NEXT_COMMITTED", "BARRIER_VALIDATED")},
                    "reconstruct_queue": {(s, "POST_COMMITTED") for s in ("POST_COMMITTED", "HYDRO_QUEUED", "HYDRO_RUNNING")} | {("POST_COMMITTED", "HYDRO_QUEUED")},
                }
                if (old, value) not in edges.get(operation, set()):
                    raise JournalError("journal state transition violates its operation")
            if field in {"post", "next"} and (old is not None or not isinstance(value, dict) or operation != "commit_" + field):
                raise JournalError("duplicate or forged artifact commit")
            if field == "hydro_attempt_started" and type(value) is not bool:
                raise JournalError("hydro attempt flag must be boolean")
            if field == "retry_count" and (operation != "reconstruct_queue" or type(value) is not int or value != old + 1):
                raise JournalError("retry mutation is not a recovery increment")
            if field == "hydro_owner" and value is not None:
                if (operation != "claim_block" or not isinstance(value, dict)
                        or set(value) != {"actor", "process", "claim_token"}
                        or not isinstance(value["actor"], str) or not value["actor"]
                        or not isinstance(value["process"], dict)
                        or set(value["process"]) != {"pid", "host_identity", "process_start_token"}
                        or type(value["process"]["pid"]) is not int or value["process"]["pid"] <= 0
                        or not all(isinstance(value["process"][k], str) and value["process"][k] for k in ("host_identity", "process_start_token"))
                        or not isinstance(value["claim_token"], str) or len(value["claim_token"]) != 32):
                    raise JournalError("hydro owner must come from a process-fenced claim")
        return
    if len(path) != 1 or not isinstance(path[0], str):
        raise JournalError("journal mutation path is outside the allowed schema")
    key = path[0]
    if kind == "append":
        if key not in _APPEND_ONLY_TOP_LISTS and key != "queue":
            raise JournalError("journal append target is not append-only")
        return
    if key not in _SETTABLE_TOP_FIELDS and key not in _APPEND_ONLY_TOP_LISTS:
        raise JournalError("journal attempted to change an immutable checkpoint field")
    if key == "recovery_backlog" and kind == "set" and not allow_backlog_rebuild:
        raise JournalError("recovery backlog replacement is only valid during explicit reconstruction")
    if key in _APPEND_ONLY_TOP_LISTS and kind != "append":
        raise JournalError("append-only lifecycle history cannot be replaced or deleted")
    if kind == "set":
        value = op.get("value")
        if key == "queue" and (not isinstance(value, list) or any(type(i) is not int or not 0 <= i < record_count for i in value) or len(value) != len(set(value))):
            raise JournalError("invalid journal queue")
        if key == "queue" and (op.get("existed") is not True or not isinstance(op.get("previous"), list)):
            raise JournalError("queue replacement lacks a durable predecessor")
        if key == "recovery_active" and type(value) is not bool:
            raise JournalError("invalid recovery flag")
        if key == "recovery_backlog_cursor" and (
                type(value) is not int or value < 0
                or operation not in {"claim_block", "reconstruct_queue"}
                or op.get("existed") is not True or type(op.get("previous")) is not int):
            raise JournalError("invalid recovery cursor")
        if key == "barrier" and operation != "validate_barrier":
            raise JournalError("barrier can only be committed by the global audit")
        if key == "promotion" and operation != "promote_next_to_current":
            raise JournalError("promotion can only be committed by promotion")
    if kind == "delete":
        raise JournalError("journal cannot delete lifecycle fields")
    if kind == "append":
        raise JournalError("append operation requires an append-only history list")


def _resolve_parent(document: Any, path: Sequence[str | int]) -> tuple[Any, str | int]:
    node = document
    for segment in path[:-1]:
        node = node[segment]
    return node, path[-1]


def validate_projection(document: Mapping[str, Any], operations) -> None:
    """Bounded post-transaction semantic checks, also applied during replay."""
    records = document["records"]
    queue = document.get("queue", [])
    if not isinstance(queue, list) or len(queue) > int(document.get("queue_depth", 16)) or len(queue) != len(set(queue)):
        raise JournalError("replayed queue violates its bound or uniqueness")
    touched = {int(op["path"][1]) for op in operations if op["path"][0] == "records"}
    touched.update(queue)
    backlog = document.get("recovery_backlog", [])
    cursor = document.get("recovery_backlog_cursor", 0)
    if type(cursor) is not int or not 0 <= cursor <= list.__len__(backlog):
        raise JournalError("replayed recovery cursor is outside backlog")
    pending_states: dict[int, list[str]] = {}
    for op in operations:
        path = op["path"]
        if len(path) != 3 or path[0] != "records":
            continue
        i, field = path[1:]
        if field == "state":
            pending_states.setdefault(i, []).append(op["value"])
        if field != "transitions":
            continue
        event = op["value"]
        record = records[i]
        pending = pending_states.get(i, [])
        if (not isinstance(event, dict) or not pending or event.get("state") != pending.pop(0)
                or event.get("ordinal") != i or event.get("screen_id") != record.get("screen_id")
                or event.get("status") != "PASS" or not isinstance(event.get("actor"), str) or not event["actor"]
                or event.get("current_generation") != document.get("current_generation")
                or event.get("next_generation") != document.get("next_generation")
                or event.get("current_content_sha256") != document.get("current_content_sha256")
                or event.get("retry_count") != record.get("retry_count")):
            raise JournalError("transition identity or state mutation binding is invalid")
        state = event["state"]
        source = record.get("current")
        if state in {"HYDRO_QUEUED", "HYDRO_RUNNING", "NEXT_COMMITTED"} or (state == "POST_COMMITTED" and event.get("reconstructed")):
            source = record.get("post")
        if state == "BARRIER_VALIDATED":
            source = record.get("next")
        if not source or event.get("source_file_sha256") != source.get("file_sha256"):
            raise JournalError("transition source hash differs from its committed artifact")
        output = record.get("post") if state == "POST_COMMITTED" and not event.get("reconstructed") else record.get("next") if state in {"NEXT_COMMITTED", "BARRIER_VALIDATED"} else None
        if output is not None and event.get("output_file_sha256") != output.get("file_sha256"):
            raise JournalError("transition output hash differs from its committed artifact")
    if any(pending_states.values()):
        raise JournalError("state mutation lacks its durable transition event")
    # A cursor is a durable consumption decision, not a free-standing index.
    # Check its bounded slices against this transaction's queue/state deltas.
    queue_seen = set()
    enqueued = set()
    rebuilding = any(op["path"] == ["recovery_backlog"] and op["op"] == "set" for op in operations)
    for op in operations:
        if op["path"] == ["queue"]:
            queue_seen.update(op.get("previous", []))
            if op["op"] == "append":
                queue_seen.add(op["value"])
            elif op["op"] == "set":
                queue_seen.update(op["value"])
        if (len(op["path"]) == 3 and op["path"][0] == "records"
                and op["path"][2] == "state" and op.get("previous") == "POST_COMMITTED"
                and op.get("value") == "HYDRO_QUEUED"):
            enqueued.add(op["path"][1])
    for op in operations:
        if op["path"] != ["recovery_backlog_cursor"]:
            continue
        old, new = op["previous"], op["value"]
        if rebuilding and new == 0:
            continue
        if not 0 <= old < new <= list.__len__(backlog) or new-old > int(document.get("queue_depth",16)):
            raise JournalError("recovery cursor moved without bounded queue consumption")
        consumed = {list.__getitem__(backlog,i) for i in range(old,new)}
        if not consumed.issubset(enqueued & queue_seen):
            raise JournalError("recovery cursor skipped records without durable enqueue")
    for i in queue_seen:
        if type(i) is not int or not 0 <= i < len(records):
            raise JournalError("queue predecessor contains an invalid screen")
        if records[i].get("state") == "HYDRO_QUEUED" and i not in queue:
            raise JournalError("queue delta lost a queued screen without a claim")
    for i in touched:
        if type(i) is not int or not 0 <= i < len(records):
            raise JournalError("replayed screen ordinal is outside checkpoint")
        record = records[i]
        state = record.get("state")
        if record.get("ordinal") != i or not record.get("transitions") or record["transitions"][-1].get("state") != state:
            raise JournalError("replayed state does not close its transition history")
        if i in queue and (state != "HYDRO_QUEUED" or record.get("next") is not None):
            raise JournalError("replayed queue disagrees with record state")
        if state in {"POST_COMMITTED", "HYDRO_QUEUED", "HYDRO_RUNNING", "NEXT_COMMITTED", "BARRIER_VALIDATED"} and record.get("post") is None:
            raise JournalError("replayed committed state lacks POST")
        if state in {"NEXT_COMMITTED", "BARRIER_VALIDATED"} and record.get("next") is None:
            raise JournalError("replayed committed state lacks NEXT")
        if state == "HYDRO_RUNNING" and not record.get("hydro_owner"):
            raise JournalError("replayed running record lacks an exclusive owner")


def _apply_op(document: TrackedDict, op: Mapping[str, Any], changes: _ChangeSet) -> None:
    kind, path = op["op"], tuple(op["path"])
    parent, final = _resolve_parent(document, path)
    if kind == "set" and "existed" in op:
        existed = final in parent if isinstance(parent, dict) else 0 <= int(final) < len(parent)
        if existed != op["existed"] or (existed and _plain(parent[final]) != op.get("previous")):
            raise JournalError("journal mutation predecessor differs from durable state")
    changes.suspended = True
    try:
        if kind == "set":
            wrapped = _wrap(op["value"], path, changes)
            if isinstance(parent, dict):
                dict.__setitem__(parent, final, wrapped)
                if path == ("recovery_backlog_cursor",):
                    changes.cursor.value = int(op["value"])
            else:
                list.__setitem__(parent, final, wrapped)
        elif kind == "append":
            target = parent[final]
            index = list.__len__(target)
            list.append(target, _wrap(op["value"], path + (index,), changes))
        elif kind == "delete":
            if isinstance(parent, dict):
                dict.__delitem__(parent, final)
            else:
                list.__delitem__(parent, final)
    finally:
        changes.suspended = False


class JournalAuthority:
    """Read and commit one lifecycle journal while the caller holds its lock."""

    def __init__(self, root: str | Path, *, checkpoint_name: str = "streaming_manifest.json"):
        self.root_path = Path(root)
        self.root = _safe_root(self.root_path)
        self.checkpoint_path = self.root / checkpoint_name
        self.journal_path = self.root / "streaming_manifest.journal.jsonl"
        self.head_path = self.root / "streaming_manifest.head.json"
        self._checkpoint_stat: tuple[int, int, int, int] | None = None
        self._journal_identity: tuple[int, int] | None = None
        self._journal_stat: tuple[int, int, int, int, int] | None = None
        self._journal_digest: Any | None = None
        self._journal_digest_offset: int | None = None
        self._journal_verified_head: JournalHead | None = None
        self.journal_digest_recheck_count = 0
        self.journal_digest_recheck_elapsed_s = 0.0
        self._record_count: int | None = None

    @classmethod
    def initialize(cls, root: str | Path, checkpoint_bytes: bytes, *, checkpoint_name: str = "streaming_manifest.json") -> "JournalAuthority":
        authority = cls(root, checkpoint_name=checkpoint_name)
        checkpoint_info = _safe_regular(authority.checkpoint_path, expected_parent=authority.root, label="manifest checkpoint")
        if checkpoint_info.st_size != len(checkpoint_bytes) or _sha(authority.checkpoint_path.read_bytes()) != _sha(checkpoint_bytes):
            raise JournalError("checkpoint bytes changed during journal initialization")
        for path in (authority.journal_path, authority.head_path):
            if path.exists() or path.is_symlink():
                raise JournalError("journal authority files already exist")
        journal_fd = os.open(str(authority.journal_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
        try:
            os.fsync(journal_fd)
        finally:
            os.close(journal_fd)
        _fsync_dir_strict(authority.root)
        checkpoint_sha = _sha(checkpoint_bytes)
        initial = JournalHead(secrets.token_hex(16), checkpoint_sha, 0, 0, checkpoint_sha)
        _atomic_head(authority.head_path, initial.public())
        try:
            authority.checkpoint_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            # The hash binding remains mandatory; unsupported chmod does not
            # permit any runtime code path to rewrite the checkpoint.
            pass
        authority._capture_checkpoint_stat()
        authority._capture_journal_stat()
        authority._journal_digest = hashlib.sha256()
        authority._journal_digest_offset = 0
        authority._journal_verified_head = initial
        return authority

    def _capture_checkpoint_stat(self) -> None:
        info = _safe_regular(self.checkpoint_path, expected_parent=self.root, label="manifest checkpoint")
        self._checkpoint_stat = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _capture_journal_stat(self) -> os.stat_result:
        info = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
        self._journal_identity = (info.st_dev, info.st_ino)
        self._journal_stat = _journal_stat_tuple(info)
        return info

    def _verify_cached_prefix(
        self,
        expected_head: JournalHead,
        *,
        prefix_length: int,
        expected_size: int,
        expected_digest: str,
        category: str,
        expected_identity: tuple[int, int] | None = None,
    ) -> tuple[Any, os.stat_result, float]:
        """Rehash a verified prefix under the caller's lifecycle lock.

        The file and HEAD are sampled on both sides of each bounded streaming
        pass.  A changing snapshot gets one retry; a stable digest mismatch is
        corruption and is never retried or repaired.
        """
        identity = expected_identity if expected_identity is not None else self._journal_identity
        last_context: dict[str, Any] = {}
        started = time.perf_counter()
        for attempt in (1, 2):
            head_before: JournalHead | None = None
            head_after: JournalHead | None = None
            stat_before: os.stat_result | None = None
            fd_after: os.stat_result | None = None
            path_after: os.stat_result | None = None
            try:
                head_before, _, _ = _strict_head(self.head_path, self.root)
                stat_before = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
                if identity != (stat_before.st_dev, stat_before.st_ino):
                    raise JournalError(
                        "manifest journal inode changed during digest verification",
                        details={
                            "category": "journal_inode_changed_during_digest",
                            "expected_identity": identity,
                            "observed_identity": (stat_before.st_dev, stat_before.st_ino),
                            "expected_head": _head_context(expected_head),
                            "head_before": _head_context(head_before),
                            "cached_stat": _journal_stat_tuple_context(self._journal_stat),
                            "expected_sha256": expected_digest,
                            "stat_before": _journal_stat_context(stat_before),
                        },
                    )
                if stat_before.st_size != expected_size:
                    raise JournalError("journal size differs from verified HEAD offset; preserving evidence")
                digest, stream_before, fd_after, path_after = _stream_journal_prefix(
                    self.journal_path, self.root, prefix_length
                )
                head_after, _, _ = _strict_head(self.head_path, self.root)
                final_path_info = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
                stable_file = (
                    _journal_stat_tuple(stat_before)
                    == _journal_stat_tuple(stream_before)
                    == _journal_stat_tuple(path_after)
                    == _journal_stat_tuple(final_path_info)
                    and _journal_identity_size(stat_before) == _journal_identity_size(fd_after)
                )
                stable_head = head_before == expected_head == head_after
                last_context = {
                    "category": category,
                    "attempt": attempt,
                    "attempt_limit": 2,
                    "expected_head": _head_context(expected_head),
                    "head_before": _head_context(head_before),
                    "head_after": _head_context(head_after),
                    "cached_stat": _journal_stat_tuple_context(self._journal_stat),
                    "expected_sha256": expected_digest,
                    "actual_sha256": digest.hexdigest(),
                    "stat_before": _journal_stat_context(stat_before),
                    "stat_fd_after": _journal_stat_context(fd_after),
                    "stat_path_after": _journal_stat_context(path_after),
                    "stat_final": _journal_stat_context(final_path_info),
                }
                if not stable_file or not stable_head:
                    if attempt == 1:
                        continue
                    last_context["category"] = "journal_snapshot_unstable"
                    last_context["reason"] = category
                    raise JournalError("journal or HEAD changed during bounded digest verification", details=last_context)
                if digest.hexdigest() != expected_digest:
                    last_context["category"] = "journal_digest_mismatch"
                    last_context["reason"] = category
                    raise JournalError("journal bytes differ from the previously verified committed prefix", details=last_context)
                return digest, final_path_info, time.perf_counter() - started
            except JournalError as error:
                if error.details:
                    raise
                # An inode race between the path check and open is a snapshot
                # instability.  Retry it once; other validation failures are
                # authoritative and fail closed immediately.
                if "inode changed before digest verification" not in str(error) or attempt == 2:
                    details = {
                        "category": "journal_digest_verification_failed",
                        "reason": category,
                        "attempt": attempt,
                        "attempt_limit": 2,
                        "expected_head": _head_context(expected_head),
                        "head_before": _head_context(head_before),
                        "head_after": _head_context(head_after),
                        "cached_stat": _journal_stat_tuple_context(self._journal_stat),
                        "expected_sha256": expected_digest,
                        "stat_before": _journal_stat_context(stat_before),
                        "stat_fd_after": _journal_stat_context(fd_after),
                        "stat_path_after": _journal_stat_context(path_after),
                        "cause": str(error),
                    }
                    raise JournalError("journal digest verification failed closed", details=details) from error
            except OSError as error:
                details = {
                    "category": "journal_digest_io_error",
                    "reason": category,
                    "attempt": attempt,
                    "attempt_limit": 2,
                    "expected_head": _head_context(expected_head),
                    "head_before": _head_context(head_before),
                    "head_after": _head_context(head_after),
                    "cached_stat": _journal_stat_tuple_context(self._journal_stat),
                    "expected_sha256": expected_digest,
                    "stat_before": _journal_stat_context(stat_before),
                    "stat_fd_after": _journal_stat_context(fd_after),
                    "stat_path_after": _journal_stat_context(path_after),
                    "cause": str(error),
                }
                raise JournalError("journal digest verification I/O failed closed", details=details) from error
        last_context["category"] = "journal_snapshot_unstable"
        last_context["reason"] = category
        raise JournalError("journal or HEAD changed during bounded digest verification", details=last_context)

    def _require_cached_digest(self, head: JournalHead) -> Any:
        if (
            self._journal_digest is None
            or self._journal_digest_offset != head.committed_offset
            or self._journal_verified_head != head
        ):
            raise JournalError(
                "journal authority lacks a raw digest bound to the requested verified HEAD",
                details={
                    "category": "journal_digest_state_missing",
                    "requested_head": _head_context(head),
                    "verified_head": _head_context(self._journal_verified_head),
                    "digest_offset": self._journal_digest_offset,
                },
            )
        return self._journal_digest

    def _publish_verified_digest(self, digest: Any, head: JournalHead, info: os.stat_result) -> None:
        """Publish digest state only after the caller has validated its snapshot."""
        self._journal_digest = digest.copy()
        self._journal_digest_offset = head.committed_offset
        self._journal_verified_head = head
        self._journal_identity = (info.st_dev, info.st_ino)
        self._journal_stat = _journal_stat_tuple(info)

    def _record_digest_recheck(self, head: JournalHead, *, reason: str, elapsed: float) -> None:
        self.journal_digest_recheck_count += 1
        self.journal_digest_recheck_elapsed_s += elapsed
        payload = {
            "pid": os.getpid(),
            "verified_sequence": head.sequence,
            "verified_offset": head.committed_offset,
            "reason": reason,
            "count": self.journal_digest_recheck_count,
            "elapsed_s": elapsed,
            "total_elapsed_s": self.journal_digest_recheck_elapsed_s,
        }
        _LOGGER.warning(
            "JOURNAL_METADATA_RECHECK %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def check_cached_checkpoint(self) -> None:
        info = _safe_regular(self.checkpoint_path, expected_parent=self.root, label="manifest checkpoint")
        current = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if self._checkpoint_stat is None or current != self._checkpoint_stat:
            raise JournalError("immutable manifest checkpoint identity changed during lifecycle use")

    def load(self, *, changes: _ChangeSet | None = None) -> tuple[TrackedDict, JournalHead, dict[str, int]]:
        if self._checkpoint_stat is None:
            self._capture_checkpoint_stat()
        else:
            self.check_cached_checkpoint()
        checkpoint_info = _safe_regular(self.checkpoint_path, expected_parent=self.root, label="manifest checkpoint")
        try:
            checkpoint_bytes = self.checkpoint_path.read_bytes()
            raw = json.loads(checkpoint_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise JournalError(f"manifest checkpoint is unreadable: {error}") from error
        if not isinstance(raw, dict):
            raise JournalError("manifest checkpoint root must be a JSON object")
        head, _, head_bytes = _strict_head(self.head_path, self.root)
        checkpoint_sha = _sha(checkpoint_bytes)
        if checkpoint_sha != head.checkpoint_sha256:
            raise JournalError("manifest checkpoint hash does not match HEAD")
        journal_info = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
        if self._journal_identity is not None and (journal_info.st_dev, journal_info.st_ino) != self._journal_identity:
            raise JournalError("manifest journal inode changed during lifecycle use")
        if journal_info.st_size != head.committed_offset:
            raise JournalError("journal size differs from committed HEAD offset; preserving uncommitted or truncated bytes")
        change_set = changes if changes is not None else _ChangeSet(int(raw.get("recovery_backlog_cursor", 0)))
        change_set.cursor.value = int(raw.get("recovery_backlog_cursor", 0))
        document = TrackedDict(raw, (), change_set)
        self._capture_checkpoint_stat()
        offset = 0
        sequence = 0
        previous = checkpoint_sha
        replayed_bytes = 0
        journal_digest = hashlib.sha256()
        with self.journal_path.open("rb") as handle:
            while offset < head.committed_offset:
                line = handle.readline(MAX_TRANSACTION_BYTES + 1)
                if not line or len(line) > MAX_TRANSACTION_BYTES or not line.endswith(b"\n"):
                    raise JournalError("journal contains an oversized, truncated, or malformed transaction")
                tx = self._decode_transaction(line[:-1], sequence + 1, previous, head.lineage_id, document)
                for op in tx["operations"]:
                    _apply_op(document, op, change_set)
                validate_projection(document, tx["operations"])
                sequence = tx["sequence"]
                previous = tx["transaction_sha256"]
                offset += len(line)
                replayed_bytes += len(line)
                journal_digest.update(line)
        if offset != head.committed_offset or sequence != head.sequence or previous != head.transaction_sha256:
            raise JournalError("journal replay does not terminate at the committed HEAD")
        head_after, _, _ = _strict_head(self.head_path, self.root)
        journal_info_after = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
        if (
            head_after != head
            or _journal_identity_size(journal_info_after) != _journal_identity_size(journal_info)
        ):
            raise JournalError(
                "journal or HEAD changed during initial replay",
                details={
                    "category": "journal_snapshot_unstable_during_load",
                    "head_before": _head_context(head),
                    "head_after": _head_context(head_after),
                    "stat_before": _journal_stat_context(journal_info),
                    "stat_after": _journal_stat_context(journal_info_after),
                    "expected_sha256": journal_digest.hexdigest(),
                },
            )
        if _journal_stat_tuple(journal_info_after) != _journal_stat_tuple(journal_info):
            checked_digest, journal_info_after, elapsed = self._verify_cached_prefix(
                head,
                prefix_length=head.committed_offset,
                expected_size=head.committed_offset,
                expected_digest=journal_digest.hexdigest(),
                category="metadata_changed_during_load",
                expected_identity=(journal_info.st_dev, journal_info.st_ino),
            )
            journal_digest = checked_digest
            self._record_digest_recheck(head, reason="metadata_changed_during_load", elapsed=elapsed)
        self._checkpoint_stat = (checkpoint_info.st_dev, checkpoint_info.st_ino, checkpoint_info.st_size, checkpoint_info.st_mtime_ns, checkpoint_info.st_ctime_ns)
        self._publish_verified_digest(journal_digest, head, journal_info_after)
        self._record_count = int(raw.get("expected_screen_count", -1))
        return document, head, {"checkpoint_bytes_read": len(checkpoint_bytes), "journal_bytes_replayed": replayed_bytes, "head_bytes_read": head_bytes}

    def refresh(self, document: TrackedDict, current: JournalHead, changes: _ChangeSet) -> tuple[JournalHead, dict[str, int]]:
        self.check_cached_checkpoint()
        head, _, head_bytes = _strict_head(self.head_path, self.root)
        if (head.lineage_id, head.checkpoint_sha256) != (current.lineage_id, current.checkpoint_sha256):
            raise JournalError("journal lineage or checkpoint identity changed")
        journal_info = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
        if self._journal_identity != (journal_info.st_dev, journal_info.st_ino):
            raise JournalError("manifest journal inode changed during lifecycle use")
        if journal_info.st_size != head.committed_offset:
            raise JournalError("journal size differs from committed HEAD offset; preserving uncommitted or truncated bytes")
        if head.sequence < current.sequence or head.committed_offset < current.committed_offset:
            raise JournalError("journal HEAD moved backwards")
        cached_digest = self._require_cached_digest(current)
        if head.sequence == current.sequence:
            if head.transaction_sha256 != current.transaction_sha256 or head.committed_offset != current.committed_offset:
                raise JournalError("journal HEAD changed without a sequence advance")
            observed = _journal_stat_tuple(journal_info)
            if self._journal_stat is None:
                raise JournalError("journal authority has no cached stat for its verified digest")
            if observed != self._journal_stat:
                checked_digest, checked_info, elapsed = self._verify_cached_prefix(
                    head,
                    prefix_length=head.committed_offset,
                    expected_size=head.committed_offset,
                    expected_digest=cached_digest.hexdigest(),
                    category="metadata_changed_without_HEAD_advance",
                )
                self._publish_verified_digest(checked_digest, head, checked_info)
                self._record_digest_recheck(head, reason="metadata_changed_without_HEAD_advance", elapsed=elapsed)
            return head, {"head_bytes_read": head_bytes, "journal_bytes_replayed": 0}

        # The lifecycle lock and append-only commit boundary bind the cached
        # prefix. Extend its raw digest only with transactions validated below.
        candidate_digest = cached_digest.copy()
        offset = current.committed_offset
        sequence = current.sequence
        previous = current.transaction_sha256
        replayed_bytes = 0
        with self.journal_path.open("rb") as handle:
            handle.seek(offset)
            while offset < head.committed_offset:
                line = handle.readline(MAX_TRANSACTION_BYTES + 1)
                if not line or len(line) > MAX_TRANSACTION_BYTES or not line.endswith(b"\n"):
                    raise JournalError("journal delta contains a truncated transaction")
                tx = self._decode_transaction(line[:-1], sequence + 1, previous, head.lineage_id, document)
                for op in tx["operations"]:
                    _apply_op(document, op, changes)
                validate_projection(document, tx["operations"])
                sequence = tx["sequence"]
                previous = tx["transaction_sha256"]
                offset += len(line)
                replayed_bytes += len(line)
                candidate_digest.update(line)
        if offset != head.committed_offset or sequence != head.sequence or previous != head.transaction_sha256:
            raise JournalError("journal delta does not terminate at the committed HEAD")
        head_after, _, _ = _strict_head(self.head_path, self.root)
        journal_info_after = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
        if head_after != head or _journal_identity_size(journal_info_after) != _journal_identity_size(journal_info):
            raise JournalError(
                "journal or HEAD changed during delta replay",
                details={
                    "category": "journal_snapshot_unstable_during_refresh",
                    "head_before": _head_context(head),
                    "head_after": _head_context(head_after),
                    "stat_before_delta": _journal_stat_context(journal_info),
                    "stat_after_delta": _journal_stat_context(journal_info_after),
                    "expected_prefix_sha256": cached_digest.hexdigest(),
                    "candidate_sha256": candidate_digest.hexdigest(),
                },
            )
        if _journal_stat_tuple(journal_info_after) != _journal_stat_tuple(journal_info):
            checked_digest, journal_info_after, elapsed = self._verify_cached_prefix(
                head,
                prefix_length=head.committed_offset,
                expected_size=head.committed_offset,
                expected_digest=candidate_digest.hexdigest(),
                category="metadata_changed_during_delta_refresh",
            )
            candidate_digest = checked_digest
            self._record_digest_recheck(head, reason="metadata_changed_during_delta_refresh", elapsed=elapsed)
        self._publish_verified_digest(candidate_digest, head, journal_info_after)
        return head, {"head_bytes_read": head_bytes, "journal_bytes_replayed": replayed_bytes}

    def _decode_transaction(self, data: bytes, expected_sequence: int, expected_previous: str, lineage_id: str, document: Mapping[str, Any]) -> dict[str, Any]:
        if len(data) > MAX_TRANSACTION_BYTES:
            raise JournalError("journal transaction exceeds size limit")
        try:
            tx = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise JournalError(f"journal transaction is unreadable: {error}") from error
        if not isinstance(tx, dict) or tx.get("schema") != JOURNAL_SCHEMA:
            raise JournalError("journal transaction schema is invalid")
        supplied_hash = tx.pop("transaction_sha256", None)
        if not isinstance(supplied_hash, str) or len(supplied_hash) != 64 or _sha(_canonical(tx)) != supplied_hash:
            raise JournalError("journal transaction checksum mismatch")
        if (
            tx.get("sequence") != expected_sequence
            or isinstance(tx.get("sequence"), bool)
            or tx.get("previous_transaction_sha256") != expected_previous
            or tx.get("lineage_id") != lineage_id
            or not isinstance(tx.get("operation"), str)
            or not isinstance(tx.get("operations"), list)
            or not tx["operations"]
        ):
            raise JournalError("journal transaction sequence, lineage, or fields are invalid")
        record_count = int(document.get("expected_screen_count", -1))
        allow_backlog_rebuild = tx["operation"] == "reconstruct_queue"
        for op in tx["operations"]:
            if not isinstance(op, dict):
                raise JournalError("journal transaction operation is not an object")
            _validate_path_op(op, record_count, operation=tx["operation"], allow_backlog_rebuild=allow_backlog_rebuild)
            if op["path"][0] != "records":
                continue
            index, field = op["path"][1:]
            if field == "transitions" and op["value"].get("actor") != tx.get("actor"):
                raise JournalError("transition actor differs from transaction actor")
            if field == "hydro_owner" and op.get("value") is not None and op["value"]["process"] != tx.get("process"):
                raise JournalError("claim process differs from transaction identity")
            if field == "hydro_owner" and op.get("value") is not None and op["value"]["actor"] != tx.get("actor"):
                raise JournalError("claim actor differs from transaction actor")
            if tx["operation"] in {"begin_hydro_screen", "commit_next"}:
                owner = document["records"][index].get("hydro_owner")
                if not isinstance(owner, dict) or owner.get("process") != tx.get("process"):
                    raise JournalError("hydro transaction is not owned by its durable process")
                if field == "transitions" and op["value"].get("actor") != owner.get("actor"):
                    raise JournalError("hydro transition actor differs from durable owner")
        tx["transaction_sha256"] = supplied_hash
        return tx

    def commit(self, operations: Sequence[Mapping[str, Any]], *, previous: JournalHead, operation: str, actor: str, process: Mapping[str, Any]) -> tuple[JournalHead, dict[str, int]]:
        if not operations:
            return previous, {"journal_bytes_written": 0, "head_bytes_written": 0, "fsync_count": 0, "replace_count": 0}
        head, _, _ = _strict_head(self.head_path, self.root)
        if head != previous:
            raise JournalError("commit attempted from a stale lifecycle HEAD")
        journal_info = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
        try:
            size = journal_info.st_size
        except OSError as error:
            raise JournalError(f"cannot stat lifecycle journal: {error}") from error
        if size != previous.committed_offset:
            raise JournalError("journal has uncommitted bytes before append; preserving evidence")
        if (journal_info.st_dev, journal_info.st_ino) != self._journal_identity:
            raise JournalError("manifest journal inode changed before commit")
        cached_digest = self._require_cached_digest(previous)
        if self._journal_stat is None:
            raise JournalError("journal authority has no cached stat for its verified digest")
        if _journal_stat_tuple(journal_info) != self._journal_stat:
            checked_digest, checked_info, elapsed = self._verify_cached_prefix(
                previous,
                prefix_length=previous.committed_offset,
                expected_size=previous.committed_offset,
                expected_digest=cached_digest.hexdigest(),
                category="metadata_changed_before_commit",
            )
            self._publish_verified_digest(checked_digest, previous, checked_info)
            self._record_digest_recheck(previous, reason="metadata_changed_before_commit", elapsed=elapsed)
            cached_digest = self._require_cached_digest(previous)
        record_count = self._record_count
        if record_count is None or record_count <= 0:
            raise JournalError("journal authority has no validated screen count")
        allow_backlog_rebuild = operation == "reconstruct_queue"
        for op in operations:
            _validate_path_op(op, record_count, operation=operation, allow_backlog_rebuild=allow_backlog_rebuild)
        body = {
            "schema": JOURNAL_SCHEMA,
            "lineage_id": previous.lineage_id,
            "sequence": previous.sequence + 1,
            "previous_transaction_sha256": previous.transaction_sha256,
            "operation": str(operation),
            "actor": str(actor),
            "process": _plain(dict(process)),
            "committed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "operations": [_plain(dict(item)) for item in operations],
        }
        digest = _sha(_canonical(body))
        body["transaction_sha256"] = digest
        line = _canonical(body) + b"\n"
        if len(line) > MAX_TRANSACTION_BYTES:
            raise JournalError("lifecycle transaction exceeds journal record size limit")
        candidate_digest = cached_digest.copy()
        candidate_digest.update(line)
        append_started = time.perf_counter()
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(str(self.journal_path), flags)
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != self._journal_identity
                or info.st_size != previous.committed_offset
            ):
                os.close(descriptor)
                raise JournalError("journal file changed before durable append")
            with os.fdopen(descriptor, "ab", buffering=0) as handle:
                view = memoryview(line)
                while view:
                    written = handle.write(view)
                    if not written:
                        raise OSError("short journal append")
                    view = view[written:]
                os.fsync(handle.fileno())
        except OSError as error:
            raise JournalError(f"durable journal append failed: {error}") from error
        append_elapsed = time.perf_counter() - append_started
        new_head = JournalHead(previous.lineage_id, previous.checkpoint_sha256, previous.sequence + 1, previous.committed_offset + len(line), digest)
        head_started = time.perf_counter()
        head_bytes = _atomic_head(self.head_path, new_head.public())
        head_elapsed = time.perf_counter() - head_started
        journal_info_after = _safe_regular(self.journal_path, expected_parent=self.root, label="manifest journal")
        if (
            journal_info_after.st_size != new_head.committed_offset
            or (journal_info_after.st_dev, journal_info_after.st_ino) != self._journal_identity
        ):
            raise JournalError(
                "journal changed after durable commit publication",
                details={
                    "category": "journal_snapshot_unstable_after_commit",
                    "expected_head": _head_context(new_head),
                    "expected_sha256": candidate_digest.hexdigest(),
                    "stat_after": _journal_stat_context(journal_info_after),
                },
            )
        self._publish_verified_digest(candidate_digest, new_head, journal_info_after)
        return new_head, {
            "journal_bytes_written": len(line),
            "head_bytes_written": head_bytes,
            "journal_append_elapsed_s": append_elapsed,
            "head_replace_elapsed_s": head_elapsed,
            "fsync_count": 3 if os.name != "nt" else 2,
            "replace_count": 1,
        }

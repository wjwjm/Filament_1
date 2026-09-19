"""Durable R/C pair orchestration for the bounded E5-1A fixture.

The coordinator records the order of already-existing scientific calls.  It
does not implement a scheduler, a new hydro operator, or a second state
authority.  Reference and candidate callbacks receive independent roots and
must return their own durable reports; a pair cannot advance until both sides,
the exact comparison, successor READY, and the reclamation step have passed.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .hr4e5_evidence import (atomic_json, compare_object_sets, sha256_file,
                              validate_durable_report, validate_paired_exact_report,
                              validate_paired_exact_report_metadata, validate_ready_receipt)
from .hr4e5_formal_entry import create_successor_root
from .hr4e5_storage import (HARD_CAP_BYTES, StorageBudget, StorageBudgetError,
                            StorageIntegrityError, _FileLock, process_identity,
                            probe_process_identity)


CAMPAIGN_SCHEMA = "khz_filament.hr4e5.e5_1a.paired_campaign.v1"
CAMPAIGN_STATE_FILENAME = "E5_1A_CAMPAIGN_STATE.json"
ADMISSION_FILENAME = "E5_1A_ADMISSION_IDENTITY.json"
_FORMAL_STEPS = frozenset({
    "reference", "candidate", "exact", "reference_successor", "reference_gc",
    "candidate_successor", "candidate_gc", "terminal",
})
_FORMAL_STATUSES = frozenset({"IN_PROGRESS", "COMMITTED", "PASS", "INTERRUPTED", "FAIL"})
_FORMAL_RECEIPT_STATUSES = frozenset({"COMMITTED", "PASS"})
_PRODUCTION_DRIVER_CAPABILITY = object()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _status(value: Any, *, default: str = "PASS") -> tuple[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        payload = dict(value)
        if not payload:
            return "FAIL", {"status": "FAIL", "error": "step returned an empty receipt"}
        if "status" not in payload:
            return "FAIL", {"status": "FAIL", "error": "step receipt must declare status"}
        status = str(payload["status"]).upper()
    elif value is True or value is None:
        payload = {"status": "FAIL", "error": "step callback returned no durable receipt"}
        status = "FAIL"
    elif value is False:
        payload = {}
        status = "FAIL"
    else:
        payload = {"value": value}
        status = default
    return status, payload


def _callback_receipt(value: Any, *, step: str) -> Any:
    """Require an explicit, persistable receipt from every successful step."""
    status, payload = _status(value, default="FAIL")
    if status != "PASS":
        return {**payload, "status": status}
    receipt_keys = {
        "receipt", "receipt_path", "report_path", "path", "ready",
        "terminal_receipt_path", "evidence_path",
    }
    if not any(key in payload for key in receipt_keys):
        return {"status": "FAIL", "error": f"{step} callback omitted a durable receipt"}
    return payload


def validate_overlap_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate overlap from persisted work events, without synthetic timing."""
    rows = [dict(item) for item in events]
    if not rows:
        return {"status": "FAIL", "reason": "no_events", "event_count": 0}
    times = []
    for row in rows:
        value = row.get("monotonic_s")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return {"status": "FAIL", "reason": "event_time_missing_or_nonfinite", "event_count": len(rows)}
        times.append(float(value))
    if any(right < left for left, right in zip(times, times[1:])):
        return {"status": "FAIL", "reason": "event_times_not_monotonic", "event_count": len(rows)}
    starts = [float(row["monotonic_s"]) for row in rows if str(row.get("event")) == "OPTICAL_START"]
    completes = [float(row["monotonic_s"]) for row in rows if str(row.get("event")) == "OPTICAL_COMPLETE"]
    hydro = [float(row["monotonic_s"]) for row in rows if str(row.get("event")) == "HYDRO_BLOCK_START"]
    if not starts or not completes or not hydro:
        return {"status": "FAIL", "reason": "required_work_events_missing", "event_count": len(rows), "optical_start_count": len(starts), "hydro_start_count": len(hydro), "optical_complete_count": len(completes)}
    optical_start, optical_complete = starts[0], completes[-1]
    overlap = any(optical_start < value < optical_complete for value in hydro)
    return {
        "status": "PASS" if overlap else "FAIL",
        "reason": "hydro_started_before_optical_complete" if overlap else "hydro_started_after_optical_complete",
        "event_count": len(rows), "optical_start": optical_start,
        "optical_complete": optical_complete, "hydro_block_starts": hydro,
    }


@dataclass(frozen=True)
class PairResult:
    pulse_index: int
    status: str
    reference_status: str
    candidate_status: str
    compare_status: str
    successor_status: str
    reclaim_status: str


def _serialized_transition(method):
    @wraps(method)
    def serialized(self, *args, **kwargs):
        with self._locked_state():
            return method(self, *args, **kwargs)
    return serialized

class PairedCampaign:
    """Persisted state machine for serial R -> C -> exact -> GC pairs."""

    def __init__(self, root: str | Path, *, create: bool = False, n_pulses: int = 3,
                 block_size: int = 8, queue_depth: int = 16,
                 max_campaign_live_bytes: int = HARD_CAP_BYTES,
                 final_output_budget_bytes: int = 64 * 1024**3,
                 safety_margin_bytes: int = 8 * 1024**3,
                 require_quota: bool = True,
                 campaign_id: str = "e5_1a_local",
                 admission_identity: Mapping[str, Any] | None = None,
                 formal: bool = False,
                 require_intents: bool | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / CAMPAIGN_STATE_FILENAME
        self.n_pulses = int(n_pulses)
        self.block_size = int(block_size)
        self.queue_depth = int(queue_depth)
        self.max_campaign_live_bytes = int(max_campaign_live_bytes)
        self.final_output_budget_bytes = int(final_output_budget_bytes)
        self.safety_margin_bytes = int(safety_margin_bytes)
        self.require_quota = bool(require_quota)
        self.campaign_id = str(campaign_id)
        self.formal = bool(formal)
        if admission_identity is not None and str(admission_identity.get("execution_mode", "")).upper() != "TEST_FIXTURE_ONLY":
            self.formal = True
        self.admission_identity = None if admission_identity is None else dict(admission_identity)
        if self.formal:
            from .hr4e5_formal_entry import validate_admission_identity
            if self.admission_identity is None and create:
                raise ValueError("formal campaign requires an immutable admission identity")
            if self.admission_identity is not None:
                self.admission_identity = validate_admission_identity(self.admission_identity, formal=True)
                if str(self.admission_identity.get("campaign_id")) != self.campaign_id:
                    raise StorageBudgetError("admission campaign identity differs from campaign")
        if self.n_pulses <= 0 or self.block_size != 8 or self.queue_depth != 16:
            raise ValueError("campaign requires positive N and frozen block8/queue16")
        if self.n_pulses > 1 and self.block_size > 0:
            # K is checked on each lifecycle; this guard prevents an accidental
            # E5 campaign from silently choosing an unsupported block contract.
            pass
        if create:
            if self.state_path.exists():
                raise FileExistsError(self.state_path)
            state = self._new_state()
            atomic_json(self.state_path, state, overwrite=False)
        if not self.state_path.is_file():
            raise FileNotFoundError(self.state_path)
        if not create:
            raw_state = self._read_state()
            if bool(raw_state.get("formal", False)):
                self.formal = True
                persisted_identity = raw_state.get("admission_identity")
                if self.admission_identity is None:
                    self.admission_identity = dict(persisted_identity or {})
                from .hr4e5_formal_entry import validate_admission_identity
                self.admission_identity = validate_admission_identity(self.admission_identity, formal=True,
                                                                       expected_hash=raw_state.get("admission_hash"))
        self.state = self._read_state()
        self._validate_state(self.state)
        if self.admission_identity is not None:
            admission_path = self.root / ADMISSION_FILENAME
            if admission_path.is_file():
                from .hr4e5_formal_entry import validate_admission_identity
                persisted_identity = validate_admission_identity(
                    json.loads(admission_path.read_text(encoding="utf-8")),
                    formal=self.formal,
                    expected_hash=self.admission_identity.get("identity_sha256"),
                )
                if persisted_identity.get("identity_sha256") != self.admission_identity.get("identity_sha256"):
                    raise StorageIntegrityError("persisted admission identity differs from campaign")
            else:
                atomic_json(admission_path, self.admission_identity, overwrite=False)
        # The campaign ledger covers every file created below the campaign
        # root, including R/C roots, receipts, manifests, diagnostics, and
        # failed-attempt residue.  A subdirectory ledger would allow a caller
        # to evade the single-campaign cap by changing directories.
        self.storage = StorageBudget(
            self.root, cap_bytes=self.max_campaign_live_bytes,
            final_output_budget_bytes=self.final_output_budget_bytes,
            safety_margin_bytes=self.safety_margin_bytes, require_quota=self.require_quota,
            campaign_id=self.campaign_id,
            require_intents=self.formal if require_intents is None else bool(require_intents),
            admission_hash=None if self.admission_identity is None else str(self.admission_identity.get("identity_sha256")),
        )

    @contextmanager
    def _locked_state(self):
        """Serialize state transitions across coordinators and processes."""
        with _FileLock(self.root / ".E5_1A_CAMPAIGN.lock"):
            self.state = self._read_state()
            self._validate_state(self.state)
            yield

    @classmethod
    def create(cls, root: str | Path, **kwargs: Any) -> "PairedCampaign":
        return cls(root, create=True, **kwargs)

    @classmethod
    def open(cls, root: str | Path, **kwargs: Any) -> "PairedCampaign":
        return cls(root, create=False, **kwargs)

    @classmethod
    def resume_from_disk(cls, root: str | Path, **kwargs: Any) -> "PairedCampaign":
        """Alias used by the new-process persistence fixture."""
        return cls.open(root, **kwargs)

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": self.campaign_id,
            "n_pulses": self.n_pulses,
            "block_size": self.block_size,
            "queue_depth": self.queue_depth,
            "max_campaign_live_bytes": self.max_campaign_live_bytes,
            "final_output_budget_bytes": self.final_output_budget_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "require_quota": self.require_quota,
            "formal": self.formal,
            "admission_identity": None if self.admission_identity is None else dict(self.admission_identity),
            "admission_hash": None if self.admission_identity is None else self.admission_identity.get("identity_sha256"),
            "status": "READY",
            "next_pair_index": 0,
            "pairs": [self._new_pair(index) for index in range(self.n_pulses)],
            "process_epochs": [],
            "active_coordinator": None,
            "created_utc": _utc(),
            "updated_utc": _utc(),
        }

    @staticmethod
    def _new_pair(index: int) -> dict[str, Any]:
        return {
            "pulse_index": int(index),
            "reference": {"status": "PENDING"},
            "candidate": {"status": "PENDING"},
            "compare": {"status": "PENDING"},
            "successor": {"status": "PENDING"},
            "reclaim": {"status": "PENDING"},
            "formal_steps": {},
            "events": [],
        }

    def _read_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("campaign state is unreadable") from error
        if not isinstance(value, dict):
            raise ValueError("campaign state must be an object")
        return value

    def _validate_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != CAMPAIGN_SCHEMA:
            raise ValueError("campaign state schema is invalid")
        if str(state.get("campaign_id", "")) != self.campaign_id:
            raise StorageBudgetError("campaign state campaign identity conflicts with requested campaign")
        if int(state.get("n_pulses", -1)) != self.n_pulses:
            raise ValueError("campaign pulse count conflicts with state")
        if int(state.get("block_size", -1)) != self.block_size or int(state.get("queue_depth", -1)) != self.queue_depth:
            raise ValueError("campaign frozen block or queue contract conflicts with state")
        if int(state.get("max_campaign_live_bytes", -1)) != self.max_campaign_live_bytes:
            raise ValueError("campaign hard cap conflicts with state")
        if int(state.get("final_output_budget_bytes", -1)) != self.final_output_budget_bytes:
            raise ValueError("campaign final output budget conflicts with state")
        if bool(state.get("require_quota", False)) != self.require_quota:
            raise StorageBudgetError("campaign quota requirement conflicts with state")
        if bool(state.get("formal", False)) != self.formal:
            raise StorageIntegrityError("campaign formal mode conflicts with state")
        if self.formal:
            from .hr4e5_formal_entry import validate_admission_identity
            persisted = state.get("admission_identity")
            if self.admission_identity is None or not isinstance(persisted, Mapping):
                raise StorageIntegrityError("formal campaign admission identity is missing")
            validate_admission_identity(persisted, formal=True, expected_hash=state.get("admission_hash"))
            if str(state.get("admission_hash")) != str(self.admission_identity.get("identity_sha256")):
                raise StorageIntegrityError("campaign admission identity conflicts with requested identity")
        pairs = state.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != self.n_pulses:
            raise ValueError("campaign pair list is invalid")
        if [int(item.get("pulse_index", -1)) for item in pairs] != list(range(self.n_pulses)):
            raise ValueError("campaign pair order is invalid")
        next_pair = state.get("next_pair_index")
        if isinstance(next_pair, bool) or not isinstance(next_pair, int) or not 0 <= next_pair <= self.n_pulses:
            raise ValueError("campaign next pair index is invalid")
        active = state.get("active_coordinator")
        if active is not None and not isinstance(active, Mapping):
            raise StorageIntegrityError("active coordinator receipt is invalid")
        if self.formal:
            for pair in pairs:
                steps = pair.get("formal_steps", {})
                if steps is not None and not isinstance(steps, Mapping):
                    raise StorageIntegrityError("formal pair step state is invalid")
            if next_pair > 0 and any(str(pairs[index].get("status")) != "PASS" for index in range(next_pair)):
                raise StorageIntegrityError("campaign index advanced before durable pair completion")
            if any(str(pair.get("status", "")) == "PASS" for pair in pairs[next_pair:]):
                raise StorageIntegrityError("campaign index is behind a durable pair completion")

    def _persist(self) -> None:
        self.state["updated_utc"] = _utc()
        self._validate_state(self.state)
        atomic_json(self.state_path, self.state)

    def _coordinator_state_enabled(self) -> bool:
        identity = self.state.get("admission_identity")
        fixture = isinstance(identity, Mapping) and identity.get("execution_mode") == "TEST_FIXTURE_ONLY"
        return bool(self.formal or fixture)

    def _active_coordinator_epoch(self, supplied: str | int | None = None) -> Any:
        """Return the live coordinator epoch and reject stale callers."""
        active = self.state.get("active_coordinator")
        if not self._coordinator_state_enabled() or not isinstance(active, Mapping):
            raise StorageIntegrityError("formal transition requires an active coordinator")
        if str(active.get("status", "")) != "ACTIVE" or active.get("epoch") is None:
            raise StorageIntegrityError("formal transition requires an active coordinator epoch")
        process_id = str(active.get("process_id", ""))
        current = next((item for item in self.state.get("process_epochs", [])
                        if str(item.get("process_id", "")) == process_id
                        and str(item.get("status", "")) == "ACTIVE"), None)
        if not isinstance(current, Mapping) or str(current.get("epoch")) != str(active.get("epoch")):
            raise StorageIntegrityError("formal active coordinator epoch is stale or inconsistent")
        if supplied is not None and str(supplied) != str(active.get("epoch")):
            raise StorageIntegrityError("formal transition uses a stale coordinator epoch")
        return active.get("epoch")

    def active_coordinator_epoch(self) -> Any:
        """Read the current epoch for a controlled driver call."""
        return self._active_coordinator_epoch()

    def _durable_receipt_fields(self, receipt: Mapping[str, Any] | None, *, label: str) -> tuple[str, str]:
        if not isinstance(receipt, Mapping):
            raise StorageIntegrityError(f"{label} requires a durable receipt")
        raw_path = receipt.get("report_path") or receipt.get("receipt_path") or receipt.get("path")
        raw_hash = receipt.get("report_sha256") or receipt.get("receipt_sha256") or receipt.get("sha256")
        if not raw_path or not raw_hash:
            raise StorageIntegrityError(f"{label} requires a durable receipt path and hash")
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if not path.is_relative_to(self.root) or path.is_symlink() or not path.is_file():
            raise StorageIntegrityError(f"{label} receipt is missing or outside campaign")
        digest = sha256_file(path)
        if digest != str(raw_hash):
            raise StorageIntegrityError(f"{label} receipt hash changed")
        return str(path), digest

    def _validate_saved_formal_receipt(self, saved: Mapping[str, Any], *, label: str) -> tuple[str, str]:
        path, digest = self._durable_receipt_fields(
            {"receipt_path": saved.get("receipt_path"), "receipt_sha256": saved.get("receipt_sha256")},
            label=label,
        )
        return path, digest

    def _validate_crash_takeover_receipt(self, receipt: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        active = self.state.get("active_coordinator")
        if not isinstance(active, Mapping) or str(active.get("status")) != "ACTIVE":
            raise StorageIntegrityError("formal crash takeover requires an active stale coordinator")
        if isinstance(receipt, Mapping):
            raw_path = receipt.get("path") or receipt.get("receipt_path")
            expected_hash = receipt.get("sha256")
        else:
            raw_path, expected_hash = receipt, None
        if raw_path is None or expected_hash is None:
            raise StorageIntegrityError("crash takeover requires a durable receipt path and hash")
        path = Path(str(raw_path)).resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or path.is_symlink():
            raise StorageIntegrityError("crash takeover receipt is missing or outside campaign")
        digest = sha256_file(path)
        if digest != str(expected_hash):
            raise StorageIntegrityError("crash takeover receipt hash changed")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageIntegrityError("crash takeover receipt is unreadable") from error
        if (not isinstance(payload, Mapping) or str(payload.get("status", "")).upper() != "PASS"
                or str(payload.get("campaign_id", self.campaign_id)) != self.campaign_id
                or payload.get("active_writers", payload.get("active_writer_ids", [])) not in ([], (), None)):
            raise StorageIntegrityError("crash takeover receipt is not a quiescent PASS")
        process_id = payload.get("coordinator_process_id", payload.get("stale_coordinator_process_id"))
        if str(process_id) != str(active.get("process_id")):
            raise StorageIntegrityError("crash takeover receipt coordinator identity changed")
        epoch = payload.get("coordinator_epoch", payload.get("stale_coordinator_epoch", payload.get("writer_epoch")))
        if str(epoch) != str(active.get("epoch")):
            raise StorageIntegrityError("crash takeover receipt epoch changed")
        if payload.get("stale") is not True and payload.get("exit_observed") is not True:
            raise StorageIntegrityError("crash takeover receipt lacks stale/exit evidence")
        return {"path": str(path), "sha256": digest, "coordinator_process_id": str(process_id),
                "coordinator_epoch": str(epoch), "status": "PASS"}

    def _authorize_crash_takeover_locked(self, receipt: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
        if receipt is not None:
            raise StorageIntegrityError("external crash takeover receipts are forbidden")
        active = self.state["active_coordinator"]
        previous = next((item for item in reversed(self.state.get("process_epochs", []))
                         if str(item.get("process_id")) == str(active.get("process_id"))
                         and item.get("status") == "ACTIVE"), None)
        if not isinstance(previous, Mapping):
            raise StorageIntegrityError("active coordinator epoch record is missing")
        probe = probe_process_identity(previous.get("process_identity", {}))
        if probe["status"] == "LIVE":
            raise StorageIntegrityError("live ACTIVE coordinator blocks takeover")
        if probe["status"] != "DEAD":
            raise StorageIntegrityError("coordinator liveness cannot be verified")
        stale = self.storage.interrupt_stale_writers(coordinator_epoch=active.get("epoch"))
        generated = self.storage.write_quiescence_receipt(
            self.root / ".takeover" / f"epoch_{active.get('epoch')}.json",
            coordinator_epoch=active.get("epoch"),
        )
        path = Path(str(generated["path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({"stale": True, "exit_observed": True,
                        "coordinator_process_id": str(active.get("process_id")),
                        "coordinator_epoch": str(active.get("epoch")),
                        "liveness_evidence": probe, "writer_transition": stale})
        atomic_json(path, payload)
        receipt = {"path": str(path), "sha256": sha256_file(path)}
        info = self._validate_crash_takeover_receipt(receipt)
        for epoch in reversed(self.state.get("process_epochs", [])):
            if str(epoch.get("process_id")) == str(active.get("process_id")) and epoch.get("status") == "ACTIVE":
                epoch["status"] = "EXITED_STALE"
                epoch["exited_utc"] = _utc()
                epoch["exit_reason"] = "durable_crash_takeover_receipt"
                epoch["exit_receipt"] = info
                break
        self.state["active_coordinator"] = None
        self.state["status"] = "READY" if self.next_pair_index < self.n_pulses else "COMPLETE"
        self.state.setdefault("events", []).append({"event": "CRASH_TAKEOVER_AUTHORIZED", **info, "timestamp_utc": _utc()})
        self._persist()
        return info

    @_serialized_transition
    def authorize_crash_takeover(self, receipt: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Authorize takeover only from hash-bound durable stale/exit evidence."""
        if not self.formal:
            raise StorageIntegrityError("crash takeover evidence is formal-only")
        return self._authorize_crash_takeover_locked(receipt)

    def _pair(self, pulse_index: int) -> dict[str, Any]:
        index = int(pulse_index)
        if index < 0 or index >= self.n_pulses:
            raise IndexError("pair pulse index is outside campaign")
        return self.state["pairs"][index]

    @property
    def next_pair_index(self) -> int:
        return int(self.state["next_pair_index"])

    @property
    def is_complete(self) -> bool:
        return self.state.get("status") == "COMPLETE"

    @_serialized_transition
    def register_process_start(self, *, process_id: str | None = None, source: str = "new_process",
                               takeover: bool = False,
                               takeover_receipt: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
        already_complete = self.is_complete
        active = self.state.get("active_coordinator")
        coordinator_enabled = self._coordinator_state_enabled()
        if coordinator_enabled and isinstance(active, Mapping) and str(active.get("status")) == "ACTIVE":
            previous = next((item for item in reversed(self.state.get("process_epochs", []))
                             if str(item.get("process_id")) == str(active.get("process_id"))), None)
            if not takeover or not isinstance(previous, Mapping):
                raise StorageIntegrityError("formal campaign already has an active coordinator")
            if previous.get("status") != "EXITED":
                self._authorize_crash_takeover_locked(takeover_receipt)
            else:
                self.state["active_coordinator"] = None
        process_name = str(process_id or f"pid:{os.getpid()}:{uuid.uuid4().hex}")
        epoch = {"epoch": len(self.state["process_epochs"]), "process_id": process_name,
                 "process_identity": process_identity(), "source": str(source),
                 "started_utc": _utc(), "status": "ACTIVE"}
        self.state["process_epochs"].append(epoch)
        if coordinator_enabled:
            self.state["active_coordinator"] = {"epoch": epoch["epoch"], "process_id": process_name, "status": "ACTIVE", "started_utc": epoch["started_utc"]}
        self.state["status"] = "COMPLETE" if already_complete else "RUNNING"
        self._persist()
        return epoch

    @_serialized_transition
    def register_process_exit(self, *, process_id: str | None = None) -> dict[str, Any]:
        target = str(process_id or f"pid:{os.getpid()}")
        for epoch in reversed(self.state["process_epochs"]):
            if epoch.get("process_id") == target and epoch.get("status") == "ACTIVE":
                epoch["status"] = "EXITED"
                epoch["exited_utc"] = _utc()
                if self._coordinator_state_enabled() and isinstance(self.state.get("active_coordinator"), Mapping) and self.state["active_coordinator"].get("process_id") == target:
                    self.state["active_coordinator"] = None
                self._persist()
                return dict(epoch)
        raise ValueError("active process epoch is missing")

    def _record_step(self, pair: dict[str, Any], name: str, value: Any, *, default: str = "PASS") -> str:
        status, payload = _status(value, default=default)
        pair[name] = {**payload, "status": status, "recorded_utc": _utc()}
        pair["events"].append({"event": f"{name.upper()}_{status}", "timestamp_utc": _utc()})
        self._persist()
        return status

    @_serialized_transition
    def formal_step(self, pulse_index: int, step: str, *, status: str,
                    receipt: Mapping[str, Any] | None = None,
                    error: str | None = None, epoch: str | int | None = None) -> dict[str, Any]:
        """Persist one fine-grained formal-driver transition.

        The method is intentionally small and receipt-oriented.  Scientific
        execution remains in the supplied trusted runner; this state store
        records only durable step boundaries and refuses index advancement
        until the coordinator commits the complete sequence.
        """
        if not self._coordinator_state_enabled():
            raise StorageIntegrityError("formal step state requires a formal campaign")
        if epoch is None:
            raise StorageIntegrityError("formal step requires the current coordinator epoch")
        active_epoch = self._active_coordinator_epoch(epoch)
        index = int(pulse_index)
        if index != self.next_pair_index:
            raise StorageIntegrityError("formal step is not on the durable pair index")
        pair = self._pair(index)
        steps = pair.setdefault("formal_steps", {})
        key = str(step)
        state = str(status).upper()
        if key not in _FORMAL_STEPS:
            raise StorageIntegrityError(f"formal step name is not in the contract: {key}")
        final_pair = index == self.n_pulses - 1
        if (final_pair and key in {"reference_successor", "reference_gc", "candidate_successor", "candidate_gc"}) or (
            not final_pair and key == "terminal"
        ):
            raise StorageIntegrityError(f"formal step is not legal for pulse {index}: {key}")
        if state not in _FORMAL_STATUSES:
            raise StorageIntegrityError(f"formal step status is not in the contract: {state}")
        saved = steps.get(key)
        if state in _FORMAL_RECEIPT_STATUSES:
            receipt_path, receipt_hash = self._durable_receipt_fields(receipt, label=f"formal {key}")
            if self.formal and key.endswith("_gc"):
                self._verify_formal_gc_receipt(key, index, receipt or {})
        else:
            receipt_path, receipt_hash = None, None
            if receipt is not None:
                raise StorageIntegrityError(f"formal {key} non-terminal state cannot carry a receipt")
        if saved is not None and str(saved.get("status", "")).upper() in _FORMAL_RECEIPT_STATUSES:
            if state not in _FORMAL_RECEIPT_STATUSES:
                raise StorageIntegrityError(f"formal step {key} cannot regress from a durable receipt")
            _, saved_hash = self._validate_saved_formal_receipt(saved, label=f"formal {key}")
            if saved_hash != receipt_hash:
                raise StorageIntegrityError(f"formal step receipt changed: {key}")
            return dict(saved)
        payload = {
            "step": key, "pulse_index": index, "status": state,
            "epoch": active_epoch, "coordinator_process_id": self.state["active_coordinator"]["process_id"],
            "recorded_utc": _utc(),
        }
        if receipt_path is not None:
            payload.update({"receipt_path": receipt_path, "receipt_sha256": receipt_hash,
                            "receipt": dict(receipt or {})})
        if error is not None:
            payload["error"] = str(error)
        steps[key] = payload
        pair["events"].append({"event": f"FORMAL_{key.upper()}_{state}", "timestamp_utc": _utc(), "epoch": payload.get("epoch")})
        self._persist()
        return dict(payload)

    def _verify_formal_gc_receipt(
        self, step: str, pulse_index: int, receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind a committed formal GC step to its durable, scoped plan."""
        plan_id = receipt.get("gc_plan_id")
        if not plan_id:
            raise StorageIntegrityError(f"formal {step} requires a durable GC plan id")
        trajectory = "R" if str(step).startswith("reference") else "C"
        verified = self.storage.verify_reclaim(
            str(plan_id), expected_trajectory=trajectory,
            expected_pulse=int(pulse_index), expected_attempt=0,
            expected_admission_hash=str(self.state.get("admission_hash", "")),
        )
        if verified.get("status") != "PASS":
            raise StorageIntegrityError(f"formal {step} GC plan is not durably complete")
        return verified

    def formal_step_state(self, pulse_index: int, step: str) -> dict[str, Any]:
        pair = self._pair(int(pulse_index))
        return dict(pair.get("formal_steps", {}).get(str(step), {"status": "PENDING", "step": str(step)}))

    @_serialized_transition
    def run_pair(
        self, pulse_index: int, *,
        reference_step: Callable[[int], Any], candidate_step: Callable[[int], Any],
        compare_step: Callable[[int, Any, Any], Any],
        successor_step: Callable[[int, Any, Any], Any] | None = None,
        reclaim_step: Callable[[int, Any, Any], Any] | None = None,
    ) -> PairResult:
        """Run one serial R -> C -> exact pair with durable checkpoints."""
        if self.formal:
            raise StorageIntegrityError(
                "formal campaign forbids the legacy run_pair callback path; use FormalPairedDriver"
            )
        index = int(pulse_index)
        if index != self.next_pair_index:
            raise ValueError(f"pair {index} is not the next durable pair {self.next_pair_index}")
        pair = self._pair(index)
        if pair["reference"].get("status") not in ("PENDING", "PASS") or pair["candidate"].get("status") not in ("PENDING", "PASS"):
            raise ValueError("pair already started or failed")
        self.state["status"] = "RUNNING"
        self._persist()
        reference_value = None
        candidate_value = None
        if pair["reference"].get("status") == "PENDING":
            try:
                reference_value = reference_step(index)
            except Exception as error:
                self._record_step(pair, "reference", {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}, default="FAIL")
                self.state["status"] = "BLOCKED"
                self._persist()
                return self.result(index)
            self._record_step(pair, "reference", reference_value)
        if pair["reference"].get("status") != "PASS":
            self.state["status"] = "BLOCKED"
            self._persist()
            return self.result(index)
        if pair["candidate"].get("status") == "PENDING":
            try:
                candidate_value = candidate_step(index)
            except Exception as error:
                self._record_step(pair, "candidate", {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}, default="FAIL")
                self.state["status"] = "BLOCKED"
                self._persist()
                return self.result(index)
            self._record_step(pair, "candidate", candidate_value)
        if pair["reference"].get("status") != "PASS" or pair["candidate"].get("status") != "PASS":
            self.state["status"] = "BLOCKED"
            self._persist()
            return self.result(index)
        # On a process resume a completed side may only be represented by its
        # durable report; callbacks are not rerun.  The compare callback is
        # therefore passed both callback values when available, otherwise the
        # persisted side payloads.
        reference_value = reference_value if reference_value is not None else dict(pair["reference"])
        candidate_value = candidate_value if candidate_value is not None else dict(pair["candidate"])
        if pair["compare"].get("status") == "PENDING":
            try:
                compared = compare_step(index, reference_value, candidate_value)
            except Exception as error:
                self._record_step(pair, "compare", {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}, default="FAIL")
                self.state["status"] = "BLOCKED"
                self._persist()
                return self.result(index)
            self._record_step(pair, "compare", compared)
        if pair["compare"].get("status") != "PASS":
            self.state["status"] = "BLOCKED"
            self._persist()
            return self.result(index)
        successor_value = None
        if successor_step is not None and pair["successor"].get("status") == "PENDING":
            try:
                successor_value = successor_step(index, reference_value, candidate_value)
            except Exception as error:
                self._record_step(pair, "successor", {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}, default="FAIL")
                self.state["status"] = "BLOCKED"
                self._persist()
                return self.result(index)
            self._record_step(pair, "successor", successor_value)
        elif pair["successor"].get("status") == "PENDING":
            self._record_step(pair, "successor", {"status": "PENDING", "reason": "successor_step_required"}, default="PENDING")
        if pair["successor"].get("status") != "PASS":
            self.state["status"] = "BLOCKED"
            self._persist()
            return self.result(index)
        reclaim_value = None
        if reclaim_step is not None and pair["reclaim"].get("status") == "PENDING":
            try:
                reclaim_value = reclaim_step(index, reference_value, candidate_value)
            except Exception as error:
                self._record_step(pair, "reclaim", {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}, default="FAIL")
                self.state["status"] = "BLOCKED"
                self._persist()
                return self.result(index)
            self._record_step(pair, "reclaim", reclaim_value)
        elif pair["reclaim"].get("status") == "PENDING":
            self._record_step(pair, "reclaim", {"status": "PENDING", "reason": "reclaim_step_required"}, default="PENDING")
        if pair["reclaim"].get("status") != "PASS":
            self.state["status"] = "BLOCKED"
            self._persist()
            return self.result(index)
        pair["status"] = "PASS"
        pair["events"].append({"event": "PAIR_READY_FOR_NEXT", "timestamp_utc": _utc()})
        self.state["next_pair_index"] = index + 1
        self.state["status"] = "COMPLETE" if self.state["next_pair_index"] == self.n_pulses else "READY"
        self._persist()
        return self.result(index)

    def result(self, pulse_index: int) -> PairResult:
        pair = self._pair(pulse_index)
        statuses = {name: str(pair[name].get("status", "PENDING")) for name in ("reference", "candidate", "compare", "successor", "reclaim")}
        status = "PASS" if all(value == "PASS" for value in statuses.values()) else ("FAIL" if any(value == "FAIL" for value in statuses.values()) else "PENDING")
        return PairResult(int(pulse_index), status, statuses["reference"], statuses["candidate"], statuses["compare"], statuses["successor"], statuses["reclaim"])

    def run(
        self, *, reference_step: Callable[[int], Any], candidate_step: Callable[[int], Any],
        compare_step: Callable[[int, Any, Any], Any],
        successor_step: Callable[[int, Any, Any], Any] | None = None,
        reclaim_step: Callable[[int, Any, Any], Any] | None = None,
    ) -> dict[str, Any]:
        """Continue from the durable next pair until completion or failure."""
        if self.formal:
            raise StorageIntegrityError(
                "formal campaign forbids the legacy run callback path; use FormalPairedDriver"
            )
        while self.next_pair_index < self.n_pulses:
            result = self.run_pair(
                self.next_pair_index, reference_step=reference_step,
                candidate_step=candidate_step, compare_step=compare_step,
                successor_step=successor_step, reclaim_step=reclaim_step,
            )
            if result.status != "PASS":
                break
        return self.report()

    def report(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "status": self.state["status"],
            "campaign_id": self.campaign_id,
            "n_pulses": self.n_pulses,
            "next_pair_index": self.next_pair_index,
            "pairs": self.state["pairs"],
            "process_epochs": self.state["process_epochs"],
            "state_path": str(self.state_path),
            "state_sha256": sha256_file(self.state_path),
            "formal": self.formal,
            "admission_hash": self.state.get("admission_hash"),
            "active_coordinator": self.state.get("active_coordinator"),
            "storage": self.storage.budget_report(),
            "updated_utc": self.state.get("updated_utc"),
        }

    @_serialized_transition
    def commit_formal_pair(self, pulse_index: int, *, epoch: str | int) -> dict[str, Any]:
        if not self._coordinator_state_enabled():
            raise StorageIntegrityError("formal pair commit requires a formal campaign")
        if epoch is None:
            raise StorageIntegrityError("formal pair commit requires the current coordinator epoch")
        active_epoch = self._active_coordinator_epoch(epoch)
        index = int(pulse_index)
        if index != self.next_pair_index:
            raise StorageIntegrityError("formal pair index is stale or already committed")
        pair = self._pair(index)
        steps = pair.get("formal_steps", {})
        required = ["reference", "candidate", "exact"]
        if index < self.n_pulses - 1:
            required.extend(["reference_successor", "reference_gc", "candidate_successor", "candidate_gc"])
        else:
            required.append("terminal")
        for name in required:
            saved = steps.get(name, {})
            if str(saved.get("status", "")).upper() not in _FORMAL_RECEIPT_STATUSES:
                raise StorageIntegrityError("formal pair cannot advance before all durable steps commit")
            self._validate_saved_formal_receipt(saved, label=f"formal {name}")
            if self.formal and name.endswith("_gc"):
                saved_receipt = saved.get("receipt")
                if not isinstance(saved_receipt, Mapping):
                    raise StorageIntegrityError(f"formal {name} durable receipt payload is missing")
                self._verify_formal_gc_receipt(name, index, saved_receipt)
        pair["status"] = "PASS"
        pair["events"].append({"event": "FORMAL_PAIR_COMMITTED", "pulse_index": index, "epoch": active_epoch, "timestamp_utc": _utc()})
        self.state["next_pair_index"] = index + 1
        self.state["status"] = "COMPLETE" if index + 1 == self.n_pulses else "READY"
        self._persist()
        return dict(pair)

    @_serialized_transition
    def pause(self, *, reason: str = "operator_pause") -> dict[str, Any]:
        self.state["status"] = "PAUSED"
        self.state["pause_reason"] = str(reason)
        self._persist()
        return self.report()

    @_serialized_transition
    def resume(self) -> dict[str, Any]:
        if self.state.get("status") == "PAUSED":
            self.state["status"] = "READY" if self.next_pair_index < self.n_pulses else "COMPLETE"
            self._persist()
        return self.report()


class FormalPairedDriver:
    """Controlled durable R/C coordinator for the formal outer entry.

    A formal runner is an object with named methods (``run_reference``,
    ``run_candidate``, ``run_exact``, ``run_successor``, ``run_gc`` and
    ``run_terminal``).  Every method must return a durable report path with a
    validated object-set binding; a mapping that merely says ``PASS`` is
    rejected.  ``fixture_only=True`` is an explicit test-only escape hatch and
    writes a TEST_FIXTURE_ONLY receipt through this same state machine.
    """

    def __init__(
        self, root: str | Path, *, admission_identity: Mapping[str, Any] | None,
        runner: Any, n_pulses: int = 3, campaign_id: str | None = None,
        max_campaign_live_bytes: int = HARD_CAP_BYTES,
         final_output_budget_bytes: int = 64 * 1024**3,
         safety_margin_bytes: int = 8 * 1024**3, require_quota: bool = True,
         fixture_only: bool = False,
         takeover: bool = False,
         takeover_receipt: str | Path | Mapping[str, Any] | None = None,
         _production_capability: object | None = None,
    ):
        self.root = Path(root).resolve()
        self.runner = runner
        self.fixture_only = bool(fixture_only)
        if not isinstance(runner, object):  # pragma: no cover - documents the trust boundary
            raise TypeError("formal runner object is required")
        if self.fixture_only:
            from .hr4e5_formal_entry import validate_admission_identity
            if admission_identity is None:
                raise ValueError("fixture coordinator still requires explicit TEST_FIXTURE_ONLY identity")
            identity = validate_admission_identity(admission_identity, formal=False)
            if identity.get("execution_mode") != "TEST_FIXTURE_ONLY":
                raise ValueError("fixture coordinator requires TEST_FIXTURE_ONLY identity")
        else:
            if _production_capability is not _PRODUCTION_DRIVER_CAPABILITY:
                raise StorageIntegrityError("formal driver can only be created by the production factory")
            from .hr4e5_formal_entry import validate_admission_identity
            identity = validate_admission_identity(admission_identity or {}, formal=True)
            if not require_quota:
                raise ValueError("formal coordinator cannot disable quota reporting")
            if takeover_receipt is not None:
                raise StorageIntegrityError("external crash takeover receipts are forbidden")
        self.admission_identity = identity
        cid = str(campaign_id or identity.get("campaign_id"))
        self.campaign = PairedCampaign(
            self.root, create=not (self.root / CAMPAIGN_STATE_FILENAME).exists(),
            n_pulses=n_pulses, campaign_id=cid,
            max_campaign_live_bytes=max_campaign_live_bytes,
            final_output_budget_bytes=final_output_budget_bytes,
            safety_margin_bytes=safety_margin_bytes, require_quota=require_quota,
            admission_identity=identity, formal=not self.fixture_only,
            require_intents=not self.fixture_only,
        )
        if self.fixture_only and getattr(runner, "budget", None) is not None:
            self.campaign.storage.provider = runner.budget.provider
        self.epoch = f"{os.getpid()}:{uuid.uuid4().hex}"
        started = self.campaign.register_process_start(
            process_id=self.epoch, source="formal_driver",
            takeover=bool(takeover), takeover_receipt=None,
        )
        self.coordinator_epoch = started["epoch"]
        self._closed = False

    def _step_state(self, pulse: int, step: str) -> dict[str, Any]:
        return self.campaign.formal_step_state(pulse, step)

    def _receipt_path(self, value: Mapping[str, Any]) -> str | None:
        for key in ("report_path", "receipt_path", "evidence_path", "path"):
            raw = value.get(key)
            if raw:
                return str(raw)
        return None

    def _validate_formal_gates(
        self, step: str, value: Mapping[str, Any], validated: Mapping[str, Any], *,
        pulse: int, check_duplicate_successor: bool,
    ) -> dict[str, Any]:
        """Revalidate successor, writer, and terminal gates on every read."""
        extras: dict[str, Any] = {}
        if "successor" in step:
            child_root = value.get("child_root")
            if not child_root:
                raise StorageIntegrityError(f"formal {step} requires a child_root")
            child_path = Path(str(child_root))
            if not child_path.is_absolute():
                child_path = self.root / child_path
            child_path = child_path.resolve()
            if not child_path.is_relative_to(self.root):
                raise StorageIntegrityError(f"formal {step} child_root escapes campaign root")
            expected_parent = value.get("parent_root") or validated.get("parent_root")
            expected_parent_generation = value.get("parent_generation") or validated.get("parent_generation")
            if expected_parent is None or expected_parent_generation is None:
                raise StorageIntegrityError(f"formal {step} requires parent root and generation")
            expected_parent_path = Path(str(expected_parent))
            if not expected_parent_path.is_absolute():
                expected_parent_path = self.root / expected_parent_path
            expected_parent = str(expected_parent_path.resolve())
            try:
                ready = validate_ready_receipt(
                    child_path, expected_parent_root=expected_parent,
                    expected_parent_generation=str(expected_parent_generation),
                )
            except Exception as error:
                raise StorageIntegrityError(f"formal {step} READY receipt is missing or invalid") from error
            if Path(str(ready.get("child_root", ""))).resolve() != child_path:
                raise StorageIntegrityError(f"formal {step} READY child_root changed")
            if str(ready.get("admission_identity_sha256", "")) != str(self.admission_identity.get("identity_sha256", "")):
                raise StorageIntegrityError(f"formal {step} READY admission identity changed")
            if check_duplicate_successor:
                normalized = str(child_path)
                seen_roots: set[str] = set()
                for pair in self.campaign.state.get("pairs", []):
                    for saved_step in pair.get("formal_steps", {}).values():
                        saved_receipt = saved_step.get("receipt", {}) if isinstance(saved_step, Mapping) else {}
                        saved_child = saved_receipt.get("child_root") if isinstance(saved_receipt, Mapping) else None
                        if saved_child:
                            prior = Path(str(saved_child))
                            if not prior.is_absolute():
                                prior = self.root / prior
                            seen_roots.add(str(prior.resolve()))
                if normalized in seen_roots:
                    raise StorageIntegrityError(f"formal successor fork or duplicate root: {normalized}")
            extras.update({
                "ready_validation": {"status": "READY", "child_root": str(child_path)},
                "child_root": str(child_path),
            })
        if step.endswith("_gc") or step == "terminal":
            writer_path_value = value.get("writer_receipt_path") or value.get("writer_receipt") or validated.get("writer_receipt_path")
            writer_hash = value.get("writer_receipt_sha256") or validated.get("writer_receipt_sha256")
            if isinstance(writer_path_value, Mapping):
                writer_hash = writer_hash or writer_path_value.get("sha256")
                writer_path_value = writer_path_value.get("path") or writer_path_value.get("receipt_path")
            if not writer_path_value or not writer_hash:
                raise StorageIntegrityError(f"formal {step} lacks durable writer receipt path/hash")
            writer_path = Path(str(writer_path_value))
            if not writer_path.is_absolute():
                writer_path = self.root / writer_path
            writer_path = writer_path.resolve()
            if not writer_path.is_relative_to(self.root) or not writer_path.is_file():
                raise StorageIntegrityError(f"formal {step} writer receipt is missing or outside campaign")
            try:
                trajectory = "R" if step.startswith("reference") else "C"
                self.campaign.storage._read_writer_receipt(
                    {"path": writer_path, "sha256": writer_hash}, expected_sha256=str(writer_hash),
                    expected_trajectory=trajectory if step.endswith("_gc") else None,
                    expected_pulse=int(pulse) if step.endswith("_gc") else None,
                    expected_attempt=0 if step.endswith("_gc") else None,
                    expected_admission_hash=(
                        str(self.admission_identity.get("identity_sha256", ""))
                        if step.endswith("_gc") else None
                    ),
                )
            except Exception as error:
                raise StorageIntegrityError(f"formal {step} writer receipt is invalid") from error
        if step.endswith("_gc"):
            plan_id = validated.get("gc_plan_id")
            if not plan_id:
                raise StorageIntegrityError(f"formal {step} report requires a durable GC plan id")
            trajectory = "R" if step.startswith("reference") else "C"
            verified_gc = self.campaign.storage.verify_reclaim(
                str(plan_id), expected_trajectory=trajectory,
                expected_pulse=int(pulse), expected_attempt=0,
                expected_admission_hash=str(self.admission_identity.get("identity_sha256", "")),
            )
            if verified_gc.get("status") != "PASS":
                raise StorageIntegrityError(f"formal {step} GC plan is not durably complete")
            extras = {**extras, "gc_verification": verified_gc, "gc_plan_id": str(plan_id)}
        if step == "terminal":
            if "expected_roles" in value:
                raise StorageIntegrityError("formal terminal roles are generated by the production contract")
            role_builder = getattr(self.runner, "expected_terminal_roles", None)
            if not callable(role_builder):
                raise StorageIntegrityError("production runner lacks terminal role contract")
            expected_roles = role_builder(int(pulse))
            if not expected_roles or isinstance(expected_roles, (str, bytes)):
                raise StorageIntegrityError("formal terminal expected_roles inventory is empty or invalid")
            inventory_root = self.root
            supplied_root = value.get("terminal_inventory_root") or validated.get("terminal_inventory_root")
            if supplied_root is not None:
                candidate_root = Path(str(supplied_root))
                if not candidate_root.is_absolute():
                    candidate_root = self.root / candidate_root
                if candidate_root.resolve() != inventory_root:
                    raise StorageIntegrityError("formal terminal inventory must cover the complete campaign root")
            try:
                terminal_inventory = self.campaign.storage.validate_terminal_inventory(
                    expected_roles, root=inventory_root, final=True,
                )
            except Exception as error:
                raise StorageIntegrityError("formal terminal inventory is missing or invalid") from error
            extras.update({"terminal_inventory": terminal_inventory, "terminal_inventory_root": str(inventory_root)})
        return extras

    def _validate_receipt(self, value: Any, *, pulse: int, step: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or str(value.get("status", "")).upper() != "PASS":
            raise StorageIntegrityError(f"formal {step} did not return PASS")
        raw_path = self._receipt_path(value)
        if raw_path is None:
            if not self.fixture_only:
                raise StorageIntegrityError(f"formal {step} omitted a durable report path")
            receipt_dir = self.root / ".formal_driver_receipts"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / f"p{int(pulse)}_{step}.json"
            atomic_json(path, {**dict(value), "scope": "TEST_FIXTURE_ONLY", "campaign_id": self.campaign.campaign_id,
                               "pulse_index": int(pulse), "step": str(step), "status": "PASS"})
            raw_path = str(path)
        path = Path(raw_path).resolve()
        if not path.is_relative_to(self.root):
            raise StorageIntegrityError(f"formal {step} report escapes campaign root")
        if self.fixture_only:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS" or (payload.get("campaign_id") not in (None, self.campaign.campaign_id)):
                raise StorageIntegrityError(f"fixture {step} report is not durable PASS")
            digest = sha256_file(path)
            return {**dict(value), "report_path": str(path), "report_sha256": digest, "scope": "TEST_FIXTURE_ONLY"}
        expected_root = self.root
        if step == "exact":
            validated = validate_paired_exact_report(
                path, admission_identity=self.admission_identity or {},
                campaign_id=self.campaign.campaign_id, pulse=int(pulse),
                root=expected_root, storage_budget=self.campaign.storage,
                expected_sha256=sha256_file(path),
            )
            return {**dict(value), **validated, "report_path": str(path),
                    "report_sha256": validated["report_sha256"]}
        trajectory = None
        if step.startswith("reference"):
            trajectory = "R"
        elif step.startswith("candidate") or step == "terminal":
            trajectory = "C"
        validated = validate_durable_report(path, campaign_id=self.campaign.campaign_id, root=expected_root,
                                             pulse=int(pulse), trajectory=trajectory)
        gates = self._validate_formal_gates(
            step, value, validated, pulse=int(pulse), check_duplicate_successor=True,
        )
        result = {**dict(value), **validated, "report_path": str(path),
                  "report_sha256": validated["report_sha256"]}
        result.update(gates)
        return result

    def _saved_receipt(self, pulse: int, step: str) -> dict[str, Any] | None:
        saved = self._step_state(pulse, step)
        if str(saved.get("status", "")).upper() not in {"COMMITTED", "PASS"}:
            return None
        saved_epoch = saved.get("epoch")
        if saved_epoch is not None:
            known_epochs = {
                token
                for item in self.campaign.state.get("process_epochs", [])
                for token in (str(item.get("epoch")), str(item.get("process_id")))
            }
            if str(saved_epoch) not in known_epochs:
                raise StorageIntegrityError(f"formal {step} receipt belongs to an unknown coordinator epoch")
        path = saved.get("receipt_path")
        digest = saved.get("receipt_sha256")
        if not path:
            raise StorageIntegrityError(f"formal {step} has no durable receipt")
        if self.fixture_only:
            file = Path(str(path)).resolve()
            if not file.is_file() or sha256_file(file) != str(digest):
                raise StorageIntegrityError(f"fixture {step} receipt changed")
            result = dict(saved.get("receipt", {}))
            if result.get("epoch") is not None and saved_epoch is not None and str(result.get("epoch")) != str(saved_epoch):
                raise StorageIntegrityError(f"fixture {step} receipt epoch changed")
            return result
        trajectory = "R" if step.startswith("reference") else ("C" if step.startswith("candidate") or step == "terminal" else None)
        validate_objects = True
        pair = self.campaign.state.get("pairs", [])[int(pulse)]
        formal_steps = pair.get("formal_steps", {}) if isinstance(pair, Mapping) else {}
        if step == "exact":
            after_gc = any(str(formal_steps.get(name, {}).get("status", "")).upper() in _FORMAL_RECEIPT_STATUSES
                           for name in ("reference_gc", "candidate_gc"))
            validator = validate_paired_exact_report_metadata if after_gc else validate_paired_exact_report
            validated = validator(
                path, admission_identity=self.admission_identity or {},
                campaign_id=self.campaign.campaign_id, pulse=int(pulse), attempt=0,
                root=self.root, storage_budget=self.campaign.storage,
                expected_sha256=str(digest),
            )
            if validated.get("epoch") is not None and saved_epoch is not None and str(validated.get("epoch")) != str(saved_epoch):
                raise StorageIntegrityError(f"formal {step} receipt epoch changed")
            return validated
        elif step.startswith("reference"):
            validate_objects = str(formal_steps.get("reference_gc", {}).get("status", "")).upper() not in _FORMAL_RECEIPT_STATUSES
        elif step.startswith("candidate"):
            validate_objects = str(formal_steps.get("candidate_gc", {}).get("status", "")).upper() not in _FORMAL_RECEIPT_STATUSES
        validated = validate_durable_report(path, expected_sha256=str(digest), campaign_id=self.campaign.campaign_id,
                                             root=self.root, pulse=int(pulse), trajectory=trajectory,
                                             validate_objects=validate_objects)
        saved_payload = saved.get("receipt", {})
        if not isinstance(saved_payload, Mapping):
            raise StorageIntegrityError(f"formal {step} durable receipt payload is missing")
        gates = self._validate_formal_gates(
            step, saved_payload, validated, pulse=int(pulse), check_duplicate_successor=False,
        )
        if validated.get("epoch") is not None and saved_epoch is not None and str(validated.get("epoch")) != str(saved_epoch):
            raise StorageIntegrityError(f"formal {step} receipt epoch changed")
        return {**validated, **gates}

    def _runner_call(self, step: str, pulse: int, *, track: str | None = None) -> Any:
        if self.fixture_only:
            names = {
                "reference": ("reference", "run_reference"), "candidate": ("candidate", "run_candidate"),
                "exact": ("compare", "run_exact"), "reference_successor": ("handoff_reference", "handoff"),
                "candidate_successor": ("handoff_candidate", "handoff"),
                "reference_gc": ("reclaim_reference", "reclaim"), "candidate_gc": ("reclaim_candidate", "reclaim"),
                "terminal": ("terminal", "run_terminal"),
            }
        else:
            names = {
                "reference": ("run_reference",), "candidate": ("run_candidate",), "exact": ("run_exact", "compare"),
                "reference_successor": ("run_successor",), "candidate_successor": ("run_successor",),
                "reference_gc": ("run_gc",), "candidate_gc": ("run_gc",), "terminal": ("run_terminal", "finalize"),
            }
        method = next((getattr(self.runner, name, None) for name in names[step] if callable(getattr(self.runner, name, None))), None)
        if method is None:
            raise StorageIntegrityError(f"formal runner lacks controlled method for {step}")
        if step in {"reference_successor", "candidate_successor", "reference_gc", "candidate_gc"}:
            if self.fixture_only and step in {"reference_gc", "candidate_gc"}:
                return method(track or ("R" if step.startswith("reference") else "C"), pulse, None)
            try:
                return method(pulse, track or ("R" if step.startswith("reference") else "C"))
            except TypeError:
                return method(track or ("R" if step.startswith("reference") else "C"), pulse)
        return method(pulse)

    def _run_step(self, pulse: int, step: str, *, track: str | None = None) -> dict[str, Any]:
        saved = self._saved_receipt(pulse, step)
        if saved is not None:
            return saved
        coordinator_epoch = self.campaign.active_coordinator_epoch()
        self.campaign.formal_step(pulse, step, status="IN_PROGRESS", epoch=coordinator_epoch)
        try:
            value = self._runner_call(step, pulse, track=track)
            receipt = self._validate_receipt(value, pulse=pulse, step=step)
            self.campaign.formal_step(
                pulse, step, status="COMMITTED", receipt=receipt,
                epoch=self.campaign.active_coordinator_epoch(),
            )
            return receipt
        except Exception as error:
            try:
                self.campaign.formal_step(
                    pulse, step, status="INTERRUPTED", error=f"{type(error).__name__}: {error}",
                    epoch=self.campaign.active_coordinator_epoch(),
                )
            except Exception:
                pass
            raise

    def run(self, *, stop_after: int | None = None) -> dict[str, Any]:
        try:
            while self.campaign.next_pair_index < self.campaign.n_pulses:
                pulse = self.campaign.next_pair_index
                self._run_step(pulse, "reference", track="R")
                self._run_step(pulse, "candidate", track="C")
                self._run_step(pulse, "exact")
                if pulse == self.campaign.n_pulses - 1:
                    self._run_step(pulse, "terminal", track="C")
                else:
                    self._run_step(pulse, "reference_successor", track="R")
                    self._run_step(pulse, "reference_gc", track="R")
                    self._run_step(pulse, "candidate_successor", track="C")
                    self._run_step(pulse, "candidate_gc", track="C")
                self.campaign.commit_formal_pair(
                    pulse, epoch=self.campaign.active_coordinator_epoch(),
                )
                if stop_after is not None and self.campaign.next_pair_index >= int(stop_after):
                    break
            return self.report()
        finally:
            self.close()

    def close(self) -> dict[str, Any]:
        if not self._closed:
            try:
                self.campaign.register_process_exit(process_id=self.epoch)
            except ValueError:
                pass
            self._closed = True
        return self.report()

    def pause(self, *, reason: str = "operator_pause") -> dict[str, Any]:
        result = self.campaign.pause(reason=reason)
        self.close()
        return result

    def resume(self, *, run: bool = True, stop_after: int | None = None,
               takeover: bool = False,
               takeover_receipt: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
        if takeover_receipt is not None:
            raise StorageIntegrityError("external crash takeover receipts are forbidden")
        self.campaign.resume()
        if self._closed:
            self.epoch = f"{os.getpid()}:{uuid.uuid4().hex}"
            started = self.campaign.register_process_start(
                process_id=self.epoch, source="formal_resume",
                takeover=bool(takeover), takeover_receipt=None,
            )
            self.coordinator_epoch = started["epoch"]
            self._closed = False
        return self.run(stop_after=stop_after) if run else self.report()

    def report(self) -> dict[str, Any]:
        return {**self.campaign.report(), "formal_driver": True, "fixture_only": self.fixture_only, "epoch": self.epoch}


def _open_production_driver(*args: Any, **kwargs: Any) -> FormalPairedDriver:
    """Private capability-bearing construction point used by the production factory."""
    if kwargs.get("fixture_only"):
        raise StorageIntegrityError("production factory cannot create a fixture driver")
    return FormalPairedDriver(*args, **kwargs, _production_capability=_PRODUCTION_DRIVER_CAPABILITY)


def compare_named_arrays(reference: Mapping[str, Any], candidate: Mapping[str, Any], *, layer: str = "pair") -> dict[str, Any]:
    """Convenience wrapper used by pair callbacks and focused tests."""
    return compare_object_sets(reference, candidate, layer=layer)


def successor_from_lifecycle(*, parent_root: str | Path, child_root: str | Path,
                             admission_identity: Mapping[str, Any] | None = None,
                             storage_budget: StorageBudget | None = None,
                             creation_intent: Mapping[str, Any] | str | None = None,
                             fixture_only: bool = False) -> dict[str, Any]:
    child, receipt = create_successor_root(
        parent_root=parent_root, child_root=child_root,
        admission_identity=admission_identity, storage_budget=storage_budget,
        creation_intent=creation_intent, fixture_only=fixture_only,
    )
    validate_ready_receipt(child.root, expected_parent_root=parent_root)
    return {"status": "PASS", "child_root": str(child.root), "ready": receipt}


__all__ = [
    "ADMISSION_FILENAME", "CAMPAIGN_SCHEMA", "CAMPAIGN_STATE_FILENAME", "HARD_CAP_BYTES", "PairResult",
    "FormalPairedDriver", "PairedCampaign", "compare_named_arrays", "successor_from_lifecycle",
]

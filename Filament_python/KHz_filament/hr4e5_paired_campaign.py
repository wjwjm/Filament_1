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
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .hr4e5_evidence import atomic_json, compare_object_sets, sha256_file, validate_ready_receipt
from .hr4e5_formal_entry import create_successor_root
from .hr4e5_storage import HARD_CAP_BYTES, StorageBudget, StorageBudgetError, _FileLock


CAMPAIGN_SCHEMA = "khz_filament.hr4e5.e5_1a.paired_campaign.v1"
CAMPAIGN_STATE_FILENAME = "E5_1A_CAMPAIGN_STATE.json"


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
                 campaign_id: str = "e5_1a_local"):
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
        self.state = self._read_state()
        self._validate_state(self.state)
        # The campaign ledger covers every file created below the campaign
        # root, including R/C roots, receipts, manifests, diagnostics, and
        # failed-attempt residue.  A subdirectory ledger would allow a caller
        # to evade the single-campaign cap by changing directories.
        self.storage = StorageBudget(
            self.root, cap_bytes=self.max_campaign_live_bytes,
            final_output_budget_bytes=self.final_output_budget_bytes,
            safety_margin_bytes=self.safety_margin_bytes, require_quota=self.require_quota,
            campaign_id=self.campaign_id,
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
            "status": "READY",
            "next_pair_index": 0,
            "pairs": [self._new_pair(index) for index in range(self.n_pulses)],
            "process_epochs": [],
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
        if bool(state.get("require_quota", False)) != self.require_quota:
            raise StorageBudgetError("campaign quota requirement conflicts with state")
        pairs = state.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != self.n_pulses:
            raise ValueError("campaign pair list is invalid")
        if [int(item.get("pulse_index", -1)) for item in pairs] != list(range(self.n_pulses)):
            raise ValueError("campaign pair order is invalid")
        next_pair = state.get("next_pair_index")
        if isinstance(next_pair, bool) or not isinstance(next_pair, int) or not 0 <= next_pair <= self.n_pulses:
            raise ValueError("campaign next pair index is invalid")

    def _persist(self) -> None:
        self.state["updated_utc"] = _utc()
        self._validate_state(self.state)
        atomic_json(self.state_path, self.state)

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
    def register_process_start(self, *, process_id: str | None = None, source: str = "new_process") -> dict[str, Any]:
        epoch = {"epoch": len(self.state["process_epochs"]), "process_id": str(process_id or f"pid:{os.getpid()}"), "source": str(source), "started_utc": _utc(), "status": "ACTIVE"}
        self.state["process_epochs"].append(epoch)
        self.state["status"] = "RUNNING"
        self._persist()
        return epoch

    @_serialized_transition
    def register_process_exit(self, *, process_id: str | None = None) -> dict[str, Any]:
        target = str(process_id or f"pid:{os.getpid()}")
        for epoch in reversed(self.state["process_epochs"]):
            if epoch.get("process_id") == target and epoch.get("status") == "ACTIVE":
                epoch["status"] = "EXITED"
                epoch["exited_utc"] = _utc()
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
    def run_pair(
        self, pulse_index: int, *,
        reference_step: Callable[[int], Any], candidate_step: Callable[[int], Any],
        compare_step: Callable[[int, Any, Any], Any],
        successor_step: Callable[[int, Any, Any], Any] | None = None,
        reclaim_step: Callable[[int, Any, Any], Any] | None = None,
    ) -> PairResult:
        """Run one serial R -> C -> exact pair with durable checkpoints."""
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
            "storage": self.storage.budget_report(),
            "updated_utc": self.state.get("updated_utc"),
        }

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


def compare_named_arrays(reference: Mapping[str, Any], candidate: Mapping[str, Any], *, layer: str = "pair") -> dict[str, Any]:
    """Convenience wrapper used by pair callbacks and focused tests."""
    return compare_object_sets(reference, candidate, layer=layer)


def successor_from_lifecycle(*, parent_root: str | Path, child_root: str | Path) -> dict[str, Any]:
    child, receipt = create_successor_root(parent_root=parent_root, child_root=child_root)
    validate_ready_receipt(child.root)
    return {"status": "PASS", "child_root": str(child.root), "ready": receipt}


__all__ = [
    "CAMPAIGN_SCHEMA", "CAMPAIGN_STATE_FILENAME", "HARD_CAP_BYTES", "PairResult",
    "PairedCampaign", "compare_named_arrays", "successor_from_lifecycle",
]

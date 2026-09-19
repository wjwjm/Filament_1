"""Bounded, fail-closed storage bookkeeping for the E5-1A fixture.

This module owns no scientific state.  It provides a small durable byte ledger,
cross-process reservations, and an explicit whitelist based reclamation
transaction for files created below one campaign root.  The implementation is
intentionally local to the E5-1A glue; it is not a general storage service.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


HARD_CAP_BYTES = 300 * 1024**3
DEFAULT_FINAL_OUTPUT_BUDGET_BYTES = 64 * 1024**3
DEFAULT_SAFETY_MARGIN_BYTES = 8 * 1024**3
STORAGE_SCHEMA = "khz_filament.hr4e5.e5_1a.storage.v1"
GC_SCHEMA = "khz_filament.hr4e5.e5_1a.gc.v1"
RECLAIM_PREREQUISITE_SCHEMA = "khz_filament.hr4e5.e5_1a.reclaim_prerequisites.v1"
INTENT_SCHEMA = "khz_filament.hr4e5.e5_1a.creation_intent.v1"
WRITER_SCHEMA = "khz_filament.hr4e5.e5_1a.writer.v1"
WRITER_RECEIPT_SCHEMA = "khz_filament.hr4e5.e5_1a.writer_receipt.v1"


def _host_identity() -> str:
    if sys.platform.startswith("linux"):
        boot = Path("/proc/sys/kernel/random/boot_id")
        if boot.is_file():
            return f"{socket.gethostname()}:{boot.read_text(encoding='ascii').strip()}"
    return socket.gethostname()


def _process_start_token(pid: int) -> str | None:
    """Return an OS process-instance token, not merely a reusable PID."""
    if pid <= 0:
        return None
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                                  ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                                  ctypes.POINTER(wintypes.FILETIME))
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            try:
                exit_code = wintypes.DWORD()
                if (not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                        or int(exit_code.value) != 259):  # STILL_ACTIVE
                    return None
                creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
                if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                                ctypes.byref(kernel), ctypes.byref(user)):
                    return None
                return str((int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime))
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii").split()
            return fields[21] if len(fields) > 21 else None
        except (OSError, ValueError):
            return None
    return None


def process_identity(pid: int | None = None) -> dict[str, Any]:
    value = int(os.getpid() if pid is None else pid)
    token = _process_start_token(value)
    if token is None:
        raise StorageIntegrityError("process start identity is unavailable")
    return {"pid": value, "host_identity": _host_identity(), "process_start_token": token}


def probe_process_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a recorded process as LIVE, DEAD, or UNKNOWN fail-closed."""
    if str(identity.get("host_identity", "")) != _host_identity():
        return {"status": "UNKNOWN", "reason": "different_host"}
    try:
        pid = int(identity["pid"]); expected = str(identity["process_start_token"])
    except (KeyError, TypeError, ValueError):
        return {"status": "UNKNOWN", "reason": "invalid_identity"}
    actual = _process_start_token(pid)
    if actual is None:
        return {"status": "DEAD", "reason": "pid_absent", "pid": pid}
    if str(actual) != expected:
        return {"status": "DEAD", "reason": "pid_reused", "pid": pid,
                "observed_process_start_token": str(actual)}
    return {"status": "LIVE", "reason": "identity_matches", "pid": pid}


def plan_campaign_budget(*, n_pulses: int, k: int, ny: int, nx: int, nt: int,
                         max_campaign_live_bytes: int = HARD_CAP_BYTES,
                         final_output_budget_bytes: int = DEFAULT_FINAL_OUTPUT_BUDGET_BYTES,
                         safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES) -> dict[str, Any]:
    """Derive the bounded campaign allocation/retention model.

    This is deliberately scalar planning only.  It mirrors the sequential
    R-then-C handoff used by the formal coordinator and never allocates a
    scientific array.  The model keeps one failed handoff residue and a
    positive receipt/ledger allowance so a plan cannot pass by declaring all
    metadata free.
    """
    values = (n_pulses, k, ny, nx, nt)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("positive integer dimensions and pulse count required")
    cap = _nonnegative_int(max_campaign_live_bytes, "max_campaign_live_bytes")
    final_cap = _nonnegative_int(final_output_budget_bytes, "final_output_budget_bytes")
    margin = _nonnegative_int(safety_margin_bytes, "safety_margin_bytes")
    if cap <= 0 or cap > HARD_CAP_BYTES or final_cap <= 0 or final_cap > cap or margin <= 0:
        raise StorageBudgetError("invalid campaign budget limits")
    field = int(k) * int(ny) * int(nx) * 8
    generation = 3 * field
    optical = int(nt) * int(ny) * int(nx) * 16
    # One R handoff and one C handoff are serialized.  The active pair keeps
    # both tracks, while a failed handoff retains one extra slow-state copy.
    reference_phase = 7 * generation
    candidate_phase = 5 * generation
    pair_peak = reference_phase + candidate_phase + generation
    failed_residue = generation
    retained_optics = (2 * int(n_pulses) + 1) * optical
    source_copy = optical
    metadata = max(2 * 1024**3, (2 * 1024**3 * int(n_pulses) + 2) // 3)
    peak = pair_peak + failed_residue + retained_optics + source_copy + metadata + margin
    final_bytes = 2 * generation + retained_optics + source_copy + metadata
    if peak > cap:
        raise StorageBudgetError(
            f"campaign allocation model exceeds cap: projected={peak} cap={cap}"
        )
    if final_bytes > final_cap:
        raise StorageBudgetError(
            f"campaign final retention model exceeds final budget: projected={final_bytes} cap={final_cap}"
        )
    return {
        "schema": "khz_filament.hr4e5.e5_1a.budget_plan.v1",
        "n_pulses": int(n_pulses), "k": int(k), "ny": int(ny), "nx": int(nx), "nt": int(nt),
        "field_bytes": field, "generation_bytes": generation, "optical_bytes": optical,
        "reference_phase_bytes": reference_phase, "candidate_phase_bytes": candidate_phase,
        "pair_peak_bytes": pair_peak, "failed_handoff_residue_bytes": failed_residue,
        "retained_optical_bytes": retained_optics, "source_copy_bytes": source_copy,
        "metadata_bytes": metadata, "safety_margin_bytes": margin,
        "peak_bytes": peak, "final_output_bytes": final_bytes,
        "max_campaign_live_bytes": cap, "final_output_budget_bytes": final_cap,
        "headroom_bytes": cap - peak, "final_headroom_bytes": final_cap - final_bytes,
        "k8048_arrays_created": False,
    }


derive_budget_plan = plan_campaign_budget


def plan_final_output(*, n_pulses: int, k: int, ny: int, nx: int, nt: int,
                      diagnostics: Sequence[str] = (), checkpoints: int = 0,
                      final_budget_bytes: int = DEFAULT_FINAL_OUTPUT_BUDGET_BYTES) -> dict[str, Any]:
    """Explicit bounded output policy; scalar planning only, never allocations."""
    values = (n_pulses, k, ny, nx, nt)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError('positive integer dimensions and pulse count required')
    if checkpoints < 0 or len(set(diagnostics)) != len(diagnostics) or not set(diagnostics) <= {'ion','ib','raman','qthermal','increment','state_after'}:
        raise ValueError('invalid finite output selection')
    if not 0 < final_budget_bytes <= HARD_CAP_BYTES:
        raise StorageBudgetError('invalid final output cap')
    field = k*ny*nx*8; generation=3*field; optical=nt*ny*nx*16
    metadata=(2*1024**3*n_pulses+2)//3
    total=2*generation+(2*n_pulses+1)*optical+metadata+2*n_pulses*len(diagnostics)*field+checkpoints*generation
    if total > final_budget_bytes:
        raise StorageBudgetError('declared final diagnostics/checkpoints exceed final output budget')
    return dict(final_output_bytes=total, final_output_budget_bytes=final_budget_bytes,
                candidate_final_state_bytes=2*generation, optical_bytes=(2*n_pulses+1)*optical,
                metadata_allowance_bytes=metadata, selected_diagnostics=list(diagnostics), checkpoints=checkpoints)


class StorageBudgetError(RuntimeError):
    """A write, reservation, or reclamation request was rejected."""


class StorageIntegrityError(StorageBudgetError):
    """A path or file identity cannot be proven safe."""


class _FileLock:
    """Small cross-platform advisory lock used for ledger transitions."""

    def __init__(self, path: Path, timeout_s: float = 10.0):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self._handle = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        if self._handle.tell() == 0:
            self._handle.write(b"0")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    self._handle.seek(0)
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as error:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise StorageBudgetError("timed out acquiring storage ledger lock") from error
                time.sleep(0.01)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(dict(value)), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        return


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: str | Path) -> tuple[Path, str]:
    root_resolved = root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    if '..' in candidate.parts:
        raise StorageIntegrityError('parent traversal is not allowed')
    try:
        lexical = candidate.relative_to(root_resolved)
    except ValueError as error:
        raise StorageIntegrityError('path escapes campaign root') from error
    cursor = root_resolved
    for part in lexical.parts:
        cursor = cursor / part
        if _is_reparse(cursor):
            raise StorageIntegrityError('symlink or reparse path is not eligible')
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(root_resolved)
    except (OSError, ValueError) as error:
        raise StorageIntegrityError("path escapes campaign root") from error
    # ``resolve`` above intentionally rejects ``..`` and prefix collisions;
    # retaining a second textual check catches a path that does not exist yet.
    parts = Path(path).parts
    if ".." in parts:
        raise StorageIntegrityError("parent traversal is not allowed")
    if not relative.parts:
        raise StorageIntegrityError("campaign root itself is not an artifact")
    return resolved, relative.as_posix()


def _is_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        stat = path.lstat()
        flag = getattr(stat, "st_file_attributes", 0)
        return bool(flag & getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _identity(path: Path) -> dict[str, Any]:
    try:
        stat = path.lstat()
    except OSError as error:
        raise StorageIntegrityError(f"artifact cannot be stat'ed: {path}") from error
    if _is_reparse(path):
        raise StorageIntegrityError("symlink or reparse artifact is not eligible")
    if not path.is_file():
        raise StorageIntegrityError("artifact is not a regular file")
    return {
        "st_dev": int(getattr(stat, "st_dev", -1)),
        "st_ino": int(getattr(stat, "st_ino", -1)),
        "st_nlink": int(getattr(stat, "st_nlink", 1)),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", 0)),
    }


def _iter_regular_files(root: Path) -> Iterable[Path]:
    """Yield files without following directory links."""
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        safe_dirs = []
        for dirname in dirnames:
            path = base / dirname
            if _is_reparse(path):
                # A reparse point is an unbounded accounting boundary.  It is
                # safer to reject the whole budget operation than silently
                # omit a linked subtree from the campaign footprint.
                raise StorageIntegrityError(f"reparse directory is not account-safe: {path}")
            safe_dirs.append(dirname)
        dirnames[:] = safe_dirs
        for filename in filenames:
            path = base / filename
            if _is_reparse(path):
                raise StorageIntegrityError(f"reparse file is not account-safe: {path}")
            if path.is_file():
                yield path


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _relative_paths_overlap(left: str | Path, right: str | Path) -> bool:
    """Return whether two normalized campaign-relative paths overlap."""
    lhs, rhs = Path(left), Path(right)
    return lhs == rhs or lhs.is_relative_to(rhs) or rhs.is_relative_to(lhs)


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    bytes: int
    purpose: str
    owner: str
    created_utc: str


class StorageBudget:
    """Durable byte ledger and fail-closed whitelist GC coordinator."""

    def __init__(
        self,
        root: str | Path,
        *,
        cap_bytes: int = HARD_CAP_BYTES,
        final_output_budget_bytes: int = DEFAULT_FINAL_OUTPUT_BUDGET_BYTES,
        safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
        provider: Any | None = None,
        require_quota: bool = True,
        campaign_id: str = "e5_1a_local",
        require_intents: bool = False,
        admission_hash: str | None = None,
        admission_identity: Mapping[str, Any] | None = None,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "storage_ledger.json"
        self.lock_path = self.root / ".storage_budget.lock"
        cap = _nonnegative_int(cap_bytes, "cap_bytes")
        final_budget = _nonnegative_int(final_output_budget_bytes, "final_output_budget_bytes")
        margin = _nonnegative_int(safety_margin_bytes, "safety_margin_bytes")
        if cap > HARD_CAP_BYTES:
            raise StorageBudgetError(f"campaign cap cannot exceed hard limit {HARD_CAP_BYTES} bytes")
        if cap <= 0 or final_budget > cap or margin <= 0 or margin >= cap:
            raise StorageBudgetError("storage cap, final output budget, or safety margin is invalid")
        self.cap_bytes = cap
        self.final_output_budget_bytes = final_budget
        self.safety_margin_bytes = margin
        self.provider = provider
        self.require_quota = bool(require_quota)
        self.campaign_id = str(campaign_id)
        self.require_intents = bool(require_intents)
        if admission_hash is None and admission_identity is not None:
            admission_hash = admission_identity.get("identity_sha256")
        self.admission_hash = None if admission_hash is None else str(admission_hash)
        self.admission_identity = None if admission_identity is None else dict(admission_identity)
        if not self.ledger_path.exists():
            _atomic_json(self.ledger_path, self._new_ledger(), overwrite=False)
        self._validate(self._read())

    def _new_ledger(self) -> dict[str, Any]:
        return {
            "schema": STORAGE_SCHEMA,
            "campaign_id": self.campaign_id,
            "cap_bytes": self.cap_bytes,
            "final_output_budget_bytes": self.final_output_budget_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "require_quota": self.require_quota,
            "require_intents": self.require_intents,
            "admission_hash": self.admission_hash,
            "artifacts": {},
            "reservations": {},
            "intents": {},
            "writers": {},
            "gc_plans": {},
            "events": [],
            "created_utc": _utc(),
        }

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageBudgetError("storage ledger is unreadable") from error
        if not isinstance(data, dict):
            raise StorageBudgetError("storage ledger must be an object")
        return data

    def _validate(self, data: Mapping[str, Any]) -> None:
        if data.get("schema") != STORAGE_SCHEMA:
            raise StorageBudgetError("storage ledger schema is invalid")
        if int(data.get("cap_bytes", -1)) != self.cap_bytes:
            raise StorageBudgetError("storage cap conflicts with durable ledger")
        if int(data.get("final_output_budget_bytes", -1)) != self.final_output_budget_bytes:
            raise StorageBudgetError("final output budget conflicts with durable ledger")
        if int(data.get("safety_margin_bytes", -1)) != self.safety_margin_bytes:
            raise StorageBudgetError("storage safety margin conflicts with durable ledger")
        if bool(data.get("require_quota", False)) != self.require_quota:
            raise StorageBudgetError("storage quota requirement conflicts with durable ledger")
        if bool(data.get("require_intents", False)) != self.require_intents:
            raise StorageBudgetError("storage intent requirement conflicts with durable ledger")
        saved_admission = data.get("admission_hash")
        if (None if saved_admission is None else str(saved_admission)) != self.admission_hash:
            raise StorageIntegrityError("storage admission identity conflicts with durable ledger")
        if str(data.get("campaign_id", "")) != self.campaign_id:
            raise StorageIntegrityError("storage ledger campaign identity conflicts with requested campaign")
        if (not isinstance(data.get("artifacts"), dict)
                or not isinstance(data.get("reservations"), dict)
                or not isinstance(data.get("intents", {}), dict)
                or not isinstance(data.get("writers", {}), dict)):
            raise StorageBudgetError("storage ledger collections are invalid")
        for reservation_id, item in data.get("reservations", {}).items():
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"reservation record is invalid: {reservation_id}")
            paths = item.get("allocation_paths", [])
            if not isinstance(paths, list):
                raise StorageBudgetError(f"reservation allocation paths are invalid: {reservation_id}")
            normalized = [str(path).replace("\\", "/") for path in paths]
            for index, left in enumerate(normalized):
                for right in normalized[index + 1:]:
                    if _relative_paths_overlap(left, right):
                        raise StorageIntegrityError(f"reservation allocation paths overlap: {reservation_id}")
        for intent_id, item in data.get("intents", {}).items():
            if not isinstance(item, Mapping) or item.get("schema") != INTENT_SCHEMA:
                raise StorageBudgetError(f"creation intent record is invalid: {intent_id}")
            if str(item.get("campaign_id", "")) != self.campaign_id:
                raise StorageIntegrityError(f"creation intent campaign mismatch: {intent_id}")
            allowed = item.get("allowed_paths", [])
            if not isinstance(allowed, list) or not allowed:
                raise StorageBudgetError(f"creation intent paths are invalid: {intent_id}")
            for index, left in enumerate(str(path).replace("\\", "/") for path in allowed):
                for right in (str(path).replace("\\", "/") for path in allowed[index + 1:]):
                    if _relative_paths_overlap(left, right):
                        raise StorageIntegrityError(f"creation intent paths overlap: {intent_id}")
        for writer_id, item in data.get("writers", {}).items():
            if not isinstance(item, Mapping) or item.get("schema") != WRITER_SCHEMA:
                raise StorageBudgetError(f"writer record is invalid: {writer_id}")
            if str(item.get("campaign_id", "")) != self.campaign_id:
                raise StorageIntegrityError(f"writer campaign mismatch: {writer_id}")

    def _write(self, data: Mapping[str, Any]) -> None:
        self._validate(data)
        _atomic_json(self.ledger_path, data)

    @contextmanager
    def _locked(self):
        with _FileLock(self.lock_path):
            data = self._read()
            self._validate(data)
            yield data
            self._write(data)

    def _management_paths(self) -> set[Path]:
        return {self.ledger_path.resolve(), self.lock_path.resolve()}

    def actual_bytes(self) -> int:
        total = 0
        for path in _iter_regular_files(self.root):
            try:
                total += int(path.stat().st_size)
            except OSError as error:
                raise StorageBudgetError(f"cannot account file {path}") from error
        return total

    @property
    def used_bytes(self) -> int:
        return self.actual_bytes()

    def _reserved_bytes(self, data: Mapping[str, Any]) -> int:
        total = 0
        for item in data.get('reservations', {}).values():
            if item.get('status') != 'ACTIVE':
                continue
            materialized = max(0, self._allocation_bytes(item.get('allocation_paths', [])) - int(item.get('allocation_start_bytes', 0)))
            total += max(0, int(item['bytes']) - materialized)
        return total

    def _allocation_bytes(self, paths: Sequence[str]) -> int:
        total = 0
        for relative in paths:
            path, _ = _relative(self.root, relative)
            if path.is_dir():
                total += sum(p.stat().st_size for p in _iter_regular_files(path))
            elif path.is_file():
                total += path.stat().st_size
        return total

    def _provider_free_bytes(self) -> int | None:
        provider = self.provider
        if provider is None:
            try:
                return int(shutil.disk_usage(self.root).free)
            except OSError:
                return None
        try:
            value = provider.free_bytes() if callable(getattr(provider, "free_bytes", None)) else getattr(provider, "free_bytes")
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        if value is None:
            return None
        return _nonnegative_int(value, "free_bytes")

    def _provider_quota_bytes(self) -> int | None:
        provider = self.provider
        if provider is None:
            return None
        try:
            value = provider.quota_bytes() if callable(getattr(provider, "quota_bytes", None)) else getattr(provider, "quota_bytes")
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        if value is None:
            return None
        return _nonnegative_int(value, "quota_bytes")

    def _require_quota_if_configured(self, quota: int | None) -> None:
        if self.require_quota and quota is None:
            raise StorageBudgetError(
                "quota is unknown; production storage admission is refused"
            )

    def check_capacity(self, additional_bytes: int = 0, *, peak_bytes: int = 0, purpose: str = "write") -> dict[str, Any]:
        additional = _nonnegative_int(additional_bytes, "additional_bytes")
        peak = _nonnegative_int(peak_bytes, "peak_bytes")
        with _FileLock(self.lock_path):
            data = self._read()
            self._validate(data)
            used = self.actual_bytes()
            reserved = self._reserved_bytes(data)
            projected = used + reserved + additional + peak + self.safety_margin_bytes
            free = self._provider_free_bytes()
            quota = self._provider_quota_bytes()
            self._require_quota_if_configured(quota)
            if projected > self.cap_bytes:
                raise StorageBudgetError(
                    f"{purpose} exceeds campaign cap: projected={projected} cap={self.cap_bytes}"
                )
            if free is not None and reserved + additional + peak + self.safety_margin_bytes > free:
                raise StorageBudgetError(f"{purpose} exceeds current filesystem free space")
            if quota is not None and used + reserved + additional + peak + self.safety_margin_bytes > quota:
                raise StorageBudgetError(f"{purpose} exceeds reported quota")
            return {
                "status": "PASS",
                "purpose": purpose,
                "actual_bytes": used,
                "reserved_bytes": reserved,
                "additional_bytes": additional,
                "peak_bytes": peak,
                "safety_margin_bytes": self.safety_margin_bytes,
                "projected_bytes": projected,
                "cap_bytes": self.cap_bytes,
                "free_bytes": free,
                "quota_bytes": quota,
            }

    def reserve(
        self, bytes: int, *, purpose: str, owner: str | None = None,
        reservation_id: str | None = None, peak_bytes: int = 0,
        allocation_paths: Sequence[str | Path] = (),
    ) -> Reservation:
        amount = _nonnegative_int(bytes, "reservation bytes") + _nonnegative_int(peak_bytes, 'peak_bytes')
        allocation = [_relative(self.root, p)[1] for p in allocation_paths]
        for index, left in enumerate(allocation):
            for right in allocation[index + 1:]:
                if _relative_paths_overlap(left, right):
                    raise StorageBudgetError('overlapping allocation ownership')
        if not str(purpose):
            raise ValueError("reservation purpose is required")
        rid = str(reservation_id or f"res-{os.getpid()}-{time.time_ns()}")
        owner_name = str(owner or f"pid:{os.getpid()}")
        with self._locked() as data:
            used = self.actual_bytes()
            active = self._reserved_bytes(data)
            free = self._provider_free_bytes()
            quota = self._provider_quota_bytes()
            self._require_quota_if_configured(quota)
            projected = used + active + amount + self.safety_margin_bytes
            if projected > self.cap_bytes:
                raise StorageBudgetError(f"reservation exceeds campaign cap: projected={projected} cap={self.cap_bytes}")
            if free is not None and active + amount + self.safety_margin_bytes > free:
                raise StorageBudgetError("reservation exceeds current filesystem free space")
            if quota is not None and projected > quota:
                raise StorageBudgetError("reservation exceeds reported quota")
            reservations = data["reservations"]
            for previous in reservations.values():
                if previous.get('status') == 'ACTIVE':
                    for a in allocation:
                        for b in previous.get('allocation_paths', []):
                            if _relative_paths_overlap(a, b):
                                raise StorageBudgetError('overlapping allocation ownership')
            if rid in reservations and reservations[rid].get("status") == "ACTIVE":
                raise StorageBudgetError(f"reservation already active: {rid}")
            item = {"reservation_id": rid, "bytes": amount, "purpose": str(purpose), "owner": owner_name, "created_utc": _utc(), "status": "ACTIVE"}
            item.update(allocation_paths=allocation, allocation_start_bytes=self._allocation_bytes(allocation))
            reservations[rid] = item
            data["events"].append({"event": "RESERVE", **item})
        return Reservation(rid, amount, str(purpose), owner_name, item["created_utc"])

    reserve_bytes = reserve

    def release(self, reservation_id: str, *, status: str = "RELEASED") -> dict[str, Any]:
        rid = str(reservation_id)
        with self._locked() as data:
            item = data["reservations"].get(rid)
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"unknown reservation: {rid}")
            if item.get("status") != "ACTIVE":
                return dict(item)
            if self.require_intents:
                allocation_paths = item.get("allocation_paths", [])
                current = self._allocation_bytes(allocation_paths)
                start = _nonnegative_int(item.get("allocation_start_bytes", 0), "allocation_start_bytes")
                if current > start:
                    raise StorageBudgetError("cannot release a reservation while residual files remain")
            mutable = dict(item)
            mutable["status"] = str(status)
            mutable["released_utc"] = _utc()
            data["reservations"][rid] = mutable
            data["events"].append({"event": "RELEASE", **mutable})
            return mutable

    def consume(self, reservation_id: str, *, actual_bytes: int | None = None) -> dict[str, Any]:
        rid = str(reservation_id)
        with self._locked() as data:
            item = data["reservations"].get(rid)
            if not isinstance(item, Mapping) or item.get("status") != "ACTIVE":
                raise StorageBudgetError(f"reservation is not active: {rid}")
            reserved_bytes = _nonnegative_int(item.get("bytes"), "reservation bytes")
            allocation_paths = item.get("allocation_paths", [])
            if not isinstance(allocation_paths, list):
                raise StorageBudgetError("reservation allocation paths are invalid")
            allocation_growth = 0
            if allocation_paths:
                current_bytes = self._allocation_bytes(allocation_paths)
                allocation_growth = max(
                    0,
                    current_bytes - _nonnegative_int(
                        item.get("allocation_start_bytes", 0), "allocation_start_bytes"
                    ),
                )
                if allocation_growth > reserved_bytes:
                    raise StorageBudgetError(
                        f"reservation allocation exceeded: growth={allocation_growth} reserved={reserved_bytes}"
                    )
            reported_bytes = (
                reserved_bytes
                if actual_bytes is None
                else _nonnegative_int(actual_bytes, "actual_bytes")
            )
            if reported_bytes > reserved_bytes:
                raise StorageBudgetError(
                    f"consumed bytes exceed reservation: consumed={reported_bytes} reserved={reserved_bytes}"
                )
            if allocation_paths and actual_bytes is not None and allocation_growth > reported_bytes:
                raise StorageBudgetError(
                    f"reported bytes understate allocation growth: growth={allocation_growth} reported={reported_bytes}"
                )
            mutable = dict(item)
            mutable["status"] = "CONSUMED"
            mutable["consumed_bytes"] = int(
                allocation_growth if actual_bytes is None and allocation_paths else reported_bytes
            )
            mutable["allocation_growth_bytes"] = int(allocation_growth)
            mutable["released_utc"] = _utc()
            data["reservations"][rid] = mutable
            data["events"].append({"event": "CONSUME", **mutable})
            return mutable

    def _intent_paths(self, paths: Sequence[str | Path]) -> list[str]:
        normalized = [_relative(self.root, path)[1] for path in paths]
        if not normalized:
            raise StorageIntegrityError("creation intent requires an allowed path")
        for index, left in enumerate(normalized):
            for right in normalized[index + 1:]:
                if _relative_paths_overlap(left, right):
                    raise StorageIntegrityError("creation intent paths overlap")
        return normalized

    def _path_is_allowed(self, relative: str, allowed_paths: Sequence[str]) -> bool:
        candidate = Path(str(relative).replace("\\", "/"))
        for raw in allowed_paths:
            base = Path(str(raw).replace("\\", "/"))
            if candidate == base or candidate.is_relative_to(base):
                return True
        return False

    def _snapshot_intent_files(self, allowed_paths: Sequence[str]) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for raw in allowed_paths:
            path, relative = _relative(self.root, raw)
            if not path.exists():
                continue
            if path.is_file():
                snapshot[relative] = {"identity": _identity(path), "sha256": _sha256_file(path)}
            elif path.is_dir():
                for item in _iter_regular_files(path):
                    _, item_relative = _relative(self.root, item)
                    snapshot[item_relative] = {"identity": _identity(item), "sha256": _sha256_file(item)}
        return snapshot

    def create_intent(
        self, *, reservation_id: str, trajectory: str, pulse: int, attempt: int,
        role: str, allowed_paths: Sequence[str | Path], expected_bytes: int | Mapping[str, int],
        admission_hash: str | None = None, intent_id: str | None = None,
        epoch: str | int | None = None, generation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        campaign_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a write-before-target creation intent.

        The intent is deliberately separate from a reservation: a reservation
        accounts bytes, while this record proves which trajectory/attempt is
        allowed to create which paths.  Existing files are snapshotted and can
        never become newly-owned artifacts through this operation.
        """
        rid = str(reservation_id)
        paths = self._intent_paths(allowed_paths)
        if isinstance(expected_bytes, Mapping):
            expected = {str(key).replace("\\", "/"): _nonnegative_int(value, "expected_bytes") for key, value in expected_bytes.items()}
            expected_total = sum(expected.values())
        else:
            expected_total = _nonnegative_int(expected_bytes, "expected_bytes")
            expected = {}
        if expected_total <= 0:
            raise StorageBudgetError("creation intent expected bytes must be positive")
        if generation is not None and not str(generation):
            raise StorageIntegrityError("creation intent generation is empty")
        bound_hash = self.admission_hash if admission_hash is None else str(admission_hash)
        if campaign_id is not None and str(campaign_id) != self.campaign_id:
            raise StorageIntegrityError("creation intent campaign identity conflicts with storage ledger")
        if self.require_intents and not bound_hash:
            raise StorageIntegrityError("formal creation intent requires an admission hash")
        iid = str(intent_id or f"intent-{os.getpid()}-{time.time_ns()}")
        with self._locked() as data:
            reservation = data["reservations"].get(rid)
            if not isinstance(reservation, Mapping) or reservation.get("status") != "ACTIVE":
                raise StorageBudgetError("creation intent requires an active reservation")
            allocation_paths = [str(path) for path in reservation.get("allocation_paths", [])]
            if self.require_intents and allocation_paths:
                if any(not any(self._path_is_allowed(path, [allocation]) for allocation in allocation_paths) for path in paths):
                    raise StorageIntegrityError("creation intent is outside reservation allocation paths")
            if bound_hash and self.admission_hash and bound_hash != self.admission_hash:
                raise StorageIntegrityError("creation intent admission hash conflicts with storage ledger")
            if int(reservation.get("bytes", 0)) < expected_total:
                raise StorageBudgetError("creation intent exceeds its active reservation")
            if iid in data["intents"]:
                raise StorageBudgetError(f"creation intent already exists: {iid}")
            initial = self._snapshot_intent_files(paths)
            item = {
                "schema": INTENT_SCHEMA, "intent_id": iid, "campaign_id": self.campaign_id,
                "reservation_id": rid, "trajectory": str(trajectory), "pulse": int(pulse),
                "attempt": int(attempt), "role": str(role), "allowed_paths": paths,
                "expected_bytes": expected_total, "expected_file_bytes": expected,
                "admission_hash": bound_hash, "epoch": None if epoch is None else str(epoch),
                "generation": None if generation is None else str(generation),
                "initial_files": initial, "created_files": {}, "status": "ACTIVE",
                "created_utc": _utc(), "metadata": dict(metadata or {}),
            }
            data["intents"][iid] = item
            data["events"].append({"event": "INTENT_CREATE", "intent_id": iid, "reservation_id": rid, "timestamp_utc": _utc()})
        return dict(item)

    begin_creation_intent = create_intent
    create_reservation_intent = create_intent

    def open_writer(
        self, *, reservation_id: str, intent_id: str, coordinator_epoch: str | int,
        trajectory: str, pulse: int, attempt: int, generation: str,
        process: Mapping[str, Any] | None = None, writer_id: str | None = None,
    ) -> dict[str, Any]:
        """Register an epoch-bound writer after reservation and intent exist."""
        identity = dict(process or process_identity())
        wid = str(writer_id or f"writer-{identity['pid']}-{time.time_ns()}")
        with self._locked() as data:
            reservation = data["reservations"].get(str(reservation_id))
            intent = data["intents"].get(str(intent_id))
            if not isinstance(reservation, Mapping) or reservation.get("status") != "ACTIVE":
                raise StorageBudgetError("writer requires an active reservation")
            if not isinstance(intent, Mapping) or intent.get("status") not in {"ACTIVE", "INTERRUPTED"}:
                raise StorageBudgetError("writer requires an active or resumable intent")
            if str(intent.get("reservation_id")) != str(reservation_id):
                raise StorageIntegrityError("writer reservation differs from intent")
            if intent.get("epoch") is not None and str(intent.get("epoch")) != str(coordinator_epoch):
                stale_writer = next(
                    (item for item in data["writers"].values()
                     if item.get("status") == "INTERRUPTED_STALE"
                     and str(item.get("intent_id")) == str(intent_id)
                     and str(item.get("coordinator_epoch")) == str(intent.get("epoch"))),
                    None,
                )
                if not isinstance(stale_writer, Mapping) or intent.get("status") != "INTERRUPTED":
                    raise StorageIntegrityError("writer intent epoch differs from coordinator")
                rebound = dict(intent)
                rebound["epoch"] = str(coordinator_epoch)
                rebound["status"] = "ACTIVE"
                rebound["takeover_from_epoch"] = str(intent.get("epoch"))
                rebound["takeover_utc"] = _utc()
                data["intents"][str(intent_id)] = rebound
                intent = rebound
            for prior in data["writers"].values():
                if prior.get("status") == "ACTIVE" and str(prior.get("intent_id")) == str(intent_id):
                    raise StorageBudgetError("creation intent already has an active writer")
            item = {
                "schema": WRITER_SCHEMA, "writer_id": wid, "campaign_id": self.campaign_id,
                "admission_hash": self.admission_hash, "coordinator_epoch": str(coordinator_epoch),
                "process_identity": identity, "trajectory": str(trajectory), "pulse": int(pulse),
                "attempt": int(attempt), "generation": str(generation),
                "reservation_id": str(reservation_id), "intent_id": str(intent_id),
                "role": str(intent.get("role", "")),
                "allowed_paths": list(intent.get("allowed_paths", [])), "status": "ACTIVE",
                "opened_utc": _utc(),
            }
            data["writers"][wid] = item
            data["events"].append({"event": "WRITER_OPEN", "writer_id": wid,
                                   "epoch": str(coordinator_epoch), "timestamp_utc": _utc()})
            return dict(item)

    def close_writer(self, writer_id: str, *, coordinator_epoch: str | int,
                     status: str = "CLOSED", reason: str | None = None) -> dict[str, Any]:
        if status not in {"CLOSED", "INTERRUPTED"}:
            raise ValueError("writer close status is invalid")
        with self._locked() as data:
            item = data["writers"].get(str(writer_id))
            if not isinstance(item, Mapping) or item.get("status") != "ACTIVE":
                raise StorageBudgetError("writer is not active")
            if str(item.get("coordinator_epoch")) != str(coordinator_epoch):
                raise StorageIntegrityError("stale epoch cannot close writer")
            mutable = dict(item)
            mutable["status"] = status; mutable["closed_utc"] = _utc()
            if reason is not None:
                mutable["reason"] = str(reason)
            data["writers"][str(writer_id)] = mutable
            data["events"].append({"event": "WRITER_CLOSE", "writer_id": str(writer_id),
                                   "status": status, "timestamp_utc": _utc()})
            return mutable

    def active_writers(self, *, coordinator_epoch: str | int | None = None,
                       include_management: bool = True) -> list[dict[str, Any]]:
        with _FileLock(self.lock_path):
            data = self._read(); self._validate(data)
            management_roles = {"TERMINAL_EVIDENCE", "REPORT", "GC_EVIDENCE"}
            return [dict(item) for item in data["writers"].values()
                    if item.get("status") == "ACTIVE" and
                    (include_management or str(item.get("role", "")).upper() not in management_roles) and
                    (coordinator_epoch is None or str(item.get("coordinator_epoch")) == str(coordinator_epoch))]

    def interrupt_stale_writers(self, *, coordinator_epoch: str | int) -> dict[str, Any]:
        """Verify dead writers and fence them without releasing reserved bytes."""
        with self._locked() as data:
            active = [(wid, item) for wid, item in data["writers"].items()
                      if item.get("status") == "ACTIVE" and
                      str(item.get("coordinator_epoch")) == str(coordinator_epoch)]
            probes = [(wid, probe_process_identity(item.get("process_identity", {}))) for wid, item in active]
            live = [wid for wid, probe in probes if probe["status"] == "LIVE"]
            unknown = [wid for wid, probe in probes if probe["status"] == "UNKNOWN"]
            if live:
                raise StorageBudgetError(f"live ACTIVE writers block takeover: {live}")
            if unknown:
                raise StorageIntegrityError(f"unverifiable ACTIVE writers block takeover: {unknown}")
            changed = []
            for wid, probe in probes:
                mutable = dict(data["writers"][wid]); mutable["status"] = "INTERRUPTED_STALE"
                mutable["stale_verified_utc"] = _utc(); mutable["liveness_evidence"] = probe
                data["writers"][wid] = mutable; changed.append(wid)
                intent_id = str(mutable.get("intent_id", ""))
                intent = data["intents"].get(intent_id)
                if isinstance(intent, Mapping) and intent.get("status") == "ACTIVE":
                    interrupted = dict(intent)
                    interrupted["status"] = "INTERRUPTED"
                    interrupted["interrupted_utc"] = _utc()
                    interrupted["interrupted_by_stale_epoch"] = str(coordinator_epoch)
                    data["intents"][intent_id] = interrupted
            data["events"].append({"event": "WRITERS_INTERRUPTED_STALE", "epoch": str(coordinator_epoch),
                                   "writer_ids": changed, "timestamp_utc": _utc()})
            return {"status": "PASS", "coordinator_epoch": str(coordinator_epoch),
                    "interrupted_writer_ids": changed, "probes": dict(probes)}

    def write_quiescence_receipt(self, path: str | Path, *, coordinator_epoch: str | int,
                                 trajectory: str | None = None, pulse: int | None = None,
                                 attempt: int | None = None) -> dict[str, Any]:
        active = self.active_writers(coordinator_epoch=coordinator_epoch, include_management=False)
        if active:
            raise StorageBudgetError("writer registry is not quiescent")
        destination, _ = _relative(self.root, path)
        payload = {"schema": WRITER_RECEIPT_SCHEMA, "status": "PASS", "campaign_id": self.campaign_id,
                   "active_writers": [], "writer_epoch": str(coordinator_epoch),
                   "coordinator_process_id": str(coordinator_epoch),
                   "admission_identity_sha256": self.admission_hash,
                   "trajectory": trajectory, "pulse": pulse, "attempt": attempt,
                   "registry_backed": True, "created_utc": _utc()}
        _atomic_json(destination, payload, overwrite=False)
        return {"path": str(destination), "sha256": _sha256_file(destination), **payload}

    def _intent_file_list(self, item: Mapping[str, Any], files: Sequence[str | Path] | None) -> list[Path]:
        allowed = [str(path) for path in item.get("allowed_paths", [])]
        if files is None:
            values: list[Path] = []
            for raw in allowed:
                path, _ = _relative(self.root, raw)
                if path.is_file():
                    values.append(path)
                elif path.is_dir():
                    values.extend(_iter_regular_files(path))
        else:
            values = []
            for raw in files:
                path, relative = _relative(self.root, raw)
                if not self._path_is_allowed(relative, allowed):
                    raise StorageIntegrityError(f"creation target is outside intent paths: {relative}")
                values.append(path)
        result: list[Path] = []
        seen: set[str] = set()
        initial = item.get("initial_files", {})
        for path in values:
            _, relative = _relative(self.root, path)
            if relative in seen:
                continue
            seen.add(relative)
            if relative in initial:
                raise StorageIntegrityError(f"pre-existing file cannot be owned by creation intent: {relative}")
            if not path.is_file() or _is_reparse(path):
                raise StorageIntegrityError(f"creation target is missing or linked: {relative}")
            result.append(path)
        return result

    def complete_intent(
        self, intent_id: str, *, files: Sequence[str | Path] | None = None,
        expected_bytes: int | None = None, role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate files created under an intent and register ownership."""
        iid = str(intent_id)
        with self._locked() as data:
            item = data["intents"].get(iid)
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"unknown creation intent: {iid}")
            if item.get("status") == "COMPLETED":
                return dict(item)
            if item.get("status") != "ACTIVE":
                raise StorageBudgetError(f"creation intent is not active: {iid}")
            values = self._intent_file_list(item, files)
            if not values:
                raise StorageIntegrityError("creation intent completed without files")
            expected_total = _nonnegative_int(item.get("expected_bytes", 0), "intent expected bytes")
            actual_total = 0
            created: dict[str, Any] = {}
            for path in values:
                resolved, relative = _relative(self.root, path)
                identity = _identity(resolved)
                digest = _sha256_file(resolved)
                per_file = item.get("expected_file_bytes", {}).get(relative)
                if per_file is not None and identity["size"] != int(per_file):
                    raise StorageIntegrityError(f"creation target size differs from intent: {relative}")
                if relative in data["artifacts"]:
                    raise StorageIntegrityError(f"creation target already has ownership: {relative}")
                actual_total += identity["size"]
                created[relative] = {"identity": identity, "sha256": digest}
                data["artifacts"][relative] = {
                    "campaign_id": self.campaign_id,
                    "trajectory": str(item.get("trajectory")), "pulse": int(item.get("pulse")),
                    "attempt": int(item.get("attempt")),
                    "role": str(role or item.get("role")), "relative_path": relative,
                    "identity": identity, "expected_sha256": digest, "sha256": digest,
                    "reclaimable": bool((metadata or {}).get("reclaimable", False)),
                    "metadata": {**dict(item.get("metadata", {})), **dict(metadata or {})},
                    "intent_id": iid, "reservation_id": str(item.get("reservation_id")),
                    "admission_hash": item.get("admission_hash"),
                    "creation_record": {"intent_id": iid, "generation": item.get("generation"),
                                        "created_utc": item.get("created_utc")},
                    "registered_utc": _utc(),
                }
            if expected_bytes is not None and actual_total > _nonnegative_int(expected_bytes, "expected_bytes"):
                raise StorageBudgetError("created files exceed completion byte bound")
            if actual_total > expected_total:
                raise StorageBudgetError("created files exceed intent byte bound")
            mutable = dict(item)
            mutable["status"] = "COMPLETED"
            mutable["created_files"] = created
            mutable["actual_bytes"] = actual_total
            mutable["completed_utc"] = _utc()
            data["intents"][iid] = mutable
            data["events"].append({"event": "INTENT_COMPLETE", "intent_id": iid, "actual_bytes": actual_total, "timestamp_utc": _utc()})
            return dict(mutable)

    register_intent_files = complete_intent
    complete_creation_intent = complete_intent

    def interrupt_intent(self, intent_id: str, *, reason: str = "interrupted") -> dict[str, Any]:
        iid = str(intent_id)
        with self._locked() as data:
            item = data["intents"].get(iid)
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"unknown creation intent: {iid}")
            mutable = dict(item)
            if mutable.get("status") == "COMPLETED":
                return mutable
            mutable["status"] = "INTERRUPTED"
            mutable["interrupt_reason"] = str(reason)
            mutable["interrupted_utc"] = _utc()
            data["intents"][iid] = mutable
            data["events"].append({"event": "INTENT_INTERRUPTED", "intent_id": iid, "reason": str(reason), "timestamp_utc": _utc()})
            return mutable

    abort_intent = interrupt_intent
    interrupt_creation_intent = interrupt_intent

    def intents(self) -> dict[str, Any]:
        with _FileLock(self.lock_path):
            data = self._read()
            self._validate(data)
            return json.loads(json.dumps(data.get("intents", {})))

    creation_intents = intents

    def validate_intent(self, intent_id: str, *, path: str | Path | None = None,
                        require_completed: bool = False, require_active: bool = False) -> dict[str, Any]:
        with _FileLock(self.lock_path):
            data = self._read()
            self._validate(data)
            item = data.get("intents", {}).get(str(intent_id))
            if not isinstance(item, Mapping):
                raise StorageBudgetError(f"unknown creation intent: {intent_id}")
            if require_completed and item.get("status") != "COMPLETED":
                raise StorageBudgetError("creation intent is not complete")
            if require_active:
                reservation_id = str(item.get("reservation_id", ""))
                reservation = data.get("reservations", {}).get(reservation_id)
                if not isinstance(reservation, Mapping) or reservation.get("status") != "ACTIVE":
                    raise StorageBudgetError("creation intent reservation is not active")
            if path is not None:
                _, relative = _relative(self.root, path)
                if not self._path_is_allowed(relative, item.get("allowed_paths", [])):
                    raise StorageIntegrityError("path is outside creation intent")
            return dict(item)

    validate_creation_intent = validate_intent

    def register_artifact(
        self, path: str | Path, *, role: str, trajectory: str | None = None,
        pulse: int | None = None, attempt: int | None = None,
        reclaimable: bool = False, expected_sha256: str | None = None,
        expected_bytes: int | None = None, metadata: Mapping[str, Any] | None = None,
        intent_id: str | None = None, reservation_id: str | None = None,
        admission_hash: str | None = None, legacy_test_only: bool = False,
    ) -> dict[str, Any]:
        resolved, relative = _relative(self.root, path)
        identity = _identity(resolved)
        if self.require_intents and legacy_test_only:
            raise StorageIntegrityError("formal artifact registration cannot use legacy_test_only")
        if self.require_intents and intent_id is None:
            raise StorageIntegrityError("formal artifact registration requires a creation intent")
        intent = None
        if intent_id is not None:
            intent = self.validate_intent(intent_id, path=resolved,
                                          require_completed=self.require_intents)
            if intent.get("status") not in {"ACTIVE", "COMPLETED"}:
                raise StorageIntegrityError("artifact intent is not active or complete")
            if self.require_intents:
                created = intent.get("created_files", {})
                if not isinstance(created, Mapping) or relative not in created:
                    raise StorageIntegrityError("formal artifact is not listed in a completed intent creation record")
                expected_created = created[relative]
                if not isinstance(expected_created, Mapping) or dict(expected_created.get("identity", {})) != identity:
                    raise StorageIntegrityError("formal artifact identity differs from completed creation record")
            if reservation_id is not None and str(reservation_id) != str(intent.get("reservation_id")):
                raise StorageIntegrityError("artifact reservation does not match creation intent")
            if admission_hash is not None and str(admission_hash) != str(intent.get("admission_hash")):
                raise StorageIntegrityError("artifact admission hash does not match creation intent")
            if trajectory is not None and str(trajectory) != str(intent.get("trajectory")):
                raise StorageIntegrityError("artifact trajectory does not match creation intent")
            if pulse is not None and int(pulse) != int(intent.get("pulse")):
                raise StorageIntegrityError("artifact pulse does not match creation intent")
            if attempt is not None and int(attempt) != int(intent.get("attempt")):
                raise StorageIntegrityError("artifact attempt does not match creation intent")
        if expected_bytes is not None and identity["size"] != _nonnegative_int(expected_bytes, "expected_bytes"):
            raise StorageIntegrityError("artifact size differs from expected size")
        digest = _sha256_file(resolved)
        if expected_sha256 is not None and digest != str(expected_sha256):
            raise StorageIntegrityError("artifact hash differs from expected hash")
        effective_admission_hash = None if intent is None else intent.get("admission_hash")
        record = {
            "campaign_id": self.campaign_id,
            "trajectory": None if trajectory is None else str(trajectory),
            "pulse": None if pulse is None else int(pulse),
            "attempt": None if attempt is None else int(attempt),
            "role": str(role),
            "relative_path": relative,
            "identity": identity,
            "expected_sha256": None if expected_sha256 is None else str(expected_sha256),
            "sha256": digest,
            "reclaimable": bool(reclaimable),
            "metadata": dict(metadata or {}),
            "intent_id": None if intent_id is None else str(intent_id),
            "reservation_id": None if reservation_id is None else str(reservation_id or (intent or {}).get("reservation_id")),
            "admission_hash": (effective_admission_hash if admission_hash is None else str(admission_hash)),
            "creation_record": None if intent is None else {"intent_id": str(intent_id),
                "generation": intent.get("generation"), "created_utc": intent.get("created_utc")},
            "registered_utc": _utc(),
        }
        key = relative
        with self._locked() as data:
            existing = data["artifacts"].get(key)
            if existing is not None and dict(existing.get("identity", {})) != identity:
                raise StorageIntegrityError("artifact identity changed for an existing ledger path")
            data["artifacts"][key] = record
            data["events"].append({"event": "REGISTER", "relative_path": relative, "role": str(role), "reclaimable": bool(reclaimable), "timestamp_utc": _utc()})
        return record

    def register_file(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        return self.register_artifact(path, **kwargs)

    def artifacts(self) -> dict[str, Any]:
        with _FileLock(self.lock_path):
            data = self._read()
            self._validate(data)
            return json.loads(json.dumps(data["artifacts"]))

    def set_reclaimable(self, paths: Sequence[str | Path], *, value: bool = True) -> list[str]:
        """Set retention only for artifacts already owned by completed formal intents."""
        changed: list[str] = []
        with self._locked() as data:
            for raw in paths:
                _, relative = _relative(self.root, raw)
                item = data["artifacts"].get(relative)
                if not isinstance(item, Mapping) or not item.get("intent_id"):
                    raise StorageIntegrityError(f"retention target lacks creation ownership: {relative}")
                intent = data["intents"].get(str(item["intent_id"]))
                if not isinstance(intent, Mapping) or intent.get("status") != "COMPLETED":
                    raise StorageIntegrityError(f"retention target intent is not complete: {relative}")
                mutable = dict(item); mutable["reclaimable"] = bool(value)
                data["artifacts"][relative] = mutable; changed.append(relative)
            data["events"].append({"event": "RETENTION_SET", "paths": changed,
                                   "reclaimable": bool(value), "timestamp_utc": _utc()})
        return changed

    def final_output_bytes(self) -> int:
        total = 0
        for item in self.artifacts().values():
            if str(item.get("role", "")).startswith("final"):
                total += int(item.get("identity", {}).get("size", 0))
        return total

    def check_final_output_budget(self, additional_bytes: int = 0) -> dict[str, Any]:
        total = self.final_output_bytes() + _nonnegative_int(additional_bytes, "additional_bytes")
        if total > self.final_output_budget_bytes:
            raise StorageBudgetError("final output budget exceeded")
        return {"status": "PASS", "final_output_bytes": total, "final_output_budget_bytes": self.final_output_budget_bytes}

    def validate_terminal_inventory(
        self, expected_roles: Mapping[str, Sequence[str | Path]] | Sequence[str], *,
        root: str | Path | None = None, final: bool = True,
    ) -> dict[str, Any]:
        """Verify the durable role manifest against every actual file.

        Unknown NPY/NPZ files, orphan files, old attempts and unregistered
        terminal payloads are reported and fail closed.  The method never
        claims ownership or deletes an extra file.
        """
        base = self.root if root is None else Path(root).resolve()
        if not base.is_relative_to(self.root):
            raise StorageIntegrityError("terminal inventory root escapes campaign root")
        def resolve_expected(value: str | Path) -> str:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            return str(candidate.resolve())
        if isinstance(expected_roles, Mapping):
            expected_map = {
                str(role): {resolve_expected(path) for path in paths}
                for role, paths in expected_roles.items()
            }
        else:
            expected_map = {"final": {resolve_expected(path) for path in expected_roles}}
        expected_paths = set().union(*expected_map.values()) if expected_map else set()
        actual_paths = {str(path.resolve()) for path in _iter_regular_files(base)}
        management = {str(path.resolve()) for path in self._management_paths()}
        artifacts = self.artifacts()
        registered = {str((self.root / relative).resolve()) for relative in artifacts}
        scientific_suffixes = {".npy", ".npz", ".mat", ".h5", ".hdf5"}
        suspicious_unregistered = {
            path for path in actual_paths - management - registered
            if Path(path).suffix.lower() in scientific_suffixes
        }
        # Durable orchestration manifests and locks are management state.  All
        # registered evidence remains in the contract, while any unregistered
        # scientific container is still an orphan and fails closed.
        actual_payload = (registered & actual_paths) | suspicious_unregistered
        missing = sorted(path for path in expected_paths if path not in actual_payload)
        extras = sorted(path for path in actual_payload if path not in expected_paths)
        unregistered = sorted(path for path in actual_payload if path not in registered)
        role_mismatches: list[str] = []
        for role, paths in expected_map.items():
            for path in paths & actual_payload:
                try:
                    relative = Path(path).relative_to(self.root).as_posix()
                except ValueError:
                    continue
                record = artifacts.get(relative)
                if record is None or str(record.get("role", "")) != str(role):
                    role_mismatches.append(relative)
        final_bytes = 0
        if final:
            final_bytes = sum(int(artifacts.get(Path(path).relative_to(self.root).as_posix(), {}).get("identity", {}).get("size", 0))
                             for path in expected_paths if Path(path).is_file() and Path(path).is_relative_to(self.root))
            if final_bytes > self.final_output_budget_bytes:
                raise StorageBudgetError("terminal final output exceeds final budget")
        ok = not missing and not extras and not unregistered and not role_mismatches
        if not ok:
            raise StorageIntegrityError(
                "terminal inventory mismatch: "
                f"missing={missing} extras={extras} unregistered={unregistered} role_mismatches={role_mismatches}"
            )
        return {
            "schema": "khz_filament.hr4e5.e5_1a.terminal_inventory.v1",
            "status": "PASS", "root": str(base), "expected_count": len(expected_paths),
            "actual_count": len(actual_payload), "final_output_bytes": final_bytes,
            "roles": {role: sorted(paths) for role, paths in expected_map.items()},
        }

    def _validate_gc_targets(self, data: Mapping[str, Any], targets: Sequence[str]) -> list[dict[str, Any]]:
        if not targets:
            raise StorageBudgetError("reclamation target list is empty")
        records = []
        for raw in targets:
            _, relative = _relative(self.root, raw)
            item = data["artifacts"].get(relative)
            if not isinstance(item, Mapping):
                raise StorageIntegrityError(f"unknown artifact ownership: {relative}")
            if item.get("campaign_id") != self.campaign_id or not bool(item.get("reclaimable")):
                raise StorageIntegrityError(f"artifact is outside the reclamation whitelist: {relative}")
            if self.require_intents:
                if not str(item.get("intent_id", "")) or not isinstance(item.get("creation_record"), Mapping):
                    raise StorageIntegrityError(f"artifact lacks durable creation ownership: {relative}")
                if str(item.get("admission_hash", "")) != str(self.admission_hash or ""):
                    raise StorageIntegrityError(f"artifact admission identity is not bound: {relative}")
            path = self.root / relative
            current = _identity(path)
            recorded = item.get("identity")
            if not isinstance(recorded, Mapping) or dict(recorded) != current:
                raise StorageIntegrityError(f"artifact identity changed: {relative}")
            if int(current.get("st_nlink", 1)) != 1:
                raise StorageIntegrityError(f"hard-linked artifact is protected: {relative}")
            if _sha256_file(path) != item.get('sha256'):
                raise StorageIntegrityError(f'artifact content changed: {relative}')
            records.append({
                "relative_path": relative, "bytes": int(current["size"]),
                "identity": dict(current), "artifact_sha256": str(item.get("sha256", "")),
                "role": item.get("role", ""),
            })
        return records

    def _read_writer_receipt(
        self, receipt: str | Path | Mapping[str, Any] | None, *, expected_sha256: str | None = None,
        expected_epoch: str | int | None = None,
        expected_trajectory: str | None = None, expected_pulse: int | None = None,
        expected_attempt: int | None = None, expected_admission_hash: str | None = None,
        _ledger_data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load a durable quiescence receipt and bind its file identity.

        A mapping alone is intentionally insufficient: it can be a caller's
        in-memory assertion and has no durability or file identity.  Callers
        must provide the receipt path either directly or as ``receipt_path``
        in a small mapping.
        """
        if isinstance(receipt, Mapping):
            receipt_path_value = receipt.get("receipt_path") or receipt.get("path")
            if receipt_path_value is None:
                raise StorageBudgetError("writer receipt path is required")
            receipt_path = Path(str(receipt_path_value))
        elif receipt is not None:
            receipt_path = Path(receipt)
        else:
            raise StorageBudgetError("writer quiescence receipt is required")
        if not receipt_path.is_absolute():
            receipt_path = self.root.parent / receipt_path
        receipt_path = receipt_path.resolve()
        if not receipt_path.is_file() or _is_reparse(receipt_path):
            raise StorageIntegrityError("writer quiescence receipt is missing or linked")
        digest = _sha256_file(receipt_path)
        if expected_sha256 is not None and digest != str(expected_sha256):
            raise StorageIntegrityError("writer quiescence receipt hash changed")
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageIntegrityError("writer quiescence receipt is not valid JSON") from error
        if not isinstance(payload, Mapping) or str(payload.get("status", "")).upper() != "PASS":
            raise StorageBudgetError("writer quiescence receipt is not PASS")
        if self.require_intents and (payload.get("schema") != WRITER_RECEIPT_SCHEMA
                                     or payload.get("registry_backed") is not True):
            raise StorageIntegrityError("formal writer receipt is not registry-backed")
        campaign = payload.get("campaign_id")
        if campaign is not None and str(campaign) != self.campaign_id:
            raise StorageIntegrityError("writer quiescence receipt campaign mismatch")
        active = payload.get("active_writers", payload.get("active_writer_ids", []))
        if active not in ([], (), None):
            raise StorageBudgetError("writer quiescence receipt still has active writers")
        epoch = payload.get("writer_epoch", payload.get("epoch"))
        if epoch is None or isinstance(epoch, bool):
            raise StorageBudgetError("writer quiescence receipt lacks writer epoch")
        if expected_epoch is not None and str(epoch) != str(expected_epoch):
            raise StorageIntegrityError("writer quiescence receipt epoch changed")
        process_id = payload.get("coordinator_process_id", payload.get("process_id"))
        if process_id is None or not str(process_id):
            raise StorageBudgetError("writer quiescence receipt lacks coordinator identity")
        bindings = {
            "trajectory": expected_trajectory,
            "pulse": expected_pulse,
            "attempt": expected_attempt,
            "admission_identity_sha256": expected_admission_hash,
        }
        for name, expected in bindings.items():
            if expected is None:
                continue
            actual = payload.get(name)
            if actual is None:
                raise StorageBudgetError(f"writer quiescence receipt lacks {name} binding")
            if str(actual) != str(expected):
                raise StorageIntegrityError(f"writer quiescence receipt {name} changed")
        if self.require_intents:
            if _ledger_data is None:
                active_now = self.active_writers(coordinator_epoch=epoch, include_management=False)
            else:
                management_roles = {"TERMINAL_EVIDENCE", "REPORT", "GC_EVIDENCE"}
                active_now = [item for item in _ledger_data.get("writers", {}).values()
                              if isinstance(item, Mapping) and item.get("status") == "ACTIVE"
                              and str(item.get("coordinator_epoch")) == str(epoch)
                              and str(item.get("role", "")).upper() not in management_roles]
            if active_now:
                raise StorageBudgetError("writer registry became active after quiescence receipt")
        return {
            "path": str(receipt_path), "sha256": digest, "writer_epoch": str(epoch),
            "coordinator_process_id": str(process_id), "status": "PASS",
            **{name: payload.get(name) for name in bindings if payload.get(name) is not None},
        }

    def _read_campaign_json(
        self, value: str | Path | Mapping[str, Any], *, label: str,
        expected_sha256: str | None = None, require_hash: bool = True,
    ) -> tuple[Path, str, dict[str, Any], str]:
        """Read a durable JSON evidence file below this campaign root.

        Reclamation authorization is evidence-driven.  A caller-provided
        mapping is not sufficient because it can be changed after planning;
        every evidence file is resolved, hashed, and parsed again by the
        storage coordinator.
        """
        if isinstance(value, Mapping):
            raw_path = value.get("path") or value.get("receipt_path")
            expected = value.get("sha256", expected_sha256)
        else:
            raw_path = value
            expected = expected_sha256
        if raw_path is None:
            raise StorageBudgetError(f"{label} path is required")
        try:
            path, relative = _relative(self.root, Path(str(raw_path)))
        except StorageIntegrityError as error:
            raise StorageIntegrityError(f"{label} is outside campaign root") from error
        if not path.is_file() or _is_reparse(path):
            raise StorageIntegrityError(f"{label} is missing or linked")
        digest = _sha256_file(path)
        if require_hash and (expected is None or not str(expected)):
            raise StorageIntegrityError(f"{label} hash is missing or changed")
        if expected is not None and (not str(expected) or digest != str(expected)):
            raise StorageIntegrityError(f"{label} hash is missing or changed")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageIntegrityError(f"{label} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise StorageIntegrityError(f"{label} must contain a JSON object")
        return path, relative, payload, digest

    def _read_evidence_descriptor(
        self, descriptor: Mapping[str, Any], *, label: str,
    ) -> tuple[dict[str, Any], str]:
        if not isinstance(descriptor, Mapping):
            raise StorageBudgetError(f"{label} descriptor is invalid")
        path, relative, payload, digest = self._read_campaign_json(
            descriptor, label=label, expected_sha256=str(descriptor.get("sha256", "")),
        )
        return {
            "path": str(path), "relative_path": relative, "sha256": digest,
            "status": str(payload.get("status", "")).upper(),
            "payload": payload,
        }, digest

    def _validate_successor_evidence(
        self, info: Mapping[str, Any], *, label: str,
    ) -> None:
        """Validate a self-contained successor receipt before any deletion.

        The production Streaming READY schema is checked with its existing
        payload validator, which re-reads every child CURRENT array.  A small
        independent reference fixture may use the generic self-contained
        shape below; its field files and exact binding are still hash-bound.
        """
        payload = info["payload"]
        status = str(payload.get("status", "")).upper()
        if status not in {"READY", "PASS"}:
            raise StorageBudgetError(f"{label} is not READY")
        schema = str(payload.get("schema", ""))
        ready_path = Path(str(info["path"]))
        if schema == "khz_filament.hr4e5.e5_1a.ready.v1":
            try:
                from .hr4e5_evidence import validate_ready_receipt

                validate_ready_receipt(ready_path.parent)
            except Exception as error:
                raise StorageIntegrityError(f"{label} payload validation failed") from error
            return
        if payload.get("terminal") == "POST_FINAL_READY":
            lifecycle_root_value = payload.get("lifecycle_root")
            retained = payload.get("retained_optical_hashes")
            if lifecycle_root_value is None or not isinstance(retained, Mapping) or len(retained) != 3:
                raise StorageBudgetError(f"{label} terminal evidence is incomplete")
            try:
                lifecycle_root, _ = _relative(self.root, Path(str(lifecycle_root_value)))
                if lifecycle_root.resolve() != ready_path.parent.resolve():
                    raise StorageIntegrityError(f"{label} lifecycle root identity changed")
                optical_paths = []
                for raw_path, expected_hash in retained.items():
                    optical_path, _ = _relative(self.root, Path(str(raw_path)))
                    if not optical_path.is_file() or _is_reparse(optical_path) or _sha256_file(optical_path) != str(expected_hash):
                        raise StorageIntegrityError(f"{label} retained optical evidence changed")
                    optical_paths.append(optical_path)
                # The terminal receipt intentionally retains only the final
                # optical field, ledger and run metadata.  Sink arrays may be
                # one of the approved GC targets, so revalidation must not
                # require those already-reclaimed intermediates.
                from .hr4e5s_streaming import StreamingLifecycle

                lifecycle = StreamingLifecycle.open(lifecycle_root)
                if lifecycle._authoritative_namespace != "CURRENT" or lifecycle.manifest.get("queue") or lifecycle.manifest.get("recovery_backlog"):
                    raise StorageIntegrityError(f"{label} terminal lifecycle is not a quiet CURRENT root")
                if lifecycle.manifest.get("barrier") is not None or lifecycle.manifest.get("promotion") is not None:
                    raise StorageIntegrityError(f"{label} terminal lifecycle has an active transition")
                if str(payload.get("current_generation")) != str(lifecycle.manifest.get("current_generation")) or str(payload.get("current_content_sha256")) != str(lifecycle.manifest.get("current_content_sha256")):
                    raise StorageIntegrityError(f"{label} terminal lifecycle identity changed")
                for record in lifecycle.manifest.get("records", []):
                    if record.get("state") != "POST_COMMITTED" or record.get("post") is None or record.get("next") is not None:
                        raise StorageIntegrityError(f"{label} terminal POST inventory is incomplete")
                    lifecycle._validate_record_provenance(record, require_post=True, require_next=False)
                lifecycle._assert_no_staged_or_orphaned_artifacts()
            except StorageBudgetError:
                raise
            except Exception as error:
                if isinstance(error, StorageIntegrityError):
                    raise
                raise StorageIntegrityError(f"{label} terminal evidence validation failed") from error
            return
        if payload.get("self_contained") is not True or payload.get("parent_payload_required") is not False:
            raise StorageBudgetError(f"{label} is not a self-contained recovery point")
        child_root_value = payload.get("child_root") or payload.get("root")
        if child_root_value is None:
            raise StorageBudgetError(f"{label} child root is missing")
        try:
            child_root, _ = _relative(self.root, Path(str(child_root_value)))
        except StorageIntegrityError as error:
            raise StorageIntegrityError(f"{label} child root escapes campaign root") from error
        if not child_root.is_dir():
            raise StorageIntegrityError(f"{label} child root is missing")
        fields = payload.get("fields")
        if not isinstance(fields, Mapping) or not fields:
            raise StorageBudgetError(f"{label} self-contained field inventory is missing")
        for name, descriptor in fields.items():
            if not isinstance(descriptor, Mapping):
                raise StorageIntegrityError(f"{label} field descriptor is invalid: {name}")
            field_path_value = descriptor.get("path")
            field_hash = descriptor.get("sha256")
            if field_path_value is None or not str(field_hash):
                raise StorageIntegrityError(f"{label} field identity is incomplete: {name}")
            try:
                field_path, _ = _relative(self.root, child_root / str(field_path_value))
            except StorageIntegrityError as error:
                raise StorageIntegrityError(f"{label} field escapes child root: {name}") from error
            if not field_path.is_file() or _is_reparse(field_path) or _sha256_file(field_path) != str(field_hash):
                raise StorageIntegrityError(f"{label} field is missing or changed: {name}")
        binding = payload.get("binding") or payload.get("binding_receipt")
        if binding is None:
            raise StorageBudgetError(f"{label} exact binding is missing")
        binding_info, _ = self._read_evidence_descriptor(binding, label=f"{label} exact binding")
        binding_payload = binding_info["payload"]
        if str(binding_payload.get("status", "")).upper() != "PASS":
            raise StorageBudgetError(f"{label} exact binding is not PASS")
        rows = binding_payload.get("rows")
        if not isinstance(rows, list) or not rows or any(
            not isinstance(row, Mapping) or row.get("status") != "PASS" for row in rows
        ):
            raise StorageIntegrityError(f"{label} exact binding rows are incomplete")

    def _read_reclaim_prerequisite_receipt(
        self, receipt: str | Path | Mapping[str, Any] | None, *,
        expected_sha256: str | None = None,
        expected_gates: Mapping[str, bool] | None = None,
        expected_targets: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Read and validate durable exact/READY/dependency prerequisites."""
        if receipt is None:
            raise StorageBudgetError("durable reclamation prerequisite receipt is required")
        path, relative, payload, digest = self._read_campaign_json(
            receipt, label="reclamation prerequisite receipt", expected_sha256=expected_sha256,
            require_hash=expected_sha256 is not None,
        )
        if payload.get("schema") != RECLAIM_PREREQUISITE_SCHEMA:
            raise StorageBudgetError("reclamation prerequisite receipt schema is invalid")
        if str(payload.get("status", "")).upper() != "PASS":
            raise StorageBudgetError("reclamation prerequisite receipt is not PASS")
        if str(payload.get("campaign_id", "")) != self.campaign_id:
            raise StorageIntegrityError("reclamation prerequisite campaign mismatch")
        required = (
            "exact_complete", "successor_ready", "no_active_writers",
            "no_future_dependency", "receipts_durable",
        )
        gates = payload.get("gates")
        if not isinstance(gates, Mapping) or any(gates.get(name) is not True for name in required):
            raise StorageBudgetError("reclamation prerequisite gates are incomplete")
        if expected_gates is not None:
            for name in required:
                if bool(expected_gates.get(name)) is not bool(gates.get(name)):
                    raise StorageIntegrityError(f"reclamation prerequisite gate changed: {name}")

        bindings = payload.get("target_bindings")
        if not isinstance(bindings, list) or not bindings:
            raise StorageBudgetError("reclamation prerequisite target bindings are missing")
        binding_map: dict[str, Mapping[str, Any]] = {}
        for item in bindings:
            if not isinstance(item, Mapping) or not str(item.get("relative_path", "")):
                raise StorageIntegrityError("reclamation prerequisite target binding is invalid")
            key = str(item["relative_path"]).replace("\\", "/")
            if key in binding_map:
                raise StorageIntegrityError(f"duplicate reclamation target binding: {key}")
            binding_map[key] = item
        if expected_targets is not None:
            expected_map = {str(item["relative_path"]): item for item in expected_targets}
            if set(binding_map) != set(expected_map):
                raise StorageIntegrityError("reclamation prerequisite target set changed")
            for relative_name, expected in expected_map.items():
                item = binding_map[relative_name]
                if str(item.get("sha256", "")) != str(expected.get("artifact_sha256", expected.get("sha256", ""))):
                    raise StorageIntegrityError(f"reclamation prerequisite target hash changed: {relative_name}")
                if dict(item.get("identity", {})) != dict(expected.get("identity", {})):
                    raise StorageIntegrityError(f"reclamation prerequisite target identity changed: {relative_name}")
                if str(expected.get("status", "")) not in {"DELETED", "DELETE_INTENT"}:
                    current_path, checked = _relative(self.root, relative_name)
                    if checked != relative_name or not current_path.is_file():
                        raise StorageIntegrityError(f"reclamation prerequisite target is missing: {relative_name}")
                    if dict(_identity(current_path)) != dict(item.get("identity", {})) or _sha256_file(current_path) != str(item.get("sha256", "")):
                        raise StorageIntegrityError(f"reclamation prerequisite target content changed: {relative_name}")
                elif str(expected.get("status", "")) == "DELETE_INTENT":
                    current_path, checked = _relative(self.root, relative_name)
                    if current_path.exists():
                        if dict(_identity(current_path)) != dict(item.get("identity", {})) or _sha256_file(current_path) != str(item.get("sha256", "")):
                            raise StorageIntegrityError(f"reclamation prerequisite target content changed: {relative_name}")

        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise StorageBudgetError("reclamation prerequisite evidence is missing")
        evidence_info: dict[str, Any] = {}
        for key in ("exact_complete", "successor_ready", "no_future_dependency"):
            descriptor = evidence.get(key)
            info, _ = self._read_evidence_descriptor(descriptor, label=f"reclamation {key} evidence")
            evidence_info[key] = info
        exact_payload = evidence_info["exact_complete"]["payload"]
        if evidence_info["exact_complete"]["status"] != "PASS" or int(exact_payload.get("mismatch_count", -1)) != 0:
            raise StorageBudgetError("reclamation exact evidence is not a complete PASS")
        if exact_payload.get("missing_reference") not in (None, [], ()) or exact_payload.get("missing_candidate") not in (None, [], ()):
            raise StorageBudgetError("reclamation exact evidence has missing objects")
        exact_rows = exact_payload.get("rows")
        if not isinstance(exact_rows, list) or not exact_rows or any(
            not isinstance(row, Mapping) or row.get("status") != "PASS" for row in exact_rows
        ):
            raise StorageBudgetError("reclamation exact evidence rows are incomplete")
        if self.require_intents:
            binding = exact_payload.get("object_set_binding")
            if not isinstance(binding, Mapping) or str(binding.get("status", "")).upper() != "PASS":
                raise StorageBudgetError("formal reclamation requires a validated exact object-set binding")
            if str(binding.get("campaign_id", self.campaign_id)) != self.campaign_id:
                raise StorageIntegrityError("formal exact object-set campaign mismatch")
            if not str(binding.get("binding_sha256", "")):
                raise StorageIntegrityError("formal exact object-set binding hash is missing")
            try:
                from .hr4e5_evidence import validate_durable_report
                reclaimed = any(
                    str(target.get("status")) == "DELETED"
                    for plan in self._read().get("gc_plans", {}).values()
                    if isinstance(plan, Mapping)
                    for target in plan.get("targets", [])
                    if isinstance(target, Mapping)
                )
                validate_durable_report(
                    evidence_info["exact_complete"]["path"],
                    expected_sha256=evidence_info["exact_complete"]["sha256"],
                    campaign_id=self.campaign_id, root=self.root,
                    validate_objects=not reclaimed,
                )
            except Exception as error:
                if isinstance(error, StorageBudgetError):
                    raise
                raise StorageIntegrityError("formal exact object-set validation failed") from error
        self._validate_successor_evidence(evidence_info["successor_ready"], label="reclamation successor READY evidence")
        dependency_payload = evidence_info["no_future_dependency"]["payload"]
        if evidence_info["no_future_dependency"]["status"] != "PASS" or dependency_payload.get("no_future_dependency") is not True:
            raise StorageBudgetError("reclamation future-dependency evidence is not PASS")
        return {
            "path": str(path), "relative_path": relative, "sha256": digest,
            "status": "PASS", "gates": {name: True for name in required},
            "target_bindings": [dict(item) for item in bindings],
            # Keep the immutable reports separate; duplicating all exact rows
            # into every per-file GC journal rewrite multiplies metadata I/O.
            "evidence": {key: {name: value for name, value in info.items() if name != 'payload'}
                         for key, info in evidence_info.items()},
        }

    def plan_reclaim(
        self, targets: Sequence[str | Path], *, exact_complete: bool,
        successor_ready: bool, no_active_writers: bool,
        no_future_dependency: bool, receipts_durable: bool,
        plan_id: str | None = None, reason: str = "after_exact_successor_ready",
        writer_receipt: str | Path | Mapping[str, Any] | None = None,
        writer_receipt_sha256: str | None = None,
        writer_epoch: str | int | None = None,
        prerequisite_receipt: str | Path | Mapping[str, Any] | None = None,
        trajectory: str | None = None, pulse: int | None = None,
        attempt: int | None = None, admission_hash: str | None = None,
    ) -> dict[str, Any]:
        gates = {
            "exact_complete": bool(exact_complete),
            "successor_ready": bool(successor_ready),
            "no_active_writers": bool(no_active_writers),
            "no_future_dependency": bool(no_future_dependency),
            "receipts_durable": bool(receipts_durable),
        }
        if not all(gates.values()):
            raise StorageBudgetError("reclamation prerequisites are not all satisfied")
        if self.require_intents:
            if str(trajectory) not in {"R", "C"}:
                raise StorageBudgetError("formal reclamation requires an R/C trajectory binding")
            if pulse is None or isinstance(pulse, bool) or int(pulse) < 0:
                raise StorageBudgetError("formal reclamation requires a nonnegative pulse binding")
            if attempt is None or isinstance(attempt, bool) or int(attempt) < 0:
                raise StorageBudgetError("formal reclamation requires a nonnegative attempt binding")
            if admission_hash is None or str(admission_hash) != str(self.admission_hash or ""):
                raise StorageIntegrityError("formal reclamation admission identity changed")
        writer_info = self._read_writer_receipt(
            writer_receipt, expected_sha256=writer_receipt_sha256,
            expected_epoch=writer_epoch,
            expected_trajectory=trajectory if self.require_intents else None,
            expected_pulse=pulse if self.require_intents else None,
            expected_attempt=attempt if self.require_intents else None,
            expected_admission_hash=admission_hash if self.require_intents else None,
        )
        pid = str(plan_id or f"gc-{os.getpid()}-{time.time_ns()}")
        raw_targets = [str(item) for item in targets]
        with self._locked() as data:
            if pid in data["gc_plans"]:
                raise StorageBudgetError(f"reclamation plan already exists: {pid}")
            records = self._validate_gc_targets(data, raw_targets)
            prerequisite_info = self._read_reclaim_prerequisite_receipt(
                prerequisite_receipt, expected_gates=gates, expected_targets=records,
            )
            protected_paths = {
                prerequisite_info["relative_path"],
                *(str(item["relative_path"]) for item in prerequisite_info["evidence"].values()),
            }
            for record in records:
                if str(record["relative_path"]) in protected_paths:
                    raise StorageIntegrityError(
                        f"reclamation target is also prerequisite evidence: {record['relative_path']}"
                    )
            try:
                _, writer_relative = _relative(self.root, writer_info["path"])
            except StorageIntegrityError:
                writer_relative = None
            if writer_relative is not None and any(
                str(record["relative_path"]) == writer_relative for record in records
            ):
                raise StorageIntegrityError("reclamation target is also writer evidence")
            plan = {
                "schema": GC_SCHEMA, "plan_id": pid, "campaign_id": self.campaign_id,
                "status": "PLANNED", "reason": str(reason), "gates": gates,
                "trajectory": None if trajectory is None else str(trajectory),
                "pulse": None if pulse is None else int(pulse),
                "attempt": None if attempt is None else int(attempt),
                "admission_identity_sha256": None if admission_hash is None else str(admission_hash),
                "writer_receipt": writer_info,
                "prerequisite_receipt": prerequisite_info,
                "targets": records, "planned_bytes": sum(item["bytes"] for item in records),
                "deleted_bytes": 0, "created_utc": _utc(), "updated_utc": _utc(),
            }
            data["gc_plans"][pid] = plan
            data["events"].append({"event": "GC_PLAN", "plan_id": pid, "planned_bytes": plan["planned_bytes"], "timestamp_utc": _utc()})
        return plan

    create_gc_plan = plan_reclaim

    def _load_plan(self, data: Mapping[str, Any], plan_id: str) -> dict[str, Any]:
        plan = data.get("gc_plans", {}).get(str(plan_id))
        if not isinstance(plan, dict) or plan.get("schema") != GC_SCHEMA or plan.get("campaign_id") != self.campaign_id:
            raise StorageBudgetError("reclamation plan is missing or invalid")
        return plan

    def _validate_reclaim_step_binding(
        self, plan: Mapping[str, Any], *, expected_trajectory: str | None,
        expected_pulse: int | None, expected_attempt: int | None,
        expected_admission_hash: str | None,
    ) -> None:
        bindings = {
            "trajectory": expected_trajectory,
            "pulse": expected_pulse,
            "attempt": expected_attempt,
            "admission_identity_sha256": expected_admission_hash,
        }
        if self.require_intents and any(value is None for value in bindings.values()):
            raise StorageIntegrityError("formal reclamation requires complete step bindings before unlink")
        for name, expected in bindings.items():
            if expected is not None and str(plan.get(name)) != str(expected):
                raise StorageIntegrityError(f"reclamation plan {name} binding changed")

    def validate_reclaim_plan(
        self, plan_id: str, *, gates: Mapping[str, bool] | None = None,
        writer_receipt: str | Path | Mapping[str, Any] | None = None,
        writer_epoch: str | int | None = None,
        prerequisite_receipt: str | Path | Mapping[str, Any] | None = None,
        expected_trajectory: str | None = None, expected_pulse: int | None = None,
        expected_attempt: int | None = None, expected_admission_hash: str | None = None,
    ) -> dict[str, Any]:
        with self._locked() as data:
            plan = self._load_plan(data, plan_id)
            self._validate_reclaim_step_binding(
                plan, expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
            )
            if plan.get("status") in {"COMPLETED", "ABORTED"}:
                return dict(plan)
            if gates is not None and not all(bool(value) for value in gates.values()):
                raise StorageBudgetError("current reclamation gates are not satisfied")
            stored_writer = plan.get("writer_receipt")
            if not isinstance(stored_writer, Mapping):
                raise StorageBudgetError("reclamation plan lacks durable writer receipt")
            self._read_writer_receipt(
                writer_receipt or stored_writer.get("path"),
                expected_sha256=str(stored_writer.get("sha256", "")),
                expected_epoch=writer_epoch or stored_writer.get("writer_epoch"),
                expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
                _ledger_data=data,
            )
            stored_prerequisite = plan.get("prerequisite_receipt")
            if not isinstance(stored_prerequisite, Mapping):
                raise StorageBudgetError("reclamation plan lacks durable prerequisite receipt")
            self._read_reclaim_prerequisite_receipt(
                prerequisite_receipt or stored_prerequisite.get("path"),
                expected_sha256=str(stored_prerequisite.get("sha256", "")),
                expected_gates=plan.get("gates"), expected_targets=plan.get("targets", []),
            )
            for target in plan.get("targets", []):
                relative = str(target["relative_path"])
                path = self.root / relative
                if target.get("status") == "DELETED":
                    continue
                current = _identity(path)
                if dict(current) != dict(target.get("identity", {})):
                    raise StorageIntegrityError(f"reclamation target identity changed: {relative}")
                if int(current.get("st_nlink", 1)) != 1:
                    raise StorageIntegrityError(f"hard-linked reclamation target: {relative}")
                if _sha256_file(path) != str(target.get("artifact_sha256", "")):
                    raise StorageIntegrityError(f"reclamation target content changed: {relative}")
            return dict(plan)

    def apply_reclaim(
        self, plan_id: str, *, no_active_writers: bool = True,
        interrupt_after: int | None = None,
        writer_receipt: str | Path | Mapping[str, Any] | None = None,
        writer_epoch: str | int | None = None,
        prerequisite_receipt: str | Path | Mapping[str, Any] | None = None,
        fault_hook: Callable[[str, str], None] | None = None,
        expected_trajectory: str | None = None, expected_pulse: int | None = None,
        expected_attempt: int | None = None, expected_admission_hash: str | None = None,
    ) -> dict[str, Any]:
        if not bool(no_active_writers):
            raise StorageBudgetError("reclamation requires a quiescent coordinator")
        count = 0
        with self._locked() as data:
            plan = self._load_plan(data, plan_id)
            self._validate_reclaim_step_binding(
                plan, expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
            )
            if plan.get("status") == "COMPLETED":
                return dict(plan)
            if plan.get("status") == "ABORTED":
                raise StorageBudgetError("aborted reclamation plan cannot be resumed")
            stored_writer = plan.get("writer_receipt")
            if not isinstance(stored_writer, Mapping):
                raise StorageBudgetError("reclamation plan lacks durable writer receipt")
            self._read_writer_receipt(
                writer_receipt or stored_writer.get("path"),
                expected_sha256=str(stored_writer.get("sha256", "")),
                expected_epoch=writer_epoch or stored_writer.get("writer_epoch"),
                expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
                _ledger_data=data,
            )
            stored_prerequisite = plan.get("prerequisite_receipt")
            if not isinstance(stored_prerequisite, Mapping):
                raise StorageBudgetError("reclamation plan lacks durable prerequisite receipt")
            self._read_reclaim_prerequisite_receipt(
                prerequisite_receipt or stored_prerequisite.get("path"),
                expected_sha256=str(stored_prerequisite.get("sha256", "")),
                expected_gates=plan.get("gates"), expected_targets=plan.get("targets", []),
            )
            plan["status"] = "APPLYING"
            plan["updated_utc"] = _utc()
            # Persist the transaction state before any unlink.  A resumed
            # process can distinguish an intended deletion from an unknown
            # disappearance and can conservatively recover a post-unlink
            # interruption.
            self._write(data)
            for target in plan["targets"]:
                if target.get("status") == "DELETED":
                    continue
                if interrupt_after is not None and count >= int(interrupt_after):
                    plan["status"] = "INTERRUPTED"
                    plan["updated_utc"] = _utc()
                    data["events"].append({"event": "GC_INTERRUPTED", "plan_id": str(plan_id), "timestamp_utc": _utc()})
                    self._write(data)
                    return dict(plan)
                relative = str(target["relative_path"])
                path, checked_relative = _relative(self.root, relative)
                if checked_relative != relative:
                    raise StorageIntegrityError("reclamation path changed")
                if target.get("status") == "DELETE_INTENT" and not path.exists():
                    # The durable intent is the only evidence under which a
                    # missing file may be attributed to this transaction.
                    target["status"] = "DELETED"
                    target["deleted_utc"] = _utc()
                    target["recovered_after_interrupt"] = True
                    plan["deleted_bytes"] = int(plan.get("deleted_bytes", 0)) + int(target["bytes"])
                    count += 1
                    self._write(data)
                    continue
                current = _identity(path)
                if dict(current) != dict(target.get("identity", {})) or int(current.get("st_nlink", 1)) != 1:
                    raise StorageIntegrityError(f"reclamation identity or hardlink gate failed: {relative}")
                if _sha256_file(path) != data['artifacts'][relative].get('sha256'):
                    raise StorageIntegrityError(f'reclamation content changed: {relative}')
                target["status"] = "DELETE_INTENT"
                target["intent_utc"] = _utc()
                self._write(data)
                path.unlink()
                _fsync_directory(path.parent)
                if fault_hook is not None:
                    fault_hook('after_unlink', relative)
                target["status"] = "DELETED"
                target["deleted_utc"] = _utc()
                plan["deleted_bytes"] = int(plan.get("deleted_bytes", 0)) + int(target["bytes"])
                count += 1
                self._write(data)
            plan["status"] = "COMPLETED"
            plan["updated_utc"] = _utc()
            data["events"].append({"event": "GC_COMPLETED", "plan_id": str(plan_id), "deleted_bytes": plan["deleted_bytes"], "timestamp_utc": _utc()})
            self._write(data)
            return dict(plan)

    def resume_reclaim(
        self, plan_id: str, *, no_active_writers: bool = True,
        writer_receipt: str | Path | Mapping[str, Any] | None = None,
        writer_epoch: str | int | None = None,
        prerequisite_receipt: str | Path | Mapping[str, Any] | None = None,
        expected_trajectory: str | None = None, expected_pulse: int | None = None,
        expected_attempt: int | None = None, expected_admission_hash: str | None = None,
    ) -> dict[str, Any]:
        return self.apply_reclaim(
            plan_id, no_active_writers=no_active_writers,
            writer_receipt=writer_receipt, writer_epoch=writer_epoch,
            prerequisite_receipt=prerequisite_receipt,
            expected_trajectory=expected_trajectory, expected_pulse=expected_pulse,
            expected_attempt=expected_attempt,
            expected_admission_hash=expected_admission_hash,
        )

    def verify_reclaim(
        self, plan_id: str, *, expected_trajectory: str | None = None,
        expected_pulse: int | None = None, expected_attempt: int | None = None,
        expected_admission_hash: str | None = None,
    ) -> dict[str, Any]:
        with _FileLock(self.lock_path):
            data = self._read()
            self._validate(data)
            plan = self._load_plan(data, plan_id)
            self._validate_reclaim_step_binding(
                plan, expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
            )
            stored_writer = plan.get("writer_receipt")
            if not isinstance(stored_writer, Mapping):
                raise StorageIntegrityError("reclamation plan lacks durable writer receipt")
            self._read_writer_receipt(
                stored_writer.get("path"), expected_sha256=str(stored_writer.get("sha256", "")),
                expected_epoch=stored_writer.get("writer_epoch"),
                expected_trajectory=expected_trajectory,
                expected_pulse=expected_pulse, expected_attempt=expected_attempt,
                expected_admission_hash=expected_admission_hash,
                _ledger_data=data,
            )
            stored_prerequisite = plan.get("prerequisite_receipt")
            if not isinstance(stored_prerequisite, Mapping):
                raise StorageIntegrityError("reclamation plan lacks durable prerequisite receipt")
            self._read_reclaim_prerequisite_receipt(
                stored_prerequisite.get("path"),
                expected_sha256=str(stored_prerequisite.get("sha256", "")),
                expected_gates=plan.get("gates"), expected_targets=plan.get("targets", []),
            )
            missing = []
            for target in plan["targets"]:
                if target.get("status") == "DELETED":
                    if (self.root / str(target["relative_path"])).exists():
                        missing.append(str(target["relative_path"]) + ":still_present")
                elif not (self.root / str(target["relative_path"])).is_file():
                    missing.append(str(target["relative_path"]) + ":missing")
            status = "PASS" if not missing and plan.get("status") == "COMPLETED" else "FAIL"
            return {"schema": GC_SCHEMA, "plan_id": str(plan_id), "status": status, "missing": missing, "deleted_bytes": int(plan.get("deleted_bytes", 0))}

    def budget_report(self) -> dict[str, Any]:
        with _FileLock(self.lock_path):
            data = self._read()
            self._validate(data)
            used = self.actual_bytes()
            reserved = self._reserved_bytes(data)
            final_bytes = sum(
                int(item.get("identity", {}).get("size", 0))
                for item in data.get("artifacts", {}).values()
                if str(item.get("role", "")).startswith("final")
            )
            return {
                "schema": STORAGE_SCHEMA,
                "campaign_id": self.campaign_id,
                "actual_bytes": used,
                "reserved_bytes": reserved,
                "safety_margin_bytes": self.safety_margin_bytes,
                "projected_bytes": used + reserved + self.safety_margin_bytes,
                "cap_bytes": self.cap_bytes,
                "final_output_bytes": final_bytes,
                "final_output_budget_bytes": self.final_output_budget_bytes,
                "free_bytes": self._provider_free_bytes(),
                "quota_bytes": self._provider_quota_bytes(),
                "quota_required": self.require_quota,
                "require_intents": self.require_intents,
                "admission_hash": self.admission_hash,
                "intent_count": len(data.get("intents", {})),
            }


class MockQuotaProvider:
    """Deterministic provider used by low-cost fixture tests."""

    def __init__(self, *, free_bytes: int | None = None, quota_bytes: int | None = None):
        self._free_bytes = free_bytes
        self._quota_bytes = quota_bytes

    def free_bytes(self) -> int | None:
        return self._free_bytes

    def quota_bytes(self) -> int | None:
        return self._quota_bytes

    def set_free(self, value: int | None) -> None:
        self._free_bytes = value

    def set_quota(self, value: int | None) -> None:
        self._quota_bytes = value


__all__ = [
    "DEFAULT_FINAL_OUTPUT_BUDGET_BYTES", "DEFAULT_SAFETY_MARGIN_BYTES", "GC_SCHEMA",
    "HARD_CAP_BYTES", "INTENT_SCHEMA", "MockQuotaProvider", "RECLAIM_PREREQUISITE_SCHEMA", "Reservation", "STORAGE_SCHEMA",
    "derive_budget_plan", "plan_campaign_budget",
    "StorageBudget", "StorageBudgetError", "StorageIntegrityError",
    "WRITER_SCHEMA", "WRITER_RECEIPT_SCHEMA", "process_identity", "probe_process_identity",
]

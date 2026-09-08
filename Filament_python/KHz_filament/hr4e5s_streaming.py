"""S2 durable CURRENT/POST/NEXT streaming lifecycle infrastructure.

This module deliberately owns orchestration only.  It never defines an HR-4
operator: hydro work invokes the frozen ``advance_hr4_single_screen`` path.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .device import to_cpu
from .hr4 import advance_hr4_single_screen
from .hr4e_timestep import sha256_array, sha256_file


SCHEMA = "khz_filament.hr4e5s.streaming.v1"
FIELDS = ("delta_n", "vx", "vy")
DEFAULT_QUEUE_DEPTH = 16
FROZEN_BLOCK_SIZE = 8


class StreamingLifecycleError(RuntimeError):
    """Base class for fail-closed streaming lifecycle failures."""


class DuplicateCommitError(StreamingLifecycleError):
    pass


class BackpressureError(StreamingLifecycleError):
    pass


class BarrierError(StreamingLifecycleError):
    pass


class _ManifestFileLock:
    """Small cross-platform advisory lock for one lifecycle manifest."""

    def __init__(self, path: Path, *, timeout_s: float = 10.0):
        self.path = path
        self.timeout_s = float(timeout_s)
        self._handle = None

    def __enter__(self) -> "_ManifestFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, os.SEEK_END)
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
                    raise StreamingLifecycleError("timed out acquiring lifecycle manifest lock") from error
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
        # Directory fsync is unavailable on some supported filesystems; file
        # fsync and atomic replacement remain mandatory.
        return


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, fields: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _validated_fields(fields)
    payload = {name: arrays[name] for name in FIELDS}
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _validated_fields(fields: Mapping[str, Any], *, shape=None, dtype=None) -> dict[str, np.ndarray]:
    if set(fields) != set(FIELDS):
        raise ValueError("streaming state requires delta_n, vx, and vy together")
    result = {name: np.asarray(fields[name]) for name in FIELDS}
    first = result["delta_n"]
    if first.ndim != 2 or first.dtype != np.dtype(np.float64) or not np.all(np.isfinite(first)):
        raise ValueError("streaming screen must be finite float64 [Ny, Nx]")
    if any(value.shape != first.shape or value.dtype != first.dtype or not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("streaming screen fields must have identical finite float64 layouts")
    if shape is not None and tuple(first.shape) != tuple(shape):
        raise ValueError("streaming screen shape does not match generation")
    if dtype is not None and first.dtype != np.dtype(dtype):
        raise ValueError("streaming screen dtype does not match generation")
    return {name: np.ascontiguousarray(value, dtype=np.float64) for name, value in result.items()}


def _field_hashes(fields: Mapping[str, Any]) -> dict[str, str]:
    return {name: sha256_array(fields[name]) for name in FIELDS}


def _content_hash(records: Sequence[Mapping[str, Any]]) -> str:
    source = json.dumps(list(records), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


class StreamingLifecycle:
    """One qualification generation with durable per-screen POST/NEXT artifacts.

    CURRENT is copied once into immutable per-screen artifacts during ``create``.
    Production callers may use memmapped/chunked inputs; no full state is ever
    copied into this coordinator's process memory.
    """

    def __init__(self, root: str | Path, manifest: Mapping[str, Any]):
        self.root = Path(root)
        self.manifest_path = self.root / "streaming_manifest.json"
        self.manifest = dict(manifest)
        self._authoritative_namespace = "CURRENT"
        self._authoritative_generation = ""
        self._validate_manifest()
        self._load_authoritative_pointer()

    @classmethod
    def create(
        cls, *, root: str | Path, current: Mapping[str, Any], screen_records: Sequence[Mapping[str, Any]],
        current_generation: str, dx_m: float, dy_m: float, queue_depth: int = DEFAULT_QUEUE_DEPTH, actor: str = "initializer",
    ) -> "StreamingLifecycle":
        base = Path(root)
        if base.exists():
            raise FileExistsError("streaming root must not already exist")
        arrays = {name: np.asarray(current[name]) for name in FIELDS}
        if set(current) != set(FIELDS) or any(value.ndim != 3 for value in arrays.values()):
            raise ValueError("CURRENT must supply three [K, Ny, Nx] fields")
        if any(value.shape != arrays["delta_n"].shape or value.dtype != np.dtype(np.float64) for value in arrays.values()):
            raise ValueError("CURRENT fields require matching float64 volume layouts")
        if any("ordinal" not in item or "screen_id" not in item or "z_m" not in item for item in screen_records):
            raise ValueError("screen records require authoritative ordinal, screen_id, and z_m")
        records = sorted((dict(item) for item in screen_records), key=lambda item: int(item["ordinal"]))
        count, shape = arrays["delta_n"].shape[0], arrays["delta_n"].shape[1:]
        if len(records) != count or [int(item["ordinal"]) for item in records] != list(range(count)):
            raise ValueError("screen records must be contiguous authoritative ordinals")
        if (
            any(not str(item["screen_id"]) or not np.isfinite(float(item["z_m"])) for item in records)
            or len({str(item["screen_id"]) for item in records}) != count
            or len({float(item["z_m"]) for item in records}) != count
        ):
            raise ValueError("screen records must have unique authoritative screen and z identities")
        if int(queue_depth) <= 0 or float(dx_m) <= 0.0 or float(dy_m) <= 0.0:
            raise ValueError("queue depth must be positive")
        base.mkdir(parents=True)
        for namespace in ("current", "post", "next"):
            (base / namespace).mkdir()
        current_records = []
        for item in records:
            ordinal = int(item["ordinal"])
            fields = {name: arrays[name][ordinal] for name in FIELDS}
            hashes = _field_hashes(fields)
            metadata = {
                "schema": SCHEMA, "namespace": "CURRENT", "ordinal": ordinal,
                "screen_id": str(item["screen_id"]), "generation": str(current_generation),
                "z_m": float(item["z_m"]), "field_sha256": hashes, "shape": list(shape), "dtype": "float64",
            }
            artifact = base / "current" / f"screen_{ordinal:06d}.npz"
            file_hash = _atomic_npz(artifact, fields, metadata)
            current_records.append({
                "ordinal": ordinal, "screen_id": str(item["screen_id"]), "z_m": float(item["z_m"]),
                "artifact": str(artifact.relative_to(base)), "file_sha256": file_hash, "field_sha256": hashes,
            })
        current_content_sha = _content_hash(current_records)
        lifecycle_records = []
        for entry in current_records:
            lifecycle_records.append({
                "ordinal": entry["ordinal"], "screen_id": entry["screen_id"], "z_m": entry["z_m"],
                "state": "CURRENT_READY", "retry_count": 0, "current": entry, "post": None, "next": None,
                "transitions": [{"state": "CURRENT_READY", "status": "PASS", "timestamp_utc": _utc(), "actor": actor, "ordinal": entry["ordinal"], "screen_id": entry["screen_id"], "current_generation": str(current_generation), "next_generation": str(current_generation) + ":next", "retry_count": 0, "current_content_sha256": current_content_sha, "source_file_sha256": entry["file_sha256"], "output_file_sha256": entry["file_sha256"]}],
            })
        manifest = {
            "schema": SCHEMA, "current_generation": str(current_generation), "current_content_sha256": current_content_sha,
            "next_generation": str(current_generation) + ":next",
            "shape": list(shape), "dtype": "float64", "dx_m": float(dx_m), "dy_m": float(dy_m), "expected_screen_count": count,
            "queue_depth": int(queue_depth), "block_size": FROZEN_BLOCK_SIZE, "queue": [], "records": lifecycle_records,
            "hydro_worker": {"qualification": "HR-4E-5P/P5", "screen_solver": "advance_hr4_single_screen", "block_size": FROZEN_BLOCK_SIZE},
            "barrier": None, "promotion": None, "rate_events": [], "telemetry_events": [], "created_utc": _utc(),
        }
        _atomic_json(base / "streaming_manifest.json", manifest)
        return cls(base, manifest)

    @classmethod
    def open(cls, root: str | Path) -> "StreamingLifecycle":
        base = Path(root)
        return cls(base, _read_json(base / "streaming_manifest.json"))

    def _validate_manifest(self) -> None:
        if self.manifest.get("schema") != SCHEMA or self.manifest.get("dtype") != "float64" or not self.manifest.get("current_generation") or not self.manifest.get("next_generation") or self.manifest["current_generation"] == self.manifest["next_generation"]:
            raise ValueError("streaming manifest schema or dtype is invalid")
        count = int(self.manifest.get("expected_screen_count", -1))
        records = self.manifest.get("records")
        if count <= 0 or not isinstance(records, list) or len(records) != count:
            raise ValueError("streaming manifest screen count is invalid")
        if [int(item.get("ordinal", -1)) for item in records] != list(range(count)):
            raise ValueError("streaming manifest order is invalid")
        if len({str(item.get("screen_id", "")) for item in records}) != count:
            raise ValueError("streaming manifest screen identity is duplicated")
        if (
            any(not str(item.get("screen_id", "")) or not np.isfinite(float(item.get("z_m", np.nan))) for item in records)
            or len({float(item["z_m"]) for item in records}) != count
            or any(not isinstance(item.get("current"), Mapping) for item in records)
        ):
            raise ValueError("streaming manifest authoritative screen identity is invalid")
        if int(self.manifest.get("queue_depth", 0)) <= 0 or int(self.manifest.get("block_size", 0)) != FROZEN_BLOCK_SIZE or float(self.manifest.get("dx_m", 0.0)) <= 0.0 or float(self.manifest.get("dy_m", 0.0)) <= 0.0:
            raise ValueError("streaming queue or frozen block contract is invalid")
        if _content_hash([item["current"] for item in records]) != self.manifest.get("current_content_sha256"):
            raise ValueError("CURRENT content provenance hash is invalid")

    def _save(self) -> None:
        self._validate_manifest()
        _atomic_json(self.manifest_path, self.manifest)

    def _telemetry_locked(self, event: str, *, actor: str, ordinal: int | None = None, block: Sequence[int] | None = None, **extra: Any) -> dict[str, Any]:
        """Append non-invasive scheduling telemetry while the manifest is owned.

        ``perf_counter`` is Linux's system-wide monotonic clock in the target
        environment.  Recording it neither reads device state nor inserts a
        CUDA synchronization.
        """
        entry = {
            "event": str(event), "monotonic_s": time.perf_counter(), "actor": str(actor),
            "worker_id": os.environ.get("HR4E5S_WORKER_ID", str(actor)),
            "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "queue_occupancy": len(self.manifest["queue"]), "queue_capacity": int(self.manifest["queue_depth"]),
            "producer_blocked": bool(extra.pop("producer_blocked", False)),
            **extra,
        }
        if ordinal is not None:
            record = self._record(int(ordinal))
            entry.update({"ordinal": int(ordinal), "screen_id": record["screen_id"], "z_m": record["z_m"]})
        if block is not None:
            entry["block"] = [int(value) for value in block]
        self.manifest.setdefault("telemetry_events", []).append(entry)
        return entry

    def record_telemetry(self, event: str, *, actor: str, ordinal: int | None = None, block: Sequence[int] | None = None, **extra: Any) -> dict[str, Any]:
        """Persist one monotonic event without touching scientific arrays."""
        with self._locked_manifest():
            entry = self._telemetry_locked(event, actor=actor, ordinal=ordinal, block=block, **extra)
            self._save()
            return entry

    @contextmanager
    def _locked_manifest(self):
        """Serialize every shared manifest read-modify-write transition."""
        with _ManifestFileLock(self.root / ".streaming_manifest.lock"):
            self.manifest = _read_json(self.manifest_path)
            self._validate_manifest()
            self._load_authoritative_pointer()
            yield

    def _load_authoritative_pointer(self) -> None:
        self._authoritative_namespace = "CURRENT"
        self._authoritative_generation = str(self.manifest["current_generation"])
        pointer_path = self.root / "authoritative_generation.json"
        if not pointer_path.is_file():
            return
        pointer = _read_json(pointer_path)
        if (
            pointer.get("schema") != SCHEMA
            or pointer.get("authoritative_namespace") != "NEXT"
            or pointer.get("source_current_generation") != self.manifest["current_generation"]
            or pointer.get("authoritative_generation") != self.manifest["next_generation"]
            or not isinstance(pointer.get("barrier"), Mapping)
            or pointer["barrier"].get("status") != "PASS"
            or not isinstance(self.manifest.get("barrier"), Mapping)
            or self.manifest["barrier"].get("status") != "PASS"
        ):
            raise StreamingLifecycleError("authoritative generation pointer is invalid")
        self._authoritative_namespace = "NEXT"
        self._authoritative_generation = str(pointer["authoritative_generation"])

    def _record(self, ordinal: int) -> dict[str, Any]:
        index = int(ordinal)
        if index < 0:
            raise IndexError("streaming screen ordinal is outside generation")
        try:
            return self.manifest["records"][index]
        except (IndexError, TypeError):
            raise IndexError("streaming screen ordinal is outside generation") from None

    def _transition(self, record: dict[str, Any], state: str, *, actor: str, **extra: Any) -> None:
        record["state"] = state
        record["transitions"].append({
            "state": state, "status": "PASS", "timestamp_utc": _utc(), "actor": actor,
            "ordinal": int(record["ordinal"]), "screen_id": str(record["screen_id"]),
            "current_generation": self.manifest["current_generation"],
            "next_generation": self.manifest["next_generation"], "retry_count": int(record["retry_count"]),
            "current_content_sha256": self.manifest["current_content_sha256"],
            "source_file_sha256": record["current"]["file_sha256"], **extra,
        })

    def _artifact_fields(self, entry: Mapping[str, Any], *, namespace: str) -> dict[str, np.ndarray]:
        path = self.root / str(entry["artifact"])
        if not path.is_file() or sha256_file(path) != str(entry["file_sha256"]):
            raise StreamingLifecycleError(f"{namespace} artifact is missing or hash-mismatched")
        with np.load(path, allow_pickle=False) as loaded:
            fields = {name: np.asarray(loaded[name]) for name in FIELDS}
            metadata = json.loads(str(loaded["metadata_json"].item()))
        fields = _validated_fields(fields, shape=self.manifest["shape"], dtype=np.float64)
        if metadata.get("schema") != SCHEMA or metadata.get("namespace") != namespace:
            raise StreamingLifecycleError(f"{namespace} artifact metadata is invalid")
        hashes = _field_hashes(fields)
        if dict(entry.get("field_sha256", {})) != hashes or dict(metadata.get("field_sha256", {})) != hashes:
            raise StreamingLifecycleError(f"{namespace} canonical hashes are invalid")
        if metadata.get("shape") != list(self.manifest["shape"]) or metadata.get("dtype") != "float64":
            raise StreamingLifecycleError(f"{namespace} artifact layout metadata is invalid")
        return fields

    def _artifact_metadata(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        path = self.root / str(entry["artifact"])
        with np.load(path, allow_pickle=False) as loaded:
            return dict(json.loads(str(loaded["metadata_json"].item())))

    def _assert_no_staged_or_orphaned_artifacts(self) -> None:
        temporary = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*.tmp"))
        if temporary:
            raise StreamingLifecycleError("staged artifact remains after interrupted commit: " + ",".join(temporary))
        expected = {Path(str(record["current"]["artifact"])).as_posix() for record in self.manifest["records"]}
        for record in self.manifest["records"]:
            for name in ("post", "next"):
                if record[name] is not None:
                    expected.add(Path(str(record[name]["artifact"])).as_posix())
        actual = {
            path.relative_to(self.root).as_posix()
            for namespace in ("current", "post", "next")
            for path in (self.root / namespace).glob("*.npz")
        }
        unexpected = sorted(actual - expected)
        if unexpected:
            raise StreamingLifecycleError("orphaned committed artifact is not represented in the manifest: " + ",".join(unexpected))

    def _validate_record_provenance(self, record: Mapping[str, Any], *, require_post: bool, require_next: bool) -> None:
        expected_ordinal = int(record["ordinal"])
        expected_screen_id = str(record["screen_id"])
        expected_z_m = float(record["z_m"])

        self._artifact_fields(record["current"], namespace="CURRENT")
        current_metadata = self._artifact_metadata(record["current"])
        if (
            int(current_metadata.get("ordinal", -1)) != expected_ordinal
            or str(current_metadata.get("screen_id", "")) != expected_screen_id
            or float(current_metadata.get("z_m", np.nan)) != expected_z_m
        ):
            raise StreamingLifecycleError("CURRENT identity is invalid")
        if (
            current_metadata.get("generation") != self.manifest["current_generation"]
            or dict(current_metadata.get("field_sha256", {})) != dict(record["current"]["field_sha256"])
        ):
            raise StreamingLifecycleError("CURRENT provenance is invalid")

        post = record.get("post")
        if post is None:
            if require_post:
                raise StreamingLifecycleError("POST artifact is missing")
            return
        self._artifact_fields(post, namespace="POST")
        post_metadata = self._artifact_metadata(post)
        if (
            int(post_metadata.get("ordinal", -1)) != expected_ordinal
            or str(post_metadata.get("screen_id", "")) != expected_screen_id
            or float(post_metadata.get("z_m", np.nan)) != expected_z_m
        ):
            raise StreamingLifecycleError("POST identity is invalid")
        if (
            post_metadata.get("current_generation") != self.manifest["current_generation"]
            or post_metadata.get("current_content_sha256") != self.manifest["current_content_sha256"]
            or dict(post_metadata.get("current_field_sha256", {})) != dict(record["current"]["field_sha256"])
            or not post_metadata.get("hr3a_authoritative")
            or not post_metadata.get("hr3b_authoritative")
        ):
            raise StreamingLifecycleError("POST provenance is invalid")

        next_entry = record.get("next")
        if next_entry is None:
            if require_next:
                raise StreamingLifecycleError("NEXT artifact is missing")
            return
        self._artifact_fields(next_entry, namespace="NEXT")
        next_metadata = self._artifact_metadata(next_entry)
        if (
            int(next_metadata.get("ordinal", -1)) != expected_ordinal
            or str(next_metadata.get("screen_id", "")) != expected_screen_id
            or float(next_metadata.get("z_m", np.nan)) != expected_z_m
        ):
            raise StreamingLifecycleError("NEXT identity is invalid")
        if (
            next_metadata.get("current_generation") != self.manifest["current_generation"]
            or next_metadata.get("current_content_sha256") != self.manifest["current_content_sha256"]
            or next_metadata.get("next_generation") != self.manifest["next_generation"]
            or next_metadata.get("post_file_sha256") != post["file_sha256"]
            or dict(next_metadata.get("post_field_sha256", {})) != dict(post["field_sha256"])
        ):
            raise StreamingLifecycleError("NEXT provenance is invalid")

    def current_fields(self, ordinal: int) -> dict[str, np.ndarray]:
        record = self._record(ordinal)
        if self._authoritative_namespace == "NEXT":
            if record["next"] is None:
                raise StreamingLifecycleError("authoritative NEXT screen is missing")
            return self._artifact_fields(record["next"], namespace="NEXT")
        return self._artifact_fields(record["current"], namespace="CURRENT")

    def begin_optical(self, ordinal: int, *, actor: str = "optical") -> None:
        with self._locked_manifest():
            record = self._record(ordinal)
            if record["state"] != "CURRENT_READY":
                raise StreamingLifecycleError("screen is not ready for optical production")
            self._transition(record, "OPTICAL_IN_PROGRESS", actor=actor)
            self._save()

    def deposition_finalized(self, ordinal: int, *, actor: str = "optical", optical_finalized_s: float | None = None) -> None:
        with self._locked_manifest():
            record = self._record(ordinal)
            if record["state"] not in ("CURRENT_READY", "OPTICAL_IN_PROGRESS"):
                raise StreamingLifecycleError("deposition finalization has invalid predecessor state")
            if record["state"] == "CURRENT_READY":
                self._transition(record, "OPTICAL_IN_PROGRESS", actor=actor)
            event = time.perf_counter() if optical_finalized_s is None else float(optical_finalized_s)
            self._transition(record, "DEPOSITION_FINALIZED", actor=actor, optical_finalized_s=event)
            self._save()

    def commit_post(self, ordinal: int, fields: Mapping[str, Any], *, actor: str = "optical", hr3a_authoritative: bool = True, hr3b_authoritative: bool = True) -> dict[str, Any]:
        with self._locked_manifest():
            record = self._record(ordinal)
            if record["state"] != "DEPOSITION_FINALIZED":
                raise StreamingLifecycleError("POST commit requires DEPOSITION_FINALIZED")
            if not hr3a_authoritative or not hr3b_authoritative:
                raise StreamingLifecycleError("POST commit requires authoritative HR-3A and HR-3B")
            if record["post"] is not None:
                raise DuplicateCommitError("duplicate POST commit")
            payload = _validated_fields(fields, shape=self.manifest["shape"], dtype=np.float64)
            hashes = _field_hashes(payload)
            metadata = {
                "schema": SCHEMA, "namespace": "POST", "ordinal": int(ordinal), "screen_id": record["screen_id"], "z_m": record["z_m"],
                "current_generation": self.manifest["current_generation"], "current_content_sha256": self.manifest["current_content_sha256"],
                "current_field_sha256": record["current"]["field_sha256"], "field_sha256": hashes,
                "shape": self.manifest["shape"], "dtype": "float64", "hr3a_authoritative": True, "hr3b_authoritative": True,
            }
            path = self.root / "post" / f"screen_{int(ordinal):06d}.npz"
            file_hash = _atomic_npz(path, payload, metadata)
            entry = {"artifact": str(path.relative_to(self.root)), "file_sha256": file_hash, "field_sha256": hashes, "metadata": metadata}
            self._artifact_fields(entry, namespace="POST")
            record["post"] = entry
            committed = time.perf_counter()
            optical_finalized = next((event.get("optical_finalized_s") for event in reversed(record["transitions"]) if event["state"] == "DEPOSITION_FINALIZED"), None)
            self.manifest["rate_events"].append({"ordinal": int(ordinal), "screen_id": record["screen_id"], "z_m": record["z_m"], "event": "POST_COMMITTED", "time_s": committed, "optical_finalized_s": optical_finalized, "queue_depth": len(self.manifest["queue"]), "backpressure": False})
            self._telemetry_locked("POST_COMMITTED", actor=actor, ordinal=ordinal)
            self._transition(record, "POST_COMMITTED", actor=actor, source_file_sha256=record["current"]["file_sha256"], output_file_sha256=file_hash, post_committed_s=committed)
            self._save()
            return entry

    def commit_post_from_delta_n(self, ordinal: int, post_delta_n: Any, **kwargs: Any) -> dict[str, Any]:
        current = self.current_fields(ordinal)
        payload = {"delta_n": np.asarray(post_delta_n, dtype=np.float64).copy(), "vx": current["vx"].copy(), "vy": current["vy"].copy()}
        return self.commit_post(ordinal, payload, **kwargs)

    def enqueue_post(
        self, ordinal: int, *, actor: str = "optical", wait_for_capacity: bool = False,
        timeout_s: float | None = 0.0, poll_s: float = 0.01,
    ) -> None:
        """Publish a POST reference, optionally pausing the producer for capacity."""
        if timeout_s is not None and float(timeout_s) < 0.0:
            raise ValueError("backpressure timeout must be non-negative or None")
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        experienced_backpressure = False
        while True:
            queue_full = False
            with self._locked_manifest():
                record = self._record(ordinal)
                if int(ordinal) in self.manifest["queue"]:
                    raise DuplicateCommitError("duplicate queue entry")
                if record["state"] != "POST_COMMITTED" or record["post"] is None:
                    raise StreamingLifecycleError("only committed POST may enter queue")
                if len(self.manifest["queue"]) >= int(self.manifest["queue_depth"]):
                    queue_full = True
                    if not experienced_backpressure:
                        self.manifest["rate_events"].append({"ordinal": int(ordinal), "screen_id": record["screen_id"], "z_m": record["z_m"], "event": "BACKPRESSURE", "time_s": time.perf_counter(), "queue_depth": len(self.manifest["queue"]), "backpressure": True})
                        self._telemetry_locked("PRODUCER_BACKPRESSURE_BEGIN", actor=actor, ordinal=ordinal, producer_blocked=True)
                        self._save()
                else:
                    self.manifest["queue"].append(int(ordinal))
                    self.manifest["queue"].sort()
                    now = time.perf_counter()
                    self.manifest["rate_events"].append({"ordinal": int(ordinal), "screen_id": record["screen_id"], "z_m": record["z_m"], "event": "ENQUEUED", "time_s": now, "queue_depth": len(self.manifest["queue"]), "backpressure": experienced_backpressure})
                    self._telemetry_locked("ENQUEUE", actor=actor, ordinal=ordinal, producer_blocked=experienced_backpressure)
                    if experienced_backpressure:
                        self._telemetry_locked("PRODUCER_BACKPRESSURE_END", actor=actor, ordinal=ordinal)
                    self._transition(record, "HYDRO_QUEUED", actor=actor, source_file_sha256=record["post"]["file_sha256"], enqueue_s=now, queue_depth=len(self.manifest["queue"]), backpressure=experienced_backpressure)
                    self._save()
                    return
            if not queue_full or not wait_for_capacity:
                raise BackpressureError("bounded POST queue is full")
            experienced_backpressure = True
            if deadline is not None and time.monotonic() >= deadline:
                raise BackpressureError("timed out waiting for bounded POST queue capacity")
            time.sleep(max(0.001, float(poll_s)))

    def claim_block(self, *, actor: str = "hydro") -> list[int]:
        with self._locked_manifest():
            ordered = sorted(int(value) for value in self.manifest["queue"])
            if len(ordered) < FROZEN_BLOCK_SIZE:
                return []
            block = ordered[:FROZEN_BLOCK_SIZE]
            for ordinal in block:
                record = self._record(ordinal)
                if record["state"] != "HYDRO_QUEUED":
                    raise StreamingLifecycleError("queue lifecycle state is inconsistent")
                self._transition(record, "HYDRO_RUNNING", actor=actor, source_file_sha256=record["post"]["file_sha256"])
            self.manifest["queue"] = [value for value in self.manifest["queue"] if int(value) not in set(block)]
            self._telemetry_locked("HYDRO_CLAIM", actor=actor, block=block)
            self._save()
            return block

    def commit_next(self, ordinal: int, fields: Mapping[str, Any], *, actor: str = "hydro") -> dict[str, Any]:
        with self._locked_manifest():
            record = self._record(ordinal)
            if record["state"] != "HYDRO_RUNNING" or record["post"] is None:
                raise StreamingLifecycleError("NEXT commit requires an owned running POST")
            if record["next"] is not None:
                raise DuplicateCommitError("duplicate NEXT commit")
            payload = _validated_fields(fields, shape=self.manifest["shape"], dtype=np.float64)
            hashes = _field_hashes(payload)
            metadata = {
                "schema": SCHEMA, "namespace": "NEXT", "ordinal": int(ordinal), "screen_id": record["screen_id"], "z_m": record["z_m"],
                "current_generation": self.manifest["current_generation"], "current_content_sha256": self.manifest["current_content_sha256"],
                "next_generation": self.manifest["next_generation"],
                "post_file_sha256": record["post"]["file_sha256"], "post_field_sha256": record["post"]["field_sha256"],
                "field_sha256": hashes, "shape": self.manifest["shape"], "dtype": "float64",
            }
            path = self.root / "next" / f"screen_{int(ordinal):06d}.npz"
            file_hash = _atomic_npz(path, payload, metadata)
            entry = {"artifact": str(path.relative_to(self.root)), "file_sha256": file_hash, "field_sha256": hashes, "metadata": metadata}
            self._artifact_fields(entry, namespace="NEXT")
            record["next"] = entry
            self._transition(record, "NEXT_COMMITTED", actor=actor, source_file_sha256=record["post"]["file_sha256"], output_file_sha256=file_hash)
            self._save()
            return entry

    def run_one_hydro_block(self, *, dt_hydro: float, n_hydro_steps: int, chi: float, nu: float, n0: float, gravity_x: float = 0.0, gravity_y: float = -9.81, cfl_limit: float = 1.0, actor: str = "hydro") -> list[int]:
        block = self.claim_block(actor=actor)
        if block:
            self.record_telemetry("HYDRO_BLOCK_START", actor=actor, block=block)
        for ordinal in block:
            incoming = self._artifact_fields(self._record(ordinal)["post"], namespace="POST")
            self.record_telemetry("HYDRO_SCREEN_START", actor=actor, ordinal=ordinal, block=block)
            result = advance_hr4_single_screen(incoming["delta_n"], incoming["vx"], incoming["vy"], dx=float(self.manifest["dx_m"]), dy=float(self.manifest["dy_m"]), dt_hydro=float(dt_hydro), chi=float(chi), nu=float(nu), n0=float(n0), gravity_x=float(gravity_x), gravity_y=float(gravity_y), cfl_limit=float(cfl_limit), n_steps=int(n_hydro_steps), require_stable=True)
            # ``advance_hr4_single_screen`` returns the active backend's arrays.
            # NEXT is a disk-backed, NumPy-float64 artifact, so this is the
            # explicit device-to-host persistence boundary.
            self.commit_next(ordinal, {name: np.asarray(to_cpu(result[name]), dtype=np.float64) for name in FIELDS}, actor=actor)
            self.record_telemetry("HYDRO_SCREEN_END", actor=actor, ordinal=ordinal, block=block)
        if block:
            self.record_telemetry("HYDRO_BLOCK_END", actor=actor, block=block)
        return block

    def reconstruct_queue(self, *, actor: str = "restart") -> list[int]:
        with self._locked_manifest():
            self._assert_no_staged_or_orphaned_artifacts()
            if self._authoritative_namespace == "NEXT":
                # A promoted generation is already terminal; it must not be
                # reconstructed as unfinished work for the prior pulse.
                for record in self.manifest["records"]:
                    self._validate_record_provenance(record, require_post=True, require_next=True)
                return []
            allowed_states = {"CURRENT_READY", "OPTICAL_IN_PROGRESS", "DEPOSITION_FINALIZED", "POST_COMMITTED", "HYDRO_QUEUED", "HYDRO_RUNNING", "NEXT_COMMITTED", "BARRIER_VALIDATED"}
            if any(record.get("state") not in allowed_states for record in self.manifest["records"]):
                raise StreamingLifecycleError("restart found an unknown lifecycle state")
            self.manifest["queue"] = []
            candidates = []
            for record in self.manifest["records"]:
                if record["next"] is not None:
                    self._validate_record_provenance(record, require_post=True, require_next=True)
                    if record["state"] not in ("NEXT_COMMITTED", "BARRIER_VALIDATED"):
                        raise StreamingLifecycleError("NEXT artifact has invalid lifecycle state")
                    continue
                if record["post"] is None:
                    if record["state"] in ("HYDRO_QUEUED", "HYDRO_RUNNING", "NEXT_COMMITTED", "BARRIER_VALIDATED"):
                        raise StreamingLifecycleError("restart found a lifecycle state without its required artifact")
                    continue
                self._validate_record_provenance(record, require_post=True, require_next=False)
                if record["state"] not in ("POST_COMMITTED", "HYDRO_QUEUED", "HYDRO_RUNNING"):
                    raise StreamingLifecycleError("POST artifact has invalid lifecycle state")
                candidates.append(int(record["ordinal"]))
                record["retry_count"] = int(record["retry_count"]) + int(record["state"] == "HYDRO_RUNNING")
                self._transition(record, "POST_COMMITTED", actor=actor, source_file_sha256=record["post"]["file_sha256"], reconstructed=True)
            for ordinal in sorted(candidates)[:int(self.manifest["queue_depth"])]:
                record = self._record(ordinal)
                self.manifest["queue"].append(ordinal)
                self._transition(record, "HYDRO_QUEUED", actor=actor, source_file_sha256=record["post"]["file_sha256"], reconstructed=True)
            self._save()
            return list(self.manifest["queue"])

    def validate_barrier(self, *, actor: str = "barrier") -> dict[str, Any]:
        self.record_telemetry("BARRIER_START", actor=actor)
        with self._locked_manifest():
            failures = []
            seen_next_identities: set[tuple[int, str, float]] = set()
            try:
                self._assert_no_staged_or_orphaned_artifacts()
            except Exception as error:
                failures.append(f"artifact_inventory_{type(error).__name__}:{error}")
            if self.manifest["queue"]:
                failures.append("unresolved_queue")
            for record in self.manifest["records"]:
                if record["state"] != "NEXT_COMMITTED" or record["next"] is None or record["post"] is None:
                    failures.append(f"screen_{record['ordinal']}_not_next_committed")
                    continue
                try:
                    self._validate_record_provenance(record, require_post=True, require_next=True)
                    next_metadata = self._artifact_metadata(record["next"])
                    identity = (int(next_metadata["ordinal"]), str(next_metadata["screen_id"]), float(next_metadata["z_m"]))
                    if identity in seen_next_identities:
                        raise StreamingLifecycleError("duplicate NEXT screen identity")
                    seen_next_identities.add(identity)
                except Exception as error:
                    failures.append(f"screen_{record['ordinal']}_{type(error).__name__}:{error}")
            result = {"schema": SCHEMA, "status": "PASS" if not failures else "FAIL", "failures": failures, "expected_screen_count": self.manifest["expected_screen_count"], "validated_utc": _utc(), "current_generation": self.manifest["current_generation"], "next_generation": self.manifest["next_generation"], "current_content_sha256": self.manifest["current_content_sha256"]}
            self.manifest["barrier"] = result
            if not failures:
                for record in self.manifest["records"]:
                    self._transition(record, "BARRIER_VALIDATED", actor=actor, source_file_sha256=record["next"]["file_sha256"], output_file_sha256=record["next"]["file_sha256"])
            self._save()
            if failures:
                self._telemetry_locked("BARRIER_FAIL", actor=actor)
                self._save()
                raise BarrierError("streaming barrier rejected generation: " + ",".join(failures))
            self._telemetry_locked("BARRIER_PASS", actor=actor)
            self._save()
            return result

    def promote_next_to_current(self, *, actor: str = "barrier") -> dict[str, Any]:
        with self._locked_manifest():
            barrier = self.manifest.get("barrier")
            if not isinstance(barrier, Mapping) or barrier.get("status") != "PASS":
                raise BarrierError("NEXT cannot become CURRENT before barrier PASS")
            pointer = {"schema": SCHEMA, "authoritative_namespace": "NEXT", "source_current_generation": self.manifest["current_generation"], "authoritative_generation": self.manifest["next_generation"], "barrier": dict(barrier), "promoted_utc": _utc(), "actor": actor}
            _atomic_json(self.root / "authoritative_generation.json", pointer)
            self.manifest["promotion"] = pointer
            self._telemetry_locked("PROMOTION", actor=actor)
            self._save()
            self._authoritative_namespace = "NEXT"
            self._authoritative_generation = str(pointer["authoritative_generation"])
            return pointer

    def rate_metrics(self) -> dict[str, Any]:
        times = [float(item["time_s"]) for item in self.manifest["rate_events"] if item["event"] == "POST_COMMITTED"]
        intervals = [right - left for left, right in zip(times, times[1:])]
        def rate(window: list[float]) -> float:
            return 0.0 if len(window) < 2 or window[-1] <= window[0] else (len(window) - 1) / (window[-1] - window[0])

        window_size = min(FROZEN_BLOCK_SIZE, len(times))
        backpressure_events = [item for item in self.manifest["rate_events"] if item["event"] == "BACKPRESSURE"]
        return {
            "post_committed_count": len(times),
            "inter_commit_intervals_s": intervals,
            "post_screens_per_s": rate(times),
            "startup_rate_screens_per_s": rate(times[:window_size]),
            "steady_state_rate_screens_per_s": rate(times[-window_size:]),
            "tail_behavior": "barrier_required_after_last_post_commit",
            "backpressure_event_count": len(backpressure_events),
            "startup_event_recorded": bool(times),
        }


def make_post_commit_hook(
    lifecycle: StreamingLifecycle, *, actor: str = "optical", backpressure_timeout_s: float | None = 60.0,
) -> Callable[..., None]:
    """Return the thin callback installed after successful HR-3B update.

    The propagation function supplies only interval metadata and the already
    computed POST delta-n.  The hook never receives or mutates the optical E
    field, q maps, or pulse ledger.
    """
    def hook(*, interval, state_after, hr3a_authoritative: bool, hr3b_authoritative: bool) -> None:
        ordinal = int(interval.index)
        lifecycle.deposition_finalized(ordinal, actor=actor)
        lifecycle.commit_post_from_delta_n(ordinal, state_after, actor=actor, hr3a_authoritative=hr3a_authoritative, hr3b_authoritative=hr3b_authoritative)
        lifecycle.enqueue_post(ordinal, actor=actor, wait_for_capacity=True, timeout_s=backpressure_timeout_s)
    return hook


__all__ = ["DEFAULT_QUEUE_DEPTH", "FIELDS", "FROZEN_BLOCK_SIZE", "BackpressureError", "BarrierError", "DuplicateCommitError", "StreamingLifecycle", "StreamingLifecycleError", "make_post_commit_hook"]

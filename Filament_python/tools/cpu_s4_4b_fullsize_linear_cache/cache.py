"""Full-size capacity adapter for the accepted CPU-S4-4A factor cache.

The numerical key, factor builder, hit path, complete-table publication, and
LRU policy remain implemented by ``cpu_s4_4a_linear_cache.cache``.  This
module only adds the full-size support domain, conservative resource
accounting, and a post-publication audit of objects owned by this cache.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_4a_linear_cache.cache import (
    CacheContractError,
    LinearFactorCache,
    _Entry,
)

from . import contract


CACHE_SCHEMA = "filament.cpu_s4_4b.fullsize_linear_cache.v1"
FULLSIZE_POLICY_ID = "CPU_S4_4B_FULLSIZE_LINEAR_FACTOR_CACHE"


def _module_source_identity(path: Path) -> dict[str, Any]:
    source = path.resolve(strict=True)
    raw = source.read_bytes()
    repo = Path(__file__).resolve().parents[3]
    relative = source.relative_to(repo).as_posix()
    git_blob = pv._git(repo, "rev-parse", "HEAD:" + relative)
    canonical = pv.canonical_lf_sha256(raw)
    if canonical != pv.canonical_lf_sha256(pv._read_git_blob(repo, git_blob)):
        raise CacheContractError("NEW_POLICY_SOURCE_DIFFERS_FROM_COMMITTED_BLOB")
    return {
        "path": relative,
        "canonical_lf_sha256": canonical,
        "hash_scope": pv.TRACKED_HASH_SCOPE,
        "classification": "tracked_text",
        "git_blob_oid": git_blob,
    }


def source_identity() -> dict[str, Any]:
    """Return the new policy's independent source identity."""

    root = Path(__file__).resolve().parent
    records = {
        name: _module_source_identity(root / name)
        for name in ("contract.py", "cache.py")
    }
    composite = (
        f"{FULLSIZE_POLICY_ID}\n"
        + "".join(
            f"{name}:{records[name]['canonical_lf_sha256']}\n"
            for name in sorted(records)
        )
    ).encode("ascii")
    return {
        "schema": "filament.cpu_s4_4b.fullsize_linear_cache.policy_sources.v1",
        "provenance_implementation": "tools.hpc_ops.provenance_v2",
        "files": records,
        "actual_loaded_paths": {name: str(root / name) for name in records},
        "composite_canonical_lf_sha256": pv.raw_sha256(composite),
    }


_SOURCE_IDENTITY = source_identity()
FULLSIZE_POLICY_SOURCE_VERSION = ":".join(
    (
        FULLSIZE_POLICY_ID,
        str(_SOURCE_IDENTITY["files"]["contract.py"]["canonical_lf_sha256"]),
        str(_SOURCE_IDENTITY["files"]["cache.py"]["canonical_lf_sha256"]),
    )
)


def policy_source_version(source_version: Any) -> str:
    """Append this policy's source identity to the original key source."""

    if not isinstance(source_version, str) or not source_version:
        raise CacheContractError("algorithm/source version is required")
    source = source_version
    token = FULLSIZE_POLICY_SOURCE_VERSION
    if token in source:
        return source
    return f"{source}:{token}"


class FullsizeLinearFactorCache(LinearFactorCache):
    """A bounded full-size cache that reuses the accepted 4A lifecycle."""

    def __init__(
        self,
        *,
        owner_identity: str,
        algorithm_source_identity: str,
        max_tables: int = contract.FULLSIZE_MAX_TABLES,
        max_bytes: int | None = None,
        resource_scope: str = contract.FULLSIZE_SCOPE,
        configured_shape: tuple[int, int, int] | None = None,
    ) -> None:
        scope = str(resource_scope)
        if scope not in contract.VALID_SCOPES:
            raise CacheContractError(f"unknown CPU-S4-4B resource scope: {scope!r}")
        if type(max_tables) is not int or int(max_tables) != int(contract.FULLSIZE_MAX_TABLES):
            raise CacheContractError("CPU-S4-4B max_tables policy was modified")
        expected_cap = contract.SMALLGRID_DEFAULT_MAX_BYTES if scope == contract.LOCAL_ENGINEERING_SCOPE else contract.FULLSIZE_MAX_BYTES
        if max_bytes is None:
            max_bytes = expected_cap
        if type(max_bytes) is not int or int(max_bytes) != expected_cap:
            raise CacheContractError("CPU-S4-4B max_bytes policy was modified")
        algorithm_source = str(algorithm_source_identity)
        if FULLSIZE_POLICY_SOURCE_VERSION not in algorithm_source:
            algorithm_source = f"{algorithm_source}:{FULLSIZE_POLICY_SOURCE_VERSION}"
        super().__init__(
            owner_identity=str(owner_identity),
            algorithm_source_identity=algorithm_source,
            max_tables=int(max_tables),
            max_bytes=int(max_bytes),
        )
        self.resource_scope = scope
        try:
            self.configured_shape = (
                contract.validate_shape(configured_shape, scope=scope)
                if configured_shape is not None
                else None
            )
        except ValueError as exc:
            raise CacheContractError(str(exc)) from exc
        self.policy_schema = contract.POLICY_SCHEMA
        self.fullsize_cache_schema = CACHE_SCHEMA
        self._global_reserve_bytes = int(contract.GLOBAL_METADATA_EVENT_RESERVE_BYTES)
        # Fixed by production construction; a private field permits a focused
        # engineering test to lower it and verify an explicit fail-closed
        # underestimated-reserve result.
        self._table_extra_reserve_bytes = int(contract.TABLE_EXTRA_RESERVE_BYTES)
        self._owned_audit_samples = 0
        self._owned_audit_max_bytes = 0
        self._last_owned_audit: dict[str, Any] = {
            "status": "NOT_RUN",
            "owned_bytes": 0,
            "reserved_bytes": int(self._global_reserve_bytes),
            "sample_count": 0,
            "max_owned_bytes": 0,
            "scope": self.resource_scope,
            "capacity_scope": "persistent_reserved_not_RSS_or_cgroup",
            "workspace_scope": "cache-owned-complete-entries-only",
        }
        self._stats.update(
            {
                "global_metadata_event_reserve_bytes": int(self._global_reserve_bytes),
                "key_guard_bytes": 0,
                "table_reserve_bytes": 0,
                "table_extra_reserve_bytes": int(self._table_extra_reserve_bytes),
                "owned_audit_samples": 0,
                "owned_audit_last_bytes": 0,
                "owned_audit_max_bytes": 0,
                "owned_audit_rejections": 0,
            }
        )
        self._sync_counts()

    def _assert_policy(self) -> None:
        if self.resource_scope not in contract.VALID_SCOPES:
            raise CacheContractError("unknown CPU-S4-4B resource scope")
        if type(self.max_tables) is not int or int(self.max_tables) != int(contract.FULLSIZE_MAX_TABLES):
            raise CacheContractError("CPU-S4-4B max_tables policy was modified")
        expected_cap = contract.SMALLGRID_DEFAULT_MAX_BYTES if self.resource_scope == contract.LOCAL_ENGINEERING_SCOPE else contract.FULLSIZE_MAX_BYTES
        if type(self.max_bytes) is not int or int(self.max_bytes) != expected_cap:
            raise CacheContractError("CPU-S4-4B max_bytes policy was modified")
        if type(self._global_reserve_bytes) is not int or self._global_reserve_bytes != int(
            contract.GLOBAL_METADATA_EVENT_RESERVE_BYTES
        ):
            raise CacheContractError("CPU-S4-4B global reserve policy was modified")
        if type(self._table_extra_reserve_bytes) is not int or self._table_extra_reserve_bytes < 0:
            raise CacheContractError("CPU-S4-4B table reserve is invalid")
        if self.configured_shape is not None:
            try:
                contract.validate_shape(self.configured_shape, scope=self.resource_scope)
            except ValueError as exc:
                raise CacheContractError(str(exc)) from exc

    def _key_reserve(self, omega_record: tuple[Any, ...], kperp_record: tuple[Any, ...]) -> int:
        if type(self._table_extra_reserve_bytes) is not int or self._table_extra_reserve_bytes < 0:
            raise CacheContractError("CPU-S4-4B table reserve is invalid")
        try:
            omega_raw = omega_record[2]
            kperp_raw = kperp_record[2]
        except (IndexError, TypeError) as exc:
            raise CacheContractError("cache key records are malformed") from exc
        if not isinstance(omega_raw, bytes) or not isinstance(kperp_raw, bytes):
            raise CacheContractError("cache key records must retain exact raw bytes")
        key_guard = int(len(omega_raw) + len(kperp_raw))
        if key_guard % 8 != 0:
            raise CacheContractError("cache key guard is not float64 aligned")
        return int(key_guard + self._table_extra_reserve_bytes)

    def _make_key(self, **kwargs: Any):
        self._assert_policy()
        kwargs = dict(kwargs)
        kwargs["source_version"] = policy_source_version(kwargs.get("source_version"))
        result = super()._make_key(**kwargs)
        try:
            contract.validate_shape(result[1], scope=self.resource_scope)
        except ValueError as exc:
            raise CacheContractError(str(exc)) from exc
        if self.configured_shape is not None and tuple(result[1]) != tuple(self.configured_shape):
            raise CacheContractError(
                f"configured shape mismatch: {tuple(result[1])}!={tuple(self.configured_shape)}"
            )
        return result

    def _sync_counts(self) -> None:
        super()._sync_counts()
        table_reserve = int(sum(entry.key_reserve_bytes for entry in self._entries.values()))
        key_guard = int(
            sum(
                max(0, int(entry.key_reserve_bytes) - int(self._table_extra_reserve_bytes))
                for entry in self._entries.values()
            )
        )
        self._persistent_bytes = int(
            self._global_reserve_bytes
            + sum(entry.factor_bytes + entry.key_reserve_bytes for entry in self._entries.values())
        )
        self._stats["global_metadata_event_reserve_bytes"] = int(self._global_reserve_bytes)
        self._stats["key_guard_bytes"] = key_guard
        self._stats["table_reserve_bytes"] = table_reserve
        self._stats["table_extra_reserve_bytes"] = int(self._table_extra_reserve_bytes)
        self._stats["persistent_bytes_reserved"] = int(self._persistent_bytes)
        self._stats["owned_audit_samples"] = int(self._owned_audit_samples)
        self._stats["owned_audit_last_bytes"] = int(self._last_owned_audit.get("owned_bytes", 0))
        self._stats["owned_audit_max_bytes"] = int(self._owned_audit_max_bytes)

    def _evict_for(self, factor_bytes: int, key_reserve_bytes: int) -> None:
        self._assert_policy()
        required = int(self._global_reserve_bytes) + int(factor_bytes) + int(key_reserve_bytes)
        if required > int(self.max_bytes):
            raise CacheContractError(
                "factor payload plus full-size global/table reserve exceeds cache capacity: "
                f"{required}>{self.max_bytes}"
            )
        # The inherited method performs the original complete-entry LRU loop;
        # this call intentionally does not change its eviction order.
        super()._evict_for(int(factor_bytes), int(key_reserve_bytes))

    def begin_half(self, **kwargs: Any) -> None:
        self._assert_policy()
        super().begin_half(**kwargs)

    @staticmethod
    def _audit_size(value: Any) -> int:
        try:
            return int(sys.getsizeof(value))
        except (TypeError, ValueError) as exc:
            raise CacheContractError("owned-object size is unknown") from exc

    def _measure_owned_objects(self) -> dict[str, Any]:
        """Measure explicit cache-owned metadata and complete tables only."""

        if self._pending is not None:
            raise CacheContractError("owned-object audit requires no pending half-step")
        seen: set[int] = set()
        labels: list[str] = []
        total = 0

        def walk(value: Any, label: str) -> None:
            nonlocal total
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            labels.append(str(label))
            total += self._audit_size(value)
            if isinstance(value, np.ndarray):
                # Read only ndarray metadata; do not inspect any element.
                if not bool(value.flags.c_contiguous):
                    raise CacheContractError(f"owned ndarray is not C-contiguous: {label}")
                if not bool(value.flags.owndata):
                    raise CacheContractError(f"owned ndarray is an external view: {label}")
                if bool(value.flags.writeable):
                    raise CacheContractError(f"owned ndarray is writable: {label}")
                if any(int(stride) <= 0 for stride in value.strides):
                    raise CacheContractError(f"owned ndarray has non-positive stride: {label}")
                _ = value.shape
                _ = value.dtype
                _ = value.nbytes
                return
            if value is None or type(value) in (str, bytes, int, float, bool):
                return
            if isinstance(value, _Entry):
                walk(value.__dict__, f"{label}.__dict__")
                return
            if isinstance(value, (dict, OrderedDict)):
                for index, (key, item) in enumerate(value.items()):
                    walk(key, f"{label}.key[{index}]")
                    walk(item, f"{label}.value[{index}]")
                return
            if type(value) in (list, tuple):
                for index, item in enumerate(value):
                    walk(item, f"{label}[{index}]")
                return
            raise CacheContractError(f"unknown owned-object type at {label}: {type(value)!r}")

        roots = (
            ("self.owner_identity", self.owner_identity),
            ("self.algorithm_source_identity", self.algorithm_source_identity),
            ("self.resource_scope", self.resource_scope),
            ("self.configured_shape", self.configured_shape),
            ("self.policy_schema", self.policy_schema),
            ("self.fullsize_cache_schema", self.fullsize_cache_schema),
            ("self.max_tables", self.max_tables),
            ("self.max_bytes", self.max_bytes),
            ("self._global_reserve_bytes", self._global_reserve_bytes),
            ("self._table_extra_reserve_bytes", self._table_extra_reserve_bytes),
            ("self._entries", self._entries),
            ("self._events", self._events),
            ("self._stats", self._stats),
            ("self._events_dropped", self._events_dropped),
            ("self._next_key_id", self._next_key_id),
            ("self._persistent_bytes", self._persistent_bytes),
            ("self._last_owned_audit", self._last_owned_audit),
        )
        total += self._audit_size(self)
        seen.add(id(self))
        # This class owns only cache metadata; never recurse into its scientific
        # state/producer owner or function globals, which are not stored here.
        walk(self.__dict__, "cache.__dict__")
        return {
            "status": "PASS",
            "owned_bytes": int(total),
            "reserved_bytes": int(self._persistent_bytes),
            "sample_count": int(len(seen)),
            "max_owned_bytes": int(max(int(self._owned_audit_max_bytes), int(total))),
            "object_count": int(len(seen)),
            "scope": self.resource_scope,
            "capacity_scope": "persistent_reserved_not_RSS_or_cgroup",
            "workspace_scope": "cache-owned-complete-entries-only",
            "array_payloads_read": False,
            "labels_sampled": int(len(labels)),
        }

    def _reject_completed_entries(self, reason: str) -> None:
        # Fail closed: release every complete entry and any active pending
        # reference.  The caller raises and cannot silently fall back.
        discarded_entries = len(self._entries)
        self._entries.clear()
        self._pending = None
        prior = dict(self._last_owned_audit)
        self._last_owned_audit = {
            "status": "REJECTED",
            "reason": str(reason),
            "owned_bytes": int(prior.get("owned_bytes", 0)),
            "reserved_bytes": int(prior.get("reserved_bytes", self._global_reserve_bytes)),
            "sample_count": int(self._owned_audit_samples),
            "max_owned_bytes": int(self._owned_audit_max_bytes),
            "scope": self.resource_scope,
            "capacity_scope": "persistent_reserved_not_RSS_or_cgroup",
            "workspace_scope": "cache-owned-complete-entries-only",
        }
        self._stats["owned_audit_rejections"] = int(self._stats["owned_audit_rejections"]) + 1
        self._record_event({"event": "owned_object_audit", "outcome": "rejected", "reason": str(reason), "discarded_entries": discarded_entries})
        self._sync_counts()

    def finish_half(self, *, success: bool) -> None:
        if not success:
            # Cleanup must remain available even when a resource setting is bad.
            super().finish_half(success=False)
            return
        self._assert_policy()
        pending = self._pending
        audit_new_table = bool(pending is not None and bool(success) and pending.entry is None)
        published_key = pending.key if audit_new_table and pending is not None else None
        super().finish_half(success=bool(success))
        if not audit_new_table:
            return
        if published_key is None or published_key not in self._entries:
            raise CacheContractError("full-size table publication disappeared before audit")
        try:
            measurement = self._measure_owned_objects()
        except BaseException as exc:
            self._reject_completed_entries(f"owned-object-audit:{exc}")
            if isinstance(exc, CacheContractError):
                raise
            raise CacheContractError("owned-object audit failed") from exc
        self._owned_audit_samples += 1
        owned_bytes = int(measurement["owned_bytes"])
        self._owned_audit_max_bytes = max(int(self._owned_audit_max_bytes), owned_bytes)
        measurement["sample_count"] = int(self._owned_audit_samples)
        measurement["max_owned_bytes"] = int(self._owned_audit_max_bytes)
        self._last_owned_audit = copy.deepcopy(measurement)
        self._sync_counts()
        if owned_bytes > int(self._persistent_bytes):
            reason = (
                "underestimated reserve: owned cache objects exceed coded persistent reserve "
                f"{owned_bytes}>{self._persistent_bytes}"
            )
            self._reject_completed_entries(reason)
            raise CacheContractError(reason)
        self._record_event(
            {
                "event": "owned_object_audit",
                "outcome": "pass",
                "owned_bytes": owned_bytes,
                "reserved_bytes": int(self._persistent_bytes),
                "object_count": int(measurement["object_count"]),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        self._assert_policy()
        self._sync_counts()
        result = super().snapshot()
        result["base_cache_schema"] = result.get("schema")
        result["schema"] = CACHE_SCHEMA
        result["persistent_capacity_scope"] = (
            "global_metadata_event_reserve_plus_factor_payload_plus_key_guard_plus_table_reserve"
        )
        result["fullsize_policy_id"] = FULLSIZE_POLICY_ID
        result["policy_source_version"] = FULLSIZE_POLICY_SOURCE_VERSION
        result["policy_source_identity"] = copy.deepcopy(_SOURCE_IDENTITY)
        result["resource_scope"] = self.resource_scope
        result["configured_shape"] = (
            list(self.configured_shape) if self.configured_shape is not None else None
        )
        result["support_domain"] = {
            "full_shape": list(contract.FULL_SHAPE),
            "local_engineering_shapes": [list(shape) for shape in contract.LOCAL_ENGINEERING_SHAPES],
            "full_shape_local_allocation_allowed": False,
        }
        result["fullsize_capacity"] = {
            "max_tables": int(contract.FULLSIZE_MAX_TABLES),
            "max_bytes": int(self.max_bytes),
            "fullsize_max_bytes": int(contract.FULLSIZE_MAX_BYTES),
            "smallgrid_default_max_bytes": int(contract.SMALLGRID_DEFAULT_MAX_BYTES),
            "global_metadata_event_reserve_bytes": int(self._global_reserve_bytes),
            "table_extra_reserve_bytes": int(self._table_extra_reserve_bytes),
            "persistent_bytes_formula": "global + sum(payload + key_guard + table_extra)",
        }
        result["owned_object_audit"] = copy.deepcopy(self._last_owned_audit)
        json.dumps(result, allow_nan=False)
        return copy.deepcopy(result)


__all__ = [
    "CACHE_SCHEMA",
    "CacheContractError",
    "FULLSIZE_POLICY_ID",
    "FULLSIZE_POLICY_SOURCE_VERSION",
    "FullsizeLinearFactorCache",
    "policy_source_version",
    "source_identity",
]

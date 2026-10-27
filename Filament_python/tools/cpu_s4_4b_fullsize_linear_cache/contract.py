"""Pure-Python capacity contract for the CPU-S4-4B private cache.

This module intentionally has no NumPy or simulation imports.  It is the
small, independently reproducible part of the full-size memory contract:
payload arithmetic, key-guard accounting, the fixed metadata reserve, and
the admitted engineering shapes.  It does not allocate a scientific array.
"""

from __future__ import annotations

from typing import Any, Iterable


POLICY_SCHEMA = "filament.cpu_s4_4b.fullsize_linear_cache.contract.v1"
FULL_SHAPE = (384, 351, 301)
MICRO_SHAPE_384 = (384, 17, 19)
MICRO_SHAPE_32 = (32, 17, 19)
SMALLGRID_SHAPE = (384, 117, 129)
LOCAL_ENGINEERING_SHAPES = (
    MICRO_SHAPE_384,
    MICRO_SHAPE_32,
    SMALLGRID_SHAPE,
)
SUPPORTED_SHAPES = LOCAL_ENGINEERING_SHAPES + (FULL_SHAPE,)

COMPLEX128_ITEMSIZE = 16
FULL_PAYLOAD_BYTES = 649_119_744
FULLSIZE_MAX_TABLES = 2
FULLSIZE_MAX_BYTES = 1_610_612_736
GLOBAL_METADATA_EVENT_RESERVE_BYTES = 4 * 1024 * 1024
TABLE_EXTRA_RESERVE_BYTES = 256 * 1024
SMALLGRID_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
SMALLGRID_DEFAULT_MAX_TABLES = 2

LOCAL_ENGINEERING_SCOPE = "LOCAL_ENGINEERING_ONLY"
FULLSIZE_SCOPE = "FULLSIZE_PRIVATE_L1"
VALID_SCOPES = (LOCAL_ENGINEERING_SCOPE, FULLSIZE_SCOPE)


def shape_tuple(shape: Iterable[int]) -> tuple[int, int, int]:
    """Return a strict positive three-dimensional integer shape."""

    values = tuple(shape)
    if len(values) != 3 or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"shape must contain three positive Python ints: {values!r}")
    return values


def validate_shape(shape: Iterable[int], *, scope: str | None = None) -> tuple[int, int, int]:
    """Validate an admitted shape without allocating an array."""

    actual = shape_tuple(shape)
    if actual not in SUPPORTED_SHAPES:
        raise ValueError(f"unsupported CPU-S4-4B shape: {actual!r}")
    if scope is not None:
        scope_name = str(scope)
        if scope_name not in VALID_SCOPES:
            raise ValueError(f"unknown CPU-S4-4B resource scope: {scope_name!r}")
        if scope_name == FULLSIZE_SCOPE and actual != FULL_SHAPE:
            raise ValueError("FULLSIZE_PRIVATE_L1 admits only the full production shape")
        if scope_name == LOCAL_ENGINEERING_SCOPE and actual == FULL_SHAPE:
            raise ValueError("full production shape is not admitted to local engineering")
    return actual


def payload_bytes(shape: Iterable[int], *, itemsize: int = COMPLEX128_ITEMSIZE) -> int:
    """Calculate one complete factor-table payload size arithmetically."""

    actual = shape_tuple(shape)
    if type(itemsize) is not int or itemsize <= 0:
        raise ValueError("itemsize must be a positive Python int")
    return int(actual[0] * actual[1] * actual[2] * itemsize)


def key_guard_bytes(shape: Iterable[int]) -> int:
    """Bytes retained by the exact Omega/kperp2 key guard."""

    nt, ny, nx = shape_tuple(shape)
    return int((nt + ny * nx) * 8)


def table_reserve_bytes(shape: Iterable[int]) -> int:
    """Payload-independent reserve for one table's key/container metadata."""

    return int(key_guard_bytes(shape) + TABLE_EXTRA_RESERVE_BYTES)


def persistent_reserved_bytes(shape: Iterable[int], table_count: int) -> int:
    """Global reserve plus exact payload/reserve for *table_count* tables."""

    actual = shape_tuple(shape)
    if type(table_count) is not int or table_count < 0:
        raise ValueError("table_count must be a non-negative Python int")
    per_table = payload_bytes(actual) + table_reserve_bytes(actual)
    return int(GLOBAL_METADATA_EVENT_RESERVE_BYTES + table_count * per_table)


def capacity_report(shape: Iterable[int] = FULL_SHAPE) -> dict[str, Any]:
    """Return a JSON-friendly, allocation-free capacity report."""

    actual = validate_shape(shape)
    payload = payload_bytes(actual)
    guard = key_guard_bytes(actual)
    reserve = table_reserve_bytes(actual)
    return {
        "schema": POLICY_SCHEMA,
        "shape": list(actual),
        "itemsize": COMPLEX128_ITEMSIZE,
        "payload_bytes": payload,
        "key_guard_bytes": guard,
        "table_reserve_bytes": reserve,
        "table_extra_reserve_bytes": TABLE_EXTRA_RESERVE_BYTES,
        "global_metadata_event_reserve_bytes": GLOBAL_METADATA_EVENT_RESERVE_BYTES,
        "max_tables": FULLSIZE_MAX_TABLES,
        "max_bytes": FULLSIZE_MAX_BYTES,
        "persistent_reserved_one_table_bytes": persistent_reserved_bytes(actual, 1),
        "persistent_reserved_two_table_bytes": persistent_reserved_bytes(actual, 2),
        "local_engineering_shape": actual in LOCAL_ENGINEERING_SHAPES,
        "fullsize_shape": actual == FULL_SHAPE,
    }


__all__ = [
    "COMPLEX128_ITEMSIZE",
    "FULL_PAYLOAD_BYTES",
    "FULL_SHAPE",
    "FULLSIZE_MAX_BYTES",
    "FULLSIZE_MAX_TABLES",
    "FULLSIZE_SCOPE",
    "GLOBAL_METADATA_EVENT_RESERVE_BYTES",
    "LOCAL_ENGINEERING_SCOPE",
    "LOCAL_ENGINEERING_SHAPES",
    "MICRO_SHAPE_32",
    "MICRO_SHAPE_384",
    "POLICY_SCHEMA",
    "SMALLGRID_DEFAULT_MAX_BYTES",
    "SMALLGRID_DEFAULT_MAX_TABLES",
    "SMALLGRID_SHAPE",
    "SUPPORTED_SHAPES",
    "TABLE_EXTRA_RESERVE_BYTES",
    "VALID_SCOPES",
    "capacity_report",
    "key_guard_bytes",
    "payload_bytes",
    "persistent_reserved_bytes",
    "shape_tuple",
    "table_reserve_bytes",
    "validate_shape",
]

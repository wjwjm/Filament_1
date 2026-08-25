"""Hashing and path-safety helpers used by the campaign CLI.

The helpers in this module deliberately operate on bytes and normalized
relative paths.  Campaign provenance must identify the bytes that were
actually written, not a re-serialized approximation of a file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator


CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA256 digest for *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA256 digest for a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically for fingerprints and manifests."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value using :func:`canonical_json_bytes`."""

    return sha256_bytes(canonical_json_bytes(value))


def normalize_relative(path: str | Path) -> str:
    """Return a POSIX relative path suitable for a manifest.

    Backslashes are normalized so manifests generated on Windows and Linux
    use the same path spelling.  Absolute paths and parent traversal are
    rejected because a manifest is a portable, repository-relative record.
    """

    raw = str(path).replace("\\", "/")
    if not raw or raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
        raise ValueError(f"manifest path must be relative: {path!s}")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"manifest path escapes its root: {path!s}")
    if not parts:
        raise ValueError("manifest path must not be empty")
    return "/".join(parts)


def ensure_within(root: Path, candidate: Path, *, allow_missing: bool = False) -> Path:
    """Resolve *candidate* and require it to be contained by *root*.

    ``Path.resolve`` follows symlinks, so this also prevents a path that
    lexically appears inside the root from escaping through a symlink.
    """

    if root.is_symlink():
        raise ValueError(f"permitted root must not be a symlink: {root}")
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=not allow_missing)
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"path is outside the permitted root: {candidate_resolved}"
        ) from exc
    return candidate_resolved


def iter_regular_files(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(absolute_path, POSIX_relative_path)`` in deterministic order.

    Symlinks are rejected rather than followed or silently omitted.  This
    makes a manifest fail closed when a result directory contains an
    unexpected link to another experiment or to a credential file.
    """

    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"manifest root is not a directory: {root}")
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for dirname in list(dirnames):
            path = current_path / dirname
            if path.is_symlink():
                raise ValueError(f"symlink directory is not allowed: {path}")
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                raise ValueError(f"symlink file is not allowed: {path}")
            if not path.is_file():
                raise ValueError(f"manifest entry is not a regular file: {path}")
            relative = normalize_relative(path.relative_to(root))
            yield path, relative


def file_record(path: Path, relative: str, artifact_class: str) -> dict[str, object]:
    """Build one stable manifest file record."""

    stat = path.stat()
    return {
        "path": normalize_relative(relative),
        "size": stat.st_size,
        "sha256": sha256_file(path),
        "artifact_class": artifact_class,
    }


def manifest_sha256(manifest: dict[str, object]) -> str:
    """Return the digest of a manifest object in canonical form."""

    return canonical_json_sha256(manifest)

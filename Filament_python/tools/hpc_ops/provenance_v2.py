#!/usr/bin/env python3
"""Create and validate cross-platform provenance manifests.

Version 2 deliberately separates two kinds of bytes:

* tracked text is bound to its committed Git blob and a canonical-LF digest;
* external artifacts are bound to their exact raw-byte SHA256 digest.

The module has no knowledge of the production simulation and does not alter
the existing v1 provenance files.  It is intentionally usable as both a
small library and a command-line tool so tests can exercise the same code as
the HPC preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlsplit


SCHEMA = "filament.provenance.v2"
VERSION = 2


class ProvenanceError(ValueError):
    """Raised when a provenance precondition or validation check fails."""


def canonical_lf_bytes(data: bytes) -> bytes:
    """Return *data* with CRLF and bare CR line endings represented as LF."""

    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_lf_sha256(data: bytes) -> str:
    """Hash text after canonicalising its line endings to LF."""

    return hashlib.sha256(canonical_lf_bytes(data)).hexdigest()


def raw_sha256(data: bytes) -> str:
    """Hash exact bytes without line-ending normalisation."""

    return hashlib.sha256(data).hexdigest()


def raw_sha256_file(path: Path) -> str:
    """Return the exact-byte SHA256 of a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise ProvenanceError(message)
    return result.stdout.strip()


def _repo_root(repo: Path) -> Path:
    return Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()


def _repo_relative(repo: Path, value: str) -> str:
    """Resolve a user path to a safe Git repo-relative POSIX path."""

    root = _repo_root(repo)
    candidate = Path(value)
    try:
        if candidate.is_absolute():
            relative = candidate.absolute().relative_to(root)
        else:
            relative = Path(os.path.normpath(str(candidate)))
    except ValueError as exc:
        raise ProvenanceError(f"tracked path is outside repository: {value}") from exc
    if str(relative) in {"", "."} or str(relative).startswith(".."):
        raise ProvenanceError(f"tracked path is not repo-relative: {value}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ProvenanceError(f"tracked path contains an unsafe component: {value}")
    tracked_path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ProvenanceError(f"tracked path must not contain a symlink: {relative}")
    return relative.as_posix()


def _ensure_clean(repo: Path) -> None:
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProvenanceError("repository worktree must be clean before create")


def _git_blob_oid(repo: Path, relative: str) -> str:
    tracked = _git(repo, "ls-files", "--error-unmatch", "--", relative, check=False)
    if tracked != relative:
        raise ProvenanceError(f"tracked path is not committed: {relative}")
    oid = _git(repo, "rev-parse", "--verify", f"HEAD:{relative}")
    if not oid or any(char not in "0123456789abcdef" for char in oid.lower()):
        raise ProvenanceError(f"invalid Git blob oid for {relative}")
    if _git(repo, "cat-file", "-t", oid) != "blob":
        raise ProvenanceError(f"Git object is not a blob: {relative}")
    return oid


def _read_tracked(repo: Path, relative: str) -> bytes:
    path = repo / Path(*relative.split("/"))
    current = repo
    for part in Path(*relative.split("/")).parts:
        current /= part
        if current.is_symlink():
            raise ProvenanceError(f"tracked path must not contain a symlink: {relative}")
    if not path.is_file():
        raise ProvenanceError(f"tracked file is missing: {relative}")
    return path.read_bytes()


def _safe_repository_identity(identity: str) -> str:
    """Reject control characters and credentials in a recorded remote URL."""

    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in identity):
        raise ProvenanceError("repository identity contains a control character")
    if "://" in identity:
        parsed = urlsplit(identity)
        if parsed.username is not None or parsed.password is not None:
            raise ProvenanceError("repository identity must not contain URL credentials")
    return identity


def _metadata(repo: Path) -> dict[str, str]:
    branch = _git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    if not branch:
        raise ProvenanceError("provenance create requires a named branch")
    identity = _safe_repository_identity(
        _git(repo, "remote", "get-url", "origin", check=False) or repo.name
    )
    return {
        "head": _git(repo, "rev-parse", "HEAD"),
        "branch": branch,
        "identity": identity,
    }


def _assert_output_target(repo: Path, output: Path) -> None:
    if output.is_symlink() or output.exists():
        raise ProvenanceError("provenance output target must not already exist or be a symlink")
    if not output.parent.exists() or not output.parent.is_dir():
        raise ProvenanceError(f"output directory does not exist: {output.parent}")
    root = _repo_root(repo)
    lexical_output = Path(os.path.abspath(output))
    try:
        lexical_output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ProvenanceError("provenance output must be outside the repository")
    resolved_output = output.resolve(strict=False)
    try:
        resolved_output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ProvenanceError("provenance output must be outside the repository")
    if "provenance_221822" in resolved_output.parts:
        raise ProvenanceError("provenance v2 must not overwrite the frozen v1 provenance area")


def create_manifest(
    repo: str | os.PathLike[str],
    output: str | os.PathLike[str],
    tracked_paths: Iterable[str],
    external_paths: Iterable[str],
) -> dict[str, Any]:
    """Create a v2 manifest after enforcing clean, committed, LF worktree state."""

    repo_path = Path(repo).resolve()
    root = _repo_root(repo_path)
    _ensure_clean(root)

    tracked_records: list[dict[str, str]] = []
    for value in tracked_paths:
        relative = _repo_relative(root, value)
        oid = _git_blob_oid(root, relative)
        data = _read_tracked(root, relative)
        if b"\r\n" in data or b"\r" in data:
            raise ProvenanceError(f"tracked text must use LF in worktree: {relative}")
        tracked_records.append(
            {
                "path": relative,
                "git_blob_oid": oid,
                "canonical_lf_sha256": canonical_lf_sha256(data),
            }
        )

    external_records: list[dict[str, str]] = []
    for value in external_paths:
        candidate = Path(value)
        if candidate.is_symlink():
            raise ProvenanceError(f"external artifact must not be a symlink: {value}")
        path = candidate.resolve()
        if not path.is_file():
            raise ProvenanceError(f"external artifact is not a regular file: {value}")
        external_records.append(
            {
                "path": str(path),
                "raw_sha256": raw_sha256_file(path),
            }
        )

    if not tracked_records and not external_records:
        raise ProvenanceError("at least one tracked or external path is required")

    metadata = _metadata(root)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "repository": metadata["identity"],
        "repository_path": str(root),
        "head": metadata["head"],
        "branch": metadata["branch"],
        "line_endings": {
            "tracked_create": "LF-required",
            "tracked_validate": "canonical-LF",
            "external": "raw-bytes",
        },
        "tracked_text": tracked_records,
        "external": external_records,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    output_path = Path(output).absolute()
    _assert_output_target(root, output_path)
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=str(output_path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
        if output_path.exists() or output_path.is_symlink():
            raise ProvenanceError("provenance output target appeared during create")
        try:
            os.link(temporary, output_path)
        except FileExistsError as exc:
            raise ProvenanceError("provenance output target appeared during create") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def _validate_tracked(repo: Path, record: dict[str, Any]) -> None:
    required = {"path", "git_blob_oid", "canonical_lf_sha256"}
    if set(record) != required:
        raise ProvenanceError("tracked_text record has an unexpected shape")
    relative = _repo_relative(repo, str(record["path"]))
    if relative != record["path"]:
        raise ProvenanceError("tracked path is not canonical")
    actual_oid = _git_blob_oid(repo, relative)
    if actual_oid != record["git_blob_oid"]:
        raise ProvenanceError(f"Git blob mismatch: {relative}")
    actual_hash = canonical_lf_sha256(_read_tracked(repo, relative))
    if actual_hash != record["canonical_lf_sha256"]:
        raise ProvenanceError(f"canonical text hash mismatch: {relative}")


def _validate_external(record: dict[str, Any]) -> None:
    required = {"path", "raw_sha256"}
    if set(record) != required:
        raise ProvenanceError("external record has an unexpected shape")
    path = Path(str(record["path"]))
    if path.is_symlink():
        raise ProvenanceError(f"external artifact must not be a symlink: {path}")
    if not path.is_file():
        raise ProvenanceError(f"external artifact is missing: {path}")
    actual_hash = raw_sha256_file(path)
    if actual_hash != record["raw_sha256"]:
        raise ProvenanceError(f"raw artifact hash mismatch: {path}")


def validate_manifest(
    repo: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Validate a v2 manifest strictly unless ``require_clean=False`` is explicit."""

    repo_path = _repo_root(Path(repo).resolve())
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read manifest: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ProvenanceError("unsupported provenance schema")
    if manifest.get("version") != VERSION:
        raise ProvenanceError("unsupported provenance version")
    current_identity = _safe_repository_identity(
        _git(repo_path, "remote", "get-url", "origin", check=False) or repo_path.name
    )
    if manifest.get("repository") != current_identity:
        raise ProvenanceError("repository identity does not match manifest")
    current_branch = _git(repo_path, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    if manifest.get("branch") != current_branch:
        raise ProvenanceError("repository branch does not match manifest")
    if manifest.get("head") != _git(repo_path, "rev-parse", "HEAD"):
        raise ProvenanceError("repository HEAD does not match manifest")
    if require_clean:
        _ensure_clean(repo_path)
    for record in manifest.get("tracked_text", []):
        if not isinstance(record, dict):
            raise ProvenanceError("tracked_text record is not an object")
        _validate_tracked(repo_path, record)
    for record in manifest.get("external", []):
        if not isinstance(record, dict):
            raise ProvenanceError("external record is not an object")
        _validate_external(record)
    if not manifest.get("tracked_text") and not manifest.get("external"):
        raise ProvenanceError("manifest contains no records")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a v2 manifest")
    create.add_argument("--repo", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--tracked", nargs="*", default=[])
    create.add_argument("--external", nargs="*", default=[])

    validate = subparsers.add_parser("validate", help="validate a v2 manifest")
    validate.add_argument("--repo", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument(
        "--non-strict",
        action="store_true",
        help="allow a dirty CRLF checkout while retaining canonical content checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest = create_manifest(args.repo, args.output, args.tracked, args.external)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        else:
            manifest = validate_manifest(args.repo, args.manifest, require_clean=not args.non_strict)
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "valid": True,
                        "head": manifest["head"],
                        "manifest": str(Path(args.manifest).resolve()),
                    },
                    ensure_ascii=False,
                )
            )
    except ProvenanceError as exc:
        print(f"provenance_v2: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

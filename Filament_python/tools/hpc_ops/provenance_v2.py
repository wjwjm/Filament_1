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
CLASSIFIED_HASH_SCOPE = "classified_by_record"
TRACKED_HASH_SCOPE = "git_blob_oid+canonical_lf_sha256"
EXTERNAL_HASH_SCOPE = "raw_bytes"
STRICT_TOP_LEVEL_KEYS = {
    "schema",
    "version",
    "repository",
    "repository_path",
    "head",
    "branch",
    "hash_scope",
    "line_endings",
    "records",
    "created_at_utc",
}
LINE_ENDING_POLICY = {
    "tracked_create": "canonical-LF-from-Git-blob",
    "tracked_validate": "canonical-LF-worktree-match",
    "external": "raw-bytes",
}


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


def _read_git_blob(repo: Path, oid: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", oid],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ProvenanceError(message or f"could not read Git blob: {oid}")
    return result.stdout


def _digest(value: Any, *, name: str, lengths: tuple[int, ...] = (64,)) -> str:
    if not isinstance(value, str) or len(value) not in lengths:
        raise ProvenanceError(f"{name} must be a hexadecimal digest")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ProvenanceError(f"{name} must be a hexadecimal digest")
    return value.lower()


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
    """Create a v2 manifest from clean committed content.

    Tracked text is hashed from the committed Git blob.  The worktree may use
    LF or CRLF, but its canonical-LF bytes must still match that blob.  This is
    what makes one manifest portable between Windows and Linux checkouts.
    """

    repo_path = Path(repo).resolve()
    root = _repo_root(repo_path)
    _ensure_clean(root)

    records: list[dict[str, str]] = []
    for value in tracked_paths:
        relative = _repo_relative(root, value)
        oid = _git_blob_oid(root, relative)
        worktree_data = _read_tracked(root, relative)
        blob_data = _read_git_blob(root, oid)
        if canonical_lf_bytes(worktree_data) != canonical_lf_bytes(blob_data):
            raise ProvenanceError(f"tracked text does not match committed Git blob: {relative}")
        records.append(
            {
                "path": relative,
                "classification": "tracked_text",
                "hash_scope": TRACKED_HASH_SCOPE,
                "git_blob_oid": oid,
                "canonical_lf_sha256": canonical_lf_sha256(blob_data),
            }
        )

    for value in external_paths:
        candidate = Path(value)
        if candidate.is_symlink():
            raise ProvenanceError(f"external artifact must not be a symlink: {value}")
        path = candidate.resolve()
        if not path.is_file():
            raise ProvenanceError(f"external artifact is not a regular file: {value}")
        records.append(
            {
                "path": str(path),
                "classification": "external",
                "hash_scope": EXTERNAL_HASH_SCOPE,
                "raw_sha256": raw_sha256_file(path),
            }
        )

    if not records:
        raise ProvenanceError("at least one tracked or external path is required")
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise ProvenanceError("each provenance path must have exactly one classification")

    metadata = _metadata(root)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "repository": metadata["identity"],
        "repository_path": str(root),
        "head": metadata["head"],
        "branch": metadata["branch"],
        "hash_scope": CLASSIFIED_HASH_SCOPE,
        "line_endings": LINE_ENDING_POLICY,
        "records": records,
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


def _validate_tracked(repo: Path, record: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    required = {"path", "git_blob_oid", "canonical_lf_sha256"}
    if strict:
        required |= {"classification", "hash_scope"}
    if set(record) != required:
        raise ProvenanceError("tracked_text record has an unexpected shape")
    if strict:
        if record.get("classification") != "tracked_text":
            raise ProvenanceError("tracked_text record classification is invalid")
        if record.get("hash_scope") != TRACKED_HASH_SCOPE:
            raise ProvenanceError("tracked_text record hash_scope is invalid")
    if not isinstance(record.get("path"), str):
        raise ProvenanceError("tracked path must be a string")
    relative = _repo_relative(repo, record["path"])
    if relative != record["path"]:
        raise ProvenanceError("tracked path is not canonical")
    expected_oid = _digest(record["git_blob_oid"], name="git_blob_oid", lengths=(40, 64))
    expected_hash = _digest(record["canonical_lf_sha256"], name="canonical_lf_sha256")
    actual_oid = _git_blob_oid(repo, relative)
    if actual_oid.lower() != expected_oid:
        raise ProvenanceError(f"Git blob mismatch: {relative}")
    actual_hash = canonical_lf_sha256(_read_tracked(repo, relative))
    if actual_hash != expected_hash:
        raise ProvenanceError(f"canonical text hash mismatch: {relative}")
    return record


def _validate_external(record: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    required = {"path", "raw_sha256"}
    if strict:
        required |= {"classification", "hash_scope"}
    if set(record) != required:
        raise ProvenanceError("external record has an unexpected shape")
    if strict:
        if record.get("classification") != "external":
            raise ProvenanceError("external record classification is invalid")
        if record.get("hash_scope") != EXTERNAL_HASH_SCOPE:
            raise ProvenanceError("external record hash_scope is invalid")
    if not isinstance(record.get("path"), str):
        raise ProvenanceError("external path must be a string")
    path = Path(record["path"])
    if strict and not path.is_absolute():
        raise ProvenanceError("external path must be absolute")
    if path.is_symlink():
        raise ProvenanceError(f"external artifact must not be a symlink: {path}")
    if not path.is_file():
        raise ProvenanceError(f"external artifact is missing: {path}")
    expected_hash = _digest(record["raw_sha256"], name="raw_sha256")
    actual_hash = raw_sha256_file(path)
    if actual_hash != expected_hash:
        raise ProvenanceError(f"raw artifact hash mismatch: {path}")
    return record


def iter_records(manifest: dict[str, Any], *, require_hash_scope: bool = False) -> list[dict[str, Any]]:
    """Return records from either a new strict manifest or a legacy v2 manifest."""

    if not isinstance(manifest, dict):
        raise ProvenanceError("provenance manifest must be an object")
    has_new = "records" in manifest or "hash_scope" in manifest
    if has_new:
        if set(manifest) != STRICT_TOP_LEVEL_KEYS:
            raise ProvenanceError("classified provenance manifest has an unexpected shape")
        if manifest.get("hash_scope") != CLASSIFIED_HASH_SCOPE:
            raise ProvenanceError("provenance manifest hash_scope is invalid")
        if set(manifest) & {"tracked_text", "external"}:
            raise ProvenanceError("new provenance manifests must use records only")
        records = manifest.get("records")
        if not isinstance(records, list):
            raise ProvenanceError("provenance manifest records must be a list")
        return records
    if require_hash_scope:
        raise ProvenanceError("strict provenance validation requires hash_scope and records")
    tracked = manifest.get("tracked_text", [])
    external = manifest.get("external", [])
    if not isinstance(tracked, list) or not isinstance(external, list):
        raise ProvenanceError("legacy provenance record groups must be lists")
    return [*tracked, *external]


def lookup_record(
    manifest: dict[str, Any], path: str, *, classification: str | None = None,
    require_hash_scope: bool = False,
) -> dict[str, Any]:
    """Find one provenance record by path and optional classification."""

    matches = []
    for record in iter_records(manifest, require_hash_scope=require_hash_scope):
        if not isinstance(record, dict) or record.get("path") != path:
            continue
        actual_classification = record.get("classification")
        if actual_classification is None and not require_hash_scope:
            actual_classification = "tracked_text" if "git_blob_oid" in record else "external"
        if classification is None or actual_classification == classification:
            matches.append(record)
    if len(matches) != 1:
        qualifier = f" ({classification})" if classification else ""
        raise ProvenanceError(f"expected exactly one provenance record for {path}{qualifier}")
    return matches[0]


find_record = lookup_record


def validate_record(
    repo: str | os.PathLike[str], record: dict[str, Any], *, require_hash_scope: bool = False,
) -> dict[str, Any]:
    """Validate one tracked or external record using the canonical hash helpers."""

    if not isinstance(record, dict):
        raise ProvenanceError("provenance record must be an object")
    classification = record.get("classification")
    if classification == "tracked_text":
        return _validate_tracked(_repo_root(Path(repo).resolve()), record, strict=require_hash_scope)
    if classification == "external":
        return _validate_external(record, strict=require_hash_scope)
    if not require_hash_scope and "classification" not in record:
        if "git_blob_oid" in record:
            return _validate_tracked(_repo_root(Path(repo).resolve()), record, strict=False)
        if "raw_sha256" in record:
            return _validate_external(record, strict=False)
    raise ProvenanceError("provenance record classification is invalid")


def validate_manifest(
    repo: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    require_clean: bool = True,
    require_hash_scope: bool = False,
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
    strict_records = manifest.get("hash_scope") == CLASSIFIED_HASH_SCOPE
    if strict_records:
        if set(manifest) != STRICT_TOP_LEVEL_KEYS:
            raise ProvenanceError("classified provenance manifest has an unexpected shape")
        if not isinstance(manifest.get("repository"), str) or not manifest["repository"]:
            raise ProvenanceError("repository identity must be a non-empty string")
        if not isinstance(manifest.get("repository_path"), str) or not manifest["repository_path"]:
            raise ProvenanceError("repository_path must be a non-empty string")
        _digest(manifest.get("head"), name="head", lengths=(40, 64))
        if not isinstance(manifest.get("branch"), str) or not manifest["branch"]:
            raise ProvenanceError("branch must be a non-empty string")
        if manifest.get("line_endings") != LINE_ENDING_POLICY:
            raise ProvenanceError("line_endings policy is invalid")
        if not isinstance(manifest.get("created_at_utc"), str) or not manifest["created_at_utc"]:
            raise ProvenanceError("created_at_utc must be a non-empty string")
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
    records = iter_records(manifest, require_hash_scope=require_hash_scope)
    if require_hash_scope and not strict_records:
        raise ProvenanceError("strict provenance validation requires classified hash_scope")
    if not records:
        raise ProvenanceError("manifest contains no records")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ProvenanceError("provenance record is not an object")
        classification = record.get("classification")
        if strict_records and classification not in {"tracked_text", "external"}:
            raise ProvenanceError("provenance record classification is invalid")
        if not strict_records:
            classification = "tracked_text" if "git_blob_oid" in record else "external"
        path_key = str(record.get("path"))
        if path_key in seen:
            raise ProvenanceError(f"duplicate provenance record: {path_key}")
        seen.add(path_key)
        validate_record(repo_path, record, require_hash_scope=strict_records)
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
    validate.add_argument(
        "--require-hash-scope",
        action="store_true",
        help="require the classified records schema and reject legacy v2 groups",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest = create_manifest(args.repo, args.output, args.tracked, args.external)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        else:
            manifest = validate_manifest(
                args.repo,
                args.manifest,
                require_clean=not args.non_strict,
                require_hash_scope=args.require_hash_scope,
            )
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

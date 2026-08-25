#!/usr/bin/env python3
"""Manage local Filament_1 campaign metadata and derived artifacts.

This command is intentionally standard-library-only.  It records provenance
and performs bounded file operations; it never starts a propagation, invokes
Slurm, edits a scientific input in place, or talks to HPC/GitHub.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import ntpath
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlsplit, parse_qsl


# Direct execution (``python tools/campaign/manage.py``) does not put the
# repository root on sys.path.  Keep imports working without a package install.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.campaign.hashing import (  # noqa: E402
    canonical_json_sha256,
    ensure_within,
    file_record,
    iter_regular_files,
    manifest_sha256,
    normalize_relative,
    sha256_file,
)
from tools.campaign.model import (  # noqa: E402
    CAMPAIGN_SCHEMA,
    LIFECYCLE_STATES,
    MANIFEST_SCHEMA,
    REGISTRY_SCHEMA,
    TERMINAL_JOB_STATES,
    CampaignError,
    GIT_SHA_RE,
    SHA256_RE,
    iter_json_strings,
    load_campaign,
    load_json,
    make_initial_campaign,
    secret_key_reason,
    validate_campaign_id,
    validate_git_sha,
    validate_sha256,
    write_json,
)


REPO_DEFAULT = Path.cwd()
PROHIBITED_GITHUB_SUFFIXES = {".npz", ".npy", ".mat", ".h5", ".hdf5"}
PROHIBITED_GITHUB_PARTS = {
    "cache",
    "caches",
    ".cache",
    "secrets",
    ".secrets",
    "secret",
    "credentials",
    ".credentials",
    ".env",
}
AUTH_QUERY_KEYS = {"token", "access_token", "refresh_token", "password", "secret", "api_key", "apikey", "credential"}


def _json_output(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "root", None) or REPO_DEFAULT).resolve()


def _campaign_dir(root: Path, campaign_id: str) -> Path:
    validate_campaign_id(campaign_id)
    return root / "results" / "campaigns" / campaign_id


def _artifacts_dir(root: Path, campaign_id: str) -> Path:
    return root / ".artifacts" / campaign_id


def _relative_repo_path(root: Path, path: Path) -> str:
    return normalize_relative(path.resolve().relative_to(root.resolve()))


def _resolve_repo_ref(root: Path, reference: str | None) -> Path | None:
    if not reference:
        return None
    path = Path(reference)
    if path.is_absolute():
        return ensure_within(root, path)
    return ensure_within(root, root / path)


def _path_from_manifest(relative: str) -> Path:
    """Convert one POSIX manifest path to a native path on any host."""

    return Path(*PurePosixPath(normalize_relative(relative)).parts)


def _campaign_hpc_root(campaign: dict[str, Any]) -> Path | None:
    value = campaign.get("hpc_root") or (campaign.get("paths") or {}).get("hpc_root")
    return Path(str(value)) if value else None


def _resolve_campaign_ref(root: Path, campaign: dict[str, Any], reference: str | None) -> Path | None:
    """Resolve a repository path or an explicitly declared HPC campaign path."""

    if not reference:
        return None
    candidate = Path(reference)
    if not candidate.is_absolute():
        return ensure_within(root, root / candidate)
    try:
        return ensure_within(root, candidate)
    except ValueError:
        hpc_root = _campaign_hpc_root(campaign)
        if hpc_root is None:
            raise
        return ensure_within(hpc_root, candidate)


def _write_campaign(path: Path, campaign: dict[str, Any]) -> None:
    write_json(path, campaign)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left.absolute() == right.absolute()


def _safe_repository_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
            raise CampaignError("repository URL must not contain credentials")
        if parsed.query or parsed.fragment:
            raise CampaignError("repository URL must not contain query or fragment data")
    return value


def _cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args)
    campaign_id = validate_campaign_id(args.campaign_id)
    campaign_dir = ensure_within(root, _campaign_dir(root, campaign_id), allow_missing=True)
    campaign_path = campaign_dir / "campaign.json"
    if campaign_path.exists():
        raise CampaignError(f"campaign already exists: {campaign_path}")

    campaign_dir.mkdir(parents=True, exist_ok=True)
    requested_dir = ensure_within(
        root, root / "configs" / "experiments" / campaign_id / "requested", allow_missing=True
    )
    resolved_dir = ensure_within(
        root, root / "configs" / "experiments" / campaign_id / "resolved", allow_missing=True
    )
    requested_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = ensure_within(root, _artifacts_dir(root, campaign_id), allow_missing=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    execution_sha = args.execution_git_sha
    execution_ref = None
    repository_url = None
    try:
        if execution_sha is None:
            execution_sha = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        execution_ref = subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
        repository_url = _safe_repository_url(subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None)
    except (OSError, subprocess.CalledProcessError):
        pass
    campaign = make_initial_campaign(
        campaign_id,
        title=args.title or "",
        scientific_purpose=args.scientific_purpose or "",
        execution_git_sha=execution_sha,
    )
    campaign["execution_ref"] = execution_ref
    campaign["repository_url"] = repository_url
    _write_campaign(campaign_path, campaign)
    return {
        "status": "created",
        "campaign_id": campaign_id,
        "campaign": _relative_repo_path(root, campaign_path),
        "artifacts": _relative_repo_path(root, artifacts_dir),
    }


def _is_absolute_config_string(value: str) -> bool:
    # Both parsers are used because this validator may run on Windows while
    # inspecting an HPC/POSIX configuration, or vice versa.
    return ntpath.isabs(value) or posixpath.isabs(value) or value.startswith("\\\\")


def _config_violations(value: Any) -> list[str]:
    violations: list[str] = []

    def visit(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                key_path = f"{path}.{key}" if path else str(key)
                reason = secret_key_reason(str(key))
                if reason:
                    violations.append(f"{key_path}: {reason}")
                visit(child, key_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")
        elif isinstance(node, str):
            parsed = urlsplit(node)
            is_url = bool(parsed.scheme and parsed.netloc)
            if not is_url and _is_absolute_config_string(node):
                violations.append(f"{path}: absolute path is not publishable")
            if parsed.scheme and parsed.netloc:
                if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
                    violations.append(f"{path}: authenticated URL is not publishable")
                query_keys = {key.casefold() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
                leaked = sorted(query_keys & AUTH_QUERY_KEYS)
                if leaked:
                    violations.append(f"{path}: URL contains credential-like query key(s): {', '.join(leaked)}")

    visit(value)
    return violations


def _config_output_path(root: Path, campaign_id: str, kind: str, output: str | None) -> Path:
    if output:
        return ensure_within(root, root / output if not Path(output).is_absolute() else Path(output), allow_missing=True)
    return ensure_within(
        root,
        root / "results" / "campaigns" / campaign_id / "configs" / kind / "config.json",
        allow_missing=True,
    )


def _cmd_publish_config(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args)
    campaign_id = validate_campaign_id(args.campaign_id)
    kind = args.kind
    if kind not in {"requested", "resolved"}:
        raise CampaignError("config kind must be requested or resolved")
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    input_path = ensure_within(root, input_path)
    value = load_json(input_path)
    if not isinstance(value, (dict, list)):
        raise CampaignError("published config must be a JSON object or array")
    violations = _config_violations(value)
    if violations:
        raise CampaignError("configuration is not publishable:\n" + "\n".join(violations))

    output_path = _config_output_path(root, campaign_id, kind, args.output)
    if _same_file(input_path, output_path):
        raise CampaignError("publish-config refuses to overwrite its input file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != rendered and not args.overwrite:
            raise CampaignError(f"refusing to overwrite differing config: {output_path}")
    output_path.write_text(rendered, encoding="utf-8", newline="\n")

    campaign_path, campaign = load_campaign(root, campaign_id)
    field_path = f"{kind}_config_path"
    field_hash = f"{kind}_config_sha256"
    campaign[field_path] = _relative_repo_path(root, output_path)
    campaign[field_hash] = sha256_file(output_path)
    _write_campaign(campaign_path, campaign)
    return {
        "status": "published_config",
        "campaign_id": campaign_id,
        "kind": kind,
        "path": campaign[field_path],
        "sha256": campaign[field_hash],
        "input": _relative_repo_path(root, input_path),
    }


def _github_forbidden(relative: str) -> str | None:
    parts = PurePosixPath(normalize_relative(relative)).parts
    if any(part.casefold() in PROHIBITED_GITHUB_PARTS for part in parts):
        return "cache/secrets/credential path"
    if Path(parts[-1]).suffix.casefold() in PROHIBITED_GITHUB_SUFFIXES:
        return "raw binary result extension"
    name = parts[-1].casefold()
    if name in {".env", "credentials.json", "credential.json", "secret.json"}:
        return "credential-like filename"
    return None


def _manifest_source(root: Path, campaign_id: str, source: str | None, campaign: dict[str, Any] | None = None) -> Path:
    if source:
        path = Path(source)
        candidate = path if path.is_absolute() else root / path
        if candidate.is_symlink():
            raise CampaignError(f"manifest root may not be a symlink: {candidate}")
        if campaign is not None:
            return _resolve_campaign_ref(root, campaign, str(candidate)) or candidate
        return ensure_within(root, candidate)
    return ensure_within(root, _artifacts_dir(root, campaign_id))


def _default_manifest_path(root: Path, campaign_id: str, artifact_class: str, source: Path) -> Path:
    names = {
        "derived": "artifact_manifest.json",
        "local": "artifact_manifest.json",
        "raw": "raw_manifest.json",
        "github": "github_manifest.json",
    }
    if artifact_class == "github":
        return root / "results" / "campaigns" / campaign_id / names[artifact_class]
    try:
        source.resolve().relative_to(root.resolve())
    except ValueError:
        return source / names[artifact_class]
    return _artifacts_dir(root, campaign_id) / names[artifact_class]


def _collect_manifest(root: Path, campaign_id: str, source: Path, artifact_class: str, output: Path | None = None) -> dict[str, Any]:
    if source.is_symlink():
        raise CampaignError(f"manifest root may not be a symlink: {source}")
    records: list[dict[str, Any]] = []
    output_relative: str | None = None
    if output is not None:
        try:
            output_relative = normalize_relative(output.resolve().relative_to(source.resolve()))
        except ValueError:
            output_relative = None
    for path, relative in iter_regular_files(source):
        # Validation receipts are local bookkeeping, not campaign artifacts.
        if relative == ".validation" or relative.startswith(".validation/"):
            continue
        if output_relative and relative == output_relative:
            continue
        if artifact_class == "github":
            reason = _github_forbidden(relative)
            if reason:
                raise CampaignError(f"GitHub artifact manifest rejects {relative}: {reason}")
        records.append(file_record(path, relative, artifact_class))
    records.sort(key=lambda record: str(record["path"]))
    try:
        source_relative = _relative_repo_path(root, source)
    except ValueError:
        source_relative = source.resolve().as_posix()
    return {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": campaign_id,
        "artifact_class": artifact_class,
        "root": source_relative,
        "files": records,
    }


def _cmd_build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args)
    campaign_id = validate_campaign_id(args.campaign_id)
    campaign_path, campaign = load_campaign(root, campaign_id)
    artifact_class = args.artifact_class
    source = _manifest_source(root, campaign_id, args.source, campaign)
    output_arg = args.output
    output = None
    if output_arg:
        output = Path(output_arg)
        candidate = output if output.is_absolute() else root / output
        try:
            output = ensure_within(root, candidate, allow_missing=True)
        except ValueError:
            hpc_root = _campaign_hpc_root(campaign)
            if hpc_root is None:
                raise
            output = ensure_within(hpc_root, candidate, allow_missing=True)
    else:
        output = _default_manifest_path(root, campaign_id, artifact_class, source)
        try:
            output = ensure_within(root, output, allow_missing=True)
        except ValueError:
            hpc_root = _campaign_hpc_root(campaign)
            if hpc_root is None:
                raise
            output = ensure_within(hpc_root, output, allow_missing=True)
    manifest = _collect_manifest(root, campaign_id, source, artifact_class, output)
    if output.exists() and not args.overwrite:
        try:
            existing = load_json(output)
        except CampaignError:
            existing = None
        if existing != manifest:
            raise CampaignError(f"refusing to overwrite differing manifest: {output}")
    else:
        write_json(output, manifest, overwrite=True)
    output_hash = sha256_file(output)
    try:
        output_reference = _relative_repo_path(root, output)
    except ValueError:
        output_reference = output.resolve().as_posix()
    field_prefix = {"derived": "artifact", "local": "artifact", "raw": "raw", "github": "github"}[artifact_class]
    try:
        output.resolve().relative_to(root.resolve())
        output_is_local = True
    except ValueError:
        output_is_local = False
    if output_is_local:
        campaign[f"{field_prefix}_manifest_path"] = output_reference
        campaign[f"{field_prefix}_manifest_sha256"] = output_hash
        _write_campaign(campaign_path, campaign)
    return {
        "status": "manifest_built",
        "campaign_id": campaign_id,
        "artifact_class": artifact_class,
        "path": output_reference,
        "sha256": output_hash,
        "file_count": len(manifest["files"]),
    }


def _manifest_records(
    path: Path,
    *,
    expected_campaign_id: str | None = None,
    allowed_classes: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    value = load_json(path)
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        raise CampaignError(f"invalid artifact manifest: {path}")
    if expected_campaign_id is not None and value.get("campaign_id") != expected_campaign_id:
        raise CampaignError(f"manifest campaign_id mismatch: {path}")
    if allowed_classes is not None and value.get("artifact_class") not in allowed_classes:
        raise CampaignError(f"manifest artifact_class is not allowed: {path}")
    if not isinstance(value.get("root"), str) or not value.get("root"):
        raise CampaignError(f"manifest root is missing: {path}")
    records = value.get("files")
    if not isinstance(records, list):
        raise CampaignError(f"manifest files must be a list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise CampaignError(f"invalid manifest record in {path}")
        relative = normalize_relative(record["path"])
        if relative in result:
            raise CampaignError(f"duplicate manifest path in {path}: {relative}")
        result[relative] = record
    return result


def _manifest_root_path(
    root: Path,
    campaign: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
) -> Path:
    root_value = manifest.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise CampaignError(f"manifest root is missing: {manifest_path}")
    candidate = Path(root_value)
    if candidate.is_absolute():
        resolved = _resolve_campaign_ref(root, campaign, root_value)
        if resolved is None:
            raise CampaignError(f"manifest root is unavailable: {root_value}")
        return resolved
    return ensure_within(root, root / candidate)


def _manifest_watch_state(
    root: Path,
    campaign: dict[str, Any],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Return cheap file-state inputs used to invalidate cached validation."""

    value = load_json(manifest_path)
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        raise CampaignError(f"invalid artifact manifest: {manifest_path}")
    source_root = _manifest_root_path(root, campaign, manifest_path, value)
    states: list[dict[str, Any]] = []
    for relative in sorted(
        _manifest_records(manifest_path, expected_campaign_id=str(campaign["campaign_id"]))
    ):
        path = ensure_within(source_root, source_root / _path_from_manifest(relative))
        if path.is_symlink() or not path.is_file():
            states.append({"path": relative, "missing": True})
            continue
        stat = path.stat()
        states.append({"path": relative, "size": stat.st_size, "sha256": sha256_file(path)})
    return states


def _directory_content_state(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_dir():
        raise CampaignError(f"evidence root must be a regular directory: {path}")
    return [
        {"path": relative, "size": file_path.stat().st_size, "sha256": sha256_file(file_path)}
        for file_path, relative in iter_regular_files(path)
    ]


def _validate_manifest_files(
    root: Path,
    campaign: dict[str, Any],
    manifest_path: Path,
    *,
    github_rules: bool = False,
) -> list[str]:
    errors: list[str] = []
    value = load_json(manifest_path)
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        return [f"invalid artifact manifest: {manifest_path}"]
    try:
        source_root = _manifest_root_path(root, campaign, manifest_path, value)
        allowed = {"github"} if github_rules else {"derived", "local", "raw"}
        records = _manifest_records(
            manifest_path,
            expected_campaign_id=str(campaign["campaign_id"]),
            allowed_classes=allowed,
        )
    except (CampaignError, OSError, ValueError) as exc:
        return [str(exc)]
    for relative, record in records.items():
        if github_rules:
            reason = _github_forbidden(relative)
            if reason:
                errors.append(f"GitHub manifest contains prohibited {relative}: {reason}")
                continue
        try:
            path = ensure_within(source_root, source_root / _path_from_manifest(relative))
        except (OSError, ValueError) as exc:
            errors.append(f"manifest path is unsafe: {relative} ({exc})")
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f"manifest file is missing or not regular: {relative}")
            continue
        if path.stat().st_size != record.get("size"):
            errors.append(f"manifest size mismatch: {relative}")
            continue
        if sha256_file(path) != record.get("sha256"):
            errors.append(f"manifest SHA256 mismatch: {relative}")
    return errors


def _cmd_diff_manifest(args: argparse.Namespace) -> dict[str, Any]:
    left = Path(args.left).resolve()
    right = Path(args.right).resolve()
    left_records = _manifest_records(left)
    right_records = _manifest_records(right)
    added = sorted(set(right_records) - set(left_records))
    removed = sorted(set(left_records) - set(right_records))
    changed = sorted(
        path
        for path in set(left_records) & set(right_records)
        if (left_records[path].get("sha256"), left_records[path].get("size"))
        != (right_records[path].get("sha256"), right_records[path].get("size"))
    )
    return {
        "status": "compared",
        "left": str(left),
        "right": str(right),
        "added": added,
        "removed": removed,
        "changed": changed,
        "identical": not (added or removed or changed),
    }


def _actual_hash(root: Path, campaign: dict[str, Any], reference: str | None) -> tuple[str | None, str | None]:
    if not reference:
        return None, None
    try:
        path = _resolve_campaign_ref(root, campaign, reference)
    except (OSError, ValueError) as exc:
        return None, f"referenced file does not exist: {reference} ({exc})"
    if path is None or not path.is_file():
        return None, f"referenced file does not exist: {reference}"
    return sha256_file(path), None


def _validation_inputs(root: Path, campaign_path: Path, campaign: dict[str, Any], level: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    configs: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for prefix in ("requested", "resolved"):
        path_ref = campaign.get(f"{prefix}_config_path")
        declared = campaign.get(f"{prefix}_config_sha256")
        if declared is not None:
            try:
                validate_sha256(declared, f"{prefix}_config_sha256")
            except CampaignError as exc:
                errors.append(str(exc))
        actual, error = _actual_hash(root, campaign, path_ref)
        configs[prefix] = {"path": path_ref, "declared": declared, "actual": actual}
        if error and declared:
            errors.append(error)
        if declared and actual and declared != actual:
            errors.append(f"{prefix} config SHA256 mismatch: {path_ref}")
    for prefix in ("artifact", "raw", "github"):
        path_ref = campaign.get(f"{prefix}_manifest_path")
        declared = campaign.get(f"{prefix}_manifest_sha256")
        if declared is not None:
            try:
                validate_sha256(declared, f"{prefix}_manifest_sha256")
            except CampaignError as exc:
                errors.append(str(exc))
        actual, error = _actual_hash(root, campaign, path_ref)
        manifests[prefix] = {"path": path_ref, "declared": declared, "actual": actual, "watch": None}
        if error and declared:
            errors.append(error)
        if declared and actual and declared != actual:
            errors.append(f"{prefix} manifest SHA256 mismatch: {path_ref}")
        if actual and path_ref:
            try:
                manifest_path = _resolve_campaign_ref(root, campaign, path_ref)
                if manifest_path is not None:
                    manifests[prefix]["watch"] = _manifest_watch_state(root, campaign, manifest_path)
            except (CampaignError, OSError, ValueError) as exc:
                errors.append(f"cannot inspect {prefix} manifest files: {exc}")
    staging_state: dict[str, Any] | None = None
    staging_ref = campaign.get("staging_path") or (campaign.get("paths") or {}).get("staging")
    if staging_ref:
        try:
            staging = _resolve_campaign_ref(root, campaign, staging_ref)
            if staging is not None and staging.exists():
                head = subprocess.run(
                    ["git", "-C", str(staging), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                status = subprocess.run(
                    ["git", "-C", str(staging), "status", "--porcelain=v1", "--untracked-files=all"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                staging_state = {"path": str(staging_ref), "head": head, "status_sha256": canonical_json_sha256(status)}
        except (OSError, ValueError, subprocess.CalledProcessError):
            staging_state = {"path": str(staging_ref), "unavailable": True}
    audit_state: dict[str, Any] | None = None
    evidence = campaign.get("submit_evidence") or {}
    audit_ref = evidence.get("batch_entry_audit_path")
    if audit_ref:
        try:
            audit_path = _resolve_campaign_ref(root, campaign, audit_ref)
            if audit_path is not None and audit_path.is_file():
                audit_state = {"path": audit_ref, "sha256": sha256_file(audit_path)}
        except (OSError, ValueError):
            audit_state = {"path": audit_ref, "unavailable": True}
    inputs = {
        "validation_level": level,
        "campaign_sha256": sha256_file(campaign_path),
        "config_sha256": configs,
        "manifest_sha256": manifests,
        "execution_git_sha": campaign.get("execution_git_sha") or "",
        "staging_state": staging_state,
        "batch_entry_audit": audit_state,
        "github_evidence": _directory_content_state(
            root / "results" / "campaigns" / str(campaign["campaign_id"]) / "artifacts"
        )
        if level in {"publish", "archive"}
        else None,
    }
    return inputs, errors


def _exit_code_is_zero(value: Any) -> bool:
    return str(value).replace(" ", "") in {"0", "0:0", "0/0"}


def _check_lite(root: Path, campaign: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        validate_git_sha(campaign.get("execution_git_sha"))
    except CampaignError as exc:
        errors.append(str(exc))
    lifecycle = campaign.get("lifecycle", "draft")
    if lifecycle not in LIFECYCLE_STATES:
        errors.append(f"unsupported lifecycle state: {lifecycle}")
    for key in ("requested_config_sha256", "resolved_config_sha256", "artifact_manifest_sha256", "raw_manifest_sha256", "github_manifest_sha256"):
        try:
            validate_sha256(campaign.get(key), key)
        except CampaignError as exc:
            errors.append(str(exc))
    local_artifacts = _artifacts_dir(root, str(campaign["campaign_id"]))
    if not local_artifacts.exists():
        errors.append(f"local artifact root does not exist: {local_artifacts}")
    return errors


def _check_submit(root: Path, campaign: dict[str, Any]) -> list[str]:
    errors = _check_lite(root, campaign)
    execution_sha = campaign.get("execution_git_sha")
    if not GIT_SHA_RE.fullmatch(str(execution_sha or "")):
        errors.append("submit requires a 40-character execution_git_sha")
    for prefix in ("requested", "resolved"):
        path_ref = campaign.get(f"{prefix}_config_path")
        declared = campaign.get(f"{prefix}_config_sha256")
        actual, error = _actual_hash(root, campaign, path_ref)
        if error:
            errors.append(error)
        if not declared:
            errors.append(f"submit requires {prefix}_config_sha256")
        if actual and declared != actual:
            errors.append(f"{prefix} config SHA256 mismatch")

    evidence = campaign.get("submit_evidence") or {}
    staging_ref = campaign.get("staging_path") or (campaign.get("paths") or {}).get("staging")
    staged_head = None
    staging_clean = False
    if not staging_ref:
        errors.append("submit requires paths.staging")
    else:
        normalized_staging = str(staging_ref).replace("\\", "/").rstrip("/")
        expected_fragment = f"/staging/{campaign['campaign_id']}/"
        if expected_fragment not in normalized_staging + "/":
            errors.append("paths.staging must be under staging/<campaign_id>/")
        try:
            staging = _resolve_campaign_ref(root, campaign, staging_ref)
        except (OSError, ValueError) as exc:
            staging = None
            errors.append(f"staging checkout is unsafe or unavailable: {exc}")
        if staging is None or not staging.is_dir():
            errors.append(f"staging checkout does not exist: {staging_ref}")
        else:
            try:
                staged_head = subprocess.run(
                    ["git", "-C", str(staging), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                status = subprocess.run(
                    ["git", "-C", str(staging), "status", "--porcelain=v1", "--untracked-files=all"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                staging_clean = not status
            except (OSError, subprocess.CalledProcessError) as exc:
                errors.append(f"unable to inspect staging checkout: {exc}")
    if staged_head != execution_sha:
        errors.append("staging HEAD does not equal execution_git_sha")
    if not staging_clean:
        errors.append("staging checkout is not clean")

    audit_path = evidence.get("batch_entry_audit_path")
    audit_passed = False
    if not audit_path:
        errors.append("submit requires batch_entry_audit_path")
    else:
        try:
            audit_file = _resolve_campaign_ref(root, campaign, audit_path)
        except (OSError, ValueError) as exc:
            audit_file = None
            errors.append(f"batch-entry audit path is unsafe or unavailable: {exc}")
        if audit_file and audit_file.is_file():
            try:
                audit_value = load_json(audit_file)
                audit_passed = (
                    isinstance(audit_value, dict)
                    and audit_value.get("schema") == "filament.hpc_batch_entry_audit.v1"
                    and audit_value.get("passed") is True
                )
            except CampaignError:
                audit_passed = False
    if not audit_passed:
        errors.append("submit requires a schema-valid passing batch-entry audit receipt")

    hpc_root = campaign.get("hpc_root") or (campaign.get("paths") or {}).get("hpc_root")
    if not hpc_root:
        errors.append("submit requires paths.hpc_root")
    else:
        normalized = str(hpc_root).replace("\\", "/").rstrip("/")
        if not normalized.endswith("/" + str(campaign["campaign_id"])) or "/campaigns/" not in normalized:
            errors.append("paths.hpc_root must be a new campaigns/<campaign_id> path")
    return errors


def _check_attempts(campaign: dict[str, Any], *, require_success: bool) -> list[str]:
    errors: list[str] = []
    attempts = campaign.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return ["at least one scheduler attempt is required"]
    has_success = False
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            errors.append(f"attempt {index + 1} must be an object")
            continue
        state = str(attempt.get("state", "")).upper()
        if state not in TERMINAL_JOB_STATES:
            errors.append(f"attempt {index + 1} is not in a terminal scheduler state")
        if state == "COMPLETED":
            if not _exit_code_is_zero(attempt.get("exit_code")):
                errors.append(f"completed attempt {index + 1} must have exit code 0:0")
            else:
                has_success = True
        if not attempt.get("job_id"):
            errors.append(f"attempt {index + 1} is missing job_id")
    if require_success and not has_success:
        errors.append("publish requires a COMPLETED/0:0 attempt")
    return errors


def _check_publish(root: Path, campaign: dict[str, Any]) -> list[str]:
    errors = _check_lite(root, campaign)
    errors.extend(_check_attempts(campaign, require_success=True))
    if not GIT_SHA_RE.fullmatch(str(campaign.get("execution_git_sha") or "")):
        errors.append("publish requires execution_git_sha")
    artifact_ref = campaign.get("artifact_manifest_path")
    artifact_hash = campaign.get("artifact_manifest_sha256")
    artifact_path = _resolve_campaign_ref(root, campaign, artifact_ref) if artifact_ref else None
    if artifact_path is None or not artifact_path.is_file() or not artifact_hash:
        errors.append("publish requires a complete derived artifact manifest")
    else:
        try:
            records = _manifest_records(
                artifact_path,
                expected_campaign_id=str(campaign["campaign_id"]),
                allowed_classes={"derived", "local"},
            )
            if not records:
                errors.append("derived artifact manifest is empty")
            errors.extend(_validate_manifest_files(root, campaign, artifact_path))
        except CampaignError as exc:
            errors.append(str(exc))
    publication = campaign.get("publication") or {}
    status = publication.get("status", "not_published")
    if status not in {"not_published", "prepared", "published"}:
        errors.append(f"unsupported publication status: {status}")
    evidence_root = root / "results" / "campaigns" / str(campaign["campaign_id"]) / "artifacts"
    if evidence_root.exists():
        try:
            for path, relative in iter_regular_files(evidence_root):
                reason = _github_forbidden(relative)
                if reason:
                    errors.append(f"GitHub evidence contains prohibited {relative}: {reason}")
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def _check_archive(root: Path, campaign: dict[str, Any]) -> list[str]:
    errors = _check_lite(root, campaign)
    errors.extend(_check_attempts(campaign, require_success=False))
    for prefix in ("raw", "artifact"):
        path_ref = campaign.get(f"{prefix}_manifest_path")
        hash_ref = campaign.get(f"{prefix}_manifest_sha256")
        path = _resolve_campaign_ref(root, campaign, path_ref) if path_ref else None
        if path is None or not path.is_file() or not hash_ref:
            errors.append(f"archive requires {prefix} manifest and hash")
        else:
            try:
                expected_classes = {"raw"} if prefix == "raw" else {"derived", "local"}
                _manifest_records(
                    path,
                    expected_campaign_id=str(campaign["campaign_id"]),
                    allowed_classes=expected_classes,
                )
                errors.extend(_validate_manifest_files(root, campaign, path))
            except CampaignError as exc:
                errors.append(str(exc))
    status = (campaign.get("publication") or {}).get("status")
    if status not in {"published", "not_published"}:
        errors.append("archive requires publication status published or not_published")
    if status == "published":
        github_ref = campaign.get("github_manifest_path")
        github_hash = campaign.get("github_manifest_sha256")
        github_path = _resolve_campaign_ref(root, campaign, github_ref) if github_ref else None
        if github_path is None or not github_path.is_file() or not github_hash:
            errors.append("published archive requires GitHub manifest and hash")
        else:
            errors.extend(_validate_manifest_files(root, campaign, github_path, github_rules=True))
        commit = (campaign.get("publication") or {}).get("github_commit")
        if not GIT_SHA_RE.fullmatch(str(commit or "")):
            errors.append("published archive requires a 40-character GitHub commit")
    hpc_root = campaign.get("hpc_root") or (campaign.get("paths") or {}).get("hpc_root")
    if not hpc_root:
        errors.append("archive requires paths.hpc_root")
    return errors


def _validation_receipt(root: Path, campaign_id: str, fingerprint: str) -> Path:
    return _artifacts_dir(root, campaign_id) / ".validation" / f"{fingerprint}.json"


def _cmd_check(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = _root(args)
    campaign_path, campaign = load_campaign(root, validate_campaign_id(args.campaign_id))
    level = args.level
    inputs, input_errors = _validation_inputs(root, campaign_path, campaign, level)
    fingerprint = canonical_json_sha256(inputs)
    receipt_path = _validation_receipt(root, args.campaign_id, fingerprint)
    if not args.no_cache and receipt_path.exists():
        receipt = load_json(receipt_path)
        if (
            isinstance(receipt, dict)
            and receipt.get("schema") == "filament.validation_receipt.v1"
            and receipt.get("campaign_id") == args.campaign_id
            and receipt.get("level") == level
            and receipt.get("fingerprint") == fingerprint
            and isinstance(receipt.get("result"), dict)
            and receipt["result"].get("fingerprint") == fingerprint
            and receipt["result"].get("campaign_id") == args.campaign_id
            and receipt["result"].get("level") == level
            and isinstance(receipt["result"].get("ok"), bool)
            and isinstance(receipt["result"].get("errors"), list)
        ):
            result = dict(receipt.get("result") or {})
            result.update({"receipt": _relative_repo_path(root, receipt_path), "reused": True, "fingerprint": fingerprint})
            return result, 0 if result.get("ok") else 1

    errors = list(input_errors)
    if level == "lite":
        errors.extend(_check_lite(root, campaign))
    elif level == "submit":
        errors.extend(_check_submit(root, campaign))
    elif level == "publish":
        errors.extend(_check_publish(root, campaign))
    elif level == "archive":
        errors.extend(_check_archive(root, campaign))
    else:
        raise CampaignError(f"unsupported validation level: {level}")
    # Preserve order while avoiding duplicate error messages from the input and
    # level-specific passes.
    errors = list(dict.fromkeys(errors))
    result = {
        "status": "passed" if not errors else "failed",
        "ok": not errors,
        "level": level,
        "campaign_id": args.campaign_id,
        "fingerprint": fingerprint,
        "errors": errors,
        "reused": False,
        "receipt": _relative_repo_path(root, receipt_path),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        receipt_path,
        {
            "schema": "filament.validation_receipt.v1",
            "campaign_id": args.campaign_id,
            "level": level,
            "fingerprint": fingerprint,
            "result": result,
        },
    )
    return result, 0 if not errors else 1


def _validate_allow_pattern(pattern: str) -> str:
    normalized = normalize_relative(pattern)
    if normalized in {"*", "**"} or normalized.startswith("**"):
        raise CampaignError("publish-plan requires a specific non-global allow pattern")
    return normalized


def _cmd_publish_plan(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = _root(args)
    campaign_id = validate_campaign_id(args.campaign_id)
    campaign_path, campaign = load_campaign(root, campaign_id)
    source = _manifest_source(root, campaign_id, args.source, campaign)
    patterns = [_validate_allow_pattern(pattern) for pattern in args.allow]
    if not patterns:
        raise CampaignError("publish-plan requires at least one --allow pattern")
    manifest_path = None
    if args.manifest:
        candidate = Path(args.manifest)
        manifest_path = ensure_within(root, candidate if candidate.is_absolute() else root / candidate)
    elif campaign.get("artifact_manifest_path"):
        manifest_path = _resolve_campaign_ref(root, campaign, campaign["artifact_manifest_path"])
    records: dict[str, dict[str, Any]]
    if manifest_path and manifest_path.is_file():
        records = _manifest_records(
            manifest_path,
            expected_campaign_id=campaign_id,
            allowed_classes={"derived", "local"},
        )
    else:
        ephemeral = _collect_manifest(root, campaign_id, source, "derived")
        records = {str(record["path"]): record for record in ephemeral["files"]}

    selected = sorted(path for path in records if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns))
    if not selected:
        raise CampaignError("allow patterns selected no files")
    if args.apply:
        gate_args = argparse.Namespace(
            campaign_id=campaign_id,
            level="publish",
            no_cache=False,
            root=root,
        )
        gate, gate_code = _cmd_check(gate_args)
        if gate_code != 0:
            return {
                "status": "failed",
                "ok": False,
                "campaign_id": campaign_id,
                "dry_run": False,
                "errors": ["publish validation gate failed"],
                "validation": gate,
                "actions": [],
            }, 1
    actions: list[dict[str, Any]] = []
    errors: list[str] = []
    destination_root = root / "results" / "campaigns" / campaign_id / "artifacts"
    if destination_root.is_symlink():
        raise CampaignError(f"publish destination root may not be a symlink: {destination_root}")
    destination_root = ensure_within(root, destination_root, allow_missing=True)
    for relative in selected:
        reason = _github_forbidden(relative)
        if reason:
            errors.append(f"selected file is not publishable: {relative} ({reason})")
            continue
        source_path = ensure_within(source, source / _path_from_manifest(relative))
        record = records[relative]
        actual_size = source_path.stat().st_size
        actual_hash = sha256_file(source_path)
        if actual_size != record.get("size") or actual_hash != record.get("sha256"):
            errors.append(f"source does not match manifest: {relative}")
            continue
        destination = ensure_within(root, destination_root / _path_from_manifest(relative), allow_missing=True)
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                errors.append(f"destination is not a regular file: {relative}")
            elif sha256_file(destination) != actual_hash:
                errors.append(f"refusing to overwrite differing destination: {relative}")
            else:
                actions.append({"path": relative, "action": "existing_same", "sha256": actual_hash, "size": actual_size})
        else:
            actions.append({"path": relative, "action": "copy" if args.apply else "would_copy", "sha256": actual_hash, "size": actual_size})
    if errors:
        result = {"status": "failed", "ok": False, "campaign_id": campaign_id, "dry_run": not args.apply, "errors": errors, "actions": actions}
        return result, 1
    if args.apply:
        for action in actions:
            if action["action"] != "copy":
                continue
            relative = action["path"]
            source_path = ensure_within(source, source / _path_from_manifest(relative))
            destination = ensure_within(root, destination_root / _path_from_manifest(relative), allow_missing=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
            temporary = Path(temporary_name)
            try:
                with open(fd, "wb", closefd=True) as handle, source_path.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, handle)
                if sha256_file(temporary) != action["sha256"]:
                    raise CampaignError(f"temporary copy hash mismatch: {relative}")
                try:
                    os.link(temporary, destination)
                except FileExistsError as exc:
                    raise CampaignError(f"destination appeared during publish: {relative}") from exc
            finally:
                if temporary.exists():
                    temporary.unlink()
        campaign.setdefault("publication", {}).setdefault("selected_files", [])
        campaign["publication"]["selected_files"] = [
            {"path": action["path"], "sha256": action["sha256"], "size": action.get("size")} for action in actions
        ]
        campaign["publication"]["status"] = "prepared"
        _write_campaign(campaign_path, campaign)
    result = {"status": "applied" if args.apply else "dry_run", "ok": True, "campaign_id": campaign_id, "dry_run": not args.apply, "actions": actions, "allow": patterns}
    return result, 0


def _cmd_register_legacy(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args)
    inventory_path = Path(args.inventory)
    inventory_path = ensure_within(root, inventory_path if inventory_path.is_absolute() else root / inventory_path)
    inventory = load_json(inventory_path)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("files"), list):
        raise CampaignError("repository inventory must contain a files list")
    prefix = "Filament_python/results/"
    groups: dict[str, dict[str, Any]] = {}
    config_paths: dict[str, list[str]] = {}
    for entry in inventory["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        path = entry["path"].replace("\\", "/")
        if path.startswith("Filament_python/configs/"):
            config_remainder = path[len("Filament_python/configs/"):]
            if "/" in config_remainder:
                config_top = config_remainder.split("/", 1)[0]
                config_paths.setdefault(config_top, []).append(path)
        if not path.startswith(prefix):
            continue
        remainder = path[len(prefix):]
        if not remainder or "/" not in remainder:
            continue
        top = remainder.split("/", 1)[0]
        bucket = groups.setdefault(top, {"file_count": 0, "total_bytes": 0, "paths": []})
        bucket["file_count"] += 1
        bucket["total_bytes"] += int(entry.get("size", 0))
        bucket["paths"].append(path)
    if len(groups) != 18:
        raise CampaignError(f"expected 18 top-level Filament_python/results directories, found {len(groups)}")
    inventory_hash = sha256_file(inventory_path)
    entries = []
    for name in sorted(groups):
        relative = f"Filament_python/results/{name}"
        hpc_paths: set[str] = set()
        result_root = root / Path(*PurePosixPath(relative).parts)
        if result_root.is_dir():
            for json_path in sorted(result_root.rglob("*.json")):
                if json_path.is_symlink() or not json_path.is_file():
                    continue
                try:
                    json_value = load_json(json_path)
                except CampaignError:
                    continue
                for _, text_value in iter_json_strings(json_value):
                    if text_value.startswith("/data/run01/scvi806/user_Wangjimin/"):
                        hpc_paths.add(text_value)
        acceptance_tokens = ("final_report", "final_classification", "final_summary", "project_state")
        acceptance_candidates = sorted(
            path for path in groups[name]["paths"]
            if any(token in PurePosixPath(path).name.casefold() for token in acceptance_tokens)
        )
        entries.append({
            "legacy_id": f"legacy_filament_python_results_{name}",
            "legacy_path": relative,
            "file_count": groups[name]["file_count"],
            "total_bytes": groups[name]["total_bytes"],
            "status": "legacy_unclassified",
            "scientific_acceptance": None,
            "acceptance_evidence_candidates": acceptance_candidates,
            "config_paths": sorted(config_paths.get(name, [])),
            "known_hpc_paths": sorted(hpc_paths),
            "inventory_sha256": inventory_hash,
            "inventory_repo_head_sha": inventory.get("repo_head_sha"),
            "inventory_generated_at": inventory.get("generated_at"),
        })
    registry = {
        "schema": REGISTRY_SCHEMA,
        "source_inventory": _relative_repo_path(root, inventory_path),
        "source_inventory_sha256": inventory_hash,
        "repository_head_sha": inventory.get("repo_head_sha"),
        "entries": entries,
    }
    output = Path(args.output)
    output = ensure_within(root, output if output.is_absolute() else root / output, allow_missing=True)
    if output.exists() and not args.overwrite:
        existing = load_json(output)
        if existing != registry:
            raise CampaignError(f"refusing to overwrite differing legacy registry: {output}")
    write_json(output, registry)
    return {"status": "registered", "path": _relative_repo_path(root, output), "entry_count": len(entries), "inventory_sha256": inventory_hash}


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=REPO_DEFAULT, help="repository root (default: current directory)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize a campaign")
    init.add_argument("campaign_id")
    init.add_argument("--title", default="")
    init.add_argument("--scientific-purpose", default="")
    init.add_argument("--execution-git-sha", default=None)
    _add_root(init)
    init.set_defaults(handler=_cmd_init)

    config = subparsers.add_parser("publish-config", help="copy a safe, de-environmented config")
    config.add_argument("campaign_id")
    config.add_argument("input", help="source JSON config")
    config.add_argument("--kind", choices=("requested", "resolved"), default="requested")
    config.add_argument("--output", default=None)
    config.add_argument("--overwrite", action="store_true")
    _add_root(config)
    config.set_defaults(handler=_cmd_publish_config)

    manifest = subparsers.add_parser("build-manifest", help="build a deterministic artifact manifest")
    manifest.add_argument("campaign_id")
    manifest.add_argument("--source", default=None)
    manifest.add_argument("--output", default=None)
    manifest.add_argument("--artifact-class", "--class", dest="artifact_class", choices=("derived", "local", "raw", "github"), default="derived")
    manifest.add_argument("--overwrite", action="store_true")
    _add_root(manifest)
    manifest.set_defaults(handler=_cmd_build_manifest)

    check = subparsers.add_parser("check", help="run a cached, level-specific validation")
    check.add_argument("campaign_id")
    check.add_argument("--level", choices=("lite", "submit", "publish", "archive"), default="lite")
    check.add_argument("--no-cache", action="store_true")
    _add_root(check)
    check.set_defaults(handler=_cmd_check)

    diff = subparsers.add_parser("diff-manifest", help="compare two manifests")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.set_defaults(handler=lambda args: (_cmd_diff_manifest(args), 0))

    plan = subparsers.add_parser("publish-plan", help="dry-run or apply an explicit evidence allowlist")
    plan.add_argument("campaign_id")
    plan.add_argument("--source", default=None)
    plan.add_argument("--manifest", default=None)
    plan.add_argument("--allow", action="append", default=[])
    plan.add_argument("--apply", action="store_true")
    _add_root(plan)
    plan.set_defaults(handler=_cmd_publish_plan)

    legacy = subparsers.add_parser("register-legacy", help="register historical results without moving them")
    legacy.add_argument("--inventory", default="docs/repo_layout/repository_inventory.json")
    legacy.add_argument("--output", default="results/campaigns/legacy_registry.json")
    legacy.add_argument("--overwrite", action="store_true")
    _add_root(legacy)
    legacy.set_defaults(handler=_cmd_register_legacy)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            payload, code = result
        else:
            payload, code = result, 0
        _json_output(payload)
        return code
    except CampaignError as exc:
        print(f"campaign error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"campaign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

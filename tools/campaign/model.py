"""Data model and validation primitives for Filament_1 campaigns."""

from __future__ import annotations

import datetime as _datetime
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


CAMPAIGN_SCHEMA = "filament.campaign.v1"
MANIFEST_SCHEMA = "filament.artifact_manifest.v1"
REGISTRY_SCHEMA = "filament.legacy_registry.v1"
CAMPAIGN_ID_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<body>[a-z0-9]+(?:[_-][a-z0-9]+)+)_v(?P<version>\d{2})$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

LIFECYCLE_STATES = (
    "draft",
    "staged",
    "submitted",
    "completed",
    "failed",
    "postprocessed",
    "reviewed",
    "published",
    "archived",
)

TERMINAL_JOB_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
}


class CampaignError(ValueError):
    """A user-correctable campaign-management validation error."""


def validate_campaign_id(value: str) -> str:
    """Validate and return a campaign ID.

    The body requires at least a topic and variant component, while allowing
    additional hyphen/underscore-separated words such as
    ``hybrid_zstart0p60``.
    """

    match = CAMPAIGN_ID_RE.fullmatch(value)
    if not match:
        raise CampaignError(
            "campaign_id must match YYYYMMDD_<topic>_<variant>_vNN "
            "using lowercase letters, digits, '-' and '_'"
        )
    try:
        _datetime.datetime.strptime(match.group("date"), "%Y%m%d")
    except ValueError as exc:
        raise CampaignError(f"campaign_id contains an invalid date: {value}") from exc
    return value


def validate_sha256(value: str | None, field: str = "sha256") -> None:
    if value is not None and value != "" and not SHA256_RE.fullmatch(value):
        raise CampaignError(f"{field} must be a 64-character lowercase SHA256")


def validate_git_sha(value: str | None, field: str = "execution_git_sha") -> None:
    if value is not None and value != "" and not GIT_SHA_RE.fullmatch(value):
        raise CampaignError(f"{field} must be a 40-character lowercase Git SHA")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise CampaignError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CampaignError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    """Write UTF-8 JSON with stable formatting.

    The CLI uses this for generated metadata only.  Input scientific config
    files are read separately and are never passed to this function.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise CampaignError(f"refusing to overwrite existing file: {path}")
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_campaign(root: Path, campaign_id: str) -> tuple[Path, dict[str, Any]]:
    validate_campaign_id(campaign_id)
    path = root / "results" / "campaigns" / campaign_id / "campaign.json"
    value = load_json(path)
    if not isinstance(value, dict):
        raise CampaignError(f"campaign JSON must be an object: {path}")
    if value.get("schema") != CAMPAIGN_SCHEMA:
        raise CampaignError(f"unsupported campaign schema in {path}")
    if value.get("campaign_id") != campaign_id:
        raise CampaignError(f"campaign_id mismatch in {path}")
    return path, value


def secret_key_reason(key: str) -> str | None:
    """Return a reason when a JSON key looks credential-bearing."""

    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    exact = {
        "token",
        "accesstoken",
        "refreshtoken",
        "password",
        "passwd",
        "secret",
        "credential",
        "credentials",
        "apikey",
        "privatekey",
        "authorization",
        "proxy",
        "proxyurl",
        "authurl",
    }
    if normalized in exact:
        return f"secret-like key: {key}"
    if any(
        term in normalized
        for term in (
            "password",
            "passwd",
            "token",
            "credential",
            "secret",
            "apikey",
            "privatekey",
            "accesskey",
            "bearer",
            "oauth",
        )
    ):
        return f"secret-like key: {key}"
    if normalized in {"pat", "githubpat", "auth", "authentication"}:
        return f"secret-like key: {key}"
    return None


def iter_json_strings(value: Any, path: str = ""):
    """Yield ``(JSON path, value)`` for every string in a JSON value."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from iter_json_strings(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_json_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def make_initial_campaign(campaign_id: str, *, title: str = "", scientific_purpose: str = "", execution_git_sha: str | None = None) -> dict[str, Any]:
    validate_campaign_id(campaign_id)
    validate_git_sha(execution_git_sha)
    config_root = f"results/campaigns/{campaign_id}/configs"
    return {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": campaign_id,
        "title": title,
        "scientific_purpose": scientific_purpose,
        "lifecycle": "draft",
        "execution_git_sha": execution_git_sha,
        "execution_ref": None,
        "repository_url": None,
        "requested_config_path": f"{config_root}/requested/config.json",
        "resolved_config_path": f"{config_root}/resolved/config.json",
        "requested_config_sha256": None,
        "resolved_config_sha256": None,
        "artifact_manifest_path": None,
        "artifact_manifest_sha256": None,
        "raw_manifest_path": None,
        "raw_manifest_sha256": None,
        "github_manifest_path": None,
        "github_manifest_sha256": None,
        "paths": {
            "repository_root": ".",
            "local_artifacts": f".artifacts/{campaign_id}",
            "github_results": f"results/campaigns/{campaign_id}",
            "hpc_root": None,
            "staging": None,
        },
        "attempts": [],
        "publication": {
            "status": "not_published",
            "github_commit": None,
            "selected_files": [],
        },
        "legacy": None,
    }

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.campaign import manage


CAMPAIGN = "20260824_integration_metrics_v01"
EXECUTION_SHA = "0123456789abcdef0123456789abcdef01234567"


def run_cli(root: Path, *args: str) -> int:
    return manage.main([*args, "--root", str(root)])


def test_init_manifest_check_and_allowlisted_publish(tmp_path: Path) -> None:
    assert run_cli(tmp_path, "init", CAMPAIGN, "--execution-git-sha", EXECUTION_SHA) == 0
    input_config = tmp_path / "input.json"
    input_config.write_text('{"physics": {"wavelength": 8e-7}, "cache_dir": "cache/ionization"}\n', encoding="utf-8")
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "requested") == 0
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "resolved") == 0

    metrics = tmp_path / ".artifacts" / CAMPAIGN / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "summary.json").write_text('{"finite": true}\n', encoding="utf-8")
    (metrics / "notes.md").write_text("review", encoding="utf-8")
    assert run_cli(tmp_path, "build-manifest", CAMPAIGN) == 0
    assert run_cli(tmp_path, "check", CAMPAIGN, "--level", "lite") == 0

    campaign_path = tmp_path / "results" / "campaigns" / CAMPAIGN / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["attempts"] = [{"job_id": "12345", "state": "COMPLETED", "exit_code": "0:0"}]
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Explicit allowlist is required and dry-run does not create results.
    assert run_cli(tmp_path, "publish-plan", CAMPAIGN, "--allow", "metrics/*.json") == 0
    assert not (tmp_path / "results" / "campaigns" / CAMPAIGN / "artifacts" / "metrics" / "summary.json").exists()
    assert run_cli(tmp_path, "publish-plan", CAMPAIGN, "--allow", "metrics/*.json", "--apply") == 0
    published = tmp_path / "results" / "campaigns" / CAMPAIGN / "artifacts" / "metrics" / "summary.json"
    assert published.read_text(encoding="utf-8") == '{"finite": true}\n'

    # A changed destination is never overwritten by --apply.
    published.write_text("changed", encoding="utf-8")
    assert run_cli(tmp_path, "publish-plan", CAMPAIGN, "--allow", "metrics/*.json", "--apply") == 1


def test_publish_validation_cache_invalidates_when_artifact_changes(tmp_path: Path) -> None:
    assert run_cli(tmp_path, "init", CAMPAIGN, "--execution-git-sha", EXECUTION_SHA) == 0
    input_config = tmp_path / "input.json"
    input_config.write_text('{"physics": {"wavelength": 8e-7}}\n', encoding="utf-8")
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "requested") == 0
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "resolved") == 0
    artifact = tmp_path / ".artifacts" / CAMPAIGN / "summary.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    assert run_cli(tmp_path, "build-manifest", CAMPAIGN) == 0
    campaign_path = tmp_path / "results" / "campaigns" / CAMPAIGN / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["attempts"] = [{"job_id": "12345", "state": "COMPLETED", "exit_code": "0:0"}]
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_cli(tmp_path, "check", CAMPAIGN, "--level", "publish") == 0
    artifact.write_text('{"ok": false}\n', encoding="utf-8")
    assert run_cli(tmp_path, "check", CAMPAIGN, "--level", "publish") == 1
    receipts = list((tmp_path / ".artifacts" / CAMPAIGN / ".validation").glob("*.json"))
    assert len(receipts) == 2


def test_failed_only_campaign_cannot_pass_publish(tmp_path: Path) -> None:
    assert run_cli(tmp_path, "init", CAMPAIGN, "--execution-git-sha", EXECUTION_SHA) == 0
    input_config = tmp_path / "input.json"
    input_config.write_text('{"physics": {"wavelength": 8e-7}}\n', encoding="utf-8")
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "requested") == 0
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "resolved") == 0
    artifact = tmp_path / ".artifacts" / CAMPAIGN / "summary.json"
    artifact.write_text('{"technical_failure": true}\n', encoding="utf-8")
    assert run_cli(tmp_path, "build-manifest", CAMPAIGN) == 0
    campaign_path = tmp_path / "results" / "campaigns" / CAMPAIGN / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["attempts"] = [{"job_id": "12345", "state": "FAILED", "exit_code": "1:0"}]
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_cli(tmp_path, "check", CAMPAIGN, "--level", "publish") == 1


def test_submit_ignores_self_reported_clean_evidence_and_checks_live_repo(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / CAMPAIGN / "Filament_1_fixture"
    staging.mkdir(parents=True)
    subprocess.run(["git", "-C", str(staging), "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(staging), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(staging), "config", "user.email", "fixture@example.invalid"], check=True)
    (staging / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(staging), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(staging), "commit", "-m", "fixture"], check=True, capture_output=True)
    head = subprocess.run(
        ["git", "-C", str(staging), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert run_cli(tmp_path, "init", CAMPAIGN, "--execution-git-sha", head) == 0
    input_config = tmp_path / "input.json"
    input_config.write_text('{"physics": {"wavelength": 8e-7}}\n', encoding="utf-8")
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "requested") == 0
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "resolved") == 0
    audit = tmp_path / "batch-audit.json"
    audit.write_text(
        json.dumps({"schema": "filament.hpc_batch_entry_audit.v1", "passed": True}) + "\n",
        encoding="utf-8",
    )
    campaign_path = tmp_path / "results" / "campaigns" / CAMPAIGN / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["paths"]["staging"] = str(staging)
    campaign["paths"]["hpc_root"] = str(tmp_path / "campaigns" / CAMPAIGN)
    campaign["submit_evidence"] = {
        "staging_head_sha": head,
        "staging_clean": True,
        "batch_entry_audit_passed": True,
        "batch_entry_audit_path": str(audit),
    }
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    assert run_cli(tmp_path, "check", CAMPAIGN, "--level", "submit") == 1


def test_publish_plan_rejects_symlink_destination_root_when_supported(tmp_path: Path) -> None:
    assert run_cli(tmp_path, "init", CAMPAIGN, "--execution-git-sha", EXECUTION_SHA) == 0
    input_config = tmp_path / "input.json"
    input_config.write_text('{"physics": {"wavelength": 8e-7}}\n', encoding="utf-8")
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "requested") == 0
    assert run_cli(tmp_path, "publish-config", CAMPAIGN, str(input_config), "--kind", "resolved") == 0
    artifact = tmp_path / ".artifacts" / CAMPAIGN / "metrics" / "summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    assert run_cli(tmp_path, "build-manifest", CAMPAIGN) == 0
    campaign_path = tmp_path / "results" / "campaigns" / CAMPAIGN / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["attempts"] = [{"job_id": "12345", "state": "COMPLETED", "exit_code": "0:0"}]
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "results" / "campaigns" / CAMPAIGN / "artifacts"
    try:
        destination.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        return
    assert run_cli(tmp_path, "publish-plan", CAMPAIGN, "--allow", "metrics/*.json", "--apply") == 2


def test_register_legacy_is_mechanical_and_complete(tmp_path: Path) -> None:
    inventory_path = tmp_path / "docs" / "repo_layout" / "repository_inventory.json"
    inventory_path.parent.mkdir(parents=True)
    files = []
    for index in range(18):
        name = f"result_{index:02d}"
        files.append({"path": f"Filament_python/results/{name}/summary.json", "size": index + 1})
        files.append({"path": f"Filament_python/results/{name}/figure.png", "size": 10})
    inventory_path.write_text(
        json.dumps(
            {
                "schema": "filament_1.repository_inventory.v1",
                "generated_at": "2026-08-23T00:00:00+00:00",
                "repo_head_sha": "a" * 40,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    assert run_cli(tmp_path, "register-legacy") == 0
    registry_path = tmp_path / "results" / "campaigns" / "legacy_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["schema"] == "filament.legacy_registry.v1"
    assert len(registry["entries"]) == 18
    assert {entry["status"] for entry in registry["entries"]} == {"legacy_unclassified"}
    assert all(entry["scientific_acceptance"] is None for entry in registry["entries"])

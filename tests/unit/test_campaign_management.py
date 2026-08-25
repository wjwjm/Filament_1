from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.campaign import manage
from tools.campaign.model import CampaignError, validate_campaign_id


def test_campaign_id_shape_and_date() -> None:
    assert validate_campaign_id("20260824_hybrid_zstart0p60_v01") == "20260824_hybrid_zstart0p60_v01"
    with pytest.raises(CampaignError):
        validate_campaign_id("20261324_hybrid_zstart0p60_v01")
    with pytest.raises(CampaignError):
        validate_campaign_id("20260824_Hybrid_zstart0p60_v01")
    with pytest.raises(CampaignError):
        validate_campaign_id("20260824_hybrid_v1")


def test_config_publish_rejects_secrets_paths_and_authenticated_urls() -> None:
    violations = manage._config_violations(
        {
            "physics": {"wavelength": 800e-9},
            "api_key": "not-a-real-key",
            "api_secret": "not-a-real-secret",
            "github_pat": "not-a-real-pat",
            "private_key_pem": "not-a-real-key",
            "access_key": "not-a-real-access-key",
            "nested": {"input_path": "/data/run01/input.json"},
            "endpoint": "https://user:password@example.invalid/api",
        }
    )
    assert any("api_key" in violation for violation in violations)
    assert any("api_secret" in violation for violation in violations)
    assert any("github_pat" in violation for violation in violations)
    assert any("private_key_pem" in violation for violation in violations)
    assert any("access_key" in violation for violation in violations)
    assert any("absolute path" in violation for violation in violations)
    assert any("authenticated URL" in violation for violation in violations)


def test_manifest_is_sorted_and_github_class_blocks_raw_results(tmp_path: Path) -> None:
    source = tmp_path / "artifacts"
    source.mkdir()
    (source / "z.json").write_text("z", encoding="utf-8")
    (source / "a.json").write_text("a", encoding="utf-8")
    manifest = manage._collect_manifest(tmp_path, "20260824_test_case_v01", source, "derived")
    assert [record["path"] for record in manifest["files"]] == ["a.json", "z.json"]

    (source / "bad.npz").write_bytes(b"raw")
    with pytest.raises(CampaignError):
        manage._collect_manifest(tmp_path, "20260824_test_case_v01", source, "github")


def test_manifest_paths_use_native_parts_without_backslash_filenames() -> None:
    path = manage._path_from_manifest("metrics/subdir/summary.json")
    assert path.parts == ("metrics", "subdir", "summary.json")


def test_manifest_campaign_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "filament.artifact_manifest.v1",
                "campaign_id": "20260825_other_case_v01",
                "artifact_class": "derived",
                "root": ".artifacts/20260825_other_case_v01",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CampaignError):
        manage._manifest_records(
            manifest,
            expected_campaign_id="20260825_expected_case_v01",
            allowed_classes={"derived"},
        )


def test_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    source = tmp_path / "artifacts"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = source / "outside.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this Windows runner")
    with pytest.raises(ValueError):
        list(manage.iter_regular_files(source))


def test_validation_receipt_is_reused(tmp_path: Path) -> None:
    campaign_id = "20260824_test_receipt_v01"
    assert manage.main(["init", campaign_id, "--root", str(tmp_path)]) == 0
    assert manage.main(["check", campaign_id, "--level", "lite", "--root", str(tmp_path)]) == 0
    # The second invocation should use the same receipt rather than producing a
    # second validation fingerprint.
    assert manage.main(["check", campaign_id, "--level", "lite", "--root", str(tmp_path)]) == 0
    receipts = list((tmp_path / ".artifacts" / campaign_id / ".validation").glob("*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["result"]["ok"] is True


def test_forged_validation_receipt_is_not_reused(tmp_path: Path) -> None:
    campaign_id = "20260825_test_forged_v01"
    assert manage.main(["init", campaign_id, "--root", str(tmp_path)]) == 0
    campaign_path = tmp_path / "results" / "campaigns" / campaign_id / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    inputs, _ = manage._validation_inputs(tmp_path, campaign_path, campaign, "lite")
    fingerprint = manage.canonical_json_sha256(inputs)
    receipt = tmp_path / ".artifacts" / campaign_id / ".validation" / f"{fingerprint}.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"result":{"ok":true}}\n', encoding="utf-8")
    assert manage.main(["check", campaign_id, "--level", "lite", "--root", str(tmp_path)]) == 0
    repaired = json.loads(receipt.read_text(encoding="utf-8"))
    assert repaired["schema"] == "filament.validation_receipt.v1"
    assert repaired["fingerprint"] == fingerprint


def test_repository_url_with_credentials_is_rejected() -> None:
    with pytest.raises(CampaignError):
        manage._safe_repository_url("https://token@example.invalid/repo.git")


def test_hpc_namespace_config_keeps_new_and_legacy_roots_separate() -> None:
    root = Path(__file__).resolve().parents[2]
    namespace = json.loads(
        (root / "configs" / "project_management" / "hpc_namespace.json").read_text(encoding="utf-8")
    )
    project_root = namespace["project_root"]
    assert namespace["schema"] == "filament.hpc_project_namespace.v1"
    assert project_root == "/data/run01/scvi806/user_Wangjimin/projects/Filament_1"
    assert namespace["account"] == "scvi806"
    assert all(path.startswith(project_root + "/") for path in namespace["canonical_paths"].values())
    assert namespace["new_campaign_path_template"].startswith(project_root + "/campaigns/")
    assert namespace["new_staging_path_template"].startswith(project_root + "/source/staging/")
    assert namespace["protected_legacy_paths"]["legacy_repository"] != project_root
    assert namespace["policies"] == {
        "new_jobs_from_legacy_repository": False,
        "permanent_deletion_authorized": False,
        "phase2_started": True,
        "symlinks_allowed": False,
    }
    assert namespace["cutover"]["receipt_sha256"] == (
        "51c805bc14fc9e27bc63437ce15639ae058e0e01107c24b1e3f5525340efd700"
    )


def test_hpc_legacy_relocation_batch1_paths_are_explicit_and_non_destructive() -> None:
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads(
        (root / "configs" / "project_management" / "hpc_legacy_relocation_batch1.json").read_text(
            encoding="utf-8"
        )
    )
    project_root = "/data/run01/scvi806/user_Wangjimin/projects/Filament_1"
    assert metadata["schema"] == "filament.hpc_legacy_relocation_batch.v1"
    assert metadata["status"] == "completed"
    assert metadata["permanent_deletion_authorized"] is False
    assert metadata["symlink_created"] is False
    assert len(metadata["campaigns"]) == 4
    assert len({campaign["campaign_id"] for campaign in metadata["campaigns"]}) == 4
    for campaign in metadata["campaigns"]:
        assert campaign["current_path"] == f"{project_root}/legacy/runs/{campaign['campaign_id']}"
        assert campaign["original_path"].startswith("/data/run01/scvi806/user_Wangjimin/")
        assert campaign["quarantine_source_path"].startswith(
            f"{project_root}/quarantine/relocated_legacy_sources_20260825/"
        )
        assert campaign["file_count"] > 0
        assert campaign["total_bytes"] > 0
        for field in (
            "source_manifest_sha256",
            "destination_manifest_sha256",
            "relocation_receipt_sha256",
            "archive_receipt_sha256",
        ):
            assert len(campaign[field]) == 64

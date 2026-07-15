from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import submit_stage


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "stages" / "stage1_single_pulse_optimization.json"


def test_dry_run_has_no_filesystem_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["submit_stage.py", "--spec", str(SPEC), "--run-id", "stage1_dry_run", "--dry-run"])
    monkeypatch.setattr(submit_stage.Path, "resolve", lambda self: self)
    assert submit_stage.main() == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["dry_run"] is True
    assert manifest["simulation_job_ids"] == {"40fs": None, "120fs": None}


def test_submission_commands_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = submit_stage.load_stage_spec(SPEC)
    base_path = (SPEC.parent / spec["base_config"]).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    configs = {case["case_id"]: submit_stage.build_case_config(base, case) for case in spec["cases"]}
    root, config_paths = submit_stage.prepare_stage_directory(spec, "run_test", tmp_path, configs)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ["git", "rev-parse"]: return subprocess.CompletedProcess(args, 0, "abc123\n", "")
        return subprocess.CompletedProcess(args, 0, f"{100 + len(calls)};cluster\n", "")

    monkeypatch.setattr(submit_stage.subprocess, "run", fake_run)
    jobs = submit_stage.submit_simulation_jobs(spec, root, config_paths, tmp_path)
    post = submit_stage.submit_stage_postprocess_job(spec, root, jobs, tmp_path)
    manifest = submit_stage.write_submission_manifest(spec, root, base_path, config_paths, jobs, post, tmp_path, False)
    assert set(jobs) == {"40fs", "120fs"}
    assert all("--dependency" not in command for command in calls[:2])
    assert "afterok:" in calls[2][3]
    assert "--gres=gpu:1" in calls[2] and "--cpus-per-task=4" in calls[2] and "--time=00:30:00" in calls[2]
    assert manifest["stage_postprocess_job_id"] == post
    assert manifest["paths"]["40fs"]["mat"] != manifest["paths"]["120fs"]["mat"]
    with pytest.raises(FileExistsError):
        submit_stage.prepare_stage_directory(spec, "run_test", tmp_path, configs)

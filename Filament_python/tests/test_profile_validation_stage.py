from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "build_profile_validation_cases.py"
SPEC_PATH = ROOT / "stages" / "transverse_profile_validation.json"
SPEC_40_PATH = ROOT / "stages" / "transverse_profile_validation_40fs.json"


def _load_submitter():
    spec = importlib.util.spec_from_file_location("profile_validation_submitter", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_stage_requires_one_gpu_and_escapes_export_commas(tmp_path, monkeypatch) -> None:
    submitter = _load_submitter()
    stage = submitter.load_stage_spec(SPEC_PATH)
    configs = submitter.load_and_validate_case_configs(stage, SPEC_PATH)
    root, config_paths = submitter.prepare_stage_directory(stage, "profile_stage_test", tmp_path, configs)
    commands: list[list[str]] = []

    def fake_sbatch(command, cwd):
        commands.append(command)
        return str(100 + len(commands))

    monkeypatch.setattr(submitter, "_sbatch", fake_sbatch)
    jobs = submitter.submit_simulation_jobs(stage, root, config_paths, tmp_path)
    submitter.submit_postprocess_job(stage, root, jobs, tmp_path)

    assert set(jobs) == {"profile_g_120", "profile_ft90_120"}
    for command in commands[:2]:
        assert "--gres=gpu:1" in command
        assert "--cpus-per-task=8" in command
        assert not any(item.startswith("--mem=") for item in command)
        assert "--time=08:00:00" in command
        export = next(item for item in command if item.startswith("--export="))
        assert "CASE_LABEL=Gaussian; 120 fs" in export or "CASE_LABEL=FT90; 120 fs" in export
    assert "--gres=gpu:1" in commands[2]
    assert "--cpus-per-task=4" in commands[2]
    assert not any(item.startswith("--mem=") for item in commands[2])
    assert "--time=00:30:00" in commands[2]


def test_profile_stage_40fs_uses_same_resources_and_dynamic_pulse_metadata(tmp_path, monkeypatch) -> None:
    submitter = _load_submitter()
    stage = submitter.load_stage_spec(SPEC_40_PATH)
    configs = submitter.load_and_validate_case_configs(stage, SPEC_40_PATH)
    assert {config["beam"]["tau_fwhm"] for config in configs.values()} == {4e-14}
    root, config_paths = submitter.prepare_stage_directory(stage, "profile_stage_40fs_test", tmp_path, configs)
    commands: list[list[str]] = []

    def fake_sbatch(command, cwd):
        commands.append(command)
        return str(200 + len(commands))

    monkeypatch.setattr(submitter, "_sbatch", fake_sbatch)
    jobs = submitter.submit_simulation_jobs(stage, root, config_paths, tmp_path)
    submitter.submit_postprocess_job(stage, root, jobs, tmp_path)

    assert set(jobs) == {"profile_g_40", "profile_ft90_40"}
    for command in commands[:2]:
        assert "--gres=gpu:1" in command
        assert "--cpus-per-task=8" in command
        export = next(item for item in command if item.startswith("--export="))
        assert "PULSE_WIDTH_FS=40" in export
    assert "--job-name=pv_post_profile_validation_40fs" in commands[2]


def test_40fs_configs_change_only_pulse_width_from_the_120fs_cases() -> None:
    pairs = (
        ("gaussian_120fs.json", "gaussian_40fs.json"),
        ("flat_top_90_120fs.json", "flat_top_90_40fs.json"),
    )
    config_dir = ROOT / "configs" / "profile_validation"
    for baseline_name, target_name in pairs:
        baseline = json.loads((config_dir / baseline_name).read_text(encoding="utf-8"))
        target = json.loads((config_dir / target_name).read_text(encoding="utf-8"))
        assert baseline["beam"]["tau_fwhm"] == 1.2e-13
        assert target["beam"]["tau_fwhm"] == 4e-14
        baseline["beam"]["tau_fwhm"] = target["beam"]["tau_fwhm"]
        assert baseline == target

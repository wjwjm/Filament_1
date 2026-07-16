from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_nonlinear_ablation_configs as builder


def _stage_path() -> Path:
    return ROOT / "stages" / "nonlinear_ablation_stage1.json"


def test_generator_writes_loadable_configs_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "_git_commit_sha", lambda _root: "a" * 40)
    manifest = builder.generate_ablation_configs(_stage_path(), tmp_path / "bundle")
    assert manifest["job_submission"] == "not_supported_by_this_generator"
    assert manifest["code_commit_sha"] == "a" * 40
    assert len(manifest["cases"]) == 12

    manifest_path = tmp_path / "bundle" / "nonlinear_ablation_manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["stage_id"] == "nonlinear_ablation_stage1"
    assert {row["variant"] for row in saved["cases"]} == {
        "vacuum",
        "electronic_kerr_only",
        "electronic_kerr_plus_raman_phase",
        "kerr_raman_ionization_plasma_no_loss",
        "kerr_raman_ionization_plasma_with_ionization_loss",
        "full_model",
    }
    for row in saved["cases"]:
        config = tmp_path / "bundle" / row["config_file"]
        assert config.is_file()
        assert row["output_filename"].endswith(".npz")
        assert row["effective_nonlinear_switches"]["use_ionization_solver"] == row["ionization_solver_enabled"]
        # The generator itself invokes confio.load_all(); this second call makes
        # that public compatibility requirement explicit in the test.
        builder._effective_switches(config)


def test_duration_variants_only_differ_by_pulse_width(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "_git_commit_sha", lambda _root: "b" * 40)
    manifest = builder.generate_ablation_configs(_stage_path(), tmp_path / "bundle")
    by_variant: dict[str, dict[str, dict]] = {}
    for row in manifest["cases"]:
        config = json.loads((tmp_path / "bundle" / row["config_file"]).read_text(encoding="utf-8"))
        by_variant.setdefault(row["variant"], {})[row["duration_case"]] = config

    for variant, configs in by_variant.items():
        cfg_40, cfg_120 = configs["40fs"], configs["120fs"]
        assert cfg_40["beam"]["tau_fwhm"] == 40e-15, variant
        assert cfg_120["beam"]["tau_fwhm"] == 120e-15, variant
        cfg_40["beam"] = dict(cfg_40["beam"])
        cfg_120["beam"] = dict(cfg_120["beam"])
        cfg_40["beam"].pop("tau_fwhm")
        cfg_120["beam"].pop("tau_fwhm")
        assert cfg_40 == cfg_120, variant

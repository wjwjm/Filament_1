from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_prepare_module():
    path = ROOT / "tools" / "prepare_phase8c_full_raman_test.py"
    spec = importlib.util.spec_from_file_location("phase8c_full_raman_prepare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_eq27_test_a_pair_has_only_feedback_switch_difference():
    module = _load_prepare_module()
    base = module.json.loads(module.BASE.read_text(encoding="utf-8"))
    on, off, differences = module.build(base)
    assert on["propagation"]["linear_precision_strategy"] == "mixed_precision"
    assert on["propagation"]["use_raman_full_operator"] is True
    assert off["propagation"]["use_raman_full_operator"] is False
    assert differences == [{
        "path": "propagation.use_raman_full_operator", "on": True, "off": False,
    }]


def test_full_eq27_submission_script_has_provenance_and_resource_guards():
    script = (ROOT / "tools" / "phase8c_full_raman_test.sbatch").read_text(encoding="utf-8")
    for token in (
        "#SBATCH --time=15:00:00", "#SBATCH --nodelist=g0609",
        "EXPECTED_GIT_SHA", "EXPECTED_CONFIG_SHA256", "EXPECTED_GPU_MODEL",
        "git status --porcelain", "git\", \"-C\", os.environ[\"REPO_DIR\"]",
        "CUPY_CACHE_DIR", "run_from_file",
    ):
        assert token in script

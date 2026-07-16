from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_nonlinear_switch_isolation as isolation


def test_switch_isolation_smoke_report_is_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation, "_git_sha", lambda: "c" * 40)
    report = isolation.run_switch_isolation_smoke()
    assert report["passed"] is True
    assert report["execution"]["saved_raw_npz"] is False
    assert report["code_commit_sha"] == "c" * 40
    assert [check["name"] for check in report["checks"]] == [
        "default_full_model_regression",
        "electronic_kerr_off",
        "raman_phase_off_with_absorption_on",
        "raman_absorption_off_with_phase_on",
        "plasma_phase_off",
        "ionization_loss_off",
    ]
    assert all(check["passed"] for check in report["checks"])

    path = tmp_path / "switch_isolation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["passed"] is True

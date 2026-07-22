from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reference_tool_emits_all_cases_candidates_and_repeat_lengths(tmp_path):
    tool_path = ROOT / "tools" / "compare_bk_nee_precision_strategies.py"
    spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    csv_path, json_path = tmp_path / "reference.csv", tmp_path / "reference.json"
    module.main(["--csv", str(csv_path), "--json", str(json_path)])
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(rows) == 1 + 7 * 3 * 4
    assert set(report["assessment"]) == {"baseline_complex64", "orthonormal_fft", "mixed_precision", "unitary_projection"}

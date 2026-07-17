from __future__ import annotations

import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_ionization_model_propagation_configs import (  # noqa: E402
    ALLOWED_LEAVES,
    BASE_CONFIGS,
    CASE_IDS,
    build,
    config_differences,
    make_talebpour_config,
)


def test_talebpour_transformation_is_limited_to_declared_species_fields():
    base = json.loads(BASE_CONFIGS["120fs"].read_text(encoding="utf-8"))
    generated = make_talebpour_config(base)
    assert set(config_differences(base, generated)) == ALLOWED_LEAVES
    by_name = {item["name"]: item for item in generated["ionization"]["species"]}
    assert by_name["N2"]["rate"] == by_name["O2"]["rate"] == "ppt_talebpour_i_lut"
    assert by_name["N2"]["Ip_eV_eff"] == 15.6 and by_name["N2"]["Zeff"] == 0.9
    assert by_name["O2"]["Ip_eV_eff"] == 12.55 and by_name["O2"]["Zeff"] == 0.53


def test_build_generates_both_widths_but_authorizes_no_submission(tmp_path):
    manifest = build(tmp_path / "talebpour")
    assert {item["case_id"] for item in manifest["cases"]} == set(CASE_IDS.values())
    assert all(item["submission_authorized"] is False for item in manifest["cases"])
    saved = json.loads((tmp_path / "talebpour" / "ionization_model_propagation_config_manifest.json").read_text(encoding="utf-8"))
    assert saved["execution_git_sha_requirement"] == "8dcd01ee38adf2167a2fd6083ae4785e94de89a0"
    with pytest.raises(FileExistsError):
        build(tmp_path / "talebpour")

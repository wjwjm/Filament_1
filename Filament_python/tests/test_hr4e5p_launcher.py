from __future__ import annotations

import json
from pathlib import Path

import pytest


def _payload(root: Path, *, source_sha: str = "70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467"):
    from KHz_filament.hr4e5p_launcher import P3_CASES, P3_INDICES

    root.mkdir(parents=True, exist_ok=True)
    cases = {}
    records = [{"ordinal": ordinal, "screen_id": f"source_index_{index:05d}", "source_index": index, "z_m": float(index), "source_array_sha256": f"hash-{index}"} for ordinal, index in enumerate(P3_INDICES)]
    for name, _, mode in P3_CASES:
        directory = root / name; directory.mkdir()
        state_name = "serial_state" if mode == "serial" else "parallel_state"
        (directory / f"{state_name}.json").write_text("{}", encoding="utf-8")
        (directory / "partition.json").write_text("{}", encoding="utf-8")
        (directory / "state_store").mkdir()
        value = {"schema": "khz_filament.hr4e5p.validation_input.v1", "source_manifest_sha256": "manifest", "source_state_file_sha256": source_sha, "source_state_array_sha256": "array", "dtype": "float64", "geometry": {"Nx": 301, "Ny": 351}, "screen_records": records, state_name: {"output_path": str(directory / "state_store"), "dtype": "float64"}}
        path = directory / "validation_input.json"; path.write_text(json.dumps(value), encoding="utf-8")
        cases[name] = path
    return cases


def _plan(tmp_path):
    from KHz_filament.hr4e5p_launcher import build_submission_plan, validate_p3_case_payloads

    cases = _payload(tmp_path / "cases")
    batch = tmp_path / "entry.sbatch"; batch.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return build_submission_plan(payload_validation=validate_p3_case_payloads(cases), expected_git_sha="a" * 40, repo=tmp_path, batch=batch)


def test_valid_p3_plan_reaches_mocked_sbatch_and_finalizes_receipt(tmp_path):
    from KHz_filament.hr4e5p_launcher import P3_CASES, submit_submission_plan

    seen = []
    result = submit_submission_plan(_plan(tmp_path), attempt_path=tmp_path / "attempt.json", receipt_path=tmp_path / "receipt.tsv", submitter=lambda command: (seen.append(command) or f"{100 + len(seen)};cluster"))
    assert len(seen) == len(P3_CASES)
    assert [f"--ntasks={gpu_count}" in command for command, (_, gpu_count, _) in zip(seen, P3_CASES, strict=True)] == [True] * len(P3_CASES)
    assert [item["job_id"] for item in result["jobs"]] == [str(101 + index) for index in range(len(P3_CASES))]
    assert (tmp_path / "receipt.tsv").is_file()


def test_missing_case_artifact_fails_before_mocked_sbatch(tmp_path):
    from KHz_filament.hr4e5p_launcher import build_submission_plan, validate_p3_case_payloads

    cases = _payload(tmp_path / "cases")
    validation = validate_p3_case_payloads(cases)
    next(iter(cases.values())).with_name("partition.json").unlink()
    batch = tmp_path / "entry.sbatch"; batch.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        build_submission_plan(payload_validation=validation, expected_git_sha="a" * 40, repo=tmp_path, batch=batch)


def test_failed_submission_has_no_final_receipt_or_fake_job_id(tmp_path):
    from KHz_filament.hr4e5p_launcher import submit_submission_plan

    attempt, receipt = tmp_path / "attempt.json", tmp_path / "receipt.tsv"
    with pytest.raises(RuntimeError):
        submit_submission_plan(_plan(tmp_path), attempt_path=attempt, receipt_path=receipt, submitter=lambda _: "not-a-job")
    saved = json.loads(attempt.read_text(encoding="utf-8"))
    assert saved["status"] == "FAILED" and saved["jobs"] == []
    assert not receipt.exists()


def test_strict_provenance_and_frozen_screen_set_reject_mismatch(tmp_path):
    from KHz_filament.hr4e5p_launcher import P3_INDICES, validate_p3_case_payloads

    cases = _payload(tmp_path / "bad_source", source_sha="0" * 64)
    with pytest.raises(ValueError, match="provenance"):
        validate_p3_case_payloads(cases)
    cases = _payload(tmp_path / "bad_order")
    path = next(iter(cases.values()))
    value = json.loads(path.read_text(encoding="utf-8")); value["screen_records"][0]["source_index"] = P3_INDICES[1]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/order"):
        validate_p3_case_payloads(cases)


def test_shell_launcher_declares_case_name_before_deriving_its_directory():
    source = (Path(__file__).parents[1] / "tools" / "hpc_ops" / "submit_hr4e5p_p3.sh").read_text(encoding="utf-8")
    assert 'local name="$1"\n  local workers="$2"\n  local dir="$RUN_ROOT/$name"' in source


def test_cli_wires_execution_subcommands_and_batch_interval_arguments():
    import importlib.util

    path = Path(__file__).parents[1] / "tools" / "run_hr4e5p.py"
    spec = importlib.util.spec_from_file_location("hr4e5p_cli_dispatch_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    commands = {
        "serial": ["--state", "state.json", "--out", "result.json", "--batch-intervals", "1"],
        "worker": ["--partition", "partition.json", "--out-dir", "workers"],
        "gather": ["--partition", "partition.json", "--worker-dir", "workers", "--out", "gather.json", "--batch-intervals", "1"],
        "compare": ["--input", "input.json", "--serial", "serial.json", "--gather", "gather.json", "--out", "report.json"],
    }
    for command, arguments in commands.items():
        seen = []
        setattr(module, f"command_{command}", lambda args, seen=seen: (seen.append(args), 0)[1])
        assert module.main([command, *arguments]) == 0
        assert len(seen) == 1


def test_p4_manifest_freezes_the_p3_plus_stratified_48_screen_contract(monkeypatch):
    from KHz_filament import hr4e5p_p4 as p4
    from KHz_filament.hr4e5p_launcher import P3_SOURCE_FILE_SHA256

    class Store:
        def read_authoritative_batch(self, start, stop):
            assert stop == start + 1
            payload = np.full((1, 3, 3), float(start), dtype=np.float64)
            return {field: payload for field in ("delta_n", "vx", "vy")}
        def close(self):
            pass

    import numpy as np
    monkeypatch.setattr(p4, "open_store_from_spec", lambda _: Store())
    input_value = {
        "source_state_file_sha256": P3_SOURCE_FILE_SHA256, "source_state_array_sha256": "array",
        "source_manifest_sha256": "manifest", "dtype": "float64", "geometry": {"Nx": 301, "Ny": 351},
        "parallel_state": {"unused": True},
        "screen_records": [
            {"ordinal": ordinal, "screen_id": f"source_index_{index:05d}", "source_index": index,
             "z_m": float(index), "source_array_sha256": f"hash-{index}"}
            for ordinal, index in enumerate(p4.P4_SOURCE_INDICES)
        ],
    }
    result = p4.build_input_manifest(input_value)
    assert len(result["screens"]) == 48
    assert [row["source_index"] for row in result["screens"]] == list(p4.P4_SOURCE_INDICES)
    assert set(p4.P3_INDICES).issubset({row["source_index"] for row in result["screens"]})
    assert all(row["input_delta_n_sha256"] for row in result["screens"])
    contract = p4.frozen_input_contract(result)
    assert len(contract) == 48
    altered = dict(result); altered["screens"] = [dict(row) for row in result["screens"]]
    altered["screens"][0]["input_vx_sha256"] = "bad"
    with pytest.raises(ValueError, match="per-field"):
        p4.frozen_input_contract(altered)


def test_p4_async_launcher_uses_the_p4_submitter_after_preflight():
    source = (Path(__file__).parents[1] / "tools" / "hpc_ops" / "launch_hr4e5p_p4_async.sh").read_text(encoding="utf-8")
    assert 'P4_LAUNCHER="$REPO/Filament_python/tools/hpc_ops/submit_hr4e5p_p4.sh"' in source
    assert 'bash "$P4_LAUNCHER" "$REPO" "$RUN_ROOT" "$EXPECTED_SHA" "$PREFLIGHT_OUT" "$LAUNCH_MODE"' in source


def test_p5_manifest_reuses_all_p4_screens_and_freezes_192_inputs(monkeypatch):
    from KHz_filament import hr4e5p_p5 as p5
    from KHz_filament.hr4e5p_launcher import P3_SOURCE_FILE_SHA256

    class Store:
        def read_authoritative_batch(self, start, stop):
            assert stop == start + 1
            value = np.full((1, 3, 3), float(start), dtype=np.float64)
            return {field: value for field in ("delta_n", "vx", "vy")}
        def close(self):
            pass

    import numpy as np
    monkeypatch.setattr(p5, "open_store_from_spec", lambda _: Store())
    value = {
        "source_manifest_sha256": "manifest", "source_state_file_sha256": P3_SOURCE_FILE_SHA256,
        "source_state_array_sha256": "array", "geometry": {"Nx": 301, "Ny": 351}, "dtype": "float64",
        "parallel_state": {"unused": True}, "screen_records": [
            {"ordinal": ordinal, "screen_id": f"source_index_{index:05d}", "source_index": index,
             "z_m": float(index), "source_array_sha256": f"hash-{index}"}
            for ordinal, index in enumerate(p5.P5_SOURCE_INDICES)
        ],
    }
    result = p5.build_input_manifest(value)
    assert len(result["screens"]) == 192 and result["block_size"] == 8
    assert set(p5.P4_SOURCE_INDICES).issubset({row["source_index"] for row in result["screens"]})
    assert all(row["dtype"] == "float64" for row in result["screens"])


def test_p5_async_launcher_uses_only_the_p5_submitter():
    source = (Path(__file__).parents[1] / "tools" / "hpc_ops" / "launch_hr4e5p_p5_async.sh").read_text(encoding="utf-8")
    assert 'P5_LAUNCHER="$REPO/Filament_python/tools/hpc_ops/submit_hr4e5p_p5.sh"' in source
    assert 'bash "$P5_LAUNCHER" "$REPO" "$RUN_ROOT" "$EXPECTED_SHA" "$PREFLIGHT_OUT" "$LAUNCH_MODE"' in source

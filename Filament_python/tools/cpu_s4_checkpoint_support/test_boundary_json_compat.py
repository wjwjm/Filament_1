"""Focused disk JSON compatibility tests for the R1 boundary loader."""
from __future__ import annotations

import json
import hashlib
import copy
import importlib
import math
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from . import g2_hpc_exact as exact
from .portable_binding import ensure_support_imports

ensure_support_imports()
from g2_boundary import load_boundary_mapping  # noqa: E402


def _receipt(root: Path, *, scalar_order: tuple[str, ...] = exact.DYNAMIC_SCALARS) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    mapping = load_boundary_mapping()
    mapping_names = [str(row["field_name"]) for row in mapping["rows"]]
    array_names = mapping_names[: len(exact.SCIENCE_ARRAY_FIELDS) + len(exact.POST_FIELDS)]
    diagnostic_names = mapping_names[len(array_names): -len(exact.DYNAMIC_SCALARS)]
    records = []
    native_layouts = {}
    for name in array_names + diagnostic_names:
        relative = Path(name + ".npy")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        value = np.asarray([1.0], dtype=np.float64)
        np.save(target, value, allow_pickle=False)
        layout = {"shape": [1], "dtype": value.dtype.str, "strides": [8],
                  "C": True, "F": True, "OWNDATA": True, "WRITEABLE": True,
                  "base_chain_depth": 0, "base_ndarray": False,
                  "pointer": int(value.__array_interface__["data"][0])}
        raw_digest = hashlib.sha256(value.tobytes()).hexdigest()
        records.append({
            "field_name": name,
            "snapshot_file": relative.as_posix(),
            "source_layout": dict(layout),
            "snapshot_layout": dict(layout),
            "nbytes": int(value.nbytes),
            "source_sha256_raw": raw_digest,
            "snapshot_sha256_raw": raw_digest,
            "snapshot_npy_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "snapshot_isolated": True,
            "source_snapshot_shares_memory": False,
        })
        native_layouts[name] = dict(records[-1]["snapshot_layout"])
    values = {
        "z": 1.0,
        "interval_index": 0,
        "dz_try": 0.1,
        "U0_baseline": 2.0,
        "U_now": 1.5,
        "adaptive_rejection_count": 0,
        "safety_mode_trigger_count": 0,
    }
    return {
        "schema": exact.BOUNDARY_RECEIPT_SCHEMA,
        "status": "PASS_BOUNDARY_CAPTURE_WITH_ISOLATED_SNAPSHOTS",
        "qualification_eligible": False,
        "snapshot_directory": str(root),
        "field_count": exact.EXPECTED_BOUNDARY_FIELDS,
        "arrays": records[: len(array_names)],
        "diagnostics": records[len(array_names):],
        "dynamic_scalars": {name: values[name] for name in scalar_order},
        "post_fields": list(exact.POST_FIELDS),
        "retained_live_array_refs": 0,
        "mapping_identity": dict(mapping["identity"]),
        "native_layouts": native_layouts,
        "aliases": {
            "input": [],
            "post": [],
            "science": [],
            "linear_result": {
                "E_layout": dict(native_layouts["E"]),
                "container_length": 2,
                "container_type": "tuple",
                "element0_index": 0,
                "element0_is_E": True,
                "element0_layout": dict(native_layouts["E"]),
                "element0_shares_memory_E": True,
                "expected_container_kind": "tuple",
                "expected_container_length": 2,
                "need_energy": False,
                "need_profile": True,
                "retained_live_bytes": 0,
                "retained_live_refs": 0,
                "source_candidate_lines": [1, 2],
                "source_scope": "test.boundary",
                "source_variable": "linear_result",
            },
        },
    }


def _write_receipt(path: Path, receipt: dict, *, sort_keys: bool) -> None:
    path.write_text(json.dumps(receipt, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


def _make_comparable_fixture(root: Path, *, scalar_order: tuple[str, ...] = exact.DYNAMIC_SCALARS,
                             scalar_overrides: dict[str, object] | None = None) -> dict:
    receipt = _receipt(root, scalar_order=scalar_order)
    if scalar_overrides:
        receipt["dynamic_scalars"].update(scalar_overrides)
    g1 = importlib.import_module("g1_exact")
    values = {record["field_name"]: np.load(root / record["snapshot_file"], allow_pickle=False)
              for record in receipt["arrays"] + receipt["diagnostics"]}
    bundle = {
        # compare_current_boundary_pair materializes mapping nodes in sorted
        # order; build the frozen tree with the same order for alias identity.
        "E": values["E"],
        "POST": {name: values[f"POST.{name}"] for name in sorted(exact.POST_FIELDS)},
        "Qacc": values["Qacc"],
        "Qacc_raman": values["Qacc_raman"],
        "diagnostics": {name.split(".", 1)[1]: values[name] for name in sorted(values) if name.startswith("diagnostics.")},
        "scalars": {name: exact._safe_to_value(value)
                    for name, value in sorted(receipt["dynamic_scalars"].items())},
    }
    tree_root = root.parent / (root.name + "_tree")
    g1.save_tree(tree_root, bundle, compressed=False)
    for member in tree_root.iterdir():
        shutil.copy2(member, root / member.name)
    return receipt


def _compare(reference, candidate, output: Path):
    return exact.compare_current_boundary_pair(reference, candidate, output)


def test_sorted_and_reversed_disk_receipts_load_by_scalar_name(tmp_path: Path):
    receipt = _receipt(tmp_path / "snapshot", scalar_order=tuple(reversed(exact.DYNAMIC_SCALARS)))
    sorted_path = tmp_path / "sorted.json"
    insertion_path = tmp_path / "insertion.json"
    _write_receipt(sorted_path, receipt, sort_keys=True)
    _write_receipt(insertion_path, receipt, sort_keys=False)

    expected = [exact.DYNAMIC_SCALARS.index(name) for name in exact.DYNAMIC_SCALARS]
    for path in (sorted_path, insertion_path):
        metadata = exact.inspect_boundary_metadata(path, require_members=True, header_only=True)
        assert metadata["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"
        assert metadata["canonical_scalar_order"] == list(exact.DYNAMIC_SCALARS)
        assert metadata["source_field_order"][:6] == list(exact.SCIENCE_ARRAY_FIELDS) + [f"POST.{name}" for name in exact.POST_FIELDS]
        assert metadata["canonical_field_order"] == metadata["source_field_order"][:105] + [f"scalars.{name}" for name in exact.DYNAMIC_SCALARS]
        loaded = exact.load_boundary_capture(path)
        assert list(name.split(".", 1)[1] for name in loaded["fields"] if name.startswith("scalars.")) == list(exact.DYNAMIC_SCALARS)
        assert loaded["json_compatibility"]["scalar_values_unchanged"] is True
    assert expected == list(range(7))


def test_original_comparator_accepts_memory_sorted_and_writer_sorted_orders(tmp_path: Path):
    receipt = _make_comparable_fixture(tmp_path / "baseline", scalar_order=exact.DYNAMIC_SCALARS)
    sorted_path = tmp_path / "writer_sort_keys_true.json"
    insertion_path = tmp_path / "writer_sort_keys_false.json"
    reversed_path = tmp_path / "reversed.json"
    another_order = ("U_now", "z", "adaptive_rejection_count", "dz_try", "safety_mode_trigger_count",
                     "interval_index", "U0_baseline")
    reversed_receipt = _receipt(tmp_path / "reversed_snapshot", scalar_order=tuple(reversed(exact.DYNAMIC_SCALARS)))
    another_receipt = _receipt(tmp_path / "another_snapshot", scalar_order=another_order)
    for receipt_path, payload, sort_keys in (
        (sorted_path, receipt, True),
        (insertion_path, receipt, False),
        (reversed_path, reversed_receipt, True),
    ):
        _write_receipt(receipt_path, payload, sort_keys=sort_keys)
    another_path = tmp_path / "another.json"
    _write_receipt(another_path, another_receipt, sort_keys=False)

    for index, candidate in enumerate((receipt, sorted_path, insertion_path, reversed_path, another_path)):
        result = _compare(sorted_path, candidate, tmp_path / f"exact_{index}.jsonl")
        assert result["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"


def test_receipt_raw_identity_is_reported_and_bound(tmp_path: Path):
    receipt = _receipt(tmp_path / "snapshot")
    path = tmp_path / "receipt.json"
    _write_receipt(path, receipt, sort_keys=True)
    raw = path.read_bytes()
    metadata = exact.inspect_boundary_metadata(
        path,
        require_members=True,
        expected_receipt_path=path,
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        expected_size_bytes=len(raw),
    )
    assert metadata["receipt_path"] == str(path.resolve())
    assert metadata["receipt_raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["receipt_size_bytes"] == len(raw)
    loaded = exact.load_boundary_capture(path)
    assert loaded["json_compatibility"]["receipt_path"] == str(path.resolve())
    assert loaded["json_compatibility"]["receipt_raw_sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(RuntimeError, match="BOUNDARY_RECEIPT_RAW_IDENTITY"):
        exact.inspect_boundary_metadata(path, expected_raw_sha256="0" * 64, require_members=False)
    recovered = exact.inspect_boundary_metadata(path, require_members=True, header_only=True)
    assert recovered["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"


def test_duplicate_json_key_is_rejected_before_receipt_validation(tmp_path: Path):
    baseline = _receipt(tmp_path / "snapshot")
    baseline_path = tmp_path / "baseline.json"
    _write_receipt(baseline_path, baseline, sort_keys=True)
    assert exact.inspect_boundary_metadata(baseline_path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="BOUNDARY_JSON_DUPLICATE_KEY"):
        exact.inspect_boundary_metadata(path, require_members=False)
    assert exact.inspect_boundary_metadata(baseline_path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"


@pytest.mark.parametrize(
    ("label", "payload", "reason"),
    (
        ("dynamic_duplicate", '{"dynamic_scalars":{"z":1,"z":2}}\n', "BOUNDARY_JSON_DUPLICATE_KEY"),
        ("nonmapping", '[]\n', "BOUNDARY_RECEIPT_INPUT"),
    ),
)
def test_json_shape_faults_have_explicit_reasons(tmp_path: Path, label: str, payload: str, reason: str):
    baseline = _receipt(tmp_path / "snapshot")
    baseline_path = tmp_path / "baseline.json"
    _write_receipt(baseline_path, baseline, sort_keys=True)
    assert exact.inspect_boundary_metadata(baseline_path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"
    path = tmp_path / f"{label}.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match=reason):
        exact.inspect_boundary_metadata(path, require_members=False)
    assert exact.inspect_boundary_metadata(baseline_path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"


@pytest.mark.parametrize(
    ("label", "mutator", "reason"),
    (
        ("missing_key", lambda d: d["dynamic_scalars"].pop("z"), "BOUNDARY_DYNAMIC_SCALAR_SET"),
        ("extra_key", lambda d: d["dynamic_scalars"].update(extra=1.0), "BOUNDARY_DYNAMIC_SCALAR_SET"),
        ("dynamic_nonmapping", lambda d: d.update(dynamic_scalars=[]), "BOUNDARY_DYNAMIC_SCALARS_MAPPING"),
        ("schema", lambda d: d.update(schema="wrong.schema.v1"), "BOUNDARY_RECEIPT_SCHEMA"),
        ("field_count", lambda d: d.update(field_count=111), "BOUNDARY_FIELD_COUNT"),
        ("name", lambda d: d["arrays"][0].update(field_name="wrong"), "BOUNDARY_ARRAY_FIELD_ORDER"),
        ("file_mapping", lambda d: d["arrays"][0].update(snapshot_file="renamed.npy"), "BOUNDARY_SNAPSHOT_FILE_MAPPING"),
        ("path", lambda d: d["arrays"][0].update(snapshot_file="../escape.npy"), "BOUNDARY_SNAPSHOT_PATH"),
    ),
)
def test_metadata_faults_reject_and_recovered_baseline_pass(tmp_path: Path, label, mutator, reason):
    baseline = _make_comparable_fixture(tmp_path / "baseline")
    baseline_path = tmp_path / "baseline.json"
    _write_receipt(baseline_path, baseline, sort_keys=True)
    assert exact.inspect_boundary_metadata(baseline_path, require_members=True)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"
    faulty = copy.deepcopy(baseline)
    mutator(faulty)
    faulty_path = tmp_path / f"{label}.json"
    _write_receipt(faulty_path, faulty, sort_keys=True)
    with pytest.raises(RuntimeError, match=reason):
        exact.inspect_boundary_metadata(faulty_path, require_members=True)
    assert exact.inspect_boundary_metadata(baseline_path, require_members=True)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"


def test_raw_identity_fault_is_rejected_before_json_schema(tmp_path: Path):
    receipt = _receipt(tmp_path / "snapshot")
    path = tmp_path / "receipt.json"
    _write_receipt(path, receipt, sort_keys=True)
    assert exact.inspect_boundary_metadata(path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"
    with pytest.raises(RuntimeError, match="BOUNDARY_RECEIPT_RAW_IDENTITY"):
        exact.inspect_boundary_metadata(path, expected_raw_sha256="0" * 64, require_members=False)
    assert exact.inspect_boundary_metadata(path, require_members=True, header_only=True)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"


@pytest.mark.parametrize(
    ("label", "mutator", "reason"),
    (
        ("native_dtype", lambda d: d["native_layouts"]["E"].update(dtype=[]), "BOUNDARY_NATIVE_LAYOUT_DTYPE"),
        ("alias_list", lambda d: d["aliases"].update(input={}), "BOUNDARY_ALIASES_LIST"),
        ("linear_result", lambda d: d["aliases"].update(linear_result=[]), "BOUNDARY_ALIASES_LINEAR_RESULT"),
        ("linear_dtype", lambda d: d["aliases"]["linear_result"]["E_layout"].update(dtype=[]), "BOUNDARY_LINEAR_RESULT_DTYPE"),
    ),
)
def test_native_and_alias_metadata_types_fail_before_header_reads(tmp_path: Path, label, mutator, reason: str):
    baseline = _receipt(tmp_path / "snapshot")
    baseline_path = tmp_path / "baseline.json"
    _write_receipt(baseline_path, baseline, sort_keys=True)
    assert exact.inspect_boundary_metadata(baseline_path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"
    faulty = copy.deepcopy(baseline)
    mutator(faulty)
    path = tmp_path / f"{label}.json"
    _write_receipt(path, faulty, sort_keys=True)
    with pytest.raises(RuntimeError, match=reason):
        exact.inspect_boundary_metadata(path, require_members=True, header_only=True)
    assert exact.inspect_boundary_metadata(baseline_path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"


def test_invalid_scalar_metadata_is_rejected_before_np_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    receipt = _receipt(tmp_path / "snapshot")
    baseline_path = tmp_path / "baseline.json"
    _write_receipt(baseline_path, receipt, sort_keys=True)
    assert exact.load_boundary_capture(baseline_path)["json_compatibility"]["scalar_values_unchanged"] is True
    receipt["dynamic_scalars"] = dict(receipt["dynamic_scalars"])
    receipt["dynamic_scalars"].pop("z")
    path = tmp_path / "missing_scalar.json"
    _write_receipt(path, receipt, sort_keys=True)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("np.load must not run after metadata failure")

    monkeypatch.setattr(np, "load", fail_if_called)
    with pytest.raises(RuntimeError, match="BOUNDARY_DYNAMIC_SCALAR_SET"):
        exact.load_boundary_capture(path)
    monkeypatch.undo()
    assert exact.load_boundary_capture(baseline_path)["json_compatibility"]["scalar_values_unchanged"] is True


def test_metadata_only_historical_receipt_does_not_claim_scientific_pass(tmp_path: Path):
    receipt = _receipt(tmp_path / "missing_snapshot")
    receipt["snapshot_directory"] = str(tmp_path / "absent_snapshot_root")
    path = tmp_path / "historical.json"
    _write_receipt(path, receipt, sort_keys=True)
    metadata = exact.inspect_boundary_metadata(path, require_members=False, header_only=True)
    assert metadata["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"
    assert metadata["scientific_comparison_status"] == "NOT_RUN_METADATA_ONLY"
    assert metadata["missing_snapshot_members"]
    with pytest.raises(RuntimeError, match="BOUNDARY_SNAPSHOT_MEMBER_MISSING"):
        exact.load_boundary_capture(path)


@pytest.mark.parametrize("mutation", ("scalar", "swap", "array"))
def test_original_scientific_comparator_rejects_value_mutations(tmp_path: Path, mutation: str):
    baseline = _make_comparable_fixture(tmp_path / "baseline")
    baseline_path = tmp_path / "baseline.json"
    _write_receipt(baseline_path, baseline, sort_keys=True)
    if mutation == "array":
        candidate_root = tmp_path / "candidate_snapshot"
        shutil.copytree(tmp_path / "baseline", candidate_root)
        value = np.load(candidate_root / "E.npy", allow_pickle=False)
        value[0] = value[0] + 1.0
        np.save(candidate_root / "E.npy", value, allow_pickle=False)
        candidate = copy.deepcopy(baseline)
        candidate["snapshot_directory"] = str(candidate_root)
        candidate_path = tmp_path / "candidate.json"
        _write_receipt(candidate_path, candidate, sort_keys=True)
    else:
        candidate = copy.deepcopy(baseline)
        values = candidate["dynamic_scalars"]
        if mutation == "scalar":
            values["z"] = float(values["z"]) + 0.25
        else:
            values["z"], values["U_now"] = values["U_now"], values["z"]
        candidate_path = tmp_path / "candidate.json"
        _write_receipt(candidate_path, candidate, sort_keys=True)
    baseline_result = _compare(baseline_path, baseline_path, tmp_path / f"baseline_{mutation}.jsonl")
    assert baseline_result["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
    with pytest.raises(RuntimeError, match="FIRST_SCIENTIFIC_NONEXACT"):
        _compare(baseline_path, candidate_path, tmp_path / f"mutated_{mutation}.jsonl")
    recovered_result = _compare(baseline_path, baseline_path, tmp_path / f"recovered_{mutation}.jsonl")
    assert recovered_result["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"


def test_signed_zero_and_nonfinite_scalar_semantics_are_preserved(tmp_path: Path):
    receipt = _make_comparable_fixture(
        tmp_path / "snapshot",
        scalar_overrides={"z": -0.0},
    )
    path = tmp_path / "special.json"
    _write_receipt(path, receipt, sort_keys=True)
    loaded = exact.load_boundary_capture(path)
    assert struct.pack("!d", loaded["fields"]["scalars.z"]).hex() == "8000000000000000"
    identity = loaded["json_compatibility"]["scalar_identity"]
    assert identity["z"]["source"]["ieee754_bits"] == "8000000000000000"
    assert identity["z"]["canonical"]["ieee754_bits"] == "8000000000000000"
    assert loaded["json_compatibility"]["scalar_values_unchanged"] is True
    comparison = _compare(path, path, tmp_path / "special_scientific_compare.jsonl")
    assert comparison["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"

    positive_zero = copy.deepcopy(receipt)
    positive_zero["dynamic_scalars"]["z"] = 0.0
    positive_path = tmp_path / "positive_zero.json"
    _write_receipt(positive_path, positive_zero, sort_keys=True)
    with pytest.raises(RuntimeError, match="FIRST_SCIENTIFIC_NONEXACT"):
        _compare(path, positive_path, tmp_path / "signed_zero_scientific_rejection.jsonl")
    assert _compare(path, path, tmp_path / "signed_zero_recovered.jsonl")["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"

    nan_receipt = copy.deepcopy(receipt)
    nan_receipt["dynamic_scalars"]["U_now"] = "NaN"
    nan_path = tmp_path / "nan.json"
    _write_receipt(nan_path, nan_receipt, sort_keys=True)
    nan_loaded = exact.load_boundary_capture(nan_path)
    assert math.isnan(nan_loaded["fields"]["scalars.U_now"])
    nan_identity = nan_loaded["json_compatibility"]["scalar_identity"]["U_now"]
    assert nan_identity["source"]["type"] == "nonfinite_float"
    assert nan_identity["canonical"]["type"] == "float"
    assert nan_identity["value_preserved_by_name"] is True
    # Preserve the pre-existing frozen scientific adjudication: a non-finite
    # scalar is rejected by the original exact comparator, not normalized.
    with pytest.raises(RuntimeError, match="FIRST_SCIENTIFIC_NONEXACT"):
        _compare(nan_path, nan_path, tmp_path / "nan_scientific_compare.jsonl")
    assert _compare(path, path, tmp_path / "nonfinite_recovered.jsonl")["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"


def test_expected_receipt_path_identity_causal_chain(tmp_path: Path):
    receipt = _receipt(tmp_path / "snapshot")
    path, other = tmp_path / "baseline.json", tmp_path / "other.json"
    _write_receipt(path, receipt, sort_keys=True)
    _write_receipt(other, receipt, sort_keys=True)
    assert exact.inspect_boundary_metadata(path, expected_receipt_path=path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"
    with pytest.raises(RuntimeError, match="BOUNDARY_RECEIPT_PATH_IDENTITY"):
        exact.inspect_boundary_metadata(path, expected_receipt_path=other)
    assert exact.inspect_boundary_metadata(path, expected_receipt_path=path)["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY"


def test_actual_historical_raw_receipt_metadata_only():
    import os
    configured = os.environ.get("CPU_BOUNDARY_HISTORICAL_RECEIPT")
    if not configured:
        pytest.skip("original historical raw receipt is required")
    path = Path(configured).resolve(strict=True)
    metadata = exact.inspect_boundary_metadata(
        path, require_members=False, header_only=True, expected_receipt_path=path,
        expected_raw_sha256="5f98b3e196a7c1edb9e6bfb4207ef7c381964c92585b36e8eec3a6499adb6d97",
        expected_size_bytes=183246)
    if metadata["scientific_comparison_status"] != "NOT_RUN_METADATA_ONLY" or metadata["array_record_count"] != 105:
        raise RuntimeError("HISTORICAL_METADATA_ONLY_SCOPE")
    root = Path(os.environ["S43SG_REPORT_ROOT"])
    target = root / "raw" / (os.environ["S43SG_STAGE"] + "_HISTORICAL_METADATA_PARSE.json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)

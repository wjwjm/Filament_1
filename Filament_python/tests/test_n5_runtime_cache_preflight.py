from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "hpc_ops" / "n5_runtime_cache_preflight.py"
)
_spec = importlib.util.spec_from_file_location("n5_runtime_cache_preflight", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)


def _runtime_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    run_root = tmp_path / "campaign" / "run"
    run_root.mkdir(parents=True)
    entry = run_root.with_name(run_root.name + "_entry")
    runtime = entry / "runtime_env"
    for name in ("home", "xdg_cache", "cupy_cache", "tmp"):
        (runtime / name).mkdir(parents=True)
    if os.name == "posix":
        for path in (entry, runtime, *(runtime / name for name in ("home", "xdg_cache", "cupy_cache", "tmp"))):
            path.chmod(0o700)
    monkeypatch.chdir(run_root)
    monkeypatch.setenv("N5_ENTRY_ROOT", str(entry))
    monkeypatch.setenv("N5_RUNTIME_ENV_ROOT", str(runtime))
    monkeypatch.setenv("HOME", str(runtime / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(runtime / "xdg_cache"))
    monkeypatch.setenv("CUPY_CACHE_DIR", str(runtime / "cupy_cache"))
    monkeypatch.setenv("TMPDIR", str(runtime / "tmp"))
    monkeypatch.setenv("TMP", str(runtime / "tmp"))
    monkeypatch.setenv("TEMP", str(runtime / "tmp"))
    tempfile.tempdir = None
    return run_root, runtime


def test_runtime_environment_checks_locations_and_atomic_write_rename(tmp_path, monkeypatch):
    run_root, runtime = _runtime_tree(tmp_path, monkeypatch)

    receipt = preflight.validate_runtime_environment(run_root)

    assert receipt["run_root"] == str(run_root)
    assert receipt["cwd"] == str(run_root)
    assert receipt["python_tempdir"] == str(runtime / "tmp")
    assert receipt["same_filesystem_as_run_root"] is True
    assert len(receipt["write_rename_probes"]) == 4
    assert all(row["write"] and row["fsync"] and row["atomic_rename"] and row["readback"] for row in receipt["write_rename_probes"])
    assert all(not list(path.iterdir()) for path in (runtime / "home", runtime / "xdg_cache", runtime / "cupy_cache", runtime / "tmp"))


def test_runtime_environment_rejects_wrong_cupy_cache_target(tmp_path, monkeypatch):
    run_root, _runtime = _runtime_tree(tmp_path, monkeypatch)
    escaped = tmp_path / "home-cache"
    escaped.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(escaped))

    with pytest.raises(ValueError, match="CUPY_CACHE_DIR"):
        preflight.validate_runtime_environment(run_root)


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-mode contract")
def test_runtime_environment_rejects_loose_permissions(tmp_path, monkeypatch):
    run_root, runtime = _runtime_tree(tmp_path, monkeypatch)
    (runtime / "cupy_cache").chmod(0o755)

    with pytest.raises(ValueError, match="mode 700"):
        preflight.validate_runtime_environment(run_root)


def test_cache_inventory_hashes_raw_artifact_bytes(tmp_path):
    cache = tmp_path / "cupy"
    cache.mkdir()
    nested = cache / "kernel"
    nested.mkdir()
    artifact = nested / "kernel.cubin"
    artifact.write_bytes(b"compiled-kernel-bytes")

    inventory = preflight.cache_inventory(cache)

    assert inventory["file_count"] == 1
    assert inventory["total_bytes"] == len(b"compiled-kernel-bytes")
    assert inventory["directories"] == ["kernel"]
    assert inventory["files"][0]["path"] == "kernel/kernel.cubin"
    assert inventory["files"][0]["sha256"] == preflight.hashlib.sha256(b"compiled-kernel-bytes").hexdigest()
    assert inventory["files"][0]["nlink"] == artifact.stat().st_nlink


def test_cold_and_fresh_warm_cache_phase_contract(tmp_path):
    cache = tmp_path / "cupy"
    cache.mkdir()
    empty = preflight.cache_inventory(cache)
    preflight.validate_phase_preconditions("cold", empty)

    compiled = cache / "kernel.bin"
    compiled.write_bytes(b"cold-cache")
    cold_after = preflight.cache_inventory(cache)
    cold_receipt = {
        "status": "PASS",
        "phase": "cold",
        "process": {"pid": 101},
        "config_sha256": "a" * 64,
        "runtime_environment": {"paths": {"CUPY_CACHE_DIR": str(cache)}},
        "cuda_visible_devices": "2",
        "cache_inventory_after": cold_after,
    }
    preflight.validate_phase_preconditions(
        "warm",
        cold_after,
        cold_receipt=cold_receipt,
        current_pid=202,
        config_sha256="a" * 64,
        environment_paths={"CUPY_CACHE_DIR": str(cache)},
        cuda_visible_devices="2",
    )
    preflight.validate_phase_postconditions("warm", cold_after, cold_after, cold_receipt=cold_receipt)

    changed = cache / "kernel.bin"
    changed.write_bytes(b"changed-cache")
    with pytest.raises(ValueError, match="warm process changed"):
        preflight.validate_phase_postconditions(
            "warm",
            cold_after,
            preflight.cache_inventory(cache),
            cold_receipt=cold_receipt,
        )


def test_warm_requires_new_process_and_same_formal_binding(tmp_path):
    cache = tmp_path / "cupy"
    cache.mkdir()
    artifact = cache / "kernel"
    artifact.write_bytes(b"cold")
    inventory = preflight.cache_inventory(cache)
    cold = {
        "status": "PASS",
        "phase": "cold",
        "process": {"pid": 44},
        "config_sha256": "b" * 64,
        "runtime_environment": {"paths": {"CUPY_CACHE_DIR": str(cache)}},
        "cuda_visible_devices": "0",
        "cache_inventory_after": inventory,
    }

    with pytest.raises(ValueError, match="fresh process"):
        preflight.validate_phase_preconditions(
            "warm",
            inventory,
            cold_receipt=cold,
            current_pid=44,
            config_sha256="b" * 64,
            environment_paths={"CUPY_CACHE_DIR": str(cache)},
            cuda_visible_devices="0",
        )
    with pytest.raises(ValueError, match="config differs"):
        preflight.validate_phase_preconditions(
            "warm",
            inventory,
            cold_receipt=cold,
            current_pid=45,
            config_sha256="c" * 64,
            environment_paths={"CUPY_CACHE_DIR": str(cache)},
            cuda_visible_devices="0",
        )


def test_receipt_creation_is_exclusive(tmp_path):
    output = tmp_path / "receipt.json"
    preflight.write_receipt_exclusive(output, {"status": "PASS", "sample": 1})
    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        preflight.write_receipt_exclusive(output, {"status": "FAIL", "sample": 2})

    assert output.read_bytes() == original
    assert json.loads(original.decode("utf-8"))["sample"] == 1


def test_probe_source_is_diagnostic_only():
    source = MODULE_PATH.read_text(encoding="utf-8")

    tree = ast.parse(source)
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "propagate_one_pulse" not in calls
    assert "isaacs_raman_stage" in calls
    assert '"scientific_source_allocated": False' in source
    assert '"propagation_called": False' in source
    assert 'dtype=cp.complex128' in source
    assert 'int(grid.Nt) != 384' in source
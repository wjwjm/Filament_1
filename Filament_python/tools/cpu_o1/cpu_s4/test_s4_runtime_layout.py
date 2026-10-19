"""Focused regressions for external S4 runtime-test storage placement."""
from pathlib import Path

import pytest

from tools.cpu_o1 import q1_t0_real as q1
from tools.cpu_o1.cpu_s4 import runtime_gate as gate


def roots(tmp_path):
    adapter = tmp_path / 'adapter'
    storage = tmp_path / 'storage'
    adapter.mkdir()
    storage.mkdir()
    return adapter, storage


def symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f'symlink support unavailable; run this check on Linux: {exc}')


def test_s4_runtime_layout_accepts_external_sibling_and_keeps_pytest_link_outside(tmp_path):
    adapter, storage = roots(tmp_path)
    expected = storage.with_name(storage.name + '.qualification')
    runtime = gate.validate_runtime_test_layout(expected,
        storage_root=storage, adapter_root=adapter)
    assert runtime == expected.resolve()
    assert not runtime.exists()

    runtime.mkdir()
    target = runtime / 'pytest_tmp' / 's4_inventory_source123'
    target.mkdir(parents=True)
    link = target.parent / 's4_inventory_sourcecurrent'
    symlink_or_skip(link, target.name, directory=True)

    frozen_storage = q1._storage_inventory(storage)
    q1._validate_storage_symlinks(frozen_storage, storage)
    external = gate.runtime_test_inventory(runtime)
    assert external['status'] == 'PASS'
    assert any(row.get('symlink_target') == target.name
               and row['path'].endswith('s4_inventory_sourcecurrent')
               for row in external['entries'])


def test_s4_runtime_layout_rejects_old_inside_root_pytest_current_link(tmp_path):
    adapter, storage = roots(tmp_path)
    runtime = storage / 'runtime_gate' / 'pytest_tmp'
    target = runtime / 's4_inventory_source123'
    target.mkdir(parents=True)
    link = runtime / 's4_inventory_sourcecurrent'
    symlink_or_skip(link, target.name, directory=True)

    inventory = q1._storage_inventory(storage)
    assert any(Path(item).name == link.name for item in inventory['symlinks'])
    with pytest.raises(RuntimeError, match='unexpected task-root symlink'):
        q1._validate_storage_symlinks(inventory, storage)


def test_s4_runtime_layout_still_rejects_unexpected_inside_symlink(tmp_path):
    adapter, storage = roots(tmp_path)
    target = tmp_path / 'target.txt'
    target.write_text('target', encoding='utf-8')
    symlink_or_skip(storage / 'unexpected', target)

    inventory = q1._storage_inventory(storage)
    with pytest.raises(RuntimeError, match='unexpected task-root symlink'):
        q1._validate_storage_symlinks(inventory, storage)


def test_s4_runtime_layout_post_test_storage_growth_uses_frozen_byte_and_file_caps(tmp_path):
    storage = tmp_path / 'storage'
    storage.mkdir()
    (storage / 'payload.bin').write_bytes(b'12345')

    with pytest.raises(RuntimeError, match='total task-root bytes exceed cap'):
        q1._check_storage_growth(storage, growth_cap_bytes=4, max_files=2)
    with pytest.raises(RuntimeError, match='storage file count exceeds cap'):
        q1._check_storage_growth(storage, growth_cap_bytes=5, max_files=0)

    checked = q1._check_storage_growth(storage, growth_cap_bytes=5, max_files=1)
    assert checked['bytes'] == 5
    assert checked['files'] == 1
    assert checked['growth_cap_bytes'] == 5
    assert checked['max_files'] == 1


@pytest.mark.parametrize('layout', ['relative', 'storage_child', 'adapter_child', 'existing'])
def test_s4_runtime_layout_rejects_invalid_existing_or_inside_roots(tmp_path, layout):
    adapter, storage = roots(tmp_path)
    expected = storage.with_name(storage.name + '.qualification')
    if layout == 'relative':
        candidate = Path('storage.qualification')
    elif layout == 'storage_child':
        candidate = storage / 'runtime_test'
    elif layout == 'adapter_child':
        candidate = adapter / 'runtime_test'
    else:
        candidate = expected
        candidate.mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        gate.validate_runtime_test_layout(candidate,
            storage_root=storage, adapter_root=adapter)


def test_s4_runtime_layout_reservation_accounts_for_external_capacity():
    reserve = gate.runtime_test_reservation()
    assert reserve['max_bytes'] == 512 * 1024**2
    assert reserve['max_entries'] == 20_000
    assert reserve['reserved_bytes'] == 644_245_095
    assert reserve['reserved_entries'] == 24_000
    result = gate.account_runtime_test_reservation(
        current_bytes=100, cap_bytes=11 * 1024**3,
        filesystem_available_bytes=reserve['reserved_bytes'] + 1000,
        filesystem_available_inodes=reserve['reserved_entries'] + 100,
        quota_available_bytes=reserve['reserved_bytes'] + 2000,
        quota_available_inodes=reserve['reserved_entries'] + 200)
    assert result['current_bytes'] == 100 + reserve['reserved_bytes']
    assert result['filesystem_available_bytes'] == 1000
    assert result['filesystem_available_inodes'] == 100
    assert result['quota_available_bytes'] == 2000
    assert result['quota_available_inodes'] == 200


@pytest.mark.parametrize('shortfall', ['bytes', 'inodes', 'cap'])
def test_s4_runtime_layout_insufficient_external_budget_fails(shortfall):
    reserve = gate.runtime_test_reservation()
    inputs = dict(current_bytes=0, cap_bytes=11 * 1024**3,
        filesystem_available_bytes=reserve['reserved_bytes'],
        filesystem_available_inodes=reserve['reserved_entries'],
        quota_available_bytes=reserve['reserved_bytes'],
        quota_available_inodes=reserve['reserved_entries'])
    if shortfall == 'bytes':
        inputs['quota_available_bytes'] -= 1
    elif shortfall == 'inodes':
        inputs['filesystem_available_inodes'] -= 1
    else:
        inputs['cap_bytes'] = reserve['reserved_bytes'] - 1
    with pytest.raises(RuntimeError, match='EXTERNAL_TEST_BUDGET'):
        gate.account_runtime_test_reservation(**inputs)


def test_s4_runtime_layout_inventory_enforces_byte_and_entry_limits(tmp_path, monkeypatch):
    adapter, storage = roots(tmp_path)
    runtime = storage.with_name(storage.name + '.qualification')
    runtime.mkdir()
    (runtime / 'payload.bin').write_bytes(b'1234')

    monkeypatch.setattr(gate, 'RUNTIME_TEST_MAX_BYTES', 3)
    by_bytes = gate.runtime_test_inventory(runtime)
    assert by_bytes['status'] == 'BLOCKED_RUNTIME_TEST_INVENTORY'
    assert 'byte_limit_exceeded' in by_bytes['limit_failures']

    monkeypatch.setattr(gate, 'RUNTIME_TEST_MAX_BYTES', 512 * 1024**2)
    monkeypatch.setattr(gate, 'RUNTIME_TEST_MAX_ENTRIES', 1)
    by_entries = gate.runtime_test_inventory(runtime)
    assert by_entries['status'] == 'BLOCKED_RUNTIME_TEST_INVENTORY'
    assert 'entry_limit_exceeded' in by_entries['limit_failures']


def test_s4_runtime_layout_inventory_scan_error_fails_closed(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime.qualification'
    runtime.mkdir()

    def failed_walk(root, *, followlinks, onerror):
        onerror(OSError('simulated directory scan failure'))
        yield str(root), [], []

    monkeypatch.setattr(gate.os, 'walk', failed_walk)
    with pytest.raises(RuntimeError, match='runtime test inventory scan failed'):
        gate.runtime_test_inventory(runtime)


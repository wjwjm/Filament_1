"""Allocation-only S4 qualification using existing probes and regression tests."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime


def allocation_clock(text, job_id, now):
    fields = dict(token.split('=', 1) for token in text.split() if '=' in token)
    if (fields.get('JobId') != str(job_id) or fields.get('JobState') != 'RUNNING'
            or fields.get('TimeLimit') != '02:00:00'):
        raise ValueError('scheduler job identity/state/time limit mismatch')
    start = datetime.fromisoformat(fields['StartTime']).timestamp()
    if not 0 < start <= now < start + 7200:
        raise ValueError('invalid or expired allocation clock')
    return start

SELECTORS = ('test_s4_admission_parent_loss_cleans_child_tree or '
    'test_s4_admission_start_failure_and_sigint_handler or '
    'test_s4_admission_failure_stops_matrix or '
    'test_s4_admission_tiny_independent_exact or test_s4_two_gates or '
    'test_s4_real_entry_quota_and_no_linux or test_s4_real_entry_allocation_clock or '
    'test_s4_runtime_layout')

RUNTIME_TEST_MAX_BYTES = 512 * 1024**2
RUNTIME_TEST_MAX_ENTRIES = 20_000
RUNTIME_TEST_RESERVE_FRACTION = 0.20


def runtime_test_reservation():
    """Return fixed external pytest/JIT allowance with a 20% headroom."""
    return {
        'max_bytes': RUNTIME_TEST_MAX_BYTES,
        'max_entries': RUNTIME_TEST_MAX_ENTRIES,
        'reserve_fraction': RUNTIME_TEST_RESERVE_FRACTION,
        'reserved_bytes': math.ceil(RUNTIME_TEST_MAX_BYTES * (1 + RUNTIME_TEST_RESERVE_FRACTION)),
        'reserved_entries': math.ceil(RUNTIME_TEST_MAX_ENTRIES * (1 + RUNTIME_TEST_RESERVE_FRACTION)),
    }


def account_runtime_test_reservation(*, current_bytes, cap_bytes,
        filesystem_available_bytes, filesystem_available_inodes,
        quota_available_bytes, quota_available_inodes):
    """Reserve bounded external gate artifacts in all storage budget inputs."""
    values = {
        'current_bytes': current_bytes,
        'cap_bytes': cap_bytes,
        'filesystem_available_bytes': filesystem_available_bytes,
        'filesystem_available_inodes': filesystem_available_inodes,
        'quota_available_bytes': quota_available_bytes,
        'quota_available_inodes': quota_available_inodes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if cap_bytes == 0:
        raise ValueError('cap_bytes must be positive')
    reserve = runtime_test_reservation()
    if (current_bytes + reserve['reserved_bytes'] > cap_bytes
            or filesystem_available_bytes < reserve['reserved_bytes']
            or quota_available_bytes < reserve['reserved_bytes']
            or filesystem_available_inodes < reserve['reserved_entries']
            or quota_available_inodes < reserve['reserved_entries']):
        raise RuntimeError('BLOCKED_RUNTIME_GATE_EXTERNAL_TEST_BUDGET')
    return {
        'current_bytes': current_bytes + reserve['reserved_bytes'],
        'filesystem_available_bytes': filesystem_available_bytes - reserve['reserved_bytes'],
        'filesystem_available_inodes': filesystem_available_inodes - reserve['reserved_entries'],
        'quota_available_bytes': quota_available_bytes - reserve['reserved_bytes'],
        'quota_available_inodes': quota_available_inodes - reserve['reserved_entries'],
        'external_runtime_reservation': reserve,
    }


def _is_within(path, root):
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def validate_runtime_test_layout(runtime_test_root, *, storage_root, adapter_root):
    """Require a fresh, absolute qualification sibling outside both roots."""
    paths = {}
    for name, value in (('runtime_test_root', runtime_test_root),
                        ('storage_root', storage_root), ('adapter_root', adapter_root)):
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            raise ValueError(f'{name} must be an absolute path')
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ValueError(f'{name} must be an absolute path')
        paths[name] = candidate
    storage = paths['storage_root'].resolve(strict=True)
    adapter = paths['adapter_root'].resolve(strict=True)
    if not storage.is_dir() or not adapter.is_dir():
        raise ValueError('storage_root and adapter_root must be directories')
    raw_runtime = paths['runtime_test_root']
    if os.path.lexists(raw_runtime):
        raise FileExistsError(raw_runtime)
    runtime = raw_runtime.resolve(strict=False)
    expected = storage.with_name(storage.name + '.qualification')
    if runtime != expected:
        raise ValueError('runtime_test_root must equal storage_root.with_name(storage_root.name + ".qualification")')
    if os.path.lexists(runtime):
        raise FileExistsError(runtime)
    if _is_within(runtime, storage) or _is_within(runtime, adapter):
        raise ValueError('runtime_test_root must be outside storage_root and adapter_root')
    return runtime


def runtime_test_inventory(path):
    """Inventory external test artifacts without following links or hashing files."""
    raw_root = Path(path)
    if raw_root.is_symlink():
        raise RuntimeError('runtime test root must not be a symlink')
    root = raw_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError('runtime test root must be a directory')
    entries = [{'path': '.', 'kind': 'directory', 'size_bytes': 0}]
    total_bytes = 0
    overflow = False

    def record(item, kind, size, target=None):
        nonlocal total_bytes, overflow
        row = {'path': item.relative_to(root).as_posix(),
               'kind': kind, 'size_bytes': int(size)}
        if target is not None:
            row['symlink_target'] = target
        entries.append(row)
        total_bytes += int(size) if kind != 'directory' else 0
        if len(entries) > RUNTIME_TEST_MAX_ENTRIES:
            overflow = True

    def fail_scan(error):
        raise RuntimeError(f'runtime test inventory scan failed: {error}') from error

    for current, dirnames, filenames in os.walk(root, followlinks=False, onerror=fail_scan):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        descend = []
        for name in dirnames:
            item = current_path / name
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode):
                record(item, 'symlink', info.st_size, os.readlink(item))
            elif stat.S_ISDIR(info.st_mode):
                record(item, 'directory', 0)
                descend.append(name)
            else:
                record(item, 'other', info.st_size)
            if overflow:
                break
        if overflow:
            break
        dirnames[:] = descend
        for name in filenames:
            item = current_path / name
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode):
                record(item, 'symlink', info.st_size, os.readlink(item))
            elif stat.S_ISDIR(info.st_mode):
                record(item, 'directory', 0)
            elif stat.S_ISREG(info.st_mode):
                record(item, 'file', info.st_size)
            else:
                record(item, 'other', info.st_size)
            if overflow:
                break
        if overflow:
            break
    entries.sort(key=lambda row: row['path'])
    reasons = []
    if overflow:
        reasons.append('entry_limit_exceeded')
    if total_bytes > RUNTIME_TEST_MAX_BYTES:
        reasons.append('byte_limit_exceeded')
    return {
        'schema': 'filament.cpu_s4.runtime_test_inventory.v1',
        'root': str(root),
        'status': 'PASS' if not reasons else 'BLOCKED_RUNTIME_TEST_INVENTORY',
        'complete': not overflow,
        'bytes': total_bytes,
        'entry_count': len(entries),
        'max_bytes': RUNTIME_TEST_MAX_BYTES,
        'max_entries': RUNTIME_TEST_MAX_ENTRIES,
        'limit_failures': reasons,
        'entries': entries,
    }


def quota_available(text):
    header = 'Filesystem type KB quota limit in_doubt grace | files quota limit in_doubt grace Remarks'
    if header not in [' '.join(line.split()) for line in text.splitlines()]:
        raise ValueError('unverified quota units or column layout')
    rows = [line.split() for line in text.splitlines()
            if line.split()[:2] == ['publicfs01', 'USR']]
    if len(rows) != 1:
        raise ValueError('ambiguous or unavailable site quota')
    row = rows[0]
    if len(row) != 14 or row[7] != '|':
        raise ValueError('truncated site quota')
    used, soft, hard, doubt = map(int, row[2:6])
    files, fsoft, fhard, fdoubt = map(int, row[8:12])
    if min(used, doubt, files, fdoubt) < 0 or min(soft, hard, fsoft, fhard) <= 0:
        raise ValueError('invalid site quota limits')
    return dict(available_bytes=max(0, min(soft, hard)-used-doubt)*1024,
                available_inodes=max(0, min(fsoft, fhard)-files-fdoubt))


def validate_junit(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    if len(cases) < 60 or any(any(c.tag in ('failure','error','skipped')
                                 for c in case) for case in cases):
        raise RuntimeError('runtime regression incomplete, failed or skipped')
    for name, count in [('parent_loss_cleans_child_tree',3),
                        ('failure_stops_matrix',7),
                        ('every_required_file_missing',21),
                        ('truncated_write',4),
                        ('tiny_independent_exact',1)]:
        if sum(name in c.get('name','') for c in cases) != count:
            raise RuntimeError('runtime regression coverage mismatch: '+name)
    return dict(testcases=len(cases), failures=0, skips=0)


def run(bound, *, approval_path, config_path, output_root):
    """Produce evidence now; never accept a caller-supplied PASS as execution."""
    if sys.platform != 'linux' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('BLOCKED_RUNTIME_GATE_REQUIRES_LINUX_ALLOCATION')
    from tools.hpc_ops import provenance_v2 as pv
    from tools.cpu_o1 import q1_t0_real as q1
    from tools.cpu_o1.cpu_s4 import admission, memory_limits, storage_budget
    from tools.cpu_o1.cpu_s4.launcher import RESOURCES, RUNTIME_CHECKS
    config=bound['config']; root=Path(config['runtime_gate_receipt']).parent.resolve()
    expected=Path(output_root).resolve().with_name(Path(output_root).name+'.runtime_gate')
    if root != expected or Path(config['runtime_gate_receipt']).name != 'RUNTIME_GATE.json':
        raise RuntimeError('runtime gate must use the dedicated sibling evidence root')
    if root.exists(): raise FileExistsError(root)
    runtime_test_root=validate_runtime_test_layout(config.get('runtime_test_root'),
        storage_root=config['storage_root'],adapter_root=config['adapter_root'])
    q1._check_expected_python(Path(config['expected_python']))
    allocation=q1._allocation_environment()
    root.mkdir()
    def save(name, value):
        admission.write_json_new(root/name, value)
    started=time.monotonic(); result={'schema':'filament.cpu_s4.linux_runtime_gate.v1',
        'status':'BLOCKED_RUNTIME_GATE','run_id':bound['run_id'],
        'job_id':bound['job_id'],'resources':RESOURCES,'checks':{},
        'authorization_sha256':pv.raw_sha256_file(approval_path),
        'run_config_sha256':pv.raw_sha256_file(config_path),
        'adapter_overlay_inventory_sha256':bound['overlay_inventory']['inventory_sha256'],
        'source_provenance_sha256':pv.raw_sha256_file(Path(config['provenance'])),
        'adapter_provenance_sha256':pv.raw_sha256_file(Path(config['adapter_provenance'])),
        'expected_inputs_sha256':pv.raw_sha256_file(Path(config['expected_inputs'])),
        'expected_source_binding_sha256':pv.raw_sha256_file(Path(config['expected_source_binding']))}
    try:
        command=['scontrol','show','job','-o',bound['job_id']]
        proc=subprocess.run(command,capture_output=True,text=True,timeout=30)
        save('scheduler_job.json',dict(command=command,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
        if proc.returncode: raise RuntimeError('scheduler clock query failed')
        start=allocation_clock(proc.stdout,bound['job_id'],time.time())
        os.environ['SLURM_JOB_START_TIME']=str(start)
        result['allocation_start_epoch']=start
        save('allocation.json',allocation)
        q1._run_allocation_probe(root)
        import psutil
        process=psutil.Process()
        memory=memory_limits.snapshot_cgroup_memory(os.getpid(),allocation_bytes=320*1024**3)
        fraction=memory_limits.memory_fractions(rss_bytes=process.memory_info().rss,
            pss_bytes=None,cgroup=memory,allocation_bytes=320*1024**3)
        save('cgroup.json',dict(snapshot=memory,fractions=fraction))
        if fraction['max_enforced_fraction'] >= .8: raise RuntimeError('MEMORY_80_NO_WINDOW')
        gate=admission.validate_preflight(bench_root=config['bench_root'],
            provenance_path=config['provenance'],adapter_root=config['adapter_root'],
            adapter_provenance_path=config['adapter_provenance'],input_root=config['input_root'],
            expected_inputs_path=config['expected_inputs'],verify_input_hashes=True)
        save('input_admission.json',gate)
        if gate['status'] != 'FULL_INPUT_IDENTITY_PASS': raise RuntimeError('input admission failed')
        command=['/usr/lpp/mmfs/bin/mmlsquota','-u','t0s000727','publicfs01']
        proc=subprocess.run(command,capture_output=True,text=True,timeout=60)
        save('quota_raw.json',dict(command=command,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
        if proc.returncode: raise RuntimeError('quota query failed')
        quota=quota_available(proc.stdout)
        storage=Path(config['storage_root']); stat=os.statvfs(storage)
        storage_before=q1._storage_inventory(storage)
        q1._validate_storage_symlinks(storage_before,storage)
        reserved=account_runtime_test_reservation(
            current_bytes=storage_before['bytes'],cap_bytes=config['storage_cap_bytes'],
            filesystem_available_bytes=stat.f_bavail*stat.f_frsize,
            filesystem_available_inodes=stat.f_favail,
            quota_available_bytes=quota['available_bytes'],
            quota_available_inodes=quota['available_inodes'])
        result['storage_quota']={
            'available_bytes':reserved['quota_available_bytes'],
            'available_inodes':reserved['quota_available_inodes'],
            'external_runtime_reservation':reserved['external_runtime_reservation']}
        budget=storage_budget.estimate_storage_budget(source_shape=(384,351,301),
            diagnostic_field_count=len(q1._frozen_diag_list_contract(Path(config['bench_root'])/'Filament_python/KHz_filament/propagate.py')),
            input_records=gate['inputs'],current_bytes=reserved['current_bytes'],
            cap_bytes=config['storage_cap_bytes'],filesystem_available_bytes=reserved['filesystem_available_bytes'],
            filesystem_available_inodes=reserved['filesystem_available_inodes'],
            quota_available_bytes=reserved['quota_available_bytes'],
            quota_available_inodes=reserved['quota_available_inodes'])
        save('storage_budget.json',budget)
        if budget['status']!='PASS': raise RuntimeError('storage budget failed')
        runtime_test_root.mkdir(parents=False,exist_ok=False)
        (runtime_test_root/'gate_only_jit_cache').mkdir()
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
            PYTHONPATH=str(Path(config['adapter_root'])/'Filament_python'),
            NUMBA_CACHE_DIR=str(runtime_test_root/'gate_only_jit_cache'))
        env.pop('PYTEST_ADDOPTS',None)
        command=[sys.executable,'-s','-B','-m','pytest','-q',
            str(Path(__file__).with_name('test_s4_admission.py')),
            str(Path(__file__).with_name('test_s4_two_gates.py')),
            str(Path(__file__).with_name('test_s4_real_entry.py')),
            str(Path(__file__).with_name('test_s4_runtime_layout.py')),
            '-k',SELECTORS,'--basetemp',str(runtime_test_root/'pytest_tmp'),
            '--junitxml',str(root/'tests.xml')]
        try:
            with (root/'tests.log').open('xb') as log:
                proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,cwd=config['adapter_root'])
                created=psutil.Process(proc.pid).create_time()
                try:
                    proc.wait(timeout=900)
                except BaseException:
                    from tools.cpu_o1.cpu_s4.supervisor import terminate
                    terminate(proc.pid,created)
                    proc.wait(timeout=8)
                    raise
        finally:
            test_inventory=runtime_test_inventory(runtime_test_root)
            save('runtime_test_inventory.json',test_inventory)
            post_test_storage=q1._storage_inventory(storage)
            save('post_test_storage.json',post_test_storage)
            post_test_growth=q1._check_storage_growth(storage,
                growth_cap_bytes=int(config['storage_cap_bytes']),
                max_files=int(config.get('max_files',50000)))
            save('post_test_storage_growth.json',post_test_growth)
            if test_inventory['status']!='PASS':
                raise RuntimeError('external runtime-test inventory exceeded its fixed bound')
        if proc.returncode: raise RuntimeError('Linux runtime regression failed')
        save('tests.json',validate_junit(root/'tests.xml'))
        result['checks']={name:'PASS' for name in RUNTIME_CHECKS}
        result['status']='S4_1_LINUX_RUNTIME_GATE_PASS'
    except BaseException as exc:
        result['error']=type(exc).__name__+': '+str(exc)
    result['elapsed_seconds']=time.monotonic()-started
    result['evidence']=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
        hash_scope='raw_bytes',sha256=pv.raw_sha256_file(p)) for p in sorted(root.rglob('*'))
        if p.is_file() and 'pytest_tmp' not in p.parts and 'gate_only_jit_cache' not in p.parts]
    save('RUNTIME_GATE.json',result)
    if result['status']!='S4_1_LINUX_RUNTIME_GATE_PASS':
        raise RuntimeError(result.get('error','runtime gate failed'))
    return result

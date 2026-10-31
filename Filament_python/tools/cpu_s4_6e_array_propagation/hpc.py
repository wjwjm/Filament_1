"""Single authorized allocation, fixed 14 serial processes, existing exact gates."""
from __future__ import annotations
import argparse
import datetime as dt
import getpass
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_5b_steady_cost.hpc import command, seal_reference_files
from tools.cpu_s4_fullsize_confirmation.controller import _tree_rss
from tools.hpc_ops import provenance_v2 as pv
from . import entry
from .protocol import MATRIX, REMOTE_PYTHON, SHAPE, THREAD_ENV, VERSIONS, plan

require = entry.require


def preflight(report):
    """Read-only: no run root, lock, receipt, propagation or scheduler call."""
    from tools.hpc_ops.audit_batch_entry import audit_batch_entry
    require(getpass.getuser() == 't0s000727', 'S46E_T0_ONLY')
    require(Path(sys.executable).absolute() == Path(REMOTE_PYTHON), 'S46E_FIXED_INTERPRETER')
    require(sys.version_info[:3] == (3, 12, 8), 'S46E_PYTHON_PATCH')
    versions = {name: importlib.import_module(name).__version__ for name in VERSIONS}
    require(versions == VERSIONS, 'S46E_FROZEN_DEPENDENCIES', versions)
    require(os.environ.get('UPPE_USE_GPU') == '0' and all(os.environ.get(k) == v for k, v in THREAD_ENV.items()), 'S46E_FROZEN_CPU_THREAD_ENV')
    binding = entry.verify_source(report, local_engineering=False)
    pv.validate_manifest(Path(__file__).resolve().parents[3], report / 'EXECUTION_SOURCE_MANIFEST.json', require_clean=True, require_hash_scope=True)
    audit, code = audit_batch_entry(Path(__file__).with_name('cpu.sbatch'), REMOTE_PYTHON, 'bash')
    require(code == 0 and audit['status'] == 'passed', 'S46E_BATCH_ENTRY_AUDIT', audit)
    help_result = command([sys.executable, '-s', '-B', '-m', 'tools.cpu_s4_6e_array_propagation.entry', '--help'])
    from .candidate import build_role, verify_bindings
    factories = {}
    for role in ('R', 'A'):
        built = build_role('PURE_OFF_SMALLGRID', capture_root=report / ('UNUSED_PREFLIGHT_'+role), role=role, local_engineering=False)
        try:
            factories[role] = verify_bindings(built[1])
        finally:
            built[1].close_intensity_pool()
        require(built[1].intensity_pool_snapshot()['closed'], 'S46E_PREFLIGHT_FACTORY_CLOSED')
    return dict(status='PASS_READONLY_BATCH_SOURCE_ENVIRONMENT_PREFLIGHT', execution_commit=binding['execution_commit'], executable=sys.executable, versions=versions, batch_audit=audit, entry_help_checked=bool(help_result), factories=factories, propagation_called=False)


def node_gate(report):
    import psutil
    from tools.cpu_o1.cpu_s4.memory_limits import snapshot_cgroup_memory
    checked = preflight(report)
    authorization = entry.read(report / 'RESOURCE_AUTHORIZATION.json')
    require(authorization.get('authorized') is True and authorization.get('execution_commit') == checked['execution_commit'] and authorization.get('resources') == plan()['resources'] and bool(authorization.get('user_instruction')), 'S46E_NEW_EXPLICIT_RESOURCE_AUTHORIZATION_REQUIRED')
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdigit() and os.environ.get('SLURM_RESTART_COUNT', '0') == '0', 'S46E_ONE_NEW_NON_REQUEUED_JOB')
    require(os.environ.get('SLURM_CPUS_PER_TASK') == '96' and os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'S46E_ONE_NODE_CPU96')
    description = command(['scontrol', 'show', 'job', job, '-o'])
    fields = dict(word.split('=', 1) for word in description.split() if '=' in word)
    require(fields.get('NumCPUs') == '96' and fields.get('NumNodes') == '1' and fields.get('TimeLimit') == '00:30:00' and fields.get('Account') == 'bscc-t' and fields.get('Partition') == 'v6_384', 'S46E_EXACT_RESOURCES', description)
    tres = dict(word.split('=', 1) for word in fields['AllocTRES'].split(','))
    require(tres.get('mem') == '32G' and 'gres/gpu' not in description and not os.environ.get('SLURM_JOB_GPUS'), 'S46E_32G_ZERO_GPU')
    affinity = psutil.Process().cpu_affinity()
    topology = command(['lscpu', '-p=CPU,CORE,SOCKET,NODE'])
    rows = [line.split(',') for line in topology.splitlines() if line and not line.startswith('#')]
    cores = {(r[2], r[1]) for r in rows if int(r[0]) in affinity}
    numas = {r[3] for r in rows if int(r[0]) in affinity}
    hardware = command(['lscpu'])
    require(len(affinity) == len(cores) == 96 and len(numas) == 4 and 'Intel(R) Xeon(R) Platinum 9242' in hardware, 'S46E_FROZEN_96_PHYSICAL_CORE_NODE')
    memory = snapshot_cgroup_memory(os.getpid(), allocation_bytes=32*1024**3)
    require(memory.get('status') == 'READ' and memory['effective_limit_bytes'] > 0, 'S46E_ACTUAL_CGROUP')
    guard = int(.8 * min(32*1024**3, memory['effective_limit_bytes']))
    require(guard > 20*1024**3, 'S46E_EXISTING_CONSERVATIVE_MEMORY_FORECAST')
    leaf = Path(memory['levels_leaf_to_mount_root'][0]['path'])
    peak_path = leaf / ('memory.max_usage_in_bytes' if memory['version'] == 1 else 'memory.peak')
    require(int(peak_path.read_text().strip()) < guard, 'S46E_INITIAL_MEMORY_GUARD')
    data = Path(os.environ['CPU_S4_DATA_ROOT']).resolve(strict=True)
    contract = entry.read(report / 'INPUT_CONTRACT.json')
    require(pv.raw_sha256_file(data / 'SMALLGRID_CONFIG.json') == contract['input_contract']['config_raw_sha256'], 'S46E_FROZEN_CONFIG')
    for row in entry.read(data / 'raw/FROZEN_LUT_RETRIEVAL.json'):
        path = data / 'inputs' / row['path']
        require(row['classification'] == 'external' and row['hash_scope'] == 'raw_bytes' and path.resolve().is_relative_to(data) and path.stat().st_size == row['size_bytes'] and pv.raw_sha256_file(path) == row['raw_sha256'], 'S46E_FROZEN_LUT', row['path'])
    refs = seal_reference_files(report)
    node = dict(status='PASS_S46E_SMALLGRID_SOURCE_RESOURCE_INPUT_GATES', job_id=job, preflight=checked,
                allocation=description, hardware=hardware, affinity=affinity, physical_cores=len(cores), numa_nodes=sorted(numas),
                memory=memory, process_tree_stop_bytes=guard, memory_peak_path=str(peak_path), sealed_reference_files=refs,
                default='I0', production_qualified=False)
    sg.write(report / 'NODE_GATE.json', node)
    return node


def launch(report, spec, reference, *, deadline, node):
    import psutil
    from tools.cpu_o1.cpu_s4.supervisor import terminate
    case, role, intervals, kind = spec
    require(time.monotonic() < deadline, 'S46E_DEADLINE_BEFORE_NEXT_WINDOW')
    folder = report / 'runs/A01' / case
    folder.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-s', '-B', '-m', 'tools.cpu_s4_6e_array_propagation.entry', '--report', str(report), '--work', str(folder / 'work'), '--role', role, '--intervals', str(intervals), '--input-manifest', str(report / 'BENCHMARK_INPUT_MANIFEST.json')]
    if reference:
        for flag, key in (('reference-output', 'output'), ('reference-boundary', 'boundary_path'), ('reference-stages', 'stages')):
            if reference.get(key) and (intervals == 2 or key == 'output'):
                argv.extend(('--'+flag, reference[key]))
    env = os.environ.copy()
    env.pop('PYTEST_ADDOPTS', None)
    env['NUMBA_CACHE_DIR'] = str(report / 'cache' / role)
    sg.write(folder / 'LAUNCH.json', dict(argv=argv, case=case, role=role, intervals=intervals, kind=kind, UTC=dt.datetime.now(dt.timezone.utc).isoformat(), warmup_intervals=0, factor_cache='EMPTY_PRIVATE_NEW_PROCESS', JIT_cache=env['NUMBA_CACHE_DIR']))
    with (folder / 'stdout.log').open('xb') as stdout, (folder / 'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    created = psutil.Process(process.pid).create_time()
    started = time.monotonic()
    peak = kernel_peak = samples = 0
    reason = None
    limit, peak_path = node['process_tree_stop_bytes'], Path(node['memory_peak_path'])
    try:
        while process.poll() is None:
            child = _tree_rss(process)
            if child < 0:
                reason = 'PROCESS_IDENTITY_LOST'
                break
            peak = max(peak, child + psutil.Process().memory_info().rss)
            kernel_peak = max(kernel_peak, int(peak_path.read_text().strip()))
            samples += 1
            if max(peak, kernel_peak) >= limit:
                reason = 'ACTUAL_MEMORY_GUARD_80_PERCENT'
                break
            if time.monotonic() >= deadline:
                reason = 'WORK_DEADLINE_1680_SECONDS'
                break
            time.sleep(1)
    finally:
        if process.poll() is None:
            terminate(process.pid, created)
    code = process.wait(timeout=15)
    kernel_peak = max(kernel_peak, int(peak_path.read_text().strip()))
    if kernel_peak >= limit:
        reason = 'KERNEL_MEMORY_HIGH_WATER_GUARD'
    receipt = dict(exit_code=code, stop_reason=reason, process_service_s=time.monotonic()-started, peak_tree_rss_bytes=peak, kernel_cgroup_peak_bytes=kernel_peak, memory_samples=samples)
    sg.write(folder / 'SUPERVISOR_RECEIPT.json', receipt)
    if code != 0 and (folder / 'work/ENTRY_FAILURE.json').is_file():
        failure = entry.read(folder / 'work/ENTRY_FAILURE.json')
        require(failure['classification'] != 'SCIENTIFIC_NONEXACT', 'S46E_STOP_SCIENCE_NONEXACT:'+case, failure)
    require(code == 0 and reason is None, 'S46E_STOP_CHILD_FAILURE:'+case, receipt)
    result = entry.read(folder / 'work/CASE_RECEIPT.json')
    result.update(case_id=case, kind=kind, supervisor=receipt)
    require(result['execution_domain'] == 'S46E_SMALLGRID' and result['shape'] == list(SHAPE), 'S46E_CASE_DOMAIN')
    print(json.dumps(dict(case=case, status=result['status'], application_service_s=result['application_service_s'])), flush=True)
    return result


def exact(result, *, terminal=True, boundary=False, stages=False):
    for name in (('terminal',) if terminal else ()) + (('boundary',) if boundary else ()) + (('stages',) if stages else ()):
        expected = 'PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT' if name == 'boundary' else 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
        require(result['comparisons'][name]['status'] == expected, 'S46E_STOP_SCIENCE_NONEXACT:'+name)
    nH = 2*result['intervals']
    route, stats = result['route'], result['cache']['stats']
    require(route['nH'] == nH and route['fresh_responses'] == 2*nH and route['math_dispatches'] == 8*nH and route['fallback'] == 0, 'S46E_ORIGINAL_ROUTE_COUNTS')
    require(all(value == nH for value in route['lifecycle'].values()) and all(route['hash_roles'].get(key) == nH for key in ('Omega_stage1', 'Omega_apply', 'Omega_stage2')), 'S46E_ORIGINAL_RETIRED_CAPABILITIES')
    require(stats['factor_requests'] == 384*nH and stats['plane_builds'] == 384 and stats['hits'] == 384*(nH-1) and result['cache_fallback'] == 0, 'S46E_CACHE_COUNTS')
    for row in route['actual_numba_kernel_executions']:
        require(row['requested_workers'] == row['active_thread_count'] == 48 and row['active_thread_ids'] == list(range(48)) and (row['thread_mask_before'], row['thread_mask_inside'], row['thread_mask_after']) == (24,48,24), 'S46E_REAL_W48_ORIGINAL_KERNEL_MASKS')
    result['exact_gate_pass'] = bool(terminal)
    result['terminal_exact_basis'] = 'COMPARED_CURRENT_WINDOW_TO_REFERENCE' if terminal else 'REFERENCE_GENERATED_AWAIT_A2_TERMINAL_PAIR'


def run(report):
    from .analysis import analyze
    report = Path(report).resolve(strict=True)
    started = time.monotonic()
    cases = {}
    try:
        node = node_gate(report)
        deadline = started + 1680
        inputs = sg.input_manifest(report)
        require(inputs['fingerprints'] == entry.read(report / 'INPUT_CONTRACT.json')['expected_input_fingerprints'], 'S46E_FROZEN_DIRECT_SMALLGRID_INPUTS')
        old = entry.read(report / 'REFERENCE_INPUT_RAW_RECORDS.json')['reference']
        for spec in MATRIX:
            case, role, intervals, kind = spec
            if case == 'QUAL_R2':
                reference = dict(boundary_path=old['boundary_path'], stages=old.get('stages'))
            elif case == 'QUAL_A2':
                reference = cases['QUAL_R2']
            elif case == 'CTRL_EARLY_R1':
                reference = old
            else:
                reference = cases['CTRL_EARLY_R1']
            result = launch(report, spec, reference, deadline=deadline, node=node)
            exact(result, terminal=case != 'QUAL_R2', boundary=intervals == 2, stages=intervals == 2)
            cases[case] = result
            if case != 'QUAL_R2':
                sg.write(report / 'runs/A01' / case / 'ACCEPTED_CASE.json', result)
            if case == 'QUAL_A2':
                cases['QUAL_R2']['exact_gate_pass'] = True
                cases['QUAL_R2']['terminal_exact_basis'] = 'R2_IS_CURRENT_L1_REFERENCE_FOR_EXACT_A2_TERMINAL_PAIR_NOT_HISTORICAL_TERMINAL_EXACT'
                sg.write(report / 'runs/A01/QUAL_R2/ACCEPTED_CASE.json', cases['QUAL_R2'])
                sg.write(report / 'QUALIFIED_STEP_CONTRACT.json', dict(nH=16, intervals=8, qualification_nH=4, qualification_intervals=2, qualification_fresh_responses=8, qualification_original_math_calls=32, qualification_cases=['QUAL_R2', 'QUAL_A2'], r2_historical_terminal_exact=False, r2_source_bound_boundary_stage_exact=True, a2_terminal_exact_against_r2=True, qualification_gate=True))
        result = analyze(cases)
        sg.write(report / 'ACCEPTED_CASES.json', cases)
        sg.write(report / 'RESULT.json', result)
        sg.write(report / 'RESOURCE_LEDGER.json', dict(job_id=node['job_id'], jobs_used=1, retry_authorized=False, serial_windows=14, controller_service_s=time.monotonic()-started, allocated_core_hours_max=48, queue_s='AWAIT_SACCT', actual_allocated_core_hours='AWAIT_SACCT', scheduler_terminal_state='AWAIT_SACCT'))
        return result
    except BaseException as error:
        text = repr(error)
        science = 'SCIENCE_NONEXACT' in text
        sg.write(report / 'JOB_FAILURE.json', dict(classification='SCIENTIFIC_NONEXACT' if science else 'ENGINEERING_OR_RESOURCE_FAILURE', error=text, traceback=traceback.format_exc(), completed_cases=list(cases), default='I0', production_qualified=False, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    result = preflight(args.report.resolve(strict=True)) if args.preflight else run(args.report)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

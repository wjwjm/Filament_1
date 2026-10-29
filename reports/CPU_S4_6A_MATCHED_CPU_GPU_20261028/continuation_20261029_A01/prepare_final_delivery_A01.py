"""One allowlisted final S46A review copy; no propagation or source changes."""
from __future__ import annotations
import csv, json, shutil, sys
from datetime import datetime
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
r = repo / 'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028'
c = r / 'continuation_20261029_A01'
out = r / 'CPU_S4_6A_REVIEW_20261029_002'
sys.path.insert(0, str(repo / 'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv

def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError('COPY_WOULD_OVERWRITE:' + str(dst))
    shutil.copyfile(src, dst)

def csv_write(path, rows):
    if not rows:
        raise RuntimeError('EMPTY_TABLE:' + str(path))
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def prepare():
    out.mkdir(exist_ok=False)
    for name in ('evidence_CPU_SMALL', 'evidence_CPU_FULL', 'evidence_GPU_COMPLETED_A01'):
        shutil.copytree(c / name, out / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(r / 'historical', out / 'historical')
    for old in ('CPU_V1_INVALID', 'CPU_R2_FAILED'):
        shutil.copytree(r / 'terminal_evidence/raw' / old, out / 'failure_history' / old)
    failure_names = [
        'GPU_STATUS_A01.json', 'GPU_REPAIR_STATUS_A01.json', 'GPU_ADDITIONAL_STATUS_A02.json',
        'GPU_REPAIR_QUAL_OFF.stderr.log', 'GPU_ADDITIONAL_QUAL_OFF.stderr.log',
        'GPU_ADDITIONAL_BOUNDARY_FIRST_FAILURE.json', 'GPU_BOUNDARY_METADATA_AUDIT_A01.json',
        'BOUNDARY_READBACK_FIX_REVIEW.json', 'SOURCE_IDENTITY_EXCEPTION.json',
        'SOURCE_IDENTITY_EXCEPTION_GPU_REPAIR.json', 'SOURCE_IDENTITY_EXCEPTION_GPU_ADDITIONAL.json',
        'SOURCE_IDENTITY_EXCEPTION_GPU_BOUNDARY_RETRY.json', 'GPU_BOUNDARY_RETRY_INSTALL.log',
        'GPU_BOUNDARY_STAGING_DIAGNOSIS_A01.log', 'GPU_BOUNDARY_WORKTREE_RECOVERY.log',
        'GPU_BOUNDARY_STAGING_RECONCILED.log', 'GPU_MONITOR_DIAGNOSTIC_RECONCILIATION_A01.json',
        'REPLAY_LINUX_TARGETED_A01.log',
    ]
    for name in failure_names:
        copy(c / name, out / 'failure_history' / name)
    for job, name in [('290147','GPU_STATUS_A01.json'),('290163','GPU_REPAIR_STATUS_A01.json'),('290181','GPU_ADDITIONAL_STATUS_A02.json')]:
        copy(c / name, out / 'failure_history' / ('GPU_' + job + '_SCHEDULER_ORIGINAL.json'))
    for name in ('INSTRUMENT_VERSION_HISTORY.json', 'FAILURE_AND_VERSION_LOG.json'):
        copy(r / name, out / 'failure_history/prior_closeout' / name)
    auth = ['AUTHORIZED_RESUME.md', 'CPU_FULL_PARALLEL_AUTHORIZATION.json',
            'GPU_ADDITIONAL_AUTHORIZATION.json', 'GPU_BOUNDARY_RETRY_AUTHORIZATION.json',
            'GPU_TELEMETRY_AND_BUDGET_DECISION.md']
    for name in auth:
        copy(c / name, out / 'authorization' / name)
    copy(r / 'USER_TASK.txt', out / 'specification/USER_TASK.txt')
    for name in ('WORKLOAD_MATCH_CONTRACT.json', 'MEASUREMENT_PLAN.json',
                 'CLOCK_BOUNDARY_CONTRACT.md', 'HISTORICAL_RATIO_RECONCILIATION.md'):
        copy(r / name, out / 'frozen_preregistration' / name)
    shutil.copytree(c / 'canonical', out / 'canonical')
    support = []
    # Complete ONLY the new delivery copy using byte-identical frozen configs.
    for manifest in sorted(out.rglob('INPUT_MANIFEST.json')):
        for rec in load(manifest)['records']:
            if Path(rec['path']).name != 'config.json':
                continue
            destination = manifest.parent / 'config.json'
            if destination.exists():
                continue
            source = c / 'canonical' / manifest.parent.name / 'config.json'
            if pv.raw_sha256_file(source) != rec['raw_sha256']:
                raise RuntimeError('SUPPORT_CONFIG_ORIGINAL_HASH_MISMATCH')
            copy(source, destination)
            support.append(dict(path=destination.relative_to(out).as_posix(),
                                original=c.relative_to(repo).as_posix() + '/canonical/' + manifest.parent.name + '/config.json',
                                hash_scope=pv.EXTERNAL_HASH_SCOPE, raw_sha256=rec['raw_sha256'],
                                original_manifest_unchanged=True))
    write(out / 'DELIVERY_CONFIG_SUPPORT_COPIES.json', dict(
        scope='Only additional byte-identical config files in new portable copy; old evidence unchanged',
        records=support))
    for name in ('CROSS_FULL_NUMERIC_A01', 'evidence_boundary_readback_A01'):
        shutil.copytree(c / name, out / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('FULL_CROSS_EXECUTION_A01.log', 'FULL_CROSS_RELAY_RECEIPT_A01.json',
                 'CROSS_FULL_ADAPTER_SELFTEST_A02.json', 'FROZEN_CONFIG_PV2_VERIFICATION_A01.json',
                 'FROZEN_CONFIG_TRANSFER_RECEIPT.json', 'MATRIX_ADDITIONAL_TRANSFER_RECEIPT.json',
                 'GPU_FULL_LIGHT_INDEPENDENT_REVIEW_A01.json'):
        copy(c / name, out / 'acquisition_receipts' / name)
    for name in ('cross_full_readonly.py',):
        if not (out / 'CROSS_FULL_NUMERIC_A01' / name).is_file():
            raise RuntimeError('CROSS_ADAPTER_MISSING')
    for name in ('scvi_run_full_cross_A01.sh',):
        copy(c / 'control' / name, out / 'acquisition_receipts' / name)
    blobs = {}
    role_rows = []
    for role, manifest in (
        ('CPU', c / 'evidence_CPU_FULL/raw/CPU_FULL/report/EXECUTION_SOURCE_MANIFEST.json'),
        ('GPU', c / 'evidence_GPU_COMPLETED_A01/raw/GPU/report/EXECUTION_SOURCE_MANIFEST.json'),
    ):
        target = out / 'source_roles' / role
        copy(manifest, target / 'EXECUTION_SOURCE_MANIFEST.json')
        mf = load(manifest)
        for record in mf['records']:
            if record['classification'] != 'tracked_text' or record['hash_scope'] != pv.TRACKED_HASH_SCOPE:
                raise RuntimeError('UNEXPECTED_SOURCE_CLASSIFICATION')
            oid = record['git_blob_oid']
            data = blobs.setdefault(oid, None)
            if data is None:
                data = pv._read_git_blob(repo, oid); blobs[oid] = data
            if pv.canonical_lf_sha256(data) != record['canonical_lf_sha256']:
                raise RuntimeError('SOURCE_CANONICAL_HASH_MISMATCH')
            dest = target / 'executed_source' / record['path']
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
        role_rows.append(dict(role=role, root=target.relative_to(out).as_posix(), execution_commit=mf['head']))
    write(out / 'SOURCE_BUNDLE_INDEX.json', dict(schema='filament.cpu.s46a.source_bundles.v1', bundles=role_rows))
    # Preserve original scoped inputs; add all three failed GPU primary scheduler records.
    scope = load(c / 'REPLAY_SCOPE.json')
    scope['ledgers'] = [
        'failure_history/CPU_V1_INVALID/SCHEDULER_TERMINAL.json',
        'failure_history/CPU_R2_FAILED/SCHEDULER_TERMINAL.json',
        'evidence_CPU_SMALL/raw/CPU_SMALL/SCHEDULER_TERMINAL.json',
        'evidence_CPU_FULL/raw/CPU_FULL/SCHEDULER_TERMINAL.json',
        'failure_history/GPU_290147_SCHEDULER_ORIGINAL.json', 'failure_history/GPU_290163_SCHEDULER_ORIGINAL.json',
        'failure_history/GPU_290181_SCHEDULER_ORIGINAL.json',
        'evidence_GPU_COMPLETED_A01/raw/GPU/SCHEDULER_TERMINAL.json',
    ]
    scope['resource_selection'] = 'Only original main scheduler rows charged once, not batch/extern/step or presentation CSV'
    scope['notes'] = [n for n in scope['notes'] if 'remote_CPU_FULL' not in n]
    write(out / 'REPLAY_SCOPE.json', scope)
    cross = out / 'CROSS_FULL_NUMERIC_A01'
    expected = {
        'CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json':'48d55398516b440132c643358210ca43a1ab96fd6a224cc10e58c74530054ae6',
        'CROSS_FULL_NUMERIC_A01_SIDECAR.json':'5e93e30530bd275dc15f572f830ea7c7c08aab9eaa6f9446887bebef16a753b5',
        'CPU_FULL_CROSS_REFERENCE_MANIFEST_A01.json':'344e3aa8d4de935ab36c657b77f202c2c425e709176ffb06e8e6b5fc116534f6',
    }
    verified = []
    for name, digest in expected.items():
        actual = pv.raw_sha256_file(cross / name)
        if actual != digest:
            raise RuntimeError('FULL_CROSS_TRANSFER_HASH_MISMATCH:' + name)
        verified.append(dict(path='CROSS_FULL_NUMERIC_A01/' + name, classification='external',
                             hash_scope=pv.EXTERNAL_HASH_SCOPE, raw_sha256=actual))
    write(out / 'acquisition_receipts/FULL_CROSS_LOCAL_VERIFICATION_A01.json', dict(
        hash_implementation='Filament_python/tools/hpc_ops/provenance_v2.py',
        records=verified, scientific_arrays_downloaded_to_disk=0))
    # Main resource rows are rebuilt from the original scheduler columns.
    rows = []
    fields_gpu = ['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','AllocCPUS','AllocTRES','Submit','Start','End','TimelimitRaw','ReqMem','MaxRSS','NodeList']
    for relative in scope['ledgers']:
        value = load(out / relative)
        if 'remote_report' in value:
            cmd = next(x for x in value['remote_report']['commands'] if x['argv'][0] == 'sacct')
            fields = fields_gpu; text = cmd['stdout']
        else:
            fields = value['fields']; text = value['stdout']
        main = next(dict(zip(fields, line.split('|'))) for line in text.splitlines()
                    if '.' not in line.split('|')[0])
        backend = 'GPU' if 'gpu' in main['JobName'].lower() else 'CPU'
        elapsed = int(main['ElapsedRaw']); cores = int(main['AllocCPUS'])
        queue = (datetime.fromisoformat(main['Start']) - datetime.fromisoformat(main['Submit'])).total_seconds()
        rows.append(dict(backend=backend, job_id=main['JobIDRaw'], state=main['State'],
                         exit_code=main['ExitCode'], node=main['NodeList'], submit_HKT=main['Submit'],
                         start_HKT=main['Start'], end_HKT=main['End'], queue_s=queue,
                         allocated_wall_s=elapsed, cpus=cores, gpus=int(backend == 'GPU'),
                         allocated_cpu_core_hours=elapsed * cores / 3600 if backend == 'CPU' else 0,
                         allocated_GPU_hours=elapsed / 3600 if backend == 'GPU' else 0,
                         GPU_host_core_hours=elapsed * cores / 3600 if backend == 'GPU' else 0,
                         original_scheduler=relative, batch_extern_charged=False))
    csv_write(out / 'RESOURCE_LEDGER.csv', rows)
    write(out / 'RESOURCE_AND_STOP_STATE.json', dict(
        status='TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED', inflight_allocations=0,
        cpu_job_ids=[row['job_id'] for row in rows if row['backend']=='CPU'],
        gpu_job_ids=[row['job_id'] for row in rows if row['backend']=='GPU'],
        cpu_allocated_core_hours=sum(row['allocated_cpu_core_hours'] for row in rows),
        gpu_allocated_hours=sum(row['allocated_GPU_hours'] for row in rows),
        gpu_host_core_hours=sum(row['GPU_host_core_hours'] for row in rows),
        caps=dict(cpu_job_ids=4,gpu_job_ids=4,cpu_core_hours=240,gpu_hours=2,gpu_host_core_hours=16),
        local_engineering_time='UNMEASURED', monitoring_wait_time='UNMEASURED', repair_rework_time='UNMEASURED',
        propagation_samples='Only current P1-P3 OFF; qualification/reference/failed/historical/AUX excluded',
        further_submissions_authorized=False, DEFAULT='I0', PRODUCTION_QUALIFIED=False,
        MANUAL_REVIEW_PENDING=True))
    forbidden = {'.npy','.npz','.mat','.h5','.hdf5','.zip','.gz','.bundle','.pyc'}
    for p in out.rglob('*'):
        if p.is_file() and (p.suffix.lower() in forbidden or '__pycache__' in p.parts):
            raise RuntimeError('FORBIDDEN_REVIEW_MEMBER:' + str(p))
    print(json.dumps(dict(directory=str(out), source_roles=role_rows, unique_git_blobs=len(blobs),
                          source_files=sum(len(load(out/row['root']/'EXECUTION_SOURCE_MANIFEST.json')['records']) for row in role_rows),
                          files=sum(p.is_file() for p in out.rglob('*')), resource_rows=len(rows),
                          support_config_copies=len(support), arrays=0)))

if __name__ == '__main__':
    prepare()

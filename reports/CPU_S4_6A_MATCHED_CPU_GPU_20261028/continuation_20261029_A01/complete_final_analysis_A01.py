"""Build the small final analysis from the acquired S46A metadata only."""
from __future__ import annotations
import csv, json, shutil, statistics, sys
from collections import Counter
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
r = repo / 'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028'
c = r / 'continuation_20261029_A01'
out = r / 'CPU_S4_6A_REVIEW_20261029_002'
sys.path.insert(0, str(repo / 'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv

def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def text(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f: f.write(value)
def table(p, rows):
    with p.open('x', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    for name in ('replay_readonly.py','replay_tests.py','build_measurement_tables.py',
                 'MATCHED_RUNTIME_TABLE.csv','GAP_CONTRIBUTIONS.csv','PROFILE_DIAGNOSTIC_COSTS.csv'):
        shutil.copyfile(c/name, out/name)
    a = load(c/'ACTUAL_ALL_BACKENDS_REPLAY_A01.json')
    # Whitelist only portable input/metadata manifests; verify executed-source
    # manifests against their separate role bundle, not unrelated report paths.
    scope=load(out/'REPLAY_SCOPE.json')
    scope['manifests'] = [
        'evidence_CPU_SMALL/PORTABLE_METADATA_MANIFEST.json',
        'evidence_CPU_FULL/PORTABLE_METADATA_MANIFEST.json',
        'evidence_GPU_COMPLETED_A01/PORTABLE_METADATA_MANIFEST.json',
        'evidence_CPU_SMALL/raw/CPU_SMALL/canonical/384x117x129/INPUT_MANIFEST.json',
        'evidence_CPU_SMALL/raw/CPU_SMALL/canonical/384x351x301/INPUT_MANIFEST.json',
        'evidence_CPU_FULL/raw/CPU_FULL/canonical/384x117x129/INPUT_MANIFEST.json',
        'evidence_CPU_FULL/raw/CPU_FULL/canonical/384x351x301/INPUT_MANIFEST.json',
        'evidence_GPU_COMPLETED_A01/raw/GPU/canonical/384x117x129/INPUT_MANIFEST.json',
        'evidence_GPU_COMPLETED_A01/raw/GPU/canonical/384x351x301/INPUT_MANIFEST.json',
        'SOURCE_BUNDLE_INDEX.json',
    ]
    (out/'REPLAY_SCOPE.json').write_text(json.dumps(scope,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    contract=load(r/'WORKLOAD_MATCH_CONTRACT.json')
    contract.update(status='COMMON_FROZEN_INPUTS_NODE_RAW_BYTES_AND_CURRENT_WORKLOAD_GATES_PASSED',
        cpu_execution_commit='ed8754041625549a80959db7a8abd804d5d219ae',
        gpu_execution_commit='6440ace43e07523ef924c790055b3cc21e7bc1fc',
        node_proofs=['evidence_CPU_FULL/raw/CPU_FULL/report/NODE_GATE.json',
                     'evidence_GPU_COMPLETED_A01/raw/GPU/report/NODE_GATE.json'],
        cross_backend_numerical_equivalence='NOT_QUALIFIED_NO_TOLERANCE_CONTRACT',
        report_scope='Both original-size and small-grid current P1-P3 OFF',
        source_binding='SOURCE_BUNDLE_INDEX.json')
    write(out/'WORKLOAD_MATCH_CONTRACT.json',contract)
    binding=load(r/'SOURCE_AND_BACKEND_BINDINGS.json')
    binding.pop('execution_commit',None)
    binding.update(cpu_execution_commit=contract['cpu_execution_commit'],gpu_execution_commit=contract['gpu_execution_commit'],
        source_provenance='SOURCE_BUNDLE_INDEX.json',gpu_hardware='RTX 5090; 33668988928 device bytes; m4gl1701; 8 host physical cores',
        cpu_hardware='Xeon Platinum 9242; 96 physical cores; SMT1; 4 NUMA; cc0902 small / ce0402 full',
        effective_cpu_ionization=dict(module='tools.cpu_o1.cpu_o1',
            callable='make_joint_propagator_factory.<locals>.factory.<locals>.evolve_candidate',
            downstream='evolve_rho_time_cpu_o1',implementation='numba_parallel',kernel_threads=24,block_pixels=32,layout='NtB',
            evidence='Source-bound factory chain; the runtime binding JSON does not directly serialize this closure identity'),
        report_tip='Recorded externally in GIT_FINAL_DELIVERY_RECEIPT.json after commit; execution commits remain distinct')
    write(out/'SOURCE_AND_BACKEND_BINDINGS.json',binding)
    for name in ('CLOCK_BOUNDARY_CONTRACT.md','HISTORICAL_RATIO_RECONCILIATION.md'):
        shutil.copyfile(r/name,out/name)
    shutil.copyfile(c/'evidence_GPU_COMPLETED_A01/raw/GPU/report/MEASUREMENT_PLAN.json',out/'MEASUREMENT_PLAN.json')
    cpu_report=out/'evidence_CPU_FULL/raw/CPU_FULL/report'
    gpu_report=out/'evidence_GPU_COMPLETED_A01/raw/GPU/report'
    small_report=out/'evidence_CPU_SMALL/raw/CPU_SMALL/report'
    observations=[]; transparency=[]; parallel=[]; retention=[]; next_leaf=[]
    sets=[('CPU','384x117x129',small_report),('CPU','384x351x301',cpu_report),
          ('GPU','384x117x129',gpu_report),('GPU','384x351x301',gpu_report)]
    for backend,shape,report in sets:
        for case in ['QUAL_OFF','QUAL_LIGHT','P1_OFF','P1_LIGHT','P2_LIGHT','P2_OFF','P3_OFF','P3_LIGHT']:
            path=report/'runs'/backend/shape/case/'work/CASE_RECEIPT.json'; v=load(path)
            comparisons=v.get('comparisons',{})
            transparency.append(dict(backend=backend,shape=shape,case=case,status=v['status'],
                path=path.relative_to(out).as_posix(), comparisons=comparisons,
                input_identity=v.get('input_identity'), numeric_equivalence_across_backends=False))
            if case.endswith('_OFF') and case.startswith('P'):
                raw_path=path.parent/('RAW_CPU_TIMING.json' if backend=='CPU' else 'RAW_GPU_TIMING.json')
                raw=load(raw_path)
                rows=raw['merged']['records'] if backend=='CPU' else raw['records']
                bodies=[row for row in rows if row['label']=='INTERVAL_BODY']
                def idx(row):
                    return int(row.get('normalized_interval',{}).get('interval_index',row.get('interval',-1)))
                def cpu(row): return int(row.get('cpu_inclusive_ns',row.get('process_cpu_ns',0)))
                first=next(row for row in bodies if idx(row)==0)
                clocks=[('application',v['application_service_s'],v['application_process_cpu_s']),
                        ('window',v['window_completed_s'],v['window_process_cpu_s']),
                        ('first_interval',v['first_interval_s'],cpu(first)/1e9),
                        ('later2to8',v['later_body_s'],sum(cpu(row) for row in bodies if 1<=idx(row)<=7)/1e9)]
                supervisor=load(path.parent.parent/'SUPERVISOR_RECEIPT.json')
                for label,wall,pcpu in clocks:
                    parallel.append(dict(backend=backend,shape=shape,case=case,scope=label,wall_s=wall,
                        process_cpu_s=pcpu,cpu_occupancy_equivalent=pcpu/wall,physical_cores=96 if backend=='CPU' else 8,
                        sampling='Exact process_time_ns / completed perf_counter_ns; later CPU is sum of interval bodies',
                        H2D_count='',H2D_known_bytes='',D2H_count='',D2H_known_bytes='',
                        memory=json.dumps(supervisor,separators=(',',':')),
                        source=path.relative_to(out).as_posix(),
                        limitation='Occupancy equivalent is not fixed active cores, useful parallel fraction or speedup potential'))
                if backend=='GPU':
                    transfers=v.get('transfers',[])
                    parallel.append(dict(backend=backend,shape=shape,case=case,scope='known_explicit_transfers',wall_s='',
                        process_cpu_s='',cpu_occupancy_equivalent='',physical_cores=8,
                        sampling='Known explicit input/output calls only, not a trace',
                        H2D_count=sum(x['direction']=='H2D' for x in transfers),
                        H2D_known_bytes=sum(x['bytes'] for x in transfers if x['direction']=='H2D'),
                        D2H_count=sum(x['direction']=='D2H' for x in transfers),
                        D2H_known_bytes=sum(x['bytes'] for x in transfers if x['direction']=='D2H'),
                        memory=json.dumps(v.get('memory_pool'),separators=(',',':')),
                        source=path.relative_to(out).as_posix(),limitation=v.get('transfer_accounting_scope','UNKNOWN')))
                index=path.parent/'OUTPUT_RAW_INDEX.json'
                if index.is_file():
                    retention.append(dict(backend=backend,shape=shape,case=case,
                        index=index.relative_to(out).as_posix(),data=v.get('normal_outputs',v.get('output')),
                        index_raw_sha256=pv.raw_sha256_file(index),hash_scope='raw_bytes',
                        local_scientific_array_bytes=0,offline_array_recheck=False,
                        retention='Leave original remote output and qualification snapshots unchanged; no deletion authorized'))
            if backend=='CPU' and shape=='384x351x301' and case.endswith('_LIGHT') and case.startswith('P'):
                raw=load(path.parent/'RAW_CPU_TIMING.json')
                selected=[row for row in raw['merged']['records'] if row['label']=='RAMAN_ENERGY'
                          and row.get('raw_context',{}).get('stage')=='none']
                next_leaf.append(dict(case=case,count=len(selected),wall_ns=sum(x['wall_exclusive_ns'] for x in selected),
                    process_cpu_ns=sum(x['cpu_exclusive_ns'] for x in selected),
                    window_s=v['window_completed_s'],application_s=v['application_service_s']))
    write(out/'BACKEND_TRANSPARENCY_EXACT_INDEX.json',dict(schema='filament.cpu_s46a.exact_index.v2',
        same_backend_only=True,original_comparison_rules_unchanged=True,
        GPU_two_resource_fields='Only gpu_allocated_step_bytes/gpu_reserved_step_bytes separately retained under approved policy',
        boundary='112 original fields; science/NaN/timing exceptions and saved native layout/alias gates retained',
        records=transparency,DEFAULT='I0',PRODUCTION_QUALIFIED=False,MANUAL_REVIEW_PENDING=True))
    table(out/'PARALLELISM_TRANSFER_AND_MEMORY.csv',parallel)
    write(out/'SCIENTIFIC_ARRAY_RETENTION.json',dict(hash_scope='raw_bytes',scientific_arrays_in_package=0,
        original_remote_outputs=retention,
        full_cross_cpu_relay=dict(bytes=966344944,files=11,local_disk_arrays=0,
            source_target_verified=True,receipt='acquisition_receipts/FULL_CROSS_RELAY_RECEIPT_A01.json'),
        original_snapshot_indexes='Each case OUTPUT_RAW_INDEX and original boundary fingerprints retained under evidence trees',
        offline_replay='Metadata only; array bytes intentionally not rechecked offline'))
    full=load(out/'CROSS_FULL_NUMERIC_A01/CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json')
    small=load(gpu_report/'CROSS_BACKEND_NUMERIC_DIFFERENCES_SMALL.json')
    numeric=[]
    for shape,v,p in [('384x117x129',small,gpu_report/'CROSS_BACKEND_NUMERIC_DIFFERENCES_SMALL.json'),
                      ('384x351x301',full,out/'CROSS_FULL_NUMERIC_A01/CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json')]:
        numeric.append(dict(shape=shape,status=v['status'],matched_input_workload=v['matched_input_workload'],
            tolerance_contract=v.get('tolerance_contract'),field_count=len(v['fields']),
            original=p.relative_to(out).as_posix(),hash_scope='raw_bytes',raw_sha256=pv.raw_sha256_file(p),
            selected=[x for x in v['fields'] if x['field'] in {'E','final_E','I_max_z','U_z','rho_onaxis_max_z'}]))
    write(out/'CROSS_BACKEND_NUMERIC_DIFFERENCES.json',dict(status='CROSS_BACKEND_NUMERICAL_EQUIVALENCE_NOT_QUALIFIED',
        new_epsilon=None,equivalence_qualified=False,datasets=numeric,
        comparator='Original cross_small body; only four SMALL->FULL acquisition path literals and function name adapted',
        full_source_proof='CROSS_FULL_NUMERIC_A01/CROSS_FULL_NUMERIC_A01_SIDECAR.json'))
    write(out/'NEXT_CPU_COST_MODEL.json',dict(scope='Only no-stage stable_field_fluence_and_loss RAMAN_ENERGY exclusive leaf',
        profile_samples=next_leaf,
        leaf_median_s=statistics.median(x['wall_ns']/1e9 for x in next_leaf),
        leaf_range_s=max(x['wall_ns']/1e9 for x in next_leaf)-min(x['wall_ns']/1e9 for x in next_leaf),
        leaf_MAD_s=statistics.median(abs(x['wall_ns']/1e9-statistics.median(y['wall_ns']/1e9 for y in next_leaf)) for x in next_leaf),
        scenarios=[dict(local_speedup=k,cases=[dict(case=x['case'],
            saved_s=x['wall_ns']/1e9*(1-1/k),
            hypothetical_window_s=x['window_s']-x['wall_ns']/1e9*(1-1/k),
            window_speedup=x['window_s']/(x['window_s']-x['wall_ns']/1e9*(1-1/k)),
            hypothetical_application_s=x['application_s']-x['wall_ns']/1e9*(1-1/k),
            application_speedup=x['application_s']/(x['application_s']-x['wall_ns']/1e9*(1-1/k))) for x in next_leaf]) for k in (2,4)],
        added_cost='UNKNOWN; zero-extra-cost hypothetical upper scenarios, not forecast or GPU lower bound',
        implementation_authorized_by_this_task=False))
    source_rows=[
        ('completed_later','Filament_python/KHz_filament/propagate.py:628','7 intervals, all operators plus normal work','native OFF completed-wall','MEASURED','Operator allocation UNKNOWN'),
        ('window_remainder','Filament_python/KHz_filament/propagate.py:628','first interval plus entry/finalization and loop gaps','native OFF completed-wall','MEASURED','Not a single numerical operator'),
        ('outside','Filament_python/tools/cpu_s4_6a/entry.py:60; gpu.py:324','application minus full propagation window','native OFF completed-wall','MEASURED','Cold-process management/input/writer costs combined'),
        ('Raman_Heun','Filament_python/tools/cpu_s4_3c/candidate.py:267; KHz_filament/raman.py:163','16 Raman halves /32 stages; existing C3 prange point helpers','CPU valid LIGHT exclusive','MEASURED_CPU_SOURCE_SUPPORTED','GPU native operator delta UNKNOWN'),
        ('stable_field_fluence_and_loss','Filament_python/KHz_filament/raman.py:479','16 no-stage calls, chunk_t32, real then imag, float64 sums','CPU valid LIGHT exclusive','MEASURED_CPU_SOURCE_SUPPORTED','Logical temporaries do not prove hardware DRAM saturation'),
        ('ionization','Filament_python/tools/cpu_o1/cpu_o1.py:1781','8 recursions; LUT species loop then existing prange pixel blocks','CPU valid LIGHT; source-bound effective factory','SOURCE_AND_CPU_MEASUREMENT','No direct serialized closure identity or busy kernel trace'),
        ('linear','Filament_python/tools/cpu_s4_4a_linear_cache/candidate.py:99; KHz_filament/linear.py','16 half steps, temporal/spatial FFT and accepted full cache','CPU valid LIGHT; GPU source','SOURCE_AND_CPU_MEASUREMENT','No native full-size cross-backend operator delta'),
        ('GPU_device_activity','Filament_python/tools/cpu_s4_6a/gpu.py:96','LIGHT event spans and separate SERIALIZED_PHASE_DIAGNOSTIC','async span / serialized own wall','LIMITED','No kernel activity trace/counters; event span includes idle gaps'),
    ]
    table(out/'DIRECTED_SOURCE_TABLE.csv',[dict(scope=x[0],source=x[1],workload=x[2],clock=x[3],evidence_strength=x[4],limitation=x[5]) for x in source_rows])
    oldfail=load(r/'FAILURE_AND_VERSION_LOG.json')
    write(out/'FAILURE_AND_VERSION_LOG.json',dict(prior_closeout=oldfail,
        continuation_failures=[dict(job='290147',reason='cgroup memory.peak unavailable at guardian startup; no scientific sample'),
            dict(job='290163',reason='Resource reserved-vector difference blocked original exact; approved telemetry-only split'),
            dict(job='290181',reason='np.load OWNDATA differs from saved native boundary metadata; original saved 112 audited; bounded readback repair'),
            dict(job='290195',reason='Scheduler COMPLETED; full LIGHT timing gate failed; OFF retained, no fifth retry')],
        acquisition='Final readonly full QUAL_OFF cross numerical comparison; no new propagation or JobID',
        replay_rework='Prior Linux /mnt/c targeted run terminated (-15), not PASS; final native-Linux package validation recorded separately',
        package_copy_rework='Only copied __pycache__ rejected and removed from new delivery copy; originals retained',
        records='failure_history/ and authorization/; immutable raw first failures remain available'))
    write(out/'MEASUREMENT_COMPLETENESS.json',dict(status='TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED',
        current_OFF_samples_per_backend_shape=3,backend_shape_count=4,OFF_total_and_macro_wall_valid=True,
        cpu_small_LIGHT='PASS',cpu_full_LIGHT='PASS',gpu_small_LIGHT='PASS',gpu_full_LIGHT='FAIL',
        gpu_full_NATIVE_operator_deltas='UNKNOWN',native_OFF_macro_arithmetic_closure='0',
        scientific_arrays_rechecked_offline=False,cross_backend_numerical_equivalence=False,
        remaining_gap='Full native GPU operator attribution, kernel activity, hardware counters and exact hardware-cause decomposition',
        DEFAULT='I0',PRODUCTION_QUALIFIED=False,MANUAL_REVIEW_PENDING=True))
    # Descriptive scalar summary; final replay output is generated independently.
    write(out/'MEASURED_RESULT_SUMMARY.json',dict(cpu_gpu_ratios=a['cpu_gpu_ratios'],
        native_off_macro_summaries=a['native_off_macro_summaries'],light_exclusive_summaries=a['light_exclusive_summaries'],
        aux_serialized_summaries=a['aux_serialized_summaries'],source='Current original CASE_RECEIPT and RAW timings',
        DEFAULT='I0',PRODUCTION_QUALIFIED=False,MANUAL_REVIEW_PENDING=True))
    print(json.dumps(dict(directory=str(out),transparency_records=len(transparency),parallel_rows=len(parallel),
        next_leaf=next_leaf,numeric=[dict(shape=x['shape'],fields=x['field_count'],selected=x['selected']) for x in numeric]),ensure_ascii=False))

if __name__=='__main__': main()

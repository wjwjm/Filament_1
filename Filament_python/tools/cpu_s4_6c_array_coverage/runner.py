"""Fixed real-input mechanism experiment; never invokes a propagation window."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
import zipfile

import numpy as np
import numba
import psutil
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_checkpoint_support import support_module
from tools.cpu_s4_6a.inputs import write, require
from . import chains


def append(path, record):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')


def stage_kwargs(ctx):
    from KHz_filament.constants import c0
    from KHz_filament.raman import resolve_raman_rot_params
    r = ctx['raman']
    omega, gamma = resolve_raman_rot_params(T2=getattr(r, 'T2', None), T_R=getattr(r, 'T_R', None),
                                          omega_R=r.omega_R, Gamma_R=r.Gamma_R)
    return dict(Omega=ctx['axes'].Omega, dt=ctx['axes'].dt,
                omega0=2.0*np.pi*c0/ctx['beam'].lam0, n0=ctx['beam'].n0,
                n_R=r.n_R, omega_R=omega, Gamma_R=gamma, method=r.method,
                chunk_pixels=r.chunk_pixels, iir_sampling=r.iir_sampling)


def fresh_factory(capture, local=False):
    from tools.cpu_s4_4a_linear_cache.candidate import build_role
    built = build_role('PURE_OFF_SMALLGRID', capture_root=capture,
                       local_engineering=local, pool_override=1 if local else None)
    state = built[1].cpu_s4_4a_state.r8_state
    fft = state.raman_namespace['xp'].fft
    return built, state, fft


def decode(node, archive, exact):
    kind = node['kind']
    if kind == 'array':
        with archive.open(node['file']) as f:
            a = np.lib.format.read_array(f, allow_pickle=False)
        # Restore the recorded native view ownership; this is not a layout copy.
        if not node['layout']['OWNDATA']:
            require(node['layout']['base_chain_depth'] == 1, 'DECLARED_VIEW_DEPTH')
            a = a.view()
        require(exact.layout(a) == node['layout'], 'CAPTURE_NATIVE_LAYOUT:' + node['file'])
        return a
    if kind == 'dict':
        return {k: decode(v, archive, exact) for k, v in node['items'].items()}
    if kind in ('list', 'tuple'):
        values = [decode(v, archive, exact) for v in node['items']]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'scalar':
        return np.asarray(node['value'], dtype=node['dtype'])[()]
    return node['value']


def check_frozen(value, prefix, frozen, exact):
    actual = exact.fingerprints(value)
    for name, row in actual.items():
        original = prefix + name[len('bundle'):]
        require(row == frozen[original], 'CAPTURE_CONTENT_IDENTITY:' + original)
    return actual


def observation():
    proc = psutil.Process()
    mem = proc.memory_full_info()
    threads = {t.id: t.user_time+t.system_time for t in proc.threads()}
    pages = Path('/proc/self/numa_maps')
    placement = {}
    if pages.is_file():
        for line in pages.read_text().splitlines():
            for item in line.split():
                if item.startswith('N') and '=' in item and item[1:item.index('=')].isdigit():
                    k, v = item.split('='); placement[k] = placement.get(k, 0) + int(v)
    return dict(rss=mem.rss, pss=getattr(mem, 'pss', None), affinity=proc.cpu_affinity(),
                numba_mask=numba.get_num_threads(), threads=threads, numa_pages=placement)


def thread_delta(before, after):
    return [{'id': int(t), 'cpu_s': max(0., after.get(t, 0.)-before.get(t, 0.))}
            for t in sorted(set(before)|set(after))]


def configurations():
    return [('REF', 0)] + [('P', n) for n in chains.WORKERS]


def call(chain, config, data, pool, refs, fft):
    mode, count = config
    E, I, response, kwargs = data
    if chain == 'A':
        return refs[0](E, kwargs['n0']) if mode == 'REF' else chains.intensity(E, kwargs['n0'], pool, mode)
    args = (I, response, kwargs['Omega'], kwargs['dt'], kwargs['n_R'])
    return refs[1](*args) if mode == 'REF' else chains.eq10(*args, fft, pool, mode)


def qualify(chain, identity, data, refs, fft, exact, root):
    expected = call(chain, ('REF', 0), data, None, refs, fft)
    fingerprint = exact.fingerprints({'E':data[0], 'I':data[1], 'response':data[2], 'Omega':data[3]['Omega']})
    output_hash = exact.fingerprints(expected)
    for config in [('P', n) for n in chains.WORKERS] + [('P_LOCAL', 24), ('P_LOCAL', 48)]:
        mode, count = config
        with chains.Workers(count) as pool:
            before = numba.get_num_threads()
            output = call(chain, config, data, pool, refs, fft)
            gate = exact.compare(expected, output, root/f'{identity}_{chain}_{mode}{count}.jsonl')
            old_identity = exact.fingerprints(output)
            another = call(chain, config, data, pool, refs, fft)
            require(exact.fingerprints(output) == old_identity, 'OLD_OUTPUT_LIFETIME')
            require(all(not np.shares_memory(a, b) for _, a in exact.arrays(output)
                        for _, b in exact.arrays(another)), 'FRESH_OUTPUT_NO_ALIAS')
            require(numba.get_num_threads() == before, 'MASK_RESTORED')
            require(exact.fingerprints({'E':data[0], 'I':data[1], 'response':data[2], 'Omega':data[3]['Omega']}) == fingerprint,
                    'INPUT_IMMUTABILITY')
            activity = pool.take_records()
            require(all(r['actual_workers'] == count for r in activity), 'ACTUAL_TARGET_WORKERS')
            append(root/'QUALIFICATION.jsonl', dict(input=identity, chain=chain, mode=mode, workers=count,
                   gate=gate, input_unchanged=True, old_output_unchanged=True, fresh_no_alias=True,
                   output_identity=old_identity, mask_before=before, mask_after=numba.get_num_threads(),
                   pool_initialization_ns=pool.initialization_ns, target_activity=activity))
            del output, another
    require(exact.fingerprints(expected) == output_hash, 'REFERENCE_LIFETIME')
    return output_hash


def measure_block(chain, identity, data, refs, fft, exact, root, configs, block, factor, output_identity, expected_mask=24):
    results = []
    for mode, count in configs:
        pool = chains.Workers(count) if count else None
        initialization = pool.initialization_ns if pool else 0
        try:
            for repeat in range(-2, 5):
                before = observation()
                cpu, wall = time.process_time_ns(), time.perf_counter_ns()
                output = call(chain, (mode,count), data, pool, refs, fft)
                wall_s, cpu_s = (time.perf_counter_ns()-wall)/1e9, (time.process_time_ns()-cpu)/1e9
                after = observation()
                fingerprint = exact.fingerprints(output)
                require(fingerprint == output_identity, 'MEASURED_OUTPUT_IDENTITY')
                target = pool.take_records() if pool else []
                require(all(r['actual_workers'] == count for r in target), 'MEASURED_WORKERS')
                require(before['numba_mask'] == after['numba_mask'] == expected_mask, 'MEASURED_MASK')
                row = dict(chain=chain, input=identity, mode=mode, workers=count, block=block, factor=factor,
                           repeat=repeat, warmup=repeat<0, wall_s=wall_s, process_cpu_s=cpu_s,
                           cpu_equivalent=cpu_s/wall_s, clock='COMPLETE_ISOLATED_CALL_WALL',
                           pool_initialization_ns=initialization, observation_before=before, observation_after=after,
                           threads_delta=thread_delta(before['threads'],after['threads']), target_activity=target,
                           output_identity=fingerprint, version=chains.VERSION)
                append(root/'TIMINGS.jsonl', row)
                if repeat>=0: results.append(row)
                del output
        finally:
            if pool: pool.close()
        print(json.dumps(dict(stage='MEASURED',input=identity,chain=chain,mode=mode,workers=count,block=block,factor=factor)),flush=True)
    return results


def allow_local48(rows, chain, identity):
    for block in (0,1):
        def med(count):
            return statistics.median(r['wall_s'] for r in rows if r['chain']==chain and r['input']==identity
                                     and r['mode']=='P' and r['workers']==count and r['block']==block)
        if med(48)/med(24) > 0.97: return False
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();root=args.report.resolve();plan=json.loads((root/'PLAN.json').read_text())
    require(os.environ.get('SLURM_JOB_ID','').isdigit(), 'COMPUTE_NODE_ONLY')
    require(os.environ.get('SLURM_CPUS_PER_TASK')=='96', 'CPU96')
    exact=support_module('g1_exact')
    stage=Path(plan['stage_root']);canonical=Path(plan['canonical'])
    for name,digest in plan['stage_metadata_sha256'].items():
        require(pv.raw_sha256_file(stage/name)==digest, 'FROZEN_METADATA:'+name)
    require(pv.raw_sha256_file(canonical/'INPUT_MANIFEST.json')==plan['canonical_manifest_sha256'], 'CANONICAL_BINDING')
    from tools.cpu_s4_6a.inputs import context
    ctx=context(canonical);kwargs=stage_kwargs(ctx)
    # Source/PRE are validated by the unchanged canonical loader, then released.
    del ctx['source'],ctx['pre_fields'],ctx['pre0'];gc.collect()
    tree=json.loads((stage/'TREE.json').read_text());frozen=json.loads((stage/'FINGERPRINTS.json').read_text())
    built,state,fft=fresh_factory(root/'UNUSED_CAPTURE')
    refs=chains.source_references(fft)
    write(root/'FACTORY_BINDING.json',dict(factory=built[3],reference_source=refs[2],
                                         kwargs={k:v for k,v in kwargs.items() if k!='Omega'}))
    datasets=[];qroot=root/'exact';qroot.mkdir()
    with zipfile.ZipFile(stage/'ARRAYS.zip') as archive:
        for index, side in enumerate(('pre','post')):
            record=tree['items'][index]['items']
            for field_key,stage_key in (('field','stage1'),('predictor','stage2')):
                identity=side+'_'+field_key
                field=decode(record[field_key],archive,exact)
                field_identity=check_frozen(field,f'bundle.{index}.{field_key}',frozen,exact)
                expected=decode(record[stage_key],archive,exact)
                check_frozen(expected,f'bundle.{index}.{stage_key}',frozen,exact)
                started=time.perf_counter()
                actual=state.stage(field,**kwargs)
                reference_s=time.perf_counter()-started
                gate=exact.compare(expected,dict(actual),qroot/(identity+'_FRESH_STAGE.jsonl'))
                require(exact.fingerprints(field)==field_identity,'FRESH_STAGE_INPUT_UNCHANGED')
                require(state.stats['reference_fallback_calls']==0,'NO_REFERENCE_FALLBACK')
                intensity=refs[0](field,kwargs['n0'])
                energy=refs[1](intensity,actual['I_R'],kwargs['Omega'],kwargs['dt'],kwargs['n_R'])
                exact.compare({k:actual[k] for k in energy},energy,qroot/(identity+'_EXTRACTED_REF.jsonl'))
                datasets.append((identity,(field,intensity,actual['I_R'],kwargs)))
                append(root/'REAL_INPUT_ADMISSION.jsonl',dict(input=identity,field_identity=field_identity,
                      frozen_prefix=f'bundle.{index}.{field_key}',stage_gate=gate,first_reference_s=reference_s,
                      input_archive=str(stage/'ARRAYS.zip'),fresh_response=True,retired_capability_reused=False))
                del expected,actual,energy
                print(json.dumps(dict(stage='REAL_INPUT_EXACT',input=identity)),flush=True)
    output_ids={}
    for identity,data in datasets:
        for chain in ('A','B'):
            output_ids[identity,chain]=qualify(chain,identity,data,refs,fft,exact,qroot)
    write(root/'ALL_REAL_EXACT.json',dict(status='PASS',inputs=4,chains=['A','B'],configs=6,C='UNAVAILABLE'))
    print('ALL_REAL_EXACT_PASS',flush=True)
    initial={identity:exact.fingerprints(dict(E=data[0],I=data[1],response=data[2],Omega=data[3]['Omega'])) for identity,data in datasets}
    rows=[]
    for block in (0,1):
        for identity,data in (datasets if block==0 else list(reversed(datasets))):
            for chain in (('A','B') if block==0 else ('B','A')):
                configs=configurations();configs=configs if block==0 else list(reversed(configs))
                rows+=measure_block(chain,identity,data,refs,fft,exact,root,configs,block,'THREAD',output_ids[identity,chain])
    eligibility={(identity,chain):allow_local48(rows,chain,identity) for identity,_ in datasets for chain in ('A','B')}
    write(root/'LOCAL48_ADMISSION.json',dict(rule='P48/P24 <= 0.97 in each reverse-order block, separately for each real input and chain',
          records=[dict(input=i,chain=c,allowed=v) for (i,c),v in eligibility.items()]))
    for block in (0,1):
        for identity,data in (datasets if block==0 else list(reversed(datasets))):
            for chain in (('A','B') if block==0 else ('B','A')):
                counts=[24,48] if eligibility[identity,chain] else [24]
                for count in counts:
                    configs=[('P',count),('P_LOCAL',count)]
                    if block: configs.reverse()
                    measure_block(chain,identity,data,refs,fft,exact,root,configs,block,'MATERIALIZATION',output_ids[identity,chain])
    for identity,data in datasets:
        require(exact.fingerprints(dict(E=data[0],I=data[1],response=data[2],Omega=data[3]['Omega']))==initial[identity], 'FINAL_INPUT_IMMUTABILITY')
    write(root/'MECHANISM_COMPLETED.json',dict(status='COMPLETE_REAL_ISOLATED_REPLAY',version=chains.VERSION,
          inputs=4,chains=['A','B'],input_unchanged=True,full_window_qualification=False,
          default='I0',production_qualified=False,review_gate='MANUAL_REVIEW_PENDING'))


if __name__=='__main__':
    main()

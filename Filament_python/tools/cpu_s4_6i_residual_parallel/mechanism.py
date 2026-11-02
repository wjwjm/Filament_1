"""Fixed two-order REF/P1/P24 mechanism on all twelve captured real inputs."""
from pathlib import Path
import statistics
import time
import numpy as np
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from .capture import fingerprint, load_capture
from .power_volume import make_power_volume_pool, power_volume_cleanup_p24
from .power_hook import PowerVolumeHook

ORDERS=(('REF','P1','P24'),('P24','P1','REF'))

def _value_identity(value):
    fp=fingerprint(value)
    return {key:fp[key] for key in ('shape','dtype','strides','c_contiguous','raw_sha256')}

def run(root,output,*,shape,execution_commit,local_engineering=False,expected_binding=None):
    root=Path(root).resolve(strict=True)
    output=Path(output).resolve()
    if output.exists():
        raise RuntimeError('S46I_MECHANISM_NON_OVERWRITE')
    package=load_capture(root,shape=shape)
    if package['source_binding']['execution_commit']!=execution_commit:
        raise RuntimeError('S46I_MECHANISM_EXECUTION_IDENTITY')
    if expected_binding is not None:
        for key in ('execution_commit','source_manifest_raw_sha256','input_contract_raw_sha256'):
            if package['source_binding'].get(key)!=expected_binding.get(key):
                raise RuntimeError('S46I_MECHANISM_INPUT_SOURCE_BINDING:'+key)
    elif not local_engineering:
        raise RuntimeError('S46I_MECHANISM_BOUND_INPUT_REQUIRED')
    pools={}
    hooks={}
    rows=[]
    blocks=[]
    failure=None
    primary_error=None
    started=time.perf_counter()
    try:
        for config,count in (('P1',1),('P24',1 if local_engineering else 24)):
            pools[config]=make_power_volume_pool(count)
            hooks[config]=PowerVolumeHook(pools[config])
        reference=hooks['P1']._original_functions['deposition._power_volume']
        for group,order in enumerate(ORDERS,1):
            for record in package['records']:
                # File mappings are reader storage only; restore an owned plain
                # ndarray for every timed call, with the same policy for REF/P1/P24.
                source=np.asarray(np.load(root/record['input_file']['path'],allow_pickle=False,mmap_mode='r'))
                expected=np.asarray(np.load(root/record['output_file']['path'],allow_pickle=False,mmap_mode='r'))
                original_identity=_value_identity(source)
                expected_identity=_value_identity(expected)
                for name,identity in (('input',original_identity),('output',expected_identity)):
                    if identity!={key:record[name+'_fingerprint'][key] for key in identity}:
                        raise RuntimeError('S46I_MECHANISM_CAPTURE_ARRAY_IDENTITY')
                times={}
                for config in order:
                    for hook in hooks.values():
                        hook.verify_binding()
                    samples=[]
                    previous=None
                    previous_identity=None
                    for index in range(7):
                        restored=np.array(source,copy=True,order='C')
                        if _value_identity(restored)!=original_identity:
                            raise RuntimeError('S46I_MECHANISM_RESTORE')
                        tick=time.perf_counter_ns()
                        if config=='REF':
                            actual=reference(restored)
                        else:
                            actual=power_volume_cleanup_p24(restored,pool=pools[config],
                                context=record['context'],interval_index=record['interval_index'])
                        elapsed=(time.perf_counter_ns()-tick)/1e9
                        activity=[] if config=='REF' else pools[config].take_records()
                        exact=(np.array_equal(actual,expected,equal_nan=True)
                            and _value_identity(restored)==original_identity
                            and actual.shape==expected.shape and actual.dtype==expected.dtype
                            and actual.strides==expected.strides and actual.flags.c_contiguous
                            and actual.flags.owndata and not np.shares_memory(actual,restored)
                            and (previous is None or (_value_identity(previous)==previous_identity
                                 and not np.shares_memory(actual,previous))))
                        if not exact:
                            path=output.parent/'NONEXACT_OUTPUT.npy'
                            with path.open('xb') as handle:
                                np.save(handle,actual,allow_pickle=False)
                            failure=dict(group=group,key=record['key'],config=config,index=index,
                                actual_file=str(path),raw_sha256=pv.raw_sha256_file(path),activity=activity)
                            raise RuntimeError('SCIENTIFIC_NONEXACT:S46I_POWER_MECHANISM')
                        if config!='REF' and (len(activity)!=2 or any(
                                row['requested_workers']!=pools[config].count for row in activity)):
                            raise RuntimeError('S46I_MECHANISM_ACTIVITY')
                        rows.append(dict(group=group,key=record['key'],context=record['context'],
                            interval_index=record['interval_index'],config=config,index=index,
                            warmup=index<2,wall_s=elapsed,exact=True,activity=activity))
                        if index>=2:
                            samples.append(elapsed)
                        previous=actual
                        previous_identity=_value_identity(actual)
                        del restored,actual
                    times[config]=statistics.median(samples)
                    del previous
                gain=1-times['P24']/times['REF']
                blocks.append(dict(group=group,key=record['key'],context=record['context'],
                    interval_index=record['interval_index'],medians_s=times,p24_gain=gain,
                    gain_pass=gain>=0.05))
                del source,expected
    except BaseException as exc:
        primary_error=exc
        try:
            sg.write(output.parent/'MECHANISM_PARTIAL.json',dict(rows=rows,blocks=blocks,
                failure=failure,error=repr(exc),execution_commit=execution_commit))
        except BaseException as recording:
            exc.add_note('S46I_PARTIAL_RECORD_FAILURE:'+repr(recording))
        raise
    finally:
        close_errors=[]
        terminal_drain={}
        snapshots={}
        for name,pool in pools.items():
            try:
                pool.close()
            except BaseException as exc:
                close_errors.append(dict(config=name,error=repr(exc)))
            try:
                terminal_drain[name]=pool.take_records()
                snapshots[name]=pool.snapshot()
                if snapshots[name].get('close_error') is not None:
                    close_errors.append(dict(config=name,error='recorded_close_error:'+str(snapshots[name]['close_error'])))
            except BaseException as exc:
                close_errors.append(dict(config=name,error='drain_snapshot:'+repr(exc)))
        failed=(bool(close_errors) or len(snapshots)!=len(pools)
                or any(terminal_drain.values()) or any(not p['closed'] or p['active_thread_ids']
                or p['pending_scalar_records'] or p.get('dropped_records',0)
                or p.get('invalid_records_dropped',0) or p.get('close_error') is not None
                for p in snapshots.values()))
        try:
            sg.write(output.parent/'MECHANISM_POOLS.json',dict(pools=snapshots,errors=close_errors,
                terminal_drain=terminal_drain,retirement_failed=failed,
                primary_error=None if primary_error is None else repr(primary_error)))
        except BaseException as exc:
            failed=True
            close_errors.append(dict(error='retirement_record:'+repr(exc)))
        if failed:
            cleanup=RuntimeError('S46I_MECHANISM_POOL_RETIREMENT:'+repr(close_errors))
            if primary_error is None:
                raise cleanup
            primary_error.add_note(str(cleanup))
    warmups=sum(row['warmup'] for row in rows)
    if warmups!=144 or len(rows)-warmups!=360 or len(blocks)!=24:
        raise RuntimeError('S46I_MECHANISM_CALL_COUNTS')
    gain_pass=all(row['gain_pass'] for row in blocks)
    status=('LOCAL_ENGINEERING_ONLY' if local_engineering else
            'PASS_SELECTED_MECHANISM' if gain_pass else 'VALID_NEGATIVE_LOCAL_GAIN')
    result=dict(schema='filament.cpu_s4_6i.selected_mechanism.v1',status=status,
        execution_commit=execution_commit,shape=list(shape),local_engineering=local_engineering,
        capture_manifest_raw_sha256=pv.raw_sha256_file(root/'CAPTURE.json'),
        source_binding=package['source_binding'],rows=rows,blocks=blocks,all_exact=True,
        warmup_calls=warmups,formal_calls=len(rows)-warmups,inputs=12,
        gain_pass=gain_pass,formal14_allowed=gain_pass and not local_engineering,
        service_s=time.perf_counter()-started,default='I0',production_qualified=False)
    sg.write(output,result)
    return result

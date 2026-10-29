"""One independent, plan-only resource check for the frozen S46A GPU shapes."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
from .inputs import require,write

SMALL=(384,117,129)
FULL=(384,351,301)
GIB=1024**3


def full_admission(probe,small_reserved_bytes,free_bytes,total_bytes):
    """Apply the preregistered forecast; this is not a strict peak proof."""
    values=(small_reserved_bytes,free_bytes,total_bytes)
    require(all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in values),'MEMORY_FORECAST_INTEGER_BYTES')
    require(0<free_bytes<=total_bytes and small_reserved_bytes>0,'MEMORY_FORECAST_POSITIVE_CAPACITY')
    if probe.get('status')!='PASS_FFT_PLAN_RESOURCE_PROBE':
        return dict(status='FULLSIZE_RESOURCE_BLOCKED',reason='FFT_PLAN_RESOURCE_PROBE_NOT_PASSED')
    full=next((v for v in probe['grids'] if v['shape']==list(FULL)),None)
    require(full is not None and int(probe['device_total_bytes'])==total_bytes,'ACTUAL_FULL_PROBE_SAME_DEVICE_CAPACITY')
    field_bytes=FULL[0]*FULL[1]*FULL[2]*16
    payload=max(32*field_bytes,7*small_reserved_bytes)
    workspace=int(full['plan_workspace_bytes'])
    require(workspace>=0,'NONNEGATIVE_FFT_WORKSPACE')
    margin=2*GIB
    required=payload+workspace+margin
    probe_free=int(full['free_after_plans_bytes'])
    passed=probe_free>=payload+margin and free_bytes>=required
    return dict(status='PASS_FULLSIZE_MEMORY_FORECAST' if passed else 'FULLSIZE_RESOURCE_BLOCKED',
        reason='SOURCE_AND_MEASURED_PLAN_FORECAST' if passed else 'AVAILABLE_MEMORY_BELOW_SOURCE_PLUS_PLAN_FORECAST',
        full_shape=list(FULL),precision='complex128',full_field_bytes=field_bytes,
        source_field_equivalents=32,source_payload_bytes=32*field_bytes,
        observed_small_reserved_bytes=small_reserved_bytes,volume_ratio=7,
        scaled_small_reserved_bytes=7*small_reserved_bytes,payload_forecast_bytes=payload,
        measured_plan_workspace_bytes=workspace,margin_bytes=margin,required_bytes=required,
        probe_free_after_full_plans_bytes=probe_free,current_free_bytes=free_bytes,total_bytes=total_bytes,
        forecast_kind='Conservative engineering forecast, not a proved runtime peak bound',
        previous_32GiB_preregistration='Preserved in earlier versions; this user-approved model was frozen before new full-size samples',
        on_OOM='Stop without precision, physics, output or algorithm fallback')


def run(report):
    import cupy as cp
    from cupy.cuda import cufft
    import cupyx.scipy.fftpack as fftpack
    from .records import load_json_strict
    require(os.environ.get('SLURM_JOB_ID','').isdigit() and cp.cuda.runtime.getDeviceCount()==1,'REAL_SINGLE_GPU_PLAN_PROBE_ONLY')
    require(not cp.fft.config.use_multi_gpus,'NO_MULTI_GPU_PLAN_PROBE')
    plan=load_json_strict(report/'MEASUREMENT_PLAN.json')
    canonical=Path(plan.get('canonical_root',str(report/'canonical')))
    rows=[]
    total=None
    for shape in (SMALL,FULL):
        config=load_json_strict(canonical/('x'.join(map(str,shape)))/'config.json')
        require(config['propagation']['linear_model']=='bk_nee','FROZEN_BK_NEE_PER_SLICE_SPATIAL_FFT')
        before,total=map(int,cp.cuda.runtime.memGetInfo())
        field_bytes=shape[0]*shape[1]*shape[2]*16
        require(before>field_bytes+2*GIB,'PLAN_PROBE_INPUT_CAPACITY')
        field=cp.empty(shape,dtype=cp.complex128)
        # BK-NEE uses one full temporal transform and 384 separate 2D slices.
        # The plan-only probe matches that organization; it never calls fft().
        temporal=fftpack.get_fft_plan(field,axes=(0,),value_type='C2C')
        spatial=fftpack.get_fft_plan(field[0],axes=(0,1),value_type='C2C')
        require(temporal.nx==shape[0] and temporal.batch==shape[1]*shape[2],'TEMPORAL_PLAN_GEOMETRY')
        require(int(temporal.fft_type)==int(cufft.CUFFT_Z2Z),'COMPLEX128_TEMPORAL_PLAN')
        def workspace(p):
            area=p.work_area
            return 0 if area is None else int(area.mem.size)
        temporal_work=workspace(temporal);spatial_work=workspace(spatial)
        cp.cuda.runtime.deviceSynchronize()
        after,after_total=map(int,cp.cuda.runtime.memGetInfo())
        require(total==after_total,'PLAN_PROBE_DEVICE_CAPACITY_STABLE')
        rows.append(dict(shape=list(shape),field_bytes=field_bytes,input_initialized=False,
            temporal_plan_type=type(temporal).__name__,temporal_nx=int(temporal.nx),
            temporal_batch=int(temporal.batch),temporal_workspace_bytes=temporal_work,
            spatial_plan_type=type(spatial).__name__,spatial_input_shape=list(field[0].shape),
            spatial_batch=1,spatial_workspace_bytes=spatial_work,
            plan_workspace_bytes=temporal_work+spatial_work,
            free_before_input_bytes=before,free_after_plans_bytes=after,
            pool_reserved_bytes=int(cp.get_default_memory_pool().total_bytes()),
            transforms_executed=0,propagations_executed=0))
        del temporal,spatial,field
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
    return dict(status='PASS_FFT_PLAN_RESOURCE_PROBE',job_id=os.environ['SLURM_JOB_ID'],
        process_id=os.getpid(),device_total_bytes=total,grids=rows,independent_process=True,
        formal_plan_reuse=False,transforms_executed=0,propagations_executed=0,
        CUPY_CACHE_DIR=os.environ.get('CUPY_CACHE_DIR'),CUDA_CACHE_PATH=os.environ.get('CUDA_CACHE_PATH'),
        policy='Fresh resource-only context; formal windows run in other new processes')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();report=args.report.resolve(strict=True)
    try:value=run(report)
    except BaseException as error:
        value=dict(status='FFT_PLAN_RESOURCE_PROBE_BLOCKED',error=repr(error),
            job_id=os.environ.get('SLURM_JOB_ID'),process_id=os.getpid(),
            transforms_executed=0,propagations_executed=0)
        write(report/'GPU_MEMORY_RESOURCE_PROBE.json',value)
        print(json.dumps(value),flush=True)
        return 1
    write(report/'GPU_MEMORY_RESOURCE_PROBE.json',value);print(json.dumps(value),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

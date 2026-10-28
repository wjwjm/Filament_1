"""Seal each grid once on CPU; both backends load the same ordinary files."""
from __future__ import annotations

import json
import os
from pathlib import Path

from tools.hpc_ops import provenance_v2 as pv
from .records import load_json_strict, validate_source_records

SHAPES = ((32, 17, 19), (384, 17, 19), (384, 117, 129), (384, 351, 301))
FULL = SHAPES[-1]


def array_filename(name):
    # Windows paths are case insensitive: axes_kx and axes_KX must differ
    # by more than case. Keep the logical names unchanged in the manifest.
    return ('axes_grid_' + name[-2:].lower() if name in ('axes_KX', 'axes_KY') else name) + '.npy'


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def seal(destination, data_root, shape, *, full_input_root=None):
    """No propagation or CUDA work. Full data are read only on a compute node."""
    import numpy as np
    from KHz_filament.device import USE_GPU
    from KHz_filament.confio import load_all
    from KHz_filament.grids import make_axes
    from KHz_filament.hr4e5_formal_entry import pre0_fields_from_delta_n
    from KHz_filament.hr4e5_evidence import sha256_array
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.hr4e5_formal_entry import build_prefix_schedule

    shape = tuple(shape)
    require(shape in SHAPES and not USE_GPU, 'DECLARED_CPU_SEAL_ONLY')
    target = Path(destination).resolve()
    require(not target.exists(), 'NON_OVERWRITING_CANONICAL_BUNDLE')
    data = Path(data_root).resolve(strict=True)
    if shape == FULL:
        require(os.environ.get('SLURM_JOB_ID', '').isdigit(), 'FULL_INPUT_COMPUTE_NODE_ONLY')
        source_root = Path(full_input_root).resolve(strict=True)
        expected = {'E_source_canonical.npy': '3fcf21546fe61b0c9e3522816f7e03e004321b0ec91ddff1ab70b5c8a25309c7',
                    'pre0_first_8.npy': 'b38a7a0efe9e43a10ae1520f063380828555228e4bf3efdacdda2434f21721c5',
                    'derived_n5_execution_config.json':'429f822af9d3845edfc6f098765be0f2d8b10a9fd643225c1af8afa23c59c87c'}
        for name, digest in expected.items():
            require(pv.raw_sha256_file(source_root / name) == digest, 'ORIGINAL_FULL_INPUT_RAW:' + name)
        config_raw = (source_root / 'derived_n5_execution_config.json').read_bytes()
        source = np.load(source_root / 'E_source_canonical.npy', allow_pickle=False)
        pre = pre0_fields_from_delta_n(np.load(source_root / 'pre0_first_8.npy', allow_pickle=False), expected_count=8)
        grid, beam, prop, ion, heat, run, raman = load_all(str(source_root / 'derived_n5_execution_config.json'))
        origin = {'kind': 'ACCEPTED_L1_ORIGINAL_SOURCE_PRE', 'root': str(source_root), 'raw_sha256': expected}
    else:
        from tools.cpu_s4_3g3_smallgrid.scientific_context import context
        ctx = context(data, shape=shape)
        source, pre = ctx['source'], ctx['pre_fields']
        grid, beam, prop, ion, heat, run, raman = (ctx[name] for name in ('grid', 'beam', 'prop', 'ion', 'heat', 'run', 'raman'))
        config = load_json_strict(data / 'SMALLGRID_CONFIG.json')
        config['grid'].update(Nt=shape[0], Ny=shape[1], Nx=shape[2])
        config_raw = (json.dumps(config, indent=2, allow_nan=False) + '\n').encode('utf-8')
        source_root = data / 'inputs'
        origin = {'kind': 'ORIGINAL_SMALLGRID_GENERATOR_ONCE', 'generator': 'tools.cpu_s4_3g3_smallgrid.scientific_context.context'}
    require(source.shape == shape and source.dtype == np.complex128, 'SOURCE_SHAPE_PRECISION')
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    target.mkdir(parents=True)
    (target / 'config.json').write_bytes(config_raw)
    arrays = dict(E_source=source, **{'PRE_' + name: value for name, value in pre.items()})
    arrays.update({'axes_' + name: getattr(axes, name) for name in ('x', 'y', 't', 'kx', 'ky', 'KX', 'KY', 'kperp2', 'Omega')})
    array_meta = {}
    for name, value in arrays.items():
        with (target / array_filename(name)).open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        array_meta[name] = dict(file=array_filename(name), shape=list(value.shape), dtype=str(value.dtype), canonical_array_sha256=sha256_array(value))
    lut_source = source_root / 'cache/rate_tables'
    import shutil
    (target / 'cache/rate_tables').mkdir(parents=True)
    for name in ('ionlut_N2_76585af6514cdd04.npz', 'ionlut_O2_1446bb99e8f9fd80.npz'):
        expected_lut={'ionlut_N2_76585af6514cdd04.npz':'1f70691ab004c410123f5e06a0dbfa626beecf4059c09b715a8052feeb4e128e',
                      'ionlut_O2_1446bb99e8f9fd80.npz':'57617bc72acbb794ae2c939241beaa57fa313eaa27a3d2981882f1fbe68d6cf8'}
        require(pv.raw_sha256_file(lut_source/name)==expected_lut[name],'ACCEPTED_LUT_RAW:'+name)
        shutil.copyfile(lut_source / name, target / 'cache/rate_tables' / name)
    records = [dict(path=p.relative_to(target).as_posix(), classification='external', hash_scope='raw_bytes',
                    raw_sha256=pv.raw_sha256_file(p), size_bytes=p.stat().st_size)
               for p in sorted(target.rglob('*')) if p.is_file()]
    full_schedule = build_longitudinal_schedule(dz=float(prop.dz),z_max=float(prop.z_max),z_start=0.,
        focus_window_step=bool(prop.focus_window_step),focus_center_m=float(prop.focus_center_m),
        focus_halfwidth_m=float(prop.focus_halfwidth_m),dz_focus=float(prop.dz_focus))
    value = dict(schema='filament.cpu_s46a.canonical_inputs.v1', hash_scope='raw_bytes', shape=list(shape),
                 precision='complex128', origin=origin, arrays=array_meta, records=records,
                 schedule=json.loads(json.dumps(build_prefix_schedule(full_schedule,8).as_metadata())),
                 spacing=dict(dx=axes.dx, dy=axes.dy, dt=axes.dt), source_readonly=True,
                 generated_once=True, independent_gpu_regeneration=False)
    write(target / 'INPUT_MANIFEST.json', value)
    return value


def context(bundle, *, verify_raw=True):
    import numpy as np
    from KHz_filament.device import xp, USE_GPU
    from KHz_filament.confio import load_all
    from KHz_filament.grids import Axes
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.hr4e5_formal_entry import build_prefix_schedule
    root = Path(bundle).resolve(strict=True)
    manifest = load_json_strict(root / 'INPUT_MANIFEST.json')
    require(manifest['schema']=='filament.cpu_s46a.canonical_inputs.v1' and manifest['hash_scope']=='raw_bytes'
            and manifest['precision']=='complex128' and tuple(manifest['shape']) in SHAPES, 'CANONICAL_MANIFEST_DOMAIN')
    if os.environ.get('SLURM_JOB_ID'):
        require(root.name=='x'.join(map(str,manifest['shape'])),'FORMAL_CANONICAL_SHAPE_NAMESPACE')
    if verify_raw:
        validate_source_records(root, manifest['records'])
    grid, beam, prop, ion, heat, run, raman = load_all(str(root / 'config.json'))
    require((grid.Nt, grid.Ny, grid.Nx) == tuple(manifest['shape']), 'LOADED_GRID')
    require(raman.operator_mode == 'full_isaacs_eq27' and raman.operator_integrator == 'heun'
            and raman.iir_sampling == 'exact_piecewise_linear' and getattr(raman, 'method', 'iir') == 'iir', 'FROZEN_RAMAN')
    source = np.load(root / 'E_source.npy', allow_pickle=False)
    require(source.shape==tuple(manifest['shape']) and source.dtype==np.complex128 and source.flags.c_contiguous, 'LOADED_CANONICAL_SOURCE')
    source.setflags(write=False)
    pre = {name: np.load(root / ('PRE_' + name + '.npy'), allow_pickle=False) for name in ('delta_n', 'vx', 'vy')}
    for array in pre.values():
        array.setflags(write=False)
    axis_names = ('x', 'y', 't', 'kx', 'ky', 'KX', 'KY', 'kperp2', 'Omega')
    host_axes = {name: np.load(root / array_filename('axes_' + name), allow_pickle=False) for name in axis_names}
    loaded=dict(E_source=source,**{'PRE_'+name:value for name,value in pre.items()},**{'axes_'+name:value for name,value in host_axes.items()})
    require(set(loaded)==set(manifest['arrays']), 'COMPLETE_CANONICAL_ARRAY_SET')
    for name,array in loaded.items():
        meta=manifest['arrays'][name]
        require(list(array.shape)==meta['shape'] and str(array.dtype)==meta['dtype'] and meta['file']==array_filename(name), 'ARRAY_METADATA:'+name)
    require(all(value.dtype==np.float64 and value.shape==(8,grid.Ny,grid.Nx) for value in pre.values()), 'PRE_SHAPE_PRECISION')
    require(manifest['spacing']==dict(dx=grid.Lx/grid.Nx,dy=grid.Ly/grid.Ny,dt=grid.Twin/grid.Nt), 'COMMON_SPACING')
    axes = Axes(**{name: xp.asarray(value) for name, value in host_axes.items()}, **manifest['spacing'])
    # Both original loaders enable these existing diagnostic clocks. Their
    # built-in synchronization remains part of each backend's natural work.
    prop.measure_performance = True
    prop.diag_bk_nee_profile = True
    ion.rate_table = dict(ion.rate_table)
    ion.rate_table['cache_dir'] = str(root / 'cache/rate_tables')
    require(not ion.rate_table.get('force_rebuild'), 'NO_LUT_REBUILD')
    full = build_longitudinal_schedule(dz=float(prop.dz), z_max=float(prop.z_max), z_start=0.,
        focus_window_step=bool(prop.focus_window_step), focus_center_m=float(prop.focus_center_m),
        focus_halfwidth_m=float(prop.focus_halfwidth_m), dz_focus=float(prop.dz_focus))
    schedule=build_prefix_schedule(full,8)
    require(json.loads(json.dumps(schedule.as_metadata()))==manifest['schedule'], 'COMMON_FROZEN_SCHEDULE')
    return dict(source=source, pre_fields=pre, pre0=pre['delta_n'], axes=axes, host_axes=host_axes,
                grid=grid, beam=beam, prop=prop, ion=ion, heat=heat, run=run, raman=raman,
                schedule=schedule, bundle=str(root), input_manifest=manifest,
                input_identity=dict(manifest_raw_sha256=pv.raw_sha256_file(root/'INPUT_MANIFEST.json'),hash_scope='raw_bytes',records=manifest['records'],schedule=manifest['schedule']),
                benchmark_input_kind='ONE_CANONICAL_RAW_BUNDLE_BOTH_BACKENDS', gpu=USE_GPU)

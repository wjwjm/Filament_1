"""Fixed local scientific fixtures using the original formal PRE/POST classes."""
from pathlib import Path
import numpy as np
from g1_common import require


def local_context(*, medium=False, energy=False, profile=False):
    from tools.cpu_p1r2.cpu_p1r2_j1 import _tiny_context
    from KHz_filament.grids import make_axes
    from KHz_filament.runner import build_transverse_input_field
    ctx = _tiny_context()
    ctx['prop'].diag_linear_halfstep_energy = bool(energy)
    ctx['prop'].diag_bk_nee_profile = bool(profile)
    if medium:
        grid = ctx['grid']
        grid.Nt, grid.Ny, grid.Nx = 384, 41, 39
        axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
        source, _ = build_transverse_input_field(axes, ctx['beam'], np.complex128)
        ctx['source'] = np.ascontiguousarray(source, dtype=np.complex128)
        ctx['pre0'] = np.zeros((8, grid.Ny, grid.Nx), dtype=np.float64)
        from KHz_filament.hr4e5_formal_entry import pre0_fields_from_delta_n
        ctx['pre_fields'] = pre0_fields_from_delta_n(ctx['pre0'], expected_count=8)
    return ctx


def prepare(ctx, directory, *, intervals):
    from KHz_filament.constants import N0_air, Ui_N2, c0, n2_air
    from KHz_filament.grids import make_axes
    from KHz_filament.longitudinal import build_deposition_contract
    from KHz_filament.hr4e5_formal_entry import (build_prefix_schedule, _all_interval_sample_plan,
        StreamingPulseReadView, StreamingPulseHook)
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle
    from KHz_filament.thermalization import ThermalDiagnosticSink
    from KHz_filament.slow_state import HR3BDiagnosticSink, validate_hr3b_parameters
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    require(intervals in (1, 2, 8), 'LOCAL_ONLY_INTERVAL_SET')
    schedule = build_prefix_schedule(ctx['schedule'], intervals, block_size=1 if intervals < 8 else 8)
    require(tuple(schedule.z_edges) == tuple(ctx['schedule'].z_edges[:intervals + 1]), 'LOCAL_CONTIGUOUS_PREFIX')
    grid, beam, prop, heat = (ctx[key] for key in ('grid', 'beam', 'prop', 'heat'))
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    field = np.array(ctx['source'], dtype=np.complex128, order='C', copy=True)
    current = {key: np.array(value[:intervals], dtype=np.float64, order='C', copy=True) for key,value in ctx['pre_fields'].items()}
    records = [dict(ordinal=i, screen_id=f'local_{i:06d}', z_m=float(schedule.z_edges[i+1])) for i in range(intervals)]
    lifecycle = StreamingLifecycle.create(root=directory/'state', current=current, screen_records=records,
        current_generation='G2_LOCAL_PRE0', dx_m=axes.dx, dy_m=axes.dy, queue_depth=8)
    view = StreamingPulseReadView(lifecycle)
    hook = StreamingPulseHook(lifecycle, final=True, view=view)
    plan = _all_interval_sample_plan(schedule)
    thermal = ThermalDiagnosticSink(plan=plan, output_path=str(directory/'sinks'), shape=(grid.Ny,grid.Nx), dtype=np.float64, enabled=True, mode='validation')
    hr3b = HR3BDiagnosticSink(plan=plan, output_path=str(directory/'sinks'), shape=(grid.Ny,grid.Nx), dtype=np.float64, enabled=True)
    beta = validate_hr3b_parameters(rho0=float(heat.rho0), Cv=float(heat.Cv), T0=float(prop.air_T), n0=float(beam.n0))
    omega0 = 2*np.pi*c0/float(beam.lam0)
    kwargs = dict(kperp2=axes.kperp2, k0=float(beam.n0)*float(omega0)/c0, omega0=float(omega0),
        dz=float(schedule.dz_intervals[0]), z_max=float(schedule.z_end-schedule.z_start), n0=float(beam.n0),
        n2=float(getattr(prop,'n2',getattr(beam,'n2_air',n2_air))), Ui=Ui_N2, N0=N0_air,
        ion_conf=ctx['ion'],dn_gas=None,dt=axes.dt,axes=axes,prop_conf=prop,raman_conf=ctx['raman'],
        record_onaxis_rho_time=True,record_every_z=1,longitudinal_schedule=schedule,
        deposition_contract=build_deposition_contract(schedule,axes=axes),thermal_sink=thermal,thermal_slow_state=view,
        hr3b_parameters=dict(rho0=float(heat.rho0),Cv=float(heat.Cv),T0=float(prop.air_T),n0=float(beam.n0),beta_th=float(beta)),
        hr3b_sink=hr3b,post_commit_hook=hook)
    return field, kwargs, view


def finish(result, view, directory, *, intervals):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle
    from g1_exact import normalized_return
    directory = Path(directory)
    require(view.complete, 'FORMAL_PRE_READ_UPDATE_COMPLETE')
    lifecycle = StreamingLifecycle.open(view.root)
    require(len(lifecycle.manifest['records']) == intervals, 'FORMAL_INTERVALS_COMPLETE')
    post = {}
    for i in range(intervals):
        require(lifecycle.has_authoritative_post(i), 'FORMAL_DURABLE_POST_MISSING')
        record = lifecycle.manifest['records'][i]
        post[str(i)] = lifecycle._artifact_fields(record['post'], namespace='POST')
    sinks = {p.name: np.load(p, allow_pickle=False) for p in sorted(directory.glob('sinks.*.npy'))}
    require(len(sinks) >= 4, 'FORMAL_SCIENTIFIC_SINKS')
    require(len(result[2]['z_axis']) == intervals, 'RETURN_CONTIGUOUS_COUNT')
    return normalized_return(result,post,sinks,directory/'RAW_LINEAR_PROFILE_SIDECARS.json')

"""Unobserved first legal interval of the original J1 source/PRE/BK-NEE path."""
from __future__ import annotations
import copy,time
from pathlib import Path
import numpy as np
from g1_common import require,write,factory,metrics
from g1_exact import normalized_return
from timing_adapter import instrument

def prepare(context,directory,*,intervals=1):
    from KHz_filament.constants import N0_air,Ui_N2,c0,n2_air
    from KHz_filament.grids import make_axes
    from KHz_filament.longitudinal import build_deposition_contract
    from KHz_filament.hr4e5_formal_entry import build_prefix_schedule,_all_interval_sample_plan
    from KHz_filament.thermalization import ThermalDiagnosticSink
    from KHz_filament.slow_state import HR3BDiagnosticSink,validate_hr3b_parameters
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    require(intervals in (1,8),'ONLY_PREFIX1_OR_AUTHORIZED8')
    schedule=build_prefix_schedule(context['schedule'],intervals,block_size=1 if intervals==1 else 8)
    require(tuple(schedule.z_edges)==tuple(context['schedule'].z_edges[:intervals+1]),'LEGAL_ORIGINAL_PREFIX')
    grid,beam,prop,heat=(context[key] for key in ('grid','beam','prop','heat'))
    source=context['source'];field=np.array(source,dtype=np.complex128,order='C',copy=True)
    require(not np.shares_memory(field,source),'J1_FRESH_SOURCE_COPY')
    view=j1.MemoryPREView(context['pre_fields'],state_origin='G1_FROZEN_PRE0',initial_pre0=True)
    axes=make_axes(grid.Nx,grid.Ny,grid.Nt,grid.Lx,grid.Ly,grid.Twin)
    omega0=2*np.pi*c0/float(beam.lam0);k0=float(beam.n0)*float(omega0)/c0
    sample_plan=_all_interval_sample_plan(schedule)
    thermal=ThermalDiagnosticSink(plan=sample_plan,output_path=str(directory/'sinks'),shape=(grid.Ny,grid.Nx),dtype=np.float64,enabled=True,mode='validation')
    hr3b=HR3BDiagnosticSink(plan=sample_plan,output_path=str(directory/'sinks'),shape=(grid.Ny,grid.Nx),dtype=np.float64,enabled=True)
    beta=validate_hr3b_parameters(rho0=float(heat.rho0),Cv=float(heat.Cv),T0=float(prop.air_T),n0=float(beam.n0))
    kwargs=dict(kperp2=axes.kperp2,k0=k0,omega0=float(omega0),dz=float(schedule.dz_intervals[0]),
        z_max=float(schedule.z_end-schedule.z_start),n0=float(beam.n0),n2=float(getattr(prop,'n2',getattr(beam,'n2_air',n2_air))),Ui=Ui_N2,N0=N0_air,
        ion_conf=context['ion'],dn_gas=None,dt=axes.dt,axes=axes,prop_conf=prop,raman_conf=context['raman'],record_onaxis_rho_time=True,record_every_z=1,
        longitudinal_schedule=schedule,deposition_contract=build_deposition_contract(schedule,axes=axes),thermal_sink=thermal,thermal_slow_state=view,
        hr3b_parameters=dict(rho0=float(heat.rho0),Cv=float(heat.Cv),T0=float(prop.air_T),n0=float(beam.n0),beta_th=float(beta)),hr3b_sink=hr3b)
    return field,kwargs,view

def finish(result,view,directory,intervals):
    directory=Path(directory)
    require(view._read==set(range(intervals)) and view._updated==view._read,'ORIGINAL_PRE_POST_COMPLETION')
    post={str(index):view.post_fields(index) for index in range(intervals)}
    sinks={path.name:np.load(path,allow_pickle=False) for path in sorted(directory.glob('sinks.*.npy'))}
    require(len(sinks)>=4,'SCIENTIFIC_SINK_PAYLOADS_RETAINED')
    require(len(result[2]['z_axis'])==intervals,'COMPLETE_PREFIX_RETURN')
    return normalized_return(result,post,sinks,directory/'RAW_LINEAR_PROFILE_SIDECARS.json')

class Stages:
    """Qualification-only retained refs. No copies, hash suppression or observer."""
    def __init__(self,prop,dispatch,role):self.prop=prop;self.dispatch=dispatch;self.role=role;self.records=[];self.restore=[]
    def attach(self):
        if self.role=='G_CONTROLLED_INPUT':self.dispatch.cpu_s4_3g_state._g_set_qualification_capture(True);return
        if self.role=='F_STRICT_INPUT':
            hand=self.dispatch.cpu_s4_3e_state;original=hand._internal_stage_inner
            def stage(field,**kwargs):
                value=original(field,**kwargs);self.records.append(dict(field=field,stage=value));return value
            hand._internal_stage_inner=stage;self.restore.append(lambda:setattr(hand,'_internal_stage_inner',original));return
        namespaces=[self.prop.__globals__,self.prop.__globals__['apply_isaacs_raman_operator_step'].__globals__]
        seen=set()
        for namespace in namespaces:
            if id(namespace) in seen:continue
            seen.add(id(namespace));original=namespace['isaacs_raman_stage']
            def stage(field,*,_original=original,**kwargs):
                value=_original(field,**kwargs);self.records.append(dict(field=field,stage=value));return value
            namespace['isaacs_raman_stage']=stage
            self.restore.append(lambda ns=namespace,fn=original:ns.__setitem__('isaacs_raman_stage',fn))
    def detach(self):
        for fn in reversed(self.restore):fn()
        self.restore.clear()
        if self.role=='G_CONTROLLED_INPUT':
            state=self.dispatch.cpu_s4_3g_state
            records=state._g_take_qualification_capture();state._g_set_qualification_capture(False)
            return [{key:row[key] for key in ('field','stage1','predictor','stage2')} for row in records]
        require(len(self.records)%2==0,'QUALIFICATION_TWO_STAGES_PER_HALF')
        return [dict(field=self.records[i]['field'],stage1=self.records[i]['stage'],predictor=self.records[i+1]['field'],stage2=self.records[i+1]['stage']) for i in range(0,len(self.records),2)]

class RoleAudit:
    """Known source-bound scalar evidence hook; no raw field argument/retention."""
    def __init__(self,prop,dispatch,role,clock):
        self.role=role;self.dispatch=dispatch;self.raw_rows=[];self.overhead_s=0.0
        self.state=None if role=='I0' else dispatch.cpu_s4_3e_state
        original=prop.__globals__['_timing_heun_end']
        def closed():
            original()
            started=time.perf_counter()
            if self.state is None:self.raw_rows.append(None)
            else:
                # Existing immutable event dictionaries contain scalar metadata,
                # never E/Omega/stage arrays. Retain refs; format/check outside math.
                state=self.state
                self.raw_rows.append((list(state._hash_records),__import__('numba').get_num_threads(),int(state._generation),getattr(state._local,'context',None) is None))
            self.overhead_s+=time.perf_counter()-started
        prop.__globals__['_timing_heun_end']=closed
    def reset(self):self.raw_rows.clear();self.overhead_s=0.0
    @property
    def rows(self):
        result=[]
        for value in self.raw_rows:
            if value is None:result.append(dict(route='I0',hash_roles='ORIGINAL_REFERENCE_CHECKS'));continue
            raw,mask,generation,closed=value
            roles=['E_stage1','Omega_stage1','E_apply','Omega_apply','Omega_stage2'] if self.role=='F_STRICT_INPUT' else [event['role'] for event in raw]
            require(len(raw)==len(roles),'ACTUAL_HASH_COUNT')
            result.append(dict(route='internal' if self.role=='F_STRICT_INPUT' else 'G_ACTUAL',hash_roles=roles,hash_events=[dict(event,role=assigned) for event,assigned in zip(raw,roles)],hash_logical_bytes=sum(int(item['logical_bytes']) for item in raw),mask_after=mask,generation=generation,context_closed=closed))
        return result

def role(role):
    from KHz_filament.propagate import propagate_one_pulse
    fac=factory(role);prop,dispatch=fac(propagate_one_pulse)
    name={'I0':'I0','F_STRICT_INPUT':'F','G_CONTROLLED_INPUT':'G'}[role]
    timed,clock,binding=instrument(prop,dispatch,name,factory=fac)
    audit=RoleAudit(timed,dispatch,role,clock)
    return timed,dispatch,clock,binding,audit

def verify_route(role,dispatch,before,after,audit,clock,*,intervals,fullsize):
    import numba
    require(numba.get_num_threads()==24,'OUTER_MASK24_RESTORED')
    require(len(clock.rows)==intervals and len(clock.half_rows)==2*intervals and len(audit.rows)==2*intervals,'ACTUAL_TWO_HEUN_HALVES')
    if role=='I0':return dict(route='I0',halves=2*intervals,hash_evidence=audit.rows,mask_after=24)
    key='internal_full_calls' if role=='F_STRICT_INPUT' else 'controlled_full_calls'
    require(after[key]-before[key]==2*intervals,'ACTUAL_CANDIDATE_CALLS')
    for key in ('strict_full_calls','strict_route_calls','failed_full_calls'):
        require(after.get(key,0)==before.get(key,0),'NO_STRICT_FALLBACK_OR_FAILURE:'+key)
    expected_ops=dict(R_PRODUCT=4*intervals,R_DERIVATIVE=4*intervals,R_RHS=4*intervals,H_PREDICTOR=2*intervals,H_CORRECTOR=2*intervals)
    require({key:after['operation_counts'][key]-before['operation_counts'][key] for key in expected_ops}==expected_ops,'FRESH_PRE_POST_TWO_STAGE_MATH')
    wanted=['E_stage1','Omega_stage1','E_apply','Omega_apply','Omega_stage2'] if role=='F_STRICT_INPUT' else ['E_stage1','Omega_stage1','Omega_apply','Omega_stage2']
    for row in audit.rows:require(row['hash_roles']==wanted and row['mask_after']==24 and row['context_closed'],'HALF_HASH_ROUTE_AND_CLEANUP',row)
    state=dispatch.cpu_s4_3e_state;all_executions=state.r8_state._runtime.get('executions',[])
    start=before.get('_g1_runtime_execution_count',len(all_executions)-16*intervals)
    require(len(all_executions)-start==16*intervals,'CURRENT_CALL_RUNTIME_RECORD_RANGE')
    executions=all_executions[start:]
    require(len(executions)==16*intervals,'FIVE_KERNEL_DISPATCH_COUNT')
    for row in executions:
        require(row['requested_workers']==48 and row['thread_mask_before']==24 and row['thread_mask_inside']==48 and row['thread_mask_after']==24,'ACTUAL_W48_MASKS',row)
        if fullsize:require(row['active_thread_ids']==list(range(48)),'ALL_48_REAL_WORKERS',row)
    if role=='G_CONTROLLED_INPUT':
        require(after['route_counts'].get('F_STRICT_FALLBACK',0)==before['route_counts'].get('F_STRICT_FALLBACK',0) and after['route_counts'].get('REJECTED',0)==before['route_counts'].get('REJECTED',0),'NO_G_FALLBACK_OR_REJECTION')
        require(not after['active_capability'],'NO_STALE_CAPABILITY')
        require(after['linear_producer_calls']-before['linear_producer_calls']==2*intervals,'TWO_ORIGINAL_BK_PRODUCERS_PER_INTERVAL')
        require(after['post_capability_issues']-before['post_capability_issues']==intervals,'ONE_LEGAL_NONLINEAR_WRITE_AND_FRESH_POST')
    return dict(route='internal' if role=='F_STRICT_INPUT' else 'G_ACTUAL',halves=2*intervals,hash_evidence=copy.deepcopy(audit.rows),actual_numba_kernel_executions=copy.deepcopy(executions),mask_after=24,operation_counts_delta=expected_ops,qualification_capture_enabled=after.get('qualification_capture_enabled',False),scalar_audit_overhead_s=audit.overhead_s,scalar_audit_is_source_bound_no_field_arguments=True,metadata_bound_scope='Existing partial O(1) ownership-container estimate; excludes borrowed arrays/static state, qualification refs, and original numerical workspaces. Not a complete Python object-graph bound.')

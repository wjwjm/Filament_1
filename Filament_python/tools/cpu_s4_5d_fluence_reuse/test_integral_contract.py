"""Original integral, directed shapes and the preregistered bounded cost probe."""
import ast,copy,importlib,json,os,time
from pathlib import Path
import numpy as np
import pytest
original=importlib.import_module('KHz_filament.diagnostics')
from .fluence import second_moment_radius_from_f2d
from .adapter import _apply_fluence_nodes,_StripS45D

@pytest.mark.parametrize('candidate',(False,True))
def test_qualification_inverse_has_only_fixed_first_save_witness(candidate):
    root=Path(__file__).resolve().parents[3]
    original_tree=ast.parse((root/'Filament_python/KHz_filament/propagate.py').read_text(encoding='utf-8'))
    tree=copy.deepcopy(original_tree);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='propagate_one_pulse')
    _apply_fluence_nodes(node,candidate_applied=candidate,qualification=True)
    restored=_StripS45D().visit(tree)
    assert ast.dump(restored,include_attributes=False)==ast.dump(original_tree,include_attributes=False)

def same(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape and a.strides==b.strides
    assert a.tobytes(order='A')==b.tobytes(order='A')

@pytest.mark.parametrize('kind',('gaussian','multipeak','plateau','tail','fortran','slice'))
def test_original_integral_identity_and_retained_inputs(kind):
    nt,ny,nx=(384,17,19) if kind in ('gaussian','multipeak') else (7,3,5)
    t=np.linspace(-2,2,nt);x=np.linspace(-1,1,nx);y=np.linspace(-.5,.5,ny)
    temporal=np.exp(-t*t);plane=np.exp(-2*((x[None,:]-.2)**2+(y[:,None]+.1)**2))
    if kind=='multipeak':plane+=.7*np.exp(-8*((x[None,:]+.6)**2+(y[:,None]-.2)**2))
    if kind=='plateau':plane[:]=1;plane[0,:]=2
    intensity=np.asarray(temporal[:,None,None]*plane[None,:,:],dtype=np.float64)
    if kind=='fortran':intensity=np.asfortranarray(intensity)
    if kind=='slice':intensity=np.repeat(intensity,2,axis=2)[:,:,::2]
    before=(intensity.tobytes(order='A'),intensity.dtype,intensity.shape,intensity.strides);dt=float(.25)
    trap=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    outer=trap(intensity,dx=dt,axis=0)
    inner=trap(np.asarray(intensity),dx=float(dt),axis=0)
    same(outer,inner);held=outer.copy(order='K');old=None
    for fraction in (0.,.75,.8,.999,1.):
        expected=original.second_moment_radius(intensity,x,y,dt=dt,frac_keep=fraction,rel_floor=1e-8)
        radius=second_moment_radius_from_f2d(outer,x,y,frac_keep=fraction,rel_floor=1e-8)
        assert np.float64(expected).tobytes()==np.float64(radius).tobytes()
        old=(radius,np.float64(radius).tobytes())
        original._fwhm_diameter_xy_center(outer,type('Axes',(),dict(x=x,y=y))(),ny//2,nx//2)
        next_fluence=trap(intensity*.5,dx=dt,axis=0)
        second_moment_radius_from_f2d(next_fluence,x,y,frac_keep=fraction,rel_floor=0.)
        assert np.float64(old[0]).tobytes()==old[1]
        same(outer,held)
        assert (intensity.tobytes(order='A'),intensity.dtype,intensity.shape,intensity.strides)==before

def test_preregistered_eight_local_cost_samples(tmp_path):
    root=Path(os.environ['S45D_REPORT_ROOT']);plan=json.loads((root/'S45D_EXECUTION_PLAN.json').read_text(encoding='utf-8'))
    definition=plan['local_qualification']['bounded_cost_diagnostic']
    assert definition['repetitions']==8 and definition['warmup']==0
    t=np.linspace(-2,2,384);x=np.linspace(-1,1,19);y=np.linspace(-.5,.5,17)
    rng=np.random.default_rng(definition['seed'])
    intensity=np.asarray(np.exp(-t[:,None,None]**2)*np.exp(-2*(x[None,None,:]**2+y[None,:,None]**2))*(1+rng.random((1,17,19))*.001),dtype=np.float64)
    dt=float(1e-15);trap=np.trapezoid if hasattr(np,'trapezoid') else np.trapz;rows=[]
    for i in range(8):
        a=time.perf_counter_ns();outer=trap(intensity,dx=dt,axis=0);b=time.perf_counter_ns()
        inner=trap(np.asarray(intensity),dx=float(dt),axis=0);c=time.perf_counter_ns()
        radius=second_moment_radius_from_f2d(outer,x,y,frac_keep=.999,rel_floor=1e-8);d=time.perf_counter_ns()
        same(outer,inner)
        rows.append(dict(repetition=i+1,first_integral_ns=b-a,second_integral_ns=c-b,retained_postprocess_ns=d-c,radius=radius))
    (tmp_path/'BOUNDED_FLUENCE_COST_DIAGNOSTIC.json').write_text(json.dumps(dict(status='PASS_LOCAL_ORIGINAL_INTEGRAL_EXACT_AND_FIXED_EIGHT_COST_SAMPLES',shape=[384,17,19],actual_dt_type=type(dt).__name__,dt=dt,rows=rows,propagation='NOT_RUN',W48_performance='NOT_MEASURED',historical_parent_scope_not_used=True),indent=2)+'\n',encoding='utf-8')

"""First-save witness, enabled only in explicit qualification windows."""
import importlib
from pathlib import Path
import numpy as np
original=importlib.import_module('KHz_filament.diagnostics')
from tools.cpu_s4_3g3_smallgrid import benchmark as sg

def require(ok,why):
    if not ok:raise RuntimeError(why)

def native(a):
    return dict(shape=list(a.shape),dtype=a.dtype.str,strides=list(a.strides),C_contiguous=bool(a.flags.c_contiguous),F_contiguous=bool(a.flags.f_contiguous),writeable=bool(a.flags.writeable))

def exact(a,b):
    require(native(a)==native(b) and a.tobytes(order='A')==b.tobytes(order='A'),'D_QUAL_F2D_VALUES_DTYPE_LAYOUT_EXACT')

class FirstSaveWitness:
    """No 3D copy/hash; release the first 2D reference after its FWHM consumer."""
    def __init__(self,root):
        self.root=Path(root);self.raw=None;self.before=None;self.value=None;self.completed=False

    def begin(self,I_now,F2D,dt,axes,y0,x0,frac_keep,rel_floor,interval_index):
        if self.value is not None:return
        require(type(dt) is float and type(I_now) is np.ndarray and I_now.dtype==np.dtype('float64'),'D_QUAL_ACTUAL_CPU_INPUT_DT_DOMAIN')
        require(np.asarray(I_now) is I_now and original.xp is np,'D_QUAL_ASARRAY_SAME_ACTUAL_INPUT')
        trap=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
        expected=trap(np.asarray(I_now),dx=float(dt),axis=0)
        exact(F2D,expected)
        expected_radius=original.second_moment_radius(I_now,axes.x,axes.y,dt=dt,frac_keep=frac_keep,rel_floor=rel_floor)
        expected_fwhm=original._fwhm_diameter_xy_center(F2D,axes,y0,x0)
        self.raw=F2D;self.before=F2D.copy(order='K')
        self.root.mkdir(parents=True,exist_ok=False)
        np.save(self.root/'FIRST_RAW_F2D.npy',F2D,allow_pickle=False)
        np.save(self.root/'ORIGINAL_INNER_INTEGRAL.npy',expected,allow_pickle=False)
        self.value=dict(schema='filament.cpu_s4_5d.first_saved_fluence.v1',interval_index=int(interval_index),input=native(I_now),F2D=native(F2D),original_inner=native(expected),input_asarray_is_same_object=True,actual_dt_type=type(dt).__name__,dt=dt,float_dt_same_bits=np.float64(dt).tobytes()==np.float64(float(dt)).tobytes(),trapezoid_function=trap.__module__+'.'+trap.__name__,original_diagnostics_backend_is_same_numpy=True,F2D_inner_values_dtype_layout_exact=True,outer_inner_shares_memory=bool(np.shares_memory(F2D,expected)),expected_radius=float(expected_radius),expected_fwhm=list(expected_fwhm),frac_keep=frac_keep,rel_floor=rel_floor,qualification_oracle_extra_integrals=2,qualification_only=True,extra_3D_copies=0,extra_3D_hashes=0,live_F2D_references=1)

    def radius(self,value):
        if self.value is not None and not self.completed:
            require(np.float64(value).tobytes()==np.float64(self.value['expected_radius']).tobytes(),'D_QUAL_ORIGINAL_RADIUS_EXACT')
            exact(self.raw,self.before);self.value.update(actual_radius=float(value),radius_exact=True,F2D_unchanged_after_radius=True)
        return value

    def fwhm(self,value):
        if self.value is not None and not self.completed:
            require(np.asarray(value,dtype=np.float64).tobytes()==np.asarray(self.value['expected_fwhm'],dtype=np.float64).tobytes(),'D_QUAL_ORIGINAL_FWHM_EXACT')
            exact(self.raw,self.before)
            self.value.update(actual_fwhm=list(value),FWHM_exact=True,F2D_unchanged_through_FWHM=True,live_F2D_references=0,status='PASS_ACTUAL_FIRST_SAVED_F2D_RADIUS_FWHM_EXACT')
            self.raw=None;self.before=None;self.completed=True
            sg.write(self.root/'FLUENCE_FIRST_SAVE_QUALIFICATION.json',self.value)
        return value

    def close(self):
        self.raw=None;self.before=None

    def snapshot(self):
        require(self.completed and self.raw is None and self.before is None,'D_QUAL_FIRST_SAVE_COMPLETED_AND_RETIRED')
        return dict(self.value)

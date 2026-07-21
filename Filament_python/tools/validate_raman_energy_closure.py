#!/usr/bin/env python3
"""One-step local/global energy closure for the opt-in full Isaacs operator."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.constants import c0, eps0
from KHz_filament.raman import apply_isaacs_raman_operator_step

N0, NR, WR, GR = 1.00027, 2.3e-23, 1.6e13, 1.3e13
OMEGA0 = 2*np.pi*c0/800e-9


def write_csv(path, rows):
    rows=list(rows)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def temporal_field(nt,dt,kind,dtype=np.complex128):
    t=(np.arange(nt)-nt//3)*dt;x=t
    width=40e-15 if kind=='40fs_tl' else 120e-15
    profile=np.exp(-4*np.log(2)*(x/width)**2);phase=np.zeros_like(t)
    if kind=='120fs_positive_chirp':phase=2.5e27*x*x
    if kind=='120fs_negative_chirp':phase=-2.5e27*x*x
    if kind=='120fs_tail':profile+=.2*np.exp(-((x-130e-15)/75e-15)**2)
    if kind=='120fs_double':profile+=.25*np.exp(-((x-210e-15)/45e-15)**2)
    profile*=5e17/profile.max()
    return (np.sqrt(2*profile/(eps0*c0*N0))*np.exp(1j*phase)).astype(dtype)


def fluence(field,dt):return np.sum(.5*eps0*c0*N0*np.abs(field)**2,axis=0)*dt
def step(field,dz,dt):
    omega=2*np.pi*np.fft.fftfreq(field.shape[0],dt)
    return apply_isaacs_raman_operator_step(field,dz,Omega=omega,dt=dt,omega0=OMEGA0,n0=N0,n_R=NR,
        omega_R=WR,Gamma_R=GR,integrator='heun',iir_sampling='exact_piecewise_linear',return_diagnostics=True)


def closure_row(field,dz,dt,dxdy,label):
    before=fluence(field,dt);after_field,diag=step(field,dz,dt);after=fluence(after_field,dt)
    target=np.asarray(diag['target_local_fluence_loss']);residual=after-before+target
    global_target=float(np.sum(target)*dxdy);global_change=float(np.sum(after-before)*dxdy);u0=float(np.sum(before)*dxdy)
    legacy_alpha=global_target/max(u0*dz,1e-300)
    double_counted_field=after_field*np.exp(-0.5*legacy_alpha*dz)
    double_counted_change=float(np.sum(fluence(double_counted_field,dt)-before)*dxdy)
    double_counting_extra_loss=-(double_counted_change-global_change)
    return after_field,{
      'case':label,'dz_m':dz,'field_energy_change_J':global_change,'target_energy_loss_J':global_target,
      'global_closure_residual':abs(global_change+global_target)/max(global_target,u0*1e-15),
      'global_residual_over_U0':abs(global_change+global_target)/max(u0,1e-300),
      'local_closure_residual':float(np.max(np.abs(residual)/np.maximum(target,np.maximum(before*1e-15,1e-300)))),
      'maximum_local_target_loss_J_m2':float(np.max(target)),
      'maximum_local_actual_loss_J_m2':float(np.max(before-after)),
      'minimum_after_fluence':float(np.min(after)),'maximum_target_over_available':float(np.max(target/np.maximum(before,1e-300))),
      'clipping_count':diag['clipping_count'],'finite':bool(np.isfinite(after_field).all()),
      'full_plus_legacy_energy_change_J':double_counted_change,
      'double_counting_extra_loss_J':double_counting_extra_loss,
      'double_counting_detected':bool(double_counting_extra_loss>max(global_target*0.5,u0*1e-15))}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    dt=.3125e-15;nt=4096;base_dz=1e-5;local=[]
    for kind in ('40fs_tl','120fs_tl','120fs_positive_chirp','120fs_negative_chirp','120fs_tail','120fs_double'):
        f=temporal_field(nt,dt,kind)[:,None,None];_,row=closure_row(f,base_dz,dt,1.0,kind);local.append(row)
    axis=np.linspace(-1,1,16);X,Y=np.meshgrid(axis,axis,indexing='xy');trans=np.exp(-2*(X*X+Y*Y));f=temporal_field(nt,dt,'120fs_tl')[:,None,None]*np.sqrt(trans)[None]
    _,row2d=closure_row(f,base_dz,dt,(20e-6)**2,'2d_gaussian_16x16');local.append(row2d)
    write_csv(a.out_dir/'raman_local_energy_closure.csv',local)
    write_csv(a.out_dir/'raman_global_energy_closure.csv',[{k:v for k,v in r.items() if k not in ('local_closure_residual','minimum_after_fluence','maximum_target_over_available')} for r in local])
    total=8e-5;solutions=[];conv=[]
    initial=temporal_field(nt,dt,'120fs_tl')[:,None,None]
    for count in (1,2,4,8,16):
        current=initial.copy();target=0.0
        for _ in range(count):
            current,diag=step(current,total/count,dt);target+=float(np.sum(np.asarray(diag['target_local_fluence_loss'])))
        solutions.append((count,current,target))
    reference=solutions[-1][1]
    for count,current,target in solutions[:-1]:
        error=float(np.linalg.norm(current-reference)/np.linalg.norm(reference));before=float(np.sum(fluence(initial,dt)));after=float(np.sum(fluence(current,dt)));res=abs(after-before+target)/max(target,before*1e-15)
        conv.append({'substeps':count,'dz_m':total/count,'field_error_to_16_substeps':error,'closure_residual':res})
    for i,row in enumerate(conv):
        if i+1<len(conv) and conv[i+1]['field_error_to_16_substeps']>0 and row['field_error_to_16_substeps']>0:
            row['estimated_order']=np.log(row['field_error_to_16_substeps']/conv[i+1]['field_error_to_16_substeps'])/np.log(2)
        else:row['estimated_order']=''
    write_csv(a.out_dir/'raman_dz_convergence.csv',conv)
    fig,ax=plt.subplots();ax.loglog([r['dz_m'] for r in conv],[r['field_error_to_16_substeps'] for r in conv],'o-',label='field convergence');ax.loglog([r['dz_m'] for r in conv],[r['closure_residual'] for r in conv],'s-',label='energy closure');ax.set(xlabel='dz (m)',ylabel='relative error',title='Full Isaacs Raman dz convergence');ax.legend();fig.tight_layout();fig.savefig(a.out_dir/'raman_energy_closure.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare the Eq. (27) Raman product derivative with the split source.

With the repository retarded-time convention tau=t-z/v_g and NumPy's
``exp(-i Omega tau)`` FFT convention, d/dtau maps to ``+i Omega`` in the
continuous Fourier representation.  Eq. (27) therefore gives
``S_exact=(1+i/omega0*d_tau)(I_R A)``.  The split implementation retains
``A(I_R+i/omega0*d_tau I_R)`` and omits ``i/omega0 I_R d_tau A``.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from KHz_filament.raman_isaacs_reference import isaacs_kernel, causal_convolution_direct

C0=299792458.; LAM=800e-9; W0=2*np.pi*C0/LAM; K0=W0/C0; NR=2.3e-23; IPEAK=5e17

def write_csv(path, rows):
 rows=list(rows); keys=sorted({k for r in rows for k in r})
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def pulse(t, kind):
 x=t-t[len(t)//3]; base=np.exp(-2*np.log(2)*(x/(120e-15/np.sqrt(2)))**2)
 if kind=='40fs_tl': base=np.exp(-2*np.log(2)*(x/(40e-15/np.sqrt(2)))**2); phase=0*x
 elif kind=='120fs_tl': phase=0*x
 elif kind=='120fs_pos_chirp': phase=2.5e27*x*x
 elif kind=='120fs_neg_chirp': phase=-2.5e27*x*x
 elif kind=='120fs_tail': base=base+.22*np.exp(-((x-120e-15)/(75e-15))**2);phase=0*x
 else: base=base+.35*np.exp(-((x-220e-15)/(45e-15))**2);phase=0*x
 return np.sqrt(IPEAK*base/base.max()).astype(complex)*np.exp(1j*phase)
def spectrum_metrics(a,dt):
 aw=np.fft.fftshift(np.fft.fft(a)); om=np.fft.fftshift(2*np.pi*np.fft.fftfreq(len(a),dt)); p=np.abs(aw)**2; s=p.sum()
 return float((om*p).sum()/s),float(np.sqrt(((om-(om*p).sum()/s)**2*p).sum()/s))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 dt=.3125e-15;t=np.arange(8192)*dt;h=isaacs_kernel(t,1.6e13,1.3e13);rows=[]; traces={}
 for kind in ('40fs_tl','120fs_tl','120fs_pos_chirp','120fs_neg_chirp','120fs_tail','120fs_double'):
  A=pulse(t,kind); IR=causal_convolution_direct(abs(A)**2,h,dt);dIR=np.gradient(IR,dt,edge_order=2);dA=np.gradient(A,dt,edge_order=2)
  exact=IR*A+1j*np.gradient(IR*A,dt,edge_order=2)/W0;split=A*(IR+1j*dIR/W0);omitted=1j*IR*dA/W0
  source=float(np.linalg.norm(exact-split)/np.linalg.norm(exact)); omitted_frac=float(np.linalg.norm(omitted)/np.linalg.norm(exact));scale=.01/max(np.max(np.abs(IR)),1e-300);ae=A+1j*scale*exact;as_=A+1j*scale*split
  energy=lambda x:float(np.sum(np.abs(x)**2)*dt); cen=lambda x:float(np.sum(t*np.abs(x)**2)/np.sum(np.abs(x)**2)); sc0,bw0=spectrum_metrics(A,dt);sc1,bw1=spectrum_metrics(as_,dt)
  front=float(np.sum(abs(A[t<t.mean()])**2)/max(np.sum(abs(A[t>=t.mean()])**2),1e-300))
  rows.append({'waveform':kind,'source_relative_l2_error':source,'max_pointwise_source_error':float(np.max(abs(exact-split))/np.max(abs(exact))),'omitted_term_norm_fraction':omitted_frac,'one_step_field_relative_error':float(np.linalg.norm(ae-as_)/np.linalg.norm(ae)),'one_step_energy_change':(energy(as_)-energy(A))/energy(A),'temporal_centroid_shift_fs':(cen(as_)-cen(A))*1e15,'spectral_centroid_shift_rad_s':sc1-sc0,'rms_spectral_bandwidth_rad_s':bw1,'front_back_temporal_asymmetry':front,'dz_equiv_m':.01/(K0*NR*IPEAK)})
  traces[kind]=(A,exact,split)
 write_csv(a.out_dir/'raman_operator_comparison.csv',rows)
 for name,kinds in (('raman_operator_40fs.png',('40fs_tl',)),('raman_operator_120fs.png',('120fs_tl',)),('raman_operator_chirped.png',('120fs_pos_chirp','120fs_neg_chirp'))):
  fig,ax=plt.subplots()
  for kind in kinds:
   A,ex,sp=traces[kind];ax.plot(t*1e15,np.abs(ex-sp)/max(np.max(abs(ex)),1e-300),label=kind)
  ax.set(xlabel='tau (fs)',ylabel='relative source error');ax.legend();fig.tight_layout();fig.savefig(a.out_dir/name,dpi=160);plt.close(fig)
 mapping={'tau_convention':'tau=t-z/v_g','fft_derivative_sign':'d_tau maps to +i Omega under exp(-i Omega tau)','source':'S_exact=(1+i/omega0*d_tau)(I_R A)','split':'A(I_R+i/omega0*d_tau I_R)','omitted':'i/omega0 I_R d_tau A'}
 (a.out_dir/'raman_operator_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
if __name__=='__main__':main()

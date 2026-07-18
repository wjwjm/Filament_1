#!/usr/bin/env python3
"""Fixed-coordinate Pop/Tal feedback event-chain analysis for Phase 6."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

INTS=(1e16,3e16,1e17,3e17,5e17); RHOS=(1e19,1e20,1e21,1e22); EN=(1e-6,1e-5,1e-4,1e-3,1e-2)
def read(p):
 rows=list(csv.DictReader(Path(p).open(encoding='utf-8')));return {k:np.asarray([float(r[k]) for r in rows]) for k in rows[0]}
def merge(a,b):
 for k,v in b.items():
  if k not in ('z_m','x_focus_cm'): a[k]=v
 return a
def cross(x,y,t):
 i=np.flatnonzero((y[:-1]<t)&(y[1:]>=t))
 if not len(i):return None
 j=int(i[0]);return float(x[j]+(t-y[j])*(x[j+1]-x[j])/(y[j+1]-y[j]))
def common_log(a,b,n=4):
 lo=max(np.nanmax([np.nanmin(a[a>0]),np.nanmin(b[b>0])]),1e-30);hi=min(np.nanmax(a),np.nanmax(b))
 return [] if not hi>lo else [float(x) for x in np.geomspace(lo*3,hi/3,n) if lo<x<hi]
def event(rows,case,field,thresholds,kind):
 return [{'case':case,'kind':kind,'field':field,'threshold':t,'x_cm':cross(rows['x_focus_cm'],np.abs(rows[field]),t),'status':'crossed_interpolated' if cross(rows['x_focus_cm'],np.abs(rows[field]),t)is not None else 'not_comparable'} for t in thresholds]
def writecsv(p,rows):
 rows=list(rows); keys=sorted({k for r in rows for k in r})
 with Path(p).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def plot(path,x,ys,title,log=False):
 fig,ax=plt.subplots(figsize=(7,4));
 for y,label in ys:ax.plot(x,y,label=label)
 if log:ax.set_yscale('log')
 ax.set(xlabel='x_focus (cm)',title=title);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--pop',required=True);p.add_argument('--tal',required=True);p.add_argument('--pop-raman',required=True);p.add_argument('--tal-raman',required=True);p.add_argument('--pycap',required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();o=a.out_dir;o.mkdir(parents=True,exist_ok=True)
 pop,tal=merge(read(a.pop),read(a.pop_raman)),merge(read(a.tal),read(a.tal_raman));x=pop['x_focus_cm'];eps=max(.1,3*float(np.median(np.diff(x))))
 rows=[]
 for case,d in [('popruzhenko',pop),('talebpour',tal)]:
  rows+=event(d,case,'I_onaxis_max_z',INTS,'intensity_onaxis')+event(d,case,'I_max_z',INTS,'intensity_max')+event(d,case,'rho_max_z',RHOS,'rho_total')+event(d,case,'rho_N2_max_z',RHOS,'rho_n2')+event(d,case,'rho_O2_max_z',RHOS,'rho_o2')
  for f in ('delta_n_plasma_min_z','dphi_plasma_raw_max_abs_z','dphi_plasma_applied_max_abs_z','alpha_ion_raw_max_z','alpha_ion_applied_max_z'):
   rows+=event(d,case,f,common_log(np.abs(pop[f]),np.abs(tal[f])),f)
  u0=1.0/(1.0+d['U_rel_change_z'][0]); rows+=event(d,case,'E_dep_cumulative_z',[q*u0 for q in EN],'energy_deposition_normalized')
 writecsv(o/'feedback_threshold_crossings.csv',rows)
 timeline=[]
 for r in rows:
  if r['x_cm'] is not None: timeline.append(r)
 writecsv(o/'feedback_event_timeline.csv',sorted(timeline,key=lambda r:(r['case'],r['x_cm'])))
 peak=[]
 for case,d in [('popruzhenko',pop),('talebpour',tal)]:
  i=int(np.argmax(d['rho_max_z']));peak.append({'case':case,'rho_peak_m3':float(d['rho_max_z'][i]),'rho_peak_x_cm':float(x[i]),'I_peak_W_m2':float(np.max(d['I_max_z'])),'I_peak_x_cm':float(x[np.argmax(d['I_max_z'])]),'o2_fraction_at_rho_peak':float(d['rho_O2_fraction_at_rho_total_max_z'][i])})
 writecsv(o/'feedback_peak_metrics.csv',peak)
 numeric=[]
 for case,d in [('popruzhenko',pop),('talebpour',tal)]:numeric.append({'case':case,'dz_min_m':float(d['dz_used_z'].min()),'dz_max_m':float(d['dz_used_z'].max()),'rejections_max':float(d['adaptive_rejection_count_z'].max()),'safety_max':float(d['safety_mode_trigger_count_z'].max())})
 writecsv(o/'feedback_numerical_path_comparison.csv',numeric)
 # low-density intensity separation test
 p19=cross(x,pop['rho_max_z'],1e19); t19=cross(x,tal['rho_max_z'],1e19); pre=[]
 for f in ('I_onaxis_max_z','I_max_z'):
  for q in INTS:
   dp,dt=cross(x,pop[f],q),cross(x,tal[f],q)
   if dp is not None and dt is not None and min(dp,dt)<=min(p19,t19):pre.append(abs(dt-dp))
 feedback='feedback_after_ionization' if pre and max(pre)<=eps else ('feedback_before_ionization' if pre else 'feedback_inconclusive')
 summary={'schema':'khz_filament.phase6.feedback_analysis.v1','coordinate_definition':'x_focus_cm = 100 * (z_m - 0.95)','epsilon_x_cm':eps,'feedback_classification':feedback,'pre_ionization_intensity_max_shift_cm':max(pre) if pre else None,'rho_1e19_crossings_cm':{'pop':p19,'tal':t19},'numerical_path':numeric,'raman_fields_present':True}
 (o/'existing_feedback_mechanism_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 (o/'existing_feedback_mechanism_report.md').write_text(f"# Existing Pop/Tal feedback mechanism\n\nClassification: **{feedback}**.\n\n- epsilon_x: `{eps:.3f} cm`\n- Maximum pre-ionization intensity threshold separation: `{summary['pre_ionization_intensity_max_shift_cm']}` cm\n- Numerical-path counters are reported in `feedback_numerical_path_comparison.csv`.\n",encoding='utf-8')
 plot(o/'01_intensity_absolute.png',x,[(pop['I_onaxis_max_z'],'Pop on-axis'),(tal['I_onaxis_max_z'],'Tal on-axis')],'On-axis intensity',True)
 m=(x>-22)&(x<-14);plot(o/'02_low_density_zoom.png',x[m],[(pop['rho_max_z'][m],'Pop rho'),(tal['rho_max_z'][m],'Tal rho')],'Low-density onset',True)
 plot(o/'03_species.png',x,[(pop['rho_N2_max_z'],'Pop N2'),(pop['rho_O2_max_z'],'Pop O2'),(tal['rho_N2_max_z'],'Tal N2'),(tal['rho_O2_max_z'],'Tal O2')],'Species density',True)
 plot(o/'04_o2_fraction.png',x,[(pop['rho_O2_fraction_at_rho_total_max_z'],'Pop'),(tal['rho_O2_fraction_at_rho_total_max_z'],'Tal')],'O2 fraction')
 plot(o/'05_plasma_phase.png',x,[(pop['dphi_plasma_applied_max_abs_z'],'Pop'),(tal['dphi_plasma_applied_max_abs_z'],'Tal')],'Applied plasma phase',True)
 plot(o/'06_ion_loss.png',x,[(pop['alpha_ion_applied_max_z'],'Pop'),(tal['alpha_ion_applied_max_z'],'Tal')],'Applied ionization loss',True)
 plot(o/'07_energy.png',x,[(pop['E_dep_cumulative_z'],'Pop'),(tal['E_dep_cumulative_z'],'Tal')],'Cumulative deposited energy',True)
 plot(o/'08_events.png',x,[(pop['dphi_rot_applied_max_abs_z'],'Pop Raman phase'),(tal['dphi_rot_applied_max_abs_z'],'Tal Raman phase')],'Raman phase',True)
 plot(o/'09_numerical_path.png',x,[(pop['dz_used_z'],'Pop dz'),(tal['dz_used_z'],'Tal dz')],'Actual step size')
 print(feedback)
if __name__=='__main__':main()

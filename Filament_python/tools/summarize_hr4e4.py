#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from KHz_filament.device import xp
from KHz_filament.hr4 import apply_hr4_boundaries, compute_hr4_rhs, laplacian_fd
from KHz_filament.hr4e_reduced_limits import load_checkpoint
from KHz_filament.hr4e_timestep import HR4_N0,json_safe

def snap(case,t): return next(x for x in case['snapshots'] if math.isclose(x['time_us'],t,abs_tol=1e-9))
def pass_norm(metric,rel,absolute): return metric['relative_Linf']<=rel or metric['absolute_Linf']<=absolute
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',action='append',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
 if a.out_dir.exists(): raise FileExistsError(a.out_dir)
 cases={x['case_id']:x for x in (json.loads(v.read_text(encoding='utf-8')) for v in a.case)}
 required={'E4A','E4Bnu','E4B0','E4Cminus','E4Cplus','E4D'}
 if set(cases)!=required: raise ValueError('E4 case set mismatch')
 result={"schema":"khz_filament.hr4e4.summary.v1","cases":{},"advection_evidence":"E4-ADV = SATISFIED_BY_REUSED_E2B_EVIDENCE"}
 # A: analytic Gaussian limits at two requested horizons.
 arows=[]
 for t in (100.0,1000.0):
  m=snap(cases['E4A'],t)['metrics']; o,r=m['observables'],m['reference_observables']; e=m['errors']['delta_n']
  centroid=max(abs(o['xc_m']-r['xc_m']),abs(o['yc_m']-r['yc_m']))<=0.5e-6; width=max(abs(o['sigma_x_m']-r['sigma_x_m'])/r['sigma_x_m'],abs(o['sigma_y_m']-r['sigma_y_m'])/r['sigma_y_m'])<=0.005; peak=abs(o['min_delta_n']-r['min_delta_n'])/abs(r['min_delta_n'])<=0.01
  ok=centroid and width and peak and m['mass_relative_drift']<=0.005 and e['relative_L2']<=0.01 and e['relative_Linf']<=0.02 and m['finite']; arows.append({"time_us":t,"metrics":m,"pass":ok})
 result['cases']['E4A']={"status":"PASS" if all(x['pass'] for x in arows) else "HARD_OPERATOR_FAILURE","rows":arows}
 # C: full-field source reference already includes the production final boundary mapping.
 for cid,sign in (('E4Cminus',1),('E4Cplus',-1)):
  m=snap(cases[cid],1.0)['metrics']; ok=m['finite'] and m['max_abs']['vx']<=1e-12 and pass_norm(m['errors']['vy'],1e-10,1e-12) and m['interior_vy_sign']*sign>0
  result['cases'][cid]={"status":"PASS" if ok else "HARD_OPERATOR_FAILURE","metrics":m,"pass":ok,"reference":"Euler buoyancy source followed by frozen open-boundary mapping"}
 # D exact zero.
 drows=[]
 for t in (0.0,100.0,1000.0):
  m=snap(cases['E4D'],t)['metrics']; ok=m['finite'] and all(v==0.0 for v in m['max_abs'].values()); drows.append({"time_us":t,"metrics":m,"pass":ok})
 result['cases']['E4D']={"status":"PASS" if all(x['pass'] for x in drows) else "HARD_OPERATOR_FAILURE","rows":drows}
 # Bnu-B0: reproduce the discrete viscosity increment and final frozen boundary mapping.
 bnu,b0=load_checkpoint(snap(cases['E4Bnu'],1.0)['checkpoint']['path']),load_checkpoint(snap(cases['E4B0'],1.0)['checkpoint']['path']); initial=load_checkpoint(snap(cases['E4Bnu'],0.0)['checkpoint']['path'])
 dt=float(bnu['metadata']['dt_hydro_s']); nu=float(bnu['metadata']['nu']); dx=float(bnu['metadata']['grid']['dx_m']); dy=float(bnu['metadata']['grid']['dy_m'])
 rhs=compute_hr4_rhs(xp.asarray(initial['delta_n']),xp.asarray(initial['vx']),xp.asarray(initial['vy']),dx=dx,dy=dy,chi=float(b0['metadata']['chi']),nu=0.0,n0=HR4_N0,gravity_y=float(b0['metadata']['gravity_y']))
 pre_vx=rhs['old_vx']+dt*rhs['rhs_vx']; increment=dt*nu*laplacian_fd(rhs['old_vx'],dx=dx,dy=dy); _,pred_vx,_=apply_hr4_boundaries(rhs['old_delta_n'],pre_vx+increment,rhs['old_vy']+dt*rhs['rhs_vy'])
 observed=xp.asarray(bnu['vx'])-xp.asarray(b0['vx']); err=npnorm(observed,pred_vx-xp.asarray(b0['vx'])); vyerr=float(abs(bnu['vy']-b0['vy']).max()); bpass=pass_norm(err,1e-10,1e-12) and vyerr<=1e-12
 result['cases']['E4B']={"status":"PASS" if bpass else "HARD_OPERATOR_FAILURE","difference_error":err,"delta_vy_max_abs":vyerr,"pass":bpass,"reference":"dt*nu*central_FD_laplacian(vx0), then frozen final boundary mapping"}
 final=all(item['status']=='PASS' for item in result['cases'].values()) and all(cases[k]['status']=='PASS' for k in cases)
 result['final_decision']='HR-4E-4 = CLOSED / PASS' if final else 'HR-4E-4 = HARD_OPERATOR_FAILURE'; result['status']='PASS' if final else 'HARD_OPERATOR_FAILURE'
 a.out_dir.mkdir(parents=True); (a.out_dir/'hr4e4_summary.json').write_text(json.dumps(json_safe(result),indent=2,sort_keys=True)+'\n',encoding='utf-8'); (a.out_dir/'HR4E4_CLOSEOUT.md').write_text(f'# HR-4E-4 Reduced Limits\n\nFinal decision: **{result["final_decision"]}**.\n',encoding='utf-8')
 print(json.dumps({'status':result['status'],'out_dir':str(a.out_dir)})); return 0 if final else 2
def npnorm(a,b):
 import numpy as np
 x,y=np.asarray(a),np.asarray(b);e=x-y; li=float(np.max(np.abs(e)));ri=float(np.max(np.abs(y)));return {'absolute_Linf':li,'relative_Linf':0.0 if ri==0 and li==0 else float('inf') if ri==0 else li/ri}
if __name__=='__main__': raise SystemExit(main())

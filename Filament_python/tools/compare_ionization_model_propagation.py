#!/usr/bin/env python3
"""Compare fixed-coordinate Popruzhenko, Talebpour, and PyCAP axial densities."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

THRESHOLDS=(1e19,1e20,1e21,1e22)

def read_csv(path: Path, y: str, scale: float=1.0):
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    x=np.asarray([float(r['x_focus_cm']) for r in rows]); v=np.asarray([float(r[y])*scale for r in rows])
    return x,v

def first_cross(x,y,t):
    hit=np.flatnonzero((y[:-1]<t)&(y[1:]>=t))
    if not len(hit): return {'status':'not_crossed','x_cm':None}
    i=int(hit[0]); return {'status':'crossed_interpolated','x_cm':float(x[i]+(t-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i]))}

def cross_desc(x,y,t,start):
    hit=np.flatnonzero((np.arange(len(x)-1)>=start)&(y[:-1]>=t)&(y[1:]<t))
    if not len(hit): return None
    i=int(hit[0]); return float(x[i]+(t-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i]))

def metrics(x,y):
    peak_i=int(np.argmax(y)); peak=float(y[peak_i]); top=np.flatnonzero(y>=.99*peak)
    left,right=int(top[0]),int(top[-1]); center=float((x[left]+x[right])/2)
    half=0.5*peak; l=first_cross(x[:peak_i+1],y[:peak_i+1],half)['x_cm']; r=cross_desc(x,y,half,peak_i)
    fwhm=None if l is None or r is None else float(r-l)
    tail=np.trapezoid(np.maximum(y[peak_i:]-half,0),x[peak_i:])
    return {'rho_peak_m3':peak,'peak_x_cm':float(x[peak_i]),'peak_top_center_cm':center,'fwhm_cm':fwhm,'post_peak_half_distance_cm':None if r is None else float(r-x[peak_i]),'tail_area_above_half_m3_cm':float(tail),'thresholds':{str(int(t)):first_cross(x,y,t) for t in THRESHOLDS}}

def rmse(a,b,w=None):
    d=a-b
    return float(math.sqrt(np.average(d*d,weights=w)))

def compare(pop,tal,paper):
    xp,yp=pop; xt,yt=tal; xc,yc=paper
    mp,mt,mc=metrics(xp,yp),metrics(xt,yt),metrics(xc,yc)
    lo=max(xp.min(),xt.min(),xc.min()); hi=min(xp.max(),xt.max(),xc.max()); grid=np.arange(lo,hi+1e-12,.025)
    p=np.interp(grid,xp,yp); t=np.interp(grid,xt,yt); c=np.interp(grid,xc,yc)
    rise_start=mp['thresholds'][str(int(1e19))]['x_cm']; rise_end=mp['peak_top_center_cm']
    w=np.where((grid>=rise_start)&(grid<=rise_end),4.,1.) if rise_start is not None else np.ones_like(grid)
    shifts=np.arange(-3,3.0001,.025); opt=min((rmse(np.interp(grid+s,xt,yt),c),float(s)) for s in shifts)
    eps=max(.10,3*float(np.median(np.diff(xp))))
    d21=mt['thresholds'][str(int(1e21))]['x_cm']-mp['thresholds'][str(int(1e21))]['x_cm']
    peak_improve=abs(mt['peak_top_center_cm']-mc['peak_top_center_cm'])+eps < abs(mp['peak_top_center_cm']-mc['peak_top_center_cm'])
    tail_bad=(mt['tail_area_above_half_m3_cm']>2*mp['tail_area_above_half_m3_cm'])
    classification='propagation_supported' if d21>eps and peak_improve and not tail_bad else 'not_supported'
    return {'coordinate_definition':'x_focus_cm = 100 * (z_m - 0.95)','epsilon_x_cm':eps,'metrics':{'popruzhenko':mp,'talebpour':mt,'pycap':mc},'unshifted_rmse_m3':{'popruzhenko_vs_pycap':rmse(p,c),'talebpour_vs_pycap':rmse(t,c),'talebpour_vs_pycap_rising_weighted':rmse(t,c,w)},'diagnostic_best_shift_cm_for_talebpour_to_pycap':opt[1],'threshold_1e21_shift_tal_minus_pop_cm':d21,'peak_center_improves_vs_pycap':peak_improve,'tail_unacceptably_worse':tail_bad,'classification':classification}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--pop',type=Path,required=True); p.add_argument('--tal',type=Path,required=True); p.add_argument('--pycap',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True); a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
 result=compare(read_csv(a.pop,'rho_max_z'),read_csv(a.tal,'rho_max_z'),read_csv(a.pycap,'rho_1e16_cm3',1e22))
 (a.out_dir/'ionization_model_propagation_120fs_comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 lines=['# 120 fs ionization-model propagation comparison','',f"Classification: **{result['classification']}**.",'',f"- epsilon_x: `{result['epsilon_x_cm']:.3f} cm`",f"- Talebpour minus Popruzhenko 1e21 onset: `{result['threshold_1e21_shift_tal_minus_pop_cm']:.3f} cm`",f"- Talebpour peak-center improves toward PyCAP: `{result['peak_center_improves_vs_pycap']}`",f"- Tail unacceptable: `{result['tail_unacceptably_worse']}`",f"- Formal curves were not shifted; diagnostic best shift is `{result['diagnostic_best_shift_cm_for_talebpour_to_pycap']:.3f} cm`."]
 (a.out_dir/'ionization_model_propagation_120fs_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(result['classification'])
if __name__=='__main__': main()

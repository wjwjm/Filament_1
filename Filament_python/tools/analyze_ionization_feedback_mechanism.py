#!/usr/bin/env python3
"""Fixed-coordinate Pop/Tal feedback analysis with physical energy normalization."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
INTS=(1e16,3e16,1e17,3e17,5e17); RHOS=(1e19,1e20,1e21,1e22); EN=(1e-6,1e-5,1e-4,1e-3,1e-2)
def read(p):
 r=list(csv.DictReader(Path(p).open(encoding="utf-8")));return {k:np.asarray([float(x[k]) for x in r],float) for k in r[0]}
def merge(a,b):
 if not np.array_equal(a["z_m"],b["z_m"]):raise ValueError("base/Raman axes differ")
 return {**a,**{k:v for k,v in b.items() if k not in("z_m","x_focus_cm")}}
def cross(x,y,t):
 i=np.flatnonzero((y[:-1]<t)&(y[1:]>=t));
 if not len(i):return None
 j=int(i[0]);return float(x[j]+(t-y[j])*(x[j+1]-x[j])/(y[j+1]-y[j]))
def rows_for(d,case,u0):
 if not u0>0:raise ValueError("U0_J must be positive")
 x=d["x_focus_cm"];out=[]
 for field,ts,kind in (("I_onaxis_max_z",INTS,"intensity_onaxis"),("I_max_z",INTS,"intensity_max"),("rho_max_z",RHOS,"rho_total"),("rho_N2_max_z",RHOS,"rho_n2"),("rho_O2_max_z",RHOS,"rho_o2")):
  for t in ts:
   z=cross(x,np.abs(d[field]),t);out.append({"case":case,"kind":kind,"field":field,"threshold":t,"threshold_fraction":None,"U0_J":u0,"E_dep_cumulative_J":None,"E_dep_fraction":None,"x_crossing_cm":z,"status":"crossed_interpolated" if z is not None else "not_crossed"})
 frac=d["E_dep_cumulative_z"]/u0
 if np.any(np.diff(d["E_dep_cumulative_z"])<-1e-10):raise ValueError("nonphysical cumulative energy decrease")
 for t in EN:
  z=cross(x,frac,t);out.append({"case":case,"kind":"energy_deposition_fraction","field":"E_dep_cumulative_z","threshold":None,"threshold_fraction":t,"U0_J":u0,"E_dep_cumulative_J":None,"E_dep_fraction":None,"x_crossing_cm":z,"status":"crossed_interpolated" if z is not None else "not_crossed"})
 return out,frac
def write(p,rows):
 rows=list(rows);keys=sorted({k for r in rows for k in r});
 with Path(p).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();
 for k in("pop","tal","pop_raman","tal_raman","pop_summary","tal_summary"):p.add_argument("--"+k.replace("_","-"),required=True,type=Path)
 p.add_argument("--out-dir",required=True,type=Path);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 pop,tal=merge(read(a.pop),read(a.pop_raman)),merge(read(a.tal),read(a.tal_raman));u={"popruzhenko":float(json.loads(a.pop_summary.read_text(encoding="utf-8"))["metrics"]["U0_J"]),"talebpour":float(json.loads(a.tal_summary.read_text(encoding="utf-8"))["metrics"]["U0_J"])}
 rows=[];fractions={}
 for case,d in (("popruzhenko",pop),("talebpour",tal)):
  r,f=rows_for(d,case,u[case]);rows+=r;fractions[case]=f
 timeline=sorted((r for r in rows if r["x_crossing_cm"] is not None),key=lambda r:(r["case"],r["x_crossing_cm"]))
 write(a.out_dir/"feedback_threshold_crossings.csv",rows);write(a.out_dir/"feedback_event_timeline.csv",timeline);write(a.out_dir/"corrected_feedback_threshold_crossings.csv",rows);write(a.out_dir/"corrected_feedback_event_timeline.csv",timeline)
 energy=[]
 for case,d in (("popruzhenko",pop),("talebpour",tal)):
  for z,e,frac in zip(d["z_m"],d["E_dep_cumulative_z"],fractions[case]):energy.append({"case":case,"z_m":z,"x_focus_cm":100*(z-.95),"U0_J":u[case],"E_dep_cumulative_J":e,"E_dep_fraction":frac})
 write(a.out_dir/"corrected_feedback_energy_normalization.csv",energy)
 eps=max(.1,3*float(np.median(np.diff(pop["x_focus_cm"]))));p19,t19=cross(pop["x_focus_cm"],pop["rho_max_z"],1e19),cross(tal["x_focus_cm"],tal["rho_max_z"],1e19);pre=[]
 for f in ("I_onaxis_max_z","I_max_z"):
  for q in INTS:
   a1,b1=cross(pop["x_focus_cm"],pop[f],q),cross(tal["x_focus_cm"],tal[f],q)
   if a1 is not None and b1 is not None and min(a1,b1)<=min(p19,t19):pre.append(abs(a1-b1))
 summary={"schema":"khz_filament.phase6.feedback_analysis.corrected.v1","coordinate_definition":"x_focus_cm = 100 * (z_m - 0.95)","epsilon_x_cm":eps,"feedback_classification":"feedback_after_ionization" if pre and max(pre)<=eps else ("feedback_before_ionization" if pre else "feedback_inconclusive"),"pre_ionization_intensity_max_shift_cm":max(pre) if pre else None,"energy_normalization":{"popruzhenko_U0_J":u["popruzhenko"],"talebpour_U0_J":u["talebpour"],"thresholds_fraction":list(EN)}}
 (a.out_dir/"existing_feedback_mechanism_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");(a.out_dir/"existing_feedback_mechanism_report.md").write_text("# Corrected existing Pop/Tal feedback mechanism\n\nEnergy-deposition thresholds use each case's diagnostic-summary `U0_J`, never `U_rel_change_z`.\n\nClassification: **"+summary["feedback_classification"]+"**.\n",encoding="utf-8")
 print(summary["feedback_classification"])
if __name__=="__main__":main()

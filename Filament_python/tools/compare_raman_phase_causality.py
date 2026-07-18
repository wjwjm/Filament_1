#!/usr/bin/env python3
"""Corrected fixed-coordinate 120 fs Raman-phase causal comparison."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Any
import numpy as np
RHO=(1e19,1e20,1e21,1e22); INT=(1e16,3e16,1e17,3e17,5e17)
def read(p:Path)->dict[str,np.ndarray]:
 r=list(csv.DictReader(p.open(encoding="utf-8"))); 
 if not r:raise ValueError(f"empty CSV: {p}")
 return {k:np.asarray([float(x[k]) for x in r],float) for k in r[0]}
def merge(a,b):
 if not np.array_equal(a["z_m"],b["z_m"]):raise ValueError("base/Raman axes differ")
 return {**a,**{k:v for k,v in b.items() if k not in("z_m","x_focus_cm")}}
def cross(x,y,t,desc=False,start=0):
 a,b=y[:-1],y[1:];m=((a>=t)&(b<t)) if desc else ((a<t)&(b>=t));i=np.flatnonzero(m&(np.arange(len(a))>=start))
 if not len(i):return None
 j=int(i[0]);return float(x[j]+(t-a[j])*(x[j+1]-x[j])/(b[j]-a[j]))
def density(x,y):
 i=int(np.argmax(y));peak=float(y[i]);top=y>=.99*peak;l=i;r=i
 while l and top[l-1]:l-=1
 while r+1<len(y) and top[r+1]:r+=1
 half=.5*peak;hl=cross(x[:i+1],y[:i+1],half);hr=cross(x,y,half,True,i)
 return {"rho_peak_m3":peak,"peak_x_cm":float(x[i]),"peak_top_center_cm":float((x[l]+x[r])/2),"fwhm_cm":None if hl is None or hr is None else float(hr-hl),"post_peak_half_distance_cm":None if hr is None else float(hr-x[i]),"tail_area_above_half_m3_cm":float(np.trapezoid(np.maximum(y[i:]-half,0),x[i:]),),"crossings":{str(int(t)):cross(x,y,t) for t in RHO}}
def err(v,ref):return None if v is None or ref is None else abs(float(v)-float(ref))
def status(f,o,p):
 if p is None:return "not_available_in_pycap"
 if f is None:return "not_crossed_by_full"
 if o is None:return "not_crossed_by_off"
 return "comparable_to_pycap"
def tail_compare(full,off,paper,tol=1e-12):
 ef,eo=abs(full-paper)/max(abs(paper),1e-300),abs(off-paper)/max(abs(paper),1e-300)
 return {"tail_full":full,"tail_off":off,"tail_pycap":paper,"tail_error_full_abs":abs(full-paper),"tail_error_off_abs":abs(off-paper),"tail_error_full_rel":ef,"tail_error_off_rel":eo,"tail_improves_vs_pycap":ef+tol<eo,"tail_worsens_vs_pycap":eo+tol<ef,"tail_full_over_off":full/off if off else None}
def is_peak_collapse(full_peak,off_peak,fraction=.5):return float(off_peak)<float(fraction)*float(full_peak)
def rmse(x,y,px,py):
 lo,hi=max(x.min(),px.min()),min(x.max(),px.max());g=np.arange(lo,hi+.00001,.025);return float(np.sqrt(np.mean((np.interp(g,x,y)-np.interp(g,px,py))**2)))
def config_diff(full:dict,off:dict):
 def flat(d,p=""):
  out={}
  if isinstance(d,dict):
   for k,v in d.items():out.update(flat(v,f"{p}.{k}" if p else k))
  else:out[p]=d
  return out
 a,b=flat(full),flat(off);return [{"path":k,"full":a.get(k),"raman_off":b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
def classify(*,valid,numerical,effect,collapse,improvements,conflict):
 if not valid or not numerical:return "raman_phase_inconclusive"
 if not effect:return "raman_phase_not_supported"
 if not collapse and improvements>=3 and not conflict:return "raman_phase_supported"
 return "raman_phase_partially_supported"
def main():
 p=argparse.ArgumentParser()
 for k in("full","full_raman","raman_off","raman_off_raman","pycap","full_summary","raman_off_summary","full_config","raman_off_config"):p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
 p.add_argument("--out-dir",type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 f,o=merge(read(a.full),read(a.full_raman)),merge(read(a.raman_off),read(a.raman_off_raman));paper=read(a.pycap);px,py=paper["x_focus_cm"],paper["rho_1e16_cm3"]*1e22;x=f["x_focus_cm"];eps=max(.1,3*float(np.median(np.diff(x))))
 if not np.array_equal(x,o["x_focus_cm"]):raise ValueError("full/off axes differ")
 fm,om,pm=density(x,f["rho_max_z"]),density(x,o["rho_max_z"]),density(px,py)
 threshold=[]
 for t in RHO:
  k=str(int(t));fv,ov,pv=fm["crossings"][k],om["crossings"][k],pm["crossings"][k];s=status(fv,ov,pv);delta=None if fv is None or ov is None else fv-ov
  threshold.append({"family":"rho_total","threshold":t,"x_full_cm":fv,"x_off_cm":ov,"delta_full_minus_off_cm":delta,"abs_delta_cm":None if delta is None else abs(delta),"effect_resolved":None if delta is None else abs(delta)>eps,"x_pycap_cm":pv,"full_error_to_pycap_cm":err(fv,pv) if s=="comparable_to_pycap" else None,"off_error_to_pycap_cm":err(ov,pv) if s=="comparable_to_pycap" else None,"pycap_comparison_status":s,"improves_vs_pycap":None if s!="comparable_to_pycap" else err(ov,pv)+eps<err(fv,pv)})
 for field in ("I_onaxis_max_z","I_max_z"):
  for t in INT:
   fv,ov=cross(x,f[field],t),cross(x,o[field],t);d=None if fv is None or ov is None else fv-ov
   threshold.append({"family":field,"threshold":t,"x_full_cm":fv,"x_off_cm":ov,"delta_full_minus_off_cm":d,"abs_delta_cm":None if d is None else abs(d),"effect_resolved":None if d is None else abs(d)>eps,"x_pycap_cm":None,"full_error_to_pycap_cm":None,"off_error_to_pycap_cm":None,"pycap_comparison_status":"not_available_in_pycap","improves_vs_pycap":None})
 fs,os=json.loads(a.full_summary.read_text(encoding="utf-8")),json.loads(a.raman_off_summary.read_text(encoding="utf-8"));u0f,u0o=float(fs["metrics"]["U0_J"]),float(os["metrics"]["U0_J"])
 feedback=[]
 for name,d,u0 in (("full",f,u0f),("raman_off",o,u0o)):
  for field in ("dphi_plasma_applied_max_abs_z","alpha_ion_applied_max_z","rho_O2_max_z","rho_N2_max_z"):
   i=int(np.argmax(np.abs(d[field])));feedback.append({"case":name,"metric":field,"peak":float(d[field][i]),"peak_x_cm":float(x[i]),"U0_J":u0})
  dep=d["E_dep_cumulative_z"]/u0;feedback.append({"case":name,"metric":"E_dep_fraction_peak","peak":float(dep.max()),"peak_x_cm":float(x[int(dep.argmax())]),"U0_J":u0})
  for q in (1e-6,1e-5,1e-4,1e-3,1e-2):feedback.append({"case":name,"metric":"E_dep_fraction_crossing","threshold_fraction":q,"peak":None,"peak_x_cm":cross(x,dep,q),"U0_J":u0})
 numeric=[]
 for name,d in (("full",f),("raman_off",o)):numeric.append({"case":name,"dz_min_m":float(d["dz_used_z"].min()),"dz_max_m":float(d["dz_used_z"].max()),"rejections_max":float(d["adaptive_rejection_count_z"].max()),"safety_max":float(d["safety_mode_trigger_count_z"].max())})
 numerical=all(numeric[0][k]==numeric[1][k] for k in ("dz_min_m","dz_max_m","rejections_max","safety_max"));collapse=is_peak_collapse(fm["rho_peak_m3"],om["rho_peak_m3"]);tc=tail_compare(fm["tail_area_above_half_m3_cm"],om["tail_area_above_half_m3_cm"],pm["tail_area_above_half_m3_cm"])
 center_err_f,center_err_o=err(fm["peak_top_center_cm"],pm["peak_top_center_cm"]),err(om["peak_top_center_cm"],pm["peak_top_center_cm"]);fwhm_err_f,fwhm_err_o=err(fm["fwhm_cm"],pm["fwhm_cm"]),err(om["fwhm_cm"],pm["fwhm_cm"]);rmf,rmo=rmse(x,f["rho_max_z"],px,py),rmse(x,o["rho_max_z"],px,py)
 improvements=sum((center_err_o+eps<center_err_f,fwhm_err_o+eps<fwhm_err_f,tc["tail_error_off_rel"]+1e-12<tc["tail_error_full_rel"],rmo<rmf));effect=any(bool(r["effect_resolved"]) for r in threshold if r["family"]=="rho_total" and r["threshold"]>=1e20);conflict=any((center_err_o+eps<center_err_f,not (fwhm_err_o+eps<fwhm_err_f),not tc["tail_worsens_vs_pycap"],not (rmo<rmf)))
 valid=bool(np.all(np.isfinite(f["rho_max_z"])) and np.all(np.isfinite(o["rho_max_z"])) and np.max(np.abs(o["dphi_rot_applied_max_abs_z"]))==0 and np.max(np.abs(o["dphi_rot_max_abs_z"]))>0 and np.max(o["alpha_R_applied_max_z"])>0)
 label=classify(valid=valid,numerical=numerical,effect=effect,collapse=collapse,improvements=improvements,conflict=conflict)
 peak={"rho_peak_full":fm["rho_peak_m3"],"rho_peak_off":om["rho_peak_m3"],"rho_peak_pycap":pm["rho_peak_m3"],"rho_peak_off_over_full":om["rho_peak_m3"]/fm["rho_peak_m3"],"peak_collapse_fraction":.5,"peak_collapse":collapse,"rho_peak_error_full":err(fm["rho_peak_m3"],pm["rho_peak_m3"]),"rho_peak_error_off":err(om["rho_peak_m3"],pm["rho_peak_m3"]),"peak_x_full_cm":fm["peak_x_cm"],"peak_x_off_cm":om["peak_x_cm"],"peak_x_pycap_cm":pm["peak_x_cm"],"peak_top_center_full_cm":fm["peak_top_center_cm"],"peak_top_center_off_cm":om["peak_top_center_cm"],"peak_top_center_pycap_cm":pm["peak_top_center_cm"],"delta_peak_center_full_minus_off_cm":fm["peak_top_center_cm"]-om["peak_top_center_cm"],"full_peak_center_error_to_pycap_cm":center_err_f,"off_peak_center_error_to_pycap_cm":center_err_o,"R_Raman_peak_center":(om["peak_top_center_cm"]-fm["peak_top_center_cm"])/(pm["peak_top_center_cm"]-fm["peak_top_center_cm"]),"fwhm_full_cm":fm["fwhm_cm"],"fwhm_off_cm":om["fwhm_cm"],"fwhm_pycap_cm":pm["fwhm_cm"],"delta_fwhm_cm":fm["fwhm_cm"]-om["fwhm_cm"],"fwhm_error_full":fwhm_err_f,"fwhm_error_off":fwhm_err_o,"post_peak_half_distance_full_cm":fm["post_peak_half_distance_cm"],"post_peak_half_distance_off_cm":om["post_peak_half_distance_cm"],"post_peak_half_distance_pycap_cm":pm["post_peak_half_distance_cm"],**tc,"rmse_full_vs_pycap":rmf,"rmse_off_vs_pycap":rmo}
 full_cfg,off_cfg=json.loads(a.full_config.read_text(encoding="utf-8")),json.loads(a.raman_off_config.read_text(encoding="utf-8"));diff=config_diff(full_cfg,off_cfg)
 def write(name,rows):
  rows=list(rows);keys=sorted({k for r in rows for k in r});
  with (a.out_dir/name).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)
 write("raman_threshold_comparison.csv",threshold);write("raman_peak_width_comparison.csv",[peak]);write("raman_feedback_comparison.csv",feedback);write("raman_numerical_path_comparison.csv",numeric);(a.out_dir/"raman_config_diff.json").write_text(json.dumps({"differences":diff,"valid_only_raman_phase_switch":diff==[{"path":"propagation.use_raman_phase","full":True,"raman_off":False}]},indent=2)+"\n",encoding="utf-8")
 summary={"schema":"khz_filament.phase6.corrected.v1","coordinate_definition":"x_focus_cm = 100 * (z_m - 0.95)","epsilon_x_cm":eps,"classification":label,"validity_gate":valid,"numerical_path_comparable":numerical,"causal_effect_resolved":effect,"peak":peak,"thresholds":threshold,"feedback_u0_J":{"full":u0f,"raman_off":u0o},"classification_basis":{"peak_collapse":collapse,"pycap_low_density_crossings_unavailable":True,"improvements_for_raman_off":improvements,"directional_conflict":conflict},"config_diff":diff}
 (a.out_dir/"phase6_corrected_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");write("phase6_corrected_decision.csv",[{"classification":label,"epsilon_x_cm":eps,"validity_gate":valid,"numerical_path_comparable":numerical,"causal_effect_resolved":effect,"peak_collapse":collapse,"rmse_full_vs_pycap":rmf,"rmse_off_vs_pycap":rmo,"tail_full_rel_error":tc["tail_error_full_rel"],"tail_off_rel_error":tc["tail_error_off_rel"]}])
 report=["# Corrected Phase 6 Raman causality","",f"Classification: **{label}**.","",f"- Raman-off/full peak ratio: `{peak['rho_peak_off_over_full']:.4f}`; collapse (<0.5): `{collapse}`.",f"- 1e21 causal full-minus-off onset: `{next(r['delta_full_minus_off_cm'] for r in threshold if r['family']=='rho_total' and r['threshold']==1e21)} cm`.",f"- Full/off RMSE: `{rmf:.4e}` / `{rmo:.4e} m^-3`.",f"- Tail relative error full/off: `{tc['tail_error_full_rel']:.4f}` / `{tc['tail_error_off_rel']:.4f}`.","- Low-density PyCAP crossings are unavailable: causal shifts are reported but are not misclassified as PyCAP failures.","- R_Raman is an endpoint diagnostic only; nonlinear propagation forbids scaling Raman parameters from it."]
 (a.out_dir/"phase6_corrected_report.md").write_text("\n".join(report)+"\n",encoding="utf-8");print(label)
if __name__=="__main__":main()

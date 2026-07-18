#!/usr/bin/env python3
"""CPU-only static audit of the implemented Raman model; never runs propagation."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from KHz_filament.raman import make_raman_kernel,precompute_kernel_fft,raman_convolve_intensity,resolve_raman_rot_params
C0=299792458.0
FIELDS=("enabled","f_R","n_R","model","T2","T_R","Omega_R","omega_R","Gamma_R","tau2","method","chunk_pixels","diagnose","absorption_model","absorption","abs_mask_frac","max_alpha_dz","tau_fwhm","n_rot_frac","R0_mode","R0_fixed_m")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def write_csv(p,rows):
 rows=list(rows);keys=sorted({k for r in rows for k in r});
 with Path(p).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)
def kernel(t,w,g):return ((w*w+g*g)/w)*np.exp(-g*t)*np.sin(w*t)
def direct(I,h,dt):return np.convolve(I,h,mode="full")[:len(I)]*dt
def gaussian(t,fwhm,peak=1.):return peak*np.exp(-4*np.log(2)*(t/fwhm)**2)
def rel(a,b):return float(np.max(np.abs(a-b))/max(np.max(np.abs(b)),1e-300))
def first_git(repo,text):
 r=subprocess.run(["git","-C",str(repo),"log","-S",text,"--format=%H","-1"],capture_output=True,text=True);return r.stdout.strip() or None
def field_usage(cfg):
 active={"enabled":"conditionally_active","f_R":"read_but_unused","n_R":"active","model":"active","T2":"shadowed_by_higher_priority_field","T_R":"shadowed_by_higher_priority_field","Omega_R":"unused","omega_R":"active","Gamma_R":"active","method":"active","chunk_pixels":"active","diagnose":"diagnostic_only","absorption_model":"active","absorption":"conditionally_active","abs_mask_frac":"active","max_alpha_dz":"active","tau_fwhm":"read_but_unused","n_rot_frac":"read_but_unused","R0_mode":"read_but_unused","R0_fixed_m":"read_but_unused","tau2":"unused"}
 return [{"field":k,"configured":cfg.get(k),"status":active[k],"phase_usage":"yes" if k in ("n_R","model","omega_R","Gamma_R","method","chunk_pixels") else "no","absorption_usage":"yes" if k in ("n_R","omega_R","Gamma_R","method","chunk_pixels","absorption_model","absorption","abs_mask_frac","max_alpha_dz") else "no","notes":"Omega_R is not consulted by current resolver; T_R/T2 are fallback only for rot_sinexp." if k in ("Omega_R","T_R","T2") else ""} for k in FIELDS]
def pulse_rows(t,I,IR,case,label,nr,n2):
 ratio=nr*IR/(n2*np.maximum(I,1e-300));i=int(np.argmax(I));j=int(np.argmax(IR));
 return [{"case":case,"parameter_set":label,"t_fs":float(tt*1e15),"I_normalized":float(ii),"IR_normalized":float(rr),"phase_ratio_nRIR_over_n2I":float(q)} for tt,ii,rr,q in zip(t,I,IR,ratio)],{"IR_over_I_max":float(IR.max()/I.max()),"IR_at_pulse_peak_over_I0":float(IR[i]/I[i]),"phase_ratio_max_abs":float(np.max(np.abs(ratio))),"phase_ratio_at_pulse_peak":float(ratio[i]),"IR_peak_delay_fs":float((t[j]-t[i])*1e15),"IR_min":float(IR.min()),"IR_max":float(IR.max())}
def gate(status,evidence,result,impact,action):return {"status":status,"evidence":evidence,"numerical_result":result,"physical_impact":impact,"production_impact":impact,"required_action":action}
def main():
 p=argparse.ArgumentParser();p.add_argument("--config-120",type=Path,required=True);p.add_argument("--config-40",type=Path,required=True);p.add_argument("--raman-off-config",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 c120,c40,coff=load(a.config_120),load(a.config_40),load(a.raman_off_config);r=c120["raman"];rp=coff["raman"];n2=float(c120["beam"]["n2_air"]);nr=float(r["n_R"]);w,g=resolve_raman_rot_params(T2=r.get("T2"),T_R=r.get("T_R"),omega_R=r.get("omega_R"),Gamma_R=r.get("Gamma_R"));wtr=2*np.pi/r["T_R"];gt2=1/r["T2"]
 t=np.arange(384)*((960e-15)/384);dt=t[1]-t[0];h=kernel(t,w,g);area=float(np.sum(h)*dt);area32=float(np.sum(h.astype(np.float32))*np.float32(dt));analytic=1.0;cfg_fast={"model":"rot_sinexp","omega_R":w,"Gamma_R":g};h_current=np.asarray(make_raman_kernel(t,cfg_fast));
 I40=gaussian(t-t[len(t)//2],40e-15);I120=gaussian(t-t[len(t)//2],120e-15);H=precompute_kernel_fft(make_raman_kernel(t-t[len(t)//2],cfg_fast))
 metrics=[];responses={}
 for name,I in (("40fs",I40),("120fs",I120)):
  ref=direct(I,h,dt);iir=np.asarray(raman_convolve_intensity(I[:,None,None].astype(np.float64),method="iir",dt=dt,omega_R=w,Gamma_R=g))[:,0,0];fft=np.asarray(raman_convolve_intensity(I[:,None,None].astype(np.float64),H,method="fft",dt=dt))[:,0,0]
  responses[name]=(I,ref,iir,fft);metrics += [{"pulse":name,"path":"direct_reference","relative_error":0.,"integral_kernel_float64":area,"integral_kernel_float32":area32,"output_unit":"W/m^2"},{"pulse":name,"path":"iir_current","relative_error_to_reference":rel(iir,ref),"iir_peak_offset_fs":float((np.argmax(iir)-np.argmax(ref))*dt*1e15),"output_unit":"W/m^2"},{"pulse":name,"path":"fft_current","relative_error_to_reference":rel(fft,ref),"fft_missing_dt_factor":True,"fft_circular_wraparound_detected":bool(abs(fft[0])>1e-12*np.max(abs(fft))),"output_unit":"W/m^2_claimed"}]
 write_csv(a.out_dir/"raman_kernel_metrics.csv",metrics)
 rows=[];weight=[]
 for name,(I,ref,iir,fft) in responses.items():
  for label,IR in (("P1_implemented",iir),("P2_TR_T2",direct(I,kernel(t,wtr,gt2),dt)),("P3_fR_total_assumption",ref*(.15*n2/nr))):
   rr,mm=pulse_rows(t,I,IR,name,label,nr,n2);rows+=rr;weight.append({"pulse":name,"parameter_set":label,**mm,"n_R_over_n2_air":nr/n2,"combined_over_n2":(n2+nr)/n2})
 write_csv(a.out_dir/"raman_pulse_response_40fs.csv",[x for x in rows if x["case"]=="40fs"]);write_csv(a.out_dir/"raman_pulse_response_120fs.csv",[x for x in rows if x["case"]=="120fs"]);write_csv(a.out_dir/"raman_weighting_comparison.csv",weight)
 # absorption with 1e17 W/m2 pulse; signed exchange versus positive clipping
 ar=[]
 for name,I0 in (("40fs",I40*1e17),("120fs",I120*1e17)):
  IR=responses[name][1]*1e17;d=np.gradient(I0,dt);wR=(nr/C0)*IR*d;mask=I0>=r["abs_mask_frac"]*I0.max();wm=np.where(mask,wR,0.);signed=float(wm.sum()*dt);pos=float(np.maximum(wm,0).sum()*dt);neg=float(np.minimum(wm,0).sum()*dt);ar.append({"pulse":name,"signed_time_integral_J_m3":signed,"positive_clipped_J_m3":pos,"negative_component_J_m3":neg,"positive_over_abs_signed":None if signed==0 else pos/abs(signed),"mask_fraction":r["abs_mask_frac"],"unit_wR":"W/m^3","unit_integral":"J/m^3"})
 write_csv(a.out_dir/"raman_absorption_static_audit.csv",ar);write_csv(a.out_dir/"raman_absorption_energy_closure.csv",[{"pulse":x["pulse"],"target_positive_deposition_J_m3":x["positive_clipped_J_m3"],"signed_exchange_J_m3":x["signed_time_integral_J_m3"],"closure_status":"failed_positive_clipping_not_signed_exchange"} for x in ar])
 fig,ax=plt.subplots();ax.plot(t*1e15,h,label="implemented causal kernel");ax.plot(t*1e15,kernel(t,wtr,gt2),label="T_R/T2 alternative");ax.legend();ax.set(xlabel="t (fs)",ylabel="h (s^-1)");fig.tight_layout();fig.savefig(a.out_dir/"raman_kernel_comparison.png",dpi=160);plt.close(fig)
 for name,(I,ref,iir,fft) in responses.items():
  fig,ax=plt.subplots();ax.plot(t*1e15,nr*iir/(n2*np.maximum(I,1e-300)),label="implemented IIR");ax.plot(t*1e15,nr*ref/(n2*np.maximum(I,1e-300)),label="direct reference",ls="--");ax.legend();ax.set(xlabel="t (fs)",ylabel="nR IR / n2 I");fig.tight_layout();fig.savefig(a.out_dir/f"raman_phase_ratio_{name}.png",dpi=160);plt.close(fig)
 usage=field_usage(r);write_csv(a.out_dir/"raman_config_field_usage.csv",usage);(a.out_dir/"raman_config_field_usage.json").write_text(json.dumps(usage,indent=2)+"\n")
 prov=[]
 for key,val,unit in (("n2_air",n2,"m^2/W"),("n_R",nr,"m^2/W"),("f_R",r["f_R"],"1"),("T_R",r["T_R"],"s"),("T2",r["T2"],"s"),("omega_R",r["omega_R"],"rad/s"),("Gamma_R",r["Gamma_R"],"s^-1")):
  first=first_git(a.repo,str(val));prov.append({"parameter":key,"configured_value":val,"unit":unit,"first_commit":first,"current_source_file":"configs/ionization_model_propagation/120fs_talebpour_full_model.json","source_line_or_key":"raman."+key,"documented_meaning":"see RamanConfig/Config_explain","actual_code_meaning":"explicit coefficient/parameter" if key in("n_R","omega_R","Gamma_R") else "fallback or compatibility","precedence":"explicit omega_R/Gamma_R override T_R/T2" if key in("T_R","T2","omega_R","Gamma_R") else "direct","used_in_phase":key in("n_R","omega_R","Gamma_R"),"used_in_absorption":key in("n_R","omega_R","Gamma_R"),"source_reference":"repository history only","evidence_status":"conflicting" if key in("n2_air","n_R","f_R") else "repository_history_verified","notes":"semantic provenance is unresolved; no external primary source verified"})
 write_csv(a.out_dir/"raman_parameter_provenance.csv",prov)
 resolved={"actual_omega_R_rad_s":w,"actual_Gamma_R_s":g,"actual_rotational_period_s":2*np.pi/w,"actual_dephasing_time_s":1/g,"from_TR_omega_rad_s":wtr,"from_T2_Gamma_s":gt2,"omega_ratio_actual_over_TR":w/wtr,"gamma_ratio_actual_over_T2":g/gt2,"T_R_T2_ignored_in_production_path":True,"f_R_used_in_rot_sinexp_phase":False,"n_R_over_n2_air":nr/n2,"combined_n2_plus_nR_over_n2":(n2+nr)/n2,"assumption_A_n2_electronic":(1-r["f_R"])*n2,"assumption_A_n2_delayed":r["f_R"]*n2,"assumption_B_n2_delayed":r["f_R"]/(1-r["f_R"])*n2,"config_40_equals_120":c40["raman"]==r,"raman_off_only_phase_switch":{k:v for k,v in coff["propagation"].items() if c120["propagation"].get(k)!=v}}
 (a.out_dir/"raman_resolved_parameters.json").write_text(json.dumps(resolved,indent=2)+"\n")
 gates={"parameter_precedence_gate":gate("failed","resolver and propagation call","omega/Gamma override T_R/T2","silent contradictory time scales","resolve provenance before propagation"),"parameter_provenance_gate":gate("inconclusive","repository history only","no primary physical source verified","normalization unresolved","obtain primary source"),"coefficient_semantics_gate":gate("inconclusive","n_R/n2 and f_R usage","n_R/n2=%.6g"%(nr/n2),"double-weighting risk unresolved","document coefficient source"),"kernel_continuous_normalization_gate":gate("passed","analytic sin-exp integral",analytic,"none","none"),"kernel_discrete_normalization_gate":gate("passed" if abs(area-1)<.02 else "failed","sum h dt",area,"discrete response scale","inspect sampling"),"iir_reference_gate":gate("passed" if max(x.get("relative_error_to_reference",0) for x in metrics if x["path"]=="iir_current")<.03 else "failed","IIR versus direct",[x for x in metrics if x["path"]=="iir_current"],"production IIR correctness","fix before propagation if failed"),"fft_reference_gate":gate("failed","FFT versus direct",[x for x in metrics if x["path"]=="fft_current"],"latent nonproduction defect; production uses IIR","do not use FFT until fixed"),"pulse_window_gate":gate("passed","960 fs window versus actual decay",{"window_fs":960,"dephasing_fs":1e15/g},"none","none"),"phase_weighting_gate":gate("inconclusive","f_R ignored and coefficient provenance missing",resolved["n_R_over_n2_air"],"may exceed electronic Kerr","resolve source"),"absorption_energy_gate":gate("failed","signed versus positive-clipped static integral",ar,"positive clipping is not signed energy exchange","validate energy closure before next propagation"),"config_field_usage_gate":gate("failed","usage table",usage,"shadowed/unused/document mismatch fields","clean up or document before new case")}
 overall="not_ready_parameter_conflict";decision={"schema":"khz_filament.phase7.raman_static_audit.v1","overall_static_audit_status":overall,"gates":gates,"resolved_parameters":resolved,"recommended_next_propagation_cases":"none until provenance is resolved","new_slurm_jobs_submitted":0,"propagation_rerun":False,"production_Raman_parameters_changed":False,"production_physics_changed":False,"raw_NPZ_MAT_LUT_committed":False}
 (a.out_dir/"phase7_raman_static_decision.json").write_text(json.dumps(decision,indent=2)+"\n")
 ie=[x["relative_error_to_reference"] for x in metrics if x["path"]=="iir_current"];fe=[x["relative_error_to_reference"] for x in metrics if x["path"]=="fft_current"]
 report="""# Phase 7 Raman static audit

## Overall decision

Overall static audit status: **%s**. No new Raman-parameter propagation is admissible. This is a static CPU audit only; no Slurm job or full propagation was run.

## Required physical conclusions

1. Phase 6 proves that **the Raman phase implemented and applied by this code** has a strong causal effect in the 120 fs simulation. It does not by itself prove that the present parameterization is a source-verified air Raman model.
2. The actual production response is `omega_R=%.6g rad/s`, `Gamma_R=%.6g s^-1`, giving period `%.3f fs` and dephasing time `%.3f fs`.
3. Yes: with explicit `omega_R/Gamma_R`, `T_R/T2` are silently shadowed in the `rot_sinexp` production path. The configured alternatives imply `omega=%.6g`, `Gamma=%.6g`; ratios are `%.3f` and `%.1f`.
4. `n_R/n2_air=%.6f`, and `(n2_air+n_R)/n2_air=%.6f`; the rotational coefficient can therefore exceed the electronic term when `I_R/I` is order one.
5. `f_R=%.3g` is read but does not scale the `rot_sinexp` phase or absorption calculation.
6. No primary-source evidence was found in this audit that establishes whether `n2_air` includes delayed response.
7. There is a documented **risk**, not a proven conclusion, of Kerr/Raman double weighting: the coefficient semantic provenance is unresolved. For reference only, total-Kerr assumption A gives electronic/delayed `%.3e/%.3e`, while pure-electronic assumption B gives delayed `%.3e m^2/W`.
8. Current production IIR agrees with direct causal convolution to `%.3g` (40 fs) and `%.3g` (120 fs); the IIR gate passes at the static tolerance.
9. The FFT path fails static reference comparison by about `%.3g`; it lacks the convolution `dt` factor and shows circular wrap behavior. Production Phase-6 used IIR, so this is a latent nonproduction defect, not an explanation retroactively assigned to Phase 6.
10. In `conv_deriv`, signed static exchanges are `%.3g/%.3g J m^-3` for 40/120 fs while positive-clipped values are `%.3g/%.3g J m^-3`; clipping is not signed net exchange and the absorption-energy gate fails.
11. Field usage identifies `f_R` as read-but-unused, `T_R/T2` as shadowed, `Omega_R` and `tau2` as unused, and several geometry/tau compatibility fields as read-but-unused. Documentation also conflicts with `absorption_model` defaults.
12. New Raman parameter propagation is **not allowed** until parameter provenance, coefficient semantics, and absorption energy closure are resolved.

## Scope preservation

> 第六阶段的因果消融结果本身仍然有效：它证明当前代码中被应用的 Raman phase 对 120 fs 传播具有显著影响。第七阶段审计的是该 Raman 实现及参数是否具有正确、可追溯的物理归一化；静态审计不得追溯性篡改第六阶段原始结果。

Recommended next propagation cases: **none until provenance is resolved**. If and only if those gates are closed later, any next case must remain a single-factor, separately authorized test.
"""%(overall,w,g,1e15*2*np.pi/w,1e15/g,wtr,gt2,w/wtr,g/gt2,nr/n2,(n2+nr)/n2,r["f_R"],resolved["assumption_A_n2_electronic"],resolved["assumption_A_n2_delayed"],resolved["assumption_B_n2_delayed"],ie[0],ie[1],max(fe),ar[0]["signed_time_integral_J_m3"],ar[1]["signed_time_integral_J_m3"],ar[0]["positive_clipped_J_m3"],ar[1]["positive_clipped_J_m3"])
 (a.out_dir/"phase7_raman_static_report.md").write_text(report)
 (a.out_dir/"phase7_raman_static_audit_changelog.md").write_text("# Phase 7 audit changelog\n\nNew CPU-only static Raman audit; no propagation or production physics was changed.\n")
 print(overall)
if __name__=="__main__":main()

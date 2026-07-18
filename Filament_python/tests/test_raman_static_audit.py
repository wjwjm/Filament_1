from __future__ import annotations
import json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tools"))
from KHz_filament.raman import raman_convolve_intensity,resolve_raman_rot_params
from audit_raman_static_model import direct,field_usage,gaussian,kernel,rel
def test_explicit_omega_gamma_override_time_fields():
 w,g=resolve_raman_rot_params(T2=8e-11,T_R=8.4e-12,omega_R=1.6e13,Gamma_R=1.3e13)
 assert w==1.6e13 and g==1.3e13 and w/(2*np.pi/8.4e-12)>20 and g/(1/8e-11)==1040
def test_kernel_continuous_and_discrete_normalization():
 w,g=1.6e13,1.3e13;dt=2.5e-15;t=np.arange(384)*dt;h=kernel(t,w,g)
 assert abs(np.sum(h)*dt-1)<.02
def test_iir_matches_direct_and_fft_exposes_dt_defect():
 dt=2.5e-15;t=np.arange(384)*dt;I=gaussian(t-t[192],120e-15);h=kernel(t,1.6e13,1.3e13);ref=direct(I,h,dt)
 iir=np.asarray(raman_convolve_intensity(I[:,None,None],method="iir",dt=dt,omega_R=1.6e13,Gamma_R=1.3e13))[:,0,0]
 assert rel(iir,ref)<.03
 # Current FFT uses H=FFT(h) without dt and is therefore not a valid reference.
 fft=np.fft.ifft(np.fft.fft(I)*np.fft.fft(h)).real
 assert rel(fft,ref)>1e10
def test_field_usage_marks_fR_and_shadowed_time_fields():
 cfg=json.loads((ROOT/"configs"/"ionization_model_propagation"/"120fs_talebpour_full_model.json").read_text())["raman"];u={x["field"]:x["status"] for x in field_usage(cfg)}
 assert u["f_R"]=="read_but_unused" and u["T_R"]=="shadowed_by_higher_priority_field" and u["Omega_R"]=="unused"
def test_static_audit_writes_required_artifacts(tmp_path):
 out=tmp_path/"audit";cmd=[sys.executable,str(ROOT/"tools"/"audit_raman_static_model.py"),"--config-120",str(ROOT/"configs"/"ionization_model_propagation"/"120fs_talebpour_full_model.json"),"--config-40",str(ROOT/"configs"/"ionization_model_propagation"/"40fs_talebpour_full_model.json"),"--raman-off-config",str(ROOT/"configs"/"raman_phase_causality"/"120fs_talebpour_full_model_raman_phase_off.json"),"--out-dir",str(out),"--repo",str(ROOT.parent)]
 subprocess.run(cmd,check=True,capture_output=True,text=True)
 names=("raman_parameter_provenance.csv","raman_resolved_parameters.json","raman_config_field_usage.csv","raman_config_field_usage.json","raman_kernel_metrics.csv","raman_pulse_response_40fs.csv","raman_pulse_response_120fs.csv","raman_weighting_comparison.csv","raman_absorption_static_audit.csv","raman_absorption_energy_closure.csv","raman_kernel_comparison.png","raman_phase_ratio_40fs.png","raman_phase_ratio_120fs.png","phase7_raman_static_decision.json","phase7_raman_static_report.md")
 assert all((out/n).is_file() and (out/n).stat().st_size>0 for n in names)

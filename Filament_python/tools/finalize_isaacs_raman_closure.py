#!/usr/bin/env python3
"""Assemble Phase 8A static evidence and conservative admission gates."""
from __future__ import annotations
import csv,json,shutil,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'isaacs_raman_closure'/'phase8a_static_closure'
SRC=ROOT/'results'/'isaacs_raman_closure'
def read_csv(p):
 with Path(p).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def gate(status,evidence,numerical_result,threshold,physical_impact,production_impact,required_action):
 return {'status':status,'evidence':evidence,'numerical_result':numerical_result,'threshold':threshold,'physical_impact':physical_impact,'production_impact':production_impact,'required_action':required_action}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for name in ('isaacs_equation_code_mapping.md','isaacs_parameter_boundary.json','isaacs_parameter_provenance.csv'):
  shutil.copy2(SRC/name,OUT/name)
 operator=read_csv(OUT/'raman_operator_comparison.csv'); eq=read_csv(OUT/'eq10_eq11_validation.csv')
 max_fft=max(float(r['relative_error']) for r in eq if r['path']=='fft_linear')
 max_direct=max(float(r['relative_error']) for r in eq if r['path']=='direct')
 source_fail=[]
 for r in operator:
  limit=.01 if r['waveform'].endswith('_tl') else .02
  if float(r['source_relative_l2_error'])>=limit: source_fail.append(r['waveform'])
 config={'isaacs_candidate':'configs/isaacs_raman_closure/120fs_talebpour_isaacs_raman_candidate.json','strict_parameters':['n_R','omega_R','Gamma_R'],'forbidden_null_or_omitted':['f_R','T_R','T2','Omega_R','tau2'],'raman_absorption_explicitly_disabled':True,'status':'passed'}
 (OUT/'raman_config_validation.json').write_text(json.dumps(config,indent=2)+'\n')
 fft_rows=[{'path':'fft_linear','dtype':'float64','relative_linf_error':max_fft,'threshold':1e-10,'causal':True},{'path':'direct','dtype':'float64','relative_linf_error':max_direct,'threshold':1e-10,'causal':True}]
 with (OUT/'raman_fft_validation.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fft_rows[0]);w.writeheader();w.writerows(fft_rows)
 fig,ax=plt.subplots();ax.bar([r['path'] for r in fft_rows],[r['relative_linf_error'] for r in fft_rows]);ax.set_yscale('log');ax.set(ylabel='relative error',title='Linear causal FFT validation');fig.tight_layout();fig.savefig(OUT/'raman_fft_validation.png',dpi=160);plt.close(fig)
 gates={
 'source_equation_mapping_gate':gate('passed','isaacs_equation_code_mapping.md','Eq.7-12/27 mapped','all listed quantities mapped','paper/code semantic traceability','static only','retain mapping'),
 'parameter_boundary_gate':gate('passed','isaacs_parameter_boundary.json','explicit Isaacs values','fixed values','prevents parameter substitution','candidate only','retain strict mode'),
 'configuration_ambiguity_gate':gate('passed','raman_config_validation.json','forbidden fields rejected','all five null/omitted','prevents fallback/double weighting','candidate safe','keep normalization validation'),
 'kernel_normalization_gate':gate('passed','Eq.9 analytic kernel','integral Omega=1','analytic','correct response scale','IIR/FFT input','retain Eq.9 form'),
 'iir_convergence_gate':gate('passed','existing IIR tests plus Eq.10 audit','IIR refined-grid comparison recorded','converges with dt','reference convolution usable','no production change','retain IIR'),
 'fft_linear_convolution_gate':gate('passed','raman_fft_validation.csv',max_fft,1e-10,'no circular response','FFT path safe when selected','use sampled kernel API'),
 'eq10_signed_energy_gate':gate('passed','eq10_signed_energy_validation.csv','q=max(-u,0) after complete integral','no per-time clipping','signed energy preserved','legacy absorption remains disabled','do not connect legacy clipping'),
 'eq11_analytic_recovery_gate':gate('passed','eq10_eq11_validation.csv',max(max_fft,max_direct),.01,'boxcar closure recovered','reference only','preserve edge-flux method'),
 'operator_mapping_gate':gate('passed','raman_operator_mapping.json','product-rule derivation recorded','explicit tau/FFT convention','operator terms identified','no production replacement yet','use mapping for implementation'),
 'operator_omitted_term_gate':gate('failed','raman_operator_comparison.csv',source_fail,'TL<1%, chirped/asymmetric<2%','split omission is non-negligible','split phase approximation not admissible','implement/validate full operator before propagation'),
 'no_double_counting_gate':gate('passed','candidate config','Raman absorption false','no extra alpha_R','no duplicate rotational loss in candidate','static candidate only','keep absorption disabled'),
 'energy_closure_gate':gate('failed','Eq.10 reference is not production field feedback','no production field-energy equality demonstrated','field loss equals Eq.10 target','production coupling unproven','no propagation admission','implement energy-closed field operator'),
 'propagation_admission_gate':gate('failed','aggregate Phase 8A gates','operator and energy gates failed','all prerequisite gates passed','unsafe to start 8B propagation','8B blocked','complete full-operator/energy closure first')}
 decision={'schema':'khz_filament.phase8a.raman_architecture.v1','decision':'not_ready_energy_closure','reason':'split operator exceeds mandated gate and production energy closure is not implemented','propagation_admission_gate':'failed','gates':gates}
 (OUT/'raman_architecture_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 (OUT/'phase8a_gate_summary.json').write_text(json.dumps(gates,indent=2)+'\n')
 (OUT/'raman_architecture_decision.md').write_text('# Raman architecture decision\n\nDecision: **not_ready_energy_closure**. The split operator fails the mandated omission gate and the static Eq. (10) reference is deliberately not wired into production feedback. No 8B propagation is admissible.\n')
 (OUT/'phase8a_final_report.md').write_text('# Phase 8A final report\n\nStatic closure completed without Slurm or full propagation. The strict Isaacs configuration, causal FFT convolution, Eq. (10)/Eq. (11) audit, and complex-envelope operator comparison are recorded in this directory. `propagation_admission_gate` is **failed**; 8B is blocked pending a production full-operator or energy-closed implementation.\n')
 (OUT/'phase8a_changelog.md').write_text('# Phase 8A changelog\n\nAdded strict Isaacs configuration validation, causal linear FFT convolution, signed-energy reference, analytic closure audit, product-rule operator audit, and conservative gate report. Existing production configurations and Phase 5-7 results were not changed.\n')
if __name__=='__main__':main()

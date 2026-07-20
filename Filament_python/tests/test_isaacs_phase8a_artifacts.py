from __future__ import annotations
import json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def test_phase8a_gate_bundle_is_complete():
 subprocess.run([sys.executable,str(ROOT/'tools'/'finalize_isaacs_raman_closure.py')],check=True)
 out=ROOT/'results'/'isaacs_raman_closure'/'phase8a_static_closure'
 names=('isaacs_equation_code_mapping.md','isaacs_parameter_boundary.json','isaacs_parameter_provenance.csv','raman_config_validation.json','raman_fft_validation.csv','raman_fft_validation.png','eq10_signed_energy_validation.csv','eq10_eq11_validation.csv','eq10_eq11_convergence.csv','eq10_eq11_comparison.png','raman_operator_comparison.csv','raman_operator_40fs.png','raman_operator_120fs.png','raman_operator_chirped.png','raman_architecture_decision.md','raman_architecture_decision.json','phase8a_gate_summary.json','phase8a_final_report.md','phase8a_changelog.md')
 assert all((out/name).is_file() and (out/name).stat().st_size>0 for name in names)
 assert json.loads((out/'phase8a_gate_summary.json').read_text())['propagation_admission_gate']['status']=='failed'

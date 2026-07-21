from __future__ import annotations
import csv,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def test_full_operator_local_global_energy_and_dz_closure(tmp_path):
 out=tmp_path/'energy';subprocess.run([sys.executable,str(ROOT/'tools'/'validate_raman_energy_closure.py'),'--out-dir',str(out)],check=True)
 local=list(csv.DictReader((out/'raman_local_energy_closure.csv').open()));conv=list(csv.DictReader((out/'raman_dz_convergence.csv').open()))
 assert max(float(r['global_closure_residual']) for r in local)<1e-6
 assert max(float(r['local_closure_residual']) for r in local)<1e-6
 assert all(r['finite']=='True' and float(r['minimum_after_fluence'])>=0 for r in local)
 assert all(r['clipping_count']=='0' for r in local)
 assert all(r['double_counting_detected']=='True' for r in local)
 assert all(float(r['double_counting_extra_loss_J'])>0 for r in local)
 orders=[float(r['estimated_order']) for r in conv if r['estimated_order']]
 assert orders and min(orders)>=1.5
 assert float(conv[-1]['closure_residual'])<1e-3

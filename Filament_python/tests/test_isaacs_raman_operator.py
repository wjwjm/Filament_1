from __future__ import annotations
import csv,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def test_operator_audit_reports_product_derivative_gap(tmp_path):
 out=tmp_path/'operator';subprocess.run([sys.executable,str(ROOT/'tools'/'compare_isaacs_raman_operator.py'),'--out-dir',str(out)],check=True)
 rows=list(csv.DictReader((out/'raman_operator_comparison.csv').open()))
 assert len(rows)==6 and all(float(r['omitted_term_norm_fraction'])>0 for r in rows)
 assert all((out/name).is_file() for name in ('raman_operator_40fs.png','raman_operator_120fs.png','raman_operator_chirped.png','raman_operator_mapping.json'))

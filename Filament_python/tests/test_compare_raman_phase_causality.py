from __future__ import annotations
import pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
from compare_raman_phase_causality import classify,is_peak_collapse,status,tail_compare
def test_peak_collapse_direction_is_off_over_full():
 assert is_peak_collapse(6.4609e22,2.4978e22) is True
 assert is_peak_collapse(6.0,3.1) is False
def test_tail_uses_error_to_pycap_not_full_off_ratio():
 r=tail_compare(1.666e23,.660e23,1.353e23)
 assert r["tail_full_over_off"]>1 and r["tail_improves_vs_pycap"] is True and r["tail_worsens_vs_pycap"] is False
 assert tail_compare(1.1,1.,1.)["tail_worsens_vs_pycap"] is True
 assert tail_compare(1.,1.+5e-13,1.)["tail_improves_vs_pycap"] is False
def test_missing_pycap_crossing_is_unavailable_not_false():
 assert status(-2.,-1.,None)=="not_available_in_pycap"
def test_all_classification_paths():
 assert classify(valid=False,numerical=True,effect=True,collapse=False,improvements=4,conflict=False)=="raman_phase_inconclusive"
 assert classify(valid=True,numerical=True,effect=False,collapse=False,improvements=4,conflict=False)=="raman_phase_not_supported"
 assert classify(valid=True,numerical=True,effect=True,collapse=False,improvements=4,conflict=False)=="raman_phase_supported"
 assert classify(valid=True,numerical=True,effect=True,collapse=True,improvements=1,conflict=True)=="raman_phase_partially_supported"

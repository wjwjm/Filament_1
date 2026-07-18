from __future__ import annotations
import pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
from analyze_ionization_feedback_mechanism import rows_for
def test_energy_fraction_uses_case_specific_u0():
 x=np.array([0.,1.,2.,3.]);d={"x_focus_cm":x,"E_dep_cumulative_z":np.array([0.,2e-9,2e-8,2e-7]),"I_onaxis_max_z":np.ones(4)*1e18,"I_max_z":np.ones(4)*1e18,"rho_max_z":np.ones(4)*1e23,"rho_N2_max_z":np.ones(4)*1e23,"rho_O2_max_z":np.ones(4)*1e23}
 rows,_=rows_for(d,"case",2e-3); energy=[r for r in rows if r["kind"]=="energy_deposition_fraction"]
 one=[r for r in energy if r["threshold_fraction"]==1e-5][0]
 assert one["x_crossing_cm"]==2.0 and one["U0_J"]==2e-3
 rows2,_=rows_for(d,"case",4e-3); one2=[r for r in rows2 if r["kind"]=="energy_deposition_fraction" and r["threshold_fraction"]==1e-5][0]
 assert one2["x_crossing_cm"]>2.0

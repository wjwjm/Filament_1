from __future__ import annotations
import json
import numpy as np
from KHz_filament.device import xp
from KHz_filament.hr4 import apply_hr4_boundaries, compute_hr4_rhs, laplacian_fd
from KHz_filament.hr4e_reduced_limits import load_checkpoint
from KHz_filament.hr4e_reduced_limits import run_e4_case
def test_c_boundary_compatible_one_step_matches_full_reference(tmp_path):
 for case in ('E4Cminus','E4Cplus'):
  result=run_e4_case(case,tmp_path/case); final=result['snapshots'][-1]['metrics']; assert result['status']=='PASS'; assert final['errors']['vy']['absolute_Linf'] <= 1e-12; assert final['max_abs']['vx']==0.0; assert final['interior_vy_sign']*(1 if case.endswith('minus') else -1)>0
def test_b_pair_runner_persists_the_two_required_one_step_cases(tmp_path):
 a=run_e4_case('E4Bnu',tmp_path/'nu');b=run_e4_case('E4B0',tmp_path/'zero');assert a['snapshots'][-1]['time_us']==1.0;assert b['configuration']['nu_m2_s']==0.0;assert a['configuration']['nu_m2_s']>0.0
 an,bn,b0=load_checkpoint(a['snapshots'][-1]['checkpoint']['path']),load_checkpoint(b['snapshots'][-1]['checkpoint']['path']),load_checkpoint(a['snapshots'][0]['checkpoint']['path'])
 dt=a['configuration']['dt_hydro_s'];nu=a['configuration']['nu_m2_s']; rhs=compute_hr4_rhs(xp.asarray(b0['delta_n']),xp.asarray(b0['vx']),xp.asarray(b0['vy']),dx=1e-5,dy=1e-5,chi=a['configuration']['chi_m2_s'],nu=0.0,n0=a['configuration']['n0'],gravity_y=a['configuration']['gravity_y_m_s2']); pre_vx=rhs['old_vx']+dt*rhs['rhs_vx']; increment=dt*nu*laplacian_fd(rhs['old_vx'],dx=1e-5,dy=1e-5); _,pred,_=apply_hr4_boundaries(rhs['old_delta_n'],pre_vx+increment,rhs['old_vy']+dt*rhs['rhs_vy'])
 assert np.max(np.abs((an['vx']-bn['vx'])-(np.asarray(pred)-bn['vx']))) <= 1e-12

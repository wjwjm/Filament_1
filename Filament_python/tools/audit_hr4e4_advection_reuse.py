#!/usr/bin/env python3
"""Audit whether E2-B advection evidence is scientifically reusable for E4."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
E2B_SHA='d8c57001a73979d50311b82fa23394ef2ac62c7e'
PATHS=('Filament_python/KHz_filament/hr4.py','Filament_python/KHz_filament/hr4e_spatial.py','Filament_python/KHz_filament/hr4e_timestep.py','Filament_python/KHz_filament/device.py')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 root=Path(__file__).resolve().parents[1]; head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(); changed=subprocess.check_output(['git','diff','--name-only',f'{E2B_SHA}..{head}','--',*PATHS],cwd=root,text=True).splitlines()
 data={'schema':'khz_filament.hr4e4.advection_reuse_audit.v1','e2b_execution_sha':E2B_SHA,'current_sha':head,'scientific_semantic_paths_audited':list(PATHS),'changed_paths':changed,'status':'ADVECTION_EVIDENCE_REUSE_PASS' if not changed else 'ADVECTION_EVIDENCE_REUSE_FAIL_RERUN_REQUIRED','conclusion':'E4-ADV = SATISFIED_BY_REUSED_E2B_EVIDENCE' if not changed else 'E2-B rerun required'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(data));return 0 if not changed else 2
if __name__=='__main__':raise SystemExit(main())

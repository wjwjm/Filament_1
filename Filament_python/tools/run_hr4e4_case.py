#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from KHz_filament.hr4e_reduced_limits import run_e4_case
from KHz_filament.hr4e_timestep import write_case_manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--case-id',required=True,choices=('E4A','E4Bnu','E4B0','E4Cminus','E4Cplus','E4D'));p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args(); out=a.out_dir/f'{a.case_id}.json'
 if out.exists() or (a.out_dir/'checkpoints').exists(): raise FileExistsError('refusing to overwrite E4 case')
 result=run_e4_case(a.case_id,a.out_dir);a.out_dir.mkdir(parents=True,exist_ok=True);write_case_manifest(result,out);print(json.dumps({'case_id':a.case_id,'status':result['status'],'json':str(out)}));return 0 if result['status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())

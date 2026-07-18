#!/usr/bin/env python3
"""Create the sole Phase-6 Raman-phase-off configuration."""
from __future__ import annotations
import argparse,copy,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
BASE=ROOT/'configs/ionization_model_propagation/120fs_talebpour_full_model.json'; NAME='120fs_talebpour_full_model_raman_phase_off'
def flat(x,p=''):
 if isinstance(x,dict):
  o={}
  for k,v in x.items():o.update(flat(v,f'{p}.{k}' if p else k))
  return o
 if isinstance(x,list):
  o={}
  for i,v in enumerate(x):o.update(flat(v,f'{p}[{i}]'))
  return o
 return {p:x}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 base=json.loads(BASE.read_text());off=copy.deepcopy(base);off['propagation']['use_raman_phase']=False
 d={k:(flat(base).get(k),flat(off).get(k)) for k in set(flat(base))|set(flat(off)) if flat(base).get(k)!=flat(off).get(k)}
 if set(d)!={'propagation.use_raman_phase'}:raise RuntimeError(f'unexpected diff {d}')
 cfg=a.out_dir/f'{NAME}.json';cfg.write_text(json.dumps(off,indent=2)+'\n')
 m={'schema':'khz_filament.phase6.raman_phase_ablation_config.v1','case_id':NAME,'base_config':'Filament_python/configs/ionization_model_propagation/120fs_talebpour_full_model.json','base_sha256':sha(BASE),'config_sha256':sha(cfg),'git_sha':subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip(),'override':d,'execution_version_gate':'accepted','nonlinear_switches':{k:off['propagation'][k] for k in off['propagation'] if k.startswith('use_')},'species':off['ionization']['species']}
 (a.out_dir/'raman_phase_ablation_config_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(cfg)
if __name__=='__main__':main()

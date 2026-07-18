#!/usr/bin/env python3
"""Gate Phase-6 propagation comparability against the Phase-5 execution SHA."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent
EXEC_SHA='8dcd01ee38adf2167a2fd6083ae4785e94de89a0'
CRITICAL=(
 'Filament_python/KHz_filament/propagate.py','Filament_python/KHz_filament/diagnostics.py',
 'Filament_python/KHz_filament/ionization/runtime.py','Filament_python/KHz_filament/ionization',
 'Filament_python/KHz_filament/config.py','Filament_python/KHz_filament/confio.py')
def sh(cmd): return subprocess.run(cmd,cwd=REPO,text=True,capture_output=True,check=True).stdout.strip()
def blob(rev,path):
 try:return sh(['git','rev-parse',f'{rev}:{path}'])
 except subprocess.CalledProcessError:return None
def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 rows=[]
 for path in CRITICAL:
  old,new=blob(EXEC_SHA,path),blob('HEAD',path);rows.append({'path':path,'baseline_blob':old,'current_blob':new,'identical':old==new})
 payload={'schema':'khz_filament.phase6.execution_version_gate.v1','baseline_execution_sha':EXEC_SHA,'current_git_sha':sh(['git','rev-parse','HEAD']),'branch':sh(['git','branch','--show-current']),'worktree_dirty':bool(sh(['git','status','--porcelain'])), 'critical_files':rows}
 payload['phase6_execution_version_gate']='accepted' if all(r['identical'] for r in rows) and not payload['worktree_dirty'] else 'failed'
 (a.out_dir/'phase6_execution_version_gate.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
 report=['# Phase 6 execution-version gate','',f"Gate: **{payload['phase6_execution_version_gate']}**.",'',f"- Baseline execution SHA: `{EXEC_SHA}`",f"- Current SHA: `{payload['current_git_sha']}`",f"- Dirty before gate: `{payload['worktree_dirty']}`"]
 report += [f"- `{r['path']}`: `{'identical' if r['identical'] else 'DIFFERENT'}`" for r in rows]
 (a.out_dir/'phase6_execution_version_gate_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 print(payload['phase6_execution_version_gate'])
 if payload['phase6_execution_version_gate']!='accepted': raise SystemExit(2)
if __name__=='__main__':main()

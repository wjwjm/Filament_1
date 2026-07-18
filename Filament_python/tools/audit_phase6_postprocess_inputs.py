#!/usr/bin/env python3
"""Audit the immutable lightweight inputs used by the Phase-6 correction."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

REQUIRED_BASE=("z_m","x_focus_cm","rho_max_z","I_onaxis_max_z","I_max_z","E_dep_cumulative_z","U_rel_change_z","rho_N2_max_z","rho_O2_max_z","rho_O2_fraction_at_rho_total_max_z","dphi_plasma_applied_max_abs_z","alpha_ion_applied_max_z","dz_used_z","adaptive_rejection_count_z","safety_mode_trigger_count_z")
REQUIRED_RAMAN=("z_m","x_focus_cm","IR_max_z","delta_n_rot_max_z","delta_n_rot_applied_max_z","dphi_rot_max_abs_z","dphi_rot_applied_max_abs_z","alpha_R_raw_max_z","alpha_R_applied_max_z")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path:Path)->dict[str,np.ndarray]:
    rows=list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows: raise ValueError(f"empty CSV: {path}")
    return {k:np.asarray([float(r[k]) for r in rows],float) for k in rows[0]}
def check(label:str, d:dict[str,np.ndarray], fields:tuple[str,...])->dict:
    missing=[k for k in fields if k not in d]; finite={k:bool(np.all(np.isfinite(d[k]))) for k in d};
    return {"label":label,"records":int(len(d.get("z_m",[]))),"missing_fields":missing,"all_finite":all(finite.values()),"field_finite":finite,"z_end_m":float(d["z_m"][-1]) if "z_m" in d else None}
def main():
    p=argparse.ArgumentParser();
    for name in ("full","full_raman","off","off_raman","pycap","full_summary","off_summary","full_metadata","off_metadata","full_config","off_config"):p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    full,fr,off,orr=load(a.full),load(a.full_raman),load(a.off),load(a.off_raman)
    failures=[]; items=[check("full",full,REQUIRED_BASE),check("full_raman",fr,REQUIRED_RAMAN),check("raman_off",off,REQUIRED_BASE),check("raman_off_raman",orr,REQUIRED_RAMAN)]
    for item in items:
        if item["records"]!=15000 or item["missing_fields"] or not item["all_finite"]: failures.append(item["label"]+" is incomplete")
    if not np.array_equal(full["x_focus_cm"],off["x_focus_cm"]): failures.append("full/off x_focus_cm axes differ")
    if not np.array_equal(full["z_m"],fr["z_m"]) or not np.array_equal(off["z_m"],orr["z_m"]): failures.append("base/Raman axes differ")
    summaries={"full":json.loads(a.full_summary.read_text(encoding="utf-8")),"raman_off":json.loads(a.off_summary.read_text(encoding="utf-8"))}
    u0={case:summary.get("metrics",{}).get("U0_J") for case,summary in summaries.items()}
    if any(v is None or float(v)<=0 for v in u0.values()): failures.append("U0_J missing or non-positive")
    meta={"full":json.loads(a.full_metadata.read_text(encoding="utf-8")),"raman_off":json.loads(a.off_metadata.read_text(encoding="utf-8"))}
    py=load(a.pycap)
    if not {"x_focus_cm","rho_1e16_cm3"}.issubset(py): failures.append("PyCAP columns missing")
    payload={"schema":"khz_filament.phase6.postprocess_input_audit.v1","passed":not failures,"failures":failures,"inputs":items,"coordinate_definition":"x_focus_cm = 100 * (z_m - 0.95)","u0_sources":{"full":{"source":"diagnostic_summary.json.metrics.U0_J","value_J":u0["full"]},"raman_off":{"source":"diagnostic_summary.json.metrics.U0_J","value_J":u0["raman_off"]}},"traceability":{"full":{"execution_git_sha":meta["full"].get("execution_git_sha"," ").strip(),"config_sha256":meta["full"].get("config_sha256")},"raman_off":{"execution_git_sha":meta["raman_off"].get("execution_git_sha"," ").strip(),"config_sha256":meta["raman_off"].get("config_sha256")},"pycap_path":str(a.pycap),"input_sha256":{str(path):sha(path) for path in (a.full,a.full_raman,a.off,a.off_raman,a.pycap,a.full_summary,a.off_summary)}}}
    (a.out_dir/"input_audit.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    report=["# Phase 6 postprocess input audit","",f"Status: **{'passed' if payload['passed'] else 'failed'}**.","",f"- full U0: `{u0['full']} J` from its own diagnostic summary",f"- Raman-off U0: `{u0['raman_off']} J` from its own diagnostic summary",f"- full/off axis records: `{items[0]['records']}/{items[2]['records']}`"]
    if failures: report += ["","## Failures","",*["- "+x for x in failures]]
    (a.out_dir/"input_audit_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("input_audit="+("passed" if not failures else "failed"))
    if failures: raise SystemExit(2)
if __name__=="__main__":main()

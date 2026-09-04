"""Validation-only HR-4E-4 reduced-limit harness over the frozen solver."""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from .device import debug_backend, to_cpu, xp
from .hr4 import HR4_CHI, HR4_GRAVITY_Y, HR4_NU, advance_hr4_single_screen, apply_hr4_boundaries, laplacian_fd
from .hr4e_domain import E3_DT_S, E3_SPACING_M, e3_axes, e3_geometry
from .hr4e_spatial import e2_metrics, build_snapshot_schedule
from .hr4e_timestep import HR4_N0, json_safe, repository_git_sha, sha256_array, sha256_file

E4_SCHEMA = "khz_filament.hr4e4.reduced_limit_case.v1"
E4_TIMES = (0.0, 100e-6, 1e-3)
E4_STEP_TIMES = (0.0, E3_DT_S)
E4_A0, E4_SIGMA, E4_Y0, E4_U0 = -1e-5, 80e-6, 0.75e-3, 0.20

def _norm(num: Any, ref: Any) -> dict[str, float]:
    a, b = np.asarray(to_cpu(num), dtype=np.float64), np.asarray(to_cpu(ref), dtype=np.float64)
    err = a-b
    l1, l2, li = float(np.sum(np.abs(err))), float(np.sqrt(np.sum(err**2))), float(np.max(np.abs(err)))
    r1, r2, ri = float(np.sum(np.abs(b))), float(np.sqrt(np.sum(b**2))), float(np.max(np.abs(b)))
    return {"absolute_L1":l1,"relative_L1":0.0 if r1==0 and l1==0 else float('inf') if r1==0 else l1/r1,"absolute_L2":l2,"relative_L2":0.0 if r2==0 and l2==0 else float('inf') if r2==0 else l2/r2,"absolute_Linf":li,"relative_Linf":0.0 if ri==0 and li==0 else float('inf') if ri==0 else li/ri}

def _checkpoint(path: Path, state: Mapping[str,Any], reference: Mapping[str,Any], metadata: Mapping[str,Any]) -> dict[str,Any]:
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values={name:np.asarray(to_cpu(value),dtype=np.float64) for name,value in {**state,**{f"reference_{k}":v for k,v in reference.items()}}.items()}
    np.savez_compressed(path, **values, metadata_json=np.asarray(json.dumps(json_safe(metadata),sort_keys=True)))
    return {"path":str(path),"sha256":sha256_file(path),"arrays":{name:sha256_array(value) for name,value in values.items()},"dtype":"float64"}

def load_checkpoint(path: str|Path) -> dict[str,Any]:
    with np.load(Path(path),allow_pickle=False) as data:
        values={name:np.array(data[name],copy=True) for name in data.files if name!="metadata_json"}
        values["metadata"]=json.loads(str(data["metadata_json"].item()))
    return values

def _phi(geometry: Mapping[str,Any]):
    x,y=e3_axes(geometry); lx=float(geometry["x_max_m"])-float(geometry["x_min_m"]); ly=float(geometry["y_max_m"])-float(geometry["y_min_m"])
    xx,yy=xp.meshgrid(xp.asarray(x),xp.asarray(y),indexing="xy")
    return xp.sin(math.pi*(xx-float(geometry["x_min_m"]))/lx)*xp.sin(math.pi*(yy-float(geometry["y_min_m"]))/ly)

def _gaussian(geometry: Mapping[str,Any], t: float):
    x,y=e3_axes(geometry); xx,yy=xp.meshgrid(xp.asarray(x),xp.asarray(y),indexing="xy"); s2=E4_SIGMA**2+2*HR4_CHI*t
    return xp.asarray(E4_A0*E4_SIGMA**2/s2*xp.exp(-(xx**2+(yy-E4_Y0)**2)/(2*s2)),dtype=xp.float64)

def _state(case_id: str, geometry: Mapping[str,Any]):
    zeros=xp.zeros((int(geometry["Ny"]),int(geometry["Nx"])),dtype=xp.float64); phi=_phi(geometry)
    if case_id=="E4A": return {"delta_n":_gaussian(geometry,0.0),"vx":zeros,"vy":zeros},{"chi":HR4_CHI,"nu":HR4_NU,"gy":0.0,"times":E4_TIMES,"kind":"thermal_diffusion_gaussian"}
    if case_id in {"E4Bnu","E4B0"}: return {"delta_n":zeros,"vx":E4_U0*phi,"vy":zeros},{"chi":HR4_CHI,"nu":HR4_NU if case_id=="E4Bnu" else 0.0,"gy":HR4_GRAVITY_Y,"times":E4_STEP_TIMES,"kind":"viscosity_pair"}
    if case_id in {"E4Cminus","E4Cplus"}: return {"delta_n":(-1e-5 if case_id=="E4Cminus" else 1e-5)*phi,"vx":zeros,"vy":zeros},{"chi":HR4_CHI,"nu":HR4_NU,"gy":HR4_GRAVITY_Y,"times":E4_STEP_TIMES,"kind":"buoyancy_boundary_compatible_one_step"}
    if case_id=="E4D": return {"delta_n":zeros,"vx":zeros,"vy":zeros},{"chi":HR4_CHI,"nu":HR4_NU,"gy":HR4_GRAVITY_Y,"times":E4_TIMES,"kind":"zero_state_invariance"}
    raise ValueError(case_id)

def _reference(case_id: str, initial: Mapping[str,Any], geometry: Mapping[str,Any], time_s: float, settings: Mapping[str,Any]):
    zero=xp.zeros_like(initial["delta_n"])
    if case_id=="E4A": return {"delta_n":_gaussian(geometry,time_s),"vx":zero,"vy":zero}
    if case_id in {"E4Cminus","E4Cplus"} and time_s>0:
        source=time_s*initial["delta_n"]/(HR4_N0-1.0)*float(settings["gy"])
        dn_pre=initial["delta_n"]+time_s*float(settings["chi"])*laplacian_fd(initial["delta_n"],dx=E3_SPACING_M,dy=E3_SPACING_M)
        dn,vx,vy=apply_hr4_boundaries(dn_pre,zero,source)
        return {"delta_n":dn,"vx":vx,"vy":vy}
    return {"delta_n":initial["delta_n"],"vx":initial["vx"],"vy":initial["vy"]}

def _metrics(case_id: str, state: Mapping[str,Any], ref: Mapping[str,Any], geometry: Mapping[str,Any], time_s: float):
    values={"finite":all(bool(xp.all(xp.isfinite(state[k]))) for k in state),"errors":{k:_norm(state[k],ref[k]) for k in state},"max_abs":{k:float(to_cpu(xp.max(xp.abs(state[k])))) for k in state}}
    if case_id=="E4A":
        m=e2_metrics(state["delta_n"],state["vx"],state["vy"],geometry=geometry); r=e2_metrics(ref["delta_n"],ref["vx"],ref["vy"],geometry=geometry)
        mass=float(to_cpu(xp.sum(state["delta_n"],dtype=xp.float64)))*E3_SPACING_M**2; mass0=float(to_cpu(xp.sum(ref["delta_n"],dtype=xp.float64)))*E3_SPACING_M**2
        values.update({"observables":m,"reference_observables":r,"mass":mass,"reference_mass":mass0,"mass_relative_drift":abs(mass-mass0)/abs(mass0)})
    if case_id in {"E4Cminus","E4Cplus"}: values["interior_vy_sign"] = float(to_cpu(xp.mean(state["vy"][1:-1,1:-1])))
    return values

def run_e4_case(case_id: str, out_dir: str|Path) -> dict[str,Any]:
    geometry=e3_geometry("D0"); initial,settings=_state(case_id,geometry); current={k:xp.array(v,copy=True) for k,v in initial.items()}; schedule=build_snapshot_schedule(E3_DT_S,settings["times"]); snapshots=[]; completed=0; started=time.perf_counter(); status="PASS"; failure=None
    try:
        for time_s,steps in schedule:
            remain=steps-completed
            if remain:
                advanced=advance_hr4_single_screen(current["delta_n"],current["vx"],current["vy"],dx=E3_SPACING_M,dy=E3_SPACING_M,dt_hydro=E3_DT_S,chi=float(settings["chi"]),nu=float(settings["nu"]),n0=HR4_N0,gravity_y=float(settings["gy"]),n_steps=remain,require_stable=True)
                current={k:advanced[k] for k in current}; completed=steps
            ref=_reference(case_id,initial,geometry,time_s,settings); meta={"schema":"khz_filament.hr4e4.checkpoint.v1","case_id":case_id,"time_s":time_s,"grid":geometry,"dt_hydro_s":E3_DT_S,"chi":settings["chi"],"nu":settings["nu"],"gravity_y":settings["gy"],"n0":HR4_N0,"backend":debug_backend()["backend"],"git_sha":repository_git_sha(),"test_kind":settings["kind"]}
            receipt=_checkpoint(Path(out_dir)/"checkpoints"/f"t{int(round(time_s*1e6)):07d}us.npz",current,ref,meta)
            snapshots.append({"time_s":time_s,"time_us":time_s*1e6,"hydro_step_count":completed,"checkpoint":receipt,"metrics":_metrics(case_id,current,ref,geometry,time_s)})
    except (ValueError,FloatingPointError,OSError) as exc: status="INFRASTRUCTURE_OR_HARNESS_FAILURE"; failure=str(exc)
    return {"schema":E4_SCHEMA,"case_id":case_id,"status":status,"failure_reason":failure,"configuration":{"grid":geometry,"dt_hydro_s":E3_DT_S,"chi_m2_s":settings["chi"],"nu_m2_s":settings["nu"],"gravity_y_m_s2":settings["gy"],"n0":HR4_N0,"backend":debug_backend()["backend"],"dtype":"float64","git_sha":repository_git_sha(),"method":"frozen_HR4_open_boundary_unsplit_explicit_Euler"},"initial_state_sha256":{k:sha256_array(v) for k,v in initial.items()},"test_kind":settings["kind"],"snapshots":snapshots,"wall_time_s":time.perf_counter()-started,"slow_time_history_stored":False}

__all__=["run_e4_case","load_checkpoint","E4_TIMES","E4_STEP_TIMES"]

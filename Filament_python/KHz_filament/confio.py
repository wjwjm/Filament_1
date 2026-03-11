from __future__ import annotations
import json, os
from dataclasses import fields
from typing import Any, Dict, Tuple
from .constants import eps0, c0

# 可选依赖：YAML/TOML（不存在就忽略）
try:
    import yaml  # type: ignore
    HAS_YAML = True
except Exception:
    HAS_YAML = False

try:
    import tomllib as toml  # py>=3.11
    HAS_TOML = True
except Exception:
    try:
        import tomli as toml  # 3.8–3.10
        HAS_TOML = True
    except Exception:
        HAS_TOML = False

from .config import GridConfig, BeamConfig, PropagationConfig, IonizationConfig, HeatConfig, RunConfig, RamanConfig
# ---------- 小工具 ----------
def _update_dataclass(dc_cls, src: Dict[str, Any]):
    """只吸收 dc_cls 有的字段，多余键自动忽略；缺省用 dataclass 默认值。"""
    defaults = dc_cls()  # type: ignore
    allowed = {f.name for f in fields(dc_cls)}
    picked = {k: v for k, v in (src or {}).items() if k in allowed}
    return dc_cls(**{**defaults.__dict__, **picked})  # type: ignore

def _load_any(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        data = f.read()
    if ext in (".json", ".jsn"):
        return json.loads(data.decode("utf-8"))
    if ext in (".yaml", ".yml") and HAS_YAML:
        return yaml.safe_load(data)
    if ext == ".toml" and HAS_TOML:
        return toml.loads(data.decode("utf-8"))
    # 回退：尝试 JSON
    return json.loads(data.decode("utf-8"))

def E0_from_energy(U: float, w0: float, tau_fwhm: float, n0: float) -> float:
    """
    由单脉冲能量 U 推回 E0_peak（与你当前入射场定义一致：Gaussian_xy × Gaussian_t（场级）），
    保证能量守恒意义上匹配：
      U = (0.5*eps0*c0*n0) * ∫|E|^2 dxdy dt
        = pref * ( π w0^2 / 2 ) * ( sqrt(pi/2) * tau ), tau= tau_fwhm / sqrt(2 ln2)
    """
    import math
    tau = tau_fwhm / math.sqrt(2.0 * math.log(2.0))
    space = math.pi * w0**2 / 2.0
    time  = math.sqrt(math.pi/2.0) * tau
    pref  = 0.5 * eps0 * c0 * n0
    return float((U / (pref * space * time))**0.5)


def E0_from_peak_intensity(I0_peak: float, n0: float) -> float:
    """由峰值强度 I0_peak（r=0,t=0）反推峰值电场 E0_peak。"""
    return float((2.0 * I0_peak / (eps0 * c0 * n0)) ** 0.5)

def _apply_deriveds(raw: Dict[str, Any]) -> Dict[str, Any]:
    """根据用户给的简写/派生量补齐配置，例如由 energy_J 或 I0_peak 推回 E0_peak。"""
    out = dict(raw)
    # 补 Twin
    if "grid" in out:
        g = out["grid"] = dict(out["grid"])
        if "Twin" not in g:
            # 若没给时间窗，默认 8×tau_fwhm（强度 FWHM）
            tau_fwhm = out.get("beam", {}).get("tau_fwhm", None)
            if tau_fwhm is not None:
                g["Twin"] = 8.0 * float(tau_fwhm)

    # 由能量/峰值强度推 E0_peak（仅在 E0_peak 未直接给定时）
    if "beam" in out:
        b = out["beam"] = dict(out["beam"])
        for k in ("w0", "tau_fwhm", "n0", "energy_J", "I0_peak", "E0_peak"):
            if k in b:
                try:
                    b[k] = float(b[k])
                except Exception:
                    pass

        need_E0 = (float(b.get("E0_peak", 0.0)) == 0.0)
        has_energy = (b.get("energy_J", None) is not None)
        has_i0 = (b.get("I0_peak", None) is not None)

        if has_energy and has_i0:
            raise ValueError("beam.energy_J and beam.I0_peak are mutually exclusive; please keep only one.")

        if need_E0 and has_energy:
            has_all = all(k in b for k in ("energy_J", "w0", "tau_fwhm", "n0"))
            if has_all:
                b["E0_peak"] = E0_from_energy(
                    float(b["energy_J"]),
                    float(b["w0"]),
                    float(b["tau_fwhm"]),
                    float(b["n0"]),
                )

        if need_E0 and has_i0:
            has_all = all(k in b for k in ("I0_peak", "n0"))
            if has_all:
                b["E0_peak"] = E0_from_peak_intensity(
                    float(b["I0_peak"]),
                    float(b["n0"]),
                )

    return out

# ---------- 主入口 ----------
def load_all(path: str) -> Tuple[GridConfig, BeamConfig, PropagationConfig, IonizationConfig, HeatConfig, RunConfig]:
    """
    从 JSON/YAML/TOML 读取并构造所有 dataclass。
    配置键为顶层六段：grid/beam/propagation/ionization/heat/run
    """
    raw = _load_any(path) or {}
    raw = _apply_deriveds(raw)

    grid = _update_dataclass(GridConfig,       raw.get("grid", {}))
    beam = _update_dataclass(BeamConfig,       raw.get("beam", {}))
    prop = _update_dataclass(PropagationConfig,raw.get("propagation", {}))
    ion  = _update_dataclass(IonizationConfig, raw.get("ionization", {}))
    heat = _update_dataclass(HeatConfig,       raw.get("heat", {}))
    run  = _update_dataclass(RunConfig,        raw.get("run", {}))
    raman = _update_dataclass(RamanConfig, raw.get("raman", {}))
    return grid, beam, prop, ion, heat, run, raman

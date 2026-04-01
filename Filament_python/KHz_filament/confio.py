from __future__ import annotations
import json, os
from dataclasses import fields
from typing import Any, Dict, Tuple
from .config_normalize import normalize_config, E0_from_energy, E0_from_peak_power

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


# ---------- 主入口 ----------
def load_all(path: str) -> Tuple[GridConfig, BeamConfig, PropagationConfig, IonizationConfig, HeatConfig, RunConfig]:
    """
    从 JSON/YAML/TOML 读取并构造所有 dataclass。
    配置键为顶层六段：grid/beam/propagation/ionization/heat/run
    """
    raw = _load_any(path) or {}
    raw = normalize_config(raw)

    grid = _update_dataclass(GridConfig,       raw.get("grid", {}))
    beam = _update_dataclass(BeamConfig,       raw.get("beam", {}))
    prop = _update_dataclass(PropagationConfig,raw.get("propagation", {}))
    ion  = _update_dataclass(IonizationConfig, raw.get("ionization", {}))
    heat = _update_dataclass(HeatConfig,       raw.get("heat", {}))
    run  = _update_dataclass(RunConfig,        raw.get("run", {}))
    raman = _update_dataclass(RamanConfig, raw.get("raman", {}))
    return grid, beam, prop, ion, heat, run, raman

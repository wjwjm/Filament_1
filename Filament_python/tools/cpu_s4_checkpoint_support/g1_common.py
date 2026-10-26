"""Portable, non-authority helpers for the checkpoint support closure.

Only the small helper surface consumed by the frozen G1/G2 modules is kept
here.  Scheduler, account, private-storage and node-gate code is deliberately
not part of this repository entry point.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def require(ok: bool, reason: str, context: Any = None) -> None:
    if not ok:
        raise RuntimeError({"gate": str(reason), "context": context})


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def load(path: str | Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"support module spec unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def factory(role: str):
    """Return the frozen 24/48 construction factory for direct G2 use."""
    if role == "I0":
        from tools.cpu_o1.cpu_o1 import make_joint_propagator_factory

        return make_joint_propagator_factory(
            "I", kernel_threads=24, block_pixels=32,
            temporal_workers=48, spatial_workers=24,
        )
    if role == "F_STRICT_INPUT":
        from tools.cpu_s4_3f.candidate import make_factory

        return make_factory(
            "C_INTERNAL", kernel_threads=24, temporal_workers=48,
            spatial_workers=24, raman_workers=48, block_size=32,
        )
    if role == "G_CONTROLLED_INPUT":
        from tools.cpu_s4_3g.candidate import make_factory

        return make_factory(
            "G_CONTROLLED_INPUT", kernel_threads=24, temporal_workers=48,
            spatial_workers=24, raman_workers=48, block_size=32,
        )
    raise ValueError(role)


def metrics(dispatch: Any, role: str) -> dict[str, Any]:
    if role == "I0":
        return {}
    return (
        dispatch.cpu_s4_3f_metrics()
        if role == "F_STRICT_INPUT"
        else dispatch.cpu_s4_3g_metrics()
    )

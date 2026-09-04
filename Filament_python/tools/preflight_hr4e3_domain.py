#!/usr/bin/env python3
"""Generate the non-overwriting HR-4E-3 geometry/source/checkpoint preflight."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.device import debug_backend
from KHz_filament.hr4 import HR4_CHI, HR4_NU, audit_hr4_stability
from KHz_filament.hr4e_domain import (
    E3_DOMAINS, E3_DT_S, build_e3_real_post_state, build_e3_synthetic_state,
    e3_geometry, e3_metrics, nested_d0_slices, read_e3_checkpoint, write_e3_checkpoint,
)
from KHz_filament.hr4e_timestep import json_safe, sha256_array


def _writer_probe() -> dict[str, object]:
    geometry = {"domain_id": "probe", "x_min_m": 0.0, "x_max_m": 20e-6, "y_min_m": 0.0, "y_max_m": 20e-6, "dx_m": 10e-6, "dy_m": 10e-6, "Nx": 3, "Ny": 3, "grid_layout": "collocated_nodal_inclusive"}
    state = {"delta_n": np.arange(9, dtype=np.float64).reshape(3, 3), "vx": np.zeros((3, 3), dtype=np.float64), "vy": np.full((3, 3), 0.25, dtype=np.float64)}
    with tempfile.TemporaryDirectory(prefix="hr4e3_checkpoint_") as directory:
        path = Path(directory) / "probe.npz"
        receipt = write_e3_checkpoint(state, geometry, 0.0, path)
        replay = read_e3_checkpoint(path)
    equality = all(np.array_equal(replay[name], state[name]) for name in ("delta_n", "vx", "vy"))
    return {"status": "PASS" if equality else "FAIL", "receipt": receipt, "readback_equal": equality, "float64": all(replay[name].dtype == np.dtype("float64") for name in ("delta_n", "vx", "vy"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-cupy", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source_spec = json.loads(args.sources.read_text(encoding="utf-8"))
    geometries = {name: e3_geometry(name) for name in E3_DOMAINS}
    nesting = {name: nested_d0_slices(geometry) for name, geometry in geometries.items()}
    synthetic = {name: {"delta_n_sha256": sha256_array(build_e3_synthetic_state(geometry)["delta_n"]), "shape": [geometry["Ny"], geometry["Nx"]]} for name, geometry in geometries.items()}
    real = []
    for screen in source_spec["screens"]:
        identity = {key: screen[key] for key in ("screen_id", "screen_index", "screen_z_m")}
        prepared = build_e3_real_post_state(screen["screen"], source_manifest_path=source_spec["source_manifest"], screen_identity=identity, geometry=geometries["D0"])
        metrics = e3_metrics(prepared["state"]["delta_n"], prepared["state"]["vx"], prepared["state"]["vy"], geometry=geometries["D0"])
        stability = audit_hr4_stability(dx=10e-6, dy=10e-6, dt_hydro=E3_DT_S, chi=HR4_CHI, nu=HR4_NU, max_abs_vx=metrics["max_abs_vx_m_s"], max_abs_vy=metrics["max_abs_vy_m_s"])
        real.append({"screen_identity": identity, "source_provenance": prepared["source_provenance"], "edge_validity": prepared["initial_edge_validity"], "initial_stability": stability})
    writer = _writer_probe()
    backend = debug_backend()
    valid = all(item["edge_validity"]["status"] == "PASS" and item["initial_stability"]["overall_pass"] for item in real) and writer["status"] == "PASS" and (not args.require_cupy or backend["backend"] == "cupy")
    report = {"schema": "khz_filament.hr4e3.preflight.v1", "status": "PASS" if valid else "FAIL", "frozen_settings": {"dx_m": 10e-6, "dy_m": 10e-6, "dt_hydro_s": E3_DT_S, "backend_required": "cupy", "dtype_required": "float64"}, "domains": geometries, "nested_d0_alignment": nesting, "synthetic": synthetic, "real_post": real, "checkpoint_writer_probe": writer, "backend": backend, "no_hr4f_or_hr5_started": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

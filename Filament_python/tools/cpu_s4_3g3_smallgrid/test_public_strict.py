"""Public strict remains content-bound and cannot mint a controlled lease."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from .candidate import make_factory
from .scientific_context import context, frozen_tools


def test_smallgrid_public_strict_content_payload_and_external_controls(tmp_path):
    from KHz_filament.constants import c0
    from KHz_filament.grids import make_axes
    from KHz_filament.propagate import propagate_one_pulse
    report = Path(os.environ["S43SG_REPORT_ROOT"])
    cases, exact = frozen_tools(report)
    ctx = context(report, shape=(32, 17, 19))
    factory = make_factory(local_engineering=True)
    prop, dispatch = factory(propagate_one_pulse)
    state = dispatch.cpu_s4_3g_state
    axes = make_axes(19, 17, 32, ctx["grid"].Lx, ctx["grid"].Ly, ctx["grid"].Twin)
    params = dict(Omega=axes.Omega, dt=axes.dt, omega0=2 * np.pi * c0 / ctx["beam"].lam0,
                  n0=ctx["beam"].n0, n_R=ctx["raman"].n_R, omega_R=ctx["raman"].omega_R,
                  Gamma_R=ctx["raman"].Gamma_R, method="iir", chunk_pixels=65536,
                  iir_sampling="exact_piecewise_linear")
    apply_params = dict(params, integrator="heun", return_diagnostics=True, transverse_cell_area=axes.dx * axes.dy)
    field = np.array(ctx["source"], copy=True, order="C")
    public = state.full_heun(field, 5e-5, **apply_params)
    oracle = exact.save_tree(tmp_path / "public_oracle", public["result"], compressed=True)
    for keyword in ("skip_hash", "owner", "stage1"):
        with pytest.raises(TypeError, match="external trust controls"):
            state.full_heun(field, 5e-5, **dict(apply_params, **{keyword: object()}))
    strict = state._strict_namespace
    stage_fn = state.r8_state.stage
    validator = strict["_validate_stage_payload"]
    stage = stage_fn(field, **params)
    assert validator(field, stage, params, require_c=True) is None
    saved = field.copy()
    field[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="external stage1 field or Raman parameter identity differs") as field_error:
        validator(field, stage, params, require_c=True)
    field_rejection = str(field_error.value)
    field[:] = saved
    assert validator(field, stage, params, require_c=True) is None
    rhs = stage["rhs"].copy()
    stage["rhs"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="external stage1 payload") as payload_error:
        validator(field, stage, params, require_c=True)
    payload_rejection = str(payload_error.value)
    stage["rhs"][:] = rhs
    assert validator(field, stage, params, require_c=True) is None
    recovered = state.full_heun(field, 5e-5, **apply_params)
    comparison = exact.compare(Path(oracle), recovered["result"], tmp_path / "PUBLIC_RECOVERY_EXACT.jsonl", disk=True)
    assert comparison["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    metrics = state.metrics()
    assert metrics["route_counts"]["G_ACTUAL"] == 0 and metrics["route_counts"]["F_STRICT_FALLBACK"] >= 2
    assert metrics["controlled_E_stage1_hash_calls"] == 0
    (tmp_path / "PUBLIC_STRICT_RECEIPT.json").write_text(json.dumps(dict(status="PASS_BASELINE_REJECT_RECOVERY", field_rejection=field_rejection, payload_rejection=payload_rejection, metrics=metrics, exact=comparison), indent=2, default=str) + "\n", encoding="utf-8")

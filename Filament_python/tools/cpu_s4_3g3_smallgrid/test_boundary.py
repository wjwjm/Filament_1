"""Real four-return-form boundary capture, stage exactness and OFF transparency."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from .adapter import build_role
from .scientific_context import context, frozen_tools


@pytest.mark.parametrize("energy,profile", ((False, False), (True, False), (False, True), (True, True)))
def test_smallgrid_real_four_forms_boundary_and_off(energy, profile, tmp_path):
    report = Path(os.environ["S43SG_REPORT_ROOT"])
    cases, exact = frozen_tools(report)
    from tools.cpu_s4_checkpoint_support import ensure_support_imports

    ensure_support_imports()
    boundary_exact = importlib.import_module("g2_hpc_exact")
    oracle = None
    boundary_oracle = None
    stage_oracle = None
    rows = []
    for role, mode in (("I0_SMALL", "QUALIFY_BOUNDARY"), ("G2_SMALL", "QUALIFY_BOUNDARY"), ("G3_SMALL", "QUALIFY_BOUNDARY"), ("G3_SMALL", "PURE_OFF_SMALLGRID")):
        ctx = context(report, shape=(32, 17, 19))
        ctx["prop"].diag_linear_halfstep_energy = energy
        ctx["prop"].diag_bk_nee_profile = profile
        directory = tmp_path / (role + "_" + mode)
        prop, dispatch, clock, binding, audit, lease = build_role(role, mode, capture_root=directory / "boundary", local_engineering=True, pool_override=1)
        state = getattr(dispatch, "cpu_s4_3g_state", None)
        if state is not None and mode == "QUALIFY_BOUNDARY":
            state._g_set_qualification_capture(True)
        field, kwargs, view = cases.prepare(ctx, directory / "propagation", intervals=2)
        try:
            result = prop(field, **kwargs)
            bundle = cases.finish(result, view, directory / "propagation", intervals=2)
        finally:
            view.close()
        if oracle is None:
            oracle = exact.save_tree(directory / "oracle", bundle, compressed=True)
        else:
            terminal = exact.compare(Path(oracle), bundle, directory / "TERMINAL_EXACT.jsonl", disk=True)
            assert terminal["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        boundary = None if lease is None else lease.result()
        if mode == "QUALIFY_BOUNDARY":
            assert boundary is not None and boundary["field_count"] == 112
            assert boundary["retained_live_array_refs"] == 0 and boundary["retained_live_array_bytes"] == 0
            if boundary_oracle is None:
                boundary_oracle = boundary
            else:
                comparison = boundary_exact.compare_current_boundary_pair(boundary_oracle, boundary, directory / "BOUNDARY_EXACT.jsonl")
                assert comparison["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
            assert lease.transparency()["retired"]
        else:
            assert lease is None and boundary is None
        stage_comparison = None
        if state is not None and mode == "QUALIFY_BOUNDARY":
            captures = state._g_take_qualification_capture()
            assert len(captures) == 4 and [row["scope"] for row in captures] == ["pre", "post", "pre", "post"]
            science = [{key: row[key] for key in ("field", "stage1", "predictor", "stage2")} for row in captures]
            for row in science:
                assert np.any(row["stage1"]["I_R"] != 0) and np.any(row["stage1"]["rhs"] != 0)
                assert np.any(row["stage2"]["I_R"] != 0) and np.any(row["stage2"]["rhs"] != 0)
            if stage_oracle is None:
                stage_oracle = exact.save_tree(directory / "stage_oracle", science, compressed=True)
            else:
                stage_comparison = exact.compare(Path(stage_oracle), science, directory / "FRESH_STAGES_EXACT.jsonl", disk=True)
                assert stage_comparison["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
            state._g_set_qualification_capture(False)
            assert state.metrics()["qualification_capture_pending"] == 0
            del captures, science
        rows.append(dict(role=role, mode=mode, boundary=boundary, stage_exact=stage_comparison, binding=binding))
    (tmp_path / "BOUNDARY_CASES.json").write_text(json.dumps(dict(energy=energy, profile=profile, rows=rows), indent=2, default=str) + "\n", encoding="utf-8")

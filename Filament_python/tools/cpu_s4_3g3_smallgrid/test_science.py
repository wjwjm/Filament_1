"""Real operator smoke and exact checks; no scheduler or performance claim."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from .adapter import build_role
from .scientific_context import context, frozen_tools


def run_real(tmp_path, shape, intervals):
    import numba

    report = Path(os.environ["S43SG_REPORT_ROOT"])
    cases, exact = frozen_tools(report)
    records = []
    oracle = None
    for role in ("I0_SMALL", "G2_SMALL", "G3_SMALL"):
        ctx = context(report, shape=shape)
        directory = tmp_path / role
        prop, dispatch, _, binding, _, _ = build_role(role, "PURE_OFF_SMALLGRID", capture_root=directory / "capture", local_engineering=True, pool_override=1)
        field, kwargs, view = cases.prepare(ctx, directory / "propagation", intervals=intervals)
        before = np.array(ctx["source"], copy=True)
        try:
            result = prop(field, **kwargs)
            bundle = cases.finish(result, view, directory / "propagation", intervals=intervals)
        finally:
            view.close()
        assert numba.get_num_threads() == 1
        assert np.array_equal(ctx["source"], before)
        assert result[0].shape == shape and result[0].dtype == np.complex128 and result[0].flags.c_contiguous
        assert np.isfinite(result[0]).all() and np.max(np.abs(result[0])) > 0
        comparison = None
        if oracle is None:
            oracle = exact.save_tree(directory / "oracle", bundle, compressed=True)
        else:
            comparison = exact.compare(Path(oracle), bundle, directory / "EXACT.jsonl", disk=True)
            assert comparison["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        metrics = None
        if role != "I0_SMALL":
            state = dispatch.cpu_s4_3g_state
            metrics = state.metrics()
            nH = int(metrics["controlled_full_calls"])
            assert nH >= 2 * intervals and metrics["route_counts"]["F_STRICT_FALLBACK"] == 0
            counts = metrics["controlled_hash_role_counts"]
            assert all(counts[name] == nH for name in ("Omega_stage1", "Omega_apply", "Omega_stage2"))
            assert counts.get("E_stage1", 0) == (nH if role == "G2_SMALL" else 0)
            assert getattr(state._local, "context", None) is None and state._g_pending_cap is None
            if role == "G3_SMALL":
                assert metrics["lifecycle_identity_created"] == nH
                assert metrics["lifecycle_identity_validated"] == nH
                assert metrics["lifecycle_identity_retired"] == nH
                assert not metrics["active_lifecycle_identity"]
        records.append(dict(role=role, binding=binding, exact=comparison, metrics=metrics, z_axis=result[2]["z_axis"].tolist()))
    (tmp_path / "REAL_CASES.json").write_text(json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8")


@pytest.mark.parametrize("intervals", (2, 8))
def test_smallgrid_real_micro_exact(intervals, tmp_path):
    run_real(tmp_path, (32, 17, 19), intervals)


def test_smallgrid_real_nt384_micro_exact(tmp_path):
    run_real(tmp_path, (384, 17, 19), 2)

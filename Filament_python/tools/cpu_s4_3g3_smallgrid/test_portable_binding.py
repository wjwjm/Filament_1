"""Low-cost checks for the repository-local G2 source binding bridge."""

from __future__ import annotations

from pathlib import Path
import inspect

from tools.cpu_s4_checkpoint_support import (
    g2_adapter_path,
    load_g2_adapter,
    repository_root,
    support_root,
)


def test_checkpoint_g2_bridge_uses_current_repo_and_real_head_blob() -> None:
    root = repository_root()
    support = support_root()
    assert support.is_relative_to(root)
    module = load_g2_adapter()
    assert Path(module.__file__).resolve() == g2_adapter_path().resolve()
    assert module._portable_checkpoint_binding is True
    assert module._frozen_tracked_record("Filament_python/KHz_filament/propagate.py")["git_blob_oid"]
    for name in (
        "_role_audit_source_contract",
        "_frozen_tracked_record",
        "_external_frozen_source_binding",
    ):
        assert getattr(module, name).__globals__ is vars(module)
    role_audit = module._role_audit_source_contract()
    assert Path(role_audit["source_path"]).resolve() == support / "g1_segment.py"
    assert role_audit["source_raw_sha256"]
    assert role_audit["closed_code"].co_name == "closed"
    mapping = module._boundary.load_boundary_mapping()
    assert mapping["field_count"] == 112
    from .benchmark import run_window

    assert inspect.signature(run_window).parameters["application_started"].default is None


def test_checkpoint_bridge_validates_legacy_raw_support_bytes() -> None:
    module = load_g2_adapter()
    path = support_root() / "g1_segment.py"
    binding = module._external_frozen_source_binding(path, path.read_bytes())
    assert binding["external_frozen_source"] is True
    assert binding["legacy_relative_path"].endswith("/g1_segment.py")
    assert binding["git_blob_oid"] is None

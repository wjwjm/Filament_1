from __future__ import annotations

import inspect
from pathlib import Path

from . import candidate


def test_candidate_api_is_l1_only_and_returns_sg_shape():
    signature = inspect.signature(candidate.build_role)
    assert list(signature.parameters)[:1] == ["mode"]
    assert "capture_root" in signature.parameters
    assert candidate.LINEAR_CACHE_ROLE == "L1_LINEAR_FACTOR_CACHE"
    assert candidate.CANDIDATE_SCHEMA.startswith("filament.cpu_s4_4a")


def test_package_does_not_eager_import_cache():
    init = Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")
    assert "import cache" not in init
    assert "from .cache" not in init

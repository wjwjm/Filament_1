"""Read-only test: configs/production/ copies byte-identical to authoritative originals."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit.verify_config_production_copies import (  # noqa: E402
    check,
    pairs,
)

REPO = Path(__file__).resolve().parents[2]


def test_authoritative_pairs_declared():
    assert len(pairs(REPO)) == 3


def test_config_production_copies_byte_identical():
    mismatches = check(REPO)
    assert mismatches == []

#!/usr/bin/env python3
"""Read-only byte-identical verification for configs/production copies.

Verifies that each copy in configs/production/ is byte-identical (SHA256) to its
authoritative original under Filament_python/. It never writes or changes files;
it only reads and compares.

Exit code: 0 when all copies match, 1 when any copy is missing or differs.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

AUTHORITATIVE_PAIRS = [
    ("Filament_python/config_ref.json", "configs/production/config_ref.json"),
    ("Filament_python/khz_config.json", "configs/production/khz_config.json"),
    ("Filament_python/khz_config_lut.json", "configs/production/khz_config_lut.json"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pairs(repo: Path = REPO) -> list[tuple[Path, Path]]:
    """Return (authoritative_original, production_copy) pairs rooted at repo."""
    return [(repo / orig, repo / copy) for orig, copy in AUTHORITATIVE_PAIRS]


def check(repo: Path = REPO) -> list[dict]:
    """Return a list of mismatch records; empty means all copies match."""
    mismatches = []
    for original, copy in pairs(repo):
        if not original.is_file():
            mismatches.append(
                {"kind": "missing_original", "original": str(original), "copy": str(copy)}
            )
            continue
        if not copy.is_file():
            mismatches.append(
                {"kind": "missing_copy", "original": str(original), "copy": str(copy)}
            )
            continue
        h_orig = _sha256(original)
        h_copy = _sha256(copy)
        if h_orig != h_copy:
            mismatches.append(
                {
                    "kind": "sha256_mismatch",
                    "original": str(original),
                    "copy": str(copy),
                    "original_sha256": h_orig,
                    "copy_sha256": h_copy,
                }
            )
    return mismatches


def main(argv=None) -> int:
    repo = REPO
    mismatches = check(repo)
    for original, copy in pairs(repo):
        h_orig = _sha256(original) if original.is_file() else "<missing>"
        h_copy = _sha256(copy) if copy.is_file() else "<missing>"
        status = "OK  " if h_orig == h_copy else "FAIL"
        print(f"{status}  {original}  ==  {copy}")
    if mismatches:
        for m in mismatches:
            print(f"FAIL: {m}")
        return 1
    print("all configs/production copies are byte-identical to authoritative originals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
